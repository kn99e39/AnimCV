#!/usr/bin/env python3
"""W108 post-hoc descriptive records (written after the analysis outcome was read; never a gate, rule or CASE input).

  * KeyTime tick deviation of the selected FBX stack: the analysis counted KeyTimes that are not exact multiples of
    the 30 fps tick; this quantifies their size (ticks / frames) and the stack's frame offset, with the independent
    reader only.
  * Static inspection of the FBX add-on source (the declared P1 importer layer): SHA of import_fbx.py /
    export_fbx_bin.py against the declared add-on SHAs, whether the importer reads KeyAttrFlags at all, the literal
    interpolation it writes, and the exporter's literal KeyAttrFlags expression.
"""
from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT/"scripts"))
import w108_fbx_subframe_contract as C  # noqa: E402


def stack_key_times(root, action):
    """KeyTime arrays of every AnimationCurve connected (via curve node and layer) to the selected stack."""
    objs = C._child(root, b"Objects")
    conns = C._child(root, b"Connections")
    kind, name, times = {}, {}, {}
    for c in objs.elems:
        if c.id in (b"AnimationStack", b"AnimationLayer", b"AnimationCurveNode", b"AnimationCurve"):
            uid = int(c.props[0])
            kind[uid] = c.id.decode()
            name[uid] = bytes(c.props[1]).split(b"\x00")[0].decode(errors="replace")
            if c.id == b"AnimationCurve":
                times[uid] = np.asarray(C._child(c, b"KeyTime").props[0], dtype=np.int64)
    parent = {}
    for c in conns.elems:
        if c.id == b"C" and len(c.props) >= 3:
            parent.setdefault(int(c.props[1]), []).append(int(c.props[2]))

    def up(uid, want):
        return next((p for p in parent.get(uid, []) if kind.get(p) == want), None)
    stack = C.select_stack({name[u] for u, k in kind.items() if k == "AnimationStack"}, action)
    out = []
    for uid, kt in times.items():
        cn = up(uid, "AnimationCurveNode")
        ly = up(cn, "AnimationLayer") if cn is not None else None
        st = up(ly, "AnimationStack") if ly is not None else None
        if st is not None and name[st] == stack:
            out.append(kt)
    return stack, out


def tick_deviation(curves, fps):
    tick = C.FBX_KTIME_PER_SECOND//int(fps)
    allk = np.concatenate(curves) if curves else np.zeros(0, np.int64)
    dev = allk - np.round(allk/tick).astype(np.int64)*tick
    return {"tick_per_frame": int(tick), "keys": int(allk.size), "nonmultiple_keys": int((dev != 0).sum()),
            "max_abs_dev_ticks": int(np.abs(dev).max()) if dev.size else 0,
            "max_abs_dev_frames": float(np.abs(dev).max()/tick) if dev.size else 0.0,
            "dev_ticks_values": sorted({int(x) for x in np.unique(dev)}),
            "first_key_frame": float(allk.min()/tick) if allk.size else None}


def addon_inspection(addon_dir, declared_sha):
    imp = (addon_dir/"import_fbx.py").read_text(encoding="utf-8")
    exp = (addon_dir/"export_fbx_bin.py").read_text(encoding="utf-8")
    shas = {f: C.sha256_file(addon_dir/f) for f in ("import_fbx.py", "export_fbx_bin.py")}
    rec = {"sha256": shas, "sha_equal_declared": {f: shas[f] == (declared_sha or {}).get(f) for f in shas}}
    rec["import_reads_KeyAttrFlags"] = "KeyAttrFlags" in imp
    rec["import_reads"] = sorted({k for k in ("b'KeyTime'", "b'KeyValueFloat'", "b'KeyAttrFlags'",
                                              "b'KeyAttrDataFloat'") if k in imp})
    rec["import_interpolation_lines"] = [ln.strip() for ln in imp.splitlines()
                                         if "INTERPOLATION_VALUE" in ln and "=" in ln][:3]
    # exporter keyattr_flags literal: evaluate the BinOp expression assigned to keyattr_flags
    vals = []
    for n in ast.walk(ast.parse(exp)):
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "keyattr_flags" for t in n.targets):
            ok = all(isinstance(x, (ast.Tuple, ast.BinOp, ast.Constant, ast.LShift, ast.BitOr, ast.Load))
                     for x in ast.walk(n.value))
            if not ok:
                raise ValueError("keyattr_flags is not a constant integer expression")
            vals.append(eval(compile(ast.Expression(n.value), "keyattr_flags", "eval"), {"__builtins__": {}}))
    rec["export_keyattr_flags_literals"] = [[int(x) for x in v] for v in vals]
    rec["export_keyattr_flags_hex"] = [[f"0x{int(x):x}" for x in v] for v in vals]
    rec["export_keyattr_interpolation"] = [[C.decode_interp(x) for x in v] for v in vals]
    rec["export_flags_depend_on_source"] = len(vals) != 1     # a single literal tuple -> independent of the source
    return rec


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--probe-dir", required=True)
    ap.add_argument("--declaration", required=True)
    ap.add_argument("--addon-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    o = ap.parse_args()
    out = Path(o.out_dir)
    out.mkdir(parents=True, exist_ok=False)
    decl = json.loads(Path(o.declaration).read_text())
    p = Path(o.probe_dir)
    rep = {"schema": C.SCHEMA + ":posthoc",
           "label": "POST-HOC descriptive records written after the analysis outcome was read; no gate, rule or CASE "
                    "depends on them", "key_time_ticks": {}}
    specs = {**decl["cases"], **decl["synthetic_controls"]}
    for cid, spec in specs.items():
        fps = spec["timing"]["fps"]/spec["timing"]["fps_base"]
        rep["key_time_ticks"][cid] = {}
        for v in ("V0", "V1"):
            root, _ = C.fbx_read(p/cid/"fbx"/f"{v}.fbx")
            stack, curves = stack_key_times(root, spec["action"])
            rep["key_time_ticks"][cid][v] = {"stack": stack, **tick_deviation(curves, fps)}
    rep["addon_static_inspection"] = addon_inspection(Path(o.addon_dir), decl["environment"]["fbx_addon_source_sha256"])
    (out/"posthoc.json").write_text(json.dumps(rep, indent=1))
    s = C.sha256_file(out/"posthoc.json")
    print(s)
    print(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
