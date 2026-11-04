#!/usr/bin/env python3
"""W110 POST-HOC diagnostic (written after the frozen analysis returned CASE D; NOT a gate, never changes the CASE).

The frozen repeat criterion compared the two fresh mixed-case V1 exports by canonical tree (only
FBXHeaderExtension/CreationTimeStamp skipped) and the adapter plans verbatim; both differed.  This script
characterises that difference without re-running anything:

  * every differing node / property between the two exports, classified (header timestamp, SceneInfo source path,
    Document / object UID, element order),
  * a UID-independent comparison: each Objects node keyed by its name / connection-derived identity
    (Model and NodeAttribute by name; AnimationCurveNode by (model, property); AnimationCurve by (model, property,
    channel); stack / layer by name), content compared with the UID removed; connections translated to those keys,
  * the adapter plans and producer directives with UIDs relabelled to (model, property, channel) keys,
  * curve key arrays by (model, property, channel).

Inputs: the stored W110 encode / consume outputs of both runs; output: <out>/posthoc_report.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT/"scripts"))
import w110_fbx_consumer_adapter as AD  # noqa: E402
import w110_fbx_metadata_recovery_contract as C  # noqa: E402
import w108_fbx_subframe_contract as C8  # noqa: E402
import w109_fbx_constant_import_contract as C9  # noqa: E402
from w108_fbx_subframe import wjson  # noqa: E402


def positional_diffs(a, b):
    out = []

    def walk(x, y, path):
        if x.id != y.id or bytes(x.props_type) != bytes(y.props_type) or len(x.props) != len(y.props) \
                or len(x.elems) != len(y.elems):
            out.append([path + "/" + x.id.decode(), "structure"])
            return
        for i, (t, u, v) in enumerate(zip(bytes(x.props_type).decode(), x.props, y.props)):
            if C9._canon(t, u) != C9._canon(t, v):
                out.append([path + "/" + x.id.decode(), i])
        for cx, cy in zip(x.elems, y.elems):
            walk(cx, cy, path + "/" + x.id.decode())
    for x, y in zip(a.elems, b.elems):
        walk(x, y, "")
    return out


def classify(diffs, a):
    """Bucket positional differences by top-level section and node kind."""
    c = Counter()
    for path, i in diffs:
        parts = path.split("/")
        c["/".join(parts[1:3]) + (f"[prop {i}]" if i != "structure" else "[structure]")] += 1
    return dict(sorted(c.items()))


def keyed_objects(root, fmap):
    """{identity key: canonical content without UID} for every Objects node + translated connections."""
    objs, conns = C8._child(root, b"Objects"), C8._child(root, b"Connections")
    name = {int(c.props[0]): AD._name(c.props[1]) for c in objs.elems}
    kind = {int(c.props[0]): c.id.decode() for c in objs.elems}
    parents = {}
    for c in conns.elems:
        parents.setdefault(int(c.props[1]), []).append((int(c.props[2]), AD._str(c.props[0]),
                                                        AD._str(c.props[3]) if len(c.props) > 3 else None))
    curve_key = {}
    for st in sorted(n for u, n in name.items() if kind[u] == "AnimationStack"):
        for r in AD.fbx_curve_map(root, st)["curves"]:
            if r["status"] == "OK":
                curve_key[r["uid"]] = ("AnimationCurve", st, r["model_name"], r["fbx_property"], r["channel"])
    key = {}
    for u, k in kind.items():
        if k == "AnimationCurve":
            key[u] = curve_key.get(u, ("AnimationCurve", "<other take>", u))
        elif k == "AnimationCurveNode":
            mp = [(p, pn) for p, t, pn in parents.get(u, []) if kind.get(p) == "Model"]
            ly = [p for p, t, pn in parents.get(u, []) if kind.get(p) == "AnimationLayer"]
            key[u] = ("AnimationCurveNode", name[mp[0][0]] if mp else None, mp[0][1] if mp else None,
                      name[ly[0]] if ly else None)
        elif k == "NodeAttribute":
            ch = [c for c, ps in parents.items() if any(p == u for p, _, _ in ps)]
            owner = [p for p, _, _ in parents.get(u, []) if kind.get(p) == "Model"] + \
                    [c for c in ch if kind.get(c) == "Model"]
            key[u] = ("NodeAttribute", name[u], tuple(sorted(name[o] for o in owner)))
        else:
            key[u] = (k, name[u])
    content = {}
    for c in objs.elems:
        u = int(c.props[0])
        h = C.tree_digest(type(root)(b"", [], b"", [type(c)(c.id, [0] + list(c.props[1:]), c.props_type, c.elems)]))
        content.setdefault(key[u], []).append(h)
    tconn = []
    for c in conns.elems:
        ch, pa = int(c.props[1]), int(c.props[2])
        tconn.append((AD._str(c.props[0]), str(key.get(ch, ("<root>",) if ch == 0 else ("?", ch))),
                      str(key.get(pa, ("<root>",) if pa == 0 else ("?", pa))),
                      AD._str(c.props[3]) if len(c.props) > 3 else None))
    dup_keys = sorted(str(k) for k, v in content.items() if len(v) > 1)
    return {str(k): sorted(v) for k, v in content.items()}, sorted(tconn), dup_keys


def relabel_plan(plan, fmap):
    by_uid = {str(r["uid"]): f'{r.get("model_name")}|{r.get("fbx_property")}|{r.get("channel")}' for r in fmap["curves"]}
    models = {str(m): v["name"] for m, v in fmap["models"].items()}
    groups = sorted(json.dumps({**{k: v for k, v in g.items() if k not in ("model", "fbx_curves")},
                                "model": models[str(g["model"])],
                                "fbx_curves": sorted(by_uid[str(u)] for u in g["fbx_curves"])}, sort_keys=True)
                    for g in plan["groups"])
    items = sorted(json.dumps({k: v for k, v in it.items() if k != "model"}, sort_keys=True)
                   for it in plan["items"].values())
    status = sorted((by_uid[u], s) for u, s in plan["curve_status"].items())
    return {"groups": groups, "items": items, "curve_status": status, "accounting": plan["accounting"]}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ("results", "out_dir"):
        ap.add_argument("--"+name.replace("_", "-"), required=True)
    o = ap.parse_args()
    R = Path(o.results)
    out = Path(o.out_dir)
    out.mkdir(parents=True, exist_ok=False)
    decl = json.loads((R/"declare/declaration.json").read_text())
    stack = decl["cases"][C.MIXED]["stack"]
    rep = {"schema": C.SCHEMA + ":posthoc", "label": "POST-HOC, NOT A GATE; the frozen CASE is unchanged", "files": {}}
    for f in ("control", "meta"):
        a, _ = C8.fbx_read(R/f"encode/{C.MIXED}/{f}.fbx")
        b, _ = C8.fbx_read(R/f"encode_repeat/{C.MIXED}/{f}.fbx")
        fa, fb = AD.fbx_curve_map(a, stack), AD.fbx_curve_map(b, stack)
        d = positional_diffs(a, b)
        ka, ca, da = keyed_objects(a, fa)
        kb, cb, db = keyed_objects(b, fb)
        hdr = {}
        for top in (b"FBXHeaderExtension", b"GlobalSettings", b"Documents", b"Takes", b"Definitions", b"References"):
            ta, tb = C8._child(a, top), C8._child(b, top)
            if ta is None or tb is None:
                hdr[top.decode()] = "absent" if ta is None and tb is None else "presence differs"
                continue
            ra = type(a)(b"", [], b"", [ta])
            rb = type(b)(b"", [], b"", [tb])
            hdr[top.decode()] = {"equal": C.tree_digest(ra) == C.tree_digest(rb),
                                 "equal_without_timestamp": C.tree_digest(ra, skip_paths=C.TIMESTAMP_PATHS)
                                 == C.tree_digest(rb, skip_paths=C.TIMESTAMP_PATHS)}
        sv = []
        for x in (a, b):
            sinfo = C8._child(C8._child(x, b"FBXHeaderExtension"), b"SceneInfo")
            p70 = C8._child(sinfo, b"Properties70") if sinfo is not None else None
            sv.append({AD._str(p.props[0]): AD._str(p.props[4]) for p in (p70.elems if p70 is not None else [])
                       if len(p.props) > 4 and isinstance(p.props[4], (bytes, bytearray))})
        si = {k: [sv[0].get(k), sv[1].get(k)] for k in sorted(set(sv[0]) | set(sv[1])) if sv[0].get(k) != sv[1].get(k)}
        arr_a, arr_b = C.curve_arrays(a, fa), C.curve_arrays(b, fb)
        rep["files"][f] = {
            "positional_differences": len(d), "positional_difference_classes": classify(d, a),
            "uids_equal": sorted(int(c.props[0]) for c in C8._child(a, b"Objects").elems)
            == sorted(int(c.props[0]) for c in C8._child(b, b"Objects").elems),
            "uid_independent_map_equal": AD.fbx_map_digest({**fa, "curves": sorted(
                ({k: v for k, v in r.items() if k not in ("uid", "curve_node", "model")} for r in fa["curves"]),
                key=json.dumps), "models": sorted(json.dumps(m, sort_keys=True) for m in fa["models"].values())})
            == AD.fbx_map_digest({**fb, "curves": sorted(
                ({k: v for k, v in r.items() if k not in ("uid", "curve_node", "model")} for r in fb["curves"]),
                key=json.dumps), "models": sorted(json.dumps(m, sort_keys=True) for m in fb["models"].values())}),
            "objects_by_identity_equal": ka == kb, "object_identity_keys": len(ka),
            "objects_differing_by_identity": sorted(k for k in set(ka) | set(kb) if ka.get(k) != kb.get(k))[:20],
            "duplicate_identity_keys": [da, db],
            "connections_by_identity_equal": ca == cb, "connections": len(ca),
            "curve_arrays_equal": set(arr_a) == set(arr_b) and all(
                np.array_equal(arr_a[k][0], arr_b[k][0]) and np.array_equal(arr_a[k][1], arr_b[k][1]) for k in arr_a),
            "header_sections": hdr, "scene_info_string_differences": si}
    cons = [json.loads((R/f"{d}/consume_report.json").read_text()) for d in ("consume", "consume_repeat")]
    enc = [json.loads((R/f"{d}/encode_report.json").read_text()) for d in ("encode", "encode_repeat")]
    plans = {}
    for k in ("control_A", "meta_A"):
        f = "control" if k.startswith("control") else "meta"
        rl = []
        for run, d in zip(cons, ("encode", "encode_repeat")):
            root, _ = C8.fbx_read(R/f"{d}/{C.MIXED}/{f}.fbx")
            rl.append(relabel_plan(run["cases"][C.MIXED][k]["adapter"]["plan"], AD.fbx_curve_map(root, stack)))
        plans[k] = {"relabelled_plan_equal": rl[0] == rl[1], "verbatim_plan_equal":
                    cons[0]["cases"][C.MIXED][k]["adapter"]["plan"] == cons[1]["cases"][C.MIXED][k]["adapter"]["plan"],
                    "changed_curves_equal": cons[0]["cases"][C.MIXED][k]["changed_curves"]
                    == cons[1]["cases"][C.MIXED][k]["changed_curves"]}
    rep["plans"] = plans
    dl = []
    for e, d in zip(enc, ("encode", "encode_repeat")):
        root, _ = C8.fbx_read(R/f"{d}/{C.MIXED}/control.fbx")
        fm = AD.fbx_curve_map(root, stack)
        key = {str(r["uid"]): f'{r["model_name"]}|{r["fbx_property"]}|{r["channel"]}' for r in fm["curves"]}
        dl.append(sorted((key[u], v[0]) for u, v in e["cases"][C.MIXED]["producer_directives"].items()))
    rep["producer_directives_relabelled_equal"] = dl[0] == dl[1]
    an = json.loads((R/"analysis/analysis_report.json").read_text())
    rep["frozen"] = {"CASE": an["CASE"], "CASE_reason": an["CASE_reason"],
                     "repeat_syn_mixed": {k: v for k, v in an["repeat"][C.MIXED].items() if k != "max_diff"},
                     "repeat_syn_mixed_max_sample_diff": max(v for v in an["repeat"][C.MIXED]["max_diff"].values()),
                     "other_decision_inputs": {k: v for k, v in an["decision_inputs"].items()
                                               if k not in ("deterministic",)}}
    s = wjson(out/"posthoc_report.json", rep)
    print(json.dumps(rep, indent=1, default=str)[:6000])
    print(s)
    return 0


if __name__ == "__main__":
    sys.exit(main())
