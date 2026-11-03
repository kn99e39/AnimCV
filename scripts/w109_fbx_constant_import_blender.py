#!/usr/bin/env python3
"""W109 Blender-side stage (run inside Blender 5.1).  Scratch copies only; no original file is ever written.

  probe  for each declared case: sample the Blender source Action (scratch .blend copy) on the declared
         integer/quarter-frame grid; then for the unchanged V1 control FBX and each validated scratch CONSTANT variant
         (``encode`` stage output): parse the file with the add-on's own parser (canonical tree digests, selected-stack
         curves, decoded flags), import it with the *unchanged default* importer into a factory-empty scene, select
         the uniquely name-matched Action, record its slot, rest pose, key data, F-curve interpolation census (object /
         pose channels) and the key data of every other imported Action, and sample it at the identical times.
         No post-import correction of any kind.

    blender -b --factory-startup -t 2 --python-exit-code 1 -P scripts/w109_fbx_constant_import_blender.py -- probe ...
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT/"src"), str(ROOT/"scripts")]

import w109_fbx_constant_import_contract as C  # noqa: E402
import w107_fbx_blender as B7  # noqa: E402  (unchanged W107 helpers, read-only reuse)
from w108_fbx_subframe_blender import _enum_values, interp_set, key_time_stats, keydata, keydata_diff  # noqa: E402
from w106_action_blender import (active_info, armature_obj, assign, channels_of, fcurves, open_case,  # noqa: E402
                                 sample, scene_timing)
from w106_action_contract import channel_digest  # noqa: E402

sha256 = C.sha256_file


def kd_digest(kd, ignore=()):
    h = hashlib.sha256()
    for k in sorted(kd):
        h.update(k.encode())
        for f in sorted(kd[k]):
            if f in ignore:
                continue
            v = kd[k][f]
            h.update(f.encode() + (repr(v).encode() if f == "meta" else v.tobytes()))
    return h.hexdigest()


def census(action):
    """F-curve interpolation census, object vs pose channels: {group: {interp-set: n_fcurves}}, keys per interp."""
    import numpy as np
    inv = {v: k for k, v in _enum_values("interpolation").items()}
    out = {"object": {}, "pose": {}}
    keys = {}
    for fc in fcurves(action):
        n = len(fc.keyframe_points)
        a = np.empty(n, dtype=np.int32)
        fc.keyframe_points.foreach_get("interpolation", a)
        vals, cnt = np.unique(a, return_counts=True)
        names = [inv[int(v)] for v in vals]
        grp = "pose" if fc.data_path.startswith("pose.bones") else "object"
        k = "+".join(sorted(names))
        out[grp][k] = out[grp].get(k, 0) + 1
        for nm, c in zip(names, cnt):
            keys[nm] = keys.get(nm, 0) + int(c)
    return {"fcurves": out, "keys": keys}


def addon_tree(path, action, variant):
    try:
        from io_scene_fbx import parse_fbx
        root, version = parse_fbx.parse(str(path))
    except Exception as exc:  # recorded; never retried
        return {"status": "PARSE_FAILED", "error": repr(exc)}
    rec = C.file_tree_record(root, action, variant)
    rec["status"], rec["fbx_version"] = "PARSED", version
    return rec


def import_one(case_dir, tag, fbx, action, times, ctrl):
    """Default import of one FBX; ``ctrl`` holds the control import's in-memory key data/rest (None for control)."""
    import bpy
    rec = {"file": tag, "fbx_sha256": sha256(fbx)}
    bpy.ops.wm.read_factory_settings(use_empty=True)
    rec["import_result"] = sorted(bpy.ops.import_scene.fbx(filepath=str(fbx)))
    arms = [o.name for o in bpy.data.objects if o.type == "ARMATURE"]
    rec["armatures"] = arms
    arm = armature_obj()
    acts = sorted(a.name for a in bpy.data.actions)
    chosen, how = B7.resolve_action(acts, action)
    rec.update({"imported_actions": acts, "selected_action": chosen, "selection": how,
                "active_after_import": active_info(arm), "imported_timing": scene_timing(),
                "armature_scale": list(arm.scale), "bones": len(arm.data.bones)})
    if chosen is None or how != "name_match":
        rec["status"] = "NO_UNIQUE_ACTION"
        return rec, None
    act = bpy.data.actions[chosen]
    rec["assigned"] = assign(arm, act)
    kd = keydata(act)
    rec["imported_interpolation"] = interp_set(kd)
    rec["census"] = census(act)
    rec["key_times"] = key_time_stats(kd)
    rec["keydata_digest_all"] = kd_digest(kd)
    rec["keydata_digest_excl_interpolation"] = kd_digest(kd, ignore=("interpolation",))
    rec["channel_digest"] = channel_digest(channels_of(act))
    others = {a.name: kd_digest(keydata(a)) for a in bpy.data.actions if a.name != chosen}
    rest = B7.rest_info(arm)
    rec["rest_sha256"] = hashlib.sha256(json.dumps(rest, sort_keys=True).encode()).hexdigest()
    if ctrl is not None:
        rec["keydata_fields_differing_from_control"] = keydata_diff(ctrl["kd"], kd)
        rec["keydata_equal_all"] = rec["keydata_fields_differing_from_control"] == []
        rec["keydata_equal_excl_interpolation"] = keydata_diff(ctrl["kd"], kd, ignore=("interpolation",)) == []
        rec["other_actions_equal_control"] = others == ctrl["others"]
        rec["other_actions_differing"] = sorted(k for k in set(others) | set(ctrl["others"])
                                                if others.get(k) != ctrl["others"].get(k))
        rec["rest_equal_control"] = rest == ctrl["rest"]
        rec["actions_equal_control"] = acts == ctrl["acts"]
    rec["other_actions"] = len(others)
    rec["other_actions_interpolation"] = sorted({i for a in bpy.data.actions if a.name != chosen
                                                 for i in interp_set(keydata(a))})
    rec["status"] = "IMPORTED"
    rec["sample"] = sample(arm, times, case_dir/f"{tag}.npz")
    (case_dir/f"{tag}_meta.json").write_text(json.dumps(rec, indent=1))
    return rec, {"kd": kd, "others": others, "rest": rest, "acts": acts}


def run_case(case_dir, cid, case, enc_case, enc_dir):
    case_dir.mkdir(parents=True)
    src = Path(case["source_blend"])
    if sha256(src) != case["source_blend_sha256"]:
        raise RuntimeError(f"STOP: source input {src} does not match its declared hash")
    work = case_dir/"input.blend"
    shutil.copyfile(src, work)
    arm, act, info = open_case(work, case["action"])
    ch = channels_of(act)
    out = {"assigned": info, "timing": scene_timing(), "source_action_digest": channel_digest(ch),
           "source_fcurves": len(ch), "source_keys_total": int(sum(c["keys"] for c in ch)),
           "source_interpolation": sorted({i for c in ch for i in c["interpolation"]}),
           "source_key_frames": sorted({f for c in ch for f in c["frames"]}),
           "slots": [s.identifier for s in getattr(act, "slots", [])]}
    out["source_sample"] = sample(arm, case["times"], case_dir/"source.npz")
    out["work_file_unchanged"] = sha256(work) == case["source_blend_sha256"]
    out["files"] = {}
    ctrl = None
    for tag in [C.CONTROL] + list(enc_case["variants"]):
        fbx = enc_dir/cid/f"{tag}.fbx"
        want = enc_case["control_sha256"] if tag == C.CONTROL else enc_case["variants"][tag]["sha256"]
        if sha256(fbx) != want:
            raise RuntimeError(f"STOP: {fbx} does not match the encode report")
        r = {"addon_tree": addon_tree(fbx, case["action"], tag)}
        try:
            imp, mem = import_one(case_dir, tag, fbx, case["action"], case["times"], ctrl)
        except Exception as exc:  # recorded, never retried with other settings
            import traceback
            imp, mem = {"status": "FAILED", "error": repr(exc), "traceback": traceback.format_exc()}, None
        r["import"] = imp
        if tag == C.CONTROL:
            ctrl = mem
            if ctrl is None:
                out["files"][tag] = r
                break
        out["files"][tag] = r
    return out


def probe(o):
    import bpy
    decl_bytes = Path(o.declaration).read_bytes()
    if hashlib.sha256(decl_bytes).hexdigest() != o.declaration_sha:
        raise RuntimeError("STOP: declaration hash mismatch")
    decl = json.loads(decl_bytes)
    enc_bytes = Path(o.encode_report).read_bytes()
    if hashlib.sha256(enc_bytes).hexdigest() != o.encode_sha:
        raise RuntimeError("STOP: encode report hash mismatch")
    enc = json.loads(enc_bytes)
    if not enc["valid"] or enc["declaration_sha256"] != o.declaration_sha:
        raise RuntimeError("STOP: encode report invalid or bound to another declaration")
    if bpy.app.version_string != decl["environment"]["blender_version"]:
        raise RuntimeError("STOP: Blender version differs from the declaration")
    out = Path(o.out_dir)
    out.mkdir(parents=True, exist_ok=False)
    enc_dir = Path(o.encode_report).parent
    rep = {"schema": C.SCHEMA + ":probe", "blender_version": bpy.app.version_string,
           "declaration_sha256": o.declaration_sha, "encode_sha256": o.encode_sha, "cases": {}}
    for cid, case in decl["cases"].items():
        rep["cases"][cid] = run_case(out/cid, cid, case, enc["cases"][cid], enc_dir)
    (out/"probe_report.json").write_text(json.dumps(rep, indent=1))
    return 0


def main():
    argv = sys.argv[sys.argv.index("--")+1:] if "--" in sys.argv else sys.argv[1:]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=("probe",))
    for name in ("declaration", "declaration_sha", "encode_report", "encode_sha", "out_dir"):
        ap.add_argument("--"+name.replace("_", "-"))
    o = ap.parse_args(argv)
    try:
        return {"probe": probe}[o.mode](o)
    except Exception as exc:  # Blender may otherwise exit 0 after a Python exception
        import traceback
        traceback.print_exc()
        print(f"W109 Blender stage failed: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    sys.exit(main())
