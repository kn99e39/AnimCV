#!/usr/bin/env python3
"""W110 Blender-side stages (run inside Blender 5.1).  Scratch copies only; no original file is ever written.

  produce  PRODUCER / GROUND-TRUTH side (source access allowed): for every declared case open a scratch copy of the
           source .blend, record the source census (per item and transform group, the key interpolation set of the
           selected Action's F-curves) and sample the source on the declared main grid and the near-boundary grid.
           For the mixed case, first build the predeclared mixed Action on a scratch copy of the W106 synthetic N0 file
           and export it once through W107's V1 call.
  consume  CONSUMER side.  Reads ONLY the consumer input written by the host ``encode`` stage (FBX paths/SHAs,
           selected-take names, sample grids) -- never the declaration, a .blend, a census or a source sample.  Per
           case and per FBX (unchanged V1 control / metadata-encoded scratch copy), each condition starts from a
           factory-empty scene and an unchanged default ``import_scene.fbx``:
             U  untouched import (negative baseline)
             A  ``w110_fbx_consumer_adapter.recover(fbx, stack)`` immediately after import (audit-logged file
                access), then integrity snapshots before/after
             K  blanket CONSTANT on every key of the selected imported Action (negative control; harness, not adapter)
           and samples the selected imported Action at the identical times.

    blender -b --factory-startup -t 2 --python-exit-code 1 -P scripts/w110_fbx_metadata_recovery_blender.py -- produce ...
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT/"src"), str(ROOT/"scripts")]

import w110_fbx_consumer_adapter as A  # noqa: E402  (imported before any audit hook is armed)

sha256_bytes = lambda b: hashlib.sha256(b).hexdigest()  # noqa: E731


def sha256(p):
    return sha256_bytes(Path(p).read_bytes())


# =====================================================================================================================
# PRODUCER / GROUND TRUTH
# =====================================================================================================================
def census(act, arm):
    import w110_fbx_metadata_recovery_contract as C
    from w106_action_blender import fcurves
    groups, unsupported, per_curve = {}, [], []
    for fc in fcurves(act):
        g = C.source_group_of(fc.data_path)
        interp = sorted({k.interpolation for k in fc.keyframe_points})
        per_curve.append([fc.data_path, fc.array_index, interp])
        if g is None:
            unsupported.append(fc.data_path)
            continue
        kind, bone, grp = g
        key = f"object|{grp}" if kind == "object" else f"bone:{bone}|{grp}"
        groups.setdefault(key, set()).update(interp)
    return {"armature_object": arm.name, "bones": sorted(b.name for b in arm.data.bones),
            "groups": {k: sorted(v) for k, v in sorted(groups.items())}, "unsupported_paths": sorted(set(unsupported)),
            "fcurves": len(per_curve), "per_curve_sha256": sha256_bytes(json.dumps(per_curve).encode())}


def source_record(case_dir, work, action, times, nb_times):
    import bpy
    from w106_action_blender import channels_of, open_case, sample, scene_timing
    from w106_action_contract import channel_digest
    from w108_fbx_subframe_blender import interp_set, keydata
    work_sha = sha256(work)
    arm, act, info = open_case(work, action)
    ch = channels_of(act)
    out = {"assigned": info, "timing": scene_timing(), "source_action_digest": channel_digest(ch),
           "source_fcurves": len(ch), "source_keys_total": int(sum(c["keys"] for c in ch)),
           "source_interpolation": interp_set(keydata(act)),
           "source_key_frames": sorted({f for c in ch for f in c["frames"]}),
           "slots": [s.identifier for s in getattr(act, "slots", [])],
           "stored_actions": sorted(a.name for a in bpy.data.actions), "census": census(act, arm)}
    out["source_sample"] = sample(arm, times, case_dir/"source.npz")
    out["source_nb_sample"] = sample(arm, nb_times, case_dir/"source_nb.npz")
    out["work_file_unchanged"] = sha256(work) == work_sha
    return out


def build_mixed(case_dir, spec, syn_input):
    """The predeclared mixed Action (contract MIXED_SPEC) on a scratch copy of the W106 synthetic N0 file."""
    import bpy
    import numpy as np
    from w106_action_blender import channels_of, fcurves, open_case, scene_timing
    from w106_action_contract import channel_digest
    from w108_fbx_subframe_blender import _enum_values, assign_slot, interp_set, keydata, keydata_diff
    base_copy = case_dir/"base_input.blend"
    shutil.copyfile(syn_input["path"], base_copy)
    if sha256(base_copy) != syn_input["sha256"]:
        raise RuntimeError("STOP: synthetic scratch input hash mismatch")
    arm, base, info = open_case(base_copy, spec["base_action"])
    kb = keydata(base)
    dup = base.copy()
    dup.name = spec["action"]
    vals = _enum_values("interpolation")
    assigned = {}
    for fc in fcurves(dup):
        g = fc.data_path.split('"')[1] if fc.data_path.startswith("pose.bones") else None
        mode = "CONSTANT" if g in spec["held_bones"] else "LINEAR" if g in spec["smooth_bones"] else None
        if mode is None:
            raise RuntimeError(f"STOP: mixed construction: undeclared channel {fc.data_path}")
        n = len(fc.keyframe_points)
        fc.keyframe_points.foreach_set("interpolation", np.full(n, vals[mode], dtype=np.int32))
        assigned[f"{fc.data_path}[{fc.array_index}]"] = mode
    kd = keydata(dup)
    rec = {"base_assigned": info, "base_digest": channel_digest(channels_of(base)), "dup_name": dup.name,
           "dup_fields_changed_excl_interpolation": keydata_diff(kb, kd, ignore=("interpolation",)),
           "dup_interpolation": interp_set(kd), "curve_interpolation": assigned,
           "key_frames": sorted({float(k.co[0]) for fc in fcurves(dup) for k in fc.keyframe_points}),
           "missing_key_frames": {b: sorted({float(k.co[0]) for fc in fcurves(dup) if f'"{b}"' in fc.data_path
                                             for k in fc.keyframe_points}) for b in spec["missing_key"]}}
    rec["assigned"] = assign_slot(arm, dup, info["slot"])
    rec["base_unchanged"] = not keydata_diff(kb, keydata(base))
    rec["dup_digest_equals_base"] = channel_digest(channels_of(dup)) == rec["base_digest"]
    rec["timing"] = scene_timing()
    work = case_dir/"input.blend"
    bpy.ops.wm.save_as_mainfile(filepath=str(work))
    rec["work_sha256"] = sha256(work)
    rec["base_input_unchanged"] = sha256(base_copy) == syn_input["sha256"]
    rec["valid_construction"] = bool(
        rec["assigned"] is not None and rec["dup_fields_changed_excl_interpolation"] == []
        and rec["dup_interpolation"] == ["CONSTANT", "LINEAR"] and rec["base_unchanged"]
        and rec["dup_digest_equals_base"] and rec["base_input_unchanged"]
        and rec["key_frames"] == spec["key_frames"]
        and all(rec["missing_key_frames"][b] == [f for f in spec["key_frames"] if f != m]
                for b, m in spec["missing_key"].items()))
    return work, rec


def export_v1(case_dir, work, action, v0_kwargs):
    from w106_action_blender import channels_of, open_case, active_info
    from w106_action_contract import channel_digest
    import w107_fbx_blender as B7
    fbx = case_dir/"V1.fbx"
    work_sha = sha256(work)
    arm, act, info = open_case(work, action)
    before = channel_digest(channels_of(act))
    exp = B7.do_export("V1", fbx, v0_kwargs)
    after = channel_digest(channels_of(act))
    return {"export": exp, "exported_active": active_info(arm), "source_assigned": info,
            "action_unmodified": before == after, "work_file_unchanged": sha256(work) == work_sha,
            "fbx": str(fbx), "fbx_sha256": sha256(fbx), "fbx_bytes": fbx.stat().st_size,
            "fbx_structure": B7.fbx_structure(fbx)}


def produce(o):
    import bpy
    decl_bytes = Path(o.declaration).read_bytes()
    if sha256_bytes(decl_bytes) != o.declaration_sha:
        raise RuntimeError("STOP: declaration hash mismatch")
    decl = json.loads(decl_bytes)
    if bpy.app.version_string != decl["environment"]["blender_version"]:
        raise RuntimeError("STOP: Blender version differs from the declaration")
    out = Path(o.out_dir)
    out.mkdir(parents=True, exist_ok=False)
    rep = {"schema": "w110:produce", "blender_version": bpy.app.version_string, "declaration_sha256": o.declaration_sha,
           "cases": {}}
    for cid, case in decl["cases"].items():
        d = out/cid
        d.mkdir()
        if case.get("build"):
            work, build = build_mixed(d, decl["mixed"], decl["synthetic_input"])
            r = {"build": build}
            if build["valid_construction"]:
                r.update(source_record(d, work, case["action"], case["times"], case["nb_times"]))
                r["v1_export"] = export_v1(d, work, case["action"], decl["exporter"]["v0_kwargs"])
            r["work_sha256_end"] = sha256(work)
        else:
            src = Path(case["source_blend"])
            if sha256(src) != case["source_blend_sha256"]:
                raise RuntimeError(f"STOP: source input {src} does not match its declared hash")
            work = d/"input.blend"
            shutil.copyfile(src, work)
            r = source_record(d, work, case["action"], case["times"], case["nb_times"])
            r["work_sha256_end"] = sha256(work)
            r["work_equals_input"] = r["work_sha256_end"] == case["source_blend_sha256"]
        rep["cases"][cid] = r
    (out/"produce_report.json").write_text(json.dumps(rep, indent=1))
    return 0


# =====================================================================================================================
# CONSUMER
# =====================================================================================================================
_AUDIT = {"on": False, "events": []}


def _audit(event, args):
    if _AUDIT["on"] and event in ("open", "os.listdir", "os.scandir", "glob.glob", "shutil.copyfile"):
        try:
            _AUDIT["events"].append([event, str(args[0]) if args else ""])
        except Exception:  # never let the hook break the adapter
            _AUDIT["events"].append([event, "<unprintable>"])


def snapshot():
    """Everything the adapter must not change, plus per-curve key data of every Action."""
    import bpy
    from w108_fbx_subframe_blender import keydata
    import w107_fbx_blender as B7
    arms = [o for o in bpy.data.objects if o.type == "ARMATURE"]
    return {"keydata": {a.name: keydata(a) for a in bpy.data.actions},
            "actions": sorted((a.name, tuple(s.identifier for s in a.slots)) for a in bpy.data.actions),
            "objects": {o.name: [o.type, o.rotation_mode, [list(r) for r in o.matrix_basis], o.parent.name
                                 if o.parent else None] for o in bpy.data.objects},
            "rest": {o.name: B7.rest_info(o) for o in arms},
            "bones": {o.name: len(o.data.bones) for o in arms},
            "pose_modes": {o.name: {pb.name: pb.rotation_mode for pb in o.pose.bones} for o in arms},
            "timing": _timing(), "filepath": bpy.data.filepath, "libraries": len(bpy.data.libraries)}


def _timing():
    import bpy
    s = bpy.context.scene
    return {"fps": s.render.fps, "fps_base": float(s.render.fps_base), "frame_start": s.frame_start,
            "frame_end": s.frame_end}


def integrity(pre, post, selected, applied):
    """Compare snapshots: only keyframe interpolation of the selected Action's directed F-curves may differ."""
    from w108_fbx_subframe_blender import keydata_diff
    rec = {}
    sel_pre, sel_post = pre["keydata"].get(selected, {}), post["keydata"].get(selected, {})
    rec["selected_fields_changed_excl_interpolation"] = keydata_diff(sel_pre, sel_post, ignore=("interpolation",))
    changed = sorted(k for k in sel_pre if k in sel_post
                     and sel_pre[k]["interpolation"].tobytes() != sel_post[k]["interpolation"].tobytes())
    rec["selected_curves_interpolation_changed"] = len(changed)
    rec["selected_curves_changed_list_sha256"] = sha256_bytes(json.dumps(changed).encode())
    rec["changed_curves_equal_applied"] = (applied is not None and len(changed) == applied["fcurves_changed"])
    others = [a for a in pre["keydata"] if a != selected]
    rec["other_actions"] = len(others)
    rec["other_actions_unchanged"] = all(keydata_diff(pre["keydata"][a], post["keydata"].get(a, {})) == []
                                         for a in others) and set(pre["keydata"]) == set(post["keydata"])
    for k in ("actions", "objects", "rest", "bones", "pose_modes", "timing", "filepath", "libraries"):
        rec[f"{k}_unchanged"] = pre[k] == post[k]
    rec["ok"] = bool(rec["selected_fields_changed_excl_interpolation"] == [] and rec["other_actions_unchanged"]
                     and rec["changed_curves_equal_applied"]
                     and all(rec[f"{k}_unchanged"] for k in ("actions", "objects", "rest", "bones", "pose_modes",
                                                            "timing", "filepath", "libraries")))
    return rec, changed


def classify_access(events, fbx, allowed_roots):
    ok, other = [], []
    fp = str(Path(fbx).resolve()).lower()
    for ev, p in events:
        q = p.lower().replace("/", "\\")
        if ev == "open" and str(Path(p).resolve()).lower() == fp:
            ok.append(p)
        elif ev == "open" and q.endswith((".py", ".pyc", ".pyd", ".dll")) and any(q.startswith(r) for r in allowed_roots):
            ok.append(p)
        else:
            other.append([ev, p])
    return {"events": len(events), "fbx_reads": sum(1 for _, p in events if str(Path(p).resolve()).lower() == fp),
            "other_access": other, "isolated": not other}


def interp_tally(act):
    """W109's F-curve interpolation tally (object / pose channels, keys per interpolation)."""
    from w109_fbx_constant_import_blender import census
    return census(act)


def import_scene(fbx):
    import bpy
    bpy.ops.wm.read_factory_settings(use_empty=True)
    return sorted(bpy.ops.import_scene.fbx(filepath=str(fbx)))


def select_action(stack):
    import bpy
    import w107_fbx_blender as B7
    from w106_action_blender import armature_obj
    arm = armature_obj()
    chosen, how = B7.resolve_action(sorted(a.name for a in bpy.data.actions), stack)
    return arm, chosen, how


def consume_one(case_dir, tag, cond, fbx, stack, times, nb_times, allowed_roots):
    import bpy
    from w106_action_blender import assign, sample
    from w108_fbx_subframe_blender import interp_set, keydata, set_interpolation
    from w109_fbx_constant_import_blender import kd_digest
    name = f"{tag}_{cond}"
    rec = {"file": tag, "condition": cond, "fbx_sha256": sha256(fbx)}
    rec["import_result"] = import_scene(fbx)
    arm, chosen, how = select_action(stack)
    rec.update({"selected_action": chosen, "selection": how, "imported_actions": len(bpy.data.actions),
                "armatures": [o.name for o in bpy.data.objects if o.type == "ARMATURE"], "bones": len(arm.data.bones)})
    if chosen is None or how != "name_match":
        rec["status"] = "NO_UNIQUE_ACTION"
        return rec
    kd0 = keydata(bpy.data.actions[chosen])
    rec["import_keydata_digest_all"] = kd_digest(kd0)
    rec["import_keydata_digest_excl_interpolation"] = kd_digest(kd0, ignore=("interpolation",))
    rec["import_interpolation"] = interp_set(kd0)
    if cond == "A":
        pre = snapshot()
        _AUDIT["events"].clear()
        t0 = time.perf_counter()
        _AUDIT["on"] = True
        try:
            rep = A.recover(str(fbx), stack)
        finally:
            _AUDIT["on"] = False
        rec["adapter_seconds"] = time.perf_counter() - t0
        post = snapshot()
        rec["adapter"] = rep
        rec["access"] = classify_access(list(_AUDIT["events"]), fbx, allowed_roots)
        rec["integrity"], changed = integrity(pre, post, chosen, rep["applied"])
        rec["changed_curves"] = changed
        rec["selected_equals_adapter_action"] = rep["plan"]["accounting"].get("target_actions") == [chosen]
    elif cond == "K":
        set_interpolation(bpy.data.actions[chosen], "CONSTANT")
    act = bpy.data.actions[chosen]
    rec["assigned"] = assign(arm, act)
    kd = keydata(act)
    rec["final_interpolation"] = interp_set(kd)
    rec["final_interp_tally"] = interp_tally(act)
    rec["final_keydata_digest_excl_interpolation"] = kd_digest(kd, ignore=("interpolation",))
    rec["final_keydata_digest_all"] = kd_digest(kd)
    rec["timing"] = _timing()
    rec["sample"] = sample(arm, times, case_dir/f"{name}.npz")
    rec["nb_sample"] = sample(arm, nb_times, case_dir/f"{name}_nb.npz")
    rec["status"] = "SAMPLED"
    (case_dir/f"{name}_meta.json").write_text(json.dumps(rec, indent=1, default=str))
    return rec


CONDITIONS = {"control": ("U", "A"), "meta": ("U", "A", "K")}


def consume(o):
    import bpy
    data = Path(o.consumer_input).read_bytes()
    if sha256_bytes(data) != o.consumer_input_sha:
        raise RuntimeError("STOP: consumer input hash mismatch")
    ci = json.loads(data)
    if bpy.app.version_string != ci["blender_version"]:
        raise RuntimeError("STOP: Blender version differs")
    out = Path(o.out_dir)
    out.mkdir(parents=True, exist_ok=False)
    allowed_roots = [str(Path(bpy.utils.resource_path("LOCAL")).parent).lower().replace("/", "\\"),
                     str(Path(sys.prefix)).lower().replace("/", "\\"), str(ROOT/"scripts").lower().replace("/", "\\")]
    sys.addaudithook(_audit)
    rep = {"schema": "w110:consume", "blender_version": bpy.app.version_string, "consumer_input_sha256":
           o.consumer_input_sha, "adapter_sha256": sha256(Path(A.__file__)), "allowed_module_roots": allowed_roots,
           "cases": {}}
    for cid, case in ci["cases"].items():
        d = out/cid
        d.mkdir()
        rc = {}
        for tag, conds in CONDITIONS.items():
            f = case["files"][tag]
            if sha256(f["path"]) != f["sha256"]:
                raise RuntimeError(f"STOP: {f['path']} does not match the consumer input")
            for cond in conds:
                try:
                    rc[f"{tag}_{cond}"] = consume_one(d, tag, cond, Path(f["path"]), case["stack"], case["times"],
                                                      case["nb_times"], allowed_roots)
                except Exception as exc:  # recorded, never retried with other settings
                    import traceback
                    rc[f"{tag}_{cond}"] = {"status": "FAILED", "error": repr(exc), "traceback": traceback.format_exc()}
        rep["cases"][cid] = rc
    (out/"consume_report.json").write_text(json.dumps(rep, indent=1, default=str))
    return 0


def main():
    argv = sys.argv[sys.argv.index("--")+1:] if "--" in sys.argv else sys.argv[1:]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=("produce", "consume"))
    for name in ("declaration", "declaration_sha", "consumer_input", "consumer_input_sha", "out_dir"):
        ap.add_argument("--"+name.replace("_", "-"))
    o = ap.parse_args(argv)
    try:
        return {"produce": produce, "consume": consume}[o.mode](o)
    except Exception as exc:  # Blender may otherwise exit 0 after a Python exception
        import traceback
        traceback.print_exc()
        print(f"W110 Blender stage failed: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    sys.exit(main())
