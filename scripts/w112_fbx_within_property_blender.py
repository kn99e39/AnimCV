#!/usr/bin/env python3
"""W112 Blender-side stages (run inside Blender 5.1).  Scratch copies only; no original file is ever written.

  produce    PRODUCER / GROUND TRUTH (source access allowed): per declared case open a scratch copy of its source
             (historical controls: byte copies of the W110 inputs; factorial cases: a duplicate of the base Action
             with the declared interpolation-only intervention -- plus, for the L family only, the separately declared
             location keys), record the per-key source census, the target structure and the source samples on the
             declared probe grid, and export built cases once through W107's V1 call.
  consume    CONSUMER (source-free): reads ONLY consumer_input.json.  Per case, each condition in a factory-empty scene
             with the unchanged default import (W110 harness ``consume_one``):
               control_U / control_A / control_K  and  meta_U / meta_A  (A = unchanged W110 adapter).
  oracle     PRIVILEGED SOURCE-KNOWN FEASIBILITY (not a consumer): reads oracle_input.json.  (i) Import-level bound:
             imports the untouched control FBX, sets every non-target group to its source interpolation and samples
             EVERY declared CONSTANT/LINEAR combination of the target group's imported F-curves (all keys).  (ii)
             FBX-level bound: imports every scratch evaluation FBX of the instrument and samples it at its key times.
  candidate  CONDITIONAL SOURCE-FREE CANDIDATE: reads ONLY candidate_input.json (cases that passed the import-level
             bound, plus the negative controls).  Condition S = ``w112_fbx_perkey_candidate.recover`` after import
             (audit-logged), integrity snapshots; negative controls also run the unchanged W110 adapter (A).

    blender -b --factory-startup -t 2 --python-exit-code 1 -P scripts/w112_fbx_within_property_blender.py -- produce ...
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

import w110_fbx_metadata_recovery_blender as B10  # noqa: E402  (W110 harness: source record, export, consume_one, audit)
import w110_fbx_consumer_adapter as AD  # noqa: E402
import w112_fbx_perkey_candidate as CAND  # noqa: E402  (imported before any audit hook is armed)

sha256_bytes = lambda b: hashlib.sha256(b).hexdigest()  # noqa: E731


def sha256(p):
    return sha256_bytes(Path(p).read_bytes())


# =====================================================================================================================
# PRODUCER / GROUND TRUTH
# =====================================================================================================================
def perkey_census(act, arm):
    """Every F-curve of the Action: path, index, key frames, values, per-key interpolation, handle digest."""
    import numpy as np
    from w106_action_blender import fcurves
    curves = []
    for fc in sorted(fcurves(act), key=lambda f: (f.data_path, f.array_index)):
        kp = fc.keyframe_points
        n = len(kp)
        co = np.empty(2*n, dtype=np.float32)
        kp.foreach_get("co", co)
        hl, hr = np.empty(2*n, dtype=np.float32), np.empty(2*n, dtype=np.float32)
        kp.foreach_get("handle_left", hl)
        kp.foreach_get("handle_right", hr)
        curves.append({"data_path": fc.data_path, "index": fc.array_index, "frames": co[0::2].astype(float).tolist(),
                       "values": co[1::2].astype(float).tolist(),
                       "interpolation": [k.interpolation for k in kp], "extrapolation": fc.extrapolation,
                       "handles_sha256": sha256_bytes(hl.tobytes() + hr.tobytes())})
    return {"armature_object": arm.name, "bones": sorted(b.name for b in arm.data.bones), "curves": curves}


def target_structure(arm, bone):
    b = arm.data.bones.get(bone)
    if b is None:
        return None
    rel = (b.parent.matrix_local.inverted() @ b.matrix_local) if b.parent else b.matrix_local
    return {"bone": bone, "parent": b.parent.name if b.parent else None,
            "rest_rotation_relative_to_parent": [[float(v) for v in r] for r in rel.to_3x3()],
            "pose_rotation_mode": arm.pose.bones[bone].rotation_mode, "object_rotation_mode": arm.rotation_mode,
            "object_scale": list(arm.scale)}


def build_case(d, cid, case, decl):
    """Scratch duplicate of the family's base Action with the declared interpolation-only intervention."""
    import bpy
    import numpy as np
    import w112_fbx_within_property_contract as C
    from bpy_extras import anim_utils
    from w106_action_blender import channels_of, fcurves, open_case, scene_timing
    from w106_action_contract import channel_digest
    from w108_fbx_subframe_blender import _enum_values, assign_slot, keydata, keydata_diff
    fam = C.FAMILIES[case["family"]]
    tpath, titem = case["target"]["source_path"], case["target"]["item"]
    src = decl["inputs"]["w106_synthetic_n0" if fam["rig"] == "synthetic" else "complete_control_source"]
    base_name = C.SYN_BASE_ACTION if fam["rig"] == "synthetic" else C.REAL_ACTION
    base_copy = d/"base_input.blend"
    shutil.copyfile(src["path"], base_copy)
    if sha256(base_copy) != src["sha256"]:
        raise RuntimeError("STOP: scratch base input hash mismatch")
    arm, base, info = open_case(base_copy, base_name)
    kb = keydata(base)
    dup = base.copy()
    dup.name = case["action"]
    rec = {"base_action": base_name, "base_assigned": info, "base_digest": channel_digest(channels_of(base)),
           "dup_name": dup.name}
    rec["assigned"] = assign_slot(arm, dup, info["slot"])
    added = []
    if case["family"] == "L":
        slot = [s for s in dup.slots if s.identifier == info["slot"]][0]
        cb = anim_utils.action_ensure_channelbag_for_slot(dup, slot)
        for i in range(3):
            fc = cb.fcurves.new(tpath, index=i, group_name=titem)
            for k, f in enumerate(C.SYN_KEY_FRAMES):
                fc.keyframe_points.insert(f, C.loc_value(k)[i], options={"FAST"})
            fc.update()
            added.append(f"{tpath}[{i}]")
    vals = _enum_values("interpolation")
    assigned = {}
    for fc in fcurves(dup):
        n = len(fc.keyframe_points)
        if fc.data_path == tpath:
            want = [C.target_interp(case["spec"], case["family"], fc.array_index, k) for k in range(n)]
        else:
            want = [C.CONSTANT]*n
        fc.keyframe_points.foreach_set("interpolation", np.asarray([vals[w] for w in want], dtype=np.int32))
        assigned[f"{fc.data_path}[{fc.array_index}]"] = want
    kd = keydata(dup)
    kb_sub = {k: v for k, v in kd.items() if k not in added}
    rec["dup_fields_changed_excl_interpolation"] = keydata_diff(kb, kb_sub, ignore=("interpolation",))
    rec["non_target_interpolation_unchanged"] = all(
        kd[k]["interpolation"].tobytes() == kb[k]["interpolation"].tobytes() for k in kb
        if not k.startswith(tpath + "["))
    rec["added_curves"] = added
    rec["target_interpolation"] = {k: v for k, v in assigned.items() if k.startswith(tpath + "[")}
    rec["target_key_frames"] = sorted({float(k.co[0]) for fc in fcurves(dup) if fc.data_path == tpath
                                       for k in fc.keyframe_points})
    if case["family"] == "L":
        rec["location_keys"] = {i: [[float(k.co[0]), float(k.co[1])] for k in fc.keyframe_points]
                                for fc in fcurves(dup) if fc.data_path == tpath for i in [fc.array_index]}
        rec["location_keys_as_declared"] = all(
            abs(rec["location_keys"][i][k][0] - f) < 1e-6 and abs(rec["location_keys"][i][k][1] - C.loc_value(k)[i]) < 1e-6
            for i in range(3) for k, f in enumerate(C.SYN_KEY_FRAMES))
        rec["location_keyset_sha256"] = sha256_bytes(json.dumps(rec["location_keys"], sort_keys=True).encode())
    rec["base_unchanged"] = not keydata_diff(kb, keydata(base))
    rec["keydata_excl_interpolation_sha256"] = sha256_bytes(b"".join(
        k.encode() + kd[k]["co"].tobytes() + kd[k]["handle_left"].tobytes() + kd[k]["handle_right"].tobytes()
        for k in sorted(kd)))
    rec["timing"] = scene_timing()
    work = d/"input.blend"
    bpy.ops.wm.save_as_mainfile(filepath=str(work))
    rec["work_sha256"] = sha256(work)
    rec["base_input_unchanged"] = sha256(base_copy) == src["sha256"]
    rec["valid_construction"] = bool(
        rec["assigned"] is not None and rec["dup_fields_changed_excl_interpolation"] == []
        and rec["non_target_interpolation_unchanged"] and rec["base_unchanged"] and rec["base_input_unchanged"]
        and len(rec["target_interpolation"]) == fam["n"]
        and (case["family"] != "L" or (rec["location_keys_as_declared"] and len(added) == 3)))
    return work, rec


def produce(o):
    import bpy
    import w112_fbx_within_property_contract as C
    from w106_action_blender import armature_obj
    decl_bytes = Path(o.declaration).read_bytes()
    if sha256_bytes(decl_bytes) != o.declaration_sha:
        raise RuntimeError("STOP: declaration hash mismatch")
    decl = json.loads(decl_bytes)
    if bpy.app.version_string != decl["environment"]["blender_version"]:
        raise RuntimeError("STOP: Blender version differs from the declaration")
    out = Path(o.out_dir)
    out.mkdir(parents=True, exist_ok=False)
    rep = {"schema": C.SCHEMA + ":produce", "blender_version": bpy.app.version_string,
           "declaration_sha256": o.declaration_sha, "cases": {}}
    for cid, case in decl["cases"].items():
        d = out/cid
        d.mkdir()
        t0 = time.perf_counter()
        if case["origin"] == "build":
            work, build = build_case(d, cid, case, decl)
            r = {"build": build}
            if not build["valid_construction"]:
                r["status"] = "INVALID_CONSTRUCTION"
                rep["cases"][cid] = r
                continue
        else:
            inp = decl["inputs"][f"{cid}_source"]
            work = d/"input.blend"
            shutil.copyfile(inp["path"], work)
            if sha256(work) != inp["sha256"]:
                raise RuntimeError(f"STOP: source copy of {cid} does not match its declared hash")
            r = {}
        r.update(B10.source_record(d, work, case["action"], case["times"], []))
        act = bpy.data.actions[case["action"]]
        arm = armature_obj()
        r["perkey_census"] = perkey_census(act, arm)
        r["target_structure"] = target_structure(arm, case["target"]["item"]) if case["target"] else None
        if case["origin"] == "build":
            r["v1_export"] = B10.export_v1(d, work, case["action"], decl["exporter"]["v0_kwargs"])
        else:
            v1 = decl["inputs"][f"{cid}_V1"]
            shutil.copyfile(v1["path"], d/"V1.fbx")
            r["v1_copy"] = {"fbx": str(d/"V1.fbx"), "fbx_sha256": sha256(d/"V1.fbx"),
                            "equals_declared": sha256(d/"V1.fbx") == v1["sha256"]}
        r["work_sha256_end"] = sha256(work)
        r["seconds"] = time.perf_counter() - t0
        r["status"] = "OK"
        rep["cases"][cid] = r
    (out/"produce_report.json").write_text(json.dumps(rep, indent=1))
    return 0


# =====================================================================================================================
# CONSUMER (source-free; unchanged W110 harness and adapter)
# =====================================================================================================================
def _allowed_roots():
    import bpy
    return [str(Path(bpy.utils.resource_path("LOCAL")).parent).lower().replace("/", "\\"),
            str(Path(sys.prefix)).lower().replace("/", "\\"), str(ROOT/"scripts").lower().replace("/", "\\")]


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
    roots = _allowed_roots()
    sys.addaudithook(B10._audit)
    rep = {"schema": "w112:consume", "blender_version": bpy.app.version_string, "consumer_input_sha256":
           o.consumer_input_sha, "adapter_sha256": sha256(Path(AD.__file__)), "allowed_module_roots": roots,
           "cases": {}}
    for cid, case in ci["cases"].items():
        d = out/cid
        d.mkdir()
        rc = {}
        for tag, conds in (("control", ("U", "A", "K")), ("meta", ("U", "A"))):
            f = case["files"].get(tag)
            if f is None:
                continue
            if sha256(f["path"]) != f["sha256"]:
                raise RuntimeError(f"STOP: {f['path']} does not match the consumer input")
            for cond in conds:
                try:
                    rc[f"{tag}_{cond}"] = B10.consume_one(d, tag, cond, Path(f["path"]), case["stack"], case["times"],
                                                          [], roots)
                except Exception as exc:  # recorded, never retried with other settings
                    import traceback
                    rc[f"{tag}_{cond}"] = {"status": "FAILED", "error": repr(exc), "traceback": traceback.format_exc()}
        rep["cases"][cid] = rc
    (out/"consume_report.json").write_text(json.dumps(rep, indent=1, default=str))
    return 0


# =====================================================================================================================
# PRIVILEGED SOURCE-KNOWN FEASIBILITY ORACLE (evaluation only; never a consumer)
# =====================================================================================================================
def _group_fcurves(pl, item, group):
    gs = [g for g in pl["groups"] if g["item"] == item and g["group"] == group and "action" in g]
    return gs[0] if len(gs) == 1 else None


def _set_group(g, values_per_curve):
    import bpy
    import numpy as np
    from bpy_extras import anim_utils
    a = bpy.data.actions[g["action"]]
    s = [x for x in a.slots if x.identifier == g["slot"]][0]
    cb = anim_utils.action_get_channelbag_for_slot(a, s)
    for (dp, i, _), v in zip(g["fcurves"], values_per_curve):
        f = cb.fcurves.find(dp, index=i)
        n = len(f.keyframe_points)
        f.keyframe_points.foreach_set("interpolation", np.full(n, v, dtype=np.int32))


def _key_frames(g):
    import bpy
    import numpy as np
    from bpy_extras import anim_utils
    a = bpy.data.actions[g["action"]]
    s = [x for x in a.slots if x.identifier == g["slot"]][0]
    cb = anim_utils.action_get_channelbag_for_slot(a, s)
    out = []
    for dp, i, _ in g["fcurves"]:
        f = cb.fcurves.find(dp, index=i)
        n = len(f.keyframe_points)
        co = np.empty(2*n, dtype=np.float32)
        f.keyframe_points.foreach_get("co", co)
        out.append(co[0::2].astype(float))
    return out


def oracle(o):
    import bpy
    import numpy as np
    from io_scene_fbx import parse_fbx
    from w106_action_blender import assign, sample
    from w108_fbx_subframe_blender import _enum_values, keydata, keydata_diff
    data = Path(o.oracle_input).read_bytes()
    if sha256_bytes(data) != o.oracle_input_sha:
        raise RuntimeError("STOP: oracle input hash mismatch")
    oi = json.loads(data)
    out = Path(o.out_dir)
    out.mkdir(parents=True, exist_ok=False)
    vals = _enum_values("interpolation")
    rep = {"schema": "w112:oracle", "blender_version": bpy.app.version_string, "oracle_input_sha256":
           o.oracle_input_sha, "cases": {}}
    for cid, case in oi["cases"].items():
        d = out/cid
        d.mkdir()
        rc = {"imp": {}, "fbx": {}}
        t0 = time.perf_counter()
        f = case["control"]
        if sha256(f["path"]) != f["sha256"]:
            raise RuntimeError(f"STOP: {f['path']} does not match the oracle input")
        rc["import_result"] = B10.import_scene(f["path"])
        arm, chosen, how = B10.select_action(case["stack"])
        if chosen is None:
            rc["status"] = "NO_UNIQUE_ACTION"
            rep["cases"][cid] = rc
            continue
        root, _ = parse_fbx.parse(str(f["path"]))
        fmap = AD.fbx_curve_map(root, case["stack"])
        pl = AD.plan(fmap, AD.describe_scene())
        kd0 = keydata(bpy.data.actions[chosen])
        others0 = {a.name: keydata(a) for a in bpy.data.actions if a.name != chosen}
        set_groups, missing = [], []
        for g in pl["groups"]:
            key = f'{g["item"]}|{g["group"]}'
            want = case["directives"].get(key)
            if want in ("CONSTANT", "LINEAR") and "action" in g:
                _set_group(g, [vals[want]]*len(g["fcurves"]))
                set_groups.append(key)
            elif want not in ("TARGET", "UNDRIVEN", None):
                missing.append([key, want])
        rc["non_target_groups_set"] = sorted(set_groups)
        rc["unusable_directives"] = missing
        tg = None
        if case["target"] is not None:
            tg = _group_fcurves(pl, case["target"]["item"], case["target"]["group"])
        rc["target_fcurves"] = [[dp, i] for dp, i, _ in tg["fcurves"]] if tg else None
        if tg is not None:
            kf = _key_frames(tg)
            rc["target_key_frames"] = [list(map(float, k)) for k in kf]
        assign(arm, bpy.data.actions[chosen])
        labels = case["imp_labels"]
        for lab in labels:
            if tg is not None and lab != "SRC":
                if len(lab) != len(tg["fcurves"]):
                    raise RuntimeError(f"STOP: label {lab} does not match {len(tg['fcurves'])} target F-curves")
                _set_group(tg, [vals["CONSTANT" if ch == "C" else "LINEAR"] for ch in lab])
            rc["imp"][lab] = sample(arm, case["times"], d/f"imp_{lab}.npz")
        kd1 = keydata(bpy.data.actions[chosen])
        rc["harness_fields_changed_excl_interpolation"] = keydata_diff(kd0, kd1, ignore=("interpolation",))
        rc["harness_other_actions_unchanged"] = all(
            keydata_diff(v, keydata(bpy.data.actions[k])) == [] for k, v in others0.items())
        import w112_fbx_within_property_contract as C
        for lab, parts in case["efiles"].items():
            e = {"parts": {}}
            merged = {}
            for pt, ef in parts.items():
                if sha256(ef["path"]) != ef["sha256"]:
                    raise RuntimeError(f"STOP: {ef['path']} does not match the oracle input")
                ptimes = C.part_times(case["times"], pt)
                B10.import_scene(ef["path"])
                arm2, ch2, _ = B10.select_action(case["stack"])
                x = {"selected_action": ch2, "probes": len(ptimes)}
                if ch2 is None:
                    x["status"] = "NO_UNIQUE_ACTION"
                    e["parts"][pt] = x
                    continue
                act2 = bpy.data.actions[ch2]
                assign(arm2, act2)
                if case["target"] is not None:
                    r2, _ = parse_fbx.parse(str(ef["path"]))
                    pl2 = AD.plan(AD.fbx_curve_map(r2, case["stack"]), AD.describe_scene())
                    tg2 = _group_fcurves(pl2, case["target"]["item"], case["target"]["group"])
                    if tg2 is not None:
                        kf2 = _key_frames(tg2)
                        t = np.asarray(ptimes, dtype=float)
                        x["target_keys"] = [int(len(k)) for k in kf2]
                        x["probe_to_key_max_frames"] = float(max(np.abs(k[np.argmin(np.abs(k[None, :] - t[:, None]),
                                                                                        axis=1)] - t).max() for k in kf2))
                x["sample"] = sample(arm2, ptimes, d/f"fbx_{lab}_{pt}.npz")
                x["status"] = "SAMPLED"
                merged[pt] = (ptimes, dict(np.load(d/f"fbx_{lab}_{pt}.npz")))
                e["parts"][pt] = x
            if len(merged) == len(parts):
                idx = {float(t): i for i, t in enumerate(case["times"])}
                first = next(iter(merged.values()))[1]
                arrs = {k: np.zeros((len(case["times"]),) + v.shape[1:], dtype=v.dtype) for k, v in first.items()
                        if k not in ("times", "bones")}
                covered = np.zeros(len(case["times"]), dtype=bool)
                for ptimes, a in merged.values():
                    if list(a["bones"]) != list(first["bones"]):
                        raise RuntimeError("STOP: evaluation parts disagree on bones")
                    rows = [idx[float(t)] for t in ptimes]
                    for k in arrs:
                        arrs[k][rows] = a[k]
                    covered[rows] = True
                e["all_probes_covered"] = bool(covered.all())
                np.savez_compressed(d/f"fbx_{lab}.npz", times=np.asarray(case["times"]), bones=first["bones"], **arrs)
                e["status"] = "SAMPLED" if covered.all() else "INCOMPLETE"
            else:
                e["status"] = "PART_FAILED"
            rc["fbx"][lab] = e
        rc["seconds"] = time.perf_counter() - t0
        rc["status"] = "OK"
        rep["cases"][cid] = rc
    (out/"oracle_report.json").write_text(json.dumps(rep, indent=1, default=str))
    return 0


# =====================================================================================================================
# CONDITIONAL SOURCE-FREE CANDIDATE
# =====================================================================================================================
def candidate_one(case_dir, name, fbx, stack, times, roots):
    import bpy
    from w106_action_blender import assign, sample
    from w108_fbx_subframe_blender import interp_set, keydata
    from w109_fbx_constant_import_blender import kd_digest
    rec = {"file": Path(fbx).name, "condition": "S", "fbx_sha256": sha256(fbx)}
    rec["import_result"] = B10.import_scene(fbx)
    arm, chosen, how = B10.select_action(stack)
    rec.update({"selected_action": chosen, "selection": how, "imported_actions": len(bpy.data.actions)})
    if chosen is None or how != "name_match":
        rec["status"] = "NO_UNIQUE_ACTION"
        return rec
    kd0 = keydata(bpy.data.actions[chosen])
    rec["import_keydata_digest_excl_interpolation"] = kd_digest(kd0, ignore=("interpolation",))
    pre = B10.snapshot()
    B10._AUDIT["events"].clear()
    t0 = time.perf_counter()
    B10._AUDIT["on"] = True
    try:
        rep = CAND.recover(str(fbx), stack)
    finally:
        B10._AUDIT["on"] = False
    rec["candidate_seconds"] = time.perf_counter() - t0
    post = B10.snapshot()
    rec["candidate"] = rep
    rec["access"] = B10.classify_access(list(B10._AUDIT["events"]), fbx, roots)
    rec["integrity"], rec["changed_curves"] = B10.integrity(pre, post, chosen, rep["applied"])
    rec["selected_equals_candidate_action"] = rep["plan"]["accounting"].get("target_actions") == [chosen]
    act = bpy.data.actions[chosen]
    rec["assigned"] = assign(arm, act)
    kd = keydata(act)
    rec["final_interpolation"] = interp_set(kd)
    rec["final_keydata_digest_excl_interpolation"] = kd_digest(kd, ignore=("interpolation",))
    rec["sample"] = sample(arm, times, case_dir/f"{name}.npz")
    rec["status"] = "SAMPLED"
    (case_dir/f"{name}_meta.json").write_text(json.dumps(rec, indent=1, default=str))
    return rec


def candidate(o):
    import bpy
    data = Path(o.candidate_input).read_bytes()
    if sha256_bytes(data) != o.candidate_input_sha:
        raise RuntimeError("STOP: candidate input hash mismatch")
    ci = json.loads(data)
    if bpy.app.version_string != ci["blender_version"]:
        raise RuntimeError("STOP: Blender version differs")
    out = Path(o.out_dir)
    out.mkdir(parents=True, exist_ok=False)
    roots = _allowed_roots()
    sys.addaudithook(B10._audit)
    rep = {"schema": "w112:candidate", "blender_version": bpy.app.version_string, "candidate_input_sha256":
           o.candidate_input_sha, "candidate_sha256": sha256(Path(CAND.__file__)),
           "adapter_sha256": sha256(Path(AD.__file__)), "cases": {}, "controls": {}}
    for sect, items in (("cases", ci["cases"]), ("controls", ci["controls"])):
        for name, item in items.items():
            d = out/sect/name
            d.mkdir(parents=True)
            fbx = Path(item["fbx"]["path"])
            if sha256(fbx) != item["fbx"]["sha256"]:
                raise RuntimeError(f"STOP: {fbx} does not match the candidate input")
            rc = {}
            try:
                rc["S"] = candidate_one(d, "S", fbx, item["stack"], item["times"], roots)
            except Exception as exc:  # recorded, never retried
                import traceback
                rc["S"] = {"status": "FAILED", "error": repr(exc), "traceback": traceback.format_exc()}
            if sect == "controls":
                try:
                    rc["A"] = B10.consume_one(d, "nc", "A", fbx, item["stack"], item["times"], [], roots)
                except Exception as exc:
                    import traceback
                    rc["A"] = {"status": "FAILED", "error": repr(exc), "traceback": traceback.format_exc()}
            rep[sect][name] = rc
    (out/"candidate_report.json").write_text(json.dumps(rep, indent=1, default=str))
    return 0


def main():
    argv = sys.argv[sys.argv.index("--")+1:] if "--" in sys.argv else sys.argv[1:]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=("produce", "consume", "oracle", "candidate"))
    for name in ("declaration", "declaration_sha", "consumer_input", "consumer_input_sha", "oracle_input",
                 "oracle_input_sha", "candidate_input", "candidate_input_sha", "out_dir"):
        ap.add_argument("--"+name.replace("_", "-"))
    o = ap.parse_args(argv)
    try:
        return {"produce": produce, "consume": consume, "oracle": oracle, "candidate": candidate}[o.mode](o)
    except Exception as exc:  # Blender may otherwise exit 0 after a Python exception
        import traceback
        traceback.print_exc()
        print(f"W112 Blender stage failed: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    sys.exit(main())
