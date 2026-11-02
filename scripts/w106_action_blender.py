#!/usr/bin/env python3
"""W106 Blender-side stages (run inside Blender 5.1).  Scratch copies only; no original file is ever opened for write.

  inventory  open scratch copies of the frozen W105 exports (and the legacy smoke file) and record Blender/file
             versions, scene timing, armature, active Action/slot, every Action's channels and value digest, bone
             hierarchy and quaternion norms.  Used for identity checks and for the predeclared selection.
  probe      for every declared case: N0 null save/reopen, E1 local rotation edit, E2 +/- key-time edit, E3 exact
             inverses, E4 save/reopen, plus an FBX export (unchanged ``src/blender/export.py``) / clean-scene import of
             N0, E1 and E2+.  Every condition is sampled at every integer and half frame (local channels, world bone
             heads/tails/rotations) into ``<case>/<condition>.npz`` + channel JSON.  The synthetic minimal armature is
             built here, and its extra boundary probes (quaternion sign, forced negative-key / colliding shifts,
             start/end windows, missing key) are run on further copies.
  render     fixed-view Workbench renders (existing W91 cameras, skinned mesh only) of declared frames per condition.

    blender -b --factory-startup -t 2 --python-exit-code 1 -P scripts/w106_action_blender.py -- probe --declaration ...
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT/"src"), str(ROOT/"scripts")]

import w106_action_contract as C  # noqa: E402

QPATH = 'pose.bones["{}"].rotation_quaternion'


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# --- Blender helpers ----------------------------------------------------------------------------------------------
def fcurves(action):
    from render_runtime_heading_rig_blender import _fcurves  # unchanged W91 helper (layered-Action aware)
    return _fcurves(action)


def armature_obj():
    import bpy
    arms = [o for o in bpy.data.objects if o.type == "ARMATURE"]
    if len(arms) != 1:
        raise RuntimeError(f"expected one armature, found {[o.name for o in arms]}")
    return arms[0]


def channels_of(action):
    out = []
    for fc in sorted(fcurves(action), key=lambda f: (f.data_path, f.array_index)):
        kps = list(fc.keyframe_points)
        out.append({"data_path": fc.data_path, "index": fc.array_index, "keys": len(kps),
                    "frames": [float(k.co[0]) for k in kps], "values": [float(k.co[1]) for k in kps],
                    "interpolation": sorted({k.interpolation for k in kps}),
                    "extrapolation": fc.extrapolation, "modifiers": len(fc.modifiers), "mute": bool(fc.mute)})
    return out


def assign(arm, action):
    arm.animation_data_create()
    arm.animation_data.action = action
    ad = arm.animation_data
    if getattr(ad, "action_slot", None) is None and len(getattr(action, "slots", [])) >= 1:
        ad.action_slot = action.slots[0]
    slot = getattr(ad, "action_slot", None)
    return {"action": ad.action.name, "slot": slot.identifier if slot is not None else None,
            "nla_tracks": len(ad.nla_tracks), "action_blend_type": ad.action_blend_type,
            "action_influence": float(ad.action_influence)}


def active_info(arm):
    ad = arm.animation_data
    if ad is None or ad.action is None:
        return {"action": None, "slot": None}
    slot = getattr(ad, "action_slot", None)
    return {"action": ad.action.name, "slot": slot.identifier if slot is not None else None,
            "nla_tracks": len(ad.nla_tracks)}


def scene_timing():
    import bpy
    s = bpy.context.scene
    return {"fps": s.render.fps, "fps_base": float(s.render.fps_base), "frame_start": s.frame_start,
            "frame_end": s.frame_end, "effective_fps": s.render.fps / s.render.fps_base}


def bones_info(arm):
    sc = arm.scale[0]
    info = {}
    for b in arm.data.bones:
        info[b.name] = {"parent": b.parent.name if b.parent else None, "length_arm": float(b.length),
                        "length_world": float(b.length*sc), "children_recursive": sorted(c.name for c in b.children_recursive),
                        "matrix_local": [[float(v) for v in row] for row in b.matrix_local]}
    return info


def sample(arm, times, out_npz):
    """Evaluate the armature at every time; returns summary, writes arrays."""
    import bpy
    import numpy as np
    scene = bpy.context.scene
    names = [b.name for b in arm.data.bones]
    T, B = len(times), len(names)
    lq = np.zeros((T, B, 4)); ll = np.zeros((T, B, 3)); ls = np.zeros((T, B, 3))
    wh = np.zeros((T, B, 3)); wt = np.zeros((T, B, 3)); wr = np.zeros((T, B, 3, 3)); om = np.zeros((T, 4, 4))
    for i, t in enumerate(times):
        f = int(math.floor(t))
        scene.frame_set(f, subframe=float(t - f))
        dg = bpy.context.evaluated_depsgraph_get()
        ev = arm.evaluated_get(dg)
        mw = ev.matrix_world
        om[i] = np.array(mw)
        for j, n in enumerate(names):
            pb = ev.pose.bones[n]
            q = pb.rotation_quaternion
            lq[i, j] = (q.w, q.x, q.y, q.z)
            ll[i, j] = tuple(pb.location)
            ls[i, j] = tuple(pb.scale)
            m = mw @ pb.matrix
            wh[i, j] = tuple(mw @ pb.head)
            wt[i, j] = tuple(mw @ pb.tail)
            wr[i, j] = np.array(m.to_3x3().normalized())
    np.savez_compressed(out_npz, times=np.array(times), bones=np.array(names), local_quat=lq, local_loc=ll,
                        local_scale=ls, world_head=wh, world_tail=wt, world_rot=wr, object_matrix=om)
    return {"samples": T, "bones": B, "npz": Path(out_npz).name, "npz_sha256": sha256(out_npz)}


def target_curves(action, bone):
    fcs = [fc for fc in fcurves(action) if fc.data_path == QPATH.format(bone)]
    fcs.sort(key=lambda f: f.array_index)
    if [f.array_index for f in fcs] != [0, 1, 2, 3]:
        raise RuntimeError(f"{bone}: expected 4 quaternion F-curves, found {[f.array_index for f in fcs]}")
    n = {len(f.keyframe_points) for f in fcs}
    frames = {tuple(float(k.co[0]) for k in f.keyframe_points) for f in fcs}
    if len(n) != 1 or len(frames) != 1:
        raise RuntimeError(f"{bone}: quaternion components are not co-keyed")
    return fcs


def edit_e1(action, bone, window, inverse=False):
    off = C.e1_offset()
    fcs = target_curves(action, bone)
    for i in window:
        q = tuple(float(f.keyframe_points[i].co[1]) for f in fcs)
        q2 = C.apply_inverse_offset(q, off) if inverse else C.apply_local_offset(q, off)
        for f, v in zip(fcs, q2):
            kp = f.keyframe_points[i]
            d = v - kp.co[1]
            kp.co[1] = v
            kp.handle_left[1] += d
            kp.handle_right[1] += d
    return [list(map(float, (f.keyframe_points[i].co[1] for f in fcs))) for i in window]


def edit_e2(action, bone, group, delta, frame_start, frame_end, enforce=True):
    fcs = target_curves(action, bone)
    frames = [float(k.co[0]) for k in fcs[0].keyframe_points]
    if enforce:
        C.shift_keys(frames, group, delta, frame_start, frame_end)  # raises ContractViolation, nothing modified
    for f in fcs:
        for i in group:
            kp = f.keyframe_points[i]
            kp.co[0] += delta
            kp.handle_left[0] += delta
            kp.handle_right[0] += delta
    return [float(k.co[0]) for k in fcs[0].keyframe_points]


# --- stage: inventory ---------------------------------------------------------------------------------------------
def inventory(o):
    import bpy
    spec = json.loads(Path(o.spec).read_text())
    rep = {"schema": C.SCHEMA + ":inventory", "blender_version": bpy.app.version_string,
           "blender_file_version": list(bpy.app.version_file), "files": {}}
    for key, item in spec["files"].items():
        p = Path(item["path"])
        before = sha256(p)
        bpy.ops.wm.open_mainfile(filepath=str(p))
        arm = armature_obj()
        acts = {}
        for a in bpy.data.actions:
            ch = channels_of(a)
            norms = []
            for bone in sorted({c["data_path"] for c in ch if c["data_path"].endswith("rotation_quaternion")}):
                comps = sorted((c for c in ch if c["data_path"] == bone), key=lambda c: c["index"])
                if len(comps) == 4:
                    norms += [math.sqrt(sum(c["values"][k]**2 for c in comps)) for k in range(comps[0]["keys"])]
            acts[a.name] = {"slots": [s.identifier for s in getattr(a, "slots", [])], "layers": len(getattr(a, "layers", [])),
                            "fcurves": len(ch), "channels": ch, "value_digest": C.channel_digest(ch),
                            "keyed_frames": sorted({f for c in ch for f in c["frames"]}),
                            "quat_norm_minmax": [min(norms), max(norms)] if norms else None,
                            "use_fake_user": bool(a.use_fake_user)}
        rep["files"][key] = {"path": str(p), "sha256_before": before, "file_version": list(bpy.data.version),
                             "timing": scene_timing(), "armature": arm.name, "armature_scale": list(arm.scale),
                             "active": active_info(arm), "actions": acts, "bones": bones_info(arm),
                             "cameras": sorted(ob.name for ob in bpy.data.objects if ob.type == "CAMERA"),
                             "mesh_children": sorted(ob.name for ob in bpy.data.objects if ob.type == "MESH" and ob.parent == arm)}
        if sha256(p) != before:
            raise RuntimeError(f"STOP: {p} changed while being inventoried")
    Path(o.out).write_text(json.dumps(rep, indent=1))
    return 0


# --- synthetic minimal armature -----------------------------------------------------------------------------------
SYN_FRAMES, SYN_MISSING = C.SYN_FRAMES, C.SYN_MISSING


def build_synthetic(path):
    import bpy
    from mathutils import Vector
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.fps, scene.render.fps_base = 30, 1.0
    scene.frame_start, scene.frame_end = 1, 25
    data = bpy.data.armatures.new("syn")
    arm = bpy.data.objects.new("root", data)
    scene.collection.objects.link(arm)
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.mode_set(mode="EDIT")
    def bone(name, head, tail, parent=None):
        b = data.edit_bones.new(name)
        b.head, b.tail = Vector(head), Vector(tail)
        if parent:
            b.parent = data.edit_bones[parent]
            b.use_connect = False
        return b
    bone("base", (0, 0, 0), (0, 0, 1))
    bone(C.TARGET_BONE, (0, 0, 1), (0.3, 0, 1))
    bone("lowerarm_l", (0.3, 0, 1), (0.6, 0, 1), C.TARGET_BONE)
    bone("hand_l", (0.6, 0, 1), (0.7, 0, 1), "lowerarm_l")
    bone("sibling", (0, 0, 1), (-0.3, 0, 1), "base")
    data.edit_bones[C.TARGET_BONE].parent = data.edit_bones["base"]
    bpy.ops.object.mode_set(mode="OBJECT")
    act = bpy.data.actions.new("SYN baseline")
    act.use_fake_user = True
    arm.animation_data_create()
    arm.animation_data.action = act
    for pb in arm.pose.bones:
        pb.rotation_mode = "QUATERNION"
    for k, f in enumerate(SYN_FRAMES):
        for j, pb in enumerate(arm.pose.bones):
            if SYN_MISSING.get(pb.name) == f:
                continue
            ang = 0.15*(k+1)*(j+1)
            q = C.qaxis_angle((math.sin(j+1), math.cos(2*j+1), 0.3*(j+1)), math.degrees(ang))
            pb.rotation_quaternion = q
            pb.keyframe_insert("rotation_quaternion", frame=f)
    for fc in fcurves(act):
        for kp in fc.keyframe_points:
            kp.interpolation = "CONSTANT"
    bpy.ops.wm.save_as_mainfile(filepath=str(path))
    return act.name


# --- stage: probe -------------------------------------------------------------------------------------------------
def open_case(path, action_name):
    import bpy
    bpy.ops.wm.open_mainfile(filepath=str(path))
    arm = armature_obj()
    if action_name not in bpy.data.actions:
        raise RuntimeError(f"action {action_name!r} missing in {path}")
    act = bpy.data.actions[action_name]
    if arm.animation_data is None or arm.animation_data.action is None or arm.animation_data.action.name != action_name:
        info = assign(arm, act)
    else:
        info = active_info(arm)
    return arm, act, info


def record(case_dir, cond, arm, act, times, extra=None):
    ch = channels_of(act)
    rec = {"condition": cond, "action": act.name, "active": active_info(arm), "timing": scene_timing(),
           "channel_digest": C.channel_digest(ch), "channels": ch}
    rec.update(sample(arm, times, case_dir/f"{cond}.npz"))
    if extra:
        rec.update(extra)
    (case_dir/f"{cond}.json").write_text(json.dumps(rec, indent=1))
    return {k: v for k, v in rec.items() if k != "channels"}


def save_reopen(case_dir, cond, arm, act, times, extra=None):
    import bpy
    mem_digest = C.channel_digest(channels_of(act))
    path = case_dir/f"{cond}.blend"
    bpy.ops.wm.save_as_mainfile(filepath=str(path))
    arm2, act2, _ = open_case(path, act.name)
    out = record(case_dir, cond + "_reopen", arm2, act2, times,
                 {"in_memory_digest_before_save": mem_digest, "blend_sha256": sha256(path)} | (extra or {}))
    return path, out


def fbx_roundtrip(case_dir, cond, blend, action_name, times, bone_names):
    """Export through the unchanged repo exporter, import into a factory-empty scene with default importer settings."""
    import bpy
    from blender.export import export_fbx
    fdir = case_dir/"fbx"
    fdir.mkdir(exist_ok=True)
    fbx = fdir/f"{cond}.fbx"
    open_case(blend, action_name)
    export_fbx(str(fbx))
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.fbx(filepath=str(fbx))
    arm = armature_obj()
    acts = [a.name for a in bpy.data.actions]
    match = [a for a in acts if a == action_name or a.endswith("|" + action_name)]
    info = {"fbx_sha256": sha256(fbx), "fbx_bytes": fbx.stat().st_size, "imported_actions": acts,
            "matched_action": match, "imported_bones": len(arm.data.bones),
            "missing_bones": sorted(set(bone_names) - {b.name for b in arm.data.bones}),
            "extra_bones": sorted({b.name for b in arm.data.bones} - set(bone_names)),
            "imported_timing": scene_timing(), "imported_armature_scale": list(arm.scale),
            "active_after_import": active_info(arm)}
    if len(match) != 1:
        info["status"] = "NO_UNIQUE_ACTION"
        (case_dir/f"fbx_{cond}.json").write_text(json.dumps(info, indent=1))
        return info
    act = bpy.data.actions[match[0]]
    info["assigned"] = assign(arm, act)
    ch = channels_of(act)
    tgt = [c for c in ch if c["data_path"] == QPATH.format(C.TARGET_BONE)]
    info["target_keys"] = [c["keys"] for c in tgt]
    info["imported_interpolation"] = sorted({i for c in ch for i in c["interpolation"]})
    info["imported_key_frames_target"] = tgt[0]["frames"] if tgt else None
    info["rest_matrix_local"] = {b.name: [[float(v) for v in r] for r in b.matrix_local] for b in arm.data.bones
                                 if b.name in bone_names}
    info["status"] = "IMPORTED"
    info.update(record(case_dir, f"fbx_{cond}", arm, act, times))
    (case_dir/f"fbx_{cond}_meta.json").write_text(json.dumps(info, indent=1))
    return info


def run_case(case_dir, case, decl, do_fbx):
    import bpy
    case_dir.mkdir(parents=True)
    src = Path(case["input_copy"])
    if sha256(src) != case["input_sha256"]:
        raise RuntimeError(f"STOP: scratch input {src} does not match its declared hash")
    work = case_dir/"input.blend"
    shutil.copyfile(src, work)
    name, bone = case["action"], C.TARGET_BONE
    arm, act, assigned = open_case(work, name)
    timing = scene_timing()
    times = C.sample_times(timing["frame_start"], timing["frame_end"])
    frames = case["target_key_frames"]
    win, grp, delta = case["e1_window"], case["e2_group"], case["e2_offset"]
    log = {"case": case["id"], "assigned": assigned, "times": len(times), "conditions": {}}
    log["conditions"]["orig"] = record(case_dir, "orig", arm, act, times)
    # N0: unedited copy saved (active Action pointer set to the declared Action, scratch-only) and reopened
    n0, log["conditions"]["n0_reopen"] = save_reopen(case_dir, "n0", arm, act, times)
    # E1 / E3(E1)
    arm, act, _ = open_case(n0, name)
    vals = edit_e1(act, bone, win)
    log["conditions"]["e1_mem"] = record(case_dir, "e1_mem", arm, act, times, {"edited_values": vals})
    e1, log["conditions"]["e1_reopen"] = save_reopen(case_dir, "e1", arm, act, times)
    arm, act, _ = open_case(e1, name)
    edit_e1(act, bone, win, inverse=True)
    _, log["conditions"]["e3_e1_reopen"] = save_reopen(case_dir, "e3_e1", arm, act, times)
    # E2 +/- and E3(E2+)
    e2 = {}
    for tag, d in (("e2p", delta), ("e2m", -delta)):
        arm, act, _ = open_case(n0, name)
        new = edit_e2(act, bone, grp, d, timing["frame_start"], timing["frame_end"])
        log["conditions"][f"{tag}_mem"] = record(case_dir, f"{tag}_mem", arm, act, times, {"new_frames": new, "delta": d})
        e2[tag], log["conditions"][f"{tag}_reopen"] = save_reopen(case_dir, tag, arm, act, times, {"delta": d})
    arm, act, _ = open_case(e2["e2p"], name)
    edit_e2(act, bone, grp, -delta, timing["frame_start"], timing["frame_end"])
    _, log["conditions"]["e3_e2p_reopen"] = save_reopen(case_dir, "e3_e2p", arm, act, times)
    # start-window negative-key refusal (contract level; Blender untouched)
    start_grp = list(range(C.E2_GROUP_KEYS))
    try:
        C.shift_keys(frames, start_grp, -max(delta, frames[0] - timing["frame_start"] + delta), timing["frame_start"], timing["frame_end"])
        log["start_negative_shift"] = "ACCEPTED"
    except C.ContractViolation as exc:
        log["start_negative_shift"] = f"REFUSED: {exc}"
    if do_fbx:
        bone_names = [b.name for b in armature_obj().data.bones]
        log["fbx"] = {}
        for cond, blend in (("n0", n0), ("e1", e1), ("e2p", e2["e2p"])):
            try:
                info = fbx_roundtrip(case_dir, cond, blend, name, times, bone_names)
                log["fbx"][cond] = {k: v for k, v in info.items() if k not in ("rest_matrix_local", "imported_key_frames_target")}
            except Exception as exc:  # recorded, never retried with other settings
                log["fbx"][cond] = {"status": "FAILED", "error": repr(exc)}
    return log


def synthetic_extras(case_dir, syn_blend, name, times):
    """Boundary probes on further copies of the synthetic Action (never on real Actions)."""
    out = {}
    frames = SYN_FRAMES
    # quaternion sign: negate one key of a non-target bone -> channels differ, rotation identical
    arm, act, _ = open_case(syn_blend, name)
    fcs = target_curves(act, "sibling")
    for f in fcs:
        f.keyframe_points[3].co[1] *= -1.0
    out["sign_flip"] = record(case_dir, "x_sign_flip", arm, act, times, {"bone": "sibling", "key_index": 3})
    # forced negative-key shift of the first group (the contract refuses this; Blender is asked anyway)
    arm, act, _ = open_case(syn_blend, name)
    grp = list(range(C.E2_GROUP_KEYS))
    try:
        C.shift_keys(frames, grp, -1.0, 1, 25)
        out["negative_contract"] = "ACCEPTED"
    except C.ContractViolation as exc:
        out["negative_contract"] = f"REFUSED: {exc}"
    new = edit_e2(act, C.TARGET_BONE, grp, -1.0, 1, 25, enforce=False)
    out["negative_forced"] = record(case_dir, "x_negative_forced", arm, act, times, {"new_frames": new})
    # forced colliding shift (+3 = key spacing) -> duplicate key times
    arm, act, _ = open_case(syn_blend, name)
    mid = C.e2_group(len(frames))
    try:
        C.shift_keys(frames, mid, 3.0, 1, 25)
        out["collision_contract"] = "ACCEPTED"
    except C.ContractViolation as exc:
        out["collision_contract"] = f"REFUSED: {exc}"
    new = edit_e2(act, C.TARGET_BONE, mid, 3.0, 1, 25, enforce=False)
    out["collision_forced"] = record(case_dir, "x_collision_forced", arm, act, times, {"new_frames": new})
    # start / end windows for E1 (CONSTANT extrapolation before the first key and after the last)
    for tag, win in (("x_e1_start", list(range(5))), ("x_e1_end", list(range(4, 9)))):
        arm, act, _ = open_case(syn_blend, name)
        edit_e1(act, C.TARGET_BONE, win)
        out[tag] = record(case_dir, tag, arm, act, times, {"window": win})
    return out


def probe(o):
    import bpy
    decl_bytes = Path(o.declaration).read_bytes()
    if hashlib.sha256(decl_bytes).hexdigest() != o.declaration_sha:
        raise RuntimeError("STOP: declaration hash mismatch")
    decl = json.loads(decl_bytes)
    out = Path(o.out_dir)
    out.mkdir(parents=True, exist_ok=False)
    rep = {"schema": C.SCHEMA + ":probe", "blender_version": bpy.app.version_string, "declaration_sha256": o.declaration_sha,
           "cases": {}}
    # synthetic first: built fresh every run (determinism covers construction too)
    syn_dir = out/"synthetic"
    syn_dir.mkdir()
    syn_src = syn_dir/"synthetic_built.blend"
    syn_name = build_synthetic(syn_src)
    syn = dict(decl["synthetic"])
    syn.update({"input_copy": str(syn_src), "input_sha256": sha256(syn_src), "action": syn_name})
    rep["synthetic_built_sha256"] = syn["input_sha256"]
    rep["cases"]["synthetic"] = run_case(syn_dir/"case", syn, decl, do_fbx=True)
    times = C.sample_times(1, 25)
    rep["synthetic_extras"] = synthetic_extras(syn_dir/"case", syn_src, syn_name, times)
    for cid, case in decl["cases"].items():
        rep["cases"][cid] = run_case(out/cid, case, decl, do_fbx=case.get("fbx", False))
    (out/"probe_report.json").write_text(json.dumps(rep, indent=1))
    return 0


# --- stage: render ------------------------------------------------------------------------------------------------
def render(o):
    import bpy
    decl = json.loads(Path(o.declaration).read_text())
    probe_dir = Path(o.probe_dir)
    out = Path(o.out_dir)
    out.mkdir(parents=True, exist_ok=False)
    rep = {"schema": C.SCHEMA + ":render", "blender_version": bpy.app.version_string, "cases": {}}
    for cid, case in decl["cases"].items():
        if not case.get("render"):
            continue
        conds = {"orig": probe_dir/cid/"input.blend", "n0": probe_dir/cid/"n0.blend", "e1": probe_dir/cid/"e1.blend",
                 "e3_e1": probe_dir/cid/"e3_e1.blend", "e2p": probe_dir/cid/"e2p.blend", "e2m": probe_dir/cid/"e2m.blend",
                 "e3_e2p": probe_dir/cid/"e3_e2p.blend"}
        frames = case["render_frames"]
        rep["cases"][cid] = {"frames": frames, "views": case["render_views"], "conditions": {}}
        for cond, blend in conds.items():
            arm, act, info = open_case(blend, case["action"])
            scene = bpy.context.scene
            keep = {arm.name} | {ob.name for ob in bpy.data.objects if ob.type == "MESH" and ob.parent == arm}
            for ob in bpy.data.objects:
                if ob.type != "CAMERA":
                    ob.hide_render = ob.name not in keep
            scene.render.resolution_x = scene.render.resolution_y = 480
            cdir = out/cid/cond
            cdir.mkdir(parents=True)
            for f in frames:
                scene.frame_set(f)
                for view in case["render_views"]:
                    scene.camera = bpy.data.objects[view]
                    scene.render.filepath = str(cdir/f"{view.replace(' ', '_')}_{f:04d}.png")
                    bpy.ops.render.render(write_still=True)
            rep["cases"][cid]["conditions"][cond] = {"blend_sha256": sha256(blend), "active": info,
                                                    "engine": scene.render.engine}
    (out/"render_report.json").write_text(json.dumps(rep, indent=1))
    return 0


def main():
    argv = sys.argv[sys.argv.index("--")+1:] if "--" in sys.argv else sys.argv[1:]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=("inventory", "probe", "render"))
    for name in ("spec", "out", "declaration", "declaration_sha", "out_dir", "probe_dir"):
        ap.add_argument("--"+name.replace("_", "-"))
    o = ap.parse_args(argv)
    try:
        return {"inventory": inventory, "probe": probe, "render": render}[o.mode](o)
    except Exception as exc:  # Blender may otherwise exit 0 after a Python exception
        import traceback
        traceback.print_exc()
        print(f"W106 Blender stage failed: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    sys.exit(main())
