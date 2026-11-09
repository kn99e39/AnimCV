#!/usr/bin/env python3
"""EXPERIMENTAL (opt-in, Batch 129) Blender-side stage for the MotionBricks left-arm inbetweening A/B/C.

Runs ONLY inside Blender's own Python (``blender -b --python-exit-code 1 -P <this file> -- <mode> ...``).  It is never
imported by the default CLI/GUI path, and it never writes into an existing AnimCV run directory.

Modes
  dump           open a saved .blend (never saved back) and report, for every keyed integer Action frame, the evaluated
                 Armature Object ``matrix_world`` and every pose bone's armature-space ``matrix``, plus every bone's rest
                 ``matrix_local`` / head / tail and the stored F-curve key values.  ``--overrides`` (control use only)
                 sets declared pose-bone / object rotations at declared frames IN MEMORY before evaluation; the file is
                 not saved, so the frozen Action is never modified.
  build_variant  open a frozen baseline .blend, COPY its Action under a new name, replace ONLY the declared left-arm
                 channels (``upperarm_l``, ``lowerarm_l`` rotation_quaternion keys) on the declared interior frames with
                 the supplied values, delete the original Action datablock from the NEW file, and save a NEW .blend
                 (refuses to overwrite).  Any channel or frame outside the declared domain is refused before any change.

The fresh-process readback / render of the resulting file reuse ``blender/experimental_action_stage.py`` unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
PROP_KEY = "animcv_experimental_provenance"
EDITABLE_CHANNELS = ("upperarm_l", "lowerarm_l")


def _armature():
    import bpy
    arms = [o for o in bpy.data.objects if o.type == "ARMATURE"]
    if len(arms) != 1:
        raise RuntimeError(f"expected exactly one armature object, found {len(arms)}")
    return arms[0]


def _fcurves(act):
    from render_runtime_heading_rig_blender import _fcurves as frozen
    return frozen(act)


def _mat(m):
    return [[float(v) for v in row] for row in m]


def _stored(act):
    out = {}
    for fc in _fcurves(act):
        ch = "__object__" if fc.data_path == "rotation_quaternion" else fc.data_path.split('"')[1]
        out.setdefault(ch, {})[str(fc.array_index)] = {str(int(round(kp.co[0]))): float(kp.co[1]) for kp in fc.keyframe_points}
    return out


def dump(o):
    import bpy
    from mathutils import Quaternion
    bpy.ops.wm.open_mainfile(filepath=str(o.blend))
    arm = _armature()
    act = arm.animation_data.action if arm.animation_data else None
    if act is None:
        raise RuntimeError("no Action assigned to the armature")
    stored = _stored(act)
    frames = sorted({int(k) for ch in stored.values() for comp in ch.values() for k in comp})
    overrides = json.loads(Path(o.overrides).read_text()) if o.overrides else {}
    want = [int(x) for x in o.frames.split(",")] if o.frames else frames
    scene = bpy.context.scene
    rest = {b.name: {"parent": b.parent.name if b.parent else None, "matrix_local": _mat(b.matrix_local),
                     "head_local": list(b.head_local), "tail_local": list(b.tail_local)} for b in arm.data.bones}
    out_frames = []
    if overrides:
        # CONTROL dump only, in memory, never saved: the Action is detached and every requested pose is set explicitly
        # (bones not named in a pose keep the identity basis; the object rotation must be given).
        arm.animation_data.action = None
        want = sorted(int(k) for k in overrides)
    for f in want:
        if overrides:
            ov = overrides[str(f)]
            for pb in arm.pose.bones:
                pb.rotation_mode = "QUATERNION"
                pb.rotation_quaternion = Quaternion(ov.get("bones", {}).get(pb.name, (1.0, 0.0, 0.0, 0.0)))
            arm.rotation_mode = "QUATERNION"
            arm.rotation_quaternion = Quaternion(ov["object"])
        else:
            scene.frame_set(int(f))
        bpy.context.view_layer.update()
        rec = {"blender_frame": int(f), "matrix_world": _mat(arm.matrix_world),
               "pose_matrix": {pb.name: _mat(pb.matrix) for pb in arm.pose.bones},
               "basis_wxyz": {pb.name: list(pb.rotation_quaternion) for pb in arm.pose.bones},
               "object_wxyz": list(arm.rotation_quaternion), "override": bool(overrides)}
        out_frames.append(rec)
    rep = {"schema": "animcv_exp_w129_dump_v1", "blender_version": bpy.app.version_string, "blend": str(o.blend),
           "blend_sha256": hashlib.sha256(Path(o.blend).read_bytes()).hexdigest(), "armature_object": arm.name,
           "action_name": act.name, "actions_in_file": sorted(a.name for a in bpy.data.actions), "rest": rest,
           "stored": stored, "keyed_frames": frames, "frames": out_frames,
           "scene": {"fps": scene.render.fps, "fps_base": scene.render.fps_base, "frame_start": scene.frame_start,
                     "frame_end": scene.frame_end}}
    Path(o.report).write_text(json.dumps(rep))
    return 0


def build_variant(o):
    import bpy
    spec = json.loads(Path(o.spec).read_text())
    out_blend = Path(spec["out_blend"])
    if out_blend.exists():
        raise RuntimeError(f"refusing to overwrite {out_blend}")
    lo, hi = spec["interior_action_frames"]
    values = spec["values_wxyz"]                       # {channel: {frame(str): [w,x,y,z]}}
    bad_ch = sorted(set(values) - set(EDITABLE_CHANNELS))
    if bad_ch:
        raise RuntimeError(f"REFUSED_CHANNEL_OUTSIDE_DOMAIN {bad_ch}")
    for ch, per in values.items():
        bad_f = sorted(int(f) for f in per if not lo <= int(f) <= hi)
        if bad_f:
            raise RuntimeError(f"REFUSED_FRAME_OUTSIDE_INTERIOR {ch} {bad_f[:5]}")
    src_sha = hashlib.sha256(Path(o.blend).read_bytes()).hexdigest()
    if spec.get("expected_blend_sha256") and src_sha != spec["expected_blend_sha256"]:
        raise RuntimeError(f"REFUSED_BASELINE_IDENTITY {src_sha}")
    bpy.ops.wm.open_mainfile(filepath=str(o.blend))
    arm = _armature()
    orig = arm.animation_data.action
    orig_name = orig.name
    act = orig.copy()
    act.name = spec["action_name"]
    act.use_fake_user = True
    slot = getattr(arm.animation_data, "action_slot", None)
    arm.animation_data.action = act
    if slot is not None and hasattr(arm.animation_data, "action_slot") and act.slots:
        arm.animation_data.action_slot = act.slots[0]
    changed = 0
    for fc in _fcurves(act):
        if fc.data_path == "rotation_quaternion":
            continue
        ch = fc.data_path.split('"')[1]
        if ch not in values:
            continue
        for kp in fc.keyframe_points:
            f = int(round(kp.co[0]))
            v = values[ch].get(str(f))
            if v is None:
                continue
            new = float(v[fc.array_index])
            if new != kp.co[1]:
                changed += 1
            kp.co[1] = new
            kp.handle_left[1] = new
            kp.handle_right[1] = new
            kp.interpolation = "CONSTANT"
        fc.update()
    bpy.data.actions.remove(orig)
    prov = {"schema": "animcv_exp_w129_variant_v1", "branch": spec["branch"], "parent_blend_sha256": src_sha,
            "parent_action_name": orig_name, "interior_action_frames": [lo, hi], "channels": sorted(values),
            "status": "EXPERIMENTAL SCRATCH PROPOSAL - not accepted, not in any lineage", "note": spec.get("note", "")}
    arm[PROP_KEY] = json.dumps(prov, sort_keys=True)
    act[PROP_KEY] = json.dumps(prov, sort_keys=True)
    out_blend.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(out_blend))
    Path(o.report).write_text(json.dumps({"schema": "animcv_exp_w129_build_variant_v1", "blender_version": bpy.app.version_string,
                                          "out_blend": str(out_blend), "blend_sha256": hashlib.sha256(out_blend.read_bytes()).hexdigest(),
                                          "parent_blend_sha256": src_sha, "action_name": act.name, "removed_action": orig_name,
                                          "key_values_changed": changed, "provenance": prov}))
    return 0


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=("dump", "build_variant"))
    for name in ("blend", "report", "overrides", "frames", "spec"):
        ap.add_argument("--" + name)
    o = ap.parse_args(argv)
    try:
        return {"dump": dump, "build_variant": build_variant}[o.mode](o)
    except SystemExit:
        raise
    except Exception:  # Blender may otherwise exit 0 after a Python exception
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    sys.exit(main())
