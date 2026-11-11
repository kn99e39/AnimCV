#!/usr/bin/env python3
"""EXPERIMENTAL (opt-in, Batch 136) Blender-side export adapter: frozen W129 B Action -> UniMate own-rig GLB.

Runs ONLY inside Blender's own Python (``blender -b --python-exit-code 1 -P <this file> -- export --blend ... ``).
Opens a BYTE COPY of a frozen .blend (the input file is never saved), and in memory:

  1. adds ONE static, unskinned edit bone ``pelvis`` (head at the thigh-head midpoint, 5 units up, roll 0) and parents
     ``spine_01``, ``thigh_l``, ``thigh_r`` to it (keep offset).  UniMate's exporter otherwise keeps only the heaviest
     root subtree and would drop both legs; the zero-weight helper roots ``interaction``/``center_of_mass`` are left
     alone (the official exporter prunes them).
  2. moves the armature OBJECT rotation (the source-derived, per-frame root orientation; object location is constant
     0, scale 0.01) onto ``pelvis``: object rotation keys are removed, the object rotation set to identity, and pelvis
     keyed with  Basis = Rest^-1 . R_obj(f) . Rest  (rotation + location about the armature origin) at EVERY frame
     1..240.  Every anatomical bone's world matrix is therefore unchanged (checked on all 240 frames; refused > 1e-5).
  3. exports a GLB (skin, active action, forced sampling) restricted to the declared frame range.

Nothing else is changed: no smoothing, no resampling, no channel or rest edit beyond the declared adapter.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ADAPTER_ROOT = "pelvis"
REPARENT = ("spine_01", "thigh_l", "thigh_r")
HELPERS = ("interaction", "center_of_mass")
WORLD_TOL = 1e-5


def _fcurves(action):
    out = []
    if hasattr(action, "layers"):
        for layer in action.layers:
            for strip in layer.strips:
                for bag in strip.channelbags:
                    out.extend((bag, fc) for fc in bag.fcurves)
    else:
        out.extend((action, fc) for fc in action.fcurves)
    return out


def export(o):
    import bpy
    from mathutils import Quaternion
    src_sha = hashlib.sha256(Path(o.blend).read_bytes()).hexdigest()
    if o.expected_sha and src_sha != o.expected_sha:
        raise RuntimeError(f"REFUSED_SOURCE_IDENTITY {src_sha}")
    out = Path(o.out)
    if out.exists():
        raise RuntimeError(f"refusing to overwrite {out}")
    f0, f1 = (int(x) for x in o.frames.split(","))
    bpy.ops.wm.open_mainfile(filepath=str(o.blend))
    arms = [ob for ob in bpy.data.objects if ob.type == "ARMATURE"]
    if len(arms) != 1:
        raise RuntimeError(f"expected one armature, found {len(arms)}")
    arm = arms[0]
    sc = bpy.context.scene
    fps = sc.render.fps / sc.render.fps_base
    if abs(fps - 30.0) > 1e-9:
        raise RuntimeError(f"REFUSED_NOT_30_FPS {fps}")
    if tuple(arm.delta_rotation_quaternion) != (1.0, 0.0, 0.0, 0.0) or arm.parent is not None or tuple(arm.location) != (0.0, 0.0, 0.0):
        raise RuntimeError("REFUSED_OBJECT_TRANSFORM unexpected delta/parent/location")
    names = [b.name for b in arm.data.bones]
    anat = [n for n in names if n not in HELPERS]
    n_frames = sc.frame_end
    before, objq = {}, {}
    for f in range(1, n_frames + 1):
        sc.frame_set(f)
        objq[f] = arm.rotation_quaternion.copy()
        before[f] = {n: (arm.matrix_world @ arm.pose.bones[n].matrix).copy() for n in names}
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.mode_set(mode="EDIT")
    eb = arm.data.edit_bones
    if ADAPTER_ROOT in eb:
        raise RuntimeError("REFUSED_ADAPTER_ROOT_EXISTS")
    p = eb.new(ADAPTER_ROOT)
    p.head = (0.0, (eb["thigh_l"].head.y + eb["thigh_r"].head.y) / 2, eb["thigh_l"].head.z)
    p.tail = (p.head.x, p.head.y, p.head.z + 5.0)
    p.roll = 0.0
    for n in REPARENT:
        if eb[n].parent is not None:
            raise RuntimeError(f"REFUSED_NOT_A_ROOT {n}")
        eb[n].parent = p
        eb[n].use_connect = False
    bpy.ops.object.mode_set(mode="OBJECT")
    act = arm.animation_data.action
    removed = 0
    for bag, fc in _fcurves(act):
        if fc.data_path == "rotation_quaternion":
            bag.fcurves.remove(fc)
            removed += 1
    if removed != 4:
        raise RuntimeError(f"REFUSED_OBJECT_ROTATION_CURVES {removed}")
    arm.rotation_mode = "QUATERNION"
    arm.rotation_quaternion = Quaternion((1.0, 0.0, 0.0, 0.0))
    pb = arm.pose.bones[ADAPTER_ROOT]
    pb.rotation_mode = "QUATERNION"
    rest = arm.data.bones[ADAPTER_ROOT].matrix_local.copy()
    for f in range(1, n_frames + 1):
        basis = rest.inverted() @ objq[f].to_matrix().to_4x4() @ rest
        loc, rq, _ = basis.decompose()
        pb.location, pb.rotation_quaternion = loc, rq
        pb.keyframe_insert("location", frame=f)
        pb.keyframe_insert("rotation_quaternion", frame=f)
    for _, fc in _fcurves(act):
        for kp in fc.keyframe_points:
            kp.interpolation = "CONSTANT"
    err, worst = 0.0, {}
    for f in range(1, n_frames + 1):
        sc.frame_set(f)
        for n in anat:
            m = arm.matrix_world @ arm.pose.bones[n].matrix
            e = max(abs(m[i][j] - before[f][n][i][j]) for i in range(4) for j in range(4))
            worst[n] = max(worst.get(n, 0.0), e)
            err = max(err, e)
    if err > WORLD_TOL:
        raise RuntimeError(f"REFUSED_WORLD_POSE_CHANGED {err}")
    sc.frame_start, sc.frame_end = f0, f1
    sc.frame_set(f0)
    out.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.export_scene.gltf(filepath=str(out), export_format="GLB", export_animations=True, export_frame_range=True,
                              export_force_sampling=True, export_skins=True, export_def_bones=False,
                              export_anim_single_armature=True, export_animation_mode="ACTIVE_ACTIONS",
                              export_bake_animation=False, export_optimize_animation_size=False,
                              export_reset_pose_bones=True, export_apply=False)
    rep = {"schema": "animcv_exp_w136_export_v1", "blender_version": bpy.app.version_string, "source_blend": str(o.blend),
           "source_blend_sha256": src_sha, "action_name": act.name, "frames": [f0, f1], "scene_fps": fps,
           "adapter": {"root": ADAPTER_ROOT, "reparented": list(REPARENT), "object_rotation_curves_removed": removed,
                       "pelvis_keys_frames": [1, n_frames], "helpers_untouched": list(HELPERS)},
           "world_matrix_max_abs_diff_anatomical": err, "world_matrix_worst_bones": sorted(worst.items(), key=lambda x: -x[1])[:5],
           "n_anatomical_bones_checked": len(anat), "frames_checked": [1, n_frames],
           "glb": str(out), "glb_sha256": hashlib.sha256(out.read_bytes()).hexdigest()}
    Path(o.report).write_text(json.dumps(rep, indent=1))
    return 0


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=("export",))
    for name in ("blend", "out", "frames", "report", "expected-sha"):
        ap.add_argument("--" + name)
    o = ap.parse_args(argv)
    try:
        return export(o)
    except SystemExit:
        raise
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    sys.exit(main())
