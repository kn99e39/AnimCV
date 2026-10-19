#!/usr/bin/env python3
"""EXPERIMENT (Worklog 91): Blender replay of FIXED / RUNTIME / GT root heading on BaseRig (torso + J1X left arm).

    Blender -b -P scripts/render_runtime_heading_rig_blender.py -- --rig examples/BaseRig.fbx \
        --rest examples/e2e_demo/baserig_blender_rest.json --cases cases.json --out-dir renders

For every case and mode one Action keys the Armature Object rotation (W60 object ownership: rest * A Rz(psi) A^-1)
and the ``upperarm_l`` / ``lowerarm_l`` local quaternions at the exact sampled frame indices (scene 30 fps, CONSTANT
interpolation; rows without an available arm keep the previous arm key).  Bone proxies only (BaseRig has no skinned
mesh).  A colored arrow from the pelvis (spine_01 head) shows the mode's body-forward direction; a thin green arrow and arm show the
GT heading and GT-reference arm (EVALUATION ONLY).  Identical cameras (video-camera view and side view), lighting and
settings for all modes; a ``.blend`` with the three Actions is saved per case.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

MODES = ("FIXED", "RUNTIME", "GT")
ARM = {"FIXED": (1.0, 0.55, 0.10, 1.0), "RUNTIME": (0.84, 0.10, 0.84, 1.0), "GT": (0.10, 0.85, 0.95, 1.0)}
GT_COLOR = (0.13, 0.75, 0.13, 1.0)
BODY_COLOR = (0.55, 0.57, 0.62, 1.0)
BONES = ("upperarm_l", "lowerarm_l")


def _args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("rig", "rest", "cases", "out_dir"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--size", type=int, default=480)
    parser.add_argument("--only", default=None)
    return parser.parse_args(argv)


def _place(segment, start, end):
    direction = end - start
    segment.location = (start + end) / 2
    segment.rotation_mode = "QUATERNION"
    segment.rotation_quaternion = direction.to_track_quat("Z", "Y")
    segment.scale = (1, 1, max(direction.length, 1e-6))


def render(o) -> int:
    import bpy
    from mathutils import Quaternion, Vector
    from blender.executor import BlenderExecutor
    from blender.framepose_object_owner_adapter import bind_imported_armature
    from retarget.framepose_target_rest import load_target_rest_pose
    from review_framepose_fk_blender import _make_bone_proxies, _render_setup, _update_bone_proxies
    from review_framepose_ik_gt_attribution_blender import _aim, _camera, _segment, _sphere

    data = json.loads(o.cases.read_text())
    bpy.ops.wm.read_factory_settings(use_empty=True)
    armature = BlenderExecutor().import_rig(str(o.rig))
    rest = load_target_rest_pose(o.rest)
    if rest.digest() != data["rest_snapshot_digest"]:
        raise ValueError("case export was built against another rest snapshot")
    binding = bind_imported_armature(armature, rest, source_path=str(o.rig))
    _render_setup(bpy, armature)
    scene = bpy.context.scene
    scene.render.resolution_x = scene.render.resolution_y = o.size
    scene.render.fps = 30
    proxies = _make_bone_proxies(bpy, armature)
    for seg in proxies.values():
        seg.color = BODY_COLOR
    rest_q = Quaternion(binding.rest_rotation_wxyz)
    loc = Vector(binding.location)
    scale = binding.scale[0]
    a = data["canonical_to_armature_xyzw"]
    aq = Quaternion((a[3], *a[:3]))
    to_q = lambda xyzw: Quaternion((xyzw[3], *xyzw[:3]))  # noqa: E731
    world_of = lambda p, objq: loc + rest_q @ (objq @ (Vector(p) * scale))  # noqa: E731
    axis = lambda v: (rest_q @ (aq @ Vector(v))).normalized()  # noqa: E731 (camera canonical axes in world)
    forward_cam, up, right = axis((0, 1, 0)), axis((0, 0, 1)), axis((1, 0, 0))
    armature.rotation_mode = "QUATERNION"
    for pb in armature.pose.bones:
        pb.rotation_mode = "QUATERNION"
        pb.rotation_quaternion = (1, 0, 0, 0)
    armature.rotation_quaternion = rest_q
    bpy.context.view_layer.update()
    pelvis_rest_local = armature.pose.bones["spine_01"].head.copy()
    cams = {"video": _camera(bpy, "W91 video"), "side": _camera(bpy, "W91 side")}
    for c in cams.values():
        c.data.ortho_scale = 2.2
    fwd_seg = _segment(bpy, "mode forward", (0, 0, 0), (0, 0, 1), ARM["RUNTIME"], 0.012)
    fwd_tip = _sphere(bpy, "mode forward tip", (0, 0, 0), ARM["RUNTIME"], 0.03)
    gt_fwd = _segment(bpy, "GT forward", (0, 0, 0), (0, 0, 1), GT_COLOR, 0.006)
    gt_up = _segment(bpy, "GT upper", (0, 0, 0), (0, 0, 1), GT_COLOR, 0.006)
    gt_lo = _segment(bpy, "GT lower", (0, 0, 0), (0, 0, 1), GT_COLOR, 0.006)
    report = {"schema": "worklog91_blender_heading_replay", "blender_version": bpy.app.version_string, "interpolation": "CONSTANT",
              "scene_fps": 30, "cases": {}}
    only = set(o.only.split(",")) if o.only else None
    root_head = data["target_chain"].get("root_head")
    for key, case in data["cases"].items():
        if only and key not in only:
            continue
        out = o.out_dir / key
        out.mkdir(parents=True, exist_ok=True)
        frames = case["frames"]
        f0 = frames[0]["frame_index"]
        bf = [fr["frame_index"] - f0 + 1 for fr in frames]
        scene.frame_start, scene.frame_end = bf[0], bf[-1]
        pelvis_w = loc + rest_q @ (pelvis_rest_local * scale)
        centre = pelvis_w + up * 0.1
        _aim(cams["video"], centre - forward_cam * 4.0, centre, up)
        _aim(cams["side"], centre - right * 4.0, centre, up)
        actions, fk = {}, {}
        for mode in MODES:
            act = bpy.data.actions.new(f"W91 {key} {mode}")
            act.use_fake_user = True
            armature.animation_data_create()
            armature.animation_data.action = act
            for fr, f in zip(frames, bf):
                m = fr["modes"].get(mode)
                yaw = fr["yaw_deg"].get(mode)
                if yaw is not None:
                    objq = to_q(fr["modes"][mode]["object_rotation_armature_xyzw"]) if m is not None else \
                        aq @ Quaternion((0, 0, 1), math.radians(yaw)) @ aq.inverted()
                    armature.rotation_quaternion = rest_q @ objq
                    armature.keyframe_insert("rotation_quaternion", frame=f)
                if m is not None:
                    for bone, field in zip(BONES, ("proximal_local_xyzw", "distal_local_xyzw")):
                        x, y, z, w = m[field]
                        armature.pose.bones[bone].rotation_quaternion = (w, x, y, z)
                        armature.pose.bones[bone].keyframe_insert("rotation_quaternion", frame=f)
            for fc in _fcurves(act):
                for kp in fc.keyframe_points:
                    kp.interpolation = "CONSTANT"
            actions[mode] = act
            for bone in BONES:
                proxies[bone].color = ARM[mode]
            fwd_seg.color = fwd_tip.color = ARM[mode]
            err = 0.0
            for i, (fr, f) in enumerate(zip(frames, bf)):
                scene.frame_set(f)
                bpy.context.view_layer.update()
                _update_bone_proxies(bpy, armature, proxies)
                pel = armature.matrix_world @ armature.pose.bones["spine_01"].head
                m = fr["modes"].get(mode)
                if m is not None:
                    objq = to_q(m["object_rotation_armature_xyzw"])
                    err = max(err, (armature.matrix_world @ armature.pose.bones["lowerarm_l"].tail - world_of(m["end"], objq)).length / scale)
                yaw = fr["yaw_deg"].get(mode)
                for obj in (fwd_seg, fwd_tip):
                    obj.hide_render = yaw is None
                if yaw is not None:
                    psi = math.radians(yaw)
                    fdir = (rest_q @ (aq @ Vector((-math.sin(psi), math.cos(psi), 0.0)))).normalized()
                    _place(fwd_seg, pel, pel + fdir * 0.5)
                    fwd_tip.location = pel + fdir * 0.5
                g = fr["modes"].get("GT_REF")
                gy = fr["yaw_deg"].get("GT")
                for obj in (gt_fwd, gt_up, gt_lo):
                    obj.hide_render = gy is None
                if gy is not None:
                    psi = math.radians(gy)
                    gdir = (rest_q @ (aq @ Vector((-math.sin(psi), math.cos(psi), 0.0)))).normalized()
                    _place(gt_fwd, pel, pel + gdir * 0.6)
                if g is not None and root_head is not None:
                    gq = to_q(g["object_rotation_armature_xyzw"])
                    r_w, m_w, e_w = world_of(root_head, gq), world_of(g["mid"], gq), world_of(g["end"], gq)
                    _place(gt_up, r_w, m_w)
                    _place(gt_lo, m_w, e_w)
                else:
                    gt_up.hide_render = gt_lo.hide_render = True
                for view, cam in cams.items():
                    scene.camera = cam
                    scene.render.filepath = str(out / f"{mode}_{view}_{i:03d}.png")
                    bpy.ops.render.render(write_still=True)
            fk[mode] = err
        for bone in BONES:
            proxies[bone].color = BODY_COLOR
        armature.animation_data.action = actions["RUNTIME"]
        bpy.ops.wm.save_as_mainfile(filepath=str(out / f"{key}.blend"), copy=True)
        report["cases"][key] = {"frames": len(frames), "fk_end_vs_export_max": fk, "actions": {m: a.name for m, a in actions.items()}}
        print(key, fk, flush=True)
    (o.out_dir / "blender_report.json").write_text(json.dumps(report, indent=1))
    return 0


def _fcurves(action):
    if hasattr(action, "fcurves") and action.fcurves is not None:
        try:
            return list(action.fcurves)
        except TypeError:
            pass
    out = []
    for layer in getattr(action, "layers", []):
        for strip in layer.strips:
            for bag in strip.channelbags:
                out.extend(bag.fcurves)
    return out


def main():
    o = _args()
    try:
        return render(o)
    except Exception as exc:
        import traceback
        traceback.print_exc()
        print(f"W91 Blender replay failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
