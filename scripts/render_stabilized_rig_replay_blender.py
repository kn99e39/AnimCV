#!/usr/bin/env python3
"""EXPERIMENT (Worklog 90): Blender replay of RAW / STABILIZED / G0X left-arm rig animation on BaseRig.

Run under Blender with ``cases.json`` from ``scripts/run_stabilized_rig_replay.py``:

    Blender -b -P scripts/render_stabilized_rig_replay_blender.py -- --rig examples/BaseRig.fbx \
        --rest examples/e2e_demo/baserig_blender_rest.json --cases cases.json --out-dir renders

For every case and mode a Blender Action keys the ``upperarm_l`` / ``lowerarm_l`` local quaternions at the exact
sampled frame indices (scene 30 fps, CONSTANT interpolation, so nothing is interpolated between model samples and
UNAVAILABLE rows hold the previous pose in every mode).  Root yaw is the export's constant psi (Worklog 60 object
ownership: Armature Object rotation = rest * A Rz(psi) A^-1, identical for all modes); every other bone stays at
rest.  The posed armature is rendered through bone proxies (BaseRig has
no skinned mesh) with identical cameras, lighting and settings for all modes: a video-camera view (looking along
+depth, image right = +X) and a side view (depth x up).  The GT-reference arm (EVALUATION ONLY) is a thin green
overlay.  A ``.blend`` with the three Actions is saved per case.  Blender FK of the keyed animation is checked against
the exported IK endpoint.
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

MODES = ("RAW", "STABILIZED", "G0X")
ARM = {"RAW": (0.10, 0.85, 0.95, 1.0), "STABILIZED": (0.84, 0.10, 0.84, 1.0), "G0X": (1.0, 0.55, 0.10, 1.0)}
GT_COLOR = (0.13, 0.75, 0.13, 1.0)
BODY_COLOR = (0.55, 0.57, 0.62, 1.0)
BONES = ("upperarm_l", "lowerarm_l")


def _args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("rig", "rest", "cases", "out_dir"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--size", type=int, default=480)
    parser.add_argument("--only", default=None, help="comma-separated case keys")
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
    for obj in bpy.data.objects:
        if obj.type == "MESH":
            obj.hide_render = True
    proxies = _make_bone_proxies(bpy, armature)
    for name, seg in proxies.items():
        seg.color = BODY_COLOR
    armature.rotation_mode = "QUATERNION"
    a = data["canonical_to_armature_xyzw"]
    aq = Quaternion((a[3], *a[:3]))
    yaw = math.radians(data.get("yaw_deg", 0.0))
    rest_q = Quaternion(binding.rest_rotation_wxyz)
    armature.rotation_quaternion = rest_q @ (aq @ Quaternion((0, 0, 1), yaw) @ aq.inverted())
    for pb in armature.pose.bones:
        pb.rotation_mode = "QUATERNION"
        pb.rotation_quaternion = (1, 0, 0, 0)
        pb.location = (0, 0, 0)
        pb.scale = (1, 1, 1)
    bpy.context.view_layer.update()
    world = armature.matrix_world.copy()
    to_world = lambda p: world @ Vector(p)  # noqa: E731
    # camera canonical axes in world: rest * A * v (independent of psi)
    axis = lambda v: (rest_q @ (aq @ Vector(v))).normalized()  # noqa: E731
    forward, up, right = axis((0, 1, 0)), axis((0, 0, 1)), axis((1, 0, 0))
    cams = {"video": _camera(bpy, "W90 video"), "side": _camera(bpy, "W90 side")}
    for c in cams.values():
        c.data.ortho_scale = 1.0
    gt_up = _segment(bpy, "GT upper", (0, 0, 0), (0, 0, 1), GT_COLOR, 0.006)
    gt_lo = _segment(bpy, "GT lower", (0, 0, 0), (0, 0, 1), GT_COLOR, 0.006)
    gt_end = _sphere(bpy, "GT wrist", (0, 0, 0), GT_COLOR, 0.012)
    report = {"schema": "worklog90_blender_rig_replay", "blender_version": bpy.app.version_string,
              "rest_snapshot_digest": rest.digest(), "yaw_deg": data.get("yaw_deg", 0.0), "interpolation": "CONSTANT", "scene_fps": 30, "cases": {}}
    only = set(o.only.split(",")) if o.only else None
    for key, case in data["cases"].items():
        if only and key not in only:
            continue
        out = o.out_dir / key
        out.mkdir(parents=True, exist_ok=True)
        frames = case["frames"]
        f0 = frames[0]["frame_index"]
        bf = [fr["frame_index"] - f0 + 1 for fr in frames]
        scene.frame_start, scene.frame_end = bf[0], bf[-1]
        # fixed framing per case (identical for all modes): centred between shoulder and the mean wrist
        centre = to_world(Vector(case["root"])) - up * 0.15
        _aim(cams["video"], centre - forward * 3.0, centre, up)
        _aim(cams["side"], centre - right * 3.0, centre, up)
        actions, fk = {}, {}
        for mode in MODES:
            act = bpy.data.actions.new(f"W90 {key} {mode}")
            act.use_fake_user = True
            armature.animation_data_create()
            armature.animation_data.action = act
            last = None
            for fr, f in zip(frames, bf):
                m = fr["modes"].get(mode)
                if m is None:
                    continue  # UNAVAILABLE: no key; CONSTANT interpolation holds the previous sample
                for bone, field in zip(BONES, ("proximal_local_xyzw", "distal_local_xyzw")):
                    x, y, z, w = m[field]
                    pb = armature.pose.bones[bone]
                    pb.rotation_quaternion = (w, x, y, z)
                    pb.keyframe_insert("rotation_quaternion", frame=f)
                last = f
            for fc in _fcurves(act):
                for kp in fc.keyframe_points:
                    kp.interpolation = "CONSTANT"
            actions[mode] = act
            if last is None:
                continue
            for seg_name in BONES:
                proxies[seg_name].color = ARM[mode]
            proxies.get("hand_l") and setattr(proxies["hand_l"], "color", ARM[mode])
            err = 0.0
            for i, (fr, f) in enumerate(zip(frames, bf)):
                scene.frame_set(f)
                bpy.context.view_layer.update()
                _update_bone_proxies(bpy, armature, proxies)
                m = fr["modes"].get(mode)
                if m is not None:
                    err = max(err, math.dist(tuple(armature.pose.bones["lowerarm_l"].tail), m["end"]))
                g = fr["modes"].get("GT_REF")
                for obj in (gt_up, gt_lo, gt_end):
                    obj.hide_render = g is None
                if g is not None:
                    r_w, m_w, e_w = to_world(case["root"]), to_world(g["mid"]), to_world(g["end"])
                    _place(gt_up, r_w, m_w)
                    _place(gt_lo, m_w, e_w)
                    gt_end.location = e_w
                for view, cam in cams.items():
                    scene.camera = cam
                    scene.render.filepath = str(out / f"{mode}_{view}_{i:03d}.png")
                    bpy.ops.render.render(write_still=True)
            fk[mode] = err
        for seg_name in BONES:
            proxies[seg_name].color = BODY_COLOR
        armature.animation_data.action = actions["STABILIZED"]
        bpy.ops.wm.save_as_mainfile(filepath=str(out / f"{key}.blend"), copy=True)
        report["cases"][key] = {"frames": len(frames), "blender_frames": [bf[0], bf[-1]], "fk_end_vs_export_max": fk,
                                "actions": {m: a.name for m, a in actions.items()}}
        print(key, fk, flush=True)
    (o.out_dir / "blender_report.json").write_text(json.dumps(report, indent=1))
    return 0


def _fcurves(action):
    """F-curves of an Action in Blender 4.x (action.fcurves) and 5.x (layered actions)."""
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
        print(f"W90 Blender replay failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
