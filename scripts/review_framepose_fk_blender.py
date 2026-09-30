#!/usr/bin/env python3
"""Render isolated, fully-known BaseRig FramePose FK poses in Blender.

Each named case is a standalone pose.  No AnimationClip conversion or
interpolation can turn an UNAVAILABLE sample into an observed rotation.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


def arguments():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rig", type=Path, required=True)
    parser.add_argument("--rest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--case", action="append", required=True,
                        help="label:path_to_fk_result:frame_index; repeat")
    return parser.parse_args(argv)


def _render_setup(bpy, armature):
    from mathutils import Vector

    scene = bpy.context.scene
    scene.render.engine = "BLENDER_WORKBENCH"
    scene.render.resolution_x = 960
    scene.render.resolution_y = 960
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.display.shading.light = "STUDIO"
    scene.display.shading.color_type = "OBJECT"
    scene.display.shading.show_cavity = True
    scene.render.film_transparent = False
    if scene.world is None:
        scene.world = bpy.data.worlds.new("FramePose FK review world")
    scene.world.color = (0.12, 0.12, 0.12)

    camera_data = bpy.data.cameras.new("FramePose FK review camera")
    camera = bpy.data.objects.new("FramePose FK review camera", camera_data)
    scene.collection.objects.link(camera)
    scene.camera = camera
    camera.location = (3.2, -4.8, 2.5)
    target = Vector((0, 0, 1.2))
    camera.rotation_euler = (target - camera.location).to_track_quat("-Z", "Y").to_euler()
    camera_data.type = "ORTHO"
    camera_data.ortho_scale = 3.0
    for obj in bpy.data.objects:
        if obj.type == "MESH":
            obj.color = (0.57, 0.61, 0.67, 1.0)
    armature.show_in_front = True


def _reset_pose(armature):
    for bone in armature.pose.bones:
        bone.rotation_mode = "QUATERNION"
        bone.rotation_quaternion = (1, 0, 0, 0)
        bone.location = (0, 0, 0)
        bone.scale = (1, 1, 1)


def _make_bone_proxies(bpy, armature):
    selected = ("spine_", "neck_", "clavicle_", "upperarm_", "lowerarm_",
                "hand_", "thigh_", "calf_", "foot_")
    proxies = {}
    for bone in armature.pose.bones:
        if not bone.name.startswith(selected):
            continue
        bpy.ops.mesh.primitive_cylinder_add(vertices=12, radius=0.013, depth=1)
        segment = bpy.context.object
        segment.name = f"FK review segment {bone.name}"
        if bone.name == "spine_01":
            segment.color = (1.0, 0.35, 0.12, 1.0)
        elif bone.name in ("upperarm_l", "lowerarm_l"):
            segment.color = (0.07, 0.9, 0.31, 1.0)
        else:
            segment.color = (0.75, 0.81, 0.92, 1.0)
        proxies[bone.name] = segment
    return proxies


def _update_bone_proxies(bpy, armature, proxies):
    for name, segment in proxies.items():
        bone = armature.pose.bones[name]
        start = armature.matrix_world @ bone.head
        end = armature.matrix_world @ bone.tail
        direction = end - start
        segment.location = (start + end) / 2
        segment.rotation_mode = "QUATERNION"
        segment.rotation_quaternion = direction.to_track_quat("Z", "Y")
        segment.scale = (1, 1, direction.length)
    bpy.context.view_layer.update()


def _apply_pose(bpy, armature, samples):
    for name, sample in samples.items():
        if sample.rotation_status != "known" or sample.rotation_local is None:
            raise ValueError(f"{name} is not fully known; refusing Blender review")
        bone = armature.pose.bones.get(name)
        if bone is None:
            raise ValueError(f"mapped bone absent from Blender armature: {name}")
        x, y, z, w = sample.rotation_local
        bone.rotation_quaternion = (w, x, y, z)
    bpy.context.view_layer.update()


def _direction_errors(armature, samples, alignment):
    from mathutils import Quaternion, Vector

    aq = Quaternion((alignment[3], *alignment[:3]))
    errors = {}
    for name, sample in samples.items():
        if sample.source_pair is None:
            continue
        bone = armature.pose.bones[name]
        actual = (bone.tail - bone.head).normalized()
        expected = aq @ Vector(sample.raw_direction)
        cosine = max(-1.0, min(1.0, actual.dot(expected.normalized())))
        errors[name] = math.degrees(math.acos(cosine))
    return errors


def main():
    args = arguments()
    try:
        import bpy
        from blender.executor import BlenderExecutor
        from blender.framepose_rest_adapter import extract_target_rest_pose
        from retarget.framepose_fk import load_fk_result
        from retarget.framepose_target_rest import load_target_rest_pose

        bpy.ops.wm.read_factory_settings(use_empty=True)
        armature = BlenderExecutor().import_rig(str(args.rig))
        rest = load_target_rest_pose(args.rest)
        actual_rest = extract_target_rest_pose(armature, rig_id=rest.rig_id,
                                               source_path=str(args.rig))
        if actual_rest.digest() != rest.digest():
            raise ValueError("execution rig rest pose differs from FK snapshot")
        args.out_dir.mkdir(parents=True, exist_ok=True)
        _render_setup(bpy, armature)
        proxies = _make_bone_proxies(bpy, armature)
        report = {"schema": "animcv_framepose_fk_blender_review_v1",
                  "rest_snapshot_digest": rest.digest(),
                  "blender_version": bpy.app.version_string,
                  "cases": {}}
        _reset_pose(armature)
        bpy.context.view_layer.update()
        _update_bone_proxies(bpy, armature, proxies)
        scene = bpy.context.scene
        scene.render.filepath = str(args.out_dir / "rest.png")
        bpy.ops.render.render(write_still=True)
        bpy.ops.wm.save_as_mainfile(filepath=str(args.out_dir / "rest.blend"))
        report["cases"]["rest"] = {"png": "rest.png", "blend": "rest.blend"}
        for item in args.case:
            label, path, index = item.split(":", 2)
            frame_index = int(index)
            result = load_fk_result(path)
            if result.provenance["rest_snapshot_digest"] != rest.digest():
                raise ValueError(f"{label}: FK rest digest mismatch")
            samples = {sample.target_bone: sample for sample in result.samples
                       if sample.frame_index == frame_index}
            if not samples:
                raise ValueError(f"{label}: frame {frame_index} absent")
            _reset_pose(armature)
            _apply_pose(bpy, armature, samples)
            _update_bone_proxies(bpy, armature, proxies)
            errors = _direction_errors(
                armature, samples,
                result.provenance["canonical_heading_zero_to_armature_xyzw"])
            scene.frame_set(frame_index)
            scene.render.filepath = str(args.out_dir / f"{label}.png")
            bpy.ops.render.render(write_still=True)
            bpy.ops.wm.save_as_mainfile(filepath=str(args.out_dir / f"{label}.blend"))
            report["cases"][label] = {
                "sequence_id": result.sequence_id, "frame_index": frame_index,
                "png": f"{label}.png", "blend": f"{label}.blend",
                "direction_error_degrees": errors,
                "local_rotations": {name: list(sample.rotation_local)
                                    for name, sample in samples.items()},
            }
        (args.out_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True))
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        print(f"FramePose FK Blender review failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
