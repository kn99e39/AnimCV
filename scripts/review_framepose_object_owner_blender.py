#!/usr/bin/env python3
"""Review BaseRig object-owned yaw and Worklog 59 bone-owned turning A/B."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))


def _args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rig", type=Path, required=True)
    parser.add_argument("--rest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--case", action="append", required=True,
                        help="label:object_result_json:frame_index")
    parser.add_argument("--turning-bone-fk", type=Path, required=True)
    parser.add_argument("--turning-frame", type=int, required=True)
    return parser.parse_args(argv)


def _root_branch_deltas(armature, rest, binding):
    from mathutils import Quaternion

    base_object_q = Quaternion(binding.rest_rotation_wxyz)
    object_delta_world = (armature.matrix_world.to_quaternion() @ base_object_q.inverted())
    out = {}
    for name in binding.top_level_bones:
        pose = armature.pose.bones[name]
        rest_q = base_object_q @ pose.bone.matrix_local.to_quaternion()
        posed_q = (armature.matrix_world @ pose.matrix).to_quaternion()
        branch_delta = posed_q @ rest_q.inverted()
        dot = abs(sum(a * b for a, b in zip(object_delta_world, branch_delta)))
        error = 2 * math.acos(min(1.0, dot))
        out[name] = {
            "inherits_object_delta_error_degrees": math.degrees(error),
            "pose_bone_rotation_wxyz": [float(v) for v in pose.rotation_quaternion],
            "same_armature_object": pose.id_data.as_pointer() == armature.as_pointer(),
        }
    return out


def _mapped_world_direction_errors(armature, samples, alignment, binding):
    from mathutils import Quaternion, Vector

    base_q = Quaternion(binding.rest_rotation_wxyz)
    aq = Quaternion((alignment[3], *alignment[:3]))
    errors = {}
    for name, sample in samples.items():
        bone = armature.pose.bones[name]
        actual = ((armature.matrix_world @ bone.tail) - (armature.matrix_world @ bone.head)).normalized()
        expected = base_q @ (aq @ Vector(sample.raw_direction))
        dot = max(-1.0, min(1.0, actual.dot(expected.normalized())))
        errors[name] = math.degrees(math.acos(dot))
    return errors


def _render(bpy, armature, proxies, out_dir, label):
    from review_framepose_fk_blender import _update_bone_proxies

    _update_bone_proxies(bpy, armature, proxies)
    bpy.context.scene.render.filepath = str(out_dir / f"{label}.png")
    bpy.ops.render.render(write_still=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(out_dir / f"{label}.blend"))


def _set_rest_object(armature, binding):
    armature.rotation_mode = "QUATERNION"
    armature.rotation_quaternion = binding.rest_rotation_wxyz
    armature.location = binding.location
    armature.scale = binding.scale


def main():
    options = _args()
    try:
        import bpy
        from blender.executor import BlenderExecutor
        from blender.framepose_object_owner_adapter import (
            apply_object_owned_frame, bind_imported_armature,
        )
        from retarget.framepose_fk import load_fk_result
        from retarget.framepose_object_owner import load_object_owned_fk
        from retarget.framepose_target_rest import load_target_rest_pose
        from review_framepose_fk_blender import _make_bone_proxies, _render_setup, _reset_pose

        bpy.ops.wm.read_factory_settings(use_empty=True)
        armature = BlenderExecutor().import_rig(str(options.rig))
        rest = load_target_rest_pose(options.rest)
        binding = bind_imported_armature(armature, rest, source_path=str(options.rig))
        _render_setup(bpy, armature)
        proxies = _make_bone_proxies(bpy, armature)
        options.out_dir.mkdir(parents=True, exist_ok=True)
        report = {"schema": "animcv_framepose_object_owner_blender_review_v1",
                  "rest_snapshot_digest": rest.digest(),
                  "blender_version": bpy.app.version_string,
                  "top_level_bones": list(binding.top_level_bones),
                  "cases": {}}

        _reset_pose(armature)
        _set_rest_object(armature, binding)
        bpy.context.view_layer.update()
        _render(bpy, armature, proxies, options.out_dir, "rest")
        report["cases"]["rest"] = {"png": "rest.png", "blend": "rest.blend"}

        for item in options.case:
            label, filename, index = item.split(":", 2)
            frame_index = int(index)
            result = load_object_owned_fk(filename)
            _set_rest_object(armature, binding)
            apply_object_owned_frame(armature, binding, result, frame_index)
            samples = {sample.target_bone: sample for sample in result.pose_bone_samples
                       if sample.frame_index == frame_index}
            branch = _root_branch_deltas(armature, rest, binding)
            errors = _mapped_world_direction_errors(
                armature, samples,
                result.provenance["canonical_heading_zero_to_armature_xyzw"], binding)
            root = next(s for s in result.rig_rotations if s.frame_index == frame_index)
            _render(bpy, armature, proxies, options.out_dir, label)
            report["cases"][label] = {
                "sequence_id": result.sequence_id, "frame_index": frame_index,
                "yaw_radians": root.yaw_radians,
                "rig_object_rotation_xyzw": list(root.rotation_armature_space),
                "object_location_unchanged": tuple(armature.location) == binding.location,
                "object_scale_unchanged": tuple(armature.scale) == binding.scale,
                "branch_coverage": branch,
                "mapped_world_direction_error_degrees": errors,
                "limb_local_xyzw": {name: list(sample.rotation_local)
                                    for name, sample in samples.items()},
                "png": f"{label}.png", "blend": f"{label}.blend",
            }

        old = load_fk_result(options.turning_bone_fk)
        turning_object = next((load_object_owned_fk(item.split(":", 2)[1])
                               for item in options.case if item.startswith("turning:")), None)
        if turning_object is None:
            raise ValueError("turning object case required for A/B")
        old_samples = {s.target_bone: s for s in old.samples
                       if s.frame_index == options.turning_frame}
        new_samples = {s.target_bone: s for s in turning_object.pose_bone_samples
                       if s.frame_index == options.turning_frame}
        if any(old_samples[name].to_dict() != sample.to_dict()
               for name, sample in new_samples.items()):
            raise ValueError("A/B changed limb-local FK samples")
        _reset_pose(armature)
        _set_rest_object(armature, binding)
        for name, sample in old_samples.items():
            if sample.rotation_status != "known":
                raise ValueError("Worklog 59 turning bone-owned frame is unavailable")
            x, y, z, w = sample.rotation_local
            armature.pose.bones[name].rotation_quaternion = (w, x, y, z)
        bpy.context.view_layer.update()
        bone_branch = _root_branch_deltas(armature, rest, binding)
        _render(bpy, armature, proxies, options.out_dir, "turning_bone_owner")
        report["cases"]["turning_bone_owner"] = {
            "frame_index": options.turning_frame,
            "same_limb_local_samples_as_object_owner": True,
            "object_location_unchanged": tuple(armature.location) == binding.location,
            "branch_coverage": bone_branch,
            "png": "turning_bone_owner.png", "blend": "turning_bone_owner.blend",
        }
        (options.out_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True))
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        print(f"object-owner Blender review failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
