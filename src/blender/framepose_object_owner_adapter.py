"""Bounded Blender execution of one whole-rig yaw plus unchanged local FK."""

from __future__ import annotations

from dataclasses import dataclass

from retarget.framepose_object_owner import ObjectOwnedFkResult
from retarget.framepose_target_rest import TargetRigRestPose


@dataclass(frozen=True)
class ImportedArmatureBinding:
    rest_snapshot_digest: str
    location: tuple[float, float, float]
    scale: tuple[float, float, float]
    rest_rotation_wxyz: tuple[float, float, float, float]
    top_level_bones: tuple[str, ...]


def bind_imported_armature(armature, rest: TargetRigRestPose, *, source_path: str) -> ImportedArmatureBinding:
    from blender.framepose_rest_adapter import extract_target_rest_pose

    if armature.type != "ARMATURE" or armature.parent is not None:
        raise ValueError("object-owned FK needs an unparented Blender Armature Object")
    actual = extract_target_rest_pose(armature, rig_id=rest.rig_id, source_path=source_path)
    if actual.digest() != rest.digest():
        raise ValueError("imported armature differs from the authoritative rest snapshot")
    _, object_q, world_scale = armature.matrix_world.decompose()
    scale = tuple(float(v) for v in world_scale)
    if min(scale) <= 0 or max(scale) - min(scale) > max(scale) * 1e-5:
        raise ValueError("object-owned yaw requires positive uniform imported object scale")
    top_level = tuple(sorted(name for name, bone in rest.bones.items() if bone.parent is None))
    pose_bone_names = {bone.name for bone in armature.pose.bones}
    if pose_bone_names != set(rest.bones):
        raise ValueError("Blender pose bones differ from rest snapshot")
    return ImportedArmatureBinding(
        rest.digest(), tuple(float(v) for v in armature.location),
        tuple(float(v) for v in armature.scale),
        (float(object_q.w), float(object_q.x), float(object_q.y), float(object_q.z)),
        top_level)


def apply_object_owned_frame(armature, binding: ImportedArmatureBinding,
                             result: ObjectOwnedFkResult, frame_index: int) -> None:
    """Apply one fully known frame; refuse unavailable samples before mutation."""
    import bpy
    from mathutils import Quaternion

    if result.provenance["rest_snapshot_digest"] != binding.rest_snapshot_digest:
        raise ValueError("FK result/rest binding mismatch")
    roots = [sample for sample in result.rig_rotations if sample.frame_index == frame_index]
    bones = [sample for sample in result.pose_bone_samples if sample.frame_index == frame_index]
    if len(roots) != 1 or not bones:
        raise ValueError("frame is missing object or pose-bone samples")
    root = roots[0]
    if root.rotation_status != "known" or root.rotation_armature_space is None:
        raise ValueError("Root Orientation unavailable; refusing Blender frame")
    if any(sample.rotation_status != "known" or sample.rotation_local is None for sample in bones):
        raise ValueError("pose-bone FK unavailable; refusing Blender frame")
    if len({sample.target_bone for sample in bones}) != len(bones):
        raise ValueError("duplicate pose-bone FK target")
    if any(armature.pose.bones.get(sample.target_bone) is None for sample in bones):
        raise ValueError("FK target absent from imported armature")

    for pose_bone in armature.pose.bones:
        pose_bone.rotation_mode = "QUATERNION"
        pose_bone.rotation_quaternion = (1, 0, 0, 0)
        pose_bone.location = (0, 0, 0)
        pose_bone.scale = (1, 1, 1)
    object_base = Quaternion(binding.rest_rotation_wxyz)
    x, y, z, w = root.rotation_armature_space
    object_delta = Quaternion((w, x, y, z))
    armature.rotation_mode = "QUATERNION"
    armature.rotation_quaternion = object_base @ object_delta
    for sample in bones:
        x, y, z, w = sample.rotation_local
        armature.pose.bones[sample.target_bone].rotation_quaternion = (w, x, y, z)
    bpy.context.view_layer.update()
    if tuple(armature.location) != binding.location or tuple(armature.scale) != binding.scale:
        raise RuntimeError("object-owned FK unexpectedly changed object translation or scale")
