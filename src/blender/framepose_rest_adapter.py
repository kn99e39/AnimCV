"""Extract the actual imported armature rest pose for FramePose FK."""

from __future__ import annotations

import hashlib
from pathlib import Path

from retarget.framepose_target_rest import RestBone, TargetRigRestPose


def extract_target_rest_pose(armature_object, *, rig_id: str, source_path: str) -> TargetRigRestPose:
    import bpy

    if armature_object.type != "ARMATURE":
        raise ValueError("expected a Blender armature object")
    bones = {}
    for bone in armature_object.data.bones:
        bones[bone.name] = RestBone(
            bone.name, bone.parent.name if bone.parent else None,
            tuple(tuple(float(x) for x in row) for row in bone.matrix_local),
            tuple(float(x) for x in bone.head_local),
            tuple(float(x) for x in bone.tail_local))
    if not bones:
        raise ValueError("imported armature has no bones")
    return TargetRigRestPose(rig_id, bones, {
        "authority": "blender_imported_armature",
        "source_fbx_sha256": hashlib.sha256(Path(source_path).read_bytes()).hexdigest(),
        "source_filename": Path(source_path).name,
        "blender_version": bpy.app.version_string,
        "armature_object": armature_object.name,
        "armature_matrix_world": [[float(x) for x in row] for row in armature_object.matrix_world],
        "importer": "bpy.ops.import_scene.fbx; factory empty scene; default importer options",
    })
