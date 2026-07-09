"""Insert AnimationClip samples as Blender pose-bone keyframes.

Only touches ``bpy`` inside the function body (lazy import), consistent
with every other adapter in this project: core logic stays importable
and testable without Blender installed, and only ``blender/*.py`` is
allowed to import ``bpy`` (Architecture_v2.md section 3.4).

Each AnimationTrack is homogeneous per Milestone 5's fk_solver: a
"direction"-mapped bone's samples all carry a rotation (location is
None), and a "landmark"/"point"-mapped bone's samples all carry a
location (rotation is the unused identity quaternion) — so which
property to keyframe is decided once per track from its first sample,
not per sample.
"""

from __future__ import annotations

from typing import Iterator

from retarget.solver import AnimationClip


def _iter_fcurves(action) -> Iterator:
    """Yield every FCurve on an Action, across Blender's animation data
    models. Blender 4.4+ moved to layered actions (layers -> strips ->
    channelbags -> fcurves) and 5.x removed the flat Action.fcurves
    compatibility accessor entirely — confirmed by actually running
    this against both a local 4.5 (has both) and 5.1 (layers only)
    install, not from documentation alone. Falls back to the legacy
    flat accessor for anything older than the layered model.
    """
    if hasattr(action, "layers"):
        for layer in action.layers:
            for strip in layer.strips:
                for channelbag in getattr(strip, "channelbags", []):
                    yield from channelbag.fcurves
    else:
        yield from action.fcurves


def write_keyframes(
    armature_object, animation_clip: AnimationClip, interpolation: str = "BEZIER"
) -> int:
    """Insert keyframes for every mapped bone track onto armature_object's pose bones.

    Bones named in the animation clip but absent from this armature are
    skipped rather than raising (section 6.5: partial mapping is
    expected). Returns the number of keyframes inserted.
    """
    import bpy

    if armature_object.animation_data is None:
        armature_object.animation_data_create()
    action = bpy.data.actions.new(name=animation_clip.name)
    armature_object.animation_data.action = action

    pose_bones = armature_object.pose.bones
    inserted = 0

    for bone_name, track in animation_clip.tracks.items():
        pose_bone = pose_bones.get(bone_name)
        if pose_bone is None or not track.samples:
            continue

        use_location = track.samples[0].location is not None
        if not use_location:
            pose_bone.rotation_mode = "QUATERNION"

        for sample in track.samples:
            bpy.context.scene.frame_set(sample.frame_index)
            if use_location:
                pose_bone.location = sample.location
                pose_bone.keyframe_insert(data_path="location", frame=sample.frame_index)
            else:
                pose_bone.rotation_quaternion = sample.rotation
                pose_bone.keyframe_insert(
                    data_path="rotation_quaternion", frame=sample.frame_index
                )
            inserted += 1

    for fcurve in _iter_fcurves(action):
        for keyframe_point in fcurve.keyframe_points:
            keyframe_point.interpolation = interpolation

    return inserted
