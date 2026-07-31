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


def _world_rotation_to_pose_local(pose_bone, rotation):
    """Express an image/world-frame delta in this imported bone's local basis.

    FBX node transforms parsed through Assimp are not a reliable substitute for
    ``bone.matrix_local``: Blender can apply FBX axis and pre/post-rotation
    conversion while importing.  The conversion must therefore happen here,
    after the actual target rig has been imported.

    Fake bpy objects used in unit tests deliberately do not implement matrix
    math, in which case the supplied quaternion is kept unchanged.
    """
    bone = getattr(pose_bone, "bone", None)
    rest_matrix = getattr(bone, "matrix_local", None)
    if rest_matrix is None:
        return rotation

    from mathutils import Quaternion

    x, y, z, w = rotation
    world_rotation = Quaternion((w, x, y, z))
    rest_rotation = rest_matrix.to_quaternion()
    local_rotation = rest_rotation.inverted() @ world_rotation @ rest_rotation
    return (
        local_rotation.x,
        local_rotation.y,
        local_rotation.z,
        local_rotation.w,
    )


def _rotation_at_frame(track, frame_index):
    """Return a track's world-space rotation at ``frame_index``.

    Optimisation leaves different key times on parent and child tracks, so a
    child key cannot assume its parent has a sample at the same frame.
    """
    exact = next((sample for sample in track.samples if sample.frame_index == frame_index), None)
    if exact is not None:
        return exact.rotation

    before = [sample for sample in track.samples if sample.frame_index < frame_index]
    after = [sample for sample in track.samples if sample.frame_index > frame_index]
    if not before:
        return after[0].rotation
    if not after:
        return before[-1].rotation

    left, right = before[-1], after[0]
    factor = (frame_index - left.frame_index) / (right.frame_index - left.frame_index)
    from mathutils import Quaternion

    lx, ly, lz, lw = left.rotation
    rx, ry, rz, rw = right.rotation
    interpolated = Quaternion((lw, lx, ly, lz)).slerp(Quaternion((rw, rx, ry, rz)), factor)
    return (interpolated.x, interpolated.y, interpolated.z, interpolated.w)


def _parent_relative_rotation(pose_bone, tracks, sample):
    """Turn a desired world delta into the delta relative to its parent.

    Applying both a parent's and a child's world-space rotation directly to
    their local pose channels compounds the parent turn at elbows and knees.
    The child's channel must instead carry parent^-1 * child.
    """
    parent = getattr(pose_bone, "parent", None)
    parent_track = tracks.get(getattr(parent, "name", "")) if parent is not None else None
    if parent_track is None:
        return sample.rotation

    from mathutils import Quaternion

    px, py, pz, pw = _rotation_at_frame(parent_track, sample.frame_index)
    cx, cy, cz, cw = sample.rotation
    relative = Quaternion((pw, px, py, pz)).inverted() @ Quaternion((cw, cx, cy, cz))
    return (relative.x, relative.y, relative.z, relative.w)


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
                # AnimationClip uses the project-wide (x, y, z, w) layout,
                # whereas Blender's RNA Quaternion property is ordered
                # (w, x, y, z).  Assigning the former directly turns an
                # identity quaternion into a 180-degree Z rotation.
                relative_rotation = _parent_relative_rotation(
                    pose_bone, animation_clip.tracks, sample
                )
                x, y, z, w = _world_rotation_to_pose_local(pose_bone, relative_rotation)
                pose_bone.rotation_quaternion = (w, x, y, z)
                pose_bone.keyframe_insert(
                    data_path="rotation_quaternion", frame=sample.frame_index
                )
            inserted += 1

    for fcurve in _iter_fcurves(action):
        for keyframe_point in fcurve.keyframe_points:
            keyframe_point.interpolation = interpolation

    return inserted
