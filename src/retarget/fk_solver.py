"""FK direction-based retargeting (Architecture_v2.md section 7.2).

v2 excludes depth estimation (section 1.3), so this solver stays
2D-direction driven per section 7.4's explicit limitation: for a
``direction`` mapping it computes the signed image-plane angle between
a bone's source direction vector now and at a reference frame, then
turns that delta into a rotation around one axis (chosen via the Bone
Mapping Profile's ``axis_hint``, default +Z / the view axis). For a
``landmark``/``point`` (single-anchor) mapping there's no direction to
rotate from, so it's tracked as a translation offset from the
reference-frame position instead. The goal is a usable editable draft,
not accurate 3D motion capture.
"""

from __future__ import annotations

import math

from common.types import Vec2
from motion.motion_graph import MotionFrame, MotionGraph
from retarget.axis_utils import axis_hint_to_vector, quaternion_from_axis_angle
from retarget.solver import BoneTransformSample
from rig.bone_mapping import BoneMappingEntry

_IDENTITY_ROTATION = (0.0, 0.0, 0.0, 1.0)


def frame_direction(motion_frame: MotionFrame, source_a: str, source_b: str) -> Vec2 | None:
    """Unit direction vector from source_a to source_b, or None if either
    point is missing/not visible/coincident this frame."""
    point_a = motion_frame.points.get(source_a)
    point_b = motion_frame.points.get(source_b)
    if point_a is None or point_b is None or not point_a.visible or not point_b.visible:
        return None
    dx = point_b.position_2d[0] - point_a.position_2d[0]
    dy = point_b.position_2d[1] - point_a.position_2d[1]
    length = math.hypot(dx, dy)
    if length == 0.0:
        return None
    return (dx / length, dy / length)


def signed_angle_delta(reference: Vec2, current: Vec2) -> float:
    """Signed angle (radians) that rotates `reference` into `current`."""
    rx, ry = reference
    cx, cy = current
    cross = rx * cy - ry * cx
    dot = rx * cx + ry * cy
    return math.atan2(cross, dot)


def solve_direction_bone(
    motion_graph: MotionGraph, bone_name: str, entry: BoneMappingEntry
) -> list[BoneTransformSample]:
    if len(entry.source_names) != 2:
        raise ValueError(
            f"direction mapping for {bone_name!r} needs exactly 2 source names, "
            f"got {entry.source_names!r}"
        )
    source_a, source_b = entry.source_names
    axis = axis_hint_to_vector(entry.axis_hint)

    reference_direction: Vec2 | None = None
    last_rotation = _IDENTITY_ROTATION
    samples: list[BoneTransformSample] = []

    for motion_frame in motion_graph.frames:
        direction = frame_direction(motion_frame, source_a, source_b)

        if direction is None:
            # Hold the last valid rotation rather than snapping to
            # identity; zero confidence flags this frame for the
            # Milestone 6 keyframe optimizer.
            samples.append(
                BoneTransformSample(
                    frame_index=motion_frame.frame_index,
                    bone_name=bone_name,
                    location=None,
                    rotation=last_rotation,
                    scale=None,
                    confidence=0.0,
                )
            )
            continue

        if reference_direction is None:
            reference_direction = direction

        angle = signed_angle_delta(reference_direction, direction)
        rotation = quaternion_from_axis_angle(axis, angle)
        last_rotation = rotation

        point_a = motion_frame.points[source_a]
        point_b = motion_frame.points[source_b]
        samples.append(
            BoneTransformSample(
                frame_index=motion_frame.frame_index,
                bone_name=bone_name,
                location=None,
                rotation=rotation,
                scale=None,
                confidence=min(point_a.confidence, point_b.confidence),
            )
        )

    return samples


def solve_anchor_bone(
    motion_graph: MotionGraph, bone_name: str, entry: BoneMappingEntry
) -> list[BoneTransformSample]:
    if len(entry.source_names) != 1:
        raise ValueError(
            f"landmark/custom_point mapping for {bone_name!r} needs exactly 1 source name, "
            f"got {entry.source_names!r}"
        )
    source_name = entry.source_names[0]

    reference_position: Vec2 | None = None
    last_location = (0.0, 0.0, 0.0)
    samples: list[BoneTransformSample] = []

    for motion_frame in motion_graph.frames:
        point = motion_frame.points.get(source_name)

        if point is None or not point.visible:
            samples.append(
                BoneTransformSample(
                    frame_index=motion_frame.frame_index,
                    bone_name=bone_name,
                    location=last_location,
                    rotation=_IDENTITY_ROTATION,
                    scale=None,
                    confidence=0.0,
                )
            )
            continue

        if reference_position is None:
            reference_position = point.position_2d

        last_location = (
            point.position_2d[0] - reference_position[0],
            point.position_2d[1] - reference_position[1],
            0.0,
        )
        samples.append(
            BoneTransformSample(
                frame_index=motion_frame.frame_index,
                bone_name=bone_name,
                location=last_location,
                rotation=_IDENTITY_ROTATION,
                scale=None,
                confidence=point.confidence,
            )
        )

    return samples
