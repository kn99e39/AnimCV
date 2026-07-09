"""FK direction-based retargeting (Architecture_v2.md section 7.2).

v2 excludes depth estimation in its original scope (section 1.3), but
this project has since added an optional Depth Anything V2 backend
(see pose/depth_estimator.py) that can populate MotionPoint.position_3d
via depth-sampled 2D landmarks. When both endpoints of a "direction"
mapping have position_3d for both the current and reference frame, this
solver computes a real 3D shortest-arc rotation instead of the
2D-image-plane-angle-around-a-fixed-axis approximation from section
7.4 — falling back to the 2D method automatically wherever 3D data
isn't available (section 1.1: depth stays optional; "system still
works without depth" is preserved). For a ``landmark``/``point``
(single-anchor) mapping there's no direction to rotate from, so it's
tracked as a translation offset from the reference-frame position
instead (3D offset when available, 2D otherwise).

Both rotation and translation deltas are computed in world/image axes
and then re-expressed in the target bone's own local space via
axis_utils.apply_rest_pose_correction[_to_vector], using the bone's
rest_local_matrix from RigProfile when the caller provides one.
"""

from __future__ import annotations

import math

from common.types import Matrix4, Quaternion, Vec2, Vec3
from motion.motion_graph import MotionFrame, MotionGraph
from retarget.axis_utils import (
    apply_rest_pose_correction,
    apply_rest_pose_correction_to_vector,
    axis_hint_to_vector,
    quaternion_from_axis_angle,
    quaternion_from_vectors,
)
from retarget.solver import BoneTransformSample
from rig.bone_mapping import BoneMappingEntry

_IDENTITY_ROTATION: Quaternion = (0.0, 0.0, 0.0, 1.0)


def frame_direction(motion_frame: MotionFrame, source_a: str, source_b: str) -> Vec2 | None:
    """Unit 2D direction vector from source_a to source_b, or None if
    either point is missing/not visible/coincident this frame."""
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


def frame_direction_3d(motion_frame: MotionFrame, source_a: str, source_b: str) -> Vec3 | None:
    """Unit 3D direction vector from source_a to source_b, or None if
    either point is missing/not visible/lacks depth/is coincident."""
    point_a = motion_frame.points.get(source_a)
    point_b = motion_frame.points.get(source_b)
    if (
        point_a is None
        or point_b is None
        or not point_a.visible
        or not point_b.visible
        or point_a.position_3d is None
        or point_b.position_3d is None
    ):
        return None
    dx = point_b.position_3d[0] - point_a.position_3d[0]
    dy = point_b.position_3d[1] - point_a.position_3d[1]
    dz = point_b.position_3d[2] - point_a.position_3d[2]
    length = math.sqrt(dx * dx + dy * dy + dz * dz)
    if length == 0.0:
        return None
    return (dx / length, dy / length, dz / length)


def signed_angle_delta(reference: Vec2, current: Vec2) -> float:
    """Signed angle (radians) that rotates `reference` into `current`."""
    rx, ry = reference
    cx, cy = current
    cross = rx * cy - ry * cx
    dot = rx * cx + ry * cy
    return math.atan2(cross, dot)


def solve_direction_bone(
    motion_graph: MotionGraph,
    bone_name: str,
    entry: BoneMappingEntry,
    rest_local_matrix: Matrix4 | None = None,
) -> list[BoneTransformSample]:
    if len(entry.source_names) != 2:
        raise ValueError(
            f"direction mapping for {bone_name!r} needs exactly 2 source names, "
            f"got {entry.source_names!r}"
        )
    source_a, source_b = entry.source_names
    axis = axis_hint_to_vector(entry.axis_hint)

    reference_direction: Vec2 | None = None
    reference_direction_3d: Vec3 | None = None
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

        direction_3d = frame_direction_3d(motion_frame, source_a, source_b)
        if direction_3d is not None and reference_direction_3d is None:
            reference_direction_3d = direction_3d

        if direction_3d is not None and reference_direction_3d is not None:
            # Real depth data available for both endpoints in both this
            # frame and the reference frame: use an actual 3D
            # shortest-arc rotation instead of the 2D-plane
            # approximation. axis_hint doesn't apply here -- the
            # rotation axis comes from the observed 3D geometry itself.
            rotation = quaternion_from_vectors(reference_direction_3d, direction_3d)
        else:
            angle = signed_angle_delta(reference_direction, direction)
            rotation = quaternion_from_axis_angle(axis, angle)

        rotation = apply_rest_pose_correction(rotation, rest_local_matrix)
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
    motion_graph: MotionGraph,
    bone_name: str,
    entry: BoneMappingEntry,
    rest_local_matrix: Matrix4 | None = None,
) -> list[BoneTransformSample]:
    if len(entry.source_names) != 1:
        raise ValueError(
            f"landmark/custom_point mapping for {bone_name!r} needs exactly 1 source name, "
            f"got {entry.source_names!r}"
        )
    source_name = entry.source_names[0]

    reference_position: Vec2 | None = None
    reference_position_3d: Vec3 | None = None
    last_location: Vec3 = (0.0, 0.0, 0.0)
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
        if point.position_3d is not None and reference_position_3d is None:
            reference_position_3d = point.position_3d

        if point.position_3d is not None and reference_position_3d is not None:
            delta: Vec3 = (
                point.position_3d[0] - reference_position_3d[0],
                point.position_3d[1] - reference_position_3d[1],
                point.position_3d[2] - reference_position_3d[2],
            )
        else:
            delta = (
                point.position_2d[0] - reference_position[0],
                point.position_2d[1] - reference_position[1],
                0.0,
            )

        last_location = apply_rest_pose_correction_to_vector(delta, rest_local_matrix)
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
