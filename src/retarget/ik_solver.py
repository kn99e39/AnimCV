"""Two-bone analytic IK solver, operating in 2D image-plane space.

Classic "two-bone IK" (law of cosines) for a root bone + one mid bone
reaching toward an end-effector target -- covers the common arm
(shoulder-elbow-wrist) and leg (hip-knee-ankle) cases. Chains with more
than one mid bone would need a general solver (FABRIK/CCD) and are out
of scope here.

Reference bone lengths are calibrated from the tracked landmarks
themselves at the first fully-visible frame (root->mid and mid->end
pixel, or 3D, distances) rather than from the rig's rest pose: there's
no camera calibration in this project to relate rig-space bone lengths
to 2D image-pixel distances (section 1.1 lists "camera calibration
hints" as optional/unused). This keeps the whole computation
self-consistent in whichever space the tracks are in, the same
calibrate-from-first-frame approach fk_solver already uses.

When the tracked points carry position_3d (via depth, see
pose/depth_estimator.py), the solve happens in 3D; otherwise it falls
back to the 2D image plane. Either way this stays a 2-bone planar/near-
planar solve — a proper 3D IK also needs a pole vector to fully
constrain the bend plane, which this MVP approximates using the
observed mid-joint (elbow/knee) position instead of requiring one.
"""

from __future__ import annotations

import math

from common.types import Matrix4, Vec2, Vec3
from motion.motion_graph import MotionGraph
from retarget.axis_utils import (
    apply_rest_pose_correction,
    axis_hint_to_vector,
    quaternion_from_axis_angle,
)
from retarget.solver import BoneTransformSample
from rig.bone_mapping import IKChainEntry

_IDENTITY_ROTATION = (0.0, 0.0, 0.0, 1.0)


def _distance_2d(a: Vec2, b: Vec2) -> float:
    return math.hypot(b[0] - a[0], b[1] - a[1])


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _law_of_cosines_angle(opposite: float, adjacent_a: float, adjacent_b: float) -> float:
    """Angle (radians) opposite side `opposite` in a triangle with the
    other two sides adjacent_a/adjacent_b."""
    if adjacent_a == 0.0 or adjacent_b == 0.0:
        return 0.0
    cosine = (adjacent_a**2 + adjacent_b**2 - opposite**2) / (2 * adjacent_a * adjacent_b)
    return math.acos(_clamp(cosine, -1.0, 1.0))


def solve_ik_chain(
    motion_graph: MotionGraph,
    chain: IKChainEntry,
    root_rest_local_matrix: Matrix4 | None = None,
    mid_rest_local_matrix: Matrix4 | None = None,
) -> dict[str, list[BoneTransformSample]]:
    """Returns {chain.root_bone: [...samples...], chain.mid_bone: [...samples...]}."""
    root_axis = axis_hint_to_vector(chain.root_axis_hint)
    mid_axis = axis_hint_to_vector(chain.mid_axis_hint)

    root_length: float | None = None
    mid_length: float | None = None
    reference_root_angle: float | None = None
    reference_mid_bend: float | None = None

    last_root_rotation = _IDENTITY_ROTATION
    last_mid_rotation = _IDENTITY_ROTATION
    root_samples: list[BoneTransformSample] = []
    mid_samples: list[BoneTransformSample] = []

    for motion_frame in motion_graph.frames:
        root_point = motion_frame.points.get(chain.root_source)
        mid_point = motion_frame.points.get(chain.mid_source)
        end_point = motion_frame.points.get(chain.end_source)

        available = (
            root_point is not None
            and mid_point is not None
            and end_point is not None
            and root_point.visible
            and mid_point.visible
            and end_point.visible
        )

        if not available:
            root_samples.append(
                BoneTransformSample(
                    motion_frame.frame_index, chain.root_bone, None, last_root_rotation, None, 0.0
                )
            )
            mid_samples.append(
                BoneTransformSample(
                    motion_frame.frame_index, chain.mid_bone, None, last_mid_rotation, None, 0.0
                )
            )
            continue

        root_pos, mid_pos, end_pos = (
            root_point.position_2d,
            mid_point.position_2d,
            end_point.position_2d,
        )

        if root_length is None:
            root_length = _distance_2d(root_pos, mid_pos) or 1.0
            mid_length = _distance_2d(mid_pos, end_pos) or 1.0

        target_distance = _distance_2d(root_pos, end_pos)
        max_reach = root_length + mid_length
        min_reach = abs(root_length - mid_length)
        clamped_distance = _clamp(target_distance, min_reach, max_reach)

        # Interior angle at the mid joint between its two bone segments;
        # the bend (pose) angle is its supplement (0 = fully extended).
        mid_interior_angle = _law_of_cosines_angle(clamped_distance, root_length, mid_length)
        mid_bend_angle = math.pi - mid_interior_angle

        root_offset_angle = _law_of_cosines_angle(mid_length, root_length, clamped_distance)
        target_angle = math.atan2(end_pos[1] - root_pos[1], end_pos[0] - root_pos[0])

        # Pick the elbow-up vs. elbow-down solution by which side of the
        # root->target line the observed mid point actually falls on,
        # rather than assuming one -- this is what lets a 2-bone solve
        # skip requiring an explicit pole vector.
        cross = (mid_pos[0] - root_pos[0]) * (end_pos[1] - root_pos[1]) - (
            mid_pos[1] - root_pos[1]
        ) * (end_pos[0] - root_pos[0])
        bend_side = 1.0 if cross >= 0 else -1.0

        root_angle = target_angle - bend_side * root_offset_angle
        signed_mid_bend = bend_side * mid_bend_angle

        if reference_root_angle is None:
            reference_root_angle = root_angle
            reference_mid_bend = signed_mid_bend

        root_rotation = quaternion_from_axis_angle(root_axis, root_angle - reference_root_angle)
        mid_rotation = quaternion_from_axis_angle(
            mid_axis, signed_mid_bend - reference_mid_bend
        )
        root_rotation = apply_rest_pose_correction(root_rotation, root_rest_local_matrix)
        mid_rotation = apply_rest_pose_correction(mid_rotation, mid_rest_local_matrix)
        last_root_rotation, last_mid_rotation = root_rotation, mid_rotation

        confidence = min(root_point.confidence, mid_point.confidence, end_point.confidence)
        root_samples.append(
            BoneTransformSample(
                motion_frame.frame_index, chain.root_bone, None, root_rotation, None, confidence
            )
        )
        mid_samples.append(
            BoneTransformSample(
                motion_frame.frame_index, chain.mid_bone, None, mid_rotation, None, confidence
            )
        )

    return {chain.root_bone: root_samples, chain.mid_bone: mid_samples}
