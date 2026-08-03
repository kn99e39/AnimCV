"""Ordered preparation of the 3D targets consumed by a future rig solver."""

from __future__ import annotations

from pose.bend_plane import stabilize_bend_planes
from pose.kinematic_reconstruction import reconstruct_kinematic_pose
from pose.pose_lifter import LiftedPoseSequence
from pose.root_motion import RootMotionSequence, estimate_root_motion


def prepare_constraint_targets(
    lifted: LiftedPoseSequence,
    min_confidence: float = 0.3,
    smoothing_window: int = 5,
    max_yaw_step_degrees: float = 15.0,
    min_bend_degrees: float = 12.0,
) -> tuple[LiftedPoseSequence, RootMotionSequence]:
    """Run the only valid R2→R4→R3 ordering.

    The final yaw pass is required because bend stabilization changes camera
    points; it regenerates matching ``character_points`` without interpreting
    the temporary pelvis fit as global translation.
    """
    kinematic = reconstruct_kinematic_pose(lifted, min_confidence)
    root_for_bend = estimate_root_motion(kinematic, smoothing_window, max_yaw_step_degrees)
    stabilized = stabilize_bend_planes(kinematic, root_for_bend, min_bend_degrees)
    final_root = estimate_root_motion(stabilized, smoothing_window, max_yaw_step_degrees)
    return stabilized, final_root
