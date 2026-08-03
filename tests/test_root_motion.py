import math

import pytest

from pose.pose_lifter import LiftedPoseFrame, LiftedPosePoint, LiftedPoseSequence
from pose.root_motion import estimate_root_motion


def _lifted_frame(index: int, yaw: float) -> LiftedPoseFrame:
    right = (math.cos(yaw), math.sin(yaw), 0.0)
    left = tuple(-value for value in right)
    points = {
        "left_shoulder": LiftedPosePoint("left_shoulder", left, 0.9, 0.1),
        "right_shoulder": LiftedPosePoint("right_shoulder", right, 0.9, 0.1),
        "left_hip": LiftedPosePoint("left_hip", left, 0.8, 0.2),
        "right_hip": LiftedPosePoint("right_hip", right, 0.8, 0.2),
        "pelvis": LiftedPosePoint("pelvis", (0.0, 0.0, 0.0), 0.9, 0.1),
    }
    return LiftedPoseFrame(index, index / 50, points)


def test_estimate_root_motion_derives_yaw_and_character_space_points():
    lifted = LiftedPoseSequence(frames=[_lifted_frame(0, math.pi / 2)], source_fps=50)

    result = estimate_root_motion(lifted, smoothing_window=1)

    frame = result.frames[0]
    assert frame.root_yaw_radians == pytest.approx(math.pi / 2)
    assert frame.forward == pytest.approx((-1.0, 0.0, 0.0))
    assert frame.character_points["right_shoulder"] == pytest.approx((1.0, 0.0, 0.0))
    assert frame.root_translation is None
    assert result.translation_observable is False


def test_estimate_root_motion_unwraps_turn_across_pi_boundary():
    lifted = LiftedPoseSequence(frames=[_lifted_frame(0, 3.1), _lifted_frame(1, -3.1)])

    result = estimate_root_motion(lifted, smoothing_window=1)

    assert result.frames[1].root_yaw_radians > result.frames[0].root_yaw_radians
    assert result.frames[1].root_yaw_radians - result.frames[0].root_yaw_radians < 0.2


def test_estimate_root_motion_holds_an_implausible_single_frame_axis_flip():
    lifted = LiftedPoseSequence(frames=[_lifted_frame(0, 0.0), _lifted_frame(1, math.pi)])

    result = estimate_root_motion(lifted, smoothing_window=1, max_yaw_step_degrees=20)

    assert result.frames[1].root_yaw_radians == pytest.approx(0.0)
    assert result.frames[1].confidence < result.frames[0].confidence
    assert result.frames[1].yaw_held


def test_estimate_root_motion_weights_reliable_torso_axis_more_heavily():
    frame = _lifted_frame(0, 0.0)
    points = dict(frame.points)
    # The hip axis contradicts shoulders but its detector confidence is low.
    points["left_hip"] = LiftedPosePoint("left_hip", (0.0, -1.0, 0.0), 0.1, 0.9)
    points["right_hip"] = LiftedPosePoint("right_hip", (0.0, 1.0, 0.0), 0.1, 0.9)
    result = estimate_root_motion(LiftedPoseSequence(frames=[LiftedPoseFrame(0, 0.0, points)]), smoothing_window=1)
    assert result.frames[0].root_yaw_radians == pytest.approx(0.0, abs=0.25)
    assert result.frames[0].yaw_sources["shoulders"] > result.frames[0].yaw_sources["hips"]
