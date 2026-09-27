import math

import pytest

from pose.pose_lifter import LiftedPoseFrame, LiftedPosePoint, LiftedPoseSequence
from pose.root_orientation_diagnostic import observe_yaw, summarize


def _frame(index: int, yaw: float, *, valid=True, hip_yaw=None) -> LiftedPoseFrame:
    def axis(angle):
        right = (math.cos(angle), math.sin(angle), 0.0)
        return tuple(-v for v in right), right

    shoulder_left, shoulder_right = axis(yaw)
    hip_left, hip_right = axis(yaw if hip_yaw is None else hip_yaw)
    points = {
        "left_shoulder": LiftedPosePoint("left_shoulder", shoulder_left, 0.9, 0.1, observation_valid=valid),
        "right_shoulder": LiftedPosePoint("right_shoulder", shoulder_right, 0.9, 0.1, observation_valid=valid),
        "left_hip": LiftedPosePoint("left_hip", hip_left, 0.8, 0.2, observation_valid=valid),
        "right_hip": LiftedPosePoint("right_hip", hip_right, 0.8, 0.2, observation_valid=valid),
    }
    return LiftedPoseFrame(index, index / 30.0, points)


def test_observe_yaw_recovers_the_known_angle():
    sequence = LiftedPoseSequence(frames=[_frame(0, math.pi / 4)], source_fps=30.0)
    observations = observe_yaw(sequence)
    assert observations[0].yaw_radians == pytest.approx(math.pi / 4)
    assert observations[0].reliable is True


def test_observe_yaw_is_unknown_when_no_bilateral_pair_is_present():
    frame = LiftedPoseFrame(0, 0.0, {})
    observations = observe_yaw(LiftedPoseSequence(frames=[frame], source_fps=30.0))
    assert observations[0].yaw_radians is None


def test_observe_yaw_reports_disagreement_between_shoulders_and_hips():
    sequence = LiftedPoseSequence(frames=[_frame(0, 0.0, hip_yaw=math.pi / 2)], source_fps=30.0)
    observations = observe_yaw(sequence)
    assert observations[0].disagreement_degrees == pytest.approx(90.0, abs=1.0)


def test_summarize_counts_flips_and_unknowns():
    sequence = LiftedPoseSequence(frames=[
        _frame(0, 0.0), _frame(1, math.pi), _frame(2, 0.0),
    ], source_fps=30.0)
    observations = observe_yaw(sequence)
    report = summarize("synthetic:flip", observations)
    assert report.flip_count == 2
    assert report.unknown_count == 0


def test_summarize_separates_reliable_from_unreliable_flip_rate():
    sequence = LiftedPoseSequence(frames=[
        _frame(0, 0.0, valid=True), _frame(1, math.pi, valid=False), _frame(2, 0.0, valid=True),
    ], source_fps=30.0)
    observations = observe_yaw(sequence)
    report = summarize("synthetic:reliability", observations)
    assert report.unreliable_frame_count == 1
    assert report.flip_rate_among_unreliable == pytest.approx(1.0)
