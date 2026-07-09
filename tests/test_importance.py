import math

import pytest

from optimize.importance import quaternion_angle, score_track_importance
from retarget.axis_utils import quaternion_from_axis_angle
from retarget.solver import AnimationTrack, BoneTransformSample


def _rotation_sample(frame_index, angle_deg, confidence=0.9):
    rotation = quaternion_from_axis_angle((0.0, 0.0, 1.0), math.radians(angle_deg))
    return BoneTransformSample(
        frame_index=frame_index,
        bone_name="upper_arm.L",
        location=None,
        rotation=rotation,
        scale=None,
        confidence=confidence,
    )


def test_quaternion_angle_identical_is_zero():
    q = quaternion_from_axis_angle((0.0, 0.0, 1.0), math.radians(30))
    assert quaternion_angle(q, q) == pytest.approx(0.0, abs=1e-9)


def test_quaternion_angle_quarter_turn():
    a = quaternion_from_axis_angle((0.0, 0.0, 1.0), 0.0)
    b = quaternion_from_axis_angle((0.0, 0.0, 1.0), math.pi / 2)
    assert quaternion_angle(a, b) == pytest.approx(math.pi / 2)


def test_quaternion_angle_handles_double_cover():
    q = quaternion_from_axis_angle((0.0, 0.0, 1.0), math.radians(45))
    negated = tuple(-c for c in q)
    assert quaternion_angle(q, negated) == pytest.approx(0.0, abs=1e-9)


def test_score_track_importance_boundaries_are_always_one():
    track = AnimationTrack(
        bone_name="upper_arm.L",
        samples=[_rotation_sample(i, angle_deg=i * 2) for i in range(6)],
    )

    scores = score_track_importance(track)

    assert scores[0] == 1.0
    assert scores[-1] == 1.0


def test_score_track_importance_locked_frame_is_one():
    track = AnimationTrack(
        bone_name="upper_arm.L",
        samples=[_rotation_sample(i, angle_deg=i * 2) for i in range(6)],
    )

    scores = score_track_importance(track, locked_frames={3})

    assert scores[3] == 1.0


def test_score_track_importance_velocity_spike_scores_higher_than_steady_motion():
    # slow drift, then a sharp velocity spike at frame 3, then slow again
    angles = [0, 5, 10, 50, 55, 60]
    track = AnimationTrack(
        bone_name="upper_arm.L",
        samples=[_rotation_sample(i, angle_deg=a) for i, a in enumerate(angles)],
    )

    scores = score_track_importance(track)

    # frame 3 is a velocity peak (40deg jump) surrounded by 5deg steps
    assert scores[3] > scores[1]
    assert scores[3] > scores[2]


def test_score_track_importance_empty_track_returns_empty():
    assert score_track_importance(AnimationTrack(bone_name="x", samples=[])) == []
