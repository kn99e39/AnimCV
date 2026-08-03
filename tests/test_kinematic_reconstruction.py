import math

import pytest

from pose.kinematic_reconstruction import reconstruct_kinematic_pose
from pose.pose_lifter import LiftedPoseFrame, LiftedPosePoint, LiftedPoseSequence


def _point(name, position):
    return LiftedPosePoint(name, position, 1.0, 0.0)


def test_reconstruction_fixes_limb_lengths_to_subject_median():
    base = {
        "left_hip": _point("left_hip", (0, 0, 0)),
        "left_knee": _point("left_knee", (0, 0, -1)),
        "left_ankle": _point("left_ankle", (0, 0, -2)),
        "right_hip": _point("right_hip", (1, 0, 0)),
        "right_knee": _point("right_knee", (1, 0, -1)),
        "right_ankle": _point("right_ankle", (1, 0, -2)),
        "left_shoulder": _point("left_shoulder", (0, 0, 2)),
        "left_elbow": _point("left_elbow", (0, 0, 3)),
        "left_wrist": _point("left_wrist", (0, 0, 4)),
        "right_shoulder": _point("right_shoulder", (1, 0, 2)),
        "right_elbow": _point("right_elbow", (1, 0, 3)),
        "right_wrist": _point("right_wrist", (1, 0, 4)),
    }
    noisy = dict(base)
    noisy["left_knee"] = _point("left_knee", (0, 0, -1.5))
    noisy["left_ankle"] = _point("left_ankle", (0, 0, -3.5))
    result = reconstruct_kinematic_pose(LiftedPoseSequence(frames=[LiftedPoseFrame(0, 0, base), LiftedPoseFrame(1, .02, noisy)]))

    for frame in result.frames:
        points = frame.points
        assert math.dist(points["left_hip"].position, points["left_knee"].position) == pytest.approx(1.25)
        assert math.dist(points["left_knee"].position, points["left_ankle"].position) == pytest.approx(1.5)
