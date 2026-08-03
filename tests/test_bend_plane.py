import math

from pose.bend_plane import stabilize_bend_planes
from pose.pose_lifter import LiftedPoseFrame, LiftedPosePoint, LiftedPoseSequence
from pose.root_motion import RootMotionFrame, RootMotionSequence


def _point(name, position): return LiftedPosePoint(name, position, 1, 0)


def test_stabilizer_reflects_a_nonambiguous_mirrored_knee():
    base = {
        "left_hip": _point("left_hip", (0, 0, 0)), "left_knee": _point("left_knee", (0, 0, -1)),
        "left_ankle": _point("left_ankle", (0, 1, -2)),
        "right_hip": _point("right_hip", (1, 0, 0)), "right_knee": _point("right_knee", (1, 0, -1)), "right_ankle": _point("right_ankle", (1, 1, -2)),
        "left_shoulder": _point("left_shoulder", (0, 0, 1)), "left_elbow": _point("left_elbow", (0, 0, 2)), "left_wrist": _point("left_wrist", (0, 1, 3)),
        "right_shoulder": _point("right_shoulder", (1, 0, 1)), "right_elbow": _point("right_elbow", (1, 0, 2)), "right_wrist": _point("right_wrist", (1, 1, 3)),
    }
    flipped = dict(base); flipped["left_ankle"] = _point("left_ankle", (0, -1, -2))
    pose = LiftedPoseSequence(frames=[LiftedPoseFrame(0, 0, base), LiftedPoseFrame(1, .02, flipped)])
    roots = RootMotionSequence(frames=[RootMotionFrame(i, i / 50, 0, (0,1,0), (1,0,0), 1, None, {n:p.position for n,p in points.items()}) for i,points in enumerate((base,flipped))])
    result = stabilize_bend_planes(pose, roots, min_bend_degrees=12)
    assert result.frames[1].points["left_ankle"].position[1] > 0
