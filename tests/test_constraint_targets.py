from pose.constraint_targets import prepare_constraint_targets
from pose.pose_lifter import LiftedPoseFrame, LiftedPosePoint, LiftedPoseSequence


def test_constraint_target_preparation_returns_matching_final_pose_and_root_motion():
    points = {
        "pelvis": (0, 0, 0), "left_hip": (-1, 0, 0), "left_knee": (-1, 0.2, -1), "left_ankle": (-1, 0.5, -2),
        "right_hip": (1, 0, 0), "right_knee": (1, 0.2, -1), "right_ankle": (1, 0.5, -2),
        "left_shoulder": (-1, 0, 1), "left_elbow": (-1, 0.2, 2), "left_wrist": (-1, 0.4, 3),
        "right_shoulder": (1, 0, 1), "right_elbow": (1, 0.2, 2), "right_wrist": (1, 0.4, 3),
    }
    lifted = LiftedPoseSequence(frames=[LiftedPoseFrame(index, index / 50, {
        name: LiftedPosePoint(name, position, 1.0, 0.0) for name, position in points.items()
    }) for index in range(3)])
    pose, root = prepare_constraint_targets(lifted, smoothing_window=1)
    assert len(pose.frames) == len(root.frames) == 3
    assert root.frames[0].character_points["right_shoulder"] == (1.0, 0.0, 1.0)
