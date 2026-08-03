import numpy as np

from pose.pose_lifter import VideoPose3DConfig, VideoPose3DLifter
from pose.pose_types import PoseFrame, PoseLandmark, PoseSequence
from pose.root_motion import estimate_root_motion


def _frame(index: int) -> PoseFrame:
    points = {
        "left_shoulder": PoseLandmark("left_shoulder", 0, 0, 0.2, True),
        "right_shoulder": PoseLandmark("right_shoulder", 1, 0, 0.2, True),
        "left_hip": PoseLandmark("left_hip", 0, 1, 0.2, True),
        "right_hip": PoseLandmark("right_hip", 1, 1, 0.2, True),
        "pelvis": PoseLandmark("pelvis", 0.5, 1, 0.2, True),
        "neck": PoseLandmark("neck", 0.5, 0, 0.2, True),
        "spine": PoseLandmark("spine", 0.5, 0.5, 0.2, True),
        "head": PoseLandmark("head", 0.5, -1, 0.2, True),
        "left_elbow": PoseLandmark("left_elbow", 0, 0, 0.2, True),
        "right_elbow": PoseLandmark("right_elbow", 1, 0, 0.2, True),
        "left_wrist": PoseLandmark("left_wrist", 0, 0, 0.2, True),
        "right_wrist": PoseLandmark("right_wrist", 1, 0, 0.2, True),
        "left_knee": PoseLandmark("left_knee", 0, 2, 0.2, True),
        "right_knee": PoseLandmark("right_knee", 1, 2, 0.2, True),
        "left_ankle": PoseLandmark("left_ankle", 0, 3, 0.2, True),
        "right_ankle": PoseLandmark("right_ankle", 1, 3, 0.2, True),
    }
    return PoseFrame(index, float(index), points)


def test_lift_and_root_motion_preserve_one_observation_policy(monkeypatch):
    poses = PoseSequence([_frame(0)], 25.0, observation_confidence_threshold=0.1)
    lifter = VideoPose3DLifter(VideoPose3DConfig("model.pth", min_observation_confidence=0.1))
    predicted = np.zeros((17, 3), dtype=np.float32)
    predicted[1, 0], predicted[4, 0] = 1.0, -1.0
    predicted[11, 0], predicted[14, 0] = 1.0, -1.0
    monkeypatch.setattr(lifter, "_predict_window", lambda *_: predicted)
    lifted = lifter.lift(poses, (100, 100))
    root = estimate_root_motion(lifted)
    assert lifted.observation_confidence_threshold == 0.1
    assert root.observation_confidence_threshold == 0.1
