from pose.pose_lifter import LiftedPoseFrame, LiftedPosePoint, LiftedPoseSequence
from pose.root_motion import estimate_root_motion
from pose.uncertainty import audit_pose_uncertainty


def test_uncertainty_marks_invalid_observation_unsafe_and_keeps_sources_traceable():
    positions = {
        "pelvis": (0, 0, 0), "left_hip": (-1, 0, 0), "left_knee": (-1, 0, -1), "left_ankle": (-1, 0, -2),
        "right_hip": (1, 0, 0), "right_knee": (1, 0, -1), "right_ankle": (1, 0, -2),
        "left_shoulder": (-1, 0, 1), "left_elbow": (-1, 0, 2), "left_wrist": (-1, 0, 3),
        "right_shoulder": (1, 0, 1), "right_elbow": (1, 0, 2), "right_wrist": (1, 0, 3),
    }
    frames = []
    for index in range(3):
        points = {name: LiftedPosePoint(name, value, 1.0, 0.0, observation_valid=name != "left_wrist")
                  for name, value in positions.items()}
        frames.append(LiftedPoseFrame(index, index / 50, points))
    raw = LiftedPoseSequence(frames=frames)
    root = estimate_root_motion(raw, smoothing_window=1)
    report = audit_pose_uncertainty(raw, raw, root)
    wrist = report["frames"][0]["joints"]["left_wrist"]
    assert wrist["unsafe"]
    assert wrist["components"]["invalid_observation"] == 1.0
    assert report["all_points_traceable"]
