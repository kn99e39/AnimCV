from pose.pose_lifter import LiftedPoseFrame, LiftedPosePoint, LiftedPoseSequence
from pose.quality_audit import audit_3d_pose
from pose.root_motion import RootMotionFrame, RootMotionSequence


def _sequence(knee_x: float = 0.1):
    points = {
        "left_hip": LiftedPosePoint("left_hip", (0, 0, 0), 1, 0),
        "left_knee": LiftedPosePoint("left_knee", (knee_x, 0, -1), 1, 0),
        "left_ankle": LiftedPosePoint("left_ankle", (0.2, 0, -2), 1, 0),
        "right_hip": LiftedPosePoint("right_hip", (1, 0, 0), 1, 0),
        "right_knee": LiftedPosePoint("right_knee", (1.1, 0, -1), 1, 0),
        "right_ankle": LiftedPosePoint("right_ankle", (1.2, 0, -2), 1, 0),
        "left_shoulder": LiftedPosePoint("left_shoulder", (0, 0, 1), 1, 0),
        "left_elbow": LiftedPosePoint("left_elbow", (0, 0, 2), 1, 0),
        "left_wrist": LiftedPosePoint("left_wrist", (0, 0, 3), 1, 0),
        "right_shoulder": LiftedPosePoint("right_shoulder", (1, 0, 1), 1, 0),
        "right_elbow": LiftedPosePoint("right_elbow", (1, 0, 2), 1, 0),
        "right_wrist": LiftedPosePoint("right_wrist", (1, 0, 3), 1, 0),
    }
    lifted = LiftedPoseSequence(frames=[LiftedPoseFrame(i, i / 50, points) for i in range(30)])
    root = RootMotionSequence(frames=[RootMotionFrame(i, i / 50, 0, (0, 1, 0), (1, 0, 0), 1, None, {n: p.position for n, p in points.items()}) for i in range(30)])
    return lifted, root


def test_audit_3d_pose_accepts_stable_character_space_sequence():
    lifted, root = _sequence()
    report = audit_3d_pose(lifted, root)
    assert report["passed"]
    assert report["knee_bend_direction_flips"] == {"left": 0, "right": 0}
    assert report["bend_direction_flips"] == {
        "left_knee": 0, "right_knee": 0, "left_elbow": 0, "right_elbow": 0,
    }


def test_audit_3d_pose_rejects_length_variation():
    lifted, root = _sequence()
    mutable = list(lifted.frames)
    changed = dict(mutable[-1].points)
    changed["left_ankle"] = LiftedPosePoint("left_ankle", (0.2, 0, -20), 1, 0)
    mutable[-1] = LiftedPoseFrame(29, 29 / 50, changed)
    report = audit_3d_pose(LiftedPoseSequence(frames=mutable), root)
    assert not report["passed"]
    assert any("limb length CV" in failure for failure in report["failures"])


def test_audit_3d_pose_detects_an_intentional_knee_mirror_flip():
    lifted, root = _sequence()
    frames = []
    for index, frame in enumerate(root.frames):
        points = dict(frame.character_points)
        points["right_hip"] = (1.0, 0.0, 0.0)
        points["right_knee"] = (1.0, 1.0, -1.0)
        points["right_ankle"] = (1.0, 0.0 if index % 2 == 0 else 3.0, -2.0)
        frames.append(RootMotionFrame(index, index / 50, 0, (0, 1, 0), (1, 0, 0), 1, None, points))
    report = audit_3d_pose(lifted, RootMotionSequence(frames=frames))
    assert not report["passed"]
    assert any("limb bend direction flips" in failure for failure in report["failures"])
