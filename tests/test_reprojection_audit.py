from pose.pose_lifter import LiftedPoseFrame, LiftedPosePoint, LiftedPoseSequence
from pose.pose_types import PoseFrame, PoseLandmark, PoseSequence
from pose.reprojection_audit import audit_weak_perspective_reprojection


def _sequence(scale: float = 1.0):
    coords = {"pelvis": (0.0, 0.0), "left_hip": (-1.0, -1.0), "right_hip": (1.0, -1.0)}
    observed = PoseSequence(frames=[PoseFrame(0, 0.0, {
        name: PoseLandmark(name, 100 + 40 * x, 200 - 40 * z, 1.0, True)
        for name, (x, z) in coords.items()
    })])
    lifted = LiftedPoseSequence(frames=[LiftedPoseFrame(0, 0.0, {
        name: LiftedPosePoint(name, (scale * x, 0.0, scale * z), 1.0, 0.0)
        for name, (x, z) in coords.items()
    })])
    return observed, lifted


def test_weak_perspective_audit_accepts_scale_equivalent_reconstruction():
    observed, raw = _sequence()
    _, reconstructed = _sequence(1.5)
    report = audit_weak_perspective_reprojection(observed, raw, reconstructed)
    assert report["passed"]
    assert report["median_error_ratio"] == 1.0


def test_weak_perspective_audit_rejects_worsened_joint_layout():
    observed, raw = _sequence()
    bad = LiftedPoseSequence(frames=[LiftedPoseFrame(0, 0.0, {
        "pelvis": LiftedPosePoint("pelvis", (0.0, 0.0, 0.0), 1.0, 0.0),
        "left_hip": LiftedPosePoint("left_hip", (-1.0, 0.0, -1.0), 1.0, 0.0),
        "right_hip": LiftedPosePoint("right_hip", (0.1, 0.0, -1.0), 1.0, 0.0),
    })])
    report = audit_weak_perspective_reprojection(observed, raw, bad)
    assert not report["passed"]
