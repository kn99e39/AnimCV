from pose.calibrated_reprojection_audit import audit_calibrated_reprojection
from pose.camera_calibration import CameraCalibration, project_camera_point
from pose.pose_lifter import LiftedPoseFrame, LiftedPosePoint, LiftedPoseSequence
from pose.pose_types import PoseFrame, PoseLandmark, PoseSequence


def _fixture(reconstructed_right_x: float = 1.0):
    camera = CameraCalibration(1920, 1080, 1000, 1000, 960, 540)
    coordinates = {"pelvis": (0.0, 0.0, 0.0), "left_hip": (-1.0, 0.1, -1.0), "right_hip": (1.0, -0.1, -1.0), "neck": (0.0, 0.2, 1.0)}
    translation = (0.2, 5.0, 0.1)
    observed = PoseSequence(frames=[PoseFrame(0, 0.0, {
        name: PoseLandmark(name, *project_camera_point(tuple(a + b for a, b in zip(point, translation)), camera), 1.0, True)
        for name, point in coordinates.items()
    })])
    raw = LiftedPoseSequence(frames=[LiftedPoseFrame(0, 0.0, {
        name: LiftedPosePoint(name, point, 1.0, 0.0) for name, point in coordinates.items()
    })])
    changed = dict(coordinates)
    changed["right_hip"] = (reconstructed_right_x, -0.1, -1.0)
    fixed = LiftedPoseSequence(frames=[LiftedPoseFrame(0, 0.0, {
        name: LiftedPosePoint(name, point, 1.0, 0.0) for name, point in changed.items()
    })])
    return observed, raw, fixed, camera


def test_calibrated_audit_accepts_exact_reconstruction():
    observed, raw, fixed, camera = _fixture()
    assert audit_calibrated_reprojection(observed, raw, fixed, camera)["passed"]


def test_calibrated_audit_rejects_changed_3d_layout():
    observed, raw, fixed, camera = _fixture(0.1)
    assert not audit_calibrated_reprojection(observed, raw, fixed, camera)["passed"]
