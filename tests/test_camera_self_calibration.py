import pytest

from pose.camera_calibration import CameraCalibration, project_camera_point
from pose.camera_self_calibration import estimate_static_camera_calibration
from pose.pose_lifter import LiftedPoseFrame, LiftedPosePoint, LiftedPoseSequence
from pose.pose_types import PoseFrame, PoseLandmark, PoseSequence


def test_static_camera_self_calibration_recovers_synthetic_focal_length():
    expected = CameraCalibration(1280, 720, 900, 900, 640, 360)
    base = {"pelvis": (0.0, 0.0, 0.0), "left_hip": (-1.0, 0.0, -1.0),
            "right_hip": (1.0, 0.1, -1.0), "neck": (0.0, 0.2, 1.0)}
    observed_frames, lifted_frames = [], []
    for index in range(12):
        translation = (0.04 * index, 5.0 + 0.03 * index, 0.0)
        observed_frames.append(PoseFrame(index, index / 30, {
            name: PoseLandmark(name, *project_camera_point(tuple(a + b for a, b in zip(point, translation)), expected), 1.0, True)
            for name, point in base.items()
        }))
        lifted_frames.append(LiftedPoseFrame(index, index / 30, {
            name: LiftedPosePoint(name, point, 1.0, 0.0) for name, point in base.items()
        }))
    calibration, report = estimate_static_camera_calibration(
        PoseSequence(observed_frames), LiftedPoseSequence(lifted_frames), 1280, 720,
        max_focal_uncertainty_ratio=3.0,
    )
    assert calibration.fx == pytest.approx(900, rel=0.02)
    assert report["accepted_for_limited_calibrated_audit"]
