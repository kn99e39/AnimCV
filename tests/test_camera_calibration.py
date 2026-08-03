import pytest

from pose.camera_calibration import CameraCalibration, project_camera_point


def test_project_camera_point_uses_animcv_camera_axes():
    camera = CameraCalibration(1920, 1080, 1000, 1000, 960, 540)
    assert project_camera_point((1.0, 2.0, 1.0), camera) == pytest.approx((1460.0, 40.0))


def test_camera_calibration_roundtrip_and_validation():
    camera = CameraCalibration(640, 480, 500, 500, 320, 240, source="checkerboard", calibration_rms_pixels=0.2)
    assert CameraCalibration.from_dict(camera.to_dict()) == camera
    with pytest.raises(ValueError, match="focal"):
        CameraCalibration(640, 480, 0, 500, 320, 240)
