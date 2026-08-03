import numpy as np
import pytest

from common.serialization import write_json
from pose.multiview_triangulation import load_calibration, triangulate
from pose.pose_types import PoseFrame, PoseLandmark, PoseSequence


def _project(point, camera):
    point_camera = np.asarray(camera) @ np.append(point, 1.0)
    return 800 * point_camera[0] / point_camera[2] + 320, 800 * point_camera[1] / point_camera[2] + 240


def test_triangulation_emits_root_relative_animcv_camera_axes(tmp_path):
    # cam1 is one metre to the right of cam0. Matrices are OpenCV world->camera.
    cam0 = np.eye(4)
    cam1 = np.eye(4)
    cam1[0, 3] = -1.0
    calibration_path = tmp_path / "calibration.json"
    write_json(calibration_path, {
        "schema": "animcv_multiview_calibration_v1", "world_units": "metres", "cameras": {
            "cam0": {"intrinsics": {"fx": 800, "fy": 800, "cx": 320, "cy": 240}, "world_to_camera": cam0.tolist()},
            "cam1": {"intrinsics": {"fx": 800, "fy": 800, "cx": 320, "cy": 240}, "world_to_camera": cam1.tolist()},
        },
    })
    world = {"pelvis": np.array((0.0, 0.0, 5.0)), "neck": np.array((0.0, 1.0, 5.0))}
    observations = {}
    for name, camera in (("cam0", cam0), ("cam1", cam1)):
        landmarks = {joint: PoseLandmark(joint, *_project(point, camera), 1.0, True) for joint, point in world.items()}
        observations[name] = PoseSequence([PoseFrame(4, 0.16, landmarks)], 25)

    output, report = triangulate(observations, load_calibration(calibration_path), "cam0")
    assert report["passed"]
    assert report["coverage"] == 1.0
    assert output.frames[0].points["pelvis"].position == (0.0, 0.0, 0.0)
    # OpenCV neck +Y (down) becomes AnimCV -Z (up) after pelvis subtraction.
    assert output.frames[0].points["neck"].position == pytest.approx((0.0, 0.0, -1.0), abs=1e-6)


def test_triangulation_marks_one_view_joint_invalid(tmp_path):
    # Validation is covered by the first test's full calibration; this isolates visibility rejection.
    calibration = {"cam0": {"intrinsics": {"fx": 800, "fy": 800, "cx": 320, "cy": 240}, "world_to_camera": np.eye(4).tolist()},
                   "cam1": {"intrinsics": {"fx": 800, "fy": 800, "cx": 320, "cy": 240}, "world_to_camera": [[1, 0, 0, -1], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]}}
    path = tmp_path / "calibration.json"
    write_json(path, {"schema": "animcv_multiview_calibration_v1", "world_units": "metres", "cameras": calibration})
    frame0 = PoseFrame(0, 0.0, {"pelvis": PoseLandmark("pelvis", 320, 240, 1.0, True)})
    frame1 = PoseFrame(0, 0.0, {"pelvis": PoseLandmark("pelvis", 160, 240, 1.0, False)})
    output, report = triangulate({"cam0": PoseSequence([frame0], 25), "cam1": PoseSequence([frame1], 25)}, load_calibration(path), "cam0")
    assert report["insufficient_view_joint_count"] == 1
    assert not output.frames[0].points["pelvis"].observation_valid
