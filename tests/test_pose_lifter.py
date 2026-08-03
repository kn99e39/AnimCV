import numpy as np
import pytest
from dataclasses import replace

from pose.pose_lifter import (
    H36M_NAMES,
    VideoPose3DConfig,
    VideoPose3DLifter,
    _to_lifted_points,
    _prepare_h36m_observations,
)
from pose.pose_types import PoseFrame, PoseLandmark, PoseSequence


def _frame(index: int = 0) -> PoseFrame:
    names = {
        "pelvis", "spine", "neck", "head", "left_shoulder", "left_elbow", "left_wrist",
        "right_shoulder", "right_elbow", "right_wrist", "left_hip", "left_knee",
        "left_ankle", "right_hip", "right_knee", "right_ankle",
    }
    return PoseFrame(
        frame_index=index,
        timestamp=index / 50,
        landmarks={name: PoseLandmark(name, 10.0, 20.0, 0.8, True) for name in names},
    )


def test_lifted_points_apply_documented_h36m_camera_transform():
    predicted = np.arange(51, dtype=np.float32).reshape(17, 3)

    points = _to_lifted_points(predicted, _frame())

    assert tuple(points["pelvis"].position) == pytest.approx((0.0, 2.0, -1.0))
    assert tuple(points["left_hip"].position) == pytest.approx((-3.0, 5.0, -4.0))
    assert points["thorax"].confidence == pytest.approx(0.8)
    assert points["thorax"].depth_uncertainty == pytest.approx(0.2)


def test_lift_uses_temporal_prediction_for_every_source_frame(monkeypatch):
    poses = PoseSequence(frames=[_frame(0), _frame(1)], source_fps=50.0)
    lifter = VideoPose3DLifter(VideoPose3DConfig(checkpoint_path="model.pth"))
    calls = []

    def fake_prediction(results, frame_index, image_size):
        calls.append((len(results), frame_index, image_size))
        return np.zeros((17, 3), dtype=np.float32)

    monkeypatch.setattr(lifter, "_predict_window", fake_prediction)
    result = lifter.lift(poses, (100, 200))

    assert [frame.frame_index for frame in result.frames] == [0, 1]
    assert calls == [(2, 0, (100, 200)), (2, 1, (100, 200))]
    assert result.coordinate_frame == "camera_root_relative"
    assert result.units == "metres"


def test_lifter_rejects_unsupported_window_length():
    with pytest.raises(ValueError, match="sequence_length=81"):
        VideoPose3DLifter(VideoPose3DConfig(checkpoint_path="model.pth", sequence_length=27))


def test_prepare_observations_marks_short_low_confidence_gap_as_interpolated():
    first, missing, last = _frame(0), _frame(1), _frame(2)
    missing = replace(
        missing,
        landmarks={
            **missing.landmarks,
            "left_wrist": replace(missing.landmarks["left_wrist"], confidence=0.1),
        },
    )
    poses = PoseSequence(frames=[first, missing, last], source_fps=50)

    arrays, flags = _prepare_h36m_observations(poses, min_confidence=0.3, max_gap=1)

    wrist_index = H36M_NAMES.index("left_wrist")
    assert arrays[1][wrist_index] == pytest.approx((10.0, 20.0))
    assert flags[1]["left_wrist"] == (False, True)
