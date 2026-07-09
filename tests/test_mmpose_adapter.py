from types import SimpleNamespace

import numpy as np

from pose.mmpose_adapter import _COCO_KEYPOINT_NAMES, _extract_canonical_landmarks


def _fake_result(keypoints: dict[str, tuple[float, float]], default_score: float = 0.9):
    coords = np.zeros((17, 2), dtype=np.float32)
    scores = np.full((17,), default_score, dtype=np.float32)
    for name, (x, y) in keypoints.items():
        idx = _COCO_KEYPOINT_NAMES.index(name)
        coords[idx] = (x, y)

    pred_instances = SimpleNamespace(
        keypoints=np.expand_dims(coords, axis=0),
        keypoint_scores=np.expand_dims(scores, axis=0),
    )
    return [SimpleNamespace(pred_instances=pred_instances)]


def test_direct_coco_landmarks_pass_through():
    results = _fake_result({"left_wrist": (10.0, 20.0)})

    landmarks = _extract_canonical_landmarks(results, visibility_threshold=0.3)

    assert landmarks["left_wrist"].x == 10.0
    assert landmarks["left_wrist"].y == 20.0
    assert landmarks["left_wrist"].visible is True


def test_derived_landmarks_are_midpoints():
    results = _fake_result(
        {
            "left_shoulder": (0.0, 0.0),
            "right_shoulder": (10.0, 0.0),
            "left_hip": (0.0, 10.0),
            "right_hip": (10.0, 10.0),
        }
    )

    landmarks = _extract_canonical_landmarks(results, visibility_threshold=0.3)

    assert landmarks["neck"].x == 5.0 and landmarks["neck"].y == 0.0
    assert landmarks["pelvis"].x == 5.0 and landmarks["pelvis"].y == 10.0
    assert landmarks["spine"].x == 5.0 and landmarks["spine"].y == 5.0


def test_low_confidence_marks_landmark_not_visible():
    results = _fake_result({"left_wrist": (10.0, 20.0)}, default_score=0.1)

    landmarks = _extract_canonical_landmarks(results, visibility_threshold=0.3)

    assert landmarks["left_wrist"].visible is False


def test_no_results_returns_empty_landmarks():
    assert _extract_canonical_landmarks([], visibility_threshold=0.3) == {}
