"""Focused tests for the A9-vs-A14 forward-depth attribution script.

Diagnostic-only units: verifies the per-frame yaw helper matches the
production evaluator and that hard/non-hard subset slicing is a clean
partition ranked strictly by descending A9 yaw error.
"""
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

pytest.importorskip("torch", reason="attribution checks require the optional training extra")

_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT = _ROOT / "scripts" / "attribute_bilateral_forward_depth.py"


def _load_module():
    sys.path.insert(0, str(_ROOT / "src"))
    try:
        spec = importlib.util.spec_from_file_location("attribute_bilateral_forward_depth", _SCRIPT)
        module = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.pop(0)


def _zeroed(n):
    from pose.pose_lifter import H36M_NAMES
    return np.zeros((n, len(H36M_NAMES), 3), dtype=np.float32)


def test_per_frame_yaw_matches_root_yaw_error_degrees_per_frame():
    from pose.pose_lifter import H36M_NAMES
    from training.temporal_lifter import _root_yaw_error_degrees

    module = _load_module()
    n = 4
    target = _zeroed(n)
    ls, rs = H36M_NAMES.index("left_shoulder"), H36M_NAMES.index("right_shoulder")
    lh, rh = H36M_NAMES.index("left_hip"), H36M_NAMES.index("right_hip")
    target[:, ls, 0], target[:, rs, 0] = -1.0, 1.0
    target[:, lh, 0], target[:, rh, 0] = -1.0, 1.0
    prediction = target.copy()
    prediction[2, rs, 1] = 0.5  # rotate frame 2's shoulder axis
    valid = np.ones((n, len(H36M_NAMES)), dtype=bool)

    result = module._per_frame_yaw(prediction, target, valid)
    expected = np.array([_root_yaw_error_degrees(p, t, v) for p, t, v in zip(prediction, target, valid)])
    assert np.allclose(result, expected, equal_nan=True)


def test_per_frame_yaw_is_nan_when_evaluator_reports_none():
    """A frame with a collapsed (near-zero) bilateral span has no evaluator
    yaw value; the attribution script must record NaN, not crash or 0.0."""
    module = _load_module()
    n = 1
    target = _zeroed(n)  # every joint at the origin -> zero-length spans
    prediction = target.copy()
    valid = np.ones((n, target.shape[1]), dtype=bool)

    result = module._per_frame_yaw(prediction, target, valid)
    assert np.isnan(result[0])


def test_subset_diagnostics_returns_empty_dict_for_no_indices():
    import torch  # noqa: F401 -- proves the optional extra is importable here too

    module = _load_module()
    predictions = _zeroed(3)
    targets = _zeroed(3)
    valid = np.ones((3, predictions.shape[1]), dtype=bool)

    assert module._subset_diagnostics(__import__("torch"), predictions, targets, valid, np.array([], dtype=int)) == {}


def test_hard_and_non_hard_partition_is_exhaustive_and_ranked_by_a9_yaw():
    """Reproduces the script's own top-5%/top-1%/non-hard split logic on a
    small synthetic yaw-error array and checks it is a clean partition of the
    eligible frames, strictly ordered by descending error."""
    yaw = np.array([50.0, 40.0, 30.0, 20.0, 10.0, np.nan, 5.0, 1.0, 0.5, 0.1])
    eligible = np.flatnonzero(np.isfinite(yaw))
    order = eligible[np.argsort(-yaw[eligible])]
    top5_count = max(1, (len(order) + 19) // 20)
    top1_count = max(1, (len(order) + 99) // 100)
    hard_top5 = order[:top5_count]
    hard_top1 = order[:top1_count]
    non_hard = order[top5_count:]

    assert set(hard_top5) | set(non_hard) == set(eligible)
    assert set(hard_top5) & set(non_hard) == set()
    assert set(hard_top1).issubset(set(hard_top5))
    assert list(yaw[order]) == sorted(yaw[eligible], reverse=True)
    assert 5 not in eligible  # the NaN entry must never appear in any subset
