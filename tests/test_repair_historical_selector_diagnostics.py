"""Focused tests for the docs/21 historical selector-diagnostic repair.

Diagnostic-only units: none of this retrains or changes historical
checkpoints/reports. Verifies the exact-evaluator-angle grid matches the
production per-frame evaluator exactly (not the historical (1-cos)
surrogate), and the frame/pair selection-overlap helpers behave correctly.
"""
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

pytest.importorskip("torch", reason="selector-diagnostic-repair checks require the optional training extra")

_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT = _ROOT / "scripts" / "repair_historical_selector_diagnostics.py"


def _load_module():
    sys.path.insert(0, str(_ROOT / "scripts"))
    sys.path.insert(0, str(_ROOT / "src"))
    try:
        spec = importlib.util.spec_from_file_location("repair_historical_selector_diagnostics", _SCRIPT)
        module = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.pop(0)
        sys.path.pop(0)


def _zeroed(n):
    from pose.pose_lifter import H36M_NAMES
    import torch
    return torch.zeros((n, len(H36M_NAMES), 3))


def test_exact_evaluator_yaw_degree_grid_matches_root_yaw_error_degrees_per_pair():
    """The vectorized (batch, pair) grid must reproduce the exact same
    per-pair angle the production evaluator computes -- not the (1-cos)
    proxy the historical diagnostic used."""
    import torch
    from pose.pose_lifter import H36M_NAMES
    from training.temporal_lifter import YAW_INDICES, _angle_delta

    module = _load_module()
    n = 5
    torch.manual_seed(3)
    target = torch.randn((n, len(H36M_NAMES), 3))
    prediction = target + torch.randn((n, len(H36M_NAMES), 3)) * 0.3
    valid = torch.ones((n, len(H36M_NAMES)), dtype=torch.bool)

    degrees, stable = module._exact_evaluator_yaw_degree_grid(torch, prediction, target, valid)

    for frame in range(n):
        for pair_index, (left, right) in enumerate(YAW_INDICES):
            predicted_axis = (prediction[frame, right, :2] - prediction[frame, left, :2]).numpy()
            target_axis = (target[frame, right, :2] - target[frame, left, :2]).numpy()
            if min(np.linalg.norm(predicted_axis), np.linalg.norm(target_axis)) <= 1e-6:
                assert not bool(stable[frame, pair_index])
                continue
            expected = abs(_angle_delta(
                np.arctan2(predicted_axis[1], predicted_axis[0]), np.arctan2(target_axis[1], target_axis[0]),
            )) * 180.0 / np.pi
            assert float(degrees[frame, pair_index]) == pytest.approx(expected, abs=1e-4)


def test_exact_evaluator_frame_combined_matches_root_yaw_error_degrees_per_frame():
    import torch
    from pose.pose_lifter import H36M_NAMES
    from training.temporal_lifter import _root_yaw_error_degrees

    module = _load_module()
    n = 4
    torch.manual_seed(7)
    target = torch.randn((n, len(H36M_NAMES), 3))
    prediction = target + torch.randn((n, len(H36M_NAMES), 3)) * 0.3
    valid = torch.ones((n, len(H36M_NAMES)), dtype=torch.bool)

    combined, frame_stable = module._exact_evaluator_frame_combined_grid(torch, prediction, target, valid)

    for frame in range(n):
        expected = _root_yaw_error_degrees(
            prediction[frame].numpy(), target[frame].numpy(), valid[frame].numpy(),
        )
        if expected is None:
            assert not bool(frame_stable[frame])
        else:
            assert float(combined[frame]) == pytest.approx(expected, abs=1e-3)


def test_exact_evaluator_grid_differs_from_1_minus_cos_proxy_in_general():
    """Confirms the bug this repair fixes actually existed: (1-cos)*180/pi
    is NOT the same quantity as the real per-pair angle in degrees, except
    at the trivial zero-error point."""
    import torch
    from pose.pose_lifter import H36M_NAMES
    from training.temporal_lifter import YAW_INDICES, _yaw_axis_error_grid

    module = _load_module()
    n = 1
    target = _zeroed(n)
    ls, rs = H36M_NAMES.index("left_shoulder"), H36M_NAMES.index("right_shoulder")
    target[0, ls, :2] = torch.tensor([-1.0, 0.0])
    target[0, rs, :2] = torch.tensor([1.0, 0.0])
    prediction = target.clone()
    import math
    theta = math.radians(60.0)
    prediction[0, ls, :2] = torch.tensor([-math.cos(theta), -math.sin(theta)])
    prediction[0, rs, :2] = torch.tensor([math.cos(theta), math.sin(theta)])
    valid = torch.ones((n, len(H36M_NAMES)), dtype=torch.bool)

    real_degrees, _stable = module._exact_evaluator_yaw_degree_grid(torch, prediction, target, valid)
    proxy_1_minus_cos, _stable2 = _yaw_axis_error_grid(torch, prediction, target, valid)
    proxy_as_fake_degrees = proxy_1_minus_cos * 180.0 / torch.pi

    # Real angle is exactly 60 degrees; the historical fake-degree surrogate
    # is a materially different number for the same rotation.
    assert float(real_degrees[0, 0]) == pytest.approx(60.0, abs=1e-2)
    assert abs(float(proxy_as_fake_degrees[0, 0]) - 60.0) > 5.0


def test_frame_selected_set_picks_the_worst_frames():
    import torch

    module = _load_module()
    errors = torch.tensor([1.0, 5.0, 2.0, 9.0, 3.0, 4.0, 6.0, 7.0, 8.0, 0.5,
                            1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 0.1])
    stable = torch.ones_like(errors, dtype=torch.bool)
    selected = module._frame_selected_set(torch, errors, stable)
    # top-5% of 20 -> 1 frame: the single largest error (9.5 at index 18)
    assert selected == {18}
