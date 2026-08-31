"""Focused tests for the A14 gradient-diagnosis instrumentation.

Diagnostic-only units: none of this changes production loss behavior. Tests
verify the raw candidate loss matches the production folding math, source
isolation only sees its own source's pairs, the endpoint-gradient helper
reports gradient on the right coordinates only, and the fixed-batch replay
is deterministic -- before any of it is trusted for the A14 diagnosis report.
"""
import importlib.util
from pathlib import Path
import sys

import pytest

pytest.importorskip("torch", reason="gradient diagnostics require the optional training extra")

_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT = _ROOT / "scripts" / "diagnose_bilateral_forward_depth_gradients.py"


def _load_module():
    sys.path.insert(0, str(_ROOT / "src"))
    try:
        spec = importlib.util.spec_from_file_location("diagnose_bilateral_forward_depth_gradients", _SCRIPT)
        module = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.pop(0)


def _zeroed(n):
    import torch
    from pose.pose_lifter import H36M_NAMES
    return torch.zeros((n, len(H36M_NAMES), 3))


def test_candidate_loss_matches_bilateral_forward_depth_residual_sum():
    import torch
    from pose.pose_lifter import H36M_NAMES
    from training.temporal_lifter import FORWARD_DEPTH_AXIS, _bilateral_forward_depth_residual_sum

    module = _load_module()
    n = 2
    target = _zeroed(n)
    prediction = target.clone()
    prediction[0, H36M_NAMES.index("right_shoulder"), FORWARD_DEPTH_AXIS] = 0.3
    prediction[1, H36M_NAMES.index("right_hip"), FORWARD_DEPTH_AXIS] = -0.2
    valid = torch.ones((n, len(H36M_NAMES)), dtype=torch.bool)

    expected_sum, expected_count = _bilateral_forward_depth_residual_sum(torch, prediction, target, valid)
    actual = module._candidate_loss(torch, prediction, target, valid)
    assert float(actual) == pytest.approx(float(expected_sum / expected_count.clamp_min(1.0)))


def test_source_restricted_candidate_loss_only_sees_its_own_source():
    import torch
    from pose.pose_lifter import H36M_NAMES
    from training.temporal_lifter import FORWARD_DEPTH_AXIS

    module = _load_module()
    n = 3
    target = _zeroed(n)
    prediction = target.clone()
    # Only frame 1 (source 1) has any error.
    prediction[1, H36M_NAMES.index("right_shoulder"), FORWARD_DEPTH_AXIS] = 0.5
    valid = torch.ones((n, len(H36M_NAMES)), dtype=torch.bool)
    source_ids = torch.tensor([0, 1, 2])

    loss_source_0 = module._source_restricted_candidate_loss(torch, prediction, target, valid, source_ids, 0)
    loss_source_1 = module._source_restricted_candidate_loss(torch, prediction, target, valid, source_ids, 1)
    loss_source_2 = module._source_restricted_candidate_loss(torch, prediction, target, valid, source_ids, 2)

    assert float(loss_source_0) == pytest.approx(0.0, abs=1e-6)
    assert float(loss_source_1) > 0.0
    assert float(loss_source_2) == pytest.approx(0.0, abs=1e-6)


def test_endpoint_gradient_isolates_forward_depth_axis_of_shoulder_and_hip():
    import torch
    from pose.pose_lifter import H36M_NAMES
    from training.temporal_lifter import FORWARD_DEPTH_AXIS

    module = _load_module()
    n = 1
    target = _zeroed(n)
    prediction = target.clone()
    prediction[0, H36M_NAMES.index("right_shoulder"), FORWARD_DEPTH_AXIS] = 0.4
    prediction.requires_grad_(True)

    loss = module._candidate_loss(torch, prediction, target,
                                   torch.ones((n, len(H36M_NAMES)), dtype=torch.bool))
    result = module._endpoint_gradient(torch, prediction, loss)

    assert result["shoulder_forward_depth_grad_l1"] > 0.0
    assert result["hip_forward_depth_grad_l1"] == pytest.approx(0.0, abs=1e-8)
    assert result["endpoint_non_forward_depth_grad_l1"] == pytest.approx(0.0, abs=1e-8)
    assert result["endpoint_l1_share"] == pytest.approx(1.0, abs=1e-6)  # nothing outside the endpoints moves


def test_source_wise_diagnostics_reports_per_source_valid_pair_counts():
    import torch
    from pose.pose_lifter import H36M_NAMES

    module = _load_module()
    n = 4
    target = _zeroed(n)
    prediction = target.clone()
    valid = torch.ones((n, len(H36M_NAMES)), dtype=torch.bool)
    valid[0, H36M_NAMES.index("left_shoulder")] = False  # frame 0's shoulder pair now invalid
    source_ids = torch.tensor([0, 0, 1, 1])

    result = module._source_wise_diagnostics(torch, prediction, target, valid, source_ids)
    assert result["MPI-INF-3DHP"]["valid_pair_count"] == 3  # frame0 hip + frame1 shoulder+hip
    assert result["3DPW"]["valid_pair_count"] == 4  # frame2+3 shoulder+hip
    assert result["AMASS"]["valid_pair_count"] == 0


def test_first_epoch_batches_replay_is_deterministic():
    import torch
    from pose.pose_lifter import H36M_NAMES
    from training.temporal_lifter import TrainingConfig, _window_offsets

    module = _load_module()
    n, window, batch_size = 40, 81, 8
    torch.manual_seed(5)
    x = torch.randn((n, len(H36M_NAMES), 3))
    offsets = torch.as_tensor(_window_offsets(n, window), dtype=torch.long)
    source_tensor = torch.zeros(n, dtype=torch.long)
    config = TrainingConfig(window=window, channels=8, epochs=1, batch_size=batch_size, seed=123,
                             source_balanced_sampling=True)

    first_inputs, first_batches = module._first_epoch_batches(
        torch, x, x, torch.ones(n, dtype=torch.bool), offsets, source_tensor, [(0, n)], config, 3,
    )
    second_inputs, second_batches = module._first_epoch_batches(
        torch, x, x, torch.ones(n, dtype=torch.bool), offsets, source_tensor, [(0, n)], config, 3,
    )
    assert torch.equal(first_inputs, second_inputs)
    assert all(torch.equal(a, b) for a, b in zip(first_batches, second_batches))
