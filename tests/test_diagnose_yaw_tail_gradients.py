"""Focused tests for the A11 gradient-diagnosis instrumentation.

Diagnostic-only units: none of this changes production loss behavior. Tests
verify the frame-level counterfactual selector, the real pooled selector's
detail/attribution, and per-source isolation match their intended semantics
before they're trusted for the A11 diagnosis report.
"""
import importlib.util
import math
from pathlib import Path
import sys

import pytest

pytest.importorskip("torch", reason="gradient diagnostics require the optional training extra")

_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT = _ROOT / "scripts" / "diagnose_yaw_tail_gradients.py"


def _load_module():
    sys.path.insert(0, str(_ROOT / "src"))
    try:
        spec = importlib.util.spec_from_file_location("diagnose_yaw_tail_gradients", _SCRIPT)
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


def _rotated(angle_deg):
    theta = math.radians(angle_deg)
    return [-math.cos(theta), -math.sin(theta)], [math.cos(theta), math.sin(theta)]


def test_frame_combined_error_averages_available_pairs_per_frame():
    import torch
    from pose.pose_lifter import H36M_NAMES

    module = _load_module()
    n = 2
    target, prediction = _zeroed(n), _zeroed(n)
    ls, rs = H36M_NAMES.index("left_shoulder"), H36M_NAMES.index("right_shoulder")
    lh, rh = H36M_NAMES.index("left_hip"), H36M_NAMES.index("right_hip")
    target[:, ls, :2] = torch.tensor([-1.0, 0.0])
    target[:, rs, :2] = torch.tensor([1.0, 0.0])
    target[:, lh, :2] = torch.tensor([-1.0, 0.0])
    target[:, rh, :2] = torch.tensor([1.0, 0.0])

    # Frame 0: both pairs at 40 degrees -> combined 40.
    sl, sr = _rotated(40.0)
    hl, hr = _rotated(40.0)
    prediction[0, ls, :2], prediction[0, rs, :2] = torch.tensor(sl), torch.tensor(sr)
    prediction[0, lh, :2], prediction[0, rh, :2] = torch.tensor(hl), torch.tensor(hr)
    # Frame 1: only shoulder valid (hip landmarks coincide -> degenerate axis), at 60 degrees.
    sl, sr = _rotated(60.0)
    prediction[1, ls, :2], prediction[1, rs, :2] = torch.tensor(sl), torch.tensor(sr)
    prediction[1, lh, :2] = prediction[1, rh, :2] = torch.tensor([0.0, 0.0])
    target[1, lh, :2] = target[1, rh, :2] = torch.tensor([0.0, 0.0])
    valid = torch.ones((n, len(H36M_NAMES)), dtype=torch.bool)

    combined, stable = module._frame_combined_error_grid(torch, prediction, target, valid)

    assert bool(stable[0]) and bool(stable[1])
    assert float(combined[0]) == pytest.approx(1.0 - math.cos(math.radians(40.0)), abs=1e-5)
    assert float(combined[1]) == pytest.approx(1.0 - math.cos(math.radians(60.0)), abs=1e-5)


def test_frame_level_selector_ranks_frames_not_pooled_pairs():
    """The exact counterexample from test_yaw_tail_loss_contract.py's pooled-
    selector test, but the frame-level selector must pick frame A (the
    officially worse *frame*), not frame B's single extreme pair."""
    import torch
    from pose.pose_lifter import H36M_NAMES
    from training.temporal_lifter import _root_yaw_error_degrees

    module = _load_module()
    n = 2
    target, prediction = _zeroed(n), _zeroed(n)
    ls, rs = H36M_NAMES.index("left_shoulder"), H36M_NAMES.index("right_shoulder")
    lh, rh = H36M_NAMES.index("left_hip"), H36M_NAMES.index("right_hip")
    target[:, ls, :2] = torch.tensor([-1.0, 0.0])
    target[:, rs, :2] = torch.tensor([1.0, 0.0])
    target[:, lh, :2] = torch.tensor([-1.0, 0.0])
    target[:, rh, :2] = torch.tensor([1.0, 0.0])

    a_sl, a_sr = _rotated(50.0)
    a_hl, a_hr = _rotated(50.0)
    b_sl, b_sr = _rotated(60.0)
    b_hl, b_hr = _rotated(5.0)
    prediction[0, ls, :2], prediction[0, rs, :2] = torch.tensor(a_sl), torch.tensor(a_sr)
    prediction[0, lh, :2], prediction[0, rh, :2] = torch.tensor(a_hl), torch.tensor(a_hr)
    prediction[1, ls, :2], prediction[1, rs, :2] = torch.tensor(b_sl), torch.tensor(b_sr)
    prediction[1, lh, :2], prediction[1, rh, :2] = torch.tensor(b_hl), torch.tensor(b_hr)
    valid = torch.ones((n, len(H36M_NAMES)), dtype=torch.bool)

    official_a = _root_yaw_error_degrees(prediction[0].numpy(), target[0].numpy(), valid[0].numpy())
    official_b = _root_yaw_error_degrees(prediction[1].numpy(), target[1].numpy(), valid[1].numpy())
    assert official_a > official_b  # frame A is the officially worse frame

    # tail_count clamps to 1 with only 2 frame candidates -> selects frame A.
    loss = module._yaw_tail_loss_frame_level(torch, prediction, target, valid)
    expected_a_combined = 1.0 - math.cos(math.radians(50.0))
    assert float(loss) == pytest.approx(expected_a_combined, abs=1e-5)


def test_pooled_selection_detail_attributes_selected_entries_by_source():
    import torch
    from pose.pose_lifter import H36M_NAMES

    module = _load_module()
    n = 4
    target, prediction = _zeroed(n), _zeroed(n)
    ls, rs = H36M_NAMES.index("left_shoulder"), H36M_NAMES.index("right_shoulder")
    lh, rh = H36M_NAMES.index("left_hip"), H36M_NAMES.index("right_hip")
    target[:, ls, :2] = torch.tensor([-1.0, 0.0])
    target[:, rs, :2] = torch.tensor([1.0, 0.0])
    target[:, lh, :2] = torch.tensor([-1.0, 0.0])
    target[:, rh, :2] = torch.tensor([1.0, 0.0])
    for i in range(n):
        prediction[i, lh, :2], prediction[i, rh, :2] = torch.tensor([-1.0, 0.0]), torch.tensor([1.0, 0.0])  # hip perfect
    # Only sample 2's shoulder pair is off -> the sole tail entry with a small (4-item * 2) pool.
    sl, sr = _rotated(70.0)
    prediction[2, ls, :2], prediction[2, rs, :2] = torch.tensor(sl), torch.tensor(sr)
    for i in (0, 1, 3):
        prediction[i, ls, :2], prediction[i, rs, :2] = torch.tensor([-1.0, 0.0]), torch.tensor([1.0, 0.0])
    valid = torch.ones((n, len(H36M_NAMES)), dtype=torch.bool)
    source_ids = torch.tensor([0, 0, 1, 2])  # sample 2 (the tail entry) belongs to source 1 (3DPW)

    detail = module._pooled_selection_detail(torch, prediction, target, valid, source_ids)

    assert detail["candidate_count"] == n * 2
    assert detail["selected_count"] >= 1
    assert detail["shoulder_only_selections"] >= 1
    assert detail["loss_share_by_source"]["3DPW"] == pytest.approx(1.0, abs=1e-4)
    assert detail["loss_share_by_source"]["MPI-INF-3DHP"] == pytest.approx(0.0, abs=1e-4)


def test_source_restricted_loss_ignores_other_sources_entirely():
    import torch
    from pose.pose_lifter import H36M_NAMES

    module = _load_module()
    n = 2
    target, prediction = _zeroed(n), _zeroed(n)
    ls, rs = H36M_NAMES.index("left_shoulder"), H36M_NAMES.index("right_shoulder")
    lh, rh = H36M_NAMES.index("left_hip"), H36M_NAMES.index("right_hip")
    target[:, ls, :2] = torch.tensor([-1.0, 0.0])
    target[:, rs, :2] = torch.tensor([1.0, 0.0])
    target[:, lh, :2] = torch.tensor([-1.0, 0.0])
    target[:, rh, :2] = torch.tensor([1.0, 0.0])
    for i in range(n):
        prediction[i, lh, :2], prediction[i, rh, :2] = torch.tensor([-1.0, 0.0]), torch.tensor([1.0, 0.0])
    sl, sr = _rotated(80.0)
    prediction[0, ls, :2], prediction[0, rs, :2] = torch.tensor(sl), torch.tensor(sr)  # only source-0 sample is wrong
    prediction[1, ls, :2], prediction[1, rs, :2] = torch.tensor([-1.0, 0.0]), torch.tensor([1.0, 0.0])
    valid = torch.ones((n, len(H36M_NAMES)), dtype=torch.bool)
    source_ids = torch.tensor([0, 1])

    restricted_to_source_1 = module._source_restricted_yaw_tail_loss(torch, prediction, target, valid, source_ids, 1)

    # Source 1's only sample is a perfect match, so restricting the pool to
    # it must yield exactly zero loss, even though source 0 has a large error.
    assert float(restricted_to_source_1) == pytest.approx(0.0, abs=1e-6)


def test_component_losses_matches_production_yaw_tail_and_is_finite():
    import torch
    from training.temporal_lifter import _yaw_tail_loss

    module = _load_module()
    n = 6
    torch.manual_seed(3)
    target = _zeroed(n)
    prediction = target + torch.randn_like(target) * 0.05
    valid = torch.ones((n, target.shape[1]), dtype=torch.bool)

    components = module._component_losses(torch, prediction, target, valid)

    assert components["yaw_tail_pooled"] == pytest.approx(float(_yaw_tail_loss(torch, prediction, target, valid)))
    assert all(math.isfinite(value) for value in components.values())
