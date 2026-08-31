"""Synthetic contract checks for the A14 bilateral forward-depth candidate.

docs/10 A14 tests whether explicitly, all-frame supervising the bilateral
forward-depth relational mode of the torso -- ``q = (y_right - y_left) /
sqrt(2)`` on AnimCV's canonical ``+Y`` (forward/depth) axis -- recovers any of
the official-test yaw failure documented in the earlier 3DPW generalization
diagnosis. This file covers Section 7's synthetic contracts (no training, no
GPU) before any fixed-batch gradient diagnosis or controlled training run.
"""
import math

import pytest

pytest.importorskip("torch", reason="A14 bilateral forward-depth checks require the optional training extra")

from pose.pose_lifter import H36M_NAMES
from training.temporal_lifter import (
    BILATERAL_DEPTH_NORMALIZATION, FORWARD_DEPTH_AXIS, TORSO_INDICES, TrainingConfig,
    _bilateral_forward_depth_diagnostics, _bilateral_forward_depth_grid, _bilateral_forward_depth_residual_sum,
    _supervision_loss,
)


def _zeroed(n):
    import torch
    return torch.zeros((n, len(H36M_NAMES), 3))


def _set_forward_y(tensor, index, left_shoulder_y, right_shoulder_y, left_hip_y, right_hip_y):
    tensor[index, H36M_NAMES.index("left_shoulder"), FORWARD_DEPTH_AXIS] = left_shoulder_y
    tensor[index, H36M_NAMES.index("right_shoulder"), FORWARD_DEPTH_AXIS] = right_shoulder_y
    tensor[index, H36M_NAMES.index("left_hip"), FORWARD_DEPTH_AXIS] = left_hip_y
    tensor[index, H36M_NAMES.index("right_hip"), FORWARD_DEPTH_AXIS] = right_hip_y


def test_normalization_constant_is_the_orthonormal_basis_factor():
    assert BILATERAL_DEPTH_NORMALIZATION == pytest.approx(1.0 / math.sqrt(2.0))


def test_forward_depth_axis_is_canonical_y_not_z():
    # AnimCV canonical camera frame: +X right, +Y forward/depth, +Z up
    # (pose_lifter._to_lifted_points). Index 1 is the forward/depth column.
    assert FORWARD_DEPTH_AXIS == 1


def test_identical_prediction_and_target_gives_zero_residual():
    import torch

    n = 3
    target = _zeroed(n)
    _set_forward_y(target, slice(0, n), 0.1, -0.2, 0.05, -0.05)
    prediction = target.clone()
    valid = torch.ones((n, len(H36M_NAMES)), dtype=torch.bool)

    total, count = _bilateral_forward_depth_residual_sum(torch, prediction, target, valid)
    assert float(total) == pytest.approx(0.0, abs=1e-6)
    assert float(count) == pytest.approx(2 * n)


def test_uniform_forward_translation_of_both_endpoints_is_invariant():
    """Shifting every joint's forward-depth by the same amount leaves the
    bilateral difference -- and therefore the residual -- unchanged."""
    import torch

    n = 1
    target = _zeroed(n)
    _set_forward_y(target, 0, 0.3, -0.1, 0.2, -0.2)
    prediction = target.clone()
    prediction[..., FORWARD_DEPTH_AXIS] += 7.5  # every joint's forward-depth shifted equally

    valid = torch.ones((n, len(H36M_NAMES)), dtype=torch.bool)
    total, _ = _bilateral_forward_depth_residual_sum(torch, prediction, target, valid)
    assert float(total) == pytest.approx(0.0, abs=1e-5)


def test_pure_common_depth_error_produces_zero_new_residual():
    """Both endpoints of a pair shifted by the *same* +Y error (a common-mode
    depth error, not an anti-symmetric one) must not register as a bilateral
    forward-depth error: q is unaffected by a shared shift."""
    import torch

    n = 1
    target = _zeroed(n)
    _set_forward_y(target, 0, 0.3, -0.1, 0.2, -0.2)
    prediction = target.clone()
    common_error = 0.4
    prediction[0, H36M_NAMES.index("left_shoulder"), FORWARD_DEPTH_AXIS] += common_error
    prediction[0, H36M_NAMES.index("right_shoulder"), FORWARD_DEPTH_AXIS] += common_error

    valid = torch.ones((n, len(H36M_NAMES)), dtype=torch.bool)
    total, _ = _bilateral_forward_depth_residual_sum(torch, prediction, target, valid)
    assert float(total) == pytest.approx(0.0, abs=1e-5)


def test_anti_symmetric_depth_error_produces_positive_residual_and_opposite_gradients():
    """right +a, left -a: the intended target mode. Contract 8/9's smooth-L1
    quadratic-region gradient scale is exercised here directly (Section 4's
    synthetic contract): in the quadratic region a pure anti-symmetric error
    of size a on each endpoint should push each endpoint's gradient by
    exactly +-a (the sqrt(2) factors cancel, see docs/10 A14 derivation)."""
    import torch

    n = 1
    target = _zeroed(n)
    _set_forward_y(target, 0, 0.0, 0.0, 0.0, 0.0)
    prediction = target.clone()
    a = 0.2  # stays inside the smooth-L1 quadratic region (|q_pred - q_target| < 1)
    left_index, right_index = H36M_NAMES.index("left_shoulder"), H36M_NAMES.index("right_shoulder")
    prediction[0, right_index, FORWARD_DEPTH_AXIS] = a
    prediction[0, left_index, FORWARD_DEPTH_AXIS] = -a
    prediction.requires_grad_(True)

    valid = torch.ones((n, len(H36M_NAMES)), dtype=torch.bool)
    total, _count = _bilateral_forward_depth_residual_sum(torch, prediction, target, valid)
    assert float(total.detach()) > 0.0
    total.backward()  # raw sum, not the pooled mean -- isolates the per-pair gradient contract

    right_grad = float(prediction.grad[0, right_index, FORWARD_DEPTH_AXIS])
    left_grad = float(prediction.grad[0, left_index, FORWARD_DEPTH_AXIS])
    assert right_grad == pytest.approx(a, abs=1e-5)
    assert left_grad == pytest.approx(-a, abs=1e-5)


def test_correct_magnitude_wrong_sign_produces_large_residual():
    import torch

    n = 1
    target = _zeroed(n)
    _set_forward_y(target, 0, -1.5, 1.5, 0.0, 0.0)  # q_shoulder_target = 3.0/sqrt(2)
    prediction = target.clone()
    prediction[0, H36M_NAMES.index("left_shoulder"), FORWARD_DEPTH_AXIS] = 1.5
    prediction[0, H36M_NAMES.index("right_shoulder"), FORWARD_DEPTH_AXIS] = -1.5  # sign flipped

    valid = torch.ones((n, len(H36M_NAMES)), dtype=torch.bool)
    q_pred, q_target, _ = _bilateral_forward_depth_grid(torch, prediction, target, valid)
    assert float(q_pred[0, 0]) == pytest.approx(-float(q_target[0, 0]), abs=1e-6)

    total, _ = _bilateral_forward_depth_residual_sum(torch, prediction, target, valid)
    # |q_error| = 6/sqrt(2) ~= 4.24, well past the beta=1.0 quadratic/linear
    # transition -> the linear-region residual (|x| - 0.5) is large.
    assert float(total) > 3.0


def test_zero_target_bilateral_depth_gives_finite_well_behaved_loss():
    import torch

    n = 1
    target = _zeroed(n)  # shoulder/hip forward-y exactly zero on both sides
    prediction = target.clone()
    prediction[0, H36M_NAMES.index("right_shoulder"), FORWARD_DEPTH_AXIS] = 0.05

    valid = torch.ones((n, len(H36M_NAMES)), dtype=torch.bool)
    total, count = _bilateral_forward_depth_residual_sum(torch, prediction, target, valid)
    assert math.isfinite(float(total))
    assert math.isfinite(float(count))
    assert float(total) > 0.0


def test_shoulder_and_hip_pairs_are_independent():
    import torch

    n = 1
    target = _zeroed(n)
    prediction = target.clone()
    prediction[0, H36M_NAMES.index("right_shoulder"), FORWARD_DEPTH_AXIS] = 0.4  # shoulder-only error

    valid = torch.ones((n, len(H36M_NAMES)), dtype=torch.bool)
    q_pred, q_target, _ = _bilateral_forward_depth_grid(torch, prediction, target, valid)
    residual = (q_pred - q_target).abs()
    assert float(residual[0, 0]) > 0.0  # shoulder column
    assert float(residual[0, 1]) == pytest.approx(0.0, abs=1e-6)  # hip column untouched


def test_gradient_reaches_only_shoulder_and_hip_forward_depth():
    """No gradient reaches unrelated joints, or the non-forward-depth
    coordinates of the shoulder/hip joints themselves."""
    import torch

    n = 1
    target = _zeroed(n)
    prediction = target.clone()
    prediction[0, H36M_NAMES.index("right_shoulder"), FORWARD_DEPTH_AXIS] = 0.4
    prediction.requires_grad_(True)

    valid = torch.ones((n, len(H36M_NAMES)), dtype=torch.bool)
    total, count = _bilateral_forward_depth_residual_sum(torch, prediction, target, valid)
    (total / count).backward()

    grad = prediction.grad[0]
    wrist = H36M_NAMES.index("left_wrist")
    right_shoulder = H36M_NAMES.index("right_shoulder")
    left_shoulder = H36M_NAMES.index("left_shoulder")
    assert grad[wrist, :].abs().sum().item() == 0.0
    assert grad[right_shoulder, 0].abs().item() == 0.0  # X (right) untouched
    assert grad[right_shoulder, 2].abs().item() == 0.0  # Z (up) untouched
    assert grad[right_shoulder, FORWARD_DEPTH_AXIS].abs().item() > 0.0
    assert grad[left_shoulder, FORWARD_DEPTH_AXIS].abs().item() > 0.0  # opposite-sign partner moves too


def test_smooth_l1_transition_matches_the_base_coordinate_loss_family():
    """Same beta/transition semantics as the base coordinate loss: quadratic
    below the default beta=1.0, linear above it, in q's own units."""
    import torch

    n = 1
    target = _zeroed(n)
    prediction = target.clone()
    left_index, right_index = H36M_NAMES.index("left_shoulder"), H36M_NAMES.index("right_shoulder")

    # |q_pred - q_target| = 0.5 (quadratic region): loss == 0.5 * q_error^2
    a_small = 0.25 * math.sqrt(2.0)  # right=+a, left=-a -> q_error = 2a/sqrt(2) = 0.5
    small_prediction = prediction.clone()
    small_prediction[0, right_index, FORWARD_DEPTH_AXIS] = a_small
    small_prediction[0, left_index, FORWARD_DEPTH_AXIS] = -a_small
    valid = torch.ones((n, len(H36M_NAMES)), dtype=torch.bool)
    total_small, count_small = _bilateral_forward_depth_residual_sum(torch, small_prediction, target, valid)
    shoulder_contribution_small = float(total_small) / float(count_small) * 2  # isolate shoulder-only sum (hip is 0)
    assert shoulder_contribution_small == pytest.approx(0.5 * 0.5 ** 2, abs=1e-5)

    # |q_pred - q_target| = 2.0 (linear region): loss == |q_error| - 0.5
    a_large = 1.0 * math.sqrt(2.0)  # q_error = 2.0
    large_prediction = prediction.clone()
    large_prediction[0, right_index, FORWARD_DEPTH_AXIS] = a_large
    large_prediction[0, left_index, FORWARD_DEPTH_AXIS] = -a_large
    total_large, count_large = _bilateral_forward_depth_residual_sum(torch, large_prediction, target, valid)
    shoulder_contribution_large = float(total_large) / float(count_large) * 2
    assert shoulder_contribution_large == pytest.approx(2.0 - 0.5, abs=1e-5)


def test_coordinate_equivalent_normalization_pools_into_the_base_coordinate_mean():
    """Section 4's contract: enabling the flag folds the relational residual
    into the *same* sum/count as the base per-scalar coordinate loss -- not a
    separately averaged-and-weighted term."""
    import torch

    n = 2
    joints = len(H36M_NAMES)
    torch.manual_seed(3)
    target = torch.randn((n, joints, 3))
    prediction = target + torch.randn((n, joints, 3)) * 0.1
    mask = torch.ones((n, joints, 1))
    valid = mask.squeeze(-1).bool()

    config_off = TrainingConfig(window=3, channels=8, epochs=1, batch_size=1,
                                 bilateral_forward_depth_supervision=False)
    config_on = TrainingConfig(window=3, channels=8, epochs=1, batch_size=1,
                                bilateral_forward_depth_supervision=True)

    coordinate_sum = (torch.nn.functional.smooth_l1_loss(prediction, target, reduction="none") * mask).sum()
    coordinate_count = mask.sum()
    relational_sum, relational_count = _bilateral_forward_depth_residual_sum(torch, prediction, target, valid)
    expected_on = (coordinate_sum + relational_sum) / (coordinate_count + relational_count)
    expected_off = coordinate_sum / coordinate_count

    assert float(_supervision_loss(torch, prediction, target, mask, config_off)) == pytest.approx(float(expected_off), abs=1e-5)
    assert float(_supervision_loss(torch, prediction, target, mask, config_on)) == pytest.approx(float(expected_on), abs=1e-5)
    # No arbitrary weight multiplying the relational term: the two totals
    # differ by exactly the accounting effect of two extra scalar coordinates
    # per valid pair, not by an independent multiplier.
    assert float(expected_on) != pytest.approx(float(expected_off) + float(relational_sum) / float(relational_count), abs=1e-5)


def test_bilateral_forward_depth_supervision_does_not_reactivate_other_terms():
    import torch

    n = 4
    target = _zeroed(n)
    prediction = target.clone()
    valid = torch.ones((n, len(H36M_NAMES), 1))
    config = TrainingConfig(window=3, channels=8, epochs=1, batch_size=1,
                             bilateral_forward_depth_supervision=True)

    assert config.bone_loss_weight == 0.0
    assert config.torso_loss_weight == 0.0
    assert config.yaw_tail_loss_weight == 0.0
    assert config.cartesian_torso_tail_loss_weight == 0.0
    # Prediction == target everywhere, so even with the flag on the folded
    # coordinate loss must still be exactly zero.
    assert float(_supervision_loss(torch, prediction, target, valid, config)) == pytest.approx(0.0)


def test_pair_validity_masks_out_invalid_shoulder_or_hip_observations():
    import torch

    n = 1
    target = _zeroed(n)
    prediction = target.clone()
    prediction[0, H36M_NAMES.index("right_shoulder"), FORWARD_DEPTH_AXIS] = 0.4

    valid = torch.ones((n, len(H36M_NAMES)), dtype=torch.bool)
    valid[0, H36M_NAMES.index("right_shoulder")] = False  # shoulder pair now invalid

    total, count = _bilateral_forward_depth_residual_sum(torch, prediction, target, valid)
    assert float(total) == pytest.approx(0.0, abs=1e-6)  # only the (now invalid) shoulder pair had error
    assert float(count) == pytest.approx(1.0)  # hip pair alone remains valid


def test_diagnostics_report_raw_meters_and_sign_disagreement_and_are_gradient_free():
    """Diagnostics use raw (un-normalized) meters, matching the prior 3DPW
    generalization-support diagnosis, and never require a gradient."""
    import torch

    n = 2
    target = _zeroed(n)
    _set_forward_y(target, 0, -0.1, 0.1, -0.05, 0.05)   # q_shoulder>0, q_hip>0
    _set_forward_y(target, 1, -0.1, 0.1, -0.05, 0.05)
    prediction = target.clone()
    # frame 0: shoulder sign flips, hip unchanged
    prediction[0, H36M_NAMES.index("left_shoulder"), FORWARD_DEPTH_AXIS] = 0.1
    prediction[0, H36M_NAMES.index("right_shoulder"), FORWARD_DEPTH_AXIS] = -0.1

    valid = torch.ones((n, len(H36M_NAMES)), dtype=torch.bool)
    diagnostics = _bilateral_forward_depth_diagnostics(torch, prediction, target, valid)

    assert diagnostics["shoulder_forward_depth_sign_disagreement"] == pytest.approx(0.5)  # 1 of 2 frames flipped
    assert diagnostics["hip_forward_depth_sign_disagreement"] == pytest.approx(0.0)
    assert diagnostics["shoulder_forward_depth_abs_residual_m"] > 0.0
    assert diagnostics["hip_forward_depth_abs_residual_m"] == pytest.approx(0.0, abs=1e-6)


def test_deterministic_replay_gives_identical_results():
    import torch

    n = 3
    torch.manual_seed(42)
    target = torch.randn((n, len(H36M_NAMES), 3))
    prediction = target + torch.randn((n, len(H36M_NAMES), 3)) * 0.05
    valid = torch.ones((n, len(H36M_NAMES)), dtype=torch.bool)

    first_total, first_count = _bilateral_forward_depth_residual_sum(torch, prediction, target, valid)
    second_total, second_count = _bilateral_forward_depth_residual_sum(torch, prediction, target, valid)
    assert float(first_total) == float(second_total)
    assert float(first_count) == float(second_count)


def test_grid_uses_the_same_pair_convention_as_torso_indices():
    """q is built on the identical shoulder-then-hip, right-minus-left
    convention TORSO_INDICES already uses -- not a new pairing."""
    assert TORSO_INDICES == (
        (H36M_NAMES.index("left_shoulder"), H36M_NAMES.index("right_shoulder")),
        (H36M_NAMES.index("left_hip"), H36M_NAMES.index("right_hip")),
    )
