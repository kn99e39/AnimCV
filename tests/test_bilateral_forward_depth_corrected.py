"""Synthetic contract checks for the docs/21 CORRECTED bilateral forward-depth
candidate (distinct from the historical, denominator-contaminated A14 flag).

Repository audit found that historical A14's coordinate loss changed the
denominator from ``mask.sum()`` (D_coord) to ``mask.sum() + relational_count``,
unintentionally attenuating the base coordinate gradient relative to plain A9.
This file verifies the corrected reduction contract:

    L_coordinate_corrected = S_coord / D_coord + S_relational / D_coord

i.e. the SAME numerator-only addition, with D_coord left exactly as historical
A9 computed it. ``bilateral_forward_depth_supervision`` (historical A14) and
``bilateral_forward_depth_supervision_corrected`` (this file) are two
different TrainingConfig flags on purpose, so historical A14 stays exactly
reproducible from current code.
"""
import math

import pytest

pytest.importorskip("torch", reason="corrected SRD checks require the optional training extra")

from pose.pose_lifter import H36M_NAMES
from training.temporal_lifter import (
    BILATERAL_DEPTH_NORMALIZATION, FORWARD_DEPTH_AXIS, TORSO_INDICES, TrainingConfig,
    _bilateral_forward_depth_residual_sum, _supervision_loss,
)


def _zeroed(n):
    import torch
    return torch.zeros((n, len(H36M_NAMES), 3))


def _set_forward_y(tensor, index, left_shoulder_y, right_shoulder_y, left_hip_y, right_hip_y):
    tensor[index, H36M_NAMES.index("left_shoulder"), FORWARD_DEPTH_AXIS] = left_shoulder_y
    tensor[index, H36M_NAMES.index("right_shoulder"), FORWARD_DEPTH_AXIS] = right_shoulder_y
    tensor[index, H36M_NAMES.index("left_hip"), FORWARD_DEPTH_AXIS] = left_hip_y
    tensor[index, H36M_NAMES.index("right_hip"), FORWARD_DEPTH_AXIS] = right_hip_y


_A9_STRUCTURAL_WEIGHTS = dict(bone_loss_weight=0.25, torso_loss_weight=0.15, hinge_loss_weight=0.15)


def _config(**overrides):
    return TrainingConfig(window=3, channels=8, epochs=1, batch_size=1, **_A9_STRUCTURAL_WEIGHTS, **overrides)


# --- 1. corrected feature OFF -> exact historical A9 loss --------------------

def test_corrected_flag_off_matches_plain_a9_loss():
    import torch

    n = 3
    torch.manual_seed(1)
    target = torch.randn((n, len(H36M_NAMES), 3))
    prediction = target + torch.randn((n, len(H36M_NAMES), 3)) * 0.1
    mask = torch.ones((n, len(H36M_NAMES), 1))
    config_off = _config(bilateral_forward_depth_supervision_corrected=False)

    a9_coordinate = (torch.nn.functional.smooth_l1_loss(prediction, target, reduction="none") * mask).sum() / mask.sum()
    total_off = _supervision_loss(torch, prediction, target, mask, config_off)
    total_a9_only = a9_coordinate  # plus the same structural terms _supervision_loss would add
    # Compare against _supervision_loss with an equivalent config that never
    # had the corrected/historical flags at all -- the default is False for both.
    default_config = _config()
    total_default = _supervision_loss(torch, prediction, target, mask, default_config)
    assert float(total_off) == pytest.approx(float(total_default))


# --- 2. feature ON but relational residual == 0 -> exact historical A9 loss/gradients --

def test_corrected_flag_on_with_zero_relational_residual_matches_a9_exactly():
    """When prediction == target at the shoulder/hip forward-depth
    coordinates, q_pred == q_target so S_relational == 0 and the corrected
    total must equal plain A9's total exactly (not just approximately)."""
    import torch

    n = 2
    torch.manual_seed(2)
    target = torch.randn((n, len(H36M_NAMES), 3))
    prediction = target.clone()  # identical everywhere, including shoulder/hip forward-depth
    prediction.requires_grad_(True)
    mask = torch.ones((n, len(H36M_NAMES), 1))

    config_corrected = _config(bilateral_forward_depth_supervision_corrected=True)
    config_a9 = _config()

    loss_corrected = _supervision_loss(torch, prediction, target, mask, config_corrected)
    loss_corrected.backward()
    grad_corrected = prediction.grad.clone()

    prediction2 = target.clone()
    prediction2.requires_grad_(True)
    loss_a9 = _supervision_loss(torch, prediction2, target, mask, config_a9)
    loss_a9.backward()
    grad_a9 = prediction2.grad.clone()

    assert float(loss_corrected.detach()) == pytest.approx(float(loss_a9.detach()), abs=1e-7)
    assert torch.allclose(grad_corrected, grad_a9, atol=1e-7)


# --- 3. relational residual nonzero -> base coordinate gradient unchanged ----

def test_delta_gradient_equals_relational_term_gradient_alone():
    """Delta_G = G_corrected - G_A9 must equal the gradient of
    S_relational / D_coord alone -- i.e. the base S_coord/D_coord
    contribution is untouched by adding the corrected term."""
    import torch

    n = 4
    torch.manual_seed(3)
    target = torch.randn((n, len(H36M_NAMES), 3))
    base_prediction = target + torch.randn((n, len(H36M_NAMES), 3)) * 0.2
    mask = torch.ones((n, len(H36M_NAMES), 1))
    valid = mask.squeeze(-1).bool()

    prediction_a9 = base_prediction.clone().requires_grad_(True)
    loss_a9 = _supervision_loss(torch, prediction_a9, target, mask, _config())
    loss_a9.backward()
    g_a9 = prediction_a9.grad.clone()

    prediction_corrected = base_prediction.clone().requires_grad_(True)
    loss_corrected = _supervision_loss(
        torch, prediction_corrected, target, mask, _config(bilateral_forward_depth_supervision_corrected=True),
    )
    loss_corrected.backward()
    g_corrected = prediction_corrected.grad.clone()

    prediction_relational_only = base_prediction.clone().requires_grad_(True)
    relational_sum, _count = _bilateral_forward_depth_residual_sum(
        torch, prediction_relational_only, target, valid,
    )
    (relational_sum / mask.sum()).backward()
    g_relational_only = prediction_relational_only.grad.clone()

    delta_g = g_corrected - g_a9
    assert torch.allclose(delta_g, g_relational_only, atol=1e-6)
    # And loss values decompose the same way.
    assert float(loss_corrected) == pytest.approx(
        float(loss_a9) + float(relational_sum / mask.sum()), abs=1e-6,
    )


# --- 4. pure common +Y translation -> relational term zero -------------------

def test_pure_common_translation_leaves_corrected_loss_equal_to_a9():
    import torch

    n = 1
    target = _zeroed(n)
    _set_forward_y(target, 0, 0.3, -0.1, 0.2, -0.2)
    prediction = target.clone()
    common_shift = 0.5
    prediction[0, H36M_NAMES.index("left_shoulder"), FORWARD_DEPTH_AXIS] += common_shift
    prediction[0, H36M_NAMES.index("right_shoulder"), FORWARD_DEPTH_AXIS] += common_shift
    mask = torch.ones((n, len(H36M_NAMES), 1))

    loss_corrected = _supervision_loss(
        torch, prediction, target, mask, _config(bilateral_forward_depth_supervision_corrected=True),
    )
    loss_a9 = _supervision_loss(torch, prediction, target, mask, _config())
    # The common-mode shift still changes S_coord (ordinary per-joint
    # residual) identically in both, and q is invariant to it -> equal totals.
    assert float(loss_corrected) == pytest.approx(float(loss_a9), abs=1e-6)


# --- 5. pure anti-symmetric +Y error -> opposite endpoint gradients ----------

def test_anti_symmetric_error_produces_opposite_endpoint_gradients_in_corrected_loss():
    import torch

    n = 1
    target = _zeroed(n)
    prediction = target.clone()
    a = 0.15
    left_index, right_index = H36M_NAMES.index("left_shoulder"), H36M_NAMES.index("right_shoulder")
    prediction[0, right_index, FORWARD_DEPTH_AXIS] = a
    prediction[0, left_index, FORWARD_DEPTH_AXIS] = -a
    prediction.requires_grad_(True)
    mask = torch.ones((n, len(H36M_NAMES), 1))

    loss = _supervision_loss(
        torch, prediction, target, mask, _config(bilateral_forward_depth_supervision_corrected=True),
    )
    loss.backward()
    right_grad = float(prediction.grad[0, right_index, FORWARD_DEPTH_AXIS])
    left_grad = float(prediction.grad[0, left_index, FORWARD_DEPTH_AXIS])
    assert right_grad > 0.0
    assert left_grad < 0.0
    assert right_grad == pytest.approx(-left_grad, abs=1e-6)


# --- 6. unrelated joints -> no relational gradient contribution -------------

def test_unrelated_joint_gradient_unaffected_by_corrected_term():
    import torch

    n = 1
    target = _zeroed(n)
    prediction = target.clone()
    wrist = H36M_NAMES.index("left_wrist")
    prediction[0, wrist, FORWARD_DEPTH_AXIS] = 0.3  # an ordinary coordinate error, unrelated to q
    prediction.requires_grad_(True)
    mask = torch.ones((n, len(H36M_NAMES), 1))

    grad_corrected = torch.autograd.grad(
        _supervision_loss(torch, prediction, target, mask, _config(bilateral_forward_depth_supervision_corrected=True)),
        prediction,
    )[0][0, wrist, FORWARD_DEPTH_AXIS]

    prediction2 = prediction.detach().clone().requires_grad_(True)
    grad_a9 = torch.autograd.grad(
        _supervision_loss(torch, prediction2, target, mask, _config()), prediction2,
    )[0][0, wrist, FORWARD_DEPTH_AXIS]

    assert float(grad_corrected) == pytest.approx(float(grad_a9), abs=1e-6)


# --- 7. shoulder/hip independently ------------------------------------------

def test_shoulder_and_hip_relational_terms_are_independent_in_corrected_loss():
    import torch

    n = 1
    target = _zeroed(n)
    prediction = target.clone()
    prediction[0, H36M_NAMES.index("right_shoulder"), FORWARD_DEPTH_AXIS] = 0.4  # shoulder-only error
    prediction.requires_grad_(True)
    mask = torch.ones((n, len(H36M_NAMES), 1))

    loss = _supervision_loss(
        torch, prediction, target, mask, _config(bilateral_forward_depth_supervision_corrected=True),
    )
    loss.backward()
    left_hip = H36M_NAMES.index("left_hip")
    right_hip = H36M_NAMES.index("right_hip")
    # Hip endpoints receive no gradient from the relational term (only from
    # their own zero ordinary coordinate residual, which is exactly zero here).
    assert float(prediction.grad[0, left_hip, FORWARD_DEPTH_AXIS]) == pytest.approx(0.0, abs=1e-7)
    assert float(prediction.grad[0, right_hip, FORWARD_DEPTH_AXIS]) == pytest.approx(0.0, abs=1e-7)


# --- 8. pair invalidity -> zero relational contribution, no denominator change --

def test_invalid_pair_contributes_zero_relational_term_and_leaves_denominator_unchanged():
    import torch

    n = 1
    target = _zeroed(n)
    prediction = target.clone()
    prediction[0, H36M_NAMES.index("right_shoulder"), FORWARD_DEPTH_AXIS] = 0.4
    mask_valid = torch.ones((n, len(H36M_NAMES), 1))
    mask_shoulder_invalid = mask_valid.clone()
    mask_shoulder_invalid[0, H36M_NAMES.index("right_shoulder"), 0] = 0.0

    loss_valid = _supervision_loss(
        torch, prediction, target, mask_valid, _config(bilateral_forward_depth_supervision_corrected=True),
    )
    loss_invalid_pair = _supervision_loss(
        torch, prediction, target, mask_shoulder_invalid, _config(bilateral_forward_depth_supervision_corrected=True),
    )
    # With the shoulder pair invalid, both S_relational's shoulder
    # contribution AND the ordinary per-joint coordinate residual for that
    # joint drop out of D_coord too (same mask feeds both) -- the two totals
    # must therefore differ (denominator did shrink from the *mask*, not
    # from relational bookkeeping), and the invalid-pair total's coordinate
    # denominator is exactly mask_shoulder_invalid.sum(), not adjusted further.
    coordinate_sum_invalid = (torch.nn.functional.smooth_l1_loss(prediction, target, reduction="none")
                               * mask_shoulder_invalid).sum()
    expected_denominator = mask_shoulder_invalid.sum()
    assert float(loss_invalid_pair) != pytest.approx(float(loss_valid))
    # Recompute the coordinate term alone (no structural weights) to confirm
    # the exact denominator contract: S_coord (masked) + 0 relational, over
    # the mask-derived count only.
    zero_structural_config = TrainingConfig(window=3, channels=8, epochs=1, batch_size=1,
                                             bilateral_forward_depth_supervision_corrected=True)
    coordinate_only = _supervision_loss(torch, prediction, target, mask_shoulder_invalid, zero_structural_config)
    assert float(coordinate_only) == pytest.approx(float(coordinate_sum_invalid / expected_denominator), abs=1e-6)


# --- 9. correct +Y axis -------------------------------------------------------

def test_forward_depth_axis_is_canonical_y():
    assert FORWARD_DEPTH_AXIS == 1


# --- 10. sqrt(2) normalization contract --------------------------------------

def test_normalization_constant_is_orthonormal_basis_factor():
    assert BILATERAL_DEPTH_NORMALIZATION == pytest.approx(1.0 / math.sqrt(2.0))


# --- 11. identical seed/config with feature disabled -> historical contract preserved --

def test_both_flags_disabled_reproduces_historical_a9_reduction_exactly():
    import torch

    n = 5
    torch.manual_seed(42)
    target = torch.randn((n, len(H36M_NAMES), 3))
    prediction = target + torch.randn((n, len(H36M_NAMES), 3)) * 0.05
    mask = torch.ones((n, len(H36M_NAMES), 1))
    config = _config(bilateral_forward_depth_supervision=False, bilateral_forward_depth_supervision_corrected=False)

    coordinate_sum = (torch.nn.functional.smooth_l1_loss(prediction, target, reduction="none") * mask).sum()
    coordinate_count = mask.sum()
    expected_coordinate = coordinate_sum / coordinate_count

    from training.temporal_lifter import BONE_INDICES, TORSO_INDICES, _hinge_loss, _vector_loss
    valid = mask.squeeze(-1).bool()
    expected_total = (
        expected_coordinate
        + 0.25 * _vector_loss(torch, prediction, target, valid, BONE_INDICES, lambda first, second: first - second)
        + 0.15 * _vector_loss(torch, prediction, target, valid, TORSO_INDICES, lambda first, second: second - first)
        + 0.15 * _hinge_loss(torch, prediction, target, valid)
    )
    actual = _supervision_loss(torch, prediction, target, mask, config)
    assert float(actual) == pytest.approx(float(expected_total), abs=1e-6)


def test_both_flags_true_simultaneously_is_rejected():
    with pytest.raises(ValueError, match="mutually exclusive"):
        TrainingConfig(bilateral_forward_depth_supervision=True, bilateral_forward_depth_supervision_corrected=True)
