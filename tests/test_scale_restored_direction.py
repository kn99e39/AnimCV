"""Contracts for the scale-restored direction-only torso candidate.

These tests exercise the counterfactual before it is allowed into a training
run.  They do not change A12's Cartesian loss or its historical artifacts.
"""

import math

import pytest

pytest.importorskip("torch", reason="direction candidate requires torch")

from pose.pose_lifter import H36M_NAMES
from training.temporal_lifter import (
    _scale_restored_direction_torso_error_grid,
    _scale_restored_direction_torso_tail_loss,
)


def _fixture(n=1):
    import torch

    target = torch.zeros((n, len(H36M_NAMES), 3), dtype=torch.float32)
    ls, rs = H36M_NAMES.index("left_shoulder"), H36M_NAMES.index("right_shoulder")
    lh, rh = H36M_NAMES.index("left_hip"), H36M_NAMES.index("right_hip")
    target[:, ls] = torch.tensor((-1.0, 0.0, 0.0))
    target[:, rs] = torch.tensor((1.0, 0.0, 0.0))
    target[:, lh] = torch.tensor((-0.5, 0.0, 0.0))
    target[:, rh] = torch.tensor((0.5, 0.0, 0.0))
    valid = torch.zeros((n, len(H36M_NAMES)), dtype=torch.bool)
    valid[:, [ls, rs, lh, rh]] = True
    return target, valid, (ls, rs, lh, rh)


def _rotate_pair(target, left, right, degrees, scale=1.0):
    import torch

    theta = math.radians(degrees)
    direction = torch.tensor((math.cos(theta), math.sin(theta), 0.0))
    midpoint = (target[:, left] + target[:, right]) / 2.0
    half_span = (target[:, right] - target[:, left]).norm(dim=-1, keepdim=True) / 2.0 * scale
    result = target.clone()
    result[:, left] = midpoint - half_span * direction
    result[:, right] = midpoint + half_span * direction
    return result


def test_identical_vectors_and_uniform_translation_are_zero():
    import torch

    target, valid, _ = _fixture()
    assert float(_scale_restored_direction_torso_tail_loss(torch, target, target, valid)) == pytest.approx(0.0)

    translated_prediction = target + torch.tensor((7.0, -3.0, 2.0))
    assert float(_scale_restored_direction_torso_tail_loss(torch, translated_prediction, target, valid)) == pytest.approx(0.0)


def test_same_direction_different_magnitude_has_zero_direction_residual():
    import torch

    target, valid, (ls, rs, lh, rh) = _fixture()
    prediction = target.clone()
    prediction[:, ls] = torch.tensor((-3.0, 0.0, 0.0))
    prediction[:, rs] = torch.tensor((3.0, 0.0, 0.0))
    prediction[:, lh] = torch.tensor((-2.0, 0.0, 0.0))
    prediction[:, rh] = torch.tensor((2.0, 0.0, 0.0))

    errors, stable, _residual, _geometry = _scale_restored_direction_torso_error_grid(
        torch, prediction, target, valid,
    )
    assert stable.all()
    assert torch.allclose(errors, torch.zeros_like(errors), atol=1e-7)


def test_rotation_and_opposite_direction_are_detected_for_both_pairs():
    import torch

    target, valid, (ls, rs, lh, rh) = _fixture()
    rotated = _rotate_pair(target, ls, rs, 35.0)
    rotated = _rotate_pair(rotated, lh, rh, 35.0, scale=1.7)
    moderate = float(_scale_restored_direction_torso_tail_loss(torch, rotated, target, valid))

    opposite = target.clone()
    opposite[:, ls], opposite[:, rs] = target[:, rs], target[:, ls]
    opposite[:, lh], opposite[:, rh] = target[:, rh], target[:, lh]
    large = float(_scale_restored_direction_torso_tail_loss(torch, opposite, target, valid))

    assert moderate > 0.0
    assert large > moderate


def test_gradient_targets_only_selected_torso_endpoints():
    import torch

    target, valid, (ls, rs, lh, rh) = _fixture()
    prediction = _rotate_pair(target, ls, rs, 45.0).requires_grad_()
    loss = _scale_restored_direction_torso_tail_loss(torch, prediction, target, valid)
    loss.backward()

    assert torch.isfinite(prediction.grad).all()
    non_torso = [index for index in range(len(H36M_NAMES)) if index not in (ls, rs, lh, rh)]
    assert torch.equal(prediction.grad[:, non_torso], torch.zeros_like(prediction.grad[:, non_torso]))
    assert prediction.grad[:, [ls, rs]].abs().sum() > 0


def test_target_length_scale_is_detached_from_optimization():
    import torch

    target, valid, (ls, rs, _lh, _rh) = _fixture()
    scale = torch.tensor(2.0, requires_grad=True)
    offset = torch.zeros_like(target)
    offset[:, ls, 0] = -scale
    offset[:, rs, 0] = scale
    target = target + offset
    prediction = _rotate_pair(target.detach(), ls, rs, 40.0)

    loss = _scale_restored_direction_torso_tail_loss(torch, prediction, target, valid)
    loss.backward()

    assert torch.isfinite(loss)
    assert scale.grad is not None
    assert float(scale.grad.abs()) == pytest.approx(0.0, abs=1e-7)


def test_collapsed_predicted_span_has_finite_loss_and_gradient():
    import torch

    target, valid, (ls, rs, _lh, _rh) = _fixture()
    prediction = target.clone()
    prediction[:, ls] = 0.0
    prediction[:, rs] = 0.0
    prediction.requires_grad_()

    loss = _scale_restored_direction_torso_tail_loss(torch, prediction, target, valid)
    loss.backward()

    assert torch.isfinite(loss)
    assert torch.isfinite(prediction.grad).all()
    assert float(prediction.grad[:, [ls, rs]].abs().sum()) > 0
