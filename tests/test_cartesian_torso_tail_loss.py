"""Contract checks for the Cartesian torso-tail loss candidate (Section 7).

Verifies the candidate actually reacts to orientation-relevant torso error,
is translation-invariant by construction, targets the right joints with
gradient, and reuses the same pooled tail-selection mechanism as the
existing (now-rejected) angular yaw_tail_loss -- so the only thing under
test across the two is the penalty's representation, not its selector.
"""
import math

import pytest

pytest.importorskip("torch", reason="Cartesian torso-tail loss checks require the optional training extra")

from pose.pose_lifter import H36M_NAMES
from training.temporal_lifter import (
    TrainingConfig, _cartesian_torso_tail_loss, _pooled_tail_mean, _supervision_loss,
    _torso_vector_error_grid, _vector_loss, TORSO_INDICES,
)


def _zeroed(n):
    import torch
    return torch.zeros((n, len(H36M_NAMES), 3))


def _set_torso(tensor, index, left_shoulder, right_shoulder, left_hip, right_hip):
    import torch
    tensor[index, H36M_NAMES.index("left_shoulder")] = torch.tensor(left_shoulder)
    tensor[index, H36M_NAMES.index("right_shoulder")] = torch.tensor(right_shoulder)
    tensor[index, H36M_NAMES.index("left_hip")] = torch.tensor(left_hip)
    tensor[index, H36M_NAMES.index("right_hip")] = torch.tensor(right_hip)


def test_correct_torso_geometry_gives_near_zero_penalty():
    import torch

    n = 1
    target = _zeroed(n)
    _set_torso(target, 0, [-1, 0, 0], [1, 0, 0], [-1, 0, -1], [1, 0, -1])
    prediction = target.clone()
    valid = torch.ones((n, len(H36M_NAMES)), dtype=torch.bool)

    assert float(_cartesian_torso_tail_loss(torch, prediction, target, valid)) == pytest.approx(0.0, abs=1e-6)


def test_incorrect_shoulder_bilateral_vector_increases_penalty():
    import torch

    n = 1
    target = _zeroed(n)
    _set_torso(target, 0, [-1, 0, 0], [1, 0, 0], [-1, 0, -1], [1, 0, -1])
    prediction = target.clone()
    prediction[0, H36M_NAMES.index("right_shoulder"), 0] = 1.5  # shoulder vector now wrong
    valid = torch.ones((n, len(H36M_NAMES)), dtype=torch.bool)

    penalty = float(_cartesian_torso_tail_loss(torch, prediction, target, valid))
    assert penalty > 0.01


def test_incorrect_hip_bilateral_vector_increases_penalty():
    import torch

    n = 1
    target = _zeroed(n)
    _set_torso(target, 0, [-1, 0, 0], [1, 0, 0], [-1, 0, -1], [1, 0, -1])
    prediction = target.clone()
    prediction[0, H36M_NAMES.index("right_hip"), 0] = 1.5  # hip vector now wrong
    valid = torch.ones((n, len(H36M_NAMES)), dtype=torch.bool)

    penalty = float(_cartesian_torso_tail_loss(torch, prediction, target, valid))
    assert penalty > 0.01


def test_torso_orientation_perturbation_with_unchanged_root_position_increases_penalty():
    """A pure in-plane rotation of the shoulder/hip pairs about the pelvis --
    the root-relative pose otherwise unchanged -- must still register as an
    orientation error, exactly the case the angular yaw_tail_loss targeted."""
    import torch

    n = 1
    target = _zeroed(n)
    _set_torso(target, 0, [-1, 0, 0], [1, 0, 0], [-1, 0, -1], [1, 0, -1])
    prediction = target.clone()
    theta = math.radians(45.0)
    rotated_left = [-math.cos(theta), -math.sin(theta), 0.0]
    rotated_right = [math.cos(theta), math.sin(theta), 0.0]
    prediction[0, H36M_NAMES.index("left_shoulder")] = torch.tensor(rotated_left)
    prediction[0, H36M_NAMES.index("right_shoulder")] = torch.tensor(rotated_right)
    valid = torch.ones((n, len(H36M_NAMES)), dtype=torch.bool)

    penalty = float(_cartesian_torso_tail_loss(torch, prediction, target, valid))
    assert penalty > 0.01


def test_uniform_translation_does_not_create_a_false_penalty():
    """A vector *difference* (right - left) is invariant to translating the
    whole skeleton by construction -- this is exactly what the angular
    formulation could not offer for free (it needed axis-length guards)."""
    import torch

    n = 1
    target = _zeroed(n)
    _set_torso(target, 0, [-1, 0, 0], [1, 0, 0], [-1, 0, -1], [1, 0, -1])
    prediction = target + 5.0  # every joint shifted by the same constant offset
    valid = torch.ones((n, len(H36M_NAMES)), dtype=torch.bool)

    assert float(_cartesian_torso_tail_loss(torch, prediction, target, valid)) == pytest.approx(0.0, abs=1e-5)


def test_gradient_reaches_only_the_perturbed_shoulder_and_hip_predictions():
    import torch

    n = 1
    target = _zeroed(n)
    _set_torso(target, 0, [-1, 0, 0], [1, 0, 0], [-1, 0, -1], [1, 0, -1])
    prediction = target.clone()
    prediction[0, H36M_NAMES.index("right_shoulder"), 0] = 1.5
    prediction.requires_grad_(True)
    valid = torch.ones((n, len(H36M_NAMES)), dtype=torch.bool)

    loss = _cartesian_torso_tail_loss(torch, prediction, target, valid)
    loss.backward()

    wrist = H36M_NAMES.index("left_wrist")
    right_shoulder = H36M_NAMES.index("right_shoulder")
    assert prediction.grad[0, wrist, :].abs().sum().item() == 0.0
    assert prediction.grad[0, right_shoulder, :].abs().sum().item() > 0.0


def test_torso_vector_error_grid_matches_the_existing_torso_structural_loss_per_pair():
    """The un-pooled per-(frame, pair) grid must reduce, when simply averaged
    over every valid pair, to exactly what the existing (already-shipped,
    equally-weighted) torso_loss_weight term computes -- proof this reuses
    torso geometry semantics rather than a new, incompatible one."""
    import torch

    n = 5
    torch.manual_seed(11)
    target = _zeroed(n)
    _set_torso(target, slice(0, n), [-1, 0, 0], [1, 0, 0], [-1, 0, -1], [1, 0, -1])
    prediction = target + torch.randn_like(target) * 0.1
    valid = torch.ones((n, len(H36M_NAMES)), dtype=torch.bool)

    errors, stable = _torso_vector_error_grid(torch, prediction, target, valid)
    manual_mean = (errors * stable).sum() / stable.sum()
    existing_torso_loss = _vector_loss(torch, prediction, target, valid, TORSO_INDICES, lambda first, second: second - first)

    assert float(manual_mean) == pytest.approx(float(existing_torso_loss), abs=1e-5)


def test_cartesian_torso_tail_loss_uses_the_same_pooled_selector_as_yaw_tail():
    """Same mechanism, different input grid -- constructed so only one of
    four pooled entries is selected (tail_count clamps to 1), and that the
    selected value is exactly the largest one, matching _pooled_tail_mean's
    own contract already verified in test_yaw_tail_loss_contract.py."""
    import torch

    n = 2
    target = _zeroed(n)
    _set_torso(target, slice(0, n), [-1, 0, 0], [1, 0, 0], [-1, 0, -1], [1, 0, -1])
    prediction = target.clone()
    prediction[0, H36M_NAMES.index("right_shoulder"), 0] = 1.3  # small error
    prediction[1, H36M_NAMES.index("right_hip"), 0] = 3.0  # the single largest pooled entry
    valid = torch.ones((n, len(H36M_NAMES)), dtype=torch.bool)

    errors, stable = _torso_vector_error_grid(torch, prediction, target, valid)
    expected = float(errors.flatten().max())
    assert float(_cartesian_torso_tail_loss(torch, prediction, target, valid)) == pytest.approx(expected, abs=1e-5)


def test_cartesian_torso_tail_loss_weight_is_isolated_from_other_terms():
    import torch

    n = 4
    target = _zeroed(n)
    prediction = target.clone()
    valid = torch.ones((n, len(H36M_NAMES), 1))
    config = TrainingConfig(window=3, channels=8, epochs=1, batch_size=1, cartesian_torso_tail_loss_weight=0.05)

    assert config.yaw_tail_loss_weight == 0.0
    assert config.hinge_flip_loss_weight == 0.0
    assert config.end_effector_loss_weight == 0.0
    assert float(_supervision_loss(torch, prediction, target, valid, config)) == pytest.approx(0.0)
