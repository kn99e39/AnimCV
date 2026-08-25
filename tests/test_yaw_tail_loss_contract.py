"""Focused contract checks for _yaw_tail_loss before it is used in a real run.

Directive: verify the loss's selection semantics, gradient isolation, and
determinism against A9's own conditions before treating an A9 + yaw_tail
training run as trustworthy evidence about the promotion gate.
"""
import math

import pytest

pytest.importorskip("torch", reason="yaw_tail_loss contract checks require the optional training extra")

from pose.pose_lifter import H36M_NAMES
from training.temporal_lifter import (
    TrainingConfig, _root_yaw_error_degrees, _supervision_loss, _yaw_axis_loss, _yaw_tail_loss,
)


def _zeroed(n):
    import torch
    return torch.zeros((n, 17, 3))


def test_yaw_tail_loss_selects_exactly_the_top_ceil_5_percent_pooled_entries():
    """With one yaw pair (hip) held degenerate/unstable everywhere, the tail
    pool is exactly the shoulder-pair error per frame. Construct 100 distinct,
    known errors and confirm the loss equals the mean of only the top
    ceil(100/20)=5 -- not diluted by the other 95."""
    import torch

    n = 100
    target, prediction = _zeroed(n), _zeroed(n)
    ls, rs = H36M_NAMES.index("left_shoulder"), H36M_NAMES.index("right_shoulder")
    target[:, ls, :2] = torch.tensor([-1.0, 0.0])
    target[:, rs, :2] = torch.tensor([1.0, 0.0])
    thetas_deg = torch.arange(1, n + 1, dtype=torch.float32)  # 1..100 degrees, all distinct
    thetas = thetas_deg * (math.pi / 180.0)
    prediction[:, ls, :2] = torch.stack([-torch.cos(thetas), -torch.sin(thetas)], dim=-1)
    prediction[:, rs, :2] = torch.stack([torch.cos(thetas), torch.sin(thetas)], dim=-1)
    # Hip left/right coincide -> zero-length axis -> _yaw_axis_error_grid marks
    # it unstable everywhere, so it never enters the tail-selection pool.
    valid = torch.ones((n, 17), dtype=torch.bool)

    loss = _yaw_tail_loss(torch, prediction, target, valid)

    top5_thetas_rad = thetas[-5:]  # 96..100 degrees, the 5 largest
    expected = (1.0 - torch.cos(top5_thetas_rad)).mean()
    assert float(loss) == pytest.approx(float(expected), abs=1e-5)


def test_yaw_tail_loss_pools_pair_observations_not_frame_combined_error():
    """Documents a real granularity gap between this loss and the P95 gate
    it claims to target: the gate (_root_yaw_error_degrees) averages the
    shoulder and hip pair per frame *before* ranking; this loss pools every
    pair observation across every frame and ranks *those* directly. A frame
    with one very large single-pair error can therefore outrank a frame whose
    combined (official) yaw error is actually higher.

    On the real A9 3DPW holdout, frames where the two pairs disagree by
    >= 20 degrees are 1,300 of 34,456 both-pairs-available frames (3.8%,
    scripts/attribute_yaw_tail.py, 2026-08-25) -- this failure mode is real
    but not dominant.
    """
    import torch

    # Frame A: both pairs at 50 degrees -> official combined error 50.
    # Frame B: shoulder at 60, hip at 5 -> official combined error 32.5,
    # despite having the single largest pooled entry (60 > 50).
    n = 2
    target, prediction = _zeroed(n), _zeroed(n)
    ls, rs = H36M_NAMES.index("left_shoulder"), H36M_NAMES.index("right_shoulder")
    lh, rh = H36M_NAMES.index("left_hip"), H36M_NAMES.index("right_hip")
    target[:, ls, :2] = torch.tensor([-1.0, 0.0])
    target[:, rs, :2] = torch.tensor([1.0, 0.0])
    target[:, lh, :2] = torch.tensor([-1.0, 0.0])
    target[:, rh, :2] = torch.tensor([1.0, 0.0])

    def rotated(angle_deg):
        theta = math.radians(angle_deg)
        return torch.tensor([-math.cos(theta), -math.sin(theta)]), torch.tensor([math.cos(theta), math.sin(theta)])

    a_shoulder_l, a_shoulder_r = rotated(50.0)
    a_hip_l, a_hip_r = rotated(50.0)
    b_shoulder_l, b_shoulder_r = rotated(60.0)
    b_hip_l, b_hip_r = rotated(5.0)
    prediction[0, ls, :2], prediction[0, rs, :2] = a_shoulder_l, a_shoulder_r
    prediction[0, lh, :2], prediction[0, rh, :2] = a_hip_l, a_hip_r
    prediction[1, ls, :2], prediction[1, rs, :2] = b_shoulder_l, b_shoulder_r
    prediction[1, lh, :2], prediction[1, rh, :2] = b_hip_l, b_hip_r
    valid = torch.ones((n, 17), dtype=torch.bool)

    official_a = _root_yaw_error_degrees(prediction[0].numpy(), target[0].numpy(), valid[0].numpy())
    official_b = _root_yaw_error_degrees(prediction[1].numpy(), target[1].numpy(), valid[1].numpy())
    assert official_a == pytest.approx(50.0, abs=0.5)
    assert official_b == pytest.approx(32.5, abs=0.5)
    assert official_a > official_b  # frame A is the officially worse frame

    # With tail_count clamped to 1 (four pooled entries total; ceil(4/20)=1
    # via clamp_min), the loss must reduce to the single largest pooled
    # entry -- frame B's shoulder pair (60 degrees) -- even though frame A is
    # the officially worse frame by the gate's own per-frame definition.
    loss = _yaw_tail_loss(torch, prediction, target, valid)
    expected_from_b_shoulder = 1.0 - math.cos(math.radians(60.0))
    assert float(loss) == pytest.approx(expected_from_b_shoulder, abs=1e-5)


def test_yaw_tail_loss_gradient_is_isolated_to_selected_pair_joints():
    """A perturbation on a joint that never enters the selected tail (an
    unrelated wrist, with both yaw pairs already perfect) must not receive
    gradient from this loss."""
    import torch

    n = 40
    target = _zeroed(n)
    prediction = target.clone().requires_grad_(True)
    ls, rs = H36M_NAMES.index("left_shoulder"), H36M_NAMES.index("right_shoulder")
    lh, rh = H36M_NAMES.index("left_hip"), H36M_NAMES.index("right_hip")
    with torch.no_grad():
        target[:, ls, :2] = torch.tensor([-1.0, 0.0])
        target[:, rs, :2] = torch.tensor([1.0, 0.0])
        target[:, lh, :2] = torch.tensor([-1.0, 0.0])
        target[:, rh, :2] = torch.tensor([1.0, 0.0])
        prediction[:, ls, :2] = torch.tensor([-1.0, 0.0])
        prediction[:, rs, :2] = torch.tensor([1.0, 0.0])
        prediction[:, lh, :2] = torch.tensor([-1.0, 0.0])
        prediction[:, rh, :2] = torch.tensor([1.0, 0.0])
        # Perturb one sample's shoulder pair so it is the sole tail entry.
        theta = math.radians(45.0)
        prediction[0, ls, :2] = torch.tensor([-math.cos(theta), -math.sin(theta)])
        prediction[0, rs, :2] = torch.tensor([math.cos(theta), math.sin(theta)])
    valid = torch.ones((n, 17), dtype=torch.bool)

    loss = _yaw_tail_loss(torch, prediction, target, valid)
    loss.backward()

    wrist = H36M_NAMES.index("left_wrist")
    assert prediction.grad[:, wrist, :].abs().sum().item() == 0.0
    assert prediction.grad[0, ls, :2].abs().sum().item() > 0.0
    # No other sample's shoulder/hip pair was selected into the tail.
    assert prediction.grad[1:, ls, :2].abs().sum().item() == 0.0
    assert prediction.grad[:, lh, :2].abs().sum().item() == 0.0


def test_yaw_tail_loss_weight_does_not_affect_other_structural_losses_when_isolated():
    """A9's planned control (yaw_tail_loss_weight=0.05, everything else forced
    off) must not silently reactivate yaw/hinge_flip/end_effector terms."""
    import torch

    n = 8
    target = _zeroed(n)
    prediction = target.clone()
    valid = torch.ones((n, 17, 1))
    config_a9 = TrainingConfig(window=3, channels=8, epochs=1, batch_size=1)
    config_a11 = TrainingConfig(window=3, channels=8, epochs=1, batch_size=1, yaw_tail_loss_weight=0.05)

    assert config_a11.yaw_loss_weight == 0.0
    assert config_a11.hinge_flip_loss_weight == 0.0
    assert config_a11.end_effector_loss_weight == 0.0
    # Matching predictions/targets: every structural term (including
    # yaw_tail) must be exactly zero regardless of which config is used.
    assert float(_supervision_loss(torch, prediction, target, valid, config_a9)) == pytest.approx(0.0)
    assert float(_supervision_loss(torch, prediction, target, valid, config_a11)) == pytest.approx(0.0)


def test_yaw_tail_loss_matches_yaw_axis_loss_for_a_single_uniform_sample():
    """Existing regression anchor (mirrors test_supervised_temporal_lifter.py):
    with one sample, the "tail" is trivially the whole batch, so yaw_tail and
    the ordinary bilateral yaw loss must agree."""
    import torch

    target = _zeroed(1)
    prediction = target.clone()
    ls, rs = H36M_NAMES.index("left_shoulder"), H36M_NAMES.index("right_shoulder")
    target[0, ls, :2] = torch.tensor([-1.0, 0.0])
    target[0, rs, :2] = torch.tensor([1.0, 0.0])
    prediction[0, ls, :2] = torch.tensor([0.0, -1.0])
    prediction[0, rs, :2] = torch.tensor([0.0, 1.0])
    valid = torch.zeros((1, 17), dtype=torch.bool)
    valid[0, [ls, rs]] = True

    assert _yaw_tail_loss(torch, prediction, target, valid) == pytest.approx(
        _yaw_axis_loss(torch, prediction, target, valid)
    )
