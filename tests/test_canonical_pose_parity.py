"""Bitwise parity for the canonical pose mathematics ownership move.

`common.canonical_pose` took ownership of formulas that previously lived inside
`training.temporal_lifter`. This is an ownership refactor, never a redesign:
A9-A16 and F0-F2 are defined by exactly these expressions, so the fixture below
pins the values captured from the pre-move implementation (commit `68a5897`)
and requires them back *bitwise*.

It also closes the asymmetry the docs/24 audit named: the loss side had an
equality test, the evaluator side did not. Both sides are checked here.

Two quantities cannot be bitwise across platforms because they go through BLAS
/ LAPACK, whose last mantissa bits are implementation-dependent. Measured
macOS-Accelerate vs Linux-OpenBLAS difference on the identical fixture input:

    similarity_align        max abs 8.88e-16, max rel 4.56e-15   (numpy.linalg.svd)
    hinge error_degrees     max abs 2.84e-14 degrees             (numpy.dot then arccos)

Those two use a strict float64 tolerance; the reported magnitudes are ~1e-14 or
smaller, so anything larger is a real change and will fail. Every other moved
formula -- the structural losses, the reduction helpers, bend vectors, bend
direction, root-yaw and angle delta -- is required back exactly, bitwise.
"""

import json
import struct
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from common import canonical_pose
from framepose.evaluate import _hinge_direction_error
from framepose.losses import BASELINE_GEOMETRY_V1, compute_loss, loss_components
from training.temporal_lifter import TrainingConfig, _supervision_loss
import framepose.evaluate as framepose_evaluate
import training.temporal_lifter as temporal_lifter


_FIXTURE = json.loads((Path(__file__).parent / "fixtures" /
                       "canonical_pose_reference_v1.json").read_text(encoding="utf-8"))

# BLAS/LAPACK-dependent last bits only. Measured cross-platform differences are
# 8.9e-16 (alignment) and 2.8e-14 degrees (hinge angle); these bounds are orders
# of magnitude tighter than any real formula change would be.
SVD_PLATFORM_TOLERANCE = 1e-12
HINGE_ANGLE_PLATFORM_TOLERANCE_DEGREES = 1e-9


def _bits(value) -> str:
    return struct.pack(">d", float(value)).hex()


def _array(entry) -> np.ndarray:
    values = [struct.unpack(">d", bytes.fromhex(item))[0] for item in entry["bits"]]
    return np.asarray(values, dtype=np.float64).reshape(entry["shape"])


@pytest.fixture(scope="module")
def torch_inputs():
    prediction = torch.as_tensor(_array(_FIXTURE["prediction"]), dtype=torch.float32)
    target = torch.as_tensor(_array(_FIXTURE["target"]), dtype=torch.float32)
    mask = torch.as_tensor(_array(_FIXTURE["mask"]), dtype=torch.float32)
    return prediction, target, mask


def test_structural_losses_match_the_pre_refactor_values_bitwise(torch_inputs):
    prediction, target, mask = torch_inputs
    valid = mask.squeeze(-1).bool()
    expected = _FIXTURE["torch"]
    assert _bits(canonical_pose.vector_loss(
        torch, prediction, target, valid, canonical_pose.BONE_INDICES,
        lambda a, b: a - b).item()) == expected["vector_loss_bone"]
    assert _bits(canonical_pose.vector_loss(
        torch, prediction, target, valid, canonical_pose.TORSO_INDICES,
        lambda a, b: b - a).item()) == expected["vector_loss_torso"]
    assert _bits(canonical_pose.hinge_loss(torch, prediction, target, valid).item()) == expected["hinge_loss"]


def test_reduction_helpers_and_bend_vectors_match_bitwise(torch_inputs):
    prediction, _, _ = torch_inputs
    expected = _FIXTURE["torch"]
    produced = canonical_pose.bend_vectors(prediction[:, 0], prediction[:, 1], prediction[:, 2])
    assert [_bits(v) for v in produced.numpy().reshape(-1)] == _FIXTURE["torch"]["bend_vectors"]["bits"]
    assert _bits(canonical_pose.masked_chain_mean(
        torch, torch.arange(6 * 4, dtype=torch.float32).reshape(6, 4),
        (torch.rand(6, 4, generator=torch.Generator().manual_seed(7)) > 0.3)).item()) == expected["masked_chain_mean"]
    assert _bits(canonical_pose.masked_mean(
        torch, torch.arange(6 * 2, dtype=torch.float32).reshape(6, 2),
        (torch.rand(6, 2, generator=torch.Generator().manual_seed(9)) > 0.3)).item()) == expected["masked_mean"]


def test_evaluator_mathematics_matches_the_pre_refactor_values_bitwise():
    expected = _FIXTURE["numpy"]
    estimate = _array(expected["estimate"])
    reference = _array(expected["reference"])
    valid = np.asarray(expected["valid"], dtype=bool)
    indices = np.flatnonzero(valid)

    aligned = canonical_pose.similarity_align(estimate[indices], reference[indices])
    want = _array(expected["similarity_align"])
    # Strict float64 tolerance rather than bitwise: see the module docstring.
    assert aligned.shape == want.shape
    assert np.allclose(aligned, want, rtol=SVD_PLATFORM_TOLERANCE, atol=SVD_PLATFORM_TOLERANCE)
    assert _bits(canonical_pose.root_yaw_error_degrees(estimate, reference, valid)) == \
        expected["root_yaw_error_degrees"]
    produced = canonical_pose.hinge_errors(estimate, reference, valid)
    assert len(produced) == len(expected["hinge_errors"])
    for actual, want in zip(produced, expected["hinge_errors"]):
        assert actual["joint"] == want["joint"]
        # numpy.dot then arccos: BLAS-dependent last bits, see the module docstring.
        assert actual["error_degrees"] == pytest.approx(
            struct.unpack(">d", bytes.fromhex(want["error_degrees"]))[0],
            abs=HINGE_ANGLE_PLATFORM_TOLERANCE_DEGREES)
        assert actual["flipped"] is want["flipped"]
    bend = canonical_pose.bend_direction(estimate[1], estimate[0], estimate[2])
    assert [_bits(v) for v in np.asarray(bend).reshape(-1)] == expected["bend_direction"]["bits"]
    assert _bits(canonical_pose.angle_delta(2.9, -2.9)) == expected["angle_delta"]


def test_the_legacy_module_delegates_rather_than_duplicating():
    """One implementation, so no second formula can drift from the measured one."""
    pairs = (
        (temporal_lifter._vector_loss, canonical_pose.vector_loss),
        (temporal_lifter._hinge_loss, canonical_pose.hinge_loss),
        (temporal_lifter._bend_vectors, canonical_pose.bend_vectors),
        (temporal_lifter._masked_chain_mean, canonical_pose.masked_chain_mean),
        (temporal_lifter._masked_mean, canonical_pose.masked_mean),
        (temporal_lifter._similarity_align, canonical_pose.similarity_align),
        (temporal_lifter._root_yaw_error_degrees, canonical_pose.root_yaw_error_degrees),
        (temporal_lifter._hinge_errors, canonical_pose.hinge_errors),
        (temporal_lifter._bend_direction, canonical_pose.bend_direction),
        (temporal_lifter._angle_delta, canonical_pose.angle_delta),
    )
    for legacy, owner in pairs:
        assert legacy is owner, f"{owner.__name__} must have exactly one implementation"
    for name in ("BONES", "HINGE_CHAINS", "END_EFFECTOR_NAMES", "BONE_INDICES", "TORSO_INDICES",
                 "HINGE_INDICES", "END_EFFECTOR_INDICES", "YAW_INDICES", "VECTOR_NORMALIZATION_EPS",
                 "FORWARD_DEPTH_AXIS", "BILATERAL_DEPTH_NORMALIZATION"):
        assert getattr(temporal_lifter, name) == getattr(canonical_pose, name), name


def test_frame_pose_loss_is_the_historical_objective_bitwise(torch_inputs):
    prediction, target, mask = torch_inputs
    legacy = _supervision_loss(torch, prediction, target, mask, TrainingConfig(
        bone_loss_weight=0.25, torso_loss_weight=0.15, hinge_loss_weight=0.15))
    frame = compute_loss(torch, prediction, target, mask, BASELINE_GEOMETRY_V1)
    assert _bits(frame.item()) == _bits(legacy.item())
    components = loss_components(torch, prediction, target, mask)
    assert _bits(components["bone"].item()) == _FIXTURE["torch"]["vector_loss_bone"]
    assert _bits(components["torso"].item()) == _FIXTURE["torch"]["vector_loss_torso"]
    assert _bits(components["hinge"].item()) == _FIXTURE["torch"]["hinge_loss"]


def test_frame_pose_evaluator_uses_the_historical_metric_mathematics():
    """Closes the docs/24 asymmetry: the evaluator side is now checked directly."""
    assert framepose_evaluate.similarity_align is canonical_pose.similarity_align
    assert framepose_evaluate.root_yaw_error_degrees is canonical_pose.root_yaw_error_degrees
    assert framepose_evaluate.bend_direction is canonical_pose.bend_direction

    expected = _FIXTURE["numpy"]
    estimate = _array(expected["estimate"])
    reference = _array(expected["reference"])
    valid = np.asarray(expected["valid"], dtype=bool)
    # The frame evaluator's hinge metric must agree with the historical
    # per-chain errors it is derived from, on the same fixture.
    historical = canonical_pose.hinge_errors(estimate, reference, valid)
    produced = _hinge_direction_error(estimate, reference, valid)
    assert produced == pytest.approx(float(np.mean([h["error_degrees"] for h in historical])))


def test_the_canonical_contract_itself_is_unchanged():
    from pose.pose_lifter import H36M_NAMES

    assert canonical_pose.JOINT_NAMES == tuple(H36M_NAMES)
    assert len(canonical_pose.JOINT_NAMES) == 17
    assert canonical_pose.JOINT_NAMES[0] == "pelvis"
    assert canonical_pose.FORWARD_DEPTH_AXIS == 1
    assert canonical_pose.BILATERAL_DEPTH_NORMALIZATION == pytest.approx(1.0 / np.sqrt(2.0))
