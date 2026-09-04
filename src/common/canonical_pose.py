"""Canonical pose mathematics — neutral, shared, owned by neither consumer.

Both the Frame Pose Core (`src/framepose/`, Architecture_v3 Layer A) and the
Legacy Temporal Pose Baseline (`src/training/temporal_lifter.py`) need the same
skeleton constants and the same geometric quantities. Before this module they
lived inside the temporal lifter, which made the *legacy* module the
implementation owner of the *new* core's mathematics — the wrong direction.

    canonical pose mathematics  (this module)
           /                 \\
    Frame Pose Core     Legacy Temporal Lifter

This module is deliberately ignorant of temporal windows, training configs,
FramePose, experiment identifiers and dataset sources. It is pure geometry over
the canonical 17-joint contract in `pose.pose_lifter.H36M_NAMES`, in AnimCV's
canonical camera frame (+X right, +Y forward/depth, +Z up).

**Historical contract.** Every formula here was moved verbatim from
`training.temporal_lifter`. Reductions, masks, epsilons and dtype behaviour are
unchanged, and `tests/test_canonical_pose_parity.py` pins them bitwise against
values captured before the move. Do not "clean up" a formula in this file: A9
through A16 and F0 through F2 are all defined by exactly these expressions.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from pose.pose_lifter import H36M_NAMES


JOINT_NAMES: tuple[str, ...] = tuple(H36M_NAMES)
JOINT_INDEX = {name: index for index, name in enumerate(JOINT_NAMES)}

# AnimCV's canonical camera frame; the forward/depth column the bilateral
# forward-depth quantity of docs/18 and docs/21 is read on.
FORWARD_DEPTH_AXIS = 1
BILATERAL_DEPTH_NORMALIZATION = 1.0 / math.sqrt(2.0)
VECTOR_NORMALIZATION_EPS = 1e-6

# Parent-to-child segments in the canonical 17-joint contract. These constrain
# shape and orientation without tying any model to an FBX rig.
BONES = (
    ("pelvis", "left_hip"), ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
    ("pelvis", "right_hip"), ("right_hip", "right_knee"), ("right_knee", "right_ankle"),
    ("pelvis", "spine"), ("spine", "thorax"), ("thorax", "neck"), ("neck", "head"),
    ("thorax", "left_shoulder"), ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
    ("thorax", "right_shoulder"), ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
)
HINGE_CHAINS = (
    ("left_shoulder", "left_elbow", "left_wrist"), ("right_shoulder", "right_elbow", "right_wrist"),
    ("left_hip", "left_knee", "left_ankle"), ("right_hip", "right_knee", "right_ankle"),
)
TORSO_PAIRS = (("left_shoulder", "right_shoulder"), ("left_hip", "right_hip"))
YAW_PAIRS = TORSO_PAIRS
# Limb-chain terminals; the joints scripts/*constraint_target* already call
# "end_effector" for IK retargeting.
END_EFFECTOR_NAMES = ("left_wrist", "right_wrist", "left_ankle", "right_ankle")

# Resolved once: these run on every CUDA batch, so repeated string lookups and
# scalar-controlled Python loops are an avoidable throughput cost.
BONE_INDICES = tuple((JOINT_INDEX[first], JOINT_INDEX[second]) for first, second in BONES)
TORSO_INDICES = tuple((JOINT_INDEX[first], JOINT_INDEX[second]) for first, second in TORSO_PAIRS)
HINGE_INDICES = tuple(tuple(JOINT_INDEX[name] for name in chain) for chain in HINGE_CHAINS)
END_EFFECTOR_INDICES = tuple(JOINT_INDEX[name] for name in END_EFFECTOR_NAMES)
YAW_INDICES = tuple((JOINT_INDEX[left], JOINT_INDEX[right]) for left, right in YAW_PAIRS)

# The metric-side hinge chains are ordered (joint, proximal, distal), which is
# the argument order `bend_direction` takes.
METRIC_HINGE_CHAINS = (
    ("left_elbow", "left_shoulder", "left_wrist"), ("right_elbow", "right_shoulder", "right_wrist"),
    ("left_knee", "left_hip", "left_ankle"), ("right_knee", "right_hip", "right_ankle"),
)


# --------------------------------------------------------------- torch side --
# `torch` is passed in rather than imported so this module stays importable in
# a geometry-only runtime, matching the lazy-import convention used throughout.

def masked_mean(torch, values, valid):
    return (values * valid).sum() / valid.sum().clamp_min(1)


def masked_chain_mean(torch, errors, valid):
    """Equal-per-chain reduction without synchronizing on CUDA scalars."""
    counts = valid.sum(dim=0)
    per_chain = (errors * valid).sum(dim=0) / counts.clamp_min(1)
    return masked_mean(torch, per_chain, counts > 0)


def bend_vectors(proximal, joint, distal):
    axis = distal - proximal
    projection = (joint - proximal).mul(axis).sum(-1, keepdim=True) / axis.square().sum(-1, keepdim=True).clamp_min(1e-8)
    return joint - (proximal + projection * axis)


def vector_loss(torch, prediction, target, valid, pairs, vector):
    """Average each segment equally without CUDA scalar control flow."""
    # Retain the historical name-pair contract for callers/tests. The training
    # path supplies pre-resolved integer pairs, so this compatibility
    # conversion never occurs inside the performance-critical loop.
    if isinstance(pairs[0][0], str):
        pairs = tuple((JOINT_NAMES.index(first), JOINT_NAMES.index(second)) for first, second in pairs)
    first, second = zip(*pairs)
    pair_valid = valid[:, first] & valid[:, second]
    predicted_vectors = vector(prediction[:, first], prediction[:, second])
    target_vectors = vector(target[:, first], target[:, second])
    errors = torch.nn.functional.smooth_l1_loss(predicted_vectors, target_vectors, reduction="none").mean(dim=-1)
    return masked_chain_mean(torch, errors, pair_valid)


def hinge_loss(torch, prediction, target, valid):
    proximal, joint, distal = zip(*HINGE_INDICES)
    chain_valid = valid[:, proximal] & valid[:, joint] & valid[:, distal]
    predicted_bends = bend_vectors(prediction[:, proximal], prediction[:, joint], prediction[:, distal])
    target_bends = bend_vectors(target[:, proximal], target[:, joint], target[:, distal])
    errors = torch.nn.functional.smooth_l1_loss(predicted_bends, target_bends, reduction="none").mean(dim=-1)
    return masked_chain_mean(torch, errors, chain_valid)


# --------------------------------------------------------------- numpy side --

def angle_delta(a: float, b: float) -> float:
    return (a - b + np.pi) % (2 * np.pi) - np.pi


def similarity_align(estimate: np.ndarray, reference: np.ndarray) -> np.ndarray:
    mean_estimate, mean_reference = estimate.mean(0), reference.mean(0)
    centered_estimate, centered_reference = estimate - mean_estimate, reference - mean_reference
    estimate_norm, reference_norm = np.linalg.norm(centered_estimate), np.linalg.norm(centered_reference)
    if estimate_norm <= 1e-12 or reference_norm <= 1e-12:
        return estimate
    source, destination = centered_estimate / estimate_norm, centered_reference / reference_norm
    u, _, vt = np.linalg.svd(source.T @ destination)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        vt[-1] *= -1
        rotation = u @ vt
    return (source @ rotation) * reference_norm + mean_reference


def root_yaw_error_degrees(estimate: np.ndarray, reference: np.ndarray, valid: np.ndarray) -> float | None:
    angles = []
    for left, right in YAW_PAIRS:
        left_index, right_index = JOINT_NAMES.index(left), JOINT_NAMES.index(right)
        if not (valid[left_index] and valid[right_index]):
            continue
        predicted_axis = estimate[right_index, :2] - estimate[left_index, :2]
        target_axis = reference[right_index, :2] - reference[left_index, :2]
        if min(np.linalg.norm(predicted_axis), np.linalg.norm(target_axis)) <= 1e-6:
            continue
        angles.append(abs(angle_delta(np.arctan2(predicted_axis[1], predicted_axis[0]),
                                      np.arctan2(target_axis[1], target_axis[0]))) * 180.0 / np.pi)
    return float(np.mean(angles)) if angles else None


def bend_direction(joint: np.ndarray, proximal: np.ndarray, distal: np.ndarray) -> np.ndarray | None:
    axis = distal - proximal
    axis_squared = float(np.dot(axis, axis))
    if axis_squared <= 1e-12:
        return None
    bend = joint - (proximal + axis * (np.dot(joint - proximal, axis) / axis_squared))
    magnitude = float(np.linalg.norm(bend))
    return bend / magnitude if magnitude > 1e-6 else None


def hinge_errors(estimate: np.ndarray, reference: np.ndarray, valid: np.ndarray) -> list[dict[str, Any]]:
    output = []
    for joint, proximal, distal in METRIC_HINGE_CHAINS:
        indexes = [JOINT_NAMES.index(name) for name in (joint, proximal, distal)]
        if not valid[indexes].all():
            continue
        predicted = bend_direction(estimate[indexes[0]], estimate[indexes[1]], estimate[indexes[2]])
        target = bend_direction(reference[indexes[0]], reference[indexes[1]], reference[indexes[2]])
        if predicted is None or target is None:
            continue
        cosine = float(np.clip(np.dot(predicted, target), -1.0, 1.0))
        output.append({"joint": joint, "error_degrees": float(np.degrees(np.arccos(cosine))), "flipped": cosine < 0})
    return output
