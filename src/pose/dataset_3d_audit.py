"""Dataset-neutral 3D pose metrics for supervised-lifter holdouts."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from pose.pose_lifter import LiftedPoseSequence
from pose.root_motion import RootMotionSequence


def audit_supervised_3d(predicted: LiftedPoseSequence, ground_truth: LiftedPoseSequence,
                        predicted_root: RootMotionSequence | None = None,
                        ground_truth_root: RootMotionSequence | None = None) -> dict[str, Any]:
    """Compare canonical root-relative sequences independently of source dataset."""
    truth = {frame.frame_index: frame for frame in ground_truth.frames}
    predicted_yaw = {frame.frame_index: frame.root_yaw_radians for frame in (predicted_root.frames if predicted_root else [])}
    truth_yaw = {frame.frame_index: frame.root_yaw_radians for frame in (ground_truth_root.frames if ground_truth_root else [])}
    raw_errors, aligned_errors, yaw_errors = [], [], []
    matched_frames, matched_joints = 0, 0
    for frame in predicted.frames:
        target = truth.get(frame.frame_index)
        if target is None:
            continue
        names = [name for name in sorted(set(frame.points) & set(target.points))
                 if frame.points[name].observation_valid and target.points[name].observation_valid]
        if len(names) < 3:
            continue
        estimate = np.asarray([frame.points[name].position for name in names], dtype=float)
        reference = np.asarray([target.points[name].position for name in names], dtype=float)
        raw_errors.extend(np.linalg.norm(estimate - reference, axis=1))
        aligned_errors.extend(np.linalg.norm(_similarity_align(estimate, reference) - reference, axis=1))
        if frame.frame_index in predicted_yaw and frame.frame_index in truth_yaw:
            yaw_errors.append(abs(_angle_delta(predicted_yaw[frame.frame_index], truth_yaw[frame.frame_index])) * 180 / math.pi)
        matched_frames += 1
        matched_joints += len(names)
    if not raw_errors:
        raise ValueError("no overlapping valid 3D joints")
    raw_mm, aligned_mm = np.asarray(raw_errors) * 1000, np.asarray(aligned_errors) * 1000
    return {"schema": "animcv_supervised_3d_audit_v1", "matched_frames": matched_frames,
            "matched_joints": matched_joints, "mpjpe_mm": float(raw_mm.mean()),
            "pa_mpjpe_mm": float(aligned_mm.mean()), "p95_joint_error_mm": float(np.quantile(raw_mm, .95)),
            "root_yaw_mae_degrees": float(np.mean(yaw_errors)) if yaw_errors else None,
            "root_yaw_p95_degrees": float(np.quantile(yaw_errors, .95)) if yaw_errors else None,
            "passed": False, "verdict": "informational: use a representative source-level holdout before applying gates"}


def _similarity_align(estimate: np.ndarray, target: np.ndarray) -> np.ndarray:
    mean_estimate, mean_target = estimate.mean(0), target.mean(0)
    centered_estimate, centered_target = estimate - mean_estimate, target - mean_target
    estimate_norm, target_norm = np.linalg.norm(centered_estimate), np.linalg.norm(centered_target)
    if estimate_norm <= 1e-12 or target_norm <= 1e-12:
        return estimate
    source, destination = centered_estimate / estimate_norm, centered_target / target_norm
    u, _, vt = np.linalg.svd(source.T @ destination)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        vt[-1] *= -1
        rotation = u @ vt
    return (source @ rotation) * target_norm + mean_target


def _angle_delta(a: float, b: float) -> float:
    return (a - b + math.pi) % (2 * math.pi) - math.pi
