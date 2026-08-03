"""Metric 3D/root-yaw audit against imported MPI-INF-3DHP ground truth."""

from __future__ import annotations

import math
import numpy as np

from pose.pose_lifter import LiftedPoseSequence
from pose.root_motion import RootMotionSequence


def audit_mpi3dhp_3d(
    predicted: LiftedPoseSequence, ground_truth: LiftedPoseSequence,
    predicted_root: RootMotionSequence, ground_truth_root: RootMotionSequence,
) -> dict:
    """Report root-relative MPJPE, PA-MPJPE, and absolute bilateral-axis yaw error."""
    truth_by_index = {frame.frame_index: frame for frame in ground_truth.frames}
    predicted_yaw = {frame.frame_index: frame.root_yaw_radians for frame in predicted_root.frames}
    truth_yaw = {frame.frame_index: frame.root_yaw_radians for frame in ground_truth_root.frames}
    mpjpe, pa_mpjpe, yaw_errors = [], [], []
    matched_frames = 0
    for frame in predicted.frames:
        truth = truth_by_index.get(frame.frame_index)
        if truth is None:
            continue
        names = sorted(set(frame.points) & set(truth.points))
        if len(names) < 3:
            continue
        estimate = np.asarray([frame.points[name].position for name in names], dtype=float)
        target = np.asarray([truth.points[name].position for name in names], dtype=float)
        mpjpe.extend(np.linalg.norm(estimate - target, axis=1).tolist())
        pa_mpjpe.extend(np.linalg.norm(_similarity_align(estimate, target) - target, axis=1).tolist())
        if frame.frame_index in predicted_yaw and frame.frame_index in truth_yaw:
            yaw_errors.append(abs(_angle_delta(predicted_yaw[frame.frame_index], truth_yaw[frame.frame_index])) * 180.0 / math.pi)
        matched_frames += 1
    if not mpjpe or not yaw_errors:
        raise ValueError("insufficient overlapping MPI-INF-3DHP 3D/root-yaw frames")
    mpjpe_mm = [value * 1000 for value in mpjpe]
    pa_mm = [value * 1000 for value in pa_mpjpe]
    return {
        "schema": "animcv_mpi3dhp_3d_audit_v1",
        "matched_frames": matched_frames,
        "matched_joints": len(mpjpe),
        "root_relative_mpjpe_mm": sum(mpjpe_mm) / len(mpjpe_mm),
        "pa_mpjpe_mm": sum(pa_mm) / len(pa_mm),
        "root_yaw_mae_degrees": sum(yaw_errors) / len(yaw_errors),
        "root_yaw_p95_degrees": _percentile(yaw_errors, 95),
        "passed": False,
        "verdict": "informational_only: thresholds require a representative multi-action benchmark",
    }


def _similarity_align(estimate: np.ndarray, target: np.ndarray) -> np.ndarray:
    mean_estimate, mean_target = estimate.mean(axis=0), target.mean(axis=0)
    centered_estimate, centered_target = estimate - mean_estimate, target - mean_target
    norm_estimate = np.linalg.norm(centered_estimate)
    norm_target = np.linalg.norm(centered_target)
    if norm_estimate <= 1e-12 or norm_target <= 1e-12:
        return estimate
    source, destination = centered_estimate / norm_estimate, centered_target / norm_target
    u, _, vt = np.linalg.svd(source.T @ destination)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        vt[-1] *= -1
        rotation = u @ vt
    return (source @ rotation) * norm_target + mean_target


def _angle_delta(a: float, b: float) -> float:
    return (a - b + math.pi) % (2 * math.pi) - math.pi


def _percentile(values: list[float], percentile: float) -> float:
    return float(np.percentile(np.asarray(values), percentile))
