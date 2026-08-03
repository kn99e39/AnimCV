"""A scale-invariant 2D preservation gate for kinematic reconstruction.

The input video has no camera intrinsics, so this is explicitly a
*weak-perspective proxy*, not calibrated reprojection.  It fits independent
per-frame image scale and translation to each candidate 3D pose, then checks
that reconstruction did not materially move trusted joints away from the
observed 2D pose.
"""

from __future__ import annotations

import math
import statistics
from typing import Any

from pose.pose_lifter import LiftedPoseSequence
from pose.pose_types import PoseSequence


def audit_weak_perspective_reprojection(
    observations: PoseSequence,
    baseline: LiftedPoseSequence,
    reconstructed: LiftedPoseSequence,
    min_confidence: float = 0.3,
    max_median_worsening_ratio: float = 1.05,
) -> dict[str, Any]:
    """Compare raw and reconstructed 3D targets against trusted 2D input."""
    _validate_sequences(observations, baseline, reconstructed)
    baseline_errors = _errors(observations, baseline, min_confidence)
    reconstructed_errors = _errors(observations, reconstructed, min_confidence)
    if not baseline_errors or not reconstructed_errors:
        raise ValueError("need at least three trusted joints with non-zero 3D span")
    baseline_stats = _summary(baseline_errors)
    reconstructed_stats = _summary(reconstructed_errors)
    if baseline_stats["median_pixels"] <= 1e-9:
        ratio = 1.0 if reconstructed_stats["median_pixels"] <= 1e-9 else float("inf")
    else:
        ratio = reconstructed_stats["median_pixels"] / baseline_stats["median_pixels"]
    return {
        "method": "per-frame weak-perspective affine proxy (not calibrated camera reprojection)",
        "trusted_joint_samples": len(baseline_errors),
        "baseline": baseline_stats,
        "reconstructed": reconstructed_stats,
        "median_error_ratio": ratio,
        "max_median_worsening_ratio": max_median_worsening_ratio,
        "passed": ratio <= max_median_worsening_ratio,
    }


def _validate_sequences(*sequences) -> None:
    counts = {len(sequence.frames) for sequence in sequences}
    if len(counts) != 1:
        raise ValueError("observation, baseline, and reconstructed frame counts must match")


def _errors(observations: PoseSequence, lifted: LiftedPoseSequence, min_confidence: float) -> list[tuple[float, float]]:
    errors: list[tuple[float, float]] = []
    for observed_frame, lifted_frame in zip(observations.frames, lifted.frames):
        if observed_frame.frame_index != lifted_frame.frame_index:
            raise ValueError("observation and lifted frame indices must match")
        samples = []
        for name, point in lifted_frame.points.items():
            source_name = "neck" if name == "thorax" else name
            landmark = observed_frame.landmarks.get(source_name)
            if landmark is None or not landmark.visible or landmark.confidence < min_confidence:
                continue
            if not point.observation_valid:
                continue
            samples.append((point.position[0], point.position[2], landmark.x, landmark.y))
        if len(samples) < 3:
            continue
        scale_x, offset_x = _fit([item[0] for item in samples], [item[2] for item in samples])
        scale_y, offset_y = _fit([item[1] for item in samples], [item[3] for item in samples])
        if scale_x is None or scale_y is None:
            continue
        xs, ys = [item[2] for item in samples], [item[3] for item in samples]
        diagonal = math.hypot(max(xs) - min(xs), max(ys) - min(ys))
        if diagonal <= 1e-6:
            continue
        for x3d, z3d, x2d, y2d in samples:
            pixel_error = math.hypot(scale_x * x3d + offset_x - x2d, scale_y * z3d + offset_y - y2d)
            errors.append((pixel_error, pixel_error / diagonal))
    return errors


def _fit(predictor: list[float], target: list[float]) -> tuple[float | None, float | None]:
    mean_predictor, mean_target = statistics.fmean(predictor), statistics.fmean(target)
    denominator = sum((value - mean_predictor) ** 2 for value in predictor)
    if denominator <= 1e-9:
        return None, None
    slope = sum((x - mean_predictor) * (y - mean_target) for x, y in zip(predictor, target)) / denominator
    return slope, mean_target - slope * mean_predictor


def _summary(errors: list[tuple[float, float]]) -> dict[str, float]:
    pixels, normalized = zip(*errors)
    ordered = sorted(pixels)
    return {
        "median_pixels": statistics.median(pixels),
        "mean_pixels": statistics.fmean(pixels),
        "p95_pixels": ordered[round(0.95 * (len(ordered) - 1))],
        "median_body_diagonal_ratio": statistics.median(normalized),
    }
