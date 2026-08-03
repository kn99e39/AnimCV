"""True pinhole reprojection audit when camera intrinsics are supplied."""

from __future__ import annotations

import math
import statistics
from typing import Any

from pose.camera_calibration import CameraCalibration, fit_pelvis_translation, project_camera_point
from pose.pose_lifter import LiftedPoseSequence
from pose.pose_types import PoseSequence


def audit_calibrated_reprojection(
    observations: PoseSequence,
    baseline: LiftedPoseSequence,
    reconstructed: LiftedPoseSequence,
    calibration: CameraCalibration,
    min_confidence: float = 0.3,
    max_median_worsening_ratio: float = 1.05,
) -> dict[str, Any]:
    if not (len(observations.frames) == len(baseline.frames) == len(reconstructed.frames)):
        raise ValueError("observation, baseline, and reconstructed frame counts must match")
    raw_errors, fixed_errors, fitted_frames = [], [], 0
    for observed, raw, fixed in zip(observations.frames, baseline.frames, reconstructed.frames):
        if observed.frame_index != raw.frame_index or raw.frame_index != fixed.frame_index:
            raise ValueError("frame indices must match")
        raw_samples = _samples(observed, raw, min_confidence)
        fixed_samples = _samples(observed, fixed, min_confidence)
        if len(raw_samples) < 4 or len(fixed_samples) < 4:
            continue
        raw_translation = fit_pelvis_translation(raw_samples, calibration)
        fixed_translation = fit_pelvis_translation(fixed_samples, calibration)
        raw_errors.extend(_project_errors(raw_samples, raw_translation, calibration))
        fixed_errors.extend(_project_errors(fixed_samples, fixed_translation, calibration))
        fitted_frames += 1
    if not raw_errors or not fixed_errors:
        raise ValueError("no frame had four trusted joints for calibrated reprojection")
    raw_stats, fixed_stats = _summary(raw_errors), _summary(fixed_errors)
    if raw_stats["median_pixels"] <= 1e-9:
        ratio = 1.0 if fixed_stats["median_pixels"] <= 1e-9 else float("inf")
    else:
        ratio = fixed_stats["median_pixels"] / raw_stats["median_pixels"]
    return {
        "method": "calibrated pinhole reprojection with fitted per-frame pelvis translation",
        "calibration": {"source": calibration.source, "rms_pixels": calibration.calibration_rms_pixels},
        "fitted_frame_count": fitted_frames,
        "trusted_joint_samples": len(raw_errors),
        "baseline": raw_stats,
        "reconstructed": fixed_stats,
        "median_error_ratio": ratio,
        "max_median_worsening_ratio": max_median_worsening_ratio,
        "passed": ratio <= max_median_worsening_ratio,
        "root_translation_note": "Fitted pelvis translations validate projection only; they are not accepted global root motion.",
    }


def _samples(observed, lifted_frame, min_confidence):
    result = []
    for name, point in lifted_frame.points.items():
        source_name = "neck" if name == "thorax" else name
        landmark = observed.landmarks.get(source_name)
        if landmark and landmark.visible and landmark.confidence >= min_confidence and point.observation_valid:
            result.append((point.position, (landmark.x, landmark.y)))
    return result


def _project_errors(samples, translation, calibration):
    return [math.dist(project_camera_point(tuple(a + b for a, b in zip(point, translation)), calibration), pixel)
            for point, pixel in samples]


def _summary(errors: list[float]) -> dict[str, float]:
    ordered = sorted(errors)
    return {"median_pixels": statistics.median(errors), "mean_pixels": statistics.fmean(errors),
            "p95_pixels": ordered[round(0.95 * (len(ordered) - 1))]}
