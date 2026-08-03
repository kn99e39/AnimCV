"""Conservative focal-length self-calibration for a static monocular camera.

This is a fallback for clips without a checkerboard calibration.  It assumes a
fixed camera, zero lens distortion, and principal point at image centre.  The
returned uncertainty gate is intentionally strict: an estimate is useful only
when the 3D/2D evidence constrains focal length rather than merely fitting a
per-frame body translation.
"""

from __future__ import annotations

import math
from typing import Any

from pose.camera_calibration import CameraCalibration, fit_pelvis_translation, project_camera_point
from pose.pose_lifter import LiftedPoseSequence
from pose.pose_types import PoseSequence


def estimate_static_camera_calibration(
    observations: PoseSequence,
    lifted: LiftedPoseSequence,
    image_width: int,
    image_height: int,
    min_confidence: float = 0.3,
    max_focal_uncertainty_ratio: float = 1.5,
) -> tuple[CameraCalibration, dict[str, Any]]:
    if len(observations.frames) != len(lifted.frames):
        raise ValueError("observation and lifted frame counts must match")
    samples_by_frame = [_samples(observed, frame, min_confidence)
                        for observed, frame in zip(observations.frames, lifted.frames)]
    samples_by_frame = [samples for samples in samples_by_frame if len(samples) >= 4]
    if len(samples_by_frame) < 10:
        raise ValueError("static camera self-calibration needs ten frames with four trusted joints")

    # Calibration quality is global, so a uniformly spaced sample is enough
    # and prevents a 120+ frame clip from turning this conservative fallback
    # into an expensive offline bundle adjustment.
    stride = max(1, math.ceil(len(samples_by_frame) / 30))
    samples_by_frame = samples_by_frame[::stride]
    centre_x, centre_y = image_width / 2.0, image_height / 2.0
    minimum, maximum = max(image_width, image_height) * 0.5, max(image_width, image_height) * 3.0

    def score(focal: float) -> float:
        camera = CameraCalibration(image_width, image_height, focal, focal, centre_x, centre_y,
                                   source="auto_pose_self_calibration")
        squared_errors = []
        for samples in samples_by_frame:
            translation = fit_pelvis_translation(samples, camera, iterations=30)
            squared_errors.extend(
                math.dist(project_camera_point(tuple(a + b for a, b in zip(point, translation)), camera), pixel) ** 2
                for point, pixel in samples
            )
        return math.sqrt(sum(squared_errors) / len(squared_errors))

    focal, rms = _minimise_positive(score, minimum, maximum)
    # A near-minimum interval exposes practical ambiguity without claiming a
    # statistical posterior. One additional pixel RMS is deliberately easier
    # to interpret than an arbitrary relative loss near zero.
    scan = [minimum * (maximum / minimum) ** (index / 40) for index in range(41)]
    plausible = [value for value in scan if score(value) <= rms + 1.0]
    interval = (min(plausible), max(plausible)) if plausible else (focal, focal)
    uncertainty_ratio = interval[1] / interval[0]
    at_search_edge = focal <= minimum * 1.02 or focal >= maximum / 1.02
    accepted = uncertainty_ratio <= max_focal_uncertainty_ratio and not at_search_edge
    calibration = CameraCalibration(
        image_width, image_height, focal, focal, centre_x, centre_y,
        source="auto_pose_self_calibration", calibration_rms_pixels=rms,
    )
    report = {
        "method": "static-camera pose self-calibration; centred principal point and zero distortion assumed",
        "fitted_frame_count": len(samples_by_frame),
        "focal_pixels": focal,
        "reprojection_rms_pixels": rms,
        "one_pixel_focal_interval": list(interval),
        "focal_uncertainty_ratio": uncertainty_ratio,
        "max_focal_uncertainty_ratio": max_focal_uncertainty_ratio,
        "search_hit_boundary": at_search_edge,
        "accepted_for_limited_calibrated_audit": accepted,
        "limitations": [
            "Not a checkerboard calibration and not evidence of lens distortion.",
            "Valid only for a static camera; moving cameras require camera tracking or bundle adjustment.",
            "Fitted pelvis translations are not global root motion.",
        ],
    }
    return calibration, report


def _samples(observed, lifted_frame, min_confidence):
    if observed.frame_index != lifted_frame.frame_index:
        raise ValueError("observation and lifted frame indices must match")
    result = []
    for name, point in lifted_frame.points.items():
        source_name = "neck" if name == "thorax" else name
        landmark = observed.landmarks.get(source_name)
        if landmark and landmark.visible and landmark.confidence >= min_confidence and point.observation_valid:
            result.append((point.position, (landmark.x, landmark.y)))
    return result


def _minimise_positive(function, lower: float, upper: float) -> tuple[float, float]:
    ratio = (math.sqrt(5.0) - 1.0) / 2.0
    left, right = lower, upper
    c, d = right - ratio * (right - left), left + ratio * (right - left)
    score_c, score_d = function(c), function(d)
    for _ in range(30):
        if score_c <= score_d:
            right, d, score_d = d, c, score_c
            c = right - ratio * (right - left)
            score_c = function(c)
        else:
            left, c, score_c = c, d, score_d
            d = left + ratio * (right - left)
            score_d = function(d)
    focal = (left + right) / 2.0
    return focal, function(focal)
