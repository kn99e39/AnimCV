"""Analysis strata for frame-level evaluation.

Strata are derived from ground truth and from the 2D observation. They exist so
a frame-level result can be read as *which observation conditions explain the
difference*, not just as one pooled number.

Leakage contract: the numeric quantities below are computed for every sample,
but every quantile threshold that turns a quantity into a bucket is fitted on
the **train split only** and then applied unchanged to validation and test. No
test ground truth ever participates in a threshold, a model parameter, or a
candidate selection.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from framepose.contract import (
    BILATERAL_DEPTH_NORMALIZATION, FORWARD_DEPTH_AXIS, JOINT_INDEX,
)


# Fixed, physically meaningful angular boundaries: these describe camera-relative
# body facing and need no data-fitted threshold.
FACING_BOUNDARIES_DEGREES = (30.0, 60.0, 120.0)
YAW_BOUNDARY_DEGREES = 45.0

_QUANTILE_STRATA = ("confidence", "torso_scale", "forward_depth", "articulation")

_HINGE_CHAINS = (
    ("left_shoulder", "left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow", "right_wrist"),
    ("left_hip", "left_knee", "left_ankle"),
    ("right_hip", "right_knee", "right_ankle"),
)
_HINGE_INDICES = tuple(tuple(JOINT_INDEX[name] for name in chain) for chain in _HINGE_CHAINS)

_PELVIS = JOINT_INDEX["pelvis"]
_THORAX = JOINT_INDEX["thorax"]
_LEFT_SHOULDER = JOINT_INDEX["left_shoulder"]
_RIGHT_SHOULDER = JOINT_INDEX["right_shoulder"]
_LEFT_HIP = JOINT_INDEX["left_hip"]
_RIGHT_HIP = JOINT_INDEX["right_hip"]


def frame_quantities(input_2d: np.ndarray, input_valid: np.ndarray,
                     target_3d: np.ndarray, target_valid: np.ndarray,
                     image_size: tuple[int, int]) -> dict[str, float | int | None]:
    """Per-frame scalar quantities the strata buckets are built from.

    `input_2d` is `(17, 3)` normalized `(x/width, y/height, confidence)` exactly
    as `training.temporal_lifter.build_dataset` writes it.
    """
    quantities: dict[str, float | int | None] = {}

    confidences = input_2d[:, 2][input_valid]
    quantities["mean_confidence"] = float(confidences.mean()) if confidences.size else None
    quantities["invalid_joint_count"] = int((~input_valid).sum())
    quantities["invalid_target_count"] = int((~target_valid).sum())

    width, height = image_size
    torso = None
    if input_valid[[_LEFT_SHOULDER, _RIGHT_SHOULDER, _LEFT_HIP, _RIGHT_HIP]].all():
        pixels = input_2d[:, :2] * np.asarray([width, height], dtype=np.float64)
        shoulder = (pixels[_LEFT_SHOULDER] + pixels[_RIGHT_SHOULDER]) / 2.0
        hip = (pixels[_LEFT_HIP] + pixels[_RIGHT_HIP]) / 2.0
        torso = float(np.linalg.norm(shoulder - hip) / max(height, 1))
    quantities["projected_torso_fraction"] = torso

    quantities["facing_angle_degrees"] = _facing_angle_degrees(target_3d, target_valid)
    quantities["shoulder_forward_depth_m"] = _bilateral_forward_depth(
        target_3d, target_valid, _LEFT_SHOULDER, _RIGHT_SHOULDER)
    quantities["hip_forward_depth_m"] = _bilateral_forward_depth(
        target_3d, target_valid, _LEFT_HIP, _RIGHT_HIP)
    quantities["min_bend_angle_degrees"] = _min_bend_angle_degrees(target_3d, target_valid)
    return quantities


def fit_thresholds(quantities: list[dict[str, float | int | None]]) -> dict[str, dict[str, float]]:
    """Fit quantile boundaries on the train split only."""
    def _quantiles(key: str, probabilities: tuple[float, ...]) -> dict[str, float]:
        values = np.asarray([item[key] for item in quantities if item.get(key) is not None], dtype=np.float64)
        if values.size == 0:
            return {}
        return {f"q{int(round(probability * 100)):02d}": float(np.quantile(values, probability))
                for probability in probabilities}

    return {
        "mean_confidence": _quantiles("mean_confidence", (0.25,)),
        "projected_torso_fraction": _quantiles("projected_torso_fraction", (0.25, 0.75)),
        "abs_shoulder_forward_depth_m": _absolute_quantiles(quantities, "shoulder_forward_depth_m", (0.25, 0.75)),
        "min_bend_angle_degrees": _quantiles("min_bend_angle_degrees", (0.10,)),
    }


def _absolute_quantiles(quantities: list[dict[str, float | int | None]], key: str,
                        probabilities: tuple[float, ...]) -> dict[str, float]:
    values = np.asarray([abs(item[key]) for item in quantities if item.get(key) is not None], dtype=np.float64)
    if values.size == 0:
        return {}
    return {f"q{int(round(probability * 100)):02d}": float(np.quantile(values, probability))
            for probability in probabilities}


def assign_strata(quantities: dict[str, float | int | None],
                  thresholds: dict[str, dict[str, float]]) -> dict[str, str]:
    """Turn per-frame quantities into named strata using fitted thresholds."""
    strata: dict[str, str] = {}

    angle = quantities.get("facing_angle_degrees")
    if angle is None:
        strata["facing"] = "unknown"
        strata["yaw"] = "unknown"
    else:
        magnitude = abs(angle)
        near, side, back = FACING_BOUNDARIES_DEGREES
        strata["facing"] = ("frontal" if magnitude <= near else
                            "near_frontal" if magnitude <= side else
                            "profile" if magnitude <= back else "back_facing")
        strata["yaw"] = "low_yaw" if magnitude <= YAW_BOUNDARY_DEGREES else "high_yaw"

    strata["visibility"] = ("fully_visible" if not quantities.get("invalid_joint_count")
                            else "partially_visible")

    strata["confidence"] = _bucket(
        quantities.get("mean_confidence"), thresholds.get("mean_confidence", {}),
        [("q25", "low_confidence")], "normal_confidence")

    strata["torso_scale"] = _bucket(
        quantities.get("projected_torso_fraction"), thresholds.get("projected_torso_fraction", {}),
        [("q25", "small_projected_torso"), ("q75", "medium_projected_torso")], "large_projected_torso")

    depth = quantities.get("shoulder_forward_depth_m")
    strata["forward_depth"] = _bucket(
        None if depth is None else abs(depth), thresholds.get("abs_shoulder_forward_depth_m", {}),
        [("q25", "near_zero_forward_depth"), ("q75", "medium_forward_depth")], "large_forward_depth")

    strata["articulation"] = _bucket(
        quantities.get("min_bend_angle_degrees"), thresholds.get("min_bend_angle_degrees", {}),
        [("q10", "rare_articulation")], "typical_articulation")
    return strata


def _bucket(value: float | None, boundaries: dict[str, float],
            ladder: list[tuple[str, str]], default: str) -> str:
    if value is None or not boundaries:
        return "unknown"
    for key, name in ladder:
        if key not in boundaries:
            return "unknown"
        if value <= boundaries[key]:
            return name
    return default


def _facing_angle_degrees(target_3d: np.ndarray, target_valid: np.ndarray) -> float | None:
    """Camera-relative body facing: 0 deg faces the camera, +-180 deg faces away.

    Body right `r = P[right_shoulder] - P[left_shoulder]`, body up
    `u = P[thorax] - P[pelvis]`, body forward `f = u x r`.  With the canonical
    camera frame (+X right, +Y forward/away, +Z up) a performer facing away has
    `f = +Y`, so `atan2(f_x, -f_y)` is 0 when facing the camera.
    """
    required = (_PELVIS, _THORAX, _LEFT_SHOULDER, _RIGHT_SHOULDER)
    if not target_valid[list(required)].all():
        return None
    right = target_3d[_RIGHT_SHOULDER] - target_3d[_LEFT_SHOULDER]
    up = target_3d[_THORAX] - target_3d[_PELVIS]
    forward = np.cross(up, right)
    norm = float(np.linalg.norm(forward))
    if norm < 1e-8:
        return None
    forward = forward / norm
    return float(math.degrees(math.atan2(float(forward[0]), float(-forward[1]))))


def _bilateral_forward_depth(target_3d: np.ndarray, target_valid: np.ndarray,
                             left: int, right: int) -> float | None:
    """`D = (y_right - y_left) / sqrt(2)` on the forward axis (docs/18, docs/21)."""
    if not (target_valid[left] and target_valid[right]):
        return None
    difference = target_3d[right, FORWARD_DEPTH_AXIS] - target_3d[left, FORWARD_DEPTH_AXIS]
    return float(difference * BILATERAL_DEPTH_NORMALIZATION)


def _min_bend_angle_degrees(target_3d: np.ndarray, target_valid: np.ndarray) -> float | None:
    angles = []
    for proximal, joint, distal in _HINGE_INDICES:
        if not (target_valid[proximal] and target_valid[joint] and target_valid[distal]):
            continue
        first = target_3d[proximal] - target_3d[joint]
        second = target_3d[distal] - target_3d[joint]
        denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
        if denominator < 1e-8:
            continue
        cosine = float(np.dot(first, second) / denominator)
        angles.append(math.degrees(math.acos(max(-1.0, min(1.0, cosine)))))
    return float(min(angles)) if angles else None


def stratum_names() -> tuple[str, ...]:
    return ("facing", "yaw", "visibility", "confidence", "torso_scale", "forward_depth", "articulation")


def summarize(samples_strata: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    summary: dict[str, dict[str, int]] = {name: {} for name in stratum_names()}
    for strata in samples_strata:
        for name in stratum_names():
            value = strata.get(name, "unknown")
            summary[name][value] = summary[name].get(value, 0) + 1
    return {name: dict(sorted(counts.items())) for name, counts in summary.items()}
