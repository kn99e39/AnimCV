"""Frame-level evaluation.

Every frame is an independently evaluable pose-reconstruction sample. For a
sequence of N frames the report carries N addressable results, so the questions
"which exact frames improved", "which regressed", "which geometry component
moved" and "which observation conditions explain it" are answerable without
re-running anything.

Temporal smoothness is deliberately absent: it is not a promotion criterion for
Layer A.
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

import numpy as np

from framepose.contract import (
    BILATERAL_DEPTH_NORMALIZATION, FORWARD_DEPTH_AXIS, FrameBank, JOINT_INDEX, JOINT_NAMES,
)
from framepose.observations import summarize as summarize_observations
from framepose.strata import stratum_names
# The historical similarity alignment and root-yaw definitions are reused
# verbatim so a frame-first number is comparable with the Legacy Temporal Pose
# Baseline's reports rather than being a differently defined metric.
from training.temporal_lifter import _root_yaw_error_degrees, _similarity_align, _bend_direction


EVALUATION_SCHEMA = "animcv_frame_pose_evaluation_v1"

# Below this the ground-truth bilateral forward depth is at the noise floor and
# its sign carries no information; reported separately rather than dropped.
STABLE_FORWARD_DEPTH_M = 0.01

_LEFT_SHOULDER = JOINT_INDEX["left_shoulder"]
_RIGHT_SHOULDER = JOINT_INDEX["right_shoulder"]
_LEFT_HIP = JOINT_INDEX["left_hip"]
_RIGHT_HIP = JOINT_INDEX["right_hip"]

_HINGE_CHAINS = (
    ("left_shoulder", "left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow", "right_wrist"),
    ("left_hip", "left_knee", "left_ankle"),
    ("right_hip", "right_knee", "right_ankle"),
)


def evaluate_predictions(bank: FrameBank, positions: Sequence[int], prediction: np.ndarray, *,
                         candidate: str, worst_frame_count: int = 25) -> dict[str, Any]:
    """Per-frame metrics plus source/stratum aggregation for one candidate."""
    positions = np.asarray(positions, dtype=np.int64)
    prediction = np.asarray(prediction, dtype=np.float64)
    if prediction.shape != (len(positions), len(JOINT_NAMES), 3):
        raise ValueError(f"prediction shape {prediction.shape} does not match {len(positions)} frames")

    targets = bank.arrays["target_3d"][positions].astype(np.float64)
    valid = bank.arrays["target_valid"][positions]

    frames: list[dict[str, Any]] = []
    joint_error_sum = np.zeros(len(JOINT_NAMES))
    joint_error_count = np.zeros(len(JOINT_NAMES))
    for order, position in enumerate(positions):
        sample = bank.samples[int(position)]
        estimate, reference, frame_valid = prediction[order], targets[order], valid[order]
        indices = np.flatnonzero(frame_valid)
        errors = np.linalg.norm(estimate[indices] - reference[indices], axis=1) * 1000.0 if len(indices) else np.asarray([])
        joint_error_sum[indices] += errors
        joint_error_count[indices] += 1
        aligned = None
        if len(indices) >= 3:
            aligned = float(np.linalg.norm(
                _similarity_align(estimate[indices], reference[indices]) - reference[indices], axis=1).mean() * 1000.0)
        record = {
            "sample_id": sample.sample_id,
            "sequence_id": sample.sequence_id,
            "frame_index": sample.frame_index,
            "source": sample.source,
            "split": sample.split,
            "observation_backend": sample.observation.backend,
            "valid_joint_count": int(len(indices)),
            "mpjpe_mm": float(errors.mean()) if len(errors) else None,
            "max_joint_error_mm": float(errors.max()) if len(errors) else None,
            "pa_mpjpe_mm": aligned,
            "root_yaw_error_degrees": _root_yaw_error_degrees(estimate, reference, frame_valid),
            "hinge_direction_mae_degrees": _hinge_direction_error(estimate, reference, frame_valid),
            "strata": {name: sample.strata.get(name, "unknown") for name in stratum_names()},
        }
        record.update(_forward_depth_metrics(estimate, reference, frame_valid, "shoulder",
                                             _LEFT_SHOULDER, _RIGHT_SHOULDER))
        record.update(_forward_depth_metrics(estimate, reference, frame_valid, "hip",
                                             _LEFT_HIP, _RIGHT_HIP))
        frames.append(record)

    observations = [bank.samples[int(position)].observation for position in positions]
    report = {
        "schema": EVALUATION_SCHEMA,
        "candidate": candidate,
        # Oracle-geometry and real-observation numbers are not comparable; the
        # label travels with the report so a later reader cannot conflate them.
        "observation_regime": sorted({item.regime for item in observations}),
        "observation": summarize_observations(observations),
        "frame_count": len(frames),
        "aggregate": aggregate(frames),
        "per_joint_mean_error_mm": {
            name: (float(joint_error_sum[index] / joint_error_count[index]) if joint_error_count[index] else None)
            for index, name in enumerate(JOINT_NAMES)
        },
        "per_source": _grouped(frames, lambda record: record["source"]),
        "per_sequence": _grouped(frames, lambda record: record["sequence_id"]),
        "per_stratum": {
            name: _grouped(frames, lambda record, key=name: record["strata"].get(key, "unknown"))
            for name in stratum_names()
        },
        "worst_frames": _worst(frames, worst_frame_count),
        "frames": frames,
    }
    return report


def aggregate(frames: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Mean / median / P90 / P95 for every frame-level metric."""
    frames = list(frames)
    summary: dict[str, Any] = {"frame_count": len(frames)}
    for key in ("mpjpe_mm", "pa_mpjpe_mm", "max_joint_error_mm", "root_yaw_error_degrees",
                "hinge_direction_mae_degrees",
                "shoulder_forward_depth_residual_mm", "hip_forward_depth_residual_mm",
                "shoulder_forward_depth_abs_residual_mm", "hip_forward_depth_abs_residual_mm"):
        summary[key] = _statistics([record.get(key) for record in frames])
    for key in ("shoulder_forward_depth_sign_disagreement", "hip_forward_depth_sign_disagreement",
                "shoulder_forward_depth_sign_disagreement_stable",
                "hip_forward_depth_sign_disagreement_stable"):
        values = [record.get(key) for record in frames if record.get(key) is not None]
        summary[key + "_rate"] = float(np.mean(values)) if values else None
        summary[key + "_count"] = len(values)
    return summary


def _statistics(values: list[Any]) -> dict[str, Any] | None:
    finite = np.asarray([value for value in values if value is not None], dtype=np.float64)
    if not finite.size:
        return None
    return {
        "mean": float(finite.mean()),
        "median": float(np.median(finite)),
        "p90": float(np.quantile(finite, 0.90)),
        "p95": float(np.quantile(finite, 0.95)),
        "count": int(finite.size),
    }


def _grouped(frames: list[dict[str, Any]], key) -> dict[str, dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = {}
    for record in frames:
        buckets.setdefault(str(key(record)), []).append(record)
    return {name: aggregate(items) for name, items in sorted(buckets.items())}


def _worst(frames: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    ranked = sorted((record for record in frames if record["mpjpe_mm"] is not None),
                    key=lambda record: -record["mpjpe_mm"])
    return [{key: record[key] for key in ("sample_id", "mpjpe_mm", "root_yaw_error_degrees", "strata")}
            for record in ranked[:count]]


def _forward_depth_metrics(estimate: np.ndarray, reference: np.ndarray, valid: np.ndarray,
                           label: str, left: int, right: int) -> dict[str, Any]:
    """`D = (y_right - y_left) / sqrt(2)` residual and sign agreement (docs/18, docs/21)."""
    if not (valid[left] and valid[right]):
        return {f"{label}_forward_depth_residual_mm": None,
                f"{label}_forward_depth_abs_residual_mm": None,
                f"{label}_forward_depth_sign_disagreement": None,
                f"{label}_forward_depth_sign_disagreement_stable": None,
                f"{label}_forward_depth_target_m": None}
    predicted = float(estimate[right, FORWARD_DEPTH_AXIS] - estimate[left, FORWARD_DEPTH_AXIS]) * BILATERAL_DEPTH_NORMALIZATION
    actual = float(reference[right, FORWARD_DEPTH_AXIS] - reference[left, FORWARD_DEPTH_AXIS]) * BILATERAL_DEPTH_NORMALIZATION
    residual = (predicted - actual) * 1000.0
    disagreement = int(np.sign(predicted) != np.sign(actual))
    return {
        f"{label}_forward_depth_residual_mm": residual,
        f"{label}_forward_depth_abs_residual_mm": abs(residual),
        f"{label}_forward_depth_sign_disagreement": disagreement,
        f"{label}_forward_depth_sign_disagreement_stable": (
            disagreement if abs(actual) >= STABLE_FORWARD_DEPTH_M else None),
        f"{label}_forward_depth_target_m": actual,
    }


def _hinge_direction_error(estimate: np.ndarray, reference: np.ndarray, valid: np.ndarray) -> float | None:
    errors = []
    for proximal, joint, distal in _HINGE_CHAINS:
        indices = (JOINT_INDEX[proximal], JOINT_INDEX[joint], JOINT_INDEX[distal])
        if not all(valid[index] for index in indices):
            continue
        predicted = _bend_direction(estimate[indices[1]], estimate[indices[0]], estimate[indices[2]])
        actual = _bend_direction(reference[indices[1]], reference[indices[0]], reference[indices[2]])
        if predicted is None or actual is None:
            continue
        cosine = float(np.clip(np.dot(predicted, actual), -1.0, 1.0))
        errors.append(float(np.degrees(np.arccos(cosine))))
    return float(np.mean(errors)) if errors else None


def compare(baseline: dict[str, Any], candidate: dict[str, Any], *, metric: str = "mpjpe_mm",
            top: int = 25) -> dict[str, Any]:
    """Frame-by-frame delta between two candidates evaluated on the same frames."""
    reference = {record["sample_id"]: record for record in baseline["frames"]}
    deltas = []
    for record in candidate["frames"]:
        other = reference.get(record["sample_id"])
        if other is None or record.get(metric) is None or other.get(metric) is None:
            continue
        deltas.append({
            "sample_id": record["sample_id"],
            "baseline": other[metric],
            "candidate": record[metric],
            "delta": record[metric] - other[metric],
            "strata": record["strata"],
        })
    if not deltas:
        raise ValueError("candidates share no comparable frames")
    values = np.asarray([item["delta"] for item in deltas])
    ordered = sorted(deltas, key=lambda item: item["delta"])
    return {
        "metric": metric,
        "baseline_candidate": baseline["candidate"],
        "candidate": candidate["candidate"],
        "compared_frame_count": len(deltas),
        "mean_delta": float(values.mean()),
        "median_delta": float(np.median(values)),
        "improved_frame_count": int((values < 0).sum()),
        "regressed_frame_count": int((values > 0).sum()),
        "most_improved": ordered[:top],
        "most_regressed": ordered[-top:][::-1],
        "delta_by_stratum": {
            name: _delta_by(deltas, name) for name in stratum_names()
        },
    }


def _delta_by(deltas: list[dict[str, Any]], stratum: str) -> dict[str, dict[str, float]]:
    buckets: dict[str, list[float]] = {}
    for item in deltas:
        buckets.setdefault(str(item["strata"].get(stratum, "unknown")), []).append(item["delta"])
    return {
        name: {"mean_delta": float(np.mean(values)), "frame_count": len(values),
               "improved_frame_count": int(sum(1 for value in values if value < 0))}
        for name, values in sorted(buckets.items())
    }
