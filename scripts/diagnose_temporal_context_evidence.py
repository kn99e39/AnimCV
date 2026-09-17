#!/usr/bin/env python3
"""Evidence-only temporal-context diagnostic for frozen Frame Pose H0.

This script does not train, smooth, or alter production predictions.  It reads
the fixed 3DPW test FrameBank and H0 once, constructs predeclared joint-level
failure cohorts, and applies one parameter-free control only on those rows:
linear interpolation of a joint's H0 location from the closest valid observed
FrameBank neighbours on both sides (at most two retained timestamps away).

The control answers whether immediately adjacent real-scene evidence contains
recoverable signal; it is neither a temporal model nor a production candidate.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from common.canonical_pose import JOINT_INDEX, JOINT_NAMES
from common.serialization import write_json
from framepose.bank import load_bank
from framepose.pose_reconciliation import HINGE_FIELDS
from framepose.replay_provenance import verify_source_identity
from framepose.signs import HINGE_CHAINS_BY_JOINT, SIGN_FIELD_NAMES, oracle_sign_states


SCHEMA = "animcv_temporal_context_evidence_v2"
MAX_NEIGHBOR_STEPS = 2
COHORTS = (
    "current_frame_observation_loss",
    "observation_degradation",
    "h0_jitter_stable_observation",
    "distal_joint_failure",
    "implausible_articulation",
    "stable_control",
)
FAILURE_COHORTS = COHORTS[:-1]
DISTAL_JOINTS = ("left_ankle", "right_ankle")
EPSILON = 1e-12
FIXED_BANK_DIGEST = "75519e6394a764e3749ddaa30555b58b73a01db582ccee14f661374b9a0ed536"
FIXED_PREDICTION_SHA256 = "6df04c2031b2bde6a4f0af37fdae0a7f0228e9fc23401e15e1aa2b3bd35695a5"
FIXED_EVALUATION_SHA256 = "a5881850b3769ca2329271d327ee895dc646f236451f05c3ba2e0ff917f580a3"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_text_with_korean_fallback(path: Path) -> str:
    raw = path.read_bytes()
    for encoding in ("utf-8-sig", "utf-8", "cp949"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            pass
    raise UnicodeDecodeError("unknown", raw, 0, len(raw), "could not decode annotation CSV")


def _sample_time(sample) -> float:
    if sample.timestamp is not None and np.isfinite(sample.timestamp):
        return float(sample.timestamp)
    if sample.fps is not None and np.isfinite(sample.fps) and sample.fps > 0:
        return float(sample.frame_index) / float(sample.fps)
    return float(sample.frame_index)


def sequence_rows(samples: list[Any], positions: np.ndarray) -> dict[str, np.ndarray]:
    """Return deterministic test-row order by sequence and timestamp/frame.

    Rows are never borrowed across sequences.  Equal time/frame keys are a
    malformed FrameBank, because interpolation would then be undefined.
    """
    grouped: dict[str, list[int]] = defaultdict(list)
    for row, position in enumerate(positions):
        grouped[samples[int(position)].sequence_id].append(row)
    result: dict[str, np.ndarray] = {}
    for sequence, rows in grouped.items():
        ordered = sorted(rows, key=lambda row: (
            _sample_time(samples[int(positions[row])]),
            int(samples[int(positions[row])].frame_index),
            samples[int(positions[row])].sample_id,
        ))
        keys = [(float(_sample_time(samples[int(positions[row])])),
                 int(samples[int(positions[row])].frame_index)) for row in ordered]
        if len(keys) != len(set(keys)):
            raise ValueError(f"duplicate timestamp/frame key in {sequence}")
        result[sequence] = np.asarray(ordered, dtype=np.int64)
    return result


def local_neighbor_rows(grouped: dict[str, np.ndarray], frame_count: int,
                        max_steps: int = MAX_NEIGHBOR_STEPS) -> tuple[list[list[int]], list[list[int]]]:
    """Indices before/after each test row, bounded inside its own sequence."""
    before, after = ([[] for _ in range(frame_count)] for _ in range(2))
    for rows in grouped.values():
        for local, row in enumerate(rows.tolist()):
            before[row] = rows[max(0, local - max_steps):local][::-1].astype(int).tolist()
            after[row] = rows[local + 1:local + 1 + max_steps].astype(int).tolist()
    return before, after


def _in_frame(observation: np.ndarray, valid: np.ndarray) -> np.ndarray:
    xy = np.asarray(observation[..., :2], dtype=np.float64)
    return (np.asarray(valid, dtype=bool) & np.isfinite(xy).all(axis=-1)
            & (xy[..., 0] >= 0.0) & (xy[..., 0] <= 1.0)
            & (xy[..., 1] >= 0.0) & (xy[..., 1] <= 1.0))


def _linear_residual(left: np.ndarray, center: np.ndarray, right: np.ndarray,
                     left_time: float, center_time: float, right_time: float) -> float:
    if not (np.isfinite(left).all() and np.isfinite(center).all() and np.isfinite(right).all()):
        return float("nan")
    span = right_time - left_time
    if not np.isfinite(span) or span <= 0.0:
        return float("nan")
    weight = (center_time - left_time) / span
    if not 0.0 < weight < 1.0:
        return float("nan")
    return float(np.linalg.norm(center - ((1.0 - weight) * left + weight * right)))


def trajectory_residuals(values: np.ndarray, valid: np.ndarray, timestamps: np.ndarray,
                          before: list[list[int]], after: list[list[int]]) -> np.ndarray:
    """Timestamp-aware local second-difference magnitude; no fitted velocity."""
    result = np.full(valid.shape, np.nan, dtype=np.float64)
    for row in range(len(values)):
        if not before[row] or not after[row]:
            continue
        left, right = before[row][0], after[row][0]
        for joint in np.flatnonzero(valid[row]):
            if valid[left, joint] and valid[right, joint]:
                result[row, joint] = _linear_residual(
                    values[left, joint], values[row, joint], values[right, joint],
                    timestamps[left], timestamps[row], timestamps[right])
    return result


def nearest_temporal_support(in_frame: np.ndarray, h0: np.ndarray, timestamps: np.ndarray,
                             before: list[list[int]], after: list[list[int]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Closest in-window neighbours with a valid observation and finite H0.

    `left`/`right` remain -1 if unavailable.  No extrapolation is represented
    by a missing side and must remain unresolved.
    """
    count, joints = in_frame.shape
    left = np.full((count, joints), -1, dtype=np.int64)
    right = np.full((count, joints), -1, dtype=np.int64)
    finite_h0 = np.isfinite(h0).all(axis=-1)
    for row in range(count):
        for joint in range(joints):
            for candidate in before[row]:
                if in_frame[candidate, joint] and finite_h0[candidate, joint]:
                    left[row, joint] = candidate
                    break
            for candidate in after[row]:
                if in_frame[candidate, joint] and finite_h0[candidate, joint]:
                    right[row, joint] = candidate
                    break
    eligible = (left >= 0) & (right >= 0)
    return left, right, eligible


def neighbor_evidence(in_frame: np.ndarray, confidence: np.ndarray, h0_error: np.ndarray,
                      before: list[list[int]], after: list[list[int]]) -> dict[str, np.ndarray]:
    """Describe the real adjacent evidence without choosing a recovery rule.

    All comparisons are deliberately local to the predeclared t±2 window.
    The oracle-error comparison is analysis-only and is never used to select a
    neighbour for the interpolation control.
    """
    shape = in_frame.shape
    observed_before = np.full(shape, -1, dtype=np.int64)
    observed_after = np.full(shape, -1, dtype=np.int64)
    higher_confidence = np.zeros(shape, dtype=bool)
    lower_oracle_error = np.zeros(shape, dtype=bool)
    for row in range(shape[0]):
        candidates = before[row] + after[row]
        for joint in range(shape[1]):
            valid_candidates = [candidate for candidate in candidates if in_frame[candidate, joint]]
            before_candidates = [candidate for candidate in before[row] if in_frame[candidate, joint]]
            after_candidates = [candidate for candidate in after[row] if in_frame[candidate, joint]]
            if before_candidates:
                observed_before[row, joint] = before_candidates[0]
            if after_candidates:
                observed_after[row, joint] = after_candidates[0]
            if np.isfinite(confidence[row, joint]):
                higher_confidence[row, joint] = any(
                    confidence[candidate, joint] > confidence[row, joint]
                    for candidate in valid_candidates if np.isfinite(confidence[candidate, joint]))
            if np.isfinite(h0_error[row, joint]):
                lower_oracle_error[row, joint] = any(
                    h0_error[candidate, joint] < h0_error[row, joint]
                    for candidate in valid_candidates if np.isfinite(h0_error[candidate, joint]))
    return {
        "observed_before": observed_before,
        "observed_after": observed_after,
        "two_sided_observation": (observed_before >= 0) & (observed_after >= 0),
        "higher_confidence_neighbor": higher_confidence,
        "lower_oracle_error_neighbor": lower_oracle_error,
    }


def observation_gap_metrics(in_frame: np.ndarray, target: np.ndarray, finite_target: np.ndarray,
                            timestamps: np.ndarray, grouped: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Measure contiguous invalid-observation spans and oracle smoothness across them.

    This is descriptive accounting for observation-loss rows.  It does not
    supply H0 values to the recovery control and it never crosses sequences.
    """
    shape = in_frame.shape
    lengths = np.zeros(shape, dtype=np.int64)
    residual = np.full(shape, np.nan, dtype=np.float64)
    for rows in grouped.values():
        ordered = rows.astype(int).tolist()
        for joint in range(shape[1]):
            local = 0
            while local < len(ordered):
                if in_frame[ordered[local], joint]:
                    local += 1
                    continue
                start = local
                while local < len(ordered) and not in_frame[ordered[local], joint]:
                    local += 1
                end = local
                run = ordered[start:end]
                for row in run:
                    lengths[row, joint] = len(run)
                if start == 0 or end == len(ordered):
                    continue
                left, right = ordered[start - 1], ordered[end]
                for row in run:
                    if finite_target[left, joint] and finite_target[row, joint] and finite_target[right, joint]:
                        residual[row, joint] = _linear_residual(
                            target[left, joint], target[row, joint], target[right, joint],
                            timestamps[left], timestamps[row], timestamps[right])
    return {"observation_gap_length_frames": lengths, "oracle_gap_residual": residual}


def interpolate_recovery(h0: np.ndarray, timestamps: np.ndarray, left: np.ndarray,
                         right: np.ndarray, eligible: np.ndarray) -> np.ndarray:
    """Joint-only deterministic temporal control; no prediction is changed in place."""
    recovered = np.asarray(h0, dtype=np.float64).copy()
    for row, joint in zip(*np.nonzero(eligible)):
        before, after = int(left[row, joint]), int(right[row, joint])
        span = timestamps[after] - timestamps[before]
        if not np.isfinite(span) or span <= 0.0:
            raise ValueError("eligible temporal support has non-positive timestamp span")
        weight = (timestamps[row] - timestamps[before]) / span
        if not 0.0 < weight < 1.0:
            raise ValueError("eligible temporal support does not bracket the current row")
        recovered[row, joint] = ((1.0 - weight) * h0[before, joint]
                                 + weight * h0[after, joint])
    return recovered


def per_joint_quantile(values: np.ndarray, percentile: float) -> np.ndarray:
    output = np.full(values.shape[1], np.nan, dtype=np.float64)
    for joint in range(values.shape[1]):
        finite = values[:, joint][np.isfinite(values[:, joint])]
        if len(finite):
            output[joint] = float(np.percentile(finite, percentile))
    return output


def _hinge_articulation_mask(h0: np.ndarray, target: np.ndarray, target_valid: np.ndarray) -> np.ndarray:
    """Current H0 hinge-sign disagreement only; does not invoke reconciliation."""
    mask = np.zeros(target_valid.shape, dtype=bool)
    target_signs = oracle_sign_states(target, target_valid)
    from framepose.signs import sign_state
    h0_signs = np.asarray([sign_state(pose, valid) for pose, valid in zip(h0, target_valid)])
    for field in HINGE_FIELDS:
        joint = field[: -len("_forward_bend")]
        proximal, middle, distal = HINGE_CHAINS_BY_JOINT[joint]
        p, m, d = (JOINT_INDEX[name] for name in (proximal, middle, distal))
        column = SIGN_FIELD_NAMES.index(field)
        known = (target_signs[:, column] != 0) & (h0_signs[:, column] != 0)
        chain_valid = target_valid[:, p] & target_valid[:, m] & target_valid[:, d]
        mask[:, m] = known & chain_valid & (target_signs[:, column] != h0_signs[:, column])
    return mask


def build_cohorts(observation: np.ndarray, observation_valid: np.ndarray, h0: np.ndarray,
                  target: np.ndarray, target_valid: np.ndarray, timestamps: np.ndarray,
                  before: list[list[int]], after: list[list[int]]) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Predeclared structural/quantile cohorts, before recovery outcomes exist."""
    in_frame = _in_frame(observation, observation_valid)
    finite_h0 = np.isfinite(h0).all(axis=-1)
    finite_target = np.isfinite(target).all(axis=-1) & target_valid
    observation_residual = trajectory_residuals(observation[..., :2], in_frame, timestamps, before, after)
    h0_residual = trajectory_residuals(h0, finite_h0, timestamps, before, after)
    h0_error = np.linalg.norm(h0 - target, axis=-1)
    h0_error[~(finite_h0 & finite_target)] = np.nan
    obs_p50, obs_p90 = per_joint_quantile(observation_residual, 50), per_joint_quantile(observation_residual, 90)
    h0_p50, h0_p90 = per_joint_quantile(h0_residual, 50), per_joint_quantile(h0_residual, 90)
    error_p50, error_p90 = per_joint_quantile(h0_error, 50), per_joint_quantile(h0_error, 90)
    local_has_before = np.asarray([bool(items) for items in before], dtype=bool)[:, None]
    local_has_after = np.asarray([bool(items) for items in after], dtype=bool)[:, None]
    cohorts = {
        "current_frame_observation_loss": (~in_frame & local_has_before & local_has_after),
        "observation_degradation": (in_frame & np.isfinite(observation_residual)
                                    & (observation_residual >= obs_p90[None, :])),
        "h0_jitter_stable_observation": (in_frame & finite_h0
                                          & np.isfinite(observation_residual) & np.isfinite(h0_residual)
                                          & (observation_residual <= obs_p50[None, :])
                                          & (h0_residual >= h0_p90[None, :])),
        "distal_joint_failure": np.zeros(in_frame.shape, dtype=bool),
        "implausible_articulation": _hinge_articulation_mask(h0, target, target_valid),
    }
    for name in DISTAL_JOINTS:
        joint = JOINT_INDEX[name]
        cohorts["distal_joint_failure"][:, joint] = (
            finite_target[:, joint] & finite_h0[:, joint] & np.isfinite(h0_error[:, joint])
            & (h0_error[:, joint] >= error_p90[joint]))
    stable_base = (in_frame & finite_h0 & finite_target
                   & np.isfinite(observation_residual) & np.isfinite(h0_residual) & np.isfinite(h0_error)
                   & (observation_residual <= obs_p50[None, :])
                   & (h0_residual <= h0_p50[None, :])
                   & (h0_error <= error_p50[None, :]))
    # A control cannot itself exhibit any predeclared failure mode.  This
    # excludes quantile ties and low-error but sign-implausible rows without
    # looking at the probe outcome.
    cohorts["stable_control"] = stable_base & ~np.logical_or.reduce(
        [cohorts[name] for name in FAILURE_COHORTS])
    quantities = {
        "in_frame": in_frame,
        "finite_h0": finite_h0,
        "finite_target": finite_target,
        "observation_residual": observation_residual,
        "h0_residual": h0_residual,
        "h0_error": h0_error,
        "observation_residual_p50": obs_p50,
        "observation_residual_p90": obs_p90,
        "h0_residual_p50": h0_p50,
        "h0_residual_p90": h0_p90,
        "h0_error_p50": error_p50,
        "h0_error_p90": error_p90,
    }
    return cohorts, quantities


def classify_temporal_rows(failure: np.ndarray, in_frame: np.ndarray, stable_observation: np.ndarray,
                           eligible: np.ndarray, target_valid: np.ndarray,
                           before_error: np.ndarray, after_error: np.ndarray) -> np.ndarray:
    """Architecture-relevant labels with no optimized error-reduction threshold."""
    labels = np.full(failure.shape, "", dtype=object)
    measurable = target_valid & np.isfinite(before_error)
    delta = before_error - after_error
    labels[failure & eligible & measurable & (delta > EPSILON)] = "T1_TEMPORALLY_RECOVERABLE_OBSERVATION_GAP"
    labels[failure & eligible & measurable & ~(delta > EPSILON)] = "T2_TEMPORAL_EVIDENCE_SIMPLE_RECOVERY_INSUFFICIENT"
    labels[failure & ~eligible & ~in_frame] = "T4_OBSERVATION_FAILURE_WITHOUT_TEMPORAL_SUPPORT"
    labels[failure & ~eligible & in_frame & stable_observation] = "T3_NON_TEMPORAL_FRAME_POSE_FAILURE"
    labels[failure & (labels == "")] = "T5_MIXED_OR_UNRESOLVED"
    return labels


def failure_mask(cohorts: dict[str, np.ndarray]) -> np.ndarray:
    """Failure rows eligible for the probe; stable control wins every tie."""
    return (np.logical_or.reduce([cohorts[name] for name in FAILURE_COHORTS])
            & ~cohorts["stable_control"])


def _quantiles_mm(values: np.ndarray) -> dict[str, Any]:
    finite = np.asarray(values, dtype=np.float64)[np.isfinite(values)] * 1000.0
    if not len(finite):
        return {"count": 0}
    return {"count": int(len(finite)), "mean": float(finite.mean()),
            "p50": float(np.percentile(finite, 50)), "p95": float(np.percentile(finite, 95))}


def _quantiles(values: np.ndarray) -> dict[str, Any]:
    finite = np.asarray(values, dtype=np.float64)[np.isfinite(values)]
    if not len(finite):
        return {"count": 0}
    return {"count": int(len(finite)), "mean": float(finite.mean()),
            "p50": float(np.percentile(finite, 50)), "p95": float(np.percentile(finite, 95))}


def summarize_cohort(name: str, mask: np.ndarray, samples: list[Any], positions: np.ndarray,
                     in_frame: np.ndarray, observation_residual: np.ndarray, h0_residual: np.ndarray,
                     before_error: np.ndarray, after_error: np.ndarray, eligible: np.ndarray,
                     labels: np.ndarray, neighbor_info: dict[str, np.ndarray],
                     gap_info: dict[str, np.ndarray]) -> dict[str, Any]:
    rows, joints = np.nonzero(mask)
    target_mask = mask & np.isfinite(before_error)
    deltas = before_error[target_mask] - after_error[target_mask]
    row_ids = sorted(set(rows.tolist()))
    sequence_ids = {samples[int(positions[row])].sequence_id for row in row_ids}
    label_counts: dict[str, int] = {}
    for label in labels[mask]:
        label_counts[str(label)] = label_counts.get(str(label), 0) + 1
    return {
        "joint_rows": int(mask.sum()), "frames": len(row_ids), "sequences": len(sequence_ids),
        "joint_names": sorted({JOINT_NAMES[joint] for joint in joints.tolist()}),
        "observation_loss_fraction": float((~in_frame[mask]).mean()) if mask.any() else float("nan"),
        "eligible_fraction": float(eligible[mask].mean()) if mask.any() else float("nan"),
        "unresolved_fraction": float((~eligible[mask]).mean()) if mask.any() else float("nan"),
        "h0_error_mm": _quantiles_mm(before_error[mask]),
        "recovery_error_mm": _quantiles_mm(after_error[mask & eligible]),
        "error_delta_mm": _quantiles_mm(deltas),
        "fraction_improved": float((deltas > EPSILON).mean()) if len(deltas) else float("nan"),
        "fraction_worsened": float((deltas < -EPSILON).mean()) if len(deltas) else float("nan"),
        "observation_trajectory_residual_normalized": _quantiles(observation_residual[mask]),
        "h0_trajectory_residual_mm": _quantiles_mm(h0_residual[mask]),
        "neighbor_evidence": {
            "two_sided_valid_2d_fraction": float(neighbor_info["two_sided_observation"][mask].mean()) if mask.any() else float("nan"),
            "higher_confidence_neighbor_fraction": float(neighbor_info["higher_confidence_neighbor"][mask].mean()) if mask.any() else float("nan"),
            "lower_oracle_h0_error_neighbor_fraction": float(neighbor_info["lower_oracle_error_neighbor"][mask].mean()) if mask.any() else float("nan"),
        },
        "observation_gap_accounting": {
            "contiguous_invalid_observation_frames": _quantiles(gap_info["observation_gap_length_frames"][mask]),
            "oracle_target_linear_residual_mm_across_gap": _quantiles_mm(gap_info["oracle_gap_residual"][mask]),
        },
        "temporal_classes": label_counts,
    }


def _seed_events(annotation_path: Path | None, manifest_path: Path | None,
                 samples: list[Any], positions: np.ndarray, labels: np.ndarray,
                 cohorts: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    if annotation_path is None or manifest_path is None:
        return []
    annotations = list(csv.DictReader(_read_text_with_korean_fallback(annotation_path).splitlines()))
    events = {entry["review_id"]: entry for entry in
              json.loads(manifest_path.read_text(encoding="utf-8"))["primary_blind_and_labelled_events"]}
    row_by_sample = {samples[int(position)].sample_id: row for row, position in enumerate(positions)}
    output = []
    for annotation in annotations:
        note = annotation.get("optional_note", "").strip()
        if not note:
            continue
        event = events.get(annotation["review_id"])
        if event is None:
            raise KeyError(f"owner annotation is absent from review manifest: {annotation['review_id']}")
        row = row_by_sample[event["sample_id"]]
        chain = event["chain"]
        joint = JOINT_INDEX[chain[1]]
        categories = []
        for name in FAILURE_COHORTS:
            if cohorts[name][row, joint]:
                categories.append(name)
        output.append({
            "review_id": annotation["review_id"], "note": note,
            "sequence_id": event["sequence_id"], "frame_index": event["center_frame"],
            "field": event["field"], "target_joint": chain[1], "test_row": int(row),
            "predeclared_cohorts_at_target_joint": categories or ["MIXED"],
            "temporal_class": str(labels[row, joint]),
        })
    return output


def _write_rows(path: Path, mask: np.ndarray, samples: list[Any], positions: np.ndarray,
                observation: np.ndarray, observation_valid: np.ndarray, quantities: dict[str, np.ndarray],
                left: np.ndarray, right: np.ndarray, eligible: np.ndarray, recovered: np.ndarray,
                after_error: np.ndarray, labels: np.ndarray, cohorts: dict[str, np.ndarray]) -> None:
    columns = [
        "test_row", "sample_id", "sequence_id", "frame_index", "timestamp", "joint",
        "observation_x_normalized", "observation_y_normalized", "detector_confidence", "observation_valid", "in_frame",
        "h0_x_m", "h0_y_m", "h0_z_m", "target_valid", "target_x_m", "target_y_m", "target_z_m",
                "observation_trajectory_residual_normalized", "h0_trajectory_residual_m", "h0_error_m",
        "neighbor_before_row", "neighbor_after_row", "observed_before_row", "observed_after_row", "two_sided_valid_2d",
        "has_higher_confidence_neighbor", "has_lower_oracle_h0_error_neighbor", "observation_gap_length_frames",
        "oracle_target_gap_residual_m", "temporally_eligible", "recovered_x_m", "recovered_y_m", "recovered_z_m",
        "recovery_error_m", "temporal_class", "cohorts",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row, joint in zip(*np.nonzero(mask)):
            sample = samples[int(positions[row])]
            writer.writerow({
                "test_row": row, "sample_id": sample.sample_id, "sequence_id": sample.sequence_id,
                "frame_index": sample.frame_index, "timestamp": _sample_time(sample), "joint": JOINT_NAMES[joint],
                "observation_x_normalized": observation[row, joint, 0], "observation_y_normalized": observation[row, joint, 1],
                "detector_confidence": observation[row, joint, 2], "observation_valid": bool(observation_valid[row, joint]),
                "in_frame": bool(quantities["in_frame"][row, joint]),
                "h0_x_m": quantities["h0"][row, joint, 0], "h0_y_m": quantities["h0"][row, joint, 1], "h0_z_m": quantities["h0"][row, joint, 2],
                "target_valid": bool(quantities["finite_target"][row, joint]),
                "target_x_m": quantities["target"][row, joint, 0], "target_y_m": quantities["target"][row, joint, 1], "target_z_m": quantities["target"][row, joint, 2],
                "observation_trajectory_residual_normalized": quantities["observation_residual"][row, joint],
                "h0_trajectory_residual_m": quantities["h0_residual"][row, joint], "h0_error_m": quantities["h0_error"][row, joint],
                "neighbor_before_row": left[row, joint], "neighbor_after_row": right[row, joint],
                "observed_before_row": quantities["neighbor_info"]["observed_before"][row, joint],
                "observed_after_row": quantities["neighbor_info"]["observed_after"][row, joint],
                "two_sided_valid_2d": bool(quantities["neighbor_info"]["two_sided_observation"][row, joint]),
                "has_higher_confidence_neighbor": bool(quantities["neighbor_info"]["higher_confidence_neighbor"][row, joint]),
                "has_lower_oracle_h0_error_neighbor": bool(quantities["neighbor_info"]["lower_oracle_error_neighbor"][row, joint]),
                "observation_gap_length_frames": quantities["gap_info"]["observation_gap_length_frames"][row, joint],
                "oracle_target_gap_residual_m": quantities["gap_info"]["oracle_gap_residual"][row, joint],
                "temporally_eligible": bool(eligible[row, joint]),
                "recovered_x_m": recovered[row, joint, 0], "recovered_y_m": recovered[row, joint, 1], "recovered_z_m": recovered[row, joint, 2],
                "recovery_error_m": after_error[row, joint], "temporal_class": labels[row, joint],
                "cohorts": ";".join(name for name in COHORTS if cohorts[name][row, joint]),
            })


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--owner-annotations", type=Path)
    parser.add_argument("--review-manifest", type=Path)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite temporal evidence output: {args.out}")
    bank = load_bank(args.bank)
    positions = bank.indices("test")
    if len(positions) != 7076 or bank.content_digest() != FIXED_BANK_DIGEST:
        raise ValueError(f"expected fixed 7,076-row test bank, got {len(positions)}")
    identity = verify_source_identity(prediction=args.prediction, evaluation=args.evaluation,
                                      bank=bank, split="test",
                                      candidate="O_BILATERAL_oracle_forward_depth_only",
                                      frames=len(positions), joints=len(JOINT_NAMES))
    if identity["prediction"]["sha256"] != FIXED_PREDICTION_SHA256 or \
            identity["evaluation"]["sha256"] != FIXED_EVALUATION_SHA256:
        raise ValueError("prediction/evaluation identity differs from docs/44-46 baseline")
    h0 = np.load(args.prediction).astype(np.float64)
    if h0.shape != (len(positions), len(JOINT_NAMES), 3):
        raise ValueError(f"unexpected H0 shape: {h0.shape}")
    observation = bank.arrays["input_2d"][positions].astype(np.float64)
    observation_valid = bank.arrays["input_valid"][positions]
    target = bank.arrays["target_3d"][positions].astype(np.float64)
    target_valid = bank.arrays["target_valid"][positions]
    grouped = sequence_rows(bank.samples, positions)
    before, after = local_neighbor_rows(grouped, len(positions))
    timestamps = np.asarray([_sample_time(bank.samples[int(position)]) for position in positions], dtype=np.float64)
    cohorts, quantities = build_cohorts(observation, observation_valid, h0, target, target_valid,
                                        timestamps, before, after)
    left, right, temporal_support = nearest_temporal_support(
        quantities["in_frame"], h0, timestamps, before, after)
    neighbor_info = neighbor_evidence(quantities["in_frame"], observation[..., 2], quantities["h0_error"],
                                      before, after)
    gap_info = observation_gap_metrics(quantities["in_frame"], target, quantities["finite_target"],
                                       timestamps, grouped)
    # Stable control is an explicit preservation set.  Quantile ties can make
    # it overlap another structural cohort; control membership wins so the
    # probe can never change a stable-control H0 joint.
    failure = failure_mask(cohorts)
    eligible = failure & temporal_support
    recovered = interpolate_recovery(h0, timestamps, left, right, eligible)
    before_error = quantities["h0_error"]
    after_error = np.linalg.norm(recovered - target, axis=-1)
    after_error[~quantities["finite_target"]] = np.nan
    stable_observation = quantities["in_frame"] & np.isfinite(quantities["observation_residual"]) & (
        quantities["observation_residual"] <= quantities["observation_residual_p50"][None, :])
    labels = classify_temporal_rows(failure, quantities["in_frame"], stable_observation, eligible,
                                    quantities["finite_target"], before_error, after_error)
    quantities.update({"h0": h0, "target": target, "neighbor_info": neighbor_info, "gap_info": gap_info})
    args.out.mkdir(parents=True, exist_ok=False)
    rows_mask = failure | cohorts["stable_control"]
    rows_path = args.out / "temporal_evidence_rows.csv"
    _write_rows(rows_path, rows_mask, bank.samples, positions, observation, observation_valid, quantities,
                left, right, eligible, recovered, after_error, labels, cohorts)
    report = {
        "schema": SCHEMA,
        "purpose": "evidence-only neighboring-frame availability and parameter-free H0 interpolation control",
        "frozen_boundaries": ["Frame-first H0 retained", "no training", "no production smoothing", "no Pose Reconciliation invocation"],
        "source": {"bank_content_digest": bank.content_digest(), "prediction_sha256": identity["prediction"]["sha256"],
                   "evaluation_sha256": identity["evaluation"]["sha256"], "test_frames": int(len(positions)),
                   "test_sequences": len(grouped)},
        "predeclared_temporal_window": {"max_neighbor_steps": MAX_NEIGHBOR_STEPS,
                                          "available_offsets": ["t-1", "t+1", "t-2", "t+2"],
                                          "cross_sequence_leakage": False},
        "cohort_rules": {
            "current_frame_observation_loss": "not in-frame/valid at t, with retained same-sequence context on both sides",
            "observation_degradation": "valid local 2D trajectory residual at or above its per-joint test P90",
            "h0_jitter_stable_observation": "valid local observation residual at or below per-joint P50 and H0 trajectory residual at or above P90",
            "distal_joint_failure": "left/right ankle target-relative H0 error at or above its per-joint test P90",
            "implausible_articulation": "known target hinge sign differs from current H0 sign at the valid middle joint; no reconciliation run",
            "stable_control": "in-frame, finite H0/target, and observation/H0/error quantities each at or below their per-joint P50",
        },
        "recovery_probe": "for failure-cohort joint rows only, interpolate H0 from closest observed+finite H0 neighbours on both sides within t±2 using timestamps; otherwise unresolved; stable controls unchanged",
        "cohorts": {name: summarize_cohort(name, cohorts[name], bank.samples, positions, quantities["in_frame"],
                                             quantities["observation_residual"], quantities["h0_residual"], before_error,
                                             after_error, eligible, labels, neighbor_info, gap_info) for name in COHORTS},
        "temporal_class_counts": {label: int((labels == label).sum()) for label in sorted(set(labels[failure].tolist()))},
        "owner_annotation_seed_events": _seed_events(args.owner_annotations, args.review_manifest, bank.samples,
                                                       positions, labels, cohorts),
        "temporal_evidence_rows": {"path": rows_path.name, "sha256": _sha256(rows_path),
                                    "joint_rows": int(rows_mask.sum())},
    }
    write_json(args.out / "temporal_context_report.json", report)
    print(json.dumps({"output": str(args.out), "failure_joint_rows": int(failure.sum()),
                      "eligible_joint_rows": int(eligible.sum()), "rows_sha256": report["temporal_evidence_rows"]["sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
