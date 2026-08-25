#!/usr/bin/env python3
"""Quantitative attribution of the A9 3DPW root-yaw P95 tail.

Diagnostic only: trains nothing, changes no evaluation semantics, and reuses
the official evaluator's own per-frame yaw building blocks (``_angle_delta``,
``YAW_INDICES``) so every number here is consistent with what
``root_yaw_p95_degrees`` in the official report already counts.

Answers, from data already available in the holdout dataset and the
checkpoint's own predictions -- no new learned estimator:

  - the yaw-error distribution around the 30 degree gate (bins at
    30/45/90/150 degrees);
  - whether the tail is broadly distributed across sequences or concentrated
    in a few;
  - whether tail frames are isolated or temporally contiguous runs;
  - which action/view slices dominate;
  - whether shoulder-pair and hip-pair evidence agree, and which one is
    missing/failing when the combined metric is high;
  - whether tail frames correlate with low 2D input confidence or a small
    screen-space shoulder/hip span (near-frontal/near-rear ambiguity in the
    2D observation the lifter actually sees).

Usage:
  python3 scripts/attribute_yaw_tail.py \
    --checkpoint reports/direct_mix.pth --holdout /data/3dpw/prepared/holdout.json \
    --out audit/a9_yaw_tail_attribution.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from training.temporal_lifter import (
    H36M_NAMES, YAW_INDICES, _angle_delta, _arrays, _frame_metadata, _model,
    _predict_batched, _root_yaw_error_degrees, _torch, load_dataset,
)

BINS_DEGREES = (30.0, 45.0, 90.0, 150.0)
_YAW_PAIR_NAMES = ("shoulder", "hip")  # matches YAW_INDICES order


def _pair_yaw_errors(estimate: np.ndarray, reference: np.ndarray, valid: np.ndarray) -> dict[str, float | None]:
    """Shoulder-only and hip-only yaw error, same math as
    ``_root_yaw_error_degrees`` but not averaged across pairs -- this is the
    only way to tell "shoulder evidence failed" apart from "hip evidence
    failed" from a metric that deliberately averages the two."""
    errors: dict[str, float | None] = {}
    for name, (left, right) in zip(_YAW_PAIR_NAMES, YAW_INDICES):
        if not (valid[left] and valid[right]):
            errors[name] = None
            continue
        predicted_axis = estimate[right, :2] - estimate[left, :2]
        target_axis = reference[right, :2] - reference[left, :2]
        if min(np.linalg.norm(predicted_axis), np.linalg.norm(target_axis)) <= 1e-6:
            errors[name] = None
            continue
        errors[name] = float(abs(_angle_delta(
            float(np.arctan2(predicted_axis[1], predicted_axis[0])),
            float(np.arctan2(target_axis[1], target_axis[0])),
        )) * 180.0 / np.pi)
    return errors


def _error_bins(errors: list[float], bins: tuple[float, ...] = BINS_DEGREES) -> dict[str, dict[str, float]]:
    """Cumulative counts/fractions of ``errors`` at or above each bin edge."""
    array = np.asarray(errors, dtype=np.float64)
    total = len(array)
    return {
        f">={edge:g}deg": {
            "count": int((array >= edge).sum()),
            "fraction": float((array >= edge).sum() / total) if total else 0.0,
        }
        for edge in bins
    }


def _tail_concentration(rows: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    """How much of the tail (error >= threshold) comes from how few actions.

    A tail spread evenly across every sequence and a tail owned almost
    entirely by two or three sequences call for different fixes -- one is a
    general-model residual, the other is closer to a data/scene problem.
    """
    by_action: dict[str, int] = {}
    for row in rows:
        if row["yaw_error_deg"] is not None and row["yaw_error_deg"] >= threshold:
            by_action[row["action"]] = by_action.get(row["action"], 0) + 1
    total = sum(by_action.values())
    ranked = sorted(by_action.items(), key=lambda item: -item[1])
    cumulative_share = []
    running = 0
    for n in (1, 3, 5, 10):
        running = sum(count for _, count in ranked[:n])
        cumulative_share.append({"top_n_actions": n, "share": running / total if total else 0.0})
    return {
        "threshold_deg": threshold,
        "total_tail_frames": total,
        "actions_touched": len(by_action),
        "worst_actions": [{"action": action, "tail_frame_count": count} for action, count in ranked[:10]],
        "top_n_concentration": cumulative_share,
    }


def _temporal_runs(rows: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    """Contiguous-run length distribution of tail frames within each action.

    Rows must already be in each action's original frame order (true for
    dataset sequences, which _frame_metadata preserves). A tail made of long
    contiguous runs looks and behaves differently from one made of scattered
    singletons, even at the same aggregate rate.
    """
    run_lengths: list[int] = []
    current_action, current_run = None, 0
    for row in rows:
        is_tail = row["yaw_error_deg"] is not None and row["yaw_error_deg"] >= threshold
        if row["action"] != current_action:
            if current_run:
                run_lengths.append(current_run)
            current_action, current_run = row["action"], 0
        if is_tail:
            current_run += 1
        else:
            if current_run:
                run_lengths.append(current_run)
            current_run = 0
    if current_run:
        run_lengths.append(current_run)

    array = np.asarray(run_lengths, dtype=np.int64) if run_lengths else np.asarray([0])
    return {
        "threshold_deg": threshold,
        "run_count": len(run_lengths),
        "singleton_runs": int((array == 1).sum()),
        "max_run_length": int(array.max()) if run_lengths else 0,
        "mean_run_length": float(array.mean()) if run_lengths else 0.0,
        "runs_of_5_or_more": int((array >= 5).sum()),
    }


def _pair_disagreement(rows: list[dict[str, Any]], disagreement_deg: float = 20.0) -> dict[str, Any]:
    both = [row for row in rows if row["shoulder_error_deg"] is not None and row["hip_error_deg"] is not None]
    disagreements = [row for row in both if abs(row["shoulder_error_deg"] - row["hip_error_deg"]) >= disagreement_deg]
    only_shoulder = [row for row in rows if row["shoulder_error_deg"] is not None and row["hip_error_deg"] is None]
    only_hip = [row for row in rows if row["hip_error_deg"] is None and row["shoulder_error_deg"] is not None]
    neither = [row for row in rows if row["shoulder_error_deg"] is None and row["hip_error_deg"] is None]
    return {
        "frames_with_both_pairs": len(both),
        "frames_disagreeing_ge_20deg": len(disagreements),
        "frames_missing_hip_pair": len(only_shoulder),
        "frames_missing_shoulder_pair": len(only_hip),
        "frames_missing_both_pairs": len(neither),
    }


def _evidence_correlation(rows: list[dict[str, Any]], tail_threshold: float) -> dict[str, Any]:
    """Compare 2D-input confidence and pair span between tail and non-tail
    frames -- both already exist in the model's own (pelvis/torso-normalized,
    per A9's ``input_coordinate_normalization``) input, no new estimator
    involved. A small shoulder/hip span in that same normalized space is the
    near-frontal/near-rear configuration where left/right can't be resolved
    from a single 2D observation, independent of the subject's distance from
    the camera.
    """
    def stats(values: list[float]) -> dict[str, float]:
        array = np.asarray(values, dtype=np.float64)
        if not len(array):
            return {"mean": None, "p10": None, "median": None}
        return {"mean": float(array.mean()), "p10": float(np.quantile(array, 0.10)), "median": float(np.median(array))}

    tail = [row for row in rows if row["yaw_error_deg"] is not None and row["yaw_error_deg"] >= tail_threshold]
    rest = [row for row in rows if row["yaw_error_deg"] is not None and row["yaw_error_deg"] < tail_threshold]
    fields = ("shoulder_screen_span", "hip_screen_span", "shoulder_confidence_mean", "hip_confidence_mean")
    return {
        field: {"tail": stats([row[field] for row in tail]), "non_tail": stats([row[field] for row in rest])}
        for field in fields
    }


def _build_attribution(
    prediction: np.ndarray, targets: np.ndarray, valid: np.ndarray, inputs: np.ndarray,
    metadata: list[dict[str, str | None]],
) -> dict[str, Any]:
    left_shoulder, right_shoulder = YAW_INDICES[0]
    left_hip, right_hip = YAW_INDICES[1]
    rows: list[dict[str, Any]] = []
    for index, (estimate, reference, frame_valid, frame_input, meta) in enumerate(
        zip(prediction, targets, valid, inputs, metadata)
    ):
        pair_errors = _pair_yaw_errors(estimate, reference, frame_valid)
        yaw_error = _root_yaw_error_degrees(estimate, reference, frame_valid)
        rows.append({
            "global_index": index,
            "action": meta.get("action") or "unknown",
            "view": meta.get("view"),
            "yaw_error_deg": yaw_error,
            "shoulder_error_deg": pair_errors["shoulder"],
            "hip_error_deg": pair_errors["hip"],
            "shoulder_screen_span": float(abs(frame_input[right_shoulder, 0] - frame_input[left_shoulder, 0])),
            "hip_screen_span": float(abs(frame_input[right_hip, 0] - frame_input[left_hip, 0])),
            "shoulder_confidence_mean": float((frame_input[left_shoulder, 2] + frame_input[right_shoulder, 2]) / 2),
            "hip_confidence_mean": float((frame_input[left_hip, 2] + frame_input[right_hip, 2]) / 2),
        })

    yaw_valid_errors = [row["yaw_error_deg"] for row in rows if row["yaw_error_deg"] is not None]
    view_slices = sorted({row["view"] for row in rows})

    return {
        "frame_count": len(rows),
        "yaw_valid_frame_count": len(yaw_valid_errors),
        "yaw_error_bins": _error_bins(yaw_valid_errors),
        "yaw_error_p95_deg": float(np.quantile(yaw_valid_errors, 0.95)) if yaw_valid_errors else None,
        "yaw_error_mae_deg": float(np.mean(yaw_valid_errors)) if yaw_valid_errors else None,
        "concentration_at_30deg": _tail_concentration(rows, 30.0),
        "concentration_at_90deg": _tail_concentration(rows, 90.0),
        "temporal_runs_at_30deg": _temporal_runs(rows, 30.0),
        "temporal_runs_at_90deg": _temporal_runs(rows, 90.0),
        "pair_disagreement": _pair_disagreement(rows),
        "evidence_correlation_at_30deg": _evidence_correlation(rows, 30.0),
        "distinct_view_labels": view_slices,
        "rows": rows,
    }


def _run(checkpoint: Path, holdout: Path, device: str):
    torch, nn = _torch()
    ck = torch.load(checkpoint, map_location=device, weights_only=True)
    dataset = load_dataset(holdout)
    inputs, targets, valid, offsets = _arrays(
        dataset, int(ck["window"]), coordinate_normalization=ck.get("input_coordinate_normalization", "image_v1"),
    )
    model = _model(nn, int(ck["channels"]), ck.get("architecture", "legacy_tcn_v1")).to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    x = torch.as_tensor(inputs, dtype=torch.float32, device=device)
    with torch.no_grad():
        prediction = _predict_batched(
            model, x, torch.as_tensor(offsets, dtype=torch.long, device=device), 1024, device.startswith("cuda"),
        ).cpu().numpy()
    metadata = _frame_metadata(dataset)
    return prediction, targets, valid, inputs, metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--holdout", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    prediction, targets, valid, inputs, metadata = _run(args.checkpoint, args.holdout, args.device)
    attribution = _build_attribution(prediction, targets, valid, inputs, metadata)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(attribution, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    summary = {k: v for k, v in attribution.items() if k != "rows"}
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
