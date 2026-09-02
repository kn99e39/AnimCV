#!/usr/bin/env python3
"""Why A16 fits bilateral forward-depth in-domain but fails to generalize
it to 3DPW official test (docs/22).

Diagnostic-only: no training, no checkpoint/report changes. The
scientifically valid comparison throughout is A15 (compiled A9 control)
vs A16 (corrected SRD candidate) -- NOT historical eager A9 vs A16 (docs/21
already established the execution backend is itself a confound for that
comparison). The frozen hard-set (top-5%/top-1% by historical A9's own
evaluator ranking) is reused exactly as established in docs/18/21 and never
redefined after observing A16.

For shoulder and hip pairs, on GT/A15/A16 predictions, computes per frame:

    delta_X = X_right - X_left        (canonical +X = right)
    delta_Y = Y_right - Y_left        (canonical +Y = forward/depth)
    magnitude = sqrt(delta_X^2 + delta_Y^2)
    angle = atan2(delta_Y, delta_X)   (exactly the production evaluator's
                                        own per-pair angle formula)

then residuals (prediction vs GT): delta_X/delta_Y absolute residual,
magnitude residual, angular error (production _angle_delta), forward-depth
sign disagreement -- Section 4.

Section 5 constructs diagnostic-only counterfactual bilateral vectors
(CF_X15_Y16: A15's delta_X with A16's delta_Y; CF_X16_Y15: the reverse)
and their angle vs GT -- mathematically exact for the pair-angle itself
(root yaw depends only on each pair's own X/Y), reported as counterfactual
angles, not claimed as a real candidate's official evaluator score.

Section 6 measures shoulder-vs-hip orientation coherence (angular
disagreement, forward-depth sign agreement) per skeleton (GT/A15/A16),
with sequence-boundary-safe temporal run accounting.

Section 7 builds a diagnostic torso local frame from canonical geometry
only (combined shoulder+hip lateral axis) and measures each pair's
consistency with it -- no training loss, no metric renormalization.

Section 8 partitions frames by frozen A15-vs-threshold and A15-vs-A16
comparison into rescued / newly-damaged / stable categories.

Section 9 analyzes forward-depth sign-transition behavior per sequence.

Usage:
  python3 scripts/diagnose_a16_generalization_gap.py \
    --a15-checkpoint /output/experiments/ablation_a15_compiled_a9_control_10e/reports/direct_mix.pth \
    --a16-checkpoint /output/experiments/ablation_a16_bilateral_forward_depth_corrected_10e/reports/direct_mix.pth \
    --historical-a9-checkpoint /output/experiments/ablation_a9_fingerprinted_baseline_10e/reports/direct_mix.pth \
    --holdout /data/3dpw/prepared/holdout.json \
    --out /output/experiments/a22_a16_generalization_diagnosis/generalization_gap.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from training.temporal_lifter import (
    YAW_INDICES, _angle_delta, _arrays, _frame_metadata, _model, _predict_batched,
    _root_yaw_error_degrees, _torch, load_dataset,
)

_FIXED_SEQUENCES = ("downtown_stairs_00:actor0", "downtown_walking_00:actor1",
                    "downtown_bus_00:actor1", "downtown_bar_00:actor0")
_PAIR_NAMES = ("shoulder", "hip")
_COHERENCE_DISAGREEMENT_DEG = 20.0  # matches attribute_yaw_tail.py's existing convention


def _predict(torch, nn, checkpoint_path: Path, dataset: dict[str, Any], device: str):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    inputs, targets, valid, offsets = _arrays(
        dataset, int(checkpoint["window"]),
        coordinate_normalization=checkpoint.get("input_coordinate_normalization", "image_v1"),
    )
    model = _model(nn, int(checkpoint["channels"]), checkpoint.get("architecture", "legacy_tcn_v1")).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    x = torch.as_tensor(inputs, dtype=torch.float32, device=device)
    with torch.no_grad():
        prediction = _predict_batched(
            model, x, torch.as_tensor(offsets, dtype=torch.long, device=device), 1024, device.startswith("cuda"),
        )
    return prediction.cpu().numpy(), targets, valid


def _pair_geometry(points: np.ndarray) -> dict[str, np.ndarray]:
    """points: (N, 17, 3). Returns, for shoulder and hip, (N,) arrays of
    delta_X, delta_Y, magnitude, angle_degrees, stacked as (N, 2) [shoulder, hip]."""
    delta_x, delta_y, magnitude, angle = [], [], [], []
    for left, right in YAW_INDICES:
        dx = points[:, right, 0] - points[:, left, 0]
        dy = points[:, right, 1] - points[:, left, 1]
        delta_x.append(dx)
        delta_y.append(dy)
        magnitude.append(np.sqrt(dx ** 2 + dy ** 2))
        angle.append(np.arctan2(dy, dx) * 180.0 / np.pi)
    return {
        "delta_x": np.stack(delta_x, axis=1), "delta_y": np.stack(delta_y, axis=1),
        "magnitude": np.stack(magnitude, axis=1), "angle_deg": np.stack(angle, axis=1),
    }


def _pair_valid(valid: np.ndarray) -> np.ndarray:
    return np.stack([valid[:, left] & valid[:, right] for left, right in YAW_INDICES], axis=1)


def _angle_delta_deg(a_deg: np.ndarray, b_deg: np.ndarray) -> np.ndarray:
    a, b = np.radians(a_deg), np.radians(b_deg)
    return np.abs(np.array([_angle_delta(float(x), float(y)) for x, y in zip(a, b)])) * 180.0 / np.pi


def _residuals(pred_geom: dict, gt_geom: dict, pair_valid: np.ndarray) -> dict[str, np.ndarray]:
    delta_x_residual = np.abs(pred_geom["delta_x"] - gt_geom["delta_x"])
    delta_y_residual = np.abs(pred_geom["delta_y"] - gt_geom["delta_y"])
    magnitude_residual = np.abs(pred_geom["magnitude"] - gt_geom["magnitude"])
    angular_error = np.full_like(pred_geom["angle_deg"], np.nan)
    for pair_index in range(pred_geom["angle_deg"].shape[1]):
        pv = pair_valid[:, pair_index] & (pred_geom["magnitude"][:, pair_index] > 1e-6) & (gt_geom["magnitude"][:, pair_index] > 1e-6)
        angular_error[pv, pair_index] = _angle_delta_deg(pred_geom["angle_deg"][pv, pair_index], gt_geom["angle_deg"][pv, pair_index])
    sign_disagreement = np.sign(pred_geom["delta_y"]) != np.sign(gt_geom["delta_y"])
    return {
        "delta_x_residual": delta_x_residual, "delta_y_residual": delta_y_residual,
        "magnitude_residual": magnitude_residual, "angular_error_deg": angular_error,
        "sign_disagreement": sign_disagreement,
    }


def _subset_summary(residuals: dict[str, np.ndarray], pair_valid: np.ndarray, indices: np.ndarray) -> dict[str, Any]:
    output: dict[str, Any] = {"frame_count": int(len(indices))}
    for pair_index, pair_name in enumerate(_PAIR_NAMES):
        pv = pair_valid[indices, pair_index]
        count = int(pv.sum())
        if count == 0:
            output[pair_name] = {"valid_pair_count": 0}
            continue
        angular = residuals["angular_error_deg"][indices, pair_index][pv]
        angular = angular[np.isfinite(angular)]
        output[pair_name] = {
            "valid_pair_count": count,
            "delta_x_abs_residual_mean": float(residuals["delta_x_residual"][indices, pair_index][pv].mean()),
            "delta_y_abs_residual_mean": float(residuals["delta_y_residual"][indices, pair_index][pv].mean()),
            "magnitude_residual_mean": float(residuals["magnitude_residual"][indices, pair_index][pv].mean()),
            "angular_error_mean_deg": float(angular.mean()) if len(angular) else None,
            "sign_disagreement_rate": float(residuals["sign_disagreement"][indices, pair_index][pv].mean()),
        }
    return output


def _counterfactual(gt_geom: dict, x_source: dict, y_source: dict, pair_valid: np.ndarray, indices: np.ndarray) -> dict[str, Any]:
    """CF = x_source's delta_X + y_source's delta_Y, per pair. Mathematically
    exact for the pair angle (root yaw depends only on that pair's own X/Y);
    reported as a counterfactual angle, not an official evaluator score."""
    cf_dx = x_source["delta_x"]
    cf_dy = y_source["delta_y"]
    cf_angle = np.arctan2(cf_dy, cf_dx) * 180.0 / np.pi
    cf_magnitude = np.sqrt(cf_dx ** 2 + cf_dy ** 2)
    output: dict[str, Any] = {}
    for pair_index, pair_name in enumerate(_PAIR_NAMES):
        pv = pair_valid[indices, pair_index] & (cf_magnitude[indices, pair_index] > 1e-6) & (gt_geom["magnitude"][indices, pair_index] > 1e-6)
        if not pv.any():
            output[pair_name] = {"valid_pair_count": 0}
            continue
        selected = indices[pv]
        error = _angle_delta_deg(cf_angle[selected, pair_index], gt_geom["angle_deg"][selected, pair_index])
        output[pair_name] = {"valid_pair_count": int(pv.sum()), "counterfactual_angular_error_mean_deg": float(error.mean())}
    return output


def _coherence(geom: dict, valid: dict) -> dict[str, np.ndarray]:
    """Shoulder-vs-hip orientation disagreement and forward-depth sign
    agreement, per skeleton (GT/A15/A16 independently)."""
    both_valid = valid[:, 0] & valid[:, 1]
    both_stable = both_valid & (geom["magnitude"][:, 0] > 1e-6) & (geom["magnitude"][:, 1] > 1e-6)
    disagreement_deg = np.full(len(both_valid), np.nan)
    if both_stable.any():
        disagreement_deg[both_stable] = _angle_delta_deg(geom["angle_deg"][both_stable, 0], geom["angle_deg"][both_stable, 1])
    sign_agreement = np.sign(geom["delta_y"][:, 0]) == np.sign(geom["delta_y"][:, 1])
    return {"disagreement_deg": disagreement_deg, "sign_agreement": sign_agreement, "both_valid": both_valid}


def _torso_local_frame(geom: dict, pair_valid: np.ndarray) -> dict[str, np.ndarray]:
    """Diagnostic-only torso local frame: combined right-axis = normalized
    sum of the shoulder and hip unit lateral axes. No training loss, no
    metric renormalization -- geometry only."""
    shoulder_unit = np.stack([geom["delta_x"][:, 0], geom["delta_y"][:, 0]], axis=1)
    hip_unit = np.stack([geom["delta_x"][:, 1], geom["delta_y"][:, 1]], axis=1)
    shoulder_norm = np.linalg.norm(shoulder_unit, axis=1, keepdims=True)
    hip_norm = np.linalg.norm(hip_unit, axis=1, keepdims=True)
    both_stable = pair_valid[:, 0] & pair_valid[:, 1] & (shoulder_norm[:, 0] > 1e-6) & (hip_norm[:, 0] > 1e-6)
    shoulder_unit = np.divide(shoulder_unit, shoulder_norm, out=np.zeros_like(shoulder_unit), where=shoulder_norm > 1e-6)
    hip_unit = np.divide(hip_unit, hip_norm, out=np.zeros_like(hip_unit), where=hip_norm > 1e-6)
    combined = shoulder_unit + hip_unit
    combined_norm = np.linalg.norm(combined, axis=1, keepdims=True)
    frame_stable = both_stable & (combined_norm[:, 0] > 1e-6)
    combined_unit = np.divide(combined, combined_norm, out=np.zeros_like(combined), where=combined_norm > 1e-6)

    def angle_to_frame(unit_vector):
        dot = np.clip((unit_vector * combined_unit).sum(axis=1), -1.0, 1.0)
        return np.degrees(np.arccos(dot))

    shoulder_frame_error = np.where(frame_stable, angle_to_frame(shoulder_unit), np.nan)
    hip_frame_error = np.where(frame_stable, angle_to_frame(hip_unit), np.nan)
    return {"shoulder_frame_error_deg": shoulder_frame_error, "hip_frame_error_deg": hip_frame_error, "frame_stable": frame_stable}


def _sequence_ids(metadata: list[dict[str, Any]]) -> np.ndarray:
    return np.array([meta.get("action") or "unknown" for meta in metadata])


def _run_lengths(flags: np.ndarray, sequence_ids: np.ndarray) -> list[int]:
    runs, current_seq, current_run = [], None, 0
    for flag, seq in zip(flags, sequence_ids):
        if seq != current_seq:
            if current_run:
                runs.append(current_run)
            current_seq, current_run = seq, 0
        if bool(flag):
            current_run += 1
        else:
            if current_run:
                runs.append(current_run)
            current_run = 0
    if current_run:
        runs.append(current_run)
    return runs


def _run_summary(flags: np.ndarray, sequence_ids: np.ndarray) -> dict[str, Any]:
    runs = _run_lengths(flags, sequence_ids)
    array = np.asarray(runs, dtype=np.int64) if runs else np.asarray([0])
    return {
        "run_count": len(runs), "singleton_runs": int((array == 1).sum()) if runs else 0,
        "max_run_length": int(array.max()) if runs else 0, "mean_run_length": float(array.mean()) if runs else 0.0,
        "runs_of_5_or_more": int((array >= 5).sum()) if runs else 0,
        "total_flagged_frames": int(flags.sum()),
    }


def _sign_transitions(sign: np.ndarray, sequence_ids: np.ndarray) -> np.ndarray:
    """Boolean array: True where sign differs from the previous frame WITHIN
    the same sequence (sequence-boundary-safe)."""
    transitions = np.zeros(len(sign), dtype=bool)
    for index in range(1, len(sign)):
        if sequence_ids[index] == sequence_ids[index - 1] and np.isfinite(sign[index]) and np.isfinite(sign[index - 1]):
            transitions[index] = sign[index] != sign[index - 1]
    return transitions


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--a15-checkpoint", required=True, type=Path)
    parser.add_argument("--a16-checkpoint", required=True, type=Path)
    parser.add_argument("--historical-a9-checkpoint", required=True, type=Path)
    parser.add_argument("--holdout", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    torch, nn = _torch()
    dataset = load_dataset(args.holdout)
    metadata = _frame_metadata(dataset)
    sequence_ids = _sequence_ids(metadata)

    a15_prediction, targets, valid = _predict(torch, nn, args.a15_checkpoint, dataset, args.device)
    a16_prediction, targets_check, valid_check = _predict(torch, nn, args.a16_checkpoint, dataset, args.device)
    a9_prediction, targets_check2, valid_check2 = _predict(torch, nn, args.historical_a9_checkpoint, dataset, args.device)
    assert np.array_equal(targets, targets_check) and np.array_equal(valid, valid_check)
    assert np.array_equal(targets, targets_check2) and np.array_equal(valid, valid_check2)

    pair_valid = _pair_valid(valid)
    gt_geom = _pair_geometry(targets)
    a15_geom = _pair_geometry(a15_prediction)
    a16_geom = _pair_geometry(a16_prediction)

    # Frozen hard-set: historical A9's own evaluator ranking, established in
    # docs/18/21, never redefined after observing A16.
    a9_yaw = np.full(len(a9_prediction), np.nan, dtype=np.float64)
    for index, (estimate, reference, frame_valid) in enumerate(zip(a9_prediction, targets, valid)):
        yaw = _root_yaw_error_degrees(estimate, reference, frame_valid)
        if yaw is not None:
            a9_yaw[index] = yaw
    eligible = np.flatnonzero(np.isfinite(a9_yaw))
    order = eligible[np.argsort(-a9_yaw[eligible])]
    top5_count = max(1, (len(order) + 19) // 20)
    top1_count = max(1, (len(order) + 99) // 100)
    hard_top5, hard_top1, non_hard = order[:top5_count], order[:top1_count], order[top5_count:]
    hard_cutoff_deg = float(a9_yaw[hard_top5[-1]])
    subsets = {"all_eligible": eligible, "hard_top5pct": hard_top5, "hard_top1pct": hard_top1, "non_hard": non_hard}

    a15_residuals = _residuals(a15_geom, gt_geom, pair_valid)
    a16_residuals = _residuals(a16_geom, gt_geom, pair_valid)

    # Section 4
    section4 = {
        "a15": {name: _subset_summary(a15_residuals, pair_valid, indices) for name, indices in subsets.items()},
        "a16": {name: _subset_summary(a16_residuals, pair_valid, indices) for name, indices in subsets.items()},
    }
    for sequence in _FIXED_SEQUENCES:
        indices = np.array([i for i, s in enumerate(sequence_ids) if sequence in s])
        if len(indices) == 0:
            continue
        section4.setdefault("sequences", {})[sequence] = {
            "a15": _subset_summary(a15_residuals, pair_valid, indices),
            "a16": _subset_summary(a16_residuals, pair_valid, indices),
        }

    # Section 5: counterfactual substitution
    section5 = {
        "cf_x15_y16": {name: _counterfactual(gt_geom, a15_geom, a16_geom, pair_valid, indices) for name, indices in subsets.items()},
        "cf_x16_y15": {name: _counterfactual(gt_geom, a16_geom, a15_geom, pair_valid, indices) for name, indices in subsets.items()},
        "a15_actual": {name: _subset_summary(a15_residuals, pair_valid, indices) for name, indices in subsets.items()},
        "a16_actual": {name: _subset_summary(a16_residuals, pair_valid, indices) for name, indices in subsets.items()},
    }

    # Section 6: shoulder/hip coherence
    gt_coherence = _coherence(gt_geom, pair_valid)
    a15_coherence = _coherence(a15_geom, pair_valid)
    a16_coherence = _coherence(a16_geom, pair_valid)

    def coherence_summary(coherence, indices):
        disagreement = coherence["disagreement_deg"][indices]
        disagreement = disagreement[np.isfinite(disagreement)]
        sign_agree = coherence["sign_agreement"][indices][coherence["both_valid"][indices]]
        return {
            "mean_disagreement_deg": float(disagreement.mean()) if len(disagreement) else None,
            "forward_depth_sign_agreement_rate": float(sign_agree.mean()) if len(sign_agree) else None,
        }

    disagreement_flag = a16_coherence["disagreement_deg"] >= _COHERENCE_DISAGREEMENT_DEG
    section6 = {
        "gt": {name: coherence_summary(gt_coherence, indices) for name, indices in subsets.items()},
        "a15": {name: coherence_summary(a15_coherence, indices) for name, indices in subsets.items()},
        "a16": {name: coherence_summary(a16_coherence, indices) for name, indices in subsets.items()},
        "a16_disagreement_ge_20deg_run_summary": _run_summary(disagreement_flag, sequence_ids),
        "a15_disagreement_ge_20deg_run_summary": _run_summary(a15_coherence["disagreement_deg"] >= _COHERENCE_DISAGREEMENT_DEG, sequence_ids),
    }

    # Section 7: diagnostic torso local frame
    gt_frame = _torso_local_frame(gt_geom, pair_valid)
    a15_frame = _torso_local_frame(a15_geom, pair_valid)
    a16_frame = _torso_local_frame(a16_geom, pair_valid)

    def frame_summary(frame, indices):
        shoulder = frame["shoulder_frame_error_deg"][indices]
        hip = frame["hip_frame_error_deg"][indices]
        shoulder, hip = shoulder[np.isfinite(shoulder)], hip[np.isfinite(hip)]
        return {
            "shoulder_vs_frame_error_mean_deg": float(shoulder.mean()) if len(shoulder) else None,
            "hip_vs_frame_error_mean_deg": float(hip.mean()) if len(hip) else None,
        }

    section7 = {
        "gt": {name: frame_summary(gt_frame, indices) for name, indices in subsets.items()},
        "a15": {name: frame_summary(a15_frame, indices) for name, indices in subsets.items()},
        "a16": {name: frame_summary(a16_frame, indices) for name, indices in subsets.items()},
    }

    # Section 8: error migration, frozen A15-vs-threshold and A15-vs-A16
    a15_yaw = np.full(len(a15_prediction), np.nan, dtype=np.float64)
    a16_yaw = np.full(len(a16_prediction), np.nan, dtype=np.float64)
    for index in range(len(a15_prediction)):
        y15 = _root_yaw_error_degrees(a15_prediction[index], targets[index], valid[index])
        y16 = _root_yaw_error_degrees(a16_prediction[index], targets[index], valid[index])
        if y15 is not None:
            a15_yaw[index] = y15
        if y16 is not None:
            a16_yaw[index] = y16
    both_valid_yaw = np.isfinite(a15_yaw) & np.isfinite(a16_yaw)
    a15_bad = both_valid_yaw & (a15_yaw >= hard_cutoff_deg)
    a15_good = both_valid_yaw & (a15_yaw < hard_cutoff_deg)
    a16_bad = both_valid_yaw & (a16_yaw >= hard_cutoff_deg)
    a16_good = both_valid_yaw & (a16_yaw < hard_cutoff_deg)
    categories = {
        "previously_bad_improved": a15_bad & a16_good,
        "previously_bad_worse": a15_bad & a16_bad,
        "previously_good_remains_good": a15_good & a16_good,
        "previously_good_newly_bad": a15_good & a16_bad,
    }

    def category_report(mask):
        indices = np.flatnonzero(mask)
        if len(indices) == 0:
            return {"frame_count": 0}
        dx15 = a15_residuals["delta_x_residual"][indices].mean(axis=0).tolist()
        dx16 = a16_residuals["delta_x_residual"][indices].mean(axis=0).tolist()
        dy15 = a15_residuals["delta_y_residual"][indices].mean(axis=0).tolist()
        dy16 = a16_residuals["delta_y_residual"][indices].mean(axis=0).tolist()
        coherence15 = a15_coherence["disagreement_deg"][indices]
        coherence16 = a16_coherence["disagreement_deg"][indices]
        frame15 = np.nanmean([a15_frame["shoulder_frame_error_deg"][indices], a15_frame["hip_frame_error_deg"][indices]])
        frame16 = np.nanmean([a16_frame["shoulder_frame_error_deg"][indices], a16_frame["hip_frame_error_deg"][indices]])
        return {
            "frame_count": int(len(indices)),
            "delta_x_residual_change_shoulder_hip": [dx16[i] - dx15[i] for i in range(2)],
            "delta_y_residual_change_shoulder_hip": [dy16[i] - dy15[i] for i in range(2)],
            "coherence_disagreement_change_deg": float(np.nanmean(coherence16) - np.nanmean(coherence15)),
            "torso_frame_error_change_deg": float(frame16 - frame15) if np.isfinite(frame15) and np.isfinite(frame16) else None,
        }

    section8 = {"hard_cutoff_deg_reused": hard_cutoff_deg, "categories": {name: category_report(mask) for name, mask in categories.items()}}

    # Section 9: temporal forward-depth sign-transition analysis
    def transition_report(geom, pv):
        report = {}
        for pair_index, pair_name in enumerate(_PAIR_NAMES):
            gt_sign = np.sign(gt_geom["delta_y"][:, pair_index])
            pred_sign = np.sign(geom["delta_y"][:, pair_index])
            valid_pair = pv[:, pair_index]
            gt_transitions = _sign_transitions(np.where(valid_pair, gt_sign, np.nan), sequence_ids)
            pred_transitions = _sign_transitions(np.where(valid_pair, pred_sign, np.nan), sequence_ids)
            disagreement = (gt_sign != pred_sign) & valid_pair
            missed_gt_transitions = gt_transitions & ~pred_transitions & valid_pair
            false_transitions = pred_transitions & ~gt_transitions & valid_pair
            report[pair_name] = {
                "gt_transition_count": int(gt_transitions.sum()),
                "pred_transition_count": int(pred_transitions.sum()),
                "missed_gt_transition_count": int(missed_gt_transitions.sum()),
                "false_transition_count": int(false_transitions.sum()),
                "sign_disagreement_run_summary": _run_summary(disagreement, sequence_ids),
            }
        return report

    section9 = {"a15": transition_report(a15_geom, pair_valid), "a16": transition_report(a16_geom, pair_valid)}

    report = {
        "holdout": str(args.holdout), "frame_count": len(targets), "eligible_frame_count": int(len(eligible)),
        "hard_top5pct_yaw_cutoff_degrees": hard_cutoff_deg,
        "section4_xy_decomposition": section4,
        "section5_counterfactual_substitution": section5,
        "section6_shoulder_hip_coherence": section6,
        "section7_torso_local_frame": section7,
        "section8_error_migration": section8,
        "section9_temporal_sign_transitions": section9,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(args.out), "eligible_frame_count": report["eligible_frame_count"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
