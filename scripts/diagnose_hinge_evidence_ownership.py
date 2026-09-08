#!/usr/bin/env python3
"""After the hidden depth branch is correct, who owns the residual hinge error?

docs/37. Three candidate owners for a residual full-3D hinge flip:

    hidden evidence          the advisor's one depth bit is not enough
    visible 2D topology      the complementary in-image-plane side, which
                             Geometry Observation may already see
    continuous axis geometry the Geometry Core simply put the limb somewhere else

This separates them on the accepted stored `O_BILATERAL` prediction with the
MINIMUM_NORM oracle-depth hinge correction. Nothing is trained, no model runs,
no RGB or VLM is touched, and SignState is not expanded.

The 2D side predicate uses only `input_2d`/`input_valid`. A benchmark detector's
2D is NOT an oracle, and is never treated as one here: it is one observation
source being tested against the target's own screen-side component.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from common.canonical_pose import FORWARD_DEPTH_AXIS, JOINT_INDEX
from common.serialization import write_json
from framepose.bank import load_bank
from framepose.branch_constraints import (
    CORRECTED, MINIMUM_NORM, apply_branch_constraints_batch,
)
from framepose.contract import JOINT_NAMES
from framepose.hinge_plane import (
    HINGE_PLANE_SCHEMA, IMAGE_TO_CANONICAL, axis_transport, bend_components, observed_screen_side,
)
from framepose.replay_provenance import verify_source_identity
from framepose.signs import (
    HINGE_CHAINS_BY_JOINT, SIGN_FIELD_NAMES, UNKNOWN, mask_fields, oracle_sign_states, sign_state,
)

SCHEMA = "animcv_frame_pose_hinge_evidence_ownership_v1"

HINGE_FIELDS = tuple(name for name in SIGN_FIELD_NAMES if name.endswith("_forward_bend"))
HINGE_JOINTS = tuple(name[: -len("_forward_bend")] for name in HINGE_FIELDS)


def _quantiles(values: np.ndarray) -> dict[str, Any]:
    values = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if not values.size:
        return {"count": 0}
    return {"count": int(values.size), "mean": float(values.mean()),
            **{f"p{q}": float(np.percentile(values, q)) for q in (50, 90, 99)},
            "max": float(values.max())}


def _confusion(observed: list[int], reference: list[int]) -> dict[str, Any]:
    """Balanced accuracy, so a class imbalance cannot flatter the predicate."""
    pairs = [(o, r) for o, r in zip(observed, reference) if o != 0 and r != 0]
    if not pairs:
        return {"valid": 0}
    matrix = Counter(pairs)
    agree = sum(count for (o, r), count in matrix.items() if o == r)
    per_class = {}
    for value in (1, -1):
        total = sum(count for (o, r), count in matrix.items() if r == value)
        right = matrix.get((value, value), 0)
        per_class[str(value)] = {"reference_count": total,
                                 "recall": (right / total) if total else None}
    recalls = [entry["recall"] for entry in per_class.values() if entry["recall"] is not None]
    return {
        "valid": len(pairs),
        "agreement": agree / len(pairs),
        "balanced_accuracy": float(np.mean(recalls)) if recalls else None,
        "positive_observed": sum(1 for o, _ in pairs if o > 0),
        "negative_observed": sum(1 for o, _ in pairs if o < 0),
        "confusion": {f"observed{o:+d}|target{r:+d}": count for (o, r), count in sorted(matrix.items())},
        "per_class_recall": per_class,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Hinge evidence-ownership diagnostic")
    parser.add_argument("--bank", required=True, type=Path)
    parser.add_argument("--source", required=True, help="LABEL=CANDIDATE:PREDICTION[:EVALUATION]")
    parser.add_argument("--split", default="test")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--review-frames", type=int, default=25)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    bank = load_bank(args.bank)
    positions = bank.indices(args.split)
    valid = bank.arrays["target_valid"][positions]
    observation = bank.arrays["input_2d"][positions].astype(np.float64)
    observed_valid = bank.arrays["input_valid"][positions]
    targets = bank.arrays["target_3d"][positions].astype(np.float64)
    oracle = oracle_sign_states(bank.arrays["target_3d"], bank.arrays["target_valid"])[positions]
    requested = mask_fields(oracle, list(HINGE_FIELDS)).astype(np.int64)

    label, rest = args.source.split("=", 1)
    parts = rest.split(":")
    candidate, prediction = parts[0], Path(parts[1])
    evaluation = Path(parts[2]) if len(parts) > 2 and parts[2] else None
    identity = verify_source_identity(prediction=prediction, evaluation=evaluation, bank=bank,
                                      split=args.split, candidate=candidate,
                                      frames=len(positions), joints=len(JOINT_NAMES))

    before = np.load(prediction).astype(np.float64)
    after, reports = apply_branch_constraints_batch(before, valid, requested,
                                                    fields=list(HINGE_FIELDS),
                                                    hinge_write_policy=MINIMUM_NORM)
    after_signs = np.stack([sign_state(after[order], valid[order]) for order in range(len(after))])

    per_field: dict[str, Any] = {}
    review: dict[str, list[dict[str, Any]]] = {
        "depth_correct_but_screen_side_wrong": [],
        "both_sides_correct_but_still_flipped": [],
        "large_axis_angle_error": [],
        "large_minimum_norm_xz_movement": [],
        "observed_2d_side_disagrees_with_target": [],
    }
    ownership_totals = Counter()

    for column, field in enumerate(HINGE_FIELDS):
        joint = HINGE_JOINTS[column]
        chain = HINGE_CHAINS_BY_JOINT[joint]
        index = SIGN_FIELD_NAMES.index(field)
        chain_indices = [JOINT_INDEX[name] for name in chain]
        middle = JOINT_INDEX[chain[1]]

        residual: list[dict[str, Any]] = []
        all_cross, flip_cross = [], []
        all_target_screen_magnitude, flip_target_screen_magnitude = [], []
        all_observed, all_target_screen = [], []
        flip_observed, flip_target_screen = [], []
        type1_observed, type1_target = [], []
        type2_observed, type2_target = [], []
        axis_angles, axis_normalized_errors, historical_errors = [], [], []
        classification = Counter()
        movement: dict[str, Any] = {}

        for order in range(len(positions)):
            if not all(bool(valid[order][i]) for i in chain_indices):
                continue
            predicted_components = bend_components(after[order], chain)
            target_components = bend_components(targets[order], chain)
            observed = observed_screen_side(observation[order], observed_valid[order], chain)
            if target_components is not None:
                all_observed.append(observed["side"] if observed["resolved"] else 0)
                all_target_screen.append(int(np.sign(target_components["c_screen"])))
                all_target_screen_magnitude.append(abs(target_components["c_screen"]) * 1000.0)
                if observed["cross"] is not None:
                    all_cross.append(abs(observed["cross"]))

            transport = axis_transport(after[order], targets[order], chain)
            if not transport.get("resolved"):
                continue
            axis_angles.append(transport["axis_angle_degrees"])
            historical_errors.append(transport["historical_error_degrees"])
            axis_normalized_errors.append(transport["axis_normalized_error_degrees"])

            entry = reports[order]["fields"][field]
            want = int(requested[order, index])
            # docs/34's TYPE split, recomputed on the pre-correction pose.
            if entry["outcome"] == CORRECTED:
                before_components = bend_components(before[order], chain)
                before_transport = axis_transport(before[order], targets[order], chain)
                if before_transport.get("resolved") and target_components is not None:
                    pair = (observed["side"] if observed["resolved"] else 0,
                            int(np.sign(target_components["c_screen"])))
                    if before_transport["historical_flipped"]:
                        type1_observed.append(pair[0]); type1_target.append(pair[1])
                    else:
                        type2_observed.append(pair[0]); type2_target.append(pair[1])

            if not transport["historical_flipped"]:
                continue
            # ---- a residual full-3D flip: attribute it -----------------------
            depth_ok = (want != UNKNOWN and int(after_signs[order, index]) == want)
            screen_predicted = (int(np.sign(predicted_components["c_screen"]))
                                if predicted_components is not None else 0)
            screen_target = (int(np.sign(target_components["c_screen"]))
                             if target_components is not None else 0)
            screen_mismatch = bool(screen_predicted and screen_target and screen_predicted != screen_target)
            axis_normalized_flip = transport["axis_normalized_flipped"]

            # "The axis is substantially different" is operationalized without a
            # threshold: the axis is implicated exactly when aligning the two
            # limb axes is enough to remove the >90-degree flip.
            axis_implicated = not axis_normalized_flip
            if screen_mismatch and not axis_implicated:
                kind = "A_screen_side_mismatch"
            elif axis_implicated and not screen_mismatch:
                kind = "B_axis_geometry_mismatch"
            elif screen_mismatch and axis_implicated:
                kind = "C_both"
            else:
                kind = "D_unexplained"
            classification[kind] += 1
            ownership_totals[kind] += 1

            flip_observed.append(observed["side"] if observed["resolved"] else 0)
            flip_target_screen.append(screen_target)
            if observed["cross"] is not None:
                flip_cross.append(abs(observed["cross"]))
            if target_components is not None:
                flip_target_screen_magnitude.append(abs(target_components["c_screen"]) * 1000.0)
            sample = bank.samples[int(positions[order])]
            record = {
                "field": field, "class": kind,
                "sample_id": sample.sample_id, "sequence_id": sample.sequence_id,
                "frame_index": int(sample.frame_index),
                "image_reference": (sample.image_reference.relative_path
                                    if sample.image_reference is not None else None),
                "chain": list(chain),
                "requested_depth_sign": want,
                "final_depth_sign": int(after_signs[order, index]),
                "depth_side_correct": bool(depth_ok),
                "predicted_c_screen_sign": screen_predicted,
                "target_c_screen_sign": screen_target,
                "observed_2d_side": observed["side"],
                "observed_2d_resolved": observed["resolved"],
                "observed_2d_cross": observed["cross"],
                "axis_angle_degrees": transport["axis_angle_degrees"],
                "historical_hinge_error_degrees": transport["historical_error_degrees"],
                "axis_normalized_hinge_error_degrees": transport["axis_normalized_error_degrees"],
                "axis_normalized_flipped": bool(axis_normalized_flip),
                "axis_implicated": bool(axis_implicated),
                "screen_implicated": bool(screen_mismatch),
                "source_middle_xyz_m": [round(float(v), 5) for v in before[order, middle]],
                "corrected_middle_xyz_m": [round(float(v), 5) for v in after[order, middle]],
                "input_2d_chain": [[round(float(v), 5) for v in observation[order][i][:2]]
                                   for i in chain_indices],
            }
            residual.append(record)
            if depth_ok and screen_mismatch and len(review["depth_correct_but_screen_side_wrong"]) < args.review_frames:
                review["depth_correct_but_screen_side_wrong"].append(record)
            if depth_ok and not screen_mismatch and axis_normalized_flip and len(review["both_sides_correct_but_still_flipped"]) < args.review_frames:
                review["both_sides_correct_but_still_flipped"].append(record)
            if len(review["large_axis_angle_error"]) < args.review_frames and transport["axis_angle_degrees"] > 45.0:
                review["large_axis_angle_error"].append(record)
            if (observed["resolved"] and screen_target and observed["side"] != screen_target
                    and len(review["observed_2d_side_disagrees_with_target"]) < args.review_frames):
                review["observed_2d_side_disagrees_with_target"].append(record)

        # Minimum-norm displacement, per chain, as an ownership question.
        delta = after[:, middle] - before[:, middle]
        moved = np.linalg.norm(delta, axis=-1) > 0
        if moved.any():
            xz = np.linalg.norm(delta[moved][:, [0, 2]], axis=-1) * 1000.0
            movement = {
                "corrections": int(moved.sum()),
                "abs_delta_x_mm": _quantiles(np.abs(delta[moved][:, 0]) * 1000.0),
                "abs_delta_y_mm": _quantiles(np.abs(delta[moved][:, FORWARD_DEPTH_AXIS]) * 1000.0),
                "abs_delta_z_mm": _quantiles(np.abs(delta[moved][:, 2]) * 1000.0),
                "xz_norm_mm": _quantiles(xz),
                "correction_norm_mm": _quantiles(np.linalg.norm(delta[moved], axis=-1) * 1000.0),
                "target_xz_error_before_mm": _quantiles(
                    np.linalg.norm((before[:, middle] - targets[:, middle])[moved][:, [0, 2]], axis=-1) * 1000.0),
                "target_xz_error_after_mm": _quantiles(
                    np.linalg.norm((after[:, middle] - targets[:, middle])[moved][:, [0, 2]], axis=-1) * 1000.0),
            }
            worst = np.argsort(-xz)[:args.review_frames]
            for rank in worst:
                order = int(np.flatnonzero(moved)[rank])
                sample = bank.samples[int(positions[order])]
                if len(review["large_minimum_norm_xz_movement"]) >= args.review_frames * 4:
                    break
                review["large_minimum_norm_xz_movement"].append({
                    "field": field, "sample_id": sample.sample_id,
                    "frame_index": int(sample.frame_index),
                    "image_reference": (sample.image_reference.relative_path
                                        if sample.image_reference is not None else None),
                    "delta_xyz_mm": [round(float(v) * 1000.0, 3) for v in delta[order]],
                    "xz_norm_mm": float(xz[rank]),
                    "source_middle_xyz_m": [round(float(v), 5) for v in before[order, middle]],
                    "corrected_middle_xyz_m": [round(float(v), 5) for v in after[order, middle]],
                })

        per_field[field] = {
            "chain": list(chain),
            "residual_full_3d_flips": len(residual),
            "classification": dict(sorted(classification.items())),
            "axis_angle_degrees": _quantiles(np.array(axis_angles)),
            "historical_hinge_error_degrees": _quantiles(np.array(historical_errors)),
            "axis_normalized_hinge_error_degrees": _quantiles(np.array(axis_normalized_errors)),
            "historical_flip_rate": float(np.mean([e > 90.0 for e in historical_errors])) if historical_errors else None,
            "axis_normalized_flip_rate": float(np.mean([e > 90.0 for e in axis_normalized_errors])) if axis_normalized_errors else None,
            "observed_screen_side_vs_target": {
                "all_frames": _confusion(all_observed, all_target_screen),
                "residual_flip_frames": _confusion(flip_observed, flip_target_screen),
                "type_1_corrections": _confusion(type1_observed, type1_target),
                "type_2_corrections": _confusion(type2_observed, type2_target),
            },
            "unresolved_observed_side": int(sum(1 for value in all_observed if value == 0)),
            "observed_line_side_magnitude": {
                "all_frames": _quantiles(np.array(all_cross)),
                "residual_flip_frames": _quantiles(np.array(flip_cross)),
            },
            "target_c_screen_magnitude_mm": {
                "all_frames": _quantiles(np.array(all_target_screen_magnitude)),
                "residual_flip_frames": _quantiles(np.array(flip_target_screen_magnitude)),
            },
            "residual_flip_records": residual,
            "minimum_norm_movement": movement,
        }

    report = {
        "schema": SCHEMA,
        "hinge_plane_schema": HINGE_PLANE_SCHEMA,
        "image_to_canonical": IMAGE_TO_CANONICAL,
        "method": ("stored prediction + MINIMUM_NORM oracle-depth hinge correction; no training, "
                   "no model execution, no RGB, no VLM, no SignState expansion"),
        "hinge_write_policy": MINIMUM_NORM,
        "observation_note": ("input_2d is a benchmark detector observation, not an oracle; it is "
                             "the source under test, never the reference"),
        "provenance": {"bank_index": str(args.bank),
                       "bank_content_digest": bank.content_digest(),
                       "observation_regime": bank.regime(),
                       "split": args.split, "frames": int(len(positions)),
                       "source": identity},
        "ownership_totals": dict(sorted(ownership_totals.items())),
        "fields": per_field,
    }
    write_json(args.out / "hinge_evidence_ownership.json", report)
    write_json(args.out / "hinge_evidence_review.json", {"schema": SCHEMA, "buckets": review})

    print(f"\n== hinge evidence ownership ({len(positions)} {args.split} frames, "
          f"policy={MINIMUM_NORM})")
    print("   residual full-3D flip classification (pooled):", dict(sorted(ownership_totals.items())))
    print("\n   %-26s %8s %8s %10s %10s" % ("field", "flips", "axisFlip%", "histMAE", "axisNormMAE"))
    for field, entry in per_field.items():
        print("   %-26s %8d %8.4f %10.3f %10.3f" % (
            field, entry["residual_full_3d_flips"], entry["axis_normalized_flip_rate"],
            entry["historical_hinge_error_degrees"]["mean"],
            entry["axis_normalized_hinge_error_degrees"]["mean"]))
    print("\n   observed 2D screen side vs target c_screen sign")
    print("   %-26s %10s %10s %10s %10s" % ("field", "all bal.acc", "flip bal.acc", "T1 bal.acc", "T2 bal.acc"))
    for field, entry in per_field.items():
        cells = []
        for key in ("all_frames", "residual_flip_frames", "type_1_corrections", "type_2_corrections"):
            value = entry["observed_screen_side_vs_target"][key]
            cells.append("-" if not value.get("valid") else "%.4f" % (value["balanced_accuracy"] or float("nan")))
        print("   %-26s %10s %10s %10s %10s" % (field, *cells))
    print("\n   minimum-norm displacement, X/Z component (mm)")
    for field, entry in per_field.items():
        m = entry["minimum_norm_movement"]
        if m:
            print("   %-26s n=%4d  p50=%7.2f p90=%7.2f p99=%7.2f max=%8.2f" % (
                field, m["corrections"], m["xz_norm_mm"]["p50"], m["xz_norm_mm"]["p90"],
                m["xz_norm_mm"]["p99"], m["xz_norm_mm"]["max"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
