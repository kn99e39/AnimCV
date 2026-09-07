#!/usr/bin/env python3
"""Replay stored Geometry-Core predictions through explicit branch constraints.

docs/33: the hypothesis under test is that SignState should not be a learned
pose feature but an explicit constraint applied to the continuous pose the
Geometry Core already produced. Nothing here trains, fine-tunes or re-runs a
model. A stored prediction array is read from disk, the oracle SignState is
read from the same bank the prediction was produced against, and
``framepose.branch_constraints`` -- parameter-free, RGB-free, GT-magnitude-free
-- is applied in output space. Every variant is then scored by the SAME
``evaluate_predictions`` used for the trained candidates, so a constraint
result and a learned-conditioning result are directly comparable.

The wrong-sign control is two endpoints only (oracle, opposite-oracle). It
answers "does the constraint follow the sign it is given, including into
error?" -- not "what accuracy does a sensor need", which no amount of replay
can establish.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

from common.serialization import write_json
from framepose.bank import load_bank
from framepose.branch_constraints import (
    APPLICATION_ORDER, BRANCH_CONSTRAINT_SCHEMA, CORRECTED, UNRESOLVED,
    apply_branch_constraints_batch, coverage,
)
from framepose.contract import JOINT_NAMES
from framepose.evaluate import evaluate_predictions
from framepose.signs import SIGN_FIELD_NAMES, UNKNOWN, mask_fields, oracle_sign_states, sign_state


SCHEMA = "animcv_frame_pose_branch_constraint_replay_v1"

_HINGE = [name for name in APPLICATION_ORDER if name.endswith("_forward_bend")]

# The variant set docs/33 Section 11 asks for. torso_facing is deliberately
# absent: this module declares no correction for it, and a variant that
# silently ignored it would misreport its own coverage.
VARIANTS: dict[str, list[str]] = {
    "C0": [],
    "C_HIP": ["hip_forward_depth"],
    "C_BILATERAL": ["shoulder_forward_depth", "hip_forward_depth"],
    "C_HINGE_ALL": list(_HINGE),
    "C_LEFT_ELBOW": ["left_elbow_forward_bend"],
    "C_RIGHT_ELBOW": ["right_elbow_forward_bend"],
    "C_LEFT_KNEE": ["left_knee_forward_bend"],
    "C_RIGHT_KNEE": ["right_knee_forward_bend"],
}

# Metrics that speak directly about the branch the constraint enforces, and
# metrics that only guard against collateral damage. Reported apart, per the
# architecture's evidence-tier rule.
PRIMARY = ("hinge_flip_rate", "hinge_direction_mae_degrees", "root_yaw_error_degrees",
           "shoulder_forward_depth_residual_mm", "hip_forward_depth_residual_mm")
GUARDRAIL = ("mpjpe_mm", "pa_mpjpe_mm", "max_joint_error_mm")


def _summary(evaluation: dict[str, Any]) -> dict[str, Any]:
    aggregate = evaluation["aggregate"]
    def read(name):
        entry = aggregate.get(name)
        return None if entry is None else {"mean": entry["mean"], "median": entry["median"],
                                           "p90": entry["p90"], "count": entry["count"]}
    return {"primary": {name: read(name) for name in PRIMARY},
            "guardrail": {name: read(name) for name in GUARDRAIL},
            "sign_agreement": evaluation.get("sign_agreement")}


def _requested(oracle: np.ndarray, fields: list[str], *, invert: bool) -> np.ndarray:
    """The sign state handed to the constraint operator.

    Only the variant's own fields carry a value; everything else is UNKNOWN, so
    a variant can never be credited with a field it did not ask for. `invert`
    flips every non-degenerate requested branch to build the wrong-sign
    endpoint -- UNKNOWN stays UNKNOWN, because "unknown" is not a branch that
    can be wrong.
    """
    requested = mask_fields(oracle, fields).astype(np.int64)
    if invert:
        known = requested != UNKNOWN
        requested[known] = -requested[known]
    return requested


def _review_rows(bank, positions, original, corrected, reports, requested, limit: int):
    """Real frames a human can check, not aggregate numbers."""
    # Every frame the constraint actually touched is a candidate, and the
    # export is an even subsample of those. Taking the first N instead draws
    # the whole export from whichever sequence happens to sort first.
    qualifying = [order for order in range(len(positions))
                  if any(entry["outcome"] in (CORRECTED, UNRESOLVED)
                         for entry in reports[order]["fields"].values())]
    if len(qualifying) > limit:
        picks = np.unique(np.floor(np.arange(limit) * (len(qualifying) / limit)).astype(np.int64))
        qualifying = [qualifying[int(index)] for index in picks]

    rows = []
    for order in qualifying:
        report = reports[order]
        touched = {field: entry for field, entry in report["fields"].items()
                   if entry["outcome"] in (CORRECTED, UNRESOLVED)}
        if not touched:
            continue
        sample = bank.samples[int(positions[order])]
        delta = (corrected[order] - original[order]) * 1000.0
        moved = np.flatnonzero(np.linalg.norm(delta, axis=-1) > 0)
        rows.append({
            "sample_id": sample.sample_id,
            "source": sample.source,
            "sequence_id": sample.sequence_id,
            "frame_index": int(sample.frame_index),
            "image_reference": (sample.image_reference.relative_path
                                if sample.image_reference is not None else None),
            "bank_position": int(positions[order]),
            "fields": {field: {
                "outcome": entry["outcome"],
                "requested_sign": int(requested[order][SIGN_FIELD_NAMES.index(field)]),
                "read_back_before": entry.get("read_back_before"),
                "read_back_after": entry.get("read_back_after", entry.get("read_back_after_attempt")),
                "reason": entry.get("reason"),
                "depth_delta_m": entry.get("depth_delta_m"),
            } for field, entry in touched.items()},
            "moved_joints": {JOINT_NAMES[joint]: [round(float(value), 3) for value in delta[joint]]
                             for joint in moved},
            "original_xyz_m": {JOINT_NAMES[joint]: [round(float(value), 5) for value in original[order][joint]]
                               for joint in moved},
            "corrected_xyz_m": {JOINT_NAMES[joint]: [round(float(value), 5) for value in corrected[order][joint]]
                                for joint in moved},
        })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Branch-constraint replay over stored predictions")
    parser.add_argument("--bank", required=True, type=Path)
    parser.add_argument("--prediction", required=True, type=Path,
                        help="stored prediction_<split>.npy from the Geometry Core candidate")
    parser.add_argument("--source-candidate", required=True,
                        help="which candidate produced --prediction (recorded, never inferred)")
    parser.add_argument("--split", default="test")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--review-frames", type=int, default=40)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    bank = load_bank(args.bank)
    positions = bank.indices(args.split)
    original = np.load(args.prediction).astype(np.float64)
    if original.shape != (len(positions), len(JOINT_NAMES), 3):
        raise ValueError(
            f"stored prediction {original.shape} does not match the {args.split} split "
            f"({len(positions)} frames); the prediction and the bank must be the same run")

    valid = bank.arrays["target_valid"][positions]
    oracle = oracle_sign_states(bank.arrays["target_3d"], bank.arrays["target_valid"])[positions]

    # What the Core produced on its own, before any constraint: the baseline
    # every variant is compared against and the C0 no-op identity check.
    base_eval = evaluate_predictions(bank, positions, original, candidate=args.source_candidate)

    report: dict[str, Any] = {
        "schema": SCHEMA,
        "constraint_schema": BRANCH_CONSTRAINT_SCHEMA,
        "bank_content_digest": bank.content_digest(),
        "split": args.split,
        "frames": int(len(positions)),
        "source_candidate": args.source_candidate,
        "source_prediction": str(args.prediction),
        "method": ("stored predictions only; no training, no fine-tuning, no RGB, no VLM, no GT XYZ "
                   "magnitude; the requested SignState is the oracle masked to each variant's own "
                   "fields; corrections are the parameter-free closed forms in "
                   "framepose.branch_constraints"),
        "sign_source": "oracle (upper bound on any sign sensor; not itself a sensor result)",
        "regime": "benchmark_detector_observation",
        "baseline": _summary(base_eval),
        "variants": {},
        "wrong_sign_control": {},
    }

    review: dict[str, Any] = {"schema": SCHEMA, "split": args.split, "variants": {}}

    for name, fields in VARIANTS.items():
        requested = _requested(oracle, fields, invert=False)
        corrected, reports = apply_branch_constraints_batch(original, valid, requested, fields=fields or None)
        if not fields:
            # C0 must be bit-identical: an operator that is not an exact no-op
            # when nothing is requested contaminates every other variant.
            if not np.array_equal(corrected, original):
                raise ValueError("C0 changed the prediction; the operator is not an exact no-op")
        evaluation = evaluate_predictions(bank, positions, corrected, candidate=f"{args.source_candidate}+{name}")
        cover = coverage(reports)
        satisfied = {
            field: float(np.mean([report["requested_branch_satisfied"][field] is True
                                  for report in reports
                                  if report["requested_branch_satisfied"][field] is not None]))
            if any(report["requested_branch_satisfied"][field] is not None for report in reports) else None
            for field in fields}
        displacement = np.linalg.norm(corrected - original, axis=-1) * 1000.0
        report["variants"][name] = {
            "fields": fields,
            "constrained_frames": int(sum(
                any(entry["outcome"] == CORRECTED for entry in item["fields"].values()) for item in reports)),
            "coverage": cover["fields"] if fields else {},
            "requested_branch_satisfied_rate": satisfied,
            "displacement_mm": {
                "mean_over_all_joints": float(displacement.mean()),
                "mean_over_moved_joints": float(displacement[displacement > 0].mean()) if (displacement > 0).any() else 0.0,
                "p99_moved": float(np.percentile(displacement[displacement > 0], 99)) if (displacement > 0).any() else 0.0,
                "max": float(displacement.max()),
            },
            "evaluation": _summary(evaluation),
        }
        if fields:
            review["variants"][name] = _review_rows(bank, positions, original, corrected, reports,
                                                    requested, args.review_frames)
            np.save(args.out / f"prediction_{args.split}_{name}.npy", corrected.astype(np.float32))

        # Wrong-sign endpoint: the same operator, the opposite branch. No
        # sweep -- a percentage sweep would invent a sensor accuracy curve that
        # no evidence here supports.
        if fields:
            wrong_requested = _requested(oracle, fields, invert=True)
            wrong, wrong_reports = apply_branch_constraints_batch(
                original, valid, wrong_requested, fields=fields)
            wrong_eval = evaluate_predictions(bank, positions, wrong,
                                              candidate=f"{args.source_candidate}+{name}+opposite_sign")
            report["wrong_sign_control"][name] = {
                "endpoints": ["oracle", "opposite_oracle"],
                "corrected_frames": int(sum(
                    any(entry["outcome"] == CORRECTED for entry in item["fields"].values())
                    for item in wrong_reports)),
                "evaluation": _summary(wrong_eval),
            }

    write_json(args.out / "branch_constraint_replay.json", report)
    write_json(args.out / "branch_constraint_review.json", review)

    def show(label, summary):
        primary, guard = summary["primary"], summary["guardrail"]
        print("   %-16s %8s %10s %9s %9s" % (
            label,
            "%.4f" % primary["hinge_flip_rate"]["mean"] if primary["hinge_flip_rate"] else "-",
            "%.3f" % primary["hinge_direction_mae_degrees"]["mean"] if primary["hinge_direction_mae_degrees"] else "-",
            "%.3f" % guard["mpjpe_mm"]["mean"], "%.3f" % guard["pa_mpjpe_mm"]["mean"]))

    print(f"\n== branch-constraint replay over {args.source_candidate} ({len(positions)} {args.split} frames)")
    print("   %-16s %8s %10s %9s %9s" % ("variant", "flip", "hingeMAE", "MPJPE", "PA-MPJPE"))
    show("baseline", report["baseline"])
    for name in VARIANTS:
        show(name, report["variants"][name]["evaluation"])
    print("\n   wrong-sign endpoint (opposite oracle)")
    for name, value in report["wrong_sign_control"].items():
        show(name, value["evaluation"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
