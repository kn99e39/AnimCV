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
import hashlib
import sys
from pathlib import Path
from typing import Any

import numpy as np

from common.canonical_pose import FORWARD_DEPTH_AXIS, JOINT_INDEX
from common.serialization import write_json
from framepose.bank import load_bank
from framepose.branch_constraints import (
    APPLICATION_ORDER, BRANCH_CONSTRAINT_SCHEMA, CORRECTED, UNRESOLVED,
    apply_branch_constraints_batch, coverage,
)
from framepose.constraint_graph import ANCHOR_ONLY, DEPENDENCY_AWARE, dependency_graph
from framepose.contract import JOINT_NAMES
from framepose.evaluate import evaluate_predictions
from framepose.signs import SIGN_FIELD_NAMES, UNKNOWN, mask_fields, oracle_sign_states, sign_state


SCHEMA = "animcv_frame_pose_branch_constraint_replay_v2"

#: Validity regimes for the correction/read-back path (docs/34 Section 3).
#: docs/33 used `target_valid` throughout, which is GT-side semantics and fine
#: for an oracle upper bound but is not production-observable. `input_valid` is
#: the observable proxy. Neither is tuned; only which mask the operator and the
#: read-back see changes.
VALIDITY_SOURCES = {"V_ORACLE": "target_valid", "V_OBSERVED": "input_valid"}

_HINGE = [name for name in APPLICATION_ORDER if name.endswith("_forward_bend")]

# The variant set docs/33 Section 11 asks for. torso_facing is deliberately
# absent: this module declares no correction for it, and a variant that
# silently ignored it would misreport its own coverage.
#: Each variant declares its fields AND its bilateral write policy, so a
#: historical pair-swap result can never be filed under a dependency-aware name
#: or the reverse.
VARIANTS: dict[str, dict[str, Any]] = {
    "C0": {"fields": [], "policy": ANCHOR_ONLY},
    # docs/33's operator, unchanged, re-run here as the comparison baseline.
    "C_HIP": {"fields": ["hip_forward_depth"], "policy": ANCHOR_ONLY},
    "C_BILATERAL": {"fields": ["shoulder_forward_depth", "hip_forward_depth"],
                    "policy": ANCHOR_ONLY},
    # docs/34's dependency-aware anchor correction.
    "C_HIP_DEP": {"fields": ["hip_forward_depth"], "policy": DEPENDENCY_AWARE},
    "C_BILATERAL_DEP": {"fields": ["shoulder_forward_depth", "hip_forward_depth"],
                        "policy": DEPENDENCY_AWARE},
    # The frozen hinge operator; no policy reaches it. Re-run for the validity
    # control and as a cross-lineage determinism check.
    "C_HINGE_ALL": {"fields": list(_HINGE), "policy": ANCHOR_ONLY},
}

# Metrics that speak directly about the branch the constraint enforces, and
# metrics that only guard against collateral damage. Reported apart, per the
# architecture's evidence-tier rule.
PRIMARY = ("hinge_flip_rate", "hinge_direction_mae_degrees", "root_yaw_error_degrees",
           "shoulder_forward_depth_residual_mm", "hip_forward_depth_residual_mm",
           "shoulder_forward_depth_abs_residual_mm", "hip_forward_depth_abs_residual_mm",
           "shoulder_forward_depth_sign_disagreement", "hip_forward_depth_sign_disagreement",
           "left_elbow_bend_error_degrees", "right_elbow_bend_error_degrees",
           "left_knee_bend_error_degrees", "right_knee_bend_error_degrees")
GUARDRAIL = ("mpjpe_mm", "pa_mpjpe_mm", "max_joint_error_mm")

#: Attachment the bilateral operators do NOT preserve, measured rather than claimed.
ATTACHMENTS = (("thorax", "left_shoulder"), ("thorax", "right_shoulder"),
               ("pelvis", "left_hip"), ("pelvis", "right_hip"))


def _digest(path: Path) -> dict[str, Any]:
    """Bind an artifact to its exact bytes, not to the label a caller passed."""
    data = path.read_bytes()
    return {"path": str(path), "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest()}


def _attachment_change_mm(original: np.ndarray, corrected: np.ndarray,
                          valid: np.ndarray) -> dict[str, Any]:
    """The cost a depth translation of an anchor does NOT avoid: the anchor's
    own attachment to the torso. Reported, never claimed preserved."""
    report: dict[str, Any] = {}
    for parent, child in ATTACHMENTS:
        parent_index, child_index = JOINT_INDEX[parent], JOINT_INDEX[child]
        usable = valid[:, parent_index] & valid[:, child_index]
        if not usable.any():
            report[f"{parent}_to_{child}"] = None
            continue
        before = np.linalg.norm(original[usable, child_index] - original[usable, parent_index], axis=-1)
        after = np.linalg.norm(corrected[usable, child_index] - corrected[usable, parent_index], axis=-1)
        change = (after - before) * 1000.0
        report[f"{parent}_to_{child}"] = {
            "frames": int(usable.sum()),
            "mean_signed_change_mm": float(change.mean()),
            "mean_abs_change_mm": float(np.abs(change).mean()),
            "p95_abs_change_mm": float(np.percentile(np.abs(change), 95)),
            "max_abs_change_mm": float(np.abs(change).max()),
            "mean_original_length_mm": float(before.mean() * 1000.0),
        }
    return report


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
    parser.add_argument("--validity", default="V_ORACLE", choices=sorted(VALIDITY_SOURCES),
                        help="which validity mask the correction and read-back path sees")
    parser.add_argument("--source-evaluation", type=Path, default=None,
                        help="optional evaluation_<split>.json of the source run, recorded by digest")
    parser.add_argument("--variants", default="",
                        help="comma-separated subset of VARIANTS; empty means all")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    bank = load_bank(args.bank)
    positions = bank.indices(args.split)
    original = np.load(args.prediction).astype(np.float64)
    if original.shape != (len(positions), len(JOINT_NAMES), 3):
        raise ValueError(
            f"stored prediction {original.shape} does not match the {args.split} split "
            f"({len(positions)} frames); the prediction and the bank must be the same run")

    # The mask the operator and the read-back see. The oracle SignState itself
    # is always derived from target_valid: it is ground truth by definition, and
    # changing how the REQUEST is built would confound the applicability
    # question with a different requested-sign distribution.
    validity_array = VALIDITY_SOURCES[args.validity]
    valid = bank.arrays[validity_array][positions]
    oracle = oracle_sign_states(bank.arrays["target_3d"], bank.arrays["target_valid"])[positions]

    target_valid = bank.arrays["target_valid"][positions]
    validity_identical = bool(np.array_equal(valid, target_valid))

    selected_variants = ([name.strip() for name in args.variants.split(",") if name.strip()]
                         or list(VARIANTS))
    unknown = [name for name in selected_variants if name not in VARIANTS]
    if unknown:
        raise ValueError(f"unknown variants {unknown}; known: {list(VARIANTS)}")

    # What the Core produced on its own, before any constraint: the baseline
    # every variant is compared against and the C0 no-op identity check.
    base_eval = evaluate_predictions(bank, positions, original, candidate=args.source_candidate)

    report: dict[str, Any] = {
        "schema": SCHEMA,
        "constraint_schema": BRANCH_CONSTRAINT_SCHEMA,
        "constraint_graph": {policy: dependency_graph(policy)
                             for policy in (ANCHOR_ONLY, DEPENDENCY_AWARE)},
        "provenance": {
            "bank_index": str(args.bank),
            "bank_content_digest": bank.content_digest(),
            "observation_regime": bank.regime(),
            "observation_regime_source": "FrameBank.regime() -- derived, never hard-coded",
            "split": args.split,
            "frames": int(len(positions)),
            "source_candidate": args.source_candidate,
            "source_prediction": _digest(args.prediction),
            "source_evaluation": (_digest(args.source_evaluation)
                                  if args.source_evaluation is not None else None),
            "constraint_schema": BRANCH_CONSTRAINT_SCHEMA,
            "constraint_validity_source": validity_array,
            "constraint_validity_regime": args.validity,
            "validity_identical_to_target_valid": validity_identical,
            "sign_request_validity_source": "target_valid (the oracle is ground truth by definition)",
        },
        # Retained at the top level for continuity with the v1 lineage.
        "bank_content_digest": bank.content_digest(),
        "split": args.split,
        "frames": int(len(positions)),
        "source_candidate": args.source_candidate,
        "method": ("stored predictions only; no training, no fine-tuning, no RGB, no VLM, no GT XYZ "
                   "magnitude; the requested SignState is the oracle masked to each variant's own "
                   "fields; corrections are the parameter-free closed forms in "
                   "framepose.branch_constraints"),
        "sign_source": "oracle (upper bound on any sign sensor; not itself a sensor result)",
        "regime": bank.regime(),
        "baseline": _summary(base_eval),
        "variants": {},
        "wrong_sign_control": {},
    }

    review: dict[str, Any] = {"schema": SCHEMA, "split": args.split, "variants": {}}

    for name in selected_variants:
        fields = VARIANTS[name]["fields"]
        policy = VARIANTS[name]["policy"]
        requested = _requested(oracle, fields, invert=False)
        corrected, reports = apply_branch_constraints_batch(
            original, valid, requested, fields=fields or None, bilateral_write_policy=policy)
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
        moved_mask = np.linalg.norm(corrected - original, axis=-1) > 0
        report["variants"][name] = {
            "fields": fields,
            "bilateral_write_policy": policy,
            "moved_joint_names": sorted(
                {JOINT_NAMES[joint] for joint in np.flatnonzero(moved_mask.any(axis=0))},
                key=lambda item: JOINT_NAMES.index(item)),
            "moved_joint_count_per_constrained_frame": (
                float(moved_mask.sum(axis=1)[moved_mask.any(axis=1)].mean())
                if moved_mask.any() else 0.0),
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
            "introduced_depth_displacement_mm": {
                "total": float(np.abs(corrected[..., FORWARD_DEPTH_AXIS]
                                      - original[..., FORWARD_DEPTH_AXIS]).sum() * 1000.0),
                "mean_per_frame": float(np.abs(corrected[..., FORWARD_DEPTH_AXIS]
                                               - original[..., FORWARD_DEPTH_AXIS]).sum(axis=1).mean() * 1000.0),
            },
            "attachment_length_change_mm": _attachment_change_mm(original, corrected, valid),
            "per_joint_mean_error_mm": evaluation["per_joint_mean_error_mm"],
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
                original, valid, wrong_requested, fields=fields, bilateral_write_policy=policy)
            wrong_eval = evaluate_predictions(bank, positions, wrong,
                                              candidate=f"{args.source_candidate}+{name}+opposite_sign")
            report["wrong_sign_control"][name] = {
                "endpoints": ["oracle", "opposite_oracle"],
                "corrected_frames": int(sum(
                    any(entry["outcome"] == CORRECTED for entry in item["fields"].values())
                    for item in wrong_reports)),
                "evaluation": _summary(wrong_eval),
                "attachment_length_change_mm": _attachment_change_mm(original, wrong, valid),
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

    print(f"\n== branch-constraint replay over {args.source_candidate} "
          f"({len(positions)} {args.split} frames, validity={args.validity}/{validity_array}, "
          f"regime={bank.regime()})")
    print("   %-16s %8s %10s %9s %9s" % ("variant", "flip", "hingeMAE", "MPJPE", "PA-MPJPE"))
    show("baseline", report["baseline"])
    for name in selected_variants:
        show(name, report["variants"][name]["evaluation"])
    print("\n   wrong-sign endpoint (opposite oracle)")
    for name, value in report["wrong_sign_control"].items():
        show(name, value["evaluation"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
