#!/usr/bin/env python3
"""Compose learned bilateral conditioning with explicit hinge constraints.

docs/35. Every SignState result so far is channel-isolated: `O_BILATERAL`
measured learned bilateral conditioning, `C_HINGE_ALL` measured explicit hinge
constraints over `S0`. "The best architecture is hybrid" is therefore a
per-channel hypothesis, not a validated composed system.

This replays the composition itself:

    geometry observation
        -> trained O_BILATERAL model            (learned bilateral conditioning)
        -> continuous pose                       [STORED, never recomputed]
        -> explicit four-field hinge constraint  (parameter-free, output space)
        -> hybrid pose

The intended ownership is that the learned channel owns global orientation and
depth magnitude while the constraint owns only the four local bend branches.
That is checked against the real stored predictions, not assumed from the code:
the replay REFUSES to report if any joint other than the four hinge middle
joints moves, or if a bilateral quantity changes.

No training. No model execution. No sensor. No RGB. No VLM. The hinge operator
and the Sign Contract are untouched.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

from common.canonical_pose import FORWARD_DEPTH_AXIS, JOINT_INDEX, hinge_errors
from common.serialization import write_json
from framepose.bank import load_bank
from framepose.branch_constraints import (
    BRANCH_CONSTRAINT_SCHEMA, CORRECTED, DEPTH_ONLY, HINGE_WRITE_POLICIES, UNRESOLVED,
    apply_branch_constraints_batch, coverage,
)
from framepose.contract import JOINT_NAMES
from framepose.evaluate import evaluate_predictions
from framepose.replay_provenance import verify_source_identity
from framepose.signs import (
    HINGE_CHAINS_BY_JOINT, SIGN_FIELD_NAMES, UNKNOWN, mask_fields, oracle_sign_states, sign_state,
)

SCHEMA = "animcv_frame_pose_hybrid_signstate_replay_v1"

HINGE_FIELDS = tuple(name for name in SIGN_FIELD_NAMES if name.endswith("_forward_bend"))
HINGE_JOINTS = tuple(name[: -len("_forward_bend")] for name in HINGE_FIELDS)
#: The only joints the hinge constraint is permitted to write.
HINGE_MIDDLE_INDICES = tuple(sorted(JOINT_INDEX[joint] for joint in HINGE_JOINTS))

#: Owned by the learned bilateral channel. Every one of these must survive the
#: hinge constraint untouched, or the ownership abstraction is violated.
BILATERAL_OWNED_METRICS = (
    "root_yaw_error_degrees",
    "shoulder_forward_depth_residual_mm", "shoulder_forward_depth_abs_residual_mm",
    "shoulder_forward_depth_sign_disagreement", "shoulder_forward_depth_sign_disagreement_stable",
    "hip_forward_depth_residual_mm", "hip_forward_depth_abs_residual_mm",
    "hip_forward_depth_sign_disagreement", "hip_forward_depth_sign_disagreement_stable",
)
BILATERAL_OWNED_SIGN_FIELDS = ("torso_facing", "shoulder_forward_depth", "hip_forward_depth")

HINGE_METRICS = ("hinge_flip_rate", "hinge_direction_mae_degrees",
                 "left_elbow_bend_error_degrees", "right_elbow_bend_error_degrees",
                 "left_knee_bend_error_degrees", "right_knee_bend_error_degrees")
GUARDRAIL_METRICS = ("mpjpe_mm", "pa_mpjpe_mm", "max_joint_error_mm")


def _stat(evaluation: dict[str, Any], name: str) -> dict[str, Any] | None:
    entry = evaluation["aggregate"].get(name)
    return None if entry is None else {key: entry[key] for key in
                                       ("mean", "median", "p90", "p95", "count")}


def _summary(evaluation: dict[str, Any]) -> dict[str, Any]:
    return {
        "orientation": {name: _stat(evaluation, name) for name in BILATERAL_OWNED_METRICS},
        "hinge": {name: _stat(evaluation, name) for name in HINGE_METRICS},
        "guardrail": {name: _stat(evaluation, name) for name in GUARDRAIL_METRICS},
        "sign_agreement": evaluation.get("sign_agreement"),
        "per_joint_mean_error_mm": evaluation["per_joint_mean_error_mm"],
    }


def _assert_ownership(before: np.ndarray, after: np.ndarray,
                      depth_only: bool = True) -> dict[str, Any]:
    """Only the four hinge middle joints may differ.

    Under the historical depth-only policy the write is additionally confined
    to the depth axis. The minimum-norm policy writes in the plane
    perpendicular to the limb axis, so it moves X and Z too -- the joint-level
    ownership boundary is the one both must satisfy, and it is the one the
    bilateral channel depends on.
    """
    moved = np.flatnonzero((before != after).any(axis=(0, 2)))
    illegal = sorted(set(moved.tolist()) - set(HINGE_MIDDLE_INDICES))
    if illegal:
        raise ValueError(
            f"hinge constraint wrote joints it does not own: {[JOINT_NAMES[i] for i in illegal]}; "
            "the ownership abstraction is violated and this replay is refused")
    axis_moved = [axis for axis in range(3)
                  if axis != FORWARD_DEPTH_AXIS and not np.array_equal(before[..., axis], after[..., axis])]
    if axis_moved and depth_only:
        raise ValueError(f"hinge constraint moved non-depth axes {axis_moved}; refused")
    return {
        "moved_joints": [JOINT_NAMES[index] for index in moved.tolist()],
        "permitted_joints": [JOINT_NAMES[index] for index in HINGE_MIDDLE_INDICES],
        "wrote_only_permitted_joints": True,
        "wrote_only_the_depth_axis": not axis_moved,
        "frames_with_any_write": int((before != after).any(axis=(1, 2)).sum()),
    }


def _assert_bilateral_preserved(base: dict[str, Any], hybrid: dict[str, Any]) -> dict[str, Any]:
    """Hard composition contract: the learned channel's metrics must not move."""
    report: dict[str, Any] = {}
    broken: list[str] = []
    for name in BILATERAL_OWNED_METRICS:
        before, after = base["orientation"][name], hybrid["orientation"][name]
        identical = before == after
        report[name] = {"identical": bool(identical), "before": before, "after": after}
        if not identical:
            broken.append(name)
    for field in BILATERAL_OWNED_SIGN_FIELDS:
        before = base["sign_agreement"][field]["agreement"]
        after = hybrid["sign_agreement"][field]["agreement"]
        report[f"sign_agreement:{field}"] = {"identical": before == after,
                                             "before": before, "after": after}
        if before != after:
            broken.append(f"sign_agreement:{field}")
    if broken:
        raise ValueError(
            f"bilateral quantities changed under hinge-only enforcement: {broken}; the learned "
            "channel does not own what it is supposed to own and this replay is refused")
    report["all_preserved_exactly"] = True
    return report


def _hinge_arrays(poses, targets, valid):
    flipped = np.full((len(poses), len(HINGE_JOINTS)), -1, dtype=np.int8)
    error = np.full((len(poses), len(HINGE_JOINTS)), np.nan)
    for order in range(len(poses)):
        for record in hinge_errors(poses[order], targets[order], valid[order]):
            column = HINGE_JOINTS.index(record["joint"])
            flipped[order, column] = 1 if record["flipped"] else 0
            error[order, column] = record["error_degrees"]
    return flipped, error


def _cross_table(reports, requested, after_signs, flipped, error):
    """docs/34's decomposition, rebuilt on the hybrid."""
    from collections import Counter

    pooled, per_field = Counter(), {}
    for column, field in enumerate(HINGE_FIELDS):
        index = SIGN_FIELD_NAMES.index(field)
        counts = Counter()
        for order in range(len(reports)):
            want = int(requested[order, index])
            state = int(flipped[order, column])
            historical = {1: "flipped", 0: "not_flipped", -1: "metric_unavailable"}[state]
            if want == UNKNOWN:
                key = f"not_requested|not_requested|{historical}"
            else:
                outcome = reports[order]["fields"][field]["outcome"]
                satisfied = "satisfied" if int(after_signs[order, index]) == want else "not_satisfied"
                key = f"{outcome}|{satisfied}|{historical}"
            counts[key] += 1
            pooled[key] += 1
        per_field[field] = {
            "cross_table": dict(sorted(counts.items())),
            "satisfied_and_flipped": sum(v for k, v in counts.items()
                                         if k.endswith("|satisfied|flipped")
                                         and not k.startswith("not_requested")),
            "flipped_in_requested_frames": sum(v for k, v in counts.items()
                                               if k.endswith("|flipped")
                                               and not k.startswith("not_requested")),
            "flipped_in_unrequested_frames": counts.get("not_requested|not_requested|flipped", 0),
        }
    return {"pooled_cross_table": dict(sorted(pooled.items())), "per_field": per_field}


def _enforcement_effect(reports, requested, base_flipped, base_error, flipped, error):
    """What enforcement did to the HISTORICAL metric, chain-frame by chain-frame.

    The Sign Contract's one-bit Y branch and the full-3D hinge metric are not
    the same predicate, so enforcing the contract can move the metric either
    way. docs/34 measured one direction (the branch is satisfied and the metric
    still says flipped). This measures the other: the contract called the
    branch wrong, the constraint dutifully flipped it, and the metric -- which
    was already content -- got worse.
    """
    from collections import Counter

    result: dict[str, Any] = {}
    for column, field in enumerate(HINGE_FIELDS):
        index = SIGN_FIELD_NAMES.index(field)
        counts, deltas = Counter(), {}
        for order in range(len(reports)):
            if int(requested[order, index]) == UNKNOWN:
                continue
            if reports[order]["fields"][field]["outcome"] != CORRECTED:
                continue
            was, now = int(base_flipped[order, column]), int(flipped[order, column])
            if was < 0 or now < 0:
                counts["metric_unavailable"] += 1
                continue
            key = {(1, 0): "fixed_by_enforcement", (0, 1): "broken_by_enforcement",
                   (1, 1): "stayed_flipped", (0, 0): "stayed_not_flipped"}[(was, now)]
            counts[key] += 1
            deltas.setdefault(key, []).append(float(error[order, column] - base_error[order, column]))
        result[field] = {
            "counts": dict(sorted(counts.items())),
            "mean_error_change_degrees": {key: float(np.mean(values))
                                          for key, values in sorted(deltas.items())},
        }
    pooled = Counter()
    for entry in result.values():
        pooled.update(entry["counts"])
    return {"per_field": result, "pooled": dict(sorted(pooled.items()))}


def _review(bank, positions, before, after, reports, requested, targets, valid,
            base_flipped, base_error, flipped, error, limit, wrong=None,
            wrong_flipped=None, wrong_error=None):
    """Frames a human can check, bucketed by the question each one answers."""
    buckets: dict[str, list[dict[str, Any]]] = {
        "hinge_corrected_by_hybrid": [],
        "branch_corrected_but_position_worsened": [],
        "y_sign_satisfied_but_still_full_3d_flipped": [],
        "contract_called_it_wrong_but_the_3d_metric_did_not": [],
        "requested_but_unresolved": [],
        "opposite_sign_catastrophic_bend": [],
    }
    candidates: dict[str, list[tuple[int, int]]] = {name: [] for name in buckets}
    after_signs = np.stack([sign_state(after[order], valid[order]) for order in range(len(after))])
    per_frame_error = lambda pose, order: float(np.linalg.norm(
        (pose[order][valid[order]] - targets[order][valid[order]]), axis=-1).mean() * 1000.0)

    for order in range(len(positions)):
        for column, field in enumerate(HINGE_FIELDS):
            index = SIGN_FIELD_NAMES.index(field)
            want = int(requested[order, index])
            if want == UNKNOWN:
                continue
            entry = reports[order]["fields"][field]
            satisfied = int(after_signs[order, index]) == want
            row = None
            if entry["outcome"] == CORRECTED and base_flipped[order, column] == 1 and flipped[order, column] == 0:
                row = "hinge_corrected_by_hybrid"
            elif entry["outcome"] == CORRECTED and base_flipped[order, column] == 0 and flipped[order, column] == 1:
                row = "contract_called_it_wrong_but_the_3d_metric_did_not"
            elif entry["outcome"] == CORRECTED and per_frame_error(after, order) > per_frame_error(before, order):
                row = "branch_corrected_but_position_worsened"
            elif satisfied and flipped[order, column] == 1:
                row = "y_sign_satisfied_but_still_full_3d_flipped"
            elif entry["outcome"] == UNRESOLVED:
                row = "requested_but_unresolved"
            if row is not None:
                candidates[row].append((order, column))
            if wrong is not None and wrong_flipped[order, column] == 1 and base_flipped[order, column] == 0:
                candidates["opposite_sign_catastrophic_bend"].append((order, column))

    # Subsample evenly across every qualifying frame rather than taking the
    # first N, which would draw a whole bucket from one sequence.
    for name, picks in candidates.items():
        chosen = picks
        if len(picks) > limit:
            step = len(picks) / limit
            chosen = [picks[int(np.floor(index * step))] for index in range(limit)]
        for order, column in chosen:
            field = HINGE_FIELDS[column]
            index = SIGN_FIELD_NAMES.index(field)
            entry = reports[order]["fields"][field]
            want = int(requested[order, index])
            sample = bank.samples[int(positions[order])]
            joint = HINGE_JOINTS[column]
            middle = JOINT_INDEX[HINGE_CHAINS_BY_JOINT[joint][1]]
            target_pose = wrong if name == "opposite_sign_catastrophic_bend" else after
            target_error = wrong_error if name == "opposite_sign_catastrophic_bend" else error
            buckets[name].append({
                "field": field,
                "sample_id": sample.sample_id,
                "sequence_id": sample.sequence_id,
                "frame_index": int(sample.frame_index),
                "image_reference": (sample.image_reference.relative_path
                                    if sample.image_reference is not None else None),
                "requested_sign": want,
                "read_back_before": entry.get("read_back_before"),
                "read_back_after": entry.get("read_back_after", entry.get("read_back_after_attempt")),
                "constraint_outcome": entry["outcome"],
                "moved_joint": JOINT_NAMES[middle],
                "depth_delta_mm": float((target_pose[order, middle, FORWARD_DEPTH_AXIS]
                                         - before[order, middle, FORWARD_DEPTH_AXIS]) * 1000.0),
                "source_xyz_m": [round(float(v), 5) for v in before[order, middle]],
                "hybrid_xyz_m": [round(float(v), 5) for v in target_pose[order, middle]],
                "full_3d_hinge_error_before": float(base_error[order, column]),
                "full_3d_hinge_error_after": float(target_error[order, column]),
                "frame_mpjpe_before_mm": per_frame_error(before, order),
                "frame_mpjpe_after_mm": per_frame_error(target_pose, order),
                "note": ("opposite-oracle endpoint, not the hybrid"
                         if name == "opposite_sign_catastrophic_bend" else None),
            })
    return buckets


def main() -> int:
    parser = argparse.ArgumentParser(description="Hybrid SignState composition replay")
    parser.add_argument("--bank", required=True, type=Path)
    parser.add_argument("--source", action="append", required=True,
                        help="LABEL=CANDIDATE:PREDICTION[:EVALUATION]")
    parser.add_argument("--split", default="test")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--review-frames", type=int, default=25)
    parser.add_argument("--hinge-write-policy", default=DEPTH_ONLY, choices=list(HINGE_WRITE_POLICIES),
                        help="depth_only is docs/33's frozen operator and the default")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    bank = load_bank(args.bank)
    positions = bank.indices(args.split)
    valid = bank.arrays["target_valid"][positions]
    targets = bank.arrays["target_3d"][positions].astype(np.float64)
    oracle = oracle_sign_states(bank.arrays["target_3d"], bank.arrays["target_valid"])[positions]
    requested = mask_fields(oracle, list(HINGE_FIELDS)).astype(np.int64)
    opposite = requested.copy()
    known = opposite != UNKNOWN
    opposite[known] = -opposite[known]

    report: dict[str, Any] = {
        "schema": SCHEMA,
        "constraint_schema": BRANCH_CONSTRAINT_SCHEMA,
        "method": ("stored predictions only; the four-field hinge Branch Constraint is applied in "
                   "output space to each learned candidate's own prediction; no training, no model "
                   "execution, no bilateral output-space constraint, no sensor"),
        "sign_source": "oracle hinge signs (upper bound; not a sensor result)",
        "hinge_fields": list(HINGE_FIELDS),
        "hinge_write_policy": args.hinge_write_policy,
        "bilateral_channel": "learned conditioning, untouched",
        "provenance": {"bank_index": str(args.bank),
                       "bank_content_digest": bank.content_digest(),
                       "observation_regime": bank.regime(),
                       "split": args.split, "frames": int(len(positions)),
                       "constraint_validity_source": "target_valid",
                       "sources": {}},
        "sources": {},
    }
    review: dict[str, Any] = {"schema": SCHEMA, "split": args.split, "sources": {}}

    for entry in args.source:
        label, rest = entry.split("=", 1)
        parts = rest.split(":")
        candidate, prediction = parts[0], Path(parts[1])
        evaluation = Path(parts[2]) if len(parts) > 2 and parts[2] else None

        identity = verify_source_identity(
            prediction=prediction, evaluation=evaluation, bank=bank, split=args.split,
            candidate=candidate, frames=len(positions), joints=len(JOINT_NAMES))
        report["provenance"]["sources"][label] = identity

        before = np.load(prediction).astype(np.float64)
        base_eval = evaluate_predictions(bank, positions, before, candidate=f"H0_{label}")
        base_summary = _summary(base_eval)

        after, reports = apply_branch_constraints_batch(
            before, valid, requested, fields=list(HINGE_FIELDS),
            hinge_write_policy=args.hinge_write_policy)
        ownership = _assert_ownership(before, after, depth_only=args.hinge_write_policy == DEPTH_ONLY)
        hybrid_eval = evaluate_predictions(bank, positions, after, candidate=f"H1_{label}_HINGE")
        hybrid_summary = _summary(hybrid_eval)
        preservation = _assert_bilateral_preserved(base_summary, hybrid_summary)

        base_flipped, base_error = _hinge_arrays(before, targets, valid)
        flipped, error = _hinge_arrays(after, targets, valid)
        after_signs = np.stack([sign_state(after[order], valid[order]) for order in range(len(after))])

        wrong, wrong_reports = apply_branch_constraints_batch(
            before, valid, opposite, fields=list(HINGE_FIELDS),
            hinge_write_policy=args.hinge_write_policy)
        _assert_ownership(before, wrong, depth_only=args.hinge_write_policy == DEPTH_ONLY)
        wrong_eval = evaluate_predictions(bank, positions, wrong, candidate=f"H1_{label}_HINGE_opposite")
        wrong_summary = _summary(wrong_eval)
        # The structural claim the wrong-sign endpoint must demonstrate: bad
        # hinge advice stays local and cannot reach the learned channel.
        wrong_preservation = _assert_bilateral_preserved(base_summary, wrong_summary)

        displacement = after - before
        moved = np.linalg.norm(displacement, axis=-1) > 0
        norms = np.linalg.norm(displacement, axis=-1)[moved] * 1000.0
        corrections = {
            "count": int(moved.sum()),
            "norm_mm": {**{f"p{q}": float(np.percentile(norms, q)) for q in (50, 90, 99, 99.9)},
                        "max": float(norms.max()) if norms.size else 0.0,
                        "mean": float(norms.mean()) if norms.size else 0.0},
            **{f"abs_delta_{name}_mm": {
                **{f"p{q}": float(np.percentile(np.abs(displacement[..., axis])[moved] * 1000.0, q))
                   for q in (50, 90, 99, 99.9)},
                "max": float((np.abs(displacement[..., axis])[moved] * 1000.0).max()) if norms.size else 0.0}
               for axis, name in ((0, "x"), (FORWARD_DEPTH_AXIS, "y"), (2, "z"))},
        }
        np.save(args.out / f"prediction_{args.split}_H1_{label}_HINGE.npy", after.astype(np.float32))
        report["sources"][label] = {
            "candidate": candidate,
            "H0": base_summary,
            "H1": hybrid_summary,
            "ownership": ownership,
            "correction_distribution": corrections,
            "bilateral_preservation": preservation,
            "coverage": coverage(reports)["fields"],
            "requested_branch_satisfied_rate": {
                field: (float(np.mean(asked)) if (asked := [
                    r["requested_branch_satisfied"][field] is True for r in reports
                    if r["requested_branch_satisfied"][field] is not None]) else None)
                for field in HINGE_FIELDS},
            "residual_hinge_attribution": _cross_table(reports, requested, after_signs, flipped, error),
            "enforcement_effect_on_historical_metric": _enforcement_effect(
                reports, requested, base_flipped, base_error, flipped, error),
            "wrong_sign_endpoint": {
                "endpoints": ["oracle", "opposite_oracle"],
                "evaluation": wrong_summary,
                "bilateral_preservation": wrong_preservation,
                "corrected_frames": int(sum(any(e["outcome"] == CORRECTED for e in r["fields"].values())
                                            for r in wrong_reports)),
            },
        }
        wrong_flipped, wrong_error = _hinge_arrays(wrong, targets, valid)
        review["sources"][label] = _review(bank, positions, before, after, reports, requested,
                                           targets, valid, base_flipped, base_error, flipped, error,
                                           args.review_frames, wrong=wrong,
                                           wrong_flipped=wrong_flipped, wrong_error=wrong_error)

    write_json(args.out / "hybrid_signstate_replay.json", report)
    write_json(args.out / "hybrid_signstate_review.json", review)

    print(f"\n== hybrid composition replay ({len(positions)} {args.split} frames, "
          f"regime={bank.regime()})")
    print("   %-14s %-6s %8s %8s %9s %9s %9s %9s" % (
        "source", "stage", "yaw", "yawP95", "flip", "hingeMAE", "MPJPE", "PA"))
    for label, entry in report["sources"].items():
        for stage in ("H0", "H1"):
            s = entry[stage]
            print("   %-14s %-6s %8.3f %8.3f %9.4f %9.3f %9.3f %9.3f" % (
                label, stage, s["orientation"]["root_yaw_error_degrees"]["mean"],
                s["orientation"]["root_yaw_error_degrees"]["p95"],
                s["hinge"]["hinge_flip_rate"]["mean"], s["hinge"]["hinge_direction_mae_degrees"]["mean"],
                s["guardrail"]["mpjpe_mm"]["mean"], s["guardrail"]["pa_mpjpe_mm"]["mean"]))
        print("   %-14s %-6s %8s %8s %9.4f %9.3f %9.3f %9.3f" % (
            label, "WRONG", "-", "-",
            entry["wrong_sign_endpoint"]["evaluation"]["hinge"]["hinge_flip_rate"]["mean"],
            entry["wrong_sign_endpoint"]["evaluation"]["hinge"]["hinge_direction_mae_degrees"]["mean"],
            entry["wrong_sign_endpoint"]["evaluation"]["guardrail"]["mpjpe_mm"]["mean"],
            entry["wrong_sign_endpoint"]["evaluation"]["guardrail"]["pa_mpjpe_mm"]["mean"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
