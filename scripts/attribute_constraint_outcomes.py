#!/usr/bin/env python3
"""Why do constrained hinges still flip, and where does bilateral damage land?

docs/34 Sections 4, 5 and 7. Two questions about docs/33's stored results, both
answered from artifacts already on disk. Nothing is trained and no model runs.

RESIDUAL HINGE FLIPS
    The hinge SignState stores one bit: the sign of `bend_direction`'s +Y
    component. The historical hinge metric compares the FULL 3D bend direction
    and calls a chain flipped when the two directions are more than 90 apart.
    Those are not the same predicate. This builds the exact cross-table of
    constraint outcome x requested-sign satisfaction x historical flip, so the
    residual flips can be attributed either to geometry the contract could not
    read or to the one-bit encoding being incomplete.

BILATERAL COLLATERAL
    docs/33 showed the anchor-only pair swap degrading all four hinge fields it
    did not ask for. Aggregate sign agreement says that happened; it does not
    say which transitions produced it. This counts correct->incorrect and
    incorrect->correct per hinge field, separating the shoulder anchor's effect
    on the elbow chains from the hip anchor's effect on the knee chains, and
    reports the same accounting under the dependency-aware write policy.

Every constrained prediction recomputed here is checked byte-for-byte against
the stored artifact it claims to be, so the analysis is bound to the real run.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from common.canonical_pose import hinge_errors
from common.serialization import write_json
from framepose.bank import load_bank
from framepose.branch_constraints import (
    ALREADY_SATISFIED, CORRECTED, UNRESOLVED, apply_branch_constraints_batch,
)
from framepose.constraint_graph import ANCHOR_ONLY, DEPENDENCY_AWARE
from framepose.contract import JOINT_NAMES
from framepose.signs import SIGN_FIELD_NAMES, UNKNOWN, mask_fields, oracle_sign_states, sign_state

SCHEMA = "animcv_frame_pose_constraint_attribution_v1"

_HINGE_FIELDS = tuple(name for name in SIGN_FIELD_NAMES if name.endswith("_forward_bend"))
_HINGE_JOINTS = tuple(name[: -len("_forward_bend")] for name in _HINGE_FIELDS)

#: Bilateral write plans the collateral accounting compares. Each is recomputed
#: from the same stored baseline; only the fields and the write policy change.
_BILATERAL_PLANS = {
    "C_SHOULDER": (["shoulder_forward_depth"], ANCHOR_ONLY),
    "C_HIP": (["hip_forward_depth"], ANCHOR_ONLY),
    "C_BILATERAL": (["shoulder_forward_depth", "hip_forward_depth"], ANCHOR_ONLY),
    "C_SHOULDER_DEP": (["shoulder_forward_depth"], DEPENDENCY_AWARE),
    "C_HIP_DEP": (["hip_forward_depth"], DEPENDENCY_AWARE),
    "C_BILATERAL_DEP": (["shoulder_forward_depth", "hip_forward_depth"], DEPENDENCY_AWARE),
}

#: Which anchor each hinge chain hangs from, so collateral can be attributed to
#: the anchor that caused it rather than pooled.
_ANCHOR_OF_CHAIN = {"left_elbow": "shoulder_forward_depth", "right_elbow": "shoulder_forward_depth",
                    "left_knee": "hip_forward_depth", "right_knee": "hip_forward_depth"}


def _digest(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    return {"path": str(path), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def _hinge_table(poses: np.ndarray, targets: np.ndarray, valid: np.ndarray):
    """Per-frame full-3D hinge error and flip for each chain, as arrays."""
    flipped = np.zeros((len(poses), len(_HINGE_JOINTS)), dtype=np.int8)
    error = np.full((len(poses), len(_HINGE_JOINTS)), np.nan)
    for order in range(len(poses)):
        for record in hinge_errors(poses[order], targets[order], valid[order]):
            column = _HINGE_JOINTS.index(record["joint"])
            flipped[order, column] = 1 if record["flipped"] else 0
            error[order, column] = record["error_degrees"]
    # -1 marks "the metric could not be computed for this chain on this frame".
    flipped[np.isnan(error)] = -1
    return flipped, error


def _bucket_stats(values: np.ndarray) -> dict[str, Any]:
    finite = values[np.isfinite(values)]
    if not finite.size:
        return {"count": 0}
    return {"count": int(finite.size), "mean_degrees": float(finite.mean()),
            "median_degrees": float(np.median(finite)),
            "p90_degrees": float(np.percentile(finite, 90)),
            "max_degrees": float(finite.max())}


def _residual_hinge_attribution(bank, positions, original, corrected, reports, requested,
                                targets, valid, sample_limit: int) -> dict[str, Any]:
    after_signs = np.stack([sign_state(corrected[order], valid[order]) for order in range(len(corrected))])
    flipped, error = _hinge_table(corrected, targets, valid)
    base_flipped, base_error = _hinge_table(original, targets, valid)

    per_field: dict[str, Any] = {}
    satisfied_but_flipped_samples: list[dict[str, Any]] = []
    totals = Counter()
    for column, field in enumerate(_HINGE_FIELDS):
        index = SIGN_FIELD_NAMES.index(field)
        cross: dict[str, int] = Counter()
        buckets: dict[str, list[float]] = {}
        for order in range(len(corrected)):
            want = int(requested[order, index])
            state_before = int(base_flipped[order, column])
            if want == UNKNOWN:
                # No branch was requested for this chain at all (the oracle
                # itself is degenerate here). A residual flip in this bucket is
                # not something any constraint declined to fix -- it was never
                # asked about -- and pooling it with the rest would overstate
                # what enforcement left behind.
                state = int(flipped[order, column])
                key = ("not_requested|not_requested|"
                       + {1: "flipped", 0: "not_flipped", -1: "metric_unavailable"}[state])
                cross[key] += 1
                totals[key] += 1
                buckets.setdefault(key, []).append(float(error[order, column]))
                continue
            outcome = reports[order]["fields"][field]["outcome"]
            satisfied = "satisfied" if int(after_signs[order, index]) == want else "not_satisfied"
            state = int(flipped[order, column])
            historical = {1: "flipped", 0: "not_flipped", -1: "metric_unavailable"}[state]
            key = f"{outcome}|{satisfied}|{historical}"
            cross[key] += 1
            totals[key] += 1
            buckets.setdefault(key, []).append(float(error[order, column]))
            if satisfied == "satisfied" and state == 1 and len(satisfied_but_flipped_samples) < sample_limit:
                satisfied_but_flipped_samples.append({
                    "field": field,
                    "sample_id": bank.samples[int(positions[order])].sample_id,
                    "frame_index": int(bank.samples[int(positions[order])].frame_index),
                    "constraint_outcome": outcome,
                    "requested_sign": want,
                    "read_back_sign": int(after_signs[order, index]),
                    "hinge_error_degrees_after": float(error[order, column]),
                    "hinge_error_degrees_before": float(base_error[order, column]),
                    "was_flipped_before": bool(base_flipped[order, column] == 1),
                })
        per_field[field] = {
            "cross_table": dict(sorted(cross.items())),
            "error_by_bucket": {key: _bucket_stats(np.asarray(values))
                                for key, values in sorted(buckets.items())},
            "satisfied_and_flipped": sum(count for key, count in cross.items()
                                         if key.endswith("|satisfied|flipped")
                                         and not key.startswith("not_requested")),
            "flipped_total": sum(count for key, count in cross.items() if key.endswith("|flipped")),
            "flipped_in_requested_frames": sum(
                count for key, count in cross.items()
                if key.endswith("|flipped") and not key.startswith("not_requested")),
            "flipped_in_unrequested_frames": cross.get("not_requested|not_requested|flipped", 0),
        }
    return {
        "axes": ["constraint_outcome", "requested_sign_satisfaction", "historical_hinge_flip"],
        "per_field": per_field,
        "pooled_cross_table": dict(sorted(totals.items())),
        "satisfied_but_flipped_samples": satisfied_but_flipped_samples,
    }


def _sign_transitions(before: np.ndarray, after: np.ndarray, reference: np.ndarray,
                      field: str) -> dict[str, Any]:
    """How a field's agreement with the oracle moved, transition by transition."""
    index = SIGN_FIELD_NAMES.index(field)
    want, was, now = reference[:, index], before[:, index], after[:, index]
    readable = want != UNKNOWN
    was_right, now_right = (was == want) & readable, (now == want) & readable
    changed = (was != now) & readable
    return {
        "readable_frames": int(readable.sum()),
        "agreement_before": float(was_right[readable].mean()) if readable.any() else None,
        "agreement_after": float(now_right[readable].mean()) if readable.any() else None,
        "changed": int(changed.sum()),
        "correct_to_incorrect": int((was_right & ~now_right).sum()),
        "incorrect_to_correct": int((~was_right & now_right).sum()),
        "correct_to_unreadable": int((was_right & (now == UNKNOWN)).sum()),
        "unreadable_to_correct": int(((was == UNKNOWN) & now_right & readable).sum()),
    }


def _bilateral_collateral(original, variants, targets, valid, reference) -> dict[str, Any]:
    base_signs = np.stack([sign_state(original[order], valid[order]) for order in range(len(original))])
    base_flipped, base_error = _hinge_table(original, targets, valid)

    result: dict[str, Any] = {}
    for name, corrected in variants.items():
        signs = np.stack([sign_state(corrected[order], valid[order]) for order in range(len(corrected))])
        flipped, error = _hinge_table(corrected, targets, valid)
        per_chain = {}
        for column, joint in enumerate(_HINGE_JOINTS):
            usable = (base_flipped[:, column] >= 0) & (flipped[:, column] >= 0)
            delta = error[usable, column] - base_error[usable, column]
            per_chain[f"{joint}_forward_bend"] = {
                "caused_by_anchor": _ANCHOR_OF_CHAIN[joint],
                **_sign_transitions(base_signs, signs, reference, f"{joint}_forward_bend"),
                "flip_rate_before": float((base_flipped[usable, column] == 1).mean()),
                "flip_rate_after": float((flipped[usable, column] == 1).mean()),
                "newly_flipped": int(((base_flipped[:, column] == 0) & (flipped[:, column] == 1)).sum()),
                "newly_unflipped": int(((base_flipped[:, column] == 1) & (flipped[:, column] == 0)).sum()),
                "hinge_error_mean_change_degrees": float(delta.mean()) if delta.size else None,
                "hinge_error_worsened_frames": int((delta > 1e-9).sum()),
                "hinge_error_improved_frames": int((delta < -1e-9).sum()),
            }
        result[name] = {
            "requested_fields": {field: _sign_transitions(base_signs, signs, reference, field)
                                 for field in ("shoulder_forward_depth", "hip_forward_depth")},
            "collateral_hinge_chains": per_chain,
            "collateral_by_anchor": {
                anchor: {
                    "newly_flipped": sum(entry["newly_flipped"] for entry in per_chain.values()
                                         if entry["caused_by_anchor"] == anchor),
                    "correct_to_incorrect": sum(entry["correct_to_incorrect"] for entry in per_chain.values()
                                                if entry["caused_by_anchor"] == anchor),
                    "incorrect_to_correct": sum(entry["incorrect_to_correct"] for entry in per_chain.values()
                                                if entry["caused_by_anchor"] == anchor),
                } for anchor in ("shoulder_forward_depth", "hip_forward_depth")},
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Residual hinge and bilateral collateral attribution")
    parser.add_argument("--bank", required=True, type=Path)
    parser.add_argument("--baseline", required=True, type=Path, help="stored S0 prediction_<split>.npy")
    parser.add_argument("--hinge-prediction", required=True, type=Path,
                        help="stored C_HINGE_ALL prediction, verified byte-identical to a recompute")
    parser.add_argument("--verify", action="append", default=[],
                        help="NAME=PATH stored bilateral prediction to verify a recompute against")
    parser.add_argument("--split", default="test")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--sample-limit", type=int, default=60)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    bank = load_bank(args.bank)
    positions = bank.indices(args.split)
    valid = bank.arrays["target_valid"][positions]
    targets = bank.arrays["target_3d"][positions].astype(np.float64)
    oracle = oracle_sign_states(bank.arrays["target_3d"], bank.arrays["target_valid"])[positions]
    original = np.load(args.baseline).astype(np.float64)
    if original.shape != (len(positions), len(JOINT_NAMES), 3):
        raise ValueError(f"baseline {original.shape} does not match the {args.split} split")

    verification: list[dict[str, Any]] = []

    # --- residual hinge flips -------------------------------------------------
    hinge_fields = list(_HINGE_FIELDS)
    requested = mask_fields(oracle, hinge_fields).astype(np.int64)
    corrected, reports = apply_branch_constraints_batch(original, valid, requested, fields=hinge_fields)
    stored = np.load(args.hinge_prediction)
    if not np.array_equal(corrected.astype(np.float32), stored):
        raise ValueError(
            f"recomputed C_HINGE_ALL does not reproduce {args.hinge_prediction}; refusing to attribute "
            "outcomes to a run these predictions did not come from")
    verification.append({"variant": "C_HINGE_ALL", "reproduced": True, **_digest(args.hinge_prediction)})

    residual = _residual_hinge_attribution(bank, positions, original, corrected, reports, requested,
                                           targets, valid, args.sample_limit)

    # --- bilateral collateral -------------------------------------------------
    expected = dict(entry.split("=", 1) for entry in args.verify)
    variants: dict[str, np.ndarray] = {}
    for name, (fields, policy) in _BILATERAL_PLANS.items():
        request = mask_fields(oracle, fields).astype(np.int64)
        result, _ = apply_branch_constraints_batch(original, valid, request, fields=fields,
                                                   bilateral_write_policy=policy)
        variants[name] = result
        if name in expected:
            path = Path(expected[name])
            if not np.array_equal(result.astype(np.float32), np.load(path)):
                raise ValueError(f"recomputed {name} does not reproduce {path}")
            verification.append({"variant": name, "reproduced": True, **_digest(path)})

    collateral = _bilateral_collateral(original, variants, targets, valid, oracle)

    report = {
        "schema": SCHEMA,
        "provenance": {
            "bank_index": str(args.bank),
            "bank_content_digest": bank.content_digest(),
            "observation_regime": bank.regime(),
            "split": args.split,
            "frames": int(len(positions)),
            "constraint_validity_source": "target_valid",
            "baseline_prediction": _digest(args.baseline),
            "stored_artifact_verification": verification,
            "method": "no training, no model execution; stored predictions and deterministic recomputes only",
        },
        "residual_hinge_attribution": residual,
        "bilateral_collateral_attribution": collateral,
    }
    write_json(args.out / "constraint_attribution.json", report)

    print("\n== residual hinge cross-table (pooled over the four chains)")
    for key, count in report["residual_hinge_attribution"]["pooled_cross_table"].items():
        print("   %-56s %6d" % (key, count))
    print("\n== satisfied-but-still-flipped, per field")
    for field, entry in residual["per_field"].items():
        print("   %-26s satisfied&flipped=%5d   flipped_total=%5d" % (
            field, entry["satisfied_and_flipped"], entry["flipped_total"]))

    print("\n== bilateral collateral: newly flipped hinge chains")
    print("   %-18s %-26s %8s %8s %8s" % ("variant", "chain", "new_flip", "c->i", "i->c"))
    for name, entry in collateral.items():
        for chain, value in entry["collateral_hinge_chains"].items():
            if value["newly_flipped"] or value["correct_to_incorrect"] or value["incorrect_to_correct"]:
                print("   %-18s %-26s %8d %8d %8d" % (
                    name, chain, value["newly_flipped"], value["correct_to_incorrect"],
                    value["incorrect_to_correct"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
