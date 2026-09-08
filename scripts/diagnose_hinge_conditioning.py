#!/usr/bin/env python3
"""Why can a CORRECT oracle hinge sign demand a multi-metre depth correction?

docs/36 Sections 3-5. Diagnostic only: the historical `_hinge_correction` is
frozen and is not called differently, not tuned, and not modified.

The current operator moves the middle joint in depth only, by

    delta_y = -2 * o_y / f,     f = (a_x^2 + a_z^2) / |a|^2

and refuses only when `sqrt(f) < UNIT_FORWARD_EPSILON`. docs/35 measured a
+3.108 m correction from a correct oracle sign. This records, for every
corrected hinge chain-frame on three stored FramePose sources, the exact
geometry that produced its correction, so the tail can be attributed rather
than guessed at.

The four hinge fields are mutually independent -- the elbow chains read
shoulder/elbow/wrist and the knee chains read hip/knee/ankle, and each writes
only its own middle joint -- so every record is computed from the source pose
directly.

No training, no model execution, no new threshold.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from common.canonical_pose import FORWARD_DEPTH_AXIS, JOINT_INDEX, hinge_errors
from common.serialization import write_json
from framepose.bank import load_bank
from framepose.branch_constraints import CORRECTED, apply_branch_constraints_batch
from framepose.contract import JOINT_NAMES
from framepose.replay_provenance import verify_source_identity
from framepose.signs import (
    HINGE_CHAINS_BY_JOINT, MIN_BEND_OFFSET_M, SIGN_FIELD_NAMES, UNIT_FORWARD_EPSILON, UNKNOWN,
    mask_fields, oracle_sign_states, sign_state,
)

SCHEMA = "animcv_frame_pose_hinge_conditioning_v1"

HINGE_FIELDS = tuple(name for name in SIGN_FIELD_NAMES if name.endswith("_forward_bend"))
HINGE_JOINTS = tuple(name[: -len("_forward_bend")] for name in HINGE_FIELDS)

#: Reported for continuity with docs/35's review ranges ONLY. These are not
#: decision thresholds and nothing branches on them.
CONTINUITY_RANGES_MM = (200.0, 500.0, 1000.0, 2000.0)


def _chain_geometry(pose: np.ndarray, field: str) -> dict[str, Any]:
    """The exact quantities the closed form is built from."""
    proximal, middle, distal = HINGE_CHAINS_BY_JOINT[field[: -len("_forward_bend")]]
    p, m, d = (pose[JOINT_INDEX[name]] for name in (proximal, middle, distal))
    axis = d - p
    axis_squared = float(axis @ axis)
    relative = m - p
    offset = relative - axis * (float(relative @ axis) / axis_squared)
    in_plane = float(axis[0] ** 2 + axis[2] ** 2) / axis_squared
    axis_length = float(np.sqrt(axis_squared))
    # Angle between the limb axis and the camera depth direction.
    depth_cosine = abs(float(axis[FORWARD_DEPTH_AXIS]) / axis_length)
    return {
        "axis": [float(v) for v in axis],
        "axis_length_m": axis_length,
        "axis_angle_to_depth_degrees": float(np.degrees(np.arccos(np.clip(depth_cosine, 0.0, 1.0)))),
        "f_in_plane_fraction": in_plane,
        "sqrt_f": float(np.sqrt(in_plane)),
        "offset": [float(v) for v in offset],
        "offset_norm_m": float(np.linalg.norm(offset)),
        "offset_forward_m": float(offset[FORWARD_DEPTH_AXIS]),
        "abs_offset_forward_m": abs(float(offset[FORWARD_DEPTH_AXIS])),
        "amplification_1_over_f": 1.0 / in_plane,
        "amplification_2_over_f": 2.0 / in_plane,
        "analytic_delta_y_m": -2.0 * float(offset[FORWARD_DEPTH_AXIS]) / in_plane,
        # The minimum displacement of the middle joint that produces the SAME
        # offset change the depth-only reflection produces. Depth-only exceeds
        # it by exactly 1/sqrt(f), because the depth direction is mostly
        # parallel to the limb axis when f is small.
        "reflection_offset_change_m": 2.0 * abs(float(offset[FORWARD_DEPTH_AXIS])) / np.sqrt(in_plane),
        "depth_only_over_minimum_ratio": 1.0 / float(np.sqrt(in_plane)),
        "middle_joint": middle,
        "chain": [proximal, middle, distal],
    }


def _hinge_state(poses, targets, valid):
    flipped = np.full((len(poses), len(HINGE_JOINTS)), -1, dtype=np.int8)
    error = np.full((len(poses), len(HINGE_JOINTS)), np.nan)
    for order in range(len(poses)):
        for record in hinge_errors(poses[order], targets[order], valid[order]):
            column = HINGE_JOINTS.index(record["joint"])
            flipped[order, column] = 1 if record["flipped"] else 0
            error[order, column] = record["error_degrees"]
    return flipped, error


def _correlations(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Continuous relationships first; no binning drives any conclusion."""
    if not records:
        return {}
    columns = {
        "abs_delta_y_mm": np.array([r["abs_delta_y_mm"] for r in records]),
        "f": np.array([r["f_in_plane_fraction"] for r in records]),
        "sqrt_f": np.array([r["sqrt_f"] for r in records]),
        "abs_offset_forward_mm": np.array([r["abs_offset_forward_m"] * 1000.0 for r in records]),
        "offset_norm_mm": np.array([r["offset_norm_m"] * 1000.0 for r in records]),
        "axis_angle_to_depth_degrees": np.array([r["axis_angle_to_depth_degrees"] for r in records]),
        "hinge_error_before_degrees": np.array([r["hinge_error_before_degrees"] for r in records]),
        "mpjpe_delta_mm": np.array([r["frame_mpjpe_after_mm"] - r["frame_mpjpe_before_mm"]
                                    for r in records]),
    }
    target = np.log10(np.maximum(columns["abs_delta_y_mm"], 1e-6))
    result: dict[str, Any] = {"note": ("Pearson r against log10|delta_y|; the closed form is "
                                       "multiplicative, so a log target is the honest scale")}
    for name, values in columns.items():
        if name == "abs_delta_y_mm":
            continue
        usable = np.isfinite(values) & np.isfinite(target)
        if usable.sum() < 3:
            result[name] = None
            continue
        result[name] = float(np.corrcoef(values[usable], target[usable])[0, 1])
    # The decomposition the closed form predicts exactly.
    log_two_o = np.log10(np.maximum(2.0 * columns["abs_offset_forward_mm"], 1e-6))
    log_inv_f = np.log10(1.0 / np.maximum(columns["f"], 1e-12))
    result["log10_decomposition"] = {
        "explanation": "log10|delta_y| = log10(2|o_y|) + log10(1/f) exactly",
        "max_abs_residual": float(np.abs(target - (log_two_o + log_inv_f)).max()),
        "variance_of_log10_2_abs_o_y": float(np.var(log_two_o)),
        "variance_of_log10_1_over_f": float(np.var(log_inv_f)),
        "variance_of_log10_abs_delta_y": float(np.var(target)),
        "covariance": float(np.cov(log_two_o, log_inv_f)[0, 1]),
    }
    return result


def _quantiles(values: np.ndarray) -> dict[str, Any]:
    if not values.size:
        return {"count": 0}
    return {"count": int(values.size),
            **{f"p{q}": float(np.percentile(values, q)) for q in (50, 90, 99, 99.9)},
            "max": float(values.max()), "mean": float(values.mean())}


def main() -> int:
    parser = argparse.ArgumentParser(description="Hinge-constraint conditioning diagnostic")
    parser.add_argument("--bank", required=True, type=Path)
    parser.add_argument("--source", action="append", required=True,
                        help="LABEL=CANDIDATE:PREDICTION[:EVALUATION]")
    parser.add_argument("--split", default="test")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--export-records", type=int, default=400,
                        help="largest-|delta_y| records written out in full")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    bank = load_bank(args.bank)
    positions = bank.indices(args.split)
    valid = bank.arrays["target_valid"][positions]
    targets = bank.arrays["target_3d"][positions].astype(np.float64)
    oracle = oracle_sign_states(bank.arrays["target_3d"], bank.arrays["target_valid"])[positions]
    requested = mask_fields(oracle, list(HINGE_FIELDS)).astype(np.int64)

    report: dict[str, Any] = {
        "schema": SCHEMA,
        "operator": {
            "status": "frozen historical operator, unmodified",
            "closed_form": "delta_y = -2 * o_y / f",
            "refusal": f"sqrt(f) < UNIT_FORWARD_EPSILON = {UNIT_FORWARD_EPSILON}",
            "min_bend_offset_m": MIN_BEND_OFFSET_M,
        },
        "provenance": {"bank_index": str(args.bank),
                       "bank_content_digest": bank.content_digest(),
                       "observation_regime": bank.regime(),
                       "split": args.split, "frames": int(len(positions)),
                       "constraint_validity_source": "target_valid",
                       "sources": {}},
        "continuity_ranges_mm": list(CONTINUITY_RANGES_MM),
        "sources": {},
    }
    exported: dict[str, Any] = {"schema": SCHEMA, "sources": {}}

    for entry in args.source:
        label, rest = entry.split("=", 1)
        parts = rest.split(":")
        candidate, prediction = parts[0], Path(parts[1])
        evaluation = Path(parts[2]) if len(parts) > 2 and parts[2] else None
        report["provenance"]["sources"][label] = verify_source_identity(
            prediction=prediction, evaluation=evaluation, bank=bank, split=args.split,
            candidate=candidate, frames=len(positions), joints=len(JOINT_NAMES))

        before = np.load(prediction).astype(np.float64)
        after, reports = apply_branch_constraints_batch(before, valid, requested,
                                                        fields=list(HINGE_FIELDS))
        base_flipped, base_error = _hinge_state(before, targets, valid)
        flipped, error = _hinge_state(after, targets, valid)
        after_signs = np.stack([sign_state(after[order], valid[order]) for order in range(len(after))])

        def frame_error(pose, order):
            usable = valid[order]
            return float(np.linalg.norm(pose[order][usable] - targets[order][usable], axis=-1).mean() * 1000.0)

        records: list[dict[str, Any]] = []
        types = Counter()
        for order in range(len(positions)):
            for column, field in enumerate(HINGE_FIELDS):
                index = SIGN_FIELD_NAMES.index(field)
                want = int(requested[order, index])
                if want == UNKNOWN:
                    continue
                accounting = reports[order]["fields"][field]
                if accounting["outcome"] != CORRECTED:
                    continue
                geometry = _chain_geometry(before[order], field)
                was, now = int(base_flipped[order, column]), int(flipped[order, column])
                # docs/36 Section 5: the branch was wrong under the one-bit
                # contract; was the full-3D metric already content?
                kind = ("TYPE_3_metric_unavailable" if was < 0
                        else ("TYPE_1_metric_also_flipped" if was == 1
                              else "TYPE_2_metric_already_acceptable"))
                types[kind] += 1
                sample = bank.samples[int(positions[order])]
                delta = float(accounting["depth_delta_m"])
                records.append({
                    "sample_id": sample.sample_id, "sequence_id": sample.sequence_id,
                    "frame_index": int(sample.frame_index),
                    "image_reference": (sample.image_reference.relative_path
                                        if sample.image_reference is not None else None),
                    "field": field, "type": kind,
                    "requested_sign": want,
                    "read_back_before": accounting["read_back_before"],
                    "read_back_after": int(after_signs[order, index]),
                    "delta_y_m": delta, "abs_delta_y_mm": abs(delta) * 1000.0,
                    "hinge_error_before_degrees": float(base_error[order, column]),
                    "hinge_error_after_degrees": float(error[order, column]),
                    "flipped_before": was, "flipped_after": now,
                    "frame_mpjpe_before_mm": frame_error(before, order),
                    "frame_mpjpe_after_mm": frame_error(after, order),
                    **geometry,
                })

        deltas = np.array([r["abs_delta_y_mm"] for r in records])
        by_type: dict[str, Any] = {}
        for kind in sorted(types):
            subset = np.array([r["abs_delta_y_mm"] for r in records if r["type"] == kind])
            error_change = np.array([r["hinge_error_after_degrees"] - r["hinge_error_before_degrees"]
                                     for r in records if r["type"] == kind
                                     and np.isfinite(r["hinge_error_after_degrees"])
                                     and np.isfinite(r["hinge_error_before_degrees"])])
            by_type[kind] = {
                "count": int(types[kind]),
                "abs_delta_y_mm": _quantiles(subset),
                "mean_hinge_error_change_degrees": (float(error_change.mean())
                                                    if error_change.size else None),
                "share_of_corrections": float(types[kind] / max(len(records), 1)),
            }

        report["sources"][label] = {
            "candidate": candidate,
            "corrections": len(records),
            "abs_delta_y_mm": _quantiles(deltas),
            "sqrt_f": _quantiles(np.array([r["sqrt_f"] for r in records])),
            "abs_offset_forward_mm": _quantiles(
                np.array([r["abs_offset_forward_m"] * 1000.0 for r in records])),
            "offset_norm_mm": _quantiles(np.array([r["offset_norm_m"] * 1000.0 for r in records])),
            "axis_angle_to_depth_degrees": _quantiles(
                np.array([r["axis_angle_to_depth_degrees"] for r in records])),
            "depth_only_over_minimum_ratio": _quantiles(
                np.array([r["depth_only_over_minimum_ratio"] for r in records])),
            "correlations": _correlations(records),
            "type_accounting": by_type,
            "continuity_counts": {f">{int(t)}mm": int((deltas > t).sum())
                                  for t in CONTINUITY_RANGES_MM},
        }
        ranked = sorted(records, key=lambda r: -r["abs_delta_y_mm"])
        exported["sources"][label] = ranked[:args.export_records]

    write_json(args.out / "hinge_conditioning.json", report)
    write_json(args.out / "hinge_conditioning_records.json", exported)

    for label, entry in report["sources"].items():
        print(f"\n== {label}: {entry['corrections']} corrections")
        q = entry["abs_delta_y_mm"]
        print("   |delta_y| mm  p50=%.1f p90=%.1f p99=%.1f p99.9=%.1f max=%.1f" % (
            q["p50"], q["p90"], q["p99"], q["p99.9"], q["max"]))
        print("   sqrt(f)       p50=%.4f p90=%.4f min-ish(p50 of lowest)=%.4f" % (
            entry["sqrt_f"]["p50"], entry["sqrt_f"]["p90"], entry["sqrt_f"]["p50"]))
        print("   correlations vs log10|delta_y|:")
        for name, value in entry["correlations"].items():
            if isinstance(value, float):
                print("      %-34s r=%+.4f" % (name, value))
        decomposition = entry["correlations"]["log10_decomposition"]
        print("      exact-decomposition max residual: %.2e" % decomposition["max_abs_residual"])
        print("      var log10(2|o_y|)=%.4f  var log10(1/f)=%.4f  cov=%.4f  var total=%.4f" % (
            decomposition["variance_of_log10_2_abs_o_y"], decomposition["variance_of_log10_1_over_f"],
            decomposition["covariance"], decomposition["variance_of_log10_abs_delta_y"]))
        print("   types:")
        for kind, value in entry["type_accounting"].items():
            print("      %-32s n=%5d (%.2f%%)  |dy| p50=%.1f max=%.1f  mean hinge err change=%s" % (
                kind, value["count"], 100 * value["share_of_corrections"],
                value["abs_delta_y_mm"]["p50"], value["abs_delta_y_mm"]["max"],
                "-" if value["mean_hinge_error_change_degrees"] is None
                else "%+.2f deg" % value["mean_hinge_error_change_degrees"]))
        print("   continuity:", entry["continuity_counts"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
