#!/usr/bin/env python3
"""Explain endpoint-fixed swivel final SignState read-back failures.

This is a read-only diagnostic over the exact docs/44 replay inputs. It records
the solver's circle/theta semantics and independently reconstructs the
canonical bend direction used by ``sign_state``. It never changes the solver,
thresholds, or production reconciliation policy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import diagnose_pose_reconciliation_attribution as attribution
import replay_pose_reconciliation as replay
from common.canonical_pose import bend_direction
from common.serialization import write_json
from framepose.bank import load_bank
from framepose.pose_reconciliation import (
    MACHINE_EPSILON,
    _basis,
    _bone_circle,
    reconcile_pose_batch,
)
from framepose.replay_provenance import verify_source_identity
from framepose.signs import (
    HINGE_CHAINS_BY_JOINT,
    SIGN_FIELD_NAMES,
    UNIT_FORWARD_EPSILON,
    UNKNOWN,
    oracle_sign_states,
    sign_state,
)


OBS = replay.R_SWIVEL_OBS
ORACLE = replay.R_SWIVEL_ORACLE_2D
EXPECTED_FAILURES = {OBS: 213, ORACLE: 195}
SOURCE_NAMES = ("stationary", "readability_boundary", "half_angle_infinity")


def _quantiles(values: list[float]) -> dict[str, Any]:
    finite = np.asarray([value for value in values if np.isfinite(value)], dtype=np.float64)
    if not len(finite):
        return {"count": 0}
    return {
        "count": int(len(finite)),
        "mean": float(finite.mean()),
        "p05": float(np.percentile(finite, 5)),
        "p50": float(np.percentile(finite, 50)),
        "p95": float(np.percentile(finite, 95)),
        "min": float(finite.min()),
        "max": float(finite.max()),
    }


def _near_machine_zero(value: float, *scales: float) -> bool:
    scale = max(1.0, *(abs(float(item)) for item in scales))
    return abs(float(value)) <= 16.0 * MACHINE_EPSILON * scale


def _candidate_semantics(pose: np.ndarray, valid: np.ndarray, field: str,
                         requested: int, solver: dict[str, Any]) -> dict[str, Any]:
    """Independently derive both read-back semantics from the attempted M."""
    chain = HINGE_CHAINS_BY_JOINT[field[: -len("_forward_bend")]]
    p_idx, m_idx, d_idx = (replay.JOINT_INDEX[name] for name in chain)
    original_pmd = np.asarray(pose[[p_idx, m_idx, d_idx]], dtype=np.float64)
    circle = _bone_circle(original_pmd[0], original_pmd[1], original_pmd[2])
    point = np.asarray(solver["candidate_point_root_relative"], dtype=np.float64)
    theta = float(solver["theta_radians"])
    radius = float(solver["circle"]["radius_m"])
    basis = _basis(circle)
    if basis is None or circle.center is None or circle.radius is None:
        raise AssertionError("a final read-back attempt must have a readable circle basis")
    u_hat, v_hat, sqrt_f = basis
    expected_point = circle.center + circle.radius * (
        u_hat * math.cos(theta) + v_hat * math.sin(theta))
    candidate = np.asarray(pose, dtype=np.float64).copy()
    candidate[m_idx] = point
    bend = bend_direction(candidate[m_idx], candidate[p_idx], candidate[d_idx])
    canonical_sign = int(sign_state(candidate, valid)[SIGN_FIELD_NAMES.index(field)])
    offset = candidate[m_idx] - circle.center
    circle_residual = float(offset @ offset - circle.radius_squared)
    solver_unit_forward = float(sqrt_f * math.cos(theta))
    solver_cosine_floor = float(UNIT_FORWARD_EPSILON / sqrt_f)
    solver_signed_margin = float(requested * solver_unit_forward - UNIT_FORWARD_EPSILON)
    canonical_forward = float(bend[1]) if bend is not None else math.nan
    canonical_signed_margin = float(requested * canonical_forward - UNIT_FORWARD_EPSILON)
    canonical_absolute_margin = float(abs(canonical_forward) - UNIT_FORWARD_EPSILON)
    basis_residual = float(np.linalg.norm(point - expected_point))
    tolerance = 16.0 * MACHINE_EPSILON * max(1.0, radius, float(np.linalg.norm(expected_point)))

    if basis_residual > tolerance:
        taxonomy = "PARAMETERIZATION_OR_BASIS_MISMATCH"
    elif abs(solver_unit_forward - canonical_forward) > 16.0 * MACHINE_EPSILON:
        taxonomy = "SOLVER_CANONICAL_FORMULA_MISMATCH"
    elif (solver.get("candidate_source") == "readability_boundary"
          and _near_machine_zero(solver_signed_margin, UNIT_FORWARD_EPSILON)
          and _near_machine_zero(canonical_signed_margin, UNIT_FORWARD_EPSILON)):
        taxonomy = "NUMERICAL_READABILITY_BOUNDARY"
    elif canonical_sign == -int(requested):
        taxonomy = "ACTUAL_WRONG_BRANCH_CANDIDATE"
    else:
        taxonomy = "OTHER"

    candidate_pmd = candidate[[p_idx, m_idx, d_idx]]
    before_lengths = (float(np.linalg.norm(original_pmd[1] - original_pmd[0])),
                      float(np.linalg.norm(original_pmd[2] - original_pmd[1])))
    after_lengths = (float(np.linalg.norm(candidate_pmd[1] - candidate_pmd[0])),
                     float(np.linalg.norm(candidate_pmd[2] - candidate_pmd[1])))
    endpoint_errors = (float(np.linalg.norm(candidate_pmd[0] - original_pmd[0])),
                       float(np.linalg.norm(candidate_pmd[2] - original_pmd[2])))
    axis = np.asarray(solver["circle"]["axis"], dtype=np.float64)
    axis_hat = axis / np.linalg.norm(axis)

    return {
        "candidate_P_M_D_root_relative_m": candidate_pmd.tolist(),
        "input_P_M_D_root_relative_m": original_pmd.tolist(),
        "theta_radians": theta,
        "circle_radius_m": radius,
        "circle_center_root_relative_m": circle.center.tolist(),
        "circle_axis": axis.tolist(),
        "circle_axis_unit": axis_hat.tolist(),
        "sqrt_f_in_plane_observability": float(sqrt_f),
        "f_in_plane_fraction": float(sqrt_f * sqrt_f),
        "cosine_theta": float(math.cos(theta)),
        "requested_times_cosine_theta": float(requested * math.cos(theta)),
        "solver_cosine_floor_unit_forward_epsilon_over_sqrt_f": solver_cosine_floor,
        "solver_implied_unit_forward_sqrt_f_times_cosine": solver_unit_forward,
        "solver_signed_margin_from_unit_forward_epsilon": solver_signed_margin,
        "solver_cosine_margin_from_floor": float(
            requested * math.cos(theta) - solver_cosine_floor),
        "canonical_actual_perpendicular_offset_m": float(np.linalg.norm(offset)),
        "canonical_bend_direction": None if bend is None else bend.tolist(),
        "canonical_bend_direction_y": canonical_forward,
        "canonical_signed_requested_margin_from_epsilon": canonical_signed_margin,
        "canonical_absolute_threshold_margin_from_epsilon": canonical_absolute_margin,
        "canonical_sign_state": canonical_sign,
        "canonical_readback": ("UNKNOWN" if canonical_sign == UNKNOWN else str(canonical_sign)),
        "circle_membership_residual_m2_signed": circle_residual,
        "circle_membership_residual_m2_abs": abs(circle_residual),
        "basis_parameterization_residual_m": basis_residual,
        "endpoint_errors_m": {"P": endpoint_errors[0], "D": endpoint_errors[1]},
        "bone_length_errors_m": {
            "P_M": after_lengths[0] - before_lengths[0],
            "M_D": after_lengths[1] - before_lengths[1],
        },
        "taxonomy": taxonomy,
    }


def boundary_probe(side: str) -> dict[str, Any]:
    """Probe adjacent representable angles around the existing exact boundary."""
    if side not in {"below", "equal", "above"}:
        raise ValueError("side must be below, equal, or above")
    p = np.asarray([0.0, 0.0, 0.0])
    d = np.asarray([0.6, 0.0, 0.0])
    radius = 0.4
    alpha = float(np.arccos(UNIT_FORWARD_EPSILON))
    theta = {
        "below": float(np.nextafter(alpha, np.inf)),
        "equal": alpha,
        "above": float(np.nextafter(alpha, -np.inf)),
    }[side]
    m = np.asarray([0.3, radius * np.cos(theta), radius * np.sin(theta)])
    pose = np.zeros((len(replay.JOINT_NAMES), 3), dtype=np.float64)
    valid = np.zeros(len(replay.JOINT_NAMES), dtype=bool)
    p_idx, m_idx, d_idx = (replay.JOINT_INDEX[name] for name in
                           HINGE_CHAINS_BY_JOINT["left_elbow"])
    pose[p_idx], pose[m_idx], pose[d_idx] = p, m, d
    valid[[p_idx, m_idx, d_idx]] = True
    canonical = int(sign_state(pose, valid)[SIGN_FIELD_NAMES.index("left_elbow_forward_bend")])
    bend = bend_direction(m, p, d)
    actual = float(bend[1])
    solver = float(np.cos(theta))
    return {
        "boundary_side": side,
        "boundary_theta_radians": alpha,
        "theta_radians": theta,
        "solver_unit_forward": solver,
        "canonical_bend_direction_y": actual,
        "solver_margin": solver - UNIT_FORWARD_EPSILON,
        "canonical_margin": abs(actual) - UNIT_FORWARD_EPSILON,
        "canonical_sign_state": canonical,
    }


def _source_accounting(entries: list[dict[str, Any]],
                       successes: list[dict[str, Any]]) -> dict[str, Any]:
    attempts = Counter()
    failures = Counter()
    margins: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {"solver": [], "canonical": [], "difference": []})
    for record in entries:
        source = record.get("candidate_source")
        if source:
            attempts[source] += 1
            if record.get("final_readback_failure"):
                failures[source] += 1
    for record in successes:
        source = record.get("candidate_source")
        if not source:
            continue
        semantics = record["semantics"]
        margins[source]["solver"].append(
            semantics["solver_signed_margin_from_unit_forward_epsilon"])
        margins[source]["canonical"].append(
            semantics["canonical_signed_requested_margin_from_epsilon"])
        margins[source]["difference"].append(
            semantics["solver_implied_unit_forward_sqrt_f_times_cosine"]
            - semantics["canonical_bend_direction_y"])
    result = {}
    for source in SOURCE_NAMES:
        result[source] = {
            "attempts_with_this_candidate_source": int(attempts[source]),
            "final_readback_failures": int(failures[source]),
            "failure_rate_given_source": (
                float(failures[source] / attempts[source]) if attempts[source] else None),
            "matched_successful_corrected_row_margin_control": {
                "rows": len(margins[source]["solver"]),
                "solver_signed_margin": _quantiles(margins[source]["solver"]),
                "canonical_signed_margin": _quantiles(margins[source]["canonical"]),
                "solver_minus_canonical_forward_component": _quantiles(
                    margins[source]["difference"]),
            },
        }
    unattributed = sum(1 for record in entries if not record.get("candidate_source"))
    return {"by_candidate_source": result,
            "attempts_without_candidate_source": unattributed}


def run_readback_diagnostic(*, bank_path: Path, prediction_path: Path,
                            evaluation_path: Path, raw_root: Path,
                            replay_report_path: Path) -> dict[str, Any]:
    replay_sha = attribution._sha256(replay_report_path)
    if replay_sha != attribution.EXPECTED_REPLAY_SHA256:
        raise ValueError(f"docs/44 replay SHA mismatch: {replay_sha}")
    original_report = attribution._read_json(replay_report_path)
    bank = load_bank(bank_path)
    positions = bank.indices("test")
    if bank.content_digest() != attribution.EXPECTED_BANK_DIGEST or len(positions) != 7076:
        raise ValueError("the exact docs/44 FrameBank was not recovered")
    identity = verify_source_identity(
        prediction=prediction_path, evaluation=evaluation_path, bank=bank, split="test",
        candidate="O_BILATERAL_oracle_forward_depth_only", frames=len(positions),
        joints=len(replay.JOINT_NAMES))
    if identity["prediction"]["sha256"] != attribution.EXPECTED_PREDICTION_SHA256:
        raise ValueError("O_BILATERAL prediction SHA differs from docs/44")
    if identity["evaluation"]["sha256"] != attribution.EXPECTED_EVALUATION_SHA256:
        raise ValueError("O_BILATERAL evaluation SHA differs from docs/44")
    if original_report.get("source_identity", {}).get("prediction", {}).get("sha256") != identity["prediction"]["sha256"]:
        raise ValueError("replay report is not bound to the recovered prediction")

    h0 = np.load(prediction_path).astype(np.float64)
    valid = bank.arrays["target_valid"][positions]
    observed_valid = bank.arrays["input_valid"][positions]
    observation = bank.arrays["input_2d"][positions].astype(np.float64)
    oracle = oracle_sign_states(bank.arrays["target_3d"], bank.arrays["target_valid"])[positions]
    requested = replay.mask_fields(oracle, list(replay.HINGE_FIELDS)).astype(np.int64)
    before = replay._hinge_sign_matrix(h0, valid)
    target_absolute, _, image_size, _, contexts, provenance, missing, usable = \
        replay._load_camera_state(bank, positions, raw_root, "test")
    if int(usable.sum()) != 7076 or missing or len(provenance) != 24:
        raise ValueError("raw camera coverage differs from docs/44")
    expected_raw = original_report["projection_context"]["raw_provenance"]
    raw_map = lambda rows: {Path(row["path"]).stem: (int(row["bytes"]), row["sha256"])
                            for row in rows}
    if raw_map(provenance) != raw_map(expected_raw):
        raise ValueError("raw camera provenance differs from docs/44")
    oracle_observation = replay._oracle_observation(
        observation, target_absolute, contexts, image_size, valid)

    by_method: dict[str, Any] = {}
    for method, method_observation, method_valid in (
            (OBS, observation, observed_valid), (ORACLE, oracle_observation, valid)):
        failures: list[dict[str, Any]] = []
        controls: list[dict[str, Any]] = []
        per_field: dict[str, Any] = {}
        readback_counts = Counter()
        taxonomy_counts = Counter()
        attempted_by_source = Counter()
        failed_by_source = Counter()
        successful_margin_by_source: dict[str, dict[str, list[float]]] = defaultdict(
            lambda: {"solver": [], "canonical": [], "difference": []})
        for field in replay.HINGE_FIELDS:
            rows = attribution._cohort_rows_from_report(original_report, "C", field)
            column = SIGN_FIELD_NAMES.index(field)
            partition = attribution.partition_c_rows(
                rows, requested[rows, column], before[rows, column])
            readable_rows = partition["readable_wrong"]
            _, batch_reports = reconcile_pose_batch(
                h0[readable_rows], valid[readable_rows], requested[readable_rows],
                method_observation[readable_rows],
                [contexts[int(row)] for row in readable_rows], fields=(field,),
                observed_valid=method_valid[readable_rows])
            attempted = [item["fields"][field] for item in batch_reports]
            failed_records: list[dict[str, Any]] = []
            successful_records: list[dict[str, Any]] = []
            p_idx, m_idx, d_idx = (replay.JOINT_INDEX[name]
                                   for name in HINGE_CHAINS_BY_JOINT[
                                       field[: -len("_forward_bend")]])
            for row, entry in zip(readable_rows, attempted):
                source = entry.get("candidate_source")
                if source:
                    attempted_by_source[source] += 1
                if entry.get("outcome") == "corrected":
                    candidate = h0[int(row)].copy()
                    candidate[m_idx] = np.asarray(
                        entry["candidate_point_root_relative"], dtype=np.float64)
                    semantics = _candidate_semantics(
                        h0[int(row)], valid[int(row)], field,
                        int(requested[int(row), column]), entry)
                    successful_records.append({"candidate_source": source,
                                               "semantics": semantics})
                    if source:
                        values = successful_margin_by_source[source]
                        values["solver"].append(semantics[
                            "solver_signed_margin_from_unit_forward_epsilon"])
                        values["canonical"].append(semantics[
                            "canonical_signed_requested_margin_from_epsilon"])
                        values["difference"].append(
                            semantics["solver_implied_unit_forward_sqrt_f_times_cosine"]
                            - semantics["canonical_bend_direction_y"])
                    controls.append({
                        "field": field,
                        "sample_id": bank.samples[int(positions[int(row)])].sample_id,
                        "candidate_source": source,
                        "solver_margin": semantics[
                            "solver_signed_margin_from_unit_forward_epsilon"],
                        "canonical_margin": semantics[
                            "canonical_signed_requested_margin_from_epsilon"],
                    })
                    continue
                if (entry.get("reason") !=
                        "candidate failed exact endpoint, length, or SignState read-back"):
                    continue
                semantics = _candidate_semantics(
                    h0[int(row)], valid[int(row)], field,
                    int(requested[int(row), column]), entry)
                record = {
                    "field": field,
                    "sample_id": bank.samples[int(positions[int(row)])].sample_id,
                    "sequence_id": bank.samples[int(positions[int(row)])].sequence_id,
                    "frame_index": int(bank.samples[int(positions[int(row)])].frame_index),
                    "requested_sign": int(requested[int(row), column]),
                    "candidate_source": source or "other",
                    "final_readback_failure": True,
                    "solver": {
                        "theta_radians": entry.get("theta_radians"),
                        "circle_radius_m": (entry.get("circle") or {}).get("radius_m"),
                        "in_plane_fraction": entry.get("in_plane_fraction"),
                        "sqrt_f": (math.sqrt(float(entry["in_plane_fraction"]))
                                   if entry.get("in_plane_fraction") is not None else None),
                        "cosine_theta": (semantics["cosine_theta"]),
                        "requested_times_cosine_theta": (
                            semantics["requested_times_cosine_theta"]),
                        "cosine_floor": (
                            UNIT_FORWARD_EPSILON / semantics[
                                "sqrt_f_in_plane_observability"]),
                        "solver_implied_unit_forward": semantics[
                            "solver_implied_unit_forward_sqrt_f_times_cosine"],
                        "signed_margin_from_UNIT_FORWARD_EPSILON": semantics[
                            "solver_signed_margin_from_unit_forward_epsilon"],
                        "minimum_reprojection_error_px": entry.get(
                            "minimum_reprojection_error_px"),
                    },
                    "solver_circle": entry.get("circle"),
                    "semantics": semantics,
                }
                failures.append(record)
                failed_records.append(record)
                failed_by_source[source or "other"] += 1
                readback_counts[semantics["canonical_readback"]] += 1
                taxonomy_counts[semantics["taxonomy"]] += 1

            if len(failed_records) + sum(
                    item.get("outcome") == "corrected" for item in attempted) != len(readable_rows):
                # Rows with a distinct non-read-back refusal are not part of the
                # target population; the reported exact failure count is checked below.
                pass
            per_field[field] = {
                "C_READABLE_WRONG_rows": int(len(readable_rows)),
                "final_readback_failure_rows": len(failed_records),
                "successful_corrected_control_rows": len(successful_records),
                "readback": dict(Counter(record["semantics"]["canonical_readback"]
                                          for record in failed_records)),
                "taxonomy": dict(Counter(record["semantics"]["taxonomy"]
                                          for record in failed_records)),
            }

        expected = EXPECTED_FAILURES[method]
        if len(failures) != expected:
            raise AssertionError(
                f"{method} exact docs/45 failure population changed: {len(failures)} != {expected}")
        source_rows = {}
        for source in SOURCE_NAMES:
            values = successful_margin_by_source[source]
            source_rows[source] = {
                "attempts_with_this_candidate_source": int(attempted_by_source[source]),
                "final_readback_failures": int(failed_by_source[source]),
                "failure_rate_given_source": (
                    float(failed_by_source[source] / attempted_by_source[source])
                    if attempted_by_source[source] else None),
                "successful_corrected_control": {
                    "rows": len(values["solver"]),
                    "solver_signed_margin": _quantiles(values["solver"]),
                    "canonical_signed_margin": _quantiles(values["canonical"]),
                    "solver_minus_canonical_forward_component": _quantiles(
                        values["difference"]),
                },
            }
        method_records = failures
        method_records.sort(key=lambda row: (
            replay.HINGE_FIELDS.index(row["field"]), row["sample_id"]))
        by_method[method] = {
            "failure_rows": len(failures),
            "per_field": per_field,
            "canonical_readback_counts": dict(readback_counts),
            "taxonomy_counts": dict(taxonomy_counts),
            "candidate_source_attribution": source_rows,
            "matched_successful_corrected_margin_control": {
                "population": "all successful R_SWIVEL rows in the same C_READABLE_WRONG field rows",
                "total_rows": sum(value["successful_corrected_control_rows"]
                                  for value in per_field.values()),
                "by_candidate_source": {
                    source: row["successful_corrected_control"]
                    for source, row in source_rows.items()},
            },
            "rows": method_records,
        }

    return {
        "schema": "animcv_pose_reconciliation_readback_diagnosis_v1",
        "scope": "diagnostic only; production solver and SignState contract unchanged",
        "identity": {
            "docs44_replay_sha256": replay_sha,
            "bank_content_digest": bank.content_digest(),
            "prediction_sha256": identity["prediction"]["sha256"],
            "evaluation_sha256": identity["evaluation"]["sha256"],
            "raw_camera_file_count": len(provenance),
        },
        "contract": {
            "unit_forward_epsilon": UNIT_FORWARD_EPSILON,
            "solver_cosine_floor": "UNIT_FORWARD_EPSILON / sqrt_f",
            "solver_implied_unit_forward": "sqrt_f * cos(theta)",
            "canonical": "bend_direction(M,P,D)[Y], UNKNOWN iff abs(Y) < UNIT_FORWARD_EPSILON or chain/offset invalid",
            "classification_precision": "16 * float64 machine epsilon with unit scale; existing contract only",
        },
        "by_method": by_method,
        "synthetic_boundary_probes": {
            "just_below": boundary_probe("below"),
            "mathematically_equal": boundary_probe("equal"),
            "just_above": boundary_probe("above"),
        },
        "no_policy_change": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", required=True, type=Path)
    parser.add_argument("--prediction", required=True, type=Path)
    parser.add_argument("--evaluation", required=True, type=Path)
    parser.add_argument("--raw-root", required=True, type=Path)
    parser.add_argument("--replay-report", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    result = run_readback_diagnostic(
        bank_path=args.bank, prediction_path=args.prediction,
        evaluation_path=args.evaluation, raw_root=args.raw_root,
        replay_report_path=args.replay_report)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.out, result)
    print(json.dumps({
        method: {"failures": value["failure_rows"],
                 "canonical_readbacks": value["canonical_readback_counts"],
                 "taxonomies": value["taxonomy_counts"],
                 "sources": value["candidate_source_attribution"]}
        for method, value in result["by_method"].items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
