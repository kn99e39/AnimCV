#!/usr/bin/env python3
"""Diagnostic-only causal attribution for the frozen docs/44 replay.

This script does not alter any solver or policy. It reuses the exact docs/44
inputs/report, partitions the predeclared C populations by H0 SignState
read-back, accounts for current refusal reasons, and calls the existing
bone-circle swivel mathematics directly for an explicitly labelled
counterfactual that bypasses only the H0-UNKNOWN early refusal.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import replay_pose_reconciliation as replay
from common.canonical_pose import JOINT_INDEX, bend_direction
from framepose.bank import load_bank
from framepose.branch_constraints import (
    CORRECTED as BRANCH_CORRECTED,
    DEPTH_ONLY,
    MINIMUM_NORM,
    UNRESOLVED as BRANCH_UNRESOLVED,
    apply_branch_constraints_batch,
)
from framepose.pose_reconciliation import (
    CORRECTED as SWIVEL_CORRECTED,
    MACHINE_EPSILON,
    ProjectionContext,
    UNRESOLVED as SWIVEL_UNRESOLVED,
    _branch_readability,
    _bone_circle,
    _solve_swivel,
    reconcile_pose_batch,
)
from framepose.replay_provenance import verify_source_identity
from framepose.signs import HINGE_CHAINS_BY_JOINT, SIGN_FIELD_NAMES, UNKNOWN, sign_state


EXPECTED_REPLAY_SHA256 = "61fde30e7e6d8d9769a0273450819478fc6c457af3841eae3d0333c1f8e16e8c"
EXPECTED_BANK_DIGEST = "75519e6394a764e3749ddaa30555b58b73a01db582ccee14f661374b9a0ed536"
EXPECTED_PREDICTION_SHA256 = "6df04c2031b2bde6a4f0af37fdae0a7f0228e9fc23401e15e1aa2b3bd35695a5"
EXPECTED_EVALUATION_SHA256 = "a5881850b3769ca2329271d327ee895dc646f236451f05c3ba2e0ff917f580a3"

H0_UNKNOWN_REFUSAL = "current_h0_branch_unreadable_early_refusal"
GEOMETRIC_REASONS = {
    "degenerate_bone_circle",
    "bend_radius_below_existing_floor",
    "axis_or_branch_unobservable",
    "no_finite_positive_depth_candidate",
}
FAILURE_REASONS = {
    "final_sign_readback_failure",
    "endpoint_bone_invariant_failure",
}
REASON_BUCKETS = (
    H0_UNKNOWN_REFUSAL,
    "chain_invalid",
    "observed_middle_invalid",
    "projection_context_invalid_or_unavailable",
    "degenerate_bone_circle",
    "bend_radius_below_existing_floor",
    "axis_or_branch_unobservable",
    "no_finite_positive_depth_candidate",
    "final_sign_readback_failure",
    "endpoint_bone_invariant_failure",
    "other",
)


def partition_c_rows(rows: Any, requested: Any, h0_readback: Any) -> dict[str, np.ndarray]:
    """Partition exact C chain rows into readable conflicts and H0-UNKNOWN."""
    rows = np.asarray(rows, dtype=np.int64)
    requested = np.asarray(requested, dtype=np.int64)
    h0_readback = np.asarray(h0_readback, dtype=np.int64)
    if requested.shape != rows.shape or h0_readback.shape != rows.shape:
        raise ValueError("C rows, requested signs, and H0 read-backs must align")
    known = requested != UNKNOWN
    readable_wrong = known & np.isin(h0_readback, (-1, 1)) & (h0_readback != requested)
    h0_unknown = known & (h0_readback == UNKNOWN)
    if not np.all(readable_wrong | h0_unknown):
        raise AssertionError("C contains a row outside the known-conflict / H0-UNKNOWN partition")
    readable_rows = rows[readable_wrong]
    unknown_rows = rows[h0_unknown]
    if np.intersect1d(readable_rows, unknown_rows).size:
        raise AssertionError("C_READABLE_WRONG and C_H0_UNKNOWN overlap")
    if not np.array_equal(np.sort(np.concatenate((readable_rows, unknown_rows))), np.sort(rows)):
        raise AssertionError("C != C_READABLE_WRONG union C_H0_UNKNOWN")
    return {"readable_wrong": readable_rows, "h0_unknown": unknown_rows}


def unresolved_reason_bucket(entry: dict[str, Any]) -> str:
    """Map existing report reason strings into the requested exclusive taxonomy."""
    reason = str(entry.get("reason", "unspecified"))
    normalized = reason.lower()
    if reason == "current hinge branch is unreadable; no guess":
        return H0_UNKNOWN_REFUSAL
    if "hinge chain validity is insufficient" in normalized or "chain joint is invalid" in normalized:
        return "chain_invalid"
    if "observed middle joint is invalid" in normalized or "observed middle 2d" in normalized:
        return "observed_middle_invalid"
    if "projectioncontext" in normalized or "projection context" in normalized or "intrinsics" in normalized:
        return "projection_context_invalid_or_unavailable"
    if "bone-preserving locus is degenerate" in normalized or "circle is degenerate" in normalized:
        return "degenerate_bone_circle"
    if "circle radius is below" in normalized or "bend floor" in normalized:
        return "bend_radius_below_existing_floor"
    if ("camera-depth branch is not observable" in normalized or
            "depth basis is undefined" in normalized or
            "no swivel point can satisfy" in normalized):
        return "axis_or_branch_unobservable"
    if "no finite positive-depth readable swivel candidate" in normalized:
        return "no_finite_positive_depth_candidate"
    if "candidate failed exact endpoint, length, or signstate read-back" in normalized:
        read_back = entry.get("read_back_after_attempt")
        requested = entry.get("requested_sign")
        if read_back is not None and requested is not None and int(read_back) != int(requested):
            return "final_sign_readback_failure"
        return "endpoint_bone_invariant_failure"
    return "other"


def unresolved_reason_accounting(entries: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Count unresolved causes and enforce an exact partition identity."""
    counts = Counter({name: 0 for name in REASON_BUCKETS})
    unresolved = 0
    for entry in entries:
        if entry.get("outcome") != SWIVEL_UNRESOLVED:
            continue
        unresolved += 1
        counts[unresolved_reason_bucket(entry)] += 1
    if sum(counts.values()) != unresolved:
        raise AssertionError("unresolved reason buckets do not partition unresolved rows")
    return {
        "unresolved": unresolved,
        "by_reason": dict(counts),
        "reason_identity_holds": sum(counts.values()) == unresolved,
    }


def decompose_coverage_gap(minimum_outcomes: Any, swivel_entries: Iterable[dict[str, Any]],
                           h0_unknown: Any) -> dict[str, Any]:
    """Close MINIMUM_NORM corrected minus current swivel corrected arithmetically.

    Positive contributions count MINIMUM_NORM-corrected rows refused by the
    swivel, categorized by the swivel's observed refusal reason. Rows corrected
    only by the swivel are reported as a negative counter-contribution.
    """
    minimum_outcomes = np.asarray(minimum_outcomes, dtype=object)
    entries = list(swivel_entries)
    h0_unknown = np.asarray(h0_unknown, dtype=bool)
    if minimum_outcomes.shape != h0_unknown.shape or len(entries) != len(h0_unknown):
        raise ValueError("coverage-gap inputs must align row-for-row")
    positive = Counter({name: 0 for name in (
        "h0_unknown_early_refusal", "true_swivel_geometric_infeasibility",
        "final_readback_or_invariant_failure", "other")})
    swivel_only = 0
    minimum_corrected = 0
    swivel_corrected = 0
    for minimum_outcome, entry, is_unknown in zip(minimum_outcomes, entries, h0_unknown):
        minimum_ok = minimum_outcome == BRANCH_CORRECTED
        swivel_ok = entry.get("outcome") == SWIVEL_CORRECTED
        minimum_corrected += int(minimum_ok)
        swivel_corrected += int(swivel_ok)
        if minimum_ok and not swivel_ok:
            if is_unknown:
                positive["h0_unknown_early_refusal"] += 1
            else:
                reason = unresolved_reason_bucket(entry)
                if reason in GEOMETRIC_REASONS:
                    positive["true_swivel_geometric_infeasibility"] += 1
                elif reason in FAILURE_REASONS:
                    positive["final_readback_or_invariant_failure"] += 1
                else:
                    positive["other"] += 1
        elif swivel_ok and not minimum_ok:
            swivel_only += 1
    gap = minimum_corrected - swivel_corrected
    attributed = sum(positive.values()) - swivel_only
    if gap != attributed:
        raise AssertionError(f"coverage gap does not close: {gap} != {attributed}")
    return {
        "minimum_norm_corrected": minimum_corrected,
        "swivel_corrected": swivel_corrected,
        "coverage_gap": gap,
        "positive_contributions": dict(positive),
        "swivel_only_corrected_counter_contribution": swivel_only,
        "decomposed_gap": attributed,
        "identity_holds": gap == attributed,
    }


def partition_outcomes(rows: Any, outcomes: Any,
                       corrected: str = SWIVEL_CORRECTED,
                       unresolved: str = SWIVEL_UNRESOLVED) -> dict[str, np.ndarray]:
    """Partition a fixed population into corrected and unresolved rows."""
    rows = np.asarray(rows, dtype=np.int64)
    outcomes = np.asarray(outcomes, dtype=object)
    if outcomes.shape != rows.shape:
        raise ValueError("rows and outcomes must align")
    corrected_rows = rows[outcomes == corrected]
    unresolved_rows = rows[outcomes == unresolved]
    if len(corrected_rows) + len(unresolved_rows) != len(rows):
        raise AssertionError("outcomes contain an unaccounted state")
    if np.intersect1d(corrected_rows, unresolved_rows).size:
        raise AssertionError("corrected and unresolved partitions overlap")
    if not np.array_equal(np.sort(np.concatenate((corrected_rows, unresolved_rows))), np.sort(rows)):
        raise AssertionError("corrected/unresolved rows do not cover the original population")
    return {"corrected": corrected_rows, "unresolved": unresolved_rows}


def counterfactual_swivel_without_prestate_refusal(
        pose: Any, valid: Any, requested: int, observed_middle_2d: Any,
        context: ProjectionContext | None, field: str, *, observed_valid: bool = True
        ) -> dict[str, Any]:
    """Invoke existing swivel geometry while bypassing only the H0-UNKNOWN guard.

    This diagnostic never mutates the input pose and never calls or changes
    ``reconcile_hinge``. All other existing feasibility, SignState, endpoint,
    and bone-length checks are applied.
    """
    original = np.asarray(pose, dtype=np.float64)
    snapshot = original.copy()
    validity = np.asarray(valid, dtype=bool)
    if int(requested) not in (-1, 1):
        raise ValueError("counterfactual requires a known requested sign")
    joint = field[: -len("_forward_bend")]
    chain = HINGE_CHAINS_BY_JOINT[joint]
    p, m, d = (JOINT_INDEX[name] for name in chain)
    base: dict[str, Any] = {"requested_sign": int(requested), "field": field,
                             "label": "COUNTERFACTUAL_SWIVEL_WITHOUT_PRESTATE_REFUSAL"}
    if not all(bool(validity[index]) for index in (p, m, d)):
        return {**base, "status": "no_feasible_solution", "reason_bucket": "chain_invalid"}
    if not observed_valid:
        return {**base, "status": "no_feasible_solution", "reason_bucket": "observed_middle_invalid"}
    if context is None or context.validation_error() is not None:
        return {**base, "status": "no_feasible_solution",
                "reason_bucket": "projection_context_invalid_or_unavailable"}
    try:
        observed_pixels = context.observation_pixels(observed_middle_2d)
    except ValueError as error:
        return {**base, "status": "no_feasible_solution", "reason_bucket": "observed_middle_invalid",
                "reason": str(error)}

    circle = _bone_circle(original[p], original[m], original[d])
    solved = _solve_swivel(circle, observed_pixels, context, int(requested))
    if not solved.get("resolved"):
        reason = str(solved.get("reason", ""))
        if "no finite positive-depth readable swivel candidate" in reason:
            status, bucket = "positive_depth_failure", "no_finite_positive_depth_candidate"
        else:
            bucket = unresolved_reason_bucket({"reason": reason})
            status = "no_feasible_solution"
        return {**base, "status": status, "reason_bucket": bucket,
                "reason": reason, "circle": solved.get("circle", circle.to_dict())}

    candidate = original.copy()
    candidate[m] = np.asarray(solved["candidate_point_root_relative"], dtype=np.float64)
    sign_column = SIGN_FIELD_NAMES.index(field)
    read_back = int(sign_state(candidate, validity)[sign_column])
    length_errors = [
        abs(float(np.linalg.norm(candidate[m] - candidate[p])) - circle.proximal_length),
        abs(float(np.linalg.norm(candidate[d] - candidate[m])) - circle.distal_length),
    ]
    endpoints_unchanged = (np.array_equal(candidate[p], original[p]) and
                           np.array_equal(candidate[d], original[d]))
    if read_back != requested:
        result = {**base, "status": "read_back_failure", "reason_bucket": "final_sign_readback_failure",
                  "read_back_after_attempt": read_back}
    elif not endpoints_unchanged or max(length_errors) > 16.0 * MACHINE_EPSILON:
        result = {**base, "status": "no_feasible_solution",
                  "reason_bucket": "endpoint_bone_invariant_failure",
                  "read_back_after_attempt": read_back}
    else:
        current_readability = _branch_readability(circle, original[m])
        current_theta = float(np.arctan2(
            current_readability["c_screen_m"], current_readability["c_depth_m"]))
        result = {
            **base,
            "status": "feasible_requested_sign_solution",
            "reason_bucket": "feasible",
            "read_back_after_attempt": read_back,
            "minimum_reprojection_error_px": solved["minimum_reprojection_error_px"],
            "candidate_source": solved["candidate_source"],
            "swivel_delta_radians": float((solved["theta_radians"] - current_theta + np.pi) %
                                           (2.0 * np.pi) - np.pi),
            "bone_length_abs_error_m": length_errors,
            "endpoint_positions_unchanged": True,
            "state": candidate,
        }
    if not np.array_equal(original, snapshot):
        raise AssertionError("counterfactual changed the caller's pose")
    return result


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _cohort_rows_from_report(report: dict[str, Any], cohort: str, field: str,
                             *, wrong_sign: bool = False) -> np.ndarray:
    if wrong_sign:
        payload = report["wrong_sign_endpoint"][replay.R_SWIVEL_OBS]
        fields = payload["cohorts"][cohort]["fields"]
    else:
        fields = report["cohorts"][cohort]["fields"]
    return np.asarray(fields[field]["frame_indices"], dtype=np.int64)


def _reason_report_for_field(rows: np.ndarray, reports: dict[int, dict[str, Any]]) -> dict[str, Any]:
    return unresolved_reason_accounting(reports[int(row)] for row in rows)


def _bend_displacement_degrees(before: np.ndarray, after: np.ndarray, field: str,
                               rows: np.ndarray) -> dict[str, Any]:
    chain = HINGE_CHAINS_BY_JOINT[field[: -len("_forward_bend")]]
    p, m, d = (JOINT_INDEX[name] for name in chain)
    angles = []
    for row in rows:
        first = bend_direction(before[row, m], before[row, p], before[row, d])
        second = bend_direction(after[row, m], after[row, p], after[row, d])
        if first is None or second is None:
            continue
        angles.append(float(np.degrees(np.arccos(np.clip(first @ second, -1.0, 1.0)))))
    return replay._quantiles(angles)


def _image_metrics(state: np.ndarray, baseline: np.ndarray, observation: np.ndarray,
                   target_absolute: np.ndarray, contexts: list[ProjectionContext | None],
                   field: str, rows: np.ndarray) -> dict[str, Any]:
    middle = replay.MIDDLE_INDEX[field]
    correction, target_error, observation_error = [], [], []
    for row in rows:
        context = contexts[int(row)]
        if context is None or context.validation_error() is not None:
            continue
        after = context.project_root_relative(state[row, middle])
        before = context.project_root_relative(baseline[row, middle])
        target = context.project_root_relative(
            target_absolute[row, middle] - np.asarray(context.root_offset_camera))
        observed = context.observation_pixels(observation[row, middle])
        correction.append(float(np.linalg.norm(after - before)))
        target_error.append(float(np.linalg.norm(after - target)))
        observation_error.append(float(np.linalg.norm(after - observed)))
    return {
        "correction_induced_image_displacement_px": replay._quantiles(correction),
        "target_projection_error_px": replay._quantiles(target_error),
        "observation_consistency_error_px": replay._quantiles(observation_error),
    }


def _endpoint_metrics(state: np.ndarray, baseline: np.ndarray, field: str,
                      rows: np.ndarray) -> dict[str, Any]:
    chain = HINGE_CHAINS_BY_JOINT[field[: -len("_forward_bend")]]
    p, _, d = (JOINT_INDEX[name] for name in chain)
    return {
        "proximal_endpoint_displacement_mm": replay._quantiles(
            np.linalg.norm(state[rows, p] - baseline[rows, p], axis=-1) * 1000.0),
        "distal_endpoint_displacement_mm": replay._quantiles(
            np.linalg.norm(state[rows, d] - baseline[rows, d], axis=-1) * 1000.0),
    }


def _candidate_field_metrics(bank, positions: np.ndarray, valid: np.ndarray,
                             requested: np.ndarray, h0: np.ndarray,
                             states: dict[str, np.ndarray],
                             reports: dict[str, dict[str, dict[int, dict[str, Any]]]],
                             observation: np.ndarray, target_absolute: np.ndarray,
                             contexts: list[ProjectionContext | None], field: str,
                             rows: np.ndarray) -> dict[str, Any]:
    column = SIGN_FIELD_NAMES.index(field)
    rows = np.asarray(rows, dtype=np.int64)
    result: dict[str, Any] = {}
    for name, state in states.items():
        known = rows[requested[rows, column] != UNKNOWN]
        signs = replay._hinge_sign_matrix(state[known], valid[known])[:, column] if len(known) else np.asarray([])
        if name == "H0":
            outcome_counts = {"corrected": 0, "unresolved": 0, "already_satisfied": 0}
        else:
            entries = [reports[name][field][int(row)] for row in rows]
            counts = Counter(entry["outcome"] for entry in entries)
            outcome_counts = {"corrected": int(counts.get(SWIVEL_CORRECTED if name.startswith("R_") else BRANCH_CORRECTED, 0)),
                              "unresolved": int(counts.get(SWIVEL_UNRESOLVED if name.startswith("R_") else BRANCH_UNRESOLVED, 0)),
                              "already_satisfied": int(counts.get("already_satisfied", 0))}
            if sum(outcome_counts.values()) != len(known):
                raise AssertionError(f"requested-sign outcomes do not close for {name}/{field}")
        metrics = replay._field_hinge_accounting(
            bank, positions, state, field, rows, f"attribution+{name}+{field}")
        bone = replay._field_bone_accounting(state, h0, field, rows)
        corrected_rows = rows
        if name != "H0":
            expected_corrected = SWIVEL_CORRECTED if name.startswith("R_") else BRANCH_CORRECTED
            corrected_rows = np.asarray([
                row for row in rows
                if reports[name][field][int(row)]["outcome"] == expected_corrected
            ], dtype=np.int64)
        result[name] = {
            "chain_frame_count": int(len(rows)),
            "requested_sign": {
                "requested_count": int(len(known)),
                "satisfied_count": int(np.sum(signs == requested[known, column])) if len(known) else 0,
                **outcome_counts,
            },
            "hinge": metrics,
            "bone": bone,
            "endpoints": _endpoint_metrics(state, h0, field, rows),
            "image": _image_metrics(state, h0, observation, target_absolute, contexts, field, rows),
            "corrected_subset": {
                "frame_count": int(len(corrected_rows)),
                "hinge": replay._field_hinge_accounting(
                    bank, positions, state, field, corrected_rows,
                    f"attribution+{name}+{field}+corrected-only"),
                "bone": replay._field_bone_accounting(state, h0, field, corrected_rows),
                "endpoints": _endpoint_metrics(state, h0, field, corrected_rows),
                "image": _image_metrics(
                    state, h0, observation, target_absolute, contexts, field, corrected_rows),
            },
            "bend_direction_displacement_degrees": (
                {"count": 0} if name == "H0" else
                _bend_displacement_degrees(h0, state, field, corrected_rows)),
        }
    return result


def _run_field_candidate(field: str, rows: np.ndarray, h0: np.ndarray, valid: np.ndarray,
                         requested: np.ndarray, observation: np.ndarray,
                         oracle_observation: np.ndarray,
                         contexts: list[ProjectionContext | None], observed_valid: np.ndarray,
                         ) -> tuple[dict[str, np.ndarray], dict[str, dict[int, dict[str, Any]]]]:
    rows = np.asarray(rows, dtype=np.int64)
    state_rows: dict[str, np.ndarray] = {"H0": h0[rows].copy()}
    reports_rows: dict[str, dict[int, dict[str, Any]]] = {}
    for name, policy in ((DEPTH_ONLY, DEPTH_ONLY), (MINIMUM_NORM, MINIMUM_NORM)):
        candidate, batch_reports = apply_branch_constraints_batch(
            h0[rows], valid[rows], requested[rows], fields=(field,), hinge_write_policy=policy)
        state_rows[name] = candidate
        reports_rows[name] = {
            int(row): batch_reports[index]["fields"][field]
            for index, row in enumerate(rows)
        }
    for name, obs, obs_valid in (
            (replay.R_SWIVEL_OBS, observation, observed_valid),
            (replay.R_SWIVEL_ORACLE_2D, oracle_observation, valid)):
        candidate, batch_reports = reconcile_pose_batch(
            h0[rows], valid[rows], requested[rows], obs[rows],
            [contexts[int(row)] for row in rows], fields=(field,), observed_valid=obs_valid[rows])
        state_rows[name] = candidate
        reports_rows[name] = {
            int(row): batch_reports[index]["fields"][field]
            for index, row in enumerate(rows)
        }
    return state_rows, reports_rows


def _summarize_counterfactual(rows: np.ndarray, field: str, pose: np.ndarray, valid: np.ndarray,
                              requested: np.ndarray, observation: np.ndarray,
                              contexts: list[ProjectionContext | None], observed_valid: np.ndarray,
                              bank, positions: np.ndarray, target_absolute: np.ndarray,
                              h0: np.ndarray) -> dict[str, Any]:
    candidate = h0.copy()
    entries = []
    feasible_rows = []
    objectives = []
    column = SIGN_FIELD_NAMES.index(field)
    middle = replay.MIDDLE_INDEX[field]
    for row in rows:
        entry = counterfactual_swivel_without_prestate_refusal(
            pose[row], valid[row], int(requested[row, column]), observation[row, middle],
            contexts[int(row)], field, observed_valid=bool(observed_valid[row, middle]))
        entries.append(entry)
        if entry["status"] == "feasible_requested_sign_solution":
            candidate[row, middle] = entry["state"][middle]
            feasible_rows.append(int(row))
            objectives.append(float(entry["minimum_reprojection_error_px"]))
    counts = Counter(entry["status"] for entry in entries)
    result: dict[str, Any] = {
        "label": "COUNTERFACTUAL_SWIVEL_WITHOUT_PRESTATE_REFUSAL",
        "total": int(len(rows)),
        "feasible_requested_sign_solution": int(counts.get("feasible_requested_sign_solution", 0)),
        "no_feasible_solution": int(counts.get("no_feasible_solution", 0)),
        "positive_depth_failure": int(counts.get("positive_depth_failure", 0)),
        "read_back_failure": int(counts.get("read_back_failure", 0)),
        "endpoint_bone_invariant_failure": int(sum(
            entry.get("reason_bucket") == "endpoint_bone_invariant_failure" for entry in entries)),
        "other_solver_failure": int(sum(
            entry["status"] == "no_feasible_solution" and entry.get("reason_bucket") not in
            ("endpoint_bone_invariant_failure", "chain_invalid", "observed_middle_invalid",
             "projection_context_invalid_or_unavailable") for entry in entries)),
        "solver_objective_error_px": replay._quantiles(objectives),
        "feasible_rows": feasible_rows,
    }
    if sum(result[key] for key in (
            "feasible_requested_sign_solution", "no_feasible_solution",
            "positive_depth_failure", "read_back_failure")) != len(rows):
        raise AssertionError("counterfactual feasibility outcomes do not partition input rows")
    if feasible_rows:
        result["continuous_geometry_on_feasible_rows"] = replay._field_hinge_accounting(
            bank, positions, candidate, field, np.asarray(feasible_rows, dtype=np.int64),
            f"counterfactual+{field}")
        result["image_metrics_on_feasible_rows"] = _image_metrics(
            candidate, h0, observation, target_absolute, contexts, field,
            np.asarray(feasible_rows, dtype=np.int64))
    return result


def _pool_reason_counts(per_field: dict[str, Any]) -> dict[str, Any]:
    pooled = Counter({name: 0 for name in REASON_BUCKETS})
    unresolved = 0
    for value in per_field.values():
        unresolved += value["unresolved"]
        pooled.update(value["by_reason"])
    if sum(pooled.values()) != unresolved:
        raise AssertionError("pooled reason taxonomy does not close")
    return {"unresolved": unresolved, "by_reason": dict(pooled), "reason_identity_holds": True}


def run_diagnostic(*, bank_path: Path, prediction_path: Path, evaluation_path: Path,
                   raw_root: Path, replay_report_path: Path) -> dict[str, Any]:
    """Analyze the exact docs/44 real-scene populations without changing policy."""
    replay_sha = _sha256(replay_report_path)
    if replay_sha != EXPECTED_REPLAY_SHA256:
        raise ValueError(f"docs/44 replay SHA mismatch: {replay_sha}")
    original_report = _read_json(replay_report_path)
    bank = load_bank(bank_path)
    positions = bank.indices("test")
    if bank.content_digest() != EXPECTED_BANK_DIGEST or len(positions) != 7076:
        raise ValueError("the exact docs/44 bank identity was not recovered")
    identity = verify_source_identity(
        prediction=prediction_path, evaluation=evaluation_path, bank=bank, split="test",
        candidate="O_BILATERAL_oracle_forward_depth_only", frames=len(positions),
        joints=len(replay.JOINT_NAMES))
    if identity["prediction"]["sha256"] != EXPECTED_PREDICTION_SHA256:
        raise ValueError("O_BILATERAL prediction SHA differs from docs/44")
    if identity["evaluation"]["sha256"] != EXPECTED_EVALUATION_SHA256:
        raise ValueError("O_BILATERAL evaluation SHA differs from docs/44")
    if original_report.get("source_identity", {}).get("prediction", {}).get("sha256") != EXPECTED_PREDICTION_SHA256:
        raise ValueError("replay report is not bound to the accepted O_BILATERAL prediction")

    h0 = np.load(prediction_path).astype(np.float64)
    valid = bank.arrays["target_valid"][positions]
    observed_valid = bank.arrays["input_valid"][positions]
    observation = bank.arrays["input_2d"][positions].astype(np.float64)
    requested = replay._requested_signs(bank, positions)
    opposite = replay._requested_signs(bank, positions, opposite=True)
    target_absolute, _, image_size, _, contexts, raw_provenance, missing, usable = \
        replay._load_camera_state(bank, positions, raw_root, "test")
    if int(usable.sum()) != 7076 or missing or len(raw_provenance) != 24:
        raise ValueError("camera provenance no longer matches exact docs/44 coverage")
    replay_raw = original_report["projection_context"]["raw_provenance"]
    def raw_digest_map(items):
        return {Path(item["path"]).stem: (int(item["bytes"]), item["sha256"]) for item in items}
    if raw_digest_map(raw_provenance) != raw_digest_map(replay_raw):
        raise ValueError("raw camera pickle bytes/SHA differ from docs/44 replay")

    before = replay._hinge_sign_matrix(h0, valid)
    oracle_observation = replay._oracle_observation(
        observation, target_absolute, contexts, image_size, valid)
    state_names = ("H0", DEPTH_ONLY, MINIMUM_NORM, replay.R_SWIVEL_OBS,
                   replay.R_SWIVEL_ORACLE_2D)
    states = {name: h0.copy() for name in state_names}
    reports: dict[str, dict[str, dict[int, dict[str, Any]]]] = {
        name: {} for name in state_names if name != "H0"
    }
    partitions: dict[str, dict[str, np.ndarray]] = {}
    c_rows_by_field: dict[str, np.ndarray] = {}
    readable_rows_by_field: dict[str, np.ndarray] = {}
    unknown_rows_by_field: dict[str, np.ndarray] = {}

    for field in replay.HINGE_FIELDS:
        rows = _cohort_rows_from_report(original_report, "C", field)
        column = SIGN_FIELD_NAMES.index(field)
        partition = partition_c_rows(rows, requested[rows, column], before[rows, column])
        c_rows_by_field[field] = rows
        readable_rows_by_field[field] = partition["readable_wrong"]
        unknown_rows_by_field[field] = partition["h0_unknown"]
        partitions[field] = {
            "C": int(len(rows)),
            "C_READABLE_WRONG": int(len(partition["readable_wrong"])),
            "C_H0_UNKNOWN": int(len(partition["h0_unknown"])),
            "union_identity_holds": len(partition["readable_wrong"]) + len(partition["h0_unknown"]) == len(rows),
            "overlap": int(np.intersect1d(partition["readable_wrong"], partition["h0_unknown"]).size),
        }
        fields_states, fields_reports = _run_field_candidate(
            field, rows, h0, valid, requested, observation, oracle_observation,
            contexts, observed_valid)
        for name, state in fields_states.items():
            if name == "H0":
                continue
            middle = replay.MIDDLE_INDEX[field]
            states[name][rows, middle] = state[:, middle]
            reports[name][field] = fields_reports[name]
    pooled_partitions = {
        key: sum(value[key] for value in partitions.values())
        for key in ("C", "C_READABLE_WRONG", "C_H0_UNKNOWN")
    }
    if any(value["overlap"] != 0 or not value["union_identity_holds"] for value in partitions.values()):
        raise AssertionError("C partition identity failed")

    unresolved_reasons: dict[str, Any] = {}
    for method in (replay.R_SWIVEL_OBS, replay.R_SWIVEL_ORACLE_2D):
        unresolved_reasons[method] = {
            field: _reason_report_for_field(c_rows_by_field[field], reports[method][field])
            for field in replay.HINGE_FIELDS
        }
        unresolved_reasons[method]["pooled"] = _pool_reason_counts(
            {field: unresolved_reasons[method][field] for field in replay.HINGE_FIELDS})

    coverage_gap: dict[str, Any] = {}
    for method in (replay.R_SWIVEL_OBS, replay.R_SWIVEL_ORACLE_2D):
        by_field = {}
        for field in replay.HINGE_FIELDS:
            rows = c_rows_by_field[field]
            _, minimum_reports = apply_branch_constraints_batch(
                h0[rows], valid[rows], requested[rows], fields=(field,),
                hinge_write_policy=MINIMUM_NORM)
            minimum_outcomes = [report["fields"][field]["outcome"] for report in minimum_reports]
            swivel_entries = [reports[method][field][int(row)] for row in rows]
            unknown_mask = before[rows, SIGN_FIELD_NAMES.index(field)] == UNKNOWN
            by_field[field] = decompose_coverage_gap(minimum_outcomes, swivel_entries, unknown_mask)
        pooled = {key: sum(value[key] for value in by_field.values())
                  for key in ("minimum_norm_corrected", "swivel_corrected", "coverage_gap",
                              "swivel_only_corrected_counter_contribution")}
        pooled_positive = Counter()
        for field in replay.HINGE_FIELDS:
            pooled_positive.update(by_field[field]["positive_contributions"])
        pooled_decomposed = sum(pooled_positive.values()) - pooled["swivel_only_corrected_counter_contribution"]
        if pooled_decomposed != pooled["coverage_gap"]:
            raise AssertionError("pooled coverage-gap decomposition failed")
        by_field["pooled"] = {**pooled,
                              "positive_contributions": dict(pooled_positive),
                              "decomposed_gap": pooled_decomposed,
                              "identity_holds": True}
        coverage_gap[method] = by_field

    readable_comparison: dict[str, Any] = {}
    readable_states = {name: h0.copy() for name in state_names}
    for field in replay.HINGE_FIELDS:
        rows = readable_rows_by_field[field]
        fields_states, _ = _run_field_candidate(
            field, rows, h0, valid, requested, observation, oracle_observation,
            contexts, observed_valid)
        for name, state in fields_states.items():
            middle = replay.MIDDLE_INDEX[field]
            readable_states[name][rows, middle] = state[:, middle]
        readable_comparison[field] = _candidate_field_metrics(
            bank, positions, valid, requested, h0, states, reports,
            observation, target_absolute, contexts, field, rows)
    readable_union = np.logical_or.reduce([
        np.isin(np.arange(len(positions)), readable_rows_by_field[field])
        for field in replay.HINGE_FIELDS])
    readable_union_rows = np.flatnonzero(readable_union).astype(np.int64)
    readable_comparison["whole_pose_union"] = {
        name: replay._matched_evaluation(
            bank, positions, readable_states[name], f"attribution+{name}+C_READABLE_WRONG", readable_union)
        for name in state_names
    }

    unknown_counterfactual: dict[str, Any] = {}
    for method, obs, obs_valid in (
            (replay.R_SWIVEL_OBS, observation, observed_valid),
            (replay.R_SWIVEL_ORACLE_2D, oracle_observation, valid)):
        method_result = {}
        for field in replay.HINGE_FIELDS:
            rows = unknown_rows_by_field[field]
            counter = _summarize_counterfactual(
                rows, field, h0, valid, requested, obs, contexts, obs_valid,
                bank, positions, target_absolute, h0)
            method_result[field] = counter
        unknown_counterfactual[method] = method_result
    for field in replay.HINGE_FIELDS:
        obs_feasible = set(unknown_counterfactual[replay.R_SWIVEL_OBS][field]["feasible_rows"])
        oracle_feasible = set(unknown_counterfactual[replay.R_SWIVEL_ORACLE_2D][field]["feasible_rows"])
        rows = set(unknown_rows_by_field[field].tolist())
        unknown_counterfactual.setdefault("obs_oracle_attribution", {})[field] = {
            "both_feasible": len(obs_feasible & oracle_feasible),
            "obs_only_feasible": len(obs_feasible - oracle_feasible),
            "oracle_only_feasible": len(oracle_feasible - obs_feasible),
            "neither_feasible": len(rows - (obs_feasible | oracle_feasible)),
            "partition_total": len(rows),
        }

    minimum_norm_unknown_outcomes = {}
    for field in replay.HINGE_FIELDS:
        rows = unknown_rows_by_field[field]
        _, batch_reports = apply_branch_constraints_batch(
            h0[rows], valid[rows], requested[rows], fields=(field,),
            hinge_write_policy=MINIMUM_NORM)
        outcomes = Counter(report["fields"][field]["outcome"] for report in batch_reports)
        minimum_norm_unknown_outcomes[field] = {
            "corrected": int(outcomes.get(BRANCH_CORRECTED, 0)),
            "unresolved": int(outcomes.get(BRANCH_UNRESOLVED, 0)),
        }

    # Wrong-sign replay, restricted to its exact existing opposite-C rows.
    wrong_requested = opposite
    wrong_h0 = replay._hinge_sign_matrix(h0, valid)
    wrong_rows_by_field = {
        field: _cohort_rows_from_report(original_report, "C", field, wrong_sign=True)
        for field in replay.HINGE_FIELDS
    }
    wrong_partitions: dict[str, dict[str, np.ndarray]] = {}
    wrong_states = {name: h0.copy() for name in state_names}
    wrong_reports: dict[str, dict[str, dict[int, dict[str, Any]]]] = {
        name: {} for name in state_names if name != "H0"
    }
    for field in replay.HINGE_FIELDS:
        rows = wrong_rows_by_field[field]
        column = SIGN_FIELD_NAMES.index(field)
        wrong_partitions[field] = partition_c_rows(rows, wrong_requested[rows, column], wrong_h0[rows, column])
        fields_states, fields_reports = _run_field_candidate(
            field, rows, h0, valid, wrong_requested, observation, oracle_observation,
            contexts, observed_valid)
        for name, state in fields_states.items():
            if name == "H0":
                continue
            middle = replay.MIDDLE_INDEX[field]
            wrong_states[name][rows, middle] = state[:, middle]
            wrong_reports[name][field] = fields_reports[name]

    wrong_sign: dict[str, Any] = {}
    wrong_union_corrected = np.zeros(len(positions), dtype=bool)
    for field in replay.HINGE_FIELDS:
        rows = wrong_rows_by_field[field]
        obs_entries = [wrong_reports[replay.R_SWIVEL_OBS][field][int(row)] for row in rows]
        row_partition = partition_outcomes(rows, [entry["outcome"] for entry in obs_entries])
        wrong_union_corrected[row_partition["corrected"]] = True
        _, min_reports = apply_branch_constraints_batch(
            h0[rows], valid[rows], wrong_requested[rows], fields=(field,), hinge_write_policy=MINIMUM_NORM)
        obs_corrected_rows = row_partition["corrected"]
        per_candidate = _candidate_field_metrics(
            bank, positions, valid, wrong_requested, h0, wrong_states, wrong_reports,
            observation, target_absolute, contexts, field, obs_corrected_rows)
        minimum_partition = partition_outcomes(
            rows, [report["fields"][field]["outcome"] for report in min_reports],
            corrected=BRANCH_CORRECTED, unresolved=BRANCH_UNRESOLVED)
        wrong_sign[field] = {
            "wrong_C_count": int(len(rows)),
            "H0_readable_wrong": int(len(wrong_partitions[field]["readable_wrong"])),
            "H0_unknown": int(len(wrong_partitions[field]["h0_unknown"])),
            "R_SWIVEL_OBS_corrected": int(len(row_partition["corrected"])),
            "R_SWIVEL_OBS_unresolved": int(len(row_partition["unresolved"])),
            "R_SWIVEL_OBS_unresolved_reasons": unresolved_reason_accounting(obs_entries),
            "common_R_SWIVEL_OBS_corrected_field_metrics": per_candidate,
            "MINIMUM_NORM_wrong_C_corrected": int(len(minimum_partition["corrected"])),
            "MINIMUM_NORM_wrong_C_unresolved": int(len(minimum_partition["unresolved"])),
            "MINIMUM_NORM_corrected_R_SWIVEL_unresolved": int(len(np.intersect1d(
                minimum_partition["corrected"], row_partition["unresolved"]))),
            "R_SWIVEL_corrected_MINIMUM_NORM_unresolved": int(len(np.intersect1d(
                row_partition["corrected"], minimum_partition["unresolved"]))),
        }
    wrong_union_rows = np.flatnonzero(wrong_union_corrected).astype(np.int64)
    wrong_sign["common_corrected_whole_pose_union"] = {
        "frames": int(len(wrong_union_rows)),
        "candidates": {
            name: replay._matched_evaluation(
                bank, positions, wrong_states[name], f"wrong-attribution+{name}+OBS-common-corrected",
                wrong_union_corrected)
            for name in state_names
        },
    }
    wrong_union_all_rows = np.logical_or.reduce([
        np.isin(np.arange(len(positions)), wrong_rows_by_field[field])
        for field in replay.HINGE_FIELDS])
    wrong_sign["whole_wrong_C_union_candidates"] = {
        "frames": int(wrong_union_all_rows.sum()),
        "candidates": {
            name: replay._matched_evaluation(
                bank, positions, wrong_states[name], f"wrong-attribution+{name}+wrong-C-union",
                wrong_union_all_rows)
            for name in state_names
        },
    }
    wrong_sign["partition_totals"] = {
        "wrong_C_union_frames": int(np.logical_or.reduce([
            np.isin(np.arange(len(positions)), wrong_rows_by_field[field])
            for field in replay.HINGE_FIELDS]).sum()),
        "R_SWIVEL_OBS_corrected_chain_frames": int(sum(
            value["R_SWIVEL_OBS_corrected"] for key, value in wrong_sign.items()
            if key in replay.HINGE_FIELDS)),
        "R_SWIVEL_OBS_unresolved_chain_frames": int(sum(
            value["R_SWIVEL_OBS_unresolved"] for key, value in wrong_sign.items()
            if key in replay.HINGE_FIELDS)),
    }

    # One-bit SignState and continuous bend direction remain explicitly separate.
    contract = {
        "signstate_contract": "requested hidden forward/depth branch only",
        "minimum_norm_behavior": "installs that branch through a specific reflected perpendicular bend geometry",
        "r_swivel_behavior": "selects the minimum-reprojection point inside the requested readable branch arc",
        "requested_sign_satisfaction_is": "contract-direct evidence",
        "full_3d_hinge_flip_and_bend_mae_are": "continuous-geometry guardrails, not direct SignState satisfaction",
    }

    return {
        "schema": "animcv_pose_reconciliation_causal_attribution_v1",
        "label": "DIAGNOSTIC_ONLY; no new architecture candidate or production behavior",
        "historical_docs44_classification_preserved": "C — SWIVEL ABSTRACTION INSUFFICIENT",
        "identity": {
            "docs44_replay_sha256": replay_sha,
            "bank_content_digest": bank.content_digest(),
            "source_identity": identity,
            "raw_camera_files": len(raw_provenance),
            "camera_frames": int(usable.sum()),
        },
        "C_partition": {"per_field": partitions, "pooled_chain_frames": pooled_partitions},
        "unresolved_reason_accounting": unresolved_reasons,
        "coverage_gap_decomposition": coverage_gap,
        "C_READABLE_WRONG_matched_comparison": {
            "per_field": readable_comparison,
            "union_frames": int(len(readable_union_rows)),
        },
        "C_H0_UNKNOWN": {
            "per_field_counts": {field: int(len(unknown_rows_by_field[field])) for field in replay.HINGE_FIELDS},
            "minimum_norm_outcomes": minimum_norm_unknown_outcomes,
            "current_swivel_outcomes": {
                method: {field: {"corrected": int(sum(
                    reports[method][field][int(row)]["outcome"] == SWIVEL_CORRECTED
                    for row in unknown_rows_by_field[field])),
                    "unresolved": int(sum(
                    reports[method][field][int(row)]["outcome"] == SWIVEL_UNRESOLVED
                    for row in unknown_rows_by_field[field]))}
                    for field in replay.HINGE_FIELDS}
                for method in (replay.R_SWIVEL_OBS, replay.R_SWIVEL_ORACLE_2D)
            },
            "counterfactual": unknown_counterfactual,
        },
        "minimum_norm_reflection_over_specification": {
            field: {
                name: readable_comparison[field][name]["bend_direction_displacement_degrees"]
                for name in (MINIMUM_NORM, replay.R_SWIVEL_OBS, replay.R_SWIVEL_ORACLE_2D)
            }
            for field in replay.HINGE_FIELDS
        },
        "wrong_sign_safety_attribution": wrong_sign,
        "contract_semantics": contract,
        "no_policy_change": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", required=True, type=Path)
    parser.add_argument("--prediction", required=True, type=Path)
    parser.add_argument("--evaluation", required=True, type=Path)
    parser.add_argument("--raw-root", required=True, type=Path)
    parser.add_argument("--replay-report", required=True, type=Path)
    args = parser.parse_args()
    result = run_diagnostic(
        bank_path=args.bank, prediction_path=args.prediction, evaluation_path=args.evaluation,
        raw_root=args.raw_root, replay_report_path=args.replay_report)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
