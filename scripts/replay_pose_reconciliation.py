#!/usr/bin/env python3
"""Controlled replay for the research-only Pose Reconciliation layer.

The replay compares the historical H0 / DEPTH_ONLY / MINIMUM_NORM endpoints
with:

    R_SWIVEL_OBS       oracle hidden SignState + stored Geometry Observation
    R_SWIVEL_ORACLE_2D oracle hidden SignState + projected target middle joint

The last candidate is an observation upper bound, never a production input.
No model is run and no historical policy is modified.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from diagnose_hinge_write_policy_observation import load_absolute_sequences, project

from common.canonical_pose import JOINT_INDEX, JOINT_NAMES, bend_direction, root_yaw_error_degrees
from common.serialization import write_json
from framepose.bank import load_bank
from framepose.branch_constraints import (
    ALREADY_SATISFIED, CORRECTED, DEPTH_ONLY, MINIMUM_NORM, UNRESOLVED,
    apply_branch_constraints_batch,
)
from framepose.evaluate import evaluate_predictions
from framepose.pose_reconciliation import (
    HINGE_FIELDS, ProjectionContext, reconcile_pose_batch,
)
from framepose.replay_provenance import verify_source_identity
from framepose.signs import HINGE_CHAINS_BY_JOINT, SIGN_FIELD_NAMES, mask_fields, oracle_sign_states, sign_state
from pose.three_dpw_adapter import _SMPL_TO_CANONICAL


SCHEMA = "animcv_frame_pose_reconciliation_replay_v1"
OBSERVATION_COORDINATE_SPACE = "normalized_full_image"
R_SWIVEL_OBS = "R_SWIVEL_OBS"
R_SWIVEL_ORACLE_2D = "R_SWIVEL_ORACLE_2D"
HINGE_JOINTS = tuple(field[: -len("_forward_bend")] for field in HINGE_FIELDS)
MIDDLE_INDEX = {
    field: JOINT_INDEX[HINGE_CHAINS_BY_JOINT[joint][1]]
    for field, joint in zip(HINGE_FIELDS, HINGE_JOINTS)
}


def _quantiles(values: Any, scale: float = 1.0) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"count": 0}
    values = values * scale
    return {"count": int(len(values)), "mean": float(values.mean()),
            "p50": float(np.percentile(values, 50)),
            "p90": float(np.percentile(values, 90)),
            "p99": float(np.percentile(values, 99)), "max": float(values.max())}


def _stats(a: np.ndarray, b: np.ndarray, scale: np.ndarray | float = 1.0) -> dict[str, Any]:
    return _quantiles(np.linalg.norm(np.asarray(a) - np.asarray(b), axis=-1),
                      scale if np.isscalar(scale) else 1.0)


def _requested_signs(bank, positions, *, opposite: bool = False) -> np.ndarray:
    oracle = oracle_sign_states(bank.arrays["target_3d"], bank.arrays["target_valid"])[positions]
    requested = mask_fields(oracle, list(HINGE_FIELDS)).astype(np.int64)
    if opposite:
        known = requested != 0
        requested[known] = -requested[known]
    return requested


def _load_camera_state(bank, positions, raw_root: Path, raw_split: str):
    sequences, provenance = load_absolute_sequences(raw_root, raw_split)
    smpl_rows = np.asarray([_SMPL_TO_CANONICAL[name] for name in JOINT_NAMES], dtype=np.int64)
    absolute_target = np.full((len(positions), len(JOINT_NAMES), 3), np.nan, dtype=np.float64)
    intrinsics = np.full((len(positions), 3, 3), np.nan, dtype=np.float64)
    image_size = np.full((len(positions), 2), np.nan, dtype=np.float64)
    missing = Counter()
    for order, position in enumerate(positions):
        sample = bank.samples[int(position)]
        entry = sequences.get(sample.sequence_id)
        if entry is None:
            missing["sequence_not_in_raw_split"] += 1
            continue
        frame = int(sample.frame_index)
        if frame >= len(entry["absolute_smpl24"]):
            missing["frame_beyond_raw_sequence"] += 1
            continue
        absolute_target[order] = entry["absolute_smpl24"][frame][smpl_rows]
        intrinsics[order] = entry["intrinsics"]
        image_size[order] = sample.image_size
    usable = ~np.isnan(absolute_target).any(axis=(1, 2))
    root = absolute_target[:, JOINT_INDEX["pelvis"]]
    contexts = [
        (ProjectionContext(intrinsics[index], tuple(image_size[index]), root[index],
                           placement_mode="research_oracle_absolute_root_placement")
         if usable[index] else None)
        for index in range(len(positions))
    ]
    return absolute_target, intrinsics, image_size, root, contexts, provenance, missing, usable


def _oracle_observation(observation: np.ndarray, absolute_target: np.ndarray,
                        contexts: list[ProjectionContext | None], image_size: np.ndarray,
                        valid: np.ndarray) -> np.ndarray:
    result = observation.copy().astype(np.float64)
    for order, context in enumerate(contexts):
        if context is None:
            continue
        for field in HINGE_FIELDS:
            middle = MIDDLE_INDEX[field]
            if not valid[order, middle]:
                continue
            target_root_relative = absolute_target[order, middle] - np.asarray(context.root_offset_camera)
            pixels = context.project_root_relative(target_root_relative)
            result[order, middle, :2] = pixels / image_size[order]
            if result.shape[-1] >= 3:
                result[order, middle, 2] = 1.0
    return result


def _state_pixel_arrays(states: dict[str, np.ndarray], root: np.ndarray,
                        middle: int, rows: np.ndarray, contexts: list[ProjectionContext | None]):
    pixels: dict[str, np.ndarray] = {}
    for name, state in states.items():
        values = []
        for order in rows:
            context = contexts[int(order)]
            if context is None:
                values.append([np.nan, np.nan])
                continue
            values.append(context.project_root_relative(state[int(order), middle]))
        pixels[name] = np.asarray(values, dtype=np.float64)
    return pixels


def _field_column(field: str) -> int:
    return SIGN_FIELD_NAMES.index(field)


def _hinge_sign_matrix(poses: np.ndarray, valid: np.ndarray) -> np.ndarray:
    return np.asarray([
        mask_fields(sign_state(pose, frame_valid)[None, :], list(HINGE_FIELDS))[0]
        for pose, frame_valid in zip(poses, valid)
    ], dtype=np.int64)


def _valid_projection_context(context: ProjectionContext | None) -> bool:
    return context is not None and context.validation_error() is None


def _target_projection_is_available(order: int, field: str, target_absolute: np.ndarray,
                                    contexts: list[ProjectionContext | None]) -> bool:
    context = contexts[order]
    middle = MIDDLE_INDEX[field]
    if not _valid_projection_context(context) or not np.isfinite(target_absolute[order, middle]).all():
        return False
    try:
        context.project_root_relative(
            target_absolute[order, middle] - np.asarray(context.root_offset_camera))
    except (TypeError, ValueError, FloatingPointError):
        return False
    return True


def build_cohorts(h0: np.ndarray, requested: np.ndarray, valid: np.ndarray,
                  observed_valid: np.ndarray, observation: np.ndarray,
                  target_absolute: np.ndarray, contexts: list[ProjectionContext | None],
                  reports_by_candidate: dict[str, list[dict[str, Any]]]) -> dict[str, dict[str, np.ndarray]]:
    """Build the pre-declared A/B/C/D/E masks without candidate-specific row dropping.

    Masks are per field because each hinge has its own chain validity and middle
    joint.  C is the primary operational observation cohort; D is deliberately
    secondary and requires every intervention mechanism to have corrected.
    """
    frame_count = len(h0)
    h0_signs = _hinge_sign_matrix(h0, valid)
    masks: dict[str, dict[str, np.ndarray]] = {
        cohort: {field: np.zeros(frame_count, dtype=bool)
                 for field in HINGE_FIELDS}
        for cohort in ("A", "B", "C", "D", "E")
    }
    for field in HINGE_FIELDS:
        column = _field_column(field)
        joint = field[: -len("_forward_bend")]
        proximal, _, distal = HINGE_CHAINS_BY_JOINT[joint]
        p, m, d = (JOINT_INDEX[name] for name in (proximal, HINGE_CHAINS_BY_JOINT[joint][1], distal))
        chain_valid = valid[:, p] & valid[:, m] & valid[:, d]
        known = requested[:, column] != 0
        conflict = known & (h0_signs[:, column] != requested[:, column]) & chain_valid
        observation_ready = np.asarray([
            conflict[order]
            and _valid_projection_context(contexts[order])
            and bool(observed_valid[order, m])
            and bool(np.isfinite(observation[order, m, :2]).all())
            for order in range(frame_count)
        ], dtype=bool)
        oracle_ready = np.asarray([
            conflict[order] and _target_projection_is_available(
                order, field, target_absolute, contexts)
            for order in range(frame_count)
        ], dtype=bool)
        masks["A"][field] = np.ones(frame_count, dtype=bool)
        masks["B"][field] = conflict
        masks["C"][field] = observation_ready
        masks["E"][field] = oracle_ready
        common_resolved = observation_ready.copy()
        for candidate in (DEPTH_ONLY, MINIMUM_NORM, R_SWIVEL_OBS):
            candidate_reports = reports_by_candidate[candidate]
            common_resolved &= np.asarray([
                candidate_reports[order]["fields"][field]["outcome"] == CORRECTED
                for order in range(frame_count)
            ], dtype=bool)
        masks["D"][field] = common_resolved
    return masks


def summarize_cohorts(cohorts: dict[str, dict[str, np.ndarray]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for cohort, fields in cohorts.items():
        field_summary = {
            field: {"frames": int(mask.sum()),
                    "frame_indices": np.flatnonzero(mask).astype(int).tolist()}
            for field, mask in fields.items()
        }
        summary[cohort] = {
            "fields": field_summary,
            "union_frames": int(np.logical_or.reduce(list(fields.values())).sum()) if fields else 0,
            "union_frame_indices": np.flatnonzero(
                np.logical_or.reduce(list(fields.values()))).astype(int).tolist() if fields else [],
        }
    return summary


def _cohort_rows(cohorts: dict[str, dict[str, np.ndarray]], cohort: str, field: str) -> np.ndarray:
    return np.flatnonzero(cohorts[cohort][field]).astype(np.int64)


def _opposite_base_name(state_name: str) -> str:
    suffix = "__opposite"
    if not state_name.endswith(suffix):
        raise ValueError(f"not an opposite state name: {state_name}")
    return state_name[:-len(suffix)]


def _reports_for_state(state_name: str, reports: dict[str, list[dict[str, Any]]],
                       wrong_reports: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    """Bind an opposite state to the report keyed by its base candidate name."""
    if state_name.endswith("__opposite"):
        return wrong_reports[_opposite_base_name(state_name)]
    return reports[state_name]


def _reconciliation_outcome_counts(reports: list[dict[str, Any]], requested: np.ndarray,
                                   cohorts: dict[str, dict[str, np.ndarray]],
                                   final_state: np.ndarray, valid: np.ndarray) -> dict[str, Any]:
    """Account only known requested signs and enforce the coverage identity."""
    final_signs = _hinge_sign_matrix(final_state, valid)
    result: dict[str, Any] = {}
    for cohort, field_masks in cohorts.items():
        result[cohort] = {}
        for field in HINGE_FIELDS:
            column = _field_column(field)
            population = _cohort_rows(cohorts, cohort, field)
            known_rows = population[requested[population, column] != 0]
            outcomes = Counter(reports[int(order)]["fields"][field]["outcome"]
                               for order in known_rows)
            already = int(outcomes.get(ALREADY_SATISFIED, 0))
            corrected = int(outcomes.get(CORRECTED, 0))
            unresolved = int(outcomes.get(UNRESOLVED, 0))
            requested_count = int(len(known_rows))
            if requested_count != already + corrected + unresolved:
                raise AssertionError(
                    f"requested coverage identity failed for {cohort}/{field}: "
                    f"{requested_count} != {already}+{corrected}+{unresolved}")
            satisfied = int(np.sum(final_signs[known_rows, column] == requested[known_rows, column]))
            result[cohort][field] = {
                "requested_count": requested_count,
                "satisfied_count": satisfied,
                "already_satisfied": already,
                "corrected": corrected,
                "unresolved": unresolved,
                "unknown_excluded": int(len(population) - requested_count),
                "outcomes": dict(outcomes),
                "identity_holds": True,
            }
    return result


def _bone_and_endpoint_accounting(state: np.ndarray, baseline: np.ndarray, valid: np.ndarray,
                                  frame_mask: np.ndarray | None = None) -> dict[str, Any]:
    if frame_mask is not None:
        state, baseline, valid = state[frame_mask], baseline[frame_mask], valid[frame_mask]
    endpoint_names = [name for field in HINGE_FIELDS for name in
                      (HINGE_CHAINS_BY_JOINT[field[: -len("_forward_bend")]][0],
                       HINGE_CHAINS_BY_JOINT[field[: -len("_forward_bend")]][2])]
    endpoint_indices = sorted({JOINT_INDEX[name] for name in endpoint_names})
    endpoint_delta = np.linalg.norm(state[:, endpoint_indices] - baseline[:, endpoint_indices], axis=-1) * 1000.0
    bone_changes = []
    for field in HINGE_FIELDS:
        chain = HINGE_CHAINS_BY_JOINT[field[: -len("_forward_bend")]]
        p, m, d = [JOINT_INDEX[name] for name in chain]
        usable = valid[:, p] & valid[:, m] & valid[:, d]
        for near, far in ((p, m), (m, d)):
            before = np.linalg.norm(baseline[usable, far] - baseline[usable, near], axis=-1)
            after = np.linalg.norm(state[usable, far] - state[usable, near], axis=-1)
            bone_changes.extend(np.abs(after - before) * 1000.0)
    return {"endpoint_max_displacement_mm": _quantiles(endpoint_delta),
            "endpoint_max_displacement_mm_scalar": float(endpoint_delta.max()) if endpoint_delta.size else None,
            "adjacent_bone_length_abs_change_mm": _quantiles(bone_changes)}


def _global_accounting(state: np.ndarray, baseline: np.ndarray, valid: np.ndarray,
                       frame_mask: np.ndarray | None = None) -> dict[str, Any]:
    if frame_mask is not None:
        state, baseline, valid = state[frame_mask], baseline[frame_mask], valid[frame_mask]
    pairs = (("left_shoulder", "right_shoulder"), ("left_hip", "right_hip"))
    bilateral = {}
    for left, right in pairs:
        li, ri = JOINT_INDEX[left], JOINT_INDEX[right]
        delta = np.abs((state[:, ri, 1] - state[:, li, 1]) -
                       (baseline[:, ri, 1] - baseline[:, li, 1])) * 1000.0
        bilateral[f"{left}_{right}_forward_depth_delta_mm"] = _quantiles(delta)
    yaw = []
    for order in range(len(state)):
        if valid[order].sum() < 1:
            continue
        value = root_yaw_error_degrees(state[order], baseline[order], valid[order])
        if value is not None:
            yaw.append(value)
    return {"root_yaw_delta_degrees": _quantiles(yaw), "bilateral": bilateral}


def _image_accounting(name: str, state: np.ndarray, baseline: np.ndarray,
                      observation: np.ndarray, target_absolute: np.ndarray, image_size: np.ndarray,
                      contexts: list[ProjectionContext | None], cohorts: dict[str, dict[str, np.ndarray]],
                      cohort: str) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    for field in HINGE_FIELDS:
        middle = MIDDLE_INDEX[field]
        rows = _cohort_rows(cohorts, cohort, field)
        if not len(rows):
            fields[field] = {
                "frames": 0,
                "target_projection_metric_count": 0,
                "observation_consistency_metric_count": 0,
            }
            continue
        if not all(_valid_projection_context(contexts[int(order)]) for order in rows):
            raise AssertionError(f"{cohort}/{field} contains a frame without a valid projection context")
        current_px = _state_pixel_arrays({"state": state, "baseline": baseline},
                                         np.zeros((len(state), 3)), middle, rows, contexts)
        target_pixels = np.asarray([
            contexts[int(order)].project_root_relative(
                target_absolute[int(order), middle] - np.asarray(contexts[int(order)].root_offset_camera))
            for order in rows
        ])
        observed_pixels = observation[rows, middle, :2] * image_size[rows]
        fields[field] = {
            "frames": int(len(rows)),
            "correction_induced_image_displacement_px": _stats(current_px["state"], current_px["baseline"]),
            "target_projection_error_px": _stats(current_px["state"], target_pixels),
            "observation_consistency_error_px": _stats(current_px["state"], observed_pixels),
            "baseline_target_projection_error_px": _stats(current_px["baseline"], target_pixels),
            "baseline_observation_consistency_error_px": _stats(current_px["baseline"], observed_pixels),
        }
        fields[field]["target_projection_metric_count"] = fields[field]["target_projection_error_px"]["count"]
        fields[field]["observation_consistency_metric_count"] = (
            fields[field]["observation_consistency_error_px"]["count"])
    return {"candidate": name, "cohort": cohort, "fields": fields}


def _field_hinge_accounting(bank, positions: np.ndarray, state: np.ndarray, field: str,
                            rows: np.ndarray, candidate: str) -> dict[str, Any]:
    """Canonical evaluator hinge metrics on exactly one field's chain rows."""
    rows = np.asarray(rows, dtype=np.int64)
    joint = field[: -len("_forward_bend")]
    middle = MIDDLE_INDEX[field]
    if not len(rows):
        return {
            "chain_frame_count": 0,
            "hinge_flip_rate": {"count": 0},
            "bend_direction_error_degrees": {"count": 0},
            "middle_joint_3d_error_mm": {"count": 0},
        }
    evaluation = evaluate_predictions(bank, positions[rows], state[rows], candidate=candidate)
    error_key = f"{joint}_bend_error_degrees"
    flip_key = f"{joint}_bend_flipped"
    bend_errors = [record.get(error_key) for record in evaluation["frames"]]
    flips = [record.get(flip_key) for record in evaluation["frames"]]
    bend_errors = np.asarray([value for value in bend_errors if value is not None], dtype=np.float64)
    flips = np.asarray([value for value in flips if value is not None], dtype=np.float64)
    targets = bank.arrays["target_3d"][positions[rows], middle].astype(np.float64)
    middle_error = np.linalg.norm(state[rows, middle] - targets, axis=-1) * 1000.0
    return {
        "chain_frame_count": int(len(rows)),
        "hinge_flip_rate": {
            "count": int(len(flips)),
            "rate": float(flips.mean()) if len(flips) else None,
        },
        "bend_direction_error_degrees": _quantiles(bend_errors),
        "middle_joint_3d_error_mm": _quantiles(middle_error),
        "canonical_evaluator_fields": {"bend_error": error_key, "flipped": flip_key},
    }


def _field_bone_accounting(state: np.ndarray, baseline: np.ndarray, field: str,
                           rows: np.ndarray) -> dict[str, Any]:
    """Endpoint/bone ownership on exactly one hinge field's chain rows."""
    rows = np.asarray(rows, dtype=np.int64)
    joint = field[: -len("_forward_bend")]
    proximal, middle_name, distal = HINGE_CHAINS_BY_JOINT[joint]
    p, m, d = (JOINT_INDEX[name] for name in (proximal, middle_name, distal))
    if not len(rows):
        empty = {"count": 0}
        return {
            "chain_frame_count": 0,
            "P-M_abs_change_mm": empty,
            "M-D_abs_change_mm": empty,
            "max_abs_change_mm": empty,
        }
    before_pm = np.linalg.norm(baseline[rows, m] - baseline[rows, p], axis=-1)
    after_pm = np.linalg.norm(state[rows, m] - state[rows, p], axis=-1)
    before_md = np.linalg.norm(baseline[rows, d] - baseline[rows, m], axis=-1)
    after_md = np.linalg.norm(state[rows, d] - state[rows, m], axis=-1)
    pm_change = np.abs(after_pm - before_pm) * 1000.0
    md_change = np.abs(after_md - before_md) * 1000.0
    return {
        "chain_frame_count": int(len(rows)),
        "P-M_abs_change_mm": _quantiles(pm_change),
        "M-D_abs_change_mm": _quantiles(md_change),
        "max_abs_change_mm": _quantiles(np.maximum(pm_change, md_change)),
        "chain": [proximal, middle_name, distal],
    }


def _baseline_requested_accounting(requested: np.ndarray, state: np.ndarray,
                                   valid: np.ndarray, field: str, rows: np.ndarray) -> dict[str, Any]:
    rows = np.asarray(rows, dtype=np.int64)
    column = _field_column(field)
    known = rows[requested[rows, column] != 0]
    final_signs = _hinge_sign_matrix(state[known], valid[known])[:, column] if len(known) else np.asarray([])
    return {
        "requested_count": int(len(known)),
        "satisfied_count": int(np.sum(final_signs == requested[known, column])),
        "unknown_excluded": int(len(rows) - len(known)),
        "mechanism_outcomes": "not_applicable_for_H0",
    }


def _matched_evaluation(bank, positions: np.ndarray, state: np.ndarray, candidate: str,
                        frame_mask: np.ndarray) -> dict[str, Any]:
    rows = np.flatnonzero(frame_mask).astype(np.int64)
    if not len(rows):
        return {"frame_count": 0, "aggregate": {"frame_count": 0}, "per_joint_mean_error_mm": {}}
    evaluation = evaluate_predictions(bank, positions[rows], state[rows], candidate=candidate)
    return {
        "frame_count": int(len(rows)),
        "aggregate": evaluation["aggregate"],
        "per_joint_mean_error_mm": evaluation["per_joint_mean_error_mm"],
    }


def _pixel_displacement(state: np.ndarray, baseline: np.ndarray, order: int, middle: int,
                        contexts: list[ProjectionContext | None]) -> float:
    context = contexts[order]
    if not _valid_projection_context(context):
        return float("nan")
    return float(np.linalg.norm(
        context.project_root_relative(state[order, middle]) -
        context.project_root_relative(baseline[order, middle])))


def _reviews(bank, positions, field: str, states: dict[str, np.ndarray], reports: dict[str, list[dict[str, Any]]],
             wrong_reports: dict[str, list[dict[str, Any]]], observation: np.ndarray,
             target_absolute: np.ndarray, contexts: list[ProjectionContext | None],
             valid: np.ndarray, cohorts: dict[str, dict[str, np.ndarray]],
             wrong_cohorts: dict[str, dict[str, np.ndarray]]) -> list[dict[str, Any]]:
    middle = MIDDLE_INDEX[field]
    chain = HINGE_CHAINS_BY_JOINT[field[: -len("_forward_bend")]]
    chain_indices = [JOINT_INDEX[name] for name in chain]
    c_rows = _cohort_rows(cohorts, "C", field)
    e_rows = _cohort_rows(cohorts, "E", field)
    wrong_rows = _cohort_rows(wrong_cohorts, "C", field)
    obs_report = reports[R_SWIVEL_OBS]
    oracle_report = reports[R_SWIVEL_ORACLE_2D]
    wrong_report = _reports_for_state(f"{R_SWIVEL_OBS}__opposite", reports, wrong_reports)

    def outcome(report, order):
        return report[int(order)]["fields"][field]["outcome"]

    rows: dict[str, tuple[int, str, str, str]] = {}
    corrected_obs = [int(order) for order in c_rows if outcome(obs_report, order) == CORRECTED]
    if corrected_obs:
        rows["OBS corrected successfully"] = (corrected_obs[0], R_SWIVEL_OBS, "C", R_SWIVEL_OBS)

    obs_unresolved_oracle_corrected = [
        int(order) for order in e_rows
        if outcome(obs_report, order) == UNRESOLVED and outcome(oracle_report, order) == CORRECTED
    ]
    if obs_unresolved_oracle_corrected:
        rows["OBS unresolved but ORACLE_2D corrected"] = (
            obs_unresolved_oracle_corrected[0], R_SWIVEL_ORACLE_2D, "E", R_SWIVEL_ORACLE_2D)

    both_unresolved = [
        int(order) for order in e_rows
        if outcome(obs_report, order) == UNRESOLVED and outcome(oracle_report, order) == UNRESOLVED
    ]
    if both_unresolved:
        rows["both OBS and ORACLE_2D unresolved"] = (
            both_unresolved[0], R_SWIVEL_OBS, "C", R_SWIVEL_OBS)

    def finite_comparison_rows(predicate):
        matches = []
        for order in corrected_obs:
            values = [_pixel_displacement(states[name], states["H0"], order, middle, contexts)
                      for name in (R_SWIVEL_OBS, DEPTH_ONLY, MINIMUM_NORM)]
            if all(np.isfinite(values)) and predicate(*values):
                matches.append((order, values))
        return matches

    worse = finite_comparison_rows(lambda obs, depth, minimum: obs > depth)
    if worse:
        rows["OBS worse than DEPTH_ONLY on same C row"] = (worse[0][0], R_SWIVEL_OBS, "C", R_SWIVEL_OBS)
    better = finite_comparison_rows(lambda obs, depth, minimum: obs < depth and obs < minimum)
    if better:
        rows["OBS better than both baselines on same C row"] = (better[0][0], R_SWIVEL_OBS, "C", R_SWIVEL_OBS)

    wrong_corrected = [int(order) for order in wrong_rows if outcome(wrong_report, order) == CORRECTED]
    if wrong_corrected:
        wrong_case = max(
            wrong_corrected,
            key=lambda order: _pixel_displacement(
                states[f"{R_SWIVEL_OBS}__opposite"], states["H0"], order, middle, contexts))
        rows["wrong-sign large damage"] = (
            wrong_case, f"{R_SWIVEL_OBS}__opposite", "wrong/C", R_SWIVEL_OBS)

    output = []
    for label, (order, after_name, cohort_name, report_name) in rows.items():
        context = contexts[order]
        before = states["H0"][order]
        after = states[after_name][order]
        report_table = wrong_reports if after_name.endswith("__opposite") else reports
        entry = _reports_for_state(after_name, reports, wrong_reports)[order]["fields"][field]
        projected_before = context.project_root_relative(before[middle]) if context else None
        projected_after = context.project_root_relative(after[middle]) if context else None
        target_root_relative = target_absolute[order, middle] - np.asarray(context.root_offset_camera) if context else None
        target_projection = context.project_root_relative(target_root_relative) if context else None
        output.append({
            "category": label, "cohort": cohort_name, "candidate": report_name,
            "sample_id": bank.samples[int(positions[order])].sample_id,
            "sequence_id": bank.samples[int(positions[order])].sequence_id,
            "frame_index": int(bank.samples[int(positions[order])].frame_index), "chain": list(chain),
            "P_M_D_before_root_relative": before[chain_indices].tolist(),
            "P_M_D_after_root_relative": after[chain_indices].tolist(),
            "bone_lengths_before_m": [float(np.linalg.norm(before[chain_indices[1]] - before[chain_indices[0]])),
                                       float(np.linalg.norm(before[chain_indices[2]] - before[chain_indices[1]]))],
            "bone_lengths_after_m": [float(np.linalg.norm(after[chain_indices[1]] - after[chain_indices[0]])),
                                      float(np.linalg.norm(after[chain_indices[2]] - after[chain_indices[1]]))],
            "requested_sign": entry.get("requested_sign"), "final_sign": entry.get("read_back_after", entry.get("read_back_before")),
            "swivel_theta_radians": entry.get("theta_radians"),
            "observed_middle_2d_normalized": observation[order, middle].tolist(),
            "projected_middle_before_px": None if projected_before is None else projected_before.tolist(),
            "projected_middle_after_px": None if projected_after is None else projected_after.tolist(),
            "target_projected_middle_px": None if target_projection is None else target_projection.tolist(),
            "pixel_delta_before_after": (None if projected_before is None else
                                          (projected_after - projected_before).tolist()),
            "report": entry,
        })
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description="Replay endpoint-fixed Pose Reconciliation")
    parser.add_argument("--bank", required=True, type=Path)
    parser.add_argument("--source", required=True, help="LABEL=CANDIDATE:PREDICTION[:EVALUATION]")
    parser.add_argument("--raw-root", required=True, type=Path)
    parser.add_argument("--raw-split", default="test")
    parser.add_argument("--split", default="test")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    bank = load_bank(args.bank)
    positions = bank.indices(args.split)
    valid = bank.arrays["target_valid"][positions]
    observed_valid = bank.arrays["input_valid"][positions]
    observation = bank.arrays["input_2d"][positions].astype(np.float64)
    targets = bank.arrays["target_3d"][positions].astype(np.float64)
    requested = _requested_signs(bank, positions)
    opposite = _requested_signs(bank, positions, opposite=True)

    label, rest = args.source.split("=", 1)
    parts = rest.split(":")
    candidate, prediction_path = parts[0], Path(parts[1])
    evaluation = Path(parts[2]) if len(parts) > 2 and parts[2] else None
    identity = verify_source_identity(prediction=prediction_path, evaluation=evaluation, bank=bank,
                                      split=args.split, candidate=candidate,
                                      frames=len(positions), joints=len(JOINT_NAMES))
    h0 = np.load(prediction_path).astype(np.float64)
    absolute_target, intrinsics, image_size, root, contexts, raw_provenance, missing, usable = \
        _load_camera_state(bank, positions, args.raw_root, args.raw_split)

    states: dict[str, np.ndarray] = {"H0": h0}
    reports: dict[str, list[dict[str, Any]]] = {}
    wrong_reports: dict[str, list[dict[str, Any]]] = {}
    for name, policy in ((DEPTH_ONLY, DEPTH_ONLY), (MINIMUM_NORM, MINIMUM_NORM)):
        states[name], reports[name] = apply_branch_constraints_batch(
            h0, valid, requested, fields=list(HINGE_FIELDS), hinge_write_policy=policy)
        states[f"{name}__opposite"], wrong_reports[name] = apply_branch_constraints_batch(
            h0, valid, opposite, fields=list(HINGE_FIELDS), hinge_write_policy=policy)

    states[R_SWIVEL_OBS], reports[R_SWIVEL_OBS] = reconcile_pose_batch(
        h0, valid, requested, observation, contexts, fields=HINGE_FIELDS, observed_valid=observed_valid)
    states[f"{R_SWIVEL_OBS}__opposite"], wrong_reports[R_SWIVEL_OBS] = reconcile_pose_batch(
        h0, valid, opposite, observation, contexts, fields=HINGE_FIELDS, observed_valid=observed_valid)
    oracle_observation = _oracle_observation(observation, absolute_target, contexts, image_size, valid)
    states[R_SWIVEL_ORACLE_2D], reports[R_SWIVEL_ORACLE_2D] = reconcile_pose_batch(
        h0, valid, requested, oracle_observation, contexts, fields=HINGE_FIELDS, observed_valid=valid)

    normal_names = ("H0", DEPTH_ONLY, MINIMUM_NORM, R_SWIVEL_OBS, R_SWIVEL_ORACLE_2D)
    intervention_names = (DEPTH_ONLY, MINIMUM_NORM, R_SWIVEL_OBS, R_SWIVEL_ORACLE_2D)
    evaluations = {
        name: evaluate_predictions(bank, positions, states[name], candidate=f"{candidate}+{name}")
        for name in normal_names
    }
    summaries = {
        name: {key: evaluation["aggregate"].get(key)
               for key in ("mpjpe_mm", "pa_mpjpe_mm", "hinge_flip_rate",
                           "hinge_direction_mae_degrees", "root_yaw_error_degrees")}
        for name, evaluation in evaluations.items()
    }
    cohorts = build_cohorts(
        h0, requested, valid, observed_valid, observation, absolute_target, contexts,
        {name: reports[name] for name in (DEPTH_ONLY, MINIMUM_NORM, R_SWIVEL_OBS)},
    )
    wrong_cohorts = build_cohorts(
        h0, opposite, valid, observed_valid, observation, absolute_target, contexts,
        {name: wrong_reports[name] for name in (DEPTH_ONLY, MINIMUM_NORM, R_SWIVEL_OBS)},
    )
    requested_sign_accounting = {
        name: _reconciliation_outcome_counts(reports[name], requested, cohorts, states[name], valid)
        for name in intervention_names
    }
    wrong_requested_sign_accounting = {
        f"{name}__opposite": _reconciliation_outcome_counts(
            wrong_reports[name], opposite, wrong_cohorts, states[f"{name}__opposite"], valid)
        for name in (DEPTH_ONLY, MINIMUM_NORM, R_SWIVEL_OBS)
    }

    def matched_results(cohort_map: dict[str, dict[str, np.ndarray]], cohort_name: str,
                        names: tuple[str, ...], sign_accounting: dict[str, Any]) -> dict[str, Any]:
        frame_mask = np.logical_or.reduce(list(cohort_map[cohort_name].values()))
        results: dict[str, Any] = {}
        for name in names:
            image = _image_accounting(
                name, states[name], h0, observation, absolute_target, image_size,
                contexts, cohort_map, cohort_name)
            field_results = {}
            for field in HINGE_FIELDS:
                rows = _cohort_rows(cohort_map, cohort_name, field)
                field_results[field] = {
                    "chain_frame_count": int(len(rows)),
                    "requested_sign": (
                        _baseline_requested_accounting(requested if cohort_map is cohorts else opposite,
                                                        states[name], valid, field, rows)
                        if name == "H0" else sign_accounting[name][cohort_name][field]),
                    "hinge": _field_hinge_accounting(
                        bank, positions, states[name], field, rows,
                        f"{candidate}+{name}+{cohort_name}+{field}"),
                    "bone": _field_bone_accounting(states[name], h0, field, rows),
                    "image": image["fields"][field],
                }
            results[name] = {
                "matched_3d": _matched_evaluation(
                    bank, positions, states[name], f"{candidate}+{name}+{cohort_name}", frame_mask),
                "endpoint_and_bones": _bone_and_endpoint_accounting(
                    states[name], h0, valid, frame_mask),
                "global": _global_accounting(states[name], h0, valid, frame_mask),
                "image": image,
                "field_results": field_results,
            }
        return {
            "primary": cohort_name == "C",
            "secondary": cohort_name == "D",
            "frame_count": int(frame_mask.sum()),
            "candidates": results,
        }

    matched_cohorts = {
        "C": matched_results(
            cohorts, "C", normal_names, requested_sign_accounting),
        "D": matched_results(
            cohorts, "D", normal_names, requested_sign_accounting),
        "E": matched_results(
            cohorts, "E", normal_names, requested_sign_accounting),
    }

    wrong_sign = {}
    for opposite_name, state in states.items():
        if not opposite_name.endswith("__opposite"):
            continue
        base_name = _opposite_base_name(opposite_name)
        wrong_sign[base_name] = {
            "state_name": opposite_name,
            "primary_cohort": "C",
            "secondary_cohort": "D",
            "cohorts": summarize_cohorts(wrong_cohorts),
            "evaluation": evaluate_predictions(
                bank, positions, state, candidate=f"{candidate}+{opposite_name}") ["aggregate"],
            "requested_sign_accounting": wrong_requested_sign_accounting[opposite_name],
            "matched_cohorts": {
                "C": matched_results(
                    wrong_cohorts, "C", ("H0", opposite_name), wrong_requested_sign_accounting),
                "D": matched_results(
                    wrong_cohorts, "D", ("H0", opposite_name), wrong_requested_sign_accounting),
            },
        }
    reviews = []
    for field in HINGE_FIELDS:
        reviews.extend(_reviews(bank, positions, field, states, reports, wrong_reports,
                                observation, absolute_target, contexts, valid, cohorts, wrong_cohorts))

    report = {
        "schema": SCHEMA,
        "architecture": "Pose Reconciliation / endpoint-fixed two-bone swivel",
        "source_identity": identity,
        "camera_placement": {
            "mode": "research_oracle_absolute_root_placement",
            "production_inference": False,
            "description": "stored root-relative prediction plus target absolute pelvis, identical to docs/39",
        },
        "projection_context": {
            "per_frame_intrinsics": True,
            "canonical_axes": "+X right, +Y forward/depth, +Z up",
            "raw_provenance": raw_provenance,
            "frames_with_camera_geometry": int(usable.sum()),
            "frames_without_camera_geometry": dict(missing),
        },
        "observation_coordinate_space": OBSERVATION_COORDINATE_SPACE,
        "sign_ownership": "existing hinge SignState selects only readable hidden forward/depth branch",
        "observation_ownership": "stored input_2d middle joint, or exact projected target for R_SWIVEL_ORACLE_2D",
        "orientation_contract": "wrist/ankle orientation is not represented by canonical XYZ; downstream IK must lock it",
        "candidates": ["H0", DEPTH_ONLY, MINIMUM_NORM, R_SWIVEL_OBS, R_SWIVEL_ORACLE_2D],
        "aggregate_3d": summaries,
        "all_test": {
            "cohort": "A",
            "candidates": {
                name: {"aggregate": evaluations[name]["aggregate"],
                       "per_joint_mean_error_mm": evaluations[name]["per_joint_mean_error_mm"]}
                for name in normal_names
            },
        },
        "cohorts": summarize_cohorts(cohorts),
        "cohort_contract": {
            "A": "all test frames; normal evaluator",
            "B": "known oracle request, H0 conflict, valid P-M-D chain",
            "C": "B plus valid ProjectionContext and observed middle in normalized_full_image",
            "D": "C plus DEPTH_ONLY, MINIMUM_NORM, and R_SWIVEL_OBS all corrected; secondary diagnostic",
            "E": "B plus camera context and target middle projection; oracle upper-bound cohort",
        },
        "requested_sign_accounting": requested_sign_accounting,
        "matched_cohorts": matched_cohorts,
        "same_C_observation_attribution": {
            "cohort": "C",
            "candidate_pair": [R_SWIVEL_OBS, R_SWIVEL_ORACLE_2D],
            "oracle_label": "R_SWIVEL_ORACLE_2D is an observation upper bound",
            "frame_count": matched_cohorts["C"]["frame_count"],
            "fields": {
                field: {
                    R_SWIVEL_OBS: matched_cohorts["C"]["candidates"][R_SWIVEL_OBS][
                        "field_results"][field],
                    R_SWIVEL_ORACLE_2D: matched_cohorts["C"]["candidates"][R_SWIVEL_ORACLE_2D][
                        "field_results"][field],
                }
                for field in HINGE_FIELDS
            },
        },
        "oracle_upper_bound_E": {
            "cohort": "E",
            "description": "broader target-projection-eligible upper-bound population",
            "matched": matched_cohorts["E"],
        },
        "wrong_sign_endpoint": wrong_sign,
        "review_records": reviews,
        "no_training": True,
        "no_production_promotion": True,
    }
    write_json(args.out / "pose_reconciliation_replay.json", report)
    print(f"\n== Pose Reconciliation replay ({len(positions)} {args.split} frames)")
    for name, summary in summaries.items():
        print("   %-22s MPJPE %s | PA-MPJPE %s | hinge flip %s | hinge MAE %s" % (
            name, summary["mpjpe_mm"], summary["pa_mpjpe_mm"],
            summary["hinge_flip_rate"], summary["hinge_direction_mae_degrees"]))
    print("   review records:", len(reviews))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
