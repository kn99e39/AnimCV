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
    CORRECTED, DEPTH_ONLY, MINIMUM_NORM, apply_branch_constraints_batch,
)
from framepose.evaluate import evaluate_predictions
from framepose.pose_reconciliation import (
    HINGE_FIELDS, ProjectionContext, reconcile_pose_batch,
)
from framepose.replay_provenance import verify_source_identity
from framepose.signs import HINGE_CHAINS_BY_JOINT, SIGN_FIELD_NAMES, mask_fields, oracle_sign_states, sign_state
from pose.three_dpw_adapter import _SMPL_TO_CANONICAL


SCHEMA = "animcv_frame_pose_reconciliation_replay_v1"
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


def _reconciliation_outcome_counts(reports: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for field in HINGE_FIELDS:
        counts = Counter(report["fields"][field]["outcome"] for report in reports)
        requested = sum(report["fields"][field]["requested_sign"] != 0 for report in reports)
        satisfied = sum(
            report["fields"][field].get("read_back_after") == report["fields"][field]["requested_sign"]
            or report["fields"][field].get("read_back_before") == report["fields"][field]["requested_sign"]
            for report in reports)
        result[field] = {"requested": int(requested), "satisfied": int(satisfied),
                         "outcomes": dict(counts), "unresolved": int(counts.get("unresolved", 0))}
    return result


def _bone_and_endpoint_accounting(state: np.ndarray, baseline: np.ndarray, valid: np.ndarray) -> dict[str, Any]:
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


def _global_accounting(state: np.ndarray, baseline: np.ndarray, valid: np.ndarray) -> dict[str, Any]:
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


def _image_accounting(name: str, state: np.ndarray, baseline: np.ndarray, targets: np.ndarray,
                      observation: np.ndarray, target_absolute: np.ndarray, image_size: np.ndarray,
                      contexts: list[ProjectionContext | None], valid: np.ndarray,
                      reports: list[dict[str, Any]] | None) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    for field in HINGE_FIELDS:
        middle = MIDDLE_INDEX[field]
        rows = np.asarray([
            order for order, context in enumerate(contexts)
            if context is not None and valid[order, middle] and
            (reports is None or reports[order]["fields"][field]["outcome"] == CORRECTED)
        ], dtype=np.int64)
        if not len(rows):
            fields[field] = {"frames": 0}
            continue
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
    return {"candidate": name, "fields": fields}


def _reviews(bank, positions, field: str, states: dict[str, np.ndarray], reports: dict[str, list[dict[str, Any]]],
             wrong_reports: dict[str, list[dict[str, Any]]], observation: np.ndarray,
             target_absolute: np.ndarray, image_size: np.ndarray, contexts: list[ProjectionContext | None],
             valid: np.ndarray) -> list[dict[str, Any]]:
    middle = MIDDLE_INDEX[field]
    chain = HINGE_CHAINS_BY_JOINT[field[: -len("_forward_bend")]]
    chain_indices = [JOINT_INDEX[name] for name in chain]
    qualifying = [order for order in range(len(positions))
                  if reports[R_SWIVEL_OBS][order]["fields"][field]["outcome"] == CORRECTED]
    rows: dict[str, int] = {}
    if qualifying:
        displacement = {
            order: float(np.linalg.norm(
                contexts[order].project_root_relative(states[R_SWIVEL_OBS][order, middle]) -
                contexts[order].project_root_relative(states["H0"][order, middle])))
            for order in qualifying if contexts[order] is not None
        }
        if displacement:
            rows["small_observation_motion"] = min(displacement, key=displacement.get)
            rows["large_swivel_correction"] = max(
                qualifying, key=lambda order: np.linalg.norm(states[R_SWIVEL_OBS][order, middle] - states["H0"][order, middle]))
            rows["foreshortened_chain"] = min(
                qualifying, key=lambda order: reports[R_SWIVEL_OBS][order]["fields"][field].get("in_plane_fraction", np.inf))
            for label, comparator in (
                ("r_swivel_better_than_both", lambda order: all(
                    displacement[order] < np.linalg.norm(
                        contexts[order].project_root_relative(states[name][order, middle]) -
                        contexts[order].project_root_relative(states["H0"][order, middle]))
                    for name in (DEPTH_ONLY, MINIMUM_NORM))),
                ("r_swivel_worse_than_depth_only", lambda order: displacement[order] > np.linalg.norm(
                    contexts[order].project_root_relative(states[DEPTH_ONLY][order, middle]) -
                    contexts[order].project_root_relative(states["H0"][order, middle]))),
                ("r_swivel_worse_than_minimum_norm", lambda order: displacement[order] > np.linalg.norm(
                    contexts[order].project_root_relative(states[MINIMUM_NORM][order, middle]) -
                    contexts[order].project_root_relative(states["H0"][order, middle]))),
            ):
                match = [order for order in qualifying if order in displacement and comparator(order)]
                if match:
                    rows[label] = match[0]
    wrong = [order for order in range(len(positions))
             if wrong_reports[R_SWIVEL_OBS][order]["fields"][field]["outcome"] == CORRECTED]
    if wrong:
        rows["wrong_sign_case"] = max(
            wrong, key=lambda order: np.linalg.norm(
                contexts[order].project_root_relative(states[f"{R_SWIVEL_OBS}__opposite"][order, middle]) -
                contexts[order].project_root_relative(states["H0"][order, middle])))
    unresolved = [order for order in range(len(positions))
                  if wrong_reports[R_SWIVEL_OBS][order]["fields"][field]["outcome"] == "unresolved"]
    if unresolved:
        rows["unresolved_case"] = unresolved[0]

    output = []
    for label, order in rows.items():
        context = contexts[order]
        before = states["H0"][order]
        after_name = R_SWIVEL_OBS if label != "wrong_sign_case" else f"{R_SWIVEL_OBS}__opposite"
        after = states[after_name][order]
        entry = reports[R_SWIVEL_OBS][order]["fields"][field] if label != "wrong_sign_case" \
            else wrong_reports[R_SWIVEL_OBS][order]["fields"][field]
        projected_before = context.project_root_relative(before[middle]) if context else None
        projected_after = context.project_root_relative(after[middle]) if context else None
        target_root_relative = target_absolute[order, middle] - np.asarray(context.root_offset_camera) if context else None
        target_projection = context.project_root_relative(target_root_relative) if context else None
        output.append({
            "category": label, "sample_id": bank.samples[int(positions[order])].sample_id,
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

    evaluations = {
        name: evaluate_predictions(bank, positions, state, candidate=f"{candidate}+{name}")
        for name, state in states.items() if "__opposite" not in name
    }
    summaries = {
        name: {key: evaluation["aggregate"].get(key)
               for key in ("mpjpe_mm", "pa_mpjpe_mm", "hinge_flip_rate",
                           "hinge_direction_mae_degrees", "root_yaw_error_degrees")}
        for name, evaluation in evaluations.items()
    }
    ownership = {
        name: {"endpoint_and_bones": _bone_and_endpoint_accounting(state, h0, valid),
               "global": _global_accounting(state, h0, valid),
               "image": _image_accounting(name, state, h0, targets, observation,
                                           absolute_target, image_size, contexts, valid,
                                           reports.get(name))}
        for name, state in states.items()
        if "__opposite" not in name
    }
    wrong_sign = {
        name: {"evaluation": evaluate_predictions(bank, positions, state, candidate=f"{candidate}+{name}__opposite")["aggregate"],
               "endpoint_and_bones": _bone_and_endpoint_accounting(state, h0, valid),
               "image": _image_accounting(name, state, h0, targets, observation,
                                           absolute_target, image_size, contexts, valid,
                                           wrong_reports.get(name))}
        for name, state in states.items() if name.endswith("__opposite")
    }
    reviews = []
    for field in HINGE_FIELDS:
        reviews.extend(_reviews(bank, positions, field, states, reports, wrong_reports,
                                observation, absolute_target, image_size, contexts, valid))

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
        "sign_ownership": "existing hinge SignState selects only readable hidden forward/depth branch",
        "observation_ownership": "stored input_2d middle joint, or exact projected target for R_SWIVEL_ORACLE_2D",
        "orientation_contract": "wrist/ankle orientation is not represented by canonical XYZ; downstream IK must lock it",
        "candidates": ["H0", DEPTH_ONLY, MINIMUM_NORM, R_SWIVEL_OBS, R_SWIVEL_ORACLE_2D],
        "aggregate_3d": summaries,
        "requested_sign_accounting": {
            name: _reconciliation_outcome_counts(reports[name])
            for name in (R_SWIVEL_OBS, R_SWIVEL_ORACLE_2D)
        },
        "ownership": ownership,
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
