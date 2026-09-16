#!/usr/bin/env python3
"""Export a deterministic, blinded visual review of MINIMUM_NORM vs R_SWIVEL_OBS.

Predictions are reconstructed independently per available test frame. By
default surrounding 3DPW RGB frames are visual context only. An explicit
display-only mode may linearly interpolate between those independent keyframes;
it is never model inference, a metric input, or production smoothing.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import diagnose_pose_reconciliation_attribution as attribution
import replay_pose_reconciliation as replay
from common.canonical_pose import bend_direction
from common.serialization import write_json
from framepose.bank import load_bank
from framepose.branch_constraints import MINIMUM_NORM, apply_branch_constraints_batch
from framepose.pose_reconciliation import reconcile_pose_batch
from framepose.replay_provenance import verify_source_identity
from framepose.signs import (
    HINGE_CHAINS_BY_JOINT,
    SIGN_FIELD_NAMES,
    UNKNOWN,
    oracle_sign_states,
)


METHOD_MINIMUM_NORM = "MINIMUM_NORM"
METHOD_SWIVEL = replay.R_SWIVEL_OBS
METHOD_ORACLE = replay.R_SWIVEL_ORACLE_2D
FPS_FALLBACK_FOR_INVALID = None
PANEL_W, PANEL_H = 480, 270
GRID_COLS, GRID_ROWS = 3, 3
VIDEO_W, VIDEO_H = PANEL_W * GRID_COLS, PANEL_H * GRID_ROWS
COLORS = {
    "p": (255, 128, 0),
    "m": (0, 230, 255),
    "d": (40, 40, 255),
    "observed": (210, 210, 210),
    "h0": (0, 210, 0),
    "candidate": (255, 0, 220),
    "keyframe": (60, 220, 60),
    "interpolated": (0, 170, 255),
    "text": (245, 245, 245),
    "muted": (130, 130, 130),
}
SKELETON_BONES = (
    ("pelvis", "left_hip"), ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
    ("pelvis", "right_hip"), ("right_hip", "right_knee"), ("right_knee", "right_ankle"),
    ("pelvis", "spine"), ("spine", "thorax"), ("thorax", "neck"), ("neck", "head"),
    ("thorax", "left_shoulder"), ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"), ("thorax", "right_shoulder"),
    ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
)
REGIME_ORDER = (
    "both_corrected",
    "minimum_norm_corrected_swivel_readback_failure",
    "large_swivel_observed_image_advantage_p90",
    "large_minimum_norm_3d_bend_advantage_p90",
    "middle_joint_3d_error_favors_swivel",
    "middle_joint_3d_error_favors_minimum_norm",
    "wrong_sign_stress_common_corrected",
    "h0_unknown_counterfactual_swivel_feasible",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _median_representative(rows: list[int], values: dict[int, float],
                           sample_ids: dict[int, str]) -> int | None:
    eligible = [row for row in rows if row in values and np.isfinite(values[row])]
    if not eligible:
        return None
    median = float(np.median([values[row] for row in eligible]))
    return min(eligible, key=lambda row: (abs(values[row] - median), sample_ids[row]))


def _p90_representative(rows: list[int], values: dict[int, float],
                        sample_ids: dict[int, str]) -> tuple[int | None, float | None]:
    positive = [row for row in rows if row in values and np.isfinite(values[row])
                and values[row] > 0.0]
    if not positive:
        return None, None
    threshold = float(np.percentile([values[row] for row in positive], 90))
    tail = [row for row in positive if values[row] >= threshold]
    return _median_representative(tail, values, sample_ids), threshold


def _project(context, points: np.ndarray) -> np.ndarray:
    result = []
    for point in np.asarray(points, dtype=np.float64):
        try:
            result.append(context.project_root_relative(point))
        except (TypeError, ValueError, FloatingPointError):
            result.append([np.nan, np.nan])
    return np.asarray(result, dtype=np.float64)


def _candidate_metrics(state: np.ndarray, h0: np.ndarray, target: np.ndarray,
                       valid: np.ndarray, field: str, requested: int,
                       observed: np.ndarray, image_size: np.ndarray,
                       target_absolute: np.ndarray, context) -> dict[str, Any]:
    joint = field[: -len("_forward_bend")]
    chain = HINGE_CHAINS_BY_JOINT[joint]
    p_idx, m_idx, d_idx = (replay.JOINT_INDEX[name] for name in chain)
    pose = state
    readback = int(replay.sign_state(pose, valid)[SIGN_FIELD_NAMES.index(field)])
    m3d = float(np.linalg.norm(pose[m_idx] - target[m_idx]) * 1000.0)
    pred_bend = bend_direction(pose[m_idx], pose[p_idx], pose[d_idx])
    target_bend = bend_direction(target[m_idx], target[p_idx], target[d_idx])
    bend_error = None
    if pred_bend is not None and target_bend is not None:
        cosine = float(np.clip(pred_bend @ target_bend, -1.0, 1.0))
        bend_error = float(np.degrees(np.arccos(cosine)))
    image_points = _project(context, pose[[p_idx, m_idx, d_idx]])
    target_point = context.project_root_relative(
        target_absolute[m_idx] - np.asarray(context.root_offset_camera))
    observation_point = observed[m_idx, :2] * image_size
    bone_before = (float(np.linalg.norm(h0[m_idx] - h0[p_idx])),
                   float(np.linalg.norm(h0[d_idx] - h0[m_idx])))
    bone_after = (float(np.linalg.norm(pose[m_idx] - pose[p_idx])),
                  float(np.linalg.norm(pose[d_idx] - pose[m_idx])))
    return {
        "canonical_readback": readback,
        "hinge_flip_vs_requested": bool(readback != requested),
        "target_projection_error_px": float(np.linalg.norm(image_points[1] - target_point)),
        "observation_consistency_error_px": float(np.linalg.norm(image_points[1] - observation_point)),
        "middle_joint_3d_error_mm": m3d,
        "bend_direction_error_degrees": bend_error,
        "endpoint_changes_mm": {
            "P": float(np.linalg.norm(pose[p_idx] - h0[p_idx]) * 1000.0),
            "D": float(np.linalg.norm(pose[d_idx] - h0[d_idx]) * 1000.0),
        },
        "bone_length_changes_mm": {
            "P_M": float((bone_after[0] - bone_before[0]) * 1000.0),
            "M_D": float((bone_after[1] - bone_before[1]) * 1000.0),
        },
        "projection_P_M_D_px": image_points.tolist(),
        "middle_joint_3d_m": pose[m_idx].tolist(),
    }


def _safe_outcome(report: dict[str, Any]) -> dict[str, Any]:
    return {key: report.get(key) for key in
            ("outcome", "reason", "read_back_before", "read_back_after",
             "read_back_after_attempt", "candidate_source", "theta_radians")}


def _field_compute(field: str, bank, positions: np.ndarray, h0: np.ndarray,
                   valid: np.ndarray, observed_valid: np.ndarray,
                   observation: np.ndarray, requested: np.ndarray,
                   contexts: list[Any], *, oracle_observation: np.ndarray | None = None
                   ) -> dict[str, Any]:
    minimum_state, minimum_reports = apply_branch_constraints_batch(
        h0, valid, requested, fields=(field,), hinge_write_policy=MINIMUM_NORM)
    swivel_state, swivel_reports = reconcile_pose_batch(
        h0, valid, requested, observation, contexts, fields=(field,),
        observed_valid=observed_valid)
    result = {
        METHOD_MINIMUM_NORM: minimum_state,
        METHOD_SWIVEL: swivel_state,
        "reports_minimum_norm": [entry["fields"][field] for entry in minimum_reports],
        "reports_swivel": [entry["fields"][field] for entry in swivel_reports],
    }
    if oracle_observation is not None:
        oracle_state, oracle_reports = reconcile_pose_batch(
            h0, valid, requested, oracle_observation, contexts, fields=(field,),
            observed_valid=valid)
        result[METHOD_ORACLE] = oracle_state
        result["reports_oracle"] = [entry["fields"][field] for entry in oracle_reports]
    return result


def _new_review_id(field: str, sample_id: str, mode: str) -> str:
    payload = f"{field}|{sample_id}|{mode}".encode("utf-8")
    return "R" + hashlib.sha256(payload).hexdigest()[:12]


def _blind_mapping(review_id: str, seed: int = 20260915) -> dict[str, str]:
    bit = hashlib.sha256(f"{seed}|{review_id}".encode()).digest()[0] & 1
    return ({"A": METHOD_MINIMUM_NORM, "B": METHOD_SWIVEL} if bit == 0
            else {"A": METHOD_SWIVEL, "B": METHOD_MINIMUM_NORM})


def _select_review_events(bank, positions: np.ndarray, requested: np.ndarray,
                          h0_signs: np.ndarray, report: dict[str, Any],
                          fields_data: dict[str, dict[str, Any]],
                          counterfactuals: dict[str, dict[int, dict[str, Any]]]
                          ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    sample_ids = {row: bank.samples[int(positions[row])].sample_id
                  for row in range(len(positions))}
    events: dict[tuple[str, str, str], dict[str, Any]] = {}
    diagnostic_events: list[dict[str, Any]] = []

    def add_standard(field: str, row: int, regime: str, request_mode: str,
                     request_sign: int, row_outcomes: dict[str, dict[str, Any]],
                     values: dict[str, Any] | None = None) -> None:
        sample = bank.samples[int(positions[row])]
        key = (field, sample.sample_id, request_mode)
        event = events.setdefault(key, {
            "review_id": _new_review_id(field, sample.sample_id, request_mode),
            "review_class": "blind_primary_ab",
            "selection_population": ("C_READABLE_WRONG" if request_mode == "oracle_correct"
                                     else "wrong-sign C common-corrected"),
            "selection_regimes": [],
            "sample_id": sample.sample_id,
            "sequence_id": sample.sequence_id,
            "center_frame": int(sample.frame_index),
            "source_fps": float(sample.fps or 30.0),
            "field": field,
            "chain": list(HINGE_CHAINS_BY_JOINT[field[: -len("_forward_bend")]]),
            "request_mode": request_mode,
            "requested_sign": int(request_sign),
            "h0_readback": int(h0_signs[row, SIGN_FIELD_NAMES.index(field)]),
            "method_outcomes": {name: _safe_outcome(value)
                                for name, value in row_outcomes.items()},
            "selection_values": {},
        })
        if regime not in event["selection_regimes"]:
            event["selection_regimes"].append(regime)
        if values:
            event["selection_values"][regime] = values

    for field, data in fields_data.items():
        column = SIGN_FIELD_NAMES.index(field)
        c_rows = attribution._cohort_rows_from_report(report, "C", field)
        known_conflict = c_rows[requested[c_rows, column] != UNKNOWN]
        readable = known_conflict[
            (h0_signs[known_conflict, column] != UNKNOWN)
            & (h0_signs[known_conflict, column] != requested[known_conflict, column])]
        outcomes = {
            METHOD_MINIMUM_NORM: data["reports_minimum_norm"],
            METHOD_SWIVEL: data["reports_swivel"],
        }
        row_metrics: dict[int, dict[str, dict[str, Any]]] = {}
        image_advantage: dict[int, float] = {}
        bend_advantage: dict[int, float] = {}
        swivel_m3d_advantage: dict[int, float] = {}
        minimum_m3d_advantage: dict[int, float] = {}
        for row in readable.tolist():
            ctx = data["contexts"][row]
            target = data["target"][row]
            row_metrics[row] = {}
            for method, states in ((METHOD_MINIMUM_NORM, data[METHOD_MINIMUM_NORM]),
                                   (METHOD_SWIVEL, data[METHOD_SWIVEL])):
                row_metrics[row][method] = _candidate_metrics(
                    states[row], data["h0"][row], target, data["valid"][row], field,
                    int(requested[row, column]), data["observation"][row],
                    data["image_size"][row], data["target_absolute"][row], ctx)
            mn = row_metrics[row][METHOD_MINIMUM_NORM]
            sw = row_metrics[row][METHOD_SWIVEL]
            image_advantage[row] = (mn["observation_consistency_error_px"]
                                    - sw["observation_consistency_error_px"])
            if mn["bend_direction_error_degrees"] is not None and \
                    sw["bend_direction_error_degrees"] is not None:
                bend_advantage[row] = (sw["bend_direction_error_degrees"]
                                       - mn["bend_direction_error_degrees"])
            swivel_m3d_advantage[row] = (mn["middle_joint_3d_error_mm"]
                                         - sw["middle_joint_3d_error_mm"])
            minimum_m3d_advantage[row] = -swivel_m3d_advantage[row]

        def metric_selector(regime: str, values: dict[int, float], *, p90: bool = False):
            if p90:
                row, threshold = _p90_representative(readable.tolist(), values, sample_ids)
                return row, threshold
            row = _median_representative(
                [idx for idx in readable.tolist() if values.get(idx, 0.0) > 0.0],
                values, sample_ids)
            return row, (float(np.median([values[idx] for idx in readable.tolist()
                                         if idx in values and values[idx] > 0.0]))
                         if row is not None else None)

        both = [int(row) for row in readable
                if outcomes[METHOD_MINIMUM_NORM][int(row)]["outcome"] == "corrected"
                and outcomes[METHOD_SWIVEL][int(row)]["outcome"] == "corrected"]
        row = _median_representative(both, image_advantage, sample_ids)
        if row is not None:
            add_standard(field, row, "both_corrected", "oracle_correct", int(requested[row, column]),
                         {METHOD_MINIMUM_NORM: outcomes[METHOD_MINIMUM_NORM][row],
                          METHOD_SWIVEL: outcomes[METHOD_SWIVEL][row]},
                         {"metric": "median observed-image error advantage within both-corrected rows",
                          "value": image_advantage[row], "population_n": len(both)})

        failures = [int(row) for row in readable
                    if outcomes[METHOD_MINIMUM_NORM][int(row)]["outcome"] == "corrected"
                    and outcomes[METHOD_SWIVEL][int(row)].get("reason") ==
                    "candidate failed exact endpoint, length, or SignState read-back"]
        row = (sorted(failures, key=lambda idx: sample_ids[idx])[len(failures) // 2]
               if failures else None)
        if row is not None:
            add_standard(field, row, "minimum_norm_corrected_swivel_readback_failure",
                         "oracle_correct", int(requested[row, column]),
                         {METHOD_MINIMUM_NORM: outcomes[METHOD_MINIMUM_NORM][row],
                          METHOD_SWIVEL: outcomes[METHOD_SWIVEL][row]},
                         {"rule": "lexicographically median sample_id among exact read-back failures",
                          "population_n": len(failures)})

        for regime, values in (
                ("large_swivel_observed_image_advantage_p90", image_advantage),
                ("large_minimum_norm_3d_bend_advantage_p90", bend_advantage)):
            row, threshold = metric_selector(regime, values, p90=True)
            if row is not None:
                add_standard(field, row, regime, "oracle_correct", int(requested[row, column]),
                             {METHOD_MINIMUM_NORM: outcomes[METHOD_MINIMUM_NORM][row],
                              METHOD_SWIVEL: outcomes[METHOD_SWIVEL][row]},
                             {"metric": ("MINIMUM_NORM observed-image error minus Swivel error"
                                         if regime.startswith("large_swivel") else
                                         "Swivel bend-direction error minus MINIMUM_NORM error"),
                              "positive_value": values[row], "p90_positive_threshold": threshold,
                              "population_n": sum(value > 0.0 for value in values.values())})

        for regime, values in (
                ("middle_joint_3d_error_favors_swivel", swivel_m3d_advantage),
                ("middle_joint_3d_error_favors_minimum_norm", minimum_m3d_advantage)):
            row, median_value = metric_selector(regime, values, p90=False)
            if row is not None:
                add_standard(field, row, regime, "oracle_correct", int(requested[row, column]),
                             {METHOD_MINIMUM_NORM: outcomes[METHOD_MINIMUM_NORM][row],
                              METHOD_SWIVEL: outcomes[METHOD_SWIVEL][row]},
                             {"metric": "paired middle-joint 3D error difference in mm; positive favors named method",
                              "median_positive_value": median_value,
                              "selected_value": values[row],
                              "population_n": sum(value > 0.0 for value in values.values())})

        # Counterfactually feasible H0-UNKNOWN rows are deliberately excluded
        # from the blind primary A/B methods. They receive a separate, labelled
        # diagnostic export below.
        unknown = attribution._cohort_rows_from_report(report, "C", field)
        unknown = unknown[(requested[unknown, column] != UNKNOWN)
                          & (h0_signs[unknown, column] == UNKNOWN)]
        feasible = [(int(idx), result) for idx, result in counterfactuals[field].items()
                    if int(idx) in set(unknown.tolist())
                    and result.get("status") == "feasible_requested_sign_solution"]
        if feasible:
            scores = [float(item.get("minimum_reprojection_error_px", math.inf))
                      for _, item in feasible]
            median_score = float(np.median(scores))
            row, counter = min(feasible, key=lambda item: (
                abs(float(item[1].get("minimum_reprojection_error_px", math.inf)) - median_score),
                sample_ids[item[0]]))
            diagnostic_events.append({
                "review_id": _new_review_id(field, sample_ids[row], "h0_unknown_counterfactual"),
                "review_class": "labelled_counterfactual_diagnostic",
                "selection_regimes": ["h0_unknown_counterfactually_feasible_swivel"],
                "selection_population": "C_H0_UNKNOWN with exact existing OBS solver feasible after bypassing only prestate refusal",
                "sample_id": sample_ids[row],
                "sequence_id": bank.samples[int(positions[row])].sequence_id,
                "center_frame": int(bank.samples[int(positions[row])].frame_index),
                "source_fps": float(bank.samples[int(positions[row])].fps or 30.0),
                "field": field,
                "chain": list(HINGE_CHAINS_BY_JOINT[field[: -len("_forward_bend")]]),
                "requested_sign": int(requested[row, column]),
                "h0_readback": UNKNOWN,
                "counterfactual_label": "COUNTERFACTUAL_SWIVEL_WITHOUT_PRESTATE_REFUSAL",
                "counterfactual_reprojection_error_px": float(
                    counter["minimum_reprojection_error_px"]),
                "method_outcomes": {
                    METHOD_MINIMUM_NORM: _safe_outcome(data["reports_minimum_norm"][row]),
                    METHOD_SWIVEL: _safe_outcome(data["reports_swivel"][row]),
                },
                "selection_rule": "median reprojection error among feasible rows for this field; sample_id tie-break",
                "counterfactual_pose": np.asarray(counter["state"], dtype=np.float64).tolist(),
            })

        # Wrong-sign stress: invert the ground-truth request, then retain only
        # exact C rows on which both operators accepted that deliberately wrong
        # request. This reproduces docs/45's safety control cohort.
        wrong_rows = attribution._cohort_rows_from_report(
            report, "C", field, wrong_sign=True)
        wrong_request = requested.copy()
        known = wrong_request[:, column] != UNKNOWN
        wrong_request[known, column] *= -1
        mn_wrong, mn_wrong_reports = apply_branch_constraints_batch(
            data["h0"], data["valid"], wrong_request, fields=(field,),
            hinge_write_policy=MINIMUM_NORM)
        sw_wrong, sw_wrong_reports = reconcile_pose_batch(
            data["h0"], data["valid"], wrong_request, data["observation"],
            data["contexts"], fields=(field,), observed_valid=data["observed_valid"])
        common = [int(idx) for idx in wrong_rows
                  if mn_wrong_reports[int(idx)]["fields"][field]["outcome"] == "corrected"
                  and sw_wrong_reports[int(idx)]["fields"][field]["outcome"] == "corrected"]
        if common:
            row = sorted(common, key=lambda idx: sample_ids[idx])[len(common) // 2]
            add_standard(field, row, "wrong_sign_stress_common_corrected",
                         "deliberately_wrong", int(wrong_request[row, column]),
                         {METHOD_MINIMUM_NORM: mn_wrong_reports[row]["fields"][field],
                          METHOD_SWIVEL: sw_wrong_reports[row]["fields"][field]},
                         {"rule": "lexicographically median sample_id in wrong-sign C common-corrected population",
                          "population_n": len(common),
                          "deliberately_wrong_sign": int(wrong_request[row, column])})
            # Preserve the wrong-request candidate arrays for the window render.
            data["wrong_states"] = {METHOD_MINIMUM_NORM: mn_wrong, METHOD_SWIVEL: sw_wrong}
            data["wrong_reports"] = {
                METHOD_MINIMUM_NORM: [entry["fields"][field] for entry in mn_wrong_reports],
                METHOD_SWIVEL: [entry["fields"][field] for entry in sw_wrong_reports],
            }

    standard = sorted(events.values(), key=lambda item: (
        item["field"], item["center_frame"], item["request_mode"], item["review_id"]))
    return standard, diagnostic_events


def _fit_rgb(image: np.ndarray) -> tuple[np.ndarray, float, float, float]:
    height, width = image.shape[:2]
    scale = min((PANEL_W - 8) / width, (PANEL_H - 42) / height)
    new_w, new_h = int(round(width * scale)), int(round(height * scale))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((PANEL_H, PANEL_W, 3), dtype=np.uint8)
    left, top = (PANEL_W - new_w) // 2, 34 + (PANEL_H - 34 - new_h) // 2
    canvas[top:top + new_h, left:left + new_w] = resized
    return canvas, scale, float(left), float(top)


def _pix(value: np.ndarray, scale: float, left: float, top: float) -> tuple[int, int] | None:
    point = np.asarray(value, dtype=np.float64)
    if point.shape != (2,) or not np.isfinite(point).all():
        return None
    return int(round(point[0] * scale + left)), int(round(point[1] * scale + top))


def _draw_point(canvas: np.ndarray, point: tuple[int, int] | None,
                color: tuple[int, int, int], label: str | None = None,
                radius: int = 5) -> None:
    if point is None:
        return
    cv2.circle(canvas, point, radius, color, -1, cv2.LINE_AA)
    if label:
        cv2.putText(canvas, label, (point[0] + 6, point[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)


def _draw_chain_rgb(canvas: np.ndarray, pixels: np.ndarray, transform: tuple[float, float, float],
                    *, observed_middle: np.ndarray | None = None,
                    chain_color: tuple[int, int, int] | None = None) -> None:
    scale, left, top = transform
    chain_points = [_pix(point, scale, left, top) for point in pixels]
    if observed_middle is not None:
        observation_point = _pix(observed_middle, scale, left, top)
        _draw_point(canvas, observation_point, COLORS["observed"], "obs", 4)
    for first, second in ((0, 1), (1, 2)):
        if chain_points[first] is not None and chain_points[second] is not None:
            cv2.line(canvas, chain_points[first], chain_points[second],
                     chain_color or COLORS["candidate"], 3, cv2.LINE_AA)
    for point, key, label in zip(chain_points, ("p", "m", "d"), ("P", "M", "D")):
        _draw_point(canvas, point, COLORS[key], label)


def _view_xy(points: np.ndarray, view: str) -> np.ndarray:
    x, y, z = np.asarray(points, dtype=np.float64).T
    if view == "camera_aligned":
        return np.column_stack((x, -z))
    azimuth, elevation = np.deg2rad(35.0), np.deg2rad(22.0)
    horizontal = np.cos(azimuth) * x - np.sin(azimuth) * y
    vertical = (-np.sin(elevation) * (np.sin(azimuth) * x + np.cos(azimuth) * y)
                + np.cos(elevation) * z)
    return np.column_stack((horizontal, vertical))


def _bounds_for_view(poses: list[np.ndarray], chain_indices: tuple[int, int, int],
                     view: str, valid_masks: list[np.ndarray] | None = None
                     ) -> tuple[float, float, float, float]:
    projected = []
    for index, pose in enumerate(poses):
        pose = np.asarray(pose, dtype=np.float64)
        mask = (np.ones(len(replay.JOINT_NAMES), dtype=bool) if valid_masks is None
                else np.asarray(valid_masks[index], dtype=bool).copy())
        mask &= np.isfinite(pose).all(axis=1)
        projected.extend(_view_xy(pose[mask], view))
    values = np.asarray(projected, dtype=np.float64)
    mins, maxs = values.min(axis=0), values.max(axis=0)
    center = (mins + maxs) / 2.0
    span = max(float((maxs - mins).max()), 0.05) * 1.6
    return (float(center[0] - span / 2), float(center[0] + span / 2),
            float(center[1] - span / 2), float(center[1] + span / 2))


def _draw_3d(canvas: np.ndarray, pose: np.ndarray | None,
             chain_indices: tuple[int, int, int], view: str,
             bounds: tuple[float, float, float, float], label: str,
             valid_mask: np.ndarray | None = None,
             chain_color: tuple[int, int, int] | None = None) -> None:
    if pose is None:
        cv2.putText(canvas, "No independent prediction at this timestamp",
                    (16, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLORS["muted"], 1, cv2.LINE_AA)
        return
    pose = np.asarray(pose, dtype=np.float64)
    projected_all = _view_xy(pose, view)
    mask = (np.ones(len(replay.JOINT_NAMES), dtype=bool) if valid_mask is None
            else np.asarray(valid_mask, dtype=bool).copy())
    mask &= np.isfinite(pose).all(axis=1)
    x0, x1, y0, y1 = bounds
    margin_left, margin_right, margin_top, margin_bottom = 44, 12, 48, 26
    sx = (PANEL_W - margin_left - margin_right) / max(x1 - x0, 1e-12)
    sy = (PANEL_H - margin_top - margin_bottom) / max(y1 - y0, 1e-12)
    scale = min(sx, sy)
    draw_w, draw_h = (x1 - x0) * scale, (y1 - y0) * scale
    base_x = int((PANEL_W - draw_w) / 2)
    base_y = int(margin_top + (PANEL_H - margin_top - margin_bottom - draw_h) / 2)
    def pixel(point: np.ndarray) -> tuple[int, int]:
        return (int(base_x + (point[0] - x0) * scale),
                int(base_y + (y1 - point[1]) * scale))

    for first, second in SKELETON_BONES:
        first_idx, second_idx = replay.JOINT_INDEX[first], replay.JOINT_INDEX[second]
        if mask[first_idx] and mask[second_idx]:
            cv2.line(canvas, pixel(projected_all[first_idx]), pixel(projected_all[second_idx]),
                     (112, 112, 112), 2, cv2.LINE_AA)
    for index in np.flatnonzero(mask):
        cv2.circle(canvas, pixel(projected_all[index]), 3, (160, 160, 160),
                   -1, cv2.LINE_AA)

    projected = projected_all[list(chain_indices)]
    pts = [pixel(point) for point in projected]
    color = chain_color or COLORS["candidate"]
    cv2.line(canvas, pts[0], pts[1], color, 4, cv2.LINE_AA)
    cv2.line(canvas, pts[1], pts[2], color, 4, cv2.LINE_AA)
    for point, key, marker in zip(pts, ("p", "m", "d"), ("P", "M", "D")):
        _draw_point(canvas, point, COLORS[key], marker, 7)
    if view == "camera_aligned":
        cv2.putText(canvas, "+X right / +Z up; depth is projected away",
                    (12, PANEL_H - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.36, COLORS["muted"], 1, cv2.LINE_AA)
    else:
        cv2.putText(canvas, "fixed oblique view: azimuth 35 deg, elevation 22 deg",
                    (12, PANEL_H - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.36, COLORS["muted"], 1, cv2.LINE_AA)


def _panel(canvas: np.ndarray, index: int, title: str) -> np.ndarray:
    col, row = index % GRID_COLS, index // GRID_COLS
    x, y = col * PANEL_W, row * PANEL_H
    tile = canvas[y:y + PANEL_H, x:x + PANEL_W]
    cv2.rectangle(tile, (0, 0), (PANEL_W - 1, PANEL_H - 1), (95, 95, 95), 1)
    cv2.rectangle(tile, (0, 0), (PANEL_W - 1, 30), (24, 24, 24), -1)
    cv2.putText(tile, title[:58], (10, 21), cv2.FONT_HERSHEY_SIMPLEX,
                0.51, COLORS["text"], 1, cv2.LINE_AA)
    return tile


def _draw_video_frame(image: np.ndarray | None, sample: Any | None,
                      data_by_field: dict[str, Any], row: int | None,
                      event: dict[str, Any], method_map: dict[str, str],
                      status: str, bounds: dict[str, tuple[float, float, float, float]],
                      center_override: np.ndarray | None = None,
                      *, blind: bool = False, counterfactual: bool = False,
                      pose_overrides: dict[str, np.ndarray] | None = None,
                      projected_overrides: dict[str, np.ndarray] | None = None,
                      visual_status: str = "keyframe") -> np.ndarray:
    canvas = np.zeros((VIDEO_H, VIDEO_W, 3), dtype=np.uint8)
    source_rgb = np.zeros((PANEL_H, PANEL_W, 3), dtype=np.uint8)
    if image is not None:
        source_rgb = _fit_rgb(image)[0]
    title = _panel(canvas, 0, "Original RGB - source sequence")
    title[34:, :] = source_rgb[34:, :]
    observation_title = ("Observed 2D joints on original RGB" if visual_status == "keyframe"
                         else "No independent 2D observation at this frame")
    title = _panel(canvas, 1, observation_title)
    if image is not None and sample is not None and row is not None and visual_status == "keyframe":
        fitted, scale, left, top = _fit_rgb(image)
        title[:] = fitted
        # Restore the panel heading after the RGB copy.
        cv2.rectangle(title, (0, 0), (PANEL_W - 1, 30), (24, 24, 24), -1)
        cv2.putText(title, observation_title, (10, 21),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.49, COLORS["text"], 1, cv2.LINE_AA)
        observed = data_by_field[event["field"]]["observation"][row]
        size = data_by_field[event["field"]]["image_size"][row]
        for joint_index, xy in enumerate(observed[:, :2] * size):
            if not np.isfinite(xy).all():
                continue
            point = _pix(xy, scale, left, top)
            if point is not None:
                is_active = joint_index in data_by_field[event["field"]]["chain_indices"]
                _draw_point(title, point, COLORS["m"] if is_active else COLORS["observed"],
                            replay.JOINT_NAMES[joint_index][:2] if is_active else None, 4 if is_active else 2)
    elif visual_status == "interpolated":
        cv2.putText(title, "Interpolation display only: no measured 2D joints here.",
                    (14, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLORS["muted"], 1, cv2.LINE_AA)
    else:
        cv2.putText(title, "No independently predicted FrameBank row",
                    (14, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLORS["muted"], 1, cv2.LINE_AA)

    field_data = data_by_field[event["field"]]
    chain_indices = field_data["chain_indices"]
    poses = {METHOD_MINIMUM_NORM: None, METHOD_SWIVEL: None, METHOD_ORACLE: None}
    if row is not None:
        state_key = "wrong_states" if event.get("request_mode") == "deliberately_wrong" else None
        states = field_data[state_key] if state_key else field_data
        poses[METHOD_MINIMUM_NORM] = states[METHOD_MINIMUM_NORM][row]
        poses[METHOD_SWIVEL] = states[METHOD_SWIVEL][row]
        if METHOD_ORACLE in states:
            poses[METHOD_ORACLE] = states[METHOD_ORACLE][row]
    if pose_overrides is not None:
        poses.update({method: pose_overrides[method] for method in poses
                      if method in pose_overrides})
    if counterfactual:
        poses[METHOD_SWIVEL] = center_override
    elif center_override is not None:
        poses[METHOD_SWIVEL] = center_override

    # H0 projection panel uses the row's own projection context only.
    tile = _panel(canvas, 2, "H0 projection on original RGB")
    if image is not None and row is not None:
        fitted, scale, left, top = _fit_rgb(image)
        tile[:] = fitted
        cv2.rectangle(tile, (0, 0), (PANEL_W - 1, 30), (24, 24, 24), -1)
        cv2.putText(tile, "H0 projection on original RGB", (10, 21),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.51, COLORS["text"], 1, cv2.LINE_AA)
        h0 = (pose_overrides["h0"] if pose_overrides is not None
              else field_data["h0"][row])
        pixels = (projected_overrides["h0"] if projected_overrides is not None
                  else _project(field_data["contexts"][row], h0[list(chain_indices)]))
        observed_middle = (field_data["observation"][row, chain_indices[1], :2]
                           * field_data["image_size"][row] if visual_status == "keyframe" else None)
        _draw_chain_rgb(tile, pixels, (scale, left, top), observed_middle=observed_middle,
                        chain_color=COLORS["keyframe"] if visual_status == "keyframe"
                        else COLORS["interpolated"])
    else:
        cv2.putText(tile, "No independently predicted FrameBank row",
                    (14, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLORS["muted"], 1, cv2.LINE_AA)

    for idx, method in ((3, method_map["A"]), (4, method_map["B"])):
        if counterfactual and method == METHOD_SWIVEL:
            projection_label = "Counterfactual Swivel (centre only)"
        else:
            projection_label = ("Candidate A" if blind and idx == 3 else
                                "Candidate B" if blind else method)
        tile = _panel(canvas, idx, f"{projection_label} projection on RGB")
        if image is not None and row is not None:
            fitted, scale, left, top = _fit_rgb(image)
            tile[:] = fitted
            cv2.rectangle(tile, (0, 0), (PANEL_W - 1, 30), (24, 24, 24), -1)
            cv2.putText(tile, f"{projection_label} projection on RGB"[:58], (10, 21),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, COLORS["text"], 1, cv2.LINE_AA)
            pose = poses[method]
            if pose is None:
                cv2.putText(tile, "Counterfactual candidate is shown at the selected centre only",
                            (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.43,
                            COLORS["muted"], 1, cv2.LINE_AA)
            else:
                pixels = (projected_overrides[method] if projected_overrides is not None
                          and method in projected_overrides else
                          _project(field_data["contexts"][row], pose[list(chain_indices)]))
                observed_middle = (field_data["observation"][row, chain_indices[1], :2]
                                   * field_data["image_size"][row]
                                   if visual_status == "keyframe" else None)
                _draw_chain_rgb(tile, pixels, (scale, left, top), observed_middle=observed_middle,
                                chain_color=COLORS["keyframe"] if visual_status == "keyframe"
                                else COLORS["interpolated"])
        else:
            cv2.putText(tile, "No independently predicted FrameBank row",
                        (14, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLORS["muted"], 1, cv2.LINE_AA)

    for idx, method, view in (
            (5, method_map["A"], "camera_aligned"),
            (6, method_map["B"], "camera_aligned"),
            (7, method_map["A"], "fixed_oblique"),
            (8, method_map["B"], "fixed_oblique")):
        if blind:
            candidate_label = "Candidate A" if idx in (5, 7) else "Candidate B"
        else:
            candidate_label = method
            if counterfactual and method == METHOD_SWIVEL:
                candidate_label = "Counterfactual Swivel (centre only)"
        view_label = "camera-aligned" if view == "camera_aligned" else "fixed oblique"
        tile = _panel(canvas, idx, f"{candidate_label} 3D - {view_label}")
        _draw_3d(tile, poses[method], chain_indices, view, bounds[view], candidate_label,
                 (pose_overrides["valid"] if pose_overrides is not None else
                  field_data["valid"][row] if row is not None else None),
                 chain_color=COLORS["keyframe"] if visual_status == "keyframe"
                 else COLORS["interpolated"])

    if visual_status != "keyframe":
        cv2.rectangle(canvas, (8, 34), (400, 59), (16, 74, 115), -1)
        cv2.putText(canvas, "INTERPOLATED VISUALIZATION - not an independent prediction",
                    (14, 51), cv2.FONT_HERSHEY_SIMPLEX, 0.39, COLORS["text"], 1, cv2.LINE_AA)
    else:
        cv2.rectangle(canvas, (8, 34), (338, 59), (32, 110, 50), -1)
        cv2.putText(canvas, "KEYFRAME - independent FrameBank prediction",
                    (14, 51), cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLORS["text"], 1, cv2.LINE_AA)

    # Persistent annotation prevents the contextual clip from being mistaken
    # for a temporal inference or smoothed animation.
    cv2.rectangle(canvas, (0, VIDEO_H - 48), (VIDEO_W, VIDEO_H), (18, 18, 18), -1)
    if counterfactual:
        cv2.putText(canvas,
                    "H0-UNKNOWN COUNTERFACTUAL ONLY; current policy refusal is unchanged.",
                    (12, VIDEO_H - 27), cv2.FONT_HERSHEY_SIMPLEX,
                    0.43, COLORS["text"], 1, cv2.LINE_AA)
        footer_y = VIDEO_H - 8
    else:
        footer_y = VIDEO_H - 11
    footer = ("Framewise independent keyframes; intervening poses are visualization-only linear interpolation."
              if visual_status != "sparse_context" else
              "Framewise independent predictions; video context is visualization only. "
              "No temporal input or pose smoothing; unscored timestamps have no pose overlay.")
    cv2.putText(canvas, footer, (12, footer_y), cv2.FONT_HERSHEY_SIMPLEX,
                0.40, COLORS["text"], 1, cv2.LINE_AA)
    return canvas


def _interpolate_keyframe_states(data: dict[str, Any], event: dict[str, Any],
                                 before: int, after: int, weight: float) -> dict[str, np.ndarray]:
    """Linear display-only interpolation between two independent FrameBank rows."""
    if not 0.0 <= weight <= 1.0:
        raise ValueError("interpolation weight must be in [0, 1]")
    states = data.get("wrong_states") if event.get("request_mode") == "deliberately_wrong" else data
    def blend(first: np.ndarray, second: np.ndarray) -> np.ndarray:
        return ((1.0 - weight) * np.asarray(first, dtype=np.float64)
                + weight * np.asarray(second, dtype=np.float64))
    result = {"h0": blend(data["h0"][before], data["h0"][after]),
              "valid": np.asarray(data["valid"][before], dtype=bool)
              & np.asarray(data["valid"][after], dtype=bool)}
    for method in (METHOD_MINIMUM_NORM, METHOD_SWIVEL):
        result[method] = blend(states[method][before], states[method][after])
    if METHOD_ORACLE in states:
        result[METHOD_ORACLE] = blend(states[METHOD_ORACLE][before], states[METHOD_ORACLE][after])
    return result


def _interpolate_keyframe_projections(data: dict[str, Any], event: dict[str, Any],
                                      before: int, after: int, weight: float
                                      ) -> dict[str, np.ndarray]:
    """Blend endpoint image positions, avoiding a claim of camera pose interpolation."""
    if not 0.0 <= weight <= 1.0:
        raise ValueError("interpolation weight must be in [0, 1]")
    states = data.get("wrong_states") if event.get("request_mode") == "deliberately_wrong" else data
    chain_indices = data["chain_indices"]
    def blend(first: np.ndarray, second: np.ndarray) -> np.ndarray:
        return ((1.0 - weight) * np.asarray(first, dtype=np.float64)
                + weight * np.asarray(second, dtype=np.float64))
    result = {
        "h0": blend(_project(data["contexts"][before], data["h0"][before, list(chain_indices)]),
                    _project(data["contexts"][after], data["h0"][after, list(chain_indices)])),
    }
    for method in (METHOD_MINIMUM_NORM, METHOD_SWIVEL):
        result[method] = blend(
            _project(data["contexts"][before], states[method][before, list(chain_indices)]),
            _project(data["contexts"][after], states[method][after, list(chain_indices)]))
    if METHOD_ORACLE in states:
        result[METHOD_ORACLE] = blend(
            _project(data["contexts"][before], states[METHOD_ORACLE][before, list(chain_indices)]),
            _project(data["contexts"][after], states[METHOD_ORACLE][after, list(chain_indices)]))
    return result


def _metrics_for_event(event: dict[str, Any], bank, positions: np.ndarray,
                       fields_data: dict[str, dict[str, Any]], requested: np.ndarray,
                       blind_mapping: dict[str, str]) -> dict[str, Any]:
    data = fields_data[event["field"]]
    row = bank.position(event["sample_id"])  # global bank position, not test-row order
    order_by_position = {int(position): order for order, position in enumerate(positions)}
    order = order_by_position[row]
    method_data = {}
    for method in (METHOD_MINIMUM_NORM, METHOD_SWIVEL):
        method_state = data[method][order]
        method_data[method] = _candidate_metrics(
            method_state, data["h0"][order], data["target"][order], data["valid"][order],
            event["field"], event["requested_sign"], data["observation"][order],
            data["image_size"][order], data["target_absolute"][order], data["contexts"][order])
    blind_metrics = {
        candidate: method_data[method]
        for candidate, method in blind_mapping.items()
    }
    return {
        "candidate_metrics_by_blind_label": blind_metrics,
        "labelled_method_metrics": method_data,
    }


def _selection_and_context_data(args) -> tuple[dict[str, Any], list[dict[str, Any]],
                                               list[dict[str, Any]], dict[str, Any]]:
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite review package: {args.out}")
    report_path = args.replay_report
    report_sha = attribution._sha256(report_path)
    if report_sha != attribution.EXPECTED_REPLAY_SHA256:
        raise ValueError(f"docs/44 replay SHA mismatch: {report_sha}")
    report = attribution._read_json(report_path)
    bank = load_bank(args.bank)
    positions = bank.indices("test")
    if len(positions) != 7076 or bank.content_digest() != attribution.EXPECTED_BANK_DIGEST:
        raise ValueError("the exact docs/44 FrameBank identity was not recovered")
    identity = verify_source_identity(
        prediction=args.prediction, evaluation=args.evaluation, bank=bank, split="test",
        candidate="O_BILATERAL_oracle_forward_depth_only", frames=len(positions),
        joints=len(replay.JOINT_NAMES))
    if identity["prediction"]["sha256"] != attribution.EXPECTED_PREDICTION_SHA256:
        raise ValueError("prediction differs from docs/44")
    if identity["evaluation"]["sha256"] != attribution.EXPECTED_EVALUATION_SHA256:
        raise ValueError("evaluation differs from docs/44")
    h0 = np.load(args.prediction).astype(np.float64)
    valid = bank.arrays["target_valid"][positions]
    observed_valid = bank.arrays["input_valid"][positions]
    observation = bank.arrays["input_2d"][positions].astype(np.float64)
    target = bank.arrays["target_3d"][positions].astype(np.float64)
    oracle = oracle_sign_states(bank.arrays["target_3d"], bank.arrays["target_valid"])[positions]
    requested = replay.mask_fields(oracle, list(replay.HINGE_FIELDS)).astype(np.int64)
    h0_signs = replay._hinge_sign_matrix(h0, valid)
    target_absolute, _, image_size, _, contexts, camera_provenance, missing, usable = \
        replay._load_camera_state(bank, positions, args.raw_root, "test")
    if int(usable.sum()) != 7076 or missing or len(camera_provenance) != 24:
        raise ValueError("raw 3DPW camera coverage does not match docs/44")
    expected_raw = report["projection_context"]["raw_provenance"]
    def digest_map(records):
        return {Path(item["path"]).stem: (int(item["bytes"]), item["sha256"])
                for item in records}
    if digest_map(camera_provenance) != digest_map(expected_raw):
        raise ValueError("raw 3DPW camera provenance differs from docs/44")
    oracle_observation = replay._oracle_observation(
        observation, target_absolute, contexts, image_size, valid)

    fields_data: dict[str, Any] = {}
    counterfactuals: dict[str, dict[int, dict[str, Any]]] = {}
    for field in replay.HINGE_FIELDS:
        computed = _field_compute(field, bank, positions, h0, valid, observed_valid,
                                  observation, requested, contexts,
                                  oracle_observation=oracle_observation)
        computed.update({
            "contexts": contexts,
            "observation": observation,
            "image_size": image_size,
            "target_absolute": target_absolute,
            "target": target,
            "valid": valid,
            "observed_valid": observed_valid,
            "h0": h0,
            "chain_indices": tuple(replay.JOINT_INDEX[name] for name in
                                    HINGE_CHAINS_BY_JOINT[field[: -len("_forward_bend")]]),
        })
        fields_data[field] = computed
        if args.no_counterfactual_diagnostics:
            counterfactuals[field] = {}
        else:
            rows = attribution._cohort_rows_from_report(report, "C", field)
            column = SIGN_FIELD_NAMES.index(field)
            unknown_rows = rows[(requested[rows, column] != UNKNOWN)
                                & (h0_signs[rows, column] == UNKNOWN)]
            unknown_results = {}
            middle = replay.MIDDLE_INDEX[field]
            for row in unknown_rows:
                unknown_results[int(row)] = attribution.counterfactual_swivel_without_prestate_refusal(
                    h0[int(row)], valid[int(row)], int(requested[int(row), column]),
                    observation[int(row), middle], contexts[int(row)], field,
                    observed_valid=bool(observed_valid[int(row), middle]))
            counterfactuals[field] = unknown_results

    standard, diagnostics = _select_review_events(
        bank, positions, requested, h0_signs, report, fields_data, counterfactuals)
    by_regime: dict[str, int] = {regime: 0 for regime in REGIME_ORDER}
    for event in standard:
        for regime in event["selection_regimes"]:
            by_regime[regime] = by_regime.get(regime, 0) + 1
    if not args.no_counterfactual_diagnostics:
        by_regime["h0_unknown_counterfactual_swivel_feasible"] = len(diagnostics)
    required_regimes = (REGIME_ORDER if not args.no_counterfactual_diagnostics else
                        tuple(regime for regime in REGIME_ORDER
                              if regime != "h0_unknown_counterfactual_swivel_feasible"))
    missing_regimes = [regime for regime in required_regimes if by_regime.get(regime, 0) == 0]
    if missing_regimes:
        raise AssertionError(f"selection did not find required review regimes: {missing_regimes}")
    if not args.no_counterfactual_diagnostics:
        expected_unknown = {"left_elbow_forward_bend": 207,
                            "right_elbow_forward_bend": 173,
                            "left_knee_forward_bend": 142,
                            "right_knee_forward_bend": 246}
        for field, expected in expected_unknown.items():
            actual = sum(result.get("status") == "feasible_requested_sign_solution"
                         for result in counterfactuals[field].values())
            if actual != expected:
                raise AssertionError(f"docs/45 OBS counterfactual feasibility changed for {field}: {actual}")
    return {
        "bank": bank,
        "positions": positions,
        "h0": h0,
        "valid": valid,
        "observed_valid": observed_valid,
        "observation": observation,
        "target": target,
        "target_absolute": target_absolute,
        "image_size": image_size,
        "requested": requested,
        "h0_signs": h0_signs,
        "contexts": contexts,
        "fields_data": fields_data,
        "counterfactuals": counterfactuals,
        "standard_events": standard,
        "diagnostic_events": diagnostics,
        "report": report,
        "report_sha": report_sha,
        "identity": identity,
        "camera_provenance": camera_provenance,
    }, standard, diagnostics, by_regime


def _view_bounds_for_event(event: dict[str, Any], data: dict[str, Any],
                           row_lookup: dict[tuple[str, int], int],
                           frame_range: tuple[int, int],
                           override: np.ndarray | None = None
                           ) -> dict[str, tuple[float, float, float, float]]:
    rows = [row for (seq, frame), row in row_lookup.items()
            if seq == event["sequence_id"] and frame_range[0] <= frame <= frame_range[1]]
    if not rows:
        rows = [row_lookup[(event["sequence_id"], event["center_frame"])]]
    states = data.get("wrong_states") if event.get("request_mode") == "deliberately_wrong" else data
    method_states = [states[METHOD_MINIMUM_NORM][row] for row in rows]
    method_states.extend(states[METHOD_SWIVEL][row] for row in rows)
    valid_masks = [data["valid"][row] for row in rows]
    valid_masks.extend(data["valid"][row] for row in rows)
    if METHOD_ORACLE in data:
        method_states.extend(data[METHOD_ORACLE][row] for row in rows)
        valid_masks.extend(data["valid"][row] for row in rows)
    if override is not None:
        method_states.append(override)
        center_order = row_lookup[(event["sequence_id"], event["center_frame"])]
        valid_masks.append(data["valid"][center_order])
    chain_indices = data["chain_indices"]
    return {view: _bounds_for_view(method_states, chain_indices, view, valid_masks)
            for view in ("camera_aligned", "fixed_oblique")}


def _render_clip(event: dict[str, Any], destination: Path, bank,
                 positions: np.ndarray, fields_data: dict[str, Any], image_root: Path,
                 method_map: dict[str, str], window_seconds: float,
                 *, blind: bool, counterfactual: np.ndarray | None = None,
                 still_path: Path | None = None,
                 interpolate_keyframes: bool = False) -> dict[str, Any]:
    sample = bank.samples[bank.position(event["sample_id"])]
    fps = float(sample.fps or 30.0)
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError(f"invalid source FPS for {sample.sample_id}")
    image_reference = sample.image_reference
    if image_reference is None:
        raise ValueError(f"selected sample has no RGB image reference: {sample.sample_id}")
    image_dir = image_reference.resolve({"3dpw_images": image_root}).parent
    if not image_dir.is_dir():
        raise FileNotFoundError(image_dir)
    available = sorted(int(match.group(1)) for path in image_dir.glob("image_*.jpg")
                       if (match := re.fullmatch(r"image_(\d+)\.jpg", path.name)))
    if not available:
        raise FileNotFoundError(f"no frame images found in {image_dir}")
    half = int(round(window_seconds * fps / 2.0))
    first_frame = max(min(available), int(event["center_frame"]) - half)
    last_frame = min(max(available), int(event["center_frame"]) + half)
    row_lookup = {(bank.samples[int(position)].sequence_id,
                   int(bank.samples[int(position)].frame_index)): order
                  for order, position in enumerate(positions)}
    sequence_keyframes = sorted((frame, row) for (sequence, frame), row in row_lookup.items()
                                if sequence == event["sequence_id"])
    keyframe_numbers = [frame for frame, _ in sequence_keyframes]
    data = fields_data[event["field"]]
    bounds = _view_bounds_for_event(
        event, data, row_lookup, (first_frame, last_frame), counterfactual)

    destination.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(destination), fourcc, fps, (VIDEO_W, VIDEO_H))
    if not writer.isOpened():
        raise RuntimeError("OpenCV could not open an MP4V writer")
    central_frame = None
    for frame in range(first_frame, last_frame + 1):
        path = image_dir / f"image_{frame:05d}.jpg"
        image = cv2.imread(str(path), cv2.IMREAD_COLOR) if path.is_file() else None
        exact_row = row_lookup.get((event["sequence_id"], frame))
        row, pose_overrides, projected_overrides, visual_status = exact_row, None, None, "keyframe"
        if interpolate_keyframes and exact_row is None:
            insertion = bisect.bisect_left(keyframe_numbers, frame)
            if 0 < insertion < len(sequence_keyframes):
                before_frame, before_row = sequence_keyframes[insertion - 1]
                after_frame, after_row = sequence_keyframes[insertion]
                weight = (frame - before_frame) / (after_frame - before_frame)
                row = before_row if weight <= 0.5 else after_row
                pose_overrides = _interpolate_keyframe_states(
                    data, event, before_row, after_row, weight)
                projected_overrides = _interpolate_keyframe_projections(
                    data, event, before_row, after_row, weight)
                visual_status = "interpolated"
            else:
                raise ValueError("requested review window extends beyond available FrameBank keyframes")
        sample_at_frame = bank.samples[int(positions[row])] if row is not None else None
        override = counterfactual if frame == event["center_frame"] else None
        frame_canvas = _draw_video_frame(
            image, sample_at_frame, fields_data, row, event, method_map,
            "sampled" if row is not None else "no_sample", bounds, override,
            blind=blind, counterfactual=counterfactual is not None,
            pose_overrides=pose_overrides, projected_overrides=projected_overrides,
            visual_status=(
                visual_status if interpolate_keyframes else "sparse_context"))
        if frame == event["center_frame"]:
            central_frame = frame_canvas.copy()
        writer.write(frame_canvas)
    writer.release()
    if central_frame is not None and still_path is not None:
        still_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(still_path), central_frame):
            raise RuntimeError(f"could not write central still {still_path}")
    return {
        "path": str(destination),
        "sha256": _sha256(destination),
        "frame_count": last_frame - first_frame + 1,
        "fps": fps,
        "start_frame": first_frame,
        "end_frame": last_frame,
        "center_frame": int(event["center_frame"]),
        "display_mode": ("keyframe_linear_interpolation_visualization"
                         if interpolate_keyframes else "sparse_independent_context"),
    }


def _oracle_still(event: dict[str, Any], destination: Path, bank,
                  fields_data: dict[str, Any], image_root: Path) -> dict[str, Any]:
    sample = bank.samples[bank.position(event["sample_id"])]
    data = fields_data[event["field"]]
    positions = bank.indices("test")
    row = int(np.where(positions == bank.position(event["sample_id"]))[0][0])
    image_path = sample.image_reference.resolve({"3dpw_images": image_root})
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(image_path)
    lookup = {(sample.sequence_id, sample.frame_index): row}
    bounds = _view_bounds_for_event(event, data, lookup,
                                    (event["center_frame"], event["center_frame"]))
    candidate_map = {"A": METHOD_SWIVEL, "B": METHOD_ORACLE}
    canvas = _draw_video_frame(image, sample, fields_data, row, event, candidate_map,
                               "sampled", bounds, blind=False)
    cv2.rectangle(canvas, (0, 0), (VIDEO_W, 33), (24, 24, 24), -1)
    cv2.putText(canvas,
                "SEPARATE UPPER-BOUND DIAGNOSTIC: Oracle target projection is unavailable to the real system",
                (12, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.58, COLORS["text"], 1, cv2.LINE_AA)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(destination), canvas):
        raise RuntimeError(f"could not write oracle diagnostic {destination}")
    return {"path": str(destination), "sha256": _sha256(destination)}


def _write_review_sheet(path: Path, events: list[dict[str, Any]]) -> None:
    columns = [
        "review_id",
        "visible_branch_correctness (A/B/both/neither/unclear)",
        "reference_image_consistency (A/B/tie/unclear)",
        "3D_articulation_plausibility (A/B/tie/unclear)",
        "animation_sequence_plausibility (A/B/tie/unclear)",
        "visible_instability_pathology (A/B/both/neither/unclear)",
        "optional_note",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for event in events:
            writer.writerow({"review_id": event["review_id"]})


def export_review(args) -> dict[str, Any]:
    package, events, diagnostics, regime_counts = _selection_and_context_data(args)
    out = args.out
    # Selection and all frame identities are persisted before any render is
    # viewed or inspected, closing the cherry-picking path by construction.
    out.mkdir(parents=True, exist_ok=False)
    for folder in ("blind", "labelled", "technical_stills", "oracle_diagnostic",
                   "counterfactual_diagnostic"):
        (out / folder).mkdir()

    blind_seed = 20260915
    blind_key: dict[str, Any] = {
        "schema": "animcv_pose_reconciliation_blind_key_v1",
        "assignment_seed": blind_seed,
        "assignment_rule": "SHA-256(seed|review_id) low bit selects the fixed A/B permutation",
        "methods": {},
    }
    manifest_events = []
    technical_manifest_events = []
    selected_oracle_events = []
    for event in events:
        mapping = _blind_mapping(event["review_id"], blind_seed)
        blind_key["methods"][event["review_id"]] = mapping
        enriched = {
            key: event[key] for key in (
                "review_id", "review_class", "selection_population", "selection_regimes",
                "sample_id", "sequence_id", "center_frame", "source_fps", "field", "chain",
                "request_mode", "requested_sign", "h0_readback")
        }
        data = package["fields_data"][event["field"]]
        global_position = package["bank"].position(event["sample_id"])
        order = int(np.where(package["positions"] == global_position)[0][0])
        states = data.get("wrong_states") if event["request_mode"] == "deliberately_wrong" else data
        metric_values = {}
        for method in (METHOD_MINIMUM_NORM, METHOD_SWIVEL):
            metric_values[method] = _candidate_metrics(
                states[method][order], data["h0"][order], data["target"][order],
                data["valid"][order], event["field"], event["requested_sign"],
                data["observation"][order], data["image_size"][order],
                data["target_absolute"][order], data["contexts"][order])
        blinded_metrics = {candidate: metric_values[method]
                           for candidate, method in mapping.items()}
        enriched["image_space_metrics"] = {
            candidate: {"target_projection_error_px": metrics["target_projection_error_px"],
                     "observation_consistency_error_px": metrics["observation_consistency_error_px"]}
            for candidate, metrics in blinded_metrics.items()}
        enriched["middle_joint_3d_error_mm"] = {
            candidate: metrics["middle_joint_3d_error_mm"]
            for candidate, metrics in blinded_metrics.items()}
        enriched["bend_direction_error_degrees"] = {
            candidate: metrics["bend_direction_error_degrees"]
            for candidate, metrics in blinded_metrics.items()}
        enriched["bone_changes_mm"] = {
            candidate: metrics["bone_length_changes_mm"]
            for candidate, metrics in blinded_metrics.items()}
        enriched["endpoint_changes_mm"] = {
            candidate: metrics["endpoint_changes_mm"]
            for candidate, metrics in blinded_metrics.items()}
        neutral_regimes = {
            "both_corrected": "both_candidates_corrected",
            "minimum_norm_corrected_swivel_readback_failure": "one_candidate_final_readback_failure",
            "large_swivel_observed_image_advantage_p90": "observed_image_error_advantage_p90",
            "large_minimum_norm_3d_bend_advantage_p90": "3d_bend_error_advantage_p90",
            "middle_joint_3d_error_favors_swivel": "middle_joint_error_advantage_candidate_1",
            "middle_joint_3d_error_favors_minimum_norm": "middle_joint_error_advantage_candidate_2",
            "wrong_sign_stress_common_corrected": "wrong_sign_stress_both_corrected",
        }
        enriched["selection_regimes"] = [neutral_regimes[regime]
                                         for regime in event["selection_regimes"]]
        enriched["A_B_mapping"] = (
            "stored separately in blind_key.json and labelled/technical_manifest.json; "
            "not included in this blind event record")
        enriched["blind_video"] = f"blind/{event['review_id']}.mp4"
        enriched["labelled_video"] = f"labelled/{event['review_id']}_labelled.mp4"
        manifest_events.append(enriched)
        technical_manifest_events.append({
            "review_id": event["review_id"],
            "sample_id": event["sample_id"],
            "sequence_id": event["sequence_id"],
            "center_frame": event["center_frame"],
            "field": event["field"],
            "chain": event["chain"],
            "requested_sign": event["requested_sign"],
            "h0_readback": event["h0_readback"],
            "request_mode": event["request_mode"],
            "A_B_mapping": mapping,
            METHOD_MINIMUM_NORM: {
                "outcome": _safe_outcome(event["method_outcomes"][METHOD_MINIMUM_NORM]),
                "metrics": metric_values[METHOD_MINIMUM_NORM],
            },
            METHOD_SWIVEL: {
                "outcome": _safe_outcome(event["method_outcomes"][METHOD_SWIVEL]),
                "metrics": metric_values[METHOD_SWIVEL],
            },
        })
        if "both_corrected" in event["selection_regimes"]:
            selected_oracle_events.append(event)

    manifest = {
        "schema": "animcv_pose_reconciliation_visual_review_manifest_v1",
        "purpose": "human qualitative review only; no automatic architecture or policy selection",
        "primary_comparison": "two frozen candidates masked as Candidate A / Candidate B; event mapping kept out of this manifest",
        "requested_sign_source": "ground-truth canonical SignState as a shared evaluation input; no VLM output",
        "primary_input_boundary": "real detector observations are shown as reference; no oracle target projection or neighboring-frame input enters the primary candidates",
        "baseline": "H0 is the common projection/reference panel",
        "temporal_scope": ("independent FrameBank keyframes with display-only linear interpolation "
                           "between them; interpolation is not a model prediction, temporal input, "
                           "or production smoothing" if args.interpolate_keyframes else
                           "framewise independent predictions; video context is visualization only; "
                           "no temporal input/interpolation/smoothing"),
        "source": {
            "bank_content_digest": package["bank"].content_digest(),
            "docs44_replay_sha256": package["report_sha"],
            "prediction_sha256": package["identity"]["prediction"]["sha256"],
            "evaluation_sha256": package["identity"]["evaluation"]["sha256"],
            "raw_camera_files": len(package["camera_provenance"]),
            "image_root": str(args.image_root),
        },
        "selection_rule": {
            "pre_render_lock": "this manifest is written before any output media is rendered",
            "both_corrected": "deterministic median representative within each field's exact C_READABLE_WRONG rows accepted by both candidates",
            "readback_failure": "lexicographically median sample_id per field among rows where one candidate fails final canonical read-back",
            "large_advantage": "per-field P90 representative in predeclared positive paired image/bend error-difference tails; median within the P90 tail, sample_id tie-break",
            "middle_joint_3d_advantage": "per-field deterministic median among positive paired middle-joint error differences on the same C_READABLE_WRONG rows",
            "wrong_sign_stress": "lexicographically median sample_id among opposite-oracle-request C rows accepted by both candidates",
            "h0_unknown_counterfactual": "per-field median reprojection-error row among exact docs/45 feasible counterfactuals; separately labelled, not a primary blind comparison",
            "no_rendered_output_used_for_selection": True,
        },
        "regime_event_counts": regime_counts,
        "primary_blind_and_labelled_events": manifest_events,
        "counterfactual_diagnostic_events": diagnostics,
        "per_event_assignment_key": (
            "blind_key.json; labelled/technical_manifest.json is labelled and should be consulted only after blind judgments"),
        "clip_window_seconds": args.window_seconds,
        "video_fps": "each sample's stored source FPS (3DPW test is 30 fps)",
        "panel_geometry": {
            "panel_grid": "3x3; RGB, observed 2D, H0, candidate A/B projections, candidate A/B camera-aligned 3D, candidate A/B fixed oblique 3D",
            "3d_bounds": "per clip, computed once from both candidates and shared across A/B and all clip frames; never independently auto-fit",
            "camera_aligned": "+X right, +Z up; camera depth +Y projected away",
            "fixed_oblique": "azimuth 35 degrees, elevation 22 degrees",
            "active_chain": "P-M-D labels and endpoint markers P/D; middle M highlighted",
        },
    }
    for event in diagnostics:
        event["diagnostic_video"] = (
            f"counterfactual_diagnostic/{event['review_id']}_prestate_bypass.mp4")
        event["diagnostic_still"] = (
            f"counterfactual_diagnostic/{event['review_id']}_prestate_bypass.png")
    write_json(out / "review_manifest.json", manifest)
    write_json(out / "blind_key.json", blind_key)
    _write_review_sheet(out / "review_sheet.csv", events)
    write_json(out / "labelled" / "technical_manifest.json", {
        "schema": "animcv_pose_reconciliation_labelled_technical_manifest_v1",
        "purpose": "method names, full outcomes, metrics, and A/B mapping; consult only after blind judgments are recorded",
        "events": technical_manifest_events,
    })

    for index, event in enumerate(events, start=1):
        print(f"render {index}/{len(events)} {event['review_id']} {event['field']} {event['sample_id']}",
              flush=True)
        mapping = blind_key["methods"][event["review_id"]]
        blind_path = out / "blind" / f"{event['review_id']}.mp4"
        _render_clip(event, blind_path, package["bank"], package["positions"],
                     package["fields_data"], args.image_root, mapping,
                     args.window_seconds, blind=True,
                     interpolate_keyframes=args.interpolate_keyframes)
        labelled_mapping = {"A": METHOD_MINIMUM_NORM, "B": METHOD_SWIVEL}
        labelled_path = out / "labelled" / f"{event['review_id']}_labelled.mp4"
        still_path = out / "technical_stills" / f"{event['review_id']}.png"
        _render_clip(event, labelled_path, package["bank"], package["positions"],
                     package["fields_data"], args.image_root, labelled_mapping,
                     args.window_seconds, blind=False, still_path=still_path,
                     interpolate_keyframes=args.interpolate_keyframes)

    # Four isolated upper-bound stills: the oracle 2D endpoint is never blended
    # into the blind comparison.
    for field in replay.HINGE_FIELDS:
        candidate = next((event for event in selected_oracle_events
                          if event["field"] == field), None)
        if candidate is None:
            continue
        path = out / "oracle_diagnostic" / f"{candidate['review_id']}_oracle_upper_bound.png"
        _oracle_still(candidate, path, package["bank"],
                      package["fields_data"], args.image_root)

    # H0-UNKNOWN feasible rows are labelled counterfactuals, not normal OBS
    # outputs and not blind primary comparisons.
    for event in diagnostics:
        bank_position = package["bank"].position(event["sample_id"])
        order = int(np.where(package["positions"] == bank_position)[0][0])
        state = np.asarray(event["counterfactual_pose"], dtype=np.float64)
        video_path = out / event["diagnostic_video"]
        still_path = out / event["diagnostic_still"]
        mapping = {"A": METHOD_MINIMUM_NORM, "B": METHOD_SWIVEL}
        event["diagnostic_render"] = _render_clip(
            event, video_path, package["bank"], package["positions"],
            package["fields_data"], args.image_root, mapping,
            args.window_seconds, blind=False, counterfactual=state,
            still_path=still_path,
            interpolate_keyframes=args.interpolate_keyframes)

    # Include hashes of every media artifact and of the two blinded-control
    # files. The manifest's own hash is recorded in the worklog to avoid a
    # self-referential digest.
    media_files = []
    for folder in ("blind", "labelled", "technical_stills", "oracle_diagnostic",
                   "counterfactual_diagnostic"):
        for path in sorted((out / folder).glob("*")):
            if path.is_file() and path.suffix.lower() in {".mp4", ".png"}:
                media_files.append({"path": str(path.relative_to(out)), "sha256": _sha256(path),
                                    "bytes": path.stat().st_size})
    write_json(out / "artifact_hashes.json", {
        "schema": "animcv_pose_reconciliation_visual_review_hashes_v1",
        "review_manifest_sha256": _sha256(out / "review_manifest.json"),
        "blind_key_sha256": _sha256(out / "blind_key.json"),
        "review_sheet_sha256": _sha256(out / "review_sheet.csv"),
        "technical_manifest_sha256": _sha256(out / "labelled" / "technical_manifest.json"),
        "media": media_files,
    })
    return {
        "output": str(out),
        "primary_events": len(events),
        "counterfactual_events": len(diagnostics),
        "media_files": len(media_files),
        "manifest_sha256": _sha256(out / "review_manifest.json"),
        "blind_key_sha256": _sha256(out / "blind_key.json"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--replay-report", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--window-seconds", type=float, default=3.0)
    parser.add_argument("--interpolate-keyframes", action="store_true",
                        help="display-only linear interpolation between independent FrameBank rows")
    parser.add_argument("--no-counterfactual-diagnostics", action="store_true",
                        help="omit non-primary H0-UNKNOWN diagnostic exports")
    args = parser.parse_args()
    if args.window_seconds <= 0:
        raise ValueError("window-seconds must be positive")
    result = export_review(args)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
