#!/usr/bin/env python3
"""Production-evidence replay: do docs/51's Root Orientation / Contact signals
survive the frozen FramePose H0 output (benchmark_detector_observation)?

Reuses docs/51's modules UNCHANGED (root_orientation_diagnostic.py,
pose.root_motion.estimate_root_motion, contact.py, contact_reference.py,
root_translation_control.py, oracle_world_reference.py) and drives them with
real H0 predictions via the new framepose_bridge.py, instead of designing any
new signal. Oracle (O) and H0 (H) are always reported separately and never
mixed into one metric. Both regimes are evaluated on the SAME matched frame
subset (the FrameBank's sampled frame_index set per sequence) so a signal
difference reflects H0 quality, not a sampling-rate artifact between the two
regimes.

Does not modify FramePose, retrain H0, or tune contact thresholds on
anything but TRAIN.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from framepose.contract import FrameBank
from framepose.bank import load_bank
from pose.contact import ContactState, ContactThresholds, classify_foot_contact, fit_contact_thresholds
from pose.contact_failure_attribution import (
    attribute_disagreements, fit_position_error_threshold, per_frame_joint_errors, summarize_attribution,
)
from pose.contact_reference import kinematic_proxy_contact_reference
from pose.framepose_bridge import assemble_h0, build_h0_lifted_sequence, sequence_ids_in_split
from pose.oracle_world_reference import WorldReferenceFrame, load_3dpw_world_reference
from pose.pose_lifter import LiftedPoseSequence
from pose.root_motion import estimate_root_motion
from pose.root_orientation_diagnostic import observe_yaw
from pose.root_orientation_diagnostic import summarize as summarize_yaw
from pose.root_translation_control import run_control_experiment
from pose.root_translation_control import summarize as summarize_translation
from pose.three_dpw_adapter import load_3dpw_ground_truth

FEET = ("left", "right")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", required=True, type=Path)
    parser.add_argument("--h0-train", required=True, type=Path)
    parser.add_argument("--h0-validation", required=True, type=Path)
    parser.add_argument("--3dpw-raw", dest="threedpw_raw", required=True, type=Path)
    parser.add_argument("--expect-bank-content-digest", default=None)
    parser.add_argument("--expect-h0-validation-sha256", default=None)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--review-out", type=Path, default=None)
    args = parser.parse_args()

    print("[1/6] Loading bank + assembling frozen H0...")
    bank = load_bank(args.bank)
    h0, identity = assemble_h0(bank, {"train": args.h0_train, "validation": args.h0_validation})
    if args.expect_bank_content_digest and identity.bank_content_digest != args.expect_bank_content_digest:
        raise SystemExit(f"bank_content_digest mismatch: {identity.bank_content_digest}")
    if args.expect_h0_validation_sha256 and identity.split_sha256["validation"] != args.expect_h0_validation_sha256:
        raise SystemExit(f"H0 validation sha256 mismatch: {identity.split_sha256['validation']}")
    print(f"      bank_content_digest={identity.bank_content_digest}")
    print(f"      h0 validation sha256={identity.split_sha256['validation']}")

    print("[2/6] Fitting T0 (oracle/TRAIN, docs/51 identical) and T1 (H0/TRAIN) contact thresholds...")
    t0_thresholds, reference_speed_threshold = _fit_oracle_train_thresholds(args.threedpw_raw)
    t1_thresholds = _fit_h0_train_thresholds(bank, h0)
    print(f"      T0: {t0_thresholds}")
    print(f"      T1: {t1_thresholds}")

    print("[3/6] Fitting TRAIN-only H0-vs-oracle position error threshold for failure attribution...")
    position_error_threshold = _fit_position_error_threshold(bank, h0, args.threedpw_raw)
    print(f"      position_error_threshold_m={position_error_threshold:.4f}")

    print("[4/6] Replaying VALIDATION sequences (O vs H0)...")
    validation_sequence_ids = sequence_ids_in_split(bank, "validation")
    actors = []
    for sequence_id in validation_sequence_ids:
        actors.append(_replay_actor(
            bank, h0, sequence_id, args.threedpw_raw, t0_thresholds, t1_thresholds,
            reference_speed_threshold, position_error_threshold,
        ))

    print("[5/6] Aggregating report...")
    report = _aggregate(identity, t0_thresholds, t1_thresholds, reference_speed_threshold,
                         position_error_threshold, actors)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(f"      wrote {args.out}")

    if args.review_out is not None:
        print("[6/6] Building qualitative review package...")
        _build_review_package(args.threedpw_raw, actors, args.review_out)
        print(f"      wrote {args.review_out}")
    else:
        print("[6/6] Skipped (no --review-out given)")


# ------------------------------------------------------------- thresholds --

def _fit_oracle_train_thresholds(threedpw_raw: Path) -> tuple[dict[str, ContactThresholds], float]:
    """Reproduces docs/51's T0 exactly: same function, same TRAIN data source."""
    train_paths = sorted((threedpw_raw / "sequenceFiles" / "train").glob("*.pkl"))
    train_lifted = []
    train_world_speeds: list[float] = []
    for path in train_paths:
        for _, _, lifted, _ in load_3dpw_ground_truth(path):
            train_lifted.append(lifted)
        for _, world_frames in load_3dpw_world_reference(path):
            for foot in FEET:
                positions = [getattr(f, f"{foot}_ankle_world") for f in world_frames]
                train_world_speeds.extend(s for s in _speeds(positions, fps=30.0) if s is not None)
    thresholds = {foot: fit_contact_thresholds(train_lifted, foot) for foot in FEET}
    train_world_speeds.sort()
    reference_speed_threshold = _percentile(train_world_speeds, 20.0)
    return thresholds, reference_speed_threshold


def _fit_h0_train_thresholds(bank: FrameBank, h0: np.ndarray) -> dict[str, ContactThresholds]:
    train_sequence_ids = sequence_ids_in_split(bank, "train")
    train_h0_sequences = [build_h0_lifted_sequence(bank, h0, sequence_id) for sequence_id in train_sequence_ids]
    return {foot: fit_contact_thresholds(train_h0_sequences, foot) for foot in FEET}


def _fit_position_error_threshold(bank: FrameBank, h0: np.ndarray, threedpw_raw: Path) -> float:
    train_sequence_ids = sequence_ids_in_split(bank, "train")
    all_errors: list[float] = []
    for sequence_id in train_sequence_ids:
        h0_sequence = build_h0_lifted_sequence(bank, h0, sequence_id)
        bank_frame_indices = [frame.frame_index for frame in h0_sequence.frames]
        oracle_sequence = _matched_oracle_sequence(sequence_id, threedpw_raw, bank_frame_indices)
        if oracle_sequence is None:
            continue
        for frame_errors in per_frame_joint_errors(oracle_sequence, h0_sequence):
            all_errors.extend(value for value in frame_errors.values() if value is not None)
    return fit_position_error_threshold(all_errors)


# ---------------------------------------------------------------- oracle --

def _parse_3dpw_sequence_id(sequence_id: str) -> tuple[str, int]:
    prefix, name, actor_part = sequence_id.split(":")
    if prefix != "3dpw" or not actor_part.startswith("actor"):
        raise ValueError(f"unrecognized 3DPW sequence_id: {sequence_id!r}")
    return name, int(actor_part[len("actor"):])


def _find_3dpw_pkl(raw_root: Path, sequence_name: str) -> Path:
    for split in ("train", "validation", "test"):
        candidate = raw_root / "sequenceFiles" / split / f"{sequence_name}.pkl"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"no 3DPW pkl found for sequence {sequence_name!r} under {raw_root}")


def _matched_oracle_sequence(
    sequence_id: str, threedpw_raw: Path, bank_frame_indices: list[int],
) -> LiftedPoseSequence | None:
    name, actor_index = _parse_3dpw_sequence_id(sequence_id)
    path = _find_3dpw_pkl(threedpw_raw, name)
    entries = load_3dpw_ground_truth(path)
    if actor_index >= len(entries):
        return None
    _, _, lifted, _ = entries[actor_index]
    by_index = {frame.frame_index: frame for frame in lifted.frames}
    missing = [i for i in bank_frame_indices if i not in by_index]
    if missing:
        raise ValueError(f"{sequence_id}: bank frame_index {missing[:5]} not present in oracle sequence")
    matched_frames = [by_index[i] for i in bank_frame_indices]
    return LiftedPoseSequence(frames=matched_frames, source_fps=lifted.source_fps, backend=lifted.backend)


def _matched_world_frames(
    sequence_id: str, threedpw_raw: Path, bank_frame_indices: list[int],
) -> list[WorldReferenceFrame] | None:
    name, actor_index = _parse_3dpw_sequence_id(sequence_id)
    path = _find_3dpw_pkl(threedpw_raw, name)
    entries = load_3dpw_world_reference(path)
    if actor_index >= len(entries):
        return None
    _, world_frames = entries[actor_index]
    if max(bank_frame_indices) >= len(world_frames):
        raise ValueError(f"{sequence_id}: bank frame_index exceeds world reference length")
    return [world_frames[i] for i in bank_frame_indices]


# --------------------------------------------------------------- per-actor --

def _replay_actor(
    bank: FrameBank, h0: np.ndarray, sequence_id: str, threedpw_raw: Path,
    t0_thresholds: dict[str, ContactThresholds], t1_thresholds: dict[str, ContactThresholds],
    reference_speed_threshold: float, position_error_threshold: float,
) -> dict[str, Any]:
    h0_sequence = build_h0_lifted_sequence(bank, h0, sequence_id)
    bank_frame_indices = [frame.frame_index for frame in h0_sequence.frames]
    oracle_sequence = _matched_oracle_sequence(sequence_id, threedpw_raw, bank_frame_indices)
    world_frames = _matched_world_frames(sequence_id, threedpw_raw, bank_frame_indices)

    # --- section 3: Root Orientation replay (unchanged observe_yaw/summarize) ---
    oracle_yaw_obs = observe_yaw(oracle_sequence)
    h0_yaw_obs = observe_yaw(h0_sequence)
    oracle_yaw_report = dataclasses.asdict(summarize_yaw(sequence_id, oracle_yaw_obs))
    h0_yaw_report = dataclasses.asdict(summarize_yaw(sequence_id, h0_yaw_obs))
    yaw_errors, sign_flips = [], 0
    for o_obs, h_obs in zip(oracle_yaw_obs, h0_yaw_obs):
        if o_obs.yaw_radians is None or h_obs.yaw_radians is None:
            continue
        delta = abs(_angle_delta(h_obs.yaw_radians, o_obs.yaw_radians)) * 180.0 / np.pi
        yaw_errors.append(delta)
        if delta > 90.0:
            sign_flips += 1

    # --- section 4: historical hold, replayed unchanged, diagnostic only ---
    try:
        h0_historical = estimate_root_motion(h0_sequence)
        h0_held_rate = sum(f.yaw_held for f in h0_historical.frames) / len(h0_historical.frames)
        suppressed_true_rotation, prevented_flip = 0, 0
        oracle_deltas_by_index = {}
        for previous, current in zip(oracle_yaw_obs, oracle_yaw_obs[1:]):
            if previous.yaw_radians is not None and current.yaw_radians is not None:
                oracle_deltas_by_index[current.frame_index] = abs(
                    _angle_delta(current.yaw_radians, previous.yaw_radians)
                ) * 180.0 / np.pi
        for frame in h0_historical.frames:
            if not frame.yaw_held:
                continue
            oracle_delta = oracle_deltas_by_index.get(frame.frame_index)
            if oracle_delta is None:
                continue
            if oracle_delta > 20.0:  # matches estimate_root_motion's own default max_yaw_step_degrees
                suppressed_true_rotation += 1
            else:
                prevented_flip += 1
    except ValueError:
        h0_held_rate = None
        suppressed_true_rotation = prevented_flip = None

    # --- sections 5-8: contact + translation control, per foot ---
    per_foot: dict[str, Any] = {}
    for foot in FEET:
        candidate_oracle = classify_foot_contact(oracle_sequence, foot, t0_thresholds[foot])
        candidate_h0_t0 = classify_foot_contact(h0_sequence, foot, t0_thresholds[foot])
        candidate_h0_t1 = classify_foot_contact(h0_sequence, foot, t1_thresholds[foot])
        reference = None
        if world_frames is not None:
            world_positions = [getattr(f, f"{foot}_ankle_world") for f in world_frames]
            reference = kinematic_proxy_contact_reference(
                world_positions, fps=oracle_sequence.source_fps or 30.0,
                speed_threshold_m_s=reference_speed_threshold,
            )

        joint_errors = per_frame_joint_errors(oracle_sequence, h0_sequence)
        attributions = (
            attribute_disagreements(joint_errors, foot, candidate_oracle, candidate_h0_t0,
                                     reference, position_error_threshold)
            if reference is not None else []
        )

        h0_camera_relative_ankle = [
            point.position if (point := frame.points.get(f"{foot}_ankle")) is not None and point.observation_valid
            else None
            for frame in h0_sequence.frames
        ]

        translation_o = translation_r0 = translation_r1 = translation_r1_t1 = None
        if world_frames is not None and reference is not None:
            oracle_camera_relative_ankle = [
                point.position if (point := frame.points.get(f"{foot}_ankle")) is not None and point.observation_valid
                else None
                for frame in oracle_sequence.frames
            ]
            translation_o = summarize_translation(
                run_control_experiment(world_frames, oracle_camera_relative_ankle, foot, reference)
            )
            translation_r0 = summarize_translation(
                run_control_experiment(world_frames, h0_camera_relative_ankle, foot, reference)
            )
            translation_r1 = summarize_translation(
                run_control_experiment(world_frames, h0_camera_relative_ankle, foot, candidate_h0_t0)
            )
            # Supplementary: R1 recomputed with the T1 (H0-train-calibrated)
            # contact selector, not asked for by name in the directive but
            # necessary to tell "H0 geometry is the bottleneck" apart from
            # "T0-on-H0 contact selection is the bottleneck" -- R1 above uses
            # T0, which section 6 already shows collapses H0 recall to ~0.
            translation_r1_t1 = summarize_translation(
                run_control_experiment(world_frames, h0_camera_relative_ankle, foot, candidate_h0_t1)
            )

        per_foot[foot] = {
            "candidate_oracle_counts": _counts(candidate_oracle),
            "candidate_h0_t0_counts": _counts(candidate_h0_t0),
            "candidate_h0_t1_counts": _counts(candidate_h0_t1),
            "reference_counts": _counts(reference) if reference is not None else None,
            "confusion_oracle_vs_reference": _confusion(candidate_oracle, reference) if reference is not None else None,
            "confusion_h0_t0_vs_reference": _confusion(candidate_h0_t0, reference) if reference is not None else None,
            "confusion_h0_t1_vs_reference": _confusion(candidate_h0_t1, reference) if reference is not None else None,
            "oracle_vs_h0_t0_disagreement_rate": _disagreement_rate(candidate_oracle, candidate_h0_t0),
            "contact_run_lengths_oracle": _run_lengths(candidate_oracle),
            "contact_run_lengths_h0_t0": _run_lengths(candidate_h0_t0),
            "failure_attribution": summarize_attribution(attributions),
            "translation_control_O_oracle_contact_oracle_geometry": translation_o,
            "translation_control_R0_oracle_contact_h0_geometry": translation_r0,
            "translation_control_R1_h0_contact_h0_geometry": translation_r1,
            "translation_control_R1_T1_h0_contact_h0_geometry": translation_r1_t1,
            "_candidate_oracle": candidate_oracle, "_candidate_h0_t0": candidate_h0_t0, "_reference": reference,
            "_joint_errors": joint_errors,
        }

    return {
        "sequence_id": sequence_id,
        "frame_count": len(h0_sequence.frames),
        "bank_frame_indices": bank_frame_indices,
        "root_orientation_oracle": oracle_yaw_report,
        "root_orientation_h0": h0_yaw_report,
        "yaw_error_degrees_mean": float(np.mean(yaw_errors)) if yaw_errors else None,
        "yaw_error_degrees_max": float(np.max(yaw_errors)) if yaw_errors else None,
        "yaw_sign_flip_count": sign_flips,
        "yaw_compared_frame_count": len(yaw_errors),
        "h0_historical_hold_rate": h0_held_rate,
        "h0_hold_suppressed_true_rotation_count": suppressed_true_rotation,
        "h0_hold_prevented_apparent_flip_count": prevented_flip,
        "per_foot": per_foot,
        "_oracle_sequence": oracle_sequence, "_h0_sequence": h0_sequence, "_world_frames": world_frames,
    }


# ----------------------------------------------------------------- helpers --

def _angle_delta(a: float, b: float) -> float:
    return (a - b + np.pi) % (2 * np.pi) - np.pi


def _speeds(positions: list[tuple[float, float, float] | None], fps: float) -> list[float | None]:
    dt = 1.0 / fps
    speeds: list[float | None] = [None] * len(positions)
    for index, current in enumerate(positions):
        if current is None:
            continue
        previous = positions[index - 1] if index - 1 >= 0 else None
        following = positions[index + 1] if index + 1 < len(positions) else None
        if previous is not None and following is not None:
            speeds[index] = float(np.linalg.norm(np.subtract(following, previous)) / (2 * dt))
        elif previous is not None:
            speeds[index] = float(np.linalg.norm(np.subtract(current, previous)) / dt)
        elif following is not None:
            speeds[index] = float(np.linalg.norm(np.subtract(following, current)) / dt)
    return speeds


def _percentile(sorted_values: list[float], percentile: float) -> float:
    if not sorted_values:
        return 0.0
    position = (len(sorted_values) - 1) * (percentile / 100.0)
    lower, upper = int(np.floor(position)), int(np.ceil(position))
    if lower == upper:
        return sorted_values[int(position)]
    return sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * (position - lower)


def _counts(states: list[ContactState] | None) -> dict[str, int] | None:
    if states is None:
        return None
    return {state.value: sum(1 for s in states if s is state) for state in ContactState}


def _disagreement_rate(a: list[ContactState], b: list[ContactState]) -> float:
    return sum(1 for x, y in zip(a, b) if x is not y) / len(a)


def _confusion(candidate: list[ContactState], reference: list[ContactState]) -> dict[str, Any]:
    tp = fp = fn = tn = 0
    for c, r in zip(candidate, reference):
        if c is ContactState.UNKNOWN or r is ContactState.UNKNOWN:
            continue
        if c is ContactState.CONTACT and r is ContactState.CONTACT:
            tp += 1
        elif c is ContactState.CONTACT and r is not ContactState.CONTACT:
            fp += 1
        elif c is not ContactState.CONTACT and r is ContactState.CONTACT:
            fn += 1
        else:
            tn += 1
    scored = tp + fp + fn + tn
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    specificity = tn / (tn + fp) if (tn + fp) else None
    balanced_accuracy = (recall + specificity) / 2.0 if recall is not None and specificity is not None else None
    return {
        "scored_frame_count": scored, "unscored_unknown_count": len(candidate) - scored,
        "true_positive": tp, "false_positive": fp, "false_negative": fn, "true_negative": tn,
        "precision": precision, "recall": recall, "specificity": specificity, "balanced_accuracy": balanced_accuracy,
    }


def _run_lengths(states: list[ContactState]) -> dict[str, float | int]:
    lengths = []
    index = 0
    while index < len(states):
        if states[index] is not ContactState.CONTACT:
            index += 1
            continue
        end = index
        while end < len(states) and states[end] is ContactState.CONTACT:
            end += 1
        lengths.append(end - index)
        index = end
    if not lengths:
        return {"count": 0, "mean": None, "median": None, "max": None}
    lengths.sort()
    return {"count": len(lengths), "mean": float(np.mean(lengths)),
            "median": float(lengths[len(lengths) // 2]), "max": int(lengths[-1])}


# ----------------------------------------------------------------- report --

def _aggregate(
    identity, t0_thresholds, t1_thresholds, reference_speed_threshold, position_error_threshold,
    actors: list[dict[str, Any]],
) -> dict[str, Any]:
    total_frames = sum(a["frame_count"] for a in actors)
    oracle_unknown = sum(a["root_orientation_oracle"]["unknown_count"] for a in actors)
    h0_unknown = sum(a["root_orientation_h0"]["unknown_count"] for a in actors)
    oracle_flips = sum(a["root_orientation_oracle"]["flip_count"] for a in actors)
    h0_flips = sum(a["root_orientation_h0"]["flip_count"] for a in actors)
    yaw_errors = [a["yaw_error_degrees_mean"] for a in actors if a["yaw_error_degrees_mean"] is not None]
    sign_flip_total = sum(a["yaw_sign_flip_count"] for a in actors)
    hold_rates = [a["h0_historical_hold_rate"] for a in actors if a["h0_historical_hold_rate"] is not None]
    suppressed = sum(a["h0_hold_suppressed_true_rotation_count"] or 0 for a in actors)
    prevented = sum(a["h0_hold_prevented_apparent_flip_count"] or 0 for a in actors)

    contact_summary = {}
    for foot in FEET:
        agg = {"oracle": {"tp": 0, "fp": 0, "fn": 0, "tn": 0}, "h0_t0": {"tp": 0, "fp": 0, "fn": 0, "tn": 0},
               "h0_t1": {"tp": 0, "fp": 0, "fn": 0, "tn": 0}}
        disagreement_rates = []
        attribution_totals: dict[str, int] = {}
        for actor in actors:
            data = actor["per_foot"][foot]
            for key, confusion_key in (("oracle", "confusion_oracle_vs_reference"),
                                        ("h0_t0", "confusion_h0_t0_vs_reference"),
                                        ("h0_t1", "confusion_h0_t1_vs_reference")):
                confusion = data[confusion_key]
                if confusion is None:
                    continue
                agg[key]["tp"] += confusion["true_positive"]
                agg[key]["fp"] += confusion["false_positive"]
                agg[key]["fn"] += confusion["false_negative"]
                agg[key]["tn"] += confusion["true_negative"]
            disagreement_rates.append(data["oracle_vs_h0_t0_disagreement_rate"])
            for k, v in data["failure_attribution"].items():
                attribution_totals[k] = attribution_totals.get(k, 0) + v

        def _derive(counts):
            tp, fp, fn, tn = counts["tp"], counts["fp"], counts["fn"], counts["tn"]
            precision = tp / (tp + fp) if (tp + fp) else None
            recall = tp / (tp + fn) if (tp + fn) else None
            specificity = tn / (tn + fp) if (tn + fp) else None
            balanced_accuracy = (recall + specificity) / 2.0 if recall is not None and specificity is not None else None
            return {**counts, "precision": precision, "recall": recall,
                    "specificity": specificity, "balanced_accuracy": balanced_accuracy}

        translation_summary = {}
        for condition in ("O_oracle_contact_oracle_geometry", "R0_oracle_contact_h0_geometry",
                          "R1_h0_contact_h0_geometry", "R1_T1_h0_contact_h0_geometry"):
            key = f"translation_control_{condition}"
            errors = [a["per_foot"][foot][key]["naive_camera_mean_error_m"] for a in actors
                      if a["per_foot"][foot][key] is not None and a["per_foot"][foot][key]["naive_camera_mean_error_m"] is not None]
            recoverable = sum(a["per_foot"][foot][key]["recoverable_segment_count"] for a in actors if a["per_foot"][foot][key] is not None)
            unresolved = sum(a["per_foot"][foot][key]["unresolved_segment_count"] for a in actors if a["per_foot"][foot][key] is not None)
            translation_summary[condition] = {
                "mean_error_m_across_actors": float(np.mean(errors)) if errors else None,
                "recoverable_segment_count": recoverable, "unresolved_segment_count": unresolved,
            }

        contact_summary[foot] = {
            "aggregate_confusion": {key: _derive(counts) for key, counts in agg.items()},
            "mean_oracle_vs_h0_t0_disagreement_rate": float(np.mean(disagreement_rates)) if disagreement_rates else None,
            "failure_attribution_totals": attribution_totals,
            "root_translation_control": translation_summary,
        }

    per_actor_public = []
    for actor in actors:
        entry = {k: v for k, v in actor.items() if not k.startswith("_")}
        entry["per_foot"] = {foot: {k: v for k, v in data.items() if not k.startswith("_")}
                             for foot, data in actor["per_foot"].items()}
        per_actor_public.append(entry)

    return {
        "schema": "animcv_root_motion_contact_h0_replay_v1",
        "h0_identity": dataclasses.asdict(identity),
        "contact_thresholds_T0_oracle_train": {
            foot: dataclasses.asdict(t) for foot, t in t0_thresholds.items()
        },
        "contact_thresholds_T1_h0_train": {
            foot: dataclasses.asdict(t) for foot, t in t1_thresholds.items()
        },
        "contact_reference_speed_threshold_m_s": reference_speed_threshold,
        "position_error_threshold_m_train_90th_pct": position_error_threshold,
        "root_orientation": {
            "total_frames": total_frames,
            "oracle_unknown_rate": oracle_unknown / total_frames if total_frames else None,
            "h0_unknown_rate": h0_unknown / total_frames if total_frames else None,
            "oracle_total_flips": oracle_flips, "h0_total_flips": h0_flips,
            "mean_yaw_error_degrees": float(np.mean(yaw_errors)) if yaw_errors else None,
            "total_yaw_sign_flip_count": sign_flip_total,
            "mean_h0_historical_hold_rate": float(np.mean(hold_rates)) if hold_rates else None,
            "total_hold_suppressed_true_rotation": suppressed,
            "total_hold_prevented_apparent_flip": prevented,
        },
        "contact": contact_summary,
        "per_actor": per_actor_public,
    }


# --------------------------------------------------------- review package --

def _build_review_package(threedpw_raw: Path, actors: list[dict[str, Any]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    events = _select_review_events(actors)
    manifest = []
    for category, event in events.items():
        if event is None:
            manifest.append({"category": category, "status": "no_qualifying_case_found"})
            continue
        name = event["sequence_id"].split(":")[1]
        image_path = threedpw_raw / "imageFiles" / name / f"image_{event['frame_index']:05d}.jpg"
        rendered_path = out_dir / f"{category}.png"
        _render_event(image_path, rendered_path, category, event)
        manifest.append({"category": category, "status": "rendered", **event, "file": rendered_path.name})
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))


def _select_review_events(actors: list[dict[str, Any]]) -> dict[str, dict[str, Any] | None]:
    events: dict[str, dict[str, Any] | None] = {
        "known_good_pose": None, "ankle_or_foot_loss": None, "turning": None,
        "tracking_loss": None, "walking": None,
    }
    best_good, worst_ankle, best_turn, longest_invalid, best_walk = (
        float("inf"), -1.0, -1.0, (-1, None), -1,
    )
    for actor in actors:
        sequence_id = actor["sequence_id"]
        mean_error_per_frame = []
        worst_ankle_per_frame = []
        for index in range(actor["frame_count"]):
            values = []
            ankle_values = []
            for foot in FEET:
                errors = actor["per_foot"][foot]["_joint_errors"][index]
                values.extend(v for v in errors.values() if v is not None)
                ankle_value = errors.get(f"{foot}_ankle")
                if ankle_value is not None:
                    ankle_values.append(ankle_value)
            mean_error_per_frame.append(float(np.mean(values)) if values else None)
            worst_ankle_per_frame.append(max(ankle_values) if ankle_values else None)

        for index, value in enumerate(mean_error_per_frame):
            if value is not None and value < best_good:
                best_good = value
                events["known_good_pose"] = _event(sequence_id, actor["bank_frame_indices"][index],
                                                    f"lowest mean H0-vs-oracle joint error ({value:.4f} m)")
        for index, value in enumerate(worst_ankle_per_frame):
            if value is not None and value > worst_ankle:
                worst_ankle = value
                events["ankle_or_foot_loss"] = _event(sequence_id, actor["bank_frame_indices"][index],
                                                       f"largest H0-vs-oracle ankle error ({value:.4f} m)")

        for index in range(1, actor["frame_count"]):
            h0_seq = actor["_h0_sequence"]
            oracle_seq = actor["_oracle_sequence"]
            h0_obs = observe_yaw(LiftedPoseSequence(frames=[h0_seq.frames[index - 1], h0_seq.frames[index]],
                                                     source_fps=h0_seq.source_fps))
            if h0_obs[0].yaw_radians is None or h0_obs[1].yaw_radians is None:
                continue
            if not (h0_obs[0].reliable and h0_obs[1].reliable):
                continue
            delta = abs(_angle_delta(h0_obs[1].yaw_radians, h0_obs[0].yaw_radians)) * 180.0 / np.pi
            if delta > best_turn:
                best_turn = delta
                events["turning"] = _event(sequence_id, actor["bank_frame_indices"][index],
                                            f"largest reliable H0 frame-to-frame yaw change ({delta:.1f} deg)")

        candidate_left = actor["per_foot"]["left"]["_candidate_h0_t0"]
        candidate_right = actor["per_foot"]["right"]["_candidate_h0_t0"]
        alternations = _alternation_count(candidate_left) + _alternation_count(candidate_right)
        if alternations > best_walk:
            best_walk = alternations
            events["walking"] = _event(sequence_id, actor["bank_frame_indices"][len(candidate_left) // 2],
                                        f"most H0 CONTACT<->MOVING alternations ({alternations})")

        h0_seq = actor["_h0_sequence"]
        invalid_run = _longest_invalid_run(h0_seq)
        if invalid_run is not None and invalid_run[0] > longest_invalid[0]:
            length, start = invalid_run
            longest_invalid = (length, (sequence_id, actor["bank_frame_indices"][start + length // 2]))

    if longest_invalid[1] is not None:
        sequence_id, frame_index = longest_invalid[1]
        events["tracking_loss"] = _event(sequence_id, frame_index,
                                          f"longest H0 any-joint-invalid run ({longest_invalid[0]} bank rows)")
    return events


def _alternation_count(states: list[ContactState]) -> int:
    scored = [s for s in states if s is not ContactState.UNKNOWN]
    return sum(1 for a, b in zip(scored, scored[1:]) if a is not b)


def _longest_invalid_run(sequence: LiftedPoseSequence) -> tuple[int, int] | None:
    validity = [all(point.observation_valid for point in frame.points.values()) for frame in sequence.frames]
    best_length, best_start = 0, None
    index = 0
    while index < len(validity):
        if validity[index]:
            index += 1
            continue
        start = index
        while index < len(validity) and not validity[index]:
            index += 1
        if index - start > best_length:
            best_length, best_start = index - start, start
    return (best_length, best_start) if best_start is not None else None


def _event(sequence_id: str, frame_index: int, notes: str) -> dict[str, Any]:
    return {"sequence_id": sequence_id, "frame_index": int(frame_index), "notes": notes}


def _render_event(image_path: Path, out_path: Path, category: str, event: dict[str, Any]) -> None:
    image = cv2.imread(str(image_path))
    if image is None:
        out_path.write_bytes(b"")
        return
    cv2.rectangle(image, (0, 0), (image.shape[1], 40), (20, 20, 20), -1)
    cv2.putText(image, f"{category}: {event['sequence_id']} frame {event['frame_index']}",
                (8, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imwrite(str(out_path), image)


if __name__ == "__main__":
    main()
