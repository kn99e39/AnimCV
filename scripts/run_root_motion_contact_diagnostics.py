#!/usr/bin/env python3
"""Root Motion / Foot Contact ownership batch — quantitative + qualitative report.

Produces the section-12/13 evidence for docs/51: Root Orientation diagnostic,
Contact candidate vs. kinematic-proxy reference, and the Root Translation
control experiment (3DPW moving-camera + MPI-INF-3DHP static-camera
contrast). Reads only existing dataset adapters and this batch's new,
independent diagnostic modules under src/pose/ — it does not modify
FramePose, VLM reliability, the temporal refiner, or occlusion reconstruction
work, and it does not touch pose/root_motion.py.

Contact thresholds are fit from the 3DPW TRAIN split only and then applied,
unchanged, to the 3DPW VALIDATION split and to MPI-INF-3DHP. Nothing here is
tuned against a held-out target.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from common.canonical_pose import JOINT_NAMES
from pose.contact import ContactState, ContactThresholds, classify_foot_contact, fit_contact_thresholds
from pose.contact_reference import kinematic_proxy_contact_reference
from pose.mpi3dhp_adapter import load_mpi3dhp_ground_truth
from pose.oracle_world_reference import load_3dpw_world_reference, load_mpi3dhp_world_reference
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
    parser.add_argument("--3dpw-raw", dest="threedpw_raw", required=True, type=Path,
                         help="Path to 3DPW's DATASET_Motion (contains sequenceFiles/, imageFiles/)")
    parser.add_argument("--mpi-root", type=Path, default=None,
                         help="Path to datasets/mpi_inf_3dhp (S*/Seq*/annot.mat) for the static-camera contrast")
    parser.add_argument("--mpi-sequences", type=int, default=4,
                         help="Number of MPI-INF-3DHP S*/Seq1 sequences to use for the static-camera contrast")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--review-out", type=Path, default=None,
                         help="Directory for the small qualitative review package (section 13)")
    args = parser.parse_args()

    train_paths = sorted((args.threedpw_raw / "sequenceFiles" / "train").glob("*.pkl"))
    validation_paths = sorted((args.threedpw_raw / "sequenceFiles" / "validation").glob("*.pkl"))
    if not train_paths or not validation_paths:
        raise SystemExit(f"no 3DPW sequenceFiles found under {args.threedpw_raw}")

    print(f"[1/5] Fitting contact thresholds from {len(train_paths)} TRAIN sequences...")
    thresholds, reference_speed_threshold = _fit_from_train(train_paths)
    print(f"      thresholds: {thresholds}")
    print(f"      reference speed threshold: {reference_speed_threshold:.4f} m/s")

    print(f"[2/5] Running 3DPW VALIDATION diagnostics on {len(validation_paths)} sequences...")
    validation_actors = _run_validation(validation_paths, thresholds, reference_speed_threshold)

    mpi_report = None
    if args.mpi_root is not None:
        print(f"[3/5] Running MPI-INF-3DHP static-camera contrast...")
        mpi_report = _run_mpi_contrast(args.mpi_root, args.mpi_sequences, thresholds, reference_speed_threshold)
    else:
        print("[3/5] Skipped (no --mpi-root given)")

    print("[4/5] Aggregating report...")
    report = _aggregate(thresholds, reference_speed_threshold, validation_actors, mpi_report)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(f"      wrote {args.out}")

    if args.review_out is not None:
        print("[5/5] Building qualitative review package...")
        _build_review_package(args.threedpw_raw, validation_actors, args.review_out)
        print(f"      wrote {args.review_out}")
    else:
        print("[5/5] Skipped (no --review-out given)")


# --------------------------------------------------------------- fitting --

def _fit_from_train(train_paths: list[Path]) -> tuple[dict[str, ContactThresholds], float]:
    train_lifted: dict[str, list[LiftedPoseSequence]] = {"left": [], "right": []}
    train_world_speeds: list[float] = []
    for path in train_paths:
        for _, _, lifted, _ in load_3dpw_ground_truth(path):
            train_lifted["left"].append(lifted)
            train_lifted["right"].append(lifted)
        for _, world_frames in load_3dpw_world_reference(path):
            for foot in FEET:
                positions = [getattr(f, f"{foot}_ankle_world") for f in world_frames]
                train_world_speeds.extend(_speeds(positions, fps=30.0))
    thresholds = {foot: fit_contact_thresholds(train_lifted[foot], foot) for foot in FEET}
    train_world_speeds = sorted(s for s in train_world_speeds if s is not None)
    reference_speed_threshold = _percentile(train_world_speeds, 20.0)
    return thresholds, reference_speed_threshold


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


# ------------------------------------------------------------ validation --

def _run_validation(
    validation_paths: list[Path], thresholds: dict[str, ContactThresholds], reference_speed_threshold: float,
) -> list[dict[str, Any]]:
    actors = []
    for path in validation_paths:
        camera_entries = load_3dpw_ground_truth(path)
        world_entries = load_3dpw_world_reference(path)
        for (sequence_id, _, lifted, _), (world_sequence_id, world_frames) in zip(camera_entries, world_entries):
            assert sequence_id == world_sequence_id
            actors.append(_diagnose_actor(
                sequence_id, path, lifted, world_frames, thresholds, reference_speed_threshold,
            ))
    return actors


def _align_lengths(lifted: LiftedPoseSequence, world_frames: list) -> tuple[LiftedPoseSequence, list]:
    """3DPW's camera-relative and world-reference reads can differ by a few
    trailing frames (three_dpw_adapter.py additionally bounds by len(poses2d),
    which this batch's world-reference loader does not need). Truncate both
    to the common length so every per-frame list stays aligned."""
    common = min(len(lifted.frames), len(world_frames))
    if common == len(lifted.frames) and common == len(world_frames):
        return lifted, world_frames
    truncated = LiftedPoseSequence(
        frames=lifted.frames[:common], source_fps=lifted.source_fps,
        coordinate_frame=lifted.coordinate_frame, units=lifted.units, backend=lifted.backend,
        observation_confidence_threshold=lifted.observation_confidence_threshold,
    )
    return truncated, world_frames[:common]


def _diagnose_actor(
    sequence_id: str, source_path: Path, lifted: LiftedPoseSequence,
    world_frames: list, thresholds: dict[str, ContactThresholds], reference_speed_threshold: float,
) -> dict[str, Any]:
    lifted, world_frames = _align_lengths(lifted, world_frames)
    yaw_observations = observe_yaw(lifted)
    yaw_report = dataclasses.asdict(summarize_yaw(sequence_id, yaw_observations))
    try:
        historical = estimate_root_motion(lifted)
        historical_held_rate = sum(f.yaw_held for f in historical.frames) / len(historical.frames)
    except ValueError:
        historical_held_rate = None

    per_foot: dict[str, Any] = {}
    for foot in FEET:
        candidate = classify_foot_contact(lifted, foot, thresholds[foot])
        world_positions = [getattr(f, f"{foot}_ankle_world") for f in world_frames]
        reference = kinematic_proxy_contact_reference(
            world_positions, fps=lifted.source_fps or 30.0, speed_threshold_m_s=reference_speed_threshold,
        )
        confusion = _confusion(candidate, reference)
        camera_relative_ankle = [
            point.position if (point := frame.points.get(f"{foot}_ankle")) is not None and point.observation_valid
            else None
            for frame in lifted.frames
        ]
        translation_results = run_control_experiment(world_frames, camera_relative_ankle, foot, reference)
        per_foot[foot] = {
            "candidate_counts": _counts(candidate),
            "reference_counts": _counts(reference),
            "confusion_vs_reference": confusion,
            "contact_run_lengths": _run_lengths(candidate, ContactState.CONTACT),
            "translation_control": summarize_translation(translation_results),
            "_candidate": candidate,
            "_reference": reference,
            "_translation_results": translation_results,
            "_camera_relative_ankle": camera_relative_ankle,
        }

    return {
        "sequence_id": sequence_id,
        "source_path": str(source_path),
        "frame_count": len(lifted.frames),
        "root_orientation": yaw_report,
        "historical_root_motion_yaw_held_rate": historical_held_rate,
        "per_foot": per_foot,
        "_world_frames": world_frames,
        "_yaw_observations": yaw_observations,
    }


def _counts(states: list[ContactState]) -> dict[str, int]:
    return {state.value: sum(1 for s in states if s is state) for state in ContactState}


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
    balanced_accuracy = (
        (recall + specificity) / 2.0 if recall is not None and specificity is not None else None
    )
    return {
        "note": "candidate (root-relative kinematics) vs. KINEMATIC_PROXY reference "
                "(world-frame GT ankle speed) -- an internal-consistency check, NOT true contact accuracy.",
        "scored_frame_count": scored,
        "unscored_unknown_count": len(candidate) - scored,
        "true_positive": tp, "false_positive": fp, "false_negative": fn, "true_negative": tn,
        "precision": precision, "recall": recall, "specificity": specificity,
        "balanced_accuracy": balanced_accuracy,
    }


def _run_lengths(states: list[ContactState], target: ContactState) -> dict[str, float | int]:
    lengths = []
    index = 0
    while index < len(states):
        if states[index] is not target:
            index += 1
            continue
        end = index
        while end < len(states) and states[end] is target:
            end += 1
        lengths.append(end - index)
        index = end
    if not lengths:
        return {"count": 0, "mean": None, "median": None, "max": None}
    lengths.sort()
    return {
        "count": len(lengths), "mean": float(np.mean(lengths)),
        "median": float(lengths[len(lengths) // 2]), "max": int(lengths[-1]),
    }


# ---------------------------------------------------------- MPI contrast --

def _run_mpi_contrast(
    mpi_root: Path, sequence_count: int, thresholds: dict[str, ContactThresholds], reference_speed_threshold: float,
) -> dict[str, Any]:
    subject_dirs = sorted(p for p in mpi_root.glob("S*") if p.is_dir())[:sequence_count]
    per_sequence = []
    for subject_dir in subject_dirs:
        seq_dir = subject_dir / "Seq1"
        annotation = seq_dir / "annot.mat"
        if not annotation.exists():
            continue
        _, lifted = load_mpi3dhp_ground_truth(annotation, camera_index=0, end_frame=999)
        world_frames = load_mpi3dhp_world_reference(annotation, camera_index=0, end_frame=999)
        lifted, world_frames = _align_lengths(lifted, world_frames)
        results_by_foot = {}
        for foot in FEET:
            candidate = classify_foot_contact(lifted, foot, thresholds[foot])
            world_positions = [getattr(f, f"{foot}_ankle_world") for f in world_frames]
            reference = kinematic_proxy_contact_reference(
                world_positions, fps=lifted.source_fps or 25.0, speed_threshold_m_s=reference_speed_threshold,
            )
            camera_relative_ankle = [
                point.position if (point := frame.points.get(f"{foot}_ankle")) is not None else None
                for frame in lifted.frames
            ]
            translation_results = run_control_experiment(world_frames, camera_relative_ankle, foot, reference)
            results_by_foot[foot] = summarize_translation(translation_results)
        per_sequence.append({"subject": subject_dir.name, "frame_count": len(lifted.frames), "per_foot": results_by_foot})
    return {
        "condition": "static_camera_contrast",
        "sequences_used": [entry["subject"] for entry in per_sequence],
        "per_sequence": per_sequence,
    }


# --------------------------------------------------------------- report --

def _aggregate(
    thresholds: dict[str, ContactThresholds], reference_speed_threshold: float,
    validation_actors: list[dict[str, Any]], mpi_report: dict[str, Any] | None,
) -> dict[str, Any]:
    yaw_reports = [actor["root_orientation"] for actor in validation_actors]
    total_frames = sum(r["frame_count"] for r in yaw_reports)
    total_unknown = sum(r["unknown_count"] for r in yaw_reports)
    total_flips = sum(r["flip_count"] for r in yaw_reports)
    unreliable_flip_rates = [r["flip_rate_among_unreliable"] for r in yaw_reports if r["flip_rate_among_unreliable"] is not None]
    reliable_flip_rates = [r["flip_rate_among_reliable"] for r in yaw_reports if r["flip_rate_among_reliable"] is not None]
    held_rates = [a["historical_root_motion_yaw_held_rate"] for a in validation_actors if a["historical_root_motion_yaw_held_rate"] is not None]

    contact_summary = {}
    for foot in FEET:
        all_confusion = [a["per_foot"][foot]["confusion_vs_reference"] for a in validation_actors]
        tp = sum(c["true_positive"] for c in all_confusion)
        fp = sum(c["false_positive"] for c in all_confusion)
        fn = sum(c["false_negative"] for c in all_confusion)
        tn = sum(c["true_negative"] for c in all_confusion)
        precision = tp / (tp + fp) if (tp + fp) else None
        recall = tp / (tp + fn) if (tp + fn) else None
        specificity = tn / (tn + fp) if (tn + fp) else None
        balanced_accuracy = (recall + specificity) / 2.0 if recall is not None and specificity is not None else None
        naive_errors = [
            a["per_foot"][foot]["translation_control"]["naive_camera_mean_error_m"] for a in validation_actors
            if a["per_foot"][foot]["translation_control"]["naive_camera_mean_error_m"] is not None
        ]
        world_errors = [
            a["per_foot"][foot]["translation_control"]["world_oracle_mean_error_m"] for a in validation_actors
            if a["per_foot"][foot]["translation_control"]["world_oracle_mean_error_m"] is not None
        ]
        correlations = [
            a["per_foot"][foot]["translation_control"]["naive_error_vs_camera_rotation_correlation"]
            for a in validation_actors
            if a["per_foot"][foot]["translation_control"]["naive_error_vs_camera_rotation_correlation"] is not None
        ]
        contact_summary[foot] = {
            "aggregate_confusion_vs_reference": {
                "true_positive": tp, "false_positive": fp, "false_negative": fn, "true_negative": tn,
                "precision": precision, "recall": recall, "specificity": specificity,
                "balanced_accuracy": balanced_accuracy,
            },
            "translation_control_3dpw_moving_camera": {
                "world_oracle_mean_error_m_across_actors": float(np.mean(world_errors)) if world_errors else None,
                "naive_camera_mean_error_m_across_actors": float(np.mean(naive_errors)) if naive_errors else None,
                "mean_naive_error_vs_rotation_correlation": float(np.mean(correlations)) if correlations else None,
            },
        }

    per_actor_public = []
    for actor in validation_actors:
        entry = {k: v for k, v in actor.items() if not k.startswith("_")}
        entry["per_foot"] = {
            foot: {k: v for k, v in data.items() if not k.startswith("_")}
            for foot, data in actor["per_foot"].items()
        }
        per_actor_public.append(entry)

    return {
        "schema": "animcv_root_motion_contact_ownership_batch_v1",
        "evaluation_regime": "oracle_geometry",
        "contact_thresholds": {
            foot: {
                "contact_speed_m_s": t.contact_speed_m_s, "moving_speed_m_s": t.moving_speed_m_s,
                "height_std_threshold_m": t.height_std_threshold_m, "min_reliable_run": t.min_reliable_run,
            } for foot, t in thresholds.items()
        },
        "contact_reference_speed_threshold_m_s": reference_speed_threshold,
        "root_orientation": {
            "total_frames": total_frames, "total_unknown": total_unknown, "unknown_rate": total_unknown / total_frames if total_frames else None,
            "total_flips": total_flips,
            "mean_flip_rate_among_unreliable_frames": float(np.mean(unreliable_flip_rates)) if unreliable_flip_rates else None,
            "mean_flip_rate_among_reliable_frames": float(np.mean(reliable_flip_rates)) if reliable_flip_rates else None,
            "mean_historical_root_motion_yaw_held_rate": float(np.mean(held_rates)) if held_rates else None,
        },
        "contact": contact_summary,
        "root_translation_control": {
            "3dpw_moving_camera": "see contact.<foot>.translation_control_3dpw_moving_camera above",
            "mpi_inf_3dhp_static_camera_contrast": mpi_report,
        },
        "per_actor": per_actor_public,
    }


# --------------------------------------------------------- review package --

_CATEGORY_JOINTS = ("left_ankle", "right_ankle")


def _build_review_package(threedpw_raw: Path, actors: list[dict[str, Any]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    events = _select_review_events(actors)
    manifest = []
    for category, event in events.items():
        if event is None:
            manifest.append({"category": category, "status": "no_qualifying_case_found"})
            continue
        image_path = (
            threedpw_raw / "imageFiles" / event["sequence_name"] / f"image_{event['frame_index']:05d}.jpg"
        )
        rendered_path = out_dir / f"{category}.png"
        _render_event(image_path, rendered_path, category, event)
        manifest.append({
            "category": category, "status": "rendered", "sequence_id": event["sequence_id"],
            "frame_index": event["frame_index"], "notes": event["notes"], "file": rendered_path.name,
        })
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))


def _select_review_events(actors: list[dict[str, Any]]) -> dict[str, dict[str, Any] | None]:
    events: dict[str, dict[str, Any] | None] = {
        "standing": None, "walking": None, "turning": None, "foot_lift": None,
        "tracking_loss": None, "moving_camera_failure": None, "ankle_short_dropout": None,
    }
    best_standing_speed = float("inf")
    best_walking_alternations = -1
    best_turning_delta = -1.0
    best_moving_camera_error = -1.0
    longest_invalid_run = (-1, None)
    shortest_qualifying_invalid_run = (float("inf"), None)

    for actor in actors:
        sequence_id = actor["sequence_id"]
        sequence_name = sequence_id.split(":")[1]
        candidate_left = actor["per_foot"]["left"]["_candidate"]
        candidate_right = actor["per_foot"]["right"]["_candidate"]
        left_speeds = [s for s in _foot_speed_series(actor, "left") if s is not None]
        right_speeds = [s for s in _foot_speed_series(actor, "right") if s is not None]
        combined_speed = left_speeds + right_speeds
        if combined_speed:
            mean_speed = float(np.mean(combined_speed))
            if mean_speed < best_standing_speed:
                best_standing_speed = mean_speed
                events["standing"] = _event(sequence_id, sequence_name, len(candidate_left) // 2,
                                             f"lowest mean ankle-relative speed ({mean_speed:.4f} m/s)")

        alternations = _alternation_count(candidate_left) + _alternation_count(candidate_right)
        if alternations > best_walking_alternations:
            best_walking_alternations = alternations
            events["walking"] = _event(sequence_id, sequence_name, len(candidate_left) // 2,
                                        f"most CONTACT<->MOVING alternations ({alternations})")

        for observation in actor["_yaw_observations"]:
            previous_reliable_known = [
                o for o in actor["_yaw_observations"]
                if o.frame_index < observation.frame_index and o.yaw_radians is not None
            ]
            if observation.yaw_radians is None or not previous_reliable_known:
                continue
            previous = previous_reliable_known[-1]
            delta = abs(_deg_delta(observation.yaw_radians, previous.yaw_radians))
            if observation.reliable and previous.reliable and delta > best_turning_delta:
                best_turning_delta = delta
                events["turning"] = _event(sequence_id, sequence_name, observation.frame_index,
                                            f"largest reliable frame-to-frame yaw change ({delta:.1f} deg)")

        for foot in FEET:
            translation_results = actor["per_foot"][foot]["_translation_results"]
            for result in translation_results:
                if result.naive_camera_error_m is not None and result.naive_camera_error_m > best_moving_camera_error:
                    best_moving_camera_error = result.naive_camera_error_m
                    events["moving_camera_failure"] = _event(
                        sequence_id, sequence_name, result.frame_index,
                        f"largest naive camera-frame translation error ({result.naive_camera_error_m:.3f} m) "
                        f"at camera rotation delta {result.camera_rotation_delta_degrees}",
                    )
            run_start = _first_run_start(actor["per_foot"][foot]["_candidate"], ContactState.MOVING, ContactState.CONTACT)
            if run_start is not None and events["foot_lift"] is None:
                events["foot_lift"] = _event(sequence_id, sequence_name, run_start,
                                              f"start of a {foot} MOVING run following sustained CONTACT")

        world_frames = actor["_world_frames"]
        invalid_run = _longest_invalid_run(world_frames)
        if invalid_run is not None:
            length, start = invalid_run
            if length > longest_invalid_run[0]:
                longest_invalid_run = (length, (sequence_id, sequence_name, start + length // 2))
            if 1 <= length <= 3 and length < shortest_qualifying_invalid_run[0]:
                shortest_qualifying_invalid_run = (length, (sequence_id, sequence_name, start))

    if longest_invalid_run[1] is not None:
        sequence_id, sequence_name, frame_index = longest_invalid_run[1]
        events["tracking_loss"] = _event(sequence_id, sequence_name, frame_index,
                                          f"longest camera-pose-invalid run ({longest_invalid_run[0]} frames)")
    if shortest_qualifying_invalid_run[1] is not None:
        sequence_id, sequence_name, frame_index = shortest_qualifying_invalid_run[1]
        events["ankle_short_dropout"] = _event(sequence_id, sequence_name, frame_index,
                                                f"short ({shortest_qualifying_invalid_run[0]}-frame) validity dropout")
    return events


def _foot_speed_series(actor: dict[str, Any], foot: str) -> list[float | None]:
    world_frames = actor["_world_frames"]
    positions = [getattr(f, f"{foot}_ankle_world") for f in world_frames]
    return _speeds(positions, fps=30.0)


def _alternation_count(states: list[ContactState]) -> int:
    scored = [s for s in states if s is not ContactState.UNKNOWN]
    return sum(1 for a, b in zip(scored, scored[1:]) if a is not b)


def _deg_delta(a: float, b: float) -> float:
    import math
    return math.degrees((a - b + math.pi) % (2 * math.pi) - math.pi)


def _first_run_start(states: list[ContactState], target: ContactState, preceding: ContactState) -> int | None:
    for index in range(1, len(states)):
        if states[index] is target and states[index - 1] is preceding:
            return index
    return None


def _longest_invalid_run(world_frames: list) -> tuple[int, int] | None:
    best_length, best_start = 0, None
    index = 0
    while index < len(world_frames):
        if world_frames[index].valid:
            index += 1
            continue
        start = index
        while index < len(world_frames) and not world_frames[index].valid:
            index += 1
        if index - start > best_length:
            best_length, best_start = index - start, start
    return (best_length, best_start) if best_start is not None else None


def _event(sequence_id: str, sequence_name: str, frame_index: int, notes: str) -> dict[str, Any]:
    return {"sequence_id": sequence_id, "sequence_name": sequence_name, "frame_index": int(frame_index), "notes": notes}


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
