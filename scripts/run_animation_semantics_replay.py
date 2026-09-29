#!/usr/bin/env python3
"""Contract replay of the FramePose H0 -> AnimationSemantics bridge (docs/53).

Validates the representation, not animation quality: builds AnimationSemantics
for every frozen-H0 VALIDATION sequence, checks deterministic replay and
serialization round-trip identity, and accounts for every semantic state.
Contact is calibrated once on H0 TRAIN (docs/52 T1 procedure, unchanged) and
persisted; nothing is fit on validation. Regime: benchmark_detector_observation.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from framepose.bank import load_bank
from motion.animation_semantics import (
    AnimationSemantics, load_animation_semantics, load_contact_calibration, save_animation_semantics,
    save_contact_calibration,
)
from motion.animation_semantics_bridge import build_animation_semantics, calibrate_contact_from_h0_train
from pose.contact import ContactState
from pose.framepose_bridge import assemble_h0, sequence_ids_in_split

SIDES = ("left", "right")

# docs/52 review manifest (~/animcv-output/root_motion_contact_h0_replay/review/manifest.json).
# Its "walking" case was selected from T0 states (1 alternation), so the
# contact-like case here is re-selected from the T1 semantics and both are shown.
DOCS52_CASES = {
    "known_good_pose": ("3dpw:courtyard_dancing_00:actor0", 424),
    "turning": ("3dpw:outdoors_crosscountry_00:actor0", 202),
    "ankle_failure": ("3dpw:outdoors_parcours_00:actor0", 384),
    "tracking_loss": ("3dpw:courtyard_hug_00:actor1", 312),
    "walking_docs52_t0_selected": ("3dpw:courtyard_rangeOfMotions_01:actor0", 234),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", required=True, type=Path)
    parser.add_argument("--h0-train", required=True, type=Path)
    parser.add_argument("--h0-validation", required=True, type=Path)
    parser.add_argument("--expect-bank-content-digest", default=None)
    parser.add_argument("--expect-h0-train-sha256", default=None)
    parser.add_argument("--expect-h0-validation-sha256", default=None)
    parser.add_argument("--docs52-report", type=Path, default=None,
                        help="docs/52 report.json; T1 thresholds must reproduce exactly")
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()

    bank = load_bank(args.bank)
    h0, identity = assemble_h0(bank, {"train": args.h0_train, "validation": args.h0_validation})
    for label, expected, actual in (
        ("bank_content_digest", args.expect_bank_content_digest, identity.bank_content_digest),
        ("h0 train sha256", args.expect_h0_train_sha256, identity.split_sha256["train"]),
        ("h0 validation sha256", args.expect_h0_validation_sha256, identity.split_sha256["validation"]),
    ):
        if expected and expected != actual:
            raise SystemExit(f"{label} mismatch: {actual}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    calibration_path = args.out_dir / "contact_calibration.json"
    save_contact_calibration(calibrate_contact_from_h0_train(bank, h0, identity), calibration_path)
    calibration = load_contact_calibration(calibration_path)
    docs52_match = None
    if args.docs52_report is not None:
        docs52 = json.loads(args.docs52_report.read_text())["contact_thresholds_T1_h0_train"]
        docs52_match = docs52 == calibration.thresholds
        if not docs52_match:
            raise SystemExit(f"T1 thresholds do not reproduce docs/52: {calibration.thresholds} vs {docs52}")

    sequences_dir = args.out_dir / "sequences"
    per_sequence, built = [], {}
    for sequence_id in sequence_ids_in_split(bank, "validation"):
        semantics = build_animation_semantics(bank, h0, identity, sequence_id, calibration)
        replay = build_animation_semantics(bank, h0, identity, sequence_id, calibration)
        path = sequences_dir / f"{sequence_id.replace(':', '__')}.semantics.json"
        save_animation_semantics(semantics, path)
        loaded = load_animation_semantics(path)
        entry = _account(semantics)
        entry["deterministic_replay"] = replay.content_digest() == semantics.content_digest()
        entry["round_trip_identity"] = (loaded == semantics
                                        and loaded.content_digest() == semantics.content_digest())
        entry["content_digest"] = semantics.content_digest()
        entry["file"] = str(path.relative_to(args.out_dir))
        per_sequence.append(entry)
        built[sequence_id] = semantics

    report = {
        "schema": "animcv_animation_semantics_replay_v1",
        "regime": bank.regime(),
        "h0_identity": {"bank_content_digest": identity.bank_content_digest,
                        "split_sha256": identity.split_sha256},
        "contact_calibration_file": calibration_path.name,
        "contact_calibration_reproduces_docs52_t1": docs52_match,
        "contact_calibration": calibration.to_dict(),
        "totals": _totals(per_sequence),
        "owner_cases": _owner_cases(built),
        "per_sequence": per_sequence,
    }
    (args.out_dir / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in ("regime", "contact_calibration_reproduces_docs52_t1", "totals")},
                     indent=2))
    print(json.dumps(report["owner_cases"], indent=2))


def _account(semantics: AnimationSemantics) -> dict[str, Any]:
    frames = semantics.frames
    known = [f for f in frames if f.root_orientation.known]
    entry: dict[str, Any] = {
        "sequence_id": semantics.sequence_id,
        "frame_count": len(frames),
        "root_orientation_known": len(known),
        "root_orientation_unknown": len(frames) - len(known),
        "root_orientation_held": sum(1 for f in known if f.root_orientation.yaw_held),
        "observation_invalid_frames": sum(1 for f in frames if not f.reliability.all_joints_valid),
        "observation_invalid_joints": sum(v is False for f in frames for v in f.reliability.joint_observation_valid),
        "out_of_frame_joints": sum(v is False for f in frames for v in (f.reliability.joint_in_frame or ())),
        "root_translation_unavailable": sum(f.root_translation.to_dict() == {"status": "unavailable"} for f in frames),
        "ground_height_unavailable": sum(f.ground_height.to_dict() == {"status": "unavailable"} for f in frames),
        "longest_held_run": max(_held_runs(semantics), default=0),
        "held_run_count": len(_held_runs(semantics)),
        "sampling_matches_calibration": semantics.provenance.contact["sampling_matches_calibration"],
        "median_row_stride_frames": semantics.provenance.contact["sequence_sampling"]["median_row_stride_frames"],
    }
    for side in SIDES:
        states = [getattr(f.foot_motion, side) for f in frames]
        entry[f"{side}_foot"] = {state.value: states.count(state) for state in ContactState}
        entry[f"{side}_alternations"] = _alternations(states)
    return entry


def _totals(per_sequence: list[dict[str, Any]]) -> dict[str, Any]:
    keys = ("frame_count", "root_orientation_known", "root_orientation_unknown", "root_orientation_held",
            "observation_invalid_frames", "observation_invalid_joints", "out_of_frame_joints",
            "root_translation_unavailable", "ground_height_unavailable")
    totals: dict[str, Any] = {key: sum(entry[key] for entry in per_sequence) for key in keys}
    for side in SIDES:
        totals[f"{side}_foot"] = {state.value: sum(e[f"{side}_foot"][state.value] for e in per_sequence)
                                  for state in ContactState}
    totals["sequence_count"] = len(per_sequence)
    totals["longest_held_run"] = max(e["longest_held_run"] for e in per_sequence)
    totals["held_run_count"] = sum(e["held_run_count"] for e in per_sequence)
    totals["mean_per_sequence_hold_rate"] = (
        sum(e["root_orientation_held"] / e["frame_count"] for e in per_sequence) / len(per_sequence))
    totals["all_deterministic_replay"] = all(e["deterministic_replay"] for e in per_sequence)
    totals["all_round_trip_identity"] = all(e["round_trip_identity"] for e in per_sequence)
    totals["sequences_sampling_matching_calibration"] = sum(e["sampling_matches_calibration"] for e in per_sequence)
    totals["validation_median_row_strides"] = sorted({e["median_row_stride_frames"] for e in per_sequence})
    return totals


def _alternations(states: list[ContactState]) -> int:
    scored = [s for s in states if s is not ContactState.UNKNOWN]
    return sum(1 for a, b in zip(scored, scored[1:]) if a is not b)


def _owner_cases(built: dict[str, AnimationSemantics]) -> dict[str, Any]:
    cases = dict(DOCS52_CASES)
    walking_id = max(built, key=lambda sid: (
        _alternations([f.foot_motion.left for f in built[sid].frames])
        + _alternations([f.foot_motion.right for f in built[sid].frames]), sid))
    cases["walking_contact_like_t1_selected"] = (walking_id, _first_transition(built[walking_id]))
    output = {}
    for category, (sequence_id, frame_index) in cases.items():
        semantics = built.get(sequence_id)
        rows = [i for i, f in enumerate(semantics.frames) if f.frame_index == frame_index] if semantics else []
        if not rows:
            output[category] = {"sequence_id": sequence_id, "frame_index": frame_index, "status": "not_in_semantics"}
            continue
        row = rows[0]
        window = semantics.frames[max(0, row - 2):row + 3]
        output[category] = {
            "sequence_id": sequence_id, "frame_index": frame_index,
            "sequence_alternations": {side: _alternations([getattr(f.foot_motion, side) for f in semantics.frames])
                                      for side in SIDES},
            "window": [_frame_summary(f) for f in window],
        }
    return output


def _first_transition(semantics: AnimationSemantics) -> int:
    """First frame whose foot state switches directly between CONTACT and MOVING."""
    decided = (ContactState.CONTACT, ContactState.MOVING)
    for previous, current in zip(semantics.frames, semantics.frames[1:]):
        for side in SIDES:
            a, b = getattr(previous.foot_motion, side), getattr(current.foot_motion, side)
            if a in decided and b in decided and a is not b:
                return current.frame_index
    return semantics.frames[len(semantics.frames) // 2].frame_index


def _held_runs(semantics: AnimationSemantics) -> list[int]:
    runs, length = [], 0
    for frame in semantics.frames:
        if frame.root_orientation.known and frame.root_orientation.yaw_held:
            length += 1
        elif length:
            runs.append(length)
            length = 0
    return runs + ([length] if length else [])


def _frame_summary(frame) -> dict[str, Any]:
    orientation = frame.root_orientation
    return {
        "frame_index": frame.frame_index,
        "timestamp": frame.timestamp,
        "root_orientation": ("unknown" if not orientation.known else
                             f"{math.degrees(orientation.yaw_radians):.1f}deg"
                             + (" held" if orientation.yaw_held else "")),
        "left": frame.foot_motion.left.value,
        "right": frame.foot_motion.right.value,
        "valid_joints": sum(frame.reliability.joint_observation_valid),
        "in_frame_joints": sum(frame.reliability.joint_in_frame or ()),
        "root_translation": frame.root_translation.to_dict()["status"],
        "ground_height": frame.ground_height.to_dict()["status"],
    }


if __name__ == "__main__":
    main()
