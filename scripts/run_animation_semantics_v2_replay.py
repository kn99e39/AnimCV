#!/usr/bin/env python3
"""Controlled frozen-H0 replay of current yaw and elapsed-time contact (validation)."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_root_motion_contact_h0_replay import _matched_oracle_sequence, _matched_world_frames, _confusion  # noqa: E402

from framepose.bank import load_bank  # noqa: E402
from motion.animation_semantics import load_animation_semantics  # noqa: E402
from motion.animation_semantics_v2 import (  # noqa: E402
    load_animation_semantics_v2, save_animation_semantics_v2, save_time_aware_calibration,
)
from motion.animation_semantics_v2_bridge import (  # noqa: E402
    build_animation_semantics_v2, calibrate_time_aware_contact_from_h0_train, current_root_orientation,
)
from pose.contact import ContactState  # noqa: E402
from pose.contact_reference import kinematic_proxy_contact_reference  # noqa: E402
from pose.contact_time_aware import TimeAwareThresholds, classify_time_aware_contact, elapsed_times  # noqa: E402
from pose.framepose_bridge import assemble_h0, build_h0_lifted_sequence, sequence_ids_in_split  # noqa: E402
from pose.root_orientation_diagnostic import observe_yaw  # noqa: E402
from pose.root_orientation_hold_attribution import circular_delta_degrees, distribution  # noqa: E402

CASES = {
    "known_good_pose": ("3dpw:courtyard_dancing_00:actor0", 424),
    "false_moving_suspected": ("3dpw:courtyard_drinking_00:actor1", 6),
    "ankle_failure": ("3dpw:outdoors_parcours_00:actor0", 384),
    "tracking_loss": ("3dpw:courtyard_hug_00:actor1", 312),
    "turning": ("3dpw:outdoors_crosscountry_00:actor0", 202),
    "dynamic": ("3dpw:outdoors_crosscountry_00:actor0", 462),
}


def _counts(states):
    return {state.value: states.count(state) for state in ContactState}


def _run_spans(states, times):
    spans = []
    start = None
    for i, state in enumerate(states + [ContactState.UNKNOWN]):
        if state is ContactState.CONTACT and start is None:
            start = i
        elif state is not ContactState.CONTACT and start is not None:
            spans.append(times[i - 1] - times[start])
            start = None
    return spans


def _summary(values):
    if not values:
        return {"count": 0, "mean": None, "median": None, "p90": None, "max": None}
    return {"count": len(values), "mean": float(np.mean(values)), "median": float(np.median(values)),
            "p90": float(np.percentile(values, 90)), "max": float(max(values))}


def _physical_world_speeds(positions, times):
    speeds = []
    for i, current in enumerate(positions):
        previous = positions[i - 1] if i else None
        following = positions[i + 1] if i + 1 < len(positions) else None
        if current is None:
            speeds.append(None)
        elif previous is not None and following is not None:
            speeds.append(float(np.linalg.norm(np.subtract(following, previous)) / (times[i + 1] - times[i - 1])))
        elif previous is not None:
            speeds.append(float(np.linalg.norm(np.subtract(current, previous)) / (times[i] - times[i - 1])))
        elif following is not None:
            speeds.append(float(np.linalg.norm(np.subtract(following, current)) / (times[i + 1] - times[i])))
        else:
            speeds.append(None)
    return speeds


def _metrics(candidate, reference, selector=None):
    if selector is not None:
        candidate = [value for value, selected in zip(candidate, selector) if selected]
        reference = [value for value, selected in zip(reference, selector) if selected]
    return _confusion(candidate, reference)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", required=True, type=Path)
    parser.add_argument("--h0-train", required=True, type=Path)
    parser.add_argument("--h0-validation", required=True, type=Path)
    parser.add_argument("--3dpw-raw", dest="threedpw_raw", required=True, type=Path)
    parser.add_argument("--legacy-semantics-dir", required=True, type=Path)
    parser.add_argument("--docs55-report", required=True, type=Path)
    parser.add_argument("--docs52-report", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()

    bank = load_bank(args.bank)
    h0, identity = assemble_h0(bank, {"train": args.h0_train, "validation": args.h0_validation})
    docs55 = json.loads(args.docs55_report.read_text())
    docs52 = json.loads(args.docs52_report.read_text())
    if identity.bank_content_digest != docs55["h0_identity"]["bank_content_digest"] or (
        identity.split_sha256["validation"] != docs55["h0_identity"]["h0_validation_sha256"]
        or identity.split_sha256 != docs52["h0_identity"]["split_sha256"]
    ):
        raise SystemExit("frozen H0 or bank differs from docs/52-55")
    legacy_report = json.loads((args.legacy_semantics_dir / "report.json").read_text())
    legacy_files = {entry["sequence_id"]: entry for entry in legacy_report["per_sequence"]}
    calibration = calibrate_time_aware_contact_from_h0_train(bank, h0, identity)
    reference_threshold = docs52["contact_reference_speed_threshold_m_s"]
    yaw_errors = []
    contact_rows = {side: [] for side in ("left", "right")}
    run_spans = {policy: {side: [] for side in ("left", "right")} for policy in ("historical_t1", "time_aware")}
    owner_cases = {}
    pending = {}
    for sid in sequence_ids_in_split(bank, "validation"):
        lifted = build_h0_lifted_sequence(bank, h0, sid)
        times = elapsed_times(lifted)
        current = current_root_orientation(lifted)
        oracle = _matched_oracle_sequence(sid, args.threedpw_raw, [frame.frame_index for frame in lifted.frames])
        oracle_yaw = observe_yaw(oracle)
        for observation, target in zip(current, oracle_yaw):
            if observation.known and target.yaw_radians is not None:
                yaw_errors.append(circular_delta_degrees(observation.yaw_radians, target.yaw_radians))
        legacy_entry = legacy_files[sid]
        legacy = load_animation_semantics(args.legacy_semantics_dir / legacy_entry["file"])
        if legacy.content_digest() != legacy_entry["content_digest"]:
            raise SystemExit(f"historical v1 digest mismatch for {sid}")
        world_frames = _matched_world_frames(sid, args.threedpw_raw, [frame.frame_index for frame in lifted.frames])
        for side in ("left", "right"):
            historical = [getattr(frame.foot_motion, side) for frame in legacy.frames]
            corrected = classify_time_aware_contact(lifted, side, TimeAwareThresholds(**calibration.thresholds[side]))
            world_positions = [getattr(frame, f"{side}_ankle_world") for frame in world_frames]
            reference = kinematic_proxy_contact_reference(world_positions, oracle.source_fps,
                                                           speed_threshold_m_s=reference_threshold)
            physical_speed = _physical_world_speeds(world_positions, times)
            for index, (old, new, ref, speed) in enumerate(zip(historical, corrected, reference, physical_speed)):
                contact_rows[side].append({"sequence_id": sid, "frame_index": lifted.frames[index].frame_index,
                                           "historical": old.value, "time_aware": new.value, "reference": ref.value,
                                           "reference_world_speed_m_s": speed})
            run_spans["historical_t1"][side].extend(_run_spans(historical, times))
            run_spans["time_aware"][side].extend(_run_spans(corrected, times))
        pending[sid] = lifted

    # A mandatory control: v2 producer must reproduce docs/55 H0-CURRENT before sidecars are written.
    yaw = distribution(yaw_errors)
    expected = docs55["oracle_error"]["all"]["H0_CURRENT"]
    for key in ("count", "mean", "median", "p90", "p95", "frac_gt_45", "frac_gt_90"):
        if not math.isclose(yaw[key], expected[key], rel_tol=0, abs_tol=1e-8):
            raise SystemExit(f"docs/55 yaw control mismatch at {key}: {yaw[key]} vs {expected[key]}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    save_time_aware_calibration(calibration, args.out_dir / "contact_calibration.json")
    per_sequence = []
    for sid in pending:
        semantics = build_animation_semantics_v2(bank, h0, identity, sid, calibration)
        repeat = build_animation_semantics_v2(bank, h0, identity, sid, calibration)
        path = args.out_dir / "sequences" / f"{sid.replace(':', '__')}.semantics.json"
        save_animation_semantics_v2(semantics, path)
        if semantics.content_digest() != repeat.content_digest() or load_animation_semantics_v2(path) != semantics:
            raise SystemExit(f"non-deterministic or non-roundtrip v2 sidecar: {sid}")
        per_sequence.append({"sequence_id": sid, "file": str(path.relative_to(args.out_dir)),
                             "content_digest": semantics.content_digest()})
        for label, (case_sid, frame_index) in CASES.items():
            if case_sid == sid:
                matching = [frame for frame in semantics.frames if frame.frame_index == frame_index]
                if matching:
                    frame = matching[0]
                    owner_cases[label] = {"sequence_id": sid, "frame_index": frame_index,
                                          "root_orientation": frame.root_orientation.to_dict(),
                                          "foot_motion": frame.foot_motion.to_dict(),
                                          "valid_joints": sum(frame.reliability.joint_observation_valid)}

    comparison = {}
    for side, rows in contact_rows.items():
        old = [ContactState(row["historical"]) for row in rows]
        new = [ContactState(row["time_aware"]) for row in rows]
        ref = [ContactState(row["reference"]) for row in rows]
        stationary = [row["reference_world_speed_m_s"] is not None and
                      row["reference_world_speed_m_s"] <= 0.5 * reference_threshold for row in rows]
        comparison[side] = {
            "historical_t1": {"counts": _counts(old), "metrics_vs_kinematic_proxy": _metrics(old, ref),
                              "contact_run_span_seconds": _summary(run_spans["historical_t1"][side])},
            "time_aware": {"counts": _counts(new), "metrics_vs_kinematic_proxy": _metrics(new, ref),
                           "contact_run_span_seconds": _summary(run_spans["time_aware"][side])},
            "state_disagreement": {"count": sum(a is not b for a, b in zip(old, new)),
                                   "rate": sum(a is not b for a, b in zip(old, new)) / len(rows),
                                   "transitions": dict(Counter(f"{a.value}->{b.value}" for a, b in zip(old, new) if a is not b))},
            "stationary_subset": {"definition": "GT world ankle speed on actual elapsed time <= half the fixed docs/52 TRAIN reference threshold",
                                  "threshold_m_s": 0.5 * reference_threshold, "row_count": sum(stationary),
                                  "historical_counts": _counts([v for v, chosen in zip(old, stationary) if chosen]),
                                  "time_aware_counts": _counts([v for v, chosen in zip(new, stationary) if chosen]),
                                  "historical_metrics": _metrics(old, ref, stationary),
                                  "time_aware_metrics": _metrics(new, ref, stationary)},
        }
    report = {"schema": "animcv_animation_semantics_v2_replay_v1", "regime": bank.regime(),
              "reference_kind": "KINEMATIC_PROXY (evaluation only)",
              "h0_identity": {"bank_content_digest": identity.bank_content_digest,
                              "split_sha256": identity.split_sha256},
              "yaw_docs55_reproduced": True, "yaw_oracle_error_degrees": yaw,
              "root_orientation_known": len(yaw_errors),
              "root_orientation_unknown": docs55["frame_count"] - len(yaw_errors),
              "calibration": calibration.to_dict(), "contact_comparison": comparison,
              "owner_cases": owner_cases, "per_sequence": per_sequence}
    (args.out_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    with (args.out_dir / "contact_rows.jsonl").open("w") as stream:
        for side, rows in contact_rows.items():
            for row in rows:
                stream.write(json.dumps({"side": side, **row}) + "\n")
    print(json.dumps({"yaw": yaw, "contact": comparison, "owner_cases": owner_cases}, indent=2))


if __name__ == "__main__":
    main()
