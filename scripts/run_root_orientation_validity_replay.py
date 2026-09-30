#!/usr/bin/env python3
"""Frozen-H0 validity control for the corrected v2 Root Orientation producer."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_root_motion_contact_h0_replay import _matched_oracle_sequence  # noqa: E402

from framepose.bank import load_bank  # noqa: E402
from motion.animation_semantics_v2 import (  # noqa: E402
    LEGACY_ROOT_POLICY_V2, ROOT_POLICY_V2, load_animation_semantics_v2,
    load_time_aware_calibration, save_animation_semantics_v2,
)
from motion.animation_semantics_v2_bridge import (  # noqa: E402
    build_animation_semantics_v2, valid_bilateral_sources,
)
from pose.framepose_bridge import assemble_h0, build_h0_lifted_sequence  # noqa: E402
from pose.root_orientation_diagnostic import observe_yaw  # noqa: E402
from pose.root_orientation_hold_attribution import circular_delta_degrees, distribution  # noqa: E402


def build_report(bank, h0, identity, old_dir: Path, raw_3dpw: Path, out_dir: Path) -> dict:
    old_report = json.loads((old_dir / "report.json").read_text())
    if old_report["h0_identity"] != {"bank_content_digest": identity.bank_content_digest,
                                       "split_sha256": identity.split_sha256}:
        raise ValueError("docs/56 bank or H0 identity changed")
    calibration = load_time_aware_calibration(old_dir / "contact_calibration.json")
    if calibration.to_dict() != old_report["calibration"]:
        raise ValueError("docs/56 contact calibration changed")
    (out_dir / "contact_calibration.json").write_bytes((old_dir / "contact_calibration.json").read_bytes())

    categories = {"both_valid": [], "one_valid": [], "neither_valid": []}
    per_sequence = []
    owner_cases = {}
    for old_entry in old_report["per_sequence"]:
        sid = old_entry["sequence_id"]
        old = load_animation_semantics_v2(old_dir / old_entry["file"], allow_legacy_policy=True)
        if old.content_digest() != old_entry["content_digest"] or (
            old.provenance.root_orientation["policy"] != LEGACY_ROOT_POLICY_V2
        ):
            raise ValueError(f"{sid}: persisted docs/56 v2 sidecar changed")
        lifted = build_h0_lifted_sequence(bank, h0, sid)
        oracle = _matched_oracle_sequence(sid, raw_3dpw, [f.frame_index for f in lifted.frames])
        oracle_yaw = observe_yaw(oracle)
        diagnostic = observe_yaw(lifted)
        new = build_animation_semantics_v2(bank, h0, identity, sid, calibration)
        replay = build_animation_semantics_v2(bank, h0, identity, sid, calibration)
        if new.content_digest() != replay.content_digest():
            raise ValueError(f"{sid}: new producer is not deterministic")
        if old.provenance.frame_pose != new.provenance.frame_pose or (
            old.provenance.contact != new.provenance.contact or old.provenance.reliability != new.provenance.reliability
        ):
            raise ValueError(f"{sid}: a non-orientation provenance owner changed")
        for source, previous, corrected, target, diag in zip(
            lifted.frames, old.frames, new.frames, oracle_yaw, diagnostic
        ):
            if previous.frame_index != corrected.frame_index or previous.timestamp != corrected.timestamp:
                raise ValueError("evaluation row changed")
            if previous.articulation != corrected.articulation or previous.foot_motion != corrected.foot_motion or (
                previous.reliability != corrected.reliability or
                previous.root_translation != corrected.root_translation or previous.ground_height != corrected.ground_height
            ):
                raise ValueError("a non-orientation semantic field changed")
            if previous.root_orientation.yaw_radians != diag.yaw_radians:
                raise ValueError("docs/56 sidecar does not match diagnostic yaw")
            sources = valid_bilateral_sources(source)
            category = ("both_valid" if len(sources) == 2 else "one_valid" if sources else "neither_valid")
            before = previous.root_orientation.yaw_radians
            after = corrected.root_orientation.yaw_radians
            oracle_value = target.yaw_radians
            row = {"sequence_id": sid, "frame_index": source.frame_index,
                   "valid_sources": list(sources), "old_yaw_radians": before,
                   "corrected_yaw_radians": after, "oracle_yaw_radians": oracle_value,
                   "old_to_corrected_delta_degrees": (
                       circular_delta_degrees(before, after) if before is not None and after is not None else None),
                   "old_oracle_error_degrees": (
                       circular_delta_degrees(before, oracle_value) if before is not None and oracle_value is not None else None),
                   "corrected_oracle_error_degrees": (
                       circular_delta_degrees(after, oracle_value) if after is not None and oracle_value is not None else None)}
            if category == "both_valid" and before != after:
                raise ValueError(f"{sid}#{source.frame_index}: fully-valid yaw differs from docs/56")
            if category == "neither_valid" and corrected.root_orientation.known:
                raise ValueError(f"{sid}#{source.frame_index}: invalid yaw is KNOWN")
            categories[category].append(row)

        path = out_dir / "sequences" / f"{sid.replace(':', '__')}.semantics.json"
        save_animation_semantics_v2(new, path)
        if load_animation_semantics_v2(path) != new:
            raise ValueError(f"{sid}: serialization round trip failed")
        per_sequence.append({"sequence_id": sid, "file": str(path.relative_to(out_dir)),
                             "content_digest": new.content_digest()})
        for label, case in old_report.get("owner_cases", {}).items():
            if case["sequence_id"] == sid:
                matches = [frame for frame in new.frames if frame.frame_index == case["frame_index"]]
                if matches:
                    frame = matches[0]
                    owner_cases[label] = {"sequence_id": sid, "frame_index": frame.frame_index,
                                          "root_orientation": frame.root_orientation.to_dict(),
                                          "foot_motion": frame.foot_motion.to_dict(),
                                          "valid_joints": sum(frame.reliability.joint_observation_valid)}

    def summarize(rows):
        return {"row_count": len(rows),
                "old_to_corrected_delta_degrees": distribution([r["old_to_corrected_delta_degrees"] for r in rows]),
                "old_oracle_error_degrees": distribution([r["old_oracle_error_degrees"] for r in rows]),
                "corrected_oracle_error_degrees": distribution([r["corrected_oracle_error_degrees"] for r in rows]),
                "corrected_unknown": sum(r["corrected_yaw_radians"] is None for r in rows)}

    report = {"schema": "animcv_root_orientation_validity_replay_v1",
              "regime": bank.regime(),
              "h0_identity": old_report["h0_identity"],
              "policy_before": LEGACY_ROOT_POLICY_V2,
              "policy_after": ROOT_POLICY_V2,
              "schema_before_after": "animcv_animation_semantics_v2",
              "both_valid_numerically_identical": True,
              "contact_and_other_fields_identical": True,
              "categories": {name: summarize(rows) for name, rows in categories.items()},
              "one_valid_rows": categories["one_valid"],
              "neither_valid_rows": categories["neither_valid"],
              "owner_cases": owner_cases,
              "per_sequence": per_sequence}
    if sum(len(rows) for rows in categories.values()) != old_report["root_orientation_known"] + old_report["root_orientation_unknown"]:
        raise ValueError("validity accounting does not cover the docs/56 rows")
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--h0-train", type=Path, required=True)
    parser.add_argument("--h0-validation", type=Path, required=True)
    parser.add_argument("--3dpw-raw", dest="raw_3dpw", type=Path, required=True)
    parser.add_argument("--docs56-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    bank = load_bank(args.bank)
    h0, identity = assemble_h0(bank, {"train": args.h0_train, "validation": args.h0_validation})
    args.out_dir.mkdir(parents=True, exist_ok=True)
    report = build_report(bank, h0, identity, args.docs56_dir, args.raw_3dpw, args.out_dir)
    print(json.dumps({"policy_before": report["policy_before"], "policy_after": report["policy_after"],
                      "categories": report["categories"]}, indent=2))


if __name__ == "__main__":
    main()
