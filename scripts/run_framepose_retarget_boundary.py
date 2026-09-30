#!/usr/bin/env python3
"""Audit BaseRig and replay direction-only v2 semantics without guessed rig alignment."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

from motion.animation_semantics_v2 import load_animation_semantics_v2
from retarget.framepose_direction_candidate import (
    audit_direction_mapping, save_direction_candidate, solve_framepose_direction_candidate,
)
from rig.bone_mapping import load_bone_mapping_profile
from rig.rig_parser import RigParser
from rig.rig_profile import load_rig_profile

CASES = {
    "known_good": ("3dpw:courtyard_dancing_00:actor0", 424),
    "turning": ("3dpw:outdoors_crosscountry_00:actor0", 202),
    "dynamic": ("3dpw:outdoors_crosscountry_00:actor0", 462),
    "tracking_loss": ("3dpw:courtyard_hug_00:actor1", 312),
    "tracking_loss_invalid_endpoint": ("3dpw:courtyard_hug_00:actor1", 406),
}


def _continuity(samples):
    previous = None
    evaluated, negative_dot, max_delta = 0, 0, 0.0
    unavailable = 0
    for sample in samples:
        if sample.rotation_status != "known":
            unavailable += 1
            previous = None
            continue
        if previous is not None:
            dot = sum(a * b for a, b in zip(previous.rotation_local, sample.rotation_local))
            evaluated += 1
            negative_dot += dot < 0
            max_delta = max(max_delta, math.degrees(2 * math.acos(min(1.0, abs(dot)))))
        previous = sample
    return {"adjacent_known_pairs": evaluated, "negative_quaternion_dots": negative_dot,
            "max_adjacent_rotation_degrees": max_delta if evaluated else None,
            "unavailable_rows": unavailable,
            "status": "evaluated" if evaluated else "not_evaluable_without_alignment_or_valid_pairs"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rig", type=Path, required=True)
    parser.add_argument("--rig-profile", type=Path,
                        help="RigParser output from this FBX when the execution container lacks pyassimp")
    parser.add_argument("--demo-mapping", type=Path, required=True)
    parser.add_argument("--real-mapping", type=Path, required=True)
    parser.add_argument("--semantics-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    rig = load_rig_profile(args.rig_profile) if args.rig_profile else RigParser().load(str(args.rig))
    if Path(rig.source_path).name != args.rig.name:
        raise ValueError("RigProfile source does not name the requested FBX")
    demo = load_bone_mapping_profile(args.demo_mapping)
    real = load_bone_mapping_profile(args.real_mapping)
    old_report = json.loads((args.semantics_dir / "report.json").read_text())
    files = {entry["sequence_id"]: entry for entry in old_report["per_sequence"]}
    if real.rig_id != rig.rig_id:
        raise ValueError("real mapping does not match parsed rig")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "schema": "animcv_framepose_retarget_boundary_replay_v1",
        "regime": old_report["regime"],
        "semantics_h0_identity": old_report["h0_identity"],
        "semantics_policy": "framepose_current_valid_bilateral_v1",
        "rig": {"rig_id": rig.rig_id, "source_path": str(args.rig), "bone_count": len(rig.bones),
                "root_bone": rig.root_bone, "parser": rig.metadata.get("parser"),
                "profile_input": str(args.rig_profile) if args.rig_profile else "RigParser.load",
                "fbx_sha256": hashlib.sha256(args.rig.read_bytes()).hexdigest(),
                "profile_digest": hashlib.sha256(json.dumps(
                    rig.to_dict(), sort_keys=True, separators=(",", ":")).encode()).hexdigest()},
        "demo_mapping": {"source_path": str(args.demo_mapping),
                         "entries": [audit_direction_mapping(rig, entry) for entry in demo.entries]},
        "existing_matching_mapping": {"source_path": str(args.real_mapping),
                                      "entries": [audit_direction_mapping(rig, entry) for entry in real.entries]},
        "canonical_to_rig_rest_alignment": {"status": "contract_gap", "value": None,
                                             "owner": "retarget session; absent from RigProfile and BoneMappingProfile"},
        "root_orientation": "separate semantic owner; not applied to limb directions",
        "cases": {},
    }
    for label, (sid, frame_index) in CASES.items():
        entry = files[sid]
        semantics = load_animation_semantics_v2(args.semantics_dir / entry["file"])
        if semantics.content_digest() != entry["content_digest"]:
            raise ValueError(f"{sid}: persisted current-policy semantics digest changed")
        result = solve_framepose_direction_candidate(semantics, rig, real, canonical_to_rig_rest=None)
        case_path = args.out_dir / "sequences" / f"{label}.direction.json"
        save_direction_candidate(result, case_path)
        mappings = {}
        for target in {mapping.target_bone for mapping in real.entries if mapping.mapping_mode == "direction"}:
            rows = [sample for sample in result.samples if sample.target_bone == target]
            selected = next((sample for sample in rows if sample.frame_index == frame_index), None)
            if selected is None:
                raise ValueError(f"{sid}#{frame_index} absent from semantics")
            mappings[target] = {
                "source_pair": list(selected.source_pair),
                "evaluated_frame_count": len(rows),
                "known_source_count": sum(row.source_status == "known" for row in rows),
                "known_rotation_count": sum(row.rotation_status == "known" for row in rows),
                "unavailable_reasons": dict(Counter(row.reason for row in rows if row.reason)),
                "case_frame": selected.to_dict(),
                "quaternion_continuity": _continuity(rows),
            }
        report["cases"][label] = {"sequence_id": sid, "frame_index": frame_index,
                                  "full_sequence_file": str(case_path.relative_to(args.out_dir)),
                                  "mappings": mappings}
    (args.out_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps({"rig": report["rig"], "demo_mapping": report["demo_mapping"],
                      "existing_matching_mapping": report["existing_matching_mapping"],
                      "case_frames": {label: case["mappings"] for label, case in report["cases"].items()}},
                     indent=2))


if __name__ == "__main__":
    main()
