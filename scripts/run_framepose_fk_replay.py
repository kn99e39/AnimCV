#!/usr/bin/env python3
"""Replay current-policy semantics with Blender-rest FramePose FK on BaseRig."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

from motion.animation_semantics_v2 import load_animation_semantics_v2
from retarget.framepose_fk import load_rig_fk_calibration, save_fk_result, solve_framepose_fk
from retarget.framepose_target_rest import load_target_rest_pose
from rig.bone_mapping import load_bone_mapping_profile

CASES = {
    "known_good": ("3dpw:courtyard_dancing_00:actor0", 424),
    "turning": ("3dpw:outdoors_crosscountry_00:actor0", 202),
    "dynamic": ("3dpw:outdoors_crosscountry_00:actor0", 462),
    "tracking_loss": ("3dpw:courtyard_hug_00:actor1", 312),
    "invalid_endpoint": ("3dpw:courtyard_hug_00:actor1", 406),
}


def rotation_accounting(samples):
    known = [sample for sample in samples if sample.rotation_status == "known"]
    deltas = []
    sign_flips = 0
    previous = None
    for sample in samples:
        if sample.rotation_status != "known":
            previous = None
            continue
        if previous is not None:
            dot = sum(a * b for a, b in zip(previous.rotation_local, sample.rotation_local))
            sign_flips += dot < 0
            deltas.append(math.degrees(2 * math.acos(min(1.0, abs(dot)))))
        previous = sample
    ordered = sorted(deltas)
    def percentile(p):
        if not ordered:
            return None
        return ordered[round((len(ordered) - 1) * p)]
    return {
        "known_rotation_count": len(known),
        "unavailable_count": len(samples) - len(known),
        "unavailable_reasons": dict(Counter(s.reason for s in samples if s.reason)),
        "max_unit_norm_error": max((abs(sum(v*v for v in s.rotation_local) - 1)
                                    for s in known), default=None),
        "adjacent_known_pairs": len(deltas),
        "adjacent_delta_degrees": {"p50": percentile(.5), "p95": percentile(.95),
                                   "max": max(deltas) if deltas else None},
        "negative_adjacent_quaternion_dots": sign_flips,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rig", type=Path, required=True)
    parser.add_argument("--rest", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--semantics-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    rest = load_target_rest_pose(args.rest)
    if rest.provenance["source_fbx_sha256"] != hashlib.sha256(args.rig.read_bytes()).hexdigest():
        raise ValueError("Blender rest snapshot does not match FBX bytes")
    mapping = load_bone_mapping_profile(args.mapping)
    calibration = load_rig_fk_calibration(args.calibration)
    old_report = json.loads((args.semantics_dir / "report.json").read_text())
    files = {entry["sequence_id"]: entry for entry in old_report["per_sequence"]}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    report = {"schema": "animcv_framepose_fk_replay_v1",
              "regime": old_report["regime"], "h0_identity": old_report["h0_identity"],
              "rest_snapshot_digest": rest.digest(),
              "rest_snapshot_provenance": rest.provenance,
              "blender_root_bones": sorted(name for name, bone in rest.bones.items()
                                           if bone.parent is None),
              "root_orientation_target_bone": calibration.root_orientation_target_bone,
              "root_owner_scope": "mapped left-arm subtree; other Blender root bones do not inherit its yaw",
              "mapping_file": str(args.mapping), "calibration_file": str(args.calibration),
              "cases": {}, "sequences": {}}
    results = {}
    for sid in sorted({value[0] for value in CASES.values()}):
        entry = files[sid]
        semantics = load_animation_semantics_v2(args.semantics_dir / entry["file"])
        if semantics.content_digest() != entry["content_digest"]:
            raise ValueError(f"{sid}: persisted current-policy semantics digest changed")
        result = solve_framepose_fk(semantics, rest, mapping, calibration)
        result_path = args.out_dir / "sequences" / (sid.replace(":", "_") + ".fk.json")
        result_path.parent.mkdir(parents=True, exist_ok=True)
        save_fk_result(result, result_path)
        results[sid] = result
        report["sequences"][sid] = {
            "file": str(result_path.relative_to(args.out_dir)),
            "semantics_digest": semantics.content_digest(),
            "mapped_bones": {bone: rotation_accounting(
                [s for s in result.samples if s.target_bone == bone])
                for bone in (calibration.root_orientation_target_bone, *
                             [entry.target_bone for entry in mapping.entries])},
        }
    for label, (sid, frame_index) in CASES.items():
        result = results[sid]
        selected = {s.target_bone: s.to_dict() for s in result.samples if s.frame_index == frame_index}
        if len(selected) != 1 + len(mapping.entries):
            raise ValueError(f"case frame {sid}#{frame_index} missing")
        report["cases"][label] = {"sequence_id": sid, "frame_index": frame_index,
                                  "samples": selected}
    output = args.out_dir / "report.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps({"report": str(output), "rest_digest": rest.digest(),
                      "cases": report["cases"], "sequences": report["sequences"]}, indent=2))


if __name__ == "__main__":
    main()
