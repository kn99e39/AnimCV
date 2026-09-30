#!/usr/bin/env python3
"""Replay current-policy v2 with Armature Object yaw and unchanged limb FK."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from motion.animation_semantics_v2 import load_animation_semantics_v2
from retarget.framepose_fk import load_fk_result, load_rig_fk_calibration
from retarget.framepose_object_owner import save_object_owned_fk, solve_object_owned_fk
from retarget.framepose_target_rest import load_target_rest_pose
from rig.bone_mapping import load_bone_mapping_profile

CASES = {
    "known_good": ("3dpw:courtyard_dancing_00:actor0", 424),
    "turning_invalid": ("3dpw:outdoors_crosscountry_00:actor0", 202),
    "turning_review": ("3dpw:outdoors_crosscountry_00:actor0", 211),
    "dynamic": ("3dpw:outdoors_crosscountry_00:actor0", 462),
    "tracking_loss": ("3dpw:courtyard_hug_00:actor1", 312),
    "invalid_endpoint": ("3dpw:courtyard_hug_00:actor1", 406),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rig", type=Path, required=True)
    parser.add_argument("--rest", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--semantics-dir", type=Path, required=True)
    parser.add_argument("--fk59-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    rest = load_target_rest_pose(args.rest)
    if rest.provenance["source_fbx_sha256"] != hashlib.sha256(args.rig.read_bytes()).hexdigest():
        raise ValueError("Blender rest snapshot differs from current FBX bytes")
    mapping = load_bone_mapping_profile(args.mapping)
    calibration = load_rig_fk_calibration(args.calibration)
    source_report = json.loads((args.semantics_dir / "report.json").read_text())
    old_report = json.loads((args.fk59_dir / "report.json").read_text())
    if old_report["rest_snapshot_digest"] != rest.digest():
        raise ValueError("Worklog 59 used a different rest snapshot")
    source_files = {row["sequence_id"]: row for row in source_report["per_sequence"]}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    report = {"schema": "animcv_framepose_object_owner_replay_v1",
              "regime": source_report["regime"],
              "h0_identity": source_report["h0_identity"],
              "rest_snapshot_digest": rest.digest(),
              "imported_top_level_bones": sorted(name for name, bone in rest.bones.items()
                                                 if bone.parent is None),
              "owner_change": "spine_01 pose bone -> Blender Armature Object",
              "sequences": {}, "cases": {}}
    results = {}
    for sid in sorted({sid for sid, _ in CASES.values()}):
        source = source_files[sid]
        semantics = load_animation_semantics_v2(args.semantics_dir / source["file"])
        if semantics.content_digest() != source["content_digest"]:
            raise ValueError(f"{sid}: persisted semantics digest changed")
        result = solve_object_owned_fk(semantics, rest, mapping, calibration)
        old_path = args.fk59_dir / old_report["sequences"][sid]["file"]
        old = load_fk_result(old_path)
        original_limbs = [sample.to_dict() for sample in old.samples
                          if sample.target_bone != calibration.root_orientation_target_bone]
        current_limbs = [sample.to_dict() for sample in result.pose_bone_samples]
        if current_limbs != original_limbs:
            raise ValueError(f"{sid}: local limb FK differs from Worklog 59")
        relative_path = Path("sequences") / (sid.replace(":", "_") + ".object_fk.json")
        output_path = args.out_dir / relative_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        save_object_owned_fk(result, output_path)
        results[sid] = result
        report["sequences"][sid] = {
            "file": str(relative_path), "source_semantics_digest": semantics.content_digest(),
            "worklog59_limb_rows_exact_match": len(current_limbs),
            "root_known": sum(s.rotation_status == "known" for s in result.rig_rotations),
            "root_unavailable": sum(s.rotation_status == "unavailable" for s in result.rig_rotations),
            "root_unavailable_reasons": dict(Counter(s.reason for s in result.rig_rotations if s.reason)),
            "limbs": {bone: {"known": sum(s.rotation_status == "known" for s in result.pose_bone_samples
                                              if s.target_bone == bone),
                             "unavailable": sum(s.rotation_status == "unavailable" for s in result.pose_bone_samples
                                                if s.target_bone == bone),
                             "unavailable_reasons": dict(Counter(s.reason for s in result.pose_bone_samples
                                                                 if s.target_bone == bone and s.reason))}
                      for bone in sorted({entry.target_bone for entry in mapping.entries})},
        }
    for label, (sid, frame_index) in CASES.items():
        result = results[sid]
        root = next((s for s in result.rig_rotations if s.frame_index == frame_index), None)
        limbs = {s.target_bone: s.to_dict() for s in result.pose_bone_samples
                 if s.frame_index == frame_index}
        if root is None or len(limbs) != len(mapping.entries):
            raise ValueError(f"{label}: case frame missing")
        report["cases"][label] = {"sequence_id": sid, "frame_index": frame_index,
                                  "rig_rotation": root.to_dict(), "pose_bones": limbs}
    output = args.out_dir / "report.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps({"report": str(output), "sequences": report["sequences"],
                      "cases": report["cases"]}, indent=2))


if __name__ == "__main__":
    main()
