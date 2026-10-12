#!/usr/bin/env python3
"""EVALUATION ONLY (Worklog 74): export review geometry for the visual-direction IK handoff.

Cases: every Worklog 62 owner case that is a matched row (unavailable ones are
listed as refused), plus three rule-based rows: the largest P2-vs-P0 endpoint
gain, the largest P2-vs-P0 endpoint loss, and the out-of-plane (|GT f| >= 0.5)
row with the median P0 endpoint error.  Overlapping picks keep one entry and
record every label.  All points are heading-relative imported-armature space
(A * Rz(-yaw_H0)); P0 is checked against the persisted Worklog 61 solve.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from retarget.framepose_fk import derive_rig_alignment, load_rig_fk_calibration
from retarget.framepose_object_owner import load_object_owned_fk
from retarget.framepose_target_rest import load_target_rest_pose
from retarget.framepose_two_bone_ik import (
    chain_forward_kinematics, load_ik_chain_set, load_two_bone_ik, target_chain_geometry,
)

NAMES = {"p0_h0_ik": "P0", "p1_e0_visual_direction": "P1", "p2_e1_visual_direction": "P2",
         "p3_gt_forward_fraction": "P3"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("rig", "rest", "calibration", "chains", "fk60_dir", "ik61_dir", "handoff_dir", "out"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    args = parser.parse_args()
    rest = load_target_rest_pose(args.rest)
    if rest.provenance["source_fbx_sha256"] != hashlib.sha256(args.rig.read_bytes()).hexdigest():
        raise ValueError("Blender rest snapshot differs from current FBX bytes")
    alignment = derive_rig_alignment(rest, load_rig_fk_calibration(args.calibration))
    target = target_chain_geometry(rest, load_ik_chain_set(args.chains).chains[0])
    report = json.loads((args.handoff_dir / "report.json").read_text())
    rows = json.loads((args.handoff_dir / "rows.json").read_text())
    row_by = {(r["sequence_id"], r["frame_index"]): r for r in rows}
    fk60 = json.loads((args.fk60_dir / "report.json").read_text())
    ik61 = json.loads((args.ik61_dir / "report.json").read_text())

    picks: dict[tuple, list[str]] = {}
    refused = {}
    for label, case in report["owner_cases"].items():
        key = (case["sequence_id"], case["frame_index"])
        if key in row_by:
            picks.setdefault(key, []).append(label)
        else:
            refused[label] = {"sequence_id": key[0], "frame_index": key[1], "reason": case["unavailable"]}

    def delta(r):
        return (r["candidates"]["p2_e1_visual_direction"]["solved_endpoint_error"]
                - r["candidates"]["p0_h0_ik"]["solved_endpoint_error"])
    rules = {"largest_p2_gain_vs_p0": min(rows, key=lambda r: (delta(r), r["sequence_id"], r["frame_index"])),
             "largest_p2_loss_vs_p0": max(rows, key=lambda r: (delta(r), r["sequence_id"], r["frame_index"]))}
    oop = sorted((r for r in rows if r["abs_gt_f"] >= 0.5),
                 key=lambda r: (r["candidates"]["p0_h0_ik"]["solved_endpoint_error"], r["sequence_id"], r["frame_index"]))
    rules["median_out_of_plane"] = oop[len(oop) // 2]
    for label, r in rules.items():
        picks.setdefault((r["sequence_id"], r["frame_index"]), []).append(label)

    loaded = {}
    out = {"schema": "animcv_framepose_visual_ik_review_cases_v1", "space": "heading_relative_armature",
           "shared_transform": "A*Rz(-yaw_H0) on every direction; GT yaw unused",
           "rest_snapshot_digest": rest.digest(), "target_chain": target.to_dict(),
           "canonical_heading_zero_to_armature_xyzw": list(alignment), "cases": {}, "refused": refused}
    checks = {"p0_vs_w61_end_max": 0.0, "p0_vs_w61_mid_max": 0.0}
    for key, labels in sorted(picks.items()):
        sid, frame_index = key
        if sid not in loaded:
            fk = load_object_owned_fk(args.fk60_dir / fk60["sequences"][sid]["file"])
            ik = load_two_bone_ik(args.ik61_dir / ik61["sequences"][sid]["ik_file"])
            loaded[sid] = (fk, {s.frame_index: s for s in ik.samples})
        fk, ik_by = loaded[sid]
        r = row_by[key]
        g = r["geometry"]
        bones = {s.target_bone: s for s in fk.pose_bone_samples if s.frame_index == frame_index}
        rig = next(s for s in fk.rig_rotations if s.frame_index == frame_index)
        fk_mid, fk_end = chain_forward_kinematics(rest, target, bones[target.proximal].rotation_local,
                                                  bones[target.distal].rotation_local)
        w61 = ik_by[frame_index]
        checks["p0_vs_w61_end_max"] = max(checks["p0_vs_w61_end_max"],
                                          math.dist(g["p0_h0_ik"]["solved_endpoint"], w61.solved_endpoint))
        checks["p0_vs_w61_mid_max"] = max(checks["p0_vs_w61_mid_max"],
                                          math.dist(g["p0_h0_ik"]["solved_mid"], w61.solved_mid))
        label = "+".join(labels)
        out["cases"][label] = {
            "labels": labels, "sequence_id": sid, "frame_index": frame_index, "root": list(target.root_head),
            "fk_mid": list(fk_mid), "fk_end": list(fk_end), "e_gt": r["e_gt"], "oracle_mid": r["oracle_mid"],
            "chains": {NAMES[n]: {"mid": g[n]["solved_mid"], "end": g[n]["solved_endpoint"],
                                  "requested": g[n]["requested_endpoint"]} for n in NAMES},
            "f": r["f"],
            "numbers": {NAMES[n]: {m: r["candidates"][n][m] for m in ("solved_endpoint_error", "direction_error_degrees",
                                                                       "elbow_error", "side_agrees_with_gt")}
                        for n in NAMES},
            "rig_rotation_armature_space_xyzw": list(rig.rotation_armature_space),
            "pose_bone_local_xyzw": {name: list(s.rotation_local) for name, s in bones.items()
                                     if s.rotation_status == "known" and s.rotation_local is not None},
        }
    out["sanity_checks_cm"] = checks
    if any(v > 1e-9 for v in checks.values()):
        raise ValueError(f"sanity check failed: {checks}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1))
    print(json.dumps({"cases": list(out["cases"]), "refused": refused, "checks": checks}, indent=1))


if __name__ == "__main__":
    main()
