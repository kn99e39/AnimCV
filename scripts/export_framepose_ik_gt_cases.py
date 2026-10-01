#!/usr/bin/env python3
"""EVALUATION ONLY: export Worklog 62 owner-case arm geometry for a Blender review.

Every point is in heading-relative imported-armature space (the shared runtime
transform A*Rz(-yaw_H0) is applied to H0 and GT alike; GT yaw is never used).
The output is a review sidecar only; nothing is fed back into runtime.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_root_motion_contact_h0_replay import _matched_oracle_sequence  # noqa: E402

from motion.animation_semantics_v2 import load_animation_semantics_v2  # noqa: E402
from retarget.framepose_fk import derive_rig_alignment, load_rig_fk_calibration  # noqa: E402
from retarget.framepose_ik_gt_attribution import (  # noqa: E402
    _runtime_side, attribute_frame, measure_chain,
)
from retarget.framepose_object_owner import load_object_owned_fk  # noqa: E402
from retarget.framepose_target_rest import load_target_rest_pose  # noqa: E402
from retarget.framepose_two_bone_ik import (  # noqa: E402
    chain_forward_kinematics, load_ik_chain_set, load_two_bone_ik, solve_two_bone,
    target_chain_geometry,
)

DANCING = "3dpw:courtyard_dancing_00:actor0"
CROSS = "3dpw:outdoors_crosscountry_00:actor0"
HUG = "3dpw:courtyard_hug_00:actor1"
CASES = (
    ("known_good", DANCING, 424),
    ("dynamic", CROSS, 462),
    ("turning_review", CROSS, 211),
    ("tracking_loss", HUG, 312),
    ("near_straight", CROSS, 525),
    ("strongly_bent_and_largest_reach_error", CROSS, 173),
    ("largest_ratio_error", HUG, 397),
    ("largest_direction_error", DANCING, 176),
    ("turning_invalid", CROSS, 202),
    ("invalid_endpoint", HUG, 406),
)
NUMBERS = ("fk_h0_endpoint_error", "ik_h0_endpoint_error", "h0_direction_gt_reach_endpoint_error",
           "gt_direction_h0_reach_endpoint_error", "oracle_ik_endpoint_error",
           "endpoint_direction_error_degrees", "bend_plane_error_degrees", "bend_side_agrees",
           "gt_ratio", "h0_ratio", "gt_reach_fraction", "h0_reach_fraction",
           "gt_bend_angle_degrees", "h0_bend_angle_degrees")


def _normalized_chain(root, measure, total):
    s = measure.upper_length + measure.lower_length
    mid = [root[i] + total * measure.upper[i] / s for i in range(3)]
    end = [root[i] + total * (measure.upper[i] + measure.lower[i]) / s for i in range(3)]
    return [list(root), mid, end]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("rig", "rest", "calibration", "chains", "semantics_dir", "fk60_dir",
                 "ik61_dir", "rows", "threedpw_raw", "out"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    args = parser.parse_args()

    rest = load_target_rest_pose(args.rest)
    if rest.provenance["source_fbx_sha256"] != hashlib.sha256(args.rig.read_bytes()).hexdigest():
        raise ValueError("Blender rest snapshot differs from current FBX bytes")
    alignment = derive_rig_alignment(rest, load_rig_fk_calibration(args.calibration))
    chain = load_ik_chain_set(args.chains).chains[0]
    target = target_chain_geometry(rest, chain)
    total = target.max_reach
    source_report = json.loads((args.semantics_dir / "report.json").read_text())
    fk60_report = json.loads((args.fk60_dir / "report.json").read_text())
    ik61_report = json.loads((args.ik61_dir / "report.json").read_text())
    source_files = {row["sequence_id"]: row for row in source_report["per_sequence"]}
    rows = {(r["sequence_id"], r["frame_index"]): r for r in json.loads(args.rows.read_text())}
    names = (chain.source_root, chain.source_mid, chain.source_end)

    loaded = {}

    def load(sid):
        if sid not in loaded:
            semantics = load_animation_semantics_v2(args.semantics_dir / source_files[sid]["file"])
            if semantics.content_digest() != source_files[sid]["content_digest"]:
                raise ValueError(f"{sid}: semantics digest changed")
            ik = load_two_bone_ik(args.ik61_dir / ik61_report["sequences"][sid]["ik_file"])
            if ik.provenance["semantics_digest"] != semantics.content_digest():
                raise ValueError(f"{sid}: IK built from other semantics")
            fk = load_object_owned_fk(args.fk60_dir / fk60_report["sequences"][sid]["file"])
            indices = [f.frame_index for f in semantics.frames]
            oracle = _matched_oracle_sequence(sid, args.threedpw_raw, indices)  # GT: evaluation only
            loaded[sid] = (semantics, ik, fk, oracle)
        return loaded[sid]

    out = {"schema": "animcv_framepose_ik_gt_review_cases_v1", "space": "heading_relative_armature",
           "shared_transform": "A*Rz(-yaw_H0) applied to H0 and GT alike; GT yaw unused",
           "rest_snapshot_digest": rest.digest(), "target_chain": target.to_dict(),
           "canonical_heading_zero_to_armature_xyzw": list(alignment),
           "cases": {}, "refused": {}}
    checks = {"oracle_end_vs_e_gt_max": 0.0, "ik_end_vs_w61_solved_max": 0.0,
              "ik_mid_vs_w61_solved_max": 0.0, "fk_end_vs_w61_comparison_max": 0.0,
              "normalized_h0_end_vs_ik_end_max": 0.0, "normalized_gt_end_vs_e_gt_max": 0.0,
              "rows_json_number_max_abs_diff": 0.0}
    for label, sid, frame_index in CASES:
        semantics, ik, fk, oracle = load(sid)
        sample = next(s for s in ik.samples if s.frame_index == frame_index)
        if sample.proximal_status != "known":
            out["refused"][label] = {"sequence_id": sid, "frame_index": frame_index,
                                     "reason": f"H0 IK UNAVAILABLE ({sample.reason})"}
            continue
        position = [f.frame_index for f in semantics.frames].index(frame_index)
        frame, gt_frame = semantics.frames[position], oracle.frames[position]
        if gt_frame.frame_index != frame_index:
            raise ValueError("GT/H0 frame alignment broken")
        lookup = {n: i for i, n in enumerate(semantics.joint_names)}
        yaw = frame.root_orientation.yaw_radians
        h0 = measure_chain(*[frame.articulation.joint_positions[lookup[n]] for n in names], yaw, alignment)
        gt = measure_chain(*[gt_frame.points[n].position for n in names], yaw, alignment)
        bones = {s.target_bone: s for s in fk.pose_bone_samples if s.frame_index == frame_index}
        rig = next(s for s in fk.rig_rotations if s.frame_index == frame_index)
        fk_mid, fk_end = chain_forward_kinematics(rest, target, bones[target.proximal].rotation_local,
                                                  bones[target.distal].rotation_local)
        row = attribute_frame(h0, gt, target, rest, fk_h0_endpoint=fk_end, fk_h0_mid=fk_mid).to_dict()
        stored = rows[(sid, frame_index)]
        for key in NUMBERS:
            if isinstance(row[key], bool) or row[key] is None:
                if row[key] != stored[key]:
                    raise ValueError(f"{label}: {key} differs from rows.json")
            else:
                checks["rows_json_number_max_abs_diff"] = max(
                    checks["rows_json_number_max_abs_diff"], abs(row[key] - stored[key]))
        e_gt = row["e_gt"]
        gt_side, _ = _runtime_side(gt, target, rest, gt.direction)
        oracle_solve = solve_two_bone(target.root_head, gt.direction, gt.reach_fraction * total,
                                      target.proximal_length, target.distal_length, gt_side)
        h0_norm = _normalized_chain(target.root_head, h0, total)
        gt_norm = _normalized_chain(target.root_head, gt, total)
        comparison = next(r for r in json.loads(
            (args.ik61_dir / ik61_report["sequences"][sid]["comparison_file"]).read_text())
            if r["frame_index"] == frame_index)
        checks["oracle_end_vs_e_gt_max"] = max(checks["oracle_end_vs_e_gt_max"],
                                               math.dist(oracle_solve.end, e_gt))
        checks["ik_end_vs_w61_solved_max"] = max(checks["ik_end_vs_w61_solved_max"],
                                                 math.dist(h0_norm[2], sample.solved_endpoint))
        checks["normalized_h0_end_vs_ik_end_max"] = checks["ik_end_vs_w61_solved_max"]
        checks["normalized_gt_end_vs_e_gt_max"] = max(checks["normalized_gt_end_vs_e_gt_max"],
                                                      math.dist(gt_norm[2], e_gt))
        checks["fk_end_vs_w61_comparison_max"] = max(checks["fk_end_vs_w61_comparison_max"],
                                                     math.dist(fk_end, comparison["fk_endpoint"]))
        # H0 IK elbow re-solved from H0 evidence equals the stored W61 elbow.
        h0_side, _ = _runtime_side(h0, target, rest, h0.direction)
        h0_solve = solve_two_bone(target.root_head, h0.direction, h0.reach_fraction * total,
                                  target.proximal_length, target.distal_length, h0_side)
        checks["ik_mid_vs_w61_solved_max"] = max(checks["ik_mid_vs_w61_solved_max"],
                                                 math.dist(h0_solve.mid, sample.solved_mid))
        out["cases"][label] = {
            "sequence_id": sid, "frame_index": frame_index,
            "root": list(target.root_head),
            "fk_mid": list(fk_mid), "fk_end": list(fk_end),
            "ik_mid": list(sample.solved_mid), "ik_end": list(sample.solved_endpoint),
            "oracle_mid": list(oracle_solve.mid), "oracle_end": list(oracle_solve.end),
            "h0_normalized_chain": h0_norm, "gt_normalized_chain": gt_norm,
            "e_gt": list(e_gt), "h0_requested_endpoint": list(sample.requested_endpoint),
            "numbers": {k: row[k] for k in NUMBERS},
            "rig_rotation_armature_space_xyzw": list(rig.rotation_armature_space),
            "pose_bone_local_xyzw": {name: list(s.rotation_local) for name, s in bones.items()
                                     if s.rotation_status == "known" and s.rotation_local is not None},
        }
    out["sanity_checks_cm"] = checks
    tolerance = 1e-9
    if any(v > tolerance for v in checks.values()):
        raise ValueError(f"sanity check failed: {checks}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1))
    print(json.dumps({"cases": list(out["cases"]), "refused": out["refused"], "checks": checks}, indent=2))


if __name__ == "__main__":
    main()
