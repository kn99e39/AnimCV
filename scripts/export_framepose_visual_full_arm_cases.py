#!/usr/bin/env python3
"""EVALUATION ONLY (Worklog 76): export Blender review geometry for the full-arm visual IK test rows.

Rows are chosen by fixed rules from the common-valid test population (F2 = E1,
F0 = H0): largest F2-F0 wrist gain / regression, largest F2-F0 elbow gain /
regression, wrist improves while elbow regresses (largest min(wrist gain,
elbow loss)), near-straight (GT bend < 15 deg) and near-folded/strong (GT bend
>= 90 deg) rows with the median F2 wrist error, strongly out-of-plane
(|GT f| >= 0.5, median F0 wrist error), lowest detector arm confidence, and the
four Worklog 75 owner rows.  Overlapping picks keep one entry with every label.

Test sequences have no Worklog 60 replay files, so the object rotation is the
Worklog 60 formula A * Rz(yaw_H0) * A^-1 (``solve_object_owned_fk``); the export
first proves that formula against the persisted Worklog 60 rig rotation on the
Worklog 74 validation rows.  The body stays in rest pose (no FK locals), as in
the Worklog 74 review where only the hidden left arm carried locals.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from retarget.axis_utils import quaternion_conjugate, quaternion_from_axis_angle, quaternion_multiply
from retarget.framepose_fk import derive_rig_alignment, load_rig_fk_calibration
from retarget.framepose_object_owner import load_object_owned_fk
from retarget.framepose_target_rest import load_target_rest_pose
from retarget.framepose_two_bone_ik import chain_forward_kinematics, load_ik_chain_set, target_chain_geometry

NAMES = {"f0_h0_full_ik": "F0", "f1_e0_full_ik": "F1", "f2_e1_full_ik": "F2"}
OWNERS = (("3dpw:downtown_bar_00:actor0", 775), ("3dpw:downtown_sitOnStairs_00:actor0", 325),
          ("3dpw:downtown_runForBus_01:actor0", 185), ("3dpw:flat_packBags_00:actor0", 446))
NUMBERS = ("wrist_error", "elbow_error", "upper_direction_error_degrees", "forearm_direction_error_degrees",
           "chain_direction_error_degrees", "bend_angle_error_degrees", "bend_plane_error_degrees", "side_agrees_with_gt")


def object_delta(alignment, yaw):
    q = quaternion_multiply(quaternion_multiply(alignment, quaternion_from_axis_angle((0.0, 0.0, 1.0), yaw)),
                            quaternion_conjugate(alignment))
    n = math.sqrt(sum(v * v for v in q))
    return tuple(v / n for v in q)


def _qdiff(a, b):
    s = -1.0 if sum(x * y for x, y in zip(a, b)) < 0 else 1.0
    return max(abs(x - s * y) for x, y in zip(a, b))


def pick(rows):
    key = lambda r: (r["sequence_id"], r["frame_index"])  # noqa: E731
    c = lambda r, n, m: r["candidates"][n][m]  # noqa: E731
    dw = lambda r: c(r, "f2_e1_full_ik", "wrist_error") - c(r, "f0_h0_full_ik", "wrist_error")  # noqa: E731
    de = lambda r: c(r, "f2_e1_full_ik", "elbow_error") - c(r, "f0_h0_full_ik", "elbow_error")  # noqa: E731
    out = {"largest_e1_wrist_improvement": min(rows, key=lambda r: (dw(r), key(r))),
           "largest_e1_wrist_regression": max(rows, key=lambda r: (dw(r), key(r))),
           "largest_e1_elbow_improvement": min(rows, key=lambda r: (de(r), key(r))),
           "largest_e1_elbow_regression": max(rows, key=lambda r: (de(r), key(r)))}
    trade = [r for r in rows if dw(r) < 0 < de(r)]
    if trade:
        out["wrist_improves_elbow_regresses"] = max(trade, key=lambda r: (min(-dw(r), de(r)), key(r)))

    def median_of(group, metric_candidate):
        g = sorted(group, key=lambda r: (c(r, metric_candidate, "wrist_error"), key(r)))
        return g[len(g) // 2] if g else None
    for label, group, cand in (("near_straight", [r for r in rows if r["gt_bend_degrees"] < 15], "f2_e1_full_ik"),
                               ("strongly_bent", [r for r in rows if r["gt_bend_degrees"] >= 90], "f2_e1_full_ik"),
                               ("strongly_out_of_plane", [r for r in rows if abs(r["f_gt"]) >= 0.5], "f0_h0_full_ik")):
        r = median_of(group, cand)
        if r is not None:
            out[label] = r
    out["low_detector_confidence"] = min(rows, key=lambda r: (r["min_arm_confidence"], key(r)))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("rig", "rest", "calibration", "chains", "fk60_dir", "full_arm_dir", "w74_dir", "out"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    args = parser.parse_args()
    rest = load_target_rest_pose(args.rest)
    if rest.provenance["source_fbx_sha256"] != hashlib.sha256(args.rig.read_bytes()).hexdigest():
        raise ValueError("Blender rest snapshot differs from current FBX bytes")
    alignment = derive_rig_alignment(rest, load_rig_fk_calibration(args.calibration))
    target = target_chain_geometry(rest, load_ik_chain_set(args.chains).chains[0])

    # Prove the Worklog 60 object-rotation formula on the persisted Worklog 74 validation rows.
    fk60 = json.loads((args.fk60_dir / "report.json").read_text())
    w74 = json.loads((args.w74_dir / "rows.json").read_text())
    formula_max, loaded = 0.0, {}
    for r in w74:
        sid = r["sequence_id"]
        if sid not in loaded:
            fk = load_object_owned_fk(args.fk60_dir / fk60["sequences"][sid]["file"])
            loaded[sid] = {s.frame_index: s.rotation_armature_space for s in fk.rig_rotations}
        formula_max = max(formula_max, _qdiff(object_delta(alignment, r["yaw_h0_radians"]), loaded[sid][r["frame_index"]]))
    if formula_max > 1e-12:
        raise ValueError(f"Worklog 60 object-rotation formula not reproduced: {formula_max}")

    rows = json.loads((args.full_arm_dir / "rows_test.json").read_text())
    common = [r for r in rows if r["reference"]["status"] == "known"
              and all(r["candidates"][n]["status"] == "known" for n in NAMES)]
    by_key = {(r["sequence_id"], r["frame_index"]): r for r in rows}
    picks, refused = {}, {}
    for label, r in pick(common).items():
        picks.setdefault((r["sequence_id"], r["frame_index"]), []).append(label)
    for sid, fi in OWNERS:
        label = f"w75_owner_{sid.split(':')[1]}_{fi}"
        r = by_key.get((sid, fi))
        if r is None or r not in common:
            refused[label] = {"sequence_id": sid, "frame_index": fi, "reason": "not a common-valid test row"}
        else:
            picks.setdefault((sid, fi), []).append(label)
    fk_mid, fk_end = chain_forward_kinematics(rest, target, (0.0, 0.0, 0.0, 1.0), (0.0, 0.0, 0.0, 1.0))
    out = {"schema": "animcv_framepose_visual_full_arm_review_cases_v1", "space": "heading_relative_armature",
           "shared_transform": "A*Rz(-yaw_H0) on every direction; object rotation A*Rz(yaw_H0)*A^-1 (Worklog 60)",
           "rest_snapshot_digest": rest.digest(), "target_chain": target.to_dict(),
           "canonical_heading_zero_to_armature_xyzw": list(alignment), "cases": {}, "refused": refused,
           "sanity_checks_cm": {"w60_object_rotation_formula_vs_persisted_max": formula_max}}
    for k, labels in sorted(picks.items()):
        r = by_key[k]
        g = r["geometry"]
        label = "+".join(labels)
        out["cases"][label] = {
            "labels": labels, "sequence_id": k[0], "frame_index": k[1], "bank_position": r["position"],
            "root": list(target.root_head), "fk_mid": list(fk_mid), "fk_end": list(fk_end),
            "e_gt": r["e_gt"], "oracle_mid": r["reference"]["mid"],
            "chains": {NAMES[n]: {"mid": g[n]["mid"], "end": g[n]["end"]} for n in NAMES},
            "f": {"e0": r["f_e0"], "e1": r["f_e1"], "gt": r["f_gt"]},
            "gt_bend_degrees": r["gt_bend_degrees"], "min_arm_confidence": r["min_arm_confidence"],
            "numbers": {NAMES[n]: {m: r["candidates"][n][m] for m in NUMBERS} for n in NAMES},
            "rig_rotation_armature_space_xyzw": list(object_delta(alignment, r["yaw_h0_radians"])),
            "pose_bone_local_xyzw": {},
        }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1))
    print(json.dumps({"cases": list(out["cases"]), "refused": refused, "checks": out["sanity_checks_cm"]}, indent=1))


if __name__ == "__main__":
    main()
