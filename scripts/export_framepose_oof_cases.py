#!/usr/bin/env python3
"""Worklog 79 exact-identity review cases for the unchanged W77 Blender renderer.

Bound to the authoritative BaseRig rest snapshot (render with the macOS Blender used for
Worklogs 76-78; its FBX import reproduces that snapshot exactly, so the strict W77 binding
gate applies unchanged).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from audit_framepose_oof_conditioning import sha, write_json
from export_framepose_visual_full_arm_cases import object_delta
from run_framepose_arm_geometry_owner_eval import OWNER_FRAMES
from retarget.framepose_fk import derive_rig_alignment, load_rig_fk_calibration
from retarget.framepose_target_rest import load_target_rest_pose
from retarget.framepose_two_bone_ik import chain_forward_kinematics, load_ik_chain_set, target_chain_geometry

DRAW = {"B0": ("b0", (.15, .45, 1., 1.)), "B1": ("b1", (.9, .15, .85, 1.)),
        "G0_OOF": ("g0", (1., .55, .1, 1.)), "G1_OOF": ("g1", (.1, .85, .95, 1.))}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("rig", "rest", "calibration", "chains", "historical_eval", "oof_eval", "out"):
        p.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    args = p.parse_args()
    rest = load_target_rest_pose(args.rest)
    if rest.provenance["source_fbx_sha256"] != sha(args.rig):
        raise ValueError("rig identity changed")
    alignment = derive_rig_alignment(rest, load_rig_fk_calibration(args.calibration))
    target = target_chain_geometry(rest, load_ik_chain_set(args.chains).chains[0])
    read = lambda directory, filename: json.loads((directory / filename).read_text())
    rows, pairs = read(args.oof_eval, "rows_test.json"), read(args.oof_eval, "pairs_test.json")
    key = lambda r: (r["sequence_id"], r["frame_index"])
    old = {key(r): r for r in read(args.historical_eval, "rows_test.json")}
    old_pairs = {key(r): r for r in read(args.historical_eval, "pairs_test.json")}
    current = {key(r): r for r in rows}
    order = {(r["sequence_id"], r["bank_order"]): r for r in rows}
    pair_by = {key(r): r for r in pairs}
    stills = {f"owner_{s.split(':')[1]}_{f}": current[(s, f)] for s, f in OWNER_FRAMES}
    strips = {}
    for g in ("g0", "g1"):
        de = lambda r: r["candidates"][g]["elbow_error"] - old[key(r)]["candidates"][g]["elbow_error"]
        stills[f"largest_{g}_oof_elbow_improvement"] = min(rows, key=lambda r: (de(r), key(r)))
        stills[f"largest_{g}_oof_elbow_regression"] = max(rows, key=lambda r: (de(r), key(r)))
        dm = lambda r: r["candidates"][g]["upper_direction_residual_degrees"] - old_pairs[key(r)]["candidates"][g]["upper_direction_residual_degrees"]
        for kind, pick in (("improvement", min), ("regression", max)):
            pair = pick(pairs, key=lambda r: (dm(r), key(r)))
            center = current[key(pair)]
            strips[f"largest_{g}_oof_temporal_{kind}"] = [r for k in (-1, 0, 1, 2)
                if (r := order.get((center["sequence_id"], center["bank_order"] + k))) is not None]
    conflict = [r for r in rows if r["gt_bend_degrees"] >= 90
                and r["candidates"]["b1"]["wrist_error"] < r["candidates"]["b0"]["wrist_error"]
                and r["candidates"]["b1"]["elbow_error"] > r["candidates"]["b0"]["elbow_error"]]
    stills["near_folded_wrist_elbow_conflict"] = max(conflict, key=lambda r: (
        min(r["candidates"]["b0"]["wrist_error"] - r["candidates"]["b1"]["wrist_error"],
            r["candidates"]["b1"]["elbow_error"] - r["candidates"]["b0"]["elbow_error"]), key(r)))
    fk_mid, fk_end = chain_forward_kinematics(rest, target, (0., 0., 0., 1.), (0., 0., 0., 1.))

    def case(label, r, strip=None, index=None):
        hist, gm, c = old[key(r)], r["geometry"], r["candidates"]
        caption = [f"W79 OOF conditioning | benchmark_detector_observation | {label}",
                   f"{r['sequence_id']} #{r['frame_index']} bank {r['position']} GT bend {r['gt_bend_degrees']:.1f} deg",
                   "blue B0; magenta B1; orange G0_OOF; cyan G1_OOF; green GT; yellow E_GT",
                   f"Frozen full-data E1 f {r['f_e1']:+.3f}; GT f {r['f_gt']:+.3f}; GT reach {r['gt_reach']:.3f}",
                   "elbow cm: " + " ".join(f"{tag} {c[n]['elbow_error']:.2f}" for tag, (n, _) in DRAW.items()),
                   "wrist cm: " + " ".join(f"{tag} {c[n]['wrist_error']:.2f}" for tag, (n, _) in DRAW.items()),
                   "IN_SAMPLE elbow / wrist cm: " + " ".join(f"{g.upper()} {hist['candidates'][g]['elbow_error']:.2f}/{hist['candidates'][g]['wrist_error']:.2f}" for g in ("g0", "g1")),
                   "reach / side err deg: " + " ".join(f"{g.upper()}_OOF {r['learned'][g.upper()]['reach']:.3f}/{c[g]['side_error_degrees']:.1f}" for g in ("g0", "g1"))]
        pair = pair_by.get(key(r))
        if pair:
            caption.append(f"pair #{pair['frame_index']} -> #{pair['next_frame_index']} upper residual IN -> OOF deg: "
                           + " ".join(f"{g.upper()} {old_pairs[key(r)]['candidates'][g]['upper_direction_residual_degrees']:.2f}->{pair['candidates'][g]['upper_direction_residual_degrees']:.2f}" for g in ("g0", "g1")))
        return {"sequence_id": r["sequence_id"], "frame_index": r["frame_index"], "bank_position": r["position"],
                "root": list(target.root_head), "fk_mid": list(fk_mid), "fk_end": list(fk_end), "e_gt": r["e_gt"],
                "oracle_mid": r["reference"]["mid"], "caption": caption,
                "draw": {tag: {"mid": gm[n]["mid"], "end": gm[n]["end"], "color": color} for tag, (n, color) in DRAW.items()},
                "numbers": {**{tag: c[n] for tag, (n, _) in DRAW.items()},
                            **{g.upper() + "_IN_SAMPLE": hist["candidates"][g] for g in ("g0", "g1")}},
                "rig_rotation_armature_space_xyzw": list(object_delta(alignment, r["yaw_h0_radians"])),
                "pose_bone_local_xyzw": {}, "strip": strip, "strip_index": index}

    cases = {label: case(label, r) for label, r in stills.items()}
    for label, frames in strips.items():
        for index, r in enumerate(frames):
            cases[f"{label}__{index}_f{r['frame_index']}"] = case(label, r, label, index)
    write_json(args.out, {"schema": "animcv_w79_oof_review_v1", "rest_snapshot_digest": rest.digest(),
        "target_chain": target.to_dict(), "canonical_heading_zero_to_armature_xyzw": list(alignment),
        "sanity_checks_cm": {}, "refused": {}, "cases": cases})
    print(json.dumps({"stills": {k: key(r) for k, r in stills.items()},
                      "strips": {k: [key(r) for r in v] for k, v in strips.items()}}, indent=2))


if __name__ == "__main__":
    main()
