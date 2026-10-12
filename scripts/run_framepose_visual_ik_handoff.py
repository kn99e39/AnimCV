#!/usr/bin/env python3
"""EVALUATION ONLY (Worklog 74): visual chain-depth -> Worklog 61 rig-space IK endpoint attribution.

Runtime observable: AnimationSemantics v2 sidecars (H0 joints, yaw_H0), Worklog 61
IK, FrameBank detector 2D, Worklog 73 E0/E1 held-out chain forward fractions.
GT / evaluation oracle: 3DPW jointPositions (P3 forward fraction, the Worklog 62
GT-anchored target and the oracle elbow).  GT never enters P0/P1/P2.

Before any comparison the script proves the coordinate/alignment contract: a
candidate rebuilt from the exact H0 direction through the visual path must
reproduce the persisted Worklog 61 solved endpoint and elbow, and every
candidate's IK must satisfy the FK numerical contract.  Otherwise it stops.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_framepose_ik_gt_attribution import SEQUENCES, _stats  # noqa: E402
from run_root_motion_contact_h0_replay import _matched_oracle_sequence  # noqa: E402

from framepose.arm_depth_probe import observed_plane  # noqa: E402
from framepose.bank import load_bank  # noqa: E402
from motion.animation_semantics_v2 import load_animation_semantics_v2  # noqa: E402
from retarget.framepose_fk import derive_rig_alignment, load_rig_fk_calibration  # noqa: E402
from retarget.framepose_ik_gt_attribution import measure_chain  # noqa: E402
from retarget.framepose_target_rest import load_target_rest_pose  # noqa: E402
from retarget.framepose_two_bone_ik import load_ik_chain_set, load_two_bone_ik, target_chain_geometry  # noqa: E402
from retarget.framepose_visual_ik_handoff import (  # noqa: E402
    CANDIDATES, CONTROLS, FK_CONTRACT_TOLERANCE, camera_chain_direction, oracle_reference, rig_direction,
    score, solve_with_direction, unavailable,
)

W73_REPORT_SHA256 = "1c05d50a5e6577329f0573786c6927b07268612ec150fbd079a1aa2bc4b03934"
W62_REPORT_SHA256 = "f7a100cde398e3117717520247d35a61b992c401fcf30d48ea2f4d86c5c9d523"
STABLE = math.sin(math.radians(10))
SHORT = {"3dpw:courtyard_dancing_00:actor0": "dancing", "3dpw:outdoors_crosscountry_00:actor0": "crosscountry",
         "3dpw:courtyard_hug_00:actor1": "hug"}
REFERENCES = ("r_gt_direction_h0_reach",)
ALL = (*CANDIDATES, *CONTROLS, *REFERENCES)
METRICS = ("direction_error_degrees", "requested_endpoint_error", "solved_endpoint_error", "elbow_error")


def _unit(v):
    n = math.sqrt(sum(x * x for x in v))
    return tuple(x / n for x in v)


def _paired(rows, a, b, metric):
    pairs = [(r["candidates"][a][metric], r["candidates"][b][metric]) for r in rows
             if r["candidates"][a][metric] is not None and r["candidates"][b][metric] is not None]
    d = [x - y for x, y in pairs]
    return {**_stats(d), "improved_share": (sum(v < 0 for v in d) / len(d)) if d else None,
            "worse_share": (sum(v > 0 for v in d) / len(d)) if d else None}


def _candidate_summary(rows, name):
    c = [r["candidates"][name] for r in rows]
    agree = [x["side_agrees_with_gt"] for x in c if x["side_agrees_with_gt"] is not None]
    return {**{m: _stats(x[m] for x in c) for m in METRICS},
            "known": sum(x["status"] == "known" for x in c),
            "unavailable": dict(Counter(x["status"] for x in c if x["status"] != "known")),
            "clamps": dict(Counter(str(x["clamp"]) for x in c)),
            "side_status": dict(Counter(str(x["side_status"]) for x in c)),
            "side_agrees_with_gt": sum(agree), "side_flips_vs_gt": len(agree) - sum(agree),
            "side_vs_gt_degrees": _stats(x["side_vs_gt_degrees"] for x in c)}


def _block(rows):
    common = [r for r in rows if all(r["candidates"][n]["status"] == "known" for n in (*CANDIDATES, *CONTROLS))]
    out = {"rows": len(rows), "common_valid_rows": len(common),
           "candidates": {n: _candidate_summary(rows, n) for n in ALL},
           "paired_common": {f"{a}-{b}": {m: _paired(common, a, b, m) for m in METRICS}
                             for a, b in (("p2_e1_visual_direction", "p1_e0_visual_direction"),
                                          ("p2_e1_visual_direction", "p0_h0_ik"),
                                          ("p1_e0_visual_direction", "p0_h0_ik"),
                                          ("p3_gt_forward_fraction", "p2_e1_visual_direction"),
                                          ("a1_h0_f_detector_plane", "p0_h0_ik"),
                                          ("p2_e1_visual_direction", "a1_h0_f_detector_plane"),
                                          ("r_gt_direction_h0_reach", "p3_gt_forward_fraction"))}}
    if common:
        def mean(name, metric):
            return sum(r["candidates"][name][metric] for r in common) / len(common)
        p0, p1, p2 = (mean(n, "solved_endpoint_error") for n in CANDIDATES[:3])
        d0, d1, d2 = (mean(n, "direction_error_degrees") for n in CANDIDATES[:3])
        out["gain_survival"] = {
            "direction_mean_deg": {"p0": d0, "p1": d1, "p2": d2},
            "endpoint_mean_cm": {"p0": p0, "p1": p1, "p2": p2},
            "relative_direction_gain_p2_vs_p1": (d1 - d2) / d1, "relative_endpoint_gain_p2_vs_p1": (p1 - p2) / p1,
            "relative_direction_gain_p2_vs_p0": (d0 - d2) / d0, "relative_endpoint_gain_p2_vs_p0": (p0 - p2) / p0}
        ladder = {n: mean(n, "solved_endpoint_error") for n in ("p2_e1_visual_direction", "p3_gt_forward_fraction",
                                                                 "r_gt_direction_h0_reach")}
        out["attribution_ladder_mean_cm"] = {
            "p2_endpoint": ladder["p2_e1_visual_direction"],
            "visual_depth_error_share (p2 - p3)": ladder["p2_e1_visual_direction"] - ladder["p3_gt_forward_fraction"],
            "observed_plane_error_share (p3 - gt_direction)": ladder["p3_gt_forward_fraction"] - ladder["r_gt_direction_h0_reach"],
            "h0_reach_error_share (gt_direction - oracle 0)": ladder["r_gt_direction_h0_reach"],
            "elbow_h0_side_vs_gt_side_on_gt_target_cm": _stats(r["w62_elbow_error_h0_side_on_gt_target"] for r in common),
            "clamped_rows": {n: sum(r["candidates"][n]["clamp"] not in ("none", None) for r in common) for n in ALL}}
    return out


def _strata(rows, key, bins):
    out = {}
    common = [r for r in rows if all(r["candidates"][n]["status"] == "known" for n in (*CANDIDATES, *CONTROLS))]
    for label, (lo, hi) in bins.items():
        group = [r for r in common if r[key] is not None and lo <= r[key] < hi]
        out[label] = {"n": len(group), **{f"{n}_endpoint_mean_cm": (
            sum(r["candidates"][n]["solved_endpoint_error"] for r in group) / len(group) if group else None)
            for n in ALL},
            "p2_minus_p1_endpoint": _paired(group, "p2_e1_visual_direction", "p1_e0_visual_direction",
                                            "solved_endpoint_error"),
            "p2_minus_p0_endpoint": _paired(group, "p2_e1_visual_direction", "p0_h0_ik", "solved_endpoint_error")}
    return out


def _quartile_bins(values):
    v = sorted(values)
    edges = [v[int(len(v) * q / 4)] for q in range(1, 4)]
    b = [-math.inf, *edges, math.inf]
    return {f"Q{i + 1} [{b[i]:.3g}, {b[i + 1]:.3g})": (b[i], b[i + 1]) for i in range(4)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("rig", "rest", "calibration", "chains", "semantics_dir", "ik61_dir", "w62_dir", "threedpw_raw",
                 "bank", "w73_dir", "out_dir"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    args = parser.parse_args()

    rest = load_target_rest_pose(args.rest)
    if rest.provenance["source_fbx_sha256"] != hashlib.sha256(args.rig.read_bytes()).hexdigest():
        raise ValueError("Blender rest snapshot differs from current FBX bytes")
    alignment = derive_rig_alignment(rest, load_rig_fk_calibration(args.calibration))
    chain = load_ik_chain_set(args.chains).chains[0]
    target = target_chain_geometry(rest, chain)
    names = (chain.source_root, chain.source_mid, chain.source_end)

    identities = {}
    for label, path, expected in (("w73_report", args.w73_dir / "report.json", W73_REPORT_SHA256),
                                  ("w62_report", args.w62_dir / "report.json", W62_REPORT_SHA256)):
        identities[label] = hashlib.sha256(path.read_bytes()).hexdigest()
        if identities[label] != expected:
            raise SystemExit(f"{label} identity mismatch: {identities[label]}")
    for label, path in (("w73_rows_heldout", args.w73_dir / "rows_heldout.json"),
                        ("w73_e0_checkpoint", args.w73_dir / "E0_3DPW_ONLY.pt"),
                        ("w73_e1_checkpoint", args.w73_dir / "E1_3DPW_PLUS_EXTERNAL.pt"),
                        ("w62_rows", args.w62_dir / "rows.json"), ("w61_report", args.ik61_dir / "report.json")):
        identities[label] = hashlib.sha256(path.read_bytes()).hexdigest()
    w62_report = json.loads((args.w62_dir / "report.json").read_text())
    w62_rows = {(r["sequence_id"], r["frame_index"]): r for r in json.loads((args.w62_dir / "rows.json").read_text())}
    w73_rows = {}
    for r in json.loads((args.w73_dir / "rows_heldout.json").read_text()):
        key = (r["sequence_id"], r["frame_index"])
        if key in w73_rows:
            raise SystemExit(f"duplicate Worklog 73 row {key}")
        w73_rows[key] = r
    bank = load_bank(args.bank)
    position = {(s.sequence_id, s.frame_index): p for p, s in enumerate(bank.samples)}
    identities["bank_content_digest"] = bank.content_digest()
    source_report = json.loads((args.semantics_dir / "report.json").read_text())
    ik61_report = json.loads((args.ik61_dir / "report.json").read_text())
    source_files = {row["sequence_id"]: row for row in source_report["per_sequence"]}

    rows, accounting, checks = [], {}, defaultdict(float)
    for sid in SEQUENCES:
        semantics = load_animation_semantics_v2(args.semantics_dir / source_files[sid]["file"])
        if semantics.content_digest() != source_files[sid]["content_digest"]:
            raise ValueError(f"{sid}: semantics digest changed")
        ik = load_two_bone_ik(args.ik61_dir / ik61_report["sequences"][sid]["ik_file"])
        if ik.provenance["semantics_digest"] != semantics.content_digest():
            raise ValueError(f"{sid}: Worklog 61 IK was built from other semantics")
        oracle = _matched_oracle_sequence(sid, args.threedpw_raw, [f.frame_index for f in semantics.frames])
        lookup = {n: i for i, n in enumerate(semantics.joint_names)}
        ik_by = {s.frame_index: s for s in ik.samples}
        acc = Counter()
        acc["w73_rows_for_sequence"] = sum(1 for k in w73_rows if k[0] == sid)
        acc["semantics_frames"] = len(semantics.frames)
        for frame, gt_frame in zip(semantics.frames, oracle.frames):
            if gt_frame.frame_index != frame.frame_index:
                raise ValueError("GT/H0 frame alignment broken")
            key = (sid, frame.frame_index)
            sample = ik_by[frame.frame_index]
            if sample.proximal_status != "known":
                acc["excluded:w61_ik_unavailable:" + sample.reason] += 1
                acc["excluded_rows_with_w73_prediction"] += key in w73_rows
                continue
            acc["w61_ik_known"] += 1
            gt_points = [gt_frame.points[n] for n in names]
            if not all(p.observation_valid for p in gt_points):
                acc["excluded:gt_invalid"] += 1
                continue
            yaw = frame.root_orientation.yaw_radians
            h0_pos = [frame.articulation.joint_positions[lookup[n]] for n in names]
            gt_pos = [p.position for p in gt_points]
            h0, gt = measure_chain(*h0_pos, yaw, alignment), measure_chain(*gt_pos, yaw, alignment)
            if h0 is None or gt is None:
                acc["excluded:degenerate_measure"] += 1
                continue
            acc["gt_valid_matched"] += 1
            e_gt, oracle_mid, gt_side, gt_side_status = oracle_reference(gt, target, rest)
            w62 = w62_rows[key]
            checks["e_gt_vs_w62_rows_max"] = max(checks["e_gt_vs_w62_rows_max"], math.dist(e_gt, w62["e_gt"]))

            # Exact identity join with Worklog 73 (no positional approximation).
            w73 = w73_rows.get(key)
            p = position.get(key)
            if w73 is None or p is None:
                acc["matched_without_w73_prediction"] += 1
                pred = None
            else:
                if w73["position"] != p or w73["sample_id"] != bank.samples[p].sample_id:
                    raise SystemExit(f"{key}: Worklog 73 row does not match the FrameBank sample")
                acc["matched_with_w73_prediction"] += 1
                pred = w73

            # Contract proof: the exact H0 direction through the visual path reproduces Worklog 61.
            h0_cam = _unit(tuple(b - a for a, b in zip(h0_pos[0], h0_pos[2])))
            rebuilt = camera_chain_direction(h0_cam[1], (h0_cam[0], h0_cam[2]))
            checks["h0_rebuilt_camera_direction_max"] = max(checks["h0_rebuilt_camera_direction_max"],
                                                            math.dist(rebuilt, h0_cam))
            parity = solve_with_direction("parity", h0, rig_direction(rebuilt, yaw, alignment), target, rest)
            checks["parity_vs_w61_solved_endpoint_max"] = max(checks["parity_vs_w61_solved_endpoint_max"],
                                                              math.dist(parity.solved_endpoint, sample.solved_endpoint))
            checks["parity_vs_w61_solved_mid_max"] = max(checks["parity_vs_w61_solved_mid_max"],
                                                         math.dist(parity.solved_mid, sample.solved_mid))
            checks["parity_vs_w61_requested_max"] = max(checks["parity_vs_w61_requested_max"],
                                                        math.dist(parity.requested_endpoint, sample.requested_endpoint))

            solves = {"p0_h0_ik": solve_with_direction("p0_h0_ik", h0, h0.direction, target, rest),
                      "r_gt_direction_h0_reach": solve_with_direction("r_gt_direction_h0_reach", h0, gt.direction,
                                                                      target, rest)}
            gt_cam = _unit(tuple(b - a for a, b in zip(gt_pos[0], gt_pos[2])))
            f = {"h0": h0_cam[1], "gt": gt_cam[1]}
            plane = None
            if pred is not None:
                f["e0"], f["e1"] = pred["chain_f_e0"], pred["chain_f_e1"]
                checks["w73_chain_f_h0_vs_semantics_max"] = max(checks["w73_chain_f_h0_vs_semantics_max"],
                                                                abs(pred["chain_f_h0"] - f["h0"]))
                checks["w73_chain_f_gt_vs_3dpw_max"] = max(checks["w73_chain_f_gt_vs_3dpw_max"],
                                                           abs(pred["chain_f_gt"] - f["gt"]))
                planes, oks = observed_plane(bank.arrays["input_2d"][p][None],
                                             np.asarray([bank.samples[p].image_size], float))
                if bool(oks["chain"][0]):
                    plane = (float(planes["chain"][0, 0]), float(planes["chain"][0, 1]))
            for name, fkey in (("p1_e0_visual_direction", "e0"), ("p2_e1_visual_direction", "e1"),
                               ("p3_gt_forward_fraction", "gt"), ("a1_h0_f_detector_plane", "h0")):
                if pred is None:
                    solves[name] = unavailable(name, "no_w73_prediction")
                elif plane is None:
                    solves[name] = unavailable(name, "degenerate_observed_plane")
                else:
                    cam = camera_chain_direction(f[fkey], plane)
                    solves[name] = solve_with_direction(name, h0, rig_direction(cam, yaw, alignment), target, rest)
            for s in solves.values():
                if s.fk_contract_error is not None:
                    checks["fk_contract_max"] = max(checks["fk_contract_max"], s.fk_contract_error)
            checks["p0_vs_w61_solved_endpoint_max"] = max(checks["p0_vs_w61_solved_endpoint_max"], math.dist(
                solves["p0_h0_ik"].solved_endpoint, sample.solved_endpoint))
            checks["r_gt_direction_vs_w62_d_endpoint_max"] = max(
                checks["r_gt_direction_vs_w62_d_endpoint_max"],
                abs(math.dist(solves["r_gt_direction_h0_reach"].solved_endpoint, e_gt)
                    - w62["gt_direction_h0_reach_endpoint_error"]))
            rows.append({
                "sequence_id": sid, "sequence": SHORT[sid], "frame_index": frame.frame_index,
                "sample_id": bank.samples[p].sample_id if p is not None else None, "bank_position": p,
                "f": f, "observed_chain_plane_xz": plane, "yaw_h0_radians": yaw,
                "abs_gt_f": abs(f["gt"]), "w62_h0_direction_error_degrees": w62["endpoint_direction_error_degrees"],
                "abs_h0_reach_error": abs(w62["reach_fraction_error"]),
                "h0_bend_angle_degrees": h0.bend_angle_degrees, "gt_bend_angle_degrees": gt.bend_angle_degrees,
                "h0_side_status": "degenerate" if h0.degenerate else "observed", "gt_side_status": gt_side_status,
                "w62_elbow_error_h0_side_on_gt_target": w62["elbow_error_h0_side_on_gt_target"],
                "e_gt": list(e_gt), "oracle_mid": list(oracle_mid) if oracle_mid else None,
                "candidates": {n: score(solves[n], gt, e_gt, oracle_mid, gt_side) for n in ALL},
                "geometry": {n: solves[n].to_dict() for n in ALL},
            })
        accounting[sid] = dict(acc)

    limits = {"e_gt_vs_w62_rows_max": 1e-9, "h0_rebuilt_camera_direction_max": 1e-12,
              "parity_vs_w61_solved_endpoint_max": 1e-9, "parity_vs_w61_solved_mid_max": 1e-9,
              "parity_vs_w61_requested_max": 1e-9, "p0_vs_w61_solved_endpoint_max": 1e-9,
              "fk_contract_max": FK_CONTRACT_TOLERANCE, "r_gt_direction_vs_w62_d_endpoint_max": 1e-9,
              "w73_chain_f_h0_vs_semantics_max": 1e-6, "w73_chain_f_gt_vs_3dpw_max": 1e-5}
    failed = {k: checks[k] for k, lim in limits.items() if checks[k] > lim}
    if failed:
        raise SystemExit(f"STOP: coordinate/alignment/identity contract not reproduced: {failed}")

    by_seq = {SHORT[sid]: [r for r in rows if r["sequence_id"] == sid] for sid in SEQUENCES}
    report = {
        "schema": "animcv_framepose_visual_ik_handoff_v1",
        "labels": {"runtime": "P0/P1/P2: H0 semantics, Worklog 61 IK, FrameBank detector 2D, Worklog 73 E0/E1 chain f",
                   "evaluation": "P3 GT chain f; Worklog 62 GT-anchored E_GT and oracle elbow; GT never in P0/P1/P2",
                   "reference": "r_gt_direction_h0_reach = Worklog 62 D endpoint (GT direction + H0 reach), H0 side"},
        "identities": identities, "target_chain": target.to_dict(),
        "contract_checks": {k: {"value": checks[k], "limit": limits.get(k)} for k in sorted(checks)},
        "shared_transform": "A*Rz(-yaw_H0) on every direction; GT yaw unused",
        "direction_construction": "reconstruct_direction(chain f, observed_plane(FrameBank detector 2D, image size))",
        "accounting": accounting,
        "pooled": _block(rows), **{f"sequence:{k}": _block(v) for k, v in by_seq.items()},
        "pooled_without_crosscountry": _block([r for r in rows if r["sequence"] != "crosscountry"]),
        "strata": {
            "abs_gt_chain_f": _strata(rows, "abs_gt_f", {"near_plane": (0, STABLE), "intermediate": (STABLE, 0.5),
                                                         "out_of_plane": (0.5, 1.01)}),
            "abs_h0_reach_error": _strata(rows, "abs_h0_reach_error", _quartile_bins([r["abs_h0_reach_error"] for r in rows])),
            "h0_bend_angle_degrees": _strata(rows, "h0_bend_angle_degrees",
                                             {"0-15": (0, 15), "15-45": (15, 45), "45-90": (45, 90), ">=90": (90, 181)}),
            "w62_h0_direction_error_quartile": _strata(rows, "w62_h0_direction_error_degrees",
                                                       _quartile_bins([r["w62_h0_direction_error_degrees"] for r in rows]))},
        "owner_cases": {},
    }
    row_by = {(r["sequence_id"], r["frame_index"]): r for r in rows}
    for label, case in w62_report["owner_cases"].items():
        key = (case["sequence_id"], case["frame_index"])
        r = row_by.get(key)
        report["owner_cases"][label] = {
            "sequence_id": key[0], "frame_index": key[1], "overlaps": case.get("overlaps"),
            "unavailable": None if r else (case.get("excluded") or "not a matched row"),
            "f": r["f"] if r else None,
            "candidates": {n: {m: r["candidates"][n][m] for m in (*METRICS, "status", "side_agrees_with_gt")}
                           for n in ALL} if r else None}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "rows.json").write_text(json.dumps(rows))
    columns = ["sequence_id", "frame_index", "abs_gt_f", "w62_h0_direction_error_degrees", "abs_h0_reach_error",
               *[f"{n}:{m}" for n in ALL for m in ("direction_error_degrees", "solved_endpoint_error", "elbow_error")]]
    with (args.out_dir / "handoff_table.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        for r in rows:
            writer.writerow([r["sequence_id"], r["frame_index"], r["abs_gt_f"], r["w62_h0_direction_error_degrees"],
                             r["abs_h0_reach_error"], *[r["candidates"][n][m] for n in ALL
                                                        for m in ("direction_error_degrees", "solved_endpoint_error",
                                                                  "elbow_error")]])
    output = args.out_dir / "report.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True, default=str))
    p = report["pooled"]
    print(json.dumps({"sha256": hashlib.sha256(output.read_bytes()).hexdigest(), "rows": len(rows),
                      "common_valid": p["common_valid_rows"], "accounting": accounting,
                      "contract_checks": {k: v["value"] for k, v in report["contract_checks"].items()},
                      "solved_endpoint_mean": {n: p["candidates"][n]["solved_endpoint_error"].get("mean") for n in ALL},
                      "direction_mean": {n: p["candidates"][n]["direction_error_degrees"].get("mean") for n in ALL}},
                     indent=1))


if __name__ == "__main__":
    main()
