#!/usr/bin/env python3
"""EVALUATION ONLY (Worklog 82): geometry-anchored visual correction (R0 / R1) vs the historical arms.

Candidates on the exact Worklog 76-81 test population (6,003 rows, 5,504 pairs):
  B0 (Worklog 77 C000), B1 (C100), J0 / J1 (frozen Worklog 80), R0 / R1 (corrected J0 arm, this batch);
  C111 / C_GT_F oracles as read-only headroom.  Every learned arm (J0/J1/R0/R1) is the predicted connected
  unit arm through the unchanged Worklog 62 measure_chain and Worklog 76 full_arm.
Correction accounting uses the stored camera-frame anchor, correction and corrected arm.

Order: identities -> Gate 1 (historical parity: population, pairs, thresholds, B0/B1/J0/J1 equal Worklog 80;
R anchor equals Worklog 80 J0) -> Gate 2 (corrected geometry valid, IK contract) -> evaluation.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_framepose_arm_geometry_owner_eval import _cmp, _table, pair_records, side_error  # noqa: E402
from run_framepose_ik_gt_attribution import _stats  # noqa: E402
from run_framepose_joint_arm_geometry_eval import j_measure  # noqa: E402
from run_framepose_visual_arm_ownership import _excludes_zero, evaluate, row_record  # noqa: E402
from run_framepose_visual_full_arm_ik import BEND_REGIMES, BOOTSTRAP, NAMES, STABLE, _qdiff, _sha, plane_for  # noqa: E402
from run_root_motion_contact_h0_replay import _matched_oracle_sequence  # noqa: E402

from framepose.bank import load_bank  # noqa: E402
from motion.animation_semantics_v2_bridge import current_root_orientation  # noqa: E402
from pose.framepose_bridge import assemble_h0, build_h0_lifted_sequence, sequence_frame_positions, sequence_ids_in_split  # noqa: E402
from retarget.framepose_fk import derive_rig_alignment, load_rig_fk_calibration  # noqa: E402
from retarget.framepose_target_rest import load_target_rest_pose  # noqa: E402
from retarget.framepose_two_bone_ik import load_ik_chain_set, target_chain_geometry  # noqa: E402
from retarget.framepose_visual_arm_ownership import RESIDUAL_METRICS  # noqa: E402
from retarget.framepose_visual_full_arm_ik import TEMPORAL_METRICS, full_arm  # noqa: E402

PINNED = {"w76_report": "dad46288436ac0fe28fea9fab7ad6882c6a39fd476e717af4eaee497ab096a2b",
          "w80_predictions": "eaf3f57c64acd9c3eb086cd84635884aef795e0fd635ea7d3a5776d08aead00f",
          "w80_eval_report": "a27eeab2721f180f4fc0ed6bd31777128e2d1fef3f23786587b6bad885ef60e5",
          "w80_rows_test": "8835e79a9a70038294ea46b5afd5e735275b3067a363304a937c21bd4eed60be",
          "w80_pairs_test": "b25e7c979ce48886a26268cebee485fce690caa658fcbcab6ee9b098121cb962",
          "w81_report": "ff1564d3e69e1a77bd4d67f25928f8220e99827b9d9f33660048150deba2b8fe"}
J = {"j0": "J0_GEOMETRY_ONLY", "j1": "J1_GEOMETRY_PLUS_VISION"}
R = {"r0": "R0_GEOMETRY_CORRECTION", "r1": "R1_VISUAL_CORRECTION"}
BASE = {"b0": "c000", "b1": "c100"}
ORACLES = ("c111", "c_gtf_h0r_h0s")
CANDIDATES = ("b0", "b1", *J, *R, *ORACLES)
KEEP_GEOMETRY = ("b0", "b1", "j0", "j1", "r0", "r1", "c111")
STATIC = ("wrist_error", "elbow_error", "upper_direction_error_degrees", "forearm_direction_error_degrees",
          "chain_direction_error_degrees", "reach_error", "side_error_degrees", "abs_bend_angle_error_degrees",
          "bend_plane_error_degrees", "proximal_rotation_vs_gt_degrees", "distal_rotation_vs_gt_degrees")
PAIRED_STATIC = ("wrist_error", "elbow_error", "upper_direction_error_degrees", "forearm_direction_error_degrees",
                 "chain_direction_error_degrees", "reach_error", "side_error_degrees", "abs_bend_angle_error_degrees")
TEMPORAL = ("upper_direction_residual_degrees", "proximal_rotation_residual_degrees", "forearm_direction_residual_degrees",
            "chain_direction_residual_degrees", "elbow_motion_residual", "wrist_motion_residual",
            "disc:proximal_rotation_change_degrees", "disc:elbow_displacement", "reach_change", "side_change_degrees",
            "chain_direction_change_degrees")
COMPARISONS = (("r1", "r0"), ("r1", "j0"), ("r1", "j1"), ("r1", "b1"), ("r1", "b0"), ("r0", "j0"), ("r0", "j1"), ("r0", "b1"))
F_BINS = {"near_plane [0,sin10)": (0, STABLE), "intermediate": (STABLE, 0.5), "out_of_plane [0.5,1]": (0.5, 1.01)}
OWNER_FRAMES = (("3dpw:downtown_bar_00:actor0", 514), ("3dpw:downtown_bar_00:actor0", 775),
                ("3dpw:downtown_sitOnStairs_00:actor0", 325), ("3dpw:flat_packBags_00:actor0", 446),
                ("3dpw:downtown_runForBus_01:actor0", 185))
# Predeclared before any R0/R1 validation or test result was computed.
DECISION = {
    "static_distal": "R1-J0 out-of-plane (|f_GT| >= 0.5) wrist < 0 AND R1-J0 forearm < 0, both sequence-cluster intervals exclude 0",
    "rgb_specific": "R1-R0 out-of-plane wrist or R1-R0 forearm < 0 with interval excluding 0",
    "temporal_vs_j1": "R1-J1 upper-arm residual < 0 with interval excluding 0 AND R1 closes >= 50% of the J1-J0 upper-residual gap",
    "elbow_vs_b1": "R1-B1 elbow mean <= 0",
    "p95": "R1 wrist p95 <= B1 wrist p95 + 1.0 cm AND R1 elbow p95 <= B1 elbow p95 + 1.0 cm",
    "distributed": "R1-J0 forearm improves in >= 2/3 of sequences",
    "validity": "no unavailable R1 rows, rig invariants <= 1e-9",
    "r0_useful": "R0-J0 forearm, out-of-plane wrist or elbow < 0 with interval excluding 0",
    "CASE_A": "all of static_distal, rgb_specific, temporal_vs_j1, elbow_vs_b1, p95, distributed, validity",
    "CASE_B": "r0_useful AND NOT rgb_specific",
    "CASE_C": "static_distal holds but temporal_vs_j1 fails (trade-off persists), or temporal_vs_j1 holds while R1-J0 forearm "
              "or out-of-plane wrist is significantly worse",
    "CASE_D": "otherwise"}


def collect(split, bank, h0_arr, w73, jp, rp, target, rest, alignment, raw):
    rows, accounting = [], {}
    for sid in sequence_ids_in_split(bank, split):
        lifted = build_h0_lifted_sequence(bank, h0_arr, sid)
        yaw = current_root_orientation(lifted)
        positions = sequence_frame_positions(bank, sid)
        oracle = _matched_oracle_sequence(sid, raw, [bank.samples[p].frame_index for p in positions])
        acc = Counter()
        for order, (frame, orient, gt_frame, p) in enumerate(zip(lifted.frames, yaw, oracle.frames, positions)):
            if gt_frame.frame_index != frame.frame_index or bank.samples[p].frame_index != frame.frame_index:
                raise SystemExit(f"{sid}: frame alignment broken")
            key = (sid, frame.frame_index)
            pred = w73.get(key)
            flags = {"no_w73_prediction": pred is None,
                     **{f"h0_source_invalid:{n}": not frame.points[n].observation_valid for n in NAMES},
                     "root_orientation_unknown": not orient.known,
                     "gt_invalid": not all(gt_frame.points[n].observation_valid for n in NAMES)}
            first = next((k for k, v in flags.items() if v), None)
            if first:
                acc[f"excluded:{first}"] += 1
                continue
            plane = plane_for(bank, p)
            if plane is None:
                acc["excluded:degenerate_observed_plane"] += 1
                continue
            ev = evaluate([frame.points[n].position for n in NAMES], [gt_frame.points[n].position for n in NAMES],
                          orient.yaw_radians, alignment, target, rest, pred["chain_f_e1"], plane)
            if ev is None:
                acc["excluded:degenerate_chain_geometry"] += 1
                continue
            arms = {b: ev["arms"][c] for b, c in BASE.items()}
            for o in ORACLES:
                arms[o] = ev["arms"][o]
            sources = {**{g: (jp[f"{c}_elbow"][p], jp[f"{c}_wrist"][p]) for g, c in J.items()},
                       **{g: (rp[f"{c}_elbow"][p], rp[f"{c}_wrist"][p]) for g, c in R.items()}}
            for g, (elbow, wrist) in sources.items():
                m = j_measure(elbow, wrist, orient.yaw_radians, alignment)
                arms[g] = full_arm(g, m, m.direction, target, rest) if m is not None else full_arm(g, ev["h0"], (0.0, 0.0, 0.0), target, rest)
                acc[f"{g}:unmeasurable"] += int(m is None)
            ev["arms"] = arms
            conf = min(float(bank.arrays["input_2d"][p][j, 2]) for j in (11, 12, 13))
            r = row_record(sid, frame.frame_index, p, order, bank.samples[p].sample_id, conf, orient.yaw_radians, pred["chain_f_e1"], ev)
            for n, a in arms.items():
                if r["candidates"][n]["status"] == "known":
                    r["candidates"][n]["side_error_degrees"] = side_error(a, ev["gt"])
                acc[f"{n}:{a.status}"] += 1
                acc[f"{n}:side:{a.side_status}"] += 1
            # Correction accounting in the normalized camera-frame arm (anchor = J0).
            e0, w0 = rp["anchor_elbow"][p], rp["anchor_wrist"][p]
            et, wt = rp["target_elbow"][p], rp["target_wrist"][p]
            corr = {}
            for g, c in R.items():
                cc = rp[f"{c}_correction"][p]
                de, dw = rp[f"{c}_elbow"][p] - e0, rp[f"{c}_wrist"][p] - w0
                need = wt - w0
                corr[g] = {"upper_rotation_degrees": math.degrees(float(np.linalg.norm(cc[0:3]))),
                           "forearm_rotation_degrees": math.degrees(float(np.linalg.norm(cc[3:6]))),
                           "rho_change": float(rp[f"{c}_rho"][p] - rp["anchor_rho"][p]), "correction": [float(x) for x in cc],
                           "elbow_displacement": float(np.linalg.norm(de)), "wrist_displacement": float(np.linalg.norm(dw)),
                           "wrist_alignment_with_needed": (float(np.dot(dw, need) / (np.linalg.norm(dw) * np.linalg.norm(need)))
                                                           if np.linalg.norm(dw) > 1e-9 and np.linalg.norm(need) > 1e-9 else None),
                           "normalized_wrist_error_change": float(np.linalg.norm(rp[f"{c}_wrist"][p] - wt) - np.linalg.norm(w0 - wt)),
                           "normalized_elbow_error_change": float(np.linalg.norm(rp[f"{c}_elbow"][p] - et) - np.linalg.norm(e0 - et))}
            r["correction"] = corr
            r["geometry"] = {n: r["geometry"][n] for n in KEEP_GEOMETRY}
            acc["common_eligible"] += 1
            rows.append(r)
        accounting[sid] = dict(acc)
    return rows, accounting


def _correction_summary(rows, pairs_by_key):
    out = {}
    for g in R:
        c = [r["correction"][g] for r in rows]
        out[g] = {k: _stats(x[k] for x in c) for k in ("upper_rotation_degrees", "forearm_rotation_degrees", "rho_change",
                                                         "elbow_displacement", "wrist_displacement", "wrist_alignment_with_needed",
                                                         "normalized_wrist_error_change", "normalized_elbow_error_change")}
        out[g]["share_reducing_wrist_error"] = sum(x["normalized_wrist_error_change"] < 0 for x in c) / len(c)
        out[g]["share_reducing_elbow_error"] = sum(x["normalized_elbow_error_change"] < 0 for x in c) / len(c)
        d = []
        for a, b in pairs_by_key:
            ca, cb = np.array(a["correction"][g]["correction"]), np.array(b["correction"][g]["correction"])
            d.append({"upper_rotation_change_degrees": math.degrees(float(np.linalg.norm(cb[0:3] - ca[0:3]))),
                      "forearm_rotation_change_degrees": math.degrees(float(np.linalg.norm(cb[3:6] - ca[3:6]))),
                      "rho_logit_change": abs(float(cb[6] - ca[6]))})
        out[g]["adjacent_correction_change"] = {k: _stats(x[k] for x in d) for k in d[0]} if d else None
    return out


def _regime(rows, pairs, row_ok, pair_ok):
    g = [r for r in rows if row_ok(r)]
    p = [x for x in pairs if pair_ok(x)]
    out = {"rows": len(g), "pairs": len(p), "sequences": len({r["sequence_id"] for r in g})}
    if not g:
        return out
    names = ("b0", "b1", "j0", "j1", "r0", "r1", "c111")
    out["static_means"] = {n: {m: sum(r["candidates"][n][m] for r in g) / len(g) for m in
                               ("wrist_error", "elbow_error", "forearm_direction_error_degrees", "chain_direction_error_degrees")} for n in names}
    keep = ("mean", "sequence_weighted_mean", "sequences_negative", "sequences_positive", "most_negative_sequence", "most_positive_sequence")
    out["contrasts"] = {f"{a}-{b}": {m: {k: v for k, v in _cmp(g, a, b, m, False).items() if k in keep}
                                     for m in ("wrist_error", "elbow_error", "forearm_direction_error_degrees")}
                        for a, b in (("r1", "j0"), ("r1", "j1"), ("r1", "r0"), ("r1", "b1"), ("r0", "j0"))}
    out["correction"] = {gg: {k: sum(r["correction"][gg][k] for r in g) / len(g) for k in ("upper_rotation_degrees", "forearm_rotation_degrees",
                                                                                         "wrist_displacement", "normalized_wrist_error_change",
                                                                                         "normalized_elbow_error_change")} for gg in R}
    if p:
        out["temporal_means"] = {n: {m: sum(x["candidates"][n][m] for x in p) / len(p) for m in (
            "upper_direction_residual_degrees", "disc:proximal_rotation_change_degrees")} for n in names}
        out["temporal_contrasts"] = {f"{a}-{b}": {k: v for k, v in _cmp(p, a, b, "upper_direction_residual_degrees", False).items() if k in keep}
                                     for a, b in (("r1", "j1"), ("r1", "j0"), ("r1", "r0"))}
    return out


def _decision(static_cmp, temporal_cmp, oop_cmp, table, coverage):
    def holds(c, sign=-1):
        return c["mean"] * sign > 0 and _excludes_zero(c)
    up = {n: table[n]["upper_direction_residual_degrees"]["mean"] for n in ("j0", "j1", "r1")} if "upper_direction_residual_degrees" in table["r1"] else None
    gap_closed = (up["j1"] - up["r1"]) / (up["j1"] - up["j0"]) if up and up["j1"] != up["j0"] else None
    c = {"static_distal": holds(oop_cmp["r1-j0"]) and holds(static_cmp["r1-j0"]["forearm_direction_error_degrees"]),
         "rgb_specific": holds(oop_cmp["r1-r0"]) or holds(static_cmp["r1-r0"]["forearm_direction_error_degrees"]),
         "temporal_vs_j1": holds(temporal_cmp["r1-j1"]["upper_direction_residual_degrees"]) and gap_closed is not None and gap_closed >= 0.5,
         "temporal_gap_closed_fraction": gap_closed,
         "elbow_vs_b1": static_cmp["r1-b1"]["elbow_error"]["mean"] <= 0,
         "p95": table["r1"]["wrist_error"]["p95"] <= table["b1"]["wrist_error"]["p95"] + 1.0
         and table["r1"]["elbow_error"]["p95"] <= table["b1"]["elbow_error"]["p95"] + 1.0,
         "distributed": static_cmp["r1-j0"]["forearm_direction_error_degrees"]["sequences_negative"]
         >= 2 / 3 * static_cmp["r1-j0"]["forearm_direction_error_degrees"]["sequences"],
         "validity": coverage,
         "r0_useful": holds(static_cmp["r0-j0"]["forearm_direction_error_degrees"]) or holds(oop_cmp["r0-j0"])
         or holds(static_cmp["r0-j0"]["elbow_error"]),
         "distal_worse": holds(static_cmp["r1-j0"]["forearm_direction_error_degrees"], 1) or holds(oop_cmp["r1-j0"], 1)}
    a_keys = ("static_distal", "rgb_specific", "temporal_vs_j1", "elbow_vs_b1", "p95", "distributed", "validity")
    if all(c[k] for k in a_keys):
        case = "A"
    elif c["r0_useful"] and not c["rgb_specific"]:
        case = "B"
    elif (c["static_distal"] and not c["temporal_vs_j1"]) or (c["temporal_vs_j1"] and c["distal_worse"]):
        case = "C"
    else:
        case = "D"
    return {**c, "mechanical_case": case}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("rig", "rest", "calibration", "chains", "bank", "h0_dir", "threedpw_raw", "w73_dir", "w76_dir",
                 "w80_joint_dir", "w80_eval_dir", "w81_dir", "correction_dir", "out_dir"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    args = parser.parse_args()
    identities = {"w76_report": _sha(args.w76_dir / "report.json"), "w80_predictions": _sha(args.w80_joint_dir / "predictions_3dpw.npz"),
                  "w80_eval_report": _sha(args.w80_eval_dir / "report.json"), "w80_rows_test": _sha(args.w80_eval_dir / "rows_test.json"),
                  "w80_pairs_test": _sha(args.w80_eval_dir / "pairs_test.json"), "w81_report": _sha(args.w81_dir / "report.json"),
                  "correction_training_report": _sha(args.correction_dir / "training_report.json"),
                  "correction_predictions": _sha(args.correction_dir / "predictions_3dpw.npz"),
                  **{f"correction_checkpoint_{g}": _sha(args.correction_dir / f"{c}.pt") for g, c in R.items()}}
    for k, exp in PINNED.items():
        if identities[k] != exp:
            raise SystemExit(f"{k} identity mismatch: {identities[k]}")
    training = json.loads((args.correction_dir / "training_report.json").read_text())
    if training["predictions_npz_sha256"] != identities["correction_predictions"] or not training["anchor"]["frozen"]:
        raise SystemExit("correction predictions / frozen-anchor record inconsistent")
    w76 = json.loads((args.w76_dir / "report.json").read_text())
    w80 = json.loads((args.w80_eval_dir / "report.json").read_text())
    identities["h0_test"] = _sha(args.h0_dir / "prediction_test.npy")
    if w76["identities"]["h0_test"] != identities["h0_test"]:
        raise SystemExit("frozen H0 test predictions differ from Worklog 76")
    rest = load_target_rest_pose(args.rest)
    if rest.provenance["source_fbx_sha256"] != _sha(args.rig):
        raise SystemExit("Blender rest snapshot differs from current FBX bytes")
    alignment = derive_rig_alignment(rest, load_rig_fk_calibration(args.calibration))
    target = target_chain_geometry(rest, load_ik_chain_set(args.chains).chains[0])
    if target.to_dict() != w76["target_chain"]:
        raise SystemExit("target chain differs from Worklog 76")
    bank = load_bank(args.bank)
    if training["identities"]["bank_content_digest"] != bank.content_digest():
        raise SystemExit("correction was trained on another FrameBank")
    h0_arr, _ = assemble_h0(bank, {"validation": args.h0_dir / "prediction_validation.npy", "test": args.h0_dir / "prediction_test.npy"})
    w73 = {(r["sequence_id"], r["frame_index"]): r for r in json.loads((args.w73_dir / "rows_heldout.json").read_text())}
    jp = dict(np.load(args.w80_joint_dir / "predictions_3dpw.npz"))
    rp = dict(np.load(args.correction_dir / "predictions_3dpw.npz"))
    rows, accounting = collect("test", bank, h0_arr, w73, jp, rp, target, rest, alignment, args.threedpw_raw)

    # --- Gate 1: historical parity ---
    old_rows = json.loads((args.w80_eval_dir / "rows_test.json").read_text())
    old = {(r["sequence_id"], r["frame_index"]): r for r in old_rows}
    mine = {(r["sequence_id"], r["frame_index"]): r for r in rows}
    g1 = {"rows": len(rows), "identity_set_equal": set(mine) == set(old), "sequences": len({k[0] for k in mine}),
          "max_abs_diff": defaultdict(float), "mismatch": Counter()}
    m = g1["max_abs_diff"]
    m["anchor_vs_w80_j0_raw"] = float(np.abs(rp["anchor_raw"] - jp["J0_GEOMETRY_ONLY_raw"]).max())
    for key, r in mine.items():
        o = old.get(key)
        if o is None:
            continue
        for n in ("b0", "b1", "j0", "j1"):
            a, b = r["geometry"][n], o["geometry"][n]
            for k in ("status", "side_status", "clamp", "reach_fraction", "direction", "side"):
                g1["mismatch"][f"{n}:{k}"] += a[k] != b[k]
            m[f"{n}:joints_cm"] = max(m[f"{n}:joints_cm"], math.dist(a["end"], b["end"]), math.dist(a["mid"], b["mid"]))
            m[f"{n}:rotations"] = max(m[f"{n}:rotations"], _qdiff(a["proximal_local"], b["proximal_local"]), _qdiff(a["distal_local"], b["distal_local"]))
            for metric in STATIC:
                x, y = r["candidates"][n].get(metric), o["candidates"][n].get(metric)
                if x is not None and y is not None:
                    m[f"{n}:metrics"] = max(m[f"{n}:metrics"], abs(x - y))
        m["reference_cm"] = max(m["reference_cm"], math.dist(r["reference"]["mid"], o["reference"]["mid"]), math.dist(r["e_gt"], o["e_gt"]))
    thresholds = w76["test"]["pooled"]["temporal"]["thresholds_gt_p99"]
    if thresholds != w80["jump_thresholds_worklog76"]:
        raise SystemExit("STOP: jump thresholds differ from Worklog 80")
    pairs = pair_records(rows, thresholds, KEEP_GEOMETRY)
    old_pairs = json.loads((args.w80_eval_dir / "pairs_test.json").read_text())
    g1["pairs"] = len(pairs)
    g1["pairs_identical"] = [(x["sequence_id"], x["frame_index"], x["next_frame_index"]) for x in pairs] == \
        [(x["sequence_id"], x["frame_index"], x["next_frame_index"]) for x in old_pairs]
    if g1["pairs_identical"]:
        for x, o in zip(pairs, old_pairs):
            for n in ("b0", "b1", "j0", "j1"):
                for t in TEMPORAL:
                    m[f"{n}:temporal"] = max(m[f"{n}:temporal"], abs(x["candidates"][n][t] - o["candidates"][n][t]))
    limits = {k: (0.0 if k.endswith("joints_cm") or k == "reference_cm" or k == "anchor_vs_w80_j0_raw" else 1e-9) for k in m}
    limits["anchor_vs_w80_j0_raw"] = 1e-6   # GPU forward of the frozen J0 (recorded in the training report)
    failed = {k: v for k, v in m.items() if v > limits[k]}
    g1.update({"limits": limits, "max_abs_diff": dict(m), "mismatch": {k: v for k, v in g1["mismatch"].items() if v}})
    g1["passed"] = (g1["identity_set_equal"] and g1["rows"] == 6003 and g1["sequences"] == 37 and g1["pairs"] == 5504
                    and g1["pairs_identical"] and not failed and not g1["mismatch"])
    if not g1["passed"]:
        raise SystemExit(f"STOP: historical parity failed: {json.dumps(g1, default=str)}")

    # --- Gate 2: corrected geometry ---
    g2 = {"unavailable": {n: sum(r["candidates"][n]["status"] != "known" for r in rows) for n in CANDIDATES},
          "side_status": {g: dict(Counter(r["geometry"][g]["side_status"] for r in rows)) for g in (*J, *R)},
          "clamps": {g: dict(Counter(r["geometry"][g]["clamp"] for r in rows)) for g in (*J, *R)},
          "integrity_max": {g: max(r["candidates"][g][k] for r in rows for k in ("integrity_upper_length_residual", "integrity_forearm_length_residual",
                                                                               "integrity_endpoint_reconstruction_residual", "integrity_ik_to_fk_residual")
                                   if r["candidates"][g]["status"] == "known") for g in R},
          "unit_arm_max_deviation": {c: float(np.abs(np.linalg.norm(rp[f"{c}_elbow"], axis=1) + np.linalg.norm(rp[f"{c}_wrist"] - rp[f"{c}_elbow"], axis=1) - 1).max())
                                     for c in R.values()},
          "anchor_frozen": training["anchor"]["frozen"], "initial_identity": {c: training["candidates"][c]["initial_identity_max_abs"] for c in R.values()}}
    g2["passed"] = (all(v <= 1e-9 for v in g2["integrity_max"].values()) and all(v <= 1e-9 for v in g2["unit_arm_max_deviation"].values())
                    and all(g2["unavailable"][g] == 0 for g in R) and g2["anchor_frozen"])
    if not g2["passed"]:
        raise SystemExit(f"STOP: corrected geometry violates the connected-arm / IK contract: {json.dumps(g2, default=str)}")
    common = [r for r in rows if all(r["candidates"][n]["status"] == "known" for n in CANDIDATES)]

    # --- Evaluation ---
    table = _table(common, CANDIDATES, STATIC)
    static_cmp = {f"{a}-{b}": {mm: _cmp(common, a, b, mm) for mm in PAIRED_STATIC} for a, b in COMPARISONS}
    oop = [r for r in common if abs(r["f_gt"]) >= 0.5]
    oop_cmp = {f"{a}-{b}": _cmp(oop, a, b, "wrist_error") for a, b in COMPARISONS}
    oop_forearm = {f"{a}-{b}": _cmp(oop, a, b, "forearm_direction_error_degrees") for a, b in COMPARISONS}
    temporal_cmp = {f"{a}-{b}": {mm: _cmp(pairs, a, b, mm) for mm in TEMPORAL} for a, b in COMPARISONS}
    ttable = _table(pairs, KEEP_GEOMETRY, (*RESIDUAL_METRICS, *TEMPORAL_METRICS, "reach_change", "side_change_degrees", "chain_direction_change_degrees"))
    regimes = {"gt_bend": {lab: _regime(common, pairs, lambda r, lo=lo, hi=hi: lo <= r["gt_bend_degrees"] < hi,
                                        lambda x, lo=lo, hi=hi: all(lo <= v < hi for v in x["gt_bend_both"])) for lab, (lo, hi) in BEND_REGIMES.items()},
               "abs_gt_f": {lab: _regime(common, pairs, lambda r, lo=lo, hi=hi: lo <= abs(r["f_gt"]) < hi,
                                         lambda x, lo=lo, hi=hi: all(lo <= v < hi for v in x["abs_f_gt_both"])) for lab, (lo, hi) in F_BINS.items()}}
    row_by = {(r["sequence_id"], r["frame_index"]): r for r in common}
    pair_rows = [(row_by[(x["sequence_id"], x["frame_index"])], row_by[(x["sequence_id"], x["next_frame_index"])]) for x in pairs]
    per_seq = {}
    for sid in sorted({r["sequence_id"] for r in common}):
        g = [r for r in common if r["sequence_id"] == sid]
        p = [x for x in pairs if x["sequence_id"] == sid]
        per_seq[sid] = {"rows": len(g), "pairs": len(p),
                        **{f"{a}-{b}_{mm}": sum(r["candidates"][a][mm] - r["candidates"][b][mm] for r in g) / len(g)
                           for a, b in (("r1", "j0"), ("r1", "j1"), ("r1", "r0"), ("r1", "b1")) for mm in ("elbow_error", "wrist_error", "forearm_direction_error_degrees")},
                        **{f"{a}-{b}_upper_residual": (sum(x["candidates"][a]["upper_direction_residual_degrees"] - x["candidates"][b]["upper_direction_residual_degrees"]
                                                           for x in p) / len(p)) if p else None for a, b in (("r1", "j0"), ("r1", "j1"), ("r1", "r0"))}}
    report = {
        "schema": "animcv_framepose_anchored_arm_correction_eval_v1", "identities": identities,
        "labels": {"runtime": "B0, B1, J0, J1, R0, R1", "evaluation_only": "C111, C_GT_F, GT reference"},
        "representation": training["representation"], "training_contract": training["training"], "anchor": training["anchor"],
        "anchor_audit": training["anchor_audit"], "control_identity": training["control_identity"], "eligibility": training["eligibility"],
        "training_summary": {c: {k: training["candidates"][c][k] for k in ("selection", "final", "steps", "draws", "source_draws", "parameters",
                                                                         "init_digest", "initial_identity_max_abs")} for c in training["candidates"]},
        "training_curves": {c: training["candidates"][c]["curve"] for c in training["candidates"]},
        "gate1_historical_parity": g1, "gate2_corrected_geometry": g2, "accounting": accounting,
        "bootstrap_contract": BOOTSTRAP, "decision_contract": DECISION, "jump_thresholds_worklog76": thresholds,
        "static": {"candidates": table, "comparisons": static_cmp, "out_of_plane_wrist": oop_cmp, "out_of_plane_forearm": oop_forearm,
                   "out_of_plane_table": _table(oop, CANDIDATES, ("wrist_error", "elbow_error", "forearm_direction_error_degrees"))},
        "temporal": {"pairs": len(pairs), "candidates": ttable | {"disc_rates": {n: {t: sum(x["candidates"][n][f"disc:{t}"] for x in pairs) / len(pairs)
                                                                                  for t in TEMPORAL_METRICS} for n in KEEP_GEOMETRY}},
                     "comparisons": temporal_cmp},
        "correction": _correction_summary(common, pair_rows), "regimes": regimes, "per_sequence": per_seq,
        "owner_frames": {f"{s}#{f}": ({n: {k: mine[(s, f)]["candidates"][n].get(k) for k in ("wrist_error", "elbow_error", "forearm_direction_error_degrees",
                                                                                           "chain_direction_error_degrees", "reach_error", "side_error_degrees")}
                                       for n in CANDIDATES} | {"correction": mine[(s, f)]["correction"], "gt_bend": mine[(s, f)]["gt_bend_degrees"],
                                                               "f_gt": mine[(s, f)]["f_gt"]})
                         if (s, f) in mine else None for s, f in OWNER_FRAMES},
    }
    report["decision_inputs"] = _decision(static_cmp, temporal_cmp, oop_cmp, {**table, **{n: {**table[n], **ttable[n]} for n in KEEP_GEOMETRY}},
                                          g2["unavailable"]["r1"] == 0 and g2["integrity_max"]["r1"] <= 1e-9)
    vrows, _ = collect("validation", bank, h0_arr, w73, jp, rp, target, rest, alignment, args.threedpw_raw)
    report["validation"] = {"note": "the split that selected the R epochs (selection-contaminated); temporal pairs are sparse", "rows": len(vrows),
                            "static": _table(vrows, ("b1", "j0", "j1", "r0", "r1"), ("wrist_error", "elbow_error", "forearm_direction_error_degrees")),
                            "comparisons": {f"{a}-{b}": {mm: _cmp(vrows, a, b, mm, False) for mm in ("wrist_error", "elbow_error", "forearm_direction_error_degrees")}
                                            for a, b in (("r1", "r0"), ("r1", "j0"), ("r1", "j1"))}}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "rows_test.json").write_text(json.dumps(rows))
    (args.out_dir / "pairs_test.json").write_text(json.dumps(pairs))
    path = args.out_dir / "report.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True, default=str))
    print(json.dumps({"sha256": _sha(path), "gate1": {k: g1[k] for k in ("rows", "pairs", "passed")}, "gate2": g2,
                      "decision": report["decision_inputs"]}, indent=1, default=str))


if __name__ == "__main__":
    main()
