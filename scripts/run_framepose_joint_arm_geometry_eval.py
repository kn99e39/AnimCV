#!/usr/bin/env python3
"""EVALUATION ONLY (Worklog 80): one-stage joint arm geometry (J0/J1) vs the historical arm baselines.

Candidates on the exact Worklog 76-79 test population (6,003 rows, 5,504 pairs):
  B0 = Worklog 77 C000 (H0 direction / reach / side), B1 = Worklog 77 C100 (E1 direction + H0 reach / side);
  J0 / J1 = the predicted connected arm (S = 0, E, W in the canonical camera frame) passed through the
  unchanged Worklog 62 measure_chain (A * Rz(-yaw_H0)) and Worklog 76 full_arm, exactly like H0's joints.
Evaluation-only decomposition (never runtime): each J-derived component (direction D, reach R,
bend side S) replaced by the GT one - the 2x2x2 corners between J (JJJ) and the GT reference (GGG) -
and "J direction + H0 reach/side" (the B1 analogue with J's direction).  Oracles C111 / C_GT_F are
reported as Worklog 77 headroom.  Metrics, temporal contract, Worklog 76 thresholds and sequence-aware
statistics are the Worklog 77/78 ones.

Order: identities -> Gate 1 (data identity: J training targets from the FrameBank equal the 3DPW raw GT
arm used by the reference) -> Gate 2 (B0/B1, population, pairs, thresholds, discontinuity rates equal
Worklog 77/78) -> Gate 3 (J output geometry valid and IK-compatible) -> evaluation.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_framepose_arm_geometry_owner_eval import _cmp, _table, pair_records, side_error  # noqa: E402
from run_framepose_ik_gt_attribution import _stats  # noqa: E402
from run_framepose_visual_arm_ownership import _contrast, _excludes_zero, evaluate, row_record  # noqa: E402
from run_framepose_visual_full_arm_ik import BEND_REGIMES, BOOTSTRAP, NAMES, STABLE, _adjacent_pairs, _paired, _qdiff, _sha, plane_for  # noqa: E402
from run_root_motion_contact_h0_replay import _matched_oracle_sequence  # noqa: E402

from framepose.bank import load_bank  # noqa: E402
from motion.animation_semantics_v2_bridge import current_root_orientation  # noqa: E402
from pose.framepose_bridge import assemble_h0, build_h0_lifted_sequence, sequence_frame_positions, sequence_ids_in_split  # noqa: E402
from retarget.framepose_fk import derive_rig_alignment, load_rig_fk_calibration  # noqa: E402
from retarget.framepose_ik_gt_attribution import measure_chain  # noqa: E402
from retarget.framepose_target_rest import load_target_rest_pose  # noqa: E402
from retarget.framepose_two_bone_ik import load_ik_chain_set, target_chain_geometry  # noqa: E402
from retarget.framepose_visual_arm_ownership import RESIDUAL_METRICS  # noqa: E402
from retarget.framepose_visual_full_arm_ik import TEMPORAL_METRICS, full_arm  # noqa: E402

PINNED = {"w73_rows_heldout": "4e17bd77399e4c016098f70592d0cfd68f508245ae60914869f756605d9897fc",
          "w76_report": "dad46288436ac0fe28fea9fab7ad6882c6a39fd476e717af4eaee497ab096a2b",
          "w77_report": "e86b7de61b934e6b74383df39ab4cf66fc13a185b439a12a21e66aa40714f29a",
          "w77_rows_test": "c39a834c978544eebdfc350cf920c7aa289398e015401763f4baaa0042839a74",
          "w78_eval_report": "8f37d6cd497c5f5b0699271b4d7da77c55094be0aa7d04a6d630a8e6db7f1ac1",
          "w79_comparison_report": "6d9ef090ce0b7e717175dcf73e6c5747dc76e7b751e309d9e0b2603f82921170"}
TOL_CM, TOL_QUAT = 1e-9, 1e-9
GT_CONVENTION_TOLERANCE = 1e-5   # float32 FrameBank targets vs float64 3DPW raw GT (Worklog 73 used 1e-5 for f)
J = {"j0": "J0_GEOMETRY_ONLY", "j1": "J1_GEOMETRY_PLUS_VISION"}
BASE = {"b0": "c000", "b1": "c100"}
ORACLES = ("c111", "c_gtf_h0r_h0s")
# Evaluation-only GT decomposition corners: (direction, reach, side) sources; "j" = the model, "g" = GT.
CORNERS = {f"d{d}r{r}s{s}": (d, r, s) for d in "jg" for r in "jg" for s in "jg" if (d, r, s) != ("j", "j", "j")}
DECOMP = tuple(f"{g}_{c}" for g in J for c in CORNERS) + tuple(f"{g}_dir_h0rs" for g in J)
CANDIDATES = ("b0", "b1", *J, *ORACLES)
KEEP_GEOMETRY = ("b0", "b1", "j0", "j1", "c111", "c_gtf_h0r_h0s")
STATIC = ("wrist_error", "elbow_error", "upper_direction_error_degrees", "forearm_direction_error_degrees",
          "chain_direction_error_degrees", "reach_error", "side_error_degrees", "abs_bend_angle_error_degrees",
          "bend_plane_error_degrees", "proximal_rotation_vs_gt_degrees", "distal_rotation_vs_gt_degrees")
PAIRED_STATIC = ("wrist_error", "elbow_error", "upper_direction_error_degrees", "forearm_direction_error_degrees",
                 "chain_direction_error_degrees", "reach_error", "side_error_degrees", "abs_bend_angle_error_degrees")
TEMPORAL = ("upper_direction_residual_degrees", "proximal_rotation_residual_degrees", "distal_rotation_residual_degrees",
            "forearm_direction_residual_degrees", "chain_direction_residual_degrees", "elbow_motion_residual",
            "wrist_motion_residual", "disc:proximal_rotation_change_degrees", "disc:elbow_displacement",
            "disc:distal_rotation_change_degrees", "disc:wrist_displacement", "reach_change", "side_change_degrees",
            "chain_direction_change_degrees")
COMPARISONS = (("j1", "b1"), ("j1", "b0"), ("j1", "j0"), ("j0", "b1"), ("j0", "b0"))
F_BINS = {"near_plane [0,sin10)": (0, STABLE), "intermediate": (STABLE, 0.5), "out_of_plane [0.5,1]": (0.5, 1.01)}
OWNER_FRAMES = (("3dpw:downtown_bar_00:actor0", 514), ("3dpw:downtown_bar_00:actor0", 775),
                ("3dpw:downtown_sitOnStairs_00:actor0", 325), ("3dpw:flat_packBags_00:actor0", 446),
                ("3dpw:downtown_runForBus_01:actor0", 185))
# Predeclared before any J0/J1 validation or test result was computed.
DECISION = {
    "static_elbow": "J-B1 elbow mean < 0, sequence-cluster 95% interval excludes 0, improves in >= 2/3 of sequences",
    "temporal": "J-B1 upper-arm motion residual < 0 AND J-B1 proximal discontinuity rate < 0, both intervals exclude 0",
    "wrist_forearm_preserved": "J-B1 wrist mean <= +0.5 cm, forearm mean <= +1.0 deg, wrist p95 <= B1 p95 + 1.0 cm, "
                               "forearm p95 <= B1 p95 + 2.0 deg",
    "near_folded": "J near-folded (GT bend >= 90) elbow mean <= B1 near-folded elbow mean",
    "coverage": "J unavailable rows <= B1's (0) and rig invariants <= 1e-9",
    "vision": "J1-J0 elbow (static) or upper-arm residual (temporal) < 0 with interval excluding 0",
    "CASE_A": "J1 meets static_elbow, temporal, wrist_forearm_preserved, near_folded, coverage, and (vision holds or J0 "
              "does not meet the same core criteria)",
    "CASE_B": "J0 meets static_elbow, temporal, wrist_forearm_preserved, coverage and vision does not hold",
    "CASE_C": "a candidate meets static_elbow but its upper residual is worse than B1 (interval excludes 0), or meets "
              "temporal while failing static_elbow or wrist_forearm_preserved with a significant elbow/wrist regression",
    "CASE_D": "otherwise"}


def j_measure(elbow, wrist, yaw, alignment):
    return measure_chain((0.0, 0.0, 0.0), tuple(float(v) for v in elbow), tuple(float(v) for v in wrist), yaw, alignment)


def decomposition_arm(name, mj, gt, corner, target, rest):
    d, r, s = corner
    m = mj
    if r == "g":
        m = replace(m, reach_fraction=gt.reach_fraction)
    if s == "g":
        m = replace(m, side=gt.side, degenerate=gt.degenerate)
    return full_arm(name, m, gt.direction if d == "g" else mj.direction, target, rest)


def collect(split, bank, h0_arr, w73, preds, target, rest, alignment, raw, gate):
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
            acc["bank_rows"] += 1
            pred = w73.get(key)
            flags = {"no_w73_prediction": pred is None,
                     **{f"h0_source_invalid:{n}": not frame.points[n].observation_valid for n in NAMES},
                     "root_orientation_unknown": not orient.known,
                     "gt_invalid": not all(gt_frame.points[n].observation_valid for n in NAMES)}
            first = next((k for k, v in flags.items() if v), None)
            if first:
                acc[f"excluded:{first}"] += 1
                continue
            if pred["position"] != p or pred["sample_id"] != bank.samples[p].sample_id:
                raise SystemExit(f"{key}: Worklog 73 row does not match the FrameBank sample")
            plane = plane_for(bank, p)
            if plane is None:
                acc["excluded:degenerate_observed_plane"] += 1
                continue
            h0_pos = [frame.points[n].position for n in NAMES]
            gt_pos = [np.asarray(gt_frame.points[n].position, float) for n in NAMES]
            ev = evaluate(h0_pos, [tuple(x) for x in gt_pos], orient.yaw_radians, alignment, target, rest, pred["chain_f_e1"], plane)
            if ev is None:
                acc["excluded:degenerate_chain_geometry"] += 1
                continue
            # Gate 1 bookkeeping: J's FrameBank training target vs the raw-GT arm used by the reference.
            total = np.linalg.norm(gt_pos[1] - gt_pos[0]) + np.linalg.norm(gt_pos[2] - gt_pos[1])
            if preds["target_valid"][p]:
                gate["target_vs_raw_gt"] = max(gate["target_vs_raw_gt"],
                                               float(np.abs(preds["target_elbow"][p] - (gt_pos[1] - gt_pos[0]) / total).max()),
                                               float(np.abs(preds["target_wrist"][p] - (gt_pos[2] - gt_pos[0]) / total).max()))
            else:
                gate["target_invalid_rows"] += 1
            arms = {b: ev["arms"][c] for b, c in BASE.items()}
            for o in ORACLES:
                arms[o] = ev["arms"][o]
            for g, cand in J.items():
                mj = j_measure(preds[f"{cand}_elbow"][p], preds[f"{cand}_wrist"][p], orient.yaw_radians, alignment)
                if mj is None:
                    acc[f"{g}:unmeasurable"] += 1
                    arms[g] = full_arm(g, ev["h0"], (0.0, 0.0, 0.0), target, rest)  # status unavailable:degenerate_direction
                    continue
                arms[g] = full_arm(g, mj, mj.direction, target, rest)
                for c, corner in CORNERS.items():
                    arms[f"{g}_{c}"] = decomposition_arm(f"{g}_{c}", mj, ev["gt"], corner, target, rest)
                arms[f"{g}_dir_h0rs"] = full_arm(f"{g}_dir_h0rs", ev["h0"], mj.direction, target, rest)
            ev["arms"] = arms
            conf = min(float(bank.arrays["input_2d"][p][j, 2]) for j in (11, 12, 13))
            r = row_record(sid, frame.frame_index, p, order, bank.samples[p].sample_id, conf, orient.yaw_radians,
                           pred["chain_f_e1"], ev)
            for n, a in arms.items():
                if r["candidates"][n]["status"] == "known":
                    r["candidates"][n]["side_error_degrees"] = side_error(a, ev["gt"])
                acc[f"{n}:{a.status}"] += 1
            for n in ("j0", "j1"):
                acc[f"{n}:side:{arms[n].side_status}"] += 1
            r["j_rho"] = {g: float(preds[f"{c}_rho"][p]) for g, c in J.items()}
            r["geometry"] = {n: r["geometry"][n] for n in KEEP_GEOMETRY if n in r["geometry"]}
            acc["common_eligible"] += 1
            rows.append(r)
        accounting[sid] = dict(acc)
    return rows, accounting


def _regime(rows, pairs, row_ok, pair_ok):
    g = [r for r in rows if row_ok(r)]
    p = [x for x in pairs if pair_ok(x)]
    out = {"rows": len(g), "pairs": len(p), "sequences": len({r["sequence_id"] for r in g})}
    if not g:
        return out
    names = ("b0", "b1", "j0", "j1", "c111")
    out["static_means"] = {n: {m: sum(r["candidates"][n][m] for r in g) / len(g) for m in
                               ("wrist_error", "elbow_error", "forearm_direction_error_degrees", "reach_error")} for n in names}
    keep = ("mean", "sequences_improved", "sequences", "largest_single_sequence_contribution", "worst_regressions")
    for a, b in (("j1", "b1"), ("j1", "j0"), ("j0", "b1")):
        out[f"{a}_vs_{b}_elbow"] = {k: v for k, v in _paired(g, a, b, "elbow_error").items() if k in keep}
    out["j1_vs_b1_wrist"] = {k: v for k, v in _paired(g, "j1", "b1", "wrist_error").items() if k in keep}
    out["j1_decomposition_elbow"] = {c: sum(r["candidates"][f"j1_{c}"]["elbow_error"] for r in g) / len(g) for c in CORNERS}
    if p:
        out["temporal_means"] = {n: {m: sum(x["candidates"][n][m] for x in p) / len(p) for m in (
            "upper_direction_residual_degrees", "disc:proximal_rotation_change_degrees")} for n in names}
        out["j1_vs_b1_upper_residual"] = {k: v for k, v in _contrast(p, lambda x: x["candidates"]["j1"]["upper_direction_residual_degrees"]
                                                                   - x["candidates"]["b1"]["upper_direction_residual_degrees"], False).items()
                                          if k in ("mean", "sequences_negative", "sequences")}
    return out


def _decomposition(rows, metric):
    out = {}
    for g in J:
        base = sum(r["candidates"][g][metric] for r in rows) / len(rows)
        corners = {c: sum(r["candidates"][f"{g}_{c}"][metric] for r in rows) / len(rows) for c in CORNERS}
        out[g] = {"model": base, "corners": corners,
                  "replace_direction_only": corners["dgrjsj"] - base, "replace_reach_only": corners["djrgsj"] - base,
                  "replace_side_only": corners["djrjsg"] - base, "replace_all_gt": corners["dgrgsg"] - base,
                  "j_direction_with_h0_reach_side": sum(r["candidates"][f"{g}_dir_h0rs"][metric] for r in rows) / len(rows)}
    return out


def _decision(static_cmp, temporal_cmp, table, regimes, coverage):
    def holds(c, sign=-1):
        return c["mean"] * sign > 0 and _excludes_zero(c)

    def criteria(g):
        el = static_cmp[f"{g}-b1"]["elbow_error"]
        out = {"static_elbow": holds(el) and el["sequences_negative"] >= 2 / 3 * el["sequences"],
               "temporal": holds(temporal_cmp[f"{g}-b1"]["upper_direction_residual_degrees"])
               and holds(temporal_cmp[f"{g}-b1"]["disc:proximal_rotation_change_degrees"]),
               "wrist_forearm_preserved": static_cmp[f"{g}-b1"]["wrist_error"]["mean"] <= 0.5
               and static_cmp[f"{g}-b1"]["forearm_direction_error_degrees"]["mean"] <= 1.0
               and table[g]["wrist_error"]["p95"] <= table["b1"]["wrist_error"]["p95"] + 1.0
               and table[g]["forearm_direction_error_degrees"]["p95"] <= table["b1"]["forearm_direction_error_degrees"]["p95"] + 2.0,
               "near_folded": regimes["near_folded [90,180]"]["static_means"][g]["elbow_error"]
               <= regimes["near_folded [90,180]"]["static_means"]["b1"]["elbow_error"],
               "coverage": coverage[g],
               "temporal_worse": holds(temporal_cmp[f"{g}-b1"]["upper_direction_residual_degrees"], sign=1),
               "elbow_worse": holds(el, sign=1), "wrist_worse": holds(static_cmp[f"{g}-b1"]["wrist_error"], sign=1)}
        return out
    j1, j0 = criteria("j1"), criteria("j0")
    vision = holds(static_cmp["j1-j0"]["elbow_error"]) or holds(temporal_cmp["j1-j0"]["upper_direction_residual_degrees"])
    core = ("static_elbow", "temporal", "wrist_forearm_preserved", "coverage")
    if all(j1[k] for k in (*core, "near_folded")) and (vision or not all(j0[k] for k in core)):
        case = "A"
    elif all(j0[k] for k in core) and not vision:
        case = "B"
    elif any((c["static_elbow"] and c["temporal_worse"]) or (c["temporal"] and (c["elbow_worse"] or c["wrist_worse"]))
             for c in (j1, j0)):
        case = "C"
    else:
        case = "D"
    return {"j1": j1, "j0": j0, "vision_contributes": vision, "mechanical_case": case}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("rig", "rest", "calibration", "chains", "bank", "h0_dir", "threedpw_raw", "w73_dir", "w76_dir", "w77_dir",
                 "w78_eval_dir", "w79_comparison_dir", "joint_dir", "out_dir"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    args = parser.parse_args()

    identities = {"w73_rows_heldout": _sha(args.w73_dir / "rows_heldout.json"), "w76_report": _sha(args.w76_dir / "report.json"),
                  "w77_report": _sha(args.w77_dir / "report.json"), "w77_rows_test": _sha(args.w77_dir / "rows_test.json"),
                  "w78_eval_report": _sha(args.w78_eval_dir / "report.json"),
                  "w79_comparison_report": _sha(args.w79_comparison_dir / "report.json"),
                  "joint_training_report": _sha(args.joint_dir / "training_report.json"),
                  "joint_predictions": _sha(args.joint_dir / "predictions_3dpw.npz"),
                  **{f"joint_checkpoint_{g}": _sha(args.joint_dir / f"{c}.pt") for g, c in J.items()}}
    for k, exp in PINNED.items():
        if identities[k] != exp:
            raise SystemExit(f"{k} identity mismatch: {identities[k]}")
    training = json.loads((args.joint_dir / "training_report.json").read_text())
    if training["predictions_npz_sha256"] != identities["joint_predictions"]:
        raise SystemExit("joint predictions differ from the training report")
    w76_report = json.loads((args.w76_dir / "report.json").read_text())
    identities["h0_test"] = _sha(args.h0_dir / "prediction_test.npy")
    if w76_report["identities"]["h0_test"] != identities["h0_test"]:
        raise SystemExit("frozen H0 test predictions differ from Worklog 76")
    rest = load_target_rest_pose(args.rest)
    if rest.provenance["source_fbx_sha256"] != _sha(args.rig):
        raise SystemExit("Blender rest snapshot differs from current FBX bytes")
    alignment = derive_rig_alignment(rest, load_rig_fk_calibration(args.calibration))
    target = target_chain_geometry(rest, load_ik_chain_set(args.chains).chains[0])
    if target.to_dict() != w76_report["target_chain"]:
        raise SystemExit("target chain differs from Worklog 76")
    bank = load_bank(args.bank)
    if training["identities"]["bank_content_digest"] != bank.content_digest():
        raise SystemExit("joint model was trained on another FrameBank")
    h0_arr, _ = assemble_h0(bank, {"validation": args.h0_dir / "prediction_validation.npy", "test": args.h0_dir / "prediction_test.npy"})
    w73 = {(r["sequence_id"], r["frame_index"]): r for r in json.loads((args.w73_dir / "rows_heldout.json").read_text())}
    preds = dict(np.load(args.joint_dir / "predictions_3dpw.npz"))
    if len(preds["target_valid"]) != len(bank):
        raise SystemExit("joint predictions do not cover the FrameBank")

    gate1 = defaultdict(float)
    rows, accounting = collect("test", bank, h0_arr, w73, preds, target, rest, alignment, args.threedpw_raw, gate1)
    gate1 = dict(gate1)
    gate1["passed"] = gate1["target_vs_raw_gt"] <= GT_CONVENTION_TOLERANCE and gate1.get("target_invalid_rows", 0) == 0
    if not gate1["passed"]:
        raise SystemExit(f"STOP: J training-target convention differs from the reference GT arm: {gate1}")

    # --- Gate 2: historical baselines ---
    w77_rows = json.loads((args.w77_dir / "rows_test.json").read_text())
    w77 = {(r["sequence_id"], r["frame_index"]): r for r in w77_rows}
    w77_report = json.loads((args.w77_dir / "report.json").read_text())
    mine = {(r["sequence_id"], r["frame_index"]): r for r in rows}
    g2 = {"rows": len(rows), "identity_set_equal": set(mine) == set(w77), "sequences": len({k[0] for k in mine}),
          "max_abs_diff": defaultdict(float), "mismatch": Counter()}
    m = g2["max_abs_diff"]
    for key, r in mine.items():
        old = w77.get(key)
        if old is None:
            continue
        for b, c in BASE.items():
            a, o = r["geometry"][b], old["geometry"][c]
            for k in ("status", "side_status", "clamp", "reach_fraction"):
                g2["mismatch"][f"{b}:{k}"] += a[k] != o[k]
            m[f"{b}:joints_cm"] = max(m[f"{b}:joints_cm"], math.dist(a["end"], o["end"]), math.dist(a["mid"], o["mid"]))
            m[f"{b}:rotations"] = max(m[f"{b}:rotations"], _qdiff(a["proximal_local"], o["proximal_local"]),
                                      _qdiff(a["distal_local"], o["distal_local"]))
            for metric in ("upper_direction_error_degrees", "forearm_direction_error_degrees"):
                m[f"{b}:{metric}"] = max(m[f"{b}:{metric}"], abs(r["candidates"][b][metric] - old["candidates"][c][metric]))
        m["reference_cm"] = max(m["reference_cm"], math.dist(r["reference"]["mid"], old["reference"]["mid"]), math.dist(r["e_gt"], old["e_gt"]))
        m["yaw"] = max(m["yaw"], abs(r["yaw_h0_radians"] - old["yaw_h0_radians"]))
    thresholds = w76_report["test"]["pooled"]["temporal"]["thresholds_gt_p99"]
    pairs = pair_records(rows, thresholds, KEEP_GEOMETRY)
    old_pairs, _ = _adjacent_pairs(w77_rows)
    g2["pairs"] = len(pairs)
    g2["pairs_identical"] = [(x["sequence_id"], x["frame_index"], x["next_frame_index"]) for x in pairs] == \
        [(a["sequence_id"], a["frame_index"], b["frame_index"]) for a, b in old_pairs]
    for b, c in BASE.items():
        for t in TEMPORAL_METRICS:
            m[f"{b}:disc_rate:{t}"] = abs(sum(x["candidates"][b][f"disc:{t}"] for x in pairs) / len(pairs)
                                          - w77_report["temporal"]["candidates"][c][f"disc:{t}"])
        for t in ("upper_direction_residual_degrees", "elbow_motion_residual"):
            m[f"{b}:{t}_mean"] = abs(sum(x["candidates"][b][t] for x in pairs) / len(pairs) - w77_report["temporal"]["candidates"][c][t]["mean"])
    limits = {k: (0.0 if "disc_rate" in k or k == "yaw" else 1e-9 if k.endswith("_mean") or "degrees" in k
                  else TOL_QUAT if "rotation" in k else TOL_CM) for k in m}
    failed = {k: v for k, v in m.items() if v > limits[k]}
    g2.update({"limits": limits, "max_abs_diff": dict(m), "mismatch": {k: v for k, v in g2["mismatch"].items() if v}})
    g2["passed"] = (g2["identity_set_equal"] and g2["rows"] == 6003 and g2["sequences"] == 37 and g2["pairs"] == 5504
                    and g2["pairs_identical"] and not failed and not g2["mismatch"])
    if not g2["passed"]:
        raise SystemExit(f"STOP: historical baseline parity failed: {json.dumps(g2, default=str)}")

    # --- Gate 3: J output geometry ---
    names = (*CANDIDATES, *DECOMP)
    g3 = {"unavailable": {n: sum(r["candidates"][n]["status"] != "known" for r in rows) for n in names},
          "clamps": {n: dict(Counter(str(r["candidates"][n].get("clamp")) for r in rows)) for n in ("b1", "j0", "j1")},
          "side_status": {g: dict(Counter(r["geometry"][g]["side_status"] for r in rows)) for g in J},
          "rho_range": {g: [float(min(r["j_rho"][g] for r in rows)), float(max(r["j_rho"][g] for r in rows))] for g in J},
          "integrity_max": {n: max(max(r["candidates"][n][k] for k in ("integrity_upper_length_residual", "integrity_forearm_length_residual",
                                                                       "integrity_endpoint_reconstruction_residual", "integrity_ik_to_fk_residual"))
                                   for r in rows if r["candidates"][n]["status"] == "known") for n in (*J, *DECOMP)},
          "nonfinite": {n: sum(not r["candidates"][n].get("integrity_finite", True) for r in rows) for n in J}}
    g3["passed"] = (all(v <= TOL_CM for v in g3["integrity_max"].values()) and not any(g3["nonfinite"].values())
                    and all(g3["unavailable"][g] == 0 for g in J))
    if not g3["passed"]:
        raise SystemExit(f"STOP: J output geometry cannot be mapped to the frozen IK contract: {json.dumps(g3, default=str)}")
    common = [r for r in rows if all(r["candidates"][n]["status"] == "known" for n in names)]

    # --- Evaluation ---
    table = _table(common, (*CANDIDATES, "j0_dir_h0rs", "j1_dir_h0rs"), STATIC)
    static_cmp = {f"{a}-{b}": {mm: _cmp(common, a, b, mm) for mm in PAIRED_STATIC} for a, b in COMPARISONS}
    static_cmp["j1_dir_h0rs-b1"] = {mm: _cmp(common, "j1_dir_h0rs", "b1", mm) for mm in ("wrist_error", "elbow_error", "chain_direction_error_degrees")}
    temporal_cmp = {f"{a}-{b}": {mm: _cmp(pairs, a, b, mm) for mm in TEMPORAL} for a, b in COMPARISONS}
    regimes = {"gt_bend": {lab: _regime(common, pairs, lambda r, lo=lo, hi=hi: lo <= r["gt_bend_degrees"] < hi,
                                        lambda x, lo=lo, hi=hi: all(lo <= v < hi for v in x["gt_bend_both"])) for lab, (lo, hi) in BEND_REGIMES.items()},
               "abs_gt_f": {lab: _regime(common, pairs, lambda r, lo=lo, hi=hi: lo <= abs(r["f_gt"]) < hi,
                                         lambda x, lo=lo, hi=hi: all(lo <= v < hi for v in x["abs_f_gt_both"])) for lab, (lo, hi) in F_BINS.items()}}
    conflict = [r for r in common if r["gt_bend_degrees"] >= 90 and r["candidates"]["b1"]["wrist_error"] < r["candidates"]["b0"]["wrist_error"]
                and r["candidates"]["b1"]["elbow_error"] > r["candidates"]["b0"]["elbow_error"]]
    conflict_block = {"rows": len(conflict)}
    if conflict:
        for n in ("b0", "b1", "j0", "j1", "c111"):
            conflict_block[f"{n}_elbow_mean"] = sum(r["candidates"][n]["elbow_error"] for r in conflict) / len(conflict)
            conflict_block[f"{n}_wrist_mean"] = sum(r["candidates"][n]["wrist_error"] for r in conflict) / len(conflict)
        conflict_block["j1_elbow_better_than_b1_share"] = sum(r["candidates"]["j1"]["elbow_error"] < r["candidates"]["b1"]["elbow_error"] for r in conflict) / len(conflict)
    coverage = {g: g3["unavailable"][g] <= g3["unavailable"]["b1"] and g3["integrity_max"][g] <= TOL_CM for g in J}
    per_seq = {}
    for sid in sorted({r["sequence_id"] for r in common}):
        g = [r for r in common if r["sequence_id"] == sid]
        p = [x for x in pairs if x["sequence_id"] == sid]
        per_seq[sid] = {"rows": len(g), "pairs": len(p),
                        **{f"{a}-{b}_{mm}": sum(r["candidates"][a][mm] - r["candidates"][b][mm] for r in g) / len(g)
                           for a, b in (("j1", "b1"), ("j1", "j0"), ("j0", "b1")) for mm in ("elbow_error", "wrist_error")},
                        **{f"{a}-{b}_upper_residual": (sum(x["candidates"][a]["upper_direction_residual_degrees"]
                                                           - x["candidates"][b]["upper_direction_residual_degrees"] for x in p) / len(p)) if p else None
                           for a, b in (("j1", "b1"), ("j1", "j0"), ("j0", "b1"))}}
    w78 = json.loads((args.w78_eval_dir / "report.json").read_text())
    w79 = json.loads((args.w79_comparison_dir / "report.json").read_text())
    history = {"w78_in_sample": {g: {mm: w78["static"]["candidates"][g][mm]["mean"] for mm in ("wrist_error", "elbow_error", "reach_error", "side_error_degrees")}
                                 | {"upper_residual": w78["temporal"]["candidates"][g]["upper_direction_residual_degrees"]["mean"],
                                    "disc_proximal": w78["temporal"]["candidates"]["disc_rates"][g]["proximal_rotation_change_degrees"]} for g in ("g0", "g1")},
               "w79_oof": {g: {mm: w79["static"]["candidates"][g][mm]["mean"] for mm in ("wrist_error", "elbow_error", "reach_error", "side_error_degrees")}
                           | {"upper_residual": w79["temporal"]["candidates"][g]["upper_direction_residual_degrees"]["mean"]} for g in ("g0_oof", "g1_oof")}}
    report = {
        "schema": "animcv_framepose_joint_arm_eval_v1", "identities": identities,
        "labels": {"runtime": "B0, B1, J0, J1", "evaluation_only": "J/GT decomposition corners, C111, C_GT_F, GT reference"},
        "representation": training["representation"], "training_contract": training["training"],
        "training_summary": {c: {k: training["candidates"][c][k] for k in ("selection", "final", "steps", "draws", "source_draws",
                                                                         "parameters", "init_digest")} for c in training["candidates"]},
        "training_curves": {c: training["candidates"][c]["curve"] for c in training["candidates"]},
        "control_identity": training["control_identity"], "eligibility": training["eligibility"],
        "gate1_data_identity": gate1, "gate2_historical_parity": g2, "gate3_joint_geometry": g3, "accounting": accounting,
        "bootstrap_contract": BOOTSTRAP, "decision_contract": DECISION, "jump_thresholds_worklog76": thresholds,
        "static": {"candidates": table, "comparisons": static_cmp,
                   "decomposition": {mm: _decomposition(common, mm) for mm in ("elbow_error", "wrist_error", "upper_direction_error_degrees",
                                                                              "forearm_direction_error_degrees")}},
        "temporal": {"pairs": len(pairs), "candidates": _table(pairs, KEEP_GEOMETRY, (*RESIDUAL_METRICS, *TEMPORAL_METRICS, "reach_change",
                                                                                     "side_change_degrees", "chain_direction_change_degrees"))
                     | {"disc_rates": {n: {t: sum(x["candidates"][n][f"disc:{t}"] for x in pairs) / len(pairs) for t in TEMPORAL_METRICS} for n in KEEP_GEOMETRY}},
                     "reference_reach_change": _stats(x["reference"]["reach_change"] for x in pairs),
                     "comparisons": temporal_cmp},
        "regimes": regimes, "near_folded_conflict_rows_worklog76": conflict_block, "per_sequence": per_seq,
        "history_context": history,
        "owner_frames": {f"{s}#{f}": ({n: {k: mine[(s, f)]["candidates"][n].get(k) for k in ("wrist_error", "elbow_error", "upper_direction_error_degrees",
                                                                                           "forearm_direction_error_degrees", "chain_direction_error_degrees",
                                                                                           "reach_error", "side_error_degrees")}
                                       for n in (*CANDIDATES, "j1_dgrjsj", "j1_djrgsg")} | {"gt_reach": mine[(s, f)]["gt_reach"], "gt_bend": mine[(s, f)]["gt_bend_degrees"]})
                         if (s, f) in mine else None for s, f in OWNER_FRAMES},
    }
    report["decision_inputs"] = _decision(static_cmp, temporal_cmp, table, regimes["gt_bend"], coverage)
    vg = defaultdict(float)
    vrows, _ = collect("validation", bank, h0_arr, w73, preds, target, rest, alignment, args.threedpw_raw, vg)
    vcommon = [r for r in vrows if all(r["candidates"][n]["status"] == "known" for n in names)]
    report["validation"] = {"note": "the split that selected the J epochs (selection-contaminated); temporal pairs are sparse",
                            "rows": len(vrows), "common": len(vcommon), "target_vs_raw_gt": vg.get("target_vs_raw_gt"),
                            "static": _table(vcommon, ("b0", "b1", "j0", "j1", "c111"), PAIRED_STATIC),
                            "comparisons": {f"{a}-{b}": {mm: _cmp(vcommon, a, b, mm, False) for mm in ("wrist_error", "elbow_error")} for a, b in COMPARISONS}}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "rows_test.json").write_text(json.dumps(rows))
    (args.out_dir / "pairs_test.json").write_text(json.dumps(pairs))
    path = args.out_dir / "report.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True, default=str))
    print(json.dumps({"sha256": _sha(path), "gate1": gate1, "gate2": {k: g2[k] for k in ("rows", "pairs", "passed")},
                      "gate3": {k: g3[k] for k in ("passed", "rho_range", "side_status", "clamps")},
                      "decision": report["decision_inputs"]}, indent=1, default=str))


if __name__ == "__main__":
    main()
