#!/usr/bin/env python3
"""EVALUATION ONLY (Worklog 78): learned reach / bend-side owner vs H0-owned geometry under the frozen E1 direction.

Candidates on the exact Worklog 76/77 test population (6,003 rows, 5,504 pairs):
  B0 = Worklog 77 C000 (H0 direction, H0 reach, H0 side)          - read-only baseline
  B1 = Worklog 77 C100 (E1 direction, H0 reach, H0 side)          - read-only baseline
  G0 / G1 = E1 direction + reach and side from the Worklog 78 learned owner (G0 zero vision, G1 Qwen vision)
  attribution arms: learned reach + H0 side, H0 reach + learned side (for G0 and G1)
  oracle headroom (evaluation only): C111 (E1 + GT reach + GT side), C_GT_F (GT f + H0 reach/side), GT reference.
All arms use the unchanged Worklog 76 ``full_arm`` and the Worklog 77 metrics, temporal
contract and Worklog 76 jump thresholds.  The learned side is a camera-frame vector moved to rig
space with the same A*Rz(-yaw_H0) as every direction; the Worklog 62 rule re-perpendicularizes it.

Order: identities -> Gate 1 (B0/B1, adjacency, thresholds and discontinuity rates equal Worklog 77/76;
frozen E1 inputs equal Worklog 73) -> Gate 2 (owner output geometry valid on every row) -> evaluation.
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
from run_framepose_ik_gt_attribution import _stats  # noqa: E402
from run_framepose_visual_arm_ownership import _contrast, _excludes_zero, evaluate, row_record  # noqa: E402
from run_framepose_visual_full_arm_ik import (  # noqa: E402
    BEND_REGIMES, BOOTSTRAP, NAMES, STABLE, _adjacent_pairs, _Arm, _paired, _qdiff, _sha, plane_for,
)
from run_root_motion_contact_h0_replay import _matched_oracle_sequence  # noqa: E402

from framepose.bank import load_bank  # noqa: E402
from motion.animation_semantics_v2_bridge import current_root_orientation  # noqa: E402
from pose.framepose_bridge import assemble_h0, build_h0_lifted_sequence, sequence_frame_positions, sequence_ids_in_split  # noqa: E402
from retarget.framepose_fk import derive_rig_alignment, load_rig_fk_calibration  # noqa: E402
from retarget.framepose_ik_gt_attribution import _perpendicular, heading_relative  # noqa: E402
from retarget.framepose_target_rest import load_target_rest_pose  # noqa: E402
from retarget.framepose_two_bone_ik import load_ik_chain_set, target_chain_geometry  # noqa: E402
from retarget.framepose_visual_arm_ownership import RESIDUAL_METRICS, motion_residual  # noqa: E402
from retarget.framepose_visual_full_arm_ik import (  # noqa: E402
    ATTRIBUTION_TEMPORAL_METRICS, TEMPORAL_METRICS, adjacent_change, classify_change, full_arm,
)

PINNED = {"w73_rows_heldout": "4e17bd77399e4c016098f70592d0cfd68f508245ae60914869f756605d9897fc",
          "w76_report": "dad46288436ac0fe28fea9fab7ad6882c6a39fd476e717af4eaee497ab096a2b",
          "w77_report": "e86b7de61b934e6b74383df39ab4cf66fc13a185b439a12a21e66aa40714f29a",
          "w77_rows_test": "c39a834c978544eebdfc350cf920c7aa289398e015401763f4baaa0042839a74"}
TOL_CM, TOL_QUAT = 1e-9, 1e-9
E1_INPUT_TOLERANCE = 1e-5
OWNERS_G = {"G0": "G0_GEOMETRY_ONLY", "G1": "G1_GEOMETRY_PLUS_VISION"}
# name -> (reach source, side source); direction is E1 for all of them.
LEARNED = {"g0": ("G0", "G0"), "g1": ("G1", "G1"), "g0r_h0s": ("G0", "h0"), "h0r_g0s": ("h0", "G0"),
           "g1r_h0s": ("G1", "h0"), "h0r_g1s": ("h0", "G1")}
BASE = {"b0": "c000", "b1": "c100"}
ORACLES = ("c111", "c110", "c101", "c_gtf_h0r_h0s")
CANDIDATES = ("b0", "b1", *LEARNED, *ORACLES)
KEEP_GEOMETRY = ("b0", "b1", "g0", "g1", "c111", "c_gtf_h0r_h0s")
STATIC = ("wrist_error", "elbow_error", "upper_direction_error_degrees", "forearm_direction_error_degrees",
          "chain_direction_error_degrees", "reach_error", "side_error_degrees", "abs_bend_angle_error_degrees",
          "bend_plane_error_degrees", "proximal_rotation_vs_gt_degrees", "distal_rotation_vs_gt_degrees")
PAIRED_STATIC = ("wrist_error", "elbow_error", "upper_direction_error_degrees", "forearm_direction_error_degrees",
                 "reach_error", "side_error_degrees")
TEMPORAL = ("upper_direction_residual_degrees", "proximal_rotation_residual_degrees", "distal_rotation_residual_degrees",
            "forearm_direction_residual_degrees", "elbow_motion_residual", "wrist_motion_residual",
            "disc:proximal_rotation_change_degrees", "disc:elbow_displacement", "disc:distal_rotation_change_degrees",
            "disc:wrist_displacement", "reach_change", "side_change_degrees")
COMPARISONS = (("g1", "b1"), ("g1", "g0"), ("g1", "b0"), ("g0", "b1"), ("g0", "b0"))
F_BINS = {"near_plane [0,sin10)": (0, STABLE), "intermediate": (STABLE, 0.5), "out_of_plane [0.5,1]": (0.5, 1.01)}
OWNER_FRAMES = (("3dpw:downtown_bar_00:actor0", 514), ("3dpw:downtown_bar_00:actor0", 775),
                ("3dpw:downtown_sitOnStairs_00:actor0", 325), ("3dpw:flat_packBags_00:actor0", 446),
                ("3dpw:downtown_runForBus_01:actor0", 185))
# Predeclared before any test result of G0/G1 was computed.
DECISION = {
    "static_elbow": "G1-B1 elbow mean < 0, sequence-cluster 95% interval excludes 0, improves in >= 2/3 of sequences",
    "temporal": "G1-B1 upper-arm motion residual < 0 AND G1-B1 proximal discontinuity rate < 0, both intervals exclude 0",
    "wrist_forearm_not_sacrificed": "G1-B1 wrist mean <= +0.5 cm AND G1-B1 forearm direction mean <= +1.0 deg",
    "near_folded": "G1 near-folded elbow mean < B1 near-folded elbow mean",
    "coverage": "no learned-arm unavailable rows beyond B1's, rig invariants <= 1e-9",
    "vision": "G1-G0 elbow (static) or upper-arm residual (temporal) < 0 with interval excluding 0 => RGB contributes",
    "CASE_A": "all of static_elbow, temporal, wrist_forearm_not_sacrificed, near_folded, coverage for G1, and either "
              "vision holds or G0 does not meet the same criteria",
    "CASE_B": "G0 meets static_elbow/temporal/wrist_forearm/coverage and vision does not hold (G1 ~ G0)",
    "CASE_D": "static_elbow holds but G1-B1 upper residual > 0 (interval excludes 0), or temporal holds but the wrist/forearm "
              "criterion fails",
    "CASE_C": "otherwise (no material static and temporal improvement)"}


def learned_arm(name, h0, d_rig, reach, side_cam, yaw, alignment, target, rest, reach_src, side_src):
    m = h0
    if reach_src != "h0":
        m = replace(m, reach_fraction=float(reach))
    if side_src != "h0":
        m = replace(m, side=heading_relative(tuple(float(v) for v in side_cam), yaw, alignment), degenerate=False)
    return full_arm(name, m, d_rig, target, rest)


def side_error(arm, gt):
    if arm.status != "known" or gt.side is None:
        return None
    ref = _perpendicular(gt.side, arm.direction)
    if ref is None:
        return None
    return math.degrees(math.acos(max(-1.0, min(1.0, sum(a * b for a, b in zip(arm.side, ref))))))


def collect(split, bank, h0_arr, w73, preds, target, rest, alignment, raw):
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
            gt_pos = [gt_frame.points[n].position for n in NAMES]
            ev = evaluate(h0_pos, gt_pos, orient.yaw_radians, alignment, target, rest, pred["chain_f_e1"], plane)
            if ev is None:
                acc["excluded:degenerate_chain_geometry"] += 1
                continue
            d_rig = ev["arms"]["c100"].direction
            arms = {b: ev["arms"][c] for b, c in BASE.items()}
            for name, (rs, ss) in LEARNED.items():
                rk = f"{OWNERS_G[rs]}_reach" if rs != "h0" else None
                sk = f"{OWNERS_G[ss]}_side" if ss != "h0" else None
                arms[name] = learned_arm(name, ev["h0"], d_rig, preds[rk][p] if rk else None, preds[sk][p] if sk else None,
                                         orient.yaw_radians, alignment, target, rest, rs, ss)
            for o in ORACLES:
                arms[o] = ev["arms"][o]
            ev["arms"] = arms
            conf = min(float(bank.arrays["input_2d"][p][j, 2]) for j in (11, 12, 13))
            r = row_record(sid, frame.frame_index, p, order, bank.samples[p].sample_id, conf, orient.yaw_radians,
                           pred["chain_f_e1"], ev)
            for n, a in arms.items():
                if r["candidates"][n]["status"] == "known":
                    r["candidates"][n]["side_error_degrees"] = side_error(a, ev["gt"])
                acc[f"{n}:{a.status}"] += 1
                acc[f"{n}:clamp:{a.clamp}"] += 1
            r["learned"] = {g: {"reach": float(preds[f"{OWNERS_G[g]}_reach"][p]), "side_cam": [float(v) for v in preds[f"{OWNERS_G[g]}_side"][p]]}
                            for g in OWNERS_G}
            r["geometry"] = {n: r["geometry"][n] for n in KEEP_GEOMETRY}
            acc["common_eligible"] += 1
            rows.append(r)
        accounting[sid] = dict(acc)
    return rows, accounting


def pair_records(rows, thresholds, names=KEEP_GEOMETRY):
    pairs, _ = _adjacent_pairs(rows)
    out = []
    for a, b in pairs:
        ra, rb = _Arm(a["reference"]), _Arm(b["reference"])
        ref_change = adjacent_change(ra, rb)
        rec = {"sequence_id": a["sequence_id"], "frame_index": a["frame_index"], "next_frame_index": b["frame_index"],
               "gt_bend_both": (a["gt_bend_degrees"], b["gt_bend_degrees"]), "abs_f_gt_both": (abs(a["f_gt"]), abs(b["f_gt"])),
               "reference": ref_change, "candidates": {}}
        for n in names:
            ca, cb = _Arm(a["geometry"][n]), _Arm(b["geometry"][n])
            m = {**adjacent_change(ca, cb), **motion_residual(ca, cb, ra, rb)}
            for t in TEMPORAL_METRICS:
                m[f"disc:{t}"] = float(classify_change(m[t], ref_change[t], thresholds[t]) == "candidate_discontinuity")
            m["reach_change"] = abs(b["geometry"][n]["reach_fraction"] - a["geometry"][n]["reach_fraction"])
            m["side_change_degrees"] = m["bend_side_change_degrees"]  # learned (or H0) side, frame to frame
            rec["candidates"][n] = m
        rec["reference"]["reach_change"] = abs(b["gt_reach"] - a["gt_reach"])
        out.append(rec)
    return out


def _cmp(items, a, b, metric, bootstrap=True):
    return _contrast(items, lambda it: (it["candidates"][a][metric] - it["candidates"][b][metric])
                     if it["candidates"][a].get(metric) is not None and it["candidates"][b].get(metric) is not None else None, bootstrap)


def _table(items, names, metrics):
    return {n: {m: _stats(it["candidates"][n].get(m) for it in items) for m in metrics} for n in names}


def _attribution(items, metric, g):
    """2x2 reach/side ownership effects under E1 direction for learned owner g (H0 = B1 corner)."""
    r, s, both = f"{g}r_h0s", f"h0r_{g}s", g
    y = lambda it, n: it["candidates"][n][metric]  # noqa: E731
    return {"reach": _contrast(items, lambda it: y(it, r) - y(it, "b1"), False)["mean"],
            "side": _contrast(items, lambda it: y(it, s) - y(it, "b1"), False)["mean"],
            "both": _contrast(items, lambda it: y(it, both) - y(it, "b1"), False)["mean"],
            "interaction": _contrast(items, lambda it: y(it, both) - y(it, r) - y(it, s) + y(it, "b1"), False)["mean"]}


def _headroom(items, metric):
    mean = {n: sum(it["candidates"][n][metric] for it in items) / len(items) for n in ("b0", "b1", "g0", "g1", "c111", "c_gtf_h0r_h0s")
            if all(n in it["candidates"] and it["candidates"][n].get(metric) is not None for it in items)}
    gap = mean.get("b1", 0) - mean.get("c111", 0)
    return {"means": mean, "reference": 0.0,
            "gap_b1_to_c111_closed": {g: (mean["b1"] - mean[g]) / gap if gap else None for g in ("g0", "g1") if g in mean}}


def _regime(rows, pairs, row_ok, pair_ok):
    g = [r for r in rows if row_ok(r)]
    p = [x for x in pairs if pair_ok(x)]
    out = {"rows": len(g), "pairs": len(p), "sequences": len({r["sequence_id"] for r in g})}
    if not g:
        return out
    out["static_means"] = {n: {m: sum(r["candidates"][n][m] for r in g) / len(g) for m in ("wrist_error", "elbow_error", "forearm_direction_error_degrees")}
                           for n in ("b0", "b1", "g0", "g1", "c111")}
    keep = ("mean", "sequences_improved", "sequences", "largest_single_sequence_contribution", "worst_regressions")
    out["g1_vs_b1_elbow"] = {k: v for k, v in _paired(g, "g1", "b1", "elbow_error").items() if k in keep}
    out["g1_vs_g0_elbow"] = {k: v for k, v in _paired(g, "g1", "g0", "elbow_error").items() if k in keep}
    out["g1_vs_b1_wrist"] = {k: v for k, v in _paired(g, "g1", "b1", "wrist_error").items() if k in keep}
    out["elbow_attribution_g1"] = _attribution(g, "elbow_error", "g1")
    if p:
        out["temporal_means"] = {n: {m: sum(x["candidates"][n][m] for x in p) / len(p) for m in (
            "upper_direction_residual_degrees", "disc:proximal_rotation_change_degrees")} for n in ("b0", "b1", "g0", "g1", "c111")}
        out["g1_vs_b1_upper_residual"] = {k: v for k, v in _contrast(p, lambda x: x["candidates"]["g1"]["upper_direction_residual_degrees"]
                                                                   - x["candidates"]["b1"]["upper_direction_residual_degrees"], False).items()
                                          if k in ("mean", "sequences_negative", "sequences")}
    return out


def _decision(static_cmp, temporal_cmp, regimes, coverage):
    def holds(c, sign=-1):
        return c["mean"] * sign > 0 and _excludes_zero(c)

    def criteria(g):
        el = static_cmp[f"{g}-b1"]["elbow_error"]
        out = {"static_elbow": holds(el) and el["sequences_negative"] >= 2 / 3 * el["sequences"],
               "temporal": holds(temporal_cmp[f"{g}-b1"]["upper_direction_residual_degrees"])
               and holds(temporal_cmp[f"{g}-b1"]["disc:proximal_rotation_change_degrees"]),
               "wrist_forearm_not_sacrificed": static_cmp[f"{g}-b1"]["wrist_error"]["mean"] <= 0.5
               and static_cmp[f"{g}-b1"]["forearm_direction_error_degrees"]["mean"] <= 1.0,
               "near_folded": regimes["near_folded [90,180]"]["static_means"][g]["elbow_error"]
               < regimes["near_folded [90,180]"]["static_means"]["b1"]["elbow_error"],
               "coverage": coverage[g],
               "temporal_worse": holds(temporal_cmp[f"{g}-b1"]["upper_direction_residual_degrees"], sign=1)}
        return out
    g1, g0 = criteria("g1"), criteria("g0")
    vision = holds(static_cmp["g1-g0"]["elbow_error"]) or holds(temporal_cmp["g1-g0"]["upper_direction_residual_degrees"])
    core = ("static_elbow", "temporal", "wrist_forearm_not_sacrificed", "coverage")
    if all(g1[k] for k in (*core, "near_folded")) and (vision or not all(g0[k] for k in core)):
        case = "A"
    elif all(g0[k] for k in core) and not vision:
        case = "B"
    elif (g1["static_elbow"] and g1["temporal_worse"]) or (g1["temporal"] and not g1["wrist_forearm_not_sacrificed"]):
        case = "D"
    else:
        case = "C"
    return {"g1": g1, "g0": g0, "vision_contributes": vision, "mechanical_case": case}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("rig", "rest", "calibration", "chains", "bank", "h0_dir", "threedpw_raw", "w73_dir", "w76_dir", "w77_dir",
                 "owner_dir", "out_dir"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    args = parser.parse_args()

    identities = {"w73_rows_heldout": _sha(args.w73_dir / "rows_heldout.json"), "w76_report": _sha(args.w76_dir / "report.json"),
                  "w77_report": _sha(args.w77_dir / "report.json"), "w77_rows_test": _sha(args.w77_dir / "rows_test.json"),
                  "owner_training_report": _sha(args.owner_dir / "training_report.json"),
                  "owner_predictions": _sha(args.owner_dir / "predictions_3dpw.npz"),
                  **{f"owner_checkpoint_{g}": _sha(args.owner_dir / f"{c}.pt") for g, c in OWNERS_G.items()}}
    for k, exp in PINNED.items():
        if identities[k] != exp:
            raise SystemExit(f"{k} identity mismatch: {identities[k]}")
    training = json.loads((args.owner_dir / "training_report.json").read_text())
    if training["predictions_npz_sha256"] != identities["owner_predictions"]:
        raise SystemExit("owner predictions differ from the training report")
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
        raise SystemExit("owner was trained on another FrameBank")
    h0_arr, _ = assemble_h0(bank, {"validation": args.h0_dir / "prediction_validation.npy", "test": args.h0_dir / "prediction_test.npy"})
    w73 = {(r["sequence_id"], r["frame_index"]): r for r in json.loads((args.w73_dir / "rows_heldout.json").read_text())}
    preds = dict(np.load(args.owner_dir / "predictions_3dpw.npz"))
    if len(preds["f_e1"]) != len(bank):
        raise SystemExit("owner predictions do not cover the FrameBank")

    rows, accounting = collect("test", bank, h0_arr, w73, preds, target, rest, alignment, args.threedpw_raw)

    # --- Gate 1: historical baselines and frozen E1 inputs ---
    w77_rows = json.loads((args.w77_dir / "rows_test.json").read_text())
    w77 = {(r["sequence_id"], r["frame_index"]): r for r in w77_rows}
    w77_report = json.loads((args.w77_dir / "report.json").read_text())
    mine = {(r["sequence_id"], r["frame_index"]): r for r in rows}
    g1 = {"rows": len(rows), "identity_set_equal": set(mine) == set(w77), "sequences": len({k[0] for k in mine}),
          "max_abs_diff": defaultdict(float), "mismatch": Counter()}
    m = g1["max_abs_diff"]
    for key, r in mine.items():
        old = w77.get(key)
        if old is None:
            continue
        for b, c in BASE.items():
            a, o = r["geometry"][b], old["geometry"][c]
            for k in ("status", "side_status", "clamp", "reach_fraction"):
                g1["mismatch"][f"{b}:{k}"] += a[k] != o[k]
            m[f"{b}:joints_cm"] = max(m[f"{b}:joints_cm"], math.dist(a["end"], o["end"]), math.dist(a["mid"], o["mid"]))
            m[f"{b}:rotations"] = max(m[f"{b}:rotations"], _qdiff(a["proximal_local"], o["proximal_local"]),
                                      _qdiff(a["distal_local"], o["distal_local"]))
        for o_name in ORACLES:
            m[f"{o_name}:elbow_error"] = max(m[f"{o_name}:elbow_error"], abs(r["candidates"][o_name]["elbow_error"] - old["candidates"][o_name]["elbow_error"]))
        m["reference_cm"] = max(m["reference_cm"], math.dist(r["reference"]["mid"], old["reference"]["mid"]), math.dist(r["e_gt"], old["e_gt"]))
        m["yaw"] = max(m["yaw"], abs(r["yaw_h0_radians"] - old["yaw_h0_radians"]))
        m["e1_input_vs_w73_chain_f"] = max(m["e1_input_vs_w73_chain_f"], abs(float(preds["f_e1"][r["position"]]) - r["f_e1"]))
    thresholds = w76_report["test"]["pooled"]["temporal"]["thresholds_gt_p99"]
    pairs = pair_records(rows, thresholds)
    old_pairs, _ = _adjacent_pairs(w77_rows)
    g1["pairs"] = len(pairs)
    g1["pairs_identical"] = [(x["sequence_id"], x["frame_index"], x["next_frame_index"]) for x in pairs] == \
        [(a["sequence_id"], a["frame_index"], b["frame_index"]) for a, b in old_pairs]
    for b, c in BASE.items():
        for t in TEMPORAL_METRICS:
            m[f"{b}:disc_rate:{t}"] = abs(sum(x["candidates"][b][f"disc:{t}"] for x in pairs) / len(pairs)
                                          - w77_report["temporal"]["candidates"][c][f"disc:{t}"])
    limits = {k: (0.0 if "disc_rate" in k or k == "yaw" else E1_INPUT_TOLERANCE if k.startswith("e1_input")
                  else TOL_QUAT if "rotation" in k else TOL_CM) for k in m}
    failed = {k: v for k, v in m.items() if v > limits[k]}
    g1.update({"limits": limits, "max_abs_diff": dict(m), "mismatch": {k: v for k, v in g1["mismatch"].items() if v}})
    g1["passed"] = (g1["identity_set_equal"] and g1["rows"] == 6003 and g1["sequences"] == 37 and g1["pairs"] == 5504
                    and g1["pairs_identical"] and not failed and not g1["mismatch"])
    if not g1["passed"]:
        raise SystemExit(f"STOP: historical baseline parity failed: {json.dumps(g1, default=str)}")

    # --- Gate 2: learned output geometry ---
    g2 = {"unavailable": {n: sum(r["candidates"][n]["status"] != "known" for r in rows) for n in CANDIDATES},
          "clamps": {n: dict(Counter(str(r["candidates"][n].get("clamp")) for r in rows)) for n in CANDIDATES},
          "reach_range": {g: [float(np.min([r["learned"][g]["reach"] for r in rows])), float(np.max([r["learned"][g]["reach"] for r in rows]))] for g in OWNERS_G},
          "integrity_max": {n: max(max(r["candidates"][n][k] for k in ("integrity_upper_length_residual", "integrity_forearm_length_residual",
                                                                       "integrity_endpoint_reconstruction_residual", "integrity_ik_to_fk_residual"))
                                   for r in rows if r["candidates"][n]["status"] == "known") for n in LEARNED},
          "nonfinite": {n: sum(not r["candidates"][n].get("integrity_finite", True) for r in rows) for n in LEARNED}}
    g2["passed"] = (all(v <= TOL_CM for v in g2["integrity_max"].values()) and not any(g2["nonfinite"].values())
                    and all(0.0 < lo and hi < 1.0 for lo, hi in g2["reach_range"].values()))
    if not g2["passed"]:
        raise SystemExit(f"STOP: learned output geometry invalid: {json.dumps(g2, default=str)}")
    common = [r for r in rows if all(r["candidates"][n]["status"] == "known" for n in CANDIDATES)]

    # --- Evaluation ---
    static_cmp = {f"{a}-{b}": {mm: _cmp(common, a, b, mm) for mm in PAIRED_STATIC} for a, b in COMPARISONS}
    temporal_cmp = {f"{a}-{b}": {mm: _cmp(pairs, a, b, mm) for mm in TEMPORAL} for a, b in COMPARISONS}
    for k in ("c111", "c_gtf_h0r_h0s"):
        temporal_cmp[f"g1-{k}"] = {mm: _cmp(pairs, "g1", k, mm, False) for mm in ("upper_direction_residual_degrees", "disc:proximal_rotation_change_degrees")}
    regimes = {"gt_bend": {lab: _regime(common, pairs, lambda r, lo=lo, hi=hi: lo <= r["gt_bend_degrees"] < hi,
                                        lambda x, lo=lo, hi=hi: all(lo <= v < hi for v in x["gt_bend_both"])) for lab, (lo, hi) in BEND_REGIMES.items()},
               "abs_gt_f": {lab: _regime(common, pairs, lambda r, lo=lo, hi=hi: lo <= abs(r["f_gt"]) < hi,
                                         lambda x, lo=lo, hi=hi: all(lo <= v < hi for v in x["abs_f_gt_both"])) for lab, (lo, hi) in F_BINS.items()}}
    # Worklog 76 near-folded conflict rows (C100 wrist better, elbow worse than C000, GT bend >= 90).
    conflict = [r for r in common if r["gt_bend_degrees"] >= 90 and r["candidates"]["b1"]["wrist_error"] < r["candidates"]["b0"]["wrist_error"]
                and r["candidates"]["b1"]["elbow_error"] > r["candidates"]["b0"]["elbow_error"]]
    conflict_block = {"rows": len(conflict), **({f"{n}_elbow_mean": sum(r["candidates"][n]["elbow_error"] for r in conflict) / len(conflict)
                                                 for n in ("b0", "b1", "g0", "g1", "c111")} if conflict else {}),
                      **({f"{n}_wrist_mean": sum(r["candidates"][n]["wrist_error"] for r in conflict) / len(conflict)
                          for n in ("b0", "b1", "g0", "g1", "c111")} if conflict else {}),
                      "g1_elbow_better_than_b1_share": (sum(r["candidates"]["g1"]["elbow_error"] < r["candidates"]["b1"]["elbow_error"] for r in conflict) / len(conflict)) if conflict else None,
                      "g1_elbow_not_worse_than_b0_share": (sum(r["candidates"]["g1"]["elbow_error"] <= r["candidates"]["b0"]["elbow_error"] for r in conflict) / len(conflict)) if conflict else None}
    coverage = {g: g2["unavailable"][g] <= g2["unavailable"]["b1"] and g2["integrity_max"][g] <= TOL_CM for g in ("g0", "g1")}
    per_seq = {}
    for sid in sorted({r["sequence_id"] for r in common}):
        g = [r for r in common if r["sequence_id"] == sid]
        p = [x for x in pairs if x["sequence_id"] == sid]
        per_seq[sid] = {"rows": len(g), "pairs": len(p),
                        **{f"{a}-{b}_elbow": sum(r["candidates"][a]["elbow_error"] - r["candidates"][b]["elbow_error"] for r in g) / len(g)
                           for a, b in (("g1", "b1"), ("g1", "g0"), ("g1", "b0"))},
                        **{f"{a}-{b}_upper_residual": (sum(x["candidates"][a]["upper_direction_residual_degrees"] - x["candidates"][b]["upper_direction_residual_degrees"] for x in p) / len(p)) if p else None
                           for a, b in (("g1", "b1"), ("g1", "g0"))},
                        "g1-b1_wrist": sum(r["candidates"]["g1"]["wrist_error"] - r["candidates"]["b1"]["wrist_error"] for r in g) / len(g)}
    report = {
        "schema": "animcv_framepose_arm_geometry_owner_eval_v1", "identities": identities,
        "labels": {"runtime": "B0, B1, G0, G1 and the learned attribution arms (runtime observable inputs only)",
                   "evaluation_only": "C111, C110, C101 (GT reach/side), C_GT_F (GT f), GT reference"},
        "architecture": training["architecture"], "training_contract": training["training"],
        "training_summary": {c: {k: training["candidates"][c][k] for k in ("selection", "final", "steps", "draws", "source_draws", "parameters", "init_digest",
                                                                         "learnability_3dpw")} for c in training["candidates"]},
        "training_curves": {c: training["candidates"][c]["curve"] for c in training["candidates"]},
        "e1": training["e1"], "control_identity": training["control_identity"], "eligibility": training["eligibility"],
        "gate1_historical_parity": g1, "gate2_learned_geometry": g2, "accounting": accounting,
        "bootstrap_contract": BOOTSTRAP, "decision_contract": DECISION, "jump_thresholds_worklog76": thresholds,
        "static": {"candidates": _table(common, CANDIDATES, STATIC), "comparisons": static_cmp,
                   "attribution": {g: {mm: _attribution(common, mm, g) for mm in ("elbow_error", "wrist_error", "upper_direction_error_degrees",
                                                                                "forearm_direction_error_degrees")} for g in ("g0", "g1")}},
        "temporal": {"pairs": len(pairs), "reference": {t: _stats(x["reference"][t] for x in pairs) for t in (*TEMPORAL_METRICS, *ATTRIBUTION_TEMPORAL_METRICS, "reach_change")},
                     "candidates": _table(pairs, KEEP_GEOMETRY, (*RESIDUAL_METRICS, *TEMPORAL_METRICS, "reach_change", "side_change_degrees"))
                     | {"disc_rates": {n: {t: sum(x["candidates"][n][f"disc:{t}"] for x in pairs) / len(pairs) for t in TEMPORAL_METRICS} for n in KEEP_GEOMETRY}},
                     "comparisons": temporal_cmp},
        "oracle_headroom": {"elbow_error": _headroom(common, "elbow_error"), "wrist_error": _headroom(common, "wrist_error"),
                            "upper_direction_residual_degrees": _headroom(pairs, "upper_direction_residual_degrees"),
                            "disc:proximal_rotation_change_degrees": _headroom(pairs, "disc:proximal_rotation_change_degrees")},
        "regimes": regimes, "near_folded_conflict_rows_worklog76": conflict_block, "per_sequence": per_seq,
        "owner_frames": {f"{s}#{f}": ({n: {k: mine[(s, f)]["candidates"][n].get(k) for k in ("wrist_error", "elbow_error", "upper_direction_error_degrees",
                                                                                           "forearm_direction_error_degrees", "reach_error", "side_error_degrees")}
                                       for n in CANDIDATES} | {"gt_reach": mine[(s, f)]["gt_reach"], "h0_reach": mine[(s, f)]["h0_reach"],
                                                               "learned": mine[(s, f)]["learned"], "gt_bend": mine[(s, f)]["gt_bend_degrees"]})
                         if (s, f) in mine else None for s, f in OWNER_FRAMES},
    }
    report["decision_inputs"] = _decision(static_cmp, temporal_cmp, regimes["gt_bend"], coverage)

    # --- Validation (owner-selection split; reported separately, never merged) ---
    vrows, vacc = collect("validation", bank, h0_arr, w73, preds, target, rest, alignment, args.threedpw_raw)
    vcommon = [r for r in vrows if all(r["candidates"][n]["status"] == "known" for n in CANDIDATES)]
    report["validation"] = {"note": "the split that selected E1 epochs and the owner epochs (selection-contaminated); temporal pairs are sparse",
                            "accounting_rows": len(vrows), "common": len(vcommon),
                            "static": _table(vcommon, ("b0", "b1", "g0", "g1", "c111"), PAIRED_STATIC),
                            "comparisons": {f"{a}-{b}": {mm: _cmp(vcommon, a, b, mm, False) for mm in ("wrist_error", "elbow_error")} for a, b in COMPARISONS}}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "rows_test.json").write_text(json.dumps(rows))
    (args.out_dir / "pairs_test.json").write_text(json.dumps(pairs))
    path = args.out_dir / "report.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True, default=str))
    print(json.dumps({"sha256": _sha(path), "gate1": {k: g1[k] for k in ("rows", "pairs", "passed", "max_abs_diff")},
                      "gate2": g2, "decision": report["decision_inputs"]}, indent=1, default=str))


if __name__ == "__main__":
    main()
