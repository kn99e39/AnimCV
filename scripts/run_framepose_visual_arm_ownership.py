#!/usr/bin/env python3
"""EVALUATION ONLY (Worklog 77): direction noise vs mixed-owner arm geometry attribution.

Full 2x2x2 factorial over the three Worklog 61 IK inputs on the exact Worklog 76
test population: direction D in {H0, E1}, reach R in {H0, GT}, bend side S in
{H0, GT} (GT = evaluation-only oracle), plus the depth-oracle diagnostic
C_GT_F (GT forward fraction on the SAME detector plane as E1, H0 reach/side).
Every candidate is the unchanged Worklog 76 ``full_arm`` (Worklog 61 solve,
Worklog 62 side rule, zero-twist rotations) scored against the same Worklog 62
GT-anchored reference arm, with the Worklog 76 adjacency contract and jump
thresholds.

Order of work (each gate stops the run on failure):
  1. frozen identities (Worklog 73/75/76 artifacts, H0, rig);
  2. Gate 1 — Worklog 76 reproduction: population, adjacency pairs, C000 = F0 and
     C100 = F2 joints/rotations/statuses, reference arm, jump thresholds and
     discontinuity counts;
  3. Gate 2 — oracle-geometry contract: GT direction + GT reach + GT side reproduces
     the reference arm; GT direction + H0 reach/side and C_GT_F reproduce the
     Worklog 75 T4/T3 endpoint errors; ownership assignment is exact;
  4. only then the counterfactual attribution.  Nothing is trained, gated or temporally filtered.
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
from run_framepose_ik_gt_attribution import _stats  # noqa: E402
from run_framepose_visual_full_arm_ik import (  # noqa: E402
    BANK_STRIDE, BEND_REGIMES, BOOTSTRAP, NAMES, STABLE, _adjacent_pairs, _Arm, _bootstrap, _paired, _qdiff, _sha,
    _unit, plane_for, pose_record,
)
from run_root_motion_contact_h0_replay import _matched_oracle_sequence  # noqa: E402

from framepose.bank import load_bank  # noqa: E402
from motion.animation_semantics_v2_bridge import current_root_orientation  # noqa: E402
from pose.framepose_bridge import assemble_h0, build_h0_lifted_sequence, sequence_frame_positions, sequence_ids_in_split  # noqa: E402
from retarget.framepose_fk import derive_rig_alignment, load_rig_fk_calibration  # noqa: E402
from retarget.framepose_ik_gt_attribution import _runtime_side, measure_chain  # noqa: E402
from retarget.framepose_target_rest import load_target_rest_pose  # noqa: E402
from retarget.framepose_two_bone_ik import load_ik_chain_set, target_chain_geometry  # noqa: E402
from retarget.framepose_visual_arm_ownership import (  # noqa: E402
    CONTRACT, DIAGNOSTICS, FACTORIAL, RESIDUAL_METRICS, RUNTIME, direction_penalty, motion_residual, owned_measure,
    ownership_effects, penalty_changes,
)
from retarget.framepose_visual_full_arm_ik import (  # noqa: E402
    ATTRIBUTION_TEMPORAL_METRICS, TEMPORAL_METRICS, adjacent_change, classify_change, compare, full_arm, gt_reference,
)
from retarget.framepose_visual_ik_handoff import camera_chain_direction, oracle_reference, rig_direction  # noqa: E402

PINNED = {"w73_report": "1c05d50a5e6577329f0573786c6927b07268612ec150fbd079a1aa2bc4b03934",
          "w73_rows_heldout": "4e17bd77399e4c016098f70592d0cfd68f508245ae60914869f756605d9897fc",
          "w75_report": "89218babea89033537895a851a013e2b9615ed97314de8945a84b1168424dea7",
          "w75_rows_test": "dc39b22b9489689dc523d5348adda003d3a86b233bd33e061adeecab721de78e",
          "w76_report": "dad46288436ac0fe28fea9fab7ad6882c6a39fd476e717af4eaee497ab096a2b",
          "w76_rows_test": "d6b339dcb5733358026a1ca4683b367340401b0b07288c91d3a25cbebe1d824a"}
H0_VALIDATION_PREFIX = "88a007fc"
TOL_CM = 1e-9      # Worklog 61/74/76 solved-geometry contract
TOL_QUAT = 1e-9    # Worklog 76 quaternion component contract
CANDIDATES = (*FACTORIAL, *DIAGNOSTICS)
ALL_ARMS = (*CANDIDATES, *CONTRACT)
W76_MAP = {"c000": "f0_h0_full_ik", "c100": "f2_e1_full_ik"}
STATIC_METRICS = ("wrist_error", "elbow_error", "upper_direction_error_degrees", "forearm_direction_error_degrees",
                  "chain_direction_error_degrees", "abs_bend_angle_error_degrees", "bend_plane_error_degrees",
                  "reach_error", "proximal_rotation_vs_gt_degrees", "distal_rotation_vs_gt_degrees")
PAIR_METRICS = (*TEMPORAL_METRICS, *ATTRIBUTION_TEMPORAL_METRICS, *RESIDUAL_METRICS)
DISC = tuple(f"disc:{m}" for m in TEMPORAL_METRICS)
F_BINS = {"near_plane [0,sin10)": (0, STABLE), "intermediate": (STABLE, 0.5), "out_of_plane [0.5,1]": (0.5, 1.01)}
OWNERS = (("3dpw:downtown_bar_00:actor0", 514), ("3dpw:downtown_bar_00:actor0", 775),
          ("3dpw:downtown_sitOnStairs_00:actor0", 325), ("3dpw:downtown_runForBus_01:actor0", 185),
          ("3dpw:flat_packBags_00:actor0", 446))
# Predeclared before any counterfactual statistic was computed.
DECISION = {
    "primary_temporal_metric": "upper_direction_residual_degrees",
    "secondary": ["proximal_rotation_residual_degrees", "disc:proximal_rotation_change_degrees",
                  "elbow_motion_residual", "elbow_error (static)"],
    "base_penalty": "P = mean(C100 - C000) on the primary metric; must be > 0 with its sequence-cluster interval excluding 0",
    "material_fraction": 0.25, "dominant_fraction": 0.5,
    "reduction": "fraction of P removed (reach / side / both oracle swaps, or C_GT_F instead of E1); counts only if the "
                 "sequence-cluster 95% interval of the change excludes 0",
    "intrinsic": "E1 penalty remaining under GT reach + GT side: P(GT,GT) / P, interval must exclude 0",
    "broad": "the reduction appears in >= 2/3 of sequences",
    "CASE_A": "intrinsic >= dominant AND C_GT_F reduction >= dominant AND best owner reduction < dominant",
    "CASE_B": "best owner reduction >= dominant AND static elbow penalty also reduced (interval) AND broad AND intrinsic < material",
    "CASE_C": "intrinsic >= material AND best owner reduction >= material (otherwise-unmatched)",
    "CASE_D": "anything else (ambiguous / inconclusive intervals)"}


# ----------------------------------------------------------------------------- rows
def evaluate(h0_pos, gt_pos, yaw, alignment, target, rest, f_e1, plane):
    h0, gt = measure_chain(*h0_pos, yaw, alignment), measure_chain(*gt_pos, yaw, alignment)
    if h0 is None or gt is None:
        return None
    f_gt = _unit(tuple(b - a for a, b in zip(gt_pos[0], gt_pos[2])))[1]
    f_h0 = _unit(tuple(b - a for a, b in zip(h0_pos[0], h0_pos[2])))[1]
    directions = {"h0": h0.direction, "gt_dir": gt.direction,
                  "e1": rig_direction(camera_chain_direction(f_e1, plane), yaw, alignment),
                  "gt_f": rig_direction(camera_chain_direction(f_gt, plane), yaw, alignment)}
    arms = {name: full_arm(name, owned_measure(h0, gt, r, s), directions[d], target, rest)
            for name, (d, r, s) in {**FACTORIAL, **DIAGNOSTICS, **CONTRACT}.items()}
    e_gt = oracle_reference(gt, target, rest)[0]
    return {"h0": h0, "gt": gt, "f_gt": f_gt, "f_h0": f_h0, "e_gt": e_gt, "ref": gt_reference(gt, target, rest),
            "arms": arms}


def row_record(sid, frame_index, p, order, sample_id, conf, yaw, f_e1, ev):
    ref = ev["ref"]
    cands = {}
    for n, a in ev["arms"].items():
        c = compare(a, ref, ev["e_gt"])
        if c["status"] == "known":
            c["abs_bend_angle_error_degrees"] = abs(c["bend_angle_error_degrees"])
            c["reach_error"] = abs(a.reach_fraction - ev["gt"].reach_fraction)
        cands[n] = c
    return {"sequence_id": sid, "frame_index": frame_index, "position": p, "bank_order": order, "sample_id": sample_id,
            "min_arm_confidence": conf, "yaw_h0_radians": yaw, "f_gt": ev["f_gt"], "f_h0": ev["f_h0"], "f_e1": f_e1,
            "h0_reach": ev["h0"].reach_fraction, "gt_reach": ev["gt"].reach_fraction,
            "h0_bend_degrees": ev["h0"].bend_angle_degrees, "gt_bend_degrees": ev["gt"].bend_angle_degrees,
            "h0_side_degenerate": ev["h0"].degenerate, "gt_side_degenerate": ev["gt"].degenerate, "e_gt": list(ev["e_gt"]),
            "reference": pose_record(ref), "geometry": {n: pose_record(a) for n, a in ev["arms"].items()},
            "candidates": cands}


def collect(bank, h0, w73, target, rest, alignment, raw):
    """Worklog 75/76 validity and identity join on the test split."""
    rows, accounting, assignment = [], {}, Counter()
    for sid in sequence_ids_in_split(bank, "test"):
        lifted = build_h0_lifted_sequence(bank, h0, sid)
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
            h0_pos = [frame.points[n].position for n in NAMES]
            gt_pos = [gt_frame.points[n].position for n in NAMES]
            plane = plane_for(bank, p)
            if plane is None:
                acc["excluded:degenerate_observed_plane"] += 1
                continue
            ev = evaluate(h0_pos, gt_pos, orient.yaw_radians, alignment, target, rest, pred["chain_f_e1"], plane)
            if ev is None:
                acc["excluded:degenerate_chain_geometry"] += 1
                continue
            if abs(pred["chain_f_h0"] - ev["f_h0"]) > 1e-6 or abs(pred["chain_f_gt"] - ev["f_gt"]) > 1e-5:
                raise SystemExit(f"{key}: Worklog 73 chain f identity differs")
            # Exact ownership assignment (reach and side source per arm).
            for n, (d, r, s) in {**FACTORIAL, **DIAGNOSTICS, **CONTRACT}.items():
                a, src_r, src_s = ev["arms"][n], ev[r], ev[s]
                if a.reach_fraction != src_r.reach_fraction:
                    assignment[f"{n}:reach_source_mismatch"] += 1
                expected, _ = _runtime_side(src_s, target, rest, a.direction)
                if (expected is None) != (a.side is None) or (expected is not None and expected != a.side):
                    assignment[f"{n}:side_source_mismatch"] += 1
            if owned_measure(ev["h0"], ev["gt"], "h0", "h0") is not ev["h0"]:
                assignment["runtime_measure_not_h0"] += 1
            conf = min(float(bank.arrays["input_2d"][p][j, 2]) for j in (11, 12, 13))
            acc["common_eligible"] += 1
            for n in ALL_ARMS:
                acc[f"{n}:{ev['arms'][n].status}"] += 1
            rows.append(row_record(sid, frame.frame_index, p, order, bank.samples[p].sample_id, conf,
                                   orient.yaw_radians, pred["chain_f_e1"], ev))
        accounting[sid] = dict(acc)
    return rows, accounting, dict(assignment)


# ----------------------------------------------------------------------------- pairs
def pair_records(rows, thresholds):
    pairs, _ = _adjacent_pairs(rows)
    out = []
    for a, b in pairs:
        ra, rb = _Arm(a["reference"]), _Arm(b["reference"])
        ref_change = adjacent_change(ra, rb)
        rec = {"sequence_id": a["sequence_id"], "frame_index": a["frame_index"], "next_frame_index": b["frame_index"],
               "gt_bend_both": (a["gt_bend_degrees"], b["gt_bend_degrees"]), "abs_f_gt_both": (abs(a["f_gt"]), abs(b["f_gt"])),
               "abs_f_change": {k: abs(b[k] - a[k]) for k in ("f_gt", "f_h0", "f_e1")},
               "reference": ref_change, "candidates": {}}
        for n in CANDIDATES:
            ca, cb = _Arm(a["geometry"][n]), _Arm(b["geometry"][n])
            m = {**adjacent_change(ca, cb), **motion_residual(ca, cb, ra, rb)}
            for t in TEMPORAL_METRICS:
                m[f"disc:{t}"] = float(classify_change(m[t], ref_change[t], thresholds[t]) == "candidate_discontinuity")
            m["excess_proximal_change"] = m["proximal_rotation_change_degrees"] - ref_change["proximal_rotation_change_degrees"]
            rec["candidates"][n] = m
        out.append(rec)
    return out


# ----------------------------------------------------------------------------- statistics
def _contrast(items, fn, bootstrap=True):
    """Mean of a per-row (or per-pair) contrast with sequence counts and Worklog 75/76 sequence-aware intervals."""
    by = defaultdict(list)
    for it in items:
        v = fn(it)
        if v is not None:
            by[it["sequence_id"]].append(v)
    vals = [v for s in by.values() for v in s]
    if not vals:
        return {"n": 0}
    seq_means = {s: sum(v) / len(v) for s, v in by.items()}
    sums = sorted(((s, sum(v)) for s, v in by.items()), key=lambda kv: kv[1])
    net = sum(v for _, v in sums)
    out = {**_stats(vals), "mean": sum(vals) / len(vals),
           "sequence_weighted_mean": sum(seq_means.values()) / len(seq_means),
           "sequences": len(seq_means), "sequences_negative": sum(v < 0 for v in seq_means.values()),
           "sequences_positive": sum(v > 0 for v in seq_means.values()),
           "most_negative_sequence": {"sequence_id": sums[0][0], "sum": sums[0][1],
                                      "share_of_net": sums[0][1] / net if net < 0 else None},
           "most_positive_sequence": {"sequence_id": sums[-1][0], "sum": sums[-1][1],
                                      "share_of_net": sums[-1][1] / net if net > 0 else None}}
    if bootstrap and len(by) > 1:
        out["bootstrap"] = _bootstrap(by)
    return out


def _ci(c, key="sequence_cluster"):
    return c["bootstrap"][key]["frame_weighted_ci95"] if "bootstrap" in c else None


def _excludes_zero(c):
    ci = _ci(c)
    return ci is not None and (ci[0] > 0 or ci[1] < 0)


def _y(item, metric):
    return {n: item["candidates"][n][metric] for n in FACTORIAL}


def _factorial(items, metric, bootstrap=True):
    out = {"direction_penalty": {k: _contrast(items, lambda it, k=k: direction_penalty(_y(it, metric))[k], bootstrap)
                                 for k in ("h0r_h0s", "gtr_h0s", "h0r_gts", "gtr_gts")},
           "penalty_change": {k: _contrast(items, lambda it, k=k: penalty_changes(_y(it, metric))[k], bootstrap)
                              for k in ("reach", "side", "both", "three_way_interaction")},
           "ownership_effects": {d: {k: _contrast(items, lambda it, d=d, k=k: ownership_effects(_y(it, metric))[d][k], False)["mean"]
                                     for k in ("reach", "side", "both", "reach_x_side")} for d in ("h0", "e1")}}
    out["gt_f_penalty"] = _contrast(items, lambda it: it["candidates"]["c_gtf_h0r_h0s"][metric] - it["candidates"]["c000"][metric], bootstrap)
    out["gt_f_reduction_vs_e1"] = _contrast(items, lambda it: it["candidates"]["c_gtf_h0r_h0s"][metric] - it["candidates"]["c100"][metric], bootstrap)
    return out


def _candidate_table(items, metrics):
    return {n: {m: _stats(it["candidates"][n][m] for it in items) for m in metrics} for n in CANDIDATES}


def _paired_vs(rows, base, metrics):
    keep = ("mean", "p50", "p95", "improved_share", "regressed_share", "sequences_improved", "sequences_regressed",
            "largest_single_sequence_contribution", "worst_regressions", "worst_row_regressions")
    return {n: {m: {k: v for k, v in _paired(rows, n, base, m).items() if k in keep} for m in metrics}
            for n in CANDIDATES if n != base}


def _decision(fac_primary, fac_elbow, fac_prox, fac_disc):
    P = fac_primary["direction_penalty"]["h0r_h0s"]
    base = P["mean"]
    out = {"base_penalty": base, "base_penalty_ci": _ci(P), "base_penalty_significant": base > 0 and _excludes_zero(P)}

    def frac(change):
        return -change["mean"] / base if base else None
    red = {k: {"fraction_removed": frac(fac_primary["penalty_change"][k]), "ci_excludes_zero": _excludes_zero(fac_primary["penalty_change"][k]),
               "sequences_reduced": fac_primary["penalty_change"][k]["sequences_negative"],
               "sequences": fac_primary["penalty_change"][k]["sequences"]} for k in ("reach", "side", "both")}
    gtf = fac_primary["gt_f_reduction_vs_e1"]
    red["gt_f_instead_of_e1"] = {"fraction_removed": frac(gtf), "ci_excludes_zero": _excludes_zero(gtf),
                                 "sequences_reduced": gtf["sequences_negative"], "sequences": gtf["sequences"]}
    out["reductions"] = red
    rem = fac_primary["direction_penalty"]["gtr_gts"]
    out["intrinsic_remaining_fraction"] = rem["mean"] / base if base else None
    out["intrinsic_ci"] = _ci(rem)
    out["intrinsic_significant"] = rem["mean"] > 0 and _excludes_zero(rem)
    valid = lambda r: r["ci_excludes_zero"] and r["fraction_removed"] is not None  # noqa: E731
    owner = max((r["fraction_removed"] for k, r in red.items() if k != "gt_f_instead_of_e1" and valid(r)), default=0.0)
    owner_key = max((k for k in ("reach", "side", "both") if valid(red[k])), key=lambda k: red[k]["fraction_removed"], default=None)
    out["best_owner_reduction"] = {"swap": owner_key, "fraction_removed": owner}
    intrinsic = out["intrinsic_remaining_fraction"] if out["intrinsic_significant"] else 0.0
    gt_f = red["gt_f_instead_of_e1"]["fraction_removed"] if valid(red["gt_f_instead_of_e1"]) else 0.0
    elbow_static_reduced = (owner_key is not None and _excludes_zero(fac_elbow["penalty_change"][owner_key])
                            and fac_elbow["penalty_change"][owner_key]["mean"] < 0)
    broad = owner_key is not None and red[owner_key]["sequences_reduced"] >= 2 / 3 * red[owner_key]["sequences"]
    out["checks"] = {"intrinsic_fraction_used": intrinsic, "gt_f_fraction_used": gt_f, "owner_fraction_used": owner,
                     "elbow_static_penalty_reduced_by_best_owner_swap": elbow_static_reduced, "owner_reduction_broad": broad}
    d, m = DECISION["dominant_fraction"], DECISION["material_fraction"]
    if not out["base_penalty_significant"]:
        case = "D"
    elif intrinsic >= d and gt_f >= d and owner < d:
        case = "A"
    elif owner >= d and elbow_static_reduced and broad and intrinsic < m:
        case = "B"
    elif intrinsic >= m and owner >= m:
        case = "C"
    else:
        case = "D"
    out["mechanical_case"] = case
    out["secondary_consistency"] = {
        name: {"base": f["direction_penalty"]["h0r_h0s"]["mean"],
               **{k: f["penalty_change"][k]["mean"] for k in ("reach", "side", "both")},
               "remaining_gtr_gts": f["direction_penalty"]["gtr_gts"]["mean"],
               "gt_f_minus_e1": f["gt_f_reduction_vs_e1"]["mean"]}
        for name, f in (("proximal_rotation_residual", fac_prox), ("disc_proximal", fac_disc), ("elbow_error_static", fac_elbow))}
    return out


def _regime(rows, pairs, row_ok, pair_ok):
    g = [r for r in rows if row_ok(r)]
    p = [x for x in pairs if pair_ok(x)]
    out = {"rows": len(g), "pairs": len(p), "sequences": len({r["sequence_id"] for r in g})}
    if not g:
        return out
    out["elbow_mean"] = {n: sum(r["candidates"][n]["elbow_error"] for r in g) / len(g) for n in CANDIDATES}
    out["wrist_mean"] = {n: sum(r["candidates"][n]["wrist_error"] for r in g) / len(g) for n in CANDIDATES}
    out["elbow_factorial"] = _factorial(g, "elbow_error", bootstrap=False)
    out["c100_vs_c000_elbow"] = {k: v for k, v in _paired(g, "c100", "c000", "elbow_error").items()
                                 if k in ("mean", "sequences_improved", "sequences", "largest_single_sequence_contribution", "worst_regressions")}
    out["c111_vs_c011_elbow"] = {k: v for k, v in _paired(g, "c111", "c011", "elbow_error").items()
                                 if k in ("mean", "sequences_improved", "sequences", "largest_single_sequence_contribution", "worst_regressions")}
    if p:
        out["upper_residual_mean"] = {n: sum(x["candidates"][n]["upper_direction_residual_degrees"] for x in p) / len(p) for n in CANDIDATES}
        out["disc_proximal_rate"] = {n: sum(x["candidates"][n]["disc:proximal_rotation_change_degrees"] for x in p) / len(p) for n in CANDIDATES}
        out["upper_residual_factorial"] = _factorial(p, "upper_direction_residual_degrees", bootstrap=False)
    return out


def _per_sequence(rows, pairs):
    out = {}
    for sid in sorted({r["sequence_id"] for r in rows}):
        g = [r for r in rows if r["sequence_id"] == sid]
        p = [x for x in pairs if x["sequence_id"] == sid]
        mean = lambda items, fn: sum(fn(i) for i in items) / len(items) if items else None  # noqa: E731
        out[sid] = {"rows": len(g), "pairs": len(p),
                    "elbow_penalty": {k: mean(g, lambda r, k=k: direction_penalty(_y(r, "elbow_error"))[k]) for k in ("h0r_h0s", "gtr_h0s", "h0r_gts", "gtr_gts")},
                    "upper_residual_penalty": {k: mean(p, lambda x, k=k: direction_penalty(_y(x, "upper_direction_residual_degrees"))[k])
                                               for k in ("h0r_h0s", "gtr_h0s", "h0r_gts", "gtr_gts")},
                    "upper_residual_gt_f_minus_c000": mean(p, lambda x: x["candidates"]["c_gtf_h0r_h0s"]["upper_direction_residual_degrees"]
                                                           - x["candidates"]["c000"]["upper_direction_residual_degrees"]),
                    "disc_proximal": {n: mean(p, lambda x, n=n: x["candidates"][n]["disc:proximal_rotation_change_degrees"]) for n in ("c000", "c100", "c111", "c011", "c_gtf_h0r_h0s")}}
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("rig", "rest", "calibration", "chains", "bank", "h0_dir", "threedpw_raw", "w73_dir", "w75_dir", "w76_dir", "out_dir"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    args = parser.parse_args()

    identities = {"w73_report": _sha(args.w73_dir / "report.json"), "w73_rows_heldout": _sha(args.w73_dir / "rows_heldout.json"),
                  "w75_report": _sha(args.w75_dir / "report.json"), "w75_rows_test": _sha(args.w75_dir / "rows_test.json"),
                  "w76_report": _sha(args.w76_dir / "report.json"), "w76_rows_test": _sha(args.w76_dir / "rows_test.json")}
    for k, exp in PINNED.items():
        if identities[k] != exp:
            raise SystemExit(f"{k} identity mismatch: {identities[k]}")
    identities["h0_validation"] = _sha(args.h0_dir / "prediction_validation.npy")
    identities["h0_test"] = _sha(args.h0_dir / "prediction_test.npy")
    if not identities["h0_validation"].startswith(H0_VALIDATION_PREFIX):
        raise SystemExit("frozen H0 validation predictions changed")
    w76_report = json.loads((args.w76_dir / "report.json").read_text())
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
    if json.loads((args.h0_dir / "materialization_report.json").read_text())["bank_content_digest"] != bank.content_digest():
        raise SystemExit("H0 materialization was built from another FrameBank")
    identities["bank_content_digest"] = bank.content_digest()
    h0, _ = assemble_h0(bank, {"validation": args.h0_dir / "prediction_validation.npy", "test": args.h0_dir / "prediction_test.npy"})
    w73 = {}
    for r in json.loads((args.w73_dir / "rows_heldout.json").read_text()):
        key = (r["sequence_id"], r["frame_index"])
        if key in w73:
            raise SystemExit(f"duplicate Worklog 73 row {key}")
        w73[key] = r

    rows, accounting, assignment = collect(bank, h0, w73, target, rest, alignment, args.threedpw_raw)

    # --- Gate 1: Worklog 76 reproduction ---
    w76_rows = json.loads((args.w76_dir / "rows_test.json").read_text())
    w76 = {(r["sequence_id"], r["frame_index"]): r for r in w76_rows}
    mine = {(r["sequence_id"], r["frame_index"]): r for r in rows}
    g1 = {"rows": len(rows), "expected_rows": len(w76), "identity_set_equal": set(mine) == set(w76),
          "sequences": len({k[0] for k in mine}), "max_abs_diff": defaultdict(float), "mismatch": Counter()}
    m = g1["max_abs_diff"]
    for key, r in mine.items():
        old = w76.get(key)
        if old is None:
            continue
        if old["position"] != r["position"] or old["sample_id"] != r["sample_id"] or old["bank_order"] != r["bank_order"]:
            g1["mismatch"]["sample_identity"] += 1
        for n, f in W76_MAP.items():
            a, b = r["geometry"][n], old["geometry"][f]
            for k in ("status", "side_status", "clamp"):
                g1["mismatch"][f"{n}:{k}"] += a[k] != b[k]
            if a["status"] != "known":
                continue
            m[f"{n}:joints_cm"] = max(m[f"{n}:joints_cm"], math.dist(a["end"], b["end"]), math.dist(a["mid"], b["mid"]))
            m[f"{n}:rotations"] = max(m[f"{n}:rotations"], _qdiff(a["proximal_local"], b["proximal_local"]),
                                      _qdiff(a["distal_local"], b["distal_local"]))
            for metric in ("wrist_error", "elbow_error"):
                m[f"{n}:{metric}"] = max(m[f"{n}:{metric}"], abs(r["candidates"][n][metric] - old["candidates"][f][metric]))
        m["reference_cm"] = max(m["reference_cm"], math.dist(r["reference"]["mid"], old["reference"]["mid"]),
                                math.dist(r["reference"]["end"], old["reference"]["end"]), math.dist(r["e_gt"], old["e_gt"]))
        m["reference_rotations"] = max(m["reference_rotations"], _qdiff(r["reference"]["proximal_local"], old["reference"]["proximal_local"]),
                                       _qdiff(r["reference"]["distal_local"], old["reference"]["distal_local"]))
    w76_temporal = w76_report["test"]["pooled"]["temporal"]
    thresholds = w76_temporal["thresholds_gt_p99"]
    my_pairs, _ = _adjacent_pairs(rows)
    old_pairs, _ = _adjacent_pairs(w76_rows)
    key_of = lambda ab: (ab[0]["sequence_id"], ab[0]["frame_index"], ab[1]["frame_index"])  # noqa: E731
    g1["pairs"] = len(my_pairs)
    g1["pairs_identical_to_w76"] = [key_of(x) for x in my_pairs] == [key_of(x) for x in old_pairs]
    ref_changes = [adjacent_change(_Arm(a["reference"]), _Arm(b["reference"])) for a, b in my_pairs]
    recomputed = {t: float(np.percentile([c[t] for c in ref_changes], 99)) for t in TEMPORAL_METRICS}
    m["thresholds"] = max(abs(recomputed[t] - thresholds[t]) for t in TEMPORAL_METRICS)
    pairs = pair_records(rows, thresholds)
    for n, f in W76_MAP.items():
        for t in TEMPORAL_METRICS:
            mine_rate = sum(x["candidates"][n][f"disc:{t}"] for x in pairs) / len(pairs)
            m[f"{n}:disc_rate:{t}"] = abs(mine_rate - w76_temporal[f][t]["candidate_discontinuity_rate"])
    limits = {k: (0.0 if "disc_rate" in k else 1e-12 if k == "thresholds" else TOL_QUAT if "rotation" in k else TOL_CM) for k in m}
    failed = {k: v for k, v in m.items() if v > limits[k]}
    g1["limits"], g1["max_abs_diff"], g1["mismatch"] = limits, dict(m), {k: v for k, v in g1["mismatch"].items() if v}
    g1["passed"] = (g1["identity_set_equal"] and g1["rows"] == 6003 and g1["sequences"] == 37 and g1["pairs"] == 5504
                    and g1["pairs_identical_to_w76"] and not failed and not g1["mismatch"])
    if not g1["passed"]:
        raise SystemExit(f"STOP: Worklog 76 reproduction failed: {json.dumps(g1, default=str)}")

    # --- Gate 2: oracle-geometry contract ---
    w75 = {(r["sequence_id"], r["frame_index"]): r for r in json.loads((args.w75_dir / "rows_test.json").read_text())}
    g2 = {"max_abs_diff": defaultdict(float), "assignment_mismatches": assignment, "unavailable": Counter()}
    m2 = g2["max_abs_diff"]
    for key, r in mine.items():
        k = r["geometry"]["k_gtdir_gtr_gts"]
        if k["status"] != "known":
            g2["unavailable"]["k_gtdir_gtr_gts"] += 1
            continue
        m2["gt_owned_arm_vs_reference_cm"] = max(m2["gt_owned_arm_vs_reference_cm"], math.dist(k["mid"], r["reference"]["mid"]),
                                                 math.dist(k["end"], r["reference"]["end"]))
        m2["gt_owned_arm_rotations_vs_reference"] = max(m2["gt_owned_arm_rotations_vs_reference"],
                                                        _qdiff(k["proximal_local"], r["reference"]["proximal_local"]),
                                                        _qdiff(k["distal_local"], r["reference"]["distal_local"]))
        m2["gt_owned_arm_elbow_error_cm"] = max(m2["gt_owned_arm_elbow_error_cm"], r["candidates"]["k_gtdir_gtr_gts"]["elbow_error"])
        m2["gt_owned_arm_wrist_error_cm"] = max(m2["gt_owned_arm_wrist_error_cm"], r["candidates"]["k_gtdir_gtr_gts"]["wrist_error"])
        old = w75[key]["candidates"]
        for n, t in (("k_gtdir_h0r_h0s", "t4_gt_direction_reference"), ("c_gtf_h0r_h0s", "t3_gt_f_oracle"),
                     ("c000", "t0_h0"), ("c100", "t2_e1_visual")):
            if r["candidates"][n]["status"] == "known":
                m2[f"{n}:wrist_vs_w75_{t}_cm"] = max(m2[f"{n}:wrist_vs_w75_{t}_cm"], abs(r["candidates"][n]["wrist_error"] - old[t]["endpoint_error"]))
        for n, (d, rr, s) in FACTORIAL.items():
            want = r["gt_reach"] if rr == "gt" else r["h0_reach"]
            m2["reach_assignment"] = max(m2["reach_assignment"], abs(r["geometry"][n]["reach_fraction"] - want))
    for n in CANDIDATES:
        g2["unavailable"][n] = sum(r["geometry"][n]["status"] != "known" for r in rows)
    g2["max_abs_diff"], g2["unavailable"] = dict(m2), dict(g2["unavailable"])
    g2["passed"] = (all(v <= (TOL_QUAT if "rotation" in k else TOL_CM) for k, v in m2.items())
                    and not assignment and not g2["unavailable"].get("k_gtdir_gtr_gts"))
    if not g2["passed"]:
        raise SystemExit(f"STOP: oracle-geometry contract failed: {json.dumps(g2, default=str)}")

    # --- Counterfactual attribution ---
    common = [r for r in rows if r["reference"]["status"] == "known" and all(r["candidates"][n]["status"] == "known" for n in CANDIDATES)]
    if len(common) != len(rows):
        raise SystemExit("unavailable counterfactual rows; candidate-specific coverage handling required")  # reported, never substituted
    fac = {mname: _factorial(common, mname) for mname in ("elbow_error", "wrist_error", "upper_direction_error_degrees",
                                                          "forearm_direction_error_degrees", "bend_plane_error_degrees")}
    tfac = {mname: _factorial(pairs, mname) for mname in ("upper_direction_residual_degrees", "proximal_rotation_residual_degrees",
                                                          "elbow_motion_residual", "chain_direction_residual_degrees",
                                                          "forearm_direction_residual_degrees", "bend_side_residual_degrees",
                                                          "wrist_motion_residual", "disc:proximal_rotation_change_degrees",
                                                          "disc:elbow_displacement", "proximal_rotation_change_degrees")}
    report = {
        "schema": "animcv_framepose_visual_arm_ownership_v1", "identities": identities,
        "labels": {"runtime": "C000 (=F0) and C100 (=F2): frozen H0 + Worklog 73 E1 on the detector plane",
                   "evaluation_only": "GT reach, GT bend side, GT forward fraction (C_GT_F), GT direction (contract checks); "
                                      "never in training, selection, runtime or production"},
        "factors": {"runtime_candidates": RUNTIME, "factorial": FACTORIAL, "diagnostics": DIAGNOSTICS, "contract_checks": CONTRACT},
        "target_chain": target.to_dict(), "tolerances": {"cm": TOL_CM, "quaternion": TOL_QUAT},
        "gate1_worklog76_reproduction": g1, "gate2_oracle_geometry_contract": g2,
        "temporal_contract": {"adjacency": f"Worklog 76: consecutive bank samples exactly {BANK_STRIDE} source frames apart, both common-valid",
                              "thresholds_gt_p99_from_worklog76": thresholds,
                              "residual": "motion_residual: candidate frame-to-frame motion minus reference motion (rotations: "
                                          "quaternion distance of relative rotations; directions: |delta_c - delta_ref| in degrees; joints cm)"},
        "bootstrap_contract": BOOTSTRAP, "decision_contract": DECISION,
        "accounting": accounting,
        "static": {"candidates": _candidate_table(common, STATIC_METRICS),
                   "paired_vs_c000": _paired_vs(common, "c000", ("wrist_error", "elbow_error", "upper_direction_error_degrees", "forearm_direction_error_degrees")),
                   "paired_vs_c100": _paired_vs(common, "c100", ("wrist_error", "elbow_error", "upper_direction_error_degrees", "forearm_direction_error_degrees")),
                   "factorial": fac},
        "temporal": {"pairs": len(pairs), "reference": {t: _stats(x["reference"][t] for x in pairs) for t in (*TEMPORAL_METRICS, *ATTRIBUTION_TEMPORAL_METRICS)},
                     "candidates": {n: {t: _stats(x["candidates"][n][t] for x in pairs) for t in PAIR_METRICS}
                                    | {d: sum(x["candidates"][n][d] for x in pairs) / len(pairs) for d in DISC} for n in CANDIDATES},
                     "abs_forward_fraction_change": {k: _stats(x["abs_f_change"][k] for x in pairs) for k in ("f_gt", "f_h0", "f_e1")},
                     "factorial": tfac},
        "regimes": {"gt_bend": {lab: _regime(common, pairs, lambda r, lo=lo, hi=hi: lo <= r["gt_bend_degrees"] < hi,
                                             lambda x, lo=lo, hi=hi: all(lo <= v < hi for v in x["gt_bend_both"]))
                                for lab, (lo, hi) in BEND_REGIMES.items()},
                    "abs_gt_f": {lab: _regime(common, pairs, lambda r, lo=lo, hi=hi: lo <= abs(r["f_gt"]) < hi,
                                              lambda x, lo=lo, hi=hi: all(lo <= v < hi for v in x["abs_f_gt_both"]))
                                 for lab, (lo, hi) in F_BINS.items()}},
        "per_sequence": _per_sequence(common, pairs),
        "owner_rows": {f"{s}#{f}": ({n: {k: mine[(s, f)]["candidates"][n][k] for k in ("wrist_error", "elbow_error", "upper_direction_error_degrees",
                                                                                     "forearm_direction_error_degrees", "reach_error")}
                                     for n in CANDIDATES} | {"gt_reach": mine[(s, f)]["gt_reach"], "h0_reach": mine[(s, f)]["h0_reach"],
                                                             "gt_bend": mine[(s, f)]["gt_bend_degrees"]})
                       if (s, f) in mine else None for s, f in OWNERS},
    }
    report["decision_inputs"] = _decision(tfac["upper_direction_residual_degrees"], fac["elbow_error"],
                                          tfac["proximal_rotation_residual_degrees"], tfac["disc:proximal_rotation_change_degrees"])
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "rows_test.json").write_text(json.dumps(rows))
    (args.out_dir / "pairs_test.json").write_text(json.dumps(pairs))
    path = args.out_dir / "report.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True, default=str))
    print(json.dumps({"sha256": _sha(path), "gate1": {k: g1[k] for k in ("rows", "sequences", "pairs", "passed", "max_abs_diff")},
                      "gate2": {k: g2[k] for k in ("passed", "max_abs_diff", "unavailable")},
                      "decision_inputs": report["decision_inputs"]}, indent=1, default=str))


if __name__ == "__main__":
    main()
