#!/usr/bin/env python3
"""EVALUATION ONLY (Worklog 81): which J0/J1 geometry factor carries J1's static gain and temporal cost.

Frozen Worklog 80 J0/J1 predictions only (no training).  Each model's ONE connected arm gives chain
direction, reach and bend side through the Worklog 62 measure_chain; the 2x2x2 factorial V_drs combines
J_d direction, J_r reach and J_s side through the unchanged Worklog 76 full_arm (Worklog 61 solve,
Worklog 62 side rule).  V000 = J0 and V111 = J1 exactly; the mixed candidates are attribution tools only.
Metrics, temporal contract, Worklog 76 thresholds and sequence-aware statistics are the Worklog 77-80 ones.

Order: identities -> Gate 1 (Worklog 80 parity: population, pairs, J0/J1/B0/B1 geometry and metrics,
reference, thresholds) -> Gate 2 (factor assignment and validity) -> attribution.
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
from run_framepose_arm_geometry_owner_eval import _table, pair_records, side_error  # noqa: E402
from run_framepose_ik_gt_attribution import _stats  # noqa: E402
from run_framepose_joint_arm_geometry_eval import j_measure  # noqa: E402
from run_framepose_visual_arm_ownership import _contrast, _excludes_zero, evaluate, row_record  # noqa: E402
from run_framepose_visual_full_arm_ik import BEND_REGIMES, BOOTSTRAP, NAMES, STABLE, _qdiff, _sha, plane_for  # noqa: E402
from run_root_motion_contact_h0_replay import _matched_oracle_sequence  # noqa: E402

from framepose.bank import load_bank  # noqa: E402
from motion.animation_semantics_v2_bridge import current_root_orientation  # noqa: E402
from pose.framepose_bridge import assemble_h0, build_h0_lifted_sequence, sequence_frame_positions, sequence_ids_in_split  # noqa: E402
from retarget.framepose_fk import derive_rig_alignment, load_rig_fk_calibration  # noqa: E402
from retarget.framepose_ik_gt_attribution import _runtime_side  # noqa: E402
from retarget.framepose_joint_factor_attribution import FACTORIAL, flat_effects, owned  # noqa: E402
from retarget.framepose_target_rest import load_target_rest_pose  # noqa: E402
from retarget.framepose_two_bone_ik import load_ik_chain_set, target_chain_geometry  # noqa: E402
from retarget.framepose_visual_arm_ownership import RESIDUAL_METRICS  # noqa: E402
from retarget.framepose_visual_full_arm_ik import TEMPORAL_METRICS, full_arm  # noqa: E402

PINNED = {"w76_report": "dad46288436ac0fe28fea9fab7ad6882c6a39fd476e717af4eaee497ab096a2b",
          "w80_training_report": "5d6e974ac72d194e2c956fa3a9ec586a63e97e3238971328444d24f8eb0e347d",
          "w80_predictions": "eaf3f57c64acd9c3eb086cd84635884aef795e0fd635ea7d3a5776d08aead00f",
          "w80_eval_report": "a27eeab2721f180f4fc0ed6bd31777128e2d1fef3f23786587b6bad885ef60e5",
          "w80_rows_test": "8835e79a9a70038294ea46b5afd5e735275b3067a363304a937c21bd4eed60be",
          "w80_pairs_test": "b25e7c979ce48886a26268cebee485fce690caa658fcbcab6ee9b098121cb962"}
J = {"0": "J0_GEOMETRY_ONLY", "1": "J1_GEOMETRY_PLUS_VISION"}
BASE = {"b0": "c000", "b1": "c100"}
V = tuple(FACTORIAL)
KEEP = (*V, "b0", "b1", "c111")
STATIC = ("wrist_error", "elbow_error", "chain_direction_error_degrees", "upper_direction_error_degrees",
          "forearm_direction_error_degrees", "reach_error", "side_error_degrees", "abs_bend_angle_error_degrees",
          "bend_plane_error_degrees", "proximal_rotation_vs_gt_degrees", "distal_rotation_vs_gt_degrees")
EFFECT_STATIC = ("wrist_error", "elbow_error", "forearm_direction_error_degrees", "upper_direction_error_degrees",
                 "chain_direction_error_degrees", "reach_error", "side_error_degrees", "abs_bend_angle_error_degrees")
TEMPORAL = ("upper_direction_residual_degrees", "proximal_rotation_residual_degrees", "forearm_direction_residual_degrees",
            "elbow_motion_residual", "wrist_motion_residual", "disc:proximal_rotation_change_degrees", "disc:elbow_displacement",
            "reach_change", "side_change_degrees", "chain_direction_change_degrees")
F_BINS = {"near_plane [0,sin10)": (0, STABLE), "intermediate": (STABLE, 0.5), "out_of_plane [0.5,1]": (0.5, 1.01)}
OWNER_FRAMES = (("3dpw:downtown_bar_00:actor0", 514), ("3dpw:downtown_bar_00:actor0", 775),
                ("3dpw:downtown_sitOnStairs_00:actor0", 325), ("3dpw:flat_packBags_00:actor0", 446),
                ("3dpw:downtown_runForBus_01:actor0", 185))
FACTORS = ("direction", "reach", "side")
INTERACTIONS = ("direction_x_reach", "direction_x_side", "reach_x_side", "three_way")
# Predeclared before any factorial result was computed.
DECISION = {
    "primary_static_benefit": "forearm_direction_error_degrees (the J1-J0 static gain whose interval excluded 0 in Worklog 80); "
                              "secondary: wrist_error overall and in out-of-plane rows",
    "primary_temporal_cost": "upper_direction_residual_degrees; secondary: disc:proximal_rotation_change_degrees",
    "total_significant": "J1-J0 (V111-V000) sequence-cluster 95% interval excludes 0",
    "dominant_factor": "the factor whose main effect has the sign of the total, interval excluding 0, and >= 50% of the total",
    "material_interaction": "any two/three-way interaction with |mean| >= 25% of |total| and interval excluding 0",
    "owner": "dominant factor if one exists and no interaction is material, else 'distributed'",
    "A_shared_owner": "the primary static benefit and the primary temporal cost have the same single-factor owner",
    "B_different_owners": "both have single-factor owners and they differ",
    "C_distributed": "either is 'distributed'",
    "D_inconclusive": "either total is not significant or factorial arms lose validity (unavailable rows)"}


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
            m = {k: j_measure(preds[f"{c}_elbow"][p], preds[f"{c}_wrist"][p], orient.yaw_radians, alignment) for k, c in J.items()}
            if m["0"] is None or m["1"] is None:
                raise SystemExit(f"{key}: J arm unmeasurable (Worklog 80 had none)")
            arms = {b: ev["arms"][c] for b, c in BASE.items()}
            arms["c111"] = ev["arms"]["c111"]
            for name, (d, r, s) in FACTORIAL.items():
                measure, direction = owned(m["0"], m["1"], d, r, s)
                arms[name] = full_arm(name, measure, direction, target, rest)
            ev["arms"] = arms
            conf = min(float(bank.arrays["input_2d"][p][j, 2]) for j in (11, 12, 13))
            r = row_record(sid, frame.frame_index, p, order, bank.samples[p].sample_id, conf, orient.yaw_radians, pred["chain_f_e1"], ev)
            for n, a in arms.items():
                if r["candidates"][n]["status"] == "known":
                    r["candidates"][n]["side_error_degrees"] = side_error(a, ev["gt"])
                acc[f"{n}:{a.status}"] += 1
                acc[f"{n}:side:{a.side_status}"] += 1
            r["factor_sources"] = {k: {"reach": mm.reach_fraction, "side": list(mm.side) if mm.side else None,
                                       "direction": list(mm.direction), "degenerate": mm.degenerate} for k, mm in m.items()}
            # Exact factor assignment check (reach source, side rule).
            for name, (d, rr, s) in FACTORIAL.items():
                a, want_side = arms[name], _runtime_side(m[s], target, rest, arms[name].direction)[0]
                acc[f"assignment_mismatch:{name}"] += int(a.reach_fraction != m[rr].reach_fraction or a.direction != tuple(
                    x / math.sqrt(sum(y * y for y in m[d].direction)) for x in m[d].direction) or a.side != want_side)
            acc["common_eligible"] += 1
            rows.append(r)
        accounting[sid] = dict(acc)
    return rows, accounting


def _y(item, metric):
    return {v: item["candidates"][v][metric] for v in V}


def _effect_block(items, metric, bootstrap=True):
    keys = list(flat_effects({v: 0.0 for v in V}))
    return {k: _contrast(items, lambda it, k=k: flat_effects(_y(it, metric))[k], bootstrap) for k in keys}


def _owner(block):
    total = block["total"]
    sig = _excludes_zero(total)
    out = {"total": total["mean"], "total_significant": sig, "main_share": {}, "dominant": None, "material_interactions": []}
    for f in FACTORS:
        c = block[f"main:{f}"]
        out["main_share"][f] = c["mean"] / total["mean"] if total["mean"] else None
        if sig and _excludes_zero(c) and c["mean"] * total["mean"] > 0 and abs(c["mean"]) >= 0.5 * abs(total["mean"]):
            if out["dominant"] is None or abs(c["mean"]) > abs(block[f"main:{out['dominant']}"]["mean"]):
                out["dominant"] = f
    for i in INTERACTIONS:
        c = block[f"interaction:{i}"]
        if _excludes_zero(c) and abs(c["mean"]) >= 0.25 * abs(total["mean"]):
            out["material_interactions"].append(i)
    out["owner"] = (out["dominant"] if out["dominant"] and not out["material_interactions"] else "distributed") if sig else "not_significant"
    return out


def _regime_effects(rows, pairs, row_ok, pair_ok):
    g = [r for r in rows if row_ok(r)]
    p = [x for x in pairs if pair_ok(x)]
    out = {"rows": len(g), "pairs": len(p), "sequences": len({r["sequence_id"] for r in g})}
    keep = ("mean", "sequence_weighted_mean", "sequences_negative", "sequences_positive", "most_negative_sequence", "most_positive_sequence")
    for metric in ("elbow_error", "wrist_error", "forearm_direction_error_degrees", "chain_direction_error_degrees", "reach_error"):
        out[metric] = {"means": {v: sum(r["candidates"][v][metric] for r in g) / len(g) for v in (*V, "b1")} if g else None,
                       "effects": {k: {kk: vv for kk, vv in c.items() if kk in keep} for k, c in _effect_block(g, metric, False).items()} if g else None}
    if p:
        metric = "upper_direction_residual_degrees"
        out[metric] = {"means": {v: sum(x["candidates"][v][metric] for x in p) / len(p) for v in (*V, "b1")},
                       "effects": {k: {kk: vv for kk, vv in c.items() if kk in keep} for k, c in _effect_block(p, metric, False).items()}}
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("rig", "rest", "calibration", "chains", "bank", "h0_dir", "threedpw_raw", "w73_dir", "w76_dir",
                 "w80_joint_dir", "w80_eval_dir", "out_dir"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    args = parser.parse_args()
    identities = {"w76_report": _sha(args.w76_dir / "report.json"),
                  "w80_training_report": _sha(args.w80_joint_dir / "training_report.json"),
                  "w80_predictions": _sha(args.w80_joint_dir / "predictions_3dpw.npz"),
                  "w80_eval_report": _sha(args.w80_eval_dir / "report.json"),
                  "w80_rows_test": _sha(args.w80_eval_dir / "rows_test.json"),
                  "w80_pairs_test": _sha(args.w80_eval_dir / "pairs_test.json"),
                  "w73_rows_heldout": _sha(args.w73_dir / "rows_heldout.json"),
                  **{f"w80_checkpoint_{c}": _sha(args.w80_joint_dir / f"{c}.pt") for c in J.values()}}
    for k, exp in PINNED.items():
        if identities[k] != exp:
            raise SystemExit(f"{k} identity mismatch: {identities[k]}")
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
    h0_arr, _ = assemble_h0(bank, {"validation": args.h0_dir / "prediction_validation.npy", "test": args.h0_dir / "prediction_test.npy"})
    w73 = {(r["sequence_id"], r["frame_index"]): r for r in json.loads((args.w73_dir / "rows_heldout.json").read_text())}
    preds = dict(np.load(args.w80_joint_dir / "predictions_3dpw.npz"))
    rows, accounting = collect("test", bank, h0_arr, w73, preds, target, rest, alignment, args.threedpw_raw)

    # --- Gate 1: Worklog 80 parity ---
    old_rows = json.loads((args.w80_eval_dir / "rows_test.json").read_text())
    old = {(r["sequence_id"], r["frame_index"]): r for r in old_rows}
    mine = {(r["sequence_id"], r["frame_index"]): r for r in rows}
    g1 = {"rows": len(rows), "identity_set_equal": set(mine) == set(old), "sequences": len({k[0] for k in mine}),
          "max_abs_diff": defaultdict(float), "mismatch": Counter()}
    m = g1["max_abs_diff"]
    for key, r in mine.items():
        o = old.get(key)
        if o is None:
            continue
        for new, ref in (("v000", "j0"), ("v111", "j1"), ("b0", "b0"), ("b1", "b1")):
            a, b = r["geometry"][new], o["geometry"][ref]
            for k in ("status", "side_status", "clamp", "reach_fraction", "direction", "side"):
                g1["mismatch"][f"{new}:{k}"] += a[k] != b[k]
            m[f"{new}:joints_cm"] = max(m[f"{new}:joints_cm"], math.dist(a["end"], b["end"]), math.dist(a["mid"], b["mid"]))
            m[f"{new}:rotations"] = max(m[f"{new}:rotations"], _qdiff(a["proximal_local"], b["proximal_local"]), _qdiff(a["distal_local"], b["distal_local"]))
            for metric in STATIC:
                x, y = r["candidates"][new].get(metric), o["candidates"][ref].get(metric)
                if x is not None and y is not None:
                    m[f"{new}:metrics"] = max(m[f"{new}:metrics"], abs(x - y))
        m["reference_cm"] = max(m["reference_cm"], math.dist(r["reference"]["mid"], o["reference"]["mid"]), math.dist(r["e_gt"], o["e_gt"]))
    thresholds = w76["test"]["pooled"]["temporal"]["thresholds_gt_p99"]
    if thresholds != w80["jump_thresholds_worklog76"]:
        raise SystemExit("STOP: jump thresholds differ from Worklog 80")
    pairs = pair_records(rows, thresholds, KEEP)
    old_pairs = json.loads((args.w80_eval_dir / "pairs_test.json").read_text())
    g1["pairs"] = len(pairs)
    g1["pairs_identical"] = [(x["sequence_id"], x["frame_index"], x["next_frame_index"]) for x in pairs] == \
        [(x["sequence_id"], x["frame_index"], x["next_frame_index"]) for x in old_pairs]
    if g1["pairs_identical"]:
        for x, o in zip(pairs, old_pairs):
            for new, ref in (("v000", "j0"), ("v111", "j1"), ("b1", "b1")):
                for t in TEMPORAL:
                    m[f"{new}:temporal"] = max(m[f"{new}:temporal"], abs(x["candidates"][new][t] - o["candidates"][ref][t]))
            m["reference_motion"] = max(m["reference_motion"], max(abs(x["reference"][t] - o["reference"][t]) for t in x["reference"]))
    for new, ref in (("v000", "j0"), ("v111", "j1")):
        for metric in ("elbow_error", "wrist_error", "forearm_direction_error_degrees"):
            m[f"{new}:w80_report_mean:{metric}"] = abs(sum(r["candidates"][new][metric] for r in rows) / len(rows)
                                                       - w80["static"]["candidates"][ref][metric]["mean"])
        m[f"{new}:w80_report_upper_residual"] = abs(sum(x["candidates"][new]["upper_direction_residual_degrees"] for x in pairs) / len(pairs)
                                                    - w80["temporal"]["candidates"][ref]["upper_direction_residual_degrees"]["mean"])
    limits = {k: (1e-9 if ("report" in k or "temporal" in k or "metrics" in k or "motion" in k) else 1e-9 if "rotation" in k else 0.0) for k in m}
    failed = {k: v for k, v in m.items() if v > limits[k]}
    g1.update({"limits": limits, "max_abs_diff": dict(m), "mismatch": {k: v for k, v in g1["mismatch"].items() if v}})
    g1["passed"] = (g1["identity_set_equal"] and g1["rows"] == 6003 and g1["sequences"] == 37 and g1["pairs"] == 5504
                    and g1["pairs_identical"] and not failed and not g1["mismatch"])
    if not g1["passed"]:
        raise SystemExit(f"STOP: Worklog 80 parity failed: {json.dumps(g1, default=str)}")

    # --- Gate 2: factor assignment and validity ---
    totals = Counter()
    for acc in accounting.values():
        totals.update(acc)
    g2 = {"assignment_mismatches": {k: v for k, v in totals.items() if k.startswith("assignment_mismatch") and v},
          "unavailable": {v: sum(r["candidates"][v]["status"] != "known" for r in rows) for v in V},
          "side_status": {v: dict(Counter(r["geometry"][v]["side_status"] for r in rows)) for v in V},
          "clamps": {v: dict(Counter(r["geometry"][v]["clamp"] for r in rows)) for v in V},
          "integrity_max": max(r["candidates"][v][k] for r in rows for v in V for k in (
              "integrity_upper_length_residual", "integrity_forearm_length_residual", "integrity_endpoint_reconstruction_residual",
              "integrity_ik_to_fk_residual") if r["candidates"][v]["status"] == "known")}
    g2["passed"] = not g2["assignment_mismatches"] and g2["integrity_max"] <= 1e-9
    if not g2["passed"]:
        raise SystemExit(f"STOP: factor assignment / geometry contract failed: {json.dumps(g2, default=str)}")
    common = [r for r in rows if all(r["candidates"][n]["status"] == "known" for n in KEEP)]
    validity_lost = len(common) != len(rows)

    # --- Attribution ---
    static_effects = {mm: _effect_block(common, mm) for mm in EFFECT_STATIC}
    temporal_effects = {mm: _effect_block(pairs, mm) for mm in TEMPORAL}
    oop = [r for r in common if abs(r["f_gt"]) >= 0.5]
    static_effects["wrist_error|out_of_plane"] = _effect_block(oop, "wrist_error")
    static_effects["forearm_direction_error_degrees|out_of_plane"] = _effect_block(oop, "forearm_direction_error_degrees")
    owners = {mm: _owner(b) for mm, b in {**static_effects, **temporal_effects}.items()}
    primary_static, primary_temporal = owners["forearm_direction_error_degrees"], owners["upper_direction_residual_degrees"]
    if validity_lost or not primary_static["total_significant"] or not primary_temporal["total_significant"]:
        hypothesis = "D"
    elif "distributed" in (primary_static["owner"], primary_temporal["owner"]):
        hypothesis = "C"
    elif primary_static["owner"] == primary_temporal["owner"]:
        hypothesis = "A"
    else:
        hypothesis = "B"
    regimes = {"gt_bend": {lab: _regime_effects(common, pairs, lambda r, lo=lo, hi=hi: lo <= r["gt_bend_degrees"] < hi,
                                                lambda x, lo=lo, hi=hi: all(lo <= v < hi for v in x["gt_bend_both"])) for lab, (lo, hi) in BEND_REGIMES.items()},
               "abs_gt_f": {lab: _regime_effects(common, pairs, lambda r, lo=lo, hi=hi: lo <= abs(r["f_gt"]) < hi,
                                                 lambda x, lo=lo, hi=hi: all(lo <= v < hi for v in x["abs_f_gt_both"])) for lab, (lo, hi) in F_BINS.items()}}
    report = {
        "schema": "animcv_framepose_joint_factor_attribution_v1", "identities": identities,
        "labels": {"runtime_models": "V000 = J0 and V111 = J1 (Worklog 80, frozen)",
                   "evaluation_only": "mixed V candidates (attribution tools, not runtime architectures), C111, GT reference"},
        "factorial": FACTORIAL, "gate1_worklog80_parity": g1, "gate2_factor_contract": g2, "accounting_totals": dict(totals),
        "decision_contract": DECISION, "bootstrap_contract": BOOTSTRAP, "jump_thresholds_worklog76": thresholds,
        "static": {"candidates": _table(common, KEEP, STATIC), "effects": static_effects},
        "temporal": {"pairs": len(pairs), "candidates": _table(pairs, KEEP, (*RESIDUAL_METRICS, *TEMPORAL_METRICS, "reach_change",
                                                                               "side_change_degrees", "chain_direction_change_degrees"))
                     | {"disc_rates": {n: {t: sum(x["candidates"][n][f"disc:{t}"] for x in pairs) / len(pairs) for t in TEMPORAL_METRICS} for n in KEEP}},
                     "reference": {t: _stats(x["reference"][t] for x in pairs) for t in ("upper_direction_change_degrees", "chain_direction_change_degrees",
                                                                                        "elbow_displacement", "reach_change")},
                     "effects": temporal_effects},
        "owners": owners, "hypothesis": hypothesis, "validity_lost": validity_lost,
        "w80_oracle_decomposition_context": w80["static"]["decomposition"],
        "regimes": regimes,
        "owner_frames": {f"{s}#{f}": ({v: {k: mine[(s, f)]["candidates"][v].get(k) for k in ("wrist_error", "elbow_error", "forearm_direction_error_degrees",
                                                                                           "chain_direction_error_degrees", "reach_error", "side_error_degrees")}
                                       for v in KEEP} | {"gt_bend": mine[(s, f)]["gt_bend_degrees"], "f_gt": mine[(s, f)]["f_gt"]})
                         if (s, f) in mine else None for s, f in OWNER_FRAMES},
    }
    vrows, _ = collect("validation", bank, h0_arr, w73, preds, target, rest, alignment, args.threedpw_raw)
    report["validation"] = {"note": "the split that selected the Worklog 80 epochs; reported separately", "rows": len(vrows),
                            "static_means": {v: {mm: sum(r["candidates"][v][mm] for r in vrows) / len(vrows) for mm in
                                                 ("elbow_error", "wrist_error", "forearm_direction_error_degrees")} for v in V},
                            "effects_mean": {mm: {k: c["mean"] for k, c in _effect_block(vrows, mm, False).items()} for mm in
                                             ("elbow_error", "wrist_error", "forearm_direction_error_degrees")}}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "rows_test.json").write_text(json.dumps(rows))
    (args.out_dir / "pairs_test.json").write_text(json.dumps(pairs))
    path = args.out_dir / "report.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True, default=str))
    print(json.dumps({"sha256": _sha(path), "gate1": {k: g1[k] for k in ("rows", "pairs", "passed")}, "gate2": g2,
                      "owners": {k: {kk: v[kk] for kk in ("owner", "total", "main_share", "material_interactions")} for k, v in owners.items()},
                      "hypothesis": hypothesis}, indent=1, default=str))


if __name__ == "__main__":
    main()
