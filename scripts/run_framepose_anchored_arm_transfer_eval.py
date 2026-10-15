#!/usr/bin/env python3
"""EVALUATION ONLY (Worklog 83): canonical connected-arm transfer of frozen J0/J1/R0/R1 to MPI-INF-3DHP test.

Reads the predictions written by ``predict_framepose_anchored_arm_transfer.py`` (which already passed the
historical reproduction gate) and scores them in canonical connected-arm geometry only (shoulder at the
origin, unit GT total arm length, AnimCV camera axes).  No rig, no H0 heading, no IK: rig-space metrics are
NOT reported (no faithful H0 heading owner exists for 3DHP).  The identical metric code is applied to the
3DPW test split (reproduced predictions) as a within-metric reference, never pooled with 3DHP.
All population, metric, contrast, uncertainty and decision rules below were committed before any 3DHP
candidate prediction existed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_framepose_ik_gt_attribution import _stats  # noqa: E402
from run_framepose_visual_arm_ownership import _contrast, _excludes_zero  # noqa: E402
from run_framepose_visual_full_arm_ik import BOOTSTRAP  # noqa: E402

from framepose.bank import load_bank  # noqa: E402
from framepose.bedlam_supervision import ExternalBank  # noqa: E402
from framepose.mpi3dhp_test_transfer import (  # noqa: E402
    BEND_REGIMES, F_BINS, SHORT_PAIR_FRAMES, motion_metrics, regime, static_metrics, stride_for,
)

NAMES = {"j0": "J0_GEOMETRY_ONLY", "j1": "J1_GEOMETRY_PLUS_VISION", "r0": "R0_GEOMETRY_CORRECTION", "r1": "R1_VISUAL_CORRECTION"}
R = ("r0", "r1")
STATIC = ("elbow_error", "wrist_error", "upper_direction_error_degrees", "forearm_direction_error_degrees",
          "chain_direction_error_degrees", "reach_error", "side_error_degrees", "abs_bend_angle_error_degrees")
TEMPORAL = ("upper_direction_residual_degrees", "forearm_direction_residual_degrees", "chain_direction_residual_degrees",
            "elbow_motion_residual", "wrist_motion_residual", "reach_change", "bend_side_change_degrees")
AMPLITUDE = ("upper_direction_change_degrees", "forearm_direction_change_degrees", "chain_direction_change_degrees",
             "elbow_displacement", "wrist_displacement", "reach_change", "bend_side_change_degrees")
JUMP_METRICS = ("upper_direction_change_degrees", "forearm_direction_change_degrees", "chain_direction_change_degrees",
                "elbow_displacement", "wrist_displacement")
COMPARISONS = (("r1", "j1"), ("r1", "j0"), ("r1", "r0"), ("r0", "j0"), ("j1", "j0"))
THREEDPW_STRIDE = 5
# Declared before any 3DHP candidate prediction was computed (Worklog 83 one-shot protocol).
PROTOCOL = {
    "source": "MPI-INF-3DHP official test set (TS1-TS6; 6 subjects, 6 sequences, one camera each)",
    "population": "every official valid_frame row (TS1-TS4: every 10th frame; TS5-TS6: nearly every frame); rows kept iff "
                  "the Worklog 73 detector association succeeds and the pelvis is valid; evaluated rows = Worklog 80 "
                  "normalized_arm_targets mask (GT-valid AND detector-visible left shoulder/elbow/wrist); no frame-rate "
                  "subsampling of TS5-TS6 (all official rows are used, so outdoor rows weigh more than one-in-ten)",
    "pairs": "primary: evaluated rows (k, k + s) of one sequence, s = 0.4 s (TS1-TS4 10 frames at 25 fps = the official "
             "valid spacing; TS5-TS6 20 frames at 50 fps), both evaluated, never through a non-valid partner; secondary "
             "(descriptive, outdoor only): TS5-TS6 pairs 8 frames = 0.16 s apart",
    "temporal_interval": "3DHP primary pairs span 0.4 s, 3DPW pairs 1/6 s; magnitudes are compared only within a "
                         "dataset/interval, never across; TS5-TS6 50 fps is the community convention (not in the README)",
    "metrics": "canonical connected-arm geometry, positions in % of the GT total arm length, directions in degrees",
    "discontinuity_threshold": "per metric, the 99th percentile of the GT arm's own adjacent change on the same 3DHP pairs",
    "uncertainty": "sequence(=subject)-cluster bootstrap (primary) and moving-block bootstraps within sequence (10, 25 rows), "
                   "2000 replicates, seed 1337; with only 6 clusters every generalization claim is weak by construction",
    "comparisons": "R1-J1, R1-J0, R1-R0, R0-J0 (J1-J0 for context)",
    "stop_rules": "feasibility failure, historical reproduction failure, frozen-artifact identity failure",
    "no_selection": "no row is removed after predictions; no threshold or candidate set is revised after evaluation"}
DECISION = {
    "holds": "paired mean in the stated direction AND sequence-cluster 95% interval excludes 0",
    "static_vs_j1": "R1-J1 wrist < 0 holds AND R1-J1 forearm direction < 0 holds AND R1-J1 elbow is not significantly worse",
    "temporal_vs_j1": "R1-J1 upper-arm residual < 0 holds AND R1-J1 forearm residual is not significantly worse",
    "visual_gain": "(R1-R0 out-of-plane wrist < 0 holds OR R1-R0 forearm < 0 holds) AND "
                   "(R1-J0 out-of-plane wrist < 0 holds OR R1-J0 forearm < 0 holds)",
    "distributed": "R1-J1 wrist and R1-J1 upper residual each improve in >= 4/6 sequences and no single sequence supplies "
                   ">= 50% of the net gain",
    "validity": "every candidate finite on every row, connected-arm contract <= 1e-9, identical candidate coverage",
    "temporal_gap_to_j0": "reported: R1-J0 upper residual > 0 holds => R1 remains materially less stable than J0",
    "CASE_A": "static_vs_j1 AND temporal_vs_j1 AND visual_gain AND distributed AND validity",
    "CASE_B": "static_vs_j1 AND visual_gain AND validity AND NOT temporal_vs_j1",
    "CASE_C": "otherwise (no reliable external advantage)",
    "CASE_D": "feasibility / reproduction STOP (reached before this runner)"}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def collect(samples, positions, pred, stride_of):
    """Rows (evaluated positions) and same-sequence adjacent pairs exactly one stride apart."""
    rows, by_seq = [], defaultdict(list)
    ok = pred["target_valid"]
    for p in positions:
        s = samples[p]
        if not ok[p]:
            continue
        ge, gw = pred["target_elbow"][p], pred["target_wrist"][p]
        r = {"sequence_id": s.sequence_id, "frame_index": int(s.frame_index), "position": int(p), "sample_id": s.sample_id,
             **regime(ge, gw), "candidates": {}, "geometry": {"gt": {"elbow": ge.tolist(), "wrist": gw.tolist()}}}
        for g, c in NAMES.items():
            e, w = pred[f"{c}_elbow"][p], pred[f"{c}_wrist"][p]
            r["candidates"][g] = static_metrics(e, w, ge, gw)
            r["geometry"][g] = {"elbow": e.tolist(), "wrist": w.tolist()}
        r["geometry"]["anchor"] = {"elbow": pred["anchor_elbow"][p].tolist(), "wrist": pred["anchor_wrist"][p].tolist()}
        r["correction"] = correction_record(pred, p, ge, gw)
        rows.append(r)
        by_seq[s.sequence_id].append(r)
    pairs, gaps = [], Counter()
    for sid, rs in by_seq.items():
        rs.sort(key=lambda r: r["frame_index"])
        at = {r["frame_index"]: r for r in rs}
        step = stride_of(sid)
        if step is None:
            continue
        for a in rs:
            b = at.get(a["frame_index"] + step)
            if b is None:
                gaps["row_without_evaluated_partner"] += 1
                continue
            x = {"sequence_id": sid, "frame_index": a["frame_index"], "next_frame_index": b["frame_index"],
                 "position": a["position"], "next_position": b["position"],
                 "f_regime": a["f_regime"], "bend_regime": a["bend_regime"], "candidates": {}}
            ga, gb = a["geometry"]["gt"], b["geometry"]["gt"]
            gt = motion_metrics(ga["elbow"], ga["wrist"], gb["elbow"], gb["wrist"], ga["elbow"], ga["wrist"], gb["elbow"], gb["wrist"])
            x["gt"] = {k: gt[k] for k in AMPLITUDE}
            for g in NAMES:
                pa, pb = a["geometry"][g], b["geometry"][g]
                x["candidates"][g] = motion_metrics(pa["elbow"], pa["wrist"], pb["elbow"], pb["wrist"],
                                                    ga["elbow"], ga["wrist"], gb["elbow"], gb["wrist"])
            x["correction_change"] = {g: _correction_change(a["correction"][g]["correction"], b["correction"][g]["correction"]) for g in R}
            pairs.append(x)
    return rows, pairs, dict(gaps)


def _correction_change(ca, cb):
    ca, cb = np.array(ca), np.array(cb)
    return {"upper_rotation_change_degrees": math.degrees(float(np.linalg.norm(cb[0:3] - ca[0:3]))),
            "forearm_rotation_change_degrees": math.degrees(float(np.linalg.norm(cb[3:6] - ca[3:6]))),
            "rho_logit_change": abs(float(cb[6] - ca[6]))}


def correction_record(pred, p, ge, gw):
    e0, w0 = pred["anchor_elbow"][p], pred["anchor_wrist"][p]
    out = {}
    for g in R:
        c = NAMES[g]
        cc = pred[f"{c}_correction"][p]
        de, dw = pred[f"{c}_elbow"][p] - e0, pred[f"{c}_wrist"][p] - w0
        need = gw - w0
        out[g] = {"upper_rotation_degrees": math.degrees(float(np.linalg.norm(cc[0:3]))),
                  "forearm_rotation_degrees": math.degrees(float(np.linalg.norm(cc[3:6]))),
                  "rho_change": float(pred[f"{c}_rho"][p] - pred["anchor_rho"][p]), "correction": [float(x) for x in cc],
                  "elbow_displacement": 100 * float(np.linalg.norm(de)), "wrist_displacement": 100 * float(np.linalg.norm(dw)),
                  "wrist_alignment_with_needed": (float(np.dot(dw, need) / (np.linalg.norm(dw) * np.linalg.norm(need)))
                                                  if np.linalg.norm(dw) > 1e-9 and np.linalg.norm(need) > 1e-9 else None),
                  "wrist_error_change_vs_anchor": 100 * float(np.linalg.norm(pred[f"{c}_wrist"][p] - gw) - np.linalg.norm(w0 - gw)),
                  "elbow_error_change_vs_anchor": 100 * float(np.linalg.norm(pred[f"{c}_elbow"][p] - ge) - np.linalg.norm(e0 - ge))}
    return out


def _cmp(items, a, b, metric, bootstrap=True):
    return _contrast(items, lambda it: (it["candidates"][a][metric] - it["candidates"][b][metric])
                     if it["candidates"][a].get(metric) is not None and it["candidates"][b].get(metric) is not None else None, bootstrap)


def _table(items, metrics):
    out = {}
    for g in NAMES:
        out[g] = {}
        for m in metrics:
            vals = [it["candidates"][g].get(m) for it in items]
            st = _stats(vals)
            by = defaultdict(list)
            for it, v in zip(items, vals):
                if v is not None:
                    by[it["sequence_id"]].append(v)
            st["sequence_weighted_mean"] = sum(sum(v) / len(v) for v in by.values()) / len(by) if by else None
            out[g][m] = st
    return out


def jump_rates(pairs):
    """Within-dataset GT p99 thresholds; candidate-specific discontinuity = candidate exceeds it while GT does not."""
    thr = {m: float(np.percentile([x["gt"][m] for x in pairs if x["gt"][m] is not None], 99)) for m in JUMP_METRICS}
    out = {"thresholds": thr, "rates": {}}
    for g in NAMES:
        out["rates"][g] = {}
        for m in JUMP_METRICS:
            quiet = [x for x in pairs if x["gt"][m] is not None and x["gt"][m] <= thr[m] and x["candidates"][g][m] is not None]
            out["rates"][g][m] = sum(x["candidates"][g][m] > thr[m] for x in quiet) / len(quiet) if quiet else None
    return out


def amplitude(pairs):
    """Candidate adjacent change relative to the GT change on the same pairs (mean ratio of means)."""
    out = {}
    for m in AMPLITUDE:
        gt = [x["gt"][m] for x in pairs if x["gt"][m] is not None]
        out[m] = {"gt_mean": sum(gt) / len(gt) if gt else None}
        for g in NAMES:
            c = [x["candidates"][g][m] for x in pairs if x["candidates"][g][m] is not None and x["gt"][m] is not None]
            out[m][g] = {"mean": sum(c) / len(c) if c else None, "ratio_to_gt": (sum(c) / len(c)) / (sum(gt) / len(gt)) if c and gt else None}
    return out


def correction_summary(rows, pairs):
    out = {}
    for g in R:
        c = [r["correction"][g] for r in rows]
        out[g] = {k: _stats(x[k] for x in c) for k in ("upper_rotation_degrees", "forearm_rotation_degrees", "rho_change",
                                                         "elbow_displacement", "wrist_displacement", "wrist_alignment_with_needed",
                                                         "wrist_error_change_vs_anchor", "elbow_error_change_vs_anchor")}
        out[g]["share_reducing_wrist_error"] = sum(x["wrist_error_change_vs_anchor"] < 0 for x in c) / len(c)
        out[g]["share_reducing_elbow_error"] = sum(x["elbow_error_change_vs_anchor"] < 0 for x in c) / len(c)
        if pairs:
            out[g]["adjacent_correction_change"] = {k: _stats(x["correction_change"][g][k] for x in pairs)
                                                    for k in ("upper_rotation_change_degrees", "forearm_rotation_change_degrees", "rho_logit_change")}
            # Excess motion: residual added on top of the frozen anchor, and how it tracks the correction change.
            ex = [x["candidates"][g]["upper_direction_residual_degrees"] - x["candidates"]["j0"]["upper_direction_residual_degrees"] for x in pairs]
            cc = [x["correction_change"][g]["upper_rotation_change_degrees"] for x in pairs]
            gm = [x["gt"]["upper_direction_change_degrees"] for x in pairs]
            out[g]["excess_upper_residual_vs_j0"] = {**_stats(ex), "share_positive": sum(v > 0 for v in ex) / len(ex),
                                                     "spearman_with_correction_change": _spearman(ex, cc),
                                                     "spearman_correction_change_with_gt_motion": _spearman(cc, gm)}
    # Errors already present in J0: rows where the anchor is in its worst quartile, and what R1 does there.
    w0 = sorted(r["candidates"]["j0"]["wrist_error"] for r in rows)
    q3 = w0[int(0.75 * (len(w0) - 1))]
    bad = [r for r in rows if r["candidates"]["j0"]["wrist_error"] >= q3]
    good = [r for r in rows if r["candidates"]["j0"]["wrist_error"] < q3]
    out["by_anchor_error"] = {name: {"rows": len(s), "j0_wrist": _stats(r["candidates"]["j0"]["wrist_error"] for r in s)["mean"],
                                     **{f"{g}_wrist_change": _stats(r["correction"][g]["wrist_error_change_vs_anchor"] for r in s)["mean"] for g in R},
                                     **{f"{g}_share_reducing": sum(r["correction"][g]["wrist_error_change_vs_anchor"] < 0 for r in s) / len(s) for g in R}}
                              for name, s in (("j0_worst_quartile", bad), ("j0_other", good)) if s}
    return out


def _spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3:
        return None
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def per_sequence(rows, pairs):
    out = {}
    for sid in sorted({r["sequence_id"] for r in rows}):
        g = [r for r in rows if r["sequence_id"] == sid]
        p = [x for x in pairs if x["sequence_id"] == sid]
        out[sid] = {"rows": len(g), "pairs": len(p),
                    "static_means": {n: {m: float(np.mean([r["candidates"][n][m] for r in g])) for m in
                                         ("elbow_error", "wrist_error", "forearm_direction_error_degrees")} for n in NAMES},
                    "temporal_means": {n: {m: float(np.mean([x["candidates"][n][m] for x in p])) for m in
                                           ("upper_direction_residual_degrees", "forearm_direction_residual_degrees")} for n in NAMES} if p else None}
    return out


def regimes(rows, pairs):
    out = {}
    bins = [("bend", k, lambda r, k=k: r["bend_regime"] == k) for k in BEND_REGIMES] + \
           [("f", k, lambda r, k=k: r["f_regime"] == k) for k in F_BINS]
    for kind, k, fn in bins:
        g, p = [r for r in rows if fn(r)], [x for x in pairs if fn(x)]
        o = {"rows": len(g), "pairs": len(p), "sequences": len({r["sequence_id"] for r in g})}
        if g:
            o["static_means"] = {n: {m: float(np.mean([r["candidates"][n][m] for r in g])) for m in
                                     ("elbow_error", "wrist_error", "upper_direction_error_degrees", "forearm_direction_error_degrees")} for n in NAMES}
            o["contrasts"] = {f"{a}-{b}": {m: {kk: v for kk, v in _cmp(g, a, b, m, False).items()
                                               if kk in ("mean", "sequences_negative", "sequences_positive", "sequences")}
                                           for m in ("elbow_error", "wrist_error", "forearm_direction_error_degrees")} for a, b in COMPARISONS}
        if p:
            o["temporal_means"] = {n: {m: float(np.mean([x["candidates"][n][m] for x in p])) for m in
                                       ("upper_direction_residual_degrees", "forearm_direction_residual_degrees")} for n in NAMES}
        out[f"{kind}:{k}"] = o
    return out


def evaluate(rows, pairs, bootstrap=True):
    oop = [r for r in rows if r["f_regime"] == "out_of_plane [0.5,1]"]
    return {"rows": len(rows), "pairs": len(pairs), "sequences": len({r["sequence_id"] for r in rows}),
            "static_table": _table(rows, STATIC), "temporal_table": _table(pairs, TEMPORAL),
            "static_contrasts": {f"{a}-{b}": {m: _cmp(rows, a, b, m, bootstrap) for m in STATIC} for a, b in COMPARISONS},
            "out_of_plane_wrist": {f"{a}-{b}": _cmp(oop, a, b, "wrist_error", bootstrap) for a, b in COMPARISONS},
            "out_of_plane_rows": len(oop),
            "temporal_contrasts": {f"{a}-{b}": {m: _cmp(pairs, a, b, m, bootstrap) for m in TEMPORAL} for a, b in COMPARISONS},
            "amplitude": amplitude(pairs), "discontinuities": jump_rates(pairs) if pairs else None,
            "regimes": regimes(rows, pairs), "per_sequence": per_sequence(rows, pairs),
            "correction": correction_summary(rows, pairs)}


def _holds(c, sign=-1):
    return c.get("n", 0) > 0 and c["mean"] * sign > 0 and _excludes_zero(c)


def _distributed(c):
    share = c["most_negative_sequence"]["share_of_net"]
    return c["mean"] < 0 and c["sequences_negative"] >= 4 and (share is None or share < 0.5)


def decide(ev, validity):
    s, t, o = ev["static_contrasts"], ev["temporal_contrasts"], ev["out_of_plane_wrist"]
    c = {"static_vs_j1": _holds(s["r1-j1"]["wrist_error"]) and _holds(s["r1-j1"]["forearm_direction_error_degrees"])
         and not _holds(s["r1-j1"]["elbow_error"], 1),
         "temporal_vs_j1": _holds(t["r1-j1"]["upper_direction_residual_degrees"])
         and not _holds(t["r1-j1"]["forearm_direction_residual_degrees"], 1),
         "rgb_specific": _holds(o["r1-r0"]) or _holds(s["r1-r0"]["forearm_direction_error_degrees"]),
         "depth_gain_vs_j0": _holds(o["r1-j0"]) or _holds(s["r1-j0"]["forearm_direction_error_degrees"]),
         "distributed": _distributed(s["r1-j1"]["wrist_error"]) and _distributed(t["r1-j1"]["upper_direction_residual_degrees"]),
         "validity": validity,
         "r1_materially_less_stable_than_j0": _holds(t["r1-j0"]["upper_direction_residual_degrees"], 1),
         "r0_useful": _holds(s["r0-j0"]["forearm_direction_error_degrees"]) or _holds(o["r0-j0"]) or _holds(s["r0-j0"]["elbow_error"])}
    c["visual_gain"] = c["rgb_specific"] and c["depth_gain_vs_j0"]
    if all(c[k] for k in ("static_vs_j1", "temporal_vs_j1", "visual_gain", "distributed", "validity")):
        case = "A"
    elif c["static_vs_j1"] and c["visual_gain"] and c["validity"] and not c["temporal_vs_j1"]:
        case = "B"
    else:
        case = "C"
    return {**c, "mechanical_case": case}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("external_bank", "bank", "predictions_dir", "out_dir"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise SystemExit("STOP: output directory is not empty; never overwrite experimental artifacts")
    prep = json.loads((args.predictions_dir / "prediction_report.json").read_text())
    if not prep["historical_reproduction"]["passed"] or not prep["frozen_unchanged"]:
        raise SystemExit("STOP: predictions did not pass the historical reproduction gate")
    if prep["predictions_3dhp_sha256"] != _sha(args.predictions_dir / "predictions_3dhp.npz"):
        raise SystemExit("STOP: 3DHP predictions changed after the prediction report")
    ext = ExternalBank.load(args.external_bank)
    if ext.content_digest() != prep["external_bank_digest"]:
        raise SystemExit("STOP: external bank identity mismatch")
    px = dict(np.load(args.predictions_dir / "predictions_3dhp.npz"))
    validity_detail = prep["external_connected_arm"]
    # Candidates only: the internal float32 anchor (Worklog 82 input contract) is not a candidate output.
    validity = all(validity_detail[c]["finite"] and validity_detail[c]["length_sum_error"] <= 1e-9
                   and validity_detail[c]["reach_over_one"] <= 1e-9 for c in NAMES.values())
    stride = {s.sequence_id: stride_for(s.sequence_id.split(":")[1]) for s in ext.samples}
    rows_x, pairs_x, gaps_x = collect(ext.samples, range(len(ext)), px, stride.__getitem__)
    ev_x = evaluate(rows_x, pairs_x)
    decision = decide(ev_x, validity)
    short = {s.sequence_id: SHORT_PAIR_FRAMES.get(s.sequence_id.split(":")[1]) for s in ext.samples}
    _, pairs_short, gaps_short = collect(ext.samples, range(len(ext)), px, short.__getitem__)
    secondary = {"pairs": len(pairs_short), "pair_gaps": gaps_short, "temporal_table": _table(pairs_short, TEMPORAL),
                 "temporal_contrasts": {f"{a}-{b}": {m: _cmp(pairs_short, a, b, m) for m in TEMPORAL} for a, b in COMPARISONS},
                 "amplitude": amplitude(pairs_short)}

    bank = load_bank(args.bank)
    if bank.content_digest() != prep["framebank_digest"]:
        raise SystemExit("STOP: FrameBank identity mismatch")
    p3 = dict(np.load(args.predictions_dir / "predictions_3dpw_reproduced.npz"))
    from framepose.joint_arm_geometry import normalized_arm_targets
    p3["target_elbow"], p3["target_wrist"], p3["target_valid"] = normalized_arm_targets(bank.arrays["target_3d"], bank.arrays["target_valid"])
    rows_3, pairs_3, gaps_3 = collect(bank.samples, bank.indices("test"), p3, lambda _sid: THREEDPW_STRIDE)
    ev_3 = evaluate(rows_3, pairs_3)

    accounting = {"bank_rows": len(ext), "evaluated_rows": len(rows_x), "pairs": len(pairs_x), "pair_gaps": gaps_x,
                  "not_evaluated_target_invalid": int(len(ext) - len(rows_x)),
                  "per_sequence_rows": dict(Counter(r["sequence_id"] for r in rows_x)),
                  "per_sequence_pairs": dict(Counter(x["sequence_id"] for x in pairs_x)),
                  "candidate_coverage": {g: len(rows_x) for g in NAMES}, "connected_arm": validity_detail}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "rows_3dhp.json").write_text(json.dumps(rows_x))
    (args.out_dir / "pairs_3dhp.json").write_text(json.dumps(pairs_x))
    report = {"schema": "animcv_framepose_anchored_arm_transfer_eval_v1", "protocol": PROTOCOL, "decision_contract": DECISION,
              "bootstrap_contract": BOOTSTRAP, "prediction_report_sha256": _sha(args.predictions_dir / "prediction_report.json"),
              "accounting_3dhp": accounting, "mpi3dhp": ev_x, "decision": decision,
              "mpi3dhp_outdoor_short_interval_secondary": secondary,
              "threedpw_test_reference": {"rows": len(rows_3), "pairs": len(pairs_3), "pair_gaps": gaps_3, **ev_3},
              "rig_space": "not reported: no faithful H0 heading owner / BaseRig alignment input exists for 3DHP"}
    (args.out_dir / "report.json").write_text(json.dumps(report, indent=1, default=str))
    print(json.dumps({"accounting": {k: v for k, v in accounting.items() if k != "connected_arm"}, "decision": decision}, indent=1))


if __name__ == "__main__":
    main()
