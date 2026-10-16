#!/usr/bin/env python3
"""EVALUATION ONLY (Worklog 85): architecture x BEDLAM-coverage comparison on 3DPW TEST and MPI-INF-3DHP TEST.

Candidates on identical observations: J0 (frozen anchor), J1 / R1 (historical), J1X / R1X (BEDLAM-expanded).
Canonical connected-arm metrics and Worklog 77 motion residuals are the unchanged Worklog 83 definitions
(``framepose.mpi3dhp_test_transfer``); intervals are the Worklog 75/76 sequence-cluster and moving-block
bootstraps.  Both test populations have informed earlier decisions: this is controlled DEVELOPMENTAL
transfer, not independent final validation.  ``PROTOCOL`` and ``DECISION`` were committed before any
expanded-model prediction existed.
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
    BEND_REGIMES, F_BINS, SHORT_PAIR_FRAMES, STABLE, motion_metrics, regime, static_metrics, stride_for,
)

NAMES = ("j0", "j1", "j1x", "r1", "r1x")
R = ("r1", "r1x")
STATIC = ("elbow_error", "wrist_error", "upper_direction_error_degrees", "forearm_direction_error_degrees",
          "chain_direction_error_degrees", "reach_error", "side_error_degrees", "abs_bend_angle_error_degrees")
TEMPORAL = ("upper_direction_residual_degrees", "forearm_direction_residual_degrees", "chain_direction_residual_degrees",
            "elbow_motion_residual", "wrist_motion_residual", "reach_change", "bend_side_change_degrees")
AMPLITUDE = ("upper_direction_change_degrees", "forearm_direction_change_degrees", "chain_direction_change_degrees",
             "elbow_displacement", "wrist_displacement", "reach_change", "bend_side_change_degrees")
INTERACTION_METRICS = ("wrist_error", "elbow_error", "forearm_direction_error_degrees", "upper_direction_error_degrees",
                       "chain_direction_error_degrees")
COMPARISONS = (("j1x", "j1"), ("r1x", "r1"), ("r1", "j1"), ("r1x", "j1x"), ("r1x", "j0"), ("r1", "j0"), ("j1x", "j0"), ("j1", "j0"))
THREEDPW_STRIDE = 5
PROTOCOL = {
    "populations": "3DPW TEST (FrameBank, 1/6 s pairs) and the fixed Worklog 83 MPI-INF-3DHP TEST bank (0.4 s primary pairs; "
                   "TS5-TS6 0.16 s secondary, descriptive) -- both developmental, neither untouched",
    "rows": "Worklog 80 normalized_arm_targets mask; every candidate on every row (identical observations)",
    "metrics": "Worklog 83 canonical connected-arm metrics (%L, degrees) and Worklog 77 motion residuals; no rig space",
    "effects": "dJ1 = J1X - J1, dR1 = R1X - R1, interaction = dR1 - dJ1 per row (negative = R1 gains more); "
               "remaining deficit R1X - J1X vs historical R1 - J1",
    "uncertainty": "sequence(=subject)-cluster bootstrap (primary) + moving-block 10/25, 2000 replicates, seed 1337",
    "no_selection": "no row, metric, threshold or candidate is changed after predictions exist"}
# Declared before any expanded-model prediction (Worklog 84/85 CASE structure; primary population = 3DHP).
DECISION = {
    "holds": "paired mean in the stated direction AND sequence-cluster 95% interval excludes 0",
    "r1_gain": "dR1 3DHP wrist < 0 holds",
    "j1_gain": "dJ1 3DHP wrist < 0 holds",
    "r1_gains_more": "interaction (dR1 - dJ1) 3DHP wrist < 0 holds",
    "deficit_narrows": "R1X - J1X 3DHP wrist is not significantly > 0, OR its mean is <= 50% of the historical R1 - J1 mean",
    "temporal_ok": "dR1 upper-arm residual is not significantly > 0 on 3DHP (0.4 s) nor on 3DPW (1/6 s)",
    "distributed": "dR1 3DHP wrist improves in >= 4/6 sequences and no single sequence supplies >= 50% of the net gain",
    "validity": "all candidates finite, connected-arm contract <= 1e-9, identical coverage; reproduction gates passed",
    "CASE_A": "validity AND r1_gain AND r1_gains_more AND deficit_narrows AND temporal_ok AND distributed "
              "(broader supervision is a material contributor to anchored-correction viability)",
    "CASE_B": "validity AND r1_gain AND j1_gain AND NOT deficit_narrows (both benefit; direct RGB remains stronger)",
    "CASE_C": "otherwise with validity (the BEDLAM coverage intervention does not resolve the transfer deficit)",
    "CASE_D": "a data, isolation or reproduction gate failed (STOP before this runner)"}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _correction(pred, g, p, ge, gw):
    e0, w0 = pred["anchor_elbow"][p], pred["anchor_wrist"][p]
    cc = pred[f"{g}_correction"][p]
    dw = pred[f"{g}_wrist"][p] - w0
    need = gw - w0
    return {"upper_rotation_degrees": math.degrees(float(np.linalg.norm(cc[0:3]))),
            "forearm_rotation_degrees": math.degrees(float(np.linalg.norm(cc[3:6]))),
            "wrist_displacement": 100 * float(np.linalg.norm(dw)), "correction": [float(x) for x in cc],
            "wrist_alignment_with_needed": (float(np.dot(dw, need) / (np.linalg.norm(dw) * np.linalg.norm(need)))
                                            if np.linalg.norm(dw) > 1e-9 and np.linalg.norm(need) > 1e-9 else None),
            "wrist_error_change_vs_anchor": 100 * float(np.linalg.norm(pred[f"{g}_wrist"][p] - gw) - np.linalg.norm(w0 - gw)),
            "elbow_error_change_vs_anchor": 100 * float(np.linalg.norm(pred[f"{g}_elbow"][p] - ge) - np.linalg.norm(e0 - ge)),
            "depth_sign_correct": bool(np.sign(pred[f"{g}_wrist"][p][1]) == np.sign(gw[1])),
            "anchor_depth_sign_correct": bool(np.sign(w0[1]) == np.sign(gw[1]))}


def collect(samples, positions, pred, stride_of):
    rows, by_seq = [], defaultdict(list)
    for p in positions:
        if not pred["target_valid"][p]:
            continue
        s = samples[p]
        ge, gw = pred["target_elbow"][p], pred["target_wrist"][p]
        r = {"sequence_id": s.sequence_id, "frame_index": int(s.frame_index), "position": int(p), "sample_id": s.sample_id,
             **regime(ge, gw), "candidates": {}, "geometry": {"gt": {"elbow": ge.tolist(), "wrist": gw.tolist()}}}
        for g in NAMES:
            e, w = pred[f"{g}_elbow"][p], pred[f"{g}_wrist"][p]
            r["candidates"][g] = static_metrics(e, w, ge, gw)
            r["geometry"][g] = {"elbow": e.tolist(), "wrist": w.tolist()}
        r["correction"] = {g: _correction(pred, g, p, ge, gw) for g in R}
        rows.append(r)
        by_seq[s.sequence_id].append(r)
    pairs, gaps = [], Counter()
    for sid, rs in by_seq.items():
        step = stride_of(sid)
        if step is None:
            continue
        at = {r["frame_index"]: r for r in rs}
        for a in sorted(rs, key=lambda r: r["frame_index"]):
            b = at.get(a["frame_index"] + step)
            if b is None:
                gaps["row_without_evaluated_partner"] += 1
                continue
            ga, gb = a["geometry"]["gt"], b["geometry"]["gt"]
            gt = motion_metrics(ga["elbow"], ga["wrist"], gb["elbow"], gb["wrist"], ga["elbow"], ga["wrist"], gb["elbow"], gb["wrist"])
            x = {"sequence_id": sid, "frame_index": a["frame_index"], "next_frame_index": b["frame_index"], "position": a["position"],
                 "f_regime": a["f_regime"], "bend_regime": a["bend_regime"], "gt": {k: gt[k] for k in AMPLITUDE}, "candidates": {}}
            for g in NAMES:
                pa, pb = a["geometry"][g], b["geometry"][g]
                x["candidates"][g] = motion_metrics(pa["elbow"], pa["wrist"], pb["elbow"], pb["wrist"], ga["elbow"], ga["wrist"], gb["elbow"], gb["wrist"])
            x["correction_change"] = {g: {"upper_rotation_change_degrees": math.degrees(float(np.linalg.norm(
                np.subtract(b["correction"][g]["correction"][0:3], a["correction"][g]["correction"][0:3])))),
                "forearm_rotation_change_degrees": math.degrees(float(np.linalg.norm(
                    np.subtract(b["correction"][g]["correction"][3:6], a["correction"][g]["correction"][3:6]))))} for g in R}
            pairs.append(x)
    return rows, pairs, dict(gaps)


def _cmp(items, a, b, metric, bootstrap=True):
    return _contrast(items, lambda it: (it["candidates"][a][metric] - it["candidates"][b][metric])
                     if it["candidates"][a].get(metric) is not None and it["candidates"][b].get(metric) is not None else None, bootstrap)


def _interaction(items, metric, bootstrap=True):
    def fn(it):
        c = it["candidates"]
        if any(c[g].get(metric) is None for g in ("j1", "j1x", "r1", "r1x")):
            return None
        return (c["r1x"][metric] - c["r1"][metric]) - (c["j1x"][metric] - c["j1"][metric])
    return _contrast(items, fn, bootstrap)


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


def amplitude(pairs):
    out = {}
    for m in AMPLITUDE:
        gt = [x["gt"][m] for x in pairs if x["gt"][m] is not None]
        out[m] = {"gt_mean": sum(gt) / len(gt) if gt else None}
        for g in NAMES:
            c = [x["candidates"][g][m] for x in pairs if x["candidates"][g][m] is not None and x["gt"][m] is not None]
            out[m][g] = (sum(c) / len(c)) / (sum(gt) / len(gt)) if c and gt else None
    return out


def anchor_dependence(rows, pairs):
    """R1 vs R1X conditional on the frozen J0 anchor's own error (GT categories: analysis only)."""
    a = sorted(r["candidates"]["j0"]["wrist_error"] for r in rows)
    q3 = a[int(0.75 * (len(a) - 1))]
    out = {"j0_wrist_q75": q3}
    near = [r for r in rows if r["f_gt"] is not None and abs(r["f_gt"]) < STABLE]
    oop = [r for r in rows if r["f_regime"] == "out_of_plane [0.5,1]"]
    for g in R:
        c = [r["correction"][g] for r in rows]
        bad = [r for r in rows if r["candidates"]["j0"]["wrist_error"] >= q3]
        good = [r for r in rows if r["candidates"]["j0"]["wrist_error"] < q3]
        o = {k: _stats(x[k] for x in c)["mean"] for k in ("upper_rotation_degrees", "forearm_rotation_degrees", "wrist_displacement",
                                                          "wrist_alignment_with_needed", "wrist_error_change_vs_anchor",
                                                          "elbow_error_change_vs_anchor")}
        o.update({"recovery_j0_worst_quartile_wrist": _stats(r["correction"][g]["wrist_error_change_vs_anchor"] for r in bad)["mean"],
                  "recovery_share_j0_worst_quartile": sum(r["correction"][g]["wrist_error_change_vs_anchor"] < 0 for r in bad) / len(bad),
                  "damage_j0_other_rows_wrist": _stats(r["correction"][g]["wrist_error_change_vs_anchor"] for r in good)["mean"],
                  "damage_share_j0_other_rows": sum(r["correction"][g]["wrist_error_change_vs_anchor"] > 0 for r in good) / len(good),
                  "near_plane_false_depth_wrist_vs_j0": _stats(r["candidates"][g]["wrist_error"] - r["candidates"]["j0"]["wrist_error"] for r in near)["mean"],
                  "out_of_plane_depth_sign_correct": sum(r["correction"][g]["depth_sign_correct"] for r in oop) / max(len(oop), 1),
                  "out_of_plane_anchor_depth_sign_correct": sum(r["correction"][g]["anchor_depth_sign_correct"] for r in oop) / max(len(oop), 1),
                  "final_vs_anchor_error_spearman": _spearman([r["candidates"]["j0"]["wrist_error"] for r in rows],
                                                              [r["candidates"][g]["wrist_error"] for r in rows])})
        if pairs:
            o["adjacent_correction_change_upper"] = _stats(x["correction_change"][g]["upper_rotation_change_degrees"] for x in pairs)["mean"]
            o["adjacent_correction_change_forearm"] = _stats(x["correction_change"][g]["forearm_rotation_change_degrees"] for x in pairs)["mean"]
            o["excess_upper_residual_vs_j0"] = _stats(x["candidates"][g]["upper_direction_residual_degrees"]
                                                      - x["candidates"]["j0"]["upper_direction_residual_degrees"] for x in pairs)["mean"]
        out[g] = o
    out["paired_r1x_minus_r1"] = {
        "j0_worst_quartile_wrist": _contrast([r for r in rows if r["candidates"]["j0"]["wrist_error"] >= q3],
                                             lambda r: r["candidates"]["r1x"]["wrist_error"] - r["candidates"]["r1"]["wrist_error"]),
        "j0_other_rows_wrist": _contrast([r for r in rows if r["candidates"]["j0"]["wrist_error"] < q3],
                                         lambda r: r["candidates"]["r1x"]["wrist_error"] - r["candidates"]["r1"]["wrist_error"]),
        "near_plane_wrist": _contrast(near, lambda r: r["candidates"]["r1x"]["wrist_error"] - r["candidates"]["r1"]["wrist_error"]),
        "out_of_plane_wrist": _contrast(oop, lambda r: r["candidates"]["r1x"]["wrist_error"] - r["candidates"]["r1"]["wrist_error"])}
    return out


def _spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def regimes(rows, pairs):
    out = {}
    bins = [("bend", k, lambda r, k=k: r["bend_regime"] == k) for k in BEND_REGIMES] + [("f", k, lambda r, k=k: r["f_regime"] == k) for k in F_BINS]
    for kind, k, fn in bins:
        g, p = [r for r in rows if fn(r)], [x for x in pairs if fn(x)]
        o = {"rows": len(g), "pairs": len(p), "sequences": len({r["sequence_id"] for r in g})}
        if g:
            o["static_means"] = {n: {m: float(np.mean([r["candidates"][n][m] for r in g])) for m in ("elbow_error", "wrist_error", "forearm_direction_error_degrees")} for n in NAMES}
            o["contrasts"] = {f"{a}-{b}": {m: {kk: v for kk, v in _cmp(g, a, b, m, False).items() if kk in ("mean", "sequences_negative", "sequences")}
                                           for m in ("elbow_error", "wrist_error")} for a, b in COMPARISONS[:4]}
            o["interaction_wrist"] = _interaction(g, "wrist_error", False).get("mean")
        if p:
            o["temporal_means"] = {n: float(np.mean([x["candidates"][n]["upper_direction_residual_degrees"] for x in p])) for n in NAMES}
        out[f"{kind}:{k}"] = o
    return out


def per_sequence(rows, pairs):
    out = {}
    for sid in sorted({r["sequence_id"] for r in rows}):
        g = [r for r in rows if r["sequence_id"] == sid]
        p = [x for x in pairs if x["sequence_id"] == sid]
        out[sid] = {"rows": len(g), "pairs": len(p),
                    "wrist": {n: float(np.mean([r["candidates"][n]["wrist_error"] for r in g])) for n in NAMES},
                    "elbow": {n: float(np.mean([r["candidates"][n]["elbow_error"] for r in g])) for n in NAMES},
                    "upper_residual": {n: float(np.mean([x["candidates"][n]["upper_direction_residual_degrees"] for x in p])) for n in NAMES} if p else None}
    return out


def evaluate(rows, pairs, bootstrap=True):
    oop = [r for r in rows if r["f_regime"] == "out_of_plane [0.5,1]"]
    return {"rows": len(rows), "pairs": len(pairs), "sequences": len({r["sequence_id"] for r in rows}),
            "static_table": _table(rows, STATIC), "temporal_table": _table(pairs, TEMPORAL),
            "static_contrasts": {f"{a}-{b}": {m: _cmp(rows, a, b, m, bootstrap) for m in STATIC} for a, b in COMPARISONS},
            "out_of_plane_wrist": {f"{a}-{b}": _cmp(oop, a, b, "wrist_error", bootstrap) for a, b in COMPARISONS},
            "interaction": {m: _interaction(rows, m, bootstrap) for m in INTERACTION_METRICS}
                           | {"out_of_plane_wrist": _interaction(oop, "wrist_error", bootstrap)}
                           | {"upper_direction_residual_degrees": _interaction(pairs, "upper_direction_residual_degrees", bootstrap)},
            "temporal_contrasts": {f"{a}-{b}": {m: _cmp(pairs, a, b, m, bootstrap) for m in TEMPORAL} for a, b in COMPARISONS},
            "amplitude": amplitude(pairs), "regimes": regimes(rows, pairs), "per_sequence": per_sequence(rows, pairs),
            "anchor_dependence": anchor_dependence(rows, pairs)}


def _holds(c, sign=-1):
    return c.get("n", 0) > 0 and c["mean"] * sign > 0 and _excludes_zero(c)


def decide(x, d3, validity):
    s, t = x["static_contrasts"], x["temporal_contrasts"]
    hist_def, new_def = s["r1-j1"]["wrist_error"], s["r1x-j1x"]["wrist_error"]
    dr1 = s["r1x-r1"]["wrist_error"]
    share = dr1.get("most_negative_sequence", {}).get("share_of_net")
    c = {"r1_gain": _holds(dr1), "j1_gain": _holds(s["j1x-j1"]["wrist_error"]),
         "r1_gains_more": _holds(x["interaction"]["wrist_error"]),
         "deficit_narrows": (not _holds(new_def, 1)) or (hist_def["mean"] > 0 and new_def["mean"] <= 0.5 * hist_def["mean"]),
         "temporal_ok": not _holds(t["r1x-r1"]["upper_direction_residual_degrees"], 1)
         and not _holds(d3["temporal_contrasts"]["r1x-r1"]["upper_direction_residual_degrees"], 1),
         "distributed": dr1["mean"] < 0 and dr1["sequences_negative"] >= 4 and (share is None or share < 0.5),
         "validity": validity,
         "historical_deficit_wrist": hist_def["mean"], "expanded_deficit_wrist": new_def["mean"]}
    if all(c[k] for k in ("validity", "r1_gain", "r1_gains_more", "deficit_narrows", "temporal_ok", "distributed")):
        case = "A"
    elif c["validity"] and c["r1_gain"] and c["j1_gain"] and not c["deficit_narrows"]:
        case = "B"
    else:
        case = "C"
    return {**c, "mechanical_case": case}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("predictions_dir", "bank", "w83_bank", "out_dir"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise SystemExit("STOP: output directory is not empty; never overwrite experimental artifacts")
    rep = json.loads((args.predictions_dir / "prediction_report.json").read_text())
    if not rep["historical_reproduction"]["passed"]:
        raise SystemExit("STOP: predictions did not pass the historical reproduction gate")
    for k in ("3dpw", "3dhp"):
        if _sha(args.predictions_dir / f"predictions_{k}.npz") != rep["predictions_sha256"][k]:
            raise SystemExit(f"STOP: {k} predictions changed after the prediction report")
    validity = all(v["finite"] and v["length_sum_error"] <= 1e-9 and v["reach_over_one"] <= 1e-9
                   for pop in rep["connected_arm"].values() for v in pop.values())
    ext = ExternalBank.load(args.w83_bank)
    px = dict(np.load(args.predictions_dir / "predictions_3dhp.npz"))
    stride = {s.sequence_id: stride_for(s.sequence_id.split(":")[1]) for s in ext.samples}
    rows_x, pairs_x, gaps_x = collect(ext.samples, range(len(ext)), px, stride.__getitem__)
    short = {s.sequence_id: SHORT_PAIR_FRAMES.get(s.sequence_id.split(":")[1]) for s in ext.samples}
    _, pairs_s, gaps_s = collect(ext.samples, range(len(ext)), px, short.__getitem__)
    bank = load_bank(args.bank)
    p3 = dict(np.load(args.predictions_dir / "predictions_3dpw.npz"))
    rows_3, pairs_3, gaps_3 = collect(bank.samples, bank.indices("test"), p3, lambda _sid: THREEDPW_STRIDE)
    ev_x, ev_3 = evaluate(rows_x, pairs_x), evaluate(rows_3, pairs_3)
    secondary = {"pairs": len(pairs_s), "pair_gaps": gaps_s, "temporal_table": _table(pairs_s, TEMPORAL),
                 "temporal_contrasts": {f"{a}-{b}": {m: _cmp(pairs_s, a, b, m) for m in ("upper_direction_residual_degrees",
                                                                                          "forearm_direction_residual_degrees", "wrist_motion_residual")}
                                        for a, b in COMPARISONS},
                 "interaction_upper_residual": _interaction(pairs_s, "upper_direction_residual_degrees"),
                 "amplitude": amplitude(pairs_s), "anchor_dependence": anchor_dependence(rows_x, pairs_s)}
    decision = decide(ev_x, ev_3, validity)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "rows_3dhp.json").write_text(json.dumps(rows_x))
    (args.out_dir / "pairs_3dhp.json").write_text(json.dumps(pairs_x))
    (args.out_dir / "pairs_3dhp_short.json").write_text(json.dumps(pairs_s))
    (args.out_dir / "rows_3dpw.json").write_text(json.dumps(rows_3))
    (args.out_dir / "pairs_3dpw.json").write_text(json.dumps(pairs_3))
    report = {"schema": "animcv_w85_expanded_eval_v1", "protocol": PROTOCOL, "decision_contract": DECISION, "bootstrap_contract": BOOTSTRAP,
              "prediction_report_sha256": _sha(args.predictions_dir / "prediction_report.json"),
              "mpi3dhp": {**ev_x, "pair_gaps": gaps_x}, "mpi3dhp_outdoor_short_interval_secondary": secondary,
              "threedpw_test": {**ev_3, "pair_gaps": gaps_3}, "validity": validity, "decision": decision}
    (args.out_dir / "report.json").write_text(json.dumps(report, indent=1, default=str))
    print(json.dumps({"rows": {"3dhp": len(rows_x), "3dpw": len(rows_3)}, "pairs": {"3dhp": len(pairs_x), "3dhp_short": len(pairs_s),
                                                                                   "3dpw": len(pairs_3)}, "decision": decision}, indent=1))


if __name__ == "__main__":
    main()
