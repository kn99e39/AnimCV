#!/usr/bin/env python3
"""EVALUATION ONLY (Worklog 86): connected-limb transfer -- right arm and right leg G1 vs G0, left-arm reference.

Canonical two-segment metrics (Worklog 83 definitions, chain-agnostic): middle = elbow/knee, distal =
wrist/ankle, proximal segment = upper arm/thigh, distal segment = forearm/shank.  Motion residuals against
the GT motion of the same pair (3DPW 1/6 s; 3DHP 0.4 s primary, TS5-TS6 0.16 s descriptive).  Intervals:
Worklog 75/76 sequence-cluster (primary) and moving-block bootstraps.  Both populations are developmental.
``PROTOCOL`` and ``DECISION`` were committed before any right-arm / leg model was trained.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict
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

CANDIDATES = {"left_arm": ("j0", "g0x", "j1x"), "right_arm": ("g0", "g1"), "right_leg": ("g0", "g1")}
CONTRASTS = {"left_arm": (("j1x", "g0x"), ("j1x", "j0"), ("g0x", "j0")), "right_arm": (("g1", "g0"),), "right_leg": (("g1", "g0"),)}
PRIMARY = {"left_arm": ("j1x", "g0x"), "right_arm": ("g1", "g0"), "right_leg": ("g1", "g0")}
STATIC = ("elbow_error", "wrist_error", "upper_direction_error_degrees", "forearm_direction_error_degrees",
          "chain_direction_error_degrees", "reach_error", "side_error_degrees", "abs_bend_angle_error_degrees")
LABELS = {"elbow_error": "middle_joint_error", "wrist_error": "distal_joint_error", "upper_direction_error_degrees": "proximal_segment_deg",
          "forearm_direction_error_degrees": "distal_segment_deg", "chain_direction_error_degrees": "chain_deg",
          "upper_direction_residual_degrees": "proximal_segment_residual_deg", "forearm_direction_residual_degrees": "distal_segment_residual_deg",
          "elbow_motion_residual": "middle_motion_residual", "wrist_motion_residual": "distal_motion_residual"}
TEMPORAL = ("upper_direction_residual_degrees", "forearm_direction_residual_degrees", "chain_direction_residual_degrees",
            "elbow_motion_residual", "wrist_motion_residual", "reach_change", "bend_side_change_degrees")
AMPLITUDE = ("upper_direction_change_degrees", "forearm_direction_change_degrees", "elbow_displacement", "wrist_displacement")
JUMPS = ("upper_direction_change_degrees", "forearm_direction_change_degrees", "wrist_displacement")
PROTOCOL = {
    "chains": "reference left arm (frozen J1X vs matched G0X, plus historical J0); right arm and right leg (data-only LEG_RULE) G0 vs G1",
    "populations": "3DPW TEST FrameBank rows and the fixed Worklog 83 MPI-INF-3DHP TEST bank; rows = chain-valid (GT valid AND detector-visible P, M, D)",
    "metrics": "canonical connected two-segment geometry, %L of the GT chain length and degrees; Worklog 77 motion residuals",
    "pairs": "3DPW consecutive FrameBank rows 5 frames (1/6 s); 3DHP rows 0.4 s apart (primary), TS5-TS6 0.16 s (descriptive)",
    "uncertainty": "sequence(=subject)-cluster bootstrap primary, moving-block 10/25 reported; 2000 replicates, seed 1337",
    "no_selection": "no row, metric, threshold or candidate is changed after limb predictions exist"}
DECISION = {
    "primary_metric": "distal-joint normalized error (wrist / ankle), G1 - G0 per row",
    "strong": "mean < 0 AND sequence-cluster 95% interval excludes 0 AND improves in >= 2/3 of sequences",
    "regression": "mean > 0 AND sequence-cluster interval excludes 0",
    "concentration_ok": "no single sequence supplies >= 50% of the net distal gain (where strong)",
    "middle_ok": "G1 - G0 middle-joint error is not a regression on either population",
    "tail_ok": "p95 distal error of G1 <= p95 of G0 + 2.0 %L on both populations",
    "temporal_ok": "3DHP 0.4 s: G1 - G0 proximal-segment residual is not a regression; 3DPW: G1 - G0 proximal residual mean "
                   "<= max(1.5 x ref, ref + 1.0 deg), ref = J1X - G0X left-arm penalty on 3DPW (the accepted reference)",
    "validity": "every candidate finite on every row, connected-chain contract <= 1e-9, identical coverage",
    "chain_transfers": "validity AND strong on >= 1 population (with concentration_ok there) AND no distal regression on either "
                       "AND middle_ok AND tail_ok AND temporal_ok",
    "CASE_A": "right arm AND right leg transfer", "CASE_B": "exactly one transfers (expansion STOPS)",
    "CASE_C": "neither transfers (expansion STOPS; return to single-frame accuracy)", "CASE_D": "a contract gate failed (before this runner)",
    "reference": "the same chain_transfers test is applied to the left arm (J1X vs G0X) and reported as calibration, not as a CASE input"}


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def collect(samples, positions, pred, chain, cands, stride_of):
    rows, by_seq = [], defaultdict(list)
    ok = pred[f"{chain}_target_valid"]
    for p in positions:
        if not ok[p]:
            continue
        s = samples[p]
        ge, gw = pred[f"{chain}_target_elbow"][p], pred[f"{chain}_target_wrist"][p]
        r = {"sequence_id": s.sequence_id, "frame_index": int(s.frame_index), "position": int(p), "sample_id": s.sample_id,
             **regime(ge, gw), "candidates": {}, "geometry": {"gt": {"elbow": ge.tolist(), "wrist": gw.tolist()}}}
        for g in cands:
            e, w = pred[f"{chain}_{g}_elbow"][p], pred[f"{chain}_{g}_wrist"][p]
            r["candidates"][g] = static_metrics(e, w, ge, gw)
            r["geometry"][g] = {"elbow": e.tolist(), "wrist": w.tolist()}
        r["confidence_min"] = None
        rows.append(r)
        by_seq[s.sequence_id].append(r)
    pairs = []
    for sid, rs in by_seq.items():
        step = stride_of(sid)
        if step is None:
            continue
        at = {r["frame_index"]: r for r in rs}
        for a in sorted(rs, key=lambda r: r["frame_index"]):
            b = at.get(a["frame_index"] + step)
            if b is None:
                continue
            ga, gb = a["geometry"]["gt"], b["geometry"]["gt"]
            gt = motion_metrics(ga["elbow"], ga["wrist"], gb["elbow"], gb["wrist"], ga["elbow"], ga["wrist"], gb["elbow"], gb["wrist"])
            x = {"sequence_id": sid, "frame_index": a["frame_index"], "next_frame_index": b["frame_index"], "position": a["position"],
                 "f_regime": a["f_regime"], "bend_regime": a["bend_regime"], "gt": {k: gt[k] for k in AMPLITUDE}, "candidates": {}}
            for g in cands:
                pa, pb = a["geometry"][g], b["geometry"][g]
                x["candidates"][g] = motion_metrics(pa["elbow"], pa["wrist"], pb["elbow"], pb["wrist"], ga["elbow"], ga["wrist"], gb["elbow"], gb["wrist"])
            pairs.append(x)
    return rows, pairs


def _cmp(items, a, b, m, bootstrap=True):
    return _contrast(items, lambda it: (it["candidates"][a][m] - it["candidates"][b][m])
                     if it["candidates"][a].get(m) is not None and it["candidates"][b].get(m) is not None else None, bootstrap)


def _table(items, cands, metrics):
    out = {}
    for g in cands:
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


def _jumps(pairs, cands):
    if not pairs:
        return None
    thr = {m: float(np.percentile([x["gt"][m] for x in pairs if x["gt"][m] is not None], 99)) for m in JUMPS}
    rates = {}
    for g in cands:
        rates[g] = {}
        for m in JUMPS:
            quiet = [x for x in pairs if x["gt"][m] is not None and x["gt"][m] <= thr[m] and x["candidates"][g][m] is not None]
            rates[g][m] = sum(x["candidates"][g][m] > thr[m] for x in quiet) / len(quiet) if quiet else None
    return {"thresholds": thr, "rates": rates}


def _amplitude(pairs, cands):
    out = {}
    for m in AMPLITUDE:
        gt = [x["gt"][m] for x in pairs if x["gt"][m] is not None]
        out[m] = {"gt_mean": float(np.mean(gt)) if gt else None}
        for g in cands:
            c = [x["candidates"][g][m] for x in pairs if x["candidates"][g][m] is not None and x["gt"][m] is not None]
            out[m][g] = float(np.mean(c) / np.mean(gt)) if c and gt else None
    return out


def _regimes(rows, pairs, a, b):
    out = {}
    bins = [("bend", k, lambda r, k=k: r["bend_regime"] == k) for k in BEND_REGIMES] + [("f", k, lambda r, k=k: r["f_regime"] == k) for k in F_BINS]
    for kind, k, fn in bins:
        g, p = [r for r in rows if fn(r)], [x for x in pairs if fn(x)]
        o = {"rows": len(g), "pairs": len(p), "sequences": len({r["sequence_id"] for r in g})}
        if g:
            o["distal"] = {c: float(np.mean([r["candidates"][c]["wrist_error"] for r in g])) for c in (a, b)}
            o["middle"] = {c: float(np.mean([r["candidates"][c]["elbow_error"] for r in g])) for c in (a, b)}
            dc = _cmp(g, a, b, "wrist_error", False)
            o["distal_contrast"] = {kk: dc.get(kk) for kk in ("mean", "sequences_negative", "sequences")}
        if p:
            o["proximal_residual"] = {c: float(np.mean([x["candidates"][c]["upper_direction_residual_degrees"] for x in p])) for c in (a, b)}
        out[f"{kind}:{k}"] = o
    return out


def evaluate(rows, pairs, chain):
    cands = CANDIDATES[chain]
    a, b = PRIMARY[chain]
    oop = [r for r in rows if r["f_regime"] == "out_of_plane [0.5,1]"]
    per_seq = {}
    for sid in sorted({r["sequence_id"] for r in rows}):
        g = [r for r in rows if r["sequence_id"] == sid]
        per_seq[sid] = {"rows": len(g), "distal": {c: float(np.mean([r["candidates"][c]["wrist_error"] for r in g])) for c in cands}}
    return {"rows": len(rows), "pairs": len(pairs), "sequences": len({r["sequence_id"] for r in rows}),
            "static_table": _table(rows, cands, STATIC), "temporal_table": _table(pairs, cands, TEMPORAL),
            "static_contrasts": {f"{x}-{y}": {m: _cmp(rows, x, y, m) for m in STATIC} for x, y in CONTRASTS[chain]},
            "out_of_plane_distal": {f"{x}-{y}": _cmp(oop, x, y, "wrist_error") for x, y in CONTRASTS[chain]},
            "temporal_contrasts": {f"{x}-{y}": {m: _cmp(pairs, x, y, m) for m in TEMPORAL} for x, y in CONTRASTS[chain]},
            "amplitude": _amplitude(pairs, cands), "discontinuities": _jumps(pairs, cands),
            "regimes": _regimes(rows, pairs, a, b), "per_sequence": per_seq}


def _strong(c):
    return c.get("n", 0) > 0 and c["mean"] < 0 and _excludes_zero(c) and c["sequences_negative"] >= math.ceil(2 / 3 * c["sequences"])


def _regress(c):
    return c.get("n", 0) > 0 and c["mean"] > 0 and _excludes_zero(c)


def transfer_test(ev, chain, ref_penalty_3dpw, validity):
    a, b = PRIMARY[chain]
    k = f"{a}-{b}"
    out = {}
    for pop in ("3dpw", "3dhp"):
        s = ev[pop]["static_contrasts"][k]
        d = s["wrist_error"]
        share = d.get("most_negative_sequence", {}).get("share_of_net")
        t = ev[pop]["static_table"]
        tp = ev[pop]["temporal_contrasts"][k]["upper_direction_residual_degrees"]
        if pop == "3dpw":
            limit = max(1.5 * ref_penalty_3dpw, ref_penalty_3dpw + 1.0)
            temporal = tp["mean"] <= limit
        else:
            limit, temporal = None, not _regress(tp)
        out[pop] = {"distal_mean": d["mean"], "strong": _strong(d), "regression": _regress(d),
                    "concentration_ok": share is None or share < 0.5, "middle_ok": not _regress(s["elbow_error"]),
                    "tail_ok": t[a]["wrist_error"]["p95"] <= t[b]["wrist_error"]["p95"] + 2.0,
                    "temporal_ok": temporal, "temporal_mean": tp["mean"], "temporal_limit": limit}
    p = out
    strong_any = any(p[x]["strong"] and p[x]["concentration_ok"] for x in p)
    out["transfers"] = bool(validity and strong_any and not any(p[x]["regression"] for x in ("3dpw", "3dhp"))
                            and all(p[x]["middle_ok"] and p[x]["tail_ok"] and p[x]["temporal_ok"] for x in ("3dpw", "3dhp")))
    out["provisional_note"] = ("strong evidence on one population only" if sum(p[x]["strong"] for x in ("3dpw", "3dhp")) == 1 else None)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("predictions_dir", "bank", "w83_bank", "out_dir"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise SystemExit("STOP: output directory is not empty; never overwrite experimental artifacts")
    rep = json.loads((args.predictions_dir / "prediction_report.json").read_text())
    if not rep["passed"]:
        raise SystemExit("STOP: predictions did not pass the reproduction gate")
    for pop in ("3dpw", "3dhp"):
        if _sha(args.predictions_dir / f"predictions_{pop}.npz") != rep["predictions_sha256"][pop]:
            raise SystemExit(f"STOP: {pop} predictions changed")
    validity = all(v["finite"] and v["length_sum_error"] <= 1e-9 for v in rep["validity"].values())
    fb, mx = load_bank(args.bank), ExternalBank.load(args.w83_bank)
    p3 = dict(np.load(args.predictions_dir / "predictions_3dpw.npz"))
    px = dict(np.load(args.predictions_dir / "predictions_3dhp.npz"))
    stride = {s.sequence_id: stride_for(s.sequence_id.split(":")[1]) for s in mx.samples}
    short = {s.sequence_id: SHORT_PAIR_FRAMES.get(s.sequence_id.split(":")[1]) for s in mx.samples}
    ev, secondary, dumps = {}, {}, {}
    for chain, cands in CANDIDATES.items():
        r3, q3 = collect(fb.samples, fb.indices("test"), p3, chain, cands, lambda _s: 5)
        rx, qx = collect(mx.samples, range(len(mx)), px, chain, cands, stride.__getitem__)
        _, qs = collect(mx.samples, range(len(mx)), px, chain, cands, short.__getitem__)
        ev[chain] = {"3dpw": evaluate(r3, q3, chain), "3dhp": evaluate(rx, qx, chain)}
        a, b = PRIMARY[chain]
        secondary[chain] = {"pairs": len(qs), "temporal_table": _table(qs, cands, TEMPORAL),
                            "proximal_residual_contrast": _cmp(qs, a, b, "upper_direction_residual_degrees"), "amplitude": _amplitude(qs, cands)}
        dumps[chain] = {"rows_3dhp": rx, "pairs_3dhp": qx, "rows_3dpw": r3}
    ref = ev["left_arm"]["3dpw"]["temporal_contrasts"]["j1x-g0x"]["upper_direction_residual_degrees"]["mean"]
    tests = {c: transfer_test(ev[c], c, ref, validity) for c in CANDIDATES}
    ok = [c for c in ("right_arm", "right_leg") if tests[c]["transfers"]]
    case = "A" if len(ok) == 2 else ("B" if len(ok) == 1 else "C")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for chain, d in dumps.items():
        for k, v in d.items():
            (args.out_dir / f"{chain}_{k}.json").write_text(json.dumps(v))
    report = {"schema": "animcv_w86_limb_eval_v1", "protocol": PROTOCOL, "decision_contract": DECISION, "labels": LABELS,
              "bootstrap_contract": BOOTSTRAP, "prediction_report_sha256": _sha(args.predictions_dir / "prediction_report.json"),
              "validity": validity, "evaluation": ev, "outdoor_short_interval_secondary": secondary,
              "reference_temporal_penalty_3dpw": ref, "transfer_tests": tests, "decision": {"transferring_chains": ok, "mechanical_case": case}}
    (args.out_dir / "report.json").write_text(json.dumps(report, indent=1, default=str))
    print(json.dumps({"transfer_tests": tests, "decision": report["decision"]}, indent=1, default=str))


if __name__ == "__main__":
    main()
