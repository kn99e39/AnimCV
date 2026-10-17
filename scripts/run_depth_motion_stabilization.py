#!/usr/bin/env python3
"""EXPERIMENT (Worklog 89): optional depth-tilt stabilizer on frozen RGB connected-limb predictions.

Comparisons on identical frozen predictions (Worklog 87 ``predictions_*.npz``; no model is run):
  A  raw RGB      frozen G1 (J1X for the left arm)
  B  geometry     frozen matched G0 (G0X for the left arm)
  C  stabilized   A + ``framepose.depth_motion_stabilizer`` (offline Kalman/RTS of the segment depth tilts)
  C_causal        the same candidate in causal mode (forward filter only), reported for the latency trade-off;
                  it does not enter the CASE rule.
Historical C1 (Worklog 48/49 gated full-body H0 interpolation) is not comparable: sparse outlier repair of absolute
full-body joints, no chain contract.  It is not rerun.

Phases.  ``validation``: calibrate the stabilizer once per chain on 3DPW VALIDATION (0.1 s rows) with the declared
moment estimator, write and hash ``config.json`` BEFORE any stabilized output is evaluated, then evaluate 3DPW
VALIDATION at 0.1 s and 0.2 s pairs plus the time-step check (stabilizer fed every second row, i.e. 0.2 s input).
``replay``: load the frozen config (hash checked; no recalibration), evaluate 3DPW TEST (1/6 s), MPI-INF-3DHP TEST
0.4 s pairs and TS5-TS6 0.16 s pairs separately, and apply the predeclared ``DECISION`` rule.

GT is used for calibration (validation phase only) and for evaluation; the stabilizer itself receives timestamps,
track keys, predicted chains and detector validity only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_visual_depth_motion_attribution import W87_PREDICTIONS, _brief  # noqa: E402

from framepose.bank import load_bank  # noqa: E402
from framepose.bedlam_supervision import ExternalBank  # noqa: E402
from framepose.connected_limb_transfer import chain_indices, chain_targets  # noqa: E402
from framepose.depth_motion_stabilizer import StabilizerConfig, calibrate, decompose, depth_tilt, stabilize  # noqa: E402
from framepose.mpi3dhp_test_transfer import FPS, SHORT_PAIR_FRAMES, stride_for  # noqa: E402
from framepose.visual_depth_motion_attribution import FALSE_MOTION, gt_motion_stratum, motion_terms  # noqa: E402

CHAINS = ("left_arm", "right_arm", "right_leg")
ARMS = ("left_arm", "right_arm")
FPS_3DPW = 30.0
# Worklog 88 depth-motion residuals (R0 G0, R1 G1; %L) that the raw baselines must reproduce.
W88_RESIDUALS = {
    "3dpw_test": {"left_arm": (7.717340801504488, 10.849913771281752), "right_arm": (7.8327421564141195, 10.888142998462358),
                  "right_leg": (6.616008009801285, 8.023732390149533)},
    "3dpw_validation_0.1s": {"left_arm": (7.1005634859760915, 9.589488798064904), "right_arm": (6.91479953969378, 9.601638779131992),
                             "right_leg": (5.129526532518284, 5.950753254832058)},
    "3dpw_validation_0.2s": {"left_arm": (11.642240682185019, 12.158010396145139), "right_arm": (10.811342131879343, 11.593561314876414),
                             "right_leg": (7.358912241716886, 7.369721299858971)}}
SEVERE_STATIC = 10.0  # %L: a frame where C's distal error exceeds A's by more than this is a severe regression
STALE = 2.0  # %L: a high-GT-motion pair whose predicted depth displacement stays below this is a false hold
DECISION = {
    "suppression": "low-GT-motion false-motion rate reduced by >= 30% relative (C vs A) AND the low-stratum residual "
                   "C - A has a sequence-cluster 95% interval below 0 AND improves in >= 75% of sequences",
    "retention": "high-stratum tracking-slope retention (slope_C - slope_B) / (slope_A - slope_B) >= 0.8 AND the "
                 "high-stratum residual C - A <= +0.5 %L",
    "static": "mean distal position error C - A <= +0.5 %L, p95 distal position error C - A <= +1.0 %L, distal "
              "camera-depth advantage retention (err_B - err_C) / (err_B - err_A) >= 0.8 where A beats B",
    "lag": "best integer row shift of C against GT distal depth = 0",
    "tails": "severe static regressions (C distal error > A + 10 %L) <= 1% of frames AND residual p99 C <= A",
    "geometry": "finite, unit u and v, rho and image-plane headings unchanged, P = 0 (else CASE D / STOP)",
    "A_practical": "all criteria for both arms on 3DPW VALIDATION 0.1 s and 3DPW TEST, right leg not regressing "
                   "(retention >= 0.5 and mean static change <= +0.5 %L), no 3DHP 0.4 s arm residual worsening > 1 %L",
    "B_limited": "otherwise, suppression for >= 1 arm on both VALIDATION 0.1 s and TEST",
    "C_insufficient": "otherwise",
    "D_blocked": "parity, timestamps, geometry or config identity fail",
    "validation_stop": "STOP before replay if slope retention < 0.5 for both arms on VALIDATION 0.1 s (systematic loss of "
                       "genuine depth motion)",
    "observability_counterexample": "reported: high GT depth motion (|dG| >= 15 %L) with detector 2D distal displacement "
                                    "below the population median"}


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


class Population:
    def __init__(self, name, bank, rows, times, keys, pred, step_of, step_s):
        self.name, self.bank, self.rows, self.times, self.keys, self.pred = name, bank, np.asarray(rows), times, keys, pred
        self.step_of, self.step_s = step_of, step_s


def pairs_for(pop, ok, step_of):
    by = defaultdict(dict)
    for p in pop.rows:
        if ok[p]:
            s = pop.bank.samples[p]
            by[s.sequence_id][int(s.frame_index)] = p
    out = []
    for sid, frames in by.items():
        step = step_of(sid)
        if step is None:
            continue
        for f, p in sorted(frames.items()):
            q = frames.get(f + step)
            if q is not None:
                out.append((sid, f, p, q))
    return out


def run_stabilizer(pop, chain, cfg, rows=None):
    """Stabilized G1 chains for every bank row of the population (others NaN)."""
    rows = pop.rows if rows is None else rows
    e = pop.pred[f"{chain}_g1_elbow"][rows]
    w = pop.pred[f"{chain}_g1_wrist"][rows]
    obs = pop.bank.arrays["input_valid"][rows][:, chain_indices(chain)].all(1)
    out_e = np.full(pop.pred[f"{chain}_g1_elbow"].shape, np.nan)
    out_w = out_e.copy()
    t0 = time.perf_counter()
    se, sw = stabilize(pop.keys[rows], pop.times[rows], e, w, obs, cfg)
    elapsed = time.perf_counter() - t0
    out_e[rows], out_w[rows] = se, sw
    return out_e, out_w, {"frames": int(len(rows)), "seconds": elapsed, "us_per_frame": 1e6 * elapsed / max(len(rows), 1)}


def geometry_check(e_raw, w_raw, e, w, rows):
    u0, v0, r0 = decompose(e_raw[rows], w_raw[rows])
    u1, v1, r1 = decompose(e[rows], w[rows])
    def head(d):
        n = np.hypot(d[:, 0], d[:, 2])
        return np.stack([d[:, 0], d[:, 2]], 1) / np.maximum(n, 1e-12)[:, None], n > 1e-6
    hu0, ok_u = head(u0)
    hv0, ok_v = head(v0)
    hu1, _ = head(u1)
    hv1, _ = head(v1)
    seg = np.linalg.norm(w[rows] - e[rows], axis=1)
    return {"finite": bool(np.isfinite(e[rows]).all() and np.isfinite(w[rows]).all()),
            "max_rho_change": float(np.abs(r1 - r0).max()),
            "max_unit_deviation": float(max(np.abs(np.linalg.norm(e[rows], axis=1) - r0).max(), np.abs(seg - (1 - r0)).max())),
            "max_heading_change": float(max(np.abs(hu1 - hu0)[ok_u].max(), np.abs(hv1 - hv0)[ok_v].max()))}


def _angle(a, b):
    return np.degrees(np.arccos(np.clip((a * b).sum(-1), -1, 1)))


def static_metrics(e, w, et, wt, rows):
    u, v, rho = decompose(e[rows], w[rows])
    ut, vt, rhot = decompose(et[rows], wt[rows])
    d_err = 100 * np.linalg.norm(w[rows] - wt[rows], axis=1)
    return {"middle": 100 * np.linalg.norm(e[rows] - et[rows], axis=1), "distal": d_err,
            "distal_depth": 100 * np.abs(w[rows, 1] - wt[rows, 1]), "u_deg": _angle(u, ut), "v_deg": _angle(v, vt),
            "bend_deg": np.abs(_angle(u, v) - _angle(ut, vt)),
            "reach": 100 * np.abs(np.linalg.norm(w[rows], axis=1) - np.linalg.norm(wt[rows], axis=1))}


def best_shift(pop, rows, w, wt, max_shift=3):
    """Row shift s minimizing mean |Y_pred(t + s) - Y_gt(t)| within tracks (positive = prediction lags)."""
    order = np.lexsort((pop.times[rows], pop.keys[rows]))
    r = rows[order]
    keys = pop.keys[r]
    cost = {}
    for s in range(-max_shift, max_shift + 1):
        a = np.arange(max(0, -s), len(r) - max(0, s))
        b = a + s
        same = keys[a] == keys[b]
        cost[s] = float(np.mean(np.abs(w[r[b[same]], 1] - wt[r[a[same]], 1])))
    return min(cost, key=cost.get), cost


def evaluate(pop, chain, models, step_of, label):
    """models: name -> (elbow array, wrist array) over all bank rows."""
    e_t, w_t, ok = chain_targets(pop.bank.arrays["target_3d"], pop.bank.arrays["target_valid"], chain)
    pairs = pairs_for(pop, ok, step_of)
    if not pairs:
        return {"pairs": 0}
    a_idx, b_idx = np.array([p[2] for p in pairs]), np.array([p[3] for p in pairs])
    seq = np.array([p[0] for p in pairs])
    strata = gt_motion_stratum(w_t[a_idx], w_t[b_idx])
    idx = chain_indices(chain)
    size = np.array([pop.bank.samples[p].image_size for p in a_idx], float)
    det = pop.bank.arrays["input_2d"]
    det_disp = np.linalg.norm((det[b_idx][:, idx[2], :2] - det[a_idx][:, idx[2], :2]) * size, axis=1)
    m = {k: motion_terms(v[1][a_idx], v[1][b_idx], w_t[a_idx], w_t[b_idx]) for k, v in models.items()}
    dG = m["A"]["gt_disp"]
    out = {"label": label, "pairs": len(pairs), "sequences": int(len(set(seq))), "pair_seconds": sorted({round(float(pop.times[q] - pop.times[p]), 4) for _, _, p, q in pairs}),
           "strata": {s: int((strata == s).sum()) for s in ("low", "mid", "high")}, "models": {}}
    for k, mk in m.items():
        r = mk["abs_residual"]
        mm = motion_terms(models[k][0][a_idx], models[k][0][b_idx], e_t[a_idx], e_t[b_idx])
        o = {"residual": float(r.mean()), "residual_p95": float(np.percentile(r, 95)), "residual_p99": float(np.percentile(r, 99)),
             "middle_residual": float(mm["abs_residual"].mean()), "amplitude": float(mk["pred_amp"].mean()), "gt_amplitude": float(mk["gt_amp"].mean())}
        lo, hi = strata == "low", strata == "high"
        o["low"] = {"residual": float(r[lo].mean()), "false_motion_rate": float((mk["pred_amp"][lo] > FALSE_MOTION).mean()),
                    "residual_p95": float(np.percentile(r[lo], 95)), "residual_p99": float(np.percentile(r[lo], 99)),
                    "over_response": float(mk["over_response"][lo].mean()), "amplitude": float(mk["pred_amp"][lo].mean())}
        dp = mk["pred_disp"]
        o["high"] = {"residual": float(r[hi].mean()), "slope": float(np.polyfit(dG[hi], dp[hi], 1)[0]),
                     "amplitude_ratio": float(mk["pred_amp"][hi].mean() / max(mk["gt_amp"][hi].mean(), 1e-9)),
                     "signed_error_along_gt": float(np.mean((dp[hi] - dG[hi]) * np.sign(dG[hi]))),
                     "sign_agreement": float(np.mean(np.sign(dp[hi]) == np.sign(dG[hi]))),
                     "missed_rate": float(np.mean((np.abs(dp[hi]) < 0.5 * np.abs(dG[hi])) | (np.sign(dp[hi]) != np.sign(dG[hi])))),
                     "stale_hold_rate": float(np.mean(np.abs(dp[hi]) < STALE))}
        # observability counterexample: large GT depth motion with small detector 2D movement
        cx = (np.abs(dG) >= 15.0) & (det_disp < np.median(det_disp))
        o["depth_motion_small_2d"] = {"pairs": int(cx.sum()),
                                      "slope": float(np.polyfit(dG[cx], dp[cx], 1)[0]) if cx.sum() > 10 else None,
                                      "amplitude_ratio": float(mk["pred_amp"][cx].mean() / max(mk["gt_amp"][cx].mean(), 1e-9)) if cx.any() else None,
                                      "residual": float(r[cx].mean()) if cx.any() else None}
        out["models"][k] = o
    # oscillation on consecutive triples (a, b) + (b, c) with GT-valid rows
    nxt = {(p[0], p[1]): i for i, p in enumerate(pairs)}
    tri = [(i, nxt[(p[0], p[1] + step_of(p[0]))]) for i, p in enumerate(pairs) if (p[0], p[1] + step_of(p[0])) in nxt]
    if tri:
        i1, i2 = np.array([t[0] for t in tri]), np.array([t[1] for t in tri])
        low2 = (strata[i1] == "low") & (strata[i2] == "low")
        for k, mk in m.items():
            dp = mk["pred_disp"]
            jerk = np.abs(dp[i2] - dp[i1])
            alt = (np.abs(dp[i1]) >= FALSE_MOTION) & (np.abs(dp[i2]) >= FALSE_MOTION) & (np.sign(dp[i1]) != np.sign(dp[i2]))
            out["models"][k]["oscillation"] = {"triples": len(tri), "jerk": float(jerk.mean()), "gt_jerk": float(np.abs(dG[i2] - dG[i1]).mean()),
                                               "low_triples": int(low2.sum()), "low_alternation_rate": float(alt[low2].mean()) if low2.any() else None}
    # paired contrasts (sequence-cluster bootstrap)
    rA = m["A"]["abs_residual"]
    con = {}
    for k in m:
        if k == "A":
            continue
        d = m[k]["abs_residual"] - rA
        con[k] = {"all": _brief(seq, d), "low": _brief(seq[strata == "low"], d[strata == "low"]),
                  "high": _brief(seq[strata == "high"], d[strata == "high"]),
                  "low_false_motion": _brief(seq[strata == "low"], (m[k]["pred_amp"] > FALSE_MOTION).astype(float)[strata == "low"]
                                             - (m["A"]["pred_amp"] > FALSE_MOTION).astype(float)[strata == "low"])}
        lo = strata == "low"
        per = defaultdict(list)
        for s_, v in zip(seq[lo], d[lo]):
            per[s_].append(v)
        con[k]["low_sequences_improved"] = [int(sum(np.mean(v) < 0 for v in per.values())), len(per)]
    out["contrast_vs_A"] = con
    # static per-frame accuracy on GT-valid rows
    rows = np.array([p for p in pop.rows if ok[p] and np.isfinite(models["C"][1][p]).all()])
    st = {k: static_metrics(v[0], v[1], e_t, w_t, rows) for k, v in models.items()}
    fseq = pop.keys[rows]
    out["static"] = {"frames": int(len(rows)),
                     **{k: {q: float(v.mean()) for q, v in s.items()} | {"distal_p95": float(np.percentile(s["distal"], 95))} for k, s in st.items()}}
    out["static_contrast_vs_A"] = {k: {"distal": _brief(fseq, st[k]["distal"] - st["A"]["distal"]),
                                       "distal_depth": _brief(fseq, st[k]["distal_depth"] - st["A"]["distal_depth"]),
                                       "severe_regression_rate": float(np.mean(st[k]["distal"] - st["A"]["distal"] > SEVERE_STATIC))}
                                   for k in models if k != "A"}
    out["lag_rows"] = {k: best_shift(pop, rows, v[1], w_t)[0] for k, v in models.items()}
    out["worst_sequences_C_minus_A_low"] = sorted(
        ({"sequence_id": s_, "mean": float(np.mean(v)), "pairs": len(v)} for s_, v in
         ((s_, [x for x, ss in zip(m["C"]["abs_residual"] - rA, seq) if ss == s_]) for s_ in sorted(set(seq)))), key=lambda z: -z["mean"])[:5]
    out["worst_regression_pairs"] = [{"sequence_id": str(seq[i]), "frame": int(pairs[i][1]), "C_minus_A": float((m["C"]["abs_residual"] - rA)[i]),
                                      "stratum": str(strata[i])} for i in np.argsort(-(m["C"]["abs_residual"] - rA))[:10]]
    return out


def criteria(r):
    """Predeclared per-chain criteria from one population's evaluation."""
    A, B, C = r["models"]["A"], r["models"]["B"], r["models"]["C"]
    ca = r["contrast_vs_A"]["C"]
    fm_rel = (A["low"]["false_motion_rate"] - C["low"]["false_motion_rate"]) / max(A["low"]["false_motion_rate"], 1e-9)
    seq_ok = ca["low_sequences_improved"][0] >= 0.75 * ca["low_sequences_improved"][1]
    low_ci = ca["low"].get("ci95")
    suppression = fm_rel >= 0.30 and low_ci is not None and low_ci[1] < 0 and seq_ok
    denom = A["high"]["slope"] - B["high"]["slope"]
    retention_ratio = (C["high"]["slope"] - B["high"]["slope"]) / denom if abs(denom) > 1e-9 else None
    retention = retention_ratio is not None and retention_ratio >= 0.8 and ca["high"]["mean"] <= 0.5
    s = r["static"]
    dadv = s["B"]["distal_depth"] - s["A"]["distal_depth"]
    depth_ret = (s["B"]["distal_depth"] - s["C"]["distal_depth"]) / dadv if dadv > 0 else None
    static = (s["C"]["distal"] - s["A"]["distal"] <= 0.5 and s["C"]["distal_p95"] - s["A"]["distal_p95"] <= 1.0
              and (depth_ret is None or depth_ret >= 0.8))
    lag = r["lag_rows"]["C"] == 0
    tails = r["static_contrast_vs_A"]["C"]["severe_regression_rate"] <= 0.01 and C["residual_p99"] <= A["residual_p99"]
    return {"false_motion_reduction": fm_rel, "suppression": bool(suppression), "slope_retention": retention_ratio,
            "retention": bool(retention), "depth_advantage_retention": depth_ret, "static": bool(static), "lag": bool(lag),
            "tails": bool(tails), "all": bool(suppression and retention and static and lag and tails)}


def build_populations(args):
    fb, mx = load_bank(args.bank), ExternalBank.load(args.w83_dir / "bank" / "mpi3dhp_test")
    for pop, h in W87_PREDICTIONS.items():
        if _sha(args.w87_dir / "predictions" / f"predictions_{pop}.npz") != h:
            raise SystemExit(f"STOP: Worklog 87 {pop} predictions changed")
    p3 = dict(np.load(args.w87_dir / "predictions" / "predictions_3dpw.npz"))
    px = dict(np.load(args.w87_dir / "predictions" / "predictions_3dhp.npz"))
    t3 = np.array([s.frame_index / FPS_3DPW for s in fb.samples], np.float64)
    k3 = np.array([s.sequence_id for s in fb.samples])
    tx = np.array([s.frame_index / FPS[s.sequence_id.split(":")[1]] for s in mx.samples], np.float64)
    kx = np.array([s.sequence_id for s in mx.samples])
    val = fb.indices("validation")
    return {"3dpw_validation_0.1s": Population("3dpw_validation_0.1s", fb, val, t3, k3, p3, lambda s: 3, 0.1),
            "3dpw_validation_0.2s": Population("3dpw_validation_0.2s", fb, val, t3, k3, p3, lambda s: 6, 0.2),
            "3dpw_test": Population("3dpw_test", fb, fb.indices("test"), t3, k3, p3, lambda s: 5, 1 / 6),
            "3dhp_0.4s": Population("3dhp_0.4s", mx, np.arange(len(mx)), tx, kx, px, lambda s: stride_for(s.split(":")[1]), 0.4),
            "3dhp_0.16s_outdoor": Population("3dhp_0.16s_outdoor", mx, np.arange(len(mx)), tx, kx, px,
                                             lambda s: SHORT_PAIR_FRAMES.get(s.split(":")[1]), 0.16)}


def parity(pop, chain):
    e_t, w_t, ok = chain_targets(pop.bank.arrays["target_3d"], pop.bank.arrays["target_valid"], chain)
    pairs = pairs_for(pop, ok, pop.step_of)
    a, b = np.array([p[2] for p in pairs]), np.array([p[3] for p in pairs])
    r = [float(motion_terms(pop.pred[f"{chain}_{g}_wrist"][a], pop.pred[f"{chain}_{g}_wrist"][b], w_t[a], w_t[b])["abs_residual"].mean())
         for g in ("g0", "g1")]
    want = W88_RESIDUALS[pop.name][chain]
    diff = max(abs(r[0] - want[0]), abs(r[1] - want[1]))
    if diff > 1e-9:
        raise SystemExit(f"STOP: {pop.name}/{chain}: Worklog 88 residuals not reproduced ({r} vs {want})")
    return diff


def models_for(pop, chain, cfgs):
    e_c, w_c, timing = run_stabilizer(pop, chain, cfgs["offline"])
    e_k, w_k, timing_k = run_stabilizer(pop, chain, cfgs["causal"])
    p = pop.pred
    return ({"A": (p[f"{chain}_g1_elbow"], p[f"{chain}_g1_wrist"]), "B": (p[f"{chain}_g0_elbow"], p[f"{chain}_g0_wrist"]),
             "C": (e_c, w_c), "C_causal": (e_k, w_k)}, {"offline": timing, "causal": timing_k})


def check_geometry(pop, chain, mods):
    geo = {}
    for k in ("C", "C_causal"):
        g = geometry_check(mods["A"][0], mods["A"][1], mods[k][0], mods[k][1], pop.rows)
        if not g["finite"] or g["max_rho_change"] > 1e-9 or g["max_unit_deviation"] > 1e-9 or g["max_heading_change"] > 1e-9:
            raise SystemExit(f"STOP: {pop.name}/{chain}/{k}: stabilized geometry invalid {g}")
        geo[k] = g
    off = stabilize(pop.keys[pop.rows], pop.times[pop.rows], mods["A"][0][pop.rows], mods["A"][1][pop.rows], np.ones(len(pop.rows), bool), None)
    if not (np.array_equal(off[0], mods["A"][0][pop.rows].astype(np.float64)) and np.array_equal(off[1], mods["A"][1][pop.rows].astype(np.float64))):
        raise SystemExit("STOP: disabled stabilizer is not the identity")
    return geo


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("validation", "replay"), required=True)
    for name in ("bank", "w83_dir", "w87_dir", "out_dir"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    args = parser.parse_args()
    out = args.out_dir / args.phase
    if out.exists() and any(out.iterdir()):
        raise SystemExit("STOP: output directory is not empty; never overwrite experimental artifacts")
    out.mkdir(parents=True, exist_ok=True)
    pops = build_populations(args)
    cfg_path = args.out_dir / "config.json"
    report = {"schema": "worklog89_depth_motion_stabilization", "phase": args.phase, "decision_contract": DECISION, "parity": {}, "results": {}}
    if args.phase == "validation":
        pv = pops["3dpw_validation_0.1s"]
        cfg = {}
        for chain in CHAINS:
            e_t, w_t, ok = chain_targets(pv.bank.arrays["target_3d"], pv.bank.arrays["target_valid"], chain)
            rows = np.array([p for p in pv.rows if ok[p]])
            u, v, _ = decompose(pv.pred[f"{chain}_g1_elbow"][rows], pv.pred[f"{chain}_g1_wrist"][rows])
            ut, vt, _ = decompose(e_t[rows], w_t[rows])
            cal = calibrate(np.stack([depth_tilt(u), depth_tilt(v)], 1), np.stack([depth_tilt(ut), depth_tilt(vt)], 1),
                            pv.keys[rows], pv.times[rows], 0.1)
            cfg[chain] = {"calibration": cal, "offline": StabilizerConfig(cal["sigma_meas"], cal["q_accel"]).as_dict(),
                          "causal": StabilizerConfig(cal["sigma_meas"], cal["q_accel"], mode="causal").as_dict()}
        text = json.dumps({"calibration_population": "3dpw_validation rows (0.1 s), G1 segment depth tilts", "chains": cfg}, indent=1)
        # the estimator is deterministic: a rerun after an implementation defect must regenerate the identical config
        if cfg_path.exists() and cfg_path.read_text() != text:
            raise SystemExit("STOP: a different config is already frozen")
        cfg_path.write_text(text)
        names = ("3dpw_validation_0.1s", "3dpw_validation_0.2s")
    else:
        if not cfg_path.exists():
            raise SystemExit("STOP: no frozen config; run the validation phase first")
        frozen = json.loads((args.out_dir / "validation" / "report.json").read_text())["config_sha256"]
        if _sha(cfg_path) != frozen:
            raise SystemExit("STOP: config changed after validation")
        names = ("3dpw_test", "3dhp_0.4s", "3dhp_0.16s_outdoor")
    report["config_sha256"] = _sha(cfg_path)
    cfg = {c: {m: StabilizerConfig(**v[m]) for m in ("offline", "causal")} for c, v in json.loads(cfg_path.read_text())["chains"].items()}
    report["config"] = json.loads(cfg_path.read_text())
    for chain in CHAINS:
        report["results"][chain] = {}
        for name in names:
            pop = pops[name]
            if name in W88_RESIDUALS:
                report["parity"][f"{name}:{chain}"] = parity(pop, chain)
            mods, timing = models_for(pop, chain, cfg[chain])
            geo = check_geometry(pop, chain, mods)
            r = evaluate(pop, chain, mods, pop.step_of, name)
            r["timing"], r["geometry"] = timing, geo
            if r["pairs"]:
                r["criteria"] = criteria(r)
            report["results"][chain][name] = r
            print(json.dumps({"chain": chain, "pop": name, "pairs": r["pairs"], "criteria": r.get("criteria")}), flush=True)
        if args.phase == "validation":
            # time-step check: the same stabilizer fed every second VALIDATION row (0.2 s input cadence)
            pop = pops["3dpw_validation_0.2s"]
            by = defaultdict(list)
            for p in pop.rows:
                by[pop.keys[p]].append(p)
            sub = np.array(sorted(p for rows in by.values() for p in rows if (pop.bank.samples[p].frame_index // 3) % 2 == 0))
            e_s, w_s, _ = run_stabilizer(pop, chain, cfg[chain]["offline"], rows=sub)
            sub_pop = Population("3dpw_validation_0.2s_input", pop.bank, sub, pop.times, pop.keys, pop.pred, pop.step_of, 0.2)
            mods = {"A": (pop.pred[f"{chain}_g1_elbow"], pop.pred[f"{chain}_g1_wrist"]), "B": (pop.pred[f"{chain}_g0_elbow"], pop.pred[f"{chain}_g0_wrist"]),
                    "C": (e_s, w_s), "C_full_input": run_stabilizer(pop, chain, cfg[chain]["offline"])[:2]}
            report["results"][chain]["3dpw_validation_0.2s_input"] = evaluate(sub_pop, chain, mods, pop.step_of, "0.2 s input cadence")
    if args.phase == "validation":
        stop = all((report["results"][c]["3dpw_validation_0.1s"]["criteria"]["slope_retention"] or 0) < 0.5 for c in ARMS)
        report["validation_stop"] = bool(stop)
    else:
        val = json.loads((args.out_dir / "validation" / "report.json").read_text())
        if val.get("validation_stop"):
            raise SystemExit("STOP: validation showed systematic loss of genuine depth motion; replay not allowed")
        report["decision"] = classify(val["results"], report["results"])
    (out / "report.json").write_text(json.dumps(report, indent=1, default=str))
    print(json.dumps(report.get("decision", {"validation_stop": report.get("validation_stop")}), indent=1))


def classify(val, rep):
    v = {c: val[c]["3dpw_validation_0.1s"]["criteria"] for c in CHAINS}
    t = {c: rep[c]["3dpw_test"]["criteria"] for c in CHAINS}
    leg_ok = all((x["right_leg"]["slope_retention"] or 0) >= 0.5 for x in (v, t)) and all(
        r["right_leg"][p]["static"]["C"]["distal"] - r["right_leg"][p]["static"]["A"]["distal"] <= 0.5
        for r, p in ((val, "3dpw_validation_0.1s"), (rep, "3dpw_test")))
    hp_ok = all(rep[c]["3dhp_0.4s"]["contrast_vs_A"]["C"]["all"]["mean"] <= 1.0 for c in ARMS)
    arms_all = all(v[c]["all"] and t[c]["all"] for c in ARMS)
    supp = [c for c in ARMS if v[c]["suppression"] and t[c]["suppression"]]
    case = "A" if arms_all and leg_ok and hp_ok else ("B" if supp else "C")
    return {"validation_criteria": v, "test_criteria": t, "right_leg_not_regressing": bool(leg_ok),
            "3dhp_arm_residual_ok": bool(hp_ok), "suppression_arms": supp, "engineering_case": case}


if __name__ == "__main__":
    main()
