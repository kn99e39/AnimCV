#!/usr/bin/env python3
"""DIAGNOSTIC (Worklog 87): near-plane false-depth attribution for frozen connected-limb models (no training).

Frozen pairs per chain (G0 = 2D only, G1 = 2D + frozen Qwen): left arm G0X / J1X, right arm G0 / G1, right leg
G0 / G1.  Populations: 3DPW TEST (primary, 37 sequences), 3DPW VALIDATION and BEDLAM VALIDATION (selection-free
corroboration), MPI-INF-3DHP TEST (6 clusters, reported).  Everything GT-derived (regimes, factor hybrids,
detector-vs-GT discrepancy) is evaluation-only.  Squared errors are in (%L)^2 so that the per-axis
decomposition |e|^2 = eX^2 + eY^2 + eZ^2 is exact; Y is camera depth.

``DIAGNOSIS`` was committed before this runner produced any output.
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

from framepose.bank import load_bank  # noqa: E402
from framepose.bedlam_supervision import ExternalBank  # noqa: E402
from framepose.connected_limb_transfer import chain_indices, chain_targets  # noqa: E402
from framepose.mpi3dhp_test_transfer import motion_metrics, stride_for  # noqa: E402
from framepose.near_plane_attribution import (  # noqa: E402
    REGIMES, STABLE, axis_squared_errors, classify, depth_components, distal_decomposition, factorial_distal_errors,
    factorial_effects, factors,
)

CHAINS = ("left_arm", "right_arm", "right_leg")
SCALE = 100.0  # normalized -> %L
DIAGNOSIS = {
    "primary_population": "3DPW TEST; corroboration = 3DPW VALIDATION (same direction required for A); 3DHP / BEDLAM VALIDATION reported",
    "penalty": "regime-wise G1 - G0 distal squared error ((%L)^2): mean > 0 AND sequence-cluster 95% interval excludes 0",
    "depth_share": "mean G1 - G0 depth-axis squared error / mean G1 - G0 total squared error (distal joint)",
    "overdepth": "G1 - G0 of (|pred depth| - |GT depth|) at the distal joint: mean > 0 AND interval excludes 0",
    "observation_dominated": "inside regime A: the higher-discrepancy half (detector vs GT 2D, max over P/M/D) has a penalty "
                             "but the lower half does not (its interval includes 0 or mean <= 0)",
    "D_insufficient": "fewer than 2 chains show a penalty in A_true_near or B_cancellation on 3DPW TEST",
    "C_observation": "otherwise, if >= 2 chains are observation_dominated",
    "A_depth_overcorrection": "otherwise, if >= 2 chains INCLUDING the left arm have penalty(A) AND depth_share(A) >= 0.6 AND "
                              "overdepth(A) on 3DPW TEST, and the 3DPW VALIDATION G1 - G0 regime-A mean is > 0 for those chains",
    "B_coupled": "otherwise (a penalty exists but is not depth-owned / not consistent). A single coherent geometry factor is "
                 "'identified' only if, for >= 2 chains, one factor's single-switch effect is >= 60% of the regime penalty and the "
                 "interaction residual is < 25% of it",
    "intervention": "only for A, or B with an identified factor; declared in writing before any training"}


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _c(seq, values, bootstrap=True):
    items = [{"sequence_id": s, "v": float(v)} for s, v in zip(seq, values) if np.isfinite(v)]
    return _contrast(items, lambda it: it["v"], bootstrap) if items else {"n": 0}


def _brief(c):
    if c.get("n", 0) == 0:
        return {"n": 0}
    b = c.get("bootstrap", {}).get("sequence_cluster", {}).get("frame_weighted_ci95")
    return {"n": c["n"], "mean": c["mean"], "ci95": b, "sequences": c["sequences"], "sequences_negative": c["sequences_negative"],
            "sequences_positive": c["sequences_positive"], "excludes_zero": _excludes_zero(c)}


def angle(a, b):
    a = a / np.maximum(np.linalg.norm(a, axis=-1, keepdims=True), 1e-12)
    b = b / np.maximum(np.linalg.norm(b, axis=-1, keepdims=True), 1e-12)
    return np.degrees(np.arctan2(np.linalg.norm(np.cross(a, b), axis=-1), (a * b).sum(-1)))


def chain_rows(pred, chain, positions, samples, disc, conf_inp):
    ok = pred[f"{chain}_target_valid"][positions]
    pos = np.asarray(positions)[ok]
    ge, gw = pred[f"{chain}_target_elbow"][pos], pred[f"{chain}_target_wrist"][pos]
    d = {"pos": pos, "seq": np.array([samples[p].sequence_id for p in pos]), "frame": np.array([samples[p].frame_index for p in pos]),
         "ge": ge, "gw": gw, "regime": classify(ge, gw)}
    for g in ("g0", "g1"):
        d[f"{g}_e"], d[f"{g}_w"] = pred[f"{chain}_{g}_elbow"][pos], pred[f"{chain}_{g}_wrist"][pos]
    idx = chain_indices(chain)
    d["disc"] = np.nanmax(disc[pos][:, idx], axis=1) if disc is not None else np.full(len(pos), np.nan)
    d["conf"] = conf_inp[pos][:, idx, 2].min(1)
    return d


def regime_block(d, mask):
    seq = d["seq"][mask]
    out = {"rows": int(mask.sum()), "sequences": int(len(set(seq.tolist())))}
    if not mask.any():
        return out
    ge, gw = d["ge"][mask] * SCALE, d["gw"][mask] * SCALE
    per = {}
    for g in ("g0", "g1"):
        e, w = d[f"{g}_e"][mask] * SCALE, d[f"{g}_w"][mask] * SCALE
        ax_d, ax_m = axis_squared_errors(w, gw), axis_squared_errors(e, ge)
        dec = distal_decomposition(e, w, ge, gw)
        u, v, rho = factors(e / SCALE, w / SCALE)
        gu, gv, grho = factors(ge / SCALE, gw / SCALE)
        per[g] = {"d_total": ax_d["total"], "d_depth": ax_d["depth"], "d_plane": ax_d["x"] + ax_d["z"], "d_signed_depth": ax_d["signed_depth"],
                  "m_total": ax_m["total"], "m_depth": ax_m["depth"], "m_plane": ax_m["x"] + ax_m["z"],
                  "overdepth": np.abs(w[:, 1]) - np.abs(gw[:, 1]), "abs_depth_pred": np.abs(w[:, 1]),
                  "dec_middle": dec["middle_part"], "dec_segment": dec["segment_part"], "dec_cross": dec["cross"],
                  "u_err": angle(u, gu), "v_err": angle(v, gv), "chain_err": angle(w, gw), "rho_err": np.abs(rho - grho),
                  "reach_err": np.abs(np.linalg.norm(w, axis=1) - np.linalg.norm(gw, axis=1)),
                  "bend_err": np.abs(angle(u, v) - angle(gu, gv)), "u_depth_err": np.abs(u[:, 1] - gu[:, 1]), "v_depth_err": np.abs(v[:, 1] - gv[:, 1])}
    out["means"] = {g: {k: float(np.mean(v)) for k, v in per[g].items()} for g in per}
    out["abs_depth_gt_mean"] = float(np.mean(np.abs(gw[:, 1])))
    out["p95_distal_sq"] = {g: float(np.percentile(per[g]["d_total"], 95)) for g in per}
    out["contrasts"] = {k: _brief(_c(seq, per["g1"][k] - per["g0"][k])) for k in
                        ("d_total", "d_depth", "d_plane", "m_total", "m_depth", "overdepth", "u_err", "v_err", "chain_err", "rho_err",
                         "reach_err", "bend_err", "dec_middle", "dec_segment", "dec_cross")}
    dt = out["contrasts"]["d_total"]
    out["depth_share_distal"] = (out["means"]["g1"]["d_depth"] - out["means"]["g0"]["d_depth"]) / dt["mean"] if dt.get("n") and abs(dt["mean"]) > 1e-12 else None
    mt = out["contrasts"]["m_total"]
    out["depth_share_middle"] = (out["means"]["g1"]["m_depth"] - out["means"]["g0"]["m_depth"]) / mt["mean"] if mt.get("n") and abs(mt["mean"]) > 1e-12 else None
    # Wrong depth sign at the distal joint (non-degenerate: GT chain depth component >= sin 10 deg in magnitude).
    _, _, fc, _ = depth_components(d["ge"][mask], d["gw"][mask])
    nd = np.abs(fc) >= STABLE
    if nd.any():
        out["wrong_sign_rate"] = {g: float(np.mean(np.sign(d[f"{g}_w"][mask][nd, 1]) != np.sign(d["gw"][mask][nd, 1]))) for g in ("g0", "g1")}
    # Hybrids are built from the unit chains (factors() needs |M| + |D - M| = 1), then expressed in (%L)^2.
    fac = factorial_distal_errors(d["g0_e"][mask], d["g0_w"][mask], d["g1_e"][mask], d["g1_w"][mask], d["ge"][mask], d["gw"][mask])
    out["factorial"] = factorial_effects({k: v * SCALE ** 2 for k, v in fac.items()})
    out["signed_depth_bias"] = {g: float(np.mean(per[g]["d_signed_depth"] * np.sign(gw[:, 1] + 1e-12))) for g in per}
    return out


def observation_split(d, mask):
    out = {}
    for key, name in (("disc", "discrepancy"), ("conf", "confidence")):
        x = d[key][mask]
        if not np.isfinite(x).any():
            continue
        med = float(np.nanmedian(x))
        hi = mask.copy()
        hi[mask] = x > med
        lo = mask.copy()
        lo[mask] = x <= med
        dt = lambda m: _brief(_c(d["seq"][m], (d["g1_w"][m] * SCALE - d["gw"][m] * SCALE) .__pow__(2).sum(1)  # noqa: E731
                                 - (d["g0_w"][m] * SCALE - d["gw"][m] * SCALE).__pow__(2).sum(1)))
        out[name] = {"median": med, "high_half": dt(hi), "low_half": dt(lo)}
    return out


def temporal(d, stride_of):
    by = defaultdict(dict)
    for i, (s, f) in enumerate(zip(d["seq"], d["frame"])):
        by[s][int(f)] = i
    rows = []
    for s, frames in by.items():
        step = stride_of(s)
        if step is None:
            continue
        for f, i in frames.items():
            j = frames.get(f + step)
            if j is not None:
                rows.append((s, i, j))
    if not rows:
        return {"pairs": 0}
    seq = np.array([r[0] for r in rows])
    i = np.array([r[1] for r in rows])
    j = np.array([r[2] for r in rows])
    gdy = (d["gw"][j, 1] - d["gw"][i, 1]) * SCALE
    out = {"pairs": len(rows)}
    res = {}
    for g in ("g0", "g1"):
        pdy = (d[f"{g}_w"][j, 1] - d[f"{g}_w"][i, 1]) * SCALE
        mm = [motion_metrics(d[f"{g}_e"][a], d[f"{g}_w"][a], d[f"{g}_e"][b], d[f"{g}_w"][b], d["ge"][a], d["gw"][a], d["ge"][b], d["gw"][b])
              for a, b in zip(i, j)]
        _, _, fca, _ = depth_components(d["ge"][i], d["gw"][i])
        _, _, fcb, _ = depth_components(d["ge"][j], d["gw"][j])
        stable_gt = (np.abs(fca) >= STABLE) & (np.abs(fcb) >= STABLE) & (np.sign(d["gw"][i, 1]) == np.sign(d["gw"][j, 1]))
        flip = (np.sign(d[f"{g}_w"][i, 1]) != np.sign(d[f"{g}_w"][j, 1])) & stable_gt
        res[g] = {"depth_motion_residual": np.abs(pdy - gdy), "depth_amplitude": np.abs(pdy),
                  "upper": np.array([m["upper_direction_residual_degrees"] for m in mm]),
                  "forearm": np.array([m["forearm_direction_residual_degrees"] for m in mm]),
                  "chain": np.array([m["chain_direction_residual_degrees"] for m in mm]),
                  "false_sign_flip": flip.astype(float), "overdepth_a": np.abs(d[f"{g}_w"][i, 1]) - np.abs(d["gw"][i, 1])}
        out[g] = {k: float(np.mean(v)) for k, v in res[g].items()}
        out[g]["gt_depth_amplitude"] = float(np.mean(np.abs(gdy)))
        out[g]["false_sign_flip_eligible_pairs"] = int(stable_gt.sum())
    out["contrasts"] = {k: _brief(_c(seq, res["g1"][k] - res["g0"][k])) for k in ("depth_motion_residual", "upper", "forearm", "chain", "false_sign_flip")}
    ra, rb = d["regime"][i], d["regime"][j]
    trans = np.array([f"{a}->{b}" if a != b else a for a, b in zip(ra, rb)])
    out["by_transition"] = {}
    for t, n in Counter(trans.tolist()).most_common():
        m = trans == t
        out["by_transition"][t] = {"pairs": int(n), "depth_motion_residual": {g: float(np.mean(res[g]["depth_motion_residual"][m])) for g in res},
                                   "upper": {g: float(np.mean(res[g]["upper"][m])) for g in res}}
    # Does the static false depth at the first frame share a cause with the temporal residual?
    near = np.isin(ra, ["A_true_near", "B_cancellation"])
    if near.sum() > 10:
        x = res["g1"]["overdepth_a"][near] - res["g0"]["overdepth_a"][near]
        y = res["g1"]["depth_motion_residual"][near] - res["g0"]["depth_motion_residual"][near]
        out["near_plane_static_vs_temporal_spearman"] = float(np.corrcoef(np.argsort(np.argsort(x)), np.argsort(np.argsort(y)))[0, 1])
    return out


def support(banks, chain):
    out = {}
    for name, (bank, rows) in banks.items():
        e, w, ok = chain_targets(bank.arrays["target_3d"][rows], bank.arrays["target_valid"][rows], chain)
        reg = classify(e[ok], w[ok])
        c = Counter(reg.tolist())
        out[name] = {"rows": int(ok.sum()), **{r: {"count": c.get(r, 0), "share": c.get(r, 0) / max(ok.sum(), 1)} for r in REGIMES}}
    return out


def diagnose(res):
    pen = lambda c, pop, r: (lambda x: x.get("n", 0) > 0 and x["mean"] > 0 and x["excludes_zero"])(res[c][pop]["regimes"][r].get("contrasts", {}).get("d_total", {"n": 0}))  # noqa: E731
    t = "3dpw_test"
    with_pen = [c for c in CHAINS if pen(c, t, "A_true_near") or pen(c, t, "B_cancellation")]
    out = {"chains_with_penalty": with_pen, "per_chain": {}}
    for c in CHAINS:
        A = res[c][t]["regimes"]["A_true_near"]
        od = A.get("contrasts", {}).get("overdepth", {"n": 0})
        obs = res[c][t]["observation_split_A"].get("discrepancy", {})
        hi, lo = obs.get("high_half", {"n": 0}), obs.get("low_half", {"n": 0})
        fac = A.get("factorial", {})
        tot = fac.get("total") or 0.0
        single = fac.get("single_switch", {})
        owner = max(single, key=lambda k: single[k]) if single else None
        val_mean = res[c]["3dpw_validation"]["regimes"]["A_true_near"].get("contrasts", {}).get("d_total", {}).get("mean")
        out["per_chain"][c] = {
            "penalty_A": pen(c, t, "A_true_near"), "penalty_B": pen(c, t, "B_cancellation"),
            "depth_share_A": A.get("depth_share_distal"),
            "overdepth_A": od.get("n", 0) > 0 and od["mean"] > 0 and od["excludes_zero"],
            "observation_dominated": (hi.get("n", 0) > 0 and hi["mean"] > 0 and hi["excludes_zero"]) and not (lo.get("n", 0) > 0 and lo["mean"] > 0 and lo["excludes_zero"]),
            "validation_A_mean_positive": val_mean is not None and val_mean > 0,
            "factor_owner_A": owner, "factor_owner_share_A": (single[owner] / tot) if owner and tot > 0 else None,
            "interaction_share_A": (abs(fac.get("interaction_residual", 0.0)) / tot) if tot > 0 else None}
    pc = out["per_chain"]
    depth_chains = [c for c in CHAINS if pc[c]["penalty_A"] and (pc[c]["depth_share_A"] or 0) >= 0.6 and pc[c]["overdepth_A"] and pc[c]["validation_A_mean_positive"]]
    factor_chains = [c for c in CHAINS if pc[c]["factor_owner_share_A"] is not None and pc[c]["factor_owner_share_A"] >= 0.6
                     and (pc[c]["interaction_share_A"] or 1.0) < 0.25]
    if len(with_pen) < 2:
        cls = "D"
    elif sum(pc[c]["observation_dominated"] for c in CHAINS) >= 2:
        cls = "C"
    elif len(depth_chains) >= 2 and "left_arm" in depth_chains:
        cls = "A"
    else:
        cls = "B"
    out.update({"depth_overcorrection_chains": depth_chains, "single_factor_chains": factor_chains,
                "single_factor_identified": len(factor_chains) >= 2, "classification": cls})
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("predictions_dir", "discrepancy_dir", "bank", "w83_bank", "bedlam_val", "hist_train", "new_train", "out_dir"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise SystemExit("STOP: output directory is not empty; never overwrite experimental artifacts")
    rep = json.loads((args.predictions_dir / "prediction_report.json").read_text())
    if not rep["passed"]:
        raise SystemExit("STOP: frozen predictions did not pass the Worklog 86 parity gate")
    for p, h in rep["predictions_sha256"].items():
        if _sha(args.predictions_dir / f"predictions_{p}.npz") != h:
            raise SystemExit(f"STOP: {p} predictions changed")
    fb, mx, bv = load_bank(args.bank), ExternalBank.load(args.w83_bank), ExternalBank.load(args.bedlam_val)
    disc = np.load(args.discrepancy_dir / "discrepancy.npz")
    preds = {p: dict(np.load(args.predictions_dir / f"predictions_{p}.npz")) for p in ("3dpw", "3dhp", "bedlam_validation")}
    pops = {"3dpw_test": ("3dpw", fb, fb.indices("test"), disc["3dpw_discrepancy"], lambda s: 5),
            "3dpw_validation": ("3dpw", fb, fb.indices("validation"), disc["3dpw_discrepancy"], lambda s: 5),
            "3dhp": ("3dhp", mx, np.arange(len(mx)), disc["mpi3dhp_test_discrepancy"], lambda s: stride_for(s.split(":")[1])),
            "bedlam_validation": ("bedlam_validation", bv, np.arange(len(bv)), disc["bedlam_validation_discrepancy"], lambda s: None)}
    hist, new = ExternalBank.load(args.hist_train), ExternalBank.load(args.new_train)
    res, dumps = {}, {}
    for chain in CHAINS:
        res[chain] = {}
        for pop, (pk, bank, positions, dd, stride_of) in pops.items():
            d = chain_rows(preds[pk], chain, positions, bank.samples, dd, bank.arrays["input_2d"])
            near_any = np.isin(d["regime"], ["A_true_near", "B_cancellation"])
            res[chain][pop] = {"rows": int(len(d["pos"])), "regime_counts": dict(Counter(d["regime"].tolist())),
                               "regimes": {r: regime_block(d, d["regime"] == r) for r in REGIMES},
                               "chain_near_plane_union": regime_block(d, near_any),
                               "observation_split_A": observation_split(d, d["regime"] == "A_true_near"),
                               "temporal": temporal(d, stride_of)}
            if pop in ("3dpw_test", "3dhp"):
                dumps[f"{chain}_{pop}"] = {"position": d["pos"].tolist(), "sequence_id": d["seq"].tolist(), "frame_index": d["frame"].tolist(),
                                           "regime": d["regime"].tolist(), "conf": d["conf"].tolist(), "disc": np.nan_to_num(d["disc"], nan=-1).tolist(),
                                           **{f"{g}_{k}": (d[f"{g}_{k}"]).tolist() for g in ("g0", "g1") for k in ("e", "w")},
                                           "ge": d["ge"].tolist(), "gw": d["gw"].tolist()}
            print(json.dumps({"chain": chain, "pop": pop, "rows": res[chain][pop]["rows"], "regimes": res[chain][pop]["regime_counts"]}), flush=True)
        res[chain]["training_support"] = support({"3dpw_train": (fb, fb.indices("train")), "bedlam_train_historical": (hist, np.arange(len(hist))),
                                                  "bedlam_train_w85_new": (new, np.arange(len(new))), "3dpw_test": (fb, fb.indices("test")),
                                                  "mpi3dhp_test": (mx, np.arange(len(mx)))}, chain)
    diagnosis = diagnose(res)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "rows.json").write_text(json.dumps(dumps))
    report = {"schema": "animcv_w87_near_plane_attribution_v1", "diagnosis_contract": DIAGNOSIS,
              "prediction_report_sha256": _sha(args.predictions_dir / "prediction_report.json"),
              "discrepancy_npz_sha256": _sha(args.discrepancy_dir / "discrepancy.npz"), "results": res, "diagnosis": diagnosis}
    (args.out_dir / "report.json").write_text(json.dumps(report, indent=1, default=str))
    print(json.dumps(diagnosis, indent=1, default=str))


if __name__ == "__main__":
    main()
