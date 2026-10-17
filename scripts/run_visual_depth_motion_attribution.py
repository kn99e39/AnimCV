#!/usr/bin/env python3
"""DIAGNOSTIC (Worklog 88): who owns the excess short-interval camera-depth motion of frozen RGB connected limbs.

Frozen models only (no training): left arm G0X / J1X, right arm G0 / G1, right leg G0 / G1.  Populations:
3DPW TEST (primary, consecutive FrameBank rows 5 frames = 1/6 s apart), 3DPW VALIDATION (same cadence,
corroboration), MPI-INF-3DHP TEST (0.4 s primary pairs and the TS5-TS6 0.16 s outdoor subset, kept separate).
BEDLAM VALIDATION has no chronological pairs and is excluded.

For every chronological pair (a, b) the frozen G1 distal-joint depth is evaluated on all 16 combinations of
four input "players" switched from frame a to frame b:
  geometry  -- the Worklog 69 pair features of the chain's detector joints (crop-normalized);
  content   -- the frozen Qwen token grid the joint readouts are sampled from;
  position  -- the sampling positions (detector joints in the frame's own 448 px crop) and their validity;
  global    -- the stored global-mean readout.
Exact Shapley values split the predicted depth change over the players.  Every non-identity combination is an
EVALUATION-ONLY sensitivity probe (an off-manifold flag is recorded), never treated as a real observation.
The all-a and all-b combinations must reproduce the frozen historical inputs and outputs exactly.

``DIAGNOSIS`` was committed before this runner produced any output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_external_visual_supervision_transfer import load_cache  # noqa: E402
from run_framepose_ik_gt_attribution import _stats  # noqa: E402
from run_framepose_visual_arm_ownership import _contrast, _excludes_zero  # noqa: E402

from framepose.bank import load_bank  # noqa: E402
from framepose.bedlam_supervision import ExternalBank  # noqa: E402
from framepose.connected_limb_transfer import GRID, chain_indices, chain_pair_geometry, chain_readout, chain_targets, token_coordinates  # noqa: E402
from framepose.joint_arm_geometry import build_joint_arm, decode_numpy  # noqa: E402
from framepose.mpi3dhp_test_transfer import SHORT_PAIR_FRAMES, stride_for  # noqa: E402
from framepose.train import geometry_tensor  # noqa: E402
from framepose.visual_depth_motion_attribution import (  # noqa: E402
    FALSE_MOTION, depth_change_terms, gt_motion_stratum, motion_terms, shapley,
)

PLAYERS = ("geometry", "content", "position", "global")
CHAINS = ("left_arm", "right_arm", "right_leg")
CKPT = {"left_arm": {"g0": "w86:left_arm_g0.pt", "g1": "w85:J1_BEDLAM_EXPANDED.pt"},
        "right_arm": {"g0": "w86:right_arm_g0.pt", "g1": "w86:right_arm_g1.pt"},
        "right_leg": {"g0": "w86:right_leg_g0.pt", "g1": "w86:right_leg_g1.pt"}}
W87_PREDICTIONS = {"3dpw": "cb9dc51e3f84ddc9b45c5ada5e941ceaaad5ee73440a6e387ee0c3f2b689c46f",
                   "3dhp": "356f8ceee545f7ca0a20510afa00efccd48782de5e6e11b78ccce2990a4fcbe1"}
W87_ATTRIBUTION_V2 = "295981b4fa03a3f9d101caabdd8f5587249a858f17a05b671e7f878310c0e316"
DIAGNOSIS = {
    "excess": "E = R1 - R0, R = mean |dP - dG| of the distal-joint depth (%L per pair), G1 vs G0 on the same pairs",
    "removal_share": "share_X = (R1 - mean |dP1 - phi_X - dG|) / E: the part of the excess that disappears when the Shapley "
                     "contribution of input group X is removed from G1's own predicted depth change",
    "visual_group": "content + position + global",
    "g1_geometry_path_excess": "(mean |phi_geometry - dG| - R0) / E: the G1 geometry pathway alone vs the G0 model",
    "observation_dominated": "pairs in the top quartile of detector 2D jitter (|d detector - d GT 2D| / GT limb diagonal, max over "
                             "P/M/D) carry >= 60% of the summed excess AND the remaining 75% of pairs show no significant excess",
    "E_blocked": "parity, readout identity or pair identity fails (STOP before classification)",
    "C_observation": ">= 2 chains observation_dominated on 3DPW TEST",
    "A_visual": "otherwise, share_visual >= 0.5 on 3DPW TEST for >= 2 chains including the left arm AND share_visual >= 0.3 on "
                "3DPW VALIDATION for those chains (sub-owner reported: content / position / global)",
    "B_model_sensitivity": "otherwise, share_visual < 0.5 AND g1_geometry_path_excess >= 0.5 for >= 2 chains on 3DPW TEST and VALIDATION",
    "D_mixed": "otherwise (interacting, population-dependent or non-identifiable)",
    "significance": "sequence-cluster bootstrap 95% interval of the per-pair quantity excludes 0 (Worklog 75/76 bootstrap)"}


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _brief(seq, values):
    items = [{"sequence_id": s, "v": float(v)} for s, v in zip(seq, values) if np.isfinite(v)]
    if not items:
        return {"n": 0}
    c = _contrast(items, lambda it: it["v"])
    b = c.get("bootstrap", {}).get("sequence_cluster", {}).get("frame_weighted_ci95")
    return {"n": c["n"], "mean": c["mean"], "sequence_weighted_mean": c["sequence_weighted_mean"], "ci95": b,
            "sequences": c["sequences"], "sequences_positive": c["sequences_positive"], "excludes_zero": _excludes_zero(c),
            "most_positive_sequence": c["most_positive_sequence"]}


def _spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 10:
        return None
    return float(np.corrcoef(np.argsort(np.argsort(x[ok])), np.argsort(np.argsort(y[ok])))[0, 1])


class Population:
    def __init__(self, name, bank, positions, grid, boxes, stored, limb, gt2d, stride_of, w87):
        self.name, self.bank, self.positions, self.grid, self.boxes, self.w87 = name, bank, positions, grid, boxes, w87
        self.stored, self.limb, self.gt2d, self.stride_of = stored, limb, gt2d, stride_of
        self.geometry = geometry_tensor(bank)


def pairs_for(pop, ok):
    by = defaultdict(dict)
    for p in pop.positions:
        if ok[p]:
            s = pop.bank.samples[p]
            by[s.sequence_id][int(s.frame_index)] = p
    out = []
    for sid, frames in by.items():
        step = pop.stride_of(sid)
        if step is None:
            continue
        for f, p in sorted(frames.items()):
            q = frames.get(f + step)
            if q is not None:
                out.append((sid, f, p, q))
    return out


def readout(pop, chain, grid_frame, coord_frame, global_frame):
    s = pop.bank.samples[coord_frame]
    rows = lambda idx, g=grid_frame: np.asarray(pop.grid[g, idx])  # noqa: E731
    glob = pop.stored[global_frame, 3].astype(np.float32)
    return chain_readout(rows, pop.bank.arrays["input_2d"][coord_frame], pop.bank.arrays["input_valid"][coord_frame],
                         tuple(s.image_size), pop.boxes[coord_frame], chain, glob).astype(np.float16)


def analyse(pop, chain, models, device):
    import torch

    e_t, w_t, ok = chain_targets(pop.bank.arrays["target_3d"], pop.bank.arrays["target_valid"], chain)
    pairs = pairs_for(pop, ok)
    if not pairs:
        return {"pairs": 0}, []
    geo = chain_pair_geometry(pop.geometry, chain).reshape(len(pop.bank), -1)
    hist = pop.stored if chain == "left_arm" else pop.limb[f"readout_{chain}"]
    coal = [frozenset(c) for k in range(5) for c in combinations(PLAYERS, k)]
    vis_in, geo_in, identity_mismatch, offgrid = [], [], 0, 0
    idx = chain_indices(chain)
    for sid, f, a, b in pairs:
        cache = {}
        for S in coal:
            g_src, c_src, m_src = (b if "content" in S else a), (b if "position" in S else a), (b if "global" in S else a)
            key = (g_src, c_src, m_src)
            if key not in cache:
                cache[key] = readout(pop, chain, g_src, c_src, m_src)
            vis_in.append(cache[key])
            geo_in.append(geo[b] if "geometry" in S else geo[a])
        for fr in (a, b):
            if not np.array_equal(cache[(fr, fr, fr)] if (fr, fr, fr) in cache else readout(pop, chain, fr, fr, fr), hist[fr]):
                identity_mismatch += 1
        cb = token_coordinates(pop.bank.arrays["input_2d"][b], tuple(pop.bank.samples[b].image_size), pop.boxes[b])[idx]
        offgrid += int(((cb < -0.5) | (cb > GRID - 0.5)).any())
    if identity_mismatch:
        raise SystemExit(f"STOP: {pop.name}/{chain}: {identity_mismatch} frames whose grid readout differs from the historical input")
    vis_in, geo_in = np.stack(vis_in), np.stack(geo_in).astype(np.float32)
    out = []
    with torch.no_grad():
        for s in range(0, len(vis_in), 4096):
            out.append(models[(chain, "g1")](torch.as_tensor(vis_in[s:s + 4096], device=device),
                                             torch.as_tensor(geo_in[s:s + 4096], device=device)).cpu().numpy().astype(np.float64))
    e_h, w_h, _ = decode_numpy(np.concatenate(out))
    n = len(pairs)
    yw = (100 * w_h[:, 1]).reshape(n, len(coal))
    ye = (100 * e_h[:, 1]).reshape(n, len(coal))
    a_idx, b_idx = np.array([p[2] for p in pairs]), np.array([p[3] for p in pairs])
    # The frozen historical outputs (Worklog 87 predictions, identical to Worklog 86) on the true inputs.
    preds = {g: (pop.w87[f"{chain}_{g}_elbow"][a_idx], pop.w87[f"{chain}_{g}_wrist"][a_idx],
                 pop.w87[f"{chain}_{g}_elbow"][b_idx], pop.w87[f"{chain}_{g}_wrist"][b_idx]) for g in ("g0", "g1")}
    # identity: all-a / all-b coalitions equal the G1 outputs on the true inputs
    full = coal.index(frozenset(PLAYERS))
    ident = float(max(np.abs(yw[:, 0] - 100 * preds["g1"][1][:, 1]).max(), np.abs(yw[:, full] - 100 * preds["g1"][3][:, 1]).max()))
    # float32 GPU outputs depend on batch composition (~1e-6 normalized after unit normalization, cf. Worklog 83);
    # 1e-3 %L = 1e-5 normalized separates that round-off from a wrong input construction.
    if ident > 1e-3:
        raise SystemExit(f"STOP: {pop.name}/{chain}: coalition identity broken ({ident})")
    phi = [shapley({S: yw[i, j] for j, S in enumerate(coal)}, PLAYERS) for i in range(n)]
    phi_m = [shapley({S: ye[i, j] for j, S in enumerate(coal)}, PLAYERS) for i in range(n)]
    seq = np.array([p[0] for p in pairs])
    gt_a, gt_b = w_t[a_idx], w_t[b_idx]
    m1 = motion_terms(preds["g1"][1], preds["g1"][3], gt_a, gt_b)
    m0 = motion_terms(preds["g0"][1], preds["g0"][3], gt_a, gt_b)
    mm1 = motion_terms(preds["g1"][0], preds["g1"][2], e_t[a_idx], e_t[b_idx])
    mm0 = motion_terms(preds["g0"][0], preds["g0"][2], e_t[a_idx], e_t[b_idx])
    dG = m1["gt_disp"]
    P = {k: np.array([p[k] for p in phi]) for k in PLAYERS}
    P["visual"] = P["content"] + P["position"] + P["global"]
    R1, R0 = float(m1["abs_residual"].mean()), float(m0["abs_residual"].mean())
    E = R1 - R0
    shares = {k: (R1 - float(np.abs(m1["pred_disp"] - P[k] - dG).mean())) / E if abs(E) > 1e-9 else None for k in (*PLAYERS, "visual")}
    geo_path = float(np.abs(P["geometry"] - dG).mean())
    # detector / crop dynamics
    size = np.array([pop.bank.samples[p].image_size for p in a_idx], float)
    det = pop.bank.arrays["input_2d"]
    dd = np.linalg.norm((det[b_idx][:, idx, :2] - det[a_idx][:, idx, :2]) * size[:, None, :], axis=2)
    if pop.gt2d is not None:
        g2 = pop.gt2d
        box = g2[a_idx][:, [11, 12, 13, 14, 15, 16, 1, 2, 3, 4, 5, 6]]
        diag = np.linalg.norm(np.nanmax(box, 1) - np.nanmin(box, 1), axis=1)
        jit = np.linalg.norm((det[b_idx][:, idx, :2] - det[a_idx][:, idx, :2]) * size[:, None, :] - (g2[b_idx][:, idx] - g2[a_idx][:, idx]), axis=2)
        jitter = np.nanmax(jit, 1) / np.maximum(diag, 1e-6)
        gt2d_disp = np.nanmax(np.linalg.norm(g2[b_idx][:, idx] - g2[a_idx][:, idx], axis=2), 1) / np.maximum(diag, 1e-6)
    else:
        jitter = gt2d_disp = np.full(n, np.nan)
        diag = np.full(n, np.nan)
    side_a, side_b = pop.boxes[a_idx, 2], pop.boxes[b_idx, 2]
    crop_scale = np.abs(np.log(side_b / side_a))
    crop_shift = np.linalg.norm(pop.boxes[b_idx, :2] + side_b[:, None] / 2 - pop.boxes[a_idx, :2] - side_a[:, None] / 2, axis=1) / side_a
    conf = det[:, idx, 2]
    dconf = np.abs(conf[b_idx] - conf[a_idx]).max(1)
    # frozen feature dynamics (the actual G1 visual inputs)
    ha, hb = hist[a_idx].astype(np.float32), hist[b_idx].astype(np.float32)
    feat = {f"slot{k}": np.linalg.norm(hb[:, k] - ha[:, k], axis=1) / np.maximum((np.linalg.norm(ha[:, k], axis=1) + np.linalg.norm(hb[:, k], axis=1)) / 2, 1e-6)
            for k in range(4)}
    excess = m1["abs_residual"] - m0["abs_residual"]
    strata = gt_motion_stratum(gt_a, gt_b)
    rep = {"pairs": n, "sequences": int(len(set(seq.tolist()))), "pair_frames": sorted({int(pop.bank.samples[q].frame_index - pop.bank.samples[p].frame_index) for _, _, p, q in pairs}),
           "R0": R0, "R1": R1, "E": E, "excess": _brief(seq, excess), "middle_R0": float(mm0["abs_residual"].mean()), "middle_R1": float(mm1["abs_residual"].mean()),
           "amplitude": {"gt": float(m1["gt_amp"].mean()), "g0": float(m0["pred_amp"].mean()), "g1": float(m1["pred_amp"].mean())},
           "over_response": {"g0": float(m0["over_response"].mean()), "g1": float(m1["over_response"].mean())},
           "static_depth_error": {g: float(np.mean(np.abs(100 * preds[g][1][:, 1] - 100 * gt_a[:, 1]))) for g in ("g0", "g1")},
           "shapley_mean_abs": {k: float(np.abs(P[k]).mean()) for k in (*PLAYERS, "visual")},
           "shapley_mean_abs_middle": {k: float(np.mean([abs(p[k]) for p in phi_m])) for k in PLAYERS},
           "removal_share": shares, "g1_geometry_path_residual": geo_path,
           "g1_geometry_path_excess": (geo_path - R0) / E if abs(E) > 1e-9 else None,
           "removal_effect": {k: _brief(seq, m1["abs_residual"] - np.abs(m1["pred_disp"] - P[k] - dG)) for k in (*PLAYERS, "visual")},
           "offgrid_position_pairs": offgrid / n,
           "geometry_terms": {g: {k: float(np.abs(v).mean()) for k, v in depth_change_terms(preds[g][0], preds[g][1], preds[g][2], preds[g][3]).items()} for g in ("g0", "g1")},
           "detector": {"joint_displacement_px": _stats(dd.max(1)), "jitter_norm": _stats(jitter), "gt2d_displacement_norm": _stats(gt2d_disp),
                        "confidence_change": _stats(dconf), "crop_scale_change": _stats(crop_scale), "crop_shift": _stats(crop_shift)},
           "features": {k: _stats(v) for k, v in feat.items()},
           "spearman": {"feature_joints_vs_visual_phi": _spearman(feat["slot0"] + feat["slot1"] + feat["slot2"], np.abs(P["visual"])),
                        "feature_joints_vs_gt_motion": _spearman(feat["slot0"] + feat["slot1"] + feat["slot2"], m1["gt_amp"]),
                        "feature_joints_vs_detector_disp": _spearman(feat["slot0"] + feat["slot1"] + feat["slot2"], dd.max(1)),
                        "feature_global_vs_visual_phi": _spearman(feat["slot3"], np.abs(P["visual"])),
                        "visual_phi_vs_gt_disp_signed": _spearman(P["visual"], dG), "geometry_phi_vs_gt_disp_signed": _spearman(P["geometry"], dG),
                        "jitter_vs_excess": _spearman(jitter, excess), "crop_scale_vs_excess": _spearman(crop_scale, excess),
                        "position_phi_abs_vs_crop_change": _spearman(np.abs(P["position"]), crop_scale + crop_shift)}}
    # GT-motion strata
    rep["gt_motion"] = {}
    for st in ("low", "mid", "high"):
        mk = strata == st
        if not mk.any():
            continue
        o = {"pairs": int(mk.sum()), "R0": float(m0["abs_residual"][mk].mean()), "R1": float(m1["abs_residual"][mk].mean()),
             "excess": _brief(seq[mk], excess[mk]), "shapley_mean_abs": {k: float(np.abs(P[k][mk]).mean()) for k in (*PLAYERS, "visual")},
             "amp": {"gt": float(m1["gt_amp"][mk].mean()), "g0": float(m0["pred_amp"][mk].mean()), "g1": float(m1["pred_amp"][mk].mean())}}
        if st == "low":
            o["false_motion_rate"] = {"g0": float((m0["pred_amp"][mk] > FALSE_MOTION).mean()), "g1": float((m1["pred_amp"][mk] > FALSE_MOTION).mean())}
        if st == "high":
            o["sign_agreement"] = {g: float(np.mean(np.sign(m["pred_disp"][mk]) == np.sign(dG[mk]))) for g, m in (("g0", m0), ("g1", m1))}
            o["tracking_slope"] = {g: float(np.polyfit(dG[mk], m["pred_disp"][mk], 1)[0]) for g, m in (("g0", m0), ("g1", m1))}
        rep["gt_motion"][st] = o
    # observation split by detector jitter
    if np.isfinite(jitter).any():
        q = np.nanpercentile(jitter, 75)
        top = jitter >= q
        tot = float(np.nansum(excess))
        rep["jitter_split"] = {"q75": float(q), "top_share_of_excess": float(np.nansum(excess[top]) / tot) if abs(tot) > 1e-9 else None,
                               "top": _brief(seq[top], excess[top]), "rest": _brief(seq[~top], excess[~top])}
    worst = sorted(((float(np.mean(excess[seq == s])), s) for s in set(seq.tolist())), reverse=True)[:5]
    rep["worst_sequences"] = [{"sequence_id": s, "mean_excess": v, "pairs": int((seq == s).sum())} for v, s in worst]
    rows = [{"sequence_id": p[0], "frame": p[1], "a": int(p[2]), "b": int(p[3]), "dG": float(dG[i]), "dP0": float(m0["pred_disp"][i]),
             "dP1": float(m1["pred_disp"][i]), "res0": float(m0["abs_residual"][i]), "res1": float(m1["abs_residual"][i]),
             "phi": {k: float(P[k][i]) for k in (*PLAYERS, "visual")}, "jitter": float(jitter[i]), "det_disp_px": float(dd.max(1)[i]),
             "dconf": float(dconf[i]), "crop_scale": float(crop_scale[i]), "crop_shift": float(crop_shift[i]),
             "feat_joints": float((feat["slot0"] + feat["slot1"] + feat["slot2"])[i] / 3), "feat_global": float(feat["slot3"][i]),
             "stratum": str(strata[i])} for i, p in enumerate(pairs)]
    return rep, rows


def classify(res):
    obs = [c for c in CHAINS if (lambda j: j is not None and j.get("top_share_of_excess") is not None and j["top_share_of_excess"] >= 0.6
                                 and not (j["rest"].get("n", 0) and j["rest"]["mean"] > 0 and j["rest"]["excludes_zero"]))(res[c]["3dpw_test"].get("jitter_split"))]
    vis_t = [c for c in CHAINS if (res[c]["3dpw_test"]["removal_share"]["visual"] or 0) >= 0.5]
    vis_ok = [c for c in vis_t if (res[c]["3dpw_validation"]["removal_share"]["visual"] or 0) >= 0.3]
    sens = [c for c in CHAINS if all((res[c][p]["removal_share"]["visual"] or 0) < 0.5 and (res[c][p]["g1_geometry_path_excess"] or 0) >= 0.5
                                     for p in ("3dpw_test", "3dpw_validation"))]
    if len(obs) >= 2:
        case = "C"
    elif len(vis_ok) >= 2 and "left_arm" in vis_ok:
        case = "A"
    elif len(sens) >= 2:
        case = "B"
    else:
        case = "D"
    sub = {c: max(("content", "position", "global"), key=lambda k: res[c]["3dpw_test"]["removal_share"][k] or -1e9) for c in CHAINS}
    return {"observation_dominated_chains": obs, "visual_chains_test": vis_t, "visual_chains_corroborated": vis_ok,
            "model_sensitivity_chains": sens, "visual_sub_owner_test": sub, "mechanical_case": case}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("bank", "cache_dir", "w83_dir", "w85_training", "w86_dir", "w87_dir", "out_dir"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    import torch

    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise SystemExit("STOP: output directory is not empty; never overwrite experimental artifacts")
    device = torch.device(args.device)
    pin = json.loads((args.w87_dir / "predictions" / "prediction_report.json").read_text())["identities"]
    paths = {"w85:J1_BEDLAM_EXPANDED.pt": args.w85_training / "J1_BEDLAM_EXPANDED.pt",
             **{f"w86:{k}": args.w86_dir / "training" / k for k in ("left_arm_g0.pt", "right_arm_g0.pt", "right_arm_g1.pt", "right_leg_g0.pt", "right_leg_g1.pt")}}
    ident = {k: _sha(p) for k, p in paths.items()}
    if any(ident[k] != pin[k.split(":", 1)[1]] for k in ident):
        raise SystemExit(f"STOP: checkpoint identity mismatch: {ident}")
    for pop, h in W87_PREDICTIONS.items():
        if _sha(args.w87_dir / "predictions" / f"predictions_{pop}.npz") != h:
            raise SystemExit(f"STOP: Worklog 87 {pop} predictions changed")
    w87v2 = args.w87_dir / "attribution_v2" / "report.json"
    if _sha(w87v2) != W87_ATTRIBUTION_V2:
        raise SystemExit("STOP: Worklog 87 attribution_v2 report changed")
    fb, mx = load_bank(args.bank), ExternalBank.load(args.w83_dir / "bank" / "mpi3dhp_test")
    r3, ident3 = load_cache(args.cache_dir, fb.content_digest(), [s.sample_id for s in fb.samples], None)
    rx, _ = load_cache(args.w83_dir / "qwen", mx.content_digest(), [s.sample_id for s in mx.samples], ident3)
    limb = {k: np.load(args.w86_dir / "readouts" / f"{k}.npz") for k in ("3dpw", "mpi3dhp_test")}
    disc = np.load(args.w87_dir / "discrepancy" / "discrepancy.npz")
    models = {}
    for chain, cands in CKPT.items():
        for g, k in cands.items():
            m = build_joint_arm(r3.shape[-1])
            m.load_state_dict(torch.load(paths[k], map_location="cpu", weights_only=False)["state_dict"])
            models[(chain, g)] = m.to(device).eval()
    g3 = np.load(args.cache_dir / "token_grid_fp16.npy", mmap_mode="r")
    gx = np.load(args.w83_dir / "qwen" / "token_grid_fp16.npy", mmap_mode="r")
    b3 = np.load(args.cache_dir / "readout.npz")["crop_box_xy_side"]
    bx = np.load(args.w83_dir / "qwen" / "readout.npz")["crop_box_xy_side"]
    p87 = {k: dict(np.load(args.w87_dir / "predictions" / f"predictions_{k}.npz")) for k in ("3dpw", "3dhp")}
    pops = {"3dpw_test": Population("3dpw_test", fb, fb.indices("test"), g3, b3, r3, limb["3dpw"], disc["3dpw_gt2d_px"], lambda s: 5, p87["3dpw"]),
            "3dpw_validation": Population("3dpw_validation", fb, fb.indices("validation"), g3, b3, r3, limb["3dpw"], disc["3dpw_gt2d_px"], lambda s: 5, p87["3dpw"]),
            "3dhp_0.4s": Population("3dhp_0.4s", mx, np.arange(len(mx)), gx, bx, rx, limb["mpi3dhp_test"], disc["mpi3dhp_test_gt2d_px"],
                                    lambda s: stride_for(s.split(":")[1]), p87["3dhp"]),
            "3dhp_0.16s_outdoor": Population("3dhp_0.16s_outdoor", mx, np.arange(len(mx)), gx, bx, rx, limb["mpi3dhp_test"], disc["mpi3dhp_test_gt2d_px"],
                                             lambda s: SHORT_PAIR_FRAMES.get(s.split(":")[1]), p87["3dhp"])}
    # parity: the frozen predictions on true inputs reproduce Worklog 87, and the Worklog 87 3DPW TEST residuals reproduce
    w87 = json.loads(w87v2.read_text())["results"]
    res, dumps, parity = {}, {}, {}
    for chain in CHAINS:
        res[chain] = {}
        for name, pop in pops.items():
            rep, rows = analyse(pop, chain, models, device)
            res[chain][name] = rep
            dumps[f"{chain}_{name}"] = rows
            print(json.dumps({"chain": chain, "pop": name, "pairs": rep["pairs"], "R0": rep.get("R0"), "R1": rep.get("R1"),
                              "share_visual": rep.get("removal_share", {}).get("visual")}), flush=True)
        for name in ("3dpw_test", "3dpw_validation"):
            for g, k in (("g0", "R0"), ("g1", "R1")):
                parity[f"{chain}:{name}:{g}"] = abs(res[chain][name][k] - w87[chain][name]["temporal"][g]["depth_motion_residual"])
    passed = max(parity.values()) <= 1e-9
    print(json.dumps({"w87_residual_parity": parity, "passed": passed}), flush=True)
    if not passed:
        raise SystemExit("STOP: Worklog 87 depth-motion residuals are not reproduced")
    diagnosis = classify(res)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "pairs.json").write_text(json.dumps(dumps))
    report = {"schema": "animcv_w88_depth_motion_attribution_v1", "diagnosis_contract": DIAGNOSIS, "players": PLAYERS,
              "identities": ident, "w87_residual_parity": parity, "results": res, "diagnosis": diagnosis}
    (args.out_dir / "report.json").write_text(json.dumps(report, indent=1, default=str))
    print(json.dumps(diagnosis, indent=1))


if __name__ == "__main__":
    main()
