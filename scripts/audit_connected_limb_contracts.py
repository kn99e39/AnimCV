#!/usr/bin/env python3
"""DIAGNOSTIC (Worklog 86): data / joint contract audit for the right-arm and leg chains, and the leg choice.

For every chain (left arm = reference, right arm, left leg, right leg) and every population (3DPW TRAIN /
VALIDATION / TEST FrameBank, BEDLAM historical TRAIN, Worklog 85 expansion TRAIN, BEDLAM VALIDATION,
MPI-INF-3DHP TEST bank): chain-valid rows (GT valid AND detector-visible on P, M, D), per-joint detector
visibility and confidence, GT bend / out-of-plane distributions, Worklog 85 coverage-cell support (the
Worklog 85 expansion was selected on LEFT-ARM geometry), and frozen-Qwen token-grid availability.
No model is trained or evaluated.

Leg choice (``LEG_RULE``, data-only, declared before any limb candidate exists): the leg whose minimum
chain-valid fraction over the seven populations is larger; tie (< 0.5 percentage point) -> the larger mean
detector confidence over those populations; still tied -> left (same side as the reference arm).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from framepose.bank import load_bank
from framepose.bedlam_supervision import ExternalBank
from framepose.connected_limb_transfer import CHAINS, chain_indices, chain_targets

LEG_RULE = "max over legs of min_population(chain-valid fraction); tie (<0.005) -> mean detector confidence; tie -> left"
BEND_BINS = [0, 15, 45, 90, 181]
ABS_F_BINS = [0.0, math.sin(math.radians(10)), 0.5, 1.01]


def cells(e, w):
    """Worklog 85 coverage cell (bend x |f| x sign f x proximal-segment elevation), chain-agnostic."""
    u = e / np.maximum(np.linalg.norm(e, axis=1, keepdims=True), 1e-12)
    v = (w - e) / np.maximum(np.linalg.norm(w - e, axis=1, keepdims=True), 1e-12)
    bend = np.degrees(np.arctan2(np.linalg.norm(np.cross(u, v), axis=1), (u * v).sum(1)))
    f = w[:, 1] / np.maximum(np.linalg.norm(w, axis=1), 1e-12)
    b = np.clip(np.digitize(bend, BEND_BINS) - 1, 0, 3)
    a = np.clip(np.digitize(np.abs(f), ABS_F_BINS) - 1, 0, 2)
    s = (f >= 0).astype(int)
    el = np.clip(np.digitize(u[:, 2], [-1.01, -0.5, 0.5, 1.01]) - 1, 0, 2)
    return ((b * 3 + a) * 2 + s) * 3 + el, bend, f


def summarize(arrays, rows, chain):
    e, w, ok = chain_targets(arrays["target_3d"][rows], arrays["target_valid"][rows], chain)
    idx = chain_indices(chain)
    inp = arrays["input_2d"][rows][:, idx]
    vis = arrays["input_valid"][rows][:, idx]
    c, bend, f = cells(e[ok], w[ok])
    return {"rows": int(len(rows)), "chain_valid": int(ok.sum()), "chain_valid_fraction": float(ok.mean()) if len(rows) else 0.0,
            "detector_visible_fraction": [float(x) for x in vis.mean(0)],
            "detector_confidence_mean": float(inp[..., 2][vis].mean()) if vis.any() else 0.0,
            "bend_bins": np.bincount(np.clip(np.digitize(bend, BEND_BINS) - 1, 0, 3), minlength=4).tolist(),
            "abs_f_ge_0.5": float((np.abs(f) >= 0.5).mean()) if ok.any() else None,
            "f_negative": float((f < 0).mean()) if ok.any() else None,
            "cell_counts": np.bincount(c, minlength=72).tolist()}


def grid_info(cache_dir: Path, n: int):
    path = cache_dir / "token_grid_fp16.npy"
    if not path.exists():
        return {"present": False}
    g = np.load(path, mmap_mode="r")
    m = json.loads((cache_dir / "manifest.json").read_text())
    return {"present": True, "shape": list(g.shape), "rows_match": int(g.shape[0]) == n,
            "readout_npz_sha256": m["readout_npz_sha256"], "bank_content_digest": m["identity"]["bank_content_digest"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("framebank", "framebank_cache", "hist_train", "hist_train_cache", "new_train", "new_train_cache",
                 "bedlam_val", "bedlam_val_cache", "mpi3dhp", "mpi3dhp_cache", "out"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit("STOP: output exists; never overwrite")
    fb = load_bank(args.framebank)
    ht, nt, bv, mx = (ExternalBank.load(p) for p in (args.hist_train, args.new_train, args.bedlam_val, args.mpi3dhp))
    pops = {"3dpw_train": (fb.arrays, fb.indices("train")), "3dpw_validation": (fb.arrays, fb.indices("validation")),
            "3dpw_test": (fb.arrays, fb.indices("test")), "bedlam_train_historical": (ht.arrays, np.arange(len(ht))),
            "bedlam_train_w85_new": (nt.arrays, np.arange(len(nt))), "bedlam_validation": (bv.arrays, np.arange(len(bv))),
            "mpi3dhp_test": (mx.arrays, np.arange(len(mx)))}
    report = {"schema": "animcv_w86_connected_limb_contract_audit_v1", "leg_rule": LEG_RULE, "chains": {}}
    for chain in CHAINS:
        report["chains"][chain] = {k: summarize(a, r, chain) for k, (a, r) in pops.items()}
        c = report["chains"][chain]
        hist = np.array(c["bedlam_train_historical"]["cell_counts"])
        exp = hist + np.array(c["bedlam_train_w85_new"]["cell_counts"])
        t = np.array(c["mpi3dhp_test"]["cell_counts"])
        c["coverage"] = {"bedlam_cells_below_20_historical": int((hist < 20).sum()), "bedlam_cells_below_20_expanded": int((exp < 20).sum()),
                         "mpi3dhp_rows_in_sparse_historical_cells": float(t[hist < 20].sum() / max(t.sum(), 1)),
                         "mpi3dhp_rows_in_sparse_expanded_cells": float(t[exp < 20].sum() / max(t.sum(), 1)),
                         "expansion_factor_chain_valid": float(exp.sum() / max(hist.sum(), 1))}
    legs = {}
    for leg in ("left_leg", "right_leg"):
        c = report["chains"][leg]
        legs[leg] = {"min_valid_fraction": min(v["chain_valid_fraction"] for k, v in c.items() if k != "coverage"),
                     "mean_confidence": float(np.mean([v["detector_confidence_mean"] for k, v in c.items() if k != "coverage"]))}
    a, b = legs["left_leg"], legs["right_leg"]
    if abs(a["min_valid_fraction"] - b["min_valid_fraction"]) >= 0.005:
        choice = "left_leg" if a["min_valid_fraction"] > b["min_valid_fraction"] else "right_leg"
    elif abs(a["mean_confidence"] - b["mean_confidence"]) > 1e-9:
        choice = "left_leg" if a["mean_confidence"] > b["mean_confidence"] else "right_leg"
    else:
        choice = "left_leg"
    report["leg_selection"] = {"criteria": legs, "selected": choice}
    report["token_grids"] = {"3dpw": grid_info(args.framebank_cache, len(fb)), "bedlam_train_historical": grid_info(args.hist_train_cache, len(ht)),
                             "bedlam_train_w85_new": grid_info(args.new_train_cache, len(nt)), "bedlam_validation": grid_info(args.bedlam_val_cache, len(bv)),
                             "mpi3dhp_test": grid_info(args.mpi3dhp_cache, len(mx))}
    report["bank_digests"] = {"3dpw": fb.content_digest(), "bedlam_train_historical": ht.content_digest(), "bedlam_train_w85_new": nt.content_digest(),
                              "bedlam_validation": bv.content_digest(), "mpi3dhp_test": mx.content_digest()}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1))
    print(json.dumps({"leg_selection": report["leg_selection"], "token_grids": report["token_grids"],
                      **{ch: {k: {kk: v[kk] for kk in ("chain_valid", "chain_valid_fraction", "detector_visible_fraction")}
                              for k, v in c.items() if k != "coverage"} | {"coverage": c["coverage"]}
                         for ch, c in report["chains"].items()}}, indent=1))
    print("sha256", hashlib.sha256(args.out.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
