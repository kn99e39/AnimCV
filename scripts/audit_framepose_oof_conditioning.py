#!/usr/bin/env python3
"""Worklog 79 diagnostic: frozen E1 distributions and saved owner reproduction.

Never trains a model or changes historical artifacts. All measurements run on
LabServer63. Output creation is exclusive, so a completed audit is immutable.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from framepose.arm_depth_probe import forward_targets
from framepose.arm_geometry_owner import (CANDIDATES, build_owner, candidate_visual,
                                         e1_direction, geometry_inputs, predict_owner)
from framepose.bank import load_bank
from framepose.bedlam_supervision import ExternalBank
from framepose.continuous_vision_depth import build_regressor, predict, sign_magnitude
from framepose.learned_vision_sensor import pair_geometry
from framepose.train import geometry_tensor
from run_external_visual_supervision_transfer import STABLE, load_cache
from train_framepose_arm_geometry_owner import E1_SHA256, W73_REPORT_SHA256, W73_ROWS_SHA256

HISTORICAL_OWNER_HASHES = {
    "G0_GEOMETRY_ONLY": "fdbd83b23029fccad75717f8d2d851539fdeb57d021006e12dd28f3501afddc1",
    "G1_GEOMETRY_PLUS_VISION": "7dd6f0e92c84e6a0e6083499b9a735aec2966f060ed0dd2ad2004e6f6029814c",
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    with Path(path).open("x") as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)


def stats(values):
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    return ({"n": int(len(v)), "mean": float(v.mean()),
             **{f"p{p:02d}": float(np.percentile(v, p)) for p in (5, 50, 95)},
             "max": float(v.max())} if len(v) else {"n": 0})


def distribution(f, gt, valid, positions, sequence_ids):
    """Descriptive chain-depth accounting, retaining source/split boundaries."""
    pos = np.asarray(positions, int)
    good = pos[np.asarray(valid)[pos]]

    def block(p):
        a, t = f[p], gt[p]
        return {"rows": int(len(p)), "absolute_error": stats(abs(a - t)),
                "prediction": stats(a), "prediction_magnitude": stats(abs(a)),
                "gt_magnitude": stats(abs(t)),
                "sign_magnitude": sign_magnitude({"e1": a}, t, STABLE)}

    by_seq = {s: block(np.array([p for p in good if sequence_ids[p] == s], int))
              for s in sorted({sequence_ids[p] for p in good})}
    ranked = sorted(by_seq, key=lambda s: by_seq[s]["absolute_error"]["mean"], reverse=True)
    total = sum(by_seq[s]["absolute_error"]["mean"] * by_seq[s]["rows"] for s in by_seq)
    out = {"population": int(len(pos)), "invalid_targets": int(len(pos) - len(good)),
           **block(good), "regimes": {}, "per_sequence": by_seq,
           "sequence_weighted_mae": float(np.mean([v["absolute_error"]["mean"] for v in by_seq.values()])),
           "largest_error_owners": [{"sequence_id": s,
                                     "error_sum_share": by_seq[s]["absolute_error"]["mean"] * by_seq[s]["rows"] / total,
                                     **by_seq[s]} for s in ranked[:5]]}
    for name, lo, hi in (("near_plane", 0, STABLE), ("intermediate", STABLE, .5), ("out_of_plane", .5, 1.01)):
        out["regimes"][name] = block(good[(abs(gt[good]) >= lo) & (abs(gt[good]) < hi)])
    return out


def load_inputs(args):
    """Check every cache's bank, sample order, backbone and preprocessing identity."""
    bank = load_bank(args.bank)
    tr, va = ExternalBank.load(args.external_train), ExternalBank.load(args.external_val)
    banks = (bank, tr, va)
    caches = (args.cache_dir, args.external_train_cache, args.external_val_cache)
    readouts, identities = [], []
    for b, c in zip(banks, caches):
        r, ident = load_cache(c, b.content_digest(), [s.sample_id for s in b.samples], identities[0] if identities else None)
        readouts.append(r)
        identities.append(ident)
    samples = [s for b in banks for s in b.samples]
    ids = [s.sample_id for s in samples]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate sample IDs")
    if {s.split for s in tr.samples} != {"train"} or {s.split for s in va.samples} != {"validation"}:
        raise ValueError("invalid external split labels")
    if {s.sequence_id for s in tr.samples} & {s.sequence_id for s in va.samples}:
        raise ValueError("external sequence overlap")
    if any(not s.detail.get("subject") for s in tr.samples + va.samples):
        raise ValueError("missing BEDLAM subject identity")
    if {s.detail["subject"] for s in tr.samples} & {s.detail["subject"] for s in va.samples}:
        raise ValueError("external subject overlap")
    if {s.image_reference.relative_path for s in tr.samples} & {s.image_reference.relative_path for s in va.samples}:
        raise ValueError("external image overlap")
    pairs = np.concatenate([pair_geometry(geometry_tensor(b)) for b in banks])
    arrays = {k: np.concatenate([b.arrays[k] for b in banks])
              for k in ("input_2d", "target_3d", "target_valid")}
    targets, masks = forward_targets(arrays["target_3d"], arrays["target_valid"])
    scopes = {"3dpw_" + split: bank.indices(split) for split in ("train", "validation", "test")}
    scopes.update(bedlam_train=len(bank) + np.arange(len(tr)),
                  bedlam_validation=len(bank) + len(tr) + np.arange(len(va)))
    return banks, samples, np.concatenate(readouts), pairs, arrays, targets, masks, scopes, identities


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("bank", "cache_dir", "external_train", "external_train_cache", "external_val", "external_val_cache",
                 "w73_dir", "historical_owner_dir", "historical_eval_dir", "out_dir"):
        p.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def main():
    import torch

    args = arguments()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    banks, samples, visual, pairs, arrays, targets, masks, scopes, cache_identities = load_inputs(args)
    identities = {"e1": sha(args.w73_dir / "E1_3DPW_PLUS_EXTERNAL.pt"),
                  "w73_report": sha(args.w73_dir / "report.json"), "w73_rows": sha(args.w73_dir / "rows_heldout.json"),
                  "banks": [b.content_digest() for b in banks], "caches": cache_identities}
    for key, expected in (("e1", E1_SHA256), ("w73_report", W73_REPORT_SHA256), ("w73_rows", W73_ROWS_SHA256)):
        if identities[key] != expected:
            raise ValueError(f"frozen artifact identity mismatch: {key}")
    device = torch.device(args.device)
    checkpoint = torch.load(args.w73_dir / "E1_3DPW_PLUS_EXTERNAL.pt", map_location="cpu", weights_only=False)
    if checkpoint["cache_identity"]["digest"] != cache_identities[0]["digest"]:
        raise ValueError("E1 cache mismatch")
    e1 = build_regressor(visual.shape[-1]).to(device).eval()
    e1.load_state_dict(checkpoint["state_dict"])
    for param in e1.parameters():
        param.requires_grad_(False)
    f = predict(e1, visual, pairs, np.arange(len(samples)), device)[:, 2]
    repeat = predict(e1, visual, pairs, np.arange(len(samples)), device)[:, 2]
    if not np.array_equal(f, repeat):
        raise ValueError("E1 inference is not deterministic")
    held = json.loads((args.w73_dir / "rows_heldout.json").read_text())
    e1_diff = float(max(abs(f[r["position"]] - r["chain_f_e1"]) for r in held))
    if e1_diff > 1e-5:
        raise ValueError("STOP: E1 held-out reproduction failed")
    sizes = np.array([s.image_size for s in samples], float)
    d, ok = e1_direction(f, arrays["input_2d"], sizes)
    geometry = geometry_inputs(pairs, f, d)
    saved = np.load(args.historical_owner_dir / "predictions_3dpw.npz")
    owner_parity = {}
    for candidate in CANDIDATES:
        path = args.historical_owner_dir / f"{candidate}.pt"
        identities[candidate] = sha(path)
        if identities[candidate] != HISTORICAL_OWNER_HASHES[candidate]:
            raise ValueError("historical owner checkpoint identity mismatch")
        model = build_owner(visual.shape[-1]).to(device).eval()
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=False)["state_dict"])
        r, side = predict_owner(model, candidate_visual(visual[:len(banks[0])], candidate), geometry,
                                np.arange(len(banks[0])), device)
        owner_parity[candidate] = {"reach_max_abs_diff": float(abs(r - saved[candidate + "_reach"]).max()),
                                   "side_max_abs_diff": float(abs(side - saved[candidate + "_side"]).max())}
        del model
    owner_parity["e1_max_abs_diff"] = float(abs(f[:len(banks[0])] - saved["f_e1"]).max())
    owner_pass = all(v <= 1e-5 for c in CANDIDATES for v in owner_parity[c].values()) and owner_parity["e1_max_abs_diff"] <= 1e-5
    # Geometry/pair parity was independently rerun with the unchanged W78 evaluator.
    new_report = json.loads((args.historical_eval_dir / "report.json").read_text())
    old_report = json.loads((args.historical_owner_dir / "eval/report.json").read_text())
    new_rows = json.loads((args.historical_eval_dir / "rows_test.json").read_text())
    old_rows = json.loads((args.historical_owner_dir / "eval/rows_test.json").read_text())
    new_pairs = json.loads((args.historical_eval_dir / "pairs_test.json").read_text())
    old_pairs = json.loads((args.historical_owner_dir / "eval/pairs_test.json").read_text())
    parity = {"checkpoint_forward": owner_parity, "checkpoint_forward_passed": owner_pass,
              "rows_exact": new_rows == old_rows, "pairs_exact": new_pairs == old_pairs,
              "rows": len(new_rows), "pairs": len(new_pairs), "e1_heldout_max_abs_diff": e1_diff,
              "baseline_gate": new_report["gate1_historical_parity"],
              "thresholds_exact": new_report["jump_thresholds_worklog76"] == old_report["jump_thresholds_worklog76"]}
    parity["passed"] = owner_pass and parity["rows_exact"] and parity["pairs_exact"] and parity["thresholds_exact"] and parity["baseline_gate"]["passed"]
    seqs = [s.sequence_id for s in samples]
    report = {"schema": "animcv_w79_e1_conditioning_audit_v1", "regime": "benchmark_detector_observation",
              "identities": identities, "historical_parity": parity,
              "scopes": {k: distribution(f, targets[:, 2], masks[:, 2], p, seqs) for k, p in scopes.items()},
              "interpretation": "Descriptive only: source-domain and selection differences do not establish causality."}
    write_json(args.out_dir / "audit.json", report)
    np.savez(args.out_dir / "full_data_e1.npz", f=f, sample_ids=np.array([s.sample_id for s in samples]),
             d=d, d_ok=ok)
    print(json.dumps({"parity": parity, "mae": {k: v["absolute_error"] for k, v in report["scopes"].items()}}, indent=2), flush=True)
    if not parity["passed"]:
        raise SystemExit("STOP: historical parity failed; CASE D, do not train or interpret OOF owners")
    if sha(args.w73_dir / "E1_3DPW_PLUS_EXTERNAL.pt") != E1_SHA256:
        raise ValueError("E1 changed during audit")


if __name__ == "__main__":
    main()
