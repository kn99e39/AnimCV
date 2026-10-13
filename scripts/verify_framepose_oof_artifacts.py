#!/usr/bin/env python3
"""Worklog 79 real-data contract verification, including saved teacher reloads."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from audit_framepose_oof_conditioning import load_inputs, sha, write_json
from framepose_oof_contract import FOLDS, assign_folds, group_key, validate_predictions
from framepose.continuous_vision_depth import build_regressor, predict
from framepose.arm_geometry_owner import (ARCHITECTURE, CANDIDATES, build_owner, candidate_visual,
                                         chain_targets, e1_direction, geometry_inputs, predict_owner)
from train_framepose_arm_geometry_owner import E1_SHA256


def main():
    import torch

    p = argparse.ArgumentParser(description=__doc__)
    for name in ("bank", "cache_dir", "external_train", "external_train_cache", "external_val", "external_val_cache",
                 "w73_dir", "audit_dir", "crossfit_dir", "out"):
        p.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    banks, samples, visual, pairs, arrays, targets, masks, scopes, identities = load_inputs(args)
    device = torch.device(args.device)
    train = np.concatenate([scopes["3dpw_train"], scopes["bedlam_train"]])
    manifest = json.loads((args.crossfit_dir / "fold_manifest.json").read_text())
    fold = assign_folds(samples, train)
    if not np.array_equal(fold, [r["fold"] for r in manifest["samples"]]):
        raise ValueError("nondeterministic assignment or changed manifest")
    if [s.sample_id for s in samples] != [r["sample_id"] for r in manifest["samples"]]:
        raise ValueError("sample/feature row mismatch")
    saved = np.load(args.crossfit_dir / "oof_predictions.npz")
    if not np.array_equal(train, saved["positions"]):
        raise ValueError("OOF row positions changed")
    provenance = [json.loads((args.crossfit_dir / f"fold_{f}_provenance.json").read_text()) for f in range(FOLDS)]
    validate_predictions(samples, train, fold, saved["sample_ids"], saved["f"], provenance)
    f_oof = np.full(len(samples), np.nan)
    f_oof[train] = saved["f"]
    diffs = {}
    for rec in provenance:
        f = rec["fold"]
        fit, sel, outer = (set(rec[k]) for k in ("initial_fit_positions", "selection_positions", "refit_positions"))
        if fit & sel or fit | sel != outer:
            raise ValueError("inner TRAIN/selection partition invalid")
        if {group_key(samples[p]) for p in fit} & {group_key(samples[p]) for p in sel}:
            raise ValueError("inner sequence-group leakage")
        if set(rec["optimizer_positions"]) != {p for p in outer if masks[p].any()}:
            raise ValueError("teacher optimizer used unexpected samples")
        path = args.crossfit_dir / f"E1_FOLD_{f}.pt"
        if sha(path) != rec["checkpoint_sha256"]:
            raise ValueError("teacher checkpoint changed")
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        if ckpt["manifest_sha256"] != sha(args.crossfit_dir / "fold_manifest.json"):
            raise ValueError("teacher provenance not bound to manifest")
        model = build_regressor(visual.shape[-1]).to(device).eval()
        model.load_state_dict(ckpt["state_dict"])
        pp = np.array(rec["prediction_positions"], int)
        predicted = predict(model, visual, pairs, pp, device)[:, 2]  # no target argument
        diffs[f] = float(abs(predicted - f_oof[pp]).max())
        if diffs[f] != 0:
            raise ValueError("saved OOF inference changed after checkpoint reload")
        del model
    full = np.load(args.audit_dir / "full_data_e1.npz")
    f_train = full["f"].copy()
    f_train[train] = f_oof[train]
    sizes = np.array([s.image_size for s in samples], float)
    d, ok = e1_direction(f_train, arrays["input_2d"], sizes)
    rt, rm, st, sm = chain_targets(arrays["target_3d"], arrays["target_valid"], d, ok)
    frozen_geometry = geometry_inputs(pairs, full["f"], full["d"])
    held = np.setdiff1d(np.arange(len(samples)), train)
    if not np.array_equal(geometry_inputs(pairs, f_train, d)[held], frozen_geometry[held]):
        raise ValueError("held-out owner inputs changed")
    ownerdir = args.crossfit_dir / "owners"
    preds = np.load(ownerdir / "predictions_3dpw.npz")
    report = json.loads((ownerdir / "training_report.json").read_text())
    if report["predictions_npz_sha256"] != sha(ownerdir / "predictions_3dpw.npz"):
        raise ValueError("owner predictions changed")
    n3 = len(banks[0])
    for name, expected in (("f_e1", full["f"]), ("d_e1", full["d"]), ("reach_target", rt),
                           ("reach_valid", rm), ("side_target", st), ("side_valid", sm)):
        if not np.array_equal(preds[name], expected[:n3]):
            raise ValueError(f"owner target/input contract changed: {name}")
    owner_diffs = {}
    for cand in CANDIDATES:
        ckpt = torch.load(ownerdir / f"{cand}.pt", map_location="cpu", weights_only=False)
        if ckpt["architecture"] != ARCHITECTURE:
            raise ValueError("owner architecture changed")
        model = build_owner(visual.shape[-1]).to(device).eval()
        model.load_state_dict(ckpt["state_dict"])
        r, s = predict_owner(model, candidate_visual(visual[:n3], cand), frozen_geometry, np.arange(n3), device)
        owner_diffs[cand] = {"reach": float(abs(r - preds[cand + "_reach"]).max()),
                             "side": float(abs(s - preds[cand + "_side"]).max())}
        if any(v != 0 for v in owner_diffs[cand].values()):
            raise ValueError("owner checkpoint output reproduction failed")
        del model
    subjects = {s.detail["subject"] for s in banks[1].samples}
    subject_folds = {s: sorted({int(fold[n3 + p]) for p, row in enumerate(banks[1].samples) if row.detail["subject"] == s})
                     for s in sorted(subjects)}
    if sha(args.w73_dir / "E1_3DPW_PLUS_EXTERNAL.pt") != E1_SHA256:
        raise ValueError("historical full-data E1 changed")
    write_json(args.out, {"passed": True, "oof_rows": len(train), "folds": FOLDS, "fold_reload_max_abs_diff": diffs,
               "owner_reload_max_abs_diff": owner_diffs, "heldout_inputs_exact": True, "targets_exact": True,
               "full_data_e1_sha256": E1_SHA256, "subject_folds": subject_folds,
               "subjects_repeated_across_folds": sum(len(v) > 1 for v in subject_folds.values()),
               "subject_limit": "sequence-level OOF, not subject-disjoint; explicitly declared before training"})
    print("PASS: real-data fold provenance, saved inference, frozen held-out inputs and target contracts", flush=True)


if __name__ == "__main__":
    main()
