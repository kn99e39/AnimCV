#!/usr/bin/env python3
"""Worklog 79: sequence cross-fitted E1 teachers and unchanged W78 owners.

Run once on LabServer63 after the immutable historical audit passes. No original
validation/test labels are used for teacher fitting or teacher selection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from audit_framepose_oof_conditioning import distribution, load_inputs, sha, write_json
from framepose_oof_contract import CONTRACT, FOLDS, assign_folds, group_key, validate_fold, validate_predictions
from framepose.arm_geometry_owner import (ARCHITECTURE, CANDIDATES, chain_targets, e1_direction,
                                         geometry_inputs, predict_owner)
from framepose.continuous_vision_depth import build_regressor, predict
from run_external_visual_supervision_transfer import SourceBalancedSampler
from run_visual_supervision_diversity import CONFIG
from train_framepose_arm_geometry_owner import E1_SHA256, TRAINING, train_owner


def teacher_fit(visual, geometry, targets, masks, primary, external, selection, device, stop_epoch=None):
    """W73 optimizer/graph/budget; internal TRAIN selection or fixed-epoch refit."""
    import torch

    torch.manual_seed(CONFIG["seed"])
    model = build_regressor(visual.shape[-1]).to(device)
    init = hashlib.sha256(b"".join(v.detach().cpu().numpy().tobytes() for v in model.state_dict().values())).hexdigest()
    opt = torch.optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"], weight_decay=CONFIG["weight_decay"])
    x, g = torch.as_tensor(visual, device=device), torch.as_tensor(geometry, device=device)
    t, m = torch.as_tensor(targets.astype(np.float32), device=device), torch.as_tensor(masks.astype(np.float32), device=device)
    sampler = SourceBalancedSampler(primary, external, 11334, CONFIG["seed"])
    schedule_steps = math.ceil(11334 / CONFIG["batch_size"]) * CONFIG["epochs"]

    def mae(pos):
        model.eval()
        total = count = 0.
        with torch.no_grad():
            for start in range(0, len(pos), 2048):
                i = torch.as_tensor(pos[start:start + 2048], device=device)
                w = m[i]
                total += float((abs(model(x[i], g[i]) - t[i]) * w).sum())
                count += float(w.sum())
        return total / max(count, 1.)

    step, best, state, curve = 0, None, None, []
    epochs = CONFIG["epochs"] if stop_epoch is None else stop_epoch + 1
    for epoch in range(epochs):
        model.train()
        order = sampler.epoch()
        for start in range(0, len(order), CONFIG["batch_size"]):
            i = torch.as_tensor(order[start:start + CONFIG["batch_size"]], device=device)
            progress = step / max(schedule_steps - 1, 1)
            lr = CONFIG["minimum_learning_rate"] + (CONFIG["learning_rate"] - CONFIG["minimum_learning_rate"]) * .5 * (1 + math.cos(math.pi * progress))
            for group in opt.param_groups:
                group["lr"] = lr
            opt.zero_grad(set_to_none=True)
            w = m[i]
            loss = (abs(model(x[i], g[i]) - t[i]) * w).sum() / w.sum().clamp_min(1.)
            if not torch.isfinite(loss):
                raise ValueError("nonfinite teacher loss")
            loss.backward()
            opt.step()
            step += 1
        if (epoch + 1) % CONFIG["evaluate_every"] == 0 or epoch == epochs - 1:
            rec = {"epoch": epoch, "train_mae": mae(np.concatenate([primary, external]))}
            if stop_epoch is None:
                rec["validation_mae"] = mae(selection)
                if best is None or rec["validation_mae"] < best["validation_mae"]:
                    best = dict(rec)
                    state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            curve.append(rec)
            print(json.dumps({"teacher_epoch": rec, "refit": stop_epoch is not None}), flush=True)
    if state is not None:
        model.load_state_dict(state)
    model.eval()
    return model, {"init_digest": init, "selection": best, "curve": curve, "steps": step,
                   "draws": epochs * 11334, "source_draws": {"3dpw": epochs * 5667, "bedlam": epochs * 5667},
                   "parameters": sum(p.numel() for p in model.parameters()), "schedule_steps": schedule_steps}


def main():
    import torch

    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("bank", "cache_dir", "external_train", "external_train_cache", "external_val", "external_val_cache",
                 "w73_dir", "audit_dir", "historical_owner_dir", "out_dir"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    audit = json.loads((args.audit_dir / "audit.json").read_text())
    if not audit["historical_parity"]["passed"]:
        raise SystemExit("STOP: historical parity gate failed")
    if args.out_dir.exists():
        raise SystemExit("STOP: output directory already exists; never overwrite experimental artifacts")
    banks, samples, visual, pairs, arrays, targets, masks, scopes, identities = load_inputs(args)
    if [b.content_digest() for b in banks] != audit["identities"]["banks"]:
        raise ValueError("banks changed since audit")
    if sha(args.w73_dir / "E1_3DPW_PLUS_EXTERNAL.pt") != E1_SHA256:
        raise ValueError("full-data E1 identity changed")
    full = np.load(args.audit_dir / "full_data_e1.npz")
    ids = [s.sample_id for s in samples]
    if list(full["sample_ids"]) != ids:
        raise ValueError("audit feature-row order mismatch")
    device = torch.device(args.device)
    train = np.concatenate([scopes["3dpw_train"], scopes["bedlam_train"]])
    fold_ids = assign_folds(samples, train)
    eligible = masks.any(1)
    plans = []
    for fold in range(FOLDS):
        pred, outer = train[fold_ids[train] == fold], train[fold_ids[train] != fold]
        inner_ids = assign_folds(samples, outer)
        inner_sel = outer[inner_ids[outer] == 0]
        fit = outer[inner_ids[outer] != 0]
        validate_fold(samples, train, pred, outer, inner_sel)
        if {group_key(samples[p]) for p in fit} & {group_key(samples[p]) for p in inner_sel}:
            raise ValueError("inner selection leakage")
        plans.append({"fold": fold, "prediction_positions": pred.tolist(), "refit_positions": outer.tolist(),
                      "initial_fit_positions": fit.tolist(), "selection_positions": inner_sel.tolist()})
    # Manifest is committed to disk BEFORE any teacher is trained. Assignment never reads labels/errors.
    args.out_dir.mkdir(parents=True)
    manifest = {"contract": CONTRACT, "samples": [{"position": p, "sample_id": s.sample_id, "source": s.source,
                 "split": s.split, "group": group_key(s), "subject": getattr(s, "detail", {}).get("subject"),
                 "fold": int(fold_ids[p])} for p, s in enumerate(samples)], "folds": plans,
                 "bank_digests": [b.content_digest() for b in banks], "audit_sha256": sha(args.audit_dir / "audit.json")}
    write_json(args.out_dir / "fold_manifest.json", manifest)
    print(json.dumps({"declared_contract": CONTRACT, "manifest_sha256": sha(args.out_dir / "fold_manifest.json")}), flush=True)
    f_oof = np.full(len(samples), np.nan)
    provenance = []
    n3 = len(banks[0])
    for plan in plans:
        fold = plan["fold"]
        fit, outer = (np.array(plan[k], int) for k in ("initial_fit_positions", "refit_positions"))
        sel = np.array([p for p in plan["selection_positions"] if p < n3 and eligible[p]], int)

        def source_positions(pos):
            return pos[(pos < n3) & eligible[pos]], pos[(pos >= n3) & eligible[pos]]

        a, b = source_positions(fit)
        model, selection_info = teacher_fit(visual, pairs, targets, masks, a, b, sel, device)
        chosen = selection_info["selection"]["epoch"]
        del model
        torch.cuda.empty_cache()
        a, b = source_positions(outer)
        model, refit_info = teacher_fit(visual, pairs, targets, masks, a, b, np.array([], int), device, chosen)
        pp = np.array(plan["prediction_positions"], int)
        first = predict(model, visual, pairs, pp, device)[:, 2]
        second = predict(model, visual, pairs, pp, device)[:, 2]
        if not np.array_equal(first, second):
            raise ValueError("nondeterministic OOF inference")
        f_oof[pp] = first
        path = args.out_dir / f"E1_FOLD_{fold}.pt"
        torch.save({"state_dict": model.state_dict(), "contract": CONTRACT, "config": CONFIG,
                    "cache_identity": identities[0], "fold": fold, "selection": selection_info["selection"],
                    "manifest_sha256": sha(args.out_dir / "fold_manifest.json")}, path)
        provenance.append({**plan, "checkpoint_sha256": sha(path), "selection_training": selection_info,
                           "refit_training": refit_info, "optimizer_positions": np.concatenate([a, b]).tolist(),
                           "prediction_repeat_max_abs_diff": float(abs(first - second).max())})
        write_json(args.out_dir / f"fold_{fold}_provenance.json", provenance[-1])
        del model
        torch.cuda.empty_cache()
    validate_predictions(samples, train, fold_ids, [ids[p] for p in train], f_oof[train], provenance)
    np.savez(args.out_dir / "oof_predictions.npz", positions=train, sample_ids=np.array(ids)[train], f=f_oof[train], fold=fold_ids[train])
    seqs = [s.sequence_id for s in samples]
    accounting = {k: distribution(f_oof, targets[:, 2], masks[:, 2], scopes[k], seqs)
                  for k in ("3dpw_train", "bedlam_train")}
    write_json(args.out_dir / "oof_distribution.json", {"scopes": accounting, "original_scopes": audit["scopes"],
               "contract": CONTRACT, "validity": True, "provenance": [sha(args.out_dir / f"fold_{f}_provenance.json") for f in range(FOLDS)]})

    # Change ONLY TRAIN conditioning; target definition comes from unchanged W78 chain_targets.
    f_full = full["f"]
    f_condition = f_full.copy()
    f_condition[train] = f_oof[train]
    sizes = np.array([s.image_size for s in samples], float)
    d, ok = e1_direction(f_condition, arrays["input_2d"], sizes)
    rt, rm, st, sm = chain_targets(arrays["target_3d"], arrays["target_valid"], d, ok)
    g_train, g_inference = geometry_inputs(pairs, f_condition, d), geometry_inputs(pairs, f_full, full["d"])
    untouched = np.setdiff1d(np.arange(len(samples)), train)
    if not np.array_equal(g_train[untouched], g_inference[untouched]):
        raise ValueError("validation/test conditioning changed")
    owner_eligible = rm & ok
    primary = scopes["3dpw_train"][owner_eligible[scopes["3dpw_train"]]]
    external = scopes["bedlam_train"][owner_eligible[scopes["bedlam_train"]]]
    val = scopes["3dpw_validation"][owner_eligible[scopes["3dpw_validation"]]]
    extval = scopes["bedlam_validation"][owner_eligible[scopes["bedlam_validation"]]]
    historical = json.loads((args.historical_owner_dir / "training_report.json").read_text())
    if len(primary) != historical["eligibility"]["3dpw_train"] or len(external) != historical["eligibility"]["bedlam_train"]:
        raise ValueError("owner training population changed")
    ownerdir = args.out_dir / "owners"
    ownerdir.mkdir()
    results = {}
    predout = {"f_e1": f_full[:n3], "d_e1": full["d"][:n3], "d_ok": full["d_ok"][:n3],
               "reach_target": rt[:n3], "reach_valid": rm[:n3], "side_target": st[:n3], "side_valid": sm[:n3]}
    for cand in CANDIDATES:
        model, info, vv = train_owner(cand, visual, g_train, d, rt, rm, st, sm, primary, external, 11334,
                                     primary, val, extval, device)
        if info["init_digest"] != historical["candidates"][cand]["init_digest"]:
            raise ValueError("owner initialization differs from historical controls")
        for key in ("steps", "draws", "source_draws", "parameters"):
            if info[key] != historical["candidates"][cand][key]:
                raise ValueError(f"owner training contract differs: {key}")
        r, s = predict_owner(model, vv, g_inference, np.arange(n3), device)
        predout[cand + "_reach"], predout[cand + "_side"] = r, s
        torch.save({"candidate": cand, "architecture": ARCHITECTURE, "training": TRAINING,
                    "selection": info["selection"], "state_dict": model.state_dict(),
                    "oof_manifest_sha256": sha(args.out_dir / "fold_manifest.json")}, ownerdir / f"{cand}.pt")
        # Standard W78 evaluator expects this descriptive field. Actual curves remain unmodified.
        info["learnability_3dpw"] = {"train_oof_conditioned": info["selection"]["train"],
                                    "validation": info["selection"]["validation"]}
        results[cand] = info
        print(json.dumps({"owner": cand, "selection": info["selection"]}), flush=True)
        del model, vv
        torch.cuda.empty_cache()
    np.savez(ownerdir / "predictions_3dpw.npz", **predout)
    report = {"schema": "animcv_w79_oof_owner_training_v1", "architecture": ARCHITECTURE, "training": TRAINING,
              "identities": historical["identities"], "eligibility": historical["eligibility"], "candidates": results,
              "control_identity": {"same_init": True, "same_parameters": True, "same_steps": True, "same_draws": True},
              "e1": {"heldout_reproduction_max_abs_diff": 0., "full_data_e1_sha256": E1_SHA256,
                     "conditioning": "OOF on TRAIN only; frozen historical full-data E1 on validation/test"},
              "predictions_npz_sha256": sha(ownerdir / "predictions_3dpw.npz"),
              "oof_manifest_sha256": sha(args.out_dir / "fold_manifest.json"), "draws_per_epoch": 11334}
    write_json(ownerdir / "training_report.json", report)
    if sha(args.w73_dir / "E1_3DPW_PLUS_EXTERNAL.pt") != E1_SHA256:
        raise ValueError("full-data E1 changed")


if __name__ == "__main__":
    main()
