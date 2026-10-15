#!/usr/bin/env python3
"""DIAGNOSTIC (Worklog 82): train R0 / R1 corrections of the frozen Worklog 80 J0 connected arm.

Order (each check stops the run):
  1. data identity (FrameBank / BEDLAM banks, Qwen caches, splits) and the J0 checkpoint hash;
  2. frozen J0 forward pass (zero visual input, its historical contract) on every row; it must
     reproduce the Worklog 80 stored J0 predictions;
  3. anchor distribution audit with a criterion declared here before any R0/R1 result exists;
  4. zero-correction identity at initialization (corrected arm == anchor arm);
  5. R0 / R1 training under the Worklog 73/78/80 contract (3DPW TRAIN + BEDLAM TRAIN 50/50,
     11,334 draws/epoch, 200 epochs, AdamW 3e-4 -> 1e-5 cosine, wd 1e-4, batch 256, seed 1337,
     selection = lowest 3DPW validation objective); J0 weights are verified unchanged afterwards.
The test split is never used for training or selection; it is only scored in the descriptive audit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_external_visual_supervision_transfer import SourceBalancedSampler, load_cache  # noqa: E402
from run_visual_supervision_diversity import CONFIG  # noqa: E402

from framepose.anchored_arm_correction import (  # noqa: E402
    CANDIDATES, REPRESENTATION, anchor_features, anchor_from_raw, build_correction, candidate_visual, correct_numpy,
    correct_torch, correction_loss, predict_correction,
)
from framepose.bank import load_bank  # noqa: E402
from framepose.bedlam_supervision import ExternalBank  # noqa: E402
from framepose.joint_arm_geometry import build_joint_arm, normalized_arm_targets, predict_joint  # noqa: E402
from framepose.learned_vision_sensor import pair_geometry  # noqa: E402
from framepose.train import geometry_tensor  # noqa: E402

J0_SHA256 = "5332b691a58b4135ce9cb4188dba0b0f4050bf11d56b478a88de03386d5f3ba5"
W80_PREDICTIONS_SHA256 = "eaf3f57c64acd9c3eb086cd84635884aef795e0fd635ea7d3a5776d08aead00f"
DRAWS_PER_EPOCH = 11334
J0_REPRODUCTION_TOLERANCE = 1e-6
IDENTITY_TOLERANCE = 1e-6
# Declared before any R0/R1 result: a severe anchor stacking mismatch exists if, within one source, the frozen
# J0 in-sample (TRAIN) connected-arm error is below half of its held-out validation error.
AUDIT_CRITERION = {"metric": "J0 normalized elbow + wrist error (Worklog 80 objective)",
                   "severe_mismatch": "TRAIN error < 0.5 x validation error within the same source (3DPW or BEDLAM)",
                   "action": "STOP and report; no cross-fitting is launched automatically"}
TRAINING = {**{k: CONFIG[k] for k in ("optimizer", "learning_rate", "minimum_learning_rate", "weight_decay", "schedule",
                                      "epochs", "batch_size", "seed", "evaluate_every")},
            "sampler": "Worklog 73 E1_SOURCE_BALANCED_50_50 over GT-valid rows (3DPW TRAIN | BEDLAM TRAIN)",
            "draws_per_epoch": DRAWS_PER_EPOCH, "selection": REPRESENTATION["selection"], "objective": REPRESENTATION["loss"]}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _state_digest(model) -> str:
    return hashlib.sha256(b"".join(v.detach().cpu().numpy().tobytes() for v in model.state_dict().values())).hexdigest()


def _objective(elbow, wrist, elbow_t, wrist_t, mask, positions):
    p = positions[mask[positions]]
    e = np.linalg.norm(elbow[p] - elbow_t[p], axis=1)
    w = np.linalg.norm(wrist[p] - wrist_t[p], axis=1)
    return {"rows": int(len(p)), "objective": float((e + w).mean()), "elbow": float(e.mean()), "wrist": float(w.mean()),
            "reach_mae": float(np.abs(np.linalg.norm(wrist[p], axis=1) - np.linalg.norm(wrist_t[p], axis=1)).mean())}


def train_correction(candidate, readouts, geometry, anchor, elbow_t, wrist_t, mask, primary, external, val_pos, ext_val_pos, device):
    import torch

    visual = candidate_visual(readouts, candidate)
    torch.manual_seed(CONFIG["seed"])
    model = build_correction(visual.shape[-1]).to(device)
    init_digest = _state_digest(model)
    x = torch.as_tensor(visual, device=device)
    g = torch.as_tensor(geometry, device=device)
    a = torch.as_tensor(anchor, device=device)
    et = torch.as_tensor(elbow_t.astype(np.float32), device=device)
    wt = torch.as_tensor(wrist_t.astype(np.float32), device=device)
    m = torch.as_tensor(mask.astype(np.float32), device=device)
    # Zero-correction identity at initialization, on every row.
    identity = 0.0
    with torch.no_grad():
        for s in range(0, len(anchor), 4096):
            i = torch.arange(s, min(s + 4096, len(anchor)), device=device)
            c = model(x[i], g[i], a[i])
            elbow, wrist = correct_torch(c, a[i])
            identity = max(identity, float(c.abs().max()), float((elbow - a[i, 7:10]).abs().max()), float((wrist - a[i, 10:13]).abs().max()))
    if identity > IDENTITY_TOLERANCE:
        raise SystemExit(f"STOP: initial corrected arm differs from the anchor: {identity}")
    opt = torch.optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"], weight_decay=CONFIG["weight_decay"])
    sampler = SourceBalancedSampler(primary, external, DRAWS_PER_EPOCH, CONFIG["seed"])
    train_pos = np.concatenate([primary, external])

    def evaluate(positions):
        model.eval()
        sums = np.zeros(4)
        with torch.no_grad():
            for s in range(0, len(positions), 4096):
                i = torch.as_tensor(positions[s:s + 4096], device=device)
                elbow, wrist = correct_torch(model(x[i], g[i], a[i]), a[i])
                e, w = (elbow - et[i]).norm(dim=-1), (wrist - wt[i]).norm(dim=-1)
                sums += [float((e * m[i]).sum()), float((w * m[i]).sum()), float(m[i].sum()),
                         float(((wrist.norm(dim=-1) - wt[i].norm(dim=-1)).abs() * m[i]).sum())]
        model.train()
        n = max(sums[2], 1)
        return {"objective": (sums[0] + sums[1]) / n, "elbow": sums[0] / n, "wrist": sums[1] / n, "reach_mae": sums[3] / n, "rows": int(sums[2])}

    steps = math.ceil(DRAWS_PER_EPOCH / CONFIG["batch_size"]) * CONFIG["epochs"]
    step, best, best_state, curve, counts = 0, None, None, [], Counter()
    curve.append({"epoch": -1, "train": evaluate(train_pos), "validation": evaluate(val_pos), "external_validation": evaluate(ext_val_pos)})
    for epoch in range(CONFIG["epochs"]):
        model.train()
        order = sampler.epoch()
        counts.update("bedlam" if p >= external.min() else "3dpw" for p in order)
        for s in range(0, len(order), CONFIG["batch_size"]):
            i = torch.as_tensor(order[s:s + CONFIG["batch_size"]], device=device)
            progress = min(step / max(steps - 1, 1), 1.0)
            lr = CONFIG["minimum_learning_rate"] + (CONFIG["learning_rate"] - CONFIG["minimum_learning_rate"]) * 0.5 * (1 + math.cos(math.pi * progress))
            for group in opt.param_groups:
                group["lr"] = lr
            opt.zero_grad(set_to_none=True)
            loss, _, _ = correction_loss(model(x[i], g[i], a[i]), a[i], et[i], wt[i], m[i])
            if not torch.isfinite(loss):
                raise SystemExit(f"STOP: non-finite training loss for {candidate} at step {step}")
            loss.backward()
            opt.step()
            step += 1
        if epoch == CONFIG["epochs"] - 1 or (epoch + 1) % CONFIG["evaluate_every"] == 0:
            rec = {"epoch": epoch, "train": evaluate(train_pos), "validation": evaluate(val_pos), "external_validation": evaluate(ext_val_pos)}
            curve.append(rec)
            print(json.dumps({"candidate": candidate, "epoch": epoch, "train": rec["train"]["objective"],
                              "validation": rec["validation"]["objective"]}), flush=True)
            if best is None or rec["validation"]["objective"] < best["validation"]["objective"]:
                best = rec
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    return model, {"selection": best, "final": curve[-1], "curve": curve, "steps": step, "draws": int(sum(counts.values())),
                   "source_draws": dict(counts), "init_digest": init_digest, "initial_identity_max_abs": identity,
                   "parameters": int(sum(p.numel() for p in model.parameters()))}, visual


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("bank", "cache_dir", "external_train", "external_train_cache", "external_val", "external_val_cache",
                 "w80_joint_dir", "out_dir"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    import torch

    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise SystemExit("STOP: output directory is not empty; never overwrite experimental artifacts")
    device = torch.device(args.device)
    identities = {"j0_checkpoint": _sha(args.w80_joint_dir / "J0_GEOMETRY_ONLY.pt"),
                  "w80_predictions": _sha(args.w80_joint_dir / "predictions_3dpw.npz")}
    if identities["j0_checkpoint"] != J0_SHA256 or identities["w80_predictions"] != W80_PREDICTIONS_SHA256:
        raise SystemExit(f"STOP: frozen J0 identity mismatch: {identities}")
    bank = load_bank(args.bank)
    ext_tr, ext_va = ExternalBank.load(args.external_train), ExternalBank.load(args.external_val)
    r3, ident3 = load_cache(args.cache_dir, bank.content_digest(), [s.sample_id for s in bank.samples], None)
    rtr, _ = load_cache(args.external_train_cache, ext_tr.content_digest(), [s.sample_id for s in ext_tr.samples], ident3)
    rva, _ = load_cache(args.external_val_cache, ext_va.content_digest(), [s.sample_id for s in ext_va.samples], ident3)
    ids = [s.sample_id for b in (bank, ext_tr, ext_va) for s in b.samples]
    if len(set(ids)) != len(ids) or {s.split for s in ext_tr.samples} != {"train"} or {s.split for s in ext_va.samples} != {"validation"}:
        raise SystemExit("STOP: sample identity or external split changed")
    if ({s.sequence_id for s in ext_tr.samples} & {s.sequence_id for s in ext_va.samples}) \
            or {s.sequence_id for s in bank.samples if s.split == "train"} & {s.sequence_id for s in bank.samples if s.split != "train"}:
        raise SystemExit("STOP: TRAIN/held-out sequence overlap")
    training_report = json.loads((args.w80_joint_dir / "training_report.json").read_text())
    if training_report["identities"]["bank_content_digest"] != bank.content_digest():
        raise SystemExit("STOP: J0 was trained on another FrameBank")
    identities.update({"bank_content_digest": bank.content_digest(), "cache_digest": ident3["digest"],
                       "external_train_digest": ext_tr.content_digest(), "external_validation_digest": ext_va.content_digest()})

    n3, ntr = len(bank), len(ext_tr)
    readouts = np.concatenate([r3, rtr, rva])
    geometry = np.concatenate([pair_geometry(geometry_tensor(b)) for b in (bank, ext_tr, ext_va)]).reshape(len(ids), -1)
    target_3d = np.concatenate([b.arrays["target_3d"] for b in (bank, ext_tr, ext_va)])
    target_valid = np.concatenate([b.arrays["target_valid"] for b in (bank, ext_tr, ext_va)])
    elbow_t, wrist_t, mask = normalized_arm_targets(target_3d, target_valid)

    # --- frozen J0 anchor (2D only; its historical zero-visual contract) ---
    j0 = build_joint_arm(readouts.shape[-1])
    j0.load_state_dict(torch.load(args.w80_joint_dir / "J0_GEOMETRY_ONLY.pt", map_location="cpu", weights_only=False)["state_dict"])
    j0 = j0.to(device).eval()
    for p in j0.parameters():
        p.requires_grad_(False)
    j0_digest_before = _state_digest(j0)
    raw0 = predict_joint(j0, np.zeros_like(readouts, dtype=np.float16), geometry, np.arange(len(ids)), device)
    w80 = np.load(args.w80_joint_dir / "predictions_3dpw.npz")
    j0_repro = float(np.abs(raw0[:n3] - w80["J0_GEOMETRY_ONLY_raw"]).max())
    if j0_repro > J0_REPRODUCTION_TOLERANCE:
        raise SystemExit(f"STOP: frozen J0 does not reproduce Worklog 80: {j0_repro}")
    u0, v0, rho0 = anchor_from_raw(raw0)
    anchor = anchor_features(u0, v0, rho0)
    e0, w0 = anchor[:, 7:10].astype(np.float64), anchor[:, 10:13].astype(np.float64)

    # --- anchor distribution audit (criterion declared above) ---
    train3, val3, test3 = bank.indices("train"), bank.indices("validation"), bank.indices("test")
    ext_train_pos, ext_val_pos = n3 + np.arange(ntr), n3 + ntr + np.arange(len(ext_va))
    audit = {"criterion": AUDIT_CRITERION, "scopes": {k: _objective(e0, w0, elbow_t, wrist_t, mask, pos) for k, pos in (
        ("3dpw_train", train3), ("3dpw_validation", val3), ("3dpw_test", test3), ("bedlam_train", ext_train_pos), ("bedlam_validation", ext_val_pos))}}
    s = audit["scopes"]
    audit["ratios_train_over_validation"] = {"3dpw": s["3dpw_train"]["objective"] / s["3dpw_validation"]["objective"],
                                             "bedlam": s["bedlam_train"]["objective"] / s["bedlam_validation"]["objective"]}
    audit["severe_mismatch"] = any(v < 0.5 for v in audit["ratios_train_over_validation"].values())
    print(json.dumps({"j0_reproduction": j0_repro, "anchor_audit": audit}, default=str), flush=True)
    if audit["severe_mismatch"]:
        raise SystemExit(f"STOP: severe frozen-anchor stacking mismatch: {audit['ratios_train_over_validation']}")

    primary, external = train3[mask[train3]], ext_train_pos[mask[ext_train_pos]]
    val_eval, ext_val_eval = val3[mask[val3]], ext_val_pos[mask[ext_val_pos]]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    preds = {"anchor_raw": raw0[:n3], "anchor_u": u0[:n3], "anchor_v": v0[:n3], "anchor_rho": rho0[:n3],
             "anchor_elbow": e0[:n3], "anchor_wrist": w0[:n3], "target_elbow": elbow_t[:n3], "target_wrist": wrist_t[:n3], "target_valid": mask[:n3]}
    results = {}
    for cand in CANDIDATES:
        model, info, visual = train_correction(cand, readouts, geometry, anchor, elbow_t, wrist_t, mask, primary, external,
                                               val_eval, ext_val_eval, device)
        c = predict_correction(model, visual, geometry, anchor, np.arange(n3), device)
        elbow, wrist, rho, u, v = correct_numpy(c, u0[:n3], v0[:n3], rho0[:n3])
        preds.update({f"{cand}_correction": c, f"{cand}_elbow": elbow, f"{cand}_wrist": wrist, f"{cand}_rho": rho,
                      f"{cand}_u": u, f"{cand}_v": v})
        torch.save({"candidate": cand, "representation": REPRESENTATION, "training": TRAINING, "selection": info["selection"],
                    "anchor_checkpoint_sha256": J0_SHA256, "state_dict": model.state_dict()}, args.out_dir / f"{cand}.pt")
        results[cand] = info
        del visual, model
        torch.cuda.empty_cache()
    j0_digest_after = _state_digest(j0)
    if j0_digest_after != j0_digest_before or _sha(args.w80_joint_dir / "J0_GEOMETRY_ONLY.pt") != J0_SHA256:
        raise SystemExit("STOP: frozen J0 changed during training")
    a, b = results[CANDIDATES[0]], results[CANDIDATES[1]]
    control = {"same_init": a["init_digest"] == b["init_digest"], "same_parameters": a["parameters"] == b["parameters"],
               "same_steps": a["steps"] == b["steps"], "same_draws": a["draws"] == b["draws"] and a["source_draws"] == b["source_draws"]}
    if not all(control.values()):
        raise SystemExit(f"STOP: R0/R1 control identity broken: {control}")
    np.savez(args.out_dir / "predictions_3dpw.npz", **preds)
    report = {"schema": "animcv_framepose_anchored_arm_correction_training_v1", "identities": identities,
              "representation": REPRESENTATION, "training": TRAINING, "control_identity": control,
              "anchor": {"j0_reproduction_max_abs_vs_w80": j0_repro, "tolerance": J0_REPRODUCTION_TOLERANCE,
                         "j0_state_digest_before": j0_digest_before, "j0_state_digest_after": j0_digest_after,
                         "frozen": j0_digest_before == j0_digest_after,
                         "note": "J0 predictions on TRAIN rows are in-sample (J0 was trained on them); validation/test are held out"},
              "anchor_audit": audit,
              "eligibility": {"3dpw_train": int(len(primary)), "bedlam_train": int(len(external)), "3dpw_validation": int(len(val_eval)),
                              "bedlam_validation": int(len(ext_val_eval))},
              "predictions_npz_sha256": _sha(args.out_dir / "predictions_3dpw.npz"), "candidates": results}
    path = args.out_dir / "training_report.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps({"sha256": _sha(path), "control": control, "anchor": report["anchor"],
                      **{c: {"selection": results[c]["selection"], "parameters": results[c]["parameters"],
                             "identity": results[c]["initial_identity_max_abs"]} for c in CANDIDATES}}, indent=1, default=str))


if __name__ == "__main__":
    main()
