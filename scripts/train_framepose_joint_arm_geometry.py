#!/usr/bin/env python3
"""DIAGNOSTIC (Worklog 80): train J0_GEOMETRY_ONLY / J1_GEOMETRY_PLUS_VISION one-stage joint arm models.

Training contract = the Worklog 73/78 contract, unchanged except that no E1 prediction is
computed or consumed: 3DPW TRAIN + BEDLAM TRAIN, exactly 50/50 per epoch over GT-valid rows,
11,334 draws/epoch, 200 epochs (9,000 steps), AdamW 3e-4 -> 1e-5 per-step cosine, weight decay
1e-4, batch 256, seed 1337, evaluation every 10 epochs; selection = lowest 3DPW validation
connected-arm objective.  The test split is never touched except for writing frozen
predictions.  J0 and J1 differ only in the visual tensor (exact zeros vs frozen Qwen readout).
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

from framepose.bank import load_bank  # noqa: E402
from framepose.bedlam_supervision import ExternalBank  # noqa: E402
from framepose.joint_arm_geometry import (  # noqa: E402
    CANDIDATES, REPRESENTATION, build_joint_arm, candidate_visual, decode_numpy, decode_torch, joint_loss,
    normalized_arm_targets, predict_joint,
)
from framepose.learned_vision_sensor import pair_geometry  # noqa: E402
from framepose.train import geometry_tensor  # noqa: E402
from framepose.visual_supervision_sampling import EpochSampler, train_sequence_groups  # noqa: E402

DRAWS_PER_EPOCH = 11334   # the Worklog 73/78 budget (S0 frame-uniform draws over 3DPW TRAIN)
TRAINING = {**{k: CONFIG[k] for k in ("optimizer", "learning_rate", "minimum_learning_rate", "weight_decay", "schedule",
                                      "epochs", "batch_size", "seed", "evaluate_every")},
            "sampler": "Worklog 73 E1_SOURCE_BALANCED_50_50 over GT-valid rows (3DPW TRAIN | BEDLAM TRAIN)",
            "draws_per_epoch": DRAWS_PER_EPOCH, "selection": REPRESENTATION["selection_metric"],
            "objective": REPRESENTATION["loss"], "inputs": "pair geometry (39) + frozen Qwen readout (J1) / zeros (J0); "
                                                          "no E1, H0, oracle sign, GT or temporal input"}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def train_joint(candidate, readouts, geometry, elbow_t, wrist_t, mask, primary, external, val_pos, ext_val_pos, device):
    import torch

    visual = candidate_visual(readouts, candidate)
    torch.manual_seed(CONFIG["seed"])
    model = build_joint_arm(visual.shape[-1]).to(device)
    init_digest = hashlib.sha256(b"".join(v.detach().cpu().numpy().tobytes() for v in model.state_dict().values())).hexdigest()
    opt = torch.optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"], weight_decay=CONFIG["weight_decay"])
    x = torch.as_tensor(visual, device=device)
    g = torch.as_tensor(geometry, device=device)
    et = torch.as_tensor(elbow_t.astype(np.float32), device=device)
    wt = torch.as_tensor(wrist_t.astype(np.float32), device=device)
    m = torch.as_tensor(mask.astype(np.float32), device=device)
    sampler = SourceBalancedSampler(primary, external, DRAWS_PER_EPOCH, CONFIG["seed"])
    train_pos = np.concatenate([primary, external])

    def evaluate(positions):
        model.eval()
        sums = np.zeros(5)
        with torch.no_grad():
            for s in range(0, len(positions), 4096):
                i = torch.as_tensor(positions[s:s + 4096], device=device)
                elbow, wrist = decode_torch(model(x[i], g[i]))
                e, w = (elbow - et[i]).norm(dim=-1), (wrist - wt[i]).norm(dim=-1)
                reach_err = (wrist.norm(dim=-1) - wt[i].norm(dim=-1)).abs()
                sums += [float((e * m[i]).sum()), float((w * m[i]).sum()), float(m[i].sum()), float((reach_err * m[i]).sum()), 0]
        model.train()
        n = max(sums[2], 1)
        return {"objective": (sums[0] + sums[1]) / n, "elbow": sums[0] / n, "wrist": sums[1] / n, "reach_mae": sums[3] / n,
                "rows": int(sums[2])}

    steps = math.ceil(DRAWS_PER_EPOCH / CONFIG["batch_size"]) * CONFIG["epochs"]
    step, best, best_state, curve, counts = 0, None, None, [], Counter()
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
            loss, _, _ = joint_loss(model(x[i], g[i]), et[i], wt[i], m[i])
            if not torch.isfinite(loss):
                raise SystemExit(f"STOP: non-finite training loss for {candidate} at step {step}")
            loss.backward()
            opt.step()
            step += 1
        if epoch == CONFIG["epochs"] - 1 or (epoch + 1) % CONFIG["evaluate_every"] == 0:
            rec = {"epoch": epoch, "train": evaluate(train_pos), "validation": evaluate(val_pos),
                   "external_validation": evaluate(ext_val_pos)}
            curve.append(rec)
            print(json.dumps({"candidate": candidate, "epoch": epoch, "train": rec["train"]["objective"],
                              "validation": rec["validation"]["objective"]}), flush=True)
            if best is None or rec["validation"]["objective"] < best["validation"]["objective"]:
                best = rec
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    return model, {"selection": best, "final": curve[-1], "curve": curve, "steps": step,
                   "draws": int(sum(counts.values())), "source_draws": dict(counts), "init_digest": init_digest,
                   "parameters": int(sum(p.numel() for p in model.parameters()))}, visual


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("bank", "cache_dir", "external_train", "external_train_cache", "external_val", "external_val_cache", "out_dir"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    import torch

    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise SystemExit("STOP: output directory is not empty; never overwrite experimental artifacts")
    device = torch.device(args.device)
    bank = load_bank(args.bank)
    ext_tr, ext_va = ExternalBank.load(args.external_train), ExternalBank.load(args.external_val)
    # --- Gate 1 (data identity): frozen Qwen caches bound to bank digests / sample order / W68 fingerprint ---
    r3, ident3 = load_cache(args.cache_dir, bank.content_digest(), [s.sample_id for s in bank.samples], None)
    rtr, _ = load_cache(args.external_train_cache, ext_tr.content_digest(), [s.sample_id for s in ext_tr.samples], ident3)
    rva, _ = load_cache(args.external_val_cache, ext_va.content_digest(), [s.sample_id for s in ext_va.samples], ident3)
    ids = [s.sample_id for b in (bank, ext_tr, ext_va) for s in b.samples]
    if len(set(ids)) != len(ids):
        raise SystemExit("STOP: duplicate sample identities")
    if {s.split for s in ext_tr.samples} != {"train"} or {s.split for s in ext_va.samples} != {"validation"}:
        raise SystemExit("STOP: external split labels changed")
    if ({s.sequence_id for s in ext_tr.samples} & {s.sequence_id for s in ext_va.samples}) \
            or ({s.detail.get("subject") for s in ext_tr.samples} & {s.detail.get("subject") for s in ext_va.samples}):
        raise SystemExit("STOP: external TRAIN/validation overlap")
    if {s.sequence_id for s in bank.samples if s.split == "train"} & {s.sequence_id for s in bank.samples if s.split != "train"}:
        raise SystemExit("STOP: 3DPW TRAIN/held-out sequence overlap")
    identities = {"bank_content_digest": bank.content_digest(), "cache_digest": ident3["digest"],
                  "external_train_digest": ext_tr.content_digest(), "external_validation_digest": ext_va.content_digest()}

    n3, ntr = len(bank), len(ext_tr)
    readouts = np.concatenate([r3, rtr, rva])
    geometry = np.concatenate([pair_geometry(geometry_tensor(b)) for b in (bank, ext_tr, ext_va)]).reshape(len(ids), -1)
    target_3d = np.concatenate([b.arrays["target_3d"] for b in (bank, ext_tr, ext_va)])
    target_valid = np.concatenate([b.arrays["target_valid"] for b in (bank, ext_tr, ext_va)])
    elbow_t, wrist_t, mask = normalized_arm_targets(target_3d, target_valid)
    train3, val3 = bank.indices("train"), bank.indices("validation")
    ext_train_pos, ext_val_pos = n3 + np.arange(ntr), n3 + ntr + np.arange(len(ext_va))
    primary, external = train3[mask[train3]], ext_train_pos[mask[ext_train_pos]]
    if EpochSampler("S0_FRAME_UNIFORM", train3, train_sequence_groups([s.sequence_id for s in bank.samples], train3, mask[:n3]),
                    CONFIG["seed"]).draws_per_epoch != DRAWS_PER_EPOCH:
        raise SystemExit("STOP: draw budget differs from Worklog 73/78")
    val_eval, ext_val_eval = val3[mask[val3]], ext_val_pos[mask[ext_val_pos]]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    results, preds = {}, {"target_elbow": elbow_t[:n3], "target_wrist": wrist_t[:n3], "target_valid": mask[:n3]}
    for cand in CANDIDATES:
        model, info, visual = train_joint(cand, readouts, geometry, elbow_t, wrist_t, mask, primary, external,
                                          val_eval, ext_val_eval, device)
        raw = predict_joint(model, visual, geometry, np.arange(n3), device)
        elbow, wrist, rho = decode_numpy(raw)
        preds[f"{cand}_raw"], preds[f"{cand}_elbow"], preds[f"{cand}_wrist"], preds[f"{cand}_rho"] = raw, elbow, wrist, rho
        torch.save({"candidate": cand, "representation": REPRESENTATION, "training": TRAINING, "selection": info["selection"],
                    "state_dict": model.state_dict()}, args.out_dir / f"{cand}.pt")
        results[cand] = info
        del visual, model
        torch.cuda.empty_cache()
    a, b = results[CANDIDATES[0]], results[CANDIDATES[1]]
    control = {"same_init": a["init_digest"] == b["init_digest"], "same_parameters": a["parameters"] == b["parameters"],
               "same_steps": a["steps"] == b["steps"], "same_draws": a["draws"] == b["draws"] and a["source_draws"] == b["source_draws"]}
    if not all(control.values()):
        raise SystemExit(f"STOP: J0/J1 control identity broken: {control}")
    np.savez(args.out_dir / "predictions_3dpw.npz", **preds)
    report = {"schema": "animcv_framepose_joint_arm_training_v1", "identities": identities, "representation": REPRESENTATION,
              "training": TRAINING, "control_identity": control,
              "eligibility": {"3dpw_train": int(len(primary)), "bedlam_train": int(len(external)),
                              "3dpw_validation": int(len(val_eval)), "bedlam_validation": int(len(ext_val_eval)),
                              "3dpw_train_rows": int(len(train3)), "bedlam_train_rows": int(ntr)},
              "predictions_npz_sha256": _sha(args.out_dir / "predictions_3dpw.npz"), "candidates": results}
    path = args.out_dir / "training_report.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps({"sha256": _sha(path), "control": control, "eligibility": report["eligibility"],
                      **{c: {"selection": results[c]["selection"], "parameters": results[c]["parameters"]} for c in CANDIDATES}},
                     indent=1, default=str))


if __name__ == "__main__":
    main()
