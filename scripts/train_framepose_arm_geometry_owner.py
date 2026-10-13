#!/usr/bin/env python3
"""DIAGNOSTIC (Worklog 78): train G0_GEOMETRY_ONLY / G1_GEOMETRY_PLUS_VISION reach + bend-side owners.

The Worklog 73 E1 checkpoint is frozen and only run forward (its held-out chain f
must reproduce Worklog 73 ``rows_heldout.json`` before anything is trained).
Training rows, sampler, draw budget, optimizer, schedule, batch, seed and the
evaluate-every-10-epochs selection cadence are the Worklog 73 E1 contract
(3DPW TRAIN + BEDLAM TRAIN, exactly 50/50 per epoch, 11,334 draws/epoch,
200 epochs, AdamW 3e-4 -> 1e-5 cosine, weight decay 1e-4, batch 256, seed 1337).
Selection: lowest 3DPW validation value of the declared objective.  The test
split is never touched here except for writing frozen predictions.  G0 and G1
differ only in the visual tensor (exact zeros vs the frozen Qwen3-VL readout).
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

from framepose.arm_geometry_owner import (  # noqa: E402
    ARCHITECTURE, CANDIDATES, build_owner, candidate_visual, chain_targets, e1_direction, geometry_inputs, owner_loss,
    perpendicular_side, predict_owner,
)
from framepose.bank import load_bank  # noqa: E402
from framepose.bedlam_supervision import ExternalBank  # noqa: E402
from framepose.continuous_vision_depth import build_regressor, predict  # noqa: E402
from framepose.learned_vision_sensor import pair_geometry  # noqa: E402
from framepose.train import geometry_tensor  # noqa: E402
from framepose.visual_supervision_sampling import EpochSampler, train_sequence_groups  # noqa: E402

E1_SHA256 = "ee002f6a889d0d1bf02f37e61feafc1c0018cb345aecae4b997fdb882d588d20"
W73_REPORT_SHA256 = "1c05d50a5e6577329f0573786c6927b07268612ec150fbd079a1aa2bc4b03934"
W73_ROWS_SHA256 = "4e17bd77399e4c016098f70592d0cfd68f508245ae60914869f756605d9897fc"
E1_REPRODUCTION_TOLERANCE = 1e-5   # the Worklog 73 E0 held-out reproduction tolerance
TRAINING = {**{k: CONFIG[k] for k in ("optimizer", "learning_rate", "minimum_learning_rate", "weight_decay", "schedule",
                                      "epochs", "batch_size", "seed", "evaluate_every")},
            "sampler": "Worklog 73 E1_SOURCE_BALANCED_50_50 over owner-eligible rows (3DPW TRAIN | BEDLAM TRAIN)",
            "draws_per_epoch": "Worklog 73 E0/E1 budget (S0_FRAME_UNIFORM draws over 3DPW TRAIN)",
            "eligibility": "GT reach valid (three GT arm joints valid, finite, non-degenerate) AND detector chain plane valid",
            "selection": "lowest 3DPW validation objective, evaluated every 10 epochs and at the last epoch",
            "objective": ARCHITECTURE["loss"]}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _angle(a, b):
    return np.degrees(np.arccos(np.clip((a * b).sum(-1), -1.0, 1.0)))


def train_owner(candidate, readouts, geometry, d, reach_t, reach_m, side_t, side_m, primary, external, draws,
                train_pos, val_pos, ext_val_pos, device):
    import torch

    visual = candidate_visual(readouts, candidate)
    torch.manual_seed(CONFIG["seed"])
    model = build_owner(visual.shape[-1]).to(device)
    init_digest = hashlib.sha256(b"".join(v.detach().cpu().numpy().tobytes() for v in model.state_dict().values())).hexdigest()
    opt = torch.optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"], weight_decay=CONFIG["weight_decay"])
    x = torch.as_tensor(visual, device=device)
    g = torch.as_tensor(geometry, device=device)
    dt = torch.as_tensor(d.astype(np.float32), device=device)
    rt = torch.as_tensor(reach_t.astype(np.float32), device=device)
    rm = torch.as_tensor(reach_m.astype(np.float32), device=device)
    st = torch.as_tensor(side_t.astype(np.float32), device=device)
    sm = torch.as_tensor(side_m.astype(np.float32), device=device)
    sampler = SourceBalancedSampler(primary, external, draws, CONFIG["seed"])

    def evaluate(positions):
        model.eval()
        sums = np.zeros(5)
        with torch.no_grad():
            for s in range(0, len(positions), 4096):
                i = torch.as_tensor(positions[s:s + 4096], device=device)
                r, v = model(x[i], g[i])
                cos = (perpendicular_side(v, dt[i]) * st[i]).sum(-1)
                sums += [float(((r - rt[i]).abs() * rm[i]).sum()), float(rm[i].sum()),
                         float(((1 - cos) * sm[i]).sum()), float(sm[i].sum()),
                         float((torch.rad2deg(torch.arccos(cos.clamp(-1, 1))) * sm[i]).sum())]
        model.train()
        reach_mae, side = sums[0] / max(sums[1], 1), sums[2] / max(sums[3], 1)
        return {"objective": reach_mae + side, "reach_mae": reach_mae, "side_one_minus_cos": side,
                "side_angle_deg": sums[4] / max(sums[3], 1), "reach_rows": int(sums[1]), "side_rows": int(sums[3])}

    steps = math.ceil(draws / CONFIG["batch_size"]) * CONFIG["epochs"]
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
            r, v = model(x[i], g[i])
            loss, _, _ = owner_loss(r, v, dt[i], rt[i], rm[i], st[i], sm[i])
            if not torch.isfinite(loss):
                raise SystemExit(f"STOP: non-finite training loss for {candidate} at step {step}")
            loss.backward()
            opt.step()
            step += 1
        if epoch == CONFIG["epochs"] - 1 or (epoch + 1) % CONFIG["evaluate_every"] == 0:
            rec = {"epoch": epoch, "train": evaluate(train_pos), "validation": evaluate(val_pos),
                   "external_validation": evaluate(ext_val_pos)}
            curve.append(rec)
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
    for name in ("bank", "cache_dir", "external_train", "external_train_cache", "external_val", "external_val_cache",
                 "w73_dir", "out_dir"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    import torch

    device = torch.device(args.device)
    identities = {"e1_checkpoint": _sha(args.w73_dir / "E1_3DPW_PLUS_EXTERNAL.pt"), "w73_report": _sha(args.w73_dir / "report.json"),
                  "w73_rows_heldout": _sha(args.w73_dir / "rows_heldout.json")}
    for k, exp in (("e1_checkpoint", E1_SHA256), ("w73_report", W73_REPORT_SHA256), ("w73_rows_heldout", W73_ROWS_SHA256)):
        if identities[k] != exp:
            raise SystemExit(f"{k} identity mismatch: {identities[k]}")
    bank = load_bank(args.bank)
    ext_tr, ext_va = ExternalBank.load(args.external_train), ExternalBank.load(args.external_val)
    r3, ident3 = load_cache(args.cache_dir, bank.content_digest(), [s.sample_id for s in bank.samples], None)
    rtr, _ = load_cache(args.external_train_cache, ext_tr.content_digest(), [s.sample_id for s in ext_tr.samples], ident3)
    rva, _ = load_cache(args.external_val_cache, ext_va.content_digest(), [s.sample_id for s in ext_va.samples], ident3)
    ckpt = torch.load(args.w73_dir / "E1_3DPW_PLUS_EXTERNAL.pt", map_location="cpu", weights_only=False)
    if ckpt["cache_identity"]["digest"] != ident3["digest"] or ckpt["candidate"] != "E1_3DPW_PLUS_EXTERNAL":
        raise SystemExit("E1 checkpoint / cache identity mismatch")
    identities.update({"bank_content_digest": bank.content_digest(), "cache_digest": ident3["digest"],
                       "external_train_digest": ext_tr.content_digest(), "external_validation_digest": ext_va.content_digest()})

    # --- stacked array space [3DPW | BEDLAM TRAIN | BEDLAM validation], as in Worklog 73 ---
    n3, ntr = len(bank), len(ext_tr)
    readouts = np.concatenate([r3, rtr, rva])
    pairs = np.concatenate([pair_geometry(geometry_tensor(b)) for b in (bank, ext_tr, ext_va)])
    input_2d = np.concatenate([b.arrays["input_2d"] for b in (bank, ext_tr, ext_va)])
    sizes = np.asarray([s.image_size for b in (bank, ext_tr, ext_va) for s in b.samples], float)
    target_3d = np.concatenate([b.arrays["target_3d"] for b in (bank, ext_tr, ext_va)])
    target_valid = np.concatenate([b.arrays["target_valid"] for b in (bank, ext_tr, ext_va)])

    # --- frozen E1 forward pass (no target input) and held-out reproduction ---
    e1 = build_regressor(readouts.shape[-1])
    e1.load_state_dict(ckpt["state_dict"])
    e1 = e1.to(device).eval()
    for p in e1.parameters():
        p.requires_grad_(False)
    everything = np.arange(len(readouts))
    f_e1 = predict(e1, readouts, pairs, everything, device)[:, 2]
    w73 = json.loads((args.w73_dir / "rows_heldout.json").read_text())
    held = np.array([r["position"] for r in w73])
    repro = float(np.max(np.abs(f_e1[held] - np.array([r["chain_f_e1"] for r in w73]))))
    if repro > E1_REPRODUCTION_TOLERANCE:
        raise SystemExit(f"STOP: frozen E1 does not reproduce Worklog 73 held-out chain f: {repro}")
    del e1
    d, d_ok = e1_direction(f_e1, input_2d, sizes)
    reach_t, reach_m, side_t, side_m = chain_targets(target_3d, target_valid, d, d_ok)
    eligible = reach_m & d_ok
    geometry = geometry_inputs(pairs, f_e1, d)

    train3, val3 = bank.indices("train"), bank.indices("validation")
    ext_train_pos, ext_val_pos = n3 + np.arange(ntr), n3 + ntr + np.arange(len(ext_va))
    primary, external = train3[eligible[train3]], ext_train_pos[eligible[ext_train_pos]]
    groups = train_sequence_groups([s.sequence_id for s in bank.samples], train3, (reach_m & d_ok)[:n3])
    draws = EpochSampler("S0_FRAME_UNIFORM", train3, groups, CONFIG["seed"]).draws_per_epoch
    if draws != 11334:
        raise SystemExit(f"draw budget differs from Worklog 73: {draws}")
    val_eval, ext_val_eval = val3[eligible[val3]], ext_val_pos[eligible[ext_val_pos]]
    held_split = {s.split for p in held for s in [bank.samples[p]]}

    args.out_dir.mkdir(parents=True, exist_ok=True)
    results, preds = {}, {"f_e1": f_e1[:n3], "d_e1": d[:n3], "d_ok": d_ok[:n3],
                          "reach_target": reach_t[:n3], "reach_valid": reach_m[:n3], "side_target": side_t[:n3], "side_valid": side_m[:n3]}
    for cand in CANDIDATES:
        model, info, visual = train_owner(cand, readouts, geometry, d, reach_t, reach_m, side_t, side_m, primary, external,
                                          draws, primary, val_eval, ext_val_eval, device)
        r_all, s_all = predict_owner(model, visual, geometry, np.arange(n3), device)
        preds[f"{cand}_reach"], preds[f"{cand}_side"] = r_all, s_all
        torch.save({"candidate": cand, "architecture": ARCHITECTURE, "training": TRAINING, "selection": info["selection"],
                    "state_dict": model.state_dict()}, args.out_dir / f"{cand}.pt")

        def learn(positions):
            m_r, m_s = reach_m[positions] & d_ok[positions], side_m[positions]
            sp = s_all[positions] - (s_all[positions] * d[positions]).sum(1, keepdims=True) * d[positions]
            sp /= np.maximum(np.linalg.norm(sp, axis=1, keepdims=True), 1e-12)
            ang = _angle(sp[m_s], side_t[positions][m_s])
            return {"reach_rows": int(m_r.sum()), "reach_mae": float(np.abs(r_all[positions] - reach_t[positions])[m_r].mean()),
                    "side_rows": int(m_s.sum()), "side_angle_mean_deg": float(ang.mean()), "side_angle_p50_deg": float(np.median(ang)),
                    "side_agreement_share": float((ang < 90).mean())}
        results[cand] = {**{k: info[k] for k in ("selection", "final", "curve", "steps", "draws", "source_draws", "init_digest", "parameters")},
                         "learnability_3dpw": {"train_in_sample": learn(train3), "validation": learn(val3)}}
        del visual, model
        torch.cuda.empty_cache()
    a, b = results[CANDIDATES[0]], results[CANDIDATES[1]]
    control = {"same_init": a["init_digest"] == b["init_digest"], "same_parameters": a["parameters"] == b["parameters"],
               "same_steps": a["steps"] == b["steps"], "same_draws": a["draws"] == b["draws"] and a["source_draws"] == b["source_draws"]}
    if not all(control.values()):
        raise SystemExit(f"STOP: G0/G1 control identity broken: {control}")
    np.savez(args.out_dir / "predictions_3dpw.npz", **preds)
    report = {"schema": "animcv_framepose_arm_geometry_owner_training_v1", "identities": identities, "architecture": ARCHITECTURE,
              "training": TRAINING, "draws_per_epoch": int(draws), "control_identity": control,
              "e1": {"heldout_reproduction_max_abs_diff": repro, "tolerance": E1_REPRODUCTION_TOLERANCE,
                     "heldout_rows": int(len(held)), "heldout_splits": sorted(held_split),
                     "training_rows_note": "E1 predictions on TRAIN rows are in-sample (E1 was trained on them); validation/test are held out"},
              "eligibility": {"3dpw_train": int(len(primary)), "bedlam_train": int(len(external)),
                              "3dpw_train_side_valid": int(side_m[primary].sum()), "bedlam_train_side_valid": int(side_m[external].sum()),
                              "3dpw_validation": int(len(val_eval)), "bedlam_validation": int(len(ext_val_eval))},
              "predictions_npz_sha256": _sha(args.out_dir / "predictions_3dpw.npz"), "candidates": results}
    path = args.out_dir / "training_report.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps({"sha256": _sha(path), "e1_repro": repro, "control": control, "eligibility": report["eligibility"],
                      **{c: {"selection": results[c]["selection"], "learn": results[c]["learnability_3dpw"]} for c in CANDIDATES}},
                     indent=1, default=str))


if __name__ == "__main__":
    main()
