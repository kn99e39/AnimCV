#!/usr/bin/env python3
"""DIAGNOSTIC (Worklog 71): S0_FRAME_UNIFORM vs S1_SEQUENCE_BALANCED for the fixed Worklog 69 visual head.

Fixed: Worklog 68 cache, Worklog 69 architecture (C1_QWEN_VISION inputs),
target, masks, optimizer, schedule, seed, batch, epochs, number of draws per
epoch, initialization, selection, validation/test distributions.  Only the
TRAIN example sampler differs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from framepose.arm_depth_probe import (
    SEGMENT_NAMES, angle_degrees, elevation_degrees, forward_targets, observed_plane, reconstruct_direction,
    segment_vectors,
)
from framepose.bank import load_bank
from framepose.continuous_vision_depth import build_regressor, candidate_visual, predict, sign_magnitude
from framepose.learned_vision_sensor import pair_geometry
from framepose.train import geometry_tensor
from framepose.visual_supervision_sampling import (
    SAMPLERS, EpochSampler, contribution_summary, cosine_by_lag, train_sequence_groups,
)

CONFIG = {"optimizer": "AdamW", "learning_rate": 3e-4, "minimum_learning_rate": 1e-5, "weight_decay": 1e-4,
          "schedule": "cosine per step", "epochs": 200, "batch_size": 256, "seed": 1337, "evaluate_every": 10,
          "loss": "masked MAE on tanh-bounded f, unweighted", "selection": "validation masked mean |f error|",
          "architecture": "Worklog 69 C1_QWEN_VISION (unchanged)"}
STABLE = math.sin(math.radians(10))
CLEAN = ("courtyard_dancing_00", "courtyard_hug_00")
F_BINS = ((0.0, STABLE), (STABLE, 0.5), (0.5, 1.0001))
LAGS = (1, 2, 4, 8)


def _stats(v):
    v = np.asarray([x for x in v if x is not None and np.isfinite(x)], float)
    return {"n": int(len(v)), "mean": float(v.mean()), "p05": float(np.percentile(v, 5)), "p50": float(np.percentile(v, 50)),
            "p95": float(np.percentile(v, 95))} if len(v) else {"n": 0}


def train(sampler, visual, geometry, targets, masks, train_pos, val_pos, sequence_ids, device):
    import torch

    torch.manual_seed(CONFIG["seed"])
    model = build_regressor(visual.shape[-1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"], weight_decay=CONFIG["weight_decay"])
    x = torch.as_tensor(visual, device=device)
    g = torch.as_tensor(geometry, device=device)
    t = torch.as_tensor(targets.astype(np.float32), device=device)
    m = torch.as_tensor(masks.astype(np.float32), device=device)

    def mae(positions):
        model.eval()
        total = count = 0.0
        with torch.no_grad():
            for s in range(0, len(positions), 2048):
                index = torch.as_tensor(positions[s:s + 2048], device=device)
                w = m[index]
                total += float((torch.abs(model(x[index], g[index]) - t[index]) * w).sum())
                count += float(w.sum())
        model.train()
        return total / max(count, 1.0)

    steps = math.ceil(sampler.draws_per_epoch / CONFIG["batch_size"]) * CONFIG["epochs"]
    step, best, best_state, curve = 0, {"epoch": None, "validation_mae": None}, None, []
    counts: dict[str, int] = defaultdict(int)
    eligible_draws = 0
    for epoch in range(CONFIG["epochs"]):
        model.train()
        order = sampler.epoch()
        for p in order:
            counts[sequence_ids[p]] += 1
        eligible_draws += int(masks[order].any(axis=1).sum())
        for s in range(0, len(order), CONFIG["batch_size"]):
            index = torch.as_tensor(order[s:s + CONFIG["batch_size"]], device=device)
            progress = min(step / max(steps - 1, 1), 1.0)
            lr = CONFIG["minimum_learning_rate"] + (CONFIG["learning_rate"] - CONFIG["minimum_learning_rate"]) * 0.5 * (1 + math.cos(math.pi * progress))
            for group in opt.param_groups:
                group["lr"] = lr
            opt.zero_grad(set_to_none=True)
            w = m[index]
            loss = (torch.abs(model(x[index], g[index]) - t[index]) * w).sum() / w.sum().clamp_min(1.0)
            loss.backward()
            opt.step()
            step += 1
        if epoch == CONFIG["epochs"] - 1 or (epoch + 1) % CONFIG["evaluate_every"] == 0:
            record = {"epoch": epoch, "train_mae": mae(train_pos), "validation_mae": mae(val_pos)}
            curve.append(record)
            if best["epoch"] is None or record["validation_mae"] < best["validation_mae"]:
                best = dict(record)
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    final = curve[-1]
    for rec in (best, final):
        rec["gap"] = rec["validation_mae"] - rec["train_mae"]
        rec["ratio"] = rec["validation_mae"] / rec["train_mae"]
    return model, {"selection": best, "final": final, "curve": curve, "steps": step,
                   "draws": int(sum(counts.values())), "eligible_draws": eligible_draws,
                   "parameters": int(sum(p.numel() for p in model.parameters()))}, dict(counts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("bank", "cache_dir", "w69_dir", "out_dir"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    import torch

    bank = load_bank(args.bank)
    manifest = json.loads((args.cache_dir / "manifest.json").read_text())
    readout_path = args.cache_dir / "readout.npz"
    if (manifest["identity"]["bank_content_digest"] != bank.content_digest()
            or manifest["identity"]["sample_ids_sha256"] != hashlib.sha256("\n".join(s.sample_id for s in bank.samples).encode()).hexdigest()
            or manifest["identity"]["model_fingerprint"] != "a0d72ded575eaa4460dad11dd1e313bc69f0403b2c97c3c1ba234af086952904"
            or manifest["readout_npz_sha256"] != hashlib.sha256(readout_path.read_bytes()).hexdigest()):
        raise SystemExit("Worklog 68 cache identity mismatch")
    w69_report = json.loads((args.w69_dir / "report.json").read_text())
    w69_ckpt = torch.load(args.w69_dir / "C1_QWEN_VISION.pt", map_location="cpu", weights_only=False)
    if w69_ckpt["config"]["batch_size"] != CONFIG["batch_size"] or w69_ckpt["config"]["epochs"] != CONFIG["epochs"] \
            or w69_ckpt["config"]["seed"] != CONFIG["seed"] or w69_ckpt["cache_identity"]["digest"] != manifest["identity"]["digest"]:
        raise SystemExit("Worklog 69 model identity mismatch")
    readouts = np.load(readout_path)["readout"]
    visual = candidate_visual(readouts, "C1_QWEN_VISION")
    geometry = pair_geometry(geometry_tensor(bank))
    targets, masks = forward_targets(bank.arrays["target_3d"], bank.arrays["target_valid"])
    eligible = masks.any(axis=1)
    train_pos, val_pos = bank.indices("train"), bank.indices("validation")
    sequence_ids = [s.sequence_id for s in bank.samples]
    groups = train_sequence_groups(sequence_ids, train_pos, eligible)
    device = torch.device(args.device)

    # --- 3. TRAIN composition and visual redundancy (diagnostic) ---
    raw_counts = defaultdict(int)
    for p in train_pos:
        raw_counts[sequence_ids[p]] += 1
    sizes = np.array([len(v) for v in groups.values()])
    shares = sizes / sizes.sum()
    composition = {"sequences": len(groups), "train_rows": int(len(train_pos)), "eligible_rows": int(sizes.sum()),
                   "largest_share": float(shares.max()), "top5_share": float(np.sort(shares)[-5:].sum()),
                   "median_size": float(np.median(sizes)), "min_size": int(sizes.min()), "max_size": int(sizes.max()),
                   "per_sequence": {}}
    arm = readouts[:, :3].reshape(len(bank), -1)
    glob = readouts[:, 3]
    redundancy = {f"lag_{lag}": {"arm": [], "global": []} for lag in LAGS}
    frame_gaps = []
    for sid, pos in groups.items():
        order = pos[np.argsort([bank.samples[p].frame_index for p in pos])]
        fi = [bank.samples[p].frame_index for p in order]
        ts = [bank.samples[p].timestamp for p in order]
        frame_gaps += list(np.diff(fi))
        composition["per_sequence"][sid] = {"train_rows": raw_counts[sid], "eligible_rows": int(len(pos)),
                                            "share_of_eligible": float(len(pos) / sizes.sum()),
                                            "frame_index_span": [int(fi[0]), int(fi[-1])],
                                            "seconds_span": float(ts[-1] - ts[0]) if ts[0] is not None else None}
        for lag in LAGS:
            redundancy[f"lag_{lag}"]["arm"] += list(cosine_by_lag(arm, order, lag))
            redundancy[f"lag_{lag}"]["global"] += list(cosine_by_lag(glob, order, lag))
    rng = np.random.default_rng(0)
    all_train = np.concatenate(list(groups.values()))
    a, b = rng.choice(all_train, 5000), rng.choice(all_train, 5000)
    cross = [(x, y) for x, y in zip(a, b) if sequence_ids[x] != sequence_ids[y]]
    xa, xb = np.array([c[0] for c in cross]), np.array([c[1] for c in cross])

    def cos(fe, i, j):
        u, v = fe[i].astype(np.float32), fe[j].astype(np.float32)
        return np.sum(u * v, 1) / np.maximum(np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1), 1e-12)
    redundancy_report = {k: {f: _stats(v[f]) for f in ("arm", "global")} for k, v in redundancy.items()}
    redundancy_report["cross_sequence_random_pairs"] = {"arm": _stats(cos(arm, xa, xb)), "global": _stats(cos(glob, xa, xb))}
    redundancy_report["frame_index_gap_between_consecutive_train_rows"] = _stats(frame_gaps)

    # --- 4-7. S0 / S1 training ---
    models, training, counts = {}, {}, {}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name in SAMPLERS:
        sampler = EpochSampler(name, train_pos, groups, CONFIG["seed"])
        models[name], training[name], counts[name] = train(sampler, visual, geometry, targets, masks, train_pos, val_pos,
                                                           sequence_ids, device)
        training[name]["contribution"] = contribution_summary(counts[name], groups)
        torch.save({"sampler": name, "config": CONFIG, "cache_identity": manifest["identity"],
                    "selection": training[name]["selection"], "state_dict": models[name].state_dict()},
                   args.out_dir / f"{name}.pt")
    if training["S0_FRAME_UNIFORM"]["draws"] != training["S1_SEQUENCE_BALANCED"]["draws"] or \
            training["S0_FRAME_UNIFORM"]["steps"] != training["S1_SEQUENCE_BALANCED"]["steps"]:
        raise SystemExit("S0/S1 optimization budget differs")
    w69_curve = w69_report["training"]["C1_QWEN_VISION"]["curve"]
    reproduction = {"max_abs_curve_difference": float(max(
        max(abs(a["train_mae"] - b["train_mae"]), abs(a["validation_mae"] - b["validation_mae"]))
        for a, b in zip(training["S0_FRAME_UNIFORM"]["curve"], w69_curve))),
        "w69_selection": w69_report["training"]["C1_QWEN_VISION"]["selection"],
        "s0_selection": training["S0_FRAME_UNIFORM"]["selection"]}

    # --- 10-15. Held-out evaluation on the Worklog 69 rows ---
    rows = json.loads((args.w69_dir / "rows_heldout.json").read_text())
    positions = np.array([r["position"] for r in rows])
    preds = {"s0": predict(models["S0_FRAME_UNIFORM"], visual, geometry, positions, device),
             "s1": predict(models["S1_SEQUENCE_BALANCED"], visual, geometry, positions, device)}
    sizes_img = np.asarray([bank.samples[p].image_size for p in positions], float)
    planes, _ = observed_plane(bank.arrays["input_2d"][positions], sizes_img)
    gt_vec = segment_vectors(bank.arrays["target_3d"][positions])
    for i, r in enumerate(rows):
        for k, seg in enumerate(SEGMENT_NAMES):
            g = gt_vec[seg][i]
            gu = (g / np.linalg.norm(g))[None]
            for mname in ("s0", "s1"):
                f = float(preds[mname][i, k])
                r[f"{seg}_f_{mname}"] = f
                r[f"{seg}_{mname}_raw_plane"] = float(angle_degrees(reconstruct_direction(np.array([f]), planes[seg][i:i + 1]), gu)[0])
    models_eval = ("s0", "s1", "c1", "h0", "d1")

    def continuous(subset):
        out = {"n": len(subset)}
        for seg in SEGMENT_NAMES:
            fg = np.array([r[f"{seg}_f_gt"] for r in subset])
            f = {mn: np.array([r[f"{seg}_f_{mn}"] for r in subset]) for mn in models_eval}
            err = {mn: {"abs_f": np.abs(f[mn] - fg), "elevation": np.abs(elevation_degrees(f[mn]) - elevation_degrees(fg)),
                        "full": np.array([r[f"{seg}_{mn}_raw_plane"] for r in subset])} for mn in models_eval}
            block = {mn: {k: _stats(v) for k, v in err[mn].items()} for mn in models_eval}
            block["s1_minus_s0"] = {k: {**_stats(err["s1"][k] - err["s0"][k]),
                                        "improved_share": float(np.mean(err["s1"][k] < err["s0"][k]))} for k in err["s1"]}
            block["s0_minus_w69_c1_abs_f"] = _stats(err["s0"]["abs_f"] - err["c1"]["abs_f"])
            block["by_abs_gt_f"] = {f"{lo:.3f}-{min(hi, 1):.3f}": {
                "n": int(((np.abs(fg) >= lo) & (np.abs(fg) < hi)).sum()),
                **{mn: float(err[mn]["abs_f"][(np.abs(fg) >= lo) & (np.abs(fg) < hi)].mean())
                   for mn in models_eval if ((np.abs(fg) >= lo) & (np.abs(fg) < hi)).any()}} for lo, hi in F_BINS}
            block["sign_magnitude"] = sign_magnitude({mn: f[mn] for mn in ("s0", "s1")}, fg, STABLE)
            out[seg] = block
        return out

    test_rows = [r for r in rows if bank.samples[r["position"]].split == "test"]
    val_rows = [r for r in rows if bank.samples[r["position"]].split == "validation"]
    scopes = {"test_all": test_rows, "validation_all": val_rows, "heldout_all": rows,
              "validation_clean_dancing_hug": [r for r in val_rows if any(c in r["sequence_id"] for c in CLEAN)],
              "validation_crosscountry_only": [r for r in val_rows if "crosscountry" in r["sequence_id"]],
              "all_other_heldout": [r for r in rows if not any(c in r["sequence_id"] for c in CLEAN) and "crosscountry" not in r["sequence_id"]]}

    per_sequence = {}
    for scope_name, scope_rows in (("test", test_rows), ("heldout", rows)):
        by_seq = defaultdict(list)
        for r in scope_rows:
            by_seq[r["sequence_id"]].append(r)
        per_sequence[scope_name] = {}
        for seg in SEGMENT_NAMES:
            gains = {}
            for sid, group in by_seq.items():
                fg = np.array([r[f"{seg}_f_gt"] for r in group])
                gains[sid] = float((np.abs(np.array([r[f"{seg}_f_s0"] for r in group]) - fg)
                                    - np.abs(np.array([r[f"{seg}_f_s1"] for r in group]) - fg)).sum())
            total = sum(gains.values())
            positive = sum(v for v in gains.values() if v > 0)
            ranked = sorted(gains.items(), key=lambda kv: kv[1])
            per_sequence[scope_name][seg] = {
                "sequences": len(gains), "improved": sum(v > 0 for v in gains.values()), "net_abs_f_gain": total,
                "largest_sequence": ranked[-1][0],
                "largest_share_of_net_gain": ranked[-1][1] / total if total > 0 else None,
                "largest_share_of_positive_gain": ranked[-1][1] / positive if positive > 0 else None,
                "worst_regressions": [{"sequence_id": s, "abs_f_sum_change": -v} for s, v in ranked[:3]]}

    report = {"schema": "animcv_visual_supervision_diversity_v1", "config": CONFIG,
              "cache_identity": manifest["identity"], "readout_npz_sha256": manifest["readout_npz_sha256"],
              "composition": composition, "redundancy": redundancy_report,
              "training": training, "s0_reproduces_worklog69": reproduction,
              "continuous": {k: continuous(v) for k, v in scopes.items()}, "per_sequence": per_sequence}
    (args.out_dir / "rows_heldout.json").write_text(json.dumps(rows))
    path = args.out_dir / "report.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps({"sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "reproduction": reproduction,
                      "training": {k: {kk: v[kk] for kk in ("selection", "final", "draws", "eligible_draws", "steps")}
                                   for k, v in training.items()}}, indent=1))


if __name__ == "__main__":
    main()
