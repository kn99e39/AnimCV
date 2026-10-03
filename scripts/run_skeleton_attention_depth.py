#!/usr/bin/env python3
"""DIAGNOSTIC (Worklog 70): A0_UNBIASED_ATTENTION vs A1_SKELETON_BIASED_ATTENTION over the frozen Qwen3-VL grid.

Same target / masks / splits / pair geometry / readouts / training contract as
Worklog 69.  Only the deterministic attention-logit bias differs between A0
(zero) and A1 (-distance to the observed 2D segment).  Worklog 69 C0/C1, frozen
H0 and Worklog 65 D1 are read-only references from the Worklog 69 rows.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import threading
import queue
from collections import defaultdict
from pathlib import Path

import numpy as np

from framepose.arm_depth_probe import (
    SEGMENT_NAMES, angle_degrees, elevation_degrees, forward_targets, observed_plane, reconstruct_direction,
    segment_vectors,
)
from framepose.bank import load_bank
from framepose.continuous_vision_depth import sign_magnitude
from framepose.skeleton_attention_depth import CANDIDATES, build_attention_model, candidate_bias, skeleton_bias
from framepose.train import geometry_tensor
from framepose.vlm_depth_advisor import shuffled_donors

CONFIG = {"optimizer": "AdamW", "learning_rate": 3e-4, "minimum_learning_rate": 1e-5, "weight_decay": 1e-4,
          "schedule": "cosine per step", "epochs": 200, "batch_size": 256, "seed": 1337, "evaluate_every": 10,
          "loss": "masked MAE on tanh-bounded f, unweighted", "selection": "validation masked mean |f error|",
          "mixed_precision": False}
STABLE = math.sin(math.radians(10))
CLEAN = ("courtyard_dancing_00", "courtyard_hug_00")
F_BINS = ((0.0, STABLE), (STABLE, 0.5), (0.5, 1.0001))
MODELS = ("a0", "a1", "a1_shuffled", "c0", "c1", "h0", "d1")


def _stats(v):
    v = np.asarray([x for x in v if x is not None and np.isfinite(x)], float)
    return {"n": int(len(v)), "mean": float(v.mean()), "p50": float(np.percentile(v, 50)),
            "p95": float(np.percentile(v, 95))} if len(v) else {"n": 0}


def batches(grid, readout, positions, size, visual_positions=None):
    """Background-prefetched (grid, readout) host batches in the given order."""
    q: queue.Queue = queue.Queue(maxsize=3)
    vp = positions if visual_positions is None else visual_positions

    def work():
        for s in range(0, len(positions), size):
            idx = np.asarray(vp[s:s + size])
            order = np.argsort(idx, kind="stable")
            rows = np.empty((len(idx),) + grid.shape[1:], dtype=grid.dtype)
            rows[order] = grid[idx[order]]
            q.put((s, rows, readout[idx]))
        q.put(None)
    threading.Thread(target=work, daemon=True).start()
    while (item := q.get()) is not None:
        yield item


def run_model(model, grid, readout, geometry, bias, positions, device, visual_positions=None, attention=False):
    import torch

    out, att = [], []
    model.eval()
    with torch.no_grad():
        for s, g_rows, r_rows in batches(grid, readout, positions, 512, visual_positions):
            p = np.asarray(positions[s:s + 512])
            res = model(torch.as_tensor(g_rows, device=device), torch.as_tensor(r_rows, device=device),
                        torch.as_tensor(geometry[p], device=device), torch.as_tensor(bias[p], device=device),
                        return_attention=attention)
            if attention:
                out.append(res[0].cpu().numpy())
                att.append(res[1].cpu().numpy())
            else:
                out.append(res.cpu().numpy())
    f = np.concatenate(out).astype(np.float64)
    return (f, np.concatenate(att)) if attention else f


def train(grid, readout, geometry, bias, targets, masks, train_pos, val_pos, device):
    import torch

    torch.manual_seed(CONFIG["seed"])
    model = build_attention_model(grid.shape[-1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"], weight_decay=CONFIG["weight_decay"])
    t = torch.as_tensor(targets.astype(np.float32), device=device)
    m = torch.as_tensor(masks.astype(np.float32), device=device)

    def mae(positions):
        f = run_model(model, grid, readout, geometry, bias, positions, device)
        w = masks[positions]
        return float((np.abs(f - targets[positions]) * w).sum() / w.sum())

    gen = torch.Generator(device="cpu").manual_seed(CONFIG["seed"])
    steps = math.ceil(len(train_pos) / CONFIG["batch_size"]) * CONFIG["epochs"]
    step, best, best_state, curve = 0, {"epoch": None, "validation_mae": None}, None, []
    for epoch in range(CONFIG["epochs"]):
        model.train()
        order = train_pos[torch.randperm(len(train_pos), generator=gen).numpy()]
        for s, g_rows, r_rows in batches(grid, readout, order, CONFIG["batch_size"]):
            p = order[s:s + CONFIG["batch_size"]]
            index = torch.as_tensor(p, device=device)
            progress = min(step / max(steps - 1, 1), 1.0)
            lr = CONFIG["minimum_learning_rate"] + (CONFIG["learning_rate"] - CONFIG["minimum_learning_rate"]) * 0.5 * (1 + math.cos(math.pi * progress))
            for group in opt.param_groups:
                group["lr"] = lr
            opt.zero_grad(set_to_none=True)
            f = model(torch.as_tensor(g_rows, device=device), torch.as_tensor(r_rows, device=device),
                      torch.as_tensor(geometry[p], device=device), torch.as_tensor(bias[p], device=device))
            w = m[index]
            loss = (torch.abs(f - t[index]) * w).sum() / w.sum().clamp_min(1.0)
            loss.backward()
            opt.step()
            step += 1
        if epoch == CONFIG["epochs"] - 1 or (epoch + 1) % CONFIG["evaluate_every"] == 0:
            record = {"epoch": epoch, "train_mae": mae(train_pos), "validation_mae": mae(val_pos)}
            curve.append(record)
            print(json.dumps(record), flush=True)
            if best["epoch"] is None or record["validation_mae"] < best["validation_mae"]:
                best = dict(record)
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    final = curve[-1]
    model.load_state_dict(best_state)
    model.eval()
    return model, {"selection": best, "final": final, "curve": curve,
                   "parameters": int(sum(p.numel() for p in model.parameters()))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("bank", "cache_dir", "w68_dir", "w69_dir", "out_dir"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    import torch

    bank = load_bank(args.bank)
    manifest = json.loads((args.cache_dir / "manifest.json").read_text())
    identity = manifest["identity"]
    readout_path = args.cache_dir / "readout.npz"
    if (identity["bank_content_digest"] != bank.content_digest()
            or identity["sample_ids_sha256"] != hashlib.sha256("\n".join(s.sample_id for s in bank.samples).encode()).hexdigest()
            or identity["model_fingerprint"] != "a0d72ded575eaa4460dad11dd1e313bc69f0403b2c97c3c1ba234af086952904"
            or manifest["readout_npz_sha256"] != hashlib.sha256(readout_path.read_bytes()).hexdigest()):
        raise SystemExit("Worklog 68 cache identity mismatch")
    grid_path = args.cache_dir / "token_grid_fp16.npy"
    digest = hashlib.sha256()
    with grid_path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(64 * 1024 * 1024), b""):
            digest.update(chunk)
    grid_sha = digest.hexdigest()
    grid = np.load(grid_path, mmap_mode="r")
    if grid.shape != (len(bank), 196, 4096) or grid.dtype != np.float16:
        raise SystemExit(f"unexpected grid {grid.shape} {grid.dtype}")
    cache = np.load(readout_path)
    readout, boxes = cache["readout"], cache["crop_box_xy_side"]

    from framepose.learned_vision_sensor import pair_geometry  # read-only reuse in a script
    geometry = pair_geometry(geometry_tensor(bank))
    targets, masks = forward_targets(bank.arrays["target_3d"], bank.arrays["target_valid"])
    raw_bias = np.stack([skeleton_bias(bank.arrays["input_2d"][p], bank.arrays["input_valid"][p],
                                       bank.samples[p].image_size, boxes[p]) for p in range(len(bank))])
    train_pos, val_pos = bank.indices("train"), bank.indices("validation")
    device = torch.device(args.device)

    biases = {c: candidate_bias(raw_bias, c) for c in CANDIDATES}
    if np.any(biases["A0_UNBIASED_ATTENTION"] != 0) or not np.array_equal(biases["A1_SKELETON_BIASED_ATTENTION"], raw_bias):
        raise SystemExit("A0/A1 bias contract violated")
    models, training = {}, {}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for c in CANDIDATES:
        print(json.dumps({"training": c}), flush=True)
        models[c], training[c] = train(grid, readout, geometry, biases[c], targets, masks, train_pos, val_pos, device)
        torch.save({"candidate": c, "config": CONFIG, "cache_identity": identity, "token_grid_sha256": grid_sha,
                    "selection": training[c]["selection"], "state_dict": models[c].state_dict()}, args.out_dir / f"{c}.pt")

    rows = json.loads((args.w69_dir / "rows_heldout.json").read_text())
    test_ids = {p for p in (r["position"] for r in rows) if bank.samples[p].split == "test"}
    positions = np.array([r["position"] for r in rows])
    donors = positions[np.array(shuffled_donors([(r["sequence_id"], r["frame_index"]) for r in rows]))]
    w68 = {r["position"]: r["donor_sample_id"] for r in json.loads((args.w68_dir / "rows_heldout.json").read_text())}
    if any(w68[p] != bank.samples[d].sample_id for p, d in zip(positions, donors)):
        raise SystemExit("shuffled donors differ from Worklog 68")
    preds = {"a0": run_model(models["A0_UNBIASED_ATTENTION"], grid, readout, geometry, biases["A0_UNBIASED_ATTENTION"], positions, device),
             "a1": run_model(models["A1_SKELETON_BIASED_ATTENTION"], grid, readout, geometry, biases["A1_SKELETON_BIASED_ATTENTION"], positions, device),
             "a1_shuffled": run_model(models["A1_SKELETON_BIASED_ATTENTION"], grid, readout, geometry,
                                      biases["A1_SKELETON_BIASED_ATTENTION"], positions, device, visual_positions=donors)}
    sizes = np.asarray([bank.samples[p].image_size for p in positions], float)
    planes, _ = observed_plane(bank.arrays["input_2d"][positions], sizes)
    gt_vec = segment_vectors(bank.arrays["target_3d"][positions])
    for i, r in enumerate(rows):
        for k, seg in enumerate(SEGMENT_NAMES):
            g = gt_vec[seg][i]
            gu = (g / np.linalg.norm(g))[None]
            for m in ("a0", "a1", "a1_shuffled"):
                f = float(preds[m][i, k])
                r[f"{seg}_f_{m}"] = f
                r[f"{seg}_{m}_raw_plane"] = float(angle_degrees(reconstruct_direction(np.array([f]), planes[seg][i:i + 1]), gu)[0])

    def continuous(subset):
        out = {"n": len(subset)}
        for seg in SEGMENT_NAMES:
            fg = np.array([r[f"{seg}_f_gt"] for r in subset])
            f = {m: np.array([r[f"{seg}_f_{m}"] for r in subset]) for m in MODELS}
            err = {m: {"abs_f": np.abs(f[m] - fg), "elevation": np.abs(elevation_degrees(f[m]) - elevation_degrees(fg)),
                       "full": np.array([r[f"{seg}_{m}_raw_plane"] for r in subset])} for m in MODELS}
            block = {m: {k: _stats(v) for k, v in err[m].items()} for m in MODELS}
            for a, b in (("a1", "a0"), ("a1", "c1"), ("a0", "c1"), ("a1", "a1_shuffled"), ("a1", "h0"), ("a1", "d1")):
                block[f"{a}_minus_{b}"] = {k: {**_stats(err[a][k] - err[b][k]),
                                               "improved_share": float(np.mean(err[a][k] < err[b][k]))}
                                           for k in ("abs_f", "elevation", "full")}
            block["by_abs_gt_f"] = {}
            for lo, hi in F_BINS:
                sel = (np.abs(fg) >= lo) & (np.abs(fg) < hi)
                block["by_abs_gt_f"][f"{lo:.3f}-{min(hi, 1):.3f}"] = {"n": int(sel.sum()),
                                                                     **{m: float(err[m]["abs_f"][sel].mean()) for m in MODELS if sel.any()}}
            block["sign_magnitude"] = sign_magnitude({m: f[m] for m in ("a0", "a1", "c1")}, fg, STABLE)
            out[seg] = block
        return out

    scopes = {"test_all": [r for r in rows if r["position"] in test_ids],
              "validation_all": [r for r in rows if r["position"] not in test_ids],
              "heldout_all": rows,
              "validation_clean_dancing_hug": [r for r in rows if r["position"] not in test_ids and any(c in r["sequence_id"] for c in CLEAN)],
              "validation_crosscountry_only": [r for r in rows if "crosscountry" in r["sequence_id"]],
              "all_other_heldout": [r for r in rows if not any(c in r["sequence_id"] for c in CLEAN) and "crosscountry" not in r["sequence_id"]]}

    per_sequence = {}
    groups = defaultdict(list)
    for r in rows:
        groups[r["sequence_id"]].append(r)
    for seg in SEGMENT_NAMES:
        per_sequence[seg] = {}
        for a, b in (("a1", "a0"), ("a1", "c1")):
            gains = {}
            for sid, group in groups.items():
                fg = np.array([r[f"{seg}_f_gt"] for r in group])
                gains[sid] = float((np.abs(np.array([r[f"{seg}_f_{b}"] for r in group]) - fg)
                                    - np.abs(np.array([r[f"{seg}_f_{a}"] for r in group]) - fg)).sum())
            total = sum(gains.values())
            top = max(gains.items(), key=lambda kv: kv[1])
            positive = sum(v for v in gains.values() if v > 0)
            per_sequence[seg][f"{a}_vs_{b}"] = {"sequences": len(gains), "improved": sum(v > 0 for v in gains.values()),
                                                "net_total_abs_f_gain": total, "largest_sequence": top[0],
                                                "largest_single_sequence_share_of_net_gain": top[1] / total if total > 0 else None,
                                                "largest_single_sequence_share_of_positive_gain": top[1] / positive if positive > 0 else None}

    owner_src = json.loads((args.w69_dir / "report.json").read_text())["owner_cases"]
    by_key = {(r["sequence_id"], r["frame_index"]): r for r in rows}
    owner, attention_export = {}, {}
    for label, c in owner_src.items():
        r = by_key.get((c["sequence_id"], c["frame_index"]))
        owner[label] = {"sequence_id": c["sequence_id"], "frame_index": c["frame_index"], "excluded": r is None,
                        **({seg: {"gt": r[f"{seg}_f_gt"], **{m: r[f"{seg}_f_{m}"] for m in ("c1", "a0", "a1", "a1_shuffled", "h0", "d1")},
                                  "full_c1_a0_a1_h0_d1": [r[f"{seg}_{m}_raw_plane"] for m in ("c1", "a0", "a1", "h0", "d1")]}
                            for seg in SEGMENT_NAMES} if r else {})}
        if r is not None:
            p = np.array([r["position"]])
            for c_name, key in (("A0_UNBIASED_ATTENTION", "a0"), ("A1_SKELETON_BIASED_ATTENTION", "a1")):
                _, att = run_model(models[c_name], grid, readout, geometry, biases[c_name], p, device, attention=True)
                attention_export[f"{r['position']}_{key}"] = att[0]
            attention_export[f"{r['position']}_bias"] = raw_bias[r["position"]]
    np.savez(args.out_dir / "owner_attention.npz", **attention_export)

    report = {"schema": "animcv_skeleton_attention_depth_v1", "config": CONFIG, "cache_identity": identity,
              "token_grid_sha256": grid_sha, "readout_npz_sha256": manifest["readout_npz_sha256"],
              "bias": "A1: -euclidean distance (token-grid units) from each token centre to the observed 2D detector segment; A0: 0; invalid-endpoint pairs: 0 (masked)",
              "training": training, "continuous": {k: continuous(v) for k, v in scopes.items()},
              "per_sequence": per_sequence, "owner_cases": owner}
    (args.out_dir / "rows_heldout.json").write_text(json.dumps(rows))
    path = args.out_dir / "report.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps({"sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "token_grid_sha256": grid_sha,
                      "training": {k: {"selection": v["selection"], "final": v["final"], "parameters": v["parameters"]}
                                   for k, v in training.items()}}, indent=1))


if __name__ == "__main__":
    main()
