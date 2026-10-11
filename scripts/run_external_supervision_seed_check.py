#!/usr/bin/env python3
"""DIAGNOSTIC (Worklog 73, secondary): seed robustness of E1 - E0.

Repeats the Worklog 73 E0/E1 pair with two further fixed seeds (1338, 1339),
changing only the seed of initialization and of both samplers.  Everything
else (arrays, caches, graph, budget, selection on 3DPW validation, evaluation
rows) is the main experiment's.  Not used for the architecture decision; it
only bounds how much of E1 - E0 a different seed could produce.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from framepose.arm_depth_probe import SEGMENT_NAMES, forward_targets
from framepose.bank import load_bank
from framepose.bedlam_supervision import ExternalBank
from framepose.continuous_vision_depth import candidate_visual, predict
from framepose.learned_vision_sensor import pair_geometry
from framepose.train import geometry_tensor
from framepose.visual_supervision_sampling import EpochSampler, train_sequence_groups
from run_external_visual_supervision_transfer import SourceBalancedSampler, load_cache
from run_visual_supervision_diversity import CONFIG, train

SEEDS = (1338, 1339)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("bank", "cache_dir", "external_train", "external_train_cache", "external_val", "external_val_cache",
                 "run_dir", "out"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    args = parser.parse_args()
    import torch

    device = torch.device("cuda")
    bank = load_bank(args.bank)
    ext_tr, ext_va = ExternalBank.load(args.external_train), ExternalBank.load(args.external_val)
    r3, ident3 = load_cache(args.cache_dir, bank.content_digest(), [s.sample_id for s in bank.samples], None)
    rtr, _ = load_cache(args.external_train_cache, ext_tr.content_digest(), [s.sample_id for s in ext_tr.samples], ident3)
    rva, _ = load_cache(args.external_val_cache, ext_va.content_digest(), [s.sample_id for s in ext_va.samples], ident3)
    n3, ntr = len(bank), len(ext_tr)
    visual = candidate_visual(np.concatenate([r3, rtr, rva]), "C1_QWEN_VISION")
    geometry = np.concatenate([pair_geometry(geometry_tensor(b)) for b in (bank, ext_tr, ext_va)])
    targets, masks = forward_targets(np.concatenate([b.arrays["target_3d"] for b in (bank, ext_tr, ext_va)]),
                                     np.concatenate([b.arrays["target_valid"] for b in (bank, ext_tr, ext_va)]))
    eligible = masks.any(axis=1)
    sequence_ids = [s.sequence_id for b in (bank, ext_tr, ext_va) for s in b.samples]
    train3, val3 = bank.indices("train"), bank.indices("validation")
    ext_train_pos = n3 + np.arange(ntr)
    groups = train_sequence_groups([s.sequence_id for s in bank.samples], train3, eligible[:n3])
    rows = json.loads((args.run_dir / "rows_heldout.json").read_text())
    positions = np.array([r["position"] for r in rows])
    fg = np.array([[r[f"{seg}_f_gt"] for seg in SEGMENT_NAMES] for r in rows])
    split = np.array([bank.samples[p].split for p in positions])

    def summary(pred_e0, pred_e1):
        out = {}
        for name in ("validation", "test"):
            sel = split == name
            for k, seg in enumerate(SEGMENT_NAMES):
                e0, e1 = np.abs(pred_e0[sel, k] - fg[sel, k]), np.abs(pred_e1[sel, k] - fg[sel, k])
                out[f"{name}_{seg}"] = {"e0": float(e0.mean()), "e1": float(e1.mean()), "e1_minus_e0": float((e1 - e0).mean()),
                                        "improved_share": float(np.mean(e1 < e0))}
        return out

    report = {"schema": "animcv_w73_seed_check_v1", "seeds": {}}
    main_rows = {"e0": np.array([[r[f"{seg}_f_e0"] for seg in SEGMENT_NAMES] for r in rows]),
                 "e1": np.array([[r[f"{seg}_f_e1"] for seg in SEGMENT_NAMES] for r in rows])}
    report["seeds"]["1337 (main run)"] = summary(main_rows["e0"], main_rows["e1"])
    for seed in SEEDS:
        CONFIG["seed"] = seed
        e0_sampler = EpochSampler("S0_FRAME_UNIFORM", train3, groups, seed)
        e1_sampler = SourceBalancedSampler(train3[eligible[train3]], ext_train_pos[eligible[ext_train_pos]],
                                           e0_sampler.draws_per_epoch, seed)
        e0, i0, _ = train(e0_sampler, visual, geometry, targets, masks, train3, val3, sequence_ids, device)
        e1, i1, _ = train(e1_sampler, visual, geometry, targets, masks, train3, val3, sequence_ids, device)
        report["seeds"][str(seed)] = {"selection": {"E0": i0["selection"], "E1": i1["selection"]},
                                      **summary(predict(e0, visual, geometry, positions, device),
                                                predict(e1, visual, geometry, positions, device))}
        print(json.dumps({"seed": seed, "E0_val": i0["selection"]["validation_mae"],
                          "E1_val": i1["selection"]["validation_mae"]}), flush=True)
    args.out.write_text(json.dumps(report, indent=1))
    for seed, r in report["seeds"].items():
        print(seed, {k: round(v["e1_minus_e0"], 4) for k, v in r.items() if k != "selection"})


if __name__ == "__main__":
    main()
