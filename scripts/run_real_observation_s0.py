#!/usr/bin/env python3
"""EXPERIMENT (Worklog 100): frozen S0 on the benchmark observation (B) and on AnimCV RTMDet/RTMPose observations.

Runs in the original FramePose environment (docker ``animcv-framepose:cuda118``).  The 3DPW frame bank is loaded unchanged;
for an observation condition only ``input_2d``/``input_valid`` of the VALIDATION and TEST rows are replaced (in memory) by
the arrays in ``--observations`` (built by ``eval_real_observation_estimator.py build-observations``).  Everything else is
the historical S0 inference contract (``run_sign_experiments._evaluate``): checkpoint ``S0_neutral_sign``, crop-normalised
geometry (``geometry_tensor``), neutral signs, ``predict`` batch 512, AMP off, rows in ``bank.indices(split)`` order.
Condition B must reproduce the pinned historical predictions bit for bit, or the run stops.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from framepose.bank import load_bank
from framepose.signs import neutral_sign_states
from framepose.train import geometry_tensor, load_checkpoint, predict

S0_CHECKPOINT_SHA = "6c12fea9f84ea0493fada8b9acc96cd5c70fdd5d9114603e954f9412f4247b44"
S0_PRED_SHA = {"validation": "402d70a2bed79ae31919ff86905e19c8972108ea988d6cfa5deac514562132b9",
               "test": "4357c7bdb46d81e7712269b08c45f39452d7a46e697a0c26fdb70d65fe1f4996"}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for n in ("bank", "checkpoint", "observations", "out_dir"):
        ap.add_argument("--" + n.replace("_", "-"), type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    import torch

    if sha(args.checkpoint) != S0_CHECKPOINT_SHA:
        raise SystemExit("STOP: S0 checkpoint identity differs")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    bank = load_bank(args.bank)
    obs = np.load(args.observations)
    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    model, ck = load_checkpoint(args.checkpoint, device=str(device))
    if ck.get("candidate", {}).get("name") != "S0_neutral_sign":
        raise SystemExit("STOP: checkpoint is not S0_neutral_sign")
    signs = neutral_sign_states(len(bank))
    original = {k: bank.arrays[k].copy() for k in ("input_2d", "input_valid")}
    report = {"bank_content_digest": bank.content_digest(), "checkpoint_sha256": S0_CHECKPOINT_SHA, "device": str(device),
              "observations_sha256": sha(args.observations), "conditions": {}}
    conditions = ["B"] + [k[len("input_2d_"):] for k in obs.files if k.startswith("input_2d_")]
    positions_obs = obs["positions"]
    for cond in conditions:
        bank.arrays["input_2d"], bank.arrays["input_valid"] = original["input_2d"].copy(), original["input_valid"].copy()
        if cond != "B":
            bank.arrays["input_2d"][positions_obs] = obs[f"input_2d_{cond}"]
            bank.arrays["input_valid"][positions_obs] = obs[f"input_valid_{cond}"]
        geometry = geometry_tensor(bank)
        report["conditions"][cond] = {}
        for split in ("validation", "test"):
            pos = bank.indices(split)
            pred = predict(model, torch, geometry, None, pos, device, signs=signs).astype(np.float32)
            path = args.out_dir / f"prediction_{cond}_{split}.npy"
            np.save(path, pred)
            report["conditions"][cond][split] = {"rows": int(len(pos)), "sha256": sha(path)}
            if cond == "B" and sha(path) != S0_PRED_SHA[split]:
                report["conditions"][cond][split]["pinned_mismatch"] = True
                (args.out_dir / "s0_report.json").write_text(json.dumps(report, indent=1))
                raise SystemExit(f"STOP: condition B does not reproduce the pinned S0 {split} predictions")
    bank.arrays["input_2d"], bank.arrays["input_valid"] = original["input_2d"], original["input_valid"]
    (args.out_dir / "s0_report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
