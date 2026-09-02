#!/usr/bin/env python3
"""Pre-training screening of frame-level loss contracts (no training).

Replays fixed real frame batches through fixed model states and reports raw loss
magnitude, gradient norm, gradient/base ratio, gradient cosine, per-joint
gradient ownership, source contribution, easy/hard-frame association and
numerical stability. No acceptance threshold is encoded.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from common.serialization import write_json
from framepose.bank import load_bank
from framepose.features import load_feature_cache
from framepose.losses import LOSS_CONTRACTS, resolve_contract
from framepose.screening import screen_contracts
from framepose.train import geometry_tensor


def main() -> int:
    parser = argparse.ArgumentParser(description="Screen frame-level loss contracts before training")
    parser.add_argument("--bank", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--contracts", default=",".join(sorted(LOSS_CONTRACTS)))
    parser.add_argument("--backbone", default="none")
    parser.add_argument("--features-root", type=Path, default=None)
    parser.add_argument("--checkpoint", action="append", default=[],
                        help="NAME=PATH of a trained frame-pose checkpoint to also replay (repeatable)")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--batch-count", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    bank = load_bank(args.bank)
    geometry = geometry_tensor(bank)
    features = None
    if args.backbone != "none":
        if args.features_root is None:
            raise ValueError("--features-root is required for a visual backbone")
        cached, _ = load_feature_cache(args.features_root, args.backbone, bank)
        features = np.asarray(cached)

    states = {}
    for value in args.checkpoint:
        import torch
        name, path = value.split("=", 1)
        payload = torch.load(path, map_location=args.device, weights_only=False)
        states[name] = payload["state_dict"]

    contracts = [resolve_contract(name.strip()) for name in args.contracts.split(",") if name.strip()]
    report = screen_contracts(bank, contracts, geometry=geometry, features=features,
                              backbone=args.backbone, device=args.device, seed=args.seed,
                              batch_count=args.batch_count, batch_size=args.batch_size,
                              states=states or None)
    write_json(args.out, report)
    print(json.dumps({state: {name: value["contracts"][name]["gradient_over_base_ratio"]["mean"]
                              for name in value["contracts"]}
                      for state, value in report["states"].items()}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
