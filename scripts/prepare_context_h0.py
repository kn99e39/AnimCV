#!/usr/bin/env python3
"""Materialize split-aligned frozen H0 predictions for context-refiner input.

This is a read-only replay of an existing Frame Pose checkpoint.  It does not
train or modify H0.  The historical ``O_BILATERAL`` lineage uses oracle
SignState as an architecture control; the generated arrays are handed to the
context refiner as frozen inputs and the sign state is never an input to the
new gate or residual model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from framepose.bank import load_bank
from framepose.signs import mask_fields, oracle_sign_states
from framepose.train import geometry_tensor, load_checkpoint, predict, sign_tensor


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--sign-source", choices=("none", "neutral", "oracle"), default="oracle")
    parser.add_argument("--sign-fields", default="shoulder_forward_depth,hip_forward_depth",
                        help="Oracle fields to retain when --sign-source=oracle")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError(f"refusing to overwrite H0 materialization: {args.out}")
    import torch

    bank = load_bank(args.bank)
    geometry = geometry_tensor(bank)
    model, checkpoint = load_checkpoint(args.checkpoint, device=args.device)
    if args.sign_source == "oracle":
        fields = [value for value in args.sign_fields.split(",") if value]
        signs = mask_fields(oracle_sign_states(bank.arrays["target_3d"], bank.arrays["target_valid"]), fields)
    else:
        fields = []
        signs = sign_tensor(bank, args.sign_source)
    args.out.mkdir(parents=True, exist_ok=True)
    entries = {}
    for split in ("train", "validation", "test"):
        positions = bank.indices(split)
        values = predict(model, torch, geometry, None, positions, torch.device(args.device),
                         signs=signs)
        path = args.out / f"prediction_{split}.npy"
        np.save(path, values.astype(np.float32))
        entries[split] = {"path": str(path), "shape": list(values.shape)}
    report = {
        "schema": "animcv_context_h0_materialization_v1",
        "bank_content_digest": bank.content_digest(),
        "bank_path": str(args.bank),
        "checkpoint": str(args.checkpoint),
        "checkpoint_candidate": checkpoint.get("candidate"),
        "sign_source": args.sign_source,
        "sign_fields": fields,
        "sign_is_used_only_for_h0_replay": True,
        "predictions": entries,
    }
    (args.out / "materialization_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
