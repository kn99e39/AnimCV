#!/usr/bin/env python3
"""Materialise frozen-backbone patch features for a frame bank.

The visual backbones are frozen for the whole controlled comparison, so their
tokens are a pure function of the frame, the crop contract and the weights.
Caching them once makes F1/F2 training cost the same as F0 and makes replay
exact.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from common.serialization import write_json
from framepose.bank import load_bank
from framepose.features import build_feature_cache


def main() -> int:
    parser = argparse.ArgumentParser(description="Cache frozen visual backbone features for a frame bank")
    parser.add_argument("--bank", required=True, type=Path)
    parser.add_argument("--backbone", required=True, help="vit_in21k | siglip")
    parser.add_argument("--image-root", action="append", default=[], help="KEY=PATH (repeatable)")
    parser.add_argument("--out", required=True, type=Path, help="Feature cache root directory")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()

    roots = {}
    for value in args.image_root:
        key, path = value.split("=", 1)
        roots[key] = path

    bank = load_bank(args.bank)
    metadata = build_feature_cache(bank, args.backbone, image_roots=roots, out_root=args.out,
                                   device=args.device, batch_size=args.batch_size, workers=args.workers)
    write_json(Path(args.out) / args.backbone / "cache_report.json", metadata)
    print(json.dumps({key: value for key, value in metadata.items() if key != "crop_contract"},
                     indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
