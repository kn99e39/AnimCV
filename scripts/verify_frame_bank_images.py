#!/usr/bin/env python3
"""Verify a frame bank's imagery against its declared contract.

Checks, on a deterministic sample of the bank, that every referenced image
exists, that its real pixel dimensions match the `image_size` the annotation
implied, and that the derived crop actually contains the observed joints. A
size mismatch would silently place every crop and every geometry token in the
wrong frame, so this is checked rather than assumed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from common.serialization import write_json
from framepose.bank import load_bank
from framepose.crops import crop_box


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify frame bank image references and crop geometry")
    parser.add_argument("--bank", required=True, type=Path)
    parser.add_argument("--image-root", action="append", required=True, help="KEY=PATH (repeatable)")
    parser.add_argument("--sample", type=int, default=500, help="0 checks every sample")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    from PIL import Image

    roots = dict(value.split("=", 1) for value in args.image_root)
    bank = load_bank(args.bank)
    positions = np.arange(len(bank))
    if args.sample and args.sample < len(bank):
        positions = np.sort(np.random.default_rng(args.seed).choice(len(bank), args.sample, replace=False))

    missing, mismatched, uncovered = [], [], []
    for position in positions:
        sample = bank.samples[int(position)]
        if sample.image_reference is None:
            missing.append(sample.sample_id)
            continue
        path = sample.image_reference.resolve(roots)
        if not path.is_file():
            missing.append(sample.sample_id)
            continue
        with Image.open(path) as handle:
            size = handle.size
        if tuple(size) != tuple(sample.image_size):
            mismatched.append({"sample_id": sample.sample_id, "declared": list(sample.image_size),
                               "actual": list(size)})
            continue
        observation = bank.arrays["input_2d"][position]
        valid = bank.arrays["input_valid"][position]
        box = crop_box(observation, valid, sample.image_size)
        pixels = observation[:, :2] * np.asarray(sample.image_size, dtype=np.float64)
        inside = ((pixels[valid] >= (box.x, box.y)).all(axis=1) &
                  (pixels[valid] <= (box.x + box.side, box.y + box.side)).all(axis=1))
        if valid.any() and not inside.all():
            uncovered.append({"sample_id": sample.sample_id,
                              "outside_joint_count": int((~inside).sum())})

    report = {
        "schema": "animcv_frame_bank_image_verification_v1",
        "bank": str(args.bank),
        "bank_content_digest": bank.content_digest(),
        "checked_samples": int(len(positions)),
        "missing_images": missing[:20],
        "missing_image_count": len(missing),
        "image_size_mismatches": mismatched[:20],
        "image_size_mismatch_count": len(mismatched),
        "crops_not_covering_observed_joints": uncovered[:20],
        "crops_not_covering_observed_joints_count": len(uncovered),
        "passed": not missing and not mismatched and not uncovered,
    }
    if args.out:
        write_json(args.out, report)
    print(json.dumps({key: value for key, value in report.items()
                      if not key.endswith(("mismatches", "images", "joints"))}, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
