#!/usr/bin/env python3
"""Recompute evaluation reports from stored predictions, without retraining.

The per-chain hinge accounting was added after the earlier sign candidates ran,
so their reports lack it. Their predictions are on disk and the evaluator is a
pure function of (bank, positions, prediction), so the missing metrics can be
recovered exactly. Nothing is retrained and no historical report is overwritten:
output goes to a separate directory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from common.serialization import write_json
from framepose.bank import load_bank
from framepose.evaluate import evaluate_predictions


def main() -> int:
    parser = argparse.ArgumentParser(description="Recompute reports from stored predictions")
    parser.add_argument("--bank", required=True, type=Path)
    parser.add_argument("--candidate", action="append", required=True,
                        help="NAME=DIRECTORY containing prediction_<split>.npy (repeatable)")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--split", action="append", default=None)
    args = parser.parse_args()

    bank = load_bank(args.bank)
    splits = args.split or ["validation", "test"]
    args.out.mkdir(parents=True, exist_ok=True)
    summary: dict[str, dict] = {}

    for entry in args.candidate:
        name, directory = entry.split("=", 1)
        summary[name] = {}
        for split in splits:
            path = Path(directory) / f"prediction_{split}.npy"
            if not path.is_file():
                continue
            positions = bank.indices(split)
            prediction = np.load(path)
            report = evaluate_predictions(bank, positions, prediction, candidate=name)
            write_json(args.out / f"evaluation_{name}_{split}.json", report)
            summary[name][split] = report["aggregate"]

    write_json(args.out / "recomputed_summary.json", {
        "schema": "animcv_frame_pose_recomputed_evaluation_v1",
        "bank_content_digest": bank.content_digest(),
        "note": ("recomputed from stored predictions with the current evaluator; the original "
                 "reports are not modified"),
        "candidates": summary,
    })
    print(json.dumps({name: {split: {
        "mpjpe_mm": value["mpjpe_mm"]["mean"],
        "hinge_flip_rate": value["hinge_flip_rate"]["mean"],
        "elbow_flip_rate": value["elbow_flip_rate"]["mean"],
        "knee_flip_rate": value["knee_flip_rate"]["mean"],
    } for split, value in splits_.items()} for name, splits_ in summary.items()},
        indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
