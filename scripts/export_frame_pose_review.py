#!/usr/bin/env python3
"""Export frame-by-frame review artifacts for fixed review sequences.

For each retained frame the export carries the ground-truth pose and every
candidate's prediction with quantitative context, so a reviewer can inspect
frame 1, frame 2, ... frame N individually. Sequence playback is a convenience;
visual smoothness is never evidence of frame-pose quality.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from common.serialization import read_json, write_json
from framepose.bank import load_bank
from framepose.contract import JOINT_NAMES


def main() -> int:
    parser = argparse.ArgumentParser(description="Export a frame-by-frame candidate review")
    parser.add_argument("--bank", required=True, type=Path)
    parser.add_argument("--experiment-root", required=True, type=Path)
    parser.add_argument("--split", default="test")
    parser.add_argument("--sequence", action="append", default=[],
                        help="sequence_id to export (repeatable); default picks the fixed selection below")
    parser.add_argument("--max-sequences", type=int, default=4)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    bank = load_bank(args.bank)
    positions = bank.indices(args.split)
    if not len(positions):
        raise ValueError(f"bank has no {args.split} split")

    candidates = {}
    for directory in sorted(args.experiment_root.iterdir()):
        prediction_path = directory / f"prediction_{args.split}.npy"
        evaluation_path = directory / f"evaluation_{args.split}.json"
        if prediction_path.is_file() and evaluation_path.is_file():
            evaluation = read_json(evaluation_path)
            candidates[directory.name] = {
                "prediction": np.load(prediction_path),
                "evaluation": evaluation,
                # Indexed once: a linear scan per frame per candidate is
                # quadratic over a full holdout split.
                "metrics": {record["sample_id"]: record for record in evaluation["frames"]},
            }
    if not candidates:
        raise ValueError(f"no candidate predictions found under {args.experiment_root}")

    order = {sample_id: index for index, sample_id in
             enumerate(bank.samples[int(position)].sample_id for position in positions)}
    sequences = args.sequence or _default_sequences(bank, positions, args.max_sequences)

    export = {
        "schema": "animcv_frame_pose_review_v1",
        "split": args.split,
        "joint_names": list(JOINT_NAMES),
        "coordinate_frame": "camera_root_relative (+X right, +Y forward, +Z up)",
        "bank_content_digest": bank.content_digest(),
        "candidates": {name: value["evaluation"]["candidate"] for name, value in candidates.items()},
        "note": ("Per-frame review. Sequence playback is a convenience only; temporal smoothness "
                 "is not evidence of frame-pose quality."),
        "sequences": {},
    }
    for sequence_id in sequences:
        frames = []
        for position in positions:
            sample = bank.samples[int(position)]
            if sample.sequence_id != sequence_id:
                continue
            index = order[sample.sample_id]
            record = {
                "sample_id": sample.sample_id,
                "frame_index": sample.frame_index,
                "timestamp": sample.timestamp,
                "image_reference": sample.image_reference.to_dict() if sample.image_reference else None,
                "strata": {key: value for key, value in sample.strata.items() if isinstance(value, str)},
                "target_valid": bank.arrays["target_valid"][position].tolist(),
                "ground_truth_3d": bank.arrays["target_3d"][position].tolist(),
                "candidates": {},
            }
            for name, value in candidates.items():
                metrics = value["metrics"].get(sample.sample_id, {})
                record["candidates"][name] = {
                    "prediction_3d": value["prediction"][index].tolist(),
                    "metrics": {key: metrics.get(key) for key in
                                ("mpjpe_mm", "pa_mpjpe_mm", "root_yaw_error_degrees",
                                 "shoulder_forward_depth_residual_mm", "hip_forward_depth_residual_mm",
                                 "shoulder_forward_depth_sign_disagreement",
                                 "hip_forward_depth_sign_disagreement")},
                }
            frames.append(record)
        if frames:
            export["sequences"][sequence_id] = {"frame_count": len(frames), "frames": frames}

    write_json(args.out, export)
    print(json.dumps({"sequences": {key: value["frame_count"]
                                    for key, value in export["sequences"].items()}}, indent=2, sort_keys=True))
    return 0


def _default_sequences(bank, positions, limit: int) -> list[str]:
    """Deterministic, content-independent selection: the alphabetically first
    sequences of the split.  Fixed before any candidate was compared."""
    identifiers = sorted({bank.samples[int(position)].sequence_id for position in positions})
    return identifiers[:limit]


if __name__ == "__main__":
    raise SystemExit(main())
