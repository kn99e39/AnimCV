#!/usr/bin/env python3
"""Run Track B: deterministic bounded nonlinear gap reconstruction.

This script consumes frozen H0 predictions and the exact structural
observation contract.  It does not call the VLM and it does not train or
modify the production Frame Pose path.  Linear interpolation is materialized
only as a historical quantitative control; it is never used as a fallback when
the nonlinear candidate lacks support.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from common.serialization import write_json
from framepose.bank import load_bank
from framepose.nonlinear_reconstruction import (
    RECONSTRUCTION_SCHEMA, evaluate_reconstruction, in_frame_mask,
    reconstruct_nonlinear,
)


FROZEN_H0_SHA256 = {
    "test": "6df04c2031b2bde6a4f0af37fdae0a7f0228e9fc23401e15e1aa2b3bd35695a5",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sample_time(sample) -> float:
    if sample.timestamp is not None and np.isfinite(float(sample.timestamp)):
        return float(sample.timestamp)
    if sample.fps and sample.fps > 0:
        return float(sample.frame_index) / float(sample.fps)
    return float(sample.frame_index)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", required=True, type=Path)
    parser.add_argument("--h0", required=True, type=Path)
    parser.add_argument("--split", choices=("train", "validation", "test"), default="test")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--allow-nonfrozen-h0", action="store_true",
                        help="only for synthetic/unit experiments; never use for the frozen test baseline")
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError(f"refusing to overwrite Track B output: {args.out}")

    bank = load_bank(args.bank)
    positions = bank.indices(args.split)
    h0 = np.asarray(np.load(args.h0), dtype=np.float64)
    expected_shape = (len(positions), len(bank.arrays["input_2d"][0]), 3)
    if h0.shape != expected_shape:
        raise ValueError(f"H0 shape {h0.shape} does not match {expected_shape}")
    h0_sha256 = _sha256(args.h0)
    if args.split in FROZEN_H0_SHA256 and not args.allow_nonfrozen_h0:
        if h0_sha256 != FROZEN_H0_SHA256[args.split]:
            raise ValueError("test H0 SHA-256 differs from the frozen context_h0_v2 artifact")

    observation = bank.arrays["input_2d"][positions]
    observation_valid = bank.arrays["input_valid"][positions]
    usable = in_frame_mask(observation, observation_valid)
    target = bank.arrays["target_3d"][positions].astype(np.float64)
    target_valid = bank.arrays["target_valid"][positions]
    samples = [bank.samples[int(position)] for position in positions]
    timestamps = np.asarray([_sample_time(sample) for sample in samples], dtype=np.float64)
    sequence_ids = [sample.sequence_id for sample in samples]
    frame_indices = [sample.frame_index for sample in samples]

    result = reconstruct_nonlinear(h0, usable, timestamps, sequence_ids,
                                   frame_indices=frame_indices)
    metrics = evaluate_reconstruction(
        h0, result.recovered, result.linear_control, target, target_valid,
        result.changed, result.gaps, timestamps, bone_records=result.bone_records)
    report = {
        "schema": RECONSTRUCTION_SCHEMA,
        "track": "B_structural_anchor_nonlinear_reconstruction",
        "bank": {"path": str(args.bank), "content_digest": bank.content_digest(),
                 "provenance_fingerprint": bank.provenance_fingerprint(),
                 "regime": bank.regime(), "split": args.split,
                 "frame_count": int(len(positions))},
        "frozen_h0": {"path": str(args.h0), "sha256": h0_sha256,
                      "expected_sha256": FROZEN_H0_SHA256.get(args.split),
                      "allow_nonfrozen_h0": bool(args.allow_nonfrozen_h0)},
        "reliability_source": {
            "track": "B",
            "usable_anchor": "input_valid AND finite normalized coordinates inside [0,1]",
            "out_of_frame_is_exact_geometry": True,
            "detector_missing_is_not_called_occluded": True,
            "vlm_used": False,
            "cross_sequence_support": False,
        },
        "representation": {
            "parent_position": True,
            "child_direction": True,
            "trusted_bone_length": True,
            "swing_orientation": "deterministic shortest arc from canonical +Z",
            "roll_twist": "not represented",
            "target_rig_bone_transform": "not represented",
        },
        "rules": {
            "position": "one timestamp-aware cubic Hermite segment with fixed endpoint tangents",
            "tangent_support": "up to A-2,A-1,A and B,B+1,B+2; no extrapolation",
            "direction": "hemisphere-continuous SQUAD over swing quaternions",
            "bone_length": "median of >=2 finite positive trusted-anchor lengths from A-2,A-1,A,B,B+1,B+2",
            "linear_control": "historical quantitative control only; never a fallback",
            "gap_bins": ["1-2", "3-5", ">5"],
        },
        "counts": {
            "structurally_usable": int(np.count_nonzero(usable)),
            "structurally_unusable": int(np.count_nonzero(~usable)),
            "gaps": int(len(result.gaps)),
            "reconstructable_gaps": int(len(result.resolved_gap_keys)),
            "unresolved_gaps": int(len(result.unresolved)),
            "changed_joint_frames": int(np.count_nonzero(result.changed)),
        },
        "gaps": [gap.to_dict() for gap in result.gaps],
        "unresolved": result.unresolved,
        "resolved_bones": result.bone_records,
        "metrics": metrics,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    np.savez(args.out / "reconstruction.npz", positions=positions,
             h0=h0.astype(np.float32), nonlinear=result.recovered.astype(np.float32),
             linear_control=result.linear_control.astype(np.float32), changed=result.changed,
             usable=usable, resolved=result.resolved)
    write_json(args.out / "reconstruction.json", report)
    print(json.dumps({
        "split": args.split,
        "gaps": len(result.gaps),
        "reconstructable_gaps": len(result.resolved_gap_keys),
        "unresolved_gaps": len(result.unresolved),
        "changed_joint_frames": int(np.count_nonzero(result.changed)),
        "median_delta_mm": metrics["median_delta_mm"],
        "output": str(args.out),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
