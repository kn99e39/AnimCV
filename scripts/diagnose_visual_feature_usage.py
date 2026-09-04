#!/usr/bin/env python3
"""Diagnose HOW a visual candidate uses its image tokens (no training).

When an RGB candidate does not beat the geometry-only control, the migration
direction requires diagnosing visual-feature usage before adding complexity.
This replays a trained checkpoint under four token conditions:

    real      the frame's own cached tokens
    zero      tokens replaced by zeros (the visual path silenced)
    shuffled  tokens taken from a different, randomly chosen frame of the same
              split (image content present, but unrelated to this pose)
    neighbour tokens taken from another frame of the SAME sequence (same scene
              and subject, different pose)

Reading: a model whose visual path carries transferable pose evidence degrades
sharply under `shuffled` on *every* split. A model that has learned scene
identity degrades under `shuffled` on the split it memorised and barely moves on
an unseen split, and is nearly unchanged under `neighbour` (same scene).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from common.serialization import write_json
from framepose.bank import load_bank
from framepose.features import load_feature_cache
from framepose.train import geometry_tensor, load_checkpoint


def _mpjpe_mm(bank, positions, prediction) -> float:
    target = bank.arrays["target_3d"][positions].astype(np.float64)
    valid = bank.arrays["target_valid"][positions]
    errors = np.linalg.norm(prediction - target, axis=-1) * 1000.0
    return float((errors * valid).sum() / max(valid.sum(), 1))


def _token_indices(bank, positions: np.ndarray, condition: str, seed: int) -> np.ndarray:
    generator = np.random.default_rng(seed)
    if condition == "real":
        return positions
    if condition == "shuffled":
        return positions[generator.permutation(len(positions))]
    if condition == "neighbour":
        buckets: dict[str, list[int]] = {}
        for position in positions:
            buckets.setdefault(bank.samples[int(position)].sequence_id, []).append(int(position))
        mapped = []
        for position in positions:
            members = buckets[bank.samples[int(position)].sequence_id]
            choice = int(generator.choice(members))
            if len(members) > 1:
                while choice == int(position):
                    choice = int(generator.choice(members))
            mapped.append(choice)
        return np.asarray(mapped, dtype=np.int64)
    raise ValueError(f"unknown token condition {condition!r}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Diagnose visual-feature usage of a frame-pose candidate")
    parser.add_argument("--bank", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--features-root", required=True, type=Path)
    parser.add_argument("--backbone", required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--visual-input-identity", type=Path, default=None,
                        help="Visual input identity JSON to verify the feature cache against")
    parser.add_argument("--allow-legacy-feature-cache", action="store_true",
                        help="Read a historical v1 cache, which recorded no image-content or "
                             "crop-contract identity; the report labels the provenance level.")
    args = parser.parse_args()

    import torch

    bank = load_bank(args.bank)
    geometry = geometry_tensor(bank)
    fingerprint = None
    if args.visual_input_identity is not None:
        from framepose.visual_input import load_identity
        fingerprint = load_identity(args.visual_input_identity)["fingerprint"]
    cached, cache_metadata = load_feature_cache(args.features_root, args.backbone, bank,
                                                visual_input_fingerprint=fingerprint,
                                                allow_legacy=args.allow_legacy_feature_cache)
    features = np.asarray(cached)
    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    model, payload = load_checkpoint(args.checkpoint, device=str(device))

    results: dict[str, dict[str, float]] = {}
    for split in ("train", "validation", "test"):
        positions = bank.indices(split)
        if not len(positions):
            continue
        split_results: dict[str, float] = {}
        for condition in ("real", "zero", "shuffled", "neighbour"):
            if condition == "zero":
                tokens = np.zeros_like(features[positions])
            else:
                tokens = features[_token_indices(bank, positions, condition, args.seed)]
            prediction = _infer(torch, model, geometry[positions], tokens, device)
            split_results[condition] = _mpjpe_mm(bank, positions, prediction)
        baseline = split_results["real"]
        split_results.update({
            f"{condition}_delta_mm": split_results[condition] - baseline
            for condition in ("zero", "shuffled", "neighbour")
        })
        results[split] = split_results

    report = {
        "schema": "animcv_frame_pose_visual_usage_v1",
        "checkpoint": str(args.checkpoint),
        "candidate": payload["candidate"]["name"],
        "backbone": payload["backbone"],
        "bank_content_digest": bank.content_digest(),
        "feature_cache_sample_order_digest": cache_metadata["sample_order_digest"],
        "feature_cache_provenance_level": cache_metadata.get("provenance_level"),
        "feature_cache_visual_input_verified": cache_metadata.get("visual_input_verified"),
        "seed": args.seed,
        "conditions": {
            "real": "the frame's own cached tokens",
            "zero": "visual path silenced",
            "shuffled": "tokens from an unrelated frame of the same split",
            "neighbour": "tokens from another frame of the same sequence",
        },
        "mpjpe_mm": results,
    }
    write_json(args.out, report)
    print(json.dumps(results, indent=2, sort_keys=True))
    return 0


def _infer(torch, model, geometry: np.ndarray, tokens: np.ndarray, device,
           batch_size: int = 512) -> np.ndarray:
    """Dense inference: the substituted tokens are already in evaluation order."""
    model.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, len(geometry), batch_size):
            stop = start + batch_size
            outputs.append(model(
                torch.as_tensor(geometry[start:stop], device=device),
                torch.as_tensor(np.asarray(tokens[start:stop], dtype=np.float32), device=device),
            ).float().cpu().numpy())
    return np.concatenate(outputs, axis=0).astype(np.float64)


if __name__ == "__main__":
    raise SystemExit(main())
