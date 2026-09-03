#!/usr/bin/env python3
"""Run the controlled F0/F1/F2 observation-backend comparison.

Every candidate sees the same frames, the same crop contract, the same
geometry, the same loss contract, the same optimizer, the same seed and the same
evaluator. The only variable is the observation backend:

    F0  geometry only
    F1  conventional pretrained vision encoder + geometry
    F2  vision-language pretrained representation + geometry

Model selection uses validation only; test ground truth is read exactly once,
for the final report.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from common.serialization import write_json
from framepose.bank import load_bank
from framepose.evaluate import compare, evaluate_predictions
from framepose.features import load_feature_cache
from framepose.train import CandidateConfig, geometry_tensor, predict, train_candidate


CANDIDATES = {
    "F0": {"name": "F0_geometry_only", "backbone": "none"},
    "F1": {"name": "F1_vision_geometry", "backbone": "vit_in21k"},
    "F2": {"name": "F2_vlm_geometry", "backbone": "siglip"},
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Controlled frame-pose observation-backend comparison")
    parser.add_argument("--bank", required=True, type=Path)
    parser.add_argument("--features-root", type=Path, default=None)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--candidates", default="F0,F1,F2")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--evaluate-every", type=int, default=5)
    parser.add_argument("--no-mixed-precision", action="store_true")
    parser.add_argument("--compile-training-graph", action="store_true")
    args = parser.parse_args()

    bank = load_bank(args.bank)
    bank.assert_split_isolation()
    # Oracle-geometry and real-observation frames must not be pooled into one
    # measurement; `regime()` raises rather than letting them mix silently.
    regime = bank.regime()
    geometry = geometry_tensor(bank)
    args.out.mkdir(parents=True, exist_ok=True)

    matrix = {
        "schema": "animcv_frame_pose_experiment_matrix_v1",
        "bank_fingerprint": bank.fingerprint(args.bank),
        "bank_metadata": {key: bank.metadata.get(key) for key in
                          ("split_counts", "sequence_counts", "modality_by_source",
                           "strata_thresholds", "require_rgb", "intake")},
        "observation_regime": regime,
        "observation": bank.observation_summary(),
        "shared": {
            "epochs": args.epochs, "batch_size": args.batch_size,
            "learning_rate": args.learning_rate, "weight_decay": args.weight_decay,
            "seed": args.seed, "loss_contract": "baseline_geometry_v1",
            "selection_split": "validation",
            "execution_backend": "compiled" if args.compile_training_graph else "eager",
        },
        "candidates": {},
    }

    reports: dict[str, dict] = {}
    for key in [item.strip() for item in args.candidates.split(",") if item.strip()]:
        if key not in CANDIDATES:
            raise ValueError(f"unknown candidate {key!r}; known: {sorted(CANDIDATES)}")
        definition = CANDIDATES[key]
        features = None
        cache_metadata = None
        if definition["backbone"] != "none":
            if args.features_root is None:
                raise ValueError(f"candidate {key} needs --features-root")
            cached, cache_metadata = load_feature_cache(args.features_root, definition["backbone"], bank)
            features = np.asarray(cached)
        config = CandidateConfig(
            name=definition["name"], backbone=definition["backbone"],
            loss_contract="baseline_geometry_v1", epochs=args.epochs, batch_size=args.batch_size,
            learning_rate=args.learning_rate, weight_decay=args.weight_decay, seed=args.seed,
            device=args.device, mixed_precision=not args.no_mixed_precision,
            compile_training_graph=args.compile_training_graph, evaluate_every=args.evaluate_every,
        )
        directory = args.out / key
        directory.mkdir(parents=True, exist_ok=True)
        training = train_candidate(bank, config, features=features, geometry=geometry,
                                   checkpoint_path=directory / "checkpoint.pt")
        if cache_metadata is not None:
            training["feature_cache"] = {branch: cache_metadata[branch] for branch in
                                         ("backbone", "sample_count", "shape", "sample_order_digest")}
        write_json(directory / "training_report.json", training)

        evaluation = _evaluate_candidate(bank, geometry, features, directory, config, args)
        for split, report in evaluation.items():
            write_json(directory / f"evaluation_{split}.json", report)
        reports[key] = {"training": training, "evaluation": evaluation}
        matrix["candidates"][key] = {
            "config": config.to_dict(),
            "backbone": training["backbone"],
            "model": training["model"],
            "selection": training["selection"],
            "performance": training["performance"],
            "execution": training["execution"],
            "aggregate": {split: report["aggregate"] for split, report in evaluation.items()},
        }

    matrix["comparisons"] = {}
    for split in ("validation", "test"):
        for baseline, candidate in (("F0", "F1"), ("F0", "F2"), ("F1", "F2")):
            if baseline not in reports or candidate not in reports:
                continue
            if split not in reports[baseline]["evaluation"] or split not in reports[candidate]["evaluation"]:
                continue
            delta = compare(reports[baseline]["evaluation"][split], reports[candidate]["evaluation"][split])
            matrix["comparisons"][f"{split}:{candidate}_vs_{baseline}"] = delta
            write_json(args.out / f"compare_{split}_{candidate}_vs_{baseline}.json", delta)

    write_json(args.out / "experiment_matrix.json", matrix)
    print(json.dumps({key: {
        "test_mpjpe_mm": value["aggregate"].get("test", {}).get("mpjpe_mm", {}).get("mean")
                         if value["aggregate"].get("test") else None,
        "validation_mpjpe_mm": value["selection"]["validation_mpjpe_mm"],
        "frames_per_second": value["performance"]["frames_per_second"],
    } for key, value in matrix["candidates"].items()}, indent=2, sort_keys=True))
    return 0


def _evaluate_candidate(bank, geometry, features, directory: Path, config: CandidateConfig, args):
    from framepose.train import load_checkpoint
    import torch

    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    model, _ = load_checkpoint(directory / "checkpoint.pt", device=str(device))
    evaluation = {}
    for split in ("validation", "test"):
        positions = bank.indices(split)
        if not len(positions):
            continue
        prediction = predict(model, torch, geometry, features, positions, device)
        np.save(directory / f"prediction_{split}.npy", prediction.astype(np.float32))
        evaluation[split] = evaluate_predictions(bank, positions, prediction, candidate=config.name)
    return evaluation


if __name__ == "__main__":
    raise SystemExit(main())
