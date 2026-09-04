#!/usr/bin/env python3
"""Run the controlled F0/F1/F2 observation-backend comparison.

Every candidate sees the same frames, the same crop contract, the same
geometry, the same loss contract, the same optimizer, the same seed and the same
evaluator.

    F0  geometry only                          (no image projection, no cross-attention)
    F1  frozen ImageNet-pretrained ViT tower   + geometry
    F2  frozen vision-language-pretrained
        (SigLIP) image tower                   + geometry

What each comparison can establish -- this is not symmetric:

    F0 vs F1/F2   the tested geometry-only architecture against the tested
                  visual-fusion architectures. F0 lacks the image projection and
                  the cross-attention sublayer, so it has a different trainable
                  parameter count; this is NOT an information-only control and
                  must never be reported as one.

    F1 vs F2      architecture-matched: identical ViT-B/16 geometry, identical
                  224x224 input, identical 14x14x768 token grid, identical
                  trainable model. The only difference is what the frozen tower
                  was pretrained on, so a difference is attributable to the
                  pretraining representation.

F2 uses the SigLIP **image tower** only -- no text encoder, no multimodal
projector, no language decoder. It is not a full-VLM path.

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
from framepose.observations import assert_quality_interpretable
from framepose.train import CandidateConfig, geometry_tensor, predict, train_candidate


# `name` is the precise terminology used by new runs. `historical_name` is the
# identifier the executed F0/F1/F2 lineage recorded; historical output
# directories, checkpoints and reports are never renamed, and
# --historical-candidate-names reproduces those identifiers exactly.
CANDIDATES = {
    "F0": {"name": "F0_geometry_only",
           "historical_name": "F0_geometry_only", "backbone": "none"},
    "F1": {"name": "F1_imagenet_vision_geometry",
           "historical_name": "F1_vision_geometry", "backbone": "vit_in21k"},
    "F2": {"name": "F2_vlpretrained_vision_geometry",
           "historical_name": "F2_vlm_geometry", "backbone": "siglip"},
}

COMPARISON_SEMANTICS = {
    "F1_vs_F0": ("tested geometry-only architecture vs tested visual-fusion architecture; "
                 "F0 lacks image projection and cross-attention, so this is not an "
                 "information-only control"),
    "F2_vs_F0": ("tested geometry-only architecture vs tested visual-fusion architecture; "
                 "F0 lacks image projection and cross-attention, so this is not an "
                 "information-only control"),
    "F2_vs_F1": ("architecture-matched: parameter-identical trainable model, identical token "
                 "grid and preprocessing; the variable is the frozen tower's pretraining"),
    "capacity_matched": False,
    "parameter_matched_geometry_control_trained": False,
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
    parser.add_argument("--compile-training-graph", action="store_true",
                        help="Accepted execution path for new FramePose training "
                             "(Architecture_v3 section 14); historical F0/F1/F2 ran eager.")
    parser.add_argument("--historical-candidate-names", action="store_true",
                        help="Record the original F0/F1/F2 candidate identifiers instead of the "
                             "precise ones, for reproducing the historical lineage exactly.")
    parser.add_argument("--visual-input-identity", type=Path, default=None,
                        help="Visual input identity JSON to verify each feature cache against "
                             "(framepose.visual_input); without it, image-content identity is "
                             "not re-verified at load and the matrix records that.")
    parser.add_argument("--allow-legacy-feature-cache", action="store_true",
                        help="Read a historical v1 feature cache, which recorded no image-content "
                             "or crop-contract identity. The matrix labels the run accordingly.")
    args = parser.parse_args()

    bank = load_bank(args.bank)
    bank.assert_split_isolation()
    visual_fingerprint = None
    if args.visual_input_identity is not None:
        from framepose.visual_input import load_identity
        visual_fingerprint = load_identity(args.visual_input_identity)["fingerprint"]
    # Frames from different observation regimes must not be pooled into one
    # measurement, and a bank whose provenance cannot be resolved cannot support
    # any claim that depends on observation quality. Both raise here rather than
    # producing a number whose meaning is unknown.
    regime = assert_quality_interpretable(bank.regime())
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
        "comparison_semantics": COMPARISON_SEMANTICS,
        "visual_input_identity": str(args.visual_input_identity) if args.visual_input_identity else None,
        "visual_input_verified": args.visual_input_identity is not None,
        "legacy_feature_cache_allowed": args.allow_legacy_feature_cache,
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
            cached, cache_metadata = load_feature_cache(
                args.features_root, definition["backbone"], bank,
                visual_input_fingerprint=visual_fingerprint,
                allow_legacy=args.allow_legacy_feature_cache)
            features = np.asarray(cached)
        candidate_name = definition["historical_name"] if args.historical_candidate_names \
            else definition["name"]
        config = CandidateConfig(
            name=candidate_name, backbone=definition["backbone"],
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
            training["feature_cache"] = {
                key: cache_metadata.get(key) for key in
                ("schema", "backbone", "sample_count", "shape", "sample_order_digest",
                 "visual_input_fingerprint", "crop_contract_digest", "feature_cache_provenance",
                 "provenance_level", "visual_input_verified", "weight_verification",
                 "not_established")}
        write_json(directory / "training_report.json", training)

        evaluation = _evaluate_candidate(bank, geometry, features, directory, config, args)
        for split, report in evaluation.items():
            write_json(directory / f"evaluation_{split}.json", report)
        reports[key] = {"training": training, "evaluation": evaluation}
        matrix["candidates"][key] = {
            "candidate_name": candidate_name,
            "historical_candidate_name": definition["historical_name"],
            "feature_cache": training.get("feature_cache"),
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
