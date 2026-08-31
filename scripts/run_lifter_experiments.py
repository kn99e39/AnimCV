#!/usr/bin/env python3
"""Run reproducible MPI/3DPW/AMASS temporal-lifter comparison candidates.

The script never places either holdout in a training dataset.  It writes every
combined training artifact, checkpoint, and metric report below ``--out`` so a
server run can be audited or repeated without reconstructing its experiment
matrix manually.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from training.temporal_lifter import TrainingConfig, combine_datasets, evaluate, load_dataset, save_dataset, train


def _paths(value: str) -> list[Path]:
    paths = [Path(part) for part in value.split(",") if part]
    if not paths:
        raise ValueError("dataset list must not be empty")
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"dataset files are missing: {missing}")
    return paths


def _combined(paths: list[Path], destination: Path) -> dict:
    dataset = combine_datasets([load_dataset(path) for path in paths])
    save_dataset(dataset, destination)
    return dataset


def _dataset_fingerprint(path: Path, dataset: dict | None = None) -> dict:
    """Return immutable input provenance needed to compare two experiments.

    Dataset paths alone are insufficient: prepared JSON files can be replaced
    in place as the source intake evolves.  Hash the exact bytes consumed by
    the run and retain the logical frame/sequence counts as a quick human
    sanity check.
    """
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    dataset = dataset if dataset is not None else load_dataset(path)
    return {
        "path": str(path),
        "sha256": digest.hexdigest(),
        "byte_size": path.stat().st_size,
        "frame_count": len(dataset["frames"]),
        "sequence_count": len(dataset.get("sequences", [dataset])),
    }


def _config(args: argparse.Namespace, *, epochs: int, init_checkpoint: Path | None = None) -> TrainingConfig:
    return TrainingConfig(
        window=args.window, channels=args.channels, epochs=epochs, batch_size=args.batch_size,
        learning_rate=args.learning_rate, device=args.device, mixed_precision=not args.no_mixed_precision,
        seed=args.seed, inference_batch_size=args.inference_batch_size,
        input_jitter_std=args.input_jitter_std, input_dropout_probability=args.input_dropout_probability,
        confidence_jitter_std=args.confidence_jitter_std,
        input_coordinate_normalization=args.input_coordinate_normalization, architecture=args.architecture,
        source_balanced_sampling=args.source_balanced_sampling,
        input_global_scale_std=args.input_global_scale_std,
        input_translation_std=args.input_translation_std,
        input_rotation_degrees=args.input_rotation_degrees,
        temporal_occlusion_probability=args.temporal_occlusion_probability,
        temporal_occlusion_frames=args.temporal_occlusion_frames,
        bone_loss_weight=args.bone_loss_weight, torso_loss_weight=args.torso_loss_weight,
        hinge_loss_weight=args.hinge_loss_weight, yaw_loss_weight=args.yaw_loss_weight,
        yaw_tail_loss_weight=args.yaw_tail_loss_weight, hinge_flip_loss_weight=args.hinge_flip_loss_weight,
        end_effector_loss_weight=args.end_effector_loss_weight,
        cartesian_torso_tail_loss_weight=args.cartesian_torso_tail_loss_weight,
        bilateral_forward_depth_supervision=args.bilateral_forward_depth_supervision,
        init_checkpoint=str(init_checkpoint) if init_checkpoint else None,
    )


def _evaluate(checkpoint: Path, validation: dict, holdouts: dict[str, dict], device: str) -> dict:
    return {
        "validation": evaluate(validation, checkpoint, device),
        "holdouts": {name: evaluate(dataset, checkpoint, device) for name, dataset in holdouts.items()},
    }


def _run_candidate(
    name: str, dataset: dict, args: argparse.Namespace, reports_dir: Path, validation: dict,
    holdouts: dict[str, dict], *, init_checkpoint: Path | None = None, epochs: int | None = None,
) -> dict:
    checkpoint = reports_dir / f"{name}.pth"
    train_report = train(dataset, checkpoint, _config(args, epochs=epochs or args.epochs, init_checkpoint=init_checkpoint))
    result = {
        "checkpoint": str(checkpoint),
        "training": train_report,
        "evaluation": _evaluate(checkpoint, validation, holdouts, args.device),
    }
    (reports_dir / f"{name}.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Run source-mixing and AMASS pretrain/fine-tune lifter experiments")
    parser.add_argument("--mpi-train", required=True)
    parser.add_argument("--three-dpw-train", required=True)
    parser.add_argument("--amass-train", required=True)
    parser.add_argument("--validation", required=True, help="comma-separated validation datasets")
    parser.add_argument("--three-dpw-holdout", required=True)
    parser.add_argument("--amass-holdout", required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--pretrain-epochs", type=int, default=None)
    parser.add_argument("--window", type=int, default=81)
    parser.add_argument("--channels", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-mixed-precision", action="store_true")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--inference-batch-size", type=int, default=1024)
    parser.add_argument("--input-jitter-std", type=float, default=0.015)
    parser.add_argument("--input-dropout-probability", type=float, default=0.05)
    parser.add_argument("--confidence-jitter-std", type=float, default=0.08)
    parser.add_argument("--input-coordinate-normalization", choices=["image_v1", "pelvis_torso_v1"],
                        default="image_v1")
    parser.add_argument("--input-global-scale-std", type=float, default=0.04)
    parser.add_argument("--input-translation-std", type=float, default=0.03)
    parser.add_argument("--input-rotation-degrees", type=float, default=12.0)
    parser.add_argument("--temporal-occlusion-probability", type=float, default=0.10)
    parser.add_argument("--temporal-occlusion-frames", type=int, default=9)
    parser.add_argument("--source-balanced-sampling", action="store_true")
    parser.add_argument("--architecture", choices=["legacy_tcn_v1", "dilated_tcn_v1"], default="dilated_tcn_v1")
    parser.add_argument("--bone-loss-weight", type=float, default=0.25)
    parser.add_argument("--torso-loss-weight", type=float, default=0.15)
    parser.add_argument("--hinge-loss-weight", type=float, default=0.15)
    parser.add_argument("--yaw-loss-weight", type=float, default=0.0)
    parser.add_argument("--yaw-tail-loss-weight", type=float, default=0.0)
    parser.add_argument("--hinge-flip-loss-weight", type=float, default=0.0)
    parser.add_argument("--end-effector-loss-weight", type=float, default=0.0)
    parser.add_argument("--cartesian-torso-tail-loss-weight", type=float, default=0.0)
    parser.add_argument("--bilateral-forward-depth-supervision", action="store_true",
                         help="all-frame signed bilateral forward-depth (+Y) supervision, "
                              "pooled into the base coordinate loss with no tunable weight (docs/10 A14)")
    parser.add_argument("--candidates", default="mpi_only,mpi_3dpw,direct_mix,amass_pretrain,amass_pretrain_mpi_3dpw_finetune",
                        help="comma-separated subset of the reproducible candidate matrix")
    args = parser.parse_args()
    if args.pretrain_epochs is not None and args.pretrain_epochs <= 0:
        raise ValueError("--pretrain-epochs must be positive")
    candidates = {item for item in args.candidates.split(",") if item}
    valid_candidates = {"mpi_only", "mpi_3dpw", "direct_mix", "amass_pretrain", "amass_pretrain_mpi_3dpw_finetune"}
    unknown_candidates = candidates.difference(valid_candidates)
    if not candidates or unknown_candidates:
        raise ValueError(f"--candidates contains unsupported values: {sorted(unknown_candidates)}")
    if "amass_pretrain_mpi_3dpw_finetune" in candidates:
        candidates.add("amass_pretrain")

    output = args.out
    datasets_dir, reports_dir = output / "datasets", output / "reports"
    datasets_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    mpi, three_dpw, amass = (Path(args.mpi_train), Path(args.three_dpw_train), Path(args.amass_train))
    for path in (mpi, three_dpw, amass):
        if not path.is_file():
            raise FileNotFoundError(path)

    validation = _combined(_paths(args.validation), datasets_dir / "validation.json")
    holdouts = {
        "3dpw_test": load_dataset(args.three_dpw_holdout),
        "amass_internal": load_dataset(args.amass_holdout),
    }
    mpi_only = load_dataset(mpi)
    mpi_three_dpw = _combined([mpi, three_dpw], datasets_dir / "mpi_3dpw_train.json")
    direct_mix = _combined([mpi, three_dpw, amass], datasets_dir / "direct_mix_train.json")
    amass_only = load_dataset(amass)

    results = {
        "schema": "animcv_lifter_experiment_matrix_v2",
        "config": asdict(_config(args, epochs=args.epochs)),
        "datasets": {
            "mpi_train": str(mpi), "three_dpw_train": str(three_dpw), "amass_train": str(amass),
            "validation": [str(path) for path in _paths(args.validation)],
            "three_dpw_holdout": args.three_dpw_holdout, "amass_holdout": args.amass_holdout,
        },
        "dataset_fingerprints": {
            "mpi_train": _dataset_fingerprint(mpi, mpi_only),
            "three_dpw_train": _dataset_fingerprint(three_dpw),
            "amass_train": _dataset_fingerprint(amass, amass_only),
            "validation": _dataset_fingerprint(datasets_dir / "validation.json", validation),
            "three_dpw_holdout": _dataset_fingerprint(Path(args.three_dpw_holdout), holdouts["3dpw_test"]),
            "amass_holdout": _dataset_fingerprint(Path(args.amass_holdout), holdouts["amass_internal"]),
        },
        "candidates": {},
    }
    if "mpi_only" in candidates:
        results["candidates"]["mpi_only"] = _run_candidate("mpi_only", mpi_only, args, reports_dir, validation, holdouts)
    if "mpi_3dpw" in candidates:
        results["candidates"]["mpi_3dpw"] = _run_candidate("mpi_3dpw", mpi_three_dpw, args, reports_dir, validation, holdouts)
    if "direct_mix" in candidates:
        results["candidates"]["direct_mix"] = _run_candidate("direct_mix", direct_mix, args, reports_dir, validation, holdouts)
    pretrain_epochs = args.pretrain_epochs or args.epochs
    pretrain = None
    if "amass_pretrain" in candidates:
        pretrain = _run_candidate("amass_pretrain", amass_only, args, reports_dir, validation, holdouts, epochs=pretrain_epochs)
        results["candidates"]["amass_pretrain"] = pretrain
    if "amass_pretrain_mpi_3dpw_finetune" in candidates:
        results["candidates"]["amass_pretrain_mpi_3dpw_finetune"] = _run_candidate(
            "amass_pretrain_mpi_3dpw_finetune", mpi_three_dpw, args, reports_dir, validation, holdouts,
            init_checkpoint=Path(pretrain["checkpoint"]),
        )
    (output / "experiment_matrix.json").write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "out": str(output),
        "candidates": list(results["candidates"]),
        "validation_frames": len(validation["frames"]),
        "holdout_frames": {name: len(dataset["frames"]) for name, dataset in holdouts.items()},
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
