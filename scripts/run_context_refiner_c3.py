#!/usr/bin/env python3
"""Run the single controlled C3 supervision-domain test.

C3 keeps the frozen C2 architecture, runtime gate, thresholds, optimizer,
learning-rate schedule, weight decay, epochs, batch size, seed, and validation
selection rule.  The only intervention is the training supervision domain:
the existing canonical coordinate-only loss is evaluated on target-valid
gate-on joints.  Gate-off output remains the exact frozen H0 value.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from common.serialization import write_json
from framepose.bank import load_bank
from framepose.context_refiner import (
    DEFAULT_HIDDEN,
    CONTEXT_DIM,
    WINDOW_OFFSETS,
    _apply_residual,
    _torch,
    apply_gate,
    build_context_features,
    build_refiner,
    fit_gate_thresholds,
    interpolate_gated_h0,
    load_checkpoint,
    parameter_report,
    predict,
    runtime_signals,
)
from framepose.contract import JOINT_COUNT
from framepose.evaluate import evaluate_predictions
from framepose.losses import COORDINATE_ONLY_V1, compute_loss

try:
    from diagnose_context_refiner_failure import test_cohort_masks
except ModuleNotFoundError:  # importable from repository-root test runners
    from scripts.diagnose_context_refiner_failure import test_cohort_masks


SCHEMA = "animcv_context_refiner_c3_experiment_v1"
FIXED_BANK_DIGEST = "75519e6394a764e3749ddaa30555b58b73a01db582ccee14f661374b9a0ed536"


def sha256(path: Path) -> str:
    import hashlib
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_h0(path: Path, count: int) -> np.ndarray:
    values = np.asarray(np.load(path), dtype=np.float32)
    expected = (count, JOINT_COUNT, 3)
    if values.shape != expected:
        raise ValueError(f"H0 shape {values.shape}, expected {expected}")
    return values


def assemble_h0(bank, paths: dict[str, Path]) -> tuple[np.ndarray, dict[str, Any]]:
    result = np.full((len(bank), JOINT_COUNT, 3), np.nan, dtype=np.float32)
    identity = {}
    for split, path in paths.items():
        positions = bank.indices(split)
        values = load_h0(path, len(positions))
        result[positions] = values
        identity[split] = {"path": str(path), "sha256": sha256(path), "shape": list(values.shape)}
    if not np.isfinite(result).all():
        raise ValueError("H0 does not cover every bank row")
    return result, identity


def train_c3(bank, h0, contexts, decision, c2_candidate: dict[str, Any],
             checkpoint_path: Path) -> dict[str, Any]:
    torch, _ = _torch()
    epochs = int(c2_candidate["epochs"])
    batch_size = int(c2_candidate["batch_size"])
    learning_rate = float(c2_candidate["learning_rate"])
    weight_decay = float(c2_candidate["weight_decay"])
    seed = int(c2_candidate["seed"])
    requested_device = str(c2_candidate["device"])
    mixed_precision = bool(c2_candidate["mixed_precision"])
    device = torch.device(requested_device if (requested_device != "cuda" or torch.cuda.is_available()) else "cpu")
    torch.manual_seed(seed)
    model = build_refiner(hidden=DEFAULT_HIDDEN).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    context_gpu = torch.as_tensor(contexts, dtype=torch.float32, device=device)
    h0_gpu = torch.as_tensor(np.nan_to_num(h0, nan=0.0, posinf=0.0, neginf=0.0), dtype=torch.float32, device=device)
    target_gpu = torch.as_tensor(bank.arrays["target_3d"], dtype=torch.float32, device=device)
    target_valid_gpu = torch.as_tensor(bank.arrays["target_valid"], dtype=torch.float32, device=device)
    gate_gpu = torch.as_tensor(decision.gate_on, dtype=torch.bool, device=device)
    amp_enabled = bool(mixed_precision and device.type == "cuda")
    if hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    else:  # pragma: no cover
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    train_positions = bank.indices("train")
    validation_positions = bank.indices("validation")
    steps_per_epoch = math.ceil(len(train_positions) / batch_size)
    total_steps = steps_per_epoch * epochs
    best_state = None
    best_validation = None
    telemetry = []
    started = perf_counter()
    step = 0
    for epoch in range(epochs):
        model.train()
        permutation = train_positions[torch.randperm(len(train_positions), generator=generator).numpy()]
        losses = []
        for start in range(0, len(permutation), batch_size):
            batch = permutation[start:start + batch_size]
            index = torch.as_tensor(batch, device=device)
            progress = min(step / max(total_steps - 1, 1), 1.0)
            lr = learning_rate * 0.5 * (1.0 + math.cos(math.pi * progress))
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                residual = model(context_gpu[index])
                refined = _apply_residual(torch, h0_gpu[index], residual, gate_gpu[index])
                gate_aligned_mask = target_valid_gpu[index][..., None] * gate_gpu[index][..., None].to(torch.float32)
                loss = compute_loss(torch, refined, target_gpu[index], gate_aligned_mask,
                                    COORDINATE_ONLY_V1)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().item()))
            step += 1
        record = {"epoch": epoch, "train_loss": float(np.mean(losses)), "learning_rate": lr}
        if len(validation_positions):
            model.eval()
            outputs = []
            with torch.no_grad():
                for start in range(0, len(validation_positions), 512):
                    batch = validation_positions[start:start + 512]
                    index = torch.as_tensor(batch, device=device)
                    outputs.append(_apply_residual(
                        torch, h0_gpu[index], model(context_gpu[index]), gate_gpu[index]
                    ).float().cpu().numpy())
            validation_prediction = np.concatenate(outputs, axis=0)
            target = bank.arrays["target_3d"][validation_positions]
            valid = bank.arrays["target_valid"][validation_positions]
            error = np.linalg.norm(validation_prediction - target, axis=-1)
            validation_mpjpe = float(error[valid].mean() * 1000.0)
            record["validation_mpjpe_mm"] = validation_mpjpe
            if best_validation is None or validation_mpjpe < best_validation:
                best_validation = validation_mpjpe
                best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
        telemetry.append(record)
    if best_state is not None:
        model.load_state_dict(best_state)
    elapsed = perf_counter() - started
    report = {
        "schema": "animcv_context_refiner_c3_training_v1",
        "candidate": {
            "name": "C3_GATE_ALIGNED_TRAINING",
            "architecture_source": "C2_support_gated_context_refiner exact",
            "hidden": DEFAULT_HIDDEN,
            "optimizer": "AdamW",
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "seed": seed,
            "device": str(device),
            "mixed_precision": amp_enabled,
        },
        "architecture": parameter_report(model),
        "supervision": {
            "loss_contract": COORDINATE_ONLY_V1.to_dict(),
            "mask": "target_valid AND runtime_gate_on joint rows",
            "gate_off_output": "exact H0",
            "structural_terms": "omitted because canonical pair/chain masking would change semantics",
            "only_changed_contract_field": "training supervision domain",
        },
        "selection": {"criterion": "whole-frame validation MPJPE", "split": "validation",
                      "test_ground_truth_used": False, "best_validation_mpjpe_mm": best_validation},
        "bank": {"content_digest": bank.content_digest(), "train_frames": int(len(train_positions)),
                 "validation_frames": int(len(validation_positions)), "test_ground_truth_used": False},
        "gate": {"activation_rate_by_split": {
            split: float(decision.gate_on[bank.indices(split)].mean())
            for split in ("train", "validation", "test")}},
        "execution": {"device": str(device), "mixed_precision": amp_enabled,
                      "torch_version": torch.__version__},
        "performance": {"training_seconds": elapsed, "steps": step},
        "epoch_telemetry": telemetry,
    }
    torch.save({"schema": "animcv_context_refiner_c3_checkpoint_v1",
                "model_config": {"context_dim": CONTEXT_DIM, "hidden": DEFAULT_HIDDEN},
                "candidate": report["candidate"], "bank_content_digest": bank.content_digest(),
                "state_dict": model.state_dict()}, checkpoint_path)
    report["checkpoint_path"] = str(checkpoint_path)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", required=True, type=Path)
    parser.add_argument("--h0-train", required=True, type=Path)
    parser.add_argument("--h0-validation", required=True, type=Path)
    parser.add_argument("--h0-test", required=True, type=Path)
    parser.add_argument("--c2-checkpoint", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError(f"refusing to overwrite C3 output: {args.out}")
    bank = load_bank(args.bank)
    bank.assert_split_isolation()
    if bank.content_digest() != FIXED_BANK_DIGEST:
        raise ValueError("unexpected bank digest")
    h0, h0_identity = assemble_h0(bank, {"train": args.h0_train, "validation": args.h0_validation, "test": args.h0_test})
    contexts, windows = build_context_features(bank, h0)
    signals = runtime_signals(bank, h0)
    threshold_path = args.c2_checkpoint.parent / "gate_thresholds.json"
    from framepose.context_refiner import GateThresholds
    thresholds = GateThresholds.from_dict(json.loads(threshold_path.read_text(encoding="utf-8")))
    decision = apply_gate(signals, thresholds)
    import torch
    c2_payload = torch.load(args.c2_checkpoint, map_location="cpu", weights_only=False)
    if c2_payload.get("schema") != "animcv_context_refiner_checkpoint_v1":
        raise ValueError("unexpected C2 checkpoint schema")
    c2_candidate = c2_payload["candidate"]
    for key, expected in {"hidden": DEFAULT_HIDDEN, "epochs": 80, "batch_size": 256,
                          "seed": 1337}.items():
        if int(c2_candidate[key]) != expected:
            raise ValueError(f"C2 {key} is not the frozen contract: {c2_candidate[key]}")
    if float(c2_candidate["learning_rate"]) != 1e-3 or float(c2_candidate["weight_decay"]) != 1e-4:
        raise ValueError("C2 optimizer hyperparameters are not the frozen contract")
    args.out.mkdir(parents=True, exist_ok=True)
    training = train_c3(bank, h0, contexts, decision, c2_candidate, args.out / "checkpoint.pt")
    write_json(args.out / "training_report.json", training)
    # C2 replay is read-only and provides the exact information-control
    # comparison.  C3 uses the same all-bank gate and context path.
    c2_model, _ = load_checkpoint(args.c2_checkpoint, device=args.device)
    c2_prediction = predict(c2_model, contexts, h0, decision.gate_on,
                             np.arange(len(bank)), device=args.device)
    c3_payload = torch.load(args.out / "checkpoint.pt", map_location=args.device, weights_only=False)
    c3_model = build_refiner(hidden=DEFAULT_HIDDEN).to(args.device)
    c3_model.load_state_dict(c3_payload["state_dict"])
    c3_model.eval()
    c3_prediction = predict(c3_model, contexts, h0, decision.gate_on,
                            np.arange(len(bank)), device=args.device)
    c1_prediction = interpolate_gated_h0(h0, signals, decision.gate_on)
    predictions = {"C0": h0, "C1": c1_prediction, "C2": c2_prediction, "C3": c3_prediction}
    test_positions = bank.indices("test")
    for name, prediction in predictions.items():
        np.save(args.out / f"prediction_test_{name}.npy", prediction[test_positions].astype(np.float32))
        write_json(args.out / f"evaluation_test_{name}.json",
                   evaluate_predictions(bank, test_positions, prediction[test_positions], candidate=name))

    test_target = bank.arrays["target_3d"][test_positions]
    test_valid = bank.arrays["target_valid"][test_positions]
    raw_masks = test_cohort_masks(bank, h0, signals)
    masks = {"ALL GATE-ON": raw_masks["ALL GATE-ON"] & decision.gate_on[test_positions]}
    masks.update({name: mask for name, mask in raw_masks.items() if name != "ALL GATE-ON"})
    slice_metrics: dict[str, Any] = {}
    accounting: dict[str, Any] = {}
    for label, cohort in {"ALL TEST": np.ones_like(test_valid, dtype=bool), **masks}.items():
        slice_metrics[label] = {}
        accounting[label] = {}
        selected = cohort & test_valid
        for name, prediction in predictions.items():
            error = np.linalg.norm(prediction[test_positions] - test_target, axis=-1) * 1000.0
            values = error[selected]
            slice_metrics[label][name] = {
                "joint_rows": int(cohort.sum()), "evaluated_rows": int(selected.sum()),
                "mpjpe_mm": float(values.mean()) if len(values) else None,
            }
            changed = np.any(prediction[test_positions] != h0[test_positions], axis=-1)
            gate = decision.gate_on[test_positions]
            delta = np.linalg.norm(h0[test_positions] - test_target, axis=-1) - np.linalg.norm(
                prediction[test_positions] - test_target, axis=-1)
            on = selected & gate
            accounting[label][name] = {
                "gate_on_count": int((cohort & gate).sum()),
                "gate_off_count": int((cohort & ~gate).sum()),
                "changed_gate_off_count": int((changed & cohort & ~gate).sum()),
                "gate_on_delta_mm": {
                    "mean": float(delta[on].mean() * 1000.0) if on.any() else None,
                    "improved": int((delta[on] > 0).sum()),
                    "worsened": int((delta[on] < 0).sum()),
                },
            }
    write_json(args.out / "slice_metrics.json", slice_metrics)
    write_json(args.out / "gate_accounting.json", accounting)
    report = {
        "schema": SCHEMA,
        "bank": {"content_digest": bank.content_digest(),
                 "split_counts": {split: int(len(bank.indices(split))) for split in ("train", "validation", "test")}},
        "frozen_h0": h0_identity,
        "frozen_c2": {"checkpoint": str(args.c2_checkpoint), "sha256": sha256(args.c2_checkpoint)},
        "architecture": parameter_report(c3_model),
        "context": {"window_offsets": list(WINDOW_OFFSETS), "context_dimension": int(contexts.shape[1]),
                     "target_frame_only": True, "sequence_window_shape": list(windows.shape)},
        "runtime_gate": {"threshold_source": str(threshold_path), "train_only_fit": True,
                         "same_gate_as_C2": True,
                         "activation_rate_by_split": {
                             split: float(decision.gate_on[bank.indices(split)].mean())
                             for split in ("train", "validation", "test")}},
        "training_report_path": str(args.out / "training_report.json"),
        "slice_metrics_path": str(args.out / "slice_metrics.json"),
        "gate_accounting_path": str(args.out / "gate_accounting.json"),
        "preserved": ["C2 architecture", "C2 gate", "C2 thresholds", "C2 optimizer and schedule",
                      "C2 epoch/batch/seed", "whole-frame validation selection", "H0", "Frame Pose Core"],
        "intervention": "gate-aligned target-valid coordinate_only_v1 supervision domain only",
    }
    write_json(args.out / "experiment_report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
