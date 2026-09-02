#!/usr/bin/env python3
"""Eager-vs-compiled verdict for the A9 temporal-lifter benchmark (docs/20).

Diagnostic-only: never imported by, and never modifies, ``train()``. Uses
the same frozen A9 benchmark config as every other script in this batch
(``_lifter_profiling_common``). Runs, in order:

  1. NUMERICAL EQUIVALENCE (Section 7) -- from two model copies with
     bitwise-identical initial parameters and the exact same fixed batch,
     compares eager vs compiled forward prediction, total supervision
     loss, the bone/torso/hinge component decomposition (recomputed from
     each path's own ``prediction`` with the same eager helpers, so this
     checks the compiled path's *output*, not a second compiled loss
     graph), and parameter gradients. Reports max abs/relative difference;
     does not assume bitwise equality.
  2. COMPILED REPRODUCIBILITY (Section 8) -- runs the compiled candidate
     twice from the same seed/init/dataset/batch sequence and compares.
  3. SHORT TRAINING A/B (Section 9) -- a bounded real training replay
     (zero_grad/forward/loss/backward/optimizer.step/scaler.update, the
     exact train() sequence) from the same initialization and batch
     sequence, eager vs compiled: loss trajectory, a training-MPJPE-style
     trajectory, and final parameter divergence.
  4. STEADY-STATE THROUGHPUT A/B (Section 10) -- the same warm-up +
     measured-window methodology as profile_temporal_lifter_training.py,
     run once eager and once compiled (compile latency absorbed into the
     warm-up window, not the measured one).

Usage:
  python3 scripts/benchmark_torch_compile_candidate.py \
    --train-dataset /output/experiments/ablation_a9_fingerprinted_baseline_10e/datasets/direct_mix_train.json \
    --device cuda --warmup-steps 50 --measure-steps 300 \
    --short-training-steps 20 \
    --out /output/experiments/a15_training_throughput_diagnosis/compile_candidate_verdict.json
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from time import perf_counter
from typing import Any

from _lifter_profiling_common import A9_CONFIG_KWARGS, build_forward_loss_callable, percentiles, setup
from training.temporal_lifter import BONE_INDICES, TORSO_INDICES, TrainingConfig, _hinge_loss, _torch, _vector_loss, load_dataset


def _component_losses(torch, prediction, target, valid_bool) -> dict[str, float]:
    """Same structural decomposition production code uses, recomputed from
    a given ``prediction`` regardless of whether it came from the eager or
    compiled path -- so this compares outputs, not a second loss graph."""
    bone = _vector_loss(torch, prediction, target, valid_bool, BONE_INDICES, lambda first, second: first - second)
    torso = _vector_loss(torch, prediction, target, valid_bool, TORSO_INDICES, lambda first, second: second - first)
    hinge = _hinge_loss(torch, prediction, target, valid_bool)
    return {"bone": float(bone.item()), "torso": float(torso.item()), "hinge": float(hinge.item())}


def _max_abs_rel_diff(a, b) -> dict[str, float]:
    a64, b64 = a.detach().float(), b.detach().float()
    diff = (a64 - b64).abs()
    max_abs = float(diff.max().item())
    denom = b64.abs().clamp_min(1e-8)
    max_rel = float((diff / denom).max().item())
    return {"max_abs_diff": max_abs, "max_rel_diff": max_rel, "exactly_equal": bool(a64.equal(b64))}


def numerical_equivalence(torch, nn, config, dataset, device) -> dict[str, Any]:
    state = setup(torch, nn, config, dataset, device)
    batch = state["batches"][0]
    windows = state["epoch_inputs"][state["offset_tensor"][batch]]
    target = state["y"][batch]
    mask = state["valid_tensor"][batch]
    valid_bool = mask.squeeze(-1).bool()

    model_eager = state["model"]
    model_compiled = copy.deepcopy(model_eager)  # identical initial parameters, independent graphs

    def run(model, use_compile):
        for parameter in model.parameters():
            parameter.grad = None
        forward_loss_fn = build_forward_loss_callable(torch, model, config, use_compile)
        with torch.amp.autocast("cuda", enabled=state["amp_enabled"]):
            prediction, loss = forward_loss_fn(windows, target, mask)
        loss.backward()
        grads = {name: parameter.grad.detach().clone() for name, parameter in model.named_parameters()
                 if parameter.grad is not None}
        return prediction.detach(), loss.detach(), grads

    eager_prediction, eager_loss, eager_grads = run(model_eager, False)
    compiled_prediction, compiled_loss, compiled_grads = run(model_compiled, True)

    prediction_diff = _max_abs_rel_diff(compiled_prediction, eager_prediction)
    loss_diff = _max_abs_rel_diff(compiled_loss.reshape(1), eager_loss.reshape(1))
    eager_components = _component_losses(torch, eager_prediction, target, valid_bool)
    compiled_components = _component_losses(torch, compiled_prediction, target, valid_bool)
    component_diffs = {
        name: abs(compiled_components[name] - eager_components[name]) for name in eager_components
    }

    missing_grads = sorted(set(eager_grads) ^ set(compiled_grads))
    grad_diffs = {
        name: _max_abs_rel_diff(compiled_grads[name], eager_grads[name])
        for name in eager_grads if name in compiled_grads
    }
    max_grad_abs = max((entry["max_abs_diff"] for entry in grad_diffs.values()), default=None)
    max_grad_rel = max((entry["max_rel_diff"] for entry in grad_diffs.values()), default=None)

    return {
        "prediction_diff": prediction_diff,
        "loss_diff": {**loss_diff, "eager_loss": float(eager_loss.item()), "compiled_loss": float(compiled_loss.item())},
        "eager_component_losses": eager_components,
        "compiled_component_losses": compiled_components,
        "component_loss_abs_diff": component_diffs,
        "gradient_missing_parameter_names": missing_grads,
        "gradient_max_abs_diff": max_grad_abs,
        "gradient_max_rel_diff": max_grad_rel,
        "gradients_finite_eager": all(bool(torch.isfinite(g).all()) for g in eager_grads.values()),
        "gradients_finite_compiled": all(bool(torch.isfinite(g).all()) for g in compiled_grads.values()),
        # torch.testing.assert_close's own default float16 tolerance is
        # rtol=1e-3/atol=1e-5; autocast runs this model's matmul/conv layers
        # in float16, so differences at that scale are expected from
        # compiler-fused reduction order, not a correctness defect. We
        # report the measured value rather than assume it fits.
        "tolerance_reference": {"dtype": "float16 (autocast)", "torch_testing_default_rtol": 1e-3, "torch_testing_default_atol": 1e-5},
    }


def compiled_reproducibility(torch, nn, config, dataset, device) -> dict[str, Any]:
    def run():
        state = setup(torch, nn, config, dataset, device)
        batch = state["batches"][0]
        windows = state["epoch_inputs"][state["offset_tensor"][batch]]
        target = state["y"][batch]
        mask = state["valid_tensor"][batch]
        forward_loss_fn = build_forward_loss_callable(torch, state["model"], config, use_compile=True)
        with torch.amp.autocast("cuda", enabled=state["amp_enabled"]):
            prediction, loss = forward_loss_fn(windows, target, mask)
        return prediction.detach(), float(loss.item())

    first_prediction, first_loss = run()
    second_prediction, second_loss = run()
    prediction_diff = _max_abs_rel_diff(second_prediction, first_prediction)
    return {
        "first_loss": first_loss, "second_loss": second_loss,
        "loss_exactly_equal": first_loss == second_loss,
        "prediction_diff": prediction_diff,
    }


def _short_training_replay(torch, nn, config, dataset, device, steps: int, use_compile: bool) -> dict[str, Any]:
    state = setup(torch, nn, config, dataset, device)
    model, optimizer, scaler = state["model"], state["optimizer"], state["scaler"]
    epoch_inputs, offset_tensor, y, valid_tensor = state["epoch_inputs"], state["offset_tensor"], state["y"], state["valid_tensor"]
    forward_loss_fn = build_forward_loss_callable(torch, model, config, use_compile)

    losses, mpjpe_mm = [], []
    for batch in state["batches"][:steps]:
        optimizer.zero_grad(set_to_none=True)
        windows = epoch_inputs[offset_tensor[batch]]
        target = y[batch]
        mask = valid_tensor[batch]
        with torch.amp.autocast("cuda", enabled=state["amp_enabled"]):
            prediction, loss = forward_loss_fn(windows, target, mask)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        losses.append(float(loss.item()))
        with torch.no_grad():
            errors = torch.linalg.vector_norm(prediction.float() - target, dim=-1)
            valid_float = mask.squeeze(-1)
            mpjpe_mm.append(float((errors * valid_float).sum().item() / valid_float.sum().clamp_min(1.0).item() * 1000))

    final_parameters = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    return {"loss_trajectory": losses, "mpjpe_mm_trajectory": mpjpe_mm, "final_parameters": final_parameters}


def short_training_ab(torch, nn, config, dataset, device, steps: int) -> dict[str, Any]:
    eager = _short_training_replay(torch, nn, config, dataset, device, steps, use_compile=False)
    compiled = _short_training_replay(torch, nn, config, dataset, device, steps, use_compile=True)

    parameter_diffs = {
        name: _max_abs_rel_diff(compiled["final_parameters"][name], eager["final_parameters"][name])
        for name in eager["final_parameters"] if name in compiled["final_parameters"]
    }
    max_parameter_abs = max((entry["max_abs_diff"] for entry in parameter_diffs.values()), default=None)
    max_parameter_rel = max((entry["max_rel_diff"] for entry in parameter_diffs.values()), default=None)

    return {
        "steps": steps,
        "eager_loss_trajectory": eager["loss_trajectory"],
        "compiled_loss_trajectory": compiled["loss_trajectory"],
        "eager_mpjpe_mm_trajectory": eager["mpjpe_mm_trajectory"],
        "compiled_mpjpe_mm_trajectory": compiled["mpjpe_mm_trajectory"],
        "final_loss_abs_diff": abs(eager["loss_trajectory"][-1] - compiled["loss_trajectory"][-1]),
        "final_mpjpe_mm_abs_diff": abs(eager["mpjpe_mm_trajectory"][-1] - compiled["mpjpe_mm_trajectory"][-1]),
        "final_parameter_max_abs_diff": max_parameter_abs,
        "final_parameter_max_rel_diff": max_parameter_rel,
    }


def _throughput_run(torch, nn, config, dataset, device, warmup_steps: int, measure_steps: int, use_compile: bool) -> dict[str, Any]:
    state = setup(torch, nn, config, dataset, device)
    model, optimizer, scaler = state["model"], state["optimizer"], state["scaler"]
    epoch_inputs, offset_tensor, y, valid_tensor = state["epoch_inputs"], state["offset_tensor"], state["y"], state["valid_tensor"]
    forward_loss_fn = build_forward_loss_callable(torch, model, config, use_compile)
    batches = state["batches"][: warmup_steps + measure_steps]

    first_call_seconds = None
    first_call_started = perf_counter()
    for index, batch in enumerate(batches[:warmup_steps]):
        optimizer.zero_grad(set_to_none=True)
        windows = epoch_inputs[offset_tensor[batch]]
        with torch.amp.autocast("cuda", enabled=state["amp_enabled"]):
            prediction, loss = forward_loss_fn(windows, y[batch], valid_tensor[batch])
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        if index == 0 and device.type == "cuda":
            torch.cuda.synchronize(device)  # isolates first-call (compile, if any) latency
            first_call_seconds = perf_counter() - first_call_started
    if device.type != "cuda":
        first_call_seconds = perf_counter() - first_call_started
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    step_wall_ms: list[float] = []
    for batch in batches[warmup_steps:]:
        step_started = perf_counter()
        optimizer.zero_grad(set_to_none=True)
        windows = epoch_inputs[offset_tensor[batch]]
        with torch.amp.autocast("cuda", enabled=state["amp_enabled"]):
            prediction, loss = forward_loss_fn(windows, y[batch], valid_tensor[batch])
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        step_wall_ms.append((perf_counter() - step_started) * 1000.0)
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    measured_wall_seconds = sum(step_wall_ms) / 1000.0
    measured_samples = measure_steps * config.batch_size
    return {
        "step_total_wall_ms": percentiles(step_wall_ms) if step_wall_ms else None,
        "samples_per_second": measured_samples / measured_wall_seconds if measured_wall_seconds else None,
        "gpu_peak_memory_allocated_mb": (torch.cuda.max_memory_allocated(device) / (1024 ** 2)) if device.type == "cuda" else None,
        "first_warmup_call_seconds": first_call_seconds,
    }


def steady_state_throughput_ab(torch, nn, config, dataset, device, warmup_steps: int, measure_steps: int) -> dict[str, Any]:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    eager = _throughput_run(torch, nn, config, dataset, device, warmup_steps, measure_steps, use_compile=False)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    compiled = _throughput_run(torch, nn, config, dataset, device, warmup_steps, measure_steps, use_compile=True)

    eager_sps = eager["samples_per_second"]
    compiled_sps = compiled["samples_per_second"]
    delta_pct = ((compiled_sps - eager_sps) / eager_sps * 100.0) if eager_sps else None
    return {"eager": eager, "compiled": compiled, "samples_per_second_delta_pct": delta_pct}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train-dataset", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--measure-steps", type=int, default=300)
    parser.add_argument("--short-training-steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    torch, nn = _torch()
    config = TrainingConfig(epochs=1, device=args.device, seed=args.seed,
                             **{k: v for k, v in A9_CONFIG_KWARGS.items() if k != "seed"})
    device = torch.device(args.device)
    dataset = load_dataset(args.train_dataset)

    report: dict[str, Any] = {
        "schema": "animcv_compile_candidate_verdict_v1",
        "config": {**A9_CONFIG_KWARGS, "seed": args.seed, "device": args.device},
    }

    report["numerical_equivalence"] = numerical_equivalence(torch, nn, config, dataset, device)
    report["compiled_reproducibility"] = compiled_reproducibility(torch, nn, config, dataset, device)
    report["short_training_ab"] = short_training_ab(torch, nn, config, dataset, device, args.short_training_steps)
    report["steady_state_throughput_ab"] = steady_state_throughput_ab(
        torch, nn, config, dataset, device, args.warmup_steps, args.measure_steps,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "out": str(args.out),
        "gradient_max_rel_diff": report["numerical_equivalence"]["gradient_max_rel_diff"],
        "compiled_reproducible": report["compiled_reproducibility"]["loss_exactly_equal"],
        "samples_per_second_delta_pct": report["steady_state_throughput_ab"]["samples_per_second_delta_pct"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
