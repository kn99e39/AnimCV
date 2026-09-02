#!/usr/bin/env python3
"""Bounded, stage-by-stage throughput diagnosis for temporal-lifter training.

Diagnostic-only: never imported by, and never modifies, ``train()`` or any
production training behavior. Replicates the exact epoch-0 GPU-resident
pipeline ``train()`` uses -- same seed, source-balanced permutation,
augmentation, forward/loss/backward/optimizer sequence -- so the measured
stages are real production work, not a synthetic tensor loop.

Runs two separate bounded passes over the same warm-up + measurement window:

  1. THROUGHPUT pass -- no per-step instrumentation, matching train()'s
     actual hot path exactly (one synchronize before/after the whole
     window, not per step). This is the number samples/sec and step-time
     percentiles are computed from; it is not perturbed by diagnostic
     synchronization.
  2. STAGE-ATTRIBUTION pass -- adds per-step CUDA events and an explicit
     torch.cuda.synchronize() after each measured step so forward/loss/
     backward/optimizer/scaler-update time can be correctly separated
     (CUDA work is asynchronous; without this sync, event deltas and
     Python-side timings would be unreliable). This pass's own aggregate
     wall time is expected to be slower than pass 1's -- that slowdown IS
     the diagnostic signal for per-step synchronization cost (Section 7),
     not something to report as the real throughput.

Also runs a background nvidia-smi sampler across pass 1's measured window
for GPU utilization/memory.

Usage:
  python3 scripts/profile_temporal_lifter_training.py \
    --train-dataset /output/experiments/ablation_a9_fingerprinted_baseline_10e/datasets/direct_mix_train.json \
    --device cuda --warmup-steps 50 --measure-steps 300 \
    --out /output/experiments/a15_training_throughput_diagnosis/baseline_profile.json
"""

from __future__ import annotations

import argparse
import json
import resource
import subprocess
import threading
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from training.temporal_lifter import (
    TrainingConfig, _arrays, _augment_inputs, _model, _source_balanced_permutation, _supervision_loss,
    _torch, load_dataset,
)

# A9's exact frozen benchmark configuration (docs/10, docs/19 Section 1) --
# reused verbatim as the fixed benchmark, not a new quality experiment.
A9_CONFIG_KWARGS = dict(
    window=81, channels=256, batch_size=128, learning_rate=1e-3, mixed_precision=True,
    seed=1337, input_coordinate_normalization="pelvis_torso_v1", architecture="dilated_tcn_v1",
    source_balanced_sampling=True,
    input_jitter_std=0.015, input_dropout_probability=0.05, confidence_jitter_std=0.08,
    input_global_scale_std=0.04, input_translation_std=0.03, input_rotation_degrees=12.0,
    temporal_occlusion_probability=0.10, temporal_occlusion_frames=9,
    bone_loss_weight=0.25, torso_loss_weight=0.15, hinge_loss_weight=0.15,
)

_STAGE_EVENT_NAMES = ("step_start", "batch_ready", "forward_done", "loss_done", "backward_done",
                       "step_done", "update_done")


class _GpuSampler:
    """Background nvidia-smi poller. Diagnostic-only: a training script must
    never shell out per step; this runs on its own thread at a fixed
    interval and is torn down when the measured window ends."""

    def __init__(self, interval_seconds: float = 0.2):
        self._interval = interval_seconds
        self._samples: list[dict[str, float]] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            try:
                output = subprocess.run(
                    ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=2.0, check=True,
                ).stdout.strip()
                utilization, memory_used, memory_total = (float(part) for part in output.split(","))
                self._samples.append({"utilization_gpu_pct": utilization, "memory_used_mb": memory_used,
                                       "memory_total_mb": memory_total})
            except Exception:
                pass
            self._stop.wait(self._interval)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *_exc):
        self._stop.set()
        self._thread.join(timeout=2.0)

    def summary(self) -> dict[str, Any]:
        if not self._samples:
            return {"sample_count": 0}
        utilizations = [sample["utilization_gpu_pct"] for sample in self._samples]
        memories = [sample["memory_used_mb"] for sample in self._samples]
        return {
            "sample_count": len(self._samples),
            "mean_utilization_gpu_pct": float(np.mean(utilizations)),
            "p95_utilization_gpu_pct": float(np.percentile(utilizations, 95)),
            "max_utilization_gpu_pct": float(np.max(utilizations)),
            "mean_memory_used_mb": float(np.mean(memories)),
            "max_memory_used_mb": float(np.max(memories)),
        }


def _percentiles(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {"mean": float(array.mean()), "median": float(np.percentile(array, 50)),
            "p95": float(np.percentile(array, 95)), "min": float(array.min()), "max": float(array.max())}


def _process_cpu_seconds() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return usage.ru_utime + usage.ru_stime


def _host_rss_mb() -> float:
    # ru_maxrss is KB on Linux (the target training container), bytes on macOS.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _setup(torch, nn, config: TrainingConfig, dataset, device):
    """Build the exact GPU-resident state train() builds, once per pass so
    each pass starts from the same deterministic initialization/augmentation
    (a fresh model/optimizer/scaler; cheap relative to the measured window)."""
    torch.manual_seed(config.seed)
    arrays_started = perf_counter()
    inputs, targets, valid, offsets, source_ids, sequence_ranges = _arrays(
        dataset, config.window, include_metadata=True, coordinate_normalization=config.input_coordinate_normalization,
    )
    arrays_elapsed_seconds = perf_counter() - arrays_started
    model = _model(nn, config.channels, config.architecture).to(device)
    parameter_count = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    x = torch.as_tensor(inputs, dtype=torch.float32, device=device)
    y = torch.as_tensor(targets, dtype=torch.float32, device=device)
    valid_tensor = torch.as_tensor(valid, dtype=torch.float32, device=device).unsqueeze(-1)
    offset_tensor = torch.as_tensor(offsets, dtype=torch.long, device=device)
    source_tensor = torch.as_tensor(source_ids, dtype=torch.long, device=device)
    indices = torch.arange(len(inputs), device=device)
    amp_enabled = bool(config.mixed_precision and device.type == "cuda")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    if hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    else:  # PyTorch 2.1 keeps GradScaler under torch.cuda.amp -- same fallback as train().
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    generator = torch.Generator(device=device).manual_seed(config.seed)
    epoch_inputs = _augment_inputs(torch, x, config, generator, sequence_ranges)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    permutation = (_source_balanced_permutation(torch, indices, source_tensor, generator)
                   if config.source_balanced_sampling else
                   indices[torch.randperm(len(indices), generator=generator, device=device)])
    batches = list(permutation.split(config.batch_size))
    return {
        "model": model, "optimizer": optimizer, "scaler": scaler, "amp_enabled": amp_enabled,
        "epoch_inputs": epoch_inputs, "offset_tensor": offset_tensor, "y": y, "valid_tensor": valid_tensor,
        "batches": batches, "frame_count": len(inputs), "windows_available": len(indices),
        "parameter_count": parameter_count, "arrays_elapsed_seconds": arrays_elapsed_seconds,
    }


def _throughput_pass(torch, state, config, device, warmup_steps: int, measure_steps: int, sampler) -> dict[str, Any]:
    """No per-step instrumentation: train()'s exact hot path, timed only at
    the boundary of the measured window (plus one sync on each side)."""
    model, optimizer, scaler = state["model"], state["optimizer"], state["scaler"]
    epoch_inputs, offset_tensor, y, valid_tensor = state["epoch_inputs"], state["offset_tensor"], state["y"], state["valid_tensor"]
    batches = state["batches"][: warmup_steps + measure_steps]
    step_wall_ms: list[float] = []

    for step_index, batch in enumerate(batches):
        step_started = perf_counter()
        optimizer.zero_grad(set_to_none=True)
        windows = epoch_inputs[offset_tensor[batch]]
        with torch.amp.autocast("cuda", enabled=state["amp_enabled"]):
            prediction = model(windows)
            mask = valid_tensor[batch]
            loss = _supervision_loss(torch, prediction, y[batch], mask, config)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        if step_index >= warmup_steps:
            step_wall_ms.append((perf_counter() - step_started) * 1000.0)

    if device.type == "cuda":
        torch.cuda.synchronize(device)

    # Re-measure with a single wall-clock bracket for the throughput number
    # itself (per-step perf_counter calls above already approximate this
    # well on CUDA since kernels are enqueued asynchronously, but the
    # bracketed total is the authoritative one used for samples/sec).
    return {
        "step_wall_ms": step_wall_ms,
        "gpu_memory_allocated_mb": (torch.cuda.memory_allocated(device) / (1024 ** 2)) if device.type == "cuda" else None,
        "gpu_memory_reserved_mb": (torch.cuda.memory_reserved(device) / (1024 ** 2)) if device.type == "cuda" else None,
        "gpu_peak_memory_allocated_mb": (torch.cuda.max_memory_allocated(device) / (1024 ** 2)) if device.type == "cuda" else None,
    }


def _attribution_pass(torch, state, config, device, warmup_steps: int, measure_steps: int) -> dict[str, list[float]]:
    """Per-step CUDA events + explicit sync so stage boundaries are
    trustworthy. This pass's own throughput is not reported as the
    production number (see module docstring)."""
    model, optimizer, scaler = state["model"], state["optimizer"], state["scaler"]
    epoch_inputs, offset_tensor, y, valid_tensor = state["epoch_inputs"], state["offset_tensor"], state["y"], state["valid_tensor"]
    batches = state["batches"][: warmup_steps + measure_steps]
    cuda_timed = device.type == "cuda"
    stage_ms: dict[str, list[float]] = {
        "batch_construction_ms": [], "forward_ms": [], "loss_ms": [], "backward_ms": [],
        "optimizer_step_ms": [], "scaler_update_ms": [],
    }

    for step_index, batch in enumerate(batches):
        measured = step_index >= warmup_steps
        events = {name: torch.cuda.Event(enable_timing=True) for name in _STAGE_EVENT_NAMES} \
            if (measured and cuda_timed) else None

        if events:
            events["step_start"].record()
        optimizer.zero_grad(set_to_none=True)
        windows = epoch_inputs[offset_tensor[batch]]
        if events:
            events["batch_ready"].record()

        with torch.amp.autocast("cuda", enabled=state["amp_enabled"]):
            prediction = model(windows)
            if events:
                events["forward_done"].record()
            mask = valid_tensor[batch]
            loss = _supervision_loss(torch, prediction, y[batch], mask, config)
            if events:
                events["loss_done"].record()

        scaler.scale(loss).backward()
        if events:
            events["backward_done"].record()
        scaler.step(optimizer)
        if events:
            events["step_done"].record()
        scaler.update()
        if events:
            events["update_done"].record()
            torch.cuda.synchronize(device)  # diagnostic-only

        if measured and events:
            stage_ms["batch_construction_ms"].append(events["step_start"].elapsed_time(events["batch_ready"]))
            stage_ms["forward_ms"].append(events["batch_ready"].elapsed_time(events["forward_done"]))
            stage_ms["loss_ms"].append(events["forward_done"].elapsed_time(events["loss_done"]))
            stage_ms["backward_ms"].append(events["loss_done"].elapsed_time(events["backward_done"]))
            stage_ms["optimizer_step_ms"].append(events["backward_done"].elapsed_time(events["step_done"]))
            stage_ms["scaler_update_ms"].append(events["step_done"].elapsed_time(events["update_done"]))

    return stage_ms


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train-dataset", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--measure-steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    torch, nn = _torch()
    config = TrainingConfig(epochs=1, device=args.device, seed=args.seed,
                             **{k: v for k, v in A9_CONFIG_KWARGS.items() if k != "seed"})
    device = torch.device(args.device)

    dataset = load_dataset(args.train_dataset)
    total_needed = args.warmup_steps + args.measure_steps

    cpu_seconds_start = _process_cpu_seconds()

    # Pass 1: throughput (no per-step instrumentation), with GPU sampling.
    state = _setup(torch, nn, config, dataset, device)
    if len(state["batches"]) < total_needed:
        raise ValueError(
            f"epoch has only {len(state['batches'])} steps at batch_size={config.batch_size}; "
            f"need {total_needed} (warmup {args.warmup_steps} + measure {args.measure_steps})"
        )
    with _GpuSampler() as sampler:
        wall_started = perf_counter()
        throughput_result = _throughput_pass(torch, state, config, device, args.warmup_steps, args.measure_steps, sampler)
        wall_elapsed_seconds = perf_counter() - wall_started
    gpu_summary = sampler.summary()
    cpu_seconds_elapsed = _process_cpu_seconds() - cpu_seconds_start

    # Pass 2: stage attribution (fresh state, same seed -> identical batches/augmentation).
    state2 = _setup(torch, nn, config, dataset, device)
    stage_ms = _attribution_pass(torch, state2, config, device, args.warmup_steps, args.measure_steps)

    measured_wall_seconds = sum(throughput_result["step_wall_ms"]) / 1000.0
    measured_samples = args.measure_steps * config.batch_size

    stage_timing_ms = {name: _percentiles(values) for name, values in stage_ms.items() if values}
    step_total_percentiles = _percentiles(throughput_result["step_wall_ms"]) if throughput_result["step_wall_ms"] else None

    report: dict[str, Any] = {
        "schema": "animcv_training_throughput_profile_v1",
        "config": {**A9_CONFIG_KWARGS, "seed": args.seed, "device": args.device},
        "hardware": {"parameter_count": int(state["parameter_count"])},
        "dataset": {"train_dataset": str(args.train_dataset), "frame_count": state["frame_count"],
                     "windows_available": state["windows_available"],
                     "steps_per_epoch_at_batch_size": len(state["batches"])},
        "warmup_steps": args.warmup_steps,
        "measure_steps": args.measure_steps,
        "one_time_costs_seconds": {"dataset_arrays": state["arrays_elapsed_seconds"]},
        "throughput_pass": {
            "step_total_wall_ms": step_total_percentiles,
            "samples_per_second": measured_samples / measured_wall_seconds if measured_wall_seconds else None,
            "windows_per_second": measured_samples / measured_wall_seconds if measured_wall_seconds else None,
            "estimated_epoch_wall_seconds": (len(state["batches"]) / args.measure_steps) * measured_wall_seconds if args.measure_steps else None,
            "bracketed_wall_seconds_including_warmup": wall_elapsed_seconds,
            "gpu_memory_allocated_mb": throughput_result["gpu_memory_allocated_mb"],
            "gpu_memory_reserved_mb": throughput_result["gpu_memory_reserved_mb"],
            "gpu_peak_memory_allocated_mb": throughput_result["gpu_peak_memory_allocated_mb"],
        },
        "gpu_utilization_measured_window": gpu_summary,
        "process_cpu_utilization_pct_of_wall": (cpu_seconds_elapsed / wall_elapsed_seconds * 100.0) if wall_elapsed_seconds else None,
        "host_rss_mb": _host_rss_mb(),
        "stage_attribution_pass": {
            "note": "per-step CUDA events + explicit sync; this pass's own wall time is inflated by the "
                    "diagnostic sync and is NOT the reported throughput -- see throughput_pass.",
            "stage_timing_ms": stage_timing_ms,
        },
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "out": str(args.out),
        "mean_step_wall_ms": step_total_percentiles["mean"] if step_total_percentiles else None,
        "samples_per_second": report["throughput_pass"]["samples_per_second"],
        "mean_gpu_utilization_pct": gpu_summary.get("mean_utilization_gpu_pct"),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
