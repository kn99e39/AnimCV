#!/usr/bin/env python3
"""Kernel-level torch.profiler diagnosis for temporal-lifter training.

Diagnostic-only: never imported by, and never modifies, ``train()``. Uses
the same frozen A9 benchmark state (``_lifter_profiling_common.setup``) as
``profile_temporal_lifter_training.py``. Profiles a small bounded
steady-state window (after an un-profiled warm-up) with
``torch.profiler.profile(activities=[CPU, CUDA])``, using
``record_function`` regions around batch construction / forward / loss /
backward / optimizer step so each stage's CPU and CUDA time can be read
directly from ``prof.key_averages()``.

Also reports, from the profiler's raw Kineto event stream:

  - per-CUDA-kernel-instance duration distribution (count, sum, median,
    P90/P95, min/max, and the fraction of kernels under a "very short"
    threshold) -- this is the direct evidence for or against the
    fragmented-small-kernel hypothesis (docs/20 Section 3), not an
    inference from GPU utilization alone;
  - the dominant CUDA kernels and dominant CPU (launch/framework) ops by
    cumulative self time.

With ``--compile``, wraps the model-forward + supervision-loss computation
in ``torch.compile`` (default settings, no backend/mode arguments) via
``_lifter_profiling_common.build_forward_loss_callable`` -- backward and
the optimizer step remain eager either way -- so the exact same profiling
methodology can be reused for the eager baseline (Section 2) and the
post-compile comparison (Section 11). A ``torch._dynamo.explain`` dry run
(outside the profiled window) reports graph count / graph breaks /
break reasons for the compiled path.

Usage:
  python3 scripts/profile_temporal_lifter_kernels.py \
    --train-dataset /output/experiments/ablation_a9_fingerprinted_baseline_10e/datasets/direct_mix_train.json \
    --device cuda --warmup-steps 20 --measure-steps 20 \
    --out /output/experiments/a15_training_throughput_diagnosis/kernel_profile_eager.json

  python3 scripts/profile_temporal_lifter_kernels.py ... --compile \
    --out /output/experiments/a15_training_throughput_diagnosis/kernel_profile_compiled.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from _lifter_profiling_common import A9_CONFIG_KWARGS, build_forward_loss_callable, build_step_callable, setup
from training.temporal_lifter import TrainingConfig, _torch, load_dataset

_STAGE_NAMES_EAGER = ("batch_construction", "forward", "loss", "backward", "optimizer_step")
_STAGE_NAMES_COMPILED = ("batch_construction", "forward_loss_compiled", "backward", "optimizer_step")
_SHORT_KERNEL_THRESHOLD_US = 10.0


def _build_regioned_step(torch, state, config, use_compile: bool):
    """A step callable identical to build_step_callable's sequence, but with
    torch.profiler.record_function regions around each stage so
    key_averages() can attribute CPU/CUDA time per stage. Kept separate
    from build_step_callable (used by the un-profiled warm-up and by other
    scripts) so record_function overhead never leaks into a non-profiled
    measurement."""
    from torch.profiler import record_function

    optimizer, scaler = state["optimizer"], state["scaler"]
    epoch_inputs, offset_tensor, y, valid_tensor = state["epoch_inputs"], state["offset_tensor"], state["y"], state["valid_tensor"]
    amp_enabled = state["amp_enabled"]
    forward_loss_fn = build_forward_loss_callable(torch, state["model"], config, use_compile)

    def step(batch):
        with record_function("batch_construction"):
            optimizer.zero_grad(set_to_none=True)
            windows = epoch_inputs[offset_tensor[batch]]
            target = y[batch]
            mask = valid_tensor[batch]
        with torch.amp.autocast("cuda", enabled=amp_enabled):
            if use_compile:
                with record_function("forward_loss_compiled"):
                    prediction, loss = forward_loss_fn(windows, target, mask)
            else:
                with record_function("forward"):
                    prediction = state["model"](windows)
                with record_function("loss"):
                    from training.temporal_lifter import _supervision_loss
                    loss = _supervision_loss(torch, prediction, target, mask, config)
        with record_function("backward"):
            scaler.scale(loss).backward()
        with record_function("optimizer_step"):
            scaler.step(optimizer)
            scaler.update()
        return prediction, loss

    return step


def _dynamo_graph_report(torch, state, config) -> dict[str, Any]:
    """One dry-run torch._dynamo.explain call on real batch shapes, outside
    the profiled window (Section 6): graph count, graph breaks, and break
    reasons for the exact compiled forward+loss callable."""
    forward_loss_fn = build_forward_loss_callable(torch, state["model"], config, use_compile=False)  # eager fn to explain
    batch = state["batches"][0]
    windows = state["epoch_inputs"][state["offset_tensor"][batch]]
    target = state["y"][batch]
    mask = state["valid_tensor"][batch]
    try:
        explanation = torch._dynamo.explain(forward_loss_fn)(windows, target, mask)
        return {
            "graph_count": explanation.graph_count,
            "graph_break_count": explanation.graph_break_count,
            "break_reasons": [str(reason) for reason in explanation.break_reasons],
        }
    except Exception as exc:  # diagnostic-only: never let explain() failure abort the profile
        return {"error": str(exc)}


def _event_duration_us(event) -> float:
    """torch 2.1.2's raw Kineto event exposes duration_us(); some other
    torch builds expose only duration_ns(). Tolerate both rather than
    pinning this diagnostic script to one exact torch release."""
    if hasattr(event, "duration_us"):
        return float(event.duration_us())
    return float(event.duration_ns()) / 1000.0


def _event_name(event) -> str:
    return event.name() if callable(getattr(event, "name", None)) else event.name


def _event_device_type_name(event) -> str:
    device_type = event.device_type() if callable(getattr(event, "device_type", None)) else event.device_type
    return device_type.name


def _kernel_duration_stats(kineto_events) -> dict[str, Any]:
    import numpy as np

    durations = [_event_duration_us(event) for event in kineto_events if _event_device_type_name(event) == "CUDA"]
    if not durations:
        return {"count": 0}
    array = np.asarray(durations, dtype=np.float64)
    short_count = int((array < _SHORT_KERNEL_THRESHOLD_US).sum())
    return {
        "count": len(durations),
        "sum_us": float(array.sum()),
        "mean_us": float(array.mean()),
        "median_us": float(np.percentile(array, 50)),
        "p90_us": float(np.percentile(array, 90)),
        "p95_us": float(np.percentile(array, 95)),
        "min_us": float(array.min()),
        "max_us": float(array.max()),
        "short_kernel_threshold_us": _SHORT_KERNEL_THRESHOLD_US,
        "short_kernel_count": short_count,
        "short_kernel_fraction": short_count / len(durations),
    }


def _dominant_ops(kineto_events, device_type_name: str, top_n: int = 10) -> list[dict[str, Any]]:
    """Aggregate raw kineto events by op name (cumulative self time), not
    the record_function-labeled stage regions -- this answers "which
    individual CUDA kernels / CPU framework ops actually dominate", where
    key_averages() by itself only answers it at the stage-region grain."""
    totals: dict[str, list[float]] = {}
    for event in kineto_events:
        if _event_device_type_name(event) != device_type_name:
            continue
        totals.setdefault(_event_name(event), []).append(_event_duration_us(event))
    ranked = sorted(totals.items(), key=lambda item: -sum(item[1]))[:top_n]
    return [{"name": name, "count": len(values), "total_us": float(sum(values)), "mean_us": float(sum(values) / len(values))}
            for name, values in ranked]


def _profile_window(torch, step_fn, batches: list) -> dict[str, Any]:
    from torch.profiler import ProfilerActivity, profile

    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)

    with profile(activities=activities, record_shapes=False, profile_memory=False, with_stack=False) as prof:
        for batch in batches:
            step_fn(batch)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    averages = prof.key_averages()
    stage_stats = {
        row.key: {
            "count": row.count,
            "cpu_time_total_us": float(row.cpu_time_total),
            "self_cpu_time_total_us": float(row.self_cpu_time_total),
            "cuda_time_total_us": float(getattr(row, "cuda_time_total", 0.0)),
            "self_cuda_time_total_us": float(getattr(row, "self_cuda_time_total", 0.0)),
        }
        for row in averages if row.key in (_STAGE_NAMES_EAGER + _STAGE_NAMES_COMPILED)
    }

    kineto_events = prof.profiler.kineto_results.events() if prof.profiler is not None else []
    cpu_op_count = sum(1 for event in kineto_events if _event_device_type_name(event) == "CPU")

    return {
        "stage_stats_us": stage_stats,
        "cpu_op_count": cpu_op_count,
        "cuda_kernel_duration_stats": _kernel_duration_stats(kineto_events),
        "dominant_cuda_kernels": _dominant_ops(kineto_events, "CUDA"),
        "dominant_cpu_ops": _dominant_ops(kineto_events, "CPU"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train-dataset", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--measure-steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    torch, nn = _torch()
    config = TrainingConfig(epochs=1, device=args.device, seed=args.seed,
                             **{k: v for k, v in A9_CONFIG_KWARGS.items() if k != "seed"})
    device = torch.device(args.device)

    dataset = load_dataset(args.train_dataset)
    total_needed = args.warmup_steps + args.measure_steps
    state = setup(torch, nn, config, dataset, device)
    if len(state["batches"]) < total_needed:
        raise ValueError(
            f"epoch has only {len(state['batches'])} steps at batch_size={config.batch_size}; "
            f"need {total_needed} (warmup {args.warmup_steps} + measure {args.measure_steps})"
        )

    dynamo_report = _dynamo_graph_report(torch, state, config) if args.compile else None

    # Warm-up (un-profiled): also where compile's one-time first-call
    # latency is paid, kept out of both the measured throughput and the
    # profiled trace (Section 6).
    from time import perf_counter
    forward_loss_fn = build_forward_loss_callable(torch, state["model"], config, use_compile=args.compile)
    warmup_step_fn = build_step_callable(torch, state, config, forward_loss_fn)
    first_call_started = perf_counter()
    warmup_step_fn(state["batches"][0])
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    first_call_seconds = perf_counter() - first_call_started
    for batch in state["batches"][1: args.warmup_steps]:
        warmup_step_fn(batch)
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    regioned_step_fn = _build_regioned_step(torch, state, config, args.compile)
    measured_batches = state["batches"][args.warmup_steps: args.warmup_steps + args.measure_steps]
    profile_result = _profile_window(torch, regioned_step_fn, measured_batches)

    report: dict[str, Any] = {
        "schema": "animcv_training_kernel_profile_v1",
        "config": {**A9_CONFIG_KWARGS, "seed": args.seed, "device": args.device},
        "compile": args.compile,
        "warmup_steps": args.warmup_steps,
        "measure_steps": args.measure_steps,
        "first_warmup_call_seconds": first_call_seconds,
        "dynamo_graph_report": dynamo_report,
        **profile_result,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "out": str(args.out),
        "compile": args.compile,
        "cuda_kernel_count": profile_result["cuda_kernel_duration_stats"].get("count"),
        "cuda_kernel_median_us": profile_result["cuda_kernel_duration_stats"].get("median_us"),
        "first_warmup_call_seconds": first_call_seconds,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
