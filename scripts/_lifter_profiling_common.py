"""Shared setup for the temporal-lifter performance diagnostic scripts.

Diagnostic-only: never imported by ``train()`` or any production training
code. ``profile_temporal_lifter_training.py`` (stage-timing),
``profile_temporal_lifter_kernels.py`` (kernel-level torch.profiler), and
``benchmark_torch_compile_candidate.py`` (docs/20 eager-vs-compiled
verdict) all build their benchmark state and step function from the
single ``setup``/``build_step_callable`` here, so no diagnostic script can
silently drift onto a different frozen configuration or a differently
constructed training step.
"""

from __future__ import annotations

import resource
import subprocess
import threading
from time import perf_counter
from typing import Any

import numpy as np

from training.temporal_lifter import (
    TrainingConfig, _arrays, _augment_inputs, _model, _source_balanced_permutation, _supervision_loss,
)

# A9's exact frozen benchmark configuration (docs/10, docs/19, docs/20) --
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


class GpuSampler:
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


def percentiles(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {"mean": float(array.mean()), "median": float(np.percentile(array, 50)),
            "p95": float(np.percentile(array, 95)), "min": float(array.min()), "max": float(array.max())}


def process_cpu_seconds() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return usage.ru_utime + usage.ru_stime


def host_rss_mb() -> float:
    # ru_maxrss is KB on Linux (the target training container), bytes on macOS.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def setup(torch, nn, config: TrainingConfig, dataset, device):
    """Build the exact GPU-resident state train() builds, once per pass/run
    so each starts from the same deterministic initialization/augmentation
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


def build_forward_loss_callable(torch, state, config: TrainingConfig, use_compile: bool):
    """Return ``forward_loss(windows, target, mask) -> (prediction, loss)``.

    docs/20 Section 4's compile candidate scope: compile the model-forward +
    existing supervision-loss computation as one graph, while backward and
    the optimizer step stay eager (built separately by the caller). No
    manual loss fusion/reduction-order change -- the exact same
    ``_supervision_loss`` call, just optionally wrapped by
    ``torch.compile`` with default settings (no backend/mode arguments).
    """
    model = state["model"]

    def _forward_loss(windows, target, mask):
        prediction = model(windows)
        loss = _supervision_loss(torch, prediction, target, mask, config)
        return prediction, loss

    return torch.compile(_forward_loss) if use_compile else _forward_loss


def build_step_callable(torch, state, config: TrainingConfig, forward_loss_fn):
    """Return ``step(batch) -> (prediction, loss)`` matching train()'s exact
    hot-path sequence: zero_grad, gather, autocast forward+loss (via
    ``forward_loss_fn`` -- eager or compiled), backward, scaler step/update.
    Identical for both paths except which ``forward_loss_fn`` is passed in.
    """
    optimizer, scaler = state["optimizer"], state["scaler"]
    epoch_inputs, offset_tensor, y, valid_tensor = state["epoch_inputs"], state["offset_tensor"], state["y"], state["valid_tensor"]
    amp_enabled = state["amp_enabled"]

    def step(batch):
        optimizer.zero_grad(set_to_none=True)
        windows = epoch_inputs[offset_tensor[batch]]
        target = y[batch]
        mask = valid_tensor[batch]
        with torch.amp.autocast("cuda", enabled=amp_enabled):
            prediction, loss = forward_loss_fn(windows, target, mask)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        return prediction, loss

    return step
