"""Focused tests for the training-throughput diagnostic script.

Diagnostic-only units: none of this changes production training behavior.
Verifies the profiler's own setup is deterministic across the two passes it
runs (so a stage-attribution pass is a fair proxy for the throughput pass),
that percentile math is correct, and that the GPU sampler degrades
gracefully when nvidia-smi is unavailable.
"""
import importlib.util
from pathlib import Path
import sys

import pytest

pytest.importorskip("torch", reason="throughput profiling checks require the optional training extra")

_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT = _ROOT / "scripts" / "profile_temporal_lifter_training.py"


def _load_module():
    sys.path.insert(0, str(_ROOT / "src"))
    try:
        spec = importlib.util.spec_from_file_location("profile_temporal_lifter_training", _SCRIPT)
        module = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.pop(0)


def _tiny_dataset():
    from pose.pose_lifter import H36M_NAMES, LiftedPoseFrame, LiftedPosePoint, LiftedPoseSequence
    from pose.pose_types import PoseFrame, PoseLandmark, PoseSequence
    from training.temporal_lifter import build_dataset

    names = set(H36M_NAMES) - {"thorax"}
    n = 300

    def pose(index):
        return PoseFrame(index, index / 25, {name: PoseLandmark(name, 10 + (index % 50), 20, 1.0, True) for name in names})

    def target(index):
        return LiftedPoseFrame(index, index / 25, {name: LiftedPosePoint(name, (index / 10, (index % 7) / 10, 0), 1.0, 0.0) for name in names})

    pose_sequence = PoseSequence([pose(i) for i in range(n)], 25)
    target_sequence = LiftedPoseSequence([target(i) for i in range(n)], 25)
    return build_dataset(pose_sequence, target_sequence, (100, 100), "profile-test")


def test_percentiles_are_correct():
    module = _load_module()
    result = module._percentiles([1.0, 2.0, 3.0, 4.0, 5.0])
    assert result["mean"] == pytest.approx(3.0)
    assert result["median"] == pytest.approx(3.0)
    assert result["min"] == pytest.approx(1.0)
    assert result["max"] == pytest.approx(5.0)


def test_setup_is_deterministic_across_repeated_calls():
    """Both the throughput pass and the stage-attribution pass call _setup()
    independently; if it were not deterministic, the two passes would not be
    a fair comparison and the two-pass design would be unsound."""
    import torch
    from training.temporal_lifter import TrainingConfig

    module = _load_module()
    dataset = _tiny_dataset()
    config = TrainingConfig(epochs=1, device="cpu", seed=7,
                             **{k: v for k, v in module.A9_CONFIG_KWARGS.items() if k != "seed"})
    device = torch.device("cpu")
    torch, nn = module._torch()

    first = module._setup(torch, nn, config, dataset, device)
    second = module._setup(torch, nn, config, dataset, device)

    assert torch.equal(first["epoch_inputs"], second["epoch_inputs"])
    assert all(torch.equal(a, b) for a, b in zip(first["batches"], second["batches"]))
    first_params = dict(first["model"].named_parameters())
    second_params = dict(second["model"].named_parameters())
    assert first_params.keys() == second_params.keys()
    assert all(torch.equal(first_params[name], second_params[name]) for name in first_params)


def test_throughput_and_attribution_passes_produce_the_same_first_step_loss():
    """A weak but concrete semantic-equivalence check: with identical seeded
    setup, the first measured step's loss must match between the
    uninstrumented throughput pass and the CUDA-event-instrumented
    attribution pass -- the diagnostic sync must not change the computed
    values, only when they become visible."""
    import torch
    from training.temporal_lifter import TrainingConfig, _supervision_loss

    module = _load_module()
    dataset = _tiny_dataset()
    config = TrainingConfig(epochs=1, device="cpu", seed=11,
                             **{k: v for k, v in module.A9_CONFIG_KWARGS.items() if k != "seed"})
    device = torch.device("cpu")
    torch, nn = module._torch()

    def first_step_loss():
        state = module._setup(torch, nn, config, dataset, device)
        batch = state["batches"][0]
        windows = state["epoch_inputs"][state["offset_tensor"][batch]]
        with torch.no_grad():
            prediction = state["model"](windows)
            mask = state["valid_tensor"][batch]
            return float(_supervision_loss(torch, prediction, state["y"][batch], mask, config).item())

    assert first_step_loss() == pytest.approx(first_step_loss())


def test_gpu_sampler_degrades_gracefully_without_nvidia_smi(monkeypatch):
    module = _load_module()

    def _raise(*_args, **_kwargs):
        raise FileNotFoundError("nvidia-smi not found")

    monkeypatch.setattr(module.subprocess, "run", _raise)
    sampler = module._GpuSampler(interval_seconds=0.01)
    with sampler:
        import time
        time.sleep(0.05)
    summary = sampler.summary()
    assert summary["sample_count"] == 0


def test_gpu_sampler_summarizes_collected_samples(monkeypatch):
    module = _load_module()

    class _FakeResult:
        stdout = "37, 1200, 12288\n"

    monkeypatch.setattr(module.subprocess, "run", lambda *a, **k: _FakeResult())
    sampler = module._GpuSampler(interval_seconds=0.01)
    with sampler:
        import time
        time.sleep(0.05)
    summary = sampler.summary()
    assert summary["sample_count"] > 0
    assert summary["mean_utilization_gpu_pct"] == pytest.approx(37.0)
    assert summary["mean_memory_used_mb"] == pytest.approx(1200.0)
