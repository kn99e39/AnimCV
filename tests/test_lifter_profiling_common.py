"""Focused tests for the shared temporal-lifter profiling setup.

Diagnostic-only units: none of this changes production training behavior.
``profile_temporal_lifter_training.py`` and ``profile_temporal_lifter_kernels.py``
both build their benchmark state from ``setup()`` here, so its determinism is
the precondition for both scripts' pass-to-pass and eager-vs-compiled
comparisons being fair.
"""
from pathlib import Path
import sys

import pytest

pytest.importorskip("torch", reason="profiling-common checks require the optional training extra")

_ROOT = Path(__file__).resolve().parent.parent


def _load_module():
    sys.path.insert(0, str(_ROOT / "scripts"))
    sys.path.insert(0, str(_ROOT / "src"))
    try:
        import _lifter_profiling_common
        return _lifter_profiling_common
    finally:
        sys.path.pop(0)
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
    return build_dataset(pose_sequence, target_sequence, (100, 100), "profile-common-test")


def test_percentiles_are_correct():
    module = _load_module()
    result = module.percentiles([1.0, 2.0, 3.0, 4.0, 5.0])
    assert result["mean"] == pytest.approx(3.0)
    assert result["median"] == pytest.approx(3.0)
    assert result["min"] == pytest.approx(1.0)
    assert result["max"] == pytest.approx(5.0)


def test_setup_is_deterministic_across_repeated_calls():
    """Every diagnostic script (throughput pass, attribution pass, eager vs
    compiled benchmark) calls setup() independently; if it were not
    deterministic, none of those comparisons would be fair."""
    import torch
    from training.temporal_lifter import TrainingConfig, _torch

    module = _load_module()
    dataset = _tiny_dataset()
    config = TrainingConfig(epochs=1, device="cpu", seed=7,
                             **{k: v for k, v in module.A9_CONFIG_KWARGS.items() if k != "seed"})
    device = torch.device("cpu")
    torch, nn = _torch()

    first = module.setup(torch, nn, config, dataset, device)
    second = module.setup(torch, nn, config, dataset, device)

    assert torch.equal(first["epoch_inputs"], second["epoch_inputs"])
    assert all(torch.equal(a, b) for a, b in zip(first["batches"], second["batches"]))
    first_params = dict(first["model"].named_parameters())
    second_params = dict(second["model"].named_parameters())
    assert first_params.keys() == second_params.keys()
    assert all(torch.equal(first_params[name], second_params[name]) for name in first_params)


def test_gpu_sampler_degrades_gracefully_without_nvidia_smi(monkeypatch):
    module = _load_module()

    def _raise(*_args, **_kwargs):
        raise FileNotFoundError("nvidia-smi not found")

    monkeypatch.setattr(module.subprocess, "run", _raise)
    sampler = module.GpuSampler(interval_seconds=0.01)
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
    sampler = module.GpuSampler(interval_seconds=0.01)
    with sampler:
        import time
        time.sleep(0.05)
    summary = sampler.summary()
    assert summary["sample_count"] > 0
    assert summary["mean_utilization_gpu_pct"] == pytest.approx(37.0)
    assert summary["mean_memory_used_mb"] == pytest.approx(1200.0)
