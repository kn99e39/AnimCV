"""Focused tests for the kernel-level torch.profiler diagnostic script.

Diagnostic-only units: none of this changes production training behavior.
Covers the event-compatibility shims (torch's raw Kineto event API differs
between torch releases -- duration_us() vs duration_ns(), method vs
attribute access) and the pure aggregation functions with fake event
objects, since a real CUDA profiler trace cannot be produced without a GPU.
"""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

pytest.importorskip("torch", reason="kernel profiling checks require the optional training extra")

_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT = _ROOT / "scripts" / "profile_temporal_lifter_kernels.py"


def _load_module():
    sys.path.insert(0, str(_ROOT / "scripts"))
    sys.path.insert(0, str(_ROOT / "src"))
    try:
        spec = importlib.util.spec_from_file_location("profile_temporal_lifter_kernels", _SCRIPT)
        module = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.pop(0)
        sys.path.pop(0)


class _MethodStyleEvent:
    """Mimics torch 2.1.2's _KinetoEvent: device_type()/name()/duration_us() are methods."""

    def __init__(self, device_type_name: str, name: str, duration_us: float):
        self._device_type_name = device_type_name
        self._name = name
        self._duration_us = duration_us

    def device_type(self):
        return SimpleNamespace(name=self._device_type_name)

    def name(self):
        return self._name

    def duration_us(self):
        return self._duration_us


class _AttributeStyleEvent:
    """Mimics a torch build where these are plain attributes, and only
    duration_ns() is available (no duration_us())."""

    def __init__(self, device_type_name: str, name: str, duration_ns: float):
        self.device_type = SimpleNamespace(name=device_type_name)
        self.name = name
        self._duration_ns = duration_ns

    def duration_ns(self):
        return self._duration_ns


def test_event_duration_us_prefers_duration_us_method():
    module = _load_module()
    event = _MethodStyleEvent("CUDA", "kernel_a", 12.5)
    assert module._event_duration_us(event) == pytest.approx(12.5)


def test_event_duration_us_falls_back_to_duration_ns():
    module = _load_module()
    event = _AttributeStyleEvent("CUDA", "kernel_a", 12500.0)
    assert module._event_duration_us(event) == pytest.approx(12.5)


def test_event_name_and_device_type_handle_both_styles():
    module = _load_module()
    method_event = _MethodStyleEvent("CPU", "aten::conv1d", 1.0)
    attribute_event = _AttributeStyleEvent("CPU", "aten::conv1d", 1000.0)
    assert module._event_name(method_event) == "aten::conv1d"
    assert module._event_name(attribute_event) == "aten::conv1d"
    assert module._event_device_type_name(method_event) == "CPU"
    assert module._event_device_type_name(attribute_event) == "CPU"


def test_kernel_duration_stats_computes_distribution_and_short_fraction():
    module = _load_module()
    events = (
        [_MethodStyleEvent("CUDA", "kernel_small", value) for value in (1.0, 2.0, 3.0, 4.0, 5.0)]
        + [_MethodStyleEvent("CUDA", "kernel_big", 500.0)]
        + [_MethodStyleEvent("CPU", "aten::conv1d", 999.0)]  # must be excluded
    )
    stats = module._kernel_duration_stats(events)
    assert stats["count"] == 6
    assert stats["sum_us"] == pytest.approx(1.0 + 2.0 + 3.0 + 4.0 + 5.0 + 500.0)
    assert stats["max_us"] == pytest.approx(500.0)
    assert stats["min_us"] == pytest.approx(1.0)
    # all 5 "small" kernels are below the 10us threshold; the 500us one is not
    assert stats["short_kernel_count"] == 5
    assert stats["short_kernel_fraction"] == pytest.approx(5 / 6)


def test_kernel_duration_stats_handles_no_cuda_events():
    module = _load_module()
    events = [_MethodStyleEvent("CPU", "aten::conv1d", 1.0)]
    stats = module._kernel_duration_stats(events)
    assert stats == {"count": 0}


def test_dominant_ops_ranks_by_cumulative_time_and_respects_top_n():
    module = _load_module()
    events = (
        [_MethodStyleEvent("CUDA", "kernel_a", 10.0) for _ in range(3)]  # total 30
        + [_MethodStyleEvent("CUDA", "kernel_b", 100.0)]  # total 100 -- should rank first
        + [_MethodStyleEvent("CPU", "aten::conv1d", 1.0)]  # different device type, excluded
    )
    ranked = module._dominant_ops(events, "CUDA", top_n=1)
    assert len(ranked) == 1
    assert ranked[0]["name"] == "kernel_b"
    assert ranked[0]["count"] == 1
    assert ranked[0]["total_us"] == pytest.approx(100.0)


def test_dynamo_graph_report_never_raises_on_eager_forward_loss():
    """A weak smoke check: even without CUDA, the dry-run explain call must
    return a dict (either a real report or an {"error": ...} fallback),
    never propagate an exception up through the profiler."""
    import torch
    from training.temporal_lifter import TrainingConfig, _torch

    module = _load_module()
    from pose.pose_lifter import H36M_NAMES, LiftedPoseFrame, LiftedPosePoint, LiftedPoseSequence
    from pose.pose_types import PoseFrame, PoseLandmark, PoseSequence
    from training.temporal_lifter import build_dataset

    names = set(H36M_NAMES) - {"thorax"}

    def pose(index):
        return PoseFrame(index, index / 25, {name: PoseLandmark(name, 10 + (index % 50), 20, 1.0, True) for name in names})

    def target(index):
        return LiftedPoseFrame(index, index / 25, {name: LiftedPosePoint(name, (index / 10, 0, 0), 1.0, 0.0) for name in names})

    dataset = build_dataset(
        PoseSequence([pose(i) for i in range(300)], 25), LiftedPoseSequence([target(i) for i in range(300)], 25),
        (100, 100), "kernel-profile-test",
    )
    config = TrainingConfig(epochs=1, device="cpu", seed=17,
                             **{k: v for k, v in module.A9_CONFIG_KWARGS.items() if k != "seed"})
    torch, nn = _torch()
    state = module.setup(torch, nn, config, dataset, torch.device("cpu"))

    result = module._dynamo_graph_report(torch, state, config)
    assert isinstance(result, dict)
    assert "graph_count" in result or "error" in result
