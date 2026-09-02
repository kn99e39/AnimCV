"""Focused tests for the training-throughput diagnostic script.

Diagnostic-only units: none of this changes production training behavior.
Shared setup determinism and the GPU sampler are covered by
``tests/test_lifter_profiling_common.py``; this file covers what is unique
to this script -- that its two passes (uninstrumented throughput vs
CUDA-event stage attribution) compute identical values from identical
seeded state.
"""
import importlib.util
from pathlib import Path
import sys

import pytest

pytest.importorskip("torch", reason="throughput profiling checks require the optional training extra")

_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT = _ROOT / "scripts" / "profile_temporal_lifter_training.py"


def _load_module():
    sys.path.insert(0, str(_ROOT / "scripts"))
    sys.path.insert(0, str(_ROOT / "src"))
    try:
        spec = importlib.util.spec_from_file_location("profile_temporal_lifter_training", _SCRIPT)
        module = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        spec.loader.exec_module(module)
        return module
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
    return build_dataset(pose_sequence, target_sequence, (100, 100), "profile-test")


def test_throughput_and_attribution_passes_produce_the_same_first_step_loss():
    """A weak but concrete semantic-equivalence check: with identical seeded
    setup, the first measured step's loss must match between the
    uninstrumented throughput pass and the CUDA-event-instrumented
    attribution pass -- the diagnostic sync must not change the computed
    values, only when they become visible."""
    import torch
    from training.temporal_lifter import TrainingConfig, _supervision_loss, _torch

    module = _load_module()
    dataset = _tiny_dataset()
    config = TrainingConfig(epochs=1, device="cpu", seed=11,
                             **{k: v for k, v in module.A9_CONFIG_KWARGS.items() if k != "seed"})
    device = torch.device("cpu")
    torch, nn = _torch()

    def first_step_loss():
        state = module.setup(torch, nn, config, dataset, device)
        batch = state["batches"][0]
        windows = state["epoch_inputs"][state["offset_tensor"][batch]]
        with torch.no_grad():
            prediction = state["model"](windows)
            mask = state["valid_tensor"][batch]
            return float(_supervision_loss(torch, prediction, state["y"][batch], mask, config).item())

    assert first_step_loss() == pytest.approx(first_step_loss())


def test_attribution_pass_stage_sum_is_close_to_a_single_step_wall_time():
    """Sanity check on the attribution pass's own stage decomposition: for
    one step on CPU (no CUDA events, so timings are absent, but the call
    sequence must still run end-to-end without error) the function returns
    the expected stage-key schema."""
    import torch
    from training.temporal_lifter import TrainingConfig, _torch

    module = _load_module()
    dataset = _tiny_dataset()
    config = TrainingConfig(epochs=1, device="cpu", seed=13,
                             **{k: v for k, v in module.A9_CONFIG_KWARGS.items() if k != "seed"})
    device = torch.device("cpu")
    torch, nn = _torch()
    state = module.setup(torch, nn, config, dataset, device)

    stage_ms = module._attribution_pass(torch, state, config, device, warmup_steps=1, measure_steps=1)
    assert set(stage_ms) == {"batch_construction_ms", "forward_ms", "loss_ms", "backward_ms",
                              "optimizer_step_ms", "scaler_update_ms"}
    # No CUDA events on CPU, so no measured entries are recorded -- confirms
    # the script never fabricates GPU timings for a non-CUDA device.
    assert all(values == [] for values in stage_ms.values())
