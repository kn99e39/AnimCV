"""Focused tests for the eager-vs-compiled verdict script.

Diagnostic-only units: none of this changes production training behavior.
Runs the full pipeline (numerical equivalence, compiled reproducibility,
short-training A/B, throughput A/B) on CPU with a tiny dataset and a
single warm-up/measure step -- exercising the real torch.compile path
(inductor also targets CPU), not a mock, while keeping runtime bounded.
"""
from pathlib import Path
import sys

import pytest

pytest.importorskip("torch", reason="compile-candidate checks require the optional training extra")

_ROOT = Path(__file__).resolve().parent.parent


def _load_module():
    sys.path.insert(0, str(_ROOT / "scripts"))
    sys.path.insert(0, str(_ROOT / "src"))
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "benchmark_torch_compile_candidate", _ROOT / "scripts" / "benchmark_torch_compile_candidate.py",
        )
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
    return build_dataset(pose_sequence, target_sequence, (100, 100), "compile-candidate-test")


def test_max_abs_rel_diff_zero_for_identical_tensors():
    import torch

    module = _load_module()
    tensor = torch.tensor([1.0, 2.0, -3.0])
    result = module._max_abs_rel_diff(tensor, tensor.clone())
    assert result["max_abs_diff"] == pytest.approx(0.0)
    assert result["max_rel_diff"] == pytest.approx(0.0)
    assert result["exactly_equal"] is True


def test_max_abs_rel_diff_detects_a_known_difference():
    import torch

    module = _load_module()
    a = torch.tensor([1.0, 10.0])
    b = torch.tensor([1.1, 10.0])
    result = module._max_abs_rel_diff(a, b)
    assert result["max_abs_diff"] == pytest.approx(0.1, abs=1e-5)
    assert result["exactly_equal"] is False


def test_component_losses_reuses_production_decomposition():
    import torch
    from pose.pose_lifter import H36M_NAMES

    module = _load_module()
    n = 2
    prediction = torch.zeros((n, len(H36M_NAMES), 3))
    target = torch.zeros((n, len(H36M_NAMES), 3))
    valid = torch.ones((n, len(H36M_NAMES)), dtype=torch.bool)

    result = module._component_losses(torch, prediction, target, valid)
    assert set(result) == {"bone", "torso", "hinge"}
    assert all(value == pytest.approx(0.0) for value in result.values())


def test_numerical_equivalence_eager_vs_compiled_on_cpu():
    """Real (not mocked) eager-vs-compiled comparison, CPU-only for test
    speed: no autocast on CPU, so differences should be near float32
    associativity noise, not large."""
    from training.temporal_lifter import TrainingConfig, _torch

    module = _load_module()
    dataset = _tiny_dataset()
    config = TrainingConfig(epochs=1, device="cpu", seed=5,
                             **{k: v for k, v in module.A9_CONFIG_KWARGS.items() if k != "seed"})
    torch, nn = _torch()
    device = torch.device("cpu")

    result = module.numerical_equivalence(torch, nn, config, dataset, device)
    assert result["prediction_diff"]["max_abs_diff"] < 1e-3
    assert result["loss_diff"]["max_abs_diff"] < 1e-3
    assert result["gradient_max_abs_diff"] < 1e-2
    assert result["gradients_finite_eager"] is True
    assert result["gradients_finite_compiled"] is True
    assert result["gradient_missing_parameter_names"] == []


def test_compiled_reproducibility_on_cpu():
    from training.temporal_lifter import TrainingConfig, _torch

    module = _load_module()
    dataset = _tiny_dataset()
    config = TrainingConfig(epochs=1, device="cpu", seed=9,
                             **{k: v for k, v in module.A9_CONFIG_KWARGS.items() if k != "seed"})
    torch, nn = _torch()
    device = torch.device("cpu")

    result = module.compiled_reproducibility(torch, nn, config, dataset, device)
    assert result["loss_exactly_equal"] is True
    assert result["prediction_diff"]["exactly_equal"] is True


def test_short_training_ab_produces_matching_length_trajectories():
    from training.temporal_lifter import TrainingConfig, _torch

    module = _load_module()
    dataset = _tiny_dataset()
    config = TrainingConfig(epochs=1, device="cpu", seed=13,
                             **{k: v for k, v in module.A9_CONFIG_KWARGS.items() if k != "seed"})
    torch, nn = _torch()
    device = torch.device("cpu")

    result = module.short_training_ab(torch, nn, config, dataset, device, steps=2)
    assert len(result["eager_loss_trajectory"]) == 2
    assert len(result["compiled_loss_trajectory"]) == 2
    assert result["final_loss_abs_diff"] < 1e-2
    assert result["final_parameter_max_abs_diff"] is not None
