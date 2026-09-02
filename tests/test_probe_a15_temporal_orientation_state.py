"""Focused tests for the docs/22 frozen-representation orientation probe.

Diagnostic-only units: no backbone training. Covers the closed-form linear
probe math, sequence-level train/test split (no frame leakage), the exact
q_shoulder/q_hip target definition, and that feature extraction from a real
(tiny) model never touches gradients.
"""
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

pytest.importorskip("torch", reason="temporal-probe checks require the optional training extra")

_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT = _ROOT / "scripts" / "probe_a15_temporal_orientation_state.py"


def _load_module():
    sys.path.insert(0, str(_ROOT / "src"))
    try:
        spec = importlib.util.spec_from_file_location("probe_a15_temporal_orientation_state", _SCRIPT)
        module = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.pop(0)


def test_q_targets_matches_corrected_srd_definition():
    from pose.pose_lifter import H36M_NAMES
    from training.temporal_lifter import FORWARD_DEPTH_AXIS

    module = _load_module()
    n = 3
    targets = np.zeros((n, len(H36M_NAMES), 3))
    ls, rs = H36M_NAMES.index("left_shoulder"), H36M_NAMES.index("right_shoulder")
    lh, rh = H36M_NAMES.index("left_hip"), H36M_NAMES.index("right_hip")
    targets[:, ls, FORWARD_DEPTH_AXIS] = -0.1
    targets[:, rs, FORWARD_DEPTH_AXIS] = 0.3
    targets[:, lh, FORWARD_DEPTH_AXIS] = -0.05
    targets[:, rh, FORWARD_DEPTH_AXIS] = 0.05

    q = module._q_targets(targets)
    assert q[0, 0] == pytest.approx((0.3 - (-0.1)) / np.sqrt(2.0))
    assert q[0, 1] == pytest.approx((0.05 - (-0.05)) / np.sqrt(2.0))


def test_linear_probe_recovers_a_perfect_linear_relationship():
    module = _load_module()
    rng = np.random.default_rng(1)
    features = rng.standard_normal((200, 8))
    true_weights = rng.standard_normal(8)
    targets = features @ true_weights + 0.5  # + intercept

    weights = module._fit_linear_probe(features, targets)
    prediction = module._apply_probe(weights, features)
    r_squared = module._r_squared(prediction, targets)
    assert r_squared == pytest.approx(1.0, abs=1e-6)


def test_r_squared_is_near_zero_for_unrelated_features_on_held_out_split():
    module = _load_module()
    rng = np.random.default_rng(2)
    features_train = rng.standard_normal((300, 8))
    targets_train = rng.standard_normal(300)  # unrelated to features
    features_test = rng.standard_normal((100, 8))
    targets_test = rng.standard_normal(100)

    weights = module._fit_linear_probe(features_train, targets_train)
    prediction = module._apply_probe(weights, features_test)
    r_squared = module._r_squared(prediction, targets_test)
    assert r_squared < 0.2  # near zero or negative; well below a real relationship


def test_sequence_split_puts_every_frame_of_a_sequence_on_one_side():
    module = _load_module()
    metadata = (
        [{"action": "seq_a"}] * 10 + [{"action": "seq_b"}] * 10
        + [{"action": "seq_c"}] * 10 + [{"action": "seq_d"}] * 10
    )
    train_indices, test_indices = module._sequence_split(metadata, test_fraction=0.25)
    train_sequences = {metadata[i]["action"] for i in train_indices}
    test_sequences = {metadata[i]["action"] for i in test_indices}
    assert train_sequences.isdisjoint(test_sequences)
    assert len(train_indices) + len(test_indices) == len(metadata)


def test_extract_features_returns_expected_shapes_from_the_real_backbone():
    """dilated_tcn_v1 (the architecture A15/A16 both use) exposes stem/blocks
    directly; this exercises _extract_features' actual no_grad, batched
    pattern against the real model class, not a mock."""
    import torch
    from pose.pose_lifter import H36M_NAMES
    from training.temporal_lifter import _model

    module = _load_module()
    model = _model(torch.nn, 16, "dilated_tcn_v1")
    model.eval()
    n, window = 5, 3
    x = torch.randn(n + window, len(H36M_NAMES), 3)
    offsets = torch.arange(n).unsqueeze(-1) + torch.arange(window)

    stem_features, block_features = module._extract_features(torch, model, x, offsets, batch_size=2)
    assert stem_features.shape == (n, 16)
    assert block_features.shape == (n, 16)
