"""Focused tests for the docs/21 corrected-candidate Delta_G diagnostic.

Diagnostic-only units: none of this changes production training behavior.
Verifies the core equivalence claim -- Delta_G (candidate - A9) equals the
gradient of the relational term alone -- on small synthetic batches, and
that the loss decomposes additively.
"""
import importlib.util
from pathlib import Path
import sys

import pytest

pytest.importorskip("torch", reason="gradient-delta diagnostic checks require the optional training extra")

_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT = _ROOT / "scripts" / "diagnose_corrected_srd_gradient_delta.py"


def _load_module():
    sys.path.insert(0, str(_ROOT / "scripts"))
    sys.path.insert(0, str(_ROOT / "src"))
    try:
        spec = importlib.util.spec_from_file_location("diagnose_corrected_srd_gradient_delta", _SCRIPT)
        module = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.pop(0)
        sys.path.pop(0)


def test_gradient_delta_equals_relational_term_gradient_alone_on_real_model():
    """The same claim as the synthetic unit tests, but through this
    script's own _gradient_delta() using a real (tiny) model forward pass,
    matching what the server-scale diagnostic actually measures."""
    import torch
    from pose.pose_lifter import H36M_NAMES
    from training.temporal_lifter import _model

    module = _load_module()
    torch.manual_seed(11)
    model = _model(torch.nn, 8, "legacy_tcn_v1")
    n = 6
    windows = torch.randn(n, 3, len(H36M_NAMES), 3)
    prediction = model(windows)
    target = prediction.detach() + torch.randn_like(prediction) * 0.2
    valid = torch.ones((n, len(H36M_NAMES)), dtype=torch.bool)
    source_ids = torch.zeros(n, dtype=torch.long)

    result = module._gradient_delta(torch, model, prediction, target, valid, source_ids)

    assert result["delta_g_vs_g_relational_rel_diff"] < 1e-4
    assert result["loss_decomposition_abs_diff"] < 1e-6
    assert result["g_candidate_norm"] > 0.0
    assert result["g_a9_norm"] >= 0.0


def test_gradient_delta_is_zero_when_relational_residual_is_zero():
    """prediction == target everywhere -> q_pred == q_target -> S_relational
    == 0 -> Delta_G must be exactly zero (candidate == A9)."""
    import torch
    from pose.pose_lifter import H36M_NAMES
    from training.temporal_lifter import _model

    module = _load_module()
    torch.manual_seed(23)
    model = _model(torch.nn, 8, "legacy_tcn_v1")
    n = 4
    windows = torch.randn(n, 3, len(H36M_NAMES), 3)
    prediction = model(windows)
    target = prediction.detach().clone()
    valid = torch.ones((n, len(H36M_NAMES)), dtype=torch.bool)
    source_ids = torch.zeros(n, dtype=torch.long)

    result = module._gradient_delta(torch, model, prediction, target, valid, source_ids)
    assert result["delta_g_norm"] == pytest.approx(0.0, abs=1e-6)
    assert result["g_relational_norm"] == pytest.approx(0.0, abs=1e-6)


def test_summarize_computes_expected_aggregate_fields():
    module = _load_module()
    batch_reports = [
        {"g_a9_norm": 1.0, "delta_g_norm": 0.1, "g_candidate_norm": 1.05, "cosine_g_a9_delta_g": 0.5,
         "delta_g_to_g_a9_ratio": 0.1, "delta_g_vs_g_relational_rel_diff": 1e-6, "loss_decomposition_abs_diff": 1e-7},
        {"g_a9_norm": 2.0, "delta_g_norm": 0.3, "g_candidate_norm": 2.1, "cosine_g_a9_delta_g": 0.3,
         "delta_g_to_g_a9_ratio": 0.15, "delta_g_vs_g_relational_rel_diff": 2e-6, "loss_decomposition_abs_diff": 2e-7},
    ]
    summary = module._summarize(batch_reports)
    assert summary["g_a9_norm_mean"] == pytest.approx(1.5)
    assert summary["delta_g_to_g_a9_ratio_mean"] == pytest.approx(0.125)
    assert summary["max_delta_g_vs_g_relational_rel_diff"] == pytest.approx(2e-6)
