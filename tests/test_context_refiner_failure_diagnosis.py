"""Pure diagnostic contracts for the frozen C2 failure analysis."""

from __future__ import annotations

import numpy as np

from scripts.diagnose_context_refiner_failure import parity_report, residual_summary


def test_c1_c2_gate_identity_has_no_population_mismatch():
    h0 = np.zeros((2, 2, 3), dtype=np.float32)
    c1 = h0.copy()
    c2 = h0.copy()
    gate = np.zeros((2, 2), dtype=bool)
    gate[0, 1] = True
    c1[0, 1, 0] = 0.25
    c2[0, 1, 0] = 0.10
    report = parity_report(h0, c1, c2, gate, np.arange(2))
    assert report["same_gate_applied_by_C1_and_C2"]
    assert report["C1_only"] == 0
    assert report["C2_only"] == 0
    assert report["both"] == 1
    assert report["neither"] == 3


def test_residual_attribution_reports_direction_and_orthogonal_error():
    target = np.asarray([[[2.0, 0.0, 0.0]]], dtype=np.float32)
    h0 = np.asarray([[[0.0, 0.0, 0.0]]], dtype=np.float32)
    candidate = np.asarray([[[1.0, 1.0, 0.0]]], dtype=np.float32)
    valid = np.ones((1, 1), dtype=bool)
    mask = valid.copy()
    report = residual_summary(target, valid, h0, candidate, mask)
    assert report["metric_rows"] == 1
    assert np.isclose(report["norm_ratio"]["mean"], np.sqrt(2.0) / 2.0)
    assert np.isclose(report["cosine_alignment"]["mean"], 1.0 / np.sqrt(2.0))
    assert np.isclose(report["signed_projection_mm"]["mean"], 1000.0)
    assert np.isclose(report["orthogonal_residual_mm"]["mean"], 1000.0)
    assert np.isclose(report["target_error_delta_mm"]["mean"], 585.7864376, atol=1e-4)


def test_residual_metrics_exclude_invalid_target_rows():
    target = np.zeros((2, 1, 3), dtype=np.float32)
    h0 = np.zeros_like(target)
    candidate = np.zeros_like(target)
    valid = np.asarray([[True], [False]])
    mask = np.ones_like(valid)
    report = residual_summary(target, valid, h0, candidate, mask)
    assert report["joint_rows"] == 2
    assert report["evaluated_rows"] == 1
    assert report["metric_rows"] == 0
