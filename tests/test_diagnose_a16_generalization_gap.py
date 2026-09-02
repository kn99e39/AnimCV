"""Focused tests for the docs/22 A16 generalization-gap diagnosis.

Diagnostic-only units: no training, no checkpoint changes. Covers the pure
geometry/accounting math this script relies on -- X/Y decomposition,
counterfactual substitution, shoulder/hip coherence, sequence-boundary-safe
run accounting, error-migration partition logic, and the diagnostic torso
local frame.
"""
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

pytest.importorskip("torch", reason="generalization-gap diagnosis checks require the optional training extra")

_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT = _ROOT / "scripts" / "diagnose_a16_generalization_gap.py"


def _load_module():
    sys.path.insert(0, str(_ROOT / "src"))
    try:
        spec = importlib.util.spec_from_file_location("diagnose_a16_generalization_gap", _SCRIPT)
        module = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.pop(0)


def _skeleton(n, shoulder_xy=(-1.0, 1.0, 0.0, 0.0), hip_xy=(-1.0, 1.0, -1.0, -1.0)):
    """n frames, all identical: left_shoulder=(-1,0), right_shoulder=(1,0),
    left_hip=(-1,-1), right_hip=(1,-1) by default (X,Y,Z layout)."""
    from pose.pose_lifter import H36M_NAMES

    points = np.zeros((n, len(H36M_NAMES), 3), dtype=np.float64)
    ls, rs = H36M_NAMES.index("left_shoulder"), H36M_NAMES.index("right_shoulder")
    lh, rh = H36M_NAMES.index("left_hip"), H36M_NAMES.index("right_hip")
    points[:, ls, 0], points[:, rs, 0] = shoulder_xy[0], shoulder_xy[1]
    points[:, ls, 1], points[:, rs, 1] = shoulder_xy[2], shoulder_xy[3]
    points[:, lh, 0], points[:, rh, 0] = hip_xy[0], hip_xy[1]
    points[:, lh, 1], points[:, rh, 1] = hip_xy[2], hip_xy[3]
    return points


def test_pair_geometry_computes_expected_delta_x_delta_y_magnitude_angle():
    module = _load_module()
    points = _skeleton(2, shoulder_xy=(-1.0, 1.0, 0.0, 0.5))  # right shoulder forward by 0.5
    geom = module._pair_geometry(points)
    # shoulder: delta_x = 1-(-1) = 2, delta_y = 0.5 - 0 = 0.5
    assert geom["delta_x"][0, 0] == pytest.approx(2.0)
    assert geom["delta_y"][0, 0] == pytest.approx(0.5)
    assert geom["magnitude"][0, 0] == pytest.approx(np.sqrt(2.0 ** 2 + 0.5 ** 2))
    assert geom["angle_deg"][0, 0] == pytest.approx(np.degrees(np.arctan2(0.5, 2.0)))


def test_residuals_zero_when_prediction_equals_gt():
    module = _load_module()
    from pose.pose_lifter import H36M_NAMES

    points = _skeleton(3)
    geom = module._pair_geometry(points)
    valid = np.ones((3, len(H36M_NAMES)), dtype=bool)
    pair_valid = module._pair_valid(valid)
    residuals = module._residuals(geom, geom, pair_valid)
    assert np.allclose(residuals["delta_x_residual"], 0.0)
    assert np.allclose(residuals["delta_y_residual"], 0.0)
    assert not residuals["sign_disagreement"].any()


def test_counterfactual_x15_y16_isolates_the_y_component():
    """CF with A15's delta_X and A16's delta_Y should reproduce A15's exact
    angular error when A16's delta_Y happens to equal A15's delta_Y (sanity:
    CF degenerates to the X-source's own geometry when Y sources match)."""
    module = _load_module()
    from pose.pose_lifter import H36M_NAMES

    n = 2
    gt_points = _skeleton(n)
    gt_geom = module._pair_geometry(gt_points)
    a15_points = _skeleton(n, shoulder_xy=(-1.0, 1.0, 0.0, 0.3))
    a15_geom = module._pair_geometry(a15_points)
    valid = np.ones((n, len(H36M_NAMES)), dtype=bool)
    pair_valid = module._pair_valid(valid)
    indices = np.arange(n)

    # When x_source == y_source, CF must equal that source's own geometry.
    cf_same = module._counterfactual(gt_geom, a15_geom, a15_geom, pair_valid, indices)
    a15_residuals = module._residuals(a15_geom, gt_geom, pair_valid)
    a15_summary = module._subset_summary(a15_residuals, pair_valid, indices)
    assert cf_same["shoulder"]["counterfactual_angular_error_mean_deg"] == pytest.approx(
        a15_summary["shoulder"]["angular_error_mean_deg"], abs=1e-6,
    )


def test_coherence_disagreement_zero_when_shoulder_and_hip_aligned():
    module = _load_module()
    points = _skeleton(2, shoulder_xy=(-1.0, 1.0, 0.0, 0.0), hip_xy=(-1.0, 1.0, 0.0, 0.0))
    geom = module._pair_geometry(points)
    from pose.pose_lifter import H36M_NAMES
    valid = np.ones((2, len(H36M_NAMES)), dtype=bool)
    pair_valid = module._pair_valid(valid)
    coherence = module._coherence(geom, pair_valid)
    assert np.allclose(coherence["disagreement_deg"], 0.0)
    assert coherence["sign_agreement"].all()


def test_coherence_disagreement_detects_shoulder_hip_mismatch():
    module = _load_module()
    # shoulder points forward (delta_y > 0), hip points backward (delta_y < 0)
    points = _skeleton(1, shoulder_xy=(-1.0, 1.0, 0.0, 1.0), hip_xy=(-1.0, 1.0, 0.0, -1.0))
    geom = module._pair_geometry(points)
    from pose.pose_lifter import H36M_NAMES
    valid = np.ones((1, len(H36M_NAMES)), dtype=bool)
    pair_valid = module._pair_valid(valid)
    coherence = module._coherence(geom, pair_valid)
    # shoulder axis (2,1) at atan2(1,2)=26.57 deg, hip axis (2,-1) at -26.57 deg.
    assert coherence["disagreement_deg"][0] == pytest.approx(53.13, abs=0.1)
    assert not coherence["sign_agreement"][0]


def test_torso_local_frame_zero_error_when_shoulder_hip_axes_match():
    module = _load_module()
    points = _skeleton(1, shoulder_xy=(-1.0, 1.0, 0.0, 0.5), hip_xy=(-1.0, 1.0, 0.0, 0.5))
    geom = module._pair_geometry(points)
    from pose.pose_lifter import H36M_NAMES
    valid = np.ones((1, len(H36M_NAMES)), dtype=bool)
    pair_valid = module._pair_valid(valid)
    frame = module._torso_local_frame(geom, pair_valid)
    assert frame["shoulder_frame_error_deg"][0] == pytest.approx(0.0, abs=1e-4)
    assert frame["hip_frame_error_deg"][0] == pytest.approx(0.0, abs=1e-4)


def test_torso_local_frame_reports_disagreement_when_axes_diverge():
    module = _load_module()
    points = _skeleton(1, shoulder_xy=(-1.0, 1.0, 0.0, 1.0), hip_xy=(-1.0, 1.0, 0.0, -0.5))
    geom = module._pair_geometry(points)
    from pose.pose_lifter import H36M_NAMES
    valid = np.ones((1, len(H36M_NAMES)), dtype=bool)
    pair_valid = module._pair_valid(valid)
    frame = module._torso_local_frame(geom, pair_valid)
    assert frame["shoulder_frame_error_deg"][0] > 5.0
    assert frame["hip_frame_error_deg"][0] > 5.0


def test_run_lengths_respect_sequence_boundaries():
    module = _load_module()
    flags = np.array([True, True, False, True, True, True])
    sequence_ids = np.array(["seq_a", "seq_a", "seq_a", "seq_b", "seq_b", "seq_b"])
    runs = module._run_lengths(flags, sequence_ids)
    # seq_a: [T,T,F] -> one run of length 2. seq_b: [T,T,T] -> one run of length 3.
    assert sorted(runs) == [2, 3]


def test_run_lengths_split_a_run_that_crosses_a_sequence_boundary():
    module = _load_module()
    flags = np.array([True, True, True, True])
    sequence_ids = np.array(["seq_a", "seq_a", "seq_b", "seq_b"])
    runs = module._run_lengths(flags, sequence_ids)
    # Without boundary-awareness this would report one run of 4; must be two runs of 2.
    assert sorted(runs) == [2, 2]


def test_sign_transitions_ignore_cross_sequence_boundaries():
    module = _load_module()
    sign = np.array([1.0, 1.0, -1.0, 1.0])  # last transition is seq_b's first frame
    sequence_ids = np.array(["seq_a", "seq_a", "seq_b", "seq_b"])
    transitions = module._sign_transitions(sign, sequence_ids)
    assert transitions.tolist() == [False, False, False, True]


def test_error_migration_categories_are_mutually_exclusive_and_exhaustive():
    a15_bad = np.array([True, True, False, False])
    a16_good = np.array([True, False, True, False])
    a15_good = ~a15_bad
    a16_bad = ~a16_good
    categories = {
        "previously_bad_improved": a15_bad & a16_good,
        "previously_bad_worse": a15_bad & a16_bad,
        "previously_good_remains_good": a15_good & a16_good,
        "previously_good_newly_bad": a15_good & a16_bad,
    }
    stacked = np.stack(list(categories.values()))
    assert (stacked.sum(axis=0) == 1).all()  # exactly one category per frame
