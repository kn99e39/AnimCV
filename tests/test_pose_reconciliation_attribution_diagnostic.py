"""Focused invariants for the diagnostic-only docs/44 attribution."""

import importlib.util
from pathlib import Path

import numpy as np

from common.canonical_pose import JOINT_INDEX
from framepose.pose_reconciliation import ProjectionContext, reconcile_hinge
from framepose.signs import SIGN_FIELD_NAMES, UNKNOWN, sign_state


_SCRIPT = Path(__file__).parents[1] / "scripts" / "diagnose_pose_reconciliation_attribution.py"
_SPEC = importlib.util.spec_from_file_location("pose_reconciliation_attribution", _SCRIPT)
attribution = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(attribution)


def _unknown_hinge_pose():
    pose = np.zeros((17, 3), dtype=np.float64)
    proximal = np.asarray([0.0, 2.0, 0.0])
    distal = np.asarray([1.0, 3.0, 0.0])
    axis_hat = (distal - proximal) / np.linalg.norm(distal - proximal)
    depth = np.asarray([0.0, 1.0, 0.0])
    u_hat = depth - axis_hat * float(axis_hat @ depth)
    u_hat /= np.linalg.norm(u_hat)
    v_hat = np.cross(axis_hat, depth)
    v_hat /= np.linalg.norm(v_hat)
    center = (proximal + distal) / 2.0
    pose[JOINT_INDEX["left_shoulder"]] = proximal
    pose[JOINT_INDEX["left_elbow"]] = center + 0.8 * v_hat
    pose[JOINT_INDEX["left_wrist"]] = distal
    return pose


def test_c_partition_is_disjoint_and_covers_exact_rows():
    rows = np.asarray([17, 29])
    result = attribution.partition_c_rows(rows, [-1, 1], [1, UNKNOWN])

    np.testing.assert_array_equal(result["readable_wrong"], [17])
    np.testing.assert_array_equal(result["h0_unknown"], [29])
    assert np.intersect1d(result["readable_wrong"], result["h0_unknown"]).size == 0
    np.testing.assert_array_equal(
        np.sort(np.concatenate((result["readable_wrong"], result["h0_unknown"]))), rows)


def test_unresolved_reason_taxonomy_partitions_only_unresolved_entries():
    entries = [
        {"outcome": "unresolved", "reason": "current hinge branch is unreadable; no guess"},
        {"outcome": "unresolved", "reason": "bone-preserving locus is degenerate"},
        {"outcome": "unresolved", "reason": "unclassified diagnostic reason"},
        {"outcome": "corrected", "reason": "not counted"},
    ]

    result = attribution.unresolved_reason_accounting(entries)

    assert result["unresolved"] == 3
    assert result["by_reason"][attribution.H0_UNKNOWN_REFUSAL] == 1
    assert result["by_reason"]["degenerate_bone_circle"] == 1
    assert result["by_reason"]["other"] == 1
    assert sum(result["by_reason"].values()) == result["unresolved"]
    assert result["reason_identity_holds"] is True


def test_coverage_gap_decomposition_closes_with_swivel_only_counter_contribution():
    minimum = ["corrected", "corrected", "corrected", "unresolved"]
    swivel = [
        {"outcome": "unresolved", "reason": "current hinge branch is unreadable; no guess"},
        {"outcome": "unresolved", "reason": "bone-preserving locus is degenerate"},
        {"outcome": "corrected"},
        {"outcome": "corrected"},
    ]

    result = attribution.decompose_coverage_gap(minimum, swivel, [True, False, False, False])

    assert result["minimum_norm_corrected"] == 3
    assert result["swivel_corrected"] == 2
    assert result["positive_contributions"] == {
        "h0_unknown_early_refusal": 1,
        "true_swivel_geometric_infeasibility": 1,
        "final_readback_or_invariant_failure": 0,
        "other": 0,
    }
    assert result["swivel_only_corrected_counter_contribution"] == 1
    assert result["coverage_gap"] == result["decomposed_gap"] == 1
    assert result["identity_holds"] is True


def test_wrong_sign_corrected_and_unresolved_partition_is_exact():
    rows = np.asarray([3, 8, 13, 21])

    partition = attribution.partition_outcomes(
        rows, ["corrected", "unresolved", "corrected", "unresolved"])

    np.testing.assert_array_equal(partition["corrected"], [3, 13])
    np.testing.assert_array_equal(partition["unresolved"], [8, 21])
    assert len(partition["corrected"]) + len(partition["unresolved"]) == len(rows)
    assert np.intersect1d(partition["corrected"], partition["unresolved"]).size == 0


def test_unknown_prestate_counterfactual_is_non_mutating_and_production_still_refuses():
    field = "left_elbow_forward_bend"
    column = SIGN_FIELD_NAMES.index(field)
    pose = _unknown_hinge_pose()
    original = pose.copy()
    valid = np.ones(17, dtype=bool)
    assert int(sign_state(pose, valid)[column]) == UNKNOWN
    context = ProjectionContext(
        np.asarray([[1000.0, 0.0, 320.0], [0.0, 1000.0, 240.0], [0.0, 0.0, 1.0]]),
        (640.0, 480.0), np.zeros(3))
    requested = 1
    observation = np.full((17, 3), 0.5, dtype=np.float64)
    observed_before = observation.copy()

    result = attribution.counterfactual_swivel_without_prestate_refusal(
        pose, valid, requested, observation[JOINT_INDEX["left_elbow"]], context, field)
    production_pose, production_report = reconcile_hinge(
        pose, valid, requested, observation[JOINT_INDEX["left_elbow"]], context, field)

    assert result["label"] == "COUNTERFACTUAL_SWIVEL_WITHOUT_PRESTATE_REFUSAL"
    assert result["status"] == "feasible_requested_sign_solution"
    assert result["read_back_after_attempt"] == requested
    assert production_report["outcome"] == "unresolved"
    assert production_report["reason"] == "current hinge branch is unreadable; no guess"
    np.testing.assert_array_equal(pose, original)
    np.testing.assert_array_equal(production_pose, original)
    np.testing.assert_array_equal(observation, observed_before)
