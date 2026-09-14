"""Contracts for the separate endpoint-fixed Pose Reconciliation layer."""

import numpy as np
import pytest

from common.canonical_pose import JOINT_INDEX
from framepose.pose_reconciliation import (
    ALREADY_SATISFIED,
    CORRECTED,
    ProjectionContext,
    UNRESOLVED,
    reconcile_hinge,
)
from framepose.signs import SIGN_FIELD_NAMES, UNKNOWN, sign_state


_FIELD = "left_elbow_forward_bend"
_CHAIN = ("left_shoulder", "left_elbow", "left_wrist")
_K = np.array([[1000.0, 0.0, 320.0], [0.0, 1000.0, 240.0], [0.0, 0.0, 1.0]])
_SIZE = (640.0, 480.0)


def _pose():
    pose = np.zeros((17, 3), dtype=np.float64)
    proximal = np.array([0.0, 2.0, 0.0])
    distal = np.array([1.0, 3.0, 0.0])
    axis_hat = (distal - proximal) / np.linalg.norm(distal - proximal)
    depth = np.array([0.0, 1.0, 0.0])
    u = depth - axis_hat * (axis_hat @ depth)
    u /= np.linalg.norm(u)
    v = np.cross(axis_hat, depth)
    v /= np.linalg.norm(v)
    center = (proximal + distal) / 2.0
    current = center + 0.8 * u + 0.6 * v
    pose[JOINT_INDEX["left_shoulder"]] = proximal
    pose[JOINT_INDEX["left_elbow"]] = current
    pose[JOINT_INDEX["left_wrist"]] = distal
    return pose, center, u, v


def _context():
    return ProjectionContext(_K, _SIZE, np.array([0.0, 5.0, 0.0]))


def _observation_for(point, context):
    pixels = context.project_root_relative(point)
    return pixels / np.asarray(_SIZE)


def _opposite_request(pose, valid):
    index = SIGN_FIELD_NAMES.index(_FIELD)
    current = int(sign_state(pose, valid)[index])
    assert current in (-1, 1)
    return -current


def test_projection_context_requires_explicit_root_placement_and_projects_canonical_axes():
    context = ProjectionContext(_K, _SIZE, np.array([0.0, 5.0, 0.0]))
    np.testing.assert_allclose(context.project_root_relative([0.0, 2.0, 0.0]), [320.0, 240.0])
    missing_root = ProjectionContext(_K, _SIZE, None)
    assert "root_offset_camera" in missing_root.validation_error()


def test_observation_guided_swivel_changes_only_middle_and_preserves_bones():
    pose, center, u, v = _pose()
    valid = np.ones(17, dtype=bool)
    context = _context()
    requested = _opposite_request(pose, valid)
    # The desired visible configuration is a point on the opposite readable
    # branch.  The solver should recover it without an angle sweep.
    target = center - 0.8 * u + 0.6 * v
    corrected, report = reconcile_hinge(
        pose, valid, requested, _observation_for(target, context), context, _FIELD)

    assert report["outcome"] == CORRECTED
    assert report["candidate_source"] in {"stationary", "readability_boundary", "half_angle_infinity"}
    np.testing.assert_array_equal(corrected[JOINT_INDEX["left_shoulder"]], pose[JOINT_INDEX["left_shoulder"]])
    np.testing.assert_array_equal(corrected[JOINT_INDEX["left_wrist"]], pose[JOINT_INDEX["left_wrist"]])
    p, m, d = (corrected[JOINT_INDEX[name]] for name in _CHAIN)
    p0, m0, d0 = (pose[JOINT_INDEX[name]] for name in _CHAIN)
    assert np.linalg.norm(m - p) == pytest.approx(np.linalg.norm(m0 - p0), abs=1e-12)
    assert np.linalg.norm(d - m) == pytest.approx(np.linalg.norm(d0 - m0), abs=1e-12)
    assert np.linalg.norm(m - center) == pytest.approx(1.0, abs=1e-12)
    assert report["circle_membership_abs_error_m2"] < 1e-12
    assert int(sign_state(corrected, valid)[SIGN_FIELD_NAMES.index(_FIELD)]) == requested
    assert report["endpoint_positions_unchanged"] is True


def test_reconciliation_is_deterministic_and_is_not_a_rig_roll_claim():
    pose, center, u, v = _pose()
    valid = np.ones(17, dtype=bool)
    context = _context()
    requested = _opposite_request(pose, valid)
    target = center - 0.6 * u - 0.8 * v
    first, first_report = reconcile_hinge(
        pose, valid, requested, _observation_for(target, context), context, _FIELD)
    second, second_report = reconcile_hinge(
        pose, valid, requested, _observation_for(target, context), context, _FIELD)
    np.testing.assert_array_equal(first, second)
    assert first_report == second_report
    assert "downstream" in first_report["orientation_contract"]


def test_already_correct_is_an_exact_noop_without_projection_context():
    pose, _, _, _ = _pose()
    valid = np.ones(17, dtype=bool)
    index = SIGN_FIELD_NAMES.index(_FIELD)
    requested = int(sign_state(pose, valid)[index])
    corrected, report = reconcile_hinge(pose, valid, requested, None, None, _FIELD)
    np.testing.assert_array_equal(corrected, pose)
    assert report["outcome"] == ALREADY_SATISFIED


def test_unknown_unreadable_or_missing_projection_never_falls_back_to_historical_policies():
    pose, center, u, v = _pose()
    valid = np.ones(17, dtype=bool)
    requested = _opposite_request(pose, valid)
    target = center - 0.8 * u + 0.6 * v
    for context in (None, ProjectionContext(_K, _SIZE, None)):
        corrected, report = reconcile_hinge(
            pose, valid, requested, _observation_for(target, _context()), context, _FIELD)
        np.testing.assert_array_equal(corrected, pose)
        assert report["outcome"] == UNRESOLVED
    corrected, report = reconcile_hinge(
        pose, valid, UNKNOWN, _observation_for(target, _context()), _context(), _FIELD)
    np.testing.assert_array_equal(corrected, pose)
    assert report["outcome"] == UNRESOLVED


def test_exact_depth_aligned_chain_is_unresolved_under_existing_sign_semantics():
    pose, _, _, _ = _pose()
    pose[JOINT_INDEX["left_shoulder"]] = [0.0, 2.0, 0.0]
    pose[JOINT_INDEX["left_wrist"]] = [0.0, 5.0, 0.0]
    pose[JOINT_INDEX["left_elbow"]] = [1.0, 3.5, 0.0]
    valid = np.ones(17, dtype=bool)
    index = SIGN_FIELD_NAMES.index(_FIELD)
    assert int(sign_state(pose, valid)[index]) == UNKNOWN
    corrected, report = reconcile_hinge(
        pose, valid, 1, np.array([0.5, 0.5]), _context(), _FIELD)
    np.testing.assert_array_equal(corrected, pose)
    assert report["outcome"] == UNRESOLVED
