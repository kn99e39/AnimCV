"""Focused accounting contracts for the controlled Pose Reconciliation replay."""

import importlib.util
from pathlib import Path

import numpy as np

from common.canonical_pose import JOINT_INDEX
from framepose.branch_constraints import ALREADY_SATISFIED, CORRECTED, UNRESOLVED
from framepose.pose_reconciliation import ProjectionContext
from framepose.signs import SIGN_FIELD_NAMES, mask_fields, sign_state


_SCRIPT = Path(__file__).parents[1] / "scripts" / "replay_pose_reconciliation.py"
_SPEC = importlib.util.spec_from_file_location("replay_pose_reconciliation", _SCRIPT)
replay = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(replay)


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
    pose[JOINT_INDEX["left_shoulder"]] = proximal
    pose[JOINT_INDEX["left_elbow"]] = center + 0.8 * u + 0.6 * v
    pose[JOINT_INDEX["left_wrist"]] = distal
    return pose


def _reports(frame_outcomes, requested):
    reports = []
    for order, outcome in enumerate(frame_outcomes):
        reports.append({
            "fields": {
                field: {
                    "outcome": (outcome if field == replay.HINGE_FIELDS[0] else "not_requested"),
                    "requested_sign": int(requested[order, SIGN_FIELD_NAMES.index(field)]),
                }
                for column, field in enumerate(replay.HINGE_FIELDS)
            }
        })
    return reports


def _two_frame_case():
    pose = _pose()
    valid = np.ones((2, 17), dtype=bool)
    h0 = np.repeat(pose[None, :, :], 2, axis=0)
    field = replay.HINGE_FIELDS[0]
    current = int(mask_fields(sign_state(pose, valid[0])[None, :], [field])[
        0, SIGN_FIELD_NAMES.index(field)])
    assert current in (-1, 1)
    requested = np.zeros((2, len(SIGN_FIELD_NAMES)), dtype=np.int64)
    requested[:, SIGN_FIELD_NAMES.index(field)] = -current
    observed_valid = np.ones((2, 17), dtype=bool)
    observation = np.zeros((2, 17, 3), dtype=np.float64)
    observation[:, :, :2] = 0.5
    target_absolute = np.repeat(pose[None, :, :], 2, axis=0)
    target_absolute[:, :, 1] += 5.0
    contexts = [
        ProjectionContext(np.diag([1000.0, 1000.0, 1.0]), (640.0, 480.0), [0.0, 5.0, 0.0]),
        ProjectionContext(np.diag([1000.0, 1000.0, 1.0]), (640.0, 480.0), [0.0, 5.0, 0.0]),
    ]
    reports = {
        replay.DEPTH_ONLY: _reports([CORRECTED, CORRECTED], requested),
        replay.MINIMUM_NORM: _reports([CORRECTED, CORRECTED], requested),
        replay.R_SWIVEL_OBS: _reports([CORRECTED, UNRESOLVED], requested),
    }
    return h0, requested, valid, observed_valid, observation, target_absolute, contexts, reports


def test_opposite_state_binds_to_base_candidate_report_key():
    normal = {replay.R_SWIVEL_OBS: [{"marker": "normal"}]}
    wrong = {replay.R_SWIVEL_OBS: [{"marker": "opposite"}]}
    assert replay._reports_for_state(
        f"{replay.R_SWIVEL_OBS}__opposite", normal, wrong)[0]["marker"] == "opposite"


def test_unknown_is_excluded_and_requested_coverage_identity_holds():
    pose = _pose()
    valid = np.ones((3, 17), dtype=bool)
    final_state = np.repeat(pose[None, :, :], 3, axis=0)
    field_column = 0
    field = replay.HINGE_FIELDS[field_column]
    sign = int(mask_fields(sign_state(pose, valid[0])[None, :], [field])[
        0, SIGN_FIELD_NAMES.index(field)])
    requested = np.zeros((3, len(SIGN_FIELD_NAMES)), dtype=np.int64)
    requested[1:, SIGN_FIELD_NAMES.index(field)] = sign
    reports = _reports([UNRESOLVED, CORRECTED, ALREADY_SATISFIED], requested)
    cohorts = {cohort: {name: np.zeros(3, dtype=bool) for name in replay.HINGE_FIELDS}
               for cohort in ("B",)}
    cohorts["B"][field][:] = True

    accounting = replay._reconciliation_outcome_counts(
        reports, requested, cohorts, final_state, valid)["B"][field]

    assert accounting["requested_count"] == 2
    assert accounting["satisfied_count"] == 2
    assert accounting["already_satisfied"] == 1
    assert accounting["corrected"] == 1
    assert accounting["unresolved"] == 0
    assert accounting["unknown_excluded"] == 1
    assert accounting["identity_holds"] is True


def test_c_keeps_unresolved_observation_rows_and_d_is_secondary_common_resolved():
    h0, requested, valid, observed_valid, observation, target_absolute, contexts, reports = _two_frame_case()
    cohorts = replay.build_cohorts(
        h0, requested, valid, observed_valid, observation, target_absolute, contexts, reports)
    field = replay.HINGE_FIELDS[0]

    np.testing.assert_array_equal(replay._cohort_rows(cohorts, "C", field), [0, 1])
    np.testing.assert_array_equal(replay._cohort_rows(cohorts, "D", field), [0])
    # The primary C rows are independent of the R_SWIVEL_OBS outcome, so all
    # four operational candidates receive exactly the same image rows.
    image_size = np.repeat(np.asarray([[640.0, 480.0]]), 2, axis=0)
    images = [replay._image_accounting(
        candidate, h0, h0, observation, target_absolute, image_size, contexts, cohorts, "C")
        for candidate in ("H0", replay.DEPTH_ONLY, replay.MINIMUM_NORM, replay.R_SWIVEL_OBS)]
    assert [image["fields"][field]["frames"] for image in images] == [2, 2, 2, 2]
    assert cohorts["D"][field].sum() < cohorts["C"][field].sum()


def test_observation_coordinate_space_is_explicit_provenance():
    assert replay.OBSERVATION_COORDINATE_SPACE == "normalized_full_image"
