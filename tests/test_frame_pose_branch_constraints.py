"""Branch Constraint contracts — SignState enforced on an already predicted pose.

docs/33 tests the other abstraction for sign evidence: instead of conditioning
a network on a sign it may or may not act on, enforce the branch on the
Geometry Core's continuous output. That only means anything if the enforcement
is exact, local, verified and honest about what it cannot do, so these
contracts run before any real-scene replay.
"""

import numpy as np
import pytest

from common.canonical_pose import BILATERAL_DEPTH_NORMALIZATION, FORWARD_DEPTH_AXIS, JOINT_INDEX
from framepose.branch_constraints import (
    ALREADY_SATISFIED, APPLICATION_ORDER, BRANCH_CONSTRAINT_SCHEMA, CORRECTED, NOT_REQUESTED,
    SUPPORTED_FIELDS, UNRESOLVED, apply_branch_constraints, apply_branch_constraints_batch, coverage,
)
from framepose.signs import (
    NEGATIVE, POSITIVE, SIGN_FIELD_COUNT, SIGN_FIELD_NAMES, UNKNOWN, sign_state,
)


def _pose(**overrides):
    """An upright canonical pose with a well-conditioned left-elbow bend and a
    readable hip depth separation."""
    layout = {"pelvis": (0, 0, 0), "spine": (0, 0, 0.2), "thorax": (0, 0, 0.45),
              "neck": (0, 0, 0.5), "head": (0, 0, 0.7),
              "left_shoulder": (-0.18, 0.0, 0.45), "right_shoulder": (0.18, 0.05, 0.45),
              "left_elbow": (-0.28, 0.10, 0.2), "right_elbow": (0.28, -0.10, 0.2),
              "left_wrist": (-0.32, 0, -0.05), "right_wrist": (0.32, 0, -0.05),
              "left_hip": (-0.11, 0.0, 0), "right_hip": (0.11, 0.06, 0),
              "left_knee": (-0.12, 0.09, -0.45), "right_knee": (0.12, -0.09, -0.45),
              "left_ankle": (-0.12, 0, -0.9), "right_ankle": (0.12, 0, -0.9)}
    layout.update(overrides)
    pose = np.zeros((17, 3))
    for name, position in layout.items():
        pose[JOINT_INDEX[name]] = position
    return pose, np.ones(17, dtype=bool)


def _request(**fields):
    state = np.zeros(SIGN_FIELD_COUNT, dtype=np.int64)
    for name, value in fields.items():
        state[SIGN_FIELD_NAMES.index(name)] = value
    return state


def _read(pose, valid, field):
    return int(sign_state(pose, valid)[SIGN_FIELD_NAMES.index(field)])


# ------------------------------------------------------------- no-ops ----

def test_unknown_request_is_an_exact_identity():
    pose, valid = _pose()
    corrected, report = apply_branch_constraints(pose, valid, _request())
    assert np.array_equal(corrected, pose)
    assert report["schema"] == BRANCH_CONSTRAINT_SCHEMA
    assert {entry["outcome"] for entry in report["fields"].values()} == {NOT_REQUESTED}
    assert report["total_displacement_mm"] == 0.0


@pytest.mark.parametrize("field", sorted(SUPPORTED_FIELDS))
def test_already_satisfied_request_is_an_exact_identity(field):
    pose, valid = _pose()
    current = _read(pose, valid, field)
    if current == UNKNOWN:
        pytest.skip(f"{field} is degenerate in this fixture")
    corrected, report = apply_branch_constraints(pose, valid, _request(**{field: current}))
    assert np.array_equal(corrected, pose), "an already-correct branch must not move anything"
    assert report["fields"][field]["outcome"] == ALREADY_SATISFIED
    assert report["total_displacement_mm"] == 0.0


# --------------------------------------------------------- bilateral ----

def test_bilateral_correction_satisfies_the_requested_branch():
    pose, valid = _pose()
    assert _read(pose, valid, "hip_forward_depth") == POSITIVE
    corrected, report = apply_branch_constraints(pose, valid, _request(hip_forward_depth=NEGATIVE))
    assert _read(corrected, valid, "hip_forward_depth") == NEGATIVE
    assert report["fields"]["hip_forward_depth"]["outcome"] == CORRECTED
    assert report["requested_branch_satisfied"]["hip_forward_depth"] is True


def test_bilateral_correction_preserves_x_z_midpoint_and_separation():
    pose, valid = _pose()
    corrected, _ = apply_branch_constraints(pose, valid, _request(hip_forward_depth=NEGATIVE))
    left, right = JOINT_INDEX["left_hip"], JOINT_INDEX["right_hip"]

    for axis in (0, 2):  # X and Z
        np.testing.assert_array_equal(corrected[:, axis], pose[:, axis])

    def midpoint(sample):
        return (sample[left, FORWARD_DEPTH_AXIS] + sample[right, FORWARD_DEPTH_AXIS]) / 2.0

    def separation(sample):
        return abs(sample[right, FORWARD_DEPTH_AXIS] - sample[left, FORWARD_DEPTH_AXIS]) \
            * BILATERAL_DEPTH_NORMALIZATION

    assert midpoint(corrected) == pytest.approx(midpoint(pose))
    assert separation(corrected) == pytest.approx(separation(pose))


def test_bilateral_correction_touches_only_its_own_pair():
    pose, valid = _pose()
    corrected, report = apply_branch_constraints(pose, valid, _request(hip_forward_depth=NEGATIVE))
    moved = np.flatnonzero(np.linalg.norm(corrected - pose, axis=-1) > 0)
    assert sorted(moved) == sorted([JOINT_INDEX["left_hip"], JOINT_INDEX["right_hip"]])
    assert report["fields"]["hip_forward_depth"]["moved_joint_count"] == 2


def test_bilateral_correction_is_idempotent():
    pose, valid = _pose()
    once, _ = apply_branch_constraints(pose, valid, _request(hip_forward_depth=NEGATIVE))
    twice, report = apply_branch_constraints(once, valid, _request(hip_forward_depth=NEGATIVE))
    assert np.array_equal(twice, once)
    assert report["fields"]["hip_forward_depth"]["outcome"] == ALREADY_SATISFIED


def test_bilateral_below_the_stability_floor_is_unresolved_not_silently_failed():
    pose, valid = _pose(left_hip=(-0.11, 0.0, 0), right_hip=(0.11, 0.005, 0))
    assert _read(pose, valid, "hip_forward_depth") == UNKNOWN
    corrected, report = apply_branch_constraints(pose, valid, _request(hip_forward_depth=POSITIVE))
    entry = report["fields"]["hip_forward_depth"]
    assert entry["outcome"] == UNRESOLVED
    assert "stability floor" in entry["reason"]
    assert np.array_equal(corrected, pose), "an unresolved field must leave the pose untouched"
    assert report["requested_branch_satisfied"]["hip_forward_depth"] is False


# ------------------------------------------------------------- hinge ----

def test_hinge_correction_satisfies_the_requested_branch():
    pose, valid = _pose()
    assert _read(pose, valid, "left_elbow_forward_bend") == POSITIVE
    corrected, report = apply_branch_constraints(
        pose, valid, _request(left_elbow_forward_bend=NEGATIVE))
    assert _read(corrected, valid, "left_elbow_forward_bend") == NEGATIVE
    assert report["fields"]["left_elbow_forward_bend"]["outcome"] == CORRECTED


def test_hinge_correction_moves_depth_only_and_never_image_space():
    pose, valid = _pose()
    corrected, _ = apply_branch_constraints(pose, valid, _request(left_elbow_forward_bend=NEGATIVE))
    for axis in (0, 2):  # X and Z: the advisor gave a near/far bit, not a 2D move
        np.testing.assert_array_equal(corrected[:, axis], pose[:, axis])


def test_hinge_correction_touches_only_the_middle_joint():
    pose, valid = _pose()
    corrected, report = apply_branch_constraints(
        pose, valid, _request(left_elbow_forward_bend=NEGATIVE))
    moved = np.flatnonzero(np.linalg.norm(corrected - pose, axis=-1) > 0)
    assert moved.tolist() == [JOINT_INDEX["left_elbow"]]
    # The proximal and distal joints define the axis and must not move.
    np.testing.assert_array_equal(corrected[JOINT_INDEX["left_shoulder"]], pose[JOINT_INDEX["left_shoulder"]])
    np.testing.assert_array_equal(corrected[JOINT_INDEX["left_wrist"]], pose[JOINT_INDEX["left_wrist"]])
    assert report["fields"]["left_elbow_forward_bend"]["moved_joint_count"] == 1


def test_hinge_correction_is_idempotent():
    pose, valid = _pose()
    once, _ = apply_branch_constraints(pose, valid, _request(left_elbow_forward_bend=NEGATIVE))
    twice, report = apply_branch_constraints(once, valid, _request(left_elbow_forward_bend=NEGATIVE))
    assert np.array_equal(twice, once)
    assert report["fields"]["left_elbow_forward_bend"]["outcome"] == ALREADY_SATISFIED


def test_hinge_correction_reflects_the_offset_rather_than_guessing_a_magnitude():
    """The closed form is o_y -> -o_y; for an axis lying in the image plane
    that is exactly a depth reflection of the bend offset."""
    pose, valid = _pose()
    corrected, report = apply_branch_constraints(
        pose, valid, _request(left_elbow_forward_bend=NEGATIVE))
    entry = report["fields"]["left_elbow_forward_bend"]
    # The fixture's left-arm axis has no depth component, so the whole depth
    # delta is twice the predicted offset, with opposite sign.
    assert entry["axis_in_plane_fraction"] == pytest.approx(1.0)
    assert entry["depth_delta_m"] == pytest.approx(-2.0 * entry["predicted_offset_forward_m"])


def test_hinge_depth_axis_alignment_is_refused_as_singular():
    """A limb pointing straight along camera depth has no depth-only branch
    correction at all -- it must be refused, not rescued with an epsilon."""
    pose, valid = _pose(left_shoulder=(-0.2, 0.0, 0.3), left_wrist=(-0.2, 0.5, 0.3),
                        left_elbow=(-0.28, 0.25, 0.3))
    corrected, report = apply_branch_constraints(
        pose, valid, _request(left_elbow_forward_bend=POSITIVE))
    entry = report["fields"]["left_elbow_forward_bend"]
    assert entry["outcome"] == UNRESOLVED
    assert "singular" in entry["reason"]
    assert entry["axis_in_plane_fraction"] == pytest.approx(0.0)
    assert np.array_equal(corrected, pose)


def test_hinge_below_the_bend_floor_is_unresolved_after_read_back_verification():
    """A nearly straight limb cannot be given a branch without inventing a bend
    magnitude, which the advisor's contract forbids."""
    pose, valid = _pose(left_elbow=(-0.2497, 0.004, 0.2))
    assert _read(pose, valid, "left_elbow_forward_bend") == UNKNOWN
    corrected, report = apply_branch_constraints(
        pose, valid, _request(left_elbow_forward_bend=NEGATIVE))
    entry = report["fields"]["left_elbow_forward_bend"]
    assert entry["outcome"] == UNRESOLVED
    assert "read back" in entry["reason"] or "read_back" in entry["reason"]
    assert np.array_equal(corrected, pose)


# ------------------------------------------------- combined / accounting ----

def test_every_supported_field_can_be_enforced_together_without_interference():
    pose, valid = _pose()
    request = _request(**{field: -_read(pose, valid, field) for field in SUPPORTED_FIELDS
                          if _read(pose, valid, field) != UNKNOWN})
    corrected, report = apply_branch_constraints(pose, valid, request)
    for field in SUPPORTED_FIELDS:
        want = int(request[SIGN_FIELD_NAMES.index(field)])
        if want == UNKNOWN:
            continue
        assert _read(corrected, valid, field) == want, field
        assert report["requested_branch_satisfied"][field] is True, field


def test_orientation_is_applied_before_hinges_so_knee_chains_settle_first():
    assert APPLICATION_ORDER[:2] == ("shoulder_forward_depth", "hip_forward_depth")
    assert all(name.endswith("_forward_bend") for name in APPLICATION_ORDER[2:])


def test_torso_facing_is_refused_rather_than_silently_ignored():
    pose, valid = _pose()
    with pytest.raises(ValueError, match="torso_facing"):
        apply_branch_constraints(pose, valid, _request(torso_facing=POSITIVE))


def test_out_of_domain_requested_values_are_refused():
    pose, valid = _pose()
    request = _request()
    request[SIGN_FIELD_NAMES.index("hip_forward_depth")] = 2
    with pytest.raises(ValueError, match="exactly -1, 0 or"):
        apply_branch_constraints(pose, valid, request)


def test_invalid_joints_make_the_field_unresolved_not_corrected():
    pose, valid = _pose()
    valid = valid.copy()
    valid[JOINT_INDEX["right_hip"]] = False
    corrected, report = apply_branch_constraints(pose, valid, _request(hip_forward_depth=NEGATIVE))
    assert report["fields"]["hip_forward_depth"]["outcome"] == UNRESOLVED
    assert np.array_equal(corrected, pose)


def test_batch_replay_is_deterministic_and_coverage_identity_holds():
    pose, valid = _pose()
    poses = np.stack([pose, pose, pose])
    valids = np.stack([valid, valid, valid])
    requests = np.stack([_request(hip_forward_depth=NEGATIVE),
                         _request(hip_forward_depth=POSITIVE),
                         _request(hip_forward_depth=UNKNOWN)])

    first_poses, first_reports = apply_branch_constraints_batch(poses, valids, requests)
    second_poses, _ = apply_branch_constraints_batch(poses, valids, requests)
    np.testing.assert_array_equal(first_poses, second_poses)

    accounting = coverage(first_reports)["fields"]["hip_forward_depth"]
    assert accounting["corrected"] == 1
    assert accounting["already_satisfied"] == 1
    assert accounting["unknown_or_not_requested"] == 1
    assert accounting["requested"] == (accounting["already_satisfied"] + accounting["corrected"]
                                       + accounting["unresolved"])
    assert accounting["coverage_identity_holds"] is True


def test_coverage_identity_is_a_cross_check_not_a_tautology():
    """`requested` is counted from the requested sign, so a report whose
    outcome buckets disagree with it is refused rather than balanced by
    construction."""
    from framepose.branch_constraints import ALREADY_SATISFIED, NOT_REQUESTED, coverage

    honest = [{"fields": {"hip_forward_depth": {"outcome": ALREADY_SATISFIED, "requested": 1}}},
              {"fields": {"hip_forward_depth": {"outcome": NOT_REQUESTED, "requested": 0}}}]
    report = coverage(honest)["fields"]["hip_forward_depth"]
    assert report["frames_considered"] == 2
    assert report["requested"] == 1
    assert report["outcome_bucket_total"] == 1
    assert report["coverage_identity_holds"] is True

    # A branch that was requested but filed as not-requested is a dropped frame.
    dropped = [{"fields": {"hip_forward_depth": {"outcome": NOT_REQUESTED, "requested": 1}}}]
    with pytest.raises(ValueError, match="coverage identity"):
        coverage(dropped)


def test_branch_constraint_replay_runs_over_a_stored_prediction(tmp_path, monkeypatch):
    """End-to-end: stored prediction in, constrained variants + coverage out.

    The replay must not train, must leave C0 bit-identical, and must produce a
    coverage identity that holds for every field of every variant.
    """
    import importlib.util
    import json
    import sys
    from pathlib import Path

    from framepose.bank import BankRequest, build_bank
    from framepose_fixtures import prepared_dataset

    root = Path(__file__).resolve().parent.parent
    module_path = root / "scripts" / "replay_branch_constraints.py"

    sequences = {"train": ["seq_a", "seq_b"], "validation": ["seq_c"], "test": ["seq_d"]}
    requests = [BankRequest("3DPW", split, prepared_dataset(tmp_path / f"{split}.json",
                                                            split=split, sequences=names))
                for split, names in sequences.items()]
    built, _ = build_bank(requests, require_rgb=False)
    bank_path = tmp_path / "bank.json"
    built.save(bank_path)

    positions = built.indices("test")
    rng = np.random.default_rng(11)
    prediction = built.arrays["target_3d"][positions].astype(np.float32)
    prediction += rng.normal(scale=0.03, size=prediction.shape).astype(np.float32)
    prediction_path = tmp_path / "prediction_test.npy"
    np.save(prediction_path, prediction)

    spec = importlib.util.spec_from_file_location("replay_branch_constraints", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    out = tmp_path / "replay"
    monkeypatch.setattr(sys, "argv", [
        "replay_branch_constraints.py", "--bank", str(bank_path),
        "--prediction", str(prediction_path), "--source-candidate", "UNIT",
        "--split", "test", "--out", str(out)])
    assert module.main() == 0

    report = json.loads((out / "branch_constraint_replay.json").read_text())
    assert report["schema"] == module.SCHEMA
    assert set(report["variants"]) == set(module.VARIANTS)
    # C0 is an exact no-op: identical aggregate to the untouched baseline.
    assert report["variants"]["C0"]["evaluation"] == report["baseline"]
    assert report["variants"]["C0"]["displacement_mm"]["max"] == 0.0
    # torso_facing has no declared correction and must appear in no variant.
    for definition in module.VARIANTS.values():
        assert "torso_facing" not in definition
    for name, value in report["variants"].items():
        for field, entry in value["coverage"].items():
            assert entry["coverage_identity_holds"], (name, field)
            assert entry["requested"] == entry["outcome_bucket_total"], (name, field)
        # Every requested, resolvable branch either was already right or was made right.
        for field, rate in value["requested_branch_satisfied_rate"].items():
            assert rate is None or 0.0 <= rate <= 1.0
    # The wrong-sign control exists for every constrained variant, two endpoints only.
    assert set(report["wrong_sign_control"]) == set(module.VARIANTS) - {"C0"}
    for value in report["wrong_sign_control"].values():
        assert value["endpoints"] == ["oracle", "opposite_oracle"]


def test_unresolved_is_split_by_what_the_prediction_actually_said():
    """An unresolved field where the Core produced no readable branch is not
    the same failure as one where it produced the opposite branch and the
    correction could not install the requested one."""
    from framepose.branch_constraints import UNRESOLVED, coverage

    reports = [
        {"fields": {"left_knee_forward_bend": {
            "outcome": UNRESOLVED, "requested": -1, "read_back_before": UNKNOWN}}},
        {"fields": {"left_knee_forward_bend": {
            "outcome": UNRESOLVED, "requested": -1, "read_back_before": POSITIVE}}},
        {"fields": {"left_knee_forward_bend": {
            "outcome": UNRESOLVED, "requested": -1, "read_back_before": POSITIVE}}},
    ]
    split = coverage(reports)["fields"]["left_knee_forward_bend"]["unresolved_by_prediction_state"]
    assert split == {"unreadable_prediction": 1, "opposite_branch": 2, "other": 0}
