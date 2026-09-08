"""Composition contracts: learned bilateral conditioning + explicit hinge constraint.

docs/35. The hybrid only means anything if each mechanism provably owns its own
channel on real data. These pin the ownership boundary, the source-artifact
identity checks, and the replay's refusal behaviour.
"""

import json

import numpy as np
import pytest

from common.canonical_pose import FORWARD_DEPTH_AXIS, JOINT_INDEX
from framepose.replay_provenance import (
    SOURCE_IDENTITY_SCHEMA, UNVERIFIABLE, digest, verify_source_identity,
)
from framepose.signs import SIGN_FIELD_NAMES


def _load_script(name, relative):
    import importlib.util
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(root / "src"))
    try:
        spec = importlib.util.spec_from_file_location(name, root / relative)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


HYBRID = _load_script("replay_hybrid_signstate", "scripts/replay_hybrid_signstate.py")


class _FakeBank:
    def content_digest(self):
        return "digest0"

    def regime(self):
        return "benchmark_detector_observation"


def _write_prediction(tmp_path, frames=4, joints=17):
    path = tmp_path / "prediction_test.npy"
    np.save(path, np.zeros((frames, joints, 3), dtype=np.float32))
    return path


def _write_evaluation(tmp_path, **overrides):
    payload = {"schema": "animcv_frame_pose_evaluation_v1",
               "candidate": "O_BILATERAL_oracle_forward_depth_only",
               "observation_regime": ["benchmark_detector_observation"],
               "frame_count": 4}
    payload.update(overrides)
    path = tmp_path / "evaluation_test.json"
    path.write_text(json.dumps(payload))
    return path


# ------------------------------------------------- source-artifact identity ----

def test_source_identity_binds_bytes_and_parses_the_evaluation(tmp_path):
    prediction = _write_prediction(tmp_path)
    evaluation = _write_evaluation(tmp_path)
    record = verify_source_identity(
        prediction=prediction, evaluation=evaluation, bank=_FakeBank(), split="test",
        candidate="O_BILATERAL_oracle_forward_depth_only", frames=4, joints=17)

    assert record["schema"] == SOURCE_IDENTITY_SCHEMA
    assert record["prediction"] == digest(prediction)
    assert record["evaluation"]["sha256"] == digest(evaluation)["sha256"]
    checks = record["checks"]
    assert checks["evaluation_candidate_matches"] is True
    assert checks["evaluation_frame_count_matches"] is True
    assert checks["evaluation_regime_matches"] is True
    # The historical evaluation schema has no split field at all: say so rather
    # than claim a verification that did not happen.
    assert checks["evaluation_split_matches"] == UNVERIFIABLE


def test_source_identity_refuses_a_mismatched_candidate_or_frame_count(tmp_path):
    prediction = _write_prediction(tmp_path)
    bank = _FakeBank()

    wrong_candidate = _write_evaluation(tmp_path, candidate="S1_oracle_sign")
    with pytest.raises(ValueError, match="refusing to replay one candidate"):
        verify_source_identity(prediction=prediction, evaluation=wrong_candidate, bank=bank,
                               split="test", candidate="O_BILATERAL_oracle_forward_depth_only",
                               frames=4, joints=17)

    wrong_frames = _write_evaluation(tmp_path, frame_count=99)
    with pytest.raises(ValueError, match="reports 99 frames"):
        verify_source_identity(prediction=prediction, evaluation=wrong_frames, bank=bank,
                               split="test", candidate="O_BILATERAL_oracle_forward_depth_only",
                               frames=4, joints=17)

    wrong_regime = _write_evaluation(tmp_path, observation_regime=["oracle_geometry"])
    with pytest.raises(ValueError, match="records regimes"):
        verify_source_identity(prediction=prediction, evaluation=wrong_regime, bank=bank,
                               split="test", candidate="O_BILATERAL_oracle_forward_depth_only",
                               frames=4, joints=17)


def test_source_identity_refuses_a_prediction_of_the_wrong_shape(tmp_path):
    prediction = _write_prediction(tmp_path, frames=3)
    with pytest.raises(ValueError, match="has shape"):
        verify_source_identity(prediction=prediction, evaluation=None, bank=_FakeBank(),
                               split="test", candidate="whatever", frames=4, joints=17)


def test_missing_evaluation_is_recorded_as_unverifiable_not_passed(tmp_path):
    record = verify_source_identity(prediction=_write_prediction(tmp_path), evaluation=None,
                                    bank=_FakeBank(), split="test", candidate="c",
                                    frames=4, joints=17)
    checks = record["checks"]
    assert checks["evaluation_present"] is False
    for name in ("evaluation_candidate_matches", "evaluation_frame_count_matches",
                 "evaluation_regime_matches", "evaluation_split_matches"):
        assert checks[name] == UNVERIFIABLE


# ------------------------------------------------------ ownership boundary ----

def test_only_the_four_hinge_middle_joints_may_be_written():
    assert HYBRID.HINGE_JOINTS == ("left_elbow", "right_elbow", "left_knee", "right_knee")
    assert set(HYBRID.HINGE_MIDDLE_INDICES) == {JOINT_INDEX[name] for name in HYBRID.HINGE_JOINTS}

    before = np.zeros((3, 17, 3))
    after = before.copy()
    after[1, JOINT_INDEX["left_knee"], FORWARD_DEPTH_AXIS] = 0.05
    record = HYBRID._assert_ownership(before, after)
    assert record["wrote_only_permitted_joints"] is True
    assert record["wrote_only_the_depth_axis"] is True
    assert record["moved_joints"] == ["left_knee"]
    assert record["frames_with_any_write"] == 1


def test_writing_an_unowned_joint_is_refused():
    before = np.zeros((2, 17, 3))
    after = before.copy()
    after[0, JOINT_INDEX["left_hip"], FORWARD_DEPTH_AXIS] = 0.02
    with pytest.raises(ValueError, match="wrote joints it does not own"):
        HYBRID._assert_ownership(before, after)


def test_writing_a_non_depth_axis_is_refused():
    before = np.zeros((2, 17, 3))
    after = before.copy()
    after[0, JOINT_INDEX["left_knee"], 0] = 0.02
    with pytest.raises(ValueError, match="moved non-depth axes"):
        HYBRID._assert_ownership(before, after)


def _summary_stub(yaw, agreement):
    orientation = {name: None for name in HYBRID.BILATERAL_OWNED_METRICS}
    orientation["root_yaw_error_degrees"] = {"mean": yaw, "median": yaw, "p90": yaw,
                                             "p95": yaw, "count": 10}
    return {"orientation": orientation,
            "sign_agreement": {field: {"agreement": agreement}
                               for field in HYBRID.BILATERAL_OWNED_SIGN_FIELDS}}


def test_bilateral_preservation_is_exact_equality_not_a_tolerance():
    base = _summary_stub(8.0, 0.95)
    same = _summary_stub(8.0, 0.95)
    record = HYBRID._assert_bilateral_preserved(base, same)
    assert record["all_preserved_exactly"] is True
    assert record["root_yaw_error_degrees"]["identical"] is True

    # A change far below any plausible tolerance is still a violation: the
    # constraint does not write these joints, so exact equality is the contract.
    drifted = _summary_stub(8.0 + 1e-9, 0.95)
    with pytest.raises(ValueError, match="bilateral quantities changed"):
        HYBRID._assert_bilateral_preserved(base, drifted)

    flipped_sign = _summary_stub(8.0, 0.94)
    with pytest.raises(ValueError, match="sign_agreement"):
        HYBRID._assert_bilateral_preserved(base, flipped_sign)


def test_the_bilateral_owned_set_covers_every_non_hinge_sign_field():
    hinge = {name for name in SIGN_FIELD_NAMES if name.endswith("_forward_bend")}
    assert set(HYBRID.BILATERAL_OWNED_SIGN_FIELDS) == set(SIGN_FIELD_NAMES) - hinge
    assert set(HYBRID.HINGE_FIELDS) == hinge


# ------------------------------------------------ end-to-end on a real bank ----

def test_hybrid_replay_preserves_bilateral_and_is_deterministic(tmp_path, monkeypatch):
    """The whole composition on a synthetic bank: only hinge middles move, every
    bilateral quantity survives exactly, and two runs agree bit for bit."""
    import sys
    from pathlib import Path

    from framepose.bank import BankRequest, build_bank
    from framepose_fixtures import prepared_dataset

    sequences = {"train": ["seq_a", "seq_b"], "validation": ["seq_c"], "test": ["seq_d"]}
    requests = [BankRequest("3DPW", split, prepared_dataset(tmp_path / f"{split}.json",
                                                            split=split, sequences=names))
                for split, names in sequences.items()]
    built, _ = build_bank(requests, require_rgb=False)
    bank_path = tmp_path / "bank.json"
    built.save(bank_path)

    positions = built.indices("test")
    rng = np.random.default_rng(5)
    prediction = built.arrays["target_3d"][positions].astype(np.float32)
    prediction += rng.normal(scale=0.04, size=prediction.shape).astype(np.float32)
    prediction_path = tmp_path / "prediction_test.npy"
    np.save(prediction_path, prediction)

    out = tmp_path / "hybrid"
    argv = ["replay_hybrid_signstate.py", "--bank", str(bank_path),
            "--source", f"UNIT=UNIT_CANDIDATE:{prediction_path}",
            "--split", "test", "--out", str(out)]
    monkeypatch.setattr(sys, "argv", argv)
    assert HYBRID.main() == 0

    report = json.loads((out / "hybrid_signstate_replay.json").read_text())
    entry = report["sources"]["UNIT"]
    assert entry["ownership"]["wrote_only_permitted_joints"] is True
    assert entry["ownership"]["wrote_only_the_depth_axis"] is True
    assert set(entry["ownership"]["moved_joints"]) <= set(HYBRID.HINGE_JOINTS)
    assert entry["bilateral_preservation"]["all_preserved_exactly"] is True
    # Wrong hinge advice must stay local too: the same contract is asserted for
    # the opposite-oracle endpoint.
    assert entry["wrong_sign_endpoint"]["bilateral_preservation"]["all_preserved_exactly"] is True
    # The evaluator agrees with the prediction-level claim.
    for name in HYBRID.BILATERAL_OWNED_METRICS:
        assert entry["H0"]["orientation"][name] == entry["H1"]["orientation"][name], name

    # No evaluation was supplied, so the parse-level checks must say so.
    checks = report["provenance"]["sources"]["UNIT"]["checks"]
    assert checks["evaluation_present"] is False
    assert checks["evaluation_candidate_matches"] == UNVERIFIABLE

    # The cross-table accounts for every requested chain-frame exactly once.
    attribution = entry["residual_hinge_attribution"]
    for field, value in attribution["per_field"].items():
        assert sum(value["cross_table"].values()) == len(positions), field

    first = np.load(out / "prediction_test_H1_UNIT_HINGE.npy")
    monkeypatch.setattr(sys, "argv", argv)
    assert HYBRID.main() == 0
    np.testing.assert_array_equal(np.load(out / "prediction_test_H1_UNIT_HINGE.npy"), first)


def test_enforcement_effect_separates_fixing_from_breaking_the_historical_metric():
    """The one-bit branch and the full-3D metric are different predicates, so
    enforcement can move the metric either way. Both directions are counted."""
    from framepose.branch_constraints import ALREADY_SATISFIED, CORRECTED

    field = "left_knee_forward_bend"
    index = SIGN_FIELD_NAMES.index(field)
    column = HYBRID.HINGE_JOINTS.index("left_knee")
    frames = 5
    reports = [{"fields": {name: {"outcome": CORRECTED if name == field else ALREADY_SATISFIED}
                           for name in HYBRID.HINGE_FIELDS}} for _ in range(frames)]
    requested = np.zeros((frames, len(SIGN_FIELD_NAMES)), dtype=np.int64)
    requested[:, index] = 1
    requested[4, index] = 0  # not requested: must be skipped entirely

    base_flipped = np.zeros((frames, 4), dtype=np.int8)
    flipped = np.zeros((frames, 4), dtype=np.int8)
    base_flipped[0, column], flipped[0, column] = 1, 0   # fixed
    base_flipped[1, column], flipped[1, column] = 0, 1   # broken
    base_flipped[2, column], flipped[2, column] = 1, 1   # stayed flipped
    base_flipped[3, column], flipped[3, column] = 0, 0   # stayed fine
    base_error = np.full((frames, 4), 100.0)
    error = np.full((frames, 4), 40.0)

    result = HYBRID._enforcement_effect(reports, requested, base_flipped, base_error, flipped, error)
    counts = result["per_field"][field]["counts"]
    assert counts == {"broken_by_enforcement": 1, "fixed_by_enforcement": 1,
                      "stayed_flipped": 1, "stayed_not_flipped": 1}
    assert result["pooled"]["broken_by_enforcement"] == 1
    # Only CORRECTED, requested chain-frames are counted -- never the other three
    # fields, which were already satisfied, nor the unrequested frame.
    assert sum(counts.values()) == 4
    assert all(result["per_field"][other]["counts"] == {} for other in HYBRID.HINGE_FIELDS
               if other != field)
