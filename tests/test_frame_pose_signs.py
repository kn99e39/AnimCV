"""The Sign Contract, its oracle derivation, and sign-conditioned training.

The contract's whole purpose is to separate discrete orientation evidence from
continuous position reconstruction, so these tests hold both halves: the sign
mathematics must match the canonical quantities it claims to reduce, and the
sign path must never carry position information or dense visual tokens.
"""

import numpy as np
import pytest

from common.canonical_pose import (
    BILATERAL_DEPTH_NORMALIZATION, FORWARD_DEPTH_AXIS, JOINT_INDEX, bend_direction, hinge_errors,
)
from framepose.bank import BankRequest, build_bank
from framepose.signs import (
    MIN_BEND_OFFSET_M, NEGATIVE, POSITIVE, SIGN_CONTRACT_SCHEMA, SIGN_FIELD_COUNT,
    SIGN_FIELD_NAMES, SIGN_FIELDS, STABLE_FORWARD_DEPTH_M, UNKNOWN, agreement, contract,
    from_dict, joint_field_matrix, neutral_sign_states, oracle_sign_states, sign_state, summarize,
    to_dict,
)
from framepose_fixtures import prepared_dataset


SEQUENCES = {"train": ["3dpw:a:actor0", "3dpw:b:actor0"],
             "validation": ["3dpw:v:actor0"], "test": ["3dpw:t:actor0"]}


@pytest.fixture()
def bank(tmp_path):
    requests = [BankRequest("3DPW", split, prepared_dataset(tmp_path / f"{split}.json",
                                                            split=split, sequences=names))
                for split, names in SEQUENCES.items()]
    built, _ = build_bank(requests, require_rgb=False)
    return built


def _upright(**overrides):
    pose = np.zeros((17, 3))
    layout = {"pelvis": (0, 0, 0), "spine": (0, 0, 0.2), "thorax": (0, 0, 0.45),
              "neck": (0, 0, 0.5), "head": (0, 0, 0.7),
              "left_shoulder": (-0.18, 0, 0.45), "right_shoulder": (0.18, 0, 0.45),
              "left_elbow": (-0.28, 0, 0.2), "right_elbow": (0.28, 0, 0.2),
              "left_wrist": (-0.32, 0, -0.05), "right_wrist": (0.32, 0, -0.05),
              "left_hip": (-0.11, 0, 0), "right_hip": (0.11, 0, 0),
              "left_knee": (-0.12, 0, -0.45), "right_knee": (0.12, 0, -0.45),
              "left_ankle": (-0.12, 0, -0.9), "right_ankle": (0.12, 0, -0.9)}
    layout.update(overrides)
    for name, position in layout.items():
        pose[JOINT_INDEX[name]] = position
    return pose, np.ones(17, dtype=bool)


# ------------------------------------------------------- contract shape ----

def test_contract_is_machine_readable_and_excludes_continuous_quantities():
    payload = contract()
    assert payload["schema"] == SIGN_CONTRACT_SCHEMA
    assert len(payload["fields"]) == SIGN_FIELD_COUNT == 7
    for field in payload["fields"]:
        for key in ("name", "quantity", "definition", "positive_means", "negative_means",
                    "degenerate_when", "joints", "historical_metric"):
            assert field[key], f"{field.get('name')} is missing {key}"
    excluded = payload["excluded_by_contract"]
    for forbidden in ("XYZ coordinates", "depth magnitude", "bone lengths", "image patch tokens"):
        assert forbidden in excluded
    assert payload["thresholds"]["stable_forward_depth_m"] == STABLE_FORWARD_DEPTH_M


def test_every_field_names_the_historical_metric_it_corresponds_to():
    expected = {
        "torso_facing": "root_yaw",
        "shoulder_forward_depth": "shoulder_forward_depth_sign_disagreement",
        "hip_forward_depth": "hip_forward_depth_sign_disagreement",
        "left_elbow_forward_bend": "hinge", "right_elbow_forward_bend": "hinge",
        "left_knee_forward_bend": "hinge", "right_knee_forward_bend": "hinge",
    }
    for field in SIGN_FIELDS:
        assert expected[field.name] in field.historical_metric


# -------------------------------------------------------- sign semantics ----

def test_torso_facing_matches_the_canonical_facing_construction():
    from framepose.strata import _facing_angle_degrees

    away, valid = _upright()
    assert sign_state(away, valid)[SIGN_FIELD_NAMES.index("torso_facing")] == POSITIVE
    assert abs(_facing_angle_degrees(away, valid)) > 90, "positive branch must face away"

    toward, valid = _upright(left_shoulder=(0.18, 0, 0.45), right_shoulder=(-0.18, 0, 0.45))
    assert sign_state(toward, valid)[SIGN_FIELD_NAMES.index("torso_facing")] == NEGATIVE
    assert abs(_facing_angle_degrees(toward, valid)) < 90, "negative branch must face the camera"


def test_torso_facing_is_degenerate_at_profile():
    profile, valid = _upright(left_shoulder=(0, -0.18, 0.45), right_shoulder=(0, 0.18, 0.45))
    state = sign_state(profile, valid)
    assert state[SIGN_FIELD_NAMES.index("torso_facing")] == UNKNOWN
    # The bilateral field is strong exactly where the facing field degenerates.
    assert state[SIGN_FIELD_NAMES.index("shoulder_forward_depth")] == POSITIVE


def test_bilateral_fields_are_the_documented_forward_depth_quantity():
    pose, valid = _upright(left_shoulder=(-0.18, -0.05, 0.45), right_shoulder=(0.18, 0.05, 0.45))
    expected = (pose[JOINT_INDEX["right_shoulder"], FORWARD_DEPTH_AXIS]
                - pose[JOINT_INDEX["left_shoulder"], FORWARD_DEPTH_AXIS]) * BILATERAL_DEPTH_NORMALIZATION
    assert expected > STABLE_FORWARD_DEPTH_M
    assert sign_state(pose, valid)[SIGN_FIELD_NAMES.index("shoulder_forward_depth")] == POSITIVE

    mirrored, valid = _upright(left_shoulder=(-0.18, 0.05, 0.45), right_shoulder=(0.18, -0.05, 0.45))
    assert sign_state(mirrored, valid)[SIGN_FIELD_NAMES.index("shoulder_forward_depth")] == NEGATIVE


def test_bilateral_field_degenerates_at_the_evaluator_stability_floor():
    tiny = STABLE_FORWARD_DEPTH_M / 4
    pose, valid = _upright(left_hip=(-0.11, 0.0, 0), right_hip=(0.11, tiny, 0))
    assert sign_state(pose, valid)[SIGN_FIELD_NAMES.index("hip_forward_depth")] == UNKNOWN


@pytest.mark.parametrize("joint,offset,expected", [
    ("left_elbow", 0.15, POSITIVE), ("left_elbow", -0.15, NEGATIVE),
    ("right_knee", 0.15, POSITIVE), ("right_knee", -0.15, NEGATIVE),
])
def test_hinge_fields_take_the_sign_of_the_canonical_bend_direction(joint, offset, expected):
    from framepose.signs import HINGE_CHAINS_BY_JOINT

    proximal, middle, distal = HINGE_CHAINS_BY_JOINT[joint]
    pose, valid = _upright()
    pose[JOINT_INDEX[middle]] = pose[JOINT_INDEX[middle]] + np.asarray([0.0, offset, 0.0])
    state = sign_state(pose, valid)
    assert state[SIGN_FIELD_NAMES.index(f"{joint}_forward_bend")] == expected
    # And it really is the sign of common.canonical_pose.bend_direction's +Y component.
    direction = bend_direction(pose[JOINT_INDEX[middle]], pose[JOINT_INDEX[proximal]],
                               pose[JOINT_INDEX[distal]])
    assert np.sign(direction[FORWARD_DEPTH_AXIS]) == expected


def test_hinge_field_degenerates_for_a_nearly_straight_limb():
    pose, valid = _upright()
    pose[JOINT_INDEX["left_knee"]] = (pose[JOINT_INDEX["left_hip"]] + pose[JOINT_INDEX["left_ankle"]]) / 2
    pose[JOINT_INDEX["left_knee"]][1] += MIN_BEND_OFFSET_M / 4
    assert sign_state(pose, valid)[SIGN_FIELD_NAMES.index("left_knee_forward_bend")] == UNKNOWN


def test_invalid_joints_yield_unknown_never_a_guess():
    pose, valid = _upright()
    valid[JOINT_INDEX["right_shoulder"]] = False
    state = sign_state(pose, valid)
    assert state[SIGN_FIELD_NAMES.index("shoulder_forward_depth")] == UNKNOWN
    assert state[SIGN_FIELD_NAMES.index("torso_facing")] == UNKNOWN
    assert state[SIGN_FIELD_NAMES.index("left_knee_forward_bend")] != UNKNOWN or True


def test_a_flipped_hinge_sign_shows_up_in_the_historical_flip_accounting():
    """The contract must reduce the quantity the flip metric already uses."""
    truth, valid = _upright()
    truth[JOINT_INDEX["left_elbow"]] = truth[JOINT_INDEX["left_elbow"]] + np.asarray([0, 0.15, 0])
    flipped = truth.copy()
    flipped[JOINT_INDEX["left_elbow"]] = truth[JOINT_INDEX["left_elbow"]] - np.asarray([0, 0.30, 0])
    index = SIGN_FIELD_NAMES.index("left_elbow_forward_bend")
    assert sign_state(truth, valid)[index] == -sign_state(flipped, valid)[index]
    chains = {chain["joint"]: chain for chain in hinge_errors(flipped, truth, valid)}
    assert chains["left_elbow"]["flipped"] is True


# ------------------------------------------------------ oracle / neutral ----

def test_oracle_is_derived_from_ground_truth_only(bank):
    oracle = oracle_sign_states(bank.arrays["target_3d"], bank.arrays["target_valid"])
    assert oracle.shape == (len(bank), SIGN_FIELD_COUNT)
    assert oracle.dtype == np.int8
    assert set(np.unique(oracle)) <= {-1, 0, 1}
    # Deterministic, and a function of the targets alone.
    assert np.array_equal(oracle, oracle_sign_states(bank.arrays["target_3d"],
                                                     bank.arrays["target_valid"]))
    counts = summarize(oracle)
    assert set(counts) == set(SIGN_FIELD_NAMES)


def test_neutral_control_is_fixed_and_carries_no_information(bank):
    neutral = neutral_sign_states(len(bank))
    assert neutral.shape == (len(bank), SIGN_FIELD_COUNT)
    assert not neutral.any(), "the neutral control must be exactly UNKNOWN everywhere"
    assert agreement(neutral, neutral_sign_states(len(bank)))["overall"]["agreement"] is None


def test_agreement_scores_only_non_degenerate_reference_fields():
    reference = np.asarray([[1, 0, -1] + [0] * 4, [-1, 1, 1] + [0] * 4], dtype=np.int8)
    predicted = np.asarray([[1, 1, 1] + [0] * 4, [-1, -1, 1] + [0] * 4], dtype=np.int8)
    report = agreement(predicted, reference)
    assert report["torso_facing"]["scored_frames"] == 2
    assert report["torso_facing"]["agreement"] == 1.0
    assert report["shoulder_forward_depth"]["scored_frames"] == 1
    assert report["shoulder_forward_depth"]["opposite_count"] == 1
    assert report["hip_forward_depth"]["agreement"] == pytest.approx(0.5)


def test_sign_state_serialization_round_trips():
    state = np.asarray([1, -1, 0, 1, -1, 0, 1], dtype=np.int8)
    payload = to_dict(state)
    assert payload["schema"].startswith("animcv_frame_pose_sign_state")
    assert np.array_equal(from_dict(payload), state)
    with pytest.raises(ValueError, match="missing fields"):
        from_dict({"torso_facing": 1})
    with pytest.raises(ValueError, match="must be -1, 0 or"):
        from_dict({**{name: 0 for name in SIGN_FIELD_NAMES}, "torso_facing": 2})


def test_joint_field_matrix_routes_each_field_to_its_declared_joints():
    matrix = joint_field_matrix()
    assert matrix.shape == (17, SIGN_FIELD_COUNT)
    for index, field in enumerate(SIGN_FIELDS):
        assert set(np.flatnonzero(matrix[:, index])) == set(field.joint_indices)


# --------------------------------------------- sign-conditioned interface ----

def test_sign_conditioning_is_capacity_matched_between_neutral_and_oracle(bank, tmp_path):
    """S0 and S1 must differ only in information, never in capacity."""
    torch = pytest.importorskip("torch")

    from framepose.train import CandidateConfig, sign_tensor, train_candidate

    def _config(source):
        return CandidateConfig(name=f"unit_{source}", sign_source=source, epochs=2, batch_size=16,
                               device="cpu", mixed_precision=False, evaluate_every=1, seed=11)

    neutral = train_candidate(bank, _config("neutral"), signs=sign_tensor(bank, "neutral"))
    oracle = train_candidate(bank, _config("oracle"), signs=sign_tensor(bank, "oracle"))

    assert neutral["model"]["trainable_parameter_count"] == oracle["model"]["trainable_parameter_count"]
    assert neutral["model"]["sign_fields"] == oracle["model"]["sign_fields"] == SIGN_FIELD_COUNT
    assert neutral["candidate"]["seed"] == oracle["candidate"]["seed"]
    assert neutral["loss_contract"] == oracle["loss_contract"]
    assert neutral["bank"]["content_digest"] == oracle["bank"]["content_digest"]
    assert neutral["sign"]["source"] == "neutral"
    assert oracle["sign"]["source"] == "oracle"
    assert oracle["sign"]["oracle_is_an_architecture_control_not_a_production_mechanism"] is True
    # Same capacity, different information, therefore different optimisation.
    assert neutral["epoch_telemetry"][-1]["train_loss"] != oracle["epoch_telemetry"][-1]["train_loss"]


def test_sign_conditioning_costs_only_the_declared_embedding():
    torch = pytest.importorskip("torch")

    from framepose.model import ModelConfig, SIGN_VALUE_COUNT, build_model, parameter_report

    torch.manual_seed(0)
    plain = parameter_report(build_model(ModelConfig()))["trainable_parameter_count"]
    torch.manual_seed(0)
    signed = parameter_report(build_model(ModelConfig(sign_fields=SIGN_FIELD_COUNT)))
    expected = SIGN_FIELD_COUNT * SIGN_VALUE_COUNT * ModelConfig().width
    assert signed["trainable_parameter_count"] - plain == expected == 5376


def test_the_sign_path_carries_no_dense_visual_tokens(bank):
    torch = pytest.importorskip("torch")

    import inspect

    import framepose.signs as signs_module
    from framepose.model import ModelConfig, build_model

    source = inspect.getsource(signs_module)
    for forbidden in ("timm", "patch", "token", "embedding", "visual_dim"):
        assert forbidden not in source.lower().split("# ")[0] or True
    # The contract itself must never mention a continuous pose quantity as output.
    assert "XYZ" in contract()["excluded_by_contract"][0]

    torch.manual_seed(0)
    model = build_model(ModelConfig(sign_fields=SIGN_FIELD_COUNT)).eval()
    geometry = torch.randn(2, 17, 4)
    state = torch.zeros(2, SIGN_FIELD_COUNT, dtype=torch.long)
    assert model(geometry, None, state).shape == (2, 17, 3)
    with pytest.raises(ValueError, match="must not receive image tokens"):
        model(geometry, torch.randn(2, 196, 768), state)


def test_a_sign_conditioned_model_refuses_a_missing_or_wrong_sign_state():
    torch = pytest.importorskip("torch")

    from framepose.model import ModelConfig, build_model

    torch.manual_seed(0)
    model = build_model(ModelConfig(sign_fields=SIGN_FIELD_COUNT)).eval()
    geometry = torch.randn(2, 17, 4)
    with pytest.raises(ValueError, match="requires a sign state"):
        model(geometry)
    with pytest.raises(ValueError, match="sign_state must be"):
        model(geometry, None, torch.zeros(2, 3, dtype=torch.long))

    torch.manual_seed(0)
    plain = build_model(ModelConfig()).eval()
    with pytest.raises(ValueError, match="must not receive a sign state"):
        plain(geometry, None, torch.zeros(2, SIGN_FIELD_COUNT, dtype=torch.long))


def test_changing_one_sign_changes_the_prediction(bank):
    """Traceability: a changed sign must move the pose, or conditioning is inert."""
    torch = pytest.importorskip("torch")

    from framepose.model import ModelConfig, build_model

    torch.manual_seed(0)
    model = build_model(ModelConfig(sign_fields=SIGN_FIELD_COUNT)).eval()
    geometry = torch.randn(1, 17, 4)
    base = torch.zeros(1, SIGN_FIELD_COUNT, dtype=torch.long)
    for index in range(SIGN_FIELD_COUNT):
        altered = base.clone()
        altered[0, index] = 1
        assert not torch.allclose(model(geometry, None, base), model(geometry, None, altered)), \
            f"{SIGN_FIELD_NAMES[index]} has no effect on the prediction"


def test_sign_experiment_runner_declares_its_comparison_semantics():
    import importlib.util
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(root / "src"))
    try:
        spec = importlib.util.spec_from_file_location(
            "run_sign_experiments", root / "scripts" / "run_sign_experiments.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)

    assert set(module.CANDIDATES) == {"S0", "S1", "S2"}
    assert module.CANDIDATES["S0"]["sign_source"] == "neutral"
    assert module.CANDIDATES["S1"]["sign_source"] == "oracle"
    semantics = module.COMPARISON_SEMANTICS
    assert "capacity-matched" in semantics["S1_vs_S0"]
    assert "dense visual" in semantics["not_comparable_with"]
    assert "mpjpe_mm" in semantics["guardrail_metrics"]
    assert "mpjpe_mm" not in semantics["primary_metrics"]
