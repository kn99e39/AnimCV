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
    """An invalid joint degrades only the fields that depend on it."""
    pose, valid = _upright()
    # Bend the left knee so its field is genuinely observable to begin with.
    pose[JOINT_INDEX["left_knee"]] = pose[JOINT_INDEX["left_knee"]] + np.asarray([0, 0.15, 0])
    assert sign_state(pose, valid)[SIGN_FIELD_NAMES.index("left_knee_forward_bend")] == POSITIVE

    valid[JOINT_INDEX["right_shoulder"]] = False
    state = sign_state(pose, valid)
    assert state[SIGN_FIELD_NAMES.index("shoulder_forward_depth")] == UNKNOWN
    assert state[SIGN_FIELD_NAMES.index("torso_facing")] == UNKNOWN
    # ... and leaves the fields that do not depend on it intact.
    assert state[SIGN_FIELD_NAMES.index("left_knee_forward_bend")] == POSITIVE
    assert state[SIGN_FIELD_NAMES.index("hip_forward_depth")] != UNKNOWN or \
        sign_state(pose, np.ones(17, dtype=bool))[SIGN_FIELD_NAMES.index("hip_forward_depth")] == UNKNOWN


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

    # The contract module must not reach for any dense visual machinery.
    source = inspect.getsource(signs_module)
    for forbidden in ("timm", "patch_token", "visual_dim", "image_tokens",
                      "FrozenVisualBackbone", "load_feature_cache"):
        assert forbidden not in source, f"framepose.signs must not reference {forbidden}"
    # And the contract must declare its exclusions rather than merely omit them.
    excluded = contract()["excluded_by_contract"]
    for forbidden in ("XYZ coordinates", "depth magnitude", "metric offsets", "bone lengths",
                      "continuous pose embedding", "image patch tokens"):
        assert any(forbidden in item for item in excluded), forbidden

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

    assert {"S0", "S1", "S2", "O_TORSO", "O_BILATERAL", "O_HINGE", "O_ORIENTATION",
            "O_SHOULDER", "O_HIP", "O_ELBOWS", "O_KNEES", "O_LEFT_ELBOW", "O_RIGHT_ELBOW",
            "O_LEFT_KNEE", "O_RIGHT_KNEE"} == set(module.CANDIDATES)
    # Single-field candidates activate exactly one field, so a group result can
    # never stand in for an individual necessity claim.
    for key in ("O_TORSO", "O_SHOULDER", "O_HIP", "O_LEFT_ELBOW", "O_RIGHT_ELBOW",
                "O_LEFT_KNEE", "O_RIGHT_KNEE"):
        assert len(module.CANDIDATES[key]["fields"]) == 1, key
    assert set(module.CANDIDATES["O_ELBOWS"]["fields"] + module.CANDIDATES["O_KNEES"]["fields"]) \
        == set(module.CANDIDATES["O_HINGE"]["fields"])
    assert set(module.CANDIDATES["O_SHOULDER"]["fields"] + module.CANDIDATES["O_HIP"]["fields"]) \
        == set(module.CANDIDATES["O_BILATERAL"]["fields"])
    for key, definition in module.CANDIDATES.items():
        assert set(definition["fields"]) <= set(SIGN_FIELD_NAMES), key
    assert module.CANDIDATES["S0"]["sign_source"] == "neutral"
    assert module.CANDIDATES["S0"]["fields"] == []
    assert module.CANDIDATES["S1"]["fields"] == list(SIGN_FIELD_NAMES)
    # The attribution groups must partition the contract without overlap.
    assert module.CANDIDATES["O_TORSO"]["fields"] == ["torso_facing"]
    assert module.CANDIDATES["O_BILATERAL"]["fields"] == ["shoulder_forward_depth",
                                                          "hip_forward_depth"]
    assert len(module.CANDIDATES["O_HINGE"]["fields"]) == 4
    assert all(name.endswith("_forward_bend") for name in module.CANDIDATES["O_HINGE"]["fields"])
    assert set(module.CANDIDATES["O_TORSO"]["fields"] + module.CANDIDATES["O_BILATERAL"]["fields"]
               + module.CANDIDATES["O_HINGE"]["fields"]) == set(SIGN_FIELD_NAMES)
    assert set(module.CANDIDATES["O_ORIENTATION"]["fields"]) == set(
        module.CANDIDATES["O_TORSO"]["fields"] + module.CANDIDATES["O_BILATERAL"]["fields"])
    semantics = module.COMPARISON_SEMANTICS
    assert "capacity-matched" in semantics["S1_vs_S0"]
    tiers = semantics["evidence_tiers"]
    assert "root_yaw_error_degrees" in tiers["structurally_coupled_downstream"]
    assert "mpjpe_mm" in tiers["independent_position_guardrails"]
    assert any("shoulder" in item for item in tiers["direct_contract_identical"])
    assert "dense visual" in semantics["not_comparable_with"]
    assert "mpjpe_mm" in semantics["guardrail_metrics"]
    assert "mpjpe_mm" not in semantics["primary_metrics"]


# ---------------------------------------------------- value-domain safety ----

def test_out_of_domain_sign_values_are_refused_not_clamped():
    """Clamping would turn a broken advisor into a confident wrong branch."""
    from framepose.signs import ALLOWED_VALUES, validate_sign_array

    assert ALLOWED_VALUES == (NEGATIVE, UNKNOWN, POSITIVE)
    good = np.asarray([[1, -1, 0, 1, 0, -1, 1]], dtype=np.int8)
    assert np.array_equal(validate_sign_array(good), good)

    for illegal in (2, -2, 7, 255):
        broken = good.copy().astype(np.int64)
        broken[0, 0] = illegal
        with pytest.raises(ValueError, match="outside the Sign Contract domain"):
            validate_sign_array(broken)


def test_sign_array_validation_checks_shape_and_dtype():
    from framepose.signs import validate_sign_array

    with pytest.raises(ValueError, match=r"shape \(n, 7\)"):
        validate_sign_array(np.zeros((3, 4), dtype=np.int8))
    with pytest.raises(ValueError, match="must have 5 rows"):
        validate_sign_array(np.zeros((3, SIGN_FIELD_COUNT), dtype=np.int8), expected_rows=5)
    with pytest.raises(ValueError, match="non-integral"):
        validate_sign_array(np.full((1, SIGN_FIELD_COUNT), 0.5))
    with pytest.raises(ValueError, match="must hold integers"):
        validate_sign_array(np.full((1, SIGN_FIELD_COUNT), "x"))
    # An integral float array is acceptable and comes back as int8.
    assert validate_sign_array(np.ones((2, SIGN_FIELD_COUNT), dtype=float)).dtype == np.int8


def test_the_model_refuses_an_out_of_domain_sign_state():
    torch = pytest.importorskip("torch")

    from framepose.model import ModelConfig, build_model

    torch.manual_seed(0)
    model = build_model(ModelConfig(sign_fields=SIGN_FIELD_COUNT)).eval()
    geometry = torch.randn(1, 17, 4)
    state = torch.zeros(1, SIGN_FIELD_COUNT, dtype=torch.long)
    state[0, 3] = 2
    with pytest.raises(ValueError, match="never clamped"):
        model(geometry, None, state)


def test_an_advisor_bank_with_illegal_values_is_refused(bank):
    from framepose.train import sign_tensor

    broken = np.zeros((len(bank), SIGN_FIELD_COUNT), dtype=np.int64)
    broken[0, 0] = 3
    with pytest.raises(ValueError, match="outside the Sign Contract domain"):
        sign_tensor(bank, "advisor", broken)


# ------------------------------------------------------- field attribution ----

def test_mask_fields_keeps_only_the_named_group(bank):
    from framepose.signs import mask_fields

    oracle = oracle_sign_states(bank.arrays["target_3d"], bank.arrays["target_valid"])
    masked = mask_fields(oracle, ["torso_facing"])
    index = SIGN_FIELD_NAMES.index("torso_facing")
    assert np.array_equal(masked[:, index], oracle[:, index])
    others = [i for i in range(SIGN_FIELD_COUNT) if i != index]
    assert not masked[:, others].any(), "every non-active field must be UNKNOWN"
    with pytest.raises(ValueError, match="unknown sign fields"):
        mask_fields(oracle, ["torso_facing", "nose_direction"])


def test_every_attribution_candidate_shares_one_parameter_count(bank):
    """Only which fields carry information may differ, never capacity."""
    torch = pytest.importorskip("torch")

    from framepose.signs import mask_fields
    from framepose.train import CandidateConfig, train_candidate

    oracle = oracle_sign_states(bank.arrays["target_3d"], bank.arrays["target_valid"])
    groups = {"torso": ["torso_facing"],
              "bilateral": ["shoulder_forward_depth", "hip_forward_depth"],
              "hinge": [name for name in SIGN_FIELD_NAMES if name.endswith("_forward_bend")]}
    counts = set()
    for name, fields in groups.items():
        report = train_candidate(
            bank,
            CandidateConfig(name=f"unit_{name}", sign_source="oracle", epochs=1, batch_size=16,
                            device="cpu", mixed_precision=False, evaluate_every=1, seed=5),
            signs=mask_fields(oracle, fields))
        counts.add(report["model"]["trainable_parameter_count"])
    assert len(counts) == 1, "attribution candidates must be capacity-matched"


# ------------------------------------------------------- advisor contract ----

def test_advisor_response_schema_is_strict():
    from framepose.sign_advisor import parse_response

    body = "{" + ", ".join(f'"{name}": "unclear"' for name in SIGN_FIELD_NAMES) + "}"
    assert parse_response(body).valid is True

    cases = {
        "prose before the object": "Sure! " + body,
        "prose after the object": body + " Hope that helps.",
        "two objects": body + " " + body,
        "extra key": body[:-1] + ', "confidence": "high"}',
        "missing key": "{" + ", ".join(f'"{n}": "unclear"' for n in SIGN_FIELD_NAMES[:-1]) + "}",
        "non-string answer": "{" + ", ".join(
            f'"{n}": ' + ("1" if n == "torso_facing" else '"unclear"') for n in SIGN_FIELD_NAMES) + "}",
        "unknown categorical": body.replace('"unclear"', '"probably_facing_away"', 1),
        "empty": "",
        "not an object": "[1, 2, 3]",
    }
    for label, text in cases.items():
        response = parse_response(text)
        assert response.valid is False, f"{label} must be rejected"
        assert response.reason, f"{label} must record why"
        assert not response.state.any(), f"{label} must fall back to all-UNKNOWN, never a guess"


def test_isolated_prompt_is_verbatim_and_asks_exactly_one_question():
    from framepose.sign_advisor import (
        ISOLATED_PROMPT_SCHEMA_VERSION, _QUESTIONS, isolated_prompt_provenance,
        isolated_prompt_text, parse_response, prompt_text,
    )

    combined = prompt_text()
    for name, question, mapping in _QUESTIONS:
        isolated = isolated_prompt_text(name)
        # The question sentence and its options are copied, not paraphrased.
        assert question in isolated and question in combined
        for option in mapping:
            assert option in isolated
        assert isolated.count("allowed answers:") == 1
        # Nothing was added. Every line either appears verbatim in the historical
        # prompt, or is one of exactly three permitted differences: the
        # singular/plural of the instruction sentence, the single question line,
        # and the single-key JSON template.
        combined_lines = set(combined.splitlines())
        singular = {"Answer these questions.": "Answer this question.",
                    "using exactly these keys:": "using exactly this key:"}
        for line in isolated.splitlines():
            reused = line in combined_lines or any(
                line == other.replace(plural, single)
                for other in combined_lines for plural, single in singular.items())
            is_question = line.strip().startswith(f"{name}:") or line.strip().startswith("1.")
            is_template = line.startswith("{")
            assert reused or is_question or is_template, f"isolated prompt added: {line!r}"
        provenance = isolated_prompt_provenance(name)
        assert provenance["schema_version"] == ISOLATED_PROMPT_SCHEMA_VERSION
        assert provenance["verbatim_question"] == question
        parsed = parse_response("{" + f'"{name}": "{list(mapping)[0]}"' + "}", fields=(name,))
        assert parsed.valid is True
        assert parsed.state[SIGN_FIELD_NAMES.index(name)] == mapping[list(mapping)[0]]

    with pytest.raises(ValueError, match="unknown sign field"):
        isolated_prompt_text("nose_direction")


def test_an_explicitly_supplied_sign_array_is_never_re_derived(bank):
    """Regression: field masks must survive into training.

    The first attribution run re-derived the full oracle inside the trainer and
    discarded the mask, so every candidate trained on all seven fields and was
    then evaluated on one group — a train/evaluate mismatch that made correct
    information look harmful.
    """
    pytest.importorskip("torch")

    from framepose.signs import mask_fields
    from framepose.train import CandidateConfig, train_candidate

    oracle = oracle_sign_states(bank.arrays["target_3d"], bank.arrays["target_valid"])
    masked = mask_fields(oracle, ["torso_facing"])
    report = train_candidate(
        bank,
        CandidateConfig(name="unit_masked", sign_source="oracle", epochs=1, batch_size=16,
                        device="cpu", mixed_precision=False, evaluate_every=1, seed=3),
        signs=masked)

    distribution = report["sign"]["distribution"]
    active = distribution["torso_facing"]
    assert active["positive"] + active["negative"] > 0, "the active field must carry information"
    for name in SIGN_FIELD_NAMES:
        if name == "torso_facing":
            continue
        assert distribution[name]["positive"] == 0 and distribution[name]["negative"] == 0, \
            f"{name} must be fully UNKNOWN in a torso-only candidate"
        assert distribution[name]["degenerate"] == len(bank)

    # And an unmasked oracle run really is different, so the check has teeth.
    full = train_candidate(
        bank,
        CandidateConfig(name="unit_full", sign_source="oracle", epochs=1, batch_size=16,
                        device="cpu", mixed_precision=False, evaluate_every=1, seed=3))
    assert full["sign"]["distribution"]["shoulder_forward_depth"]["degenerate"] < len(bank)


def test_the_model_requires_exact_sign_membership_not_a_range():
    """A range test lets 0.5 through and `.long()` then truncates it to 0.

    NaN and inf also pass an unordered range comparison, so the boundary must
    check exact membership in {-1, 0, +1}.
    """
    torch = pytest.importorskip("torch")

    from framepose.model import ModelConfig, build_model

    torch.manual_seed(0)
    model = build_model(ModelConfig(sign_fields=SIGN_FIELD_COUNT)).eval()
    geometry = torch.randn(1, 17, 4)

    for legal in (-1.0, 0.0, 1.0):
        state = torch.full((1, SIGN_FIELD_COUNT), legal)
        assert model(geometry, None, state).shape == (1, 17, 3)

    for illegal in (0.5, -0.5, 0.999, float("nan"), float("inf"), float("-inf")):
        state = torch.zeros(1, SIGN_FIELD_COUNT)
        state[0, 2] = illegal
        with pytest.raises(ValueError, match="exactly -1, 0 or"):
            model(geometry, None, state)


def test_per_chain_hinge_metrics_are_reported(bank):
    from framepose.evaluate import evaluate_predictions

    positions = bank.indices("test")
    prediction = bank.arrays["target_3d"][positions].copy()
    # Flip only the left elbow's bend across the shoulder-wrist axis.
    left_shoulder, left_elbow, left_wrist = (JOINT_INDEX["left_shoulder"],
                                             JOINT_INDEX["left_elbow"], JOINT_INDEX["left_wrist"])
    axis_point = (prediction[:, left_shoulder] + prediction[:, left_wrist]) / 2
    prediction[:, left_elbow] = 2 * axis_point - prediction[:, left_elbow]
    report = evaluate_predictions(bank, positions, prediction, candidate="left_elbow_flipped")

    aggregate = report["aggregate"]
    assert aggregate["left_elbow_bend_flipped"]["mean"] > 0.5
    assert aggregate["right_elbow_bend_flipped"]["mean"] == pytest.approx(0.0, abs=1e-9)
    assert aggregate["knee_flip_rate"]["mean"] == pytest.approx(0.0, abs=1e-9)
    assert aggregate["elbow_flip_rate"]["mean"] > aggregate["knee_flip_rate"]["mean"]
    frame = report["frames"][0]
    assert set(frame) >= {"left_elbow_bend_error_degrees", "right_elbow_bend_flipped",
                          "elbow_flip_rate", "knee_bend_error_degrees"}
