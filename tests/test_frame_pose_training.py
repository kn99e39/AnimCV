import numpy as np
import pytest

torch = pytest.importorskip("torch")

from framepose.bank import BankRequest, build_bank
from framepose.contract import JOINT_NAMES
from framepose.losses import (
    BASELINE_GEOMETRY_V1, COORDINATE_ONLY_V1, LOSS_CONTRACTS, LossContract, compute_loss, resolve_contract,
)
from framepose.model import ModelConfig, build_model
from framepose.screening import fixed_batches, screen_contracts
from framepose.train import CandidateConfig, geometry_tensor, load_checkpoint, predict, train_candidate
from framepose_fixtures import prepared_dataset
from training.temporal_lifter import TrainingConfig, _supervision_loss


SEQUENCES = {"train": ["3dpw:a:actor0", "3dpw:b:actor0"],
             "validation": ["3dpw:v:actor0"], "test": ["3dpw:t:actor0"]}


@pytest.fixture()
def bank(tmp_path):
    requests = [BankRequest("3DPW", split, prepared_dataset(tmp_path / f"{split}.json",
                                                            split=split, sequences=names))
                for split, names in SEQUENCES.items()]
    built, _ = build_bank(requests, require_rgb=False)
    return built


def _config(**overrides):
    base = dict(name="unit", epochs=2, batch_size=16, device="cpu", mixed_precision=False,
                evaluate_every=1, seed=17)
    base.update(overrides)
    return CandidateConfig(**base)


def test_shared_loss_contract_is_the_legacy_stable_geometry_objective():
    torch.manual_seed(0)
    prediction, target = torch.randn(6, 17, 3), torch.randn(6, 17, 3)
    mask = (torch.rand(6, 17, 1) > 0.25).float()
    legacy = _supervision_loss(torch, prediction, target, mask, TrainingConfig(
        bone_loss_weight=0.25, torso_loss_weight=0.15, hinge_loss_weight=0.15))
    assert compute_loss(torch, prediction, target, mask, BASELINE_GEOMETRY_V1) == pytest.approx(float(legacy))
    assert BASELINE_GEOMETRY_V1.name in LOSS_CONTRACTS
    with pytest.raises(ValueError):
        resolve_contract("yaw_tail_v2")
    with pytest.raises(ValueError):
        LossContract(name="negative", bone_weight=-1.0)


def test_geometry_tensor_is_identical_for_every_candidate(bank):
    first = geometry_tensor(bank)
    second = geometry_tensor(bank)
    assert np.array_equal(first, second)
    assert first.shape == (len(bank), len(JOINT_NAMES), 4)
    assert set(np.unique(first[:, :, 3])) <= {0.0, 1.0}


def test_geometry_only_candidate_trains_and_selects_on_validation(bank, tmp_path):
    report = train_candidate(bank, _config(), checkpoint_path=tmp_path / "f0.pt")
    assert report["model"]["trainable_parameter_count"] > 0
    assert report["selection"]["split"] == "validation"
    assert report["selection"]["test_ground_truth_used"] is False
    assert report["selection"]["validation_mpjpe_mm"] is not None
    assert report["performance"]["frames_per_second"] > 0
    assert report["backbone"]["kind"] == "none"
    assert len(report["epoch_telemetry"]) == 2


def test_visual_candidates_require_and_consume_cached_features(bank, tmp_path):
    features = np.zeros((len(bank), 196, 768), dtype=np.float32)
    with pytest.raises(ValueError, match="requires cached"):
        train_candidate(bank, _config(backbone="vit_in21k"))
    with pytest.raises(ValueError, match="must not be given visual features"):
        train_candidate(bank, _config(), features=features)
    report = train_candidate(bank, _config(backbone="siglip"), features=features,
                             checkpoint_path=tmp_path / "f2.pt")
    assert report["backbone"]["kind"] == "vision_language"
    assert report["model"]["visual_tokens"] == 196


def test_candidate_replay_is_deterministic(bank):
    first = train_candidate(bank, _config())
    second = train_candidate(bank, _config())
    assert first["epoch_telemetry"][-1]["train_loss"] == pytest.approx(
        second["epoch_telemetry"][-1]["train_loss"])
    assert first["selection"]["validation_mpjpe_mm"] == pytest.approx(
        second["selection"]["validation_mpjpe_mm"])


def test_candidate_configuration_is_isolated(bank):
    """Changing only the loss contract must change only the objective."""
    baseline = train_candidate(bank, _config(loss_contract="baseline_geometry_v1"))
    coordinate = train_candidate(bank, _config(loss_contract="coordinate_only_v1"))
    assert baseline["candidate"]["seed"] == coordinate["candidate"]["seed"]
    assert baseline["bank"] == coordinate["bank"]
    assert baseline["loss_contract"]["name"] != coordinate["loss_contract"]["name"]
    assert baseline["epoch_telemetry"][0]["train_loss"] != coordinate["epoch_telemetry"][0]["train_loss"]
    assert baseline["augmentation"]["enabled"] is False


def test_parameter_efficient_backbone_adaptation_is_gated_off():
    with pytest.raises(ValueError, match="gated on frozen-F2 evidence"):
        CandidateConfig(name="adapted", backbone="siglip", adapt_backbone=True)


def test_checkpoint_round_trip_reproduces_predictions(bank, tmp_path):
    path = tmp_path / "candidate.pt"
    train_candidate(bank, _config(), checkpoint_path=path)
    model, payload = load_checkpoint(path)
    assert payload["bank_content_digest"] == bank.content_digest()
    assert payload["loss_contract"]["name"] == "baseline_geometry_v1"
    positions = bank.indices("test")
    geometry = geometry_tensor(bank)
    first = predict(model, torch, geometry, None, positions, torch.device("cpu"))
    second = predict(model, torch, geometry, None, positions, torch.device("cpu"))
    assert first.shape == (len(positions), 17, 3)
    assert np.array_equal(first, second)


def test_compiled_execution_path_is_recorded_and_equivalent(bank):
    eager = train_candidate(bank, _config(epochs=1))
    assert eager["execution"]["execution_backend"] == "eager"
    try:
        compiled = train_candidate(bank, _config(epochs=1, compile_training_graph=True))
    except Exception as error:  # pragma: no cover - torch.compile unavailable on this platform
        pytest.skip(f"torch.compile unavailable: {error}")
    assert compiled["execution"]["execution_backend"] == "compiled"
    assert compiled["epoch_telemetry"][-1]["train_loss"] == pytest.approx(
        eager["epoch_telemetry"][-1]["train_loss"], rel=1e-3)


def test_loss_screening_measures_without_encoding_a_threshold(bank):
    geometry = geometry_tensor(bank)
    report = screen_contracts(bank, [BASELINE_GEOMETRY_V1, COORDINATE_ONLY_V1], geometry=geometry,
                              batch_count=3, batch_size=8, device="cpu")
    assert report["acceptance_thresholds"] is None
    state = report["states"]["initialization"]
    for name in ("baseline_geometry_v1", "coordinate_only_v1"):
        measured = state["contracts"][name]
        assert measured["numerically_stable"] is True
        assert measured["gradient_norm"]["mean"] > 0
        assert measured["gradient_over_base_ratio"]["mean"] > 0
        assert -1.0 <= measured["gradient_cosine_with_base"]["mean"] <= 1.0
    # The base contract screened against itself is by definition perfectly aligned.
    assert state["contracts"]["baseline_geometry_v1"]["gradient_cosine_with_base"]["mean"] == pytest.approx(1.0, abs=1e-4)
    ownership = state["components"]["coordinate"]["per_joint_gradient_ownership"]
    assert set(ownership) == set(JOINT_NAMES)
    assert sum(ownership.values()) == pytest.approx(1.0, abs=1e-5)
    assert state["frame_association"]["source_contribution_mean"]["3DPW"] == pytest.approx(1.0)


def test_screening_batches_are_fixed_across_contracts(bank):
    first = fixed_batches(bank, batch_count=4, batch_size=8, seed=99)
    second = fixed_batches(bank, batch_count=4, batch_size=8, seed=99)
    assert all(np.array_equal(a, b) for a, b in zip(first, second))
