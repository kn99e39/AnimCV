import json

import numpy as np
import pytest

from common.serialization import read_json, write_json
from framepose.bank import BankRequest, build_bank
from framepose.contract import BANK_SCHEMA, FrameBank
from framepose.observations import (
    BACKEND_DATASET_DETECTOR, BACKEND_DATASET_GROUND_TRUTH, BACKEND_MMPOSE,
    BACKEND_SYNTHETIC_PROJECTION, DATASET_OBSERVATIONS, INTERPRETABLE_REGIMES,
    REGIME_BENCHMARK_DETECTOR, REGIME_HISTORICAL_UNKNOWN, REGIME_ORACLE, REGIME_REAL_ANIMCV,
    ObservationProvenance, UNRECORDED, assert_quality_interpretable, assert_single_regime,
    migrate_regime, mmpose_observation, observation_cache_key, resolve_dataset_observation,
)
from framepose.sources import SOURCE_SPECS, frames_from_prepared_dataset, load_prepared_dataset, resolve_spec
from framepose_fixtures import prepared_dataset, write_images


TRAIN = ["3dpw:a:actor0"]
VALIDATION = ["3dpw:v:actor0"]
TEST = ["3dpw:t:actor0"]


def _mmpose(**overrides):
    base = dict(pose_config="rtmpose-t.py", pose_checkpoint="rtmpose-t.pth",
                detector_config="rtmdet-t.py", detector_checkpoint="rtmdet-t.pth",
                visibility_threshold=0.3, input_size="256x192", mmpose_version="1.3.2",
                pose_weights_sha256="aaa", detector_weights_sha256="bbb")
    base.update(overrides)
    return mmpose_observation(**base)


def _bank(tmp_path, mixed=False):
    images = write_images(tmp_path / "imageFiles", TRAIN + VALIDATION + TEST)
    requests = [BankRequest("3DPW", split, prepared_dataset(tmp_path / f"{split}.json",
                                                            split=split, sequences=names))
                for split, names in (("train", TRAIN), ("validation", VALIDATION), ("test", TEST))]
    return build_bank(requests, image_roots={"3dpw_images": images}, require_rgb=True)


def test_dataset_observations_are_read_from_the_prepared_artifact(tmp_path):
    path = prepared_dataset(tmp_path / "train.json", split="train", sequences=TRAIN)
    payload = load_prepared_dataset(path)
    # The fixture writes no input_kind, so the SourceSpec default applies.
    samples, _ = frames_from_prepared_dataset(payload, spec=resolve_spec("3DPW"), split="train")
    assert samples[0].observation.backend == BACKEND_DATASET_DETECTOR
    # 3DPW ships detector output, not ground truth: it must not be oracle.
    assert samples[0].observation.regime == REGIME_BENCHMARK_DETECTOR

    # An artifact that declares its own input_kind wins over the table default.
    payload["sequences"][0]["source"] = {"dataset": "3DPW", "input_kind": "dataset_ground_truth_2d"}
    samples, _ = frames_from_prepared_dataset(payload, spec=resolve_spec("3DPW"), split="train")
    assert samples[0].observation.observation_type == "projected_ground_truth_2d"


def test_every_declared_source_maps_to_a_registered_observation():
    expected = {"3DPW": REGIME_BENCHMARK_DETECTOR, "MPI-INF-3DHP": REGIME_ORACLE,
                "AMASS": REGIME_ORACLE}
    for name, spec in SOURCE_SPECS.items():
        assert spec.default_input_kind in DATASET_OBSERVATIONS, name
        provenance = resolve_dataset_observation(spec.default_input_kind)
        assert provenance.regime == expected[name], name
        # Dataset-provided geometry must never be labelled as AnimCV's own sensor.
        assert provenance.backend != BACKEND_MMPOSE


def test_a_detector_is_never_oracle_and_ground_truth_always_is():
    detector = resolve_dataset_observation("official_3dpw_2d_detection")
    assert detector.backend == BACKEND_DATASET_DETECTOR
    assert detector.regime == REGIME_BENCHMARK_DETECTOR
    assert detector.regime != REGIME_ORACLE
    for kind in ("dataset_ground_truth_2d", "synthetic_virtual_camera_gt_2d"):
        assert resolve_dataset_observation(kind).regime == REGIME_ORACLE
    assert resolve_dataset_observation("dataset_ground_truth_2d").backend == BACKEND_DATASET_GROUND_TRUTH
    assert resolve_dataset_observation("synthetic_virtual_camera_gt_2d").backend == BACKEND_SYNTHETIC_PROJECTION
    # The invariant is enforced, not merely conventional.
    with pytest.raises(ValueError, match="never oracle geometry"):
        ObservationProvenance(BACKEND_DATASET_DETECTOR, "dataset_shipped_detector_2d", REGIME_ORACLE)


def test_regime_migration_resolves_from_the_backend_and_refuses_to_guess():
    # A v2 artifact that mislabelled 3DPW detector output as oracle resolves
    # deterministically to the benchmark-detector regime.
    assert migrate_regime(BACKEND_DATASET_DETECTOR, "oracle_geometry") == REGIME_BENCHMARK_DETECTOR
    # Ground truth and synthetic projection stay oracle -- not every old
    # oracle_geometry label maps to the same new regime.
    assert migrate_regime(BACKEND_DATASET_GROUND_TRUTH, "oracle_geometry") == REGIME_ORACLE
    assert migrate_regime(BACKEND_SYNTHETIC_PROJECTION, "oracle_geometry") == REGIME_ORACLE
    assert migrate_regime(BACKEND_MMPOSE, "real_observation") == REGIME_REAL_ANIMCV
    # An unrecognised backend and label is not guessed at.
    assert migrate_regime("some_future_sensor", "who_knows") == REGIME_HISTORICAL_UNKNOWN

    loaded = ObservationProvenance.from_dict(
        {"backend": BACKEND_DATASET_DETECTOR, "observation_type": "dataset_shipped_detector_2d",
         "regime": "oracle_geometry", "detail": {}})
    assert loaded.regime == REGIME_BENCHMARK_DETECTOR
    assert loaded.detail["migrated_from_regime"] == "oracle_geometry"


def test_quality_interpretation_is_refused_for_unresolvable_artifacts():
    for regime in INTERPRETABLE_REGIMES:
        assert assert_quality_interpretable(regime) == regime
    with pytest.raises(ValueError, match="no resolvable 2D observation provenance"):
        assert_quality_interpretable(REGIME_HISTORICAL_UNKNOWN)


def test_unknown_input_kind_is_refused_rather_than_defaulted():
    with pytest.raises(ValueError, match="register it in"):
        resolve_dataset_observation("some_new_detector_2d")


def test_mmpose_observation_is_the_real_observation_regime():
    provenance = _mmpose()
    assert provenance.backend == BACKEND_MMPOSE
    assert provenance.regime == REGIME_REAL_ANIMCV
    assert provenance.detail["adapter"] == "pose.mmpose_adapter.PoseEstimator"
    # A dataset backend may not claim the real-observation regime and vice versa.
    with pytest.raises(ValueError):
        ObservationProvenance(BACKEND_MMPOSE, "estimated_2d", REGIME_ORACLE)
    with pytest.raises(ValueError):
        ObservationProvenance(BACKEND_DATASET_DETECTOR, "dataset_shipped_detector_2d", REGIME_REAL_ANIMCV)


@pytest.mark.parametrize("field,value", [
    ("pose_config", "rtmpose-l.py"),
    ("pose_checkpoint", "other.pth"),
    ("pose_weights_sha256", "ccc"),
    ("detector_config", "rtmdet-l.py"),
    ("detector_checkpoint", "other-det.pth"),
    ("visibility_threshold", 0.5),
    ("input_size", "384x288"),
])
def test_cached_observation_is_invalidated_by_model_weights_config_or_preprocessing(field, value):
    baseline = _mmpose().cache_key()
    assert _mmpose(**{field: value}).cache_key() != baseline, field
    assert _mmpose().cache_key() == baseline, "the key must be stable for an unchanged sensor"


def test_cached_observation_is_invalidated_by_the_input_image():
    provenance = _mmpose()
    first = observation_cache_key(provenance, "seq/image_00000.jpg")
    assert observation_cache_key(provenance, "seq/image_00001.jpg") != first
    assert observation_cache_key(provenance, "seq/image_00000.jpg") == first
    assert observation_cache_key(_mmpose(pose_checkpoint="x.pth"), "seq/image_00000.jpg") != first


def test_bank_is_labelled_with_exactly_one_regime(tmp_path):
    bank, report = _bank(tmp_path)
    assert report["regime"] == REGIME_BENCHMARK_DETECTOR
    assert bank.regime() == REGIME_BENCHMARK_DETECTOR
    assert report["observation"]["backends"] == {BACKEND_DATASET_DETECTOR: len(bank)}
    assert report["intake"][0]["observation"]["regime"] == REGIME_BENCHMARK_DETECTOR


def test_mixed_regime_frames_are_refused_as_one_measurement(tmp_path):
    bank, _ = _bank(tmp_path)
    provenances = [sample.observation for sample in bank.samples]
    assert assert_single_regime(provenances) == REGIME_BENCHMARK_DETECTOR
    object.__setattr__(bank.samples[0], "observation", _mmpose())
    with pytest.raises(ValueError, match="mixes evaluation regimes"):
        bank.regime()


def test_observation_provenance_does_not_disturb_the_content_digest(tmp_path):
    """Adding provenance to a bank must not invalidate caches keyed to its frames."""
    bank, _ = _bank(tmp_path)
    baseline = bank.content_digest()
    object.__setattr__(bank.samples[0], "observation", _mmpose())
    assert bank.content_digest() == baseline
    # A genuinely different observation changes the 2D array, and that does move it.
    bank.arrays["input_2d"][0, 0, 0] += 0.01
    assert bank.content_digest() != baseline


def test_a_pre_provenance_bank_still_loads_and_is_marked_unlabelled(tmp_path):
    bank, _ = _bank(tmp_path)
    index_path, _ = bank.save(tmp_path / "bank.json")
    payload = read_json(index_path)
    assert payload["schema"] == BANK_SCHEMA
    payload["schema"] = "animcv_frame_pose_bank_v1"
    for sample in payload["samples"]:
        sample.pop("observation")
    write_json(index_path, payload)

    legacy = FrameBank.load(index_path)
    assert legacy.samples[0].observation == UNRECORDED
    assert legacy.regime() == REGIME_HISTORICAL_UNKNOWN
    assert legacy.content_digest() == bank.content_digest()


def test_evaluation_reports_carry_the_regime_label(tmp_path):
    from framepose.evaluate import evaluate_predictions

    bank, _ = _bank(tmp_path)
    positions = bank.indices("test")
    report = evaluate_predictions(bank, positions, bank.arrays["target_3d"][positions], candidate="oracle")
    assert report["observation_regime"] == [REGIME_BENCHMARK_DETECTOR]
    assert report["observation"]["backends"] == {BACKEND_DATASET_DETECTOR: len(positions)}
    assert report["frames"][0]["observation_backend"] == BACKEND_DATASET_DETECTOR


def test_content_digest_is_pinned_against_metadata_schema_drift(tmp_path):
    """The digest covers frames, not the index schema label.

    Bumping the bank schema to carry observation provenance once moved this
    digest, which silently invalidated every feature cache and experiment
    report keyed to it. The literal below pins the covered content so a future
    metadata change cannot repeat that.
    """
    from framepose.contract import CONTENT_DIGEST_DOMAIN

    bank, _ = _bank(tmp_path)
    assert CONTENT_DIGEST_DOMAIN == "animcv_frame_pose_bank_v1"
    assert bank.content_digest() == (
        "0d9c5213b8153260534ca7fadc4b9bf0182676c9eeb1493f1cdb042b623ab27f")


def test_provenance_fingerprint_and_content_digest_have_separate_jobs(tmp_path):
    bank, report = _bank(tmp_path)
    content = bank.content_digest()
    provenance = bank.provenance_fingerprint()
    assert report["provenance_fingerprint"] == provenance

    # A provenance change moves the provenance fingerprint and leaves the
    # content digest -- and therefore feature-cache validity -- alone.
    object.__setattr__(bank.samples[0], "observation", _mmpose())
    assert bank.content_digest() == content
    assert bank.provenance_fingerprint() != provenance

    # A change to the frames themselves moves the content digest.
    moved = bank.provenance_fingerprint()
    bank.arrays["input_2d"][0, 0, 0] += 0.01
    assert bank.content_digest() != content
    assert bank.provenance_fingerprint() == moved


def test_bank_fingerprint_reports_both_digests(tmp_path):
    bank, _ = _bank(tmp_path)
    index_path, _ = bank.save(tmp_path / "bank.json")
    fingerprint = bank.fingerprint(index_path)
    assert fingerprint["content_digest"] == bank.content_digest()
    assert fingerprint["provenance_fingerprint"] == bank.provenance_fingerprint()
    assert fingerprint["content_digest"] != fingerprint["provenance_fingerprint"]
