import math

import numpy as np
import pytest

from framepose.bank import BankRequest, build_bank
from framepose.contract import (
    BANK_SCHEMA, COORDINATE_FRAME, FORWARD_DEPTH_AXIS, FrameBank, ImageReference, JOINT_NAMES,
    Modality, make_sample_id, modality_summary,
)
from framepose.sources import SOURCE_SPECS, frames_from_prepared_dataset, load_prepared_dataset, resolve_spec
from framepose.strata import stratum_names
from framepose_fixtures import prepared_dataset, write_images


TRAIN = ["3dpw:seq_train_a:actor0", "3dpw:seq_train_b:actor0"]
VALIDATION = ["3dpw:seq_validation:actor0"]
TEST = ["3dpw:seq_test:actor0"]


def _bank(tmp_path, *, require_rgb=True, strides=(1, 1, 1)):
    images = write_images(tmp_path / "imageFiles", TRAIN + VALIDATION + TEST)
    requests = []
    for split, sequences, stride in (("train", TRAIN, strides[0]),
                                     ("validation", VALIDATION, strides[1]),
                                     ("test", TEST, strides[2])):
        path = prepared_dataset(tmp_path / f"{split}.json", split=split, sequences=sequences)
        requests.append(BankRequest("3DPW", split, path, stride=stride))
    return build_bank(requests, image_roots={"3dpw_images": images}, require_rgb=require_rgb)


def test_sample_identity_is_stable_and_sortable():
    assert make_sample_id("3dpw:courtyard_box_00:actor1", 42) == "3dpw:courtyard_box_00:actor1#000042"
    identifiers = [make_sample_id("s", index) for index in (2, 10, 1)]
    assert sorted(identifiers) == [identifiers[2], identifiers[0], identifiers[1]]
    with pytest.raises(ValueError):
        make_sample_id("bad#id", 0)


def test_sequence_and_frame_metadata_survive_the_frame_contract(tmp_path):
    path = prepared_dataset(tmp_path / "train.json", split="train", sequences=TRAIN)
    samples, arrays = frames_from_prepared_dataset(load_prepared_dataset(path),
                                                   spec=resolve_spec("3DPW"), split="train")
    assert len(samples) == 48
    first = samples[0]
    assert first.sequence_id == TRAIN[0]
    assert first.frame_index == 0
    assert first.fps == 30.0
    assert first.timestamp == pytest.approx(0.0)
    assert samples[1].timestamp == pytest.approx(1 / 30.0)
    # Frame-first must not discard sequence identity or neighbour addressability.
    assert first.neighbors["previous"] is None
    assert first.neighbors["next"] == samples[1].sample_id
    assert samples[23].neighbors["next"] is None, "neighbours must not cross a sequence boundary"
    assert samples[24].sequence_id == TRAIN[1]
    assert arrays["input_2d"].shape == (48, len(JOINT_NAMES), 3)


def test_stride_decimation_keeps_neighbours_inside_the_bank(tmp_path):
    path = prepared_dataset(tmp_path / "train.json", split="train", sequences=TRAIN[:1])
    samples, _ = frames_from_prepared_dataset(load_prepared_dataset(path), spec=resolve_spec("3DPW"),
                                              split="train", stride=3)
    identifiers = {sample.sample_id for sample in samples}
    assert [sample.frame_index for sample in samples] == list(range(0, 24, 3))
    for sample in samples:
        for neighbour in sample.neighbors.values():
            assert neighbour is None or neighbour in identifiers


def test_modality_availability_is_declared_per_source_and_never_fabricated():
    assert SOURCE_SPECS["3DPW"].modality == Modality(has_2d=True, has_3d=True, has_rgb=True, has_camera=True)
    assert SOURCE_SPECS["MPI-INF-3DHP"].modality.has_rgb is False
    assert SOURCE_SPECS["AMASS"].modality.has_rgb is False
    for name in ("MPI-INF-3DHP", "AMASS"):
        assert SOURCE_SPECS[name].image_reference is None, f"{name} must not be given an image path"


def test_geometry_only_source_is_excluded_from_the_paired_subset(tmp_path):
    images = write_images(tmp_path / "imageFiles", TRAIN)
    rgb = prepared_dataset(tmp_path / "rgb.json", split="train", sequences=TRAIN)
    geometry = prepared_dataset(tmp_path / "geometry.json", split="train",
                                sequences=["amass:walk:actor0"], dataset="AMASS")
    requests = [BankRequest("3DPW", "train", rgb), BankRequest("AMASS", "train", geometry)]
    paired, report = build_bank(requests, image_roots={"3dpw_images": images}, require_rgb=True)
    assert {sample.source for sample in paired.samples} == {"3DPW"}
    assert report["intake"][1]["retained_frames"] == 0
    # 3DPW ships detector output and AMASS is a synthetic projection, so pooling
    # them mixes observation regimes; that needs an explicit opt-in.
    with pytest.raises(ValueError, match="mixes evaluation regimes"):
        build_bank(requests, image_roots={"3dpw_images": images}, require_rgb=False)
    both, mixed_report = build_bank(requests, image_roots={"3dpw_images": images},
                                    require_rgb=False, allow_mixed_regime=True)
    assert mixed_report["regime"] == "mixed"
    assert set(mixed_report["observation"]["regimes"]) == {
        "benchmark_detector_observation", "oracle_geometry"}
    assert {sample.source for sample in both.samples} == {"3DPW", "AMASS"}
    summary = modality_summary(both.samples)
    assert summary["AMASS"]["has_rgb"] == 0
    assert summary["3DPW"]["has_rgb"] == summary["3DPW"]["sample_count"]


def test_image_reference_resolves_through_a_named_root_only(tmp_path):
    reference = ImageReference("3dpw_images", "courtyard_box_00/image_00007.jpg")
    assert reference.resolve({"3dpw_images": tmp_path}).name == "image_00007.jpg"
    with pytest.raises(KeyError):
        reference.resolve({})


def test_bank_round_trip_preserves_schema_and_fingerprint(tmp_path):
    bank, report = _bank(tmp_path)
    index_path, array_path = bank.save(tmp_path / "bank.json")
    assert array_path.is_file()
    reloaded = FrameBank.load(index_path)
    assert reloaded.content_digest() == bank.content_digest()
    assert reloaded.samples[0].strata["facing"] in {"frontal", "near_frontal", "profile", "back_facing"}
    fingerprint = reloaded.fingerprint(index_path)
    assert fingerprint["content_digest"] == report["content_digest"]
    assert fingerprint["sample_count"] == len(bank)
    assert set(fingerprint) >= {"index_sha256", "array_sha256", "split_counts"}


def test_bank_construction_is_deterministic(tmp_path):
    first, _ = _bank(tmp_path / "a")
    second, _ = _bank(tmp_path / "b")
    assert first.content_digest() == second.content_digest()
    strided, _ = _bank(tmp_path / "c", strides=(2, 1, 1))
    assert strided.content_digest() != first.content_digest()


def test_split_isolation_is_enforced(tmp_path):
    bank, _ = _bank(tmp_path)
    bank.assert_split_isolation()
    leaked = prepared_dataset(tmp_path / "leak.json", split="test", sequences=TRAIN[:1])
    images = tmp_path / "imageFiles"
    requests = [
        BankRequest("3DPW", "train", prepared_dataset(tmp_path / "t.json", split="train", sequences=TRAIN)),
        BankRequest("3DPW", "test", leaked),
    ]
    with pytest.raises(ValueError, match="both train and test"):
        build_bank(requests, image_roots={"3dpw_images": images}, require_rgb=True)


def test_strata_thresholds_are_fitted_on_train_only(tmp_path):
    bank, report = _bank(tmp_path)
    assert report["strata_threshold_fit_split"] == "train"
    assert set(report["strata_counts"]) == {"train", "validation", "test"}
    for name in stratum_names():
        assert name in report["strata_counts"]["test"]
    # Thresholds are a property of the bank, applied unchanged to every split.
    assert report["strata_thresholds"]["projected_torso_fraction"]


def test_canonical_coordinate_semantics_are_unchanged(tmp_path):
    bank, _ = _bank(tmp_path)
    assert bank.arrays["target_3d"].shape[1] == 17
    assert list(JOINT_NAMES)[0] == "pelvis"
    assert COORDINATE_FRAME == "camera_root_relative"
    assert FORWARD_DEPTH_AXIS == 1
    index_path, _ = bank.save(tmp_path / "bank.json")
    from common.serialization import read_json
    payload = read_json(index_path)
    assert payload["schema"] == BANK_SCHEMA
    assert payload["coordinate_frame"] == COORDINATE_FRAME
    assert payload["forward_depth_axis"] == 1
    # Targets are pelvis-relative: the root joint sits at the origin.
    assert np.abs(bank.arrays["target_3d"][:, 0, :]).max() < 1e-5


def test_facing_strata_follow_the_camera_convention(tmp_path):
    from framepose.strata import _facing_angle_degrees

    target = np.zeros((17, 3))
    valid = np.ones(17, dtype=bool)
    target[8] = (0.0, 0.0, 0.45)          # thorax above pelvis
    target[14], target[11] = (0.18, 0.0, 0.45), (-0.18, 0.0, 0.45)
    assert _facing_angle_degrees(target, valid) == pytest.approx(180.0, abs=1e-6)
    target[14], target[11] = (-0.18, 0.0, 0.45), (0.18, 0.0, 0.45)
    assert _facing_angle_degrees(target, valid) == pytest.approx(0.0, abs=1e-6)
    target[14], target[11] = (0.0, 0.18, 0.45), (0.0, -0.18, 0.45)
    assert abs(_facing_angle_degrees(target, valid)) == pytest.approx(90.0, abs=1e-6)
