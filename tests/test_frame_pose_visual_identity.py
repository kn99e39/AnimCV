"""Visual-input and feature-cache identity contracts.

A frozen visual feature is a pure function of the exact image bytes, the
geometry that built the crop, the crop contract and the backbone preprocessing.
These tests hold the executable semantics to that: a cache may not be reused
when any of those changed, and a historical cache may not be represented as
having recorded provenance it never recorded.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from common.serialization import read_json, write_json
from framepose.bank import BankRequest, build_bank
from framepose.crops import CROP_CONTRACT, CROP_RESOLUTION, crop_contract_digest
from framepose.features import (
    CACHE_SCHEMA, LEGACY_CACHE_SCHEMAS, WEIGHT_VERIFICATION, cache_directory, load_feature_cache,
    sample_order_digest,
)
from framepose.observations import image_content_digest, mmpose_observation, observation_cache_key
from framepose.visual_input import (
    image_content_digests, image_content_summary, preprocessing_identity, visual_input_fingerprint,
    visual_input_identity,
)
from framepose_fixtures import prepared_dataset, write_images


TRAIN = ["3dpw:a:actor0"]
VALIDATION = ["3dpw:v:actor0"]
TEST = ["3dpw:t:actor0"]
_PREPROCESSING = preprocessing_identity([0.5, 0.5, 0.5], [0.5, 0.5, 0.5], 224, 0)


@pytest.fixture()
def paired(tmp_path):
    images = write_images(tmp_path / "imageFiles", TRAIN + VALIDATION + TEST)
    requests = [BankRequest("3DPW", split, prepared_dataset(tmp_path / f"{split}.json",
                                                            split=split, sequences=names))
                for split, names in (("train", TRAIN), ("validation", VALIDATION), ("test", TEST))]
    bank, _ = build_bank(requests, image_roots={"3dpw_images": images}, require_rgb=True)
    index_path, _ = bank.save(tmp_path / "bank.json")
    return bank, index_path, Path(images)


def _identity(bank, images, **overrides):
    kwargs = {"image_roots": {"3dpw_images": images}, "preprocessing": _PREPROCESSING,
              "crop_resolution": CROP_RESOLUTION}
    kwargs.update(overrides)
    return visual_input_identity(bank, **kwargs)


def _write_cache(directory: Path, bank, *, schema=CACHE_SCHEMA, fingerprint=None,
                 crop_digest=None, weights="deadbeef", tokens=196, dim=768):
    directory.mkdir(parents=True, exist_ok=True)
    np.save(directory / "tokens.npy",
            np.zeros((len(bank), tokens, dim), dtype=np.float16))
    metadata = {
        "schema": schema,
        "backbone": {"key": "siglip", "timm_model": "vit_base_patch16_siglip_224.webli",
                     "weights_sha256": weights, "embed_dim": dim, "token_count": tokens,
                     "input_resolution": 224, "frozen": True},
        "crop_contract": CROP_CONTRACT,
        "bank_content_digest": bank.content_digest(),
        "sample_order_digest": sample_order_digest([s.sample_id for s in bank.samples]),
        "sample_count": len(bank), "dtype": "float16",
        "shape": [len(bank), tokens, dim], "array": "tokens.npy",
    }
    if schema == CACHE_SCHEMA:
        metadata.update({
            "crop_contract_digest": crop_digest or crop_contract_digest(),
            "visual_input_fingerprint": fingerprint,
            "feature_cache_provenance": "x" * 64,
            "weight_verification": WEIGHT_VERIFICATION,
            "provenance_level": "verified_v2",
            "token_shape": [tokens, dim],
        })
    write_json(directory / "meta.json", metadata)
    return metadata


# ------------------------------------------------ observation image identity --

def test_observation_cache_key_binds_image_bytes_not_the_path(tmp_path):
    provenance = mmpose_observation(pose_config="a.py", pose_checkpoint="b.pth",
                                    visibility_threshold=0.3)
    path = tmp_path / "image_00000.jpg"
    path.write_bytes(b"original pixels")
    first = observation_cache_key(provenance, image_content_digest(path))

    # Same path, different content: the key must move.
    path.write_bytes(b"replaced pixels")
    second = observation_cache_key(provenance, image_content_digest(path))
    assert second != first

    # Same content at a different path: the key must not move.
    other = tmp_path / "elsewhere" / "image_00000.jpg"
    other.parent.mkdir()
    other.write_bytes(b"replaced pixels")
    assert observation_cache_key(provenance, image_content_digest(other)) == second


def test_observation_cache_key_refuses_a_path(tmp_path):
    provenance = mmpose_observation(pose_config="a.py", pose_checkpoint="b.pth",
                                    visibility_threshold=0.3)
    for wrong in ("seq/image_00000.jpg", "image_00000.jpg", "abc123"):
        with pytest.raises(ValueError, match="SHA-256 of the image bytes"):
            observation_cache_key(provenance, wrong)
    assert observation_cache_key(provenance, None)


# ------------------------------------------------- visual input fingerprint --

def test_visual_input_fingerprint_is_deterministic_and_path_independent(paired, tmp_path):
    bank, _, images = paired
    first = _identity(bank, images)
    second = _identity(bank, images)
    assert first["fingerprint"] == second["fingerprint"]
    assert len(first["fingerprint"]) == 64

    # The same bytes reached through a different root produce the same identity.
    import shutil
    moved = tmp_path / "relocated"
    shutil.copytree(images, moved)
    assert _identity(bank, moved)["fingerprint"] == first["fingerprint"]


def test_image_content_change_moves_the_visual_fingerprint(paired):
    bank, _, images = paired
    baseline = _identity(bank, images)["fingerprint"]
    target = images / bank.samples[0].image_reference.relative_path
    payload = bytearray(target.read_bytes())
    payload[-1] ^= 0xFF
    target.write_bytes(bytes(payload))
    assert _identity(bank, images)["fingerprint"] != baseline


def test_crop_contract_change_moves_the_visual_fingerprint(paired):
    bank, _, images = paired
    identity = _identity(bank, images)
    altered = visual_input_fingerprint(
        image_content=identity["image_content_summary"],
        bank_content_digest=identity["bank_content_digest"],
        sample_order_digest=identity["sample_order_digest"],
        crop_digest="0" * 64,
        crop_resolution=identity["crop_resolution"],
        preprocessing=identity["preprocessing"])
    assert altered != identity["fingerprint"]
    resized = visual_input_fingerprint(
        image_content=identity["image_content_summary"],
        bank_content_digest=identity["bank_content_digest"],
        sample_order_digest=identity["sample_order_digest"],
        crop_digest=identity["crop_contract_digest"],
        crop_resolution=112,
        preprocessing=identity["preprocessing"])
    assert resized != identity["fingerprint"]


def test_preprocessing_change_moves_the_visual_fingerprint(paired):
    bank, _, images = paired
    baseline = _identity(bank, images)["fingerprint"]
    other = _identity(bank, images,
                      preprocessing=preprocessing_identity([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225], 224, 1))
    assert other["fingerprint"] != baseline


def test_bank_geometry_change_moves_the_visual_fingerprint(paired):
    bank, _, images = paired
    digests = image_content_digests(bank, {"3dpw_images": images})
    baseline = _identity(bank, images, digests=digests)["fingerprint"]
    bank.arrays["input_2d"][0, 0, 0] += 0.01
    assert _identity(bank, images, digests=digests)["fingerprint"] != baseline


def test_shared_images_are_hashed_once_and_summary_is_order_sensitive(paired):
    bank, _, images = paired
    digests = image_content_digests(bank, {"3dpw_images": images})
    assert len(digests) == len(bank)
    assert all(len(value) == 64 for value in digests)
    assert image_content_summary(digests) != image_content_summary(list(reversed(digests)))


# --------------------------------------------------- feature cache contract --

def test_v2_cache_loads_only_against_its_own_visual_input(paired, tmp_path):
    bank, _, images = paired
    identity = _identity(bank, images)
    directory = cache_directory(tmp_path / "features", "siglip")
    _write_cache(directory, bank, fingerprint=identity["fingerprint"])

    array, metadata = load_feature_cache(tmp_path / "features", "siglip", bank,
                                         visual_input_fingerprint=identity["fingerprint"])
    assert array.shape == (len(bank), 196, 768)
    assert metadata["visual_input_verified"] is True
    assert metadata["provenance_level"] == "verified_v2"

    with pytest.raises(ValueError, match="visual-input fingerprint does not match"):
        load_feature_cache(tmp_path / "features", "siglip", bank,
                           visual_input_fingerprint="f" * 64)


def test_v2_cache_is_refused_under_a_different_crop_contract(paired, tmp_path):
    bank, _, images = paired
    directory = cache_directory(tmp_path / "features", "siglip")
    _write_cache(directory, bank, fingerprint=_identity(bank, images)["fingerprint"],
                 crop_digest="a" * 64)
    with pytest.raises(ValueError, match="different crop contract"):
        load_feature_cache(tmp_path / "features", "siglip", bank)


def test_v2_cache_requires_a_recorded_backbone_and_weight_digest(paired, tmp_path):
    bank, _, images = paired
    directory = cache_directory(tmp_path / "features", "siglip")
    _write_cache(directory, bank, fingerprint=_identity(bank, images)["fingerprint"], weights="")
    with pytest.raises(ValueError, match="no backbone weight digest"):
        load_feature_cache(tmp_path / "features", "siglip", bank)

    metadata = read_json(directory / "meta.json")
    metadata["backbone"]["key"] = "vit_in21k"
    metadata["backbone"]["weights_sha256"] = "abc"
    write_json(directory / "meta.json", metadata)
    with pytest.raises(ValueError, match="different visual backbone"):
        load_feature_cache(tmp_path / "features", "siglip", bank)


def test_loading_without_an_identity_does_not_claim_image_verification(paired, tmp_path):
    """Weaker call, honestly labelled -- never a stronger claim than performed."""
    bank, _, images = paired
    directory = cache_directory(tmp_path / "features", "siglip")
    _write_cache(directory, bank, fingerprint=_identity(bank, images)["fingerprint"])
    _, metadata = load_feature_cache(tmp_path / "features", "siglip", bank)
    assert metadata["visual_input_verified"] is False
    assert metadata["weight_verification"] == WEIGHT_VERIFICATION
    assert "does not re-download or re-hash" in metadata["weight_verification"]


def test_historical_v1_cache_needs_an_explicit_path_and_is_labelled(paired, tmp_path):
    bank, _, _ = paired
    directory = cache_directory(tmp_path / "features", "siglip")
    _write_cache(directory, bank, schema=LEGACY_CACHE_SCHEMAS[0])

    with pytest.raises(ValueError, match="historical schema"):
        load_feature_cache(tmp_path / "features", "siglip", bank)

    _, metadata = load_feature_cache(tmp_path / "features", "siglip", bank, allow_legacy=True)
    assert metadata["provenance_level"] == "historical_v1"
    assert metadata["visual_input_verified"] is False
    assert any("image content" in item for item in metadata["not_established"])
    assert any("crop-contract" in item for item in metadata["not_established"])


def test_a_cache_built_for_another_bank_is_still_refused(paired, tmp_path):
    bank, _, images = paired
    directory = cache_directory(tmp_path / "features", "siglip")
    _write_cache(directory, bank, fingerprint=_identity(bank, images)["fingerprint"])
    bank.arrays["target_3d"][0, 0, 0] += 1.0
    with pytest.raises(ValueError, match="different frame bank"):
        load_feature_cache(tmp_path / "features", "siglip", bank)
