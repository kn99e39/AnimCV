import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from common.serialization import read_json
from framepose.bank import BankRequest, build_bank, load_bank
from framepose.contract import FrameBank
from framepose.crops import CROP_RESOLUTION
from framepose.features import build_feature_cache, cache_directory, load_feature_cache, read_crop
from framepose_fixtures import IMAGE_SIZE, prepared_dataset, write_images


_ROOT = Path(__file__).resolve().parent.parent
TRAIN = ["3dpw:a:actor0", "3dpw:b:actor0"]
VALIDATION = ["3dpw:v:actor0"]
TEST = ["3dpw:t:actor0"]


def _load(script: str):
    sys.path.insert(0, str(_ROOT / "src"))
    try:
        spec = importlib.util.spec_from_file_location(script, _ROOT / "scripts" / f"{script}.py")
        module = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.pop(0)


def _dataset_paths(tmp_path):
    return {split: prepared_dataset(tmp_path / f"{split}.json", split=split, sequences=names)
            for split, names in (("train", TRAIN), ("validation", VALIDATION), ("test", TEST))}


class _StubBackbone:
    """Stands in for a frozen timm tower so the cache path is testable offline."""

    def __init__(self, spec, device="cpu"):
        self.spec = spec
        self.calls = 0

    def tokens(self, crops):
        self.calls += 1
        assert crops.shape[1:] == (self.spec.input_resolution, self.spec.input_resolution, 3)
        summary = crops.reshape(len(crops), -1).mean(axis=1)
        return np.tile(summary[:, None, None], (1, self.spec.token_count, self.spec.embed_dim)).astype(np.float32)

    def provenance(self):
        return {**self.spec.to_dict(), "weights_sha256": "stub", "parameter_count": 0,
                "trainable_parameter_count": 0, "frozen": True,
                "preprocessing": {"mean": [0.5] * 3, "std": [0.5] * 3,
                                  "input_resolution": self.spec.input_resolution,
                                  "prefix_tokens_dropped": 0},
                "text_encoder_loaded": False, "language_decoder_loaded": False,
                "autoregressive_generation_used": False}


@pytest.fixture()
def paired(tmp_path):
    images = write_images(tmp_path / "imageFiles", TRAIN + VALIDATION + TEST)
    paths = _dataset_paths(tmp_path)
    requests = [BankRequest("3DPW", split, path) for split, path in paths.items()]
    bank, _ = build_bank(requests, image_roots={"3dpw_images": images}, require_rgb=True)
    index_path, _ = bank.save(tmp_path / "bank.json")
    return bank, index_path, images


def test_build_frame_bank_script_writes_a_fingerprinted_paired_bank(tmp_path, capsys):
    module = _load("build_frame_bank")
    images = write_images(tmp_path / "imageFiles", TRAIN + VALIDATION + TEST)
    paths = _dataset_paths(tmp_path)
    argv = ["build_frame_bank", "--out", str(tmp_path / "bank.json"), "--require-rgb",
            "--image-root", f"3dpw_images={images}", "--train-stride", "2"]
    for split, path in paths.items():
        argv += ["--source", f"3DPW:{split}={path}"]
    sys.argv = argv
    assert module.main() == 0
    capsys.readouterr()

    bank = load_bank(tmp_path / "bank.json")
    report = read_json(tmp_path / "bank_report.json")
    assert report["require_rgb"] is True
    assert report["split_counts"]["train"] == 24, "train stride 2 over two 24-frame sequences"
    assert report["split_counts"]["validation"] == 24
    assert report["fingerprint"]["content_digest"] == bank.content_digest()
    assert report["modality_by_source"]["3DPW"]["has_rgb"] == len(bank)
    bank.assert_split_isolation()


def test_read_crop_uses_the_bank_image_reference(paired):
    bank, _, images = paired
    crop = read_crop(bank, 0, {"3dpw_images": images})
    assert crop.shape == (CROP_RESOLUTION, CROP_RESOLUTION, 3)
    assert crop.dtype == np.uint8


def test_feature_cache_is_keyed_to_its_bank(paired, tmp_path, monkeypatch):
    bank, index_path, images = paired
    import framepose.features as features_module

    monkeypatch.setattr(features_module, "FrozenVisualBackbone", _StubBackbone)
    metadata = build_feature_cache(bank, "siglip", image_roots={"3dpw_images": images},
                                  out_root=tmp_path / "features", device="cpu", batch_size=8, workers=2)
    assert metadata["shape"] == [len(bank), 196, 768]
    assert metadata["backbone"]["frozen"] is True
    assert metadata["backbone"]["autoregressive_generation_used"] is False
    assert metadata["crop_contract"]["resolution"] == CROP_RESOLUTION

    array, loaded = load_feature_cache(tmp_path / "features", "siglip", bank)
    assert array.shape == (len(bank), 196, 768)
    assert loaded["bank_content_digest"] == bank.content_digest()

    other = FrameBank([sample for sample in bank.samples],
                      {key: value.copy() for key, value in bank.arrays.items()})
    other.arrays["target_3d"][0, 0, 0] += 1.0
    with pytest.raises(ValueError, match="different frame bank"):
        load_feature_cache(tmp_path / "features", "siglip", other)
    assert (cache_directory(tmp_path / "features", "siglip") / "tokens.npy").is_file()


def test_geometry_only_candidate_needs_no_cache(paired):
    bank, _, _ = paired
    from framepose.features import build_feature_cache as build

    with pytest.raises(ValueError, match="needs no feature cache"):
        build(bank, "none", image_roots={}, out_root="/tmp/unused")


def test_experiment_runner_produces_isolated_candidates_and_comparisons(paired, tmp_path, capsys):
    pytest.importorskip("torch")
    bank, index_path, images = paired
    features_root = tmp_path / "features"
    for key in ("vit_in21k", "siglip"):
        directory = cache_directory(features_root, key)
        directory.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(0 if key == "vit_in21k" else 1)
        np.save(directory / "tokens.npy",
                rng.normal(size=(len(bank), 196, 768)).astype(np.float16))
        from common.serialization import write_json
        from framepose.features import sample_order_digest
        write_json(directory / "meta.json", {
            "schema": "animcv_frame_pose_feature_cache_v1",
            "backbone": {"key": key, "frozen": True},
            "crop_contract": {}, "bank_content_digest": bank.content_digest(),
            "sample_order_digest": sample_order_digest([s.sample_id for s in bank.samples]),
            "sample_count": len(bank), "dtype": "float16",
            "shape": [len(bank), 196, 768], "array": "tokens.npy"})

    module = _load("run_frame_pose_experiments")
    sys.argv = ["run_frame_pose_experiments", "--bank", str(index_path),
                "--features-root", str(features_root), "--out", str(tmp_path / "experiments"),
                "--epochs", "2", "--batch-size", "16", "--device", "cpu",
                "--no-mixed-precision", "--evaluate-every", "1"]
    assert module.main() == 0
    capsys.readouterr()

    matrix = read_json(tmp_path / "experiments" / "experiment_matrix.json")
    assert set(matrix["candidates"]) == {"F0", "F1", "F2"}
    shared = matrix["shared"]
    assert shared["loss_contract"] == "baseline_geometry_v1"
    assert shared["selection_split"] == "validation"
    # Only the observation backend may differ between candidates.
    variable = {key: value["config"]["backbone"] for key, value in matrix["candidates"].items()}
    assert variable == {"F0": "none", "F1": "vit_in21k", "F2": "siglip"}
    for key, value in matrix["candidates"].items():
        for field in ("epochs", "batch_size", "learning_rate", "seed", "loss_contract"):
            assert value["config"][field] == matrix["candidates"]["F0"]["config"][field], field
    assert (matrix["candidates"]["F1"]["model"]["trainable_parameter_count"]
            == matrix["candidates"]["F2"]["model"]["trainable_parameter_count"])
    assert "test:F2_vs_F1" in matrix["comparisons"]
    assert matrix["comparisons"]["test:F1_vs_F0"]["compared_frame_count"] > 0
    assert matrix["bank_fingerprint"]["content_digest"] == bank.content_digest()

    review = _load("export_frame_pose_review")
    sys.argv = ["export_frame_pose_review", "--bank", str(index_path),
                "--experiment-root", str(tmp_path / "experiments"),
                "--split", "test", "--out", str(tmp_path / "review.json")]
    assert review.main() == 0
    capsys.readouterr()
    exported = read_json(tmp_path / "review.json")
    sequence = exported["sequences"][TEST[0]]
    assert sequence["frame_count"] == 24
    first = sequence["frames"][0]
    assert set(first["candidates"]) == {"F0", "F1", "F2"}
    assert len(first["ground_truth_3d"]) == 17
    assert len(first["candidates"]["F1"]["prediction_3d"]) == 17
    assert first["candidates"]["F0"]["metrics"]["mpjpe_mm"] is not None


def test_image_verification_script_detects_a_declared_size_mismatch(paired, tmp_path, capsys):
    module = _load("verify_frame_bank_images")
    bank, index_path, images = paired
    argv = ["verify_frame_bank_images", "--bank", str(index_path),
            "--image-root", f"3dpw_images={images}", "--sample", "0",
            "--out", str(tmp_path / "verification.json")]
    sys.argv = argv
    assert module.main() == 0
    capsys.readouterr()
    report = read_json(tmp_path / "verification.json")
    assert report["passed"] is True
    assert report["checked_samples"] == len(bank)
    assert report["image_size_mismatch_count"] == 0

    from PIL import Image
    reference = bank.samples[0].image_reference
    path = Path(images) / reference.relative_path
    Image.fromarray(np.zeros((IMAGE_SIZE[1] // 2, IMAGE_SIZE[0] // 2, 3), dtype=np.uint8)).save(path)
    sys.argv = argv
    assert module.main() == 1
    capsys.readouterr()
    report = read_json(tmp_path / "verification.json")
    assert report["passed"] is False
    assert report["image_size_mismatch_count"] == 1
