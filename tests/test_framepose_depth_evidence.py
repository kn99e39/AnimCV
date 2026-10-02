import inspect
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from framepose.depth_evidence import (
    MIN_RELATIVE_STD, FramePoseDepthEvidence, cache_identity, evidence_from_raw,
    forward_depth_evidence, model_provenance, normalize_relative, sample_nearest, sha256_bytes,
)
from framepose.depth_evidence_probe import CANDIDATES, candidate_geometry, predict_depth_candidate

ROOT = Path(__file__).resolve().parent.parent
CKPT = "715fade13be8f229f8a70cc02066f656f2423a59effd0579197bbf57860e1378"


def _prov(**overrides):
    base = dict(code_revision="a561b84", torch_version="2.1.2", device="cuda",
                checkpoint_path="/x/depth_anything_v2_vits.pth", upstream={"revision": "03876f86"})
    base.update(overrides)
    return model_provenance(CKPT, **base)


def test_exact_nearest_pixel_sampling_and_invalid_or_out_of_frame_joints():
    depth = np.arange(30.0).reshape(5, 6)  # H=5, W=6
    points = np.array([[2 / 6, 3 / 5, 1.0],    # pixel (2, 3) -> 20
                       [0.0, 0.0, 1.0],        # pixel (0, 0) -> 0
                       [1.2, 0.5, 1.0],        # out of frame
                       [0.5, 0.5, 1.0]])       # invalid joint
    valid = np.array([True, True, True, False])
    raw, ok = sample_nearest(depth, points, valid, (6, 5))
    assert raw[:2].tolist() == [20.0, 0.0]
    assert ok.tolist() == [True, True, False, False]
    assert raw[2] == raw[3] == 0.0  # no neighbouring pixel is searched
    with pytest.raises(ValueError):
        sample_nearest(depth, points, valid, (5, 6))


def test_per_frame_normalization_and_scale_offset_invariance():
    raw = np.array([2.0, 4.0, 6.0, 100.0])
    ok = np.array([True, True, True, False])
    rel, avail, frame = normalize_relative(raw, ok)
    assert frame and avail.tolist() == ok.tolist()
    assert rel[:3] == pytest.approx([-1.224744871, 0.0, 1.224744871])
    assert rel[3] == 0.0
    scaled, _, _ = normalize_relative(raw * 37.0 + 5.0, ok)
    assert scaled == pytest.approx(rel)


def test_depth_sign_convention_larger_depth_anything_is_closer_so_less_forward():
    raw = np.array([1.0, 3.0])  # joint 1 has the larger Depth Anything value = closer
    forward, ok, frame = forward_depth_evidence(raw, np.array([True, True]))
    assert frame and ok.all()
    assert forward[1] < forward[0]  # closer to the camera -> smaller canonical +Y evidence


def test_degenerate_depth_map_or_too_few_joints_is_unavailable():
    flat = np.full(5, 3.0)
    rel, avail, frame = normalize_relative(flat, np.ones(5, bool))
    assert not frame and not avail.any() and not rel.any()
    near_flat = flat + np.array([0, MIN_RELATIVE_STD / 10, 0, 0, 0])
    assert not normalize_relative(near_flat, np.ones(5, bool))[2]
    assert not normalize_relative(np.array([1.0, 2.0]), np.array([True, False]))[2]


def test_evidence_record_identity_and_cache_key():
    sample = SimpleNamespace(sample_id="3dpw:a:actor0#000001", sequence_id="3dpw:a:actor0", frame_index=1,
                             split="validation")
    raw = np.array([1.0, 2.0, 3.0])
    ev = evidence_from_raw(sample, sha256_bytes(b"jpeg-bytes"), CKPT, raw, np.ones(3, bool))
    assert ev.provider == "depth_anything_v2" and ev.frame_available
    other_image = evidence_from_raw(sample, sha256_bytes(b"other-bytes"), CKPT, raw, np.ones(3, bool))
    assert other_image.digest != ev.digest  # image bytes, not filename, are identity
    assert "not metric" in ev.to_dict()["semantics"]
    with pytest.raises(ValueError):
        FramePoseDepthEvidence("s", "q", 0, "train", "i", CKPT, (1.0,), (0.5,), (False,), True)
    ident = cache_identity(_prov(), "bankdigest", "inputdigest")
    assert ident["checkpoint_sha256"] == CKPT and ident["encoder"] == "vits" and ident["input_size"] == 518
    assert ident["digest"] != cache_identity(_prov(code_revision="other"), "bankdigest", "inputdigest")["digest"]


def test_checkpoint_provenance_is_pinned_in_the_build_script():
    script = (ROOT / "scripts/build_framepose_depth_evidence.py").read_text()
    assert CKPT in script and "03876f8651c73a60fe4c2c48294e09fcb6838fcf" in script
    assert "is not the official vits file" in script
    assert "sha256_bytes(data)" in script and "imdecode(np.frombuffer(data" in script


def test_d0_fifth_channel_is_exactly_zero_and_d1_is_only_depth_evidence():
    geometry = np.random.default_rng(0).normal(size=(3, 17, 4)).astype(np.float32)
    evidence = np.random.default_rng(1).normal(size=(3, 17)).astype(np.float32)
    d0 = candidate_geometry(geometry, evidence, "D0_ZERO_DEPTH")
    d1 = candidate_geometry(geometry, evidence, "D1_DEPTH_ANYTHING")
    assert d0.shape == d1.shape == (3, 17, 5)
    assert (d0[..., 4] == 0).all()
    assert np.array_equal(d0[..., :4], d1[..., :4]) and np.array_equal(d0[..., :4], geometry)
    assert np.array_equal(d1[..., 4], evidence)
    assert CANDIDATES == ("D0_ZERO_DEPTH", "D1_DEPTH_ANYTHING")
    with pytest.raises(ValueError):
        candidate_geometry(geometry, evidence, "D2")


def test_same_split_target_objective_and_no_gt_at_inference():
    script = (ROOT / "scripts/run_framepose_depth_evidence_ab.py").read_text()
    assert "forward_targets(bank.arrays[\"target_3d\"]" in script
    assert "train_depth_candidate(inputs[name], signs, targets, masks, train, validation" in script
    assert 'SIGN_FIELDS = ("shoulder_forward_depth", "hip_forward_depth")' in script
    module = (ROOT / "src/framepose/depth_evidence_probe.py").read_text()
    assert "from framepose.arm_depth_probe import OBJECTIVE" in module
    params = list(inspect.signature(predict_depth_candidate).parameters)
    assert params == ["model", "interpret", "geometry5", "signs", "positions", "device"]
    infer = module[module.index("def predict_depth_candidate"):]
    assert not re.search(r"target|three_dpw|jointPositions", infer)


def test_d0_d1_identical_graph_parameter_count_and_deterministic_eval():
    torch = pytest.importorskip("torch")
    from framepose.depth_evidence_probe import build_depth_conditioned
    from framepose.model import ModelConfig, build_model, parameter_report

    config = ModelConfig(sign_fields=7)
    torch.manual_seed(1337)
    a, interpret = build_depth_conditioned(config)
    torch.manual_seed(1337)
    b, _ = build_depth_conditioned(config)
    assert parameter_report(a) == parameter_report(b)
    assert parameter_report(a)["parameter_count"] == parameter_report(build_model(config))["parameter_count"] + 256
    for (ka, va), (kb, vb) in zip(a.state_dict().items(), b.state_dict().items()):
        assert ka == kb and torch.equal(va, vb)  # identical initialization for D0 and D1
    geometry = np.zeros((4, 17, 5), np.float32)
    signs = np.zeros((4, 7), np.int64)
    p1 = predict_depth_candidate(a, interpret, geometry, signs, np.arange(4), device="cpu")
    p2 = predict_depth_candidate(a, interpret, geometry, signs, np.arange(4), device="cpu")
    assert p1.shape == (4, 3) and (p1 == p2).all() and (np.abs(p1) <= 1).all()
    geometry[..., 4] = 1.0  # the fifth channel actually reaches the output
    assert not np.allclose(predict_depth_candidate(a, interpret, geometry, signs, np.arange(4), device="cpu"), p1)


def test_legacy_depth_modules_and_worklog64_probe_untouched():
    assert "framepose" not in (ROOT / "src/pose/depth_sampling.py").read_text()
    assert "depth_evidence" not in (ROOT / "src/pose/depth_estimator.py").read_text()
    assert "depth_evidence" not in (ROOT / "src/framepose/arm_depth_probe.py").read_text()
    for path in ("src/framepose/model.py", "src/framepose/train.py"):
        assert "depth_evidence" not in (ROOT / path).read_text()
