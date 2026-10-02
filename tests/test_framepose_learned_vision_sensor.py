import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from framepose.contract import JOINT_INDEX
from framepose.learned_vision_sensor import (
    CLASSES, GRID, MODEL_FINGERPRINT, MODEL_REVISION, PAIRS, RESOLUTION, STABLE_SINE, TOKEN_PIXELS,
    bilinear_sample, pair_geometry, pair_labels, readout, sensor_inputs, token_coordinates,
)
from framepose.vlm_depth_advisor import shuffled_donors, v0_select

ROOT = Path(__file__).resolve().parent.parent


def test_pinned_model_identity_and_feature_source():
    assert MODEL_REVISION == "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
    assert MODEL_FINGERPRINT == "a0d72ded575eaa4460dad11dd1e313bc69f0403b2c97c3c1ba234af086952904"
    assert GRID == 14 and RESOLUTION == 448 and TOKEN_PIXELS == 32
    script = (ROOT / "scripts/extract_qwen3vl_visual_features.py").read_text()
    assert "get_image_features" in script and "generate(" not in script and "apply_chat_template" not in script


def test_detector_geometry_maps_to_token_grid_and_bilinear_sampling():
    box = SimpleNamespace(x=100.0, y=200.0, side=448.0)  # 1 crop px per image px
    x = np.zeros((17, 3))
    x[0, :2] = ((100 + 16) / 1000, (200 + 16) / 1000)          # centre of token (0, 0)
    x[1, :2] = ((100 + 32 * 5 + 16) / 1000, (200 + 32 * 3 + 16) / 1000)  # centre of token (row 3, col 5)
    coords = token_coordinates(x, (1000, 1000), box)
    assert coords[0] == pytest.approx([0.0, 0.0]) and coords[1] == pytest.approx([5.0, 3.0])
    grid = np.arange(GRID * GRID, dtype=np.float32).reshape(GRID, GRID, 1)
    assert bilinear_sample(grid, 5.0, 3.0)[0] == pytest.approx(3 * GRID + 5)   # row-major
    assert bilinear_sample(grid, 5.5, 3.0)[0] == pytest.approx(3 * GRID + 5.5)
    assert bilinear_sample(grid, -3.0, 99.0)[0] == pytest.approx(13 * GRID + 0)  # clamped


def test_readout_order_validity_and_global_mean():
    tokens = np.random.default_rng(0).normal(size=(GRID * GRID, 8)).astype(np.float32)
    box = SimpleNamespace(x=0.0, y=0.0, side=448.0)
    x = np.zeros((17, 3))
    valid = np.zeros(17, bool)
    for joint, (u, v) in {"left_shoulder": (16, 16), "left_elbow": (48, 16)}.items():
        x[JOINT_INDEX[joint], :2] = (u / 448, v / 448)
        valid[JOINT_INDEX[joint]] = True
    r, ok = readout(tokens, x, valid, (448, 448), box)
    assert ok.tolist() == [True, True, False]
    assert r[0] == pytest.approx(tokens[0]) and r[1] == pytest.approx(tokens[1])
    assert not r[2].any()
    assert r[3] == pytest.approx(tokens.mean(axis=0))
    a, _ = readout(tokens, x, valid, (448, 448), box)
    assert (a == r).all()  # deterministic
    with pytest.raises(ValueError):
        readout(tokens[:10], x, valid, (448, 448), box)


def test_fixed_pair_targets():
    f = np.array([[0.5, -0.5, 0.0], [STABLE_SINE + 1e-6, -STABLE_SINE - 1e-6, STABLE_SINE]])
    labels, masks = pair_labels(f, np.ones((2, 3), bool))
    assert labels.tolist() == [[0, 1, 2], [0, 1, 2]]
    assert CLASSES == ("FIRST_CLOSER", "SECOND_CLOSER", "UNCLEAR")
    assert [p[0] for p in PAIRS] == ["shoulder_elbow", "elbow_wrist", "shoulder_wrist"]


def test_pair_geometry_is_runtime_only():
    g = np.zeros((1, 17, 4), np.float32)
    g[0, JOINT_INDEX["left_shoulder"]] = (0.0, 0.0, 0.9, 1)
    g[0, JOINT_INDEX["left_elbow"]] = (0.3, 0.4, 0.8, 1)
    g[0, JOINT_INDEX["left_wrist"]] = (0.3, 0.9, 0.7, 0)
    out = pair_geometry(g)
    assert out.shape == (1, 3, 13)
    assert out[0, 0, 8:] == pytest.approx([0.3, 0.4, 0.5, 0.6, 0.8])
    assert not out[0, 1, 8:].any()  # invalid wrist -> no pairwise geometry


def test_g0_zero_and_g1_exact_cache():
    r = np.random.default_rng(1).normal(size=(3, 4, 16)).astype(np.float16)
    assert not sensor_inputs(r, None, "G0_ZERO_VISION").any()
    assert sensor_inputs(r, None, "G1_QWEN_VISION") is r
    with pytest.raises(ValueError):
        sensor_inputs(r, None, "G2")


def test_identical_graph_initialization_and_deterministic_inference():
    torch = pytest.importorskip("torch")
    from framepose.learned_vision_sensor import build_sensor

    torch.manual_seed(1337)
    a = build_sensor(16)
    torch.manual_seed(1337)
    b = build_sensor(16)
    for (ka, va), (kb, vb) in zip(a.state_dict().items(), b.state_dict().items()):
        assert ka == kb and torch.equal(va, vb)
    v = torch.zeros(2, 4, 16)
    g = torch.zeros(2, 3, 13)
    out1, out2 = a(v, g), a(v, g)
    assert out1.shape == (2, 3, 3) and torch.equal(out1, out2)


def test_no_h0_depth_or_gt_in_sensor_inputs_and_selector_rules():
    script = (ROOT / "scripts/run_learned_vision_sensor.py").read_text()
    train_call = script[script.index("for c in CANDIDATES:"):script.index("train_class_counts")]
    assert "train(inputs[c], geometry, labels, masks" in train_call
    assert not re.search(r"f_h0|relation|vlm|d1", train_call)
    extract = (ROOT / "scripts/extract_qwen3vl_visual_features.py").read_text()
    assert "target_3d" not in extract and "target_valid" not in extract
    assert v0_select(0.3, 0.4, "SECOND_CLOSER") == (0.3, "h0_same_sign")
    assert v0_select(0.3, -0.4, "UNCLEAR") == (0.3, "h0_vlm_unclear_or_invalid")
    assert v0_select(0.3, -0.4, "SECOND_CLOSER") == (-0.4, "d1_vlm_agrees")
    keys = [("a", 0), ("a", 1), ("b", 0), ("c", 0)]
    assert all(keys[d][0] != keys[i][0] for i, d in enumerate(shuffled_donors(keys)))


def test_no_production_import():
    for path in sorted((ROOT / "src").rglob("*.py")):
        if path.name == "learned_vision_sensor.py":
            continue
        assert "learned_vision_sensor" not in path.read_text(), path
