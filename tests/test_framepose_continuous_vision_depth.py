import re
from pathlib import Path

import numpy as np
import pytest

from framepose.continuous_vision_depth import CANDIDATES, SEGMENT_OF_PAIR, candidate_visual, sign_magnitude

ROOT = Path(__file__).resolve().parent.parent


def test_pair_order_matches_worklog64_segments_and_target_identity():
    from framepose.arm_depth_probe import SEGMENT_NAMES

    from framepose import continuous_vision_depth as cvd
    from framepose import learned_vision_sensor as lvs

    assert SEGMENT_OF_PAIR == SEGMENT_NAMES == ("upper", "lower", "chain")
    assert (cvd.PAIRS, cvd.GEOMETRY_FEATURES, cvd.VISUAL_PROJECTION, cvd.HIDDEN) == \
        (lvs.PAIRS, lvs.GEOMETRY_FEATURES, lvs.VISUAL_PROJECTION, lvs.HIDDEN)
    script = (ROOT / "scripts/run_continuous_visual_arm_depth.py").read_text()
    assert 'forward_targets(bank.arrays["target_3d"], bank.arrays["target_valid"])' in script
    assert "Worklog 68 visual cache identity mismatch" in script and 'manifest["readout_npz_sha256"]' in script
    assert 'bank.indices("train"), bank.indices("validation")' in script


def test_c0_exact_zero_and_c1_exact_cache():
    r = np.random.default_rng(0).normal(size=(4, 4, 8)).astype(np.float16)
    assert not candidate_visual(r, "C0_ZERO_VISION").any()
    assert candidate_visual(r, "C1_QWEN_VISION") is r
    assert CANDIDATES == ("C0_ZERO_VISION", "C1_QWEN_VISION")
    with pytest.raises(ValueError):
        candidate_visual(r, "C2")


def test_identical_architecture_initialization_tanh_range_and_deterministic_inference():
    torch = pytest.importorskip("torch")
    from framepose.continuous_vision_depth import build_regressor, predict

    torch.manual_seed(1337)
    a = build_regressor(16)
    torch.manual_seed(1337)
    b = build_regressor(16)
    for (ka, va), (kb, vb) in zip(a.state_dict().items(), b.state_dict().items()):
        assert ka == kb and torch.equal(va, vb)
    visual = (np.random.default_rng(1).normal(size=(5, 4, 16)) * 100).astype(np.float32)
    geometry = (np.random.default_rng(2).normal(size=(5, 3, 13)) * 100).astype(np.float32)
    p1 = predict(a, visual, geometry, np.arange(5), "cpu")
    p2 = predict(a, visual, geometry, np.arange(5), "cpu")
    assert p1.shape == (5, 3) and (p1 == p2).all() and (np.abs(p1) <= 1).all()
    donor = predict(a, visual, geometry, np.arange(5), "cpu", visual_positions=np.array([1, 2, 3, 4, 0]))
    assert not np.allclose(donor, p1)  # the visual donor replaces only the visual input


def test_sign_vs_magnitude_decomposition():
    gt = np.array([0.6, -0.6, 0.6, 0.05])
    pred = {"a": np.array([0.2, -0.5, -0.6, 0.3]), "b": np.array([0.55, 0.4, 0.6, -0.3])}
    out = sign_magnitude(pred, gt, np.sin(np.radians(10)))
    assert out["stable_rows"] == 3
    assert out["a"]["stable_sign_accuracy"] == pytest.approx(2 / 3)
    assert out["b"]["stable_sign_accuracy"] == pytest.approx(2 / 3)
    assert out["a"]["magnitude_error_when_own_sign_correct"] == pytest.approx((0.4 + 0.1) / 2)
    both = out["both_sign_correct:a|b"]
    assert both["rows"] == 1  # only row 0 has both signs right
    assert both["a_magnitude_error"] == pytest.approx(0.4) and both["b_magnitude_error"] == pytest.approx(0.05)
    assert out["sign_flip_counts:a|b"] == {"a_only_correct": 1, "b_only_correct": 1}


def test_no_h0_d1_depth_in_training_input_and_no_production_import():
    script = (ROOT / "scripts/run_continuous_visual_arm_depth.py").read_text()
    call = script[script.index("for c in CANDIDATES:"):script.index("rows = json.loads")]
    assert "train(visual[c], geometry, targets, masks, train_pos, val_pos, device)" in call
    assert not re.search(r"f_h0|f_d1|relation|evidence|vlm", call)
    assert "shuffled donors differ from Worklog 68" in script
    for path in sorted((ROOT / "src").rglob("*.py")):
        if path.name == "continuous_vision_depth.py":
            continue
        assert "continuous_vision_depth" not in path.read_text(), path
