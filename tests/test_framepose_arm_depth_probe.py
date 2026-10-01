import inspect
import math
import re
from pathlib import Path

import numpy as np
import pytest

from framepose.arm_depth_probe import (
    OUTPUT_TOKEN, SEGMENT_NAMES, angle_degrees, forward_fraction, forward_targets,
    observed_plane, predict_probe, reconstruct_direction,
)
from framepose.contract import JOINT_INDEX

ROOT = Path(__file__).resolve().parent.parent


def _pose(shoulder, elbow, wrist):
    p = np.zeros((1, 17, 3))
    p[0, JOINT_INDEX["left_shoulder"]] = shoulder
    p[0, JOINT_INDEX["left_elbow"]] = elbow
    p[0, JOINT_INDEX["left_wrist"]] = wrist
    return p


def test_forward_target_construction_and_masks():
    pose = _pose((0, 0, 0), (0.3, 0.4, 0), (0.3, 0.4, -0.25))
    valid = np.ones((1, 17), bool)
    f, mask = forward_targets(pose, valid)
    assert f[0] == pytest.approx([0.8, 0.0, 0.4 / math.sqrt(0.09 + 0.16 + 0.0625)])
    assert mask.all()
    valid[0, JOINT_INDEX["left_wrist"]] = False
    _, mask = forward_targets(pose, valid)
    assert mask[0].tolist() == [True, False, False]
    degenerate = _pose((0, 0, 0), (0, 0, 0), (0, 1, 0))
    _, mask = forward_targets(degenerate, np.ones((1, 17), bool))
    assert mask[0].tolist() == [False, True, True]


def test_scale_invariance_and_same_direction_different_lengths():
    v = np.array([[0.2, -0.3, 0.5]])
    assert forward_fraction(v)[0] == pytest.approx(forward_fraction(v * 7.5)[0])
    assert forward_fraction(v)[0] == pytest.approx(forward_fraction(v * 1e-3)[0])
    pose = _pose((1, 2, 3), (1.2, 1.7, 3.5), (1.4, 1.4, 4.0))
    f1, _ = forward_targets(pose, np.ones((1, 17), bool))
    f2, _ = forward_targets(pose * 100.0, np.ones((1, 17), bool))
    assert f1 == pytest.approx(f2)


def test_correct_f_and_plane_reconstruct_original_direction():
    rng = np.random.default_rng(0)
    v = rng.normal(size=(50, 3))
    unit = v / np.linalg.norm(v, axis=1, keepdims=True)
    f, _ = forward_fraction(v * 3.0)
    plane = np.stack([v[:, 0], v[:, 2]], axis=1) * 0.37  # plane length is irrelevant
    assert reconstruct_direction(f, plane) == pytest.approx(unit, abs=1e-12)
    assert angle_degrees(reconstruct_direction(f, plane), unit) == pytest.approx(np.zeros(50), abs=1e-6)


def test_zero_f_is_image_plane_and_sign_is_preserved():
    plane = np.array([[3.0, -4.0]])
    d0 = reconstruct_direction(np.array([0.0]), plane)
    assert d0[0] == pytest.approx([0.6, 0.0, -0.8])
    assert reconstruct_direction(np.array([0.5]), plane)[0, 1] > 0
    assert reconstruct_direction(np.array([-0.5]), plane)[0, 1] < 0


def test_near_depth_axis_is_finite_and_deterministic():
    plane = np.array([[1e-3, 2e-3]] * 4)
    f = np.array([1.0, -1.0, 1.0 - 1e-15, 1.0 + 1e-12])  # last is clipped, not NaN
    a = reconstruct_direction(f, plane)
    b = reconstruct_direction(f, plane)
    assert np.isfinite(a).all() and (a == b).all()
    assert a[0] == pytest.approx([0, 1, 0]) and a[1] == pytest.approx([0, -1, 0])
    assert np.linalg.norm(a, axis=1) == pytest.approx(np.ones(4))
    with pytest.raises(ValueError):
        reconstruct_direction(np.array([0.2]), np.array([[0.0, 0.0]]))


def test_observed_plane_maps_image_y_down_to_minus_z_with_aspect():
    x = np.zeros((1, 17, 3))
    x[0, JOINT_INDEX["left_shoulder"], :2] = (0.5, 0.5)
    x[0, JOINT_INDEX["left_elbow"], :2] = (0.5, 0.6)   # straight down in the image
    x[0, JOINT_INDEX["left_wrist"], :2] = (0.6, 0.6)   # right in normalized units
    planes, ok = observed_plane(x, np.array([[1080.0, 1920.0]]))
    assert ok["upper"][0] and planes["upper"][0] == pytest.approx([0.0, -1.0])
    assert planes["lower"][0] == pytest.approx([1.0, 0.0])
    chain = np.array([108.0, -192.0]) / math.hypot(108.0, 192.0)
    assert planes["chain"][0] == pytest.approx(chain)


def test_output_tokens_are_fixed_and_distinct():
    assert SEGMENT_NAMES == ("upper", "lower", "chain")
    assert len({OUTPUT_TOKEN[name] for name in SEGMENT_NAMES}) == 3


def test_candidate_input_identity_split_identity_and_no_gt_at_inference():
    script = (ROOT / "scripts/run_framepose_arm_depth_probe.py").read_text()
    h0_script = (ROOT / "scripts/prepare_context_h0.py").read_text()
    # Same geometry tensor and the same two oracle sign fields as the H0 materialization.
    assert "geometry = geometry_tensor(bank)" in script
    assert 'SIGN_FIELDS = ("shoulder_forward_depth", "hip_forward_depth")' in script
    assert "shoulder_forward_depth,hip_forward_depth" in h0_script
    assert "input_identity > 1e-4" in script
    # Splits come from the bank's own identity; test never selects.
    assert 'bank.indices("train")' in script and "validation, model_config, config" in script
    # Inference takes only geometry + signs (+ positions); no target array.
    params = list(inspect.signature(predict_probe).parameters)
    assert params == ["model", "interpret", "geometry", "signs", "positions", "device"]
    module = (ROOT / "src/framepose/arm_depth_probe.py").read_text()
    infer = module[module.index("def predict_probe"):]
    assert not re.search(r"target|three_dpw|jointPositions", infer)


def test_probe_reuses_unchanged_estimator_with_identical_parameter_count():
    torch = pytest.importorskip("torch")
    from framepose.arm_depth_probe import build_probe
    from framepose.model import ModelConfig, build_model, parameter_report

    config = ModelConfig(sign_fields=7)
    torch.manual_seed(0)
    probe, interpret = build_probe(config)
    assert parameter_report(probe) == parameter_report(build_model(config))
    geometry = np.zeros((4, 17, 4), np.float32)
    signs = np.zeros((4, 7), np.int64)
    a = predict_probe(probe, interpret, geometry, signs, np.arange(4), device="cpu")
    b = predict_probe(probe, interpret, geometry, signs, np.arange(4), device="cpu")
    assert a.shape == (4, 3) and (a == b).all() and (np.abs(a) <= 1).all()


def test_historical_modules_untouched_by_probe():
    for path in ("src/framepose/model.py", "src/framepose/train.py", "src/framepose/losses.py"):
        assert "arm_depth_probe" not in (ROOT / path).read_text()
