import re
from pathlib import Path

import numpy as np
import pytest

from framepose.skeleton_attention_depth import (
    ARM_JOINT_INDEX, CANDIDATES, GRID, TOKEN_CENTRES, candidate_bias, joint_token_coordinates,
    point_segment_distance, skeleton_bias,
)

ROOT = Path(__file__).resolve().parent.parent


def test_constants_match_worklog68_and_joint_indices():
    from framepose import learned_vision_sensor as lvs
    from framepose import skeleton_attention_depth as sad
    from framepose.contract import JOINT_INDEX

    assert (sad.GRID, sad.RESOLUTION, sad.TOKEN_PIXELS, sad.PAIRS, sad.GEOMETRY_FEATURES, sad.VISUAL_PROJECTION, sad.HIDDEN) == \
        (lvs.GRID, lvs.RESOLUTION, lvs.TOKEN_PIXELS, lvs.PAIRS, lvs.GEOMETRY_FEATURES, lvs.VISUAL_PROJECTION, lvs.HIDDEN)
    assert ARM_JOINT_INDEX == tuple(JOINT_INDEX[j] for j in lvs.ARM)


def test_token_grid_geometry_is_row_major():
    assert TOKEN_CENTRES.shape == (GRID * GRID, 2)
    assert TOKEN_CENTRES[0].tolist() == [0, 0] and TOKEN_CENTRES[1].tolist() == [1, 0]
    assert TOKEN_CENTRES[GRID].tolist() == [0, 1]  # second row starts after GRID tokens


def test_detector_to_token_coordinates_match_worklog68():
    from types import SimpleNamespace
    from framepose.learned_vision_sensor import token_coordinates

    x = np.random.default_rng(0).uniform(0, 1, (17, 3))
    box = (35.0, -12.0, 600.0)
    a = joint_token_coordinates(x, (1080, 1920), box)
    b = token_coordinates(x, (1080, 1920), SimpleNamespace(x=box[0], y=box[1], side=box[2]))
    assert a == pytest.approx(b)


def test_point_to_segment_distance():
    pts = np.array([[0.0, 0.0], [2.0, 1.0], [5.0, 0.0], [-3.0, 4.0]])
    d = point_segment_distance(pts, np.array([0.0, 0.0]), np.array([4.0, 0.0]))
    assert d == pytest.approx([0.0, 1.0, 1.0, 5.0])
    assert point_segment_distance(pts, np.array([1.0, 1.0]), np.array([1.0, 1.0]))[0] == pytest.approx(np.sqrt(2))


def test_deterministic_bias_a0_zero_a1_minus_distance_and_invalid_pairs():
    x = np.zeros((17, 3))
    valid = np.zeros(17, bool)
    for j, (u, v) in zip(ARM_JOINT_INDEX, ((0.25, 0.25), (0.5, 0.5), (0.5, 0.75))):
        x[j, :2] = (u, v)
        valid[j] = True
    box = (0.0, 0.0, 448.0)
    bias = skeleton_bias(x, valid, (448, 448), box)
    again = skeleton_bias(x, valid, (448, 448), box)
    assert (bias == again).all() and bias.shape == (3, GRID * GRID)
    coords = joint_token_coordinates(x, (448, 448), box)
    expected = -point_segment_distance(TOKEN_CENTRES, coords[ARM_JOINT_INDEX[0]], coords[ARM_JOINT_INDEX[1]])
    assert bias[0] == pytest.approx(expected) and (bias <= 0).all() and bias[0].max() == pytest.approx(0, abs=0.6)
    assert not candidate_bias(bias, "A0_UNBIASED_ATTENTION").any()
    assert np.array_equal(candidate_bias(bias, "A1_SKELETON_BIASED_ATTENTION"), bias)
    valid[ARM_JOINT_INDEX[2]] = False
    masked = skeleton_bias(x, valid, (448, 448), box)
    assert not masked[1].any() and not masked[2].any() and masked[0].any()
    assert CANDIDATES == ("A0_UNBIASED_ATTENTION", "A1_SKELETON_BIASED_ATTENTION")


def test_identical_graph_initialization_bias_effect_and_determinism():
    torch = pytest.importorskip("torch")
    from framepose.skeleton_attention_depth import build_attention_model

    torch.manual_seed(1337)
    a = build_attention_model(16)
    torch.manual_seed(1337)
    b = build_attention_model(16)
    for (ka, va), (kb, vb) in zip(a.state_dict().items(), b.state_dict().items()):
        assert ka == kb and torch.equal(va, vb)
    g = torch.randn(2, GRID * GRID, 16)
    r = torch.randn(2, 4, 16)
    geo = torch.randn(2, 3, 13)
    zero = torch.zeros(2, 3, GRID * GRID)
    bias = -torch.rand(2, 3, GRID * GRID) * 10
    f0, att0 = a(g, r, geo, zero, return_attention=True)
    f0b = a(g, r, geo, zero)
    assert torch.equal(f0, f0b) and (f0.abs() <= 1).all()
    assert torch.allclose(att0.sum(-1), torch.ones(2, 3))
    f1, att1 = a(g, r, geo, bias, return_attention=True)
    assert not torch.allclose(att0, att1)  # only the bias changed the attention
    donor = a(g.flip(0), r, geo, bias)     # shuffled vision: only the visual grid changes
    assert not torch.allclose(donor, f1)


def test_no_gt_in_bias_no_h0_depth_inputs_same_contract_and_no_production_import():
    module = (ROOT / "src/framepose/skeleton_attention_depth.py").read_text()
    code = module.split('"""', 2)[2]
    assert not re.search(r"target_3d|target_valid|f_gt", code)
    script = (ROOT / "scripts/run_skeleton_attention_depth.py").read_text()
    assert 'forward_targets(bank.arrays["target_3d"], bank.arrays["target_valid"])' in script
    assert "skeleton_bias(bank.arrays[\"input_2d\"][p], bank.arrays[\"input_valid\"][p]" in script
    call = script[script.index("for c in CANDIDATES:"):script.index("rows = json.loads")]
    assert "train(grid, readout, geometry, biases[c], targets, masks, train_pos, val_pos, device)" in call
    assert not re.search(r"f_h0|f_d1|relation|evidence|vlm", call)
    assert "shuffled donors differ from Worklog 68" in script and "Worklog 68 cache identity mismatch" in script
    for path in sorted((ROOT / "src").rglob("*.py")):
        if path.name == "skeleton_attention_depth.py":
            continue
        assert "skeleton_attention_depth" not in path.read_text(), path
