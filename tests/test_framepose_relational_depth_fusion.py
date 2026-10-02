import inspect
import re
from pathlib import Path

import numpy as np
import pytest

from framepose.contract import JOINT_INDEX
from framepose.relational_depth_fusion import (
    CANDIDATES, depth_relations, fusion_inputs, h0_forward, predict_fusion,
)

ROOT = Path(__file__).resolve().parent.parent
S, E, W = (JOINT_INDEX[j] for j in ("left_shoulder", "left_elbow", "left_wrist"))


def test_exact_relational_delta_construction_and_unavailable_semantics():
    ev = np.zeros((2, 17))
    ev[:, S], ev[:, E], ev[:, W] = [0.1, 0.1], [0.5, 0.5], [-0.2, -0.2]
    ok = np.ones((2, 17), bool)
    ok[1, E] = False  # elbow without evidence on frame 1
    rel, mask = depth_relations(ev, ok)
    assert rel[0] == pytest.approx([0.4, -0.7, -0.3])
    assert mask[0].tolist() == [True, True, True]
    assert mask[1].tolist() == [False, False, True]
    assert rel[1, 0] == 0.0 and rel[1, 1] == 0.0 and rel[1, 2] == pytest.approx(-0.3)


def test_frozen_h0_forward_fraction_extraction():
    pose = np.zeros((1, 17, 3))
    pose[0, E] = (0.0, 0.3, 0.4)      # upper: f = 0.6
    pose[0, W] = (0.0, 0.3, 0.4 + 1)  # lower: straight up -> f = 0
    f = h0_forward(pose)
    assert f[0] == pytest.approx([0.6, 0.0, 0.3 / np.linalg.norm([0, 0.3, 1.4])])
    assert h0_forward(pose * 10)[0] == pytest.approx(f[0])


def test_r0_relations_exactly_zero_r1_exactly_worklog65_relations():
    h0_f = np.random.default_rng(0).uniform(-1, 1, (5, 3))
    rel = np.random.default_rng(1).normal(size=(5, 3))
    r0 = fusion_inputs(h0_f, rel, "R0_ZERO_RELATION")
    r1 = fusion_inputs(h0_f, rel, "R1_DEPTH_RELATION")
    assert r0.shape == r1.shape == (5, 6)
    assert np.array_equal(r0[:, :3], r1[:, :3]) and (r0[:, 3:] == 0).all()
    assert np.array_equal(r1[:, 3:], rel.astype(np.float32))
    assert CANDIDATES == ("R0_ZERO_RELATION", "R1_DEPTH_RELATION")
    with pytest.raises(ValueError):
        fusion_inputs(h0_f, rel, "R2")


def test_identical_graph_initialization_and_deterministic_inference():
    torch = pytest.importorskip("torch")
    from framepose.relational_depth_fusion import build_fusion

    torch.manual_seed(1337)
    a = build_fusion()
    torch.manual_seed(1337)
    b = build_fusion()
    assert sum(p.numel() for p in a.parameters()) == 6 * 32 + 32 + 32 * 32 + 32 + 32 * 3 + 3
    for (ka, va), (kb, vb) in zip(a.state_dict().items(), b.state_dict().items()):
        assert ka == kb and torch.equal(va, vb)
    x = np.random.default_rng(2).normal(size=(4, 6)).astype(np.float32)
    p1, p2 = predict_fusion(a, x, np.arange(4)), predict_fusion(a, x, np.arange(4))
    assert p1.shape == (4, 3) and (p1 == p2).all() and (np.abs(p1) <= 1).all()


def test_same_target_objective_splits_and_no_gt_gating_or_threshold():
    script = (ROOT / "scripts/run_framepose_relational_depth_fusion.py").read_text()
    module = (ROOT / "src/framepose/relational_depth_fusion.py").read_text()
    assert 'forward_targets(bank.arrays["target_3d"], bank.arrays["target_valid"])' in script
    assert "train_fusion(inputs[name], targets, masks, train, validation, config)" in script
    assert "from framepose.arm_depth_probe import (\n    OBJECTIVE" in module
    # Inputs come only from frozen H0 and Worklog 65 relations; GT bins appear only in evaluation.
    assert "inputs = {name: fusion_inputs(h0_f, relations, name) for name in CANDIDATES}" in script
    assert script.index("F_BINS:") > script.index("def metrics")
    assert not re.search(r"\bthreshold\b|STABLE_SINE|F_BINS", module)
    assert list(inspect.signature(predict_fusion).parameters) == ["model", "inputs", "positions"]


def test_worklog65_cache_identity_is_checked_and_no_production_imports():
    script = (ROOT / "scripts/run_framepose_relational_depth_fusion.py").read_text()
    assert 'manifest["evidence_npz_sha256"]' in script and "Worklog 65 evidence cache identity changed" in script
    for path in [*sorted((ROOT / "src/framepose").glob("*.py")), *sorted((ROOT / "src/motion").glob("*.py")),
                 *sorted((ROOT / "src/retarget").glob("*.py")), ROOT / "src/pose/framepose_bridge.py"]:
        if path.name == "relational_depth_fusion.py":
            continue
        assert "relational_depth_fusion" not in path.read_text(), path
