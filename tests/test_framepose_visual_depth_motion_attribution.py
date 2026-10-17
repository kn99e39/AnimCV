import itertools
import re
from pathlib import Path

import numpy as np
import pytest

from framepose.visual_depth_motion_attribution import (
    HIGH_GT_MOTION, LOW_GT_MOTION, depth_change_terms, gt_motion_stratum, motion_terms, shapley,
)

ROOT = Path(__file__).resolve().parent.parent
RNG = np.random.default_rng(88)
PLAYERS = ("geometry", "content", "position", "global")


def _coalitions():
    return [frozenset(c) for k in range(5) for c in itertools.combinations(PLAYERS, k)]


def test_shapley_is_exact_and_recovers_additive_and_interaction_structure():
    coef = dict(zip(PLAYERS, RNG.normal(size=4)))
    values = {S: sum(coef[p] for p in S) for S in _coalitions()}
    phi = shapley(values, PLAYERS)
    assert all(phi[p] == pytest.approx(coef[p]) for p in PLAYERS)
    # pure interaction of content x position is split equally, and efficiency holds for arbitrary games
    values = {S: (2.0 if {"content", "position"} <= S else 0.0) for S in _coalitions()}
    phi = shapley(values, PLAYERS)
    assert phi["content"] == pytest.approx(1.0) and phi["position"] == pytest.approx(1.0) and phi["geometry"] == pytest.approx(0.0)
    values = {S: float(RNG.normal()) for S in _coalitions()}
    phi = shapley(values, PLAYERS)
    assert sum(phi.values()) == pytest.approx(values[frozenset(PLAYERS)] - values[frozenset()])


def _chain(n):
    u = RNG.normal(size=(n, 3))
    v = RNG.normal(size=(n, 3))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    rho = RNG.uniform(0.3, 0.7, n)
    e = rho[:, None] * u
    return e, e + (1 - rho)[:, None] * v


def test_depth_change_terms_sum_exactly_to_the_distal_depth_change():
    ea, wa = _chain(300)
    eb, wb = _chain(300)
    t = depth_change_terms(ea, wa, eb, wb)
    assert np.allclose(t["proximal_direction"] + t["distal_direction"] + t["ratio"], 100 * (wb[:, 1] - wa[:, 1]))


def test_motion_terms_and_strata_definitions():
    ga, gb = np.zeros((3, 3)), np.array([[0, 0.01, 0], [0, 0.10, 0], [0, 0.20, 0]])
    pa, pb = np.zeros((3, 3)), np.array([[0, 0.08, 0], [0, 0.10, 0], [0, -0.05, 0]])
    m = motion_terms(pa, pb, ga, gb)
    assert np.allclose(m["pred_disp"], [8, 10, -5]) and np.allclose(m["gt_disp"], [1, 10, 20])
    assert np.allclose(m["abs_residual"], [7, 0, 25]) and np.allclose(m["over_response"], [7, 0, -15])
    assert gt_motion_stratum(ga, gb).tolist() == ["low", "mid", "high"]
    assert LOW_GT_MOTION == 4.0 and HIGH_GT_MOTION == 15.0


def test_runner_contracts():
    run = (ROOT / "scripts/run_visual_depth_motion_attribution.py").read_text()
    assert run.index("DIAGNOSIS = {") < run.index("def main():")
    for check in ("STOP: Worklog 87 depth-motion residuals are not reproduced", "STOP: checkpoint identity mismatch",
                  "grid readout differs from the historical input", "coalition identity broken", "EVALUATION-ONLY"):
        assert check in run
    assert not re.search(r"\.backward\(|optim|smooth|train_joint|requires_grad_\(True\)", run)
    mod = (ROOT / "src/framepose/visual_depth_motion_attribution.py").read_text()
    assert "EVALUATION-ONLY" in mod
    for path in sorted((ROOT / "src").rglob("*.py")):
        if path.name != "visual_depth_motion_attribution.py":
            assert "visual_depth_motion_attribution" not in path.read_text(), path


def test_validation_cadence_is_declared_and_uses_available_pairs():
    run = (ROOT / "scripts/run_visual_depth_motion_attribution.py").read_text()
    assert '"validation_cadence"' in run and 'Population("3dpw_validation", fb, fb.indices("validation"), g3, b3, r3, limb["3dpw"], disc["3dpw_gt2d_px"], lambda s: 6' in run
    assert "3dpw_validation_0.1s" in run
