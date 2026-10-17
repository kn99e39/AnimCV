import math
import re
from pathlib import Path

import numpy as np
import pytest

from framepose.near_plane_attribution import (
    REGIMES, STABLE, axis_squared_errors, classify, compose, depth_components, distal_decomposition, factorial_distal_errors,
    factorial_effects, factors,
)

ROOT = Path(__file__).resolve().parent.parent
RNG = np.random.default_rng(87)


def _chain(u, v, rho):
    u, v = np.asarray(u, float), np.asarray(v, float)
    u, v = u / np.linalg.norm(u), v / np.linalg.norm(v)
    m = rho * u
    return m, m + (1 - rho) * v


def test_regimes_separate_true_near_plane_from_cancellation():
    # A: both segments in the image plane (Y = depth component ~ 0)
    e, w = _chain([1, 0.0, -0.3], [1, 0.05, -1], 0.5)
    # B: chain near-plane because the segments' depths cancel (upper toward camera, forearm away)
    e2, w2 = _chain([1, -0.6, -0.2], [1, 0.6, -0.2], 0.5)
    # C: strongly out of plane
    e3, w3 = _chain([0.2, -1, 0], [0.2, -1, -0.1], 0.5)
    # D: folded, reach < 0.25
    e4, w4 = _chain([1, 0, 0], [-1, 0.02, 0.05], 0.55)
    # I: intermediate
    e5, w5 = _chain([1, -0.3, -0.5], [1, -0.3, -0.4], 0.5)
    E, W = np.stack([e, e2, e3, e4, e5]), np.stack([w, w2, w3, w4, w5])
    assert classify(E, W).tolist() == ["A_true_near", "B_cancellation", "C_out_of_plane", "D_degenerate", "I_intermediate"]
    fp, fd, fc, reach = depth_components(E, W)
    assert abs(fc[1]) < STABLE and abs(fp[1]) > STABLE and abs(fd[1]) > STABLE and fp[1] * fd[1] < 0
    assert set(REGIMES) == {"A_true_near", "B_cancellation", "I_intermediate", "C_out_of_plane", "D_degenerate"}


def test_axis_decomposition_is_exact_and_depth_is_camera_y():
    p, g = RNG.normal(size=(500, 3)), RNG.normal(size=(500, 3))
    a = axis_squared_errors(p, g)
    assert np.allclose(a["x"] + a["depth"] + a["z"], ((p - g) ** 2).sum(1))
    assert np.allclose(a["signed_depth"], p[:, 1] - g[:, 1]) and np.allclose(a["depth"], (p[:, 1] - g[:, 1]) ** 2)


def test_distal_decomposition_identity():
    e, w, ge, gw = (RNG.normal(size=(300, 3)) for _ in range(4))
    d = distal_decomposition(e, w, ge, gw)
    assert np.allclose(d["middle_part"] + d["segment_part"] + d["cross"], ((w - gw) ** 2).sum(1))


def test_factorial_hybrids_are_valid_chains_and_effects_add_up():
    n = 400
    u0, v0, u1, v1 = (RNG.normal(size=(n, 3)) for _ in range(4))
    r0, r1 = RNG.uniform(0.3, 0.7, n), RNG.uniform(0.3, 0.7, n)
    e0, w0 = compose(u0 / np.linalg.norm(u0, axis=1, keepdims=True), v0 / np.linalg.norm(v0, axis=1, keepdims=True), r0)
    e1, w1 = compose(u1 / np.linalg.norm(u1, axis=1, keepdims=True), v1 / np.linalg.norm(v1, axis=1, keepdims=True), r1)
    fu, fv, fr = factors(e1, w1)
    assert np.allclose(fr, r1) and np.allclose(np.linalg.norm(fu, axis=1), 1)
    ge, gw = compose(*factors(RNG.normal(size=(n, 3)), RNG.normal(size=(n, 3)) * 2)[:2], np.full(n, 0.5))
    y = factorial_distal_errors(e0, w0, e1, w1, ge, gw)
    assert np.allclose(y["000"], ((w0 - gw) ** 2).sum(1)) and np.allclose(y["111"], ((w1 - gw) ** 2).sum(1))
    eff = factorial_effects(y)
    s = eff["single_switch"]
    assert eff["total"] == pytest.approx(s["u"] + s["v"] + s["rho"] + eff["interaction_residual"])
    same = factorial_effects(factorial_distal_errors(e0, w0, e0, w0, ge, gw))
    assert abs(same["total"]) < 1e-12 and all(abs(v) < 1e-12 for v in same["single_switch"].values())


def test_runner_contracts_and_no_runtime_gt():
    run = (ROOT / "scripts/run_near_plane_attribution.py").read_text()
    assert run.index("DIAGNOSIS = {") < run.index("def main():")
    assert "STOP: frozen predictions did not pass the Worklog 86 parity gate" in run
    assert not re.search(r"import torch|\.backward\(|optim|smooth", run)
    pred = (ROOT / "scripts/predict_near_plane_diagnostics.py").read_text()
    assert "STOP: frozen predictions do not reproduce Worklog 86" in pred and "train_joint" not in pred
    mod = (ROOT / "src/framepose/near_plane_attribution.py").read_text()
    assert "EVALUATION-ONLY" in mod
    for path in sorted((ROOT / "src").rglob("*.py")):
        if path.name != "near_plane_attribution.py":
            assert "near_plane_attribution" not in path.read_text(), path


def test_runner_builds_factor_hybrids_from_unit_chains():
    run = (ROOT / "scripts/run_near_plane_attribution.py").read_text()
    line = next(l for l in run.splitlines() if "fac = factorial_distal_errors(" in l)
    assert "SCALE" not in line and "* SCALE ** 2" in run
