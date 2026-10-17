import inspect
import re
from pathlib import Path

import numpy as np
import pytest

from framepose.depth_motion_stabilizer import (
    StabilizerConfig, calibrate, decompose, depth_tilt, kalman_cv, stabilize, tracks, with_tilt,
)

ROOT = Path(__file__).resolve().parent.parent
RNG = np.random.default_rng(89)
CFG = StabilizerConfig(sigma_meas=0.08, q_accel=20.0)


def _chains(n):
    u = RNG.normal(size=(n, 3))
    v = RNG.normal(size=(n, 3))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    rho = RNG.uniform(0.3, 0.7, n)
    e = rho[:, None] * u
    return e, e + (1 - rho)[:, None] * v


def _track(n, dt=0.1, key="s"):
    return np.array([key] * n), np.arange(n) * dt


def test_disabled_stabilizer_is_the_exact_identity():
    e, w = _chains(50)
    keys, t = _track(50)
    e2, w2 = stabilize(keys, t, e, w, np.ones(50, bool), None)
    assert np.array_equal(e2, e) and np.array_equal(w2, w)


def test_connected_geometry_rho_and_image_plane_heading_are_preserved():
    e, w = _chains(400)
    # include segments pointing almost straight at the camera (tilt near +-90 deg)
    e[:5] = np.array([[0.0, 0.5, 1e-4]]) * 1.0
    keys, t = _track(400)
    e2, w2 = stabilize(keys, t, e, w, np.ones(400, bool), CFG)
    u0, v0, r0 = decompose(e, w)
    u1, v1, r1 = decompose(e2, w2)
    assert np.isfinite(e2).all() and np.isfinite(w2).all()
    assert np.allclose(r1, r0) and np.allclose(np.linalg.norm(u1, axis=1), 1) and np.allclose(np.linalg.norm(v1, axis=1), 1)
    assert np.allclose(np.linalg.norm(w2 - e2, axis=1), 1 - r0)
    for d0, d1 in ((u0, u1), (v0, v1)):
        n0, n1 = np.hypot(d0[:, 0], d0[:, 2]), np.hypot(d1[:, 0], d1[:, 2])
        ok = n0 > 1e-6
        assert (n1[ok] > 0).all()
        assert np.allclose(d1[ok][:, [0, 2]] / n1[ok, None], d0[ok][:, [0, 2]] / n0[ok, None])
    # only the depth tilt changes
    assert np.allclose(with_tilt(u0, depth_tilt(u0)), u0)


def test_order_independence_and_track_state_isolation():
    e, w = _chains(60)
    keys = np.array(["a"] * 30 + ["b"] * 30)
    t = np.concatenate([np.arange(30) * 0.1, np.arange(30) * 0.1])
    valid = np.ones(60, bool)
    ref = stabilize(keys, t, e, w, valid, CFG)
    perm = RNG.permutation(60)
    out = stabilize(keys[perm], t[perm], e[perm], w[perm], valid[perm], CFG)
    assert np.allclose(out[1], ref[1][perm])
    # changing track b leaves track a untouched
    e_b, w_b = e.copy(), w.copy()
    e_b[30:], w_b[30:] = _chains(30)
    alt = stabilize(keys, t, e_b, w_b, valid, CFG)
    assert np.array_equal(alt[1][:30], ref[1][:30])


def test_gaps_reset_the_state_and_invalid_observations_are_not_used():
    keys = np.array(["s"] * 6)
    t = np.array([0.0, 0.1, 0.2, 2.0, 2.1, 2.2])
    assert [list(x) for x in tracks(keys, t, 1.0)] == [[0, 1, 2], [3, 4, 5]]
    z = np.array([0.1, 0.2, 0.15, 0.3, -0.4, 0.2])
    valid = np.array([True, True, True, True, False, True])
    a = kalman_cv(t[3:], z[3:], valid[3:], CFG)
    z2 = z.copy()
    z2[4] = 1.2  # an invalid observation's value must not matter
    b = kalman_cv(t[3:], z2[3:], valid[3:], CFG)
    assert np.array_equal(a, b)


def test_determinism_and_causal_mode_has_no_future_dependency():
    t = np.arange(40) * 0.1
    z = np.sin(t) + RNG.normal(0, 0.05, 40)
    valid = np.ones(40, bool)
    causal = StabilizerConfig(0.08, 20.0, mode="causal")
    assert np.array_equal(kalman_cv(t, z, valid, CFG), kalman_cv(t, z, valid, CFG))
    z2 = z.copy()
    z2[25:] += 1.0
    assert np.array_equal(kalman_cv(t, z, valid, causal)[:25], kalman_cv(t, z2, valid, causal)[:25])
    # the offline smoother is declared to use future frames
    assert not np.allclose(kalman_cv(t, z, valid, CFG)[:25], kalman_cv(t, z2, valid, CFG)[:25])


@pytest.mark.parametrize("times", [np.arange(30) * 0.1, np.arange(30) * 0.4, np.cumsum(RNG.uniform(0.02, 0.4, 30))])
def test_constant_velocity_motion_is_preserved_at_any_time_step(times):
    z = 0.2 + 0.8 * (times - times[0]) / (times[-1] - times[0])
    for mode in ("offline", "causal"):
        out = kalman_cv(times, z, np.ones(len(times), bool), StabilizerConfig(0.08, 20.0, mode=mode))
        # no lag or attenuation of a ramp: only the zero-mean velocity prior at a track start leaves < 0.15 deg
        assert np.abs(out - z).max() < 2.5e-3 and np.abs(out - z)[10:].max() < 1e-4


def test_physical_parameters_smooth_less_at_coarser_sampling():
    noise = RNG.normal(0, 0.08, 400)
    fine = kalman_cv(np.arange(400) * 0.05, noise, np.ones(400, bool), CFG)
    coarse = kalman_cv(np.arange(400) * 0.4, noise, np.ones(400, bool), CFG)
    assert np.std(fine) < np.std(coarse) < np.std(noise)


def test_moment_calibration_recovers_the_generating_scales():
    dt, n, sub = 0.1, 20000, 20
    q, sigma = 20.0, 0.08
    # continuous-time white-noise acceleration, integrated on a fine grid and sampled every dt
    h = dt / sub
    vel = np.cumsum(RNG.normal(0, np.sqrt(q * h), n * sub))
    theta = np.cumsum(vel * h)[sub - 1::sub]
    pred = theta + RNG.normal(0, sigma, n)
    cal = calibrate(pred[:, None], theta[:, None], np.zeros(n), np.arange(n) * dt, dt)
    assert cal["sigma_meas"] == pytest.approx(sigma, rel=0.1)
    assert cal["q_accel"] == pytest.approx(q, rel=0.25)


def test_runtime_interface_has_no_ground_truth_and_no_production_coupling():
    params = list(inspect.signature(stabilize).parameters)
    assert params == ["keys", "times", "e", "w", "valid", "cfg"]
    run = (ROOT / "scripts/run_depth_motion_stabilization.py").read_text()
    body = run.split("def run_stabilizer", 1)[1].split("\ndef ", 1)[0]
    assert "target" not in body and "gt" not in body.lower().replace("length", "")
    assert run.index("DECISION = {") < run.index("def main():")
    for check in ("STOP: config changed after validation", "STOP: disabled stabilizer is not the identity",
                  "Worklog 88 residuals not reproduced", "stabilized geometry invalid", "replay not allowed"):
        assert check in run
    assert not re.search(r"\.backward\(|optim\.|load_state_dict", run)
    for path in sorted((ROOT / "src").rglob("*.py")):
        if path.name != "depth_motion_stabilizer.py":
            assert "depth_motion_stabilizer" not in path.read_text(), path
