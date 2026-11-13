"""W139 synthetic contract tests (NO pretrained model, NO GPU required).  Real pretrained-model results are produced only
by scripts/w139_two_stage_harness.py and are reported separately.

The official-equivalence tests import the pinned official UniMate ``motion_inbetweening`` module read-only from the W136
scratch checkout (skipped if it is absent) and drive it and the W139 scheduled sampler with the same stub velocity model.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest

import app.experimental_unimate as U
import app.experimental_two_stage as TS

torch = pytest.importorskip("torch")
import app.experimental_two_stage_sampler as SS  # noqa: E402

UM_ROOT = Path("C:/Users/dna10/animcv-w136/ext/UniMate")
W136_COND = Path("C:/Users/dna10/animcv-w136/results/preflight/rig/cond.npy")
OFFICIAL_SHA = "ec02743b"


class _Transport:
    train_eps = 0.0
    sample_eps = 0.0

    def check_interval(self, *a, **k):
        return 0.0, 1.0


def _stub(scale=1.0, early_bomb=None):
    """Deterministic state- and time-dependent velocity (no weights)."""
    def f(x, t, cond=None):
        v = scale * (torch.tanh(x) * 0.7 - 0.3 * x + torch.cos(3 * t)[:, None, None, None] * 0.1)
        if early_bomb is not None and float(t[0]) < early_bomb:
            v = v + 1e3
        return v
    return f


def _official():
    if not (UM_ROOT / "unimate" / "inference" / "motion_inbetweening.py").is_file():
        pytest.skip("pinned official UniMate checkout absent")
    src = (UM_ROOT / "unimate" / "inference" / "motion_inbetweening.py").read_bytes()
    assert hashlib.sha256(src).hexdigest().startswith(OFFICIAL_SHA)
    sys.path.insert(0, str(UM_ROOT))
    try:
        from unimate.inference.motion_inbetweening import inbetween_sample_ode
    finally:
        sys.path.remove(str(UM_ROOT))
    return inbetween_sample_ode


def _setup(J=6, nvalid=4, T=10, kind="joint_time"):
    shape = (1, J, 12, T)
    g = torch.Generator().manual_seed(3)
    x1 = torch.randn(shape, generator=g)
    if kind == "joint_time":
        keep = torch.ones(shape, dtype=torch.bool)
        keep[:, nvalid:] = False                         # padding free, as the official joint mask
        keep[:, 2:4, :, 3:7] = False                    # free block on two valid joints
        keep[:, 2:4, 9:12, 2] = False                   # + velocity channels one frame earlier
    else:
        keep = torch.zeros((1, 1, 1, T), dtype=torch.bool)
        keep[..., :3] = True
        keep[..., 7:] = True
    km = torch.broadcast_to(keep, shape).numpy()[0]
    G = torch.from_numpy(TS.sched_set(km, nvalid)[None])
    return shape, x1, keep, G


def _run(fn, shape, x1, keep, seed=0, **kw):
    torch.manual_seed(seed)
    return fn(sample_model=kw.pop("model", _stub()), transport=_Transport(), cond={}, x1_known=x1, keep_mask=keep,
              motion_shape=shape, device=torch.device("cpu"), **kw)


# ----------------------------------------------------------------------------------------- sampler vs official ---
@pytest.mark.parametrize("kind", ["joint_time", "frames"])
def test_k0_bit_identical_to_official(kind):
    off = _official()
    shape, x1, keep, G = _setup(kind=kind)
    a = _run(off, shape, x1, keep)
    b = _run(SS.scheduled_inbetween_sample_ode, shape, x1, keep, sched_mask=G, k_s=0)
    c = _run(SS.scheduled_inbetween_sample_ode, shape, x1, keep)                       # no schedule at all
    assert torch.equal(a, b) and torch.equal(a, c)


def test_full_schedule_copies_x1_on_keep_or_G():
    shape, x1, keep, G = _setup()
    y = _run(SS.scheduled_inbetween_sample_ode, shape, x1, keep, sched_mask=G, k_s=50)
    m = torch.broadcast_to(keep, shape) | G
    assert torch.equal(y[m], x1[m])


def test_state_ks_is_training_marginal_and_next_state_is_euler():
    shape, x1, keep, G = _setup()
    rec = SS.StateRecorder(4, 25)
    _run(SS.scheduled_inbetween_sample_ode, shape, x1, keep, sched_mask=G, k_s=25, recorder=rec)
    ar = rec.arrays()
    t = ar["t"]
    eps = ar["eps"]
    g = G.numpy()[:, :4]
    x1v = x1.numpy()[:, :4]
    st = ar["states"]
    exp25 = TS.interpolant(eps, x1v, t[25])
    assert np.array_equal(st[25][g], exp25[g])
    dt = np.float32(t[26] - t[25])
    nxt = st[25] + dt * ar["velocities"][25]
    assert np.allclose(st[26][g], nxt[g], atol=1e-6)
    assert not np.array_equal(st[26][g], TS.interpolant(eps, x1v, t[26])[g])     # G released after state 25
    assert rec.calls == 1 and len(st) == 51 and len(ar["velocities"]) == 50
    chk = TS.check_recorded_states(st, eps, x1v, np.broadcast_to(keep.numpy(), shape)[0, :4], g[0], 25, t)
    assert chk["ok"]


def test_discarded_steps_have_no_effect():
    shape, x1, keep, G = _setup()
    a = _run(SS.scheduled_inbetween_sample_ode, shape, x1, keep, sched_mask=G, k_s=25)
    b = _run(SS.scheduled_inbetween_sample_ode, shape, x1, keep, sched_mask=G, k_s=25, model=_stub(early_bomb=0.49))
    valid = torch.zeros(shape, dtype=torch.bool)
    valid[:, :4] = True
    m = (torch.broadcast_to(keep, shape) | G) & valid                                      # every valid entry is pinned for t<t_s
    assert m[valid].all()
    assert torch.equal(a[valid], b[valid])


def test_x1_outside_schedule_never_read():
    shape, x1, keep, G = _setup()
    x1b = x1.clone()
    x1b[G] += 5.0                                                                           # Stage-1 content changed only on G
    a = _run(SS.scheduled_inbetween_sample_ode, shape, x1, keep, sched_mask=G, k_s=0)
    b = _run(SS.scheduled_inbetween_sample_ode, shape, x1b, keep, sched_mask=G, k_s=0)
    assert torch.equal(a, b)                                                                # k_s=0: G content invisible
    c = _run(SS.scheduled_inbetween_sample_ode, shape, x1b, keep, sched_mask=G, k_s=25)
    assert not torch.equal(a, c)                                                            # k_s=25: visible


def test_padding_is_not_scheduled():
    shape, x1, keep, G = _setup()
    assert not G[:, 4:].any()
    x1p = x1.clone()
    x1p[:, 4:] += 9.0
    a = _run(SS.scheduled_inbetween_sample_ode, shape, x1, keep, sched_mask=G, k_s=25)
    b = _run(SS.scheduled_inbetween_sample_ode, shape, x1p, keep, sched_mask=G, k_s=25)
    assert torch.equal(a[:, :4], b[:, :4])


@pytest.mark.parametrize("bad", [-1, 51, 2.5, True, "25"])
def test_invalid_k_s_refused(bad):
    shape, x1, keep, G = _setup()
    with pytest.raises((ValueError, TypeError)):
        _run(SS.scheduled_inbetween_sample_ode, shape, x1, keep, sched_mask=G, k_s=bad)
    with pytest.raises(TS.TwoStageStop):
        TS.validate_k_s(bad)


def test_non_bool_sched_refused():
    shape, x1, keep, G = _setup()
    with pytest.raises(ValueError):
        _run(SS.scheduled_inbetween_sample_ode, shape, x1, keep, sched_mask=G.float(), k_s=1)


def test_same_seed_same_eps_sha():
    shape, x1, keep, G = _setup()
    r1, r2 = SS.StateRecorder(4, 0), SS.StateRecorder(4, 0)
    _run(SS.scheduled_inbetween_sample_ode, shape, x1, keep, sched_mask=G, k_s=0, recorder=r1)
    _run(SS.scheduled_inbetween_sample_ode, shape, x1 * 2, keep, sched_mask=G, k_s=25, recorder=r2)
    assert r1.eps_sha256 == r2.eps_sha256
    r3 = SS.StateRecorder(4, 0)
    _run(SS.scheduled_inbetween_sample_ode, shape, x1, keep, seed=1, sched_mask=G, k_s=0, recorder=r3)
    assert r3.eps_sha256 != r1.eps_sha256


# ----------------------------------------------------------------------------------------- decomposition ---
def _traj(x1, eps, g, k_s, v_fn):
    t = np.linspace(0, 1, 51)
    x = eps.copy()
    states, vels = [], []
    m0 = g
    x = np.where(m0, TS.interpolant(eps, x1, t[0]), x)
    states.append(x.copy())
    for i in range(50):
        v = v_fn(x, t[i])
        vels.append(v)
        x = x + (t[i + 1] - t[i]) * v
        if i + 1 <= k_s:
            x = np.where(g, TS.interpolant(eps, x1, t[i + 1]), x)
        states.append(x.copy())
    return {"states": np.stack(states), "velocities": np.stack(vels), "t": t, "eps": eps}


def _pair(v_fn):
    rng = np.random.default_rng(0)
    eps = rng.standard_normal((1, 3, 12, 8)).astype(np.float64)
    x1B = rng.standard_normal((1, 3, 12, 8))
    x1C = x1B.copy()
    g = np.zeros((3, 12, 8), bool)
    g[1, :, 2:6] = True
    x1C[0][g] += rng.standard_normal(g.sum()) * 2
    rB = _traj(x1B, eps, g[None], 25, v_fn(x1B))
    rC = _traj(x1C, eps, g[None], 25, v_fn(x1C))
    return TS.transfer_decomposition(rB, rC, x1B, x1C, g, 25)


def test_passive_state_blind_network_gives_null_values():
    d = _pair(lambda x1: (lambda x, t: np.full_like(x, 0.3)))
    assert abs(d["beta"] - 0.5) < 1e-6 and abs(d["gamma_first_free"] - 0.5) < 1e-6 and d["network_norm"] < 1e-6
    assert d["carry_through_check_max_abs"] < TS.CROSS_TOL and d["network_term_sum_check_max_abs"] < 1e-9


def test_perfect_denoiser_reads_stage1():
    d = _pair(lambda x1: (lambda x, t: (x1 - x) / max(1 - t, 1e-6)))
    assert abs(d["gamma_first_free"] - 1.0) < 1e-6 and d["beta"] > 0.99 and d["cos_network_dx1"] > 0.99


def test_context_regenerator_ignores_stage1():
    target = np.zeros((1, 3, 12, 8))
    d = _pair(lambda x1: (lambda x, t: (target - x) / max(1 - t, 1e-6)))
    assert abs(d["gamma_first_free"]) < 1e-6 and d["cos_network_dx1"] < -0.99


def test_null_output_and_power_ratio():
    outB = np.zeros((1, 3, 12, 8), np.float32)
    x1B, x1C = np.zeros_like(outB), np.ones_like(outB)
    g = np.zeros((3, 12, 8), bool)
    g[1] = True
    n = TS.null_output(outB, x1B, x1C, g, 0.5)
    assert np.all(n[0][g] == 0.5) and np.all(n[0][~g] == 0)
    r = TS.injected_signal_ratio(x1B, x1C, g, 0.5, slots=(1,))
    assert abs(r["G"] - 1.0) < 1e-9


def test_information_path_rules():
    base = {"gamma_first_free": 0.9, "network_over_carry": 0.5, "network_norm": 10.0, "n_G": 100, "cos_network_dx1": 0.5}
    ok = TS.information_path(True, base, 50.0, 0.1, 1e-4)
    assert ok["learned_path"] and ok["direction"] == "AMPLIFIES_STAGE1"
    assert not TS.information_path(True, {**base, "gamma_first_free": 0.6}, 50.0, 0.1, 1e-4)["learned_path"]
    assert not TS.information_path(True, base, 50.0, 6.0, 1e-4)["learned_path"]          # below 10x noise floor
    assert not TS.information_path(False, base, 50.0, 0.1, 1e-4)["learned_path"]
    assert not TS.information_path(True, {**base, "network_over_carry": 0.1}, 50.0, 0.1, 1e-4)["I3"]


# ----------------------------------------------------------------------------------------- quality / judgment ---
def _wbf(hand_z, sh_z, f0=171, f1=199):
    return {f: {"world": {"hand_l": {"head": [0, 0, hand_z(f)]}, "upperarm_l": {"head": [0, 0, sh_z]}}} for f in range(f0, f1 + 1)}


def test_intent_metrics():
    up = TS.intent_metrics(_wbf(lambda f: 1.5 + 0.01 * (f - 171), 1.4))
    assert up["P1"] and up["P2"]
    drop = TS.intent_metrics(_wbf(lambda f: 1.5 - (0.6 if 180 <= f <= 185 else 0), 1.4))
    assert not drop["P1"] and not drop["P2"]


def _q(peak=20, flick=0, entry=20, straight=()):
    return {"arm_local": {"boundary_c0": {"entry_step_mm": entry, "exit_step_mm": 20}, "peak_step_mm": peak, "n_reversal_flicks": flick},
            "elbow": {"elbow_entry_step_mm": 15, "elbow_exit_step_mm": 15, "elbow_peak_step_mm": 15, "elbow_reversal_flicks": []},
            "near_straight_frames": list(straight), "hinge_opposite_frames": []}


def test_acceptability_and_superiority():
    intent = {"P1": True, "P2": True}
    ref = _q(peak=25, entry=24)
    assert TS.acceptable(_q(), intent, ref)
    assert not TS.acceptable(_q(peak=150), intent, ref)                                    # unusable step
    assert not TS.acceptable(_q(flick=1), intent, ref)
    assert not TS.acceptable(_q(), {"P1": False, "P2": True}, ref)
    assert not TS.acceptable(_q(entry=70), intent, ref)                                    # > 2*24+10
    cand = {"acceptable": True, "fidelity_dev_A_mean_mm": 100}
    sup = TS.superiority(cand, {"X": {"acceptable": False, "fidelity_dev_A_mean_mm": 50}, "Y": {"acceptable": True, "fidelity_dev_A_mean_mm": 130}})
    assert sup["all"]
    sup2 = TS.superiority(cand, {"Y": {"acceptable": True, "fidelity_dev_A_mean_mm": 110}})
    assert not sup2["all"]                                                                 # margin 20 mm not met


def test_judgment_mapping():
    lp = {"learned_path": True}
    assert TS.judgment(False, True, lp, True, True)["judgment"] == "BLOCKED"
    assert TS.judgment(True, False, lp, True, True)["judgment"] == "BLOCKED"
    assert TS.judgment(True, True, {"learned_path": False}, True, True)["judgment"] == "NOT_YET_SUPPORTED"
    assert TS.judgment(True, True, lp, True, False)["judgment"] == "PARTIALLY_SUPPORTED"
    assert TS.judgment(True, True, lp, False, True)["judgment"] == "PARTIALLY_SUPPORTED"
    assert TS.judgment(True, True, lp, True, True)["judgment"] == "SUPPORTED"


# ----------------------------------------------------------------------------------------- representation / native ---
def _rq(rng, n):
    return U.qnorm(rng.standard_normal((n, 4)))


def test_undo_endpoint_correction_roundtrip():
    rng = np.random.default_rng(1)
    raw = _rq(rng, 30)
    dS, dE = _rq(rng, 1)[0], _rq(rng, 1)[0]
    one = np.array([1.0, 0, 0, 0])
    corr = np.stack([U.qmul(U.qmul(raw[f], TS.qslerp(one, dS, 1 - f / 29)), TS.qslerp(one, dE, f / 29)) for f in range(30)])
    to_xyzw = lambda q: np.concatenate([q[..., 1:], q[..., :1]], -1)                     # noqa: E731
    rec, _ = TS.undo_endpoint_correction(to_xyzw(corr), to_xyzw(dS), to_xyzw(dE))
    assert np.max(U.qdist(rec, raw)) < 1e-9


def test_slerp_hidden_and_poison():
    rng = np.random.default_rng(2)
    q = _rq(rng, 60 * 3).reshape(60, 3, 4)
    out = TS.slerp_hidden_rotations(q, [1], 15, 44)
    assert np.array_equal(out[:16, 1], q[:16, 1]) and np.array_equal(out[44:, 1], q[44:, 1]) and np.array_equal(out[:, 0], q[:, 0])
    q2 = q.copy()
    q2[20:30, 1] = np.nan                                                                  # hidden interior never read
    assert np.array_equal(TS.slerp_hidden_rotations(q2, [1], 15, 44)[:, 1], out[:, 1])
    q3 = q.copy()
    q3[15, 1] = np.nan
    with pytest.raises(TS.TwoStageStop):
        TS.slerp_hidden_rotations(q3, [1], 15, 44)
    f = np.zeros((60, 3, 12))
    g = np.zeros((3, 12, 60), bool)
    g[2, :, 16:44] = True
    p = TS.poison_hidden(f, g)
    assert np.isnan(p[16:44, 2]).all() and not np.isnan(p[:16]).any() and not np.isnan(p[:, :2]).any()


def test_blind_key_is_permutation_and_deterministic():
    labels = ["S2A", "S2B", "S2C", "D"]
    k1, k2 = TS.blind_key("abc", labels), TS.blind_key("abc", labels)
    assert k1 == k2 and sorted(k1.values()) == sorted(labels)


def test_frame_mask_and_state_mask():
    m = TS.frame_mask([0, 1, 5])
    assert m.shape == (1, 1, 1, 60) and m.sum() == 3
    keep = np.ones((3, 12, 60), bool)
    keep[1, :, 16:44] = False
    g = TS.sched_set(keep, 3)
    assert np.array_equal(TS.state_mask(keep, g, 25, 25), np.ones_like(keep))
    assert np.array_equal(TS.state_mask(keep, g, 26, 25), keep)


@pytest.mark.skipif(not W136_COND.is_file(), reason="W136 own-rig asset absent")
def test_real_rig_w136_free_set_is_5424():
    c = np.load(W136_COND, allow_pickle=True).item()["S1_PROXY_B"]
    names, par = [str(n) for n in c["joint_names"]], np.asarray(c["parents"])
    km = U.build_keep_mask(names, par, 71)
    info = TS.check_w136_free_set(km, names, par)
    assert info["n_sched"] == TS.N_FREE_W136
    g = TS.sched_set(km)
    assert not g[50:].any() and g[TS.SLOT_UPPER, 3:9, 16:44].all() and g[TS.SLOT_LOWER, 3:9, 16:44].all()
    assert not g[names.index("upperarm_l")].any()                                          # upperarm_l slot (= clavicle rot) kept
    bad = km.copy()
    bad[names.index("spine_03"), 0, 20] = False
    with pytest.raises(U.UniStop):
        TS.check_w136_free_set(bad, names, par)


# ----------------------------------------------------------------------------------------- time map / import / normalisation ---
def test_time_map_and_window():
    assert U.action_to_index(156) == 0 and U.action_to_index(172) == 16 and U.action_to_index(199) == 43 and U.action_to_index(215) == 59
    assert U.index_to_action(16) == 172 and U.source_of(172) == 250 and U.INTERIOR == (172, 199)
    assert U.CONTEXT_LAST == 171 and U.TARGET_FIRST == 200


def test_import_domain_refusals():
    stored = {ch: {str(c): {str(f): (1.0 if c == 0 else 0.0) for f in range(1, 241)} for c in range(4)} for ch in U.EDIT_CHANNELS}
    dec = {b: np.tile(np.array([1.0, 0, 0, 0]), (U.T_MODEL, 1)) for b in U.EDIT_CHANNELS}
    assert set(U.import_values(dec, stored)["upperarm_l"]) == {str(f) for f in range(172, 200)}
    for frames, code in ((range(171, 200), "DOMAIN_FRAMES"), (range(172, 201), "DOMAIN_FRAMES"), (range(132, 141), "PIN_OVERLAP")):
        with pytest.raises(U.UniStop) as e:
            U.import_values(dec, stored, frames=frames)
        assert e.value.code == code
    with pytest.raises(U.UniStop):
        U.import_values({**dec, "hand_l": dec["upperarm_l"]}, stored)
    bad = {b: v.copy() for b, v in dec.items()}
    bad["lowerarm_l"][20] = np.nan
    with pytest.raises(U.UniStop):
        U.import_values(bad, stored)


W136_FEATS = Path("C:/Users/dna10/animcv-w136/results/preflight/features/features.npz")
W136_IO = Path("C:/Users/dna10/animcv-w136/results/run/model/raw_model_io.npz")


@pytest.mark.skipif(not (W136_FEATS.is_file() and W136_IO.is_file()), reason="W136 evidence absent")
def test_offline_normaliser_reproduces_official_x1_bit_exactly():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import w139_two_stage_harness as W
    mean, std, _ = W.stats_norm()
    x1 = W.normalise(np.load(W136_FEATS)["feats"], mean, std)
    off = np.load(W136_IO)["x1_known_proxyB_normalised"][:, :U.N_JOINTS]
    assert x1.shape == off.shape and np.array_equal(x1, off)
