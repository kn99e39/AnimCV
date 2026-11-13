"""W138 Kimodo adapter: SYNTHETIC tests only (no model, no Kimodo package, no S1 data, no Blender)."""
import numpy as np
import pytest

import app.experimental_kimodo as K
import app.experimental_motionbricks as M
import app.experimental_unimate as U


def _stop(code, fn, *a, **kw):
    with pytest.raises(K.KimodoStop) as e:
        fn(*a, **kw)
    assert e.value.code == code, e.value


# ---------------------------------------------------------------------------------------------------- time ---
def test_window_and_keyframes_contract():
    assert K.N_WIN == 46 and K.index_to_action(0) == 163 and K.index_to_action(45) == 208
    assert K.key_indices() == list(range(0, 9)) + list(range(37, 46)) and len(K.KEY_ACTION) == 18 < 20
    assert K.interior_indices() == list(range(9, 37)) and [K.index_to_action(i) for i in (9, 36)] == [172, 199]
    assert not set(K.key_indices()) & set(K.interior_indices())
    assert set(K.key_indices()) | set(K.interior_indices()) == set(range(46))
    assert K.check_keyframes(K.key_indices())


@pytest.mark.parametrize("bad,code", [
    ([i + 1 for i in range(0, 9)] + list(range(37, 46)), "KEYFRAMES_DOMAIN"),
    (sorted(list(range(0, 9)) + list(range(37, 46)) + [9]), "KEYFRAMES_DOMAIN"),
    (sorted(list(range(0, 9)) + list(range(37, 46)) + [36]), "KEYFRAMES_DOMAIN"),
    (list(range(0, 8)) + list(range(37, 46)), "KEYFRAMES_DOMAIN"),
    (list(range(0, 9)) + list(range(38, 46)), "KEYFRAMES_DOMAIN"),
    (list(range(46)), "KEYFRAMES_DOMAIN"),
    (list(range(0, 9)) + list(range(37, 45)) + [46], "KEYFRAMES_DOMAIN"),
    ([-1] + list(range(0, 9)) + list(range(37, 45)), "KEYFRAMES_DOMAIN"),
    (list(reversed(list(range(0, 9)) + list(range(37, 46)))), "KEYFRAMES_FORMAT"),
    ([0, 0] + list(range(1, 9)) + list(range(37, 46)), "KEYFRAMES_FORMAT"),
    ([], "KEYFRAMES_FORMAT"),
    (["a"], "KEYFRAMES_FORMAT"),
])
def test_keyframe_refusals(bad, code):
    _stop(code, K.check_keyframes, bad)


@pytest.mark.parametrize("f", [162, 209, 0, 240])
def test_window_domain(f):
    _stop("WINDOW_DOMAIN", K.action_to_index, f)


def test_fps_and_length():
    assert K.check_fps_frames(30, 46)
    _stop("NOT_30_FPS", K.check_fps_frames, 25, 46)
    _stop("NOT_30_FPS", K.check_fps_frames, 29.97, 46)
    _stop("WINDOW_LENGTH", K.check_fps_frames, 30, 60)


# ---------------------------------------------------------------------------------------------------- mask ---
def _mask():
    m = np.zeros((46, 300), bool)
    m[K.key_indices(), :95] = True
    return m


def test_mask_contract_and_refusals():
    rep = K.validate_motion_mask(_mask())
    assert rep["constrained_rows"] == K.key_indices() and rep["n_constrained_rows"] == 18
    assert K.validate_motion_mask(_mask()[None])["n_constrained_rows"] == 18
    assert K.validate_motion_mask(_mask().astype(np.int64))
    for i, v in ((9, True), (36, True), (8, False), (37, False)):
        m = _mask()
        m[i, :] = v
        _stop("MASK_DOMAIN", K.validate_motion_mask, m)
    _stop("MASK_DOMAIN", K.validate_motion_mask, np.roll(_mask(), 1, axis=0))
    _stop("MASK_EMPTY", K.validate_motion_mask, np.zeros((46, 300), bool))
    _stop("MASK_SHAPE", K.validate_motion_mask, _mask()[:45])
    _stop("MASK_SHAPE", K.validate_motion_mask, np.stack([_mask(), _mask()]))
    _stop("MASK_DTYPE", K.validate_motion_mask, _mask() * 0.5)


def test_null_text_only():
    assert K.check_null_text("") and K.check_null_text(["", "   "])
    _stop("TEXT_MODE_BLOCKED", K.check_null_text, "A person presses overhead.")
    _stop("TEXT_MODE_BLOCKED", K.check_null_text, ["", "x"])


# ---------------------------------------------------------------------------------------------------- skeleton ---
def _synthetic_soma77():
    """A 77-joint tree that contains every W129 SOMA model joint and alignment point, with random offsets."""
    rng = np.random.default_rng(3)
    names = list(M.SOMA_MODEL_JOINTS) + ["HeadEnd", "LeftHandMiddle1", "LeftHandThumb1", "RightHandMiddle1", "RightHandThumb1",
                                          "LeftToeEnd", "RightToeEnd"]
    names += [f"Extra{i}" for i in range(77 - len(names))]
    par = {"Hips": -1}
    for child, parent in (("Spine1", "Hips"), ("Spine2", "Spine1"), ("Chest", "Spine2"), ("Neck1", "Chest"), ("Neck2", "Neck1"), ("Head", "Neck2"),
                          ("Jaw", "Head"), ("LeftEye", "Head"), ("RightEye", "Head"), ("HeadEnd", "Head")):
        par[child] = parent
    for s in ("Left", "Right"):
        for child, parent in ((f"{s}Shoulder", "Chest"), (f"{s}Arm", f"{s}Shoulder"), (f"{s}ForeArm", f"{s}Arm"), (f"{s}Hand", f"{s}ForeArm"),
                              (f"{s}HandThumbEnd", f"{s}Hand"), (f"{s}HandMiddleEnd", f"{s}Hand"), (f"{s}HandMiddle1", f"{s}Hand"),
                              (f"{s}HandThumb1", f"{s}Hand"), (f"{s}Leg", "Hips"), (f"{s}Shin", f"{s}Leg"), (f"{s}Foot", f"{s}Shin"),
                              (f"{s}ToeBase", f"{s}Foot"), (f"{s}ToeEnd", f"{s}ToeBase")):
            par[child] = parent
    order = []
    def visit(n):
        if n in order:
            return
        if par.get(n, "Hips") not in (-1, None) and par.get(n, "Hips") != n:
            visit(par.get(n, "Hips"))
        order.append(n)
    for n in names:
        visit(n)
    parents = [-1 if par.get(n, "Hips") == -1 else order.index(par.get(n, "Hips")) for n in order]
    neutral = np.zeros((77, 3))
    for i, n in enumerate(order):
        if parents[i] >= 0:
            step = rng.normal(scale=0.05, size=3)
            if "Leg" in n or "Shin" in n or "Foot" in n:
                step[1] = -0.3
            neutral[i] = neutral[parents[i]] + step
    return order, parents, neutral


def test_kimodo_soma_reference_and_fk():
    names, parents, neutral = _synthetic_soma77()
    s = K.kimodo_soma(names, parents, neutral)
    h = -neutral[:, 1].min()
    assert s.ref[0, :3].tolist() == [0.0, h, 0.0] and np.allclose(s.ref[:, 3:], [0, 0, 0, 1])
    _, P = s.fk(s.ref)
    assert np.allclose(P - P[0], neutral, atol=1e-12)                   # identity rest == neutral joints (up to the root)
    rng = np.random.default_rng(1)
    mats = np.stack([M.quat_xyzw_to_mat(q / np.linalg.norm(q)) for q in rng.normal(size=(77, 4))])
    sl = K.soma_local_from_mats(s, mats, [0.1, 0.9, -0.2])
    assert np.allclose(K.mats_from_soma_local(sl), mats, atol=1e-12) and sl[0, :3].tolist() == [0.1, 0.9, -0.2]
    assert K.positions30(s, sl).shape == (30, 3)


def test_kimodo_soma_refusals():
    names, parents, neutral = _synthetic_soma77()
    off = neutral.copy()
    off[0] = [0, 0.1, 0]
    _stop("SOMA_SKELETON_ROOT", K.kimodo_soma, names, parents, off)
    _stop("SOMA_SKELETON_SCALE", K.kimodo_soma, names, parents, neutral * 10)
    _stop("SOMA_SKELETON_SHAPE", K.kimodo_soma, names[:30], parents[:30], neutral[:30])
    s = K.kimodo_soma(names, parents, neutral)
    bad = np.tile(np.eye(3), (77, 1, 1))
    bad[11] = -np.eye(3)
    _stop("RIG_PROJECTION_FAILED", K.soma_local_from_mats, s, bad, [0, 1, 0])
    bad[11] = np.eye(3) * np.nan
    _stop("RIG_PROJECTION_FAILED", K.soma_local_from_mats, s, bad, [0, 1, 0])
    _stop("RIG_PROJECTION_FAILED", K.soma_local_from_mats, s, np.tile(np.eye(3), (30, 1, 1)), [0, 1, 0])


def test_canonical_shift():
    r = np.array([[0.3, 0.9, -0.4], [0.35, 0.91, -0.38]])
    c = r + K.canonical_shift(r)
    assert K.check_canonical(c) and np.allclose(c[:, 1], r[:, 1]) and np.allclose(c[1] - c[0], r[1] - r[0])
    _stop("CANONICAL_ORIGIN", K.check_canonical, r)


# ---------------------------------------------------------------------------------------------------- decode/import ---
def _stored(seq):
    """BaseRig-style stored keys {ch: {comp: {Action: v}}} on Action 1..240 from window quaternions placed at 163..208."""
    st = {}
    rng = np.random.default_rng(5)
    for ch in K.EDIT_CHANNELS:
        st[ch] = {str(c): {} for c in range(4)}
        for f in range(1, 241):
            q = seq[ch][f - 163] if 163 <= f <= 208 else rng.normal(size=4)
            q = q / np.linalg.norm(q)
            for c in range(4):
                st[ch][str(c)][str(f)] = float(q[c])
    return st


def _smooth_seq():
    t = np.linspace(0, 1, 46)
    out = {}
    for k, ch in enumerate(K.EDIT_CHANNELS):
        ang = 0.4 + 1.2 * t + 0.1 * k
        out[ch] = np.stack([np.cos(ang / 2), np.sin(ang / 2) * 0.6, np.sin(ang / 2) * 0.8, 0 * t], 1)
    return out


def test_time_identity_and_shift_refusal():
    seq = _smooth_seq()
    st = _stored(seq)
    rep = K.time_identity(seq, st)
    assert all(v["shift0"] < 1e-12 for v in rep.values())
    _stop("TIME_SHIFT", K.time_identity, {c: np.roll(v, 1, axis=0) for c, v in seq.items()}, st, tol=1.0)
    _stop("REPRESENTATION_BLOCKED", K.time_identity, {c: np.roll(v, 1, axis=0) for c, v in seq.items()}, st)


def test_import_values_domain_sign_and_refusals():
    seq = _smooth_seq()
    st = _stored(seq)
    neg = {c: -v for c, v in seq.items()}
    vals = K.import_values(neg, st)
    assert set(vals) == set(K.EDIT_CHANNELS) and sorted(int(f) for f in vals["upperarm_l"]) == list(range(172, 200))
    for ch in K.EDIT_CHANNELS:
        for f, q in vals[ch].items():
            ref = [st[ch][str(c)][f] for c in range(4)]
            assert np.dot(q, ref) > 0 and abs(np.linalg.norm(q) - 1) < 1e-12
    z = {c: v.copy() for c, v in seq.items()}
    z["lowerarm_l"][K.action_to_index(180)] = 0.0
    _stop("RIG_PROJECTION_FAILED", K.import_values, z, st)
    with pytest.raises(U.UniStop):
        K.import_values({**seq, "clavicle_l": seq["upperarm_l"]}, st)


# ---------------------------------------------------------------------------------------------------- conditioning ---
def _positions():
    rng = np.random.default_rng(7)
    base = rng.normal(size=(1, 30, 3))
    t = np.linspace(0, 1, 46)[:, None, None]
    return base + t * np.array([0.0, 0.5, 0.1])


def test_conditioning_report_and_detectors():
    p = _positions()
    raw = p.copy()
    raw[K.interior_indices()] += 0.02
    raw[K.key_indices()] += 0.01 / np.sqrt(3)
    rep = K.conditioning_report(raw, p)
    assert abs(rep["key_err_mean_m"] - 0.01) < 1e-9 and rep["interior_exact_equal_fraction"] == 0.0
    assert set(rep["groups"]) == set(K.GROUPS30) and sum(len(v) for v in K.group_index().values()) == 30
    assert K.check_conditioning(rep)
    far = raw.copy()
    far[K.key_indices()] += 0.1
    _stop("CONDITIONING_FIDELITY", K.check_conditioning, K.conditioning_report(far, p))
    _stop("INTERIOR_COPIED", K.check_conditioning, K.conditioning_report(p.copy(), p))
    _stop("POSITION_SHAPE", K.conditioning_report, p[:45], p[:45])
    nanr = raw.copy()
    nanr[0, 0, 0] = np.nan
    _stop("CONDITIONING_FIDELITY", K.check_conditioning, K.conditioning_report(nanr, p))


def test_continuity_flicks_and_raw_rule():
    p = _positions()
    rep = K.continuity(p[[K.action_to_index(f) for f in K.STEP_FRAMES], 13])
    assert len(rep["steps_mm"]) == 29 and rep["n_flicks"] == 0
    zig = np.zeros((30, 3))
    zig[1::2, 0] = 0.05
    assert K.continuity(zig)["n_flicks"] == 28
    raw = p.copy()
    hyb = p.copy()
    att = K.attribution(raw, p, hyb)
    body = K.body_report(raw, p)
    assert not K.raw_unusable(att, body)["unusable"]
    jump = p.copy()
    jump[K.action_to_index(172):K.action_to_index(199) + 1, K.SOMA30.index("LeftHand")] += np.array([0.2, 0.0, 0.0])
    att2 = K.attribution(jump, p, hyb)
    r2 = K.raw_unusable(att2, K.body_report(jump, p))
    assert r2["unusable"] and any("wrist" in s for s in r2["reasons"])
    assert att2["raw_vs_projected"]["wrist"]["raw_minus_projected_max_mm"] == pytest.approx(200.0)
    assert att2["raw_vs_projected"]["wrist"]["parent_orientation_part_max_mm"] == pytest.approx(200.0)


def test_hybrid_takes_only_the_two_arm_rotations():
    names, parents, neutral = _synthetic_soma77()
    s = K.kimodo_soma(names, parents, neutral)
    rng = np.random.default_rng(2)
    a = np.tile(s.ref, (4, 1, 1))
    b = a.copy()
    b[..., 3:] = rng.normal(size=b[..., 3:].shape)
    b[..., :3] += 1.0
    h = K.hybrid_local(a, b, s)
    ch = {s.names[j] for j in range(77) if not np.array_equal(h[:, j], a[:, j])}
    assert ch == {"LeftArm", "LeftForeArm"}
    assert np.array_equal(h[:, :, :3], a[:, :, :3])


def test_neck_height():
    p = np.zeros((46, 30, 3))
    p[:, K.SOMA30.index("Neck2"), 1] = 0.6
    assert K.neck_height_mm(p)[0] == pytest.approx(600.0)
