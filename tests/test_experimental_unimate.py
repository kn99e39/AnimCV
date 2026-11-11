"""Synthetic contract tests for the W136 UniMate scratch instruments (no Blender, no model, no S1 data)."""
from __future__ import annotations

import numpy as np
import pytest

import app.experimental_unimate as U

RIG = [("spine_01", None), ("spine_02", "spine_01"), ("spine_03", "spine_02"), ("spine_04", "spine_03"), ("spine_05", "spine_04"),
       ("neck_01", "spine_05"), ("neck_02", "neck_01"), ("clavicle_l", "spine_05"), ("upperarm_l", "clavicle_l"),
       ("lowerarm_l", "upperarm_l"), ("hand_l", "lowerarm_l")]
for side in ("l", "r"):
    if side == "r":
        RIG += [("clavicle_r", "spine_05"), ("upperarm_r", "clavicle_r"), ("lowerarm_r", "upperarm_r"), ("hand_r", "lowerarm_r")]
    for fing, base in (("index", "metacarpal"), ("middle", "metacarpal"), ("pinky", "metacarpal"), ("ring", "metacarpal")):
        RIG += [(f"{fing}_{base}_{side}", f"hand_{side}"), (f"{fing}_01_{side}", f"{fing}_{base}_{side}"), (f"{fing}_02_{side}", f"{fing}_01_{side}")]
    RIG += [(f"thumb_01_{side}", f"hand_{side}"), (f"thumb_02_{side}", f"thumb_01_{side}")]
RIG += [("thigh_r", None), ("calf_r", "thigh_r"), ("foot_r", "calf_r"), ("thigh_l", None), ("calf_l", "thigh_l"), ("foot_l", "calf_l"),
        ("interaction", None), ("center_of_mass", None)]
RIG_PARENT = dict(RIG)


def skeleton():
    """The adapter skeleton in BFS order: pelvis root + the 49 anatomical bones, with labels."""
    names, parents = ["pelvis"], [-1]
    queue = ["pelvis"]
    kids = {}
    for n, p in RIG:
        if n in U.EXCLUDED_HELPERS:
            continue
        kids.setdefault(p if p is not None else "pelvis", []).append(n)
    while queue:
        cur = queue.pop(0)
        for c in kids.get(cur, []):
            names.append(c)
            parents.append(names.index(cur))
            queue.append(c)
    labels = []
    for n in names:
        lab = U.EXPECTED_LABELS.get(n)
        if lab is None:
            side = "Left " if n.endswith("_l") else ("Right " if n.endswith("_r") else "")
            lab = side + n.split("_")[0].capitalize()
        labels.append(lab)
    return names, np.array(parents), labels


def rand_q(rng, n=None):
    q = rng.normal(size=(4,) if n is None else (n, 4))
    return U.qnorm(q)


# ------------------------------------------------------------------------------------------------ time ---
def test_window_and_time_mapping():
    assert U.check_window()
    assert U.action_to_index(156) == 0 and U.action_to_index(215) == 59
    assert U.index_to_action(16) == 172 and U.index_to_action(43) == 199
    assert U.source_of(172) == 250 and U.source_of(199) == 277
    with pytest.raises(U.UniStop):
        U.action_to_index(216)
    with pytest.raises(U.UniStop) as e:
        U.check_fps(24)
    assert e.value.code == "NOT_30_FPS"
    with pytest.raises(U.UniStop):
        U.check_fps(20)
    assert U.check_fps(30)


# ------------------------------------------------------------------------------------------------ rotations ---
def test_quaternion_and_6d_roundtrip():
    rng = np.random.default_rng(0)
    for _ in range(50):
        q = rand_q(rng)
        assert U.qdist(U.mat_to_quat(U.quat_to_mat(q)), q) < 1e-12
        assert U.qdist(U.mat_to_quat(U.sixd_to_mat(U.quat_to_6d(q))), q) < 1e-12
        assert np.allclose(U.qmul(q, U.qinv(q)), [1, 0, 0, 0], atol=1e-12)
    m = U.quat_to_mat(rand_q(rng))
    assert np.isclose(np.linalg.det(m), 1.0)


def test_degenerate_6d_refused_not_identity():
    with pytest.raises(U.UniStop) as e:
        U.sixd_to_mat(np.zeros(6))
    assert e.value.code == "RIG_PROJECTION_FAILED"
    with pytest.raises(U.UniStop):
        U.sixd_to_mat(np.array([1, 0, 0, 2, 0, 0.0]))            # parallel axes
    with pytest.raises(U.UniStop):
        U.sixd_to_mat(np.array([1, 0, np.nan, 0, 1, 0.0]))


# ------------------------------------------------------------------------------------------------ decode ---
def test_decode_inverts_official_rebase_and_hml_order():
    rng = np.random.default_rng(1)
    names, par, _ = skeleton()
    J = len(names)
    tq = rand_q(rng, J)
    T = U.T_MODEL
    basis = {b: rand_q(rng, T) for b in U.EDIT_CHANNELS}
    feats = np.zeros((T, J, U.FEAT))
    feats[:, :, 3:9] = U.quat_to_6d(np.array([1.0, 0, 0, 0]))
    for b in U.EDIT_CHANNELS:
        child = U.children(par)[names.index(b)][0]
        feats[:, child, 3:9] = U.encode_slot(basis[b], names, par, tq, b)
    dec = U.decode_basis(feats, names, par, tq)
    for b in U.EDIT_CHANNELS:
        assert U.qdist(dec[b], basis[b]).max() < 1e-10


def test_decode_refuses_nonfinite_and_bad_shape():
    names, par, _ = skeleton()
    tq = np.tile([1.0, 0, 0, 0], (len(names), 1))
    f = np.zeros((U.T_MODEL, len(names), U.FEAT))
    f[:, :, 3:9] = U.quat_to_6d(np.array([1.0, 0, 0, 0]))
    assert set(U.decode_basis(f, names, par, tq)) == set(U.EDIT_CHANNELS)
    f[3, 0, 0] = np.inf
    with pytest.raises(U.UniStop):
        U.decode_basis(f, names, par, tq)
    with pytest.raises(U.UniStop):
        U.decode_basis(np.zeros((U.T_MODEL, len(names), 9)), names, par, tq)


def _stored_from(dec, frames_offset=0):
    st = {}
    for b, q in dec.items():
        st[b] = {str(c): {str(U.index_to_action(i) + frames_offset): float(q[i][c]) for i in range(len(q))} for c in range(4)}
        for c in range(4):                                   # neighbours for the +-1 shift probes
            st[b][str(c)][str(U.WINDOW[0] - 1)] = float(q[0][c])
            st[b][str(c)][str(U.WINDOW[1] + 1)] = float(q[-1][c])
    return st


def test_time_identity_detects_shift():
    rng = np.random.default_rng(2)
    dec = {b: rand_q(rng, U.T_MODEL) for b in U.EDIT_CHANNELS}
    st = _stored_from(dec)
    rep = U.time_identity(dec, st)
    assert all(v["shift0"] < 1e-12 for v in rep.values())
    shifted = {b: np.roll(v, 1, axis=0) for b, v in dec.items()}
    with pytest.raises(U.UniStop) as e:
        U.time_identity(shifted, st)
    assert e.value.code in ("REPRESENTATION_BLOCKED", "TIME_SHIFT")
    static = {b: np.tile(v[:1], (U.T_MODEL, 1)) for b, v in dec.items()}
    with pytest.raises(U.UniStop) as e:                    # a motionless clip cannot prove frame identity
        U.time_identity(static, _stored_from(static))
    assert e.value.code == "TIME_SHIFT"


# ------------------------------------------------------------------------------------------------ mask ---
def test_mask_support_is_exactly_left_arm_descendants_at_footprint():
    names, par, _ = skeleton()
    km = U.build_keep_mask(names, par, 71)
    info = U.validate_mask(km, names, par)
    assert "upperarm_l" not in info["generated_joints"] and "clavicle_l" not in info["generated_joints"]
    assert {"lowerarm_l", "hand_l", "thumb_02_l"} <= set(info["generated_joints"]) and len(info["generated_joints"]) == 16
    assert not any(n.endswith("_r") for n in info["generated_joints"])
    free = ~km[:len(names)]
    t_free = sorted({U.index_to_action(t) for t in np.nonzero(free)[2]})
    assert t_free[0] == 171 and t_free[-1] == 199
    i171 = U.action_to_index(171)
    assert set(np.nonzero(free[:, :, i171])[1]) == set(U.VEL)
    assert info["n_free"] == 16 * 12 * 28 + 16 * 3
    assert km[len(names):].sum() == 0                       # padding rows as the official joint mask


@pytest.mark.parametrize("mutate,code", [
    (lambda m, n: m.__setitem__((n.index("spine_01"), slice(None), 20), False), "MASK_DOMAIN"),
    (lambda m, n: m.__setitem__((n.index("lowerarm_r"), 0, 20), False), "MASK_DOMAIN"),
    (lambda m, n: m.__setitem__((n.index("upperarm_l"), 4, 20), False), "MASK_DOMAIN"),
    (lambda m, n: m.__setitem__((n.index("lowerarm_l"), 0, U.action_to_index(171)), False), "MASK_DOMAIN"),
    (lambda m, n: m.__setitem__((n.index("hand_l"), 0, U.action_to_index(200)), False), "MASK_DOMAIN"),
    (lambda m, n: m.__setitem__((n.index("hand_l"), 5, 30), True), "MASK_INCOMPLETE"),
    (lambda m, n: m.__setitem__((slice(0, len(n)),), True), "MASK_EMPTY"),
])
def test_mask_controls_fail_closed(mutate, code):
    names, par, _ = skeleton()
    km = U.build_keep_mask(names, par, 71)
    mutate(km, names)
    with pytest.raises(U.UniStop) as e:
        U.validate_mask(km, names, par)
    assert e.value.code == code


# ------------------------------------------------------------------------------------------------ asset ---
def test_asset_gate_and_controls():
    names, par, labels = skeleton()
    assert U.validate_asset(names, par, labels, U.FACE_PAIR, RIG_PARENT, raw_annotation_labels=labels)
    swapped = list(labels)
    i, j = names.index("lowerarm_l"), names.index("lowerarm_r")
    swapped[i], swapped[j] = swapped[j], swapped[i]
    with pytest.raises(U.UniStop) as e:
        U.validate_asset(names, par, swapped, U.FACE_PAIR, RIG_PARENT)
    assert e.value.code in ("LABEL_MISMATCH", "SIDE_MISMATCH")
    k = names.index("hand_l")
    with pytest.raises(U.UniStop):
        U.validate_asset(names[:k] + names[k + 1:], np.r_[par[:k], par[k + 1:]], labels[:k] + labels[k + 1:], U.FACE_PAIR, RIG_PARENT)
    with pytest.raises(U.UniStop) as e:
        U.validate_asset(names, par, labels, U.FACE_PAIR[::-1], RIG_PARENT)
    assert e.value.code == "FACING_PAIR"
    perm = list(names)
    a, b = perm.index("upperarm_l"), perm.index("upperarm_r")
    perm[a], perm[b] = perm[b], perm[a]
    with pytest.raises(U.UniStop):
        U.validate_asset(perm, par, labels, U.FACE_PAIR, RIG_PARENT)
    with pytest.raises(U.UniStop):
        U.validate_asset(names, par, labels, U.FACE_PAIR, RIG_PARENT, raw_annotation_labels=swapped)


def _tpose(names):
    tp = np.zeros((len(names), 3))
    for i, n in enumerate(names):
        x = 0.2 if n.endswith("_l") else (-0.2 if n.endswith("_r") else 0.0)
        y = 1.6 if n.startswith("neck") else (1.0 if n == "pelvis" else (0.5 if n.startswith(("calf", "foot")) else 1.2))
        tp[i] = (x, y, 0.0)
    return tp


def test_facing_and_handedness():
    names, _, _ = skeleton()
    tp = _tpose(names)
    assert U.check_facing(tp, names) > 0.99
    mir = tp.copy()
    mir[:, 0] *= -1
    with pytest.raises(U.UniStop):
        U.check_facing(mir, names)
    upside = tp.copy()
    upside[:, 1] *= -1
    with pytest.raises(U.UniStop):
        U.check_facing(upside, names)


def test_scale_uniform_vs_nonuniform():
    names, par, _ = skeleton()
    rng = np.random.default_rng(3)
    head = {n: rng.normal(size=3) for n in names if n != "pelvis"}
    off = np.zeros((len(names), 3))
    for j, n in enumerate(names[1:], 1):
        p = names[par[j]]
        off[j] = (np.subtract(head[n], head[p]) if p != "pelvis" else rng.normal(size=3)) * 0.01
    s, spread = U.check_scale(off, names, par, head)
    assert abs(s - 0.01) < 1e-12 and spread < 1e-9
    bad = off.copy()
    bad[:, 1] *= 1.05
    with pytest.raises(U.UniStop) as e:
        U.check_scale(bad, names, par, head)
    assert e.value.code == "SCALING"


# ------------------------------------------------------------------------------------------------ import ---
def test_domain_and_import_values():
    rng = np.random.default_rng(4)
    dec = {b: rand_q(rng, U.T_MODEL) for b in U.EDIT_CHANNELS}
    st = _stored_from({b: -v for b, v in dec.items()})          # reference keys with the opposite sign
    vals = U.import_values(dec, st)
    assert set(vals) == set(U.EDIT_CHANNELS)
    assert sorted(int(f) for f in vals["upperarm_l"]) == list(range(172, 200))
    for ch in vals:
        for f, q in vals[ch].items():
            q = np.asarray(q)
            assert abs(np.linalg.norm(q) - 1) < 1e-12
            ref = [st[ch][str(c)][f] for c in range(4)]
            assert np.dot(q, ref) >= 0                           # sign canonicalised, same rotation
            assert U.qdist(q, dec[ch][U.action_to_index(int(f))]) < 1e-12
    with pytest.raises(U.UniStop) as e:
        U.check_domain(range(172, 200), channels=("spine_01",))
    assert e.value.code == "DOMAIN_CHANNEL"
    with pytest.raises(U.UniStop) as e:
        U.check_domain(range(171, 201))
    assert e.value.code == "DOMAIN_FRAMES"
    with pytest.raises(U.UniStop) as e:
        U.check_domain(range(130, 158))
    assert e.value.code == "PIN_OVERLAP"
    with pytest.raises(U.UniStop):
        U.import_values(dec, st, frames=range(171, 200))


# ------------------------------------------------------------------------------------------------ source 2D ---
def test_affine_camera_fit_exact_on_synthetic_projection():
    rng = np.random.default_rng(5)
    P = rng.normal(size=(2, 4)) * 100
    X = rng.normal(size=(40, 3))
    uv = U.project(P, X)
    Pf, rms = U.fit_affine_camera(X, uv)
    assert rms < 1e-8 and np.allclose(Pf, P, atol=1e-8)
