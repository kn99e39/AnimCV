"""Synthetic tests for the W135 experimental elbow-swivel prior (no Blender, no model; SYNTHETIC two-bone rig)."""
from __future__ import annotations

import copy

import numpy as np
import pytest

import app.experimental_elbow_prior as P


def _rest_matrix(head, direction, up):
    y = np.asarray(direction, float) / np.linalg.norm(direction)
    z = np.asarray(up, float) - np.dot(up, y) * y
    z /= np.linalg.norm(z)
    x = np.cross(y, z)
    m = np.eye(4)
    m[:3, :3] = np.column_stack([x, y, z])
    m[:3, 3] = head
    return m


def _rest():
    c = _rest_matrix([0, 0, 140], [1, 0, -0.1], [0, 0, 1])
    u_head = c[:3, 3] + c[:3, 1] * 18.0
    u = _rest_matrix(u_head, [0.5, 0, -1], [0, -1, 0])
    l_head = u_head + u[:3, 1] * 28.0
    lo = _rest_matrix(l_head, [0.4, -0.6, -0.6], [0, 0, 1])
    h_head = l_head + lo[:3, 1] * 27.0
    h = _rest_matrix(h_head, [0.4, -0.6, -0.6], [0, 0, 1])
    out = {}
    for name, m, parent, tail in ((P.PARENT, c, None, u_head), (P.UPPER, u, P.PARENT, l_head), (P.LOWER, lo, P.UPPER, h_head),
                                  (P.HAND, h, P.LOWER, h_head + h[:3, 1] * 4.0)):
        out[name] = {"parent": parent, "matrix_local": m.tolist(), "head_local": m[:3, 3].tolist(), "tail_local": list(tail)}
    return out


def _quat(axis, deg):
    a = np.asarray(axis, float) / np.linalg.norm(axis)
    h = np.radians(deg) / 2
    return np.r_[np.cos(h), np.sin(h) * a]


def _fk(rest, K, qu, ql):
    R = {b: np.asarray(rest[b]["matrix_local"], float) for b in rest}
    Pu = K @ np.linalg.inv(R[P.PARENT]) @ R[P.UPPER] @ P._m4(P.wxyz_to_mat(qu), np.zeros(3))
    Pl = Pu @ np.linalg.inv(R[P.UPPER]) @ R[P.LOWER] @ P._m4(P.wxyz_to_mat(ql), np.zeros(3))
    Ph = Pl @ np.linalg.inv(R[P.LOWER]) @ R[P.HAND]
    return {P.PARENT: K, P.UPPER: Pu, P.LOWER: Pl, P.HAND: Ph}


def _dump(rest, poses, sha="b" * 64, fps=30, frames=range(160, 211)):
    """poses: f -> (K, qu, ql)"""
    fr, stored = [], {P.UPPER: {str(c): {} for c in range(4)}, P.LOWER: {str(c): {} for c in range(4)}}
    for f in frames:
        K, qu, ql = poses(f)
        pm = _fk(rest, K, qu, ql)
        fr.append({"blender_frame": f, "pose_matrix": {b: m.tolist() for b, m in pm.items()}})
        for ch, q in ((P.UPPER, qu), (P.LOWER, ql)):
            for c in range(4):
                stored[ch][str(c)][str(f)] = float(q[c])
    return {"rest": rest, "frames": fr, "stored": stored, "blend_sha256": sha, "actions_in_file": ["X"],
            "scene": {"fps": fps, "fps_base": 1.0, "frame_start": 1, "frame_end": 240}}


def _K(f):
    R = P.wxyz_to_mat(_quat([0.2, 0.3, 1.0], 10 + 0.5 * f))
    return P._m4(R, [1.0 + 0.01 * f, -2.0, 0.5]) @ np.asarray(_rest()[P.PARENT]["matrix_local"], float)


def _poses_b(f):
    t = (f - 160) / 50.0
    return _K(f), _quat([1, 0.2, 0.1], 20 + 60 * t), _quat([1, 0, 0], 50 + 20 * t)


def _poses_c(f):
    t = (f - 160) / 50.0
    return _K(f), _quat([0.3, 1, 0.2], -30 + 40 * t), _quat([1, 0, 0], 70 - 10 * t)


@pytest.fixture(scope="module")
def dumps():
    rest = _rest()
    return _dump(rest, _poses_b, sha="b" * 64), _dump(rest, _poses_c, sha="c" * 64), _dump(rest, _poses_b, sha="a" * 64)


def test_quaternion_roundtrip_and_sign():
    rng = np.random.default_rng(0)
    for _ in range(50):
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        back = P.mat_to_wxyz(P.wxyz_to_mat(q))
        assert min(np.abs(back - q).max(), np.abs(back + q).max()) < 1e-12
        assert np.dot(P.canon_sign(-q, q), q) > 0


def test_time_mapping_and_pin_are_declared_constants():
    assert P.INTERIOR == (172, 199) and (P.INTERIOR[0] + 78, P.INTERIOR[1] + 78) == (250, 277)
    assert P.CONTEXT_LAST + 78 == 249 and P.TARGET_FIRST + 78 == 278
    assert (P.PIN_ACTION[0] + 78, P.PIN_ACTION[-1] + 78) == (210, 218) and P.FPS == 30.0


def test_swivel_rotates_with_elbow_and_ignores_wrist_distance_and_parent():
    S, W = np.zeros(3), np.array([0.0, 0.0, -40.0])
    E = np.array([10.0, 0.0, -20.0])
    r = np.array([1.0, 0.0, 0.0])
    psi0 = P.swivel(S, E, W, r)[0]
    for th in (15, 60, 170):
        R = P.axis_angle(W - S, np.radians(th))
        assert abs(P.wrap(P.swivel(S, R @ E, W, r)[0] - psi0 - np.radians(th))) < 1e-12
    # wrist distance along the axis does not change the swivel
    assert abs(P.swivel(S, E, W * 0.7, r)[0] - psi0) < 1e-12
    # a common rigid motion of the whole arm-local frame (points and reference) does not change it
    Q = P.axis_angle([1, 2, 3], 0.7)
    assert abs(P.swivel(Q @ S + 3, Q @ E + 3, Q @ W + 3, Q @ r)[0] - psi0) < 1e-12


def test_mirrored_matrix_refused():
    m = np.eye(4)
    m[0, 0] = -1
    with pytest.raises(P.PriorStop) as e:
        P.rot(m)
    assert e.value.code == "IMPROPER_ROTATION"


def test_solver_preserves_wrist_lengths_and_lowerarm_and_hits_target(dumps):
    db, dc, da = dumps
    prior = P.build_prior(dc, "c" * 64)
    vals, diag = P.solve(db, da, prior["psi_rad"])
    for f, d in diag["per_frame"].items():
        assert d["wrist_err"] < 1e-6 and d["upper_len_err"] < 1e-6 and d["lower_len_err"] < 1e-6
        assert d["swivel_err_rad"] < 1e-9
        assert np.allclose(vals[P.LOWER][f], [db["stored"][P.LOWER][str(c)][f] for c in range(4)])
    assert max(d["elbow_shift"] for d in diag["per_frame"].values()) > 1.0     # the learned swivel actually moved the elbow


def test_neutral_prior_is_identity(dumps):
    db, _, da = dumps
    vals, diag = P.solve(db, da, P.neutral_prior(db, "b" * 64)["psi_rad"])
    for f, d in diag["per_frame"].items():
        assert d["elbow_shift"] < 1e-9 and abs(d["delta_rad"]) < 1e-12
        q = np.array(vals[P.UPPER][f])
        qb = np.array([db["stored"][P.UPPER][str(c)][f] for c in range(4)])
        assert min(np.abs(q - qb).max(), np.abs(q + qb).max()) < 1e-12


def test_prior_reads_only_swivel_not_wrist_path(dumps):
    """Changing C's wrist distance (forearm flexion) while keeping its swivel leaves the prior unchanged."""
    db, dc, _ = dumps
    rest = dc["rest"]
    p0 = P.build_prior(dc, "c" * 64)["psi_rad"]
    # same upperarm, different elbow flexion -> different wrist path, same bend plane
    dc2 = _dump(rest, lambda f: (_K(f), _poses_c(f)[1], _quat([1, 0, 0], 40)), sha="c" * 64)
    p2 = P.build_prior(dc2, "c" * 64)["psi_rad"]
    W1 = np.array([P.ArmFrame(dc, f).SEW()[2] for f in range(172, 200)])
    W2 = np.array([P.ArmFrame(dc2, f).SEW()[2] for f in range(172, 200)])
    assert np.linalg.norm(W1 - W2, axis=1).max() > 1.0
    # C's parent / root / global travel (clavicle, spine, object) does not enter the prior at all
    moved = lambda f: P._m4(P.wxyz_to_mat(_quat([1, -0.5, 0.3], 3.0 * f)), [5.0 * f, 2.0, -f]) @ _K(f)  # noqa: E731
    dc3 = _dump(rest, lambda f: (moved(f), *_poses_c(f)[1:]), sha="c" * 64)
    p3 = P.build_prior(dc3, "c" * 64)["psi_rad"]
    assert max(abs(p3[k] - p0[k]) for k in p0) < 1e-9
    # the swivel is defined about each branch's own S-W axis, so it is not literally constant under flexion;
    # what must hold is that the prior carries no position / timing / root data at all:
    assert set(P.build_prior(dc, "c" * 64)) >= {"psi_rad", "content_sha256", "source_blend_sha256"}
    assert all(isinstance(v, float) for v in p0.values()) and all(isinstance(v, float) for v in p2.values())


def test_prior_verification_fails_closed(dumps):
    db, dc, _ = dumps
    good = P.build_prior(dc, "c" * 64)
    assert P.verify_prior(good, dc, "LEARNED_ELBOW_SWIVEL")
    mirrored = copy.deepcopy(good)
    mirrored["psi_rad"] = {k: -v for k, v in good["psi_rad"].items()}
    mirrored["content_sha256"] = P.sha256_json(mirrored["psi_rad"])
    shifted = copy.deepcopy(good)
    shifted["psi_rad"] = {str(int(k) + 1): v for k, v in good["psi_rad"].items()}
    shifted["content_sha256"] = P.sha256_json(shifted["psi_rad"])
    corrupt = copy.deepcopy(good)
    corrupt["psi_rad"]["180"] += 0.01
    side = copy.deepcopy(good)
    side["side"] = "r"
    for bad, code in ((mirrored, "PRIOR_PROVENANCE_MISMATCH"), (shifted, "PRIOR_FRAME_IDENTITY"),
                      (corrupt, "PRIOR_CONTENT_HASH"), (side, "PRIOR_SIDE_MISMATCH")):
        with pytest.raises(P.PriorStop) as e:
            P.verify_prior(bad, dc, "LEARNED_ELBOW_SWIVEL")
        assert e.value.code == code
    with pytest.raises(P.PriorStop) as e:
        P.build_prior(dc, "d" * 64)
    assert e.value.code == "SOURCE_IDENTITY"
    with pytest.raises(P.PriorStop) as e:
        P.verify_prior(good, db, "LEARNED_ELBOW_SWIVEL")                  # prior bound to C, verified against B
    assert e.value.code == "SOURCE_IDENTITY"


def test_domain_pin_and_channel_refusals(dumps):
    db, dc, da = dumps
    pri = P.build_prior(dc, "c" * 64)["psi_rad"]
    for frames, code in ((range(171, 200), "DOMAIN_FRAMES"), (range(172, 201), "DOMAIN_FRAMES"), (range(130, 142), "PIN_OVERLAP")):
        with pytest.raises(P.PriorStop) as e:
            P.solve(db, da, pri, frames=frames)
        assert e.value.code == code
    with pytest.raises(P.PriorStop) as e:
        P.check_domain(range(172, 200), channels=("spine_01",))
    assert e.value.code == "DOMAIN_CHANNEL"


def test_unreachable_wrist_refused(dumps):
    db, dc, da = dumps
    targets = {}
    for f in range(172, 200):
        S, _, W = P.ArmFrame(db, f).SEW()
        targets[f] = S + (W - S) / np.linalg.norm(W - S) * 200.0
    with pytest.raises(P.PriorStop) as e:
        P.solve(db, da, P.neutral_prior(db, "b" * 64)["psi_rad"], wrist_targets=targets)
    assert e.value.code == "WRIST_UNREACHABLE"


def test_elbow_flip_refused(dumps):
    db, _, da = dumps
    pri = dict(P.neutral_prior(db, "b" * 64)["psi_rad"])
    pri["185"] = P.wrap(pri["185"] + np.pi)
    with pytest.raises(P.PriorStop) as e:
        P.solve(db, da, pri)
    assert e.value.code == "ELBOW_FLIP"


def test_non_30fps_refused(dumps):
    db, dc, da = dumps
    d24 = copy.deepcopy(dc)
    d24["scene"]["fps"] = 24
    with pytest.raises(P.PriorStop) as e:
        P.build_prior(d24, "c" * 64)
    assert e.value.code == "NOT_30_FPS"


def test_reference_singularity_refused():
    S, E = np.zeros(3), np.array([5.0, 10.0, -20.0])
    W = np.array([40.0, 0.0, 0.0])                     # arm axis parallel to the lateral reference
    _, mr, _ = P.swivel(S, E, W, np.array([1.0, 0.0, 0.0]))
    assert mr < P.MIN_REF_MARGIN_SIN


def test_select_reference_is_max_min_margin(dumps):
    db, dc, _ = dumps
    best, rows = P.select_reference([db, dc], range(171, 201))
    assert best in P.REFERENCE_CANDIDATES and rows[best] == max(rows.values())
