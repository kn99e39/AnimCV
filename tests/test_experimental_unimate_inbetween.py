"""Synthetic contract tests for the W137 official-inbetween UniMate scratch instruments (no Blender, no model, no S1 data)."""
from __future__ import annotations

import numpy as np
import pytest

import app.experimental_unimate as U
import app.experimental_unimate_inbetween as T


def _rot(axis, deg):
    a = np.radians(deg)
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * K @ K


# ------------------------------------------------------------------------------------------------ keep frames / time ---
def test_keep_frames_declared_set():
    assert len(T.KEEP_FRAMES) == 32 and len(T.FREE_FRAMES) == 28
    assert set(T.KEEP_FRAMES) | set(T.FREE_FRAMES) == set(range(60)) and not set(T.KEEP_FRAMES) & set(T.FREE_FRAMES)
    assert [U.index_to_action(i) for i in T.FREE_FRAMES] == list(range(172, 200))
    assert U.index_to_action(15) == 171 and U.index_to_action(44) == 200
    assert T.parse_keep_arg(T.KEEP_FRAMES_ARG) == list(T.KEEP_FRAMES)
    rep = T.check_keep_frames(T.parse_keep_arg(T.KEEP_FRAMES_ARG))
    assert rep == {"n_keep": 32, "n_free": 28, "free_action": [172, 199]}
    assert U.source_of(172) == 250 and U.source_of(199) == 277


@pytest.mark.parametrize("bad", [[0, -1], list(range(0, 15)) + list(range(44, 60)), list(range(0, 16)) + list(range(45, 60)),
                                 sorted(list(T.KEEP_FRAMES) + [16]), [i + 1 for i in T.KEEP_FRAMES if i < 59],
                                 list(reversed(T.KEEP_FRAMES)), list(T.KEEP_FRAMES) + [60], list(T.KEEP_FRAMES) + [15]])
def test_keep_frames_refuses(bad):
    with pytest.raises(U.UniStop) as e:
        T.check_keep_frames(bad)
    assert e.value.code in ("KEEP_FRAMES_FORMAT", "KEEP_FRAMES_DOMAIN")


def test_unsupported_frame_rate_and_window():
    for fps in (24, 25, 20, 60):
        with pytest.raises(U.UniStop):
            U.check_fps(fps)
    assert U.check_fps(30)
    assert U.check_window()


# ------------------------------------------------------------------------------------------------ temporal mask ---
def test_temporal_mask_exact():
    m = T.temporal_mask()
    assert m.shape == (1, 1, 1, 60) and m.dtype == bool and int(m.sum()) == 32
    rep = T.validate_temporal_mask(m)
    assert rep["kept"] == list(T.KEEP_FRAMES)
    # the official sampler broadcasts this mask over every joint and channel
    full = np.broadcast_to(m, (1, 71, 12, 60))
    assert full[0, :, :, 15].all() and full[0, :, :, 44].all() and not full[0, :, :, 16:44].any()


@pytest.mark.parametrize("mut,code", [
    (lambda m: np.roll(m, 1, axis=3), "MASK_DOMAIN"), (lambda m: np.roll(m, -1, axis=3), "MASK_DOMAIN"),
    (lambda m: np.ones_like(m), "MASK_EMPTY"), (lambda m: np.zeros_like(m), "MASK_EMPTY"),
    (lambda m: np.broadcast_to(m, (1, 50, 1, 60)).copy(), "MASK_NOT_TEMPORAL"),
    (lambda m: np.broadcast_to(m, (1, 1, 12, 60)).copy(), "MASK_NOT_TEMPORAL"),
    (lambda m: m[..., :59], "MASK_NOT_TEMPORAL"), (lambda m: m.astype(np.int64), "MASK_DTYPE")])
def test_temporal_mask_refuses(mut, code):
    with pytest.raises(U.UniStop) as e:
        T.validate_temporal_mask(mut(T.temporal_mask()))
    assert e.value.code == code


def test_single_frame_mask_errors():
    for i, v in ((15, False), (44, False), (16, True), (43, True), (0, False), (59, False)):
        m = T.temporal_mask()
        m[0, 0, 0, i] = v
        with pytest.raises(U.UniStop) as e:
            T.validate_temporal_mask(m)
        assert e.value.code == "MASK_DOMAIN"


def test_mode_flags():
    assert T.check_mode(True, False)
    for args in ((False, True), (True, True), (False, False), (True, False, True)):
        with pytest.raises(U.UniStop) as e:
            T.check_mode(*args)
        assert e.value.code == "CONDITIONING_BLOCKED"


# ------------------------------------------------------------------------------------------------ conditioning ---
NAMES = ["pelvis", "spine_01", "spine_02", "spine_03", "spine_04", "spine_05", "neck_01", "neck_02", "clavicle_l", "upperarm_l",
         "lowerarm_l", "hand_l", "index_01_l", "clavicle_r", "upperarm_r", "lowerarm_r", "hand_r", "index_01_r",
         "thigh_l", "calf_l", "foot_l", "thigh_r", "calf_r", "foot_r"]
PARENTS = [-1, 0, 1, 2, 3, 4, 5, 6, 5, 8, 9, 10, 11, 5, 13, 14, 15, 16, 0, 18, 19, 0, 21, 22]


def _feats(seed=0):
    return np.random.default_rng(seed).normal(size=(60, len(NAMES), 12))


def test_conditioning_report_and_gate():
    g = _feats()
    s = g.copy()
    s[16:44] += np.random.default_rng(1).normal(size=s[16:44].shape)
    rep = T.conditioning_report(s, g, NAMES)
    assert rep["kept_max_abs"] == 0.0 and rep["interior_exact_equal_fraction"] == 0.0
    assert set(rep["groups"]) >= {"root", "torso", "head", "left_clavicle", "left_arm", "right_arm", "legs", "fingers_left", "fingers_right"}
    assert T.check_conditioning(rep)


@pytest.mark.parametrize("joint,frame", [("pelvis", 0), ("spine_03", 15), ("lowerarm_r", 50), ("hand_l", 44), ("foot_l", 59)])
def test_conditioning_kept_violation(joint, frame):
    g = _feats()
    s = g.copy()
    s[16:44] += 1.0
    s[frame, NAMES.index(joint), 5] += 1e-3
    with pytest.raises(U.UniStop) as e:
        T.check_conditioning(T.conditioning_report(s, g, NAMES))
    assert e.value.code == "CONDITIONING_FIDELITY"


def test_conditioning_interior_copy_detected():
    g = _feats()
    with pytest.raises(U.UniStop) as e:
        T.check_conditioning(T.conditioning_report(g.copy(), g, NAMES))
    assert e.value.code == "INTERIOR_COPIED"
    s = g.copy()                                    # only the body generated, left arm copied -> still refused
    s[16:44] += 1.0
    for j in ("upperarm_l", "lowerarm_l", "hand_l"):
        s[16:44, NAMES.index(j)] = g[16:44, NAMES.index(j)]
    with pytest.raises(U.UniStop):
        T.check_conditioning(T.conditioning_report(s, g, NAMES))


def test_conditioning_shape_and_finite():
    g = _feats()
    with pytest.raises(U.UniStop):
        T.conditioning_report(g[:59], g[:59], NAMES)
    s = g.copy()
    s[20, 3, 4] = np.nan
    with pytest.raises(U.UniStop) as e:
        T.conditioning_report(s, g, NAMES)
    assert e.value.code == "RIG_PROJECTION_FAILED"


def test_hybrid_features_only_arm_slots():
    g, s = _feats(0), _feats(1)
    h = T.hybrid_features(g, s, NAMES, PARENTS)
    changed = {NAMES[j] for j in range(len(NAMES)) if not np.array_equal(h[:, j], g[:, j])}
    assert changed == {"lowerarm_l", "hand_l"}                              # slots of bvh(upperarm_l), bvh(lowerarm_l)
    assert np.array_equal(h[:, NAMES.index("hand_l"), 3:9], s[:, NAMES.index("hand_l"), 3:9])
    assert np.array_equal(h[:, NAMES.index("hand_l"), [0, 1, 2, 9, 10, 11]], g[:, NAMES.index("hand_l"), [0, 1, 2, 9, 10, 11]])


def test_degenerate_6d_refused():
    with pytest.raises(U.UniStop):
        U.sixd_to_mat(np.zeros(6))
    with pytest.raises(U.UniStop):
        U.sixd_to_mat(np.array([1, 0, 0, 2, 0, 0.0]))


# ------------------------------------------------------------------------------------------------ world map ---
def test_similarity_fit_recovers_and_refuses_reflection():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(200, 3))
    R = _rot([1, 2, 3], 40)
    Y = 0.37 * X @ R.T + np.array([1.0, -2.0, 0.5])
    s, R2, t, rms = T.similarity_fit(X, Y)
    assert abs(s - 0.37) < 1e-9 and np.allclose(R2, R) and rms < 1e-9
    Xm = X.copy()
    Xm[:, 0] *= -1                                                           # mirrored (handedness) input
    assert T.similarity_fit(Xm, Y)[3] > 0.05


def _readback(pos_by_frame, bones):
    return {"frames": [{"blender_frame": f, "world": {b: {"head": list(map(float, p[k]))} for k, b in enumerate(bones)}}
                       for f, p in pos_by_frame.items()]}


def test_fit_world_map_time_and_joint_identity():
    rng = np.random.default_rng(4)
    bones = ["spine_01", "hand_l", "hand_r", "foot_l"]
    names = ["pelvis"] + bones
    model = np.cumsum(rng.normal(scale=0.05, size=(60, len(names), 3)), axis=0)
    R = _rot([0, 1, 0], 90)
    world = {156 + i: 0.5 * model[i, 1:] @ R.T + 0.2 for i in range(60)}
    rb = _readback(world, bones)
    wm = T.fit_world_map(model, names, rb, bones)
    assert wm["rms_m"] < 1e-9 and abs(wm["s"] - 0.5) < 1e-9
    with pytest.raises(U.UniStop):
        T.fit_world_map(np.roll(model, 1, axis=0), names, rb, bones)
    sw = list(names)
    sw[2], sw[3] = sw[3], sw[2]
    with pytest.raises(U.UniStop):
        T.fit_world_map(model, sw, rb, bones)


# ------------------------------------------------------------------------------------------------ continuity / attribution ---
def _arm_world(names, lift):
    """(60, J, 3) synthetic world positions: shoulder fixed, elbow/wrist rise linearly by ``lift`` metres over the window."""
    P = np.zeros((60, len(names), 3))
    for i in range(60):
        P[i, names.index("upperarm_l")] = [0.2, 0.0, 1.4]
        P[i, names.index("lowerarm_l")] = [0.2, 0.05, 1.1 + lift * i / 59]
        P[i, names.index("hand_l")] = [0.2, 0.1, 1.0 + 2 * lift * i / 59]
    return P


def test_continuity_steps_and_flicks():
    pts = {f: np.array([0.0, 0.0, 0.01 * (f - 171)]) for f in range(156, 216)}
    c = T.continuity(pts)
    assert len(c["steps_mm"]) == 29 and abs(c["entry_mm"] - 10) < 1e-9 and abs(c["peak_mm"] - 10) < 1e-9 and c["n_flicks"] == 0
    pts[185] = pts[185] + np.array([0.0, 0.1, 0.0])                          # out-and-back spike -> one reversal flick
    c = T.continuity(pts)
    assert c["n_flicks"] == 1 and c["flick_at_action"] == [185] and c["peak_mm"] > 100


def test_attribution_identical_views_and_parent_shift():
    names = NAMES
    gt = _arm_world(names, 0.3)
    raw = gt.copy()
    hyb = gt.copy()
    att = T.attribution(raw, gt, hyb, names)
    assert att["raw_vs_projected"]["wrist"]["raw_minus_projected_max_mm"] == 0.0
    assert abs(att["raw_excess_step_over_proxyB"]["wrist"]["max_excess_mm"]) < 1e-9
    # raw parent (shoulder) translated by 50 mm in the interior; arm-local identical -> pure shoulder-translation part
    raw2 = gt.copy()
    for i in T.FREE_FRAMES:
        raw2[i, [names.index("upperarm_l"), names.index("lowerarm_l"), names.index("hand_l")]] += [0.05, 0, 0]
    att2 = T.attribution(raw2, gt, hyb, names)
    w = att2["raw_vs_projected"]["wrist"]
    assert abs(w["raw_minus_projected_max_mm"] - 50) < 1e-6 and abs(w["shoulder_translation_part_max_mm"] - 50) < 1e-6
    assert w["parent_orientation_part_max_mm"] < 1e-6
    st = 2 * 0.3 / 59 * 1e3                                                  # PROXY_B wrist step (mm)
    assert att2["raw_excess_step_over_proxyB"]["wrist"]["max_excess_mm"] == pytest.approx(np.hypot(50, st) - st, abs=1e-6)


def test_raw_rule_and_body_report():
    names = NAMES
    gt = _arm_world(names, 0.3)
    raw = gt.copy()
    body = T.body_report(raw, gt, names)
    att = T.attribution(raw, gt, gt, names)
    assert not T.raw_unusable(att, body)["unusable"]
    raw[16:44, names.index("thigh_r")] += [0.0, 0.0, 0.15]                    # a 150 mm leg pop at entry and exit
    body = T.body_report(raw, gt, names)
    r = T.raw_unusable(att, body)
    assert r["unusable"] and any("legs" in x for x in r["reasons"])
    assert body["legs"]["interior_dev_max_mm"] == pytest.approx(150.0)


# ------------------------------------------------------------------------------------------------ authority / pin ---
def test_unauthorised_edits_and_pin_overlap_refused():
    assert U.check_domain(range(172, 200))
    for kw, code in ((dict(frames=range(171, 200)), "DOMAIN_FRAMES"), (dict(frames=range(172, 201)), "DOMAIN_FRAMES"),
                     (dict(frames=range(172, 200), channels=("spine_01",)), "DOMAIN_CHANNEL"),
                     (dict(frames=range(172, 200), channels=("clavicle_l",)), "DOMAIN_CHANNEL"),
                     (dict(frames=range(130, 158)), "PIN_OVERLAP")):
        with pytest.raises(U.UniStop) as e:
            U.check_domain(**kw)
        assert e.value.code == code
    assert not any(U.WINDOW[0] <= f <= U.WINDOW[1] for f in U.PIN_ACTION)
