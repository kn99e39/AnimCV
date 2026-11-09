"""Focused tests for the EXPERIMENTAL (Batch 131) CondMDI adapter.

Evidence kinds (kept separate from real-model / real-Blender evidence, which the harness produces):
  * the REAL BaseRig rest snapshot ``examples/e2e_demo/baserig_blender_rest.json`` with SYNTHETIC poses through the numpy FK;
  * pure numpy maths (time map, mask contract, HML263 encode/decode, projection, 20->30 reconstruction).
No model is loaded here.
"""
from __future__ import annotations

import json
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import app.experimental_condmdi as C  # noqa: E402
from app.experimental_motionbricks import AdapterStop, canon_sign  # noqa: E402

REST_JSON = ROOT / "examples" / "e2e_demo" / "baserig_blender_rest.json"


def _q(axis, deg):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    h = np.radians(deg) / 2
    return np.array([np.cos(h), *(np.sin(h) * axis)])


@pytest.fixture(scope="module")
def rig():
    r = json.loads(REST_JSON.read_text())
    rest = {n: {"parent": b["parent"], "matrix_local": b["matrix_local"], "head_local": b["head"], "tail_local": b["tail"]}
            for n, b in r["bones"].items()}
    return C.Rig(rest)


def _stored(n=240, arm_fn=None):
    """Synthetic keyed Action: identity everywhere except an optional left-arm schedule."""
    st = {ch: {str(c): {str(f): (1.0 if c == 0 else 0.0) for f in range(1, n + 1)} for c in range(4)} for ch in C.KEYED}
    if arm_fn:
        for f in range(1, n + 1):
            for ch, q in arm_fn(f).items():
                for c in range(4):
                    st[ch][str(c)][str(f)] = float(q[c])
    return st


def test_time_map_is_duration_preserving_and_half_frame():
    tm = C.time_map()
    assert tm["n_model"] == 160 and tm["duration_s"] == Fraction(8)
    assert C.validate_time_map(tm)
    assert tm["u"][0] == 1 and tm["u"][1] == Fraction(5, 2) and tm["u"][2] == 4
    assert tm["kind"].count("EXACT") == 80 and tm["kind"].count("INTERPOLATED_HALF_FRAME") == 80
    k = C.interior_samples(tm)
    assert k == list(range(114, 133))
    assert tm["u"][113] == Fraction(341, 2) and tm["u"][133] == Fraction(401, 2)     # 170.5 / 200.5: outside 172..199
    assert tm["s"][171] == 114 and tm["s"][198] == 132                                # Action 172 / 199 are exact samples


def test_time_map_refusals():
    with pytest.raises(AdapterStop) as e:
        C.time_map(n_action=239)
    assert e.value.code == "DURATION_NOT_REPRESENTABLE"
    tm = C.time_map()
    bad = dict(tm, u=list(tm["u"]))
    bad["u"][5], bad["u"][6] = bad["u"][6], bad["u"][5]
    with pytest.raises(AdapterStop) as e:
        C.validate_time_map(bad)
    assert e.value.code == "TIME_MAP_NON_MONOTONE"
    with pytest.raises(AdapterStop) as e:
        C.validate_time_map(dict(tm, u=list(tm["u"])[:-1]))
    assert e.value.code == "TIME_MAP_DURATION_CHANGED"


def test_observation_mask_contract():
    tm = C.time_map()
    k = C.interior_samples(tm)
    m = C.observation_mask(160, k)
    assert C.validate_mask(m, k)
    assert not m[67:].any() and not m[1:3].any() and m[0].all() and m[3].all()
    for j in (10, 11, 15, 19, 21):
        assert not m[4 + 3 * (j - 1): 4 + 3 * j].any()
    lw = 4 + 3 * 19
    assert not m[lw, k].any() and m[lw, :114].all() and m[lw, 133:].all()
    for f, code in ((4 + 3 * 18, "MASK_OBSERVES_UNKNOWN"), (100, "MASK_OBSERVES_UNKNOWN"), (200, "MASK_OBSERVES_UNKNOWN"),
                    (260, "MASK_OBSERVES_UNKNOWN"), (1, "MASK_OBSERVES_UNKNOWN")):
        bad = m.copy()
        bad[f, :] = True
        with pytest.raises(AdapterStop) as e:
            C.validate_mask(bad, k)
        assert e.value.code == code
    bad = m.copy()
    bad[lw, 120] = True
    with pytest.raises(AdapterStop) as e:
        C.validate_mask(bad, k)
    assert e.value.code == "MASK_OBSERVES_EDIT_DOMAIN"
    bad = m.copy()
    bad[4 + 3 * 17: 4 + 3 * 18, :114] = False
    with pytest.raises(AdapterStop) as e:
        C.validate_mask(bad, k)
    assert e.value.code == "MASK_NO_SOURCE_ARM_EVIDENCE"


def test_axes_and_handedness(rig):
    assert C.check_axes(C.A_BLENDER_TO_HML, rig)
    # a proper up-preserving rotation is only a yaw (removed by canonicalisation); handedness can flip only by mirroring
    assert C.check_axes(np.array([[-1.0, 0, 0], [0, 0, 1], [0, 1, 0]]), rig)
    for A, code in ((np.diag([-1.0, 1, 1]) @ C.A_BLENDER_TO_HML, "AXIS_MAPPING_IMPROPER"), (np.eye(3), "AXIS_UP_MISMATCH")):
        with pytest.raises(AdapterStop) as e:
            C.check_axes(A, rig)
        assert e.value.code == code
    J = rig.hml_joints({}, (1, 0, 0, 0))[None]
    assert C.check_left_right(J)
    assert J[0, 16, 0] > J[0, 17, 0] and J[0, 1, 0] > J[0, 2, 0]                    # rig left = HML +X
    sw = J.copy()
    sw[:, [18, 19]] = sw[:, [19, 18]]
    with pytest.raises(AdapterStop) as e:
        C.check_left_right(sw)
    assert e.value.code == "LEFT_RIGHT_MISMATCH"


def test_missing_bone_refused(rig):
    rest = {k: v for k, v in rig.rest.items() if k != "upperarm_l"}
    with pytest.raises(AdapterStop) as e:
        C.Rig(rest)
    assert e.value.code == "BASERIG_MISSING_BONES"


def test_encode_invariances_and_unknown_isolation(rig):
    base = rig.hml_joints({}, (1, 0, 0, 0))
    yaw = rig.hml_joints({}, _q([0, 0, 1], 35))
    e0 = C.encode(np.stack([base, base]), "supported", 0.05)["relative"]
    e1 = C.encode(np.stack([yaw, yaw]), "supported", 0.05)["relative"]
    assert np.abs(e0[:, 4:67] - e1[:, 4:67]).max() < 1e-9                           # yaw invariant (canonical frame)
    arm = rig.hml_joints({"upperarm_l": _q([1, 0, 0], -50)}, (1, 0, 0, 0))   # swing; local Y would be pure twist
    # compare the SAME frame of two sequences (frame 0's root rotation is forced to identity by the official recipe)
    r0 = C.encode(np.stack([base, base]), "supported", 0.05)["relative"][1, 4:67].reshape(21, 3)
    ra = C.encode(np.stack([base, arm]), "supported", 0.05)["relative"][1, 4:67].reshape(21, 3)
    moved = {C.HML_JOINTS[j + 1] for j in np.nonzero(np.linalg.norm(ra - r0, axis=1) > 1e-6)[0]}
    assert moved == {"left_elbow", "left_wrist"}
    unk = rig.hml_joints({"upperarm_r": _q([1, 0, 0], 40), "foot_l": _q([0, 1, 0], 30), "neck_02": _q([1, 0, 0], 25),
                          "hand_l": _q([0, 0, 1], 30)}, (1, 0, 0, 0))
    a = C.rel_to_abs(C.encode(np.stack([base, base]), "supported", 0.05)["relative"])
    b = C.rel_to_abs(C.encode(np.stack([unk, unk]), "supported", 0.05)["relative"])
    m = C.observation_mask(2, [])
    assert np.array_equal(a[m.T], b[m.T]) and not np.array_equal(a, b)            # UNKNOWN moves masked features only


def test_encode_decode_round_trip_and_projection(rig):
    rng = np.random.default_rng(0)
    seq = [rig.hml_joints({"spine_01": _q([1, 0, 0], 10 * np.sin(t)), "upperarm_l": _q([0, 1, 0], -40 - 30 * t),
                           "lowerarm_l": _q([1, 0, 0], 20 + 10 * t)}, _q([0, 0, 1], 15 * t)) for t in np.linspace(0, 1, 30)]
    J = np.stack(seq)
    enc = C.encode(J, "supported", 0.05)
    ab = C.rel_to_abs(enc["relative"])
    Jd = C.abs_to_joints(ab)
    obs = list(C.OBSERVED_JOINTS)
    assert np.abs(Jd[:, obs] - enc["canonical"][:, obs]).max() < 1e-9
    U, V = C.arm_dirs_from_abs(ab, enc["r0"], list(range(len(J))))
    want = C.normalize(J[:, 18] - J[:, 16]) @ C.A_BLENDER_TO_HML                    # HML world -> Blender world
    assert np.abs(U - want).max() < 1e-9
    basis = {"spine_01": _q([1, 0, 0], 5)}
    obj = _q([0, 0, 1], 20)
    ref_u, ref_l = _q([0, 1, 0], -30), _q([1, 0, 0], 30)
    tgt_u, tgt_l = _q([0, 1, 0], -70), _q([1, 0, 0], 60)
    Mw, P = rig.pose({**basis, "upperarm_l": tgt_u, "lowerarm_l": tgt_l}, obj)
    u = rig.point(Mw, P, "H:lowerarm_l") - rig.point(Mw, P, "H:upperarm_l")
    v = rig.point(Mw, P, "H:hand_l") - rig.point(Mw, P, "H:lowerarm_l")
    bu, bl = C.project_arm(rig, basis, obj, ref_u, ref_l, u, v)
    M2, P2 = rig.pose({**basis, "upperarm_l": bu, "lowerarm_l": bl}, obj)
    u2 = rig.point(M2, P2, "H:lowerarm_l") - rig.point(M2, P2, "H:upperarm_l")
    v2 = rig.point(M2, P2, "H:hand_l") - rig.point(M2, P2, "H:lowerarm_l")
    # float32 Blender rest matrices are re-orthonormalised in rot_part: residual ~1e-7
    assert np.abs(C.normalize(u2) - C.normalize(u)).max() < 1e-6 and np.abs(C.normalize(v2) - C.normalize(v)).max() < 1e-6
    assert abs(np.linalg.norm(u2) - np.linalg.norm(u)) < 1e-12                      # rotation only: limb length unchanged
    _ = rng


def test_reconstruct_30_exact_and_slerp():
    per = {k: _q([0, 0, 1], 3.0 * k) for k in range(114, 133)}
    out = C.reconstruct_30(per, range(172, 200))
    assert np.allclose(out[172], per[114]) and np.allclose(out[199], per[132])
    assert np.allclose(canon_sign(out[173], per[114]), _q([0, 0, 1], 3.0 * (114 + 2 / 3)))
    with pytest.raises(AdapterStop) as e:
        C.reconstruct_30({114: per[114]}, [173])
    assert e.value.code == "RECONSTRUCTION_SAMPLE_MISSING"


def test_pose_at_half_frame_is_slerp_and_override():
    st = _stored(arm_fn=lambda f: {"upperarm_l": _q([0, 1, 0], f)})
    b, o = C.pose_at(st, 10.5)
    assert np.allclose(b["upperarm_l"], _q([0, 1, 0], 10.5)) and np.allclose(o, [1, 0, 0, 0])
    fn = C.b_slerp_fn(st, "upperarm_l")
    assert np.allclose(fn(171), _q([0, 1, 0], 171)) and np.allclose(fn(200), _q([0, 1, 0], 200))
    b2, _ = C.pose_at(st, 180.5, {"upperarm_l": fn})
    assert np.allclose(b2["upperarm_l"], _q([0, 1, 0], 180.5))


def test_protected_overlap():
    with pytest.raises(AdapterStop) as e:
        C.protected_check((130, 160), [132, 140])
    assert e.value.code == "PROTECTED_OVERLAP"
    assert C.protected_check(C.INTERIOR_ACTION, list(range(132, 141))) is None


def test_default_paths_do_not_reference_slice():
    for f in ("src/app/cli.py", "src/app/gui.py"):
        assert "experimental_condmdi" not in (ROOT / f).read_text(encoding="utf-8")
