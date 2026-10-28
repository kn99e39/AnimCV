"""Worklog 99: HMR 2.0b predicted SMPL hip roll vs zero-twist leg IK (gates, adapter R, isolation)."""

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest

from motion.torso_frame import quaternion_to_matrix

ROOT = Path(__file__).resolve().parent.parent
E2E = ROOT / "examples/e2e_demo"


def _script(name):
    sys.path.insert(0, str(ROOT / "scripts"))
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def m():
    return _script("run_learned_rotation_vs_geometry_ik")


@pytest.fixture(scope="module")
def w():
    return _script("run_leg_knee_hinge_twist"), _script("run_pelvis_frame_leg_articulation"), _script("run_pelvis_torso_ownership")


@pytest.fixture(scope="module")
def ctx(w):
    w96, w95, _ = w
    from retarget.framepose_fk import derive_rig_alignment, load_rig_fk_calibration
    from retarget.framepose_target_rest import load_target_rest_pose

    rest = load_target_rest_pose(E2E / "baserig_blender_rest.json")
    a = derive_rig_alignment(rest, load_rig_fk_calibration(E2E / "framepose_fk_calibration.json"))
    c = {"alignment": a, "Am": quaternion_to_matrix(a), "rest": rest, "legs": w95.leg_targets(rest), "lf": w96.leg_frames(rest)}
    c["psi0"] = w96.rest_offsets(c["lf"], c["Am"])
    return c


C = np.array([[1.0, 0, 0], [0, 0, 1.0], [0, -1.0, 0]])


def _ry(deg):
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _rx(deg):
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _row(m, w, ctx, thigh, shin, hip_l=np.eye(3), hip_r=np.eye(3), smpl=None):
    """Synthetic row: Object = identity body frame, HMR global orient chosen so its pelvis maps onto that frame."""
    w96, w95, w94 = w
    j = w95.fixture_skeleton(np.eye(3), np.eye(3), thigh, shin)
    ev = w95.evaluate(j, np.eye(3), np.eye(3), ctx)
    go = C.T @ np.eye(3) @ w94.SMPL_BODY.T  # C . go . SMPL_BODY = I
    bp = np.tile(np.eye(3), (23, 1, 1))
    bp[0], bp[1] = hip_l, hip_r
    r = {"H": {"T": {"ev": ev, "Ro": np.eye(3)}}, "E": {"legs": m.hmr_leg_frames(go, bp, C)}}
    return r, m.leg_conditions(r, ctx, smpl, w96)


def test_frozen_identities_and_predeclared(m):
    assert m.W97_PREDICTIONS_SHA == "fc5d330565205f4792cd3641671633decbab4e24d70522d4dd85f045d87a7003"
    text = (ROOT / "scripts/run_learned_rotation_vs_geometry_ik.py").read_text()
    assert 'raise SystemExit(f"STOP: frozen input changed' in text
    R = m.PREDECLARED["evaluation"]["rules"]
    assert set(R) == {"R1_deep_roll", "R2_low_flex_roll", "R3_deep_off_hinge", "R4_low_flex_off_hinge", "R5_temporal", "R6_hip_axial", "blender"}
    assert m.SMPL_SIDES == {"l": (1, 4, 7), "r": (2, 5, 8)}


def test_smpl_hierarchy_and_canonical_hinge_axis(m):
    go = np.diag([1.0, -1.0, -1.0])  # upright, facing the camera (OpenCV)
    bp = np.tile(np.eye(3), (23, 1, 1))
    L = m.hmr_leg_frames(go, bp, C)
    for s in "lr":
        assert np.allclose(L[s]["axis"], (1, 0, 0))  # template +x (subject's left) = camera right for a facing subject
        assert np.linalg.det(L[s]["G"]) == pytest.approx(1.0)
    bp[0] = _rx(40)  # left hip flexion about the hinge axis leaves that axis unchanged
    assert np.allclose(m.hmr_leg_frames(go, bp, C)["l"]["axis"], (1, 0, 0))
    assert np.allclose(m.hmr_leg_frames(go, bp, C)["l"]["knee_local"], np.eye(3))  # body_pose[3] = left knee (joint 4)


def test_R_equals_oracle_construction_and_tracks_axis_rotation(m, w, ctx):
    w96 = w[0]
    thigh, shin = np.array([0, 0.5, -0.87]), np.array([0, -0.5, -0.87])
    r, out = _row(m, w, ctx, thigh, shin, hip_l=_ry(25), hip_r=_ry(-25))
    sm = {s: {"lateral": r["E"]["legs"][s]["axis"], "knee_axis": None, "knee_angle": 0.0} for s in "lr"}
    _, out2 = _row(m, w, ctx, thigh, shin, hip_l=_ry(25), hip_r=_ry(-25), smpl=sm)
    for s in "lr":
        R, O = out2["R"][s], out2["O"][s]
        assert R["roll"] == pytest.approx(O["roll"], abs=1e-9) and np.allclose(R["q_p"], O["q_p"], atol=1e-12)
        assert abs(R["smpl_roll_err"]) < 1e-6  # with the reference equal to the HMR axis, R has zero roll error
    _, base = _row(m, w, ctx, thigh, shin)
    d = {s: m.wrap(out["R"][s]["roll"] - base["R"][s]["roll"]) for s in "lr"}
    # a 25 deg rotation about the template vertical moves the hinge about the 30-deg-flexed thigh by ~25 cos(30) deg
    assert d["l"] * d["r"] < 0 and all(20.0 < abs(v) < 25.5 for v in d.values())


def test_world_geometry_identical_across_Z_R_and_round_trip(m, w, ctx):
    for thigh, shin in (((0, 0.5, -0.87), (0, -0.5, -0.87)), ((0.2, 0.9, -0.3), (0.1, -0.2, -0.97)), ((0, 0, -1.0), (0, 0, -1.0))):
        _, out = _row(m, w, ctx, np.array(thigh), np.array(shin), hip_l=_ry(30) @ _rx(10), hip_r=_ry(-15))
        for s in "lr":
            for k in ("Z", "R"):
                x = out[k][s]
                assert x["fk_parity"] < 1e-9 and x["thigh_world_dev"] < 1e-6 and x["shin_world_dev"] < 1e-6
                assert sum(c * c for c in x["q_p"]) == pytest.approx(1.0, abs=1e-12)
            assert out["Z"][s]["roll"] == 0.0


def test_straight_leg_roll_defined_and_mirror_parity(m, w, ctx):
    _, base = _row(m, w, ctx, np.array([0, 0, -1.0]), np.array([0, 0, -1.0]))
    for s in "lr":
        assert base["R"][s] is not None and abs(base["R"][s]["roll"]) < 3.0  # identity SMPL hips ~ rig rest hinge (psi0)
    _, rot = _row(m, w, ctx, np.array([0, 0, -1.0]), np.array([0, 0, -1.0]), hip_l=_ry(20), hip_r=_ry(-20))
    dl = m.wrap(rot["R"]["l"]["roll"] - base["R"]["l"]["roll"])
    dr = m.wrap(rot["R"]["r"]["roll"] - base["R"]["r"]["roll"])
    assert dl == pytest.approx(-dr, abs=1.0) and abs(dl) == pytest.approx(20.0, abs=2.0)  # mirrored hip rotations, mirrored rolls


def test_knee_components_and_degeneracy(m, w):
    w96 = w[0]
    k = m.knee_components(_rx(35), w96.rot_axis, w96.twist_deg)
    assert k["hinge_twist"] == pytest.approx(35.0, abs=1e-6) and k["off_hinge_swing"] == pytest.approx(0.0, abs=1e-6)
    k = m.knee_components(_ry(8) @ _rx(35), w96.rot_axis, w96.twist_deg)
    assert k["off_hinge_swing"] > 5.0
    phi, perp = m.r_phi((0, 0, -1.0), (1, 0, 0), np.eye(3), np.eye(3), (0, 0, 1.0), 0.0)
    assert phi is None and perp < 0.1  # hinge axis along the thigh -> unavailable, never substituted


def test_fk_positions_chain(m):
    J = np.zeros((9, 3))
    J[1], J[4], J[7] = (0.1, -0.1, 0), (0.1, -0.5, 0), (0.1, -0.9, 0)
    parents = [-1, 0, 0, 0, 1, 2, 3, 4, 5]
    bp = np.tile(np.eye(3), (23, 1, 1))
    bp[3] = _rx(90)  # left knee 90 deg
    P, _ = m.fk_positions(np.eye(3), bp, J, parents)
    assert np.allclose(P[4], J[4]) and np.allclose(P[7] - P[4], _rx(90) @ (J[7] - J[4]))


def test_gt_isolation_of_candidate_path(m):
    text = (ROOT / "scripts/run_learned_rotation_vs_geometry_ik.py").read_text()
    for fn, nxt in (("def hmr_leg_frames", "def r_phi"), ("def r_phi", "def knee_components")):
        body = text.split(fn, 1)[1].split(nxt, 1)[0].lower().replace("smpl_sides", "")
        for bad in ("smpl_side", "lateral", "r_smpl", "gt_", "cam_poses", "smpl_legs", "joints3"):
            assert bad not in body, (fn, bad)
    lc = text.split("def leg_conditions", 1)[1].split("def flex_bin", 1)[0]
    assert 'phi, perp = r_phi(pose.upper, dp0 @ f["n_rest"], Ro, Am, hl["axis"], ctx["psi0"][s])' in lc  # R reads HMR axis only
    assert 'c, e = r["H"].get("T"), r.get("E")' in lc  # fixed W98 T body frame
    for path in sorted((ROOT / "src").rglob("*.py")):
        assert "run_learned_rotation_vs_geometry_ik" not in path.read_text(), path


def test_temporal_accounting(m, w):
    _, w95, _ = w
    from retarget.framepose_two_bone_ik import quaternion_angle_degrees

    def leg(roll, q):
        return {"roll": roll, "q_p": q, "smpl_roll_err": roll}
    q0, q1 = (0.0, 0.0, 0.0, 1.0), (0.0, 0.0, math.sin(math.radians(60)), math.cos(math.radians(60)))  # 120 deg apart
    gl = {"l": [np.array([0, 0, -1.0])], "r": [np.array([0, 0, -1.0])]}
    a = {"gt_legs": gl, "W99": {"Z": {"l": leg(0, q0), "r": leg(0, q0)}, "R": {"l": leg(10, q0), "r": leg(10, q0)}}}
    b = {"gt_legs": gl, "W99": {"Z": {"l": leg(0, q0), "r": leg(0, q0)}, "R": {"l": leg(130, q1), "r": leg(10, q0)}}}
    t = m.temporal({"s": [a, b]}, quaternion_angle_degrees, w95.angle, ("Z", "R"))
    assert t["Z"]["severe_roll_steps"] == 0 and t["R"]["severe_roll_steps"] == 1 and t["R"]["severe_thigh_jumps"] == 1
    assert t["R"]["roll_jitter_mean"] == pytest.approx(60.0)
