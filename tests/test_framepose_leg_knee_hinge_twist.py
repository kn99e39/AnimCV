"""Worklog 96: knee-hinge / thigh-roll diagnostics and the opt-in hinge-aware candidate (unwired)."""

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
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def mods():
    return _script("run_leg_knee_hinge_twist"), _script("run_pelvis_frame_leg_articulation")


@pytest.fixture(scope="module")
def ctx(mods):
    m, w95 = mods
    from retarget.framepose_fk import derive_rig_alignment, load_rig_fk_calibration
    from retarget.framepose_target_rest import load_target_rest_pose

    rest = load_target_rest_pose(E2E / "baserig_blender_rest.json")
    a = derive_rig_alignment(rest, load_rig_fk_calibration(E2E / "framepose_fk_calibration.json"))
    c = {"alignment": a, "Am": quaternion_to_matrix(a), "rest": rest, "legs": w95.leg_targets(rest), "lf": m.leg_frames(rest)}
    c["psi0"] = m.rest_offsets(c["lf"], c["Am"])
    return c


def _leg(mods, ctx, thigh, shin, Ro=np.eye(3)):
    m, w95 = mods
    j = w95.fixture_skeleton(np.eye(3), np.eye(3), thigh, shin)
    ev = w95.evaluate(j, Ro, np.eye(3), ctx)
    return ev, m.process(None, ev, Ro, ctx, None, ("Z", "T"))


def test_rest_frames_conventions_and_mirroring(mods, ctx):
    m, _ = mods
    for s in ("l", "r"):
        f = ctx["lf"][s]
        for B in (f["Bt"], f["Bc"]):
            assert np.allclose(B.T @ B, np.eye(3), atol=1e-9) and np.linalg.det(B) == pytest.approx(1.0)
        assert np.allclose(f["Bt"][:, 1], f["t0"], atol=1e-6)  # bone local Y = bone direction
        assert abs(f["n_rest"][0]) > 0.98  # rest hinge = lateral armature axis
        assert 5 < math.degrees(math.acos(abs(float(np.dot(f["n_rest"], f["n_bone_z"]))))) < 9  # declared convention gap
    assert ctx["lf"]["l"]["n_rest"][0] * ctx["lf"]["r"]["n_rest"][0] > 0  # same lateral sense for both knees
    assert ctx["psi0"]["l"] == pytest.approx(-ctx["psi0"]["r"], abs=0.5)  # mirrored rest offsets


def test_zero_twist_round_trip_and_candidate_preserves_directions(mods, ctx):
    m, _ = mods
    out_rot = m.rot_axis((0, 0, 1), 30)
    ev, res = _leg(mods, ctx, out_rot @ np.array([0, 0.5, -0.87]), out_rot @ np.array([0, -0.5, -0.87]))
    for s in ("l", "r"):
        z, t = res[s]["Z"], res[s]["T"]
        assert z["fk_parity"] < 1e-9 and t["fk_parity"] < 1e-9  # parent/local round trip, same mid/end (directions, lengths)
        for q in (t["q_p"], t["q_d"]):
            assert sum(c * c for c in q) == pytest.approx(1.0, abs=1e-12)
        assert abs(z["off_hinge"]) > 15 and abs(t["off_hinge"]) < 1e-6  # zero twist bends the knee off its hinge; T does not
        assert t["hinge_w95"] < 1e-6 and abs(t["calf_twist"]) < 1e-6 and t["weight"] == 1.0
        assert abs(t["roll"]) <= 90.0


def test_straight_leg_degeneracy_and_backward_knee(mods, ctx):
    _, res = _leg(mods, ctx, (0, 0, -1.0), (0, 0, -1.0))
    for s in ("l", "r"):
        assert res[s]["T"]["roll"] == 0.0 and res[s]["T"]["weight"] == 0.0  # no plane -> zero twist
    _, res = _leg(mods, ctx, (0, -0.4, -0.92), (0, 0.4, -0.92))  # observed bend opposite to the rig hinge
    for s in ("l", "r"):
        t = res[s]["T"]
        assert t["sign"] == -1 and abs(t["roll"]) < 15 and t["hinge_w95"] > 170  # kept as hyperextension, no 180-degree roll


def test_partial_weight_between_5_and_20_deg(mods, ctx):
    m, _ = mods
    thigh = m.rot_axis((0, 0, 1), 30) @ m.rot_axis((1, 0, 0), -6) @ np.array([0, 0, -1.0])
    shin = m.rot_axis((0, 0, 1), 30) @ m.rot_axis((1, 0, 0), 6) @ np.array([0, 0, -1.0])
    _, res = _leg(mods, ctx, thigh, shin)
    t = res["l"]["T"]
    assert 5 < t["flex"] < 20 and 0 < t["weight"] < 1
    assert t["roll"] == pytest.approx(t["weight"] * t["raw_phi"])


def test_crouch_supine_prone_and_root_invariance(mods, ctx):
    m, _ = mods
    cases = {"crouch": ((0, 0.95, -0.3), (0, -0.45, -0.9)), "supine_knees_up": ((0, 0.3, 0.95), (0, 0.6, -0.8)),
             "prone_shins_up": ((0, -1.0, 0.0), (0, -0.3, 0.95))}
    for name, (th, sh) in cases.items():
        _, a = _leg(mods, ctx, th, sh)
        _, b = _leg(mods, ctx, th, sh, Ro=m.rot_axis((1, 1, 0), 40))  # a different parent Object
        for s in ("l", "r"):
            assert abs(a[s]["T"]["off_hinge"]) < 1e-6, name
            assert a[s]["T"]["flex"] == pytest.approx(b[s]["T"]["flex"], abs=1e-9)
            # with a full-weight candidate the world hinge follows the observed plane, whatever the Object frame
            assert b[s]["T"]["hinge_w95"] == pytest.approx(a[s]["T"]["hinge_w95"], abs=1e-6)


def test_smpl_oracle_nulls_the_reference_roll(mods, ctx):
    m, w95 = mods
    j = w95.fixture_skeleton(np.eye(3), np.eye(3), (0, 0.5, -0.87), (0, -0.5, -0.87))
    ev = w95.evaluate(j, np.eye(3), np.eye(3), ctx)
    lat = {s: m.rot_axis(ev["legs"][s]["thigh_w"], 25) @ np.array([-1.0, 0.0, 0.0]) for s in ("l", "r")}
    smpl = {s: {"lateral": lat[s], "knee_axis": None, "knee_angle": 0.0} for s in ("l", "r")}
    res = m.process(None, ev, np.eye(3), ctx, smpl, ("Z", "O"))
    for s in ("l", "r"):
        assert abs(res[s]["O"]["smpl_roll_err"]) < 1e-6


def test_predeclared_constants_gt_isolation_and_frozen_inputs(mods):
    m, _ = mods
    P = m.PREDECLARED
    assert P["candidate_T"]["degeneracy_weight"] == {"zero_below_flexion_deg": 5.0, "full_at_flexion_deg": 20.0}
    assert P["evaluation"]["hip_axial_rotation_max_deg"] == 45.0 and P["evaluation"]["knee_off_hinge_max_deg"] == 10.0
    text = (ROOT / "scripts/run_leg_knee_hinge_twist.py").read_text()
    body = text.split("def candidate_phi", 1)[1].split("class SmplLegs", 1)[0]
    assert "smpl" not in body.lower() and "gt_legs" not in body  # the candidate uses only the posed chain and rig rest
    assert 'raise SystemExit(f"STOP: frozen input changed' in text and m.S0_SHA["validation"].startswith("402d70a2")
    for path in sorted((ROOT / "src").rglob("*.py")):
        assert "run_leg_knee_hinge_twist" not in path.read_text(), path
