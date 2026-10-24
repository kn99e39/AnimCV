"""Worklog 95: experimental articulated-leg path and root-frame factorization (unwired)."""

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest

from motion.torso_frame import quaternion_to_matrix, torso_frames, yaw_frame

ROOT = Path(__file__).resolve().parent.parent
E2E = ROOT / "examples/e2e_demo"


def _script(name):
    sys.path.insert(0, str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def leg():
    return _script("run_pelvis_frame_leg_articulation")


@pytest.fixture(scope="module")
def ctx(leg):
    from retarget.framepose_fk import derive_rig_alignment, load_rig_fk_calibration
    from retarget.framepose_target_rest import load_target_rest_pose

    rest = load_target_rest_pose(E2E / "baserig_blender_rest.json")
    a = derive_rig_alignment(rest, load_rig_fk_calibration(E2E / "framepose_fk_calibration.json"))
    return {"alignment": a, "Am": quaternion_to_matrix(a), "rest": rest, "legs": leg.leg_targets(rest)}


def _rot(axis, angle):
    a = np.asarray(axis, float) / np.linalg.norm(axis)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(angle) * K + (1 - math.cos(angle)) * K @ K


def test_baserig_leg_hierarchy_and_chain_contract(ctx):
    rest = ctx["rest"]
    for s in ("l", "r"):
        assert rest.bones[f"thigh_{s}"].parent is None and rest.bones[f"calf_{s}"].parent == f"thigh_{s}"
        assert rest.bones[f"foot_{s}"].parent == f"calf_{s}"
        t = ctx["legs"][s]["target"]
        assert t.proximal == f"thigh_{s}" and t.distal == f"calf_{s}"
        assert 43 < t.proximal_length < 44 and 42 < t.distal_length < 43
        assert abs(ctx["legs"][s]["hinge"][0]) > 0.98  # rest knee hinge = lateral armature axis
        assert math.dist(rest.bones[f"calf_{s}"].tail, rest.bones[f"foot_{s}"].head) > 1e-6  # why no end anchor is used


def test_leg_ik_fk_parity_and_world_parity_across_object_frames(leg, ctx):
    from retarget.framepose_two_bone_ik import chain_forward_kinematics

    j = leg.fixture_skeleton(np.eye(3), np.eye(3), (0.0, 0.6, -0.8), (0.0, -0.5, -0.86))
    worlds = []
    for Ro in [np.eye(3), yaw_frame(1.1), _rot((1, 2, 3), 0.7), _rot((0, 1, 0), 2.0)]:
        for s in ("l", "r"):
            p = leg.solve_leg(j, s, Ro, ctx)
            assert p.status == "known"
            mid, end = chain_forward_kinematics(ctx["rest"], ctx["legs"][s]["target"], p.proximal_local, p.distal_local)
            assert np.allclose(mid, p.mid, atol=1e-9) and np.allclose(end, p.end, atol=1e-9)  # parent/local round trip
            worlds.append((s, Ro @ (ctx["Am"].T @ p.upper), Ro @ (ctx["Am"].T @ p.forearm)))
    for s in ("l", "r"):
        ws = [w for w in worlds if w[0] == s]
        for w in ws:
            assert np.allclose(w[1], ws[0][1], atol=1e-9) and np.allclose(w[2], ws[0][2], atol=1e-9)


def test_hip_and_spine_angles_on_fixtures(leg, ctx):
    f = leg.fixtures(ctx)
    for name, row in f.items():
        assert row["world_ankle_direction_max_diff_across_policies"] < 1e-9, name
    up = f["upright_neutral"]
    assert abs(up["B"]["hip_flex_l"]) < 1e-6 and up["B"]["violations"] == []
    bend = f["stable_pelvis_waist_bend_60"]  # trunk-axis frame turns a lumbar bend into a hip hinge (within ROM)
    assert bend["A"]["hip_flex_l"] == pytest.approx(45.8, abs=0.2) and abs(bend["A"]["spine_flex"]) < 1e-6
    assert abs(bend["U"]["hip_flex_l"]) < 1e-6 and bend["U"]["spine_flex"] == pytest.approx(45.8, abs=0.2)
    assert bend["A"]["violations"] == [] and bend["U"]["violations"] == []
    assert f["lying_supine"]["U"]["violations"] == ["spine_extension"] and f["lying_supine"]["A"]["violations"] == []
    assert f["lying_prone"]["U"]["violations"] == ["hip_extension"] and f["lying_prone"]["A"]["violations"] == []
    assert f["crouch"]["B"]["hip_flex_l"] == pytest.approx(103.9, abs=0.5) and f["crouch"]["B"]["knee_l"] > 90
    assert f["body_turn_90"]["U"]["object_vs_true_pelvis"] < 1e-6
    assert f["transition_15"]["U"]["violations"] == [] and f["transition_30"]["U"]["violations"] == ["spine_extension"]


def test_knee_hinge_and_degeneracy(leg, ctx):
    # natural knee bend under the true pelvis: knee forward, shin back -> small hinge deviation; backward bend -> ~180
    fwd = leg.fixture_skeleton(np.eye(3), np.eye(3), (0.0, 0.5, -0.87), (0.0, -0.5, -0.87))
    back = leg.fixture_skeleton(np.eye(3), np.eye(3), (0.0, -0.5, -0.87), (0.0, 0.5, -0.87))
    hf = leg.knee_hinge(leg.solve_leg(fwd, "l", np.eye(3), ctx), "l", ctx)
    hb = leg.knee_hinge(leg.solve_leg(back, "l", np.eye(3), ctx), "l", ctx)
    assert hf < 15 and hb > 150
    straight = leg.fixture_skeleton(np.eye(3), np.eye(3), (0.0, 0.0, -1.0), (0.0, 0.0, -1.0))
    e = leg.evaluate(straight, np.eye(3), np.eye(3), ctx)
    assert e["legs"]["l"]["hinge"] is None  # knee flexion below the declared 20 deg evaluation floor
    j = straight.copy()
    j[2] = j[1]  # zero-length thigh
    assert leg.solve_leg(j, "l", np.eye(3), ctx) is None


def test_spine_angle_decomposition(leg):
    f, l, t = leg.spine_angles(np.eye(3), _rot((1, 0, 0), math.radians(-30)))
    assert f == pytest.approx(30.0) and abs(l) < 1e-9 and abs(t) < 1e-9
    f, l, t = leg.spine_angles(np.eye(3), yaw_frame(math.radians(25)))
    assert abs(f) < 1e-9 and t == pytest.approx(25.0)
    f, l, t = leg.spine_angles(np.eye(3), _rot((0, 1, 0), math.radians(20)))
    assert l == pytest.approx(20.0)


def test_upper_body_unchanged_by_leg_path(leg, ctx):
    own = _script("run_pelvis_torso_ownership")
    j = leg.fixture_skeleton(_rot((1, 0, 0), -0.4), _rot((1, 0, 0), -0.9), (0.0, 0.0, -1.0), (0.0, 0.0, -1.0))
    Rt, Rp = torso_frames(j)
    a = ctx["alignment"]
    for Ro in (Rp, own.upright_root(Rt, j), own.lower_trunk_frame(j)):
        assert np.allclose(own.conj(a, Ro) @ own.conj(a, Ro.T @ Rt), own.conj(a, Rt), atol=1e-12)


def test_predeclared_rule_gt_isolation_and_frozen_inputs(leg):
    P = leg.PREDECLARED
    assert P["hip_deg"] == {"flexion_max": 120.0, "extension_max": 30.0, "abduction_max": 45.0, "adduction_max": 30.0}
    assert P["spine_deg"] == {"flexion_max": 90.0, "extension_max": 30.0, "lateral_max": 40.0, "axial_max": 45.0}
    assert P["knee_hinge"]["deviation_max_deg"] == 15.0 and P["severe_jump"]["local_thigh_step_deg"] == 90.0
    text = (ROOT / "scripts/run_pelvis_frame_leg_articulation.py").read_text()
    assert 'r["S"][m] = evaluate(js, r["F"][m][1], r["F"][m][0], ctx, r["gt_legs"])' in text  # S0 legs, W94 S0 frames
    assert "Rts = r[\"F\"][\"A_S0\"][0]" in text and 'jm = summary["temporal"][m]["by_stratum"].get(s, 0)' in text
    assert 'raise SystemExit(f"STOP: frozen input changed' in text
    assert leg.S0_SHA["test"].startswith("4357c7bd") and leg.W87_PREDICTIONS["3dhp"].startswith("356f8cee")
    for path in sorted((ROOT / "src").rglob("*.py")):
        assert "run_pelvis_frame_leg_articulation" not in path.read_text(), path


def test_renderer_leg_fk_check_is_optional():
    render = (ROOT / "scripts/render_runtime_heading_rig_blender.py").read_text()
    assert 'pts = fr.get("fk_points_armature", {}).get(mode)' in render
    assert 'if any(fk_extra.values()):' in render
