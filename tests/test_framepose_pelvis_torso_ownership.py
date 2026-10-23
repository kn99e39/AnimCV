"""Worklog 94: root / pelvis / thorax ownership diagnostics (experimental, unwired)."""

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest

from motion.torso_frame import geodesic_degrees, quaternion_to_matrix, torso_frames, yaw_frame

ROOT = Path(__file__).resolve().parent.parent
E2E = ROOT / "examples/e2e_demo"


def _script(name):
    sys.path.insert(0, str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def own():
    return _script("run_pelvis_torso_ownership")


@pytest.fixture(scope="module")
def rig(own):
    from retarget.framepose_fk import derive_rig_alignment, load_rig_fk_calibration
    from retarget.framepose_target_rest import load_target_rest_pose
    from retarget.framepose_two_bone_ik import load_ik_chain_set, target_chain_geometry

    rest = load_target_rest_pose(E2E / "baserig_blender_rest.json")
    a = derive_rig_alignment(rest, load_rig_fk_calibration(E2E / "framepose_fk_calibration.json"))
    M = np.asarray(rest.bones["spine_01"].matrix_local, float)[:3, :3]
    return {"rest": rest, "a": a, "A": quaternion_to_matrix(a), "basis": M / np.linalg.norm(M, axis=0, keepdims=True),
            "target": target_chain_geometry(rest, load_ik_chain_set(E2E / "framepose_ik_chains.json").chains[0]),
            "legs": own.rest_leg_dirs(rest, quaternion_to_matrix(a))}


def _rot(axis, angle):
    a = np.asarray(axis, float) / np.linalg.norm(axis)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(angle) * K + (1 - math.cos(angle)) * K @ K


def _random(n, seed):
    rng = np.random.default_rng(seed)
    return [_rot(rng.normal(size=3), rng.uniform(0, math.pi)) for _ in range(n)]


def _proper(R):
    return np.allclose(R.T @ R, np.eye(3), atol=1e-9) and np.linalg.det(R) == pytest.approx(1.0)


def test_candidate_pelvis_frames_are_right_handed(own):
    for Pp, Pt in zip(_random(10, 1), _random(10, 2)):
        j = own.fixture_joints(Pp, Pt, Pp @ (0.0, 0.0, -1.0))
        for R in (own.lower_trunk_frame(j), own.leg_frame(j), own.upright_root(torso_frames(j)[0], j)):
            assert R is not None and _proper(R)


def test_w93_pelvis_and_thorax_share_the_trunk_axis_so_spine_01_is_pure_twist(own):
    for Pp, Pt in zip(_random(12, 3), _random(12, 4)):
        Rt, Rp = torso_frames(own.fixture_joints(Pp, Pt, (0.0, 0.0, -1.0)))
        assert np.allclose(Rt[:, 2], Rp[:, 2], atol=1e-12)
        rel = Rp.T @ Rt
        assert np.allclose(rel[:, 2], (0.0, 0.0, 1.0), atol=1e-9)  # rotation about the shared axis only: no flexion DOF


def test_fixtures_separate_waist_flexion_whole_body_tilt_pelvis_tilt_and_twist(own, rig):
    f = own.fixtures(rig["legs"])
    erect, flex, lying, tilt, twist = (f[k] for k in ("erect", "waist_flexion_60", "lying_supine", "pelvis_tilt_20_torso_compensating", "axial_twist_40"))
    # upright: every owner keeps the Object near upright (only the fixture's lumbar offset)
    assert all(erect[k]["object_tilt"] < 12 for k in ("A_W93", "L_lower_trunk", "U_upright_root", "B_true_pelvis"))
    # pure waist flexion: W93 converts it into whole-body (Object) rotation; the trunk-relative flexion is lost
    assert flex["A_W93"]["object_tilt"] > 40 and flex["A_W93"]["spine_swing"] < 1e-4 and flex["A_W93"]["spine_vs_true_relative"] > 55
    assert flex["B_true_pelvis"]["object_tilt"] < 1e-6 and flex["U_upright_root"]["object_tilt"] < 1e-6
    assert flex["A_W93"]["rest_thigh_error"] > 40 > flex["U_upright_root"]["rest_thigh_error"]
    # whole body lying: W93 is right, the upright root is not
    assert lying["A_W93"]["object_tilt"] > 85 and lying["A_W93"]["rest_thigh_error"] < 5
    assert lying["U_upright_root"]["object_tilt"] < 1e-6 and lying["U_upright_root"]["rest_thigh_error"] > 85
    # pelvis tilted under a compensating upright torso: neither trunk-axis nor upright owners can see it
    assert tilt["A_W93"]["object_vs_true_pelvis"] > 15 and tilt["U_upright_root"]["object_vs_true_pelvis"] > 15
    assert tilt["B_true_pelvis"]["rest_thigh_error"] > 15  # a correct pelvis with REST legs still gives wrong legs
    # pure axial twist: W93 keeps it on spine_01 (no Object tilt), the hip-yaw root as well
    assert twist["A_W93"]["spine_vs_true_relative"] < 1 and twist["U_upright_root"]["object_vs_true_pelvis"] < 1


def test_object_rotation_is_the_only_transform_reaching_the_thighs(rig, own):
    rest = rig["rest"]
    top = {n for n, b in rest.bones.items() if b.parent is None}
    assert {"spine_01", "thigh_l", "thigh_r"} <= top and "pelvis" not in rest.bones
    assert rest.bones["clavicle_l"].parent == "spine_05"
    A = rig["A"]
    for R in _random(5, 6):  # world thigh = (A R A^-1)(A d) = A R d  for a top-level rest thigh
        d = rig["legs"]["thigh_l"]
        assert np.allclose(own.conj(rig["a"], R) @ (A @ d), A @ (R @ d), atol=1e-12)
    assert rig["legs"]["thigh_l"][2] < -0.99 and rig["legs"]["thigh_r"][2] < -0.99


def test_spine_and_arm_parent_frames_are_consistent_across_owners(own, rig):
    a, B, rest = rig["a"], rig["basis"], rig["rest"]
    head, sh = np.asarray(rest.bones["spine_01"].head, float), np.asarray(rest.bones["upperarm_l"].head, float)
    Rt = _random(1, 9)[0]
    pose = own.arm_pose(np.array([0.05, -0.04, -0.27]), np.array([0.12, -0.15, -0.48]), Rt, a, rig["target"], rest)
    worlds, shoulders = [], []
    for Ro in [Rt, yaw_frame(0.4)] + _random(4, 10):
        local = quaternion_to_matrix(own.spine_local_xyzw(a, Ro.T @ Rt, B))
        Q = B @ local @ B.T  # spine_01 armature-space delta
        O = own.conj(a, Ro)
        assert np.allclose(O @ Q, own.conj(a, Rt), atol=1e-9)
        worlds.append(O @ Q @ np.asarray(pose.upper))
        shoulders.append(O @ Q @ (sh - head))
    for w, s in zip(worlds, shoulders):
        assert np.allclose(w, worlds[0], atol=1e-9) and np.allclose(s, shoulders[0], atol=1e-9)


def test_camera_frame_versus_gravity_frame(own):
    Rwc = np.diag([1.0, -1.0, -1.0])  # upright camera, world +Y up
    assert np.allclose(own.CV_TO_CANON @ Rwc @ (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    # SMPL zero pose seen by that camera faces the camera: right axis on -X, forward toward the camera (-Y)
    R = own.CV_TO_CANON @ Rwc @ own.SMPL_BODY
    assert _proper(R) and np.allclose(R, yaw_frame(math.pi), atol=1e-12)
    for g in ((0.0, 0.0, 1.0), (0.0, 0.2, 0.98), (0.3, -0.1, 0.95)):
        g = np.asarray(g) / np.linalg.norm(g)
        G = own.level(g)
        assert _proper(G) and np.allclose(G @ g, (0.0, 0.0, 1.0), atol=1e-12)
    t14 = math.degrees(math.acos(own.hp_gravity("TS2")[2]))
    t56 = math.degrees(math.acos(own.hp_gravity("TS5")[2]))
    assert 10 < t14 < 15 and 3 < t56 < 8  # calibration camera tilts (evaluation-only references)


def test_gt_isolation_frozen_inputs_and_s0_owners(own):
    text = (ROOT / "scripts/run_pelvis_torso_ownership.py").read_text()
    assert '"A_S0": (Rts, Rps) if s0_valid else (Ys, Ys)' in text
    assert '"L_S0": (Rts, low_s) if s0_valid and low_s is not None else (Ys, Ys)' in text
    assert '"U_S0": (Rts, upright_root(Rts, js)) if s0_valid else (Ys, Ys)' in text
    assert "low, low_s = lower_trunk_frame(j), lower_trunk_frame(js) if s0_valid else None" in text
    assert 'raise SystemExit(f"STOP: frozen input changed' in text
    assert own.S0_SHA["3dhp"].startswith("a78adeb0") and own.W87_PREDICTIONS["3dpw"].startswith("cb9dc51e")
    for path in sorted((ROOT / "src").rglob("*.py")):
        assert "run_pelvis_torso_ownership" not in path.read_text(), path


def test_renderer_and_composer_extensions_are_optional():
    render = (ROOT / "scripts/render_runtime_heading_rig_blender.py").read_text()
    assert 'legs = fr.get("reference_legs_cam")' in render and "seg.hide_render = legs is None" in render
    assert 'fr["modes"].get(f"GT_REF_{mode}", fr["modes"].get("GT_REF"))' in render
    compose = (ROOT / "scripts/compose_runtime_heading_rig_video.py").read_text()
    assert 'if "plot_series" in case:' in compose
    assert 'case.get("header_note", "J1X left arm via W61/W76 IK | thin green = GT heading / GT-reference arm (evaluation only)")' in compose
