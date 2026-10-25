"""Worklog 98: GT-free HMR 2.0b body-orientation handoff (adapter, frames, isolation)."""

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest

from motion.torso_frame import geodesic_degrees, yaw_frame

ROOT = Path(__file__).resolve().parent.parent


def _script(name):
    sys.path.insert(0, str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def h():
    return _script("run_external_orientation_handoff")


def _rx(deg):
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], float)


def _template_joints():
    """24 SMPL joints of an upright template body in SMPL template axes (x subject-left, y up, z forward)."""
    j = np.zeros((24, 3))
    j[1], j[2] = (0.1, -0.05, 0), (-0.1, -0.05, 0)        # hips
    j[4], j[5] = (0.1, -0.45, 0), (-0.1, -0.45, 0)        # knees
    j[7], j[8] = (0.1, -0.85, 0), (-0.1, -0.85, 0)        # ankles
    j[3], j[6], j[9], j[12], j[15] = (0, 0.1, 0), (0, 0.25, 0), (0, 0.35, 0), (0, 0.5, 0), (0, 0.6, 0)
    j[16], j[17] = (0.18, 0.4, 0), (-0.18, 0.4, 0)        # shoulders
    j[18], j[19], j[20], j[21] = (0.2, 0.1, 0), (-0.2, 0.1, 0), (0.2, -0.15, 0), (-0.2, -0.15, 0)
    return j


def _posed(go):
    """SMPL joints rotated by a global orient (template -> OpenCV camera axes)."""
    return _template_joints() @ np.asarray(go).T


def test_upright_facing_camera_maps_to_yaw_180(h):
    go = np.diag([1.0, -1.0, -1.0])  # SMPL upright, facing the camera, in OpenCV axes (y down, z forward)
    R_p, R_t, j17 = h.handoff_frames(go, _posed(go))
    assert np.allclose(R_p, yaw_frame(math.pi), atol=1e-9)
    assert R_t is not None and geodesic_degrees(R_t, R_p) < 1e-6  # template torso is straight: thorax = pelvis
    assert np.linalg.det(R_p) == pytest.approx(1.0) and j17[10, 2] > 0.5  # head up in canonical axes


def test_lying_and_rotated_fixtures(h):
    base = np.diag([1.0, -1.0, -1.0])
    for deg in (30, 90, -90):
        go = base @ _rx(deg)  # rotate the body about its own left-right axis
        R_p, R_t, _ = h.handoff_frames(go, _posed(go))
        assert math.degrees(math.acos(np.clip(R_p[2, 2], -1, 1))) == pytest.approx(abs(deg), abs=1e-6)
        assert geodesic_degrees(R_t, R_p) < 1e-6
    yawed = base @ np.array([[math.cos(0.7), 0, math.sin(0.7)], [0, 1, 0], [-math.sin(0.7), 0, math.cos(0.7)]])
    R_p, _, _ = h.handoff_frames(yawed, _posed(yawed))
    assert R_p[2, 2] == pytest.approx(1.0) and geodesic_degrees(R_p, yaw_frame(math.pi)) == pytest.approx(math.degrees(0.7), abs=1e-6)


def test_smpl_basis_is_not_the_armature_basis(h):
    w94 = sys.modules.get("run_pelvis_torso_ownership") or _script("run_pelvis_torso_ownership")
    S = w94.SMPL_BODY
    assert not np.allclose(S, np.eye(3)) and np.linalg.det(S) == pytest.approx(1.0)  # template axes != AnimCV body axes
    R_p, _, _ = h.handoff_frames(np.eye(3), _posed(np.eye(3)))  # identity SMPL orient = upside-down in OpenCV axes
    assert R_p[2, 2] == pytest.approx(-1.0)


def test_chain_thorax_uses_spine_locals(h):
    go = np.diag([1.0, -1.0, -1.0])
    bp = np.tile(np.eye(3), (23, 1, 1))
    assert geodesic_degrees(h.smpl_chain_thorax(go, bp), yaw_frame(math.pi)) < 1e-9
    bp[2] = _rx(30)  # spine1 (joint 3) flexion
    assert geodesic_degrees(h.smpl_chain_thorax(go, bp), yaw_frame(math.pi)) == pytest.approx(30.0, abs=1e-6)


def test_parent_local_composition_reproduces_thorax(h):
    _script("run_torso_frame_rig_replay")
    from motion.torso_frame import quaternion_to_matrix
    from retarget.framepose_fk import derive_rig_alignment, load_rig_fk_calibration
    from retarget.framepose_target_rest import load_target_rest_pose

    rest = load_target_rest_pose(ROOT / "examples/e2e_demo/baserig_blender_rest.json")
    a = derive_rig_alignment(rest, load_rig_fk_calibration(ROOT / "examples/e2e_demo/framepose_fk_calibration.json"))
    M = np.asarray(rest.bones["spine_01"].matrix_local, float)[:3, :3]
    B = M / np.linalg.norm(M, axis=0, keepdims=True)
    R_p, R_t = yaw_frame(2.0) @ _rx(20), yaw_frame(2.3) @ _rx(50)
    Q = B @ quaternion_to_matrix(h.spine_local_xyzw(a, R_p.T @ R_t, B)) @ B.T
    assert np.allclose(h.conj(a, R_p) @ Q, h.conj(a, R_t), atol=1e-9)


def test_gt_free_adapter_and_frozen_identities(h):
    text = (ROOT / "scripts/run_external_orientation_handoff.py").read_text()
    body = text.split("def handoff_frames", 1)[1].split("def smpl_chain_thorax", 1)[0]
    for forbidden in ("R_smpl", "smpl", "gt", "joints3", "cam_poses", "intrinsics", "stratum"):
        assert forbidden not in body.lower().replace("smpl24_to_canonical17", "").replace("smpl_body", "")
    assert h.W97_PREDICTIONS_SHA == "fc5d330565205f4792cd3641671633decbab4e24d70522d4dd85f045d87a7003"
    assert 'raise SystemExit(f"STOP: frozen input changed' in text
    rc = text.split("def row_conditions", 1)[1].split("def metrics_row", 1)[0]
    assert 'out["P"] = cond_eval(r, e["j17"], e["R_p"], Rts' in rc and 'out["T"] = cond_eval(r, e["j17"], e["R_p"], e["R_t"]' in rc
    assert 'out["T_S"] = cond_eval(r, r["S17"], e["R_p"], e["R_t"]' in rc  # only the leg source changes
    for path in sorted((ROOT / "src").rglob("*.py")):
        assert "run_external_orientation_handoff" not in path.read_text(), path


def test_error_structure_helpers(h):
    d = np.array([0.3, -0.2, 1.0])
    R = h.ray_rotation(d)
    assert np.allclose(R @ [0, 0, 1], d / np.linalg.norm(d)) and np.linalg.det(R) == pytest.approx(1.0)
    assert np.allclose(h.ray_rotation([0, 0, 1]), np.eye(3))
    assert np.allclose(h.chordal_mean([_rx(10), _rx(-10)]), np.eye(3), atol=1e-12)
    Rs = [yaw_frame(0.1 * k) for k in range(5)]
    rows = [(s, Rx @ _rx(5), Rx, None) for s in ("3dpw_validation", "3dpw_test") for Rx in Rs]
    rows = [(s, a, b, a) for s, a, b, _ in rows]
    out = h.pelvis_error_structure(rows)
    assert out["validation_offsets_deg"]["body"] == pytest.approx(5.0, abs=1e-6)  # constant body-side offset is recovered...
    assert out["3dpw_test"]["geodesic_after_val_body_offset_removed"] == pytest.approx(0.0, abs=1e-4)  # ...and removed (arccos precision)
