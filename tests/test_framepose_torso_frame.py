"""Worklog 93: experimental full torso frame and its rig-replay contract (no production wiring)."""

import importlib.util
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import pytest

from motion.torso_frame import (
    L_HIP, L_SHOULDER, PELVIS, R_HIP, R_SHOULDER, THORAX, compat_yaw, geodesic_degrees, inclination_degrees,
    matrix_to_quaternion_xyzw, quaternion_to_matrix, torso_frames, yaw_frame,
)

ROOT = Path(__file__).resolve().parent.parent
E2E = ROOT / "examples/e2e_demo"


def _script(name):
    sys.path.insert(0, str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def rig():
    return _script("run_torso_frame_rig_replay")


@pytest.fixture(scope="module")
def ctx():
    from retarget.framepose_fk import derive_rig_alignment, load_rig_fk_calibration
    from retarget.framepose_target_rest import load_target_rest_pose
    from retarget.framepose_two_bone_ik import load_ik_chain_set, target_chain_geometry

    rest = load_target_rest_pose(E2E / "baserig_blender_rest.json")
    alignment = derive_rig_alignment(rest, load_rig_fk_calibration(E2E / "framepose_fk_calibration.json"))
    target = target_chain_geometry(rest, load_ik_chain_set(E2E / "framepose_ik_chains.json").chains[0])
    M = np.asarray(rest.bones["spine_01"].matrix_local, float)[:3, :3]
    return {"rest": rest, "alignment": alignment, "target": target, "basis": M / np.linalg.norm(M, axis=0, keepdims=True)}


def _torso(R, trunk=0.5, shoulder=0.2, hip=0.12):
    """(17, 3) joints whose torso is the body frame R (columns x right, y forward, z up) at the origin."""
    j = np.zeros((17, 3))
    x, z = R[:, 0], R[:, 2]
    j[THORAX] = trunk * z
    j[L_SHOULDER], j[R_SHOULDER] = j[THORAX] - shoulder * x, j[THORAX] + shoulder * x
    j[L_HIP], j[R_HIP] = -hip * x, hip * x
    return j


def _rot(axis, angle):
    a = np.asarray(axis, float) / np.linalg.norm(axis)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(angle) * K + (1 - math.cos(angle)) * K @ K


def _random_rotations(n, seed=0):
    rng = np.random.default_rng(seed)
    return [_rot(rng.normal(size=3), rng.uniform(0, math.pi)) for _ in range(n)]


def test_frames_are_right_handed_orthonormal_and_recover_the_body_frame():
    for R in _random_rotations(20):
        Rt, Rp = torso_frames(_torso(R))
        for F in (Rt, Rp):
            assert np.allclose(F.T @ F, np.eye(3), atol=1e-12) and np.linalg.det(F) == pytest.approx(1.0)
            assert geodesic_degrees(F, R) < 1e-5


def test_upright_frame_is_the_yaw_only_contract():
    for yaw in (0.0, 0.7, math.pi, -2.0):
        Rt, Rp = torso_frames(_torso(yaw_frame(yaw)))
        assert np.allclose(Rt, yaw_frame(yaw), atol=1e-12) and np.allclose(Rp, yaw_frame(yaw), atol=1e-12)
        assert compat_yaw(Rt) == pytest.approx(math.atan2(math.sin(yaw), math.cos(yaw)))
        assert inclination_degrees(Rt) == pytest.approx(0.0, abs=1e-6)
    # facing the camera = 180 deg: right shoulder on image left, chest normal toward the camera (-Y depth)
    Rt, _ = torso_frames(_torso(yaw_frame(math.pi)))
    assert np.allclose(Rt[:, 1], (0.0, -1.0, 0.0), atol=1e-12)


def test_lying_fixtures_carry_inclination_and_chest_sign(rig):
    supine = _rot((1, 0, 0), math.pi / 2) @ yaw_frame(0.0)  # on the back: chest normal up the camera Z axis
    prone = _rot((1, 0, 0), -math.pi / 2) @ yaw_frame(0.0)
    for R, label in ((supine, "BACK"), (prone, "STOMACH")):
        Rt, _ = torso_frames(_torso(R))
        assert inclination_degrees(Rt) == pytest.approx(90.0)
        assert rig.gt_label_from_frame(Rt) == label
    # yaw-only cannot express it: every yaw frame stays upright
    assert min(geodesic_degrees(yaw_frame(y), supine) for y in np.linspace(-math.pi, math.pi, 73)) >= 90.0 - 1e-6


def test_degenerate_torsos_are_invalid():
    j = _torso(np.eye(3))
    nan = j.copy()
    nan[L_SHOULDER, 0] = np.nan
    assert torso_frames(nan) == (None, None)
    short = j.copy()
    short[THORAX] = short[PELVIS] + 1e-5
    assert torso_frames(short) == (None, None)
    along = j.copy()  # bilateral shoulder axis parallel to the spine -> thorax frame undefined, pelvis still valid
    along[L_SHOULDER], along[R_SHOULDER] = along[THORAX] - (0, 0, 0.1), along[THORAX] + (0, 0, 0.1)
    Rt, Rp = torso_frames(along)
    assert Rt is None and Rp is not None
    hips = j.copy()
    hips[L_HIP], hips[R_HIP] = (0, 0, -0.1), (0.01, 0, 0.1)  # 3 deg off the spine (< MIN_SIDE_ANGLE_DEG)
    assert torso_frames(hips)[1] is None


def test_quaternion_round_trip_covers_every_branch():
    rots = _random_rotations(30, seed=3) + [_rot(a, math.pi) for a in ((1, 0, 0), (0, 1, 0), (0, 0, 1))] + [np.eye(3)]
    for R in rots:
        q = matrix_to_quaternion_xyzw(R)
        assert sum(c * c for c in q) == pytest.approx(1.0, abs=1e-12)
        assert np.allclose(quaternion_to_matrix(q), R, atol=1e-9)


def test_world_arm_direction_is_invariant_and_yaw_frames_match_the_w91_ik(rig, ctx):
    w91 = _script("run_runtime_heading_rig_replay")
    a, t, r = ctx["alignment"], ctx["target"], ctx["rest"]
    mid, end = np.array([0.05, -0.04, -0.27]), np.array([0.12, -0.15, -0.48])
    worlds = []
    for R in [yaw_frame(0.9)] + _random_rotations(6, seed=5):
        pose = rig.arm_pose(mid, end, R, a, t, r)
        assert pose is not None and pose.status == "known"
        worlds.append(rig.conj(a, R) @ np.asarray(pose.upper))  # object(R_p) * spine(R_p^T R_t) * arm(R_t^T v) = A v
    for w in worlds[1:]:
        assert np.allclose(w, worlds[0], atol=1e-9)
    for yaw in (0.0, 0.9, math.pi, -1.7):  # the full-frame chain with R = Rz(psi) is exactly the W61/W76 yaw-only IK
        full, old = rig.arm_pose(mid, end, yaw_frame(yaw), a, t, r), w91.solve(mid, end, yaw, a, t, r)
        assert np.allclose(full.proximal_local, old.proximal_local, atol=1e-9)
        assert np.allclose(full.distal_local, old.distal_local, atol=1e-9)


def test_spine_local_conjugation_and_yaw_only_baseline(rig, ctx):
    w91 = _script("run_runtime_heading_rig_replay")
    from retarget.framepose_two_bone_ik import quaternion_angle_degrees

    a, B = ctx["alignment"], ctx["basis"]
    for R_rel in _random_rotations(5, seed=7):
        local = quaternion_to_matrix(rig.spine_local_xyzw(a, R_rel, B))
        assert np.allclose(B @ local @ B.T, rig.conj(a, R_rel), atol=1e-9)  # top-level bone: armature delta = B L B^T
    for yaw in (0.0, 1.2, math.pi):  # yaw-only: R_t = R_p = Rz(psi) -> W60 object rotation, identity trunk twist
        Y = yaw_frame(yaw)
        assert quaternion_angle_degrees(matrix_to_quaternion_xyzw(rig.conj(a, Y)), w91.object_rotation(a, yaw)) < 1e-4
        assert quaternion_angle_degrees(rig.spine_local_xyzw(a, Y.T @ Y, B), (0.0, 0.0, 0.0, 1.0)) < 1e-4


def test_runtime_candidate_is_gt_free_and_nothing_in_production_imports_it():
    mod = (ROOT / "src/motion/torso_frame.py").read_text().split('"""', 2)[2]
    assert "target_3d" not in mod and "oracle" not in mod
    text = (ROOT / "scripts/run_torso_frame_rig_replay.py").read_text()
    assert "Rt_s, Rp_s = torso_frames(preds_s0[p])" in text  # F_S0 = frozen S0 joints only
    assert '"F_S0": (Rt_s, Rp_s) if s0_valid else (Y_s0, Y_s0)' in text  # invalid -> W91 yaw-only fallback
    assert '"RUNTIME": "F_S0"' in text and '"GT": "F_GT"' in text
    for path in sorted((ROOT / "src").rglob("*.py")):
        if path.name != "torso_frame.py":
            assert not re.search(r"(from|import)\s+motion\.torso_frame", path.read_text()), path


def test_renderer_and_composer_defaults_are_unchanged(tmp_path):
    render = (ROOT / "scripts/render_runtime_heading_rig_blender.py").read_text()
    assert '"--ortho-scale", type=float, default=2.2' in render
    assert '"--show-mesh", action="store_true"' in render
    assert '"--equivalent-rig-tolerance", type=float, default=None' in render
    pytest.importorskip("cv2")
    compose = _script("compose_runtime_heading_rig_video")
    assert compose.rig_note(tmp_path) == "BaseRig bone proxies (no skinned mesh)"
    (tmp_path / "blender_report.json").write_text(json.dumps({"show_mesh": False}))
    assert compose.rig_note(tmp_path) == "BaseRig bone proxies (no skinned mesh)"
    (tmp_path / "blender_report.json").write_text(json.dumps({"show_mesh": True}))
    assert "skinned mannequin" in compose.rig_note(tmp_path)


def test_frozen_inputs_are_hash_pinned_and_statistics_cluster_by_sequence(rig):
    text = (ROOT / "scripts/run_torso_frame_rig_replay.py").read_text()
    assert rig.W87_PREDICTIONS == {"3dpw": "cb9dc51e3f84ddc9b45c5ada5e941ceaaad5ee73440a6e387ee0c3f2b689c46f",
                                   "3dhp": "356f8ceee545f7ca0a20510afa00efccd48782de5e6e11b78ccce2990a4fcbe1"}
    assert set(rig.S0_SHA) == {"validation", "test", "3dhp"} and 'raise SystemExit(f"STOP: frozen input changed' in text
    # contrasts bootstrap over sequence clusters (3DPW sequence ids are per actor); temporal steps never cross a sequence
    assert '_brief(seq[mask], per[a]["upper"][mask] - per[b]["upper"][mask])' in text and 'seq = np.array([r["sid"] for r in use])' in text
    assert "for rs in seq_rows.values():\n            for a, b in zip(rs[:-1], rs[1:]):" in text
