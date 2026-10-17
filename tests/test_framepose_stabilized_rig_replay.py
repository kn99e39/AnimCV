import importlib.util
import json
import math
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
RNG = np.random.default_rng(90)
REST = ROOT / "examples/e2e_demo/baserig_blender_rest.json"


def _script(name):
    sys.path.insert(0, str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def rig():
    pytest.importorskip("cv2")
    from retarget.framepose_fk import derive_rig_alignment, load_rig_fk_calibration
    from retarget.framepose_target_rest import load_target_rest_pose
    from retarget.framepose_two_bone_ik import load_ik_chain_set, target_chain_geometry

    rest = load_target_rest_pose(REST)
    alignment = derive_rig_alignment(rest, load_rig_fk_calibration(ROOT / "examples/e2e_demo/framepose_fk_calibration.json"))
    target = target_chain_geometry(rest, load_ik_chain_set(ROOT / "examples/e2e_demo/framepose_ik_chains.json").chains[0])
    return rest, alignment, target


def _chain():
    u, v = RNG.normal(size=3), RNG.normal(size=3)
    u, v = u / np.linalg.norm(u), v / np.linalg.norm(v)
    rho = RNG.uniform(0.35, 0.65)
    return rho * u, rho * u + (1 - rho) * v


def test_rotvec_matches_the_quaternion_angle():
    mod = _script("run_stabilized_rig_replay")
    from retarget.axis_utils import quaternion_from_axis_angle
    from retarget.framepose_two_bone_ik import quaternion_angle_degrees

    for _ in range(20):
        axis = RNG.normal(size=3)
        axis /= np.linalg.norm(axis)
        a = quaternion_from_axis_angle(tuple(RNG.normal(size=3) / 2 + 0.1), RNG.uniform(0, 3))
        d = quaternion_from_axis_angle(tuple(axis), RNG.uniform(0, 3))
        from retarget.axis_utils import quaternion_multiply
        b = quaternion_multiply(d, a)
        rv = mod.rotvec(a, b)
        assert np.linalg.norm(rv) == pytest.approx(quaternion_angle_degrees(a, b), abs=1e-6)


def test_ik_endpoint_lengths_and_quaternions_are_exact_and_gt_free(rig):
    rest, alignment, target = rig
    from retarget.framepose_ik_gt_attribution import measure_chain
    from retarget.framepose_visual_full_arm_ik import full_arm

    for _ in range(50):
        e, w = _chain()
        m = measure_chain((0.0, 0.0, 0.0), tuple(e), tuple(w), math.pi, alignment)
        pose = full_arm("RAW", m, m.direction, target, rest)
        assert pose.status == "known"
        assert pose.integrity["endpoint_reconstruction_residual"] < 1e-9 and pose.integrity["ik_to_fk_residual"] < 1e-6
        assert pose.integrity["upper_length_residual"] < 1e-9 and pose.integrity["quaternion_norm_error"] < 1e-9
        # rig lengths are the BaseRig rest lengths and the reach is the predicted |D| (no GT length enters)
        lo, hi = abs(target.proximal_length - target.distal_length), target.proximal_length + target.distal_length
        assert math.dist(pose.root, pose.end) == pytest.approx(min(max(np.linalg.norm(w) * target.max_reach, lo), hi), rel=1e-9)


def test_constant_yaw_only_changes_the_torso_frame_not_the_world_arm(rig):
    """World arm direction = object rotation (A Rz(psi) A^-1) applied to the heading-relative arm: psi-independent."""
    rest, alignment, target = rig
    from retarget.axis_utils import quaternion_conjugate, quaternion_from_axis_angle, quaternion_multiply, rotate_vector_by_quaternion
    from retarget.framepose_ik_gt_attribution import heading_relative

    for _ in range(20):
        v = tuple(RNG.normal(size=3))
        world = []
        for psi in (0.0, math.pi, 0.7):
            local = heading_relative(v, psi, alignment)
            obj = quaternion_multiply(quaternion_multiply(alignment, quaternion_from_axis_angle((0, 0, 1), psi)), quaternion_conjugate(alignment))
            world.append(np.asarray(rotate_vector_by_quaternion(local, obj)))
        assert np.allclose(world[0], world[1]) and np.allclose(world[0], world[2])


def test_disabled_stabilizer_reproduces_the_raw_rig_poses_exactly(rig):
    rest, alignment, target = rig
    mod = _script("run_stabilized_rig_replay")
    from framepose.depth_motion_stabilizer import stabilize

    n = 12
    chains = [_chain() for _ in range(n)]
    e = np.array([c[0] for c in chains])
    w = np.array([c[1] for c in chains])
    e2, w2 = stabilize(np.array(["s"] * n), np.arange(n) / 6, e, w, np.ones(n, bool), None)
    rows = list(range(n))
    poses = mod.solve_rows(None, rows, {"RAW": (e, w), "STABILIZED": (e2, w2)}, target, rest, alignment, math.pi)
    for p in rows:
        assert poses["RAW"][p].proximal_local == poses["STABILIZED"][p].proximal_local
        assert poses["RAW"][p].distal_local == poses["STABILIZED"][p].distal_local


def test_review_windows_are_track_contiguous_and_isolated():
    mod = _script("run_stabilized_rig_replay")
    samples = [SimpleNamespace(sequence_id=s, frame_index=f) for s, fs in (("a", range(0, 300, 5)), ("b", range(0, 300, 5))) for f in fs]
    samples[10] = SimpleNamespace(sequence_id="a", frame_index=51)  # a break inside track a
    pop = SimpleNamespace(rows=np.arange(len(samples)), bank=SimpleNamespace(samples=samples))
    wins = mod.track_windows(pop, 5, 36)
    for w in wins:
        seqs = {samples[p].sequence_id for p in w}
        frames = [samples[p].frame_index for p in w]
        assert len(seqs) == 1 and len(w) == 36 and all(b - a == 5 for a, b in zip(frames[:-1], frames[1:]))


def test_contracts_no_gt_at_runtime_and_no_production_changes():
    run = (ROOT / "scripts/run_stabilized_rig_replay.py").read_text()
    assert run.index("DECISION = {") < run.index("def main():")
    for check in ("STOP: Worklog 89 config changed", "STOP: Worklog 89 stabilized output not reproduced",
                  "STOP: Blender rest snapshot differs", "EVALUATION ONLY"):
        assert check in run
    # GT enters only as the GT_REF evaluation reference; the runtime chains are the frozen predictions / stabilizer
    modes = run.split('chains = {"RAW"', 1)[1].split("poses = solve_rows", 1)[0]
    assert "g1_elbow" in modes and "e_c, w_c" in modes and "g0_elbow" in modes and modes.count("e_t") == 1 and '"GT_REF": (e_t, w_t)' in modes
    assert not re.search(r"\.backward\(|optim\.|load_state_dict", run)
    blender = (ROOT / "scripts/render_stabilized_rig_replay_blender.py").read_text()
    assert '"CONSTANT"' in blender and "keyframe_insert" in blender
    for path in sorted((ROOT / "src").rglob("*.py")):
        text = path.read_text()
        assert "stabilized_rig_replay" not in text, path
    stab = (ROOT / "src/framepose/depth_motion_stabilizer.py").read_text()
    assert "EXPERIMENTAL (Worklog 89)" in stab


def test_compose_angle_is_sign_invariant():
    pytest.importorskip("cv2")
    mod = _script("compose_stabilized_rig_replay_video")
    q = np.array([0.1, 0.2, 0.3, 0.9])
    q /= np.linalg.norm(q)
    assert mod.qangle(q, -q) == pytest.approx(0.0, abs=1e-5)
