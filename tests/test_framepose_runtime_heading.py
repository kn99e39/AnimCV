import importlib.util
import math
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _script(name):
    sys.path.insert(0, str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def ev():
    return _script("run_runtime_heading_eval")


def test_circular_difference_wraps_around(ev):
    assert ev.circ(math.radians(179), math.radians(-179)) == pytest.approx(-2.0)
    assert ev.circ(math.radians(-179), math.radians(179)) == pytest.approx(2.0)
    assert abs(ev.circ(0.0, math.pi)) == pytest.approx(180.0)
    assert ev.circ(math.radians(10), math.radians(370)) == pytest.approx(0.0, abs=1e-9)


def test_view_strata_follow_the_facing_convention(ev):
    # facing the camera: right shoulder at image left -> axis (-1, 0) -> yaw 180
    assert ev.view_class(math.pi) == "front" and ev.view_class(0.0) == "back"
    assert ev.view_class(math.radians(90)) == "side" and ev.view_class(math.radians(-90)) == "side"
    assert ev.view_class(math.radians(135)) == "oblique"


def test_unknown_is_never_scored_as_correct(ev):
    rows = [{"sequence_id": "s", "frame": i, "t": i / 10, "gt": 0.0, "known": i % 2 == 0,
             "yaw": 0.0 if i % 2 == 0 else None, "conf": 0.9, "reliability": None} for i in range(20)]
    out = ev.evaluate(rows)
    assert out["coverage"] == pytest.approx(0.5) and out["known"] == 10


def test_sudden_flip_and_true_turn_accounting(ev):
    rows = []
    for i in range(40):
        gt = math.radians(min(i, 20) * 6.0)  # GT turns 120 deg over 2 s, then holds
        yaw = gt + (math.pi if i == 30 else 0.0)  # one spurious 180 deg flip while GT is steady
        rows.append({"sequence_id": "s", "frame": 3 * i, "t": i * 0.1, "gt": gt, "known": True, "yaw": yaw,
                     "conf": 0.9, "reliability": None})
    out = ev.evaluate(rows)
    assert out["temporal"]["sudden_flips"] == 2  # into and out of the flipped row
    assert out["temporal"]["true_turn_windows"] > 0 and out["temporal"]["true_turn_tracked_rate"] == pytest.approx(1.0)
    assert out["error"]["front_back_gt135"] == pytest.approx(1 / 40)


def test_runtime_providers_never_read_ground_truth(ev):
    text = (ROOT / "scripts/run_runtime_heading_eval.py").read_text()
    body = text.split("def provider_lifted", 1)[1].split("\ndef ", 1)[0]
    assert "target" not in body and "oracle" not in body
    providers = text.split('providers = {"A_h0_oracle"', 1)[1].split("report = ", 1)[0]
    assert "_matched_oracle_sequence" not in providers and "target_3d" not in providers
    assert "EVALUATION-ONLY" in text
    assert not re.search(r"\.backward\(|optim\.|load_state_dict", text)


def _lifted(frames):
    from pose.pose_lifter import LiftedPoseFrame, LiftedPosePoint, LiftedPoseSequence

    out = []
    for i, joints in enumerate(frames):
        pts = {name: LiftedPosePoint(name=name, position=pos, confidence=conf, depth_uncertainty=1 - conf, observation_valid=valid)
               for name, (pos, conf, valid) in joints.items()}
        out.append(LiftedPoseFrame(i, i / 30.0, pts))
    return LiftedPoseSequence(frames=out, source_fps=30.0, backend="test")


def _torso(yaw_sh, yaw_hip, conf=0.9, valid=True):
    def pair(yaw, left, right):
        dx, dy = math.cos(yaw) * 0.2, math.sin(yaw) * 0.2
        return {left: ((-dx, -dy, 0.0), conf, valid), right: ((dx, dy, 0.0), conf, valid)}
    return {**pair(yaw_sh, "left_shoulder", "right_shoulder"), **pair(yaw_hip, "left_hip", "right_hip")}


def test_runtime_heading_states_and_unchanged_yaw():
    pytest.importorskip("numpy")
    from motion.animation_semantics_v2_bridge import current_root_orientation
    from motion.runtime_heading import MAX_PAIR_DISAGREEMENT_DEG, MIN_TORSO_CONFIDENCE, runtime_heading

    seq = _lifted([_torso(math.pi, math.pi), _torso(math.pi, math.pi, conf=MIN_TORSO_CONFIDENCE / 2),
                   _torso(math.radians(90), math.radians(90 + MAX_PAIR_DISAGREEMENT_DEG + 10)), _torso(0.0, 0.0, valid=False)])
    out = runtime_heading(seq)
    assert [h.status for h in out] == ["KNOWN", "UNRELIABLE", "UNRELIABLE", "UNKNOWN"]
    assert out[1].reasons == ("low_torso_confidence",) and out[2].reasons == ("shoulder_hip_disagreement",)
    base = current_root_orientation(seq)
    for h, b in zip(out, base):  # the yaw value is never altered by the reliability state
        assert (h.yaw_radians is None) == (not b.known) and (h.yaw_radians is None or h.yaw_radians == b.yaw_radians)
    # facing the camera (right shoulder at image left) is 180 deg
    assert abs(abs(out[0].yaw_radians) - math.pi) < 1e-9


def test_runtime_heading_is_stateless_per_frame_and_track():
    from motion.runtime_heading import runtime_heading

    a = _lifted([_torso(math.radians(170), math.radians(175)), _torso(math.radians(10), math.radians(5))])
    b = _lifted([_torso(math.radians(10), math.radians(5))])
    # a genuine 160 deg turn between rows is reported immediately (no hold), and history never leaks into a frame
    assert runtime_heading(a)[1].yaw_radians == runtime_heading(b)[0].yaw_radians
    text = (ROOT / "src/motion/runtime_heading.py").read_text()
    assert "target" not in text.split('"""', 2)[2] and "oracle" not in text.split('"""', 2)[2]


def test_rig_replay_heading_contract():
    rig = _script("run_runtime_heading_rig_replay")
    from retarget.axis_utils import rotate_vector_by_quaternion
    import numpy as np

    a = (0.0, 0.0, 0.7071067811865476, 0.7071067811865476)  # any unit alignment
    for yaw in (0.0, 1.0, math.pi):
        q = rig.object_rotation(a, yaw)
        assert abs(sum(c * c for c in q) - 1.0) < 1e-12
    q0 = rig.object_rotation(a, 0.0)
    assert np.allclose(rotate_vector_by_quaternion((1.0, 2.0, 3.0), q0), (1.0, 2.0, 3.0))
    text = (ROOT / "scripts/run_runtime_heading_rig_replay.py").read_text()
    assert '"RUNTIME": h.yaw_radians' in text and "EVALUATION-ONLY" in text
    for path in sorted((ROOT / "src").rglob("*.py")):
        if not path.name.startswith("runtime_heading"):  # the experimental runtime_heading* modules themselves
            assert "runtime_heading" not in path.read_text(), path


def test_front_back_and_profile_are_carried_only_by_the_lifted_depth_sign():
    """Same 2D evidence, mirrored lifted depth: the heading mirrors and the reliability state cannot tell (3DHP TS2)."""
    from motion.runtime_heading import runtime_heading

    # side-on subject: right-minus-left axis along +depth (yaw +90) vs the same image with mirrored depth (yaw -90)
    side = _lifted([_torso(math.radians(90), math.radians(90)), _torso(math.radians(-90), math.radians(-90))])
    a, b = runtime_heading(side)
    assert abs(math.degrees(a.yaw_radians) - 90) < 1e-9 and abs(math.degrees(b.yaw_radians) + 90) < 1e-9
    assert a.status == b.status == "KNOWN"  # a mirrored profile is not flagged: documented limitation
    # facing the camera vs facing away differ by the image-x order of the shoulders, which 2D does observe
    front_back = _lifted([_torso(math.pi, math.pi), _torso(0.0, 0.0)])
    f, k = runtime_heading(front_back)
    assert abs(abs(math.degrees(f.yaw_radians)) - 180) < 1e-9 and abs(math.degrees(k.yaw_radians)) < 1e-9


@pytest.fixture(scope="module")
def rig_assets():
    from retarget.framepose_fk import derive_rig_alignment, load_rig_fk_calibration
    from retarget.framepose_target_rest import load_target_rest_pose
    from retarget.framepose_two_bone_ik import load_ik_chain_set, target_chain_geometry

    rest = load_target_rest_pose(ROOT / "examples/e2e_demo/baserig_blender_rest.json")
    alignment = derive_rig_alignment(rest, load_rig_fk_calibration(ROOT / "examples/e2e_demo/framepose_fk_calibration.json"))
    target = target_chain_geometry(rest, load_ik_chain_set(ROOT / "examples/e2e_demo/framepose_ik_chains.json").chains[0])
    return rest, alignment, target


def test_ik_parity_with_the_unchanged_w76_path_and_world_invariance(rig_assets):
    import numpy as np
    from retarget.axis_utils import rotate_vector_by_quaternion
    from retarget.framepose_ik_gt_attribution import measure_chain
    from retarget.framepose_visual_full_arm_ik import full_arm

    rest, alignment, target = rig_assets
    rig = _script("run_runtime_heading_rig_replay")
    rng = np.random.default_rng(91)
    for _ in range(25):
        u, v = rng.normal(size=3), rng.normal(size=3)
        u, v = u / np.linalg.norm(u), v / np.linalg.norm(v)
        rho = rng.uniform(0.35, 0.65)
        e, w = rho * u, rho * u + (1 - rho) * v
        worlds = []
        for yaw in (0.0, math.pi, rng.uniform(-math.pi, math.pi)):
            pose = rig.solve(e, w, yaw, alignment, target, rest)
            m = measure_chain((0.0, 0.0, 0.0), tuple(e), tuple(w), yaw, alignment)
            ref = full_arm("arm", m, m.direction, target, rest)
            # identical code path; python-float vs numpy-float inputs differ only at ULP level
            assert np.allclose(pose.proximal_local, ref.proximal_local, atol=1e-12) and np.allclose(pose.distal_local, ref.distal_local, atol=1e-12)
            assert pose.integrity["endpoint_reconstruction_residual"] < 1e-9 and pose.integrity["ik_to_fk_residual"] < 1e-6
            q = rig.object_rotation(alignment, yaw)
            worlds.append(np.asarray(rotate_vector_by_quaternion(tuple(pose.upper), q)))
        # the root rotation and the heading-relative arm cancel in the world frame (Worklog 90/91)
        assert np.allclose(worlds[0], worlds[1], atol=1e-9) and np.allclose(worlds[0], worlds[2], atol=1e-9)
