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
        if path.name != "runtime_heading.py":
            assert "runtime_heading" not in path.read_text(), path
