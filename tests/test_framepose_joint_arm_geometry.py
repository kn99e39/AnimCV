import math
import re
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from framepose.contract import JOINT_COUNT, JOINT_INDEX
from framepose.joint_arm_geometry import CANDIDATES, candidate_visual, decode_numpy, normalized_arm_targets
from retarget.axis_utils import quaternion_from_axis_angle, rotate_vector_by_quaternion
from retarget.framepose_ik_gt_attribution import measure_chain
from retarget.framepose_target_rest import RestBone, TargetRigRestPose
from retarget.framepose_two_bone_ik import TwoBoneIkChain, observed_bend_side, source_chain_geometry, target_chain_geometry
from retarget.framepose_visual_full_arm_ik import full_arm, gt_reference

ROOT = Path(__file__).resolve().parent.parent
I4 = ((1., 0., 0., 0.), (0., 1., 0., 0.), (0., 0., 1., 0.), (0., 0., 0., 1.))
ALIGN = quaternion_from_axis_angle((0.1, 0.2, 0.97), 2.9)
S, E, W = (JOINT_INDEX[j] for j in ("left_shoulder", "left_elbow", "left_wrist"))


def _target():
    l1, l2 = 27.771168, 27.251084
    s, e, w = (0., 0., 140.), (0., l1, 140.), (0., l1, 140. - l2)

    def bone(n, p, h, t):
        return RestBone(n, p, I4, h, t)
    rest = TargetRigRestPose("rig", {"body": bone("body", None, (0, 0, 0), (0, 0, 1)), "upper": bone("upper", "body", s, e),
                                     "lower": bone("lower", "upper", e, w), "hand": bone("hand", "lower", w, (w[0], w[1], w[2] - 5))},
                             {"authority": "blender_imported_armature", "fixture": "w80"})
    return rest, target_chain_geometry(rest, TwoBoneIkChain("arm", "s", "e", "w", "upper", "lower", "hand"))


def _arm(theta, bend, forward=0.3, upper=0.3, lower=0.25, origin=(0.1, 2.0, 1.4)):
    t, b = math.radians(theta), math.radians(bend)
    u = np.array([upper * math.cos(t), forward * upper, -upper * math.sin(t)])
    e = np.array(origin) + u
    l = np.array([math.cos(t + b), -forward, -math.sin(t + b)])
    return np.array(origin), e, e + lower * l / np.linalg.norm(l)


def _bank(arms):
    p = np.full((len(arms), JOINT_COUNT, 3), np.nan)
    v = np.zeros((len(arms), JOINT_COUNT), bool)
    for i, (s, e, w) in enumerate(arms):
        p[i, S], p[i, E], p[i, W] = s, e, w
        v[i, [S, E, W]] = True
    return p, v


def _raw_for(s, e, w):
    """Raw outputs that decode exactly to the arm (s, e, w)."""
    up, lo = e - s, w - e
    rho = np.linalg.norm(up) / (np.linalg.norm(up) + np.linalg.norm(lo))
    return np.concatenate([3.0 * up, 0.5 * lo, [math.log(rho / (1 - rho))]])


ARMS = [_arm(30, 50), _arm(20, 120, forward=-0.6), _arm(60, 5, forward=0.9), _arm(10, 170, forward=-0.2),
        _arm(45, 80, upper=0.35, lower=0.18), _arm(-30, 60, forward=0.0, upper=0.2, lower=0.31),
        (np.array([0.1, 2.0, 1.4]), np.array([0.4, 2.1, 1.3]), np.array([0.64, 2.18, 1.22]))]  # last one straight


def test_decode_is_one_connected_unit_length_arm_with_worklog61_derived_quantities():
    raw = np.array([_raw_for(*a) for a in ARMS])
    elbow, wrist, rho = decode_numpy(raw)
    for i, (s, e, w) in enumerate(ARMS):
        total = np.linalg.norm(e - s) + np.linalg.norm(w - e)
        assert np.allclose(elbow[i], (e - s) / total, atol=1e-12) and np.allclose(wrist[i], (w - s) / total, atol=1e-12)
        assert np.linalg.norm(elbow[i]) + np.linalg.norm(wrist[i] - elbow[i]) == pytest.approx(1.0, abs=1e-12)
        g = source_chain_geometry(tuple(elbow[i]), tuple(wrist[i] - elbow[i]))
        ref = source_chain_geometry(tuple(e - s), tuple(w - e))
        assert g.reach_fraction == pytest.approx(ref.reach_fraction, abs=1e-12) == pytest.approx(np.linalg.norm(wrist[i]), abs=1e-12)
        assert np.allclose(g.endpoint_direction, ref.endpoint_direction, atol=1e-12)
        side, ref_side = observed_bend_side(g), observed_bend_side(ref)
        assert (side is None) == (ref_side is None) and (side is None or np.allclose(side, ref_side, atol=1e-9))
    assert observed_bend_side(source_chain_geometry(tuple(elbow[-1]), tuple(wrist[-1] - elbow[-1]))) is None
    # Any raw output is a valid arm: reach <= 1, rho in (0, 1).
    rnd = np.random.default_rng(0).normal(size=(500, 7)) * 5
    e2, w2, r2 = decode_numpy(rnd)
    assert (np.linalg.norm(w2, axis=1) <= 1 + 1e-12).all() and ((r2 > 0) & (r2 < 1)).all()


def test_targets_are_scale_free_rotation_equivariant_and_masked():
    p, v = _bank(ARMS)
    elbow, wrist, ok = normalized_arm_targets(p, v)
    assert ok.all()
    p2, v2 = _bank([tuple(3.7 * x + 5 for x in a) for a in ARMS])
    e2, w2, _ = normalized_arm_targets(p2, v2)
    assert np.allclose(e2, elbow, atol=1e-9) and np.allclose(w2, wrist, atol=1e-9)
    q = quaternion_from_axis_angle((0.3, -0.5, 0.8), 1.1)
    rot = lambda x: np.array(rotate_vector_by_quaternion(tuple(x), q))  # noqa: E731
    p3, v3 = _bank([tuple(rot(x) for x in a) for a in ARMS])
    e3, w3, _ = normalized_arm_targets(p3, v3)
    assert np.allclose(e3, [rot(x) for x in elbow], atol=1e-9)
    p[0, E] = p[0, S]
    v[1, W] = False
    p[2, S, 0] = np.nan
    _, _, ok = normalized_arm_targets(p, v)
    assert list(ok[:3]) == [False, False, False] and ok[3:].all()


def test_perfect_prediction_reproduces_the_worklog62_reference_arm_under_any_yaw():
    rest, target = _target()
    for s, e, w in ARMS[:6]:
        raw = _raw_for(s, e, w)[None]
        elbow, wrist, _ = decode_numpy(raw)
        for yaw in (0.0, 0.9, -2.2):
            gt = measure_chain(tuple(s), tuple(e), tuple(w), yaw, ALIGN)
            j = measure_chain((0.0, 0.0, 0.0), tuple(elbow[0]), tuple(wrist[0]), yaw, ALIGN)
            arm, ref = full_arm("j", j, j.direction, target, rest), gt_reference(gt, target, rest)
            assert arm.status == ref.status == "known"
            assert math.dist(arm.end, ref.end) < 1e-9 and math.dist(arm.mid, ref.mid) < 1e-9
            assert np.allclose(arm.proximal_local, ref.proximal_local, atol=1e-9)
            assert np.allclose(arm.distal_local, ref.distal_local, atol=1e-9)
            assert arm.integrity["ik_to_fk_residual"] < 1e-9 and arm.integrity["quaternion_norm_error"] < 1e-12
            assert arm.integrity["upper_length_residual"] < 1e-9 and arm.integrity["forearm_length_residual"] < 1e-9
    # Straight predicted arm: valid direction/reach, historical rest-plane side fallback.
    s, e, w = ARMS[-1]
    elbow, wrist, _ = decode_numpy(_raw_for(s, e, w)[None])
    j = measure_chain((0.0, 0.0, 0.0), tuple(elbow[0]), tuple(wrist[0]), 0.3, ALIGN)
    arm = full_arm("j", j, j.direction, target, rest)
    assert j.degenerate and arm.status == "known" and arm.side_status == "degenerate_rest_plane_fallback"


def test_target_rig_lengths_do_not_inherit_source_proportions():
    rest, target = _target()
    for upper, lower in ((0.2, 0.35), (0.35, 0.2)):
        s, e, w = _arm(30, 70, upper=upper, lower=lower)
        elbow, wrist, rho = decode_numpy(_raw_for(s, e, w)[None])
        lu, ll = np.linalg.norm(e - s), np.linalg.norm(w - e)
        assert rho[0] == pytest.approx(lu / (lu + ll))
        j = measure_chain((0.0, 0.0, 0.0), tuple(elbow[0]), tuple(wrist[0]), 0.0, ALIGN)
        arm = full_arm("j", j, j.direction, target, rest)
        assert np.linalg.norm(arm.upper) == pytest.approx(target.proximal_length)
        assert np.linalg.norm(arm.forearm) == pytest.approx(target.distal_length)


def test_visual_control_and_model_identity():
    r = np.random.default_rng(1).normal(size=(3, 4, 8)).astype(np.float16)
    assert (candidate_visual(r, "J0_GEOMETRY_ONLY") == 0).all() and candidate_visual(r, "J1_GEOMETRY_PLUS_VISION") is r
    with pytest.raises(ValueError):
        candidate_visual(r, "x")
    assert CANDIDATES == ("J0_GEOMETRY_ONLY", "J1_GEOMETRY_PLUS_VISION")
    torch = pytest.importorskip("torch")
    from framepose.joint_arm_geometry import build_joint_arm, decode_torch, joint_loss
    models = []
    for _ in range(2):
        torch.manual_seed(1337)
        models.append(build_joint_arm(8))
    assert all(torch.equal(a, b) for a, b in zip(models[0].state_dict().values(), models[1].state_dict().values()))
    assert sum(p.numel() for p in models[0].parameters()) == 8 * 64 + 64 + 295 * 128 + 128 + 128 * 128 + 128 + 128 * 7 + 7
    raw = torch.as_tensor(np.array([_raw_for(*a) for a in ARMS]), dtype=torch.float64)
    elbow, wrist = decode_torch(raw)
    p, v = _bank(ARMS)
    et, wt, ok = normalized_arm_targets(p, v)
    loss, le, lw = joint_loss(raw, torch.as_tensor(et), torch.as_tensor(wt), torch.ones(len(ARMS), dtype=torch.float64))
    assert float(loss) < 1e-9
    # A correct chain direction with a misplaced elbow is NOT a good loss.
    bad = raw.clone()
    bad[0, 0:3] = torch.as_tensor([0.0, 0.0, 1.0])
    _, le_bad, _ = joint_loss(bad, torch.as_tensor(et), torch.as_tensor(wt), torch.ones(len(ARMS), dtype=torch.float64))
    assert float(le_bad) > 0.01
    _, _, lw0 = joint_loss(bad, torch.as_tensor(et), torch.as_tensor(wt), torch.zeros(len(ARMS), dtype=torch.float64))
    assert float(lw0) == 0


def test_static_contracts():
    train = (ROOT / "scripts/train_framepose_joint_arm_geometry.py").read_text()
    for check in ("SourceBalancedSampler", "STOP: J0/J1 control identity broken", "torch.manual_seed(CONFIG[\"seed\"])",
                  "STOP: draw budget differs", "load_cache", "never overwrite"):
        assert check in train
    assert 'bank.indices("test")' not in train and "E1_3DPW_PLUS_EXTERNAL" not in train and "chain_f_e1" not in train
    module = (ROOT / "src/framepose/joint_arm_geometry.py").read_text().split('"""', 2)[2]
    network = module.split("def build_joint_arm", 1)[1]
    assert not re.search(r"h0|e1|target|yaw|temporal|oracle", network.split("def joint_loss", 1)[0], re.IGNORECASE)
    for path in sorted((ROOT / "src").rglob("*.py")):
        if path.name != "joint_arm_geometry.py":
            assert "joint_arm_geometry" not in path.read_text(), path


def test_evaluation_decomposition_corners_are_exact():
    import run_framepose_joint_arm_geometry_eval as ev
    rest, target = _target()
    s, e, w = ARMS[1]
    gs, ge, gw = _arm(35, 90, forward=-0.4)
    yaw = 0.8
    elbow, wrist, _ = decode_numpy(_raw_for(s, e, w)[None])
    mj = ev.j_measure(elbow[0], wrist[0], yaw, ALIGN)
    gt = measure_chain(tuple(gs), tuple(ge), tuple(gw), yaw, ALIGN)
    ref = gt_reference(gt, target, rest)
    allgt = ev.decomposition_arm("x", mj, gt, ("g", "g", "g"), target, rest)
    assert math.dist(allgt.end, ref.end) < 1e-9 and math.dist(allgt.mid, ref.mid) < 1e-9
    model = full_arm("j", mj, mj.direction, target, rest)
    none = ev.decomposition_arm("x", mj, gt, ("j", "j", "j"), target, rest)
    assert none.end == model.end and none.mid == model.mid
    r_only = ev.decomposition_arm("x", mj, gt, ("j", "g", "j"), target, rest)
    assert r_only.reach_fraction == gt.reach_fraction and r_only.direction == model.direction
    assert set(ev.CORNERS) == {f"d{d}r{r}s{s}" for d in "jg" for r in "jg" for s in "jg"} - {"djrjsj"}
    assert ev.j_measure((0, 0, 0), (0, 0, 0), 0.0, ALIGN) is None  # zero arm -> unmeasurable, reported unavailable


def test_eval_runner_contracts_static():
    run = (ROOT / "scripts/run_framepose_joint_arm_geometry_eval.py").read_text()
    for check in ("STOP: J training-target convention differs", "STOP: historical baseline parity failed",
                  "STOP: J output geometry cannot be mapped", 'thresholds = w76_report["test"]["pooled"]["temporal"]["thresholds_gt_p99"]',
                  'g2["pairs"] == 5504', "identity mismatch"):
        assert check in run
    body = run.split("def main():", 1)[1]
    assert body.index("STOP: J training-target convention") < body.index("STOP: historical baseline parity") \
        < body.index("STOP: J output geometry") < body.index("static_cmp = ")
    assert run.index("DECISION = {") < run.index("def main():")
    assert not re.search(r"\.backward\(|optimizer|smooth|torch", run)
    # GT enters J arms only in the evaluation-only decomposition corners.
    collect = run.split("def collect", 1)[1].split("def _regime", 1)[0]
    assert "j_measure(preds[" in collect and "ev[\"gt\"]" in collect.split("for c, corner in CORNERS.items()", 1)[1]
