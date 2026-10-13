import math
import re
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from framepose.arm_geometry_owner import (
    CANDIDATES, DEGENERATE_BEND_SINE, candidate_visual, chain_targets, e1_direction, geometry_inputs,
)
from framepose.contract import JOINT_COUNT, JOINT_INDEX
from retarget.axis_utils import quaternion_from_axis_angle, rotate_vector_by_quaternion
from retarget.framepose_ik_gt_attribution import heading_relative, measure_chain
from retarget.framepose_target_rest import RestBone, TargetRigRestPose
from retarget.framepose_two_bone_ik import (
    DEGENERATE_BEND_SINE as W61_SINE, TwoBoneIkChain, observed_bend_side, source_chain_geometry, target_chain_geometry,
)
from retarget.framepose_visual_full_arm_ik import full_arm
from retarget.framepose_visual_ik_handoff import camera_chain_direction, rig_direction

ROOT = Path(__file__).resolve().parent.parent
I4 = ((1., 0., 0., 0.), (0., 1., 0., 0.), (0., 0., 1., 0.), (0., 0., 0., 1.))
ALIGN = quaternion_from_axis_angle((0.1, 0.2, 0.97), 2.9)
S, E, W = JOINT_INDEX["left_shoulder"], JOINT_INDEX["left_elbow"], JOINT_INDEX["left_wrist"]


def _target():
    l1, l2 = 27.771168, 27.251084
    s, e, w = (0., 0., 140.), (0., l1, 140.), (0., l1, 140. - l2)

    def bone(n, p, h, t):
        return RestBone(n, p, I4, h, t)
    rest = TargetRigRestPose("rig", {"body": bone("body", None, (0, 0, 0), (0, 0, 1)), "upper": bone("upper", "body", s, e),
                                     "lower": bone("lower", "upper", e, w), "hand": bone("hand", "lower", w, (w[0], w[1], w[2] - 5))},
                             {"authority": "blender_imported_armature", "fixture": "w78"})
    return rest, target_chain_geometry(rest, TwoBoneIkChain("arm", "s", "e", "w", "upper", "lower", "hand"))


def _arm(theta, bend, forward=0.3, upper=0.3, lower=0.25, origin=(0.1, 2.0, 1.4)):
    t, b = math.radians(theta), math.radians(bend)
    u = np.array([upper * math.cos(t), forward * upper, -upper * math.sin(t)])
    e = np.array(origin) + u
    l = np.array([math.cos(t + b), -forward, -math.sin(t + b)])
    return np.array(origin), e, e + lower * l / np.linalg.norm(l)


def _bank_points(arms):
    p = np.full((len(arms), JOINT_COUNT, 3), np.nan)
    v = np.zeros((len(arms), JOINT_COUNT), bool)
    for i, (s, e, w) in enumerate(arms):
        p[i, S], p[i, E], p[i, W] = s, e, w
        v[i, [S, E, W]] = True
    return p, v


def _d(arms, noise=0.0):
    out = []
    for s, _, w in arms:
        c = (w - s) / np.linalg.norm(w - s)
        out.append(c + noise * np.array([0.05, -0.1, 0.08]))
    out = np.array(out)
    return out / np.linalg.norm(out, axis=1, keepdims=True)


def test_targets_follow_worklog61_62_definitions_across_bends_depths_and_rotations():
    arms = [_arm(30, 50), _arm(20, 120, forward=-0.6), _arm(60, 5, forward=0.9), _arm(10, 175, forward=-0.2),
            (np.array([0.1, 2.0, 1.4]), np.array([0.4, 2.1, 1.3]), np.array([0.64, 2.18, 1.22]))]  # collinear
    p, v = _bank_points(arms)
    d = _d(arms, noise=1.0)
    reach, rm, side, sm = chain_targets(p, v, d, np.ones(len(arms), bool))
    assert DEGENERATE_BEND_SINE == W61_SINE
    for i, (s, e, w) in enumerate(arms):
        g = source_chain_geometry(tuple(e - s), tuple(w - e))
        assert rm[i] and reach[i] == pytest.approx(g.reach_fraction, abs=1e-12)
        obs = observed_bend_side(g)
        if obs is None:
            assert not sm[i]  # straight arm: reach-only supervision
            continue
        o = np.array(obs)
        want = o - (o @ d[i]) * d[i]
        assert sm[i] and np.allclose(side[i], want / np.linalg.norm(want), atol=1e-12)
        assert abs(side[i] @ d[i]) < 1e-12
    assert not sm[4] and rm[4] and reach[4] == pytest.approx(1.0)
    # Rotation equivariance: reach invariant, side rotates with the scene.
    q = quaternion_from_axis_angle((0.3, -0.5, 0.8), 1.1)
    rot = lambda x: np.array(rotate_vector_by_quaternion(tuple(x), q))  # noqa: E731
    pr, vr = _bank_points([tuple(rot(x) for x in a) for a in arms])
    r2, _, s2, sm2 = chain_targets(pr, vr, np.array([rot(x) for x in d]), np.ones(len(arms), bool))
    assert np.allclose(r2, reach, atol=1e-12) and (sm2 == sm).all()
    assert np.allclose(s2[sm], np.array([rot(x) for x in side[sm]]), atol=1e-12)


def test_invalid_and_degenerate_geometry_is_masked():
    arms = [_arm(30, 50), _arm(30, 50), _arm(30, 50)]
    p, v = _bank_points(arms)
    v[1, E] = False                      # invalid GT joint
    p[2, E] = p[2, S]                    # zero-length upper arm
    reach, rm, _, sm = chain_targets(p, v, _d(arms), np.array([False, True, True]))
    assert list(rm) == [True, False, False] and list(sm) == [False, False, False]  # no direction -> no side label
    inp = np.zeros((2, JOINT_COUNT, 3))
    inp[:, S, :2], inp[:, W, :2] = (0.5, 0.5), (0.5, 0.5)
    inp[1, W, :2] = (0.6, 0.7)
    d, ok = e1_direction(np.array([0.2, 0.2]), inp, np.array([[640, 480], [640, 480]], float))
    assert list(ok) == [False, True] and (d[0] == 0).all() and np.linalg.norm(d[1]) == pytest.approx(1.0)


def test_perfect_prediction_reproduces_worklog77_oracle_arm_and_ik_contracts():
    rest, target = _target()
    h0_pts, gt_pts = _arm(30, 50), _arm(38, 95, forward=-0.4, upper=0.31, lower=0.24)
    for yaw in (0.0, 0.7, -2.4):
        h0 = measure_chain(*map(tuple, h0_pts), yaw, ALIGN)
        gt = measure_chain(*map(tuple, gt_pts), yaw, ALIGN)
        d_cam = np.array(camera_chain_direction(0.25, (0.6, -0.8)))
        d_rig = rig_direction(tuple(d_cam), yaw, ALIGN)
        p, v = _bank_points([gt_pts])
        reach, _, side, sm = chain_targets(p, v, d_cam[None], np.array([True]))
        assert sm[0]
        learned = replace(h0, reach_fraction=float(reach[0]), side=heading_relative(tuple(side[0]), yaw, ALIGN), degenerate=False)
        arm = full_arm("g", learned, d_rig, target, rest)
        oracle = full_arm("c111", replace(h0, reach_fraction=gt.reach_fraction, side=gt.side, degenerate=gt.degenerate), d_rig, target, rest)
        assert math.dist(arm.end, oracle.end) < 1e-9 and math.dist(arm.mid, oracle.mid) < 1e-9
        assert np.allclose(arm.proximal_local, oracle.proximal_local, atol=1e-9)
        assert arm.integrity["ik_to_fk_residual"] < 1e-9 and arm.integrity["upper_length_residual"] < 1e-9
        # An unprojected raw side gives the same arm (the Worklog 62 rule re-perpendicularizes it).
        raw = side[0] + 3.0 * d_cam
        same = full_arm("g", replace(learned, side=heading_relative(tuple(raw), yaw, ALIGN)), d_rig, target, rest)
        assert math.dist(same.mid, arm.mid) < 1e-9
        # A raw side parallel to the direction is unavailable: no fallback, no substitution.
        bad = full_arm("g", replace(learned, side=heading_relative(tuple(2.0 * d_cam), yaw, ALIGN)), d_rig, target, rest)
        assert bad.status == "unavailable:bend_side"
        # Any reach in (0, 1) is IK-valid; tiny reach is clamped by the Worklog 61 solver, not by the model.
        tiny = full_arm("g", replace(learned, reach_fraction=1e-4), d_rig, target, rest)
        assert tiny.clamp == "clamped_to_min_reach" and tiny.status == "known"


def test_inputs_and_visual_control():
    r = np.random.default_rng(0).normal(size=(3, 4, 8)).astype(np.float16)
    assert (candidate_visual(r, "G0_GEOMETRY_ONLY") == 0).all() and candidate_visual(r, "G0_GEOMETRY_ONLY").shape == r.shape
    assert candidate_visual(r, "G1_GEOMETRY_PLUS_VISION") is r
    with pytest.raises(ValueError):
        candidate_visual(r, "other")
    g = geometry_inputs(np.ones((3, 3, 13)), np.array([0.1, 0.2, 0.3]), np.eye(3))
    assert g.shape == (3, 43) and g[1, 39] == pytest.approx(0.2) and (g[:, 40:] == np.eye(3)).all()
    assert CANDIDATES == ("G0_GEOMETRY_ONLY", "G1_GEOMETRY_PLUS_VISION")


def test_model_identity_and_loss_contract():
    torch = pytest.importorskip("torch")
    from framepose.arm_geometry_owner import build_owner, owner_loss, perpendicular_side
    models = []
    for _ in range(2):
        torch.manual_seed(1337)
        models.append(build_owner(8))
    sd0, sd1 = models[0].state_dict(), models[1].state_dict()
    assert all(torch.equal(sd0[k], sd1[k]) for k in sd0)
    assert sum(p.numel() for p in models[0].parameters()) == 8 * 64 + 64 + 299 * 128 + 128 + 128 * 128 + 128 + 129 + 387
    vis = torch.zeros(5, 4, 8)
    geo = torch.randn(5, 43)
    reach, side = models[0](vis, geo)
    assert reach.shape == (5,) and side.shape == (5, 3) and bool(((reach > 0) & (reach < 1)).all())
    d = torch.nn.functional.normalize(torch.randn(5, 3), dim=-1)
    ps = perpendicular_side(side, d)
    assert torch.allclose(ps.norm(dim=-1), torch.ones(5), atol=1e-5) and float((ps * d).sum(-1).abs().max()) < 1e-5
    loss, rt, st = owner_loss(reach, side, d, reach.detach(), torch.ones(5), ps.detach(), torch.ones(5))
    assert float(rt) == 0 and float(st) < 1e-6
    _, _, st0 = owner_loss(reach, side, d, reach.detach(), torch.ones(5), ps.detach(), torch.zeros(5))
    assert float(st0) == 0  # side-invalid rows never contribute


def test_static_contracts():
    train = (ROOT / "scripts/train_framepose_arm_geometry_owner.py").read_text()
    for check in ("E1_SHA256", "STOP: frozen E1 does not reproduce Worklog 73", "requires_grad_(False)", "SourceBalancedSampler",
                  "draw budget differs from Worklog 73", "STOP: G0/G1 control identity broken", "torch.manual_seed(CONFIG[\"seed\"])"):
        assert check in train
    assert train.index("STOP: frozen E1 does not reproduce") < train.index("for cand in CANDIDATES")
    assert 'bank.indices("test")' not in train  # selection never sees test
    module = (ROOT / "src/framepose/arm_geometry_owner.py").read_text().split('"""', 2)[2]
    # GT enters only chain_targets (labels); inputs never take target arrays; no H0 / temporal input.
    assert "target" not in module.split("def geometry_inputs", 1)[1].split("def build_owner", 1)[0]
    assert not re.search(r"oracle|sign_tensor|smooth|load_bank|h0_dir|prediction_test", module, re.IGNORECASE)
    network = module.split("def geometry_inputs", 1)[1]
    assert not re.search(r"h0|temporal|yaw|target_3d", network, re.IGNORECASE)
    for path in sorted((ROOT / "src").rglob("*.py")):
        if path.name != "arm_geometry_owner.py":
            assert "arm_geometry_owner" not in path.read_text(), path


def test_learned_arm_assembly_and_side_error():
    import run_framepose_arm_geometry_owner_eval as ev
    rest, target = _target()
    yaw = 0.6
    h0 = measure_chain(*map(tuple, _arm(30, 50)), yaw, ALIGN)
    gt = measure_chain(*map(tuple, _arm(40, 100, forward=-0.3)), yaw, ALIGN)
    d_cam = np.array(camera_chain_direction(0.2, (0.5, -0.7)))
    d_rig = rig_direction(tuple(d_cam), yaw, ALIGN)
    side_cam = np.array([0.3, -0.2, 0.9])
    b1 = full_arm("b1", h0, d_rig, target, rest)
    assert ev.learned_arm("x", h0, d_rig, None, None, yaw, ALIGN, target, rest, "h0", "h0") == replace(b1, name="x")
    r_only = ev.learned_arm("x", h0, d_rig, 0.6, side_cam, yaw, ALIGN, target, rest, "G1", "h0")
    assert r_only.reach_fraction == 0.6 and r_only.side == b1.side
    s_only = ev.learned_arm("x", h0, d_rig, 0.6, side_cam, yaw, ALIGN, target, rest, "h0", "G1")
    assert s_only.reach_fraction == h0.reach_fraction and s_only.side != b1.side
    perfect = full_arm("p", replace(h0, side=gt.side), d_rig, target, rest)
    assert ev.side_error(perfect, gt) == pytest.approx(0.0, abs=1e-6)
    assert ev.side_error(b1, gt) > 0 and set(ev.LEARNED) == {"g0", "g1", "g0r_h0s", "h0r_g0s", "g1r_h0s", "h0r_g1s"}


def test_eval_runner_contracts_static():
    run = (ROOT / "scripts/run_framepose_arm_geometry_owner_eval.py").read_text()
    for check in ("STOP: historical baseline parity failed", "STOP: learned output geometry invalid", "identity mismatch",
                  "owner predictions differ from the training report", 'thresholds = w76_report["test"]["pooled"]["temporal"]["thresholds_gt_p99"]',
                  "E1_INPUT_TOLERANCE = 1e-5", 'g1["pairs"] == 5504'):
        assert check in run
    body = run.split("def main():", 1)[1]
    assert body.index("STOP: historical baseline parity failed") < body.index("STOP: learned output geometry invalid") \
        < body.index("static_cmp = ")
    assert run.index("DECISION = {") < run.index("def main():")
    assert not re.search(r"\.backward\(|optimizer|smooth|torch", run)
