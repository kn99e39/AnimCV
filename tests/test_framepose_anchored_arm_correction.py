import math
import re
from pathlib import Path

import numpy as np
import pytest

from framepose.anchored_arm_correction import (
    CANDIDATES, anchor_features, anchor_from_raw, candidate_visual, correct_numpy, rodrigues_numpy,
)
from retarget.axis_utils import quaternion_from_axis_angle
from retarget.framepose_ik_gt_attribution import measure_chain
from retarget.framepose_target_rest import RestBone, TargetRigRestPose
from retarget.framepose_two_bone_ik import TwoBoneIkChain, target_chain_geometry
from retarget.framepose_visual_full_arm_ik import full_arm, gt_reference

ROOT = Path(__file__).resolve().parent.parent
I4 = ((1., 0., 0., 0.), (0., 1., 0., 0.), (0., 0., 1., 0.), (0., 0., 0., 1.))
ALIGN = quaternion_from_axis_angle((0.1, 0.2, 0.97), 2.9)
RNG = np.random.default_rng(7)


def _target():
    l1, l2 = 27.771168, 27.251084
    s, e, w = (0., 0., 140.), (0., l1, 140.), (0., l1, 140. - l2)

    def bone(n, p, h, t):
        return RestBone(n, p, I4, h, t)
    rest = TargetRigRestPose("rig", {"body": bone("body", None, (0, 0, 0), (0, 0, 1)), "upper": bone("upper", "body", s, e),
                                     "lower": bone("lower", "upper", e, w), "hand": bone("hand", "lower", w, (w[0], w[1], w[2] - 5))},
                             {"authority": "blender_imported_armature", "fixture": "w82"})
    return rest, target_chain_geometry(rest, TwoBoneIkChain("arm", "s", "e", "w", "upper", "lower", "hand"))


def _unit(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def _rotation_between(a, b):
    """Rotation vector taking unit a to unit b (shortest arc)."""
    axis = np.cross(a, b)
    s, c = np.linalg.norm(axis), float(np.dot(a, b))
    return axis / s * math.atan2(s, c) if s > 1e-12 else np.zeros(3)


def test_zero_correction_is_the_anchor_and_output_is_one_connected_unit_arm():
    raw = RNG.normal(size=(200, 7)) * 3
    u0, v0, rho0 = anchor_from_raw(raw)
    a = anchor_features(u0, v0, rho0)
    elbow, wrist, rho, u, v = correct_numpy(np.zeros((200, 7)), u0, v0, rho0)
    assert np.allclose(elbow, a[:, 7:10], atol=1e-6) and np.allclose(wrist, a[:, 10:13], atol=1e-6)
    assert np.allclose(u, u0, atol=1e-15) and np.allclose(rho, rho0, atol=1e-12)
    c = RNG.normal(size=(200, 7)) * 2
    elbow, wrist, rho, u, v = correct_numpy(c, u0, v0, rho0)
    assert np.allclose(np.linalg.norm(u, axis=1), 1, atol=1e-12) and np.allclose(np.linalg.norm(v, axis=1), 1, atol=1e-12)
    assert np.allclose(np.linalg.norm(elbow, axis=1) + np.linalg.norm(wrist - elbow, axis=1), 1, atol=1e-12)
    assert (np.linalg.norm(wrist, axis=1) <= 1 + 1e-12).all() and ((rho > 0) & (rho < 1)).all()


def test_rodrigues_is_a_rotation_and_matches_torch():
    w = RNG.normal(size=(100, 3)) * 2
    x = _unit(RNG.normal(size=(100, 3)))
    y = rodrigues_numpy(w, x)
    assert np.allclose(np.linalg.norm(y, axis=1), 1, atol=1e-12)
    k = _unit(w)
    assert np.allclose((y * k).sum(1), (x * k).sum(1), atol=1e-12)  # component along the axis preserved
    assert np.allclose(rodrigues_numpy(np.zeros((1, 3)), x[:1]), x[:1])
    assert np.allclose(rodrigues_numpy(np.array([[0, 0, math.pi / 2]]), np.array([[1.0, 0, 0]])), [[0, 1, 0]], atol=1e-12)
    torch = pytest.importorskip("torch")
    from framepose.anchored_arm_correction import rodrigues_torch
    for scale in (1.0, 1e-6):
        wt = torch.as_tensor(w * scale)
        assert np.allclose(rodrigues_torch(wt, torch.as_tensor(x)).numpy(), rodrigues_numpy(w * scale, x), atol=1e-10)


def test_perfect_correction_reproduces_the_worklog62_reference_arm():
    rest, target = _target()
    for theta, bend, fwd, yaw in ((30, 50, 0.3, 0.4), (10, 120, -0.6, -1.7), (60, 95, 0.8, 2.6)):
        t, b = math.radians(theta), math.radians(bend)
        e = np.array([0.30 * math.cos(t), 0.3 * fwd, -0.30 * math.sin(t)])
        l = np.array([math.cos(t + b), -fwd, -math.sin(t + b)])
        w = e + 0.27 * l / np.linalg.norm(l)
        u_gt, v_gt = _unit(e), _unit(w - e)
        rho_gt = np.linalg.norm(e) / (np.linalg.norm(e) + np.linalg.norm(w - e))
        raw = RNG.normal(size=(1, 7))
        u0, v0, rho0 = anchor_from_raw(raw)
        c = np.concatenate([_rotation_between(u0[0], u_gt), _rotation_between(v0[0], v_gt),
                            [math.log(rho_gt / (1 - rho_gt)) - math.log(rho0[0] / (1 - rho0[0]))]])[None]
        elbow, wrist, _, _, _ = correct_numpy(c, u0, v0, rho0)
        total = np.linalg.norm(e) + np.linalg.norm(w - e)
        assert np.allclose(elbow[0], e / total, atol=1e-12) and np.allclose(wrist[0], w / total, atol=1e-12)
        mr = measure_chain((0.0, 0.0, 0.0), tuple(elbow[0]), tuple(wrist[0]), yaw, ALIGN)
        gt = measure_chain((0.0, 0.0, 0.0), tuple(e), tuple(w), yaw, ALIGN)
        arm, ref = full_arm("r", mr, mr.direction, target, rest), gt_reference(gt, target, rest)
        assert math.dist(arm.end, ref.end) < 1e-9 and math.dist(arm.mid, ref.mid) < 1e-9
        assert np.allclose(arm.proximal_local, ref.proximal_local, atol=1e-9)
        assert arm.integrity["ik_to_fk_residual"] < 1e-9 and arm.integrity["quaternion_norm_error"] < 1e-12
        assert np.linalg.norm(arm.upper) == pytest.approx(target.proximal_length)


def test_visual_control_identity_init_and_loss():
    r = RNG.normal(size=(3, 4, 8)).astype(np.float16)
    assert (candidate_visual(r, "R0_GEOMETRY_CORRECTION") == 0).all() and candidate_visual(r, "R1_VISUAL_CORRECTION") is r
    with pytest.raises(ValueError):
        candidate_visual(r, "x")
    assert CANDIDATES == ("R0_GEOMETRY_CORRECTION", "R1_VISUAL_CORRECTION")
    torch = pytest.importorskip("torch")
    from framepose.anchored_arm_correction import build_correction, correct_torch, correction_loss
    models = []
    for _ in range(2):
        torch.manual_seed(1337)
        models.append(build_correction(8))
    assert all(torch.equal(a, b) for a, b in zip(models[0].state_dict().values(), models[1].state_dict().values()))
    assert sum(p.numel() for p in models[0].parameters()) == 8 * 64 + 64 + 308 * 128 + 128 + 128 * 128 + 128 + 128 * 7 + 7
    u0, v0, rho0 = anchor_from_raw(RNG.normal(size=(5, 7)))
    a = torch.as_tensor(anchor_features(u0, v0, rho0))
    c = models[0](torch.as_tensor(r[:1].repeat(5, 0)).float() * 0, torch.randn(5, 39), a)
    assert float(c.abs().max()) == 0.0  # zero-initialized head -> identity
    elbow, wrist = correct_torch(c, a)
    assert float((elbow - a[:, 7:10]).abs().max()) < 1e-6 and float((wrist - a[:, 10:13]).abs().max()) < 1e-6
    loss, _, _ = correction_loss(c, a, a[:, 7:10], a[:, 10:13], torch.ones(5))
    assert float(loss) < 1e-5
    # Zero-initialized head still receives gradient when the anchor is wrong (the correction is trainable).
    wrong, _, _ = correction_loss(c, a, a[:, 7:10] + 0.1, a[:, 10:13] - 0.1, torch.ones(5))
    wrong.backward()
    assert float(models[0].head.weight.grad.abs().sum()) > 0


def test_static_contracts():
    train = (ROOT / "scripts/train_framepose_anchored_arm_correction.py").read_text()
    for check in ("J0_SHA256", "STOP: frozen J0 does not reproduce Worklog 80", "requires_grad_(False)", "STOP: frozen J0 changed",
                  "STOP: initial corrected arm differs from the anchor", "STOP: severe frozen-anchor stacking mismatch",
                  "STOP: R0/R1 control identity broken", "SourceBalancedSampler", "AUDIT_CRITERION = {"):
        assert check in train
    assert train.index("AUDIT_CRITERION = {") < train.index("def main():")
    assert train.index("STOP: severe frozen-anchor") < train.index("for cand in CANDIDATES")
    # The anchor never receives RGB: its forward pass uses exact zeros.
    assert "predict_joint(j0, np.zeros_like(readouts" in train
    assert "E1_3DPW_PLUS_EXTERNAL" not in train and "chain_f_e1" not in train and "h0" not in train.lower().replace("j0", "")
    module = (ROOT / "src/framepose/anchored_arm_correction.py").read_text().split('"""', 2)[2]
    network = module.split("def build_correction", 1)[1].split("def correction_loss", 1)[0]
    assert not re.search(r"target|h0|e1|yaw|temporal|oracle", network, re.IGNORECASE)
    for path in sorted((ROOT / "src").rglob("*.py")):
        if path.name != "anchored_arm_correction.py":
            assert "anchored_arm_correction" not in path.read_text(), path


def test_eval_runner_contracts_static():
    run = (ROOT / "scripts/run_framepose_anchored_arm_correction_eval.py").read_text()
    for check in ("STOP: historical parity failed", "STOP: corrected geometry violates", "identity mismatch",
                  'g1["pairs"] == 5504', "STOP: jump thresholds differ from Worklog 80", "anchor_vs_w80_j0_raw"):
        assert check in run
    body = run.split("def main():", 1)[1]
    assert body.index("STOP: historical parity failed") < body.index("STOP: corrected geometry violates") < body.index("static_cmp = ")
    assert run.index("DECISION = {") < run.index("def main():")
    assert not re.search(r"\.backward\(|optimizer|smooth|torch|load_state_dict", run)
