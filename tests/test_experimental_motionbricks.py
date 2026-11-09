"""Focused tests for the EXPERIMENTAL W129 MotionBricks adapter (src/app/experimental_motionbricks.py).

Fixtures are labelled:
  * SYNTHETIC SOMA-like skeleton (T-pose, identity rest rotations, the 30 model joint names + the helper joints the
    alignment table needs) - NOT the NVIDIA SOMA asset, which is not redistributed in this repository;
  * the REAL BaseRig rest snapshot ``examples/e2e_demo/baserig_blender_rest.json``;
  * posed BaseRig frames built by an independent numpy armature FK (MOCK of Blender's pose evaluation).
One test runs real Blender (skipped when Blender or the W129 scratch inputs are absent).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import app.experimental_motionbricks as M  # noqa: E402

REST_JSON = ROOT / "examples" / "e2e_demo" / "baserig_blender_rest.json"
OBJ_REST = [1.0, -1.410649872468639e-07, 0.0, 0.0]
OBJ_SCALE = 0.009999999776482582


def _quat(axis, deg):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    h = np.radians(deg) / 2
    return np.array([np.cos(h), *(np.sin(h) * axis)])


def synthetic_soma():
    """SYNTHETIC SOMA-like T-pose skeleton (Y up, faces +Z, left = +X), identity rest rotations."""
    J = [("Hips", None, (0, 1.0, 0)), ("Spine1", "Hips", (0, .05, 0)), ("Spine2", "Spine1", (0, .07, 0)),
         ("Chest", "Spine2", (0, .07, 0)), ("Neck1", "Chest", (0, .26, 0)), ("Neck2", "Neck1", (0, .08, 0)),
         ("Head", "Neck2", (0, .07, .03)), ("HeadEnd", "Head", (0, .16, 0)), ("Jaw", "Head", (0, -.03, .05)),
         ("LeftEye", "Head", (.03, .05, .08)), ("RightEye", "Head", (-.03, .05, .08))]
    for s, x in (("Left", 1), ("Right", -1)):
        J += [(f"{s}Shoulder", "Chest", (x * .02, .23, 0)), (f"{s}Arm", f"{s}Shoulder", (x * .15, 0, 0)),
              (f"{s}ForeArm", f"{s}Arm", (x * .29, 0, 0)), (f"{s}Hand", f"{s}ForeArm", (x * .27, 0, 0)),
              (f"{s}HandThumb1", f"{s}Hand", (x * .02, 0, .03)), (f"{s}HandThumbEnd", f"{s}HandThumb1", (x * .05, 0, .02)),
              (f"{s}HandMiddle1", f"{s}Hand", (x * .09, 0, 0)), (f"{s}HandMiddleEnd", f"{s}HandMiddle1", (x * .08, 0, 0))]
    for s, x in (("Left", 1), ("Right", -1)):
        J += [(f"{s}Leg", "Hips", (x * .1, -.08, 0)), (f"{s}Shin", f"{s}Leg", (0, -.43, 0)), (f"{s}Foot", f"{s}Shin", (0, -.42, 0)),
              (f"{s}ToeBase", f"{s}Foot", (0, -.05, .13)), (f"{s}ToeEnd", f"{s}ToeBase", (0, 0, .06))]
    names = [j[0] for j in J]
    parents = [-1 if j[1] is None else names.index(j[1]) for j in J]
    ref = np.array([[*j[2], 0, 0, 0, 1] for j in J], float)
    return M.SomaSkeleton(names, parents, ref)


def base_rest_dict():
    r = json.loads(REST_JSON.read_text())
    return {n: {"parent": b["parent"], "matrix_local": b["matrix_local"], "head_local": b["head"], "tail_local": b["tail"]}
            for n, b in r["bones"].items()}


def posed_frame(rest, basis_wxyz, obj_wxyz):
    """MOCK armature FK: P_b = P_parent . (B_p^-1 B_b) . basis (4x4, armature space)."""
    P = {}

    def get(b):
        if b in P:
            return P[b]
        Bb = np.asarray(rest[b]["matrix_local"], float)
        Bas = np.eye(4)
        Bas[:3, :3] = M.wxyz_to_mat(basis_wxyz.get(b, (1, 0, 0, 0)))
        p = rest[b]["parent"]
        P[b] = (Bb @ Bas) if p is None else get(p) @ np.linalg.inv(np.asarray(rest[p]["matrix_local"], float)) @ Bb @ Bas
        return P[b]
    for b in rest:
        get(b)
    Mw = np.eye(4)
    Mw[:3, :3] = M.wxyz_to_mat(obj_wxyz) * OBJ_SCALE
    return M.BaseFrame(rest, {"matrix_world": Mw, "pose_matrix": P, "basis_wxyz": {b: list(basis_wxyz.get(b, (1, 0, 0, 0))) for b in rest}})


@pytest.fixture(scope="module")
def mapping():
    rest = base_rest_dict()
    return M.Mapping(synthetic_soma(), M.rest_frame(rest, OBJ_REST, OBJ_SCALE)), rest


def test_quaternion_matrix_roundtrip():
    rng = np.random.default_rng(0)
    for _ in range(50):
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        q2 = M.mat_to_quat_xyzw(M.quat_xyzw_to_mat(q))
        assert min(np.abs(q - q2).max(), np.abs(q + q2).max()) < 1e-12
    a, b = _quat([1, 0, 0], 10), _quat([0, 1, 0], 80)
    assert np.allclose(M.slerp_wxyz(a, b, 0.0), a) and np.allclose(M.slerp_wxyz(a, b, 1.0), b)


def test_episode_s1_time_contract():
    ep = M.episode(79, 249, 278, 9, protected_source=list(range(210, 219)))
    assert ep["valid_samples"] == 30 and ep["raw_frames"] == 36
    assert ep["context_action"] == [168, 169, 170, 171] and ep["target_action"] == [200, 201, 202, 203]
    assert ep["interior_action"] == [172, 199] and ep["interior_source"] == [250, 277]
    assert ep["valid_source"][0] == 249 and ep["valid_source"][-1] == 278 and len(ep["valid_source"]) == 30
    s3 = M.episode(0, 109, 138, 9, protected_source=list(range(41, 50)))
    assert s3["interior_action"] == [111, 138]


def test_episode_refusals():
    with pytest.raises(M.AdapterStop) as e:
        M.episode(79, 249, 280, 9)
    assert e.value.code == "UNSUPPORTED_TIME_ALIGNMENT"
    with pytest.raises(M.AdapterStop) as e:
        M.episode(79, 249, 278, 9, protected_source=[255])
    assert e.value.code == "PROTECTED_OVERLAP"


def test_rest_alignment_and_identity_roundtrip(mapping):
    mp, rest = mapping
    assert all(v["primary_residual_deg"] < 1e-4 for v in mp.align_report.values())
    L = mp.to_soma_local(mp.base_rest)
    back = mp.base_basis_from_soma(L)
    for b in M.EDIT_CHANNELS:
        assert min(np.abs(back[b] - [1, 0, 0, 0]).max(), np.abs(back[b] + [1, 0, 0, 0]).max()) < 1e-9
    assert M.no_mirroring_check(mp, L)["ok"]


def test_posed_roundtrip_both_directions(mapping):
    mp, rest = mapping
    rng = np.random.default_rng(1)
    for _ in range(20):
        basis = {b: _quat(rng.normal(size=3), rng.uniform(-80, 80)) for b in ("spine_01", "thigh_l", "calf_l", "upperarm_l", "lowerarm_l")}
        bf = posed_frame(rest, basis, _quat(rng.normal(size=3), rng.uniform(-40, 40)))
        L = mp.to_soma_local(bf)
        back = mp.base_basis_from_soma(L)
        ind = mp.basis_from_base_frame(bf)
        for b in M.EDIT_CHANNELS:
            for q in (back[b], ind[b]):
                assert min(np.abs(q - basis[b]).max(), np.abs(q + basis[b]).max()) < 1e-9
        # primary bone directions agree in SOMA space
        _, P = mp.soma.fk(L)
        for j in ("LeftArm", "LeftForeArm", "Spine1", "LeftLeg"):
            (sa, sb), (ba, bb), _ = M.ALIGN[j]
            ds = P[mp.soma.idx[sb]] - P[mp.soma.idx[sa]]
            db = mp.A @ (bf.point(bb) - bf.point(ba))
            assert np.dot(ds, db) / np.linalg.norm(ds) / np.linalg.norm(db) > 1 - 1e-9


def test_left_right_and_parent_child(mapping):
    mp, rest = mapping
    r0 = mp.to_soma_local(posed_frame(rest, {}, OBJ_REST))
    r1 = mp.to_soma_local(posed_frame(rest, {"upperarm_l": _quat([0, 0, 1], -70)}, OBJ_REST))
    _, P0 = mp.soma.fk(r0)
    _, P1 = mp.soma.fk(r1)
    i = mp.soma.idx
    assert np.linalg.norm(P1[i["LeftHand"]] - P0[i["LeftHand"]]) > 0.05
    assert np.linalg.norm(P1[i["RightHand"]] - P0[i["RightHand"]]) < 1e-9
    r2 = mp.to_soma_local(posed_frame(rest, {"spine_01": _quat([1, 0, 0], 30)}, OBJ_REST))
    q0, q2 = r0[i["LeftArm"], 3:], r2[i["LeftArm"], 3:]
    assert min(np.abs(q0 - q2).max(), np.abs(q0 + q2).max()) < 1e-9          # local arm channel invariant to its parents


def test_axis_negative_controls(mapping):
    mp, rest = mapping
    with pytest.raises(M.AdapterStop) as e:
        M.Mapping(mp.soma, mp.base_rest, A=np.diag([-1.0, 1, 1]) @ M.A_BLENDER_TO_SOMA)
    assert e.value.code == "AXIS_MAPPING_IMPROPER"
    wrong = M.Mapping(mp.soma, mp.base_rest, A=np.eye(3))                          # proper but wrong axes
    assert not M.no_mirroring_check(wrong, wrong.to_soma_local(mp.base_rest))["ok"]


def test_missing_joint_and_bone_refused(mapping):
    mp, rest = mapping
    names = list(mp.soma.names)
    names[mp.soma.idx["LeftForeArm"]] = "Nope"
    with pytest.raises(M.AdapterStop) as e:
        M.SomaSkeleton(names, mp.soma.parents, mp.soma.ref)
    assert e.value.code == "SOMA_SKELETON_MISSING_JOINTS"
    r = {k: v for k, v in rest.items() if k != "upperarm_l"}
    with pytest.raises(M.AdapterStop) as e:
        M.Mapping(mp.soma, M.rest_frame(r, OBJ_REST, OBJ_SCALE))
    assert e.value.code == "BASERIG_MISSING_BONES"


def test_coverage_labels_unknowns():
    cov = M.coverage()
    assert set(cov) == set(M.SOMA_MODEL_JOINTS)
    assert cov["RightArm"].startswith("RIG_REST") and cov["Jaw"].startswith("SYNTHETIC")
    assert cov["LeftArm"].startswith("DRIVEN") and "not observed truth" in cov["LeftArm"]
    assert "SYNTHETIC zero travel" in cov["Hips"]


def _rb(frames_world, subs=()):
    return {"frames": [{"blender_frame": f, "world": w, "matrix_world": [[1]]} for f, w in frames_world],
            "subframes": [{"blender_frame": f, "subframe": s, "world": w, "matrix_world": [[1]]} for f, s, w in subs]}


def test_preservation_and_changed_keys():
    base = _rb([(f, {"hand_l": {"head": [0, 0, f]}, "spine_01": {"head": [0, 0, 0]}}) for f in range(1, 11)])
    ok = _rb([(f, {"hand_l": {"head": [0, 1, f] if 4 <= f <= 6 else [0, 0, f]}, "spine_01": {"head": [0, 0, 0]}}) for f in range(1, 11)])
    p = M.preservation(ok, base, (4, 6))
    assert p["inside_changed_frames"] == [4, 5, 6] and not p["outside_changed"] and not p["nonarm_changed"]
    bad = _rb([(f, {"hand_l": {"head": [0, 1, f] if f == 8 else [0, 0, f]}, "spine_01": {"head": [0, 0, 1] if f == 2 else [0, 0, 0]}}) for f in range(1, 11)])
    p = M.preservation(bad, base, (4, 6))
    assert p["outside_changed"] == [[8, "hand_l"]] and p["nonarm_changed"] == [[2, "spine_01"]]
    sb = {"upperarm_l": {"0": {"5": 1.0, "9": 1.0}}}
    sv = {"upperarm_l": {"0": {"5": 0.5, "9": 1.0}}}
    assert M.changed_keys(sv, sb) == [["upperarm_l", 0, 5]]


def _arm_rb(wrists, n):
    fr = []
    for f in range(1, n + 1):
        w = wrists[f - 1]
        fr.append({"blender_frame": f, "world": {"upperarm_l": {"head": [0.0, 0.0, 1.4]}, "lowerarm_l": {"head": [0.25, 0.0, 1.2]},
                                                 "hand_l": {"head": list(w)}}, "matrix_world": [[1]]})
    return {"frames": fr, "subframes": []}


def test_quality_flick_detection():
    n = 40
    smooth = np.array([[0.25 + 0.01 * k, -0.05, 1.0 + 0.004 * k] for k in range(n)])
    flick = smooth.copy()
    flick[20] += [0.0, -0.10, 0.0]                                               # out-and-back one-frame excursion
    A = _arm_rb(smooth, n)
    qs = M.quality(A, A, (15, 30), (8, 37))
    qf = M.quality(_arm_rb(flick, n), A, (15, 30), (8, 37))
    assert qs["n_reversal_flicks"] == 0 and qf["n_reversal_flicks"] >= 1
    assert qf["wrist_dev_from_A_max_mm"] == pytest.approx(100.0, abs=1e-6)


def test_armlocal_quality_flick_and_boundary():
    n = 40
    smooth = np.array([[0.01 * k, 0.0, 0.0] for k in range(n)])
    flick = smooth.copy()
    flick[20] += [0.0, 0.1, 0.0]
    qs = M.armlocal_quality(smooth, (15, 30), (1, n))
    qf = M.armlocal_quality(flick, (15, 30), (1, n))
    assert qs["n_reversal_flicks"] == 0 and qf["n_reversal_flicks"] >= 1
    assert qs["boundary_c0"]["entry_step_mm"] == pytest.approx(10.0)


def test_armlocal_wrist_factors_out_parent_motion():
    pc = np.eye(4)
    ph = np.eye(4)
    ph[:3, 3] = [10.0, 0, 0]
    rot = np.eye(4)
    rot[:3, :3] = M.wxyz_to_mat(_quat([0, 0, 1], 50))
    rep = {"frames": [{"blender_frame": 1, "pose_matrix": {"clavicle_l": pc.tolist(), "hand_l": ph.tolist()}},
                      {"blender_frame": 2, "pose_matrix": {"clavicle_l": (rot @ pc).tolist(), "hand_l": (rot @ ph).tolist()}}]}
    w = M.armlocal_wrist(rep, 1, 2)
    assert np.allclose(w[0], w[1]) and np.allclose(w[0], [0.1, 0, 0])


def test_default_paths_do_not_reference_slice():
    for f in ("src/app/cli.py", "src/app/gui.py"):
        assert "experimental_motionbricks" not in (ROOT / f).read_text(encoding="utf-8")


BLENDER = os.environ.get("ANIMCV_BLENDER", "C:/Program Files/Blender Foundation/Blender 5.1/blender.exe")
S1_COPY = Path("C:/Users/dna10/animcv-w129/inputs/S1_baseline.blend")


@pytest.mark.skipif(not (Path(BLENDER).exists() and S1_COPY.exists()), reason="real Blender / W129 scratch input not available")
def test_real_blender_build_variant_refuses_out_of_domain(tmp_path):
    out_blend = tmp_path / "x.blend"
    spec = {"out_blend": str(out_blend), "interior_action_frames": [172, 199], "values_wxyz": {"spine_01": {"180": [1, 0, 0, 0]}},
            "branch": "T", "action_name": "T"}
    (tmp_path / "spec.json").write_text(json.dumps(spec))
    r = subprocess.run([BLENDER, "-b", "--python-exit-code", "1", "-P", str(ROOT / "src/blender/experimental_motionbricks_stage.py"), "--",
                        "build_variant", "--blend", str(S1_COPY), "--spec", str(tmp_path / "spec.json"), "--report", str(tmp_path / "r.json")],
                       capture_output=True, text=True)
    assert r.returncode != 0 and "REFUSED_CHANNEL_OUTSIDE_DOMAIN" in (r.stdout + r.stderr) and not out_blend.exists()
