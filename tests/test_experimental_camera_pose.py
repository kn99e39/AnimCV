"""W140 focused tests (synthetic / no estimator run).  Independent references: scipy Rotation, cv2.projectPoints."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

import app.experimental_camera_pose as CP

Rot = pytest.importorskip("scipy.spatial.transform").Rotation
ROOT = Path(__file__).resolve().parents[1]


def rand_T(rng):
    return CP.make_T(Rot.random(random_state=int(rng.integers(1e9))).as_matrix(), rng.normal(size=3))


# ------------------------------------------------------------------------------------------------ SE(3) / quaternions ---
def test_inverse_and_composition():
    rng = np.random.default_rng(0)
    A, B = rand_T(rng), rand_T(rng)
    assert np.allclose(CP.inv_T(A) @ A, np.eye(4), atol=1e-12)
    assert np.allclose(CP.inv_T(A @ B), CP.inv_T(B) @ CP.inv_T(A), atol=1e-12)
    assert np.allclose(np.linalg.inv(A), CP.inv_T(A), atol=1e-12)


def test_camera_center_vs_extrinsic_translation():
    rng = np.random.default_rng(1)
    T_WC = rand_T(rng)
    T_CW = CP.inv_T(T_WC)
    assert np.allclose(CP.camera_center_from_T_CW(T_CW), T_WC[:3, 3])
    assert not np.allclose(T_CW[:3, 3], T_WC[:3, 3])                       # extrinsic t != centre


def test_quaternion_xyzw_matches_scipy_and_normalises():
    rng = np.random.default_rng(2)
    for _ in range(50):
        r = Rot.random(random_state=int(rng.integers(1e9)))
        q = r.as_quat()                                                      # scipy is scalar-last xyzw
        assert np.allclose(CP.quat_xyzw_to_R(q), r.as_matrix(), atol=1e-12)
        assert np.allclose(CP.quat_xyzw_to_R(3.7 * q), r.as_matrix(), atol=1e-12)
        q2 = CP.R_to_quat_xyzw(r.as_matrix())
        assert np.allclose(np.abs(np.dot(q2, q)), 1.0, atol=1e-12)
    with pytest.raises(CP.CamStop):
        CP.quat_xyzw_to_R([0, 0, 0, 0])


def test_swapped_quaternion_order_is_detectable():
    r = Rot.from_euler("xyz", [30, -20, 50], degrees=True)
    q = r.as_quat()
    wrong = CP.quat_xyzw_to_R([q[3], q[0], q[1], q[2]])                     # scalar-first misread
    assert CP.geodesic_deg(wrong, r.as_matrix()) > 10


def test_blender_to_optical_axes():
    Rb = np.eye(3)                                                           # Blender camera looking down world -Z, +Y up
    Ro = CP.blender_cam_to_optical(Rb)
    assert np.allclose(Ro @ [0, 0, 1], [0, 0, -1]) and np.allclose(Ro @ [0, 1, 0], [0, -1, 0]) and CP.is_rotation(Ro)


# ------------------------------------------------------------------------------------------------ projection ---
def test_projection_matches_cv2_and_detects_bad_focal_axis():
    cv2 = pytest.importorskip("cv2")
    rng = np.random.default_rng(3)
    K = np.array([[517.3, 0, 318.6], [0, 516.5, 255.3], [0, 0, 1.0]])
    T_WC = CP.make_T(Rot.from_euler("xyz", [5, -10, 3], degrees=True).as_matrix(), [0.2, -0.1, -2.0])
    T_CW = CP.inv_T(T_WC)
    X = rng.uniform(-1, 1, (20, 3))
    uv, z = CP.project(K, T_WC, X)
    rv, _ = cv2.Rodrigues(T_CW[:3, :3])
    ref, _ = cv2.projectPoints(X, rv, T_CW[:3, 3], K, None)
    assert np.all(z > 0) and np.allclose(uv, ref[:, 0], atol=1e-9)
    K2 = K.copy()
    K2[0, 0] *= 1.1
    assert np.abs(CP.project(K2, T_WC, X)[0] - uv).max() > 1
    flip = T_WC @ CP.make_T(np.diag([1.0, -1.0, -1.0]), [0, 0, 0])
    _, zf = CP.project(K, flip, X)
    assert np.all(zf < 0)                                                    # axis-sign error puts the scene behind the camera


@pytest.mark.parametrize("K", [np.array([[0, 0, 320], [0, 500, 240], [0, 0, 1.0]]), np.array([[500, 0, 900], [0, 500, 240], [0, 0, 1.0]]),
                               np.array([[500, 1, 320], [0, 500, 240], [0, 0, 1.0]]), np.full((3, 3), np.nan)])
def test_invalid_calibration_refused(K):
    with pytest.raises(CP.CamStop):
        CP.check_intrinsics(K, 640, 480)


# ------------------------------------------------------------------------------------------------ TUM I/O and association ---
def write(tmp_path, name, lines):
    p = tmp_path / name
    p.write_text("\n".join(lines) + "\n")
    return p


def test_malformed_trajectory_refused(tmp_path):
    ok = ["# c", "1.0 0 0 0 0 0 0 1", "1.01 0 0 0 0 0 0 1"]
    assert CP.read_groundtruth(write(tmp_path, "a.txt", ok)).shape == (2, 8)
    for bad in (["1.0 0 0 0 0 0 1"], ["1.0 0 0 0 0 0 0 1", "0.9 0 0 0 0 0 0 1"], ["1.0 0 0 0 0 0 0 2"], ["1.0 0 0 nan 0 0 0 1"]):
        with pytest.raises(CP.CamStop):
            CP.read_groundtruth(write(tmp_path, "b.txt", bad))


def test_interpolation_and_gap_refusal():
    R0, R1 = np.eye(3), Rot.from_euler("z", 20, degrees=True).as_matrix()
    gt = np.array([[10.00, 0, 0, 0, *Rot.from_matrix(R0).as_quat()], [10.01, 1, 0, 0, *Rot.from_matrix(R1).as_quat()],
                   [10.50, 2, 0, 0, *Rot.from_matrix(R1).as_quat()]])
    T, info = CP.interpolate_gt(gt, 10.0025)
    assert np.allclose(T[:3, 3], [0.25, 0, 0]) and abs(CP.geodesic_deg(T[:3, :3], Rot.from_euler("z", 5, degrees=True).as_matrix())) < 1e-9
    assert CP.interpolate_gt(gt, 10.2)[0] is None                            # bracket gap 0.49 s > 0.05 s
    assert CP.interpolate_gt(gt, 9.0)[0] is None and CP.interpolate_gt(gt, 11.0)[0] is None


def test_deterministic_uniform_selection():
    a, b = CP.select_uniform(798), CP.select_uniform(798)
    assert a == b and len(a) == 300 and a[0] == 0 and a[-1] == 797 and a == sorted(set(a))
    assert CP.select_uniform(120) == list(range(120))


# ------------------------------------------------------------------------------------------------ alignment / metrics ---
def traj(n=60, seed=4):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 6, n)
    P = np.stack([np.sin(t), 0.5 * np.cos(0.7 * t), 0.2 * t], 1)
    Rs = [Rot.from_euler("xyz", [10 * np.sin(x), 20 * np.cos(x), 5 * x], degrees=True).as_matrix() for x in t]
    gt = np.array([[1000 + x, *p, *Rot.from_matrix(R).as_quat()] for x, p, R in zip(t, P, Rs)])
    return gt, P, Rs, rng


def est_from(gt_rows, s, Rg, tg, noise=0.0, rng=None, invert=False):
    """Estimate in an arbitrary gauge: X_est = (R^T (X_gt - t)) / s, i.e. exactly Sim(3)-related to GT."""
    out = []
    for i, r in enumerate(gt_rows):
        T = CP.T_WC_from_tum(r[1:4], r[4:8])
        P = Rg.T @ (T[:3, 3] - tg) / s + (rng.normal(scale=noise, size=3) if noise else 0)
        Te = CP.make_T(Rg.T @ T[:3, :3], P)
        if invert:
            Te = CP.inv_T(Te)
        out.append({"frame_id": i, "timestamp": float(r[0]), "status": "ESTIMATED", "T_WC": Te.tolist()})
    return out


def gt_dense(gt):
    """GT sampled densely (100 Hz) so that every estimate timestamp is bracketed within the gap tolerance."""
    t = np.arange(gt[0, 0] - 0.05, gt[-1, 0] + 0.05, 0.01)
    rows = []
    for x in t:
        j = int(np.clip(np.searchsorted(gt[:, 0], x), 1, len(gt) - 1))
        w = np.clip((x - gt[j - 1, 0]) / (gt[j, 0] - gt[j - 1, 0]), 0, 1)
        p = (1 - w) * gt[j - 1, 1:4] + w * gt[j, 1:4]
        R = CP.slerp_R(CP.quat_xyzw_to_R(gt[j - 1, 4:8]), CP.quat_xyzw_to_R(gt[j, 4:8]), w)
        rows.append([x, *p, *CP.R_to_quat_xyzw(R)])
    return np.asarray(rows)


def test_sim3_recovers_gauge_and_scale_exactly():
    gt, P, Rs, rng = traj()
    Rg, tg, s = Rot.from_euler("xyz", [40, -30, 70], degrees=True).as_matrix(), np.array([3.0, -1.0, 2.0]), 0.37
    ev = CP.evaluate(est_from(gt, s, Rg, tg), gt_dense(gt))
    f = ev["full_sim3"]
    assert f["ate"]["rmse"] < 1e-3 and f["rot"]["max_deg"] < 0.05 and abs(f["sim3"]["scale"] - s) < 1e-3
    assert f["time_lag"]["best_lag_frames"] == 0 and ev["coverage_estimated"] == 1.0


def test_metric_vs_scale_ambiguous():
    gt, P, Rs, rng = traj()
    X = np.asarray([r[1:4] for r in gt])
    s, R, t = CP.umeyama_sim3(X * 0.3, X)
    assert abs(s - 1 / 0.3) < 1e-9
    s2, R2, t2 = CP.umeyama_sim3(X * 0.3, X, with_scale=False)
    assert s2 == 1.0 and np.linalg.norm((X * 0.3) @ R2.T + t2 - X, axis=1).mean() > 0.1    # SE(3)-only cannot absorb scale


def test_known_noise_gives_known_ate():
    gt, P, Rs, rng = traj(n=400)
    ev = CP.evaluate(est_from(gt, 1.0, np.eye(3), np.zeros(3), noise=0.01, rng=rng), gt_dense(gt))
    assert abs(ev["full_sim3"]["ate"]["rmse"] - 0.01 * np.sqrt(3)) < 0.003


def test_negative_inverted_transforms_and_time_shift():
    gt, P, Rs, rng = traj()
    good = CP.evaluate(est_from(gt, 1.0, np.eye(3), np.zeros(3)), gt_dense(gt))
    inv = CP.evaluate(est_from(gt, 1.0, np.eye(3), np.zeros(3), invert=True), gt_dense(gt))
    assert inv["full_sim3"]["ate"]["rmse"] > 50 * max(good["full_sim3"]["ate"]["rmse"], 1e-6)
    shifted = est_from(gt, 1.0, np.eye(3), np.zeros(3))
    for k in range(len(shifted) - 1):                                         # poses carry the NEXT frame's timestamp
        shifted[k]["timestamp"] = shifted[k + 1]["timestamp"]
    shifted = shifted[:-1]
    ev = CP.evaluate(shifted, gt_dense(gt))
    assert ev["full_sim3"]["time_lag"]["best_lag_frames"] != 0


def test_insufficient_and_degenerate_estimates():
    gt, P, Rs, rng = traj()
    est = est_from(gt, 1.0, np.eye(3), np.zeros(3))
    for e in est[2:]:
        e["status"], e["T_WC"] = "LOST", None
    ev = CP.evaluate(est, gt_dense(gt))
    assert "uncomputable" in ev and ev["coverage_estimated"] < 0.1
    with pytest.raises(CP.CamStop):
        CP.umeyama_sim3(np.zeros((5, 3)), np.random.default_rng(0).normal(size=(5, 3)))
    g = CP.real_gates("xyz", ev)
    assert g["pass"] is False and "uncomputable" in g


def test_stationary_baseline_and_gates():
    gt, P, Rs, rng = traj()
    ev = CP.evaluate(est_from(gt, 1.0, np.eye(3), np.zeros(3)), gt_dense(gt))
    b = ev["baseline_stationary"]
    assert b["ate_rmse"] > ev["full_sim3"]["ate"]["rmse"] and b["rot_median_deg"] > ev["full_sim3"]["rot"]["median_deg"]
    assert CP.real_gates("xyz", ev)["beats_baseline"] and CP.real_gates("rpy", ev)["beats_baseline"]


def test_synthetic_stationary_gate_detects_invented_motion():
    gt, P, Rs, rng = traj()
    gt[:10, 1:] = gt[0, 1:]                                                   # first 10 frames truly stationary
    est = est_from(gt, 1.0, np.eye(3), np.zeros(3))
    ev = CP.evaluate(est, gt_dense(gt))
    assert CP.synthetic_gates(ev, list(range(10)), True)["stationary_ok"]
    for k in range(10):                                                       # invent 5 cm wobble
        T = np.asarray(est[k]["T_WC"])
        T[:3, 3] += [0.05 * (-1) ** k, 0, 0]
        est[k]["T_WC"] = T.tolist()
    ev2 = CP.evaluate(est, gt_dense(gt))
    assert not CP.synthetic_gates(ev2, list(range(10)), True)["stationary_ok"]


def test_judgment_mapping():
    assert CP.judgment(True, True, True)["judgment"] == "SUPPORTED"
    assert CP.judgment(True, False, False)["judgment"] == "PARTIALLY_SUPPORTED"
    assert CP.judgment(False, False, True)["judgment"] == "PARTIALLY_SUPPORTED"
    assert CP.judgment(False, False, False)["judgment"] == "NOT_YET_SUPPORTED"
    assert CP.judgment(True, True, True, blocked="x")["judgment"] == "BLOCKED"


# ------------------------------------------------------------------------------------------------ GT leakage guard ---
def test_estimator_refuses_gt_depth_or_extra_inputs(tmp_path):
    sys.path.insert(0, str(ROOT / "scripts"))
    import w140_colmap_estimate as E
    d = tmp_path / "in"
    (d / "images").mkdir(parents=True)
    (d / "images" / "a.png").write_bytes(b"x")
    (d / "frames.json").write_text("[]")
    (d / "camera.json").write_text("{}")
    E.refuse_leaks(d)
    for extra in ("groundtruth.txt", "depth.txt", "accelerometer.txt", "notes.txt"):
        (d / extra).write_text("1")
        with pytest.raises(SystemExit):
            E.refuse_leaks(d)
        (d / extra).unlink()
    (d / "images" / "gt_pose.txt").write_text("1")
    with pytest.raises(SystemExit):
        E.refuse_leaks(d)
