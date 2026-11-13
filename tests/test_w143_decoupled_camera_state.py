"""W143 focused tests: decoupled orientation validation / translation observability / 6DoF availability.

W140 (tests/test_experimental_camera_pose.py), W141 (tests/test_w141_dynamic_camera_pose.py) and W142
(tests/test_w142_background_rotation.py) tests are reused unchanged.
"""
from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from app import experimental_background_rotation as BR  # noqa: E402
from app import experimental_camera_pose as CP  # noqa: E402
from app import experimental_decoupled_camera_state as DC  # noqa: E402

cv2 = pytest.importorskip("cv2")
K = np.array([[535.4, 0, 320.1], [0, 539.2, 247.6], [0, 0, 1.0]])


def rot(axis, deg):
    from scipy.spatial.transform import Rotation
    a = np.asarray(axis, float)
    return Rotation.from_rotvec(np.radians(deg) * a / np.linalg.norm(a)).as_matrix()


def scene(n=1200, near=1.0, far=5.0, planar=False, seed=0):
    rng = np.random.default_rng(seed)
    if planar:
        return np.column_stack([rng.uniform(-2, 2, n), rng.uniform(-1.5, 1.5, n), np.full(n, 3.0)])
    z = rng.uniform(near, far, n)
    return np.column_stack([rng.uniform(-0.8, 0.8, n) * z, rng.uniform(-0.6, 0.6, n) * z, z])


def views(X, R, t, noise=0.4, k1=0.0, seed=7):
    x1, z1 = CP.project(K, CP.make_T(np.eye(3), [0, 0, 0]), X)
    x2, z2 = CP.project(K, CP.make_T(R, t), X)
    if k1:
        def dist(x):
            u = (x - K[:2, 2]) / np.array([K[0, 0], K[1, 1]])
            return u * (1 + k1 * np.sum(u * u, 1, keepdims=True)) * np.array([K[0, 0], K[1, 1]]) + K[:2, 2]
        x1, x2 = dist(x1), dist(x2)
    ok = (z1 > 0) & (z2 > 0) & np.all((x1 > 0) & (x1 < [640, 480]), 1) & np.all((x2 > 0) & (x2 < [640, 480]), 1)
    rng = np.random.default_rng(seed)
    return x1[ok] + rng.normal(0, noise, (ok.sum(), 2)), x2[ok] + rng.normal(0, noise, (ok.sum(), 2))


def edge(R, t=(0, 0, 0), **kw):
    planar = kw.pop("planar", False)
    near = kw.pop("near", 1.0)
    far = kw.pop("far", 5.0)
    x1, x2 = views(scene(planar=planar, near=near, far=far), R, t, **kw)
    e = DC.analyse_edge(K, x1, x2, seed=1)
    e.pop("_inl")
    return e


# ------------------------------------------------------------------------------------------------ orientation validation ---
def test_homography_rotation_recovers_pure_rotation_and_deviates_for_planar_translation():
    R = rot([0.2, 1, 0.1], 3.0)
    assert CP.geodesic_deg(DC.homography_rotation(K, K @ R @ np.linalg.inv(K)), R) < 1e-6
    H = K @ (np.eye(3) + np.outer([0.2, 0, 0], [0, 0, 1]) / 3.0) @ np.linalg.inv(K)       # translation 0.2 over a plane at 3 m
    assert CP.geodesic_deg(DC.homography_rotation(K, H), np.eye(3)) > 1.0
    assert DC.homography_rotation(K, None) is None


def test_pure_rotation_edge_accepted_and_accurate():
    R = rot([0.3, 1, 0.1], 3.0)
    e = edge(R)
    assert e["rotation_candidate"] and e["fit_model"] == "R"
    assert CP.geodesic_deg(np.asarray(e["M"]), R.T) < 0.02 and e["translation_edge"] == "UNRESOLVED"


def test_homography_preferred_but_rotation_still_valid_is_decoupled():
    """Residual radial distortion makes the homography win GRIC, yet the rotation measurement is accurate and accepted."""
    R = rot([0.3, 1, 0.1], 3.0)
    e = edge(R, k1=0.05)
    assert e["fit_model"] == "H" and e["rotation_candidate"]
    assert CP.geodesic_deg(np.asarray(e["M"]), R.T) < 0.1
    assert e["translation_edge"] == "UNRESOLVED"                      # homography preference never implies translation


def test_planar_translation_rotation_rejected_and_not_translation_observed():
    e = edge(np.eye(3), t=(0.2, 0, 0), planar=True)
    assert e["fit_model"] == "H" and not e["rotation_candidate"]
    assert "rotation_vs_homography" in e["reject"] and e["translation_edge"] != "DIRECTION_CANDIDATE"


def test_small_translation_bias_accepted_small_and_rejected_large():
    R = rot([0.3, 1, 0.1], 0.5)
    small = edge(R, t=(0.002, 0, 0))
    assert small["rotation_candidate"] and CP.geodesic_deg(np.asarray(small["M"]), R.T) < 0.1
    large = edge(R, t=(0.03, 0, 0))
    assert not large["rotation_candidate"] and {"rotation_support", "rotation_residual"} & set(large["reject"])
    assert large["rotation_bias_bound_deg"] > small["rotation_bias_bound_deg"]


def test_general_translation_gives_direction_candidate_and_rotation_rejected():
    e = edge(rot([0, 1, 0], 2.0), t=(0.3, 0, 0.05))
    assert e["translation_edge"] == "DIRECTION_CANDIDATE" and not e["rotation_candidate"] and e["parallax_px"] >= DC.PARALLAX_MIN_PX
    d = np.asarray(e["translation_direction_in_cam_i"])
    t = np.array([0.3, 0, 0.05]) / np.linalg.norm([0.3, 0, 0.05])
    assert math.degrees(math.acos(np.clip(d @ t, -1, 1))) < 5


def test_too_few_matches_insufficient():
    e = DC.analyse_edge(K, np.zeros((5, 2)), np.zeros((5, 2)), seed=0)
    assert e["translation_edge"] == "INSUFFICIENT" and not e["rotation_candidate"]


# ------------------------------------------------------------------------------------------------ cycles / translation confirmation ---
def _chain(n=12, corrupt=None):
    R = [rot([0, 1, 0], 0.5 * k) for k in range(n)]
    edges = []
    for i in range(n):
        for s in (1, 2, 4):
            j = i + s
            if j < n:
                M = R[j].T @ R[i]
                if corrupt == (i, j):
                    M = rot([1, 0, 0], 3.0) @ M
                edges.append({"i": i, "j": j, "span": s, "M": M.tolist(), "rotation_candidate": True})
    return edges


def test_cycle_consistency_flags_corrupted_edge_only():
    edges = _chain(corrupt=(4, 6))
    sc = DC.cycle_scores(edges)
    med = {k: float(np.median(v)) for k, v in sc.items() if v}
    assert med[(4, 6)] > DC.CYCLE_MAX_DEG
    assert sum(v > DC.CYCLE_MAX_DEG for v in med.values()) > 1          # naive scoring also flags innocent triangle partners
    rej, final = DC.cycle_reject(edges)
    assert rej == {(4, 6)}                                              # greedy removal isolates the corrupted edge
    assert all(float(np.median(v)) < 1e-4 for v in final.values() if v)            # arccos precision
    assert DC.cycle_reject(_chain())[0] == set()


def test_translation_confirmation_across_spans():
    d = [1.0, 0, 0]
    edges = [{"i": 0, "j": 1, "translation_edge": "DIRECTION_CANDIDATE", "translation_direction_in_cam_i": d},
             {"i": 0, "j": 2, "translation_edge": "DIRECTION_CANDIDATE", "translation_direction_in_cam_i": [0.98, 0.2, 0]},
             {"i": 3, "j": 4, "translation_edge": "DIRECTION_CANDIDATE", "translation_direction_in_cam_i": d},
             {"i": 3, "j": 5, "translation_edge": "DIRECTION_CANDIDATE", "translation_direction_in_cam_i": [0, 1.0, 0]},
             {"i": 6, "j": 7, "translation_edge": "UNRESOLVED"}]
    DC.confirm_translation(edges)
    assert [e["translation_edge"] for e in edges] == ["DIRECTION_OBSERVED", "DIRECTION_OBSERVED", "UNRESOLVED_UNCONFIRMED",
                                                      "UNRESOLVED_UNCONFIRMED", "UNRESOLVED"]


def test_translation_window_semantics():
    t = np.arange(10) * 0.1
    obs = [{"i": i, "j": i + 1, "translation_edge": "DIRECTION_OBSERVED", "rotation_accepted": False, "rotation_deg": 0.0} for i in range(9)]
    w = DC.frame_translation_window(t, obs)
    assert w[5]["translation"] == "TRANSLATION_DIRECTION_ONLY" and w[5]["regime"] == "TRANSLATING"
    un = [{"i": i, "j": i + 1, "translation_edge": "UNRESOLVED", "rotation_accepted": True, "rotation_deg": 0.01} for i in range(9)]
    w = DC.frame_translation_window(t, un)
    assert w[5]["translation"] == "TRANSLATION_UNRESOLVED" and w[5]["regime"] == "NEAR_STATIC_OR_BELOW_DETECTABILITY"
    assert DC.frame_translation_window(t, [])[0]["translation"] == "TRANSLATION_INSUFFICIENT_EVIDENCE"


# ------------------------------------------------------------------------------------------------ SfM validity / routing ---
def _sfm(n=10, reg=10, landmarks=5000, reproj=0.8, obs=500):
    return {"primary_model_id": 0, "models": [{"model_id": 0, "n_points3D": landmarks, "mean_reproj_px": reproj}],
            "frames": [{"frame_id": k, "status": "ESTIMATED" if k < reg else "LOST", "T_WC": np.eye(4).tolist() if k < reg else None,
                        "n_points3D_observed": obs} for k in range(n)]}


def test_sfm_validity_from_reconstruction_evidence_only():
    assert DC.sfm_validity(_sfm())["model_valid"]
    assert not DC.sfm_validity(_sfm(landmarks=18))["model_valid"]                    # W141 walking_static_B-like
    assert not DC.sfm_validity(_sfm(reg=3))["model_valid"]
    assert not DC.sfm_validity(_sfm(reproj=3.0))["model_valid"]
    v = DC.sfm_validity(_sfm(obs=50))
    assert v["model_valid"] and not any(f["valid"] for f in v["frames"].values())
    assert DC.sfm_validity(None)["status"] == "SFM_NOT_AVAILABLE"


def test_camera_state_never_upgraded_by_translation_label():
    win_t = {"translation": "TRANSLATION_DIRECTION_ONLY"}
    assert DC.camera_state(True, None, win_t) == ("ORIENTATION_ONLY_AVAILABLE", "TRANSLATION_DIRECTION_ONLY")
    assert DC.camera_state(True, {"valid": False}, win_t)[0] == "ORIENTATION_ONLY_AVAILABLE"
    assert DC.camera_state(False, {"valid": True}, win_t) == ("FULL_6DOF_AVAILABLE", "TRANSLATION_ESTIMATED_SCALE_AMBIGUOUS")
    assert DC.camera_state(False, None, {"translation": "TRANSLATION_UNRESOLVED"})[0] == "CAMERA_MOTION_UNRESOLVED"


# ------------------------------------------------------------------------------------------------ estimator integration ---
def _texture(seed, h, w):
    rng = np.random.default_rng(seed)
    img = np.full((h, w, 3), 128, np.uint8)
    for _ in range(int(h * w / 340)):
        cv2.circle(img, (int(rng.integers(0, w)), int(rng.integers(0, h))), int(rng.integers(3, 14)), tuple(int(v) for v in rng.integers(0, 255, 3)), -1)
    return img


def _sequence(tmp, n=8, leak=False):
    tex = _texture(5, 900, 1200)
    ind = tmp / "in"
    (ind / "images").mkdir(parents=True)
    md = tmp / "keep"
    md.mkdir()
    Kp = np.array([[500.0, 0, 319.5], [0, 500.0, 239.5], [0, 0, 1]])
    Kt = np.array([[500.0, 0, 599.5], [0, 500.0, 449.5], [0, 0, 1]])
    Rs, frames = [], []
    for k in range(n):
        R_WC = rot([0, 1, 0], 0.6 * k) @ rot([1, 0, 0], 0.25 * k)
        Rs.append(R_WC)
        img = cv2.warpPerspective(tex, Kp @ R_WC.T @ np.linalg.inv(Kt), (640, 480))
        nm = f"f{k:04d}_{50 + 0.1 * k:.6f}.png"
        cv2.imwrite(str(ind / "images" / nm), img)
        keep = np.full((480, 640), 255, np.uint8)
        keep[180:300, 260:380] = 0                                   # an excluded 'person' box
        cv2.imwrite(str(md / (nm + ".png")), keep)
        frames.append({"frame_id": 100 + k, "timestamp": 50 + 0.1 * k, "image": nm})
    json.dump(frames, open(ind / "frames.json", "w"))
    json.dump({"model": "PINHOLE", "width": 640, "height": 480, "fx": 500.0, "fy": 500.0, "cx": 319.5, "cy": 239.5}, open(ind / "camera.json", "w"))
    json.dump({}, open(ind / "input_manifest.json", "w"))
    if leak:
        (ind / "groundtruth.txt").write_text("0 0 0 0 0 0 0 1\n")
    return ind, md, Rs, frames


def _run(ind, md, out, sfm=None):
    cmd = [sys.executable, "-I", str(ROOT / "scripts" / "w143_decoupled_estimate.py"), "--input", str(ind), "--out", str(out),
           "--keep-masks", str(md), "--mask-label", "rgbmask"] + (["--sfm", str(sfm)] if sfm else [])
    return subprocess.run(cmd, capture_output=True, text=True)


def test_estimator_end_to_end_orientation_only_and_null_translation(tmp_path):
    ind, md, Rs, frames = _sequence(tmp_path)
    r = _run(ind, md, tmp_path / "out")
    assert r.returncode == 0, r.stderr[-2000:]
    est = json.load(open(tmp_path / "out" / "estimate.json"))
    assert [f["frame_id"] for f in est["frames"]] == [f["frame_id"] for f in frames]
    assert [f["timestamp"] for f in est["frames"]] == [f["timestamp"] for f in frames]
    assert est["kp_on_excluded_total"] == 0
    for f, R in zip(est["frames"], Rs):
        assert f["orientation"]["status"] == "ORIENTATION_ESTIMATED" and f["camera_state"] == "ORIENTATION_ONLY_AVAILABLE"
        assert f["translation"]["value"] is None and f["translation"]["status"] != "TRANSLATION_ESTIMATED_SCALE_AMBIGUOUS"
        assert CP.geodesic_deg(np.asarray(f["orientation"]["R_WC"]), Rs[0].T @ R) < 0.1
    assert est["counts"]["numeric_translation_without_valid_sfm"] == 0


def test_estimator_full_6dof_only_from_valid_sfm(tmp_path):
    ind, md, Rs, frames = _sequence(tmp_path, n=6)
    good = _sfm(n=0)
    good["frames"] = [{"frame_id": f["frame_id"], "status": "ESTIMATED", "T_WC": CP.make_T(np.eye(3), [0.1 * k, 0, 0]).tolist(), "n_points3D_observed": 400}
                      for k, f in enumerate(frames)]
    p = tmp_path / "sfm_good.json"
    p.write_text(json.dumps(good))
    r = _run(ind, md, tmp_path / "o1", p)
    assert r.returncode == 0, r.stderr[-2000:]
    est = json.load(open(tmp_path / "o1" / "estimate.json"))
    assert all(f["camera_state"] == "FULL_6DOF_AVAILABLE" and f["translation"]["scale_status"] == "SCALE_AMBIGUOUS" for f in est["frames"])
    bad = dict(good)
    bad["models"] = [{"model_id": 0, "n_points3D": 18, "mean_reproj_px": 0.1}]
    p2 = tmp_path / "sfm_bad.json"
    p2.write_text(json.dumps(bad))
    r = _run(ind, md, tmp_path / "o2", p2)
    est = json.load(open(tmp_path / "o2" / "estimate.json"))
    assert all(f["camera_state"] != "FULL_6DOF_AVAILABLE" and f["translation"]["value"] is None for f in est["frames"])


def test_estimator_refuses_gt_leak_and_gt_as_sfm(tmp_path):
    ind, md, _, _ = _sequence(tmp_path / "a", n=3, leak=True)
    assert _run(ind, md, tmp_path / "a" / "out").returncode != 0
    ind, md, _, _ = _sequence(tmp_path / "b", n=3)
    g = tmp_path / "b" / "groundtruth.json"
    g.write_text(json.dumps(_sfm()))
    assert _run(ind, md, tmp_path / "b" / "out", g).returncode != 0


def test_harness_held_out_isolation_and_runs():
    import importlib.util
    spec = importlib.util.spec_from_file_location("w143h", ROOT / "scripts" / "w143_harness.py")
    H = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(H)
    assert [r[1] for r in H.RUNS143] == list(H.DATASETS) and "sitting_static" in H.DATASETS
    assert str(H.SFM_SRC["sitting_static"]).startswith("NEW:") and H.SFM_SRC["walking_static"].name == "estimate.json"
    src = (ROOT / "scripts" / "w143_decoupled_estimate.py").read_text()
    assert "read_groundtruth" not in src and "groundtruth.txt" not in src
