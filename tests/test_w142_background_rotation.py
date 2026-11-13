"""W142 focused tests: background rotation / motion-state estimator (geometry, observability, masks, contracts, evaluation).

The W140 (tests/test_experimental_camera_pose.py) and W141 (tests/test_w141_dynamic_camera_pose.py) tests are reused unchanged.
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

cv2 = pytest.importorskip("cv2")
K = np.array([[500.0, 0, 319.5], [0, 500.0, 239.5], [0, 0, 1]])


def rot(axis, deg):
    from scipy.spatial.transform import Rotation
    return Rotation.from_rotvec(np.radians(deg) * np.asarray(axis, float) / np.linalg.norm(axis)).as_matrix()


def scene_points(n=600, seed=0, near=3.0, far=9.0, planar=False):
    rng = np.random.default_rng(seed)
    if planar:
        X = np.column_stack([rng.uniform(-4, 4, n), rng.uniform(-3, 3, n), np.full(n, 5.0)])
    else:
        X = np.column_stack([rng.uniform(-4, 4, n), rng.uniform(-3, 3, n), rng.uniform(near, far, n)])
    return X


def views(X, T1, T2, noise=0.3, seed=1):
    rng = np.random.default_rng(seed)
    x1, z1 = CP.project(K, T1, X)
    x2, z2 = CP.project(K, T2, X)
    ok = (z1 > 0) & (z2 > 0) & (x1[:, 0] > 0) & (x1[:, 0] < 640) & (x2[:, 0] > 0) & (x2[:, 0] < 640) & \
        (x1[:, 1] > 0) & (x1[:, 1] < 480) & (x2[:, 1] > 0) & (x2[:, 1] < 480)
    return x1[ok] + rng.normal(0, noise, (ok.sum(), 2)), x2[ok] + rng.normal(0, noise, (ok.sum(), 2))


# ------------------------------------------------------------------------------------------------ rotation geometry ---
def test_kabsch_and_rotation_convention_M_is_R_CjCi():
    R_WC1 = rot([0, 1, 0], 10) @ rot([1, 0, 0], 4)
    R_WC2 = R_WC1 @ rot([0.2, 1, 0.1], 3.0)                    # camera 2 rotated w.r.t. camera 1, same centre
    T1, T2 = CP.make_T(R_WC1, [0, 0, 0]), CP.make_T(R_WC2, [0, 0, 0])
    X = scene_points() @ R_WC1.T                                # points in front of camera 1
    x1, x2 = views(X, T1, T2, noise=0.0)
    R, inl = BR.rotation_ransac(K, x1, x2, seed=0)
    assert inl.all()
    M_true = R_WC2.T @ R_WC1                                    # R_C2C1
    assert CP.geodesic_deg(R, M_true) < 1e-6
    assert CP.geodesic_deg(R_WC1 @ R.T, R_WC2) < 1e-6          # R_WCj = R_WCi M^T


def test_optical_flow_direction_for_rightward_pan():
    """Camera pans right (rotation about optical +Y moves +Z towards +X): static points flow LEFT in the image."""
    R_WC2 = rot([0, 1, 0], 5.0)
    X = scene_points()
    x1, x2 = views(X, CP.make_T(np.eye(3), [0, 0, 0]), CP.make_T(R_WC2, [0, 0, 0]), noise=0.0)
    assert np.median(x2[:, 0] - x1[:, 0]) < -30


def test_rotation_robust_to_incoherent_outliers():
    R_WC2 = rot([0, 1, 0], 2.0)
    X = scene_points(800)
    x1, x2 = views(X, CP.make_T(np.eye(3), [0, 0, 0]), CP.make_T(R_WC2, [0, 0, 0]))
    n_out = int(0.35 * len(x1))
    x2 = x2.copy()
    x2[:n_out] += np.random.default_rng(9).uniform(-30, 30, (n_out, 2))     # 35 % gross mismatches
    r = BR.analyse_pair(K, x1, x2, seed=3)
    assert r["state"] == "ROTATION_ONLY_CONSISTENT"
    assert CP.geodesic_deg(np.asarray(r["M"]), R_WC2.T) < 0.05


def test_coherent_foreground_motion_can_masquerade_as_translation():
    """Documented confound (why foreground exclusion matters): a rigidly shifting foreground group lies on a consistent
    epipolar pencil, so the essential model explains it as parallax and a static-rotation camera is classified as
    translating.  Masking that group restores the rotation-only decision."""
    R_WC2 = rot([0, 1, 0], 2.0)
    X = scene_points(800)
    x1, x2 = views(X, CP.make_T(np.eye(3), [0, 0, 0]), CP.make_T(R_WC2, [0, 0, 0]))
    n_out = int(0.35 * len(x1))
    x2 = x2.copy()
    x2[:n_out] += np.array([25.0, 3.0])                         # coherent 'person' motion on 35 % of the matches
    assert BR.analyse_pair(K, x1, x2, seed=3)["state"] == "TRANSLATION_OBSERVED"
    r = BR.analyse_pair(K, x1[n_out:], x2[n_out:], seed=3)     # foreground excluded
    assert r["state"] == "ROTATION_ONLY_CONSISTENT" and CP.geodesic_deg(np.asarray(r["M"]), R_WC2.T) < 0.05


def test_constant_camera_no_detectable_motion():
    X = scene_points()
    x1, x2 = views(X, CP.make_T(np.eye(3), [0, 0, 0]), CP.make_T(np.eye(3), [0, 0, 0]))
    r = BR.analyse_pair(K, x1, x2, seed=0)
    assert r["state"] == "NO_DETECTABLE_MOTION" and r["rotation_deg"] < r["detectability_deg"]
    assert r["translation"] == "UNRESOLVED_AT_THIS_SPAN"


def test_small_translation_is_not_reported_as_zero_or_as_observed():
    """1 mm translation at 3-9 m depth: indistinguishable from pure rotation -> translation UNRESOLVED (never 0)."""
    X = scene_points()
    x1, x2 = views(X, CP.make_T(np.eye(3), [0, 0, 0]), CP.make_T(rot([0, 1, 0], 1.0), [0.001, 0, 0]))
    r = BR.analyse_pair(K, x1, x2, seed=0)
    assert r["state"] in ("ROTATION_ONLY_CONSISTENT",) and r["translation"] == "UNRESOLVED_AT_THIS_SPAN"
    assert r["translation_to_depth_bound"] > 0


def test_large_translation_with_parallax_is_observed_with_direction():
    X = scene_points(near=2.0, far=12.0)
    t = np.array([0.4, 0.0, 0.1])
    x1, x2 = views(X, CP.make_T(np.eye(3), [0, 0, 0]), CP.make_T(rot([0, 1, 0], 2.0), t))
    r = BR.analyse_pair(K, x1, x2, seed=0)
    assert r["state"] == "TRANSLATION_OBSERVED" and r["translation"] == "DIRECTION_ONLY_SCALE_UNOBSERVABLE"
    d = np.asarray(r["translation_direction_in_cam_i"])
    assert math.degrees(math.acos(np.clip(d @ (t / np.linalg.norm(t)), -1, 1))) < 6.0     # 0.3 px noise
    assert CP.geodesic_deg(np.asarray(r["M"]), rot([0, 1, 0], 2.0).T) < 0.2


def test_planar_scene_translation_is_not_classified_as_pure_rotation():
    """A homography fits a translating camera over a plane; it must NOT be read as rotation-only evidence."""
    X = scene_points(planar=True)
    x1, x2 = views(X, CP.make_T(np.eye(3), [0, 0, 0]), CP.make_T(np.eye(3), [0.5, 0.0, 0.0]))
    H, _ = cv2.findHomography(x1, x2, cv2.RANSAC, 1.5)
    assert np.median(np.sqrt(BR.homog_sym_err2(H, x1, x2))) < 1.0          # the homography does fit
    r = BR.analyse_pair(K, x1, x2, seed=0)
    assert r["state"] not in ("ROTATION_ONLY_CONSISTENT", "NO_DETECTABLE_MOTION")
    assert r["state"] in ("TRANSLATION_PRESENT_PLANAR_AMBIGUOUS", "TRANSLATION_OBSERVED")


def test_pure_rotation_over_plane_stays_rotation():
    X = scene_points(planar=True)
    x1, x2 = views(X, CP.make_T(np.eye(3), [0, 0, 0]), CP.make_T(rot([1, 0, 0], 3.0), [0, 0, 0]))
    r = BR.analyse_pair(K, x1, x2, seed=0)
    assert r["state"] == "ROTATION_ONLY_CONSISTENT"


def test_wrong_focal_negative_control():
    X = scene_points()
    x1, x2 = views(X, CP.make_T(np.eye(3), [0, 0, 0]), CP.make_T(rot([0, 1, 0], 6.0), [0, 0, 0]), noise=0.0)
    Kw = K.copy()
    Kw[0, 0] *= 1.3
    Kw[1, 1] *= 1.3
    Rg, _ = BR.rotation_ransac(K, x1, x2, seed=0)
    Rw, _ = BR.rotation_ransac(Kw, x1, x2, seed=0)
    e_g = np.median(BR.rot_sym_err2(K, Rg, x1, x2))
    e_w = np.median(BR.rot_sym_err2(Kw, Rw, x1, x2))
    assert e_w > 10 * max(e_g, 1e-6) and CP.geodesic_deg(Rw, rot([0, 1, 0], 6.0).T) > 0.5


def test_transposed_relative_rotation_is_detected():
    R_WC2 = rot([0, 1, 0], 4.0) @ rot([1, 0, 0], 2.0)
    M = R_WC2.T                                                     # R_C2C1 with R_WC1 = I
    assert CP.geodesic_deg(np.eye(3) @ M.T, R_WC2) < 1e-9
    assert CP.geodesic_deg(np.eye(3) @ M, R_WC2) > 5.0


def test_degenerate_few_matches_insufficient():
    r = BR.analyse_pair(K, np.zeros((10, 2)) + 100, np.zeros((10, 2)) + 100, seed=0)
    assert r["state"] == "INSUFFICIENT_EVIDENCE" and not r["rotation_valid"]


def test_gric_prefers_simpler_model_on_equal_fit():
    e = np.full(500, 0.3)
    assert BR.gric(e, 2, 3) < BR.gric(e, 2, 8) < BR.gric(e, 3, 5)


# ------------------------------------------------------------------------------------------------ averaging / regimes ---
def test_rotation_averaging_recovers_chain_and_rejects_outlier_edge():
    rng = np.random.default_rng(0)
    n = 30
    R_true = [rot([0, 1, 0], 0.4 * k) @ rot([1, 0, 0], 0.2 * math.sin(k / 3)) for k in range(n)]
    edges, Ms = [], []
    for i in range(n):
        for s in (1, 2, 4):
            j = i + s
            if j < n:
                M = R_true[j].T @ R_true[i]
                M = rot(rng.normal(size=3), 0.02) @ M
                edges.append((i, j))
                Ms.append(M)
    Ms[10] = rot([0, 0, 1], 30) @ Ms[10]                            # one gross outlier edge
    R, info = BR.rotation_averaging(n, edges, Ms)
    err = [CP.geodesic_deg(R_true[0] @ R[k], R_true[k]) for k in range(n)]
    assert max(err) < 0.1 and info["n_outlier_edges"] == 1 and info["reference"] == 0


def test_rotation_averaging_components_other_gauge():
    M = np.eye(3)
    R, info = BR.rotation_averaging(6, [(0, 1), (1, 2), (2, 3), (4, 5)], [M] * 4)
    assert info["component"] == [0, 1, 2, 3] and R[4] is None and R[5] is None


def test_frame_regimes_and_sequence_regime():
    times = np.arange(20) * 0.1
    rot_edges = [{"i": i, "j": i + 1, "state": "ROTATION_ONLY_CONSISTENT", "rotation_deg": 0.5} for i in range(19)]
    reg = BR.frame_regimes(times, rot_edges)
    assert all(r["regime"] == "ROTATING_TRANSLATION_UNRESOLVED" for r in reg[3:-3])
    still = [{"i": i, "j": i + 1, "state": "NO_DETECTABLE_MOTION", "rotation_deg": 0.0} for i in range(19)]
    assert BR.sequence_regime(BR.frame_regimes(times, still))["regime"] == "NEAR_STATIC_TRANSLATION_UNRESOLVED"
    tr = [{"i": i, "j": i + 1, "state": "TRANSLATION_OBSERVED", "rotation_deg": 0.1} for i in range(19)]
    assert BR.sequence_regime(BR.frame_regimes(times, tr))["regime"] == "TRANSLATING"
    assert BR.frame_regimes(times, [])[0]["regime"] == "INSUFFICIENT_EVIDENCE"


def test_gt_regime_labels():
    t = np.arange(30) * 0.1
    P = np.zeros((30, 3))
    R = np.stack([np.eye(3)] * 30)
    assert set(BR.gt_regime_labels(t, P, R)) == {"NEAR_STATIC"}
    P[:, 0] = np.arange(30) * 0.02
    assert set(BR.gt_regime_labels(t, P, R)) == {"TRANSLATING"}


# ------------------------------------------------------------------------------------------------ evaluation ---
def test_orientation_alignment_and_constant_baseline():
    Rg = np.stack([rot([0, 1, 0], 0.5 * k) for k in range(20)])
    G0 = rot([1, 2, 3], 40)
    Re = np.einsum("ij,njk->nik", G0.T, Rg)                          # estimate in another gauge
    Ra, err = BR.orientation_errors(Rg, Re)
    assert err.max() < 1e-4                                          # arccos precision
    _, berr = BR.orientation_errors(Rg, np.broadcast_to(np.eye(3), Rg.shape))
    assert berr.max() > 4.0
    e, b, pairs = BR.relative_errors(np.arange(20) * 0.1, Rg, Re)
    assert e.max() < 1e-4 and abs(np.median(b) - 0.5) < 1e-4 and len(pairs) == 19


# ------------------------------------------------------------------------------------------------ masks ---
def _texture(seed=0, h=480, w=640):
    rng = np.random.default_rng(seed)
    img = np.full((h, w, 3), 128, np.uint8)
    for _ in range(900):
        x, y = rng.integers(0, w), rng.integers(0, h)
        cv2.circle(img, (int(x), int(y)), int(rng.integers(3, 14)), tuple(int(v) for v in rng.integers(0, 255, 3)), -1)
    return img


def test_masked_features_never_on_excluded_pixels_and_static_preserved():
    img = _texture()
    keep = np.ones((480, 640), bool)
    keep[100:380, 200:420] = False
    xy_all, _, _ = BR.sift_features(img, None)
    xy, desc, dropped = BR.sift_features(img, keep)
    assert len(xy) > 200 and BR.on_keep(xy, keep).all()
    outside = xy_all[BR.on_keep(xy_all, keep)]
    assert len(xy) >= 0.8 * min(len(outside), BR.SIFT_NFEATURES)


def test_on_keep_rounding_convention():
    keep = np.ones((5, 5), bool)
    keep[2, 3] = False
    assert BR.on_keep(np.array([[3.0, 2.0], [3.4, 2.4], [3.6, 2.0], [2.6, 2.0]]), keep).tolist() == [False, False, True, False]


# ------------------------------------------------------------------------------------------------ estimator integration ---
def _make_rotation_sequence(tmp, n=6, bad_cam=False, leak=False, missing=False):
    tex = _texture(5, 900, 1200)
    ind = tmp / "in"
    (ind / "images").mkdir(parents=True)
    Rs, frames = [], []
    for k in range(n):
        R_WC = rot([0, 1, 0], 0.8 * k) @ rot([1, 0, 0], 0.3 * k)
        Rs.append(R_WC)
        Kt = np.array([[500.0, 0, 599.5], [0, 500.0, 449.5], [0, 0, 1]])   # big texture plane at infinity
        H = K @ R_WC.T @ np.linalg.inv(Kt)
        img = cv2.warpPerspective(tex, H, (640, 480))
        nm = f"f{k:04d}_{100 + 0.1 * k:.6f}.png"
        if not (missing and k == 3):
            cv2.imwrite(str(ind / "images" / nm), img)
        else:
            cv2.imwrite(str(ind / "images" / nm), np.zeros((10, 10, 3), np.uint8))   # invalid frame
        frames.append({"frame_id": 10 + k, "timestamp": 100 + 0.1 * k, "image": nm})
    json.dump(frames, open(ind / "frames.json", "w"))
    cam = {"model": "PINHOLE", "width": 640, "height": 480, "fx": 500.0, "fy": 500.0, "cx": 700.0 if bad_cam else 319.5, "cy": 239.5}
    json.dump(cam, open(ind / "camera.json", "w"))
    json.dump({}, open(ind / "input_manifest.json", "w"))
    if leak:
        (ind / "groundtruth.txt").write_text("0 0 0 0 0 0 0 1\n")
    return ind, Rs, frames


def _run(ind, out):
    return subprocess.run([sys.executable, "-I", str(ROOT / "scripts" / "w142_background_rotation_estimate.py"), "--input", str(ind), "--out",
                           str(out), "--mask-label", "none"], capture_output=True, text=True)


def test_estimator_end_to_end_rotation_identity_and_null_translation(tmp_path):
    ind, Rs, frames = _make_rotation_sequence(tmp_path)
    r = _run(ind, tmp_path / "out")
    assert r.returncode == 0, r.stderr[-2000:]
    est = json.load(open(tmp_path / "out" / "estimate.json"))
    assert [f["frame_id"] for f in est["frames"]] == [f["frame_id"] for f in frames]
    assert [f["timestamp"] for f in est["frames"]] == [f["timestamp"] for f in frames]
    for f, R in zip(est["frames"], Rs):
        assert f["orientation"]["status"] == "ESTIMATED" and f["translation"]["value"] is None
        assert CP.geodesic_deg(np.asarray(f["orientation"]["R_WC"]), Rs[0].T @ R) < 0.1     # reference = frame 0
    assert all(e["state"] in ("ROTATION_ONLY_CONSISTENT", "NO_DETECTABLE_MOTION") for e in est["edges"] if e["state"] != "INSUFFICIENT_EVIDENCE")


def test_estimator_refuses_gt_leak_and_bad_calibration(tmp_path):
    ind, _, _ = _make_rotation_sequence(tmp_path / "a", n=3, leak=True)
    assert _run(ind, tmp_path / "a" / "out").returncode != 0
    ind, _, _ = _make_rotation_sequence(tmp_path / "b", n=3, bad_cam=True)
    assert _run(ind, tmp_path / "b" / "out").returncode != 0


def test_estimator_invalid_frame_is_kept_in_denominator(tmp_path):
    ind, _, frames = _make_rotation_sequence(tmp_path, n=6, missing=True)
    r = _run(ind, tmp_path / "out")
    assert r.returncode == 0, r.stderr[-2000:]
    est = json.load(open(tmp_path / "out" / "estimate.json"))
    assert len(est["frames"]) == 6 and est["frames"][3]["orientation"]["status"] == "INSUFFICIENT_EVIDENCE"
    assert est["frames"][3]["evidence"].get("invalid_input") is True


def test_harness_declared_runs_and_gates():
    import importlib.util
    spec = importlib.util.spec_from_file_location("w142h", ROOT / "scripts" / "w142_harness.py")
    H = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(H)
    names = [r[0] for r in H.RUNS]
    assert len(names) == len(set(names)) == 19
    assert not any(r[2] == "oracle" for r in H.RUNS if not r[3].startswith("syn_"))      # oracle never on real data / S1
    assert {r[2] for r in H.RUNS if r[3] == "walking_static"} == {"rgbmask", None, "areactrl"}
    times = [k / 30 for k in range(140)]
    c0, R0 = np.zeros(3), np.eye(3)
    A1 = H.trajectory("A1", times, c0, R0)
    A2 = H.trajectory("A2", times, c0, R0)
    A3 = H.trajectory("A3", times, c0, R0)
    assert all(np.allclose(c, 0) and np.allclose(R, np.eye(3)) for c, R in A1)
    assert all(np.allclose(c, 0) for c, _ in A2) and max(CP.geodesic_deg(np.eye(3), R) for _, R in A2) > 8
    assert abs(np.linalg.norm(A3[-1][0] - A3[0][0]) - math.hypot(0.04, 0.015)) < 1e-9
