"""W141 focused tests: dynamic-scene camera pose instruments (masks, feature exclusion, accounting, evaluation, gates).

The W140 camera geometry tests (tests/test_experimental_camera_pose.py) are reused unchanged and run alongside these.
"""
from __future__ import annotations

import importlib.util
import json
import math
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from app import experimental_camera_pose as CP  # noqa: E402
from app import experimental_dynamic_camera_pose as DP  # noqa: E402

cv2 = pytest.importorskip("cv2")


def _load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _rot(axis, deg):
    from scipy.spatial.transform import Rotation
    return Rotation.from_rotvec(np.radians(deg) * np.asarray(axis, float)).as_matrix()


def _traj(n=120, motion="xyz", seed=0):
    t = np.arange(n) * 0.1                               # motion parameter; timestamps are 0.02 s apart (< W140 50 ms GT gap rule)
    P = np.zeros((n, 3))
    if motion == "xyz":
        P[:, 0] = 0.4 * np.sin(t)
        P[:, 1] = 0.3 * np.cos(0.7 * t)
        P[:, 2] = 0.2 * np.sin(0.5 * t)
    R = np.stack([_rot([0, 0, 1], 10 * math.sin(0.3 * k)) @ _rot([1, 0, 0], 5 * math.cos(0.2 * k)) for k in range(n)])
    return 1000.0 + 0.2 * t, P, R


def _gt_array(t, P, R):
    return np.column_stack([t, P, CP.R_to_quat_xyzw(R)])


def _est_from(t, P, R, s=2.5, Rg=None, tg=(1, -2, 0.5), status=None):
    Rg = _rot([0.3, 1, 0.2] / np.linalg.norm([0.3, 1, 0.2]), 70) if Rg is None else Rg
    est = []
    for i in range(len(t)):
        # GT = s Rg X + tg  ->  X = Rg^T (GT - tg) / s
        c = Rg.T @ (P[i] - np.asarray(tg)) / s
        Rw = Rg.T @ R[i]
        st = "ESTIMATED" if status is None else status[i]
        est.append({"frame_id": i, "timestamp": float(t[i]), "status": st, "T_WC": CP.make_T(Rw, c).tolist() if st == "ESTIMATED" else None})
    return est


# ------------------------------------------------------------------------------------------------ masks / coordinates ---
def test_keypoint_mask_coordinate_convention_colmap_corner_origin():
    m = np.zeros((4, 6), bool)
    m[2, 3] = True                                       # pixel row 2, col 3 covers COLMAP x in [3, 4), y in [2, 3)
    xy = np.array([[3.0, 2.0], [3.99, 2.99], [2.99, 2.5], [3.5, 3.0], [3.5, 2.5]])
    assert DP.keypoints_in_mask(xy, m).tolist() == [True, True, False, False, True]
    # OpenCV centre (3, 2) == COLMAP (3.5, 2.5)
    assert DP.keypoints_in_mask(np.array([[3 + 0.5, 2 + 0.5]]), m).tolist() == [True]


def test_dilation_contains_raw_and_has_declared_radius():
    raw = np.zeros((100, 100), bool)
    raw[50, 50] = True
    B = DP.disk_dilate(raw, 15)
    assert B[50, 50] and not (raw & ~B).any()
    ys, xs = np.nonzero(B)
    assert xs.min() == 35 and xs.max() == 65 and ys.min() == 35 and ys.max() == 65
    assert not B[50 - 11, 50 - 11]                      # disk, not square (corner at distance ~15.6 > 15)


def test_keep_mask_is_colmap_convention():
    E = np.zeros((3, 3), bool)
    E[1, 1] = True
    k = DP.keep_from_exclusion(E)
    assert k.dtype == np.uint8 and k[1, 1] == 0 and k[0, 0] == 255 and set(np.unique(k)) == {0, 255}


def test_area_control_is_disjoint_area_matched_and_deterministic():
    H = np.zeros((48, 64), bool)
    H[10:30, 5:20] = True
    C, info = DP.area_control_mask(H)
    C2, _ = DP.area_control_mask(H)
    assert (C == C2).all()
    assert int(C.sum()) == int(H.sum()) and info["shortfall_px"] == 0 and info["disjoint"] and not (C & H).any()
    assert info["offset"] == [0.5, 0.0]                  # shifted copy lands on the opposite half (no overlap)


def test_area_control_empty_and_heavy_occlusion():
    C, info = DP.area_control_mask(np.zeros((10, 10), bool))
    assert C.sum() == 0 and info["area_px"] == 0
    H = np.ones((20, 20), bool)
    H[:, :6] = False                                     # 70 % occupied: only 30 % background exists
    C, info = DP.area_control_mask(H)
    assert not (C & H).any() and info["shortfall_px"] == int(H.sum()) - int((~H).sum()) and int(C.sum()) == int((~H).sum())


def test_mask_quality():
    a = np.zeros((10, 10), bool)
    b = np.zeros((10, 10), bool)
    a[:5] = True
    b[:4] = True
    q = DP.mask_quality(a, b)
    assert q["iou"] == pytest.approx(0.8) and q["recall"] == 1.0 and q["precision"] == pytest.approx(0.8)


def test_masked_feature_exclusion_and_static_preservation_simulated():
    """COLMAP's mask filter semantics: keep exactly the keypoints on non-zero keep pixels (no redistribution)."""
    rng = np.random.default_rng(0)
    xy = rng.uniform([0, 0], [64, 48], (500, 2))
    E = np.zeros((48, 64), bool)
    E[10:40, 20:35] = True
    keep = DP.keep_from_exclusion(E)
    kept = xy[keep[np.floor(xy[:, 1]).astype(int), np.floor(xy[:, 0]).astype(int)] > 0]
    assert DP.keypoints_in_mask(kept, E).sum() == 0
    assert len(kept) == len(xy) - DP.keypoints_in_mask(xy, E).sum()


# ------------------------------------------------------------------------------------------------ wrapper contract ---
def _mk_input(tmp, n=3, h=12, w=16):
    ind = tmp / "in"
    (ind / "images").mkdir(parents=True)
    fr = []
    for i in range(n):
        nm = f"f{i:04d}_{100 + i * 0.1:.6f}.png"
        cv2.imwrite(str(ind / "images" / nm), np.full((h, w, 3), 100, np.uint8))
        fr.append({"frame_id": i, "timestamp": 100 + i * 0.1, "image": nm})
    json.dump(fr, open(ind / "frames.json", "w"))
    json.dump({"model": "PINHOLE", "width": w, "height": h, "fx": 10, "fy": 10, "cx": 7.5, "cy": 5.5}, open(ind / "camera.json", "w"))
    md = tmp / "keep"
    md.mkdir()
    for f in fr:
        cv2.imwrite(str(md / (f["image"] + ".png")), np.full((h, w), 255, np.uint8))
    return ind, md, fr


def test_wrapper_mask_contract_accepts_valid(tmp_path):
    W = _load_script("w141_colmap_masked_estimate")
    ind, md, fr = _mk_input(tmp_path)
    rec = W.check_masks(ind, md)
    assert set(rec) == {f["image"] for f in fr} and all(v["keep_fraction"] == 1.0 for v in rec.values())


@pytest.mark.parametrize("bad", ["missing", "extra_gt_file", "wrong_shape", "non_binary", "colour"])
def test_wrapper_refuses_invalid_masks(tmp_path, bad):
    W = _load_script("w141_colmap_masked_estimate")
    ind, md, fr = _mk_input(tmp_path)
    p = md / (fr[0]["image"] + ".png")
    if bad == "missing":
        p.unlink()
    elif bad == "extra_gt_file":
        (md / "groundtruth.txt").write_text("0 0 0 0 0 0 0 1")
    elif bad == "wrong_shape":
        cv2.imwrite(str(p), np.full((13, 16), 255, np.uint8))
    elif bad == "non_binary":
        cv2.imwrite(str(p), np.full((12, 16), 128, np.uint8))
    else:
        cv2.imwrite(str(p), np.full((12, 16, 3), 255, np.uint8))
    with pytest.raises(SystemExit):
        W.check_masks(ind, md)


def test_wrapper_runs_w140_estimator_unchanged_and_input_guard_still_applies(tmp_path):
    src = (ROOT / "scripts" / "w141_colmap_masked_estimate.py").read_text()
    assert "w140_colmap_estimate.py" in src and "mask_path" in src
    W140 = _load_script("w140_colmap_estimate")
    ind, md, fr = _mk_input(tmp_path)
    W140.refuse_leaks(ind)                                # clean input passes
    (ind / "depth").mkdir()
    with pytest.raises(SystemExit):
        W140.refuse_leaks(ind)


# ------------------------------------------------------------------------------------------------ COLMAP readers ---
def test_read_db_keypoints(tmp_path):
    db = tmp_path / "d.db"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE images (image_id INTEGER PRIMARY KEY, name TEXT, camera_id INTEGER)")
    con.execute("CREATE TABLE keypoints (image_id INTEGER PRIMARY KEY, rows INTEGER, cols INTEGER, data BLOB)")
    a = np.array([[1.5, 2.5, 0, 0, 0, 0], [10.25, 3.75, 1, 0, 0, 1]], np.float32)
    con.execute("INSERT INTO images VALUES (1, 'a.png', 1)")
    con.execute("INSERT INTO keypoints VALUES (1, 2, 6, ?)", (a.tobytes(),))
    con.commit()
    con.close()
    k = DP.read_db_keypoints(db)
    assert np.allclose(k["a.png"], [[1.5, 2.5], [10.25, 3.75]])


def test_read_images_txt(tmp_path):
    p = tmp_path / "images.txt"
    p.write_text("# header\n1 1 0 0 0 0 0 0 1 a.png\n1.5 2.5 7 3.0 4.0 -1 5.5 6.5 9\n")
    o = DP.read_images_txt(p)
    assert o["a.png"]["pid"].tolist() == [7, 9] and np.allclose(o["a.png"]["xy"], [[1.5, 2.5], [5.5, 6.5]])


# ------------------------------------------------------------------------------------------------ calibration / identity ---
def test_fr3_calibration_identity_and_not_fr1():
    H = _load_script("w141_dynamic_camera_harness")
    vals = [float(v) for v in H.CALIB_ROW.split()[3:]]
    assert vals[:4] == [H.FR3["fx"], H.FR3["fy"], H.FR3["cx"], H.FR3["cy"]] and vals[4:] == [0] * 5
    assert H.CALIB_ROW.startswith("Freiburg 3 RGB") and H.FR3["fx"] != 517.3
    K = np.array([[H.FR3["fx"], 0, H.FR3["cx"]], [0, H.FR3["fy"], H.FR3["cy"]], [0, 0, 1.0]])
    assert CP.check_intrinsics(K, H.FR3["width"], H.FR3["height"])


def test_frame_selection_reproducible_and_timestamps_preserved():
    a, b = CP.select_uniform(743), CP.select_uniform(743)
    assert a == b and len(a) == 300 and a[0] == 0 and a[-1] == 742
    assert len(CP.select_uniform(859)) == 300


def test_declared_runs_matched():
    H = _load_script("w141_dynamic_camera_harness")
    names = [r[0] for r in H.RUNS]
    assert len(names) == 11 and len(set(names)) == 11
    for seq in H.SEQS:
        rs = [r for r in H.RUNS if r[3] == seq]
        assert {r[2] for r in rs} == {None, "rgbmask", "areactrl"} and len({r[1] for r in rs}) == 1     # same input for A/B/C
    assert not any(r[2] in ("oracle", "oracle_areactrl") for r in H.RUNS if r[3] != "synthetic")       # oracle never on real data


# ------------------------------------------------------------------------------------------------ evaluation ---
def test_evaluate_dynamic_exact_recovery_and_coverage_denominator():
    t, P, R = _traj()
    gt = _gt_array(t, P, R)
    status = ["ESTIMATED"] * len(t)
    for i in (40, 41, 42, 90):
        status[i] = "LOST"
    est = _est_from(t, P, R, status=status)
    ev = DP.evaluate_dynamic(est, gt)
    assert ev["coverage_estimated"] == pytest.approx(116 / 120)
    assert ev["full_sim3"]["ate"]["rmse"] < 1e-9 and ev["full_sim3"]["rot"]["median_deg"] < 1e-5
    assert [(r["first_frame_id"], r["last_frame_id"], r["n"]) for r in ev["lost_intervals"]] == [(40, 42, 3), (90, 90, 1)]
    assert ev["spikes_gt_referenced"]["n"] == 0
    assert ev["early_reference"]["status"] == "OBSERVABLE"
    pa = ev["early_reference"]["pose_aware"]
    assert pa["ate_rest"]["rmse"] < 1e-9 and abs(pa["scale_ratio_to_full"] - 1) < 1e-9


def test_spike_detection_and_path_length():
    t, P, R = _traj()
    gt = _gt_array(t, P, R)
    est = _est_from(t, P, R)
    T = np.asarray(est[60]["T_WC"])
    T[:3, 3] += np.array([0.05, 0, 0])                   # one-frame jump (0.125 m in GT units)
    est[60]["T_WC"] = T.tolist()
    ev = DP.evaluate_dynamic(est, gt)
    pairs = [tuple(s["pair_frame_ids"]) for s in ev["spikes_gt_referenced"]["spikes"]]
    assert (59, 60) in pairs and (60, 61) in pairs
    assert ev["path_length"]["est_scaled_m"] > ev["path_length"]["gt_m"]


def test_early_reference_unobservable_when_stationary_start():
    t, P, R = _traj()
    P[:40] = P[40]                                       # stationary early interval
    gt = _gt_array(t, P, R)
    ev = DP.evaluate_dynamic(_est_from(t, P, R), gt)
    assert ev["early_reference"]["status"] == "ROTATION_ONLY" and "pose_aware" not in ev["early_reference"]
    assert ev["early_reference"]["rotation_only"]["rot_rest"]["median_deg"] < 1e-5


def test_pose_aware_alignment_recovers_rotation_from_orientations():
    t, P, R = _traj()
    est = _est_from(t, P, R)
    Pe = np.asarray([np.asarray(e["T_WC"])[:3, 3] for e in est])
    Re = np.asarray([np.asarray(e["T_WC"])[:3, :3] for e in est])
    s, Ra, tt = DP.pose_aware_sim3(Pe[:5], Re[:5], P[:5], R[:5])
    assert abs(s - 2.5) < 1e-9 and np.allclose(Ra @ Re[50], R[50], atol=1e-9)


def test_evaluate_on_ids_common_subset():
    t, P, R = _traj()
    gt = _gt_array(t, P, R)
    est = _est_from(t, P, R)
    o = DP.evaluate_on_ids(est, gt, list(range(10, 60)))
    assert o["n"] == 50 and len(o["frame_ids"]) == 50 and o["ate_rmse"] < 1e-9


def test_scale_ambiguity_preserved():
    t, P, R = _traj()
    gt = _gt_array(t, P, R)
    est = _est_from(t, P, R, s=0.01)
    ev = DP.evaluate_dynamic(est, gt)
    # estimate units are 1/0.01 = 100x GT metres; the Sim(3) scale absorbs it and is reported as evaluation-only
    assert ev["full_sim3"]["sim3"]["scale"] == pytest.approx(0.01) and "SCALE_AMBIGUOUS" in ev["scale_status"]


def test_gt_duplicate_timestamps_removed_not_picked(tmp_path):
    p = tmp_path / "groundtruth.txt"
    rows = ["# c", "1.00 0 0 0 0 0 0 1", "1.01 0 0 0 0 0 0 1", "1.02 1 0 0 0 0 0 1", "1.02 2 0 0 0 0 0 1", "1.03 0 0 0 0 0 0 1"]
    p.write_text("\n".join(rows) + "\n")
    with pytest.raises(CP.CamStop):
        CP.read_groundtruth(p)                            # W140 reader refuses (unchanged)
    a, info = DP.read_groundtruth_dedup(p)
    assert a[:, 0].tolist() == [1.00, 1.01, 1.03] and info["n_removed"] == 2 and info["duplicated_timestamps"] == [1.02]
    p.write_text("\n".join(rows[:3] + ["1.005 0 0 0 0 0 0 1"]) + "\n")
    with pytest.raises(CP.CamStop):
        DP.read_groundtruth_dedup(p)                      # genuinely out-of-order GT is still refused


# ------------------------------------------------------------------------------------------------ gates ---
def _ev(cov, ate, rot, bate=0.02, brot=1.0):
    return {"coverage_estimated": cov, "full_sim3": {"ate": {"rmse": ate}, "rot": {"median_deg": rot}},
            "baseline_stationary": {"ate_rmse": bate, "rot_median_deg": brot}}


def test_config_gate_requires_baseline_on_walking_static():
    g = DP.config_gate("walking_static", _ev(0.95, 0.03, 0.8, bate=0.025, brot=1.0))
    assert g["accuracy_target_met"] and not g["beats_baseline"] and not g["pass"]
    assert DP.config_gate("walking_static", _ev(0.95, 0.01, 0.5))["pass"]
    assert not DP.config_gate("walking_xyz", _ev(0.79, 0.01, 0.5, 0.3, 3))["pass"]
    assert not DP.config_gate("walking_xyz", _ev(0.9, 0.16, 0.5, 0.3, 3))["pass"]


def _cm(n, ate, rot):
    return {"n": n, "ate_rmse": ate, "rot_median_deg": rot}


def test_masking_benefit_verdicts():
    gp, gf = {"pass": True}, {"pass": False}
    # B better, C not -> DEMONSTRATED
    r = DP.masking_benefit({"A": 1, "B": 1, "C": 1}, {"A": _cm(200, 0.10, 2), "B": _cm(200, 0.05, 2), "C": _cm(200, 0.10, 2)}, {"A": gf})
    assert r["verdict"] == "DEMONSTRATED"
    # B and C equally better -> not attributable
    r = DP.masking_benefit({"A": 1, "B": 1, "C": 1}, {"A": _cm(200, 0.10, 2), "B": _cm(200, 0.05, 2), "C": _cm(200, 0.05, 2)}, {"A": gf})
    assert r["verdict"] == "NOT_ATTRIBUTABLE_TO_HUMAN_EXCLUSION"
    # A already good, no change -> UNNECESSARY
    r = DP.masking_benefit({"A": 1, "B": 1, "C": 1}, {"A": _cm(200, 0.010, 1), "B": _cm(200, 0.011, 1), "C": _cm(200, 0.010, 1)}, {"A": gp})
    assert r["verdict"] == "UNNECESSARY"
    # B loses coverage -> HARMFUL
    r = DP.masking_benefit({"A": 1, "B": 0.5, "C": 1}, {"A": _cm(150, 0.01, 1), "B": _cm(150, 0.01, 1), "C": _cm(150, 0.01, 1)}, {"A": gp})
    assert r["verdict"] == "HARMFUL" and r["B_coverage_loss"]
    # coverage rescue by B only
    r = DP.masking_benefit({"A": 0.4, "B": 0.95, "C": 0.4}, {"A": _cm(10, 0.1, 2), "B": _cm(10, 0.1, 2), "C": _cm(10, 0.1, 2)}, {"A": gf})
    assert r["verdict"] == "DEMONSTRATED" and r["B_vs_A"]["coverage_rescue"]
    # tiny absolute gain does not count
    r = DP.masking_benefit({"A": 1, "B": 1, "C": 1}, {"A": _cm(200, 0.004, 0.3), "B": _cm(200, 0.002, 0.2), "C": _cm(200, 0.004, 0.3)}, {"A": gf})
    assert not r["B_vs_A"]["improves"]


def test_synthetic_disturbance_and_judgment():
    clean = {"coverage_estimated": 1.0, "full_sim3": {"ate": {"rmse": 0.001}, "rot": {"median_deg": 0.05}}}
    bad = {"coverage_estimated": 1.0, "full_sim3": {"ate": {"rmse": 0.02}, "rot": {"median_deg": 0.05}}}
    assert DP.synthetic_disturbance(bad, clean, 1.7, True)["disturbed"]
    assert not DP.synthetic_disturbance(clean, clean, 1.7, True)["disturbed"]
    assert DP.synthetic_disturbance(clean, clean, 1.7, False)["disturbed"]
    assert DP.judgment({"a": True, "b": True}, {})["judgment"] == "SUPPORTED"
    assert DP.judgment({"a": True, "b": False}, {"a": True, "b": False})["judgment"] == "PARTIALLY_SUPPORTED"
    assert DP.judgment({"a": False, "b": False}, {"a": False, "b": False})["judgment"] == "NOT_YET_SUPPORTED"
    assert DP.judgment({}, {}, blocked="x")["judgment"] == "BLOCKED"


def test_dynamic_foreground_contamination_synthetic_toy():
    """A toy two-view check that features on an independently moving object violate the static epipolar geometry."""
    rng = np.random.default_rng(3)
    K = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1.0]])
    Xs = rng.uniform([-2, -1, 4], [2, 1, 8], (200, 3))                       # static points
    Xd = rng.uniform([-0.3, -0.5, 3], [0.3, 0.5, 3.5], (60, 3))              # moving object points
    T1 = np.eye(4)
    T2 = CP.make_T(np.eye(3), [0.2, 0, 0])                                   # camera translates along +x
    def proj(T, X):
        return CP.project(K, T, X)[0]
    x1s, x2s = proj(T1, Xs), proj(T2, Xs)
    x1d, x2d = proj(T1, Xd), proj(T2, Xd + np.array([0, 0.15, 0]))           # object moves in y between views
    F, _ = cv2.findFundamentalMat(x1s, x2s, cv2.FM_8POINT)
    def epi(x1, x2):
        l = (F @ np.column_stack([x1, np.ones(len(x1))]).T).T
        return np.abs(np.sum(l * np.column_stack([x2, np.ones(len(x2))]), 1)) / np.linalg.norm(l[:, :2], axis=1)
    assert np.median(epi(x1s, x2s)) < 1e-6 and np.median(epi(x1d, x2d)) > 5.0
