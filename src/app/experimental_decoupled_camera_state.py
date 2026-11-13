"""W143 (EXPERIMENTAL, opt-in, scratch only): DECOUPLED camera orientation / translation observability / 6DoF availability.

Builds on, and never modifies, W142 (``app.experimental_background_rotation``: SIFT, matching, rotation-only RANSAC,
homography / essential / GRIC, rotation averaging, regimes) and W140 (``app.experimental_camera_pose``: conventions).
NOT integrated into Production, FramePose, MotionGraph, the S1 calibration, root motion or the Generator.

W142 coupled four questions into one GRIC decision.  W143 answers them separately, per edge (frame pair):
  1. GEOMETRIC MODEL FIT        which of rotation-only / homography / essential explains the matches best (W142 GRIC, kept
                                only as a descriptive label);
  2. ROTATION MEASUREMENT       is the rotation-only estimate a valid ORIENTATION measurement?  Decided from image evidence
                                independent of (1): rotation support of the background consensus set, its rotational
                                residual, agreement with the rotation implied by the homography, and multi-span cycle
                                consistency.  Its translation-induced bias is bounded by residual / f and reported;
  3. TRANSLATION OBSERVABILITY  direction only, and only when the essential model wins, decomposes with cheirality, the
                                rotation-compensated parallax is >= PARALLAX_MIN_PX and another span from the same frame
                                confirms the direction.  A homography preference alone never implies translation;
  4. FULL 6DoF AVAILABILITY     only from a frozen SfM reconstruction whose own quality evidence passes (model coverage,
                                landmarks, reprojection; per-frame observed landmarks).  Never from a translation label.

Conventions (W140/W142): optical OpenCV axes, R_WC, scalar-last quaternions; per-edge M_ij = R_CjCi; W143 orientation is
expressed in the optical frame of its reference camera; an SfM 6DoF pose is expressed in that SfM model's own gauge with
SCALE_AMBIGUOUS translation.  The two gauges are never mixed into one trajectory.

States
  orientation   ORIENTATION_ESTIMATED | ORIENTATION_UNSUPPORTED
  translation   TRANSLATION_ESTIMATED_SCALE_AMBIGUOUS (SfM only) | TRANSLATION_DIRECTION_ONLY | TRANSLATION_UNRESOLVED |
                TRANSLATION_INSUFFICIENT_EVIDENCE      (value null unless SCALE_AMBIGUOUS SfM; never a measured zero)
  camera        FULL_6DOF_AVAILABLE | ORIENTATION_ONLY_AVAILABLE | CAMERA_MOTION_UNRESOLVED
"""
from __future__ import annotations

import math

import numpy as np

from app import experimental_background_rotation as BR
from app import experimental_camera_pose as CP

SCHEMA = "animcv_exp_w143_decoupled_camera_state_v1"
# ---- frozen parameters (declared before any W143 estimator output) ----
MIN_INLIERS = BR.MIN_INLIERS             # 30
SUPPORT_RATIO_MIN = 0.8                  # rotation inliers / best-model inliers
CONSENSUS_P90_MAX_PX = 1.5               # p90 rotation residual over the best-model consensus set (= RANSAC threshold)
AGREE_RH_MAX_DEG = 0.5                   # rotation-only vs homography-nearest rotation
CYCLE_MAX_DEG = 0.5                      # median triangle (s, s, 2s) cycle error of an edge
PARALLAX_MIN_PX = 2.0                    # rotation-compensated median displacement of essential inliers
DIR_CONFIRM_DEG = 20.0                   # translation direction confirmation by another span from the same frame
SFM_MODEL_MIN_REG_FRAC = 0.5
SFM_MODEL_MIN_LANDMARKS = 1000
SFM_MODEL_MAX_REPROJ_PX = 2.0
SFM_FRAME_MIN_OBS = 100
TRANSLATION_STATES = ("TRANSLATION_ESTIMATED_SCALE_AMBIGUOUS", "TRANSLATION_DIRECTION_ONLY", "TRANSLATION_UNRESOLVED",
                      "TRANSLATION_INSUFFICIENT_EVIDENCE")
CAMERA_STATES = ("FULL_6DOF_AVAILABLE", "ORIENTATION_ONLY_AVAILABLE", "CAMERA_MOTION_UNRESOLVED")


def nearest_rotation(A):
    U, S, Vt = np.linalg.svd(np.asarray(A, float))
    return U @ np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))]) @ Vt


def homography_rotation(K, H):
    """Closest rotation to K^-1 H K (scale-normalised).  Equals the camera rotation for a pure rotation; deviates by
    O(|t|/d) for a translating camera over a plane."""
    if H is None:
        return None
    A = np.linalg.inv(K) @ np.asarray(H, float) @ K
    s = np.linalg.svd(A, compute_uv=False)
    A = A / s[1]
    if np.linalg.det(A) < 0:
        A = -A
    return nearest_rotation(A)


def analyse_edge(K, x1, x2, seed):
    """All four W143 questions' per-edge evidence from the same tentative background matches."""
    import cv2
    n = len(x1)
    out = {"n_matches": int(n)}
    if n < MIN_INLIERS:
        out.update({"fit_model": None, "rotation_candidate": False, "reject": ["too_few_matches"], "translation_edge": "INSUFFICIENT",
                    "_inl": np.zeros(n, bool)})
        return out
    x1 = np.asarray(x1, np.float64)
    x2 = np.asarray(x2, np.float64)
    fpx = float((K[0, 0] + K[1, 1]) / 2)
    R_r, _ = BR.rotation_ransac(K, x1, x2, seed)
    e_r = BR.rot_sym_err2(K, R_r, x1, x2)
    cv2.setRNGSeed(int(seed) % (2 ** 31))
    H, _ = cv2.findHomography(x1, x2, cv2.RANSAC, BR.RANSAC_THR_PX, maxIters=2000, confidence=0.999)
    e_h = BR.homog_sym_err2(H, x1, x2) if H is not None else np.full(n, 1e12)
    cv2.setRNGSeed(int(seed) % (2 ** 31))
    E, _ = cv2.findEssentialMat(x1, x2, K, method=cv2.RANSAC, prob=0.999, threshold=BR.RANSAC_THR_PX)
    R_e, t_e, n_cheir = None, None, 0
    Kinv = np.linalg.inv(K)
    if E is not None and E.shape[0] >= 3:
        E = E[:3]
        e_e = BR.sampson_err2(Kinv.T @ E @ Kinv, x1, x2)
        inl_e = e_e < BR.RANSAC_THR_PX ** 2
        try:
            n_cheir, R_e, t_e, _ = cv2.recoverPose(E, x1[inl_e], x2[inl_e], K)
        except cv2.error:
            R_e = None
        if R_e is not None and inl_e.sum() >= 8 and np.linalg.norm(t_e) > 0:
            R_e, t_e = BR.refine_essential(K, np.asarray(R_e), t_e, x1[inl_e], x2[inl_e])
            e_e = BR.sampson_err2(Kinv.T @ BR.skew(t_e) @ R_e @ Kinv, x1, x2)
            inl_e = e_e < BR.RANSAC_THR_PX ** 2
    else:
        e_e = np.full(n, 1e12)
        inl_e = np.zeros(n, bool)
    thr2 = BR.RANSAC_THR_PX ** 2
    inl = {"R": e_r < thr2, "H": e_h < thr2, "E": inl_e}
    g = {"R": BR.gric(e_r, 2, 3), "H": BR.gric(e_h, 2, 8), "E": BR.gric(e_e, 3, 5)}
    fit = min(("R", "H", "E"), key=lambda m: (g[m], "RHE".index(m)))
    n_best = int(max(v.sum() for v in inl.values()))
    cons = inl[fit] if inl[fit].sum() >= MIN_INLIERS else inl[max(inl, key=lambda m: inl[m].sum())]
    res_c = np.sqrt(e_r[cons]) if cons.any() else np.array([np.inf])
    support = float(inl["R"].sum() / max(n_best, 1))
    p50, p90 = float(np.median(res_c)), float(np.percentile(res_c, 90))
    R_h = homography_rotation(K, H)
    agree_rh = float(CP.geodesic_deg(R_r, R_h)) if R_h is not None else None
    cheir_ok = R_e is not None and n_cheir >= BR.CHEIRALITY_MIN_FRAC * max(int(inl_e.sum()), 1) and n_cheir >= MIN_INLIERS
    agree_re = float(CP.geodesic_deg(R_r, np.asarray(R_e))) if cheir_ok else None
    reject = []
    if int(inl["R"].sum()) < MIN_INLIERS:
        reject.append("rotation_inliers")
    if support < SUPPORT_RATIO_MIN:
        reject.append("rotation_support")
    if not p90 <= CONSENSUS_P90_MAX_PX:
        reject.append("rotation_residual")
    if agree_rh is None or agree_rh > AGREE_RH_MAX_DEG:
        reject.append("rotation_vs_homography")
    out.update({"fit_model": fit, "gric": g, "n_inliers": {m: int(v.sum()) for m, v in inl.items()}, "n_best": n_best,
                "rot_support_ratio": support, "rot_consensus_resid_p50_px": p50, "rot_consensus_resid_p90_px": p90,
                "rotation_bias_bound_deg": math.degrees(p90 / fpx), "agree_rot_vs_homography_deg": agree_rh,
                "agree_rot_vs_essential_deg": agree_re, "M": R_r.tolist(), "rotation_deg": float(CP.geodesic_deg(np.eye(3), R_r)),
                "rotation_candidate": not reject, "reject": reject, "flow_raw_median_px": float(np.median(np.linalg.norm(x2 - x1, axis=1))),
                "_inl": inl["R"]})
    # ---- translation (independent of the rotation decision) ----
    if n_best < MIN_INLIERS:
        out["translation_edge"] = "INSUFFICIENT"
        return out
    par = None
    if cheir_ok:
        p2, z = BR.project_bearing(K, (np.asarray(R_e) @ BR.bearings(K, x1[inl_e]).T).T)
        par = float(np.median(np.linalg.norm(x2[inl_e] - p2, axis=1)))
        d = -(np.asarray(R_e).T @ np.asarray(t_e).reshape(3))
        out["translation_direction_in_cam_i"] = (d / max(np.linalg.norm(d), 1e-12)).tolist()
    out["parallax_px"] = par
    out["translation_edge"] = ("DIRECTION_CANDIDATE" if fit == "E" and cheir_ok and par is not None and par >= PARALLAX_MIN_PX
                               else "UNRESOLVED")
    return out


def cycle_scores(edges):
    """Median (s, s, 2s) triangle cycle error per rotation-candidate edge.  edges: dicts with i, j, span, M,
    rotation_candidate.  Returns {(i, j): [errors]}."""
    M = {(e["i"], e["j"]): np.asarray(e["M"]) for e in edges if e.get("rotation_candidate")}
    errs = {k: [] for k in M}
    spans = sorted({e["span"] for e in edges})
    for s in spans:
        if 2 * s not in spans:
            continue
        for (i, j), Mij in list(M.items()):
            if j - i != s:
                continue
            k = j + s
            if (j, k) in M and (i, k) in M:
                err = float(CP.geodesic_deg(M[(j, k)] @ Mij, M[(i, k)]))
                for key in ((i, j), (j, k), (i, k)):
                    errs[key].append(err)
    return errs


def cycle_reject(edges):
    """Greedy cycle-consistency validation: repeatedly remove the rotation-candidate edge with the largest median triangle
    error while it exceeds CYCLE_MAX_DEG (removing one bad edge first stops it contaminating its triangle partners).
    Returns (rejected set of (i, j), final {(i, j): [errors]})."""
    cand = [e for e in edges if e.get("rotation_candidate")]
    rejected = set()
    while True:
        live = [e for e in cand if (e["i"], e["j"]) not in rejected]
        sc = cycle_scores(live)
        worst, wv, wkey = None, -1.0, None
        for k, v in sc.items():
            if v:
                m = float(np.median(v))
                key = (round(m, 6), len(v), -k[0], -k[1])       # ties -> the edge with more triangle evidence, then earliest
                if wkey is None or key > wkey:
                    worst, wv, wkey = k, m, key
        if worst is None or wv <= CYCLE_MAX_DEG:
            return rejected, sc
        rejected.add(worst)


def confirm_translation(edges):
    """DIRECTION_CANDIDATE edges become DIRECTION_OBSERVED when another span from the same start frame agrees in direction."""
    by_i = {}
    for e in edges:
        if e.get("translation_edge") == "DIRECTION_CANDIDATE":
            by_i.setdefault(e["i"], []).append(e)
    for e in edges:
        if e.get("translation_edge") != "DIRECTION_CANDIDATE":
            continue
        d = np.asarray(e["translation_direction_in_cam_i"])
        ok = any(o is not e and math.degrees(math.acos(np.clip(d @ np.asarray(o["translation_direction_in_cam_i"]), -1, 1))) <= DIR_CONFIRM_DEG
                 for o in by_i.get(e["i"], []))
        e["translation_edge"] = "DIRECTION_OBSERVED" if ok else "UNRESOLVED_UNCONFIRMED"
    return edges


def sfm_validity(sfm):
    """Model- and frame-level validity of a frozen SfM estimate from its own reconstruction evidence (no GT)."""
    if sfm is None:
        return {"status": "SFM_NOT_AVAILABLE", "model_valid": False, "frames": {}}
    n = len(sfm["frames"])
    prim = next((m for m in sfm.get("models", []) if m["model_id"] == sfm.get("primary_model_id")), None)
    reg = sum(f["status"] == "ESTIMATED" for f in sfm["frames"])
    mv = bool(prim is not None and reg / max(n, 1) >= SFM_MODEL_MIN_REG_FRAC and prim["n_points3D"] >= SFM_MODEL_MIN_LANDMARKS
              and prim.get("mean_reproj_px") is not None and prim["mean_reproj_px"] <= SFM_MODEL_MAX_REPROJ_PX)
    fr = {}
    for f in sfm["frames"]:
        ok = mv and f["status"] == "ESTIMATED" and (f.get("n_points3D_observed") or 0) >= SFM_FRAME_MIN_OBS
        fr[f["frame_id"]] = {"sfm_status": f["status"], "valid": bool(ok), "T_WC": f.get("T_WC") if ok else None,
                             "n_points3D_observed": f.get("n_points3D_observed")}
    return {"status": "SFM_MODEL_VALID" if mv else "SFM_MODEL_INVALID", "model_valid": mv, "registered_fraction": reg / max(n, 1),
            "landmarks": prim["n_points3D"] if prim else 0, "mean_reproj_px": prim.get("mean_reproj_px") if prim else None, "frames": fr}


def frame_translation_window(times, edges):
    times = np.asarray(times, float)
    out = []
    for k in range(len(times)):
        lo, hi = times[k] - BR.WINDOW_S - 1e-9, times[k] + BR.WINDOW_S + 1e-9
        win = [e for e in edges if lo <= times[e["i"]] <= hi and lo <= times[e["j"]] <= hi and e.get("translation_edge") != "INSUFFICIENT"]
        if len(win) < BR.MIN_WINDOW_EDGES:
            out.append({"translation": "TRANSLATION_INSUFFICIENT_EVIDENCE", "n_valid_edges": len(win)})
            continue
        frac = sum(e["translation_edge"] == "DIRECTION_OBSERVED" for e in win) / len(win)
        rot = [e for e in win if e.get("rotation_accepted")]
        rate = float(np.median([e["rotation_deg"] / max(times[e["j"]] - times[e["i"]], 1e-6) for e in rot])) if rot else None
        out.append({"translation": "TRANSLATION_DIRECTION_ONLY" if frac >= BR.REGIME_T_FRAC else "TRANSLATION_UNRESOLVED",
                    "direction_observed_fraction": frac, "rotation_rate_deg_s": rate, "n_valid_edges": len(win),
                    "regime": ("TRANSLATING" if frac >= BR.REGIME_T_FRAC else
                               ("ROTATING_TRANSLATION_UNRESOLVED" if rate is not None and rate >= BR.ROT_RATE_DEG_S
                                else ("NEAR_STATIC_OR_BELOW_DETECTABILITY" if rate is not None else "INSUFFICIENT_EVIDENCE")))})
    return out


def camera_state(orientation_ok, sfm_frame, window):
    """Per-frame combination: FULL_6DOF only from valid SfM; never upgraded by translation labels."""
    if sfm_frame and sfm_frame.get("valid"):
        return "FULL_6DOF_AVAILABLE", "TRANSLATION_ESTIMATED_SCALE_AMBIGUOUS"
    tr = window["translation"]
    if orientation_ok:
        return "ORIENTATION_ONLY_AVAILABLE", tr
    return "CAMERA_MOTION_UNRESOLVED", tr
