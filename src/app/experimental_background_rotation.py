"""W142 (EXPERIMENTAL, opt-in, scratch only): observability-aware background camera ROTATION and motion-state estimation.

Complements -- never replaces -- the frozen W140 COLMAP SfM (``app.experimental_camera_pose``, whose conventions are reused
unchanged).  NOT integrated into Production, FramePose, MotionGraph, the S1 camera calibration, root motion or the Generator.

CONVENTIONS (W140, unchanged)
  * Camera optical frame (OpenCV): +X right, +Y down, +Z forward.  T_WC maps camera-optical to world; R_WC its rotation.
  * Quaternions scalar-last [qx, qy, qz, qw].  Image points use the OpenCV pixel-centre convention.
  * World here = the optical frame of the REFERENCE camera (first frame of the largest connected pair-graph component):
    R_WC(reference) = I.  It is a scene-local, reference-camera-anchored frame, NOT the GT or any metric world.
  * Bearing f = normalise(K^-1 [u, v, 1]).  A pure rotation maps f_j = R_CjCi f_i.  The relative rotation stored per edge
    is M_ij = R_CjCi (OpenCV recoverPose convention x_j = R x_i + t); hence R_WCj = R_WCi M_ij^T.

WHAT THE ESTIMATOR MAY CONSUME: RGB frames, their timestamps, the approved intrinsics and an RGB-derived keep mask.
NEVER: GT poses, GT scale, depth, IMU or oracle masks (except the labelled synthetic oracle diagnostic).

OBSERVABILITY (per frame pair = edge, model selection by Torr's GRIC on the same tentative matches)
  NO_DETECTABLE_MOTION             rotation-only model wins and its angle is below the evidence-based detectability limit
  ROTATION_ONLY_CONSISTENT         rotation-only model wins: rotation OBSERVED; translation NOT resolvable at this span
                                   (it may be zero or merely small relative to scene depth -- never reported as zero)
  TRANSLATION_PRESENT_PLANAR_AMBIGUOUS  a general homography wins but the rotation-only model does not: the warp is not a
                                   pure rotation (translation present) yet direction/structure is ambiguous (planar case)
  TRANSLATION_OBSERVED             the essential-matrix model wins: rotation and translation DIRECTION observed; scale never
  INSUFFICIENT_EVIDENCE            too few background matches / inliers
Translation MAGNITUDE and camera POSITION are never produced by this estimator (value always null).
"""
from __future__ import annotations

import math

import numpy as np

from app import experimental_camera_pose as CP

SCHEMA = "animcv_exp_w142_partial_camera_state_v1"
# ---- frozen parameters (declared before any W142 estimator output) ----
SIFT_NFEATURES = 4000
RATIO = 0.8
RANSAC_THR_PX = 1.5
ROT_RANSAC_ITERS = 1000
ROT_REFINE_ROUNDS = 3
GRIC_SIGMA_PX = 1.0
MIN_INLIERS = 30
SPANS = (1, 2, 4, 8, 16, 32)
ROT_DET_FLOOR_DEG = 0.02
CHEIRALITY_MIN_FRAC = 0.5
IRLS_C = 0.05                 # Cauchy scale on chordal residual (~2 deg)
IRLS_ITERS = 10
EDGE_OUTLIER_DEG = 5.0
WINDOW_S = 0.5                # regime window: edges with both endpoints within +-0.5 s of the frame
REGIME_T_FRAC = 0.5
ROT_RATE_DEG_S = 2.0
MIN_WINDOW_EDGES = 3
STATES = ("NO_DETECTABLE_MOTION", "ROTATION_ONLY_CONSISTENT", "TRANSLATION_PRESENT_PLANAR_AMBIGUOUS", "TRANSLATION_OBSERVED",
          "INSUFFICIENT_EVIDENCE")
REGIMES = ("NEAR_STATIC_TRANSLATION_UNRESOLVED", "ROTATING_TRANSLATION_UNRESOLVED", "TRANSLATING", "INSUFFICIENT_EVIDENCE")


# ------------------------------------------------------------------------------------------------ features ---
def sift_features(img_bgr, keep=None):
    """SIFT on the grey image restricted to keep-mask pixels; returns (xy (N,2) OpenCV centre coords, desc (N,128), n_dropped).
    Keypoints whose rounded pixel lies outside the keep mask are dropped again explicitly (belt and braces)."""
    import cv2
    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if img_bgr.ndim == 3 else img_bgr
    m = None if keep is None else np.where(np.asarray(keep, bool), 255, 0).astype(np.uint8)
    s = cv2.SIFT_create(nfeatures=SIFT_NFEATURES)
    kps, desc = s.detectAndCompute(g, m)
    if not kps:
        return np.zeros((0, 2)), np.zeros((0, 128), np.float32), 0
    xy = np.asarray([k.pt for k in kps], float)
    dropped = 0
    if keep is not None:
        ok = on_keep(xy, keep)
        dropped = int((~ok).sum())
        xy, desc = xy[ok], desc[ok]
    return xy, desc, dropped


def on_keep(xy, keep):
    """OpenCV centre coordinates -> containing pixel (round); True if that pixel is kept."""
    k = np.asarray(keep, bool)
    ix = np.clip(np.floor(np.asarray(xy)[:, 0] + 0.5).astype(int), 0, k.shape[1] - 1)
    iy = np.clip(np.floor(np.asarray(xy)[:, 1] + 0.5).astype(int), 0, k.shape[0] - 1)
    return k[iy, ix]


def mutual_ratio_matches(d1, d2, ratio=RATIO):
    import cv2
    if len(d1) < 2 or len(d2) < 2:
        return np.zeros((0, 2), int)
    bf = cv2.BFMatcher(cv2.NORM_L2)

    def best(a, b):
        out = {}
        for m in bf.knnMatch(a, b, k=2):
            if len(m) == 2 and m[0].distance < ratio * m[1].distance:
                out[m[0].queryIdx] = m[0].trainIdx
        return out
    f, r = best(d1, d2), best(d2, d1)
    pairs = [(i, j) for i, j in f.items() if r.get(j) == i]
    return np.asarray(sorted(pairs), int).reshape(-1, 2)


# ------------------------------------------------------------------------------------------------ geometry ---
def bearings(K, xy):
    h = np.column_stack([np.asarray(xy, float), np.ones(len(xy))])
    b = (np.linalg.inv(K) @ h.T).T
    return b / np.linalg.norm(b, axis=1, keepdims=True)


def project_bearing(K, f):
    p = (K @ np.asarray(f, float).T).T
    with np.errstate(divide="ignore", invalid="ignore"):
        return p[:, :2] / p[:, 2:3], p[:, 2]


def kabsch(a, b, w=None):
    """R minimising sum w ||R a_i - b_i||^2 (rows are vectors); proper rotation."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    w = np.ones(len(a)) if w is None else np.asarray(w, float)
    Hm = (a * w[:, None]).T @ b
    U, _, Vt = np.linalg.svd(Hm)
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(Vt.T @ U.T))])
    return Vt.T @ D @ U.T


def rot_sym_err2(K, R, x1, x2):
    """Symmetric transfer error^2 (mean of both directions, pixels^2) of the rotation-only homography K R K^-1."""
    f1, f2 = bearings(K, x1), bearings(K, x2)
    p2, z2 = project_bearing(K, (R @ f1.T).T)
    p1, z1 = project_bearing(K, (R.T @ f2.T).T)
    e = (np.sum((p2 - x2) ** 2, 1) + np.sum((p1 - x1) ** 2, 1)) / 2
    e[(z2 <= 0) | (z1 <= 0) | ~np.isfinite(e)] = 1e12
    return e


def homog_sym_err2(H, x1, x2):
    def tr(Hm, x):
        p = (Hm @ np.column_stack([x, np.ones(len(x))]).T).T
        with np.errstate(divide="ignore", invalid="ignore"):
            return p[:, :2] / p[:, 2:3]
    e = (np.sum((tr(H, x1) - x2) ** 2, 1) + np.sum((tr(np.linalg.inv(H), x2) - x1) ** 2, 1)) / 2
    e[~np.isfinite(e)] = 1e12
    return e


def sampson_err2(F, x1, x2):
    h1 = np.column_stack([x1, np.ones(len(x1))])
    h2 = np.column_stack([x2, np.ones(len(x2))])
    Fx1, Ftx2 = (F @ h1.T).T, (F.T @ h2.T).T
    num = np.sum(h2 * Fx1, 1) ** 2
    den = Fx1[:, 0] ** 2 + Fx1[:, 1] ** 2 + Ftx2[:, 0] ** 2 + Ftx2[:, 1] ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        e = num / den
    e[~np.isfinite(e)] = 1e12
    return e


def rotation_ransac(K, x1, x2, seed, thr=RANSAC_THR_PX, iters=ROT_RANSAC_ITERS):
    """2-point minimal rotation-only RANSAC on bearings, refined by Kabsch on inliers. Returns (R = R_CjCi, inlier mask)."""
    n = len(x1)
    if n < 2:
        return np.eye(3), np.zeros(n, bool)
    f1, f2 = bearings(K, x1), bearings(K, x2)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, (iters, 2))
    idx = idx[idx[:, 0] != idx[:, 1]]
    a = f1[idx]                                   # (m, 2, 3)
    b = f2[idx]
    Hm = np.einsum("mki,mkj->mij", a, b)
    U, _, Vt = np.linalg.svd(Hm)
    d = np.sign(np.linalg.det(np.einsum("mji,mkj->mik", Vt, U)))
    D = np.zeros((len(idx), 3, 3))
    D[:, 0, 0] = D[:, 1, 1] = 1
    D[:, 2, 2] = d
    Rs = np.einsum("mji,mjk,mlk->mil", Vt, D, U)  # V D U^T
    p = np.einsum("mij,nj->mni", Rs, f1)
    with np.errstate(divide="ignore", invalid="ignore"):
        q = np.einsum("ij,mnj->mni", K, p)
        uv = q[..., :2] / q[..., 2:3]
    err = np.sum((uv - x2[None]) ** 2, -1)
    err[~np.isfinite(err) | (p[..., 2] <= 0)] = 1e12
    cnt = (err < thr ** 2).sum(1)
    R = Rs[int(np.argmax(cnt))]
    inl = rot_sym_err2(K, R, x1, x2) < thr ** 2
    for _ in range(ROT_REFINE_ROUNDS):
        if inl.sum() < 2:
            break
        R = kabsch(f1[inl], f2[inl])
        inl = rot_sym_err2(K, R, x1, x2) < thr ** 2
    return R, inl


def skew(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def refine_essential(K, R, t, x1, x2):
    """Robust (soft-L1, scale = RANSAC threshold) Sampson refinement of (R, t) on the given matches; returns (R, unit t)."""
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation
    Kinv = np.linalg.inv(K)
    t = np.asarray(t, float).reshape(3)
    t = t / np.linalg.norm(t)
    a = np.cross(t, [1.0, 0, 0]) if abs(t[0]) < 0.9 else np.cross(t, [0, 1.0, 0])
    a /= np.linalg.norm(a)
    b = np.cross(t, a)

    def unpack(p):
        Rr = Rotation.from_rotvec(p[:3]).as_matrix() @ R
        tt = t + p[3] * a + p[4] * b
        return Rr, tt / np.linalg.norm(tt)

    def res(p):
        Rr, tt = unpack(p)
        F = Kinv.T @ skew(tt) @ Rr @ Kinv
        return np.sqrt(sampson_err2(F, x1, x2))
    s = least_squares(res, np.zeros(5), loss="soft_l1", f_scale=RANSAC_THR_PX, max_nfev=200)
    return unpack(s.x)


def gric(e2, d, k, n_dims=4, sigma=GRIC_SIGMA_PX):
    """Torr's Geometric Robust Information Criterion (lower is better)."""
    n = len(e2)
    if n == 0:
        return float("inf")
    rho = np.minimum(np.asarray(e2) / sigma ** 2, 2.0 * (n_dims - d))
    return float(rho.sum() + math.log(n_dims) * d * n + math.log(n_dims * n) * k)


def angle_deg(R):
    return float(CP.geodesic_deg(np.eye(3), R))


def analyse_pair(K, x1, x2, seed):
    """Fit rotation-only / homography / essential / fundamental models to the same tentative background matches and decide
    the observability state of this edge.  Returns a JSON-able dict (+ '_inliers' boolean mask for the chosen model)."""
    import cv2
    n = len(x1)
    out = {"n_matches": int(n)}
    if n < MIN_INLIERS:
        out.update({"state": "INSUFFICIENT_EVIDENCE", "why": "too few matches", "rotation_valid": False, "_inliers": np.zeros(n, bool)})
        return out
    x1 = np.asarray(x1, np.float64)
    x2 = np.asarray(x2, np.float64)
    R_r, inl_r = rotation_ransac(K, x1, x2, seed)
    e_r = rot_sym_err2(K, R_r, x1, x2)
    cv2.setRNGSeed(int(seed) % (2 ** 31))
    H, mh = cv2.findHomography(x1, x2, cv2.RANSAC, RANSAC_THR_PX, maxIters=2000, confidence=0.999)
    e_h = homog_sym_err2(H, x1, x2) if H is not None else np.full(n, 1e12)
    cv2.setRNGSeed(int(seed) % (2 ** 31))
    E, me = cv2.findEssentialMat(x1, x2, K, method=cv2.RANSAC, prob=0.999, threshold=RANSAC_THR_PX)
    R_e, t_e, n_cheir = None, None, 0
    if E is not None and E.shape[0] >= 3:
        E = E[:3]
        Kinv = np.linalg.inv(K)
        e_e = sampson_err2(Kinv.T @ E @ Kinv, x1, x2)
        inl_e = e_e < RANSAC_THR_PX ** 2
        try:
            n_cheir, R_e, t_e, _ = cv2.recoverPose(E, x1[inl_e], x2[inl_e], K)
        except cv2.error:
            R_e = None
        if R_e is not None and inl_e.sum() >= 8 and np.linalg.norm(t_e) > 0:
            R_e, t_e = refine_essential(K, np.asarray(R_e), t_e, x1[inl_e], x2[inl_e])
            e_e = sampson_err2(Kinv.T @ skew(t_e) @ R_e @ Kinv, x1, x2)
            inl_e = e_e < RANSAC_THR_PX ** 2
    else:
        e_e = np.full(n, 1e12)
        inl_e = np.zeros(n, bool)
    cv2.setRNGSeed(int(seed) % (2 ** 31))
    F, mf = cv2.findFundamentalMat(x1, x2, cv2.FM_RANSAC, RANSAC_THR_PX, 0.999)
    e_f = sampson_err2(F[:3], x1, x2) if F is not None and F.shape[0] >= 3 else np.full(n, 1e12)
    g = {"R": gric(e_r, 2, 3), "H": gric(e_h, 2, 8), "E": gric(e_e, 3, 5), "F": gric(e_f, 3, 7)}
    inl = {"R": e_r < RANSAC_THR_PX ** 2, "H": e_h < RANSAC_THR_PX ** 2, "E": inl_e, "F": e_f < RANSAC_THR_PX ** 2}
    calib_best = min(("R", "H", "E"), key=lambda m: (g[m], "RHE".index(m)))
    kfree_best = min(("H", "F"), key=lambda m: (g[m], "HF".index(m)))
    n_best = int(inl[calib_best].sum())
    fpx = float((K[0, 0] + K[1, 1]) / 2)
    out.update({"gric": g, "n_inliers": {m: int(v.sum()) for m, v in inl.items()}, "best_calibrated_model": calib_best,
                "best_kfree_model": kfree_best, "flow_raw_median_px": float(np.median(np.linalg.norm(x2 - x1, axis=1))),
                "rot_residual_median_px": float(np.sqrt(np.median(e_r[inl[calib_best]]))) if n_best else None,
                "rot_residual_p90_px": float(np.sqrt(np.percentile(e_r[inl[calib_best]], 90))) if n_best else None})
    if n_best < MIN_INLIERS:
        out.update({"state": "INSUFFICIENT_EVIDENCE", "why": "too few inliers for the best model", "rotation_valid": False,
                    "_inliers": inl[calib_best]})
        return out
    if calib_best == "R":
        th = angle_deg(R_r)
        det = max(ROT_DET_FLOOR_DEG, math.degrees(3 * GRIC_SIGMA_PX / (fpx * math.sqrt(max(int(inl["R"].sum()), 1)))))
        out.update({"state": "NO_DETECTABLE_MOTION" if th < det else "ROTATION_ONLY_CONSISTENT", "M": R_r.tolist(), "rotation_valid": True,
                    "rotation_source": "rotation_only_model", "rotation_deg": th, "detectability_deg": det,
                    "translation": "UNRESOLVED_AT_THIS_SPAN",
                    "translation_to_depth_bound": float(np.sqrt(np.percentile(e_r[inl["R"]], 90))) / fpx,
                    "_inliers": inl["R"]})
        return out
    cheir_ok = R_e is not None and n_cheir >= CHEIRALITY_MIN_FRAC * max(int(inl_e.sum()), 1) and n_cheir >= MIN_INLIERS
    if cheir_ok:
        d_i = -(R_e.T @ np.asarray(t_e).reshape(3))
        d_i = d_i / max(np.linalg.norm(d_i), 1e-12)
        out.update({"M": np.asarray(R_e).tolist(), "rotation_valid": True, "rotation_source": "essential_recoverPose",
                    "rotation_deg": angle_deg(R_e), "n_cheirality": int(n_cheir)})
    else:
        out.update({"rotation_valid": False, "rotation_source": None, "n_cheirality": int(n_cheir)})
    if calib_best == "H":
        out.update({"state": "TRANSLATION_PRESENT_PLANAR_AMBIGUOUS", "translation": "PRESENT_DIRECTION_AMBIGUOUS", "_inliers": inl["H"]})
    else:
        out.update({"state": "TRANSLATION_OBSERVED", "translation": "DIRECTION_ONLY_SCALE_UNOBSERVABLE", "_inliers": inl["E"]})
        if cheir_ok:
            out["translation_direction_in_cam_i"] = d_i.tolist()
    return out


# ------------------------------------------------------------------------------------------------ rotation averaging ---
def components(n, edges):
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    for i, j in edges:
        parent[find(i)] = find(j)
    roots = [find(i) for i in range(n)]
    return roots


def rotation_averaging(n, edges, Ms, ref=None):
    """Robust chordal rotation averaging.  edges: [(i, j)], Ms: M_ij = R_CjCi.  Unknowns Q_i = R_CiW (= R_WCi^T) with
    Q_j = M_ij Q_i and Q_ref = I.  Linear least squares (3 independent columns) + Cauchy IRLS, SO(3) projection.
    Returns (R_WC list or None per node, info)."""
    if not edges:
        return [None] * n, {"component": [], "reference": None}
    roots = components(n, [e for e in edges])
    used = {i for e in edges for i in e}
    comp_sizes = {}
    for i in used:
        comp_sizes[roots[i]] = comp_sizes.get(roots[i], 0) + 1
    big = max(comp_sizes, key=lambda r: (comp_sizes[r], -min(i for i in used if roots[i] == r)))
    nodes = sorted(i for i in used if roots[i] == big)
    ref = nodes[0] if ref is None or ref not in nodes else ref
    loc = {v: k for k, v in enumerate(nodes)}
    E = [(loc[i], loc[j], np.asarray(M, float)) for (i, j), M in zip(edges, Ms) if i in loc and j in loc]
    m, N = len(E), len(nodes)
    w = np.ones(m)
    free = [k for k in range(N) if k != loc[ref]]
    col = {k: c for c, k in enumerate(free)}
    Q = None
    for it in range(IRLS_ITERS + 1):
        A = np.zeros((3 * m, 3 * len(free)))
        B = np.zeros((3 * m, 3))
        for e, (a, b, M) in enumerate(E):
            sw = math.sqrt(w[e])
            r = slice(3 * e, 3 * e + 3)
            # Q_b - M Q_a = 0
            if b in col:
                A[r, 3 * col[b]:3 * col[b] + 3] += sw * np.eye(3)
            else:
                B[r] -= sw * np.eye(3)
            if a in col:
                A[r, 3 * col[a]:3 * col[a] + 3] -= sw * M
            else:
                B[r] += sw * M
        X = np.linalg.lstsq(A, B, rcond=None)[0]
        Q = [np.eye(3)] * N
        Q = list(Q)
        for k, c in col.items():
            U, _, Vt = np.linalg.svd(X[3 * c:3 * c + 3])
            Q[k] = U @ np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))]) @ Vt
        res = np.asarray([np.linalg.norm(Q[b] - M @ Q[a]) for a, b, M in E])
        if it < IRLS_ITERS:
            w = 1.0 / (1.0 + (res / IRLS_C) ** 2)
    res_deg = np.asarray([CP.geodesic_deg(Q[b], M @ Q[a]) for a, b, M in E])
    out = [None] * n
    support = np.zeros(N, int)
    for (a, b, _), r in zip(E, res_deg):
        if r <= EDGE_OUTLIER_DEG:
            support[a] += 1
            support[b] += 1
    for k, v in enumerate(nodes):
        out[v] = Q[k].T                          # R_WC = Q^T
    return out, {"reference": ref, "component": nodes, "n_edges": m, "edge_residual_deg": res_deg.tolist(),
                 "n_outlier_edges": int((res_deg > EDGE_OUTLIER_DEG).sum()), "support": {v: int(support[loc[v]]) for v in nodes},
                 "n_components_with_edges": len(comp_sizes)}


# ------------------------------------------------------------------------------------------------ regimes ---
def frame_regimes(times, edges_info):
    """edges_info: list of dicts with i, j, state, rotation_deg.  Per-frame regime from edges inside the +-WINDOW_S window."""
    times = np.asarray(times, float)
    out = []
    for k in range(len(times)):
        lo, hi = times[k] - WINDOW_S, times[k] + WINDOW_S
        win = [e for e in edges_info if lo - 1e-9 <= times[e["i"]] <= hi + 1e-9 and lo - 1e-9 <= times[e["j"]] <= hi + 1e-9]
        valid = [e for e in win if e["state"] != "INSUFFICIENT_EVIDENCE"]
        if len(valid) < MIN_WINDOW_EDGES:
            out.append({"regime": "INSUFFICIENT_EVIDENCE", "n_valid_edges": len(valid), "n_edges": len(win)})
            continue
        tf = sum(e["state"] in ("TRANSLATION_OBSERVED", "TRANSLATION_PRESENT_PLANAR_AMBIGUOUS") for e in valid) / len(valid)
        rates = [e["rotation_deg"] / max(times[e["j"]] - times[e["i"]], 1e-6) for e in valid if e.get("rotation_deg") is not None]
        rate = float(np.median(rates)) if rates else 0.0
        if tf >= REGIME_T_FRAC:
            reg = "TRANSLATING"
        elif rate >= ROT_RATE_DEG_S:
            reg = "ROTATING_TRANSLATION_UNRESOLVED"
        else:
            reg = "NEAR_STATIC_TRANSLATION_UNRESOLVED"
        out.append({"regime": reg, "translation_fraction": tf, "rotation_rate_deg_s": rate, "n_valid_edges": len(valid), "n_edges": len(win)})
    return out


def sequence_regime(fr):
    c = {}
    for r in fr:
        if r["regime"] != "INSUFFICIENT_EVIDENCE":
            c[r["regime"]] = c.get(r["regime"], 0) + 1
    if not c:
        return {"regime": "INSUFFICIENT_EVIDENCE", "counts": {}}
    best = sorted(c.items(), key=lambda kv: (-kv[1], REGIMES.index(kv[0])))
    if len(best) > 1 and best[0][1] == best[1][1]:
        return {"regime": "INSUFFICIENT_EVIDENCE", "counts": c, "why": "tie"}
    return {"regime": best[0][0], "counts": c}


# ------------------------------------------------------------------------------------------------ evaluation (GT side) ---
def chordal_align(Rg, Re):
    """Evaluation-only orientation-reference alignment: R_a minimising sum ||R_a Re_i - Rg_i||_F."""
    M = np.einsum("nij,nkj->ik", np.asarray(Rg, float), np.asarray(Re, float))
    U, _, Vt = np.linalg.svd(M)
    return U @ np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))]) @ Vt


def orientation_errors(Rg, Re):
    Ra = chordal_align(Rg, Re)
    g = CP.geodesic_deg(np.einsum("ij,njk->nik", Ra, np.asarray(Re)), np.asarray(Rg))
    return Ra, g


def relative_errors(times, Rg, Re, dt=None):
    """Consecutive (dt None) or ~dt-second relative-rotation errors and the identity-baseline errors on the same pairs."""
    times = np.asarray(times)
    est, base, pairs = [], [], []
    for i in range(len(times)):
        j = i + 1 if dt is None else int(np.searchsorted(times, times[i] + dt))
        if j >= len(times):
            continue
        Ge = np.asarray(Re[i]).T @ np.asarray(Re[j])
        Gg = np.asarray(Rg[i]).T @ np.asarray(Rg[j])
        est.append(float(CP.geodesic_deg(Ge, Gg)))
        base.append(float(CP.geodesic_deg(np.eye(3), Gg)))
        pairs.append((i, j))
    return np.asarray(est), np.asarray(base), pairs


def gt_regime_labels(times, P, R, disp_m=0.05, rot_deg=1.0):
    """Evaluation labels: TRANSLATING if GT centre displacement within +-WINDOW_S >= disp_m; ROTATING if rotation range >=
    rot_deg; else NEAR_STATIC."""
    times = np.asarray(times)
    out = []
    for k in range(len(times)):
        ix = np.nonzero(np.abs(times - times[k]) <= WINDOW_S + 1e-9)[0]
        Pw, Rw = np.asarray(P)[ix], np.asarray(R)[ix]
        d = float(np.linalg.norm(Pw - Pw[:1], axis=1).max()) if len(ix) else 0.0
        d = max(d, float(np.linalg.norm(Pw[:, None] - Pw[None], axis=-1).max()) if len(ix) > 1 else 0.0)
        r = float(CP.geodesic_deg(Rw[:, None], Rw[None]).max()) if len(ix) > 1 else 0.0
        out.append("TRANSLATING" if d >= disp_m else ("ROTATING" if r >= rot_deg else "NEAR_STATIC"))
    return out
