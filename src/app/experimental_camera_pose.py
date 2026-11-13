"""W140 (EXPERIMENTAL, opt-in, scratch only): static-scene camera pose reconstruction -- pure numpy contracts and evaluation.

Isolated experimental camera-pose interface; NOT integrated into Production FramePose, MotionGraph, the S1 camera
calibration or any Blender root/spine/bone transform.

CONVENTIONS (frozen; never changed to improve a metric)
  * Camera optical frame "C" (OpenCV): +X right, +Y down, +Z forward (viewing direction).
  * T_WC(t) = [R_WC | c] maps camera-optical coordinates to world: X_W = R_WC X_C + c.  c is the CAMERA CENTER.
  * T_CW = T_WC^-1 = [R_CW | t_CW] with R_CW = R_WC^T, t_CW = -R_WC^T c (the "extrinsic translation" -- NOT the center).
  * Projection (OpenCV pixel convention: integer pixel coordinates are pixel CENTERS):
        x_C = R_CW X_W + t_CW;  u = fx x/z + cx,  v = fy y/z + cy.
    COLMAP uses a pixel-CORNER origin, so cx_colmap = cx_opencv + 0.5, cy_colmap = cy_opencv + 0.5.
  * Quaternions are stored SCALAR-LAST [qx, qy, qz, qw] (TUM order), unit norm, as rotation R_WC.
  * TUM groundtruth.txt "timestamp tx ty tz qx qy qz qw" = optical-center position and orientation of the RGB camera in
    the motion-capture world frame, i.e. exactly T_WC in the TUM world (metres).
  * Blender camera local axes: +X right, +Y up, -Z viewing.  R_WC(optical) = R_W,blendercam @ diag(1, -1, -1).
  * Monocular estimates live in an ARBITRARY world gauge with ARBITRARY scale ("scale_status": SCALE_AMBIGUOUS).  Any
    comparison to GT uses an evaluation-only Sim(3) alignment that never feeds back into estimation.

Frame status categories: ESTIMATED, PREDICTED_INTERPOLATED, LOST, UNOBSERVABLE, INVALID.
"""
from __future__ import annotations

import hashlib
import json
import math

import numpy as np

SCHEMA = "animcv_exp_w140_camera_pose_v1"
STATUSES = ("ESTIMATED", "PREDICTED_INTERPOLATED", "LOST", "UNOBSERVABLE", "INVALID")
N_SELECT = 300
GT_MAX_BRACKET_GAP_S = 0.05        # GT interpolation only between samples at most 50 ms apart (TUM GT ~100 Hz)
EARLY_FRACTION = 0.2               # early-segment alignment: first 20 % (by time) of associated estimated frames
RPE_DT_S = 1.0                     # RPE at ~1 s separation (plus consecutive selected frames)
SPIKE_FACTOR = 5.0                 # temporal spike: step > 5 x the median step of the same trajectory
# ---- prospective Pilot gates ----
GATE_COVERAGE = 0.70
GATE_XYZ_ATE_M = 0.20
GATE_RPY_MED_ROT_DEG = 5.0
SYN_STATIONARY_MAX_REL = 0.01      # stationary interval: max centre displacement <= 1 % of trajectory extent (aligned)
SYN_STATIONARY_MAX_ROT_DEG = 0.5
SYN_ATE_MAX_REL = 0.02             # synthetic recovery: aligned ATE RMSE <= 2 % of GT trajectory extent
SYN_MED_ROT_DEG = 1.0
SYN_COVERAGE = 0.90


class CamStop(RuntimeError):
    def __init__(self, code, detail=""):
        super().__init__(f"{code}: {detail}")
        self.code, self.detail = code, detail


# ------------------------------------------------------------------------------------------------ rotations ---
def quat_xyzw_to_R(q):
    q = np.asarray(q, float)
    n = np.linalg.norm(q, axis=-1, keepdims=True)
    if np.any(~np.isfinite(q)) or np.any(n < 1e-9):
        raise CamStop("BAD_QUATERNION", "non-finite or zero quaternion")
    x, y, z, w = np.moveaxis(q / n, -1, 0)
    R = np.stack([np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
                  np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
                  np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1)], -2)
    return R


def R_to_quat_xyzw(R):
    R = np.asarray(R, float)
    single = R.ndim == 2
    Rs = R[None] if single else R
    out = []
    for m in Rs:
        t = np.trace(m)
        if t > 0:
            s = math.sqrt(t + 1.0) * 2
            q = [(m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s, 0.25 * s]
        elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
            s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
            q = [0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s, (m[2, 1] - m[1, 2]) / s]
        elif m[1, 1] > m[2, 2]:
            s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
            q = [(m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s, (m[0, 2] - m[2, 0]) / s]
        else:
            s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
            q = [(m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s, (m[1, 0] - m[0, 1]) / s]
        q = np.asarray(q)
        q = q / np.linalg.norm(q)
        out.append(q if q[3] >= 0 else -q)
    out = np.asarray(out)
    return out[0] if single else out


def geodesic_deg(Ra, Rb):
    """Angle of Ra^T Rb in degrees (batched)."""
    M = np.einsum("...ji,...jk->...ik", np.asarray(Ra, float), np.asarray(Rb, float))
    c = (np.trace(M, axis1=-2, axis2=-1) - 1.0) / 2.0
    return np.degrees(np.arccos(np.clip(c, -1.0, 1.0)))


def is_rotation(R, tol=1e-6):
    R = np.asarray(R, float)
    return bool(np.allclose(R @ R.T, np.eye(3), atol=tol) and abs(np.linalg.det(R) - 1.0) < tol)


def slerp_R(R0, R1, w):
    q0, q1 = R_to_quat_xyzw(R0), R_to_quat_xyzw(R1)
    d = float(np.dot(q0, q1))
    if d < 0:
        q1, d = -q1, -d
    if d > 0.9995:
        q = q0 + w * (q1 - q0)
    else:
        th = math.acos(min(1.0, d))
        q = (math.sin((1 - w) * th) * q0 + math.sin(w * th) * q1) / math.sin(th)
    return quat_xyzw_to_R(q)


# ------------------------------------------------------------------------------------------------ SE(3) ---
def make_T(R, t):
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = np.asarray(R, float), np.asarray(t, float)
    return T


def inv_T(T):
    T = np.asarray(T, float)
    R, t = T[:3, :3], T[:3, 3]
    return make_T(R.T, -R.T @ t)


def camera_center_from_T_CW(T_CW):
    """Camera centre in world from a World->Camera transform: c = -R_CW^T t_CW."""
    T_CW = np.asarray(T_CW, float)
    return -T_CW[:3, :3].T @ T_CW[:3, 3]


def T_WC_from_tum(t, q_xyzw):
    return make_T(quat_xyzw_to_R(q_xyzw), t)


def blender_cam_to_optical(R_world_blendercam):
    return np.asarray(R_world_blendercam, float) @ np.diag([1.0, -1.0, -1.0])


def project(K, T_WC, X_W):
    """OpenCV-convention projection of world points (N,3) through T_WC; returns (N,2) pixels and depth z."""
    T_CW = inv_T(T_WC)
    Xc = (T_CW[:3, :3] @ np.asarray(X_W, float).T).T + T_CW[:3, 3]
    z = Xc[:, 2]
    u = K[0, 0] * Xc[:, 0] / z + K[0, 2]
    v = K[1, 1] * Xc[:, 1] / z + K[1, 2]
    return np.stack([u, v], -1), z


def check_intrinsics(K, width, height):
    K = np.asarray(K, float)
    if K.shape != (3, 3) or not np.isfinite(K).all():
        raise CamStop("BAD_CALIBRATION", "K shape/finite")
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    if not (fx > 0 and fy > 0 and 0 < cx < width and 0 < cy < height and K[0, 1] == 0 and K[1, 0] == 0 and K[2, 2] == 1
            and K[2, 0] == 0 and K[2, 1] == 0 and 0.2 < fx / fy < 5):
        raise CamStop("BAD_CALIBRATION", f"fx {fx} fy {fy} cx {cx} cy {cy} for {width}x{height}")
    return True


# ------------------------------------------------------------------------------------------------ TUM I/O ---
def read_tum_list(path, ncols):
    """Parse a TUM text file (comment lines start with '#'); refuses malformed rows."""
    rows = []
    for ln, line in enumerate(open(path, encoding="utf-8")):
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        parts = s.split()
        if len(parts) != ncols:
            raise CamStop("MALFORMED", f"{path}:{ln + 1}: {len(parts)} columns, expected {ncols}")
        rows.append(parts)
    return rows


def read_groundtruth(path):
    rows = read_tum_list(path, 8)
    a = np.asarray([[float(v) for v in r] for r in rows])
    if not np.isfinite(a).all():
        raise CamStop("MALFORMED", "non-finite GT")
    if np.any(np.diff(a[:, 0]) <= 0):
        raise CamStop("MALFORMED", "GT timestamps not strictly increasing")
    qn = np.linalg.norm(a[:, 4:8], axis=1)
    if np.any(np.abs(qn - 1) > 1e-2):
        raise CamStop("MALFORMED", f"GT quaternion norm off by {np.abs(qn - 1).max():.3g}")
    return a


def read_rgb_list(path):
    rows = read_tum_list(path, 2)
    ts = np.asarray([float(r[0]) for r in rows])
    if np.any(np.diff(ts) <= 0):
        raise CamStop("MALFORMED", "rgb timestamps not strictly increasing")
    return ts, [r[1] for r in rows]


def select_uniform(n_total, n_select=N_SELECT):
    """Deterministic uniform selection over the full sequence by index (rgb.txt order = timestamp order)."""
    if n_total <= 0:
        raise CamStop("NO_FRAMES", "empty sequence")
    if n_total <= n_select:
        return list(range(n_total))
    return sorted({int(round(v)) for v in np.linspace(0, n_total - 1, n_select)})


# ------------------------------------------------------------------------------------------------ association ---
def interpolate_gt(gt, t, max_gap=GT_MAX_BRACKET_GAP_S):
    """GT pose at time t: linear position + SLERP rotation between the bracketing GT samples (both required, gap <= max_gap).
    Returns (T_WC or None, info)."""
    ts = gt[:, 0]
    j = int(np.searchsorted(ts, t))
    near = float(np.min(np.abs(ts - t)))
    if j == 0 or j >= len(ts):
        return None, {"nearest_dt": near, "reason": "outside_gt_span"}
    t0, t1 = ts[j - 1], ts[j]
    if t1 - t0 > max_gap:
        return None, {"nearest_dt": near, "reason": f"gt_gap_{t1 - t0:.3f}s"}
    w = (t - t0) / (t1 - t0)
    p = (1 - w) * gt[j - 1, 1:4] + w * gt[j, 1:4]
    R = slerp_R(quat_xyzw_to_R(gt[j - 1, 4:8]), quat_xyzw_to_R(gt[j, 4:8]), w)
    return make_T(R, p), {"nearest_dt": near, "bracket_gap": float(t1 - t0), "w": float(w)}


# ------------------------------------------------------------------------------------------------ alignment ---
def umeyama_sim3(X, Y, with_scale=True):
    """Least-squares s, R, t with Y ~ s R X + t (Umeyama 1991); proper rotation enforced."""
    X, Y = np.asarray(X, float), np.asarray(Y, float)
    if len(X) < 3:
        raise CamStop("ALIGNMENT", "fewer than 3 correspondences")
    mx, my = X.mean(0), Y.mean(0)
    Xc, Yc = X - mx, Y - my
    var_x = (Xc ** 2).sum() / len(X)
    if var_x < 1e-18:
        raise CamStop("ALIGNMENT_DEGENERATE", "estimated positions have no spread")
    S = Yc.T @ Xc / len(X)
    U, D, Vt = np.linalg.svd(S)
    E = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        E[2, 2] = -1
    R = U @ E @ Vt
    s = float(np.trace(np.diag(D) @ E) / var_x) if with_scale else 1.0
    t = my - s * R @ mx
    return s, R, t


def apply_sim3(sim, T_WC_list):
    s, R, t = sim
    out = []
    for T in T_WC_list:
        out.append(make_T(R @ T[:3, :3], s * R @ T[:3, 3] + t))
    return out


def ate(P_est_aligned, P_gt):
    e = np.linalg.norm(np.asarray(P_est_aligned) - np.asarray(P_gt), axis=1)
    return {"rmse": float(np.sqrt(np.mean(e ** 2))), "median": float(np.median(e)), "mean": float(e.mean()), "max": float(e.max()),
            "n": int(len(e)), "per_frame": e.tolist()}


def rot_errors(R_est_aligned, R_gt):
    g = geodesic_deg(R_est_aligned, R_gt)
    return {"median_deg": float(np.median(g)), "p90_deg": float(np.percentile(g, 90)), "max_deg": float(g.max()),
            "mean_deg": float(g.mean()), "per_frame": g.tolist()}


def rpe(T_est, T_gt, times, scale, dt=RPE_DT_S):
    """Relative pose error for consecutive frames and for ~dt-second pairs.  Estimated translations scaled by the
    evaluation Sim(3) scale (gauge-invariant otherwise)."""
    def pairs_for(step_fn):
        out_t, out_r = [], []
        for i in range(len(times)):
            j = step_fn(i)
            if j is None:
                continue
            Ge = inv_T(T_est[i]) @ T_est[j]
            Gg = inv_T(T_gt[i]) @ T_gt[j]
            Ge = make_T(Ge[:3, :3], scale * Ge[:3, 3])
            E = inv_T(Gg) @ Ge
            out_t.append(float(np.linalg.norm(E[:3, 3])))
            out_r.append(float(geodesic_deg(np.eye(3), E[:3, :3])))
        return out_t, out_r

    def nxt(i):
        return i + 1 if i + 1 < len(times) else None

    def by_dt(i):
        j = int(np.searchsorted(times, times[i] + dt))
        return j if j < len(times) else None
    res = {}
    for nm, fn in (("consecutive", nxt), (f"dt_{dt:g}s", by_dt)):
        tt, rr = pairs_for(fn)
        res[nm] = {"n": len(tt), "trans_rmse": float(np.sqrt(np.mean(np.square(tt)))) if tt else None,
                   "trans_median": float(np.median(tt)) if tt else None, "rot_median_deg": float(np.median(rr)) if rr else None,
                   "rot_rmse_deg": float(np.sqrt(np.mean(np.square(rr)))) if rr else None}
    return res


def stationary_baseline(P_gt, R_gt):
    """Most favourable constant pose: the GT mean position and the chordal L2 mean of the GT rotations."""
    P_gt, R_gt = np.asarray(P_gt, float), np.asarray(R_gt, float)
    c = P_gt.mean(0)
    U, _, Vt = np.linalg.svd(R_gt.sum(0))
    Rm = U @ np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))]) @ Vt
    e = np.linalg.norm(P_gt - c, axis=1)
    g = geodesic_deg(np.broadcast_to(Rm, R_gt.shape), R_gt)
    return {"ate_rmse": float(np.sqrt(np.mean(e ** 2))), "ate_median": float(np.median(e)), "rot_median_deg": float(np.median(g)),
            "rot_p90_deg": float(np.percentile(g, 90)), "rot_max_deg": float(g.max()), "definition": "constant pose = GT mean centre + chordal-mean GT rotation (best case for a stationary baseline)"}


def temporal_increments(T_list, times, scale=1.0):
    P = np.asarray([T[:3, 3] for T in T_list]) * scale
    R = np.asarray([T[:3, :3] for T in T_list])
    dp = np.linalg.norm(np.diff(P, axis=0), axis=1)
    dr = geodesic_deg(R[:-1], R[1:])
    dtt = np.diff(np.asarray(times))
    md, mr = (float(np.median(dp)) if len(dp) else 0.0), (float(np.median(dr)) if len(dr) else 0.0)
    return {"trans_step_median": md, "trans_step_max": float(dp.max()) if len(dp) else 0.0,
            "rot_step_median_deg": mr, "rot_step_max_deg": float(dr.max()) if len(dr) else 0.0,
            "trans_spikes": [int(i) for i in np.nonzero(dp > SPIKE_FACTOR * max(md, 1e-12))[0]],
            "rot_spikes": [int(i) for i in np.nonzero(dr > SPIKE_FACTOR * max(mr, 1e-12))[0]],
            "dt_median_s": float(np.median(dtt)) if len(dtt) else None, "dt_max_s": float(dtt.max()) if len(dtt) else None,
            "trans_steps": dp.tolist(), "rot_steps_deg": dr.tolist()}


def best_time_lag(P_est_aligned, P_gt, max_lag=5):
    """Lag (in frames) minimising aligned position error; 0 expected (time-shift negative control)."""
    best = None
    n = len(P_gt)
    for lag in range(-max_lag, max_lag + 1):
        a = P_est_aligned[max(0, lag):n + min(0, lag)]
        b = P_gt[max(0, -lag):n + min(0, -lag)]
        r = float(np.sqrt(np.mean(np.sum((a - b) ** 2, axis=1))))
        if best is None or r < best[1]:
            best = (lag, r)
    return {"best_lag_frames": best[0], "rmse_at_best": best[1]}


# ------------------------------------------------------------------------------------------------ evaluation driver ---
def evaluate(est, gt, early_fraction=EARLY_FRACTION):
    """est: list of dicts {frame_id, timestamp, status, T_WC (4x4 list) or None}; gt: (N,8) TUM array.
    Returns the full evaluation dict (GT is accessed ONLY here)."""
    n_sel = len(est)
    ok = [e for e in est if e["status"] == "ESTIMATED"]
    assoc, dts, unassoc = [], [], []
    for e in ok:
        Tg, info = interpolate_gt(gt, e["timestamp"])
        dts.append(info["nearest_dt"])
        if Tg is None:
            unassoc.append({"frame_id": e["frame_id"], **info})
            continue
        assoc.append((e, Tg))
    rep = {"n_selected": n_sel, "status_counts": {s: sum(1 for e in est if e["status"] == s) for s in STATUSES},
           "coverage_estimated": len(ok) / n_sel if n_sel else 0.0, "n_associated": len(assoc), "unassociated": unassoc,
           "nearest_gt_dt_s": {"median": float(np.median(dts)) if dts else None, "p95": float(np.percentile(dts, 95)) if dts else None,
                               "max": float(np.max(dts)) if dts else None}}
    if len(assoc) < 3:
        rep["uncomputable"] = "fewer than 3 associated estimated frames"
        return rep
    times = np.asarray([e["timestamp"] for e, _ in assoc])
    Te = [np.asarray(e["T_WC"], float) for e, _ in assoc]
    Tg = [g for _, g in assoc]
    Pe, Pg = np.asarray([T[:3, 3] for T in Te]), np.asarray([T[:3, 3] for T in Tg])
    Rg = np.asarray([T[:3, :3] for T in Tg])
    # (1) standard full-trajectory Sim(3)
    sim = umeyama_sim3(Pe, Pg)
    Ta = apply_sim3(sim, Te)
    Pa, Ra = np.asarray([T[:3, 3] for T in Ta]), np.asarray([T[:3, :3] for T in Ta])
    full = {"sim3": {"scale": sim[0], "R": sim[1].tolist(), "t": sim[2].tolist(), "R_angle_deg": float(geodesic_deg(np.eye(3), sim[1]))},
            "ate": ate(Pa, Pg), "rot": rot_errors(Ra, Rg), "rpe": rpe(Te, Tg, times, sim[0]),
            "time_lag": best_time_lag(Pa, Pg)}
    # (2) early-segment Sim(3) applied unchanged to the rest
    k = max(3, int(math.ceil(early_fraction * len(assoc))))
    early = {"n_fit": k, "fit_time_span_s": [float(times[0]), float(times[k - 1])]}
    try:
        sim_e = umeyama_sim3(Pe[:k], Pg[:k])
        Tae = apply_sim3(sim_e, Te)
        Pae, Rae = np.asarray([T[:3, 3] for T in Tae]), np.asarray([T[:3, :3] for T in Tae])
        early.update({"sim3": {"scale": sim_e[0], "R_angle_deg": float(geodesic_deg(np.eye(3), sim_e[1]))},
                      "ate_rest": ate(Pae[k:], Pg[k:]) if len(assoc) > k else None, "rot_rest": rot_errors(Rae[k:], Rg[k:]) if len(assoc) > k else None,
                      "ate_fit_segment": ate(Pae[:k], Pg[:k])})
    except CamStop as e:
        early["uncomputable"] = f"{e.code}: {e.detail}"
    # GT spread (observability) and baselines
    gt_extent = float(np.linalg.norm(Pg.max(0) - Pg.min(0)))
    gt_rot_range = float(geodesic_deg(Rg[:, None], Rg[None, :]).max())
    rep.update({"times": times.tolist(), "frame_ids": [e["frame_id"] for e, _ in assoc], "full_sim3": full, "early_sim3": early,
                "baseline_stationary": stationary_baseline(Pg, Rg), "gt_extent_m": gt_extent, "gt_rotation_range_deg": gt_rot_range,
                "temporal_est_scaled": temporal_increments(Te, times, sim[0]), "temporal_gt": temporal_increments(Tg, times),
                "aligned_positions": Pa.tolist(), "gt_positions": Pg.tolist(), "aligned_rotations": Ra.tolist(), "gt_rotations": Rg.tolist(),
                "scale_status": "SCALE_AMBIGUOUS (monocular RGB; the Sim(3) scale is evaluation-only, not a recovered metric scale)"})
    return rep


# ------------------------------------------------------------------------------------------------ gates ---
def real_gates(name, ev):
    g = {"coverage": ev.get("coverage_estimated", 0.0), "coverage_ok": bool(ev.get("coverage_estimated", 0.0) >= GATE_COVERAGE)}
    if "full_sim3" not in ev:
        g["uncomputable"] = ev.get("uncomputable", "no evaluation")
        g["pass"] = False
        return g
    f, b = ev["full_sim3"], ev["baseline_stationary"]
    if name == "xyz":
        g["ate_rmse_m"] = f["ate"]["rmse"]
        g["ate_ok"] = bool(f["ate"]["rmse"] < GATE_XYZ_ATE_M)
        g["beats_baseline"] = bool(f["ate"]["rmse"] < b["ate_rmse"])
        g["baseline_ate_rmse_m"] = b["ate_rmse"]
        g["pass"] = bool(g["coverage_ok"] and g["ate_ok"] and g["beats_baseline"])
    else:
        g["rot_median_deg"] = f["rot"]["median_deg"]
        g["rot_ok"] = bool(f["rot"]["median_deg"] < GATE_RPY_MED_ROT_DEG)
        g["beats_baseline"] = bool(f["rot"]["median_deg"] < b["rot_median_deg"])
        g["baseline_rot_median_deg"] = b["rot_median_deg"]
        g["pass"] = bool(g["coverage_ok"] and g["rot_ok"] and g["beats_baseline"])
    return g


def synthetic_gates(ev, stationary_ids, contracts_ok):
    g = {"contracts_ok": bool(contracts_ok), "coverage": ev.get("coverage_estimated", 0.0)}
    if "full_sim3" not in ev:
        g["pass"] = False
        g["uncomputable"] = ev.get("uncomputable")
        return g
    f = ev["full_sim3"]
    ext = ev["gt_extent_m"]
    ids = ev["frame_ids"]
    Pa = np.asarray(ev["aligned_positions"])
    Ra = np.asarray(ev["aligned_rotations"])
    st = [i for i, fid in enumerate(ids) if fid in set(stationary_ids)]
    if len(st) >= 2:
        disp = float(np.max(np.linalg.norm(Pa[st] - Pa[st].mean(0), axis=1)) * 2)
        rot = float(geodesic_deg(Ra[st][:, None], Ra[st][None, :]).max())
    else:
        disp, rot = None, None
    g.update({"stationary_frames_estimated": len(st), "stationary_max_disp_rel": (disp / ext) if disp is not None else None,
              "stationary_max_rot_deg": rot, "ate_rel": f["ate"]["rmse"] / ext, "rot_median_deg": f["rot"]["median_deg"],
              "time_lag_frames": f["time_lag"]["best_lag_frames"]})
    g["stationary_ok"] = bool(disp is not None and disp / ext <= SYN_STATIONARY_MAX_REL and rot <= SYN_STATIONARY_MAX_ROT_DEG)
    g["recovery_ok"] = bool(g["ate_rel"] <= SYN_ATE_MAX_REL and g["rot_median_deg"] <= SYN_MED_ROT_DEG and g["coverage"] >= SYN_COVERAGE
                            and g["time_lag_frames"] == 0)
    g["pass"] = bool(g["contracts_ok"] and g["stationary_ok"] and g["recovery_ok"])
    return g


def judgment(syn_pass, xyz_pass, rpy_pass, blocked=None):
    if blocked:
        return {"judgment": "BLOCKED", "why": blocked}
    if xyz_pass and rpy_pass and syn_pass:
        return {"judgment": "SUPPORTED", "why": "synthetic + both real Pilot gates pass (Pilot utility, not Production readiness)"}
    if syn_pass or xyz_pass or rpy_pass:
        return {"judgment": "PARTIALLY_SUPPORTED", "why": "some but not all of synthetic / xyz / rpy gates pass"}
    return {"judgment": "NOT_YET_SUPPORTED", "why": "no gate passes on the selected scenes"}


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def dump(p, obj):
    s = json.dumps(obj, indent=1, sort_keys=True, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        f.write(s)
    return hashlib.sha256(s.encode()).hexdigest()
