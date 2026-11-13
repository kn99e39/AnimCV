"""W141 (EXPERIMENTAL, opt-in, scratch only): dynamic-scene camera pose robustness -- masks, feature accounting, evaluation.

Isolated experimental extension of W140 (``app.experimental_camera_pose``, imported and reused UNCHANGED: conventions,
SE(3)/quaternion maths, GT association, Umeyama Sim(3), ATE/RPE, stationary baseline).  NOT integrated into Production,
FramePose, MotionGraph, the S1 camera calibration, root motion or the Dual-Stage Generator.

Conventions are W140's (T_WC optical OpenCV axes, scalar-last quaternions, SCALE_AMBIGUOUS monocular estimates, evaluation-
only alignment).  Additional W141 conventions:
  * Exclusion mask E (uint8, 255 = exclude from camera geometry) in image pixel coordinates (row = y, col = x).
  * COLMAP keep mask = 255 - E  (COLMAP drops keypoints on zero pixels).  COLMAP keypoint (x, y) uses a pixel-CORNER origin,
    so the pixel containing it is (floor(y), floor(x)); OpenCV centre coordinates are (x - 0.5, y - 0.5).
  * Condition B mask  = disk-dilate(RGB person mask, DILATE_PX).
  * Condition C mask  = area-matched static-region control built deterministically from B's mask (``area_control_mask``).
  * Condition D mask  = disk-dilate(oracle foreground mask, DILATE_PX)  (synthetic only, ORACLE diagnostic).
"""
from __future__ import annotations

import math
import sqlite3

import numpy as np

from app import experimental_camera_pose as CP

SCHEMA = "animcv_exp_w141_dynamic_camera_pose_v1"
DILATE_PX = 15                        # disk radius added around every predicted person (boundary error, blur, SIFT support)
AREA_OFFSETS = ((0.5, 0.0), (0.25, 0.0), (0.75, 0.0), (0.5, 0.5), (0.25, 0.5), (0.75, 0.5), (0.0, 0.5))  # (dx/W, dy/H)
EARLY_FRACTION = 0.2                  # early-reference interval: first 20 % (by time) of associated ESTIMATED frames
EARLY_MIN_FRAMES = 10
EARLY_MIN_SPREAD_M = 0.05             # largest GT position singular value (RMS spread, m) needed to observe scale/translation
SPIKE_FACTOR = 5.0
SPIKE_FLOOR_M = 0.02
SPIKE_FLOOR_DEG = 2.0
# ---- prospective Pilot gates (frozen) ----
COVERAGE_FLOOR = 0.80
ATE_TARGET_M = {"walking_static": 0.05, "walking_xyz": 0.15}
ROT_MED_TARGET_DEG = 5.0
BENEFIT_REL = 0.20                    # B must beat A by >= 20 % AND by an absolute margin
BENEFIT_ABS_ATE_M = 0.005
BENEFIT_ABS_ROT_DEG = 0.5
COVERAGE_LOSS_MAX = 0.10
MIN_COMMON = 30
SYN_DISTURB_ATE_RATIO = 2.0           # synthetic disturbance: ATE_rel > 2x clean AND > clean + 0.2 % of extent, ...
SYN_DISTURB_ATE_ABS_REL = 0.002
SYN_DISTURB_ROT_DEG = 0.5
SYN_DISTURB_COVERAGE = 0.10


# ------------------------------------------------------------------------------------------------ masks ---
def binary(mask):
    m = np.asarray(mask)
    if m.ndim != 2:
        raise CP.CamStop("MASK_CONTRACT", f"mask must be 2-D, got {m.shape}")
    return m > 127 if m.dtype == np.uint8 else m.astype(bool)


def disk_dilate(mask, r=DILATE_PX):
    import cv2
    m = binary(mask).astype(np.uint8)
    if r <= 0:
        return m.astype(bool)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    return cv2.dilate(m, k).astype(bool)


def keep_from_exclusion(excl):
    """COLMAP keep mask (uint8 0/255) from a boolean exclusion mask."""
    return np.where(binary(excl), 0, 255).astype(np.uint8)


def area_control_mask(H, offsets=AREA_OFFSETS):
    """Deterministic static-region control with the same pixel area as exclusion mask H, disjoint from H.

    1. For each declared offset (fractions of W, H; wrap-around shift), shift H; pick the offset with the least overlap with
       H (ties -> first declared).  2. C = shifted \\ H.  3. If |C| < |H|, grow C by adding non-H pixels in order of their
       distance to C (ties in raster order) until |C| = |H| or the background is exhausted (shortfall reported).
    Uses only the (RGB-derived) mask; never GT."""
    H = binary(H)
    h, w = H.shape
    a = int(H.sum())
    info = {"target_px": a}
    if a == 0:
        info.update({"offset": None, "area_px": 0, "shortfall_px": 0, "overlap_px": 0})
        return np.zeros_like(H), info
    best = None
    for k, (fx, fy) in enumerate(offsets):
        S = np.roll(H, (int(round(fy * h)), int(round(fx * w))), axis=(0, 1))
        ov = int((S & H).sum())
        if best is None or ov < best[0]:
            best = (ov, k, S)
    ov, k, S = best
    C = S & ~H
    deficit = a - int(C.sum())
    if deficit > 0:
        import cv2
        free = ~H & ~C
        if C.any():
            dist = cv2.distanceTransform((~C).astype(np.uint8), cv2.DIST_L2, 5)
        else:
            dist = np.zeros(H.shape, np.float32)
        cand = np.flatnonzero(free.ravel())
        order = cand[np.lexsort((cand, dist.ravel()[cand]))]
        add = order[:deficit]
        C = C.copy().ravel()
        C[add] = True
        C = C.reshape(H.shape)
    info.update({"offset": list(offsets[k]), "overlap_px": ov, "area_px": int(C.sum()), "shortfall_px": int(a - C.sum()),
                 "disjoint": bool(not (C & H).any())})
    return C, info


def keypoints_in_mask(xy_colmap, mask):
    """Boolean per keypoint: does its pixel (floor of COLMAP corner-origin coordinates) lie in the mask?"""
    m = binary(mask)
    xy = np.asarray(xy_colmap, float).reshape(-1, 2)
    if len(xy) == 0:
        return np.zeros(0, bool)
    ix = np.clip(np.floor(xy[:, 0]).astype(int), 0, m.shape[1] - 1)
    iy = np.clip(np.floor(xy[:, 1]).astype(int), 0, m.shape[0] - 1)
    return m[iy, ix]


def mask_quality(pred, ref):
    p, r = binary(pred), binary(ref)
    inter, uni = int((p & r).sum()), int((p | r).sum())
    return {"iou": inter / uni if uni else None, "recall": inter / int(r.sum()) if r.any() else None,
            "precision": inter / int(p.sum()) if p.any() else None, "pred_px": int(p.sum()), "ref_px": int(r.sum())}


# ------------------------------------------------------------------------------------------------ COLMAP readers ---
def read_db_keypoints(db_path):
    """{image name: (N, 2) COLMAP corner-origin keypoint xy} straight from the COLMAP sqlite database."""
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        names = dict(con.execute("SELECT image_id, name FROM images"))
        out = {}
        for iid, rows, cols, data in con.execute("SELECT image_id, rows, cols, data FROM keypoints"):
            a = np.frombuffer(data, np.float32).reshape(rows, cols) if rows else np.zeros((0, max(cols, 2)), np.float32)
            out[names[iid]] = a[:, :2].astype(float)
        return out
    finally:
        con.close()


def read_images_txt(path):
    """{image name: {"xy": (M,2) observations with a 3D point (COLMAP corner origin), "pid": (M,) point3D ids}}."""
    lines = [l for l in open(path, encoding="utf-8").read().splitlines() if not l.startswith("#")]
    out = {}
    for i in range(0, len(lines) - 1, 2):
        hdr, pts = lines[i].split(), lines[i + 1].split()
        if len(hdr) < 10:
            continue
        arr = np.asarray(pts, float).reshape(-1, 3) if pts else np.zeros((0, 3))
        keep = arr[:, 2] >= 0
        out[hdr[9]] = {"xy": arr[keep, :2], "pid": arr[keep, 2].astype(np.int64)}
    return out


# ------------------------------------------------------------------------------------------------ evaluation ---
def read_groundtruth_dedup(path):
    """W140 GT reader semantics, except that samples sharing a DUPLICATED timestamp are all removed (no arbitrary pick).

    W141 amendment 1 (before any GT score): the official fr3 walking_static groundtruth.txt contains one timestamp twice
    with two slightly different poses, which CP.read_groundtruth refuses.  Removing every sample of a duplicated timestamp
    leaves the W140 bracket-gap rule (<= 50 ms) to decide association.  Returns (array, info)."""
    rows = CP.read_tum_list(path, 8)
    a = np.asarray([[float(v) for v in r] for r in rows])
    ts, counts = np.unique(a[:, 0], return_counts=True)
    dup = set(ts[counts > 1].tolist())
    keep = np.asarray([t not in dup for t in a[:, 0]], bool)
    info = {"n_raw": int(len(a)), "duplicated_timestamps": sorted(dup), "n_removed": int((~keep).sum()),
            "removed_rows": a[~keep].tolist()}
    a = a[keep]
    if not np.isfinite(a).all():
        raise CP.CamStop("MALFORMED", "non-finite GT")
    if np.any(np.diff(a[:, 0]) <= 0):
        raise CP.CamStop("MALFORMED", "GT timestamps not strictly increasing after duplicate removal (out-of-order samples)")
    qn = np.linalg.norm(a[:, 4:8], axis=1)
    if np.any(np.abs(qn - 1) > 1e-2):
        raise CP.CamStop("MALFORMED", f"GT quaternion norm off by {np.abs(qn - 1).max():.3g}")
    return a, info


def chordal_rotation(Rg, Re):
    """R minimising sum ||R Re_i - Rg_i||_F (orientation-based gauge rotation)."""
    M = np.einsum("nij,nkj->ik", np.asarray(Rg, float), np.asarray(Re, float))
    U, _, Vt = np.linalg.svd(M)
    return U @ np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))]) @ Vt


def pose_aware_sim3(Pe, Re, Pg, Rg):
    """Evaluation-only alignment using orientations for the rotation and positions for scale/translation."""
    R = chordal_rotation(Rg, Re)
    Pe, Pg = np.asarray(Pe, float), np.asarray(Pg, float)
    q = (R @ (Pe - Pe.mean(0)).T).T
    g = Pg - Pg.mean(0)
    den = float((q * q).sum())
    s = float((q * g).sum() / den) if den > 1e-18 else float("nan")
    t = Pg.mean(0) - s * R @ Pe.mean(0)
    return s, R, t


def observability(Pg, Rg, Pe=None):
    Pg = np.asarray(Pg, float)
    sv = np.linalg.svd(Pg - Pg.mean(0), compute_uv=False) / math.sqrt(max(len(Pg), 1)) if len(Pg) >= 2 else np.zeros(3)
    rot_div = float(CP.geodesic_deg(np.asarray(Rg)[:, None], np.asarray(Rg)[None]).max()) if len(Rg) >= 2 else 0.0
    out = {"n": int(len(Pg)), "gt_position_sv_rms_m": [float(v) for v in sv], "gt_orientation_diversity_deg": rot_div,
           "gt_position_rank_1cm": int((sv >= 0.01).sum())}
    if Pe is not None and len(Pe) >= 2:
        out["est_position_sv_rms"] = [float(v) for v in np.linalg.svd(np.asarray(Pe) - np.mean(Pe, 0), compute_uv=False) / math.sqrt(len(Pe))]
    return out


def slope(t, y):
    t, y = np.asarray(t, float), np.asarray(y, float)
    if len(t) < 3 or np.ptp(t) <= 0:
        return None
    return float(np.polyfit(t, y, 1)[0])


def consecutive_relative_errors(Te, Tg, scale):
    out = []
    for i in range(len(Te) - 1):
        Ge = CP.inv_T(Te[i]) @ Te[i + 1]
        Gg = CP.inv_T(Tg[i]) @ Tg[i + 1]
        Ge = CP.make_T(Ge[:3, :3], scale * Ge[:3, 3])
        E = CP.inv_T(Gg) @ Ge
        out.append((float(np.linalg.norm(E[:3, 3])), float(CP.geodesic_deg(np.eye(3), E[:3, :3]))))
    return np.asarray(out).reshape(-1, 2)


def lost_intervals(est):
    runs, cur = [], None
    for e in est:
        if e["status"] != "ESTIMATED":
            if cur is None:
                cur = {"first_frame_id": e["frame_id"], "t0": e["timestamp"], "n": 0}
            cur.update({"last_frame_id": e["frame_id"], "t1": e["timestamp"]})
            cur["n"] += 1
        elif cur is not None:
            runs.append(cur)
            cur = None
    if cur is not None:
        runs.append(cur)
    return runs


def evaluate_dynamic(est, gt, early_fraction=EARLY_FRACTION):
    """W140 evaluation (unchanged) plus W141 diagnostics: per-frame errors, GT-referenced spikes, path lengths, lost intervals,
    early-interval observability and the pose-aware early-reference alignment (evaluation only)."""
    ev = CP.evaluate(est, gt, early_fraction)
    ev["lost_intervals"] = lost_intervals(est)
    ev["w140_position_only_early_alignment_note"] = ("early_sim3 is W140's position-only protocol, kept for continuity only: it is "
                                                     "ill-conditioned on low-rank early motion and is NOT interpreted as drift in W141")
    if "full_sim3" not in ev:
        return ev
    times = np.asarray(ev["times"])
    ids = ev["frame_ids"]
    by_id = {e["frame_id"]: e for e in est}
    Te = [np.asarray(by_id[i]["T_WC"], float) for i in ids]
    Pg, Rg = np.asarray(ev["gt_positions"]), np.asarray(ev["gt_rotations"])
    Tg = [CP.make_T(R, p) for R, p in zip(Rg, Pg)]
    Pe = np.asarray([T[:3, 3] for T in Te])
    Re = np.asarray([T[:3, :3] for T in Te])
    f = ev["full_sim3"]
    s = f["sim3"]["scale"]
    Pa = np.asarray(ev["aligned_positions"])
    rel = consecutive_relative_errors(Te, Tg, s)
    if len(rel):
        mt, mr = float(np.median(rel[:, 0])), float(np.median(rel[:, 1]))
        thr_t, thr_r = max(SPIKE_FACTOR * mt, SPIKE_FLOOR_M), max(SPIKE_FACTOR * mr, SPIKE_FLOOR_DEG)
        spikes = [{"pair_frame_ids": [ids[i], ids[i + 1]], "t": float(times[i + 1]), "trans_err": float(rel[i, 0]),
                   "rot_err_deg": float(rel[i, 1])} for i in range(len(rel)) if rel[i, 0] > thr_t or rel[i, 1] > thr_r]
    else:
        mt = mr = thr_t = thr_r = None
        spikes = []
    ev["spikes_gt_referenced"] = {"definition": f"consecutive relative pose error > max({SPIKE_FACTOR} x median, {SPIKE_FLOOR_M} m / "
                                                f"{SPIKE_FLOOR_DEG} deg)", "median_trans": mt, "median_rot_deg": mr, "thr_trans": thr_t,
                                  "thr_rot_deg": thr_r, "n": len(spikes), "spikes": spikes,
                                  "per_pair_trans": rel[:, 0].tolist() if len(rel) else [], "per_pair_rot_deg": rel[:, 1].tolist() if len(rel) else []}
    ev["path_length"] = {"est_scaled_m": float(np.sum(np.linalg.norm(np.diff(Pa, axis=0), axis=1))),
                         "gt_m": float(np.sum(np.linalg.norm(np.diff(Pg, axis=0), axis=1))),
                         "est_rotation_path_deg": float(np.sum(CP.geodesic_deg(Re[:-1], Re[1:]))),
                         "gt_rotation_path_deg": float(np.sum(CP.geodesic_deg(Rg[:-1], Rg[1:])))}
    ev["drift_full_sim3"] = {"pos_err_slope_per_s": slope(times, f["ate"]["per_frame"]), "rot_err_slope_deg_per_s": slope(times, f["rot"]["per_frame"])}
    ev["observability_full"] = observability(Pg, Rg, Pe)
    k = max(3, int(math.ceil(early_fraction * len(ids))))
    obs = observability(Pg[:k], Rg[:k], Pe[:k])
    er = {"n_fit": k, "fit_time_span_s": [float(times[0]), float(times[k - 1])], "observability": obs,
          "rule": f"OBSERVABLE if n_fit >= {EARLY_MIN_FRAMES}, GT RMS spread sv1 >= {EARLY_MIN_SPREAD_M} m and fitted scale > 0; "
                  "otherwise ROTATION_ONLY (gauge rotation from orientations; scale/translation UNOBSERVABLE). Never repaired with later GT."}
    R_only = chordal_rotation(Rg[:k], Re[:k])
    Rrest = np.einsum("ij,njk->nik", R_only, Re[k:])
    er["rotation_only"] = {"rot_rest": CP.rot_errors(Rrest, Rg[k:]) if len(ids) > k else None,
                           "rot_slope_deg_per_s": slope(times[k:], CP.geodesic_deg(Rrest, Rg[k:])) if len(ids) > k + 2 else None}
    status = "ROTATION_ONLY"
    if k >= EARLY_MIN_FRAMES and obs["gt_position_sv_rms_m"][0] >= EARLY_MIN_SPREAD_M:
        se, Ra_, te = pose_aware_sim3(Pe[:k], Re[:k], Pg[:k], Rg[:k])
        if np.isfinite(se) and se > 0:
            status = "OBSERVABLE"
            Par = (se * (Ra_ @ Pe.T)).T + te
            Rar = np.einsum("ij,njk->nik", Ra_, Re)
            if len(ids) > k:
                e_rest = np.linalg.norm(Par[k:] - Pg[k:], axis=1)
                er["pose_aware"] = {"scale": se, "scale_ratio_to_full": se / s, "R_angle_to_full_deg": float(CP.geodesic_deg(Ra_, np.asarray(f["sim3"]["R"]))),
                                    "ate_rest": CP.ate(Par[k:], Pg[k:]), "rot_rest": CP.rot_errors(Rar[k:], Rg[k:]),
                                    "pos_slope_per_s": slope(times[k:], e_rest), "rot_slope_deg_per_s": slope(times[k:], CP.geodesic_deg(Rar[k:], Rg[k:]))}
    er["status"] = status
    ev["early_reference"] = er
    return ev


def evaluate_on_ids(est, gt, ids):
    """Accuracy restricted to a frame-ID subset (e.g. the common registered intersection), with its own Sim(3).
    Coverage numbers in the returned dict refer to the subset only and are NOT the coverage result."""
    keep = set(ids)
    sub = [e for e in est if e["frame_id"] in keep]
    ev = CP.evaluate(sub, gt)
    out = {"n": len(sub)}
    if "full_sim3" in ev:
        f = ev["full_sim3"]
        out.update({"ate_rmse": f["ate"]["rmse"], "ate_median": f["ate"]["median"], "ate_p90": float(np.percentile(f["ate"]["per_frame"], 90)),
                    "rot_median_deg": f["rot"]["median_deg"], "rot_p90_deg": f["rot"]["p90_deg"], "rpe": f["rpe"], "scale": f["sim3"]["scale"],
                    "frame_ids": ev["frame_ids"], "ate_per_frame": f["ate"]["per_frame"], "rot_per_frame": f["rot"]["per_frame"]})
    else:
        out["uncomputable"] = ev.get("uncomputable")
    return out


# ------------------------------------------------------------------------------------------------ gates ---
def config_gate(seq, ev):
    """Per real sequence x condition: coverage floor, exploratory accuracy targets, and beating the stationary baseline."""
    g = {"coverage": ev.get("coverage_estimated", 0.0)}
    g["coverage_ok"] = bool(g["coverage"] >= COVERAGE_FLOOR)
    if "full_sim3" not in ev:
        g.update({"pass": False, "uncomputable": ev.get("uncomputable", "no evaluation")})
        return g
    f, b = ev["full_sim3"], ev["baseline_stationary"]
    g.update({"ate_rmse_m": f["ate"]["rmse"], "rot_median_deg": f["rot"]["median_deg"], "ate_target_m": ATE_TARGET_M[seq],
              "baseline_ate_rmse_m": b["ate_rmse"], "baseline_rot_median_deg": b["rot_median_deg"]})
    g["accuracy_target_met"] = bool(f["ate"]["rmse"] <= ATE_TARGET_M[seq] and f["rot"]["median_deg"] <= ROT_MED_TARGET_DEG)
    g["beats_baseline"] = bool(f["ate"]["rmse"] < b["ate_rmse"] and f["rot"]["median_deg"] < b["rot_median_deg"])
    g["pass"] = bool(g["coverage_ok"] and g["accuracy_target_met"] and g["beats_baseline"])
    return g


def _better(x, y, abs_margin):
    """x improves on y by >= BENEFIT_REL relative AND >= abs_margin absolute."""
    if x is None or y is None:
        return False
    return bool(x <= (1 - BENEFIT_REL) * y and (y - x) >= abs_margin)


def improvement(cov_x, cov_a, cx, ca):
    """Does condition x improve on A?  cx / ca: common-intersection accuracy dicts (evaluate_on_ids)."""
    rescue = bool(cov_x >= COVERAGE_FLOOR and cov_a < COVERAGE_FLOOR)
    acc = False
    if cx.get("n", 0) >= MIN_COMMON and "ate_rmse" in cx and "ate_rmse" in ca:
        acc = _better(cx["ate_rmse"], ca["ate_rmse"], BENEFIT_ABS_ATE_M) or _better(cx["rot_median_deg"], ca["rot_median_deg"], BENEFIT_ABS_ROT_DEG)
    return {"coverage_rescue": rescue, "accuracy_improvement": acc, "improves": bool(rescue or acc)}


def masking_benefit(cov, common, gates):
    """cov: {A,B,C: coverage}; common: {A,B,C: evaluate_on_ids on the A∩B∩C frames}; gates: {A,B,C: config_gate}."""
    iB = improvement(cov["B"], cov["A"], common["B"], common["A"])
    iC = improvement(cov["C"], cov["A"], common["C"], common["A"])
    loss = bool(cov["B"] < cov["A"] - COVERAGE_LOSS_MAX)
    b_vs_c = False
    if common["B"].get("n", 0) >= MIN_COMMON and "ate_rmse" in common["B"] and "ate_rmse" in common["C"]:
        b_vs_c = _better(common["B"]["ate_rmse"], common["C"]["ate_rmse"], BENEFIT_ABS_ATE_M) or \
            _better(common["B"]["rot_median_deg"], common["C"]["rot_median_deg"], BENEFIT_ABS_ROT_DEG)
    c_distinct = bool((not iC["improves"]) or b_vs_c or (iB["coverage_rescue"] and not iC["coverage_rescue"]))
    harm = False
    if loss:
        harm = True
    elif common["A"].get("n", 0) >= MIN_COMMON and "ate_rmse" in common["A"] and "ate_rmse" in common["B"]:
        harm = _better(common["A"]["ate_rmse"], common["B"]["ate_rmse"], BENEFIT_ABS_ATE_M) or \
            _better(common["A"]["rot_median_deg"], common["B"]["rot_median_deg"], BENEFIT_ABS_ROT_DEG)
    if iB["improves"] and not loss and c_distinct:
        verdict = "DEMONSTRATED"
    elif iB["improves"] and not c_distinct:
        verdict = "NOT_ATTRIBUTABLE_TO_HUMAN_EXCLUSION"
    elif harm:
        verdict = "HARMFUL"
    elif gates["A"].get("pass"):
        verdict = "UNNECESSARY"
    else:
        verdict = "NOT_DEMONSTRATED"
    return {"B_vs_A": iB, "C_vs_A": iC, "B_better_than_C": b_vs_c, "C_distinct": c_distinct, "B_coverage_loss": loss,
            "B_harms": harm, "verdict": verdict, "n_common": common["A"].get("n", 0)}


def synthetic_disturbance(ev_x, ev_clean, extent, stationary_ok_x):
    """Does condition x show a disturbance relative to the clean static run?"""
    if "full_sim3" not in ev_x:
        return {"disturbed": True, "why": ["no trajectory"]}
    why = []
    ax, ac = ev_x["full_sim3"]["ate"]["rmse"] / extent, ev_clean["full_sim3"]["ate"]["rmse"] / extent
    if ax > SYN_DISTURB_ATE_RATIO * ac and ax - ac > SYN_DISTURB_ATE_ABS_REL:
        why.append(f"ATE_rel {ax:.5f} vs clean {ac:.5f}")
    rx, rc = ev_x["full_sim3"]["rot"]["median_deg"], ev_clean["full_sim3"]["rot"]["median_deg"]
    if rx - rc > SYN_DISTURB_ROT_DEG:
        why.append(f"median rotation {rx:.3f} vs clean {rc:.3f} deg")
    if ev_clean["coverage_estimated"] - ev_x["coverage_estimated"] > SYN_DISTURB_COVERAGE:
        why.append(f"coverage {ev_x['coverage_estimated']:.3f} vs clean {ev_clean['coverage_estimated']:.3f}")
    if not stationary_ok_x:
        why.append("stationary-interval spurious motion above the W140 thresholds")
    return {"disturbed": bool(why), "why": why, "ate_rel": ax, "ate_rel_clean": ac, "rot_median_deg": rx}


def judgment(real_pass, real_partial, blocked=None):
    """real_pass: {seq: bool (A or B passes all config gates)}; real_partial: {seq: bool (some config covers >= floor and beats baseline)}."""
    if blocked:
        return {"judgment": "BLOCKED", "why": blocked}
    if real_pass and all(real_pass.values()):
        return {"judgment": "SUPPORTED", "why": "an RGB-only configuration (A or B) passes coverage, accuracy targets and the stationary "
                                                "baseline on both moving-human sequences (Pilot utility, not Production readiness)"}
    if any(real_pass.values()) or any(real_partial.values()):
        return {"judgment": "PARTIALLY_SUPPORTED", "why": "camera trajectories recovered in some sequence/condition, not all gates on both"}
    return {"judgment": "NOT_YET_SUPPORTED", "why": "no RGB-only configuration establishes useful camera reconstruction on either sequence"}
