"""DIAGNOSTIC (Worklog 83): MPI-INF-3DHP official test set as an independent evaluation source.

Source contract (verified on the downloaded data, see docs/83):
  * ``mpi_inf_3dhp_test_set.zip`` (vcai.mpi-inf.mpg.de, Last-Modified 2021-07-11);
    sequences TS1..TS6, one subject each, one camera each;
  * per sequence ``annot_data.mat``: ``annot2`` (pixels), ``annot3`` (camera
    coordinates, millimetres), ``valid_frame``; joints are the 17-joint test
    order below (subject's anatomical left/right);
  * images ``imageSequence/img_%06d.jpg`` numbered from 1, so annotation row
    ``k`` is image ``k + 1``;
  * camera axes are OpenCV (+x right, +y down, +z forward); AnimCV canonical
    (X right, Y depth, Z up) = (x, z, -y), metres, pelvis subtracted -- the
    same conversion as the 3DPW/BEDLAM adapters;
  * frame rate: TS1-TS4 studio 25 fps, TS5-TS6 outdoor GoPro 50 fps (community convention; the official
    README does not state it -- the GT walking speed of TS3 supports 25 fps for the studio sequences);
  * the official evaluation frames (``valid_frame``) are every 10th frame in TS1-TS4 and almost every frame
    in TS5-TS6; non-valid frames are never used, neither as rows nor as temporal partners.
Only the subject's left shoulder/elbow/wrist and the pelvis enter any metric;
the other target joints are carried for FrameBank shape only.
GT never enters a model input: 2D GT is used only to decide which detected
person is the subject (identity association, reported), GT 3D only to build
the evaluation reference.
"""

from __future__ import annotations

import math

import numpy as np

from pose.pose_lifter import H36M_NAMES

TEST17_NAMES = ("head_top", "neck", "right_shoulder", "right_elbow", "right_wrist", "left_shoulder", "left_elbow",
                "left_wrist", "right_hip", "right_knee", "right_ankle", "left_hip", "left_knee", "left_ankle",
                "pelvis", "spine", "head")
# FrameBank joint -> 3DHP test joint.  Arm/hip/leg joints are exact; thorax/neck/head are the nearest
# 3DHP landmarks (neck, neck, head) and never enter a metric.
CANONICAL_FROM_TEST17 = {"pelvis": 14, "left_hip": 11, "left_knee": 12, "left_ankle": 13, "right_hip": 8,
                         "right_knee": 9, "right_ankle": 10, "spine": 15, "thorax": 1, "neck": 1, "head": 16,
                         "left_shoulder": 5, "left_elbow": 6, "left_wrist": 7, "right_shoulder": 2,
                         "right_elbow": 3, "right_wrist": 4}
CANONICAL_TEST17 = np.array([CANONICAL_FROM_TEST17[n] for n in H36M_NAMES])
# SMPL-X body index (the Worklog 73 association table's GT indexing) -> 3DHP test joint.
SMPLX_FROM_TEST17 = {16: 5, 17: 2, 18: 6, 19: 3, 20: 7, 21: 4, 1: 11, 2: 8, 4: 12, 5: 9, 7: 13, 8: 10}
SEQUENCES = ("TS1", "TS2", "TS3", "TS4", "TS5", "TS6")
FPS = {"TS1": 25, "TS2": 25, "TS3": 25, "TS4": 25, "TS5": 50, "TS6": 50}
# Every official valid frame is an evaluation row.  Primary temporal pairs: two valid rows of one sequence
# exactly 0.4 s apart (10 frames at 25 fps = the TS1-TS4 valid-frame spacing; 20 frames at 50 fps).
# Secondary (descriptive, outdoor only): TS5-TS6 pairs 8 frames = 0.16 s apart, the closest to 3DPW's 1/6 s.
PAIR_INTERVAL_SECONDS = 0.4
SHORT_PAIR_FRAMES = {"TS5": 8, "TS6": 8}
BEND_REGIMES = {"near_straight [0,15)": (0, 15), "moderate [15,45)": (15, 45), "strong [45,90)": (45, 90),
                "near_folded [90,180]": (90, 181)}
STABLE = math.sin(math.radians(10))
F_BINS = {"near_plane [0,sin10)": (0, STABLE), "intermediate": (STABLE, 0.5), "out_of_plane [0.5,1]": (0.5, 1.01)}
DEGENERATE_BEND_SINE = math.sin(math.radians(1.0))  # Worklog 61 observed-bend-side contract


def stride_for(sequence: str) -> int:
    """Primary pair spacing in frames (0.4 s)."""
    return int(round(FPS[sequence] * PAIR_INTERVAL_SECONDS))


def select_frames(valid_frame: np.ndarray) -> np.ndarray:
    """Every annotation row the dataset marks ``valid_frame`` (label-blind: never reads a pose)."""
    return np.flatnonzero(np.asarray(valid_frame).reshape(-1).astype(bool))


def image_name(row: int) -> str:
    return f"img_{row + 1:06d}.jpg"


def camera_mm_to_animcv(points_mm: np.ndarray) -> np.ndarray:
    p = np.asarray(points_mm, dtype=np.float64) / 1000.0
    return np.stack([p[..., 0], p[..., 2], -p[..., 1]], axis=-1)


def canonical_target(annot3_17: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(17,3) pelvis-relative AnimCV target (metres) and GT validity (finite, in front of the camera)."""
    picked = np.asarray(annot3_17, dtype=np.float64)[CANONICAL_TEST17]
    valid = np.isfinite(picked).all(1) & (picked[:, 2] > 0)
    animcv = camera_mm_to_animcv(picked)
    return (animcv - animcv[0]).astype(np.float32), valid


def association_uv(annot2_17: np.ndarray) -> np.ndarray:
    """(22, 2) GT pixels in SMPL-X body indexing for the unchanged Worklog 73 ``associate`` (identity only)."""
    out = np.full((22, 2), np.nan)
    for smplx, j in SMPLX_FROM_TEST17.items():
        out[smplx] = np.asarray(annot2_17, dtype=np.float64)[j]
    return out


def parse_calibration(text: str, image_size: tuple[int, int]) -> dict:
    """Official ``tc camera calibration v0.3`` block -> pinhole K, OpenCV distortion (verification only).

    ``pixelAspect`` is documented as y / x, so fy = fx * pixelAspect.  Intrinsics never enter a model input
    or a metric; they only verify that ``annot3`` (camera mm, +y down) reprojects onto ``annot2``.
    """
    import re

    def nums(key):
        m = re.search(rf"^\s*{key}\s+([-0-9.eE\s]+?)(#|$)", text, re.MULTILINE)
        return [float(x) for x in m.group(1).split()]
    w, h = image_size
    sw, sh = nums("sensorSize")
    focal, aspect = nums("focalLength")[0], nums("pixelAspect")[0]
    ox, oy = nums("centerOffset")
    dist = nums("distortion")
    fx = focal / sw * w
    k = np.array([[fx, 0, w / 2 + ox / sw * w], [0, fx * aspect, h / 2 + oy / sh * h], [0, 0, 1]])
    return {"K": k, "dist": np.array(dist + [0.0] * (5 - len(dist)))}


def project_opencv(points_mm: np.ndarray, k: np.ndarray, dist: np.ndarray) -> np.ndarray:
    """OpenCV pinhole + (k1, k2, p1, p2, k3) distortion projection of camera-space points."""
    p = np.asarray(points_mm, dtype=np.float64)
    x, y = p[..., 0] / p[..., 2], p[..., 1] / p[..., 2]
    k1, k2, p1, p2, k3 = dist
    r2 = x * x + y * y
    radial = 1 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3
    xd = x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
    yd = y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
    return np.stack([k[0, 0] * xd + k[0, 2], k[1, 1] * yd + k[1, 2]], axis=-1)


def fit_pinhole(points_mm: np.ndarray, uv: np.ndarray) -> dict:
    """Least-squares (fx, fy, cx, cy) mapping camera points onto pixels, and its residual (verification only)."""
    p, u = np.asarray(points_mm, np.float64).reshape(-1, 3), np.asarray(uv, np.float64).reshape(-1, 2)
    x, y = p[:, 0] / p[:, 2], p[:, 1] / p[:, 2]
    fx, cx = np.linalg.lstsq(np.stack([x, np.ones_like(x)], 1), u[:, 0], rcond=None)[0]
    fy, cy = np.linalg.lstsq(np.stack([y, np.ones_like(y)], 1), u[:, 1], rcond=None)[0]
    r = np.hypot(fx * x + cx - u[:, 0], fy * y + cy - u[:, 1])
    return {"fx": float(fx), "fy": float(fy), "cx": float(cx), "cy": float(cy),
            "residual_median_px": float(np.median(r)), "residual_p95_px": float(np.percentile(r, 95))}


# --- canonical connected-arm metrics (shoulder at the origin, unit total arm length) ---

def _unit(x):
    x = np.asarray(x, dtype=np.float64)
    n = np.linalg.norm(x)
    return x / n if n > 1e-12 else None


def _angle(a, b):
    a, b = _unit(a), _unit(b)
    if a is None or b is None:
        return None
    return math.degrees(math.atan2(float(np.linalg.norm(np.cross(a, b))), float(np.dot(a, b))))


def bend_side(elbow, wrist):
    """Worklog 61 observed bend side: unit offset of the elbow from the shoulder->wrist line, or None."""
    e, w = np.asarray(elbow, np.float64), np.asarray(wrist, np.float64)
    d = _unit(w)
    u, v = _unit(e), _unit(w - e)
    if d is None or u is None or v is None or np.linalg.norm(np.cross(u, v)) < DEGENERATE_BEND_SINE:
        return None
    return _unit(e - d * float(np.dot(e, d)))


def bend_angle(elbow, wrist):
    e, w = np.asarray(elbow, np.float64), np.asarray(wrist, np.float64)
    return _angle(e, w - e)


def side_error(elbow, wrist, gt_elbow, gt_wrist):
    """Angle between bend sides, the GT side re-perpendicularized to the candidate chain direction."""
    side, gt = bend_side(elbow, wrist), bend_side(gt_elbow, gt_wrist)
    d = _unit(wrist)
    if side is None or gt is None or d is None:
        return None
    ref = _unit(gt - d * float(np.dot(gt, d)))
    return None if ref is None else _angle(side, ref)


def static_metrics(elbow, wrist, gt_elbow, gt_wrist) -> dict:
    """Per-row canonical errors; positions in percent of the GT total arm length."""
    e, w, ge, gw = (np.asarray(x, np.float64) for x in (elbow, wrist, gt_elbow, gt_wrist))
    ba, bg = bend_angle(e, w), bend_angle(ge, gw)
    return {"elbow_error": 100 * float(np.linalg.norm(e - ge)), "wrist_error": 100 * float(np.linalg.norm(w - gw)),
            "upper_direction_error_degrees": _angle(e, ge), "forearm_direction_error_degrees": _angle(w - e, gw - ge),
            "chain_direction_error_degrees": _angle(w, gw),
            "reach_error": 100 * abs(float(np.linalg.norm(w) - np.linalg.norm(gw))),
            "side_error_degrees": side_error(e, w, ge, gw),
            "abs_bend_angle_error_degrees": abs(ba - bg) if ba is not None and bg is not None else None}


def _chord_degrees(a0, a1, r0, r1):
    """Worklog 77 direction motion residual: |delta_candidate - delta_reference| (unit vectors), degrees of chord."""
    vs = [_unit(x) for x in (a0, a1, r0, r1)]
    if any(v is None for v in vs):
        return None
    return math.degrees(float(np.linalg.norm((vs[1] - vs[0]) - (vs[3] - vs[2]))))


def motion_metrics(e0, w0, e1, w1, ge0, gw0, ge1, gw1) -> dict:
    """Per-pair motion residual against the GT motion of the SAME pair, plus motion amplitude and variation."""
    e0, w0, e1, w1, ge0, gw0, ge1, gw1 = (np.asarray(x, np.float64) for x in (e0, w0, e1, w1, ge0, gw0, ge1, gw1))
    s0, s1 = bend_side(e0, w0), bend_side(e1, w1)
    return {"upper_direction_residual_degrees": _chord_degrees(e0, e1, ge0, ge1),
            "forearm_direction_residual_degrees": _chord_degrees(w0 - e0, w1 - e1, gw0 - ge0, gw1 - ge1),
            "chain_direction_residual_degrees": _chord_degrees(w0, w1, gw0, gw1),
            "elbow_motion_residual": 100 * float(np.linalg.norm((e1 - e0) - (ge1 - ge0))),
            "wrist_motion_residual": 100 * float(np.linalg.norm((w1 - w0) - (gw1 - gw0))),
            "upper_direction_change_degrees": _angle(e0, e1),
            "forearm_direction_change_degrees": _angle(w0 - e0, w1 - e1),
            "chain_direction_change_degrees": _angle(w0, w1),
            "elbow_displacement": 100 * float(np.linalg.norm(e1 - e0)),
            "wrist_displacement": 100 * float(np.linalg.norm(w1 - w0)),
            "reach_change": 100 * abs(float(np.linalg.norm(w1) - np.linalg.norm(w0))),
            "bend_side_change_degrees": _angle(s0, s1) if s0 is not None and s1 is not None else None}


def regime(gt_elbow, gt_wrist) -> dict:
    """Evaluation-only GT bins: bend angle and |chain depth component| (out-of-plane f)."""
    b = bend_angle(gt_elbow, gt_wrist)
    d = _unit(gt_wrist)
    f = float(d[1]) if d is not None else None
    bend_bin = next((k for k, (lo, hi) in BEND_REGIMES.items() if b is not None and lo <= b < hi), None)
    f_bin = next((k for k, (lo, hi) in F_BINS.items() if f is not None and lo <= abs(f) < hi), None)
    return {"gt_bend_degrees": b, "f_gt": f, "bend_regime": bend_bin, "f_regime": f_bin}
