"""EVALUATION ONLY: where does FramePose shoulder->wrist 3D direction error enter?

GT 3DPW geometry is evaluation evidence.  Nothing here is imported by FramePose,
AnimationSemantics, FK, IK or reliability code.  All vectors are in the
canonical camera frame (+X right, +Y forward/depth, +Z up), so the image-plane
component of a 3D vector is XZ and the depth component is Y.  2D observations
are compared in source-image pixels (y down); angles are convention-invariant
as long as both sides share the convention.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from pose.pose_lifter import H36M_NAMES

ARM = ("left_shoulder", "left_elbow", "left_wrist")
JOINT_INDEX = {name: index for index, name in enumerate(H36M_NAMES)}

# SMPL-24 joints for the 12 detector keypoints the 3DPW adapter maps directly.
_SMPL_DIRECT = {
    "left_shoulder": 16, "right_shoulder": 17, "left_elbow": 18, "right_elbow": 19,
    "left_wrist": 20, "right_wrist": 21, "left_hip": 1, "right_hip": 2,
    "left_knee": 4, "right_knee": 5, "left_ankle": 7, "right_ankle": 8,
}
# The detector "head" is the OpenPose nose, which SMPL lacks; SMPL head (15) is
# the closest GT point and is recorded as a definitional substitution.
_SMPL_HEAD = 15


def _unit(v: np.ndarray) -> np.ndarray | None:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 and math.isfinite(n) else None


def angle_degrees(a: np.ndarray, b: np.ndarray) -> float | None:
    ua, ub = _unit(np.asarray(a, float)), _unit(np.asarray(b, float))
    if ua is None or ub is None:
        return None
    cross = np.cross(ua, ub) if ua.size == 3 else ua[0] * ub[1] - ua[1] * ub[0]
    return math.degrees(math.atan2(float(np.linalg.norm(cross)), float(ua @ ub)))


# ------------------------------------------------------------------ 2D


def oracle_input_2d(smpl_pixels: np.ndarray, image_size: tuple[int, int],
                    detector_input: np.ndarray, *, joints: tuple[str, ...] | None = None) -> np.ndarray:
    """GT-projected `(17, 3)` input in the bank convention, detector confidence kept.

    Mirrors ``three_dpw_adapter._canonical_2d`` + ``temporal_lifter.build_dataset``:
    direct joints, head, neck = shoulder midpoint, pelvis = hip midpoint,
    spine = neck/pelvis midpoint, thorax aliased to neck, then x/width, y/height.
    ``joints`` restricts substitution (e.g. the left arm only); every other row
    stays the detector observation.  Validity is never changed here.
    """
    width, height = float(image_size[0]), float(image_size[1])
    px = {name: np.asarray(smpl_pixels[index], float) for name, index in _SMPL_DIRECT.items()}
    px["head"] = np.asarray(smpl_pixels[_SMPL_HEAD], float)
    px["neck"] = (px["left_shoulder"] + px["right_shoulder"]) / 2
    px["pelvis"] = (px["left_hip"] + px["right_hip"]) / 2
    px["spine"] = (px["neck"] + px["pelvis"]) / 2
    px["thorax"] = px["neck"]
    out = np.array(detector_input, dtype=np.float64, copy=True)
    for name in (joints or H36M_NAMES):
        i = JOINT_INDEX[name]
        out[i, 0], out[i, 1] = px[name][0] / width, px[name][1] / height
    return out


def detector_vs_oracle_2d(detector_pixels: np.ndarray, oracle_pixels: np.ndarray) -> dict[str, Any]:
    """Arm joint pixel errors and segment / chain 2D direction errors.

    Inputs are `(3, 2)` shoulder, elbow, wrist pixels.
    """
    d, o = np.asarray(detector_pixels, float), np.asarray(oracle_pixels, float)
    out = {f"{name}_2d_error_px": float(np.linalg.norm(d[i] - o[i])) for i, name in enumerate(ARM)}
    for label, (a, b) in {"shoulder_wrist": (0, 2), "shoulder_elbow": (0, 1), "elbow_wrist": (1, 2)}.items():
        out[f"{label}_2d_direction_error_degrees"] = angle_degrees(d[b] - d[a], o[b] - o[a])
    chain = float(np.linalg.norm(o[1] - o[0]) + np.linalg.norm(o[2] - o[1]))
    out["oracle_chain_length_px"] = chain
    out["shoulder_wrist_2d_endpoint_error_over_chain"] = (
        float(np.linalg.norm((d[2] - d[0]) - (o[2] - o[0])) / chain) if chain > 1e-9 else None)
    return out


# ------------------------------------------------------------------ 3D


def image_plane_depth(h0: np.ndarray, gt: np.ndarray) -> dict[str, Any]:
    """Separate XZ image-plane angle, depth (Y) components and full 3D angle."""
    h0, gt = np.asarray(h0, float), np.asarray(gt, float)
    uh, ug = _unit(h0), _unit(gt)
    if uh is None or ug is None:
        raise ValueError("degenerate 3D vector")

    def elevation(u):
        return math.degrees(math.atan2(u[1], math.hypot(u[0], u[2])))
    return {
        "full_3d_angle_degrees": angle_degrees(h0, gt),
        "xz_image_plane_angle_degrees": angle_degrees(h0[[0, 2]], gt[[0, 2]]),
        "h0_forward_fraction": float(uh[1]), "gt_forward_fraction": float(ug[1]),
        "forward_fraction_error": float(uh[1] - ug[1]),
        "depth_elevation_error_degrees": elevation(uh) - elevation(ug),
        "depth_sign_disagrees": bool(np.sign(uh[1]) != np.sign(ug[1])
                                     and abs(ug[1]) > math.sin(math.radians(10))),
    }


def component_hybrids(h0: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    """Raw-displacement XZ / Y substitution, normalized only afterwards, scored vs GT."""
    h0, gt = np.asarray(h0, float), np.asarray(gt, float)
    variants = {
        "h0_full": h0,
        "h0_image_plane_gt_depth": np.array([h0[0], gt[1], h0[2]]),
        "gt_image_plane_h0_depth": np.array([gt[0], h0[1], gt[2]]),
        "gt_full": gt,
    }
    return {name: angle_degrees(v, gt) for name, v in variants.items()}


def segment_hybrids(h0_upper: np.ndarray, h0_lower: np.ndarray,
                    gt_upper: np.ndarray, gt_lower: np.ndarray) -> dict[str, float]:
    """Shoulder->wrist from segment DIRECTIONS weighted by fixed GT lengths, vs GT."""
    lu, ll = float(np.linalg.norm(gt_upper)), float(np.linalg.norm(gt_lower))
    dirs = {"h0_u": _unit(np.asarray(h0_upper, float)), "h0_l": _unit(np.asarray(h0_lower, float)),
            "gt_u": _unit(np.asarray(gt_upper, float)), "gt_l": _unit(np.asarray(gt_lower, float))}
    if any(v is None for v in dirs.values()):
        raise ValueError("degenerate segment")
    target = np.asarray(gt_upper, float) + np.asarray(gt_lower, float)
    combos = {"h0_upper_h0_lower": ("h0_u", "h0_l"), "gt_upper_h0_lower": ("gt_u", "h0_l"),
              "h0_upper_gt_lower": ("h0_u", "gt_l"), "gt_upper_gt_lower": ("gt_u", "gt_l")}
    out = {name: angle_degrees(lu * dirs[u] + ll * dirs[l], target) for name, (u, l) in combos.items()}
    out["upper_direction_error_degrees"] = angle_degrees(h0_upper, gt_upper)
    out["lower_direction_error_degrees"] = angle_degrees(h0_lower, gt_lower)
    return out


def chain_direction_row(h0_points: np.ndarray, gt_points: np.ndarray) -> dict[str, Any]:
    """All 3D attributions for one `(3, 3)` shoulder/elbow/wrist pair (same frame)."""
    h, g = np.asarray(h0_points, float), np.asarray(gt_points, float)
    if not (np.isfinite(h).all() and np.isfinite(g).all()):
        raise ValueError("non-finite chain")
    row = image_plane_depth(h[2] - h[0], g[2] - g[0])
    row.update({f"component_{k}": v for k, v in component_hybrids(h[2] - h[0], g[2] - g[0]).items()})
    row.update({f"segment_{k}": v for k, v in segment_hybrids(h[1] - h[0], h[2] - h[1],
                                                               g[1] - g[0], g[2] - g[1]).items()})
    row["upper_xz_angle_degrees"] = angle_degrees((h[1] - h[0])[[0, 2]], (g[1] - g[0])[[0, 2]])
    row["lower_xz_angle_degrees"] = angle_degrees((h[2] - h[1])[[0, 2]], (g[2] - g[1])[[0, 2]])
    return row


def joint_coordinate_errors(h0_joints: np.ndarray, gt_joints: np.ndarray,
                            h0_pelvis: np.ndarray, gt_pelvis: np.ndarray) -> dict[str, float]:
    """Pelvis-relative signed X / Y(depth) / Z error per arm joint, in metres."""
    out = {}
    for i, name in enumerate(ARM):
        delta = (np.asarray(h0_joints[i]) - h0_pelvis) - (np.asarray(gt_joints[i]) - gt_pelvis)
        for axis, value in zip("xyz", delta):
            out[f"{name}_{axis}_error_m"] = float(value)
        out[f"{name}_error_m"] = float(np.linalg.norm(delta))
    return out


def project_animcv_to_pixels(points_animcv: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    """Absolute AnimCV camera points -> OpenCV camera (x, -Z, Y) -> pixels (y down)."""
    p = np.asarray(points_animcv, float)
    x, y, z = p[..., 0], -p[..., 2], p[..., 1]
    if np.any(z <= 0):
        raise ValueError("point behind the camera")
    return np.stack([intrinsics[0, 0] * x / z + intrinsics[0, 2],
                     intrinsics[1, 1] * y / z + intrinsics[1, 2]], axis=-1)


def xz_vs_image_direction(vector_3d: np.ndarray, direction_2d_pixels: np.ndarray) -> float | None:
    """Angle between a 3D vector's XZ component and a 2D pixel direction (u right, v down -> X, -Z)."""
    v = np.asarray(vector_3d, float)
    d = np.asarray(direction_2d_pixels, float)
    return angle_degrees(v[[0, 2]], np.array([d[0], -d[1]]))
