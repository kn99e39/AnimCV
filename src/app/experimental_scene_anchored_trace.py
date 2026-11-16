"""W144 scratch instrumentation. No estimator, smoother or production integration.

All rotations are active. Matrices map local to parent. Positions carry explicit
units at the call site; no depth/metric world translation is inferred from RGB.
"""
from __future__ import annotations

import numpy as np


def action_frame(source: int, first: int = 79, last: int = 318) -> int:
    if not isinstance(source, (int, np.integer)) or not first <= source <= last:
        raise ValueError("source frame outside frozen segment")
    return int(source - first + 1)


def summary(values):
    a = np.asarray(values, float).ravel()
    a = a[np.isfinite(a)]
    return {"n": int(a.size), **({k: float(v) for k, v in zip(
        ("min", "median", "p90", "p95", "p99", "max"),
        np.percentile(a, [0, 50, 90, 95, 99, 100]))} if a.size else {})}


def motion(points, valid=None):
    p = np.asarray(points, float)
    v = np.isfinite(p).all(-1)
    if valid is not None:
        v &= np.asarray(valid, bool)
    d = np.linalg.norm(np.diff(p, axis=0), axis=-1)
    d[~(v[1:] & v[:-1])] = np.nan
    acc = np.linalg.norm(np.diff(p, n=2, axis=0), axis=-1)
    acc[~(v[2:] & v[1:-1] & v[:-2])] = np.nan
    return {"step": summary(d), "second_difference": summary(acc),
            "peak_to_action": int(np.nanargmax(d) + 2) if np.isfinite(d).any() else None,
            "valid_frames": int(v.sum())}, d


def rotation_angle(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    r = np.swapaxes(a, -1, -2) @ b
    # atan2 retains accuracy near identity and pi.
    sin = np.linalg.norm(np.stack((r[..., 2, 1]-r[..., 1, 2],
                                   r[..., 0, 2]-r[..., 2, 0],
                                   r[..., 1, 0]-r[..., 0, 1]), -1), axis=-1) / 2
    cos = np.clip((np.trace(r, axis1=-2, axis2=-1)-1)/2, -1, 1)
    return np.degrees(np.arctan2(sin, cos))


def fk(rest, basis, object_matrix):
    """Standard rest-relative FK, with Blender inherit defaults verified separately.

    P_b = P_parent @ inverse(L_parent) @ L_b @ basis_b;
    top-level P_b = L_b @ basis_b; World_b = Object @ P_b.
    """
    arm, world = {}, {}
    def visit(name):
        if name in arm:
            return arm[name]
        rec = rest[name]
        local = np.asarray(rec["matrix_local"], float)
        parent = rec["parent"]
        base = np.asarray(basis.get(name, np.eye(4)), float)
        arm[name] = (visit(parent) @ np.linalg.inv(rest[parent]["matrix_local"]) @ local @ base
                     if parent else local @ base)
        world[name] = np.asarray(object_matrix) @ arm[name]
        return arm[name]
    for name in rest:
        visit(name)
    return arm, world


def replace_factor(current, previous, factor):
    """One endpoint counterfactual; arrays copied, never modifies source state."""
    import copy
    result = copy.deepcopy(current)
    if factor in ("object_rotation", "object_translation"):
        sl = (slice(0, 3), slice(0, 3)) if factor == "object_rotation" else (slice(0, 3), 3)
        result["object"][sl] = previous["object"][sl]
    elif factor.startswith("bone:"):
        name = factor[5:]
        result["basis"][name] = previous["basis"][name].copy()
    else:
        raise ValueError("unknown single factor")
    return result


def camera_to_world(points_camera, R_cw, centre_world):
    if centre_world is None:
        raise ValueError("metric translation unavailable; orientation is not a camera centre")
    return np.asarray(points_camera) @ np.asarray(R_cw).T + np.asarray(centre_world)


def world_to_camera(points_world, R_cw, centre_world):
    if centre_world is None:
        raise ValueError("camera centre unavailable")
    return (np.asarray(points_world)-np.asarray(centre_world)) @ np.asarray(R_cw)


def require_intermediate(data, key):
    if key not in data:
        raise ValueError(f"intermediate unavailable: {key}; do not fabricate it")
    return data[key]
