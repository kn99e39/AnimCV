#!/usr/bin/env python3
"""Geometry-only feasibility audit for the middle-joint hinge write contract.

This is a diagnostic module, not a branch-constraint policy.  It formalizes
the set

    |M' - P| = |M - P|,  |D - M'| = |D - M|

and intersects that set with a calibrated perspective camera ray.  It also
contains deterministic synthetic contracts and a camera-aware lower-bound
solver for a *diagnostic* comparison.  Nothing in this file is imported by
``framepose.branch_constraints`` and it does not change a production default.

Degeneracy decisions use exact algebraic comparisons.  The only approximate
membership check is scaled machine precision, used to recognize the result of
the same floating-point arithmetic as an on-circle point; it is not a pose or
performance threshold.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from framepose.signs import MIN_BEND_OFFSET_M, UNIT_FORWARD_EPSILON


MACHINE_EPSILON = np.finfo(np.float64).eps


def _vec(value: Any) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (3,):
        raise ValueError(f"expected a 3-vector, got {result.shape}")
    return result


def _unit(value: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(value))
    if norm == 0.0:
        raise ValueError("a direction must be non-zero")
    return value / norm


def _machine_zero(value: float, scale: float = 1.0) -> bool:
    """Recognize arithmetic zero at machine precision, never a fitted epsilon."""
    # The fixed factor is an operation-count margin for the dot products and
    # subtraction used to form the residual; it is unrelated to pose scale,
    # camera geometry, or measured performance.
    return abs(float(value)) <= 16.0 * MACHINE_EPSILON * max(1.0, abs(float(scale)))


def _json_vector(value: np.ndarray | None) -> list[float] | None:
    return None if value is None else [float(item) for item in value]


def bone_preserving_locus(proximal: Any, distal: Any, proximal_length: float,
                          distal_length: float) -> dict[str, Any]:
    """Return the exact intersection of the two bone-length spheres.

    For a non-zero proximal/distal axis, ``status == "circle"`` means:

        C = P + lambda (D-P)
        Pi = {X : (X-C) dot (D-P) = 0}
        radius = sqrt(r_P^2 - lambda^2 |D-P|^2)

    Tangency is returned as ``zero_radius`` rather than silently promoted to a
    circle.  A zero-length axis is handled separately because concentric
    spheres intersect as a sphere, a point, or not at all.
    """
    p, d = _vec(proximal), _vec(distal)
    rp, rd = float(proximal_length), float(distal_length)
    if not np.isfinite([rp, rd]).all() or rp < 0.0 or rd < 0.0:
        raise ValueError("sphere radii must be finite and non-negative")

    axis = d - p
    axis_squared = float(axis @ axis)
    axis_length = float(np.sqrt(axis_squared))
    result: dict[str, Any] = {
        "proximal": _json_vector(p),
        "distal": _json_vector(d),
        "proximal_radius": rp,
        "distal_radius": rd,
        "axis_length": axis_length,
        "axis": _json_vector(axis),
        "plane_normal": None,
        "circle_center": None,
        "circle_radius": None,
        "circle_radius_squared": None,
    }

    if axis_squared == 0.0:
        if rp != rd:
            result["status"] = "inconsistent_concentric_spheres"
        elif rp == 0.0:
            result.update({"status": "singleton", "circle_center": _json_vector(p),
                           "circle_radius": 0.0, "circle_radius_squared": 0.0})
        else:
            result.update({"status": "sphere", "circle_center": _json_vector(p),
                           "circle_radius": rp, "circle_radius_squared": rp * rp})
        return result

    if axis_length > rp + rd or axis_length < abs(rp - rd):
        result["status"] = "inconsistent_spheres"
        return result

    lam = (rp * rp - rd * rd + axis_squared) / (2.0 * axis_squared)
    center = p + lam * axis
    radius_squared = rp * rp - lam * lam * axis_squared
    result.update({
        "lambda": float(lam),
        "circle_center": _json_vector(center),
        "plane_normal": _json_vector(axis),
        "circle_radius_squared": float(radius_squared),
    })
    if radius_squared < 0.0:
        result["status"] = "inconsistent_spheres"
    elif radius_squared == 0.0:
        result.update({"status": "zero_radius", "circle_radius": 0.0})
    else:
        result.update({"status": "circle", "circle_radius": float(np.sqrt(radius_squared))})
    return result


def _locus_arrays(locus: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, float]:
    if locus.get("status") not in {"circle", "zero_radius"}:
        raise ValueError(f"a circle locus is required, got {locus.get('status')!r}")
    center = _vec(locus["circle_center"])
    normal = _vec(locus["plane_normal"])
    radius = float(locus["circle_radius"])
    return center, normal, radius


def ray_locus_intersections(locus: dict[str, Any], origin: Any, direction: Any) -> dict[str, Any]:
    """Intersect a forward camera ray with a circle, point, or concentric sphere."""
    o = _vec(origin)
    v = _unit(_vec(direction))
    status = locus.get("status")
    output: dict[str, Any] = {
        "relation": None,
        "intersection_count": 0,
        "parameters": [],
        "points": [],
        "plane_denominator": None,
        "plane_offset": None,
        "circle_residuals": [],
    }

    if status == "inconsistent_spheres" or status == "inconsistent_concentric_spheres":
        output["relation"] = "empty_locus"
        return output

    if status in {"singleton", "zero_radius"}:
        point = _vec(locus["circle_center"])
        t = float((point - o) @ v)
        residual = float(np.linalg.norm(point - (o + t * v)))
        output["relation"] = "point"
        output["point_line_residual"] = residual
        if t >= 0.0 and _machine_zero(residual, np.linalg.norm(point) + 1.0):
            output.update({"intersection_count": 1, "parameters": [t],
                           "points": [_json_vector(point)]})
        return output

    if status == "sphere":
        center = _vec(locus["circle_center"])
        radius = float(locus["circle_radius"])
        b = 2.0 * float((o - center) @ v)
        c = float((o - center) @ (o - center) - radius * radius)
        discriminant = b * b - 4.0 * c
        output["relation"] = "concentric_axis_sphere"
        output["discriminant"] = discriminant
        roots = [] if discriminant < 0.0 else (
            [-b / 2.0] if discriminant == 0.0 else
            [(-b - np.sqrt(discriminant)) / 2.0, (-b + np.sqrt(discriminant)) / 2.0])
        roots = [float(t) for t in roots if t >= 0.0]
        output["parameters"] = roots
        output["points"] = [_json_vector(o + t * v) for t in roots]
        output["intersection_count"] = len(roots)
        return output

    if status != "circle":
        output["relation"] = "unsupported_locus"
        return output

    center, normal, radius = _locus_arrays(locus)
    denominator = float(normal @ v)
    offset = float(normal @ (o - center))
    output["plane_denominator"] = denominator
    output["plane_offset"] = offset

    if denominator != 0.0:
        t = -offset / denominator
        output["relation"] = "ray_meets_circle_plane_once"
        if t < 0.0:
            output["relation"] = "plane_intersection_behind_ray"
            return output
        point = o + t * v
        residual = float((point - center) @ (point - center) - radius * radius)
        output["circle_residuals"] = [residual]
        if _machine_zero(residual, np.linalg.norm(point - center) ** 2 + radius * radius + 1.0):
            output.update({"intersection_count": 1, "parameters": [float(t)],
                           "points": [_json_vector(point)]})
        else:
            output["relation"] = "ray_plane_candidate_misses_circle"
        return output

    if offset != 0.0:
        output["relation"] = "ray_parallel_to_circle_plane_disjoint"
        return output

    # The line is contained in the circle plane.  Intersect it with the
    # circle; only non-negative parameters belong to the camera ray.
    output["relation"] = "ray_contained_in_circle_plane"
    w = o - center
    b = 2.0 * float(w @ v)
    c = float(w @ w - radius * radius)
    discriminant = b * b - 4.0 * c
    output["discriminant"] = discriminant
    if discriminant < 0.0:
        return output
    if discriminant == 0.0:
        roots = [-b / 2.0]
    else:
        roots = [(-b - np.sqrt(discriminant)) / 2.0,
                 (-b + np.sqrt(discriminant)) / 2.0]
    roots = sorted(float(t) for t in roots if t >= 0.0)
    output["parameters"] = roots
    output["points"] = [_json_vector(o + t * v) for t in roots]
    output["intersection_count"] = len(roots)
    return output


def circle_basis(locus: dict[str, Any]) -> dict[str, Any] | None:
    """Return the Sign Contract's depth/screen basis in the circle plane."""
    if locus.get("status") != "circle":
        return None
    center, normal, radius = _locus_arrays(locus)
    axis_hat = normal / np.linalg.norm(normal)
    depth = np.array([0.0, 1.0, 0.0])
    u = depth - axis_hat * float(axis_hat @ depth)
    u_norm = float(np.linalg.norm(u))
    if u_norm == 0.0:
        return {"axis_hat": _json_vector(axis_hat), "depth_basis_defined": False,
                "in_plane_fraction": 0.0, "circle_radius": radius}
    u_hat = u / u_norm
    v_hat = np.cross(axis_hat, depth) / u_norm
    return {"axis_hat": _json_vector(axis_hat), "u_hat": _json_vector(u_hat),
            "v_hat": _json_vector(v_hat), "depth_basis_defined": True,
            "in_plane_fraction": float(u_norm * u_norm), "circle_radius": radius,
            "center": _json_vector(center)}


def branch_coordinates(point: Any, locus: dict[str, Any]) -> dict[str, Any]:
    """Report continuous bend coordinates and the frozen SignState readability."""
    if locus.get("status") != "circle":
        return {"readable": False, "sign": 0, "reason": "locus is not a non-zero circle"}
    basis = circle_basis(locus)
    if not basis or not basis["depth_basis_defined"]:
        return {"readable": False, "sign": 0, "reason": "axis is exactly camera-depth aligned"}
    q = _vec(point)
    center = _vec(locus["circle_center"])
    u_hat, v_hat = _vec(basis["u_hat"]), _vec(basis["v_hat"])
    radius = float(locus["circle_radius"])
    offset = q - center
    c_depth = float(offset @ u_hat)
    c_screen = float(offset @ v_hat)
    unit_forward = float((offset / radius)[1])
    sign = 0 if unit_forward == 0.0 else (1 if unit_forward > 0.0 else -1)
    readable = (radius >= MIN_BEND_OFFSET_M and
                np.sqrt(float(basis["in_plane_fraction"])) >= UNIT_FORWARD_EPSILON and
                abs(unit_forward) >= UNIT_FORWARD_EPSILON)
    return {"readable": bool(readable), "sign": int(sign if readable else 0),
            "c_depth": c_depth, "c_screen": c_screen,
            "offset_norm": float(np.linalg.norm(offset)), "unit_forward": unit_forward,
            "in_plane_fraction": float(basis["in_plane_fraction"]), "radius": radius,
            "reason": None if readable else "frozen Sign Contract readability semantics reject it"}


def opposite_branch_available(locus: dict[str, Any], requested: int) -> dict[str, Any]:
    """Whether this exact bone-preserving circle contains the requested branch."""
    if requested not in (-1, 1):
        return {"available": False, "reason": "requested branch is UNKNOWN"}
    basis = circle_basis(locus)
    if basis is None or not basis["depth_basis_defined"]:
        return {"available": False, "reason": "camera-depth component is not observable"}
    radius = float(locus["circle_radius"])
    sqrt_f = np.sqrt(float(basis["in_plane_fraction"]))
    if radius < MIN_BEND_OFFSET_M:
        return {"available": False, "reason": "circle offset is below Sign Contract bend floor"}
    if sqrt_f < UNIT_FORWARD_EPSILON:
        return {"available": False, "reason": "no circle point can satisfy Sign Contract forward floor"}
    return {"available": True, "minimum_abs_cosine": float(UNIT_FORWARD_EPSILON / sqrt_f),
            "requested": int(requested)}


def _poly_add(*terms: np.ndarray) -> np.ndarray:
    size = max(len(term) for term in terms)
    result = np.zeros(size, dtype=np.float64)
    for term in terms:
        result[:len(term)] += term
    return result


def _poly_mul(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    return np.polynomial.polynomial.polymul(first, second)


def _trig_quadratic(constant: float, cosine: float, sine: float) -> np.ndarray:
    """Polynomial numerator for constant + cosine*cos(theta) + sine*sin(theta)."""
    return _poly_add(np.array([constant, 0.0, constant]),
                     cosine * np.array([1.0, 0.0, -1.0]),
                     sine * np.array([0.0, 2.0]))


def minimum_image_displacement_on_branch(point: Any, locus: dict[str, Any], intrinsics: Any,
                                         requested: int, origin: Any = (0.0, 0.0, 0.0)) -> dict[str, Any]:
    """Minimize perspective pixel displacement over the readable branch arc.

    The parameter is the exact circle angle.  The stationary candidates are
    obtained from the polynomial derivative after the tangent-half-angle
    substitution; the two Sign Contract readability-boundary endpoints are
    added explicitly.  No circle sampling, weight, fit, or performance-derived
    tolerance is used.

    This is deliberately a diagnostic lower-bound calculation.  It does not
    select or expose a production write operator.
    """
    if locus.get("status") != "circle":
        return {"resolved": False, "reason": "the feasible set is not a non-zero circle"}
    if requested not in (-1, 1):
        return {"resolved": False, "reason": "requested branch is UNKNOWN"}
    basis = circle_basis(locus)
    availability = opposite_branch_available(locus, requested)
    if not availability["available"]:
        return {"resolved": False, "reason": availability["reason"]}
    K = np.asarray(intrinsics, dtype=np.float64)
    if K.shape != (3, 3):
        raise ValueError(f"intrinsics must be (3, 3), got {K.shape}")
    o = _vec(origin)
    center = _vec(locus["circle_center"])
    u_hat, v_hat = _vec(basis["u_hat"]), _vec(basis["v_hat"])
    radius = float(locus["circle_radius"])
    m = _vec(point)
    mrel = m - o
    if mrel[1] == 0.0:
        return {"resolved": False, "reason": "current point lies on the camera plane"}

    fx, fy = float(K[0, 0]), float(K[1, 1])
    x0, z0 = float(mrel[0] / mrel[1]), float(mrel[2] / mrel[1])
    crel = center - o
    # q_x - x0*q_y and q_z - z0*q_y, each as a trig numerator.
    hx = _trig_quadratic(float(crel[0] - x0 * crel[1]),
                         float(radius * (u_hat[0] - x0 * u_hat[1])),
                         float(radius * (v_hat[0] - x0 * v_hat[1])))
    hz = _trig_quadratic(float(crel[2] - z0 * crel[1]),
                         float(radius * (u_hat[2] - z0 * u_hat[1])),
                         float(radius * (v_hat[2] - z0 * v_hat[1])))
    ypoly = _trig_quadratic(float(crel[1]), float(radius * u_hat[1]), float(radius * v_hat[1]))
    numerator = _poly_add(fx * fx * _poly_mul(hx, hx), fy * fy * _poly_mul(hz, hz))
    derivative_numerator = np.polynomial.polynomial.polyder(numerator)
    derivative_y = np.polynomial.polynomial.polyder(ypoly)
    stationary = _poly_add(_poly_mul(derivative_numerator, ypoly),
                           -2.0 * _poly_mul(numerator, derivative_y))

    sqrt_f = np.sqrt(float(basis["in_plane_fraction"]))
    cosine_floor = float(UNIT_FORWARD_EPSILON / sqrt_f)
    alpha = float(np.arccos(np.clip(cosine_floor, -1.0, 1.0)))
    if requested > 0:
        endpoints = (-alpha, alpha)
    else:
        endpoints = (np.pi - alpha, np.pi + alpha)
    candidates: list[tuple[float, str]] = [(float(theta), "readability_boundary") for theta in endpoints]
    for root in np.polynomial.polynomial.polyroots(stationary):
        if abs(float(np.imag(root))) <= MACHINE_EPSILON * max(1.0, abs(float(np.real(root)))):
            candidates.append((float(2.0 * np.arctan(float(np.real(root)))), "stationary"))
    # t = infinity in the half-angle chart is theta = pi.
    candidates.append((float(np.pi), "half_angle_infinity"))

    def evaluate(theta: float) -> tuple[float, np.ndarray] | None:
        cosine, sine = np.cos(theta), np.sin(theta)
        if requested * cosine < cosine_floor - MACHINE_EPSILON:
            return None
        q = center + radius * (u_hat * cosine + v_hat * sine)
        qrel = q - o
        if qrel[1] == 0.0:
            return None
        pixels = np.array([fx * qrel[0] / qrel[1], -fy * qrel[2] / qrel[1]])
        target_pixels = np.array([fx * x0, -fy * z0])
        return float(np.linalg.norm(pixels - target_pixels)), q

    evaluated = []
    for theta, source in candidates:
        value = evaluate(theta)
        if value is not None and np.isfinite(value[0]):
            evaluated.append((value[0], theta, source, value[1]))
    if not evaluated:
        return {"resolved": False, "reason": "no finite readable branch candidate"}
    best = min(evaluated, key=lambda item: (item[0], item[1]))
    return {
        "resolved": True,
        "objective": "minimum perspective pixel displacement from current middle joint",
        "feasible_set": "bone-preserving circle intersected with the existing readable requested-sign arc",
        "requested": int(requested),
        "minimum_pixel_displacement": float(best[0]),
        "theta_radians": float(best[1]),
        "candidate_source": best[2],
        "candidate_count": len(evaluated),
        "candidate_point": _json_vector(best[3]),
        "stationary_polynomial_degree": int(max(0, len(stationary) - 1)),
        "circle_radius_m": radius,
        "cosine_floor": cosine_floor,
        "no_sampling": True,
        "no_weighted_3d_term": True,
    }


def _point_on_circle(locus: dict[str, Any], cosine: float, sine: float) -> np.ndarray:
    basis = circle_basis(locus)
    return (_vec(locus["circle_center"]) + float(locus["circle_radius"]) *
            (_vec(basis["u_hat"]) * cosine + _vec(basis["v_hat"]) * sine))


def _fixture(name: str, proximal: Any, distal: Any, radius: float, middle: Any,
             origin: Any = (0.0, 0.0, 0.0)) -> dict[str, Any]:
    p, d, m, o = map(_vec, (proximal, distal, middle, origin))
    rp, rd = float(np.linalg.norm(m - p)), float(np.linalg.norm(m - d))
    # ``radius`` documents the intended construction radius.  The audited
    # locus always derives from the two actual adjacent lengths, avoiding a
    # second rounded representation of the same fixture.
    locus = bone_preserving_locus(p, d, rp, rd)
    intersections = ray_locus_intersections(locus, o, m - o)
    current = branch_coordinates(m, locus)
    requested = -int(current["sign"]) if current["sign"] in (-1, 1) else None
    opposite = (opposite_branch_available(locus, requested)
                if requested is not None else {"available": False, "reason": "current SignState is UNKNOWN"})
    opposite_hits = []
    for point in intersections["points"]:
        hit = branch_coordinates(point, locus)
        if requested is not None and hit["sign"] == requested:
            opposite_hits.append(point)
    return {
        "name": name,
        "P": _json_vector(p), "D": _json_vector(d), "M": _json_vector(m),
        "bone_preserving_locus": locus,
        "camera_ray": {"origin": _json_vector(o), "direction": _json_vector(_unit(m - o))},
        "camera_ray_intersection": intersections,
        "current_branch": current,
        "requested_opposite_branch": requested,
        "opposite_branch_available_on_circle": opposite,
        "opposite_branch_intersection_count": len(opposite_hits),
        "non_trivial_branch_changing_solution": bool(opposite_hits),
    }


def synthetic_contracts() -> dict[str, Any]:
    """Construct the deterministic geometry fixtures required by the audit."""
    # Ordinary oblique axis: the middle point has both depth and screen-plane
    # bend components, and the camera ray meets the circle plane once.
    p, d = np.array([0.0, 2.0, 0.0]), np.array([1.0, 3.0, 0.0])
    construction = bone_preserving_locus(p, d, np.sqrt(1.5), np.sqrt(1.5))
    ordinary_m = _point_on_circle(construction, 0.8, 0.6)

    # Fronto-parallel means the limb axis has no camera-depth component.
    p_front, d_front = np.array([0.0, 3.0, 0.0]), np.array([1.0, 3.0, 1.0])
    front_locus = bone_preserving_locus(p_front, d_front, np.sqrt(1.5), np.sqrt(1.5))
    front_m = _point_on_circle(front_locus, 0.8, 0.6)

    # Exact and near depth alignment exercise the frozen SignState readability
    # semantics without inventing a new threshold.
    p_depth, d_depth = np.array([0.0, 2.0, 0.0]), np.array([0.0, 5.0, 0.0])
    depth_m = np.array([1.0, 3.5, 0.0])
    p_near, d_near = np.array([0.0, 2.0, 0.0]), np.array([0.01, 5.0, 0.0])
    near_radius = np.sqrt(1.0 + (np.linalg.norm(d_near - p_near) / 2.0) ** 2)
    near_locus = bone_preserving_locus(p_near, d_near, near_radius, near_radius)
    near_m = _point_on_circle(near_locus, 0.8, 0.6)

    # A ray contained in the plane can cut a circle twice.  Both points are in
    # front of this camera, and they carry opposite readable depth branches.
    p_parallel, d_parallel = np.array([-1.0, 3.0, 0.0]), np.array([1.0, 3.0, 0.0])
    parallel_m = np.array([0.0, 5.0, 0.0])

    tangent_p, tangent_d = np.array([-1.0, 3.0, 0.0]), np.array([1.0, 3.0, 0.0])
    tangent_m = np.array([0.0, 3.0, 0.0])

    cases = [
        _fixture("ordinary_oblique", p, d, np.sqrt(1.5), ordinary_m),
        _fixture("fronto_parallel", p_front, d_front, np.sqrt(1.5), front_m),
        _fixture("depth_aligned", p_depth, d_depth, np.sqrt(3.25), depth_m),
        _fixture("near_depth_aligned", p_near, d_near, near_radius, near_m),
        _fixture("ray_parallel_and_contained_in_circle_plane", p_parallel, d_parallel,
                 np.sqrt(5.0), parallel_m),
        _fixture("degenerate_zero_radius_circle", tangent_p, tangent_d, 1.0, tangent_m),
        {"name": "inconsistent_spheres", "bone_preserving_locus": bone_preserving_locus(
            tangent_p, tangent_d, 0.5, 0.5), "camera_ray_intersection": None,
         "non_trivial_branch_changing_solution": False},
        {"name": "zero_length_axis_equal_spheres", "bone_preserving_locus": bone_preserving_locus(
            [0.0, 3.0, 0.0], [0.0, 3.0, 0.0], 1.0, 1.0),
         "camera_ray_intersection": None, "non_trivial_branch_changing_solution": None},
    ]
    return {"schema": "animcv_hinge_feasibility_synthetic_v1", "cases": cases}


def branch_constraint_input_contract() -> dict[str, Any]:
    """The actual runtime/research API, recorded without changing it."""
    return {
        "callable": "framepose.branch_constraints.apply_branch_constraints",
        "required_inputs": {
            "pose": "root-relative canonical (17, 3) prediction",
            "valid": "one (17,) boolean validity mask",
            "requested": "one (7,) -1/0/+1 SignState array",
        },
        "optional_inputs": {"fields": "selected SignState fields",
                            "bilateral_write_policy": "bilateral-only policy",
                            "hinge_write_policy": "DEPTH_ONLY or MINIMUM_NORM"},
        "not_received": ["input_2d", "camera intrinsics", "camera extrinsics",
                          "absolute root depth", "image size"],
        "diagnostic_only_sources": ["docs/38/39 oracle absolute root placement",
                                    "3DPW raw cam_intrinsics/cam_poses",
                                    "stored input_2d"],
        "production_default": "DEPTH_ONLY",
    }


def _quantiles(values: list[float]) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return {"count": 0}
    return {"count": int(len(values)), "p50": float(np.percentile(values, 50)),
            "p90": float(np.percentile(values, 90)), "p99": float(np.percentile(values, 99)),
            "max": float(np.max(values))}


def real_scene_feasibility(bank_path: Path, source_spec: str, raw_root: Path,
                           raw_split: str = "test", split: str = "test") -> dict[str, Any]:
    """Account for every corrected real-scene hinge frame when artifacts exist.

    This intentionally reuses docs/39's verified reconstruction and oracle
    absolute-pelvis placement.  The caller must supply the exact bank,
    prediction, and raw 3DPW pickles; this function never substitutes a
    synthetic camera or an inferred root depth.
    """
    # The existing docs/39 loader is the provenance owner for 3DPW camera
    # reconstruction.  Import it only for this optional diagnostic path.
    scripts_root = str(Path(__file__).resolve().parent)
    if scripts_root not in sys.path:
        sys.path.insert(0, scripts_root)
    from diagnose_hinge_write_policy_observation import load_absolute_sequences
    from common.canonical_pose import JOINT_INDEX, JOINT_NAMES
    from framepose.bank import load_bank
    from framepose.branch_constraints import CORRECTED, apply_branch_constraints_batch
    from framepose.signs import HINGE_CHAINS_BY_JOINT, SIGN_FIELD_NAMES, mask_fields, oracle_sign_states
    from pose.three_dpw_adapter import _SMPL_TO_CANONICAL

    label, rest = source_spec.split("=", 1)
    parts = rest.split(":")
    candidate, prediction_path = parts[0], Path(parts[1])
    bank = load_bank(bank_path)
    positions = bank.indices(split)
    valid = bank.arrays["target_valid"][positions]
    oracle = oracle_sign_states(bank.arrays["target_3d"], bank.arrays["target_valid"])[positions]
    hinge_fields = tuple(name for name in SIGN_FIELD_NAMES if name.endswith("_forward_bend"))
    requested = mask_fields(oracle, list(hinge_fields)).astype(np.int64)
    h0 = np.load(prediction_path).astype(np.float64)
    states, reports = apply_branch_constraints_batch(h0, valid, requested, fields=list(hinge_fields))
    sequences, raw_provenance = load_absolute_sequences(raw_root, raw_split)
    smpl_rows = np.array([_SMPL_TO_CANONICAL[name] for name in JOINT_NAMES], dtype=np.int64)
    absolute_target = np.full((len(positions), len(JOINT_NAMES), 3), np.nan)
    intrinsics = np.full((len(positions), 3, 3), np.nan)
    for order, position in enumerate(positions):
        sample = bank.samples[int(position)]
        entry = sequences.get(sample.sequence_id)
        if entry is None or int(sample.frame_index) >= len(entry["absolute_smpl24"]):
            continue
        absolute_target[order] = entry["absolute_smpl24"][int(sample.frame_index)][smpl_rows]
        intrinsics[order] = entry["intrinsics"]

    root = absolute_target[:, JOINT_INDEX["pelvis"]][:, None, :]
    usable = ~np.isnan(absolute_target).any(axis=(1, 2))
    fields: dict[str, Any] = {}
    for field in hinge_fields:
        joint = field[: -len("_forward_bend")]
        proximal, middle, distal = HINGE_CHAINS_BY_JOINT[joint]
        indices = [JOINT_INDEX[name] for name in (proximal, middle, distal)]
        field_index = SIGN_FIELD_NAMES.index(field)
        records: list[dict[str, Any]] = []
        counts = Counter()
        incidence = []
        plane_offset = []
        radius_values = []
        axis_lengths = []
        lower_bounds = []
        for order, position in enumerate(positions):
            if not usable[order] or reports[order]["fields"][field]["outcome"] != CORRECTED:
                continue
            p, m, d = (states[order, index] + root[order, 0] for index in indices)
            locus = bone_preserving_locus(p, d, float(np.linalg.norm(m - p)),
                                          float(np.linalg.norm(m - d)))
            ray = ray_locus_intersections(locus, (0.0, 0.0, 0.0), m)
            branch = branch_coordinates(m, locus)
            wanted = int(requested[order, field_index])
            opposite = sum(1 for point in ray["points"]
                           if branch_coordinates(point, locus)["sign"] == wanted)
            counts[str(ray["intersection_count"])] += 1
            if ray["intersection_count"] not in (0, 1, 2):
                counts["other"] += 1
            if ray["plane_denominator"] is not None:
                normal_length = float(np.linalg.norm(np.asarray(locus["plane_normal"])))
                incidence.append(abs(float(ray["plane_denominator"])) / normal_length)
                plane_offset.append(abs(float(ray["plane_offset"])) / normal_length)
            radius_values.append(float(locus["circle_radius"]))
            axis_lengths.append(float(locus["axis_length"]))
            lower = minimum_image_displacement_on_branch(
                m, locus, intrinsics[order], wanted, origin=(0.0, 0.0, 0.0))
            lower_bounds.append(lower)
            records.append({
                "sample_id": bank.samples[int(position)].sample_id,
                "sequence_id": bank.samples[int(position)].sequence_id,
                "frame_index": int(bank.samples[int(position)].frame_index),
                "requested_sign": wanted,
                "current_sign": int(branch["sign"]),
                "circle_status": locus["status"],
                "ray_plane_relation": ray["relation"],
                "ray_circle_intersection_count": int(ray["intersection_count"]),
                "opposite_branch_intersection_count": int(opposite),
                "ray_plane_incidence_abs": (incidence[-1] if ray["plane_denominator"] is not None else None),
                "ray_plane_offset_abs_m": (plane_offset[-1] if ray["plane_offset"] is not None else None),
                "axis_length_m": float(locus["axis_length"]),
                "circle_radius_m": float(locus["circle_radius"]),
                "in_plane_fraction": branch.get("in_plane_fraction"),
                "lower_bound": lower,
            })
        fields[field] = {
            "middle_joint": middle,
            "corrected_frames_with_camera_geometry": len(records),
            "ray_circle_intersection_counts": dict(counts),
            "second_opposite_branch_intersections": int(sum(
                record["opposite_branch_intersection_count"] > 0 for record in records)),
            "ray_plane_incidence_abs": _quantiles(incidence),
            "ray_plane_offset_abs_m": _quantiles(plane_offset),
            "axis_length_m": _quantiles(axis_lengths),
            "circle_radius_m": _quantiles(radius_values),
            "lower_bound_resolved": int(sum(item.get("resolved", False) for item in lower_bounds)),
            "lower_bound_pixel_displacement": _quantiles([
                float(item["minimum_pixel_displacement"])
                for item in lower_bounds if item.get("resolved")]),
            "records": records,
        }
    return {
        "schema": "animcv_hinge_feasibility_real_v1",
        "source": {"label": label, "candidate": candidate, "prediction": str(prediction_path)},
        "camera_placement": "oracle_absolute_root_placement, identical to docs/39",
        "raw_provenance": raw_provenance,
        "fields": fields,
        "no_production_policy": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Synthetic hinge feasibility contracts")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--bank", type=Path)
    parser.add_argument("--source", help="LABEL=CANDIDATE:PREDICTION[:EVALUATION]")
    parser.add_argument("--raw-root", type=Path)
    parser.add_argument("--raw-split", default="test")
    parser.add_argument("--split", default="test")
    args = parser.parse_args()
    real_args = (args.bank, args.source, args.raw_root)
    if any(value is not None for value in real_args) and not all(value is not None for value in real_args):
        parser.error("--bank, --source and --raw-root must be supplied together for real-scene accounting")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"geometry": synthetic_contracts(),
               "branch_constraint_input_contract": branch_constraint_input_contract(),
               "camera_aware_lower_bound": {"implemented": True,
                   "status": "diagnostic_only; no production policy"}}
    if all(value is not None for value in real_args):
        payload["real_scene_feasibility"] = real_scene_feasibility(
            args.bank, args.source, args.raw_root, raw_split=args.raw_split, split=args.split)
    else:
        payload["real_scene_feasibility"] = {
            "status": "not_run",
            "reason": "verified docs/39 bank, prediction and raw 3DPW camera pickles are not present in this workspace",
        }
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
