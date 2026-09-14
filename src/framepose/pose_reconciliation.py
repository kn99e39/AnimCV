"""Research-only endpoint-fixed two-bone swivel reconciliation.

This module is intentionally separate from ``branch_constraints``.  The
historical ``DEPTH_ONLY`` and ``MINIMUM_NORM`` operators remain untouched.

The layer treats a hinge chain as a two-bone IK chain with fixed endpoints:

    P (proximal endpoint) -- M (swivel mediator) -- D (distal endpoint)

The only free variable is the angle of ``M`` around the exact bone-preserving
circle.  Geometry Observation selects the visible point on the readable arc;
SignState selects only the hidden forward/depth branch.

This is a FramePose position-space reconciliation.  It does not author rig
bone-roll quaternions or claim wrist/ankle orientation preservation.  That
orientation requirement belongs to downstream articulated-rig IK/retargeting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from common.canonical_pose import JOINT_INDEX, JOINT_NAMES
from framepose.signs import (
    HINGE_CHAINS_BY_JOINT,
    MIN_BEND_OFFSET_M,
    SIGN_FIELD_NAMES,
    UNIT_FORWARD_EPSILON,
    UNKNOWN,
    sign_state,
)


POSE_RECONCILIATION_SCHEMA = "animcv_frame_pose_reconciliation_v1"
CORRECTED = "corrected"
ALREADY_SATISFIED = "already_satisfied"
UNRESOLVED = "unresolved"
OUTCOMES = (CORRECTED, ALREADY_SATISFIED, UNRESOLVED)
HINGE_FIELDS = tuple(name for name in SIGN_FIELD_NAMES if name.endswith("_forward_bend"))
MACHINE_EPSILON = np.finfo(np.float64).eps


@dataclass(frozen=True)
class ProjectionContext:
    """Explicit projection state for research reconciliation.

    ``pose`` values are root-relative canonical coordinates.  The context must
    carry the camera-space root placement used to project them.  The current
    controlled replay uses ``research_oracle_absolute_root_placement`` from
    docs/39; this is not production root inference.
    """

    intrinsics: np.ndarray
    image_size: tuple[float, float]
    root_offset_camera: np.ndarray | None
    camera_origin: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    placement_mode: str = "research_oracle_absolute_root_placement"

    def validation_error(self) -> str | None:
        K = np.asarray(self.intrinsics, dtype=np.float64)
        size = np.asarray(self.image_size, dtype=np.float64)
        origin = np.asarray(self.camera_origin, dtype=np.float64)
        if K.shape != (3, 3) or not np.isfinite(K).all():
            return "projection intrinsics must be finite (3, 3)"
        if size.shape != (2,) or not np.isfinite(size).all() or (size <= 0).any():
            return "projection image_size must be two positive finite values"
        if self.root_offset_camera is None:
            return "root_offset_camera is required; root depth must not be invented"
        root = np.asarray(self.root_offset_camera, dtype=np.float64)
        if root.shape != (3,) or not np.isfinite(root).all():
            return "root_offset_camera must be a finite 3-vector"
        if origin.shape != (3,) or not np.isfinite(origin).all():
            return "camera_origin must be a finite 3-vector"
        if K[0, 0] == 0.0 or K[1, 1] == 0.0:
            return "projection focal lengths must be non-zero"
        return None

    def observation_pixels(self, observed_middle_2d: Any) -> np.ndarray:
        observation = np.asarray(observed_middle_2d, dtype=np.float64)
        if observation.shape not in {(2,), (3,)}:
            raise ValueError("observed middle 2D must have shape (2,) or (3,)")
        if not np.isfinite(observation[:2]).all():
            raise ValueError("observed middle 2D coordinates must be finite")
        return observation[:2] * np.asarray(self.image_size, dtype=np.float64)

    def project_root_relative(self, point: Any) -> np.ndarray:
        error = self.validation_error()
        if error:
            raise ValueError(error)
        relative = np.asarray(point, dtype=np.float64) + np.asarray(self.root_offset_camera)
        relative = relative - np.asarray(self.camera_origin, dtype=np.float64)
        if relative.shape != (3,) or not np.isfinite(relative).all():
            raise ValueError("point to project must be a finite 3-vector")
        if relative[1] <= 0.0:
            raise ValueError("point to project must have positive camera depth")
        K = np.asarray(self.intrinsics, dtype=np.float64)
        return np.asarray([
            K[0, 0] * relative[0] / relative[1] + K[0, 2],
            -K[1, 1] * relative[2] / relative[1] + K[1, 2],
        ])

    def to_dict(self) -> dict[str, Any]:
        return {
            "intrinsics": np.asarray(self.intrinsics, dtype=float).tolist(),
            "image_size": [float(v) for v in self.image_size],
            "root_offset_camera": (None if self.root_offset_camera is None else
                                    np.asarray(self.root_offset_camera, dtype=float).tolist()),
            "camera_origin": np.asarray(self.camera_origin, dtype=float).tolist(),
            "placement_mode": self.placement_mode,
            "research_only": True,
        }


def _machine_zero(value: float, scale: float = 1.0) -> bool:
    return abs(float(value)) <= 16.0 * MACHINE_EPSILON * max(1.0, abs(float(scale)))


def _vector(value: Any, name: str = "vector") -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (3,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite 3-vector")
    return result


def _unit(value: np.ndarray) -> np.ndarray | None:
    norm = float(np.linalg.norm(value))
    return None if norm == 0.0 else value / norm


@dataclass(frozen=True)
class BoneCircle:
    status: str
    center: np.ndarray | None
    axis: np.ndarray
    axis_length: float
    radius: float | None
    radius_squared: float | None
    proximal_length: float
    distal_length: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "center": None if self.center is None else self.center.tolist(),
            "axis": self.axis.tolist(),
            "axis_length_m": self.axis_length,
            "radius_m": self.radius,
            "radius_squared_m2": self.radius_squared,
            "proximal_length_m": self.proximal_length,
            "distal_length_m": self.distal_length,
        }


def _bone_circle(proximal: np.ndarray, middle: np.ndarray, distal: np.ndarray) -> BoneCircle:
    axis = distal - proximal
    axis_squared = float(axis @ axis)
    axis_length = float(np.sqrt(axis_squared))
    proximal_length = float(np.linalg.norm(middle - proximal))
    distal_length = float(np.linalg.norm(middle - distal))
    if axis_squared == 0.0:
        if proximal_length != distal_length:
            status = "inconsistent_concentric_spheres"
        elif proximal_length == 0.0:
            status = "singleton"
        else:
            status = "sphere"
        return BoneCircle(status, proximal.copy(), axis, axis_length,
                          proximal_length, proximal_length * proximal_length,
                          proximal_length, distal_length)
    if axis_length > proximal_length + distal_length or axis_length < abs(proximal_length - distal_length):
        return BoneCircle("inconsistent_spheres", None, axis, axis_length, None, None,
                          proximal_length, distal_length)
    lam = (proximal_length ** 2 - distal_length ** 2 + axis_squared) / (2.0 * axis_squared)
    center = proximal + lam * axis
    radius_squared = proximal_length ** 2 - lam ** 2 * axis_squared
    if radius_squared < 0.0:
        return BoneCircle("inconsistent_spheres", center, axis, axis_length, None, radius_squared,
                          proximal_length, distal_length)
    if radius_squared == 0.0:
        return BoneCircle("zero_radius", center, axis, axis_length, 0.0, 0.0,
                          proximal_length, distal_length)
    return BoneCircle("circle", center, axis, axis_length, float(np.sqrt(radius_squared)),
                      float(radius_squared), proximal_length, distal_length)


def _basis(circle: BoneCircle) -> tuple[np.ndarray, np.ndarray, float] | None:
    if circle.status != "circle":
        return None
    axis_hat = circle.axis / circle.axis_length
    depth = np.asarray([0.0, 1.0, 0.0])
    u = depth - axis_hat * float(axis_hat @ depth)
    sqrt_f = float(np.linalg.norm(u))
    if sqrt_f == 0.0:
        return None
    return u / sqrt_f, np.cross(axis_hat, depth) / sqrt_f, sqrt_f


def _branch_readability(circle: BoneCircle, point: np.ndarray) -> dict[str, Any]:
    basis = _basis(circle)
    if basis is None:
        return {"readable": False, "sign": UNKNOWN, "reason": "depth basis is undefined"}
    if circle.radius is None or circle.center is None:
        return {"readable": False, "sign": UNKNOWN, "reason": "circle is degenerate"}
    u_hat, v_hat, sqrt_f = basis
    offset = point - circle.center
    unit_forward = float((offset / circle.radius)[1])
    sign = UNKNOWN if unit_forward == 0.0 else (1 if unit_forward > 0.0 else -1)
    readable = (circle.radius >= MIN_BEND_OFFSET_M and
                sqrt_f >= UNIT_FORWARD_EPSILON and
                abs(unit_forward) >= UNIT_FORWARD_EPSILON)
    return {
        "readable": bool(readable),
        "sign": int(sign if readable else UNKNOWN),
        "unit_forward": unit_forward,
        "in_plane_fraction": sqrt_f * sqrt_f,
        "reason": None if readable else "existing Sign Contract readability semantics reject it",
        "c_depth_m": float(offset @ u_hat),
        "c_screen_m": float(offset @ v_hat),
    }


def _poly_add(*terms: np.ndarray) -> np.ndarray:
    size = max(len(term) for term in terms)
    result = np.zeros(size, dtype=np.float64)
    for term in terms:
        result[:len(term)] += term
    return result


def _poly_mul(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    return np.polynomial.polynomial.polymul(first, second)


def _trig_quadratic(constant: float, cosine: float, sine: float) -> np.ndarray:
    return _poly_add(np.array([constant, 0.0, constant]),
                     cosine * np.array([1.0, 0.0, -1.0]),
                     sine * np.array([0.0, 2.0]))


def _solve_swivel(circle: BoneCircle, observed_pixels: np.ndarray, context: ProjectionContext,
                  requested: int) -> dict[str, Any]:
    if circle.status != "circle":
        return {"resolved": False, "reason": "bone-preserving locus is degenerate", "circle": circle.to_dict()}
    basis = _basis(circle)
    if basis is None:
        return {"resolved": False, "reason": "camera-depth branch is not observable", "circle": circle.to_dict()}
    u_hat, v_hat, sqrt_f = basis
    if circle.radius is None or circle.radius < MIN_BEND_OFFSET_M:
        return {"resolved": False, "reason": "circle radius is below Sign Contract bend floor",
                "circle": circle.to_dict()}
    if sqrt_f < UNIT_FORWARD_EPSILON:
        return {"resolved": False, "reason": "no swivel point can satisfy Sign Contract forward floor",
                "circle": circle.to_dict()}
    K = np.asarray(context.intrinsics, dtype=np.float64)
    origin = np.asarray(context.camera_origin, dtype=np.float64)
    center = circle.center + np.asarray(context.root_offset_camera) - origin
    target_x = (float(observed_pixels[0]) - K[0, 2]) / K[0, 0]
    target_z = -(float(observed_pixels[1]) - K[1, 2]) / K[1, 1]
    radius = float(circle.radius)

    hx = _trig_quadratic(float(center[0] - target_x * center[1]),
                         radius * float(u_hat[0] - target_x * u_hat[1]),
                         radius * float(v_hat[0] - target_x * v_hat[1]))
    hz = _trig_quadratic(float(center[2] - target_z * center[1]),
                         radius * float(u_hat[2] - target_z * u_hat[1]),
                         radius * float(v_hat[2] - target_z * v_hat[1]))
    ypoly = _trig_quadratic(float(center[1]), radius * float(u_hat[1]), radius * float(v_hat[1]))
    numerator = _poly_add(K[0, 0] ** 2 * _poly_mul(hx, hx),
                          K[1, 1] ** 2 * _poly_mul(hz, hz))
    derivative = _poly_add(
        _poly_mul(np.polynomial.polynomial.polyder(numerator), ypoly),
        -2.0 * _poly_mul(numerator, np.polynomial.polynomial.polyder(ypoly)))

    cosine_floor = float(UNIT_FORWARD_EPSILON / sqrt_f)
    alpha = float(np.arccos(np.clip(cosine_floor, -1.0, 1.0)))
    endpoints = ((-alpha, alpha) if requested > 0 else (np.pi - alpha, np.pi + alpha))
    candidates: list[tuple[float, str]] = [(float(theta), "readability_boundary") for theta in endpoints]
    for root in np.polynomial.polynomial.polyroots(derivative):
        real = float(np.real(root))
        if abs(float(np.imag(root))) <= 16.0 * MACHINE_EPSILON * max(1.0, abs(real)):
            candidates.append((float(2.0 * np.arctan(real)), "stationary"))
    candidates.append((float(np.pi), "half_angle_infinity"))

    evaluated: list[tuple[float, float, str, np.ndarray, np.ndarray]] = []
    for theta, source in candidates:
        cosine, sine = float(np.cos(theta)), float(np.sin(theta))
        if requested * cosine < cosine_floor - 16.0 * MACHINE_EPSILON:
            continue
        point = circle.center + radius * (u_hat * cosine + v_hat * sine)
        try:
            projected = context.project_root_relative(point)
        except ValueError:
            continue
        error = float(np.linalg.norm(projected - observed_pixels))
        if np.isfinite(error):
            evaluated.append((error, theta, source, point, projected))
    if not evaluated:
        return {"resolved": False, "reason": "no finite positive-depth readable swivel candidate",
                "circle": circle.to_dict()}
    best = min(evaluated, key=lambda item: (item[0], item[1]))
    return {
        "resolved": True,
        "circle": circle.to_dict(),
        "objective": "minimum middle-joint reprojection error to observed 2D",
        "feasible_set": "exact bone circle intersected with existing readable requested-sign arc",
        "requested_sign": int(requested),
        "minimum_reprojection_error_px": float(best[0]),
        "theta_radians": float(best[1]),
        "candidate_source": best[2],
        "candidate_count": len(evaluated),
        "stationary_polynomial_degree": int(max(0, len(derivative) - 1)),
        "in_plane_fraction": float(sqrt_f * sqrt_f),
        "candidate_point_root_relative": best[3].tolist(),
        "candidate_projection_px": best[4].tolist(),
        "no_angle_sampling": True,
        "no_3d_weight": True,
    }


def reconcile_hinge(pose: Any, valid: Any, requested: int, observed_middle_2d: Any,
                    context: ProjectionContext | None, field: str,
                    *, observed_valid: bool = True) -> tuple[np.ndarray, dict[str, Any]]:
    """Reconcile one hinge field using the endpoint-fixed swivel contract."""
    result = np.asarray(pose, dtype=np.float64)
    validity = np.asarray(valid, dtype=bool)
    if result.shape != (len(JOINT_NAMES), 3):
        raise ValueError(f"pose must be ({len(JOINT_NAMES)}, 3), got {result.shape}")
    if validity.shape != (len(JOINT_NAMES),):
        raise ValueError(f"valid must be ({len(JOINT_NAMES)},), got {validity.shape}")
    if field not in HINGE_FIELDS:
        raise ValueError(f"pose reconciliation supports hinge fields only, got {field!r}")
    if int(requested) not in (-1, UNKNOWN, 1):
        raise ValueError("requested SignState must be -1, 0 or +1")

    chain = HINGE_CHAINS_BY_JOINT[field[: -len("_forward_bend")]]
    indexes = [JOINT_INDEX[name] for name in chain]
    index = SIGN_FIELD_NAMES.index(field)
    before = int(sign_state(result, validity)[index])
    base: dict[str, Any] = {
        "field": field, "chain": list(chain), "requested_sign": int(requested),
        "read_back_before": before, "changed_joints": [],
        "orientation_contract": "wrist/ankle orientation is downstream IK/retargeting only",
    }
    if requested == UNKNOWN:
        return result.copy(), {**base, "outcome": UNRESOLVED,
                               "reason": "requested SignState is UNKNOWN; no guess"}
    if before == requested:
        return result.copy(), {**base, "outcome": ALREADY_SATISFIED,
                               "reason": "requested branch already satisfied; exact no-op"}
    if before == UNKNOWN:
        return result.copy(), {**base, "outcome": UNRESOLVED,
                               "reason": "current hinge branch is unreadable; no guess"}
    if not all(bool(validity[item]) for item in indexes):
        return result.copy(), {**base, "outcome": UNRESOLVED,
                               "reason": "hinge chain validity is insufficient"}
    if not observed_valid:
        return result.copy(), {**base, "outcome": UNRESOLVED,
                               "reason": "observed middle joint is invalid"}
    if context is None:
        return result.copy(), {**base, "outcome": UNRESOLVED,
                               "reason": "ProjectionContext is unavailable"}
    context_error = context.validation_error()
    if context_error:
        return result.copy(), {**base, "outcome": UNRESOLVED, "reason": context_error}
    try:
        observed_pixels = context.observation_pixels(observed_middle_2d)
    except ValueError as error:
        return result.copy(), {**base, "outcome": UNRESOLVED, "reason": str(error)}

    proximal, middle, distal = (result[item] for item in indexes)
    circle = _bone_circle(proximal, middle, distal)
    solved = _solve_swivel(circle, observed_pixels, context, int(requested))
    if not solved.get("resolved"):
        return result.copy(), {**base, **solved, "outcome": UNRESOLVED}

    candidate = result.copy()
    middle_index = indexes[1]
    candidate[middle_index] = np.asarray(solved["candidate_point_root_relative"], dtype=np.float64)
    after = int(sign_state(candidate, validity)[index])
    p_after, m_after, d_after = (candidate[item] for item in indexes)
    length_errors = [
        abs(float(np.linalg.norm(m_after - p_after)) - circle.proximal_length),
        abs(float(np.linalg.norm(d_after - m_after)) - circle.distal_length),
    ]
    endpoint_unchanged = (np.array_equal(candidate[indexes[0]], result[indexes[0]]) and
                          np.array_equal(candidate[indexes[2]], result[indexes[2]]))
    if after != requested or not endpoint_unchanged or max(length_errors) > 16.0 * MACHINE_EPSILON:
        return result.copy(), {**base, **solved, "outcome": UNRESOLVED,
                               "reason": "candidate failed exact endpoint, length, or SignState read-back",
                               "read_back_after_attempt": after,
                               "bone_length_abs_error_m": length_errors}
    circle_residual = abs(float((m_after - circle.center) @ (m_after - circle.center) -
                                circle.radius_squared))
    current_theta = float(np.arctan2(
        _branch_readability(circle, middle)["c_screen_m"],
        _branch_readability(circle, middle)["c_depth_m"]))
    delta_theta = float((solved["theta_radians"] - current_theta + np.pi) % (2.0 * np.pi) - np.pi)
    return candidate, {
        **base, **solved, "outcome": CORRECTED, "read_back_after": after,
        "changed_joints": [chain[1]], "swivel_theta_before_radians": current_theta,
        "swivel_delta_radians": delta_theta,
        "bone_length_abs_error_m": length_errors,
        "circle_membership_abs_error_m2": circle_residual,
        "endpoint_positions_unchanged": True,
        "middle_displacement_m": float(np.linalg.norm(candidate[middle_index] - result[middle_index])),
    }


def reconcile_pose(pose: Any, valid: Any, requested: Any, observed_2d: Any,
                   context: ProjectionContext | None, *, fields: Sequence[str] | None = None,
                   observed_valid: Any = None) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply reconciliation to selected hinge fields, with exact accounting."""
    result = np.asarray(pose, dtype=np.float64).copy()
    validity = np.asarray(valid, dtype=bool)
    requested_array = np.asarray(requested)
    observation = np.asarray(observed_2d, dtype=np.float64)
    if requested_array.shape != (len(SIGN_FIELD_NAMES),):
        raise ValueError(f"requested must have shape ({len(SIGN_FIELD_NAMES)},)")
    if observation.shape[0:2] != (len(JOINT_NAMES), 3):
        raise ValueError("observed_2d must have shape (17, 3)")
    selected = tuple(HINGE_FIELDS if fields is None else fields)
    observation_valid_array = (np.ones(len(JOINT_NAMES), dtype=bool) if observed_valid is None
                               else np.asarray(observed_valid, dtype=bool))
    if observation_valid_array.shape != (len(JOINT_NAMES),):
        raise ValueError("observed_valid must have shape (17,)")
    reports: dict[str, Any] = {}
    for field_name in selected:
        middle = JOINT_INDEX[HINGE_CHAINS_BY_JOINT[field_name[: -len("_forward_bend")]][1]]
        result, reports[field_name] = reconcile_hinge(
            result, validity, int(requested_array[SIGN_FIELD_NAMES.index(field_name)]),
            observation[middle], context, field_name, observed_valid=bool(observation_valid_array[middle]))
    final = sign_state(result, validity)
    displacement = np.linalg.norm(result - np.asarray(pose, dtype=np.float64), axis=-1)
    return result, {
        "schema": POSE_RECONCILIATION_SCHEMA,
        "layer": "Pose Reconciliation",
        "fields": reports,
        "projection_context": None if context is None else context.to_dict(),
        "final_sign_state": {name: int(value) for name, value in zip(SIGN_FIELD_NAMES, final)},
        "moved_joints": [JOINT_NAMES[index] for index in np.flatnonzero(displacement > 0.0)],
        "max_joint_displacement_m": float(displacement.max()),
        "wrist_ankle_orientation": "not represented; mandatory downstream IK/retargeting contract",
    }


def reconcile_pose_batch(poses: np.ndarray, valid: np.ndarray, requested: np.ndarray,
                         observed_2d: np.ndarray, contexts: Sequence[ProjectionContext | None],
                         *, fields: Sequence[str] | None = None,
                         observed_valid: np.ndarray | None = None) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Batch wrapper for controlled replay; each frame may have its own camera context."""
    poses = np.asarray(poses, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    requested = np.asarray(requested)
    observed_2d = np.asarray(observed_2d, dtype=np.float64)
    if poses.ndim != 3 or len(poses) != len(contexts):
        raise ValueError("poses must be (N, 17, 3) and contexts must have N entries")
    corrected = np.empty_like(poses)
    reports = []
    for index in range(len(poses)):
        frame, report = reconcile_pose(
            poses[index], valid[index], requested[index], observed_2d[index], contexts[index],
            fields=fields, observed_valid=None if observed_valid is None else observed_valid[index])
        corrected[index] = frame
        reports.append(report)
    return corrected, reports
