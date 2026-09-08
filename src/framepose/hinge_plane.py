"""The local plane a hinge bend lives in, and who can observe each of its axes.

docs/37. A hinge bend direction is a unit vector perpendicular to the chain's
proximal->distal axis, so it lives in a 2D plane. That plane has two natural
axes:

    u_hat   the projected camera-depth direction -- HIDDEN evidence, the one
            the Sign Contract's `*_forward_bend` bit already encodes
    v_hat   perpendicular to both the limb axis and the camera depth axis, so
            it lies exactly in the canonical X/Z image plane -- VISIBLE

Splitting the bend this way is a diagnostic instrument for asking who owns the
residual hinge disagreement. **Nothing here is added to SignState**, no new
sign is emitted, and the Sign Contract is untouched.

Sign convention for `v_hat`, fixed by construction and never chosen from a
measured result:

    v_hat = normalize(cross(a_hat, e_y))

`cross(a, e_y)` has zero Y component for any `a`, which is the analytic reason
`v_hat` lies in the image plane; see `test_v_hat_lies_in_the_canonical_image_plane`.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from common.canonical_pose import (
    FORWARD_DEPTH_AXIS, JOINT_INDEX, VECTOR_NORMALIZATION_EPS,
)

HINGE_PLANE_SCHEMA = "animcv_frame_pose_hinge_plane_v1"

#: Measured from `bank_3dpw_paired_v2`, not assumed: over all 7,076 test frames
#: the per-frame correlation of `input_2d[:, 0]` with canonical X is positive on
#: 100% of frames (median r +0.975) and of `input_2d[:, 1]` with canonical Z is
#: negative on 100% of frames (median r -0.992), with no cross-coupling
#: (|r| <= 0.03). So the observation's image axes map to the canonical image
#: plane as (X, Z) = (+x, -y): the standard convention with image y downward.
IMAGE_TO_CANONICAL = {"canonical_x_from": "+input_2d_x", "canonical_z_from": "-input_2d_y"}


def _unit(vector: np.ndarray) -> np.ndarray | None:
    norm = float(np.linalg.norm(vector))
    if norm < VECTOR_NORMALIZATION_EPS:
        return None
    return vector / norm


def local_basis(pose: np.ndarray, chain: tuple[str, str, str]) -> dict[str, Any] | None:
    """Orthonormal `(a_hat, u_hat, v_hat)` for one hinge chain, or None.

    `u_hat` and `v_hat` span the plane perpendicular to the limb axis. Returns
    None when the axis is degenerate or lies exactly along the camera depth
    axis, where the depth direction has no projection into that plane at all.
    """
    proximal, _, distal = chain
    axis = pose[JOINT_INDEX[distal]] - pose[JOINT_INDEX[proximal]]
    a_hat = _unit(axis)
    if a_hat is None:
        return None
    depth = np.zeros(3)
    depth[FORWARD_DEPTH_AXIS] = 1.0
    u = depth - a_hat * float(a_hat @ depth)
    u_hat = _unit(u)
    v_hat = _unit(np.cross(a_hat, depth))
    if u_hat is None or v_hat is None:
        return None
    return {"a_hat": a_hat, "u_hat": u_hat, "v_hat": v_hat,
            "in_plane_fraction": float(u @ u),
            "axis_length_m": float(np.linalg.norm(axis))}


def bend_components(pose: np.ndarray, chain: tuple[str, str, str]) -> dict[str, Any] | None:
    """Decompose the perpendicular bend offset as `o = c_depth*u_hat + c_screen*v_hat`.

    `sign(c_depth)` is the same quantity the Sign Contract's `*_forward_bend`
    field reads: `o . u_hat = o_y / sqrt(f)` and `sqrt(f) > 0`, so the two signs
    agree wherever the contract finds the field readable. `c_screen` is the
    complementary in-image-plane side, which the contract does not encode.
    """
    basis = local_basis(pose, chain)
    if basis is None:
        return None
    proximal, middle, distal = chain
    relative = pose[JOINT_INDEX[middle]] - pose[JOINT_INDEX[proximal]]
    offset = relative - basis["a_hat"] * float(relative @ basis["a_hat"])
    return {
        **basis,
        "offset": offset,
        "offset_norm_m": float(np.linalg.norm(offset)),
        "c_depth": float(offset @ basis["u_hat"]),
        "c_screen": float(offset @ basis["v_hat"]),
        "along_axis_m": float(relative @ basis["a_hat"]),
    }


def observed_screen_side(input_2d: np.ndarray, input_valid: np.ndarray,
                         chain: tuple[str, str, str]) -> dict[str, Any]:
    """Which side of the proximal->distal line the middle joint is observed on.

    A pure orientation predicate on the 2D observation: no magnitude, no learned
    feature, no RGB, no ground truth. Derived, not fitted:

        c_screen = (o . v_hat) = (a_X * r_Z - a_Z * r_X) / sqrt(f)

    with `a = distal - proximal` and `r = middle - proximal`, because the
    axis-parallel parts cancel. `sqrt(f) > 0`, so `sign(c_screen)` is the sign of
    the 2D cross product of the axis with the proximal->middle vector, taken in
    the canonical image plane. Substituting `(X, Z) = (+x, -y)` from
    `IMAGE_TO_CANONICAL` gives, in raw observation coordinates,

        side = sign( (d_y - p_y)(m_x - p_x) - (d_x - p_x)(m_y - p_y) )

    An exactly collinear observation has no side. Rather than introduce a
    collinearity tolerance -- which would be a policy threshold chosen from
    results -- only the exact zero is called unresolved, and the continuous
    magnitude is returned so its distribution can be reported.
    """
    proximal, middle, distal = chain
    indices = [JOINT_INDEX[name] for name in (proximal, middle, distal)]
    if not all(bool(input_valid[index]) for index in indices):
        return {"side": 0, "cross": None, "resolved": False, "reason": "a chain joint is unobserved"}
    p, m, d = (np.asarray(input_2d[index][:2], dtype=np.float64) for index in indices)
    cross = float((d[1] - p[1]) * (m[0] - p[0]) - (d[0] - p[0]) * (m[1] - p[1]))
    if cross == 0.0:
        return {"side": 0, "cross": cross, "resolved": False,
                "reason": "the observed chain is exactly collinear, so no side exists"}
    return {"side": 1 if cross > 0 else -1, "cross": cross, "resolved": True, "reason": None}


def axis_transport(predicted: np.ndarray, target: np.ndarray,
                   chain: tuple[str, str, str]) -> dict[str, Any]:
    """Rotate the target bend by the minimal rotation aligning the two axes.

    Historical hinge error mixes bend-side error with limb-axis orientation
    error. Transporting the target bend direction through the minimal rotation
    that maps the target axis onto the predicted axis removes the second, so
    what remains answers: *if both had the same limb axis, would the bend still
    be on the wrong side?*

    Degeneracies are reported, never smoothed: an antiparallel axis pair admits
    no unique minimal rotation, and a vanishing bend has no direction.
    """
    proximal, middle, distal = chain
    predicted_axis = _unit(predicted[JOINT_INDEX[distal]] - predicted[JOINT_INDEX[proximal]])
    target_axis = _unit(target[JOINT_INDEX[distal]] - target[JOINT_INDEX[proximal]])
    if predicted_axis is None or target_axis is None:
        return {"resolved": False, "reason": "a limb axis is degenerate"}

    cosine = float(np.clip(target_axis @ predicted_axis, -1.0, 1.0))
    axis_angle = float(np.degrees(np.arccos(cosine)))
    cross = np.cross(target_axis, predicted_axis)
    sine = float(np.linalg.norm(cross))
    if sine < VECTOR_NORMALIZATION_EPS:
        if cosine > 0:
            rotation = np.eye(3)          # already aligned: transport is identity
        else:
            return {"resolved": False, "axis_angle_degrees": axis_angle,
                    "reason": "the limb axes are antiparallel, so the minimal rotation is not unique"}
    else:
        k = cross / sine
        skew = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
        rotation = np.eye(3) + skew * sine + skew @ skew * (1.0 - cosine)

    from common.canonical_pose import bend_direction

    predicted_bend = bend_direction(predicted[JOINT_INDEX[middle]],
                                    predicted[JOINT_INDEX[proximal]],
                                    predicted[JOINT_INDEX[distal]])
    target_bend = bend_direction(target[JOINT_INDEX[middle]],
                                 target[JOINT_INDEX[proximal]],
                                 target[JOINT_INDEX[distal]])
    if predicted_bend is None or target_bend is None:
        return {"resolved": False, "axis_angle_degrees": axis_angle,
                "reason": "a bend direction is degenerate"}

    transported = rotation @ target_bend
    aligned_cosine = float(np.clip(predicted_bend @ transported, -1.0, 1.0))
    raw_cosine = float(np.clip(predicted_bend @ target_bend, -1.0, 1.0))
    return {
        "resolved": True,
        "axis_angle_degrees": axis_angle,
        "historical_error_degrees": float(np.degrees(np.arccos(raw_cosine))),
        "axis_normalized_error_degrees": float(np.degrees(np.arccos(aligned_cosine))),
        "historical_flipped": raw_cosine < 0,
        "axis_normalized_flipped": aligned_cosine < 0,
        "transported_target_bend": transported,
        "predicted_bend": predicted_bend,
    }
