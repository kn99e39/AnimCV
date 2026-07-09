"""Axis correction utilities for the retarget solver (Architecture_v2.md section 7).

Keeps axis-vector/quaternion math out of fk_solver.py so hint parsing
and axis-angle rotation construction are testable and reusable in
isolation, and so the solver never hardcodes an assumption like "Y axis
points along the bone" — everything axis-related funnels through here.
"""

from __future__ import annotations

import math

from common.types import Matrix4, Quaternion, Vec3

_AXIS_VECTORS: dict[str, Vec3] = {
    "+X": (1.0, 0.0, 0.0),
    "-X": (-1.0, 0.0, 0.0),
    "+Y": (0.0, 1.0, 0.0),
    "-Y": (0.0, -1.0, 0.0),
    "+Z": (0.0, 0.0, 1.0),
    "-Z": (0.0, 0.0, -1.0),
}

# The 2D image-plane direction delta the fk_solver computes is, by
# default, applied as a rotation around the view axis (+Z) unless the
# Bone Mapping Profile's axis_hint says otherwise.
DEFAULT_ROTATION_AXIS: Vec3 = _AXIS_VECTORS["+Z"]


def axis_hint_to_vector(axis_hint: str | None, default: Vec3 = DEFAULT_ROTATION_AXIS) -> Vec3:
    if axis_hint is None:
        return default
    try:
        return _AXIS_VECTORS[axis_hint.upper()]
    except KeyError as exc:
        raise ValueError(
            f"unknown axis hint {axis_hint!r}; expected one of {sorted(_AXIS_VECTORS)}"
        ) from exc


def quaternion_from_axis_angle(axis: Vec3, angle_radians: float) -> Quaternion:
    """Unit quaternion (x, y, z, w) rotating by angle_radians around axis."""
    length = math.sqrt(sum(component * component for component in axis))
    if length == 0.0:
        return (0.0, 0.0, 0.0, 1.0)
    ax, ay, az = (component / length for component in axis)
    half = angle_radians / 2.0
    sin_half = math.sin(half)
    return (ax * sin_half, ay * sin_half, az * sin_half, math.cos(half))


def quaternion_multiply(a: Quaternion, b: Quaternion) -> Quaternion:
    """Hamilton product a * b (apply b first, then a)."""
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    )


def quaternion_conjugate(q: Quaternion) -> Quaternion:
    x, y, z, w = q
    return (-x, -y, -z, w)


def _normalize_quaternion(q: Quaternion) -> Quaternion:
    length = math.sqrt(sum(component * component for component in q))
    if length == 0.0:
        return (0.0, 0.0, 0.0, 1.0)
    return tuple(component / length for component in q)


def quaternion_from_matrix(matrix: Matrix4) -> Quaternion:
    """Extract the rotation as a unit quaternion from a 4x4 matrix's
    upper-left 3x3 submatrix (Shepperd's method). Columns are
    normalized first so a rig's rest-pose scale doesn't distort the
    result -- this only extracts orientation, never scale."""
    raw = [[matrix[r][c] for c in range(3)] for r in range(3)]
    for c in range(3):
        col_length = math.sqrt(sum(raw[r][c] ** 2 for r in range(3))) or 1.0
        for r in range(3):
            raw[r][c] /= col_length
    m = raw

    trace = m[0][0] + m[1][1] + m[2][2]
    if trace > 0:
        s = math.sqrt(trace + 1.0) * 2
        w = 0.25 * s
        x = (m[2][1] - m[1][2]) / s
        y = (m[0][2] - m[2][0]) / s
        z = (m[1][0] - m[0][1]) / s
    elif m[0][0] > m[1][1] and m[0][0] > m[2][2]:
        s = math.sqrt(1.0 + m[0][0] - m[1][1] - m[2][2]) * 2
        w = (m[2][1] - m[1][2]) / s
        x = 0.25 * s
        y = (m[0][1] + m[1][0]) / s
        z = (m[0][2] + m[2][0]) / s
    elif m[1][1] > m[2][2]:
        s = math.sqrt(1.0 + m[1][1] - m[0][0] - m[2][2]) * 2
        w = (m[0][2] - m[2][0]) / s
        x = (m[0][1] + m[1][0]) / s
        y = 0.25 * s
        z = (m[1][2] + m[2][1]) / s
    else:
        s = math.sqrt(1.0 + m[2][2] - m[0][0] - m[1][1]) * 2
        w = (m[1][0] - m[0][1]) / s
        x = (m[0][2] + m[2][0]) / s
        y = (m[1][2] + m[2][1]) / s
        z = 0.25 * s

    return _normalize_quaternion((x, y, z, w))


def _cross(a: Vec3, b: Vec3) -> Vec3:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _normalize_vec3(v: Vec3) -> Vec3:
    length = math.sqrt(sum(component * component for component in v))
    if length == 0.0:
        return v
    return (v[0] / length, v[1] / length, v[2] / length)


def quaternion_from_vectors(a: Vec3, b: Vec3) -> Quaternion:
    """Shortest-arc unit quaternion that rotates unit vector a onto unit
    vector b. Used for 3D-driven retargeting when depth-derived
    position_3d is available (see fk_solver.solve_direction_bone) — the
    2D-only path uses quaternion_from_axis_angle with axis_hint instead,
    since a 2D image-plane delta has no inherent rotation axis to derive
    from geometry alone."""
    dot = a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
    dot = max(-1.0, min(1.0, dot))

    if dot > 0.999999:
        return (0.0, 0.0, 0.0, 1.0)

    if dot < -0.999999:
        # a and b point in opposite directions: any axis perpendicular
        # to a works for a 180-degree rotation. Pick whichever world
        # axis is least aligned with a to avoid a near-zero cross product.
        fallback = (1.0, 0.0, 0.0) if abs(a[0]) < 0.9 else (0.0, 1.0, 0.0)
        axis = _normalize_vec3(_cross(a, fallback))
        return quaternion_from_axis_angle(axis, math.pi)

    cross = _cross(a, b)
    return _normalize_quaternion((cross[0], cross[1], cross[2], 1.0 + dot))


def rotate_vector_by_quaternion(v: Vec3, q: Quaternion) -> Vec3:
    qv: Quaternion = (v[0], v[1], v[2], 0.0)
    result = quaternion_multiply(quaternion_multiply(q, qv), quaternion_conjugate(q))
    return (result[0], result[1], result[2])


def apply_rest_pose_correction(
    delta_rotation: Quaternion, rest_local_matrix: Matrix4 | None
) -> Quaternion:
    """Reinterpret a rotation computed in world/image space as being in
    the bone's own local space, using its rest-pose orientation.

    fk_solver computes rotations in world/view axes (e.g. "rotate
    around +Z by this angle observed in the image"), but Blender
    applies a pose bone's ``rotation_quaternion`` relative to that
    bone's REST orientation in its own local space. Without this
    correction, a bone whose rest pose points sideways (its local axes
    not aligned with world axes) would visibly spin around the wrong
    axis. The conjugation below (rest^-1 * delta * rest) reinterprets
    the delta in the bone's local frame. It's still an approximation —
    the whole retargeting approach is 2D/image-driven, not true 3D
    reconstruction (section 7.4) — not a claim of physical correctness.
    """
    if rest_local_matrix is None:
        return delta_rotation
    rest_quat = quaternion_from_matrix(rest_local_matrix)
    return quaternion_multiply(
        quaternion_multiply(quaternion_conjugate(rest_quat), delta_rotation), rest_quat
    )


def apply_rest_pose_correction_to_vector(delta: Vec3, rest_local_matrix: Matrix4 | None) -> Vec3:
    """Same idea as apply_rest_pose_correction but for a translation
    offset: re-expresses a world/image-space delta in the bone's local
    axes using its rest-pose orientation."""
    if rest_local_matrix is None:
        return delta
    rest_quat = quaternion_from_matrix(rest_local_matrix)
    return rotate_vector_by_quaternion(delta, quaternion_conjugate(rest_quat))
