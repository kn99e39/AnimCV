"""EXPERIMENTAL (Worklog 93): camera-relative anatomical torso frames for rig replay.

Not a production default; no existing RootOrientation / AnimationSemantics schema is changed.

Frames (right-handed, columns = body axes expressed in AnimCV camera coordinates X right, Y forward depth, Z up):
  spine axis  u = normalize(thorax - pelvis)                          (body +Z, "up" along the trunk)
  thorax frame R_t: x_t = normalize(perp_u(right_shoulder - left_shoulder)), z_t = u, y_t = z_t x x_t
  pelvis frame R_p: x_p = normalize(perp_u(right_hip - left_hip)),           z_p = u, y_p = z_p x x_p
  (y = body forward / chest normal; for an upright torso R = Rz(psi) exactly as the yaw-only contract, with
  psi = atan2 of the right-minus-left axis, facing the camera = 180 deg).
These are CAMERA-relative anatomical frames, not gravity-aligned world orientations: a tilted camera tilts them.

Validity: the trunk length must be > ``MIN_TRUNK`` (in the input's units) and the bilateral axis must keep at least
sin ``MIN_SIDE_ANGLE_DEG`` of its length perpendicular to the spine; otherwise the frame is invalid (None) and the
caller decides the fallback (Worklog 93 rig replay: the W91 yaw-only frame, flagged).

Rig ownership used by the Worklog 93 replay (BaseRig has no root/pelvis bone; ``spine_01`` is top-level and the arms
hang from spine_05 -> clavicle):
  Armature Object rotation  = A R_p A^-1        (whole-body orientation, legs included)
  spine_01 local rotation   = conjugate of A (R_p^T R_t) A^-1 by the spine_01 rest basis (trunk twist above the pelvis)
  arm local rotations       = Worklog 61/76 IK on the arm chain expressed in the thorax frame: A R_t^T v
so the world-frame arm direction is unchanged (A R_p (R_p^T R_t) R_t^T v = A v), as with the yaw-only contract.
"""

from __future__ import annotations

import math

import numpy as np

MIN_TRUNK = 1e-3
MIN_SIDE_ANGLE_DEG = 20.0
PELVIS, L_HIP, R_HIP, THORAX, L_SHOULDER, R_SHOULDER = 0, 1, 4, 8, 11, 14


def _frame(u, side):
    s = np.asarray(side, float)
    perp = s - np.dot(s, u) * u
    n = float(np.linalg.norm(perp))
    if n < math.sin(math.radians(MIN_SIDE_ANGLE_DEG)) * max(float(np.linalg.norm(s)), 1e-12) or n < 1e-12:
        return None
    x = perp / n
    y = np.cross(u, x)
    return np.stack([x, y, u], axis=1)


def torso_frames(joints):
    """(R_thorax, R_pelvis) from (17, 3) H36M-order joints; either may be None when degenerate."""
    j = np.asarray(joints, float)
    if not np.isfinite(j[[PELVIS, L_HIP, R_HIP, THORAX, L_SHOULDER, R_SHOULDER]]).all():
        return None, None
    trunk = j[THORAX] - j[PELVIS]
    length = float(np.linalg.norm(trunk))
    if length < MIN_TRUNK:
        return None, None
    u = trunk / length
    return _frame(u, j[R_SHOULDER] - j[L_SHOULDER]), _frame(u, j[R_HIP] - j[L_HIP])


def yaw_frame(yaw):
    """The yaw-only contract as a frame: Rz(yaw)."""
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def compat_yaw(R):
    """Yaw of the frame's right axis projected on the camera horizontal plane (None if vertical)."""
    x = R[:, 0]
    return math.atan2(x[1], x[0]) if math.hypot(x[0], x[1]) > 1e-9 else None


def inclination_degrees(R):
    return math.degrees(math.acos(max(-1.0, min(1.0, float(R[2, 2])))))


def geodesic_degrees(Ra, Rb):
    c = (np.trace(Ra.T @ Rb) - 1.0) / 2.0
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def axis_angle_degrees(a, b):
    return math.degrees(math.acos(max(-1.0, min(1.0, float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))))))


def matrix_to_quaternion_xyzw(R):
    """Unit quaternion (x, y, z, w) of a rotation matrix (Shepperd)."""
    m = np.asarray(R, float)
    t = np.trace(m)
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        q = ((m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s, 0.25 * s)
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        q = (0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s, (m[2, 1] - m[1, 2]) / s)
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        q = ((m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s, (m[0, 2] - m[2, 0]) / s)
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        q = ((m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s, (m[1, 0] - m[0, 1]) / s)
    q = np.asarray(q, float)
    return tuple(q / np.linalg.norm(q))


def quaternion_to_matrix(q):
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
