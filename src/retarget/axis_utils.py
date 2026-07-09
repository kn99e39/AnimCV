"""Axis correction utilities for the retarget solver (Architecture_v2.md section 7).

Keeps axis-vector/quaternion math out of fk_solver.py so hint parsing
and axis-angle rotation construction are testable and reusable in
isolation, and so the solver never hardcodes an assumption like "Y axis
points along the bone" — everything axis-related funnels through here.
"""

from __future__ import annotations

import math

from common.types import Quaternion, Vec3

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
