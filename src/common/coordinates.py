"""Canonical coordinate conversions shared by future 3D retarget stages.

The current MotionGraph's ``position_2d`` remains image-space data.  These
helpers establish the non-ambiguous bridge that 3D pose lifting and root
estimation must use before data reaches a rig solver.
"""

from __future__ import annotations

from common.types import Vec2, Vec3


def image_delta_to_camera(delta: Vec2, depth: float = 0.0) -> Vec3:
    """Map an image-plane delta into the canonical camera coordinate frame.

    Image coordinates are (+X right, +Y down).  Canonical camera coordinates
    are (+X right, +Y forward/away from the camera, +Z up), so an image-space
    downward move maps to negative camera Z. ``depth`` must use the same
    metric scale as X and Z; relative monocular depth is deliberately not
    accepted as a substitute by callers of the 3D retarget solver.
    """
    dx, dy = delta
    return (dx, depth, -dy)


def camera_to_character(point: Vec3, root_yaw_radians: float) -> Vec3:
    """Rotate a camera-space point about +Z into character/root space.

    Root yaw is intentionally an explicit input.  Treating the camera frame
    as character space is the old fixed-axis behaviour that breaks when the
    performer turns sideways.
    """
    import math

    x, y, z = point
    cosine = math.cos(root_yaw_radians)
    sine = math.sin(root_yaw_radians)
    return (cosine * x + sine * y, -sine * x + cosine * y, z)


def character_to_camera(point: Vec3, root_yaw_radians: float) -> Vec3:
    """Inverse of :func:`camera_to_character`."""
    import math

    x, y, z = point
    cosine = math.cos(root_yaw_radians)
    sine = math.sin(root_yaw_radians)
    return (cosine * x - sine * y, sine * x + cosine * y, z)
