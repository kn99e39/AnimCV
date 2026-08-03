"""Temporal knee/elbow bend-plane stabilization for fixed-length 3D targets."""

from __future__ import annotations

from dataclasses import replace
import math

from common.coordinates import character_to_camera
from pose.pose_lifter import LiftedPoseFrame, LiftedPoseSequence
from pose.root_motion import RootMotionSequence


_CHAINS = (
    ("left_hip", "left_knee", "left_ankle"),
    ("right_hip", "right_knee", "right_ankle"),
    ("left_shoulder", "left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow", "right_wrist"),
)


def stabilize_bend_planes(
    lifted: LiftedPoseSequence, root_motion: RootMotionSequence, min_bend_degrees: float = 12.0
) -> LiftedPoseSequence:
    """Keep each limb's non-ambiguous bend on a temporally consistent side.

    A straight limb has no meaningful pole direction. Frames below
    ``min_bend_degrees`` retain their observed position and do not update the
    limb state. For a later mirror flip, lower-segment direction is reflected
    across the plane spanned by the upper limb and character hinge axis +X.
    """
    if len(lifted.frames) != len(root_motion.frames):
        raise ValueError("lifted pose and root motion must have matching frame counts")
    states: dict[tuple[str, str, str], int] = {}
    output = []
    for pose_frame, root_frame in zip(lifted.frames, root_motion.frames):
        character = dict(root_frame.character_points)
        for chain in _CHAINS:
            root, mid, end = chain
            upper = _unit(_subtract(character[mid], character[root]))
            lower = _unit(_subtract(character[end], character[mid]))
            bend = math.degrees(math.acos(max(-1.0, min(1.0, _dot(upper, lower)))))
            signed = _dot(_cross(upper, lower), (1.0, 0.0, 0.0))
            if bend < min_bend_degrees or abs(signed) < 1e-6:
                continue
            current = 1 if signed > 0 else -1
            previous = states.get(chain)
            if previous is None:
                states[chain] = current
                continue
            if current != previous:
                plane_normal = _unit(_cross((1.0, 0.0, 0.0), upper))
                reflected = _subtract(lower, _scale(plane_normal, 2.0 * _dot(lower, plane_normal)))
                lower_length = _length(character[end], character[mid])
                character[end] = _add(character[mid], _scale(reflected, lower_length))

        points = {
            name: replace(point, position=character_to_camera(character[name], root_frame.root_yaw_radians))
            for name, point in pose_frame.points.items()
        }
        output.append(LiftedPoseFrame(pose_frame.frame_index, pose_frame.timestamp, points))
    return LiftedPoseSequence(
        frames=output, source_fps=lifted.source_fps, coordinate_frame=lifted.coordinate_frame,
        units=lifted.units, backend=f"{lifted.backend}+bend_plane_stabilized",
    )


def _subtract(a, b): return tuple(x - y for x, y in zip(a, b))
def _add(a, b): return tuple(x + y for x, y in zip(a, b))
def _scale(a, factor): return tuple(x * factor for x in a)
def _dot(a, b): return sum(x * y for x, y in zip(a, b))
def _cross(a, b): return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])
def _length(a, b): return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))
def _unit(a):
    length = _length(a, (0.0, 0.0, 0.0))
    if length < 1e-6: raise ValueError("coincident joints cannot define a bend plane")
    return _scale(a, 1.0 / length)
