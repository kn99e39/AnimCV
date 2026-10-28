"""Keyframe importance scoring (Architecture_v2.md section 8.2).

The doc's example formula references signals this project doesn't have
yet (rotation_error / endpoint_error need a reconstruction reference
this MVP's 2D-direction solver doesn't produce, and visibility_change
needs per-sample visibility that BoneTransformSample doesn't carry —
only confidence). This substitutes the signals that ARE available from
Milestone 5's output: angular velocity, angular acceleration, a
confidence delta (as a proxy for tracking-quality change), and local
extrema detection, plus unconditional 1.0 for locked/boundary frames.
"""

from __future__ import annotations

import math

import numpy as np

from common.types import Quaternion
from retarget.solver import AnimationTrack

MIN_IMPORTANCE_TO_KEEP = 0.75


def quaternion_angle(a: Quaternion, b: Quaternion) -> float:
    """Angular distance (radians) between two unit quaternions.

    Uses abs(dot) because q and -q represent the same rotation
    (quaternion double cover) — without it, sign flips between frames
    would register as a near-180-degree jump that isn't really there.
    """
    dot = sum(x * y for x, y in zip(a, b))
    dot = max(-1.0, min(1.0, abs(dot)))
    return 2.0 * math.acos(dot)


def score_track_importance(
    track: AnimationTrack, locked_frames: set[int] | None = None
) -> list[float]:
    """One importance score in [0, 1] per sample in track.samples, aligned by index."""
    locked_frames = locked_frames or set()
    samples = track.samples
    n = len(samples)
    if n == 0:
        return []

    # vectorised over samples (same formula as the per-sample definition)
    q = np.array([sample.rotation for sample in samples], float)
    c = np.array([sample.confidence for sample in samples], float)
    velocities = np.zeros(n)
    if n > 1:
        velocities[1:] = 2.0 * np.arccos(np.clip(np.abs(np.sum(q[1:] * q[:-1], axis=1)), -1.0, 1.0))
    accelerations = np.zeros(n)
    accelerations[1:] = np.abs(np.diff(velocities))
    confidence_deltas = np.zeros(n)
    confidence_deltas[1:] = np.abs(np.diff(c))

    max_velocity = float(velocities.max()) or 1.0
    max_acceleration = float(accelerations.max()) or 1.0

    scores = np.ones(n)
    if n > 2:
        v_prev, v, v_next = velocities[:-2], velocities[1:-1], velocities[2:]
        extremum = ((v_prev < v) & (v > v_next)) | ((v_prev > v) & (v < v_next))
        inner = (0.35 * (v / max_velocity) + 0.30 * (accelerations[1:-1] / max_acceleration)
                 + 0.15 * confidence_deltas[1:-1] + 0.20 * extremum)
        scores[1:-1] = np.minimum(1.0, inner)
    if locked_frames:
        scores[[i for i, sample in enumerate(samples) if sample.frame_index in locked_frames]] = 1.0
    return scores.tolist()
