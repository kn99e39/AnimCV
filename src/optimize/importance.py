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

    velocities = [0.0] * n
    for i in range(1, n):
        velocities[i] = quaternion_angle(samples[i - 1].rotation, samples[i].rotation)

    accelerations = [0.0] * n
    for i in range(1, n):
        accelerations[i] = abs(velocities[i] - velocities[i - 1])

    confidence_deltas = [0.0] * n
    for i in range(1, n):
        confidence_deltas[i] = abs(samples[i].confidence - samples[i - 1].confidence)

    max_velocity = max(velocities) or 1.0
    max_acceleration = max(accelerations) or 1.0

    scores: list[float] = []
    for i, sample in enumerate(samples):
        if i == 0 or i == n - 1 or sample.frame_index in locked_frames:
            scores.append(1.0)
            continue

        is_local_extremum = (
            velocities[i - 1] < velocities[i] > velocities[i + 1]
            or velocities[i - 1] > velocities[i] < velocities[i + 1]
        )

        score = (
            0.35 * (velocities[i] / max_velocity)
            + 0.30 * (accelerations[i] / max_acceleration)
            + 0.15 * confidence_deltas[i]
            + 0.20 * (1.0 if is_local_extremum else 0.0)
        )
        scores.append(min(1.0, score))

    return scores
