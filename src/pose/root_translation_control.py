"""Root Translation control experiment (sections 8 & 9 of docs/51).

Tests one narrow geometric question, not a production solver: *if* a foot is
truly planted in the world, does enforcing its world-space stationarity let
a monocular pipeline recover the frame-to-frame root displacement — and does
that recovery survive when the camera itself moves?

Two conditions are compared per frame where a foot is planted (by the
KINEMATIC_PROXY reference in contact_reference.py — see its caveats):

  world_oracle        Uses the GT world-frame pelvis-relative ankle position.
                       Immune to camera motion by construction (it IS the
                       ground-truth world geometry), so its error is a sanity
                       check that the identity itself is sound — a tautology
                       up to floating point, not a claim about what AnimCV
                       can currently observe.

  naive_camera_frame   Applies the exact same identity to the per-frame
                       camera-relative pelvis-relative ankle position — what
                       a real monocular Frame Pose output looks like today.
                       This is contaminated whenever the camera itself
                       rotates between frames, because Frame Pose's
                       coordinate frame is anchored to the camera, not to the
                       body or the world.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math

import numpy as np

from pose.contact import ContactState
from pose.oracle_world_reference import WorldReferenceFrame


class CameraMotionCondition(Enum):
    STATIC = "static"
    MOVING = "moving"


@dataclass(frozen=True)
class TranslationRecoveryFrame:
    frame_index: int
    foot: str
    resolved: bool  # False -> UNKNOWN: insufficient evidence this frame
    world_true_delta_m: float | None = None
    world_oracle_error_m: float | None = None
    naive_camera_error_m: float | None = None
    camera_rotation_delta_degrees: float | None = None


def rotation_angle_degrees(rotation_a: np.ndarray, rotation_b: np.ndarray) -> float:
    relative = rotation_a @ rotation_b.T
    cosine = np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
    return math.degrees(math.acos(cosine))


def run_control_experiment(
    world_frames: list[WorldReferenceFrame],
    camera_relative_ankle: list[tuple[float, float, float] | None],
    foot: str,
    planted_reference: list[ContactState],
) -> list[TranslationRecoveryFrame]:
    """Evaluate root-displacement recoverability at each planted-foot frame.

    ``camera_relative_ankle`` and ``planted_reference`` must be aligned
    frame-for-frame with ``world_frames`` (same length, same indexing).
    """
    if not (len(world_frames) == len(camera_relative_ankle) == len(planted_reference)):
        raise ValueError("world_frames, camera_relative_ankle and planted_reference must be the same length")

    ankle_attr = f"{foot}_ankle_world"
    results = []
    for index in range(1, len(world_frames)):
        previous, current = world_frames[index - 1], world_frames[index]
        planted = (
            planted_reference[index] is ContactState.CONTACT
            and planted_reference[index - 1] is ContactState.CONTACT
        )
        if not planted or not previous.valid or not current.valid:
            results.append(TranslationRecoveryFrame(index, foot, resolved=False))
            continue

        pelvis_prev = np.asarray(previous.pelvis_world)
        pelvis_curr = np.asarray(current.pelvis_world)
        world_ankle_prev = np.asarray(getattr(previous, ankle_attr))
        world_ankle_curr = np.asarray(getattr(current, ankle_attr))
        true_root_delta = pelvis_curr - pelvis_prev

        world_rel_prev = world_ankle_prev - pelvis_prev
        world_rel_curr = world_ankle_curr - pelvis_curr
        world_oracle_root_delta = -(world_rel_curr - world_rel_prev)
        world_oracle_error = float(np.linalg.norm(world_oracle_root_delta - true_root_delta))

        naive_error = None
        camera_prev, camera_curr = camera_relative_ankle[index - 1], camera_relative_ankle[index]
        if camera_prev is not None and camera_curr is not None:
            naive_root_delta = -(np.asarray(camera_curr) - np.asarray(camera_prev))
            naive_error = float(np.linalg.norm(naive_root_delta - true_root_delta))

        rotation_delta = None
        if previous.camera_rotation is not None and current.camera_rotation is not None:
            rotation_delta = rotation_angle_degrees(current.camera_rotation, previous.camera_rotation)

        results.append(TranslationRecoveryFrame(
            frame_index=index, foot=foot, resolved=True,
            world_true_delta_m=float(np.linalg.norm(true_root_delta)),
            world_oracle_error_m=world_oracle_error,
            naive_camera_error_m=naive_error,
            camera_rotation_delta_degrees=rotation_delta,
        ))
    return results


def summarize(results: list[TranslationRecoveryFrame]) -> dict:
    resolved = [r for r in results if r.resolved]
    unresolved_count = len(results) - len(resolved)
    world_errors = [r.world_oracle_error_m for r in resolved if r.world_oracle_error_m is not None]
    naive_errors = [r.naive_camera_error_m for r in resolved if r.naive_camera_error_m is not None]
    rotations = [r.camera_rotation_delta_degrees for r in resolved if r.camera_rotation_delta_degrees is not None]
    # Correlate naive-condition error with camera rotation magnitude, when both exist.
    paired = [
        (r.naive_camera_error_m, r.camera_rotation_delta_degrees)
        for r in resolved
        if r.naive_camera_error_m is not None and r.camera_rotation_delta_degrees is not None
    ]
    correlation = None
    if len(paired) >= 2:
        errors, deltas = zip(*paired)
        errors_arr, deltas_arr = np.asarray(errors), np.asarray(deltas)
        if errors_arr.std() > 1e-9 and deltas_arr.std() > 1e-9:
            correlation = float(np.corrcoef(errors_arr, deltas_arr)[0, 1])
    return {
        "frame_count": len(results),
        "recoverable_segment_count": len(resolved),
        "unresolved_segment_count": unresolved_count,
        "world_oracle_mean_error_m": float(np.mean(world_errors)) if world_errors else None,
        "world_oracle_max_error_m": float(np.max(world_errors)) if world_errors else None,
        "naive_camera_mean_error_m": float(np.mean(naive_errors)) if naive_errors else None,
        "naive_camera_max_error_m": float(np.max(naive_errors)) if naive_errors else None,
        "mean_camera_rotation_delta_degrees": float(np.mean(rotations)) if rotations else None,
        "naive_error_vs_camera_rotation_correlation": correlation,
    }
