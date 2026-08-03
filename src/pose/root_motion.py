"""Root/body orientation estimation for temporally lifted 3D poses.

VideoPose3D outputs pelvis-relative camera-space joints.  This module derives
the missing body yaw from bilateral hips and shoulders, then rotates every
joint into character space.  It deliberately does not invent global root
translation: monocular, root-relative lifting cannot observe it reliably.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import statistics
from pathlib import Path
from typing import Any

from common.coordinates import camera_to_character
from common.serialization import read_json, write_json
from common.types import Vec3
from pose.pose_lifter import LiftedPoseSequence, load_lifted_pose_sequence


@dataclass(frozen=True)
class RootMotionFrame:
    frame_index: int
    timestamp: float
    root_yaw_radians: float
    forward: Vec3
    right: Vec3
    confidence: float
    # Intentionally None until foot-contact/root-motion estimation.  This is
    # more honest than treating the pelvis-relative model output as a global
    # translation.
    root_translation: Vec3 | None
    character_points: dict[str, Vec3]
    yaw_sources: dict[str, float] = field(default_factory=dict)
    yaw_agreement_degrees: float = 0.0
    yaw_held: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "timestamp": self.timestamp,
            "root_yaw_radians": self.root_yaw_radians,
            "forward": list(self.forward),
            "right": list(self.right),
            "confidence": self.confidence,
            "root_translation": list(self.root_translation) if self.root_translation else None,
            "character_points": {name: list(value) for name, value in self.character_points.items()},
            "yaw_sources": self.yaw_sources,
            "yaw_agreement_degrees": self.yaw_agreement_degrees,
            "yaw_held": self.yaw_held,
        }


@dataclass(frozen=True)
class RootMotionSequence:
    frames: list[RootMotionFrame] = field(default_factory=list)
    source_fps: float = 0.0
    coordinate_frame: str = "character_root_relative"
    translation_observable: bool = False
    observation_confidence_threshold: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_fps": self.source_fps,
            "coordinate_frame": self.coordinate_frame,
            "translation_observable": self.translation_observable,
            "observation_confidence_threshold": self.observation_confidence_threshold,
            "frames": [frame.to_dict() for frame in self.frames],
        }


def save_root_motion_sequence(sequence: RootMotionSequence, path: str | Path) -> None:
    write_json(path, sequence.to_dict())


def estimate_root_motion(
    lifted: LiftedPoseSequence,
    smoothing_window: int = 5,
    max_yaw_step_degrees: float = 20.0,
) -> RootMotionSequence:
    if smoothing_window < 1 or smoothing_window % 2 == 0:
        raise ValueError("smoothing_window must be a positive odd integer")
    if max_yaw_step_degrees <= 0:
        raise ValueError("max_yaw_step_degrees must be positive")
    observations = [_fused_yaw_observation(frame) for frame in lifted.frames]
    raw = [observation[0] for observation in observations]
    unwrapped = _unwrap_angles(raw)
    smoothed = _median_smooth(unwrapped, smoothing_window)
    stabilized, accepted = _limit_yaw_velocity(smoothed, math.radians(max_yaw_step_degrees))
    frames = []
    for frame, yaw, is_accepted, observation in zip(lifted.frames, stabilized, accepted, observations):
        right = (math.cos(yaw), math.sin(yaw), 0.0)
        forward = (-math.sin(yaw), math.cos(yaw), 0.0)
        _, source_confidence, source_weights, agreement = observation
        confidence = source_confidence * (1.0 if is_accepted else 0.25)
        points = {
            name: camera_to_character(point.position, yaw)
            for name, point in frame.points.items()
        }
        frames.append(RootMotionFrame(
            frame_index=frame.frame_index,
            timestamp=frame.timestamp,
            root_yaw_radians=yaw,
            forward=forward,
            right=right,
            confidence=confidence,
            root_translation=None,
            character_points=points,
            yaw_sources=source_weights,
            yaw_agreement_degrees=agreement,
            yaw_held=not is_accepted,
        ))
    return RootMotionSequence(
        frames=frames, source_fps=lifted.source_fps,
        observation_confidence_threshold=lifted.observation_confidence_threshold,
    )


def load_root_motion_sequence(path: str | Path) -> RootMotionSequence:
    data = read_json(path)
    return RootMotionSequence(
        source_fps=data["source_fps"],
        coordinate_frame=data["coordinate_frame"],
        translation_observable=data["translation_observable"],
        observation_confidence_threshold=data.get("observation_confidence_threshold"),
        frames=[
            RootMotionFrame(
                frame_index=frame["frame_index"], timestamp=frame["timestamp"],
                root_yaw_radians=frame["root_yaw_radians"], forward=tuple(frame["forward"]),
                right=tuple(frame["right"]), confidence=frame["confidence"],
                root_translation=(tuple(frame["root_translation"])
                                  if frame["root_translation"] is not None else None),
                character_points={name: tuple(point) for name, point in frame["character_points"].items()},
                yaw_sources={name: float(weight) for name, weight in frame.get("yaw_sources", {}).items()},
                yaw_agreement_degrees=float(frame.get("yaw_agreement_degrees", 0.0)),
                yaw_held=bool(frame.get("yaw_held", False)),
            ) for frame in data["frames"]
        ],
    )


def _fused_yaw_observation(frame) -> tuple[float, float, dict[str, float], float]:
    pairs = (("left_shoulder", "right_shoulder"), ("left_hip", "right_hip"))
    vectors = []
    weights: dict[str, float] = {}
    for label, (left, right) in zip(("shoulders", "hips"), pairs):
        if left not in frame.points or right not in frame.points:
            continue
        a, b = frame.points[left], frame.points[right]
        dx = b.position[0] - a.position[0]
        dy = b.position[1] - a.position[1]
        length = math.hypot(dx, dy)
        if length > 1e-6:
            observation_confidence = (a.confidence + b.confidence) / 2.0
            validity = float(a.observation_valid and b.observation_valid)
            weight = length * observation_confidence * validity
            if weight > 1e-6:
                vectors.append((math.atan2(dy, dx), weight, observation_confidence))
                weights[label] = weight
    if not vectors:
        raise ValueError(f"frame {frame.frame_index} lacks bilateral torso points for root yaw")
    x = sum(math.cos(angle) * weight for angle, weight, _ in vectors)
    y = sum(math.sin(angle) * weight for angle, weight, _ in vectors)
    if math.hypot(x, y) < 1e-6:
        raise ValueError(f"frame {frame.frame_index} has ambiguous bilateral torso direction")
    yaw = math.atan2(y, x)
    total_weight = sum(weight for _, weight, _ in vectors)
    confidence = sum(weight * item_confidence for _, weight, item_confidence in vectors) / total_weight
    agreement = max(
        (abs(_angle_delta(first[0], second[0])) * 180.0 / math.pi for first in vectors for second in vectors),
        default=0.0,
    )
    # Shoulder/hip disagreement is a useful R5 uncertainty signal, but it is
    # not itself a detector failure: walking and torso twist routinely create
    # it. Keep detector reliability as the root-yaw confidence and expose the
    # disagreement separately instead of incorrectly classifying valid motion
    # as low-confidence yaw.
    return yaw, confidence, weights, agreement


def _unwrap_angles(angles: list[float]) -> list[float]:
    if not angles:
        return []
    output = [angles[0]]
    for angle in angles[1:]:
        delta = _angle_delta(angle, output[-1])
        output.append(output[-1] + delta)
    return output


def _angle_delta(a: float, b: float) -> float:
    return (a - b + math.pi) % (2 * math.pi) - math.pi


def _median_smooth(values: list[float], window: int) -> list[float]:
    radius = window // 2
    return [statistics.median(values[max(0, i - radius):i + radius + 1]) for i in range(len(values))]


def _limit_yaw_velocity(values: list[float], max_step: float) -> tuple[list[float], list[bool]]:
    """Hold orientation outliers instead of treating a 3D-lifter flip as turn.

    A sudden 180-degree bilateral-axis reversal is a common monocular lifting
    ambiguity, not a physically plausible one-frame root turn at 50 fps.
    """
    if not values:
        return [], []
    output = [values[0]]
    accepted = [True]
    for value in values[1:]:
        if abs(value - output[-1]) > max_step:
            output.append(output[-1])
            accepted.append(False)
        else:
            output.append(value)
            accepted.append(True)
    return output, accepted
