"""Temporal 2D-to-3D pose lifting via MMPose VideoPose3D.

This module deliberately produces a representation separate from
``MotionGraph.position_3d``.  The latter may contain sampled *relative* depth
with image-pixel X/Y and is not usable as a 3D rig target.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from common.serialization import read_json, write_json
from common.types import Vec3
from pose.pose_types import PoseFrame, PoseSequence

# H36M order emitted by MMPose's VideoPose3D codec.
H36M_NAMES = (
    "pelvis", "left_hip", "left_knee", "left_ankle", "right_hip",
    "right_knee", "right_ankle", "spine", "thorax", "neck", "head",
    "left_shoulder", "left_elbow", "left_wrist", "right_shoulder",
    "right_elbow", "right_wrist",
)


@dataclass(frozen=True)
class LiftedPosePoint:
    name: str
    position: Vec3
    confidence: float
    depth_uncertainty: float
    observation_valid: bool = True
    interpolated_input: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "position": list(self.position),
            "confidence": self.confidence,
            "depth_uncertainty": self.depth_uncertainty,
            "observation_valid": self.observation_valid,
            "interpolated_input": self.interpolated_input,
        }


@dataclass(frozen=True)
class LiftedPoseFrame:
    frame_index: int
    timestamp: float
    points: dict[str, LiftedPosePoint]

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "timestamp": self.timestamp,
            "points": {name: point.to_dict() for name, point in self.points.items()},
        }


@dataclass(frozen=True)
class LiftedPoseSequence:
    frames: list[LiftedPoseFrame] = field(default_factory=list)
    source_fps: float = 0.0
    coordinate_frame: str = "camera_root_relative"
    units: str = "metres"
    backend: str = "videopose3d_h36m_81f"
    observation_confidence_threshold: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_fps": self.source_fps,
            "coordinate_frame": self.coordinate_frame,
            "units": self.units,
            "backend": self.backend,
            "observation_confidence_threshold": self.observation_confidence_threshold,
            "frames": [frame.to_dict() for frame in self.frames],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LiftedPoseSequence":
        return cls(
            source_fps=data["source_fps"],
            coordinate_frame=data["coordinate_frame"],
            units=data["units"],
            backend=data["backend"],
            observation_confidence_threshold=data.get("observation_confidence_threshold"),
            frames=[
                LiftedPoseFrame(
                    frame_index=frame["frame_index"],
                    timestamp=frame["timestamp"],
                    points={
                        name: LiftedPosePoint(
                            name=point["name"],
                            position=tuple(point["position"]),
                            confidence=point["confidence"],
                            depth_uncertainty=point["depth_uncertainty"],
                            observation_valid=point.get("observation_valid", True),
                            interpolated_input=point.get("interpolated_input", False),
                        )
                        for name, point in frame["points"].items()
                    },
                )
                for frame in data["frames"]
            ],
        )


def save_lifted_pose_sequence(sequence: LiftedPoseSequence, path: str | Path) -> None:
    write_json(path, sequence.to_dict())


def load_lifted_pose_sequence(path: str | Path) -> LiftedPoseSequence:
    return LiftedPoseSequence.from_dict(read_json(path))


@dataclass(frozen=True)
class VideoPose3DConfig:
    checkpoint_path: str
    config_path: str | None = None
    device: str = "cpu"
    sequence_length: int = 81
    min_observation_confidence: float = 0.3
    max_interpolation_gap: int = 5


class VideoPose3DLifter:
    """Lift a tracked canonical PoseSequence with a non-causal temporal model."""

    def __init__(self, config: VideoPose3DConfig):
        if config.sequence_length != 81:
            raise ValueError("the bundled VideoPose3D backend requires sequence_length=81")
        if not 0.0 <= config.min_observation_confidence <= 1.0:
            raise ValueError("min_observation_confidence must be in [0, 1]")
        if config.max_interpolation_gap < 0:
            raise ValueError("max_interpolation_gap must be non-negative")
        self.config = config
        self._model = None

    def lift(self, poses: PoseSequence, image_size: tuple[int, int]) -> LiftedPoseSequence:
        if not poses.frames:
            return LiftedPoseSequence(source_fps=poses.source_fps)
        pose_results, observations = _prepare_h36m_observations(
            poses, self.config.min_observation_confidence, self.config.max_interpolation_gap
        )
        frames: list[LiftedPoseFrame] = []
        for index, source_frame in enumerate(poses.frames):
            predicted = self._predict_window(pose_results, index, image_size)
            points = _to_lifted_points(predicted, source_frame, observations[index])
            frames.append(LiftedPoseFrame(source_frame.frame_index, source_frame.timestamp, points))
        return LiftedPoseSequence(
            frames=frames, source_fps=poses.source_fps,
            observation_confidence_threshold=self.config.min_observation_confidence,
        )

    def _predict_window(self, pose_results, frame_index: int, image_size: tuple[int, int]) -> np.ndarray:
        """Return the target H36M frame as a (17, 3) camera-space array.

        Kept as a narrow method so unit tests can exercise schema/coordinate
        conversion without importing MMPose or model weights.
        """
        try:
            from mmpose.apis import extract_pose_sequence, inference_pose_lifter_model, init_model
        except ImportError as exc:
            raise ImportError("VideoPose3D requires the optional mmpose dependencies") from exc

        if self._model is None:
            config_path = self.config.config_path or _default_config_path()
            self._model = init_model(config_path, self.config.checkpoint_path, device=self.config.device)
        mmpose_results = [_to_mmpose_sample(keypoints, image_size) for keypoints in pose_results]
        window = extract_pose_sequence(
            mmpose_results, frame_index, causal=False, seq_len=self.config.sequence_length
        )
        result = inference_pose_lifter_model(
            self._model, window, image_size=image_size, norm_pose_2d=True
        )
        if not result:
            raise RuntimeError(f"VideoPose3D returned no prediction at frame {frame_index}")
        raw = np.asarray(result[0].pred_instances.keypoints)
        return np.squeeze(raw, axis=0)


def _default_config_path() -> str:
    import mmpose

    return str(
        Path(mmpose.__file__).resolve().parent
        / ".mim/configs/body_3d_keypoint/video_pose_lift/h36m/"
        "video-pose-lift_tcn-81frm-supv_8xb128-160e_h36m.py"
    )


_H36M_SOURCE_NAMES = (
    "pelvis", "left_hip", "left_knee", "left_ankle", "right_hip",
    "right_knee", "right_ankle", "spine", "neck", "neck", "head",
    "left_shoulder", "left_elbow", "left_wrist", "right_shoulder",
    "right_elbow", "right_wrist",
)


def _to_h36m_keypoints(frame: PoseFrame) -> np.ndarray:
    """Convert canonical landmarks to the H36M 2D layout used by VideoPose3D."""
    landmarks = frame.landmarks
    def xy(name: str) -> tuple[float, float]:
        point = landmarks[name]
        return (point.x, point.y)

    return np.array([xy(name) for name in _H36M_SOURCE_NAMES], dtype=np.float32)


def _prepare_h36m_observations(
    poses: PoseSequence, min_confidence: float, max_gap: int
) -> tuple[list[np.ndarray], list[dict[str, tuple[bool, bool]]]]:
    """Prepare MMPose input without hiding low-confidence observations.

    Invalid observations are used only as short-gap, linearly interpolated
    model input. The accompanying flags remain false/true respectively in the
    lifted artifact, so later quality gates can reject them.
    """
    raw = np.stack([_to_h36m_keypoints(frame) for frame in poses.frames])
    valid = np.array([
        [frame.landmarks[name].visible and frame.landmarks[name].confidence >= min_confidence
         for name in _H36M_SOURCE_NAMES]
        for frame in poses.frames
    ], dtype=bool)
    filled = raw.copy()
    interpolated = np.zeros(valid.shape, dtype=bool)
    for joint in range(raw.shape[1]):
        index = 0
        while index < len(poses.frames):
            if valid[index, joint]:
                index += 1
                continue
            end = index
            while end < len(poses.frames) and not valid[end, joint]:
                end += 1
            gap = end - index
            if 0 < index and end < len(poses.frames) and gap <= max_gap:
                start_value, end_value = raw[index - 1, joint], raw[end, joint]
                for offset in range(gap):
                    filled[index + offset, joint] = start_value + (end_value - start_value) * ((offset + 1) / (gap + 1))
                    interpolated[index + offset, joint] = True
            else:
                # At a sequence boundary there is no pair to interpolate.
                # Hold the nearest valid observation for model context only;
                # the output stays observation_valid=False and is rejected by
                # downstream quality gates. A zero sentinel is much worse: it
                # contaminates a temporal model's surrounding valid frames.
                if 0 < index and end < len(poses.frames):
                    for offset in range(gap):
                        filled[index + offset, joint] = (
                            raw[index - 1, joint] if offset < gap / 2 else raw[end, joint]
                        )
                        interpolated[index + offset, joint] = True
                else:
                    nearest = raw[end, joint] if end < len(poses.frames) else raw[index - 1, joint]
                    filled[index:end, joint] = nearest
                    interpolated[index:end, joint] = True
            index = end
    flags = [
        {name: (bool(valid[i, joint]), bool(interpolated[i, joint]))
         for joint, name in enumerate(H36M_NAMES)}
        for i in range(len(poses.frames))
    ]
    return list(filled), flags


def _to_mmpose_sample(keypoints: np.ndarray, image_size: tuple[int, int]):
    """Wrap one H36M 2D layout in MMPose's inference data structure."""
    try:
        from mmengine.structures import InstanceData
        from mmpose.structures import PoseDataSample
    except ImportError as exc:
        raise ImportError("VideoPose3D requires the optional mmpose dependencies") from exc

    keypoints = np.asarray(keypoints, dtype=np.float32)[None, ...]
    width, height = image_size
    sample = PoseDataSample()
    sample.pred_instances = InstanceData(
        keypoints=keypoints,
        bboxes=np.array([[0.0, 0.0, float(width), float(height)]], dtype=np.float32),
    )
    sample.gt_instances = InstanceData()
    sample.track_id = 0
    return [sample]


def _to_lifted_points(
    predicted: np.ndarray, source: PoseFrame, observations: dict[str, tuple[bool, bool]] | None = None
) -> dict[str, LiftedPosePoint]:
    if predicted.shape != (17, 3):
        raise ValueError(f"expected VideoPose3D output shape (17, 3), got {predicted.shape}")
    # MMPose's documented visualization transform: H36M (x, y, z) is
    # rearranged to (x, z, y) with X/Z signs flipped, yielding our canonical
    # camera frame (+X right, +Y forward, +Z up). Root remains pelvis-relative.
    converted = predicted[:, [0, 2, 1]].copy()
    converted[:, 0] *= -1.0
    converted[:, 2] *= -1.0
    result = {}
    for index, name in enumerate(H36M_NAMES):
        source_name = "neck" if name == "thorax" else name
        landmark = source.landmarks.get(source_name)
        confidence = landmark.confidence if landmark is not None else 0.0
        observation_valid, interpolated_input = (observations or {}).get(name, (True, False))
        # VideoPose3D has no calibrated posterior variance. Keep an explicit,
        # conservative input-confidence proxy rather than pretending it is a
        # metric uncertainty estimate.
        result[name] = LiftedPosePoint(
            name=name,
            position=tuple(float(value) for value in converted[index]),
            confidence=confidence,
            depth_uncertainty=1.0 - confidence,
            observation_valid=observation_valid,
            interpolated_input=interpolated_input,
        )
    return result
