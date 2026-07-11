"""MMPose backend adapter (Architecture_v2.md section 3.2).

Project code must not depend on MMPose internals directly; only this
module imports ``mmpose``, and the import happens lazily inside
``_load_model``/``process_frame`` so the rest of the project can run
without MMPose installed (Milestone 1 acceptance criteria).

MMPose's default top-down models emit the 17 COCO keypoints, which do not
include pelvis/spine/neck/head directly. Those four canonical landmarks
are derived as midpoints (see ``_extract_canonical_landmarks``) so that
downstream code always sees the schema in ``pose/schemas.py``.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass

import numpy as np

from mediaio.frame_sequence import Frame, FrameSequence
from pose.pose_types import PoseFrame, PoseLandmark, PoseSequence


@contextlib.contextmanager
def _mmengine_checkpoint_compat():
    """PyTorch >=2.6 flipped ``torch.load``'s default to
    ``weights_only=True``, but mmengine's checkpoint loader (as of 0.10.7,
    the newest release compatible with mmdet's ``mmcv<2.2.0`` pin) calls
    ``torch.load`` with no override and its checkpoints carry more than
    plain tensors, so loading breaks under the new default. Restoring the
    old default only around model init (not process-wide) fixes this
    without trusting arbitrary pickles anywhere else."""
    import torch

    original_load = torch.load

    def patched_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original_load(*args, **kwargs)

    torch.load = patched_load
    try:
        yield
    finally:
        torch.load = original_load

_COCO_KEYPOINT_NAMES = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]

_DIRECT_CANONICAL_FROM_COCO = {
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
}


@dataclass
class MMPoseConfig:
    config_path: str
    checkpoint_path: str
    device: str = "cpu"
    visibility_threshold: float = 0.3


class PoseEstimator:
    """Wraps MMPose top-down inference behind the PoseFrame/PoseSequence schema."""

    def __init__(self, config: MMPoseConfig):
        self._config = config
        self._model = None

    def _load_model(self):
        if self._model is None:
            try:
                from mmpose.apis import init_model
            except ImportError as exc:
                raise ImportError(
                    "MMPose is not installed. Install the optional 'pose' extra: "
                    "pip install -e '.[pose]'"
                ) from exc
            with _mmengine_checkpoint_compat():
                self._model = init_model(
                    self._config.config_path,
                    self._config.checkpoint_path,
                    device=self._config.device,
                )
        return self._model

    def process_frame(self, frame: Frame) -> PoseFrame:
        from mmpose.apis import inference_topdown

        model = self._load_model()
        results = inference_topdown(model, frame.image)
        landmarks = _extract_canonical_landmarks(results, self._config.visibility_threshold)
        return PoseFrame(frame_index=frame.index, timestamp=frame.timestamp, landmarks=landmarks)

    def process_sequence(self, frames: FrameSequence) -> PoseSequence:
        pose_frames = [self.process_frame(frame) for frame in frames.frames]
        return PoseSequence(
            frames=pose_frames, source_fps=frames.fps, landmark_schema="canonical_v1"
        )


def _extract_canonical_landmarks(
    results: list, visibility_threshold: float
) -> dict[str, PoseLandmark]:
    """Convert one ``inference_topdown`` result list into canonical landmarks.

    Only the first (highest-confidence) detected person is used; this
    project excludes multi-character tracking (Architecture_v2.md 1.3).
    """
    if not results:
        return {}

    instance = results[0].pred_instances
    keypoints = np.asarray(instance.keypoints[0])
    scores = np.asarray(instance.keypoint_scores[0])

    coco_landmarks: dict[str, PoseLandmark] = {}
    for name, (x, y), score in zip(_COCO_KEYPOINT_NAMES, keypoints, scores):
        coco_landmarks[name] = PoseLandmark(
            name=name,
            x=float(x),
            y=float(y),
            confidence=float(score),
            visible=float(score) >= visibility_threshold,
        )

    landmarks: dict[str, PoseLandmark] = {}
    for name in _DIRECT_CANONICAL_FROM_COCO:
        if name in coco_landmarks:
            landmarks[name] = coco_landmarks[name]

    if "nose" in coco_landmarks:
        nose = coco_landmarks["nose"]
        landmarks["head"] = PoseLandmark(
            name="head", x=nose.x, y=nose.y, confidence=nose.confidence, visible=nose.visible
        )

    neck = _midpoint(landmarks.get("left_shoulder"), landmarks.get("right_shoulder"), "neck")
    if neck is not None:
        landmarks["neck"] = neck

    pelvis = _midpoint(landmarks.get("left_hip"), landmarks.get("right_hip"), "pelvis")
    if pelvis is not None:
        landmarks["pelvis"] = pelvis

    if neck is not None and pelvis is not None:
        landmarks["spine"] = _midpoint(neck, pelvis, "spine")

    return landmarks


def _midpoint(
    a: PoseLandmark | None, b: PoseLandmark | None, name: str
) -> PoseLandmark | None:
    if a is None or b is None:
        return None
    return PoseLandmark(
        name=name,
        x=(a.x + b.x) / 2,
        y=(a.y + b.y) / 2,
        confidence=min(a.confidence, b.confidence),
        visible=a.visible and b.visible,
    )
