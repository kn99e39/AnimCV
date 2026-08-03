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
from pose.subject_tracker import Detection, SubjectTracker


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
    subject_box: tuple[float, float, float, float] | None = None
    detector_config_path: str | None = None
    detector_checkpoint_path: str | None = None


class PoseEstimator:
    """Wraps MMPose top-down inference behind the PoseFrame/PoseSequence schema."""

    def __init__(self, config: MMPoseConfig):
        self._config = config
        self._model = None
        self._detector = None
        self.last_tracking_report: dict[str, int | float] = {}

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

    def _load_detector(self):
        if self._config.detector_config_path is None or self._config.detector_checkpoint_path is None:
            from pose.default_detector import get_default_detector_checkpoint_path, get_default_detector_config_path
            self._config.detector_config_path = get_default_detector_config_path()
            self._config.detector_checkpoint_path = get_default_detector_checkpoint_path()
        if self._detector is None:
            from mmdet.apis import init_detector
            # MMDetection's compatible mmengine release has the same
            # PyTorch >=2.6 checkpoint-loading issue as MMPose.
            with _mmengine_checkpoint_compat():
                self._detector = init_detector(
                    self._config.detector_config_path,
                    self._config.detector_checkpoint_path,
                    device=self._config.device,
                )
        return self._detector

    def process_frame(self, frame: Frame) -> PoseFrame:
        """Estimate one frame through the same detector-first path as clips.

        A top-down pose model over a whole image is not a valid production
        fallback: it silently fails on a small or off-centre subject.  Keep
        the public one-frame API, but make its contract detector-first too.
        """
        sequence = FrameSequence(
            frames=[frame], fps=1.0, width=frame.width, height=frame.height,
            source_path="single_frame",
        )
        return self._process_tracked_sequence(sequence).frames[0]

    def process_sequence(self, frames: FrameSequence) -> PoseSequence:
        return self._process_tracked_sequence(frames)

    def process_sequence_with_evaluation_boxes(
        self, frames: FrameSequence, ground_truth: PoseSequence, padding: float = 1.25
    ) -> PoseSequence:
        """Run top-down pose only inside GT-derived boxes for benchmark diagnosis.

        This is deliberately named as an *evaluation* method: it separates
        top-down keypoint regression from person detection, but must never be
        used to claim an end-user video pipeline has solved tracking.
        """
        from mmpose.apis import inference_topdown

        truth_by_index = {frame.frame_index: frame for frame in ground_truth.frames}
        model = self._load_model()
        output = []
        for frame in frames.frames:
            truth = truth_by_index.get(frame.index)
            if truth is None or not truth.landmarks:
                output.append(PoseFrame(frame.index, frame.timestamp, {}))
                continue
            xs = [point.x for point in truth.landmarks.values()]
            ys = [point.y for point in truth.landmarks.values()]
            cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
            width, height = (max(xs) - min(xs)) * padding, (max(ys) - min(ys)) * padding
            bbox = (cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2)
            results = inference_topdown(model, frame.image, [bbox])
            output.append(PoseFrame(frame.index, frame.timestamp, _extract_canonical_landmarks(
                results, self._config.visibility_threshold
            )))
        return PoseSequence(
            output, source_fps=frames.fps, landmark_schema="canonical_v1",
            observation_confidence_threshold=self._config.visibility_threshold,
        )

    def _process_tracked_sequence(self, frames: FrameSequence) -> PoseSequence:
        from mmdet.apis import inference_detector
        from mmpose.apis import inference_topdown
        from mmengine.registry import init_default_scope

        detector = self._load_detector()
        model = self._load_model()
        tracker = SubjectTracker(self._config.subject_box)
        pose_frames = []
        detector_candidates = 0
        selected_frames = 0
        for frame in frames.frames:
            # inference_topdown switches the global default scope to mmpose;
            # switch it back before constructing MMDetection's pipeline.
            init_default_scope("mmdet")
            result = inference_detector(detector, frame.image)
            instances = result.pred_instances
            detections = [
                Detection(tuple(float(value) for value in bbox), float(score))
                for bbox, score, label in zip(instances.bboxes.cpu().numpy(), instances.scores.cpu().numpy(), instances.labels.cpu().numpy())
                if int(label) == 0 and float(score) >= 0.3
            ]
            detector_candidates += len(detections)
            selected = tracker.select(detections)
            if selected is None:
                landmarks = {}
            else:
                selected_frames += 1
                results = inference_topdown(model, frame.image, [selected.bbox])
                landmarks = _extract_canonical_landmarks(results, self._config.visibility_threshold)
            pose_frames.append(PoseFrame(frame_index=frame.index, timestamp=frame.timestamp, landmarks=landmarks))
        total = len(frames.frames)
        self.last_tracking_report = {
            "frame_count": total,
            "tracked_frame_count": selected_frames,
            "tracking_success_rate": selected_frames / total if total else 0.0,
            "no_detection_frame_count": total - selected_frames,
            "mean_person_candidates_per_frame": detector_candidates / total if total else 0.0,
        }
        return PoseSequence(
            frames=pose_frames, source_fps=frames.fps, landmark_schema="canonical_v1",
            observation_confidence_threshold=self._config.visibility_threshold,
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
