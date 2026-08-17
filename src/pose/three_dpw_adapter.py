"""Import 3DPW's paired 2D detections and 3D SMPL joints.

3DPW stores one pickle per recorded sequence.  Its ``jointPositions`` are
world-space SMPL-24 joints and ``cam_poses`` are world-to-camera extrinsics;
this adapter applies that extrinsic before converting the OpenCV camera axes
to AnimCV's camera convention.  The result is root-relative metres, matching
the temporal lifter's training contract.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np

from pose.pose_lifter import LiftedPoseFrame, LiftedPosePoint, LiftedPoseSequence
from pose.pose_types import PoseFrame, PoseLandmark, PoseSequence
from training.temporal_lifter import build_dataset, combine_datasets, save_dataset


# 3DPW calls this "Coco-Format", but its 18 entries use the legacy OpenPose
# ordering (including an explicit neck).  We intentionally derive pelvis and
# spine exactly like the production COCO/MMPose adapter does.
_COCO18 = (
    "nose", "neck", "right_shoulder", "right_elbow", "right_wrist",
    "left_shoulder", "left_elbow", "left_wrist", "right_hip", "right_knee",
    "right_ankle", "left_hip", "left_knee", "left_ankle", "right_eye",
    "left_eye", "right_ear", "left_ear",
)
_DIRECT = {
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee",
    "right_knee", "left_ankle", "right_ankle",
}

# SMPL-24 body-joint order.  The two torso points are retained because the
# lifter's 17-joint target layout includes them; ``build_dataset`` aliases
# thorax input to neck but can supervise both targets independently.
_SMPL_TO_CANONICAL = {
    "pelvis": 0, "left_hip": 1, "right_hip": 2, "spine": 3,
    "left_knee": 4, "right_knee": 5, "thorax": 9, "left_ankle": 7,
    "right_ankle": 8, "neck": 12, "head": 15, "left_shoulder": 16,
    "right_shoulder": 17, "left_elbow": 18, "right_elbow": 19,
    "left_wrist": 20, "right_wrist": 21,
}


def load_3dpw_ground_truth(path: str | Path, *, fps: float = 30.0) -> list[tuple[str, PoseSequence, LiftedPoseSequence, tuple[int, int]]]:
    """Load every actor in one official 3DPW sequence pickle.

    Invalid camera-alignment frames and missing 2D detections are retained as
    invalid landmarks rather than silently becoming supervision.  Frames with
    no usable pelvis are filtered by ``build_dataset`` later.
    """
    if fps <= 0:
        raise ValueError("fps must be positive")
    source_path = Path(path)
    with source_path.open("rb") as handle:
        raw = pickle.load(handle, encoding="latin1")
    required = {"sequence", "poses2d", "jointPositions", "cam_poses", "campose_valid", "cam_intrinsics"}
    missing = required.difference(raw)
    if missing:
        raise ValueError(f"3DPW sequence is missing required fields: {sorted(missing)}")

    camera = np.asarray(raw["cam_poses"], dtype=float)
    intrinsics = np.asarray(raw["cam_intrinsics"], dtype=float)
    if camera.ndim != 3 or camera.shape[1:] != (4, 4) or intrinsics.shape != (3, 3):
        raise ValueError("invalid 3DPW camera arrays")
    image_size = (int(round(intrinsics[0, 2] * 2)), int(round(intrinsics[1, 2] * 2)))
    if min(image_size) <= 0:
        raise ValueError("invalid 3DPW image size inferred from intrinsics")

    output = []
    for actor, (raw_2d, raw_3d, aligned) in enumerate(zip(raw["poses2d"], raw["jointPositions"], raw["campose_valid"])):
        points_2d = np.asarray(raw_2d, dtype=float)
        joints_world = np.asarray(raw_3d, dtype=float).reshape(-1, 24, 3)
        aligned = np.asarray(aligned, dtype=bool)
        frame_count = min(len(points_2d), len(joints_world), len(aligned), len(camera))
        if points_2d.ndim != 3 or points_2d.shape[1:] != (3, 18):
            raise ValueError("3DPW poses2d must have shape (frames, 3, 18)")
        poses: list[PoseFrame] = []
        lifted: list[LiftedPoseFrame] = []
        for frame_index in range(frame_count):
            landmarks = _canonical_2d(points_2d[frame_index]) if aligned[frame_index] else {}
            poses.append(PoseFrame(frame_index, frame_index / fps, landmarks))
            camera_joints = _world_to_animcv_camera(joints_world[frame_index], camera[frame_index])
            camera_joints -= camera_joints[0]
            target_points = {
                name: LiftedPosePoint(name, tuple(float(v) for v in camera_joints[index]), 1.0, 0.0,
                                      observation_valid=bool(aligned[frame_index]))
                for name, index in _SMPL_TO_CANONICAL.items()
            }
            lifted.append(LiftedPoseFrame(frame_index, frame_index / fps, target_points))
        sequence_id = f"3dpw:{raw['sequence']}:actor{actor}"
        output.append((sequence_id, PoseSequence(poses, source_fps=fps, landmark_schema="canonical_v1"),
                       LiftedPoseSequence(lifted, source_fps=fps, backend="3dpw_ground_truth"), image_size))
    if not output:
        raise ValueError("3DPW sequence contains no actors")
    return output


def import_3dpw_dataset(path: str | Path, out: str | Path, *, split: str) -> dict[str, Any]:
    """Convert all actors of a 3DPW pickle into one trainable v2 dataset."""
    if split not in {"train", "validation", "holdout"}:
        raise ValueError("split must be train, validation, or holdout")
    datasets = []
    for sequence_id, pose, lifted, image_size in load_3dpw_ground_truth(path):
        dataset = build_dataset(pose, lifted, image_size, sequence_id)
        source = {"dataset": "3DPW", "split": split, "annotation_path": str(path),
                  "input_kind": "official_3dpw_2d_detection", "coordinate_frame": "camera_root_relative"}
        dataset["source"] = source
        dataset["sequences"][0]["source"] = source
        datasets.append(dataset)
    combined = combine_datasets(datasets, expected_split=split)
    combined["source"] = {"dataset": "3DPW", "split": split, "annotation_path": str(path)}
    save_dataset(combined, out)
    return {"split": split, "sequence_count": len(combined["sequences"]), "frame_count": len(combined["frames"]),
            "valid_joint_count": sum(sum(frame["target_valid"]) for frame in combined["frames"])}


def _canonical_2d(raw: np.ndarray) -> dict[str, PoseLandmark]:
    coco = {}
    for index, name in enumerate(_COCO18):
        x, y, confidence = raw[:, index]
        valid = bool(np.isfinite((x, y, confidence)).all() and confidence > 0)
        coco[name] = PoseLandmark(name, float(x) if valid else 0.0, float(y) if valid else 0.0,
                                  float(confidence) if valid else 0.0, valid)
    result = {name: coco[name] for name in _DIRECT}
    result["head"] = _rename(coco["nose"], "head")
    result["neck"] = _midpoint(result["left_shoulder"], result["right_shoulder"], "neck")
    result["pelvis"] = _midpoint(result["left_hip"], result["right_hip"], "pelvis")
    result["spine"] = _midpoint(result["neck"], result["pelvis"], "spine")
    return result


def _rename(point: PoseLandmark, name: str) -> PoseLandmark:
    return PoseLandmark(name, point.x, point.y, point.confidence, point.visible)


def _midpoint(a: PoseLandmark, b: PoseLandmark, name: str) -> PoseLandmark:
    visible = a.visible and b.visible
    return PoseLandmark(name, (a.x + b.x) / 2, (a.y + b.y) / 2, min(a.confidence, b.confidence), visible)


def _world_to_animcv_camera(joints_world: np.ndarray, world_to_camera: np.ndarray) -> np.ndarray:
    """Apply 3DPW's OpenCV world→camera matrix, then convert to AnimCV axes."""
    rotation = world_to_camera[:3, :3]
    translation = world_to_camera[:3, 3]
    camera = joints_world @ rotation.T + translation
    # OpenCV camera (+X right, +Y down, +Z forward) -> AnimCV (+X right,
    # +Y forward, +Z up), without changing metre units.
    return np.column_stack((camera[:, 0], camera[:, 2], -camera[:, 1]))
