"""Turn AMASS SMPL+H motion parameters into synthetic paired 2D/3D clips.

The raw AMASS archive contains motion parameters, not joints or images.  This
adapter evaluates a locally installed SMPL+H body model, renders no pixels,
and projects the resulting SMPL-24 joints through a deterministic virtual
camera.  The emitted input is deliberately marked as synthetic GT 2D so it
cannot be confused with production MMPose detections.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from pose.pose_lifter import LiftedPoseFrame, LiftedPosePoint, LiftedPoseSequence
from pose.pose_types import PoseFrame, PoseLandmark, PoseSequence
from training.temporal_lifter import build_dataset, save_dataset


_SMPL = {
    "pelvis": 0, "left_hip": 1, "right_hip": 2, "spine": 3,
    "left_knee": 4, "right_knee": 5, "thorax": 9, "left_ankle": 7,
    "right_ankle": 8, "neck": 12, "head": 15, "left_shoulder": 16,
    "right_shoulder": 17, "left_elbow": 18, "right_elbow": 19,
    "left_wrist": 20, "right_wrist": 21,
}
_DIRECT_2D = (
    "head", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee",
    "right_knee", "left_ankle", "right_ankle",
)


def import_amass_motion(
    motion_path: str | Path, out: str | Path, *, body_model_root: str | Path,
    split: str, image_size: tuple[int, int] = (1920, 1080), focal_length: float = 1500.0,
    camera_yaw_degrees: float = 0.0, camera_pitch_degrees: float = 0.0,
    camera_distance_meters: float = 4.5, max_frames: int | None = None, target_fps: float = 30.0,
    device: str = "cpu", source_identifier: str | None = None,
) -> dict[str, Any]:
    """Evaluate one AMASS ``.npz`` and write a v2 supervised clip.

    ``body_model_root`` must contain ``smplh/SMPLH_{MALE,FEMALE,NEUTRAL}.pkl``.
    The model and AMASS file stay outside Git; only the generated JSON is
    written to ``out``.
    """
    if split not in {"train", "validation", "holdout"}:
        raise ValueError("split must be train, validation, or holdout")
    width, height = image_size
    if width <= 0 or height <= 0 or focal_length <= 0 or camera_distance_meters <= 0:
        raise ValueError("image_size, focal_length, and camera_distance_meters must be positive")
    source_path = Path(motion_path)
    with np.load(source_path, allow_pickle=False) as raw:
        required = {"poses", "trans", "betas", "gender", "mocap_framerate"}
        missing = required.difference(raw.files)
        if missing:
            raise ValueError(f"AMASS motion is missing required fields: {sorted(missing)}")
        poses = np.asarray(raw["poses"], dtype=np.float32)
        trans = np.asarray(raw["trans"], dtype=np.float32)
        betas = np.asarray(raw["betas"], dtype=np.float32)
        gender = _gender(raw["gender"])
        source_fps = float(np.asarray(raw["mocap_framerate"]).item())
    if poses.ndim != 2 or poses.shape[1] != 156 or trans.shape != (len(poses), 3):
        raise ValueError("AMASS SMPL+H poses/trans must have shapes (frames, 156) and (frames, 3)")
    if source_fps <= 0 or target_fps <= 0:
        raise ValueError("AMASS mocap_framerate must be positive")
    stride = max(1, round(source_fps / target_fps))
    poses, trans = poses[::stride], trans[::stride]
    source_fps /= stride
    if max_frames is not None:
        if max_frames <= 0:
            raise ValueError("max_frames must be positive")
        poses, trans = poses[:max_frames], trans[:max_frames]
    joints = _evaluate_smplh(poses, trans, betas, gender, Path(body_model_root), device)
    camera = _virtual_camera(joints[:, 0], camera_yaw_degrees, camera_pitch_degrees, camera_distance_meters)
    camera_joints = _to_camera_axes(joints, camera)
    image_points, visible = _project(camera_joints, width, height, focal_length)

    pose_frames: list[PoseFrame] = []
    lifted_frames: list[LiftedPoseFrame] = []
    for frame_index, (joints_3d, joints_2d, valid) in enumerate(zip(camera_joints, image_points, visible)):
        landmarks = _canonical_2d(joints_2d, valid)
        root_relative = joints_3d - joints_3d[_SMPL["pelvis"]]
        target = {
            name: LiftedPosePoint(name, tuple(float(v) for v in root_relative[index]), 1.0, 0.0,
                                  observation_valid=bool(valid[index]))
            for name, index in _SMPL.items()
        }
        pose_frames.append(PoseFrame(frame_index, frame_index / source_fps, landmarks))
        lifted_frames.append(LiftedPoseFrame(frame_index, frame_index / source_fps, target))
    sequence_id = amass_sequence_id(source_identifier or source_path.stem, camera_yaw_degrees)
    dataset = build_dataset(
        PoseSequence(pose_frames, source_fps=source_fps, landmark_schema="canonical_v1"),
        LiftedPoseSequence(lifted_frames, source_fps=source_fps, backend="amass_smplh_virtual_camera"),
        image_size, sequence_id,
    )
    source = {
        "dataset": "AMASS", "split": split, "motion_path": str(source_path),
        "input_kind": "synthetic_virtual_camera_gt_2d", "body_model": "SMPL+H",
        "camera_yaw_degrees": camera_yaw_degrees, "camera_pitch_degrees": camera_pitch_degrees,
        "camera_distance_meters": camera_distance_meters, "image_size": list(image_size), "focal_length": focal_length,
    }
    dataset["source"] = source
    dataset["sequences"][0]["source"] = source
    save_dataset(dataset, out)
    return {"split": split, "sequence_id": sequence_id, "frame_count": len(dataset["frames"]),
            "valid_joint_count": sum(sum(frame["target_valid"]) for frame in dataset["frames"])}


def amass_sequence_id(source_identifier: str, camera_yaw_degrees: float) -> str:
    """Return a stable ID; callers preparing a corpus should pass a root-relative path."""
    normalized = source_identifier.replace("\\", "/").removesuffix(".npz").strip("/")
    if not normalized:
        raise ValueError("AMASS source identifier must not be empty")
    return f"amass:{normalized}:yaw{camera_yaw_degrees:g}"


def _evaluate_smplh(
    poses: np.ndarray, trans: np.ndarray, betas: np.ndarray, gender: str, root: Path, device: str,
) -> np.ndarray:
    try:
        import torch
        import smplx
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise ImportError("AMASS conversion requires the training extra with smplx") from exc
    model_file = root / "smplh" / f"SMPLH_{gender.upper()}.pkl"
    if not model_file.is_file():
        raise FileNotFoundError(f"missing SMPL+H body model: {model_file}")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"requested AMASS conversion device is unavailable: {device}")
    model = _load_smplh_model(str(root), gender, device)
    beta = np.zeros(10, dtype=np.float32)
    beta[:min(10, len(betas))] = betas[:10]
    batches = []
    with torch.no_grad():
        for start in range(0, len(poses), 256):
            pose = torch.from_numpy(poses[start:start + 256]).to(device)
            batch = len(pose)
            output = model(
                global_orient=pose[:, :3], body_pose=pose[:, 3:66],
                left_hand_pose=pose[:, 66:111], right_hand_pose=pose[:, 111:156],
                betas=torch.from_numpy(np.repeat(beta[None], batch, axis=0)).to(device),
                transl=torch.from_numpy(trans[start:start + 256]).to(device),
            )
            batches.append(output.joints[:, :24].cpu().numpy())
    return np.concatenate(batches, axis=0)


@lru_cache(maxsize=6)
def _load_smplh_model(root: str, gender: str, device: str):
    import smplx

    model = smplx.create(root, model_type="smplh", gender=gender, ext="pkl", use_pca=False)
    model.to(device).eval()
    return model


def _gender(value: np.ndarray) -> str:
    item = np.asarray(value).item()
    if isinstance(item, bytes):
        item = item.decode("utf-8")
    normalized = str(item).lower()
    return normalized if normalized in {"male", "female", "neutral"} else "neutral"


def _virtual_camera(
    pelvis: np.ndarray, yaw_degrees: float, pitch_degrees: float = 0.0, distance_meters: float = 4.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    target = pelvis.mean(axis=0) + np.array((0.0, 0.8, 0.0), dtype=np.float32)
    angle = np.deg2rad(yaw_degrees)
    pitch = np.deg2rad(pitch_degrees)
    horizontal = distance_meters * np.cos(pitch)
    position = target + np.array((horizontal * np.sin(angle), 1.2 + distance_meters * np.sin(pitch),
                                  horizontal * np.cos(angle)), dtype=np.float32)
    forward = _normalize(target - position)
    right = _normalize(np.cross(forward, np.array((0.0, 1.0, 0.0), dtype=np.float32)))
    up = _normalize(np.cross(right, forward))
    return position, right, forward, up


def _to_camera_axes(joints: np.ndarray, camera: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
    position, right, forward, up = camera
    relative = joints - position
    return np.stack((relative @ right, relative @ forward, relative @ up), axis=-1)


def _project(joints: np.ndarray, width: int, height: int, focal: float) -> tuple[np.ndarray, np.ndarray]:
    depth = joints[..., 1]
    pixels = np.empty(joints.shape[:-1] + (2,), dtype=np.float32)
    pixels[..., 0] = focal * joints[..., 0] / np.maximum(depth, 1e-6) + width / 2
    pixels[..., 1] = height / 2 - focal * joints[..., 2] / np.maximum(depth, 1e-6)
    visible = (depth > 1e-4) & (pixels[..., 0] >= 0) & (pixels[..., 0] < width) & (pixels[..., 1] >= 0) & (pixels[..., 1] < height)
    return pixels, visible


def _canonical_2d(points: np.ndarray, visible: np.ndarray) -> dict[str, PoseLandmark]:
    direct = {
        name: PoseLandmark(name, float(points[index, 0]), float(points[index, 1]), 1.0, bool(visible[index]))
        for name, index in _SMPL.items() if name in _DIRECT_2D
    }
    neck = _midpoint(direct["left_shoulder"], direct["right_shoulder"], "neck")
    pelvis = _midpoint(direct["left_hip"], direct["right_hip"], "pelvis")
    direct["neck"] = neck
    direct["pelvis"] = pelvis
    direct["spine"] = _midpoint(neck, pelvis, "spine")
    return direct


def _midpoint(a: PoseLandmark, b: PoseLandmark, name: str) -> PoseLandmark:
    return PoseLandmark(name, (a.x + b.x) / 2, (a.y + b.y) / 2, min(a.confidence, b.confidence), a.visible and b.visible)


def _normalize(vector: np.ndarray) -> np.ndarray:
    magnitude = np.linalg.norm(vector)
    if magnitude <= 1e-8:
        raise ValueError("degenerate virtual camera")
    return vector / magnitude
