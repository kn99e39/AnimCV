"""Oracle world / fixed-camera reference loaders for the Root Translation
control experiment (sections 8 & 9 of docs/51).

These EXPOSE ground-truth global joint positions that the existing
production adapters (three_dpw_adapter.py, mpi3dhp_adapter.py) deliberately
discard once they make their output root-relative
(three_dpw_adapter.py:89 `camera_joints -= camera_joints[0]`,
mpi3dhp_adapter.py:87 `converted -= converted[14]`). Nothing here changes
what those adapters output to production training/evaluation; this reads the
same source files a second time, for a separate, research-only control
experiment, and imports their joint-index constants rather than
reimplementing them so the two stay in agreement by construction.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pickle

import numpy as np

from pose.mpi3dhp_adapter import _CANONICAL_FROM_17, _JOINTS_17
from pose.three_dpw_adapter import _SMPL_TO_CANONICAL


@dataclass(frozen=True)
class WorldReferenceFrame:
    frame_index: int
    pelvis_world: tuple[float, float, float] | None
    left_ankle_world: tuple[float, float, float] | None
    right_ankle_world: tuple[float, float, float] | None
    valid: bool
    # World-to-camera rotation for this frame, 3DPW only (None => static/no
    # extrinsics tracked, e.g. MPI-INF-3DHP's fixed studio cameras).
    camera_rotation: np.ndarray | None = None


def load_3dpw_world_reference(
    path: str | Path, *, fps: float = 30.0,
) -> list[tuple[str, list[WorldReferenceFrame]]]:
    """Per-actor world-space pelvis/ankle GT plus per-frame camera rotation.

    3DPW's camera is handheld and moves between frames; ``camera_rotation``
    lets the control experiment correlate translation-recovery failure with
    how much the camera itself rotated (section 9).
    """
    source_path = Path(path)
    with source_path.open("rb") as handle:
        raw = pickle.load(handle, encoding="latin1")
    required = {"sequence", "jointPositions", "cam_poses", "campose_valid"}
    missing = required.difference(raw)
    if missing:
        raise ValueError(f"3DPW sequence is missing required fields: {sorted(missing)}")
    camera = np.asarray(raw["cam_poses"], dtype=float)
    if camera.ndim != 3 or camera.shape[1:] != (4, 4):
        raise ValueError("invalid 3DPW camera arrays")

    output = []
    for actor, (raw_3d, aligned) in enumerate(zip(raw["jointPositions"], raw["campose_valid"])):
        joints_world = np.asarray(raw_3d, dtype=float).reshape(-1, 24, 3)
        aligned = np.asarray(aligned, dtype=bool)
        frame_count = min(len(joints_world), len(aligned), len(camera))
        frames = []
        for index in range(frame_count):
            valid = bool(aligned[index])
            pelvis = joints_world[index, _SMPL_TO_CANONICAL["pelvis"]]
            left_ankle = joints_world[index, _SMPL_TO_CANONICAL["left_ankle"]]
            right_ankle = joints_world[index, _SMPL_TO_CANONICAL["right_ankle"]]
            frames.append(WorldReferenceFrame(
                frame_index=index,
                pelvis_world=tuple(float(v) for v in pelvis) if valid else None,
                left_ankle_world=tuple(float(v) for v in left_ankle) if valid else None,
                right_ankle_world=tuple(float(v) for v in right_ankle) if valid else None,
                valid=valid,
                camera_rotation=camera[index, :3, :3].copy(),
            ))
        output.append((f"3dpw:{raw['sequence']}:actor{actor}", frames))
    return output


def load_mpi3dhp_world_reference(
    annotation_path: str | Path, camera_index: int, *, fps: float = 25.0,
    start_frame: int = 0, end_frame: int | None = None,
) -> list[WorldReferenceFrame]:
    """Fixed-studio-camera-space pelvis/ankle GT, pre pelvis-subtraction.

    MPI-INF-3DHP's cameras do not move within a sequence, so this per-camera
    3D position stands in for a "world" reference up to one constant
    rotation/offset — a clean static-camera contrast to 3DPW's moving camera.
    """
    try:
        from scipy.io import loadmat
    except ImportError as exc:  # pragma: no cover - optional data import dependency
        raise ImportError("MPI-INF-3DHP import requires scipy; install the pose evaluation extra") from exc
    data = loadmat(annotation_path)
    if "annot3" not in data:
        raise ValueError("MPI-INF-3DHP annot.mat must contain annot3")
    raw_3d = np.asarray(data["annot3"][camera_index, 0]).reshape(-1, 28, 3)
    stop = len(raw_3d) if end_frame is None else min(end_frame + 1, len(raw_3d))
    if not 0 <= start_frame < stop:
        raise ValueError("requested MPI-INF-3DHP frame range is empty")
    raw_3d = raw_3d[start_frame:stop][:, _JOINTS_17]

    frames = []
    for offset, frame_3d in enumerate(raw_3d):
        # Dataset camera axes +X right, +Y down, +Z forward, mm -> AnimCV
        # +X right, +Y forward, +Z up, metres (matches mpi3dhp_adapter.py:86).
        converted = np.column_stack((frame_3d[:, 0], frame_3d[:, 2], -frame_3d[:, 1])) * 0.001
        pelvis = converted[_CANONICAL_FROM_17["pelvis"]]
        left_ankle = converted[_CANONICAL_FROM_17["left_ankle"]]
        right_ankle = converted[_CANONICAL_FROM_17["right_ankle"]]
        frames.append(WorldReferenceFrame(
            frame_index=offset,
            pelvis_world=tuple(float(v) for v in pelvis),
            left_ankle_world=tuple(float(v) for v in left_ankle),
            right_ankle_world=tuple(float(v) for v in right_ankle),
            valid=True,
            camera_rotation=None,
        ))
    return frames
