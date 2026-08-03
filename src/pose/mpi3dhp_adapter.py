"""Import a locally licensed MPI-INF-3DHP sequence for calibrated evaluation.

The raw dataset is deliberately never copied into the repository.  This
adapter converts its documented 28-joint camera annotation into AnimCV's
canonical 2D and root-relative 3D schemas, so a user-provided local cache can
be used as an objective evaluation input.
"""

from __future__ import annotations

from pathlib import Path
import re

import numpy as np

from pose.camera_calibration import CameraCalibration
from pose.pose_lifter import LiftedPoseFrame, LiftedPosePoint, LiftedPoseSequence
from pose.pose_types import PoseFrame, PoseLandmark, PoseSequence


# Official MMPose MPI-INF-3DHP preprocessing uses this raw-28 subset.  The
# resulting 17-joint order is documented in its ``mpi_inf_3dhp.py`` dataset
# definition; keeping the mapping here makes our import independently
# reproducible without importing MMPose at runtime.
_JOINTS_17 = (7, 5, 14, 15, 16, 9, 10, 11, 23, 24, 25, 18, 19, 20, 4, 3, 6)
_CANONICAL_FROM_17 = {
    "head": 16, "neck": 1, "spine": 15, "pelvis": 14,
    "right_shoulder": 2, "right_elbow": 3, "right_wrist": 4,
    "left_shoulder": 5, "left_elbow": 6, "left_wrist": 7,
    "right_hip": 8, "right_knee": 9, "right_ankle": 10,
    "left_hip": 11, "left_knee": 12, "left_ankle": 13,
}


def load_mpi3dhp_calibration(path: str | Path, camera_index: int) -> CameraCalibration:
    """Read one camera's intrinsics from an official calibration text file."""
    text = Path(path).read_text(encoding="utf-8")
    blocks = re.split(r"(?=^name\s+\d+\s*$)", text, flags=re.MULTILINE)
    selected = next((block for block in blocks if re.search(rf"^name\s+{camera_index}\s*$", block, re.MULTILINE)), None)
    if selected is None:
        raise ValueError(f"camera {camera_index} is absent from {path}")

    size = _numbers_after(selected, "size")
    intrinsic = _numbers_after(selected, "intrinsic")
    if len(size) != 2 or len(intrinsic) != 16:
        raise ValueError("invalid MPI-INF-3DHP camera calibration block")
    return CameraCalibration(
        image_width=int(size[0]), image_height=int(size[1]),
        fx=intrinsic[0], fy=intrinsic[5], cx=intrinsic[2], cy=intrinsic[6],
        source=f"MPI-INF-3DHP official S1/Seq1 camera {camera_index}",
    )


def load_mpi3dhp_ground_truth(
    annotation_path: str | Path, camera_index: int, *, fps: float = 25.0,
    start_frame: int = 0, end_frame: int | None = None,
) -> tuple[PoseSequence, LiftedPoseSequence]:
    """Convert one camera's 2D/3D ground truth into AnimCV evaluation data."""
    try:
        from scipy.io import loadmat
    except ImportError as exc:  # pragma: no cover - optional data import dependency
        raise ImportError("MPI-INF-3DHP import requires scipy; install the pose evaluation extra") from exc
    data = loadmat(annotation_path)
    if "annot2" not in data or "annot3" not in data:
        raise ValueError("MPI-INF-3DHP annot.mat must contain annot2 and annot3")
    raw_2d = np.asarray(data["annot2"][camera_index, 0]).reshape(-1, 28, 2)
    raw_3d = np.asarray(data["annot3"][camera_index, 0]).reshape(-1, 28, 3)
    stop = len(raw_2d) if end_frame is None else min(end_frame + 1, len(raw_2d))
    if not 0 <= start_frame < stop:
        raise ValueError("requested MPI-INF-3DHP frame range is empty")
    raw_2d, raw_3d = raw_2d[start_frame:stop, _JOINTS_17], raw_3d[start_frame:stop, _JOINTS_17]

    poses, lifted_frames = [], []
    for offset, (frame_2d, frame_3d) in enumerate(zip(raw_2d, raw_3d)):
        # ``extract-frames`` intentionally numbers a trimmed clip from zero.
        # Keep imported GT in that same local-clip index space; callers retain
        # ``start_frame`` separately as dataset provenance.
        frame_index = offset
        landmarks = {
            name: PoseLandmark(name, float(frame_2d[index, 0]), float(frame_2d[index, 1]), 1.0, True)
            for name, index in _CANONICAL_FROM_17.items()
        }
        poses.append(PoseFrame(frame_index, frame_index / fps, landmarks))
        # Dataset camera axes are +X right, +Y down, +Z forward in mm.
        # AnimCV's camera axes are +X right, +Y forward, +Z up in metres.
        converted = np.column_stack((frame_3d[:, 0], frame_3d[:, 2], -frame_3d[:, 1])) * 0.001
        converted -= converted[14]  # Root-relative, matching VideoPose3D output contract.
        points = {
            name: LiftedPosePoint(name, tuple(float(v) for v in converted[index]), 1.0, 0.0)
            for name, index in _CANONICAL_FROM_17.items()
        }
        lifted_frames.append(LiftedPoseFrame(frame_index, frame_index / fps, points))
    return (
        PoseSequence(poses, source_fps=fps, landmark_schema="canonical_v1"),
        LiftedPoseSequence(lifted_frames, source_fps=fps, backend="mpi_inf_3dhp_ground_truth"),
    )


def _numbers_after(block: str, label: str) -> list[float]:
    match = re.search(rf"^\s*{label}\s+(.+)$", block, flags=re.MULTILINE)
    return [] if match is None else [float(value) for value in match.group(1).split()]
