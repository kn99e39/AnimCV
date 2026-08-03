"""Adapters that turn installed research datasets into trainable AnimCV clips."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pose.mpi3dhp_adapter import load_mpi3dhp_ground_truth
from training.temporal_lifter import build_dataset, save_dataset


def import_mpi3dhp_dataset(
    annotation: str | Path, camera_index: int, image_size: tuple[int, int], sequence_id: str,
    out: str | Path, start_frame: int = 0, end_frame: int | None = None, split: str = "train",
) -> dict[str, Any]:
    """Import one MPI-INF-3DHP camera clip as a v2 supervised dataset.

    The official annotation already supplies matching 2D and metric camera-space
    3D points, so this route avoids treating triangulation as a prerequisite.
    """
    pose, target = load_mpi3dhp_ground_truth(annotation, camera_index, start_frame=start_frame, end_frame=end_frame)
    dataset = build_dataset(pose, target, image_size, sequence_id)
    dataset["source"] = {
        "dataset": "MPI-INF-3DHP", "split": split, "camera_index": camera_index,
        "annotation_path": str(annotation), "source_start_frame": start_frame, "source_end_frame": end_frame,
        "input_kind": "dataset_ground_truth_2d",
    }
    dataset["sequences"][0]["source"] = dataset["source"]
    save_dataset(dataset, out)
    return {"sequence_id": sequence_id, "split": split, "frame_count": len(dataset["frames"]),
            "valid_joint_count": sum(sum(frame["target_valid"]) for frame in dataset["frames"])}
