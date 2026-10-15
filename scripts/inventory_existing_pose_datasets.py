#!/usr/bin/env python3
"""DIAGNOSTIC (Worklog 84): inventory of human-pose datasets already stored on LabServer63.

Read-only and offline: walks the declared dataset roots, counts what is actually on disk (RGB images,
image archives, videos, 3D annotation files, camera calibration files), reads the historical bank /
build reports for prior usage, and classifies every dataset for the Worklog 84 question "can it
supply NEW paired RGB<->3D arm supervision?".  It never downloads, never opens a network connection
and never reads credential files.  Standard library only (runs with the host ``python3 -I``).

Classification (``classify``; first matching rule wins):
  absent                       -- nothing on disk;
  evaluation_reserved          -- reserved as an evaluation source (MPI-INF-3DHP official TEST);
  already_used_same_purpose    -- already supplies the historical J1/R1 paired training/selection data;
  annotation_only              -- 3D labels without on-disk RGB (cannot be paired RGB<->3D supervision);
  no_3d_camera_labels          -- RGB without camera-space 3D joints;
  eligible_new_paired_training -- RGB + camera-space 3D + calibration, unused for this purpose.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

IMAGE_EXT = (".jpg", ".jpeg", ".png")
VIDEO_EXT = (".avi", ".mp4", ".mov", ".mkv")
ARCHIVE_EXT = (".tar", ".zip", ".tgz", ".gz")

# Declared before the walk; ``roots`` are relative to the home directory.
DATASETS = {
    "3DPW": {"roots": ["animcv-data/datasets/3dpw"], "labels_3d": [".pkl"], "calibration_in_labels": True,
             "role": "historical paired training (TRAIN), selection (VALIDATION), development evaluation (TEST)",
             "used_for_paired_training": True, "evaluation_reserved": False},
    "BEDLAM": {"roots": ["animcv-data/bedlam"], "labels_3d": [".npz", ".tar.gz"], "calibration_in_labels": True,
               "role": "historical paired training/validation (Worklog 73 external banks)",
               "used_for_paired_training": True, "evaluation_reserved": False},
    "AGORA (labels shipped inside the BEDLAM label bundle)": {
        "roots": ["animcv-data/bedlam/labels/all_npz_12_training"], "name_filter": "agora",
        "labels_3d": [".npz"], "calibration_in_labels": True, "role": "never used (Worklog 73 builder skips agora*)",
        "used_for_paired_training": False, "evaluation_reserved": False},
    "MPI-INF-3DHP official TEST": {"roots": ["animcv-data/mpi_inf_3dhp_test/mpi_inf_3dhp_test_set"],
                                   "labels_3d": [".mat"], "calibration": [".calib"],
                                   "role": "Worklog 83 independent evaluation (evaluation-only)",
                                   "used_for_paired_training": False, "evaluation_reserved": True},
    "MPI-INF-3DHP TRAIN (S1-S8)": {"roots": ["AnimCV/datasets/mpi_inf_3dhp", "animcv-output/data/mpi_inf_3dhp"],
                                   "labels_3d": [".mat"], "calibration": [".calibration"],
                                   "role": "legacy annotation-only lifter training (docs 04/06/09/10)",
                                   "used_for_paired_training": False, "evaluation_reserved": False},
    "AMASS": {"roots": ["animcv-data/amass/raw"], "labels_3d": [".npz"], "calibration": [],
              "role": "legacy synthetic-projection lifter training (docs 09/10); mocap only",
              "used_for_paired_training": False, "evaluation_reserved": False},
    "Human3.6M": {"roots": ["animcv-data/human36m"], "labels_3d": [".cdf", ".h5", ".npz"], "calibration": [],
                  "role": "placeholder directories only", "used_for_paired_training": False, "evaluation_reserved": False},
}


def classify(rgb: int, labels_3d: int, calibration: bool, used_for_paired_training: bool, evaluation_reserved: bool) -> str:
    if rgb == 0 and labels_3d == 0:
        return "absent"
    if evaluation_reserved:
        return "evaluation_reserved"
    if used_for_paired_training:
        return "already_used_same_purpose"
    if rgb == 0:
        return "annotation_only"
    if labels_3d == 0 or not calibration:
        return "no_3d_camera_labels"
    return "eligible_new_paired_training"


def walk(root: Path, spec: dict) -> dict:
    out = {"root": str(root), "exists": root.exists(), "images": 0, "videos": 0, "archives": 0, "archive_bytes": 0,
           "labels_3d": 0, "calibration_files": 0, "bytes": 0}
    if not root.exists():
        return out
    name_filter = spec.get("name_filter")
    for dirpath, _dirs, files in os.walk(root):
        for f in files:
            low = f.lower()
            if name_filter and name_filter not in low:
                continue
            p = Path(dirpath) / f
            try:
                size = p.stat().st_size
            except OSError:
                continue
            out["bytes"] += size
            if low.endswith(IMAGE_EXT):
                out["images"] += 1
            elif low.endswith(VIDEO_EXT):
                out["videos"] += 1
            elif any(low.endswith(e) for e in spec.get("labels_3d", [])) and "image" not in dirpath.lower() \
                    and not low.endswith(".tar") and "png" not in low:
                out["labels_3d"] += 1
            elif low.endswith(ARCHIVE_EXT):
                out["archives"] += 1
                out["archive_bytes"] += size
            if any(low.endswith(e) for e in spec.get("calibration", [])):
                out["calibration_files"] += 1
    return out


def rgb_count(walks: list[dict], name: str) -> int:
    """On-disk RGB observations: loose images + videos, plus BEDLAM's image tars (members counted by its verify report)."""
    n = sum(w["images"] + w["videos"] for w in walks)
    if name == "BEDLAM":
        n += sum(w["archives"] for w in walks)  # image tars (train 6/30 fps scenes + validation PNG parts)
    return n


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--home", type=Path, default=Path.home())
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit("STOP: inventory output exists; never overwrite")
    report = {"schema": "animcv_w84_dataset_inventory_v1", "home": str(args.home), "datasets": {}}
    for name, spec in DATASETS.items():
        walks = [walk(args.home / r, spec) for r in spec["roots"]]
        rgb = rgb_count(walks, name)
        labels = sum(w["labels_3d"] for w in walks)
        calibration = spec.get("calibration_in_labels", False) or any(w["calibration_files"] for w in walks)
        report["datasets"][name] = {
            "walks": walks, "rgb_observations_or_archives": rgb, "label_files_3d": labels, "camera_calibration": calibration,
            "role": spec["role"], "classification": classify(rgb, labels, calibration, spec["used_for_paired_training"],
                                                             spec["evaluation_reserved"])}
    report["eligible_new_paired_training"] = [n for n, d in report["datasets"].items()
                                              if d["classification"] == "eligible_new_paired_training"]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1))
    print(json.dumps({n: {k: d[k] for k in ("rgb_observations_or_archives", "label_files_3d", "camera_calibration", "classification")}
                      for n, d in report["datasets"].items()} | {"eligible_new_paired_training": report["eligible_new_paired_training"]},
                     indent=1))


if __name__ == "__main__":
    main()
