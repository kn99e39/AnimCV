#!/usr/bin/env python3
"""Build bounded, restartable AMASS synthetic pretraining splits.

AMASS source clips can be extremely numerous and high-frame-rate.  This tool
selects a deterministic bounded corpus, downsamples each source to 30 FPS,
keeps source files as individual artifacts, and combines only complete clips.
It therefore remains practical on the 12 GB training GPU and never lets a
temporal window cross an AMASS source-motion boundary.
"""

from __future__ import annotations

import argparse
from collections import defaultdict, deque
import json
from pathlib import Path

import numpy as np

from pose.amass_adapter import amass_sequence_id, import_amass_motion
from training.temporal_lifter import combine_datasets, load_dataset, save_dataset


_VALIDATION_SUBSETS = {"HumanEva", "SFU"}
_HOLDOUT_SUBSETS = {"SSM_synced", "Transitions_mocap"}
_REQUIRED_FIELDS = {"poses", "trans", "betas", "gender", "mocap_framerate"}


def _split(path: Path, raw_root: Path) -> str:
    subset = path.relative_to(raw_root).parts[0]
    if subset in _HOLDOUT_SUBSETS:
        return "holdout"
    if subset in _VALIDATION_SUBSETS:
        return "validation"
    return "train"


def _stratified_sources(sources: list[Path], raw_root: Path, limit: int) -> list[Path]:
    """Select deterministically in subset round-robin order."""
    buckets: dict[str, deque[Path]] = defaultdict(deque)
    for source in sorted(sources):
        buckets[source.relative_to(raw_root).parts[0]].append(source)
    selected: list[Path] = []
    names = sorted(buckets)
    while len(selected) < limit and names:
        remaining = []
        for name in names:
            if buckets[name] and len(selected) < limit:
                selected.append(buckets[name].popleft())
            if buckets[name]:
                remaining.append(name)
        names = remaining
    return selected


def _source_metadata_error(source: Path) -> str | None:
    """Reject AMASS auxiliary archives such as per-subject ``shape.npz``."""
    try:
        with np.load(source, allow_pickle=False) as raw:
            missing = sorted(_REQUIRED_FIELDS.difference(raw.files))
    except Exception as exc:  # malformed archives must not abort the whole corpus
        return f"{type(exc).__name__}: {exc}"
    return f"missing fields: {', '.join(missing)}" if missing else None


def _repair_cached_sequence_id(dataset: dict, sequence_id: str) -> bool:
    """Upgrade clips produced before corpus-relative AMASS IDs were introduced."""
    changed = dataset.get("sequence_id") != sequence_id
    dataset["sequence_id"] = sequence_id
    for sequence in dataset.get("sequences", []):
        changed = changed or sequence.get("sequence_id") != sequence_id
        sequence["sequence_id"] = sequence_id
    return changed


def _camera_views(value: str) -> list[tuple[float, float, float, float]]:
    """Parse ``yaw,pitch,distance,focal;...`` without an implicit Cartesian product."""
    views = []
    for item in value.split(";"):
        parts = [part.strip() for part in item.split(",")]
        if len(parts) != 4:
            raise ValueError("--camera-views must be yaw,pitch,distance,focal;...")
        try:
            yaw, pitch, distance, focal = (float(part) for part in parts)
        except ValueError as exc:
            raise ValueError("--camera-views values must be numbers") from exc
        if distance <= 0 or focal <= 0:
            raise ValueError("camera distance and focal length must be positive")
        views.append((yaw, pitch, distance, focal))
    if not views:
        raise ValueError("at least one camera view is required")
    return views


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare bounded AMASS synthetic 2D/3D lifter datasets")
    parser.add_argument("--raw", required=True, type=Path, help="AMASS root containing raw/<subset>/*.npz")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--body-model-root", required=True, type=Path)
    parser.add_argument("--max-frames-per-clip", type=int, default=120)
    parser.add_argument("--train-clips", type=int, default=1000)
    parser.add_argument("--validation-clips", type=int, default=100)
    parser.add_argument("--holdout-clips", type=int, default=100)
    parser.add_argument("--device", default="cpu", help="SMPL+H evaluation device, e.g. cuda")
    parser.add_argument("--camera-views", default="0,0,4.5,1500",
                        help="semicolon-separated yaw,pitch,distance,focal views; each selected source uses every view")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if min(args.max_frames_per_clip, args.train_clips, args.validation_clips, args.holdout_clips) <= 0:
        raise ValueError("all clip limits must be positive")
    camera_views = _camera_views(args.camera_views)
    raw_root = args.raw / "raw"
    if not raw_root.is_dir():
        raise ValueError(f"AMASS raw directory is missing: {raw_root}")
    limits = {"train": args.train_clips, "validation": args.validation_clips, "holdout": args.holdout_clips}
    available = {split: [] for split in limits}
    exclusions: dict[str, int] = defaultdict(int)
    npz_count = 0
    for source in sorted(raw_root.rglob("*.npz")):
        npz_count += 1
        error = _source_metadata_error(source)
        if error:
            exclusions[error] += 1
            continue
        split = _split(source, raw_root)
        available[split].append(source)
    selected = {
        split: _stratified_sources(sources, raw_root, limits[split])
        for split, sources in available.items()
    }
    missing = [split for split, sources in selected.items() if not sources]
    if missing:
        raise ValueError(f"no AMASS source clips available for {missing}")

    report: dict[str, object] = {
        "schema": "animcv_amass_preparation_v1", "raw": str(args.raw), "splits": {},
        "source_scan": {
            "npz_files": npz_count,
            "eligible_motion_files": sum(len(sources) for sources in available.values()),
            "excluded_files": sum(exclusions.values()),
            "exclusions": dict(sorted(exclusions.items())),
        },
        "camera_views": [
            {"yaw_degrees": yaw, "pitch_degrees": pitch, "distance_meters": distance, "focal_length": focal}
            for yaw, pitch, distance, focal in camera_views
        ],
    }
    for split, sources in selected.items():
        clips = []
        rebuilt = 0
        repaired_ids = 0
        for source in sources:
            base_relative = source.relative_to(raw_root).with_suffix("")
            for view_index, (yaw, pitch, distance, focal) in enumerate(camera_views):
                # Preserve the original on-disk cache and sequence IDs for the
                # historical single frontal view; new view sets are explicit.
                legacy_view = len(camera_views) == 1 and (yaw, pitch, distance, focal) == (0.0, 0.0, 4.5, 1500.0)
                relative = Path(f"{base_relative}.json" if legacy_view else f"{base_relative}.view{view_index}.json")
                source_identifier = base_relative.as_posix() if legacy_view else f"{base_relative.as_posix()}:view{view_index}"
                sequence_id = amass_sequence_id(source_identifier, yaw)
                destination = args.out / "clips" / split / relative
                if args.force or not destination.exists():
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    import_amass_motion(
                        source, destination, body_model_root=args.body_model_root, split=split,
                        max_frames=args.max_frames_per_clip, target_fps=30.0, device=args.device,
                        source_identifier=source_identifier, camera_yaw_degrees=yaw,
                        camera_pitch_degrees=pitch, camera_distance_meters=distance, focal_length=focal,
                    )
                    rebuilt += 1
                clip = load_dataset(destination)
                if _repair_cached_sequence_id(clip, sequence_id):
                    save_dataset(clip, destination)
                    repaired_ids += 1
                clips.append(clip)
        combined = combine_datasets(clips, expected_split=split)
        combined["source"] = {"dataset": "AMASS", "split": split, "raw_root": str(args.raw),
                              "max_frames_per_clip": args.max_frames_per_clip, "target_fps": 30.0}
        destination = args.out / f"{split}.json"
        save_dataset(combined, destination)
        report["splits"][split] = {"requested_clips": limits[split], "available_clips": len(available[split]),
                                    "source_clips": len(sources), "rebuilt_clips": rebuilt,
                                    "repaired_sequence_ids": repaired_ids,
                                    "camera_view_count": len(camera_views),
                                    "sequence_count": len(combined["sequences"]), "frame_count": len(combined["frames"]),
                                    "dataset": str(destination)}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "preparation_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
