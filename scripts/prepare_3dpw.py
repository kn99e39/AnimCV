#!/usr/bin/env python3
"""Prepare an installed 3DPW archive into restartable AnimCV lifter splits.

The official archive is never changed.  Each source pickle becomes one small
supervised JSON artifact under ``--out/<split>/``; the corresponding combined
split is then rebuilt from those complete artifacts, so temporal windows never
cross a source sequence or actor boundary.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pose.three_dpw_adapter import import_3dpw_dataset
from training.temporal_lifter import combine_datasets, load_dataset, save_dataset


_SPLITS = {"train": "train", "validation": "validation", "test": "holdout"}


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare official 3DPW labels for AnimCV training/evaluation")
    parser.add_argument("--raw", required=True, type=Path, help="3DPW archive root containing sequenceFiles/")
    parser.add_argument("--out", required=True, type=Path, help="Writable prepared-data directory")
    parser.add_argument("--force", action="store_true", help="Rebuild already prepared source artifacts")
    args = parser.parse_args()
    source_root = args.raw / "sequenceFiles"
    if not source_root.is_dir():
        raise ValueError(f"3DPW sequenceFiles directory is missing: {source_root}")

    report: dict[str, object] = {"schema": "animcv_3dpw_preparation_v1", "raw": str(args.raw), "splits": {}}
    for official_split, animcv_split in _SPLITS.items():
        annotations = sorted((source_root / official_split).glob("*.pkl"))
        if not annotations:
            raise ValueError(f"no 3DPW annotations found for {official_split}: {source_root / official_split}")
        split_dir = args.out / animcv_split
        split_dir.mkdir(parents=True, exist_ok=True)
        prepared = []
        rebuilt = 0
        for annotation in annotations:
            destination = split_dir / f"{annotation.stem}.json"
            if args.force or not destination.exists():
                import_3dpw_dataset(annotation, destination, split=animcv_split)
                rebuilt += 1
            prepared.append(load_dataset(destination))
        combined = combine_datasets(prepared, expected_split=animcv_split)
        combined["source"] = {"dataset": "3DPW", "split": animcv_split,
                              "raw_root": str(args.raw), "official_split": official_split}
        destination = args.out / f"{animcv_split}.json"
        save_dataset(combined, destination)
        report["splits"][animcv_split] = {"official_split": official_split, "source_files": len(annotations),
                                            "rebuilt_files": rebuilt, "sequence_count": len(combined["sequences"]),
                                            "frame_count": len(combined["frames"]), "dataset": str(destination)}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "preparation_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
