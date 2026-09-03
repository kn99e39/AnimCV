#!/usr/bin/env python3
"""Build the deterministic frame research bank (Architecture_v3 section 4).

The bank is the shared, fingerprinted frame set every frame-first candidate
trains and is evaluated on. It is built once and then reused; nothing in the
research loop is allowed to rebuild it implicitly.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from common.serialization import write_json
from framepose.bank import BankRequest, build_bank


def _requests(values: list[str], stride: dict[str, int]) -> list[BankRequest]:
    requests = []
    for value in values:
        parts = value.split("=", 1)
        if len(parts) != 2:
            raise ValueError(f"--source expects SOURCE:SPLIT=PATH, got {value!r}")
        head, path = parts
        if ":" not in head:
            raise ValueError(f"--source expects SOURCE:SPLIT=PATH, got {value!r}")
        source, split = head.split(":", 1)
        requests.append(BankRequest(source=source, split=split, dataset_path=Path(path),
                                    stride=stride.get(split, 1)))
    return requests


def _image_roots(values: list[str]) -> dict[str, str]:
    roots = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"--image-root expects KEY=PATH, got {value!r}")
        key, path = value.split("=", 1)
        roots[key] = path
    return roots


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a deterministic AnimCV frame pose research bank")
    parser.add_argument("--source", action="append", required=True,
                        help="SOURCE:SPLIT=PREPARED_DATASET.json (repeatable)")
    parser.add_argument("--image-root", action="append", default=[], help="KEY=PATH (repeatable)")
    parser.add_argument("--out", required=True, type=Path, help="Bank index path (.json)")
    parser.add_argument("--train-stride", type=int, default=1)
    parser.add_argument("--validation-stride", type=int, default=1)
    parser.add_argument("--test-stride", type=int, default=1)
    parser.add_argument("--require-rgb", action="store_true",
                        help="Build the paired-modality subset: only frames with real, existing imagery")
    parser.add_argument("--no-verify-images", action="store_true",
                        help="Skip on-disk existence checks (not recommended)")
    args = parser.parse_args()

    stride = {"train": args.train_stride, "validation": args.validation_stride, "test": args.test_stride}
    bank, report = build_bank(
        _requests(args.source, stride),
        image_roots=_image_roots(args.image_root),
        require_rgb=args.require_rgb,
        verify_images=not args.no_verify_images,
    )
    index_path, array_path = bank.save(args.out)
    fingerprint = bank.fingerprint(index_path)
    write_json(args.out.with_name(args.out.stem + "_report.json"),
               {"schema": "animcv_frame_pose_bank_report_v1", **report, "fingerprint": fingerprint})
    print(json.dumps({"index": str(index_path), "arrays": str(array_path),
                      "split_counts": report["split_counts"],
                      "sequence_counts": report["sequence_counts"],
                      "observation_regime": report["regime"],
                      "observation": report["observation"],
                      "content_digest": report["content_digest"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
