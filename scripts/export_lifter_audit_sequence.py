#!/usr/bin/env python3
"""Export a contiguous ground-truth + predicted skeleton window for video review.

``scripts/export_lifter_audit_frames.py`` answers "does this specific frame
look like a flip"; this script answers "does the motion around it look
wrong" by exporting every frame in a chosen window of one holdout action as
both reference and estimate skeletons, ready for
``scripts/render_lifter_audit_video.py`` to render as a single overlaid MP4
(no retarget, no rig, no mesh -- just the two skeletons superimposed).

Usage:
  python3 scripts/export_lifter_audit_sequence.py \
    --checkpoint reports/direct_mix.pth --holdout /data/3dpw/prepared/holdout.json \
    --action 3dpw:downtown_stairs_00:actor0 --start-frame 200 --end-frame 350 \
    --out audit/stairs_window.json

  blender --background --python scripts/render_lifter_audit_video.py -- \
    --sequence audit/stairs_window.json --out audit/stairs_window.mp4
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from training.temporal_lifter import H36M_NAMES, _arrays, _frame_metadata, _model, _predict_batched, _torch, load_dataset

SCHEMA = "animcv_lifter_audit_sequence_v1"


def _action_bounds(metadata: list[dict[str, str | None]], action: str) -> tuple[int, int]:
    """Return the inclusive [start, end] global index range for one action.

    Frames for a single 3DPW/AMASS sequence are stored contiguously, so this
    also verifies that assumption instead of silently sampling a non-clip
    window if a dataset ever interleaves sequences.
    """
    indices = [index for index, meta in enumerate(metadata) if meta.get("action") == action]
    if not indices:
        raise ValueError(f"no frames found for action {action!r}")
    start, end = indices[0], indices[-1]
    if indices != list(range(start, end + 1)):
        raise ValueError(f"action {action!r} frames are not contiguous in this dataset; cannot take a clip window")
    return start, end


def _resolve_window(metadata: list[dict[str, str | None]], action: str, start_frame: int, end_frame: int | None) -> tuple[int, int]:
    action_start, action_end = _action_bounds(metadata, action)
    action_length = action_end - action_start + 1
    local_end = end_frame if end_frame is not None else action_length - 1
    if start_frame < 0 or local_end < start_frame or local_end >= action_length:
        raise ValueError(
            f"requested window [{start_frame}, {local_end}] is outside action {action!r} "
            f"({action_length} frames, valid local range [0, {action_length - 1}])"
        )
    return action_start + start_frame, action_start + local_end


def _character_points(row: np.ndarray) -> dict[str, list[float]]:
    return {name: [float(value) for value in row[index]] for index, name in enumerate(H36M_NAMES)}


def _build_sequence_export(
    prediction: np.ndarray, targets: np.ndarray, global_start: int, global_end: int, action: str, fps: float,
) -> dict[str, Any]:
    frames = [
        {
            "frame_index": local_index,
            "reference": _character_points(targets[global_index]),
            "estimate": _character_points(prediction[global_index]),
        }
        for local_index, global_index in enumerate(range(global_start, global_end + 1))
    ]
    return {"schema": SCHEMA, "action": action, "fps": fps, "frames": frames}


def _run(checkpoint: Path, holdout: Path, device: str):
    torch, nn = _torch()
    ck = torch.load(checkpoint, map_location=device, weights_only=True)
    dataset = load_dataset(holdout)
    inputs, targets, valid, offsets = _arrays(
        dataset, int(ck["window"]), coordinate_normalization=ck.get("input_coordinate_normalization", "image_v1"),
    )
    model = _model(nn, int(ck["channels"]), ck.get("architecture", "legacy_tcn_v1")).to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    x = torch.as_tensor(inputs, dtype=torch.float32, device=device)
    with torch.no_grad():
        prediction = _predict_batched(
            model, x, torch.as_tensor(offsets, dtype=torch.long, device=device), 1024, device.startswith("cuda"),
        ).cpu().numpy()
    metadata = _frame_metadata(dataset)
    fps = 0.0
    for sequence in dataset.get("sequences", [{"frames": dataset["frames"]}]):
        source_fps = sequence.get("source_fps")
        if source_fps:
            fps = float(source_fps)
            break
    return prediction, targets, metadata, fps


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--holdout", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--action", required=True)
    parser.add_argument("--start-frame", type=int, default=0, help="0-based local frame index within the action")
    parser.add_argument("--end-frame", type=int, default=None, help="inclusive; defaults to the action's last frame")
    parser.add_argument("--fps", type=float, default=None, help="overrides the dataset's recorded source FPS")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    prediction, targets, metadata, dataset_fps = _run(args.checkpoint, args.holdout, args.device)
    global_start, global_end = _resolve_window(metadata, args.action, args.start_frame, args.end_frame)
    fps = args.fps if args.fps is not None else dataset_fps
    if not fps:
        raise ValueError("dataset has no recorded source FPS; pass --fps explicitly")

    export = _build_sequence_export(prediction, targets, global_start, global_end, args.action, fps)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(export, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(json.dumps({
        "action": args.action,
        "frame_count": len(export["frames"]),
        "fps": fps,
        "global_index_range": [global_start, global_end],
        "out": str(args.out),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
