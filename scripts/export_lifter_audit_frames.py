#!/usr/bin/env python3
"""Export ground-truth and predicted skeletons for static-frame Blender audit.

A lifter evaluation report reduces every holdout frame to a handful of
aggregate numbers (PA-MPJPE, yaw P95, hinge flip rate). Those numbers cannot
say whether a flagged frame actually *looks* like a flip, or whether a clip
ranked clean by its aggregate flip rate still has an isolated bad frame. This
script re-runs a checkpoint over a holdout dataset, ranks the frames of each
requested action by hinge-direction error and root-yaw error, and writes the
worst-hinge, worst-yaw, and a mid-error frame per action as ground-truth and
predicted skeletons in the ``character_points`` schema that
``scripts/render_3d_audit.py`` already renders. No retarget step or Blender
``.blend`` is required, so the check runs in seconds.

Usage:
  python3 scripts/export_lifter_audit_frames.py \
    --checkpoint reports/direct_mix.pth --holdout /data/3dpw/prepared/holdout.json \
    --actions 3dpw:downtown_stairs_00:actor0,3dpw:downtown_bar_00:actor1 \
    --out-gt audit/gt.json --out-pred audit/pred.json --out-picks audit/picks.json

  blender --background --python scripts/render_3d_audit.py -- \
    --root-motion audit/gt.json --out-dir audit/render_gt --frames <global indices>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from training.temporal_lifter import (
    H36M_NAMES, _arrays, _bend_direction, _frame_metadata, _hinge_errors, _model,
    _predict_batched, _root_yaw_error_degrees, _torch, load_dataset,
)

SCHEMA = "animcv_lifter_static_frame_audit_v1"
PICK_LABELS = ("worst_hinge", "worst_yaw", "mid")


def _frame_diagnostics(
    prediction: np.ndarray, targets: np.ndarray, valid: np.ndarray, metadata: list[dict[str, str | None]],
) -> list[dict[str, Any]]:
    """Per-frame worst-hinge-chain error, any-chain flip flag, and yaw error.

    Reuses the exact ``_hinge_errors``/``_root_yaw_error_degrees`` functions the
    official evaluation report calls, so a frame flagged here is guaranteed to
    be flagged the same way in ``reports/*.json``.
    """
    diagnostics = []
    for index, (estimate, reference, frame_valid, meta) in enumerate(zip(prediction, targets, valid, metadata)):
        hinges = _hinge_errors(estimate, reference, frame_valid)
        yaw = _root_yaw_error_degrees(estimate, reference, frame_valid)
        diagnostics.append({
            "global_index": index,
            "action": meta.get("action"),
            "worst_hinge_deg": max((hinge["error_degrees"] for hinge in hinges), default=0.0),
            "flipped": any(hinge["flipped"] for hinge in hinges),
            "yaw_error_deg": yaw if yaw is not None else None,
        })
    return diagnostics


def _select_picks(
    diagnostics: list[dict[str, Any]], actions: list[str] | None,
) -> dict[str, dict[str, dict[str, Any]]]:
    """Pick the worst-hinge, worst-yaw, and a mid-error frame per action.

    ``actions`` narrows the export to specific clips; omit it to cover every
    action present in the dataset (fine for a holdout, likely too much for a
    full training set).
    """
    by_action: dict[str, list[dict[str, Any]]] = {}
    for row in diagnostics:
        action = row["action"] or "unknown"
        if actions is not None and action not in actions:
            continue
        by_action.setdefault(action, []).append(row)

    picks: dict[str, dict[str, dict[str, Any]]] = {}
    for action, rows in by_action.items():
        by_hinge = max(rows, key=lambda row: row["worst_hinge_deg"])
        yawed = [row for row in rows if row["yaw_error_deg"] is not None]
        by_yaw = max(yawed, key=lambda row: row["yaw_error_deg"]) if yawed else by_hinge
        mid = sorted(rows, key=lambda row: row["worst_hinge_deg"])[len(rows) // 2]
        picks[action] = {"worst_hinge": by_hinge, "worst_yaw": by_yaw, "mid": mid}
    return picks


def _character_points(row: np.ndarray) -> dict[str, list[float]]:
    return {name: [float(value) for value in row[index]] for index, name in enumerate(H36M_NAMES)}


def _build_export(
    picks: dict[str, dict[str, dict[str, Any]]], targets: np.ndarray, prediction: np.ndarray,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    seen: set[int] = set()
    gt_frames, pred_frames = [], []
    for action, group in picks.items():
        for label in PICK_LABELS:
            row = group[label]
            index = row["global_index"]
            if index in seen:
                continue
            seen.add(index)
            tag = f"{action.replace(':', '_')}_{label}_f{index}"
            gt_frames.append({"frame_index": index, "tag": tag, "character_points": _character_points(targets[index])})
            pred_frames.append({"frame_index": index, "tag": tag, "character_points": _character_points(prediction[index])})
    gt_frames.sort(key=lambda frame: frame["frame_index"])
    pred_frames.sort(key=lambda frame: frame["frame_index"])
    return (
        {"schema": SCHEMA, "frames": gt_frames},
        {"schema": SCHEMA, "frames": pred_frames},
        {"schema": SCHEMA, "picks": picks},
    )


def _run(checkpoint: Path, holdout: Path, device: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, str | None]]]:
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
    return prediction, targets, valid, _frame_metadata(dataset)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--holdout", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--actions", help="comma-separated action labels to export; omit for every action present")
    parser.add_argument("--out-gt", required=True, type=Path)
    parser.add_argument("--out-pred", required=True, type=Path)
    parser.add_argument("--out-picks", type=Path)
    args = parser.parse_args()

    actions = [action for action in args.actions.split(",") if action] if args.actions else None
    prediction, targets, valid, metadata = _run(args.checkpoint, args.holdout, args.device)
    diagnostics = _frame_diagnostics(prediction, targets, valid, metadata)
    picks = _select_picks(diagnostics, actions)
    if not picks:
        raise ValueError(f"no frames matched --actions {actions!r}; check the holdout's action labels")

    gt_export, pred_export, picks_export = _build_export(picks, targets, prediction)
    args.out_gt.parent.mkdir(parents=True, exist_ok=True)
    args.out_pred.parent.mkdir(parents=True, exist_ok=True)
    args.out_gt.write_text(json.dumps(gt_export, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.out_pred.write_text(json.dumps(pred_export, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.out_picks:
        args.out_picks.parent.mkdir(parents=True, exist_ok=True)
        args.out_picks.write_text(json.dumps(picks_export, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    frame_indices = sorted({row["global_index"] for group in picks.values() for row in group.values()})
    print(json.dumps({
        "actions": sorted(picks),
        "exported_frame_count": len(frame_indices),
        "frame_indices": frame_indices,
        "out_gt": str(args.out_gt),
        "out_pred": str(args.out_pred),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
