#!/usr/bin/env python3
"""Worklog 70 qualitative review: owner-case crop + 2D segment + A0 / A1 attention heatmaps per pair.

Qualitative diagnostics only.  The encoder crop had no overlay; segments and
heatmaps are drawn on this panel only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from framepose.bank import load_bank
from framepose.crops import crop_box, render_crop
from framepose.skeleton_attention_depth import ARM_JOINT_INDEX, GRID, PAIRS, RESOLUTION, joint_token_coordinates

CELL = RESOLUTION


def heat(crop: np.ndarray, weights: np.ndarray) -> np.ndarray:
    w = weights.reshape(GRID, GRID)
    w = w / max(w.max(), 1e-12)
    up = cv2.resize(w.astype(np.float32), (CELL, CELL), interpolation=cv2.INTER_NEAREST)
    colour = cv2.applyColorMap((up * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)[:, :, ::-1]
    return (0.45 * crop + 0.55 * colour).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("bank", "image_root", "run_dir", "cache_dir", "out_dir"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    args = parser.parse_args()
    bank = load_bank(args.bank)
    report = json.loads((args.run_dir / "report.json").read_text())
    rows = {(r["sequence_id"], r["frame_index"]): r for r in json.loads((args.run_dir / "rows_heldout.json").read_text())}
    attention = np.load(args.run_dir / "owner_attention.npz")
    boxes = np.load(args.cache_dir / "readout.npz")["crop_box_xy_side"]
    font = ImageFont.load_default()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    seen = {}
    for label, case in report["owner_cases"].items():
        r = rows.get((case["sequence_id"], case["frame_index"]))
        if r is not None:
            seen.setdefault(r["position"], []).append(label)
    for position, labels in seen.items():
        sample = bank.samples[position]
        row = rows[(sample.sequence_id, sample.frame_index)]
        data = (args.image_root / sample.image_reference.relative_path).read_bytes()
        rgb = cv2.cvtColor(cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        box = crop_box(bank.arrays["input_2d"][position], bank.arrays["input_valid"][position], sample.image_size)
        if not np.allclose([box.x, box.y, box.side], boxes[position]):
            raise SystemExit("crop box differs from the Worklog 68 cache")
        crop = render_crop(rgb, box, RESOLUTION)
        coords = joint_token_coordinates(bank.arrays["input_2d"][position], sample.image_size, boxes[position])
        a0, a1 = attention[f"{position}_a0"], attention[f"{position}_a1"]
        canvas = Image.new("RGB", (3 * CELL, 3 * CELL + 120), (20, 20, 20))
        draw = ImageDraw.Draw(canvas)
        for k, (field, ia, ib, seg) in enumerate(PAIRS):
            pa = (coords[ARM_JOINT_INDEX[ia]] + 0.5) * 32
            pb = (coords[ARM_JOINT_INDEX[ib]] + 0.5) * 32
            cells = [crop.copy(), heat(crop, a0[k]), heat(crop, a1[k])]
            for c, image in enumerate(cells):
                tile = Image.fromarray(image)
                d = ImageDraw.Draw(tile)
                d.line([tuple(pa), tuple(pb)], fill=(0, 255, 120), width=3)
                d.text((6, 6), f"{seg} | {['crop + 2D segment', 'A0 unbiased attention', 'A1 skeleton-biased attention'][c]}",
                       fill=(255, 255, 255), font=font)
                canvas.paste(tile, (c * CELL, k * CELL))
        lines = [f"W70 attention review | {' = '.join(labels)} | {sample.sample_id}",
                 "f per segment  GT / C1(W69) / A0 / A1 / A1-shuffled / H0 / D1:"]
        for seg in ("upper", "lower", "chain"):
            lines.append(f"  {seg:6s} " + " / ".join(f"{row[f'{seg}_f_{m}']:+.2f}" for m in ("gt", "c1", "a0", "a1", "a1_shuffled", "h0", "d1"))
                         + "   full err C1/A0/A1: " + " / ".join(f"{row[f'{seg}_{m}_raw_plane']:.1f}" for m in ("c1", "a0", "a1")))
        lines.append("Heatmaps are qualitative diagnostics only (normalized per map).")
        for i, line in enumerate(lines):
            draw.text((8, 3 * CELL + 8 + 16 * i), line, fill=(235, 235, 235), font=font)
        canvas.save(args.out_dir / f"{labels[0]}_{sample.sequence_id.split(':')[1]}_{sample.frame_index}.png")
    print(json.dumps({"rendered": [v[0] for v in seen.values()]}))


if __name__ == "__main__":
    main()
