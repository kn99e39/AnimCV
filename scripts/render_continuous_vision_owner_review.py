#!/usr/bin/env python3
"""Worklog 69 qualitative review: owner-case person crops with continuous forward fractions.

The S/E/W dots are drawn only on this review panel; the encoder input had no overlay.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from framepose.bank import load_bank
from framepose.contract import JOINT_INDEX
from framepose.learned_vision_sensor import ARM, RESOLUTION, crop_for_vision

SIGN_TEXT = {1: "first closer", -1: "second closer", 0: "unclear/0"}
STATE_SIGN = {"FIRST_CLOSER": 1, "SECOND_CLOSER": -1, "UNCLEAR": 0, "UNKNOWN": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("bank", "image_root", "sensor_dir", "out_dir"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    args = parser.parse_args()
    bank = load_bank(args.bank)
    report = json.loads((args.sensor_dir / "report.json").read_text())
    rows = {(r["sequence_id"], r["frame_index"]): r for r in json.loads((args.sensor_dir / "rows_heldout.json").read_text())}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    font = ImageFont.load_default()
    done = {}
    for label, case in report["owner_cases"].items():
        row = rows.get((case["sequence_id"], case["frame_index"]))
        if row is None:
            continue
        done.setdefault(row["position"], []).append(label)
    for position, labels in done.items():
        case = report["owner_cases"][labels[0]]
        row = rows[(case["sequence_id"], case["frame_index"])]
        sample = bank.samples[position]
        data = (args.image_root / sample.image_reference.relative_path).read_bytes()
        rgb = cv2.cvtColor(cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        crop, box = crop_for_vision(rgb, bank.arrays["input_2d"][position], bank.arrays["input_valid"][position],
                                    sample.image_size)
        image = Image.fromarray(crop)
        draw = ImageDraw.Draw(image)
        w, h = sample.image_size
        for letter, joint in zip("SEW", ARM):
            x, y = bank.arrays["input_2d"][position, JOINT_INDEX[joint], :2]
            u = (x * w - box.x) / box.side * RESOLUTION
            v = (y * h - box.y) / box.side * RESOLUTION
            draw.ellipse((u - 4, v - 4, u + 4, v + 4), outline=(255, 64, 64), width=2)
            draw.text((u + 6, v - 12), letter, fill=(255, 64, 64), font=font)
        lines = [f"W69 continuous visual arm depth review | {' = '.join(labels)}",
                 f"{sample.sample_id} | dots are review-only (encoder input had no overlay)",
                 "seg    f: GT     C0      C1      C1shuf  H0      D1     | full err C0 / C1 / H0 / D1 (deg)"]
        for seg, pair in (("upper", "S-E"), ("lower", "E-W"), ("chain", "S-W")):
            fs = "".join(f"{row[f'{seg}_f_{m}']:+.2f}   " for m in ("gt", "c0", "c1", "c1_shuffled", "h0", "d1"))
            errs = " / ".join(f"{row[f'{seg}_{m}_raw_plane']:.1f}" for m in ("c0", "c1", "h0", "d1"))
            lines.append(f"{pair}    {fs}| {errs}")
        lines.append("f = unit(segment).Y; +Y = farther from the camera.")
        panel = Image.new("RGB", (max(RESOLUTION, 760), RESOLUTION + 16 * len(lines) + 16), (20, 20, 20))
        panel.paste(image, (0, 0))
        d = ImageDraw.Draw(panel)
        for i, line in enumerate(lines):
            d.text((8, RESOLUTION + 8 + 16 * i), line, fill=(235, 235, 235), font=font)
        panel.save(args.out_dir / f"{labels[0]}_{sample.sequence_id.split(':')[1]}_{sample.frame_index}.png")
    print(json.dumps({"rendered": [v[0] for v in done.values()]}))


if __name__ == "__main__":
    main()
