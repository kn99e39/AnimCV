#!/usr/bin/env python3
"""Worklog 68 qualitative review: owner-case person crops (as seen by the frozen encoder) with orderings.

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
    args.out_dir.mkdir(parents=True, exist_ok=True)
    font = ImageFont.load_default()
    done = {}
    for label, case in report["owner_cases"].items():
        row = case["row"]
        if row is None:
            continue
        done.setdefault(row["position"], []).append(label)
    for position, labels in done.items():
        row = report["owner_cases"][labels[0]]["row"]
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
        lines = [f"W68 learned vision near/far review | {' = '.join(labels)}",
                 f"{sample.sample_id} | dots are review-only (encoder input had no overlay)",
                 "pair  GT             H0             D1             W67 VLM        G0             G1             V1 source         full err H0/D1/V0/V1"]
        for seg, pair in (("upper", "S-E"), ("lower", "E-W"), ("chain", "S-W")):
            gt = {0: 1, 1: -1, 2: 0}[row[f"{seg}_label"]]
            cells = [SIGN_TEXT[gt], SIGN_TEXT[int(np.sign(row[f"{seg}_f_h0"]))], SIGN_TEXT[int(np.sign(row[f"{seg}_f_d1"]))],
                     SIGN_TEXT[STATE_SIGN[row[f"{seg}_vlm"]]], SIGN_TEXT[STATE_SIGN[row[f"{seg}_g0_state"]]],
                     SIGN_TEXT[STATE_SIGN[row[f"{seg}_g1_state"]]]]
            errs = "/".join(f"{row[f'{seg}_{m}_raw_plane']:.1f}" for m in ("h0", "d1", "v0", "v1"))
            lines.append(f"{pair}   " + "".join(f"{c:15s}" for c in cells) + f"{row[f'{seg}_v1_source']:18s}{errs}")
        lines.append("'first closer' = proximal point nearer the camera (GT f > sin 10 deg).")
        panel = Image.new("RGB", (max(RESOLUTION, 1000), RESOLUTION + 16 * len(lines) + 16), (20, 20, 20))
        panel.paste(image, (0, 0))
        d = ImageDraw.Draw(panel)
        for i, line in enumerate(lines):
            d.text((8, RESOLUTION + 8 + 16 * i), line, fill=(235, 235, 235), font=font)
        panel.save(args.out_dir / f"{labels[0]}_{sample.sequence_id.split(':')[1]}_{sample.frame_index}.png")
    print(json.dumps({"rendered": [v[0] for v in done.values()]}))


if __name__ == "__main__":
    main()
