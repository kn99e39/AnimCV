#!/usr/bin/env python3
"""Worklog 67 qualitative review: owner-case advisor crops with GT / H0 / DA / VLM orderings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def _order(value, first, second):
    if value is None:
        return "n/a"
    if value > 0:
        return f"{first} closer"
    if value < 0:
        return f"{second} closer"
    return "tie"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vlm-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    analysis = json.loads((args.vlm_dir / "analysis" / "analysis.json").read_text())
    args.out_dir.mkdir(parents=True, exist_ok=True)
    font = ImageFont.load_default()
    pairs = (("upper", "S", "E", "shoulder_elbow"), ("lower", "E", "W", "elbow_wrist"),
             ("chain", "S", "W", "shoulder_wrist"))
    seen = {}
    for label, case in analysis["owner_cases"].items():
        if "excluded" in case:
            continue
        key = (case["sequence_id"], case["frame_index"])
        seen.setdefault(key, []).append(label)
    for (sid, frame), labels in seen.items():
        case = analysis["owner_cases"][labels[0]]
        crop = Image.open(args.vlm_dir / "owner_images" / f"{sid.split(':')[1]}_{frame}.png").convert("RGB")
        lines = [f"W67 VLM near/far review | {' = '.join(labels)}",
                 f"{sid} #{frame} | runtime H0-vs-DA disagreement frame: {case['primary']}",
                 "pair    GT         H0         DepthAny   VLM(raw)       V0 source        full err H0/D1/V0"]
        for seg, a, b, field in pairs:
            v = case[seg]
            vlm = case["vlm"][field]
            vlm_text = {"FIRST_CLOSER": f"{a} closer", "SECOND_CLOSER": f"{b} closer"}.get(vlm, vlm)
            errs = "/".join(f"{x:.1f}" for x in v["full_h0_d1_v0"])
            lines.append(f"{a}-{b}     {_order(v['gt_f'], a, b):10s} {_order(v['h0_f'], a, b):10s} "
                         f"{_order(v['da_relation'], a, b):10s} {vlm_text:14s} {v['v0_source']:16s} {errs}")
        lines.append("S/E/W = benchmark detector points; f>0 means the distal point is farther (first closer).")
        panel = Image.new("RGB", (max(crop.width, 760), crop.height + 16 * len(lines) + 16), (20, 20, 20))
        panel.paste(crop, (0, 0))
        draw = ImageDraw.Draw(panel)
        for i, line in enumerate(lines):
            draw.text((8, crop.height + 8 + 16 * i), line, fill=(235, 235, 235), font=font)
        panel.save(args.out_dir / f"{labels[0]}_{sid.split(':')[1]}_{frame}.png")
    print(json.dumps({"rendered": [f"{v[0]}" for v in seen.values()]}))


if __name__ == "__main__":
    main()
