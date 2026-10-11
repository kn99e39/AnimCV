#!/usr/bin/env python3
"""Worklog 73 qualitative review sheets (images only).

A. BEDLAM external samples: the FramePose 448 px crop with the detector's
   left shoulder/elbow/wrist (the model input; drawn only here) next to a side
   view (forward depth Y right, up Z) of the GT left arm, with GT f per segment.
B. 3DPW held-out rows where E1 most improved / most regressed over E0
   (forearm + chain |f| error): crop with detector S/E/W and a side view of the
   GT arm with the E0 and E1 reconstructed forearm/chain directions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from framepose.arm_depth_probe import SEGMENT_NAMES, forward_targets, observed_plane, reconstruct_direction, segment_vectors
from framepose.bank import load_bank
from framepose.bedlam_supervision import ExternalBank
from framepose.learned_vision_sensor import RESOLUTION, crop_for_vision

ARM_IDX = (11, 12, 13)
COLORS = {"gt": (40, 40, 40), "e0": (220, 120, 0), "e1": (0, 140, 220), "det": (255, 40, 40)}
PANEL = 448
SIDE = 300


def crop_with_detector(rgb, inp, valid, size):
    crop, box = crop_for_vision(rgb, inp, valid, size)
    img = Image.fromarray(crop)
    d = ImageDraw.Draw(img)
    pts = []
    for j in ARM_IDX:
        if valid[j]:
            u = (inp[j, 0] * size[0] - box.x) / box.side * RESOLUTION
            v = (inp[j, 1] * size[1] - box.y) / box.side * RESOLUTION
            pts.append((u, v))
            d.ellipse([u - 5, v - 5, u + 5, v + 5], outline=COLORS["det"], width=3)
    if len(pts) > 1:
        d.line(pts, fill=COLORS["det"], width=2)
    return img


def side_view(target, extra=None, title=""):
    """Side view: forward depth Y to the right, up Z upwards; shoulder at the centre."""
    img = Image.new("RGB", (SIDE, PANEL), "white")
    d = ImageDraw.Draw(img)
    cx, cy, scale = SIDE // 2, PANEL // 2, 330
    d.line([(cx, 20), (cx, PANEL - 40)], fill=(220, 220, 220))
    d.line([(20, cy), (SIDE - 20, cy)], fill=(220, 220, 220))
    d.text((SIDE - 95, cy + 4), "farther ->", fill=(120, 120, 120))
    d.text((8, cy + 4), "<- nearer", fill=(120, 120, 120))
    s, e, w = (target[j] - target[11] for j in ARM_IDX)

    def p(v):
        return (cx + v[1] * scale, cy - v[2] * scale)
    d.line([p(s), p(e), p(w)], fill=COLORS["gt"], width=4)
    for v in (s, e, w):
        d.ellipse([p(v)[0] - 4, p(v)[1] - 4, p(v)[0] + 4, p(v)[1] + 4], fill=COLORS["gt"])
    if extra:
        for name, vectors in extra.items():
            fore, chain = vectors
            length_fore, length_chain = np.linalg.norm(w - e), np.linalg.norm(w - s)
            d.line([p(e), p(e + fore * length_fore)], fill=COLORS[name], width=3)
            d.line([p(s), p(s + chain * length_chain)], fill=COLORS[name], width=2)
    d.text((8, 6), title, fill=(0, 0, 0))
    return img


def sheet(panels, cols, path, header):
    w, h = panels[0].size
    out = Image.new("RGB", (cols * w, 30 + ((len(panels) + cols - 1) // cols) * h), "white")
    ImageDraw.Draw(out).text((8, 8), header, fill=(0, 0, 0))
    for i, p in enumerate(panels):
        out.paste(p, ((i % cols) * w, 30 + (i // cols) * h))
    out.save(path)


def labelled(crop, side, lines):
    img = Image.new("RGB", (PANEL + SIDE, PANEL + 18 * len(lines) + 6), "white")
    img.paste(crop, (0, 0))
    img.paste(side, (PANEL, 0))
    d = ImageDraw.Draw(img)
    for i, line in enumerate(lines):
        d.text((6, PANEL + 4 + 18 * i), line, fill=(0, 0, 0))
    return img


def load_rgb(path):
    return cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("external_train", "external_val", "bedlam_images", "framebank", "framebank_images", "run_dir", "out_dir"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for split, path, count in (("train", args.external_train, 12), ("validation", args.external_val, 6)):
        bank = ExternalBank.load(path)
        f, m = forward_targets(bank.arrays["target_3d"], bank.arrays["target_valid"])
        picks = np.unique(np.linspace(0, len(bank) - 1, count).round().astype(int))
        panels = []
        for p in picks:
            s = bank.samples[p]
            rgb = load_rgb(args.bedlam_images / s.image_reference.relative_path)
            crop = crop_with_detector(rgb, bank.arrays["input_2d"][p], bank.arrays["input_valid"][p], s.image_size)
            side = side_view(bank.arrays["target_3d"][p], title="GT left arm (side view)")
            fs = "  ".join(f"{seg}:{f[p, k]:+.2f}" if m[p, k] else f"{seg}:n/a" for k, seg in enumerate(SEGMENT_NAMES))
            panels.append(labelled(crop, side, [s.sequence_id.split(":", 1)[1][-60:],
                                                f"subject {s.detail.get('subject')}  GT f  {fs}"]))
        sheet(panels, 3, args.out_dir / f"A_bedlam_{split}_samples.png",
              f"BEDLAM {split}: red = detector S/E/W (model input); right = GT arm side view (farther = right)")

    bank = load_bank(args.framebank)
    rows = json.loads((args.run_dir / "rows_heldout.json").read_text())
    change = np.array([sum(abs(r[f"{seg}_f_e1"] - r[f"{seg}_f_gt"]) - abs(r[f"{seg}_f_e0"] - r[f"{seg}_f_gt"])
                           for seg in ("lower", "chain")) for r in rows])
    order = np.argsort(change)
    for name, idx in (("improved", order[:9]), ("regressed", order[::-1][:9])):
        panels = []
        for i in idx:
            r = rows[i]
            p = r["position"]
            s = bank.samples[p]
            rgb = load_rgb(args.framebank_images / s.image_reference.relative_path)
            crop = crop_with_detector(rgb, bank.arrays["input_2d"][p], bank.arrays["input_valid"][p], s.image_size)
            planes, _ = observed_plane(bank.arrays["input_2d"][p:p + 1], np.asarray([s.image_size], float))
            extra = {m_: (reconstruct_direction(np.array([r[f"lower_f_{m_}"]]), planes["lower"])[0],
                          reconstruct_direction(np.array([r[f"chain_f_{m_}"]]), planes["chain"])[0]) for m_ in ("e0", "e1")}
            side = side_view(bank.arrays["target_3d"][p], extra, "GT (black)  E0 (orange)  E1 (blue)")
            lines = [f"{r['sequence_id']}#{r['frame_index']}  split {s.split}",
                     "  ".join(f"{seg}: gt {r[f'{seg}_f_gt']:+.2f} e0 {r[f'{seg}_f_e0']:+.2f} e1 {r[f'{seg}_f_e1']:+.2f}"
                               for seg in ("lower", "chain"))]
            panels.append(labelled(crop, side, lines))
        sheet(panels, 3, args.out_dir / f"B_3dpw_heldout_e1_{name}.png",
              f"3DPW held-out: E1 most {name} vs E0 (forearm+chain |f| error). red = detector S/E/W; forearm thick, chain thin")
    print(json.dumps({"out": str(args.out_dir), "files": sorted(p.name for p in args.out_dir.glob("*.png"))}))


if __name__ == "__main__":
    main()
