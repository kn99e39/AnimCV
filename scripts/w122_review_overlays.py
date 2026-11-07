#!/usr/bin/env python3
"""W122 blind review aid (written AFTER render and BEFORE any rating; reads ONLY the blind real-time MP4s and the blind trail
plots, never render/sealed/mapping.json).  For every blind pair, every decoded frame of the 30 fps blind MP4 in order, the
world_front and char_side views as an A/B overlay: red channel = A, green+blue = B (identical pixels grey; A-only red,
B-only cyan), plus the source and U tiles of the first frame and the A/B/U trail plot.  AGENT REVIEW material only."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

R = Path(sys.argv[1] if len(sys.argv) > 1 else "C:/Users/dna10/animcv-w122/results/render")
OUT = R / "review_overlays"
SIZE, HDR, CELL, COLS = 320, 28, 150, 13
VIEWS = {"world_front": 0, "char_side": 2}


def overlay(im, row):
    a = im[HDR + row * SIZE:HDR + (row + 1) * SIZE, 2 * SIZE:3 * SIZE]
    b = im[HDR + row * SIZE:HDR + (row + 1) * SIZE, 3 * SIZE:4 * SIZE]
    ga, gb = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY), cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
    o = np.stack([gb, gb, ga], -1)          # BGR: blue=B, green=B, red=A
    c = o[40:280, 40:280]
    return cv2.resize(c, (CELL, CELL), interpolation=cv2.INTER_AREA)


def main():
    OUT.mkdir(exist_ok=False)
    rr = json.loads((R / "render_report.json").read_text())
    pid_of = {v["pair_id"]: rid for rid, v in rr["videos"].items() if "pair_id" in v}
    made = {}
    # pin frames only (from the frozen declaration's selection / W121 holds via the build records; no outcome field is read)
    pins = {}
    for p, rid in pid_of.items():
        sid, w = rid.split("_")[0], rid.split("_")[1]
        b = json.loads((R.parent / "build_r1/build_report.json").read_text()) if not pins else b
        pins[p] = set(b["sources"][sid]["records"][rid]["pin_frames"])
    for p in rr["pair_ids_review_order"]:
        frames = rr["videos"][pid_of[p]]["frames"]
        cap = cv2.VideoCapture(str(R / "videos" / f"{p}_blind_realtime_30fps.mp4"))
        dec = []
        while True:
            ok, im = cap.read()
            if not ok:
                break
            dec.append(im)
        cap.release()
        rows = math.ceil(len(dec) / COLS)
        blk = rows * (CELL + 14) + 22
        trails = cv2.imread(str(R / "review_sheets" / f"{p}_trails.png"))
        tw = COLS * CELL
        trails = cv2.resize(trails, (tw, int(trails.shape[0] * tw / trails.shape[1])))
        H = 30 + 2 * blk + trails.shape[0]
        sh = np.full((H, tw, 3), 252, np.uint8)
        cv2.putText(sh, f"{p}: A/B overlay of every decoded frame of the 30 fps blind MP4 (red = A only, cyan = B only, grey = same)",
                    (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 1, cv2.LINE_AA)
        y0 = 30
        for vn, row in VIEWS.items():
            cv2.putText(sh, vn, (6, y0 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 40, 160), 1, cv2.LINE_AA)
            for i, im in enumerate(dec):
                r_, c_ = divmod(i, COLS)
                y, x = y0 + 22 + r_ * (CELL + 14), c_ * CELL
                sh[y:y + CELL, x:x + CELL] = overlay(im, row)
                pin = frames[i] in pins[p]
                cv2.putText(sh, f"f{frames[i]}{' PIN' if pin else ''}", (x + 3, y + CELL + 11), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                            (30, 60, 200) if pin else (50, 50, 50), 1, cv2.LINE_AA)
            y0 += blk
        sh[y0:y0 + trails.shape[0]] = trails
        cv2.imwrite(str(OUT / f"{p}_overlay.jpg"), sh, [cv2.IMWRITE_JPEG_QUALITY, 90])
        made[p] = len(dec)
    (OUT / "overlays.json").write_text(json.dumps({"made": made, "reads_mapping": False}, indent=1))
    print(made)


if __name__ == "__main__":
    main()
