#!/usr/bin/env python3
"""W127 SUPPLEMENTARY visual (written AFTER the declaration; not a gate, not in the declared instrument list).

Why: the declared Workbench A/B renders the rig's skinned mesh; the stock exporter writes an armature-only FBX, so its
re-import has no mesh and its panel is blank.  This script draws the bone skeletons directly from the EVALUATOR's
fresh-process readbacks already on disk (no new Blender evaluation, no new numbers): original (white) vs imported
(red = stock export+import, green = opt-in package + adapter), front orthographic projection (scene X right, Z up),
over the declared quarter-frame window (centred on the stock path's worst subframe sample) and over the pin window.

  python scripts/w127_supplementary_skeleton_visual.py --real-dir <results/real> --visual-dir <results/visual> --out-dir NEW
"""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


def load(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def samples(rb):
    out = {(f["blender_frame"], 0.0): f for f in rb["frames"]}
    out.update({(s["blender_frame"], s["subframe"]): s for s in rb["subframes"]})
    return out


def draw(img, rec, bones, proj, col, thick):
    for b in bones:
        h, t = proj(rec["world"][b]["head"]), proj(rec["world"][b]["tail"])
        cv2.line(img, h, t, col, thick, cv2.LINE_AA)


def lab(img, txt, y, col=(255, 255, 255)):
    cv2.putText(img, txt, (6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, txt, (6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real-dir", required=True)
    ap.add_argument("--visual-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=False)
    res = {}
    for src in ("S1", "S3"):
        d = Path(a.real_dir) / src / "w126v"
        orig = load(d / "verify" / "original_readback.json")
        stock = load(d / "stock_eval" / "imported_readback.json")
        optin = load(d / "verify" / "imported_readback.json")
        if sha(d / "verify" / "original_readback.json") != load(d / "verify" / "fbx_verify_report.json")["original_readback_sha256"]:
            raise SystemExit("readback hash mismatch")
        bones = orig["bones"]
        so, ss, sp = samples(orig), samples(stock), samples(optin)
        vis = load(Path(a.visual_dir) / "visual.json")[src]
        lo, hi = vis["quarter_window_action_frames"]
        pin = load(d / "verify" / "fbx_verify_report.json")["pin"]["action_frames"]
        windows = {"worst_stock_window": [(f, s) for f in range(lo, hi + 1) for s in (0.0, 0.25, 0.5, 0.75)],
                   "pin_window": [(f, s) for f in range(min(pin) - 2, max(pin) + 2) for s in (0.0, 0.25, 0.5, 0.75)]}
        pts = np.array([so[k]["world"][b][p] for k in sum(windows.values(), []) for b in bones for p in ("head", "tail")])
        c = (pts.min(0) + pts.max(0)) / 2
        ext = float(max(pts[:, 0].ptp(), pts[:, 2].ptp())) * 1.2 + 0.2
        S = 360

        def proj(v):
            return (int(S / 2 + (v[0] - c[0]) / ext * S), int(S / 2 - (v[2] - c[2]) / ext * S))
        res[src] = {}
        for wname, keys in windows.items():
            frames = []
            for k in keys:
                if k not in so or k not in ss or k not in sp:
                    continue
                tiles = []
                for name, other, col in (("ORIGINAL .blend", None, None), ("STOCK FBX (red) vs original", ss, (60, 60, 255)),
                                         ("OPT-IN (green) vs original", sp, (60, 220, 60))):
                    img = np.full((S, S, 3), 40, np.uint8)
                    draw(img, so[k], bones, proj, (235, 235, 235), 3)
                    err = None
                    if other is not None:
                        draw(img, other[k], bones, proj, col, 1)
                        err = max(float(np.linalg.norm(np.array(so[k]["world"][b][p]) - np.array(other[k]["world"][b][p])))
                                  for b in bones for p in ("head", "tail"))
                    lab(img, name, 16)
                    lab(img, f"Action t = {k[0] + k[1]:.2f}" + ("  PIN" if k[0] in pin else ""), 34)
                    if err is not None:
                        lab(img, f"max bone-point gap {err * 1000:.1f} mm", 52, col)
                    tiles.append(img)
                frames.append(np.hstack(tiles))
            vid = out / f"skeleton_{src}_{wname}_4x_slow.mp4"
            h, w = frames[0].shape[:2]
            wr = cv2.VideoWriter(str(vid), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (w, h))
            for fr in frames:
                wr.write(fr)
            wr.release()
            sheet = out / f"skeleton_{src}_{wname}_sheet.png"
            cv2.imwrite(str(sheet), np.vstack(frames[: min(8, len(frames))]))
            res[src][wname] = {"video": str(vid), "sha256": sha(vid), "samples": len(frames), "sheet": str(sheet),
                               "sheet_sha256": sha(sheet), "action_frames": [keys[0][0], keys[-1][0]]}
    (out / "supplementary_visual.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
