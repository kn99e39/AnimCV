#!/usr/bin/env python3
"""REVIEW (Worklog 90): side-by-side MP4s of the Blender rig replay (RAW J1X | STABILIZED J1X | G0X reference).

Layout per frame (each tile 480 px): top row = input RGB, then the three modes in the video-camera view; bottom row =
wrist camera-depth and upper-arm rotation-change trajectories with a cursor, then the three modes in the side view
(depth x up).  Every tile is labelled with the per-step changes of that mode (upper-arm / forearm local rotation
change in degrees, wrist depth change in cm) against the GT reference (EVALUATION ONLY, thin green arm in the rig
tiles).  Clips play the sampled rows in real time (6 fps TEST, 10 fps VALIDATION); nothing is interpolated.
UNAVAILABLE rows (invalid detector arm joints) hold the previous pose in every mode and are flagged.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

MODES = ("RAW", "STABILIZED", "G0X")
LABEL = {"RAW": "RAW J1X", "STABILIZED": "STABILIZED J1X (W89)", "G0X": "G0X geometry-only", "GT_REF": "GT reference"}
BGR = {"RAW": (242, 217, 26), "STABILIZED": (214, 26, 214), "G0X": (26, 140, 255), "GT_REF": (34, 170, 34)}
MPL = {"RAW": "#1ad9f2", "STABILIZED": "#d61ad6", "G0X": "#ff8c1a", "GT_REF": "#22aa22"}
T = 480


def qangle(a, b):
    """Sign-invariant angle (deg) between two xyzw quaternions."""
    return math.degrees(2 * math.acos(min(1.0, abs(float(np.dot(a, b))))))


def tile_text(img, lines, color=(235, 235, 235)):
    if lines:
        width = max(cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0][0] for text, _ in lines)
        cv2.rectangle(img, (2, 4), (14 + width, 10 + 20 * len(lines)), (25, 25, 25), -1)
    for i, (text, c) in enumerate(lines):
        cv2.putText(img, text, (8, 20 + 20 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.45, c or color, 1, cv2.LINE_AA)
    return img


def fit(img, size=T):
    h, w = img.shape[:2]
    s = size / max(h, w)
    small = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    out = np.zeros((size, size, 3), np.uint8)
    y, x = (size - small.shape[0]) // 2, (size - small.shape[1]) // 2
    out[y:y + small.shape[0], x:x + small.shape[1]] = small
    return out


def series(case):
    frames = case["frames"]
    t = np.array([f["time_s"] - frames[0]["time_s"] for f in frames])
    depth = {k: np.array([f["modes"][k]["wrist_depth_cm"] if k in f["modes"] else np.nan for f in frames]) for k in (*MODES, "GT_REF")}
    rot = {}
    for k in (*MODES, "GT_REF"):
        r = [np.nan]
        for a, b in zip(frames[:-1], frames[1:]):
            r.append(qangle(a["modes"][k]["proximal_local_xyzw"], b["modes"][k]["proximal_local_xyzw"])
                     if k in a["modes"] and k in b["modes"] else np.nan)
        rot[k] = np.array(r)
    return t, depth, rot


def plot_tile(t, depth, rot, i, title):
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(4.8, 4.8), dpi=100, sharex=True)
    for k in ("GT_REF", "G0X", "RAW", "STABILIZED"):
        lw = 2.6 if k == "GT_REF" else 1.5
        a1.plot(t, depth[k], color=MPL[k], lw=lw, label=LABEL[k].split(" (")[0])
        a2.plot(t, rot[k], color=MPL[k], lw=lw)
    for a in (a1, a2):
        a.axvline(t[i], color="k", lw=0.8)
        a.tick_params(labelsize=7)
    a1.set_ylabel("wrist camera depth (cm)", fontsize=8)
    a2.set_ylabel("upper-arm rotation change / step (deg)", fontsize=8)
    a2.set_xlabel("time (s)", fontsize=8)
    a1.legend(fontsize=6, loc="upper left", ncol=2)
    a1.set_title(title, fontsize=8)
    fig.tight_layout()
    fig.canvas.draw()
    img = cv2.cvtColor(np.asarray(fig.canvas.buffer_rgba())[:, :, :3], cv2.COLOR_RGB2BGR)
    plt.close(fig)
    return cv2.resize(img, (T, T))


def compose(key, case, renders, image_root, out_dir):
    frames = case["frames"]
    t, depth, rot = series(case)
    fps = case["fps_sampled"]
    path = out_dir / f"{key}.mp4"
    writer = None
    last = {}
    for i, fr in enumerate(frames):
        rgb = cv2.imread(str(image_root / fr["image"]))
        rgb = fit(rgb) if rgb is not None else np.zeros((T, T, 3), np.uint8)
        tile_text(rgb, [(f"input RGB  frame {fr['frame_index']}  t={t[i]:.2f}s", None),
                        (f"detector arm conf {fr['detector_min_confidence']:.2f}", None)]
                  + ([("UNAVAILABLE: detector arm invalid -> pose held", (60, 60, 255))] if not fr["available"] else []))
        top, bottom = [rgb], [plot_tile(t, depth, rot, i, f"{key}  ({case['sequence_id'].split(':', 1)[1]})")]
        prev = frames[i - 1] if i > 0 else None
        g_now = fr["modes"].get("GT_REF")
        for k in MODES:
            lines = [(LABEL[k], BGR[k])]
            m = fr["modes"].get(k)
            if m is not None:
                last[k] = m
            if prev is not None and m is not None and k in prev["modes"]:
                pm = prev["modes"][k]
                lines.append((f"upper {qangle(pm['proximal_local_xyzw'], m['proximal_local_xyzw']):5.1f}  fore "
                              f"{qangle(pm['distal_local_xyzw'], m['distal_local_xyzw']):5.1f} deg  dDepth {m['wrist_depth_cm'] - pm['wrist_depth_cm']:+5.1f} cm", None))
                if g_now is not None and "GT_REF" in prev["modes"]:
                    gp = prev["modes"]["GT_REF"]
                    lines.append((f"GT    {qangle(gp['proximal_local_xyzw'], g_now['proximal_local_xyzw']):5.1f}       "
                                  f"{qangle(gp['distal_local_xyzw'], g_now['distal_local_xyzw']):5.1f}      {g_now['wrist_depth_cm'] - gp['wrist_depth_cm']:+5.1f}", BGR["GT_REF"]))
            if m is None:
                lines.append(("UNAVAILABLE (held)", (60, 60, 255)))
            for view, row in (("video", top), ("side", bottom)):
                img = cv2.imread(str(renders / key / f"{k}_{view}_{i:03d}.png"))
                img = cv2.resize(img, (T, T)) if img is not None else np.zeros((T, T, 3), np.uint8)
                tile_text(img, lines if view == "video" else [(f"{LABEL[k]}  side view (depth x up)", BGR[k])])
                row.append(img)
        frame = np.vstack([np.hstack(top), np.hstack(bottom)])
        band = np.full((34, frame.shape[1], 3), 20, np.uint8)
        cv2.putText(band, f"W90 rig replay | {key} | {case['sequence_id']} | sampled {fps:g} Hz, real time, no interpolation | "
                          "BaseRig left arm via W61/W76 IK, constant root yaw (torso faces camera) | thin green = GT reference (evaluation only)",
                    (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (235, 235, 235), 1, cv2.LINE_AA)
        frame = np.vstack([band, frame])
        if writer is None:
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"avc1"), fps, (frame.shape[1], frame.shape[0]))
            if not writer.isOpened():
                writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame.shape[1], frame.shape[0]))
        writer.write(frame)
        if i == len(frames) - 1:
            cv2.imwrite(str(out_dir / f"{key}.png"), frame)
    writer.release()
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("cases", "renders", "image_root", "out_dir"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    data = json.loads(args.cases.read_text())
    for key, case in data["cases"].items():
        print(compose(key, case, args.renders, args.image_root, args.out_dir), flush=True)


if __name__ == "__main__":
    main()
