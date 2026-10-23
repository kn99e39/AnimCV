#!/usr/bin/env python3
"""REVIEW (Worklog 91): side-by-side MP4s of the root-heading rig replay (FIXED 180 | RUNTIME GT-free | GT heading).

Top row: input RGB (frame, time, runtime reliability state), then FIXED / RUNTIME / GT (evaluation-only) in the
video-camera view.  Bottom row: heading trajectories (GT, RUNTIME with UNRELIABLE marks, FIXED) and the upper-arm local
rotation error against the GT-reference arm, then the three modes in the side view (depth x up).  Arrows from the
pelvis show the body-forward direction (mode colour) and the GT heading (thin green).  Clips play the sampled rows in
real time (6 fps TEST, 10 fps VALIDATION) without interpolation.  Bone proxies only: BaseRig has no skinned mesh.
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

MODES = ("FIXED", "RUNTIME", "GT")
LABEL = {"FIXED": "FIXED heading 180 (W90 control)", "RUNTIME": "RUNTIME GT-free heading (S0)", "GT": "GT heading (evaluation only)"}
BGR = {"FIXED": (26, 140, 255), "RUNTIME": (214, 26, 214), "GT": (242, 217, 26)}
MPL = {"FIXED": "#ff8c1a", "RUNTIME": "#d61ad6", "GT": "#22aa22"}
T = 480


def qangle(a, b):
    return math.degrees(2 * math.acos(min(1.0, abs(float(np.dot(a, b))))))


def circ(a, b):
    return (a - b + 180.0) % 360.0 - 180.0


def text(img, lines):
    if lines:
        width = max(cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0][0] for t, _ in lines)
        cv2.rectangle(img, (2, 4), (14 + width, 10 + 20 * len(lines)), (25, 25, 25), -1)
    for i, (t, c) in enumerate(lines):
        cv2.putText(img, t, (8, 20 + 20 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.45, c or (235, 235, 235), 1, cv2.LINE_AA)
    return img


def fit(img):
    h, w = img.shape[:2]
    s = T / max(h, w)
    small = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    out = np.zeros((T, T, 3), np.uint8)
    y, x = (T - small.shape[0]) // 2, (T - small.shape[1]) // 2
    out[y:y + small.shape[0], x:x + small.shape[1]] = small
    return out


def plot(case, i):
    fr = case["frames"]
    labels = case.get("short_labels", {k: k for k in MODES})
    t = np.array([f["time_s"] - fr[0]["time_s"] for f in fr])
    gt = np.array([f["yaw_deg"]["GT"] if f["yaw_deg"]["GT"] is not None else np.nan for f in fr])
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(4.8, 4.8), dpi=100, sharex=True)
    ref = np.nan_to_num(gt, nan=180.0)
    for k in ("GT", "FIXED", "RUNTIME"):
        y = np.array([f["yaw_deg"][k] if f["yaw_deg"][k] is not None else np.nan for f in fr])
        y = ref + np.array([circ(a, b) if np.isfinite(a) else np.nan for a, b in zip(y, ref)])  # unwrap around GT
        a1.plot(t, y, color=MPL[k], lw=2.6 if k == "GT" else 1.5, label=labels.get(k, k))
        if k != "GT":
            err = [qangle(f["modes"][k]["proximal_local_xyzw"], f["modes"]["GT_REF"]["proximal_local_xyzw"])
                   if k in f["modes"] and "GT_REF" in f["modes"] else np.nan for f in fr]
            a2.plot(t, err, color=MPL[k], lw=1.5)
    bad = [j for j, f in enumerate(fr) if f["runtime_status"] != "KNOWN"]
    if bad:
        a1.scatter(t[bad], [ref[j] for j in bad], marker="x", color="red", s=18, label="RUNTIME not KNOWN", zorder=5)
    for a in (a1, a2):
        a.axvline(t[i], color="k", lw=0.8)
        a.tick_params(labelsize=7)
    a1.set_ylabel("body heading yaw (deg)", fontsize=8)
    a2.set_ylabel("upper-arm local rot. error vs GT ref (deg)", fontsize=8)
    a2.set_xlabel("time (s)", fontsize=8)
    a1.legend(fontsize=6, loc="best")
    a1.set_title(f"{case['case']} ({case['sequence_id'].split(':', 1)[1]})", fontsize=8)
    fig.tight_layout()
    fig.canvas.draw()
    img = cv2.cvtColor(np.asarray(fig.canvas.buffer_rgba())[:, :, :3], cv2.COLOR_RGB2BGR)
    plt.close(fig)
    return cv2.resize(img, (T, T))


def compose(key, case, renders, images, image_root, out_dir):
    fr = case["frames"]
    label = case.get("labels", LABEL)  # optional per-case mode labels (Worklog 92); defaults unchanged
    fps = case["fps_sampled"]
    writer = None
    for i, f in enumerate(fr):
        img = cv2.imread(str(image_root / f.get("image", images.get(f["bank_position"], ""))))
        rgb = fit(img) if img is not None else np.zeros((T, T, 3), np.uint8)
        status = f["runtime_status"] + (f" ({', '.join(f['runtime_reasons'])})" if f["runtime_reasons"] else "")
        text(rgb, [(f"input RGB  frame {f['frame_index']}  t={f['time_s'] - fr[0]['time_s']:.2f}s", None),
                   (f"runtime heading: {status}", (60, 60, 255) if f["runtime_status"] != "KNOWN" else None)]
             + ([("arm UNAVAILABLE (detector) -> arm pose held", (60, 60, 255))] if not f["available"] else []))
        top, bottom = [rgb], [plot(case, i)]
        g = f["yaw_deg"]["GT"]
        for k in MODES:
            y = f["yaw_deg"][k]
            lines = [(label[k], BGR[k])]
            if y is not None and g is not None and k != "GT":
                lines.append((f"heading err {abs(circ(y, g)):5.1f} deg", None))
            if k in f["modes"] and "GT_REF" in f["modes"]:
                lines.append((f"upper-arm local err {qangle(f['modes'][k]['proximal_local_xyzw'], f['modes']['GT_REF']['proximal_local_xyzw']):5.1f} deg", None))
            for view, row in (("video", top), ("side", bottom)):
                r = cv2.imread(str(renders / key / f"{k}_{view}_{i:03d}.png"))
                r = cv2.resize(r, (T, T)) if r is not None else np.zeros((T, T, 3), np.uint8)
                text(r, lines if view == "video" else [(f"{k}  side view (depth x up)", BGR[k])])
                row.append(r)
        frame = np.vstack([np.hstack(top), np.hstack(bottom)])
        band = np.full((34, frame.shape[1], 3), 20, np.uint8)
        cv2.putText(band, f"{case.get('title', 'W91 root-heading rig replay')} | {key} | sampled {fps:g} Hz, real time | BaseRig bone proxies (no skinned mesh) | "
                          "J1X left arm via W61/W76 IK | thin green = GT heading / GT-reference arm (evaluation only)",
                    (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (235, 235, 235), 1, cv2.LINE_AA)
        frame = np.vstack([band, frame])
        if writer is None:
            path = out_dir / f"{key}.mp4"
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"avc1"), fps, (frame.shape[1], frame.shape[0]))
            if not writer.isOpened():
                writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame.shape[1], frame.shape[0]))
        writer.write(frame)
        if i == len(fr) // 2:
            cv2.imwrite(str(out_dir / f"{key}.png"), frame)
    writer.release()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("cases", "renders", "bank", "image_root", "out_dir"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    args = parser.parse_args()
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    from framepose.bank import load_bank

    args.out_dir.mkdir(parents=True, exist_ok=True)
    data = json.loads(args.cases.read_text())
    images = {}
    if any("image" not in f for c in data["cases"].values() for f in c["frames"]):
        bank = load_bank(args.bank)
        images = {f["bank_position"]: bank.samples[f["bank_position"]].image_reference.relative_path
                  for c in data["cases"].values() for f in c["frames"]}
    for key, case in data["cases"].items():
        compose(key, case, args.renders, images, args.image_root, args.out_dir)
        print(key, flush=True)


if __name__ == "__main__":
    main()
