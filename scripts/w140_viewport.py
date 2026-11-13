#!/usr/bin/env python3
"""W140 POST-HOC visualisation (host side): third-person 3D perspective viewport videos of the camera motion.

Reads only FROZEN W140 outputs (estimate.json, points3D.npy, evaluation.json Sim(3)) and the GT files; never changes any
W140 result, gate or classification.  For each dataset: builds a spec (GT T_WC interpolated at every selected frame;
ESTIMATED frames mapped by the evaluation-only full-trajectory Sim(3); landmarks mapped likewise), renders it in Blender
(scripts/w140_viewport_blender.py), then composes a video paced by the REAL frame timestamps (30 fps output, each selected
frame held for its true inter-frame interval) with a legend, timestamp, status (ESTIMATED / TRACKING LOST) and the RGB
frame the camera saw as an inset.

    PY=C:/Projects/AnimCV/.venv_build_full/Scripts/python.exe; export PYTHONPATH=src
    $PY -X utf8 scripts/w140_viewport.py --out-dir C:/Users/dna10/animcv-w140/results/viewport
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src")]
import cv2                                                            # noqa: E402
import numpy as np                                                    # noqa: E402

import app.experimental_camera_pose as CP                             # noqa: E402

W140 = Path("C:/Users/dna10/animcv-w140")
R = W140 / "results"
BLENDER = "C:/Program Files/Blender Foundation/Blender 5.1/blender.exe"
GT = {"synthetic": W140 / "data" / "synthetic_v3" / "gt" / "camera_gt.txt",
      "xyz": W140 / "data" / "extracted" / "rgbd_dataset_freiburg1_xyz" / "groundtruth.txt",
      "rpy": W140 / "data" / "extracted" / "rgbd_dataset_freiburg1_rpy" / "groundtruth.txt"}
MAX_POINTS = 4000
RADIUS_M = {"synthetic": 6.0, "xyz": 2.5, "rpy": 2.5}


def spec_for(nm):
    ev = json.load(open(R / "evaluate" / "evaluation.json"))["datasets"][nm]
    est = json.load(open(R / "estimate" / nm / "estimate.json"))
    pf = json.load(open(R / "preflight" / "preflight.json"))
    cam = json.load(open(Path(pf[f"input_{nm}"]["dir"]) / "camera.json"))
    sim = ev["full_sim3"]["sim3"]
    s, Rs, ts = sim["scale"], np.asarray(sim["R"]), np.asarray(sim["t"])
    gt = CP.read_groundtruth(GT[nm])
    frames = []
    for f in est["frames"]:
        Tg, _ = CP.interpolate_gt(gt, f["timestamp"])
        if Tg is None and nm == "synthetic":                                    # synthetic GT sampled exactly at frame times
            k = int(np.argmin(np.abs(gt[:, 0] - f["timestamp"])))
            Tg = CP.T_WC_from_tum(gt[k, 1:4], gt[k, 4:8]) if abs(gt[k, 0] - f["timestamp"]) < 1e-6 else None
        Te = CP.apply_sim3((s, Rs, ts), [np.asarray(f["T_WC"])])[0] if f["status"] == "ESTIMATED" else None
        frames.append({"frame_id": f["frame_id"], "timestamp": f["timestamp"], "image": f["image"], "status": f["status"],
                       "gt_T": Tg.tolist() if Tg is not None else None, "est_T": Te.tolist() if Te is not None else None})
    pts = np.load(R / "estimate" / nm / "points3D.npy")
    P = (s * (Rs @ pts[:, :3].T)).T + ts if len(pts) else np.zeros((0, 3))
    c = np.mean([np.asarray(f["gt_T"])[:3, 3] for f in frames if f["gt_T"]], axis=0)
    P = P[np.linalg.norm(P - c, axis=1) < RADIUS_M[nm]]
    P = P[np.random.default_rng(0).permutation(len(P))[:MAX_POINTS]] if len(P) > MAX_POINTS else P
    return {"dataset": nm, "K": {k: cam[k] for k in ("fx", "fy", "cx", "cy", "width", "height")}, "frames": frames, "points": P.tolist(),
            "sim3": sim, "view_azimuth_deg": 30.0, "input_dir": pf[f"input_{nm}"]["dir"], "grid_at_floor": nm == "synthetic"}


def label(img, txt, org, col=(255, 255, 255), sc=0.6, th=1):
    cv2.putText(img, txt, org, cv2.FONT_HERSHEY_SIMPLEX, sc, (0, 0, 0), th + 3, cv2.LINE_AA)
    cv2.putText(img, txt, org, cv2.FONT_HERSHEY_SIMPLEX, sc, col, th, cv2.LINE_AA)


def compose(nm, spec, fdir, out, gz):
    frames = spec["frames"]
    t = np.asarray([f["timestamp"] for f in frames])
    t0 = t[0]
    vid = out / f"{nm}_viewport3d_realtime.mp4"
    wr = cv2.VideoWriter(str(vid), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (1280, 720))
    n_out, acc = 0, 0.0
    for i, f in enumerate(frames):
        v = cv2.imread(str(fdir / f"v{i:04d}.png"))
        rgb = cv2.imread(str(Path(spec["input_dir"]) / "images" / f["image"]))
        inset = cv2.resize(rgb, (320, 240))
        cv2.rectangle(inset, (0, 0), (319, 239), (255, 255, 255), 2)
        v[720 - 290:720 - 50, 1280 - 336:1280 - 16] = inset
        label(v, "camera view (RGB input)", (1280 - 334, 720 - 298), sc=0.5)
        label(v, f"W140 {nm}  t = {f['timestamp'] - t0:6.2f} s   frame {f['frame_id']}", (16, 34), sc=0.7, th=2)
        label(v, "GT camera / path (motion capture)" if nm != "synthetic" else "GT camera / path (synthetic exact)", (16, 600), (240, 240, 240), sc=0.55)
        label(v, "ALIGNED ESTIMATE camera / path (RGB-only COLMAP; Sim(3) for evaluation only)", (16, 624), (13, 115, 255), sc=0.55)
        gtxt = ("grid: scene floor z = 0" if gz == 0 else f"grid: horizontal plane z = {gz:.1f} m (parallel to floor, z up)")
        label(v, gtxt + "; 0.1/1 m; red X, green Y", (16, 648), (200, 200, 200), sc=0.48)
        st = "ESTIMATED" if f["status"] == "ESTIMATED" else ("TRACKING LOST" if f["status"] == "LOST" else f["status"])
        label(v, st, (16, 700), (80, 230, 80) if st == "ESTIMATED" else (60, 60, 255), sc=0.9, th=2)
        label(v, "external 3rd-person view (fixed); real-time pacing", (820, 700), (200, 200, 200), sc=0.5)
        dt = (t[i + 1] - t[i]) if i + 1 < len(t) else float(np.median(np.diff(t)))
        acc += dt * 30.0
        k = int(round(acc)) - n_out
        for _ in range(max(k, 1 if i == 0 else 0)):
            wr.write(v)
            n_out += 1
    wr.release()
    cp = cv2.VideoCapture(str(vid))
    nd = 0
    while cp.read()[0]:
        nd += 1
    fps = cp.get(cv2.CAP_PROP_FPS)
    cp.release()
    stills = [0, len(frames) // 3, 2 * len(frames) // 3, len(frames) - 1]
    lost = [i for i, f in enumerate(frames) if f["status"] == "LOST"]
    if lost:
        stills[2] = lost[0]
    tiles = [cv2.resize(cv2.imread(str(fdir / f"v{i:04d}.png")), (640, 360)) for i in stills]
    sheet = out / f"{nm}_viewport3d_stills.png"
    cv2.imwrite(str(sheet), np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:])]))
    return {"video": str(vid), "frames_written": n_out, "frames_decoded": nd, "fps": fps,
            "duration_s": n_out / 30.0, "sequence_span_s": float(t[-1] - t[0]), "stills": str(sheet)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--datasets", default="synthetic,xyz,rpy")
    a = ap.parse_args()
    out = Path(a.out_dir)
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}")
    out.mkdir(parents=True)
    rep = {"label": "POST-HOC visualisation of frozen W140 outputs; no result, gate or classification changed", "datasets": {}}
    for nm in a.datasets.split(","):
        spec = spec_for(nm)
        sp = out / f"{nm}_spec.json"
        sp.write_text(json.dumps(spec))
        fdir = out / f"frames_{nm}"
        r = subprocess.run([BLENDER, "-b", "--factory-startup", "--python-exit-code", "1", "-P", str(ROOT / "scripts" / "w140_viewport_blender.py"),
                            "--", "--spec", str(sp), "--out", str(fdir)], capture_output=True, text=True, encoding="utf-8", errors="replace")
        (out / f"{nm}_blender.log").write_text(r.stdout + "\n--- stderr ---\n" + r.stderr, encoding="utf-8")
        if r.returncode != 0:
            raise SystemExit(f"Blender failed for {nm}: {r.stderr.strip().splitlines()[-3:]}")
        view = json.load(open(fdir / "view.json"))
        rep["datasets"][nm] = {**compose(nm, spec, fdir, out, view["grid_z_m"]), "n_points": len(spec["points"]), "view": view}
    (out / "viewport.json").write_text(json.dumps(rep, indent=1))
    ue = ROOT / "user_eval" / "W140_camera_pose"
    ue.mkdir(parents=True, exist_ok=True)
    for nm, d in rep["datasets"].items():
        for k in ("video", "stills"):
            shutil.copyfile(d[k], ue / Path(d[k]).name)
    print(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
