#!/usr/bin/env python3
"""W140 (EXPERIMENTAL) RGB-only camera pose estimator: COLMAP incremental Structure-from-Motion via pycolmap.

Runs ONLY in the scratch environment C:/Users/dna10/animcv-w140/env/pc311 (pycolmap 4.2.1, CPU).

    pc311/Scripts/python.exe -I scripts/w140_colmap_estimate.py --input <in_dir> --out <out_dir> [--smoke N]

INPUT CONTRACT (the only information the estimator receives):
    <in_dir>/images/*.png      RGB frames (already undistorted with the published calibration, for TUM fr1)
    <in_dir>/frames.json       [{frame_id, timestamp, image}] in acquisition order (original timestamps)
    <in_dir>/camera.json       PINHOLE intrinsics in the OpenCV pixel convention (fx, fy, cx, cy, width, height)
  The runner REFUSES an input directory containing any ground-truth, depth, IMU/accelerometer or pose file.

FROZEN CONFIGURATION (COLMAP defaults except where the input contract requires otherwise):
    SIFT (CPU, max 8192 features), sequential pairing overlap 10 (quadratic), no loop detection (no vocabulary tree),
    single shared PINHOLE camera with the given intrinsics (COLMAP cx/cy = OpenCV + 0.5), intrinsics NOT refined
    (ba_refine_focal_length / principal_point / extra_params = False, abs_pose_refine_* = False), incremental mapper
    defaults (incl. COLMAP's own initialisation relaxation), random_seed 0, num_threads 8, multiple_models allowed.

OUTPUT (isolated experimental format, schema animcv_exp_w140_camera_pose_v1):
    estimate.json : per selected frame {frame_id, timestamp, image, status, T_WC (4x4, arbitrary gauge & scale),
                    model_id, n_keypoints, n_points3D_observed, mean_reproj_px}; models summary; diagnostics; versions
    points3D.npy  : primary-model landmarks [x y z r g b error track_len]
Status: ESTIMATED (registered in the primary = largest model), INVALID (registered only in a secondary model: a
different gauge), LOST (never registered).  No interpolation or prediction is ever produced.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

FORBIDDEN = ("groundtruth", "gt", "depth", "accelerometer", "imu", "pose", "trajectory")


def refuse_leaks(in_dir):
    for p in Path(in_dir).rglob("*"):
        stem = p.name.lower()
        if p.is_file() and p.parent.name != "images" and any(k in stem for k in FORBIDDEN):
            raise SystemExit(f"STOP GT_LEAKAGE: forbidden input file {p}")
        if p.is_dir() and p.name.lower() in ("gt", "depth", "groundtruth"):
            raise SystemExit(f"STOP GT_LEAKAGE: forbidden input directory {p}")
    bad_img = [p.name for p in (Path(in_dir) / "images").iterdir() if p.suffix.lower() != ".png"]
    if bad_img:
        raise SystemExit(f"STOP GT_LEAKAGE: non-image files in images/ {bad_img[:5]}")
    allowed = {"images", "frames.json", "camera.json", "input_manifest.json"}
    extra = [p.name for p in Path(in_dir).iterdir() if p.name not in allowed]
    if extra:
        raise SystemExit(f"STOP GT_LEAKAGE: unexpected input entries {extra}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--smoke", type=int, default=0, help="DEV ONLY: use only the first N frames")
    a = ap.parse_args()
    import pycolmap
    in_dir, out = Path(a.input), Path(a.out)
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}")
    refuse_leaks(in_dir)
    out.mkdir(parents=True)
    cam = json.load(open(in_dir / "camera.json"))
    frames = json.load(open(in_dir / "frames.json"))
    if a.smoke:
        frames = frames[:a.smoke]
    names = [f["image"] for f in frames]
    for n in names:
        if not (in_dir / "images" / n).is_file():
            raise SystemExit(f"STOP MISSING_IMAGE {n}")
    if cam["model"] != "PINHOLE" or not (cam["fx"] > 0 and cam["fy"] > 0):
        raise SystemExit("STOP BAD_CALIBRATION")
    params = f"{cam['fx']},{cam['fy']},{cam['cx'] + 0.5},{cam['cy'] + 0.5}"
    pycolmap.set_random_seed(0)
    db = out / "database.db"
    t0 = time.perf_counter()
    reader = pycolmap.ImageReaderOptions()
    reader.camera_model = "PINHOLE"
    reader.camera_params = params
    ext = pycolmap.FeatureExtractionOptions()
    ext.num_threads = 8
    ext.use_gpu = False
    pycolmap.extract_features(db, in_dir / "images", image_names=names, camera_mode=pycolmap.CameraMode.SINGLE,
                              reader_options=reader, extraction_options=ext, device=pycolmap.Device.cpu)
    t1 = time.perf_counter()
    mopt = pycolmap.FeatureMatchingOptions()
    mopt.num_threads = 8
    mopt.use_gpu = False
    popt = pycolmap.SequentialPairingOptions()
    popt.overlap = 10
    popt.quadratic_overlap = True
    popt.loop_detection = False
    pycolmap.match_sequential(db, matching_options=mopt, pairing_options=popt, device=pycolmap.Device.cpu)
    t2 = time.perf_counter()
    opt = pycolmap.IncrementalPipelineOptions()
    opt.random_seed = 0
    opt.num_threads = 8
    opt.ba_refine_focal_length = False
    opt.ba_refine_principal_point = False
    opt.ba_refine_extra_params = False
    opt.mapper.abs_pose_refine_focal_length = False
    opt.mapper.abs_pose_refine_extra_params = False
    sparse = out / "sparse"
    sparse.mkdir()
    recs = pycolmap.incremental_mapping(db, in_dir / "images", sparse, options=opt)
    t3 = time.perf_counter()
    # ---- database diagnostics (keypoints, verified pair inliers) ----
    dbo = pycolmap.Database.open(str(db)) if hasattr(pycolmap.Database, "open") else pycolmap.Database(str(db))
    name_to_dbid = {im.name: im.image_id for im in dbo.read_all_images()}
    nkp = {n: int(dbo.read_keypoints(i).shape[0]) for n, i in name_to_dbid.items()}
    pair_stats = []
    for k in range(len(names) - 1):
        i1, i2 = name_to_dbid[names[k]], name_to_dbid[names[k + 1]]
        try:
            raw = dbo.read_matches(i1, i2)
            n_raw = int(len(raw)) if raw is not None else 0
        except Exception:
            n_raw = 0
        try:
            tvg = dbo.read_two_view_geometry(i1, i2)
            n_in = int(len(tvg.inlier_matches))
            cfg = str(tvg.config)
        except Exception:
            n_in, cfg = 0, "NONE"
        pair_stats.append({"pair": [frames[k]["frame_id"], frames[k + 1]["frame_id"]], "raw_matches": n_raw, "inliers": n_in,
                           "inlier_ratio": (n_in / n_raw) if n_raw else None, "config": cfg})
    dbo.close()
    # ---- models ----
    models = []
    for mid, r in sorted(recs.items()):
        models.append({"model_id": int(mid), "n_registered": int(r.num_reg_images()), "n_points3D": int(r.num_points3D()),
                       "mean_reproj_px": float(r.compute_mean_reprojection_error()) if r.num_points3D() else None,
                       "mean_track_length": float(r.compute_mean_track_length()) if r.num_points3D() else None,
                       "mean_obs_per_image": float(r.compute_mean_observations_per_reg_image()) if r.num_reg_images() else None})
    primary = max(recs.items(), key=lambda kv: (kv[1].num_reg_images(), -kv[0]))[0] if recs else None
    reg = {}
    for mid, r in recs.items():
        for iid, im in r.images.items():
            if im.has_pose:
                reg.setdefault(im.name, []).append((mid, r, im))
    est = []
    for f in frames:
        rec = {"frame_id": f["frame_id"], "timestamp": f["timestamp"], "image": f["image"], "n_keypoints": nkp.get(f["image"], 0),
               "status": "LOST", "T_WC": None, "model_id": None}
        hits = reg.get(f["image"], [])
        prim = [h for h in hits if h[0] == primary]
        if prim:
            mid, r, im = prim[0]
            T_CW = np.vstack([np.asarray(im.cam_from_world().matrix() if callable(im.cam_from_world) else im.cam_from_world.matrix()), [0, 0, 0, 1]])
            R, t = T_CW[:3, :3], T_CW[:3, 3]
            T_WC = np.eye(4)
            T_WC[:3, :3], T_WC[:3, 3] = R.T, -R.T @ t
            errs = []
            for p2 in im.points2D:
                if p2.has_point3D():
                    errs.append(r.points3D[p2.point3D_id].error)
            rec.update({"status": "ESTIMATED", "T_WC": T_WC.tolist(), "model_id": int(mid), "n_points3D_observed": int(im.num_points3D),
                        "mean_track_error_px_of_observed_points": float(np.mean(errs)) if errs else None,
                        "projection_center_check": float(np.linalg.norm(np.asarray(im.projection_center()) - T_WC[:3, 3]))})
        elif hits:
            rec.update({"status": "INVALID", "model_id": int(hits[0][0]), "reason": "registered only in a secondary model (different gauge)"})
        est.append(rec)
    pts = []
    if primary is not None:
        r = recs[primary]
        for pid, p in r.points3D.items():
            pts.append([*np.asarray(p.xyz).tolist(), *np.asarray(p.color).tolist(), float(p.error), int(p.track.length())])
    np.save(out / "points3D.npy", np.asarray(pts, np.float64) if pts else np.zeros((0, 8)))
    if primary is not None:
        (out / "primary_model_txt").mkdir()
        recs[primary].write_text(str(out / "primary_model_txt"))
    res = {"schema": "animcv_exp_w140_camera_pose_v1", "estimator": "COLMAP incremental SfM (pycolmap)", "pycolmap": pycolmap.__version__,
           "numpy": np.__version__, "python": sys.version.split()[0], "input": str(in_dir), "n_frames": len(frames), "smoke": a.smoke,
           "camera_params_colmap": params, "config": {"sift_max_features": ext.sift.max_num_features, "overlap": popt.overlap,
                                                      "quadratic_overlap": popt.quadratic_overlap, "loop_detection": popt.loop_detection,
                                                      "refine_intrinsics": False, "random_seed": 0, "num_threads": 8,
                                                      "init_min_tri_angle": opt.mapper.init_min_tri_angle, "multiple_models": opt.multiple_models},
           "models": models, "primary_model_id": primary, "frames": est, "pair_stats_consecutive": pair_stats,
           "status_counts": {s: sum(1 for e in est if e["status"] == s) for s in ("ESTIMATED", "PREDICTED_INTERPOLATED", "LOST", "UNOBSERVABLE", "INVALID")},
           "timing_s": {"extract": t1 - t0, "match": t2 - t1, "map": t3 - t2, "total": t3 - t0},
           "frame_convention": "T_WC: camera-optical (OpenCV +X right, +Y down, +Z forward) -> arbitrary COLMAP world; scale arbitrary",
           "scale_status": "SCALE_AMBIGUOUS", "interpolation": "none"}
    (out / "estimate.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ("status_counts", "models", "primary_model_id", "timing_s")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
