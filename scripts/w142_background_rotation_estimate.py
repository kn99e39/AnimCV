#!/usr/bin/env python3
"""W142 (EXPERIMENTAL) observability-aware background rotation / motion-state estimator.

    python -I scripts/w142_background_rotation_estimate.py --input <in_dir> --out <out_dir> [--keep-masks <dir>] --mask-label L

INPUT CONTRACT (W140/W141 format, enforced by the unchanged W140 leak guard):
    <in_dir>/images/*.png, frames.json (frame_id, original timestamp, image), camera.json (PINHOLE fx fy cx cy; optional
    "calibration_status"), input_manifest.json.  Optional keep masks in a SEPARATE directory (W141 COLMAP convention:
    <image>.png, 0 = exclude, 255 = keep), validated by the unchanged W141 mask contract.
Never reads GT, depth, IMU or poses.  Outputs (estimate.json, features.npz, edges.npz) are frozen by the harness.

Method (frozen parameters in app.experimental_background_rotation):
  SIFT (<= 4000) on keep-mask pixels -> mutual ratio-test matches for pairs at spans 1,2,4,8,16,32 frames ->
  rotation-only (2-point RANSAC + Kabsch) / homography / essential / fundamental fits on the same matches -> GRIC model
  selection -> per-edge observability state -> robust chordal rotation averaging over rotation-valid edges ->
  per-frame +-0.5 s regime windows.  Orientation is output in the reference camera's optical frame; translation value is
  ALWAYS null with an explicit status.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src")]
import numpy as np                                                    # noqa: E402

import app.experimental_background_rotation as BR                     # noqa: E402
import app.experimental_camera_pose as CP                             # noqa: E402


def _script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main():
    import cv2
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--keep-masks")
    ap.add_argument("--mask-label", required=True)
    a = ap.parse_args()
    ind, out = Path(a.input), Path(a.out)
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}")
    _script("w140_colmap_estimate").refuse_leaks(ind)
    masks = _script("w141_colmap_masked_estimate").check_masks(ind, a.keep_masks) if a.keep_masks else None
    if (a.keep_masks is None) != (a.mask_label == "none"):
        raise SystemExit("STOP: --mask-label none iff no masks")
    out.mkdir(parents=True)
    cam = json.load(open(ind / "camera.json"))
    frames = json.load(open(ind / "frames.json"))
    K = np.array([[cam["fx"], 0, cam["cx"]], [0, cam["fy"], cam["cy"]], [0, 0, 1.0]])
    CP.check_intrinsics(K, cam["width"], cam["height"])
    t0 = time.perf_counter()
    feats, per_frame, kp_dropped = [], [], 0
    for f in frames:
        img = cv2.imread(str(ind / "images" / f["image"]), cv2.IMREAD_COLOR)
        if img is None or img.shape[:2] != (cam["height"], cam["width"]):
            feats.append((np.zeros((0, 2)), np.zeros((0, 128), np.float32)))
            per_frame.append({"n_kp": 0, "invalid_input": True})
            continue
        keep = (cv2.imread(str(Path(a.keep_masks) / (f["image"] + ".png")), cv2.IMREAD_UNCHANGED) > 0) if a.keep_masks else None
        xy, desc, dr = BR.sift_features(img, keep)
        kp_dropped += dr
        on_excl = int((~BR.on_keep(xy, keep)).sum()) if keep is not None and len(xy) else 0
        feats.append((xy, desc))
        per_frame.append({"n_kp": int(len(xy)), "kp_on_excluded": on_excl, "keep_fraction": float(keep.mean()) if keep is not None else 1.0})
    t1 = time.perf_counter()
    times = [f["timestamp"] for f in frames]
    edges, inl_store = [], {}
    for i in range(len(frames)):
        for s in BR.SPANS:
            j = i + s
            if j >= len(frames):
                continue
            (x1, d1), (x2, d2) = feats[i], feats[j]
            mm = BR.mutual_ratio_matches(d1, d2)
            p1, p2 = (x1[mm[:, 0]], x2[mm[:, 1]]) if len(mm) else (np.zeros((0, 2)), np.zeros((0, 2)))
            r = BR.analyse_pair(K, p1, p2, seed=1000003 * i + j)
            inl = r.pop("_inliers")
            r.update({"i": i, "j": j, "span": s, "frame_ids": [frames[i]["frame_id"], frames[j]["frame_id"]],
                      "dt_s": times[j] - times[i]})
            inl_store[f"e{i}_{j}"] = mm[inl] if len(mm) else np.zeros((0, 2), int)
            edges.append(r)
    t2 = time.perf_counter()
    rv = [e for e in edges if e.get("rotation_valid")]
    R_wc, info = BR.rotation_averaging(len(frames), [(e["i"], e["j"]) for e in rv], [np.asarray(e["M"]) for e in rv])
    reg = BR.frame_regimes(times, edges)
    seqreg = BR.sequence_regime(reg)
    t3 = time.perf_counter()
    by_pair = {(e["i"], e["j"]): e for e in edges}
    res_frames = []
    for k, f in enumerate(frames):
        valid_edges = [e for e in edges if k in (e["i"], e["j"]) and e["state"] != "INSUFFICIENT_EVIDENCE"]
        if R_wc[k] is not None and info["support"].get(k, 0) > 0:
            o = {"status": "ESTIMATED", "R_WC": R_wc[k].tolist(), "q_xyzw": CP.R_to_quat_xyzw(R_wc[k]).tolist(),
                 "reference_frame_id": frames[info["reference"]]["frame_id"], "support_edges": info["support"][k]}
        elif R_wc[k] is not None:
            o = {"status": "UNSUPPORTED", "R_WC": None, "why": "all incident edges are rotation-averaging outliers"}
        elif valid_edges:
            o = {"status": "OTHER_COMPONENT", "R_WC": None, "why": "connected only to a smaller pair-graph component (other gauge)"}
        else:
            o = {"status": "INSUFFICIENT_EVIDENCE", "R_WC": None}
        prev = by_pair.get((k - 1, k))
        rg = reg[k]["regime"]
        if rg == "TRANSLATING":
            tr = {"value": None, "status": "TRANSLATION_DIRECTION_OBSERVED_SCALE_UNOBSERVABLE",
                  "note": "position is not produced by this estimator; a 6DoF SfM trajectory may exist separately"}
        elif rg == "INSUFFICIENT_EVIDENCE":
            tr = {"value": None, "status": "INSUFFICIENT_EVIDENCE"}
        else:
            win = [e for e in edges if e.get("translation_to_depth_bound") is not None and abs(times[e["i"]] - times[k]) <= BR.WINDOW_S
                   and abs(times[e["j"]] - times[k]) <= BR.WINDOW_S]
            tr = {"value": None, "status": "UNRESOLVED_BELOW_PARALLAX_DETECTABILITY",
                  "translation_to_depth_ratio_bound_p90": max((e["translation_to_depth_bound"] for e in win), default=None),
                  "note": "consistent with zero OR small translation relative to scene depth; NOT a measured zero"}
        res_frames.append({"frame_id": f["frame_id"], "timestamp": f["timestamp"], "image": f["image"], "orientation": o,
                           "rotation_observability": "OBSERVED" if o["status"] == "ESTIMATED" else "UNOBSERVED",
                           "translation": tr, "motion_state_from_previous": prev["state"] if prev else None,
                           "regime_window": reg[k], "evidence": per_frame[k],
                           "provenance": f"background features outside the '{a.mask_label}' keep mask"})
    np.savez_compressed(out / "features.npz", **{f"f{k}": feats[k][0] for k in range(len(frames))})
    np.savez_compressed(out / "edges.npz", **inl_store)
    st = {s: sum(e["state"] == s for e in edges) for s in BR.STATES}
    res = {"schema": BR.SCHEMA, "estimator": "W142 background rotation + GRIC observability (OpenCV SIFT, numpy)", "input": str(ind),
           "mask_label": a.mask_label, "keep_masks": a.keep_masks, "mask_files_sha256": ({k: v["sha256"] for k, v in masks.items()} if masks else None),
           "camera": cam, "calibration_status": cam.get("calibration_status", "PUBLISHED"), "n_frames": len(frames),
           "params": {k: getattr(BR, k) for k in ("SIFT_NFEATURES", "RATIO", "RANSAC_THR_PX", "ROT_RANSAC_ITERS", "GRIC_SIGMA_PX", "MIN_INLIERS",
                                                  "SPANS", "ROT_DET_FLOOR_DEG", "CHEIRALITY_MIN_FRAC", "IRLS_C", "IRLS_ITERS", "EDGE_OUTLIER_DEG",
                                                  "WINDOW_S", "REGIME_T_FRAC", "ROT_RATE_DEG_S", "MIN_WINDOW_EDGES")},
           "versions": {"opencv": cv2.__version__, "numpy": np.__version__, "python": sys.version.split()[0]},
           "frames": res_frames, "edges": edges, "edge_state_counts": st, "rotation_averaging": {k: v for k, v in info.items() if k != "edge_residual_deg"},
           "edge_residual_deg": info.get("edge_residual_deg"), "sequence_regime": seqreg,
           "kp_dropped_after_mask_recheck": kp_dropped, "kp_on_excluded_total": int(sum(p.get("kp_on_excluded", 0) for p in per_frame)),
           "orientation_status_counts": {s: sum(fr["orientation"]["status"] == s for fr in res_frames)
                                         for s in ("ESTIMATED", "UNSUPPORTED", "OTHER_COMPONENT", "INSUFFICIENT_EVIDENCE")},
           "timing_s": {"features": t1 - t0, "pairs": t2 - t1, "averaging_regimes": t3 - t2, "total": time.perf_counter() - t0},
           "frame_convention": "R_WC: camera optical (OpenCV) -> reference-camera optical frame; translation never produced"}
    (out / "estimate.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ("edge_state_counts", "sequence_regime", "orientation_status_counts", "timing_s", "kp_on_excluded_total")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
