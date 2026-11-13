#!/usr/bin/env python3
"""W143 (EXPERIMENTAL) decoupled camera-state estimator: validated background ORIENTATION, separate translation
observability, and 6DoF availability from a frozen SfM reconstruction's own evidence.

    python -I scripts/w143_decoupled_estimate.py --input <in_dir> --out <out_dir> --keep-masks <dir> --mask-label L [--sfm <estimate.json>]

Inputs as W142 (W140 leak guard, W141 mask contract).  --sfm is a frozen SfM estimate.json (an estimator output, never GT).
Pipeline: W142 features / matches / spans (unchanged functions) -> per-edge W143 evidence (app.experimental_decoupled_camera_state.
analyse_edge) -> rotation candidates -> (s, s, 2s) cycle check -> robust rotation averaging (W142 function) on accepted edges
-> translation direction confirmation across spans -> +-0.5 s windows -> per-frame camera state.
"""
from __future__ import annotations

import argparse
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
import app.experimental_decoupled_camera_state as DC                  # noqa: E402


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
    ap.add_argument("--sfm")
    a = ap.parse_args()
    ind, out = Path(a.input), Path(a.out)
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}")
    _script("w140_colmap_estimate").refuse_leaks(ind)
    masks = _script("w141_colmap_masked_estimate").check_masks(ind, a.keep_masks) if a.keep_masks else None
    if (a.keep_masks is None) != (a.mask_label == "none"):
        raise SystemExit("STOP: --mask-label none iff no masks")
    sfm = json.load(open(a.sfm)) if a.sfm else None
    if sfm is not None and "groundtruth" in str(a.sfm).lower():
        raise SystemExit("STOP GT_LEAKAGE: --sfm must be an estimator output")
    out.mkdir(parents=True)
    cam = json.load(open(ind / "camera.json"))
    frames = json.load(open(ind / "frames.json"))
    K = np.array([[cam["fx"], 0, cam["cx"]], [0, cam["fy"], cam["cy"]], [0, 0, 1.0]])
    CP.check_intrinsics(K, cam["width"], cam["height"])
    t0 = time.perf_counter()
    feats, per_frame = [], []
    for f in frames:
        img = cv2.imread(str(ind / "images" / f["image"]), cv2.IMREAD_COLOR)
        if img is None or img.shape[:2] != (cam["height"], cam["width"]):
            feats.append((np.zeros((0, 2)), np.zeros((0, 128), np.float32)))
            per_frame.append({"n_kp": 0, "invalid_input": True})
            continue
        keep = (cv2.imread(str(Path(a.keep_masks) / (f["image"] + ".png")), cv2.IMREAD_UNCHANGED) > 0) if a.keep_masks else None
        xy, desc, _ = BR.sift_features(img, keep)
        feats.append((xy, desc))
        per_frame.append({"n_kp": int(len(xy)), "kp_on_excluded": int((~BR.on_keep(xy, keep)).sum()) if keep is not None and len(xy) else 0})
    t1 = time.perf_counter()
    times = [f["timestamp"] for f in frames]
    edges, store = [], {}
    for i in range(len(frames)):
        for s in BR.SPANS:
            j = i + s
            if j >= len(frames):
                continue
            mm = BR.mutual_ratio_matches(feats[i][1], feats[j][1])
            p1, p2 = (feats[i][0][mm[:, 0]], feats[j][0][mm[:, 1]]) if len(mm) else (np.zeros((0, 2)), np.zeros((0, 2)))
            e = DC.analyse_edge(K, p1, p2, seed=1000003 * i + j)
            inl = e.pop("_inl")
            store[f"e{i}_{j}"] = mm[inl] if len(mm) else np.zeros((0, 2), int)
            e.update({"i": i, "j": j, "span": s, "frame_ids": [frames[i]["frame_id"], frames[j]["frame_id"]], "dt_s": times[j] - times[i]})
            edges.append(e)
    t2 = time.perf_counter()
    cyc_rej, cyc = DC.cycle_reject(edges)          # greedy: worst inconsistent edge removed first
    for e in edges:
        c = cyc.get((e["i"], e["j"]), [])
        e["cycle_n"] = len(c)
        e["cycle_median_deg"] = float(np.median(c)) if c else None
        if (e["i"], e["j"]) in cyc_rej:
            e["reject"] = e["reject"] + ["cycle_inconsistent"]
        e["rotation_accepted"] = bool(e.get("rotation_candidate") and "cycle_inconsistent" not in e["reject"])
        e["cycle_verified"] = bool(e["rotation_accepted"] and e["cycle_n"] > 0)
    DC.confirm_translation(edges)
    acc = [e for e in edges if e["rotation_accepted"]]
    R_wc, info = BR.rotation_averaging(len(frames), [(e["i"], e["j"]) for e in acc], [np.asarray(e["M"]) for e in acc])
    res_by_edge = {}
    if acc and info.get("component"):
        loc = set(info["component"])
        k = 0
        for e in acc:
            if e["i"] in loc and e["j"] in loc:
                res_by_edge[(e["i"], e["j"])] = info["edge_residual_deg"][k]
                k += 1
    for e in edges:
        e["averaging_residual_deg"] = res_by_edge.get((e["i"], e["j"]))
    win = DC.frame_translation_window(times, edges)
    sv = DC.sfm_validity(sfm)
    t3 = time.perf_counter()
    out_frames = []
    for k, f in enumerate(frames):
        inc = [e for e in acc if k in (e["i"], e["j"]) and (res_by_edge.get((e["i"], e["j"])) or 99) <= BR.EDGE_OUTLIER_DEG]
        ok = R_wc[k] is not None and info.get("support", {}).get(k, 0) > 0
        o = {"status": "ORIENTATION_ESTIMATED" if ok else "ORIENTATION_UNSUPPORTED",
             "R_WC": R_wc[k].tolist() if ok else None, "q_xyzw": CP.R_to_quat_xyzw(R_wc[k]).tolist() if ok else None,
             "reference_frame_id": frames[info["reference"]]["frame_id"] if ok else None, "reference_frame": "W143 reference camera optical frame",
             "evidence": {"accepted_edges": len(inc), "cycle_verified_edges": sum(e["cycle_verified"] for e in inc),
                          "rotation_bias_indicator_deg_median": float(np.median([e["rotation_bias_bound_deg"] for e in inc])) if inc else None,
                          "confidence": ("CYCLE_VERIFIED" if any(e["cycle_verified"] for e in inc) else "UNVERIFIED") if ok else None},
             "why_unsupported": None if ok else ("OTHER_COMPONENT" if R_wc[k] is None and any(k in (e["i"], e["j"]) for e in acc)
                                                 else "NO_ACCEPTED_ROTATION_EDGE")}
        sf = sv["frames"].get(f["frame_id"])
        cs, ts = DC.camera_state(ok, sf, win[k])
        tr = {"status": ts, "value": None, "scale_status": None, "reference_frame": None}
        if cs == "FULL_6DOF_AVAILABLE":
            T = np.asarray(sf["T_WC"])
            tr.update({"value": T[:3, 3].tolist(), "scale_status": "SCALE_AMBIGUOUS", "reference_frame": "frozen SfM model gauge",
                       "sfm_R_WC": T[:3, :3].tolist()})
        out_frames.append({"frame_id": f["frame_id"], "timestamp": f["timestamp"], "image": f["image"], "camera_state": cs,
                           "orientation": o, "translation": tr, "translation_window": win[k],
                           "sfm": {"model_status": sv["status"], "frame_status": sf["sfm_status"] if sf else None,
                                   "frame_valid": bool(sf and sf["valid"])},
                           "evidence": per_frame[k], "provenance": f"background features outside the '{a.mask_label}' keep mask; "
                           f"SfM: {a.sfm or 'none'}"})
    np.savez_compressed(out / "features.npz", **{f"f{k}": feats[k][0] for k in range(len(frames))})
    np.savez_compressed(out / "edges.npz", **store)
    cnt = lambda key, vals: {v: sum(1 for x in out_frames if key(x) == v) for v in vals}   # noqa: E731
    res = {"schema": DC.SCHEMA, "input": str(ind), "mask_label": a.mask_label, "keep_masks": a.keep_masks,
           "mask_files_sha256": ({k: v["sha256"] for k, v in masks.items()} if masks else None), "sfm_source": a.sfm,
           "sfm_validity": {k: v for k, v in sv.items() if k != "frames"}, "camera": cam,
           "calibration_status": cam.get("calibration_status", "PUBLISHED"), "n_frames": len(frames),
           "params": {k: getattr(DC, k) for k in dir(DC) if k.isupper() and not k.endswith("STATES") and k not in ("SCHEMA", "BR", "CP")},
           "w142_params_reused": {k: getattr(BR, k) for k in ("SIFT_NFEATURES", "RATIO", "RANSAC_THR_PX", "GRIC_SIGMA_PX", "SPANS", "IRLS_C",
                                                              "EDGE_OUTLIER_DEG", "WINDOW_S", "REGIME_T_FRAC", "ROT_RATE_DEG_S")},
           "frames": out_frames, "edges": edges, "rotation_averaging": {k: v for k, v in info.items() if k != "edge_residual_deg"},
           "counts": {"camera_state": cnt(lambda x: x["camera_state"], DC.CAMERA_STATES),
                      "translation": cnt(lambda x: x["translation"]["status"], DC.TRANSLATION_STATES),
                      "orientation": cnt(lambda x: x["orientation"]["status"], ("ORIENTATION_ESTIMATED", "ORIENTATION_UNSUPPORTED")),
                      "edges_rotation_accepted": sum(e["rotation_accepted"] for e in edges), "edges_total": len(edges),
                      "edges_fit_model": {m: sum(e.get("fit_model") == m for e in edges) for m in ("R", "H", "E", None)},
                      "edge_reject_reasons": {r: sum(r in e["reject"] for e in edges) for r in ("too_few_matches", "rotation_inliers", "rotation_support",
                                                                                                  "rotation_residual", "rotation_vs_homography", "cycle_inconsistent")},
                      "edges_translation": {s: sum(e["translation_edge"] == s for e in edges) for s in ("DIRECTION_OBSERVED", "UNRESOLVED_UNCONFIRMED",
                                                                                                        "UNRESOLVED", "INSUFFICIENT")},
                      "numeric_translation_without_valid_sfm": sum(1 for x in out_frames if x["translation"]["value"] is not None
                                                                   and x["camera_state"] != "FULL_6DOF_AVAILABLE")},
           "kp_on_excluded_total": int(sum(p.get("kp_on_excluded", 0) for p in per_frame)),
           "timing_s": {"features": t1 - t0, "pairs": t2 - t1, "validation_averaging_state": t3 - t2, "total": time.perf_counter() - t0},
           "versions": {"opencv": cv2.__version__, "numpy": np.__version__}}
    (out / "estimate.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({"counts": res["counts"], "sfm": res["sfm_validity"], "timing_s": res["timing_s"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
