#!/usr/bin/env python3
"""DIAGNOSTIC (Worklog 73): does the re-run OpenPose COCO-18 detector reproduce 3DPW's shipped detections?

The 3DPW FrameBank observation is the dataset-shipped OpenPose COCO-18 output.
Before running the same detector family on external RGB, measure on a fixed
deterministic subset of 3DPW TRAIN rows (every 10th TRAIN position) how close
the re-run is to the shipped keypoints.  Association: the detected person with
the smallest median normalized distance to the shipped keypoints over joints
valid in both (>= 4 shared).  No 3D GT is read.  Descriptive only: nothing is
tuned from it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import cv2
import numpy as np

from framepose.bank import load_bank
from framepose.openpose_coco18 import OpenPoseCOCO18, to_canonical_input
from pose.pose_lifter import H36M_NAMES

ARM = ("left_shoulder", "left_elbow", "left_wrist")
DIRECT = ("left_shoulder", "left_elbow", "left_wrist", "right_shoulder", "right_elbow", "right_wrist",
          "left_hip", "left_knee", "left_ankle", "right_hip", "right_knee", "right_ankle", "head")
STRIDE = 10


def _stats(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return {"n": int(len(v)), "mean": float(v.mean()), "p50": float(np.median(v)), "p95": float(np.percentile(v, 95))} \
        if len(v) else {"n": 0}


def associate(shipped_px, shipped_valid, people, image_size, scale):
    best, best_d = None, np.inf
    for person in people:
        ours, vis = to_canonical_input(person["keypoints"], image_size)
        ours_px = ours[:, :2] * image_size
        both = shipped_valid & vis
        idx = [H36M_NAMES.index(n) for n in DIRECT]
        both_direct = [i for i in idx if both[i]]
        if len(both_direct) < 4:
            continue
        d = float(np.median(np.linalg.norm(ours_px[both_direct] - shipped_px[both_direct], axis=1)) / scale)
        if d < best_d:
            best, best_d = (ours, vis), d
    return best, best_d


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    bank = load_bank(args.bank)
    detector = OpenPoseCOCO18()
    positions = bank.indices("train")[::STRIDE]
    rows, t0 = [], time.time()
    for p in positions:
        s = bank.samples[int(p)]
        path = args.image_root / s.image_reference.relative_path
        data = path.read_bytes()
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        size = np.array(s.image_size, float)
        shipped = bank.arrays["input_2d"][p]
        shipped_valid = bank.arrays["input_valid"][p]
        shipped_px = shipped[:, :2] * size
        vp = shipped_px[shipped_valid]
        scale = float(np.linalg.norm(vp.max(0) - vp.min(0))) if len(vp) >= 2 else float("nan")
        people = detector(img)
        match, dist = associate(shipped_px, shipped_valid, people, size, scale)
        row = {"position": int(p), "sequence_id": s.sequence_id, "image_sha256": hashlib.sha256(data).hexdigest(),
               "people_detected": len(people), "matched": match is not None, "median_norm_dist": dist, "joints": {}}
        if match is not None:
            ours, vis = match
            for name in DIRECT:
                j = H36M_NAMES.index(name)
                row["joints"][name] = {
                    "shipped_valid": bool(shipped_valid[j]), "ours_valid": bool(vis[j]),
                    "norm_err": float(np.linalg.norm(ours[j, :2] * size - shipped_px[j]) / scale)
                    if shipped_valid[j] and vis[j] else None,
                    "shipped_conf": float(shipped[j, 2]), "ours_conf": float(ours[j, 2])}
        rows.append(row)
    seconds = time.time() - t0

    matched = [r for r in rows if r["matched"]]
    summary = {"rows": len(rows), "stride": STRIDE, "matched_share": len(matched) / len(rows),
               "people_detected": _stats([r["people_detected"] for r in rows]),
               "median_norm_dist": _stats([r["median_norm_dist"] for r in matched]), "joints": {}}
    for name in DIRECT:
        js = [r["joints"][name] for r in matched]
        sv = [j for j in js if j["shipped_valid"]]
        both = [j for j in sv if j["ours_valid"]]
        summary["joints"][name] = {
            "shipped_valid_rows": len(sv), "ours_also_valid_share": len(both) / max(len(sv), 1),
            "ours_valid_when_shipped_invalid_share": float(np.mean([j["ours_valid"] for j in js if not j["shipped_valid"]]))
            if any(not j["shipped_valid"] for j in js) else None,
            "norm_err": _stats([j["norm_err"] for j in both]),
            "share_within_0.05": float(np.mean([j["norm_err"] < 0.05 for j in both])) if both else None,
            "conf_corr": float(np.corrcoef([j["shipped_conf"] for j in both], [j["ours_conf"] for j in both])[0, 1])
            if len(both) > 2 else None,
            "shipped_conf": _stats([j["shipped_conf"] for j in both]), "ours_conf": _stats([j["ours_conf"] for j in both])}
    report = {"schema": "animcv_openpose_provider_parity_v1", "detector": detector.identity(),
              "bank_content_digest": bank.content_digest(), "seconds": seconds,
              "normalization": "pixel distance / diagonal of the shipped valid-keypoint bounding box",
              "summary": summary, "rows": rows}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1))
    print(json.dumps({"seconds": seconds, **{k: v for k, v in summary.items() if k != "joints"},
                      "arm": {n: summary["joints"][n] for n in ARM}}, indent=1))


if __name__ == "__main__":
    main()
