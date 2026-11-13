#!/usr/bin/env python3
"""W141 (EXPERIMENTAL) RGB-only person segmentation for dynamic-human feature exclusion.

FROZEN MECHANISM (declared before any W141 estimator output; never tuned on camera GT):
  model       torchvision maskrcnn_resnet50_fpn_v2, weights MaskRCNN_ResNet50_FPN_V2_Weights.COCO_V1
              (https://download.pytorch.org/models/maskrcnn_resnet50_fpn_v2_coco-73cbd019.pth), eval mode, CPU, float32
  input       the estimator's own RGB PNG (BGR->RGB, /255), nothing else (no depth, no GT, no neighbouring frames)
  keep        detections with COCO label 1 ('person') and score >= SCORE_THR
  mask        union over kept detections of (soft mask >= MASK_THR)       -> raw person mask (255 = person)
  The model's own resize transform (min_size 800 / max_size 1333) is used unchanged; output masks are in input pixel
  coordinates (same H x W as the image).  Dilation for feature exclusion is applied later by the harness (DILATE_PX).

    python -I scripts/w141_segment_people.py --images <dir> --frames <frames.json> --out <dir> --torch-home <dir>
Writes <out>/<image_name>  (uint8 PNG, 0 / 255) and <out>/../<name>_detections.json (per-frame scores, boxes, areas, timing).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

SCORE_THR = 0.5
MASK_THR = 0.5
PERSON_LABEL = 1
WEIGHTS_URL = "https://download.pytorch.org/models/maskrcnn_resnet50_fpn_v2_coco-73cbd019.pth"
WEIGHTS_SHA256 = "73cbd0190fcbe3ba339921fbce2c3a0b6bb9126c9a133c85e43a2a8e060a109e"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--frames", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--torch-home", required=True)
    ap.add_argument("--threads", type=int, default=16)
    a = ap.parse_args()
    os.environ["TORCH_HOME"] = a.torch_home
    import cv2
    import hashlib
    import numpy as np
    import torch
    import torchvision
    from torchvision.models.detection import MaskRCNN_ResNet50_FPN_V2_Weights, maskrcnn_resnet50_fpn_v2
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}")
    w = MaskRCNN_ResNet50_FPN_V2_Weights.COCO_V1
    if w.url != WEIGHTS_URL:
        raise SystemExit(f"STOP MODEL_IDENTITY: {w.url}")
    ck = Path(a.torch_home) / "hub" / "checkpoints" / Path(WEIGHTS_URL).name
    h = hashlib.sha256(ck.read_bytes()).hexdigest() if ck.is_file() else None
    if h is not None and h != WEIGHTS_SHA256:
        raise SystemExit(f"STOP MODEL_IDENTITY: checkpoint sha {h}")
    if w.meta["categories"][PERSON_LABEL] != "person":
        raise SystemExit("STOP MODEL_IDENTITY: label 1 is not person")
    torch.set_num_threads(a.threads)
    torch.manual_seed(0)
    model = maskrcnn_resnet50_fpn_v2(weights=w).eval()
    h = hashlib.sha256(ck.read_bytes()).hexdigest()
    if h != WEIGHTS_SHA256:
        raise SystemExit(f"STOP MODEL_IDENTITY: checkpoint sha {h}")
    out.mkdir(parents=True)
    frames = json.load(open(a.frames, encoding="utf-8"))
    det = {"model": "torchvision maskrcnn_resnet50_fpn_v2 COCO_V1", "weights_url": WEIGHTS_URL, "weights_sha256": h,
           "torch": torch.__version__, "torchvision": torchvision.__version__, "device": "cpu", "score_thr": SCORE_THR,
           "mask_thr": MASK_THR, "label": "person (COCO 1)", "frames": []}
    t_all = time.perf_counter()
    for f in frames:
        img = cv2.imread(str(Path(a.images) / f["image"]), cv2.IMREAD_COLOR)
        if img is None:
            det["frames"].append({"frame_id": f["frame_id"], "image": f["image"], "error": "unreadable"})
            continue
        x = torch.from_numpy(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float() / 255.0
        t0 = time.perf_counter()
        with torch.no_grad():
            o = model([x])[0]
        dt = time.perf_counter() - t0
        keep = (o["labels"] == PERSON_LABEL) & (o["scores"] >= SCORE_THR)
        m = np.zeros(img.shape[:2], bool)
        boxes, scores, areas = [], [], []
        for k in np.nonzero(keep.numpy())[0]:
            mk = o["masks"][k, 0].numpy() >= MASK_THR
            m |= mk
            boxes.append([round(float(v), 2) for v in o["boxes"][k]])
            scores.append(round(float(o["scores"][k]), 4))
            areas.append(int(mk.sum()))
        if m.shape != img.shape[:2]:
            raise SystemExit("STOP MASK_GEOMETRY")
        cv2.imwrite(str(out / f["image"]), (m * 255).astype(np.uint8))
        det["frames"].append({"frame_id": f["frame_id"], "image": f["image"], "n_person": len(scores), "scores": scores, "boxes_xyxy": boxes,
                              "instance_areas_px": areas, "union_area_px": int(m.sum()), "union_fraction": float(m.mean()),
                              "n_other_detections_ge_thr": int(((o["labels"] != PERSON_LABEL) & (o["scores"] >= SCORE_THR)).sum()),
                              "infer_s": round(dt, 3)})
    det["total_s"] = time.perf_counter() - t_all
    json.dump(det, open(out.parent / f"{out.name}_detections.json", "w", encoding="utf-8"), indent=1)
    print(json.dumps({"n": len(frames), "total_s": det["total_s"],
                      "mean_fraction": float(np.mean([d.get("union_fraction", 0) for d in det["frames"]]))}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
