#!/usr/bin/env python3
"""DIAGNOSTIC (Worklog 68): immutable frozen Qwen3-VL visual-token cache for every FrameBank sample.

Same pinned model and runtime as Worklog 67 (4-bit NF4 load), but no text is
generated: only ``get_image_features`` runs on the 448 px FramePose person crop
(no overlay).  Stores the full 14x14x4096 merger grid (fp16 memmap) and the
deterministic readout [shoulder, elbow, wrist bilinear samples, global mean].
No GT is read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from framepose.bank import load_bank
from framepose.learned_vision_sensor import (
    CHANNELS, CROP_CONTRACT, FEATURE_SOURCE, GRID, MODEL_FINGERPRINT, MODEL_REVISION, RESOLUTION, crop_for_vision,
    readout,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("bank", "image_root", "snapshot", "w67_manifest", "out_dir"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    import cv2
    import torch
    import transformers
    from PIL import Image
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration

    manifest = json.loads(args.w67_manifest.read_text())
    if args.snapshot.name != MODEL_REVISION or manifest["aggregate_sha256"] != MODEL_FINGERPRINT:
        raise SystemExit("not the pinned Worklog 67 Qwen3-VL artifact")
    for name, entry in manifest["files"].items():
        path = args.snapshot / name
        if path.stat().st_size != entry["bytes"]:
            raise SystemExit(f"snapshot file changed: {name}")
    bank = load_bank(args.bank)
    processor = AutoProcessor.from_pretrained(args.snapshot)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.snapshot, device_map="cuda:0",
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                               bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=False))
    model.eval()
    merger_fc2 = type(model.model.visual.merger.linear_fc2).__name__

    args.out_dir.mkdir(parents=True, exist_ok=True)
    n = len(bank)
    grid_path = args.out_dir / "token_grid_fp16.npy"
    grids = np.lib.format.open_memmap(grid_path, mode="w+", dtype=np.float16, shape=(n, GRID * GRID, CHANNELS))
    readouts = np.zeros((n, 4, CHANNELS), np.float16)
    joint_valid = np.zeros((n, 3), bool)
    image_sha = np.empty(n, dtype="<U64")
    crop_sha = np.empty(n, dtype="<U64")
    boxes = np.zeros((n, 3), np.float64)
    image_cache = {}

    def load(position):
        rel = bank.samples[position].image_reference.relative_path
        if rel not in image_cache:
            if len(image_cache) > 64:
                image_cache.clear()
            data = (args.image_root / rel).read_bytes()
            bgr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            image_cache[rel] = (hashlib.sha256(data).hexdigest(), cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        return image_cache[rel]

    def features(crops):
        inputs = processor.image_processor(images=[Image.fromarray(c) for c in crops], return_tensors="pt")
        grid_thw = inputs["image_grid_thw"]
        if not bool((grid_thw == torch.tensor([1, 2 * GRID, 2 * GRID])).all()):
            raise SystemExit(f"unexpected image grid {grid_thw.tolist()}")
        with torch.no_grad():
            embeds, _deepstack = model.model.get_image_features(inputs["pixel_values"].to(model.device),
                                                                grid_thw.to(model.device))
        return [e.float().cpu().numpy() for e in embeds], inputs

    started = time.time()
    preprocess = None
    order = sorted(range(n), key=lambda p: bank.samples[p].image_reference.relative_path)
    for start in range(0, n, args.batch_size):
        batch = order[start:start + args.batch_size]
        crops, record = [], []
        for p in batch:
            sample = bank.samples[p]
            sha, rgb = load(p)
            crop, box = crop_for_vision(rgb, bank.arrays["input_2d"][p], bank.arrays["input_valid"][p], sample.image_size)
            crops.append(crop)
            record.append((p, sha, box, hashlib.sha256(crop.tobytes()).hexdigest()))
        tokens, inputs = features(crops)
        if preprocess is None:
            preprocess = {"pixel_values_shape_per_image": [int(inputs["pixel_values"].shape[0] // len(batch)),
                                                           int(inputs["pixel_values"].shape[1])],
                          "image_grid_thw": inputs["image_grid_thw"][0].tolist()}
        for (p, sha, box, csha), t in zip(record, tokens):
            sample = bank.samples[p]
            grids[p] = t.astype(np.float16)
            r, v = readout(t, bank.arrays["input_2d"][p], bank.arrays["input_valid"][p], sample.image_size, box)
            readouts[p], joint_valid[p] = r.astype(np.float16), v
            image_sha[p], crop_sha[p] = sha, csha
            boxes[p] = (box.x, box.y, box.side)
        if (start // args.batch_size) % 100 == 0:
            print(json.dumps({"done": start + len(batch), "of": n, "seconds": round(time.time() - started, 1)}), flush=True)
    grids.flush()

    # Determinism self-check: re-extract the first batch and compare.
    first = order[:args.batch_size]
    again, _ = features([crop_for_vision(load(p)[1], bank.arrays["input_2d"][p], bank.arrays["input_valid"][p],
                                         bank.samples[p].image_size)[0] for p in first])
    determinism = float(max(np.abs(a.astype(np.float16).astype(np.float32) - grids[p].astype(np.float32)).max()
                            for a, p in zip(again, first)))

    np.savez(args.out_dir / "readout.npz", readout=readouts, joint_valid=joint_valid, image_sha256=image_sha,
             crop_sha256=crop_sha, crop_box_xy_side=boxes)
    readout_sha = hashlib.sha256((args.out_dir / "readout.npz").read_bytes()).hexdigest()
    identity = {"bank_content_digest": bank.content_digest(),
                "sample_ids_sha256": hashlib.sha256("\n".join(s.sample_id for s in bank.samples).encode()).hexdigest(),
                "crop_contract": CROP_CONTRACT["schema"], "render_resolution": RESOLUTION, "overlay": "none",
                "model_repository": manifest["repository"], "model_revision": MODEL_REVISION,
                "model_fingerprint": MODEL_FINGERPRINT, "quantization": manifest["quantization"],
                "processor": {"class": type(processor.image_processor).__name__, **preprocess},
                "feature_source": FEATURE_SOURCE, "merger_linear_fc2_class": merger_fc2,
                "token_grid": [GRID, GRID], "token_order": "row-major over (h/2, w/2) merged 2x2 patch groups",
                "token_grid_dtype": "float16", "channels": CHANNELS,
                "readout": "bilinear at detector left shoulder/elbow/wrist (token centres at 32*(i+0.5) px) + global mean",
                "transformers": transformers.__version__, "torch": torch.__version__,
                "device": torch.cuda.get_device_name(0)}
    identity["digest"] = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    (args.out_dir / "manifest.json").write_text(json.dumps(
        {"identity": identity, "readout_npz_sha256": readout_sha, "samples": n,
         "determinism_max_abs_diff_fp16": determinism, "seconds": time.time() - started,
         "joint_valid_counts": joint_valid.sum(axis=0).tolist()}, indent=1, sort_keys=True))
    print(json.dumps({"samples": n, "seconds": round(time.time() - started, 1), "determinism": determinism,
                      "merger_fc2": merger_fc2, "readout_sha256": readout_sha}))


if __name__ == "__main__":
    main()
