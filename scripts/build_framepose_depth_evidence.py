#!/usr/bin/env python3
"""DIAGNOSTIC (Worklog 65): Depth Anything V2 vits evidence for every FrameBank sample.

Each image is read once as bytes, hashed, decoded from those same bytes and
passed to the unchanged legacy ``pose.depth_estimator.DepthEstimator`` (vits,
input 518).  Every sample of that image is then sampled at its own benchmark
2D joints.  The cache is keyed by image SHA-256, checkpoint SHA-256, encoder,
input size and sampling/normalization policy; never by filename alone.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from framepose.bank import load_bank
from framepose.depth_evidence import (
    ENCODER, INPUT_SIZE, cache_identity, forward_depth_evidence, model_provenance, sample_nearest,
    sha256_bytes,
)
from pose.depth_estimator import DepthEstimator, DepthEstimatorConfig

UPSTREAM = {"repository": "huggingface.co/depth-anything/Depth-Anything-V2-Small",
            "revision": "03876f8651c73a60fe4c2c48294e09fcb6838fcf",
            "file": "depth_anything_v2_vits.pth",
            "lfs_sha256": "715fade13be8f229f8a70cc02066f656f2423a59effd0579197bbf57860e1378",
            "license": "apache-2.0"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True, help="root for image_reference 3dpw_images")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--code-dir", type=Path, required=True, help="third_party/Depth-Anything-V2 checkout")
    parser.add_argument("--code-revision", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    import torch

    checkpoint_sha = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    if checkpoint_sha != UPSTREAM["lfs_sha256"]:
        raise SystemExit(f"checkpoint SHA-256 {checkpoint_sha} is not the official vits file")
    bank = load_bank(args.bank)
    estimator = DepthEstimator(DepthEstimatorConfig(str(args.checkpoint), encoder=ENCODER,
                                                    device="auto", input_size=INPUT_SIZE))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    provenance = model_provenance(checkpoint_sha, code_revision=args.code_revision,
                                  torch_version=torch.__version__, device=device,
                                  checkpoint_path=str(args.checkpoint), upstream=UPSTREAM)
    input_digest = hashlib.sha256(bank.arrays["input_2d"].tobytes() + bank.arrays["input_valid"].tobytes()).hexdigest()
    identity = cache_identity(provenance, bank.content_digest(), input_digest)

    by_image = defaultdict(list)
    for position, sample in enumerate(bank.samples):
        if sample.image_reference is None or sample.image_reference.root_key != "3dpw_images":
            raise SystemExit(f"{sample.sample_id}: missing 3dpw_images reference")
        by_image[sample.image_reference.relative_path].append(position)

    n = len(bank)
    raw = np.zeros((n, 17), np.float32)
    sampled = np.zeros((n, 17), bool)
    forward = np.zeros((n, 17), np.float32)
    available = np.zeros((n, 17), bool)
    frame_ok = np.zeros(n, bool)
    image_sha = np.empty(n, dtype="<U64")
    image_index = {}
    started = time.time()
    for k, (relative, positions) in enumerate(sorted(by_image.items())):
        data = (args.image_root / relative).read_bytes()
        digest = sha256_bytes(data)
        image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)  # BGR, as upstream run.py
        depth = estimator.infer_frame(image)
        image_index[relative] = digest
        for p in positions:
            sample = bank.samples[p]
            if (image.shape[1], image.shape[0]) != tuple(sample.image_size):
                raise SystemExit(f"{sample.sample_id}: image size differs from the bank")
            r, s = sample_nearest(depth, bank.arrays["input_2d"][p], bank.arrays["input_valid"][p],
                                  sample.image_size)
            f, ok, frame = forward_depth_evidence(r, s)
            raw[p], sampled[p], forward[p], available[p], frame_ok[p] = r, s, f, ok, frame
            image_sha[p] = digest
        if k % 1000 == 0:
            print(json.dumps({"images": k, "of": len(by_image), "seconds": round(time.time() - started, 1)}), flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out / "evidence.npz", raw_depth=raw, sampled=sampled, forward_evidence=forward,
                        available=available, frame_available=frame_ok, image_sha256=image_sha)
    manifest = {
        "schema": "animcv_framepose_depth_evidence_cache_v1", "identity": identity, "provenance": provenance,
        "code_dir_git": subprocess.run(["git", "-C", str(args.code_dir), "rev-parse", "HEAD"],
                                       capture_output=True, text=True).stdout.strip() or args.code_revision,
        "samples": n, "unique_images": len(by_image),
        "sample_ids_sha256": sha256_bytes("\n".join(s.sample_id for s in bank.samples).encode()),
        "frames_available": int(frame_ok.sum()), "joints_sampled": int(sampled.sum()),
        "joints_input_valid": int(bank.arrays["input_valid"].sum()),
        "per_split_frames_available": {split: int(frame_ok[bank.indices(split)].sum())
                                       for split in ("train", "validation", "test")},
        "image_sha256_by_relative_path": image_index,
        "evidence_npz_sha256": hashlib.sha256((args.out / "evidence.npz").read_bytes()).hexdigest(),
        "inference_seconds": time.time() - started,
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True))
    print(json.dumps({k: v for k, v in manifest.items() if k != "image_sha256_by_relative_path"}, indent=1))


if __name__ == "__main__":
    main()
