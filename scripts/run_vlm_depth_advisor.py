#!/usr/bin/env python3
"""DIAGNOSTIC (Worklog 67): run Qwen3-VL-8B (4-bit NF4, greedy) as a left-arm near/far sensor.

Real advisor images for every eligible held-out left-arm row (the Worklog 65/66
rows) and shuffled-RGB controls for the runtime H0-vs-Depth-Anything
disagreement population.  Results are appended to a JSONL cache keyed by
(kind, sample_id, advisor image SHA, model fingerprint, prompt SHA, generation)
and the run is resumable.  GT never enters any input.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from framepose.arm_depth_probe import SEGMENT_NAMES
from framepose.bank import load_bank
from framepose.relational_depth_fusion import depth_relations
from framepose.vlm_depth_advisor import (
    ADVISOR_RESOLUTION, GENERATION, MODEL_REPOSITORY, MODEL_REVISION, PROMPT, PROMPT_SHA256, QUANTIZATION,
    advisor_image, parse_response, runtime_disagreement, sha256_bytes, shuffled_donors,
)


def model_manifest(snapshot: Path) -> dict:
    files = {}
    for path in sorted(p for p in snapshot.iterdir() if p.is_file() or p.is_symlink()):
        data = path.resolve().read_bytes()
        files[path.name] = {"sha256": sha256_bytes(data), "bytes": len(data)}
    aggregate = sha256_bytes(json.dumps(files, sort_keys=True).encode())
    return {"repository": MODEL_REPOSITORY, "revision": MODEL_REVISION, "snapshot": str(snapshot),
            "files": files, "aggregate_sha256": aggregate}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("bank", "image_root", "evidence_dir", "w65_dir", "w66_report", "snapshot", "out_dir"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None, help="smoke runs only")
    args = parser.parse_args()
    import cv2
    import torch
    import transformers
    from PIL import Image
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration

    if args.snapshot.name != MODEL_REVISION:
        raise SystemExit("snapshot directory is not the pinned revision")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out_dir / "model_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
    else:
        manifest = model_manifest(args.snapshot)
        manifest.update({"transformers": transformers.__version__, "torch": torch.__version__,
                         "quantization": QUANTIZATION, "generation": GENERATION, "device": torch.cuda.get_device_name(0),
                         "prompt": PROMPT, "prompt_sha256": PROMPT_SHA256, "advisor_resolution": ADVISOR_RESOLUTION,
                         "batch_size": args.batch_size})
        manifest_path.write_text(json.dumps(manifest, indent=1, sort_keys=True))
    fingerprint = manifest["aggregate_sha256"]

    bank = load_bank(args.bank)
    evidence = np.load(args.evidence_dir / "evidence.npz")
    relations, relation_ok = depth_relations(evidence["forward_evidence"], evidence["available"])
    rows = json.loads((args.w65_dir / "rows_validation.json").read_text()) + \
        json.loads((args.w65_dir / "rows_test.json").read_text())
    population = []
    for r in rows:
        p = r["position"]
        disagree = {name: runtime_disagreement(r[f"{name}_f_h0"], float(relations[p, k]) if relation_ok[p, k] else None)
                    for k, name in enumerate(SEGMENT_NAMES)}
        population.append({"position": p, "sample_id": bank.samples[p].sample_id,
                           "sequence_id": r["sequence_id"], "frame_index": r["frame_index"],
                           "disagree": disagree, "primary": any(disagree.values())})
    primary = [e for e in population if e["primary"]]
    donors = shuffled_donors([(e["sequence_id"], e["frame_index"]) for e in primary])
    (args.out_dir / "population.json").write_text(json.dumps(
        {"rows": population, "primary_count": len(primary),
         "shuffled_donor_sample_id": {e["sample_id"]: primary[d]["sample_id"] for e, d in zip(primary, donors)}}))
    owner_keys = {(c["sequence_id"], c["frame_index"]) for c in json.loads(args.w66_report.read_text())["owner_cases"].values()}

    jobs = [("real", e, e) for e in population] + [("shuffled", e, primary[d]) for e, d in zip(primary, donors)]
    if args.limit:
        jobs = jobs[:args.limit]
    results_path = args.out_dir / "responses.jsonl"
    done = set()
    if results_path.exists():
        for line in results_path.read_text().splitlines():
            item = json.loads(line)
            done.add((item["kind"], item["sample_id"]))
    jobs = [j for j in jobs if (j[0], j[1]["sample_id"]) not in done]

    processor = AutoProcessor.from_pretrained(args.snapshot)
    processor.tokenizer.padding_side = "left"
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.snapshot, device_map="cuda:0",
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                               bnb_4bit_compute_dtype=torch.bfloat16,
                                               bnb_4bit_use_double_quant=False))
    model.eval()
    image_cache = {}

    def image_for(position):
        sample = bank.samples[position]
        rel = sample.image_reference.relative_path
        if rel not in image_cache:
            if len(image_cache) > 64:
                image_cache.clear()
            data = (args.image_root / rel).read_bytes()
            bgr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            image_cache[rel] = (sha256_bytes(data), cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        return image_cache[rel]

    started = time.time()
    with results_path.open("a") as out:
        for start in range(0, len(jobs), args.batch_size):
            batch = jobs[start:start + args.batch_size]
            images, records = [], []
            for kind, target, source in batch:
                p = source["position"]
                sample = bank.samples[p]
                image_sha, rgb = image_for(p)
                rendered, crop = advisor_image(rgb, bank.arrays["input_2d"][p], bank.arrays["input_valid"][p],
                                               sample.image_size)
                images.append(Image.fromarray(rendered))
                records.append({"kind": kind, "sample_id": target["sample_id"], "image_source_sample_id": source["sample_id"],
                                "image_sha256": image_sha, "crop": crop, "advisor_image_sha256": crop["rendered_sha256"],
                                "model_fingerprint": fingerprint, "prompt_sha256": PROMPT_SHA256})
                if kind == "real" and (target["sequence_id"], target["frame_index"]) in owner_keys:
                    (args.out_dir / "owner_images").mkdir(exist_ok=True)
                    Image.fromarray(rendered).save(args.out_dir / "owner_images" /
                                                   f"{target['sequence_id'].split(':')[1]}_{target['frame_index']}.png")
            messages = [[{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}]
                        for _ in batch]
            texts = [processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in messages]
            inputs = processor(text=texts, images=images, return_tensors="pt", padding=True).to(model.device)
            with torch.no_grad():
                generated = model.generate(**inputs, **GENERATION)
            outputs = processor.batch_decode(generated[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            for record, text in zip(records, outputs):
                parsed = parse_response(text)
                record.update({"raw_text": text, "parse_status": parsed.status, "fence_normalized": parsed.fence_normalized,
                               "states": parsed.states})
                out.write(json.dumps(record) + "\n")
            out.flush()
            if (start // args.batch_size) % 25 == 0:
                print(json.dumps({"done": start + len(batch), "of": len(jobs),
                                  "seconds": round(time.time() - started, 1)}), flush=True)
    print(json.dumps({"finished": len(jobs), "seconds": round(time.time() - started, 1),
                      "primary_population": len(primary), "all_rows": len(population)}))


if __name__ == "__main__":
    main()
