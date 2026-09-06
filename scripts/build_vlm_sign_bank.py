#!/usr/bin/env python3
"""Run the VLM Sign Advisor over a deterministic frame subset.

The advisor sees one person crop from frame *n* and returns only the structured
SignState. Nothing continuous, nothing temporal, no free-form prose persisted
into the pose path, and malformed output rejected rather than guessed.

The resulting bank is separate from the F1/F2 dense patch-token cache, which is
not reused as VLM evidence.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from common.serialization import write_json
from framepose.bank import load_bank
from framepose.crops import crop_box
from framepose.features import read_crop
from framepose.observations import image_content_digest
from framepose.sign_advisor import (
    ADVISOR_CROP_RESOLUTION, bank_metadata, parse_response, prompt_text,
)
from framepose.signs import SIGN_FIELD_COUNT, SIGN_FIELD_NAMES, agreement, oracle_sign_states


def deterministic_subset(bank, split: str, count: int, seed: int) -> np.ndarray:
    """A fixed, reproducible subset: evenly strided over the split's frames.

    Stride rather than random choice, so the subset is a function of the bank
    alone and stays spread across every sequence.
    """
    positions = bank.indices(split)
    if count <= 0 or count >= len(positions):
        return positions
    step = len(positions) / count
    picked = np.unique(np.floor(np.arange(count) * step).astype(np.int64))
    return positions[picked]


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a VLM sign bank for a frame subset")
    parser.add_argument("--bank", required=True, type=Path)
    parser.add_argument("--image-root", action="append", required=True, help="KEY=PATH")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--split", action="append", default=None,
                        help="split to cover (repeatable); default validation and test")
    parser.add_argument("--frames-per-split", type=int, default=1200)
    parser.add_argument("--model", default="Qwen/Qwen2-VL-2B-Instruct")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=0, help="throughput probe: stop after N frames")
    args = parser.parse_args()

    import torch
    from PIL import Image
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

    roots = dict(value.split("=", 1) for value in args.image_root)
    bank = load_bank(args.bank)
    splits = args.split or ["validation", "test"]

    torch.manual_seed(args.seed)
    processor = AutoProcessor.from_pretrained(args.model, revision=args.revision)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model, revision=args.revision, torch_dtype=torch.float16, device_map=args.device)
    model.eval()
    instruction = prompt_text()

    signs = np.zeros((len(bank), SIGN_FIELD_COUNT), dtype=np.int8)
    covered = np.zeros(len(bank), dtype=bool)
    records: list[dict] = []
    invalid = 0
    started = time.perf_counter()

    for split in splits:
        positions = deterministic_subset(bank, split, args.frames_per_split, args.seed)
        for order, position in enumerate(positions):
            if args.limit and len(records) >= args.limit:
                break
            sample = bank.samples[int(position)]
            crop = read_crop(bank, int(position), roots, ADVISOR_CROP_RESOLUTION)
            image = Image.fromarray(crop)
            messages = [{"role": "user", "content": [{"type": "image"},
                                                     {"type": "text", "text": instruction}]}]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[text], images=[image], return_tensors="pt").to(model.device)
            with torch.no_grad():
                generated = model.generate(**inputs, max_new_tokens=args.max_new_tokens,
                                           do_sample=False)
            reply = processor.batch_decode(
                generated[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
            response = parse_response(reply)
            if not response.valid:
                invalid += 1
            signs[position] = response.state
            covered[position] = True
            box = crop_box(bank.arrays["input_2d"][position], bank.arrays["input_valid"][position],
                           sample.image_size)
            records.append({
                "sample_id": sample.sample_id,
                "split": split,
                "bank_position": int(position),
                "image_reference": sample.image_reference.to_dict(),
                "image_content_digest": image_content_digest(
                    sample.image_reference.resolve(roots)),
                "crop_box": box.to_dict(),
                "predicted": {name: int(value) for name, value in zip(SIGN_FIELD_NAMES, response.state)},
                "valid_response": response.valid,
                "rejection_reason": response.reason,
                "raw_response": response.raw[:600],
            })
            if len(records) % 50 == 0:
                rate = len(records) / max(time.perf_counter() - started, 1e-9)
                print(f"{len(records)} frames, {rate:.2f} frames/s, {invalid} invalid", flush=True)

    elapsed = time.perf_counter() - started
    oracle = oracle_sign_states(bank.arrays["target_3d"], bank.arrays["target_valid"])
    for record in records:
        position = record["bank_position"]
        record["oracle"] = {name: int(value)
                            for name, value in zip(SIGN_FIELD_NAMES, oracle[position])}
        record["per_field_correct"] = {
            name: (None if record["oracle"][name] == 0
                   else bool(record["predicted"][name] == record["oracle"][name]))
            for name in SIGN_FIELD_NAMES}

    scored = np.flatnonzero(covered)
    metadata = bank_metadata(
        model_id=args.model, revision=args.revision,
        weights_sha256=None,
        parameter_count=sum(p.numel() for p in model.parameters()),
        dtype="float16",
        generation={"max_new_tokens": args.max_new_tokens, "do_sample": False, "seed": args.seed},
        bank_content_digest=bank.content_digest(), sample_count=int(len(scored)))
    metadata.update({
        "splits": splits,
        "frames_per_split": args.frames_per_split,
        "covered_positions": int(covered.sum()),
        "invalid_response_count": invalid,
        "invalid_response_rate": invalid / max(len(records), 1),
        "frames_per_second": len(records) / max(elapsed, 1e-9),
        "elapsed_seconds": elapsed,
        "accuracy_vs_oracle": agreement(signs[scored], oracle[scored]),
    })

    args.out.mkdir(parents=True, exist_ok=True)
    np.savez(args.out / "signs.npz", signs=signs, covered=covered)
    write_json(args.out / "sign_bank.json", {**metadata, "records": records})
    write_json(args.out / "sign_bank_summary.json", metadata)
    print(json.dumps({key: metadata[key] for key in
                      ("covered_positions", "invalid_response_count", "invalid_response_rate",
                       "frames_per_second")}, indent=2))
    print(json.dumps({name: metadata["accuracy_vs_oracle"][name]["agreement"]
                      for name in list(SIGN_FIELD_NAMES) + ["overall"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
