#!/usr/bin/env python3
"""Why did the sign advisor answer a constant branch? — a controlled diagnosis.

Two controls, each changing exactly one thing against the frozen historical
prompt:

    prompt mode   combined (all seven questions in one response, historical)
                  vs isolated_question_v1 (one historical question per request,
                  wording and options copied verbatim)

    image         the correct person crop
                  vs another person's crop from the same subset, with the
                  original sample's question and oracle label kept

Nothing else moves: same model, same weights, same decoding, same crop contract,
same frames. No prompt was rewritten, no decoding parameter swept, no pose model
touched. This produces sensor evidence only.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from common.serialization import write_json
from framepose.bank import load_bank
from framepose.features import read_crop
from framepose.observations import image_content_digest
from framepose.sign_advisor import (
    ADVISOR_CROP_RESOLUTION, isolated_prompt_provenance, isolated_prompt_text, parse_response,
    prompt_provenance, prompt_text, resolve_snapshot,
)
from framepose.signs import NEGATIVE, POSITIVE, SIGN_FIELD_NAMES, UNKNOWN, oracle_sign_states


def balanced_subset(bank, oracle: np.ndarray, splits: list[str], count: int) -> np.ndarray:
    """A deterministic stride subset that contains both branches of every field.

    Stride, not selection by model behaviour: the subset is a function of the
    bank alone. It is rejected outright if any field lacks a branch, so a
    constant answer can never look competent on it.
    """
    pool = np.concatenate([bank.indices(split) for split in splits])
    step = max(len(pool) / count, 1.0)
    picked = pool[np.unique(np.floor(np.arange(count) * step).astype(np.int64))]
    for index, name in enumerate(SIGN_FIELD_NAMES):
        values = oracle[picked, index]
        if not ((values == POSITIVE).any() and (values == NEGATIVE).any()):
            raise ValueError(f"subset lacks both branches for {name}; enlarge --frames")
    return picked


def shuffled_partner(order: int, total: int) -> int:
    """Fixed derangement: pair each sample with the one half a subset away."""
    return (order + total // 2) % total


def _entropy(counts: dict[int, int]) -> float:
    total = sum(counts.values())
    if not total:
        return 0.0
    return float(-sum((n / total) * math.log2(n / total) for n in counts.values() if n))


def _field_metrics(predicted: np.ndarray, reference: np.ndarray) -> dict:
    scored = reference != UNKNOWN
    truth = reference[scored]
    guess = predicted[scored]
    counts = {value: int((predicted == value).sum()) for value in (NEGATIVE, UNKNOWN, POSITIVE)}
    confusion = {f"oracle{t:+d}_pred{p:+d}": int(((truth == t) & (guess == p)).sum())
                 for t in (NEGATIVE, POSITIVE) for p in (NEGATIVE, UNKNOWN, POSITIVE)}
    per_class = []
    for value in (NEGATIVE, POSITIVE):
        mask = truth == value
        if mask.any():
            per_class.append(float((guess[mask] == value).mean()))
    majority = 0.0
    if truth.size:
        majority = float(max((truth == POSITIVE).mean(), (truth == NEGATIVE).mean()))
    return {
        "oracle_positive": int((reference == POSITIVE).sum()),
        "oracle_negative": int((reference == NEGATIVE).sum()),
        "oracle_degenerate": int((reference == UNKNOWN).sum()),
        "predicted_positive": counts[POSITIVE],
        "predicted_negative": counts[NEGATIVE],
        "predicted_unknown": counts[UNKNOWN],
        "scored": int(truth.size),
        "accuracy": float((guess == truth).mean()) if truth.size else None,
        "majority_class_baseline": majority if truth.size else None,
        "balanced_accuracy": float(np.mean(per_class)) if per_class else None,
        "confusion": confusion,
        "output_entropy_bits": _entropy(counts),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Diagnose the sign advisor's constant answers")
    parser.add_argument("--bank", required=True, type=Path)
    parser.add_argument("--image-root", action="append", required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--split", action="append", default=None)
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--model", default="Qwen/Qwen2-VL-2B-Instruct")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--hf-cache", default=None)
    args = parser.parse_args()

    import torch
    import transformers
    from PIL import Image
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

    roots = dict(value.split("=", 1) for value in args.image_root)
    bank = load_bank(args.bank)
    splits = args.split or ["validation", "test"]
    oracle = oracle_sign_states(bank.arrays["target_3d"], bank.arrays["target_valid"])
    positions = balanced_subset(bank, oracle, splits, args.frames)

    torch.manual_seed(args.seed)
    processor = AutoProcessor.from_pretrained(args.model, revision=args.revision)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model, revision=args.revision, torch_dtype=torch.float16, device_map=args.device)
    model.eval()

    crops = {int(position): read_crop(bank, int(position), roots, ADVISOR_CROP_RESOLUTION)
             for position in positions}

    def ask(image: np.ndarray, instruction: str, fields: tuple[str, ...]):
        messages = [{"role": "user", "content": [{"type": "image"},
                                                 {"type": "text", "text": instruction}]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[Image.fromarray(image)],
                           return_tensors="pt").to(model.device)
        with torch.no_grad():
            generated = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
        reply = processor.batch_decode(generated[:, inputs["input_ids"].shape[1]:],
                                       skip_special_tokens=True)[0]
        return parse_response(reply, fields=fields), reply

    modes = ("combined_real", "combined_shuffled", "isolated_real", "isolated_shuffled")
    predictions = {mode: np.zeros((len(positions), len(SIGN_FIELD_NAMES)), dtype=np.int8)
                   for mode in modes}
    invalid = {mode: 0 for mode in modes}
    records = []
    combined_instruction = prompt_text()
    started = time.perf_counter()

    for order, position in enumerate(positions):
        position = int(position)
        partner = int(positions[shuffled_partner(order, len(positions))])
        sample = bank.samples[position]
        entry = {
            "sample_id": sample.sample_id, "bank_position": position, "split": sample.split,
            "image_reference": sample.image_reference.to_dict(),
            "image_content_digest": image_content_digest(sample.image_reference.resolve(roots)),
            "shuffled_donor_sample_id": bank.samples[partner].sample_id,
            "oracle": {name: int(value) for name, value in zip(SIGN_FIELD_NAMES, oracle[position])},
            "answers": {},
        }
        for mode, image in (("combined_real", crops[position]), ("combined_shuffled", crops[partner])):
            response, reply = ask(image, combined_instruction, tuple(SIGN_FIELD_NAMES))
            predictions[mode][order] = response.state
            invalid[mode] += int(not response.valid)
            entry["answers"][mode] = {"valid": response.valid, "reason": response.reason,
                                      "raw": reply[:400],
                                      "state": {n: int(v) for n, v in
                                                zip(SIGN_FIELD_NAMES, response.state)}}
        for mode, image in (("isolated_real", crops[position]),
                            ("isolated_shuffled", crops[partner])):
            state = np.zeros(len(SIGN_FIELD_NAMES), dtype=np.int8)
            details = {}
            for index, name in enumerate(SIGN_FIELD_NAMES):
                response, reply = ask(image, isolated_prompt_text(name), (name,))
                if not response.valid:
                    invalid[mode] += 1
                state[index] = response.state[index]
                details[name] = {"valid": response.valid, "reason": response.reason,
                                 "raw": reply[:200], "value": int(response.state[index])}
            predictions[mode][order] = state
            entry["answers"][mode] = {"per_field": details,
                                      "state": {n: int(v) for n, v in zip(SIGN_FIELD_NAMES, state)}}
        records.append(entry)
        if (order + 1) % 10 == 0:
            rate = (order + 1) / max(time.perf_counter() - started, 1e-9)
            print(f"{order + 1}/{len(positions)} frames, {rate:.3f} frames/s", flush=True)

    reference = oracle[positions]
    report = {
        "schema": "animcv_sign_advisor_diagnostic_v1",
        "bank_content_digest": bank.content_digest(),
        "splits": splits,
        "frame_count": int(len(positions)),
        "subset_class_counts": {
            name: {"positive": int((reference[:, i] == POSITIVE).sum()),
                   "negative": int((reference[:, i] == NEGATIVE).sum()),
                   "degenerate": int((reference[:, i] == UNKNOWN).sum())}
            for i, name in enumerate(SIGN_FIELD_NAMES)},
        "model": {"id": args.model, "requested_revision": args.revision,
                  "transformers_version": transformers.__version__,
                  "torch_version": torch.__version__, "dtype": "float16",
                  "parameter_count": int(sum(p.numel() for p in model.parameters())),
                  "generation": {"max_new_tokens": args.max_new_tokens, "do_sample": False,
                                 "seed": args.seed}},
        "prompts": {"combined": prompt_provenance(),
                    "isolated": {name: isolated_prompt_provenance(name)
                                 for name in SIGN_FIELD_NAMES}},
        "shuffle_rule": "deterministic derangement: subset order i pairs with (i + n//2) % n",
        "invalid_response_counts": invalid,
        "elapsed_seconds": time.perf_counter() - started,
        "metrics": {},
    }
    if args.hf_cache:
        snapshot = resolve_snapshot(args.model, args.hf_cache)
        report["model"]["snapshot"] = (
            {key: snapshot[key] for key in ("resolved_commit", "refs", "file_count",
                                            "weight_fingerprint")} if snapshot else None)
        report["model"]["exact_weight_fingerprint_established"] = snapshot is not None

    for mode in modes:
        report["metrics"][mode] = {
            name: _field_metrics(predictions[mode][:, index], reference[:, index])
            for index, name in enumerate(SIGN_FIELD_NAMES)}
    report["grounding"] = {}
    for prompt_mode, real, shuffled in (("combined", "combined_real", "combined_shuffled"),
                                        ("isolated", "isolated_real", "isolated_shuffled")):
        report["grounding"][prompt_mode] = {
            name: {
                "real_vs_shuffled_agreement": float(
                    (predictions[real][:, index] == predictions[shuffled][:, index]).mean()),
                "prediction_change_rate_under_shuffle": float(
                    (predictions[real][:, index] != predictions[shuffled][:, index]).mean()),
            } for index, name in enumerate(SIGN_FIELD_NAMES)}

    args.out.mkdir(parents=True, exist_ok=True)
    np.savez(args.out / "predictions.npz", positions=positions, reference=reference,
             **{mode: predictions[mode] for mode in modes})
    write_json(args.out / "diagnostic.json", report)
    write_json(args.out / "diagnostic_records.json",
               {"schema": "animcv_sign_advisor_diagnostic_records_v1", "records": records})
    print(json.dumps({mode: {name: {
        "acc": report["metrics"][mode][name]["accuracy"],
        "bal": report["metrics"][mode][name]["balanced_accuracy"],
        "H": round(report["metrics"][mode][name]["output_entropy_bits"], 3),
    } for name in SIGN_FIELD_NAMES} for mode in modes}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
