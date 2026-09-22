#!/usr/bin/env python3
"""Run Track A: one-frame VLM joint-reliability evidence and shuffle control.

The VLM sees only a deterministic RGB crop and this fixed prompt.  Detector
confidence, 2D coordinates, target 3D, and temporal neighbours never enter the
request.  The structural labels in the report are deliberately separate:
out-of-frame is exact geometry, detector-missing is an existing observation
contract fact, and no OCCLUDED ground truth is invented.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from common.serialization import write_json
from framepose.bank import load_bank
from framepose.features import read_crop
from framepose.observations import image_content_digest
from framepose.reliability_advisor import (
    ADVISOR_CROP_RESOLUTION, INDEX_TO_STATE, JOINT_NAMES, RELIABILITY_SCHEMA, STATE_TO_INDEX,
    parse_response, prompt_provenance, prompt_text,
)
from framepose.sign_advisor import resolve_snapshot


def _structural_labels(observation: np.ndarray, valid: np.ndarray) -> dict[str, np.ndarray]:
    xy = np.asarray(observation[..., :2], dtype=np.float64)
    finite = np.isfinite(xy).all(axis=-1)
    exact_out = valid & finite & ((xy[..., 0] < 0.0) | (xy[..., 0] > 1.0)
                                  | (xy[..., 1] < 0.0) | (xy[..., 1] > 1.0))
    in_frame = valid & finite & ~exact_out & (xy[..., 0] >= 0.0) & (xy[..., 0] <= 1.0) \
        & (xy[..., 1] >= 0.0) & (xy[..., 1] <= 1.0)
    return {"in_frame": in_frame, "exact_out_of_frame": exact_out,
            "detector_missing": ~valid, "structural_invalid": ~in_frame}


def _select_positions(bank, splits: list[str], frames: int, include_ids: list[str]) -> np.ndarray:
    pool = np.concatenate([bank.indices(split) for split in splits])
    explicit = [bank.position(identifier) for identifier in include_ids]
    explicit_set = set(explicit)
    if frames < len(explicit):
        frames = len(explicit)
    remaining = np.asarray([position for position in pool if int(position) not in explicit_set], dtype=np.int64)
    count = min(max(frames - len(explicit), 0), len(remaining))
    if count:
        step = max(len(remaining) / count, 1.0)
        chosen = remaining[np.unique(np.floor(np.arange(count) * step).astype(np.int64))]
    else:
        chosen = np.asarray([], dtype=np.int64)
    positions = np.asarray(explicit + [int(value) for value in chosen], dtype=np.int64)
    if len(positions) < 2:
        raise ValueError("Track A needs at least two frames for the deterministic RGB shuffle")
    return positions


def _shuffle_partner(order: int, total: int) -> int:
    return (order + total // 2) % total


def _confusion(truth: np.ndarray, predicted: np.ndarray) -> dict[str, Any]:
    truth = np.asarray(truth, dtype=bool)
    predicted = np.asarray(predicted, dtype=bool)
    if truth.shape != predicted.shape:
        raise ValueError("confusion input shape mismatch")
    tp = int(np.count_nonzero(truth & predicted))
    tn = int(np.count_nonzero(~truth & ~predicted))
    fp = int(np.count_nonzero(~truth & predicted))
    fn = int(np.count_nonzero(truth & ~predicted))
    sensitivity = tp / (tp + fn) if tp + fn else None
    specificity = tn / (tn + fp) if tn + fp else None
    return {"true_positive": tp, "true_negative": tn, "false_positive": fp,
            "false_negative": fn,
            "balanced_accuracy": ((sensitivity + specificity) / 2.0
                                   if sensitivity is not None and specificity is not None else None),
            "recall": sensitivity, "specificity": specificity}


def _state_counts(values: np.ndarray) -> dict[str, int]:
    return {name: int(np.count_nonzero(values == index)) for index, name in INDEX_TO_STATE.items()}


def _metric_by_joint(predicted: np.ndarray, labels: dict[str, np.ndarray]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for joint, name in enumerate(JOINT_NAMES):
        prediction = predicted[:, joint]
        exact_known = labels["in_frame"][:, joint] | labels["exact_out_of_frame"][:, joint]
        predicted_oof = prediction == STATE_TO_INDEX["OUT_OF_FRAME"]
        structural_prediction = predicted_oof | (prediction == STATE_TO_INDEX["OCCLUDED"])
        output[name] = {
            "state_counts": _state_counts(prediction),
            "non_unknown_coverage": float(np.mean(prediction != STATE_TO_INDEX["UNKNOWN"])),
            "exact_out_of_frame": _confusion(labels["exact_out_of_frame"][:, joint][exact_known],
                                               predicted_oof[exact_known]),
            "structural_invalid_proxy": _confusion(labels["structural_invalid"][:, joint],
                                                     structural_prediction),
            "detector_missing_exact_count": int(np.count_nonzero(labels["detector_missing"][:, joint])),
            "in_frame_reference_count": int(np.count_nonzero(labels["in_frame"][:, joint])),
        }
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", required=True, type=Path)
    parser.add_argument("--image-root", action="append", required=True, help="KEY=PATH")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--split", action="append", default=None)
    parser.add_argument("--frames", type=int, default=128)
    parser.add_argument("--include-sample-id", action="append", default=[])
    parser.add_argument("--model", default="Qwen/Qwen2-VL-2B-Instruct")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=180)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--hf-cache", default=None)
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError(f"refusing to overwrite Track A output: {args.out}")
    roots = dict(value.split("=", 1) for value in args.image_root)
    bank = load_bank(args.bank)
    splits = args.split or ["validation", "test"]
    positions = _select_positions(bank, splits, args.frames, args.include_sample_id)
    labels = _structural_labels(bank.arrays["input_2d"][positions], bank.arrays["input_valid"][positions])
    crops = {int(position): read_crop(bank, int(position), roots, ADVISOR_CROP_RESOLUTION)
             for position in positions}

    if args.hf_cache:
        os.environ.setdefault("HF_HOME", args.hf_cache)
        os.environ.setdefault("TRANSFORMERS_CACHE", str(Path(args.hf_cache) / "hub"))
    import torch
    import transformers
    from PIL import Image
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

    torch.manual_seed(args.seed)
    processor = AutoProcessor.from_pretrained(args.model, revision=args.revision)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model, revision=args.revision, torch_dtype=torch.float16, device_map=args.device)
    model.eval()

    def ask(image: np.ndarray):
        messages = [{"role": "user", "content": [{"type": "image"},
                                                     {"type": "text", "text": prompt_text()}]}]
        rendered = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[rendered], images=[Image.fromarray(image)], return_tensors="pt").to(model.device)
        with torch.no_grad():
            generated = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
        reply = processor.batch_decode(generated[:, inputs["input_ids"].shape[1]:],
                                       skip_special_tokens=True)[0]
        return parse_response(reply), reply

    modes = ("real", "shuffled")
    predictions = {mode: np.full((len(positions), len(JOINT_NAMES)), STATE_TO_INDEX["UNKNOWN"], dtype=np.int8)
                   for mode in modes}
    invalid = {mode: 0 for mode in modes}
    records: list[dict[str, Any]] = []
    started = time.perf_counter()
    for order, position_value in enumerate(positions):
        position = int(position_value)
        partner = int(positions[_shuffle_partner(order, len(positions))])
        sample = bank.samples[position]
        entry: dict[str, Any] = {
            "sample_id": sample.sample_id,
            "bank_position": position,
            "split": sample.split,
            "image_reference": sample.image_reference.to_dict() if sample.image_reference else None,
            "image_content_digest": image_content_digest(sample.image_reference.resolve(roots)),
            "shuffled_donor_sample_id": bank.samples[partner].sample_id,
            "structural": {name: {joint_name: bool(values[order, joint])
                                   for joint, joint_name in enumerate(JOINT_NAMES)}
                            for name, values in labels.items()},
            "answers": {},
        }
        for mode, image in (("real", crops[position]), ("shuffled", crops[partner])):
            response, reply = ask(image)
            predictions[mode][order] = response.state
            invalid[mode] += int(not response.valid)
            entry["answers"][mode] = {
                "valid": response.valid, "reason": response.reason,
                "fence_normalized": response.fence_normalized, "raw": reply[:1000],
                "state": {name: INDEX_TO_STATE[int(value)]
                          for name, value in zip(JOINT_NAMES, response.state)},
            }
        records.append(entry)
        if (order + 1) % 10 == 0:
            rate = (order + 1) / max(time.perf_counter() - started, 1e-9)
            print(f"{order + 1}/{len(positions)} frames, {rate:.3f} frames/s", flush=True)

    report: dict[str, Any] = {
        "schema": RELIABILITY_SCHEMA,
        "track": "A_vlm_visual_reliability_advisor",
        "bank": {"path": str(args.bank), "content_digest": bank.content_digest(),
                 "provenance_fingerprint": bank.provenance_fingerprint(),
                 "regime": bank.regime(), "splits": splits,
                 "frame_count": int(len(positions))},
        "model": {"id": args.model, "requested_revision": args.revision,
                  "transformers_version": transformers.__version__, "torch_version": torch.__version__,
                  "dtype": "float16", "parameter_count": int(sum(p.numel() for p in model.parameters())),
                  "generation": {"max_new_tokens": args.max_new_tokens, "do_sample": False,
                                 "seed": args.seed}},
        "prompt": prompt_provenance(),
        "crop": {"resolution": ADVISOR_CROP_RESOLUTION,
                  "source": "framepose.features.read_crop / fixed crop contract"},
        "shuffle_rule": "deterministic derangement: subset order i pairs with (i + n//2) % n",
        "invalid_response_counts": invalid,
        "elapsed_seconds": time.perf_counter() - started,
        "reference_inventory": {
            "exact_geometry_label": "OUT_OF_FRAME from finite valid normalized coordinates outside [0,1]",
            "exact_observation_contract_label": "detector_missing from input_valid=false",
            "structural_invalid_proxy": "detector_missing OR OUT_OF_FRAME; not an occlusion label",
            "true_occlusion_label_available": False,
            "human_qualitative_annotation_used_as_metric_ground_truth": False,
            "detector_confidence": "recorded in bank but never sent to VLM; not promoted to reliability ground truth",
        },
        "metrics": {"real": _metric_by_joint(predictions["real"], labels)},
        "grounding": {
            name: {
                "real_vs_shuffled_agreement": float(np.mean(predictions["real"][:, index] == predictions["shuffled"][:, index])),
                "prediction_change_rate_under_shuffle": float(np.mean(predictions["real"][:, index] != predictions["shuffled"][:, index])),
            } for index, name in enumerate(JOINT_NAMES)
        },
    }
    if args.hf_cache:
        snapshot = resolve_snapshot(args.model, args.hf_cache)
        report["model"]["snapshot"] = ({key: snapshot[key] for key in
                                          ("resolved_commit", "refs", "file_count", "weight_fingerprint")}
                                         if snapshot else None)
        report["model"]["exact_weight_fingerprint_established"] = snapshot is not None

    args.out.mkdir(parents=True, exist_ok=True)
    np.savez(args.out / "predictions.npz", positions=positions,
             exact_out_of_frame=labels["exact_out_of_frame"], in_frame=labels["in_frame"],
             detector_missing=labels["detector_missing"], structural_invalid=labels["structural_invalid"],
             real=predictions["real"], shuffled=predictions["shuffled"])
    write_json(args.out / "diagnostic.json", report)
    write_json(args.out / "diagnostic_records.json", {"schema": RELIABILITY_SCHEMA + "_records_v1",
                                                       "records": records})
    print(json.dumps({"frames": len(positions), "invalid": invalid,
                      "real_unknown_fraction": float(np.mean(predictions["real"] == STATE_TO_INDEX["UNKNOWN"])),
                      "output": str(args.out)}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
