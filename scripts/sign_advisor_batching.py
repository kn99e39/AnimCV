"""Execution-only batching helpers for the frozen Sign Advisor evaluation.

These helpers preserve crop bytes and pair order. They do not own or alter the
prompt, model, decoder, parser, population, or real-versus-shuffled assignment.
"""

from __future__ import annotations

import hashlib
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Sequence

import numpy as np


def interleave_pairs(real: Sequence[Any], shuffled: Sequence[Any]) -> list[Any]:
    """Return real_1, shuffled_1, real_2, shuffled_2, ... in receiver order."""
    if len(real) != len(shuffled):
        raise ValueError("real and shuffled batches must have the same length")
    return [item for pair in zip(real, shuffled) for item in pair]


def split_interleaved(values: Sequence[Any]) -> tuple[list[Any], list[Any]]:
    """Restore interleaved values to aligned real and shuffled receiver rows."""
    if len(values) % 2:
        raise ValueError("an interleaved real/shuffled batch must have even length")
    return list(values[::2]), list(values[1::2])


def crop_sha256(crop: np.ndarray) -> str:
    """Hash the exact C-order RGB bytes used by the sequential evaluator."""
    return hashlib.sha256(crop.tobytes()).hexdigest()


@dataclass
class PreparedPairBatch:
    entries: list[Any]
    crop_pairs: list[tuple[np.ndarray, np.ndarray]]
    crop_hash_pairs: list[tuple[str, str]]
    inputs: Any
    crop_read_seconds: float
    pil_conversion_seconds: float
    processor_seconds: float


class OrderedBatchPreparer:
    """Prepare one bounded, ordered batch of receiver/control crop pairs."""

    def __init__(self, read_pair: Callable[[Any], tuple[np.ndarray, np.ndarray]],
                 processor: Any, rendered_prompt: str, *, crop_workers: int = 4,
                 crop_resolution: int = 448):
        if crop_workers < 1:
            raise ValueError("crop_workers must be positive")
        self._read_pair = read_pair
        self._processor = processor
        self._rendered_prompt = rendered_prompt
        self._crop_resolution = crop_resolution
        self._pool = ThreadPoolExecutor(max_workers=crop_workers,
                                        thread_name_prefix="sign-advisor-crop")

    def close(self) -> None:
        self._pool.shutdown(wait=True)

    def prepare(self, entries: Sequence[Any]) -> PreparedPairBatch:
        if not entries:
            raise ValueError("cannot prepare an empty frame batch")
        started = time.perf_counter()
        pairs = list(self._pool.map(self._read_pair, entries))
        crop_read_seconds = time.perf_counter() - started
        for pair in pairs:
            if len(pair) != 2:
                raise ValueError("each receiver must produce exactly real and shuffled crops")
            for crop in pair:
                if (not isinstance(crop, np.ndarray) or crop.ndim != 3
                        or crop.shape != (self._crop_resolution,
                                          self._crop_resolution, 3)
                        or crop.dtype != np.uint8):
                    raise ValueError("crop bytes/shape differ from the frozen RGB crop contract")

        crop_hash_pairs = [(crop_sha256(real), crop_sha256(shuffled))
                           for real, shuffled in pairs]
        crops = interleave_pairs([pair[0] for pair in pairs],
                                 [pair[1] for pair in pairs])

        from PIL import Image

        started = time.perf_counter()
        images = [Image.fromarray(crop) for crop in crops]
        pil_conversion_seconds = time.perf_counter() - started
        started = time.perf_counter()
        inputs = self._processor(
            text=[self._rendered_prompt] * len(images),
            images=images,
            return_tensors="pt",
        )
        processor_seconds = time.perf_counter() - started
        return PreparedPairBatch(
            entries=list(entries), crop_pairs=pairs,
            crop_hash_pairs=crop_hash_pairs, inputs=inputs,
            crop_read_seconds=crop_read_seconds,
            pil_conversion_seconds=pil_conversion_seconds,
            processor_seconds=processor_seconds,
        )


def prefetch_ordered_batches(batches: Iterable[Sequence[Any]],
                             preparer: OrderedBatchPreparer):
    """Prefetch at most one next batch without changing the input order."""
    iterator = iter(batches)
    with ThreadPoolExecutor(max_workers=1,
                            thread_name_prefix="sign-advisor-prefetch") as pool:
        try:
            current = next(iterator)
        except StopIteration:
            return
        pending = pool.submit(preparer.prepare, current)
        while True:
            prepared = pending.result()
            try:
                following = next(iterator)
            except StopIteration:
                yield prepared
                return
            pending = pool.submit(preparer.prepare, following)
            yield prepared


def generate_prepared_batch(prepared: PreparedPairBatch, *, processor: Any,
                            model: Any, torch: Any, parse_response: Callable[..., Any],
                            fields: Sequence[str], max_new_tokens: int
                            ) -> tuple[list[Any], list[str], dict[str, float]]:
    """Run one greedy batch and return aligned parse objects and raw strings."""
    timings = {
        "crop_read_seconds": prepared.crop_read_seconds,
        "pil_conversion_seconds": prepared.pil_conversion_seconds,
        "processor_seconds": prepared.processor_seconds,
    }

    def synchronize() -> None:
        if torch.cuda.is_available() and str(model.device).startswith("cuda"):
            torch.cuda.synchronize(model.device)

    started = time.perf_counter()
    synchronize()
    device_inputs = prepared.inputs.to(model.device)
    synchronize()
    timings["host_to_device_seconds"] = time.perf_counter() - started

    started = time.perf_counter()
    with torch.inference_mode():
        generated = model.generate(
            **device_inputs, max_new_tokens=max_new_tokens, do_sample=False)
    synchronize()
    timings["generate_seconds"] = time.perf_counter() - started

    started = time.perf_counter()
    output_tokens = generated[:, device_inputs["input_ids"].shape[1]:]
    raw_responses = processor.batch_decode(output_tokens, skip_special_tokens=True)
    parsed_responses = [parse_response(raw, fields=tuple(fields))
                        for raw in raw_responses]
    timings["decode_parse_seconds"] = time.perf_counter() - started
    if len(parsed_responses) != 2 * len(prepared.entries):
        raise AssertionError("generation response count does not match interleaved frame pairs")
    return parsed_responses, raw_responses, timings


def compare_output_rows(reference_rows: dict[str, dict[str, Any]],
                        candidate_rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Compare both control modes and return explicit raw/parse/state equivalence."""
    seen: set[str] = set()
    raw_matches = parse_matches = state_matches = relevant_matches = 0
    raw_mismatches: list[dict[str, str]] = []
    relevant_mismatches: list[dict[str, str]] = []
    request_count = 0
    for candidate in candidate_rows:
        sample_id = candidate["sample_id"]
        if sample_id in seen:
            raise ValueError(f"duplicate candidate sample_id: {sample_id}")
        seen.add(sample_id)
        if sample_id not in reference_rows:
            raise ValueError(f"candidate sample_id absent from sequential reference: {sample_id}")
        reference = reference_rows[sample_id]
        for mode in ("real", "shuffled"):
            old, new = reference[mode], candidate[mode]
            request_count += 1
            if old["raw"] == new["raw"]:
                raw_matches += 1
            else:
                raw_mismatches.append({"sample_id": sample_id, "mode": mode})
            if (old["valid"], old.get("reason")) == (new["valid"], new.get("reason")):
                parse_matches += 1
            relevant_equal = (
                old["valid"] == new["valid"]
                and old.get("reason") == new.get("reason")
                and old["state"] == new["state"]
            )
            if old["state"] == new["state"]:
                state_matches += 1
            if relevant_equal:
                relevant_matches += 1
            else:
                relevant_mismatches.append({"sample_id": sample_id, "mode": mode})
    denominator = request_count or 1
    return {
        "rows": len(seen),
        "requests": request_count,
        "raw_exact_matches": raw_matches,
        "raw_exact_match_rate": raw_matches / denominator if request_count else None,
        "parse_status_exact_matches": parse_matches,
        "parse_status_exact_match_rate": parse_matches / denominator if request_count else None,
        "state_exact_matches": state_matches,
        "state_exact_match_rate": state_matches / denominator if request_count else None,
        "evaluation_relevant_exact_matches": relevant_matches,
        "evaluation_relevant_exact_match_rate": (
            relevant_matches / denominator if request_count else None),
        "raw_mismatches": raw_mismatches,
        "evaluation_relevant_mismatches": relevant_mismatches,
    }


def validate_resume_identity(reference: dict[str, Any], current: dict[str, Any]) -> None:
    """Refuse resuming a run under a different evaluator or execution mode."""
    ignored = {"record_type", "run_started_utc"}
    mismatched = [key for key in set(current) - ignored
                  if reference.get(key) != current[key]]
    if mismatched:
        raise ValueError(f"resume provenance mismatch in fields: {sorted(mismatched)}")
