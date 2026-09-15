#!/usr/bin/env python3
"""Bounded throughput and output-equivalence benchmark for Sign Advisor batching.

Only execution grouping changes. The frozen model, weights, prompt, crop,
greedy decoder, parser, donor mapping, and sequential reference are verified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import evaluate_current_sign_advisor as evaluation
from common.serialization import write_json
from framepose.bank import load_bank
from framepose.features import read_crop
from framepose.sign_advisor import (
    ADVISOR_CROP_RESOLUTION,
    PROMPT_SCHEMA_VERSION,
    parse_response,
    prompt_provenance,
    prompt_text,
    resolve_snapshot,
)
from framepose.signs import SIGN_FIELD_NAMES
import framepose.features as feature_module
import framepose.sign_advisor as sign_advisor_module
from sign_advisor_batching import (
    OrderedBatchPreparer,
    compare_output_rows,
    crop_sha256,
    generate_prepared_batch,
    prefetch_ordered_batches,
)


BENCHMARK_FRAME_COUNTS = (4, 8, 16)
EQUIVALENCE_FRAMES = 64
PROFILE_FRAMES = 8
SAFE_HEADROOM_MIB = 2048
MATERIAL_GAIN = 0.10


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()


class GpuSampler:
    """Sample host-visible utilization and memory without blocking inference."""

    def __init__(self, interval_seconds: float = 0.5):
        self.interval_seconds = interval_seconds
        self.samples: list[dict[str, int]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @staticmethod
    def sample() -> dict[str, int] | None:
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
                 "--format=csv,noheader,nounits"],
                check=True, capture_output=True, text=True, timeout=3)
            first = result.stdout.strip().splitlines()[0].split(",")
            return {"utilization_gpu_percent": int(first[0].strip()),
                    "memory_used_mib": int(first[1].strip()),
                    "memory_total_mib": int(first[2].strip())}
        except (OSError, subprocess.SubprocessError, ValueError, IndexError):
            return None

    def _run(self) -> None:
        while not self._stop.is_set():
            sample = self.sample()
            if sample is not None:
                self.samples.append(sample)
            self._stop.wait(self.interval_seconds)

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True,
                                       name="sign-advisor-gpu-sampler")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def summary(self) -> dict[str, Any]:
        if not self.samples:
            return {"samples": 0, "mean_gpu_utilization_percent": None,
                    "max_gpu_utilization_percent": None,
                    "peak_host_memory_used_mib": None,
                    "gpu_memory_total_mib": None}
        utils = [item["utilization_gpu_percent"] for item in self.samples]
        memories = [item["memory_used_mib"] for item in self.samples]
        return {
            "samples": len(self.samples),
            "mean_gpu_utilization_percent": statistics.mean(utils),
            "max_gpu_utilization_percent": max(utils),
            "peak_host_memory_used_mib": max(memories),
            "gpu_memory_total_mib": max(item["memory_total_mib"]
                                         for item in self.samples),
        }


def _read_reference(path: Path, bank: Any, positions: np.ndarray,
                    prompt_sha: str) -> tuple[dict[str, Any], list[dict[str, Any]], str]:
    if not path.is_file():
        raise FileNotFoundError(f"sequential reference JSONL does not exist: {path}")
    records: dict[str, Any] = {}
    identity = None
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row.get("record_type") == "run_identity":
                if identity is not None:
                    raise ValueError("reference contains more than one run identity")
                identity = row
            elif row.get("record_type") == "frame":
                sample_id = row["sample_id"]
                if sample_id in records:
                    raise ValueError(f"duplicate sequential sample_id: {sample_id}")
                records[sample_id] = row
    if identity is None:
        raise ValueError("sequential reference has no run identity row")
    expected = {
        "bank_content_digest": evaluation.EXPECTED_BANK_DIGEST,
        "test_rows": int(len(positions)),
        "model_id": evaluation.EXPECTED_MODEL_ID,
        "resolved_model_commit": evaluation.EXPECTED_MODEL_REVISION,
        "weight_fingerprint": evaluation.EXPECTED_WEIGHT_FINGERPRINT,
        "prompt_sha256": prompt_sha,
        "crop_resolution": ADVISOR_CROP_RESOLUTION,
        "dtype": "float16",
    }
    mismatched = [key for key, value in expected.items() if identity.get(key) != value]
    if mismatched:
        raise ValueError(f"sequential reference provenance mismatch: {mismatched}")
    if not 64 <= len(records) < len(positions):
        raise ValueError("reference must be a partial run with at least 64 completed rows")
    by_order = {int(row["test_order"]): row for row in records.values()}
    if len(by_order) != len(records):
        raise ValueError("reference test_order values are not unique")
    ordered = [by_order[key] for key in sorted(by_order)]
    selection_indexes = np.linspace(0, len(ordered) - 1,
                                    num=EQUIVALENCE_FRAMES, dtype=np.int64)
    selected = [ordered[int(index)] for index in selection_indexes]
    bank_by_order = [bank.samples[int(position)] for position in positions]
    position_to_order = {int(position): order for order, position in enumerate(positions)}
    sample_to_position = {sample.sample_id: index
                          for index, sample in enumerate(bank.samples)}
    for row in selected:
        order = int(row["test_order"])
        if order < 0 or order >= len(bank_by_order):
            raise ValueError("reference contains a test_order outside the frozen population")
        expected_sample = bank_by_order[order]
        if row["sample_id"] != expected_sample.sample_id:
            raise ValueError("reference row is mapped to a different receiver sample")
        if int(row["bank_position"]) != int(positions[order]):
            raise ValueError("reference row bank position differs from the frozen test order")
        donor_id = row["shuffled_donor_sample_id"]
        donor_position = sample_to_position.get(donor_id)
        if donor_position is None:
            raise ValueError("reference donor sample_id is absent from the FrameBank")
        donor_order = position_to_order.get(donor_position)
        if donor_order is None:
            raise ValueError("reference shuffled donor is outside the test population")
        if bank_by_order[donor_order].sequence_id == expected_sample.sequence_id:
            raise ValueError("reference shuffled donor is from the receiver sequence")
    digest = _sha256(path)
    return records, selected, digest


def _summarize_timings(totals: dict[str, float], batch_latencies: list[float],
                       frames: int, elapsed: float, gpu: dict[str, Any],
                       torch: Any) -> dict[str, Any]:
    try:
        allocated_mib = torch.cuda.max_memory_allocated() / (1024 ** 2)
        reserved_mib = torch.cuda.max_memory_reserved() / (1024 ** 2)
        device_total_mib = torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory / (1024 ** 2)
    except Exception:
        allocated_mib = reserved_mib = device_total_mib = None
    host_peak = gpu.get("peak_host_memory_used_mib")
    peak_candidates = [value for value in (host_peak, reserved_mib) if value is not None]
    peak_mib = max(peak_candidates) if peak_candidates else None
    totals_mib = [value for value in (gpu.get("gpu_memory_total_mib"),
                                      device_total_mib) if value is not None]
    total_mib = min(totals_mib) if totals_mib else None
    headroom = total_mib - peak_mib if total_mib is not None and peak_mib is not None else None
    return {
        "frames": frames,
        "vlm_requests": 2 * frames,
        "elapsed_seconds": elapsed,
        "frames_per_second": frames / elapsed if elapsed else None,
        "requests_per_second": 2 * frames / elapsed if elapsed else None,
        "estimated_full_test_seconds": 7076 / (frames / elapsed)
        if frames and elapsed else None,
        "estimated_full_test_hours": 7076 / (frames / elapsed) / 3600
        if frames and elapsed else None,
        "batch_latency_seconds": {
            "median": statistics.median(batch_latencies) if batch_latencies else None,
            "p95": float(np.percentile(batch_latencies, 95)) if batch_latencies else None,
            "per_batch": batch_latencies,
        },
        "stage_seconds": totals,
        "gpu": gpu,
        "memory": {
            "torch_peak_allocated_mib": allocated_mib,
            "torch_peak_reserved_mib": reserved_mib,
            "peak_vram_mib": peak_mib,
            "total_vram_mib": total_mib,
            "minimum_headroom_mib": headroom,
        },
        "safe_memory_headroom": headroom is not None and headroom >= SAFE_HEADROOM_MIB,
    }


def run(args) -> dict[str, Any]:
    import torch
    import transformers
    from PIL import Image
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

    if transformers.__version__ != "4.49.0":
        raise ValueError("throughput benchmark requires frozen transformers 4.49.0")
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite benchmark evidence: {args.out}")
    args.out.mkdir(parents=True)
    roots = dict(value.split("=", 1) for value in args.image_root)
    bank = load_bank(args.bank)
    positions = bank.indices("test")
    if len(positions) != 7076 or bank.content_digest() != evaluation.EXPECTED_BANK_DIGEST:
        raise ValueError("FrameBank differs from the frozen 7,076-row Sign Advisor population")
    prompt = prompt_text()
    prompt_sha = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    snapshot = resolve_snapshot(args.model, args.hf_cache)
    evaluation.validate_frozen_backend(
        args.model, args.revision, snapshot, prompt_sha,
        max_new_tokens=evaluation.EXPECTED_MAX_NEW_TOKENS,
        seed=evaluation.EXPECTED_SEED,
        crop_resolution=ADVISOR_CROP_RESOLUTION)
    advisor_sha = _sha256(Path(sign_advisor_module.__file__).resolve())
    feature_sha = _sha256(Path(feature_module.__file__).resolve())
    dockerfile = Path(sign_advisor_module.__file__).resolve().parents[2] / "Dockerfile.signadvisor"
    dockerfile_sha = _sha256(dockerfile)
    if (advisor_sha != evaluation.EXPECTED_ADVISOR_SOURCE_SHA256
            or feature_sha != evaluation.EXPECTED_FEATURE_SOURCE_SHA256
            or dockerfile_sha != evaluation.EXPECTED_DOCKERFILE_SHA256):
        raise ValueError("Sign Advisor, crop, or runtime source differs from the frozen contract")
    reference_rows, selected_reference, reference_sha = _read_reference(
        args.reference_jsonl, bank, positions, prompt_sha)
    sequence_ids = [bank.samples[int(position)].sequence_id for position in positions]
    donor_orders = evaluation.different_sequence_donors(sequence_ids)
    entries = []
    for row in selected_reference:
        order = int(row["test_order"])
        position = int(positions[order])
        donor_position = int(positions[int(donor_orders[order])])
        sample = bank.samples[position]
        donor = bank.samples[donor_position]
        if row["sample_id"] != sample.sample_id:
            raise ValueError("selected reference row does not match deterministic receiver order")
        if row["shuffled_donor_sample_id"] != donor.sample_id:
            raise ValueError("selected reference donor differs from frozen deterministic mapping")
        entries.append((order, position, donor_position))
    if len(entries) != EQUIVALENCE_FRAMES:
        raise AssertionError("deterministic equivalence subset is not exactly 64 frames")

    torch.manual_seed(evaluation.EXPECTED_SEED)
    processor = AutoProcessor.from_pretrained(args.model, revision=args.revision)
    batch_preparation_processor = AutoProcessor.from_pretrained(
        args.model, revision=args.revision)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model, revision=args.revision, torch_dtype=torch.float16,
        device_map=args.device)
    model.eval()
    parameter_count = int(sum(parameter.numel() for parameter in model.parameters()))
    if parameter_count != 2_208_985_600:
        raise ValueError("loaded parameter count differs from the recovered Qwen snapshot")
    messages = [{"role": "user", "content": [
        {"type": "image"}, {"type": "text", "text": prompt}]}]
    rendered_prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)

    # Exclude one serial warm-up request so startup kernels do not distort the
    # short baseline profile. It uses the same crop, processor, and decoder.
    warmup_started = time.perf_counter()
    warmup_crop = read_crop(bank, entries[0][1], roots, ADVISOR_CROP_RESOLUTION)
    warmup_inputs = processor(
        text=[rendered_prompt], images=[Image.fromarray(warmup_crop)],
        return_tensors="pt").to(model.device)
    with torch.inference_mode():
        warmup_output = model.generate(
            **warmup_inputs, max_new_tokens=evaluation.EXPECTED_MAX_NEW_TOKENS,
            do_sample=False)
    if torch.cuda.is_available() and str(model.device).startswith("cuda"):
        torch.cuda.synchronize(model.device)
    warmup_raw = processor.batch_decode(
        warmup_output[:, warmup_inputs["input_ids"].shape[1]:],
        skip_special_tokens=True)[0]
    parse_response(warmup_raw, fields=tuple(SIGN_FIELD_NAMES))
    warmup_seconds = time.perf_counter() - warmup_started
    del warmup_output, warmup_inputs, warmup_crop

    # Short profile of the exact existing serial path: eight frames, both images.
    serial_totals = {name: 0.0 for name in (
        "crop_read_seconds", "pil_conversion_seconds", "processor_seconds",
        "host_to_device_seconds", "generate_seconds", "decode_parse_seconds")}
    serial_rows: list[dict[str, Any]] = []
    serial_gpu_sampler = GpuSampler()
    serial_gpu_sampler.start()

    def synchronize() -> None:
        if torch.cuda.is_available() and str(model.device).startswith("cuda"):
            torch.cuda.synchronize(model.device)

    serial_started = time.perf_counter()
    for entry, reference in zip(entries[:PROFILE_FRAMES], selected_reference[:PROFILE_FRAMES]):
        order, position, donor_position = entry
        sample = bank.samples[position]
        donor = bank.samples[donor_position]
        outputs = {}
        for mode, crop_position in (("real", position), ("shuffled", donor_position)):
            started = time.perf_counter()
            crop = read_crop(bank, crop_position, roots, ADVISOR_CROP_RESOLUTION)
            serial_totals["crop_read_seconds"] += time.perf_counter() - started
            expected_digest = reference[f"image_digest_{mode}_crop_rgb"]
            if crop_sha256(crop) != expected_digest:
                raise ValueError(f"{mode} crop bytes differ from sequential reference for {sample.sample_id}")
            started = time.perf_counter()
            image = Image.fromarray(crop)
            serial_totals["pil_conversion_seconds"] += time.perf_counter() - started
            started = time.perf_counter()
            inputs = processor(text=[rendered_prompt], images=[image], return_tensors="pt")
            serial_totals["processor_seconds"] += time.perf_counter() - started
            started = time.perf_counter()
            synchronize()
            inputs = inputs.to(model.device)
            synchronize()
            serial_totals["host_to_device_seconds"] += time.perf_counter() - started
            started = time.perf_counter()
            with torch.inference_mode():
                generated = model.generate(
                    **inputs, max_new_tokens=evaluation.EXPECTED_MAX_NEW_TOKENS,
                    do_sample=False)
            synchronize()
            serial_totals["generate_seconds"] += time.perf_counter() - started
            started = time.perf_counter()
            raw = processor.batch_decode(
                generated[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
            parsed = parse_response(raw, fields=tuple(SIGN_FIELD_NAMES))
            serial_totals["decode_parse_seconds"] += time.perf_counter() - started
            outputs[mode] = {"valid": bool(parsed.valid), "reason": parsed.reason,
                             "state": [int(value) for value in parsed.state], "raw": raw}
        serial_rows.append({"sample_id": sample.sample_id, **outputs})
    serial_elapsed = time.perf_counter() - serial_started
    serial_gpu_sampler.stop()
    serial_equivalence = compare_output_rows(
        {row["sample_id"]: row for row in reference_rows.values()}, serial_rows)

    profile = {
        "warmup": {"requests": 1, "elapsed_seconds": warmup_seconds,
                   "excluded_from_profile": True},
        "frames": PROFILE_FRAMES,
        "vlm_requests": 2 * PROFILE_FRAMES,
        "elapsed_seconds": serial_elapsed,
        "frames_per_second": PROFILE_FRAMES / serial_elapsed if serial_elapsed else None,
        "requests_per_second": 2 * PROFILE_FRAMES / serial_elapsed if serial_elapsed else None,
        "estimated_full_test_seconds": 7076 / (PROFILE_FRAMES / serial_elapsed)
        if serial_elapsed else None,
        "estimated_full_test_hours": 7076 / (PROFILE_FRAMES / serial_elapsed) / 3600
        if serial_elapsed else None,
        "stage_seconds": serial_totals,
        "gpu": serial_gpu_sampler.summary(),
        "sequential_output_equivalence": serial_equivalence,
    }
    candidates: list[dict[str, Any]] = []
    selected_size = None
    selected_equivalence = None
    selected_safe = False
    selected_gain = False
    previous_rate = profile["frames_per_second"]
    serial_matches = profile["sequential_output_equivalence"].get(
        "evaluation_relevant_exact_match_rate") == 1.0
    stop_reason = None if serial_matches else (
        "short serial replay did not reproduce evaluation-relevant sequential outputs")

    for frame_batch_size in BENCHMARK_FRAME_COUNTS if serial_matches else ():
        torch.cuda.reset_peak_memory_stats()
        gpu_sampler = GpuSampler()
        gpu_sampler.start()
        candidate_started = time.perf_counter()
        totals = {name: 0.0 for name in (
            "crop_read_seconds", "pil_conversion_seconds", "processor_seconds",
            "host_to_device_seconds", "generate_seconds", "decode_parse_seconds")}
        latencies: list[float] = []
        candidate_rows: list[dict[str, Any]] = []
        preparer = OrderedBatchPreparer(
        lambda entry: (
            read_crop(bank, entry[1], roots, ADVISOR_CROP_RESOLUTION),
            read_crop(bank, entry[2], roots, ADVISOR_CROP_RESOLUTION)),
            batch_preparation_processor, rendered_prompt, crop_workers=4,
            crop_resolution=ADVISOR_CROP_RESOLUTION)
        status = "COMPLETE"
        error = None
        try:
            chunks = (entries[start:start + frame_batch_size]
                      for start in range(0, len(entries), frame_batch_size))
            for prepared in prefetch_ordered_batches(chunks, preparer):
                batch_started = time.perf_counter()
                parsed, raw, timings = generate_prepared_batch(
                    prepared, processor=processor, model=model, torch=torch,
                    parse_response=parse_response, fields=SIGN_FIELD_NAMES,
                    max_new_tokens=evaluation.EXPECTED_MAX_NEW_TOKENS)
                real_parsed, shuffled_parsed = evaluation.split_interleaved(parsed)
                real_raw, shuffled_raw = evaluation.split_interleaved(raw)
                # Retain the exact crop hashes and response strings for audit.
                for index, entry in enumerate(prepared.entries):
                    order, position, donor_position = entry
                    sample = bank.samples[position]
                    donor = bank.samples[donor_position]
                    row = {
                        "sample_id": sample.sample_id,
                        "test_order": order,
                        "shuffled_donor_sample_id": donor.sample_id,
                        "image_digest_real_crop_rgb": prepared.crop_hash_pairs[index][0],
                        "image_digest_shuffled_crop_rgb": prepared.crop_hash_pairs[index][1],
                        "real": {"valid": bool(real_parsed[index].valid),
                                 "reason": real_parsed[index].reason,
                                 "state": [int(value) for value in real_parsed[index].state],
                                 "raw": real_raw[index]},
                        "shuffled": {"valid": bool(shuffled_parsed[index].valid),
                                     "reason": shuffled_parsed[index].reason,
                                     "state": [int(value) for value in shuffled_parsed[index].state],
                                     "raw": shuffled_raw[index]},
                    }
                    candidate_rows.append(row)
                    for mode in ("real", "shuffled"):
                        reference_row = reference_rows[sample.sample_id]
                        if row[f"image_digest_{mode}_crop_rgb"] != reference_row[
                                f"image_digest_{mode}_crop_rgb"]:
                            raise ValueError("batched path changed crop bytes")
                for name, value in timings.items():
                    totals[name] += value
                latencies.append(time.perf_counter() - batch_started)
        except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
            if isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in str(exc).lower():
                status = "OOM"
                error = str(exc)
                torch.cuda.empty_cache()
            else:
                raise
        finally:
            preparer.close()
            gpu_sampler.stop()
        elapsed = time.perf_counter() - candidate_started
        gpu_summary = gpu_sampler.summary()
        performance = _summarize_timings(
            totals, latencies, len(candidate_rows), elapsed, gpu_summary, torch)
        equivalence = compare_output_rows(reference_rows, candidate_rows) if candidate_rows else {
            "rows": 0, "requests": 0, "raw_exact_match_rate": None,
            "parse_status_exact_match_rate": None, "state_exact_match_rate": None,
            "evaluation_relevant_exact_match_rate": None,
            "raw_mismatches": [], "evaluation_relevant_mismatches": [],
        }
        candidate_path = args.out / f"batch_{frame_batch_size}_equivalence.jsonl"
        if candidate_rows:
            _write_jsonl(candidate_path, candidate_rows)
            candidate_sha = _sha256(candidate_path)
        else:
            candidate_sha = None
        candidate = {
            "frame_batch_size": frame_batch_size,
            "request_batch_size": 2 * frame_batch_size,
            "status": status,
            "error": error,
            "performance": performance,
            "equivalence": equivalence,
            "candidate_rows_path": candidate_path.name if candidate_rows else None,
            "candidate_rows_sha256": candidate_sha,
        }
        candidates.append(candidate)
        if status == "OOM":
            stop_reason = f"OOM at frame batch size {frame_batch_size}"
            break
        safe = bool(performance["safe_memory_headroom"])
        if not safe:
            stop_reason = (f"unsafe VRAM headroom below {SAFE_HEADROOM_MIB} MiB "
                           f"at frame batch size {frame_batch_size}")
            break
        rate = performance["frames_per_second"]
        gain = rate / previous_rate - 1 if previous_rate else None
        candidate["gain_vs_previous_percent"] = gain * 100 if gain is not None else None
        if gain is None or gain < MATERIAL_GAIN:
            if selected_size is None:
                stop_reason = "first batched size did not beat the serial profile materially"
            else:
                stop_reason = "throughput gain over previous size was below 10%"
            break
        selected_size = frame_batch_size
        selected_equivalence = equivalence
        selected_safe = safe
        selected_gain = rate / profile["frames_per_second"] - 1 >= MATERIAL_GAIN
        previous_rate = rate

    report = {
        "schema": "animcv_sign_advisor_throughput_equivalence_v1",
        "started_utc": args.started_utc,
        "sequential_reference": {
            "path": str(args.reference_jsonl),
            "sha256": reference_sha,
            "completed_frame_rows": len(reference_rows),
            "selected_subset_rule": "64 evenly spaced completed rows by ascending test_order",
            "selected_frame_count": len(entries),
            "selected_sample_ids": [row["sample_id"] for row in selected_reference],
        },
        "frozen_contract": {
            "model_id": args.model,
            "resolved_revision": snapshot["resolved_commit"],
            "weight_fingerprint": snapshot["weight_fingerprint"],
            "dtype": "float16",
            "prompt_schema": PROMPT_SCHEMA_VERSION,
            "prompt_sha256": prompt_sha,
            "prompt_provenance": prompt_provenance(),
            "max_new_tokens": evaluation.EXPECTED_MAX_NEW_TOKENS,
            "do_sample": False,
            "seed": evaluation.EXPECTED_SEED,
            "crop_resolution": ADVISOR_CROP_RESOLUTION,
            "crop_source_sha256": feature_sha,
            "sign_advisor_source_sha256": advisor_sha,
            "runtime_dockerfile_sha256": dockerfile_sha,
            "parameter_count": parameter_count,
            "processor_class": f"{processor.__class__.__module__}.{processor.__class__.__name__}",
            "batch_preparation_processor_class": (
                f"{batch_preparation_processor.__class__.__module__}."
                f"{batch_preparation_processor.__class__.__name__}"),
            "transformers_version": transformers.__version__,
            "torch_version": torch.__version__,
        },
        "benchmark_limits": {
            "frame_batch_sizes": list(BENCHMARK_FRAME_COUNTS),
            "maximum_vlm_request_batch_size": 2 * max(BENCHMARK_FRAME_COUNTS),
            "safe_vram_headroom_mib": SAFE_HEADROOM_MIB,
            "material_throughput_gain_fraction": MATERIAL_GAIN,
            "prefetch_crop_workers": 4,
            "prefetch_max_pending_batches": 1,
        },
        "sequential_profile": profile,
        "candidate_benchmarks": candidates,
        "selected_frame_batch_size": selected_size,
        "selected_request_batch_size": 2 * selected_size if selected_size else None,
        "selected_performance": next((item["performance"] for item in candidates
                                       if item["frame_batch_size"] == selected_size), None),
        "selected_throughput_multiplier_vs_sequential": (
            next((item["performance"]["frames_per_second"] / profile["frames_per_second"]
                  for item in candidates if item["frame_batch_size"] == selected_size), None)
            if selected_size and profile["frames_per_second"] else None),
        "selected_equivalence": selected_equivalence,
        "selected_batch_safe": selected_safe,
        "selected_material_gain": selected_gain,
        "stop_reason": stop_reason,
        "evaluator_script_sha256": _sha256(Path(evaluation.__file__).resolve()),
        "batching_helper_sha256": _sha256(Path(__file__).with_name("sign_advisor_batching.py")),
        "benchmark_script_sha256": _sha256(Path(__file__).resolve()),
    }
    report_path = args.out / "throughput_equivalence_report.json"
    write_json(report_path, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", required=True, type=Path)
    parser.add_argument("--image-root", action="append", required=True, help="KEY=PATH")
    parser.add_argument("--reference-jsonl", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--model", default=evaluation.EXPECTED_MODEL_ID)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--hf-cache", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--started-utc", required=True)
    args = parser.parse_args()
    result = run(args)
    print(json.dumps({
        "selected_frame_batch_size": result["selected_frame_batch_size"],
        "selected_request_batch_size": result["selected_request_batch_size"],
        "selected_equivalence": result["selected_equivalence"],
        "stop_reason": result["stop_reason"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
