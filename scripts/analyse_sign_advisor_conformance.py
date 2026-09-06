#!/usr/bin/env python3
"""Separate schema conformance from answer content in a stored diagnostic.

The strict contract requires exactly one bare JSON object. Qwen wraps its reply
in a ```json fence, so strict parsing rejects it and every strict metric then
measures *fencing* rather than perception. This re-reads the stored raw text —
no new inference, no prompt change — and reports the two things apart:

    conformance   how often the reply satisfies the strict contract
    content       what the model actually answered, recovered leniently

Only content is comparable across prompt modes. Conformance is reported as its
own finding.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

from common.serialization import read_json, write_json
from framepose.sign_advisor import _ANSWERS
from framepose.signs import NEGATIVE, POSITIVE, SIGN_FIELD_NAMES, UNKNOWN


_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", flags=re.IGNORECASE)


def lenient_state(raw: str, fields: tuple[str, ...]) -> tuple[np.ndarray, bool, str]:
    """Recover the answer a lenient reader would have taken from this reply."""
    state = np.zeros(len(SIGN_FIELD_NAMES), dtype=np.int8)
    text = _FENCE.sub("", raw or "").strip()
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match is None:
        return state, False, "no JSON object recoverable"
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError as error:
        return state, False, f"truncated or invalid JSON: {error.msg}"
    if not isinstance(payload, dict):
        return state, False, "recovered value is not an object"
    recovered = 0
    for name in fields:
        answer = payload.get(name)
        if isinstance(answer, str) and answer.strip().lower() in _ANSWERS[name]:
            state[SIGN_FIELD_NAMES.index(name)] = _ANSWERS[name][answer.strip().lower()]
            recovered += 1
    return state, recovered == len(fields), f"recovered {recovered}/{len(fields)} fields"


def _metrics(predicted: np.ndarray, reference: np.ndarray) -> dict:
    scored = reference != UNKNOWN
    truth, guess = reference[scored], predicted[scored]
    per_class = [float((guess[truth == value] == value).mean())
                 for value in (NEGATIVE, POSITIVE) if (truth == value).any()]
    return {
        "scored": int(truth.size),
        "accuracy": float((guess == truth).mean()) if truth.size else None,
        "majority_class_baseline": float(max((truth == POSITIVE).mean(), (truth == NEGATIVE).mean()))
        if truth.size else None,
        "balanced_accuracy": float(np.mean(per_class)) if per_class else None,
        "predicted_positive": int((predicted == POSITIVE).sum()),
        "predicted_negative": int((predicted == NEGATIVE).sum()),
        "predicted_unknown": int((predicted == UNKNOWN).sum()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Split conformance from content")
    parser.add_argument("--diagnostic", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    records = read_json(args.diagnostic / "diagnostic_records.json")["records"]
    strict = read_json(args.diagnostic / "diagnostic.json")
    fields = tuple(SIGN_FIELD_NAMES)
    modes = ("combined_real", "combined_shuffled", "isolated_real", "isolated_shuffled")

    content = {mode: np.zeros((len(records), len(fields)), dtype=np.int8) for mode in modes}
    conformance = {mode: {"strict_valid": 0, "lenient_recovered": 0, "unrecoverable": 0}
                   for mode in modes}
    reference = np.zeros((len(records), len(fields)), dtype=np.int8)

    for order, record in enumerate(records):
        reference[order] = [record["oracle"][name] for name in fields]
        for mode in ("combined_real", "combined_shuffled"):
            answer = record["answers"][mode]
            conformance[mode]["strict_valid"] += int(answer["valid"])
            state, complete, _ = lenient_state(answer["raw"], fields)
            content[mode][order] = state
            conformance[mode]["lenient_recovered"] += int(complete)
            conformance[mode]["unrecoverable"] += int(not state.any() and not complete)
        for mode in ("isolated_real", "isolated_shuffled"):
            state = np.zeros(len(fields), dtype=np.int8)
            for name in fields:
                detail = record["answers"][mode]["per_field"][name]
                conformance[mode]["strict_valid"] += int(detail["valid"])
                recovered, complete, _ = lenient_state(detail["raw"], (name,))
                state[SIGN_FIELD_NAMES.index(name)] = recovered[SIGN_FIELD_NAMES.index(name)]
                conformance[mode]["lenient_recovered"] += int(complete)
            content[mode][order] = state

    report = {
        "schema": "animcv_sign_advisor_conformance_v1",
        "source_diagnostic": str(args.diagnostic),
        "frame_count": len(records),
        "note": ("strict conformance and answer content are different measurements; only content "
                 "is comparable across prompt modes"),
        "strict_conformance": {
            mode: {**conformance[mode],
                   "requests": len(records) * (1 if mode.startswith("combined") else len(fields)),
                   "strict_valid_rate": conformance[mode]["strict_valid"] /
                                        (len(records) * (1 if mode.startswith("combined") else len(fields)))}
            for mode in modes},
        "content_metrics": {
            mode: {name: _metrics(content[mode][:, index], reference[:, index])
                   for index, name in enumerate(fields)} for mode in modes},
        "content_grounding": {
            label: {name: {
                "real_vs_shuffled_agreement": float(
                    (content[real][:, index] == content[shuffled][:, index]).mean()),
                "prediction_change_rate_under_shuffle": float(
                    (content[real][:, index] != content[shuffled][:, index]).mean()),
            } for index, name in enumerate(fields)}
            for label, real, shuffled in (("combined", "combined_real", "combined_shuffled"),
                                          ("isolated", "isolated_real", "isolated_shuffled"))},
        "strict_metrics_reference": strict["metrics"],
    }
    args.out.mkdir(parents=True, exist_ok=True)
    np.savez(args.out / "content_predictions.npz", reference=reference,
             **{mode: content[mode] for mode in modes})
    write_json(args.out / "conformance.json", report)
    print(json.dumps(report["strict_conformance"], indent=2, sort_keys=True))
    for mode in modes:
        print(f"\n== {mode} (content)")
        for name in fields:
            m = report["content_metrics"][mode][name]
            print("   %-28s acc %.3f  major %.3f  bal %-6s +1 %3d  -1 %3d  ? %3d" % (
                name, m["accuracy"], m["majority_class_baseline"],
                ("%.3f" % m["balanced_accuracy"]) if m["balanced_accuracy"] is not None else "-",
                m["predicted_positive"], m["predicted_negative"], m["predicted_unknown"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
