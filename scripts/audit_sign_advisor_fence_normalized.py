#!/usr/bin/env python3
"""Audit stored Sign Advisor content after one predeclared fence-only transform.

This is diagnostic-only.  It never loads a VLM, regenerates crops, changes the
production parser, or writes to the frozen full-test run directory.  It reads
the immutable raw ledger and may remove exactly one outer Markdown code fence
before delegating all schema/content validation to ``parse_response``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common.serialization import write_json
import evaluate_current_sign_advisor as frozen_evaluation
from framepose.sign_advisor import AdvisorResponse, parse_response
from framepose.signs import SIGN_FIELD_NAMES


_ONE_OUTER_FENCE = re.compile(
    r"\A```(?P<language>json)?\r?\n(?P<payload>.*?)\r?\n```\Z", re.DOTALL)
_MODES = ("real", "shuffled")


@dataclass(frozen=True)
class DiagnosticResponse:
    """Strict parser result, plus whether the sole allowed transform was used."""

    response: AdvisorResponse
    outer_fence: str | None


def parse_fence_normalized_response(raw: str) -> DiagnosticResponse:
    """Optionally remove one complete outer `````json``/````` fence, and no more.

    A bare strict JSON object remains acceptable without normalization.  Any
    response that needs prose removal, inner-fence removal, JSON repair, or
    categorical repair is delegated to the unchanged strict parser and fails.
    """
    stripped = (raw or "").strip()
    match = _ONE_OUTER_FENCE.fullmatch(stripped)
    if match is None:
        return DiagnosticResponse(parse_response(raw, fields=SIGN_FIELD_NAMES), None)
    response = parse_response(match.group("payload"), fields=SIGN_FIELD_NAMES)
    return DiagnosticResponse(response, "json" if match.group("language") == "json" else "plain")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_ledger(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    if not rows or rows[0].get("record_type") != "run_identity":
        raise ValueError("ledger must begin with one run_identity record")
    records = [row for row in rows[1:] if row.get("record_type") == "frame"]
    if len(records) != len(rows) - 1:
        raise ValueError("ledger contains an unexpected non-frame record")
    records.sort(key=lambda row: row["test_order"])
    if [row["test_order"] for row in records] != list(range(len(records))):
        raise ValueError("ledger frame records do not have contiguous test_order values")
    return rows[0], records


def _diagnostic_records(records: list[dict[str, Any]]) -> tuple[
        list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, np.ndarray]]:
    accounting: dict[str, dict[str, Any]] = {
        mode: {
            "total_responses": len(records),
            "exact_outer_fence_responses": 0,
            "outer_fence_language": Counter(),
            "already_strict_valid": 0,
            "valid_after_fence_only_normalization": 0,
            "still_invalid_after_normalization": 0,
            "invalid_reason_taxonomy": Counter(),
        }
        for mode in _MODES
    }
    normalized_records: list[dict[str, Any]] = []
    states = {mode: np.zeros((len(records), len(SIGN_FIELD_NAMES)), dtype=np.int8)
              for mode in _MODES}
    for order, record in enumerate(records):
        normalized: dict[str, Any] = {}
        for mode in _MODES:
            raw = record[mode]["raw"]
            result = parse_fence_normalized_response(raw)
            response = result.response
            audit = accounting[mode]
            audit["exact_outer_fence_responses"] += int(result.outer_fence is not None)
            if result.outer_fence is not None:
                audit["outer_fence_language"][result.outer_fence] += 1
            else:
                audit["already_strict_valid"] += int(response.valid)
            audit["valid_after_fence_only_normalization"] += int(response.valid)
            audit["still_invalid_after_normalization"] += int(not response.valid)
            if not response.valid:
                audit["invalid_reason_taxonomy"][response.reason or "unspecified"] += 1
            states[mode][order] = response.state
            normalized[mode] = {
                "valid": bool(response.valid),
                "reason": response.reason,
                "state": [int(value) for value in response.state],
            }
        normalized_records.append(normalized)
    for mode in _MODES:
        audit = accounting[mode]
        audit["outer_fence_language"] = dict(sorted(audit["outer_fence_language"].items()))
        audit["invalid_reason_taxonomy"] = dict(sorted(audit["invalid_reason_taxonomy"].items()))
        audit["only_contract_violation_outer_fence"] = audit["valid_after_fence_only_normalization"]
        audit["only_contract_violation_outer_fence_rate"] = (
            audit["only_contract_violation_outer_fence"] / audit["total_responses"]
            if audit["total_responses"] else None)
    return normalized_records, accounting, states


def _raw_and_state_changes(records: list[dict[str, Any]],
                           states: dict[str, np.ndarray]) -> dict[str, Any]:
    raw_changed = np.asarray(
        [record["real"]["raw"] != record["shuffled"]["raw"] for record in records], dtype=bool)
    field_changed = states["real"] != states["shuffled"]
    return {
        "rows": len(records),
        "raw_string_change_count": int(raw_changed.sum()),
        "raw_string_change_rate": float(raw_changed.mean()) if len(raw_changed) else None,
        "normalized_signstate_any_field_change_count": int(field_changed.any(axis=1).sum()),
        "normalized_signstate_any_field_change_rate": (
            float(field_changed.any(axis=1).mean()) if len(field_changed) else None),
        "per_field_normalized_sign_change_rate": {
            field: float(field_changed[:, index].mean()) if len(field_changed) else None
            for index, field in enumerate(SIGN_FIELD_NAMES)},
    }


def _frozen_interface_accounting(records: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for mode in _MODES:
        valid = sum(bool(record[mode]["valid"]) for record in records)
        result[mode] = {
            "responses": len(records),
            "strict_valid": valid,
            "strict_invalid": len(records) - valid,
            "operational_signstate_coverage": sum(
                any(value != 0 for value in record[mode]["state"]) for record in records),
        }
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    run_identity, records = _load_ledger(args.ledger)
    ledger_sha256 = _sha256(args.ledger)
    if args.expected_ledger_sha256 and ledger_sha256 != args.expected_ledger_sha256:
        raise ValueError("immutable raw ledger SHA-256 differs from the expected value")
    frozen = _frozen_interface_accounting(records)
    for mode in _MODES:
        if frozen[mode]["strict_valid"] != 0:
            raise ValueError("this audit is scoped to the recorded all-strict-invalid full run")

    normalized_records, normalization, states = _diagnostic_records(records)
    reference = np.asarray([record["oracle_sign_state"] for record in records], dtype=np.int8)
    h0_signs = np.asarray([record["h0_sign_state"] for record in records], dtype=np.int8)
    replay_report = json.loads(args.replay_report.read_text(encoding="utf-8"))
    c_readable_wrong, c_h0_unknown = frozen_evaluation.exact_docs45_c_populations(
        replay_report, reference, h0_signs)
    metrics = frozen_evaluation.summarize_predictions(
        normalized_records, reference, c_readable_wrong, c_h0_unknown)

    frozen_metrics_sha256 = _sha256(args.frozen_metrics) if args.frozen_metrics else None
    frozen_manifest_sha256 = _sha256(args.frozen_manifest) if args.frozen_manifest else None
    if args.frozen_manifest:
        manifest = json.loads(args.frozen_manifest.read_text(encoding="utf-8"))
        if manifest.get("status") != "COMPLETE" or manifest.get("response_jsonl_sha256") != ledger_sha256:
            raise ValueError("frozen manifest does not identify this complete raw ledger")
    report = {
        "schema": "animcv_sign_advisor_fence_normalized_audit_v1",
        "diagnostic_name": "DIAGNOSTIC_FENCE_NORMALIZED",
        "scope": ("post-hoc content audit of immutable raw ledger only; no model, prompt, crop, "
                  "decoder, production parser, or production behavior changed"),
        "source": {
            "ledger_path": str(args.ledger),
            "ledger_sha256": ledger_sha256,
            "run_identity": run_identity,
            "frozen_metrics_sha256": frozen_metrics_sha256,
            "frozen_manifest_sha256": frozen_manifest_sha256,
            "replay_report_sha256": _sha256(args.replay_report),
        },
        "frozen_strict_interface_result_unchanged": frozen,
        "normalization_contract": {
            "allowed_transform": "remove exactly one complete outer ```json or ``` fence",
            "after_transform": "unchanged parse_response strict JSON/category validation",
            "forbidden": ["prose stripping", "multiple-fence stripping", "JSON repair",
                          "field repair", "left/right inversion", "semantic guessing"],
        },
        "normalization_accounting": normalization,
        "population_identity": {
            "all_oracle_readable_test_frames": len(records),
            "C_READABLE_WRONG": {field: len(rows) for field, rows in c_readable_wrong.items()},
            "C_H0_UNKNOWN": {field: len(rows) for field, rows in c_h0_unknown.items()},
        },
        "semantic_metrics": metrics,
        "raw_vs_normalized_signstate_change": _raw_and_state_changes(records, states),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    write_json(args.out / "fence_normalized_audit.json", report)
    np.savez_compressed(args.out / "fence_normalized_states.npz", reference=reference,
                        h0_signs=h0_signs, real=states["real"], shuffled=states["shuffled"])
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", required=True, type=Path)
    parser.add_argument("--replay-report", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--expected-ledger-sha256")
    parser.add_argument("--frozen-metrics", type=Path)
    parser.add_argument("--frozen-manifest", type=Path)
    args = parser.parse_args()
    report = run(args)
    summary = {
        mode: report["normalization_accounting"][mode]
        for mode in _MODES
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
