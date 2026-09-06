#!/usr/bin/env python3
"""Recover, or refuse to guess, the exact weights a historical advisor run used.

The executed S2 sign bank recorded only a repository id, which is not an
experiment identity: the same id resolves to different bytes over time. This
walks the local HuggingFace cache and, when the snapshot is unambiguous, writes
an **addendum** beside the historical artifact linking the proven provenance to
it. The historical artifact itself is never rewritten.

When the snapshot cannot be identified unambiguously the addendum says exactly
that — `exact_weight_fingerprint_not_established` — rather than naming a
plausible revision.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from common.serialization import read_json, write_json
from framepose.sign_advisor import resolve_snapshot


def main() -> int:
    parser = argparse.ArgumentParser(description="Recover historical advisor weight provenance")
    parser.add_argument("--sign-bank", required=True, type=Path)
    parser.add_argument("--hf-cache", required=True, type=Path)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    summary = read_json(args.sign_bank / "sign_bank_summary.json")
    model_id = summary["model"]["id"]
    snapshot = resolve_snapshot(model_id, args.hf_cache)

    addendum = {
        "schema": "animcv_sign_advisor_provenance_addendum_v1",
        "applies_to": str(args.sign_bank / "sign_bank_summary.json"),
        "historical_artifact_modified": False,
        "model_id": model_id,
        "recorded_revision_at_run_time": summary["model"].get("revision"),
        "recorded_weights_sha256_at_run_time": summary["model"].get("weights_sha256"),
    }
    if snapshot is None:
        addendum.update({
            "exact_weight_fingerprint_established": False,
            "status": "exact_weight_fingerprint_not_established",
            "reason": ("the local cache holds zero or several snapshots for this model id, so the "
                       "snapshot used by the historical run cannot be proven; a revision is not "
                       "guessed"),
        })
    else:
        addendum.update({
            "exact_weight_fingerprint_established": True,
            "status": "recovered_from_local_snapshot",
            "resolved_commit": snapshot["resolved_commit"],
            "refs": snapshot["refs"],
            "snapshot_dir": snapshot["snapshot_dir"],
            "file_count": snapshot["file_count"],
            "weight_fingerprint": snapshot["weight_fingerprint"],
            "files": snapshot["files"],
            "argument": ("the run loaded the model with revision=None, which resolves to refs/main; "
                         "the cache holds exactly one snapshot for this id and refs/main points at "
                         "it, so those are the bytes the run consumed"),
        })
        refs = snapshot["refs"]
        if refs.get("main") and refs["main"] != snapshot["resolved_commit"]:
            addendum.update({"exact_weight_fingerprint_established": False,
                             "status": "exact_weight_fingerprint_not_established",
                             "reason": "refs/main does not point at the only cached snapshot"})

    destination = args.out or (args.sign_bank / "provenance_addendum.json")
    write_json(destination, addendum)
    print(json.dumps({key: addendum[key] for key in
                      ("status", "exact_weight_fingerprint_established", "model_id",
                       "resolved_commit", "weight_fingerprint", "file_count")
                      if key in addendum}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
