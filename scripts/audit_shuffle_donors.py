#!/usr/bin/env python3
"""Audit what the shuffled-image control actually controlled for.

No VLM inference. This reads the stored diagnostic records and the FrameBank and
answers two questions the original run asserted rather than measured:

1. **Donor identity.** The pairing rule `(i + n//2) % n` guarantees a different
   *sample*. It does not by itself guarantee a different sequence or a different
   performer, so the claim "another person's crop" needs checking.

2. **Conditioned grounding.** An unconditioned change rate under shuffling is
   weak evidence: if the donor happens to carry the same oracle branch, a
   grounded model *should* answer the same way. Stratifying by the donor's own
   oracle sign separates "did not look" from "looked and agreed".

The original diagnostic artifact is not modified.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from common.serialization import read_json, write_json
from framepose.bank import load_bank
from framepose.signs import NEGATIVE, POSITIVE, SIGN_FIELD_NAMES, UNKNOWN


def _identity(sequence_id: str) -> dict[str, str]:
    """`3dpw:<sequence>:actor<k>` -> its representable identity components."""
    parts = sequence_id.split(":")
    return {"sequence_id": sequence_id,
            "sequence": parts[1] if len(parts) > 1 else sequence_id,
            "actor": parts[2] if len(parts) > 2 else ""}


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit shuffled-image donor identity and grounding")
    parser.add_argument("--bank", required=True, type=Path)
    parser.add_argument("--diagnostic", required=True, type=Path)
    parser.add_argument("--conformance", required=True, type=Path,
                        help="content predictions from analyse_sign_advisor_conformance")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    bank = load_bank(args.bank)
    records = read_json(args.diagnostic / "diagnostic_records.json")["records"]
    with np.load(args.conformance / "content_predictions.npz") as handle:
        content = {key: handle[key] for key in handle.files}

    position_of = {sample.sample_id: index for index, sample in enumerate(bank.samples)}
    donor_rows = []
    for record in records:
        source = _identity(bank.samples[record["bank_position"]].sequence_id)
        donor_id = record["shuffled_donor_sample_id"]
        donor_position = position_of[donor_id]
        donor = _identity(bank.samples[donor_position].sequence_id)
        donor_rows.append({
            "sample_id": record["sample_id"], "donor_sample_id": donor_id,
            "donor_position": donor_position,
            "same_sample": record["sample_id"] == donor_id,
            "same_sequence_id": source["sequence_id"] == donor["sequence_id"],
            "same_sequence": source["sequence"] == donor["sequence"],
            "same_actor_index_within_sequence": (source["sequence"] == donor["sequence"]
                                                 and source["actor"] == donor["actor"]),
        })

    total = len(donor_rows)
    identity = {
        "pairing_rule": "(i + n//2) % n over the deterministic subset order",
        "donor_count": total,
        "same_sample_rate": sum(row["same_sample"] for row in donor_rows) / total,
        "same_sequence_id_rate": sum(row["same_sequence_id"] for row in donor_rows) / total,
        "same_sequence_rate": sum(row["same_sequence"] for row in donor_rows) / total,
        "different_sequence_rate": 1 - sum(row["same_sequence"] for row in donor_rows) / total,
        "same_actor_within_sequence_rate": sum(row["same_actor_index_within_sequence"]
                                               for row in donor_rows) / total,
        "representable_identity": ("3DPW sequence_id encodes sequence and actor index. Performer "
                                   "identity across different sequences is NOT representable from "
                                   "the bank, so 'a different person' cannot be asserted beyond "
                                   "'a different sequence'."),
    }

    oracle = np.stack([[record["oracle"][name] for name in SIGN_FIELD_NAMES]
                       for record in records]).astype(np.int8)
    # The donor's own oracle sign, looked up by the donor's subset row.
    donor_index = {record["sample_id"]: order for order, record in enumerate(records)}
    donor_oracle = np.stack([
        oracle[donor_index[row["donor_sample_id"]]] for row in donor_rows]).astype(np.int8)

    conditioned = {}
    for label, real_key, shuffled_key in (("combined", "combined_real", "combined_shuffled"),
                                          ("isolated", "isolated_real", "isolated_shuffled")):
        per_field = {}
        for index, name in enumerate(SIGN_FIELD_NAMES):
            real = content[real_key][:, index]
            shuffled = content[shuffled_key][:, index]
            own = oracle[:, index]
            donor = donor_oracle[:, index]
            groups = {
                "donor_same_branch": (donor != UNKNOWN) & (own != UNKNOWN) & (donor == own),
                "donor_opposite_branch": (donor != UNKNOWN) & (own != UNKNOWN) & (donor == -own),
                "donor_degenerate": donor == UNKNOWN,
            }
            per_field[name] = {
                key: {"frames": int(mask.sum()),
                      "prediction_change_rate": (float((real[mask] != shuffled[mask]).mean())
                                                 if mask.any() else None)}
                for key, mask in groups.items()}
            per_field[name]["unconditioned_change_rate"] = float((real != shuffled).mean())
        conditioned[label] = per_field

    report = {
        "schema": "animcv_shuffle_donor_audit_v1",
        "source_diagnostic": str(args.diagnostic),
        "original_artifact_modified": False,
        "donor_identity": identity,
        "conditioned_grounding": conditioned,
        "reading": ("a grounded model may legitimately answer the same under a donor carrying the "
                    "same branch; the discriminating number is the change rate when the donor "
                    "carries the OPPOSITE branch"),
        "donors": donor_rows,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    write_json(args.out / "shuffle_donor_audit.json", report)
    print(json.dumps(identity, indent=2, sort_keys=True))
    for label, per_field in conditioned.items():
        print(f"\n== {label}")
        print("   %-28s %8s %8s %8s %8s" % ("field", "uncond", "same", "opposite", "degen"))
        for name, value in per_field.items():
            def fmt(key):
                item = value[key]
                return "-" if item["prediction_change_rate"] is None else "%.3f" % item["prediction_change_rate"]
            print("   %-28s %8.3f %8s %8s %8s" % (
                name, value["unconditioned_change_rate"], fmt("donor_same_branch"),
                fmt("donor_opposite_branch"), fmt("donor_degenerate")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
