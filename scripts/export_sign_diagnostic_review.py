#!/usr/bin/env python3
"""Export RGB crops and the four advisor answers for human inspection.

These exports exist so a person can check whether the oracle label corresponds
to what is actually visible in the frame. The numeric GT mapping alone does not
establish that the visual semantics are right, and this script deliberately
makes no such claim.

Prioritises frames where the modes disagree — isolated vs combined, real vs
shuffled, advisor vs oracle — because those are where the mechanism is legible.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from common.serialization import read_json, write_json
from framepose.bank import load_bank
from framepose.features import read_crop
from framepose.sign_advisor import ADVISOR_CROP_RESOLUTION
from framepose.signs import SIGN_FIELDS, SIGN_FIELD_NAMES, UNKNOWN


def main() -> int:
    parser = argparse.ArgumentParser(description="Export sign-advisor review crops")
    parser.add_argument("--bank", required=True, type=Path)
    parser.add_argument("--image-root", action="append", required=True)
    parser.add_argument("--diagnostic", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--per-field", type=int, default=4)
    args = parser.parse_args()

    from PIL import Image

    roots = dict(value.split("=", 1) for value in args.image_root)
    bank = load_bank(args.bank)
    records = read_json(args.diagnostic / "diagnostic_records.json")["records"]
    definitions = {field.name: field for field in SIGN_FIELDS}

    crops_dir = args.out / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)
    exported: dict[str, list] = {}

    for name in SIGN_FIELD_NAMES:
        scored = []
        for record in records:
            oracle = record["oracle"][name]
            if oracle == UNKNOWN:
                continue
            combined = record["answers"]["combined_real"]["state"][name]
            isolated = record["answers"]["isolated_real"]["state"][name]
            shuffled = record["answers"]["isolated_shuffled"]["state"][name]
            # Rank by how much this frame discriminates the hypotheses.
            interest = ((isolated != combined) * 4 + (isolated != shuffled) * 2
                        + (isolated != oracle))
            scored.append((interest, record, combined, isolated, shuffled, oracle))
        scored.sort(key=lambda item: -item[0])
        picked = []
        for interest, record, combined, isolated, shuffled, oracle in scored[:args.per_field]:
            position = record["bank_position"]
            sample = bank.samples[position]
            crop = read_crop(bank, position, roots, ADVISOR_CROP_RESOLUTION)
            filename = f"{name}__{sample.sample_id.replace(':', '_').replace('#', '_')}.png"
            Image.fromarray(crop).save(crops_dir / filename)
            picked.append({
                "sample_id": sample.sample_id,
                "crop_png": str(Path("crops") / filename),
                "image_reference": record["image_reference"],
                "image_content_digest": record["image_content_digest"],
                "shuffled_donor_sample_id": record["shuffled_donor_sample_id"],
                "field": name,
                "plain_language_definition": definitions[name].definition,
                "positive_means": definitions[name].positive_means,
                "negative_means": definitions[name].negative_means,
                "oracle_sign": oracle,
                "combined_seven_question_answer": combined,
                "isolated_question_answer": isolated,
                "shuffled_image_answer": shuffled,
                "isolated_differs_from_combined": isolated != combined,
                "isolated_differs_from_shuffled": isolated != shuffled,
                "isolated_disagrees_with_oracle": isolated != oracle,
                "raw_isolated_response": record["answers"]["isolated_real"]["per_field"][name]["raw"],
            })
        exported[name] = picked

    write_json(args.out / "review.json", {
        "schema": "animcv_sign_advisor_review_v1",
        "note": ("crops and answers for human inspection. Whether the oracle label matches what a "
                 "human sees in the frame is for the reader to judge; this export asserts only what "
                 "the model answered and what the geometry says."),
        "crop_resolution": ADVISOR_CROP_RESOLUTION,
        "fields": exported,
    })
    print(json.dumps({name: len(items) for name, items in exported.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
