#!/usr/bin/env python3
"""DIAGNOSTIC (Worklog 73): apply the fixed subject split to the BEDLAM banks.

TRAIN: every row's person is recovered from the official scene CSV's body
list for its sequence, disambiguated by the label's SMPL-X betas
(``SceneBodyIndex``); rows of validation subjects (sha256(subject) mod 5 == 0)
and rows whose person cannot be recovered are removed.  VALIDATION: rows were
already restricted to validation subjects at selection (official ``sub``
field); the recovered person must agree with ``sub``.  Writes ``<out-dir>/bedlam_train_final`` and
``bedlam_validation_final`` plus a split report.  No target value is read.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from framepose.bedlam_supervision import ExternalBank, ExternalSample, SceneBodyIndex, is_validation_subject, subject_key


def with_identity(bank: ExternalBank, labels: dict[str, Path], index: SceneBodyIndex, split: str):
    keep, samples, reasons, cache, distances = [], [], Counter(), {}, []
    for i, s in enumerate(bank.samples):
        name = s.detail["label_file"]
        if name not in cache:
            cache[name] = np.load(labels[name])["shape"]
        _, scene, seq = s.sequence_id.split(":")
        person, info = index.lookup(scene, seq, cache[name][s.detail["label_row"]])
        if person is None:
            reasons[f"person_unrecovered:{info['reason']}"] += 1
            continue
        distances.append(info["beta_distance"])
        if split == "validation" and person != s.detail.get("person"):
            raise SystemExit(f"betas-derived person {person} disagrees with official sub {s.detail.get('person')}")
        if is_validation_subject(person) != (split == "validation"):
            reasons["other_split_subject"] += 1
            continue
        keep.append(i)
        samples.append(ExternalSample(s.sample_id, s.source, s.sequence_id, s.frame_index, s.split, s.image_size,
                                      s.image_reference, {**s.detail, "person": person, "subject": subject_key(person)}))
    arrays = {k: bank.arrays[k][keep] for k in ExternalBank.ARRAYS}
    reasons["beta_distance_max"] = float(max(distances)) if distances else None
    reasons["beta_distance_nonzero_rows"] = int(sum(d > 0 for d in distances))
    return ExternalBank(samples, arrays, {**bank.metadata, "subject_split": "sha256(subject) mod 5 == 0 -> validation"}), reasons


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("train_bank", "validation_bank", "train_labels", "validation_labels", "gt_dir", "motion_dir", "out_dir"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    args = parser.parse_args()
    index = SceneBodyIndex(args.gt_dir, args.motion_dir)
    report = {}
    finals = {}
    for split, bank_path, label_dir in (("train", args.train_bank, args.train_labels),
                                        ("validation", args.validation_bank, args.validation_labels)):
        bank = ExternalBank.load(bank_path)
        labels = {p.name: p for p in label_dir.glob("*.npz")}
        final, reasons = with_identity(bank, labels, index, split)
        final.save(args.out_dir / f"bedlam_{split}_final")
        finals[split] = final
        report[split] = {"rows_before": len(bank), "rows_after": len(final), "dropped": dict(reasons),
                         "persons": len({s.detail["person"] for s in final.samples}),
                         "subjects": len({s.detail["subject"] for s in final.samples}),
                         "sequences": len({s.sequence_id for s in final.samples}),
                         "scenes": len({s.sequence_id.split(":")[1] for s in final.samples}),
                         "content_digest": final.content_digest()}
    tr = {s.detail["subject"] for s in finals["train"].samples}
    va = {s.detail["subject"] for s in finals["validation"].samples}
    if tr & va:
        raise SystemExit(f"subject overlap after split: {sorted(tr & va)[:5]}")
    report["subject_overlap"] = 0
    report["all_subjects_in_motion_release"] = len({subject_key(p.name) for p in args.motion_dir.iterdir() if p.is_dir()})
    (args.out_dir / "bedlam_subject_split_report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
