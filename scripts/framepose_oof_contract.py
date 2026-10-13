"""Worklog 79 sequence cross-fit contracts; diagnostic scripts only."""
from __future__ import annotations

import hashlib
from collections import Counter, defaultdict

import numpy as np

FOLDS = 5
SEED = 1337
CONTRACT = {
    "folds": FOLDS,
    "grouping": "3DPW capture (all actors); BEDLAM sequence (all people/frames)",
    "assignment": "per source, decreasing row count; SHA256(seed|group) tie order; least-loaded fold",
    "subject_limit": "BEDLAM subjects may repeat across distinct sequences/folds; not subject-generalization evidence",
    "selection": "inner sequence-grouped 5-way partition of outer TRAIN complement; inner fold 0 is selection only",
    "teacher_fit": "W73 graph/target/50-50 sampler/optimizer/200 epochs; select lowest inner 3DPW masked MAE",
    "refit": "fresh seed-1337 model on all outer TRAIN complement for selected epoch+1, same 200-epoch LR schedule",
    "inference": "exactly one refitted outer teacher per TRAIN row; original frozen E1 everywhere else",
    "reason": "5 folds preserve 80% outer training coverage; inner selection keeps original validation/test labels out of teachers",
    "coverage_limit": "OOF removes in-sample optimism AND reduces teacher coverage; refit/inner selection differs from W73 selection",
}


def group_key(sample):
    sid = sample.sequence_id
    if sample.source.upper() == "3DPW":
        return sid.rsplit(":actor", 1)[0]
    return sid


def assign_folds(samples, train_positions, folds=FOLDS):
    """Label-blind deterministic row-balanced assignment; whole capture groups."""
    groups = defaultdict(list)
    for p in np.asarray(train_positions, int):
        s = samples[p]
        if s.split != "train":
            raise ValueError("held-out row in cross-fit training population")
        groups[(s.source.upper(), group_key(s))].append(int(p))
    out = np.full(len(samples), -1, np.int64)
    for source in sorted({s for s, _ in groups}):
        keys = [k for k in groups if k[0] == source]
        if len(keys) < folds:
            raise ValueError("too few sequence groups for the declared folds")
        ordered = sorted(keys, key=lambda k: (-len(groups[k]), hashlib.sha256(f"{SEED}|{k[1]}".encode()).hexdigest()))
        loads = [0] * folds
        for k in ordered:
            f = min(range(folds), key=lambda j: (loads[j], j))
            out[groups[k]] = f
            loads[f] += len(groups[k])
    return out


def validate_fold(samples, all_train, predicted, teacher_train, selection):
    """Validate actual optimizer and selection populations, including group isolation."""
    train = set(map(int, all_train))
    pred, fit, sel = (set(map(int, v)) for v in (predicted, teacher_train, selection))
    if not pred or not fit or not sel or not (pred | fit | sel) <= train:
        raise ValueError("empty or unauthorized teacher population")
    if pred & (fit | sel):
        raise ValueError("OOF row used for teacher fit/selection")
    pg = {group_key(samples[p]) for p in pred}
    if pg & {group_key(samples[p]) for p in fit | sel}:
        raise ValueError("sequence/capture leakage into teacher")
    # Refit can legitimately include inner selection rows after the epoch was fixed.
    if any(samples[p].split != "train" for p in pred | fit | sel):
        raise ValueError("validation/test row used by teacher")


def validate_predictions(samples, train_positions, fold_ids, prediction_ids, predicted_f, provenance):
    expected = [samples[p].sample_id for p in train_positions]
    if expected != list(prediction_ids) or len(set(prediction_ids)) != len(expected):
        raise ValueError("sample/feature-row identity mismatch")
    if not np.isfinite(predicted_f).all() or len(predicted_f) != len(expected):
        raise ValueError("incomplete OOF predictions")
    coverage = Counter()
    for rec in provenance:
        pred = rec["prediction_positions"]
        validate_fold(samples, train_positions, pred, rec["refit_positions"], rec["selection_positions"])
        if any(fold_ids[p] != rec["fold"] for p in pred):
            raise ValueError("teacher provenance / fold mismatch")
        coverage.update(pred)
    if set(coverage) != set(map(int, train_positions)) or any(v != 1 for v in coverage.values()):
        raise ValueError("each TRAIN row must have exactly one OOF prediction")
    return True
