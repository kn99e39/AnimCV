"""Deterministic frame research bank.

The bank exists so architecture and loss research can run in minutes on a fixed,
fingerprinted, split-safe set of frames before anything expensive is trained.

Determinism: selection is by per-sequence stride only. There is no RNG in bank
construction, so the same inputs always produce the same `content_digest`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from framepose import strata as strata_module
from framepose.contract import FrameBank, FrameSample, assert_split_isolation, modality_summary
from framepose.observations import REGIME_HISTORICAL_UNKNOWN, REGIME_MIXED
from framepose.sources import frames_from_prepared_dataset, load_prepared_dataset, resolve_spec


@dataclass(frozen=True)
class BankRequest:
    """One prepared dataset contributing one split to the bank."""

    source: str
    split: str
    dataset_path: Path
    stride: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {"source": self.source, "split": self.split,
                "dataset_path": str(self.dataset_path), "stride": self.stride}


def build_bank(requests: list[BankRequest], *, image_roots: dict[str, str | Path] | None = None,
               require_rgb: bool = False, verify_images: bool = True,
               allow_mixed_regime: bool = False) -> tuple[FrameBank, dict[str, Any]]:
    """Assemble, filter, stratify and validate a frame bank.

    `require_rgb=True` builds the paired-modality subset used for the controlled
    F0/F1/F2 comparison: every retained sample has real imagery that actually
    exists on disk, so all three candidates see exactly the same frames.

    Sources whose 2D geometry belongs to different observation regimes are
    refused by default -- pooling oracle geometry with benchmark detector output
    silently changes what a number means. `allow_mixed_regime=True` is the
    explicit opt-in for a deliberately multi-regime bank; it records the
    breakdown and labels the bank `mixed`, and downstream interpretation must
    still handle that rather than assume one regime.
    """
    if not requests:
        raise ValueError("at least one bank request is required")
    image_roots = {key: Path(value) for key, value in (image_roots or {}).items()}

    samples: list[FrameSample] = []
    columns: dict[str, list[np.ndarray]] = {key: [] for key in
                                            ("input_2d", "input_valid", "target_3d", "target_valid")}
    intake: list[dict[str, Any]] = []
    for request in requests:
        spec = resolve_spec(request.source)
        payload = load_prepared_dataset(request.dataset_path)
        source_samples, arrays = frames_from_prepared_dataset(
            payload, spec=spec, split=request.split, stride=request.stride)
        kept = _filter(source_samples, image_roots, require_rgb=require_rgb, verify_images=verify_images)
        intake.append({**request.to_dict(), "available_frames": len(source_samples),
                       "retained_frames": len(kept), "modality": spec.modality.to_dict(),
                       "observation": (source_samples[0].observation.to_dict() if source_samples else None)})
        if not kept:
            continue
        keep_index = np.asarray(kept, dtype=np.int64)
        samples.extend(source_samples[position] for position in kept)
        for key in columns:
            columns[key].append(arrays[key][keep_index])

    if not samples:
        raise ValueError("no frame samples survived bank construction")
    # Checked before assembly so a leaking intake reports leakage rather than
    # colliding sample ids.
    assert_split_isolation(samples)
    merged = {key: np.concatenate(values, axis=0) for key, values in columns.items()}
    bank = FrameBank(samples, merged)
    bank.assert_split_isolation()

    report = _stratify(bank)
    report.update({
        "intake": intake,
        "modality_by_source": modality_summary(bank.samples),
        "split_counts": {split: int(len(bank.indices(split))) for split in ("train", "validation", "test")},
        "sequence_counts": {split: len(sequences)
                            for split, sequences in bank.split_sequences().items()},
        "require_rgb": require_rgb,
        "content_digest": bank.content_digest(),
        "observation": bank.observation_summary(),
        "provenance_fingerprint": bank.provenance_fingerprint(),
        # A newly built bank must land in a labelled regime; the oracle /
        # benchmark-detector / real-AnimCV distinction is what makes its numbers
        # readable at all.
        "regime": _bank_regime(bank, allow_mixed_regime),
    })
    if REGIME_HISTORICAL_UNKNOWN in report["observation"]["regimes"]:
        raise ValueError(
            "bank construction produced unresolvable 2D observation provenance; register the prepared "
            "dataset's input_kind in framepose.observations.DATASET_OBSERVATIONS")
    bank.metadata.update(report)
    return bank, report


def _bank_regime(bank: FrameBank, allow_mixed_regime: bool) -> str:
    try:
        return bank.regime()
    except ValueError:
        if not allow_mixed_regime:
            raise
        return REGIME_MIXED


def _filter(samples: list[FrameSample], image_roots: dict[str, Path], *,
            require_rgb: bool, verify_images: bool) -> list[int]:
    """Deterministic retention.  Never silently drops for an unrecorded reason."""
    kept: list[int] = []
    for position, sample in enumerate(samples):
        if require_rgb:
            if not sample.modality.has_rgb or sample.image_reference is None:
                continue
            if verify_images:
                if sample.image_reference.root_key not in image_roots:
                    raise KeyError(
                        f"image root {sample.image_reference.root_key!r} must be mapped to verify RGB samples")
                if not sample.image_reference.resolve(image_roots).is_file():
                    continue
        kept.append(position)
    return kept


def _stratify(bank: FrameBank) -> dict[str, Any]:
    """Compute per-frame quantities, fit thresholds on train, assign strata."""
    quantities = [
        strata_module.frame_quantities(
            bank.arrays["input_2d"][position], bank.arrays["input_valid"][position],
            bank.arrays["target_3d"][position], bank.arrays["target_valid"][position],
            bank.samples[position].image_size)
        for position in range(len(bank))
    ]
    train_positions = bank.indices("train")
    fitting = [quantities[position] for position in train_positions] or quantities
    thresholds = strata_module.fit_thresholds(fitting)

    for position, sample in enumerate(bank.samples):
        assigned = strata_module.assign_strata(quantities[position], thresholds)
        sample.strata.clear()
        sample.strata.update(assigned)
        sample.strata.update({key: value for key, value in quantities[position].items()})

    per_split = {}
    for split in ("train", "validation", "test"):
        positions = bank.indices(split)
        if not len(positions):
            continue
        per_split[split] = strata_module.summarize([bank.samples[position].strata for position in positions])
    return {
        "strata_thresholds": thresholds,
        "strata_threshold_fit_split": "train",
        "strata_counts": per_split,
    }


def load_bank(index_path: str | Path) -> FrameBank:
    return FrameBank.load(index_path)
