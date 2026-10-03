"""DIAGNOSTIC (Worklog 71): TRAIN-split sampling contracts for the fixed Worklog 69 visual head.

S0_FRAME_UNIFORM is the Worklog 69 epoch exactly: one random permutation of
all TRAIN positions per epoch from a CPU generator seeded 1337.
S1_SEQUENCE_BALANCED draws the same number of examples per epoch: a TRAIN
sequence uniformly, then one eligible frame of it uniformly (with
replacement).  Neither sampler looks at targets, depth or errors beyond the
fixed eligibility mask (at least one GT-valid arm segment).  Validation and
test are never resampled.  Not imported by production code.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np

SAMPLERS = ("S0_FRAME_UNIFORM", "S1_SEQUENCE_BALANCED")


def train_sequence_groups(sequence_ids: list[str], train_positions: np.ndarray,
                          eligible: np.ndarray) -> dict[str, np.ndarray]:
    """TRAIN positions per sequence, restricted to eligible rows, in position order."""
    groups: dict[str, list[int]] = {}
    for p in np.asarray(train_positions):
        if bool(eligible[p]):
            groups.setdefault(sequence_ids[p], []).append(int(p))
    return {sid: np.asarray(v, dtype=np.int64) for sid, v in sorted(groups.items())}


class EpochSampler:
    """Deterministic per-epoch example orders for one sampling contract."""

    def __init__(self, name: str, train_positions: np.ndarray, groups: dict[str, np.ndarray], seed: int):
        import torch

        if name not in SAMPLERS:
            raise ValueError(f"unknown sampler {name!r}")
        self.name = name
        self.train_positions = np.asarray(train_positions, dtype=np.int64)
        self.groups = groups
        self.sequences = list(groups)
        self.draws_per_epoch = len(self.train_positions)
        self.generator = torch.Generator(device="cpu").manual_seed(seed)

    def epoch(self) -> np.ndarray:
        import torch

        if self.name == "S0_FRAME_UNIFORM":
            return self.train_positions[torch.randperm(len(self.train_positions), generator=self.generator).numpy()]
        seq = torch.randint(len(self.sequences), (self.draws_per_epoch,), generator=self.generator).numpy()
        u = torch.rand(self.draws_per_epoch, generator=self.generator, dtype=torch.float64).numpy()
        out = np.empty(self.draws_per_epoch, dtype=np.int64)
        for k, s in enumerate(seq):
            frames = self.groups[self.sequences[s]]
            out[k] = frames[min(int(u[k] * len(frames)), len(frames) - 1)]
        return out


def contribution_counts(orders: list[np.ndarray], sequence_ids: list[str]) -> dict[str, int]:
    counter: Counter = Counter()
    for order in orders:
        counter.update(sequence_ids[p] for p in order)
    return dict(sorted(counter.items()))


def contribution_summary(counts: dict[str, int], groups: dict[str, np.ndarray]) -> dict[str, Any]:
    """How closely sampled counts follow frame frequency vs uniform-over-sequence."""
    total = sum(counts.values())
    frames = {sid: len(v) for sid, v in groups.items()}
    total_frames = sum(frames.values())
    share = {sid: counts.get(sid, 0) / total for sid in groups}
    frame_share = {sid: frames[sid] / total_frames for sid in groups}
    uniform = 1.0 / len(groups)
    return {"total_draws": total, "sequences": len(groups),
            "max_abs_deviation_from_frame_share": max(abs(share[s] - frame_share[s]) for s in groups),
            "max_abs_deviation_from_uniform": max(abs(share[s] - uniform) for s in groups),
            "largest_share": max(share.values()), "smallest_share": min(share.values()),
            "per_sequence": {sid: {"draws": counts.get(sid, 0), "share": share[sid],
                                   "eligible_frames": frames[sid], "frame_share": frame_share[sid]}
                             for sid in groups}}


def cosine_by_lag(features: np.ndarray, ordered_positions: np.ndarray, lag: int) -> np.ndarray:
    """Cosine similarity between rows ``lag`` apart in a sequence's frame order."""
    if len(ordered_positions) <= lag:
        return np.zeros(0)
    a = np.asarray(features[ordered_positions[:-lag]], dtype=np.float32)
    b = np.asarray(features[ordered_positions[lag:]], dtype=np.float32)
    na, nb = np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1)
    return np.sum(a * b, axis=1) / np.maximum(na * nb, 1e-12)
