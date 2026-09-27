"""Bridge from a FrameBank + frozen H0 array to pose.pose_lifter.LiftedPoseSequence.

docs/51's Root Orientation / Contact / Root Translation modules all consume
`pose.pose_lifter.LiftedPoseSequence`, not `framepose.contract.FrameBank`. This
module is the (previously nonexistent) conversion, written so a "does the
signal survive real FramePose output" replay can drive those UNCHANGED
modules with H0 instead of oracle GT.

It reads ONLY `bank.arrays["input_2d"]`, `bank.arrays["input_valid"]`, and the
caller-supplied H0 array -- never `bank.arrays["target_3d"]` or
`bank.arrays["target_valid"]` -- so nothing built here can leak evaluation
ground truth into what is meant to be a production-evidence replay (see
tests/test_framepose_bridge.py for the enforcement test).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from framepose.contract import FrameBank, JOINT_NAMES, SPLITS
from pose.pose_lifter import LiftedPoseFrame, LiftedPosePoint, LiftedPoseSequence


@dataclass(frozen=True)
class H0Identity:
    """What exactly was assembled, so a report can cite it instead of a path."""

    bank_content_digest: str
    split_paths: dict[str, str] = field(default_factory=dict)
    split_sha256: dict[str, str] = field(default_factory=dict)


def assemble_h0(bank: FrameBank, split_paths: dict[str, Path | str]) -> tuple[np.ndarray, H0Identity]:
    """Reassemble a full-bank-length (len(bank), 17, 3) H0 array from per-split .npy files.

    Positions not covered by any split in ``split_paths`` are left as NaN.
    Mirrors the load/assemble pattern in scripts/run_context_refiner.py's
    ``_load_split``/``_assemble_h0`` (not itself an importable function there).
    """
    result = np.full((len(bank), len(JOINT_NAMES), 3), np.nan, dtype=np.float32)
    sha256_by_split: dict[str, str] = {}
    covered: list[int] = []
    for split, path in split_paths.items():
        if split not in SPLITS:
            raise ValueError(f"unknown split {split!r}")
        path = Path(path)
        positions = bank.indices(split)
        values = np.asarray(np.load(path), dtype=np.float32)
        expected_shape = (len(positions), len(JOINT_NAMES), 3)
        if values.shape != expected_shape:
            raise ValueError(f"H0 split {split!r} has shape {values.shape}, expected {expected_shape}")
        result[positions] = values
        sha256_by_split[split] = _sha256_file(path)
        covered.extend(int(p) for p in positions)
    if covered and not np.isfinite(result[covered]).all():
        raise ValueError("assembled H0 contains non-finite values within a supplied split")
    identity = H0Identity(
        bank_content_digest=bank.content_digest(),
        split_paths={key: str(value) for key, value in split_paths.items()},
        split_sha256=sha256_by_split,
    )
    return result, identity


def sequence_frame_positions(bank: FrameBank, sequence_id: str) -> list[int]:
    """Bank row positions for one sequence, sorted by frame_index ascending."""
    positions = [i for i, sample in enumerate(bank.samples) if sample.sequence_id == sequence_id]
    positions.sort(key=lambda i: bank.samples[i].frame_index)
    return positions


def sequence_ids_in_split(bank: FrameBank, split: str) -> list[str]:
    seen: dict[str, None] = {}
    for sample in bank.samples:
        if sample.split == split:
            seen.setdefault(sample.sequence_id, None)
    return list(seen)


def build_h0_lifted_sequence(
    bank: FrameBank, h0: np.ndarray, sequence_id: str, *, fps: float | None = None,
) -> LiftedPoseSequence:
    """One sequence's LiftedPoseSequence from H0 xyz + INPUT-side validity/confidence.

    Deliberately reads ``bank.arrays["input_2d"]`` / ``["input_valid"]`` only.
    """
    positions = sequence_frame_positions(bank, sequence_id)
    if not positions:
        raise ValueError(f"no bank samples for sequence_id {sequence_id!r}")
    input_2d = bank.arrays["input_2d"]
    input_valid = bank.arrays["input_valid"]
    resolved_fps = fps if fps is not None else (bank.samples[positions[0]].fps or 30.0)
    frames = []
    for position in positions:
        sample = bank.samples[position]
        joint_xyz = h0[position]
        joint_finite = np.isfinite(joint_xyz).all(axis=-1)
        points = {}
        for joint_index, name in enumerate(JOINT_NAMES):
            valid = bool(input_valid[position, joint_index]) and bool(joint_finite[joint_index])
            confidence = float(input_2d[position, joint_index, 2])
            points[name] = LiftedPosePoint(
                name=name,
                position=tuple(float(v) for v in joint_xyz[joint_index]),
                confidence=confidence, depth_uncertainty=1.0 - confidence,
                observation_valid=valid,
            )
        timestamp = sample.timestamp if sample.timestamp is not None else sample.frame_index / resolved_fps
        frames.append(LiftedPoseFrame(sample.frame_index, timestamp, points))
    return LiftedPoseSequence(frames=frames, source_fps=resolved_fps, backend="frozen_h0_replay")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
