"""Frame Pose Contract — the frame is the primary data unit.

One sample is one frame of one actor: independently addressable, carrying its
own modality availability, its own provenance, and an explicit reference to its
neighbours so a future Layer B (temporal context) needs no re-ingest.

Numeric arrays are stored in an `.npz` companion rather than inlined into JSON,
and imagery is referenced by `(root_key, relative_path)` rather than copied.
Both files are fingerprinted together.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from common.serialization import read_json, write_json
from framepose.observations import (
    ObservationProvenance, UNRECORDED, assert_single_regime, summarize as summarize_observations,
)
from pose.pose_lifter import H36M_NAMES


# v2 adds the 2D observation provenance contract. A v1 bank still loads; its
# samples come back with `UNRECORDED` provenance and the `unlabeled` regime,
# which is never assigned to a newly built bank.
BANK_SCHEMA = "animcv_frame_pose_bank_v2"
LEGACY_BANK_SCHEMAS = ("animcv_frame_pose_bank_v1",)
SAMPLE_SCHEMA = "animcv_frame_pose_sample_v2"

JOINT_NAMES: tuple[str, ...] = tuple(H36M_NAMES)
JOINT_COUNT = len(JOINT_NAMES)
JOINT_INDEX = {name: index for index, name in enumerate(JOINT_NAMES)}

# Canonical camera frame, identical to the Legacy Temporal Pose Baseline:
# +X right, +Y forward/depth, +Z up.  Layer A introduces no new axis semantics.
COORDINATE_FRAME = "camera_root_relative"
FORWARD_DEPTH_AXIS = 1
BILATERAL_DEPTH_NORMALIZATION = 1.0 / math.sqrt(2.0)

SPLITS = ("train", "validation", "test")

_ARRAY_KEYS = ("input_2d", "input_valid", "target_3d", "target_valid")


@dataclass(frozen=True)
class Modality:
    """What a sample's source actually provides.  Never fabricated."""

    has_2d: bool
    has_3d: bool
    has_rgb: bool
    has_camera: bool

    def to_dict(self) -> dict[str, bool]:
        return {"has_2d": self.has_2d, "has_3d": self.has_3d,
                "has_rgb": self.has_rgb, "has_camera": self.has_camera}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Modality":
        return cls(bool(payload["has_2d"]), bool(payload["has_3d"]),
                   bool(payload["has_rgb"]), bool(payload["has_camera"]))


@dataclass(frozen=True)
class ImageReference:
    """A stable pointer to imagery, resolved through a named root at runtime.

    Storing a root key rather than an absolute path is what lets a bank built on
    one machine be consumed on another (Architecture_v3 section 9).
    """

    root_key: str
    relative_path: str

    def to_dict(self) -> dict[str, str]:
        return {"root_key": self.root_key, "relative_path": self.relative_path}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ImageReference":
        return cls(str(payload["root_key"]), str(payload["relative_path"]))

    def resolve(self, roots: dict[str, str | Path]) -> Path:
        if self.root_key not in roots:
            raise KeyError(f"image root {self.root_key!r} is not mapped; pass --image-root {self.root_key}=<path>")
        return Path(roots[self.root_key]) / self.relative_path


@dataclass(frozen=True)
class FrameSample:
    """Metadata for one frame.  Arrays live in the bank's aligned `.npz`."""

    sample_id: str
    source: str
    sequence_id: str
    frame_index: int
    split: str
    image_size: tuple[int, int]
    modality: Modality
    observation: ObservationProvenance = UNRECORDED
    timestamp: float | None = None
    fps: float | None = None
    image_reference: ImageReference | None = None
    neighbors: dict[str, str | None] = field(default_factory=dict)
    strata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SAMPLE_SCHEMA,
            "sample_id": self.sample_id,
            "source": self.source,
            "sequence_id": self.sequence_id,
            "frame_index": self.frame_index,
            "split": self.split,
            "image_size": list(self.image_size),
            "modality": self.modality.to_dict(),
            "observation": self.observation.to_dict(),
            "timestamp": self.timestamp,
            "fps": self.fps,
            "image_reference": self.image_reference.to_dict() if self.image_reference else None,
            "neighbors": dict(self.neighbors),
            "strata": dict(self.strata),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "FrameSample":
        reference = payload.get("image_reference")
        width, height = payload["image_size"]
        return cls(
            sample_id=str(payload["sample_id"]),
            source=str(payload["source"]),
            sequence_id=str(payload["sequence_id"]),
            frame_index=int(payload["frame_index"]),
            split=str(payload["split"]),
            image_size=(int(width), int(height)),
            modality=Modality.from_dict(payload["modality"]),
            observation=ObservationProvenance.from_dict(payload.get("observation")),
            timestamp=payload.get("timestamp"),
            fps=payload.get("fps"),
            image_reference=ImageReference.from_dict(reference) if reference else None,
            neighbors=dict(payload.get("neighbors") or {}),
            strata=dict(payload.get("strata") or {}),
        )


def make_sample_id(sequence_id: str, frame_index: int) -> str:
    """`<source>:<sequence>:<actor>#<frame_index:06d>` — stable and sortable."""
    if "#" in sequence_id:
        raise ValueError("sequence_id must not contain '#'")
    if frame_index < 0:
        raise ValueError("frame_index must be non-negative")
    return f"{sequence_id}#{frame_index:06d}"


class FrameBank:
    """A deterministic, split-safe, fingerprinted set of frame samples."""

    def __init__(self, samples: list[FrameSample], arrays: dict[str, np.ndarray],
                 metadata: dict[str, Any] | None = None) -> None:
        count = len(samples)
        if count == 0:
            raise ValueError("frame bank must contain at least one sample")
        missing = [key for key in _ARRAY_KEYS if key not in arrays]
        if missing:
            raise ValueError(f"frame bank arrays are missing: {missing}")
        expected = {
            "input_2d": (count, JOINT_COUNT, 3),
            "input_valid": (count, JOINT_COUNT),
            "target_3d": (count, JOINT_COUNT, 3),
            "target_valid": (count, JOINT_COUNT),
        }
        for key, shape in expected.items():
            if tuple(arrays[key].shape) != shape:
                raise ValueError(f"frame bank array {key} has shape {arrays[key].shape}, expected {shape}")
        identifiers = [sample.sample_id for sample in samples]
        if len(set(identifiers)) != count:
            raise ValueError("frame bank sample_id values must be unique")
        unknown = sorted({sample.split for sample in samples} - set(SPLITS))
        if unknown:
            raise ValueError(f"unknown split values in frame bank: {unknown}")
        self.samples = samples
        self.arrays = {
            "input_2d": np.asarray(arrays["input_2d"], dtype=np.float32),
            "input_valid": np.asarray(arrays["input_valid"], dtype=bool),
            "target_3d": np.asarray(arrays["target_3d"], dtype=np.float32),
            "target_valid": np.asarray(arrays["target_valid"], dtype=bool),
        }
        self.metadata = dict(metadata or {})
        self._index = {identifier: position for position, identifier in enumerate(identifiers)}

    def __len__(self) -> int:
        return len(self.samples)

    def position(self, sample_id: str) -> int:
        return self._index[sample_id]

    def indices(self, split: str) -> np.ndarray:
        if split not in SPLITS:
            raise ValueError(f"unknown split {split!r}")
        return np.asarray([i for i, sample in enumerate(self.samples) if sample.split == split], dtype=np.int64)

    def subset_indices(self, *, split: str | None = None, source: str | None = None,
                       require_rgb: bool = False) -> np.ndarray:
        selected = []
        for position, sample in enumerate(self.samples):
            if split is not None and sample.split != split:
                continue
            if source is not None and sample.source != source:
                continue
            if require_rgb and not (sample.modality.has_rgb and sample.image_reference is not None):
                continue
            selected.append(position)
        return np.asarray(selected, dtype=np.int64)

    def split_sequences(self) -> dict[str, set[str]]:
        buckets: dict[str, set[str]] = {name: set() for name in SPLITS}
        for sample in self.samples:
            buckets[sample.split].add(sample.sequence_id)
        return buckets

    def assert_split_isolation(self) -> None:
        """No sequence may appear in two splits.  Guards against test leakage."""
        assert_split_isolation(self.samples)

    def observation_summary(self) -> dict[str, Any]:
        return summarize_observations([sample.observation for sample in self.samples])

    def regime(self) -> str:
        """The single evaluation regime this bank measures.  Raises if mixed."""
        return assert_single_regime([sample.observation for sample in self.samples])

    # ------------------------------------------------------------------ io --

    def save(self, index_path: str | Path) -> tuple[Path, Path]:
        index_path = Path(index_path)
        array_path = index_path.with_suffix(".npz")
        index_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(array_path, **self.arrays)
        write_json(index_path, {
            "schema": BANK_SCHEMA,
            "joint_names": list(JOINT_NAMES),
            "coordinate_frame": COORDINATE_FRAME,
            "forward_depth_axis": FORWARD_DEPTH_AXIS,
            "sample_count": len(self.samples),
            "arrays": array_path.name,
            "metadata": self.metadata,
            "samples": [sample.to_dict() for sample in self.samples],
        })
        return index_path, array_path

    @classmethod
    def load(cls, index_path: str | Path) -> "FrameBank":
        index_path = Path(index_path)
        payload = read_json(index_path)
        if payload.get("schema") not in (BANK_SCHEMA, *LEGACY_BANK_SCHEMAS):
            raise ValueError(f"unsupported frame bank schema: {payload.get('schema')!r}")
        if payload.get("joint_names") != list(JOINT_NAMES):
            raise ValueError("frame bank joint schema mismatch")
        if payload.get("coordinate_frame") != COORDINATE_FRAME:
            raise ValueError("frame bank coordinate frame mismatch")
        array_path = index_path.parent / payload["arrays"]
        with np.load(array_path) as handle:
            arrays = {key: handle[key] for key in _ARRAY_KEYS}
        samples = [FrameSample.from_dict(item) for item in payload["samples"]]
        return cls(samples, arrays, payload.get("metadata"))

    def fingerprint(self, index_path: str | Path) -> dict[str, Any]:
        """Content fingerprint of the exact bytes a run consumed.

        Paths alone are insufficient — prepared artifacts get rebuilt in place.
        Mirrors `scripts/run_lifter_experiments._dataset_fingerprint`.
        """
        index_path = Path(index_path)
        array_path = index_path.with_suffix(".npz")
        report = {
            "index_path": str(index_path),
            "index_sha256": _sha256(index_path),
            "index_byte_size": index_path.stat().st_size,
            "array_path": str(array_path),
            "array_sha256": _sha256(array_path),
            "array_byte_size": array_path.stat().st_size,
            "sample_count": len(self.samples),
            "content_digest": self.content_digest(),
        }
        report["split_counts"] = {name: int(len(self.indices(name))) for name in SPLITS}
        report["observation"] = self.observation_summary()
        return report

    def content_digest(self) -> str:
        """Path-independent digest of sample identity and numeric content.

        Deliberately covers sample identity, split, source and the numeric
        arrays only. Observation provenance is metadata *about* those arrays:
        if the sensor changes, `input_2d` changes and the digest changes with
        it. Keeping provenance out means adding the provenance contract to an
        existing bank does not invalidate feature caches keyed to this digest,
        while a genuinely different observation still does.
        """
        digest = hashlib.sha256()
        digest.update(BANK_SCHEMA.encode("utf-8"))
        for sample in self.samples:
            digest.update(f"{sample.sample_id}|{sample.split}|{sample.source}".encode("utf-8"))
        for key in _ARRAY_KEYS:
            array = self.arrays[key]
            digest.update(key.encode("utf-8"))
            digest.update(np.ascontiguousarray(array).tobytes())
        return digest.hexdigest()


def assert_split_isolation(samples: Iterable[FrameSample]) -> None:
    """No sequence may appear in two splits.  Guards against test leakage.

    Checked on the sample list rather than only on an assembled bank, so a
    leaking intake is reported as leakage instead of as a duplicate-id error.
    """
    buckets: dict[str, set[str]] = {name: set() for name in SPLITS}
    for sample in samples:
        buckets[sample.split].add(sample.sequence_id)
    for first in range(len(SPLITS)):
        for second in range(first + 1, len(SPLITS)):
            shared = buckets[SPLITS[first]] & buckets[SPLITS[second]]
            if shared:
                raise ValueError(
                    f"sequences appear in both {SPLITS[first]} and {SPLITS[second]}: {sorted(shared)[:5]}"
                )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def modality_summary(samples: Iterable[FrameSample]) -> dict[str, dict[str, int]]:
    """Per-source modality availability, reported rather than assumed."""
    summary: dict[str, dict[str, int]] = {}
    for sample in samples:
        bucket = summary.setdefault(sample.source, {
            "sample_count": 0, "has_2d": 0, "has_3d": 0, "has_rgb": 0, "has_camera": 0,
            "resolvable_image_reference": 0,
        })
        bucket["sample_count"] += 1
        for key, value in sample.modality.to_dict().items():
            bucket[key] += int(value)
        bucket["resolvable_image_reference"] += int(sample.image_reference is not None)
    return summary
