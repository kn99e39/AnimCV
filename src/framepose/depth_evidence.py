"""DIAGNOSTIC: Depth Anything V2 relative depth as a FramePose evidence provider (Worklog 65).

This is a FramePose-side contract, separate from the legacy PoseLandmark.z
path (``pose/depth_sampling.py``), whose behaviour is unchanged.  Depth
Anything output is RELATIVE monocular depth: it is never metres, canonical Y,
ground truth or absolute camera distance.

Contract (fixed before any result):
  sampling       nearest depth-map pixel at the benchmark 2D joint
                 (``int(round(x * width))``, ``int(round(y * height))``), as
                 the historical sampler does; invalid or out-of-frame joints
                 are unavailable and never searched around.
  normalization  per frame over the valid joints:
                 d_rel = (d - mean) / std; std below ``MIN_RELATIVE_STD`` or
                 fewer than two valid joints -> frame unavailable.
  orientation    Depth Anything: larger = closer.  Canonical +Y: farther.
                 forward_depth_evidence = -d_rel.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

import numpy as np

PROVIDER = "depth_anything_v2"
ENCODER = "vits"
INPUT_SIZE = 518
SAMPLING_POLICY = "nearest_pixel_at_benchmark_2d_joint_v1"
NORMALIZATION_POLICY = "per_frame_valid_joint_zscore_v1"
ORIENTATION = "forward_depth_evidence = -zscore(depth_anything); depth_anything larger = closer"
MIN_RELATIVE_STD = 1e-6
MIN_VALID_JOINTS = 2


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sample_nearest(depth_map: np.ndarray, input_2d: np.ndarray, input_valid: np.ndarray,
                   image_size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Raw Depth Anything value at each joint's nearest pixel, plus availability.

    ``input_2d`` is the bank convention ``[x / width, y / height, conf]``.
    """
    depth = np.asarray(depth_map, dtype=np.float64)
    height, width = depth.shape[:2]
    if (width, height) != tuple(image_size):
        raise ValueError(f"depth map {width}x{height} differs from the sample image {image_size}")
    points = np.asarray(input_2d, dtype=np.float64)
    valid = np.asarray(input_valid, dtype=bool)
    raw = np.zeros(points.shape[0], dtype=np.float64)
    ok = np.zeros(points.shape[0], dtype=bool)
    for joint in range(points.shape[0]):
        if not valid[joint] or not np.isfinite(points[joint, :2]).all():
            continue
        px = int(round(points[joint, 0] * width))
        py = int(round(points[joint, 1] * height))
        if 0 <= px < width and 0 <= py < height and np.isfinite(depth[py, px]):
            raw[joint] = depth[py, px]
            ok[joint] = True
    return raw, ok


def normalize_relative(raw: np.ndarray, available: np.ndarray) -> tuple[np.ndarray, np.ndarray, bool]:
    """Per-frame z-score over available joints; unavailable joints stay 0.

    Returns (d_rel, availability, frame_ok).  A degenerate frame makes every
    joint unavailable rather than inventing a scale.
    """
    raw = np.asarray(raw, dtype=np.float64)
    available = np.asarray(available, dtype=bool)
    rel = np.zeros_like(raw)
    if int(available.sum()) < MIN_VALID_JOINTS:
        return rel, np.zeros_like(available), False
    values = raw[available]
    std = float(values.std())
    if not np.isfinite(std) or std < MIN_RELATIVE_STD:
        return rel, np.zeros_like(available), False
    rel[available] = (values - float(values.mean())) / std
    return rel, available.copy(), True


def forward_depth_evidence(raw: np.ndarray, available: np.ndarray) -> tuple[np.ndarray, np.ndarray, bool]:
    """Canonical-orientation relative evidence: larger = farther along +Y."""
    rel, ok, frame_ok = normalize_relative(raw, available)
    return np.where(ok, -rel, 0.0), ok, frame_ok


def model_provenance(checkpoint_sha256: str, *, code_revision: str, torch_version: str,
                     device: str, checkpoint_path: str, upstream: dict[str, Any]) -> dict[str, Any]:
    return {"provider": PROVIDER, "encoder": ENCODER, "input_size": INPUT_SIZE,
            "checkpoint_sha256": checkpoint_sha256, "checkpoint_path": checkpoint_path,
            "upstream": upstream, "code_revision": code_revision,
            "torch_version": torch_version, "device": device,
            "output_semantics": "relative monocular depth; larger = closer; not metric"}


def cache_identity(provenance: dict[str, Any], bank_digest: str, input_2d_digest: str) -> dict[str, Any]:
    """Identity of a whole evidence cache.  Per-sample identity adds the image SHA."""
    identity = {"provider": PROVIDER, "encoder": provenance["encoder"],
                "input_size": provenance["input_size"],
                "checkpoint_sha256": provenance["checkpoint_sha256"],
                "code_revision": provenance["code_revision"],
                "sampling_policy": SAMPLING_POLICY, "normalization_policy": NORMALIZATION_POLICY,
                "orientation": ORIENTATION, "bank_content_digest": bank_digest,
                "input_2d_digest": input_2d_digest}
    identity["digest"] = sha256_bytes(json.dumps(identity, sort_keys=True).encode())
    return identity


@dataclass(frozen=True)
class FramePoseDepthEvidence:
    sample_id: str
    sequence_id: str
    frame_index: int
    split: str
    image_sha256: str
    checkpoint_sha256: str
    raw_depth: tuple[float, ...]
    forward_evidence: tuple[float, ...]
    available: tuple[bool, ...]
    frame_available: bool
    provider: str = PROVIDER

    def __post_init__(self) -> None:
        if not (len(self.raw_depth) == len(self.forward_evidence) == len(self.available)):
            raise ValueError("per-joint evidence arrays must align")
        if not self.frame_available and any(self.available):
            raise ValueError("an unavailable frame carries no joint evidence")
        if any(not ok and value != 0.0 for ok, value in zip(self.available, self.forward_evidence)):
            raise ValueError("unavailable joints carry zero evidence")

    @property
    def digest(self) -> str:
        payload = {"sample_id": self.sample_id, "image_sha256": self.image_sha256,
                   "checkpoint_sha256": self.checkpoint_sha256, "sampling": SAMPLING_POLICY,
                   "normalization": NORMALIZATION_POLICY,
                   "raw": [round(v, 9) for v in self.raw_depth], "available": list(self.available)}
        return sha256_bytes(json.dumps(payload, sort_keys=True).encode())

    def to_dict(self) -> dict[str, Any]:
        return {"sample_id": self.sample_id, "sequence_id": self.sequence_id,
                "frame_index": self.frame_index, "split": self.split, "provider": self.provider,
                "image_sha256": self.image_sha256, "checkpoint_sha256": self.checkpoint_sha256,
                "raw_depth": list(self.raw_depth), "forward_evidence": list(self.forward_evidence),
                "available": list(self.available), "frame_available": self.frame_available,
                "semantics": "relative monocular depth evidence; not metric, not canonical Y, not GT",
                "digest": self.digest}


def evidence_from_raw(sample, image_sha256: str, checkpoint_sha256: str, raw: np.ndarray,
                      sampled: np.ndarray) -> FramePoseDepthEvidence:
    forward, ok, frame_ok = forward_depth_evidence(raw, sampled)
    return FramePoseDepthEvidence(sample.sample_id, sample.sequence_id, sample.frame_index, sample.split,
                                  image_sha256, checkpoint_sha256,
                                  tuple(float(v) for v in np.where(sampled, raw, 0.0)),
                                  tuple(float(v) for v in forward), tuple(bool(v) for v in ok), frame_ok)
