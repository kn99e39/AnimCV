"""2D observation provenance — where a frame's explicit geometry came from.

The Frame Pose Core consumes explicit 2D joints as a first-class input, so
"which sensor produced these joints" is part of the sample contract, not a
detail of the ingest script. Without it a bank of dataset-shipped annotations
and a bank of AnimCV's own MMPose output are indistinguishable, and a number
measured on one would silently be read as a number about the other.

Three evaluation regimes are named explicitly. The distinction that matters is
**whether a learned 2D detector is part of the observation error**, not whether
the keypoints happened to ship with a dataset:

`oracle_geometry`
    annotated/projected ground-truth 2D, or deterministic synthetic projection
    from known 3D. No learned detector contributes error. Purpose: isolate the
    3D reconstruction architecture from 2D sensor error.

`benchmark_detector_observation`
    a fixed external detector's output distributed with a benchmark — 3DPW's
    shipped OpenPose-format keypoints are the case here. Detector error is
    already present, so this regime does **not** isolate the 3D core from 2D
    observation error; but the observation is not produced by AnimCV's current
    sensor either. Purpose: compare 3D reconstruction architectures under one
    fixed, externally defined observation.

`real_animcv_observation`
    AnimCV's current Real Observation backend of the Geometry Observation Layer
    — MMPose + RTMDet (`pose/mmpose_adapter.py`) with its detector, checkpoint
    and preprocessing. The layer is the abstraction; MMPose is one backend of
    it. Purpose: measure actual AnimCV perception behaviour end to end.

Results from different regimes are never comparable without the label,
dataset-shipped detector output is never called oracle geometry, and
dataset-provided geometry is never described as "the MMPose pipeline".
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


OBSERVATION_SCHEMA = "animcv_frame_pose_observation_provenance_v1"

REGIME_ORACLE = "oracle_geometry"
REGIME_BENCHMARK_DETECTOR = "benchmark_detector_observation"
REGIME_REAL_ANIMCV = "real_animcv_observation"
# For artifacts whose provenance cannot be determined. Never assigned to a newly
# built bank, and any interpretation that depends on observation quality must
# refuse it (`assert_quality_interpretable`).
REGIME_HISTORICAL_UNKNOWN = "historical_unknown"
# A bank label only, never a sample label: a deliberately multi-regime bank
# built with the explicit opt-in. No single sample is ever "mixed".
REGIME_MIXED = "mixed"
REGIMES = (REGIME_ORACLE, REGIME_BENCHMARK_DETECTOR, REGIME_REAL_ANIMCV, REGIME_HISTORICAL_UNKNOWN)

# Regimes in which a claim about 2D observation quality is meaningful.
INTERPRETABLE_REGIMES = (REGIME_ORACLE, REGIME_BENCHMARK_DETECTOR, REGIME_REAL_ANIMCV)

# Historical regime spellings, mapped forward deterministically on load.
# `oracle_geometry` is deliberately absent: an old artifact carrying it is not
# resolved by the label alone, only by its backend (see `migrate_regime`).
_LEGACY_REGIME_RENAMES = {"real_observation": REGIME_REAL_ANIMCV,
                          "unlabeled": REGIME_HISTORICAL_UNKNOWN}

BACKEND_DATASET_GROUND_TRUTH = "dataset_ground_truth"
BACKEND_DATASET_DETECTOR = "dataset_detector"
BACKEND_SYNTHETIC_PROJECTION = "synthetic_projection"
BACKEND_MMPOSE = "mmpose"
BACKEND_UNRECORDED = "unrecorded"
BACKENDS = (BACKEND_DATASET_GROUND_TRUTH, BACKEND_DATASET_DETECTOR,
            BACKEND_SYNTHETIC_PROJECTION, BACKEND_MMPOSE, BACKEND_UNRECORDED)

# Backend and regime stay separate concepts, but they are not independent: what
# produced an observation determines whether a learned detector is in its error.
# This table is the invariant, and it is enforced rather than inferred from
# whether the provider was "a dataset".
# Backends whose observations are produced by *reading RGB pixels*. Their cache
# identity must bind the exact image bytes, because the same sensor run over
# different pixels is a different observation.
#
# The other backends do not read an image to produce their keypoints:
#   dataset_ground_truth   projected from the dataset's own 3D annotation
#   synthetic_projection   projected from mocap through a virtual camera
#   dataset_detector       consumed as a distributed keypoint artifact; AnimCV
#                          never re-runs that detector, so the identity source
#                          is the annotation artifact (already covered by the
#                          bank's content digest), not an image AnimCV read
# so an image digest is optional for them and its absence is meaningful rather
# than a missing check.
IMAGE_GENERATED_BACKENDS = (BACKEND_MMPOSE,)

BACKEND_REGIME = {
    BACKEND_DATASET_GROUND_TRUTH: REGIME_ORACLE,
    BACKEND_SYNTHETIC_PROJECTION: REGIME_ORACLE,
    BACKEND_DATASET_DETECTOR: REGIME_BENCHMARK_DETECTOR,
    BACKEND_MMPOSE: REGIME_REAL_ANIMCV,
    BACKEND_UNRECORDED: REGIME_HISTORICAL_UNKNOWN,
}


@dataclass(frozen=True)
class ObservationProvenance:
    """Auditable identity of one sample's 2D observation source."""

    backend: str
    observation_type: str
    regime: str
    detail: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.backend not in BACKENDS:
            raise ValueError(f"unknown observation backend {self.backend!r}; known: {list(BACKENDS)}")
        if self.regime not in REGIMES:
            raise ValueError(f"unknown observation regime {self.regime!r}; known: {list(REGIMES)}")
        expected = BACKEND_REGIME[self.backend]
        if self.regime != expected:
            raise ValueError(
                f"backend {self.backend!r} always belongs to regime {expected!r}, not {self.regime!r}; "
                "in particular, detector output is never oracle geometry merely because it shipped "
                "with a dataset")

    def to_dict(self) -> dict[str, Any]:
        return {"schema": OBSERVATION_SCHEMA, "backend": self.backend,
                "observation_type": self.observation_type, "regime": self.regime,
                "detail": dict(self.detail)}

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "ObservationProvenance":
        """Load provenance, migrating a historical artifact deterministically.

        An artifact written before the three-way taxonomy may carry a regime
        that no longer means what it says -- most importantly 3DPW's shipped
        detector keypoints were labelled `oracle_geometry`. `migrate_regime`
        resolves those from the recorded backend, which is unambiguous, and
        refuses to guess when it is not.
        """
        if not payload:
            return UNRECORDED
        backend = str(payload["backend"])
        regime = migrate_regime(backend, str(payload["regime"]))
        detail = dict(payload.get("detail") or {})
        if regime != payload["regime"]:
            detail = {**detail, "migrated_from_regime": payload["regime"]}
        return cls(backend, str(payload["observation_type"]), regime, detail)

    def cache_key(self) -> str:
        """Digest of everything that makes a cached observation stale.

        Covers the backend identity plus, for an estimated observation, its
        model, weights, config and preprocessing. It deliberately does not cover
        the input image; that is `observation_cache_key`'s job.

        The four identities and what each is responsible for:

        `FrameBank.content_digest`        numeric bank identity: frames, split,
                                          source, arrays. It does **not** cover
                                          image bytes, so it cannot detect an
                                          in-place JPEG replacement.
        `FrameBank.provenance_fingerprint`  recorded observation and modality
                                          provenance, and image *references*.
        `observation_cache_key`           this key plus the exact image bytes,
                                          wherever the sensor consumes RGB.
        `visual_input.visual_input_fingerprint`  exact visual-input identity for
                                          frozen-feature caches.
        """
        payload = {"backend": self.backend, "observation_type": self.observation_type,
                   "regime": self.regime, "detail": self.detail}
        return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()


_CONTENT_DIGEST_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def image_content_digest(path: str | Path) -> str:
    """SHA-256 over the exact bytes of one image.

    Content, not location: a file can be replaced while keeping its path, its
    name and even its mtime, and an observation cached from the old pixels would
    then be silently reused for the new ones.
    """
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def observation_cache_key(provenance: ObservationProvenance, image_digest: str | None) -> str:
    """Per-sample invalidation key: sensor identity plus exact image content.

    `image_digest` must be an `image_content_digest` value, never a path. A path
    is rejected rather than hashed, because hashing a path binds the filename
    and not the pixels the sensor actually saw.

    For an **image-generated** observation (`IMAGE_GENERATED_BACKENDS` — today
    the MMPose + RTMDet Real AnimCV backend) the digest is **mandatory**. A
    Real AnimCV observation is produced by reading an RGB frame, so an identity
    that omits which frame is not an identity at all; omitting it would let a
    cache built from one set of pixels be reused for another.

    It stays optional only where the observation is not produced from pixels:
    projected ground truth, synthetic projection, and dataset-distributed
    detector keypoints that AnimCV consumes as artifacts rather than
    regenerating.
    """
    if provenance.backend in IMAGE_GENERATED_BACKENDS and image_digest is None:
        raise ValueError(
            f"{provenance.backend} observations are produced by reading an RGB frame, so their "
            "cache identity must bind the exact image bytes; pass "
            "framepose.observations.image_content_digest(path)")
    if image_digest is not None and not _CONTENT_DIGEST_PATTERN.match(image_digest):
        raise ValueError(
            "observation_cache_key expects a 64-character SHA-256 of the image bytes "
            "(framepose.observations.image_content_digest), not a path or filename")
    digest = hashlib.sha256()
    digest.update(provenance.cache_key().encode("utf-8"))
    digest.update(b"|")
    digest.update((image_digest or "").encode("utf-8"))
    return digest.hexdigest()


def migrate_regime(backend: str, recorded: str) -> str:
    """Resolve a recorded regime label onto the current taxonomy.

    The backend wins whenever it is known, because it determines the regime by
    definition (`BACKEND_REGIME`). An unknown backend with an unrecognised label
    is not guessed at -- it becomes `historical_unknown`.
    """
    if backend in BACKEND_REGIME:
        return BACKEND_REGIME[backend]
    if recorded in REGIMES:
        return recorded
    return _LEGACY_REGIME_RENAMES.get(recorded, REGIME_HISTORICAL_UNKNOWN)


def assert_quality_interpretable(regime: str) -> str:
    """Refuse a claim about observation quality on an unresolvable artifact."""
    if regime not in INTERPRETABLE_REGIMES:
        raise ValueError(
            f"regime {regime!r} carries no resolvable 2D observation provenance; any interpretation "
            "that depends on observation quality must not be made from this artifact")
    return regime


UNRECORDED = ObservationProvenance(
    backend=BACKEND_UNRECORDED, observation_type="unrecorded", regime=REGIME_HISTORICAL_UNKNOWN,
    detail={"note": "artifact predates the observation-provenance contract"},
)

# Keyed by the `input_kind` the prepared lifter datasets already record, so the
# provenance is read from the upstream artifact rather than guessed per source.
DATASET_OBSERVATIONS: dict[str, ObservationProvenance] = {
    "dataset_ground_truth_2d": ObservationProvenance(
        backend=BACKEND_DATASET_GROUND_TRUTH, observation_type="projected_ground_truth_2d",
        regime=REGIME_ORACLE,
        detail={"note": "dataset ground-truth 3D projected with the dataset's own calibration"}),
    "official_3dpw_2d_detection": ObservationProvenance(
        backend=BACKEND_DATASET_DETECTOR, observation_type="dataset_shipped_detector_2d",
        regime=REGIME_BENCHMARK_DETECTOR,
        detail={"note": "2D keypoints shipped inside the 3DPW release (OpenPose-format, 18 joints); "
                        "detector output, but not produced by AnimCV and not MMPose"}),
    "synthetic_virtual_camera_gt_2d": ObservationProvenance(
        backend=BACKEND_SYNTHETIC_PROJECTION, observation_type="synthetic_virtual_camera_2d",
        regime=REGIME_ORACLE,
        detail={"note": "mocap joints projected through a synthetic virtual camera"}),
}


def mmpose_observation(*, pose_config: str, pose_checkpoint: str,
                       detector_config: str | None = None, detector_checkpoint: str | None = None,
                       visibility_threshold: float, input_size: str | None = None,
                       mmpose_version: str | None = None,
                       pose_weights_sha256: str | None = None,
                       detector_weights_sha256: str | None = None,
                       extra: dict[str, Any] | None = None) -> ObservationProvenance:
    """Provenance for the current Real AnimCV backend of the Geometry Observation
    Layer — MMPose + RTMDet.

    Every argument that changes the produced keypoints is part of `cache_key`,
    so a cached observation bank is invalidated by a change of model, weights,
    config or preprocessing. Mirrors what `pose.mmpose_adapter.MMPoseConfig`
    actually parameterises.
    """
    detail = {
        "pose_config": pose_config,
        "pose_checkpoint": pose_checkpoint,
        "pose_weights_sha256": pose_weights_sha256,
        "detector_config": detector_config,
        "detector_checkpoint": detector_checkpoint,
        "detector_weights_sha256": detector_weights_sha256,
        "visibility_threshold": visibility_threshold,
        "input_size": input_size,
        "mmpose_version": mmpose_version,
        "adapter": "pose.mmpose_adapter.PoseEstimator",
        "landmark_schema": "canonical_v1",
    }
    detail.update(extra or {})
    return ObservationProvenance(
        backend=BACKEND_MMPOSE, observation_type="estimated_2d", regime=REGIME_REAL_ANIMCV, detail=detail)


def resolve_dataset_observation(input_kind: str | None) -> ObservationProvenance:
    """Map a prepared dataset's declared `input_kind` onto provenance."""
    if input_kind is None:
        return UNRECORDED
    if input_kind not in DATASET_OBSERVATIONS:
        raise ValueError(
            f"unknown prepared-dataset input_kind {input_kind!r}; register it in "
            "framepose.observations.DATASET_OBSERVATIONS rather than defaulting it")
    return DATASET_OBSERVATIONS[input_kind]


def summarize(provenances: list[ObservationProvenance]) -> dict[str, Any]:
    backends: dict[str, int] = {}
    regimes: dict[str, int] = {}
    keys: dict[str, str] = {}
    for provenance in provenances:
        backends[provenance.backend] = backends.get(provenance.backend, 0) + 1
        regimes[provenance.regime] = regimes.get(provenance.regime, 0) + 1
        keys[provenance.backend] = provenance.cache_key()
    return {"backends": dict(sorted(backends.items())), "regimes": dict(sorted(regimes.items())),
            "cache_keys": dict(sorted(keys.items()))}


def assert_single_regime(provenances: list[ObservationProvenance]) -> str:
    """Refuse to treat a mixed-regime set as one measurement."""
    present = sorted({provenance.regime for provenance in provenances})
    if len(present) != 1:
        raise ValueError(
            "frame set mixes evaluation regimes " + str(present) +
            "; oracle geometry, benchmark detector output and real AnimCV observations are not "
            "comparable without a label")
    return present[0]
