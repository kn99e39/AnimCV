"""2D observation provenance — where a frame's explicit geometry came from.

The Frame Pose Core consumes explicit 2D joints as a first-class input, so
"which sensor produced these joints" is part of the sample contract, not a
detail of the ingest script. Without it a bank of dataset-shipped annotations
and a bank of AnimCV's own MMPose output are indistinguishable, and a number
measured on one would silently be read as a number about the other.

Two evaluation regimes are therefore named explicitly:

`oracle_geometry`
    dataset-provided 2D (projected ground truth, dataset-shipped detections, or
    synthetic projection). Purpose: isolate the 3D reconstruction architecture
    from 2D sensor error.

`real_observation`
    AnimCV's own Geometry Observation Layer — MMPose (`pose/mmpose_adapter.py`)
    with its detector, checkpoint and preprocessing. Purpose: measure actual
    AnimCV perception behaviour end to end.

Results from the two regimes are never comparable without the label, and
dataset-provided geometry is never described as "the MMPose pipeline".
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any


OBSERVATION_SCHEMA = "animcv_frame_pose_observation_provenance_v1"

REGIME_ORACLE = "oracle_geometry"
REGIME_REAL = "real_observation"
# Reserved for artifacts written before this contract existed; never assigned
# to a newly built bank.
REGIME_UNLABELED = "unlabeled"
REGIMES = (REGIME_ORACLE, REGIME_REAL, REGIME_UNLABELED)

BACKEND_DATASET_GROUND_TRUTH = "dataset_ground_truth"
BACKEND_DATASET_DETECTOR = "dataset_detector"
BACKEND_SYNTHETIC_PROJECTION = "synthetic_projection"
BACKEND_MMPOSE = "mmpose"
BACKEND_UNRECORDED = "unrecorded"
BACKENDS = (BACKEND_DATASET_GROUND_TRUTH, BACKEND_DATASET_DETECTOR,
            BACKEND_SYNTHETIC_PROJECTION, BACKEND_MMPOSE, BACKEND_UNRECORDED)


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
        if self.backend == BACKEND_MMPOSE and self.regime != REGIME_REAL:
            raise ValueError("an MMPose observation belongs to the real_observation regime")
        if self.backend in (BACKEND_DATASET_GROUND_TRUTH, BACKEND_DATASET_DETECTOR,
                            BACKEND_SYNTHETIC_PROJECTION) and self.regime != REGIME_ORACLE:
            raise ValueError(f"{self.backend} is dataset-provided geometry and belongs to oracle_geometry")

    def to_dict(self) -> dict[str, Any]:
        return {"schema": OBSERVATION_SCHEMA, "backend": self.backend,
                "observation_type": self.observation_type, "regime": self.regime,
                "detail": dict(self.detail)}

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "ObservationProvenance":
        if not payload:
            return UNRECORDED
        return cls(str(payload["backend"]), str(payload["observation_type"]),
                   str(payload["regime"]), dict(payload.get("detail") or {}))

    def cache_key(self) -> str:
        """Digest of everything that makes a cached observation stale.

        Covers the backend identity plus, for an estimated observation, its
        model, weights, config and preprocessing. It deliberately does not cover
        the input image: an image change is covered per sample by
        `observation_cache_key`, and bank-wide by `FrameBank.content_digest`.
        """
        payload = {"backend": self.backend, "observation_type": self.observation_type,
                   "regime": self.regime, "detail": self.detail}
        return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def observation_cache_key(provenance: ObservationProvenance, image_relative_path: str | None) -> str:
    """Per-sample invalidation key: sensor identity plus the exact input frame."""
    digest = hashlib.sha256()
    digest.update(provenance.cache_key().encode("utf-8"))
    digest.update(b"|")
    digest.update((image_relative_path or "").encode("utf-8"))
    return digest.hexdigest()


UNRECORDED = ObservationProvenance(
    backend=BACKEND_UNRECORDED, observation_type="unrecorded", regime=REGIME_UNLABELED,
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
        regime=REGIME_ORACLE,
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
    """Provenance for AnimCV's own Geometry Observation Layer.

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
        backend=BACKEND_MMPOSE, observation_type="estimated_2d", regime=REGIME_REAL, detail=detail)


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
            "; oracle_geometry and real_observation results are not comparable without a label")
    return present[0]
