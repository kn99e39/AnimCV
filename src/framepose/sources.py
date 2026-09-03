"""Dataset adapters — the only place dataset-specific knowledge is allowed.

Everything downstream (`bank`, `model`, `losses`, `train`, `evaluate`) consumes
the Frame Pose Contract plus modality metadata and nothing else. Onboarding a
future paired commercial dataset means adding one `SourceSpec` here, not
touching the core (Architecture_v3 section 9).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from common.serialization import read_json
from framepose.contract import (
    JOINT_COUNT, JOINT_NAMES, FrameSample, ImageReference, Modality, make_sample_id,
)
from framepose.observations import (
    ObservationProvenance, mmpose_observation, resolve_dataset_observation,
)


# The prepared lifter datasets these adapters read are the historical
# `animcv_supervised_3d_lifter_dataset_v2` artifacts; they are consumed
# read-only and never rewritten.
_SUPPORTED_DATASET_SCHEMAS = (
    "animcv_supervised_3d_lifter_dataset_v2",
    "animcv_supervised_3d_lifter_dataset_v1",
)

THREE_DPW_IMAGE_ROOT_KEY = "3dpw_images"


@dataclass(frozen=True)
class SourceSpec:
    """How one dataset maps onto the Frame Pose Contract.

    `default_input_kind` is the provenance a prepared artifact is expected to
    declare. The artifact's own `input_kind` wins when present, so a source that
    is later re-ingested through a different sensor is recorded as what it
    actually is rather than as what this table assumed.
    """

    name: str
    modality: Modality
    image_root_key: str | None = None
    image_reference: Callable[[str, int], ImageReference | None] | None = None
    default_input_kind: str | None = None


def _three_dpw_image_reference(sequence_id: str, frame_index: int) -> ImageReference | None:
    """`3dpw:<sequence>:actor<k>` -> `imageFiles/<sequence>/image_<index:05d>.jpg`.

    3DPW's frame indices are the image indices, and every actor of a sequence
    shares one image, so the actor suffix is dropped from the path.
    """
    parts = sequence_id.split(":")
    if len(parts) < 2 or parts[0] != "3dpw":
        return None
    return ImageReference(THREE_DPW_IMAGE_ROOT_KEY, f"{parts[1]}/image_{frame_index:05d}.jpg")


SOURCE_SPECS: dict[str, SourceSpec] = {
    # 3DPW ships the recorded imagery alongside its annotations, so it is the
    # only intaken source that can carry the RGB modality honestly.
    "3DPW": SourceSpec(
        name="3DPW",
        modality=Modality(has_2d=True, has_3d=True, has_rgb=True, has_camera=True),
        image_root_key=THREE_DPW_IMAGE_ROOT_KEY,
        image_reference=_three_dpw_image_reference,
        default_input_kind="official_3dpw_2d_detection",
    ),
    # Only `annot.mat` + `camera.calibration` are intaken for MPI-INF-3DHP; the
    # video frames are not part of this repository's data intake.
    "MPI-INF-3DHP": SourceSpec(
        name="MPI-INF-3DHP",
        modality=Modality(has_2d=True, has_3d=True, has_rgb=False, has_camera=True),
        default_input_kind="dataset_ground_truth_2d",
    ),
    # AMASS is marker-derived mocap with synthetic projection: there is no
    # photograph of the performer to restore, and none is fabricated.
    "AMASS": SourceSpec(
        name="AMASS",
        modality=Modality(has_2d=True, has_3d=True, has_rgb=False, has_camera=False),
        default_input_kind="synthetic_virtual_camera_gt_2d",
    ),
}


def load_prepared_dataset(path: str | Path) -> dict[str, Any]:
    payload = read_json(path)
    if payload.get("schema") not in _SUPPORTED_DATASET_SCHEMAS:
        raise ValueError(f"unsupported prepared dataset schema: {payload.get('schema')!r}")
    if payload.get("joint_names") != list(JOINT_NAMES):
        raise ValueError("prepared dataset joint schema mismatch")
    if not payload.get("sequences") and not payload.get("frames"):
        raise ValueError("prepared dataset contains no frames")
    return payload


def observation_for(payload: dict[str, Any], sequence: dict[str, Any],
                    spec: SourceSpec) -> ObservationProvenance:
    """Read the 2D observation provenance the prepared artifact declares.

    Prepared lifter datasets already record `source.input_kind`; the sequence's
    own source wins over the dataset default, and the `SourceSpec` default is
    only the fallback for an artifact that predates that field.
    """
    sequence_source = sequence.get("source") or {}
    dataset_source = payload.get("source") or {}
    input_kind = sequence_source.get("input_kind") or dataset_source.get("input_kind") or spec.default_input_kind
    return resolve_dataset_observation(input_kind)


def frames_from_prepared_dataset(payload: dict[str, Any], *, spec: SourceSpec, split: str,
                                 stride: int = 1) -> tuple[list[FrameSample], dict[str, np.ndarray]]:
    """Convert one prepared lifter dataset into frame-contract samples.

    `stride` decimates temporally within each sequence. 3DPW is 30 fps and
    adjacent frames are near-duplicates; decimation is applied per sequence so
    the surviving samples still cover every sequence, and it never mixes
    sequences.
    """
    if stride < 1:
        raise ValueError("stride must be at least 1")
    sequences = payload.get("sequences") or [{
        "sequence_id": payload.get("sequence_id"), "frames": payload["frames"],
        "source_fps": payload.get("source_fps"), "image_size": payload.get("image_size"),
    }]
    default_size = payload.get("image_size")
    default_fps = payload.get("source_fps")

    samples: list[FrameSample] = []
    input_2d: list[list[list[float]]] = []
    input_valid: list[list[bool]] = []
    target_3d: list[list[list[float]]] = []
    target_valid: list[list[bool]] = []

    for sequence in sequences:
        sequence_id = str(sequence.get("sequence_id") or payload.get("sequence_id") or "")
        if not sequence_id:
            raise ValueError("prepared dataset sequence is missing sequence_id")
        frames = sequence.get("frames") or []
        if not frames:
            continue
        size = sequence.get("image_size") or default_size
        if not size:
            raise ValueError(f"sequence {sequence_id} has no image_size")
        width, height = int(size[0]), int(size[1])
        fps = sequence.get("source_fps") or default_fps
        fps = float(fps) if fps else None
        observation = observation_for(payload, sequence, spec)
        # Sample ids of the retained frames, so `neighbors` can point at real
        # bank members instead of dangling at decimated-away frames.
        retained = frames[::stride]
        identifiers = [make_sample_id(sequence_id, int(frame["frame_index"])) for frame in retained]
        for position, frame in enumerate(retained):
            frame_index = int(frame["frame_index"])
            observations = np.asarray(frame["input_2d"], dtype=np.float64)
            targets = np.asarray(frame["target_3d"], dtype=np.float64)
            supervised = np.asarray(frame["target_valid"], dtype=bool)
            if observations.shape != (JOINT_COUNT, 3) or targets.shape != (JOINT_COUNT, 3):
                raise ValueError(f"frame {frame_index} of {sequence_id} has a non-canonical joint layout")
            # `build_dataset` writes confidence 0 for a landmark the detector did
            # not produce, and its `target_valid` already means
            # "observed AND supervised".
            observed = observations[:, 2] > 0.0
            reference = spec.image_reference(sequence_id, frame_index) if spec.image_reference else None
            samples.append(FrameSample(
                sample_id=identifiers[position],
                source=spec.name,
                sequence_id=sequence_id,
                frame_index=frame_index,
                split=split,
                image_size=(width, height),
                modality=spec.modality,
                observation=observation,
                timestamp=(frame_index / fps) if fps else None,
                fps=fps,
                image_reference=reference,
                neighbors={
                    "previous": identifiers[position - 1] if position > 0 else None,
                    "next": identifiers[position + 1] if position + 1 < len(identifiers) else None,
                },
            ))
            input_2d.append(observations.tolist())
            input_valid.append(observed.tolist())
            target_3d.append(targets.tolist())
            target_valid.append(supervised.tolist())

    if not samples:
        raise ValueError("prepared dataset produced no frame samples")
    arrays = {
        "input_2d": np.asarray(input_2d, dtype=np.float32),
        "input_valid": np.asarray(input_valid, dtype=bool),
        "target_3d": np.asarray(target_3d, dtype=np.float32),
        "target_valid": np.asarray(target_valid, dtype=bool),
    }
    return samples, arrays


def resolve_spec(name: str) -> SourceSpec:
    if name not in SOURCE_SPECS:
        raise ValueError(f"unknown frame-pose source {name!r}; known: {sorted(SOURCE_SPECS)}")
    return SOURCE_SPECS[name]


def mmpose_source_spec(name: str, *, modality: Modality, image_root_key: str | None = None,
                       image_reference: Callable[[str, int], ImageReference | None] | None = None,
                       **provenance) -> tuple[SourceSpec, ObservationProvenance]:
    """Declare a source whose 2D geometry comes from AnimCV's own MMPose sensor.

    The Real Observation regime's ingest is not built in this batch — no MMPose
    outputs exist for the current research bank — but the contract does, so a
    future observation bank records exactly which model, weights, config and
    preprocessing produced its geometry (`framepose.observations`).
    """
    spec = SourceSpec(name=name, modality=modality, image_root_key=image_root_key,
                      image_reference=image_reference, default_input_kind=None)
    return spec, mmpose_observation(**provenance)
