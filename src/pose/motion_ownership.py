"""Ownership and observability model for Root Motion / Foot Contact.

Architecture_v3_FramePose.md lists "Root motion, contact, foot locking,
constraints" as Layer D of the Frame Pose contract, not implemented. Before
any estimator for that layer is written, this module fixes which stage owns
which quantity, so nothing downstream folds articulation, heading,
translation, ground height and contact into one "root confidence" scalar
(see docs/51).

This module is deliberately data, not an algorithm: it is read by the
diagnostics in this package and by anyone deciding what a later Root Motion
/ Contact estimator is allowed to claim.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Owner(Enum):
    """Which stage is responsible for a quantity — never split across two."""

    FRAME_POSE = "frame_pose"                # A: local articulation
    ROOT_ORIENTATION = "root_orientation"    # B: body heading / yaw
    ROOT_MOTION = "root_motion"              # C: global horizontal translation
    GROUND_CONTACT = "ground_contact"        # D: vertical placement vs. ground
    CONTACT = "contact"                      # E: foot planted / moving state
    RELIABILITY = "reliability"              # F: observation reliability metadata


class Observability(Enum):
    """How defensible a quantity's estimate is from current evidence alone."""

    DIRECTLY_OBSERVABLE = "directly_observable"
    WEAKLY_INFERABLE = "weakly_inferable"
    ONLY_TEMPORALLY_INFERABLE = "only_temporally_inferable"
    NOT_OBSERVABLE = "not_observable"


@dataclass(frozen=True)
class QuantityAudit:
    name: str
    owner: Owner
    observability: Observability
    rationale: str


OWNERSHIP_MODEL: tuple[QuantityAudit, ...] = (
    QuantityAudit(
        name="local_articulation",
        owner=Owner.FRAME_POSE,
        observability=Observability.DIRECTLY_OBSERVABLE,
        rationale=(
            "Pelvis-relative joint position is exactly what Frame Pose Core "
            "(Layer A) already regresses per frame; this is its existing "
            "contract and stays untouched by this batch."
        ),
    ),
    QuantityAudit(
        name="body_heading_yaw",
        owner=Owner.ROOT_ORIENTATION,
        observability=Observability.WEAKLY_INFERABLE,
        rationale=(
            "Bilateral shoulder/hip axes give a per-frame yaw observation, but "
            "it degrades under torso twist, near-frontal/near-lateral "
            "ambiguity, and monocular-lifter axis flips — root_motion.py's own "
            "max-yaw-step hold exists to paper over exactly this. See "
            "root_orientation_diagnostic.py for the measured failure rates "
            "before assuming that heuristic is the right long-term owner."
        ),
    ),
    QuantityAudit(
        name="global_horizontal_translation",
        owner=Owner.ROOT_MOTION,
        observability=Observability.NOT_OBSERVABLE,
        rationale=(
            "A root-relative Frame Pose output has the pelvis pinned to the "
            "origin every frame by construction; that representation cannot "
            "itself imply global translation, and monocular RGB carries no "
            "camera-motion estimate to separate character motion from camera "
            "motion. Only conditionally recoverable through an external "
            "constraint such as a planted foot (root_translation_control.py), "
            "and only while the camera is not itself moving/rotating."
        ),
    ),
    QuantityAudit(
        name="ground_height_placement",
        owner=Owner.GROUND_CONTACT,
        observability=Observability.NOT_OBSERVABLE,
        rationale=(
            "No dataset wired into AnimCV (3DPW, MPI-INF-3DHP, AMASS) exposes "
            "a ground-plane or floor-height reference in Frame Pose's "
            "coordinate frame. Do not infer vertical ground placement from "
            "ankle height alone without an actual floor calibration."
        ),
    ),
    QuantityAudit(
        name="foot_planted_or_moving",
        owner=Owner.CONTACT,
        observability=Observability.ONLY_TEMPORALLY_INFERABLE,
        rationale=(
            "A single frame's ankle position says nothing about contact; only "
            "a short window of ankle velocity/height consistency is "
            "informative, and even then only as a candidate signal "
            "(contact.py), never as ground truth — no wired dataset ships a "
            "contact label (contact_reference.py)."
        ),
    ),
    QuantityAudit(
        name="observation_reliability",
        owner=Owner.RELIABILITY,
        observability=Observability.DIRECTLY_OBSERVABLE,
        rationale=(
            "observation_valid / in_frame_mask / ObservationProvenance are "
            "already computed directly from the detector's own output. The "
            "VLM reliability track (docs/50) found no additional signal worth "
            "trusting as an owner; nothing here should fold these signals "
            "together with any quantity above into one scalar."
        ),
    ),
)


def audit_for(name: str) -> QuantityAudit:
    for entry in OWNERSHIP_MODEL:
        if entry.name == name:
            return entry
    raise KeyError(f"no ownership audit entry for {name!r}")
