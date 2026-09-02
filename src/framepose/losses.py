"""Frame-level loss contracts.

Section 15 of the migration direction: the first architecture comparison must
not reopen the loss search. The shared starting contract is exactly the Legacy
Temporal Pose Baseline's established stable geometry objective (A5/A9:
coordinate smooth-L1 + bone 0.25 + torso 0.15 + hinge 0.15), evaluated on single
frames instead of window centres.

The structural terms are imported from `training.temporal_lifter` rather than
reimplemented: those helpers are already `(B, 17, 3)`-shaped and frame-local, so
reusing them is what makes "the same objective as A9" a checkable statement
rather than a claim. Nothing in the legacy module is modified.

Candidate isolation: a loss contract is a property of a *run*, never of a frame.
Two contracts are compared by training two candidates on the same frames, never
by assigning different contracts to different frames inside one run — that would
mix gradients and destroy attribution.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

from training.temporal_lifter import (
    BONE_INDICES, TORSO_INDICES, _hinge_loss, _vector_loss,
)


LOSS_SCHEMA = "animcv_frame_pose_loss_contract_v1"


@dataclass(frozen=True)
class LossContract:
    """One named, fully specified training objective."""

    name: str
    coordinate_weight: float = 1.0
    bone_weight: float = 0.0
    torso_weight: float = 0.0
    hinge_weight: float = 0.0
    description: str = ""

    def __post_init__(self) -> None:
        if self.coordinate_weight < 0 or min(self.bone_weight, self.torso_weight, self.hinge_weight) < 0:
            raise ValueError("loss weights must be non-negative")
        if not any((self.coordinate_weight, self.bone_weight, self.torso_weight, self.hinge_weight)):
            raise ValueError("a loss contract must weight at least one term")

    def to_dict(self) -> dict[str, Any]:
        return {"schema": LOSS_SCHEMA, **asdict(self)}


BASELINE_GEOMETRY_V1 = LossContract(
    name="baseline_geometry_v1",
    coordinate_weight=1.0, bone_weight=0.25, torso_weight=0.15, hinge_weight=0.15,
    description=("A5/A9 established stable geometry objective, frame-local: masked smooth-L1 "
                 "coordinate loss plus canonical bone, torso-axis and hinge-bend vector terms."),
)

COORDINATE_ONLY_V1 = LossContract(
    name="coordinate_only_v1", coordinate_weight=1.0,
    description="Masked smooth-L1 coordinate loss alone; the structural-term ablation reference.",
)

LOSS_CONTRACTS: dict[str, LossContract] = {
    contract.name: contract for contract in (BASELINE_GEOMETRY_V1, COORDINATE_ONLY_V1)
}


def resolve_contract(name: str) -> LossContract:
    if name not in LOSS_CONTRACTS:
        raise ValueError(f"unknown loss contract {name!r}; known: {sorted(LOSS_CONTRACTS)}")
    return LOSS_CONTRACTS[name]


def coordinate_loss(torch, prediction, target, mask):
    """Masked smooth-L1, reduced over valid joint coordinates only."""
    numerator = (torch.nn.functional.smooth_l1_loss(prediction, target, reduction="none") * mask).sum()
    return numerator / mask.sum().clamp_min(1.0)


def loss_components(torch, prediction, target, mask) -> dict[str, Any]:
    """Every raw term, unweighted.  Used for reporting and gradient screening."""
    valid = mask.squeeze(-1).bool()
    return {
        "coordinate": coordinate_loss(torch, prediction, target, mask),
        "bone": _vector_loss(torch, prediction, target, valid, BONE_INDICES,
                             lambda first, second: first - second),
        "torso": _vector_loss(torch, prediction, target, valid, TORSO_INDICES,
                              lambda first, second: second - first),
        "hinge": _hinge_loss(torch, prediction, target, valid),
    }


def compute_loss(torch, prediction, target, mask, contract: LossContract):
    """Total weighted objective for one candidate's contract."""
    components = loss_components(torch, prediction, target, mask)
    total = contract.coordinate_weight * components["coordinate"]
    if contract.bone_weight:
        total = total + contract.bone_weight * components["bone"]
    if contract.torso_weight:
        total = total + contract.torso_weight * components["torso"]
    if contract.hinge_weight:
        total = total + contract.hinge_weight * components["hinge"]
    return total
