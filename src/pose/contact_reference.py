"""Contact reference inventory (section 7 of docs/51).

No dataset wired into AnimCV — 3DPW, MPI-INF-3DHP, AMASS — ships an explicit
foot-contact ground-truth label (confirmed by inspection of every adapter
under src/pose/*_adapter.py; none parses a contact/mesh-ground field). Every
reference built here is therefore a KINEMATIC_PROXY derived from GT *joint
position* velocity, never an EXACT_LABEL, and must never be reported as true
contact accuracy.
"""

from __future__ import annotations

from enum import Enum

import numpy as np

from pose.contact import ContactState


class ContactReferenceKind(Enum):
    EXACT_LABEL = "exact_label"                        # not available for any wired dataset
    KINEMATIC_PROXY = "kinematic_proxy"                 # derived from GT joint position velocity
    HEURISTIC_PSEUDO_LABEL = "heuristic_pseudo_label"   # derived from a noisy/detector observation


def kinematic_proxy_contact_reference(
    world_or_fixed_camera_positions: list[tuple[float, float, float] | None],
    fps: float,
    *,
    speed_threshold_m_s: float,
) -> list[ContactState]:
    """Reference contact state from world-frame (or fixed-camera) GT joint speed.

    This is a KINEMATIC_PROXY, not ground truth: a genuinely planted foot can
    still read nonzero speed from annotation noise, and a foot that is
    momentarily still mid-air reads as CONTACT under this rule alone. Use it
    only to sanity-check a candidate detector's internal consistency
    (contact.py), or as the "planted" precondition for the root-translation
    control experiment (root_translation_control.py) — never to report a
    "true" contact accuracy number.
    """
    if fps <= 0:
        raise ValueError("fps must be positive")
    if speed_threshold_m_s < 0:
        raise ValueError("speed_threshold_m_s must be non-negative")
    dt = 1.0 / fps
    states: list[ContactState] = []
    for index, position in enumerate(world_or_fixed_camera_positions):
        if position is None:
            states.append(ContactState.UNKNOWN)
            continue
        previous = world_or_fixed_camera_positions[index - 1] if index - 1 >= 0 else None
        following = (
            world_or_fixed_camera_positions[index + 1]
            if index + 1 < len(world_or_fixed_camera_positions) else None
        )
        speed = None
        if previous is not None and following is not None:
            speed = float(np.linalg.norm(np.asarray(following) - np.asarray(previous)) / (2 * dt))
        elif previous is not None:
            speed = float(np.linalg.norm(np.asarray(position) - np.asarray(previous)) / dt)
        elif following is not None:
            speed = float(np.linalg.norm(np.asarray(following) - np.asarray(position)) / dt)
        if speed is None:
            states.append(ContactState.UNKNOWN)
        elif speed <= speed_threshold_m_s:
            states.append(ContactState.CONTACT)
        else:
            states.append(ContactState.MOVING)
    return states
