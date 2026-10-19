"""EXPERIMENTAL (Worklog 91): GT-free runtime body heading with explicit reliability.

Not a production default.  Provider id ``framepose_s0_valid_bilateral_reliability_v1``.

Inputs (all runtime observations):
  * a ``LiftedPoseSequence`` from the frozen S0 neutral-sign FramePose model (``sign_v1/S0``: trained with GT 3D
    supervision only; every sign field UNKNOWN at training and inference, so no GT enters inference) built by the
    unchanged ``pose.framepose_bridge.build_h0_lifted_sequence`` (detector ``input_valid`` / confidence);
  * nothing else -- no GT, no sequence identity, no temporal state.

Heading: exactly the unchanged AnimationSemantics v2 ``current_root_orientation`` (docs/57 valid-only circular fusion
of the right-minus-left shoulder and hip axes, yaw = atan2(dy, dx), facing the camera = 180 deg).  Front/back is
resolved by S0's own lifted depth of the bilateral joints (learned from 2D context, not supplied by an oracle).

Reliability (declared from physical meaning before any test-set inspection; checked only on 3DPW VALIDATION):
  * ``UNKNOWN``    no valid bilateral pair (unchanged docs/57 semantics);
  * ``UNRELIABLE`` yaw computed, but the detector barely sees the torso (minimum 2D confidence of the four
                   contributing torso joints < ``MIN_TORSO_CONFIDENCE`` = 0.3) or the shoulder and hip axes disagree by
                   more than ``MAX_PAIR_DISAGREEMENT_DEG`` = 45 deg (beyond ordinary torso twist; the two pairs then
                   tell different headings);
  * ``KNOWN``      otherwise.
The yaw value is never altered by the reliability state; a consumer that needs a dense signal decides what to do with
UNRELIABLE frames (the Worklog 91 rig replay keys only KNOWN frames, so Blender holds the previous key).
Per-frame and causal: no hold, smoothing, or carry, so it cannot create stale-heading runs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from motion.animation_semantics_v2_bridge import current_root_orientation

PROVIDER_ID = "framepose_s0_valid_bilateral_reliability_v1"
MIN_TORSO_CONFIDENCE = 0.3
MAX_PAIR_DISAGREEMENT_DEG = 45.0
PAIRS = (("shoulders", "left_shoulder", "right_shoulder"), ("hips", "left_hip", "right_hip"))


@dataclass(frozen=True)
class RuntimeHeading:
    status: str  # KNOWN | UNRELIABLE | UNKNOWN
    yaw_radians: float | None
    min_torso_confidence: float | None
    pair_disagreement_degrees: float | None
    reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.status not in ("KNOWN", "UNRELIABLE", "UNKNOWN"):
            raise ValueError(self.status)
        if (self.status == "UNKNOWN") != (self.yaw_radians is None):
            raise ValueError("UNKNOWN carries no yaw; KNOWN/UNRELIABLE carry one")


def _pair_angles(frame):
    out, conf = {}, []
    for label, left, right in PAIRS:
        a, b = frame.points.get(left), frame.points.get(right)
        if a is None or b is None or not (a.observation_valid and b.observation_valid):
            continue
        dx, dy = b.position[0] - a.position[0], b.position[1] - a.position[1]
        if math.hypot(dx, dy) <= 1e-6:
            continue
        out[label] = math.atan2(dy, dx)
        conf += [a.confidence, b.confidence]
    return out, (min(conf) if conf else None)


def runtime_heading(lifted) -> list[RuntimeHeading]:
    out = []
    for frame, current in zip(lifted.frames, current_root_orientation(lifted)):
        if not current.known:
            out.append(RuntimeHeading("UNKNOWN", None, None, None, ("no_valid_bilateral_pair",)))
            continue
        angles, conf = _pair_angles(frame)
        dis = None
        if len(angles) == 2:
            dis = abs((math.degrees(angles["shoulders"] - angles["hips"]) + 180.0) % 360.0 - 180.0)
        reasons = []
        if conf is not None and conf < MIN_TORSO_CONFIDENCE:
            reasons.append("low_torso_confidence")
        if dis is not None and dis > MAX_PAIR_DISAGREEMENT_DEG:
            reasons.append("shoulder_hip_disagreement")
        out.append(RuntimeHeading("UNRELIABLE" if reasons else "KNOWN", current.yaw_radians, conf, dis, tuple(reasons)))
    return out
