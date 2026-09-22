"""Frame-level visual reliability advisor contract.

This module is deliberately separate from :mod:`framepose.sign_advisor`.
The advisor does not estimate a pose, coordinates, depth, a correction, or a
temporal interpolation.  It answers one categorical observation question per
canonical joint so a later bounded experiment can decide whether a structural
anchor is trustworthy.

The parser accepts one optional *outer* Markdown JSON fence because Qwen's
chat template occasionally wraps an otherwise valid object.  No prose,
nested explanation, confidence score, or extra key is accepted.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

from common.canonical_pose import JOINT_NAMES


PROMPT_SCHEMA_VERSION = "animcv_vlm_reliability_advisor_prompt_v1"
RELIABILITY_SCHEMA = "animcv_vlm_joint_reliability_v1"
ADVISOR_CROP_RESOLUTION = 448


class ReliabilityState(str, Enum):
    RELIABLE = "RELIABLE"
    WEAK = "WEAK"
    OCCLUDED = "OCCLUDED"
    OUT_OF_FRAME = "OUT_OF_FRAME"
    UNKNOWN = "UNKNOWN"


RELIABILITY_STATES: tuple[str, ...] = tuple(state.value for state in ReliabilityState)
STATE_TO_INDEX = {state: index for index, state in enumerate(RELIABILITY_STATES)}
INDEX_TO_STATE = {index: state for state, index in STATE_TO_INDEX.items()}
_FENCE_RE = re.compile(r"^```(?:json)?\s*\n?(.*?)\n?```$", re.IGNORECASE | re.DOTALL)


def prompt_text() -> str:
    """Return the fixed one-frame prompt used for every advisor request."""
    joints = ", ".join(JOINT_NAMES)
    return "\n".join([
        "You are inspecting one cropped RGB frame containing one person.",
        "For every canonical joint listed below, classify only whether the joint's visual observation is reliable.",
        "Use the person's own left/right labels, not the image's left/right.",
        "",
        "State definitions:",
        "RELIABLE = clearly visible and the joint location can be identified.",
        "WEAK = visible but blurred, tiny, ambiguous, or only weakly identifiable.",
        "OCCLUDED = the joint is hidden by the person, an object, or another subject.",
        "OUT_OF_FRAME = the joint is outside the image boundary.",
        "UNKNOWN = the image does not support a defensible decision.",
        "",
        "Do not estimate XYZ, depth, a metric correction, a transform, a confidence number,",
        "a detector score, an interpolation, or any text explanation. Do not use temporal context.",
        f"Canonical joints (all are required): {joints}",
        "",
        "Reply with exactly one JSON object whose keys are exactly the canonical joint names",
        "and whose values are exactly one of RELIABLE, WEAK, OCCLUDED, OUT_OF_FRAME, UNKNOWN.",
        "Example shape only:",
        '{"pelvis":"UNKNOWN", "left_hip":"UNKNOWN", "right_hip":"UNKNOWN", "spine":"UNKNOWN", "thorax":"UNKNOWN", "neck":"UNKNOWN", "head":"UNKNOWN", "left_knee":"UNKNOWN", "right_knee":"UNKNOWN", "left_ankle":"UNKNOWN", "right_ankle":"UNKNOWN", "left_shoulder":"UNKNOWN", "right_shoulder":"UNKNOWN", "left_elbow":"UNKNOWN", "right_elbow":"UNKNOWN", "left_wrist":"UNKNOWN", "right_wrist":"UNKNOWN"}',
    ])


def prompt_provenance() -> dict[str, Any]:
    return {
        "schema_version": PROMPT_SCHEMA_VERSION,
        "prompt": prompt_text(),
        "joint_names": list(JOINT_NAMES),
        "states": list(RELIABILITY_STATES),
        "temporal_context": "none; frame n RGB only",
        "detector_confidence_input": False,
        "forbidden_outputs": [
            "XYZ", "depth", "metric correction", "transform", "detector confidence",
            "interpolation", "free-form explanation",
        ],
    }


@dataclass(frozen=True)
class ReliabilityResponse:
    valid: bool
    state: np.ndarray
    reason: str | None = None
    raw: str = ""
    fence_normalized: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": bool(self.valid),
            "state": {name: INDEX_TO_STATE[int(value)]
                       for name, value in zip(JOINT_NAMES, self.state)},
            "reason": self.reason,
            "raw": self.raw,
            "fence_normalized": bool(self.fence_normalized),
        }


def _unknown_state() -> np.ndarray:
    return np.full(len(JOINT_NAMES), STATE_TO_INDEX["UNKNOWN"], dtype=np.int8)


def parse_response(text: str) -> ReliabilityResponse:
    """Strictly parse one reliability object, allowing one outer JSON fence."""
    raw = text or ""
    stripped = raw.strip()
    fence_normalized = False
    match = _FENCE_RE.match(stripped)
    if match:
        stripped = match.group(1).strip()
        fence_normalized = True
    if not stripped:
        return ReliabilityResponse(False, _unknown_state(), "empty response", raw, fence_normalized)
    if not (stripped.startswith("{") and stripped.endswith("}")):
        return ReliabilityResponse(False, _unknown_state(),
                                  "response is not one JSON object or one outer JSON fence",
                                  raw, fence_normalized)
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError as error:
        return ReliabilityResponse(False, _unknown_state(), f"invalid JSON: {error.msg}", raw,
                                  fence_normalized)
    if not isinstance(payload, dict):
        return ReliabilityResponse(False, _unknown_state(), "top-level JSON value is not an object",
                                  raw, fence_normalized)
    expected = set(JOINT_NAMES)
    missing = sorted(expected - set(payload))
    extra = sorted(set(payload) - expected)
    if missing:
        return ReliabilityResponse(False, _unknown_state(), f"missing joints: {missing}", raw,
                                  fence_normalized)
    if extra:
        return ReliabilityResponse(False, _unknown_state(), f"unexpected keys: {extra}", raw,
                                  fence_normalized)
    state = _unknown_state()
    for index, name in enumerate(JOINT_NAMES):
        value = payload[name]
        if not isinstance(value, str):
            return ReliabilityResponse(False, _unknown_state(), f"{name} is not a string", raw,
                                      fence_normalized)
        normalized = value.strip().upper()
        if normalized not in STATE_TO_INDEX:
            return ReliabilityResponse(False, _unknown_state(),
                                      f"{name} has unrecognised state {value!r}", raw,
                                      fence_normalized)
        state[index] = STATE_TO_INDEX[normalized]
    return ReliabilityResponse(True, state, None, raw, fence_normalized)


def state_dict(state: np.ndarray) -> dict[str, str]:
    values = np.asarray(state)
    if values.shape != (len(JOINT_NAMES),):
        raise ValueError(f"reliability state must have shape ({len(JOINT_NAMES)},), got {values.shape}")
    return {name: INDEX_TO_STATE[int(value)] for name, value in zip(JOINT_NAMES, values)}


def state_matrix_to_strings(states: np.ndarray) -> list[list[str]]:
    values = np.asarray(states)
    if values.ndim != 2 or values.shape[1] != len(JOINT_NAMES):
        raise ValueError("reliability state matrix has the wrong shape")
    return [[INDEX_TO_STATE[int(value)] for value in row] for row in values]
