"""VLM Sign Advisor — discrete orientation evidence from one RGB frame.

The advisor answers exactly the seven questions of the Sign Contract
(`framepose.signs`) and nothing else. It never emits XYZ, depth magnitude,
metric offsets, bone lengths, a continuous embedding, or image patch tokens; its
only interface to the Frame Pose Core is a validated categorical `SignState`.

It may reason internally in language, but no free-form prose is persisted into
the pose path: the response is parsed against a strict schema and **rejected**
rather than guessed at when it does not conform.

Frame-level only, by contract: the advisor sees frame *n*'s person crop and no
neighbour, no flow, no temporal vote.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

import numpy as np

from framepose.signs import NEGATIVE, POSITIVE, SIGN_FIELD_NAMES, UNKNOWN


PROMPT_SCHEMA_VERSION = "animcv_sign_advisor_prompt_v1"
SIGN_BANK_SCHEMA = "animcv_vlm_sign_bank_v1"

# The image the advisor sees is the same deterministic person-centric box the
# pose model's crop contract defines, rasterised larger purely for legibility.
# The box, margin rule and mapping are unchanged; only the raster size differs,
# and it is recorded in provenance.
ADVISOR_CROP_RESOLUTION = 448

# Each contract field, phrased as one visual question with two branches and an
# explicit "unclear". The mapping from answer to sign is fixed here so the
# prompt and the contract cannot drift apart.
_QUESTIONS: tuple[tuple[str, str, dict[str, int]], ...] = (
    ("torso_facing",
     "Is the person's chest turned toward the camera, or is their back turned toward the camera?",
     {"chest_toward_camera": NEGATIVE, "back_toward_camera": POSITIVE, "unclear": UNKNOWN}),
    ("shoulder_forward_depth",
     "Which of the person's own shoulders is CLOSER to the camera?",
     {"their_right_shoulder_closer": NEGATIVE, "their_left_shoulder_closer": POSITIVE,
      "unclear": UNKNOWN}),
    ("hip_forward_depth",
     "Which of the person's own hips is CLOSER to the camera?",
     {"their_right_hip_closer": NEGATIVE, "their_left_hip_closer": POSITIVE, "unclear": UNKNOWN}),
    ("left_elbow_forward_bend",
     "Think of the straight line from the person's own LEFT shoulder to their own LEFT wrist. "
     "Is their LEFT elbow bent toward the camera (nearer than that line) or away from the camera "
     "(farther than that line)?",
     {"toward_camera": NEGATIVE, "away_from_camera": POSITIVE, "unclear": UNKNOWN}),
    ("right_elbow_forward_bend",
     "Think of the straight line from the person's own RIGHT shoulder to their own RIGHT wrist. "
     "Is their RIGHT elbow bent toward the camera or away from the camera?",
     {"toward_camera": NEGATIVE, "away_from_camera": POSITIVE, "unclear": UNKNOWN}),
    ("left_knee_forward_bend",
     "Think of the straight line from the person's own LEFT hip to their own LEFT ankle. "
     "Is their LEFT knee bent toward the camera or away from the camera?",
     {"toward_camera": NEGATIVE, "away_from_camera": POSITIVE, "unclear": UNKNOWN}),
    ("right_knee_forward_bend",
     "Think of the straight line from the person's own RIGHT hip to their own RIGHT ankle. "
     "Is their RIGHT knee bent toward the camera or away from the camera?",
     {"toward_camera": NEGATIVE, "away_from_camera": POSITIVE, "unclear": UNKNOWN}),
)

_FIELD_ORDER = tuple(name for name, _, _ in _QUESTIONS)
assert _FIELD_ORDER == SIGN_FIELD_NAMES, "the prompt must cover exactly the Sign Contract fields"

_ANSWERS = {name: mapping for name, _, mapping in _QUESTIONS}


def prompt_text() -> str:
    """The exact instruction sent with every frame."""
    lines = [
        "You are looking at a cropped photograph of one person.",
        "Answer ONLY about the orientation of that person's body relative to the camera.",
        "Never estimate coordinates, distances, depths in metres, or joint positions.",
        "",
        "Answer these questions. 'Their own left/right' means the person's own left and right,",
        "not the left and right of the image. If you genuinely cannot tell, answer \"unclear\".",
        "",
    ]
    for index, (name, question, mapping) in enumerate(_QUESTIONS, start=1):
        options = " | ".join(key for key in mapping)
        lines.append(f"{index}. {name}: {question}")
        lines.append(f"   allowed answers: {options}")
    lines += [
        "",
        "Reply with a single JSON object and nothing else, using exactly these keys:",
        "{" + ", ".join(f'"{name}": "<allowed answer>"' for name in _FIELD_ORDER) + "}",
    ]
    return "\n".join(lines)


def prompt_provenance() -> dict[str, Any]:
    return {
        "schema_version": PROMPT_SCHEMA_VERSION,
        "prompt": prompt_text(),
        "fields": list(_FIELD_ORDER),
        "answer_mapping": {name: dict(mapping) for name, _, mapping in _QUESTIONS},
        "advisor_crop_resolution": ADVISOR_CROP_RESOLUTION,
        "temporal_context": "none; frame n only",
        "forbidden_outputs": ["XYZ", "depth magnitude", "metric offsets", "bone lengths",
                              "continuous embeddings", "image patch tokens", "free-form prose"],
    }


@dataclass(frozen=True)
class AdvisorResponse:
    """One parsed response: either a valid sign state, or an explicit rejection."""

    valid: bool
    state: np.ndarray
    reason: str | None = None
    raw: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"valid": self.valid, "state": [int(value) for value in self.state],
                "reason": self.reason, "raw": self.raw}


def _extract_object(text: str) -> dict[str, Any] | None:
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match is None:
        return None
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def parse_response(text: str) -> AdvisorResponse:
    """Validate a response against the schema.  Malformed output is rejected.

    A rejected response yields an all-`UNKNOWN` state, which is the neutral
    value the contract already defines — never a guessed branch.
    """
    unknown = np.zeros(len(_FIELD_ORDER), dtype=np.int8)
    payload = _extract_object(text or "")
    if payload is None:
        return AdvisorResponse(False, unknown, "no JSON object in response", text or "")
    missing = [name for name in _FIELD_ORDER if name not in payload]
    if missing:
        return AdvisorResponse(False, unknown, f"missing fields: {missing}", text)
    values = []
    for name in _FIELD_ORDER:
        answer = payload[name]
        if not isinstance(answer, str):
            return AdvisorResponse(False, unknown, f"{name} is not a string", text)
        key = answer.strip().lower()
        if key not in _ANSWERS[name]:
            return AdvisorResponse(False, unknown, f"{name} has unrecognised answer {answer!r}", text)
        values.append(_ANSWERS[name][key])
    return AdvisorResponse(True, np.asarray(values, dtype=np.int8), None, text)


def bank_metadata(*, model_id: str, revision: str | None, weights_sha256: str | None,
                  parameter_count: int, dtype: str, generation: dict[str, Any],
                  bank_content_digest: str, sample_count: int) -> dict[str, Any]:
    return {
        "schema": SIGN_BANK_SCHEMA,
        "model": {"id": model_id, "revision": revision, "weights_sha256": weights_sha256,
                  "parameter_count": int(parameter_count), "dtype": dtype,
                  "role": "discrete sign advisor only; never a pose estimator"},
        "generation": generation,
        "prompt": prompt_provenance(),
        "sign_contract_fields": list(SIGN_FIELD_NAMES),
        "bank_content_digest": bank_content_digest,
        "sample_count": int(sample_count),
        "separate_from": "the F1/F2 dense visual feature cache, which is not reused here",
    }
