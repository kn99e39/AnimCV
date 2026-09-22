from __future__ import annotations

import json

import numpy as np

from common.canonical_pose import JOINT_NAMES
from framepose.reliability_advisor import (
    INDEX_TO_STATE, STATE_TO_INDEX, parse_response, prompt_text,
)


def _payload(value: str = "UNKNOWN") -> dict[str, str]:
    return {name: value for name in JOINT_NAMES}


def test_reliability_parser_accepts_exact_object_and_outer_json_fence():
    payload = _payload()
    payload["left_wrist"] = "RELIABLE"
    direct = parse_response(json.dumps(payload))
    fenced = parse_response("```json\n" + json.dumps(payload) + "\n```")

    assert direct.valid and fenced.valid
    assert not direct.fence_normalized
    assert fenced.fence_normalized
    assert INDEX_TO_STATE[int(direct.state[JOINT_NAMES.index("left_wrist")])] == "RELIABLE"


def test_reliability_parser_rejects_prose_missing_extra_and_nested_values():
    payload = _payload()
    assert not parse_response("Here is the answer:\n" + json.dumps(payload)).valid
    missing = dict(payload)
    missing.pop("head")
    assert not parse_response(json.dumps(missing)).valid
    extra = dict(payload)
    extra["explanation"] = "no"
    assert not parse_response(json.dumps(extra)).valid
    nested = dict(payload)
    nested["head"] = {"state": "UNKNOWN"}
    assert not parse_response(json.dumps(nested)).valid


def test_prompt_has_all_joints_and_no_detector_confidence_input():
    prompt = prompt_text()
    assert all(name in prompt for name in JOINT_NAMES)
    assert "detector score" in prompt
    assert "confidence number" in prompt
    assert STATE_TO_INDEX["UNKNOWN"] == 4
