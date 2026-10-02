"""DIAGNOSTIC (Worklog 67): Qwen3-VL as an RGB near/far ordering sensor for the left arm.

The VLM never predicts XYZ, depth values or confidences.  It answers exactly
three pairwise questions (S vs E, E vs W, S vs W) with FIRST_CLOSER /
SECOND_CLOSER / UNCLEAR about a deterministic person-centric crop on which
the benchmark DETECTOR's left shoulder / elbow / wrist are labelled S / E / W.
Nothing on the image or in the prompt encodes GT, H0 or Depth Anything.

Not imported by any production module.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any

import numpy as np

from framepose.contract import JOINT_INDEX
from framepose.crops import CROP_CONTRACT, crop_box, render_crop

MODEL_REPOSITORY = "Qwen/Qwen3-VL-8B-Instruct"
MODEL_REVISION = "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
QUANTIZATION = {"load_in_4bit": True, "bnb_4bit_quant_type": "nf4", "bnb_4bit_compute_dtype": "bfloat16",
                "bnb_4bit_use_double_quant": False}
GENERATION = {"do_sample": False, "max_new_tokens": 64, "num_beams": 1}
# The crop region, margin, padding and resampling are the FramePose crop
# contract; only the rendered side length is fixed here for the advisor image
# (twice the 224-pixel model crop), once, before any evaluation.
ADVISOR_RESOLUTION = 448
LABELS = (("S", "left_shoulder"), ("E", "left_elbow"), ("W", "left_wrist"))
STATES = ("FIRST_CLOSER", "SECOND_CLOSER", "UNCLEAR")
FIELDS = ("shoulder_elbow", "elbow_wrist", "shoulder_wrist")
FIELD_SEGMENT = {"shoulder_elbow": "upper", "elbow_wrist": "lower", "shoulder_wrist": "chain"}
SEGMENT_FIELD = {v: k for k, v in FIELD_SEGMENT.items()}

PROMPT = (
    "This image shows one person. Three points on the same person's LEFT arm are marked by "
    "an automatic 2D keypoint detector: S = left shoulder, E = left elbow, W = left wrist. "
    "For each pair, decide which marked point is physically nearer to the camera in 3D. "
    "You may use perspective, overlap, foreshortening and occlusion visible in the image. "
    "If the image does not support a defensible ordering for a pair, answer UNCLEAR.\n"
    "Pairs (FIRST vs SECOND): shoulder_elbow = S vs E; elbow_wrist = E vs W; shoulder_wrist = S vs W.\n"
    "Allowed values: FIRST_CLOSER, SECOND_CLOSER, UNCLEAR.\n"
    "Reply with exactly one JSON object and nothing else, in this form:\n"
    '{"shoulder_elbow": "...", "elbow_wrist": "...", "shoulder_wrist": "..."}'
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


PROMPT_SHA256 = sha256_bytes(PROMPT.encode())


# ------------------------------------------------------------ advisor image


def advisor_image(image_rgb: np.ndarray, input_2d: np.ndarray, input_valid: np.ndarray,
                  image_size: tuple[int, int]) -> tuple[np.ndarray, dict[str, Any]]:
    """Deterministic crop with detector S/E/W labels; returns (uint8 RGB, crop record)."""
    from PIL import Image, ImageDraw, ImageFont

    if (image_rgb.shape[1], image_rgb.shape[0]) != tuple(image_size):
        raise ValueError("image does not match the sample's image size")
    box = crop_box(input_2d, input_valid, image_size)
    crop = render_crop(image_rgb, box, ADVISOR_RESOLUTION)
    canvas = Image.fromarray(crop)
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    width, height = image_size
    points = {}
    for label, joint in LABELS:
        index = JOINT_INDEX[joint]
        if not bool(input_valid[index]):
            raise ValueError(f"{joint} detector point is not valid")
        px = float(input_2d[index, 0]) * width
        py = float(input_2d[index, 1]) * height
        u = (px - box.x) / box.side * ADVISOR_RESOLUTION
        v = (py - box.y) / box.side * ADVISOR_RESOLUTION
        points[label] = (round(u, 3), round(v, 3))
        r = 4
        draw.ellipse((u - r, v - r, u + r, v + r), fill=(255, 255, 0), outline=(0, 0, 0))
        tx, ty = u + 6, v - 12
        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            draw.text((tx + dx, ty + dy), label, fill=(0, 0, 0), font=font)
        draw.text((tx, ty), label, fill=(255, 255, 0), font=font)
    rendered = np.asarray(canvas, dtype=np.uint8)
    record = {"crop_contract": CROP_CONTRACT["schema"], "box": box.to_dict(), "resolution": ADVISOR_RESOLUTION,
              "label_pixels": points, "label_source": "benchmark detector input_2d",
              "rendered_sha256": sha256_bytes(rendered.tobytes())}
    return rendered, record


# ------------------------------------------------------------------ parser


@dataclass(frozen=True)
class ParsedResponse:
    states: dict[str, str] | None
    status: str  # valid / invalid_json / invalid_schema / empty
    fence_normalized: bool

    def state(self, field: str) -> str:
        return self.states[field] if self.states is not None else "UNKNOWN"


_FENCE = re.compile(r"^```(?:json)?\s*\n(.*)\n```$", re.DOTALL)


def parse_response(text: str) -> ParsedResponse:
    """Strict: one JSON object, exactly the three fields, each an allowed state.

    One outer Markdown code fence may be removed (recorded).  Nothing else is
    repaired; any other deviation makes every field UNKNOWN.
    """
    raw = (text or "").strip()
    if not raw:
        return ParsedResponse(None, "empty", False)
    fenced = _FENCE.match(raw)
    body = fenced.group(1).strip() if fenced else raw
    try:
        value = json.loads(body)
    except json.JSONDecodeError:
        return ParsedResponse(None, "invalid_json", bool(fenced))
    if not isinstance(value, dict) or set(value) != set(FIELDS) or any(value[f] not in STATES for f in FIELDS):
        return ParsedResponse(None, "invalid_schema", bool(fenced))
    return ParsedResponse({f: value[f] for f in FIELDS}, "valid", bool(fenced))


# ------------------------------------------------------- canonical mapping


def ordering_sign(state: str) -> int:
    """Pair (proximal, distal): FIRST_CLOSER -> distal farther -> +1 in canonical f.

    AnimCV +Y is farther from the camera, so a positive forward fraction of
    proximal->distal means the distal point is farther (the first is closer).
    UNCLEAR / UNKNOWN -> 0.
    """
    return {"FIRST_CLOSER": 1, "SECOND_CLOSER": -1}.get(state, 0)


def gt_ordering(f_gt: float, stable_sine: float) -> tuple[int, bool]:
    """Evaluation only: (sign of GT f, whether |f| exceeds the stable threshold)."""
    return int(np.sign(f_gt)), bool(abs(f_gt) > stable_sine)


def runtime_disagreement(h0_f: float, da_relation: float | None) -> bool:
    """Runtime-only: both non-zero and opposite signs.  GT is never consulted."""
    if da_relation is None:
        return False
    a, b = np.sign(h0_f), np.sign(da_relation)
    return bool(a != 0 and b != 0 and a != b)


def v0_select(h0_f: float, d1_f: float, vlm_state: str) -> tuple[float, str]:
    """V0_VLM_SELECTOR for one segment: (chosen forward fraction, source)."""
    sh, sd = np.sign(h0_f), np.sign(d1_f)
    if sh == sd or sh == 0 or sd == 0:
        return float(h0_f), "h0_same_sign"
    vote = ordering_sign(vlm_state)
    if vote == 0:
        return float(h0_f), "h0_vlm_unclear_or_invalid"
    if vote == sh:
        return float(h0_f), "h0_vlm_agrees"
    return float(d1_f), "d1_vlm_agrees"


def shuffled_donors(keys: list[tuple[str, int]]) -> list[int]:
    """Deterministic donor index for each row: half-way rotation, next row from another sequence."""
    order = sorted(range(len(keys)), key=lambda i: keys[i])
    n = len(order)
    donors = [0] * n
    for rank, i in enumerate(order):
        step = n // 2
        for k in range(n):
            j = order[(rank + step + k) % n]
            if keys[j][0] != keys[i][0]:
                donors[i] = j
                break
        else:
            raise ValueError("no donor from a different sequence")
    return donors
