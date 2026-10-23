"""EXPERIMENTAL (Worklog 92): frozen-VLM torso up/down advisor for the GT-free heading's mirrored-sign failure.

Not a production default; complements ``motion.runtime_heading`` (Worklog 91).  Declared before any VLM response was
seen.

Failure it targets (Worklog 92 first gate): when the torso is lying or strongly bent, the heading -- the yaw of the
right-minus-left shoulder axis -- reduces to which shoulder is nearer the camera, and S0 sometimes picks the mirrored
depth order (3DHP: all 31 rows > 135 deg are lying poses; mirroring fixes 30).  The magnitude is right; the depth sign
of the shoulder axis is wrong.

Contract (one mechanism, three decisions KEEP / MIRROR / ABSTAIN; never an arbitrary yaw):
  * invocation (runtime only): S0's lifted torso is not upright -- the angle between S0 pelvis->thorax and vertical is
    >= ``INVOKE_MIN_INCLINATION_DEG`` (45: the torso is closer to horizontal than vertical) -- and the detector torso
    (both shoulders and hips) is valid;
  * visual input: the FramePose crop (observation-derived square box, ``crop_box`` / ``render_crop``) rendered at
    448 px, with NO overlay (no joints, labels, GT, S0 or correctness);
  * model: Qwen/Qwen3-VL-8B-Instruct, revision 0c351dd0..., 4-bit NF4, greedy (the Worklog 67 pinned environment);
  * question: torso-specific -- is the person on their BACK (chest/belly up) or STOMACH (chest/belly down), or UNCLEAR.
    Asked twice with the option order swapped (``PROMPTS``); an answer is used only when both agree on BACK/STOMACH;
  * geometry: for a lying torso the chest normal is n = +Z (BACK) or -Z (STOMACH); with the torso axis b = thorax -
    pelvis, the shoulder axis is s = n x b, whose depth component is s_y = n_z * b_x.  b_x is the image-x component of
    the detector's shoulder-midpoint minus hip-midpoint (observed in 2D, normalized by the 2D torso length).  The
    predicted depth sign of the shoulder axis is sign(n_z * b_x); S0's is sign(sin psi);
  * decision: ABSTAIN if the answers disagree or are UNCLEAR/unparseable, if |b_x| < ``MIN_AXIS_X`` (torso axis toward
    the camera), or if |sin psi| < sin ``MIN_SIDE_DEG`` (the depth sign barely matters); else KEEP when the signs
    agree, MIRROR (psi -> -psi) when they disagree.  ABSTAIN keeps the S0 heading.
Head direction is never used.  Expected failures: seated/leaning torsos (BACK/STOMACH ill-posed; should be UNCLEAR),
side-lying bodies, multi-person crops, and VLM answer priors.
"""

from __future__ import annotations

import hashlib
import math
import re

import numpy as np

MODEL_REPOSITORY = "Qwen/Qwen3-VL-8B-Instruct"
MODEL_REVISION = "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
GENERATION = {"do_sample": False, "max_new_tokens": 8, "num_beams": 1}
ADVISOR_RESOLUTION = 448
INVOKE_MIN_INCLINATION_DEG = 45.0
MIN_AXIS_X = 0.25
MIN_SIDE_DEG = 20.0
PROMPTS = {
    "back_first": ("Look at the person in this image. Is the person lying or reclining on their back (chest and belly facing up, "
                   "toward the ceiling or sky) or on their stomach (chest and belly facing down, toward the floor)? If the person "
                   "is not lying down, or you cannot tell, answer UNCLEAR. Answer with exactly one word: BACK, STOMACH, or UNCLEAR."),
    "stomach_first": ("Look at the person in this image. Is the person lying or reclining on their stomach (chest and belly facing down, "
                      "toward the floor) or on their back (chest and belly facing up, toward the ceiling or sky)? If the person "
                      "is not lying down, or you cannot tell, answer UNCLEAR. Answer with exactly one word: STOMACH, BACK, or UNCLEAR."),
}
PROMPT_SHA256 = {k: hashlib.sha256(v.encode()).hexdigest() for k, v in PROMPTS.items()}
ANSWERS = ("BACK", "STOMACH", "UNCLEAR")


def parse_answer(text: str) -> str:
    """BACK / STOMACH / UNCLEAR, or INVALID when not exactly one allowed word is present."""
    words = set(re.findall(r"[A-Z]+", text.upper()))
    hits = [a for a in ANSWERS if a in words]
    return hits[0] if len(hits) == 1 else "INVALID"


def torso_inclination_degrees(lifted_frame) -> float | None:
    """Angle between the lifted pelvis->thorax vector and vertical (+Z), in degrees."""
    p, t = lifted_frame.points.get("pelvis"), lifted_frame.points.get("thorax")
    if p is None or t is None:
        return None
    v = np.subtract(t.position, p.position)
    n = float(np.linalg.norm(v))
    if not math.isfinite(n) or n < 1e-9:
        return None
    return math.degrees(math.acos(max(-1.0, min(1.0, float(v[2]) / n))))


def torso_axis_x(input_2d_row, input_valid_row, image_size) -> float | None:
    """Image-x component of (shoulder midpoint - hip midpoint) over the 2D torso length; None if the torso is invalid."""
    x = np.asarray(input_2d_row, float)
    v = np.asarray(input_valid_row, bool)
    if not (v[1] and v[4] and v[11] and v[14]):
        return None
    size = np.asarray(image_size, float)
    d = ((x[11, :2] + x[14, :2]) - (x[1, :2] + x[4, :2])) / 2 * size
    n = float(np.linalg.norm(d))
    return float(d[0] / n) if n > 1e-6 else None


def should_invoke(inclination: float | None, axis_x: float | None) -> bool:
    return inclination is not None and axis_x is not None and inclination >= INVOKE_MIN_INCLINATION_DEG


def decide(yaw, answers: dict, axis_x):
    """(advised yaw, action, reason) with action in {'KEEP', 'MIRROR', 'ABSTAIN'}."""
    if yaw is None:
        return yaw, "ABSTAIN", "no_heading"
    a, b = answers.get("back_first"), answers.get("stomach_first")
    if a != b or a not in ("BACK", "STOMACH"):
        return yaw, "ABSTAIN", "inconsistent_or_unclear"
    if axis_x is None or abs(axis_x) < MIN_AXIS_X:
        return yaw, "ABSTAIN", "torso_axis_toward_camera"
    if abs(math.sin(yaw)) < math.sin(math.radians(MIN_SIDE_DEG)):
        return yaw, "ABSTAIN", "depth_sign_irrelevant"
    n_z = 1.0 if a == "BACK" else -1.0
    predicted = np.sign(n_z * axis_x)
    if predicted == np.sign(math.sin(yaw)):
        return yaw, "KEEP", "agree"
    return -yaw, "MIRROR", "disagree"
