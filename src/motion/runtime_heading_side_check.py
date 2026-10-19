"""EXPERIMENTAL (Worklog 91A): runtime side-view facing check for the GT-free heading, from the 2D nose.

Not a production default; complements ``motion.runtime_heading`` (Worklog 91) for its documented failure: in side-on
views out of domain, S0's lifted shoulder/hip depth can pick the wrong profile direction (3DHP TS2).

Observation (runtime only): the detector's 2D nose (FrameBank ``head`` joint = OpenPose nose), shoulders and hips.
  offset = (x_nose - x_shoulder_midpoint) / |shoulder_midpoint - hip_midpoint|   (image pixels; + = image right)
Geometry: with heading yaw psi (right-minus-left shoulder axis angle, facing the camera = 180 deg) the body-forward
direction in AnimCV camera axes is f = (-sin psi, cos psi), so its image-x component is -sin psi.  In a side view the
nose lies ahead of the shoulders along f, hence sign(offset) predicts sign(-sin psi).  In near-front/back views
(small |sin psi|) the offset carries little profile information and the check does not apply.

Rule (one mechanism; thresholds fixed from geometry before any test-set inspection, checked on 3DPW VALIDATION only):
  applies when the heading is side-ish (|sin psi| >= sin 45 deg), the nose is valid with confidence >= 0.3 and
  |offset| >= 0.10 torso lengths; if sign(-sin psi) disagrees with sign(offset) the profile is mirrored by flipping the
  depth half-plane, psi -> -psi (the shoulder axis's depth sign is negated, its image-x component kept).
  Otherwise the yaw is unchanged.  Per-frame, stateless, causal.
Anticipated failures: head turned relative to the torso (looking over the shoulder), nose mis-detections, and
near-threshold side views.
"""

from __future__ import annotations

import math

import numpy as np

MIN_ABS_SIN = math.sin(math.radians(45.0))
MIN_NOSE_CONFIDENCE = 0.3
MIN_NOSE_OFFSET = 0.10
NOSE, L_SHOULDER, R_SHOULDER, L_HIP, R_HIP = 10, 11, 14, 1, 4


def nose_offset(input_2d_row, input_valid_row, image_size):
    """(offset in torso lengths or None, nose confidence) from one FrameBank input row."""
    x = np.asarray(input_2d_row, float)
    v = np.asarray(input_valid_row, bool)
    conf = float(x[NOSE, 2])
    if not (v[NOSE] and v[L_SHOULDER] and v[R_SHOULDER] and v[L_HIP] and v[R_HIP]):
        return None, conf
    size = np.asarray(image_size, float)
    sh = (x[L_SHOULDER, :2] + x[R_SHOULDER, :2]) / 2 * size
    hp = (x[L_HIP, :2] + x[R_HIP, :2]) / 2 * size
    torso = float(np.linalg.norm(sh - hp))
    if torso <= 1e-6:
        return None, conf
    return float((x[NOSE, 0] * size[0] - sh[0]) / torso), conf


def side_view_check(yaw, offset, conf):
    """(checked yaw, action) with action in {'not_applicable', 'agree', 'flipped'}."""
    if yaw is None or offset is None or conf < MIN_NOSE_CONFIDENCE or abs(offset) < MIN_NOSE_OFFSET \
            or abs(math.sin(yaw)) < MIN_ABS_SIN:
        return yaw, "not_applicable"
    if np.sign(-math.sin(yaw)) == np.sign(offset):
        return yaw, "agree"
    return -yaw, "flipped"
