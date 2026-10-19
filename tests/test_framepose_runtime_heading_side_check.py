import math
from pathlib import Path

import numpy as np
import pytest

from motion.runtime_heading_side_check import (
    MIN_ABS_SIN, MIN_NOSE_CONFIDENCE, MIN_NOSE_OFFSET, nose_offset, side_view_check,
)

ROOT = Path(__file__).resolve().parent.parent
SIZE = (1000, 1000)


def _row(nose_x, nose_conf=0.9, valid=True):
    x = np.zeros((17, 3), np.float32)
    v = np.zeros(17, bool)
    for j, (px, py) in {11: (0.50, 0.30), 14: (0.50, 0.30), 1: (0.50, 0.60), 4: (0.50, 0.60), 10: (nose_x, 0.22)}.items():
        x[j] = (px, py, 0.9)
        v[j] = True
    x[10, 2] = nose_conf
    v[10] = valid
    return x, v


def test_offset_is_normalized_by_torso_length_and_needs_valid_joints():
    off, conf = nose_offset(*_row(0.56), SIZE)
    assert off == pytest.approx(0.06 / 0.30) and conf == pytest.approx(0.9)
    assert nose_offset(*_row(0.56, valid=False), SIZE)[0] is None


def test_geometry_convention_and_gating():
    # body forward f = (-sin psi, cos psi): yaw -90 deg faces image right (+X), so the nose sits right of the shoulders
    right = math.radians(-90)
    assert side_view_check(right, +0.2, 0.9) == (right, "agree")
    assert side_view_check(right, -0.2, 0.9) == (-right, "flipped")  # mirrored profile: psi -> -psi
    # near-front/back headings, weak or low-confidence noses, and unknown headings are not touched
    assert side_view_check(math.pi, -0.5, 0.9)[1] == "not_applicable"
    assert side_view_check(math.radians(-90), 0.5 * MIN_NOSE_OFFSET, 0.9)[1] == "not_applicable"
    assert side_view_check(math.radians(-90), 0.5, 0.5 * MIN_NOSE_CONFIDENCE)[1] == "not_applicable"
    assert side_view_check(None, 0.5, 0.9) == (None, "not_applicable")
    edge = math.asin(MIN_ABS_SIN) * 0.99
    assert side_view_check(-edge, -0.5, 0.9)[1] == "not_applicable"


def test_rejected_check_is_not_wired_into_any_provider_or_production_module():
    for path in sorted((ROOT / "src").rglob("*.py")):
        if path.name != "runtime_heading_side_check.py":
            assert "runtime_heading_side_check" not in path.read_text(), path
    text = (ROOT / "src/motion/runtime_heading_side_check.py").read_text()
    assert "target" not in text.split('"""', 2)[2] and "oracle" not in text.split('"""', 2)[2]
