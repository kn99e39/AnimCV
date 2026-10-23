import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from motion.vlm_heading_sign_advisor import (
    INVOKE_MIN_INCLINATION_DEG, MIN_AXIS_X, MIN_SIDE_DEG, PROMPTS, decide, parse_answer, should_invoke,
    torso_axis_x, torso_inclination_degrees,
)

ROOT = Path(__file__).resolve().parent.parent
BOTH = lambda a: {"back_first": a, "stomach_first": a}  # noqa: E731


def test_answer_parsing_is_strict():
    assert parse_answer("BACK") == "BACK" and parse_answer(" stomach.") == "STOMACH" and parse_answer("Unclear") == "UNCLEAR"
    assert parse_answer("BACK or STOMACH") == "INVALID" and parse_answer("") == "INVALID"
    # the two prompts only swap the option order
    assert PROMPTS["back_first"] != PROMPTS["stomach_first"] and sorted(PROMPTS["back_first"].split()) != []


def test_mirrored_heading_geometry_on_a_supine_torso():
    """Supine (chest up), head toward image left: the right shoulder is nearer the camera, shoulder axis -> -Y, yaw -90."""
    b = np.array([-1.0, 0.0, 0.0])  # thorax - pelvis points to image left
    n = np.array([0.0, 0.0, 1.0])  # chest up
    s = np.cross(n, b)  # right - left shoulder axis
    gt_yaw = math.atan2(s[1], s[0])
    assert math.degrees(gt_yaw) == pytest.approx(-90.0)
    axis_x = -1.0
    assert decide(-gt_yaw, BOTH("BACK"), axis_x) == (gt_yaw, "MIRROR", "disagree")  # S0 picked the mirror: corrected
    assert decide(gt_yaw, BOTH("BACK"), axis_x) == (gt_yaw, "KEEP", "agree")
    # prone with the same axis predicts the opposite depth sign
    assert decide(gt_yaw, BOTH("STOMACH"), axis_x)[1] == "MIRROR"


def test_abstention_policy():
    y = math.radians(-90)
    assert decide(y, {"back_first": "BACK", "stomach_first": "STOMACH"}, -1.0)[1:] == ("ABSTAIN", "inconsistent_or_unclear")
    assert decide(y, BOTH("UNCLEAR"), -1.0)[1] == "ABSTAIN" and decide(y, BOTH("INVALID"), -1.0)[1] == "ABSTAIN"
    assert decide(y, BOTH("BACK"), 0.5 * MIN_AXIS_X)[1:] == ("ABSTAIN", "torso_axis_toward_camera")
    near_front = math.pi - math.radians(MIN_SIDE_DEG) / 2
    assert decide(near_front, BOTH("BACK"), -1.0)[1:] == ("ABSTAIN", "depth_sign_irrelevant")
    assert decide(None, BOTH("BACK"), -1.0)[1] == "ABSTAIN"
    # ABSTAIN keeps the S0 heading; no arbitrary yaw is ever produced
    for ans in ("BACK", "STOMACH", "UNCLEAR"):
        out, act, _ = decide(y, BOTH(ans), -1.0)
        assert out in (y, -y)


def test_invocation_uses_runtime_quantities_only():
    pt = lambda xyz: SimpleNamespace(position=xyz)  # noqa: E731
    lying = SimpleNamespace(points={"pelvis": pt((0, 0, 0)), "thorax": pt((-0.5, 0.0, 0.05))})
    upright = SimpleNamespace(points={"pelvis": pt((0, 0, 0)), "thorax": pt((0.0, 0.0, 0.5))})
    assert torso_inclination_degrees(lying) > INVOKE_MIN_INCLINATION_DEG > torso_inclination_degrees(upright)
    x = np.zeros((17, 3))
    v = np.zeros(17, bool)
    for j, (px, py) in {11: (0.3, 0.5), 14: (0.3, 0.52), 1: (0.6, 0.5), 4: (0.6, 0.52)}.items():
        x[j, :2], v[j] = (px, py), True
    ax = torso_axis_x(x, v, (1000, 1000))
    assert ax == pytest.approx(-1.0)
    assert should_invoke(torso_inclination_degrees(lying), ax) and not should_invoke(torso_inclination_degrees(upright), ax)
    v[11] = False
    assert torso_axis_x(x, v, (1000, 1000)) is None


def test_gt_isolation_and_no_production_wiring():
    mod = (ROOT / "src/motion/vlm_heading_sign_advisor.py").read_text().split('"""', 2)[2]
    run = (ROOT / "scripts/run_vlm_heading_sign_advisor.py").read_text()
    for text in (mod, run.split('"""', 2)[2]):
        assert "target_3d" not in text and "oracle" not in text and "_matched_oracle_sequence" not in text
    for path in sorted((ROOT / "src").rglob("*.py")):
        if path.name != "vlm_heading_sign_advisor.py":
            assert "vlm_heading_sign_advisor" not in path.read_text(), path
