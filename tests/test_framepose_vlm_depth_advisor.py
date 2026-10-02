import re
from pathlib import Path

import numpy as np
import pytest

from framepose.contract import JOINT_INDEX
from framepose.vlm_depth_advisor import (
    ADVISOR_RESOLUTION, FIELDS, MODEL_REPOSITORY, MODEL_REVISION, PROMPT, QUANTIZATION, GENERATION,
    advisor_image, gt_ordering, ordering_sign, parse_response, runtime_disagreement, shuffled_donors, v0_select,
)

ROOT = Path(__file__).resolve().parent.parent


def _sample():
    image = np.random.default_rng(0).integers(0, 255, (120, 80, 3), dtype=np.uint8)  # H=120, W=80
    x = np.zeros((17, 3))
    valid = np.zeros(17, bool)
    for joint, (u, v) in {"left_shoulder": (0.4, 0.3), "left_elbow": (0.5, 0.45), "left_wrist": (0.55, 0.6),
                          "right_shoulder": (0.6, 0.3), "pelvis": (0.5, 0.7)}.items():
        x[JOINT_INDEX[joint], :2] = (u, v)
        x[JOINT_INDEX[joint], 2] = 0.9
        valid[JOINT_INDEX[joint]] = True
    return image, x, valid


def test_deterministic_crop_and_detector_overlay_identity():
    pytest.importorskip("PIL")
    image, x, valid = _sample()
    a, ra = advisor_image(image, x, valid, (80, 120))
    b, rb = advisor_image(image, x, valid, (80, 120))
    assert a.shape == (ADVISOR_RESOLUTION, ADVISOR_RESOLUTION, 3)
    assert ra == rb and (a == b).all()
    assert ra["label_source"] == "benchmark detector input_2d"
    # Labels sit at the detector points mapped into the crop.
    box = ra["box"]
    su, sv = ra["label_pixels"]["S"]
    assert su == pytest.approx((0.4 * 80 - box["x"]) / box["side"] * ADVISOR_RESOLUTION, abs=1e-3)
    assert sv == pytest.approx((0.3 * 120 - box["y"]) / box["side"] * ADVISOR_RESOLUTION, abs=1e-3)
    moved = x.copy()
    moved[JOINT_INDEX["left_wrist"], :2] = (0.3, 0.6)
    c, rc = advisor_image(image, moved, valid, (80, 120))
    assert rc["rendered_sha256"] != ra["rendered_sha256"]
    invalid = valid.copy()
    invalid[JOINT_INDEX["left_elbow"]] = False
    with pytest.raises(ValueError):
        advisor_image(image, x, invalid, (80, 120))


def test_strict_parser_and_invalid_to_unknown():
    good = '{"shoulder_elbow": "FIRST_CLOSER", "elbow_wrist": "UNCLEAR", "shoulder_wrist": "SECOND_CLOSER"}'
    p = parse_response(good)
    assert p.status == "valid" and not p.fence_normalized and p.state("elbow_wrist") == "UNCLEAR"
    fenced = parse_response("```json\n" + good + "\n```")
    assert fenced.status == "valid" and fenced.fence_normalized
    for bad, status in (("", "empty"), ("closer: S", "invalid_json"),
                        ('{"shoulder_elbow": "FIRST_CLOSER", "elbow_wrist": "UNCLEAR"}', "invalid_schema"),
                        (good.replace("UNCLEAR", "unclear"), "invalid_schema"),
                        (good[:-1] + ', "confidence": 0.9}', "invalid_schema"),
                        ("Here: " + good, "invalid_json")):
        parsed = parse_response(bad)
        assert parsed.status == status and parsed.states is None
        assert all(parsed.state(f) == "UNKNOWN" for f in FIELDS)


def test_fixed_model_provenance_and_prompt_contract():
    assert MODEL_REPOSITORY == "Qwen/Qwen3-VL-8B-Instruct"
    assert MODEL_REVISION == "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
    assert QUANTIZATION["bnb_4bit_quant_type"] == "nf4" and GENERATION["do_sample"] is False
    for forbidden in ("H0", "Depth Anything", "ground truth", "GT", "AnimCV", "confidence"):
        assert forbidden not in PROMPT
    assert "UNCLEAR" in PROMPT and "nearer to the camera" in PROMPT


def test_canonical_near_far_mapping_and_gt_ordering():
    # proximal->distal f > 0: distal farther (+Y) -> the FIRST point is closer.
    assert ordering_sign("FIRST_CLOSER") == 1
    assert ordering_sign("SECOND_CLOSER") == -1
    assert ordering_sign("UNCLEAR") == ordering_sign("UNKNOWN") == 0
    assert gt_ordering(0.3, np.sin(np.radians(10))) == (1, True)
    assert gt_ordering(-0.1, np.sin(np.radians(10))) == (-1, False)


def test_runtime_only_disagreement_population():
    assert runtime_disagreement(0.4, -0.2) is True
    assert runtime_disagreement(0.4, 0.2) is False
    assert runtime_disagreement(0.0, -0.2) is False
    assert runtime_disagreement(0.4, None) is False
    script = (ROOT / "scripts/run_vlm_depth_advisor.py").read_text()
    selection = script[script.index("population = []"):script.index("primary = [")]
    assert "f_gt" not in selection and "target_3d" not in script


def test_v0_selector_rules():
    assert v0_select(0.3, 0.5, "SECOND_CLOSER") == (0.3, "h0_same_sign")
    assert v0_select(0.3, -0.5, "UNCLEAR") == (0.3, "h0_vlm_unclear_or_invalid")
    assert v0_select(0.3, -0.5, "UNKNOWN") == (0.3, "h0_vlm_unclear_or_invalid")
    assert v0_select(0.3, -0.5, "FIRST_CLOSER") == (0.3, "h0_vlm_agrees")
    assert v0_select(0.3, -0.5, "SECOND_CLOSER") == (-0.5, "d1_vlm_agrees")
    assert v0_select(-0.3, 0.5, "FIRST_CLOSER") == (0.5, "d1_vlm_agrees")


def test_shuffled_pairing_is_deterministic_and_cross_sequence():
    keys = [("a", i) for i in range(5)] + [("b", i) for i in range(4)] + [("c", 0)]
    donors = shuffled_donors(keys)
    assert donors == shuffled_donors(keys)
    assert all(keys[d][0] != keys[i][0] for i, d in enumerate(donors))
    with pytest.raises(ValueError):
        shuffled_donors([("a", 0), ("a", 1)])


def test_gt_never_enters_vlm_input_and_no_production_imports():
    module = (ROOT / "src/framepose/vlm_depth_advisor.py").read_text()
    assert not re.search(r"target_3d|jointPositions|three_dpw", module)
    script = (ROOT / "scripts/run_vlm_depth_advisor.py").read_text()
    assert "target_3d" not in script and "f_gt" not in script
    for path in [*sorted((ROOT / "src").rglob("*.py"))]:
        if path.name == "vlm_depth_advisor.py":
            continue
        assert "vlm_depth_advisor" not in path.read_text(), path
