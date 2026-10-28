"""Worklog 100: real AnimCV observation adapter, identity accounting and parity helpers (no model or GT needed)."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent


def _script(name):
    sys.path.insert(0, str(ROOT / "scripts"))
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def ev():
    return _script("eval_real_observation_estimator")


def _canon_landmarks(ev, scale=1.0, offset=(0.0, 0.0), score=0.9, hide=()):
    """A synthetic upright body in pixels as the production canonical_v1 landmark array (16, 4)."""
    pts = {"head": (500, 100), "left_shoulder": (560, 200), "right_shoulder": (440, 200), "left_elbow": (580, 300),
           "right_elbow": (420, 300), "left_wrist": (590, 380), "right_wrist": (410, 380), "left_hip": (540, 420),
           "right_hip": (460, 420), "left_knee": (545, 560), "right_knee": (455, 560), "left_ankle": (550, 700),
           "right_ankle": (450, 700)}
    lm = np.full((16, 4), np.nan)
    for n, (x, y) in pts.items():
        lm[ev.CANON.index(n)] = (x * scale + offset[0], y * scale + offset[1], score, float(n not in hide))
    ls, rs, lh, rh = (lm[ev.CANON.index(n)] for n in ("left_shoulder", "right_shoulder", "left_hip", "right_hip"))
    neck = ((ls[0] + rs[0]) / 2, (ls[1] + rs[1]) / 2, min(ls[2], rs[2]), float(ls[3] and rs[3]))
    pelvis = ((lh[0] + rh[0]) / 2, (lh[1] + rh[1]) / 2, min(lh[2], rh[2]), float(lh[3] and rh[3]))
    lm[ev.CANON.index("neck")] = neck
    lm[ev.CANON.index("pelvis")] = pelvis
    lm[ev.CANON.index("spine")] = ((neck[0] + pelvis[0]) / 2, (neck[1] + pelvis[1]) / 2, min(neck[2], pelvis[2]), float(neck[3] and pelvis[3]))
    return lm


def test_benchmark_representation_rules(ev):
    lm = _canon_landmarks(ev, hide=("left_wrist",))
    x, v = ev.to_input(lm, (1000, 800))
    J = ev.JOINTS
    assert J == ["pelvis", "left_hip", "left_knee", "left_ankle", "right_hip", "right_knee", "right_ankle", "spine", "thorax", "neck",
                 "head", "left_shoulder", "left_elbow", "left_wrist", "right_shoulder", "right_elbow", "right_wrist"]
    assert np.array_equal(x[J.index("thorax")], x[J.index("neck")])  # thorax and neck both read the shoulder midpoint
    assert x[J.index("neck")][:2] == pytest.approx((0.5, 0.25))  # x/W, y/H
    assert x[J.index("head")][:2] == pytest.approx((0.5, 0.125))  # head = nose
    assert not v[J.index("left_wrist")] and np.all(x[J.index("left_wrist")] == 0)  # invisible -> invalid, zeros
    assert v.sum() == 16 and x[J.index("pelvis")][2] == pytest.approx(0.9)
    x0, v0 = ev.to_input(None, (1000, 800))
    assert not v0.any() and not x0.any()  # no subject -> all joints invalid


def test_identity_classes_and_w73_distance(ev):
    size = (1000, 800)
    b, vb = ev.to_input(_canon_landmarks(ev), size)
    same, vs = ev.to_input(_canon_landmarks(ev, offset=(3.0, -2.0)), size)
    other, vo = ev.to_input(_canon_landmarks(ev, offset=(300.0, 0.0)), size)
    assert ev.w73_distance(same, vs, b, vb, size) < ev.SAME_PERSON < ev.w73_distance(other, vo, b, vb, size)
    few = vs.copy()
    few[ev.DIRECT[3:]] = False
    assert ev.w73_distance(same, few, b, vb, size) is None  # < 4 shared limb joints -> unmatchable
    assert ev.w73_distance(b, vb, b, vb, size) == 0.0


def test_raw_coords_keep_invisible_landmarks_for_mask_probe(ev):
    lm = _canon_landmarks(ev, hide=("left_knee",))
    x, have = ev.raw_coords(lm, (1000, 800))
    k = ev.JOINTS.index("left_knee")
    assert have[k] and x[k][0] > 0  # coordinates of an invisible joint remain available to the Mmask probe


def test_smpl_projection_joint_map(ev):
    assert ev.SMPL_OF == {1: 1, 2: 4, 3: 7, 4: 2, 5: 5, 6: 8, 11: 16, 12: 18, 13: 20, 14: 17, 15: 19, 16: 21}


def test_gt_free_construction_of_runtime_condition(ev):
    text = (ROOT / "scripts/eval_real_observation_estimator.py").read_text()
    body = text.split("def to_input", 1)[1].split("def raw_coords", 1)[0]
    for bad in ("target", "gt", "smpl", "R_smpl", "actor"):
        assert bad not in body.lower()
    bo = text.split("def build_observations", 1)[1].split("def gt_2d", 1)[0]
    assert "target_3d" not in bo and "jointPositions" not in bo  # no 3D/2D GT while building observations
    assert 'out["M"][i], val["M"][i] = to_input(lm, size)' in bo  # M = tracked production subject only


def test_mmpose_gate_helpers():
    pytest.importorskip("cv2")
    try:
        m = _script("run_real_observation_mmpose")
    except ImportError as exc:  # pose environment not installed
        pytest.skip(str(exc))
    a = np.zeros((3, 16, 4))
    a[..., 2], a[..., 3] = 0.8, 1.0
    b = a.copy()
    b[0, 0, 0] += 0.2
    b[1, 1, 2], b[1, 1, 3] = 0.2995, 0.0
    a[1, 1, 2] = 0.3001
    b[2] = np.nan
    r = m.compare(a, b)
    assert r["max_xy_px"] == pytest.approx(0.2) and r["visibility_flips"] == 1 and r["visibility_flips_near_threshold"] == 1
    assert r["presence_mismatch"] == 16 and r["frames_with_landmarks"] == [3, 2]
    assert m.compare_dets([[((0, 0, 1, 1), 0.9)]], [[((0, 0, 1.3, 1), 0.9)]])["max_box_px"] == pytest.approx(0.3)
    assert m.iou((0, 0, 2, 2), (0, 0, 2, 2)) == 1.0
    assert m.CHECKPOINT_SHA["rtmpose_tiny"].startswith("77ffc7e8") and m.HISTORICAL_COMMIT == "01c0dc3"
