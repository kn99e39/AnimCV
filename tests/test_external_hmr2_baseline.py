"""Worklog 97: external HMR 2.0b baseline adapter (population, box, canonical mapping, GT isolation, metrics)."""

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent


def _script(name):
    sys.path.insert(0, str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def hmr():
    return _script("run_external_hmr2_baseline")


def test_checkpoint_and_asset_identities_are_pinned(hmr):
    P = hmr.PREDECLARED["model"]
    assert P["commit"] == "efe18deff163b29dff87ddbd575fa29b716a356c"
    assert P["checkpoint_sha256"] == "c9ea09c834161468638c9de7a782c614e300c622709bf52967274c543ec72fcf"
    assert "98e65c74ad9b998783132f00880d1025a8d64b158e040e6ef13a557e5098bc42" in P["smpl"]
    docker = (ROOT / "scripts/w97_hmr2.Dockerfile").read_text()
    assert "FROM animcv-external:cu124" in docker and "chumpy@580566eafc9ac68b2614b64d6f7aaa84eebb70da" in docker


def test_box_uses_only_valid_detections_and_fixed_factor(hmr):
    xy = np.zeros((17, 3))
    xy[:, 0] = np.linspace(0.4, 0.6, 17)
    xy[:, 1] = np.linspace(0.2, 0.8, 17)
    valid = np.ones(17, bool)
    valid[0] = False
    xy[0, :2] = (0.0, 0.0)  # an invalid detection far away must not enlarge the box
    box = hmr.person_box(xy, valid, (1000, 2000))
    lo = xy[1:, :2].min(0) * [1000, 2000]
    hi = xy[1:, :2].max(0) * [1000, 2000]
    c, half = (lo + hi) / 2, (hi - lo) / 2 * 1.2
    assert np.allclose(box, [c[0] - half[0], c[1] - half[1], c[0] + half[0], c[1] + half[1]])
    v = np.zeros(17, bool)
    v[:5] = True
    assert hmr.person_box(xy, v, (1000, 2000)) is None  # fewer than 6 valid joints -> no box


def test_canonical_mapping_axes_and_joint_map(hmr):
    from pose.three_dpw_adapter import _SMPL_TO_CANONICAL
    names = ("pelvis", "left_hip", "left_knee", "left_ankle", "right_hip", "right_knee", "right_ankle", "spine", "thorax", "neck",
             "head", "left_shoulder", "left_elbow", "left_wrist", "right_shoulder", "right_elbow", "right_wrist")
    assert tuple(_SMPL_TO_CANONICAL[n] for n in names) == hmr.SMPL24_TO_17  # identical to the 3DPW GT adapter
    j = np.zeros((24, 3))
    j[15] = (0.0, -0.6, 0.0)  # head above the pelvis in OpenCV axes (y down)
    j[1], j[2] = (0.1, 0.0, 0.0), (-0.1, 0.0, 0.0)  # SMPL left hip at +x (subject facing the camera)
    j[12] = (0.0, 0.0, -0.2)  # neck nearer to the camera (z forward)
    c = hmr.smpl24_to_canonical17(j)
    assert np.allclose(c[0], 0) and c[10, 2] == pytest.approx(0.6)  # head up (+Z)
    assert c[9, 1] == pytest.approx(-0.2)  # nearer the camera = -Y depth
    assert c[1, 0] == pytest.approx(0.1)  # left hip on +X (image right) for a subject facing the camera
    R = hmr.CV_TO_CANON
    assert np.allclose(R @ R.T, np.eye(3)) and np.linalg.det(R) == pytest.approx(1.0)


def test_inference_is_gt_free_and_single_frame(hmr):
    text = (ROOT / "scripts/run_external_hmr2_baseline.py").read_text()
    body = text.split("def cmd_infer", 1)[1].split("def main", 1)[0]
    for forbidden in ("target_3d", "oracle", "smpl_v1", "R_smpl", "joints3", "gt_"):
        assert forbidden not in body
    assert "ViTDetDataset(cfg, img, np.array([r[\"box\"]]" in body and "load_hmr2(DEFAULT_CHECKPOINT)" in body
    pop = text.split("def cmd_population", 1)[1].split("def cmd_infer", 1)[0]
    assert "target_3d" not in pop and 'input_2d' in pop and 'input_valid' in pop


def test_source_metrics_and_material_rule():
    ev = _script("eval_external_hmr2_baseline")
    rng = np.random.default_rng(0)
    G = rng.normal(size=(17, 3)) * 0.3
    m = ev.source_metrics(G.copy(), G)
    assert m["mpjpe"] == pytest.approx(0.0, abs=1e-9) and m["dir_thigh_l"] == pytest.approx(0.0, abs=1e-6)
    P = G * 1.1 + 0.05
    assert ev.source_metrics(P, G)["pa_mpjpe"] == pytest.approx(0.0, abs=1e-6)
    good = {"mpjpe": {"ci95": [-9.0, -2.0], "relative": -0.2}, "dir_thigh": {"ci95": [-3.0, -1.0], "relative": -0.15},
            "dir_shin": {"ci95": [-1.0, 0.5], "relative": -0.2}, "knee_plane": {"ci95": [-5.0, -1.0], "relative": -0.05}}
    assert ev.material(good) == {"mpjpe": True, "dir_thigh": True, "dir_shin": False, "knee_plane": False}
    R = ev.pelvis_from_global_orient(np.diag([1.0, -1.0, -1.0]))  # SMPL upright, facing the camera (OpenCV)
    assert np.allclose(R[:, 2], (0, 0, 1)) and np.allclose(R[:, 1], (0, -1, 0))


def test_frozen_inputs_and_no_production_wiring():
    text = (ROOT / "scripts/eval_external_hmr2_baseline.py").read_text()
    assert 'raise SystemExit(f"STOP: frozen input changed' in text and "W87_PREDICTIONS" in text
    for path in sorted((ROOT / "src").rglob("*.py")):
        src = path.read_text()
        assert "run_external_hmr2_baseline" not in src and "eval_external_hmr2_baseline" not in src, path
