"""Worklog 97: HMR 2.0 access gate (read-only body-model / upstream / GPU checks)."""

import importlib.util
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def _gate():
    spec = importlib.util.spec_from_file_location("check_external_hmr2_access", ROOT / "scripts/check_external_hmr2_access.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _model(joints, pose_blend, shape_blend=10, hands=False):
    d = {"J_regressor": np.zeros((joints, 6890)), "kintree_table": np.zeros((2, joints)), "weights": np.zeros((6890, joints)),
         "shapedirs": np.zeros((6890, 3, shape_blend)), "posedirs": np.zeros((6890, 3, pose_blend)), "v_template": np.zeros((6890, 3)),
         "f": np.zeros((13776, 3))}
    if hands:
        d["hands_componentsl"] = np.zeros((45, 45))
    return d


def test_smpl_v1_is_accepted_and_smplh_renamed_is_rejected(tmp_path):
    g = _gate()
    v1 = tmp_path / "basicModel_neutral_lbs_10_207_0_v1.0.0.pkl"
    v1.write_bytes(pickle.dumps(_model(24, 207)))
    fake = tmp_path / "SMPL_NEUTRAL.pkl"  # the filename upstream checks, but SMPL+H inside
    fake.write_bytes(pickle.dumps(_model(52, 459, 16)))
    a, b = g.inspect_body_model(v1), g.inspect_body_model(fake)
    assert a["family"] == "SMPL" and a["verdict"] == "accepted_smpl_v1"
    assert b["family"] == "SMPL+H" and b["verdict"] == "rejected_smplplush"
    assert g.inspect_body_model(tmp_path / "absent.pkl")["verdict"] == "missing"


def test_symlinked_npz_and_chumpy_stubs(tmp_path):
    g = _gate()
    real = tmp_path / "SMPLH_NEUTRAL.npz"
    np.savez(real, **_model(52, 459, 16))
    link = tmp_path / "SMPL_NEUTRAL.npz"
    link.symlink_to(real)
    r = g.inspect_body_model(link)
    assert r["is_symlink"] and r["resolves_to"].endswith("SMPLH_NEUTRAL.npz") and r["family"] == "SMPL+H"
    # a legacy pickle whose arrays are chumpy objects is read structurally without chumpy installed
    import types

    mod = types.ModuleType("chumpy.ch")

    class Ch:
        def __init__(self, x):
            self.x = x

    Ch.__module__, Ch.__qualname__ = "chumpy.ch", "Ch"
    mod.Ch = Ch
    sys.modules["chumpy"], sys.modules["chumpy.ch"] = types.ModuleType("chumpy"), mod
    try:
        d = _model(24, 207)
        d["shapedirs"], d["J_regressor"] = Ch(d["shapedirs"]), Ch(d["J_regressor"])
        raw = pickle.dumps(d)
    finally:
        del sys.modules["chumpy"], sys.modules["chumpy.ch"]
    assert b"chumpy.ch" in raw
    p = tmp_path / "legacy.pkl"
    p.write_bytes(raw)
    r = g.inspect_body_model(p)
    assert r["structure"]["joints"] == 24 and r["structure"]["shape_blend"] == 10 and r["verdict"] == "accepted_smpl_v1"


def test_gpu_parsing_marks_foreign_use_busy():
    g = _gate()
    s = g.gpu_status("7450, 12288, 82\n")
    assert s["available"] and s["gpus"][0]["busy"] and s["gpus"][0]["memory_free_mib"] == 4838.0
    assert not g.gpu_status("3, 12288, 0\n")["gpus"][0]["busy"]


def test_gate_is_read_only_and_unwired():
    text = (ROOT / "scripts/check_external_hmr2_access.py").read_text()
    for forbidden in ("urlretrieve", "requests.", "wget", "curl", "download_models", "torch.load", "load_hmr2("):
        assert forbidden not in text
    for path in sorted((ROOT / "src").rglob("*.py")):
        assert "check_external_hmr2_access" not in path.read_text(), path
