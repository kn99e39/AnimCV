"""W107 FBX round-trip failure ownership: contract, metric, CASE and stored-output tests (no Blender required)."""
from __future__ import annotations

import ast
import hashlib
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT/"scripts"))
import w107_fbx_contract as C  # noqa: E402

BASE = Path(os.environ.get("W107_BASE", "C:/Users/dna10/animcv-w107"))
W106 = Path(os.environ.get("W106_BASE", "C:/Users/dna10/animcv-w106"))
DECL = BASE/"results/declare/declaration.json"
ANALYSIS = BASE/"results/analysis/analysis_report.json"


# --- exporter freeze / one-factor variants -------------------------------------------------------------------------
def test_v0_kwargs_parsed_from_unchanged_exporter():
    kw = C.export_fbx_kwargs((ROOT/"src/blender/export.py").read_text(encoding="utf-8"))
    assert kw == {"use_selection": False, "object_types": {"ARMATURE"}, "bake_anim": True}


def test_variants_change_exactly_one_property():
    v0 = {"use_selection": False, "object_types": ["ARMATURE"], "bake_anim": True}
    assert C.variant_kwargs(v0, "V0d") == v0
    for v in C.CAUSAL_VARIANTS:
        kw = C.variant_kwargs(v0, v)
        diff = {k for k in kw if kw[k] != v0.get(k, object())}
        assert diff == set(C.VARIANTS[v]["override"]) and len(diff) == 1
    defaults = {"bake_anim_simplify_factor": 1.0, "bake_anim_use_all_actions": True}
    assert all(r["ok"] for r in C.one_factor_ok(v0, defaults).values())
    # a null intervention (control value == default) or unsupported name is refused
    assert not C.one_factor_ok(v0, {"bake_anim_simplify_factor": 0.0, "bake_anim_use_all_actions": True})["V1"]["ok"]
    assert not C.one_factor_ok(v0, {"bake_anim_use_all_actions": True})["V1"]["ok"]


def test_variant_cannot_override_production_argument():
    with pytest.raises(ValueError):
        old = C.VARIANTS["V1"]["override"]
        C.VARIANTS["V1"]["override"] = {"bake_anim": False}
        try:
            C.variant_kwargs({"bake_anim": True}, "V1")
        finally:
            C.VARIANTS["V1"]["override"] = old


def test_exporter_never_patched_by_w107_scripts():
    for name in ("w107_fbx_blender.py", "w107_fbx_ownership.py", "w107_fbx_contract.py"):
        src = (ROOT/"scripts"/name).read_text(encoding="utf-8")
        tree = ast.parse(src)
        for n in ast.walk(tree):
            if isinstance(n, ast.Assign):
                for t in n.targets:
                    assert not (isinstance(t, ast.Attribute) and t.attr in ("export_fbx", "fbx")), name
        assert "save_mainfile(" not in src.replace("save_as_mainfile(", "")
        assert "import torch" not in src
    bl = (ROOT/"scripts/w107_fbx_blender.py").read_text(encoding="utf-8")
    assert "from blender.export import export_fbx" in bl


# --- V3 world-equivalent construction (pure math) -----------------------------------------------------------------
def _qrot(q, v):
    w, x, y, z = q
    p = C.qmul(C.qmul(q, (0.0, *v)), (w, -x, -y, -z))
    return np.array(p[1:])


def test_v3_static_and_animated_cells_are_world_equivalent_mathematically():
    rest_q = C.qaxis_angle((1, 0, 0), -90.0)   # bone pointing along +Z: rest maps local Y to armature Z
    for f in (1.0, 17.0, 49.0):
        g = C.v3_global_rotation(f)
        qb = C.qmul(C.qmul((rest_q[0], -rest_q[1], -rest_q[2], -rest_q[3]), g), rest_q)
        static = C.qmul(rest_q, qb)            # armature-space base orientation: Rrest * qb
        animated = C.qmul(g, rest_q)           # object G * Rrest
        for v in ((1, 0, 0), (0, 1, 0), (0.3, -0.2, 0.9)):
            assert np.allclose(_qrot(static, v), _qrot(animated, v), atol=1e-12)
        for s in (1.0, 0.01):                 # uniform scale commutes with rotation
            assert np.allclose(_qrot(g, np.array([0.3, 0, 0.6])/s)*s, _qrot(g, [0.3, 0, 0.6]), atol=1e-12)


def test_v3_design_is_fixed_2x2():
    assert set(C.V3_CELLS) == {"static_s1", "static_s001", "animated_s1", "animated_s001"}
    assert {(C.V3_ANIMATED[c], C.V3_SCALE[c]) for c in C.V3_CELLS} == {(a, s) for a in (False, True) for s in (1.0, 0.01)}
    assert C.V3_REFERENCE == "static_s1" and C.V3_FRAMES == [float(f) for f in range(1, 50)]
    q = C.v3_global_rotation(5.0)
    assert math.isclose(sum(x*x for x in q), 1.0, abs_tol=1e-12)


# --- metrics -------------------------------------------------------------------------------------------------------
def _fake(T=5, B=3, seed=0):
    rng = np.random.default_rng(seed)
    times = np.array(sorted([1.0 + k for k in range(T)] + [1.5 + k for k in range(T-1)]))
    n = len(times)
    om = np.tile(np.diag([0.01, 0.01, 0.01, 1.0]), (n, 1, 1))
    q = np.tile([1.0, 0, 0, 0], (n, B, 1))
    return {"times": times, "bones": np.array([f"b{i}" for i in range(B)]), "world_head": rng.normal(size=(n, B, 3)),
            "world_tail": rng.normal(size=(n, B, 3)), "world_rot": np.tile(np.eye(3), (n, B, 1, 1)),
            "object_matrix": om, "local_quat": q, "local_loc": np.zeros((n, B, 3)), "local_scale": np.ones((n, B, 3))}


def test_parity_gate_metric_is_w106_max_abs_coordinate_on_integer_frames():
    a = _fake()
    b = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in a.items()}
    b["bones"] = np.array(list(a["bones"]) + ["extra_end"])
    for k in ("world_head", "world_tail", "world_rot", "local_quat", "local_loc", "local_scale"):
        pad = np.zeros_like(a[k][:, :1])
        b[k] = np.concatenate([a[k], pad], axis=1)
    b["world_tail"][2, 1, 0] += 0.004     # times[2] == 2.0 (integer)
    b["world_head"][1, 0, 2] -= 0.9       # times[1] == 1.5 (half frame): excluded from the gate
    p = C.parity(a, b)
    assert p["gate_max_abs_coord_m"] == pytest.approx(0.004)
    assert p["half_frame_max_abs_coord_m"] == pytest.approx(0.9)
    assert p["frames_gt_1mm"] == 1 and p["worst_bones"][0][0] == "b1"
    assert p["bones_compared"] == 3 and p["missing_in_import"] == []
    assert C.collateral(p["per_bone_max_abs_coord_m"], {"b0": 0.0, "b1": 0.0041, "b2": 0.0002}) == ["b2"]


def test_arrays_max_diff_detects_identity_and_mismatch():
    a = _fake()
    assert C.arrays_max_diff(a, a) == 0.0
    b = dict(a, bones=np.array(["x", "y", "z"]))
    assert C.arrays_max_diff(a, b) == math.inf


# --- causal rule and CASE mapping ----------------------------------------------------------------------------------
def _per(vals, collateral=(), rep=True, ident=True):
    return {c: {"v0": v0, "vf": vf, "collateral": list(collateral), "repeat_ok": rep, "action_identity_ok": ident}
            for c, (v0, vf) in vals.items()}


def test_factor_support_rule():
    s = C.factor_support(_per({"a": (1.3e-3, 2e-4), "b": (6.1e-3, 1.5e-3), "c": (7.5e-3, 3e-3)}))
    assert s["supported"] and s["any_1mm"] and not s["all_1mm"]
    s = C.factor_support(_per({"a": (1.3e-3, 2e-4), "b": (6.1e-3, 5e-3), "c": (7.5e-3, 3e-3)}))
    assert not s["supported"] and s["mixed"]
    s = C.factor_support(_per({"a": (1.3e-3, 1.29e-3), "b": (6.1e-3, 6.1e-3), "c": (7.5e-3, 7.5e-3)}))
    assert not s["supported"] and not s["mixed"]
    s = C.factor_support(_per({"a": (1.3e-3, 2e-4), "b": (6.1e-3, 1e-3), "c": (7.5e-3, 1e-3)}, collateral=["x"]))
    assert not s["supported"] and s["mixed"]
    s = C.factor_support(_per({"a": (1.3e-3, 2e-4), "b": (6.1e-3, 2e-3), "c": (7.5e-3, 3e-3)}, rep=False))
    assert not s["supported"]


def _pre(**kw):
    base = {"preflight_ok": True, "source_fidelity_ok": True, "v0_reproduces_w106": True, "v0_deterministic": True,
            "v0d_equals_v0": True, "actions_unmodified": True, "originals_unchanged": True, "synthetic_control_ok": True,
            "v3_status": "TESTED", "variants_testable": {"V1": True, "V2": True},
            "factors": {"V1": {"supported": False, "mixed": False}, "V2": {"supported": False, "mixed": False}}}
    base.update(kw)
    return base


def test_case_mapping():
    assert C.decide_case(_pre())[0] == "B"
    assert C.decide_case(_pre(factors={"V1": {"supported": True, "mixed": False},
                                       "V2": {"supported": False, "mixed": False}}))[0] == "A"
    assert C.decide_case(_pre(factors={"V1": {"supported": False, "mixed": True},
                                       "V2": {"supported": False, "mixed": False}}))[0] == "C"
    assert C.decide_case(_pre(v3_status="NOT_TESTABLE"))[0] == "C"
    assert C.decide_case(_pre(v3_status="NOT_TESTABLE", factors={"V1": {"supported": True, "mixed": False},
                                                                 "V2": {"supported": False, "mixed": False}}))[0] == "C"
    for k in ("preflight_ok", "source_fidelity_ok", "v0_reproduces_w106", "v0_deterministic", "v0d_equals_v0",
              "actions_unmodified", "originals_unchanged"):
        assert C.decide_case(_pre(**{k: False}, factors={"V1": {"supported": True, "mixed": False},
                                                         "V2": {"supported": True, "mixed": False}}))[0] == "D"


def test_fixed_cases_and_actions():
    assert C.REAL_CASES == ("complete_control", "arm_natural_most", "real_clip")
    assert all(a.endswith(" FIXED") for a in C.REAL_ACTIONS.values())
    assert C.TOL["fbx_rt_parity_m"] == 1e-3


# --- stored outputs (skipped before the run) -----------------------------------------------------------------------
needs_decl = pytest.mark.skipif(not DECL.exists(), reason="declaration not written yet")
needs_analysis = pytest.mark.skipif(not ANALYSIS.exists(), reason="analysis not written yet")


@needs_decl
def test_stored_declaration_frozen_before_outcomes():
    d = json.loads(DECL.read_text())
    s = hashlib.sha256(DECL.read_bytes()).hexdigest()
    assert (DECL.parent/"declaration.sha256").read_text().strip() == s
    assert d["outcomes_read"] is False
    assert d["variant_order"] == list(C.VARIANT_ORDER)
    assert d["exporter"]["v0_kwargs"] == {"use_selection": False, "object_types": ["ARMATURE"], "bake_anim": True}
    assert set(d["cases"]) == set(C.REAL_CASES) | {C.W106_SYNTHETIC}


@needs_analysis
def test_stored_analysis_consistent():
    d = json.loads(DECL.read_text())
    a = json.loads(ANALYSIS.read_text())
    assert a["declaration_sha256"] == hashlib.sha256(DECL.read_bytes()).hexdigest()
    assert a["CASE"] in "ABCD"
    assert C.decide_case(a["decision_inputs"])[0] == a["CASE"]
    for c in C.REAL_CASES:
        assert a["cases"][c]["w106_v0_recomputed_gate_m"] == pytest.approx(
            {"complete_control": 1.29e-3, "arm_natural_most": 6.10e-3, "real_clip": 7.51e-3}[c], abs=5e-6)
    assert set(a["cases"][C.REAL_CASES[0]]["variants"]) == set(d["variant_order_testable"])
    assert a["originals_unchanged"]["ok"]
