"""Focused tests for the EXPERIMENTAL (Batch 132) CondMDI causal gate (pure numpy; SYNTHETIC tensors; no model, no Blender).

Real-model evidence (replay, flag isolation from runtime instrumentation, leak control, official-mask parity) and real
Blender evidence are produced by ``scripts/w132_condmdi_causal_harness.py``, not here.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import app.experimental_condmdi as C  # noqa: E402
import app.experimental_condmdi_causal as K  # noqa: E402


# ------------------------------------------------------------------------------------------- official masks ---
def test_middle_frames_match_official_rule():
    obs, hid = K.official_middle_frames(120, 19)
    assert hid == list(range(50, 69)) and len(hid) == 19
    assert obs == list(range(0, 50)) + list(range(69, 120))
    assert K.official_middle_frames(160, 19)[1] == list(range(70, 89))


def test_pos_mask_is_subset_and_differs_only_by_rot_vel_and_yaw():
    prv = K.official_middle_mask(120, 19, "pos_rot_vel")
    pos = K.official_middle_mask(120, 19, "pos")
    assert prv.shape == pos.shape == (263, 120)
    assert np.all(pos <= prv)
    groups = {C.feature_owner(int(f))[0] for f in np.nonzero((prv ^ pos).any(1))[0]}
    assert groups == {"root_yaw", "rot", "vel"}
    assert not prv[:, 50:69].any() and not pos[:, 50:69].any()           # nothing observed in the hidden interval
    # official semantics: pelvis pos = root xz + height; contact always included
    assert pos[1:4, 0].all() and not pos[0, 0] and prv[0, 0]
    assert pos[259:263, 0].all() and prv[259:263, 0].all()
    census = K.mask_group_census(pos)
    assert census["contact"] == 4 * 101 and census["root_xz"] == 2 * 101 and census["rot"] == 0 and census["vel"] == 0


def test_feature_mask_rejects_unknown_mode():
    with pytest.raises(ValueError):
        K.official_feature_mask(np.ones((22, 4), bool), "rot")


# ------------------------------------------------------------------------------------------- model-space metrics ---
def _abs(T=160, seed=0):
    rng = np.random.default_rng(seed)
    a = np.zeros((T, 263))
    a[:, 3] = 0.9
    a[:, 4:67] = rng.normal(0, 0.01, (T, 63)) + np.tile(np.linspace(-0.5, 0.5, 63), (T, 1))
    return a


def _set_wrist(a, k, xyz):
    a[k, 4 + 3 * 19: 4 + 3 * 20] = xyz


def test_s1_boundary_internal_vs_effective():
    inp = _abs()
    out = inp.copy()
    _set_wrist(inp, 113, [0.2, 1.6, 0.1])
    _set_wrist(out, 113, [0.2, 1.7, 0.1])                                 # context error 0.1 m
    _set_wrist(out, 114, [0.2, 1.9, 0.1])                                 # internal 0.2, effective 0.3
    b = K.s1_boundary(out, inp, None)["left_wrist"]["ric"]
    assert b["internal_entry_m"] == pytest.approx(0.2)
    assert b["effective_entry_m"] == pytest.approx(0.3)
    assert b["context_err_113_m"] == pytest.approx(0.1)


def test_observed_residuals_and_context_gate():
    inp = _abs()
    mask = C.observation_mask(160, list(range(114, 133)))
    out = inp.copy()
    r = K.observed_residuals(out, inp, mask)
    assert r["max_abs_feature_err"] == 0.0 and r["arm_window_ric_err_max_m"] == 0.0
    assert K.context_gate(r)["pass"]
    _set_wrist(out, 110, out[110, 61:64] + [0.0, 0.06, 0.0])               # 60 mm at an observed boundary-window sample
    r = K.observed_residuals(out, inp, mask)
    assert r["arm_window_ric_err_max_m"] == pytest.approx(0.06)
    assert not K.context_gate(r)["pass"]
    out2 = inp.copy()
    _set_wrist(out2, 120, out2[120, 61:64] + [0.0, 0.5, 0.0])              # interior (masked) change is not a context error
    assert K.observed_residuals(out2, inp, mask)["max_abs_feature_err"] == 0.0
    out3 = inp.copy()
    out3[:, 0] += np.radians(6.0)                                          # root yaw error > 5 deg
    assert not K.context_gate(K.observed_residuals(out3, inp, mask))["pass"]


def test_refctl_metrics_identity_and_boundary_excess():
    ref = _abs(120, 1)
    mask = K.official_middle_mask(120, 19, "pos_rot_vel")
    m = K.refctl_metrics(ref, ref, mask, range(50, 69))
    assert m["ric"]["hidden_mpjpe_mm"] == 0.0 and m["ric"]["boundary_max_excess_mm"] <= 1e-9
    out = ref.copy()
    out[50:69, 4:67] += 0.1                                                # hidden block shifted -> boundary jump
    m = K.refctl_metrics(out, ref, mask, range(50, 69))
    assert m["ric"]["boundary_max_excess_mm"] > 50 and m["observed_feature_residual_max"] == 0.0
    cells = {c: m for c in ("i", "ii", "iii", "iv")}
    rd = K.refctl_readings(cells)
    assert all(rd["discontinuity_present"].values()) and rd["replacement_effect_pos"] == "no class change"


# ------------------------------------------------------------------------------------------- flag isolation ---
def _log(impute, n=100):
    h = lambda s: hashlib.sha256(s.encode()).hexdigest()                  # noqa: E731
    base = {"mask", "lengths", "text", "text_tokens", "diffusion_steps", "text_scale", "reconstruction_guidance"}
    xs = [h("x0"), h("x0")] + [h(f"x{i}{'on' if impute else 'off'}") for i in range(2, 2 * n)]
    return {"request": {"seed": 42, "text": "t", "length": 160, "guidance": 2.5, "respacing": "ddim100", "impute": impute},
            "n_net_calls": 2 * n, "obs_x0_sha": [h("obs")], "obs_mask_sha": [h("mask")], "text_tokens_sha": [h("tok")],
            "y_keys": sorted(base | (K.IMPUTE_KEYS if impute else set())), "x_sha": xs, "out_sha": [h("o0"), h("o1"), h("o2" + str(impute))],
            "requires_imputation": [impute] * n, "requires_reconstruction_guidance": [False] * n}


def _io():
    z = np.zeros((3, 3))
    return {"input_relative": z, "input_absolute": z, "observation_mask_abs": z.astype(bool), "effective_conditioning_normalized": z}


def test_flag_isolation_passes_only_for_pure_replacement_switch():
    ok, c = K.flag_isolation(_log(True), _log(False), _io(), _io(), 100)
    assert ok and c["first_divergent_net_call"] == 2
    bad = _log(False)
    bad["y_keys"] = bad["y_keys"] + ["reconstruction_weight"]
    assert not K.flag_isolation(_log(True), bad, _io(), _io(), 100)[0]
    bad = _log(False)
    bad["obs_x0_sha"] = ["other"]
    assert not K.flag_isolation(_log(True), bad, _io(), _io(), 100)[0]
    bad = _log(True)
    bad["requires_imputation"][5] = False
    assert not K.flag_isolation(bad, _log(False), _io(), _io(), 100)[0]
    bad = _log(False)
    bad["x_sha"][0] = "diverged"
    assert not K.flag_isolation(_log(True), bad, _io(), _io(), 100)[0]
    io2 = _io()
    io2["effective_conditioning_normalized"] = np.ones((3, 3))
    assert not K.flag_isolation(_log(True), _log(False), _io(), io2, 100)[0]
    bad = _log(False)
    bad["request"]["seed"] = 43
    assert not K.flag_isolation(_log(True), bad, _io(), _io(), 100)[0]


# ------------------------------------------------------------------------------------------- frozen rules ---
def _b(ie, ix, ee, ex):
    return {"left_wrist": {"ric": {"internal_entry_m": ie, "internal_exit_m": ix, "effective_entry_m": ee, "effective_exit_m": ex}}}


def test_mechanism_classes():
    on = _b(0.3, 0.17, 0.3, 0.17)
    assert K.mechanism(on, _b(0.25, 0.1, 0.25, 0.1), True)[0] == "PERSISTS_UNDER_LEARNED_CONDITIONING"
    assert K.mechanism(on, _b(0.02, 0.02, 0.2, 0.02), True)[0] == "OFF_CONTINUOUS_ONLY_BY_DEPARTING_FROM_CONTEXT"
    assert K.mechanism(on, _b(0.02, 0.02, 0.03, 0.02), False)[0] == "OFF_CONTINUOUS_ONLY_BY_DEPARTING_FROM_CONTEXT"
    assert K.mechanism(on, _b(0.02, 0.02, 0.03, 0.02), True)[0] == "REPLACEMENT_IMPUTATION_PRIMARY_SUPPORTED"


def _q(entry, exitv, jerk, dev, flicks=0, dip_pass=True):
    return {"arm_local": {"n_reversal_flicks": flicks, "boundary_c0": {"entry_step_mm": entry, "exit_step_mm": exitv}, "rms_jerk_mps3": jerk},
            "near_straight_frames": [], "bend_direction_flips": [], "hinge_opposite_frames": [], "wrist_dev_from_A_mean_mm": dev,
            "press_intent": {"pass": dip_pass}}


def test_candidate_qa_and_case_precedence():
    q = {"B": _q(24, 25, 2564, 148), "C_ON": _q(429, 153, 7338, 292), "C_OFF": _q(30, 30, 3000, 120)}
    rv = {"V1_flick": ["C_ON"], "V2_most_natural": "C_OFF", "V3_wrong_elbow": [], "V4_rank_most_to_least_faithful": ["C_OFF", "B", "A", "C_ON"],
          "V6_upward_press": {"C_OFF": "yes"}}
    qa = K.candidate_qa(q, rv)
    assert all(qa.values())
    assert K.case_rule(False, True, False, qa) == "A"
    q["C_OFF"] = _q(30, 30, 3000, 160)                                   # fidelity worse than B
    qa = K.candidate_qa(q, rv)
    assert not qa["QA4"] and K.case_rule(False, True, False, qa) == "B"
    assert not K.candidate_qa(q, None)["QA6"]
    assert K.case_rule(True, False, True, qa) == "DATA_BLOCKED"
    assert K.case_rule(False, False, True, qa) == "D"
    assert K.case_rule(False, True, True, None) == "C"


# ------------------------------------------------------------------------------------------- declaration guard ---
def _harness():
    spec = importlib.util.spec_from_file_location("w132_harness", str(ROOT / "scripts" / "w132_condmdi_causal_harness.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_declaration_and_instrument_tamper_refused(tmp_path):
    h = _harness()
    f = "src/app/experimental_condmdi_causal.py"
    decl = tmp_path / "declaration.json"
    decl.write_text(json.dumps({"instrument_sha256_lf": {f: h.lf_sha(f)}}))
    sha = hashlib.sha256(decl.read_bytes()).hexdigest()
    assert h.check_declaration(argparse.Namespace(declaration=str(decl), declaration_sha=sha))
    with pytest.raises(SystemExit, match="BLOCKED_DECLARATION_CHANGED"):
        h.check_declaration(argparse.Namespace(declaration=str(decl), declaration_sha="0" * 64))
    decl.write_text(json.dumps({"instrument_sha256_lf": {f: "0" * 64}}))
    sha = hashlib.sha256(decl.read_bytes()).hexdigest()
    with pytest.raises(SystemExit, match="BLOCKED_INSTRUMENT_CHANGED"):
        h.check_declaration(argparse.Namespace(declaration=str(decl), declaration_sha=sha))
