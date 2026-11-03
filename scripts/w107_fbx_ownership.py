#!/usr/bin/env python3
"""W107 host-side stages: FBX Action round-trip failure ownership (evidence only).

  verify    W106 COMPLETE marker, W106 evidence-manifest identity of every reused W106 artifact, frozen exporter
            source (SHA vs W106 + AST-parsed V0 arguments), worker lock, Git state; byte-verified scratch copies of the
            W106 N0 .blend files into <base>/inputs
  declare   frozen cases, variant order, exact operator properties (runtime names/defaults from the ``options``
            stage), metrics, samples, tolerances, negative controls, hypothesis map, CASE rule, stop conditions and
            resource cap -> declaration.json + SHA-256 (``outcomes_read: false``)
  analyze   V0 reproduction, source fidelity, diagnostic-path equality, determinism, per-case/per-variant parity,
            causal-support rule, synthetic controls, V3 2x2 -> analysis_report.json and the CASE
  manifest  hashes of every output, plus a re-check that no W106 original changed

No production exporter/importer, rig, Action original or scene setting is modified.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT/"scripts"))
import w107_fbx_contract as C  # noqa: E402

sha256 = C.sha256_file
W106_MANIFEST_SHA = "f3003fe54e68c6cf182313c0b784468275b7ed039f50f574ad10e64eb8fb97db"
W106_DECLARATION_SHA = "a72f6dc57fc305e808b4fb92a5a449fddacb96c96b643ac6d4170fb1f516714d"
W106_WORKLOG = "docs/106_WORKLOG_EXISTING_RIG_ACTION_EDITABILITY_AND_ROUNDTRIP.md"
W106_DRIFT_MM = {"complete_control": 1.29, "arm_natural_most": 6.10, "real_clip": 7.51}   # as published
EXPORTER = "src/blender/export.py"
CASE_DIRS = {"complete_control": "complete_control", "arm_natural_most": "arm_natural_most",
             "real_clip": "real_clip", C.W106_SYNTHETIC: "synthetic/case"}
W106_PER_CASE = ["n0.blend", "n0_reopen.npz", "n0_reopen.json", "fbx_n0.npz", "fbx_n0.json", "fbx_n0_meta.json",
                 "fbx/n0.fbx"]
W106_GLOBAL = ["results/declare/declaration.json", "results/analysis/analysis_report.json",
               "results/verify/verify_report.json", "results/fbx_attribution/fbx_attribution.json"]
SHARED_PATHS = ["src", "tests", "scripts", "examples", "docs", "Architecture_v2.md", "Architecture_v3_FramePose.md"]


def wjson(path, obj):
    Path(path).write_text(json.dumps(obj, indent=1))
    return sha256(path)


def w106_used():
    out = [f"results/probe/{d}/{f}" for d in CASE_DIRS.values() for f in W106_PER_CASE]
    return out + W106_GLOBAL


def git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()


# --- verify -------------------------------------------------------------------------------------------------------
def verify(o):
    base = Path(o.base)
    out = base/"results"/"verify"
    out.mkdir(parents=True, exist_ok=False)
    w106 = Path(o.w106)
    rep = {"schema": C.SCHEMA + ":verify", "checks": {}, "stop": []}
    wl = (ROOT/W106_WORKLOG).read_text(encoding="utf-8")
    rep["checks"]["w106_complete_marker"] = wl.rstrip().endswith("AUTOMATION_BATCH_STATUS: COMPLETE\nBATCH_NUMBER: 106")
    msha = sha256(w106/"evidence_manifest.json")
    man = json.loads((w106/"evidence_manifest.json").read_text())
    rep["checks"]["w106_manifest_sha"] = {"value": msha, "ok": msha == W106_MANIFEST_SHA}
    rep["checks"]["w106_manifest_originals_unchanged"] = bool(man.get("all_originals_unchanged"))
    rep["checks"]["w106_manifest_declaration"] = man.get("declaration_sha256") == W106_DECLARATION_SHA
    files = {}
    for rel in w106_used():
        p = w106/rel
        rec = {"exists": p.exists(), "manifest_sha256": man["files"].get(rel, {}).get("sha256")}
        rec["sha256"] = sha256(p) if p.exists() else None
        rec["ok"] = rec["exists"] and rec["sha256"] == rec["manifest_sha256"]
        files[rel] = rec
    rep["w106_files"] = files
    # frozen exporter: SHA equals W106's recorded value; V0 arguments parsed from its source
    w106_ver = json.loads((w106/"results/verify/verify_report.json").read_text())
    exp_sha = sha256(ROOT/EXPORTER)
    rep["checks"]["exporter_sha_equals_w106"] = {"value": exp_sha, "w106": w106_ver["rig_assets"][EXPORTER],
                                                 "ok": exp_sha == w106_ver["rig_assets"][EXPORTER]}
    rep["exporter_v0_kwargs"] = C.export_fbx_kwargs((ROOT/EXPORTER).read_text(encoding="utf-8"))
    rep["exporter_v0_kwargs"] = {k: (sorted(v) if isinstance(v, set) else v) for k, v in rep["exporter_v0_kwargs"].items()}
    # repository and worker state
    head = git("rev-parse", "HEAD")
    status = git("status", "--porcelain").splitlines()
    tracked_changes = [s for s in status if not s.startswith("??")]
    untracked = [s[3:] for s in status if s.startswith("??")]
    allowed = all(u.startswith(("scripts/w107_", "tests/test_w107_")) for u in untracked)
    rep["git"] = {"head": head, "branch": git("branch", "--show-current"), "tracked_changes": tracked_changes,
                  "untracked": untracked}
    rep["checks"]["starting_head"] = {"value": head, "ok": head == o.starting_head}
    rep["checks"]["working_tree_only_new_w107_files"] = not tracked_changes and allowed
    lock = Path(o.lock)
    lk = json.loads(lock.read_text()) if lock.exists() else {}
    rep["worker_lock"] = lk
    rep["checks"]["worker_lock_batch_107_running"] = lk.get("batch") == 107 and lk.get("state") == "RUNNING"
    rep["checks"]["blender_present"] = Path(o.blender).exists()
    for k, v in rep["checks"].items():
        if not (v if isinstance(v, bool) else v["ok"]):
            rep["stop"].append(k)
    rep["stop"] += [f"w106:{r}" for r, v in files.items() if not v["ok"]]
    # byte-verified scratch copies of the W106 N0 .blend (active Action already the declared M_Main Action)
    inputs = base/"results"/"inputs"
    inputs.mkdir(exist_ok=False)
    copies = {}
    for cid, d in CASE_DIRS.items():
        srcp = w106/"results/probe"/d/"n0.blend"
        dst = inputs/f"{cid}_n0.blend"
        shutil.copyfile(srcp, dst)
        copies[cid] = {"source": str(srcp), "path": str(dst), "sha256": sha256(dst),
                       "copy_identical": sha256(dst) == sha256(srcp)}
        if not copies[cid]["copy_identical"]:
            rep["stop"].append(f"copy:{cid}")
    rep["inputs"] = copies
    rep["ok"] = not rep["stop"]
    wjson(out/"verify_report.json", rep)
    print(json.dumps({"ok": rep["ok"], "stop": rep["stop"]}))
    return 0 if rep["ok"] else 2


# --- declare ------------------------------------------------------------------------------------------------------
def declare(o):
    base = Path(o.base)
    out = base/"results"/"declare"
    out.mkdir(parents=True, exist_ok=False)
    w106 = Path(o.w106)
    ver = json.loads(Path(o.verify).read_text())
    if not ver["ok"]:
        raise SystemExit("STOP: verify failed")
    opts = json.loads(Path(o.options).read_text())
    exp_props = opts["export_scene.fbx"]
    defaults = {k: v["default"] for k, v in exp_props.items()}
    v0 = ver["exporter_v0_kwargs"]
    missing = [k for k in v0 if k not in exp_props]
    if missing:
        raise SystemExit(f"STOP: export_fbx argument(s) {missing} not supported by this Blender")
    ofk = C.one_factor_ok(v0, defaults)
    testable = {"V0": True, "V0d": True} | {v: ofk[v]["ok"] for v in C.CAUSAL_VARIANTS}
    w106_decl = json.loads((w106/"results/declare/declaration.json").read_text())
    cases = {}
    for cid in list(C.REAL_CASES) + [C.W106_SYNTHETIC]:
        d = CASE_DIRS[cid]
        n0meta = json.loads((w106/"results/probe"/d/"n0_reopen.json").read_text())
        entry = {"id": cid, "kind": "real" if cid in C.REAL_CASES else "synthetic_w106 (separately labelled, never pooled)",
                 "input_copy": ver["inputs"][cid]["path"], "input_sha256": ver["inputs"][cid]["sha256"],
                 "w106_source": ver["inputs"][cid]["source"],
                 "action": C.REAL_ACTIONS.get(cid, C.W106_SYNTHETIC_ACTION),
                 "w106_n0_channel_digest": n0meta["channel_digest"], "w106_n0_timing": n0meta["timing"],
                 "w106_reference": {f: f"results/probe/{d}/{f}" for f in W106_PER_CASE}}
        if cid in C.REAL_CASES:
            entry["w106_declared_action_digest"] = w106_decl["cases"][cid]["action_value_digest"]
            entry["regime"] = w106_decl["cases"][cid].get("timebase", {}).get("regime")
            entry["timebase"] = w106_decl["cases"][cid].get("timebase")
        cases[cid] = entry
    decl = {
        "schema": C.SCHEMA + ":declaration", "outcomes_read": False,
        "note": "Frozen before any W107 variant export/import was executed. W106 V0 outcomes are historical inputs.",
        "starting_head": o.starting_head, "verify_sha256": sha256(o.verify), "options_sha256": sha256(o.options),
        "environment": {"blender_version": opts["blender_version"], "blender_build_hash": opts["blender_build_hash"],
                        "fbx_addon": {k: v for k, v in opts["fbx_addon"].items() if k != "source_sha256"},
                        "fbx_addon_source_sha256": opts["fbx_addon"].get("source_sha256"),
                        "blender_exe": o.blender, "launch": "-b --factory-startup -t 2 --python-exit-code 1"},
        "scripts_sha256": {p: sha256(ROOT/p) for p in ("scripts/w107_fbx_contract.py", "scripts/w107_fbx_blender.py",
                                                        "scripts/w107_fbx_ownership.py", "scripts/w106_action_blender.py",
                                                        "scripts/w106_action_contract.py", EXPORTER)},
        "cases": cases,
        "exporter": {"v0_function": "blender.export.export_fbx (unchanged, called, never patched)",
                     "v0_kwargs": v0, "runtime_defaults_of_relevant_properties":
                         {k: defaults.get(k) for k in sorted(set(v0) | {"bake_anim_simplify_factor", "bake_anim_use_all_actions",
                                                                       "bake_anim_step", "bake_anim_use_nla_strips",
                                                                       "bake_anim_force_startend_keying", "add_leaf_bones",
                                                                       "axis_forward", "axis_up", "apply_scale_options",
                                                                       "global_scale", "apply_unit_scale",
                                                                       "bake_space_transform", "primary_bone_axis",
                                                                       "secondary_bone_axis", "armature_nodetype",
                                                                       "bake_anim_use_all_bones"})},
                     "variants": C.VARIANTS, "one_factor_check": ofk,
                     "diagnostic_kwargs": {v: C.variant_kwargs(v0, v) for v in C.VARIANT_ORDER if v != "V0"}},
        "importer": C.IMPORTER, "importer_probe": C.IMPORTER_PROBE, "rotation_conversion": C.ROTATION_CONVERSION,
        "importer_runtime_defaults": {k: v["default"] for k, v in opts["import_scene.fbx"].items()},
        "variant_order": list(C.VARIANT_ORDER), "variants_testable": testable,
        "variant_order_testable": [v for v in C.VARIANT_ORDER if testable[v]],
        "v3": {"cells": list(C.V3_CELLS), "reference": C.V3_REFERENCE, "frames": C.V3_FRAMES, "bones": C.V3_BONES,
               "scale": C.V3_SCALE, "animated": C.V3_ANIMATED, "base_prerotation": C.V3_BASE_PREROT,
               "settings": "V0 production export_fbx + default importer only (no V1/V2 crossing)",
               "construction": "static cells key the base bone with Rrest^-1 G(t) Rrest (base head at the origin); "
                               "animated cells key the object rotation with G(t) and leave the base bone at rest; "
                               "scale-0.01 cells multiply bone geometry by 100; child local rotations identical",
               "equivalence_invariant": "Blender-evaluated world heads/tails of every bone in every cell equal the "
                                        "reference cell within v3_equivalence_m before export; otherwise V3 is NOT "
                                        "TESTABLE and is never used as evidence",
               "interpretation": "descriptive mechanism probe; never pooled with real Actions; not causal for the "
                                 "real rig by itself"},
        "samples": "every integer and half frame of each source scene range (W106 sample_times); imported files are "
                   "sampled at the identical times regardless of their imported frame range",
        "metrics": {"gate": "gate_max_abs_coord_m: max |coordinate| difference of world heads and tails, all source "
                            "bones present in the import, integer frames (W106 metric, 1 mm gate)",
                    "diagnostics": ["euclidean integer-frame head/tail error (max/p95/median over bone-frames and "
                                    "per-frame max)", "frames > 1 mm", "half-frame max abs coordinate",
                                    "armature-object rotation (deg) / translation (m) / scale error",
                                    "armature-space tail error (world metres)",
                                    "local quaternion angle (bones with comparable rest rotation only)",
                                    "per-bone gate metric, worst 5 bones / frames",
                                    "imported Action count/names/selection, interpolation, F-curve and key counts, "
                                    "object channels", "FBX stacks/takes/curves/curve keys (parsed file)",
                                    "fps / frame range / file size", "source Action digest before/after export"]},
        "tolerances": C.TOL,
        "negative_controls": {
            "NC1_source_fidelity": "W107 source samples == W106 n0_reopen samples (bitwise)",
            "NC2_v0_reproduces_w106": "W107 V0 imported samples vs W106 fbx_n0 <= deterministic_m and gate metric "
                                      "within deterministic_m of the W106 recomputed value",
            "NC3_v0d_equals_v0": "diagnostic operator call with V0 arguments == production export_fbx (samples "
                                 "<= deterministic_m)",
            "NC4_repeat": "every variant on a fresh repeat probe <= deterministic_m",
            "NC5_action_unmodified": "source Action digest identical before/after every export; work file unchanged",
            "NC6_synthetic_control": "W106 synthetic V0 gate metric <= synthetic_control_m",
            "NC7_originals_unchanged": "every reused W106 file hash unchanged after the run"},
        "hypotheses": {
            "H107_V1": "default key simplification owns a material share of the real-rig integer-frame drift -> V1 "
                       "reduces gate metric to <= 0.5*V0 on every real case, no collateral bone > 0.1 mm",
            "H107_V2": "take/Action enumeration changes the selected Action's geometry -> V2 reduces the gate metric "
                       "(take count/identity reported separately; take contamination != geometric parity)",
            "H107_V3": "animated armature object and/or 0.01 object scale produce drift under V0 in a world-equivalent "
                       "synthetic (descriptive 2x2; mechanism only)",
            "importer": "NOT_RUN (see importer_probe)", "rotation_conversion": "NOT_INDEPENDENTLY_MANIPULABLE"},
        "factor_rule": C.factor_support.__doc__.strip(),
        "case_rule": C.decide_case.__doc__.strip(),
        "stop_conditions": ["declaration hash mismatch", "Blender version differs from declaration",
                            "scratch input hash mismatch", "V0 non-reproduction or non-determinism -> CASE D",
                            "source Action or original artifact mutated -> CASE D",
                            "shared/production file changed -> STOP", "resource cap exceeded"],
        "resource_cap": {"blender_runs": "options + 2 probes", "wall_clock_per_probe_s": 900, "disk_gb": 3,
                         "gpu": "none (no rendering planned)", "network": "none"},
        "renders": "none planned; only if numerical evidence is ambiguous, a few predetermined outlier frames",
    }
    path = out/"declaration.json"
    wjson(path, decl)
    s = sha256(path)
    (out/"declaration.sha256").write_text(s + "\n")
    print(s)
    return 0


# --- analyze ------------------------------------------------------------------------------------------------------
def _meta(d, v):
    p = d/f"{v}_meta.json"
    return json.loads(p.read_text()) if p.exists() else None


def _selection_ok(m, action):
    if m is None or m.get("status") != "IMPORTED":
        return False
    if m["selection"] == "name_match":
        return True
    return m["selection"] == "single_take" and m["exported_active"]["action"] == action


def analyze_case(cid, case, p1, p2, w106, variants):
    d1, d2 = p1/cid, p2/cid
    w = w106/"results/probe"/CASE_DIRS[cid]
    src = C.load_npz(d1/"source.npz")
    rest = json.loads((d1/"source_rest.json").read_text())
    res = {"kind": case["kind"], "action": case["action"]}
    w106_n0 = C.load_npz(w/"n0_reopen.npz")
    res["source_fidelity_max_diff"] = C.arrays_max_diff(src, w106_n0)
    rep = json.loads((p1/"probe_report.json").read_text())["cases"][cid]
    res["source"] = {k: rep[k] for k in ("timing", "samples", "source_action_digest", "source_fcurves",
                                         "source_keys_total", "source_interpolation", "stored_actions", "armature_scale")}
    res["source"]["stored_action_count"] = len(rep["stored_actions"])
    res["source_digest_equals_w106_n0"] = rep["source_action_digest"] == case["w106_n0_channel_digest"]
    if "w106_declared_action_digest" in case:
        res["source_digest_equals_w106_declared"] = rep["source_action_digest"] == case["w106_declared_action_digest"]
    res["work_file_unchanged"] = rep["work_sha256_end"] == case["input_sha256"]
    # W106 baseline recomputed with the W107 metric (must equal the published values)
    w106_fbx = C.load_npz(w/"fbx_n0.npz")
    res["w106_v0_recomputed_gate_m"] = C.parity(w106_n0, w106_fbx)["gate_max_abs_coord_m"]
    res["variants"] = {}
    for v in variants:
        m = _meta(d1, v)
        if m is None or m.get("status") != "IMPORTED":
            res["variants"][v] = {"status": (m or {}).get("status", rep["variants"].get(v, {}).get("status", "NOT_RUN")),
                                  "error": rep["variants"].get(v, {}).get("error")}
            continue
        imp = C.load_npz(d1/f"{v}.npz")
        par = C.parity(src, imp, m.get("rest_rotation_comparable") or None)
        rv = {"status": "IMPORTED", "parity": par,
              "export": {k: m[k] for k in ("export", "fbx_bytes", "fbx_sha256", "fbx_structure", "action_unmodified",
                                           "work_file_unchanged", "exported_active")},
              "import": {k: m.get(k) for k in ("imported_actions", "imported_action_count", "selected_action", "selection",
                                               "active_after_import", "imported_timing", "imported_armature_scale",
                                               "imported_bones", "imported_fcurves", "imported_keys_total",
                                               "imported_interpolation", "target_keys", "object_channels",
                                               "object_keys", "imported_channel_digest")},
              "selection_ok": _selection_ok(m, case["action"])}
        m2 = _meta(d2, v)
        if m2 is not None and m2.get("status") == "IMPORTED":
            rv["repeat_max_diff"] = C.arrays_max_diff(imp, C.load_npz(d2/f"{v}.npz"))
            rv["repeat_fbx_bytes_equal"] = m2["fbx_bytes"] == m["fbx_bytes"]
            rv["repeat_fbx_sha_equal"] = m2["fbx_sha256"] == m["fbx_sha256"]
        else:
            rv["repeat_max_diff"] = None
        rv["repeat_ok"] = rv["repeat_max_diff"] is not None and rv["repeat_max_diff"] <= C.TOL["deterministic_m"]
        res["variants"][v] = rv
    v0 = res["variants"].get("V0", {})
    if v0.get("status") == "IMPORTED":
        imp0 = C.load_npz(d1/"V0.npz")
        res["v0_vs_w106_fbx_max_diff"] = C.arrays_max_diff(imp0, w106_fbx)
        res["v0_gate_minus_w106_gate_m"] = v0["parity"]["gate_max_abs_coord_m"] - res["w106_v0_recomputed_gate_m"]
        if res["variants"].get("V0d", {}).get("status") == "IMPORTED":
            res["v0d_vs_v0_max_diff"] = C.arrays_max_diff(C.load_npz(d1/"V0d.npz"), imp0)
        for v in C.CAUSAL_VARIANTS:
            vf = res["variants"].get(v, {})
            if vf.get("status") == "IMPORTED":
                vf["collateral_bones"] = C.collateral(v0["parity"]["per_bone_max_abs_coord_m"],
                                                      vf["parity"]["per_bone_max_abs_coord_m"])
                vf["gate_delta_vs_v0_m"] = vf["parity"]["gate_max_abs_coord_m"] - v0["parity"]["gate_max_abs_coord_m"]
    T = C.TOL["deterministic_m"]
    res["controls"] = {
        "source_fidelity_ok": res["source_fidelity_max_diff"] <= C.TOL["source_fidelity_m"] and res["source_digest_equals_w106_n0"],
        "v0_reproduces_w106": bool(v0.get("status") == "IMPORTED" and res.get("v0_vs_w106_fbx_max_diff", np.inf) <= T
                                   and abs(res.get("v0_gate_minus_w106_gate_m", np.inf)) <= T),
        "v0_deterministic": bool(v0.get("repeat_ok")),
        "v0d_equals_v0": bool(res.get("v0d_vs_v0_max_diff", np.inf) <= T),
        "actions_unmodified": all(x.get("export", {}).get("action_unmodified", False)
                                  and x.get("export", {}).get("work_file_unchanged", False)
                                  for x in res["variants"].values() if x.get("status") == "IMPORTED")
                              and res["work_file_unchanged"]}
    return res


def analyze_v3(p1, p2):
    rep = json.loads((p1/"probe_report.json").read_text()).get("v3", {})
    if rep.get("status") == "FAILED" or "cells" not in rep:
        return {"status": "NOT_TESTABLE", "reason": rep.get("error", "V3 not run")}
    out = {"cells": {}}
    eq = {c: rep["cells"][c]["equivalence_vs_reference_m"] for c in C.V3_CELLS}
    out["equivalence_vs_reference_m"] = eq
    out["equivalence_ok"] = all(v <= C.TOL["v3_equivalence_m"] for v in eq.values())
    if not out["equivalence_ok"]:
        out["status"] = "NOT_TESTABLE"
        out["reason"] = "world-equivalence invariant failed; V3 outcomes are not evidence"
        return out
    g = {}
    for c in C.V3_CELLS:
        d = p1/"v3"/c
        m = _meta(d, "V0")
        if m is None or m.get("status") != "IMPORTED":
            out["status"] = "NOT_TESTABLE"
            out["reason"] = f"{c}: import failed ({(m or {}).get('status')})"
            return out
        src, imp = C.load_npz(d/"source.npz"), C.load_npz(d/"V0.npz")
        par = C.parity(src, imp, m.get("rest_rotation_comparable") or None)
        rec = {"object_animated": C.V3_ANIMATED[c], "object_scale": C.V3_SCALE[c], "parity": par,
               "import": {k: m.get(k) for k in ("imported_action_count", "selected_action", "selection",
                                                "imported_interpolation", "imported_keys_total", "object_keys",
                                                "imported_armature_scale", "imported_timing")},
               "fbx_structure": m["fbx_structure"], "source_fcurves": rep["cells"][c]["source_fcurves"]}
        if (p2/"v3"/c/"V0.npz").exists():
            rec["repeat_max_diff"] = C.arrays_max_diff(imp, C.load_npz(p2/"v3"/c/"V0.npz"))
        rec["repeat_ok"] = rec.get("repeat_max_diff") is not None and rec["repeat_max_diff"] <= C.TOL["deterministic_m"]
        out["cells"][c] = rec
        g[c] = par["gate_max_abs_coord_m"]
    out["gate_m"] = g
    out["contrasts_m"] = {
        "scale_effect_static": g["static_s001"] - g["static_s1"],
        "scale_effect_animated": g["animated_s001"] - g["animated_s1"],
        "animation_effect_s1": g["animated_s1"] - g["static_s1"],
        "animation_effect_s001": g["animated_s001"] - g["static_s001"],
        "interaction": (g["animated_s001"] - g["animated_s1"]) - (g["static_s001"] - g["static_s1"])}
    out["cells_gt_1mm"] = sorted(c for c in C.V3_CELLS if g[c] > C.TOL["fbx_rt_parity_m"])
    out["repeat_ok"] = all(out["cells"][c]["repeat_ok"] for c in C.V3_CELLS)
    out["status"] = "TESTED" if out["repeat_ok"] else "NOT_TESTABLE"
    if not out["repeat_ok"]:
        out["reason"] = "V3 repeat not deterministic"
    return out


def analyze(o):
    decl_bytes = Path(o.declaration).read_bytes()
    if hashlib.sha256(decl_bytes).hexdigest() != o.declaration_sha:
        raise SystemExit("STOP: declaration hash mismatch")
    decl = json.loads(decl_bytes)
    out = Path(o.out_dir)
    out.mkdir(parents=True, exist_ok=False)
    p1, p2, w106 = Path(o.probe_dir), Path(o.repeat_dir), Path(o.w106)
    variants = decl["variant_order_testable"]
    rep = {"schema": C.SCHEMA + ":analysis", "declaration_sha256": o.declaration_sha, "cases": {}}
    for cid, case in decl["cases"].items():
        rep["cases"][cid] = analyze_case(cid, case, p1, p2, w106, variants)
    rep["v3"] = analyze_v3(p1, p2)
    real = list(C.REAL_CASES)
    factors = {}
    for v in C.CAUSAL_VARIANTS:
        if not decl["variants_testable"].get(v):
            continue
        per = {}
        for c in real:
            r = rep["cases"][c]["variants"]
            vf = r.get(v, {})
            per[c] = {"v0": r["V0"]["parity"]["gate_max_abs_coord_m"],
                      "vf": vf["parity"]["gate_max_abs_coord_m"] if vf.get("status") == "IMPORTED" else None,
                      "collateral": vf.get("collateral_bones", ["NOT_IMPORTED"]),
                      "repeat_ok": vf.get("repeat_ok", False), "action_identity_ok": vf.get("selection_ok", False)}
        factors[v] = C.factor_support(per)
        factors[v]["inputs"] = per
    rep["factors"] = factors
    # synthetic W106 control (separately labelled; also reports V1/V2 there, never pooled)
    syn = rep["cases"][C.W106_SYNTHETIC]
    rep["synthetic_control"] = {"v0_gate_m": syn["variants"]["V0"]["parity"]["gate_max_abs_coord_m"]
                                if syn["variants"].get("V0", {}).get("status") == "IMPORTED" else None}
    rep["synthetic_control"]["ok"] = bool(rep["synthetic_control"]["v0_gate_m"] is not None
                                          and rep["synthetic_control"]["v0_gate_m"] <= C.TOL["synthetic_control_m"])
    allc = rep["cases"].values()
    ctrl = {k: all(c["controls"][k] for c in allc) for k in
            ("source_fidelity_ok", "v0_reproduces_w106", "v0_deterministic", "v0d_equals_v0", "actions_unmodified")}
    unchanged = {rel: (w106/rel).exists() and sha256(w106/rel) == v["sha256"]
                 for rel, v in json.loads(Path(o.verify).read_text())["w106_files"].items()}
    rep["originals_unchanged"] = {"files": unchanged, "ok": all(unchanged.values())}
    shared = git("status", "--porcelain", "--", *SHARED_PATHS).splitlines()
    rep["shared_tracked_changes"] = [s for s in shared if not s.startswith("??")]
    pre = {"preflight_ok": json.loads(Path(o.verify).read_text())["ok"] and not rep["shared_tracked_changes"],
           **ctrl, "originals_unchanged": rep["originals_unchanged"]["ok"],
           "synthetic_control_ok": rep["synthetic_control"]["ok"], "v3_status": rep["v3"]["status"],
           "variants_testable": {v: decl["variants_testable"][v] for v in C.CAUSAL_VARIANTS},
           "factors": {v: {k: f[k] for k in ("supported", "mixed", "all_1mm", "any_1mm")} for v, f in factors.items()}}
    case, why = C.decide_case(pre)
    rep["decision_inputs"] = pre
    rep["CASE"], rep["CASE_reason"] = case, why
    rep["published_w106_drift_mm"] = W106_DRIFT_MM
    s = wjson(out/"analysis_report.json", rep)
    summary = {"CASE": case, "reason": why, "sha256": s, "controls": ctrl, "synthetic_control": rep["synthetic_control"],
               "v3": {k: rep["v3"].get(k) for k in ("status", "gate_m", "contrasts_m", "equivalence_ok")},
               "gate_m": {c: {v: r["variants"][v]["parity"]["gate_max_abs_coord_m"] for v in r["variants"]
                              if r["variants"][v].get("status") == "IMPORTED"} for c, r in rep["cases"].items()},
               "factors": {v: {k: f[k] for k in ("supported", "mixed", "all_1mm", "any_1mm")} | {"cases": f["cases"]}
                           for v, f in factors.items()}}
    print(json.dumps(summary, indent=1))
    return 0


# --- v3-posthoc (descriptive decomposition of the V3 V0 error; added after the V3 outcome was read; never a gate) -
def v3_posthoc(o):
    out = Path(o.out_dir)
    out.mkdir(parents=True, exist_ok=False)
    p1 = Path(o.probe_dir)
    rep = {"schema": C.SCHEMA + ":v3_posthoc",
           "label": "POST-HOC descriptive per-bone head/tail decomposition of the V3 V0 round-trip error, written after "
                    "the V3 gate outcome was read; does not change any gate, rule or CASE", "cells": {}}
    for c in C.V3_CELLS:
        d = p1/"v3"/c
        s, i = C.load_npz(d/"source.npz"), C.load_npz(d/"V0.npz")
        bs, bi = [str(b) for b in s["bones"]], [str(b) for b in i["bones"]]
        ix = [bi.index(b) for b in bs]
        it = s["times"] == np.floor(s["times"])
        dh = np.abs(s["world_head"][it] - i["world_head"][it][:, ix]).max(axis=(0, 2))
        dt = np.abs(s["world_tail"][it] - i["world_tail"][it][:, ix]).max(axis=(0, 2))
        base_tail = np.linalg.norm(s["world_tail"][it][:, 0] - i["world_tail"][it][:, 0], axis=-1)
        rest_s = json.loads((d/"source_rest.json").read_text())
        rep["cells"][c] = {
            "head_max_abs_coord_m": dict(zip(bs, map(float, dh))), "tail_max_abs_coord_m": dict(zip(bs, map(float, dt))),
            "base_tail_euclid_m": {"min": float(base_tail.min()), "max": float(base_tail.max())},
            "base_children": [n for n, _, _, p in C.V3_BONES if p == "base"],
            "base_rest_length_armature_units": float(np.linalg.norm(np.array(rest_s["base"])[:3, 1])),
            "imported_bones": bi,
            "max_excluding_base_tail_m": float(max(dh.max(), np.delete(dt, 0).max()))}
    g = {c: rep["cells"][c]["max_excluding_base_tail_m"] for c in C.V3_CELLS}
    rep["max_excluding_base_tail_m"] = g
    rep["contrasts_excluding_base_tail_m"] = {
        "scale_effect_static": g["static_s001"] - g["static_s1"], "scale_effect_animated": g["animated_s001"] - g["animated_s1"],
        "animation_effect_s1": g["animated_s1"] - g["static_s1"], "animation_effect_s001": g["animated_s001"] - g["static_s001"]}
    s = wjson(out/"v3_posthoc.json", rep)
    print(s)
    print(json.dumps({"max_excluding_base_tail_m": g, "contrasts": rep["contrasts_excluding_base_tail_m"],
                      "base_tail": {c: rep["cells"][c]["base_tail_euclid_m"] for c in C.V3_CELLS}}, indent=1))
    return 0


# --- manifest -----------------------------------------------------------------------------------------------------
def manifest(o):
    base = Path(o.base)
    files = {}
    for p in sorted(base.rglob("*")):
        if p.is_file() and p.name != "evidence_manifest.json":
            files[p.relative_to(base).as_posix()] = {"sha256": sha256(p), "bytes": p.stat().st_size}
    ver = json.loads(Path(o.verify).read_text())
    w106 = Path(o.w106)
    unchanged = {rel: (w106/rel).exists() and sha256(w106/rel) == v["sha256"] for rel, v in ver["w106_files"].items()}
    unchanged["w106/evidence_manifest.json"] = sha256(w106/"evidence_manifest.json") == W106_MANIFEST_SHA
    rep = {"schema": C.SCHEMA + ":manifest", "starting_head": o.starting_head, "declaration_sha256": o.declaration_sha,
           "originals_unchanged": unchanged, "all_originals_unchanged": all(unchanged.values()), "files": files}
    s = wjson(base/"evidence_manifest.json", rep)
    print(s, len(files), rep["all_originals_unchanged"])
    return 0 if rep["all_originals_unchanged"] else 2


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=("verify", "declare", "analyze", "v3-posthoc", "manifest"))
    for name in ("base", "w106", "verify", "options", "lock", "blender", "starting_head", "declaration",
                 "declaration_sha", "probe_dir", "repeat_dir", "out_dir"):
        ap.add_argument("--"+name.replace("_", "-"))
    o = ap.parse_args()
    return {"verify": verify, "declare": declare, "analyze": analyze, "v3-posthoc": v3_posthoc,
            "manifest": manifest}[o.mode](o)


if __name__ == "__main__":
    sys.exit(main())
