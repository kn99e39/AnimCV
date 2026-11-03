#!/usr/bin/env python3
"""W108 host-side stages: FBX subframe motion semantics and interpolation ownership (evidence only).

  verify    W107 COMPLETE marker and evidence-manifest / declaration identity, every W107 manifest file, W106 manifest
            identity, frozen exporter (SHA + AST arguments), W106/W107 script SHAs, worker lock, Git state, no other
            Blender process; byte-verified scratch copies of the W107 inputs into <base>/results/inputs
  declare   cases, identities, exact sample times, interval classes, metrics, tolerances, P0 references, the single P2
            intervention, synthetic controls, expected signatures, CASE rule, stop conditions and resource cap ->
            declaration.json + SHA-256 (``outcomes_read: false``)
  analyze   P0 replay of W107, P1 boundary tracing (source / FBX encoding by two parsers / import), P2, P3, repeat
            determinism -> analysis_report.json and the CASE
  manifest  hashes of every output, plus a re-check that no W106/W107 original changed

No production exporter/importer, rig, Action original, scene setting or historical artifact is modified.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT/"scripts"))
import w108_fbx_subframe_contract as C  # noqa: E402

sha256 = C.sha256_file
W107_MANIFEST_SHA = "20fcb1b9f6ec8879247ac91ae3a21ca1185e995340976079bc80936316f57278"
W107_DECLARATION_SHA = "aeb9ce710d01878e8d55a0b26d1d117e9ddfc9868ae397c6d88d1567975f5a76"
W106_MANIFEST_SHA = "f3003fe54e68c6cf182313c0b784468275b7ed039f50f574ad10e64eb8fb97db"
W107_WORKLOG = "docs/107_WORKLOG_FBX_ROUNDTRIP_FAILURE_OWNERSHIP.md"
EXPORTER = "src/blender/export.py"
EXPORTER_SHA = "521a63d96401f6f82b072ab8c82800e7795623a66370d70a257d72591f76b5fa"
W107_SCRIPTS = {"scripts/w107_fbx_contract.py": "9af0a3289c137e159859f4e065bee3f067f3194ea3fc3d6902c5a893fe034262",
                "scripts/w107_fbx_blender.py": "985fc4bc5ccea755896d7b18230a69598ed9b08ed7c7dbd8161d6300dd04af85",
                "scripts/w106_action_blender.py": "8d33879c30e3d1941cb0e986f1e03ee2bd2f3140dbf0a8f41ab6962b5c2716ae",
                "scripts/w106_action_contract.py": "d7ff860f0024e4d0686f1186696b901eadea2798ce13d0bf5f053ddf151f725d"}
W107_OWNERSHIP_FINAL_PREFIX = "a8fdebc5"   # published in the W107 Worklog (post-declaration v3-posthoc stage)
# Self-referential manifest logs: each was captured empty when its own manifest hashed it (the stage's stdout was
# still being redirected into it).  Known, recorded, and the only exceptions allowed.
MANIFEST_LOG_EXCEPTIONS = {"w107": "manifest.log", "w106": "results/manifest.log"}
EXPECTED_ENV = {"blender_version": "5.1.0", "fbx_addon_version": [5, 15, 0]}
SHARED_PATHS = ["src", "tests", "scripts", "examples", "docs", "Architecture_v2.md", "Architecture_v3_FramePose.md"]
W108_NEW = ("scripts/w108_", "tests/test_w108_", "docs/108_WORKLOG_")


def wjson(path, obj):
    Path(path).write_text(json.dumps(obj, indent=1))
    return sha256(path)


def git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def export_kwargs(text):
    tree = ast.parse(text)
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
             and n.func.attr == "fbx" and isinstance(n.func.value, ast.Attribute) and n.func.value.attr == "export_scene"]
    if len(calls) != 1:
        raise ValueError("expected one export_scene.fbx call")
    kw = {k.arg: ast.literal_eval(k.value) for k in calls[0].keywords if k.arg != "filepath"}
    return {k: (sorted(v) if isinstance(v, set) else v) for k, v in kw.items()}


def check_manifest(base, man, exception):
    bad, n = [], 0
    for rel, v in man["files"].items():
        p = Path(base)/rel
        n += 1
        if not p.exists():
            bad.append([rel, "missing"])
        elif sha256(p) != v["sha256"] and rel != exception:
            bad.append([rel, "sha"])
    exc = Path(base)/exception
    return {"files": n, "bad": bad, "ok": not bad,
            "exception": {"file": exception, "manifest_sha256": man["files"].get(exception, {}).get("sha256"),
                          "manifest_bytes": man["files"].get(exception, {}).get("bytes"),
                          "actual_sha256": sha256(exc) if exc.exists() else None,
                          "actual_bytes": exc.stat().st_size if exc.exists() else None}}


# --- verify -------------------------------------------------------------------------------------------------------
def verify(o):
    base = Path(o.base)
    out = base/"results"/"verify"
    out.mkdir(parents=True, exist_ok=False)
    w107, w106 = Path(o.w107), Path(o.w106)
    rep = {"schema": C.SCHEMA + ":verify", "checks": {}, "stop": []}
    wl = (ROOT/W107_WORKLOG).read_text(encoding="utf-8")
    rep["checks"]["w107_complete_marker"] = wl.rstrip().endswith("AUTOMATION_BATCH_STATUS: COMPLETE\nBATCH_NUMBER: 107")
    m7sha = sha256(w107/"evidence_manifest.json")
    m7 = json.loads((w107/"evidence_manifest.json").read_text())
    rep["checks"]["w107_manifest_sha"] = {"value": m7sha, "ok": m7sha == W107_MANIFEST_SHA}
    rep["checks"]["w107_manifest_originals_unchanged"] = bool(m7.get("all_originals_unchanged"))
    d7sha = sha256(w107/"results/declare/declaration.json")
    rep["checks"]["w107_declaration_sha"] = {"value": d7sha, "ok": d7sha == W107_DECLARATION_SHA
                                             and m7.get("declaration_sha256") == W107_DECLARATION_SHA}
    rep["w107_files"] = check_manifest(w107, m7, MANIFEST_LOG_EXCEPTIONS["w107"])
    rep["checks"]["w107_files_intact"] = rep["w107_files"]["ok"]
    m6sha = sha256(w106/"evidence_manifest.json")
    m6 = json.loads((w106/"evidence_manifest.json").read_text())
    rep["checks"]["w106_manifest_sha"] = {"value": m6sha, "ok": m6sha == W106_MANIFEST_SHA}
    rep["w106_files"] = check_manifest(w106, m6, MANIFEST_LOG_EXCEPTIONS["w106"])
    rep["checks"]["w106_files_intact"] = rep["w106_files"]["ok"]
    # frozen exporter and reused scripts
    exp_sha = sha256(ROOT/EXPORTER)
    rep["checks"]["exporter_sha"] = {"value": exp_sha, "ok": exp_sha == EXPORTER_SHA}
    rep["exporter_v0_kwargs"] = export_kwargs((ROOT/EXPORTER).read_text(encoding="utf-8"))
    d7 = json.loads((w107/"results/declare/declaration.json").read_text())
    rep["checks"]["exporter_kwargs_equal_w107"] = rep["exporter_v0_kwargs"] == d7["exporter"]["v0_kwargs"]
    scr = {p: sha256(ROOT/p) for p in W107_SCRIPTS}
    rep["reused_scripts_sha256"] = scr
    rep["checks"]["reused_scripts_unchanged"] = all(scr[p] == s for p, s in W107_SCRIPTS.items())
    own = sha256(ROOT/"scripts/w107_fbx_ownership.py")
    rep["checks"]["w107_ownership_final"] = {"value": own, "ok": own.startswith(W107_OWNERSHIP_FINAL_PREFIX)}
    # repository / worker / process state
    head = git("rev-parse", "HEAD")
    status = git("status", "--porcelain").splitlines()
    tracked = [s for s in status if not s.startswith("??")]
    untracked = [s[3:] for s in status if s.startswith("??")]
    rep["git"] = {"head": head, "branch": git("branch", "--show-current"), "origin": git("rev-parse", "origin/" + o.branch),
                  "tracked_changes": tracked, "untracked": untracked}
    rep["checks"]["starting_head"] = {"value": head, "ok": head == o.starting_head}
    rep["checks"]["branch"] = rep["git"]["branch"] == o.branch
    rep["checks"]["origin_equals_head"] = rep["git"]["origin"] == head
    rep["checks"]["working_tree_only_new_w108_files"] = not tracked and all(u.startswith(W108_NEW) for u in untracked)
    lk = json.loads(Path(o.lock).read_text()) if Path(o.lock).exists() else {}
    rep["worker_lock"] = lk
    rep["checks"]["worker_lock_batch_108_running"] = lk.get("batch") == 108 and lk.get("state") == "RUNNING" \
        and lk.get("start_commit") == o.starting_head
    tl = subprocess.run(["tasklist", "/FO", "CSV", "/NH"], capture_output=True, text=True, errors="replace").stdout
    blenders = [ln for ln in tl.splitlines() if ln.lower().startswith('"blender')]
    rep["other_blender_processes"] = blenders
    rep["checks"]["no_other_blender_process"] = not blenders
    rep["checks"]["blender_present"] = Path(o.blender).exists()
    for k, v in rep["checks"].items():
        if not (v if isinstance(v, bool) else v["ok"]):
            rep["stop"].append(k)
    # byte-verified scratch copies of the W107 inputs (themselves byte copies of the W106 N0 files)
    inputs = base/"results"/"inputs"
    inputs.mkdir(exist_ok=False)
    copies = {}
    for cid in list(C.REAL_CASES) + [C.SYN_INPUT]:
        rel = f"results/inputs/{cid}_n0.blend"
        srcp = w107/rel
        dst = inputs/f"{cid}_n0.blend"
        shutil.copyfile(srcp, dst)
        copies[cid] = {"source": str(srcp), "path": str(dst), "sha256": sha256(dst),
                       "w107_manifest_sha256": m7["files"][rel]["sha256"],
                       "w107_declared_sha256": d7["cases"][cid]["input_sha256"]}
        copies[cid]["ok"] = copies[cid]["sha256"] == copies[cid]["w107_manifest_sha256"] == copies[cid]["w107_declared_sha256"]
        if not copies[cid]["ok"]:
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
    w107, w106 = Path(o.w107), Path(o.w106)
    ver = json.loads(Path(o.verify).read_text())
    if not ver["ok"]:
        raise SystemExit("STOP: verify failed")
    opts = json.loads(Path(o.options).read_text())
    if opts["blender_version"] != EXPECTED_ENV["blender_version"] or \
            opts["fbx_addon"]["version"] != EXPECTED_ENV["fbx_addon_version"]:
        raise SystemExit(f"STOP: unexpected environment {opts['blender_version']} / {opts['fbx_addon']['version']}")
    d7 = json.loads((w107/"results/declare/declaration.json").read_text())
    if opts["fbx_addon"].get("source_sha256") != d7["environment"]["fbx_addon_source_sha256"]:
        raise SystemExit("STOP: FBX add-on sources differ from W107")
    a7 = json.loads((w107/"results/analysis/analysis_report.json").read_text())
    p7 = json.loads((w107/"results/probe/probe_report.json").read_text())
    exp_props = opts["export_scene.fbx"]
    v0 = ver["exporter_v0_kwargs"]
    if any(k in v0 for k in C.V1_OVERRIDE) or exp_props["bake_anim_simplify_factor"]["default"] == 0.0:
        raise SystemExit("STOP: V1 would not be a one-property change from V0")
    cases = {}
    for cid in C.REAL_CASES:
        n0 = json.loads((w106/"results/probe"/cid/"n0_reopen.json").read_text())
        tm = n0["timing"]
        keys = sorted({f for c in n0["channels"] for f in c["frames"]})
        times = C.sample_times(tm["frame_start"], tm["frame_end"])
        cls = C.interval_classes(tm["frame_start"], tm["frame_end"], keys)
        w7 = {v: a7["cases"][cid]["variants"][v]["parity"] for v in ("V0", "V1")}
        cases[cid] = {
            "id": cid, "kind": "real", "regime": d7["cases"][cid].get("regime"),
            "input_copy": ver["inputs"][cid]["path"], "input_sha256": ver["inputs"][cid]["sha256"],
            "action": C.REAL_ACTIONS[cid], "expected_slot": n0["active"]["slot"],
            "w107_source_action_digest": p7["cases"][cid]["source_action_digest"],
            "w106_declared_action_digest": d7["cases"][cid]["w106_declared_action_digest"],
            "timing": tm, "source_key_frames": keys, "times": times,
            "interval_classes": {str(k): v for k, v in cls.items()},
            "n_transition_intervals": sum(v == "transition" for v in cls.values()),
            "n_quiet_intervals": sum(v == "quiet" for v in cls.values()),
            "timebase_note": d7["cases"][cid].get("timebase"),
            "p0_reference": {
                "w107_npz": {k: f"results/probe/{cid}/{k}.npz" for k in ("source", "V0", "V1")},
                "w107_npz_sha256": {k: sha256(w107/f"results/probe/{cid}/{k}.npz") for k in ("source", "V0", "V1")},
                "w107_gate_m": {v: w7[v]["gate_max_abs_coord_m"] for v in w7},
                "w107_half_frame_m": {v: w7[v]["half_frame_max_abs_coord_m"] for v in w7}}}
    n0s = json.loads((w106/"results/probe/synthetic/case/n0_reopen.json").read_text())
    tm = n0s["timing"]
    skeys = sorted({f for c in n0s["channels"] for f in c["frames"]})
    scls = C.interval_classes(tm["frame_start"], tm["frame_end"], skeys)
    controls = {}
    for ctrl, spec in C.SYN_CONTROLS.items():
        controls[ctrl] = {**spec, "kind": "synthetic control (separately labelled, never pooled)", "timing": tm,
                          "source_key_frames": skeys, "times": C.sample_times(tm["frame_start"], tm["frame_end"]),
                          "interval_classes": {str(k): v for k, v in scls.items()}}
    decl = {
        "schema": C.SCHEMA + ":declaration", "outcomes_read": False,
        "note": "Frozen before any W108 export/import/sampling ran. W107 outcomes are historical P0 references.",
        "starting_head": o.starting_head, "verify_sha256": sha256(o.verify), "options_sha256": sha256(o.options),
        "environment": {"blender_version": opts["blender_version"], "blender_build_hash": opts["blender_build_hash"],
                        "fbx_addon_version": opts["fbx_addon"]["version"],
                        "fbx_addon_source_sha256": opts["fbx_addon"].get("source_sha256"),
                        "blender_exe": o.blender, "launch": "-b --factory-startup -t 2 --python-exit-code 1"},
        "scripts_sha256": {p: sha256(ROOT/p) for p in ("scripts/w108_fbx_subframe_contract.py",
                                                       "scripts/w108_fbx_subframe_blender.py",
                                                       "scripts/w108_fbx_subframe.py", *W107_SCRIPTS, EXPORTER)},
        "cases": cases,
        "synthetic_input": {"path": ver["inputs"][C.SYN_INPUT]["path"], "sha256": ver["inputs"][C.SYN_INPUT]["sha256"],
                            "base_action": C.SYN_BASE_ACTION, "construction": C.SYN_CONSTRUCTION},
        "synthetic_controls": controls,
        "exporter": {"v0": "blender.export.export_fbx (unchanged, called, never patched)", "v0_kwargs": v0,
                     "v1": "bpy.ops.export_scene.fbx(V0 kwargs + bake_anim_simplify_factor=0.0) -- W107 V1, diagnostic "
                           "only, not a production fix", "v1_override": C.V1_OVERRIDE,
                     "runtime_defaults": {k: exp_props[k]["default"] for k in ("bake_anim_simplify_factor",
                                                                              "bake_anim_step", "bake_anim_use_all_actions",
                                                                              "bake_anim_use_nla_strips",
                                                                              "bake_anim_force_startend_keying")}},
        "importer": "bpy.ops.import_scene.fbx(filepath=...) defaults into read_factory_settings(use_empty=True)",
        "importer_runtime_defaults": {k: v["default"] for k, v in opts["import_scene.fbx"].items()},
        "samples": "every integer frame of the source scene range plus +0.25/+0.50/+0.75 of every frame but the last "
                   "(explicit lists per case); imported files are sampled at the identical times; W107 integer+half "
                   "times are a subset used for P0",
        "interval_classes": C.interval_classes.__doc__.strip(),
        "metrics": {
            "primary": "max |coordinate| difference of world bone heads and tails over all source bones present in "
                       "the import (W106/W107 gate form), by phase (integer, .25, .5, .75), over all subframes, and "
                       "over subframes in transition / quiet intervals",
            "diagnostics": ["euclidean head/tail error max/p95/median and per-sample max", "samples > 1 mm",
                            "world head-only max abs coordinate", "world bone rotation angle (deg)",
                            "armature-object rotation (deg) / translation (m)", "armature-space tail error",
                            "per-bone subframe max, worst 5 bones / times"],
            "isolated_time_varying": "world head max abs coordinate and world rotation angle -- the synthetic metric "
                                     "if the synthetic V1 integer gate exceeds synthetic_rest_confound_m"},
        "p0": {"source_fidelity": "W108 source samples == W107 source.npz at common (integer+half) times, bitwise",
               "replay": "W108 V0 / V1 imported samples vs W107 V0 / V1 npz at common times <= deterministic_m",
               "metrics": "recomputed integer gate and half-frame max vs W107 analysis values <= w107_metric_reproduce_m",
               "identity": "unique name-matched Action, slot, timing, source digest == W107 and W106 declared digest, "
                           "source interpolation CONSTANT, source key frames == declared",
               "failure": "CASE D"},
        "p1": {"layers": ["source curves (interpolation, key times) and Blender evaluation (held between keys?)",
                          "FBX selected-stack AnimationCurve KeyTime grid and KeyAttrFlags interpolation bits, decoded "
                          "by an independent reader (this contract) and by io_scene_fbx.parse_fbx; reliable only if "
                          "both agree and KeyAttrRefCount covers every key",
                          "default-imported Action (interpolation, key times, keys per curve)",
                          "importer source handling of KeyAttrFlags: static inspection of import_fbx.py (recorded, "
                          "not a behavioural test)"],
               "fbx_interp_bits": {f"0x{k:x}": v for k, v in C.FBX_INTERP_BITS.items()},
               "boundary_rule": C.boundary.__doc__.strip()},
        "p2": C.P2,
        "p3": {"controls": list(C.SYN_CONTROLS), "path": "same V0 / V1 export, default import and P2 as the real cases",
               "validity": "construction valid (only interpolation differs from SYN baseline), step source held at "
                           "every subframe (bitwise equal to its integer frame), smooth source differs from its "
                           "integer frame in every interval (> 1e-6 m)",
               "expected": {"syn_step": "P2 subframe max <= material_reduction_ratio * V1 subframe max",
                            "syn_smooth": "P2 subframe max >= smooth_harm_ratio * V1 subframe max"},
               "confound": "if a control's V1 integer gate > synthetic_rest_confound_m the original gate is marked "
                           "confounded and the contrast is evaluated on the isolated time-varying metric (CASE <= C)"},
        "hypothesis": {
            "H108": "source hold semantics are lost at a discrete export/import interpolation boundary while V1 integer "
                    "keys are adequate: source CONSTANT, FBX and/or import non-CONSTANT; V1 subframe error concentrated "
                    "in transition intervals; P2 materially reduces subframe error with unchanged integer motion/key "
                    "data; step control improves under P2, smooth control is harmed",
            "expected_signatures": {
                "S1": "source interpolation {CONSTANT} and held by evaluation; FBX stack interpolation {LINEAR}; "
                      "imported {LINEAR}",
                "S2": "V1 quiet-interval subframe max <= sufficiency_m while transition-interval max > sufficiency_m "
                      "(descriptive; real_clip has no quiet intervals)",
                "S3": "P2 subframe max <= material_reduction_ratio * V1 subframe max on every real case; P2 integer "
                      "samples == V1 integer samples (<= deterministic_m); key data unchanged except interpolation",
                "S4": "P3 contrasts as declared"},
            "sufficiency": "P2 subframe max <= sufficiency_m (reported per case; not a CASE gate)"},
        "tolerances": C.TOL,
        "case_rule": C.decide_case.__doc__.strip(),
        "negative_controls": {"NC1": "source samples W108 == W107 (bitwise)", "NC2": "V0/V1 W108 == W107 at common times",
                              "NC3": "fresh repeat probe of every sample set <= deterministic_m",
                              "NC4": "P2_pre (fresh import, original Action) == V1 samples <= deterministic_m",
                              "NC5": "source Action digest unchanged across exports; scratch work file unchanged",
                              "NC6": "W106/W107 originals unchanged after the run",
                              "NC7": "smooth synthetic control (forcing CONSTANT must harm continuous motion)"},
        "not_run": {"importer_behavioural_flag_probe": "would need a hand-edited CONSTANT-flagged FBX (a second "
                                                       "intervention); importer handling is recorded by static source "
                                                       "inspection only",
                    "frame_end / takes / 50->30 fps timebase": "preserved, unresolved, not repaired",
                    "renders": "none planned; only if numerical evidence is ambiguous"},
        "stop_conditions": ["declaration hash mismatch", "Blender version differs", "scratch input hash mismatch",
                            "sample-time mismatch vs declaration", "P0 failure -> CASE D", "nondeterminism -> CASE D",
                            "original or source Action mutated -> CASE D", "shared/production file changed -> STOP",
                            "resource cap exceeded"],
        "resource_cap": {"blender_runs": "options + 2 probes", "wall_clock_per_probe_s": 900, "disk_gb": 3,
                         "gpu": "none", "network": "none"},
    }
    path = out/"declaration.json"
    wjson(path, decl)
    s = sha256(path)
    (out/"declaration.sha256").write_text(s + "\n")
    print(s)
    return 0


# --- analyze ------------------------------------------------------------------------------------------------------
def _mask_int(t):
    return C.phase_of(t) == 0


def fbx_check(fbx, fps, addon_summary, action):
    """Independent-reader summary of the selected stack, cross-checked with the add-on parser's summary."""
    root, version = C.fbx_read(fbx)
    ind = C.fbx_anim_summary(root, fps)
    ind["fbx_version"] = version
    st_i = C.select_stack(ind.get("stacks", {}), action)
    st_a = C.select_stack(addon_summary.get("stacks", {}), action) if addon_summary.get("status") == "PARSED" else None
    sel_i = ind["stacks"].get(st_i) if st_i else None
    sel_a = addon_summary["stacks"].get(st_a) if st_a else None
    all_interp = sorted({i for s in ind.get("stacks", {}).values() for i in s["interpolation"]})
    return {"version": version, "stack_name": st_i, "stack_count": len(ind.get("stacks", {})),
            "selected_independent": sel_i, "selected_addon": sel_a,
            "parsers_agree_all_stacks": addon_summary.get("stacks") == ind.get("stacks"),
            "all_stacks_interpolation": all_interp}


def analyze_flow(d1, d2, rep1, rep2, spec, kind):
    """Shared per-case / per-control analysis of one probe flow."""
    times = np.array(spec["times"])
    classes = {int(k): v for k, v in spec["interval_classes"].items()}
    src = C.load_npz(d1/"source.npz")
    res = {"times_equal_declared": bool(rep1.get("times_equal_declared")) and np.array_equal(src["times"], times),
           "source": {k: rep1.get(k) for k in ("assigned", "timing", "source_action_digest", "source_fcurves",
                                               "source_keys_total", "source_interpolation", "source_extrapolation",
                                               "source_modifiers", "source_key_times", "stored_actions",
                                               "armature_scale", "slots")}}
    res["source"]["key_frames_equal_declared"] = rep1.get("source_key_frames") == spec["source_key_frames"]
    # source hold semantics by evaluation: every subframe equals its integer frame (bitwise)?
    ph = C.phase_of(times)
    fl = np.floor(times)
    idx_int = {float(t): i for i, t in enumerate(times) if ph[i] == 0}
    per_interval = {}
    hold = 0.0
    for i, t in enumerate(times):
        if ph[i] == 0:
            continue
        j = idx_int[float(fl[i])]
        dev = float(max(np.abs(src["world_head"][i] - src["world_head"][j]).max(),
                        np.abs(src["world_tail"][i] - src["world_tail"][j]).max()))
        hold = max(hold, dev)
        per_interval[int(fl[i])] = max(per_interval.get(int(fl[i]), 0.0), dev)
    res["source_subframe_vs_floor_max_m"] = hold
    res["source_held_by_evaluation"] = hold == 0.0
    res["source_min_interval_change_m"] = min(per_interval.values()) if per_interval else None
    res["paths"] = {}
    for v in C.EXPORT_PATHS:
        r = rep1["paths"].get(v, {})
        if r.get("status") != "IMPORTED":
            res["paths"][v] = {"status": r.get("status", "NOT_RUN"), "error": r.get("error")}
            continue
        imp = C.load_npz(d1/f"{v}.npz")
        fps = rep1["timing"]["fps"]/rep1["timing"]["fps_base"]
        pv = {"status": "IMPORTED", "errors": C.subframe_report(src, imp, times, classes),
              "import": {k: r.get(k) for k in ("imported_action_count", "selected_action", "selection", "assigned",
                                               "imported_timing", "imported_fcurves", "imported_keys_total",
                                               "imported_interpolation", "target_keys", "imported_key_times",
                                               "object_channels")},
              "export": {k: r.get(k) for k in ("export", "fbx_bytes", "fbx_sha256", "action_unmodified",
                                               "work_file_unchanged")},
              "fbx": fbx_check(d1/"fbx"/f"{v}.fbx", fps, r.get("fbx_addon_summary", {}), spec["action"])}
        r2 = rep2["paths"].get(v, {}) if rep2 else {}
        pv["repeat_max_diff"] = C.arrays_max_diff(imp, C.load_npz(d2/f"{v}.npz")) if r2.get("status") == "IMPORTED" else None
        res["paths"][v] = pv
    res["source_repeat_max_diff"] = C.arrays_max_diff(src, C.load_npz(d2/"source.npz")) if rep2 else None
    # P2
    p = rep1.get("P2", {})
    pr = {"status": p.get("status", "NOT_RUN"), "record": {k: p.get(k) for k in p if k not in ("traceback",)}}
    if p.get("status") in ("TESTED", "NOT_TESTABLE") and (d1/"P2.npz").exists():
        p2 = C.load_npz(d1/"P2.npz")
        pre = C.load_npz(d1/"P2_pre.npz")
        v1 = C.load_npz(d1/"V1.npz")
        mi = _mask_int(times)
        pr["errors"] = C.subframe_report(src, p2, times, classes)
        pr["pre_vs_v1_max_diff"] = C.arrays_max_diff(pre, v1)
        pr["integer_vs_v1_max_diff"] = C.arrays_max_diff(p2, v1, mask_a=mi, mask_b=mi)
        pr["integer_unchanged"] = pr["integer_vs_v1_max_diff"] <= C.TOL["deterministic_m"]
        pr["keydata_ok"] = bool(p.get("keydata_ok"))
        kt = p.get("original_key_times", {})
        pr["imported_key_times_integer"] = kt.get("max_dev_from_integer", 1.0) <= C.TOL["key_time_integer_eps"]
        v1sub = res["paths"]["V1"]["errors"]["subframe"]["max_abs_coord_m"]
        p2sub = pr["errors"]["subframe"]["max_abs_coord_m"]
        pr["v1_subframe_max_m"], pr["p2_subframe_max_m"] = v1sub, p2sub
        pr["ratio_p2_over_v1"] = p2sub/v1sub if v1sub > 0 else None
        pr["material"] = bool(v1sub > 0 and p2sub <= C.TOL["material_reduction_ratio"]*v1sub)
        pr["sufficient"] = bool(p2sub <= C.TOL["sufficiency_m"])
        iso = {"head_max_abs_coord_m": {"v1": res["paths"]["V1"]["errors"]["subframe"]["head_max_abs_coord_m"],
                                        "p2": pr["errors"]["subframe"]["head_max_abs_coord_m"]}}
        iso["world_rot_deg_max"] = {"v1": res["paths"]["V1"]["errors"]["subframe"]["world_rot_deg"]["max"],
                                    "p2": pr["errors"]["subframe"]["world_rot_deg"]["max"]}
        pr["isolated"] = iso
        if rep2 and (d2/"P2.npz").exists():
            pr["repeat_max_diff"] = C.arrays_max_diff(p2, C.load_npz(d2/"P2.npz"))
            pr["repeat_pre_max_diff"] = C.arrays_max_diff(pre, C.load_npz(d2/"P2_pre.npz"))
        else:
            pr["repeat_max_diff"] = None
    res["P2"] = pr
    # determinism over every sample set of this flow
    reps = [res["source_repeat_max_diff"], pr.get("repeat_max_diff"), pr.get("repeat_pre_max_diff")] + \
        [res["paths"][v].get("repeat_max_diff") for v in C.EXPORT_PATHS]
    res["deterministic"] = all(x is not None and x <= C.TOL["deterministic_m"] for x in reps)
    res["actions_unmodified"] = all(res["paths"][v].get("export", {}).get("action_unmodified", False)
                                    and res["paths"][v].get("export", {}).get("work_file_unchanged", False)
                                    for v in C.EXPORT_PATHS) and bool(rep1.get("work_file_unchanged"))
    # P1 boundary
    v1 = res["paths"].get("V1", {})
    if v1.get("status") == "IMPORTED":
        src_interp = rep1.get("source_interpolation") if res["source_held_by_evaluation"] or kind != "real" else \
            ["NOT_HELD_BY_EVALUATION"]
        res["boundary"] = C.boundary(src_interp, v1["fbx"]["selected_independent"], v1["fbx"]["selected_addon"],
                                     v1["import"]["imported_interpolation"])
    else:
        res["boundary"] = {"boundary": "UNRESOLVED_NO_IMPORT"}
    return res


def analyze_real(cid, case, p1, p2, w107):
    rep1 = json.loads((p1/"probe_report.json").read_text())["cases"][cid]
    rep2 = json.loads((p2/"probe_report.json").read_text())["cases"][cid]
    res = analyze_flow(p1/cid, p2/cid, rep1, rep2, case, "real")
    res["kind"], res["action"], res["regime"] = "real", case["action"], case.get("regime")
    # P0: replay of W107
    src = C.load_npz(p1/cid/"source.npz")
    ref = case["p0_reference"]
    p0 = {"w107_npz_sha_unchanged": all(sha256(w107/ref["w107_npz"][k]) == ref["w107_npz_sha256"][k]
                                        for k in ref["w107_npz"])}
    w7s = C.load_npz(w107/ref["w107_npz"]["source"])
    ma, mb = C.common_time_masks(src["times"], w7s["times"])
    p0["common_times"] = int(ma.sum())
    p0["w107_times_all_present"] = bool(mb.all())
    p0["source_vs_w107_max_diff"] = C.arrays_max_diff(src, w7s, mask_a=ma, mask_b=mb)
    for v in C.EXPORT_PATHS:
        if res["paths"][v].get("status") != "IMPORTED":
            p0[f"{v}_vs_w107_max_diff"] = None
            continue
        imp = C.load_npz(p1/cid/f"{v}.npz")
        p0[f"{v}_vs_w107_max_diff"] = C.arrays_max_diff(imp, C.load_npz(w107/ref["w107_npz"][v]), mask_a=ma, mask_b=mb)
        g = C.w107_gate(src, imp)
        p0[f"{v}_gate_m"], p0[f"{v}_half_m"] = g["gate_max_abs_coord_m"], g["half_frame_max_abs_coord_m"]
        p0[f"{v}_gate_minus_w107_m"] = g["gate_max_abs_coord_m"] - ref["w107_gate_m"][v]
        p0[f"{v}_half_minus_w107_m"] = g["half_frame_max_abs_coord_m"] - ref["w107_half_frame_m"][v]
    T, Tm = C.TOL["deterministic_m"], C.TOL["w107_metric_reproduce_m"]
    p0["ok"] = bool(p0["w107_npz_sha_unchanged"] and p0["w107_times_all_present"]
                    and p0["source_vs_w107_max_diff"] <= C.TOL["source_fidelity_m"]
                    and all(p0[f"{v}_vs_w107_max_diff"] is not None and p0[f"{v}_vs_w107_max_diff"] <= T
                            and abs(p0[f"{v}_gate_minus_w107_m"]) <= Tm and abs(p0[f"{v}_half_minus_w107_m"]) <= Tm
                            for v in C.EXPORT_PATHS))
    res["P0"] = p0
    s = res["source"]
    res["identity_ok"] = bool(
        s["source_action_digest"] == case["w107_source_action_digest"] == case["w106_declared_action_digest"]
        and s["assigned"]["action"] == case["action"] and s["assigned"]["slot"] == case["expected_slot"]
        and s["timing"] == case["timing"] and s["key_frames_equal_declared"] and s["source_interpolation"] == ["CONSTANT"]
        and res["times_equal_declared"]
        and all(res["paths"][v].get("import", {}).get("selection") == "name_match" for v in C.EXPORT_PATHS))
    return res


def analyze_control(ctrl, spec, p1, p2):
    rep1 = json.loads((p1/"probe_report.json").read_text())["controls"][ctrl]
    rep2 = json.loads((p2/"probe_report.json").read_text())["controls"][ctrl]
    if rep1.get("status") == "FAILED":
        return {"kind": "synthetic", "status": "FAILED", "error": rep1.get("error"), "valid": False, "testable": False}
    res = analyze_flow(p1/ctrl, p2/ctrl, rep1, rep2, spec, "synthetic")
    res["kind"], res["action"] = "synthetic", spec["action"]
    res["build"] = rep1["build"]
    res["build_repeat_identical_work"] = rep1["build"]["work_sha256"] == rep2.get("build", {}).get("work_sha256")
    if spec["source_interpolation"] == "CONSTANT":
        sem_ok = res["source_held_by_evaluation"]
    else:
        sem_ok = res["source_min_interval_change_m"] is not None and res["source_min_interval_change_m"] > 1e-6
    res["source_semantics_ok"] = bool(sem_ok)
    res["valid"] = bool(rep1["build"]["valid_construction"] and sem_ok and res["times_equal_declared"]
                        and res["source"]["key_frames_equal_declared"]
                        and res["source"]["source_interpolation"] == [spec["source_interpolation"]])
    pr = res["P2"]
    res["testable"] = bool(pr.get("status") == "TESTED" and pr.get("integer_unchanged"))
    v1 = res["paths"].get("V1", {})
    gate_int = v1.get("errors", {}).get("integer", {}).get("max_abs_coord_m") if v1.get("status") == "IMPORTED" else None
    res["v1_integer_gate_m"] = gate_int
    res["confounded"] = bool(gate_int is None or gate_int > C.TOL["synthetic_rest_confound_m"])
    if res["testable"]:
        if res["confounded"]:
            a, b = pr["isolated"]["head_max_abs_coord_m"]["v1"], pr["isolated"]["head_max_abs_coord_m"]["p2"]
            res["contrast_metric"] = "isolated head max abs coord (confounded gate)"
        else:
            a, b = pr["v1_subframe_max_m"], pr["p2_subframe_max_m"]
            res["contrast_metric"] = "primary subframe max abs coord"
        if spec["source_interpolation"] == "CONSTANT":
            res["contrast_ok"] = bool(a > 0 and b <= C.TOL["material_reduction_ratio"]*a)
        else:
            res["contrast_ok"] = bool(b >= C.TOL["smooth_harm_ratio"]*a)
        res["contrast_values"] = {"v1": a, "p2": b}
    else:
        res["contrast_ok"] = False
    return res


def analyze(o):
    decl_bytes = Path(o.declaration).read_bytes()
    if hashlib.sha256(decl_bytes).hexdigest() != o.declaration_sha:
        raise SystemExit("STOP: declaration hash mismatch")
    decl = json.loads(decl_bytes)
    out = Path(o.out_dir)
    out.mkdir(parents=True, exist_ok=False)
    p1, p2, w107, w106 = Path(o.probe_dir), Path(o.repeat_dir), Path(o.w107), Path(o.w106)
    rep = {"schema": C.SCHEMA + ":analysis", "declaration_sha256": o.declaration_sha, "cases": {}, "controls": {}}
    for cid, case in decl["cases"].items():
        rep["cases"][cid] = analyze_real(cid, case, p1, p2, w107)
    for ctrl, spec in decl["synthetic_controls"].items():
        rep["controls"][ctrl] = analyze_control(ctrl, spec, p1, p2)
    ver = json.loads(Path(o.verify).read_text())
    unchanged = {}
    for tag, b, rec in (("w107", w107, ver["w107_files"]), ("w106", w106, ver["w106_files"])):
        man = json.loads((b/"evidence_manifest.json").read_text())
        chk = check_manifest(b, man, MANIFEST_LOG_EXCEPTIONS[tag])
        unchanged[tag] = {"ok": chk["ok"], "bad": chk["bad"], "files": chk["files"]}
    unchanged["w107_manifest_sha"] = sha256(w107/"evidence_manifest.json") == W107_MANIFEST_SHA
    unchanged["w106_manifest_sha"] = sha256(w106/"evidence_manifest.json") == W106_MANIFEST_SHA
    rep["originals_unchanged"] = unchanged
    shared = git("status", "--porcelain", "--", *SHARED_PATHS).splitlines()
    rep["shared_tracked_changes"] = [s for s in shared if not s.startswith("??")]
    real = rep["cases"]
    ctr = rep["controls"]
    pre = {"preflight_ok": ver["ok"] and not rep["shared_tracked_changes"],
           "sources_ok": all(real[c]["identity_ok"] for c in C.REAL_CASES),
           "deterministic": all(r["deterministic"] for r in list(real.values()) + [c for c in ctr.values() if "deterministic" in c]),
           "originals_unchanged": all(unchanged[k]["ok"] for k in ("w107", "w106")) and unchanged["w107_manifest_sha"]
           and unchanged["w106_manifest_sha"] and all(real[c]["actions_unmodified"] for c in C.REAL_CASES),
           "synthetic_valid": all(ctr[k].get("valid") for k in C.SYN_CONTROLS),
           "p0": {c: real[c]["P0"]["ok"] for c in C.REAL_CASES},
           "boundary": {c: real[c]["boundary"] for c in C.REAL_CASES},
           "p2": {c: {"testable": real[c]["P2"]["status"] == "TESTED",
                      "keydata_ok": bool(real[c]["P2"].get("keydata_ok")),
                      "integer_unchanged": bool(real[c]["P2"].get("integer_unchanged")),
                      "material": bool(real[c]["P2"].get("material")),
                      "sufficient": bool(real[c]["P2"].get("sufficient"))} for c in C.REAL_CASES},
           "synthetic": {k: {"testable": bool(ctr[k].get("testable")), "contrast_ok": bool(ctr[k].get("contrast_ok")),
                             "confounded": bool(ctr[k].get("confounded"))} for k in C.SYN_CONTROLS}}
    case, why = C.decide_case(pre)
    rep["decision_inputs"] = pre
    rep["CASE"], rep["CASE_reason"] = case, why
    s = wjson(out/"analysis_report.json", rep)

    def brief(r):
        b = {"boundary": r["boundary"].get("boundary")}
        for v in C.EXPORT_PATHS:
            e = r["paths"].get(v, {}).get("errors")
            if e:
                b[v] = {k: e[k].get("max_abs_coord_m") for k in ("integer", "phase_0.25", "phase_0.50", "phase_0.75",
                                                                 "subframe_transition", "subframe_quiet")}
        e = r["P2"].get("errors")
        if e:
            b["P2"] = {k: e[k].get("max_abs_coord_m") for k in ("integer", "phase_0.25", "phase_0.50", "phase_0.75",
                                                                "subframe_transition", "subframe_quiet")}
            b["P2_flags"] = {k: r["P2"].get(k) for k in ("integer_unchanged", "keydata_ok", "material", "sufficient",
                                                         "ratio_p2_over_v1", "imported_key_times_integer")}
        return b
    summary = {"CASE": case, "reason": why, "sha256": s,
               "p0": {c: real[c]["P0"] for c in C.REAL_CASES},
               "cases": {c: brief(real[c]) for c in C.REAL_CASES},
               "controls": {k: brief(ctr[k]) | {x: ctr[k].get(x) for x in ("valid", "testable", "confounded",
                                                                           "contrast_ok", "contrast_values")}
                            for k in C.SYN_CONTROLS if "boundary" in ctr[k]},
               "decision_inputs": {k: v for k, v in pre.items() if k not in ("boundary",)}}
    print(json.dumps(summary, indent=1, default=str))
    return 0


# --- manifest -----------------------------------------------------------------------------------------------------
def manifest(o):
    base = Path(o.base)
    files = {}
    for p in sorted(base.rglob("*")):
        if p.is_file() and p.name != "evidence_manifest.json":
            files[p.relative_to(base).as_posix()] = {"sha256": sha256(p), "bytes": p.stat().st_size}
    w107, w106 = Path(o.w107), Path(o.w106)
    unchanged = {}
    for tag, b in (("w107", w107), ("w106", w106)):
        chk = check_manifest(b, json.loads((b/"evidence_manifest.json").read_text()), MANIFEST_LOG_EXCEPTIONS[tag])
        unchanged[tag] = chk["ok"]
    unchanged["w107_manifest_sha"] = sha256(w107/"evidence_manifest.json") == W107_MANIFEST_SHA
    unchanged["w106_manifest_sha"] = sha256(w106/"evidence_manifest.json") == W106_MANIFEST_SHA
    rep = {"schema": C.SCHEMA + ":manifest", "starting_head": o.starting_head, "declaration_sha256": o.declaration_sha,
           "originals_unchanged": unchanged, "all_originals_unchanged": all(unchanged.values()), "files": files}
    s = wjson(base/"evidence_manifest.json", rep)
    print(s, len(files), rep["all_originals_unchanged"])
    return 0 if rep["all_originals_unchanged"] else 2


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=("verify", "declare", "analyze", "manifest"))
    for name in ("base", "w106", "w107", "verify", "options", "lock", "blender", "starting_head", "branch",
                 "declaration", "declaration_sha", "probe_dir", "repeat_dir", "out_dir"):
        ap.add_argument("--"+name.replace("_", "-"))
    o = ap.parse_args()
    return {"verify": verify, "declare": declare, "analyze": analyze, "manifest": manifest}[o.mode](o)


if __name__ == "__main__":
    sys.exit(main())
