#!/usr/bin/env python3
"""W112 host-side stages: FBX within-property interpolation recoverability boundary (evidence only).

  verify    W111 COMPLETE marker, W111 / W110 declaration and evidence-manifest identity and every manifest file,
            every W106-W109 manifest file, frozen exporter / add-on / Blender exe vs the W111 declaration, reused-script
            SHAs vs the W111 declaration, worker lock (Batch 112 RUNNING), Git state, no other Blender process; byte
            copies of the W110 inputs into <base>/results/inputs
  declare   case matrix, constructions, producer rule, FBX-level instrument, privileged assignment spaces and their
            exact-minimum proof, source-free candidate rule and eligibility, conditions, grids, metrics, tolerances,
            controls, CASE rule, repeat, resource cap -> declaration.json + SHA (outcomes_read = false)
  encode    PRODUCER (source-informed): per-key flags from the frozen rule; obstruction report; per-key run-length
            re-encoding of KeyAttrFlags only (keys untouched) with validity checks; the FBX-level evaluation files; the
            negative-control files; source-free consumer_input.json and source-known oracle_input.json
  gate      import-level bound per case -> candidate eligibility -> source-free candidate_input.json
  analyze   bounds, conditions, candidate, coverage, isolation, integrity, controls, instrument, repeat, originals
            -> analysis_report.json + CASE
  manifest  hashes of every output plus a re-check that no W106-W111 original changed

No production exporter/importer, rig, Action original, scene setting or historical artifact is modified.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT/"scripts"))
import w112_fbx_within_property_contract as C  # noqa: E402
import w111_fbx_semantic_identity as S  # noqa: E402
import w110_fbx_metadata_recovery as H10  # noqa: E402  (W110 host helpers: manifests; read-only reuse)
import w110_fbx_consumer_adapter as AD  # noqa: E402  (pure FBX-side map)
import w108_fbx_subframe_contract as C8  # noqa: E402
from w108_fbx_subframe import check_manifest, git, wjson  # noqa: E402

sha256 = C.sha256_file
W111_WORKLOG = "docs/111_WORKLOG_FBX_SEMANTIC_REPEATABILITY_UID_PATH_VARIATION.md"
W111_DECLARATION_SHA = "0ffa9d3cb51e45d31316ce1e429f0bb0b8d02951e7281601cd7c721bf4c29d89"
W111_MANIFEST_SHA = "11ea4de969429bb4dcf2ab3fb1a013942d7b345123a6be9f66363fa5b5927a8f"
W110_DECLARATION_SHA = "223b8c81ab345507ee5e60cbceca70fdad2e03bf97236afe87d94932a3ec049d"
W110_MANIFEST_SHA = "e46329f5f2e2d11b016f06ffd93f3a3beb5ffe39579f5de9ba26e318f8215fd4"
W110_OPTIONS_SHA_PREFIX = "7d1a5472"
W112_SCRIPTS = ("scripts/w112_fbx_within_property_contract.py", "scripts/w112_fbx_perkey_candidate.py",
                "scripts/w112_fbx_within_property_blender.py", "scripts/w112_fbx_within_property.py")
W112_NEW = ("scripts/w112_", "tests/test_w112_", "docs/112_WORKLOG_")
INPUT_FILES = ("syn_step_source.blend", "syn_step_V1.fbx", "syn_smooth_source.blend", "syn_smooth_V1.fbx",
               "complete_control_source.blend", "complete_control_V1.fbx", "w106_synthetic_n0.blend")
CONSUMER_KEYS = {"schema", "blender_version", "cases"}
CANDIDATE_KEYS = {"schema", "blender_version", "cases", "controls"}
BANNED_CONSUMER_TOKENS = (".blend", "census", "declaration", "source", ".npz", "produce", "oracle", "efbx")
NC_SPECS = {
    "NC0_base": ("BASELINE", "unmodified metadata FBX of the NC base case (reference for the controls below)"),
    "NC1_interval_flag": ("LOCALIZED", "every target-group curve: the flag of one held transition key -> LINEAR; the "
                                       "candidate must change exactly that key and the motion only in that interval "
                                       "(> 1 cm harm there, identical elsewhere)"),
    "NC1b_one_axis_flag": ("UNRESOLVED", "the same key flipped on ONE target curve only: per-key axis disagreement; "
                                         "the target group must stay unresolved and unchanged"),
    "NC2_key_time": ("UNRESOLVED", "one target curve: middle KeyTime + 1 tick; the group must stay unresolved"),
    "NC3_key_value": ("DETECTED_BY_EVALUATION", "one target curve: middle KeyValueFloat + 1.0; the candidate plan is "
                                                "unchanged (values are not metadata) and the evaluated motion departs "
                                                "from the source by > 1 mm at that key"),
    "NC4_ambiguous_name": ("UNRESOLVED_OR_IMPORT_FAILED", "the target LimbNode Model renamed to the sibling LimbNode's "
                                                          "name: ambiguous correspondence; both items' groups must stay "
                                                          "unresolved and unchanged"),
}


class _NS:
    def __init__(self, **k):
        self.__dict__.update(k)


# --- verify -------------------------------------------------------------------------------------------------------
def verify(o):
    base = Path(o.base)
    out = base/"results"/"verify"
    out.mkdir(parents=True, exist_ok=False)
    rep = {"schema": C.SCHEMA + ":verify", "checks": {}, "stop": []}
    wl = (ROOT/W111_WORKLOG).read_text(encoding="utf-8")
    rep["checks"]["w111_complete_marker"] = wl.rstrip().endswith("AUTOMATION_BATCH_STATUS: COMPLETE\nBATCH_NUMBER: 111")
    w111, w110 = Path(o.w111), Path(o.w110)
    for tag, b, dsha, msha in (("w111", w111, W111_DECLARATION_SHA, W111_MANIFEST_SHA),
                               ("w110", w110, W110_DECLARATION_SHA, W110_MANIFEST_SHA)):
        m_s, d_s = sha256(b/"evidence_manifest.json"), sha256(b/"results/declare/declaration.json")
        rep["checks"][f"{tag}_manifest_sha"] = {"value": m_s, "ok": m_s == msha}
        rep["checks"][f"{tag}_declaration_sha"] = {"value": d_s, "ok": d_s == dsha}
        man = json.loads((b/"evidence_manifest.json").read_text())
        chk = check_manifest(b, man, "manifest_stdout.txt")
        rep[f"{tag}_manifest"] = {"files": chk["files"], "bad": chk["bad"], "exception": chk["exception"],
                                  "all_originals_unchanged": man.get("all_originals_unchanged")}
        rep["checks"][f"{tag}_files_intact"] = chk["ok"] and bool(man.get("all_originals_unchanged"))
    _, man = H10._manifests(o)
    rep["manifests"] = man
    for tag in man:
        rep["checks"][f"{tag}_manifest_sha"] = man[tag]["manifest_sha_ok"]
        rep["checks"][f"{tag}_files_intact"] = man[tag]["files_ok"]
    d11 = json.loads((w111/"results/declare/declaration.json").read_text())
    scr = {p: sha256(ROOT/p) for p in d11["scripts_sha256"]}
    rep["reused_scripts_sha256"] = scr
    rep["checks"]["reused_scripts_equal_w111_declaration"] = all(scr[p] == d11["scripts_sha256"][p] for p in scr)
    addon = Path(o.addon_dir)
    a_sha = {p.name: sha256(p) for p in sorted(addon.glob("*.py"))}
    rep["fbx_addon_source_sha256"] = a_sha
    rep["checks"]["fbx_addon_unchanged_vs_w111"] = a_sha == d11["environment"]["fbx_addon_source_sha256"]
    rep["blender_exe_sha256"] = sha256(o.blender)
    rep["checks"]["blender_exe_equals_w111"] = rep["blender_exe_sha256"] == d11["environment"]["blender_exe_sha256"]
    rep["environment_w111"] = {k: d11["environment"][k] for k in ("blender_version", "blender_build_hash",
                                                                 "fbx_addon_version")}
    head = git("rev-parse", "HEAD")
    status = git("status", "--porcelain", "--untracked-files=all").splitlines()
    tracked = [s for s in status if not s.startswith("??")]
    untracked = [s[3:] for s in status if s.startswith("??")]
    rep["git"] = {"head": head, "branch": git("branch", "--show-current"), "origin": git("rev-parse", "origin/" + o.branch),
                  "tracked_changes": tracked, "untracked": untracked}
    rep["checks"]["starting_head"] = {"value": head, "ok": head == o.starting_head}
    rep["checks"]["branch"] = rep["git"]["branch"] == o.branch
    rep["checks"]["origin_equals_head"] = rep["git"]["origin"] == head
    rep["checks"]["working_tree_only_new_w112_files"] = not tracked and all(u.startswith(W112_NEW) for u in untracked)
    lk = json.loads(Path(o.lock).read_text()) if Path(o.lock).exists() else {}
    rep["worker_lock"] = lk
    rep["checks"]["worker_lock_batch_112_running"] = lk.get("batch") == 112 and lk.get("state") == "RUNNING" \
        and lk.get("start_commit") == o.starting_head
    tl = subprocess.run(["tasklist", "/FO", "CSV", "/NH"], capture_output=True, text=True, errors="replace").stdout
    blenders = [ln for ln in tl.splitlines() if ln.lower().startswith('"blender')]
    rep["other_blender_processes"] = blenders
    rep["checks"]["no_other_blender_process"] = not blenders
    for k, v in rep["checks"].items():
        if not (v if isinstance(v, bool) else v["ok"]):
            rep["stop"].append(k)
    m10 = json.loads((w110/"evidence_manifest.json").read_text())
    inputs = base/"results"/"inputs"
    inputs.mkdir(exist_ok=False)
    copies = {}
    for name in INPUT_FILES:
        rel = f"results/inputs/{name}"
        want = m10["files"].get(rel, {}).get("sha256")
        dst = inputs/name
        shutil.copyfile(w110/rel, dst)
        copies[name] = {"origin": f"w110:{rel}", "path": str(dst), "sha256": sha256(dst), "w110_manifest_sha256": want}
        copies[name]["ok"] = want is not None and copies[name]["sha256"] == want
        if not copies[name]["ok"]:
            rep["stop"].append(f"copy:{name}")
    rep["inputs"] = copies
    rep["ok"] = not rep["stop"]
    wjson(out/"verify_report.json", rep)
    print(json.dumps({"ok": rep["ok"], "stop": rep["stop"]}))
    return 0 if rep["ok"] else 2


# --- declare ------------------------------------------------------------------------------------------------------
def _frames(cid):
    return (1, 70) if C.CASES[cid]["family"] == "RR" else (1, 25)


def declare(o):
    base, w110, w111 = Path(o.base), Path(o.w110), Path(o.w111)
    out = base/"results"/"declare"
    out.mkdir(parents=True, exist_ok=False)
    ver = json.loads(Path(o.verify).read_text())
    if not ver["ok"]:
        raise SystemExit("STOP: verify failed")
    d10 = json.loads((w110/"results/declare/declaration.json").read_text())
    d11 = json.loads((w111/"results/declare/declaration.json").read_text())
    opt_sha = sha256(o.options)
    inp = {}
    for name, rec in ver["inputs"].items():
        key = name.rsplit(".", 1)[0]
        inp[key] = {"path": rec["path"], "sha256": rec["sha256"], "origin": rec["origin"]}
    cases = {}
    for cid, c in C.CASES.items():
        fs, fe = _frames(cid)
        rec = dict(c)
        rec.update({"id": cid, "stack": "root|" + c["action"], "frame_start": fs, "frame_end": fe,
                    "times": C.grid(fs, fe), "within_property": c["spec"] in C.WITHIN_SPECS})
        if c["family"] is not None:
            fam = C.FAMILIES[c["family"]]
            rec["target"] = {"item": C.TARGET_BONE, "group": fam["group"], "fbx_property": fam["fbx_property"],
                             "source_path": fam["source_path"], "imported_fcurves": fam["n"]}
            rec["spec_text"] = C.SPECS[c["spec"]]
        else:
            rec["target"] = None
        if c["origin"] == "w110_input":
            rec["w110_case"] = d10["cases"][cid]["id"]
            rec["expected_source_action_digest"] = d10["cases"][cid]["expected_source_action_digest"]
        cases[cid] = rec
    scripts = {p: sha256(ROOT/p) for p in W112_SCRIPTS}
    scripts.update({p: sha256(ROOT/p) for p in d11["scripts_sha256"]})
    decl = {
        "schema": C.SCHEMA + ":declaration", "outcomes_read": False,
        "note": "Frozen before any W112 build, export, encode, import, adapter, oracle or candidate run. Exploratory "
                "construction debugging was confined to C:/Users/dna10/animcv-w112-explore on non-declared plumbing "
                "and is not evidence.",
        "starting_head": o.starting_head, "verify_sha256": sha256(o.verify), "options_sha256": opt_sha,
        "options_equals_w110": opt_sha == sha256(w110/"results/options.json"),
        "history": {"w110_declaration_sha256": W110_DECLARATION_SHA, "w110_manifest_sha256": W110_MANIFEST_SHA,
                    "w111_declaration_sha256": W111_DECLARATION_SHA, "w111_manifest_sha256": W111_MANIFEST_SHA,
                    "w110_case": "D (immutable)", "w111_case": "A bounded (immutable)"},
        "environment": {"blender_version": ver["environment_w111"]["blender_version"],
                        "blender_build_hash": ver["environment_w111"]["blender_build_hash"],
                        "fbx_addon_version": ver["environment_w111"]["fbx_addon_version"],
                        "blender_exe": str(o.blender), "blender_exe_sha256": ver["blender_exe_sha256"],
                        "fbx_addon_source_sha256": ver["fbx_addon_source_sha256"],
                        "launch": "-b --factory-startup -t 2 --python-exit-code 1 (W110 launch; no system env)"},
        "scripts_sha256": scripts,
        "presentation_only_scripts_not_bound": ["scripts/w112_fbx_within_property_figures.py"],
        "inputs": inp,
        "exporter": {"v0_kwargs": d10["exporter"]["v0_kwargs"],
                     "v1": "bpy.ops.export_scene.fbx(V0 kwargs + bake_anim_simplify_factor=0.0) via W110 export_v1 "
                           "(W107 do_export 'V1'); one export per built case per run; historical controls reuse "
                           "their frozen W107/W108 V1 FBX byte-for-byte"},
        "cases": cases,
        "families": C.FAMILIES, "specs": C.SPECS, "target_bone": C.TARGET_BONE,
        "within_cases": list(C.WITHIN_CASES),
        "uniform_controls": {"linear": C.UNIFORM_LINEAR_OF, "held": C.UNIFORM_HELD_OF},
        "constructions": {
            "synthetic": "scratch copy of the W110 input copy of the W106 synthetic N0 file; 'SYN baseline' (9 "
                         "quaternion keys per bone at frames 1,4,...,25; sibling misses 13; all CONSTANT) is duplicated "
                         "with Action.copy() and assigned with the original slot identifier; only keyframe "
                         "interpolation is written (foreach_set); rig, fps (30), frame range (1-25), times, values, "
                         "handles untouched; every non-target curve stays CONSTANT",
            "location_keys": {"rule": "L family only, SEPARATELY DECLARED KEY CONSTRUCTION (not an interpolation "
                                      "intervention): three new F-curves pose.bones['upperarm_l'].location[0..2] with "
                                      "keys at frames 1+3k (k = 0..8) and the values below (bone-local metres), "
                                      "inserted with keyframe_points.insert(FAST); identical key set in every L case; "
                                      "only their interpolation differs between L cases",
                              "values": [list(C.loc_value(k)) for k in range(9)]},
            "real_rig": "scratch copy of the W110 input copy of complete_control (W91 complete_control FIXED: 24 "
                        "quaternion keys per channel at frames 1,4,...,70, all CONSTANT); the Action is duplicated and "
                        "only the interpolation of pose.bones['upperarm_l'].rotation_quaternion[0..3] is written; "
                        "SYNTHETIC INTERVENTION ON A REAL RIG, reported separately from real-scene replay",
            "historical_controls": "syn_step, syn_smooth and complete_control reuse the W110 input .blend and V1 FBX "
                                   "byte copies (unmodified)"},
        "producer": {
            "rule": "PRODUCER ONLY (source-informed).  Per selected-take FBX AnimationCurve mapped by the W110 FBX map "
                    "to (item, Lcl property) and from there to the source (item, transform group): FBX key j (frame "
                    "f_j = frame_start + (KeyTime_j - KeyTime_0)/ticks_per_frame) receives the interpolation in force "
                    "on (f_j, f_j + dt) in every source F-curve of that group (interpolation of the last source key at "
                    "or before f_j; constant extrapolation before the first key).  All source axes agree -> that class "
                    "(CONSTANT -> 0x6002, LINEAR -> 0x6104).  Source axes disagree -> OBSTRUCTED: an FBX axis of the "
                    "property is not a source axis (quaternion -> Euler, rest-frame / parent-frame mixing), so no "
                    "per-FBX-axis semantics are defined by the source and the case STOPS at the encoding layer.  No "
                    "source F-curve -> UNDRIVEN (flags untouched).  Anything else -> UNSUPPORTED (STOP).",
            "encoding": "KeyAttrFlags / KeyAttrDataFloat / KeyAttrRefCount of a curve whose per-key classes are not "
                        "all LINEAR are rewritten as run-length runs (one flag + the original 4-float data group + "
                        "the run length per run); KeyTime, KeyValueFloat and every other element, take and the rig "
                        "are untouched; the tree is re-serialised with the W111 minimal writer",
            "validity": ["re-read meta tree equals the control tree except the three KeyAttr arrays of the re-encoded "
                         "curves", "every selected-take KeyTime / KeyValueFloat bitwise equal", "decoded per-key "
                         "classes equal the producer classes on every selected-take curve", "RefCount sums equal key "
                         "counts", "W111 canonical comparison control vs meta differs in exactly the re-encoded curve "
                         "records (0 header / connection differences)"]},
        "consumer": {"adapter": "scripts/w110_fbx_consumer_adapter.py (unchanged W110 file)",
                     "conditions": {"control_U": "untouched V1 import", "control_A": "W110 adapter on V1",
                                    "control_K": "blanket CONSTANT on the selected imported Action (harness control)",
                                    "meta_U": "metadata FBX untouched (metadata-only stock import)",
                                    "meta_A": "W110 adapter on the metadata FBX"},
                     "isolation": "consumer_input.json structural whitelist; W110 audit hook around recover()"},
        "oracle": {
            "role": "PRIVILEGED SOURCE-KNOWN FEASIBILITY BOUNDS -- never a runtime consumer, never source-free recovery",
            "import_level": "untouched control V1 import; every non-target (item, group) set to its source uniform "
                            "interpolation (CONSTANT / LINEAR; undriven untouched); the target group's imported "
                            "F-curves receive EVERY combination in {CONSTANT, LINEAR}^n (n = 4 quaternion / 3 "
                            "location), uniform over keys; sampled per combination",
            "fbx_level": "FBX-level evaluation instrument: scratch evaluation FBX per combination in {CONSTANT, "
                         "LINEAR}^3 of the target FBX curves (d|X, d|Y, d|Z of the target Lcl property); every "
                         "selected-take curve is re-keyed at probe phases of every interval carrying the FBX-level "
                         "value (hold of key j, or lerp between existing keys j and j+1); two files per assignment "
                         "(part a: phases 0/.25/.5/.75 + final key; part b: phase .99 only -- Blender merges keys "
                         "closer than 0.01 frame); non-target curves use the producer classes (undriven -> LINEAR); "
                         "imported unchanged; each part sampled only at its own key times and the parts merged.  KEYS ARE ADDED ONLY IN THIS EVALUATION FILE; it is never a "
                         "product, metadata or consumer input",
            "exact_minimum": "the error at a probe inside (k_j, k_j+1) depends only on key j's interpolation of the "
                             "target curves (CONSTANT / LINEAR ignore handles; every other curve fixed), so the "
                             "minimum over the product space of per-interval assignments equals the per-interval "
                             "minimum over the sampled uniform combinations (ties -> first label in declared order); "
                             "integer samples are key times (assignment-independent, checked)",
            "space_limits": "CUBIC / tangent data, CONSTANT-next, BEZIER handles, key insertion or value change are "
                            "outside both declared spaces",
            "derived_diagnostics": ["source-aligned per-axis assignment (same index; import level only, same "
                                    "representation)", "group-uniform per interval (all C or all L: the space a "
                                    "per-key group-level consumer can reach)"]},
        "candidate": {
            "module": "scripts/w112_fbx_perkey_candidate.py", "rule": "see module docstring (frozen)",
            "eligibility": "a case is given to the candidate only if its metadata encoding is VALID and its "
                           "import-level bound is within 1 mm; negative controls are always run (behaviour evidence "
                           "only, never recovery evidence)",
            "isolation": "candidate_input.json structural whitelist; W110 audit hook around recover(); static "
                         "import / token guard (test)"},
        "negative_controls": {"base_case": C.NC_BASE_CASE, "controls": NC_SPECS,
                              "nc1_key": "first FBX key j whose producer class is CONSTANT on the target group and "
                                         "whose next frame f_j + 1 is a source key frame (a held transition)"},
        "positive_controls": {
            "uniform_held_recovered_by_w110": "syn_step and complete_control: control_U harms > 1 cm, meta_A within 1 mm",
            "smooth_harmed_by_blanket_constant": "syn_smooth: control_K harms > 1 cm, meta_A within 1 mm",
            "importer_ignores_flags": "meta_U equals control_U (<= 1e-9 m) on every encoded case",
            "adapter_noop_on_linear_flags": "control_A equals control_U (<= 1e-9 m) and changes 0 F-curves",
            "candidate_reduces_to_w110_on_uniform": "on eligible uniform cases the candidate changes the same F-curves "
                                                    "as W110 meta_A and its motion equals meta_A (<= 1e-9 m)",
            "historical_replay": "control_U of syn_step / syn_smooth / complete_control equals the W110 consume "
                                 "control_U samples at common times (<= 1e-9 m)"},
        "instrument_checks": ["every fbx_<label> integer sample equals control_U (<= 1e-9 m)",
                              "every evaluation import has a key at every probe of its part (|probe - key| <= "
                              "1e-4 frame) and exactly as many target keys as probes", "import-level integer samples identical across combinations "
                              "(<= 1e-9 m)", "oracle harness changed only interpolation", "uniform-held family "
                              "members (syn_step, syn_L_uniC, complete_control): FBX-level bound within 1 mm"],
        "samples": {"grid": "f + " + str(list(C.PHASES)) + " for every interval [f, f+1) of the frame range, plus "
                            "frame_end (integer, quarter frames and the near-boundary f + 0.99 everywhere)"},
        "metrics": "max |coordinate| world head / tail error over the common bones (W108 gate form); per integer / "
                   "per target interval class (held / smooth / axis_mixed); near-boundary f + 0.99 of transition "
                   "intervals; per-interval best assignment",
        "tolerances": C.TOL,
        "gates": {"bound_ok": "privileged bound <= 1 mm at every probe", "sf_ok": "candidate eligible, every probe "
                  "<= 1 mm, 0 unresolved groups, 0 unmapped imported F-curves, clean isolation and integrity",
                  "harm": "> 1 cm", "identity": "<= 1e-9 m"},
        "case_rule": C.decide_case.__doc__,
        "repeat": "two independent full runs (fresh build, export, encode, consume, oracle, gate, candidate); every "
                  "decision input must be equal and every reported maximum equal within 1e-9 m; built FBX compared "
                  "under the W111 canonicaliser (UIDs differ by design)",
        "stop_conditions": ["remote / branch / lock conflict", "missing or changed historical artifact",
                            "invalid construction or export", "encoding validity failure", "source-free isolation "
                            "violation", "non-target change", "masked negative control", "post-hoc gate change"],
        "resource_cap": {"blender_runs": "2 x (produce, consume, oracle, candidate)", "wall_s_per_run": 1800,
                         "disk_gb": 6, "gpu": False, "network": False},
        "hypothesis": "H112: for within-property axis- or interval-mixed interpolation on one item / Lcl property, "
                      "the unchanged Blender 5.1 import followed by CONSTANT/LINEAR interpolation-only edits informed "
                      "by FBX metadata recovers the evaluated source motion within 1 mm.",
    }
    s = wjson(out/"declaration.json", decl)
    print(s)
    return 0


def _decl(path, sha):
    if sha256(path) != sha:
        raise SystemExit("STOP: declaration hash mismatch")
    return json.loads(Path(path).read_text())


# --- encode (producer) --------------------------------------------------------------------------------------------
def _sel_curves(root, fmap):
    objs = C8._child(root, b"Objects")
    by_uid = {int(e.props[0]): e for e in objs.elems if e.id == b"AnimationCurve"}
    return {r["uid"]: by_uid[r["uid"]] for r in fmap["curves"]}


def _set_child(e, name, value, t):
    for c in e.elems:
        if c.id == name:
            c.props[0] = value
            c.props_type = t
            return
    raise KeyError(name)


def _keyattr(e):
    return [np.asarray(C8._child(e, n).props[0]) for n in (b"KeyAttrFlags", b"KeyAttrDataFloat", b"KeyAttrRefCount")]


def encode_meta(root, fmap, perkey):
    """Run-length re-encoding of the per-key classes; returns (meta tree, changed uids, intended per-key flags)."""
    m = S.mcopy(root)
    mc = _sel_curves(m, fmap)
    changed, intended = [], {}
    for uid, rec in perkey.items():
        e = mc.get(uid)
        if e is None:
            continue
        flags, data, refs = _keyattr(e)
        nk = len(C8._child(e, b"KeyTime").props[0])
        orig = C.expand_flags(flags, refs)
        if rec["kind"] != "PERKEY":
            intended[uid] = orig
            continue
        want = [C.FLAG[c] for c in rec["per_key"]]
        intended[uid] = want
        if want == orig:
            continue
        if len(flags) != 1 or int(flags[0]) != C.FLAG[C.LINEAR] or len(data) != 4 or nk != len(want):
            raise ValueError(f"unexpected original KeyAttr layout on curve {uid}")
        runs = C.rle(want)
        _set_child(e, b"KeyAttrFlags", np.asarray([f for f, _ in runs], dtype=np.int32), b"i")
        _set_child(e, b"KeyAttrDataFloat", np.tile(np.asarray(data, dtype=np.float32), len(runs)), b"f")
        _set_child(e, b"KeyAttrRefCount", np.asarray([n for _, n in runs], dtype=np.int32), b"i")
        changed.append(uid)
    return m, changed, intended


def masked_equal(a, b, masked_uids):
    """Tree equality with the three KeyAttr arrays of ``masked_uids`` masked (ids, types, values, order)."""
    masked = set(masked_uids)
    bad = []

    def walk(x, y, path, in_masked):
        if bytes(x.id) != bytes(y.id) or bytes(x.props_type) != bytes(y.props_type) or len(x.elems) != len(y.elems):
            bad.append(path)
            return
        skip = in_masked and bytes(x.id) in (b"KeyAttrFlags", b"KeyAttrDataFloat", b"KeyAttrRefCount")
        if not skip:
            for t, u, v in zip(bytes(x.props_type).decode(), x.props, y.props):
                if isinstance(u, np.ndarray) or isinstance(v, np.ndarray):
                    if np.asarray(u).tobytes() != np.asarray(v).tobytes():
                        bad.append(path)
                        return
                elif u != v:
                    bad.append(path)
                    return
        nm = bytes(x.id) == b"AnimationCurve" and len(x.props) > 0 and int(x.props[0]) in masked
        for cx, cy in zip(x.elems, y.elems):
            walk(cx, cy, path + "/" + bytes(cx.id).decode(errors="replace"), nm)
    for x, y in zip(a.elems, b.elems):
        walk(x, y, bytes(x.id).decode(), False)
    return not bad and len(a.elems) == len(b.elems), bad[:10]


def build_efile(root, fmap, sem_of_uid, part):
    """FBX-level evaluation instrument: every selected-take curve densified at the part's probe phases (hold / lerp)."""
    phases, end = C.EPARTS[part]
    m = S.mcopy(root)
    mc = _sel_curves(m, fmap)
    for uid, e in mc.items():
        t = np.asarray(C8._child(e, b"KeyTime").props[0], dtype=np.int64)
        v = np.asarray(C8._child(e, b"KeyValueFloat").props[0], dtype=np.float32)
        tt, vv = C.densify(t, v, sem_of_uid[uid], phases, end)
        flags, data, refs = _keyattr(e)
        _set_child(e, b"KeyTime", tt, b"l")
        _set_child(e, b"KeyValueFloat", vv, b"f")
        _set_child(e, b"KeyAttrFlags", np.asarray([C.FLAG[C.LINEAR]], dtype=np.int32), b"i")
        _set_child(e, b"KeyAttrDataFloat", np.asarray(data[:4], dtype=np.float32), b"f")
        _set_child(e, b"KeyAttrRefCount", np.asarray([len(tt)], dtype=np.int32), b"i")
    return m


def directives(fmap, census):
    """Per FBX model name | group: the source uniform interpolation (oracle non-target assignment)."""
    groups = C.census_groups(census)
    out = {}
    for m in fmap["models"].values():
        for g in ("location", "rotation", "scale"):
            if m["type"] == "Null":
                gk = f"object|{g}" if m["name"] == census["armature_object"] else None
            else:
                gk = f"bone:{m['name']}|{g}"
            src = groups.get(gk, []) if gk else []
            s = sorted({i for c in src for i in c["interpolation"]})
            out[f"{m['name']}|{g}"] = "UNDRIVEN" if not s else s[0] if len(s) == 1 else "MIXED:" + "/".join(s)
    return out


def _target_uids(fmap, target):
    t = {r["channel"]: r["uid"] for r in fmap["curves"] if r["status"] == "OK" and r["model_name"] == target["item"]
         and r["fbx_property"] == target["fbx_property"]}
    return [t[i] for i in range(3)] if sorted(t) == [0, 1, 2] else None


def _write(tree, path, version):
    Path(path).write_bytes(S.write_fbx(tree, version))
    return {"path": str(path), "sha256": sha256(path), "bytes": Path(path).stat().st_size}


def nc_files(d, root, meta_tree, fmap, perkey, case, census, version, fps):
    """Negative-control FBX files built from the NC base case's metadata tree."""
    tgt = _target_uids(fmap, case["target"])
    frames, _ = C.key_frames(np.asarray(C8._child(_sel_curves(root, fmap)[tgt[0]], b"KeyTime").props[0]),
                             case["frame_start"], fps)
    src_keys = {f for c in census["curves"] if c["data_path"] == case["target"]["source_path"] for f in c["frames"]}
    per = perkey[tgt[0]]["per_key"]
    j = next((i for i in range(len(per) - 1) if per[i] == C.CONSTANT and frames[i] + 1 in src_keys), None)
    out = {"nc1_key": j, "nc1_frame": frames[j] if j is not None else None, "files": {}}
    if j is None:
        return out
    mid = len(frames)//2
    model_uid = next(r["model"] for r in fmap["curves"] if r["uid"] == tgt[0])
    sib = [u for u, m in fmap["models"].items() if m["name"] == "sibling"]
    for name in NC_SPECS:
        t = S.mcopy(meta_tree)
        mc = _sel_curves(t, fmap)
        rec = {"expect": NC_SPECS[name][0]}
        if name in ("NC1_interval_flag", "NC1b_one_axis_flag"):
            for u in (tgt if name == "NC1_interval_flag" else tgt[:1]):
                e = mc[u]
                flags, data, refs = _keyattr(e)
                pk = C.expand_flags(flags, refs)
                pk[j] = C.FLAG[C.LINEAR]
                runs = C.rle(pk)
                _set_child(e, b"KeyAttrFlags", np.asarray([f for f, _ in runs], dtype=np.int32), b"i")
                _set_child(e, b"KeyAttrDataFloat", np.tile(np.asarray(data[:4], dtype=np.float32), len(runs)), b"f")
                _set_child(e, b"KeyAttrRefCount", np.asarray([n for _, n in runs], dtype=np.int32), b"i")
            rec.update({"key": j, "frame": frames[j]})
        elif name == "NC2_key_time":
            kt = np.asarray(C8._child(mc[tgt[0]], b"KeyTime").props[0], dtype=np.int64).copy()
            kt[mid] += 1
            _set_child(mc[tgt[0]], b"KeyTime", kt, b"l")
            rec.update({"key": mid, "frame": frames[mid]})
        elif name == "NC3_key_value":
            kv = np.asarray(C8._child(mc[tgt[0]], b"KeyValueFloat").props[0], dtype=np.float32).copy()
            kv[mid] += np.float32(1.0)
            _set_child(mc[tgt[0]], b"KeyValueFloat", kv, b"f")
            rec.update({"key": mid, "frame": frames[mid]})
        elif name == "NC4_ambiguous_name":
            objs = C8._child(t, b"Objects")
            mdl = [e for e in objs.elems if e.id == b"Model" and int(e.props[0]) == model_uid][0]
            mdl.props[1] = b"sibling\x00\x01Model"
            rec.update({"renamed_model": case["target"]["item"], "to": "sibling", "sibling_models": len(sib)})
        rec.update(_write(t, d/f"{name.lower()}.fbx", version))
        out["files"][name] = rec
    return out


def encode(o):
    decl = _decl(o.declaration, o.declaration_sha)
    out = Path(o.out_dir)
    out.mkdir(parents=True, exist_ok=False)
    prod = json.loads((Path(o.produce_dir)/"produce_report.json").read_text())
    rep = {"schema": C.SCHEMA + ":encode", "declaration_sha256": o.declaration_sha, "cases": {}}
    ci = {"schema": "w112_consumer_input", "blender_version": decl["environment"]["blender_version"], "cases": {}}
    oi = {"schema": "w112_oracle_input", "blender_version": decl["environment"]["blender_version"], "cases": {}}
    for cid, case in decl["cases"].items():
        pr = prod["cases"].get(cid, {})
        r = {}
        rep["cases"][cid] = r
        if pr.get("status") != "OK":
            r["status"] = "NO_SOURCE"
            continue
        d = out/cid
        d.mkdir()
        v1 = Path(o.produce_dir)/cid/"V1.fbx"
        want = (pr.get("v1_export") or {}).get("fbx_sha256") or (pr.get("v1_copy") or {}).get("fbx_sha256")
        if sha256(v1) != want:
            raise SystemExit(f"STOP: {v1} changed after produce")
        shutil.copyfile(v1, d/"control.fbx")
        root, version = C8.fbx_read(d/"control.fbx")
        fmap = AD.fbx_curve_map(root, case["stack"])
        census = pr["perkey_census"]
        sel = _sel_curves(root, fmap)
        ticks = {u: np.asarray(C8._child(e, b"KeyTime").props[0], dtype=np.int64) for u, e in sel.items()}
        fps = pr["timing"]["fps"]/pr["timing"]["fps_base"]
        perkey = C.producer_perkey(fmap, census, ticks, case["frame_start"], fps)
        kinds = {}
        for v in perkey.values():
            k = v["kind"] if v["kind"] != "PERKEY" else ("PERKEY_" + "/".join(sorted(set(v["per_key"]))))
            kinds[k] = kinds.get(k, 0) + 1
        obstructed = []
        for uid, v in perkey.items():
            if v["kind"] == "PERKEY" and any(c not in C.SUPPORTED for c in v["per_key"]):
                rr = next(x for x in fmap["curves"] if x["uid"] == uid)
                frames, _ = C.key_frames(ticks[uid], case["frame_start"], fps)
                obstructed.append({"model": rr["model_name"], "property": rr["fbx_property"], "channel": rr["channel"],
                                   "classes": sorted(set(v["per_key"])),
                                   "frames": [f for f, c in zip(frames, v["per_key"]) if c not in C.SUPPORTED]})
        bad_kind = [k for k in kinds if k in ("UNMATCHED", "BAD_KEY_GRID")]
        r.update({"fbx_map_status": fmap["status"], "curves": len(fmap["curves"]), "producer_kinds": kinds,
                  "obstructed_curves": obstructed, "unmatched_or_bad": bad_kind,
                  "control": {"path": str(d/"control.fbx"), "sha256": sha256(d/"control.fbx")}})
        can_c, rej_c = S.try_canonicalize(root)
        r["control_canonical_digest"] = S.canonical_digest(can_c) if can_c else None
        r["control_canonical_rejection"] = rej_c
        if fmap["status"] != "OK" or bad_kind:
            r["status"] = "UNSUPPORTED"
        elif obstructed:
            r["status"] = "OBSTRUCTED"
        else:
            r["status"] = "VALID"
        r["per_key"] = {str(u): {"kind": v["kind"], "runs": C.rle(v["per_key"]) if v["per_key"] else None}
                        for u, v in perkey.items() if v["kind"] == "PERKEY" and set(v["per_key"]) != {C.LINEAR}}
        meta_tree = None
        if r["status"] == "VALID":
            meta_tree, changed, intended = encode_meta(root, fmap, perkey)
            r["meta"] = _write(meta_tree, d/"meta.fbx", version)
            mroot, _ = C8.fbx_read(d/"meta.fbx")
            eq, bad = masked_equal(root, mroot, changed)
            msel = _sel_curves(mroot, AD.fbx_curve_map(mroot, case["stack"]))
            dec_ok = all(C.expand_flags(*[np.asarray(C8._child(msel[u], n).props[0]) for n in
                                          (b"KeyAttrFlags", b"KeyAttrRefCount")]) == intended[u] for u in msel)
            can_m, rej_m = S.try_canonicalize(mroot)
            cmp = S.compare(can_c, can_m) if can_c and can_m else None
            exp_ids = {k for k, cl in can_c["class_of"].items() if cl == "AnimationCurve"
                       and can_c["uid_of"][k] in set(changed)} if can_c else set()
            r["meta_checks"] = {
                "changed_curves": len(changed), "masked_tree_equal": eq, "masked_tree_bad": bad,
                "decoded_per_key_equal_intended": dec_ok,
                "refcount_ok": all(sum(int(x) for x in C8._child(e, b"KeyAttrRefCount").props[0]) ==
                                   len(C8._child(e, b"KeyTime").props[0]) for e in msel.values()),
                "canonical_rejection": rej_m,
                "canonical_differences": S.public(cmp) if cmp else None,
                "canonical_differ_exactly_changed": bool(cmp and cmp["_sets"]["differ"] == exp_ids
                                                         and not cmp["header_sections_differ"]
                                                         and not cmp["n_connections_only_a"]
                                                         and not cmp["n_objects_only_a"] and not cmp["n_objects_only_b"]),
                "meta_canonical_digest": S.canonical_digest(can_m) if can_m else None}
            r["meta_checks"]["valid"] = bool(eq and dec_ok and r["meta_checks"]["refcount_ok"] and rej_m is None
                                             and r["meta_checks"]["canonical_differ_exactly_changed"])
        # FBX-level evaluation instrument (privileged; evaluation only)
        efiles = {}
        sem = {}
        for uid, v in perkey.items():
            n = len(ticks[uid])
            sem[uid] = v["per_key"] if v["kind"] == "PERKEY" and all(c in C.SUPPORTED for c in v["per_key"]) else \
                [C.LINEAR]*n if v["kind"] == "UNDRIVEN" else None
        tgt = _target_uids(fmap, case["target"]) if case["target"] else None
        ed = d/"efbx"
        ed.mkdir()
        if case["target"] is not None and tgt is not None:
            non_target_ok = all(sem[u] is not None for u in sem if u not in tgt)
            r["instrument_non_target_semantics_ok"] = non_target_ok
            if non_target_ok:
                for lab in C.COMBOS3:
                    s2 = dict(sem)
                    for i, u in enumerate(tgt):
                        s2[u] = [C.LETTER[lab[i]]]*len(ticks[u])
                    efiles[lab] = {pt: _write(build_efile(root, fmap, s2, pt), ed/f"e_{lab}_{pt}.fbx", version)
                                   for pt in C.EPARTS}
        elif case["target"] is None and all(s is not None for s in sem.values()):
            r["instrument_non_target_semantics_ok"] = True
            efiles["SRC"] = {pt: _write(build_efile(root, fmap, sem, pt), ed/f"e_SRC_{pt}.fbx", version)
                             for pt in C.EPARTS}
        else:
            r["instrument_non_target_semantics_ok"] = False
        r["efiles"] = efiles
        # negative controls
        if cid == C.NC_BASE_CASE and meta_tree is not None:
            nd = out/"controls"
            nd.mkdir()
            r["negative_controls"] = nc_files(nd, root, meta_tree, fmap, perkey, case, census, version, fps)
        ci["cases"][cid] = {"stack": case["stack"], "times": case["times"],
                            "files": {"control": r["control"], **({"meta": {k: r["meta"][k] for k in ("path", "sha256")}}
                                                                  if "meta" in r else {})}}
        ci["cases"][cid]["files"]["control"] = {k: r["control"][k] for k in ("path", "sha256")}
        dirs = directives(fmap, census)
        if case["target"] is not None:
            dirs[f"{case['target']['item']}|{case['target']['group']}"] = "TARGET"
        r["oracle_directives"] = dirs
        n = case["target"]["imported_fcurves"] if case["target"] else 0
        oi["cases"][cid] = {"stack": case["stack"], "times": case["times"], "control": ci["cases"][cid]["files"]["control"],
                            "target": ({"item": case["target"]["item"], "group": case["target"]["group"]}
                                       if case["target"] else None),
                            "directives": dirs, "imp_labels": C.combos(n) if n else ["SRC"],
                            "efiles": {k: {pt: {"path": x["path"], "sha256": x["sha256"]} for pt, x in v.items()}
                                       for k, v in efiles.items()}}
    rep["consumer_input_clean"] = consumer_clean(ci, out)
    if not rep["consumer_input_clean"]:
        raise SystemExit("STOP: consumer input failed the whitelist")
    rep["consumer_input_sha256"] = wjson(out/"consumer_input.json", ci)
    rep["oracle_input_sha256"] = wjson(out/"oracle_input.json", oi)
    wjson(out/"encode_report.json", rep)
    print(json.dumps({c: v.get("status") for c, v in rep["cases"].items()}))
    return 0


def consumer_clean(ci, root=None, keys=CONSUMER_KEYS, names=("control.fbx", "meta.fbx")):
    """Structural whitelist: declared keys only; every FBX a declared name under ``root``; no source-side reference."""
    if set(ci) != keys:
        return False
    blob = json.dumps(ci).lower()
    if any(w in blob for w in BANNED_CONSUMER_TOKENS):
        return False
    for sect in [k for k in ("cases", "controls") if k in ci]:
        for v in ci[sect].values():
            files = v["files"].values() if "files" in v else [v["fbx"]]
            for f in files:
                if set(f) != {"path", "sha256"}:
                    return False
                p = Path(f["path"])
                if p.name not in names and not (p.name.startswith("nc") and p.name.endswith(".fbx")):
                    return False
                if root is not None and Path(root).resolve() not in p.resolve().parents:
                    return False
    return True


# --- gate ---------------------------------------------------------------------------------------------------------
def _load_labels(d, prefix, labels):
    return {lab: C.load_npz(d/f"{prefix}_{lab}.npz") for lab in labels}


def gate(o):
    decl = _decl(o.declaration, o.declaration_sha)
    out = Path(o.out_dir)
    out.mkdir(parents=True, exist_ok=False)
    enc = json.loads((Path(o.encode_dir)/"encode_report.json").read_text())
    oi = json.loads((Path(o.encode_dir)/"oracle_input.json").read_text())
    orc = json.loads((Path(o.oracle_dir)/"oracle_report.json").read_text())
    rep = {"schema": C.SCHEMA + ":gate", "cases": {}}
    ci = {"schema": "w112_candidate_input", "blender_version": decl["environment"]["blender_version"], "cases": {},
          "controls": {}}
    for cid, case in decl["cases"].items():
        e = enc["cases"].get(cid, {})
        o_ = orc["cases"].get(cid)
        r = {"encoding": e.get("status")}
        if o_ is None or o_.get("status") != "OK":
            r["imp_bound_m"] = None
            r["eligible"] = False
        else:
            src = C.load_npz(Path(o.produce_dir)/cid/"source.npz")
            labs = oi["cases"][cid]["imp_labels"]
            b = C.exact_bound(src, _load_labels(Path(o.oracle_dir)/cid, "imp", labs), labs)
            r["imp_bound_m"] = b["bound_m"]
            r["harness_ok"] = o_["harness_fields_changed_excl_interpolation"] == [] and o_["harness_other_actions_unchanged"]
            r["eligible"] = bool(e.get("status") == "VALID" and r["harness_ok"] and b["bound_m"] <= C.TOL["recovered_m"])
        if r["eligible"]:
            ci["cases"][cid] = {"stack": case["stack"], "times": case["times"],
                                "fbx": {k: e["meta"][k] for k in ("path", "sha256")}}
        rep["cases"][cid] = r
    base = decl["cases"][C.NC_BASE_CASE]
    nce = enc["cases"].get(C.NC_BASE_CASE, {})
    if "negative_controls" in nce:
        ci["controls"]["NC0_base"] = {"stack": base["stack"], "times": base["times"],
                                      "fbx": {k: nce["meta"][k] for k in ("path", "sha256")}}
        for name, f in nce["negative_controls"]["files"].items():
            ci["controls"][name] = {"stack": base["stack"], "times": base["times"],
                                    "fbx": {"path": f["path"], "sha256": f["sha256"]}}
    rep["candidate_input_clean"] = consumer_clean(ci, Path(o.encode_dir), CANDIDATE_KEYS, ("meta.fbx",))
    if not rep["candidate_input_clean"]:
        raise SystemExit("STOP: candidate input failed the whitelist")
    rep["candidate_input_sha256"] = wjson(out/"candidate_input.json", ci)
    wjson(out/"gate_report.json", rep)
    print(json.dumps({c: [v["encoding"], v["imp_bound_m"], v["eligible"]] for c, v in rep["cases"].items()}))
    return 0


# --- analyze ------------------------------------------------------------------------------------------------------
def _err(src, imp):
    return C.per_time_error(src, imp)


def _arrays_diff(a, b, idx=None):
    m = 0.0
    for k in ("world_head", "world_tail"):
        x, y = a[k], b[k]
        if idx is not None:
            x, y = x[idx], y[idx]
        m = max(m, float(np.abs(x - y).max()) if x.size else 0.0)
    return m


def _cond_report(src, imp, times, classes, trans):
    e = _err(src, imp)
    rep = C.class_maxima(e, times, classes)
    rep["near_boundary_transition"] = C.near_boundary_max(e, times, trans)
    return rep


def analyze_run(rd, decl, w110):
    rd = Path(rd)
    prod = json.loads((rd/"produce/produce_report.json").read_text())
    enc = json.loads((rd/"encode/encode_report.json").read_text())
    oi = json.loads((rd/"encode/oracle_input.json").read_text())
    cons = json.loads((rd/"consume/consume_report.json").read_text())
    orc = json.loads((rd/"oracle/oracle_report.json").read_text())
    gat = json.loads((rd/"gate/gate_report.json").read_text())
    cand = json.loads((rd/"candidate/candidate_report.json").read_text())
    res = {"cases": {}, "controls": {}}
    for cid, case in decl["cases"].items():
        pr, e, co, orr = prod["cases"].get(cid, {}), enc["cases"].get(cid, {}), cons["cases"].get(cid, {}), \
            orc["cases"].get(cid, {})
        r = {"family": case["family"], "spec": case["spec"], "regime": case["regime"], "encoding": e.get("status")}
        res["cases"][cid] = r
        if pr.get("status") != "OK":
            r["status"] = "NO_SOURCE"
            continue
        times = case["times"]
        fs, fe = case["frame_start"], case["frame_end"]
        src = C.load_npz(rd/"produce"/cid/"source.npz")
        census = pr["perkey_census"]
        if case["target"] is not None:
            tc = sorted([c for c in census["curves"] if c["data_path"] == case["target"]["source_path"]],
                        key=lambda c: c["index"])
        else:
            tc = census["curves"]
        classes = {f: C.interval_class(tc, f) for f in range(fs, fe)}
        keys = {f for c in tc for f in c["frames"]}
        trans = {f for f in range(fs, fe) if float(f + 1) in keys}
        r["interval_classes"] = {k: sum(1 for v in classes.values() if v == k) for k in sorted(set(classes.values()))}
        r["target_structure"] = pr.get("target_structure")
        r["source"] = {"fcurves": pr["source_fcurves"], "keys": pr["source_keys_total"],
                       "interpolation": pr["source_interpolation"], "digest": pr["source_action_digest"],
                       "target_per_key": {f"{c['data_path']}[{c['index']}]": C.rle(c["interpolation"]) for c in tc}
                       if case["target"] else None}
        r["build_valid"] = pr.get("build", {}).get("valid_construction", True)
        r["export_ok"] = True
        if "v1_export" in pr:
            x = pr["v1_export"]
            r["export_ok"] = bool(x["export"]["result"] == ["FINISHED"] and x["action_unmodified"]
                                  and x["work_file_unchanged"])
            r["v1_fbx"] = {"sha256": x["fbx_sha256"], "bytes": x["fbx_bytes"], "structure": x["fbx_structure"]}
        else:
            r["export_ok"] = bool(pr["v1_copy"]["equals_declared"])
        r["source_work_unchanged"] = pr.get("work_file_unchanged", True) and (
            pr.get("work_sha256_end") == (pr.get("build", {}).get("work_sha256") or decl["inputs"][f"{cid}_source"]["sha256"]))
        if case["origin"] == "w110_input":
            r["source_digest_equals_w110"] = pr["source_action_digest"] == case["expected_source_action_digest"]
        # encoding
        r["producer_kinds"] = e.get("producer_kinds")
        r["obstructed_curves"] = e.get("obstructed_curves")
        r["meta_checks_valid"] = e.get("meta_checks", {}).get("valid") if e.get("status") == "VALID" else None
        r["meta_changed_curves"] = e.get("meta_checks", {}).get("changed_curves")
        r["control_canonical_digest"] = e.get("control_canonical_digest")
        r["meta_canonical_digest"] = e.get("meta_checks", {}).get("meta_canonical_digest")
        # consumer conditions
        conds = {}
        for k, v in co.items():
            if v.get("status") != "SAMPLED":
                conds[k] = {"status": v.get("status"), "error": v.get("error")}
                continue
            imp = C.load_npz(rd/"consume"/cid/f"{k}.npz")
            conds[k] = _cond_report(src, imp, times, classes, trans)
            if k.endswith("_A"):
                a = v["adapter"]
                acc = a["plan"]["accounting"]
                conds[k]["adapter"] = {
                    "fbx_curves": acc["fbx_curves"], "fbx_curves_mapped": acc["fbx_curves_mapped"],
                    "group_directives": acc["group_directives"], "unresolved": a["plan"]["unresolved"],
                    "imported_fcurves": acc["imported_fcurves_in_target_actions"],
                    "imported_fcurves_mapped": acc["imported_fcurves_mapped"],
                    "fcurves_changed": (a["applied"] or {}).get("fcurves_changed"),
                    "changed_curves": v["changed_curves"]}
                conds[k]["isolated"] = v["access"]["isolated"] and v["access"]["fbx_reads"] >= 1
                conds[k]["integrity_ok"] = v["integrity"]["ok"]
        r["conditions"] = conds
        cu = C.load_npz(rd/"consume"/cid/"control_U.npz") if co.get("control_U", {}).get("status") == "SAMPLED" else None
        if cu is not None and case["target"] is not None:
            r["target_local_control_U"] = {"integer": C.target_local(src, cu, case["target"]["item"], True),
                                           "all": C.target_local(src, cu, case["target"]["item"])}
        if cu is not None and co.get("meta_U", {}).get("status") == "SAMPLED":
            r["meta_U_equals_control_U_m"] = _arrays_diff(cu, C.load_npz(rd/"consume"/cid/"meta_U.npz"))
        if cu is not None and co.get("control_A", {}).get("status") == "SAMPLED":
            r["control_A_equals_control_U_m"] = _arrays_diff(cu, C.load_npz(rd/"consume"/cid/"control_A.npz"))
        # historical replay vs W110 consume control_U
        if case["origin"] == "w110_input" and cu is not None:
            wd = Path(w110)/"results/consume"/cid
            diffs = []
            for nm in ("control_U.npz", "control_U_nb.npz"):
                if (wd/nm).exists():
                    h = C.load_npz(wd/nm)
                    ht = np.round(np.asarray(h["times"], dtype=float), 6)
                    mt = np.round(np.asarray(times, dtype=float), 6)
                    common = [t for t in ht if t in set(mt)]
                    ia = [int(np.where(mt == t)[0][0]) for t in common]
                    ib = [int(np.where(ht == t)[0][0]) for t in common]
                    if common:
                        ci_, ia2, ib2 = C8.common_index(cu, h)
                        for k in ("world_head", "world_tail"):
                            diffs.append(float(np.abs(cu[k][ia][:, ia2] - h[k][ib][:, ib2]).max()))
                    r.setdefault("historical_replay_common_times", 0)
                    r["historical_replay_common_times"] += len(common)
            r["historical_replay_max_m"] = max(diffs) if diffs else None
        # privileged bounds
        bounds = {}
        if orr.get("status") == "OK":
            labs = oi["cases"][cid]["imp_labels"]
            imp = _load_labels(rd/"oracle"/cid, "imp", labs)
            b = C.exact_bound(src, imp, labs)
            bounds["imp"] = _bound_summary(b, classes, case, tc, labs, align=True)
            elabs = [k for k, v in orr["fbx"].items() if v.get("status") == "SAMPLED"]
            if elabs and len(elabs) == len(oi["cases"][cid]["efiles"]):
                ef = _load_labels(rd/"oracle"/cid, "fbx", elabs)
                bf = C.exact_bound(src, ef, elabs)
                bounds["fbx"] = _bound_summary(bf, classes, case, tc, elabs, align=False)
                ints = [i for i, t in enumerate(times) if float(t) == int(t)]
                bounds["fbx_integer_equals_control_U_m"] = max(_arrays_diff(cu, v, ints) for v in ef.values()) \
                    if cu is not None else None
                bounds["fbx_probe_to_key_max_frames"] = max(
                    [x.get("probe_to_key_max_frames", 0.0) for v in orr["fbx"].values()
                     for x in v.get("parts", {}).values()] or [1.0])
                bounds["fbx_target_keys"] = sorted({(pt, tuple(x.get("target_keys", []))) for v in orr["fbx"].values()
                                                    for pt, x in v.get("parts", {}).items()})
                bounds["fbx_target_keys_expected"] = {pt: len(C.part_times(times, pt)) for pt in C.EPARTS}
                bounds["fbx_target_keys_ok"] = all(
                    all(k == len(C.part_times(times, pt)) for k in x.get("target_keys", [-1]))
                    for v in orr["fbx"].values() for pt, x in v.get("parts", {}).items()) if case["target"] else True
            bounds["harness_ok"] = orr["harness_fields_changed_excl_interpolation"] == [] and \
                orr["harness_other_actions_unchanged"]
            bounds["imp_integer_spread_m"] = b["integer_spread_m"]
            bounds["target_fcurves"] = orr.get("target_fcurves")
            bounds["unusable_directives"] = orr.get("unusable_directives")
        r["bounds"] = bounds
        # candidate
        g = gat["cases"].get(cid, {})
        r["eligible"] = g.get("eligible")
        cs = cand["cases"].get(cid, {}).get("S")
        if cs is not None:
            if cs.get("status") == "SAMPLED":
                s = C.load_npz(rd/"candidate/cases"/cid/"S.npz")
                r["candidate"] = _cond_report(src, s, times, classes, trans)
                acc = cs["candidate"]["accounting"]
                r["candidate"].update({
                    "perkey": [{k: v for k, v in x.items() if k != "per_key"} for x in cs["candidate"]["perkey"]],
                    "groups_unresolved": acc["groups_unresolved_after_perkey"],
                    "imported_fcurves_unmapped": acc["imported_fcurves_unmapped"],
                    "fcurves_changed": (cs["candidate"]["applied"] or {}).get("fcurves_changed"),
                    "changed_curves": cs["changed_curves"],
                    "isolated": cs["access"]["isolated"] and cs["access"]["fbx_reads"] >= 1,
                    "integrity_ok": cs["integrity"]["ok"]})
                if case["target"] is not None:
                    r["candidate"]["target_local"] = C.target_local(src, s, case["target"]["item"])
                ma = co.get("meta_A")
                if ma and ma.get("status") == "SAMPLED":
                    r["candidate"]["equals_meta_A_m"] = _arrays_diff(C.load_npz(rd/"consume"/cid/"meta_A.npz"), s)
                    r["candidate"]["same_changed_curves_as_meta_A"] = cs["changed_curves"] == ma["changed_curves"]
            else:
                r["candidate"] = {"status": cs.get("status"), "error": cs.get("error")}
    res["controls"] = _analyze_ncs(rd, decl, enc, cand)
    return res


def _bound_summary(b, classes, case, tc, labels, align):
    per = b["per_interval"]
    out = {"bound_m": b["bound_m"], "integer_m": b["integer_m"], "integer_spread_m": b["integer_spread_m"],
           "labels": len(labels)}
    by_class = {}
    for f, v in per.items():
        by_class.setdefault(classes[f], []).append(v["best_m"])
    out["by_class_max_m"] = {k: max(v) for k, v in by_class.items()}
    best = {}
    for v in per.values():
        best[v["best"]] = best.get(v["best"], 0) + 1
    out["best_label_counts"] = best
    out["intervals_over_1mm"] = sorted(f for f, v in per.items() if v["best_m"] > C.TOL["recovered_m"])
    out["worst_interval"] = max(per, key=lambda f: per[f]["best_m"]) if per else None
    if out["worst_interval"] is not None:
        w = per[out["worst_interval"]]
        out["worst"] = {"interval": out["worst_interval"], "class": classes[out["worst_interval"]],
                        "best": w["best"], "best_m": w["best_m"],
                        "top3": sorted(w["errors_m"].items(), key=lambda kv: kv[1])[:3]}
    n = len(labels[0]) if labels and labels[0] != "SRC" else 0
    if n:
        cl, ll = "C"*n, "L"*n
        gu = max([min(v["errors_m"][cl], v["errors_m"][ll]) for v in per.values()] + [b["integer_m"]])
        out["group_uniform_bound_m"] = gu
        out["group_uniform_intervals_over_1mm"] = sorted(
            f for f, v in per.items() if min(v["errors_m"][cl], v["errors_m"][ll]) > C.TOL["recovered_m"])
        if align and case["target"] is not None and len(tc) == n:
            al = []
            for f, v in per.items():
                lab = "".join("C" if C.in_force(c["frames"], c["interpolation"], f) == C.CONSTANT else "L" for c in tc)
                al.append(v["errors_m"][lab])
            out["source_aligned_bound_m"] = max(al + [b["integer_m"]])
    out["per_interval"] = {str(f): {"class": classes[f], "best": v["best"], "best_m": v["best_m"]}
                           for f, v in per.items()}
    return out


def _analyze_ncs(rd, decl, enc, cand):
    base = decl["cases"][C.NC_BASE_CASE]
    nce = enc["cases"].get(C.NC_BASE_CASE, {}).get("negative_controls")
    out = {"available": nce is not None, "results": {}}
    if nce is None:
        return out
    times = base["times"]
    src = C.load_npz(rd/"produce"/C.NC_BASE_CASE/"source.npz")
    ref = cand["controls"].get("NC0_base", {})
    ref_s = C.load_npz(rd/"candidate/controls/NC0_base/S.npz") if ref.get("S", {}).get("status") == "SAMPLED" else None
    j = nce["nc1_key"]
    jf = nce["nc1_frame"]
    tgt_item, tgt_group = base["target"]["item"], base["target"]["group"]
    for name in NC_SPECS:
        rc = cand["controls"].get(name, {})
        rr = {"expect": NC_SPECS[name][0]}
        for cond in ("S", "A"):
            v = rc.get(cond)
            if v is None:
                continue
            if v.get("status") != "SAMPLED":
                rr[cond] = {"status": v.get("status"), "error": (v.get("error") or "")[:300]}
                continue
            rep = v["candidate"] if cond == "S" else v["adapter"]
            pl = rep["plan"]
            tgt_groups = [x for x in pl["groups"] if x["item"] == tgt_item and x["group"] == tgt_group]
            pk = [x for x in rep.get("perkey", []) if x["item"] == tgt_item and x["group"] == tgt_group]
            changed = v["changed_curves"]
            tgt_changed = [c for c in changed if f'"{tgt_item}"' in c]
            npz = rd/"candidate/controls"/name/("S.npz" if cond == "S" else "nc_A.npz")
            arr = C.load_npz(npz)
            e = _err(src, arr)
            rr[cond] = {"target_directives": [x["directive"] for x in tgt_groups],
                        "target_perkey_status": [x["status"] for x in pk], "target_curves_changed": len(tgt_changed),
                        "curves_changed": len(changed), "max_err_m": float(e.max()),
                        "isolated": v["access"]["isolated"], "integrity_ok": v["integrity"]["ok"],
                        "unresolved": pl["unresolved"],
                        "twin_directives": [x["directive"] for x in pl["groups"] if x["item"] == "sibling"],
                        "twin_curves_changed": len([c for c in changed if '"sibling' in c])}
            if cond == "S" and ref_s is not None:
                iv = C.interval_index(times)
                ins = [i for i, f in enumerate(iv) if f is not None and int(f) == int(jf)] if jf is not None else []
                outs = [i for i in range(len(times)) if i not in ins]
                rr[cond]["diff_vs_base_outside_interval_m"] = _arrays_diff(ref_s, arr, outs)
                rr[cond]["diff_vs_base_inside_interval_m"] = _arrays_diff(ref_s, arr, ins) if ins else None
                rr[cond]["err_inside_interval_m"] = float(e[ins].max()) if ins else None
                bpk = [x for x in ref["S"]["candidate"]["perkey"] if x["item"] == tgt_item]
                rr[cond]["perkey_equal_to_base"] = [x.get("per_key") for x in pk] == [x.get("per_key") for x in bpk]
                if name == "NC1_interval_flag" and pk and bpk and pk[0].get("per_key") and bpk[0].get("per_key"):
                    rr[cond]["perkey_diff_keys"] = [i for i, (a, b) in enumerate(zip(pk[0]["per_key"],
                                                                                       bpk[0]["per_key"])) if a != b]
                if name == "NC3_key_value":
                    fr = nce["files"][name]["frame"]
                    k = [i for i, t in enumerate(times) if abs(t - fr) < 1e-9]
                    rr[cond]["err_at_changed_key_m"] = float(e[k].max()) if k else None
        rr["judged"] = _judge_nc(name, rr, j)
        out["results"][name] = rr
    out["nc1_key"], out["nc1_frame"] = j, jf
    return out


def _judge_nc(name, rr, j):
    s, a = rr.get("S", {}), rr.get("A", {})
    exp = NC_SPECS[name][0]
    if exp == "BASELINE":
        return bool(s.get("isolated") and s.get("integrity_ok"))
    if name == "NC4_ambiguous_name":
        def unres(x):
            if x.get("status") is not None:                 # import / harness failure: nothing was mapped
                return True
            return bool(x.get("target_curves_changed") == 0 and x.get("twin_curves_changed") == 0
                        and x.get("twin_directives") and all(d == "UNRESOLVED" for d in x["twin_directives"])
                        and all(d == "UNRESOLVED" for d in x.get("target_directives", [])))
        return bool(unres(s) and unres(a))
    if not (s.get("isolated") and s.get("integrity_ok") and a.get("isolated") and a.get("integrity_ok")):
        return False
    a_ok = a.get("target_curves_changed") == 0      # W110 never resolves a per-key group
    if exp == "LOCALIZED":
        return bool(a_ok and s.get("perkey_diff_keys") == [j] and s.get("diff_vs_base_outside_interval_m", 1) <=
                    C.TOL["identity_m"] and (s.get("err_inside_interval_m") or 0) > C.TOL["harm_m"])
    if exp == "UNRESOLVED":
        return bool(a_ok and s.get("target_curves_changed") == 0 and s.get("target_perkey_status")
                    and all(x != "OK" for x in s["target_perkey_status"]))
    if exp == "DETECTED_BY_EVALUATION":
        return bool(a_ok and s.get("perkey_equal_to_base") and (s.get("err_at_changed_key_m") or 0) >
                    C.TOL["recovered_m"])
    return False


def _flat(x, prefix=""):
    out = {}
    if isinstance(x, dict):
        for k, v in x.items():
            out.update(_flat(v, f"{prefix}/{k}"))
    elif isinstance(x, list):
        for i, v in enumerate(x):
            out.update(_flat(v, f"{prefix}[{i}]"))
    else:
        out[prefix] = x
    return out


def compare_runs(a, b):
    """Every leaf equal; floats within 1e-9 (metres / frames); FBX canonical digests equal."""
    fa, fb = _flat(a), _flat(b)
    diffs = []
    for k in sorted(set(fa) | set(fb)):
        x, y = fa.get(k, "<missing>"), fb.get(k, "<missing>")
        if k.endswith(("seconds", "sha256")):
            continue
        if isinstance(x, float) and isinstance(y, float):
            if abs(x - y) > C.TOL["identity_m"]:
                diffs.append([k, x, y])
        elif x != y:
            diffs.append([k, x, y])
    return diffs


def decision_inputs(res, decl, verify_ok, bound_ok_scripts, originals_ok):
    cases = res["cases"]
    W = C.WITHIN_CASES
    tol = C.TOL["recovered_m"]

    def bound(c, layer):
        x = cases[c].get("bounds", {}).get(layer)
        return x["bound_m"] if x else None

    imp_ok = {c: (bound(c, "imp") is not None and bound(c, "imp") <= tol) for c in decl["cases"]}
    fbx_ok = {c: (bound(c, "fbx") is not None and bound(c, "fbx") <= tol) for c in decl["cases"]}
    enc = {c: cases[c]["encoding"] == "VALID" for c in decl["cases"]}

    def sf(c):
        s = cases[c].get("candidate")
        return bool(cases[c].get("eligible") and s and s.get("all") is not None and s["all"] <= tol
                    and s["groups_unresolved"] == 0 and not s["imported_fcurves_unmapped"] and s["isolated"]
                    and s["integrity_ok"])
    pre = {"preflight_ok": verify_ok, "declaration_bound": bound_ok_scripts, "originals_unchanged": originals_ok}
    pre["sources_ok"] = all(v.get("build_valid") and v.get("source_work_unchanged") and
                            v.get("source_digest_equals_w110", True) for v in cases.values())
    pre["export_ok"] = all(v.get("export_ok") for v in cases.values())
    pre["encoding_integrity"] = all(v["meta_checks_valid"] for v in cases.values() if v["encoding"] == "VALID") and \
        all(v["encoding"] in ("VALID", "OBSTRUCTED") for v in cases.values())
    inst = []
    for c, v in cases.items():
        b = v.get("bounds", {})
        inst.append(bool(b.get("harness_ok") and b.get("imp_integer_spread_m", 1) <= C.TOL["identity_m"]
                         and "fbx" in b and (b.get("fbx_integer_equals_control_U_m") or 0) <= C.TOL["identity_m"]
                         and b.get("fbx_probe_to_key_max_frames", 1) <= C.TOL["time_match_frames"]
                         and b.get("fbx_target_keys_ok")
                         and not b.get("unusable_directives")))
    inst.append(all(fbx_ok[c] for c in C.UNIFORM_HELD_OF.values()))
    pre["instrument_valid"] = all(inst)
    iso, integ = [], []
    for v in cases.values():
        for k, x in v.get("conditions", {}).items():
            if k.endswith("_A"):
                iso.append(bool(x.get("isolated")))
                integ.append(bool(x.get("integrity_ok")))
        s = v.get("candidate")
        if s and "isolated" in s:
            iso.append(s["isolated"])
            integ.append(s["integrity_ok"])
    ncs = res["controls"]["results"]
    for x in ncs.values():
        for cnd in ("S", "A"):
            if "isolated" in x.get(cnd, {}):
                iso.append(x[cnd]["isolated"])
                integ.append(x[cnd]["integrity_ok"])
    pre["isolation_ok"] = all(iso) and bool(iso)
    pre["integrity_ok"] = all(integ) and bool(integ)
    h, harm = tol, C.TOL["harm_m"]

    def cond(c, k, field="all"):
        return (cases[c].get("conditions", {}).get(k) or {}).get(field)
    pos = {
        "syn_step_U_harms": (cond("syn_step", "control_U") or 0) > harm,
        "syn_step_meta_A_recovers": (cond("syn_step", "meta_A") or 1) <= h,
        "complete_control_U_harms": (cond("complete_control", "control_U") or 0) > harm,
        "complete_control_meta_A_recovers": (cond("complete_control", "meta_A") or 1) <= h,
        "syn_smooth_K_harms": (cond("syn_smooth", "control_K") or 0) > harm,
        "syn_smooth_meta_A_preserves": (cond("syn_smooth", "meta_A") or 1) <= h,
        "importer_ignores_flags": all(v.get("meta_U_equals_control_U_m", 0) <= C.TOL["identity_m"]
                                      for v in cases.values() if v["encoding"] == "VALID"),
        "adapter_noop_on_linear": all(v.get("control_A_equals_control_U_m", 1) <= C.TOL["identity_m"] and
                                      (v["conditions"].get("control_A", {}).get("adapter", {}).get("fcurves_changed") == 0)
                                      for v in cases.values() if "conditions" in v),
        "historical_replay": all((cases[c].get("historical_replay_max_m") is not None and
                                  cases[c]["historical_replay_max_m"] <= C.TOL["identity_m"])
                                 for c in ("syn_step", "syn_smooth", "complete_control")),
        "candidate_reduces_to_w110_on_uniform": all(
            (v["candidate"].get("equals_meta_A_m", 1) <= C.TOL["identity_m"] and
             v["candidate"].get("same_changed_curves_as_meta_A"))
            for c, v in cases.items() if v.get("candidate") and "all" in v.get("candidate", {})
            and decl["cases"][c]["spec"] not in ("INT", "INT1") and c not in C.WITHIN_CASES),
        "negative_controls_available": res["controls"]["available"],
        "negative_controls_as_declared": res["controls"]["available"] and all(x["judged"] for x in ncs.values())}
    pre["positive_negative_controls"] = pos
    pre["controls_ok"] = all(pos.values())
    pre["repeat_ok"] = True  # set by the caller from the run comparison
    pre["imp_ok"], pre["fbx_ok"], pre["encoded"] = ({c: d[c] for c in W} for d in (imp_ok, fbx_ok, enc))
    pre["sf_ok"] = {c: sf(c) for c in W}
    pre["imp_ok_uniL"] = {f: imp_ok[C.UNIFORM_LINEAR_OF[f]] for f in C.FAMILIES}
    pre["fbx_ok_uniL"] = {f: fbx_ok[C.UNIFORM_LINEAR_OF[f]] for f in C.FAMILIES}
    pre["all_imp_ok"], pre["all_fbx_ok"] = imp_ok, fbx_ok
    return pre


def analyze(o):
    decl = _decl(o.declaration, o.declaration_sha)
    out = Path(o.out_dir)
    out.mkdir(parents=True, exist_ok=False)
    ver = json.loads(Path(o.verify).read_text())
    scripts_now = {p: sha256(ROOT/p) for p in decl["scripts_sha256"]}
    bound_scripts = scripts_now == decl["scripts_sha256"]
    inputs_ok = all(sha256(v["path"]) == v["sha256"] for v in decl["inputs"].values())
    _, man = H10._manifests(o)
    originals = all(m["manifest_sha_ok"] and m["files_ok"] for m in man.values())
    for tag, b, msha in (("w110", Path(o.w110), W110_MANIFEST_SHA), ("w111", Path(o.w111), W111_MANIFEST_SHA)):
        chk = check_manifest(b, json.loads((b/"evidence_manifest.json").read_text()), "manifest_stdout.txt")
        originals = originals and chk["ok"] and sha256(b/"evidence_manifest.json") == msha
    runs = {}
    for rd in o.runs:
        runs[Path(rd).name] = analyze_run(rd, decl, o.w110)
    names = list(runs)
    pres = {n: decision_inputs(runs[n], decl, ver["ok"], bound_scripts and inputs_ok, originals) for n in names}
    rep_diff = compare_runs({k: v for k, v in runs[names[0]].items()}, {k: v for k, v in runs[names[1]].items()}) \
        if len(names) == 2 else [["runs", len(names), 2]]
    canon = {}
    if len(names) == 2:
        for cid in decl["cases"]:
            a, b = runs[names[0]]["cases"][cid], runs[names[1]]["cases"][cid]
            canon[cid] = {"control_equal": a.get("control_canonical_digest") == b.get("control_canonical_digest")
                          and a.get("control_canonical_digest") is not None,
                          "meta_equal": a.get("meta_canonical_digest") == b.get("meta_canonical_digest")}
    pre_cmp = compare_runs(pres[names[0]], pres[names[1]]) if len(names) == 2 else [["runs"]]
    repeat_ok = not rep_diff and not pre_cmp and all(v["control_equal"] and v["meta_equal"] for v in canon.values())
    final = dict(pres[names[0]])
    final["repeat_ok"] = repeat_ok
    case, why, detail = C.decide_case(final)
    rep = {"schema": C.SCHEMA + ":analysis", "declaration_sha256": o.declaration_sha, "case": case, "reason": why,
           "detail": detail, "decision_inputs": final, "runs": runs,
           "repeat": {"ok": repeat_ok, "result_differences": rep_diff[:50], "n_result_differences": len(rep_diff),
                      "decision_input_differences": pre_cmp[:50], "canonical": canon},
           "scripts_bound": bound_scripts, "inputs_unchanged": inputs_ok, "originals_unchanged": originals}
    s = wjson(out/"analysis_report.json", rep)
    print(s, case, why, json.dumps(detail))
    return 0


# --- manifest -----------------------------------------------------------------------------------------------------
def manifest(o):
    base = Path(o.base)
    files = {}
    for p in sorted(base.rglob("*")):
        if p.is_file() and p.name not in ("evidence_manifest.json", "manifest_stdout.txt"):
            files[p.relative_to(base).as_posix()] = {"sha256": sha256(p), "bytes": p.stat().st_size}
    _, man = H10._manifests(o)
    unchanged = {t: man[t]["manifest_sha_ok"] and man[t]["files_ok"] for t in man}
    for tag, b, msha in (("w110", Path(o.w110), W110_MANIFEST_SHA), ("w111", Path(o.w111), W111_MANIFEST_SHA)):
        chk = check_manifest(b, json.loads((b/"evidence_manifest.json").read_text()), "manifest_stdout.txt")
        unchanged[tag] = chk["ok"] and sha256(b/"evidence_manifest.json") == msha
    rep = {"schema": C.SCHEMA + ":manifest", "starting_head": o.starting_head, "declaration_sha256": o.declaration_sha,
           "originals_unchanged": unchanged, "all_originals_unchanged": all(unchanged.values()), "files": files}
    s = wjson(base/"evidence_manifest.json", rep)
    print(s, len(files), rep["all_originals_unchanged"])
    return 0 if rep["all_originals_unchanged"] else 2


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", choices=("verify", "declare", "encode", "gate", "analyze", "manifest"))
    for name in ("base", "w106", "w107", "w108", "w109", "w110", "w111", "lock", "blender", "addon_dir",
                 "starting_head", "branch", "verify", "options", "declaration", "declaration_sha", "produce_dir",
                 "encode_dir", "oracle_dir", "out_dir"):
        ap.add_argument("--" + name.replace("_", "-"))
    ap.add_argument("--runs", nargs="*")
    o = ap.parse_args()
    return {"verify": verify, "declare": declare, "encode": encode, "gate": gate, "analyze": analyze,
            "manifest": manifest}[o.stage](o)


if __name__ == "__main__":
    sys.exit(main())
