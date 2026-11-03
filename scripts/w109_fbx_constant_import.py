#!/usr/bin/env python3
"""W109 host-side stages: FBX importer CONSTANT-flag behavioural ownership (evidence only).

  verify    W108 COMPLETE marker, W108 manifest / declaration identity and every W108 manifest file, W107 / W106
            manifests and files, frozen exporter, unchanged FBX add-on sources, reused-script SHAs, worker lock, Git
            state, no other Blender process; byte-verified scratch copies of the inputs (W107 V1 FBX + W107 source
            .blend per real case; W108 V1 FBX + W108 control .blend per synthetic control) into <base>/results/inputs
  declare   fixed cases, selected-stack curve identifiers and original flags (read from the inputs), the two CONSTANT
            encodings, expected source / FBX / import signatures, sample grid, metrics, tolerances, P0 references,
            negative controls, repeat requirement, CASE rule, stop conditions and resource cap -> declaration.json +
            SHA-256 (``outcomes_read: false``)
  encode    scratch copies: control (byte copy), no-op identity check, and per target case the two byte-exact
            KeyAttrFlags variants; independent-reader tree verification of every non-target field -> encode_report
  analyze   parser agreement (independent vs add-on), P0 baseline, source identity, matching, importer behaviour
            per variant and case, smooth negative control, repeat determinism, originals -> analysis_report + CASE
  manifest  hashes of every output, plus a re-check that no W106/W107/W108 original changed

No production exporter/importer, rig, Action original, scene setting or historical artifact is modified.
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
import w109_fbx_constant_import_contract as C  # noqa: E402
import w108_fbx_subframe_contract as C8  # noqa: E402
from w108_fbx_subframe import check_manifest, git, wjson  # noqa: E402  (read-only reuse)

sha256 = C.sha256_file
W108_WORKLOG = "docs/108_WORKLOG_FBX_SUBFRAME_INTERPOLATION_OWNERSHIP.md"
W108_MANIFEST_SHA = "bffcd9ca551744d4ebd4e5087f42ea99a0ad6c8e29395c0d0895ddd8c8768375"
W108_DECLARATION_SHA = "0509a38c882386b35a2d895a8963570a8a3ebb875f2f475818297096b343112d"
W107_MANIFEST_SHA = "20fcb1b9f6ec8879247ac91ae3a21ca1185e995340976079bc80936316f57278"
W106_MANIFEST_SHA = "f3003fe54e68c6cf182313c0b784468275b7ed039f50f574ad10e64eb8fb97db"
EXPORTER = "src/blender/export.py"
EXPORTER_SHA = "521a63d96401f6f82b072ab8c82800e7795623a66370d70a257d72591f76b5fa"
# self-referential stage logs captured empty when their own manifest hashed them (recorded in W106-W108)
MANIFEST_LOG_EXCEPTIONS = {"w108": "manifest_stdout.txt", "w107": "manifest.log", "w106": "results/manifest.log"}
REUSED_SCRIPTS = ("scripts/w108_fbx_subframe_contract.py", "scripts/w108_fbx_subframe_blender.py",
                  "scripts/w108_fbx_subframe.py", "scripts/w107_fbx_blender.py", "scripts/w107_fbx_contract.py",
                  "scripts/w106_action_blender.py", "scripts/w106_action_contract.py")
W109_SCRIPTS = ("scripts/w109_fbx_constant_import_contract.py", "scripts/w109_fbx_constant_import_blender.py",
                "scripts/w109_fbx_constant_import.py")
EXPECTED_ENV = {"blender_version": "5.1.0", "fbx_addon_version": [5, 15, 0]}
SHARED_PATHS = ["src", "tests", "scripts", "examples", "docs", "Architecture_v2.md", "Architecture_v3_FramePose.md"]
W109_NEW = ("scripts/w109_", "tests/test_w109_", "docs/109_WORKLOG_")


def _inputs_for(cid):
    """(source blend, V1 FBX) inside the W107 / W108 bases, relative paths."""
    if cid in C.REAL_CASES:
        return ("w107", f"results/inputs/{cid}_n0.blend"), ("w107", f"results/probe/{cid}/fbx/V1.fbx")
    return ("w108", f"results/probe/{cid}/input.blend"), ("w108", f"results/probe/{cid}/fbx/V1.fbx")


def _manifests(o):
    bases = {"w108": Path(o.w108), "w107": Path(o.w107), "w106": Path(o.w106)}
    shas = {"w108": W108_MANIFEST_SHA, "w107": W107_MANIFEST_SHA, "w106": W106_MANIFEST_SHA}
    out = {}
    for tag, b in bases.items():
        s = sha256(b/"evidence_manifest.json")
        man = json.loads((b/"evidence_manifest.json").read_text())
        chk = check_manifest(b, man, MANIFEST_LOG_EXCEPTIONS[tag])
        out[tag] = {"manifest_sha256": s, "manifest_sha_ok": s == shas[tag], "files": chk["files"], "bad": chk["bad"],
                    "files_ok": chk["ok"], "exception": chk["exception"],
                    "all_originals_unchanged": bool(man.get("all_originals_unchanged"))}
    return bases, out


# --- verify -------------------------------------------------------------------------------------------------------
def verify(o):
    base = Path(o.base)
    out = base/"results"/"verify"
    out.mkdir(parents=True, exist_ok=False)
    rep = {"schema": C.SCHEMA + ":verify", "checks": {}, "stop": []}
    wl = (ROOT/W108_WORKLOG).read_text(encoding="utf-8")
    rep["checks"]["w108_complete_marker"] = wl.rstrip().endswith("AUTOMATION_BATCH_STATUS: COMPLETE\nBATCH_NUMBER: 108")
    bases, man = _manifests(o)
    rep["manifests"] = man
    for tag in man:
        rep["checks"][f"{tag}_manifest_sha"] = man[tag]["manifest_sha_ok"]
        rep["checks"][f"{tag}_files_intact"] = man[tag]["files_ok"]
    d8sha = sha256(bases["w108"]/"results/declare/declaration.json")
    rep["checks"]["w108_declaration_sha"] = {"value": d8sha, "ok": d8sha == W108_DECLARATION_SHA}
    d8 = json.loads((bases["w108"]/"results/declare/declaration.json").read_text())
    exp_sha = sha256(ROOT/EXPORTER)
    rep["checks"]["exporter_sha"] = {"value": exp_sha, "ok": exp_sha == EXPORTER_SHA}
    scr = {p: sha256(ROOT/p) for p in REUSED_SCRIPTS}
    rep["reused_scripts_sha256"] = scr
    rep["checks"]["reused_scripts_equal_w108_declaration"] = all(
        scr[p] == d8["scripts_sha256"][p] for p in REUSED_SCRIPTS if p in d8["scripts_sha256"])
    addon = Path(o.addon_dir)
    a_sha = {p.name: sha256(p) for p in sorted(addon.glob("*.py"))}
    rep["fbx_addon_source_sha256"] = a_sha
    rep["checks"]["fbx_addon_unchanged_vs_w108"] = a_sha == d8["environment"]["fbx_addon_source_sha256"]
    rep["blender_exe_sha256"] = sha256(o.blender)
    head = git("rev-parse", "HEAD")
    status = git("status", "--porcelain").splitlines()
    tracked = [s for s in status if not s.startswith("??")]
    untracked = [s[3:] for s in status if s.startswith("??")]
    rep["git"] = {"head": head, "branch": git("branch", "--show-current"), "origin": git("rev-parse", "origin/" + o.branch),
                  "tracked_changes": tracked, "untracked": untracked}
    rep["checks"]["starting_head"] = {"value": head, "ok": head == o.starting_head}
    rep["checks"]["branch"] = rep["git"]["branch"] == o.branch
    rep["checks"]["origin_equals_head"] = rep["git"]["origin"] == head
    rep["checks"]["working_tree_only_new_w109_files"] = not tracked and all(
        u.startswith(W109_NEW) or u.rstrip("/") in ("scripts", "tests") for u in untracked)
    lk = json.loads(Path(o.lock).read_text()) if Path(o.lock).exists() else {}
    rep["worker_lock"] = lk
    rep["checks"]["worker_lock_batch_109_running"] = lk.get("batch") == 109 and lk.get("state") == "RUNNING" \
        and lk.get("start_commit") == o.starting_head
    tl = subprocess.run(["tasklist", "/FO", "CSV", "/NH"], capture_output=True, text=True, errors="replace").stdout
    blenders = [ln for ln in tl.splitlines() if ln.lower().startswith('"blender')]
    rep["other_blender_processes"] = blenders
    rep["checks"]["no_other_blender_process"] = not blenders
    for k, v in rep["checks"].items():
        if not (v if isinstance(v, bool) else v["ok"]):
            rep["stop"].append(k)
    inputs = base/"results"/"inputs"
    inputs.mkdir(exist_ok=False)
    mans = {t: json.loads((bases[t]/"evidence_manifest.json").read_text()) for t in ("w107", "w108")}
    copies = {}
    for cid in C.ALL_CASES:
        rec = {}
        for kind, (tag, rel) in zip(("blend", "fbx"), _inputs_for(cid)):
            dst = inputs/f"{cid}_{'source.blend' if kind == 'blend' else 'V1.fbx'}"
            shutil.copyfile(bases[tag]/rel, dst)
            r = {"source": f"{tag}:{rel}", "path": str(dst), "sha256": sha256(dst),
                 "manifest_sha256": mans[tag]["files"].get(rel, {}).get("sha256")}
            r["ok"] = r["sha256"] == r["manifest_sha256"]
            if not r["ok"]:
                rep["stop"].append(f"copy:{cid}:{kind}")
            rec[kind] = r
        copies[cid] = rec
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
    ver = json.loads(Path(o.verify).read_text())
    if not ver["ok"]:
        raise SystemExit("STOP: verify failed")
    opts = json.loads(Path(o.options).read_text())
    if opts["blender_version"] != EXPECTED_ENV["blender_version"] or \
            opts["fbx_addon"]["version"] != EXPECTED_ENV["fbx_addon_version"]:
        raise SystemExit(f"STOP: unexpected environment {opts['blender_version']} / {opts['fbx_addon']['version']}")
    if opts["fbx_addon"].get("source_sha256") != ver["fbx_addon_source_sha256"]:
        raise SystemExit("STOP: runtime FBX add-on differs from the verified sources")
    w108, w107 = Path(o.w108), Path(o.w107)
    d8 = json.loads((w108/"results/declare/declaration.json").read_text())
    p8 = json.loads((w108/"results/probe/probe_report.json").read_text())
    a8 = json.loads((w108/"results/analysis/analysis_report.json").read_text())
    cases = {}
    for cid in C.ALL_CASES:
        real = cid in C.REAL_CASES
        spec = d8["cases"][cid] if real else d8["synthetic_controls"][cid]
        prb = p8["cases" if real else "controls"][cid]
        ana = a8["cases" if real else "controls"][cid]
        inp = ver["inputs"][cid]
        data = Path(inp["fbx"]["path"]).read_bytes()
        root, version = C8.fbx_read(inp["fbx"]["path"])
        sel, tg, smap = C.target_uids(root, spec["action"])
        fl = C.curve_flags(root)
        locs = C.curve_flag_locations(data)
        tflags = sorted({v for u in tg for v in fl[u]})
        if tflags != [C.SOURCE_FLAG] or sorted({v for vs in fl.values() for v in vs}) != [C.SOURCE_FLAG]:
            raise SystemExit(f"STOP: {cid}: source flags {tflags} contradict the declared 0x{C.SOURCE_FLAG:x} reading")
        if sel is None or not tg or any(locs[u]["encoding"] != 0 or locs[u]["type"] != b"i" for u in tg):
            raise SystemExit(f"STOP: {cid}: no unique selected stack or a target flag array is not patchable in place")
        summ = C8.fbx_anim_summary(root, spec["timing"]["fps"]/spec["timing"]["fps_base"])["stacks"][sel]
        role = ("real" if real else ("step control (mutated)" if cid == C.STEP_CONTROL
                                     else "smooth LINEAR negative control (never mutated)"))
        cases[cid] = {
            "id": cid, "role": role, "regime": spec.get("regime") if real else "synthetic (separately labelled)",
            "action": spec["action"], "source_blend": inp["blend"]["path"], "source_blend_sha256": inp["blend"]["sha256"],
            "source_blend_origin": inp["blend"]["source"], "v1_fbx": inp["fbx"]["path"],
            "v1_fbx_sha256": inp["fbx"]["sha256"], "v1_fbx_origin": inp["fbx"]["source"],
            "variants": list(C.VARIANTS) if cid in C.TARGET_CASES else [],
            "expected_source_interpolation": ["CONSTANT"] if cid != C.SMOOTH_CONTROL else ["LINEAR"],
            "expected_source_action_digest": prb["source_action_digest"],
            "expected_source_slot": prb["assigned"]["slot"], "expected_imported_slot": prb["paths"]["V1"]["assigned"]["slot"],
            "expected_imported_action": prb["paths"]["V1"]["selected_action"],
            "timing": spec["timing"], "source_key_frames": spec["source_key_frames"], "times": spec["times"],
            "interval_classes": spec["interval_classes"],
            "fbx": {"version": version, "selected_stack": sel, "target_curves": len(tg),
                    "target_uids": tg, "target_uids_sha256": hashlib.sha256(",".join(map(str, tg)).encode()).hexdigest(),
                    "curves_total": len(fl), "stacks": sorted({s for s in smap.values() if s is not None}),
                    "original_target_flags_hex": [f"0x{v:x}" for v in tflags],
                    "original_all_flags_hex": sorted({f"0x{v:x}" for vs in fl.values() for v in vs}),
                    "selected_stack_keys": summ["keys"], "selected_stack_distinct_frames": summ["distinct_frames"],
                    "flag_array_lengths": sorted({locs[u]["length"] for u in tg})},
            "p0_reference": {"w108_source_npz": str(w108/f"results/probe/{cid}/source.npz"),
                             "w108_source_npz_sha256": sha256(w108/f"results/probe/{cid}/source.npz"),
                             "w108_v1_npz": str(w108/f"results/probe/{cid}/V1.npz"),
                             "w108_v1_npz_sha256": sha256(w108/f"results/probe/{cid}/V1.npz"),
                             "w108_v1_subframe_max_m": ana["paths"]["V1"]["errors"]["subframe"]["max_abs_coord_m"],
                             "w108_v1_integer_max_m": ana["paths"]["V1"]["errors"]["integer"]["max_abs_coord_m"]}}
        if real:
            cases[cid]["p0_reference"]["w107_v1_npz"] = str(w107/f"results/probe/{cid}/V1.npz")
            cases[cid]["p0_reference"]["w107_v1_npz_sha256"] = sha256(w107/f"results/probe/{cid}/V1.npz")
    decl = {
        "schema": C.SCHEMA + ":declaration", "outcomes_read": False,
        "note": "Frozen before any W109 FBX mutation, import or sampling ran. W107/W108 outcomes are historical "
                "references; only the unmodified V1 inputs were parsed (to bind the selected-stack curves and "
                "original flags).",
        "starting_head": o.starting_head, "verify_sha256": sha256(o.verify), "options_sha256": sha256(o.options),
        "environment": {"blender_version": opts["blender_version"], "blender_build_hash": opts["blender_build_hash"],
                        "blender_exe": o.blender, "blender_exe_sha256": ver["blender_exe_sha256"],
                        "fbx_addon_version": opts["fbx_addon"]["version"],
                        "fbx_addon_source_sha256": ver["fbx_addon_source_sha256"],
                        "launch": "-b --factory-startup -t 2 --python-exit-code 1"},
        "scripts_sha256": {p: sha256(ROOT/p) for p in (*W109_SCRIPTS, *REUSED_SCRIPTS, EXPORTER)},
        "cases": cases,
        "key_data_basis": "frozen W107 V1 FBX (no key simplification) for the three real Actions; W108 V1 FBX for "
                          "the synthetic controls (W108 built them on the same V1 path)",
        "intervention": {
            "what": "byte-exact in-place rewrite of the KeyAttrFlags int32 values of the AnimationCurves connected "
                    "(curve -> curve node -> layer -> stack) to the uniquely name-matched AnimationStack only; file "
                    "length, every other byte, every other curve/take/object/connection unchanged",
            "source_flag": f"0x{C.SOURCE_FLAG:x}", "source_flag_decoded": C.decode_flag(C.SOURCE_FLAG),
            "derivation": C.variant_flag.__doc__.strip(),
            "variants": {v: {"flag_hex": f"0x{C.EXPECTED_FLAGS[v]:x}", "decoded": C.decode_flag(C.EXPECTED_FLAGS[v]),
                             "expected_function": C.EXPECTED_FUNCTION[v]} for v in C.VARIANTS},
            "bit_semantics": {"0x2": "eInterpolationConstant", "0x4": "eInterpolationLinear",
                              "0x8": "eInterpolationCubic",
                              "0x100": "eConstantNext under CONSTANT; eTangentAuto (cubic-only) otherwise -- the "
                                       "exporter literal sets it as tangent-auto, so a CONSTANT-standard key must "
                                       "clear it",
                              "0x2000|0x4000": "generic time-independent / clamp-progressive tangent bits; retained, "
                                               "inert for non-cubic keys"},
            "targets": "real cases and syn_step only; syn_smooth is never mutated",
            "not_testable_if": "a target array is compressed / not int32 / holds a value other than 0x6104; any byte "
                               "outside the target value spans differs; the independent-reader or add-on tree digest "
                               "with targets masked differs from the control; target flags do not decode to the "
                               "declared values; the two parsers disagree",
            "no_op_identity": "the patcher run with the source value (0x6104) must reproduce the control bytes exactly"},
        "importer": "bpy.ops.import_scene.fbx(filepath=...) defaults into read_factory_settings(use_empty=True); no "
                    "post-import correction",
        "importer_static_note": "W108 static inspection: import_fbx.py never reads KeyAttrFlags and fills LINEAR. "
                                "Recorded as the basis of H109's prediction, not as evidence.",
        "samples": "W108's explicit per-case lists: every integer frame plus +0.25/+0.50/+0.75 of every frame but "
                   "the last; interval classes from W108 (source key times only)",
        "metrics": {
            "primary": "max |coordinate| difference of world bone heads/tails vs the source Action (W106-W108 gate "
                       "form), by phase, all subframes and transition / quiet subframes (W108 subframe_report)",
            "declared_function": C.declared_function_error.__doc__.strip(),
            "delta_vs_control": "max abs difference of every sampled array (local quat/loc/scale, world head/tail/"
                                "rot, object matrix) between the variant import and the control import",
            "census": "imported F-curve interpolation sets per object / pose channel, keys per interpolation, other "
                      "imported Actions' interpolation and key-data digests",
            "diagnostics": ["world rotation angle", "object rotation / translation", "armature-space tail error",
                            "worst bones / times"]},
        "classification": C.classify.__doc__.strip(),
        "p0": {"control_identity": "control.fbx bytes == V1 input bytes; independent and add-on parser record "
                                   "identical; selected stack flags {0x6104}",
               "source_fidelity": "fresh source samples == W108 source.npz (bitwise); source Action digest, "
                                  "interpolation, slot and key frames == declared",
               "control_replay": "control import samples vs W108 V1.npz <= deterministic_m at every time; real cases "
                                 "also vs W107 V1.npz at the common integer+half times; control subframe max and "
                                 "integer max vs W108 analysis <= w108_metric_reproduce_m; control interpolation "
                                 "{LINEAR}; imported Action == expected name, slot == expected",
               "failure": "CASE D"},
        "smooth_control": "syn_smooth control import: interpolation {LINEAR}, P0 replay as above, subframe max vs "
                          "source <= smooth_sufficiency_m, source changes in every interval "
                          "(> smooth_min_interval_change_m); never mutated",
        "matching": "for every import: exactly one armature, unique name-matched Action == expected, slot == expected, "
                    "rest pose / Action inventory equal to the control import; non-selected Actions' key data equal "
                    "to the control import (no flag leakage); add-on tree selected stack and target uids == declared",
        "hypothesis": {
            "H109": "a well-formed FBX differing from W107 V1 only in a valid CONSTANT KeyAttrFlags value on the "
                    "selected curves is still imported as LINEAR by the unchanged Blender 5.1 importer, producing "
                    "essentially the same subframe error (falsified if the import follows the declared CONSTANT "
                    "function)",
            "expected_signatures": {
                "source": "real + syn_step: interpolation {CONSTANT}, held by evaluation; syn_smooth {LINEAR}",
                "fbx": "control: every curve 0x6104 (LINEAR); variants: selected curves 0x6002 / 0x6102, every other "
                       "curve 0x6104; masked trees identical to the control under both parsers",
                "import_if_H109": "LINEAR_UNCHANGED for both variants on every target case",
                "import_if_honoured": "CONSTANT_standard FOLLOWS_DECLARED (floor); CONSTANT_next FOLLOWS_DECLARED "
                                      "(ceil) if the next mode is also honoured"}},
        "tolerances": C.TOL,
        "case_rule": C.decide_case.__doc__.strip(),
        "negative_controls": {"NC1": "no-op patch reproduces the control bytes",
                              "NC2": "control import == W108 / W107 V1 samples (P0)",
                              "NC3": "syn_smooth left LINEAR stays smooth (<= 1 mm vs source)",
                              "NC4": "non-selected stacks keep 0x6104 and their imported Actions keep identical key "
                                     "data",
                              "NC5": "fresh repeat of encode + probe: identical variant bytes, decoded flags, key-data "
                                     "digests and samples <= deterministic_m",
                              "NC6": "W106/W107/W108 originals and source .blend copies unchanged"},
        "repeat": "encode and probe are run twice into fresh directories; every comparison above must hold for both "
                  "and agree between them",
        "not_run": {"mixed single synthetic Action (one held + one smooth curve)": "optional in the Batch; not built",
                    "importer patch / post-import correction": "out of scope by contract",
                    "V0 FBX": "historical negative baseline only; not mutated",
                    "non-Blender consumers, FBX SDK validation": "not available on Main",
                    "frame_end / takes / 50->30 fps / V1 cost": "preserved, unresolved"},
        "stop_conditions": ["declaration / encode hash mismatch", "Blender version differs", "input hash mismatch",
                            "source flags not 0x6104 or target not patchable in place",
                            "non-isolated mutation (encode invalid) -> NOT TESTABLE, no import",
                            "P0 / identity / matching / determinism failure -> CASE D",
                            "shared/production file changed -> STOP", "resource cap exceeded"],
        "resource_cap": {"blender_runs": "options + 2 probes", "wall_clock_per_probe_s": 900, "disk_gb": 2,
                         "gpu": "none", "network": "none"},
    }
    path = out/"declaration.json"
    wjson(path, decl)
    s = sha256(path)
    (out/"declaration.sha256").write_text(s + "\n")
    print(s)
    return 0


# --- encode -------------------------------------------------------------------------------------------------------
def encode(o):
    decl_bytes = Path(o.declaration).read_bytes()
    if hashlib.sha256(decl_bytes).hexdigest() != o.declaration_sha:
        raise SystemExit("STOP: declaration hash mismatch")
    decl = json.loads(decl_bytes)
    out = Path(o.out_dir)
    out.mkdir(parents=True, exist_ok=False)
    rep = {"schema": C.SCHEMA + ":encode", "declaration_sha256": o.declaration_sha, "cases": {}, "problems": []}
    for cid, case in decl["cases"].items():
        d = out/cid
        d.mkdir()
        src = Path(case["v1_fbx"])
        data = src.read_bytes()
        if hashlib.sha256(data).hexdigest() != case["v1_fbx_sha256"]:
            raise SystemExit(f"STOP: {src} hash mismatch")
        (d/"control.fbx").write_bytes(data)
        root0, _ = C8.fbx_read(d/"control.fbx")
        ctrl = C.file_tree_record(root0, case["action"], C.CONTROL)
        tg = case["fbx"]["target_uids"]
        noop, _ = C.patch_flags(data, tg, C.SOURCE_FLAG)
        crec = {"control_sha256": sha256(d/"control.fbx"), "control_equals_input": sha256(d/"control.fbx") == case["v1_fbx_sha256"],
                "noop_identity": noop == data, "control_tree_independent": ctrl,
                "control_ok": (ctrl["target_uids_sha256"] == case["fbx"]["target_uids_sha256"]
                               and ctrl["selected_stack"] == case["fbx"]["selected_stack"]
                               and ctrl["target_flags_hex"] == [f"0x{C.SOURCE_FLAG:x}"]),
                "variants": {}}
        for v in case["variants"]:
            want = C.EXPECTED_FLAGS[v]
            try:
                new, recs = C.patch_flags(data, tg, want)
            except ValueError as exc:
                crec["variants"][v] = {"valid": False, "error": str(exc)}
                rep["problems"].append(f"{cid}:{v}:{exc}")
                continue
            p = d/f"{v}.fbx"
            p.write_bytes(new)
            diff = C.byte_diff_spans(data, new, [(r["offset"], r["bytes"]) for r in recs])
            root1, _ = C8.fbx_read(p)
            tr = C.file_tree_record(root1, case["action"], v)
            masked_ctrl = C.tree_digest(root0, tg)
            st = C8.fbx_anim_summary(root1, case["timing"]["fps"]/case["timing"]["fps_base"])["stacks"]
            sel = st.get(case["fbx"]["selected_stack"], {})
            vr = {"sha256": sha256(p), "bytes": p.stat().st_size, "patched_curves": len(recs),
                  "patched_bytes_total": sum(r["bytes"] for r in recs), "byte_diff": diff, "tree_independent": tr,
                  "selected_stack_summary": sel,
                  "other_stacks_interpolation": sorted({i for n, s in st.items() if n != case["fbx"]["selected_stack"]
                                                        for i in s["interpolation"]}),
                  "checks": {"same_length": diff["same_length"], "diff_only_in_target_spans": diff["all_in_targets"],
                             "every_target_changed": diff["diff_bytes"] is not None and diff["diff_bytes"] > 0,
                             "masked_tree_equals_control": tr["digest_targets_masked"] == masked_ctrl
                             == ctrl["digest_targets_masked"],
                             "full_tree_differs": tr["digest_full"] != ctrl["digest_full"],
                             "target_uids_equal_declared": tr["target_uids_sha256"] == case["fbx"]["target_uids_sha256"],
                             "target_flags_equal_declared": tr["target_flags_hex"] == [f"0x{want:x}"],
                             "decoded_as_declared": tr["target_flags_decoded"] == [C.decode_flag(want)],
                             "nontarget_flags_unchanged": tr["nontarget_flags_hex"] == ctrl["nontarget_flags_hex"],
                             "selected_stack_counts_unchanged": sel.get("keys") == case["fbx"]["selected_stack_keys"]
                             and sel.get("refcount_covers_keys") is True and sel.get("values_match_times") is True,
                             "selected_stack_interpolation_constant": sel.get("interpolation") == ["CONSTANT"],
                             "other_stacks_linear": vr_other_linear(st, case)}}
            vr["valid"] = all(vr["checks"].values())
            if not vr["valid"]:
                rep["problems"].append(f"{cid}:{v}:{[k for k, x in vr['checks'].items() if not x]}")
            crec["variants"][v] = vr
        if not (crec["control_equals_input"] and crec["noop_identity"] and crec["control_ok"]):
            rep["problems"].append(f"{cid}:control")
        rep["cases"][cid] = crec
    rep["valid"] = not rep["problems"]
    s = wjson(out/"encode_report.json", rep)
    print(json.dumps({"valid": rep["valid"], "problems": rep["problems"], "sha256": s}))
    return 0 if rep["valid"] else 2


def vr_other_linear(stacks, case):
    others = [s for n, s in stacks.items() if n != case["fbx"]["selected_stack"]]
    return all(s["interpolation"] == ["LINEAR"] for s in others)


# --- analyze ------------------------------------------------------------------------------------------------------
TREE_KEYS = ("selected_stack", "target_curves", "target_uids_sha256", "curves_total", "unmapped_curves", "stacks",
             "target_flags_hex", "nontarget_flags_hex", "digest_full", "digest_targets_masked")


def _int_mask(t):
    return C8.phase_of(t) == 0


def analyze_case(cid, case, enc, prb, d, enc_dir):
    T = C.TOL
    times = np.array(case["times"])
    classes = {int(k): v for k, v in case["interval_classes"].items()}
    src = C.load_npz(d/"source.npz")
    ref = case["p0_reference"]
    res = {"role": case["role"], "regime": case["regime"], "action": case["action"]}
    # source identity
    s = {k: prb.get(k) for k in ("assigned", "timing", "source_action_digest", "source_fcurves", "source_keys_total",
                                 "source_interpolation", "slots", "work_file_unchanged")}
    s["ok"] = bool(prb["source_action_digest"] == case["expected_source_action_digest"]
                   and prb["assigned"]["action"] == case["action"] and prb["assigned"]["slot"] == case["expected_source_slot"]
                   and prb["timing"] == case["timing"] and prb["source_key_frames"] == case["source_key_frames"]
                   and prb["source_interpolation"] == case["expected_source_interpolation"]
                   and prb["work_file_unchanged"] and np.array_equal(src["times"], times))
    res["source"] = s
    # parser agreement + matching per file
    files = {}
    for tag, fr in prb["files"].items():
        etree = enc["control_tree_independent"] if tag == C.CONTROL else enc["variants"][tag]["tree_independent"]
        atree = fr["addon_tree"]
        agree = atree.get("status") == "PARSED" and all(atree.get(k) == etree.get(k) for k in TREE_KEYS) \
            and atree.get("target_flags_decoded") == etree.get("target_flags_decoded")
        im = fr["import"]
        m = {"parsers_agree": bool(agree),
             "addon_masked_equals_control": atree.get("digest_targets_masked")
             == prb["files"][C.CONTROL]["addon_tree"].get("digest_targets_masked"),
             "imported": im.get("status") == "IMPORTED"}
        if m["imported"]:
            m.update({"one_armature": len(im["armatures"]) == 1,
                      "action_expected": im["selected_action"] == case["expected_imported_action"],
                      "slot_expected": im["assigned"]["slot"] == case["expected_imported_slot"],
                      "key_times_integer": im["key_times"]["max_dev_from_integer"] == 0.0})
            if tag != C.CONTROL:
                m.update({"rest_equal_control": im["rest_equal_control"],
                          "actions_equal_control": im["actions_equal_control"],
                          "other_actions_equal_control": im["other_actions_equal_control"]})
        files[tag] = {"checks": m, "matching_ok": all(m.values()),
                      "import": {k: im.get(k) for k in ("selected_action", "selection", "assigned", "imported_interpolation",
                                                       "census", "key_times", "keydata_digest_all",
                                                       "keydata_digest_excl_interpolation", "imported_timing",
                                                       "other_actions", "other_actions_interpolation",
                                                       "keydata_fields_differing_from_control", "imported_actions")}}
    res["files"] = files
    res["matching_ok"] = all(f["matching_ok"] for f in files.values()) and set(files) == {C.CONTROL, *case["variants"]}
    # P0 (control baseline)
    p0 = {"w108_npz_sha_unchanged": sha256(ref["w108_source_npz"]) == ref["w108_source_npz_sha256"]
          and sha256(ref["w108_v1_npz"]) == ref["w108_v1_npz_sha256"],
          "control_bytes_equal_input": enc["control_equals_input"], "noop_identity": enc["noop_identity"],
          "control_tree_ok": enc["control_ok"]}
    p0["source_vs_w108_max_diff"] = C8.arrays_max_diff(src, C.load_npz(ref["w108_source_npz"]))
    res["P0"] = p0
    if not files[C.CONTROL]["checks"]["imported"]:
        p0["ok"] = False
        res["source_held_by_evaluation"] = None
        return res
    ctrl = C.load_npz(d/f"{C.CONTROL}.npz")
    p0["control_vs_w108_v1_max_diff"] = C8.arrays_max_diff(ctrl, C.load_npz(ref["w108_v1_npz"]))
    if "w107_v1_npz" in ref:
        w7 = C.load_npz(ref["w107_v1_npz"])
        ma, mb = C8.common_time_masks(ctrl["times"], w7["times"])
        p0["w107_v1_npz_sha_unchanged"] = sha256(ref["w107_v1_npz"]) == ref["w107_v1_npz_sha256"]
        p0["w107_times_all_present"] = bool(mb.all())
        p0["control_vs_w107_v1_max_diff"] = C8.arrays_max_diff(ctrl, w7, mask_a=ma, mask_b=mb)
    ce = C8.subframe_report(src, ctrl, times, classes)
    p0["control_subframe_max_m"] = ce["subframe"]["max_abs_coord_m"]
    p0["control_integer_max_m"] = ce["integer"]["max_abs_coord_m"]
    p0["subframe_minus_w108_m"] = p0["control_subframe_max_m"] - ref["w108_v1_subframe_max_m"]
    p0["integer_minus_w108_m"] = p0["control_integer_max_m"] - ref["w108_v1_integer_max_m"]
    p0["control_interpolation"] = files[C.CONTROL]["import"]["imported_interpolation"]
    p0["ok"] = bool(p0["w108_npz_sha_unchanged"] and p0["control_bytes_equal_input"] and p0["noop_identity"]
                    and p0["control_tree_ok"] and p0["source_vs_w108_max_diff"] <= T["source_fidelity_m"]
                    and p0["control_vs_w108_v1_max_diff"] <= T["deterministic_m"]
                    and abs(p0["subframe_minus_w108_m"]) <= T["w108_metric_reproduce_m"]
                    and abs(p0["integer_minus_w108_m"]) <= T["w108_metric_reproduce_m"]
                    and p0["control_interpolation"] == ["LINEAR"]
                    and ("w107_v1_npz" not in ref or (p0["w107_v1_npz_sha_unchanged"] and p0["w107_times_all_present"]
                                                     and p0["control_vs_w107_v1_max_diff"] <= T["deterministic_m"])))
    # source semantics by evaluation
    ph = C8.phase_of(times)
    idx = {float(t): i for i, t in enumerate(times) if ph[i] == 0}
    per = {}
    for i, t in enumerate(times):
        if ph[i] > 0:
            j = idx[float(np.floor(t))]
            dv = float(max(np.abs(src["world_head"][i] - src["world_head"][j]).max(),
                           np.abs(src["world_tail"][i] - src["world_tail"][j]).max()))
            per[int(np.floor(t))] = max(per.get(int(np.floor(t)), 0.0), dv)
    res["source_held_by_evaluation"] = max(per.values()) == 0.0
    res["source_min_interval_change_m"] = min(per.values())
    res["control"] = {"errors_vs_source": ce,
                      "declared_floor": C.declared_function_error(ctrl, ctrl, times, "floor", classes),
                      "declared_ceil": C.declared_function_error(ctrl, ctrl, times, "ceil", classes),
                      "census": files[C.CONTROL]["import"]["census"]}
    mi = _int_mask(times)
    res["variants"] = {}
    for v in case["variants"]:
        if not files.get(v, {}).get("checks", {}).get("imported"):
            res["variants"][v] = {"outcome": "NOT_IMPORTED", "keydata_ok": False, "integer_ok": False}
            continue
        imp = C.load_npz(d/f"{v}.npz")
        mode = C.EXPECTED_FUNCTION[v]
        im = prb["files"][v]["import"]
        de = C.declared_function_error(imp, ctrl, times, mode, classes)
        other = C.declared_function_error(imp, ctrl, times, "ceil" if mode == "floor" else "floor", classes)
        disc = res["control"][f"declared_{mode}"]["max_m"] > T["discriminable_m"]
        r = {"imported_interpolation": im["imported_interpolation"], "keydata_equal_all": im["keydata_equal_all"],
             "keydata_equal_excl_interpolation": im["keydata_equal_excl_interpolation"],
             "delta_vs_control_m": C8.arrays_max_diff(imp, ctrl),
             "integer_vs_control_m": C8.arrays_max_diff(imp, ctrl, mask_a=mi, mask_b=mi),
             "declared_error_m": de["max_m"], "discriminable": bool(disc)}
        r["outcome"] = C.classify(r)
        r["keydata_ok"] = bool(r["keydata_equal_excl_interpolation"])
        r["integer_ok"] = bool(r["integer_vs_control_m"] <= T["deterministic_m"])
        r["flag_read"] = im["imported_interpolation"] != ["LINEAR"]   # importer F-curves reflect the flag at all
        r["census"] = im["census"]
        r["declared_function_error"] = de
        r["other_function_error"] = other
        r["errors_vs_source"] = C8.subframe_report(src, imp, times, classes)
        res["variants"][v] = r
    return res


MISSING = {"outcome": "NOT_IMPORTED", "keydata_ok": False, "integer_ok": False}


def analyze(o):
    decl_bytes = Path(o.declaration).read_bytes()
    if hashlib.sha256(decl_bytes).hexdigest() != o.declaration_sha:
        raise SystemExit("STOP: declaration hash mismatch")
    decl = json.loads(decl_bytes)
    out = Path(o.out_dir)
    out.mkdir(parents=True, exist_ok=False)
    runs = []
    for e, p in ((o.encode_dir, o.probe_dir), (o.encode_repeat_dir, o.repeat_dir)):
        enc = json.loads((Path(e)/"encode_report.json").read_text())
        prb = json.loads((Path(p)/"probe_report.json").read_text())
        runs.append((Path(e), enc, Path(p), prb))
    rep = {"schema": C.SCHEMA + ":analysis", "declaration_sha256": o.declaration_sha, "cases": {}, "repeat": {}}
    rep["encode_valid"] = [r[1]["valid"] and r[1]["declaration_sha256"] == o.declaration_sha for r in runs]
    rep["probe_bound"] = [r[3]["declaration_sha256"] == o.declaration_sha
                          and r[3]["encode_sha256"] == sha256(r[0]/"encode_report.json") for r in runs]
    for cid, case in decl["cases"].items():
        rr = [analyze_case(cid, case, r[1]["cases"][cid], r[3]["cases"][cid], r[2]/cid, r[0]) for r in runs]
        res = rr[0]
        res["repeat_run_summary"] = {k: rr[1].get(k) for k in ("matching_ok",)}
        res["repeat_run_summary"]["p0_ok"] = rr[1]["P0"].get("ok")
        res["repeat_run_summary"]["source_ok"] = rr[1]["source"]["ok"]
        res["repeat_run_summary"]["outcomes"] = {v: x.get("outcome") for v, x in rr[1].get("variants", {}).items()}
        # determinism between the two runs
        d1, d2 = runs[0][2]/cid, runs[1][2]/cid
        e1, e2 = runs[0][1]["cases"][cid], runs[1][1]["cases"][cid]
        det = {"source": C8.arrays_max_diff(C.load_npz(d1/"source.npz"), C.load_npz(d2/"source.npz"))}
        bytes_eq, kd_eq = {}, {}
        for tag in [C.CONTROL, *case["variants"]]:
            det[tag] = C8.arrays_max_diff(C.load_npz(d1/f"{tag}.npz"), C.load_npz(d2/f"{tag}.npz")) \
                if (d1/f"{tag}.npz").exists() and (d2/f"{tag}.npz").exists() else None
            bytes_eq[tag] = (e1["control_sha256"] == e2["control_sha256"]) if tag == C.CONTROL else \
                (e1["variants"][tag]["sha256"] == e2["variants"][tag]["sha256"])
            i1 = runs[0][3]["cases"][cid]["files"][tag]["import"]
            i2 = runs[1][3]["cases"][cid]["files"][tag]["import"]
            kd_eq[tag] = (i1.get("keydata_digest_all") == i2.get("keydata_digest_all") is not None
                          and i1.get("census") == i2.get("census")
                          and runs[0][3]["cases"][cid]["files"][tag]["addon_tree"]
                          == runs[1][3]["cases"][cid]["files"][tag]["addon_tree"])
        rep["repeat"][cid] = {"max_diff": det, "fbx_bytes_equal": bytes_eq, "import_and_tree_equal": kd_eq,
                              "outcomes_equal": res["repeat_run_summary"]["outcomes"]
                              == {v: x.get("outcome") for v, x in res.get("variants", {}).items()}}
        rep["repeat"][cid]["ok"] = bool(all(x is not None and x <= C.TOL["deterministic_m"] for x in det.values())
                                        and all(bytes_eq.values()) and all(kd_eq.values())
                                        and rep["repeat"][cid]["outcomes_equal"])
        rep["cases"][cid] = res
    # originals
    _, man = _manifests(o)
    rep["originals"] = {t: {k: man[t][k] for k in ("manifest_sha_ok", "files", "bad", "files_ok")} for t in man}
    shared = git("status", "--porcelain", "--", *SHARED_PATHS).splitlines()
    rep["shared_tracked_changes"] = [s for s in shared if not s.startswith("??")]
    ver = json.loads(Path(o.verify).read_text())
    cs = rep["cases"]
    sm = cs[C.SMOOTH_CONTROL]
    smooth_valid = bool(sm["P0"].get("ok") and sm["source"]["ok"] and sm["matching_ok"] and "control" in sm
                        and sm.get("source_min_interval_change_m") is not None
                        and sm["source_min_interval_change_m"] > C.TOL["smooth_min_interval_change_m"]
                        and sm["control"]["errors_vs_source"]["subframe"]["max_abs_coord_m"] <= C.TOL["smooth_sufficiency_m"]
                        and sm["files"][C.CONTROL]["import"]["imported_interpolation"] == ["LINEAR"])
    held_ok = all(cs[c].get("source_held_by_evaluation") for c in C.TARGET_CASES)
    enc_valid = all(rep["encode_valid"]) and all(cs[c]["files"][t]["checks"]["parsers_agree"]
                                                 and cs[c]["files"][t]["checks"]["addon_masked_equals_control"]
                                                 for c in cs for t in cs[c]["files"])
    pre = {"preflight_ok": bool(ver["ok"] and not rep["shared_tracked_changes"] and all(rep["probe_bound"])),
           "encoding_valid": bool(enc_valid),
           "sources_ok": bool(all(cs[c]["source"]["ok"] for c in cs) and held_ok),
           "matching_ok": bool(all(cs[c]["matching_ok"] for c in cs)),
           "deterministic": bool(all(rep["repeat"][c]["ok"] for c in cs)),
           "originals_unchanged": bool(all(v["manifest_sha_ok"] and v["files_ok"] for v in rep["originals"].values())),
           "p0": {c: bool(cs[c]["P0"].get("ok")) for c in cs},
           "smooth_valid": smooth_valid,
           "results": {v: {c: {k: cs[c].get("variants", {}).get(v, MISSING)[k] for k in ("outcome", "keydata_ok",
                                                                                         "integer_ok")}
                           for c in C.TARGET_CASES} for v in C.VARIANTS}}
    case, why = C.decide_case(pre)
    rep["decision_inputs"] = pre
    rep["CASE"], rep["CASE_reason"] = case, why
    s = wjson(out/"analysis_report.json", rep)

    def brief(r):
        b = {"P0_ok": r["P0"].get("ok"), "source_ok": r["source"]["ok"], "matching_ok": r["matching_ok"]}
        if "control" in r:
            b["control_subframe_m"] = r["control"]["errors_vs_source"]["subframe"]["max_abs_coord_m"]
        for v, x in r.get("variants", {}).items():
            b[v] = {k: x.get(k) for k in ("outcome", "imported_interpolation", "flag_read", "delta_vs_control_m",
                                          "declared_error_m")}
        return b
    print(json.dumps({"CASE": case, "reason": why, "sha256": s, "cases": {c: brief(cs[c]) for c in cs},
                      "decision": {k: v for k, v in pre.items() if k != "results"}}, indent=1, default=str))
    return 0


# --- manifest -----------------------------------------------------------------------------------------------------
def manifest(o):
    base = Path(o.base)
    files = {}
    for p in sorted(base.rglob("*")):
        if p.is_file() and p.name not in ("evidence_manifest.json", "manifest_stdout.txt"):
            files[p.relative_to(base).as_posix()] = {"sha256": sha256(p), "bytes": p.stat().st_size}
    _, man = _manifests(o)
    unchanged = {t: man[t]["manifest_sha_ok"] and man[t]["files_ok"] for t in man}
    rep = {"schema": C.SCHEMA + ":manifest", "starting_head": o.starting_head, "declaration_sha256": o.declaration_sha,
           "originals_unchanged": unchanged, "all_originals_unchanged": all(unchanged.values()), "files": files}
    s = wjson(base/"evidence_manifest.json", rep)
    print(s, len(files), rep["all_originals_unchanged"])
    return 0 if rep["all_originals_unchanged"] else 2


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=("verify", "declare", "encode", "analyze", "manifest"))
    for name in ("base", "w106", "w107", "w108", "verify", "options", "lock", "blender", "addon_dir", "starting_head",
                 "branch", "declaration", "declaration_sha", "out_dir", "encode_dir", "encode_repeat_dir", "probe_dir",
                 "repeat_dir"):
        ap.add_argument("--"+name.replace("_", "-"))
    o = ap.parse_args()
    return {"verify": verify, "declare": declare, "encode": encode, "analyze": analyze, "manifest": manifest}[o.mode](o)


if __name__ == "__main__":
    sys.exit(main())
