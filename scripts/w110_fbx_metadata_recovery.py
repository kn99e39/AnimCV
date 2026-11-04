#!/usr/bin/env python3
"""W110 host-side stages: FBX-embedded interpolation-metadata recovery feasibility (evidence only).

  verify    W109 COMPLETE marker and manifest / declaration identity, every W106-W109 manifest file, frozen exporter,
            unchanged FBX add-on sources, reused-script SHAs, worker lock (Batch 110 RUNNING), Git state, no other
            Blender process; byte-verified scratch copies of the inputs into <base>/results/inputs
  declare   fixed cases, selected takes and FBX-side curve identities (adapter map of the unmodified V1 inputs), the
            producer rule, the consumer mapping rule (adapter SHA), the mixed construction, sample grids, metrics,
            tolerances, negative controls, CASE rule, repeat, stop conditions, resource cap -> declaration.json + SHA
  encode    PRODUCER: per case, flags from the frozen rule + the Blender source census; byte-exact in-place rewrite of
            only the targeted KeyAttrFlags (0x6104 -> 0x6002); validity (byte spans, masked trees, decoded flags,
            unchanged map); writes control.fbx / meta.fbx and the source-free consumer_input.json
  analyze   P0, construction, coverage, isolation, integrity, attribution, recovery, components, controls, repeat,
            originals -> analysis_report.json + CASE
  manifest  hashes of every output plus a re-check that no W106-W109 original changed

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
import w110_fbx_metadata_recovery_contract as C  # noqa: E402
import w110_fbx_consumer_adapter as AD  # noqa: E402  (pure FBX-side map only; same rule the consumer runs)
import w108_fbx_subframe_contract as C8  # noqa: E402
import w109_fbx_constant_import_contract as C9  # noqa: E402
from w108_fbx_subframe import check_manifest, git, wjson  # noqa: E402  (read-only reuse)

sha256 = C.sha256_file
W109_WORKLOG = "docs/109_WORKLOG_FBX_IMPORTER_CONSTANT_FLAG_OWNERSHIP.md"
MANIFEST_SHA = {"w109": "e62e8c657267345dcc40d6f41f099269255a120b61bd931d3599d2cb6ababdd1",
                "w108": "bffcd9ca551744d4ebd4e5087f42ea99a0ad6c8e29395c0d0895ddd8c8768375",
                "w107": "20fcb1b9f6ec8879247ac91ae3a21ca1185e995340976079bc80936316f57278",
                "w106": "f3003fe54e68c6cf182313c0b784468275b7ed039f50f574ad10e64eb8fb97db"}
W109_DECLARATION_SHA = "b6bf24049e4be1270c3b3d64ff2d119188b87a8c89b79ac8ac951eac2b986f73"
W108_DECLARATION_SHA = "0509a38c882386b35a2d895a8963570a8a3ebb875f2f475818297096b343112d"
EXPORTER = "src/blender/export.py"
EXPORTER_SHA = "521a63d96401f6f82b072ab8c82800e7795623a66370d70a257d72591f76b5fa"
MANIFEST_LOG_EXCEPTIONS = {"w109": "manifest_stdout.txt", "w108": "manifest_stdout.txt", "w107": "manifest.log",
                           "w106": "results/manifest.log"}
REUSED_SCRIPTS = ("scripts/w109_fbx_constant_import_contract.py", "scripts/w109_fbx_constant_import_blender.py",
                  "scripts/w108_fbx_subframe_contract.py", "scripts/w108_fbx_subframe_blender.py",
                  "scripts/w108_fbx_subframe.py", "scripts/w107_fbx_blender.py", "scripts/w107_fbx_contract.py",
                  "scripts/w106_action_blender.py", "scripts/w106_action_contract.py",
                  "scripts/render_runtime_heading_rig_blender.py")
W110_SCRIPTS = ("scripts/w110_fbx_metadata_recovery_contract.py", "scripts/w110_fbx_consumer_adapter.py",
                "scripts/w110_fbx_metadata_recovery_blender.py", "scripts/w110_fbx_metadata_recovery.py")
EXPECTED_ENV = {"blender_version": "5.1.0", "fbx_addon_version": [5, 15, 0]}
SHARED_PATHS = ["src", "tests", "scripts", "examples", "docs", "Architecture_v2.md", "Architecture_v3_FramePose.md"]
W110_NEW = ("scripts/w110_", "tests/test_w110_", "docs/110_WORKLOG_")
CONSUMER_KEYS = {"schema", "blender_version", "cases"}
CONSUMER_CASE_KEYS = {"stack", "files", "times", "nb_times"}


def _inputs_for(cid):
    if cid in C.REAL_CASES:
        return ("w107", f"results/inputs/{cid}_n0.blend"), ("w107", f"results/probe/{cid}/fbx/V1.fbx")
    return ("w108", f"results/probe/{cid}/input.blend"), ("w108", f"results/probe/{cid}/fbx/V1.fbx")


def _manifests(o):
    bases = {t: Path(getattr(o, t)) for t in ("w109", "w108", "w107", "w106")}
    out = {}
    for tag, b in bases.items():
        s = sha256(b/"evidence_manifest.json")
        man = json.loads((b/"evidence_manifest.json").read_text())
        chk = check_manifest(b, man, MANIFEST_LOG_EXCEPTIONS[tag])
        out[tag] = {"manifest_sha256": s, "manifest_sha_ok": s == MANIFEST_SHA[tag], "files": chk["files"],
                    "bad": chk["bad"], "files_ok": chk["ok"], "exception": chk["exception"],
                    "all_originals_unchanged": bool(man.get("all_originals_unchanged"))}
    return bases, out


# --- verify -------------------------------------------------------------------------------------------------------
def verify(o):
    base = Path(o.base)
    out = base/"results"/"verify"
    out.mkdir(parents=True, exist_ok=False)
    rep = {"schema": C.SCHEMA + ":verify", "checks": {}, "stop": []}
    wl = (ROOT/W109_WORKLOG).read_text(encoding="utf-8")
    rep["checks"]["w109_complete_marker"] = wl.rstrip().endswith("AUTOMATION_BATCH_STATUS: COMPLETE\nBATCH_NUMBER: 109")
    bases, man = _manifests(o)
    rep["manifests"] = man
    for tag in man:
        rep["checks"][f"{tag}_manifest_sha"] = man[tag]["manifest_sha_ok"]
        rep["checks"][f"{tag}_files_intact"] = man[tag]["files_ok"]
    d9sha = sha256(bases["w109"]/"results/declare/declaration.json")
    d8sha = sha256(bases["w108"]/"results/declare/declaration.json")
    rep["checks"]["w109_declaration_sha"] = {"value": d9sha, "ok": d9sha == W109_DECLARATION_SHA}
    rep["checks"]["w108_declaration_sha"] = {"value": d8sha, "ok": d8sha == W108_DECLARATION_SHA}
    d9 = json.loads((bases["w109"]/"results/declare/declaration.json").read_text())
    exp_sha = sha256(ROOT/EXPORTER)
    rep["checks"]["exporter_sha"] = {"value": exp_sha, "ok": exp_sha == EXPORTER_SHA}
    scr = {p: sha256(ROOT/p) for p in REUSED_SCRIPTS}
    rep["reused_scripts_sha256"] = scr
    rep["checks"]["reused_scripts_equal_w109_declaration"] = all(
        scr[p] == d9["scripts_sha256"][p] for p in REUSED_SCRIPTS if p in d9["scripts_sha256"])
    rep["reused_scripts_not_in_w109_declaration"] = [p for p in REUSED_SCRIPTS if p not in d9["scripts_sha256"]]
    addon = Path(o.addon_dir)
    a_sha = {p.name: sha256(p) for p in sorted(addon.glob("*.py"))}
    rep["fbx_addon_source_sha256"] = a_sha
    rep["checks"]["fbx_addon_unchanged_vs_w109"] = a_sha == d9["environment"]["fbx_addon_source_sha256"]
    rep["blender_exe_sha256"] = sha256(o.blender)
    rep["checks"]["blender_exe_equals_w109"] = rep["blender_exe_sha256"] == d9["environment"]["blender_exe_sha256"]
    head = git("rev-parse", "HEAD")
    status = git("status", "--porcelain", "--untracked-files=all").splitlines()
    tracked = [s for s in status if not s.startswith("??")]
    untracked = [s[3:] for s in status if s.startswith("??")]
    rep["git"] = {"head": head, "branch": git("branch", "--show-current"), "origin": git("rev-parse", "origin/" + o.branch),
                  "tracked_changes": tracked, "untracked": untracked}
    rep["checks"]["starting_head"] = {"value": head, "ok": head == o.starting_head}
    rep["checks"]["branch"] = rep["git"]["branch"] == o.branch
    rep["checks"]["origin_equals_head"] = rep["git"]["origin"] == head
    rep["checks"]["working_tree_only_new_w110_files"] = not tracked and all(u.startswith(W110_NEW) for u in untracked)
    lk = json.loads(Path(o.lock).read_text()) if Path(o.lock).exists() else {}
    rep["worker_lock"] = lk
    rep["checks"]["worker_lock_batch_110_running"] = lk.get("batch") == 110 and lk.get("state") == "RUNNING" \
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
    for cid in C.REAL_CASES + (C.STEP, C.SMOOTH):
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
    rel = "results/inputs/w106_synthetic_n0.blend"
    dst = inputs/"w106_synthetic_n0.blend"
    shutil.copyfile(bases["w108"]/rel, dst)
    copies["synthetic_n0"] = {"source": f"w108:{rel}", "path": str(dst), "sha256": sha256(dst),
                              "manifest_sha256": mans["w108"]["files"].get(rel, {}).get("sha256")}
    copies["synthetic_n0"]["ok"] = copies["synthetic_n0"]["sha256"] == copies["synthetic_n0"]["manifest_sha256"]
    if not copies["synthetic_n0"]["ok"]:
        rep["stop"].append("copy:synthetic_n0")
    rep["inputs"] = copies
    rep["ok"] = not rep["stop"]
    wjson(out/"verify_report.json", rep)
    print(json.dumps({"ok": rep["ok"], "stop": rep["stop"]}))
    return 0 if rep["ok"] else 2


# --- declare ------------------------------------------------------------------------------------------------------
def _map_summary(fmap):
    st = {}
    for r in fmap["curves"]:
        st[r["status"]] = st.get(r["status"], 0) + 1
    return {"status": fmap["status"], "curves": len(fmap["curves"]), "curve_status": st, "models": len(fmap["models"]),
            "models_separable": sum(1 for m in fmap["models"].values() if m["separable"]),
            "uids_sha256": hashlib.sha256(",".join(str(r["uid"]) for r in fmap["curves"]).encode()).hexdigest(),
            "map_digest_without_flags": AD.fbx_map_digest(fmap, with_flags=False),
            "flags": sorted({h for r in fmap["curves"] for h in r["flags_hex"]})}


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
    w109, w108 = Path(o.w109), Path(o.w108)
    d9 = json.loads((w109/"results/declare/declaration.json").read_text())
    d8 = json.loads((w108/"results/declare/declaration.json").read_text())
    cases = {}
    for cid in C.REAL_CASES + (C.STEP, C.SMOOTH):
        c9 = d9["cases"][cid]
        inp = ver["inputs"][cid]
        root, version = C8.fbx_read(inp["fbx"]["path"])
        stack = c9["fbx"]["selected_stack"]
        fmap = AD.fbx_curve_map(root, stack)
        ms = _map_summary(fmap)
        if fmap["status"] != "OK" or ms["curves"] != c9["fbx"]["target_curves"] or ms["flags"] != ["0x6104"]:
            raise SystemExit(f"STOP: {cid}: FBX-side map of the unmodified input is not as W109 recorded: {ms}")
        classes = {int(k): v for k, v in c9["interval_classes"].items()}
        cases[cid] = {
            "id": cid, "role": "real (held source)" if cid in C.REAL_CASES else
            ("synthetic held-step control" if cid == C.STEP else "synthetic smooth LINEAR control"),
            "regime": c9["regime"], "action": c9["action"], "stack": stack,
            "source_blend": inp["blend"]["path"], "source_blend_sha256": inp["blend"]["sha256"],
            "source_blend_origin": inp["blend"]["source"], "v1_fbx": inp["fbx"]["path"],
            "v1_fbx_sha256": inp["fbx"]["sha256"], "v1_fbx_origin": inp["fbx"]["source"],
            "expected_source_action_digest": c9["expected_source_action_digest"],
            "expected_source_slot": c9["expected_source_slot"], "expected_source_interpolation":
            c9["expected_source_interpolation"], "expected_imported_action": c9["expected_imported_action"],
            "expected_imported_slot": c9["expected_imported_slot"], "timing": c9["timing"],
            "source_key_frames": c9["source_key_frames"], "times": c9["times"],
            "interval_classes": c9["interval_classes"], "nb_times": C.near_boundary_times(classes),
            "fbx_side_map_of_unmodified_v1": ms, "fbx_version": version,
            "p0_reference": c9["p0_reference"]}
    sy = d8["synthetic_controls"][C.STEP]
    classes = {int(k): v for k, v in sy["interval_classes"].items()}
    cases[C.MIXED] = {
        "id": C.MIXED, "role": "synthetic mixed held/smooth (new, predeclared)", "regime": "synthetic (separately "
        "labelled)", "action": C.MIXED_SPEC["action"], "stack": C.MIXED_SPEC["stack"], "build": True,
        "timing": sy["timing"], "source_key_frames": C.MIXED_SPEC["key_frames"], "times": sy["times"],
        "interval_classes": sy["interval_classes"], "nb_times": C.near_boundary_times(classes),
        "expected_fbx": C.MIXED_SPEC["expected_fbx"]}
    if sy["source_key_frames"] != C.MIXED_SPEC["key_frames"]:
        raise SystemExit("STOP: synthetic key frames differ from the mixed specification")
    adapter_sha = sha256(ROOT/"scripts/w110_fbx_consumer_adapter.py")
    decl = {
        "schema": C.SCHEMA + ":declaration", "outcomes_read": False,
        "note": "Frozen before the mixed Action is built, before any W110 FBX is encoded or imported and before any "
                "adapter run. Only unmodified historical inputs were parsed (FBX-side map, flags).",
        "starting_head": o.starting_head, "verify_sha256": sha256(o.verify), "options_sha256": sha256(o.options),
        "environment": {"blender_version": opts["blender_version"], "blender_build_hash": opts["blender_build_hash"],
                        "blender_exe": o.blender, "blender_exe_sha256": ver["blender_exe_sha256"],
                        "fbx_addon_version": opts["fbx_addon"]["version"],
                        "fbx_addon_source_sha256": ver["fbx_addon_source_sha256"],
                        "launch": "-b --factory-startup -t 2 --python-exit-code 1"},
        "scripts_sha256": {p: sha256(ROOT/p) for p in (*W110_SCRIPTS, *REUSED_SCRIPTS, EXPORTER)},
        "cases": cases,
        "synthetic_input": {"path": ver["inputs"]["synthetic_n0"]["path"],
                            "sha256": ver["inputs"]["synthetic_n0"]["sha256"],
                            "origin": ver["inputs"]["synthetic_n0"]["source"]},
        "mixed": C.MIXED_SPEC,
        "exporter": {"v0_kwargs": d8["exporter"]["v0_kwargs"], "v1": d8["exporter"]["v1"],
                     "used_for": "the mixed case only (one export per produce run); real / step / smooth reuse their "
                                 "frozen V1 FBX"},
        "producer": {
            "rule": C.producer_directives.__doc__.strip(),
            "flag_written": f"0x{C.CONSTANT_FLAG:x} (W109 CONSTANT-standard; derived from 0x{C.SOURCE_FLAG:x} by "
                           "clearing 0xE|0x100 and setting 0x2); CONSTANT-next 0x6102 never used",
            "edit": "byte-exact in-place rewrite of the KeyAttrFlags int32 value of selected-take curves whose producer "
                    "directive is CONSTANT (W109 patch_flags); every other byte unchanged",
            "source_access": "allowed here only (Blender census of the selected source Action + host rule)",
            "refuse_if": "any selected-take curve UNMATCHED (model not a source item) or UNSUPPORTED (mixed / BEZIER "
                         "source interpolation); a target flag not 0x6104; compressed / non-int32 flag array"},
        "consumer": {
            "adapter": "scripts/w110_fbx_consumer_adapter.py", "adapter_sha256": adapter_sha,
            "inputs": "FBX file (bytes, via io_scene_fbx.parse_fbx), selected take name, the freshly imported scene",
            "mapping_rule": AD.__doc__.strip(),
            "plan": AD.plan.__doc__.strip(),
            "edit": "keyframe_points.foreach_set('interpolation', ...) on F-curves of CONSTANT / LINEAR groups whose "
                    "current interpolation differs; nothing else; no fcurve.update()",
            "isolation": {"static": "the adapter module imports only stdlib, numpy, bpy, bpy_extras and "
                                    "io_scene_fbx.parse_fbx; no repo module; no path other than its FBX argument",
                          "runtime": "consume stage reads only consumer_input.json (host whitelist: stack, FBX paths/"
                                     "SHAs, sample grids); a Python audit hook records every file open during "
                                     "recover(); only the FBX and Python module files are allowed",
                          "integrity": "snapshots before/after recover(): selected Action key data except "
                                       "interpolation, every other Action, Action/slot inventory, object types / "
                                       "rotation modes / matrix_basis / parents, rest matrices, bone counts, pose "
                                       "rotation modes, scene fps/range, filepath, libraries -- all unchanged; "
                                       "changed-curve count == adapter count"}},
        "conditions": {"control_U": "unchanged V1 FBX, untouched import (P0 vs W108/W107)",
                       "control_A": "unchanged V1 FBX + adapter (consumer alone; expected no change)",
                       "meta_U": "metadata FBX, untouched import (producer alone; expected == control_U, W109)",
                       "meta_A": "metadata FBX + adapter (the tested route)",
                       "meta_K": "metadata FBX + blanket CONSTANT on every key of the selected imported Action "
                                 "(negative control; harness, not adapter)"},
        "samples": {"main": "W108's explicit per-case grid: every integer frame plus +0.25/+0.50/+0.75 of every frame "
                            "but the last (mixed: the W108 synthetic grid)",
                    "near_boundary": "t = f + 0.99 for every transition interval [f, f+1) (W108 classes; source key "
                                     "at f+1)",
                    "interval_classes": "W108 (source key times only)"},
        "metrics": {
            "world": "max |coordinate| world head/tail error vs the source Action on common source bones (W108 gate "
                     "form), by phase, transition / quiet, integer, near-boundary and all samples; object rotation / "
                     "translation and armature-space tail error as diagnostics",
            "mixed_components": C.MIXED_SPEC["component_points"],
            "channel_function": C.channel_function_error.__doc__.strip(),
            "coverage": "adapter plan accounting: FBX curves mapped / unresolved by status, groups by directive, "
                        "imported F-curves mapped / unmapped, F-curves claimed twice, per-axis 1:1 count",
            "identity": "masked FBX tree == control; import key data (excluding interpolation) of meta_U / meta_A / "
                        "meta_K == control_U; meta_A interpolation changes only on directed curves",
            "cost": "FBX bytes / key counts (unchanged by construction), stage wall clock, adapter seconds"},
        "tolerances": C.TOL,
        "gates": {"recovered": "meta_A all-sample max <= recovered_m and integer max <= integer_gate_m",
                  "smooth_preserved": "syn_smooth meta_A all-sample max <= recovered_m",
                  "mixed_ok": "mixed meta_A held and smooth component max <= recovered_m, all-sample max <= "
                              "recovered_m, integer <= integer_gate_m",
                  "controls_discriminate": "meta_U all-sample max > harm_m on every held case; mixed meta_U held "
                                           "component > harm_m; syn_smooth meta_K and mixed meta_K smooth component "
                                           "> harm_m",
                  "attribution": "control_A and meta_U each within deterministic_m of control_U on every case (and "
                                 "control_A changes no curve)",
                  "coverage": "per case: map status OK; all selected-take FBX curves mapped (0 unresolved); every "
                              "imported F-curve of the target Action mapped; 0 claimed twice; adapter CONSTANT curve "
                              "set == producer CONSTANT set (meta) / empty (control)"},
        "case_rule": C.decide_case.__doc__.strip(),
        "p0": "historical cases: fresh source samples == W108 source.npz (bitwise); source Action digest / slot / "
              "timing / key frames / interpolation == declared; control_U vs W108 V1.npz <= deterministic_m; real "
              "cases also vs W107 V1.npz at common times; control_U subframe and integer max vs W108 analysis <= "
              "w108_metric_reproduce_m; mixed: valid construction (key data == SYN baseline except interpolation, "
              "declared per-bone interpolation, key frames, base unchanged) and its V1 export leaves the Action and "
              "scratch file unchanged",
        "negative_controls": {"NC1": "no-op identity of the patcher (0x6104 rewrite reproduces the bytes)",
                              "NC2": "control_U P0 replay",
                              "NC3": "control_A: adapter on the unmodified V1 FBX changes nothing (consumer alone)",
                              "NC4": "meta_U == control_U (producer alone; W109 importer discard)",
                              "NC5": "meta_K: blanket CONSTANT harms the smooth components",
                              "NC6": "non-selected takes keep 0x6104 and their imported Actions are unchanged by the "
                                     "adapter",
                              "NC7": "fresh repeat (produce, encode, consume)",
                              "NC8": "W106-W109 originals unchanged"},
        "repeat": "produce, encode and consume run twice into fresh directories; mixed V1 FBX compared by canonical "
                  "tree with FBXHeaderExtension/CreationTimeStamp skipped (exporter writes the wall clock there); all "
                  "other FBX bytes, maps, directives, plans and samples must agree (<= deterministic_m)",
        "stop_conditions": ["declaration / consumer-input hash mismatch", "Blender version differs",
                            "input hash mismatch", "FBX-side map of an unmodified input differs from W109",
                            "producer directive UNMATCHED / UNSUPPORTED -> encode invalid, no import",
                            "invalid mixed construction -> no export", "shared/production file changed",
                            "resource cap exceeded"],
        "resource_cap": {"blender_runs": "options + 2 produce + 2 consume", "wall_clock_per_run_s": 900,
                         "disk_gb": 3, "gpu": "none", "network": "none"},
        "hypothesis": {"H110": "per-curve FBX CONSTANT/LINEAR metadata plus the frozen deterministic correspondence is "
                               "sufficient for a consumer-only adapter to reconstruct the held real Actions, syn_step "
                               "and the mixed held/smooth Action through the stock importer while preserving LINEAR "
                               "channels"},
    }
    path = out/"declaration.json"
    wjson(path, decl)
    s = sha256(path)
    (out/"declaration.sha256").write_text(s + "\n")
    print(s)
    return 0


# --- encode (producer) ---------------------------------------------------------------------------------------------
def _consumer_clean(ci, root=None):
    """Structural whitelist: only the declared keys; every file is a control.fbx / meta.fbx (under ``root`` when
    given); no .blend / census / declaration / source reference anywhere.  (Take names legitimately carry historical
    tags such as 'W108 SYN step CONSTANT', so tags are not banned lexically.)"""
    if set(ci) != CONSUMER_KEYS:
        return False
    blob = json.dumps(ci).lower()
    if any(w in blob for w in (".blend", "census", "declaration", "source", ".npz")):
        return False
    for c in ci["cases"].values():
        if set(c) != CONSUMER_CASE_KEYS or set(c["files"]) != {"control", "meta"}:
            return False
        for tag, f in c["files"].items():
            p = Path(f["path"])
            if p.name != f"{tag}.fbx" or (root is not None and Path(root).resolve() not in p.resolve().parents):
                return False
    return True


def encode(o):
    decl_bytes = Path(o.declaration).read_bytes()
    if hashlib.sha256(decl_bytes).hexdigest() != o.declaration_sha:
        raise SystemExit("STOP: declaration hash mismatch")
    decl = json.loads(decl_bytes)
    prod = json.loads(Path(o.produce_report).read_text())
    if prod["declaration_sha256"] != o.declaration_sha:
        raise SystemExit("STOP: produce report bound to another declaration")
    out = Path(o.out_dir)
    out.mkdir(parents=True, exist_ok=False)
    rep = {"schema": C.SCHEMA + ":encode", "declaration_sha256": o.declaration_sha,
           "produce_report_sha256": sha256(o.produce_report), "cases": {}, "problems": []}
    ci = {"schema": C.SCHEMA + ":consumer_input", "blender_version": decl["environment"]["blender_version"], "cases": {}}
    for cid, case in decl["cases"].items():
        pr = prod["cases"][cid]
        d = out/cid
        d.mkdir()
        if case.get("build"):
            if not pr.get("build", {}).get("valid_construction") or "v1_export" not in pr:
                rep["problems"].append(f"{cid}:invalid construction / no export")
                continue
            src = Path(pr["v1_export"]["fbx"])
            want = pr["v1_export"]["fbx_sha256"]
        else:
            src, want = Path(case["v1_fbx"]), case["v1_fbx_sha256"]
        data = src.read_bytes()
        if hashlib.sha256(data).hexdigest() != want:
            raise SystemExit(f"STOP: {src} hash mismatch")
        (d/"control.fbx").write_bytes(data)
        root0, version = C8.fbx_read(d/"control.fbx")
        fmap = AD.fbx_curve_map(root0, case["stack"])
        ms = _map_summary(fmap)
        crec = {"control_sha256": sha256(d/"control.fbx"), "control_bytes": len(data), "fbx_version": version,
                "map_summary": ms}
        if "fbx_side_map_of_unmodified_v1" in case:
            crec["map_equals_declared"] = ms == case["fbx_side_map_of_unmodified_v1"]
        else:
            ef = case["expected_fbx"]
            crec["map_equals_declared"] = fmap["status"] == "OK" and ms["curves"] == ef["selected_stack_curves"]
        dirs = C.producer_directives(fmap, pr["census"])
        counts = C.producer_counts(dirs)
        crec["producer_counts"] = counts
        crec["census"] = {k: pr["census"][k] for k in ("armature_object", "groups", "unsupported_paths", "fcurves")}
        crec["producer_directives"] = {str(u): v for u, v in sorted(dirs.items())}
        const = sorted(u for u, (dv, _) in dirs.items() if dv == "CONSTANT")
        crec["constant_uids"] = const
        noop, _ = C9.patch_flags(data, [r["uid"] for r in fmap["curves"]], C.SOURCE_FLAG)
        crec["noop_identity"] = noop == data
        bad = {k: v for k, v in counts.items() if k in ("UNMATCHED", "UNSUPPORTED")}
        if bad or fmap["status"] != "OK" or not crec["map_equals_declared"] or not crec["noop_identity"]:
            rep["problems"].append(f"{cid}:producer {bad} / map {fmap['status']} / declared "
                                   f"{crec['map_equals_declared']} / noop {crec['noop_identity']}")
            rep["cases"][cid] = crec
            continue
        if "expected_fbx" in case:
            ef = case["expected_fbx"]
            crec["mixed_expected_counts_ok"] = (counts.get("CONSTANT", 0) == ef["constant_curves"]
                                                and counts.get("LINEAR", 0) == ef["linear_driven_curves"]
                                                and counts.get("UNDRIVEN", 0) == ef["undriven_curves"])
            if not crec["mixed_expected_counts_ok"]:
                rep["problems"].append(f"{cid}:producer counts {counts} differ from the declared expectation")
        if const:
            new, recs = C9.patch_flags(data, const, C.CONSTANT_FLAG)
        else:
            new, recs = data, []
        p = d/"meta.fbx"
        p.write_bytes(new)
        diff = C9.byte_diff_spans(data, new, [(r["offset"], r["bytes"]) for r in recs])
        root1, _ = C8.fbx_read(p)
        fmap1 = AD.fbx_curve_map(root1, case["stack"])
        fl1 = {r["uid"]: r["flags_hex"] for r in fmap1["curves"]}
        all1 = C9.curve_flags(root1)
        all0 = C9.curve_flags(root0)
        st = C8.fbx_anim_summary(root1, case["timing"]["fps"]/case["timing"]["fps_base"])["stacks"]
        st0 = C8.fbx_anim_summary(root0, case["timing"]["fps"]/case["timing"]["fps_base"])["stacks"]
        vr = {"sha256": sha256(p), "bytes": p.stat().st_size, "patched_curves": len(recs), "byte_diff": diff,
              "map_digest_without_flags": AD.fbx_map_digest(fmap1, with_flags=False),
              "selected_stack_summary": st.get(case["stack"])}
        vr["checks"] = {
            "same_length": diff["same_length"],
            "diff_only_in_target_spans": diff["all_in_targets"],
            "every_target_changed": (diff["diff_bytes"] or 0) > 0 if const else diff["diff_bytes"] == 0,
            "masked_tree_equals_control": C.tree_digest(root1, const) == C.tree_digest(root0, const),
            "map_unchanged_except_flags": vr["map_digest_without_flags"] == ms["map_digest_without_flags"],
            "constant_targets_decoded": all(fl1[u] == [f"0x{C.CONSTANT_FLAG:x}"] for u in const)
            and AD.flag_class(C.CONSTANT_FLAG) == "CONSTANT",
            "every_other_curve_unchanged": all(all1[u] == all0[u] for u in all0 if u not in set(const)),
            "selected_stack_keys_unchanged": (st.get(case["stack"], {}).get("keys") == st0.get(case["stack"], {}).get("keys")
                                              and st.get(case["stack"], {}).get("refcount_covers_keys") is True),
            "other_stacks_unchanged": {k: v for k, v in st.items() if k != case["stack"]}
            == {k: v for k, v in st0.items() if k != case["stack"]}}
        vr["valid"] = all(vr["checks"].values())
        if not vr["valid"]:
            rep["problems"].append(f"{cid}:meta {[k for k, x in vr['checks'].items() if not x]}")
        crec["meta"] = vr
        crec["tree_digest_full_control"] = C.tree_digest(root0)
        crec["tree_digest_control_no_timestamp"] = C.tree_digest(root0, skip_paths=C.TIMESTAMP_PATHS)
        crec["tree_digest_meta_no_timestamp"] = C.tree_digest(root1, skip_paths=C.TIMESTAMP_PATHS)
        rep["cases"][cid] = crec
        ci["cases"][cid] = {"stack": case["stack"], "times": case["times"], "nb_times": case["nb_times"],
                            "files": {"control": {"path": str(d/"control.fbx"), "sha256": crec["control_sha256"]},
                                      "meta": {"path": str(p), "sha256": vr["sha256"]}}}
    rep["consumer_input_clean"] = _consumer_clean(ci, out)
    if not rep["consumer_input_clean"]:
        rep["problems"].append("consumer input not source-free")
    rep["valid"] = not rep["problems"]
    cs = wjson(out/"consumer_input.json", ci)
    rep["consumer_input_sha256"] = cs
    s = wjson(out/"encode_report.json", rep)
    print(json.dumps({"valid": rep["valid"], "problems": rep["problems"], "sha256": s, "consumer_input_sha256": cs,
                      "counts": {c: r.get("producer_counts") for c, r in rep["cases"].items()}}))
    return 0 if rep["valid"] else 2


# --- analyze ------------------------------------------------------------------------------------------------------
def _ld(p):
    return C.load_npz(p)


def _cond(d, name):
    return _ld(d/f"{name}.npz"), _ld(d/f"{name}_nb.npz")


def coverage_of(adapter_rec, enc_case, tag):
    pl = adapter_rec["plan"]
    acc = pl.get("accounting", {})
    want_const = set(map(str, enc_case["constant_uids"])) if tag == "meta" else set()
    got_const = {u for u, v in pl["curve_status"].items() if v == "CONSTANT"}
    ok = bool(pl["status"] == "OK" and acc.get("fbx_curves_unresolved") == 0
              and acc.get("fbx_curves_mapped") == acc.get("fbx_curves") and acc.get("fbx_curves", 0) > 0
              and not acc.get("imported_fcurves_unmapped") and acc.get("fcurves_claimed_twice") == 0
              and acc.get("items_unresolved") == 0 and got_const == want_const
              and adapter_rec.get("applied") is not None)
    return {"ok": ok, "accounting": acc, "constant_set_equals_producer": got_const == want_const,
            "unresolved": pl.get("unresolved", [])[:20]}


def analyze_case(cid, case, enc, prb, cons, pdir, cdir, mixed_spec):
    T = C.TOL
    times = np.array(case["times"])
    classes = {int(k): v for k, v in case["interval_classes"].items()}
    src, src_nb = _ld(pdir/"source.npz"), _ld(pdir/"source_nb.npz")
    res = {"role": case["role"], "regime": case["regime"], "action": case["action"], "stack": case["stack"]}
    # source identity / construction
    if case.get("build"):
        b = prb["build"]
        ex = prb["v1_export"]
        held_t = []
        ph = C8.phase_of(times)
        idx = {float(t): i for i, t in enumerate(times) if ph[i] == 0}
        comp = C.component_points(mixed_spec)
        hp = C._points(src, comp["held"]["heads"], comp["held"]["tails"])
        sp = C._points(src, comp["smooth"]["heads"], comp["smooth"]["tails"])
        sm = []
        for i, t in enumerate(times):
            if ph[i] > 0:
                j = idx[float(np.floor(t))]
                held_t.append(float(np.abs(hp[i] - hp[j]).max()))
                sm.append(float(np.abs(sp[i] - sp[j]).max()))
        res["source"] = {"valid_construction": b["valid_construction"], "export_action_unmodified":
                         ex["action_unmodified"], "export_work_unchanged": ex["work_file_unchanged"],
                         "source_interpolation": prb["source_interpolation"],
                         "held_component_exactly_held": max(held_t) == 0.0,
                         "smooth_component_min_interval_change_m": min(sm),
                         "slot": prb["assigned"]["slot"], "timing": prb["timing"]}
        res["source"]["ok"] = bool(b["valid_construction"] and ex["action_unmodified"] and ex["work_file_unchanged"]
                                   and prb["work_file_unchanged"] and res["source"]["held_component_exactly_held"]
                                   and min(sm) > 1e-6 and prb["source_interpolation"] == ["CONSTANT", "LINEAR"]
                                   and prb["source_key_frames"] == case["source_key_frames"]
                                   and np.array_equal(src["times"], times))
        res["P0"] = {"ok": res["source"]["ok"], "note": "new case: construction validity replaces historical replay"}
    else:
        s = {"digest_ok": prb["source_action_digest"] == case["expected_source_action_digest"],
             "slot_ok": prb["assigned"]["slot"] == case["expected_source_slot"],
             "timing_ok": prb["timing"] == case["timing"], "frames_ok": prb["source_key_frames"] == case["source_key_frames"],
             "interp_ok": prb["source_interpolation"] == case["expected_source_interpolation"],
             "work_unchanged": prb["work_file_unchanged"] and prb["work_equals_input"]}
        s["ok"] = all(s.values()) and bool(np.array_equal(src["times"], times))
        res["source"] = s
        ref = case["p0_reference"]
        p0 = {"source_vs_w108_max_diff": C8.arrays_max_diff(src, _ld(ref["w108_source_npz"])),
              "w108_npz_unchanged": sha256(ref["w108_source_npz"]) == ref["w108_source_npz_sha256"]
              and sha256(ref["w108_v1_npz"]) == ref["w108_v1_npz_sha256"]}
        if (cdir/"control_U.npz").exists():
            cu = _ld(cdir/"control_U.npz")
            p0["control_vs_w108_v1_max_diff"] = C8.arrays_max_diff(cu, _ld(ref["w108_v1_npz"]))
            ce = C8.subframe_report(src, cu, times, classes)
            p0["subframe_minus_w108_m"] = ce["subframe"]["max_abs_coord_m"] - ref["w108_v1_subframe_max_m"]
            p0["integer_minus_w108_m"] = ce["integer"]["max_abs_coord_m"] - ref["w108_v1_integer_max_m"]
            if "w107_v1_npz" in ref:
                w7 = _ld(ref["w107_v1_npz"])
                ma, mb = C8.common_time_masks(cu["times"], w7["times"])
                p0["w107_all_present"] = bool(mb.all())
                p0["control_vs_w107_v1_max_diff"] = C8.arrays_max_diff(cu, w7, mask_a=ma, mask_b=mb)
            p0["ok"] = bool(p0["w108_npz_unchanged"] and p0["source_vs_w108_max_diff"] <= T["source_fidelity_m"]
                            and p0["control_vs_w108_v1_max_diff"] <= T["deterministic_m"]
                            and abs(p0["subframe_minus_w108_m"]) <= T["w108_metric_reproduce_m"]
                            and abs(p0["integer_minus_w108_m"]) <= T["w108_metric_reproduce_m"]
                            and ("w107_v1_npz" not in ref or (p0["w107_all_present"]
                                                             and p0["control_vs_w107_v1_max_diff"] <= T["deterministic_m"])))
        else:
            p0["ok"] = False
        res["P0"] = p0
    # conditions
    conds = {}
    for name, r in cons.items():
        if r.get("status") != "SAMPLED":
            conds[name] = {"status": r.get("status"), "error": r.get("error")}
            continue
        imp, imp_nb = _cond(cdir, name)
        e = C.overall(src, imp, src_nb, imp_nb, times, classes)
        c = {"status": "SAMPLED", "errors": e, "all_samples_max_m": e["all_samples_max_m"],
             "integer_max_m": e["integer"]["max_abs_coord_m"], "subframe_max_m": e["subframe"]["max_abs_coord_m"],
             "near_boundary_max_m": e["near_boundary"]["max_abs_coord_m"],
             "final_interpolation": r["final_interpolation"], "final_interp_tally": r["final_interp_tally"],
             "keydata_excl_interp_equals_control_U": r["final_keydata_digest_excl_interpolation"]
             == cons["control_U"].get("final_keydata_digest_excl_interpolation"),
             "import_keydata_equals_control_U": r["import_keydata_digest_all"]
             == cons["control_U"].get("import_keydata_digest_all"),
             "timing": r["timing"], "selected_action": r["selected_action"]}
        if case.get("build"):
            c["components"] = C.component_errors(src, imp, mixed_spec)
            c["components_nb"] = C.component_errors(src_nb, imp_nb, mixed_spec)
            c["components_max"] = {k: max(c["components"][k], c["components_nb"][k]) for k in c["components"]}
            c["channel_function"] = {b: C.channel_function_error(imp, times, b, "floor")
                                     for b in mixed_spec["held_bones"]}
            c["channel_function"].update({f"{b}(lerp)": C.channel_function_error(imp, times, b, "lerp")
                                          for b in mixed_spec["smooth_bones"]})
        if name.endswith("_A"):
            tag = name.split("_")[0]
            ad = r["adapter"]
            c["coverage"] = coverage_of(ad, enc, tag)
            c["access"] = r["access"]
            c["integrity"] = r["integrity"]
            c["applied"] = {k: ad["applied"][k] for k in ("fcurves_changed", "fcurves_already_correct", "keys_changed")} \
                if ad.get("applied") else None
            c["adapter_seconds"] = r.get("adapter_seconds")
            c["adapter_map_digest"] = ad["fbx_map_digest"]
            c["adapter_map_digest_without_flags"] = ad["fbx_map_digest_without_flags"]
            c["map_equals_host"] = ad["fbx_map_digest_without_flags"] == enc["map_summary"]["map_digest_without_flags"]
            c["isolation_ok"] = bool(r["access"]["isolated"] and r["access"]["fbx_reads"] >= 1
                                     and ad["fbx_unchanged_during_read"] and not ad["escape_mismatch"]
                                     and r["selected_equals_adapter_action"] and ad["fbx_sha256"] == r["fbx_sha256"])
        conds[name] = c
    res["conditions"] = conds
    ok = all(conds.get(k, {}).get("status") == "SAMPLED" for k in ("control_U", "control_A", "meta_U", "meta_A",
                                                                  "meta_K"))
    res["all_sampled"] = ok
    if not ok:
        return res
    cu, ca, mu = (_ld(cdir/f"{k}.npz") for k in ("control_U", "control_A", "meta_U"))
    cu_nb, ca_nb, mu_nb = (_ld(cdir/f"{k}_nb.npz") for k in ("control_U", "control_A", "meta_U"))
    res["attribution"] = {"control_A_vs_control_U_m": max(C8.arrays_max_diff(ca, cu), C8.arrays_max_diff(ca_nb, cu_nb)),
                          "meta_U_vs_control_U_m": max(C8.arrays_max_diff(mu, cu), C8.arrays_max_diff(mu_nb, cu_nb)),
                          "control_A_changed_curves": conds["control_A"]["integrity"]["selected_curves_interpolation_changed"],
                          "meta_U_import_keydata_equal": conds["meta_U"]["import_keydata_equals_control_U"]}
    a = res["attribution"]
    a["ok"] = bool(a["control_A_vs_control_U_m"] <= T["deterministic_m"] and a["meta_U_vs_control_U_m"] <= T["deterministic_m"]
                   and a["control_A_changed_curves"] == 0 and a["meta_U_import_keydata_equal"])
    ma = conds["meta_A"]
    res["recovered"] = bool(ma["all_samples_max_m"] <= T["recovered_m"] and ma["integer_max_m"] <= T["integer_gate_m"])
    res["coverage_ok"] = bool(conds["control_A"]["coverage"]["ok"] and ma["coverage"]["ok"])
    res["isolation_ok"] = bool(conds["control_A"]["isolation_ok"] and ma["isolation_ok"])
    res["integrity_ok"] = bool(conds["control_A"]["integrity"]["ok"] and ma["integrity"]["ok"]
                               and ma["keydata_excl_interp_equals_control_U"]
                               and conds["meta_K"]["keydata_excl_interp_equals_control_U"]
                               and ma["map_equals_host"] and conds["control_A"]["map_equals_host"])
    return res


def analyze(o):
    decl_bytes = Path(o.declaration).read_bytes()
    if hashlib.sha256(decl_bytes).hexdigest() != o.declaration_sha:
        raise SystemExit("STOP: declaration hash mismatch")
    decl = json.loads(decl_bytes)
    out = Path(o.out_dir)
    out.mkdir(parents=True, exist_ok=False)
    runs = []
    for pd, ed, cd in ((o.produce_dir, o.encode_dir, o.consume_dir),
                       (o.produce_repeat_dir, o.encode_repeat_dir, o.consume_repeat_dir)):
        runs.append({"pd": Path(pd), "ed": Path(ed), "cd": Path(cd),
                     "prod": json.loads((Path(pd)/"produce_report.json").read_text()),
                     "enc": json.loads((Path(ed)/"encode_report.json").read_text()),
                     "cons": json.loads((Path(cd)/"consume_report.json").read_text())})
    rep = {"schema": C.SCHEMA + ":analysis", "declaration_sha256": o.declaration_sha, "cases": {}, "repeat": {}}
    rep["bindings"] = [{"produce": r["prod"]["declaration_sha256"] == o.declaration_sha,
                        "encode": r["enc"]["declaration_sha256"] == o.declaration_sha
                        and r["enc"]["produce_report_sha256"] == sha256(r["pd"]/"produce_report.json"),
                        "consume": r["cons"]["consumer_input_sha256"] == sha256(r["ed"]/"consumer_input.json")
                        and r["cons"]["consumer_input_sha256"] == r["enc"]["consumer_input_sha256"],
                        "adapter_sha_declared": r["cons"]["adapter_sha256"] == decl["consumer"]["adapter_sha256"],
                        "encode_valid": r["enc"]["valid"], "consumer_input_clean": r["enc"]["consumer_input_clean"]}
                       for r in runs]
    mixed = decl["mixed"]
    for cid, case in decl["cases"].items():
        rr = [analyze_case(cid, case, r["enc"]["cases"][cid], r["prod"]["cases"][cid], r["cons"]["cases"].get(cid, {}),
                           r["pd"]/cid, r["cd"]/cid, mixed) for r in runs]
        res = rr[0]
        res["encode"] = {k: runs[0]["enc"]["cases"][cid].get(k) for k in ("producer_counts", "control_bytes",
                                                                         "map_summary", "noop_identity",
                                                                         "map_equals_declared", "census")}
        res["encode"]["meta"] = {k: runs[0]["enc"]["cases"][cid].get("meta", {}).get(k)
                                 for k in ("bytes", "patched_curves", "byte_diff", "checks", "valid",
                                           "selected_stack_summary")}
        res["repeat_run"] = {k: rr[1].get(k) for k in ("recovered", "coverage_ok", "isolation_ok", "integrity_ok")}
        # determinism between the two runs
        d1p, d2p = runs[0]["pd"]/cid, runs[1]["pd"]/cid
        d1c, d2c = runs[0]["cd"]/cid, runs[1]["cd"]/cid
        e1, e2 = runs[0]["enc"]["cases"][cid], runs[1]["enc"]["cases"][cid]
        det = {"source": C8.arrays_max_diff(_ld(d1p/"source.npz"), _ld(d2p/"source.npz")),
               "source_nb": C8.arrays_max_diff(_ld(d1p/"source_nb.npz"), _ld(d2p/"source_nb.npz"))}
        for name in ("control_U", "control_A", "meta_U", "meta_A", "meta_K"):
            for suf in ("", "_nb"):
                p1, p2 = d1c/f"{name}{suf}.npz", d2c/f"{name}{suf}.npz"
                det[name + suf] = C8.arrays_max_diff(_ld(p1), _ld(p2)) if p1.exists() and p2.exists() else None
        if case.get("build"):
            fbx_eq = (e1.get("tree_digest_control_no_timestamp") == e2.get("tree_digest_control_no_timestamp")
                      and e1.get("tree_digest_meta_no_timestamp") == e2.get("tree_digest_meta_no_timestamp"))
            byte_eq = {"control": e1.get("control_sha256") == e2.get("control_sha256"),
                       "meta": e1.get("meta", {}).get("sha256") == e2.get("meta", {}).get("sha256")}
        else:
            fbx_eq = e1.get("control_sha256") == e2.get("control_sha256") and \
                e1.get("meta", {}).get("sha256") == e2.get("meta", {}).get("sha256")
            byte_eq = {"control": fbx_eq, "meta": fbx_eq}
        c1, c2 = runs[0]["cons"]["cases"].get(cid, {}), runs[1]["cons"]["cases"].get(cid, {})
        plan_eq = all(c1.get(k, {}).get("adapter", {}).get(f) == c2.get(k, {}).get("adapter", {}).get(f)
                      for k in ("control_A", "meta_A") for f in ("fbx_map_digest", "plan"))
        changed_eq = all(c1.get(k, {}).get("changed_curves") == c2.get(k, {}).get("changed_curves")
                         for k in ("control_A", "meta_A"))
        census_eq = runs[0]["prod"]["cases"][cid].get("census") == runs[1]["prod"]["cases"][cid].get("census")
        dir_eq = e1.get("producer_directives") is not None and \
            sorted(map(tuple, e1["producer_directives"].values())) == sorted(map(tuple, e2["producer_directives"].values())) \
            and e1.get("producer_counts") == e2.get("producer_counts")
        rep["repeat"][cid] = {"max_diff": det, "fbx_equal": fbx_eq, "fbx_bytes_equal": byte_eq, "plans_equal": plan_eq,
                              "changed_curves_equal": changed_eq, "census_equal": census_eq, "directives_equal": dir_eq}
        rep["repeat"][cid]["ok"] = bool(all(x is not None and x <= C.TOL["deterministic_m"] for x in det.values())
                                        and fbx_eq and plan_eq and changed_eq and census_eq and dir_eq)
        rep["cases"][cid] = res
    # producer key identity: mixed held bones' FBX curves == syn_step's, sibling's == syn_smooth's
    rep["mixed_key_identity"] = _mixed_key_identity(runs[0], decl)
    _, man = _manifests(o)
    rep["originals"] = {t: {k: man[t][k] for k in ("manifest_sha_ok", "files", "bad", "files_ok")} for t in man}
    shared = git("status", "--porcelain", "--", *SHARED_PATHS).splitlines()
    rep["shared_tracked_changes"] = [s for s in shared if not s.startswith("??")]
    ver = json.loads(Path(o.verify).read_text())
    cs = rep["cases"]
    T = C.TOL

    def cond(c, k, f):
        return cs[c]["conditions"].get(k, {}).get(f)
    sampled = all(cs[c].get("all_sampled") for c in cs)
    pre = {"preflight_ok": bool(ver["ok"] and not rep["shared_tracked_changes"]
                                and all(all(b.values()) for b in rep["bindings"])),
           "encoding_valid": bool(all(r["enc"]["valid"] for r in runs)),
           "isolation_ok": bool(sampled and all(cs[c]["isolation_ok"] for c in cs)),
           "adapter_integrity": bool(sampled and all(cs[c]["integrity_ok"] for c in cs)),
           "sources_ok": bool(all(cs[c]["source"]["ok"] for c in cs)),
           "deterministic": bool(all(rep["repeat"][c]["ok"] for c in cs)),
           "originals_unchanged": bool(all(v["manifest_sha_ok"] and v["files_ok"] for v in rep["originals"].values())),
           "p0": {c: bool(cs[c]["P0"].get("ok")) for c in cs},
           "coverage": {c: bool(cs[c].get("coverage_ok")) for c in cs},
           "recovered": {c: bool(cs[c].get("recovered")) for c in cs}}
    if sampled:
        mx = cs[C.MIXED]["conditions"]
        pre["smooth_preserved"] = bool(cond(C.SMOOTH, "meta_A", "all_samples_max_m") <= T["recovered_m"])
        pre["mixed_held_ok"] = bool(mx["meta_A"]["components_max"]["held"] <= T["recovered_m"])
        pre["mixed_ok"] = bool(pre["mixed_held_ok"] and mx["meta_A"]["components_max"]["smooth"] <= T["recovered_m"]
                               and mx["meta_A"]["all_samples_max_m"] <= T["recovered_m"]
                               and mx["meta_A"]["integer_max_m"] <= T["integer_gate_m"])
        pre["controls_discriminate"] = bool(
            all(cond(c, "meta_U", "all_samples_max_m") > T["harm_m"] for c in C.HELD_CASES)
            and mx["meta_U"]["components_max"]["held"] > T["harm_m"]
            and cond(C.SMOOTH, "meta_K", "all_samples_max_m") > T["harm_m"]
            and mx["meta_K"]["components_max"]["smooth"] > T["harm_m"])
        pre["attribution"] = bool(all(cs[c]["attribution"]["ok"] for c in cs))
    else:
        pre.update({"smooth_preserved": False, "mixed_held_ok": False, "mixed_ok": False,
                    "controls_discriminate": False, "attribution": False})
    case, why = C.decide_case(pre)
    rep["decision_inputs"] = pre
    rep["CASE"], rep["CASE_reason"] = case, why
    s = wjson(out/"analysis_report.json", rep)

    def brief(r):
        b = {"P0": r["P0"].get("ok"), "source": r["source"]["ok"], "coverage": r.get("coverage_ok"),
             "recovered": r.get("recovered")}
        for k, c in r.get("conditions", {}).items():
            b[k] = {f: c.get(f) for f in ("all_samples_max_m", "integer_max_m", "near_boundary_max_m")}
            if "components_max" in c:
                b[k]["components"] = c["components_max"]
            if "applied" in c:
                b[k]["applied"] = c["applied"]
        return b
    print(json.dumps({"CASE": case, "reason": why, "sha256": s, "cases": {c: brief(cs[c]) for c in cs},
                      "decision": {k: v for k, v in pre.items()}}, indent=1, default=str))
    return 0


def _mixed_key_identity(run, decl):
    """Selected-take FBX key arrays of the mixed V1 vs the W108 step / smooth V1 files, per (model, property, axis)."""
    enc = run["enc"]["cases"]
    if C.MIXED not in enc or "meta" not in enc[C.MIXED]:
        return {"status": "NOT_AVAILABLE"}
    rm, _ = C8.fbx_read(run["ed"]/C.MIXED/"control.fbx")
    am = C.curve_arrays(rm, AD.fbx_curve_map(rm, decl["cases"][C.MIXED]["stack"]))
    ref = {}
    for cid in (C.STEP, C.SMOOTH):
        r, _ = C8.fbx_read(run["ed"]/cid/"control.fbx")
        ref[cid] = C.curve_arrays(r, AD.fbx_curve_map(r, decl["cases"][cid]["stack"]))
    held, smooth = set(decl["mixed"]["held_bones"]), set(decl["mixed"]["smooth_bones"])
    out = {"curves": len(am), "equal_step": 0, "equal_smooth": 0, "expected_match": 0, "mismatch": [],
           "max_abs_diff_vs_expected_reference": 0.0, "key_times_equal": True}

    def mdiff(cid, k, t, v):
        if k not in ref[cid] or not np.array_equal(t, ref[cid][k][0]):
            return np.inf
        return float(np.abs(v.astype(np.float64) - ref[cid][k][1].astype(np.float64)).max()) if v.size else 0.0
    for k, (t, v) in sorted(am.items()):
        ds, dl = mdiff(C.STEP, k, t, v), mdiff(C.SMOOTH, k, t, v)
        es, el = ds == 0.0, dl == 0.0
        out["equal_step"] += es
        out["equal_smooth"] += el
        out["key_times_equal"] &= bool(np.isfinite(ds) and np.isfinite(dl))
        want, dv = (es, ds) if k[0] in held else (el, dl) if k[0] in smooth else (es and el, max(ds, dl))
        out["expected_match"] += bool(want)
        out["max_abs_diff_vs_expected_reference"] = max(out["max_abs_diff_vs_expected_reference"], dv)
        if not want:
            out["mismatch"].append([*map(str, k), dv])
    out["mismatch"] = out["mismatch"][:20]
    out["exact"] = out["expected_match"] == out["curves"]
    return out


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
    for name in ("base", "w106", "w107", "w108", "w109", "verify", "options", "lock", "blender", "addon_dir",
                 "starting_head", "branch", "declaration", "declaration_sha", "out_dir", "produce_report",
                 "produce_dir", "encode_dir", "consume_dir", "produce_repeat_dir", "encode_repeat_dir",
                 "consume_repeat_dir"):
        ap.add_argument("--"+name.replace("_", "-"))
    o = ap.parse_args()
    return {"verify": verify, "declare": declare, "encode": encode, "analyze": analyze,
            "manifest": manifest}[o.mode](o)


if __name__ == "__main__":
    sys.exit(main())
