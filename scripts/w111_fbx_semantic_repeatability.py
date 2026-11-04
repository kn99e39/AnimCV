#!/usr/bin/env python3
"""W111 host-side stages: FBX semantic repeatability across exporter UID and source-path variation (evidence only).

  verify    W110 COMPLETE marker, W110 declaration / evidence-manifest identity and every W110 manifest file, every
            W106-W109 manifest file, frozen exporter, FBX add-on sources and Blender exe vs W110, reused-script SHAs vs
            the W110 declaration, worker lock (Batch 111 RUNNING), Git state, no other Blender process; byte copies of
            the six W110 source .blend files to the two declared scratch source paths
  declare   cases, producer matrix (seeds x paths + same-cell repeat, two runs), launches, exporter options, the
            canonicalisation rule and its excluded fields, mapping (W110 adapter, unchanged), controls, grids,
            metrics, tolerances, gates, CASE rule, resource cap -> declaration.json + SHA (outcomes_read = false)
  cells     PRODUCER: one Blender export process per cell with the declared PYTHONHASHSEED and sanitised environment
  encode    PRODUCER: W110 metadata edit per variant (census-driven 0x6104 -> 0x6002 on CONSTANT-driven selected-take
            curves, W110 validity checks) -> control.fbx / meta.fbx and the source-free consumer_input.json
  consume   CONSUMER: one Blender process (W110 launch) over every variant: U / A / A2 / K
  controls  negative / invariance controls on scratch copies of the reference cell's metadata FBX
  analyze   canonical records, raw-variation attribution, normalised plans, scene digests, motion, gates -> CASE
  manifest  hashes of every output plus a re-check that no W106-W110 original changed

No production exporter/importer, rig, Action original, scene setting or historical artifact is modified.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT/"scripts"))
import w111_fbx_semantic_repeatability_contract as C  # noqa: E402
import w111_fbx_semantic_identity as S  # noqa: E402
import w110_fbx_metadata_recovery_contract as C10  # noqa: E402
import w110_fbx_metadata_recovery as H10  # noqa: E402  (W110 host helpers: manifests, map summary; read-only reuse)
import w110_fbx_consumer_adapter as AD  # noqa: E402
import w108_fbx_subframe_contract as C8  # noqa: E402
import w109_fbx_constant_import_contract as C9  # noqa: E402
import w107_fbx_contract as C7  # noqa: E402
from w108_fbx_subframe import check_manifest, git, wjson  # noqa: E402

sha256 = C10.sha256_file
W110_WORKLOG = "docs/110_WORKLOG_FBX_METADATA_INTERPOLATION_RECOVERY_FEASIBILITY.md"
W110_DECLARATION_SHA = "223b8c81ab345507ee5e60cbceca70fdad2e03bf97236afe87d94932a3ec049d"
W110_MANIFEST_SHA = "e46329f5f2e2d11b016f06ffd93f3a3beb5ffe39579f5de9ba26e318f8215fd4"
W111_SCRIPTS = ("scripts/w111_fbx_semantic_identity.py", "scripts/w111_fbx_semantic_repeatability_contract.py",
                "scripts/w111_fbx_semantic_repeatability_blender.py", "scripts/w111_fbx_semantic_repeatability.py")
W110_SCRIPTS = H10.W110_SCRIPTS
REUSED_SCRIPTS = H10.REUSED_SCRIPTS + ("scripts/w107_fbx_ownership.py",)
EXPORTER = H10.EXPORTER
W111_NEW = ("scripts/w111_", "tests/test_w111_", "docs/111_WORKLOG_")
SHARED_PATHS = H10.SHARED_PATHS
BLENDER_SCRIPT = ROOT/"scripts/w111_fbx_semantic_repeatability_blender.py"
CONSUMER_KEYS = {"schema", "blender_version", "variants"}
CONSUMER_VARIANT_KEYS = {"cell", "case", "stack", "fbx", "times", "nb_times"}
MAP_FIELDS = ("status", "curves", "curve_status", "models", "models_separable", "flags")


def _w110_source(cid):
    return "results/produce/syn_mixed/input.blend" if cid == C.MIXED else f"results/inputs/{cid}_source.blend"


def _w110_hist_fbx(cid):
    if cid == C.MIXED:
        return ["results/encode/syn_mixed/control.fbx", "results/encode_repeat/syn_mixed/control.fbx"]
    return [f"results/inputs/{cid}_V1.fbx"]


class _NS:
    def __init__(self, **k):
        self.__dict__.update(k)


# --- verify -------------------------------------------------------------------------------------------------------
def verify(o):
    base = Path(o.base)
    out = base/"results"/"verify"
    out.mkdir(parents=True, exist_ok=False)
    rep = {"schema": C.SCHEMA + ":verify", "checks": {}, "stop": []}
    wl = (ROOT/W110_WORKLOG).read_text(encoding="utf-8")
    rep["checks"]["w110_complete_marker"] = wl.rstrip().endswith("AUTOMATION_BATCH_STATUS: COMPLETE\nBATCH_NUMBER: 110")
    w110 = Path(o.w110)
    msha, dsha = sha256(w110/"evidence_manifest.json"), sha256(w110/"results/declare/declaration.json")
    rep["checks"]["w110_manifest_sha"] = {"value": msha, "ok": msha == W110_MANIFEST_SHA}
    rep["checks"]["w110_declaration_sha"] = {"value": dsha, "ok": dsha == W110_DECLARATION_SHA}
    m10 = json.loads((w110/"evidence_manifest.json").read_text())
    chk = check_manifest(w110, m10, "manifest_stdout.txt")
    rep["w110_manifest"] = {"files": chk["files"], "bad": chk["bad"], "exception": chk["exception"],
                            "all_originals_unchanged": m10.get("all_originals_unchanged")}
    rep["checks"]["w110_files_intact"] = chk["ok"] and bool(m10.get("all_originals_unchanged"))
    bases, man = H10._manifests(o)
    rep["manifests"] = man
    for tag in man:
        rep["checks"][f"{tag}_manifest_sha"] = man[tag]["manifest_sha_ok"]
        rep["checks"][f"{tag}_files_intact"] = man[tag]["files_ok"]
    d10 = json.loads((w110/"results/declare/declaration.json").read_text())
    scr = {p: sha256(ROOT/p) for p in (*W110_SCRIPTS, *H10.REUSED_SCRIPTS, EXPORTER)}
    rep["reused_scripts_sha256"] = scr
    rep["checks"]["reused_scripts_equal_w110_declaration"] = all(scr[p] == d10["scripts_sha256"][p] for p in scr)
    rep["checks"]["exporter_sha"] = {"value": scr[EXPORTER], "ok": scr[EXPORTER] == H10.EXPORTER_SHA}
    rep["extra_reused_scripts_sha256"] = {p: sha256(ROOT/p) for p in REUSED_SCRIPTS if p not in scr}
    addon = Path(o.addon_dir)
    a_sha = {p.name: sha256(p) for p in sorted(addon.glob("*.py"))}
    rep["fbx_addon_source_sha256"] = a_sha
    rep["checks"]["fbx_addon_unchanged_vs_w110"] = a_sha == d10["environment"]["fbx_addon_source_sha256"]
    rep["blender_exe_sha256"] = sha256(o.blender)
    rep["checks"]["blender_exe_equals_w110"] = rep["blender_exe_sha256"] == d10["environment"]["blender_exe_sha256"]
    head = git("rev-parse", "HEAD")
    status = git("status", "--porcelain", "--untracked-files=all").splitlines()
    tracked = [s for s in status if not s.startswith("??")]
    untracked = [s[3:] for s in status if s.startswith("??")]
    rep["git"] = {"head": head, "branch": git("branch", "--show-current"), "origin": git("rev-parse", "origin/" + o.branch),
                  "tracked_changes": tracked, "untracked": untracked}
    rep["checks"]["starting_head"] = {"value": head, "ok": head == o.starting_head}
    rep["checks"]["branch"] = rep["git"]["branch"] == o.branch
    rep["checks"]["origin_equals_head"] = rep["git"]["origin"] == head
    rep["checks"]["working_tree_only_new_w111_files"] = not tracked and all(u.startswith(W111_NEW) for u in untracked)
    lk = json.loads(Path(o.lock).read_text()) if Path(o.lock).exists() else {}
    rep["worker_lock"] = lk
    rep["checks"]["worker_lock_batch_111_running"] = lk.get("batch") == 111 and lk.get("state") == "RUNNING" \
        and lk.get("start_commit") == o.starting_head
    tl = subprocess.run(["tasklist", "/FO", "CSV", "/NH"], capture_output=True, text=True, errors="replace").stdout
    blenders = [ln for ln in tl.splitlines() if ln.lower().startswith('"blender')]
    rep["other_blender_processes"] = blenders
    rep["checks"]["no_other_blender_process"] = not blenders
    for k, v in rep["checks"].items():
        if not (v if isinstance(v, bool) else v["ok"]):
            rep["stop"].append(k)
    copies = {}
    for cid in C.CASES:
        rel = _w110_source(cid)
        want = m10["files"].get(rel, {}).get("sha256")
        rec = {"origin": f"w110:{rel}", "w110_manifest_sha256": want}
        for pk in C.PATHS:
            dst = Path(C.source_path(base, cid, pk))
            dst.parent.mkdir(parents=True, exist_ok=False)
            shutil.copyfile(w110/rel, dst)
            rec[pk] = {"path": str(dst), "sha256": sha256(dst)}
        rec["ok"] = all(rec[pk]["sha256"] == want for pk in C.PATHS) and want is not None
        if not rec["ok"]:
            rep["stop"].append(f"copy:{cid}")
        copies[cid] = rec
    rep["sources"] = copies
    rep["ok"] = not rep["stop"]
    wjson(out/"verify_report.json", rep)
    print(json.dumps({"ok": rep["ok"], "stop": rep["stop"]}))
    return 0 if rep["ok"] else 2


# --- declare ------------------------------------------------------------------------------------------------------
def declare(o):
    base, w110 = Path(o.base), Path(o.w110)
    out = base/"results"/"declare"
    out.mkdir(parents=True, exist_ok=False)
    ver = json.loads(Path(o.verify).read_text())
    if not ver["ok"]:
        raise SystemExit("STOP: verify failed")
    opts = json.loads(Path(o.options).read_text())
    if opts["blender_version"] != H10.EXPECTED_ENV["blender_version"] or \
            opts["fbx_addon"]["version"] != H10.EXPECTED_ENV["fbx_addon_version"] or \
            opts["fbx_addon"].get("source_sha256") != ver["fbx_addon_source_sha256"]:
        raise SystemExit("STOP: unexpected environment")
    d10 = json.loads((w110/"results/declare/declaration.json").read_text())
    m10 = json.loads((w110/"evidence_manifest.json").read_text())["files"]
    e10 = json.loads((w110/"results/encode/encode_report.json").read_text())
    p10 = json.loads((w110/"results/produce/produce_report.json").read_text())
    cases = {}
    for cid in C.CASES:
        c10 = d10["cases"][cid]
        ms = e10["cases"][cid]["map_summary"]
        cases[cid] = {
            "id": cid, "role": c10["role"], "regime": c10["regime"], "action": c10["action"], "stack": c10["stack"],
            "full_matrix_minimum": cid in C.PRIMARY_CASES,
            "times": c10["times"], "nb_times": c10["nb_times"], "interval_classes": c10["interval_classes"],
            "timing": c10["timing"], "source_key_frames": c10["source_key_frames"],
            "sources": {pk: ver["sources"][cid][pk] for pk in C.PATHS},
            "source_origin": ver["sources"][cid]["origin"],
            "w110_source_npz": {k: {"path": str(w110/f"results/produce/{cid}/{k}.npz"),
                                    "sha256": m10[f"results/produce/{cid}/{k}.npz"]["sha256"]}
                                for k in ("source", "source_nb")},
            "expected_source_action_digest": p10["cases"][cid]["source_action_digest"],
            "expected_census": {k: p10["cases"][cid]["census"][k] for k in ("armature_object", "groups",
                                                                             "unsupported_paths", "fcurves")},
            "expected_map": {k: ms[k] for k in MAP_FIELDS},
            "expected_producer_counts": e10["cases"][cid]["producer_counts"],
            "historical_fbx": [{"path": str(w110/r), "sha256": m10[r]["sha256"], "origin": f"w110:{r}"}
                               for r in _w110_hist_fbx(cid)]}
    v0 = d10["exporter"]["v0_kwargs"]
    decl = {
        "schema": C.SCHEMA + ":declaration", "outcomes_read": False,
        "note": "Frozen before any W111 export, encode, import, adapter run or control.  Before freezing, only single "
                "historical W110 files were parsed (structure inventory, canonicaliser / control development); no two "
                "files were compared under the W111 canonicaliser and no fresh export existed.",
        "starting_head": o.starting_head, "verify_sha256": sha256(o.verify), "options_sha256": sha256(o.options),
        "w110": {"declaration_sha256": W110_DECLARATION_SHA, "evidence_manifest_sha256": W110_MANIFEST_SHA,
                 "case": "D (final; unchanged by W111)"},
        "environment": {"blender_version": opts["blender_version"], "blender_build_hash": opts["blender_build_hash"],
                        "blender_exe": o.blender, "blender_exe_sha256": ver["blender_exe_sha256"],
                        "fbx_addon_version": opts["fbx_addon"]["version"],
                        "fbx_addon_source_sha256": ver["fbx_addon_source_sha256"],
                        "producer_launch": C.PRODUCER_LAUNCH, "consumer_launch": C.CONSUMER_LAUNCH,
                        "producer_env": "all PYTHON* variables removed; PYTHONHASHSEED=<cell seed>; PYTHONNOUSERSITE=1 "
                                        "(--python-use-system-env is needed for Blender to honour PYTHONHASHSEED; "
                                        "pre-declaration plumbing probe: without it two PYTHONHASHSEED=1 launches "
                                        "printed different hash('w111-probe'))",
                        "consumer_env": "all PYTHON* variables removed; W110 launch (per-process random hash seed, as "
                                        "in W110)"},
        "scripts_sha256": {p: sha256(ROOT/p) for p in (*W111_SCRIPTS, *W110_SCRIPTS, *REUSED_SCRIPTS, EXPORTER)},
        "cases": cases,
        "matrix": {"seeds": C.SEEDS, "paths": C.PATHS, "cells": [list(c) for c in C.CELLS], "reference_cell": C.REF_CELL,
                   "runs": list(C.RUNS), "pairs": {k: [list(p) for p in v] for k, v in C.PAIRS.items()},
                   "conditions": list(C.CONDITIONS),
                   "fixed": "Blender exe / add-on, source .blend bytes (identical copies at both paths), Action and "
                            "slot, scene timing, exporter V1 options, selected take, W110 producer rule and flag, "
                            "W110 adapter, sample grids; the consumer environment"},
        "exporter": {"v0_kwargs": v0, "v1": d10["exporter"]["v1"], "v1_kwargs": C7.variant_kwargs(v0, "V1"),
                     "call": "W110 export_v1 (open_case + W107 do_export('V1')) on the source at the cell's path"},
        "producer": d10["producer"],
        "consumer": {"adapter": "scripts/w110_fbx_consumer_adapter.py", "adapter_sha256": sha256(ROOT/"scripts/"
                                                                                            "w110_fbx_consumer_adapter.py"),
                     "inputs": "metadata FBX bytes, selected take name, freshly imported scene (unchanged W110 rule)",
                     "consumer_input": "host whitelist: variant cell / case id, stack, metadata-FBX path / SHA under the "
                                       "encode dir, sample grids; nothing else"},
        "canonicalisation": {"rule": S.__doc__.strip(), "excluded_fields": S.EXCLUDED_FIELDS,
                             "named_classes": S.NAMED, "anonymous_classes": S.ANONYMOUS,
                             "comparison": "full in-memory equality of every retained value (bytes) of every header "
                                           "section, every Objects element keyed by identity and the Connections "
                                           "multiset; stored records carry per-identity digests",
                             "raw_variation": S.raw_variation.__doc__.strip(),
                             "plan_normalisation": S.normalize_plan.__doc__.strip()},
        "controls": {"cases": list(C.CONTROL_CASES), "cell": C.REF_CELL, "file": "meta.fbx",
                     "writer": "w111 write_fbx (uncompressed arrays; scratch copies only; never imported)",
                     "list": {k: list(v) for k, v in S.CONTROLS.items()},
                     "judge": S.judge_control.__doc__.strip()},
        "metrics": {
            "semantic": "canonical compare(reference, variant) for control.fbx and meta.fbx of every variant; reference "
                        "= run1 / s1_pA",
            "attribution": "raw_variation over the declared pairs (seed_only, path_only, same_cell, seed_and_path) per "
                           "case and run; Python hash probes per cell",
            "plans": "UID-normalised W110 adapter plans (A, A2) vs the reference; A2 vs A per variant",
            "scene": "consume scene digests per condition vs the reference; final key-data digests",
            "motion": "W110 overall / component metrics vs the variant's own source samples; per condition max "
                      "difference of every sampled array vs the reference variant",
            "historical": "diagnostic only, not a gate: reference fresh control vs the W107/W108 V1 FBX (W110 copies) "
                          "and vs both W110 mixed control FBX; W110 mixed run vs repeat under the W111 canonicaliser "
                          "(does not alter W110 CASE D)",
            "cost": "FBX bytes, selected-take keys, export / encode / import / adapter / canonicalise seconds, storage"},
        "tolerances": C.TOL,
        "gates": {
            "semantic_identical": "every variant's control and meta canonical records equal the reference (0 "
                                  "differences)",
            "plans_identical": "normalised A and A2 plans equal the reference; 0 unmapped UIDs",
            "scene_identical": "scene digests (selected key data with/without interpolation, other Actions, Action "
                               "inventory, rig, timing) and final key-data digests equal the reference per condition",
            "motion_stable": "every condition's arrays within deterministic_m of the reference variant",
            "motion_material": "some condition's arrays differ from the reference by more than recovered_m",
            "motion_gate": "W110 gates per variant: held cases meta A all-samples and integer <= 1 mm; syn_smooth A "
                           "<= 1 mm; mixed A held / smooth components, all samples and integer <= 1 mm (A2 too)",
            "coverage": "W110 coverage per variant (map OK, all curves mapped, all imported F-curves mapped, 0 claimed "
                        "twice, 0 unresolved items, CONSTANT identity set == producer CONSTANT identity set)",
            "controls_discriminate": "per variant: U harms every held case and the mixed held component > 1 cm; K harms "
                                     "syn_smooth and the mixed smooth component > 1 cm",
            "variation_realized": "per case and run: every seed_only pair has uid_changed > 0, every path_only pair "
                                  "has source_path_differs; s1 / s2 hash probes differ, same-seed probes equal",
            "attribution_clean": "same_cell pairs: 0 UID changes, equal object / connection order, same path and "
                                 "Document UID; path_only pairs: 0 UID changes and equal order; seed_only pairs: same "
                                 "path",
            "repeat_ok": "both runs complete and the per-run aggregate decision inputs are equal",
            "negative_controls_ok": "every control in both runs judged ok (EQUAL / DETECTED_LOCAL / declared REJECTED)",
            "canonical_injective": "no genuine variant rejected"},
        "case_rule": C.decide_case.__doc__.strip(),
        "stop_conditions": ["declaration / consumer-input hash mismatch", "Blender version differs", "source hash "
                            "mismatch", "a Blender stage exits non-zero (recorded; not retried with other settings)",
                            "shared / production file changed", "resource cap exceeded"],
        "resource_cap": {"blender_runs": "options + 2 x (5 export cells + 1 consume)", "wall_clock_per_run_s": 1200,
                         "disk_gb": 5, "gpu": "none", "network": "none"},
        "hypothesis": {"H111": "the W110 FBX interpolation metadata plus the unchanged source-free W110 consumer mapping "
                               "is invariant to exporter hash-seed UIDs, UID-induced order and the embedded source path "
                               "while the canonicaliser still detects key, flag, take and hierarchy changes"},
    }
    path = out/"declaration.json"
    wjson(path, decl)
    s = sha256(path)
    (out/"declaration.sha256").write_text(s + "\n")
    print(s)
    return 0


def _decl(o):
    b = Path(o.declaration).read_bytes()
    if hashlib.sha256(b).hexdigest() != o.declaration_sha:
        raise SystemExit("STOP: declaration hash mismatch")
    return json.loads(b)


# --- cells (producer processes) ------------------------------------------------------------------------------------
def cells(o):
    decl = _decl(o)
    run = Path(o.run_dir)
    (run/"cells").mkdir(parents=True, exist_ok=False)
    runs = []
    for cell, sk, pk in C.CELLS:
        env = C.producer_env(os.environ, C.SEEDS[sk])
        cmd = [o.blender, *C.PRODUCER_LAUNCH, "-P", str(BLENDER_SCRIPT), "--", "export", "--declaration",
               str(Path(o.declaration).resolve()), "--declaration-sha", o.declaration_sha, "--cell", cell,
               "--out-dir", str(run/"cells"/cell)]
        t0 = time.perf_counter()
        p = subprocess.run(cmd, env=env, capture_output=True, text=True, errors="replace")
        dt = time.perf_counter() - t0
        (run/"cells"/f"{cell}.log").write_text(p.stdout + "\n--- stderr ---\n" + p.stderr, encoding="utf-8")
        runs.append({"cell": cell, "seed": C.SEEDS[sk], "path_key": pk, "returncode": p.returncode, "seconds": dt,
                     "env_pythonhashseed": env["PYTHONHASHSEED"]})
        print(json.dumps(runs[-1]))
        if p.returncode != 0:
            break
    wjson(run/"cells"/"cells_runs.json", {"declaration_sha256": o.declaration_sha, "runs": runs})
    return 0 if all(r["returncode"] == 0 for r in runs) and len(runs) == len(C.CELLS) else 2


# --- encode (producer) ---------------------------------------------------------------------------------------------
def consumer_clean(ci, root=None):
    """Structural whitelist: only declared keys; every FBX is a meta.fbx under ``root``; no .blend / census /
    declaration / source / sample / source-path reference anywhere."""
    if set(ci) != CONSUMER_KEYS:
        return False
    blob = json.dumps(ci).lower()
    if any(w in blob for w in (".blend", "census", "declaration", "source", ".npz", "srcpaths")):
        return False
    for v in ci["variants"].values():
        if set(v) != CONSUMER_VARIANT_KEYS or set(v["fbx"]) != {"path", "sha256"}:
            return False
        p = Path(v["fbx"]["path"])
        if p.name != "meta.fbx" or (root is not None and Path(root).resolve() not in p.resolve().parents):
            return False
    return True


def encode_variant(d, v1, v1_sha, case, census):
    """W110 producer edit and validity checks on one fresh V1 export (same rule, flag and checks as W110 encode)."""
    t0 = time.perf_counter()
    data = Path(v1).read_bytes()
    if hashlib.sha256(data).hexdigest() != v1_sha:
        raise SystemExit(f"STOP: {v1} hash mismatch")
    (d/"control.fbx").write_bytes(data)
    root0, version = C8.fbx_read(d/"control.fbx")
    fmap = AD.fbx_curve_map(root0, case["stack"])
    ms = H10._map_summary(fmap)
    rec = {"control_sha256": sha256(d/"control.fbx"), "control_bytes": len(data), "fbx_version": version,
           "map_summary": ms, "map_equals_declared": {k: ms[k] for k in MAP_FIELDS} == case["expected_map"]}
    dirs = C10.producer_directives(fmap, census)
    counts = C10.producer_counts(dirs)
    rec["producer_counts"] = counts
    rec["producer_counts_equal_w110"] = counts == case["expected_producer_counts"]
    rec["producer_directives"] = {str(u): v for u, v in sorted(dirs.items())}
    const = sorted(u for u, (dv, _) in dirs.items() if dv == "CONSTANT")
    rec["constant_uids"] = const
    noop, _ = C9.patch_flags(data, [r["uid"] for r in fmap["curves"]], C10.SOURCE_FLAG)
    rec["noop_identity"] = noop == data
    problems = []
    bad = {k: v for k, v in counts.items() if k in ("UNMATCHED", "UNSUPPORTED")}
    if bad or fmap["status"] != "OK" or not rec["map_equals_declared"] or not rec["noop_identity"] \
            or not rec["producer_counts_equal_w110"]:
        problems.append(f"producer {bad} / map {fmap['status']} / declared {rec['map_equals_declared']} / noop "
                        f"{rec['noop_identity']} / counts {counts}")
        return rec, problems
    new, recs = (C9.patch_flags(data, const, C10.CONSTANT_FLAG) if const else (data, []))
    p = d/"meta.fbx"
    p.write_bytes(new)
    diff = C9.byte_diff_spans(data, new, [(r["offset"], r["bytes"]) for r in recs])
    root1, _ = C8.fbx_read(p)
    fmap1 = AD.fbx_curve_map(root1, case["stack"])
    fl1 = {r["uid"]: r["flags_hex"] for r in fmap1["curves"]}
    all1, all0 = C9.curve_flags(root1), C9.curve_flags(root0)
    fps = case["timing"]["fps"]/case["timing"]["fps_base"]
    st, st0 = C8.fbx_anim_summary(root1, fps)["stacks"], C8.fbx_anim_summary(root0, fps)["stacks"]
    vr = {"sha256": sha256(p), "bytes": p.stat().st_size, "patched_curves": len(recs), "byte_diff": diff,
          "map_digest_without_flags": AD.fbx_map_digest(fmap1, with_flags=False),
          "selected_stack_summary": st.get(case["stack"])}
    vr["checks"] = {
        "same_length": diff["same_length"], "diff_only_in_target_spans": diff["all_in_targets"],
        "every_target_changed": (diff["diff_bytes"] or 0) > 0 if const else diff["diff_bytes"] == 0,
        "masked_tree_equals_control": C10.tree_digest(root1, const) == C10.tree_digest(root0, const),
        "map_unchanged_except_flags": vr["map_digest_without_flags"] == ms["map_digest_without_flags"],
        "constant_targets_decoded": all(fl1[u] == [f"0x{C10.CONSTANT_FLAG:x}"] for u in const),
        "every_other_curve_unchanged": all(all1[u] == all0[u] for u in all0 if u not in set(const)),
        "selected_stack_keys_unchanged": (st.get(case["stack"], {}).get("keys") == st0.get(case["stack"], {}).get("keys")
                                          and st.get(case["stack"], {}).get("refcount_covers_keys") is True),
        "other_stacks_unchanged": {k: v for k, v in st.items() if k != case["stack"]}
        == {k: v for k, v in st0.items() if k != case["stack"]}}
    vr["valid"] = all(vr["checks"].values())
    if not vr["valid"]:
        problems.append(f"meta {[k for k, x in vr['checks'].items() if not x]}")
    rec["meta"] = vr
    rec["selected_stack_keys"] = (st0.get(case["stack"]) or {}).get("keys")
    rec["seconds"] = time.perf_counter() - t0
    return rec, problems


def encode(o):
    decl = _decl(o)
    run = Path(o.run_dir)
    out = run/"encode"
    out.mkdir(parents=True, exist_ok=False)
    rep = {"schema": C.SCHEMA + ":encode", "declaration_sha256": o.declaration_sha, "cells": {}, "problems": []}
    ci = {"schema": C.SCHEMA + ":consumer_input", "blender_version": decl["environment"]["blender_version"],
          "variants": {}}
    for cell, _, _ in C.CELLS:
        cp = run/"cells"/cell/"cell_report.json"
        cr = json.loads(cp.read_text())
        if cr["declaration_sha256"] != o.declaration_sha:
            raise SystemExit("STOP: cell report bound to another declaration")
        rep["cells"][cell] = {"cell_report_sha256": sha256(cp), "cases": {}}
        for cid, case in decl["cases"].items():
            pr = cr["cases"][cid]
            d = out/cell/cid
            d.mkdir(parents=True)
            rec, probs = encode_variant(d, pr["v1_export"]["fbx"], pr["v1_export"]["fbx_sha256"], case, pr["census"])
            rep["cells"][cell]["cases"][cid] = rec
            rep["problems"] += [f"{cell}:{cid}:{p}" for p in probs]
            if not probs:
                ci["variants"][f"{cell}:{cid}"] = {"cell": cell, "case": cid, "stack": case["stack"],
                                                   "times": case["times"], "nb_times": case["nb_times"],
                                                   "fbx": {"path": str(d/"meta.fbx"), "sha256": rec["meta"]["sha256"]}}
    rep["consumer_input_clean"] = consumer_clean(ci, out)
    if not rep["consumer_input_clean"]:
        rep["problems"].append("consumer input not source-free")
    rep["valid"] = not rep["problems"]
    rep["consumer_input_sha256"] = wjson(out/"consumer_input.json", ci)
    s = wjson(out/"encode_report.json", rep)
    print(json.dumps({"valid": rep["valid"], "problems": rep["problems"][:10], "sha256": s,
                      "consumer_input_sha256": rep["consumer_input_sha256"], "variants": len(ci["variants"])}))
    return 0 if rep["valid"] else 2


# --- consume (consumer process) ------------------------------------------------------------------------------------
def consume(o):
    _decl(o)
    run = Path(o.run_dir)
    ci = run/"encode"/"consumer_input.json"
    cs = sha256(ci)
    cmd = [o.blender, *C.CONSUMER_LAUNCH, "-P", str(BLENDER_SCRIPT), "--", "consume", "--consumer-input", str(ci),
           "--consumer-input-sha", cs, "--out-dir", str(run/"consume")]
    t0 = time.perf_counter()
    p = subprocess.run(cmd, env=C.consumer_env(os.environ), capture_output=True, text=True, errors="replace")
    dt = time.perf_counter() - t0
    (run/"consume.log").write_text(p.stdout + "\n--- stderr ---\n" + p.stderr, encoding="utf-8")
    wjson(run/"consume_run.json", {"returncode": p.returncode, "seconds": dt, "consumer_input_sha256": cs})
    print(json.dumps({"returncode": p.returncode, "seconds": dt}))
    return p.returncode


# --- controls -------------------------------------------------------------------------------------------------------
def controls(o):
    decl = _decl(o)
    run = Path(o.run_dir)
    out = run/"controls"
    out.mkdir(parents=True, exist_ok=False)
    rep = {"schema": C.SCHEMA + ":controls", "declaration_sha256": o.declaration_sha, "cases": {}}
    for cid in C.CONTROL_CASES:
        f = run/"encode"/C.REF_CELL/cid/"meta.fbx"
        if not f.exists():
            rep["cases"][cid] = {"file": str(f), "base_rejection": {"code": "MISSING_FILE"}, "controls": {}}
            continue
        root, version = C8.fbx_read(f)
        base, rej = S.try_canonicalize(root)
        crec = {"file": str(f), "sha256": sha256(f), "base_rejection": rej, "controls": {}}
        if base is None:
            rep["cases"][cid] = crec
            continue
        (out/cid).mkdir()
        for name in S.CONTROLS:
            try:
                mt, rec = S.mutate(root, base, decl["cases"][cid]["stack"], name)
                p = out/cid/f"{name}.fbx"
                p.write_bytes(S.write_fbx(mt, version))
                mcan, mrej = S.try_canonicalize(C8.fbx_read(p)[0])
                j = S.judge_control(rec, base, mcan, mrej)
                crec["controls"][name] = {**{k: v for k, v in rec.items() if k not in ("allowed_ids",)},
                                          "allowed_ids_count": len(rec["allowed_ids"]), "sha256": sha256(p),
                                          "bytes": p.stat().st_size, **j}
            except Exception as exc:  # recorded as a failed control, never retried
                crec["controls"][name] = {"ok": False, "outcome": "ERROR", "error": repr(exc)}
        rep["cases"][cid] = crec
    rep["ok"] = all(r.get("base_rejection") is None and len(r["controls"]) == len(S.CONTROLS)
                    and all(c["ok"] for c in r["controls"].values()) for r in rep["cases"].values())
    s = wjson(out/"controls_report.json", rep)
    print(json.dumps({"ok": rep["ok"], "sha256": s, "outcomes": {c: {k: v["outcome"] for k, v in r["controls"].items()}
                                                                 for c, r in rep["cases"].items()}}, indent=1))
    return 0


# --- analyze --------------------------------------------------------------------------------------------------------
_ld = C10.load_npz
LIGHT = ("uid_of", "order", "conn_order", "document_uid", "timestamp", "source_path")


def _load_run(base, run):
    d = Path(base)/"results"/run
    r = {"dir": d, "cells": {}, "cells_sha": {}}
    for cell, _, _ in C.CELLS:
        p = d/"cells"/cell/"cell_report.json"
        if p.exists():
            r["cells"][cell] = json.loads(p.read_text())
            r["cells_sha"][cell] = sha256(p)
    for k, rel in (("cells_runs", "cells/cells_runs.json"), ("enc", "encode/encode_report.json"),
                   ("cons", "consume/consume_report.json"), ("ctl", "controls/controls_report.json"),
                   ("consume_run", "consume_run.json")):
        p = d/rel
        r[k] = json.loads(p.read_text()) if p.exists() else None
    ci = d/"encode"/"consumer_input.json"
    r["ci_sha"] = sha256(ci) if ci.exists() else None
    return r


def _cov(adapter_rec, nplan, const_ids):
    acc = adapter_rec["plan"].get("accounting", {})
    got = {k for k, v in nplan["curve_status"].items() if v == "CONSTANT"}
    ok = bool(adapter_rec["plan"]["status"] == "OK" and acc.get("fbx_curves_unresolved") == 0
              and acc.get("fbx_curves_mapped") == acc.get("fbx_curves") and acc.get("fbx_curves", 0) > 0
              and not acc.get("imported_fcurves_unmapped") and acc.get("fcurves_claimed_twice") == 0
              and acc.get("items_unresolved") == 0 and got == const_ids and adapter_rec.get("applied") is not None
              and nplan["_unmapped_uids"] == 0)
    return ok, acc


def _isolation(r):
    ad = r["adapter"]
    return bool(r["access"]["isolated"] and r["access"]["fbx_reads"] >= 1 and ad["fbx_unchanged_during_read"]
                and not ad["escape_mismatch"] and r["selected_equals_adapter_action"] and ad["fbx_sha256"] == r["fbx_sha256"])


def analyze_case(cid, case, runs, decl, out):
    """Every variant of one case against the reference (run1 / s1_pA)."""
    T = C.TOL
    times = np.array(case["times"])
    classes = {int(k): v for k, v in case["interval_classes"].items()}
    w110_src = _ld(case["w110_source_npz"]["source"]["path"])
    w110_nb = _ld(case["w110_source_npz"]["source_nb"]["path"])
    mixed = cid == C.MIXED
    if cid not in C.HELD_CASES + (C.SMOOTH, C.MIXED):
        raise ValueError(f"undeclared case role: {cid}")
    spec = C10.MIXED_SPEC
    ref = None
    light, variants = {}, {}
    rec_dir = out/"records"
    for run in C.RUNS:
        R = runs[run]
        for cell, _, _ in C.CELLS:
            vid = f"{run}/{cell}"
            v = {"run": run, "cell": cell}
            variants[vid] = v
            cr = R["cells"].get(cell, {}).get("cases", {}).get(cid)
            er = (R["enc"] or {}).get("cells", {}).get(cell, {}).get("cases", {}).get(cid)
            co = (R["cons"] or {}).get("variants", {}).get(f"{cell}:{cid}")
            v["complete"] = bool(cr and er and er.get("meta") and co
                                 and all(co.get(k, {}).get("status") == "SAMPLED" for k in C.CONDITIONS))
            if not (cr and er and er.get("meta")):
                continue
            ed = R["dir"]/"encode"/cell/cid
            cans = {}
            for tag in ("control", "meta"):
                t0 = time.perf_counter()
                root, _ = C8.fbx_read(ed/f"{tag}.fbx")
                can, rej = S.try_canonicalize(root)
                v[f"{tag}_canonicalise_seconds"] = time.perf_counter() - t0
                v[f"{tag}_rejection"] = rej
                if can is None:
                    continue
                v[f"{tag}_accounting"] = can["accounting"]
                v[f"{tag}_canonical_digest"] = S.canonical_digest(can)
                cans[tag] = can
            del root
            if ref is None and vid == "run1/" + C.REF_CELL and len(cans) == 2:
                ref = {"can": cans, "vid": vid}
                rec_dir.mkdir(exist_ok=True)
                for tag in ("control", "meta"):
                    wjson(rec_dir/f"{cid}_reference_{tag}.json", S.record_summary(cans[tag]))
            if ref is None:
                v["no_reference"] = True
                continue
            for tag in cans:
                cmp = S.compare(ref["can"][tag], cans[tag])
                v[f"{tag}_vs_reference"] = S.public(cmp)
                if not cmp["equal"]:
                    rec_dir.mkdir(exist_ok=True)
                    wjson(rec_dir/f"{cid}_{run}_{cell}_{tag}.json", S.record_summary(cans[tag]))
            if "control" in cans:
                light[vid] = {k: cans["control"][k] for k in LIGHT}
                v["control_sha256"] = er["control_sha256"]
                ndir = S.normalize_directives(er["producer_directives"], cans["control"]["uid_of"])
                v["directives_digest"] = S.plan_digest(ndir)
                const_ids = {k for k, x in ndir.items() if x[0] == "CONSTANT"}
                v["_const_ids"] = const_ids
                v["constant_curves"] = len(const_ids)
            # source identity / environment
            src, src_nb = _ld(R["dir"]/"cells"/cell/cid/"source.npz"), _ld(R["dir"]/"cells"/cell/cid/"source_nb.npz")
            v["source_vs_w110_max_diff"] = max(C8.arrays_max_diff(src, w110_src), C8.arrays_max_diff(src_nb, w110_nb))
            v["source_ok"] = bool(v["source_vs_w110_max_diff"] <= T["source_fidelity_m"]
                                  and cr["source_action_digest"] == case["expected_source_action_digest"]
                                  and {k: cr["census"][k] for k in case["expected_census"]} == case["expected_census"]
                                  and cr["source_unchanged"] and cr["v1_export"]["action_unmodified"]
                                  and cr["v1_export"]["work_file_unchanged"] and cr["work_file_unchanged"]
                                  and np.array_equal(src["times"], times))
            ex = cr["v1_export"]
            pk = C.cell_of(cell)["path_key"]
            v["environment_ok"] = bool(ex["export"].get("kwargs") == decl["exporter"]["v1_kwargs"]
                                       and ex["exported_active"]["action"] == case["action"]
                                       and Path(cr["blend_filepath_at_export"]) == Path(case["sources"][pk]["path"])
                                       and cr["assigned"]["slot"] is not None)
            v["encode_valid"] = bool(er.get("meta", {}).get("valid") and er.get("producer_counts_equal_w110")
                                     and er.get("map_equals_declared") and er.get("noop_identity"))
            v["fbx_bytes"], v["meta_bytes"] = er["control_bytes"], er["meta"]["bytes"]
            v["selected_stack_keys"] = er.get("selected_stack_keys")
            v["export_seconds"] = cr["seconds"]["export"]
            v["encode_seconds"] = er.get("seconds")
            if not co or not v["complete"]:
                continue
            # consumer: plans, isolation, integrity, scene, motion
            meta_can = cans.get("meta")
            conds = {}
            for k in C.CONDITIONS:
                r = co[k]
                imp, imp_nb = (_ld(R["dir"]/"consume"/cell/cid/f"meta_{k}{s}.npz") for s in ("", "_nb"))
                e = C10.overall(src, imp, src_nb, imp_nb, times, classes)
                c = {"all_samples_max_m": e["all_samples_max_m"], "integer_max_m": e["integer"]["max_abs_coord_m"],
                     "subframe_max_m": e["subframe"]["max_abs_coord_m"],
                     "near_boundary_max_m": e["near_boundary"]["max_abs_coord_m"], "scene": r["scene"],
                     "final_keydata_digest_all": r["final_keydata_digest_all"],
                     "final_keydata_digest_excl_interpolation": r["final_keydata_digest_excl_interpolation"],
                     "import_keydata_digest_all": r["import_keydata_digest_all"],
                     "final_interp_tally": r["final_interp_tally"], "import_seconds": r.get("import_seconds")}
                if mixed:
                    cm = C10.component_errors(src, imp, spec)
                    cn = C10.component_errors(src_nb, imp_nb, spec)
                    c["components_max"] = {q: max(cm[q], cn[q]) for q in cm}
                if k in ("A", "A2"):
                    np_ = S.normalize_plan(r["adapter"]["plan"], meta_can["uid_of"]) if meta_can else None
                    c["plan_digest"] = S.plan_digest(np_) if np_ else None
                    c["_plan"] = np_
                    c["coverage_ok"], c["accounting"] = _cov(r["adapter"], np_, v.get("_const_ids", set())) \
                        if np_ else (False, {})
                    c["isolation_ok"] = _isolation(r)
                    c["access_events"] = r["access"]["events"]
                    c["integrity_ok"] = bool(r["integrity"]["ok"])
                    c["applied"] = {q: r["adapter"]["applied"][q] for q in ("fcurves_changed", "fcurves_already_correct",
                                                                           "keys_changed")}
                    c["adapter_seconds"] = r.get("adapter_seconds")
                c["_imp"] = (imp, imp_nb)
                conds[k] = c
            v["conditions"] = conds
            v["isolation_ok"] = all(conds[k]["isolation_ok"] for k in ("A", "A2"))
            v["integrity_ok"] = bool(all(conds[k]["integrity_ok"] for k in ("A", "A2"))
                                     and all(conds[k]["final_keydata_digest_excl_interpolation"]
                                             == conds["U"]["final_keydata_digest_excl_interpolation"]
                                             for k in ("A", "A2", "K"))
                                     and all(conds[k]["scene"]["other_actions"] == conds["U"]["scene"]["other_actions"]
                                             and conds[k]["scene"]["rig"] == conds["U"]["scene"]["rig"]
                                             and conds[k]["scene"]["timing"] == conds["U"]["scene"]["timing"]
                                             for k in ("A", "A2", "K")))
            v["coverage_ok"] = all(conds[k]["coverage_ok"] for k in ("A", "A2"))
            v["adapter_repeat_equal"] = conds["A"]["plan_digest"] == conds["A2"]["plan_digest"] and \
                conds["A"]["_plan"] == conds["A2"]["_plan"]
            A_ = conds["A"]
            ok = []
            for k in ("A", "A2"):
                a = conds[k]
                if cid in C.HELD_CASES:
                    ok.append(a["all_samples_max_m"] <= T["recovered_m"] and a["integer_max_m"] <= T["integer_gate_m"])
                elif cid == C.SMOOTH:
                    ok.append(a["all_samples_max_m"] <= T["recovered_m"])
                elif cid == C.MIXED:
                    ok.append(a["components_max"]["held"] <= T["recovered_m"] and a["components_max"]["smooth"]
                              <= T["recovered_m"] and a["all_samples_max_m"] <= T["recovered_m"]
                              and a["integer_max_m"] <= T["integer_gate_m"])
            v["motion_gate"] = all(ok)
            if cid in C.HELD_CASES:
                v["controls_discriminate"] = conds["U"]["all_samples_max_m"] > T["harm_m"]
            elif cid == C.SMOOTH:
                v["controls_discriminate"] = conds["K"]["all_samples_max_m"] > T["harm_m"]
            elif cid == C.MIXED:
                v["controls_discriminate"] = (conds["U"]["components_max"]["held"] > T["harm_m"]
                                              and conds["K"]["components_max"]["smooth"] > T["harm_m"])
            del A_
    # cross-variant comparisons against the reference variant
    rv = variants.get("run1/" + C.REF_CELL, {})
    for vid, v in variants.items():
        if not v.get("complete") or "conditions" not in v or "conditions" not in rv:
            continue
        v["semantic_equal"] = bool(v.get("control_vs_reference", {}).get("equal")
                                   and v.get("meta_vs_reference", {}).get("equal"))
        v["directives_equal"] = v.get("directives_digest") == rv.get("directives_digest")
        v["plans_equal"] = all(v["conditions"][k]["_plan"] == rv["conditions"]["A"]["_plan"] for k in ("A", "A2"))
        sc = {}
        md = {}
        for k in C.CONDITIONS:
            a, b = v["conditions"][k], rv["conditions"][k]
            sc[k] = bool(a["scene"] == b["scene"] and all(a[f] == b[f] for f in (
                "final_keydata_digest_all", "final_keydata_digest_excl_interpolation", "import_keydata_digest_all")))
            md[k] = max(C8.arrays_max_diff(a["_imp"][0], b["_imp"][0]), C8.arrays_max_diff(a["_imp"][1], b["_imp"][1]))
        v["scene_equal"] = sc
        v["motion_max_diff_vs_reference_m"] = md
    for v in variants.values():
        for c in v.get("conditions", {}).values():
            c.pop("_imp", None)
            c.pop("_plan", None)
        v.pop("_const_ids", None)
    # raw variation over the declared pairs
    pairs = {}
    for run in C.RUNS:
        for kind, plist in C.PAIRS.items():
            for a, b in plist:
                va, vb = f"{run}/{a}", f"{run}/{b}"
                if va in light and vb in light:
                    pairs[f"{run}:{kind}:{a}~{b}"] = {"run": run, "kind": kind, **S.raw_variation(
                        light[va], light[vb], variants[va].get("control_sha256"), variants[vb].get("control_sha256"))}
    for cell, _, _ in C.CELLS:
        va, vb = f"run1/{cell}", f"run2/{cell}"
        if va in light and vb in light:
            pairs[f"cross_run:{cell}"] = {"run": "both", "kind": "cross_run_same_cell", **S.raw_variation(
                light[va], light[vb], variants[va].get("control_sha256"), variants[vb].get("control_sha256"))}
    # historical diagnostics (not gates)
    hist = []
    if ref is not None:
        for h in case["historical_fbx"]:
            root, _ = C8.fbx_read(h["path"])
            can, rej = S.try_canonicalize(root)
            rec = {"origin": h["origin"], "sha256_ok": sha256(h["path"]) == h["sha256"], "rejection": rej}
            if can is not None:
                rec["vs_reference_control"] = S.public(S.compare(ref["can"]["control"], can))
                rec["raw_vs_reference_control"] = S.raw_variation({k: ref["can"]["control"][k] for k in LIGHT},
                                                                  {k: can[k] for k in LIGHT})
                rec["_can"] = can
            hist.append(rec)
        if cid == C.MIXED and len(hist) == 2 and all("_can" in h for h in hist):
            hist.append({"origin": "w110 mixed run vs repeat (W111 canonicaliser; diagnostic only, W110 CASE D "
                                   "unchanged)", "comparison": S.public(S.compare(hist[0]["_can"], hist[1]["_can"])),
                         "raw": S.raw_variation({k: hist[0]["_can"][k] for k in LIGHT},
                                                {k: hist[1]["_can"][k] for k in LIGHT})})
        for h in hist:
            h.pop("_can", None)
    return {"variants": variants, "pairs": pairs, "historical": hist, "reference": ref["vid"] if ref else None}


def _per_run(res, run, cid):
    vs = [v for k, v in res["variants"].items() if v["run"] == run]
    allc = lambda f: bool(vs) and all(v.get(f) is True for v in vs)  # noqa: E731
    return {
        "complete": allc("complete"),
        "semantic_identical": allc("semantic_equal"),
        "plans_identical": allc("plans_equal") and allc("adapter_repeat_equal") and allc("directives_equal"),
        "scene_identical": bool(vs) and all(all(v.get("scene_equal", {}).values()) and v.get("scene_equal") for v in vs),
        "motion_stable": bool(vs) and all(v.get("motion_max_diff_vs_reference_m") and
                                          max(v["motion_max_diff_vs_reference_m"].values()) <= C.TOL["deterministic_m"]
                                          for v in vs),
        "motion_material": any(max((v.get("motion_max_diff_vs_reference_m") or {"x": 0.0}).values())
                               > C.TOL["recovered_m"] for v in vs),
        "motion_gate": allc("motion_gate"), "coverage": allc("coverage_ok"),
        "controls_discriminate": allc("controls_discriminate"),
        "isolation_ok": allc("isolation_ok"), "integrity_ok": allc("integrity_ok"), "source_ok": allc("source_ok"),
        "environment_ok": allc("environment_ok"), "encode_valid": allc("encode_valid"),
        "canonical_injective": bool(vs) and all(v.get("control_rejection") is None and v.get("meta_rejection") is None
                                                and "control_accounting" in v and "meta_accounting" in v for v in vs),
        "variation_realized": all(p["uid_changed"] > 0 for p in res["pairs"].values()
                                  if p["run"] == run and p["kind"] == "seed_only")
        and all(p["source_path_differs"] for p in res["pairs"].values() if p["run"] == run and p["kind"] == "path_only")
        and sum(1 for p in res["pairs"].values() if p["run"] == run and p["kind"] in ("seed_only", "path_only")) == 4,
        "attribution_clean": all(p["uid_changed"] == 0 and p["object_order_equal"] and p["connection_order_equal"]
                                 and not p["source_path_differs"] and not p["document_uid_changed"]
                                 for p in res["pairs"].values() if p["run"] == run and p["kind"] == "same_cell")
        and all(p["uid_changed"] == 0 and p["object_order_equal"] and p["connection_order_equal"]
                and not p["document_uid_changed"]
                for p in res["pairs"].values() if p["run"] == run and p["kind"] == "path_only")
        and all(not p["source_path_differs"] for p in res["pairs"].values() if p["run"] == run and p["kind"] == "seed_only")}


def _probe_check(runs):
    out = {}
    for run in C.RUNS:
        pr = {c: r["python_env"] for c, r in runs[run]["cells"].items()}
        same = all(pr[a]["probes"] == pr[b]["probes"] for a, b in (("s1_pA", "s1_pB"), ("s1_pA", "s1_pA_r"),
                                                                  ("s2_pA", "s2_pB")) if a in pr and b in pr)
        diff = all(pr[a]["probes"][p] != pr[b]["probes"][p] for a, b in (("s1_pA", "s2_pA"),) if a in pr and b in pr
                   for p in pr[a]["probes"])
        out[run] = {"probes": {c: v["probes"] for c, v in pr.items()}, "same_seed_equal": same, "seeds_differ": diff,
                    "seed_as_declared": all(r["seed_as_declared"] for r in runs[run]["cells"].values()),
                    "no_user_site": all(not v["sys_path_user_site"] for v in pr.values()), "cells": len(pr)}
    return out


def analyze(o):
    decl = _decl(o)
    base = Path(o.base)
    out = Path(o.out_dir)
    out.mkdir(parents=True, exist_ok=False)
    runs = {r: _load_run(base, r) for r in C.RUNS}
    rep = {"schema": C.SCHEMA + ":analysis", "declaration_sha256": o.declaration_sha, "cases": {}, "per_run": {}}
    bind = {}
    for run, R in runs.items():
        enc, cons = R["enc"] or {}, R["cons"] or {}
        bind[run] = {
            "cells_bound": len(R["cells"]) == len(C.CELLS) and all(c["declaration_sha256"] == o.declaration_sha
                                                                    for c in R["cells"].values()),
            "cells_exit_0": bool(R["cells_runs"]) and all(x["returncode"] == 0 for x in R["cells_runs"]["runs"]),
            "encode_bound": enc.get("declaration_sha256") == o.declaration_sha and all(
                enc.get("cells", {}).get(c, {}).get("cell_report_sha256") == R["cells_sha"].get(c) for c in R["cells"]),
            "consume_bound": cons.get("consumer_input_sha256") == R["ci_sha"] == enc.get("consumer_input_sha256"),
            "adapter_sha_declared": cons.get("adapter_sha256") == decl["consumer"]["adapter_sha256"],
            "consume_exit_0": (R["consume_run"] or {}).get("returncode") == 0,
            "encode_valid": bool(enc.get("valid")), "consumer_input_clean": bool(enc.get("consumer_input_clean")),
            "controls_present": R["ctl"] is not None}
    rep["bindings"] = bind
    rep["hash_probes"] = _probe_check(runs)
    t0 = time.perf_counter()
    for cid, case in decl["cases"].items():
        rep["cases"][cid] = analyze_case(cid, case, runs, decl, out)
        print(f"analysed {cid} ({time.perf_counter() - t0:.0f} s)", flush=True)
    for run in C.RUNS:
        rep["per_run"][run] = {cid: _per_run(rep["cases"][cid], run, cid) for cid in decl["cases"]}
    rep["controls"] = {run: {"ok": (runs[run]["ctl"] or {}).get("ok"),
                             "outcomes": {c: {k: [x["outcome"], x["ok"]] for k, x in r["controls"].items()}
                                          for c, r in (runs[run]["ctl"] or {}).get("cases", {}).items()}}
                       for run in C.RUNS}
    # originals and tracked files
    o2 = _NS(w106=o.w106, w107=o.w107, w108=o.w108, w109=o.w109)
    _, man = H10._manifests(o2)
    w110 = Path(o.w110)
    m10 = json.loads((w110/"evidence_manifest.json").read_text())
    chk = check_manifest(w110, m10, "manifest_stdout.txt")
    rep["originals"] = {t: {k: man[t][k] for k in ("manifest_sha_ok", "files", "bad", "files_ok")} for t in man}
    rep["originals"]["w110"] = {"manifest_sha_ok": sha256(w110/"evidence_manifest.json") == W110_MANIFEST_SHA,
                                "declaration_sha_ok": sha256(w110/"results/declare/declaration.json")
                                == W110_DECLARATION_SHA, "files": chk["files"], "bad": chk["bad"], "files_ok": chk["ok"]}
    rep["sources_end"] = {cid: {pk: sha256(c["sources"][pk]["path"]) == c["sources"][pk]["sha256"] for pk in C.PATHS}
                          for cid, c in decl["cases"].items()}
    shared = git("status", "--porcelain", "--", *SHARED_PATHS).splitlines()
    rep["shared_tracked_changes"] = [s for s in shared if not s.startswith("??")]
    rep["scripts_sha256_end_equal_declared"] = {p: sha256(ROOT/p) == s for p, s in decl["scripts_sha256"].items()}
    ver = json.loads(Path(o.verify).read_text())
    PR = rep["per_run"]

    def every(f):
        return all(PR[r][c][f] for r in C.RUNS for c in PR[r])
    hp = rep["hash_probes"]
    agg = {r: {c: {k: v for k, v in PR[r][c].items()} for c in PR[r]} for r in C.RUNS}
    pre = {
        "preflight_ok": bool(ver["ok"] and not rep["shared_tracked_changes"] and all(all(b.values()) for b in bind.values())
                             and all(rep["scripts_sha256_end_equal_declared"].values())),
        "environment_fixed": bool(every("environment_ok") and all(h["seed_as_declared"] and h["no_user_site"]
                                                                  for h in hp.values())),
        "all_complete": every("complete"),
        "encoding_valid": every("encode_valid") and all(b["encode_valid"] for b in bind.values()),
        "consumer_input_clean": all(b["consumer_input_clean"] for b in bind.values()),
        "isolation_ok": every("isolation_ok"),
        "adapter_integrity": every("integrity_ok"),
        "sources_ok": every("source_ok") and all(all(v.values()) for v in rep["sources_end"].values()),
        "canonical_injective": every("canonical_injective"),
        "negative_controls_ok": all(rep["controls"][r]["ok"] is True for r in C.RUNS),
        "repeat_ok": bool(agg["run1"] == agg["run2"] and every("complete")),
        "originals_unchanged": bool(all(v["manifest_sha_ok"] and v["files_ok"] for v in rep["originals"].values())
                                    and rep["originals"]["w110"]["declaration_sha_ok"]),
        "semantic_identical": {c: all(PR[r][c]["semantic_identical"] for r in C.RUNS) for c in decl["cases"]},
        "plans_identical": {c: all(PR[r][c]["plans_identical"] for r in C.RUNS) for c in decl["cases"]},
        "scene_identical": {c: all(PR[r][c]["scene_identical"] for r in C.RUNS) for c in decl["cases"]},
        "motion_stable": {c: all(PR[r][c]["motion_stable"] for r in C.RUNS) for c in decl["cases"]},
        "motion_material": {c: any(PR[r][c]["motion_material"] for r in C.RUNS) for c in decl["cases"]},
        "motion_gate": {c: all(PR[r][c]["motion_gate"] for r in C.RUNS) for c in decl["cases"]},
        "coverage": {c: all(PR[r][c]["coverage"] for r in C.RUNS) for c in decl["cases"]},
        "controls_discriminate": every("controls_discriminate"),
        "variation_realized": bool(every("variation_realized") and all(h["seeds_differ"] for h in hp.values())),
        "attribution_clean": bool(every("attribution_clean") and all(h["same_seed_equal"] for h in hp.values()))}
    case, why = C.decide_case(pre)
    rep["decision_inputs"] = pre
    rep["CASE"], rep["CASE_reason"] = case, why
    rep["analysis_seconds"] = time.perf_counter() - t0
    s = wjson(out/"analysis_report.json", rep)
    print(json.dumps({"CASE": case, "reason": why, "sha256": s, "decision": pre}, indent=1, default=str))
    return 0


# --- manifest -----------------------------------------------------------------------------------------------------
def manifest(o):
    base = Path(o.base)
    files = {}
    for p in sorted(base.rglob("*")):
        if p.is_file() and p.name not in ("evidence_manifest.json", "manifest_stdout.txt"):
            files[p.relative_to(base).as_posix()] = {"sha256": sha256(p), "bytes": p.stat().st_size}
    _, man = H10._manifests(o)
    w110 = Path(o.w110)
    chk = check_manifest(w110, json.loads((w110/"evidence_manifest.json").read_text()), "manifest_stdout.txt")
    unchanged = {t: man[t]["manifest_sha_ok"] and man[t]["files_ok"] for t in man}
    unchanged["w110"] = chk["ok"] and sha256(w110/"evidence_manifest.json") == W110_MANIFEST_SHA
    rep = {"schema": C.SCHEMA + ":manifest", "starting_head": o.starting_head, "declaration_sha256": o.declaration_sha,
           "originals_unchanged": unchanged, "all_originals_unchanged": all(unchanged.values()), "files": files}
    s = wjson(base/"evidence_manifest.json", rep)
    print(s, len(files), rep["all_originals_unchanged"])
    return 0 if rep["all_originals_unchanged"] else 2


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=("verify", "declare", "cells", "encode", "consume", "controls", "analyze",
                                     "manifest"))
    for name in ("base", "w106", "w107", "w108", "w109", "w110", "verify", "options", "lock", "blender", "addon_dir",
                 "starting_head", "branch", "declaration", "declaration_sha", "run_dir", "out_dir"):
        ap.add_argument("--"+name.replace("_", "-"))
    o = ap.parse_args()
    return {"verify": verify, "declare": declare, "cells": cells, "encode": encode, "consume": consume,
            "controls": controls, "analyze": analyze, "manifest": manifest}[o.mode](o)


if __name__ == "__main__":
    sys.exit(main())
