#!/usr/bin/env python3
"""W127 evidence harness: opt-in FBX continuous-time interchange of the verified editable Action (EXPERIMENTAL).

Stages (each refuses an existing output directory; every stage after ``declare`` refuses to run if any instrument file
changed since the declaration):

  preflight  hash-verified copies of the W126 S1/S3 saved-artifact runs (file-by-file against the W126 evidence manifest),
             a second S1 copy at another path (exporter path variation), target states (baseline, the W126-authorized
             accepted W126V state, the current head) verified, rig topology probe of every target .blend.
  declare    freeze instrument hashes, the control matrix, the real-run expectations, metrics, precedence and the CASE rule
             BEFORE any W127 control or real interchange outcome is examined.
  real       per source x state: stock ``fbx`` (unchanged), stock FBX on the evaluator grid, opt-in ``fbx-interchange`` ->
             ``fbx-consume`` -> ``fbx-verify`` and the raw (stock importer only) A/B; run snapshots + status before/after;
             fresh repeat with exporter UID + source-path variation (S1, W126V).
  visual     30 fps A/B: source | original .blend | stock FBX import | opt-in recovered (S1, S3 W126V) + a quarter-frame
             review window (4x slow) and contact sheet; exact video hashes and decoded frame counts.
  pytest     focused (W127 + W124/W125/W126 suites) and bounded regression (junit xml).
  assess     frozen gates -> CASE.
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

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "src", ROOT / "scripts", ROOT / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
PY = sys.executable
W126 = Path("C:/Users/dna10/animcv-w126")
W126_MANIFEST_SHA = "f9139cd609232011e4dd82c7265b7c2dc8568291d5ee0011d7a7ee29db105e8c"
W127 = Path("C:/Users/dna10/animcv-w127")
SOURCES = ("S1", "S3")
STATE_TAGS = ("baseline", "w126v", "head")
BLENDER = os.environ.get("ANIMCV_BLENDER", "C:/Program Files/Blender Foundation/Blender 5.1/blender.exe")
STARTING_COMMIT = "4229fe7eca6ce79bbb39a34678815d04cb45e177"
INSTRUMENT = ["src/app/experimental_video_action.py", "src/blender/experimental_action_stage.py",
              "src/blender/experimental_fbx_interchange.py", "tests/test_experimental_fbx_interchange.py",
              "tests/w127_fbx_controls_blender.py", "scripts/w127_fbx_interchange_harness.py"]
UNCHANGED = ["src/blender/export.py", "src/blender/executor.py", "src/app/experimental_action_txn.py", "src/app/cli.py",
             "scripts/w110_fbx_consumer_adapter.py", "tests/test_experimental_video_action.py", "tests/test_experimental_action_txn.py",
             "tests/test_experimental_gate_integrity.py", "docs/next_batch.md"]
FOCUSED = ["tests/test_experimental_fbx_interchange.py", "tests/test_experimental_video_action.py", "tests/test_experimental_action_txn.py",
           "tests/test_experimental_gate_integrity.py"]
REGRESSION = ["tests/test_cli.py", "tests/test_gui_app.py", "tests/test_blender_executor.py", "tests/test_motion_graph.py",
              "tests/test_keyframe_collapse.py", "tests/test_w116_semantic_pin.py", "tests/test_w118_transition_edit.py",
              "tests/test_w105_rig_utility.py", "tests/test_fk_solver.py", "tests/test_canonical_pose_parity.py", "tests/test_retarget_solver.py",
              "tests/test_w121_independent_transfer.py", "tests/test_w122_realtime_utility.py", "tests/test_render_blender_animation_video.py",
              "tests/test_w117_temporal_pin.py", "tests/test_w120_factor_attribution.py", "tests/test_w107_fbx_ownership.py",
              "tests/test_w108_fbx_subframe.py", "tests/test_w109_fbx_constant_import.py", "tests/test_w110_fbx_metadata_recovery.py",
              "tests/test_w111_fbx_semantic_repeatability.py", "tests/test_w112_fbx_within_property.py"]
RAW_HELD = ["SUBFRAME_WORLD_DEVIATION_OVER_1MM", "INTERPOLATION_NOT_RECOVERED", "SCENE_FRAME_RANGE_NOT_CARRIED"]


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def sha_lf(p):
    return hashlib.sha256(Path(p).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def load(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def dump(p, obj):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    Path(p).write_text(json.dumps(obj, indent=1, sort_keys=True, default=str), encoding="utf-8")
    return sha(p)


def new_out(p):
    p = Path(p)
    if p.exists():
        raise SystemExit(f"STOP: {p} exists (outputs are never overwritten)")
    p.mkdir(parents=True)
    return p


def git(*a):
    return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def env():
    e = dict(os.environ, PYTHONPATH=str(ROOT / "src"), ANIMCV_BLENDER=BLENDER)
    for k in ("ANIMCV_EXP_TXN_FAULT", "ANIMCV_EXP_FBXI_FAULT"):
        e.pop(k, None)
    return e


def cli(*args, extra_env=None):
    t0 = time.time()
    r = subprocess.run([PY, "-X", "utf8", "-m", "app.cli", "experimental-video-action", *[str(x) for x in args]], cwd=ROOT,
                       env=dict(env(), **(extra_env or {})), capture_output=True, text=True, timeout=3600)
    return {"cmd": " ".join(str(x) for x in args), "exit": r.returncode, "stdout": r.stdout, "stderr": r.stderr[-4000:],
            "seconds": round(time.time() - t0, 3)}


def snapshot(run):
    return {str(p.relative_to(run)).replace("\\", "/"): sha(p) for p in sorted(Path(run).rglob("*")) if p.is_file()
            and p.name != "txn.lock"}


def targets(run):
    from app import experimental_video_action as E
    R = load(run / "run.json")
    refs = E.read_refs(run)[0]
    w = load(run / "edits" / "W126V" / "commit.json")
    return {"baseline": R["baseline_state_id"], "w126v": w["head_after"], "head": refs["head"]}


# ------------------------------------------------------------------------------------------------- preflight ---
TOPO_PROBE = r'''
import bpy, sys, json, hashlib
sys.path.insert(0, sys.argv[sys.argv.index("--") + 1])
import blender.experimental_fbx_interchange as X
out = {}
for b in sys.argv[sys.argv.index("--") + 3:]:
    bpy.ops.wm.open_mainfile(filepath=b)
    arms = [o for o in bpy.data.objects if o.type == "ARMATURE"]
    out[b] = {"armatures": len(arms), "topology": X.topology_digest({x.name: (x.parent.name if x.parent else None) for x in arms[0].data.bones}),
              "objects": sorted([o.name, o.type] for o in bpy.data.objects), "actions": sorted(a.name for a in bpy.data.actions)}
open(sys.argv[sys.argv.index("--") + 2], "w").write(json.dumps(out))
'''


def cmd_preflight(a):
    from app import experimental_video_action as E
    import blender.experimental_fbx_interchange as X
    out = new_out(a.out_dir)
    if sha(W126 / "results" / "evidence_manifest.json") != W126_MANIFEST_SHA:
        raise SystemExit("STOP DATA BLOCKED: W126 evidence manifest hash differs")
    files = load(W126 / "results" / "evidence_manifest.json")["files"]
    rep = {"git_head": git("rev-parse", "HEAD"), "next_batch_blob": git("rev-parse", "HEAD:docs/next_batch.md"),
           "blender": subprocess.run([BLENDER, "--version"], capture_output=True, text=True).stdout.splitlines()[0],
           "python": sys.version, "copies": {}, "targets": {}}
    for src in SOURCES:
        name = f"{src}_replay"
        man = {k[len(f"runs/{name}/"):]: v for k, v in files.items() if k.startswith(f"runs/{name}/")}
        for dst in ([W127 / "runs" / name] + ([W127 / "runs_pathvar" / "copy_for_path_variation" / name] if src == "S1" else [])):
            if dst.exists():
                raise SystemExit(f"STOP: {dst} exists")
            shutil.copytree(W126 / "runs" / name, dst)
            got = snapshot(dst)
            diff = sorted(k for k in set(got) | set(man) if got.get(k) != man.get(k) and k != "lineage/txn.lock")
            rep["copies"][str(dst)] = {"files": len(got), "manifest_files": len(man), "differences": diff[:50], "n_differences": len(diff)}
            if diff:
                raise SystemExit(f"STOP DATA BLOCKED: {dst} differs from the W126 evidence manifest: {diff[:5]}")
        run = W127 / "runs" / name
        R, mi = E.load_run(run)
        clip = Path(mi["intake"]["clip_path"])
        rep["targets"][src] = {"clip_exists": clip.exists(), "clip_sha256_ok": clip.exists() and E.sha256_file(clip) == mi["intake"]["clip_sha256"]}
        if not rep["targets"][src]["clip_sha256_ok"]:
            raise SystemExit(f"STOP DATA BLOCKED: {src} source clip missing or changed")
        refs, raw, kind = E.read_refs(run)
        for tag, sid in targets(run).items():
            rec, rb = E.verify_state_blend(run, sid, refs, kind)
            rep["targets"][src][tag] = {"state_id": sid, "role": rec["role"], "semantic_authorization": rec["semantic_authorization"],
                                        "blend": rec["blend"], "blend_sha256": rec["blend_sha256"], "value_digest": rec["value_digest"]}
    probe = out / "topology_probe.py"
    probe.write_text(TOPO_PROBE, encoding="utf-8")
    blends = [str(W127 / "runs" / f"{s}_replay" / rep["targets"][s][t]["blend"]) for s in SOURCES for t in STATE_TAGS]
    r = subprocess.run([BLENDER, "-b", "--python-exit-code", "1", "-P", str(probe), "--", str(ROOT / "src"), str(out / "topology.json"),
                        *blends], capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit("STOP: topology probe failed\n" + r.stdout[-2000:] + r.stderr[-2000:])
    topo = load(out / "topology.json")
    rep["topology"] = topo
    rep["topology_all_known_baserig"] = all(v["topology"] == X.KNOWN_BASERIG_TOPOLOGY_SHA256 and v["armatures"] == 1 for v in topo.values())
    if not rep["topology_all_known_baserig"]:
        raise SystemExit("STOP: a target .blend is not the known BaseRig topology")
    rep["note"] = "SAVED-ARTIFACT REAL-VIDEO REPLAY copies (W124-W126 S1/S3); not fresh perception"
    print(json.dumps({"preflight_sha256": dump(out / "preflight.json", rep), "targets": rep["targets"]}, indent=1))


# --------------------------------------------------------------------------------------------------- declare ---
def cmd_declare(a):
    import test_experimental_fbx_interchange as T
    out = new_out(a.out_dir)
    pf = load(a.preflight)
    d = {
        "schema": "w127_declaration_v1", "written_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "outcomes_inspected": False,
        "starting_commit": STARTING_COMMIT, "git_head_at_declaration": git("rev-parse", "HEAD"),
        "preflight_sha256": sha(a.preflight),
        "instrument_sha256_lf": {f: sha_lf(ROOT / f) for f in INSTRUMENT},
        "unchanged_vs_starting_commit": UNCHANGED,
        "development_disclosure": [
            "Implementation was developed on the W125/W126 8-frame SYNTHETIC-VALUE BaseRig fixture only.  Seen before this "
            "declaration: the C1 user path (nonzero pin head) INTERCHANGE_PASS (~1e-6 m) and its raw A/B FAIL "
            "(subframe 0.34 m, interpolation, scene range); two development findings that shaped the exporter arguments "
            "(armature-only export without BindPose rebuilds rest from the posed frame: up to 0.92 m; a GLOBAL scene take "
            "re-imports one frame late with the stock importer's default anim_offset).",
            "No W127 S1/S3 interchange outcome and no other control outcome was examined before this declaration.  The stock "
            "S1/S3 FAIL figures (W126) were known."],
        "contract": {
            "producer": "fbx-interchange: verified published state -> NEW package/{animation.fbx, package_manifest.json} "
                        "(manifest sealed, written LAST); exporter kwargs " + json.dumps({k: sorted(v) if isinstance(v, set) else v for k, v in
                                                                                         __import__("blender.experimental_fbx_interchange",
                                                                                                    fromlist=["x"]).EXPORT_KWARGS.items()}),
            "flags": "CONSTANT-driven (item, transform group) FBX curves 0x6104 -> 0x6002 byte-exact in place; LINEAR / UNDRIVEN keep 0x6104",
            "consumer": "fbx-consume: reads ONLY the package manifest + FBX (audited); stock importer default flags; per-group "
                        "interpolation from the FBX flags cross-checked with the manifest; scene range from the manifest",
            "evaluator": "fbx-verify / evaluate_pair: fresh Blender readback of the original .blend AND of the imported .blend; "
                         "every non-finger bone head/tail at every integer frame and at +0.25/+0.5/+0.75/+0.99",
            "dependency": "raw FBX alone is NOT claimed lossless; recovery REQUIRES the package manifest and the Blender 5.1 adapter"},
        "metrics": {"world_drift_threshold_m": 1e-3, "threshold_note": "historical W106/W107 EXPERIMENTAL comparison diagnostic, not a "
                    "Production joint tolerance", "object_matrix_tol": 1e-5, "rest_tol_m": 1e-5, "pin_residual_threshold_m": 1e-3,
                    "grid_subframes": [0.25, 0.5, 0.75, 0.99]},
        "mismatch_codes": ["BONE_SET_DIFFERS", "FRAME_SET_DIFFERS", "SUBFRAME_SET_DIFFERS", "INTEGER_FRAME_WORLD_DRIFT_OVER_1MM",
                           "SUBFRAME_WORLD_DEVIATION_OVER_1MM", "OBJECT_TRANSFORM_DIFFERS", "REST_GEOMETRY_DIFFERS",
                           "ARMATURE_SCALE_DIFFERS", "INTERPOLATION_NOT_RECOVERED", "MULTIPLE_ACTIONS_AFTER_IMPORT", "KEY_RANGE_CHANGED",
                           "SCENE_FPS_DIFFERS", "SCENE_FRAME_RANGE_NOT_CARRIED", "PIN_RESIDUAL_OVER_1MM"],
        "consumer_refusal_precedence": ["REFUSED_EXISTING_OUTPUT", "PACKAGE_INCOMPLETE (manifest)", "MANIFEST_UNPARSABLE",
                                        "MANIFEST_SCHEMA_UNSUPPORTED", "MANIFEST_SEAL_INVALID", "UNSUPPORTED_CONSUMER_BLENDER_VERSION",
                                        "UNSUPPORTED_PRODUCER_BLENDER_VERSION", "UNSUPPORTED_IMPORTER_VERSION", "PACKAGE_INCOMPLETE (fbx)",
                                        "FBX_HASH_MISMATCH", "FBX_PARSE_ERROR", "TAKE_NOT_UNIQUE", "TAKE_MISMATCH", "TIMEBASE_MISMATCH (FBX)",
                                        "TAKE_RANGE_MISMATCH", "UNSUPPORTED_RIG", "FBX_CURVE_MAP_*", "METADATA_FBX_DISAGREE (group table)",
                                        "CONSUMER_IMPORT_FAILED", "RIG_MISMATCH", "SCENE_OBJECTS_MISMATCH", "ACTION_NOT_UNIQUE_OR_WRONG_TAKE",
                                        "ACTION_NOT_ASSIGNED", "TIMEBASE_MISMATCH (scene)", "UNRESOLVED_CURVES",
                                        "METADATA_FBX_DISAGREE (directives)", "KEY_RANGE_MISMATCH", "POST_APPLY_CENSUS_MISMATCH"],
        "producer_refusal_precedence": ["UNSUPPORTED_ARMATURE_COUNT", "UNSUPPORTED_RIG", "scene checks (objects, NLA, drivers, "
                                        "constraints)", "per-F-curve census (channel, modifier, non-integer keys, empty)",
                                        "per-group interpolation (UNSUPPORTED_INTERPOLATION_MIXTURE / _<KIND>)", "fps_base",
                                        "UNSUPPORTED_TIME_ORIGIN_OR_RANGE", "PRODUCER_* (export / take / map / patch checks)"],
        "controls_focused_tests": {
            "C1": "fixture head with accepted nonzero pin P1 (all CONSTANT): fbx-interchange/consume/verify INTERCHANGE_PASS, "
                  "pin residual <= 1 mm; raw mismatches == " + json.dumps(T.RAW_HELD) + "; baseline package PASS; all run files and "
                  "status unchanged; exact revert after export; package verified against its state, refused for another state",
            "C2": "smooth LINEAR control: 0 patched curves, PASS; blanket CONSTANT on the recovered copy -> SUBFRAME_WORLD_DEVIATION_OVER_1MM "
                  "(integer <= 1 mm); raw mismatches == [SCENE_FRAME_RANGE_NOT_CARRIED]",
            "C3": "per-group mixed (upperarm_l LINEAR, others CONSTANT): 21 patched curves, PASS; blanket CONSTANT -> SUBFRAME FAIL; "
                  "raw mismatches == RAW_HELD",
            "C4": T.UNSUPPORTED_SOURCES, "C5a": "extra unassigned Action: excluded (listed), single take, PASS",
            "C5_C6_package": T.PACKAGE_NEGATIVES,
            "C6_producer": "stage fault after_raw_export -> PRODUCER_STAGE_FAILED, STATUS FAILED; host kill before_manifest -> exit 86, "
                           "STATUS IN_PROGRESS, FBX present without manifest; both consumed -> PACKAGE_INCOMPLETE; existing out refused",
            "C7": "stock `fbx` code identical to the starting commit; still FAIL with SUBFRAME, INTERPOLATION_CHANGED, SCENE_FRAME_RANGE",
            "C8": "second export of the same head from a copied run at another path in another process: raw exporter bytes differ; "
                  "group table / interpolation / take / timebase / rig equal; recovered readbacks bit-equal",
            "C9": "two concurrent exports to one --out: exactly one COMPLETE, the other REFUSED_EXISTING_OUTPUT; an export during an "
                  "in-flight commit (W125 hang before publication) -> RECOVERY_REQUIRED, no output; the commit then completes"},
        "real_expectations": {
            "sources": "S1, S3 SAVED-ARTIFACT REAL-VIDEO REPLAY copies; states baseline, w126v (AUTHORIZED_AT_COMMIT), head (reverted)",
            "opt_in": "fbx-verify INTERCHANGE_PASS for all 6 (integer and subframe <= 1 mm, interpolation recovered, key/scene range, "
                      "fps, single Action, rest <= 1e-5 m, object <= 1e-5; w126v pin residual <= 1 mm); consumer UNDECLARED reads empty",
            "raw": "fbx-verify FAIL with mismatches == " + json.dumps(RAW_HELD) + " (integer <= 1 mm) for all 6",
            "stock_fbx_command": "roundtrip_verdict FAIL for all 6 (characterisation, unchanged code); w126v figures expected equal to W126 "
                                 "(S1 3.90 mm / 0.69 m, S3 6.77 mm / 0.21 m)",
            "stock_same_grid": "FAIL for all 6, with BONE_SET_DIFFERS (leaf bones), SUBFRAME, INTERPOLATION_NOT_RECOVERED, "
                               "SCENE_FRAME_RANGE_NOT_CARRIED among the mismatches",
            "authoring_safety": "run snapshot unchanged except NEW files under fbx/ (stock command); status output identical; source "
                                ".blend hashes equal",
            "repeat": "S1 w126v from the path-variation copy: raw exporter bytes differ; recovered readbacks bit-equal to the first"},
        "gates": {
            "G1_user_path_and_default_isolation": "C7 + default-isolation + W124 parser-digest tests PASS",
            "G2_mechanism_controls": "C1, C2, C3, C5a PASS",
            "G3_negative_controls_safety": "every C4 / C5_C6 / C6_producer control refused WITHOUT a package or recovered .blend "
                                           "(a failing negative test whose ONLY failed assertion is marked CLASSIFICATION: is not a breach)",
            "G3b_negative_controls_classification": "every refusal carries its declared code (all C4 / C5_C6 / C6 tests PASS)",
            "G4_repeatability": "C8 PASS and the real S1 repeat equal",
            "G5_isolation_concurrency": "C9 PASS and every real consumer audit has no UNDECLARED read",
            "G6_real": "all real_expectations met (opt-in, raw, stock, stock same-grid, authoring safety)",
            "G7_regression": "W124 35/35, W125 58/58, W126 40/41 with ONLY the historical N12 failing, bounded regression all PASS",
            "G8_visual": "S1 and S3 4-panel 30 fps videos written and decoded with frame count = Action frames; quarter-frame review written"},
        "case_rule": {
            "D": "instrument invalid or contaminated: a consumer UNDECLARED read, an evaluator readback not of its .blend, an "
                 "instrument file changed after this declaration, a run/source contamination (G6 authoring safety false), or any "
                 "negative control producing a package / recovered .blend (G3 false)",
            "A": "not D and G1..G8 all true",
            "B": "not D, not A, G6 opt-in INTERCHANGE_PASS on all 6 real states, and G2, G3 (safety), G5 true",
            "C": "not D, not A, not B (in particular: the real opt-in path does not pass on all 6 states)"},
        "optional_non_gating": "UNITY probe (Unity editors are installed on Main): MAY be attempted after the gated stages; it never "
                               "enters the CASE; if it cannot run it is reported NOT TESTED.  Unreal / other engines: NOT available.",
    }
    s = dump(out / "declaration.json", d)
    print(json.dumps({"declaration": str(out / "declaration.json"), "sha256": s}, indent=1))


def check_decl(a):
    if sha(a.declaration) != a.declaration_sha:
        raise SystemExit("STOP: declaration hash mismatch")
    d = load(a.declaration)
    bad = [f for f, h in d["instrument_sha256_lf"].items() if sha_lf(ROOT / f) != h]
    if bad:
        raise SystemExit(f"STOP: instrument changed after the declaration: {bad}")
    return d


# ------------------------------------------------------------------------------------------------------ real ---
def _status(run):
    r = cli("status", "--run", run)
    return r["stdout"]


def cmd_real(a):
    from app import experimental_video_action as E
    import blender.experimental_fbx_interchange as X
    check_decl(a)
    out = new_out(a.out_dir)
    idx = {"sources": {}}
    for src in SOURCES:
        run = W127 / "runs" / f"{src}_replay"
        R = load(run / "run.json")
        tg = targets(run)
        snap0, st0 = snapshot(run), _status(run)
        srec = {"targets": tg, "states": {}}
        for tag in STATE_TAGS:
            sid = tg[tag]
            d = out / src / tag
            d.mkdir(parents=True)
            rec = {"state_id": sid}
            reg = E.blend_registry(run)
            blend = run / reg[sid]["blend"]
            rec["blend_sha256_before"] = sha(blend)
            # 1. stock fbx command (unchanged)
            r = cli("fbx", "--run", run, "--state", sid)
            stock_dir = run / "fbx" / f"{sid}_{sid[:12]}"
            rec["stock_cmd"] = {"exit": r["exit"], "seconds": r["seconds"], "dir": str(stock_dir)}
            if r["exit"] == 0:
                sr = load(stock_dir / "fbx_roundtrip_report.json")
                rec["stock_cmd"].update({k: sr.get(k) for k in ("roundtrip_verdict", "mismatches", "integer_world_max_err_m",
                                                                 "subframe_world_max_err_m", "integer_world_err_median_m", "fbx_fcurves",
                                                                 "fbx_interpolation", "fbx_actions", "fbx_key_range", "fbx_scene")})
                fbx = stock_dir / f"{sid[:12]}.fbx"
                rec["stock_cmd"]["fbx_bytes"] = fbx.stat().st_size
                # 2. stock FBX on the evaluator grid
                (d / "stock_import").mkdir()
                rc = E._stage(BLENDER, "fbx_import_save", d / "stock_import" / "import.log", fbx=fbx,
                              out_blend=d / "stock_import" / "stock_import.blend", report=d / "stock_import" / "import.json")
                if rc == 0:
                    (d / "stock_eval").mkdir()
                    ev = E.evaluate_pair(BLENDER, blend, d / "stock_import" / "stock_import.blend", d / "stock_eval",
                                         E.pin_context(run, sid), "STOCK_EXPORTER_STOCK_IMPORTER")
                    dump(d / "stock_eval" / "eval_report.json", ev)
                    rec["stock_same_grid"] = _brief(ev)
                else:
                    rec["stock_same_grid"] = {"error": f"import exit {rc}"}
            # 3-5. opt-in
            r = cli("fbx-interchange", "--run", run, "--state", sid, "--out", d / "export")
            rec["export"] = {"exit": r["exit"], "seconds": r["seconds"], "stderr": r["stderr"][-800:]}
            if r["exit"] == 0:
                rec["export"]["status"] = load(d / "export" / "STATUS.json")
                rec["export"]["manifest_sha256"] = sha(d / "export" / "package" / X.PACKAGE_MANIFEST)
                for mode, cd, vd in (("adapter", "consume", "verify"), ("raw", "consume_raw", "verify_raw")):
                    rc_ = cli("fbx-consume", "--package", d / "export" / "package", "--out", d / cd, "--mode", mode)
                    rec[cd] = {"exit": rc_["exit"], "seconds": rc_["seconds"], "stderr": rc_["stderr"][-800:]}
                    if rc_["exit"] == 0:
                        cr = load(d / cd / "consume_report.json")
                        rec[cd].update({"verdict": cr["verdict"], "audit": {k: len(v) for k, v in cr["file_access_audit"]["classified"].items()},
                                        "undeclared": cr["file_access_audit"]["classified"]["UNDECLARED"],
                                        "declared_inputs": cr["file_access_audit"]["classified"]["DECLARED_INPUT"],
                                        "consumer_seconds": cr.get("seconds"), "census": cr.get("recovered_interpolation_census")})
                        rv = cli("fbx-verify", "--run", run, "--consumed", d / cd, "--out", d / vd)
                        rec[vd] = {"exit": rv["exit"], "seconds": rv["seconds"], "stderr": rv["stderr"][-800:]}
                        if rv["exit"] == 0:
                            vr = load(d / vd / "fbx_verify_report.json")
                            rec[vd].update(_brief(vr), interchange_verdict=vr["interchange_verdict"])
            rec["blend_sha256_after"] = sha(blend)
            srec["states"][tag] = rec
            dump(d / "record.json", rec)
        snap1, st1 = snapshot(run), _status(run)
        changed = sorted(k for k in set(snap0) | set(snap1) if snap0.get(k) != snap1.get(k))
        srec["authoring_safety"] = {"files_before": len(snap0), "files_after": len(snap1),
                                    "changed_or_new_outside_fbx": [k for k in changed if not k.startswith("fbx/")],
                                    "removed": [k for k in snap0 if k not in snap1],
                                    "new_under_fbx": len([k for k in changed if k.startswith("fbx/") and k not in snap0]),
                                    "modified_under_fbx": [k for k in changed if k.startswith("fbx/") and k in snap0],
                                    "status_identical": st0 == st1}
        srec["authoring_safety"]["ok"] = (not srec["authoring_safety"]["changed_or_new_outside_fbx"] and not srec["authoring_safety"]["removed"]
                                          and not srec["authoring_safety"]["modified_under_fbx"] and srec["authoring_safety"]["status_identical"])
        dump(out / src / "status_before.json", {"stdout": st0})
        idx["sources"][src] = srec
    # fresh repeat with exporter UID + source path variation (S1, w126v)
    runv = W127 / "runs_pathvar" / "copy_for_path_variation" / "S1_replay"
    sid = targets(runv)["w126v"]
    d = out / "S1" / "repeat"
    d.mkdir()
    r = cli("fbx-interchange", "--run", runv, "--state", sid, "--out", d / "export")
    rep = {"export_exit": r["exit"], "stderr": r["stderr"][-800:]}
    if r["exit"] == 0:
        rc_ = cli("fbx-consume", "--package", d / "export" / "package", "--out", d / "consume")
        rep["consume_exit"] = rc_["exit"]
        m1 = load(out / "S1" / "w126v" / "export" / "package" / X.PACKAGE_MANIFEST)
        m2 = load(d / "export" / "package" / X.PACKAGE_MANIFEST)
        rep["raw_exporter_bytes_differ"] = m1["fbx"]["raw_exporter_output_sha256"] != m2["fbx"]["raw_exporter_output_sha256"]
        rep["package_fbx_bytes_differ"] = m1["fbx"]["sha256"] != m2["fbx"]["sha256"]
        rep["semantic_fields_equal"] = all(m1[k] == m2[k] for k in ("take", "timebase", "rig", "interpolation")) and \
            m1["fbx"]["group_table"] == m2["fbx"]["group_table"]
        if rc_["exit"] == 0:
            for n, bl in (("first", out / "S1" / "w126v" / "consume" / "recovered.blend"), ("repeat", d / "consume" / "recovered.blend")):
                E._stage(BLENDER, "readback", d / f"rb_{n}.log", blend=bl, report=d / f"rb_{n}.json", subframes="0.25,0.5,0.75,0.99",
                         frozen_audit="0")
            r1, r2 = load(d / "rb_first.json"), load(d / "rb_repeat.json")
            rep["recovered_integer_world_bit_equal"] = [f["world"] for f in r1["frames"]] == [f["world"] for f in r2["frames"]]
            rep["recovered_subframe_world_bit_equal"] = [s["world"] for s in r1["subframes"]] == [s["world"] for s in r2["subframes"]]
            rep["recovered_census_equal"] = r1["fcurve_census"]["interpolation_census_curves"] == r2["fcurve_census"]["interpolation_census_curves"]
    rep["ok"] = all(rep.get(k) for k in ("raw_exporter_bytes_differ", "semantic_fields_equal", "recovered_integer_world_bit_equal",
                                         "recovered_subframe_world_bit_equal", "recovered_census_equal")) and rep["export_exit"] == 0
    idx["repeat_S1_w126v"] = rep
    print(json.dumps({"run_index_sha256": dump(out / "run_index.json", idx)}, indent=1))


def _brief(v):
    return {"verdict": v["verdict"], "mismatches": v["mismatches"], "integer_max_m": v["integer"]["max_m"],
            "integer_median_m": v["integer"]["median_m"], "integer_frames_over_tol": len(v["integer"]["frames_over_tol"]),
            "integer_worst_at": v["integer"]["worst_at"], "subframe_max_m": v["subframe"]["max_m"], "subframe_median_m": v["subframe"]["median_m"],
            "subframe_by_offset_max_m": v["subframe"]["by_offset_max_m"], "subframe_worst_at": v["subframe"]["worst_at"],
            "subframe_samples_over_tol": v["subframe"]["samples_over_tol"], "bones_over_tol": v["bones_over_tol"],
            "object_matrix_world_max_abs": v["object_matrix_world_max_abs"], "rest_max_m": v["rest_max_m"],
            "interpolation_census": [v["interpolation"]["source_census_curves"], v["interpolation"]["imported_census_curves"]],
            "fcurves": [v["interpolation"]["source_fcurves"], v["interpolation"]["imported_fcurves"]],
            "key_range": v["key_range"], "scene": v["scene"], "actions": v["actions"],
            "pin": (v.get("pin") or {}).get("residual"), "bone_set": v.get("bone_set")}


# ---------------------------------------------------------------------------------------------------- visual ---
def cmd_visual(a):
    import cv2
    import numpy as np
    from app import experimental_video_action as E
    check_decl(a)
    out = new_out(a.out_dir)
    res = {}
    for src in SOURCES:
        run = W127 / "runs" / f"{src}_replay"
        sid = targets(run)["w126v"]
        rd = Path(a.run_dir) / src / "w126v"
        R, man = E.load_run(run)
        reg = E.blend_registry(run)
        sidecar = load(run / "baseline" / "action_sidecar.json")
        mapping = sidecar["frame_mapping"]
        pin = E.pin_context(run, sid)
        base_rb = load(run / reg[sid]["readback"])
        allp = np.array([fr["world"][b][p] for fr in base_rb["frames"] for b in fr["world"] for p in ("head", "tail")])
        centre = ((allp.min(0) + allp.max(0)) / 2).tolist()
        ext = float(np.max(allp.max(0) - allp.min(0)))
        views = [{"name": "body", "centre": centre, "offset": [0.0, -6.0, 0.0], "ortho_scale": ext * 1.15},
                 {"name": "arm", "centre": pin["target"], "offset": [0.0, -6.0, 0.0], "ortho_scale": 0.9}]
        size = 300
        pinset = set(pin["action_frames"])
        jobs = [{"blender_frame": r["action_frame"], "markers": [{"name": "pin_target", "position": pin["target"], "color": (0.1, 0.85, 0.2, 1),
                                                                   "radius": 0.02}] if r["action_frame"] in pinset else []} for r in mapping]
        d = out / src
        d.mkdir()
        E.dump(d / "jobs.json", {"size": size, "views": views, "frames": jobs})
        blends = {"original": run / reg[sid]["blend"], "stock": rd / "stock_import" / "stock_import.blend",
                  "optin": rd / "consume" / "recovered.blend"}
        t0 = time.time()
        for name, bl in blends.items():
            rc = E._stage(BLENDER, "render", d / f"{name}_render.log", blend=bl, jobs=d / "jobs.json", out_dir=d / f"frames_{name}")
            if rc != 0:
                raise SystemExit(f"STOP: render {src} {name} failed")
        render_s = time.time() - t0
        cap = cv2.VideoCapture(man["intake"]["clip_path"])
        want = {r["source_frame"] for r in mapping}
        srcf, i = {}, 0
        while True:
            ok, im = cap.read()
            if not ok:
                break
            if i in want:
                srcf[i] = im
            i += 1
        cap.release()

        def lab(img, txt, y, col=(255, 255, 255), s=0.4):
            cv2.putText(img, txt, (5, y), cv2.FONT_HERSHEY_SIMPLEX, s, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(img, txt, (5, y), cv2.FONT_HERSHEY_SIMPLEX, s, col, 1, cv2.LINE_AA)
            return img
        H = size
        titles = {"original": "ORIGINAL .blend Action", "stock": "STOCK FBX export+import", "optin": "OPT-IN package + adapter"}
        frames_out = []
        for r in mapping:
            k, f = r["source_frame"], r["action_frame"]
            s_im = srcf[k]
            sc = 2 * H / s_im.shape[0]
            s_im = lab(cv2.resize(s_im, (int(s_im.shape[1] * sc), 2 * H)), "SOURCE (full frame)", 16)
            lab(s_im, f"src {k}  Action f{f}  t={r['source_time_s']:.3f}s", 34)
            tiles = [s_im]
            for name in ("original", "stock", "optin"):
                body = cv2.imread(str(d / f"frames_{name}" / f"body_f{f:04d}.png"))
                arm = cv2.imread(str(d / f"frames_{name}" / f"arm_f{f:04d}.png"))
                lab(body, titles[name], 16)
                lab(arm, "left-arm close-up" + ("  PIN" if f in pinset else ""), 16, (120, 255, 120) if f in pinset else (255, 255, 255))
                tiles.append(np.vstack([body, arm]))
            frames_out.append(np.hstack(tiles))
        vid = d / f"ab4_{src}_w126v_30fps.mp4"
        h, w = frames_out[0].shape[:2]
        wr = cv2.VideoWriter(str(vid), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (w, h))
        for fr in frames_out:
            wr.write(fr)
        wr.release()
        cap = cv2.VideoCapture(str(vid))
        n = 0
        while cap.read()[0]:
            n += 1
        vfps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        for f in sorted({min(pinset) - 1, min(pinset), max(pinset), max(pinset) + 1}):
            if 1 <= f <= len(frames_out):
                cv2.imwrite(str(d / f"still_f{f:04d}.jpg"), frames_out[f - 1])
        # quarter-frame review window centred on the stock path's worst subframe sample (visualisation only)
        se = load(rd / "stock_eval" / "eval_report.json")
        wf = int(se["subframe"]["worst_at"][0])
        lo, hi = max(1, wf - 3), min(len(mapping) - 1, wf + 3)
        sjobs = [{"blender_frame": f, "subframe": s, "markers": []} for f in range(lo, hi + 1) for s in (0.0, 0.25, 0.5, 0.75)]
        E.dump(d / "sub_jobs.json", {"size": size, "views": views, "frames": sjobs})
        for name, bl in blends.items():
            rc = E._stage(BLENDER, "render", d / f"{name}_sub_render.log", blend=bl, jobs=d / "sub_jobs.json", out_dir=d / f"sub_{name}")
            if rc != 0:
                raise SystemExit(f"STOP: subframe render {src} {name} failed")
        sub_frames = []
        for j in sjobs:
            f, s = j["blender_frame"], j["subframe"]
            tag = f"{f:04d}" if s == 0.0 else f"{f:04d}s{int(round(s * 100)):02d}"
            tiles = []
            for name in ("original", "stock", "optin"):
                body = cv2.imread(str(d / f"sub_{name}" / f"body_f{tag}.png"))
                lab(body, titles[name], 16)
                lab(body, f"Action t = {f + s:.2f} frames", 34)
                tiles.append(body)
            sub_frames.append(np.hstack(tiles))
        svid = d / f"quarterframe_{src}_w126v_4x_slow.mp4"
        h2, w2 = sub_frames[0].shape[:2]
        wr = cv2.VideoWriter(str(svid), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (w2, h2))
        for fr in sub_frames:
            wr.write(fr)
        wr.release()
        sheet = np.vstack([sub_frames[k] for k in range(len(sjobs)) if sjobs[k]["blender_frame"] == wf])
        cv2.imwrite(str(d / f"contact_sheet_{src}_f{wf}.png"), sheet)
        res[src] = {"video": str(vid), "sha256": sha(vid), "frames_written": len(frames_out), "frames_decoded": n, "fps": vfps,
                    "seconds": n / vfps if vfps else None, "action_frames": len(mapping), "native_rate": abs(vfps - 30.0) < 1e-6,
                    "render_seconds_3_blends": round(render_s, 1), "quarter_frame_video": str(svid), "quarter_frame_sha256": sha(svid),
                    "quarter_frame_samples": len(sub_frames), "quarter_window_action_frames": [lo, hi], "worst_stock_subframe": se["subframe"]["worst_at"],
                    "contact_sheet": str(d / f"contact_sheet_{src}_f{wf}.png"), "contact_sheet_sha256": sha(d / f"contact_sheet_{src}_f{wf}.png"),
                    "panels": ["source", "original .blend", "stock FBX export+import", "opt-in package + adapter"]}
    print(json.dumps({"visual_sha256": dump(out / "visual.json", res), "res": res}, indent=1))


# ---------------------------------------------------------------------------------------------------- pytest ---
def cmd_pytest(a):
    check_decl(a)
    out = new_out(a.out_dir)
    e = env()
    res = {}
    for name, files in (("focused", FOCUSED), ("regression", [f for f in REGRESSION if (ROOT / f).exists()])):
        t0 = time.time()
        r = subprocess.run([PY, "-X", "utf8", "-m", "pytest", "-q", "-p", "no:cacheprovider", *files, f"--junitxml={out / (name + '.xml')}"],
                           cwd=ROOT, env=e, capture_output=True, text=True)
        (out / f"{name}.log").write_text(r.stdout + "\n" + r.stderr, encoding="utf-8")
        res[name] = {"files": files, "exit": r.returncode, "summary": r.stdout.strip().splitlines()[-1] if r.stdout.strip() else "",
                     "seconds": round(time.time() - t0, 1), "junit_sha256": sha(out / f"{name}.xml")}
    print(json.dumps({"pytest_sha256": dump(out / "pytest.json", res), "res": res}, indent=1))


def _junit(p):
    import xml.etree.ElementTree as ET
    r = ET.parse(p).getroot()
    cases = {}
    for tc in r.iter("testcase"):
        k = tc.get("classname", "") + "::" + tc.get("name")
        cases[k] = "FAIL" if tc.find("failure") is not None or tc.find("error") is not None else ("SKIP" if tc.find("skipped") is not None else "PASS")
    return cases


# ---------------------------------------------------------------------------------------------------- assess ---
def cmd_assess(a):
    d = check_decl(a)
    out = new_out(a.out_dir)
    idx = load(Path(a.run_dir) / "run_index.json")
    vis = load(Path(a.visual_dir) / "visual.json")
    jf = _junit(Path(a.pytest_dir) / "focused.xml")
    jr = _junit(Path(a.pytest_dir) / "regression.xml")
    t127 = {k.split("::")[-1]: v for k, v in jf.items() if "fbx_interchange" in k}
    t124 = {k: v for k, v in jf.items() if k.split("::")[0].endswith("test_experimental_video_action")}
    t125 = {k: v for k, v in jf.items() if "action_txn" in k}
    t126 = {k.split("::")[-1]: v for k, v in jf.items() if "gate_integrity" in k}

    def ok(prefix):
        ks = [k for k in t127 if k.startswith(prefix)]
        return bool(ks) and all(t127[k] == "PASS" for k in ks), {k: t127[k] for k in ks}
    g = {}
    g["G1_user_path_and_default_isolation"] = all([ok("test_C7")[0], ok("test_default_cli_isolation")[0],
                                                   any(v == "PASS" for k, v in t124.items() if "default_cli_commands_unchanged" in k)])
    g["G2_mechanism_controls"] = all(ok(p)[0] for p in ("test_C1_", "test_C2_", "test_C3_", "test_C5a_"))
    neg = {k: v for k, v in t127.items() if k.startswith(("test_C4_", "test_C5_C6_", "test_C6_"))}
    breach = _negative_safety_breaches(Path(a.pytest_dir))
    g["G3_negative_controls_safety"] = bool(neg) and not breach
    g["G3b_negative_controls_classification"] = bool(neg) and all(v == "PASS" for v in neg.values())
    g["G4_repeatability"] = ok("test_C8_")[0] and bool(idx["repeat_S1_w126v"]["ok"])
    real = idx["sources"]
    undeclared = [(s, t, m) for s in real for t, r in real[s]["states"].items() for m in ("consume", "consume_raw")
                  if r.get(m, {}).get("undeclared")]
    g["G5_isolation_concurrency"] = ok("test_C9_")[0] and not undeclared
    chk = {}
    for s, sr in real.items():
        for t, r in sr["states"].items():
            v, vr = r.get("verify", {}), r.get("verify_raw", {})
            chk[f"{s}/{t}"] = {
                "opt_in_pass": v.get("interchange_verdict") == "INTERCHANGE_PASS",
                "raw_fail_as_declared": vr.get("mismatches") == RAW_HELD and (vr.get("integer_max_m") or 1) <= 1e-3,
                "stock_cmd_fail": r.get("stock_cmd", {}).get("roundtrip_verdict") == "FAIL",
                "stock_same_grid_fail_as_declared": r.get("stock_same_grid", {}).get("verdict") == "FAIL" and
                set(["BONE_SET_DIFFERS", "SUBFRAME_WORLD_DEVIATION_OVER_1MM", "INTERPOLATION_NOT_RECOVERED",
                     "SCENE_FRAME_RANGE_NOT_CARRIED"]) <= set(r.get("stock_same_grid", {}).get("mismatches", [])),
                "source_blend_unchanged": r["blend_sha256_before"] == r["blend_sha256_after"]}
        chk[f"{s}/authoring_safety"] = {"ok": sr["authoring_safety"]["ok"]}
    g6_opt = all(c.get("opt_in_pass") for k, c in chk.items() if "authoring" not in k)
    safety = all(c["ok"] for k, c in chk.items() if "authoring" in k) and all(c["source_blend_unchanged"] for k, c in chk.items()
                                                                             if "authoring" not in k)
    g["G6_real"] = all(all(c.values()) for c in chk.values())
    w126_fail = sorted(k for k, v in t126.items() if v != "PASS")
    g["G7_regression"] = (len(t124) == 35 and all(v == "PASS" for v in t124.values()) and len(t125) == 58 and all(v == "PASS" for v in t125.values())
                          and len(t126) == 41 and w126_fail == ["test_negative_control_refused_without_any_publication[N12]"] and bool(jr) and all(v == "PASS" for v in jr.values()))
    g["G8_visual"] = all(vis[s]["frames_decoded"] == vis[s]["action_frames"] == vis[s]["frames_written"] and vis[s]["native_rate"]
                         and vis[s]["quarter_frame_samples"] > 0 for s in SOURCES)
    instrument_ok = all(sha_lf(ROOT / f) == h for f, h in d["instrument_sha256_lf"].items())
    unchanged = {f: subprocess.run(["git", "diff", "--quiet", STARTING_COMMIT, "--", f], cwd=ROOT).returncode == 0 for f in UNCHANGED}
    D = (not instrument_ok) or bool(undeclared) or (not safety) or (not g["G3_negative_controls_safety"]) or not all(unchanged.values())
    if D:
        case = "D"
    elif all(g.values()):
        case = "A"
    elif g6_opt and g["G2_mechanism_controls"] and g["G3_negative_controls_safety"] and g["G5_isolation_concurrency"]:
        case = "B"
    else:
        case = "C"
    rep = {"gates": g, "real_checks": chk, "undeclared_reads": undeclared, "negative_safety_breaches": breach, "instrument_unchanged_since_declaration": instrument_ok,
           "files_unchanged_vs_starting_commit": unchanged, "w127_tests": t127, "w126_failures": w126_fail,
           "counts": {"w124": [sum(v == "PASS" for v in t124.values()), len(t124)], "w125": [sum(v == "PASS" for v in t125.values()), len(t125)],
                      "w126": [sum(v == "PASS" for v in t126.values()), len(t126)], "w127": [sum(v == "PASS" for v in t127.values()), len(t127)],
                      "regression": [sum(v == "PASS" for v in jr.values()), len(jr)]},
           "case": case, "case_rule": d["case_rule"]}
    print(json.dumps({"assessment_sha256": dump(out / "assessment.json", rep), "case": case, "gates": g}, indent=1))


def _negative_safety_breaches(pdir):
    """Failed negative-control tests whose failure is NOT only a ``CLASSIFICATION:`` assertion (= a safety breach)."""
    import xml.etree.ElementTree as ET
    out = []
    for tc in ET.parse(pdir / "focused.xml").getroot().iter("testcase"):
        if "fbx_interchange" in tc.get("classname", "") and tc.get("name", "").startswith(("test_C4_", "test_C5_C6_", "test_C6_")):
            f = tc.find("failure") if tc.find("failure") is not None else tc.find("error")
            if f is not None and "CLASSIFICATION:" not in (f.get("message", "") + (f.text or "")):
                out.append(tc.get("name"))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="stage", required=True)
    p = sub.add_parser("preflight")
    p.add_argument("--out-dir", required=True)
    p = sub.add_parser("declare")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--preflight", required=True)
    for name in ("real", "visual", "pytest", "assess"):
        p = sub.add_parser(name)
        p.add_argument("--out-dir", required=True)
        p.add_argument("--declaration", required=True)
        p.add_argument("--declaration-sha", required=True)
        if name in ("visual", "assess"):
            p.add_argument("--run-dir", required=True)
        if name == "assess":
            p.add_argument("--visual-dir", required=True)
            p.add_argument("--pytest-dir", required=True)
    a = ap.parse_args()
    {"preflight": cmd_preflight, "declare": cmd_declare, "real": cmd_real, "visual": cmd_visual, "pytest": cmd_pytest,
     "assess": cmd_assess}[a.stage](a)


if __name__ == "__main__":
    main()
