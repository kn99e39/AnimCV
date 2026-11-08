#!/usr/bin/env python3
"""W126 evidence harness: commit-time semantic gate integrity of the EXPERIMENTAL opt-in WORLD_POSITION_PIN editor.

NOT part of the product path.  It drives the user command ``python -m app.cli experimental-video-action ...`` (PYTHONPATH=src)
on SAVED-ARTIFACT REAL-VIDEO REPLAYS (copies of the W125 final S1/S3 runs, verified file-by-file against the W125 evidence
manifest; the originals are only read and re-hashed) and audits every step from this separate process, including a FRESH
Blender readback of the head .blend.

  preflight  W125 control hashes, source clips, W125 baselines, and a reproduction of the W125 proposal-edit bypass on the
             STARTING-COMMIT code (git archive export) on (a) the real-Blender BaseRig synthetic fixture and (b) a copy of S1.
  declare    frozen scenario + expectations + gates + CASE rule + instrument SHA-256 (before any real W126 outcome); binds the
             prospective matrix declaration written before the first W126 test run.
  run        execute the declared scenario on S1 and S3
  pytest     focused (W124 + W125 + W126) and bounded regression tests (junit xml)
  assess     gates and CASE from the recorded outputs only
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
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
BASE = Path(os.environ.get("W126_BASE", "C:/Users/dna10/animcv-w126"))
W125 = Path(os.environ.get("W125_BASE", "C:/Users/dna10/animcv-w125"))
PY = os.environ.get("W126_PYTHON", "C:/Projects/AnimCV/.venv_build_full/Scripts/python.exe")
BLENDER = os.environ.get("ANIMCV_BLENDER", "C:/Program Files/Blender Foundation/Blender 5.1/blender.exe")
START_COMMIT = "a475c422477684d0284ea50d80c8c5315e5f6201"
W125_MANIFEST_SHA = "9da2652173f1d25735bc9e8eb7011765c962b1b4debb119f967d1b076055d40c"
MATRIX_DECL_SHA = "aafbebed8d16477e96d252d93223eea757973cc85ff108c5be984b3beba86f94"
CLIP_SHA = {"S1": "de8feb9d515daa2bfaeed7167fdcc62028dd3f6450872c4fd81f8fb03dd6359c",
            "S3": "f6b90f2f72e9020dcc261fadb09725913fa31719a8614de26fab372826987f62"}
PIN = {"S1": {"frames": [210, 218], "capture": 210}, "S3": {"frames": [41, 49], "capture": 41}}
LEGACY_UNDECIDED = {"S1": "S1_N_UNAUTH", "S3": None}   # S3 has no undecided pre-W126 proposal (all decided in W124/W125)
AUTH = "REFUSED_SEMANTIC_AUTHORIZATION"

# ---- frozen scenario (declared before any real W126 outcome) --------------------------------------------------------------
def scenario(src):
    s = [("s01_status", "status", {}, {"exit": 0, "legacy_gate": True, "historical_not_reclassified": True})]
    if LEGACY_UNDECIDED[src]:
        s.append(("s02_accept_legacy_proposal", "commit", {"rid": LEGACY_UNDECIDED[src], "decision": "accept"},
                  {"exit": 2, "stop": "BLOCKED_LEGACY_PROPOSAL_REQUIRES_REPROPOSE", "head_unchanged": True, "files_unchanged": True}))
    s += [
        ("s03_propose_V", "propose", {"req": "W126V"}, {"exit": 0, "gate": "ACCEPT"}),
        ("s04_propose_R", "propose", {"req": "W126R"}, {"exit": 0, "gate": "ACCEPT"}),
        ("s05_propose_T", "propose", {"req": "W126T"}, {"exit": 0, "gate": "REJECT", "reasons": ["UNREACHABLE"], "candidate_file": True}),
        ("s06_T_decision_hand_edited_accept", "commit", {"rid": "W126T", "decision": "accept", "tamper": "decision_only"},
         {"exit": 2, "stop": AUTH, "primary": "PROPOSAL_SEAL_INVALID", "head_unchanged": True, "no_false_accept_display": True}),
        ("s07_T_forged_resealed_accept", "commit", {"rid": "W126T", "decision": "accept", "tamper": "forged_resealed"},
         {"exit": 2, "stop": AUTH, "primary": "RECOMPUTED_GATE_REJECT", "head_unchanged": True, "no_false_accept_display": True}),
        ("s08_T_restored_user_reject", "commit", {"rid": "W126T", "decision": "reject", "tamper": "restore"},
         {"exit": 0, "final": "REJECTED", "head_unchanged": True}),
        ("s09_reject_R", "commit", {"rid": "W126R", "decision": "reject"}, {"exit": 0, "final": "REJECTED", "head_unchanged": True}),
        ("s10_accept_V", "commit", {"rid": "W126V", "decision": "accept"},
         {"exit": 0, "final": "ACCEPTED", "authorized": True, "generation_plus_1": True, "residual_le_m": 1e-3, "footprint_in_pin": True}),
        ("s11_review_V", "review", {"rid": "W126V"}, {"exit": 0, "native_rate": True}),
        ("s12_fbx_head", "fbx", {"state": "head"}, {"exit": 0, "roundtrip": "FAIL (expected, unchanged baseline; characterisation only)"}),
        ("s13_revert", "revert", {}, {"exit": 0, "exact_undo": True, "generation_plus_1": True}),
    ]
    return s


GATES = {
    "G1_bypass_closed": "starting-commit code PUBLISHES the hand-edited UNREACHABLE proposal (fixture and S1 copy; preflight) AND the W126 "
                        "code refuses it (focused N01/N02 and real s06/s07 on S1 and S3)",
    "G2_negative_matrix": "every NEG control of the prospective matrix gives its frozen exit/STOP/primary with every no-publication "
                          "invariant; RECORDED cases as frozen (focused tests)",
    "G3_valid_blender": "V01-V05 pass on the real Blender BaseRig (focused tests)",
    "G4_transactions": "R01/R02 as frozen; W125 focused suite and W124 suite all PASS",
    "G5_real_replay": "S1 and S3: every declared step meets its expectation and every fresh-process audit passes",
    "G6_native_rate": "S1 and S3 A/B review decodes one frame per Action frame at 30 fps",
    "G7_regression_isolation": "bounded regression PASS, frozen code unchanged, W125 originals unchanged after the run",
}
CASE_RULE = ("DATA_BLOCKED/NO CASE if the W125 S1/S3 artifacts are missing or differ before the run.  D if the prospective matrix or "
             "an expectation is changed after outcomes, frozen code changes, or an audit shows a ghost/unverified accepted head.  A if "
             "G1-G7 all PASS (supports only bounded opt-in semantic gate integrity).  B if G1-G4 and G7 PASS and the real path passes "
             "on >=1 source but some G5/G6 element or declared class is limited.  C otherwise.")
FBX_NOTE = "FBX round-trip is characterisation only (expected FAIL, unchanged W124/W125 baseline); never a W126 gate."
INSTRUMENT = ["src/app/experimental_video_action.py", "src/app/experimental_action_txn.py", "src/blender/experimental_action_stage.py",
              "src/app/cli.py", "tests/test_experimental_video_action.py", "tests/test_experimental_action_txn.py",
              "tests/test_experimental_gate_integrity.py", "scripts/w126_gate_integrity_harness.py", "docs/EXPERIMENTAL_VIDEO_ACTION_WORLD_PIN.md"]
REGRESSION = ["tests/test_cli.py", "tests/test_gui_app.py", "tests/test_blender_executor.py", "tests/test_motion_graph.py",
              "tests/test_keyframe_collapse.py", "tests/test_w116_semantic_pin.py", "tests/test_w118_transition_edit.py",
              "tests/test_w105_rig_utility.py", "tests/test_fk_solver.py", "tests/test_canonical_pose_parity.py", "tests/test_retarget_solver.py",
              "tests/test_w121_independent_transfer.py", "tests/test_w122_realtime_utility.py", "tests/test_render_blender_animation_video.py",
              "tests/test_w117_temporal_pin.py", "tests/test_w120_factor_attribution.py"]
FOCUSED = ["tests/test_experimental_gate_integrity.py", "tests/test_experimental_video_action.py", "tests/test_experimental_action_txn.py"]


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def sha_lf(p):
    return hashlib.sha256(Path(p).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def dump(p, o):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    Path(p).write_text(json.dumps(o, indent=1, sort_keys=True, default=str), encoding="utf-8")
    return sha(p)


def load(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def new_out(p):
    p = Path(p)
    if p.exists():
        raise SystemExit(f"refusing to overwrite {p}")
    p.mkdir(parents=True)
    return p


def tree(root):
    return {p.relative_to(root).as_posix(): sha(p) for p in sorted(Path(root).rglob("*")) if p.is_file()}


def w125_tree_check(src):
    man = W125 / "results" / "evidence_manifest.json"
    if sha(man) != W125_MANIFEST_SHA:
        return {"ok": False, "reason": "W125 evidence manifest hash differs"}
    files = load(man)["files"]
    rel = f"runs/{src.name}/"
    exp = {k[len(rel):]: v for k, v in files.items() if k.startswith(rel)}
    got = tree(src)
    diff = sorted(k for k in set(exp) | set(got) if exp.get(k) != got.get(k))
    return {"ok": not diff and bool(exp), "files": len(exp), "n_differences": len(diff), "differences": diff[:20]}


def git(*a):
    return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def req(rid, parent, src, offset=None):
    t = {"kind": "CAPTURED_FROM_EVALUATED_STATE", "capture_frame": PIN[src]["capture"]} if offset is None else \
        {"kind": "CAPTURED_PLUS_DECLARED_OFFSET", "capture_frame": PIN[src]["capture"], "offset_m": offset}
    return {"schema": "animcv_exp_world_pin_request_v1", "request_id": rid, "parent_state_id": parent, "type": "WORLD_POSITION_PIN",
            "semantic_joint": "left_wrist", "coordinate_frame": "SCENE_WORLD", "unit": "m", "frames": PIN[src]["frames"], "target": t,
            "authorized_channels": ["lowerarm_l", "upperarm_l"]}


OFFSETS = {"W126V": None, "W126R": [0.0, 0.0, 0.03], "W126T": [0.0, 0.0, 1.0]}


def cli(code_root, run, *args, env_extra=None):
    env = dict(os.environ, PYTHONPATH=str(Path(code_root) / "src"))
    env.pop("ANIMCV_EXP_TXN_FAULT", None)
    env.update(env_extra or {})
    t0 = time.time()
    r = subprocess.run([PY, "-X", "utf8", "-m", "app.cli", "experimental-video-action", args[0], "--run", str(run), *args[1:]],
                       cwd=code_root, env=env, capture_output=True, text=True, timeout=1800)
    return {"cmd": " ".join(["experimental-video-action", *args[:1], "--run", "<run>", *args[1:]]), "exit": r.returncode,
            "stdout": r.stdout[-6000:], "stderr": r.stderr[-4000:], "seconds": round(time.time() - t0, 2), "_stdout_full": r.stdout}


# ------------------------------------------------------------------------------------------------- preflight ---
OLD_FIXTURE = r"""
import sys, json
from pathlib import Path
root = Path(sys.argv[1]); out = Path(sys.argv[2])
sys.path[:0] = [str(root / 'src'), str(root / 'scripts'), str(root / 'tests')]
import test_experimental_action_txn as T
v0 = T.build_fixture_run(out)
print(json.dumps({'baseline': v0['state_id']}))
"""


def residual_of_head(run, prop):
    """Hard-pin residual of the published head from its recorded fresh readback (independent of the proposal's claims)."""
    import numpy as np
    from app import experimental_video_action as E
    refs = load(Path(run) / "lineage" / "refs.json")
    rb = load(Path(run) / refs["head_blend"]["readback"])
    X = E.readback_by_source(rb, load(Path(run) / "baseline" / "action_sidecar.json"))
    t = np.array(prop["constraint"]["target"])
    return max(float(np.linalg.norm(np.array(X[k]["world"]["lowerarm_l"]["tail"]) - t)) for k in prop["constraint"]["frames"])


def cmd_preflight(a):
    out = new_out(a.out_dir)
    rep = {"start_commit": START_COMMIT, "git_head": git("rev-parse", "HEAD"), "instruction_blob": git("rev-parse", "HEAD:docs/next_batch.md"),
           "w125_manifest_sha256": sha(W125 / "results" / "evidence_manifest.json"), "w125_runs": {}, "clips": {}, "w125_baselines": {}}
    for s in ("S1", "S3"):
        run = W125 / "runs" / f"{s}_replay"
        rep["w125_runs"][s] = w125_tree_check(run)
        man = load(run / "manifest" / "input_manifest.json")
        clip = Path(man["intake"]["clip_path"])
        rep["clips"][s] = {"path": str(clip), "exists": clip.exists(), "sha256": sha(clip) if clip.exists() else None, "expected": CLIP_SHA[s]}
        refs = load(run / "lineage" / "refs.json")
        edits = {}
        for ed in sorted((run / "edits").iterdir()):
            p = load(ed / "proposal.json")
            c = load(ed / "commit.json") if (ed / "commit.json").exists() else None
            edits[ed.name] = {"gate": p["decision"], "candidate": p.get("candidate"), "final": c and c["final_status"],
                              "residual": p.get("hard_pin_residual_max_m"), "sealed": "proposal_seal" in p}
        rep["w125_baselines"][s] = {"refs": refs, "refs_sha256": sha(run / "lineage" / "refs.json"), "edits": edits,
                                    "baseline_blend_sha256": sha(run / "baseline" / "baseline.blend"),
                                    "review_sha256": {r.name: load(r / "review_report.json")["sha256"] for r in (run / "review").iterdir()}}
    # starting-commit code export
    code = BASE / "prechange"
    if not (code / "src" / "app" / "experimental_video_action.py").exists():
        code.mkdir(parents=True, exist_ok=True)
        subprocess.run(f'git archive {START_COMMIT} | tar -x -C "{code}"', cwd=ROOT, shell=True, check=True)
    rep["prechange_code"] = {"dir": str(code), "experimental_video_action_sha256_lf": sha_lf(code / "src/app/experimental_video_action.py")}
    # (a) fixture reproduction of the W125 bypass on the starting-commit code
    fx = out / "oldcode_fixture"
    r = subprocess.run([PY, "-X", "utf8", "-c", OLD_FIXTURE, str(code), str(fx / "run")], cwd=code, capture_output=True, text=True,
                       env=dict(os.environ, PYTHONPATH=str(code / "src")))
    repro = {"fixture_build_exit": r.returncode, "fixture_build_stderr": r.stderr[-2000:]}
    if r.returncode == 0:
        repro.update(_bypass(code, fx / "run", "REACH"))
    rep["oldcode_bypass_fixture"] = repro
    # (b) S1 copy: same bypass with a real UNREACHABLE request on the starting-commit code
    s1 = out / "oldcode_S1" / "run"
    shutil.copytree(W125 / "runs" / "S1_replay", s1)
    refs = load(s1 / "lineage" / "refs.json")
    rq = req("OLD_T", refs["head"], "S1", OFFSETS["W126T"])
    dump(out / "oldcode_S1" / "OLD_T.json", rq)
    pr = cli(code, s1, "propose", "--request", str(out / "oldcode_S1" / "OLD_T.json"))
    pr.pop("_stdout_full", None)
    rep["oldcode_bypass_S1"] = {"propose": pr, **(_bypass(code, s1, "OLD_T") if pr["exit"] == 0 else {})}
    rep["w125_runs_unchanged_after_preflight"] = {s: w125_tree_check(W125 / "runs" / f"{s}_replay")["ok"] for s in ("S1", "S3")}
    print(dump(out / "preflight.json", rep))


def _bypass(code, run, rid):
    p = load(run / "edits" / rid / "proposal.json")
    orig = p["decision"]
    before = load(run / "lineage" / "refs.json")
    p["decision"] = {"decision": "ACCEPT", "failure_class": "NONE", "reasons": []}
    (run / "edits" / rid / "proposal.json").write_text(json.dumps(p, indent=1), encoding="utf-8")
    c = cli(code, run, "commit", "--request-id", rid, "--decision", "accept")
    c.pop("_stdout_full", None)
    after = load(run / "lineage" / "refs.json")
    out = {"original_decision": orig, "tamper": "proposal.json decision -> ACCEPT (no other change)", "commit": c,
           "head_before": before["head"], "head_after": after["head"], "published": after["head"] == p["candidate"]}
    if out["published"]:
        out["hard_pin_residual_max_m_of_published"] = residual_of_head(run, p)
    return out


# --------------------------------------------------------------------------------------------------- declare ---
def cmd_declare(a):
    out = new_out(a.out_dir)
    from app import experimental_video_action as E
    md = BASE / "declare" / "matrix_declaration.json"
    d = {"schema": "w126_declaration_v1", "declared_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "outcomes_inspected_real": False,
         "matrix_declaration": {"path": str(md), "sha256": sha(md), "expected_sha256": MATRIX_DECL_SHA},
         "scenario": {s: scenario(s) for s in ("S1", "S3")}, "requests": {k: {"offset_m": v} for k, v in OFFSETS.items()}, "pin": PIN,
         "gates": GATES, "case_rule": CASE_RULE, "fbx": FBX_NOTE, "reason_order": list(E.AUTH_REASON_ORDER),
         "instrument_sha256_lf": {p: sha_lf(ROOT / p) for p in INSTRUMENT}, "git_head": git("rev-parse", "HEAD"),
         "frozen_code_sha256_lf": {k: v["sha256_lf"] for k, v in E.frozen_code_report().items()}}
    print(dump(out / "declaration.json", d))


def check_decl(a):
    if sha(a.declaration) != a.declaration_sha:
        raise SystemExit("declaration hash mismatch")
    d = load(a.declaration)
    bad = [p for p, h in d["instrument_sha256_lf"].items() if sha_lf(ROOT / p) != h]
    if bad:
        raise SystemExit(f"instrument changed since declaration: {bad}")
    if sha(d["matrix_declaration"]["path"]) != MATRIX_DECL_SHA:
        raise SystemExit("prospective matrix declaration changed")
    return d


# ------------------------------------------------------------------------------------------------------- run ---
def audit(run, src, step):
    """Separate-process audit: authority chain + FRESH Blender readback of the head + display truth + no ghost acceptance."""
    import numpy as np
    from app import experimental_video_action as E
    import w118_transition_edit_base as XB
    import w116_semantic_pin_base as S
    run = Path(run)
    a = {"step": step}
    refs, raw, kind = E.read_refs(run)
    a["refs_kind"] = kind
    a["head_verified"] = E.verify_head(run, refs)
    hb = refs["head_blend"]
    tmp = BASE / "tmp_audit" / f"{src}_{step}"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    E.run_blender(BLENDER, "readback", tmp / "rb.log", blend=run / hb["blend"], report=tmp / "rb.json",
                  subframes=",".join(str(s) for s in E.SUBFRAMES))
    fresh, rec = load(tmp / "rb.json"), load(run / hb["readback"])
    ints, subs = E._frames_equal(fresh, rec)
    a["fresh_readback"] = {"value_digest_equal": fresh["value_digest"] == hb["value_digest"], "blend_sha_bound": fresh["blend_sha256"] == hb["blend_sha256"],
                           "integer_frames_bit_equal": ints, "subframes_bit_equal": subs}
    st = json.loads(cli(ROOT, run, "status")["_stdout_full"])
    chain = [x["state_id"] for x in st["lineage"]]
    a["no_ghost_acceptance"] = all(e.get("decision") != "ACCEPTED" or load(run / "edits" / rid / "commit.json")["head_after"] in chain
                                   for rid, e in st["edits"].items())
    a["no_unpublished_accepted_files"] = not st["orphan_accepted_folder_blends_not_published"]
    a["recovery_required"] = st["recovery_required"]
    a["w126_edits"] = {rid: {k: e.get(k) for k in ("gate", "decision", "semantic_authorization", "proposal_integrity", "authorization_refusals")}
                       for rid, e in st["edits"].items() if rid.startswith("W126")}
    a["false_accept_display"] = [rid for rid, e in st["edits"].items() if rid.startswith("W126") and e["gate"] == "ACCEPT"
                                 and e.get("proposal_integrity") != "OK"]
    # the head's semantic truth, recomputed here from the fresh readback when it is a W126 accepted edit
    if refs.get("op") == "ACCEPT" and hb.get("request_id", "").startswith("W126"):
        prop = load(run / "edits" / hb["request_id"] / "proposal.json")
        sidecar = load(run / "baseline" / "action_sidecar.json")
        X = E.readback_by_source(fresh, sidecar)
        U = E.readback_by_source(load(run / prop["head_blend_at_proposal"]["readback"]), sidecar)
        t = np.array(prop["constraint"]["target"])
        a["head_hard_pin_residual_max_m_fresh"] = max(float(np.linalg.norm(np.array(X[k]["world"]["lowerarm_l"]["tail"]) - t))
                                                      for k in prop["pin_frames"])
        frames = [r["source_frame"] for r in sidecar["frame_mapping"]]
        import w116_semantic_pin_contract as C
        R = load(run / "run.json")
        rig = C.rig_from(R["rest_json"], sidecar["rig_info"])
        fp = E._norm(XB.derive_footprint(X, U, frames, prop["pin_frames"], S.domain_points(rig, ["lowerarm_l", "upperarm_l"])))
        a["head_footprint_fresh"] = {"frames": fp["frames_changed"], "bones": sorted(fp["channel_frames"]),
                                     "equals_proposal": fp["frames_changed"] == prop["footprint"]["frames_changed"]
                                     and fp["channel_frames"] == prop["footprint"]["channel_frames"],
                                     "inside_pin_frames": set(fp["frames_changed"]) <= set(prop["pin_frames"])}
        sub = E.subframe_report(fresh, load(run / prop["head_blend_at_proposal"]["readback"]), sidecar, prop["constraint"],
                                {"bone": "lowerarm_l", "point": "tail"})
        a["head_subframes_fresh"] = sub
    R = load(run / "run.json")
    a["immutable_ok"] = all(E.sha256_file(run / k) == v for k, v in R["immutable_files"].items())
    man = load(run / "manifest" / "input_manifest.json")
    a["clip_ok"] = sha(man["intake"]["clip_path"]) == CLIP_SHA[src]
    a["ok"] = bool(a["head_verified"]["ok"] and all(a["fresh_readback"].values()) and a["no_ghost_acceptance"]
                   and a["no_unpublished_accepted_files"] and not a["recovery_required"] and not a["false_accept_display"]
                   and a["immutable_ok"] and a["clip_ok"]
                   and (a.get("head_hard_pin_residual_max_m_fresh", 0.0) <= 1e-3)
                   and (a.get("head_footprint_fresh", {}).get("equals_proposal", True))
                   and (a.get("head_footprint_fresh", {}).get("inside_pin_frames", True)))
    return a, st


def _reseal_forged(p):
    from app import experimental_video_action as E
    p = json.loads(json.dumps(p))
    p["decision"] = {"decision": "ACCEPT", "failure_class": "NONE", "reasons": []}
    p["status"] = "PROPOSED_GATE_PASS"
    p["provenance"]["gate_view_digest"] = E._digest(E.gate_view(p))
    return E._sealed(p, "proposal_seal")


def cmd_run(a):
    check_decl(a)
    out = new_out(a.out_dir)
    idx = {"started": time.strftime("%Y-%m-%dT%H:%M:%S"), "sources": {}}
    for src in ("S1", "S3"):
        orig = W125 / "runs" / f"{src}_replay"
        pre = w125_tree_check(orig)
        if not pre["ok"]:
            idx["sources"][src] = {"DATA_BLOCKED": pre}
            continue
        run = BASE / "runs" / f"{src}_replay"
        if run.exists():
            raise SystemExit(f"refusing to overwrite {run}")
        shutil.copytree(orig, run)
        copy_ok = tree(run) == tree(orig)
        hist = {p.relative_to(run).as_posix(): sha(p) for p in (run / "edits").rglob("commit.json")}
        recs, saved = [], {}
        for sid, cmd, kw, exp in scenario(src):
            refs0 = load(run / "lineage" / "refs.json")
            files0 = tree(run)
            rec = {"step": sid, "cmd": cmd, "kw": kw, "expect": exp, "head_before": refs0["head"], "generation_before": refs0.get("generation")}
            if cmd == "status":
                r = cli(ROOT, run, "status")
            elif cmd == "propose":
                rq = req(kw["req"], refs0["head"], src, OFFSETS[kw["req"]])
                dump(out / src / f"{kw['req']}.json", rq)
                r = cli(ROOT, run, "propose", "--request", str(out / src / f"{kw['req']}.json"))
            elif cmd == "commit":
                pp = run / "edits" / kw["rid"] / "proposal.json"
                t = kw.get("tamper")
                if t and kw["rid"] not in saved:
                    saved[kw["rid"]] = pp.read_bytes()
                if t == "decision_only":
                    p = json.loads(saved[kw["rid"]])
                    p["decision"] = {"decision": "ACCEPT", "failure_class": "NONE", "reasons": []}
                    pp.write_text(json.dumps(p, indent=1), encoding="utf-8")
                elif t == "forged_resealed":
                    pp.write_text(json.dumps(_reseal_forged(json.loads(saved[kw["rid"]])), indent=1), encoding="utf-8")
                elif t == "restore":
                    pp.write_bytes(saved[kw["rid"]])
                rec["tamper"] = t
                files0 = tree(run)
                r = cli(ROOT, run, "commit", "--request-id", kw["rid"], "--decision", kw["decision"])
            elif cmd == "review":
                r = cli(ROOT, run, "review", "--request-id", kw["rid"])
            elif cmd == "fbx":
                r = cli(ROOT, run, "fbx", "--state", kw["state"])
            elif cmd == "revert":
                r = cli(ROOT, run, "revert")
            r.pop("_stdout_full", None)
            rec.update(r)
            refs1 = load(run / "lineage" / "refs.json")
            rec.update(head_after=refs1["head"], generation_after=refs1.get("generation"))
            files1 = tree(run)
            rec["preexisting_files_changed"] = sorted(k for k in files0 if files0[k] != files1.get(k)
                                                      and not (k.endswith("txn.lock") or k.endswith(".holder.json")))
            aud, st = audit(run, src, sid)
            rec["audit"] = aud
            rec["status_after"] = {"edits": {k: v for k, v in st["edits"].items()
                                             if k.startswith("W126") or k == LEGACY_UNDECIDED[src] or k in ("R1",)}}
            if cmd == "status":
                rec["status"] = {"legacy": {k: v.get("gate", v.get("state")) for k, v in st["edits"].items() if not k.startswith("W126")},
                                 "historical_accepted": {k: v["semantic_authorization"] for k, v in st["edits"].items()
                                                         if v.get("decision") == "ACCEPTED"}}
            if cmd == "commit" and (run / "edits" / kw["rid"] / "commit.json").exists():
                rec["commit_json"] = load(run / "edits" / kw["rid"] / "commit.json")
            refusals = sorted((run / "edits" / kw.get("rid", "-") / "authorization_refusals").glob("*.json")) if kw.get("rid") else []
            if refusals:
                rec["last_refusal"] = load(refusals[-1])
            if cmd == "propose":
                rec["proposal"] = {k: load(run / "edits" / kw["req"] / "proposal.json").get(k)
                                   for k in ("decision", "status", "candidate", "hard_pin_residual_max_m", "pin_frames")}
                rec["proposal"]["candidate_file"] = (run / "edits" / kw["req"] / "candidate.blend").exists()
                rec["proposal"]["changed_frames"] = (load(run / "edits" / kw["req"] / "proposal.json").get("footprint") or {}).get("frames_changed")
            if cmd == "review" and r["exit"] == 0:
                rec["review"] = load(run / "review" / kw["rid"] / "review_report.json")
            if cmd == "fbx" and r["exit"] == 0:
                fx = run / "fbx" / f"head_{refs1['head'][:12]}"
                rec["fbx"] = {k: v for k, v in load(fx / "fbx_roundtrip_report.json").items() if k not in ("export", "blend_scene", "fbx_scene")}
            if cmd == "revert" and r["exit"] == 0:
                h = refs1["head"]
                rec["revert_report"] = load(run / "reverted" / f"reverted_{h[:12]}_report.json")
            rec["ok"] = step_ok(rec)
            recs.append(rec)
            print(src, sid, rec["exit"], rec["ok"]["ok"], flush=True)
        hist_after = {p.relative_to(run).as_posix(): sha(p) for p in (run / "edits").rglob("commit.json")
                      if not p.relative_to(run).as_posix().startswith("edits/W126")}
        idx["sources"][src] = {"copy_equal_to_w125": copy_ok, "pre_check": pre, "steps": [r["step"] for r in recs],
                               "historical_commit_json_unchanged": hist_after == hist,
                               "steps_file_sha256": dump(out / f"{src}_steps.json", recs)}
    idx["w125_originals_unchanged_after"] = {s: w125_tree_check(W125 / "runs" / f"{s}_replay")["ok"] for s in ("S1", "S3")}
    idx["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    print(dump(out / "run_index.json", idx))


def step_ok(rec):
    e, chk = rec["expect"], {}
    if "exit" in e:
        chk["exit"] = rec["exit"] == e["exit"]
    if "stop" in e:
        chk["stop"] = f"STOP {e['stop']}" in rec.get("stderr", "")
    if "primary" in e:
        chk["primary"] = (rec.get("last_refusal") or {}).get("primary") == e["primary"]
    if e.get("head_unchanged"):
        chk["head_unchanged"] = rec["head_after"] == rec["head_before"] and rec["generation_after"] == rec["generation_before"]
    if e.get("files_unchanged"):
        chk["files_unchanged"] = not rec["preexisting_files_changed"]
    if e.get("generation_plus_1"):
        chk["generation_plus_1"] = rec["generation_after"] == rec["generation_before"] + 1
    if "gate" in e:
        chk["gate"] = (rec.get("proposal") or {}).get("decision", {}).get("decision") == e["gate"]
    if "reasons" in e:
        chk["reasons"] = (rec.get("proposal") or {}).get("decision", {}).get("reasons") == e["reasons"]
    if e.get("candidate_file"):
        chk["candidate_file"] = bool((rec.get("proposal") or {}).get("candidate_file"))
    if "final" in e:
        chk["final"] = (rec.get("commit_json") or {}).get("final_status") == e["final"]
    if e.get("authorized"):
        chk["authorized"] = ((rec.get("commit_json") or {}).get("semantic_authorization") or {}).get("authorized") is True
    if "residual_le_m" in e:
        chk["residual"] = rec["audit"].get("head_hard_pin_residual_max_m_fresh", 1.0) <= e["residual_le_m"]
    if e.get("footprint_in_pin"):
        f = rec["audit"].get("head_footprint_fresh", {})
        chk["footprint"] = bool(f.get("inside_pin_frames") and f.get("equals_proposal") and f.get("frames")
                                and f.get("bones") == ["lowerarm_l", "upperarm_l"])
        chk["subframes"] = bool(rec["audit"].get("head_subframes_fresh", {}).get("changed_intervals_within_pin_frames"))
    if e.get("no_false_accept_display"):
        w = rec["status_after"]["edits"].get(rec["kw"]["rid"], {})
        chk["no_false_accept_display"] = w.get("gate") != "ACCEPT" and w.get("decision", "").startswith("UNDECIDED") \
            and w.get("semantic_authorization") is None
        chk["no_preexisting_file_changed_by_refusal"] = rec["preexisting_files_changed"] == []
    if e.get("native_rate"):
        rv = rec.get("review") or {}
        chk["native_rate"] = bool(rv.get("native_rate") and rv.get("frames_decoded") == rv.get("frames_written"))
    if "roundtrip" in e:
        chk["fbx_characterised"] = (rec.get("fbx") or {}).get("roundtrip_verdict") in ("PASS", "FAIL")
    if e.get("exact_undo"):
        chk["exact_undo"] = bool((rec.get("revert_report") or {}).get("exact_undo"))
    if e.get("legacy_gate"):
        chk["legacy_gate"] = all(str(g).startswith("LEGACY_UNVERIFIED") for g in rec["status"]["legacy"].values())
    if e.get("historical_not_reclassified"):
        chk["historical_not_reclassified"] = all(str(v).startswith("NOT_ESTABLISHED") for v in rec["status"]["historical_accepted"].values())
    chk["audit"] = rec["audit"]["ok"]
    return {"ok": all(chk.values()), "checks": chk}


# ---------------------------------------------------------------------------------------------------- pytest ---
def cmd_pytest(a):
    check_decl(a)
    out = new_out(a.out_dir)
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
    res = {}
    for name, files in (("focused", FOCUSED), ("regression", [f for f in REGRESSION if (ROOT / f).exists()])):
        t0 = time.time()
        r = subprocess.run([PY, "-X", "utf8", "-m", "pytest", "-q", "-p", "no:cacheprovider", *files, f"--junitxml={out / (name + '.xml')}"],
                           cwd=ROOT, env=env, capture_output=True, text=True)
        (out / f"{name}.log").write_text(r.stdout + "\n" + r.stderr, encoding="utf-8")
        res[name] = {"files": files, "exit": r.returncode, "summary": r.stdout.strip().splitlines()[-1] if r.stdout.strip() else "",
                     "seconds": round(time.time() - t0, 1), "junit_sha256": sha(out / f"{name}.xml")}
    dump(out / "pytest.json", res)
    print(json.dumps(res, indent=1))


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
    pf = load(a.preflight)
    idx = load(Path(a.run_dir) / "run_index.json")
    jf = _junit(Path(a.pytest_dir) / "focused.xml")
    jr = _junit(Path(a.pytest_dir) / "regression.xml")
    t126 = {k.split("::")[-1]: v for k, v in jf.items() if "gate_integrity" in k}
    t125 = {k: v for k, v in jf.items() if "action_txn" in k}
    t124 = {k: v for k, v in jf.items() if k.split("::")[0].endswith("test_experimental_video_action")}
    steps = {s: load(Path(a.run_dir) / f"{s}_steps.json") for s in ("S1", "S3") if (Path(a.run_dir) / f"{s}_steps.json").exists()}
    by = lambda s, sid: next((r for r in steps.get(s, []) if r["step"] == sid), None)  # noqa: E731
    neg = {k: v for k, v in t126.items() if "negative_control" in k or "N26" in k or "N30" in k or "N23" in k}
    g = {}
    fx, s1o = pf["oldcode_bypass_fixture"], pf["oldcode_bypass_S1"]
    g["G1_bypass_closed"] = {"oldcode_fixture_published": fx.get("published"), "oldcode_S1_published": s1o.get("published"),
                             "w126_N01": t126.get("test_negative_control_refused_without_any_publication[N01]"),
                             "w126_N02": t126.get("test_negative_control_refused_without_any_publication[N02]"),
                             "real": {s: [by(s, sid)["ok"]["ok"] if by(s, sid) else None for sid in ("s06_T_decision_hand_edited_accept",
                                                                                                    "s07_T_forged_resealed_accept")]
                                      for s in ("S1", "S3")}}
    g["G1_bypass_closed"]["pass"] = bool(fx.get("published") and s1o.get("published") and g["G1_bypass_closed"]["w126_N01"] == "PASS"
                                         and g["G1_bypass_closed"]["w126_N02"] == "PASS"
                                         and all(all(v) for v in g["G1_bypass_closed"]["real"].values()))
    g["G2_negative_matrix"] = {"tests": neg, "recorded": t126.get("test_recorded_rejections_never_move_head"),
                               "failed": sorted(k for k, v in neg.items() if v != "PASS")}
    g["G2_negative_matrix"]["pass"] = not g["G2_negative_matrix"]["failed"] and g["G2_negative_matrix"]["recorded"] == "PASS"
    val = {k: v for k, v in t126.items() if k.startswith("test_V0") or "fixture_proposals" in k}
    g["G3_valid_blender"] = {"tests": val, "pass": bool(val) and all(v == "PASS" for v in val.values())}
    rec = {k: v for k, v in t126.items() if k.startswith("test_R0")}
    g["G4_transactions"] = {"recovery": rec, "w125_suite": {"n": len(t125), "fail": sorted(k for k, v in t125.items() if v != "PASS")},
                            "w124_suite": {"n": len(t124), "fail": sorted(k for k, v in t124.items() if v != "PASS")}}
    g["G4_transactions"]["pass"] = (bool(rec) and all(v == "PASS" for v in rec.values()) and not g["G4_transactions"]["w125_suite"]["fail"]
                                    and not g["G4_transactions"]["w124_suite"]["fail"] and len(t125) > 0 and len(t124) > 0)
    g["G5_real_replay"] = {s: {"steps": {r["step"]: r["ok"]["ok"] for r in steps.get(s, [])},
                               "failed_checks": {r["step"]: {k: v for k, v in r["ok"]["checks"].items() if not v}
                                                 for r in steps.get(s, []) if not r["ok"]["ok"]},
                               "historical_commit_json_unchanged": idx["sources"].get(s, {}).get("historical_commit_json_unchanged")}
                           for s in ("S1", "S3")}
    g["G5_real_replay"]["pass"] = all(steps.get(s) and all(r["ok"]["ok"] for r in steps[s])
                                      and idx["sources"][s].get("historical_commit_json_unchanged") for s in ("S1", "S3"))
    g["G6_native_rate"] = {s: (by(s, "s11_review_V") or {}).get("review") for s in ("S1", "S3")}
    g["G6_native_rate"]["pass"] = all(by(s, "s11_review_V") and by(s, "s11_review_V")["ok"]["ok"] for s in ("S1", "S3"))
    from app import experimental_video_action as E
    g["G7_regression_isolation"] = {"regression": {"n": len(jr), "fail": sorted(k for k, v in jr.items() if v != "PASS")},
                                    "frozen_code_unchanged": {k: v["sha256_lf"] for k, v in E.frozen_code_report().items()} == d["frozen_code_sha256_lf"],
                                    "w125_originals_unchanged": idx.get("w125_originals_unchanged_after")}
    g["G7_regression_isolation"]["pass"] = (not g["G7_regression_isolation"]["regression"]["fail"] and len(jr) > 0
                                            and g["G7_regression_isolation"]["frozen_code_unchanged"]
                                            and all(g["G7_regression_isolation"]["w125_originals_unchanged"].values()))
    data_blocked = any("DATA_BLOCKED" in v for v in idx["sources"].values())
    ghost = any(not r["audit"]["no_ghost_acceptance"] or not r["audit"]["head_verified"]["ok"] for s in steps.values() for r in s)
    P = {k: v["pass"] for k, v in g.items()}
    if data_blocked:
        case = "DATA_BLOCKED / NO CASE"
    elif ghost or not g["G7_regression_isolation"]["frozen_code_unchanged"]:
        case = "D"
    elif all(P.values()):
        case = "A"
    elif all(P[k] for k in ("G1_bypass_closed", "G2_negative_matrix", "G3_valid_blender", "G4_transactions", "G7_regression_isolation")) \
            and any(steps.get(s) and all(r["ok"]["ok"] for r in steps[s]) for s in ("S1", "S3")):
        case = "B"
    else:
        case = "C"
    rep = {"gates": g, "gate_pass": P, "case_by_frozen_rule": case, "case_rule": CASE_RULE, "fbx": FBX_NOTE,
           "fbx_results": {s: (by(s, "s12_fbx_head") or {}).get("fbx") for s in ("S1", "S3")}}
    print(dump(out / "assessment.json", rep))
    print(json.dumps({"gate_pass": P, "case": case}, indent=1))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("preflight", "declare", "run", "pytest", "assess"):
        p = sub.add_parser(name)
        p.add_argument("--out-dir", required=True)
        if name in ("run", "pytest", "assess"):
            p.add_argument("--declaration", required=True)
            p.add_argument("--declaration-sha", required=True)
        if name == "assess":
            p.add_argument("--preflight", required=True)
            p.add_argument("--run-dir", required=True)
            p.add_argument("--pytest-dir", required=True)
    a = ap.parse_args()
    {"preflight": cmd_preflight, "declare": cmd_declare, "run": cmd_run, "pytest": cmd_pytest, "assess": cmd_assess}[a.cmd](a)


if __name__ == "__main__":
    main()
