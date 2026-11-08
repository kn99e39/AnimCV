#!/usr/bin/env python3
"""W125 evidence harness: crash-consistent ACCEPT / REJECT / REVERT of the EXPERIMENTAL opt-in editable-Action slice.

NOT part of the product path.  It drives the user command ``python -m app.cli experimental-video-action ...`` (PYTHONPATH=src)
on SAVED-ARTIFACT REPLAYS of the W124 authentic S1/S3 runs (copies; the W124 originals are only read and re-hashed), injects
deterministic faults / forced terminations at declared boundaries, and after every step audits the run from this separate
process, including a FRESH Blender readback of the head .blend.

  preflight  W124 control hashes (every run file vs the W124 evidence manifest), source clips, and a reproduction of one
             code-inspected W124 failure window on the STARTING-COMMIT code (git archive export; a copy of the S1 run).
  declare    frozen scenario, failure points, expected outcomes, gates and CASE rule + instrument SHA-256 (before outcomes)
  run        execute the declared scenario on S1 and S3
  pytest     focused + bounded regression tests (junit xml)
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
import tarfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
BASE = Path(os.environ.get("W125_BASE", "C:/Users/dna10/animcv-w125"))
W124 = Path(os.environ.get("W124_BASE", "C:/Users/dna10/animcv-w124"))
PY = os.environ.get("W125_PYTHON", "C:/Projects/AnimCV/.venv_build_full/Scripts/python.exe")
BLENDER = os.environ.get("ANIMCV_BLENDER", "C:/Program Files/Blender Foundation/Blender 5.1/blender.exe")
START_COMMIT = "c5d23da7f40f4e8ca904941d80d33426fc1181a0"
W124_MANIFEST_SHA = "507099dc9ef8d720b8f87fd722003e10d20b05db529507e7ecae57e384c1e067"
CLIP_SHA = {"S1": "de8feb9d515daa2bfaeed7167fdcc62028dd3f6450872c4fd81f8fb03dd6359c",
            "S3": "f6b90f2f72e9020dcc261fadb09725913fa31719a8614de26fab372826987f62"}
W124_EDIT = {"S1": {"accepted_state": "986840be1e1dd70f738483cfc18f181f5ed398295121d43d9a44f74ec6c2af11", "frames": [210, 218], "capture": 210,
                    "changed": list(range(211, 219)), "head": "49dbe510f459999bcdf6cfdfd8ef4367a7711b4ab57ca7f099b480a5b8e43eca",
                    "baseline": "1284d6ce91afce30377fd83e94154ded7ce098cef133cccc7dfa11a3f11eda11"},
             "S3": {"accepted_state": "508d72692024", "frames": [41, 49], "capture": 41, "changed": list(range(42, 50)),
                    "head": "c7dc430ba05edf1a6cdca3d9b43ddcb80c23b90ead4c1531151f465b88d68120",
                    "baseline": "07667f65b21d299c6d2a45122ad3c2925e6870c29943e03a79edb9643849472d"}}
ARM = ["lowerarm_l", "upperarm_l"]

# ---- frozen scenario (declared before any S1/S3 outcome) -------------------------------------------------------------
# fault: (point, mode).  mode kill = hang at the boundary, then TerminateProcess from this harness (separate process);
# kill_in_blender = no fault env; the CLI python is terminated while its Blender child is building (orphan Blender left running).
SCENARIO = {
    "S1": [
        ("s01_status_legacy", "status", {}, None, {"refs_kind": "LEGACY_W124"}),
        ("s02_propose_before_adopt", "propose", {"req": "R0"}, None, {"exit": 2, "stop": "BLOCKED_LEGACY_RUN_REQUIRES_ADOPT"}),
        ("s03_adopt", "adopt", {}, None, {"exit": 0, "generation": 2, "head": "@W124_HEAD"}),
        ("s04_propose_R1", "propose", {"req": "R1"}, None, {"exit": 0, "gate": "ACCEPT"}),
        ("s05_accept_R1_killed_after_blend_copy", "commit", {"rid": "R1", "decision": "accept"}, ("after_blend_copy", "kill"),
         {"head_unchanged": True, "pending_published": False}),
        ("s06_recover", "recover", {}, None, {"exit": 0, "outcome": "ROLLED_BACK", "head_unchanged": True}),
        ("s07_accept_R1", "commit", {"rid": "R1", "decision": "accept"}, None, {"exit": 0, "final": "ACCEPTED", "generation": 3}),
        ("s08_propose_R2", "propose", {"req": "R2"}, None, {"exit": 0, "gate": "ACCEPT"}),
        ("s09_reject_R2", "commit", {"rid": "R2", "decision": "reject"}, None, {"exit": 0, "final": "REJECTED", "head_unchanged": True}),
        ("s10_revert_killed_during_blender_build", "revert", {}, ("-", "kill_in_blender"), {"head_unchanged": True, "pending_published": False}),
        ("s11_recover", "recover", {}, None, {"exit": 0, "outcome": "ROLLED_BACK", "head_unchanged": True}),
        ("s12_revert", "revert", {}, None, {"exit": 0, "exact_undo": True, "generation": 4}),
        ("s13_review_R1", "review", {"rid": "R1"}, None, {"exit": 0, "native_rate": True}),
        ("s14_fbx_R1_state", "fbx", {"state": "@R1"}, None, {"exit": 0, "roundtrip": "FAIL (expected, unchanged W107-W109 baseline)"}),
    ],
    "S3": [
        ("s01_status_legacy", "status", {}, None, {"refs_kind": "LEGACY_W124"}),
        ("s03_adopt", "adopt", {}, None, {"exit": 0, "generation": 2, "head": "@W124_HEAD"}),
        ("s04_propose_R1", "propose", {"req": "R1"}, None, {"exit": 0, "gate": "ACCEPT"}),
        ("s05_accept_R1_exit_after_publish", "commit", {"rid": "R1", "decision": "accept"}, ("after_publish", "exit"),
         {"head_moved_to_candidate": True, "pending_published": True}),
        ("s06_recover", "recover", {}, None, {"exit": 0, "outcome": "ROLLED_FORWARD", "final_after": "ACCEPTED", "generation": 3}),
        ("s08_propose_R2", "propose", {"req": "R2"}, None, {"exit": 0, "gate": "ACCEPT"}),
        ("s09_reject_R2", "commit", {"rid": "R2", "decision": "reject"}, None, {"exit": 0, "final": "REJECTED", "head_unchanged": True}),
        ("s10_revert_exception_before_publish", "revert", {}, ("before_publish", "raise"),
         {"exit": 2, "stop": "TRANSACTION_ABORTED_ROLLED_BACK", "head_unchanged": True, "pending": 0}),
        ("s12_revert", "revert", {}, None, {"exit": 0, "exact_undo": True, "generation": 4}),
        ("s13_review_R1", "review", {"rid": "R1"}, None, {"exit": 0, "native_rate": True}),
        ("s14_fbx_R1_state", "fbx", {"state": "@R1"}, None, {"exit": 0, "roundtrip": "FAIL (expected, unchanged W107-W109 baseline)"}),
    ],
}
GATES = {
    "T1_fault_matrix": "every declared ACCEPT/REVERT/REJECT boundary x {exception, os._exit} gives the frozen outcome (pre-publication: "
                       "old head authoritative, staged files quarantined, retry succeeds; post-publication: new verified head, recover "
                       "rolls forward exactly once) - focused real-Blender BaseRig tests all PASS",
    "T2_process_kill_restart": "separate-process forced termination (TerminateProcess) + restart recovery give the frozen outcome, in the "
                               "focused tests AND on the real S1/S3 replays (s05/s10 S1, s05 S3)",
    "T3_concurrency": "competing CLI operations on one parent: at most one publishes, the other fails BLOCKED_RUN_LOCKED or STALE_PARENT; "
                      "no lost update, no waiting/stealing",
    "T4_tamper": "tampered staged .blend refused; tampered refs/journal/head .blend STOP (never repaired); torn log tail and stray temp "
                 "repaired by explicit recover with the bytes quarantined",
    "T5_real_transactions": "S1 and S3 replays: every declared step meets its expected outcome; edited values canon-equal to the W124 "
                            "accepted edit; actual changed frames/bones equal W124's; exact undo bit-equal at integer frames and subframes",
    "T6_independent_audit": "after every real step: head -> state -> .blend -> FRESH Blender readback equal; no accepted commit outside "
                            "the published history; no unpublished file in accepted/; footprint display truth ok; immutable baseline, "
                            "source clip and W124 originals unchanged",
    "T7_native_rate_review": "30 fps A/B video of the recovered accepted edit decodes one frame per Action frame",
    "T9_regression_isolation": "focused (W124 + W125) and bounded regression tests pass; default parser digest unchanged",
}
FBX_NOTE = "FBX round-trip is characterisation only (expected FAIL, unchanged W124 baseline); never a W125 gate."
CASE_RULE = ("DATA_BLOCKED if the W124 S1/S3 artifacts are missing or differ from the W124 evidence manifest before the run.  D if the "
             "W124 originals, frozen code or a declared expectation is changed after outcomes, or T6 shows a ghost/unverified accepted "
             "head.  A if T1-T7 and T9 all PASS (scope: process-level failures on local NTFS / Windows 11; power loss, OS crash, "
             "network or other filesystems are NOT tested and NOT claimed).  B if T1-T4 and T9 PASS and the real path passes on >=1 "
             "source but some T5-T7 element is limited.  C otherwise.")
INSTRUMENT = ["src/app/experimental_video_action.py", "src/app/experimental_action_txn.py", "src/blender/experimental_action_stage.py",
              "src/app/cli.py", "tests/test_experimental_video_action.py", "tests/test_experimental_action_txn.py", "scripts/w125_txn_harness.py",
              "docs/EXPERIMENTAL_VIDEO_ACTION_WORLD_PIN.md"]


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


def w124_tree_check(src):
    """Every file of a W124 run dir vs the W124 evidence manifest (which was written before the W124 Worklog)."""
    man = W124 / "results" / "evidence_manifest.json"
    if sha(man) != W124_MANIFEST_SHA:
        return {"ok": False, "reason": "W124 evidence manifest hash differs"}
    files = load(man)["files"]
    rel_root = f"runs/{src.name}/"
    exp = {k[len(rel_root):]: v for k, v in files.items() if k.startswith(rel_root)}
    got = {str(p.relative_to(src)).replace("\\", "/"): sha(p) for p in src.rglob("*") if p.is_file()}
    diff = sorted(k for k in set(exp) | set(got) if exp.get(k) != got.get(k))
    return {"ok": not diff and bool(exp), "files": len(exp), "differences": diff[:20], "n_differences": len(diff)}


# ------------------------------------------------------------------------------------------------- preflight ---
def cmd_preflight(a):
    out = new_out(a.out_dir)
    rep = {"start_commit": START_COMMIT, "git_head": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip(),
           "w124_manifest_sha256": sha(W124 / "results" / "evidence_manifest.json"), "w124_runs": {}, "clips": {}}
    for s in ("S1", "S3"):
        run = W124 / "runs" / f"{s}_live"
        rep["w124_runs"][s] = w124_tree_check(run)
        man = load(run / "manifest" / "input_manifest.json")
        clip = Path(man["intake"]["clip_path"])
        rep["clips"][s] = {"path": str(clip), "exists": clip.exists(), "sha256": sha(clip) if clip.exists() else None,
                           "expected": CLIP_SHA[s]}
        rep["w124_runs"][s].update(refs=load(run / "lineage" / "refs.json"), refs_sha256=sha(run / "lineage" / "refs.json"),
                                   log_sha256=sha(run / "lineage" / "log.jsonl"), blends_sha256=sha(run / "lineage" / "blends.json"),
                                   baseline_blend_sha256=sha(run / "baseline" / "baseline.blend"),
                                   accepted=[(p.name, sha(p)) for p in sorted((run / "accepted").glob("*.blend"))],
                                   reverted=[(p.name, sha(p)) for p in sorted((run / "reverted").glob("*.blend"))])
    rep["old_failure_windows_code_inspected"] = {
        "W1_commit_head_before_artifacts": "cmd_commit: ps.save() writes refs.json (head = candidate) BEFORE the accepted copy, its SHA check, "
                                           "the registry, commit.json and timeline_final; a crash in between leaves head -> unregistered state, "
                                           "no accepted copy, and a proposal that still looks undecided",
        "W2_refs_non_atomic": "PersistentStore.save: Path.write_text truncates then writes refs.json (torn/empty on kill); the log is appended "
                              "separately afterwards (refs/log divergence)",
        "W3_revert_head_before_blender": "cmd_revert: ps.save() publishes the revert state BEFORE build_state_blend/readback; a Blender failure "
                                         "leaves a head with no .blend (propose then fails on reg[head])",
        "W4_registry_non_atomic": "register_blend: blends.json rewritten in place",
        "W5_no_cross_process_exclusion": "two processes load the same refs; W116 Store.commit compares heads only in memory; last writer wins",
        "W6_state_put_non_atomic": "W116 Store.put writes the state file in place (torn state under a valid id on kill)",
        "W7_accepted_name_by_count": "accepted_NN chosen by counting files: an orphan copy shifts/collides names",
    }
    rep["old_code_reproduction"] = reproduce_old_window(out)
    rep["data_blocked"] = not all(r["ok"] for r in rep["w124_runs"].values()) or not all(
        c["sha256"] == c["expected"] for c in rep["clips"].values())
    print(json.dumps({"data_blocked": rep["data_blocked"], "w124": {s: r["ok"] for s, r in rep["w124_runs"].items()},
                      "old_code_reproduction": rep["old_code_reproduction"]["summary"]}, indent=1))
    dump(out / "preflight.json", rep)


def reproduce_old_window(out):
    """W3 on the STARTING-COMMIT code: a Blender failure during REVERT, on a copy of the authentic S1 run."""
    code = out / "oldcode"
    code.mkdir()
    tarp = out / "oldcode.tar"
    subprocess.run(["git", "archive", "--format=tar", "-o", str(tarp), START_COMMIT, "src", "scripts", "examples"], cwd=ROOT, check=True)
    with tarfile.open(tarp) as t:
        t.extractall(code)
    run = out / "old_S1_copy"
    shutil.copytree(W124 / "runs" / "S1_live", run)
    r = load(run / "run.json")
    os.chmod(run / "run.json", 0o644)
    r["blender"] = str(out / "NO_BLENDER_INJECTED_FAILURE.exe")
    (run / "run.json").write_text(json.dumps(r, indent=1), encoding="utf-8")
    env = dict(os.environ, PYTHONPATH=str(code / "src"))
    head0 = load(run / "lineage" / "refs.json")["head"]
    rv = subprocess.run([PY, "-X", "utf8", "-m", "app.cli", "experimental-video-action", "revert", "--run", str(run)], cwd=code, env=env,
                        capture_output=True, text=True)
    head1 = load(run / "lineage" / "refs.json")["head"]
    reg = load(run / "lineage" / "blends.json")
    pr = subprocess.run([PY, "-X", "utf8", "-m", "app.cli", "experimental-video-action", "status", "--run", str(run)], cwd=code, env=env,
                        capture_output=True, text=True)
    (out / "old_revert_stderr.txt").write_text(rv.stderr, encoding="utf-8")
    res = {"code": f"git archive {START_COMMIT} (W124 code)", "run": "copy of W124 runs/S1_live with run.json blender -> missing exe",
           "revert_exit": rv.returncode, "head_before": head0, "head_after": head1, "head_moved": head1 != head0,
           "new_head_has_blend": head1 in reg, "status_exit": pr.returncode, "stderr_tail": rv.stderr[-400:]}
    res["summary"] = ("REPRODUCED: revert failed in Blender but the head moved to a state with no .blend" if res["head_moved"] and not
                      res["new_head_has_blend"] else "NOT REPRODUCED")
    return res


# --------------------------------------------------------------------------------------------------- declare ---
def cmd_declare(a):
    out = new_out(a.out_dir)
    from app import experimental_action_txn as TX
    d = {"schema": "w125_declaration_v1", "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "outcomes_inspected": False,
         "real_population": {"S1": "saved-artifact replay: copy of W124 runs/S1_live (authentic live S1 clip, segment 79-318, 240 frames)",
                             "S3": "saved-artifact replay: copy of W124 runs/S3_live (authentic live S3 clip, segment 0-153, 154 frames)",
                             "label": "SAVED-ARTIFACT REPLAY of W124 live outputs - NOT a fresh live-perception E2E; engineering regression "
                                      "sources, not new independent sources"},
         "requests": {s: {"R0": "same as R1 (must STOP before adopt)", "R1": f"hold left wrist {W124_EDIT[s]['frames']} capture {W124_EDIT[s]['capture']}",
                          "R2": f"hold {W124_EDIT[s]['frames']} capture {W124_EDIT[s]['capture']} + 3 cm up (user REJECT)"} for s in W124_EDIT},
         "scenario": SCENARIO, "fault_points": list(TX.FAULT_POINTS), "gates": GATES, "case_rule": CASE_RULE, "fbx": FBX_NOTE,
         "parity_expectations": {s: {"R1_candidate_values_canon_equal_w124_accepted": True, "R1_changed_frames": W124_EDIT[s]["changed"],
                                     "R1_changed_bones": sorted(ARM)} for s in W124_EDIT},
         "platform_scope": {"tested": "Windows 11 Pro 10.0.26200, local NTFS, Python 3.11.9, Blender 5.1.0; failures = Python exception, "
                                      "os._exit, TerminateProcess of the CLI process (incl. while its Blender child runs)",
                            "not_tested_not_claimed": "power loss, OS crash, disk-full, network/other filesystems, POSIX (code path exists, untested)"},
         "instrument_sha256_lf": {f: sha_lf(ROOT / f) for f in INSTRUMENT}}
    dump(out / "declaration.json", d)
    print(sha(out / "declaration.json"))


def check_decl(a):
    if sha(a.declaration) != a.declaration_sha:
        raise SystemExit("declaration hash mismatch")
    d = load(a.declaration)
    changed = [f for f, h in d["instrument_sha256_lf"].items() if sha_lf(ROOT / f) != h and f != "docs/EXPERIMENTAL_VIDEO_ACTION_WORLD_PIN.md"]
    if changed:
        raise SystemExit(f"instrument changed after declaration: {changed}")
    return d


# ------------------------------------------------------------------------------------------------------- run ---
def cli(run, args, fault=None, hang=None, wait=True):
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
    for k in ("ANIMCV_EXP_TXN_FAULT", "ANIMCV_EXP_TXN_FAULT_MODE", "ANIMCV_EXP_TXN_HANG_FILE"):
        env.pop(k, None)
    if fault and fault[1] in ("raise", "exit", "hang"):
        env.update(ANIMCV_EXP_TXN_FAULT=fault[0], ANIMCV_EXP_TXN_FAULT_MODE=fault[1])
        if hang:
            env["ANIMCV_EXP_TXN_HANG_FILE"] = str(hang)
    cmd = [PY, "-X", "utf8", "-m", "app.cli", "experimental-video-action", args[0], "--run", str(run), *args[1:]]
    if not wait:
        return subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True), cmd
    t0 = time.time()
    r = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, timeout=1800)
    return {"cmd": cmd, "exit": r.returncode, "stdout": r.stdout, "stderr": r.stderr, "seconds": time.time() - t0}


def fresh_readback(blend, out_json):
    log = Path(str(out_json) + ".log")
    r = subprocess.run([BLENDER, "-b", "-t", "2", "--python-exit-code", "1", "-P", str(ROOT / "src/blender/experimental_action_stage.py"), "--",
                        "readback", "--blend", str(blend), "--report", str(out_json), "--subframes", "0.25,0.5,0.75"],
                       cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
    log.write_text(r.stdout + "\n" + r.stderr, encoding="utf-8")
    return load(out_json) if r.returncode == 0 else None


def audit(run, out, tag, clip):
    """Independent audit from this process (not the CLI): authority, Blender truth, ghost/orphan checks, immutables."""
    from app import experimental_video_action as E
    import w118_transition_edit_base as XB
    import w116_semantic_pin_base as S
    import w116_semantic_pin_contract as C
    A = {"tag": tag}
    refs, raw, kind = E.read_refs(run)
    A.update(refs_kind=kind, head=refs["head"], generation=refs.get("generation"), refs_sha256=hashlib.sha256(raw).hexdigest())
    A["pending"] = [p.name for p, _ in E.pending_journals(run)]
    R = load(run / "run.json")
    A["immutable_unchanged"] = all(sha(run / k) == v for k, v in R["immutable_files"].items())
    A["source_clip_unchanged"] = sha(clip) == CLIP_SHA[tag.split(":")[0]]
    if kind != "V2":
        reg = load(run / "lineage" / "blends.json")
        e = reg[refs["head"]]
        hb = {"blend": e.get("accepted_copy") or e["blend"], "readback": e["readback"], "blend_sha256": e["blend_sha256"]}
        A["head_verified"] = {"ok": None, "problems": ["LEGACY (unsealed)"]}
    else:
        hb = refs["head_blend"]
        A["head_verified"] = E.verify_head(run, refs)
    A["head_blend"], A["head_blend_sha256"] = hb["blend"], sha(run / hb["blend"])
    fr = fresh_readback(run / hb["blend"], out / f"{tag.replace(':', '_')}_fresh_head_readback.json")
    rec_rb = load(run / hb["readback"])
    A["fresh_blender_readback_equals_recorded"] = fr is not None and fr["value_digest"] == rec_rb["value_digest"] and \
        [{k: f[k] for k in ("world", "matrix_world", "stored_wxyz")} for f in fr["frames"]] == \
        [{k: f[k] for k in ("world", "matrix_world", "stored_wxyz")} for f in rec_rb["frames"]] and \
        [{k: f[k] for k in ("world", "matrix_world")} for f in fr["subframes"]] == [{k: f[k] for k in ("world", "matrix_world")} for f in rec_rb["subframes"]]
    A["fresh_readback_value_digest"] = fr["value_digest"] if fr else None
    st = S.Store(run / "lineage")
    chain = [x["state_id"] for x in st.lineage(refs["head"])]
    A["lineage_len"], A["lineage_verifies"] = len(chain), all(x["verifies"] for x in st.lineage(refs["head"]))
    reg = load(run / "lineage" / "blends.json")
    sidecar = load(run / "baseline" / "action_sidecar.json")
    frames = [r["source_frame"] for r in sidecar["frame_mapping"]]
    ghosts, disp = [], {}
    for ed in sorted(p for p in (run / "edits").iterdir() if p.is_dir()):
        cj = ed / "commit.json"
        if cj.exists() != (ed / "timeline_final.png").exists():
            ghosts.append(f"{ed.name}: commit.json/timeline_final mismatch")
        if not cj.exists():
            continue
        cm = load(cj)
        if cm["final_status"] == "ACCEPTED":
            if cm["head_after"] not in chain and cm["head_after"] not in reg:
                ghosts.append(f"{ed.name}: ACCEPTED but not in published history")
            if ed.name.startswith("R"):     # W125 edits: recompute footprint + display truth from the ACCEPTED .blend readback
                prop = load(ed / "proposal.json")
                arb = cm.get("accepted_readback")
                if arb:
                    rig = C.rig_from(R["rest_json"], sidecar["rig_info"])
                    dom = S.domain_points(rig, sorted(ARM))
                    X = E.readback_by_source(load(run / arb), sidecar)
                    U = E.readback_by_source(load(run / prop["head_blend_at_proposal"]["readback"]), sidecar)
                    fp = E._norm(XB.derive_footprint(X, U, frames, prop["pin_frames"], dom))
                    dc = XB.check_display(cm["display_final"], X, U, frames, prop["pin_frames"], prop["candidate"], decision="ACCEPT")
                    disp[ed.name] = {"changed_frames": fp["frames_changed"], "changed_bones": sorted(fp["channel_frames"]),
                                     "equals_proposal": fp["frames_changed"] == prop["footprint"]["frames_changed"] and
                                     fp["channel_frames"] == prop["footprint"]["channel_frames"], "display_final_check_ok": dc["ok"]}
    A["ghost_or_mismatch"] = ghosts
    A["footprint_truth"] = disp
    published_acc = {str(Path(r["accepted_copy"])) for r in reg.values() if r.get("accepted_copy")}
    published_acc |= {str(Path(x.get("accepted_copy"))) for r in reg.values() for x in r.get("additional_records", []) if x.get("accepted_copy")}
    if kind == "V2":
        published_acc.add(str(Path(refs["head_blend"]["blend"])))
    A["unpublished_files_in_accepted"] = [p.name for p in sorted((run / "accepted").glob("*.blend")) if str(p.relative_to(run)) not in published_acc] \
        if (run / "accepted").exists() else []
    A["quarantine_files"] = sorted(str(p.relative_to(run)) for p in (run / "quarantine").rglob("*") if p.is_file()) if (run / "quarantine").exists() else []
    A["ok"] = bool(A["immutable_unchanged"] and A["source_clip_unchanged"] and A["fresh_blender_readback_equals_recorded"] and A["lineage_verifies"]
               and not ghosts and all(v["equals_proposal"] and v["display_final_check_ok"] for v in disp.values())
               and (kind != "V2" or (A["head_verified"]["ok"] and (A["pending"] or not A["unpublished_files_in_accepted"]))))
    return A


def make_requests(s, head, out):
    e = W124_EDIT[s]
    base = {"schema": "animcv_exp_world_pin_request_v1", "parent_state_id": head, "type": "WORLD_POSITION_PIN", "semantic_joint": "left_wrist",
            "coordinate_frame": "SCENE_WORLD", "unit": "m", "frames": e["frames"], "authorized_channels": list(ARM)}
    reqs = {"R0": dict(base, request_id="R0", target={"kind": "CAPTURED_FROM_EVALUATED_STATE", "capture_frame": e["capture"]}),
            "R1": dict(base, request_id="R1", target={"kind": "CAPTURED_FROM_EVALUATED_STATE", "capture_frame": e["capture"]}),
            "R2": dict(base, request_id="R2", target={"kind": "CAPTURED_PLUS_DECLARED_OFFSET", "capture_frame": e["capture"], "offset_m": [0.0, 0.0, 0.03]})}
    paths = {}
    for k, r in reqs.items():
        paths[k] = out / "requests" / f"{s}_{k}.json"
        dump(paths[k], r)
    return paths


def run_step(run, out, s, name, op, args, fault, req_paths, state):
    from app import experimental_video_action as E
    rec = {"step": name, "op": op, "fault": fault, "head_before": E.read_refs(run)[0]["head"]}
    if op == "propose":
        if args["req"] in ("R2",):     # R2 is authored against the CURRENT head (the user copies it from `status`)
            r = load(req_paths["R2"])
            r["parent_state_id"] = rec["head_before"]
            dump(req_paths["R2"], r)
        cargs = ["propose", "--request", str(req_paths[args["req"]])]
    elif op == "commit":
        cargs = ["commit", "--request-id", args["rid"], "--decision", args["decision"]]
    elif op == "review":
        cargs = ["review", "--request-id", args["rid"]]
    elif op == "fbx":
        cargs = ["fbx", "--state", state["R1_candidate"]]
    else:
        cargs = [op]
    if fault and fault[1] in ("kill", "kill_in_blender"):
        hang = out / f"{s}_{name}_hang"
        pr, cmd = cli(run, cargs, fault=(fault[0], "hang") if fault[1] == "kill" else None, hang=hang, wait=False)
        t0 = time.time()
        if fault[1] == "kill":
            while not Path(str(hang) + ".reached").exists() and time.time() - t0 < 900 and pr.poll() is None:
                time.sleep(0.05)
            rec["reached_boundary"] = Path(str(hang) + ".reached").exists()
        else:   # wait until the revert's Blender build has started in its txn staging dir, then kill ONLY the CLI python
            pat = None
            while time.time() - t0 < 900 and pr.poll() is None:
                hits = list((run / "lineage" / "txn").glob("*.stage/*_provenance.json")) if (run / "lineage" / "txn").exists() else []
                if hits:
                    pat = hits[0]
                    break
                time.sleep(0.02)
            time.sleep(0.5)
            rec["blender_build_started_marker"] = str(pat) if pat else None
            rec["stage_blend_existed_at_kill"] = bool(pat) and (pat.parent / pat.name.replace("_provenance.json", ".blend")).exists()
        pr.kill()                      # TerminateProcess: no Python cleanup, lock released by the OS
        so, se = pr.communicate(timeout=120)
        rec.update(cmd=cmd, exit=pr.returncode, stdout=so, stderr=se, killed=True)
        if fault[1] == "kill_in_blender" and rec.get("blender_build_started_marker"):
            rec["orphan_stage_dir"] = str(Path(rec["blender_build_started_marker"]).parent)
    else:
        rec.update(cli(run, cargs, fault=fault))
    # first command after a kill may meet the OS's asynchronous lock release: status is lock-free, recover retries on LOCKED only
    st = cli(run, ["status"])
    rec["status_exit"] = st["exit"]
    rec["status"] = json.loads(st["stdout"]) if st["exit"] == 0 else st["stderr"]
    rec["head_after"] = E.read_refs(run)[0]["head"]
    return rec


def cmd_run(a):
    d = check_decl(a)
    out = new_out(a.out_dir)
    from app import experimental_video_action as E
    pre = load(a.preflight)
    if pre["data_blocked"]:
        dump(out / "DATA_BLOCKED.json", pre)
        raise SystemExit("DATA BLOCKED")
    index = {"declaration_sha256": a.declaration_sha, "sources": {}}
    for s in SCENARIO:
        src = W124 / "runs" / f"{s}_live"
        run = BASE / "runs" / f"{s}_replay"
        if run.exists():
            raise SystemExit(f"refusing to overwrite {run}")
        shutil.copytree(src, run)
        copy_check = {"files_equal_w124": all(sha(p) == sha(run / p.relative_to(src)) for p in src.rglob("*") if p.is_file())}
        clip = Path(load(run / "manifest" / "input_manifest.json")["intake"]["clip_path"])
        req_paths = make_requests(s, W124_EDIT[s]["head"], out)
        state, steps = {}, []
        steps.append({"step": "s00_copy", "copy_check": copy_check, "audit": audit(run, out, f"{s}:s00", clip)})
        for name, op, args, fault, expect in d["scenario"][s]:
            fault = tuple(fault) if fault else None
            rec = run_step(run, out, s, name, op, args, fault, req_paths, state)
            if op == "recover" and rec["exit"] == 2 and "BLOCKED_RUN_LOCKED" in rec["stderr"]:
                rec["retry_after_os_lock_release"] = rec["stderr"]
                time.sleep(1.0)
                rec.update(cli(run, ["recover"]))
            if name == "s04_propose_R1":
                state["R1_candidate"] = load(run / "edits" / "R1" / "proposal.json").get("candidate")
            if fault and fault[1] == "kill_in_blender" and rec.get("orphan_stage_dir"):
                state["orphan_stage_dir"] = rec["orphan_stage_dir"]
            if name == "s11_recover" and state.get("orphan_stage_dir"):
                sd = Path(state["orphan_stage_dir"])
                t0 = time.time()
                while time.time() - t0 < 120 and not list(sd.glob("*_build.json")):
                    time.sleep(0.2)
                rec["orphan_blender_late_output"] = sorted(p.name for p in sd.iterdir())
                rec["status_after_orphan"] = json.loads(cli(run, ["status"])["stdout"])
            rec["expect"] = expect
            rec["audit"] = audit(run, out, f"{s}:{name}", clip)
            steps.append(rec)
            dump(out / f"{s}_steps.json", steps)
        index["sources"][s] = {"run": str(run), "steps": f"{s}_steps.json"}
        index["sources"][s]["parity"] = parity(run, s, state)
    index["w124_originals_after"] = {s: w124_tree_check(W124 / "runs" / f"{s}_live") for s in SCENARIO}
    dump(out / "run_index.json", index)
    print(json.dumps({s: v["parity"] for s, v in index["sources"].items()}, indent=1))


def parity(run, s, state):
    import w116_semantic_pin_base as S
    w = W124 / "runs" / f"{s}_live" / "lineage" / "states"
    wid = [p for p in w.glob("*.json") if p.stem.startswith(W124_EDIT[s]["accepted_state"][:12])][0]
    old = json.loads(wid.read_text())
    new = json.loads((run / "lineage" / "states" / f"{state['R1_candidate']}.json").read_text())
    prop = load(run / "edits" / "R1" / "proposal.json")
    w_reg = load(W124 / "runs" / f"{s}_live" / "lineage" / "blends.json")[old["state_id"]]
    return {"R1_values_canon_equal_w124_accepted": S.canon(old["values"]) == S.canon(new["values"]),
            "R1_value_digest": prop["candidate_blend"]["value_digest"], "w124_accepted_value_digest": w_reg["value_digest"],
            "value_digest_equal": prop["candidate_blend"]["value_digest"] == w_reg["value_digest"],
            "R1_changed_frames": prop["footprint"]["frames_changed"], "R1_changed_bones": sorted(prop["footprint"]["channel_frames"]),
            "R1_hard_pin_residual_max_m": prop["hard_pin_residual_max_m"], "w124_accepted_state": old["state_id"], "R1_candidate": new["state_id"]}


# ---------------------------------------------------------------------------------------------------- pytest ---
REGRESSION = ["tests/test_cli.py", "tests/test_gui_app.py", "tests/test_blender_executor.py", "tests/test_motion_graph.py",
              "tests/test_keyframe_collapse.py", "tests/test_w116_semantic_pin.py", "tests/test_w118_transition_edit.py",
              "tests/test_w105_rig_utility.py", "tests/test_fk_solver.py", "tests/test_canonical_pose_parity.py", "tests/test_retarget_solver.py",
              "tests/test_w121_independent_transfer.py", "tests/test_w122_realtime_utility.py", "tests/test_render_blender_animation_video.py",
              "tests/test_w117_temporal_pin.py", "tests/test_w120_factor_attribution.py"]


def cmd_pytest(a):
    check_decl(a)
    out = new_out(a.out_dir)
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
    res = {}
    for name, files in (("focused", ["tests/test_experimental_video_action.py", "tests/test_experimental_action_txn.py"]),
                        ("regression", [f for f in REGRESSION if (ROOT / f).exists()])):
        t0 = time.time()
        r = subprocess.run([PY, "-X", "utf8", "-m", "pytest", "-q", "-p", "no:cacheprovider", *files, f"--junitxml={out / (name + '.xml')}"],
                           cwd=ROOT, env=env, capture_output=True, text=True)
        (out / f"{name}.log").write_text(r.stdout + "\n" + r.stderr, encoding="utf-8")
        res[name] = {"files": files, "exit": r.returncode, "summary": r.stdout.strip().splitlines()[-1] if r.stdout.strip() else "", "seconds": time.time() - t0}
    dump(out / "pytest.json", res)
    print(json.dumps(res, indent=1))


# ---------------------------------------------------------------------------------------------------- assess ---
def _junit(p):
    import xml.etree.ElementTree as ET
    r = ET.parse(p).getroot()
    ts = r if r.tag == "testsuite" else r.find("testsuite")
    cases = {}
    for tc in ts.iter("testcase"):
        k = tc.get("classname", "") + "::" + tc.get("name")
        cases[k] = "FAIL" if tc.find("failure") is not None or tc.find("error") is not None else ("SKIP" if tc.find("skipped") is not None else "PASS")
    return cases


def step_ok(rec):
    e, st = rec["expect"], rec.get("status") if isinstance(rec.get("status"), dict) else {}
    chk = {}
    if "exit" in e:
        chk["exit"] = rec["exit"] == e["exit"]
    if "stop" in e:
        chk["stop"] = e["stop"] in rec.get("stderr", "")
    if "refs_kind" in e:
        chk["refs_kind"] = st.get("refs_kind") == e["refs_kind"]
    if e.get("head_unchanged"):
        chk["head_unchanged"] = rec["head_after"] == rec["head_before"]
    if e.get("head_moved_to_candidate"):
        chk["head_moved"] = rec["head_after"] != rec["head_before"]
    if "generation" in e:
        chk["generation"] = st.get("generation") == e["generation"]
    if "pending_published" in e:
        pt = st.get("pending_transactions") or []
        chk["pending_published"] = len(pt) == 1 and pt[0].get("published") == e["pending_published"] and st.get("recovery_required") is True
    if "pending" in e:
        chk["pending"] = len(st.get("pending_transactions") or []) == e["pending"]
    if "outcome" in e:
        chk["outcome"] = e["outcome"] in rec.get("stdout", "")
    if "gate" in e:
        chk["gate"] = f'"decision": "{e["gate"]}"' in rec.get("stdout", "")
    if "final" in e:
        chk["final"] = f'"final_status": "{e["final"]}"' in rec.get("stdout", "")
    if "final_after" in e:
        chk["final_after"] = (st.get("edits") or {}).get("R1", {}).get("decision") == e["final_after"]
    if e.get("exact_undo"):
        chk["exact_undo"] = '"exact_undo": true' in rec.get("stdout", "")
    if e.get("native_rate"):
        chk["native_rate"] = '"native_rate": true' in rec.get("stdout", "")
    if e.get("head") == "@W124_HEAD":
        chk["head"] = st.get("head") in {v["head"] for v in W124_EDIT.values()}
    return {"ok": all(chk.values()) and rec["audit"]["ok"], "checks": chk, "audit_ok": rec["audit"]["ok"]}


def cmd_assess(a):
    d = check_decl(a)
    out = new_out(a.out_dir)
    idx = load(Path(a.run_dir) / "run_index.json")
    py = load(Path(a.pytest_dir) / "pytest.json")
    foc = _junit(Path(a.pytest_dir) / "focused.xml")
    reg = _junit(Path(a.pytest_dir) / "regression.xml")
    G, per = {}, {}
    sel = lambda pat: {k: v for k, v in foc.items() if pat in k}  # noqa: E731
    allpass = lambda dct: bool(dct) and all(v == "PASS" for v in dct.values())  # noqa: E731
    G["T1_fault_matrix"] = allpass(sel("fault_matrix")) and allpass(sel("interrupted_explicit_reject"))
    for s, v in idx["sources"].items():
        steps = load(Path(a.run_dir) / v["steps"])
        per[s] = {r["step"]: step_ok(r) for r in steps if "expect" in r}
        per[s]["s00_copy"] = {"ok": steps[0]["copy_check"]["files_equal_w124"] and steps[0]["audit"]["ok"]}
    real_kill = all(per[s][k]["ok"] for s in per for k in per[s] if "killed" in k or "exit_after" in k or k.endswith("recover"))
    G["T2_process_kill_restart"] = allpass(sel("forced_termination")) and allpass(sel("lock_is_exclusive")) and real_kill
    G["T3_concurrency"] = allpass(sel("competing")) and allpass(sel("simultaneous"))
    G["T4_tamper"] = allpass(sel("tampered")) and allpass(sel("interrupted_propose")) and allpass(sel("legacy"))
    par = {s: v["parity"] for s, v in idx["sources"].items()}
    par_ok = {s: p["R1_values_canon_equal_w124_accepted"] and p["value_digest_equal"] and p["R1_changed_frames"] == W124_EDIT[s]["changed"]
              and p["R1_changed_bones"] == sorted(ARM) for s, p in par.items()}
    src_ok = {s: all(x["ok"] for k, x in per[s].items() if not k.startswith(("s13", "s14"))) and par_ok[s] for s in per}
    G["T5_real_transactions"] = all(src_ok.values())
    G["T6_independent_audit"] = all(x.get("audit_ok", x["ok"]) for s in per for x in per[s].values()) and \
        all(v["ok"] for v in idx["w124_originals_after"].values())
    G["T7_native_rate_review"] = all(per[s].get("s13_review_R1", {}).get("ok") for s in per)
    G["T9_regression_isolation"] = py["focused"]["exit"] == 0 and py["regression"]["exit"] == 0 and \
        "FAIL" not in set(foc.values()) | set(reg.values())
    contamination = not all(v["ok"] for v in idx["w124_originals_after"].values())
    if contamination:
        case = "D"
    elif all(G.values()):
        case = "A"
    elif G["T1_fault_matrix"] and G["T2_process_kill_restart"] and G["T3_concurrency"] and G["T4_tamper"] and G["T9_regression_isolation"] \
            and any(src_ok.values()):
        case = "B"
    else:
        case = "C"
    rep = {"gates": G, "per_step": per, "parity": par, "parity_ok": par_ok, "source_ok": src_ok, "case": case, "case_rule": d["case_rule"],
           "focused_counts": {k: sum(1 for v in foc.values() if v == k) for k in ("PASS", "FAIL", "SKIP")},
           "regression_counts": {k: sum(1 for v in reg.values() if v == k) for k in ("PASS", "FAIL", "SKIP")}}
    dump(out / "assessment.json", rep)
    print(json.dumps({k: rep[k] for k in ("gates", "case", "parity_ok", "focused_counts", "regression_counts")}, indent=1))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for n in ("preflight", "declare", "run", "pytest", "assess"):
        p = sub.add_parser(n)
        p.add_argument("--out-dir", required=True)
        if n in ("run", "pytest", "assess"):
            p.add_argument("--declaration", required=True)
            p.add_argument("--declaration-sha", required=True)
        if n == "run":
            p.add_argument("--preflight", required=True)
        if n == "assess":
            p.add_argument("--run-dir", required=True)
            p.add_argument("--pytest-dir", required=True)
    a = ap.parse_args()
    {"preflight": cmd_preflight, "declare": cmd_declare, "run": cmd_run, "pytest": cmd_pytest, "assess": cmd_assess}[a.cmd](a)


if __name__ == "__main__":
    main()
