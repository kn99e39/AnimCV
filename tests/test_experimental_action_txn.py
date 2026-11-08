"""Batch 125 focused tests: crash-consistent ACCEPT / REJECT / REVERT transactions of the EXPERIMENTAL opt-in slice.

Test kinds (reported separately in the W125 Worklog):
  * unit              - pure Python primitives (atomic write, torn-log detection, seals); no Blender.
  * process           - separate OS processes: lock exclusion, forced termination (TerminateProcess / kill), ``os._exit``.
  * blender           - REAL Blender 5.1 on the REAL W105 BaseRig: a small SYNTHETIC-VALUE run (8 frames, not video, not a
                        quality claim) built with the slice's own build/readback stage, then driven through the user CLI
                        (``python -m app.cli experimental-video-action ...``) with deterministic fault injection.
Expected outcomes were frozen before the real S1/S3 replay (W125 declaration); they are not relaxed after a failure.
"""
from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys
import time
from argparse import Namespace
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from app import experimental_video_action as E  # noqa: E402
from app import experimental_action_txn as TX  # noqa: E402
import w116_semantic_pin_base as S  # noqa: E402
import w116_semantic_pin_contract as C  # noqa: E402

BLENDER = os.environ.get("ANIMCV_BLENDER", "C:/Program Files/Blender Foundation/Blender 5.1/blender.exe")
HAS_BLENDER = Path(BLENDER).exists()
needs_blender = pytest.mark.skipif(not HAS_BLENDER, reason="Blender 5.1 not available")
ENV = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
ENV.pop(TX.FAULT_ENV, None)


# ------------------------------------------------------------------------------------------------- unit ---
def test_atomic_write_replaces_and_leaves_no_temp(tmp_path):
    p = tmp_path / "a.json"
    TX.atomic_dump(p, {"x": 1})
    TX.atomic_dump(p, {"x": 2})
    assert json.loads(p.read_text()) == {"x": 2}
    assert [q.name for q in tmp_path.iterdir()] == ["a.json"]


def test_torn_log_tail_detected(tmp_path):
    p = tmp_path / "log.jsonl"
    TX.append_line_durable(p, json.dumps({"a": 1}))
    with open(p, "ab") as f:
        f.write(b'{"b": 2, "c"')
    recs, torn, bad = TX.read_jsonl(p)
    assert recs == [{"a": 1}] and torn == b'{"b": 2, "c"' and bad == 0


def test_refs_seal_detects_tamper_and_corruption(tmp_path):
    run = tmp_path / "run"
    (run / "lineage").mkdir(parents=True)
    refs = E.make_refs("a" * 64, 3, {"state_id": "a" * 64}, "t1", None, "INIT")
    TX.atomic_dump(run / "lineage" / "refs.json", refs)
    assert E.read_refs(run)[2] == "V2"
    TX.atomic_dump(run / "lineage" / "refs.json", dict(refs, head="b" * 64))
    with pytest.raises(E.SliceStop) as ei:
        E.read_refs(run)
    assert ei.value.code == "BLOCKED_REFS_TAMPERED"
    (run / "lineage" / "refs.json").write_bytes(b'{"head": "aaa')
    with pytest.raises(E.SliceStop) as ei:
        E.read_refs(run)
    assert ei.value.code == "BLOCKED_REFS_CORRUPT"
    TX.atomic_dump(run / "lineage" / "refs.json", {"head": "c" * 64})
    assert E.read_refs(run)[2] == "LEGACY_W124"


def test_fault_points_are_declared_and_inert_without_env(monkeypatch):
    monkeypatch.delenv(TX.FAULT_ENV, raising=False)
    for p in TX.FAULT_POINTS:
        TX.fault(p)                      # no-op
    monkeypatch.setenv(TX.FAULT_ENV, "before_publish")
    with pytest.raises(TX.InjectedFault):
        TX.fault("before_publish")


# -------------------------------------------------------------------------------------------- process ---
LOCK_HOLDER = """
import sys, time, pathlib
sys.path.insert(0, sys.argv[1])
from app import experimental_action_txn as TX
class Stop(RuntimeError):
    def __init__(self, code, detail=''):
        super().__init__(code); self.code = code
with TX.RunLock(sys.argv[2], 'holder', Stop):
    pathlib.Path(sys.argv[3]).write_text('held')
    time.sleep(120)
"""


def _wait(path, timeout=60):
    t0 = time.time()
    while not Path(path).exists():
        if time.time() - t0 > timeout:
            raise AssertionError(f"timeout waiting for {path}")
        time.sleep(0.05)


def test_run_lock_is_exclusive_across_processes_and_released_by_os_on_kill(tmp_path):
    lin = tmp_path / "lineage"
    lin.mkdir()
    flag = tmp_path / "held.flag"
    pr = subprocess.Popen([sys.executable, "-c", LOCK_HOLDER, str(ROOT / "src"), str(lin), str(flag)])
    try:
        _wait(flag)
        with pytest.raises(E.SliceStop) as ei:
            with TX.RunLock(lin, "second", E.SliceStop):
                pass
        assert ei.value.code == "BLOCKED_RUN_LOCKED"
    finally:
        pr.kill()                                      # forced termination: no cleanup code runs in the holder
        pr.wait(30)
    # The OS releases the dead holder's lock by itself (Windows: asynchronously, "depending on system resources").  The
    # lock stays NON-blocking; this test only measures how long the OS takes (bounded at 10 s), it never steals the lock.
    t0, delay = time.time(), None
    while delay is None:
        try:
            with TX.RunLock(lin, "after-kill", E.SliceStop):
                delay = time.time() - t0
        except E.SliceStop as e:
            assert e.code == "BLOCKED_RUN_LOCKED" and time.time() - t0 < 10.0, "OS did not release the killed holder's lock"
            time.sleep(0.05)
    print(f"OS lock release after kill: {delay:.3f} s")


# ------------------------------------------------------------------------ real Blender synthetic-value run ---
def _q(axis, deg):
    return S.q_axis_angle(axis, math.radians(deg)).tolist()


FRAMES = list(range(10, 18))


def _fixture_state():
    vals = {c: [[0.0, 0.0, 0.0, 1.0] for _ in FRAMES] for c in C.REAL_CH}
    vals["upperarm_l"] = [_q([0, 0, 1], 10 + 6 * i) for i in range(len(FRAMES))]
    vals["lowerarm_l"] = [_q([0, 0, 1], 25 + 5 * i) for i in range(len(FRAMES))]
    return S.make_state(None, "BaseRig", FRAMES, C.REAL_CH, vals, {"kind": "ROOT_SYNTHETIC_TEST_FIXTURE"},
                        {"kind": "SYNTHETIC_TEST_FIXTURE (not video)"})


def _req(rid, parent, frames, capture, offset=None):
    t = {"kind": "CAPTURED_FROM_EVALUATED_STATE", "capture_frame": capture} if offset is None else \
        {"kind": "CAPTURED_PLUS_DECLARED_OFFSET", "capture_frame": capture, "offset_m": offset}
    return {"schema": "animcv_exp_world_pin_request_v1", "request_id": rid, "parent_state_id": parent, "type": "WORLD_POSITION_PIN",
            "semantic_joint": "left_wrist", "coordinate_frame": "SCENE_WORLD", "unit": "m", "frames": frames, "target": t,
            "authorized_channels": ["lowerarm_l", "upperarm_l"]}


def build_fixture_run(run):
    """Same lineage/sidecar/binding steps as ``build`` after perception, on synthetic values (labelled)."""
    import numpy as np
    run.mkdir(parents=True)
    rest_json, rig_fbx = E.PROJECT_ROOT / E.RIG_FILES["rest"][0], E.PROJECT_ROOT / E.RIG_FILES["fbx"][0]
    v0 = _fixture_state()
    man = {"schema": E.SLICE_ID + ":input_manifest", "intake": {"clip_path": str(run / "NO_CLIP_SYNTHETIC_FIXTURE"), "clip_sha256": "0" * 64,
                                                                "frame_size": [64, 48]},
           "perception": {"mode": "SYNTHETIC_TEST_FIXTURE", "label": "synthetic values on the real BaseRig - not video"}}
    man_sha = E.dump(run / "manifest" / "input_manifest.json", man)
    mapping = [{"source_frame": k, "source_time_s": k / 30.0, "action_frame": k - FRAMES[0] + 1,
                "action_time_s_at_scene_fps": (k - FRAMES[0]) / 30.0} for k in FRAMES]
    rec, b, rb = E.build_state_blend(run, BLENDER, v0, "baseline", "baseline", {"fixture": True}, rest_json, rig_fbx)
    rig = C.rig_from(rest_json, b["rig"])
    ev = E.readback_by_source(rb, {"frame_mapping": mapping})
    ok, failed = {}, {}
    for j, cands in C.REAL_BINDING_CANDIDATES.items():
        r = S.verify_binding(j, cands, "BaseRig", b["rig"]["rest_snapshot_digest"], rb["rest"], [ev[k] for k in FRAMES], v0["channels"], rig)
        (ok.__setitem__(r["binding_id"], r) if r["status"] == "BOUND" else failed.__setitem__(j, r))
    E.dump(run / "baseline" / "bindings.json", {"bound": ok, "failed": failed})
    sc_sha = E.dump(run / "baseline" / "action_sidecar.json", {"frame_mapping": mapping, "rig_info": b["rig"], "source": {"container_fps": 30.0}})
    E.init_lineage(run, v0, rec)
    E.dump(run / "run.json", {"schema": E.SLICE_ID + ":run", "input_manifest_sha256": man_sha, "baseline_state_id": v0["state_id"],
                              "immutable_files": {"baseline/baseline.blend": rec["blend_sha256"], "baseline/action_sidecar.json": sc_sha,
                                                  "manifest/input_manifest.json": man_sha},
                              "blender": BLENDER, "rig_fbx": str(rig_fbx), "rest_json": str(rest_json)})
    reqs = {"P1": _req("P1", v0["state_id"], [12, 15], 12), "P2": _req("P2", v0["state_id"], [13, 16], 13),
            "P3": _req("P3", v0["state_id"], [12, 15], 12), "REACH": _req("REACH", v0["state_id"], [12, 15], 12, [0.0, 0.0, 1.0])}
    for rid, rq in reqs.items():
        E.dump(run / "requests" / f"{rid}.json", rq)
        E.cmd_propose(Namespace(run=str(run), request=str(run / "requests" / f"{rid}.json")))
    return v0


@pytest.fixture(scope="session")
def base_run(tmp_path_factory):
    if not HAS_BLENDER:
        pytest.skip("Blender 5.1 not available")
    run = tmp_path_factory.mktemp("w125") / "base"
    v0 = build_fixture_run(run)
    return run, v0


@pytest.fixture
def run(base_run, tmp_path):
    src, v0 = base_run
    dst = tmp_path / "run"
    shutil.copytree(src, dst)
    return dst, v0


def cli(run, *args, fault=None, mode="raise", hang=None, wait=True):
    env = dict(ENV)
    if fault:
        env[TX.FAULT_ENV], env[TX.FAULT_MODE_ENV] = fault, mode
        if hang:
            env[TX.HANG_FILE_ENV] = str(hang)
    cmd = [sys.executable, "-X", "utf8", "-m", "app.cli", "experimental-video-action", args[0], "--run", str(run), *args[1:]]
    if not wait:
        return subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, timeout=600)


def cli_after_kill(run, *args, timeout_s=10.0):
    """First command after a hard kill: retry ONLY on BLOCKED_RUN_LOCKED while the OS releases the dead process's lock."""
    t0 = time.time()
    while True:
        r = cli(run, *args)
        if not (r.returncode == 2 and "BLOCKED_RUN_LOCKED" in r.stderr) or time.time() - t0 > timeout_s:
            return r
        time.sleep(0.1)


def status(run):
    r = cli(run, "status")
    assert r.returncode == 0, r.stderr
    return json.loads(r.stdout)


def head(run):
    return E.read_refs(run)[0]


def assert_consistent(run, v0):
    """Independent (new-process view) check: head -> state -> .blend -> readback; no ghost acceptance; baseline intact."""
    st = status(run)
    assert st["refs_kind"] == "V2" and st["head_verified"]["ok"], st["head_verified"]
    assert not st["recovery_required"] and not st["pending_transactions"]
    assert not st["orphan_accepted_folder_blends_not_published"]
    chain = [x["state_id"] for x in st["lineage"]]
    for rid, e in st["edits"].items():
        cj = run / "edits" / rid / "commit.json"
        assert cj.exists() == (run / "edits" / rid / "timeline_final.png").exists()
        if e["decision"] == "ACCEPTED":
            assert load(cj)["head_after"] in chain, (rid, chain)       # an accepted edit is in the published history
        else:
            assert e["candidate"] not in chain or e["decision"] == "ACCEPTED"
    assert E.sha256_file(run / "baseline" / "baseline.blend") == load(run / "run.json")["immutable_files"]["baseline/baseline.blend"]
    assert chain[-1] == v0["state_id"]
    return st


def load(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


@needs_blender
def test_normal_accept_reject_gate_reject_stale_and_exact_revert(run):
    run, v0 = run
    st0 = status(run)
    assert st0["edits"]["P1"]["decision"].startswith("UNDECIDED") and st0["generation"] == 0
    r = cli(run, "commit", "--request-id", "REACH", "--decision", "accept")
    assert r.returncode == 0 and head(run)["head"] == v0["state_id"]
    assert load(run / "edits/REACH/commit.json")["store_record"]["reasons"][0] == "GATE_REJECTED_CANNOT_ACCEPT"
    r = cli(run, "commit", "--request-id", "P3", "--decision", "reject")
    assert r.returncode == 0 and head(run)["head"] == v0["state_id"] and head(run)["generation"] == 0
    r = cli(run, "commit", "--request-id", "P1", "--decision", "accept")
    assert r.returncode == 0, r.stderr
    h = head(run)
    p1 = load(run / "edits/P1/proposal.json")
    assert h["head"] == p1["candidate"] and h["generation"] == 1 and h["op"] == "ACCEPT"
    assert h["head_blend"]["blend"].startswith("accepted") and E.sha256_file(run / h["head_blend"]["blend"]) == p1["candidate_blend"]["blend_sha256"]
    cm = load(run / "edits/P1/commit.json")
    assert cm["final_status"] == "ACCEPTED" and cm["publication_checks"]["ok"]
    assert cm["publication_checks"]["actual_changed_frames"] == p1["footprint"]["frames_changed"] != []
    assert cm["publication_checks"]["actual_changed_bones"] == ["lowerarm_l", "upperarm_l"]
    r = cli(run, "commit", "--request-id", "P2", "--decision", "accept")     # proposed against v0: stale now
    assert r.returncode == 0 and load(run / "edits/P2/commit.json")["store_record"]["reasons"] == ["STALE_PARENT"]
    assert head(run)["head"] == p1["candidate"]
    r = cli(run, "revert")
    assert r.returncode == 0, r.stderr
    h2 = head(run)
    rep = load(run / f"reverted/reverted_{h2['head'][:12]}_report.json")
    assert rep["exact_undo"] and rep["blender_eval_integer_frames_bit_equal"] and rep["blender_eval_subframes_bit_equal"]
    assert h2["generation"] == 2 and S.canon(E.PersistentStore(run).store.get(h2["head"])["values"]) == S.canon(v0["values"])
    st = assert_consistent(run, v0)
    assert st["edits"]["P1"]["decision"] == "ACCEPTED" and st["edits"]["P2"]["decision"] == "REJECTED"
    recs, torn, bad = TX.read_jsonl(run / "lineage/log.jsonl")
    assert torn is None and bad == 0 and len({r["txn_id"] for r in recs}) == len(recs)


# Frozen expected outcome per boundary (W125 declaration): published? -> head after recovery.
ACCEPT_PRE = ["before_stage", "after_journal_prepared", "after_state_write", "after_blend_copy", "during_blender", "before_publish"]
ACCEPT_POST = ["after_publish", "after_log_append", "torn_log_append", "before_timeline_final", "before_finalize"]


@needs_blender
@pytest.mark.parametrize("mode", ["raise", "exit"])
@pytest.mark.parametrize("point", ACCEPT_PRE + ACCEPT_POST)
def test_accept_fault_matrix(run, point, mode):
    run, v0 = run
    cand = load(run / "edits/P1/proposal.json")["candidate"]
    r = cli(run, "commit", "--request-id", "P1", "--decision", "accept", fault=point, mode=mode)
    assert r.returncode != 0
    published = point in ACCEPT_POST
    assert head(run)["head"] == (cand if published else v0["state_id"])      # the authority is exactly old or new
    st = status(run)
    if mode == "raise" and not published:
        # in-process rollback: nothing pending, staged files quarantined (never deleted), proposal still undecided
        assert ("TRANSACTION_ABORTED_ROLLED_BACK" in r.stderr) or (point == "before_stage" and "INJECTED_FAULT" in r.stderr), r.stderr
        assert not st["recovery_required"]
    elif point == "before_stage":                       # killed before any journal: nothing to recover
        assert not st["recovery_required"]
    else:
        assert st["recovery_required"] and st["pending_transactions"][0]["published"] == published, st
        r2 = cli(run, "commit", "--request-id", "P1", "--decision", "accept")           # blocked until recovery
        assert r2.returncode == 2 and "RECOVERY_REQUIRED" in r2.stderr
        rc = cli(run, "recover")
        assert rc.returncode == 0, rc.stderr
        out = json.loads(rc.stdout)
        assert out["actions"] and ("ROLLED_FORWARD" if published else "ROLLED_BACK") in json.dumps(out["actions"])
    assert not (run / "edits/P1/commit.json").exists() or published
    if not published:
        assert status(run)["edits"]["P1"]["decision"].startswith("UNDECIDED")
        assert not list((run / "accepted").glob("*.blend")) if (run / "accepted").exists() else True
        r3 = cli(run, "commit", "--request-id", "P1", "--decision", "accept")       # the user can simply retry
        assert r3.returncode == 0, r3.stderr
    st = assert_consistent(run, v0)
    assert head(run)["head"] == cand and head(run)["generation"] == 1
    assert st["edits"]["P1"]["decision"] == "ACCEPTED"
    recs, torn, bad = TX.read_jsonl(run / "lineage/log.jsonl")
    assert torn is None and sum(1 for x in recs if x.get("candidate") == cand and x.get("head_after") == cand) == 1


REVERT_PRE = ["before_stage", "after_journal_prepared", "after_state_write", "during_blender", "before_publish"]
REVERT_POST = ["after_publish", "torn_log_append", "before_finalize"]


@needs_blender
@pytest.mark.parametrize("mode", ["raise", "exit"])
@pytest.mark.parametrize("point", REVERT_PRE + REVERT_POST)
def test_revert_fault_matrix(run, point, mode):
    run, v0 = run
    assert cli(run, "commit", "--request-id", "P1", "--decision", "accept").returncode == 0
    acc = head(run)
    r = cli(run, "revert", fault=point, mode=mode)
    assert r.returncode != 0
    published = point in REVERT_POST
    h = head(run)
    if not published:
        assert h == acc                                    # a failed REVERT leaves the prior accepted head intact
    else:
        assert h["op"] == "REVERT" and h["prev_head"] == acc["head"]
    if mode == "exit" or published:
        if point != "before_stage":
            assert status(run)["recovery_required"]
        assert cli(run, "recover").returncode == 0
    assert_consistent(run, v0)
    if not published:
        assert head(run) == acc
        assert cli(run, "revert").returncode == 0           # retry succeeds with exact undo
    h = head(run)
    rep = load(run / f"reverted/reverted_{h['head'][:12]}_report.json")
    assert rep["exact_undo"] and h["generation"] == 2
    assert_consistent(run, v0)


@needs_blender
@pytest.mark.parametrize("point,published", [("during_blender", False), ("after_publish", True)])
def test_forced_termination_of_separate_process_then_restart_recovery(run, tmp_path, point, published):
    """A REVERT process is hard-terminated (TerminateProcess / SIGKILL) while it holds the lock at a boundary."""
    run, v0 = run
    assert cli(run, "commit", "--request-id", "P1", "--decision", "accept").returncode == 0
    acc = head(run)
    hang = tmp_path / "hang"
    pr = cli(run, "revert", fault=point, mode="hang", hang=hang, wait=False)
    _wait(str(hang) + ".reached", 300)
    pr.kill()
    pr.wait(60)
    assert pr.returncode != 0
    h = head(run)
    assert (h["op"] == "REVERT") == published and (h == acc) == (not published)
    st = status(run)
    assert st["recovery_required"] and st["pending_transactions"][0]["published"] == published
    rc = cli_after_kill(run, "recover")
    assert rc.returncode == 0, rc.stderr
    assert_consistent(run, v0)
    if not published:
        assert head(run) == acc and cli(run, "revert").returncode == 0
    assert load(run / f"reverted/reverted_{head(run)['head'][:12]}_report.json")["exact_undo"]


@needs_blender
def test_competing_commits_same_parent_at_most_one_publishes(run, tmp_path):
    run, v0 = run
    hang = tmp_path / "hang"
    a = cli(run, "commit", "--request-id", "P1", "--decision", "accept", fault="before_publish", mode="hang", hang=hang, wait=False)
    _wait(str(hang) + ".reached", 300)
    b = cli(run, "commit", "--request-id", "P2", "--decision", "accept")
    assert b.returncode == 2 and "BLOCKED_RUN_LOCKED" in b.stderr
    assert cli(run, "revert").returncode == 2 and cli(run, "recover").returncode == 2    # fail closed, no waiting/stealing
    Path(str(hang) + ".release").write_text("go")
    a.wait(300)
    assert a.returncode == 0, a.stderr.read()
    b2 = cli(run, "commit", "--request-id", "P2", "--decision", "accept")
    assert b2.returncode == 0 and load(run / "edits/P2/commit.json")["store_record"]["reasons"] == ["STALE_PARENT"]
    h = head(run)
    assert h["head"] == load(run / "edits/P1/proposal.json")["candidate"] and h["generation"] == 1
    assert_consistent(run, v0)


@needs_blender
def test_simultaneous_commits_race_no_lost_update(run):
    run, v0 = run
    ps = [cli(run, "commit", "--request-id", rid, "--decision", "accept", wait=False) for rid in ("P1", "P2")]
    outs = [(p.wait(600), p.stderr.read()) for p in ps]
    acc = [rid for rid in ("P1", "P2") if (run / f"edits/{rid}/commit.json").exists()
           and load(run / f"edits/{rid}/commit.json")["final_status"] == "ACCEPTED"]
    assert len(acc) == 1, outs
    other = "P2" if acc == ["P1"] else "P1"
    oc = run / f"edits/{other}/commit.json"
    assert (not oc.exists() and any("BLOCKED_RUN_LOCKED" in e for _, e in outs)) or load(oc)["store_record"]["reasons"] == ["STALE_PARENT"]
    assert head(run)["generation"] == 1 and head(run)["head"] == load(run / f"edits/{acc[0]}/proposal.json")["candidate"]
    assert_consistent(run, v0)


@needs_blender
def test_tampered_staged_candidate_blend_is_refused(run):
    run, v0 = run
    p = run / "edits/P1/candidate.blend"
    b = bytearray(p.read_bytes())
    b[len(b) // 2] ^= 0x01
    p.write_bytes(bytes(b))
    r = cli(run, "commit", "--request-id", "P1", "--decision", "accept")
    assert r.returncode == 2 and "INVALID_CANDIDATE_BLEND_TAMPERED" in r.stderr
    assert head(run)["head"] == v0["state_id"]
    assert not list((run / "lineage").glob("txn/*ACCEPT*"))
    assert not (run / "edits/P1/commit.json").exists()


@needs_blender
def test_tampered_refs_journal_and_head_blend_stop(run):
    run, v0 = run
    assert cli(run, "commit", "--request-id", "P1", "--decision", "accept").returncode == 0
    refs_p = run / "lineage/refs.json"
    good = refs_p.read_bytes()
    refs = json.loads(good)
    refs_p.write_text(json.dumps(dict(refs, head=v0["state_id"])))            # seal no longer binds
    for args in (("status",), ("revert",), ("recover",), ("commit", "--request-id", "P2", "--decision", "accept")):
        r = cli(run, *args)
        assert r.returncode == 2 and "BLOCKED_REFS_TAMPERED" in r.stderr, (args, r.stderr)
    refs_p.write_bytes(good)
    hb = run / refs["head_blend"]["blend"]
    hb_bytes = hb.read_bytes()
    os.chmod(hb, 0o644)
    hb.write_bytes(hb_bytes[:-1] + bytes([hb_bytes[-1] ^ 1]))
    r = cli(run, "revert")
    assert r.returncode == 2 and "BLOCKED_HEAD_UNVERIFIED" in r.stderr
    assert status(run)["head_verified"]["ok"] is False
    hb.write_bytes(hb_bytes)
    r = cli(run, "revert", fault="before_publish", mode="exit")
    assert r.returncode == TX.FAULT_EXIT_CODE
    jp = sorted((run / "lineage/txn").glob("*REVERT*.json"))[-1]
    j = load(jp)
    jp.write_text(json.dumps(dict(j, expected_head="f" * 64)))                  # tampered journal
    r = cli(run, "recover")
    assert r.returncode == 2 and "BLOCKED_JOURNAL_CORRUPT_OR_TAMPERED" in r.stderr
    assert head(run) == refs                                                     # nothing guessed, head intact


@needs_blender
def test_interrupted_propose_and_stray_refs_temp(run):
    run, v0 = run
    (run / "edits/HALF").mkdir()
    (run / "edits/HALF/request.json").write_text("{}")
    r = cli(run, "commit", "--request-id", "HALF", "--decision", "accept")
    assert r.returncode == 2 and "PROPOSAL_MISSING_OR_INCOMPLETE" in r.stderr
    (run / "lineage/.refs.json.999.tmp").write_bytes(b'{"head": "torn')            # SIMULATED torn temp of a killed publish
    st = status(run)
    assert st["recovery_required"] and st["stray_temp_files"] and st["edits"]["HALF"]["state"].startswith("INCOMPLETE")
    assert cli(run, "recover").returncode == 0
    assert head(run)["head"] == v0["state_id"] and not status(run)["stray_temp_files"]
    assert list((run / "quarantine").rglob(".refs.json.999.tmp"))


@needs_blender
def test_legacy_w124_run_readable_requires_explicit_adopt(run):
    run, v0 = run
    legacy = json.dumps({"head": v0["state_id"]}, indent=1).encode()
    (run / "lineage/refs.json").write_bytes(legacy)
    st = status(run)
    assert st["refs_kind"] == "LEGACY_W124" and st["head"] == v0["state_id"]
    r = cli(run, "commit", "--request-id", "P1", "--decision", "accept")
    assert r.returncode == 2 and "BLOCKED_LEGACY_RUN_REQUIRES_ADOPT" in r.stderr
    assert (run / "lineage/refs.json").read_bytes() == legacy                    # never silently rewritten
    r = cli(run, "adopt")
    assert r.returncode == 0, r.stderr
    assert (run / "lineage/refs.legacy_w124.json").read_bytes() == legacy
    assert head(run)["generation"] == 0 and head(run)["op"] == "ADOPT_LEGACY_W124"
    assert cli(run, "commit", "--request-id", "P1", "--decision", "accept").returncode == 0
    assert_consistent(run, v0)


@needs_blender
def test_nothing_to_revert_and_already_decided(run):
    run, v0 = run
    r = cli(run, "revert")
    assert r.returncode == 2 and "NOTHING_TO_REVERT" in r.stderr
    assert cli(run, "commit", "--request-id", "P3", "--decision", "reject").returncode == 0
    r = cli(run, "commit", "--request-id", "P3", "--decision", "accept")
    assert r.returncode == 2 and "REFUSED_ALREADY_DECIDED" in r.stderr
    assert head(run)["head"] == v0["state_id"]


@needs_blender
@pytest.mark.parametrize("mode", ["raise", "exit"])
def test_interrupted_explicit_reject_never_moves_head(run, mode):
    run, v0 = run
    r = cli(run, "commit", "--request-id", "P3", "--decision", "reject", fault="reject_before_record", mode=mode)
    assert r.returncode != 0 and head(run)["head"] == v0["state_id"]
    if mode == "exit":
        assert cli(run, "recover").returncode == 0
    assert status(run)["edits"]["P3"]["decision"].startswith("UNDECIDED")
    assert cli(run, "commit", "--request-id", "P3", "--decision", "reject").returncode == 0
    assert head(run)["head"] == v0["state_id"] and head(run)["generation"] == 0
    assert_consistent(run, v0)


@needs_blender
def test_cli_killed_while_blender_child_builds_orphan_writes_only_to_staging(run):
    """Only the CLI python is terminated while its Blender child is mid-build (the child keeps running as an orphan)."""
    run, v0 = run
    assert cli(run, "commit", "--request-id", "P1", "--decision", "accept").returncode == 0
    acc = head(run)
    pr = cli(run, "revert", wait=False)
    t0, marker = time.time(), None
    while time.time() - t0 < 300 and pr.poll() is None and marker is None:
        hits = list((run / "lineage" / "txn").glob("*.stage/*_provenance.json")) if (run / "lineage" / "txn").exists() else []
        marker = hits[0] if hits else None
        time.sleep(0.02)
    assert marker is not None, "revert finished before its Blender build could be observed"
    time.sleep(0.3)
    pr.kill()
    pr.wait(60)
    assert head(run) == acc
    rc = cli_after_kill(run, "recover")
    assert rc.returncode == 0, rc.stderr
    assert "ROLLED_BACK" in rc.stdout and head(run) == acc
    t0 = time.time()
    while time.time() - t0 < 120 and not list(marker.parent.glob("*_build.json")):
        time.sleep(0.2)
    assert not list((run / "reverted").glob("*")) if (run / "reverted").exists() else True   # nothing under final names
    st = assert_consistent(run, v0)
    assert marker.parent.name in st["staging_dirs_never_authority"]
    r = cli(run, "revert")
    assert r.returncode == 0, r.stderr
    assert load(run / f"reverted/reverted_{head(run)['head'][:12]}_report.json")["exact_undo"]
    assert_consistent(run, v0)


@needs_blender
@pytest.mark.parametrize("op", ["revert", "accept"])
def test_real_blender_failure_never_moves_head(run, op):
    """A/B with the W124 code (W3 reproduced in the W125 preflight): the Blender executable is missing at commit/revert time."""
    run, v0 = run
    if op == "revert":
        assert cli(run, "commit", "--request-id", "P1", "--decision", "accept").returncode == 0
    before = head(run)
    R = load(run / "run.json")
    R["blender"] = str(run / "NO_BLENDER_INJECTED.exe")
    (run / "run.json").write_text(json.dumps(R))
    args = ("revert",) if op == "revert" else ("commit", "--request-id", "P2", "--decision", "accept")
    r = cli(run, *args)
    assert r.returncode == 2 and "TRANSACTION_ABORTED_ROLLED_BACK" in r.stderr, r.stderr
    assert head(run) == before and not status(run)["recovery_required"]
    if op == "accept":
        assert status(run)["edits"]["P2"]["decision"].startswith("UNDECIDED")
