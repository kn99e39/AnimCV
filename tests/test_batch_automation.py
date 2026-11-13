"""Dry-run safety tests for the Batch dispatch worker on isolated synthetic git repositories (never the real repo)."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
import batch_automation as ba

BRANCH = "arch/single_frame_first"
SCRIPT = (Path(__file__).resolve().parents[1]/"scripts"/"batch_automation.py").as_posix()


def instruction(n, extra=""):
    body = "\n".join(f"Context line {i} describing the architecture question in sufficient detail." for i in range(12))
    return (f"# BATCH {n} — synthetic dry-run instruction\n\nDIRECTION\n\n{body}\n\nPRESERVE / CHANGE ONLY / DO NOT TUNE\n\n"
            f"STOP CONDITIONS\n\nStop if unsafe.\n\nCOMPLETION CONDITION\n\nThe synthetic batch is complete when its Worklog exists.\n{extra}")


def run(cwd, *args, check=True):
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=check)


@pytest.fixture
def env(tmp_path, monkeypatch):
    for k, v in {"GIT_AUTHOR_NAME": "dry", "GIT_AUTHOR_EMAIL": "dry@example.invalid",
                 "GIT_COMMITTER_NAME": "dry", "GIT_COMMITTER_EMAIL": "dry@example.invalid"}.items():
        monkeypatch.setenv(k, v)
    remote = tmp_path/"remote.git"; seed = tmp_path/"seed"; clone = tmp_path/"clone"; owner = tmp_path/"chatgpt"
    run(tmp_path, "init", "--bare", "-q", "-b", BRANCH, str(remote))
    run(tmp_path, "init", "-q", "-b", BRANCH, str(seed))
    (seed/"docs").mkdir()
    for i in range(1, 7):
        (seed/"docs"/f"{i}_WORKLOG_SYNTH_{i}.md").write_text(f"# Worklog {i}\n")
    run(seed, "add", "."); run(seed, "commit", "-qm", "seed"); run(seed, "remote", "add", "origin", str(remote))
    run(seed, "push", "-q", "origin", BRANCH)
    run(tmp_path, "clone", "-q", "-b", BRANCH, str(remote), str(clone))
    run(tmp_path, "clone", "-q", "-b", BRANCH, str(remote), str(owner))
    cfg = dict(ba.DEFAULTS, repo=str(clone), state=str(tmp_path/"state"), host=None, gpu_busy_check=False, notify=False)
    Path(cfg["state"]).mkdir()
    hooks = clone/".git"/"hooks"
    (tmp_path/"cfg.json").write_text(json.dumps(cfg))
    py = Path(sys.executable).as_posix(); conf = (tmp_path/"cfg.json").as_posix()
    (hooks/"pre-commit").write_text(f'#!/bin/sh\nexec "{py}" "{SCRIPT}" hook-pre-commit --config "{conf}"\n')
    (hooks/"pre-push").write_text(f'#!/bin/sh\nexec "{py}" "{SCRIPT}" hook-pre-push --config "{conf}" "$@"\n')

    def publish(text, path=ba.INSTRUCTION):
        run(owner, "pull", "-q", "--ff-only")
        (owner/path).write_text(text)
        run(owner, "add", path); run(owner, "commit", "-qm", "ChatGPT publishes instruction"); run(owner, "push", "-q")

    def remote_head():
        return run(tmp_path, "--git-dir", str(remote), "rev-parse", BRANCH).stdout.strip()

    return {"cfg": cfg, "clone": clone, "owner": owner, "publish": publish, "remote_head": remote_head, "tmp": tmp_path}


def executor(kind, calls):
    def go(cfg, batch, blob, start, worker):
        calls.append(batch)
        clone = Path(cfg["repo"])
        if kind == "crash":
            return 1
        verdict = "CASE D (negative)" if kind == "negative" else "CASE A"
        marker = "" if kind == "no_marker" else f"\nAUTOMATION_BATCH_STATUS: COMPLETE\nBATCH_NUMBER: {batch}\n"
        (clone/"docs"/f"{batch}_WORKLOG_SYNTH_RESULT.md").write_text(f"# Worklog {batch}\n{verdict}\n{marker}")
        if kind in ("remote_moved", "conflict"):
            # the Batch runs on its starting documents: the central doc is the pre-update one
            calls.append((clone/"docs"/"CENTRAL.md").read_text() if (clone/"docs"/"CENTRAL.md").exists() else None)
            if kind == "conflict":
                (clone/"docs"/"CENTRAL.md").write_text("Batch edit\n")
            # meanwhile ChatGPT commits an agreed central-document update to the remote
            owner = Path(cfg["repo"]).parent/"chatgpt"
            run(owner, "pull", "-q", "--ff-only")
            (owner/"docs"/"CENTRAL.md").write_text("Architecture decision v2\n")
            run(owner, "add", "-A"); run(owner, "commit", "-qm", "ChatGPT: central doc update"); run(owner, "push", "-q")
        if kind == "edit_instruction":
            (clone/ba.INSTRUCTION).write_text("tampered")
            run(clone, "add", "-A"); run(clone, "commit", "-q", "--no-verify", "-m", f"tamper\n\n{ba.TRAILER}: {batch}")
        else:
            run(clone, "add", "-A"); run(clone, "commit", "-qm", f"Worklog {batch}\n\n{ba.TRAILER}: {batch}")
        pushed = run(clone, "push", "-q", "origin", BRANCH, check=False)
        assert (pushed.returncode != 0) == (kind in ("remote_moved", "conflict"))   # rejected: remote advanced
        return 0
    return go


def lock(env):
    return ba.read_json(ba.state_paths(env["cfg"])["lock"])


# A
def test_A_missing_instruction_is_a_clean_noop(env):
    head = env["remote_head"](); calls = []
    s = ba.poll(env["cfg"], executor("ok", calls))
    assert s["outcome"] == "noop" and "no docs/next_batch.md" in s["reason"]
    assert calls == [] and lock(env) is None and env["remote_head"]() == head


# B
@pytest.mark.parametrize("text", [instruction(7).replace("BATCH 7", "BATCH"), instruction(7, "\nSee also BATCH 9.\n"),
                                  "BATCH 7\nDIRECTION\nshort"])
def test_B_invalid_or_conflicting_number_never_executes(env, text):
    env["publish"](text); calls = []
    s = ba.poll(env["cfg"], executor("ok", calls))
    assert s["outcome"] == "stop" and "malformed" in s["reason"] and calls == [] and lock(env) is None


# C
def test_C_batch_not_newer_than_latest_worklog_is_noop(env):
    env["publish"](instruction(6)); calls = []
    assert ba.poll(env["cfg"], executor("ok", calls))["outcome"] == "noop" and calls == []


def test_C2_skipped_number_stops(env):
    env["publish"](instruction(9)); calls = []
    s = ba.poll(env["cfg"], executor("ok", calls))
    assert s["outcome"] == "stop" and "expected 7" in s["reason"] and calls == []


# D
def test_D_previous_worklog_incomplete_blocks(env):
    env["publish"]("# Worklog 6\nAUTOMATION_BATCH_STATUS: IN_PROGRESS\nBATCH_NUMBER: 6\n", "docs/6_WORKLOG_SYNTH_6.md")
    env["publish"](instruction(7)); calls = []
    s = ba.poll(env["cfg"], executor("ok", calls))
    assert s["outcome"] == "noop" and "unfinished" in s["reason"] and calls == []


def test_D2_automation_executed_previous_without_marker_blocks(env):
    ba.write_json(ba.state_paths(env["cfg"])["journal"], [{"batch": 6}])
    env["publish"](instruction(7)); calls = []
    assert "unfinished" in ba.poll(env["cfg"], executor("ok", calls))["reason"] and calls == []


# E
def test_E_active_lock_owned_by_another_worker_blocks(env):
    env["publish"](instruction(7)); calls = []
    ba.acquire_lock(env["cfg"], {"batch": 7, "worker": "other", "state": "CLAIMED", "poller_pid": os.getpid()})
    s = ba.poll(env["cfg"], executor("ok", calls))
    assert s["outcome"] == "noop" and calls == [] and lock(env)["worker"] == "other"


# F
def test_F_simultaneous_claims_exactly_one_succeeds(env):
    start = time.time() + 2
    code = ("import sys,time,json;sys.path.insert(0,r'%s');import batch_automation as b;"
            "time.sleep(max(0,%f-time.time()));print(b.acquire_lock({'state':r'%s'},{'worker':sys.argv[1]}))"
            % (str(Path(SCRIPT).parent), start, env["cfg"]["state"]))
    procs = [subprocess.Popen([sys.executable, "-c", code, str(i)], stdout=subprocess.PIPE, text=True) for i in range(8)]
    results = [p.communicate()[0].strip() for p in procs]
    assert results.count("True") == 1 and results.count("False") == 7


# G
def test_G_instruction_change_before_acquisition_discards_attempt(env, monkeypatch):
    env["publish"](instruction(7)); calls = []
    real = ba.acquire_lock
    def racing(cfg, record):
        ok = real(cfg, record)
        env["publish"](instruction(7, "\nRevised by the reviewer.\n"))
        return ok
    monkeypatch.setattr(ba, "acquire_lock", racing)
    s = ba.poll(env["cfg"], executor("ok", calls))
    assert s["outcome"] == "noop" and "changed during claim" in s["reason"] and calls == [] and lock(env) is None


# H + I
@pytest.mark.parametrize("kind", ["ok", "negative"])
def test_H_I_completed_batch_never_reexecutes_and_negative_case_is_complete(env, kind):
    env["publish"](instruction(7)); calls = []
    s = ba.poll(env["cfg"], executor(kind, calls))
    assert s["outcome"] == "complete" and calls == [7] and lock(env) is None
    journal = ba.read_json(ba.state_paths(env["cfg"])["journal"])
    assert [e["batch"] for e in journal] == [7]
    for _ in range(3):
        assert ba.poll(env["cfg"], executor(kind, calls))["outcome"] == "noop"
    assert calls == [7]
    text = (Path(env["cfg"]["repo"])/"docs"/"7_WORKLOG_SYNTH_RESULT.md").read_text()
    assert ("CASE D" in text) == (kind == "negative")     # verdict untouched, not retried


# J
@pytest.mark.parametrize("kind", ["crash", "no_marker"])
def test_J_unexpected_termination_preserves_lock_and_never_reruns(env, kind):
    env["publish"](instruction(7)); calls = []
    s = ba.poll(env["cfg"], executor(kind, calls))
    assert s["outcome"] == "stop" and lock(env)["state"] == "NEEDS_REVIEW"
    for _ in range(2):
        again = ba.poll(env["cfg"], executor("ok", calls))
        assert again["outcome"] == "stop" and "ambiguous lock" in again["reason"]
    assert calls == [7]


def test_J2_dead_running_executor_is_not_stolen(env):
    env["publish"](instruction(7)); calls = []
    ba.acquire_lock(env["cfg"], {"batch": 7, "worker": "w", "state": "RUNNING", "executor_pid": 999999})
    assert "ambiguous lock" in ba.poll(env["cfg"], executor("ok", calls))["reason"] and calls == []
    assert lock(env)["state"] == "RUNNING"


# K
def test_K_instruction_edits_are_refused(env):
    with pytest.raises(ba.Refuse):
        ba.guard_write("docs/next_batch.md")
    env["publish"](instruction(7))
    run(env["clone"], "pull", "-q", "--ff-only")
    (env["clone"]/ba.INSTRUCTION).write_text("tampered")
    run(env["clone"], "add", ba.INSTRUCTION)
    r = run(env["clone"], "commit", "-qm", "edit", check=False)
    assert r.returncode != 0 and "REFUSED" in r.stderr
    run(env["clone"], "reset", "-q", "HEAD", "--", ba.INSTRUCTION); run(env["clone"], "checkout", "--", ba.INSTRUCTION)


def test_K2_agent_that_bypasses_hook_is_caught_by_post_verification(env):
    env["publish"](instruction(7)); calls = []
    s = ba.poll(env["cfg"], executor("edit_instruction", calls))
    assert s["outcome"] == "stop" and "modified docs/next_batch.md" in s["reason"] and lock(env)["state"] == "NEEDS_REVIEW"


# M: the remote advances during a Batch (ChatGPT central-document updates)
def test_M_remote_docs_update_during_batch_is_merged_and_batch_completes(env):
    env["publish"]("Architecture decision v1\n", "docs/CENTRAL.md")
    env["publish"](instruction(7)); calls = []
    blob = ba.remote_view(env["cfg"]).blob
    s = ba.poll(env["cfg"], executor("remote_moved", calls))
    assert s["outcome"] == "complete" and lock(env) is None
    assert calls == [7, "Architecture decision v1\n"]          # the running Batch kept its starting documents
    remote = f"origin/{BRANCH}"
    run(env["clone"], "fetch", "-q")
    show = lambda p: run(env["clone"], "show", f"{remote}:{p}").stdout
    assert show("docs/CENTRAL.md") == "Architecture decision v2\n"      # the update is kept, not overwritten
    assert "AUTOMATION_BATCH_STATUS: COMPLETE" in show("docs/7_WORKLOG_SYNTH_RESULT.md")
    assert run(env["clone"], "rev-parse", f"{remote}:{ba.INSTRUCTION}").stdout.strip() == blob
    tip = run(env["clone"], "show", "-s", "--format=%P%n%B", remote).stdout
    assert len(tip.splitlines()[0].split()) == 2 and f"{ba.TRAILER}: 7" in tip   # one ordinary merge commit
    journal = ba.read_json(ba.state_paths(env["cfg"])["journal"])
    assert journal[-1]["batch"] == 7 and len(journal[-1]["merged_remote_updates"]) == 1
    assert ba.poll(env["cfg"], executor("ok", calls))["outcome"] == "noop" and calls.count(7) == 1


def test_M2_conflicting_remote_update_is_never_forced(env):
    env["publish"](instruction(7)); calls = []
    s = ba.poll(env["cfg"], executor("conflict", calls))
    assert s["outcome"] == "stop" and "conflicts" in s["reason"] and lock(env)["state"] == "NEEDS_REVIEW"
    assert lock(env)["publication"]["result"] == "refused"
    clone = env["clone"]
    assert run(clone, "status", "--porcelain").stdout == "" and not (clone/".git"/"MERGE_HEAD").exists()
    run(clone, "fetch", "-q")
    assert run(clone, "show", f"origin/{BRANCH}:docs/CENTRAL.md").stdout == "Architecture decision v2\n"
    assert run(clone, "ls-tree", "--name-only", f"origin/{BRANCH}", "docs/7_WORKLOG_SYNTH_RESULT.md").stdout == ""
    again = ba.poll(env["cfg"], executor("ok", calls))           # never retried automatically
    assert "ambiguous lock" in again["reason"] and lock(env)["state"] == "NEEDS_REVIEW"


def test_M4_review_lock_from_pre_merge_worker_is_published_on_next_poll(env, monkeypatch):
    env["publish"](instruction(7)); calls = []
    monkeypatch.setattr(ba, "publish_pending", lambda *a, **k: {"result": "nothing"})   # the old worker had no merge step
    s = ba.poll(env["cfg"], executor("remote_moved", calls))
    assert s["outcome"] == "stop" and set(lock(env)["problems"]) == {"local commits not published",
                                                                       "expected exactly one Worklog 7, found []"}
    monkeypatch.undo()
    s = ba.poll(env["cfg"], executor("ok", calls))
    assert s["outcome"] == "complete" and lock(env) is None and calls.count(7) == 1
    journal = ba.read_json(ba.state_paths(env["cfg"])["journal"])
    assert journal[-1]["batch"] == 7 and journal[-1]["merged_remote_updates"]


def test_M3_unfinished_batch_is_not_published_even_if_remote_moved(env):
    env["publish"](instruction(7))
    ba.poll(env["cfg"], lambda *a: 0)                           # executor wrote nothing, exited 0
    assert lock(env)["state"] == "NEEDS_REVIEW"
    run(env["owner"], "pull", "-q", "--ff-only")
    start = lock(env)["start_commit"]
    (env["clone"]/"docs"/"7_WORKLOG_X.md").write_text("# Worklog 7\npartial\n")
    run(env["clone"], "add", "-A"); run(env["clone"], "commit", "-qm", "partial")
    env["publish"]("update\n", "docs/CENTRAL.md")
    r = ba.publish_pending(env["cfg"], 7, start)
    assert r["result"] == "refused" and "not final" in r["reason"]
    assert run(env["clone"], "rev-list", "--merges", f"{start}..HEAD").stdout == ""


def test_executor_prompt_carries_resumption_rule():
    text = ba.EXECUTOR_PROMPT.format(batch=7, blob="b", start="s", branch=BRANCH, python="p", trailer=ba.TRAILER)
    assert f'--grep="^{ba.TRAILER}: 7$"' in text and "RESUMPTION" in text and "immutable" in text


# L
def test_L_wrong_branch_or_dirty_or_diverged_clone_is_never_repaired(env):
    env["publish"](instruction(7)); calls = []
    clone = env["clone"]
    run(clone, "checkout", "-q", "-b", "synthetic-other")
    assert "not on the authorized branch" in ba.poll(env["cfg"], executor("ok", calls))["reason"]
    assert run(clone, "rev-parse", "--abbrev-ref", "HEAD").stdout.strip() == "synthetic-other"   # no checkout performed
    run(clone, "checkout", "-q", BRANCH)
    (clone/"scratch.txt").write_text("user work")
    assert "dirty" in ba.poll(env["cfg"], executor("ok", calls))["reason"] and (clone/"scratch.txt").exists()
    (clone/"scratch.txt").unlink()
    (clone/"local.txt").write_text("x"); run(clone, "add", "."); run(clone, "commit", "-qm", "local only")
    assert "diverged" in ba.poll(env["cfg"], executor("ok", calls))["reason"]
    assert calls == [] and lock(env) is None


def test_L2_pre_push_refuses_other_branch_deletion_and_non_fast_forward(env):
    clone = env["clone"]
    r = run(clone, "push", "origin", f"{BRANCH}:refs/heads/other", check=False)
    assert r.returncode != 0 and "REFUSED" in r.stderr
    r = run(clone, "push", "origin", f":refs/heads/{BRANCH}", check=False)
    assert r.returncode != 0
    run(clone, "commit", "-q", "--allow-empty", "-m", "a"); run(clone, "push", "-q", "origin", BRANCH)
    run(clone, "reset", "-q", "--hard", "HEAD~1"); run(clone, "commit", "-q", "--allow-empty", "-m", "b")
    r = run(clone, "push", "--force", "origin", BRANCH, check=False)
    assert r.returncode != 0 and "non-fast-forward" in r.stderr


def test_host_guard_and_pause(env, monkeypatch):
    env["publish"](instruction(7)); calls = []
    (ba.state_paths(env["cfg"])["pause"]).write_text("")
    assert ba.poll(env["cfg"], executor("ok", calls))["outcome"] == "paused"
    ba.state_paths(env["cfg"])["pause"].unlink()
    cfg = dict(env["cfg"], host="SOME-OTHER-HOST")
    assert "not the designated executor" in ba.poll(cfg, executor("ok", calls))["reason"] and calls == []


def test_repeated_noop_polls_are_quiet(env):
    calls = []
    for _ in range(5):
        ba.poll(env["cfg"], executor("ok", calls))
    log = ba.state_paths(env["cfg"])["logs"]/"poll.log"
    assert len(log.read_text().splitlines()) == 1


def test_poll_pulls_remote_state_into_clone_before_reading(env):
    env["publish"]("notes\n", "docs/notes.md"); calls = []
    s = ba.poll(env["cfg"], executor("ok", calls))
    assert s["outcome"] == "noop" and calls == []
    clone = env["clone"]
    assert run(clone, "rev-parse", "HEAD").stdout.strip() == env["remote_head"]() and (clone/"docs"/"notes.md").exists()


def test_extra_checkout_is_fast_forwarded_but_never_forced(env):
    user = env["tmp"]/"user"
    run(env["tmp"], "clone", "-q", "-b", BRANCH, str(env["tmp"]/"remote.git"), str(user))
    cfg = dict(env["cfg"], pull_paths=[str(user)]); calls = []
    env["publish"]("a\n", "docs/a.md")
    ba.poll(cfg, executor("ok", calls))
    assert run(user, "rev-parse", "HEAD").stdout.strip() == env["remote_head"]()
    (user/"docs"/"a.md").write_text("user edit\n")                       # tracked modification: skipped, untouched
    env["publish"]("b\n", "docs/b.md")
    assert ba.poll(cfg, executor("ok", calls))["outcome"] == "noop"
    assert (user/"docs"/"a.md").read_text() == "user edit\n" and not (user/"docs"/"b.md").exists()
    assert "dirty" in (ba.state_paths(cfg)["logs"]/"poll.log").read_text()
    run(user, "checkout", "--", "docs/a.md")
    (user/"local.txt").write_text("x"); run(user, "add", "."); run(user, "commit", "-qm", "local")   # diverged: skipped
    before = run(user, "rev-parse", "HEAD").stdout.strip()
    ba.poll(cfg, executor("ok", calls))
    assert run(user, "rev-parse", "HEAD").stdout.strip() == before and calls == []


def test_clone_is_not_pulled_while_a_batch_holds_the_lock(env):
    user = env["tmp"]/"user"
    run(env["tmp"], "clone", "-q", "-b", BRANCH, str(env["tmp"]/"remote.git"), str(user))
    cfg = dict(env["cfg"], pull_paths=[str(user)]); calls = []
    before = run(env["clone"], "rev-parse", "HEAD").stdout.strip()
    ba.acquire_lock(cfg, {"batch": 7, "worker": "w", "state": "CLAIMED", "poller_pid": os.getpid()})
    env["publish"]("c\n", "docs/c.md")
    assert ba.poll(cfg, executor("ok", calls))["outcome"] == "noop" and calls == []
    assert run(env["clone"], "rev-parse", "HEAD").stdout.strip() == before            # running Batch's tree untouched
    assert run(user, "rev-parse", "HEAD").stdout.strip() == env["remote_head"]()      # interactive checkout still updated


def test_marker_and_instruction_parsing():
    assert ba.parse_instruction(instruction(104))[0] == 104
    assert ba.worklog_marker("x\n    AUTOMATION_BATCH_STATUS: COMPLETE\n    BATCH_NUMBER: 104\n") == {"status": "COMPLETE", "number": 104}
    assert ba.worklog_marker("no marker") is None
    assert ba.worklog_numbers(["docs/99A_WORKLOG_X.md", "docs/103_WORKLOG_Y.md", "docs/AUDIT.md"]) == {99: ["99A_WORKLOG_X.md"], 103: ["103_WORKLOG_Y.md"]}
