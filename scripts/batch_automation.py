#!/usr/bin/env python3
"""AnimCV autonomous Batch dispatch worker (Main only; fails closed).

ChatGPT publishes one Agent-ready Batch in `docs/next_batch.md` on the authorized branch; this worker, run by the host
scheduler, executes each newly published Batch exactly once in a dedicated clone, then verifies the published Worklog.

Commands:
  poll            scheduled entry point: pull remote state (ff-only), then lightweight checks; executes only when
                  every eligibility rule holds
  status          print the lock / last-poll status
  clear-lock      HUMAN ONLY: remove a preserved lock after inspection (--batch N --confirm-human-reviewed)
  selftest-claude verify the headless Claude entry point from the current (e.g. scheduler) context
  hook-pre-commit / hook-pre-push   git hooks installed in the dedicated clone

The remote branch may advance while a Batch runs (ChatGPT committing central-document updates agreed with the user):
the running Batch keeps the documents of its starting commit, the updates apply from the next Batch, and the finished
Batch is published with one clean merge commit; a conflicting merge is aborted and left for human review.

The worker never writes `docs/next_batch.md`, never creates/switches/deletes branches, never force-pushes, never resets,
and never steals or expires a lock. Only standard library is used so polling stays cheap.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import datetime as dt
import glob
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys

INSTRUCTION = "docs/next_batch.md"
MARKER_STATUS = re.compile(r"^[ 	>]*AUTOMATION_BATCH_STATUS:\s*(\S+)\s*$", re.M)
MARKER_NUMBER = re.compile(r"^[ 	>]*BATCH_NUMBER:\s*(\d+)\s*$", re.M)
NOWIN = 0x08000000 if os.name == "nt" else 0   # CREATE_NO_WINDOW: the scheduler runs pythonw, keep children hidden
BATCH_ID = re.compile(r"\bBATCH\s+(\d+)\b")
WORKLOG = re.compile(r"^(\d+)[A-Z]?_.*\.md$")
REQUIRED_SECTIONS = ("DIRECTION", "STOP CONDITIONS", "COMPLETION CONDITION")
TRAILER = "Automation-Batch"

DEFAULT_ROOT = Path(os.environ.get("USERPROFILE", str(Path.home())))/"animcv-automation"
DEFAULTS = {"branch": "arch/single_frame_first", "remote": "origin", "repo": str(DEFAULT_ROOT/"repo"),
            "state": str(DEFAULT_ROOT/"state"), "host": None, "python": r"C:\Projects\AnimCV\.venv_build_full\Scripts\python.exe",
            "claude": "auto", "timeout_hours": 48, "gpu_busy_check": True, "notify": True,
            # extra checkouts (e.g. the interactive one) fast-forwarded on every poll; never stashed/reset/merged/switched
            "pull_paths": []}


class Refuse(Exception):
    """A write the worker must never perform."""


# ------------------------------------------------------------------ config / small utilities
def load_config(path=None):
    cfg = dict(DEFAULTS)
    path = Path(path) if path else DEFAULT_ROOT/"state"/"config.json"
    if path.exists():
        cfg.update(json.loads(path.read_text(encoding="utf-8")))
    return cfg


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def git(repo, *args, check=True):
    r = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, encoding="utf-8", creationflags=NOWIN)
    if check and r.returncode:
        raise RuntimeError(f"git {' '.join(args)} failed: {r.stderr.strip()}")
    return r.stdout.strip() if check else r


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(value, indent=1), encoding="utf-8")
    os.replace(tmp, path)


def guard_write(rel):
    """Every repository write path the worker owns goes through here; the instruction file is never writable."""
    if Path(rel).as_posix().lstrip("./") == INSTRUCTION:
        raise Refuse("docs/next_batch.md is ChatGPT-owned and read-only for Claude")


# ------------------------------------------------------------------ instruction / worklog parsing (pure)
def parse_instruction(text):
    """-> (batch number or None, problems). Exactly one distinct `BATCH N`, required sections present."""
    problems = []
    numbers = sorted({int(n) for n in BATCH_ID.findall(text)})
    if not numbers:
        problems.append("no BATCH number")
    elif len(numbers) > 1:
        problems.append(f"conflicting BATCH numbers {numbers}")
    elif numbers[0] <= 0:
        problems.append("BATCH number must be positive")
    for section in REQUIRED_SECTIONS:
        if section not in text:
            problems.append(f"missing section {section}")
    if len(re.findall(r"^\s*#*\s*COMPLETION CONDITION", text, re.M)) > 1:
        problems.append("multiple Batch prompts (repeated COMPLETION CONDITION)")
    if len(text.strip()) < 400:
        problems.append("instruction too short to be Agent-ready")
    return (numbers[0] if len(numbers) == 1 and not problems else None), problems


def worklog_numbers(names):
    out = {}
    for name in names:
        m = WORKLOG.match(Path(name).name)
        if m:
            out.setdefault(int(m.group(1)), []).append(Path(name).name)
    return out


def worklog_marker(text):
    s, n = MARKER_STATUS.findall(text), MARKER_NUMBER.findall(text)
    if not s and not n:
        return None
    return {"status": s[-1] if s else None, "number": int(n[-1]) if n else None}


# ------------------------------------------------------------------ remote view (read-only)
@dataclass
class View:
    head: str
    blob: str | None
    text: str | None
    worklogs: dict = field(default_factory=dict)


def remote_view(cfg):
    repo, ref = cfg["repo"], f"{cfg['remote']}/{cfg['branch']}"
    git(repo, "fetch", "--quiet", cfg["remote"], f"+refs/heads/{cfg['branch']}:refs/remotes/{ref}")
    head = git(repo, "rev-parse", ref)
    r = git(repo, "rev-parse", "--verify", "--quiet", f"{ref}:{INSTRUCTION}", check=False)
    blob = r.stdout.strip() if r.returncode == 0 else None
    text = git(repo, "cat-file", "-p", blob) if blob else None
    names = git(repo, "ls-tree", "--name-only", f"{ref}", "docs/").splitlines()
    return View(head, blob, text, worklog_numbers(names))


def worklog_text(cfg, ref, name):
    return git(cfg["repo"], "show", f"{ref}:docs/{name}")


# ------------------------------------------------------------------ state: lock, journal, status
def state_paths(cfg):
    s = Path(cfg["state"])
    return {"lock": s/"active_batch.json", "journal": s/"completed.json", "status": s/"status.json",
            "pause": s/"PAUSE", "logs": s/"logs", "blockers": s/"blockers"}


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return default


def acquire_lock(cfg, record):
    """Atomic create-if-absent (O_EXCL). Exactly one concurrent caller on this host can succeed."""
    path = state_paths(cfg)["lock"]
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        json.dump(record, stream, indent=1)
    return True


def update_lock(cfg, **changes):
    path = state_paths(cfg)["lock"]
    lock = read_json(path)
    if lock is None or lock.get("worker") != changes.pop("expect_worker", lock.get("worker")):
        raise RuntimeError("lock ownership lost")
    lock.update(changes)
    write_json(path, lock)
    return lock


def release_lock(cfg, worker, allowed_states):
    """Remove only our own lock, only in a state where no Batch work can be in flight."""
    path = state_paths(cfg)["lock"]
    lock = read_json(path)
    if lock is None or lock.get("worker") != worker or lock.get("state") not in allowed_states:
        return False
    path.unlink()
    return True


def process_alive(pid, name_hint=None):
    if not pid:
        return False
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.OpenProcess.restype = wintypes.HANDLE
        handle = k32.OpenProcess(0x1000, False, int(pid))   # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return False
        try:
            code = wintypes.DWORD()
            if not k32.GetExitCodeProcess(handle, ctypes.byref(code)) or code.value != 259:   # STILL_ACTIVE
                return False
            if name_hint:
                buf = ctypes.create_unicode_buffer(1024); size = wintypes.DWORD(1024)
                if k32.QueryFullProcessImageNameW(handle, 0, buf, ctypes.byref(size)):
                    return name_hint.lower() in Path(buf.value).name.lower()
            return True
        finally:
            k32.CloseHandle(handle)
    try:
        os.kill(int(pid), 0)   # POSIX only: signal 0 is a pure existence probe
        return True
    except OSError:
        return False


def gpu_busy():
    """Another compute job (python process on the GPU) is running: defer, never contend."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=20, creationflags=NOWIN).stdout
    except (OSError, subprocess.TimeoutExpired):
        return "nvidia-smi unavailable"
    names = [line.split(",")[-1].strip() for line in out.splitlines() if line.strip()]
    hits = [n for n in names if Path(n).name.lower().startswith("python")]
    return f"GPU compute process running: {hits}" if hits else None


# ------------------------------------------------------------------ eligibility (pure given the view)
def evaluate(cfg, view, journal):
    """-> (action, reason, batch). action in noop | stop | execute."""
    if view.text is None:
        return "noop", "no docs/next_batch.md", None
    n, problems = parse_instruction(view.text)
    if n is None:
        return "stop", "malformed instruction: " + "; ".join(problems), None
    latest = max(view.worklogs) if view.worklogs else 0
    done = {e["batch"] for e in journal}
    if n in done or n <= latest:
        return "noop", f"BATCH {n} not newer than latest Worklog {latest} (or already completed)", n
    if n != latest + 1:
        return "stop", f"unexpected BATCH {n}: expected {latest + 1} (latest Worklog {latest}); numbers are never skipped", n
    previous = view.worklogs.get(latest, [])
    if len(previous) != 1:
        return "stop", f"Worklog {latest} is ambiguous: {previous}", n
    return "check_previous", previous[0], n


def previous_complete(cfg, view, journal, latest, name):
    """Automation-executed Batches need their COMPLETE marker; manual Worklogs count once committed on the remote."""
    text = worklog_text(cfg, f"{cfg['remote']}/{cfg['branch']}", name)
    marker = worklog_marker(text)
    automated = any(e["batch"] == latest for e in journal)
    if marker is None:
        return (False, f"Worklog {latest} was automation-executed but has no completion marker") if automated else (True, "manual Worklog committed on remote")
    if marker["status"] == "COMPLETE" and marker["number"] == latest:
        return True, "completion marker present"
    return False, f"Worklog {latest} marker not complete: {marker}"


# ------------------------------------------------------------------ status / notification
def record_status(cfg, outcome, reason, batch=None, notify_key=None):
    paths = state_paths(cfg)
    previous = read_json(paths["status"], {})
    status = {"time": now(), "outcome": outcome, "reason": reason, "batch": batch}
    changed = (previous.get("outcome"), previous.get("reason")) != (outcome, reason)
    write_json(paths["status"], status)
    if changed:
        paths["logs"].mkdir(parents=True, exist_ok=True)
        with open(paths["logs"]/"poll.log", "a", encoding="utf-8") as stream:
            stream.write(json.dumps(status)+"\n")
    if notify_key and changed:
        paths["blockers"].mkdir(parents=True, exist_ok=True)
        (paths["blockers"]/f"{notify_key}.txt").write_text(f"{now()}\n{reason}\n", encoding="utf-8")
        notify(cfg, f"AnimCV automation {outcome}: {reason}")
    return status


def notify(cfg, message):
    if not cfg.get("notify") or os.name != "nt":
        return
    text = message.replace("'", "''")[:250]
    script = ("[Windows.UI.Notifications.ToastNotificationManager,Windows.UI.Notifications,ContentType=WindowsRuntime]|Out-Null;"
              "$t=[Windows.UI.Notifications.ToastNotificationManager]::GetTemplateContent([Windows.UI.Notifications.ToastTemplateType]::ToastText02);"
              "$x=$t.GetElementsByTagName('text');$x.Item(0).AppendChild($t.CreateTextNode('AnimCV automation'))|Out-Null;"
              f"$x.Item(1).AppendChild($t.CreateTextNode('{text}'))|Out-Null;"
              "[Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier('AnimCV').Show([Windows.UI.Notifications.ToastNotification]::new($t))")
    subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script], capture_output=True, timeout=30, creationflags=NOWIN)


# ------------------------------------------------------------------ execution
EXECUTOR_PROMPT = """You are the AnimCV Execution Agent, launched unattended by the scheduled automation worker on Main.

Execute exactly the Batch published in docs/next_batch.md of this checkout. The worker bound it to:
  BATCH {batch}; instruction blob {blob}; starting commit {start}; branch {branch}.
First verify `git rev-parse HEAD:docs/next_batch.md` equals {blob} and HEAD equals {start}. If not, stop without changes.

Permanent rules (they override any conflicting habit):
- Read AGENTS.md and the documents the Batch names. Read docs/next_batch.md; it is READ-ONLY. Never create, edit,
  delete, copy, archive or summarize-in-full that file, never change its Batch number, never append another Batch.
- Work, commit and push only on the current branch {branch}. Never create/switch/rename/merge/rebase/delete branches,
  never force-push, never reset or rewrite history, never skip git hooks.
- Run everything on this machine (Main, local RTX 5080, Python {python}). Do not connect to LabServer63/69 or any other
  machine. Stop if the Batch cannot run here.
- Every commit message you create must end with the trailer line `{trailer}: {batch}` (before Co-Authored-By).
- Respect the Batch's PRESERVE / CHANGE ONLY / DO NOT TUNE and STOP CONDITIONS. Do not tune after seeing results, do
  not expand into production integration or a downstream subsystem, do not start another Batch.
- Write the Worklog docs/{batch}_WORKLOG_<DESCRIPTIVE_NAME>.md following AGENTS.md conventions, with: original task scope,
  starting commit, final source revision, completed/uncompleted work, evidence and population, tests executed, tests
  skipped/failed, CASE/verdict, negative findings, unresolved uncertainties, reproduction, changed files, resource and
  environment constraints. Do not copy the full Batch prompt into it. Do NOT write Next Work Recommendation, Next
  Architecture Decision, Recommended Next Experiment, Proposed Next Batch or Next Implementation Plan.
- This Batch runs entirely against the documents of its starting commit {start}. The remote branch may advance while
  you work (e.g. ChatGPT committing central-document updates agreed with the user); those updates apply from the NEXT
  Batch. Never pull, merge or rebase them into this checkout, and do not re-read or follow them.
- Only as the very last change, when all artifacts and the Worklog are final, add these two lines at the end of the
  Worklog, commit, fetch, and push (fast-forward only):
    AUTOMATION_BATCH_STATUS: COMPLETE
    BATCH_NUMBER: {batch}
  A negative result, CASE D or a valid stop-condition outcome is still COMPLETE. If you are interrupted or cannot
  finish, do NOT write the marker; leave evidence of the partial state.
- If any push is rejected only because the remote branch advanced, do not merge, rebase or force: keep committing
  locally, finish the Batch normally (including the marker) and exit. After you exit the worker publishes your local
  commits with one clean merge commit, or stops for human review if that merge would conflict.
"""


def find_claude(cfg):
    if cfg.get("claude") not in (None, "auto"):
        return cfg["claude"]
    pattern = str(Path(os.environ.get("USERPROFILE", str(Path.home())))/".vscode"/"extensions"/"anthropic.claude-code-*-win32-x64"/
                  "resources"/"native-binary"/"claude.exe")
    def version(p):
        m = re.search(r"claude-code-(\d+)\.(\d+)\.(\d+)", p)
        return tuple(map(int, m.groups())) if m else (0,)
    found = sorted(glob.glob(pattern), key=version)
    if not found:
        raise RuntimeError("Claude CLI not found")
    return found[-1]


def run_claude(cfg, batch, blob, start, worker):
    paths = state_paths(cfg)
    paths["logs"].mkdir(parents=True, exist_ok=True)
    prompt = EXECUTOR_PROMPT.format(batch=batch, blob=blob, start=start, branch=cfg["branch"], python=cfg["python"], trailer=TRAILER)
    log = paths["logs"]/f"batch_{batch}_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    with open(log, "w", encoding="utf-8") as out:
        proc = subprocess.Popen([find_claude(cfg), "-p", prompt, "--dangerously-skip-permissions", "--output-format", "text"],
                                cwd=cfg["repo"], stdout=out, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                creationflags=NOWIN)
        update_lock(cfg, expect_worker=worker, state="RUNNING", executor_pid=proc.pid, executor_log=str(log), running_since=now())
        try:
            code = proc.wait(timeout=float(cfg["timeout_hours"])*3600)
        except subprocess.TimeoutExpired:
            return None   # still running: state stays RUNNING, never killed or released automatically
    return code


def post_verify(cfg, batch, blob, start):
    """Independent of the Agent's own report: marker, history, instruction ownership, publication."""
    repo, ref = cfg["repo"], f"{cfg['remote']}/{cfg['branch']}"
    problems = []
    view = remote_view(cfg)
    if git(repo, "merge-base", "--is-ancestor", start, ref, check=False).returncode:
        problems.append("remote branch no longer descends from the starting commit (history rewritten)")
    if git(repo, "rev-parse", "--abbrev-ref", "HEAD") != cfg["branch"]:
        problems.append("dedicated clone is not on the authorized branch")
    if git(repo, "status", "--porcelain"):
        problems.append("dedicated clone has uncommitted changes")
    if git(repo, "rev-list", f"{ref}..HEAD"):
        problems.append("local commits not published")
    # merge commits only join remote updates (publish_pending); the Batch's own changes are in its non-merge commits
    for sha in git(repo, "rev-list", "--no-merges", f"{start}..{ref}").splitlines():
        msg = git(repo, "show", "-s", "--format=%B", sha)
        if re.search(rf"^{TRAILER}:\s*{batch}\s*$", msg, re.M):
            touched = git(repo, "show", "--name-only", "--format=", sha).splitlines()
            if INSTRUCTION in touched:
                problems.append(f"Batch commit {sha[:10]} modified {INSTRUCTION}")
    names = view.worklogs.get(batch, [])
    if len(names) != 1:
        problems.append(f"expected exactly one Worklog {batch}, found {names}")
    else:
        marker = worklog_marker(worklog_text(cfg, ref, names[0]))
        if not marker or marker["status"] != "COMPLETE" or marker["number"] != batch:
            problems.append(f"Worklog {names[0]} lacks the final COMPLETE marker for BATCH {batch}: {marker}")
    return (not problems), problems, view.head


def publish_pending(cfg, batch, start, attempts=3):
    """Publish a finished Batch whose push was refused because the remote advanced meanwhile.

    The remote advancing during a Batch is expected (ChatGPT commits central-document updates agreed with the user);
    the running Batch keeps its starting documents and the updates apply from the next Batch. Acts only when the local
    commits carry this Batch's final COMPLETE Worklog and both sides still descend from the starting commit; joins the
    two histories with one ordinary merge commit (no rebase, reset or force) and pushes fast-forward. A conflicting
    merge is aborted and left for human review. -> dict with "result" in nothing/pushed/merged/refused.
    """
    repo, ref = cfg["repo"], f"{cfg['remote']}/{cfg['branch']}"
    for _ in range(attempts):
        git(repo, "fetch", "--quiet", cfg["remote"], f"+refs/heads/{cfg['branch']}:refs/remotes/{ref}")
        if not git(repo, "rev-list", f"{ref}..HEAD"):
            return {"result": "nothing"}
        if git(repo, "rev-parse", "--abbrev-ref", "HEAD") != cfg["branch"]:
            return {"result": "refused", "reason": "dedicated clone is not on the authorized branch"}
        if git(repo, "status", "--porcelain", "--untracked-files=no"):
            return {"result": "refused", "reason": "dedicated clone has uncommitted tracked changes"}
        for side in ("HEAD", ref):
            if git(repo, "merge-base", "--is-ancestor", start, side, check=False).returncode:
                return {"result": "refused", "reason": f"{side} no longer descends from the starting commit"}
        local = worklog_numbers(git(repo, "ls-tree", "--name-only", "HEAD", "docs/").splitlines()).get(batch, [])
        marker = worklog_marker(git(repo, "show", f"HEAD:docs/{local[0]}")) if len(local) == 1 else None
        if not marker or marker["status"] != "COMPLETE" or marker["number"] != batch:
            return {"result": "refused", "reason": f"local Worklog {batch} is not final (no COMPLETE marker); not publishing"}
        incoming = git(repo, "rev-list", "--reverse", f"HEAD..{ref}").splitlines()
        if incoming:
            msg = (f"Merge remote {cfg['branch']} updates into Batch {batch} publication\n\n"
                   f"The remote advanced by {len(incoming)} commit(s) while Batch {batch} ran; the Batch used the documents\n"
                   f"of its starting commit {start[:10]} and the remote updates apply from the next Batch.\n"
                   f"Clean merge by the automation worker (no rebase, reset or force).\n\n{TRAILER}: {batch}")
            r = git(repo, "merge", "--no-ff", "--no-edit", "-m", msg, ref, check=False)
            if r.returncode:
                git(repo, "merge", "--abort", check=False)
                return {"result": "refused", "reason": "merging the remote updates conflicts; aborted, nothing published",
                        "incoming": incoming}
        r = git(repo, "push", "--quiet", cfg["remote"], f"HEAD:refs/heads/{cfg['branch']}", check=False)
        if r.returncode == 0:
            return {"result": "merged" if incoming else "pushed", "incoming": incoming, "head": git(repo, "rev-parse", "HEAD")}
        # the remote moved again between fetch and push: the merge commit stays local and the next attempt builds on it
    return {"result": "refused", "reason": f"push still rejected after {attempts} attempts: {r.stderr.strip()[-200:]}"}


def sync_checkout(cfg, path):
    """Pull the authorized branch into one checkout, fast-forward only.

    Skips (and says why) instead of stashing, resetting, merging or switching: wrong branch, tracked modifications,
    or local commits not on the remote. Untracked files are left alone; git itself refuses if one would be overwritten.
    """
    path = Path(path)
    if not (path/".git").exists():
        return {"path": str(path), "result": "skipped", "reason": "not a git checkout"}
    if git(path, "rev-parse", "--abbrev-ref", "HEAD") != cfg["branch"]:
        return {"path": str(path), "result": "skipped", "reason": "not on the authorized branch; no checkout is performed"}
    if git(path, "status", "--porcelain", "--untracked-files=no"):
        return {"path": str(path), "result": "skipped", "reason": "dirty: tracked modifications; nothing is stashed or reset"}
    ref = f"{cfg['remote']}/{cfg['branch']}"
    git(path, "fetch", "--quiet", cfg["remote"], f"+refs/heads/{cfg['branch']}:refs/remotes/{ref}")
    before = git(path, "rev-parse", "HEAD")
    if git(path, "merge-base", "--is-ancestor", "HEAD", ref, check=False).returncode:
        return {"path": str(path), "result": "skipped", "reason": "diverged from the remote branch; no merge or rebase"}
    r = git(path, "pull", "--ff-only", "--quiet", cfg["remote"], cfg["branch"], check=False)
    if r.returncode:
        return {"path": str(path), "result": "skipped", "reason": "pull --ff-only refused: " + r.stderr.strip().splitlines()[-1][:200]}
    after = git(path, "rev-parse", "HEAD")
    return {"path": str(path), "result": "updated" if after != before else "up_to_date", "from": before, "to": after}


def log_sync(cfg, results):
    """Only real updates/skips are logged; an up-to-date pull stays quiet."""
    noisy = [r for r in results if r["result"] != "up_to_date"]
    if noisy:
        paths = state_paths(cfg)
        paths["logs"].mkdir(parents=True, exist_ok=True)
        with open(paths["logs"]/"poll.log", "a", encoding="utf-8") as stream:
            stream.write(json.dumps({"time": now(), "sync": noisy}, ensure_ascii=False)+"\n")


def poll(cfg, executor=run_claude):
    paths = state_paths(cfg)
    Path(cfg["state"]).mkdir(parents=True, exist_ok=True)
    worker = f"{socket.gethostname()}:{os.getpid()}"
    if cfg.get("host") and os.environ.get("COMPUTERNAME", socket.gethostname()).lower() != cfg["host"].lower():
        return record_status(cfg, "stop", f"host {socket.gethostname()} is not the designated executor {cfg['host']}", notify_key="host")
    if paths["pause"].exists():
        return record_status(cfg, "paused", "PAUSE file present")
    # Step 0: pull the remote state here first. Extra checkouts always; the dedicated clone only when no Batch holds
    # the lock (a running Batch owns its working tree and its bound start commit).
    lock = read_json(paths["lock"])
    synced = [sync_checkout(cfg, p) for p in cfg.get("pull_paths", []) if Path(p).resolve() != Path(cfg["repo"]).resolve()]
    clone = sync_checkout(cfg, cfg["repo"]) if lock is None else None
    log_sync(cfg, synced + ([clone] if clone else []))
    if clone and clone["result"] == "skipped":
        return record_status(cfg, "stop", f"dedicated clone {clone['reason']}", notify_key="clone_sync")
    # Step 1: lock
    if lock is not None:
        # Only an in-flight claim/run with a live owner is a quiet no-op; any other state (NEEDS_REVIEW, COMPLETE not
        # released, unknown) is preserved and reported, regardless of whether some process with that pid exists.
        alive = {"RUNNING": lambda: process_alive(lock.get("executor_pid"), "claude"),
                 "CLAIMED": lambda: process_alive(lock.get("poller_pid"))}.get(lock.get("state"), lambda: False)()
        if alive:
            return record_status(cfg, "noop", f"BATCH {lock.get('batch')} {lock.get('state')} by {lock.get('worker')}", lock.get("batch"))
        retried = retry_publication(cfg, lock)
        if retried is not None:
            return retried
        return record_status(cfg, "stop", f"ambiguous lock for BATCH {lock.get('batch')} in state {lock.get('state')}; "
                             "preserved for human review", lock.get("batch"), notify_key=f"lock_{lock.get('batch')}_{lock.get('state')}")
    # Steps 2-7
    repo = cfg["repo"]
    if git(repo, "rev-parse", "--abbrev-ref", "HEAD") != cfg["branch"]:
        return record_status(cfg, "stop", "dedicated clone is not on the authorized branch; no checkout is performed", notify_key="branch")
    view = remote_view(cfg)
    journal = read_json(paths["journal"], [])
    action, reason, batch = evaluate(cfg, view, journal)
    if action == "check_previous":
        latest = max(view.worklogs)
        ok, why = previous_complete(cfg, view, journal, latest, reason)
        if not ok:
            return record_status(cfg, "noop", f"previous Batch unfinished: {why}", batch)
        action = "execute"
    if action != "execute":
        key = f"instr_{view.blob[:12]}" if action == "stop" and view.blob else None
        return record_status(cfg, action, reason, batch, notify_key=key)
    # Step 8: no other work
    busy = gpu_busy() if cfg.get("gpu_busy_check") else None
    if busy:
        return record_status(cfg, "noop", f"deferred: {busy}", batch)
    # Step 7 (local side): the clone must be updatable without destructive operations
    if git(repo, "status", "--porcelain"):
        return record_status(cfg, "stop", "dedicated clone is dirty; nothing is stashed or reset", batch, notify_key="dirty")
    ref = f"{cfg['remote']}/{cfg['branch']}"
    if git(repo, "merge-base", "--is-ancestor", "HEAD", ref, check=False).returncode:
        return record_status(cfg, "stop", "dedicated clone has diverged from the remote branch", batch, notify_key="diverged")
    # Step 9: claim
    record = {"batch": batch, "instruction_blob": view.blob, "start_commit": view.head, "worker": worker,
              "poller_pid": os.getpid(), "claimed_at": now(), "state": "CLAIMED"}
    if not acquire_lock(cfg, record):
        return record_status(cfg, "noop", "lost the claim race", batch)
    # Step 10: revalidate after the claim
    again = remote_view(cfg)
    a2, r2, b2 = evaluate(cfg, again, read_json(paths["journal"], []))
    stale = again.blob != view.blob or b2 != batch or a2 not in ("check_previous", "execute") or max(again.worklogs) != max(view.worklogs)
    if not stale and a2 == "check_previous":
        stale = not previous_complete(cfg, again, read_json(paths["journal"], []), max(again.worklogs), r2)[0]
    if stale:
        release_lock(cfg, worker, ("CLAIMED",))
        return record_status(cfg, "noop", "instruction or completion state changed during claim; discarded, will revalidate", batch)
    git(repo, "merge", "--ff-only", "--quiet", ref)
    start = git(repo, "rev-parse", "HEAD")
    if start != again.head or git(repo, "rev-parse", f"HEAD:{INSTRUCTION}") != view.blob:
        release_lock(cfg, worker, ("CLAIMED",))
        return record_status(cfg, "stop", "clone could not be bound to the claimed instruction", batch, notify_key=f"bind_{batch}")
    update_lock(cfg, expect_worker=worker, start_commit=start)
    # Step 11: execute
    record_status(cfg, "running", f"BATCH {batch} started", batch)
    code = executor(cfg, batch, view.blob, start, worker)
    if code is None:
        return record_status(cfg, "stop", f"BATCH {batch} exceeded {cfg['timeout_hours']} h; lock preserved", batch, notify_key=f"timeout_{batch}")
    return conclude(cfg, worker, batch, view.blob, start, code)


def conclude(cfg, worker, batch, blob, start, code):
    """Step 12-13: join remote updates made during the Batch (e.g. ChatGPT central docs), verify, journal, release."""
    paths = state_paths(cfg)
    published = publish_pending(cfg, batch, start) if code == 0 else None
    if published and published["result"] != "nothing":
        update_lock(cfg, expect_worker=worker, publication=published)
    ok, problems, final = post_verify(cfg, batch, blob, start)
    if code != 0:
        ok, problems = False, problems + [f"executor exit code {code}"]
    if published and published["result"] == "refused":
        ok, problems = False, problems + [f"publication: {published['reason']}"]
    if not ok:
        update_lock(cfg, expect_worker=worker, state="NEEDS_REVIEW", problems=problems, ended_at=now())
        return record_status(cfg, "stop", f"BATCH {batch} not verified complete: {problems}", batch, notify_key=f"review_{batch}")
    update_lock(cfg, expect_worker=worker, state="COMPLETE", final_commit=final, ended_at=now())
    journal = read_json(paths["journal"], [])
    entry = {"batch": batch, "instruction_blob": blob, "start_commit": start, "final_commit": final, "completed_at": now()}
    if published and published["result"] == "merged":
        entry["merged_remote_updates"] = published["incoming"]
    journal.append(entry)
    write_json(paths["journal"], journal)
    release_lock(cfg, worker, ("COMPLETE",))
    return record_status(cfg, "complete", f"BATCH {batch} complete at {final[:10]}", batch, notify_key=f"done_{batch}")


# Problems a pre-merge-step worker recorded when the only issue was a remote that advanced during the Batch.
PUBLICATION_ONLY = ("local commits not published", "expected exactly one Worklog {batch}, found []")


def retry_publication(cfg, lock):
    """A NEEDS_REVIEW lock whose sole problems are 'not published because the remote advanced' (written by a worker
    that predates the merge step) is concluded once more with the merge step; anything else stays for human review."""
    batch, start = lock.get("batch"), lock.get("start_commit")
    allowed = {p.format(batch=batch) for p in PUBLICATION_ONLY}
    problems = lock.get("problems") or []
    if (lock.get("state") != "NEEDS_REVIEW" or not start or not lock.get("instruction_blob") or not problems
            or not set(problems) <= allowed or process_alive(lock.get("executor_pid"), "claude")):
        return None
    worker = lock["worker"]
    update_lock(cfg, expect_worker=worker, state="RUNNING", retried_publication_at=now(), problems=[])
    status = conclude(cfg, worker, batch, lock["instruction_blob"], start, 0)
    if status["outcome"] != "complete":
        # record the attempt so the same lock is never retried again
        update_lock(cfg, expect_worker=worker, problems=read_json(state_paths(cfg)["lock"])["problems"] + ["publication retry failed"])
    return status


# ------------------------------------------------------------------ git hooks (dedicated clone only)
def hook_pre_commit(repo="."):
    staged = git(repo, "diff", "--cached", "--name-only").splitlines()
    if INSTRUCTION in staged:
        sys.stderr.write("REFUSED: docs/next_batch.md is ChatGPT-owned and read-only for the automation\n")
        return 1
    return 0


def hook_pre_push(cfg, repo, lines):
    for line in lines:
        if not line.strip():
            continue
        local_ref, local_sha, remote_ref, remote_sha = line.split()
        if remote_ref != f"refs/heads/{cfg['branch']}":
            sys.stderr.write(f"REFUSED: push to {remote_ref}; only {cfg['branch']} is authorized\n"); return 1
        if set(local_sha) == {"0"}:
            sys.stderr.write("REFUSED: branch deletion\n"); return 1
        if set(remote_sha) != {"0"} and git(repo, "merge-base", "--is-ancestor", remote_sha, local_sha, check=False).returncode:
            sys.stderr.write("REFUSED: non-fast-forward push\n"); return 1
    return 0


# ------------------------------------------------------------------ CLI
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=("poll", "status", "clear-lock", "selftest-claude", "hook-pre-commit", "hook-pre-push"))
    ap.add_argument("--config")
    ap.add_argument("--batch", type=int)
    ap.add_argument("--confirm-human-reviewed", action="store_true")
    ap.add_argument("hook_args", nargs="*")
    args = ap.parse_intermixed_args(argv)   # git appends `<remote> <url>` after the options for pre-push
    cfg = load_config(args.config)
    if args.command == "poll":
        try:
            print(json.dumps(poll(cfg)))
        except Exception as error:   # fail closed: record, never retry automatically
            record_status(cfg, "stop", f"poll error: {error!r}", notify_key="error")
            return 2
    elif args.command == "status":
        paths = state_paths(cfg)
        print(json.dumps({"status": read_json(paths["status"]), "lock": read_json(paths["lock"]),
                          "completed": read_json(paths["journal"], []), "paused": paths["pause"].exists()}, indent=1))
    elif args.command == "clear-lock":
        lock = read_json(state_paths(cfg)["lock"])
        if not (args.confirm_human_reviewed and lock and lock.get("batch") == args.batch):
            print("refused: requires --batch matching the lock and --confirm-human-reviewed"); return 1
        if lock.get("state") == "RUNNING" and process_alive(lock.get("executor_pid"), "claude"):
            print("refused: executor process is still alive"); return 1
        state_paths(cfg)["lock"].unlink()
        print(f"lock for BATCH {args.batch} removed by human request")
    elif args.command == "selftest-claude":
        # Same flags as the real executor, so an unattended-mode confirmation prompt would surface here, not mid-Batch.
        out = subprocess.run([find_claude(cfg), "-p", "Reply with exactly: AUTOMATION_SELFTEST_OK", "--max-turns", "1",
                              "--dangerously-skip-permissions", "--output-format", "text"],
                             capture_output=True, text=True, timeout=300, stdin=subprocess.DEVNULL, creationflags=NOWIN)
        result = {"time": now(), "returncode": out.returncode, "ok": "AUTOMATION_SELFTEST_OK" in out.stdout,
                  "stdout": out.stdout[-500:], "stderr": out.stderr[-500:], "claude": find_claude(cfg)}
        Path(cfg["state"]).mkdir(parents=True, exist_ok=True)
        write_json(Path(cfg["state"])/"selftest_claude.json", result)
        print(json.dumps(result))
        return 0 if result["ok"] else 1
    elif args.command == "hook-pre-commit":
        return hook_pre_commit()
    elif args.command == "hook-pre-push":
        return hook_pre_push(cfg, ".", sys.stdin.read().splitlines())
    return 0


if __name__ == "__main__":
    sys.exit(main())
