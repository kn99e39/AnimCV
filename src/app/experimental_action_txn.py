"""EXPERIMENTAL (Batch 125): crash-consistency primitives for the opt-in ``experimental-video-action`` slice.

Stdlib only (imported lazily by ``app.experimental_video_action``; never by the default CLI/GUI path).

Provides, and nothing more:

  * ``atomic_write_bytes`` / ``atomic_dump`` - write a sibling temp file, flush + ``os.fsync``, then ``os.replace``.
    ``os.replace`` is an atomic rename on the same volume on NTFS (MoveFileEx REPLACE_EXISTING) and POSIX filesystems: a
    reader sees either the old or the new complete file, never a torn one.  On POSIX the directory is fsynced afterwards;
    on Windows directory fsync is not available from Python, so the *rename* itself is not claimed durable across power
    loss (process-kill durability only; see the W125 Worklog).
  * ``RunLock`` - one exclusive, NON-BLOCKING, OS-level advisory lock per run (``msvcrt.locking`` on Windows, ``fcntl.flock``
    on POSIX).  The OS releases it when the holding process exits or is killed, so there is no stale lock to steal or
    expire.  A second process fails closed with ``BLOCKED_RUN_LOCKED``; it never waits, steals or overrides.
  * ``fault`` - deterministic fault injection at named boundaries, active ONLY when ``ANIMCV_EXP_TXN_FAULT`` names that
    boundary (test / evidence harness use).  Modes: ``raise`` (exception), ``exit`` (``os._exit``: no cleanup, no finally),
    ``hang`` (write ``<ANIMCV_EXP_TXN_HANG_FILE>.reached`` and block until ``.release`` appears, so an external process can
    hard-terminate it or release it).
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

FAULT_ENV = "ANIMCV_EXP_TXN_FAULT"
FAULT_MODE_ENV = "ANIMCV_EXP_TXN_FAULT_MODE"
HANG_FILE_ENV = "ANIMCV_EXP_TXN_HANG_FILE"
FAULT_EXIT_CODE = 86
HANG_MAX_S = 600.0

# Named failure boundaries (frozen before any real S1/S3 outcome; see the W125 declaration).
FAULT_POINTS = (
    "before_stage",              # validated + locked, nothing staged, no journal yet
    "after_journal_prepared",    # journal PREPARED written, no output yet
    "after_state_write",         # content-addressed state file written (accept candidate / revert state)
    "after_blend_copy",          # accepted .blend copy written + renamed into accepted/, not yet read back
    "during_blender",            # REVERT: .blend built, independent readback not yet run (ACCEPT: before readback)
    "before_publish",            # everything staged + verified, journal STAGED, refs not replaced
    "after_publish",             # refs.json replaced (publication point), nothing ancillary written
    "after_log_append",          # lineage log appended, registry / commit.json / timeline not yet written
    "before_timeline_final",     # commit.json written, final timeline not drawn, journal not FINALIZED
    "before_finalize",           # all ancillary written, journal not FINALIZED
    "torn_log_append",           # writes HALF of the lineage log line, then fails (torn append)
    "reject_before_record",      # explicit REJECT: journal written, commit.json (the REJECT publication point) not yet
)


class InjectedFault(RuntimeError):
    """Raised by ``fault`` in ``raise`` mode.  Deliberately NOT a SliceStop: it simulates an unexpected crash."""


def fault(point: str):
    if os.environ.get(FAULT_ENV) != point:
        return
    mode = os.environ.get(FAULT_MODE_ENV, "raise")
    if mode == "exit":
        sys.stdout.flush()
        os._exit(FAULT_EXIT_CODE)
    if mode == "hang":
        hf = os.environ.get(HANG_FILE_ENV)
        if not hf:
            raise InjectedFault(f"{point}: hang mode needs {HANG_FILE_ENV}")
        Path(hf + ".reached").write_text(json.dumps({"point": point, "pid": os.getpid()}), encoding="utf-8")
        t0 = time.time()
        while not Path(hf + ".release").exists():
            if time.time() - t0 > HANG_MAX_S:
                raise InjectedFault(f"{point}: hang not released within {HANG_MAX_S} s")
            time.sleep(0.05)
        return
    raise InjectedFault(f"injected fault at {point}")


def _fsync_dir(d: Path):
    if os.name == "nt":       # not available on Windows from Python; documented durability limit
        return False
    fd = os.open(str(d), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    return True


def atomic_write_bytes(path, data: bytes):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    _replace(tmp, path)
    _fsync_dir(path.parent)


def _replace(src, dst, attempts=20, pause_s=0.05):
    """``os.replace``.  On Windows a concurrent reader holding the destination open (without FILE_SHARE_DELETE) makes the
    rename fail with a sharing violation; it is retried a bounded number of times (<= ~1 s), then the error propagates
    (the caller's transaction then fails closed).  This is an I/O retry, not a decision retry."""
    for i in range(attempts):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if os.name != "nt" or i == attempts - 1:
                raise
            time.sleep(pause_s)


def atomic_dump(path, obj, default=None):
    atomic_write_bytes(path, json.dumps(obj, indent=1, sort_keys=True, default=default).encode("utf-8"))


def fsync_file(path):
    with open(path, "rb+") as f:
        os.fsync(f.fileno())


def append_line_durable(path, line: str, torn_point=None):
    """Append one complete JSON line and fsync.  ``torn_point`` (fault point name) lets the harness tear the write."""
    data = (line + "\n").encode("utf-8")
    with open(path, "ab") as f:
        if torn_point is not None and os.environ.get(FAULT_ENV) == torn_point:
            f.write(data[: max(1, len(data) // 2)])
            f.flush()
            os.fsync(f.fileno())
            fault(torn_point)
        f.write(data)
        f.flush()
        os.fsync(f.fileno())


def read_jsonl(path):
    """-> (records, torn_tail_bytes or None, n_bad_lines).  A final line without a newline that does not parse is a torn tail."""
    p = Path(path)
    if not p.exists():
        return [], None, 0
    raw = p.read_bytes()
    recs, bad, torn = [], 0, None
    lines = raw.split(b"\n")
    for i, ln in enumerate(lines):
        if not ln.strip():
            continue
        try:
            recs.append(json.loads(ln.decode("utf-8")))
        except (ValueError, UnicodeDecodeError):
            if i == len(lines) - 1:          # no trailing newline -> torn tail
                torn = ln
            else:
                bad += 1
    return recs, torn, bad


class RunLock:
    """Exclusive non-blocking OS advisory lock on ``<lineage>/txn.lock``.  Released by the OS on exit / kill."""

    def __init__(self, lineage_dir, op, stop_cls):
        self.path = Path(lineage_dir) / "txn.lock"
        self.op, self.stop_cls, self.fh = op, stop_cls, None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fh = open(self.path, "a+b")
        try:
            if os.name == "nt":
                import msvcrt
                fh.seek(0)
                msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            fh.close()
            holder = None
            hp = self.path.with_name("txn.lock.holder.json")
            if hp.exists():
                try:
                    holder = json.loads(hp.read_text(encoding="utf-8"))
                except ValueError:
                    holder = "unreadable"
            raise self.stop_cls("BLOCKED_RUN_LOCKED", f"another experimental-video-action process holds {self.path} "
                                f"(last holder record: {holder}); not waiting, not stealing - retry when it has finished")
        self.fh = fh
        try:   # informational only (never used to decide ownership)
            atomic_dump(self.path.with_name("txn.lock.holder.json"), {"pid": os.getpid(), "op": self.op, "at": time.time()})
        except OSError:
            pass
        return self

    def __exit__(self, *exc):
        if self.fh is not None:
            try:
                if os.name == "nt":
                    import msvcrt
                    self.fh.seek(0)
                    msvcrt.locking(self.fh.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(self.fh.fileno(), fcntl.LOCK_UN)
            finally:
                self.fh.close()
                self.fh = None
        return False
