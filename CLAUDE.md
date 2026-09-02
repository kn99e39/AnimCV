# Working on this repo

## Multiple machines, same branch

This project is actively worked on from more than one machine (at least
one macOS session and one Windows session), often both against the
`On_Work` branch. Nothing enforces that only one of them is active at a
time.

**Before starting any work session on a branch, run `git fetch` and
compare against the remote** (e.g. `git log HEAD..origin/<branch>`) —
don't assume your local branch is current. If the remote has moved,
pull/merge before making new changes, not after. Skipping this doesn't
usually cause conflicts within a single edit, but it lets the two
sessions' histories diverge silently, which then surfaces as a merge
(possibly with real conflicts, e.g. this has already happened once on
`src/pose/mmpose_adapter.py`, which both a macOS and a Windows session
modified independently for unrelated reasons) the next time either side
pushes.

Corollary: don't assume a stale mental model of "what's on this branch"
from earlier in a conversation is still accurate if meaningful time has
passed — re-check `git log`/`git status` against the remote rather than
trusting a summary from a previous turn.

## Experiments run on LabServer63

Training and every GPU experiment run on the SSH host `LabServer63`
(`~/.ssh/config`, user `nd`) in the checkout at `/home/nd/AnimCV`. The
macOS and Windows checkouts are for authoring; nothing is measured
there. That makes LabServer63 a third clone under the same
"fetch before you start" rule above.

When a branch is created, switched or pushed locally, mirror it on the
server (`ssh LabServer63 'cd ~/AnimCV && git fetch origin && git
checkout <branch>'`) — otherwise the experiment environment is silently
still on the old branch. Check `git status` there first: that checkout
normally carries untracked `.animcv_sync_stage/` and `docker/` plus
stashes, and none of them should be disturbed.

Datasets live outside the repo at `~/animcv-data` (3DPW with its
`imageFiles/`, AMASS, body models) and outputs at `~/animcv-output`;
MPI-INF-3DHP annotations are inside the repo at
`datasets/mpi_inf_3dhp/`. GPU: one RTX 3080 Ti, 12 GB.

## Branch layout

- `main`: kept lean on purpose — no `tests/`, and evaluation/verification
  narrative is intentionally light. Only content needed to actually run
  the framework.
- `On_Work`: the real active development branch, full test suite
  included. Most work happens here.

## OS-specific scripts

Windows-only and macOS-only scripts live under `windows/` and `mac/`
respectively (not mixed at the repo root), so it's never ambiguous which
one to run on which machine. Each script hops back to the project root
itself on startup (`cd /d "%~dp0.."` / `cd "$(dirname "${BASH_SOURCE[0]}")/.."`),
so it works the same whether invoked from the repo root or by
double-clicking inside the subfolder.
