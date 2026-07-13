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
