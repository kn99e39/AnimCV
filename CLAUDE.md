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

## Architecture precedence

`Architecture_v3_FramePose.md` is normative for AnimCV's perception stage:
perception ownership, 2D pose observation (the Geometry Observation Layer
is the abstraction; MMPose + RTMDet is its current Real AnimCV backend, and
a 2D one), frame-pose learning, the role of temporal lifting, and
visual/VLM evidence fusion. Where it and
`Architecture_v2.md` disagree on those subjects, v3 wins and v2 is
historical.

`Architecture_v2.md` remains authoritative, unchanged, for everything v3
does not replace: video intake, rig parsing and RigProfile, bone mapping,
Motion Graph, keyframe collapse, the Blender isolation boundary, and
retargeting/downstream animation contracts.

Results are labelled by evaluation regime, and never compared without the
label: `oracle_geometry` (annotated/projected GT or synthetic projection —
no detector error), `benchmark_detector_observation` (a benchmark's own
shipped detector keypoints, e.g. 3DPW — detector error already present),
`real_animcv_observation` (AnimCV's own MMPose + RTMDet sensor). A
benchmark detector is not an oracle. Everything measured so far is
`benchmark_detector_observation`.

A Real AnimCV (MMPose) observation cache identity must bind the SHA-256 of
the exact image bytes; building one without it is refused. Visual feature
caches are keyed to a `visual_input_fingerprint` binding image bytes, bank
geometry, the crop contract and backbone preprocessing.
Historical v1 caches recorded none of that and are readable only with an
explicit legacy flag, labelled `historical_v1`.

Canonical pose mathematics (bone/torso/hinge geometry, similarity
alignment, root-yaw and bend-direction metrics) is owned by
`src/common/canonical_pose.py`. Both the Frame Pose Core and the Legacy
Temporal Pose Baseline consume it; neither owns it. Its formulas are
pinned bitwise by `tests/test_canonical_pose_parity.py` because A9-A16 and
F0-F2 are defined by exactly those expressions — do not "clean up" one.

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
