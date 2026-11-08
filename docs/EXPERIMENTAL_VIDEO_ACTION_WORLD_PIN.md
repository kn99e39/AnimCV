# EXPERIMENTAL — Video → Editable Blender Action → Left-Wrist World Pin (opt-in)

Status: **EXPERIMENTAL, opt-in, introduced in Batch 124** (evidence: `docs/124_WORKLOG_OPT_IN_VIDEO_TO_EDITABLE_ACTION_WORLD_PIN.md`).
It is **not** the default pipeline. The default `motion-tool` commands and the GUI (MotionGraph → RetargetSolver →
AnimationClip → `export-blender`) are unchanged and never call this code. It is not a general retargeter, not a trained
generator, not a style system, and not a quality guarantee.

## What it does

One bounded workflow on the **known W105 BaseRig only** (`examples/skinned_mannequin/BaseRig_Skinned.fbx`):

1. **build** — authenticate one source clip (SHA-256 you declare, a provenance record, declared fps), check the container's
   presentation timestamps, run the **frozen** W102 M1 + S0 perception live, map the S0 hypothesis to the BaseRig with the
   **frozen** W105 GT-free FIXED-slot adapter, and write a **new** `baseline.blend` with one editable Action. A second Blender
   process re-opens the file and reads it back (F-curves, interpolation, timebase, world pose at integer frames and at
   subframes 0.25/0.5/0.75).
2. **propose** — one user-authored `WORLD_POSITION_PIN` request for `left_wrist`, in Blender scene-world metres, on an
   inclusive source-frame interval. The frozen W116 contract validates it, the frozen W116 analytic two-bone resolver
   changes **only** `upperarm_l`/`lowerarm_l` on the pin frames (no automatic out-of-pin edit), the candidate is keyed into
   its own `.blend`, read back independently, and gated (hard pin ≤ 1 mm *experimental diagnostic*, scope, locality, root,
   bone lengths, joint limit, truthful footprint). Nothing is committed. A timeline PNG shows requested pin frames, the
   permitted domain, the **actual** changed frames/bones from Blender, and the gate status.
3. **commit** — explicit `--decision accept|reject`. Atomic; a stale parent is refused; a gate-rejected candidate can never
   be accepted. An accepted edit is copied to `accepted/accepted_NN_<id>.blend`; the baseline file is never modified.
4. **revert** — creates a new state restoring the head's parent values and a new `.blend`; its readback must be bit-equal
   to the restored state's.
5. **review** — 30 fps side-by-side video: original source | baseline Action | accepted edited Action, with the per-frame
   footprint overlaid.
6. **fbx** — exports a state's `.blend` through the existing `blender.export.export_fbx` and re-imports it with the existing
   default importer, and **reports** integer-frame/subframe/interpolation/timebase differences. Nothing is repaired: an FBX
   file on disk is not a round-trip PASS (see W107–W112 for the known failures).

## Requirements

- Blender 5.1 (`--blender` or `ANIMCV_BLENDER`); the project Python environment with OpenCV, numpy, matplotlib, torch and
  mmpose (for live perception).
- The frozen perception assets, described in a JSON like `examples/experimental_video_action/perception_assets.template.json`.
  Their SHA-256 identities are fixed in `src/app/experimental_video_action.py` (`PERCEPTION_EXPECTED`); any mismatch stops.
- A source clip at **exactly 30 fps** with container presentation steps within 1 ms of 1000/30. Other cadences, variable
  frame rate, or a declared fps that disagrees with the container are refused (`UNSUPPORTED_*`), never resampled.
- Run from the repository root with `PYTHONPATH=src` (an installed `app` package from another checkout would shadow it).

## Example

```bash
export PYTHONPATH=src
PY=python   # the project environment
$PY -m app.cli experimental-video-action build \
    --clip /data/clip.webm --clip-sha256 <sha256> --source-record source.json --declared-fps 30 \
    --segment 79 318 --perception-assets perception_assets.json --blender "<blender.exe>" --out runs/my_run
# copy the printed baseline_state_id into a request (see examples/experimental_video_action/left_wrist_pin_request.example.json)
$PY -m app.cli experimental-video-action propose --run runs/my_run --request my_pin.json
$PY -m app.cli experimental-video-action commit  --run runs/my_run --request-id hold_left_wrist_example --decision accept
$PY -m app.cli experimental-video-action review  --run runs/my_run --request-id hold_left_wrist_example
$PY -m app.cli experimental-video-action fbx     --run runs/my_run --state head
$PY -m app.cli experimental-video-action revert  --run runs/my_run
$PY -m app.cli experimental-video-action status  --run runs/my_run
$PY -m app.cli experimental-video-action recover --run runs/my_run   # only if status says recovery_required
```

`source.json` must contain `title`, `license`, `author`, `source_uri`. The slice records these values; it does not judge them.

## Request format

```json
{"schema": "animcv_exp_world_pin_request_v1", "request_id": "hold_1", "parent_state_id": "<current head>",
 "type": "WORLD_POSITION_PIN", "semantic_joint": "left_wrist", "coordinate_frame": "SCENE_WORLD", "unit": "m",
 "frames": [210, 218],
 "target": {"kind": "CAPTURED_FROM_EVALUATED_STATE", "capture_frame": 210},
 "authorized_channels": ["lowerarm_l", "upperarm_l"]}
```

- `target.kind` is `CAPTURED_FROM_EVALUATED_STATE` (hold the wrist where Blender evaluates it at `capture_frame`) or
  `CAPTURED_PLUS_DECLARED_OFFSET` with `offset_m` = [x, y, z] in Blender scene-world metres. Absolute targets are not
  supported in this slice (frozen W116 target-provenance contract).
- Only `left_wrist` and exactly `{upperarm_l, lowerarm_l}` are supported. Other joints, ambiguous bindings, unknown joints,
  other coordinate frames/units, non-finite numbers, frames outside the Action, and a stale `parent_state_id` are rejected
  with a reason and no change.

## Output directory (one run; nothing is overwritten)

```
run/
  run.json                       run record (immutable-file hashes)
  manifest/input_manifest.json   read-only: clip identity, timebase, perception identity, rig identity, frozen-code hashes
  manifest/inspection.json       clip + decoded-frame identity for the frozen diagnostic
  perception/m1s0/               frozen M1+S0 outputs (live) and log
  rig/video_cases.json           W105 FIXED-slot per-frame rig values
  baseline/baseline.blend        unedited Action (immutable)  + action_sidecar.json, bindings.json, build/readback reports
  edits/<request_id>/            request.json, proposal.json, candidate.blend (if any), timeline.png, commit.json, timeline_final.png
  accepted/accepted_NN_<id>.blend
  reverted/reverted_<id>.blend   + exact-undo report
  lineage/                       content-addressed read-only states, refs.json (authority), log.jsonl, blends.json,
                                 txn/<txn>.json journals + txn/<txn>.stage/ staging (never authority), txn.lock
  quarantine/<txn>/              unpublished files moved (never deleted) by a rollback / recover
  recovery/<id>.json             what each `recover` did
  review/<request_id>/           30 fps A/B video, stills, render logs
  fbx/<state>/                   exported .fbx, re-import readback, round-trip report
```

## Transactions, interruption and recovery (Batch 125)

Evidence: `docs/125_WORKLOG_CRASH_CONSISTENT_EDITABLE_ACTION_TRANSACTIONS.md`. Still EXPERIMENTAL and opt-in.

**Authority.** `lineage/refs.json` is the only authority for the current head. Since W125 it is a sealed record
(`schema: animcv_exp_refs_v2`) that binds, in one file, the head state id, a generation counter, and the head's `.blend`
path + SHA-256 + readback report + SHA-256. A `.blend`, a `proposal.json`, a `candidate.blend` or a file in `accepted/`
is **never** an accepted edit by its existence; only what `refs.json` names is.

**What `commit --decision accept` and `revert` do, in order:**

1. take a per-run OS lock (`lineage/txn.lock`; non-blocking: a second process stops with `BLOCKED_RUN_LOCKED` and never waits,
   steals or expires it; the OS releases it if the holder dies);
2. refuse to start if an earlier transaction is unfinished (`RECOVERY_REQUIRED`) or the head does not verify;
3. write a sealed journal `lineage/txn/<txn>.json` (PREPARED) naming every file it will create;
4. stage: the content-addressed state (atomic write), the `.blend` and a fresh-process Blender readback, all inside
   `lineage/txn/<txn>.stage/`; verify them (accept: bit-identical to the gated candidate and the actual frame/bone
   footprint equal to the proposal's; revert: exact undo at integer frames and subframes); only then give them their final
   names in `accepted/` or `reverted/`;
5. **publish**: compare the `refs.json` bytes with those read under the lock, then replace `refs.json` atomically
   (temp file + fsync + rename). Before this point the old head is authoritative; after it the new head is;
6. afterwards append the lineage log, update `lineage/blends.json`, write `commit.json` and `timeline_final.png`, and mark
   the journal FINALIZED.

`commit --decision reject` (and a gate-rejected or stale accept) never moves the head; its publication point is the atomic
`commit.json`.

**If a command fails or is killed:**

| Where it stopped | What you see | What to do |
|---|---|---|
| exception before publication | `STOP TRANSACTION_ABORTED_ROLLED_BACK`; head unchanged; staged files moved to `quarantine/<txn>/` | retry the command |
| killed before publication | `status` → `recovery_required: true`, pending txn `published: false` | `recover` (rolls back), then retry |
| after publication (exception or kill) | `status` → pending txn `published: true` ("head ALREADY moved"); no success message may have been printed | `recover` (completes log, registry, `commit.json`, timeline) |
| torn log line / stray temp file | `status` → `recovery_required: true` | `recover` (bytes kept in `quarantine/`) |
| tampered `refs.json`, journal, head `.blend`, manifest or baseline | `STOP BLOCKED_REFS_TAMPERED` / `BLOCKED_JOURNAL_CORRUPT_OR_TAMPERED` / `BLOCKED_HEAD_UNVERIFIED` / `BLOCKED_IMMUTABLE_OUTPUT_CHANGED` | not repaired automatically: inspect by hand |

`recover` never deletes a file (it moves unpublished ones to `quarantine/`), never marks an incomplete operation ACCEPTED,
and STOPs instead of guessing when a journal and `refs.json` disagree. Other commands refuse to run while recovery is
needed. Right after a hard kill on Windows the OS may take a moment to release the dead process's lock; `recover` then
stops with `BLOCKED_RUN_LOCKED` and can simply be re-run.

**W124 runs** (unversioned `refs.json`) stay readable (`status`, `review`, `fbx`). Editing commands stop with
`BLOCKED_LEGACY_RUN_REQUIRES_ADOPT` until you run `adopt`, which verifies the run and writes v2 refs, keeping the original as
`lineage/refs.legacy_w124.json`.

**Example with a simulated interruption** (`ANIMCV_EXP_TXN_FAULT*` variables are for tests and evidence only):

```bash
export PYTHONPATH=src
$PY -m app.cli experimental-video-action propose --run R --request hold.json
ANIMCV_EXP_TXN_FAULT=after_publish ANIMCV_EXP_TXN_FAULT_MODE=exit   $PY -m app.cli experimental-video-action commit --run R --request-id hold_1 --decision accept   # exits 86, no message
$PY -m app.cli experimental-video-action status  --run R   # head = the candidate; pending txn published: true; recovery_required
$PY -m app.cli experimental-video-action recover --run R   # ROLLED_FORWARD; commit.json + timeline_final.png written
$PY -m app.cli experimental-video-action status  --run R   # recovery_required: false; edits.hold_1.decision = ACCEPTED
```

**Not protected (W125 finding; superseded by Batch 126 — see the next section).** At W125, `edits/<id>/proposal.json` was not sealed. `commit` re-verifies that the accepted `.blend`
is bit-identical to the candidate and that its actual footprint equals the proposal's, but it reads the **gate decision**
from `proposal.json`. A hand-edited proposal that flips a gate REJECT to ACCEPT is therefore published (shown on a synthetic
fixture: an UNREACHABLE candidate with a 136 mm pin residual). The statement "a gate-rejected candidate can never be
accepted" holds only if `proposal.json` is not tampered with. The W124 code had the same exposure.

**Durability scope.** Tested: Python exceptions, `os._exit`, and hard termination of the CLI process (including while its
Blender child is running) on Windows 11 / local NTFS. Not tested and not claimed: power loss, OS crash, full disk,
network or other filesystems. On Windows the directory entry of a rename is not fsynced (Python cannot do that), so a
published head is durable against process death, not guaranteed against power loss.

## Commit-time semantic authority (Batch 126)

Since Batch 126 (evidence: `docs/126_WORKLOG_COMMIT_TIME_WORLD_PIN_GATE_INTEGRITY.md`, Agent-run) **`proposal.json` is never the
gate authority.** `commit --decision accept` re-derives the frozen W116/W118 decision itself and publishes only if it agrees.

**What `propose` now writes.** A versioned proposal (`"schema": "animcv_exp_proposal_v2"`) with a `provenance` block and a
`proposal_seal`. The provenance binds the exact request bytes, the parent head (state id, `.blend` and readback hashes from
`refs.json`), the binding record, the candidate state file, `.blend` and readback, and a digest of every gate field. The seal
is an unkeyed SHA-256: a **consistency check**, not authentication. Somebody able to rewrite every file can recompute it.

**What `commit --decision accept` does before the W125 transaction may publish:**

1. Refuses a pre-W126 (unsealed) proposal: `BLOCKED_LEGACY_PROPOSAL_REQUIRES_REPROPOSE`. Its gate cannot be re-established
   and is never grandfathered. Re-propose the request under a new request id, or `commit --decision reject` it.
2. Checks the hash-bound evidence (seal, decision/status consistency, request bytes, binding record, parent, candidate state,
   `.blend` and readback). The binding record is **re-derived from the immutable baseline** with the frozen `verify_binding`
   and compared with `baseline/bindings.json`.
3. A sealed gate REJECT is recorded as REJECTED (`GATE_REJECTED_CANNOT_ACCEPT`), and a stale parent as `STALE_PARENT`, as
   before; the head never moves.
4. **Re-derives the gate** from the original request bytes, the verified head and its readback, and the re-derived bindings,
   with the unchanged frozen validator, resolver, evaluator, footprint, display check and subframe rule. The recomputed
   candidate must be the proposal's candidate (same content id), the decision must be ACCEPT, and every gate field must equal
   the sealed proposal.
5. Inside the W125 transaction, after staging the accepted copy, a **fresh Blender process** reads it back and step 4 is
   repeated on that readback.
6. Right before the publication point, every decision input is re-hashed (an edit while the lock is held is caught).

Any failure stops with `REFUSED_SEMANTIC_AUTHORIZATION` (exit 2). The head, the accepted Action and every input file are left
exactly as they are; nothing is repaired. A refusal inside the transaction is rolled back like any other pre-publication
failure. Each refusal is recorded as a new file `edits/<id>/authorization_refusals/<time>_<id>.json` (`stage`, `primary`,
`reasons`). The reason order (first present = primary) is: provenance reasons (`PROPOSAL_SEAL_INVALID`,
`PROPOSAL_DECISION_STATUS_CONFLICT`, `REQUEST_MISSING`, `REQUEST_BYTES_CHANGED`, `BINDING_RECORD_CHANGED`, …,
`CANDIDATE_READBACK_NOT_OF_BLEND`), then `RECOMPUTED_GATE_REJECT`, `CANDIDATE_NOT_REPRODUCED`, `PROPOSAL_GATE_RECORD_DIVERGES`,
then the `FRESH_…` equivalents, then `INPUTS_CHANGED_DURING_TRANSACTION`. Earlier, pre-existing checks keep their W125 codes
(`INVALID_CANDIDATE_TAMPERED`, `INVALID_CANDIDATE_BLEND_TAMPERED`, `BLOCKED_HEAD_UNVERIFIED`).

**Identity policy.** `request.json` is bound by **exact bytes**: even a reformat is refused (`REQUEST_BYTES_CHANGED`) — re-propose.
`proposal.json` and `bindings.json` are bound by **content**: a re-indent or key reorder is accepted.

**What `status` shows.** `gate` is the proposal's decision only if its provenance verifies. Otherwise it reads
`UNTRUSTED (<reason>)`, `REFUSED_AT_COMMIT (<primary>)` or `LEGACY_UNVERIFIED:<decision>`. `semantic_authorization` is
`AUTHORIZED_AT_COMMIT` for W126 acceptances and `NOT_ESTABLISHED (…)` for acceptances made before W126. Those historical
edits are not reclassified and their files are not rewritten. An explicit REJECT of an unverified proposal is recorded with
`proposal_integrity` and its final timeline is titled `UNTRUSTED PROPOSAL`.

**Recovery.** `recover` never creates an authorization. A published W126 ACCEPT whose journal does not record
`authorized: true` stops with `BLOCKED_DIVERGENT_JOURNAL`. A published pre-W126 journal is rolled forward and marked
`NOT_ESTABLISHED`.

**Example (end user):**

```bash
export PYTHONPATH=src
$PY -m app.cli experimental-video-action propose --run R --request reach_1m.json   # gate REJECT (UNREACHABLE)
# hand-edit edits/reach_1m/proposal.json: "decision": {"decision": "ACCEPT", ...}
$PY -m app.cli experimental-video-action commit --run R --request-id reach_1m --decision accept
#   STOP REFUSED_SEMANTIC_AUTHORIZATION: PROVENANCE: PROPOSAL_SEAL_INVALID ...; head unchanged   (exit 2)
$PY -m app.cli experimental-video-action propose --run R --request hold.json       # gate ACCEPT
$PY -m app.cli experimental-video-action commit --run R --request-id hold_1 --decision accept   # ACCEPTED (authorized)
$PY -m app.cli experimental-video-action revert --run R                            # exact undo
```

**Not covered:** an actor who rewrites the run consistently (including `run.json`, the immutable-file record, sealed
`refs.json`, the frozen code or this module); cryptographic user authentication; power loss / OS crash (W125 limits).

## Opt-in FBX continuous-time interchange (Batch 127)

Evidence: `docs/127_WORKLOG_OPT_IN_FBX_CONTINUOUS_TIME_INTERCHANGE.md` (Agent-run, not independently re-executed). Still
EXPERIMENTAL and opt-in. The `fbx` command above is **unchanged** and still reports the stock exporter/importer FAIL; the
default exporter `src/blender/export.py` and Blender's importer are not changed or patched.

Three separate commands with separate inputs:

| Command | Role | Reads | Writes |
|---|---|---|---|
| `fbx-interchange --run R [--state head\|baseline\|<id>] --out NEW` | producer | the run (read-only) and ONE verified published state's `.blend` | a NEW directory: `package/` + `producer/` evidence + `STATUS.json` |
| `fbx-consume --package NEW/package --out NEW2 [--mode adapter\|raw]` | consumer | **only** `package/package_manifest.json` and `package/animation.fbx` (every Python-level file open is audited) | a NEW `recovered.blend` (or `raw_import.blend`) + `consume_report.json` |
| `fbx-verify --run R --consumed NEW2 --out NEW3` | evaluator | the original `.blend` and the consumed `.blend`, each re-opened in a fresh Blender process | `fbx_verify_report.json` |

**What the package contains.**

- `animation.fbx`: a standard binary FBX. It is exported with:
  - no key simplification (a key at every frame);
  - exactly one take, for the selected Action;
  - the BaseRig skinned mesh, so the file carries a BindPose and the bone rest survives;
  - no `_end` leaf bones.

  Curves whose driving source property is uniformly CONSTANT carry the FBX CONSTANT interpolation flag (`KeyAttrFlags`
  `0x6002`). All other curves keep the exporter's LINEAR flag.
- `package_manifest.json`: a sealed record of take, timebase (fps, frame range), rig topology, per-property interpolation
  directives and source provenance. The seal is an unkeyed SHA-256: a consistency check, not authentication. The manifest
  is written last, so a directory without it is never a package.

**Dependency (do not read more into it).** The package FBX imports into any FBX consumer as a file. Lossless recovery of
held motion and of the scene frame range **requires both the manifest and the Blender 5.1 consumer adapter**.

The package FBX imported alone with Blender's stock importer (`--mode raw`) keeps integer frames, but it:
- interpolates every held key between frames (S1: up to 0.90 m at subframes);
- loses the scene range.

Unity, Unreal and other FBX consumers were **not tested**.

**Supported (anything else STOPs with a code, and nothing is written as a package):**
- the known W105 BaseRig;
- one armature, plus meshes bound to it;
- one assigned Action, with no NLA, drivers or constraints;
- per object/bone transform property, keys either all CONSTANT or all LINEAR;
- integer key times, with keys spanning scene frames 1..N.

Refused with a STOP code:
- mixed interpolation inside a property, per key or per axis: `UNSUPPORTED_INTERPOLATION_MIXTURE`;
- BEZIER interpolation: `UNSUPPORTED_INTERPOLATION_BEZIER`;
- another rig: `UNSUPPORTED_RIG`.

Other Actions in the file are excluded from the take and listed in the manifest.

**Consumer refusals** (exit 2; no `.blend` written; package files never repaired):

| Code | Cause |
|---|---|
| `PACKAGE_INCOMPLETE` | no manifest (e.g. a killed export) |
| `MANIFEST_SEAL_INVALID` | the manifest was edited without resealing |
| `FBX_HASH_MISMATCH` | the FBX is not the one the manifest names (including a stale FBX from another package) |
| `FBX_PARSE_ERROR` | the FBX is truncated |
| `TAKE_NOT_UNIQUE` / `TAKE_MISMATCH` | the FBX has more than one take, or a take other than the declared one |
| `TIMEBASE_MISMATCH` | the frame rate differs |
| `UNSUPPORTED_RIG` / `RIG_MISMATCH` | the manifest declares an unknown rig, or the imported rig is not the declared one |
| `METADATA_FBX_DISAGREE` | the FBX flags and the manifest directives disagree |
| `UNSUPPORTED_*_VERSION` | the Blender or importer version is not 5.1 |

**One complete user path** (Git Bash, repository root):

```bash
export PYTHONPATH=src ANIMCV_BLENDER="C:/Program Files/Blender Foundation/Blender 5.1/blender.exe"
PY=python   # the project environment
$PY -m app.cli experimental-video-action status          --run R                      # note head / generation
$PY -m app.cli experimental-video-action fbx-interchange --run R --state head --out out/pkg_head
$PY -m app.cli experimental-video-action fbx-consume     --package out/pkg_head/package --out out/consumed_head
$PY -m app.cli experimental-video-action fbx-verify      --run R --consumed out/consumed_head --out out/verify_head
#   -> "interchange_verdict": "INTERCHANGE_PASS" | "FAIL" and "mismatches"; details in out/verify_head/fbx_verify_report.json
$PY -m app.cli experimental-video-action fbx             --run R --state head         # stock path, still FAIL (unchanged)
$PY -m app.cli experimental-video-action revert          --run R                      # undo the head's edit (unaffected by export)
```

Open `out/consumed_head/recovered.blend` in Blender 5.1 to use the recovered Action (`root|root|AnimCV_EXP_<id>`). An export
never changes the run, the head, the generation, the journals or the source `.blend`. A package stays bound to the state it
was made from: after a later commit or revert, `fbx-verify --state head` refuses it (`PACKAGE_NOT_OF_THIS_STATE`).

**Cost (S1, 240 frames).** The package FBX is 1.15 MB, against 0.37 MB for the stock export (3.1×). It holds 468 curves
and 112,320 keys. Export, consume and verify take about 1 s, 1 s and 2.5 s.

## Known limitations (do not read the output as more than it is)

- **Partial rig.** Driven: Armature Object rotation, `spine_01`, `thigh_l/r`, `calf_l/r`, `upperarm_l`, `lowerarm_l`.
  The right arm, upper spine, neck/head, clavicles, hands, fingers and feet stay at rest. No root translation, no foot
  contact. Left/right arm assignment and 3D accuracy are unverified GT-free hypotheses; subject identity is `UNKNOWN`.
- **Keys are CONSTANT** (frozen keying rule): playback holds each source frame; subframes equal the previous integer frame.
- **Hard pin only on the pin frames.** Release/entry discontinuities at the pin boundaries are not smoothed (no automatic
  transition edit in this slice); they are disclosed in `proposal.json` (`quality_disclosure`).
- **The 1 mm pin check is an experimental diagnostic**, not a Production tolerance.
- **FBX:** the existing exporter/importer are used unchanged; integer-frame drift, CONSTANT→LINEAR interpolation and scene
  range loss are expected (W107–W112) and reported, not fixed. Since W127, the opt-in `fbx-interchange` / `fbx-consume`
  path recovers the Action in a Blender 5.1 consumer, but only with its package manifest and adapter. Raw FBX alone is
  still not lossless, and no game-engine import was tested.
- Only the BaseRig is supported. Any other rig, cadence, joint or edit type is out of scope.
