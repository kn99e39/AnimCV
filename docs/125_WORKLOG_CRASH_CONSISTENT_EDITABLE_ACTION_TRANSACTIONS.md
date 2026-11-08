# Worklog — Crash-Consistent Editable Blender Action Transactions and Recovery (2026-11-09)

Batch 125. A bounded engineering-integration batch on `arch/single_frame_first`, run unattended on Main: local RTX 5080,
Python `C:\Projects\AnimCV\.venv_build_full` (3.11.9), Blender 5.1.0. `docs/next_batch.md` was read only.

Every number, test result and hash below is **AGENT-RUN** on Main. Nobody else has re-executed any of it.

## 1. Official CASE and scope

**Verdict: CASE A under the frozen rule — TRANSACTION CAPABILITY support only.** Source:
`results/assess/assessment.json` (SHA-256 `2f887a4bbb3fd788c0e7091c563b059a3d50b26d9db31e677392f29ba2b61e9d`).

- Every declared gate passes: T1 fault matrix, T2 process kill + restart, T3 concurrency, T4 tamper, T5 real S1/S3
  transactions, T6 independent audit, T7 native-rate review, T9 regression/isolation.
- **What CASE A covers:**
  - process-level failures only: Python exceptions, `os._exit`, and hard termination (`TerminateProcess`) of the CLI
    process, including while its Blender child is running;
  - competing CLI processes;
  - Windows 11 / local NTFS.
- **What CASE A does not cover:**
  - power loss, OS crash, full disk, network or other filesystems, and POSIX. These were declared out of scope before
    the run and are not claimed.
  - animation quality, source fidelity, FBX portability, or Production readiness.
- **Post-outcome negative finding** (§9): a hand-edited `proposal.json` can still publish a gate-rejected candidate. This
  is outside the declared gates and does not change the CASE, but it limits what "a gate-rejected candidate can never
  be accepted" means.
- **The sources are not new evidence.** S1 and S3 are **saved-artifact replays** of the W124 live runs: copies of the W124
  run directories, verified file-by-file. They are not a fresh live-perception run. They are the W121/W122 engineering
  sources, not new independent sources.
- Historical CASEs, including W124 CASE B, are untouched. No historical Worklog was edited.

## 2. Starting state, revisions, environment

| Item | Value |
|---|---|
| Instruction blob | `94355c1c1cdaa48d5ab5db7a21926555a3990ebb`, verified first |
| Starting commit | `c5d23da7f40f4e8ca904941d80d33426fc1181a0`. It equalled `origin/arch/single_frame_first` after fetch, with no concurrent changes before the instrument push. |
| Pre-outcome instrument commit | `78f3533f0a021caa68308571f6ad00d4b4ff4588`: code, tests, harness and usage doc, pushed before any S1/S3 W125 outcome |
| Declaration | `C:/Users/dna10/animcv-w125/results/declare/declaration.json`, SHA-256 `cae950472de76dc5946f09a16b48eb5cf6e763fed109eba565c07078dfa4aabe`, `outcomes_inspected: false`, with the LF SHA-256 of every instrument file. The harness refuses to run if any of them changed. |
| Final source revision | the commit adding this Worklog. It changes only documentation; no `src/`, `tests/` or `scripts/` file changed after `78f3533`. The COMPLETE marker is a separate, final commit. |
| Machine | Main, Windows 11 Pro 10.0.26200, NTFS. Python 3.11.9, Blender 5.1.0. No LabServer connection, no training, no network use. |

**Debug use.** Before the declaration, a copy of the W124 **S2 debug run** was used once for a harness dry run. The
original is `animcv-w124/debug/harness_base/runs/S1_live`, which is the S2 clip despite its name. The output is
`animcv-w125/debug/`. This debug use is **excluded from all evidence**. The dry run found one harness defect, fixed
before the freeze: the audit's `ok` field returned a list instead of a boolean while a transaction was pending.

## 3. Pre-change authority map and old failure windows

**W124 normal-case invariants, recorded before changes:**

- `lineage/refs.json = {"head": id}` is the only head pointer.
- `lineage/states/` holds content-addressed, read-only W116 states.
- `blends.json` maps a state to its `.blend` and readback.
- `log.jsonl` is append-only.
- `edits/<id>/commit.json` records the user decision.
- `accepted/accepted_NN_<id>.blend` is the accepted copy.
- W116 `Store.commit` compares the head with the expected parent **in memory only**.

**Failure windows, found by code inspection** (`results/preflight/preflight.json`, SHA-256 `1e6ef08e…`):

| Window | Where | Exposure |
|---|---|---|
| W1 | `cmd_commit` | The head is saved before the accepted copy, its SHA check, the registry, `commit.json` and the timeline. |
| W2 | `PersistentStore.save` | `write_text` truncates and rewrites `refs.json` in place, so a kill can leave it torn or empty. The log is appended separately. |
| W3 | `cmd_revert` | The head is saved before the Blender build and readback. |
| W4 | `blends.json` | Rewritten in place. |
| W5 | all commands | No cross-process exclusion: last writer wins. |
| W6 | W116 `Store.put` | Writes the state file in place. |
| W7 | accepted copies | `accepted_NN` is chosen by counting files, so an orphan shifts the numbering. |

**W3 was reproduced**, not only inspected, on authentic data with the **starting-commit code** (`git archive c5d23da`,
run against a copy of the W124 S1 run):

- The Blender executable was made unavailable and the old `revert` was run.
- The old `revert` crashed with `FileNotFoundError`, but the head moved from `49dbe510…` to `d77109d9…`.
- That new head has **no `.blend` registered**.

W1, W2 and W4–W7 remain code-inspected only.

**W124 controls.** All 1,128 S1 and 692 S3 run files equal the W124 evidence manifest (`507099dc…`). The S1 and S3 clips
match their declared SHA-256. The same check after the run again gave 0 differences: the W124 originals were never
modified.

## 4. Transaction design and durability limits

All new code is opt-in. It lives in `src/app/experimental_video_action.py` and a new stdlib-only helper,
`src/app/experimental_action_txn.py`.

**Authority.** `lineage/refs.json` becomes a sealed v2 record. One file binds:

- the head state id;
- a `generation` counter;
- `head_blend`: the `.blend` path and SHA-256, the readback report path and SHA-256, and the value digest;
- the publishing transaction id and operation.

The seal is a SHA-256 over the record. Any mismatch stops with `BLOCKED_REFS_TAMPERED`; the record is never repaired.

**Exclusion.** Each run has one OS advisory lock: `msvcrt.locking` on Windows, `flock` on POSIX.

- It is **non-blocking**: a second process fails closed with `BLOCKED_RUN_LOCKED`.
- There is no waiting, stealing or expiry.
- The OS releases the lock when the holder dies. Measured release after a kill: 0.05 s, asynchronous on Windows.

**ACCEPT and REVERT order:**

1. Take the lock.
2. Refuse if any journal is unfinished (`RECOVERY_REQUIRED`) or the head does not verify.
3. Write a sealed journal `lineage/txn/<txn>.json` in state PREPARED, naming every final output.
4. Write the content-addressed state atomically. The frozen W116 `Store` is subclassed only so that `put` writes via
   temp file + fsync + rename.
5. Stage the `.blend` and a **fresh-process Blender readback** inside `lineage/txn/<txn>.stage/`. Verify them:
   - **ACCEPT:**
     - the copy's SHA equals the gated candidate's;
     - the readback is bound to that copy;
     - the value digest, every integer frame and every subframe are bit-equal to the gated candidate's readback;
     - the actual footprint (frozen W118 `derive_footprint`, exact equality) equals the proposal's frames and bones;
   - **REVERT:** exact undo, meaning the values, the value digest, and the integer-frame and subframe Blender evaluation
     are bit-equal to the restored state.
6. Move the verified files to their final names in `accepted/` or `reverted/`, still unpublished. Set the journal to
   STAGED.
7. **Publication point:** re-read the `refs.json` bytes and compare them with those read under the lock
   (compare-and-swap), then `os.replace` a fsynced temp file.
8. After publication: append the log line (idempotent per transaction), update the registry, write `commit.json` and
   `timeline_final.png` (both written atomically), and set the journal to FINALIZED.

**REJECT.** A REJECT, a gate-rejected accept, or a stale-parent accept never moves the head. Its publication point is the
atomic `commit.json`.

**Failure handling:**

| When | Behaviour |
|---|---|
| Exception before publication | In-process rollback. Planned outputs are moved, never deleted, to `quarantine/<txn>/`. A valid content-addressed state is kept but unreferenced. The journal becomes ABORTED. Stop code `TRANSACTION_ABORTED_ROLLED_BACK`. |
| Exception after publication | `RECOVERY_REQUIRED_AFTER_PUBLICATION`. The head is already the new verified state. |
| Kill | The journal stays pending. Every editing command, plus `review` and `fbx`, stops with `RECOVERY_REQUIRED`. |

**`recover`.** An explicit command, run under the lock:

- If `refs.json` names the transaction, it rolls **forward**: it completes the log, registry, `commit.json` (marked
  `completed_by_recovery`) and the timeline.
- Otherwise it requires that the head and generation still equal the journal's expectation, then rolls **back**. If
  they differ, it stops with `BLOCKED_DIVERGENT_JOURNAL`.
- A torn final log line has its bytes moved to quarantine and the complete prefix kept.
- Stray temp files are quarantined.
- A corrupt or tampered journal stops with `BLOCKED_JOURNAL_CORRUPT_OR_TAMPERED`; nothing is guessed.
- **It never marks an incomplete operation ACCEPTED** and never deletes a file.

**Kill after publication, before the success message.** The head is the new, fully verified state. No success message
appears. `status` shows the pending transaction with `published: true` and "head ALREADY moved". `recover` completes the
remaining records.

**Orphan Blender.** On Windows, killing the CLI Python does not kill its Blender child. Because every Blender output goes
to the transaction's staging directory, a late orphan write can never land under a final name. This was shown on the
real S1 run (§6).

**W124 runs.** Runs with unversioned refs stay readable through `status`, `review` and `fbx`. Editing commands stop with
`BLOCKED_LEGACY_RUN_REQUIRES_ADOPT`. The explicit `adopt` command:

- verifies the lineage, the log's head chain, and the head `.blend` and readback;
- keeps the original refs byte-for-byte as `refs.legacy_w124.json`;
- writes v2 refs.

Runs are never rewritten silently.

**Durability limits:**

- Files and their temp copies are fsynced.
- Directory fsync after a rename is not available from Python on Windows, so the rename is **durable against process
  death, not guaranteed against power loss** or an OS crash.
- POSIX does fsync the directory, but that path is untested.
- On a Windows sharing violation during a rename (a concurrent reader holding the destination), the rename is retried at
  most about 1 s, then the transaction fails closed. This is an I/O retry, not a decision retry.

**Unchanged:**

- W116 hard-pin semantics, resolver and evaluator; W118 footprint and display;
- thresholds, keying, the FBX exporter and importer;
- MotionGraph and AnimationClip schemas;
- the default CLI and GUI;
- the frozen-code hash list (still verified).

## 5. Fault-point matrix, kill/restart, concurrency, tamper (focused tests, real Blender BaseRig)

**Test fixture.** `tests/test_experimental_action_txn.py` builds a **synthetic-value** 8-frame run on the **real** W105
BaseRig with real Blender, using the slice's own build, readback and binding steps. It then drives the **user CLI in
separate processes**. The values are synthetic, not video.

**Pre-publication boundaries:**

| Boundaries | Exception | `os._exit` |
|---|---|---|
| ACCEPT: before_stage, after_journal_prepared, after_state_write, after_blend_copy, during_blender, before_publish | Old head kept. Rolled back in-process (before_stage: nothing staged). Proposal still UNDECIDED; retry succeeds. | Old head kept. `status` shows recovery required (except before_stage). Other commands blocked. `recover` rolls back; retry succeeds. |
| REVERT: same boundaries (during_blender = between the Blender build and its readback) | Prior accepted head intact. Retry gives an exact undo. | Prior accepted head intact. Retry gives an exact undo. |

**Post-publication boundaries:**

| Boundaries | Exception | `os._exit` |
|---|---|---|
| ACCEPT: after_publish, after_log_append, torn_log_append, before_timeline_final, before_finalize | Head is the candidate; `RECOVERY_REQUIRED_AFTER_PUBLICATION`. | Head is the candidate; recovery required. |
| REVERT: after_publish, torn_log_append, before_finalize | Head is the revert state. | Head is the revert state. |

In every post-publication case, `recover` rolls forward with exactly one log record, writes `commit.json` and
`timeline_final`, and the status reads ACCEPTED. For REVERT, the undo is exact.

**Other cases:**

| Case | Outcome |
|---|---|
| Explicit REJECT interrupted before its record (exception / exit) | Head never moves. The proposal stays UNDECIDED; a re-reject succeeds. |
| Separate-process `TerminateProcess` while holding the lock (REVERT during_blender / after_publish) | Rolled back / rolled forward as declared. |
| CLI Python killed **while its Blender child builds** | The orphan writes only into staging. `recover` rolls back; the retry is exact. |
| Real Blender failure (missing executable) for REVERT and ACCEPT, the A/B of W3 | `TRANSACTION_ABORTED_ROLLED_BACK`, head unchanged. |
| Two commits on one parent, overlap forced | The second gets `BLOCKED_RUN_LOCKED`, and so do `revert` and `recover`. After release, it gets `STALE_PARENT`. One publication, generation 1. |
| Two commits launched simultaneously (unforced) | Exactly one ACCEPTED; the other LOCKED or `STALE_PARENT`. |
| Tampered staged `candidate.blend` | `INVALID_CANDIDATE_BLEND_TAMPERED`. No journal, no decision. |
| Tampered `refs.json` (status/revert/recover/commit) | `BLOCKED_REFS_TAMPERED` (STOP) |
| Byte flipped in the head `.blend` | `BLOCKED_HEAD_UNVERIFIED` (STOP); `status` shows `head_verified.ok = false` |
| Tampered pending journal | `BLOCKED_JOURNAL_CORRUPT_OR_TAMPERED`; head intact |
| Torn log tail / stray `.refs.json.*.tmp` (**simulated**) | Repaired by `recover`, bytes quarantined. Head unchanged. |
| Interrupted propose (no `proposal.json`) | `PROPOSAL_MISSING_OR_INCOMPLETE`; `status` shows INCOMPLETE |
| Legacy refs | Readable. Editing stops; legacy bytes unchanged. `adopt` keeps them and then commits work. |

**Repairable versus must-STOP:**

- **Repaired safely by `recover`:** interrupted staging, pending publication, torn log tail, stray temp files.
- **Must STOP:** tampered refs, journal or head `.blend`; immutable baseline or manifest; divergent journal.

## 6. Real S1/S3 saved-artifact replay (Agent-run)

**Setup.** The W124 S1 and S3 run directories were copied and verified equal file by file to W124, then audited. After
**every** step, a separate harness process:

- re-read the authority;
- checked head → state → `.blend` → readback hashes;
- ran a **fresh Blender readback of the head `.blend`** and compared it bitwise with the recorded readback (value digest,
  integer frames and subframes);
- checked that no ACCEPTED commit lies outside the published history and that no unpublished file sits in `accepted/`;
- recomputed the footprint and display truth from the accepted readback;
- checked the immutable baseline and manifest and the source clip.

All 27 step audits passed (S1: 15, S3: 12). Logs: `results/run/S1_steps.json` (`ae16c884…`) and `S3_steps.json` (`c17b1fce…`).

**Requests:**

- **R1** holds the left wrist at its evaluated position (S1: frames 210–218, capture 210; S3: 41–49, capture 41).
- **R2** is the same window with the target 3 cm up; the user rejects it.

**S1 steps:**

| Step | Result |
|---|---|
| `status` on the legacy run | `LEGACY_W124` |
| `propose` before adopt | `BLOCKED_LEGACY_RUN_REQUIRES_ADOPT`; nothing written |
| `adopt` | Generation 2, head `49dbe510…` |
| propose R1 | Gate ACCEPT |
| commit accept R1, **TerminateProcess at after_blend_copy** | Exit 1. Head unchanged. `status`: pending ACCEPT, `published: false`, "staged files are NOT accepted" |
| `recover` | ROLLED_BACK. The candidate state is kept unreferenced; the staged copy stays in `txn/<id>.stage/`. |
| commit accept R1 | ACCEPTED, generation 3, head `51f9aae1…`, `accepted_02_51f9aae14b1b.blend` |
| propose R2 → user REJECT | Head unchanged |
| revert, **CLI Python killed while its Blender child was building** | The stage `.blend` did not yet exist at the kill. Head unchanged; pending REVERT, not published. |
| `recover` | ROLLED_BACK. The orphan Blender later wrote `reverted_1994e410e991.blend` and its `build.json` **only into the aborted stage dir**. `status` afterwards: no recovery needed. |
| revert | Exact undo, generation 4, head `1994e410…` |

**S3 steps:**

| Step | Result |
|---|---|
| `adopt` | Generation 2 |
| commit accept R1 with **`os._exit` at after_publish** | Exit 86, no message. Head = `17911f0b…`. `status`: pending, `published: true`, "head ALREADY moved". |
| `recover` | ROLLED_FORWARD. `commit.json` ACCEPTED with `completed_by_recovery`. Generation 3. |
| R2 | REJECT |
| revert with an **exception at before_publish** | `TRANSACTION_ABORTED_ROLLED_BACK`. Head still R1. 8 staged files moved to `quarantine/…REVERT_69bbc804/`. |
| revert | Exact undo, generation 4, head `676ee129…` |

**Parity with W124 on both sources:**

- R1's edited state values are canon-equal to W124's accepted edit (`986840be…` and `508d7269…`).
- The Blender value digest is equal (S1 `3a3eb7eb…`, S3 `9c49233b…`).
- Actual changed frames: S1 211–218, S3 42–49 (8 each). Changed bones: `upperarm_l` and `lowerarm_l` only.
- Hard-pin residual from the accepted copy's fresh readback: 6.4e-7 m (S1) and 5.6e-7 m (S3), integer frames.
- Every publication check was true: SHA, bit-equal integer and subframe evaluation, footprint equal to the proposal.
- R2 changed frames: 210–218 (S1) and 41–49 (S3), with residual about 6.7e-7 / 5.2e-7 m. Rejected; head unchanged.

**Undo.** Both final reverts are bit-equal to the restored state at every integer frame and at subframes 0.25/0.5/0.75.
Their values equal the restored values.

**Hashes:**

| | S1 | S3 |
|---|---|---|
| Baseline `.blend` (unchanged) | `0803ab2bf31210fb…` | `e1cfbae49e3af0e7…` |
| Accepted R1 `.blend` | `ca0ce1aa512a1ce22959298a860747b83c36c3c430ddc6fcbca1ad79471901d4` | `70fdb447a5922ca076da30bc045f5168e489410f02d405f396efb704e2107c7b` |
| Final reverted `.blend` | `25706d4eef32891288a0ee6f21db3c199aeb2583e73cf3b007ec878210dbdfa3` | `6bd570e2b7979d53f1ecfd2f868ffdc4565ca08bc85d5f8a23b82e68af0c51b2` |
| Final `refs.json` | `9d43f147…` (gen 4) | `4c703c71…` (gen 4) |
| `refs.legacy_w124.json` (= W124 `refs.json` bytes) | `fd7bda12…` | `ed11373b…` |

**Unrecovered orphans.** None. What remains is all recorded and never authority:

- S1: the staged copy of the aborted ACCEPT, and the late orphan-Blender files in the aborted REVERT stage dir;
- S3: an empty aborted stage dir, and 8 quarantined files;
- two unreferenced but valid content-addressed states, which were later reused by the successful retries.

## 7. Native 30 fps A/B, quality, FBX (unchanged baselines)

**A/B review** of the recovered accepted R1 edit: original source | baseline `.blend` | accepted `.blend`.

| Source | Frames written / decoded | Fps | Duration | SHA-256 |
|---|---|---|---|---|
| S1 | 240 / 240 | 30.0 | 8.000 s | `9b058d36…` |
| S3 | 154 / 154 | 30.0 | 5.133 s | `4a59a5f5…` |

Both videos are **byte-identical to W124's**, so W124's agent review and its negatives apply unchanged; no new visual
review was done:

- the S1 left arm disagrees with the source (the kettlebells are held at the chest);
- the S3 close-up does not show the occluded wrist;
- 20–22 mm one-frame release jumps at the pin boundaries;
- the S3 elbow reaches 6.4°, close to straight;
- the rig is partial: right arm at rest, no root translation or foot contact.

**A correct transaction is not evidence of animation quality.**

**FBX** (existing exporter and default importer, unchanged), on the R1 state:

| | S1 | S3 |
|---|---|---|
| Integer-frame drift (max) | 3.90 mm | 6.77 mm |
| Subframe deviation (max) | 0.69 m | 0.21 m |
| Interpolation | 519 LINEAR | 519 LINEAR |
| `frame_end` | 250 | 250 |
| Round-trip | **FAIL** | **FAIL** |

These are identical to W124. Nothing was repaired. Export left the `.blend` unchanged.

## 8. Tests (RUN / PASS / FAIL / NOT RUN)

| Set | Kind | Result |
|---|---|---|
| `tests/test_experimental_action_txn.py` (58) | 4 unit, 1 process (lock/kill), 53 **real Blender BaseRig** with synthetic values through separate CLI processes (fault matrix 38, kill 2 + orphan 1, concurrency 2, normal path / tamper / legacy / reject / real Blender failure 10) | **58 PASS** |
| `tests/test_experimental_video_action.py` (35, W124, unmodified) | unit + MOCK FK + 1 real Blender | **35 PASS** (also 35 PASS before changes) |
| Regression: `test_cli`, `test_gui_app`, `test_blender_executor`, `test_motion_graph`, `test_keyframe_collapse`, `test_w116_semantic_pin`, `test_w118_transition_edit`, `test_w105_rig_utility`, `test_fk_solver`, `test_canonical_pose_parity`, `test_retarget_solver`, `test_w121_independent_transfer`, `test_w122_realtime_utility`, `test_render_blender_animation_video`, `test_w117_temporal_pin`, `test_w120_factor_attribution` | existing | **226 PASS** (same files and count as W124) |
| Real S1/S3 saved-artifact replay | real Blender | **RUN**: every declared step as expected, 27/27 audits pass |
| Old-code reproduction of W3 | real | **RUN**: reproduced |
| FBX export + reimport | FBX | **RUN**: round-trip FAIL (characterisation) |

The JUnit files are `results/pytest/focused.xml` (`b701c91b…`) and `regression.xml` (`8b0a4892…`).

**Isolation:**

- The default parser digest is unchanged (W124 test).
- `app.cli` loads no heavy module; the new helper is stdlib only.
- `gui.py` does not reference the slice.

**Not run:**

- the full repository suite;
- a fresh live-perception E2E (the W124 saved artifacts were replayed instead);
- POSIX, power-loss and OS-crash tests;
- other filesystems;
- independent re-execution by anyone else.

**Failed:** none in the frozen run. Two defects were found and fixed **before** the instrument commit and declaration:

- a test glob also matched the new `.stage` directory;
- the harness audit returned a list instead of a boolean.

## 9. Negative findings

- **`proposal.json` is not sealed.** POST-OUTCOME EXPLORATORY, on a synthetic fixture, not a gate:
  `results/exploratory/proposal_tamper/result.json` (`8f679902…`).
  - The UNREACHABLE candidate's `proposal.json` decision was hand-edited to ACCEPT.
  - `commit --decision accept` then **published** that gate-rejected candidate: hard-pin residual 136 mm, exit 0.
  - The new publication checks bind the `.blend` and footprint to the proposal, not the gate decision. W124 had the same
    exposure.
  - Not fixed: a post-outcome code change would alter the evaluated instrument. Documented in the usage doc.
- **Old window W3 was real**, reproduced on W124 code with authentic artifacts. W1, W2 and W4–W7 were not reproduced on
  the old code.
- **Staging directories accumulate.** Directories of aborted transactions are kept (one held a full staged `.blend`
  copy). Only empty ones are removed after finalize.
- **Valid unreferenced states remain** in `lineage/states/` after a rollback. They are visible in `status` as
  `unreferenced_states_not_accepted`.
- **Lock release is asynchronous on Windows** after a kill. Measured at 0.05 s here, but unbounded by the OS. The first
  command after a kill may meet `BLOCKED_RUN_LOCKED` and must be re-run.
- **Commit is slower:** 1.8 s versus under 1 s in W124, because of the fresh Blender readback of the accepted copy.
- **No power-loss guarantee on Windows** (no directory fsync).

## 10. Unresolved uncertainties

- Behaviour under power loss, OS crash, full disk, antivirus or file-indexer locks held for longer than the bounded
  retry, network drives, and POSIX.
- A long-lived orphan Blender keeping files open could make a later `recover` quarantine fail with a sharing violation.
  It would fail closed and could be re-run. Not observed.
- Whether users understand `status` / `recover` without guidance. No human used the workflow.
- Behaviour of tampering classes other than those tested. `proposal.json`, `bindings.json` and `request.json` are
  unsealed.

## 11. Performance, maintenance, rollback

**Wall clock (Main):**

| Operation | Time |
|---|---|
| propose | 2.4–2.7 s |
| commit accept | 1.8 s |
| reject | 0.8–0.9 s |
| revert | 2.0 s |
| recover | 0.3–0.8 s |
| adopt | 0.2–0.3 s |
| review | 10–17 s |
| fbx | about 2 s |
| Focused tests | 159 s |
| Regression tests | 12 s |

**Storage:** 463 MB of runs and 514 MB of results, under `C:/Users/dna10/animcv-w125`, outside the repository.

**Maintenance.** Fault injection is active only when `ANIMCV_EXP_TXN_FAULT` is set; it is inert otherwise (tested). The
v2 refs schema and journal are slice-private. The frozen W116 `Store` is subclassed, not edited.

**Rollback.** Revert the code changes of `78f3533`. v2 `refs.json` still carries the W124 `"head"` key, so the W124
reader can locate the head of a W125 run. Adopted runs also keep their original refs as `refs.legacy_w124.json`.

## 12. Evidence, reproduction, changed files

**Evidence.** All under `C:/Users/dna10/animcv-w125`; local, not published.
`results/evidence_manifest.json` lists 5,547 files and was written before this Worklog. SHA-256
`9da2652173f1d25735bc9e8eb7011765c962b1b4debb119f967d1b076055d40c`.

Key files:

| File | SHA-256 |
|---|---|
| `declare/declaration.json` | `cae95047…` |
| `preflight/preflight.json` | `1e6ef08e…` |
| `run/run_index.json` | `917f6c0e…` |
| `pytest/pytest.json` | `0fd5f489…` |
| `assess/assessment.json` | `2f887a4b…` |

Runs are `runs/S1_replay` and `runs/S3_replay`.

**Reproduction** (Git Bash, repository root at `78f3533`; every stage refuses an existing output):

```
PY=C:/Projects/AnimCV/.venv_build_full/Scripts/python.exe; R=C:/Users/dna10/animcv-w125/results; export PYTHONPATH=src
D="--declaration $R/declare/declaration.json --declaration-sha cae950472de76dc5946f09a16b48eb5cf6e763fed109eba565c07078dfa4aabe"
$PY -X utf8 scripts/w125_txn_harness.py preflight --out-dir $R/preflight
$PY -X utf8 scripts/w125_txn_harness.py declare   --out-dir $R/declare
$PY -X utf8 scripts/w125_txn_harness.py run       --out-dir $R/run $D --preflight $R/preflight/preflight.json
$PY -X utf8 scripts/w125_txn_harness.py pytest    --out-dir $R/pytest $D
$PY -X utf8 scripts/w125_txn_harness.py assess    --out-dir $R/assess --run-dir $R/run --pytest-dir $R/pytest $D
```

**User-visible interruption and recovery sequence** (S3 replay, steps s05–s06):

```
ANIMCV_EXP_TXN_FAULT=after_publish ANIMCV_EXP_TXN_FAULT_MODE=exit $PY -m app.cli experimental-video-action commit --run runs/S3_replay --request-id R1 --decision accept
$PY -m app.cli experimental-video-action status  --run runs/S3_replay
$PY -m app.cli experimental-video-action recover --run runs/S3_replay
$PY -m app.cli experimental-video-action status  --run runs/S3_replay
```

The visible results, in order:

1. The commit exits 86 with no message.
2. The first `status` shows head `17911f0b…`, `recovery_required: true` and a pending ACCEPT with `published: true`.
3. `recover` prints ROLLED_FORWARD.
4. The second `status` shows `recovery_required: false` and `edits.R1.decision = ACCEPTED`.
5. The accepted `.blend` is `accepted_02_17911f0b74cf.blend` (`70fdb447…`).

**Changed files (repository):**

- New:
  - `src/app/experimental_action_txn.py`
  - `tests/test_experimental_action_txn.py`
  - `scripts/w125_txn_harness.py`
  - this Worklog
- Modified:
  - `src/app/experimental_video_action.py`: transactions, `recover`, `adopt`, extended `status`, atomic proposal and
    timeline writes.
  - `docs/EXPERIMENTAL_VIDEO_ACTION_WORLD_PIN.md`: W125 section and limitations.
  - `docs/README.md`, `docs/00_마스터_플랜.md`, `docs/01_프레임워크_현황.md`: dated 2026-11-09 W125 notes, labelled
    experimental and Agent-run.
- Unchanged:
  - `docs/next_batch.md` and every historical Worklog;
  - `scripts/batch_automation.py` and its test;
  - the W1xx scripts and frozen algorithms, the exporter and importer, the CLI registration (`src/app/cli.py`) and the
    GUI;
  - models, the rig, and the normative architecture documents.

AUTOMATION_BATCH_STATUS: COMPLETE
BATCH_NUMBER: 125
