# Worklog — Commit-Time Semantic Gate Integrity for the Reversible World Pin (2026-11-09)

Batch 126. A bounded Production-engineering batch on `arch/single_frame_first`, run unattended on Main: local RTX 5080,
Python `C:\Projects\AnimCV\.venv_build_full` (3.11.9), Blender 5.1.0, Windows 11 Pro 10.0.26200, NTFS. `docs/next_batch.md`
was read only.

Every number, test result and hash below is **AGENT-RUN** on Main. No Reviewer has re-executed any of it.

## 1. Official verdict and scope

**Official verdict: CASE C under the frozen CASE rule.**

- Source: `results/assess/assessment.json` (SHA-256 `854866fb6f587d674e1851bff5b1bd15507c9b38e3c120d9593b003abd383a09`).
- Gate results:
  - G1 (bypass closed), G3 (valid Blender cases), G4 (transactions), G5 (real replay), G6 (native rate) and G7 (regression
    / isolation): **PASS**.
  - G2 (negative-control matrix): **FAIL**, on one control only.
- The frozen rule allows B only if G1–G4 and G7 pass, so the rule yields **C**. This Worklog does not upgrade the verdict.

**What failed.** Control **N12**: `request.json` changed **and** its hash updated in the proposal, then resealed.

- The commit **refused** it (`REFUSED_SEMANTIC_AUTHORIZATION`, exit 2).
- Every no-publication invariant held: head and generation unchanged, no accepted `.blend`, no `commit.json`, no false
  ACCEPTED display, no input file repaired.
- What failed is the **frozen prediction of the primary refusal class**. Declared: `CANDIDATE_NOT_REPRODUCED`. Observed:
  `RECOMPUTED_GATE_REJECT`, with `CANDIDATE_NOT_REPRODUCED` and `PROPOSAL_GATE_RECORD_DIVERGES` also present.
- Cause: the frozen declaration contradicted itself.
  - Its own reason order puts `RECOMPUTED_GATE_REJECT` before `CANDIDATE_NOT_REPRODUCED`.
  - For N12 the recomputed pin is evaluated against the stored readback of the *original* candidate, so the recomputed gate
    rejects.
  - N17 has the same reason set and was correctly predicted as `RECOMPUTED_GATE_REJECT`.
- Neither the implementation nor the expectation was changed after the outcome.

**Reading the result against the Batch's own CASE text** (for the Reviewer; not a verdict change):

- The Batch defines C as "trustworthy commit-time semantic authorization is not demonstrated".
- It defines B as "a declared class … remains limited and transparently bounded".
- Here, every declared negative control was refused without publication, and both real sources behaved as declared.
- The only failure is classification-precedence prediction. The Agent states this; it does not select the verdict.

**What this evidence supports at most:** a bounded, opt-in commit-time semantic acceptance boundary under the declared
threat model (accidental or deliberate edits of run files, short of a consistent rewrite of the whole run).

**What it does not support:**

- cryptographic authenticity;
- animation quality or source fidelity;
- FBX portability;
- a Production default.

**Historical verdicts:** W121 D, W122 B, W123 B, W124 B and W125 A (transaction capability only) are untouched. No
historical Worklog or historical run file was edited.

## 2. Starting state, revisions, environment

| Item | Value |
|---|---|
| Instruction blob | `2a8d65b1b0716dbfa3ea72debe134e952e0cb513`, verified first |
| Starting commit | `a475c422477684d0284ea50d80c8c5315e5f6201`; equal to `origin/arch/single_frame_first` at fetch |
| W125 instruction blob (provenance only) | `94355c1c1cdaa48d5ab5db7a21926555a3990ebb` |
| Prospective matrix declaration | `C:/Users/dna10/animcv-w126/declare/matrix_declaration.json`, SHA-256 `aafbebed8d16477e96d252d93223eea757973cc85ff108c5be984b3beba86f94`. Written 2026-11-08T19:02:01Z, **before the first W126 test run**, `outcomes_inspected: false`. |
| Pre-outcome instrument commit | `4a048011afb220535e14a7d22714487f2b74d6bb`: code, focused tests, harness, usage doc |
| Harness fix commit | `69e78795…` (harness only; see §8) |
| Real-run declaration | `results/declare_v2/declaration.json`, SHA-256 `d5ae2f772a06be0983952635b65cd9c8cd0ab4c9c500c2a9bc3f99634c18fb58` (instrument hashes; refuses to run if any changed) |
| Final source revision | `69e78795…` for code, tests and harness. The commit adding this Worklog changes documentation only. The COMPLETE marker is a separate, final commit. |
| Machine | Main only. No LabServer, no training, no network use beyond `git fetch/push`. |

**Pre-outcome hashes.** At the matrix declaration, the implementation `src/app/experimental_video_action.py` had LF
SHA-256 `1f2c09cc9f2685385dd943328f9b4ce94987687a759dc9d51f94cccb012af022`. The final committed module is the **same**. The
implementation was never changed after any W126 outcome was seen.

**The control tables did not change** after the declaration: `NEG`, `RECORDED`, `VALID`, `RECOVERY` and the reason order
are identical to the declared ones (checked programmatically). One test **assertion** outside the tables did change (§8).

## 3. The W125 bypass: code explanation and reproduction

**Code path at `a475c42`:**

- `_propose` writes `edits/<id>/proposal.json` with a mutable `decision` plus `constraint`, `footprint`, `display`,
  `candidate_blend` and `pin_frames`.
- `_commit` branches on `prop["decision"]["decision"]` and passes that decision to W116 `Store.commit`.
- `_publication_checks` checks that the accepted copy is bit-equal to the candidate and that its footprint equals the
  proposal's. Both comparisons are against the **same editable proposal**. It computes the pin residual but never gates on
  it.
- `run.json`'s `immutable_files` does not guard `baseline/bindings.json` or `edits/<id>/request.json`.

**Reproduced with the starting-commit code** (a `git archive a475c42` export; `results/preflight/preflight.json`, SHA-256
`33ecf08b…`):

| Fixture | Tamper | Result on old code |
|---|---|---|
| Real-Blender BaseRig synthetic fixture (the W125 fixture builder), proposal `REACH`: gate REJECT, INFEASIBLE_REQUEST/UNREACHABLE | `proposal.json` decision → ACCEPT, nothing else changed | `commit --decision accept` exit 0; head moved to the candidate; published hard-pin residual **136.16 mm**. Matches W125's exploratory 136 mm. |
| Copy of the W125 S1 replay; new request: left wrist 210–218, capture 210, +1 m up (gate REJECT UNREACHABLE, with a real candidate `.blend`) | same | exit 0; head `1994e410…` → `73452512…`; published residual **557.27 mm** on real saved-artifact data |

## 4. Trust boundary, binding strategy and compatibility (frozen before outcomes)

**Threat model.**

- **In scope:**
  - edits to `proposal.json` (any field, with or without a recomputed seal), `request.json`, `bindings.json`, the
    candidate state, `.blend` or readback;
  - a stale parent;
  - edits while a transaction holds the lock;
  - unsealed pre-W126 proposals;
  - tampered journals.
- **Out of scope, not claimed:**
  - an actor who rewrites the whole run consistently, including `run.json`, its immutable-file record, sealed `refs.json`,
    the frozen code or the slice module. All seals are unkeyed SHA-256: **consistency checks, not authentication.**
  - a malicious Blender binary;
  - power loss and OS crash (W125 limits).

**The decision authority is a recomputation, not a file.** `commit --decision accept` re-derives the frozen gate from:

1. the original request **bytes**;
2. the head state and readback verified through sealed `refs.json` (W125 `verify_head`);
3. bindings **re-derived from the immutable baseline** (baseline state, the baseline readback bound to the immutable
   baseline `.blend` SHA, the sidecar and the known rig files) with the frozen W116 `verify_binding`;
4. the unchanged frozen W116 `validate_request`, `captured_target`, `resolve_frame`, `evaluate_candidate` and `decide`, the
   W118 `derive_footprint`, `display_spec`, `check_display` and `decide_x`, and the slice's subframe rule.

`propose` and `commit` call **the same function** (`_gate_record`), so both use one code path.

**The recomputation runs twice:**

- **PRE_STAGING:** on the hash-bound stored candidate readback, before any journal.
- **FRESH_READBACK:** inside the W125 transaction, on a **fresh Blender process readback of the staged accepted copy**.

Each run must give a decision of ACCEPT. The recomputed candidate must have the same content id and canonical bytes as the
proposal's candidate. All 19 gate fields of the sealed proposal must be canonically equal to the recomputation.

**Pre-publish re-hash.** Immediately before the W125 compare-and-swap publication, every decision input is re-hashed:
proposal, request, bindings, candidate state, `.blend` and readback.

**Recomputed versus hash-bound:**

| Recomputed at commit | Hash-bound at commit (consistency only) |
|---|---|
| request validation | proposal seal and gate-view digest |
| pin target re-capture | request exact bytes |
| bindings | bindings content, compared with the re-derived bindings |
| resolver statuses and candidate state | parent head evidence, compared with sealed refs |
| W116 evaluator: hard pin ≤ `TOL_M`, scope, locality, lengths, root, joint limit | candidate state file, `.blend` and readback |
| W118 footprint, display and decision | immutable baseline, manifest and sidecar, source clip (`load_run`) |
| subframe rule | frozen algorithm sources (now also required at commit) |
| | all inputs re-hashed again before publication |

**Declared precedence:**

| Step | Check | Outcome |
|---|---|---|
| P0 | run guards | existing STOP |
| P1 | proposal missing or already decided | existing STOP |
| P2 | unsealed legacy proposal (accept only) | `BLOCKED_LEGACY_PROPOSAL_REQUIRES_REPROPOSE` |
| P3 | candidate content address | W125 code `INVALID_CANDIDATE_TAMPERED` |
| P3b | candidate `.blend` SHA | W125 code `INVALID_CANDIDATE_BLEND_TAMPERED` |
| P4 | provenance | `REFUSED_SEMANTIC_AUTHORIZATION` |
| P5 | sealed gate REJECT or no candidate | recorded REJECT `GATE_REJECTED_CANNOT_ACCEPT`, exit 0, as in W125 |
| P6 | stale parent | recorded `STALE_PARENT`, exit 0, as in W125 |
| P7 | PRE_STAGING recomputation | refusal |
| P8 | FRESH_READBACK recomputation | refusal, rolled back |
| P9 | W125 publication checks | as in W125 |
| P10 | PRE_PUBLISH re-hash | refusal, rolled back |
| P11 | compare-and-swap | publication |

Within a refusal, the first reason present in the declared order is the primary class.

**Identity policy:**

- `request.json`: **exact bytes**. A reformat is refused; re-propose.
- `proposal.json` and `bindings.json`: **content**. A re-indent or key reorder is benign.

**Compatibility:**

- Pre-W126 (unsealed) proposals stay inspectable. `status` shows them as `LEGACY_UNVERIFIED:<decision>`.
- `accept` of such a proposal stops with `BLOCKED_LEGACY_PROPOSAL_REQUIRES_REPROPOSE` and changes nothing. It is never
  grandfathered. Explicit `reject` stays possible.
- Historical ACCEPTED edits are shown as `NOT_ESTABLISHED (accepted before W126; historical, not reclassified)`. Their
  `commit.json` files are not rewritten (verified byte-equal after the run on S1 and S3).

**Display:**

- `status` shows a proposal's gate only if its provenance verifies. Otherwise it shows `UNTRUSTED (<reason>)` or
  `REFUSED_AT_COMMIT (<primary>)`, and the footprint is shown as "NOT SHOWN (unverified)".
- An explicit REJECT of an unverified proposal is recorded with `proposal_integrity`, skips the display check, and draws
  `timeline_final.png` titled `UNTRUSTED PROPOSAL`.

**Recovery:**

- `recover` never creates an authorization.
- A published W126 ACCEPT whose journal does not record `authorized: true` stops with `BLOCKED_DIVERGENT_JOURNAL`.
- A published pre-W126 journal is rolled forward and marked `NOT_ESTABLISHED`.

**Refusal evidence.** Each refusal writes a **new** file `edits/<id>/authorization_refusals/<time>_<uuid>.json` with stage,
primary and reasons. Nothing is overwritten or repaired.

## 5. Prospective negative-control matrix (focused tests, real Blender BaseRig, synthetic values)

**Fixture.** The W125 8-frame **synthetic-value** run on the **real** W105 BaseRig, built with real Blender. It is not
video. Extra proposals were added: `NOOP`, `WB` (right-wrist binding id) and `UA` (only `lowerarm_l` authorized).

**Harness.** The user CLI is run in separate processes. The MOCK case N23 runs in-process.

**Invariants checked for every refused control:**

- refs record byte-identical, so head and generation are unchanged;
- no accepted `.blend`;
- no `commit.json`;
- every ACCEPT journal ABORTED;
- **no pre-existing file changed**, so tampered bytes are left as they are and nothing is repaired;
- `status` gate ≠ ACCEPT, decision UNDECIDED, `semantic_authorization` null;
- a refusal record carrying the declared primary class.

**Refused controls** (the "Expected" column is the frozen primary class or STOP code):

| ID | Control | Expected | Result |
|---|---|---|---|
| N01 | **W125 bypass**: REACH decision → ACCEPT, no reseal | `PROPOSAL_SEAL_INVALID` | PASS |
| N02 | REACH forged decision + status, resealed | `RECOMPUTED_GATE_REJECT` | PASS |
| N03 | REACH forged low residual + `pins_satisfied` + decision, resealed | `RECOMPUTED_GATE_REJECT` | PASS |
| N04 | P1 residual set to 0, resealed | `PROPOSAL_GATE_RECORD_DIVERGES` | PASS |
| N05 | P1 pin target +5 cm, resealed | `PROPOSAL_GATE_RECORD_DIVERGES` | PASS |
| N06 | P1 pin frames shortened, resealed | `PROPOSAL_GATE_RECORD_DIVERGES` | PASS |
| N07 | P1 constraint digest altered, resealed | `PROPOSAL_GATE_RECORD_DIVERGES` | PASS |
| N08 | P1 authorized channels widened, resealed | `PROPOSAL_GATE_RECORD_DIVERGES` | PASS |
| N09 | P1 binding id → right wrist, resealed | `PROPOSAL_GATE_RECORD_DIVERGES` | PASS |
| N10 | P1 footprint shrunk, resealed | `PROPOSAL_GATE_RECORD_DIVERGES` | PASS |
| N11 | `request.json` changed, provenance not updated | `REQUEST_BYTES_CHANGED` | PASS |
| **N12** | `request.json` changed **and** hash updated, resealed | `CANDIDATE_NOT_REPRODUCED` | **FAIL (prediction)**: refused, primary `RECOMPUTED_GATE_REJECT`; all invariants held (§1) |
| N13 | substituted binding record in `bindings.json` | `BINDING_RECORD_CHANGED` | PASS |
| N15 | `request.json` reformatted (same JSON) | `REQUEST_BYTES_CHANGED` | PASS |
| N16 | candidate state values edited | `INVALID_CANDIDATE_TAMPERED` | PASS |
| N17 | REACH candidate state, `.blend` and readback substituted into P1, all hashes consistent | `RECOMPUTED_GATE_REJECT` | PASS |
| N18 | candidate `.blend` byte flip | `INVALID_CANDIDATE_BLEND_TAMPERED` | PASS |
| N19 | REACH `.blend` placed under P1 with readback and hashes made consistent | `FRESH_READBACK_GATE_REJECT` | PASS: caught only by the fresh readback; transaction ABORTED, head unchanged |
| N21 | status PASS with decision REJECT, resealed | `PROPOSAL_DECISION_STATUS_CONFLICT` | PASS |
| N23 | **MOCK** resolver status NON_CONVERGED with a real candidate file; forged ACCEPT, resealed | `RECOMPUTED_GATE_REJECT` (recomputed class SOLVER_FAILURE) | PASS |
| N24 | candidate readback deleted | `CANDIDATE_READBACK_MISSING` | PASS |
| N25 | readback values altered | `CANDIDATE_READBACK_CHANGED` | PASS |
| N25b | REACH readback forged to show the pin satisfied, hashes updated, forged ACCEPT | `RECOMPUTED_GATE_REJECT`: the resolver's UNREACHABLE comes from the request, not the readback | PASS |
| N26 | stale parent **and** unsealed edit | `PROPOSAL_SEAL_INVALID` (provenance before stale) | PASS |
| N27 | request changed **and** proposal edited | `PROPOSAL_SEAL_INVALID` | PASS |
| N28 | `.blend` flipped **and** proposal edited | `INVALID_CANDIDATE_BLEND_TAMPERED` (W125 check first) | PASS |
| N29 | proposal stripped to the pre-W126 format with decision ACCEPT | `BLOCKED_LEGACY_PROPOSAL_REQUIRES_REPROPOSE`; explicit reject then recorded with `LEGACY_UNSEALED_PROPOSAL` | PASS |
| N30 | `request.json` edited **while** the accept transaction holds the lock, after the fresh readback | `INPUTS_CHANGED_DURING_TRANSACTION` | PASS: rolled back, 3 promoted files quarantined |
| N32 | wrong-binding proposal (no candidate), forged ACCEPT + PASS, resealed | `PROPOSAL_DECISION_STATUS_CONFLICT` | PASS |
| N33 | unauthorized-DOF proposal (no candidate), forged ACCEPT, resealed | `PROPOSAL_DECISION_STATUS_CONFLICT` | PASS |
| N34 | parent (head) readback altered | `BLOCKED_HEAD_UNVERIFIED` (W125) | PASS |

**Recorded cases.** These are W125 behaviour, preserved; exit 0 and the head never moves.

| ID | Case | Recorded result | Result |
|---|---|---|---|
| N20 | stale parent | `STALE_PARENT` | PASS |
| N22 | honest UNREACHABLE proposal with the accept flag | `GATE_REJECTED_CANNOT_ACCEPT` | PASS |
| N31 | tampered proposal, explicit REJECT | REJECTED with `proposal_integrity` `PROPOSAL_SEAL_INVALID`; no display; `status` shows UNTRUSTED | PASS |

## 6. Valid Blender cases, recovery and W125 parity

| ID | Case | Result |
|---|---|---|
| V01 | Valid nonzero pin (P1) | ACCEPTED; `semantic_authorization.authorized = true`; pre-staging and fresh gates ACCEPT; fresh-readback residual ≤ 1 mm; actual changed frames ⊆ pin frames; bones `lowerarm_l` and `upperarm_l` |
| V01 (independent) | A new Blender process run by the **test**, not the CLI, reopens the published head `.blend` | value digest and SHA equal; residual ≤ 1 mm; changed frames equal the authorization's; subframe changes inside the pin frames; `status` shows `AUTHORIZED_AT_COMMIT` |
| V02 | No-op | ACCEPT `TRIVIALLY_SATISFIED_NOT_EVIDENCE`; empty footprint. **Not evidence of improvement.** |
| V03 | Explicit REJECT, then the accept flag | REJECTED; then `REFUSED_ALREADY_DECIDED` |
| V04 | Exact REVERT | bit-equal at integer frames and subframes |
| V05 | Benign reserialization of `bindings.json` and `proposal.json` | ACCEPTED; `bindings_file_bytes_equal_proposal = false` is reported |
| R01 | Published W126 ACCEPT whose journal is resealed with `authorized = false` | `recover` → `BLOCKED_DIVERGENT_JOURNAL`; nothing invented |
| R02 | Published journal without `semantic_authorization` (pre-W126 shape) | rolled forward; `commit.json` `authorized: null` with a NOT ESTABLISHED note |

**W125 parity.** The W125 suite was re-run **in full**, not as a subset: **58/58 PASS** with the W126 code. That covers the
38-case fault matrix, kill / restart / orphan Blender, concurrency, tamper, legacy, reject and real Blender failure. The
W124 suite also passed: **35/35**.

## 7. Real S1/S3 saved-artifact replays (Agent-run)

**Label.** SAVED-ARTIFACT REAL-VIDEO REPLAY. These are copies of the W125 final `S1_replay` and `S3_replay` runs, verified
file by file against the W125 evidence manifest `9da26521…`: 2169 and 1390 files, 0 differences. The clips match their
declared SHA-256.

**This is not fresh perception.** S1 and S3 are the correlated W121–W125 engineering sources, not new generalization
evidence.

**Requests:**

- **W126V:** hold the left wrist at its evaluated position. S1: frames 210–218, capture 210. S3: frames 41–49, capture 41.
- **W126R:** the same window, target +3 cm up.
- **W126T:** the same window, target +1 m up.

**Audits.** After **every** step, a separate harness process:

- re-read sealed `refs.json` and checked head → state → `.blend` → readback;
- reopened the head `.blend` in a **fresh Blender process** and required the value digest, every integer frame and every
  subframe to be bit-equal;
- checked that no ACCEPTED edit lies outside the published lineage and that no unpublished file sits in `accepted/`;
- checked that no W126 edit is displayed as ACCEPT without verified integrity;
- checked the immutable baseline and the source clip;
- for an accepted W126 head, recomputed the residual, the W118 footprint and the subframe report from the fresh readback.

**All 25 steps** (S1: 13, S3: 12) met their frozen expectation, and all audits passed. Logs: `results/run/S1_steps.json`
(`81850a64…`) and `S3_steps.json` (`408d8c78…`).

| Step | S1 | S3 |
|---|---|---|
| status | legacy proposals `LEGACY_UNVERIFIED`; historical ACCEPTED edits (R1, `S1_E1`) `NOT_ESTABLISHED`, not reclassified | the same (R1, `S3_E1`) |
| accept a legacy proposal (`S1_N_UNAUTH`) | `BLOCKED_LEGACY_PROPOSAL_REQUIRES_REPROPOSE`; no file changed | **N/A**: S3 has no undecided pre-W126 proposal |
| propose W126V / W126R | gate ACCEPT, residual 6.4e-7 / 6.2e-7 m | gate ACCEPT, residual 5.6e-7 / 5.2e-7 m |
| propose W126T | gate REJECT UNREACHABLE; real candidate; residual **557.3 mm** | gate REJECT UNREACHABLE; residual **12.4 mm** |
| W126T, decision hand-edited (the W125 bypass) | `REFUSED_SEMANTIC_AUTHORIZATION` PROVENANCE / `PROPOSAL_SEAL_INVALID`; head and generation 4 unchanged; status gate UNTRUSTED | the same |
| W126T, forged + resealed | REFUSED PRE_STAGING / `RECOMPUTED_GATE_REJECT` (+ `PROPOSAL_GATE_RECORD_DIVERGES`); head unchanged | the same |
| W126T restored, explicit REJECT | REJECTED | REJECTED |
| W126R explicit REJECT | REJECTED | REJECTED |
| W126V accept | ACCEPTED, generation 5, `authorized: true` | ACCEPTED, generation 5 |
| W126V hard-pin residual (fresh readback, audit) | **6.36e-7 m** | **5.61e-7 m** |
| W126V actual changed frames / bones (Blender evaluator) | 211–218 / `lowerarm_l`, `upperarm_l` | 42–49 / the same |
| W126V subframe changes inside the pin frames; subframe residual | yes; 6.36e-7 m | yes; 5.61e-7 m |
| review, 30 fps | 240 frames written / 240 decoded, 8.000 s | 154 / 154, 5.133 s |
| revert | exact undo (integer and subframe bit-equal), generation 6 | the same |

**Notes:**

- On S1, W126T's resolver clamps to maximum reach, which leaves a 557 mm residual. On S3 the clamped residual is 12.4 mm. Both
  are UNREACHABLE and both exceed the unchanged 1 mm experimental diagnostic.
- W126V's state values are canon-equal to W125's R1 accepted edit. The Blender value digests are equal too (S1 `3a3eb7eb…`,
  S3 `9c49233b…`).
- The accepted copies are `accepted_03_674ba278e9a0.blend` (`7e2ab165…`, S1) and `accepted_03_fd249fbb203f.blend`
  (`702ead26…`, S3).
- The final reverted `.blend` files are `1ec3cda7…` (S1) and `0de50493…` (S3). The baseline `.blend` files are unchanged
  (`0803ab2b…`, `e1cfbae4…`).
- Historical `commit.json` files were byte-equal after the run. The W125 originals were unchanged after the run.

**30 fps A/B.** The review videos are **byte-identical to W124/W125**: S1 `9b058d36…`, S3 `4a59a5f5…`. That is expected,
since the edited values and the restored parent values are equal. No new visual review was done. The W124/W125 negatives
apply unchanged:

- the S1 left arm disagrees with the source;
- the S3 close-up does not show the occluded wrist;
- 20–22 mm one-frame release jumps at the pin boundaries;
- the S3 elbow is near straight (6.4°);
- the rig is partial: right arm at rest, no root translation or foot contact;
- subject identity and GT are unverified.

**A correct authorization is not evidence of animation quality.**

**FBX** (existing exporter and importer, unchanged), on the W126V head:

| | S1 | S3 |
|---|---|---|
| Integer-frame drift (max) | 3.90 mm | 6.77 mm |
| Subframe deviation (max) | 0.69 m | 0.21 m |
| Interpolation | 519 LINEAR | 519 LINEAR |
| Scene range | not carried | not carried |
| Round-trip | **FAIL** | **FAIL** |

These are the same figures as W124/W125. Nothing was repaired, and export left the `.blend` unchanged.

## 8. Tests (RUN / PASS / FAIL / NOT RUN) and post-outcome changes

| Set | Kind | Result |
|---|---|---|
| `tests/test_experimental_gate_integrity.py` (41) | 2 unit, 38 real Blender BaseRig with synthetic values (CLI in separate processes), 1 MOCK solver status (N23) | **40 PASS, 1 FAIL** (N12, prediction) |
| `tests/test_experimental_action_txn.py` (58, W125, unmodified) | real Blender, process | **58 PASS** |
| `tests/test_experimental_video_action.py` (35, W124, unmodified) | unit, MOCK FK, 1 real Blender | **35 PASS** |
| Regression (the same 16 files as W124/W125) | existing | **226 PASS** |
| Real S1/S3 saved-artifact replay | real Blender | **RUN**: 25/25 steps, all audits pass |
| Old-code bypass reproduction | real Blender, fixture + S1 | **RUN**: reproduced |
| FBX export and reimport | FBX | **RUN**: FAIL (characterisation) |

**Baseline before changes.** The W124 and W125 suites ran on a `git archive` of the starting commit: **93/93 PASS**.

**Official files.** JUnit: `results/pytest/focused.xml` (`7eb56afb…`) and `regression.xml` (`a48e1ed4…`). Command: the
harness `pytest` stage, with `PYTHONPATH=src` and `-p no:cacheprovider`.

**Isolation.** These are the W124 tests, which passed:

- the default parser digest is unchanged;
- `app.cli` and the default help load no experimental dependencies;
- `gui.py` does not reference the slice.

`src/app/cli.py` and `experimental_action_txn.py` are unchanged. No Production schema was added.

**Post-outcome changes, all disclosed:**

1. **Test assertion, N19.** I had asserted that the aborted transaction quarantined files. By the unchanged W125 design, a
   refusal *before promotion* leaves the staged copy in the never-authoritative `txn/<id>.stage/` directory, and
   `quarantined` is empty. The assertion was corrected to state that. The control's frozen exit, code and primary class were
   met and are unchanged.
2. **Harness, after the first real-run attempt.** The attempt stopped at its first, read-only step (`status`): the harness
   parsed a stdout truncated to 6000 characters. The harness fix also binds the FBX report to the current head's directory,
   because W124 runs carry an older `head_*` directory. The fix is commit `69e7879`, and the run was re-declared as
   `declare_v2`. The aborted attempt's run copy is kept as `runs_attempt1_harness_defect/`: byte-equal to W125, 0
   differences.

No implementation code changed after the matrix declaration.

**Not run:**

- the full repository suite;
- fresh live perception;
- POSIX, power-loss and OS-crash tests;
- adversaries able to rewrite the whole run;
- independent re-execution by anyone else;
- a new human visual review.

## 9. Negative findings

- **N12 prediction error** (§1). The frozen declaration was internally inconsistent: N12's expected primary class
  conflicted with its own reason order. The implementation follows the order. Safety was not affected, but the declared
  matrix was not fully met.
- **`request.json` is byte-bound.** A harmless reformat (N15) is refused and requires a re-propose. This is strict by
  declaration and may surprise users.
- **Legacy proposals cannot be accepted.** All W124/W125 undecided proposals must be re-proposed. Only S1 had one
  (`S1_N_UNAUTH`, a gate REJECT anyway).
- **Seals are unkeyed.** A consistent forger defeats every hash check. The control is the recomputation (N02/N03/N17/N19/
  N25b show that consistent forgeries are still refused). A forger who also rewrites `run.json`, the baseline record, sealed
  refs or the code is not covered.
- **The baseline readback is not guarded on the synthetic fixture.** It is not in the fixture's `immutable_files` (the
  real runs guard it). `recompute_bindings` still requires its `blend_sha256` to equal the immutable baseline `.blend`.
- **Refusal records and aborted staging directories accumulate.** They are never deleted.
- **A sealed REJECT is trusted without recomputation** (P5). Trusting a REJECT is fail-safe, but it means a forged
  ACCEPT → REJECT is recorded as a user decision without being detected as a forgery. N32/N33 are caught only through the
  decision/status consistency rule.

## 10. Unresolved uncertainties

- Whether every gate field is bit-deterministic across machines and NumPy versions. Recomputation equality was exact on
  Main; another machine could produce false divergences, which would fail closed.
- Behaviour of tamper classes outside the matrix, such as edits to `run.json`, the sidecar or the rig files. `load_run`
  and the rig check cover some of these; others are out of scope.
- Whether users understand `REFUSED_SEMANTIC_AUTHORIZATION` and the re-propose procedure. No human used the workflow.
- Whether the Reviewer reads the outcome as C (the frozen rule) or B (the Batch's text) (§1).

## 11. Performance, storage, maintenance, rollback

**Extra commit-time cost (in-process timing, Main):**

| Component | S1 | S3 |
|---|---|---|
| Binding re-derivation | 0.17 s | 0.04 s |
| Integrity checks | 0.05 s | 0.04 s |
| One gate recomputation (run twice) | about 0.06 s | about 0.06 s |

**End to end:**

- `commit accept`: 2.15 s (S1) and 1.81 s (S3), against 1.8 s in W125.
- Refusal before staging: 0.3–0.5 s.
- propose: 2.4–2.7 s, unchanged.

**Storage:**

- `proposal.json` grows by about 4.6 KB (provenance and seal).
- Each refusal record is about 1–2 KB.
- Under `C:/Users/dna10/animcv-w126`, outside the repository: `runs` 680 MB, `results` 306 MB, the aborted-attempt copy
  293 MB, and the `prechange` code export 2.2 GB (a `git archive` including example assets).

**Maintenance:**

- One extra opt-in module section in `experimental_video_action.py`. The proposal schema `animcv_exp_proposal_v2` and the
  refusal record are slice-private.
- The frozen W116/W118 files are unchanged (hash list verified, and now also required at commit).
- No new tolerance, constraint type or threshold.

**Rollback.**

- Reverting commit `4a04801` restores W125 behaviour.
- v2 proposals keep every W125 field, so W125 code can still read them.

## 12. Evidence, reproduction, changed files

**Evidence.** All under `C:/Users/dna10/animcv-w126`; local, not published.
`results/evidence_manifest.json` lists 9,751 files and was written before this Worklog. SHA-256
`f9139cd609232011e4dd82c7265b7c2dc8568291d5ee0011d7a7ee29db105e8c`.

Key files:

| File | SHA-256 |
|---|---|
| `declare/matrix_declaration.json` | `aafbebed…` |
| `results/preflight/preflight.json` | `33ecf08b…` |
| `results/declare_v2/declaration.json` | `d5ae2f77…` |
| `results/run/run_index.json` | `bea92cc4…` |
| `results/pytest/pytest.json` | `33c850e3…` |
| `results/assess/assessment.json` | `854866fb…` |

**Reproduction** (Git Bash, repository root at `69e7879`; every stage refuses an existing output):

```
PY=C:/Projects/AnimCV/.venv_build_full/Scripts/python.exe; R=C:/Users/dna10/animcv-w126/results; export PYTHONPATH=src
D="--declaration $R/declare_v2/declaration.json --declaration-sha d5ae2f772a06be0983952635b65cd9c8cd0ab4c9c500c2a9bc3f99634c18fb58"
$PY -X utf8 scripts/w126_gate_integrity_harness.py preflight --out-dir $R/preflight
$PY -X utf8 scripts/w126_gate_integrity_harness.py declare   --out-dir $R/declare_v2
$PY -X utf8 scripts/w126_gate_integrity_harness.py run       --out-dir $R/run $D
$PY -X utf8 scripts/w126_gate_integrity_harness.py pytest    --out-dir $R/pytest $D
$PY -X utf8 scripts/w126_gate_integrity_harness.py assess    --out-dir $R/assess $D --preflight $R/preflight/preflight.json --run-dir $R/run --pytest-dir $R/pytest
```

**Minimal end-user reproduction** (S1 replay, steps s05–s13; request files under `results/run/S1/`):

```
$PY -m app.cli experimental-video-action propose --run runs/S1_replay --request W126T.json   # gate REJECT UNREACHABLE
#   edit runs/S1_replay/edits/W126T/proposal.json: "decision": {"decision": "ACCEPT", ...}
$PY -m app.cli experimental-video-action commit --run runs/S1_replay --request-id W126T --decision accept
#   STOP REFUSED_SEMANTIC_AUTHORIZATION: PROVENANCE: PROPOSAL_SEAL_INVALID ...; head unchanged   (exit 2)
$PY -m app.cli experimental-video-action propose --run runs/S1_replay --request W126V.json   # gate ACCEPT
$PY -m app.cli experimental-video-action commit --run runs/S1_replay --request-id W126V --decision accept   # ACCEPTED, authorized
$PY -m app.cli experimental-video-action revert --run runs/S1_replay                         # exact undo
```

**Changed files (repository):**

- New:
  - `tests/test_experimental_gate_integrity.py`
  - `scripts/w126_gate_integrity_harness.py`
  - this Worklog
- Modified:
  - `src/app/experimental_video_action.py`: shared `_gate_record`, v2 sealed proposal with provenance,
    `recompute_bindings`, `proposal_integrity`, `_authorize`, legacy policy, refusal records, the fresh-readback gate and the
    pre-publish re-hash inside `_accept`, the `_staging_failed` code mapping, `_finish` / `_recover_one` authorization
    handling, untrusted timeline, and `status` display truth.
  - `docs/EXPERIMENTAL_VIDEO_ACTION_WORLD_PIN.md`: W126 section; the W125 "not protected" note is marked superseded.
  - `docs/README.md`, `docs/00_마스터_플랜.md`, `docs/01_프레임워크_현황.md`: dated 2026-11-09 W126 notes, labelled
    experimental, Agent-run, CASE C by the frozen rule.
- Unchanged:
  - `docs/next_batch.md` and every historical Worklog;
  - `src/app/experimental_action_txn.py`, `src/app/cli.py`, the GUI and the Blender stage;
  - the frozen W1xx scripts and algorithms, the FBX exporter and importer, models and the rig;
  - the normative architecture documents, `scripts/batch_automation.py`, and the W124/W125 test files.

AUTOMATION_BATCH_STATUS: COMPLETE
BATCH_NUMBER: 126
