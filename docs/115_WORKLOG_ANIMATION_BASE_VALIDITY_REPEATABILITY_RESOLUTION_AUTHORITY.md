# Worklog — Animation Base Validity: Repeatability, Numerical Resolution and Rig Authority (2026-11-05)

Batch 115. Evidence-only architecture diagnostic on `arch/single_frame_first`, run unattended on Main (local RTX 5080
host; no GPU inference was needed). No `src/` module, schema, model, threshold, rig, solver, FK/IK, FBX/Blender exporter,
renderer, default, W113/W114 script, test, Worklog, CASE or artefact, and not `docs/next_batch.md`, was changed.

## 1. Official verdict and scope

**Verdict: CASE D — INVALID.** This is not a viability verdict.

The frozen decision (`results/decision/decision.json`, `8f6dc061…`) returns D because exactly one validity gate failed:

- **V8 (numeric attribution) is false.** I declared that the C1 float64 exact-direction diagnostic would produce **0**
  stored (float32) channel-frame changes. It produced **3**: `upperarm_l` at rows 88, 100 and 116.
  - The float64 arithmetic noise of the frozen rig is at most 3.4e-15 per component. In these three frames it fell on
    the `upperarm_l` z component, whose magnitude is only 1.6e-10 to 2.7e-9.
  - At that magnitude the float32 spacing is 1.4e-17 to 2.2e-16, i.e. finer than an 8.3e-17 float64 delta, so the
    float32 cast changes.
  - None of the three stored changes moved any Blender-evaluated world position.
- **Every other gate passed:** V1–V7, V9, V10, all of G1–G9. All four boundary triggers T1–T4 hold.
- **Mechanical alternative, not assigned:** with V8 removed, the rule would output B. I do not assign B. The V8
  prediction was declared before outcomes, and changing or dropping it afterwards would be an outcome-driven gate
  change. The prediction itself contradicts the value-dependent resolution account I wrote into the same declaration
  ("smaller components have finer spacing").

So the declared numerical-attribution premise was incomplete. Within the frozen pipeline, bit-exact stored-value
comparison cannot certify "no rig influence" for an analytically invariant perturbation: floating-point noise reaches
float32 storage wherever a quaternion component is near zero.

**What this Batch established, by evidence level.** All of these are separate from the CASE.

| Level | Finding |
|---|---|
| Instrument (exact) | The storage model reproduces every audited Action F-curve value exactly, across 19 cases × 2 slots: 9 real-clip cases with 199 keys each and 10 mixed-window cases with 24 keys each. That covers every channel, key and component, with no tolerance. |
| Repeatability | Two independent builds are equivalent under the declared field-scoped path semantics. 4/4 positive controls are equivalent. 18/18 adversarial source mutations are rejected (0 false accepts). |
| W114 diagnosis | W114 V6 failed on two omitted MotionGraph fields. Separately, its prefix exemption for `.synthetic*` silently absorbs substantive synthetic changes. W114 G8 compared a 1e-9 float64 rule with exact float32 storage; the two disagree in both directions. |
| Contract (replayed) | The unchanged W114 candidate's integrity, lineage, locality, reversal, stale/permission/identity controls all hold again (G1–G7). They also hold with a storage-aware rig (G8, G9). |
| Rig influence (storage-aware) | Every A/B/D stored channel set equals the declared expectation (6 real conditions, 10 mixed condition-window pairs). The unconsumed-source controls are bit-identical on every row. |
| Lock semantics | Local channel-value preservation and posed world-space preservation are separately assessable, and they diverge in both directions. Neither is adopted. |

## 2. Verified baseline identities

- **Starting commit:** `d017d1b425fa3f5c1c3423e48b1401ab68380ba0`, equal to HEAD at start and to `origin/arch/single_frame_first`
  (the remote showed no divergence at start).
- **Instruction blob:** `docs/next_batch.md` = `54698373c7b91d4d367c9e091a394f83e80cc889`. It was verified before any
  change and was read only.
- **Main-local lock:** batch 115, this blob and this start commit.
- **Historical checks** (preflight `dee72caf…`):
  - W113 and W114 Worklog blobs at start equal the blobs now, and both COMPLETE markers are present.
  - The W114 frozen scripts and test are identical across the W114 final commit `f7239ca`, HEAD and the worktree:
    `w114_source_preserving_base.py`, `w114_rig_influence.py`, `w114_source_preserving_contract.py`,
    `w114_source_preserving_posthoc.py` and `test_w114_source_preserving_base.py`.
  - The W114 evidence manifest is `9f2af3d6…`. The W114 declaration is `f5fb2451…` and the W114 decision `57ea33fc…`
    with `case: D`. Repeat, Blender-compare, render audit, rig cases, both build reports and the C1 post-hoc each
    re-hash to their manifest entries.
- **Inputs:** every W114 input re-hashed locally and equals its W114 value. These are the W102 observations/predictions
  `6784357f…`, the clip `0dda8a5f…`, frozen HMR2 `fc5d3305…`, frozen J1X `cb9dc51e…`, the W105 manifest and its five
  entries, the S0 checkpoint `6c12fea9…` (equal to `S0_CHECKPOINT_SHA`), and the 3DPW bank, arrays and VALIDATION
  prediction.
- **Runtime:** Python 3.11.9, numpy 1.26.4, torch 2.13.0+cu130 (imported by reused helpers only), Blender 5.1.0,
  Windows 10.0.26200.

## 3. Prospective contract (declaration)

`results/declare/declaration.json`, SHA-256 `82ce9bf6e0e4148881c389ea21858c6f4a424e559afc72b434080de878b14117`, was written
with `outcomes_inspected: false`. The instrument was committed in `d7cd2d8` after the declaration was hashed and before
any W115 build existed. No instrument file changed afterwards. The post-hoc script (§11) was added after the decision.

**Disclosed prior knowledge and pre-declaration work:**
- The W114 outcomes for the same interventions are published and were read while designing the declaration. The A/B/D
  channel expectations equal W114's predictions and observations.
- **Blender rest-rotation probe:** the W105 rig was imported exactly as the historical renderer does. The probe recorded
  `rest_q` = (1, −1.410649872468639e-07, 0, 0).
- **Storage-model calibration:** run on the unperturbed historical FIXED Action of all 8 W114 cases. It was exact, and no
  RUNTIME (perturbed) value was read.
- **Posed-joint extraction smoke test:** run on one historical W114 `.blend` (A1, the published no-effect case). It
  showed the evaluated rotations equal the audited F-curves. The scratch directory was deleted before preflight.
- **One test-expectation bug fixed before the declaration:** 1e-12 added to a zero quaternion component is representable
  in float32.

**Identical to W114:**
- the datasets and frames;
- the perturbations: `C.RIG_CONDITIONS`, a 0.01 m camera-Z delta, C1 along unit(thorax − pelvis), even rows perturbed
  and odd rows as locality controls;
- the directives RS1, RS1_arm, RS_head, RS2, RSB, SS1 and SS1_arm;
- the edit programmes and controls, all called unchanged.

**New W115 parameters,** all declared before outcomes:
- the storage-aware rig: a W114 `RigModel` whose channels are the predicted stored float32 values, with `tol = 0`;
- the C1 float64 diagnostic;
- the constraint set `LW_D2`, built to separate local from world locks under the unchanged D2 intervention;
- Blender replay of the 10 mixed condition-window pairs;
- the world-point mapping and the representative-edit table.

**Numeric contract** (exact at every layer; no tolerance declared or used):

| Layer | Representation | Changed iff |
|---|---|---|
| L0 source | candidate state float32 | float32 bytes differ |
| L1 case | frozen rig in float64 → quaternion components | any bit differs. Continuous deltas are reported (hemisphere-aligned max component, geodesic degrees). |
| L2 predicted storage | Bone keys: `float32(component)`. Object key: float32 mathutils product `rest_q @ to_q(objq)` in BLI `mul_qt_qtqt` term order. The renderer's key order applies (mode locals overwrite extra locals at the same frame). | float32 values differ. Rotation change = nonzero hemisphere-aligned difference; sign flips are reported separately. |
| L3 Action | audited F-curve keyframe values | any of the 4 values differs |

Each channel-frame is classified into exactly one of four classes:
- **invariant:** L1 is bit-identical;
- **below output resolution:** L1 changed but L2 is identical;
- **material stored:** L2 changed;
- **missing/contradictory:** the NaN pattern mismatches, or L2 changed without an L1 change.

**C1 analytic account.** `torso_frames` builds `R_t = frame(u, R_sh − L_sh)` with `perp_u(s) = s − (s·u)u`. Adding a·u to
the right shoulder leaves `perp_u` unchanged exactly. Nothing else reads the right shoulder: R_p, u, the legs and the
left-arm vectors do not. So C1 is analytically invariant for every channel, and Object and the legs have no code path
from it at all.

**CASE rule (frozen):** D if any V fails; otherwise C if any G fails; otherwise B if any T holds; otherwise A.
- A is a defined predicate, but it is predicted unreachable on these populations. T1 (identity) and T2 (time) are
  structural facts already reported by W114.

## 4. Path semantics and repeatability (Required Evidence 2)

Builds `build_r1` and `build_r2` ran as separate processes into distinct directories, 16.6 s and 16.8 s.

**(A) Raw file-byte identity.** 22 files per build:
- 17 are byte-identical: the real and benchmark bundles, the rig case file, `lock_spec.json`, the pose and AnimationClip
  files, and `synthetic_source.npz`.
- 5 differ: `build_report.json`, `existing_path/legacy_motion.json`, and the synthetic bundle's `evidence_meta.json`,
  `constraints.json` and `lineage.json`.

**(B) Semantic equality.** All of these hold:
- real and benchmark evidence digests (`4ba03f34…`, unchanged from W114) and the benchmark digest;
- the normalized synthetic chain;
- observations and hypotheses: every array is inside the normalized digest;
- edit lineage: real 9-state and benchmark 3-state ids are identical raw;
- rig outputs: every layered classification, both mixed windows, and the rig case file sha `1de44824…`.

The in-process repeated same-input controls were bit-identical in both builds: evidence digest, base case frames, and
storage-rig channels.

**(C) Declared build-path differences.** 30 declared-difference entries were observed: 4 file-level and 26 build-report
fields. Every one is a declared allowlist entry whose guard held:
- `PV0` build_root, `PV1` build tag and `PV2` wall time;
- `PV3`, the legacy MotionGraph `source_metadata.pose_source`. That is the absolute path of the build's own
  reconstructed PoseSequence, plus its report copy and the report's MotionGraph sha256. The two MotionGraphs are
  canonically equal once **only** that field is replaced. The pose files it references are byte-identical.
- `PV4`, the synthetic source path. The W114 synthetic population writes its npz into the build directory. The recorded
  source **sha256 is not normalized**, and the referent npz is byte-identical. Normalize-then-rehash makes the evidence
  digest, constraints digests, edit-record digests and the state ids equal. The report fields derived from them each
  equal their own bundle's raw digest.

**How normalization was limited.**
- It is field-scoped, never prefix- or pattern-scoped.
- A value is normalized only if it lies under the build's own recorded root, has the declared relative remainder, and
  has equal referent bytes that equal the recorded sha.
- Never normalized:
  - any sha or digest;
  - frame digests;
  - model or checkpoint identity;
  - any path in a non-allowlisted field;
  - frame order, timestamps, joint names, constraints, parent links and values.
- No alternative allowlist was tried.

**Positive controls,** all equivalent:
- P0: an unmutated byte copy of r2, in another directory, vs r1;
- P1: r1 vs r2;
- P2: r1 vs r1, with 0 differing files;
- P3: r2 vs r1.

**Adversarial negatives:** applied to byte copies of r2; **18/18 rejected, 0 false accepts.**

The nine bundle mutations below were *re-chained* into self-consistent forgeries: every digest was recomputed, and
integrity passed in both builds. Every one was still rejected by the semantic chain.

| Control | Rejected by |
|---|---|
| N1 same filename, different pose bytes | PV3 referent guard (bytes differ); MotionGraph sha not explained |
| N1b same filename, different synthetic source bytes | synthetic referent ≠ recorded sha / r1 |
| N2a changed source sha256 · N2b changed frame digest | semantic chain (re-chained forgery) |
| N3 changed identity-bearing source path (another location) | semantic chain (re-chained) |
| N4a array changed without re-digest · N4b recorded digest replaced | integrity (stale hash) |
| N5a timestamp + 1 frame · N5b rows swapped · N5c joint renamed | semantic chain (re-chained) |
| N6a edit re-parented · N6b record bound to another permission digest | semantic chain (re-chained) |
| N7a MotionGraph confidence changed | not equal after PV3-only normalization |
| N7b AnimationClip rotation changed | raw bytes, no declared semantics |
| N8a source path set to `<build_root>/existing_path/…npz` (real file copied there, same sha) | the field is not allowlisted, so it is not normalized; semantic chain |
| N8b `pose_source` pointing at the other build's file | guard: not under the own root |
| N9 synthetic score + 0.01 (re-chained) | normalized synthetic chain |
| N10 report field under the synthetic prefix | field-scoped report diff |

**W114 V6 diagnosis.** The frozen W114 `compare` was rerun on copies of W114's own historical build reports.
- Unmodified, it reproduces W114's failure: `repeatable: false`, and the unexplained list is exactly the two MotionGraph
  fields.
- Two injected substantive changes to synthetic fields (`controls_summary.passed`, `scores.above_1`) were counted as
  "synthetic_source_path_induced": 22 instead of 21. That is, they were absorbed.
- So W114's premise failed in two ways:
  - it omitted known path fields;
  - its prefix exemption would have hidden real synthetic mutations even with complete path fields.

## 5. Representation-resolution rig test (Required Evidence 3)

**Storage exactness (V4).** For all 19 cases and both slots, the predicted stored values equal the audited F-curve
values exactly. Key sets equal the prediction. Every real case's FIXED value digest equals W105's `4d168d95…`.
Evaluated pose rotations after `frame_set` equal the audited values on every keyed channel-frame.

**Real clip** (`real_animcv_observation`; 100 perturbed rows of 199):
- Unperturbed rows show 0 L1 bit changes in every condition.
- Stored channel sets equal the declaration for A1, A2, B1, B2, D1 and D2.
- Every storage-aware authority outcome equals the independent re-derivation and is atomic.

| Condition | Stored rotation channels (perturbed rows) | Class counts / magnitude | Storage-aware authority |
|---|---|---|---|
| A1 `right_wrist` | none (L1 bit-identical, all rows) | — | NO_RIG_INFLUENCE |
| A2 `head` | none (L1 bit-identical, all rows) | — | NO_RIG_INFLUENCE |
| B1 `left_wrist` | upperarm_l 100, lowerarm_l 100 | max 0.58° / 2.60° | OUTSIDE |
| B2 `left_elbow`+`left_wrist` | upperarm_l 100, lowerarm_l 100 | 1.44° / 1.62° | ACCEPT |
| C1 `right_shoulder` along trunk, float32 | spine_01 75, upperarm_l 33, lowerarm_l 31; Object and 4 legs invariant (bit-identical) | below resolution: spine_01 25, upperarm_l 67, lowerarm_l 69; L1 max 1.2e-8 per component, ≤2.4e-6°; stored max 6.0e-8 | OUTSIDE (consistent) |
| C1 float64 exact diagnostic | **upperarm_l 3** (V8 predicted 0) | L1 changes: spine_01 67, upperarm_l 28, lowerarm_l 49, all ≤3.4e-15; stored max 2.2e-16 | not a candidate state |
| D1 `thorax` | all 8 channels, 100 each | 0.05°–0.58° | OUTSIDE |
| D2 `left_shoulder` | spine_01, upperarm_l, lowerarm_l (100 each) | 0.39° / 1.37° / 1.73° | OUTSIDE (under `LW_D2`: ACCEPT) |

**How W114 C1/G8 is explained.**
- The audited Action C1 counts (spine_01 75, upperarm_l 33, lowerarm_l 31) equal W115's L2 prediction exactly.
- W114's float64 1e-9 counts are reproduced exactly (76, 31 and 41).
- The two rules disagree in both directions:
  - some deltas below 1e-9 still flip the float32 storage of small components (upperarm_l 33 > 31);
  - some deltas above 1e-9 stay inside one float32 rounding cell (lowerarm_l 31 < 41).
- W114's Action-equals-case premise was therefore ill-posed.

**Attribution of the C1 float32 changes.**
- The candidate stores the 0.01 m trunk-axis displacement in float32. That leaves a perpendicular residual of up to
  1.62e-8 m (median 4.4e-9 m) on every perturbed row.
- No row has an L1 change with a zero residual.
- These are genuine sub-micrometre source changes, not instrument artefacts. They reach storage on 75/33/31
  channel-frames and move posed joints by ≤1.24e-7 m.

**The V8 failure (C1 float64).**
- Analytic invariance holds in exact arithmetic.
- Float64 arithmetic noise flips float32 storage on 3 `upperarm_l` channel-frames, as described in §1.
- Bit-exact storage comparison therefore classifies pure numerical noise as a material stored change. The batch's
  categories "analytically invariant", "below output resolution" and "material change" cannot be separated without a
  rule I did not declare.

**Mixed W102 windows** (`real_animcv_observation` on 3DPW images + frozen HMR2 + frozen J1X):
- Windows: `complete_control` (3DPW VALIDATION) and `arm_natural_most` (3DPW TEST), 24 rows each, with 12 perturbed.
- Reproduction is 24/24 in both. Unperturbed rows show 0 L1 changes. All 10 condition-window pairs equal the
  declaration.

| Condition | Both windows |
|---|---|
| MA1 S0 `left_wrist` | **none**, L1 bit-identical on all 24 rows (negative control: S0 does not drive the mixed arm) |
| MA2 S0 `left_shoulder` | none (HMR2 owns the torso frames) |
| MB1 S0 `left_knee` | thigh_l 12, calf_l 12 (0.69°–1.57°) |
| MB2 J1X wrist | upperarm_l 12, lowerarm_l 12 (0.19°–1.29°) |
| MD1 HMR2 joints24[16] | spine_01, upperarm_l, lowerarm_l, 12 each (0.60°–1.12°) |

**Independent ownership derivation.** The measured owner of each channel agrees with the declared ownership in both
configurations.
- `real_s0_only`: S0 torso frames drive Object and spine_01. S0 leg IK drives the legs. S0 left-arm IK drives the arm.
  The right arm is unkeyed rest.
- `mixed_w102`: HMR2 drives Object and spine_01. S0 drives the legs. J1X drives the arm.
- An S0 perturbation reaches a channel only where S0 is the declared owner.

No global dependency or derivative is inferred: there is one magnitude and one direction per condition.

## 6. Edit authority vs user intent (Required Evidence 4)

**Programmes and controls, replayed with the storage-aware rig:**
- Real programme: E1 ACCEPT (38 = 38 cells, 0 unintended). E2 ACCEPT (10 = 10). E3 STALE_PARENT. E4 STALE_EVIDENCE.
  E5 IDENTITY_UNVERIFIED. RV2, RV1 and RV1_tampered behave as declared (REVERT_MISMATCH for the tampered one).
- Rig steps:
  - R9 is OUTSIDE (upperarm_l depicts the non-permitted elbow).
  - R10 is ACCEPT with stored changes of exactly upperarm_l 38 and lowerarm_l 38. The Blender Action likewise changes
    exactly 38 + 38 keys.
  - R11 restores the channels bit-exactly.
  - R12 is NO_RIG_INFLUENCE. R13 is ACCEPT with 0 changed channel-frames. R14 then restores.
  - The final state equals the hypothesis bytes. The lineage audit recomputes all 9 ids.
- Benchmark programme: B1 changes 113 = 113 cells, then LOCK_CONFLICT_EDIT, STALE_PARENT, IDENTITY_UNVERIFIED, and a
  revert to the hypothesis.
- Evidence digest, arrays and source files are unchanged after every operation, and in-place writes are refused.
- Scores > 1 are kept exactly: 69 real, 4 benchmark. Clipping is caught as `SOURCE_MISMATCH`. No 3D reliability is
  manufactured.

**Two lock interpretations, evaluated separately.**
- Local means bit-identical stored values of every keyed channel that depicts the joint (tail-joint projection).
- World means bit-identical Blender-evaluated positions of the declared rig points.
- Blender world units are metres; the armature scale is 0.01 on the FBX cm rig.

`np` = non-permitted cells (preserve + default prohibit); `pres` = explicitly preserved.

| Mock edit | Candidate | Local np | World np | Local pres | World pres |
|---|---|---|---|---|---|
| M1 E1 under RS1 | OUTSIDE | VIOLATED (left_elbow 38) | VIOLATED (left_elbow 38, ≤1.65 mm) | SATISFIED (1,997) | SATISFIED (1,997) |
| M1b E1 under RS1_arm | ACCEPT | SATISFIED | SATISFIED | SATISFIED | SATISFIED |
| M2 B1 `left_wrist` | OUTSIDE | VIOLATED (elbow 100) | VIOLATED (elbow, ≤2.75 mm) | vacuous | vacuous |
| M3 B2 elbow + wrist | ACCEPT | SATISFIED | SATISFIED | vacuous | vacuous |
| **M4 D2 under `LW_D2`** | **ACCEPT** | **SATISFIED** (2,883 cells) | **VIOLATED** (500): right_wrist ≤3.41 mm, right_elbow ≤2.41 mm, spine ≤0.19 mm, head ≤0.067 mm, neck ≤0.018 mm | SATISFIED (796; all 796 depicted by no keyed channel) | **VIOLATED** (400 of 796) |
| M5 D1 `thorax` | OUTSIDE | VIOLATED (11 joints) | VIOLATED (16 joints, ≤16.3 mm) | vacuous | vacuous |
| M6 A1 `right_wrist` | NO_RIG_INFLUENCE | SATISFIED | SATISFIED | vacuous | vacuous |
| M7 C1 float32 | OUTSIDE | VIOLATED (214: thorax 75, left_shoulder 75, elbow 33, wrist 31) | VIOLATED (189 cells, ≤1.24e-7 m) | vacuous | vacuous |
| M7b C1 float64 diagnostic | — | VIOLATED (left_elbow 3) | SATISFIED | vacuous | vacuous |
| M8 mixed MA1, both windows | — | SATISFIED | SATISFIED | — | — |
| M9 mixed MB2, both windows | — | VIOLATED (elbow 12) | VIOLATED (elbow ≤0.94 mm `arm_natural_most`, ≤1.36 mm `complete_control`) | — | — |
| M10 E5 cross-frame copy | IDENTITY_UNVERIFIED | UNKNOWABLE | UNKNOWABLE | UNKNOWABLE | UNKNOWABLE |

Every outcome matched its declared prediction except M7b local (predicted SATISFIED). M7 had no prediction; it was
declared as reported only.

**How the two interpretations relate:**
- **They diverge in both directions.**
  - M4 is a locally allowed edit, accepted by the candidate, that moves world-locked joints by up to 3.4 mm.
  - It is **not** a user-preserving edit under world semantics.
  - Local semantics cannot see the right arm, neck or head at all, because no keyed channel depicts them.
  - M7b is the opposite case: a local stored-value violation caused by float noise, with no world movement.
- **Joint permission is not bone authority (T4).** In B1/M2, a single-joint wrist change moves the stored `upperarm_l`
  channel and the elbow's world position.
- **Which preservation rule a user means is an unresolved human product choice.** Neither rule was adopted, and nothing
  above settles it.

**Identity and time:**
- Per-frame subject identity is UNKNOWN on 199/199 real frames and 319/319 benchmark rows. Cross-frame operations fail
  closed there:
  - real: E5 and the identity control give 2/2 rejected, 0 eligible;
  - benchmark: B4 and the control give 2/2 rejected.
- On synthetic data the operation behaves as declared: a copy over verified frames 2→3 is ACCEPTED, and copies
  involving frame 5 are rejected.
- The Action keys `frame_index + 1` at 30 fps against a 50 fps source, so 198/199 keys are mismatched. At the last key
  the mismatch is 2.64 s; the durations are 3.96 s vs 6.60 s.
- The association sidecar recovers source time. That is not evidence that the Action or FBX preserves time; FBX was not
  exercised.

## 7. Populations and accounting (Required Evidence 5)

Regimes are never pooled.

**Cells, scores and support:**

| | Real (199 × 17) | Benchmark (319 × 17) | Synthetic (8 × 17) |
|---|---|---|---|
| present / absent-UNKNOWN | 3,383 / 0 | 5,310 / 113 | 135 / 1 |
| valid / invalid | 3,345 / 38 (all left wrist) | 5,310 / 113 | declared |
| observed / derived 2D kind | 2,587 / 796 | adapter kind only | — |
| scores > 1 (frames) / max | 69 (53) / 1.0791 | 4 / 1.0254 | 1 |
| 3D input-supported / unsupported / unknown | 3,345 / 38 / 0 | 5,310 / 113 / 0 | — |
| controls applicable / passed / NA | 23 / 23 / 2 | 20 / 20 / 5 | 25 / 25 / 0 |

**Edits and outputs:**
- **Authored → reverted edit cells:**
  - real: E1 38→38, E2 10→10, R10 38→38, R13 38 (head)→38;
  - benchmark: EB1 113→113.
- **Output channel-frames:**
  - R10: 76 stored (38 + 38), with the Action giving 76.
  - R13: 0.
- **Comparator:** 0/18 false-accepted mutations; 4/4 positives equivalent.

**Identity and lock accounting:**
- Source identity verified/unknown: real 0/199 frames; benchmark 0/319 rows; synthetic 7/8 (frame 5 declared
  unverified).
- Local/world lock outcomes: see §6. There are 11 assessable representative edits (13 evaluations with both mixed
  windows) and 1 UNKNOWABLE (M10).
- World instrument: 30,248 world-point/frame cells had all their dependency channels bit-identical, and 0 of them moved.
  FIXED world positions are bit-identical across all cases that share a FIXED Action.

**Qualitative before/after on concrete frames.** These come from `blender_compare.json` `examples`, reporting only.
Values are stored F-curve wxyz and Blender world positions.

- **C1 float32, frame 38** (source 0.76 s; Action key 39, 1.27 s at 30 fps):
  - `spine_01` changes in the 8th decimal of y: −0.03664044 → −0.03664043.
  - `upperarm_l` changes in w: 0.07318678 → 0.07318677.
  - `__object__` is identical.
  - The left wrist, elbow, right shoulder and thorax world positions are bit-identical at this frame. A stored-value
    change does not always reach world positions.
- **C1 float64, same frame:** all channels are identical.
- **RS1 / RS1_arm, frame 113** (2.26 s; key 114):
  - `lowerarm_l` changes from (0.77356, −0.54570, 0.05204, 0.31798) to (0.76981, −0.55208, 0.05115, 0.31622), and
    `upperarm_l` changes too.
  - The left wrist moves 3.53 mm and the left elbow 0.28 mm.
  - The left shoulder and right wrist do not move. `spine_01` is identical.
  - The edit is permitted under RS1_arm but not under RS1, where the elbow is not permitted.
- **D2 under `LW_D2`, frame 110** (2.20 s; key 111):
  - `spine_01`, `upperarm_l` and `lowerarm_l` change.
  - The left wrist moves 2.79 mm (permitted).
  - The world-locked joints also move: right wrist 0.48 mm, right elbow 0.34 mm, head 9.5e-3 mm, neck 2.6e-3 mm.
- **Mixed negative control MA1, window position 0** (W105 window time 2.20 s / 3.67 s; Action key 1): both arm
  channels and the elbow/wrist world positions are bit-identical in both windows.
- **Mixed positive MB2, position 0:**
  - both arm channels change;
  - `complete_control`: elbow 0.91 mm, wrist 1.00 mm;
  - `arm_natural_most`: elbow 0.63 mm, wrist 2.02 mm.

**Identity and provenance of these frames:**
- Real frames: `w102_historical_video:0dda8a5f16c67992` (clip sha `0dda8a5f…`, per-frame decoded digests), S0
  checkpoint `6c12fea9…`.
- Mixed windows: `3dpw:*` sequence ids from W105 `cases.json`, with HMR2 `fc5d3305…` and J1X `cb9dc51e…`.
- The renders (8,124 PNGs) are diagnostic replays of source-to-rig influence. They are not quality evidence.

**Kinds of claim kept apart:**
- **Synthetic contract correctness:** V7 and the synthetic tests.
- **Authentic replay:** real and benchmark G1–G7.
- **Causal rig influence:** §5, measured for one magnitude.
- **Not shown by this Batch:** practical animation quality and Architecture viability. The mock edits are offsets of
  stored values, not corrections or recovered animation.

## 8. Tests

All tests are AGENT-RUN; none has been reviewer-verified. Command prefix:
`C:\Projects\AnimCV\.venv_build_full\Scripts\python.exe -X utf8 -m pytest -q -p no:cacheprovider`.

- **`tests/test_w115_validity.py` + `tests/test_w114_source_preserving_base.py`: 27 passed** (15 + 12), recorded in
  `w115_tests.xml`. The W115 tests cover:
  - Blender quaternion product and key order (mode overrides);
  - the float32 cast at a resolution boundary;
  - the layer classification, including sign flips;
  - local/world divergence and world dependencies from the rest rig;
  - the root guard and field patterns that are not prefixes;
  - normalized-chain equality across roots with differing raw digests;
  - rejection of re-chained forgeries and of a changed source sha;
  - an out-of-root allowlisted path, a stale digest, and lineage rollback in a bundle;
  - frozen-rig reproduction of W105 on the real clip, with the storage rig.
- **Regressions: 178 passed** (`w115_regression.xml`). Files:
  - `test_w113_animation_base_contract`;
  - `test_animation_semantics` and `test_animation_semantics_v2`;
  - `test_framepose_bridge`, `test_frame_pose_observations`;
  - `test_motion_builder`, `test_motion_graph`;
  - `test_retarget_solver`, `test_fk_solver`, `test_keyframe_collapse`;
  - `test_w105_rig_utility`, `test_canonical_pose_parity`.
- **Failed:** none. One W115 test expectation was wrong before the declaration (see §3) and was corrected then.
- **NOT RUN:**
  - the full suite;
  - Blender-dependent suites (Blender was exercised only through the W105 render and audit and the W115 restq/posed
    stages);
  - FBX export/import;
  - any game engine.

## 9. Negative findings

**The verdict is D because my own numeric prediction was wrong.**
- The prediction was that the float64 control gives 0 stored changes. That is false for near-zero components.
- So the declared contract cannot separate "analytically invariant" from "material stored change" by bit-exact
  comparison.
- The same applies to any exact storage-level authority rule. It will attribute floating-point noise as rig influence,
  which in the authority check would mean false `OUTPUT_OUTSIDE_AUTHORIZED_DOMAIN`.

**C1 is not "no influence" for the float32 candidate.**
- The float32 state itself carries a genuine perpendicular source residual of ≤1.62e-8 m.
- That residual changes 139 stored channel-frames and moves posed joints by ≤1.24e-7 m.
- The candidate correctly returns OUTSIDE, so W114's prediction (NO_RIG_INFLUENCE) does not hold for a float32 state.

**W114's frozen V6 comparator was unsound in a second way:** its prefix exemption accepts substantive synthetic changes.

**Local tail-joint authority is blind to joints that no keyed channel depicts.** On the real rig that is the right arm,
neck, head and spine.
- An edit accepted under local authority (M4) moved world-locked joints by up to 3.4 mm.
- Conversely, a stored-value change can leave world positions bit-identical. This happened in M7b and on individual C1
  frames.

**Joint-level permission does not map to bone authority.** A wrist-only permission cannot authorize the IK-coupled
`upperarm_l` change.

**Authentic populations:**
- Per-frame subject identity is unavailable on every authentic population, so cross-frame operations cannot run there.
- The Action timebase (30 fps) differs from source time (50 fps), and the Action carries no source time.

## 10. Unresolved uncertainties and alternative explanations

**Policy.** Which lock semantics a user means remains a human product decision. W115 shows that the two semantics
differ; it does not choose between them.

**Numeric criterion.**
- A representation-aware criterion that would separate float noise from causal influence would need a declared rule:
  for example, value-relative spacing, world-space magnitude, or exact-arithmetic replay.
- No such rule was declared, and choosing one after these outcomes would be outcome-fitted.
- So completion condition 2 is only partly met:
  - **Met:** the float32/Blender storage semantics fully explain W114's C1 Action counts and the W114 G8 mismatch,
    without any tolerance.
  - **Not met:** the declared float64 control did not give an analytic-zero result at storage resolution.

**Storage model scope.**
- The Object composition and the `mul_qt_qtqt` term order are validated empirically: exact on all 38 audited Actions.
  This is not a proof for other inputs or other Blender builds.
- Only the `object_rotation_xyzw` root path is modelled. A yaw-only frame raises, and none occurred.

**World points.** The mapping is a declared approximate correspondence:
- pelvis = `spine_01` head;
- thorax = `neck_01` head;
- head = `neck_02` tail;
- spine = `spine_03` head.

The torso, neck and head verdicts depend on it. The limb joints map to bone heads directly.

**Exact world equality.** It is sensitive to float32 evaluation noise in the same way as storage. The C1 world
violations are ≤1.24e-7 m, and no magnitude threshold was used.

**Narrow data.**
- One private real clip with one performer, whose only low-evidence joint is the left wrist.
- Two mixed windows.
- One perturbation magnitude and direction per condition.
- The benchmark is not an untouched holdout.

**Mixed windows.** They have no candidate evidence store. Their lock evaluation uses masks constructed from the
perturbed rows, not candidate constraints.

**Benchmark adapter.** Its per-joint 2D derivation is still untraced.

## 11. Post-hoc diagnostic (labelled; no CASE)

`scripts/w115_validity_posthoc.py c1-stored-mechanism` was written after the frozen decision was inspected. Output:
`results/posthoc_c1_stored_mechanism/c1_stored_mechanism.json`, `ac26d085…`. It changes no gate and assigns no CASE.

**Float64 diagnostic:**
- 341 L1-changed components.
- 2 of them have a delta above half the local float32 spacing.
- 3 stored components changed. The third is a rounding-boundary crossing.
- All 3 are `upperarm_l` z with |value| 1.6e-10 to 2.7e-9 and spacing 1.4e-17 to 2.2e-16, at a delta of 8.3e-17.

**Float32 candidate:**
- 1,195 L1-changed components.
- 221 of them are above half spacing.
- 262 stored components changed.

## 12. Reproduction (PowerShell on Main, repo root)

```
$py="C:\Projects\AnimCV\.venv_build_full\Scripts\python.exe"; $bl="C:\Program Files\Blender Foundation\Blender 5.1\blender.exe"
$o="C:/Users/dna10/animcv-w115/results"; $s="82ce9bf6e0e4148881c389ea21858c6f4a424e559afc72b434080de878b14117"
$d="--declaration $o/declare/declaration.json --declaration-sha $s"
& $py -X utf8 scripts/w115_validity_contract.py preflight --out-dir $o/preflight
& $py -X utf8 scripts/w115_validity_contract.py declare --preflight $o/preflight/preflight.json --out-dir $o/declare
& $py -X utf8 scripts/w115_validity_contract.py build $d.Split() --tag r1 --out-dir $o/build_r1
& $py -X utf8 scripts/w115_validity_contract.py build $d.Split() --tag r2 --out-dir $o/build_r2
& $py -X utf8 scripts/w115_validity_contract.py repeat --first $o/build_r1 --second $o/build_r2 --out-dir $o/repeat
& $bl -b -t 2 --python-exit-code 1 -P scripts/w105_rig_blender.py -- render --rig examples/skinned_mannequin/BaseRig_Skinned.fbx --rest examples/e2e_demo/baserig_blender_rest.json --show-mesh --size 480 --ortho-scale 3.2 --equivalent-rig-tolerance 0.0001 --modes FIXED,RUNTIME --cases $o/build_r1/w115_rig_cases.json --out-dir $o/rig_replay/render
& $bl -b -t 2 --python-exit-code 1 -P scripts/w105_rig_blender.py -- audit --render-dir $o/rig_replay/render --out $o/rig_replay/render_audit.json
& $bl -b -t 2 --python-exit-code 1 -P scripts/w115_blender_stages.py -- posed --render-dir $o/rig_replay/render --out $o/rig_replay/posed.json
& $py -X utf8 scripts/w115_validity_contract.py blender-compare $d.Split() --build-dir $o/build_r1 --audit $o/rig_replay/render_audit.json --posed $o/rig_replay/posed.json --out-dir $o/blender_compare
& $py -X utf8 scripts/w115_validity_contract.py decide $d.Split() --preflight $o/preflight/preflight.json --build-dir $o/build_r1 --repeat $o/repeat/repeat_report.json --blender $o/blender_compare/blender_compare.json --out-dir $o/decision
# labelled post-hoc (no CASE)
& $py -X utf8 scripts/w115_validity_posthoc.py c1-stored-mechanism $d.Split() --out-dir $o/posthoc_c1_stored_mechanism
& $py -X utf8 scripts/w115_validity_contract.py manifest --base C:/Users/dna10/animcv-w115 --declaration-sha $s
& $py -X utf8 -m pytest tests/test_w115_validity.py tests/test_w114_source_preserving_base.py -q
```

Each stage refuses an existing output directory. The declaration embeds this host's absolute input paths. The preflight
stage runs `w115_blender_stages.py restq` itself.

**Key artefacts (SHA-256):**

| Artefact | SHA-256 |
|---|---|
| `declare/declaration.json` | `82ce9bf6…` |
| `decision/decision.json` | `8f6dc061…` |
| `preflight/preflight.json` | `dee72caf…` |
| `repeat/repeat_report.json` | `754998d5…` |
| `blender_compare/blender_compare.json` | `acd851d0…` |
| `build_r1/build_report.json` / `build_r2/build_report.json` | `f06f74e4…` / `93629377…` |
| `build_r1/w115_rig_cases.json` | `1de44824…` |
| `build_r1/lock_spec.json` | `630f7dfc…` |
| `rig_replay/render_audit.json` / `rig_replay/posed.json` | `d2ca3ae0…` / `36ad309b…` |
| `posthoc_c1_stored_mechanism/c1_stored_mechanism.json` | `ac26d085…` |

`evidence_manifest.json` (`5988f4fb8c5f848aaa23b1f0eb4500e73176d234b8b399201ffe56c539de1d2d`) covers 513 files, plus the
8,124 PNGs through a combined digest.

## 13. Changed files and revisions

All files are new; no pre-existing tracked file was modified.
- `scripts/w115_validity_base.py`: storage model, layered classification, storage-aware RigModel, mixed case frames,
  lock semantics, and semantic comparison helpers.
- `scripts/w115_validity_contract.py`: stages preflight, declare, build, repeat, blender-compare, decide and manifest.
- `scripts/w115_blender_stages.py`: the Blender `restq` probe and posed world-joint extraction.
- `scripts/w115_validity_posthoc.py`: post-hoc and labelled, added after the decision.
- `tests/test_w115_validity.py`.
- This Worklog.

**Revisions:**
- starting commit `d017d1b`;
- `d7cd2d8`, the pre-outcome instrument;
- the commit adding this Worklog and the post-hoc script; its hash is the final source revision and the branch head
  after publication;
- a final commit that adds only the COMPLETE marker.

**Completed and not completed.** Every Required Evidence part was executed.
- **Not completed in the declared sense:** a valid numeric attribution (V8), and therefore any non-D CASE.
- **Not attempted:** FBX/engine time survival, real occlusion recovery, a Generator, and Production integration.

## 14. Resource and environment constraints

**Cost:**

| Item | Cost |
|---|---|
| Each build | 16.6–16.8 s |
| Repeat with 19 directory copies | about a minute |
| Blender render (19 cases, 2 modes × 2 views; 8,124 PNGs) | 3 min 16 s |
| Audit and posed extraction | seconds each |
| Focused tests | 1.1 s |
| Regressions | 5.7 s |
| Storage under `C:\Users\dna10\animcv-w115\` | 2.1 GB, mostly PNGs and adversarial build copies; local only, not published |

**Environment:**
- Main only. No GPU inference, network or model download, package change, elevated access, or LabServer63/69 use.
- Every array came from a hash-verified stored artefact. No evaluation GT was read: neither the J1X `*_target_*` arrays
  nor the bank's `target_3d`/strata.
- The source clip was not copied or published.
- Single branch `arch/single_frame_first`. No branch, merge, rebase, reset or force operation.

AUTOMATION_BATCH_STATUS: COMPLETE
BATCH_NUMBER: 115
