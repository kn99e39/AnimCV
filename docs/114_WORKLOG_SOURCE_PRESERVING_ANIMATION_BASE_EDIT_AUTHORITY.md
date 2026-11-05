# Worklog — Source-Preserving Animation Base & Edit Authority Feasibility (2026-11-05)

Batch 114. Evidence-only architecture diagnostic on `arch/single_frame_first`, run unattended on Main (local RTX 5080
host; no GPU inference was needed). No `src/` module, schema, model, threshold, rig, solver, FBX/Blender exporter,
default, historical artefact or `docs/next_batch.md` was changed. W113's frozen CASE D and its post-hoc labels were not
touched (Worklog blob unchanged; frozen decision hash re-verified).

## 1. Official verdict and scope

**Verdict: CASE D — INVALID.** This is not a viability verdict.

The frozen decision stage (`results/decision/decision.json`, `57ea33fc…`) returns D because one validity gate failed.
- **V6 (repeatability), as implemented in the frozen `compare` stage, returned `repeatable: false`.** Its "unexplained"
  list holds exactly two fields of the existing-path **control**:
  - the legacy MotionGraph's `source_metadata.pose_source`, which is the absolute path of each build's own reconstructed
    PoseSequence;
  - that MotionGraph file's sha256.
- I did not list these as path-induced in the frozen code, although W113 had already identified the same field as
  path-induced.
- Every quantity the declaration text names for V6 was identical across the two builds:
  - real and benchmark evidence digests;
  - the real lineage's state ids;
  - the Blender case file;
  - all rig outcomes.
- A labelled post-hoc check (`posthoc_repeat_pathnorm`) finds the two MotionGraphs byte-identical once the build
  directory is replaced by a placeholder. The frozen result still stands.

A second, independent instrument failure exists, and on its own it would also block a trustworthy verdict.
- **G8's Blender clause fails for exactly one rig condition, C1.** C1 moves `right_shoulder` 0.01 m along the trunk
  axis.
  - At case level its thorax-frame channels moved by at most 1.4e-8 per quaternion component. That is above my declared
    tolerance of 1e-9 but below the float32 resolution of the Blender F-curves.
  - So the Action-level and case-level change counts differ: spine_01 75 vs 76, lowerarm_l 31 vs 41, upperarm_l 33 vs 31.
- Re-running the frozen decision on the path-normalized repeat (post-hoc input, `posthoc_decision_inputs_pathnorm`)
  mechanically outputs C, because G8 is false.
- **That C is not assigned.**
  - The mismatch comes from my tolerance choice: 1e-9 is below the float32 resolution of the candidate state and of
    Blender.
  - It does not show that the evidence/state separation cannot provide its guarantees.
  - Calling it C would misstate the evidence.

**What every other gate measured** (all on the frozen build; reported as decision inputs, not as a verdict):

| Gate | Result |
|---|---|
| V1 inputs, V2 W113 intact, V3 evidence = source bytes, V4 positive ingest, V5 rig reproduction, V7 synthetic controls, V8 rig positive control | all true |
| V6 repeatability (frozen code) | **false** (2 path-induced control fields; true after post-hoc path normalization) |
| G1 evidence immutable, G2 edit locality, G3 reversal, G4 stale rejection, G5 permission, G6 score semantics, G7 real/bench controls | all true |
| G8 rig influence | **false**, only through the C1 Blender clause; every other G8 component true |
| T1 identity, T2 time, T3 authority (B-triggers) | all true |

**Scope as executed.** I built a scratch-only representation and drove it on three separate populations with frozen
inputs:
- **The representation:**
  - immutable source evidence;
  - a versioned editable state;
  - synthetic user constraints;
  - an observed-influence rig-authority check.
- **The three populations:**
  - the authentic GT-free real clip, `real_animcv_observation`, W102;
  - the W113 3DPW benchmark sequence, `benchmark_detector_observation`;
  - a synthetic contract sequence (`oracle_geometry`).
- **The comparison against the unchanged existing paths:**
  - the W113 legacy CLI replay;
  - the FramePose bridge replay;
  - the W105 frozen rig and Action.
- **The influence tests:** controlled source perturbations of two frozen W105 rig configurations, replayed at case level
  and in Blender.
- **Not done, and not authorized:**
  - Generator training or implementation;
  - production schema or data-format changes;
  - estimator, detector, FK/IK or FBX repair;
  - confidence calibration;
  - Batch 115.

**Regime labels.** Regimes are never pooled.

| Population | Regime and details |
|---|---|
| Real | `real_animcv_observation`: W102 M1 MMPose+RTMDet on `examples/production_demo/source.mp4`; 199 frames at 50 fps; GT-free, private, not unseen. |
| Benchmark | `benchmark_detector_observation`: 3DPW shipped detector, `3dpw:courtyard_basketball_01:actor0`, VALIDATION, 319 rows. Not a new holdout. |
| Mixed rig windows | `real_animcv_observation` on 3DPW images (W102 M_Main S0), plus frozen HMR2 and frozen J1X; two W105 windows of 24 frames. Used only as a rig-ownership probe; the identity labels were never read. |
| Synthetic | `oracle_geometry`: 8 frames at 25 fps. Contract test only. |

No evaluation GT was read anywhere:
- the J1X npz `*_target_*` arrays were never accessed;
- neither were the bank's `target_3d` array or its stratification.

## 2. Verified historical baseline identities

- **Starting commit:** `fd24ef0817eb185e4d9ed2ad80bfb282f7bb78b6`.
- **Instruction blob:** `docs/next_batch.md` = `6d8ce038a4a2a878aef3136f2f4723d6efebd68c` (Batch 114). Verified before any
  change. The remote was in sync at start and again before publication.
- **Main-local lock** (`state/active_batch.json`): batch 114, this blob, this start commit. It is Main-local only.
- **W113 checks:**
  - the Worklog blob at the start equals the blob now, and the COMPLETE marker is present;
  - `decision_frozen.json` = `2369289f…` with `case: D`;
  - the declaration = `c3f7485c…`;
  - the W113 evidence manifest = `c4c1a6e7…`.

Inputs (preflight `0b887720…`; every later stage re-verifies them against the declaration):

| Input | SHA-256 |
|---|---|
| W102 `observations_predictions.npz` | `6784357f7c6145e4b60e2e9201e32accd3ae00001486283a1e76f0adae61e7bd` |
| W102 `report.json` / `inspection.json` | `e6e64307…` / `dd25b7af…` |
| source clip (repo copy) | `0dda8a5f16c67992330a1a8534cf0bfefeb5e0f2aec438577a9e1dbace773275` |
| W102 `hmr_pred_full.npz` (frozen HMR2) | `fc5d3305…` |
| W102 `w102_j1x_predictions_3dpw.npz` (frozen J1X) | `cb9dc51e…` |
| W105 `evidence_manifest.json` | `cb0eca93…`; `video_sources.npz`, `video_cases.json`, `cases.json`, `sources.npz` and `render_clip_audit.json` each equal their manifest entries |
| S0 checkpoint | `6c12fea9…`, equal to `run_real_observation_s0.S0_CHECKPOINT_SHA` |
| 3DPW bank index / arrays / frozen VALIDATION prediction | `6fe13cdc…` / `909090f7…` / `402d70a2…`; content digest `75519e63…` |

Runtime: Python 3.11.9, numpy 1.26.4, torch 2.13.0+cu130 (imported by the reused helpers, not used for inference),
Blender 5.1.0, Windows 10.0.26200.

## 3. Declared hypothesis, controls and frozen declaration

The declaration is `results/declare/declaration.json`, SHA-256
`f5fb2451b06126f9133aeccaabdd68f0eeb3fca1243a6e45fda36ab6851f1aa1`. It was written with `outcomes_inspected: false`.
The instrument and tests were committed in `dc5bfd3` after the declaration was hashed and before any real, benchmark or
perturbed-rig outcome existed. The frozen scripts have not changed since `dc5bfd3`.

**Disclosed pre-declaration development.** Two things were run before the declaration:
- The instrument was iterated on synthetic data.
- The unperturbed frozen rig was replayed to check that it reproduces W105 byte-for-byte: real clip 199/199,
  `complete_control` 24/24, `arm_natural_most` 24/24.

No perturbed output and no candidate outcome was computed beforehand.

**H1.** Immutable source evidence plus a versioned editable state can establish a minimum contract for controlled
editing, without changing the production MotionGraph schema. H1 was not assumed.

**Control.** The unchanged existing paths, run with unchanged code through the W113 helpers:
- `PoseSequence → MotionGraph → AnimationClip`;
- the FramePose bridge;
- the W105 rig and Action.

**Candidate.** `scripts/w114_source_preserving_base.py`, scratch only. It has five logical layers:
1. **Evidence** holds the raw 2D xy and the raw detector score. The score is stored exactly as float64 of the producer's
   value and is never bounded, clipped or transformed. Evidence also holds:
   - the present/visible flags;
   - the producer's validity;
   - the observed/derived kind;
   - frame index, timestamp and per-frame image digests;
   - source sha256 values;
   - the frozen 3D hypothesis, with its frame, unit, model identity and direct-input support status;
   - the subject-identity status.

   Evidence arrays are read-only and bound to a content digest. 3D reliability is recorded as UNAVAILABLE and never
   carries a value.
2. **Editable state** is a copied float32 value array with a per-cell origin code. Its `state_id` hashes the evidence
   digest, the parent id, the edit-record digest and the values.
3. **Edit records** carry:
   - base state id, evidence digest and constraints digest;
   - targets, with the source timestamp of each target;
   - coordinate frame and unit;
   - the exact original values and the replacement values;
   - the changed and unchanged counts.

   Applying an edit is atomic. Reverts must restore the parent's bytes.
4. **Constraints** are `preserve` / `permit` / explicit `prohibit`, with default prohibit. Preserve∧permit and
   permit∧prohibit are rejected, never resolved by priority.
5. **Output association** recomputes the frozen rig on the parent and child states and records the channel-frames that
   actually change. The declared authority rule is applied to that set:
   - each channel's local value places one or more depicted joints (the tail-joint projection):

     | Channel | Depicted joints |
     |---|---|
     | Object | pelvis and both hips |
     | `spine_01` | thorax and both shoulders |
     | thigh / calf | knee / ankle |
     | `upperarm_l` / `lowerarm_l` | elbow / wrist |

   - a changed channel-frame is authorized only if all its depicted joints are permitted and none is preserved;
   - a rig-claimed edit with no observed change is rejected as `NO_RIG_INFLUENCE`.

**Score contract.**
- A finite score of any magnitude is accepted.
- A present observation with a non-finite score rejects the evidence with `SCORE_MISSING`. This is the declared
  fail-safe; no operation proceeds.
- An absent observation is NaN with `present=False` (UNKNOWN). Any value placed there is rejected.
- `1 − score` is never formed.

**Rig perturbations.**
- Every perturbation is +0.01 m in camera Z. This is W113's declared mock delta.
- C1 instead uses 0.01 m along the per-frame unit vector of thorax − pelvis.
- Perturbed rows are the even frame indices (100 of 199) or the even window positions (12 of 24). The odd rows serve as
  temporal-locality controls.
- Tolerance: a channel counts as changed when a hemisphere-aligned quaternion component moves by more than 1e-9.
  Bit-level changes are reported separately.
- The expected channel sets and the authority outcomes were predicted from code analysis before any perturbed output
  existed. Rig-dependent outcomes are graded on whether they agree with an independent re-derivation from the observed
  influence, written as a separate code path. Prediction mismatches are reported, not gated.

**CASE rule (frozen).** D if any V fails; otherwise C if any G fails; otherwise B if any T holds; otherwise A.

## 4. Producer/consumer ownership matrix (static code facts + measured counts)

| Quantity | Boundary | Status | Location / measured |
|---|---|---|---|
| raw 2D xy | adapter → PoseSequence → MotionGraph | PRESENT | MotionGraph xy exact on 3,184/3,184 present canonical cells |
| raw 2D xy | MotionGraph → AnimationClip | LOST | `src/retarget/fk_solver.py:108-158` |
| raw 2D xy of invalid joints | real observation → S0 input | LOST (zero-filled) | 38/38 invalid cells zero-filled (`scripts/w101_reference_estimator.py:60-75`) |
| detector score | → MotionGraph | PRESENT (raw) | exact on 3,184/3,184 cells, including the 69 values > 1 |
| detector score | FrameBank → Lifted (bridge) | DERIVED | `depth_uncertainty = 1 − score` is negative in 69 cells (min −0.0791); the confidence of the 38 invalid cells is 0, so their raw score is lost there |
| detector score | → AnimationClip | DERIVED | min over a bone's joints; 38 samples with 0.0 = held previous rotation, none of them a measured zero |
| detector score / validity | → W105 Action | LOST | `scripts/w105_rig_utility.py:657-735` |
| observed vs derived 2D | adapter / MotionGraph | LOST (no flag) | 796 derived cells (pelvis, spine, thorax, neck); thorax = neck duplicate |
| S0 3D | → Lifted / AnimationSemantics | PRESENT (camera_root_relative, metres) | articulation exact |
| S0 3D | → MotionGraph / CLI AnimationClip | UNAVAILABLE | no `position_3d`; a mixed-unit probe is rejected |
| S0 3D | → W105 real-clip Action | DERIVED (rotations) | see §9 |
| 3D reliability | everywhere | UNAVAILABLE | `src/pose/framepose_bridge.py:107` |
| source/model provenance | → MotionGraph / AnimationClip / Action | LOST | `source_metadata` holds only the pose path; clip name is `Generated_Motion` |
| frame index | → Action | PRESENT as key index (`frame_index + 1`) | 199/199 keys |
| timestamp | → AnimationClip / Action | LOST (Clip: derivable from fps; Action: 30 fps vs 50 fps) | 198/199 Action keys mismatched |
| per-frame tracked subject | adapter | LOST before serialization | W102 keeps only aggregates (tracked 199/199, mean 1.76 candidates per frame) |
| user preserve/permit | everywhere | UNAVAILABLE | `MotionFrame.locked` True on 0/199 frames; the collapse lock is frame-wide (1,592 vs 158 keys) |
| bone/channel source ownership | W105 rig | measured in §9 | — |

**The candidate retains every field listed as PRESENT upstream.** It keeps the raw score, xy, validity, kind, frame,
time, frame digests, the 3D hypothesis, model identity and source hashes. On the real clip all six arrays checked
against the source matched byte-for-byte: score, xy, 3D, validity, frame index and timestamp. Source time stays
recoverable at the rig boundary only through the association sidecar, which maps each Action key to a `frame_index` and
from there to the source time. The unchanged Action does not carry it.

**Four meanings kept apart.** In the candidate these are four separate things, and none is derived from another:
- the detector's numeric score;
- observation validity, which is the producer's flag;
- calibrated probability, which is UNAVAILABLE;
- 3D reliability, which is UNAVAILABLE;
- edit permission, which comes only from the constraints.

On the real clip, validity equals `score ≥ 0.3`:
- in 2,587/2,587 cells of observed-kind joints;
- in 796/796 cells of derived-kind joints.

## 5. Source score semantics and validity accounting

| | Real (199 × 17 = 3,383 cells) | Benchmark (319 × 17 = 5,423 cells) |
|---|---|---|
| present / absent (UNKNOWN) | 3,383 / 0 | 5,310 / 113 |
| finite scores on present cells | 3,383 | 5,310 |
| scores > 1 (all `input_valid`) | **69**: head 6, left ankle 45, right ankle 17, right elbow 1; 53 frames; max 1.0791 | **4**: head 2, left elbow 1, left wrist 1; max 1.0254 |
| scores < 0 | 0 | 0 |
| cells where `1 − score` would be negative | 69 | 4 |
| ingest of unchanged evidence | PASS (scores > 1 do not invalidate) | PASS |
| directly observed / derived 2D cells | 2,587 / 796 | 5,423 benchmark-adapter cells (per-joint derivation not traced) |
| 3D input-supported / unsupported / UNKNOWN | 3,345 / 38 / 0 | 5,310 / 113 / 0 |

On the benchmark source side, the bank stores score 0 at all 113 invalid cells and xy (0, 0) at 104 of them. These are
placeholder values. The candidate represents all 113 as UNKNOWN, not zero, and the `unknown_to_zero_coercion` control is
exercised on these authentic cells.

## 6. Candidate information-preservation results

Real clip (`real_animcv_observation`), evidence digest `4ba03f34…`:
- **Evidence:**
  - verifies, and equals its source byte-for-byte;
  - an in-place write is refused because the arrays are read-only;
  - the evidence digest, every array byte, the metadata and the five source-file hashes were unchanged after every
    operation (programme, controls, rig conditions);
  - a modified copy whose digest was not recomputed is rejected with `EVIDENCE_MODIFIED`.
- **Clipped scores:** a copy with scores clipped to ≤ 1 still passes structural verification, because no bound exists.
  It is caught against the source with `SOURCE_MISMATCH`.
- **Lineage:** 9 states in the real programme (root, E1, E2, RV2, RV1, R10, R11, R13, R14). Every id recomputed exactly
  and the root is the hypothesis.

## 7. Edit locality and reversal accounting

**Real programme** (RS1: preserve 1,997 / permit 38 / default-prohibited 1,348; all in `est3d` cells):

| Step | What | Expected | Observed |
|---|---|---|---|
| E1 | +0.01 m z on the 38 permitted left-wrist cells (frames 109–118, 154–181) | ACCEPT | ACCEPT; 38 changed = 38 requested, 0 unintended; 3,345 unedited cells byte-identical; origin changed on exactly the 38; recorded originals = parent bytes |
| E2 | +0.005 m x on the 10 cells in frames 109–118 | ACCEPT | ACCEPT; 10 = 10, 0 unintended |
| E3 | E2 cells prepared against the post-E1 state | STALE_PARENT | STALE_PARENT, head unchanged |
| E4 | bound to evidence digest `f…f` | STALE_EVIDENCE | STALE_EVIDENCE, head unchanged |
| E5 | copy `left_wrist` from frame 108 to 109 (cross-frame) | IDENTITY_UNVERIFIED | IDENTITY_UNVERIFIED |
| RV2 | revert E2 | ACCEPT | state bytes and origin = post-E1 state |
| RV1_tampered | revert E1 with one restore byte flipped | REVERT_MISMATCH | REVERT_MISMATCH, head unchanged |
| RV1 | revert E1 | ACCEPT | state = hypothesis bytes (values and origin) |
| R11 / R14 | revert R10 / R13 | ACCEPT | after R11 the rig channels are bit-equal to the hypothesis channels; the final state = hypothesis bytes |

**Benchmark programme** (RSB: preserve 5,310 input-supported / permit 113 input-unsupported cells):
- EB1 changed exactly 113 = 113 cells, with 0 unintended.
- A preserved-cell edit was rejected with `LOCK_CONFLICT_EDIT`.
- The stale parent was rejected with `STALE_PARENT`.
- The cross-frame copy was rejected with `IDENTITY_UNVERIFIED`.
- The revert gave the hypothesis bytes.
- The lineage audit is exact (3 states).

**Totals.** Restored cells after reversal:
- real: 38/38 for E1 and 10/10 for E2;
- benchmark: 113/113.

Unintentionally changed cells: 0 on every accepted edit.

## 8. Real and synthetic results, reported separately

**Synthetic correctness** (`oracle_geometry`, toy rig declared in the code):
- 25/25 controls applicable, 25 passed, 0 failed, 0 NOT APPLICABLE.
- Programme steps all as declared.
- Rig steps R9/R10/R12/R13 matched both their predictions and the independent rule.
- A cross-frame copy over verified frames 2→3 was **accepted**. This is the only positive exercise of a cross-frame
  operation; no authentic population has verified continuity.

**Real-observation contract behaviour:**
- 23 applicable, 23 passed, 0 failed.
- 2 NOT APPLICABLE, neither counted as a pass:
  - `unknown_to_zero_coercion`: the clip has no absent observation;
  - `edit_on_unknown_target`: it has no non-finite 3D cell.
- Synthetic evidence for both appears above.

**Benchmark** (separate population):
- 20 applicable, 20 passed, 0 failed.
- 5 NOT APPLICABLE:
  - 3 rig controls: no frozen rig configuration consumes the benchmark-detector S0 predictions;
  - `unauthorized_edit`: RSB covers every finite cell with either preserve or permit, so no default-prohibited cell
    exists;
  - `edit_on_unknown_target`.

Every rejection in every population was atomic: the lineage head and the stored-state count were unchanged, and so was
the evidence.

Control codes exercised, each matched exactly where applicable:
- identity: `EDIT_NOT_AUTHORIZED`, `LOCK_CONFLICT`, `LOCK_CONFLICT_EDIT`;
- staleness: `STALE_EVIDENCE`, `STALE_PARENT`;
- evidence: `EVIDENCE_MISSING`, `PROVENANCE_MISSING`;
- frame and time: `FRAME_MISMATCH`, `TIME_MISMATCH`, `IDENTITY_UNVERIFIED`;
- structure: `JOINT_DUPLICATE`, `COORDINATE_MISMATCH`, `OBSERVATION_IMMUTABLE`;
- integrity: `EVIDENCE_MODIFIED`, `UNKNOWN_COERCED_TO_ZERO`, `CONFIDENCE_MANUFACTURED_3D`, `SCORE_MISSING`,
  `SOURCE_MISMATCH`;
- rig: `ASSOCIATION_INVALID`, `NO_RIG_INFLUENCE`, `OUTPUT_OUTSIDE_AUTHORIZED_DOMAIN`;
- reversal and targets: `REVERT_MISMATCH`, `EDIT_UNKNOWN_TARGET`.

## 9. Actual source-to-rig influence accounting

**Ownership (code + measurement).** The two frozen W105 configurations differ.

**`real_s0_only`** (W105 real clip):
- Object and spine_01 come from W93 torso frames built on S0 pelvis, hips, thorax and shoulders.
- The legs are S0 through W95 two-bone IK in the S0 pelvis frame.
- The left arm is S0 through the existing two-bone arm chain in the S0 thorax frame.
- The right arm is rest and not keyed.

**`mixed_w102`** (W105 3DPW windows):
- Object comes from HMR2 `global_orient`.
- spine_01 comes from the HMR2 pelvis vs the HMR2 joints thorax frame.
- The legs are S0.
- The left arm is frozen J1X.
- The right arm is rest.

My replicas reproduce the stored W105 exports byte-for-byte: real 199/199 frames; `complete_control` 24/24;
`arm_natural_most` 24/24.

Real clip: 100 perturbed rows. Unperturbed rows showed 0 bit changes in every condition.

| Condition | Kind | Predicted | Observed (changed frames of 100) | W113 map predicted but unchanged | Authority (predicted / observed / independent) |
|---|---|---|---|---|---|
| A1 `right_wrist` | A | none | **none** (bit-identical) | — | NO_RIG_INFLUENCE / same / same |
| A2 `head` | A | none | **none** (bit-identical) | — | NO_RIG_INFLUENCE / same / same |
| B1 `left_wrist` | B | upperarm_l, lowerarm_l | upperarm_l 100 (max 0.58°), lowerarm_l 100 (max 2.60°) | — (W113's map **omits** upperarm_l for the wrist) | OUTSIDE / OUTSIDE / OUTSIDE |
| B2 `left_elbow` + `left_wrist` | B | upperarm_l, lowerarm_l | upperarm_l 100 (1.44°), lowerarm_l 100 (1.62°) | — | ACCEPT / ACCEPT / ACCEPT |
| C1 `right_shoulder` along the trunk axis | C | none | Object and 4 legs **bit-identical**; spine_01 76, lowerarm_l 41, upperarm_l 31 at ≤ 1.4e-8 per component | Object, thigh/calf l/r (5 of 8) | NO_RIG_INFLUENCE / **OUTSIDE** / OUTSIDE |
| D1 `thorax` | D | all 8 | all 8 channels, 100 each (0.05–0.58°) | — | OUTSIDE / OUTSIDE / OUTSIDE |
| D2 `left_shoulder` | D | spine_01, upperarm_l, lowerarm_l | spine_01 100 (0.39°), upperarm_l 100 (1.37°), lowerarm_l 100 (1.73°); Object and legs bit-identical | Object, thigh/calf l/r (5 of 8) | OUTSIDE / OUTSIDE / OUTSIDE |

Mixed W102 windows: 12 perturbed rows per window; unperturbed rows showed 0 bit changes. All 10 condition-window pairs
matched their prediction.

| Condition | `complete_control` (3DPW VALIDATION) | `arm_natural_most` (3DPW TEST) |
|---|---|---|
| MA1 S0 `left_wrist` | **none** | **none** |
| MA2 S0 `left_shoulder` | **none** | **none** |
| MB1 S0 `left_knee` | thigh_l 12, calf_l 12 | thigh_l 12, calf_l 12 |
| MB2 J1X wrist | upperarm_l 12, lowerarm_l 12 | upperarm_l 12, lowerarm_l 12 |
| MD1 HMR2 joints24[16] (left shoulder) | spine_01, upperarm_l, lowerarm_l (12 each) | same |

**Rig-level edit authority on the authentic 38-cell left-wrist region:**
- **RS1 (wrist permitted):** the edit is source-authorized, but rejected at rig level with
  `OUTPUT_OUTSIDE_AUTHORIZED_DOMAIN`. upperarm_l, which depicts `left_elbow`, changes on all 38 frames, and the elbow is
  not permitted.
- **RS1_arm (elbow and wrist permitted on the same runs):** the edit is accepted. The observed change is exactly
  upperarm_l 38 and lowerarm_l 38 channel-frames, and nothing else. The Blender Action also changes on exactly 38 + 38
  keys. The revert restores the channels bit-for-bit.
- **Head edit claiming rig influence:** `NO_RIG_INFLUENCE`. Without the claim it is accepted, with 0 changed
  channel-frames recorded.

**Blender Action replay** (unchanged W105 renderer; 8 cases; 6,368 diagnostic PNGs):
- In all 8 cases the FIXED Action value digest equals W105's FIXED digest `4d168d95…`.
- Keys sit at `frame_index + 1` on all 199 frames; scene fps is 30, with 32 F-curves and CONSTANT interpolation.
- Action-level changed counts equal the case-level counts in 7/8 cases, every channel included.
- C1 differs, as described in §1.

**Expected vs actual affected channels, overall:**
- real conditions: predictions matched in 6 of 7;
- mixed condition-window pairs: matched in 10 of 10;
- programme rig steps: matched in 4 of 4.

Every candidate authority decision equalled the independent re-derivation: 7 of 7 conditions, 4 of 4 programme rig
steps and 2 applicable real rig controls. Every one was atomic.

**C1 mechanism** (post-hoc, `posthoc_c1_float32`, labelled):
- The candidate state is float32. Storing the 0.01 m trunk-axis displacement in float32 leaves a component perpendicular
  to the axis of up to 1.62e-8 m (median 4.4e-9 m).
- With the exact float64 displacement on float64 values, the same frozen rig changes nothing beyond 3.4e-15. No channel
  exceeds the tolerance.
- So the trunk-axis invariance of the thorax frame holds analytically. The declared C1 test could not resolve it at a
  1e-9 tolerance on a float32 state.
- The frozen C1 outcome remains recorded as a prediction mismatch.

## 10. Identity, provenance and timebase limitations

**Identity:**

| Identity | Status |
|---|---|
| Source sequence | PRESENT: clip sha256 plus 199 decoded-frame digests |
| Frame | PRESENT: `frame_index`, timestamp and per-frame digest |
| Candidate animation | PRESENT: the `state_id` lineage |
| Rig channel | PRESENT: Action name and F-curve `data_path[index]` |
| Tracked subject | **UNKNOWN per frame** on 199/199 real frames and 319/319 benchmark rows |

For the benchmark, the dataset's actor label is not runtime-verified. Cross-frame operations therefore fail closed on
both authentic populations.

**Time:**
- The W105 Action keys `frame_index + 1` at 30 fps against a 50 fps source.
- 198/199 keys are mismatched; frame 0 is the exception.
- At frame 198 the source time is 3.96 s but the Action time is 6.60 s, a mismatch of 2.64 s. Durations are 3.96 s vs
  6.6 s.
- The candidate's association maps every key back to source time.
- No claim is made that an unchanged Action or FBX preserves that mapping. FBX export was not exercised.

**Provenance:** the evidence binds every source sha256 and the S0 checkpoint. The unchanged MotionGraph, AnimationClip and
Action carry none of this.

## 11. Representative positive and negative cases (`results/review/review.json`, reporting only)

All edits below are MOCK edits.

**Reliable observed joint that must stay unchanged.** Frame 150, `left_knee`:
- score 0.688, valid, input-supported;
- after E1 its 3D value is byte-identical, with origin 0.

**Input-unsupported joint permitted for a mock edit.** Frame 113, `left_wrist`:
- score 0.152, invalid, xy (707.1, 247.4) px;
- z changes from −0.1705 to −0.1605 m, and the origin becomes 1.

**Low-score observation not assumed wrong.** The `left_wrist` in frames 109–118, scores 0.152–0.299:
- y is 451.7–467.5 px in 7 frames, near the hip at y = 440.9–445.0 px;
- y is 241.1–247.4 px in frames 113, 114 and 116, near the nose at y = 195.5–201.8 px.

The candidate keeps the raw values with the status "observed below validity threshold" and attaches no correctness
label.

**Score above 1.** Frame 0, `left_ankle`: score 1.0217, valid, preserved.

**Conflicting directive.** RS2 permits `right_knee` in frames 90–110 while frames 0–99 are preserved, so it fails with
`LOCK_CONFLICT` on 10 cells.

**Source edit with no rig effect.** A1 `right_wrist` at frame 110: all 8 channel quaternions are unchanged (0.0°).

**Source edit with a rig effect.** B1 `left_wrist` at frame 110:
- lowerarm_l moves 0.857°, from (−0.5698, 0.0633, 0.3026, 0.7614) to (−0.5759, 0.0627, 0.3005, 0.7577);
- upperarm_l moves 0.048°;
- the other 6 channels are unchanged.

**Timebase mismatch.**

| Frame | Source time | Action time | Action key |
|---|---|---|---|
| 100 | 2.00 s | 3.33 s | 101 |
| 198 | 3.96 s | 6.60 s | 199 |

**Unresolved subject identity.** A copy from frame 108 to 109 is rejected with `IDENTITY_UNVERIFIED`. The subject status
records that the per-frame association is unavailable.

The renders are diagnostic replays of source-to-rig influence. They are not evidence of quality.

## 12. Alternative explanations and unresolved uncertainty

**The D is driven by my instrument, not by the separation.**
- The frozen repeat check omitted a known path-induced control field.
- The rig tolerance (1e-9) was declared below float32 resolution.
- Neither failure says anything about H1. Equally, neither can be repaired inside this Batch without revising acceptance
  criteria after seeing the outcomes, so I did not repair them.

**The authority rule is a declared choice.**
- The tail-joint projection judges each channel by the joints its local value places.
- A world-position reading would judge differently. Under it, a parent rotation moves every descendant joint, so a
  preserved wrist would be violated by an upper-arm change.
- The evidence shows which channels change. It does not settle which reading of "preserve" a user intends.

**Influence is local.** One magnitude (0.01 m), one direction per condition, two windows plus one clip. No global
bone-dependency model is inferred. The tolerance-level C1 changes show that numerical resolution itself enters
influence classification.

**Narrow data.**
- The real clip is single, private and not unseen, with one performer.
- Its only low-evidence joint is the left wrist.
- Its two failure windows are the only edit-permission case.
- The mixed windows are `real_animcv_observation` on 3DPW images, not a benchmark-detector regime.

**Per-frame subject identity is lost before serialization.**
- Recovering it would need a new tracker run, which is not authorized.
- It is unknown whether the 38 low-score wrist cells belong to the same person throughout.

**The 3DPW adapter's per-joint 2D derivation was not traced.** The benchmark cells are labelled with the adapter's kind
only.

**The synthetic programme's evidence digests depend on its own source path.** This is by construction, and the repeat
report lists it.

## 13. Changed files and tests

**Changed files.** All are new; no pre-existing tracked file was modified.
- `scripts/w114_source_preserving_base.py`: candidate evidence, constraints, lineage, edits, reverts, output association
  and authority check.
- `scripts/w114_rig_influence.py`: frozen-rig harness. It replicates the W105 `build_clip` real-clip frame and the
  `window_case` mixed frame, writes the Blender case-file layout and computes channel deltas.
- `scripts/w114_source_preserving_contract.py`: stages preflight, declare, build, blender-compare, compare, decide and
  manifest.
- `scripts/w114_source_preserving_posthoc.py`: written after the outcomes. Stages repeat-pathnorm, c1-float32 and review,
  all labelled. It changes no gate.
- `tests/test_w114_source_preserving_base.py`.
- This Worklog.

**Tests.** All are AGENT-RUN; none has been reviewer-verified.

`tests/test_w114_source_preserving_base.py`: **12 passed** (`w114_tests.xml`). It covers:
- score > 1 preserved; clipping detected as `SOURCE_MISMATCH`;
- read-only evidence and digest binding;
- unknown, manufactured and NaN-score rejections;
- edit locality, reversal and lineage; tampered revert; revert conflict;
- permission, conflict, identity, time, frame, unit and immutable-observation codes;
- rig authority on the toy rig, plus association validation;
- the full synthetic control suite (25/25 applicable) and the synthetic programme;
- quaternion delta semantics;
- byte reproduction of the W105 real-clip rig export.

**Regressions: 171 passed** (`w114_regression.xml`). Files:
- `test_w113_animation_base_contract`;
- `test_animation_semantics` and `test_animation_semantics_v2`;
- `test_framepose_bridge`, `test_frame_pose_observations`;
- `test_motion_builder`, `test_motion_graph`;
- `test_retarget_solver`, `test_fk_solver`, `test_keyframe_collapse`;
- `test_w105_rig_utility`.

**Failed:** none.

**NOT RUN:**
- the full test suite;
- Blender-dependent test suites (Blender was exercised only through the W105 render and audit stages);
- FBX export/import;
- any game-engine import.

No external runtime or engine compatibility is claimed.

## 14. Reproduction (PowerShell on Main, repo root)

```
$py="C:\Projects\AnimCV\.venv_build_full\Scripts\python.exe"; $bl="C:\Program Files\Blender Foundation\Blender 5.1\blender.exe"
$o="C:/Users/dna10/animcv-w114/results"; $s="f5fb2451b06126f9133aeccaabdd68f0eeb3fca1243a6e45fda36ab6851f1aa1"
$d="--declaration $o/declare/declaration.json --declaration-sha $s"
& $py -X utf8 scripts/w114_source_preserving_contract.py preflight --out-dir $o/preflight
& $py -X utf8 scripts/w114_source_preserving_contract.py declare --preflight $o/preflight/preflight.json --out-dir $o/declare
& $py -X utf8 scripts/w114_source_preserving_contract.py build $d.Split() --tag r1 --out-dir $o/build_r1
& $py -X utf8 scripts/w114_source_preserving_contract.py build $d.Split() --tag r2 --out-dir $o/build_r2
& $py -X utf8 scripts/w114_source_preserving_contract.py compare --first $o/build_r1 --second $o/build_r2 --out-dir $o/repeat
& $bl -b -t 2 --python-exit-code 1 -P scripts/w105_rig_blender.py -- render --rig examples/skinned_mannequin/BaseRig_Skinned.fbx --rest examples/e2e_demo/baserig_blender_rest.json --show-mesh --size 480 --ortho-scale 3.2 --equivalent-rig-tolerance 0.0001 --modes FIXED,RUNTIME --cases $o/build_r1/w114_rig_cases.json --out-dir $o/rig_replay/render
& $bl -b -t 2 --python-exit-code 1 -P scripts/w105_rig_blender.py -- audit --render-dir $o/rig_replay/render --out $o/rig_replay/render_audit.json
& $py -X utf8 scripts/w114_source_preserving_contract.py blender-compare $d.Split() --build-dir $o/build_r1 --audit $o/rig_replay/render_audit.json --out-dir $o/blender_compare
& $py -X utf8 scripts/w114_source_preserving_contract.py decide $d.Split() --preflight $o/preflight/preflight.json --build-dir $o/build_r1 --repeat $o/repeat/repeat_report.json --blender $o/blender_compare/blender_compare.json --out-dir $o/decision
# labelled post-hoc (no CASE assigned)
& $py -X utf8 scripts/w114_source_preserving_posthoc.py repeat-pathnorm --frozen-repeat $o/repeat/repeat_report.json --first $o/build_r1 --second $o/build_r2 --out-dir $o/posthoc_repeat_pathnorm
& $py -X utf8 scripts/w114_source_preserving_contract.py decide $d.Split() --preflight $o/preflight/preflight.json --build-dir $o/build_r1 --repeat $o/posthoc_repeat_pathnorm/repeat_pathnorm_report.json --blender $o/blender_compare/blender_compare.json --out-dir $o/posthoc_decision_inputs_pathnorm
& $py -X utf8 scripts/w114_source_preserving_posthoc.py c1-float32 $d.Split() --out-dir $o/posthoc_c1_float32
& $py -X utf8 scripts/w114_source_preserving_posthoc.py review $d.Split() --out-dir $o/review
& $py -X utf8 scripts/w114_source_preserving_contract.py manifest --base C:/Users/dna10/animcv-w114 --declaration-sha $s
& $py -X utf8 -m pytest tests/test_w114_source_preserving_base.py -q
```

Each stage refuses an existing output directory. The declaration embeds this host's absolute input paths.

Key artefacts (SHA-256):

| Artefact | SHA-256 |
|---|---|
| `decision/decision.json` | `57ea33fc…` |
| `repeat/repeat_report.json` | `cfd8a7d6…` |
| `posthoc_repeat_pathnorm/…` | `f2c2e044…` |
| `posthoc_decision_inputs_pathnorm/decision.json` | `1db9daa5…` |
| `blender_compare/blender_compare.json` | `51829daf…` |
| `build_r1/build_report.json` | `f677096b…` |
| `build_r1/w114_rig_cases.json` | `5c314c0a…` |
| `rig_replay/render_audit.json` | `8b1632d4…` |
| `posthoc_c1_float32/…` | `2dcf13dc…` |
| `review/review.json` | `87694561…` |

`evidence_manifest.json` (`9f2af3d64aaae2f021d66f5b94beae1982e74097ca49ab90e20a69fb3208a2a0`) covers 44 files, plus the
6,368 PNGs through the combined digest `4466c977…`.

## 15. Explicit non-claims and scope limitations

- No CASE was obtained, so no viability claim is made for H1.
- Nothing here shows Generator quality, reconstruction correctness, calibrated reliability, production-ready editing or
  real editing integration.
- The mock edits are synthetic offsets of stored values, not corrections.
- The accepted RS1_arm rig edit shows only that a source-backed change can be confined to its authorized channels under
  one frozen configuration. It is not animation-quality evidence.
- The representation is scratch-only and is not an implicit production format.
- No sidecar survival through Blender/FBX or any engine was tested.

**Starting commit:** `fd24ef0`. **Commits on this branch:**
- `dc5bfd3`: the pre-outcome instrument;
- the commit adding this Worklog and the post-hoc script; its hash is the branch head after publication;
- a final commit that adds only the COMPLETE marker.

## 16. Negative findings

- The frozen W114 instrument did not produce a trustworthy verdict, for two instrument-design reasons:
  - an incomplete list of path-induced fields;
  - a tolerance set below float32 resolution.
- Joint-level permission does not map one-to-one to bone authority in the frozen real rig. A single-joint `left_wrist`
  change also moves `upperarm_l` (two-bone IK), so the W113-retained wrist-only permission cannot authorize a rig edit
  under the declared rule.
- W113's conservative dependency map is wrong in both directions:
  - it lists torso dependencies that produce bit-identical channels (5 of 8 channels for each shoulder perturbation);
  - it omits `upperarm_l` for the wrist.
- The name `left_wrist` does not identify ownership. In the W102 mixed contract an S0 left-wrist change has no effect at
  all, because J1X owns the arm.
- Existing path:
  - raw scores > 1 reach MotionGraph unchanged;
  - the FramePose bridge derives negative "uncertainty";
  - invalid joints are zero-filled before S0 and carry confidence 0 in the bridge;
  - AnimationClip and Action drop score, validity, identity and source time.
- The benchmark bank stores zero placeholders at all 113 invalid cells.
- Per-frame subject identity is unavailable on every authentic population, so cross-frame operations cannot run there.

## 17. Resource and environment constraints

**Cost:**

| Item | Cost |
|---|---|
| Each build (synthetic + real + rig conditions + existing path + benchmark) | 14.5 s |
| Blender render (8 cases × 199 frames × 2 modes × 2 views) | 2 min 30 s |
| Audit | a few seconds |
| Focused tests | 0.5 s |
| Regressions | 4.7 s |
| Storage under `C:\Users\dna10\animcv-w114\` | 1.05 GB, mostly diagnostic PNGs; local only, not published |

**Environment:**
- Main only. No GPU inference, network access, model download, package change, elevated access, or LabServer63/69 use.
- Every array came from a hash-verified stored artefact.
- The source clip was not copied or published.
- Single branch `arch/single_frame_first`. No branch, rebase, reset or force operation.

AUTOMATION_BATCH_STATUS: COMPLETE
BATCH_NUMBER: 114
