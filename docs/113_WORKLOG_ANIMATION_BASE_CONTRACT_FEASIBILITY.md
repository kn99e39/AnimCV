# Worklog — Animation Base Contract Feasibility (2026-11-04)

Batch 113. Evidence-only architecture diagnostic on `arch/single_frame_first`, run unattended on Main (local RTX 5080
host; no GPU inference was needed). No `src/`, schema, model, threshold, rig, FBX/Blender exporter, default, historical
artefact or `docs/next_batch.md` was changed.

## Human assessment

**Verdict: CASE D — INVALID.** This is not a viability verdict.

**Reason.** The frozen declaration required a detector confidence in `[0, 1]`. Real data violates that assumption.
- Real clip: the RTMPose scores stored for `real_animcv_observation` exceed 1 in 69 of 3,383 present joint cells.
  - These cells fall in 53 frames: head 6, left ankle 45, right ankle 17, right elbow 1.
  - The maximum is 1.0791, and all 69 cells are `input_valid`.
- Benchmark: the 3DPW shipped-detector scores exceed 1 in 1,350 of 353,457 valid bank cells (maximum 1.2852).

As a result, the unchanged real view fails its own positive control with `CONFIDENCE_AMBIGUOUS`.
- The same happens in two independent builds, and again in a third run after the post-freeze code edit.
- Under the frozen precedence (D > B > A > C), a failed declared control is D.
- The gate was not relaxed to obtain a verdict.

**What the D does tell the reviewer.** Every consumer in the repository calls this score "confidence", but it is not a
probability. The real clip shows the consequence:
- `pose.framepose_bridge` derives `depth_uncertainty = 1 − confidence`, which gives 69 negative values (min −0.079).
- MotionGraph passes the raw score through unchanged: all 3,184 MotionGraph cells (16 canonical joints × 199 frames,
  without the derived thorax duplicate) are identical to the source, including the 69 values above 1.
- Neither path declares what scale its thresholds (visibility 0.3, retarget mean-confidence 0.30) assume.

This is a measured, reproducible finding. It is not an animation-quality result.

**Post-hoc partial diagnostics.** These are labelled, assign no CASE, and use a single override: the confidence
upper bound is lifted. With that override, every other declared check ran on the authentic GT-free clip, and:
- The 38 low-evidence left-wrist cells (frames 109–118 and 154–181) can be isolated as the only cells eligible for edit.
- A synthetic mock perturbation then changed exactly those 38 `est3d` cells and nothing else, byte for byte.
- A lock-conflicting directive set and lock-violating edits were rejected, not resolved by priority.
- 19 of 20 view controls discriminated with their exact code. The 20th (`unknown_coerced_to_zero`) is NOT APPLICABLE:
  the real clip has no absent landmark and no unknown root yaw.
- The existing W105 real-clip Blender Action replays byte-identically (value digest).

The post-hoc pass also left three ownership gaps visible:
1. No serialized per-frame actor/track identity.
2. Joint-level permission does not map to bone-level edit authority. Every W105 bone channel depends on the shared torso
   frame, so a 38-cell wrist permission reaches 0 editable bone-frames under the declared conservative rule.
3. The Action keeps frame numbers but runs at 30 fps against a 50 fps source, so source time exists only in a sidecar.

Nothing here implies a production-ready Generator/Editor, calibrated reliability or real editing integration.

**Regime labels.**
- `real_animcv_observation`: the W102 M1 MMPose+RTMDet observation of `examples/production_demo/source.mp4`. GT-free,
  private, not unseen.
- `benchmark_detector_observation`: 3DPW shipped detector, one VALIDATION sequence, never pooled.
- `oracle_geometry` / `synthetic_projection`: an 8-frame synthetic contract sequence. Contract test only.
- No evaluation GT was examined anywhere in W113.

## 1. Scope, starting point and final revision

- **Task scope (as executed).** Diagnose whether the existing FramePose observation/reconstruction, MotionGraph,
  AnimationSemantics and Rig/AnimationClip handoff can furnish a minimum reference-guided, reliability-aware,
  user-constrained Animation Base. Allowed means:
  - static ownership tracing;
  - a scratch-only versioned join;
  - synthetic directives with a deterministic mock edit;
  - negative controls;
  - GT-free real replay;
  - an existing-path rig association;
  - repeatability.
- **Not authorized and not done:**
  - production schema changes, a Generator/Editor, training or inference;
  - threshold changes, mask repair or IK/retarget changes;
  - UI or runtime integration;
  - Batch 114.
- **Starting commit:** `011d79112ade873cc2097d694c04cdfe6c0031ce`.
- **Instruction blob:** `docs/next_batch.md` = `ce299256622abb2ece5cbbd7196b2d6b8c59645f` (Batch 113). Verified before any
  change.
- **Preconditions:** W112 Worklog ends with `AUTOMATION_BATCH_STATUS: COMPLETE` / `BATCH_NUMBER: 112`. The remote was in
  sync at start.
- **Main-local lock** (`state/active_batch.json`): batch 113, this blob, this start commit, executor PID 31648.
  - It is a Main-local lock only and does not prevent executors on other hosts.
  - Two other interactive `claude.exe` processes were present on Main. They were not lock claimants.
- **Commits:**
  - `9fe6aa0`: the instrument and tests, committed after the declaration was hashed and before any real/benchmark
    outcome existed.
  - The commit adding this Worklog: the post-hoc script, the frozen-gate test and the constant refactor below.
  - A final commit adds only the COMPLETE marker.
- **Post-freeze code edits** (disclosed; neither changes the frozen behaviour):
  1. In `scripts/w113_animation_base_view.py`, the literal `1` in the confidence gate became `CONFIDENCE_UPPER = 1.0`.
     - Only the post-hoc script overrides it.
     - Re-running the frozen build with the edited code (`build_r3_postfreeze_code_check`) fails at the same call with
       the same code.
  2. Added `scripts/w113_animation_base_posthoc.py` and one test pinning the frozen gate.

## 2. Inputs, hashes and runtime (preflight `272b11be…`)

All authenticated before the declaration and re-verified by every later stage:

| Input | SHA-256 |
|---|---|
| W102 `results/historical_video/observations_predictions.npz` (landmarks, input_2d/valid, geometry, S0 predictions, frame_index, timestamp) | `6784357f7c6145e4b60e2e9201e32accd3ae00001486283a1e76f0adae61e7bd` |
| W102 `results/historical_video/report.json` | `e6e643071aadc1def75d86697acaeed56a0db6014c6620b42908a6359408b4fa` |
| W102 `results/video_inspection/inspection.json` | `dd25b7af0183754b7e78d07df9df71c7f718e43fb3f7489545d9c8b5c10d3149` |
| `examples/production_demo/source.mp4` = W102 `historical_source.mp4` | `0dda8a5f16c67992330a1a8534cf0bfefeb5e0f2aec438577a9e1dbace773275` |
| W105 `evidence_manifest.json` | `cb0eca93c7847a70d497bd36c57856bd8295f3d8e3a7a72913ebf876ebe96899` |
| 3DPW bank index / arrays (content digest `75519e63…`) | `6fe13cdc…` / `909090f7…` |
| Frozen S0 VALIDATION prediction (W101 = W102) | `402d70a2bed79ae31919ff86905e19c8972108ea988d6cfa5deac514562132b9` |

Cross-checks:
- The W102 manifest entries match the NPZ and report, and the report binds the NPZ hash.
- The W105 manifest entries match `video_sources.npz`, `video_cases.json` and `render_clip_audit.json`.
- The S0 checkpoint equals `run_real_observation_s0.S0_CHECKPOINT_SHA`.
- The W102 report and inspection agree on the 199 decoded-BGR frame digests, the clip hash and the timestamps.
- The full input list with paths and hashes is in `declare/declaration.json`.

Runtime: Python 3.11.9, numpy 1.26.4, torch 2.13.0+cu130 (imported, not used for inference), matplotlib 3.11.0,
Blender 5.1.0, Windows 10.0.26200.

## 3. Frozen declaration (`declare/declaration.json`, SHA-256 `c3f7485c73fe9b9d6a6933fd9c1f45357b349b423177cdb248fe48c66258ef6b`)

It was written with `outcomes_inspected: false`, and every later stage refuses to run without this hash. It contains:
- **Frame selection:**
  - real: all 199 decoded frames, original order, `t = i/50`;
  - benchmark: the first VALIDATION sequence in bank order;
  - synthetic: 8 frames at 25 fps.
- **Joint order:** H36M-17. For each joint, how the MMPose adapter produces it in 2D:
  - limb joints are observed;
  - `head` = nose, renamed;
  - `neck` and `thorax` = shoulder midpoint (thorax duplicates neck);
  - `pelvis` = hip midpoint;
  - `spine` = neck–pelvis midpoint.
- **Status vocabulary:**
  - 2D observation: observed-valid / below-threshold / absent (UNKNOWN);
  - 3D estimate: inferred-3D input-supported / input-unsupported / UNKNOWN non-finite;
  - other: derived, unavailable, synthetic-user-directive, synthetic-edit.
- **UNAVAILABLE by design:** root translation, ground height, physical contact, calibrated 3D confidence, actual 3D
  error, and world/camera motion.
- **Minimum fields:**
  - M1: 2D xy + confidence + status;
  - M2: observed vs derived 2D;
  - M3: 3D + frame/unit/finite;
  - M4: observed vs inferred;
  - M5a: sequence/frame/time;
  - M5b: actor/track per frame;
  - M6: source provenance;
  - M7: directive (synthetic; no production producer);
  - M8: UNAVAILABLE classification.
- **Controls and their exact expected codes:**
  - 20 view controls, hard-reject;
  - 13 directive/edit controls;
  - the cases that must become UNKNOWN rather than reject (absent landmark; finite S0 output on an invalid input,
    labelled unsupported; non-computable yaw; contact-like evidence without a regime-matched calibration).
- **Real directive set RS1:**
  - preserve all joints in frames 0–99;
  - preserve the right arm in all frames;
  - permit `left_wrist` `est3d` cells in maximal runs where `input_valid` is false and frame ≥ 100;
  - default prohibit.
- **RS2:** RS1 plus a permit for `right_knee` in frames 90–110. Expected result: `LOCK_CONFLICT`.
- **Edit requests:** `E_ok` and its rejections. An empty permitted set would make the test vacuous and rule out A.
- **Rig handoff:**
  - the unchanged W105 replay;
  - the bone→joint dependency map, taken from `w105_rig_utility.build_clip` and `torso_frames`;
  - the bone permission rule (locked if any dependency cell is preserved);
  - the legacy CLI replay.
- **Counterexample definitions and failure windows:** 109–118 and 154–181.
- **CASE logic.** B applies only to a field that is present in an upstream **serialized** artefact and lost downstream
  with no identity-safe scratch recovery. An internal, unserialized variable counts as a missing producer, which is C.

## 4. Ownership matrix and actual wired paths (P0, static; CODE)

| Quantity | Boundary | Status | Location |
|---|---|---|---|
| 2D xy/conf/visible | MMPose adapter → PoseSequence | PRODUCED; midpoints/renamed joints carry no derived flag | `src/pose/mmpose_adapter.py:280-339` |
| 2D xy/conf/visible | PoseSequence → MotionGraph | PASSED | `src/motion/motion_builder.py:27-40` |
| 2D xy/conf/visible | MotionGraph → AnimationClip | DROPPED; only `min(conf_a, conf_b)` per bone-frame; 0.0 = held rotation | `src/retarget/fk_solver.py:108-158` |
| 2D for FramePose | real observation → S0 input (script-only) | PASSED/AMBIGUOUS; invalid joints zero-filled (xy=0, conf=0) | `scripts/w101_reference_estimator.py:60-75` |
| confidence | FrameBank → LiftedPoseSequence | AMBIGUOUS; `depth_uncertainty = 1 − conf` is a 2D proxy | `src/pose/framepose_bridge.py:101-110` |
| 2D xy/conf | Lifted → AnimationSemantics v1/v2 | DROPPED; only `joint_observation_valid`, `joint_in_frame` | `src/motion/animation_semantics.py:218-243` |
| 3D estimate | S0 → Lifted → AnimationSemantics | PASSED (camera_root_relative, metres) | `src/motion/animation_semantics_v2_bridge.py:101-121` |
| 3D estimate | → MotionGraph / CLI AnimationClip | UNAVAILABLE; `position_3d` is mixed-unit or None | `src/motion/motion_builder.py:33-37`, `src/app/cli.py:951-996` |
| 3D estimate | → W105 S0-only Action (scripts) | PASSED as rotations only | `scripts/w105_rig_utility.py:657-720` |
| 3D confidence | everywhere | UNAVAILABLE | `src/pose/framepose_bridge.py:107` |
| sample/sequence/actor/frame/time/image/provenance | FrameSample | PRODUCED (benchmark bank) | `src/framepose/contract.py:105-160` |
| sequence/actor id | PoseSequence / MotionGraph | DROPPED; `source_metadata = {'pose_source': path}` | `src/motion/motion_graph.py:96-121` |
| per-frame tracked subject | MMPose adapter | PRODUCED internally, never serialized | `src/pose/mmpose_adapter.py:189-232` |
| identity + source | AnimationClip | DROPPED; name `Generated_Motion` | `src/retarget/solver.py:66-96` |
| timestamp | → AnimationClip | DROPPED (derivable from fps) | same |
| time | → W105 Action | AMBIGUOUS; keys at `frame_index+1`, scene 30 fps vs source 50 | W105 §10 |
| root yaw / contact-like | AnimationSemantics | PRODUCED (derived / calibration-bound) | `animation_semantics_v2_bridge.py:24-98` |
| root translation / ground height | AnimationSemantics | UNAVAILABLE (explicit; refused on load) | `animation_semantics.py:110-150` |
| `MotionFrame.locked` | MotionGraph | PRODUCED, always False; no producer sets it | `motion_builder.py:42-50` |
| `MotionFrame.locked` | → optimize | DROPPED: `collapse_animation_clip(locked_frames=…)` exists, but the CLI never passes it | `src/app/cli.py:999-1008` |
| `BoneMappingEntry.locked` | mapping | PRODUCED, never consumed | `src/rig/bone_mapping.py:25-47` |
| per-joint lock / edit permission / edit provenance | everywhere | UNAVAILABLE | — |

**Wired paths.** `src/app/cli.py` and `src/ui/gui_app.py` wire only:
`estimate-pose` → PoseSequence → `build-motion` → MotionGraph → `retarget` → AnimationClip → `optimize` →
`export-blender`. That path is 2D, with optional mixed-unit depth.

The other paths exist only as library code that scripts call:
- the FramePose S0 → LiftedPose → AnimationSemantics bridges;
- FramePose → rig (`retarget/framepose_fk.py`, `framepose_two_bone_ik.py`, `framepose_object_owner.py`,
  `motion/torso_frame.py`);
- the real-observation → S0 input conversion.

No entry point references them.

Two further observations:
- The benchmark FrameSample stratification is computed from `target_3d` (`src/framepose/bank.py:138-160`). Its fields
  (`facing`, `facing_angle_degrees`, `forward_depth`, `hip/shoulder_forward_depth_m`, `invalid_target_count`,
  `min_bend_angle_degrees`) are therefore GT-derived, and a consumer that reads sample metadata wholesale would leak
  GT. The view excludes them.
- Architecture_v2 §1.3 makes AnimCV single-character.

## 5. Frozen run (the verdict)

| View | Frozen `validate_view` |
|---|---|
| synthetic (8 × 17) | PASS |
| real (199 × 17) | `CONFIDENCE_AMBIGUOUS` |
| benchmark (319 × 17) | `CONFIDENCE_AMBIGUOUS` |

- `build_r1` and `build_r2` stop at `validate_view(real)`. Their logs are identical (path-normalized), and so are their
  synthetic artefacts: `view_synthetic.npz` `5eaa1e19…` and `view_synthetic_edited.npz` `9d84ed05…`.
- `decision_frozen/decision_frozen.json` (`2369289f…`) records CASE D and the confidence-domain evidence.

## 6. Post-hoc partial diagnostics (labelled; override `CONFIDENCE_UPPER = inf`; no CASE)

### 6.1 Real clip coverage (GT-free; numerator / denominator)

| Quantity | Count (of 3,383 joint-frame cells = 199 × 17) |
|---|---|
| present 2D | 3,383 (absent/UNKNOWN: 0) |
| observed valid | 3,345 |
| present below the validity threshold | 38, all `left_wrist`: 10 in frames 109–118, 28 in 154–181; score 0.152–0.299 |
| derived 2D (pelvis, spine, thorax, neck) | 796 (thorax 2D equals neck exactly in all frames) |
| visible flag ≠ `input_valid` | 0 |
| `in_frame` | 3,345 (it requires valid input) |
| 3D finite | 3,383 |
| 3D input-supported / unsupported / UNKNOWN | 3,345 / 38 / 0 |
| root yaw known (v2 `current_root_orientation`, camera frame) | 199 / 199 frames |
| joinable / provenance-covered / lockable | 3,383 / 3,383 / 3,383 |
| RS1: preserved / eligible to edit / prohibited | 1,997 / 38 / 1,348 |
| eligible cells by status | all 38 are inferred-3D input-unsupported |

Other structural checks:
- Upstream zero-fill: the S0 input zero-fills all 38 of 38 invalid cells. The raw scores survive only in the W102
  landmarks, and the view recovers them by a row join within the same hashed NPZ.
- S0 output is pelvis-centred to within 0.0175 m. The median thigh length is 0.353 m, consistent with metres.

### 6.2 Controls

| Population | View controls | Directive/edit controls |
|---|---|---|
| synthetic | positive PASS; 20/20 exact code | 14/14, including `EDIT_UNKNOWN_TARGET` and `LOCK_CONFLICT` |
| real | positive PASS (post-hoc); 19/20 exact code; `unknown_coerced_to_zero` NOT APPLICABLE (no UNKNOWN cell exists) | 13/13 |
| benchmark | positive PASS (post-hoc); 20/20 exact code | — |

Every negative control has an exact expected code, and each one is distinct:
- identity and time: `IDENTITY_MISMATCH`, `TIME_ORDER`, `TIME_DUPLICATE`, `FRAME_ORDER`, `FPS_MISMATCH`;
- joints: `JOINT_DUPLICATE`, `JOINT_MISSING`;
- provenance: `PROVENANCE_STALE`, `PROVENANCE_MISSING`, `DIGEST_STALE`;
- coordinates: `COORDINATE_MISMATCH` (frame and unit);
- confidence: `CONFIDENCE_MISSING` (field and cell), `CONFIDENCE_MANUFACTURED_3D`;
- unknown values: `UNKNOWN_COERCED_TO_ZERO`, `UNAVAILABLE_COERCED` (zero root translation);
- GT: `GT_LEAKAGE` (`target_3d` and stratification).

The mixed-unit probe on a serialized MotionGraph is rejected with `MIXED_UNIT_3D`.

### 6.3 Mask locality and preservation (real, RS1, `E_ok`, synthetic perturbation +0.01 m in z, not animation)

- Changed `est3d` cells: 38, exactly the 38 requested, all inside the permitted set.
- Per-cell provenance: switched to `synthetic_mock_edit` on exactly those 38 cells.
- Everything else is byte-identical:
  - all unedited 3D values;
  - all other arrays: 2D, confidence, validity, in-frame, frame index, timestamp, yaw;
  - all metadata apart from the edit log and the parent digest.
- UNKNOWN masks are preserved.
- The edited view validates, records its parent digest and round-trips exactly.

Rejections, each with its exact code and with no partial application:
- `RS2` → `LOCK_CONFLICT`;
- an edit on (50, `left_wrist`) → `LOCK_CONFLICT_EDIT`;
- an edit on (150, `left_knee`) → `EDIT_NOT_AUTHORIZED`;
- an edit of `obs2d` → `OBSERVATION_IMMUTABLE`;
- an edit against a stale digest → `DIRECTIVE_STALE_VIEW`.

### 6.4 Counterexamples and failure windows (GT-free; not verified errors)

**Structurally valid yet possibly wrong 3D** (S0 limb step > 45° with every limb joint `input_valid`): **0 events**.
- All of W102's four S0 limb steps above 45° involve invalid-input wrist frames.
- So no high-confidence, structurally valid counterexample is established on this clip.

**Low-confidence yet useful raw 2D: not established** without GT. In frames 109–118 the below-threshold left wrist
behaves two ways:
- In 7 frames it lies near the hip: y = 452–468 px, against a hip at 441–445 px.
- In frames 113, 114 and 116 it lies at head height: y = 241–247 px, against a nose at about 196–202 px. This is W105's
  wrist jump.

The same low-confidence channel is therefore plausible in some frames and implausible in others. In both windows every
wrist cell is input-unsupported inferred 3D, and yaw stays known.

### 6.5 Existing serialization paths: loss and recovery

**Legacy CLI path, run with unchanged code.** The input was a PoseSequence reconstructed losslessly from the W102
landmarks (x, y, score, visible; fps 50; threshold 0.3). It then went through `MotionGraphBuilder` → JSON →
`RetargetSolver(validate_quality=False)` with `examples/production_demo/mapping.json` and `examples/BaseRig.fbx`.

MotionGraph:
- xy, confidence and visible are exact on all 3,184 cells (16 canonical joints × 199); timestamps are exact.
- `locked = True` on 0 of 199 frames.
- No `position_3d`, so 3D is UNAVAILABLE in MotionGraph.
- No derived-2D flag, no sequence/actor id and no hash. `source_metadata` holds only the pose path.

AnimationClip (8 tracks × 199 = 1,592 samples):
- No timestamps, though fps × frame index reproduces them exactly. No source identity: the name is `Generated_Motion`.
  No lock or permission field.
- Confidence:
  - 38 samples have confidence 0.0. All of them are `lowerarm_l` at a missing or invisible wrist, and all hold the
    previous rotation. None is a measured zero.
  - In the 1,554 observed samples, confidence equals the minimum source score.
  - No sample is above 1, because of the min over two joints.
- The input quality gate passes at the CLI defaults.

Sidecar association by unique `frame_index`: 0 unmatched samples. The clip carries no source digest, so a sidecar
must bind (clip sha256, view digest) at creation time. Frame index alone is not identity-safe.

**`collapse_animation_clip(locked_frames=…)`** is the only existing lock consumer. RS1 preserves the right arm in every
frame, so a frame-wide lock marks all 199 frames as locked:
- With those frames locked: 1,592 keys kept.
- CLI default, no locks: 158 keys kept.

A joint-level directive therefore degrades to "every frame locked", and the lock only means key retention.

**AnimationSemantics pieces, from existing bridges on a scratch FrameBank.** The bank's NaN/False target placeholders
are required by the class and are never read; a test enforces this.
- `build_h0_lifted_sequence`: frame index, timestamps and articulation are exact; `observation_valid` equals
  `input_valid ∧ finite`.
- `in_frame_mask` and the yaw known/unknown flags equal the view's.
- The 38 invalid cells carry confidence 0 in the bridge, so the raw 0.152–0.299 scores are lost there. They are
  recoverable by the view's row join.
- 2D xy/score, visible flag, derived kind, actor, frame digests, lock and edit provenance are all absent from the
  semantics.
- A full `AnimationSemanticsV2` was **not built**: it requires a contact calibration. None exists for this regime at
  50 fps, and transferring the 3DPW TRAIN calibration is not authorized. Contact-like evidence is therefore UNKNOWN.

### 6.6 Rig handoff (P4)

**Source identity.**
- W105 `video_sources.pred_M` is byte-equal to the W102 S0 predictions, and `input_valid` and the timestamps are equal.
- `video_cases` frames match `frame_index` and `time_s` exactly.
- The cases carry no validity or confidence, and their identity label is `single_subject_GT_free`.

**Fresh replay.** I re-ran the unchanged `scripts/w105_rig_blender.py` render and audit (Blender 5.1) on W105's
`video_cases.json` into scratch. The Action value digests equal W105's:

| Action | Value digest |
|---|---|
| FIXED (M_Main) | `4d168d95…` |
| RUNTIME | `9f7e4e31…` |

- Keys sit at `frame_index + 1` on all 199 frames, with 32 F-curves and CONSTANT interpolation.
- Channels: object, `spine_01`, thigh/calf l/r, `upperarm_l`, `lowerarm_l`.

**Time.** The Action scene rate is 30 fps against a 50 fps source, so Action time is not source time. Source time is
recoverable only by the sidecar's frame index.

**Permission semantics.** Every channel depends on the torso frame (pelvis, both hips, thorax, both shoulders). RS1
preserves `right_shoulder` in all frames and every joint in frames 0–99. Under the declared rule, all 8 channels are
therefore locked on 199 of 199 frames, and **0 bone-frames are editable** despite 38 permitted joint cells.

No existing owner defines bone-level edit authority. A cell that is merely a dependency is not necessarily a cell that
changes: editing the wrist alone changes the `lowerarm_l` channel but not the torso. The rig-level semantics are
therefore ambiguous.

Neither Blender/FBX nor a game engine was shown to preserve any sidecar, and FBX export was not exercised.

### 6.7 Benchmark (separate; never pooled)

The sequence is `3dpw:courtyard_basketball_01:actor0` (VALIDATION; 319 rows; unique sample ids).
- Rows are 3–4 source frames apart, with `t = frame_index / 30`. The view validates this rather than assuming
  contiguity.
- 113 invalid cells become UNKNOWN because the bank keeps no raw below-threshold coordinate. Unlike the real W102 NPZ,
  that information is unrecoverable in the bank.
- Yaw is known on 319 of 319 frames.
- The GT-derived stratification is excluded, and the leak check found nothing.
- All 20 controls behave as declared.
- The coverage counter labels all 5,423 cells "not directly observed". This is a labelling artefact: the 3DPW adapter's
  per-joint 2D derivation was not traced.

### 6.8 Decision inputs on the post-hoc build (no CASE assigned)

- **Validity gates:** all true except `real_controls`, which fails only because of the NOT APPLICABLE control.
- **B hits:** none.
- **A conditions unmet:**
  - `valid` (the same control);
  - `M5b_has_serialized_producer`;
  - `rig_permission_semantics_unambiguous`.
- **A conditions met:**
  - producer fields joinable;
  - directives enforced non-vacuously (38 cells);
  - rig replay identical;
  - rig association.
- File: `posthoc_gates/posthoc_gate_inputs.json` (`12c1ed90…`).

## 7. Identity, time and coordinate audits

- **Identity:**
  - The real sequence is bound by the clip hash and 199 decoded-frame digests.
  - The actor is `single_tracked_subject` under the single-character contract. The tracker averaged 1.76 person
    candidates per frame (W102), and the per-frame selection is not serialized. A subject switch could not be detected
    from any stored artefact.
- **Time:**
  - frame index 0–198, strictly increasing;
  - `timestamp == frame_index / 50` to within 1e-9;
  - FPS from the W102 report, equal to the inspection;
  - no gaps on the real clip; the benchmark uses 3–4 frame strides.
- **Coordinates:**
  - 2D in image `I` pixels;
  - 3D camera_root_relative metres, as stored, with no transform;
  - yaw in camera frame radians;
  - legacy `MotionPoint.position_3d` is rejected for metric use.

## 8. Repeat evidence (`posthoc_repeat/repeat_report.json`, `86759942…`)

The two post-hoc builds (`posthoc_build_r1` and `posthoc_build_r2`) were compared:
- The real, edited-real and benchmark view digests are identical: `0489b4bb…`, `42954a15…` and `e9801576…`.
- 13 artefacts are byte-identical, including the views, masks, edited 3D, reconstructed pose and AnimationClip.
- The only report differences are the 7 declared path-induced fields:
  - the synthetic source path inside the synthetic view;
  - the MotionGraph `pose_source` path.
- Each of those is identical after path normalization.
- The frozen builds r1/r2 also fail identically.

## 9. Visual review

`figures/real_timeline.png` (`1d3624ff…`) has three panels:
- 2D status per joint-frame, with derived joints labelled.
- RS1 masks: preserved, permitted and prohibited.
- The frozen S0 left-wrist z trajectory, with the 38 synthetic +0.01 m points marked "SYNTHETIC mock (not animation)".

The permitted spans coincide exactly with the two below-threshold wrist runs. The figure shows no GT and makes no
accuracy claim.

## 10. Cost and resources

| Item | Cost |
|---|---|
| Each build (synthetic + real + benchmark + legacy + semantics + rig association) | 1.3–1.5 s, about 7 ms per real frame |
| Blender replay | render 20 s; audit < 1 s |
| Focused tests | 0.4 s |
| Regressions | 4.6 s |
| Scratch storage | `C:\Users\dna10\animcv-w113\` 130 MB, mostly 796 replay PNGs; about 4 MB per build |

- `evidence_manifest.json`, SHA-256 `c4c1a6e70a848ea94533faa12c393654bd8c2d2956cdb3e52e80b28ed9491d72`, covers 76 files
  plus the PNGs through a combined digest `7d2695b2…`.
- No GPU inference, network access, package change or LabServer use. The source clip was not copied or published.

## 11. Tests

All are **AGENT-RUN**; none has been reviewer-verified.
- `tests/test_w113_animation_base_contract.py`: **37 passed**. That is 36 before the declaration, plus 1 frozen-gate
  test added afterwards. They cover:
  - exact round-trip;
  - no filling and no manufactured 3D confidence;
  - every negative control's exact code;
  - stale source file;
  - nested GT key, plus a no-false-positive check;
  - directive masks and conflict;
  - exact edit locality, plus a non-vacuous wrong-mask check;
  - edit rejections with no partial application;
  - directive identity and staleness;
  - mixed-unit rejection;
  - source-only consumer isolation (a guard raises on any `target*` read);
  - CASE precedence;
  - the run helper;
  - the frozen `[0, 1]` gate.
- **Regressions: 134 passed.** Files: `test_animation_semantics`, `test_animation_semantics_v2`,
  `test_framepose_bridge`, `test_frame_pose_observations`, `test_motion_builder`, `test_motion_graph`,
  `test_retarget_solver`, `test_fk_solver`, `test_keyframe_collapse`, `test_w105_rig_utility`.
- **Failed:** none.
- **Not run:** the full suite, Blender-dependent test suites, FBX export/import.

## 12. Failed, unknown and unsupported cases (preserved)

- `preflight_attempt1_failed`: `tasklist` output decoding failed; fixed before the declaration.
- `declare_attempt1_failed`: an empty directory left because a shell pipe masked the failed preflight's exit code.
- `build_r1` and `build_r2`: the frozen D.
- `build_r3_postfreeze_code_check`: the same D.
- The real `unknown_coerced_to_zero` control: NOT APPLICABLE. UNKNOWN-cell handling is therefore exercised only on the
  synthetic and benchmark views, never on authentic real input.
- Contact-like evidence on the real clip: UNKNOWN (no regime-matched calibration). Full `AnimationSemanticsV2`
  serialization: not executed.

## 13. Negative findings

- Detector "confidence" is an unbounded score in both the real and benchmark regimes. The `1 − confidence` proxy goes
  negative.
- The real observation → FramePose input conversion zero-fills invalid joints, and the semantics and bridge then show
  confidence 0 for them.
- MotionGraph carries no derived-2D flag, no identity and no source hash. `MotionFrame.locked` has no producer. No
  existing contract carries a per-joint lock.
- AnimationClip has no timestamps, no source identity and no permission field. Its confidence 0.0 means "held
  rotation", not a measured value.
- The only lock consumer (keyframe collapse) is frame-wide, retention-only and unwired in the CLI.
- In W105's rig handoff, a shared torso dependency turns joint-level locks into all-bone locks, and the Action timebase
  differs from the source.
- The per-frame subject selection is not serialized anywhere.

## 14. Unresolved uncertainties

- The intended scale and semantics of the RTMPose / 3DPW scores used by existing thresholds are undeclared.
- The bone dependency map was derived by W113 from code and is conservative. Dependency is not the same as change.
  Additional export-time inputs, such as calibration, are not represented.
- The 3DPW adapter's per-joint 2D derivation was not traced.
- There is one private, not-unseen, single-performer real clip, and its only low-evidence joint is the left wrist.
  Nothing generalizes to other footage, multiple actors or leg occlusion.
- No sidecar survival through Blender/FBX or a game engine was tested.
- The binary evidence is local to Main. A GitHub reviewer cannot replay it without the bundle.

## 15. Reproduction (PowerShell on Main, repo root)

```
$py="C:\Projects\AnimCV\.venv_build_full\Scripts\python.exe"; $bl="C:\Program Files\Blender Foundation\Blender 5.1\blender.exe"
$o="C:/Users/dna10/animcv-w113/results"; $s="c3f7485c73fe9b9d6a6933fd9c1f45357b349b423177cdb248fe48c66258ef6b"
$d="--declaration $o/declare/declaration.json --declaration-sha $s"
& $py -X utf8 scripts/w113_animation_base_contract.py preflight --out-dir $o/preflight
& $py -X utf8 scripts/w113_animation_base_contract.py declare --preflight $o/preflight/preflight.json --out-dir $o/declare
& $bl -b -t 2 --python-exit-code 1 -P scripts/w105_rig_blender.py -- render --rig examples/skinned_mannequin/BaseRig_Skinned.fbx --rest examples/e2e_demo/baserig_blender_rest.json --show-mesh --size 480 --ortho-scale 3.2 --equivalent-rig-tolerance 0.0001 --modes FIXED,RUNTIME --cases C:/Users/dna10/animcv-w105/results/build/video_cases.json --out-dir $o/rig_replay/render_clip
& $bl -b -t 2 --python-exit-code 1 -P scripts/w105_rig_blender.py -- audit --render-dir $o/rig_replay/render_clip --out $o/rig_replay/render_clip_audit.json
# frozen verdict (expected to stop with CONFIDENCE_AMBIGUOUS), twice
& $py -X utf8 scripts/w113_animation_base_contract.py build $d.Split() --tag r1 --rig-audit $o/rig_replay/render_clip_audit.json --out-dir $o/build_r1
& $py -X utf8 scripts/w113_animation_base_contract.py build $d.Split() --tag r2 --rig-audit $o/rig_replay/render_clip_audit.json --out-dir $o/build_r2
& $py -X utf8 scripts/w113_animation_base_posthoc.py frozen-failure $d.Split() --out-dir $o/decision_frozen
# labelled post-hoc partial diagnostics (no CASE)
& $py -X utf8 scripts/w113_animation_base_posthoc.py build $d.Split() --tag r1 --rig-audit $o/rig_replay/render_clip_audit.json --out-dir $o/posthoc_build_r1
& $py -X utf8 scripts/w113_animation_base_posthoc.py build $d.Split() --tag r2 --rig-audit $o/rig_replay/render_clip_audit.json --out-dir $o/posthoc_build_r2
& $py -X utf8 scripts/w113_animation_base_contract.py compare --first $o/posthoc_build_r1 --second $o/posthoc_build_r2 --out-dir $o/posthoc_repeat
& $py -X utf8 scripts/w113_animation_base_contract.py figures --build-dir $o/posthoc_build_r1 --out-dir $o/figures
& $py -X utf8 scripts/w113_animation_base_posthoc.py gates $d.Split() --preflight $o/preflight/preflight.json --build-dir $o/posthoc_build_r1 --repeat $o/posthoc_repeat/repeat_report.json --out-dir $o/posthoc_gates
& $py -X utf8 scripts/w113_animation_base_contract.py manifest --base C:/Users/dna10/animcv-w113 --declaration-sha $s
& $py -X utf8 -m pytest tests/test_w113_animation_base_contract.py -q
```

Each stage refuses an existing output directory, so a rerun needs a fresh base directory. The declaration embeds this
host's absolute input paths.

## 16. Changed files

All new; no pre-existing tracked file was modified:
- `scripts/w113_animation_base_view.py`: scratch view, validation, directives, mock edit, controls.
- `scripts/w113_animation_base_contract.py`: stages preflight, declare, build, compare, figures, decide, manifest.
- `scripts/w113_animation_base_posthoc.py`: the frozen-D record, the labelled post-hoc build and the gate inputs.
- `tests/test_w113_animation_base_contract.py`.
- This Worklog.

## 17. Resource and environment constraints

- Main only. Inference was neither needed nor run, and every array came from a hash-verified stored artefact.
- Blender Workbench ran for about 20 s.
- No elevated access, network access, model download, package change, or LabServer63/69 use.
- Single branch `arch/single_frame_first`. No branch, rebase or force operation.
