# Worklog — Opt-in Real-Video to Editable Blender Action with Reversible World Pin (2026-11-09)

Batch 124. An engineering-integration batch on `arch/single_frame_first`, run unattended on Main: local RTX 5080, Python
`C:\Projects\AnimCV\.venv_build_full` (3.11.9), Blender 5.1.0. `docs/next_batch.md` was read only.

Every number, render, test and manifest below is **AGENT-RUN** on Main. Nobody else has re-executed any of it. Every
visual judgement is the agent's own **AGENT REVIEW**, not animator, user or professional validation.

## 1. Official CASE and scope

**Verdict: CASE B** under the frozen rule (`results/assess/assessment.json`, SHA-256 `ed0a1d347370781214b43fdb1f9fdea07c549549ce2466069f9255d76b344119`).

- G1–G6 pass on **both** declared live sources, S1 and S3:
  - authentic live-video E2E with perception parity;
  - a usable editable baseline `.blend`;
  - a nonzero hard pin with a truthful footprint;
  - lineage and exact undo;
  - a native-rate review video.
- All integrity gates pass: G4 negative controls, G5 lineage, G8 regression/isolation, G9 source/runtime controls.
- **G7 FBX round-trip FAILs** on every exported state. This is the only reason the verdict is B and not A.
- **Scope.** One opt-in EXPERIMENTAL workflow, on the known W105 BaseRig only:
  - left-wrist WORLD_POSITION_PIN only;
  - no automatic out-of-pin editing;
  - 30 fps sources only.
- **The sources are not new evidence.** S1 and S3 are the W121/W122 engineering replays, not new independent sources.
  Nothing here is a generalisation claim, a quality guarantee or a Production default migration.

Historical CASEs are untouched (W113–W115 D, W116 B, W117 C, W118 D, W119 D, W120 B, W121 D, W122 B, W123 B). No historical
Worklog was edited (G8 `historical_worklogs_changed = []`).

## 2. Starting state, revisions, environment

| Item | Value |
|---|---|
| Instruction blob | `7b9666260ca8132f0ec2f288523fddaf67affefe`, verified first |
| Starting commit | `0d6435a34f6312023f7ea3ddc3bb888dc4134158`. It equalled `origin/arch/single_frame_first` after fetch, with no concurrent changes. |
| Pre-outcome instrument commit | `ed480ab`: code, tests, usage doc and harness, committed before any S1/S3 outcome |
| Declaration | `C:/Users/dna10/animcv-w124/results/declare/declaration.json`, SHA-256 `cf74df453d75715d2a922aaa65cce19fbab94b736c0e123f31b638176096dfb4`. It records `outcomes_inspected: false` and the SHA-256 of every instrument file. |
| Final source revision | the Worklog commit following `ed480ab`. It adds this Worklog and the dated status-doc notes. The COMPLETE marker is a separate, final commit. |
| Machine | Main, Windows 11 (10.0.26200), RTX 5080 (perception: M1 on CPU batch 1, S0 on CUDA). Python 3.11.9, numpy 1.26.4, torch 2.13.0+cu130, mmpose 1.3.2, OpenCV 4.11.0, Blender 5.1.0. No LabServer connection, no network, no training. |

**Real versus mocked stages:**

| Stage | Status |
|---|---|
| Perception (S1, S3) | **REAL live** |
| Rig adapter | **REAL** |
| Blender build/readback, edit, commit, revert | **REAL** |
| Review render/video | **REAL** |
| FBX export/reimport | **REAL** |
| Synthetic-rig edit tests | **MOCK**: the Python FK evaluator stands in for Blender and is labelled as such |
| Intake tests | synthetic MJPG clips |

**S2 debug use.** S2 (crowded juggling clip) was used before the declaration **only as a debug fixture**:

- an offline replay plus one live harness dry run;
- outputs are in `animcv-w124/debug`;
- **excluded from all evidence and denominators**.

The debug dry run surfaced and fixed two harness/stage defects before the freeze: an FBX-readback audit crash, and a
relative-path bug. Its failed attempts are retained as `fbx_attempt1_audit_crash`, `run1_attempt1_relpath` and
`harness_base_attempt1`.

## 3. What was built: the opt-in workflow and dependency/authority map

**Entry point.** `motion-tool experimental-video-action {build, propose, commit, revert, review, fbx, status}` (run with
`PYTHONPATH=src`). Usage, request format, output tree and limitations are in
`docs/EXPERIMENTAL_VIDEO_ACTION_WORLD_PIN.md`.

**Reproducible example** (the exact commands are in `results/run/command_log.json`):

```
export PYTHONPATH=src; PY=C:/Projects/AnimCV/.venv_build_full/Scripts/python.exe
$PY -X utf8 -m app.cli experimental-video-action build --clip <S1 original .webm> \
    --clip-sha256 de8feb9d515daa2bfaeed7167fdcc62028dd3f6450872c4fd81f8fb03dd6359c --source-record source_S1.json \
    --declared-fps 30 --segment 79 318 --perception-assets perception_assets_main.json --python $PY \
    --blender "C:/Program Files/Blender Foundation/Blender 5.1/blender.exe" --out C:/Users/dna10/animcv-w124/runs/S1_live
$PY -X utf8 -m app.cli experimental-video-action propose --run <run> --request S1_E1_hold_210_218.json
$PY -X utf8 -m app.cli experimental-video-action commit  --run <run> --request-id S1_E1_hold_210_218 --decision accept
$PY -X utf8 -m app.cli experimental-video-action review  --run <run> --request-id S1_E1_hold_210_218
$PY -X utf8 -m app.cli experimental-video-action fbx     --run <run> --state head
$PY -X utf8 -m app.cli experimental-video-action revert  --run <run>
```

**Dependency and authority map.** These boundaries were inspected in code before implementing.

*Reused unchanged, CALLED not rewritten:*

- Perception: the frozen W102 M1 + S0 diagnostic, run from the W102 code snapshot.
- Rig adapter: `w121_independent_transfer.fixed_slot` / `state_values_from_cases`, i.e. the W105 FIXED slot through
  `w105_rig_utility`, `w102_rig_review`, `motion.torso_frame` and `retarget.framepose_*`.
- Edit core:
  - the W116 state and contract: `make_state`, `Store`, `verify_binding`, `make_pin`, `validate_request`;
  - the W116 analytic resolver, `resolve_frame`;
  - the W116 evaluator: `evaluate_candidate`, `decide`;
  - the W118 footprint, display and gate functions: `derive_footprint`, `display_spec`, `check_display`, `decide_x`,
    `quality`.
- Blender: the W116 Blender helpers (`_setup_w105` via `BlenderExecutor.import_rig`, `_key_state`, `_audit`), the
  existing `blender.export.export_fbx` and `BlenderExecutor.import_rig`.

*Guards on the reused code:*

- The reused files are pinned by LF-normalised SHA-256 in `examples/experimental_video_action/frozen_code_sha256.json`.
  A change stops the command with `BLOCKED_FROZEN_ALGORITHM_CHANGED`.
- The perception identities are pinned in `PERCEPTION_EXPECTED`: S0 `6c12fea9…`, RTMDet `78e30dcc…`, RTMPose
  `77ffc7e8…`, gate report `d0be1936…`, lock `5bf8e2ca…` and diagnostic `6a2eff92…`. Their paths come only from a user
  JSON, never from hardcoded paths in `src/`.

*New code:*

| File | Purpose |
|---|---|
| `src/app/experimental_video_action.py` | Host orchestrator. Only stdlib is imported at module level. |
| `src/app/experimental_frozen_launch.py` | Launcher for the frozen snapshot; every path is an explicit argument. |
| `src/blender/experimental_action_stage.py` | Blender modes: `build_blend`, `readback`, `export_fbx`, `fbx_readback`, `render`. |
| 11 lines in `src/app/cli.py` | Parser registration and dispatch only. |

**Default-path isolation:**

- The 35 pre-existing subcommands have an identical argparse structure. Their digest
  `ca4f2105…` was computed from the parser at `0d6435a` and is pinned by a test.
- Importing `app.cli` and building the parser loads none of: cv2, torch, bpy, numpy, matplotlib, W116, W121.
- `src/app/gui.py` has no reference to the slice.
- MotionGraph, AnimationClip, the exporter, the importer, keyframe collapse and every W1xx script are unchanged.

**Output tree per run** (outside the repository; nothing is ever overwritten):

| Path | Contents |
|---|---|
| `manifest/input_manifest.json` | read-only |
| `perception/m1s0/` | perception outputs |
| `rig/video_cases.json` | per-frame rig values |
| `baseline/baseline.blend` | plus sidecar, bindings, build/readback reports |
| `edits/<id>/` | request, proposal, `candidate.blend`, `timeline.png`, `commit.json`, `timeline_final.png` |
| `accepted/accepted_NN_<id>.blend` | accepted copy |
| `reverted/` | `.blend` and exact-undo report |
| `lineage/` | read-only content-addressed states, `refs.json`, `log.jsonl`, `blends.json` |
| `review/<id>/` | review video and stills |
| `fbx/<state>/` | FBX export and readback |

## 4. Provenance, timebase, binding, UNKNOWN and support accounting

| | S1 (Double Kettlebell Strict Press, side view) | S3 (Bent-over row) |
|---|---|---|
| Licence / author | CC BY-SA 4.0 / Taco fleur | CC BY 3.0 / FitnessScape |
| Clip SHA-256 | `de8feb9d515daa2b…` (= W121) | `f6b90f2f72e9020d…` (= W121) |
| Container | 30.000 fps, 1631 frames; cadence within 1 ms | 30.000 fps, 154 frames; cadence within 1 ms |
| Segment → Action | source 79–318 → Blender frames 1–240 | source 0–153 → Blender frames 1–154 |
| Live observation file | `7988bf4d…` | `2a5f3ae5…` |
| Missing-subject frames in segment | 0 | 0 |
| Input manifest | `2b191a2a…` | `784fcf45…` |

- **Perception parity.** The live observation files are byte-equal to the W121 stored arrays: all seven arrays and the
  file SHA match.
- **Time mapping.** Source time = index / 30. The Action frame is index − first + 1, recorded per frame in the sidecar.
  Playback factor is 1.0 at the 30 fps scene. No resampling or retiming.
- **Subject identity.** `video_subject_identity = UNKNOWN` on every state. No verified-identity claim is made.
- **Bindings** (the W116 rule against the independent `.blend` readback):
  - left_wrist, left_elbow, left_shoulder, right_wrist, right_shoulder and pelvis_root are BOUND;
  - left_hand, right_elbow and thorax are AMBIGUOUS;
  - nose is MISSING.
- **Driven channels:**
  - Armature Object rotation (location not keyed);
  - spine_01, thigh_l/r, calf_l/r, upperarm_l, lowerarm_l.

  The other 44 of the rig's 51 bones (finger and helper bones included) stay **at rest**, including the whole right arm, spine_02–05, neck, head, clavicles,
  hands and feet. There is no root translation and no foot contact.
- **Unavailable frames:** none in either segment. The slice would refuse with `UNSUPPORTED_FRAMES` rather than fill a frame.

## 5. Real clip → Blender Action evidence (G1, G2)

On both sources, from a **fresh Blender process that re-opened the saved `baseline.blend`**:

- **Content.**
  - One Action (`AnimCV_EXP baseline <id>`), assigned to the armature object `root` (slot `OBroot`).
  - 32 F-curves: object `rotation_quaternion` plus 7 pose-bone `rotation_quaternion` paths, all **CONSTANT**.
  - Keys on integer frames 1..N. Scene at 30 fps, `frame_start = 1`, `frame_end = N`.
  - Provenance JSON is embedded on the object and the Action.
- **Bitwise checks.**
  - The readback equals the in-process evaluation bitwise at every integer frame.
  - Subframes 0.25 / 0.5 / 0.75 equal the preceding integer frame (CONSTANT hold).
- **Python FK model vs Blender:** 1.31e-6 m (S1) and 1.26e-6 m (S3); the gate is 1e-5.
- **Parity with frozen history.**
  - The Action value digest equals the W121 digest: `25a09802…` (S1) and `b9058745…` (S3).
  - The state values are canon-equal to the W121 parent states, frame for frame.

**Baseline `.blend` SHA-256:**

| Source | Baseline `.blend` |
|---|---|
| S1 | `0803ab2bf31210fb…` |
| S3 | `e1cfbae49e3af0e7…` |

Both are in the evidence manifest.

## 6. User pin, resolver, hard-pin residual, footprint, timeline, undo (G3–G5)

**Primary requests.** `CAPTURED_FROM_EVALUATED_STATE`: hold the left wrist at its Blender-evaluated position at the
capture frame over a 9-frame interval. The windows are the W121 LOW windows, chosen before outcomes.

| | S1_E1 (210–218, capture 210) | S3_E1 (41–49, capture 41) |
|---|---|---|
| Gate decision → user decision | ACCEPT → ACCEPTED | ACCEPT → ACCEPTED |
| Wrist distance from target before edit (max) | 20.5 mm | 22.2 mm |
| Hard-pin residual after edit (Blender, integer frames / subframes) | 6.4e-7 m / 6.4e-7 m | 5.6e-7 m / 5.6e-7 m |
| Actual changed frames (Blender) | 211–218 (8) | 42–49 (8) |
| Actual changed bones | upperarm_l, lowerarm_l | upperarm_l, lowerarm_l |
| Changed subframe intervals | only 211–218 | only 42–49 |
| Root / spine / other bones / out-of-pin frames | bit-unchanged | bit-unchanged |
| Display truth check (PROPOSED and final ACCEPTED) | ok / ok | ok / ok |
| Edited state, accepted `.blend` SHA-256 | `986840be…`, `19a8caa5…` | `508d7269…`, `11ecc822…` |

The capture frame itself is unchanged: its residual is 0, so no change is needed there.

**Timelines** (`edits/<id>/timeline.png` and `timeline_final.png`) show, from the evaluator footprint only:

- the requested pin frames;
- the permitted domain (budget 0 = pin frames);
- the actual changed frames per bone, with wrist millimetres;
- the unchanged frames and the status.

The views are the whole Action plus a zoom. Rejected candidates are hatched and labelled "not applied".

**Lineage (S1, all on the live run):**

| Step | Result |
|---|---|
| User REJECT of a gate-passing candidate (E3) | head unchanged |
| ACCEPT of the gate-rejected UNREACHABLE candidate (N_REACH) | refused with `GATE_REJECTED_CANNOT_ACCEPT`; head unchanged |
| E1 ACCEPT | head moves |
| ACCEPT of E4 (made against v0) | `STALE_PARENT`; head unchanged |
| Exploratory E2 (capture + 3 cm up; gate ACCEPT, residual 6.2e-7 m) | committed after E1, so `STALE_PARENT`; never applied |
| Revert | new state `49dbe510…` |
| Post-revert proposal against the reverted-away E1 | `STALE_PARENT` |

**Revert result.** The reverted state's values equal v0's, its value digest equals v0's, and its Blender evaluation is
bit-equal at all integer frames and subframes (**exact undo**). S3 revert is likewise exact. The immutable baseline
files, manifest and sidecar were unchanged across every commit.

**Negative controls on the live S1 baseline (G4).** All 11 were REJECTed with the declared class and reason, and none
produced a candidate except REACH:

| Control | Class / reason |
|---|---|
| REACH, +1 m | INFEASIBLE / `UNREACHABLE` (candidate built, gate-rejected) |
| WRONGBIND, right_wrist binding id on a left_wrist pin | `BINDING_JOINT_MISMATCH` |
| Unknown joint `left_ankle` | `BINDING_UNDECLARED_JOINT` |
| `left_hand` | `BINDING_AMBIGUOUS` |
| `right_wrist`, bound but outside the slice | `JOINT_NOT_SUPPORTED_BY_SLICE` |
| NaN offset | `INVALID_NUMERIC_TARGET` |
| `ARMATURE_LOCAL` | `COORDINATE_FRAME_UNSUPPORTED` |
| `cm` | `UNIT_UNSUPPORTED` |
| spine_01 added to the authorised DOF | `UNAUTHORIZED_DOF_FOR_SLICE` (+ W116 `ROOT_NOT_PERMITTED`) |
| Stale parent (W121 state id) | `STALE_PARENT` |
| Frames 300–330 beyond the Action | `FRAME_NOT_IN_STATE` |

**Source and runtime controls (G9).** All STOP with the declared code:

| Control | Stop code |
|---|---|
| Boxing clip, container 30 000 fps | `UNSUPPORTED_FPS_MISMATCH` |
| Historical W102 50 fps clip | `UNSUPPORTED_CADENCE`: the 50 fps clip is **not** presented as a 30 fps success |
| Flipped byte | `BLOCKED_SOURCE_IDENTITY` |
| Declared 25 fps | `UNSUPPORTED_FPS_MISMATCH` |
| Missing checkpoint | `DATA_BLOCKED_MISSING_CHECKPOINT` |
| Absent Blender | `BLOCKED_BLENDER_UNAVAILABLE` |
| Missing observation | `DATA_BLOCKED_MISSING_OBSERVATION` |
| Stale manifest, on a copy of the S3 run | `BLOCKED_STALE_MANIFEST` |
| Tampered baseline `.blend`, on a copy | `BLOCKED_IMMUTABLE_OUTPUT_CHANGED` |

## 7. Native-rate A/B, quality side effects, review attribution (G6)

**Videos.** `review/<id>/ab_<id>_30fps.mp4` shows source crop + full frame | baseline `.blend` render | accepted `.blend`
render. Each render has a body view and a left-arm close-up, with the per-frame footprint text and a timeline bar.

| Source | Frames | Fps | Duration | SHA-256 |
|---|---|---|---|---|
| S1 | 240 | 30.0 | 8.000 s (= source) | `9b058d36…` |
| S3 | 154 | 30.0 | 5.133 s (= source) | `4a59a5f5…` |

**Pin compliance is separate from source fidelity and continuity.** Blender-evaluated disclosure from the frozen W118
`quality`:

| | S1_E1 | S3_E1 |
|---|---|---|
| Release/entry wrist compensation step (B1) | 20.5 mm | 22.2 mm |
| Raw wrist step excess over source at boundary | +12.3 mm | +20.6 mm |
| Introduced bend flips / near-straight (< 5°) frames | 0 / 0 | 0 / 0 |
| Min elbow flex (window) | 102.7° | **6.4°** (frame 49; source hypothesis 23.7°) |
| Peak arm-rotation step excess | 1.3° | 17.2° |
| Peak elbow deviation from source hypothesis | 11.7 mm | 47.0 mm |

The hold is a pure hard pin with a 0-frame transition budget. At the end of the pin run the wrist jumps back to the
source path in one frame. On S3 the hold drives the elbow close to straight. These are real side effects of the edit, not
improvements.

**Agent review** (`results/review/visual_review.json`, `021f1647…`; stills and timelines inspected, video timing verified
by decoding):

- **S1.** The edited wrist sits on the target marker on the pin frames. The change is visible only in the close-up.
  - **Negative, source fidelity:** the performer holds the kettlebells at the chest with both arms. The rig's left arm
    is raised out to the side and the right arm hangs at rest. The pin preserves this GT-free hypothesis; it does not
    make it more faithful.
- **S3.** The trunk bend and hanging left arm are broadly plausible.
  - **Negative:** the declared front close-up does not show the pinned wrist; the target is occluded. The S3 video
    therefore does not visually demonstrate the S3 edit.
- **Not done:** the agent cannot perceive native-rate playback as a human does, and no animator or user review was
  performed.

## 8. `.blend` vs FBX verdicts (G7)

| | Result |
|---|---|
| Blender `.blend` Action | **USABLE (experimental)**: editable F-curves, exact readback, source-time mapping, pin, undo |
| FBX export (existing `export_fbx`, unchanged) | file written: S1 head `38fc0eea…` (370 KB), S1 baseline `2171467d…`, S3 head `a9908159…`. The source `.blend` was unchanged by export. |
| FBX reimport (existing default importer) | succeeds: 1 Action (`root|root|AnimCV_EXP …`), key range 1..N preserved |
| Round-trip | **FAIL** on all three exported states |

**Round-trip mismatches:**

- **Integer-frame world drift.** Max 3.90 mm (S1, both states; median 0.55 mm; 62/240 frames > 1 mm). Max 6.77 mm (S3;
  median 0.96 mm; 75/154 frames > 1 mm).
- **Subframes.** Max 0.69 m (S1) and 0.21 m (S3).
- **Interpolation.** 32 CONSTANT curves become **519 LINEAR** curves (object location/rotation/scale paths are added).
- **Scene range.** `frame_end` imports as **250**.

These are the W107 (simplification drift), W108/W109 (LINEAR flags, importer discards CONSTANT) and W107 (`frame_end`
250) known failures, reproduced. Nothing was repaired. No exporter or importer flag, key insertion or global default was
changed. Multi-take leakage did **not** appear: each `.blend` holds exactly one Action.

## 9. Tests (RUN / PASS / FAIL / NOT RUN)

| Test set | Kind | Result |
|---|---|---|
| `tests/test_experimental_video_action.py` (35) | unit + mock synthetic rig + 1 real-Blender BaseRig test | **35 PASS** (`results/run/pytest_focused.xml`); also 35 PASS before the freeze |
| Regression: `test_cli`, `test_gui_app`, `test_blender_executor`, `test_motion_graph`, `test_keyframe_collapse`, `test_w116_semantic_pin`, `test_w117_temporal_pin`, `test_w118_transition_edit`, `test_w105_rig_utility`, `test_fk_solver`, `test_canonical_pose_parity`, `test_retarget_solver`, `test_w120_factor_attribution`, `test_w121_independent_transfer`, `test_w122_realtime_utility`, `test_render_blender_animation_video` | unit / existing | **226 PASS** (`pytest_regression.xml`). The pre-change baseline of 13 of these files gave 172 PASS. |
| Real-video E2E, S1 and S3 (live) | real | **RUN**: G1–G6 pass |
| Real Blender build/readback/revert/render | Blender | **RUN**: pass |
| FBX export + reimport | FBX | **RUN**: round-trip **FAIL** (characterisation) |
| Negative, source and tamper controls | real / Blender | **RUN**: 11 + 9, all as declared |

**Coverage of the focused tests:**

- Default parser digest, import isolation and GUI isolation.
- Frozen-code tamper detection.
- Intake: 30 fps accept and time mapping; changed byte, wrong declared fps, 25 fps cadence, missing file, bad segment and
  missing provenance.
- Missing checkpoint, absent Blender and missing observation.
- Unsupported frames are not fabricated; unknown Blender frame mapping.
- Synthetic rig:
  - a feasible nonzero pin;
  - a no-op pin (`TRIVIALLY_SATISFIED_NOT_EVIDENCE`);
  - 12 invalid requests;
  - hidden out-of-pin change, unauthorised bone change and display tamper.
- Store reject/accept/stale/exact undo with persistence.
- Real-Blender quaternion (xyzw → wxyz) and time mapping, CONSTANT hold, and the no-overwrite rule.

**Not run:**

- the full repository test suite;
- engines other than Workbench;
- other FBX consumers;
- independent re-execution by anyone else.

**Failed:** none.

## 10. Negative findings, safety, performance, maintenance, rollback

**Negative findings:**

- FBX round-trip FAIL (§8).
- The S1 left-arm hypothesis visibly disagrees with the source (§7).
- The pin creates one-frame release jumps of 20–22 mm and, on S3, a near-straight elbow (6.4°). It improves
  user-constraint compliance only.
- The S3 review close-up is uninformative.
- The partial rig makes the full-body result unusable as general animation.
- **Possible duplicated content in S3.** S3 shows 14/153 near-zero 2D-motion transitions and triplet-repeating flex
  values. Its effective content cadence may be lower than its valid 30 fps container. This was not verified, and the
  cadence rule cannot detect it.

**Safety:**

- The slice refuses rather than fabricates.
- Hard pins are never relaxed.
- No W123 reliability output is used.
- Gate decisions cannot be overridden by `accept`.
- Baseline, manifest and sidecar are hash-guarded.
- Lineage states are content-addressed and read-only.
- Licence fields are recorded, not judged: the historical clip passed the field check with "UNDOCUMENTED" and was stopped
  only by cadence.

**Performance** (wall clock, Main):

| Step | Time |
|---|---|
| S1 live build | 346 s (CPU M1 over all 1631 frames) |
| S3 live build | 40 s |
| Each propose | 2.3–2.6 s with Blender, 0.7–0.9 s for contract rejects |
| Commit | < 1 s |
| Revert | about 2 s |
| Review render | 10–16 s |
| FBX export + readback | about 2 s |
| Whole declared plan | 456 s |
| Storage | 251 MB of runs and 168 MB of results, outside the repository |

**Maintenance risks:**

- The slice depends on frozen research scripts under `scripts/`. They are guarded by hash, so any edit to them stops the
  slice until `frozen_code_sha256.json` is re-frozen deliberately.
- It requires `PYTHONPATH=src`, because this venv resolves `app` to another checkout (`C:\Projects\AnimCV`).
- The frozen W102 code snapshot and environment lock are external, user-supplied paths. A driver change alters the S0
  environment fingerprint and stops live perception.

**Rollback.** Remove the 11-line registration in `src/app/cli.py` (or the three new modules). No default behaviour,
schema or file depends on them.

## 11. Unresolved uncertainties

- Usefulness to an animator or user. No human used the workflow or watched the videos at native rate.
- Whether the slice works on any source other than these two engineering replays. A new 30 fps source would need a fully
  supported segment; the slice refuses partial ones.
- Whether S3's effective motion cadence is 30 fps.
- Whether the BaseRig mesh in the review renders occludes the edit in other views. Only one fixed front view was rendered.
- FBX behaviour in non-Blender consumers.

## 12. Evidence, reproduction and changed files

**Evidence.** All under `C:/Users/dna10/animcv-w124`; local, not published.

| Artefact | SHA-256 |
|---|---|
| `results/declare/declaration.json` | `cf74df45…` |
| `results/run/command_log.json` | `fc6ea8a1…` |
| `results/run/run_index.json` | `1be8a1bc…` |
| `results/run/pytest_focused.json` / `pytest_regression.json` | `fa2fbab6…` / `d6573d5d…` |
| `results/assess/assessment.json` | `ed0a1d34…` |
| `results/review/visual_review.json` | `021f1647…` |
| `results/evidence_manifest.json` (3,240 files in results + runs; written before this Worklog) | `507099dc9ef8d720b8f87fd722003e10d20b05db529507e7ecae57e384c1e067` |

Runs are in `runs/S1_live` and `runs/S3_live`.

**Reproduction** (Git Bash, repository root at `ed480ab`; every stage refuses an existing output):

```
PY=C:/Projects/AnimCV/.venv_build_full/Scripts/python.exe; R=C:/Users/dna10/animcv-w124/results
D="--declaration $R/declare/declaration.json --declaration-sha cf74df453d75715d2a922aaa65cce19fbab94b736c0e123f31b638176096dfb4"
$PY -X utf8 scripts/w124_vertical_slice_harness.py declare --out-dir $R/declare
$PY -X utf8 scripts/w124_vertical_slice_harness.py run    --out-dir $R/run $D        # needs setup/source_*.json + perception_assets_main.json
$PY -X utf8 scripts/w124_vertical_slice_harness.py pytest --out-dir $R/x --run-dir $R/run $D
$PY -X utf8 scripts/w124_vertical_slice_harness.py assess --out-dir $R/assess --run-dir $R/run $D
```

`setup/` holds:

- the four source records (S1, S3, the Boxing control, the historical-clip control);
- the Main perception-assets JSON, pointing at the frozen W101/W102 assets.

**Changed files (repository):**

- New:
  - `src/app/experimental_video_action.py`
  - `src/app/experimental_frozen_launch.py`
  - `src/blender/experimental_action_stage.py`
  - `tests/test_experimental_video_action.py`
  - `scripts/w124_vertical_slice_harness.py`
  - `examples/experimental_video_action/frozen_code_sha256.json`
  - `examples/experimental_video_action/perception_assets.template.json`
  - `examples/experimental_video_action/left_wrist_pin_request.example.json`
  - `docs/EXPERIMENTAL_VIDEO_ACTION_WORLD_PIN.md`
  - this Worklog
- Modified:
  - `src/app/cli.py`: +11 lines, opt-in registration only.
  - `docs/README.md`, `docs/00_마스터_플랜.md`, `docs/01_프레임워크_현황.md`: dated 2026-11-09 W124 status notes, labelled
    experimental and Agent-run. Existing text is not rewritten.
- Unchanged: `docs/next_batch.md`, every historical Worklog, every W1xx script, the models, the rig and the
  normative architecture documents.
