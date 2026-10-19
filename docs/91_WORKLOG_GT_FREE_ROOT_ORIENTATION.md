# Worklog — GT-Free Root Orientation and Runtime Heading Contract (2026-10-19)

Branch: `arch/single_frame_first`
Starting HEAD: `4392499`. Local was fast-forwarded from `44f586d` to include the upstream PRODUCT_VISION commit; the
worktree was clean and LabServer63 was at the same commit.

**Execution.** LabServer63 was occupied by another project for the whole session (one GPU process, 20–100% use; idle
monitor log kept). As the owner allowed, the batch ran without the server:
- frozen artifacts were copied once (bandwidth-limited);
- the heading evaluation, IK replay and Blender ran locally at low priority (`nice`).

**Unchanged:** no model was trained; no checkpoint, H0, J1X, W89 stabilizer, FK/IK solver, BaseRig or production
default was touched.

## 0. Answer

**Engineering CASE A — GT-free heading usable (opt-in, experimental), with a weak reliability component.**

**The fix is a re-sourcing, not a new estimator.** The existing AnimationSemantics v2 heading
(`current_root_orientation`, the docs/57 valid-only bilateral fusion) becomes GT-free once it is fed by the already
existing **S0 neutral-sign FramePose** model instead of the oracle-conditioned H0. The heading code is unchanged.

| 3DPW TEST, 7,076 rows, 37 sequences | heading error mean (95% CI) | p95 | >90° | front/back >135° |
|---|---|---|---|---|
| H0 oracle-conditioned (evaluation reference, not runtime) | 7.1° | 18.6° | 0.03% | — |
| **S0 runtime** | **10.0° (9.1–11.1)** | **28.1°** | **0.14%** | **0.07%** |
| Fixed 180° control (W90) | 67.5° | — | 31.9% | — |

**What S0 runtime delivers:**
- Sudden flips on 0.13% of pairs.
- 94% of real 1-s turns tracked.
- No lag.
- No hold, so no stale-heading mechanism.

**Effect on the rig:**
- The torso turns correctly.
- The J1X left arm's torso-relative (local) upper-arm rotation error against the GT reference falls from 28.0° to
  15.1° on TEST. The heading's own share falls from 24.5° to 4.6°.

**Weaknesses:**
- **Reliability.** The added UNRELIABLE flag separates errors only modestly.
- **Failure regimes:**
  - occluded or seated torsos with low detector confidence;
  - some oblique views with a sustained ~50° bias.

## 1. Oracle provenance (first gate)

**Where the oracle signs come from.**
- `shoulder_forward_depth` and `hip_forward_depth` are computed from **3DPW GT 3D joints**:
  - `framepose/signs.py:209-214` takes the right-minus-left forward-depth sign of `target_3d`;
  - `oracle_sign_states` is at `:258-267`.
- They are not stored in the bank; they are recomputed from `target_3d`.

**How they reach H0.**
- They are **network inputs at inference** for O_BILATERAL (`model.py:167-199`: sign embeddings are added to the shoulder
  and hip queries before global attention).
- Frozen H0 = `context_h0_v2`, materialized by `scripts/prepare_context_h0.py` with
  `--sign-source oracle --sign-fields shoulder_forward_depth,hip_forward_depth` on **train, validation and test**
  (`materialization_report.json`: `"sign_source": "oracle"`).

**How they reach heading.**
- `current_root_orientation` computes yaw = atan2(dy, dx) of H0's 3D shoulder and hip axes.
- The sign of `dy` selects the front/back half-plane, and that sign is exactly what the oracle fields inject.
- H0 yaw is therefore a **derived quantity that inherits GT**.

**Classification by category**

| category | what |
|---|---|
| (1) GT as training supervision (legitimate) | `target_3d` loss of H0, S0, J1X and W86 |
| (2) GT for evaluation (legitimate) | matched-oracle `observe_yaw`; `root_yaw_error_degrees`; W62 GT chains |
| (3) GT supplied at inference (illegitimate) | the O_BILATERAL sign inputs on every split |
| (4) Derived quantities inheriting GT (illegitimate at runtime) | **every H0 joint** (signs spread through attention), `current_root_orientation` on H0, H0-based FK/IK local rotations (W59–W61), H0 reach/direction/side (W74–W79), context refiner (W47–49), reconciliation (W41–46), W52 contact replay |
| (5) Genuine runtime observations | detector `input_2d` / `geometry_tensor`, validity masks, Qwen token readouts, J1X/W86 outputs, **S0 outputs** |

**Note on docs/57.** Its sentence "3DPW oracle torso geometry is evaluation only" describes only the yaw *reference*;
H0's inputs were oracle-conditioned throughout.

### Historical-result impact (actual scope)

**Not affected:**
- J1X (W80) and the W86 connected limbs: inputs are detector geometry and Qwen only (`joint_arm_geometry.py:1-5`,
  `connected_limb_transfer.py:8-10`).
- W87–W89.
- W90: constant yaw.
- Rotation-invariant chain metrics in W62/W76/W80, because the same yaw is applied to prediction and GT.

**Affected:**
- Any absolute heading-accuracy claim built on H0: docs/52–57 and docs/60 Root Orientation.
- H0-articulation rig replays (W59–W61).
- Comparisons that used H0 as a baseline: W74–W79 and W75 T0, where docs/75 already noted the tilt.

These are **runtime-legitimacy** problems of those baselines, not invalidations of their model-internal findings. The
docs/29–30 sign attribution already measured the GT-free S0 with this same graph and seed: MPJPE 81.98 against 79.23
for O_BILATERAL.

**Separate production-critical blocker (recorded, not changed).** Production AnimationSemantics, FK and IK replays are
wired to `context_h0_v2`, so the whole H0 articulation (not only heading) is oracle-conditioned. A GT-free whole-body
path must take its joints from S0 (or another GT-free model).

## 2. Existing GT-free capabilities audited

| candidate | verdict |
|---|---|
| **S0 neutral-sign FramePose** (`framepose/sign_v1/S0`) | Same bank digest `75519e63…`, same graph and seed as O_BILATERAL, every sign field UNKNOWN in training and inference, checkpoint chosen on validation MPJPE (test GT unused). Predictions exist for validation (`402d70a2…`) and test (`4357c7bd…`). **GT-free.** |
| O_BILATERAL checkpoint run with neutral signs | possible, never done; off-distribution; not used |
| VLM Sign Advisor (docs 28–29) | below the majority-class baseline per field; S2 yaw P95 126°; rejected historically |
| 2D-only bilateral heading | none exists; every yaw estimator uses lifted 3D x–y |
| `current_root_orientation` (docs/57) | input-agnostic, stateless, UNKNOWN when no valid pair; **reusable unchanged** |

**Baseline B:** S0 → unchanged `build_h0_lifted_sequence` → unchanged `current_root_orientation`. This is the simplest
valid existing GT-free policy.

## 3. Implemented candidate C (bounded)

**Why C was added.** B's large errors (> 45°: 0.85% of VAL rows, 1.38% of TEST rows) cluster in two runtime-visible
situations, found from VALIDATION and from inspecting the failing rows:
- a barely detected torso: minimum 2D confidence of the shoulders and hips near 0–0.4 (e.g. a seated subject behind a
  café table);
- strongly disagreeing shoulder and hip axes.

**What C is.** `src/motion/runtime_heading.py`, provider id `framepose_s0_valid_bilateral_reliability_v1`,
experimental, imported by no production module.

| item | definition |
|---|---|
| Inputs | the S0 lifted sequence only (detector validity and confidence via the unchanged bridge); no GT, no sequence identity, no temporal state |
| Yaw | **identical to B** |
| `UNKNOWN` | no valid bilateral pair (docs/57) |
| `UNRELIABLE` | minimum torso 2D confidence < 0.3, or shoulder–hip axis disagreement > 45° |
| `KNOWN` | otherwise |
| Conventions | AnimCV camera axes; yaw of the right-minus-left axis; facing the camera = 180° |
| Front/back | resolved by S0's learned lifted depth of the bilateral joints. The projected shoulder axis alone is not assumed to resolve 360°: the 2D left/right order fixes facing toward vs away, and S0's depth resolves the side-on half |
| Temporal behaviour | per-frame and causal; no hold, so no stale runs and no false hold on turns |
| Fallback | the consumer decides. The rig replay keys every B yaw and flags non-KNOWN frames |

**Thresholds.** Fixed from their physical meaning (barely seen torso; disagreement beyond ordinary torso twist) and
checked only on VALIDATION before the TEST replay. VALIDATION has no > 90° failure, so it cannot calibrate a severe-failure
gate. No sweep was run.

**Anticipated failures:**
- occluded torso with plausible but wrong S0 depth (not caught when confidence stays above 0.3);
- sustained oblique bias.

## 4. Heading evaluation

**Method.** `scripts/run_runtime_heading_eval.py` uses the same frames, detector observations and conventions for every
provider.
- **GT heading** (offline only) = `observe_yaw` of the matched 3DPW GT joints, as in docs/55–57.
- **Parity:** H0 reproduces the docs/57 VALIDATION mean error, 6.096° (docs: about 6.10°).
- **Bootstrap:** sequence-cluster intervals.
- **UNKNOWN is never scored as correct;** coverage is reported separately.

| | VAL coverage | VAL mean / p95 | VAL >45° / >90° | TEST coverage | TEST mean (CI) / p95 | TEST >45° / >90° / >135° |
|---|---|---|---|---|---|---|
| A H0 oracle (reference) | 100% | 6.10 / 16.2 | 0.21% / 0 | 100% | 7.09 (6.59–7.69) / 18.6 | 0.04% / 0.03% / 0.01% |
| **B S0 runtime** | 100% | **7.59** / 21.6 | 0.85% / 0 | 100% | **9.99 (9.08–11.10)** / 28.1 | 1.38% / 0.14% / 0.07% |
| C S0 + reliability | 99.27% | 7.43 / 21.1 | 0.65% / 0 | 98.66% | 9.84 / 27.7 | 1.23% / 0.10% / 0.03% |
| F fixed 180° | 100% | 65.7 / 158.8 | 54% / 32% | 100% | 67.5 / 164.0 | 56% / 32% / 17% |

**By GT view, TEST, B** (mean / p95 / >90°):

| view | mean | p95 | >90° |
|---|---|---|---|
| front | 10.2 | 26.0 | 0.08% |
| oblique | 11.0 | 36.0 | 0.05% |
| side | 7.6 | 22.7 | 0.4% |
| back | 11.8 | 30.4 | 0 |

Fixed 180° is right only for front views (13.9°); for back views it is 163.8°.

**Temporal behaviour, TEST** (B / A / F):

| metric | B | A | F |
|---|---|---|---|
| sudden flips (>90° step while GT changes <45°) | 9 of 7,039 pairs (0.13%) | 2 | 0 |
| true 1-s turns (GT ≥45°) tracked at ≥50% amplitude and same direction | 94.2% | 98.4% | 0 |
| best lag | 0 rows in all 37 sequences | — | — |
| incorrect runs ≥1 s (error > 45°) | 3, longest 4.0 s, 0.6% of rows | — | 77, longest 54 s |

On VALIDATION, B has 2 flips, tracks 96.3% of turns and has no incorrect run ≥ 1 s.

**Reliability:**
- B errors by torso-confidence quartile, TEST: 11.2 / 8.9 / 9.4 / 10.5°. The lowest quartile holds most of the > 90°
  rows (0.45%).
- C UNRELIABLE flags 25 VAL rows and 95 TEST rows. It removes 3 of 10 TEST > 90° rows, 3 of 5 front/back rows and 12 of 98 > 45° rows for a
  1.3% coverage cost.
- UNKNOWN runs are short: longest 8 rows on TEST, one run ≥ 6 rows.
- **The flag is useful but weak.** It does not catch the sustained oblique bias (`office_phoneCall_00` actor 1, about
  50°, KNOWN).

**Failure concentration** (TEST, B mean error by sequence):

| sequence | mean | note |
|---|---|---|
| office_phoneCall_00 a1 | 19.3° | sustained bias |
| downtown_car_00 | 17.2° | |
| downtown_cafe_00 a1 | 15.2° | 2.2% > 90°: the seated, occluded torso |
| downtown_walking_00 a1 | 15.1° | |

## 5. Rig-level real-scene validation

**Method.** `scripts/run_runtime_heading_rig_replay.py`:
- **Identical inputs in every mode:** frozen J1X left arm (W87 `cb9dc51e…`) through the unchanged W61/W76 IK, plus the
  W60 object-owned root rotation rest·A·Rz(ψ)·A⁻¹.
- **Modes:**
  - FIXED ψ = 180°;
  - RUNTIME ψ = B;
  - GT ψ, evaluation-only.
- **Reference arm:** the GT chain with GT heading.
- **Scope:** TEST 6,003 and VAL 3,056 available frames.

| metric | TEST FIXED / RUNTIME | VAL FIXED / RUNTIME |
|---|---|---|
| torso heading error | 60.9 / **9.9** | 64.0 / **7.0** |
| upper-arm **local** rotation vs GT-reference arm | 28.0 / **15.1** (Δ −12.9 [−16.3, −9.8]; RUNTIME worse in 1 of 37 sequences) | 31.1 / **11.3** (Δ −19.7 [−26.0, −11.7]; 0 of 16) |
| upper-arm local rotation p95 | 79.1 / 37.9 | 95.9 / 29.6 |
| forearm local rotation vs GT reference | 46.1 / 22.2 | 42.8 / 17.7 |
| heading's own share (same J1X chain under GT heading) | 24.5 / **4.6** | 29.6 / 3.7 |
| world-frame arm direction across modes | identical (max diff 3.7e-13) | identical |
| IK endpoint / IK→FK residual | ≤ 2.4e-13 | ≤ 2.1e-13 |
| RUNTIME armature-rotation step | mean 7.9°, p95 24.7°, max 167°; 9 sudden spins in 7,039 steps | mean 6.2°, p95 20.3°; 2 in 3,391 |

**Reading the table:**
- As W90 found, the heading cancels in the world arm direction, so wrist position cannot judge heading.
- Torso orientation and local arm rotation can. There RUNTIME removes most of the fixed-heading error, and the remainder
  (15.1° vs 4.6° from heading) belongs to J1X's own arm geometry.
- **Actor isolation:** the provider is stateless per frame and per track (tested). `downtown_arguing_00` actor 1 is
  replayed separately from actor 0.

## 6. Blender review package

`scripts/render_runtime_heading_rig_blender.py` and `scripts/compose_runtime_heading_rig_video.py` produce 11 MP4 clips
plus a mid-clip PNG each at `~/animcv-output/91_framepose_runtime_heading/`.

**What the renders are**
- Actual BaseRig Actions: the Armature Object rotation and two arm bones keyed at the exact sampled frames, with CONSTANT
  interpolation.
- Rendered as **bone proxies: BaseRig has no skinned mesh**. The separate 90A mannequin was not used.
- Panels: RGB with the runtime reliability state, then FIXED / RUNTIME / GT in the video-camera and side views.
- Body-forward arrows: mode colour for the mode, thin green for GT.
- A heading-trajectory plot and an upper-arm local-error plot.
- Blender FK vs the exported IK endpoint ≤ 4.3e-5 cm.

**Selection rules** (in `CASE_RULES`):
- turning
- walking away
- side-on
- stationary facing
- worst runtime
- two actors

The worst-runtime case was selected by rule, not to flatter the result.

| clip | observation |
|---|---|
| test_turning (cafe_00 a1) | Runtime follows a fast 180° turn with GT; FIXED stays front-facing. The arm-error spike at t = 3 s also appears under GT heading, so it is J1X arm geometry |
| test_walking_away (walkBridge_01) | FIXED renders a front-facing torso for a subject walking away, which puts the arm on the wrong side of the torso. Runtime is within a few degrees of GT |
| test_worst_runtime (office_phoneCall_00 a1) | A sustained ~50° runtime bias while the subject holds a phone, flagged KNOWN. Still far better than FIXED (140°) |
| test_two_actors, side_on, stationary_facing; VAL cases | Runtime ≈ GT. No coordinate-frame inconsistency, IK failure or actor contamination visible |

## 7. Tests

`tests/test_framepose_runtime_heading.py`, 8 tests, all passing locally. They cover:
- circular wraparound
- the facing convention and view strata
- UNKNOWN never scored as correct
- flip and turn accounting
- runtime providers never reading GT
- KNOWN / UNRELIABLE / UNKNOWN states with unchanged yaw
- statelessness and immediate turns (no false hold)
- root-rotation / limb-transform consistency and no production imports

IK parity and deterministic replay rely on the unchanged W90 functions and were re-checked by the replay's own
residuals.

The server suite was not run because LabServer63 stayed occupied. No shared production contract changed.

## 8. CASE

**CASE A, opt-in experimental runtime provider.** Against the acceptance criteria:

| criterion | status |
|---|---|
| no privileged input | met |
| usable heading for all four view strata (mean 7.6–11.8°) | met |
| real turns handled (94%, no lag, no hold) | met |
| large flips rare (0.13% of pairs; 0.07% front/back) | met |
| works through the existing rig conventions | met |
| large practical benefit over fixed heading (67.5° → 10.0°; local arm rotation 28° → 15°) | met |
| meaningful reliability behaviour | **only weakly met** |

**About the reliability criterion:**
- The C flag improves the tails a little.
- It misses sustained biases.
- B emits UNKNOWN only when no valid pair exists, which never happened on 3DPW.

**Accounting for the "new candidate":** the main delivered value is that the existing GT-free S0 plus the existing
heading code already satisfy the contract. C adds only the reliability annotation.

**No production default changed.**

## 9. Remaining integration blockers

1. **The production FramePose source is still oracle-conditioned H0.** It has to switch to S0 (or another GT-free model)
   for **all** joints, not only heading: S0 MPJPE 81.98 vs 79.23 mm.
2. **The reliability semantics are weak.** There is no detector for sustained occlusion or bias. Animation export
   currently holds the previous key on non-KNOWN frames.
3. **Real-video inputs untested.** Only 3DPW's official OpenPose 2D detections and FrameBank crops were tested. A real
   video needs the production detector adapter, actor tracking and timestamps.
4. **Partial body.** Root translation and ground contact remain UNAVAILABLE. Only the left arm (J1X) is wired to IK; the
   torso articulation beyond the object rotation stays at rest in these replays.
5. **Cadence.** Bank cadence is 6–10 Hz; native-cadence behaviour is untested (also W90 blocker 5).

## 10. Recommended next milestone

**A first fully GT-free whole-body experimental replay:**
- S0 for torso and legs;
- J1X / W86 for the arms;
- this runtime heading for the root;
- through the existing FK/IK into BaseRig;
- on 1–2 ordinary (non-benchmark) user videos with the production detector.

It should report which body parts are reconstructed, held or UNAVAILABLE. This is not started automatically.

## Artifacts and files

**Local archive** (`~/animcv-local-data/w91/archive/`; the server was occupied):
- `heading_eval_report.json` (SHA-256 `6362c2a9…`)
- `rig_replay_report.json` (`72c1c3ef…`)
- `rig_replay_cases.json`
- `blender_report.json`
- 11 `.blend` files

**New files:**

| file | role |
|---|---|
| `src/motion/runtime_heading.py` | the experimental provider |
| `scripts/run_runtime_heading_eval.py` | heading evaluation |
| `scripts/run_runtime_heading_rig_replay.py` | rig-level replay |
| `scripts/render_runtime_heading_rig_blender.py` | Blender render |
| `scripts/compose_runtime_heading_rig_video.py` | review clip composition |
| `tests/test_framepose_runtime_heading.py` | the 8 focused tests |

No existing file was modified.
