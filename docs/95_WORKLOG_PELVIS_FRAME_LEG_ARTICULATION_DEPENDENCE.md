# Worklog — Pelvis Orientation Dependence and Articulated Lower-Body Rig (2026-10-24)

## Starting state and environment

- **Branch and HEAD:** `arch/single_frame_first`, starting HEAD `ebdcd5e` (Worklog 94). Local, origin and LabServer63
  were in sync and the worktree was clean. I read PRODUCT_VISION.md and Worklogs 91–94.
- **Where it ran:**
  - Everything ran locally at low priority: the replay (≈ 70 s) and Blender 5.1.2.
  - LabServer63 was busy with other projects (load ≈ 33–35). It was used only to copy 36 RGB frames and for one short
    docker test run.

**What was not changed.** No checkpoint, S0, J1X, W91–W94 module, production IK/FK, BaseRig hierarchy, rest pose,
`RootOrientation` / AnimationSemantics schema or production default was changed. Nothing was trained or tuned.

**No new mechanisms.** No root translation, contact, foot lock, gravity estimator or temporal smoothing was added.
Gravity is used only for evaluation strata.

## 0. Answer

**CASE C: leg articulation does not isolate the pelvis question; pelvis necessity remains unconfirmed.**

**The factorization is exact.** With both legs articulated through the existing two-bone IK, every root policy
reproduces the same world-space thigh and shin directions (differences ≤ 4e-4, mostly ≤ 1e-14). Upper-body output is
bit-identical (≤ 5e-14).

**Upright and standing waist bends: no pelvis necessity shows.**
- Two GT-free-compatible frames meet the predeclared acceptance rule with GT leg geometry: the hip-axis yaw (U) on both
  3DPW splits, and the W93 trunk axis (A) on TEST.
- On joint range-of-motion (ROM) bounds alone, the hip/spine ROM violation rate is ≤ 3% for B, U and A.
- What changes is **how the bend is split between hip and spine**:
  - A puts the whole bend at the hips (hip flexion 58°, straight `spine_01`).
  - U puts it in the spine (49°, pelvis upright).
  - The SMPL pelvis (B) splits it 26° / 33°.
- That split is exactly the pelvis tilt W94 showed to be unobservable. It changes the visual style (a stiff-back hinge
  vs. a curved back), not ROM plausibility.

**Lying and seated: the frame choice matters.**
- U fails lying. 3DHP lying rows: 84% ROM violations (spine hyperextension, hip ±90°+), and visibly folded bodies.
- A trunk-following frame (A, 23%; L, 3%) does not fail lying.
- Choosing between them would need posture/pelvis/gravity evidence (W94).
- On seated/crouched rows both A and U exceed B by about 25 percentage points under the predeclared rule.

**The predeclared rule is dominated by something other than the pelvis.**
- Its dominant term is the **knee-hinge deviation**, which comes from the existing zero-twist IK convention: thigh
  roll is not modelled. It fires on 24–62% of GT-geometry rows **even under the true SMPL pelvis**. On 3DHP every policy
  fails the absolute rule in every stratum, upright included.
- With the frozen S0 legs as the GT-free source, sensor error dominates:
  - thigh direction error is 9–10° (3DPW) and 13–32° (3DHP);
  - knee-hinge violations rise to 45–84% for every policy, including the SMPL-pelvis control;
  - prone/lying S0 legs bend backwards.

## 1. Frozen inputs and baseline reproduction

**Frozen inputs** (hash-pinned; the script STOPs on change):
- J1X `cb9dc51e…` / `356f8cee…`.
- S0 `402d70a2…` (VAL), `4357c7bd…` (TEST) and `a78adeb0…` (3DHP).

**References** (EVALUATION ONLY):
- W94 SMPL pelvis from 3DPW `poses[:3]` + `cam_poses`.
- W94 gravity strata (3DPW world +Y; 3DHP calibration).

**Baseline reproduction before any new contrast:**
- A rerun of the W94 script is byte-identical (`report.json`, `cases_3dpw.json`, `cases_3dhp.json`).
- A W94 clip re-rendered with the extended renderer is pixel-identical (216 frames).

The W95 script imports the W94 row builder and frames unchanged, so W94's rows, frames, strata and windows are reused
as is.

## 2. Rig and IK contract audit

**BaseRig leg hierarchy:**
- `thigh_l` / `thigh_r` are top-level (parent = Armature Object).
- `calf_*` is the child of `thigh_*`, connected.
- `foot_*` is the child of `calf_*`, but sits 0.04–0.06 rig units off the calf tail.
- There is no pelvis bone. The Object is the only transform above both thighs and `spine_01`.

**Existing IK is chain-generic.**
- `TwoBoneIkChain` + `target_chain_geometry` accept `thigh_* → calf_*` without a production change.
- The end anchor is left out because the foot is not connected within the contract's 1e-6 tolerance.
- Leg lengths are 43.34 / 42.22 rig units. The rest bend normal ≈ the lateral armature axis, which is the knee hinge.

**The leg is solved with the W61/W76 path unchanged:**
- W93 `measure_chain_full` with R = R_o (the thigh's parent frame is the Object);
- `full_arm`: observed bend side, rest-plane fallback, closed form;
- zero-twist shortest-arc locals.

The experimental chains live only in `scripts/run_pelvis_frame_leg_articulation.py`;
`examples/e2e_demo/framepose_ik_chains.json` is untouched.

**What is controllable vs missing:**

| Item | Status |
|---|---|
| Hip swing (thigh direction) | controllable: thigh local rotation |
| Knee | controllable: calf local rotation |
| Thigh twist / hip internal-external rotation | **not modelled**: zero-twist convention, so the knee hinge orientation follows R_o |
| Foot | stays at rest |
| Joint limits, twist limits | **none exist** in BaseRig or AnimCV IK |

The bounds used below are evaluation-only, predeclared and taken from commonly cited normal adult ROM.

**Propagation.**
- A change of R_o rotates both thighs and `spine_01`.
- The IK recomputes the thigh locals as R_oᵀ (world), so world leg geometry is invariant.
- The hip local rotation, the thigh roll and the `spine_01` rotation absorb the change.
- The rest pose and pelvis mesh follow R_o. The mesh orientation is therefore *not* separable from the reference-frame
  choice in this hierarchy.

## 3. Factorization and its limits

**Invariance.**
- The world thigh is R_o · (R_oᵀ t) = t, for any R_o, up to the rig's thigh/shin ratio. The same holds for the shin and
  the ankle direction.
- The thorax and arm are invariant as in W94.

**Not invariant:**
- the pelvis (Object) orientation, and so the skinned pelvis region;
- the hip local rotation (flexion/abduction relative to the Object);
- the `spine_01` rotation R_oᵀR_t;
- the thigh roll under zero twist, and with it the knee hinge relative to the knee plane;
- temporal continuity of those locals.

**Predeclared evaluation** (`PREDECLARED` in the script, fixed before any real-data result was read):

| Item | Value |
|---|---|
| Hip | flexion ≤ 120°, extension ≤ 30°, abduction ≤ 45°, adduction ≤ 30° (thigh direction in the Object frame) |
| Knee | flexion ≤ 150° |
| Spine (R_oᵀR_t) | flexion ≤ 90°, extension ≤ 30°, lateral ≤ 40°, axial ≤ 45° |
| Knee hinge | deviation ≤ 15° (angle between the posed rig hinge and the posed knee-plane normal), evaluated when knee flexion ≥ 20° |
| Severe hip jump | thigh local step > 90° while the GT thigh steps < 45° |
| Acceptance (3DPW) | any-violation rate ≤ B + 5 pp per posture stratum, and no more severe jumps than B in that stratum |
| Acceptance (3DHP) | any-violation rate ≤ 10% |

**Implementation correction.** The first implementation compared whole-population jump counts. It was corrected to
the declared per-stratum rule; the rule itself did not change.

**Post-hoc addition.** The **ROM-only rate** (knee hinge excluded) was added after reading the predeclared result and is
labelled as post hoc throughout.

## 4. Synthetic fixtures

Construction: hips and spine joint on the true pelvis frame, shoulders on the true thorax frame, explicit
thigh/shin. B = true pelvis.

| Fixture | B | U hip yaw | A W93 trunk axis | L lower trunk |
|---|---|---|---|---|
| upright neutral | ok | ok | ok (2.5° lumbar offset) | ok (11° offset → hip −11°) |
| stable pelvis, waist bend 60° | hip 0, spine 46 | hip 0, spine 46 | **hip 46, spine 0** (hip hinge) | hip −11, spine 57 |
| bend with flexed hips (40 + 30) | hip 40, spine 21 | hip 0, spine 61 | hip 61, spine 0 | hip 29, spine 32 |
| lying supine | ok | **hip 90, spine −92: spine_extension** | ok | ok |
| lying prone | ok | **hip −90: hip_extension** | ok | ok |
| crouch (knee 99°) | hip 104, hinge 3.7° | hip 74, hinge 5.4° | hip 113, hinge 2.9° | hip 93, hinge 4.4° |
| body turn 90° | ok | ok | ok | ok |
| transition 0→90° (whole body tilts back) | ok | spine_extension from 30° | ok | ok |

**Endpoint parity.** In every fixture the world ankle direction is identical across policies (≤ 3e-16).

**Different hip rotations required.** The waist-bend and flexed-hip fixtures show alternative frames demanding
different local hip rotations for the same skeleton. All of them are within ROM.

Every fixture is representable without a hierarchy change.

## 5. Coverage

| Population | Eligible (W94 rows) | GT-geometry phase evaluated | S0-leg phase evaluated | Excluded |
|---|---|---|---|---|
| 3DPW VALIDATION | 3407 | 3407 | 3109 | 298 S0 2D leg joints invalid |
| 3DPW TEST | 7076 | 7076 | 6339 | 737 S0 2D leg joints invalid |
| MPI-INF-3DHP | 2728 | 2532 | 2532 | 196 GT leg joints invalid |

**Gravity strata (3DPW TEST):** upright 5908, waist_bend 241, lying 3, other (seated/crouched/kneeling) 924.

**Gravity strata (3DHP, evaluated rows):** upright 1458, waist_bend 241, lying 88, other 745.

3DPW has an SMPL pelvis reference; 3DHP does not, so there are no pelvis-accuracy claims for 3DHP.

## 6. Controlled GT-geometry comparison

Setup: GT thorax, GT legs, J1X arm.

**3DPW TEST** (any-violation rate predeclared; ROM-only post hoc; contrast vs B with sequence-cluster 95% interval):

| Stratum | B any / ROM | U any / ROM | A any / ROM | L any / ROM | Acceptable (predeclared) |
|---|---|---|---|---|---|
| upright (5908) | 0.241 / 0.004 | 0.257 / 0.008 | 0.250 / 0.001 | 0.306 / 0.121 | U, A |
| waist_bend (241) | 0.622 / 0.000 | 0.631 / 0.008 | 0.651 / 0.000 | 0.614 / 0.000 | U, A, L |
| other (924) | 0.545 / 0.016 | 0.791 / 0.096 | 0.799 / 0.035 | 0.408 / 0.062 | L |
| lying (3) | — | — | hip flexion > 120 in 3/3 | — | not interpretable |

**3DPW VALIDATION:**
- upright: U and A acceptable;
- waist_bend: U and L acceptable; A is not (knee-hinge excess +13.6 pp; ROM-only +0.0);
- other: L acceptable.

**3DPW TEST waist bends, how the bend is split:**

| | hip flexion | `spine_01` flexion | pelvis vs SMPL |
|---|---|---|---|
| B | 25.7° | 32.8° | 0° |
| U | 10.1° | 48.9° | 19.4° |
| A | 58.5° | 0° | 33.2° |
| L | 8.6° | 49.9° | 17.2° (12° constant offset) |

**3DPW knee-hinge deviation** (mean / p95):

| | upright | other |
|---|---|---|
| B | 11.9 / 27.7 | 15.2 / 37.2 |
| U | 12.2 / 28.9 | 21.6 / 42.1 |
| A | 12.5 / 29.4 | 25.8 / 53.2 |
| L | 10.7 / 26.3 | 11.8 / 30.8 |

On seated/crouched rows the pelvis frame moves the knee hinge through the zero-twist thigh roll. That is the main
source of the +25 pp excess of U and A.

**L's hip-extension violations** (12% upright) come from its constant 12° lower-trunk offset (W94).

**MPI-INF-3DHP** (no B; any / ROM-only):

| Stratum | U | A | L |
|---|---|---|---|
| upright (1458) | 0.422 / 0.002 | 0.416 / 0.003 | 0.420 / 0.001 |
| waist_bend (241) | 0.631 / **0.232** (spine flexion > 90 in 20%) | 0.631 / 0.071 | 0.560 / 0.021 |
| lying (88) | 0.909 / **0.841** (spine extension 70%, hip 91%) | 0.523 / 0.227 (hip flexion > 120 in 22%) | 0.455 / 0.034 |
| other (745) | 0.762 / 0.279 | 0.668 / 0.142 | 0.603 / 0.017 |

On the predeclared absolute rule, no policy is acceptable in any 3DHP stratum. Even upright, the knee hinge alone is
42%.

**Temporal (severe hip jumps):**
- 3DPW: 0 everywhere except A on VAL (4, other).
- 3DHP: U 25 (lying; the thighs flip around the upright pelvis), A 6, L 0.

**Integrity:**
- World leg direction across policies ≤ 4e-4.
- Leg IK residual ≤ 1e-13.
- Upper body identical (≤ 5e-14).

## 7. GT-free source (S0 legs)

Setup: frozen S0 thorax (W93 fallback) and S0 legs. GT is used only for scoring. B_S0 is the SMPL-pelvis control on the
same S0 geometry.

| Population / stratum | B_S0 any | A_S0 | U_S0 | L_S0 | Acceptable (predeclared) |
|---|---|---|---|---|---|
| 3DPW TEST upright (5228) | 0.446 | 0.471 | 0.465 | 0.519 | U_S0 |
| 3DPW TEST waist_bend (227) | 0.705 | 0.692 | 0.634 | 0.630 | U_S0, L_S0 |
| 3DPW TEST other (881) | 0.840 | 0.899 | 0.849 | 0.776 | none |
| 3DPW VAL upright / waist / other | 0.307 / 0.693 / 0.649 | 0.317 / 0.760 / 0.740 | 0.315 / 0.754 / 0.643 | 0.363 / 0.749 / 0.633 | upright: A_S0, U_S0; other: U_S0, L_S0 |
| 3DHP upright / waist / lying / other | — | 0.52 / 0.83 / 0.92 / 0.88 | 0.51 / 0.67 / 0.99 / 0.86 | 0.51 / 0.65 / 0.93 / 0.81 | none |

**Sensor-driven vs factorization-driven failures:**
- **Sensor-driven** (identical across policies, including B_S0):
  - thigh/shin direction errors: 3DPW TEST 10.4 / 11.1°; 3DHP 18.3 / 16.1°, lying 31.8 / 33.0°;
  - the knee-hinge rise from 24% to 45% upright (3DPW TEST, B vs B_S0);
  - backward-bent S0 legs in prone 3DHP frames (hinge deviation > 120°).
- **Factorization-driven** (differences between policies on the same S0 geometry):
  - U_S0's lying failures (3DHP ROM 83%);
  - A_S0's hip-flexion / abduction violations in waist bends and other;
  - severe hip jumps (3DPW TEST: A_S0 9, U_S0 9, L_S0 2, B_S0 0; 3DHP: A_S0 29, U_S0 36, L_S0 27). These come from S0
    frame flips and fallback switching (W93/W94) carried into the hip locals.

W86 leg outputs were not used, so there is no mixed-input control.

## 8. Skinned Blender comparison

**Setup.**
- Skinned mannequin `examples/skinned_mannequin/BaseRig_Skinned.fbx` (equivalence 6.1e-5).
- Animated: Object, `spine_01`, both thigh/calf chains and the J1X arm.
- Identical camera, timestamps and CONSTANT interpolation.
- Thin green = GT legs and GT arm (evaluation only).

**FK checks.**
- Leg FK vs export ≤ 7.7e-5.
- Arm ≤ 1.02e-4.
- A re-render is pixel-identical (216 frames).

**Clips:** `~/animcv-output/95_framepose_pelvis_frame_leg_articulation/` (11 mp4 + 11 png).

**Window means:**

| Clip (slots) | Hip flexion L (mean) | `spine_01` flexion | What is visible |
|---|---|---|---|
| 3dpw_bent_or_seated (B / U / A, GT legs) | 11 / −2 / 54 | 43 / 56 / 0 | all three bend over plausibly; A is a straight-back hip hinge with the pelvis tilted forward, U a curved back over an upright pelvis; the legs are identical |
| 3dpw_pelvis_stable_bend (B / U / A) | 35 / 28 / 60 | 24 / 32 / 0 | same pattern |
| 3dpw_seated_or_crouched (flat_guitar_01 500, fixed rule) | 50 / 76 / 95 | 43 / 21 / 0 | all read as seated; A's pelvis sits upright with the trunk rigid on it; knee-hinge deviation 36 / 43 / 52° |
| 3dpw_walking_upright | 1 / 20 / 2 | 0 / −18 / 0 | **U leans the hips and spine by the camera pitch**: camera-vertical pelvis, so a 17° camera tilt becomes hip flexion plus spine extension |
| 3dpw_turn_180 | −5 / −7 / −6 | ≈ 0 | turning unaffected |
| 3dhp_ts2_supine (A / U / L) | 71 / 104 / 53 | 0 / −66 / 19 | A and L lie with the knees raised as in the image; U keeps an upright pelvis and folds the trunk back 66° |
| 3dhp_ts4_lying (standing waist bend) | 91 / 81 / 66 | 0 / −13 / 25 | deep bend: A all at the hips, U at the spine, L between; all plausible at the shown frame; U flips elsewhere in the window |
| 3dhp_upright_lying_transition | 48 / 11 / 35 | 0 / 37 / 13 | identical while upright; differ once bending |
| 3dpw_bent_or_seated_s0 (B_S0 / U_S0 / A_S0) | 12 / −1 / 52 | 38 / 52 / 0 | same split; S0 legs close to GT here |
| 3dhp_ts2_supine_s0, 3dhp_s0_spins_s0 (A_S0 / U_S0 / L_S0) | — | — | S0 legs bend backwards / knees up in a prone subject under **every** policy (sensor); U_S0 additionally stands the pelvis up |

**Answers to the review questions:**
- **Does the pelvis turn or pitch incorrectly?** Under A it pitches with the trunk. Under U it stays camera-vertical
  (wrong when lying, camera-tilted when walking).
- **Does the torso bend independently from the legs?** Yes under every policy now that the legs are articulated.
- **Are lying poses coherent?** A and L yes, U no.
- **Do the hips compensate with unnatural rotations?** U when lying (≥ 90°), A on deep seated/bent rows (> 120°).
- **Do normal walkers stay believable?** Yes for A and B; U shows the camera-pitch lean as hip/spine motion.
- **Do transitions create discontinuities?** Only through S0 frame flips (S0 phase) and through U's thigh flips when
  lying.
- **Is the rig more usable than the rest-leg baseline?** Yes. Legs now follow the observed geometry under every policy,
  so W94's rest-leg errors are gone.

## 9. Attribution

| Owner | Evidence | Effect |
|---|---|---|
| Root frame / anatomical pelvis | B vs U/A/L on GT geometry | hip/spine split (unobservable, within ROM) in upright and standing bends; posture-specific failures (U lying; U/A seated through the knee hinge; A lying hip > 120) |
| Mathematical joint compensation | world leg parity ≤ 4e-4 | exact for limb directions; not for pelvis mesh, hip roll or knee hinge |
| Existing IK convention (zero twist) | B knee-hinge violations 24–62% | dominates the predeclared rule independent of the pelvis |
| Existing IK limits | none exist | all ROM figures are evaluation bounds, not rig behaviour |
| S0 reconstruction accuracy | B_S0 vs B, same policy | thigh error +9–30°, knee hinge +20–40 pp, backward knees in prone frames |
| Pelvis mesh orientation | Blender review | follows R_o: A tilts the pelvis with the trunk, U keeps it camera-vertical |
| Temporal discontinuities | severe hip jumps | S0 frame flips / fallback (all S0 policies); U thigh flips when lying |
| Missing root translation / contact | not added | no foot locking; legs are articulated in place |
| Camera vs gravity | U on the upright walker | camera tilt becomes hip flexion plus spine extension under a camera-vertical root |

## 10. Negative cases and alternative explanations

**Knee hinge.** The 15° bound is strict against a zero-twist thigh. Real knees also allow tibial rotation, and SMPL's
own leg twist is not reproduced. A looser bound would raise every policy's acceptance roughly equally; it does not
change the policy ranking.

**3DPW lying.** It has 3 rows. Lying evidence comes from 3DHP, which has no pelvis reference.

**L's offset.** L's upright hip-extension violations reflect its constant offset, not its tilt tracking (W94: tilt
error ≈ 5° after offset on GT joints, 18° on S0 joints).

**Seated rows.** The seated/crouched "other" stratum mixes sitting, crouching and kneeling.

**Strata.** Strata use GT gravity (evaluation only). The runtime cannot tell lying from a standing bend (W94).

## 11. Tests and files

**Tests:** `tests/test_framepose_pelvis_frame_leg_articulation.py` (8 tests):
- BaseRig leg hierarchy and chain contract;
- leg IK/FK parent/local round trip and world parity across Object frames;
- the fixtures: waist bend vs body tilt, supine/prone, crouch, turn and transition;
- knee hinge and degeneracy (straight leg, zero-length thigh);
- spine angle decomposition;
- upper-body invariance;
- predeclared rule constants, GT isolation of the S0 phase, frozen hashes and no production import;
- the optional renderer leg FK check.

**Runs:**
- Together with the W91/W91A/W92/W93/W94 suites: 44 passed locally and 44 passed on LabServer63 (docker
  `animcv-external:cu124`).
- Broad regression was not run, because no shared production contract changed.
- The W95 replay rerun is byte-identical. Blender re-renders of W95 and W94 clips are pixel-identical.

**New files:**
- `scripts/run_pelvis_frame_leg_articulation.py`.
- `tests/test_framepose_pelvis_frame_leg_articulation.py`.
- This worklog.

**Changed (optional, backward compatible):** `scripts/render_runtime_heading_rig_blender.py` gains an optional
per-slot `fk_points_armature` FK check, reported as `fk_points_vs_export_max`. Frame-level `extra_bone_locals_xyzw` was
already supported and now carries the leg locals.

**Reproduce:** the W94 command plus `--w94-cases <w94>/replay`, with the script `run_pelvis_frame_leg_articulation.py`.
Rendering uses the W94 Blender command with `--cases <w95>/replay/cases_{3dpw,3dhp}.json`.

**Archive:** reports, cases, the log and the .blend files are on LabServer63 at
`~/animcv-output/framepose_pelvis_frame_leg_articulation_archive/`.

## 12. CASE and unresolved uncertainty

**CASE C.**

**Geometric feasibility is established.** Any root frame reproduces the observed limb directions once the legs are
articulated, within the existing hierarchy.

**Credible animation within upright and standing waist bends.** GT-free-compatible frames (U; A on TEST) meet the
predeclared rule with GT legs. They differ from the SMPL pelvis only in the unobservable hip/spine split.

**Outside that range the comparison is not isolated:**
- The predeclared rule is dominated by the zero-twist knee-hinge convention, which fails under the true pelvis too.
- With S0 legs, sensor error dominates.
- U is unusable for lying and A/L are not, but selecting between them needs posture evidence the runtime lacks.

So the batch neither shows that an accurate anatomical pelvis is required, nor that a single GT-free frame suffices
across postures.

**Unresolved:**
- the effect of a thigh-twist/knee-hinge policy, which does not exist in the current IK;
- the pelvis evidence for lying vs bending;
- a gravity reference;
- 3DPW lying coverage.
