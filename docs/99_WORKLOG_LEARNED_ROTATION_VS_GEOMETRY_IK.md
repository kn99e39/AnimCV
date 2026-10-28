# Worklog — Learned Joint Rotation vs Geometry-Only Rig Retargeting (2026-10-28)

## Starting state and environment

**Branch and HEAD.** `arch/single_frame_first`, starting HEAD `5cbf77d` (Worklog 98, CASE B). Local, origin and
LabServer63 were in sync and the worktree was clean. I read PRODUCT_VISION.md and Worklogs 94–98. During the batch a
documentation-only commit landed on the branch (`4cc453e`, cumulative architecture/codebase audit). It changes no code,
and this worklog's code and results do not depend on it.

**Prior decisions.**
- W98 CASE B is accepted within its scope and unchanged.
- W96 candidate T was not enabled or tuned.
- No production file changed.

**Machines.**
- Local M1 Pro: the replay, evaluation and Blender 5.1.2 rendering.
- LabServer63: the SMPL forward-kinematics audit (CPU, in docker `animcv-w97-hmr2:cu124`, which holds the licensed SMPL
  v1 model) and the test suite (docker `animcv-external:cu124`). No GPU job ran.

**What was not changed:**
- no model, checkpoint, prediction or preprocessing;
- no root/pelvis/thorax policy, W95 zero-twist IK or W96 parameter;
- no rig, FK/IK, schema or production default;
- no smoothing, interpolation, confidence threshold, translation, contact or gravity heuristic.

## 0. Answer

**CASE B — local benefit with limited applicability.**

HMR 2.0b's predicted SMPL hip rotation is a valid, GT-free, fully reconcilable extra degree of freedom:
- SMPL forward kinematics of its rotations reproduces its own joints to 3e-7 m;
- the only information it adds beyond the joint centres is thigh axial roll.

Used through one isolated adapter (R), it materially improves the rig only where zero twist was already known to fail:
deep knee flexion and seated/crouched poses. In ordinary low-flexion movement, which is most frames, it adds no
accurate roll information.

**3DPW TEST, per leg** (14,152 legs; same W98 T body frame and HMR joints; roll error vs the evaluation-only SMPL
thigh hinge; off-hinge = how far the rig calf leaves its knee hinge plane, W96):

| Knee flexion | Legs | Roll error Z → R | R − Z [95% CI] | Off-hinge Z → R | R − Z [95% CI] |
|---|---|---|---|---|---|
| 0–5° | 152 | 11.9 → 11.3 | −0.6 [−2.7, 1.1] | 1.9 → 1.8 | −0.1 |
| 5–20° | 2470 | 11.5 → 12.7 | **+1.2 [0.9, 1.6]** | 6.5 → 5.9 | −0.6 |
| 20–45° | 6245 | 11.1 → 12.8 | **+1.7 [1.3, 2.2]** | 6.9 → 5.6 | −1.4 |
| 45–90° | 4104 | 12.3 → 12.5 | +0.2 [−0.6, 1.1] | 7.3 → 5.0 | **−2.4 [−2.6, −2.1]** |
| ≥ 90° | 1181 | 20.0 → **11.2** | **−8.8 [−12.8, −4.1]** | 17.4 → **9.1** | **−8.3 [−11.7, −4.6]** |
| Seated/crouched/other | 1848 | 17.5 → **11.4** | **−6.1 [−9.8, −2.4]** | 14.1 → **7.8** | **−6.3 [−9.3, −3.5]** |

VALIDATION repeats the pattern:
- ≥ 90°: roll −6.4 [−9.9, −2.1], off-hinge −6.8;
- 45–90°: −1.2;
- 20–45°: +1.4;
- 5–20°: +1.0.

**Predeclared rules** (fixed before any R result):

| Rule | TEST | VALIDATION |
|---|---|---|
| R1 deep roll (flexion ≥ 45°, interval below 0) | **fail**: −1.77 [−3.89, 0.11] | pass: −1.82 [−3.49, −0.57] |
| R2 low-flexion roll ≤ +2° | pass (+1.12) | pass (+0.93) |
| R3 deep off-hinge (interval below 0) | pass (−3.69) | pass (−1.98) |
| R4 low-flexion off-hinge ≤ +2° | pass (−0.54) | pass (−0.71) |
| R5 temporal (no added severe jumps, jitter ≤ Z + 1°) | pass | pass |
| R6 hip axial \|roll\| > 45° ≤ 2% | pass (0%) | pass (0%) |

**Attribution.** Under the oracle SMPL pelvis, zero twist is 4.6–5.5° from SMPL roll below 90° flexion. HMR's own
thigh roll is 11–13° from it at every flexion. So:
- at low flexion the predicted rotation is less accurate than simply carrying no twist from a correct pelvis;
- under the GT-free pelvis, the two come out roughly equal only because the W98 pelvis error spoils zero twist by
  about the same amount;
- the rotation channel's clear value is confined to ≥ 90° flexion and seated poses.

## 1. Historical reproduction and frozen inputs

**Frozen inputs** are identical to W98 (pins checked at run time; STOP if changed):
- W97 HMR 2.0b predictions `fc5d3305…7003` and population `533895c9…cdb9`;
- S0 TEST / VALIDATION / 3DHP predictions `4357c7bd…` / `402d70a2…` / `a78adeb0…`;
- J1X arms `cb9dc51e…` / `356f8cee…`.

The model provenance is W97 §1: 4D-Humans `efe18def…`, `epoch=35-step=1000000.ckpt` `c9ea09c8…`, SMPL v1 neutral
`98e65c74…` (non-commercial research licence).

**W98 baseline reproduced exactly.** Z under the W98 T frame equals W98 condition T to the last digit:

| | W98 T | W99 Z |
|---|---|---|
| TEST roll vs SMPL (7076 rows) | 12.243794 | 12.243794 |
| TEST off-hinge | 7.776947 | 7.776947 |
| VALIDATION roll vs SMPL (3407 rows) | 11.175499 | 11.175499 |
| VALIDATION off-hinge | 6.517196 | 6.517196 |

## 2. Gate 1 — SMPL rotation and rest-frame audit

**SMPL kinematic chain.**
- Joints: hips 1 (L) / 2 (R), parent pelvis 0; knees 4 / 5; ankles 7 / 8.
- `body_pose[j − 1]` is joint j's rotation relative to its parent.
- The thigh's global frame is G = global_orient · body_pose[hip − 1], in OpenCV crop-camera axes. AnimCV canonical is
  C · G with C = (x, z, −y).

**Forward-kinematics consistency** (server, `fk-audit`, all 13,211 predictions, SMPL v1 joint regressor with each
prediction's betas):

| Check | Result |
|---|---|
| Root-relative joint position error (max / mean) | 3.0e-7 / 1.3e-7 m |
| Thigh / shin direction error (max) | 1.1e-5° / 1.3e-5° |
| Thigh template direction spread across betas | 1.75° |

The predicted rotations and the 24 joints are one consistent kinematic state, so the HMR thigh frame's swing *is* the
HMR hip→knee direction. What the rotation adds beyond the joint centres is (b) axial rotation about the thigh, plus the
knee's non-hinge components (d).

**Separated components:**
- (a) **Swing:** identical by construction. World thigh/shin directions differ by ≤ 1.2e-6° between Z and R.
- (b) **Axial roll:** the only experimental variable.
- (c) **Knee flexion:** taken from the joint geometry in both.
- (d) **Knee non-hinge:** HMR's own knee local rotation has a 9.4° off-hinge swing component on average (≥ 90°:
  17.6°).
- (e) **Pelvis:** W98 T, unchanged.

**Template and rig rest frames** (zero betas; rig from `baserig_blender_rest.json`):

| | Left | Right |
|---|---|---|
| SMPL rest thigh vs template +x (hinge axis) | 84.8° | 95.7° |
| SMPL rest knee bend | 9.1° | 9.3° |
| Rig rest knee bend | 5.0° | 5.0° |
| Rig rest hinge n_rest vs SMPL +x (body axes) | 8.4° | 8.1° |
| Rig thigh bone-local Z vs SMPL +x | 3.1° | 3.3° |
| psi0, analytic rest roll offset (W96, used) | +7.82° | −7.53° |
| psi0 if bone-local Z were the hinge | −0.01° | −0.99° |

**Mapping.**
- The mapping is analytic: the SMPL template +x hinge is related to the rig's rest hinge by psi0, a signed angle about
  the rig rest thigh in body axes. Nothing was fitted.
- **The W96 7–8° rest-axis gap remains a convention uncertainty.** n_rest and bone-local Z differ by 7.8° / 6.5°.
- R's roll error vs SMPL is *independent* of psi0, because the same offset is applied in R and in the evaluation and
  cancels.
- R's rig pose (off-hinge, visuals) and Z's roll error do depend on psi0, within that gap.
- SMPL's rest knee is bent 9°, so its rest plane is ill-defined. The template +x axis is used as the hinge, as in W96,
  where the SMPL knee rotation axis was within 5.2° of it.

Gate 1 passed: the mapping is defensible without evaluation GT.

## 3. Gate 2 — what the rotation adds (observability)

**Is roll already fixed by the joints?** Angle between HMR's hinge axis and HMR's own hip–knee–ankle plane normal
(flexion ≥ 20°; a pure hinge would give 0):

| Flexion | TEST | VALIDATION |
|---|---|---|
| 20–45° | 17.1° | 13.9° |
| 45–90° | 11.1° | 11.7° |
| ≥ 90° | 16.3° | 16.2° |

**Reading:**
- The predicted roll is a genuinely separate degree of freedom: it is not recoverable from the 17 joints.
- But it also does not agree with HMR's own joint plane. HMR's knee is far from a hinge about its thigh axis: 9.4° of
  off-hinge knee rotation, versus about 5° for the 3DPW SMPL fits (W96).

**Numerical validity:**
- every prediction yields a non-degenerate hinge axis (14,152 / 6,814 / 5,064 legs; 0 unavailable);
- R's hip axial rotation stays within ±45° on every leg in every split (0%; mean 3.9–5.3°).

**Shared-model caveat.**
- HMR's rotations and joints come from one network and one SMPL layer, so they are not independent sensors.
- The SMPL roll reference shares the body-model family. Agreement with it can partly reward shared conventions.

## 4. Candidate definitions (declared before results)

All conditions share:
- the W98 T body frame (HMR pelvis Object, HMR joint thorax);
- the HMR 17 joints, J1X arm, W95 two-bone geometry, bone lengths, frames and timestamps.

| Condition | Thigh axial roll | Role |
|---|---|---|
| Z | none: shortest arc from the rest thigh in the Object frame (W95/W98) | baseline |
| R | φ = signed angle about the posed world thigh axis from Z's rig hinge to HMR's hinge axis a = C·G·e_x, minus psi0; no weighting, flip or clamp | candidate (GT-free) |
| O | the same construction with the 3DPW SMPL thigh axis | evaluation-only reference |
| Z_O | zero twist under the oracle SMPL pelvis (W98 O) | attribution only |

**Calf.** It is a dependent transform: the shortest arc from the rolled rest calf to the same target direction
(W96 `rolled`), so the world shin direction is unchanged. As a side effect, R's calf twist relative to the thigh is 0 by
construction. Calf twist is therefore reported, but it is not evidence for R.

**R's inputs.** R reads only `global_orient`, `body_pose`, the HMR joints and the rig. A source test asserts this.

## 5. GT-controlled comparison

### By flexion and posture

The flexion table is in §0. Further groups (TEST unless stated):

| Subset | Legs | Roll Z → R (R − Z) | Off-hinge Z → R | R − Z_O roll |
|---|---|---|---|---|
| Upright | 11,816 | 11.6 → 12.9 (+1.3) | 6.8 → 5.4 | +8.3 |
| Waist bend | 482 | 8.1 → 8.0 (−0.1) | 7.5 → 6.0 | +1.7 |
| Seated/crouched/other | 1848 | 17.5 → 11.4 (**−6.1**) | 14.1 → 7.8 | −3.7 [−7.4, 0.9] |
| GT knee flexion ≥ 45° | 5098 | 14.4 → 12.7 (−1.7, n.s.) | 10.3 → 6.7 | +4.0 |
| S0 wrong knee plane (occluded) | 980 | 16.1 → 18.0 (+1.9, n.s.) | 9.1 → 6.6 | +12.8 |
| Out-of-plane limbs | 6180 | 13.5 → 12.6 (−0.9, n.s.) | 9.5 → 6.6 | +4.6 |
| Gravity-lying | 6 | 15.1 → 12.9 | 8.2 → 5.5 | six legs; no claim |
| VALIDATION seated/other | 2006 | 11.9 → 10.3 (−1.7) | 8.3 → 6.4 | +1.6 |

**What the oracle comparison shows (O − Z, R − Z_O):**
- **Below 90° flexion, R is 6–8° worse than zero twist under the oracle pelvis** (TEST 5–20°: +7.8; 20–45°: +8.1;
  45–90°: +7.0; VALIDATION similar).
- **At ≥ 90°, R beats even oracle-pelvis zero twist on TEST** (−6.7 [−9.9, −1.8]). On VALIDATION it only matches it
  (−0.5, n.s.).
- **The GT roll oracle O has more off-hinge than Z or R** (TEST 9.3° vs Z 7.8° and R 5.7°). This is because HMR's joint
  plane is not consistent with the true roll. **R's off-hinge reduction is partly structural:** R's hinge comes from the
  same model that generated the joint plane. It is not independent proof of anatomical correctness.

### Other quantities (TEST; VALIDATION in brackets)

**Hip axial rotation:**
- R |roll| averages 4.2° [3.9°], rising to 9.2° [7.9°] at ≥ 90°. None exceeds 45°.
- The GT oracle needs 12.2° [11.2°] under the HMR pelvis, with 7.7% [1.0%] above 45° at ≥ 90°. Much of the GT roll
  relative to the HMR Object is pelvis-error compensation.

**Left/right.** The signed roll error is biased on the right leg for both Z (+5.5° [+4.9°]) and R (+3.7° [+4.6°]),
against left +0.6° [+2.9°] for R. R's error is psi0-invariant, so the right-leg bias lies in HMR's predicted thigh roll
or the crop camera, not in the rest offset.

**Knee axis vs SMPL knee axis** (flexion ≥ 20°): Z 15.3°, R 13.4°, O 9.7°.

**Parent/local validity:**
- R FK vs IK targets ≤ 2.0e-13 m on every leg.
- Z has one leg out of 14,152 (TEST, downtown_walking_00 actor1, frame 1214, right) where the W95 IK locals miss their
  own target by 4.9 cm. This is inherited from W95 and present in W98 T. R reconstructs it exactly.
- Blender FK vs export ≤ 9.6e-5 m.

**Coverage.**
- 3DPW TEST: 7076 rows (all with T and R).
- VALIDATION: 3407.
- 3DHP: 2532 of 2728 (W98 T coverage; 196 rows lack the GT leg reference the evaluator needs).
- R unavailable: 0 legs.

### Temporal (adjacent matched frames, no smoothing)

| | TEST Z | TEST R | VAL Z | VAL R |
|---|---|---|---|---|
| Severe thigh-local jumps | 0 | 0 | 0 | 0 |
| Severe roll steps | 0 | 0 | 0 | 0 |
| Roll jitter vs SMPL (mean / p95) | 3.88 / 12.4 | 4.28 / 13.8 | 3.66 / 11.5 | 3.78 / 12.1 |

**Reading:**
- **Jitter** is the adjacent-frame change of the signed roll error, so the subject's own roll motion is removed.
- **R adds 0.1–0.4° of per-frame roll noise** from HMR's independent single-frame predictions.
- **Z's jitter is the W98 pelvis noise passed through zero twist.** R's world roll does not depend on the Object, so it
  replaces that parent-frame noise with HMR's own thigh-roll noise.
- **The GT oracle O shows 4 / 6 severe jumps.** These are the W98 HMR-pelvis Object spins that O must compensate for.

**3DHP** (2532 rows; rig-internal only, no roll reference; HMR is in-domain there):
- no severe jumps for Z or R;
- off-hinge Z 8.2 → R 5.7 (≥ 90°: 12.6 → 7.0; lying: 10.5 → 8.2);
- R |roll| averages 5.3°.

## 6. Information-addition attribution

**Source held constant.** The source joints, swing, pelvis, thorax, arm and bone lengths are identical. Every Z/R
difference is therefore thigh roll plus its dependent calf transform.

**Where R helps (≥ 90° flexion, seated):**
- Zero twist bends the calf 17° out of the hinge, a sideways shin.
- HMR's predicted thigh roll halves the roll error (20 → 11°) and the off-hinge (17 → 9°).
- This is the W96 genuine deep-flexion defect. The predicted rotation fixes about half of it on its own.

**Where R does not help (< 45°, most frames):**
- The joint plane is weak or undefined, so roll is underdetermined by geometry. That is exactly where an independent
  rotation channel would be most valuable.
- HMR's thigh roll is 11–13° from SMPL there, against about 5° for zero twist from a correct pelvis.
- Its learned prior does not select the correct axial rotation among the geometrically equivalent ones.

**What is circular:**
- R's improvement against SMPL roll shares the SMPL body model with HMR.
- R's off-hinge improvement is structurally tied to using the same network's rotation.
- The oracle O having *more* off-hinge than R shows that "less off-hinge" is not by itself "more correct".

**Not attributable to R:**
- the W97/W98 upright-shin regression (HMR knee depth/flexion; identical in Z and R, see the `hmr_upright_shin_regression`
  clip);
- the W98 pelvis error;
- the occluded-leg failures (S0 wrong-plane / bar scenes, where R is +1.9° worse in roll).

**Explicit probes:**

| Probe | Case | Observation |
|---|---|---|
| Joint directions agree, predicted roll differs | `worst_R_vs_Z`, downtown_bar_00 a1 /1074, occluded standing legs | R rolls +6/+10°, GT needs −22/−20°. R reduces off-hinge yet moves away from the reference roll. Visually the legs differ little; both are far from the GT legs |
| Near-straight, roll underdetermined | `near_straight_max_roll`, arguing_00 a1 /283, flexion 5–13° | R's roll stays at ±5° (no needless twisting); the GT oracle needs +26/+28°, mostly W98 pelvis compensation that R inherits from HMR's own global orient |
| Deep flexion exposes a knee-axis defect | `seated_or_crouched`, `worst_hinge`, flat_guitar_01, flexion 100–127° | Z off-hinge 35–52° (sideways shin), R 23–28°, O 0–9°. R's roll is −21/+17° against the oracle's −45 to −55/+34° |
| Rotation conflicts with joint directions | HMR hinge vs own plane | 11–17° mean (Gate 2) |

## 7. Skinned-rig review

**Settings.**
- BaseRig skinned mannequin; FIXED = Z, RUNTIME = R, GT = O (evaluation-only roll oracle).
- On 3DHP the third slot repeats Z and is labelled so, because there is no SMPL roll reference.
- Identical HMR pelvis/thorax/joints, J1X arm, timestamps, camera, rig scale and export.
- Full body: ortho 3.2 / centre height 0.6. Knee close-ups: 1.3 / 0.45.
- The plots show hip axial roll and knee off-hinge for the leg with more flexion in the window.

**Selection (fixed rules; provenance):**
- W96 windows: walking_upright, pelvis_stable_bend, seated_or_crouched, turn_180, worst_hinge.
- W97 window: s0_wrong_knee_plane.
- W98 window: worst_hmr_pelvis.
- New rules declared in `PREDECLARED`:
  - `hmr_upright_shin_regression`: downtown_bus_00 a1 /640;
  - `near_straight_max_roll` (failure probe): arguing_00 a1 /283;
  - `worst_R_vs_Z` (failure probe): bar_00 a1 /1074.
- 3DHP TS2 /500 and TS4 /340.
- Close-ups: seated, worst_hinge, walking_upright, near_straight, worst_R_vs_Z.

**Observations:**
- **Deep seated knees** (flat_guitar_01): the visible Z defect is a calf bent 35–47° out of the hinge, so the shin and
  foot turn sideways. R reduces it to 22–28°, and its shin/foot orientation lies between Z and the GT-roll oracle. The
  improvement is visible but partial. The close-up framing (centre 0.45) cuts the seated pose near the knees, so the
  full-body view is the more informative one here.
- **Upright walking** (back-facing, walkBridge): R's roll stays within ±6° of zero twist, and Z and R are visually
  indistinguishable. The oracle departs by −20 to −28° in the first second, where the W98 pelvis is wrong; R does not
  follow it.
- **Near-straight legs:** no visible twisting in R; the roll is constant at about ±5°.
- **Turn 180°:** R moves in the oracle's direction during the turn (−9° vs −30°) and otherwise matches Z.
- **HMR upright-shin regression:** identical in Z and R. Both shins are over-flexed relative to the GT legs.
- **Occluded bar scenes** (downtown_bar_00 /367 and /1074): R rolls +5 to +10°, while the reference needs −20 to −43°.
  R lowers off-hinge (right knee about 10° vs Z about 17°), while the oracle bends the HMR knee 26–45° off its hinge. The
  occluded HMR leg geometry contradicts the true roll, and both Z and R legs stay far from the GT legs.
- **3DHP TS4 (bent/lying):** R's roll flickers by ±10° between sampled frames where Z is static. Off-hinge is similar.
  This is the one visible temporal cost of R.
- **3DHP TS2 supine** (legs raised on the sofa): R's roll varies within ±6° and off-hinge tracks Z. Rows without a T
  body frame are labelled "unavailable" in all slots, with no fallback.
- **Pelvis mesh and hip attachment:** unchanged in every case; only the thigh/shin roll differs.

**Blender criterion.** No visible new twisting of near-straight legs or feet. Deep seated knees are better hinged, but
only partially. One visible temporal flicker on 3DHP.

## 8. Coverage and uncertainty

**Intervals.** Sequence-cluster bootstrap, 95%. TEST has 37 sequence-actors and VALIDATION 16. The ≥ 90° bin is small
(1181 / 299 legs) and its effect differs between splits: R − Z_O is −6.7 on TEST and −0.5 on VALIDATION.

**Uncertainties:**
- The 7–8° rig hinge convention gap: it affects R's rig pose and off-hinge, not R's SMPL-roll error.
- The 9° SMPL rest knee bend.
- The crop-camera frame: R's world axis is in HMR's crop camera; W98 measured about 1–2° of yaw from this.
- Six gravity-lying TEST legs: no lying claim.
- 3DHP has no roll reference and is in-domain for HMR.

**Not tested:**
- the W96 joint-plane candidate T on HMR joints (not authorised);
- any combination of R with T or flexion weighting (would be tuning).

## 9. Tests and files

**New files:**
- `scripts/run_learned_rotation_vs_geometry_ik.py`: rotation readout, adapter R, Z/R/O replay, evaluation, temporal
  accounting, Blender cases, and the `fk-audit` subcommand.
- `tests/test_learned_rotation_vs_geometry_ik.py` (9 tests).
- This worklog.

No existing file changed. The renderer and composer were used unchanged.

**Tests (9):**
- frozen identity and predeclared rule set;
- SMPL hierarchy and canonical hinge axis (body_pose indices, upright fixture);
- R equals the oracle construction for an equal axis, and tracks an axial hip rotation;
- world geometry, FK round trip and unit quaternions identical across Z and R (bent, skewed and straight legs);
- straight-leg roll is defined, with mirror parity for mirrored hip rotations;
- knee swing/twist components and the unavailable status for a degenerate axis;
- SMPL FK chain;
- GT isolation of the candidate path and the fixed T body frame, with no production import;
- temporal accounting.

**Test runs.** The W93–W99 suites (59 tests) passed locally and on LabServer63. Broad regression was not run: no shared
contract changed.

**Omitted:** a Blender re-render of the 3DPW cases. Only the 3DHP cases were re-rendered (§10).

**Fixed during the run.** The close-up case list was first filtered character-wise, which included every case. It was
fixed to the declared five, and the existing close-up file was filtered by the same rule before rendering. The
determinism re-run (§10) was made with the fixed code.

**Outputs.**
- Videos and contact sheets: `~/animcv-output/99_framepose_learned_rotation_vs_geometry_ik/`.
- Report, FK audit, cases and .blend files: archived on LabServer63 at
  `~/animcv-output/framepose_learned_rotation_vs_geometry_ik_archive/`.

## 10. Reproducibility

- A full replay re-run into a separate directory, with the final code including the close-up fix, is byte-identical:
  `report.json`, `cases_3dpw.json`, `cases_3dhp.json` and `cases_3dpw_closeup.json`.
- A re-render of the 3DHP cases gives 432 identical PNGs (maximum pixel difference 0).
- Z reproduces W98 condition T exactly (§1).
- The FK audit is a deterministic pass over the frozen predictions.

## 11. CASE and evidence limits

**CASE B.**
- **The representation contract holds:** exact FK consistency, analytic rest mapping, identical world limb geometry,
  no GT, no added severe jumps.
- **The predicted SMPL hip rotation adds a genuine axial degree of freedom.** It materially improves BaseRig knee
  articulation in deep flexion and seated/crouched poses: roll −8.8 / −6.4°, off-hinge −8.3 / −6.8° at ≥ 90° on TEST /
  VALIDATION, with a partial visible improvement in the skinned rig.
- **It is not generally useful:**
  - predeclared R1 fails on TEST;
  - at low and moderate flexion HMR's thigh roll is 11–13° from the reference, 6–8° worse than zero twist from a correct
    pelvis;
  - it adds small per-frame roll noise;
  - part of the off-hinge gain is structural, because rotation and joints come from the same network;
  - the SMPL roll reference shares HMR's body-model family.

**It is not CASE A:** the general improvement is unsupported.

**It is not CASE C:** there is a rig-internal and visual improvement in a bounded, important subset, and on VALIDATION
it is supported by the deep-flexion roll rule.

**It is not CASE D:** both gates passed.

**Answer to the completion question.** Yes, but only in a bounded regime. HMR 2.0b's predicted hip rotation carries
information beyond its 3D joint centres that improves the final rig in deep knee flexion and seated poses. It does not
carry accurate axial information in ordinary movement, where joint geometry leaves roll underdetermined.
