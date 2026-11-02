# Worklog — Source-Only Restoration: Rig-Level Utility and Failure Boundaries (2026-11-02)

## Scope, starting state and principle

Task scope (summarized, not copied): an evidence-only rig diagnostic. It asks whether W104's fixed source-only
raw-joint restoration (R11) gives better **BaseRig animation** than unchanged M_Main, or only a better 3D metric. It
uses the same frozen geometry and the unchanged W102 BaseRig/FK/IK/render contract. It also locates where any remaining
failure sits: 2D observation, frozen S0, retargeting, identity or reference.

No observation threshold, mask, S0, checkpoint, detector, tracking, crop, IK, retargeting, root policy, rig asset,
renderer setting or runtime was changed. R11 is not adopted as a policy.

- Branch `arch/single_frame_first`, starting HEAD `12311cc31fe51aa6e1de4ddcddaea2c25cbd5cec`; instruction blob
  `a90bbb53ee3ff3baa11e4a018afdcc38b9a17c0f`. Both were verified before any work.
- `git fetch` showed no divergence from `origin/arch/single_frame_first`, and the working tree was clean.
- Run unattended by the scheduled automation worker, entirely on **Main**:
  - RTX 5080, Python 3.11.9, `C:\Projects\AnimCV\.venv_build_full`;
  - Blender 5.1.0 at `C:\Program Files\Blender Foundation\Blender 5.1\blender.exe`.
- GPU utilization was 0% before the run, and no other job was visible.
- No LabServer63/69 job, no package install, no network data access.
- W95–W104 verdicts, artifacts and the W104 COMPLETE marker are unchanged.

Regimes, kept apart throughout:
- `real_animcv_observation`: M_Main, and R11 (the evaluation-only W104 rule).
- `evaluation-only GT_RIG`: GT 3D mapped through the same fixed rig. This is a reference, never a candidate.
- The real clip is GT-free throughout.

The 3DPW same-person population has been reused for architecture diagnosis many times; it is **not a holdout**.

## Answer

**CASE A under the declared rule (conditional, scoped), in both the primary run and the reference-validity
sensitivity run.** The scope matters as much as the verdict.

1. **Restored legs (417 PARTIAL rows, 20 videos): the source gain carries through to the rig.**
   - Under M_Main, 411 of 417 rows have a failed rig leg: folded or wrong-branch knee, or a bone more than 45° off.
   - Under R11 this falls to 131, with 0 newly failed rows.
   - Mean rig leg-bone direction error vs GT-through-rig falls from 61.1° to 21.6°: **−39.5° [−45.0, −29.9]**.
   - Temporal direction-step events fall from 439 to 53. Local-thigh pops fall from 88 to 0.
   - In the rendered windows, M_Main crouches standing people. R11 stands them up (see §8).
2. **Arm-only restorations (806 rows) have no rig effect under the W102 contract.**
   - The W102 rig consumes S0 geometry only in `thigh_l/calf_l/thigh_r/calf_r`.
   - Rig leg direction change: −0.05° [−0.21, +0.14].
   - W104's arm-side gains (and its largest harm, `parcours_00` #1018) are invisible in this rig.
3. **R11 still fails, mostly through low-confidence 2D hallucination.**
   - 131 P_leg rows still have a rig failure.
   - Of the 151 failing legs, 99 are attributed to restored 2D tokens more than 0.10 off (category A).
   - Seated `downtown_cafe_00` actor1 rows improve at source level (485 → 313 mm) but get **worse in the rig**
     (51° → 89°). Here the restored leg joints are badly localized.
4. **The real clip (left wrist only) gives a mixed result.** It does not test leg recovery.
   - Long back-facing run (frames 154–181):
     - M_Main holds a stable ~95° elbow fold; complete frames have a median of 12°.
     - R11 gives a moderately bent, varying arm (34–71°).
   - Side-on run (frames 113, 114, 116):
     - The raw wrist (confidence 0.15–0.16) jumps about 215 px up to head height.
     - R11 renders this as a hand-to-face flicker: three steps of 125–137° at 50 fps.
   - M_Main's 4 large arm steps occur at the mask transitions; R11 has none there but 4 inside the run.

The CASE is an evidence verdict. It does **not** authorize a threshold change, mask policy, default change, retargeting
fix or production integration. None was made.

## 1. Preflight and provenance (`verify/verify_report.json`, SHA `167b0e95…`)

W104 identities:
- Manifest `a6fa3874b7959b74c8859a5ffbf983728daaed67519e1a9598be899a10c91aeb`; all 22 files match.
- Declaration `6a64c8d1ffc57f928fdf2fbaaf40dffd5b848a927b662e90ff584257bfbae487`.
- The committed W104 Worklog ends with its COMPLETE marker.

W102 and W103 identities:
- W102 manifest `5186a170…`. The 13 W102 files used here all match: rows, HMR2, J1X, observations, real-clip
  predictions/inspection, rig cases/render report and population.
- W103 manifest `3aad853c…`, S0 checkpoint `6c12fea9…`, bank digest `75519e63…`.

Rig identity:
- The rest snapshot, FK calibration, chain file and skinned FBX are equal (LF-normalized) to W102's Main source
  snapshot. So are 9 rig code files: the W102 rig review, the W91 renderer, W95/W93 IK/torso, the HMR handoff, and
  `src/retarget` and `src/motion` modules.
- Recorded SHAs: rest `1dcdc826…` (CRLF checkout), skinned FBX `c2947ba7…`.

Rows and alignment:
- All 10,483 actor-frames: row order, bank position, sample id, frame index and timestamp are equal to the bank.
- `rows.json` stores **root-relative** joints. It equals the stored M_Main prediction after root-centering (max diff 0).

R11 input (GT-free; the identity label is never consulted):
- The fixed W104 rule, rebuilt from the retained raw M1 landmarks, reproduces W104's 1,225 rows, new-token masks and
  R11 tensors **byte-exactly**.
- 5,763 same-person rows have no restorable token; their R11 tensor is byte-equal to M_Main's.

Real clip:
- `examples/production_demo/source.mp4` (`0dda8a5f…`) equals W102's `historical_source.mp4`.
- 199 decoded frames at 50 fps; per-frame decoded-BGR digests and timestamps equal W102's inspection.
- `examples/production_demo/README.md` describes a 120-frame audit. The clip itself decodes to 199 frames, which W102
  used. This is a documentation discrepancy only.
- Content (W102 inventory sheet): one performer, full body, standing → walking toward camera → turning → walking away.
  Rights undocumented; private diagnostic; not redistributed; not unseen.

## 2. Declaration (hashed before any rig outcome)

`declare/declaration.json`, SHA-256 **`6238b78f271f56f7c112aa499c07e81c39a68530fcb5b4368fe18665a99afae7`**, records
`rig_outputs_read: false`. Build/analyze refuse to run without this hash. The full text of the contract is in the
JSON. In short:

**Populations.**
- Full W102 denominator: 10,483 (VAL 3,407 / TEST 7,076).
- W104 restoration rows: 1,225. Of these, 1,223 are PARTIAL:
  - **P_leg** (≥ 1 restored hip/knee/ankle): 417.
  - **P_arm** (arm-only): 806.
- Every PARTIAL row has an HMR parent and thorax. No degenerate chain occurred, so rig-eligible = 1,223, with 0
  exclusions.

**Metrics** (GT is evaluation-only):
- **rig_dir_ref**: world direction of each rig leg bone vs the same bone of GT mapped through the same rig and parent.
  Also rig_dir_gt, against raw GT segments.
- Rig knee absolute error.
- Rig knee branch (W97 rule: both ≥ 20°, normals > 90°).
- **Rig limb failure**: any bone > 45° (W102/W104 scale) or a wrong rig knee branch.
- **False fold**: rig knee more than 45° above the GT-through-rig knee.
- W95 leg ROM violations.
- Source-to-rig discrepancy (W102).
- Ankle endpoint relative to the hip, in armature units.
- Temporal events, on original 10 Hz same-person neighbours:
  - rig bone step > 45° while the GT step < 15° (W102);
  - local thigh step > 90° while the GT thigh step < 45° (W95/W102).
- Video-clustered bootstrap: 2,000 replicates, seed 1337.

**Gates on P_leg:**

| Gate | Requirement |
|---|---|
| RIG_BENEFIT | Δrig_dir_ref CI upper < 0 and rig-failure rate change CI upper < 0 |
| MATERIAL | Net recovered rig-failure rows ≥ 10% of M's |
| RIG_HARM | New rig failures > 0.25 × recovered; or the same ratio for false folds or leg-ROM rows; or p99 worse; or temporal event Δ CI lower > 0 |
| SPLIT | Δ < 0 in both VAL and TEST |

- **A** = RIG_BENEFIT ∧ MATERIAL ∧ SPLIT ∧ ¬RIG_HARM.
- **B** = source benefit ∧ (¬RIG_BENEFIT ∨ RIG_HARM).
- **C** otherwise.
- **D** = identity, reproduction, rig-equivalence or Action-identity failure.

**Harm attribution** (ordered rules): F GT invalid → E identity → D source-to-rig/IK → A raw 2D > 0.10 → B frozen S0
response → G unresolved. C (crop) is not repeated; W104's R01 bounds it.

**Render windows** (24 samples each; W102's cafe window is 36):

| Window | Sequence, start frame | Notes |
|---|---|---|
| leg_natural_most | bar_00 a1 #276 | 24 leg-restored rows |
| arm_natural_most | windowShopping_00 a0 #110 | 19 arm-only rows |
| w104_largest_harm | parcours_00 #982 | centred on #1018 |
| new_wrong_knee | parcours_01 #602 | |
| w102_max_source_regression | cafe_00 a1 #685 | W102's window; 32 leg rows |
| high_motion | runForBus_00 a1 #320 | |
| identity_caveat | bar_00 a0 #276 | 22 unresolved + 2 other-person rows, all changed by the rule |
| complete_control | basketball_01 a0 #66 | all no-op |

## 3. Paired source-to-rig contract

**Bones consuming S0 geometry: `thigh_l`, `calf_l`, `thigh_r`, `calf_r` only.** The rest of the rig is shared:
- Armature Object = frozen W97 HMR2 pelvis; `spine_01` = HMR thorax.
- Left arm = frozen W87 J1X (`cb9dc51e…`); right arm at rest.
- These are shared byte-identically by both conditions, checked per frame in the build.
- The right arm has no rig chain at all. 483 PARTIAL rows with restored right-arm tokens cannot be expressed in this rig.

Invariants checked:
- Same row/frame/timestamp/sequence in both slots (stored test over every rendered frame).
- Identical rest snapshot digest, alignment, chains, camera, lights, Workbench settings, 480 px, orthographic scale 3.2.
- Keys at original frame indices, CONSTANT interpolation.
- Skinned rig bound within **6.10e−5** (≤ 1e−4).
- Blender FK bone tails vs exported points ≤ **6.84e−5** (≤ 1e−3); arm end vs export ≤ 1.04e−4.
- Action audit (`render_audit.json`):
  - M and R11 Actions have identical channels (32 F-curves) and keyed frames in every window.
  - 0 value mismatches on the 77 rendered samples whose source is unchanged.
  - The complete-control window's two Actions have **identical value digests**.
- All 5,763 unchanged same-person rows give an identical rig export.
- Unavailable samples:
  - Rows without a tracked subject are blanked "UNAVAILABLE"; no pose is keyed for them.
  - The historical renderer's CONSTANT hold inside the `.blend` remains a property of that file and is not presented
    as a reconstruction.
- No GT, oracle or benchmark value enters a candidate export. The construction code is statically tested for this.
  The GT-through-rig reference is written to a separate case file and rendered in a separate Blender run.
- The historical renderer saves every Action created so far into each per-case `.blend`. The audit selects Actions by
  name.

This is an evaluation-only Blender output. It is not an end-to-end production pipeline.

## 4. Primary rig-level comparison (`analysis/analysis_report.json`, SHA `258b4578…`)

**P_leg** (417 rows, 20 videos; VAL 46 / TEST 371):

| Metric | M_Main | R11 | Δ R11 − M [95% video CI] |
|---|---:|---:|---:|
| Source MPJPE (mm) | 175.9 | 108.3 | −67.6 [−78.2, −43.5] |
| Source severe rows | 108 | 12 | |
| **Rig leg-bone dir vs GT-through-rig (°)** | **61.1** | **21.6** | **−39.5 [−45.0, −29.9]** |
| Rig dir vs raw GT segment (°) | 61.1 | 21.6 | −39.5 [−45.0, −29.9] |
| Rig knee abs error (°) | 64.6 | 21.9 | −42.7 [−53.9, −28.4] |
| Rig ankle endpoint error (armature units ×1000) | 59.7 | 21.4 | −38.3 [−44.9, −26.6] |
| p99 rig dir (°) | 100.2 | 58.1 | |
| **Rig-failure rows** (new / recovered) | **411** | **131** (0 / 280) | rate −0.67 [−0.71, −0.56] |
| False-fold rows (new / removed) | 314 | 30 (4 / 288) | |
| Leg-ROM rows (new / removed) | 416 | 314 (1 / 103) | |
| Rig knee branch, 449 both-eligible legs (wrong) | 178 | 17 (13 new / 174 fixed) | |
| Source-to-rig max discrepancy (mean; rows > 45°) | 28.6° (77) | 7.1° (5) | |

Splits:
- VAL −21.7° [−28.3, −14.8]; TEST −41.7° [−47.0, −33.2]. VAL again shows about half of TEST's gain.
- Sensitivity without the 162 root-behind-camera rows (407 P_leg rows): −40.2° [−45.5, −31.0]; failures 401 → 123.
  CASE is still A.

**P_arm** (806 rows, 33 videos):

| Metric | Value |
|---|---|
| Source MPJPE Δ | −16.5 mm [−19.3, −13.7] |
| Rig leg dir Δ | **−0.05° [−0.21, +0.14]** |
| Rig-failure rows M → R11 (new / recovered) | 56 → 57 (6 / 5) |
| Leg-ROM rows M → R11 (new / removed) | 468 → 438 (48 / 78) |

**P_all** (1,223 rows): rig dir 28.7° → 15.2°, Δ −13.5° [−23.5, −5.0]; rig-failure rows 467 → 188.

Strata, P_leg Δrig_dir_ref:

| Stratum | Δrig_dir_ref |
|---|---|
| Restored leg tokens: 1 / 2 / 3 | −23.0 / −45.7 / −47.8° |
| Lowest restored confidence: [0, .1) / [.1, .2) / [.2, .3) | −46.6 / −46.1 / −30.8° |
| Ankle-only / knee or hip present | −38.6 / −40.5° |
| Restored 2D all within 0.10 (post-hoc, GT-referenced) / some beyond | −27.8 / −42.8° |

The confidence and token-count strata are confounded, as in W104.

**Translation.** Source leg-segment change and rig change correlate at r = 0.94. Using a ±5° band:
- 398 of 417 rows improve in both.
- 5 rows get worse in both.
- 1 row improves at source but gets worse in the rig; 4 are unchanged at source but worse in the rig.

## 5. Source vs retargeting attribution

The fixed IK is **not** the dominant limitation for legs:
- GT mapped through the same rig departs from the GT segments by a median of 0.8° and a maximum of 5.7° (834 legs;
  none > 45°).
- W102's large source-to-rig discrepancies come from S0's inconsistent folded-chain geometry (thigh/shin length ratio
  far from the target ratio). R11 reduces rows with discrepancy > 45° from 77 to 5.
- The remaining fixed-length, zero-twist IK limitation is real but small here: 21 of the 151 R11 failing legs are
  category D.

Attribution of rig failures on P_leg:

| Condition | Failing legs | Categories |
|---|---:|---|
| M_Main | 680 | 655 missing-token fold (W104 mechanism); 15 frozen S0 with no missing token on that leg; 10 F |
| R11 | 151 | **99 A raw 2D localization**; 21 D IK; 14 B frozen S0; 9 G unresolved; 8 F GT invalid |

Stable-but-wrong vs jitter, among failing rows:
- M_Main: 305 of 411 have a neighbour step > 45° (jitter-like); 106 are stable-but-wrong.
- R11: 45 of 131 jitter-like; 86 stable-but-wrong.
- So R11's remaining failures are mostly consistently wrong poses, not flicker.

## 6. Temporal boundaries (10 Hz, original same-person neighbours; no interpolation evaluated)

P_leg: 487 pairs, 301 of them at M leg-mask transitions.

| Pairs | Dir-step events M → R11 | Thigh-step events M → R11 |
|---|---|---|
| All 487 | 439 → 53 (Δ/pair −0.79 [−0.91, −0.61]) | 88 → 0 (Δ/pair −0.18 [−0.23, −0.10]) |
| At mask transitions | 426 → 30 | 83 → 0 |
| Elsewhere | 13 → **23** | — |

M_Main's leg discontinuities are concentrated at valid → invalid → valid transitions, where the leg folds and
unfolds. R11 removes most of them. Away from transitions, R11 adds 10 direction events: a small new flicker from
frame-to-frame variation in the restored low-confidence coordinates.

The P_all temporal view agrees: 460 → 75 direction events and 88 → 0 thigh events. No propagation, hold or filter was
implemented. The 10 Hz cadence is not presented as high-frequency motion.

## 7. Harm cases, alternative explanations and transfer limits

All 30 R11 harm rows in PARTIAL were inspected:
- 11 source newly-wrong knee, 1 source newly severe, 18 rig newly-failed legs, 8 with rig Δ > +15° on a leg, plus
  W104's #1018.
- The 18 rig newly-failed legs mostly sit on rows whose other leg was already failing. So P_leg's row-level "new" count
  is 0.

Per-leg attribution: 23 A, 10 F, 4 G, 3 B, 3 D, and 3 B-indirect (an arm restoration that moved a leg). Notable cases:
- **`downtown_cafe_00` actor1 #659/668/685/690 (seated).**
  - Source MPJPE improves 485–531 → 281–330 mm, but the rig gets worse: 47–51° → 59–90°.
  - 6–8 tokens were restored at confidence 0.02–0.25, with knee/ankle 2D errors of 0.08–0.62 (mostly 0.24–0.48) of the limb-box diagonal.
  - **A source MPJPE gain here is a rig regression.** This is the main counter-example to "better MPJPE means better
    animation".
- **`outdoors_crosscountry_00`** (10 legs): GT root behind camera (F). This is a reference defect, not evidence.
- **`parcours_00` #1018** (W104's largest source harm, restored left elbow+wrist): the rig legs are essentially
  unchanged (17.3° → 21.0°). The arm-side misplacement is not expressed by the W102 rig.
- **`downtown_car_00` #789/795**: the restored knees are accurate (2D error 0.02–0.12), yet the rig gets worse
  (B / D).

The distinction W104 drew still holds:
- Restored coordinates are often inaccurate.
- Restoration is better *for S0* than absence, because absence makes S0 fold the limb.
- Low-confidence coordinates are not trustworthy in general; where they are badly wrong, R11 produces a different
  wrong pose.

Not tested: other detectors, other confidence regimes, unseen footage.

## 8. Identity and coverage

The full denominator is unchanged:
- 6,988 same-person, 1,157 other-person, 2,299 unresolved, 30 unmatchable, 7 no-subject, 2 ambiguous.
- Rig evidence covers only the 1,223 uniquely associated same-person PARTIAL rows: 17.5% of same-person rows and
  11.7% of all actor-frames.

The same GT-free rule would also change 818 non-same-person rows (174 other-person, 614 unresolved, 30 unmatchable):
- To keep this visible, R11 for those rows was computed by a fresh S0 forward: an additional, labelled run in a new
  layout.
- Their accuracy is **unmeasured**, because GT belongs to the target actor and not necessarily to the observed person.
- In the `identity_caveat` window, R11 stands up an unresolved-association track that M_Main had folded. This looks
  plausible in the render, but it is not evidence.
- Nothing fills one actor's joints from another actor, and no association is inferred from GT.

## 9. Secondary: S0 left arm through the existing chain (not the W102 arm policy; never in a gate)

489 PARTIAL rows with restored left-arm tokens, using the shared HMR thorax and the existing left-arm two-bone chain:
- Rig arm direction vs GT-through-chain: 58.4° → 37.8°, Δ −20.6° [−26.4, −15.1].
- Arm-failure rows: 445 → 229 (20 new / 236 recovered).

So arm-side restoration would carry into a rig *if* S0 arms were consumed. Under the preserved W102 contract (frozen
J1X arm), they are not.

## 10. Real-video evidence (GT-free; private; not unseen; legs not validated)

The W102 contract cannot run on this clip: there is no HMR parent and no J1X for it. A labelled **S0-only
configuration**, the same for both conditions, was used instead:
- W93 torso frames from each condition's own S0 output as Object/thorax (W95 A_S0).
- S0 legs; S0 left arm through the existing chain; right arm at rest.
- Keys at the original 50 fps frame indices (1–199). The historical renderer's scene rate stays 30 fps, so `.blend`
  playback is 0.6× real time. The review MP4 uses the original timestamps.

Reproduction against W102 and W104:
- M input, geometry and predictions reproduce W102 byte-exactly.
- R11 changes 38 frames (left wrist only); mean joint deviation 52.897 mm, equal to W104.
- Action audit: identical channels, 199 keyed frames, 0 mismatches on the 161 unchanged frames.

| GT-free quantity (38 changed frames) | M_Main | R11 |
|---|---|---|
| Rig left-elbow flexion, median | 93° (complete frames: 12°) | 40° |
| Left-arm rig change vs M | — | mean 116° |
| Parent pelvis change | — | ≤ 2.6° |
| Leg rig change | — | ≤ 7.2° |
| Left-arm steps > 45° (199 frames) | 4, all at the 4 wrist-mask transitions (69–114°) | 4, none at transitions; all inside frames 112–117 (126–137°) |
| Leg steps > 45° | 0 | 0 |
| Leg-ROM frames | 102 | 101 |

Visual judgement from the renders (frames 115, 170 and others):
- **Frames 154–181** (walking away, wrist occluded by the body):
  - M_Main holds a forearm folded ~95° across the body for 0.56 s.
  - R11 gives a moderately bent, varying arm. This is plausibly closer to a walking arm swing, but without GT it is
    not a verified improvement.
- **Frames 109–118** (side-on):
  - The raw left wrist at confidence 0.15–0.16 lies near the hip in most frames.
  - In frames 113, 114 and 116 it is placed at head height (y ≈ 241–247 px vs nose ≈ 197).
  - R11 renders a **hand-to-face flicker**: 80 ms, four steps of 126–137°. This is clearly harmful animation and a
    category-A hallucination.
- Editability: the export is inspectable. There are 32 F-curves per Action, keys only at sampled frame indices, and
  CONSTANT interpolation. This is not a claim of authoring capability.

Real-video verdict: **indeterminate / mixed for the arm. No evidence about low-confidence leg recovery.**

## 11. Representative renders (local, evaluation-only)

Location: `C:\Users\dna10\animcv-w105\results\review\*.mp4|jpg`. Panels: source image | M_Main | R11 |
GT-through-rig (labelled EVALUATION ONLY, from a separate Blender run). Rows: video camera, then side camera.

- **`leg_natural_most` #352**: M_Main crouches with folded knees beside a bar; R11 stands like the GT reference.
- **`w102_max_source_regression` (cafe actor1, seated) #751**: R11 is closer to seated than M's raised leg, but the
  shins still differ from GT. Rows #659–690 in this window are the rig regressions from §7.
- **`w104_largest_harm` #1018**: M and R11 are visually the same (arm restoration is not consumed).
- **`new_wrong_knee` #638** (stepping up): both conditions extend the shin forward instead of folding it back like GT.
  This is a shared failure, not specific to R11.
- **`complete_control`**: identical; Action digests are equal.
- **Real clip**: `review_clip/real_clip.mp4`, with stills at frames 115 and 170. The source image here is the unchanged
  private clip, viewed locally only.

## 12. CASE, negative findings and unresolved uncertainty

**CASE A (declared rule; primary and sensitivity).**

| Gate | Result |
|---|---|
| RIG_BENEFIT | −39.5° [−45.0, −29.9]; failure rate −0.67 [−0.71, −0.56] |
| MATERIAL | net 280 of 411 |
| SPLIT | VAL −21.7°, TEST −41.7° |
| RIG_HARM components | rig-fail 0 vs 70; false-fold 4 vs 72; ROM 1 vs 25.75; p99 58 < 100; temporal CI upper < 0 — none met |
| Contract (D checks) | all pass |

Scope of CASE A: restored hip/knee/ankle tokens on 417 uniquely associated same-person 3DPW rows (20 videos, reused
data), under the W102 mixed-source rig (HMR parent, J1X arm).

Negative findings and limits:
- **Arm restorations: no measurable rig utility under the contract.** Their rig utility is untestable without changing
  the arm policy; the §9 secondary suggests it would transfer.
- **Real clip: mixed.** R11 removes a stable fold but introduces a visible hand-to-face flicker from hallucinated
  coordinates.
- **R11 still leaves 31% of P_leg rows with a failed rig leg**, mostly from bad low-confidence 2D.
- **Source MPJPE gains can be rig regressions** (seated cafe rows).
- **Leg-ROM violations stay common** under R11 (314 of 417 rows). The W95 limits flag many poses that GT itself may
  violate; this is not investigated here.
- Away from mask transitions, R11 adds 10 small temporal direction events.
- Rig evidence is conditional on unique same-person association (66.7% of actor-frames are same-person). The rule's effect on 818
  non-same-person rows is unmeasured.
- VAL gains are about half of TEST's. The population has informed earlier decisions; nothing here is a holdout.
- The selected windows are illustrative. Two of them (leg_natural_most, cafe) overlap rows reported by W104.

Unresolved:
- Whether the low-confidence arm coordinates would help under an S0-consuming arm policy at runtime.
- Whether a temporal check could separate the frame-113 hallucination from the frame-170 occlusion. Only evidence is
  recorded here; nothing was implemented.
- How cross-detector behaviour compares.
- Why S0 folds absent limbs (W104 open question).

## 13. Tests

`tests/test_w105_rig_utility.py`, final run **15 passed** (`results/w105_tests_final.xml`). Covered:
- **Synthetic rig checks:**
  - shared parent and only-legs-consume-S0 invariant; an arm-only change leaves the export identical;
  - unavailable parent → None, never filled;
  - determinism and length ratio;
  - leg metrics: identity, fold, knee-branch flip;
  - pair events require GT-quiet steps.
- **GT isolation:** the GT-free construction functions contain no GT/projection reference; the identity label is not
  consulted.
- **Selection and decision mapping:** restoration partition, deterministic disjoint windows, decision mapping including
  temporal harm, and Action identity detecting a changed unchanged-sample key.
- **Stored-real checks:**
  - R11 byte reproduction and 5,763 no-op identity;
  - per-frame row/frame/timestamp/sequence alignment and shared parent/arm across slots;
  - renderer report, PNG counts per slot/view, Action identity, and keys only at sampled indices;
  - analysis denominators, labels and GT-reference slot isolation.

Earlier run: pre-declaration, 11 passed and 4 stored-real skipped (`w105_tests_predeclare_r1.xml`).

Regression (W105 imports their helpers): `test_w104_missing_joint.py`, `test_w103_observability.py` and
`test_w102_same_environment.py` — **50 passed** (`w102_w104_regression.xml`).

Execution failures, retained and not counted as passes:
- **verify, attempt 1, stopped.** The alignment check compared absolute stored M predictions with `rows.json`'s
  root-relative joints. The check was corrected to root-relative before the declaration, and verify was rerun.
- **build, attempt 1, crashed after the S0 and rig stages.** It passed the window dict instead of its row list. This
  was fixed after the declaration; the fix is a construction bug, not a contract change. The failed outputs are kept
  as `build_attempt1_failed`; its `rig_rows.json` hash equals the final one.

Not executed:
- Broad production regression (no shared contract changed).
- Training, detector, HMR2 or J1X inference.
- Any temporal filling.
- LabServer jobs.
- Unseen-video or contact evaluation: foot-slide and contact proxies are not computable without root translation in
  this contract.

## 14. Reproduction (PowerShell on Main, repo root)

```
$py="C:\Projects\AnimCV\.venv_build_full\Scripts\python.exe"; $bl="C:\Program Files\Blender Foundation\Blender 5.1\blender.exe"
$a="--w102 C:/Users/dna10/animcv-w102 --w101-data C:/Users/dna10/animcv-w101/data --w103 C:/Users/dna10/animcv-w103 --w104 C:/Users/dna10/animcv-w104"
$o="C:/Users/dna10/animcv-w105/results"
$d="--declaration $o/declare/declaration.json --declaration-sha 6238b78f271f56f7c112aa499c07e81c39a68530fcb5b4368fe18665a99afae7"
$r="--rig examples/skinned_mannequin/BaseRig_Skinned.fbx --rest examples/e2e_demo/baserig_blender_rest.json --show-mesh --size 480 --ortho-scale 3.2 --equivalent-rig-tolerance 0.0001"
& $py -X utf8 scripts/w105_rig_utility.py verify  $a.Split() --out-dir $o/verify
& $py -X utf8 scripts/w105_rig_utility.py declare $a.Split() --out-dir $o/declare
& $py -X utf8 scripts/w105_rig_utility.py build   $a.Split() $d.Split() --out-dir $o/build
& $bl -b -t 2 --python-exit-code 1 -P scripts/w105_rig_blender.py -- render $r.Split() --modes FIXED,RUNTIME --cases $o/build/cases.json --out-dir $o/render
& $bl -b -t 2 --python-exit-code 1 -P scripts/w105_rig_blender.py -- render $r.Split() --modes FIXED,RUNTIME --cases $o/build/video_cases.json --out-dir $o/render_clip
& $bl -b -t 2 --python-exit-code 1 -P scripts/w105_rig_blender.py -- audit --render-dir $o/render --out $o/render_audit.json
& $bl -b -t 2 --python-exit-code 1 -P scripts/w105_rig_blender.py -- audit --render-dir $o/render_clip --out $o/render_clip_audit.json
& $py -X utf8 scripts/w105_rig_utility.py analyze $a.Split() $d.Split() --build-dir $o/build --render-report $o/render/blender_report.json --audit-report $o/render_audit.json --out-dir $o/analysis
& $bl -b -t 2 --python-exit-code 1 -P scripts/w105_rig_blender.py -- render $r.Split() --modes RUNTIME --cases $o/analysis/gt_reference_cases.json --out-dir $o/render_gt_reference
& $py -X utf8 scripts/w105_rig_utility.py compose --cases $o/build/cases.json --gt-cases $o/analysis/gt_reference_cases.json --render-dir $o/render --gt-render-dir $o/render_gt_reference --image-root C:/Users/dna10/animcv-w102/data/images --out-dir $o/review
& $py -X utf8 scripts/w105_rig_utility.py compose --cases $o/build/video_cases.json --render-dir $o/render_clip --clip --out-dir $o/review_clip
& $py -X utf8 scripts/w105_rig_utility.py manifest --base C:/Users/dna10/animcv-w105 --starting-head 12311cc31fe51aa6e1de4ddcddaea2c25cbd5cec --declaration-sha 6238b78f271f56f7c112aa499c07e81c39a68530fcb5b4368fe18665a99afae7
& $py -X utf8 -m pytest tests/test_w105_rig_utility.py -q
```

- Each stage refuses an existing output directory.
- Measured runtimes: verify 60 s; declare about 5 min; build 197 s (GPU peak 159 MB); candidate renders 20 s; clip
  render 15 s; analyze 184 s; focused tests 3 s; regression 66 s.

Outputs are local in `C:\Users\dna10\animcv-w105\` (516 MB) and are not committed.
- `evidence_manifest.json`, SHA `cb0eca93c7847a70d497bd36c57856bd8295f3d8e3a7a72913ebf876ebe96899`.
- It covers 207 files (reports, case files, `.blend` exports, review videos/stills, logs including the failed
  attempts, and test XML).
- It also covers 2,020 render PNGs through a combined digest `35daba49…`.

## 15. Changed files, resources and environment

New files only:
- `scripts/w105_rig_utility.py`: verify / declare / build / analyze / compose / manifest.
- `scripts/w105_rig_blender.py`: Blender render wrapper around the unchanged W91 renderer, and the Action audit.
- `tests/test_w105_rig_utility.py`.
- This Worklog.

No `src/`, historical script, renderer, rig asset, checkpoint, dataset, W102–W104 artifact, `docs/next_batch.md` or
automation file changed.

Resources: Main only; GPU about 1 min in total; Blender Workbench about 1 min in total; single-threaded torch; no package
change. The source video was not copied, redistributed or published.

Final source revision: the commit adding this Worklog on `arch/single_frame_first`, the next commit after `12311cc`.
The COMPLETE marker is added in a separate final commit.

AUTOMATION_BATCH_STATUS: COMPLETE
BATCH_NUMBER: 105
