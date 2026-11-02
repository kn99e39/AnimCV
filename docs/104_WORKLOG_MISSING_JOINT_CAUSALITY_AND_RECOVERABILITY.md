# Worklog — Missing-Joint Causality and Recoverability in Frozen S0 (2026-11-02)

## Scope, starting state and principle

Task scope (summarized, not copied): an evidence-only Estimator/Observation diagnostic. It tests whether removing specific
real M1 joints from frozen S0's input **causes** severe and knee-branch failures, how much of that is crop normalization,
and whether naturally missing evidence is recoverable from information a GT-free runtime actually has. No model,
threshold, mask policy, crop rule, rig or production behaviour may change.

- Branch `arch/single_frame_first`, starting HEAD `218335e24e2c0811cfa7b7c890213d0d5d3bdef6`; instruction blob
  `7b1ca87cecdf80518d05be6ec383e3561197abee` verified before any work. `git fetch` showed no divergence from
  `origin/arch/single_frame_first`; the working tree was clean.
- Run unattended by the scheduled automation worker, entirely on **Main** (RTX 5080, driver 596.49, Python 3.11.9,
  Torch 2.13.0+cu130, `C:\Projects\AnimCV\.venv_build_full`). No LabServer63/69 job, no package install, no precision or
  environment change. GPU utilization was 0% before the run and no other compute workload was visible; W104 used the GPU
  for about 2.5 min in total.
- W95–W103 verdicts, numbers and artifacts are unchanged. W103 CASE C and its reference-validity sensitivity stay
  separately identifiable; this Worklog does not rewrite either.

Evaluation regimes, kept apart throughout:
- `real_animcv_observation`: M_Main and the source-only restoration R11.
- `counterfactual_diagnostic`: F10, F01, F11, R10, R01 and Rc. These are not valid production observations.
- `oracle_geometry` privileged upper bound: P_oracle.
- `benchmark_detector_observation` privileged reference: P_bench.

The 3DPW same-person population has been reused for architecture diagnosis many times. It is **not an untouched holdout**.

## Answer

**CASE A under the predeclared rule**, in the primary run and in the reference-validity sensitivity run. Every gate
passed with a wide margin.

Within this diagnostic, removing joint evidence **causally** produces frozen S0's catastrophic failures. Crop
normalization is a secondary contributor except where bilateral leg removal shrinks the crop drastically. The naturally
missing joints are, in this population, **recoverable from information M1 already computes and then discards**: every
naturally invalid token has a finite raw RTMPose coordinate below the 0.3 visibility threshold.

Restoring those coordinates under one fixed, source-only rule improves the naturally partial rows substantially, with
very little collateral harm.

Per the Batch definition, CASE A supports **further consideration** of a bounded observation-handling architecture. It
does **not** authorize adopting a mask rule, threshold change, fallback or any production change. None was made.

Four findings, each kept separate:

1. **Missing tokens cause the failures (input sensitivity).** On 4,803 well-observed rows × 26 removal patterns:
   - Removal at a fixed crop (F10) adds **+84.5 mm** MPJPE [80.1, 89.1]. Crop recomputation alone (F01) adds +10.0 mm
     [9.3, 10.8].
   - Mask main effect **+80.0 mm**, crop main effect +5.6 mm, interaction −9.0 mm.
   - The mechanism is visible directly. Without a knee or ankle, S0 **folds that leg**: median knee flexion goes from about
     20° to **120–140°** (GT ≈ 20°).
   - 36.7% [32.8, 40.4] of correct eligible knees become wrong-branch, versus 0.05% under the count-matched
     mirrored-arm control and 0.8% under crop-only.
   - About 23% of row×pattern removals become newly severe, versus 0.9% for crop-only.
2. **Crop/mask interaction is large only for bilateral leg loss.** For ankles or knees+ankles on both sides, F01
   is +32 / +85 mm and the interaction is −22 / −80 mm. Original joints then fall outside the shrunken crop in 93–99% of
   cases, so F01 is out-of-distribution there and is labelled uninterpretable as an isolated crop effect.
3. **Natural missingness is quantitatively consistent with the controlled removal.** Over the 8 natural pattern classes
   with ≥ 30 rows, F11's failure rate is **0.94×** the natural rows' failure rate. The mean MPJPEs are close too (e.g.
   single wrist: natural 98 mm vs F11 87–96 mm; bilateral ankles: 157 vs 170 mm).
4. **Source-only restoration (R11) recovers most of the damage.** On the 1,223 naturally partial same-person rows:
   - MPJPE **−33.9 mm** [−49.5, −20.6]; severe frames **133 → 24**, with 1 newly severe and 110 recovered.
   - Wrong knee: 174 fixed vs 13 newly wrong. p99 MPJPE 305 → 221 mm. No row is harmed by more than 50 mm.
   - The effect comes from the new coordinates: the restored-crop-only R01 gives −0.4 mm, and fixing restored confidence
     at 0.3 (Rc) gives −0.2 mm relative to R11.
   - The privileged GT-projection upper bound adds only a further −12.3 mm [−17.5, −7.8] over R11.

## 1. Preflight and provenance

All checks are in `verify/verify_report.json` (SHA `d0f7f419…`).

Artifact identities:
- W102 evidence manifest `5186a170…`. Its 94 files match their manifest entries: the 21 W103-pinned files plus all 73
  raw M1 `results/observe/*` archives and metadata.
- W103 manifest `3aad853c…`, 33 files matching, declaration `fda5c08b…`.
- S0 checkpoint `6c12fea9…`; bank content digest `75519e63…`.
- All five W102 geometry tensors reproduce `geometry_sha256` exactly.

Raw M1 provenance:
- The retained selected-subject landmarks (`(16,4)` x, y, confidence, visible per frame; source manifest `6492b3fc…`;
  each npz checked against its own recorded SHA) reproduce `input_2d_M` / `input_valid_M` **exactly on all 10,483
  actor-frames**.
- Image/timestamp alignment is checked per row.

Natural missingness = discarded predictions, not missing observations:
- Same-person rows have 2,193 invalid tokens (2,155 limb). **All 2,193 have a finite raw coordinate**; none is absent.
- Their raw confidences are all < 0.3 (max 0.29992), and valid tokens are all ≥ 0.3 (min 0.30007).
- So every naturally "missing" M1 joint is a below-threshold prediction that the unchanged production policy discards.

Reproduction (`infer/infer_report.json`, SHA `97ee988e…`):
- Same-stack S0 forward of M_Main in W102's batch layout is **byte-equal** to W102's stored predictions on both splits.
- Factorial F00 tensors are byte-equal to the M_Main geometry.
- F00 recomputed in the counterfactual batch layout differs from the stored predictions by at most **0.0007 mm**
  (budget 0.1 mm), with **zero** discrete decision changes (severe / knee / depth / torso / limb).
- The environment fingerprint (`9a032024…`) differs from W102's only in `model_source_sha256`, which is CRLF checkout
  bytes. The LF-normalized source hashes equal W102's, the same finding as W103.

## 2. Declaration (hashed before any W104 S0 outcome)

`declare/declaration.json`, SHA-256 **`6a64c8d1ffc57f928fdf2fbaaf40dffd5b848a927b662e90ff584257bfbae487`**.

- It records `GT_arrays_read: false` and `S0_outputs_read: false`. The infer, analyze, video and posthoc stages refuse
  to run without this hash.
- Pattern construction reads only M validity masks. A test pins that construction functions reference no GT, projection
  or prediction array.
- Substrate membership uses the frozen W103 `quality_M` label. That label is GT-2D-referenced but S0-independent; no
  S0-error column was read.

Populations, per split (VAL / TEST):

| Population | VAL | TEST |
|---|---:|---:|
| Same-person rows | 2,745 | 4,243 |
| Factorial primary substrate: complete M input, W103 WELL | 1,874 | 2,929 |
| Factorial secondary substrate: complete M input, mislocalized | 496 | 466 |
| Natural PARTIAL (≥ 1 invalid limb joint) | 375 | 848 |

Sampling was exhaustive; seed 1337 is used only for the video bootstrap.

**Patterns (27).** Selected from 2D masks only:
- 12 single limb joints.
- 8 same-limb chains: knee+ankle, hip+knee+ankle, elbow+wrist and shoulder+elbow+wrist per side.
- 4 bilateral: ankles, knees+ankles, wrists, elbows+wrists.
- 2 natural extras from the 12 most frequent natural sets: L knee+L ankle+R ankle, and L elbow+L wrist+both
  knees+ankles.
- 1 added mirrored control.

The 8 most frequent natural sets were already covered by the declared list: single wrists (269 each), R elbow+wrist (102),
bilateral ankles (79), L elbow+wrist (60), L ankle (56), R elbow (55) and bilateral knees+ankles (36).

**Controls and derived tokens.**
- Count-matched unrelated control = the same-side leg↔arm mirror; mixed-limb patterns have none.
- Removal applies the production derived-token rule: a hip invalidates pelvis and spine; a shoulder invalidates neck,
  thorax and spine.

**Factorial, per row and pattern** (exact invariants are pinned by tests):

| Condition | Validity | Crop |
|---|---|---|
| F00 | original | original (equals the M_Main tensor) |
| F10 | intervention mask | original |
| F01 | original | recomputed from the intervention validity |
| F11 | intervention mask | production recrop |

- F10 differs from F00 only on the removed tokens, which are zeroed.
- F01 keeps validity and confidence and is an exact affine image of F00.
- F11 equals the production `crop_box` → `geometry_in_crop` path on the modified observation.
- Crop strata:
  - invariant: F01 bytes equal F00;
  - small: side ratio within 1% and centre shift < 1% of side;
  - large: anything else.
- F01 tokens outside [−1, 1] are labelled out-of-range and kept in the denominator.
- No pattern was refused: every declared joint was valid on every complete row.

**Source-only restoration** (natural rows, evaluation-only):
- R11: every M-invalid token with a finite raw landmark becomes valid with its raw coordinate and confidence. Derived
  tokens follow the production midpoint rule. Production recrop.
- R10: the same tokens with the original crop.
- R01: original validity with the restored crop.
- Rc: R11 with restored-token confidence fixed at 0.3, the unchanged boundary value.
- No threshold was searched and no joint was selected by GT or by predicted 3D. M_Main is unchanged.

**Privileged references**, never pooled with real-source rates:
- P_oracle: R11 validity and confidence, with restored base-joint coordinates replaced by the SMPL GT projection. Only
  rows whose GT root is in front of the camera.
- P_bench: M-missing base joints filled with benchmark 2D where it is valid.

**Gates and CASE mapping**, fixed in the declaration:

| Gate | Requirement |
|---|---|
| CAUSAL | (a) F10 induced rate, target minus mirrored control on the target limbs: CI lower > 0; (b) F10 − F01 induced: CI lower > 0; (c) MPJPE(F11 − F00): CI lower ≥ +10 mm |
| CROP_SECONDARY | \|crop main\| and \|interaction\| each ≤ 0.5 × mask main |
| EXPLAINED | F11 / natural failure ratio ≥ 0.5 over natural classes with ≥ 30 rows |
| BENEFIT | R11 − M MPJPE CI upper < 0, and severe-rate change CI upper < 0 |
| HARM | newly severe > 0.25 × recovered, or newly wrong knee > 0.25 × fixed, or p99 worse |

- A requires CAUSAL, CROP_SECONDARY, EXPLAINED, BENEFIT and not HARM.
- B requires the first three and not BENEFIT.
- C covers everything else; D is a reproduction or provenance failure.

**One construction correction before inference.** The first pre-inference test run (`w104_tests_preinfer.xml`) failed 2
of 14 tests:
- `privileged_oracle_row` recomputed derived-token confidence as a minimum, instead of copying R11's confidence verbatim.
  The two are identical on real M1 landmarks, whose midpoint rows already carry the minimum.
- A synthetic knee fixture was below the 20° eligibility bend.

Both were corrected before `infer` ran; the second run passed 14/14. The declaration text and hash did not change.

## 3. Controlled joint removal (factorial)

`analysis/analysis_report.json` (SHA `c7f95ab4…`):
- Primary substrate: 4,803 rows, 34 videos, 124,878 target row×pattern pairs.
- Crop strata among those pairs: invariant 58,512, small 12,093, large 54,273. F01 out-of-range: 19,095.

Pooled = mean over target patterns per row, then video bootstrap (2,000 replicates, seed 1337). Counterfactual copies of
one frame are never treated as independent.

| Primary (WELL), pooled | Mean [95% video CI] |
|---|---:|
| MPJPE F10 − F00 (mask only) | **+84.5 [80.1, 89.1]** |
| MPJPE F01 − F00 (crop only) | +10.0 [9.3, 10.8] |
| MPJPE F11 − F00 (production-consistent) | **+85.6 [82.2, 89.1]** |
| Mask main / crop main / interaction | **+80.0** [76.4, 83.9] / +5.6 [5.0, 6.1] / −9.0 [−10.6, −7.3] |
| Induced failure F10 / F01 / F11 | 0.871 / 0.052 / 0.873 |
| Induced F10 − F01 | +0.819 [0.804, 0.833] |
| Induced F10, target − mirrored control (target limbs) | +0.647 [0.620, 0.670] |

The secondary (mislocalized, 962 rows) substrate behaves the same way: F11 +72.6 mm, mask main +67.1, crop main +5.5,
interaction −5.0. F00 itself had 3 severe rows in the primary substrate.

**The declared "induced failure" is close to definitional for an end joint.** A missing wrist or ankle leaves its own
segment unobserved, so the declared limb failure (segment > 45°) nearly always follows. The post-hoc views below
(`posthoc/posthoc_report.json`, SHA `41dd2aec…`) therefore use only knee branch and SEVERE. They were written after
reading the primary results and are **not** part of any gate.

| Post-hoc, primary | F10 | F01 | F11 | Mirrored arm control (F10) |
|---|---:|---:|---:|---:|
| Correct eligible knee → wrong branch, pure-leg patterns (2,830 rows) | **36.7% [32.8, 40.4]** | 0.8% [0.5, 1.2] | 38.4% [35.2, 41.3] | 0.05% [0.0, 0.15] |
| Newly severe, target patterns | **23.1% [20.5, 25.9]** | 0.9% [0.5, 1.2] | 24.9% [22.6, 27.6] | target − control +1.0 pp [0.8, 1.3] |

**Severity is a whole-body consequence of removal, not limb-specific.** The mirrored "unrelated" removal is itself a
removal: shoulder loss produces as many severe frames as hip loss. So the severe target-minus-control difference is small,
even though removal (F10) versus crop (F01) is very large.

Per pattern (primary; F11 new severe of 4,803; F11 knee newly wrong / both-eligible on the affected legs):

| Pattern | F10 / F01 / F11 ΔMPJPE (mm) | New severe | Knee newly wrong |
|---|---|---:|---:|
| single knee L / R | +38.1 / −0.0 / +38.0; +41.9 / 0.0 / +41.9 | 23 / 96 | 706/1,881; 1,269/2,166 |
| single ankle L / R | +40.5 / +2.5 / +41.9; +47.1 / +1.4 / +46.9 | 25 / 70 | 607/1,883; 705/2,168 |
| single wrist L / R | +14.8 / +0.3 / +15.2; +23.0 / +0.1 / +24.1 | 3 / 33 | — |
| single elbow L / R | +28.4 / +0.3 / +28.3; +27.4 / −0.2 / +27.7 | 3 / 124 | — |
| single hip L / R (pelvis+spine lost) | +99.1 / 0.0 / +99.2; +87.2 / 0.0 / +87.2 | 926 / 582 | 41/1,287; 35/1,635 |
| single shoulder L / R (neck+thorax+spine lost) | +108.5 / +0.1 / +108.5; +108.8 / 0.0 / +109.0 | 1,498 / 1,334 | — |
| hip+knee+ankle L / R | +147.0 / +2.6 / +150.8; +132.0 / +1.5 / +134.2 | 3,595 / 2,625 | 630/1,847; 1,109/1,646 |
| bilateral ankles | +87.4 / +32.0 / +97.9 (interaction −21.5) | 668 | 1,129/4,051 |
| bilateral knees+ankles | +161.3 / +84.8 / +166.1 (interaction −79.9) | 4,389 | 2,597/4,035 |
| natural L knee+ankle+R ankle | +115.4 / +34.6 / +125.7 | 1,963 | 1,549/4,034 |

**Where S0 puts a removed joint** (post-hoc, F10, medians):
- For removed shoulders, elbows, wrists, knees and ankles, it is placed far from the truth: 359–817 mm joint error,
  versus 56–106 mm when observed. Removed hips are an exception at 48–54 mm (17–19 mm observed).
- A removed ankle sits 354–439 mm from the root (GT 876–880 mm).
- With a knee or ankle removed, knee flexion is 119–140° (F00 20–24°, GT 20–21°).
- Removed elbows produce upper arms of 562–625 mm (GT 257 mm).

So frozen S0 does not impute anatomically plausible joints; it **folds** the affected limb. For hips and shoulders the
derived torso tokens are lost too, and the damage spreads to the other joints (+93 to +108 mm on non-removed joints for
hips).

**Strata.** Pooled F11 ΔMPJPE in mm:

| Stratum | Pooled F11 ΔMPJPE |
|---|---|
| Removed joints: 1 / 2 / 3 / 4 / 6 | +55.7 / +65.1 / +142.1 / +130.6 / +194.4 |
| Anatomy: arm / leg / mixed | +70.2 / +91.4 / +194.4 |
| Crop stratum: invariant / small / large | +69.8 / +61.9 / +110.4 |
| Posture: upright / other / waist-bend / lying (2 rows) | +89.5 / +72.2 / +70.1 / +46.2 |
| Affected GT knee flexion: < 20° / 20–60° / ≥ 60° | +101.7 / +93.2 / +81.6 |
| Baseline MPJPE: < 50 / 50–100 / 100–200 mm | +98.6 / +85.1 / +68.8 |
| Split: VAL / TEST | +86.0 / +85.3 |

- The effect does **not** reverse in difficult poses. It is smaller where the baseline is already worse or the knee
  already flexed, consistent with S0's folding prior coinciding with those poses.
- The effect is largest in crop-invariant cases that still have large mask effects, which shows the crop is not
  necessary.
- The heaviest sequence-level clusters are parcours_00, windowShopping_00, basketball_01, parcours_01 and enterShop_00
  (induced share 0.84–0.89). They are the largest WELL sequences; no sequence is spared.

**Temporal events.** 4,304 original-neighbour WELL pairs:
- Under F00: 43 S0 segment events (S0 step > 45° while GT step < 15°) among 29,147 eligible steps, and 0 torso events.
- Under F11, summed over target patterns: 3,043 / 757,822 segment events and 183 torso events. That is +0.017 [0.012,
  0.022] extra segment events per pair-pattern; under F01, +0.0004 [−0.0005, +0.0012].
- Persistent removal mostly produces a stable but wrong (folded) pose rather than flicker.

## 4. Natural missingness and source-only recoverability

**EXPLAINED (declared).** Over the eight natural classes with ≥ 30 rows, the weighted F11 failure count is 837.7 against
889 natural failures, a **ratio of 0.94**. The classes are both single wrists, R elbow, L elbow, L ankle, the elbow+wrist
pairs, bilateral ankles and bilateral knees+ankles.

Natural rows also carry pose and occlusion confounds that the substrate does not. The ratio is an aggregate consistency
check, not a per-row causal identification.

**Restoration, PARTIAL rows** (1,223 rows, 33 videos; 2 extra rows had only derived/head tokens restored):

| Condition vs M | ΔMPJPE [video CI] | Severe M → cond (new / recovered) | Knee newly wrong / fixed | p99 MPJPE |
|---|---:|---:|---:|---:|
| **R11** (raw coords + raw conf, recrop) | **−33.9 [−49.5, −20.6]** | 133 → 24 (1 / 110) | 13 / 174 | 305 → 221 |
| R10 (restored, original crop) | −30.8 [−42.7, −20.4] | 133 → 27 (1 / 107) | 11 / 169 | 305 → 222 |
| R01 (restored crop only) | −0.4 [−1.6, +0.6] | 133 → 137 (10 / 6) | 24 / 3 | 305 → 297 |
| Rc (R11, restored conf = 0.3) | −33.7 [−48.9, −20.5] | 133 → 25 | 12 / 170 | 305 → 221 |

- R11 − Rc (the confidence value of restored tokens) is −0.24 mm [−0.59, +0.08]. The R10/R01 interaction is −2.8 mm
  [−7.1, +0.1]. **The benefit is the newly visible coordinates**, not their confidence or the crop.
- R11 collateral, counted honestly:
  - 89 new limb failures against 760 recovered.
  - 238 new depth-sign errors against 1,016 fixed.
  - Torso flips 7 → 8.
  - 0 rows harmed by more than 50 mm; 282 improved by more than 50 mm.
  - The largest harm is outdoors_parcours_00 #1018, +45.8 mm (85 → 131), where a restored left elbow is misplaced.
- Severe-rate change for R11: −0.089 [−0.173, −0.027]. HARM gate: newly severe 1 vs 0.25 × 110; knee 13 vs 0.25 × 174;
  p99 not worse. **Not met.**

**R11 by stratum** (ΔMPJPE in mm, rows in parentheses):

| Stratum | ΔMPJPE |
|---|---|
| Split: TEST / VAL | −40.4 [−58.5, −21.9] / −19.3 [−25.8, −16.6] |
| Restored limb tokens: 1 / 2 / 3 / ≥ 4 | −14.9 (720) / −37.4 (293) / −62.9 (89) / −117.0 (121) |
| Lowest restored confidence: [0, .1) / [.1, .2) / [.2, .3) | −100.9 (51) / −51.1 (343) / −22.7 (829) |
| Baseline > 200 mm | −103.7, severe 133 → 23 |
| Single restored wrist L / R | −10.3 / −14.3 |
| Single restored ankle L / R | −28.6 / −30.8 |

The lowest-confidence rows are also the rows with the most missing joints, so the confidence and count strata are
confounded.

**Restored 2D accuracy** (evaluation-only, after inference, 2,088 limb tokens, crosscountry excluded):
- Median error 0.127 of the GT limb-box diagonal; PCK@0.10 = 42.5%; 97.9% inside the image.
- By confidence: [.2, .3) PCK 52%; [.1, .2) 22%; [0, .1) 1.5%.
- By joint: knees 75–82%, wrists 37–48%, ankles 16–21%.
- Post-hoc and GT-stratified, so not a runtime rule: rows whose restored tokens are all within 0.10 improve −24.1 mm;
  rows with some restored token beyond 0.10 improve **−39.6 mm**.
- So restored coordinates are often inaccurate, yet still far better for S0 than absence, which makes S0 fold the limb.
- The COCO/SMPL semantic offset is included in these errors.

**Privileged references.** Separate regimes; not deployment evidence.

| Reference | Rows | − M | − R11 | Severe M / R11 / reference |
|---|---:|---:|---:|---:|
| P_oracle (GT projection coords; `oracle_geometry`) | 1,183 | −46.5 [−66.4, −28.7] | **−12.3 [−17.5, −7.8]** | 130 / 22 / 15 |
| P_bench (benchmark 2D fills; `benchmark_detector_observation`) | 448 | −42.2 [−57.2, −27.2] | +8.9 [−5.9, +29.3] | 90 / 15 / 33 |

- Perfect 2D for the restored tokens would recover roughly a further third of R11's gain.
- The benchmark detector is not better than the discarded M1 coordinates on the rows where it can fill.
- Meaningful support: P_oracle needs a valid camera (41 crosscountry PARTIAL rows excluded). P_bench is only defined
  where the benchmark marks the M-missing joint valid.

## 5. Temporal evidence availability (measurement only; nothing filled)

The 2,155 naturally missing limb tokens are on 1,223 rows. Neighbours come from the original bank chain only (10 Hz
sparse cadence), within the same actor-sequence, must be same-person and must have the joint valid in M.

| Within ±k samples | Earlier | Later | Either | Bracketed |
|---|---:|---:|---:|---:|
| 1 | 32.2% | 31.5% | 48.1% | 15.5% |
| 2 | 47.6% | 46.8% | 64.4% | 30.0% |
| 3 | 57.5% | 56.3% | 73.1% | 40.7% |
| 5 | 69.5% | 68.3% | 82.2% | 55.5% |

- Missing-run length: median 5 samples, p90 11.
- Nearest earlier observation: median 0.30 s back, p90 0.81 s.
- 52 bracketed cases have intervening non-same-person rows, which adds association uncertainty.
- GT-free 2D motion between the bracketing observations: median 0.14, p90 0.39 of the body-box diagonal. Many gaps span
  substantial motion.
- Availability is lowest for ankles (50–55% within ±5; 23–30% adjacent) and highest for elbows/wrists (90–97% within ±5).
- Evaluation-only evidence content: the nearest earlier observation, held unchanged, is within 0.10 of the current GT
  projection for 49.1% of tokens. The raw current low-confidence coordinate is within 0.10 for 42.5%. Over 1,433 paired
  tokens, the past observation is closer in 49.3%.
- Neither source dominates. This is potential evidence, not an implemented or validated propagation.

## 6. Real-video consequences (GT-free; no rig render)

Historical W102 clip (199 frames, 50 fps, private, rights undocumented, not unseen):
- The M input, geometry and S0 predictions reproduce **byte-exactly**.
- 161 frames are complete. The other 38 lack only the **left wrist**, with raw confidence 0.15–0.30.

`video/video_report.json` (SHA `967bfeeb…`) and the post-hoc flexion check:

| Condition (frames changed) | Mean deviation from M | Knee flexion, affected knee: M → condition (median) | Frames with flexion change > 45° | Knee-normal flips vs M | S0 segment steps > 45° (M: 4) |
|---|---:|---|---:|---:|---:|
| R11, left wrist restored (38) | 52.9 mm | unchanged | 0 | 0 | 4 |
| F11 single L knee (161) | 75.6 mm | 16.5° → 122.0° | 161 / 161 | 16 | 8 |
| F11 single L ankle (161) | 70.0 mm | 16.5° → 123.8° | 161 / 161 | 32 | 8 |
| F11 single R knee (161) | 85.0 mm | 23.7° → 122.6° | 161 / 161 | 22 | 8 |
| F11 single R ankle (161) | 85.2 mm | 23.7° → 105.6° | 151 / 161 | 85 | 8 |
| F11 bilateral ankles (161) | 134.9 mm | 16.5° / 23.7° → 129.7° / 114.5° | both knees, all frames | 121 | 11 |

- On real walking footage, losing one knee or ankle produces a **folded leg in essentially every frame**. Because the fold
  is temporally stable, discontinuities only double (4 → 8–11).
- That failure would be visible as unusable rig motion: a crouched, bent leg during walking. W102's existing
  `maximum_source_regression` / `wrong_knee` BaseRig renders, reviewed in W103, show the natural counterpart.
- R11 changes the arm-side output by about 53 mm with no knee or discontinuity effect. Without GT, whether that change is
  better is **unknown**.
- No new BaseRig/Blender render was produced. The declaration did not require one, and the source-level folding
  signature is unambiguous. No full-body animation success is claimed.

## 7. Reference-validity sensitivity

`outdoors_crosscountry_00:actor0` is 162 VAL rows with the GT root behind the camera (W103). All 162 are same-person;
none is in the WELL substrate. The sensitivity run removes them everywhere.
- The factorial gates are identical.
- EXPLAINED ratio 0.944.
- R11 on 1,183 PARTIAL rows: −34.2 mm [−49.1, −20.1], severe 130 → 22.
- **CASE A** is unchanged. The primary run is retained and the sensitivity run is reported separately.

## 8. CASE and evidence limits

**CASE A (declared rule; primary and sensitivity).**

| Gate | Value | Result |
|---|---|---|
| CAUSAL | target − control lower 0.620; F10 − F01 lower 0.804; F11 MPJPE lower +82.2 mm | pass |
| CROP_SECONDARY | crop 5.6 and \|interaction\| 9.0 vs mask 80.0 mm | pass |
| EXPLAINED | ratio 0.94 | pass |
| BENEFIT | MPJPE CI upper −20.6; severe CI upper −0.027 | pass |
| HARM | — | not met |

Negative findings and limits:
- **Criterion (a) is partly definitional** for end-joint removal. The non-tautological post-hoc views (knee branch 36.7%
  vs 0.05% control; severe 23% vs 0.9% crop-only) support the same conclusion. Severe failure is **not limb-specific**:
  any comparable removal causes it.
- **Crop matters for bilateral leg loss.** There the F01 condition is out-of-distribution (93–99% out-of-range) and the
  mask/crop interaction is large (−22 to −106 mm). The crop is unnecessary for the effect, but it is not inert.
- **Restoration is evaluated only where it can be scored**: the 1,223 uniquely associated same-person PARTIAL rows.
  - Its effect on other-person, unresolved or ambiguous rows (W102: about one third of actor-frames) is **unmeasured**.
  - Complete same-person rows are unaffected by construction (5,763 no-op rows).
  - The real clip shows an output change but cannot score it.
- **VAL shows about half of TEST's gain** (−19 vs −40 mm). Both intervals exclude zero, but the population is 33 videos
  and has informed earlier decisions. TEST is not a holdout.
- **Not all restored coordinates are good.** 57.5% are outside 0.10 of GT, and ankles are the worst (16–21% within).
  The benefit is relative to S0's folding behaviour when a token is absent; it is not evidence that the restored 2D is
  accurate. Whether S0 would behave the same with other detectors, other clips or other confidence regimes is not tested.
- The natural/controlled consistency (0.94) is aggregate. Natural rows have correlated occlusion and pose difficulty.
- **Temporal availability is potential evidence only.** It is measured at 10 Hz bank cadence with the W102
  association proxy, and gaps often span large motion.
- **Unresolved:**
  - Why S0 folds absent limbs: training-time missingness priors versus architecture. Not investigated; no training was
    done.
  - Whether rig-level animation quality follows the source-level improvement.
  - Whether R11's arm change on the real clip is an improvement.

## 9. Tests

`tests/test_w104_missing_joint.py`, final run **16 passed** (`results/w104_tests_final.xml`). Covered:
- Derived-token closure.
- The four factorial conditions with exact invariants (F00 = production; F10 removed-only; F01 affine with unchanged
  validity/confidence; F11 = production path).
- No-op and crop-invariant equality; degenerate and non-applicable refusal.
- Absence of GT, projection or prediction references in construction code.
- Deterministic, outcome-free pattern and control derivation.
- The restoration rule (finite raw only; originally valid tokens bit-identical; Rc confidence; R10/R01 isolation).
- Privileged oracle isolation and regime labels.
- Knee/depth/severe/limb metrics; per-row pooling and video bootstrap (no CI below 2 videos); the decision mapping;
  GT-stable temporal events.
- Stored-real checks: landmark→M reproduction on all 10,483 rows, retention of finite low-confidence raw tokens,
  same-sequence neighbour isolation, the stored reproduction gate and crop-invariant F01 = F00, and analysis labels and
  denominators including the 162 invalid-root rows.

Earlier runs:
- Pre-inference run 1: 12 passed, 2 failed (test-side fixture plus the oracle confidence contract; see §2), 2 skipped.
- Pre-inference run 2: 14 passed, 2 skipped (result-dependent).

Regression: `tests/test_w103_observability.py` **18 passed** (the W104 scripts import its helpers).

Not executed:
- Broad production regression (no shared contract changed).
- Rig/Blender rendering.
- Training, detector or HMR2 inference.
- Any temporal filling.
- LabServer jobs.

## 10. Reproduction

PowerShell on Main, from the repo root:

```
$py="C:\Projects\AnimCV\.venv_build_full\Scripts\python.exe"
$a="--w102 C:/Users/dna10/animcv-w102 --w101-data C:/Users/dna10/animcv-w101/data --w103 C:/Users/dna10/animcv-w103"
$o="C:/Users/dna10/animcv-w104/results"
$d="--declaration $o/declare/declaration.json --declaration-sha 6a64c8d1ffc57f928fdf2fbaaf40dffd5b848a927b662e90ff584257bfbae487"
& $py -X utf8 scripts/w104_missing_joint.py verify  $a.Split() --out-dir $o/verify
& $py -X utf8 scripts/w104_missing_joint.py declare $a.Split() --out-dir $o/declare
& $py -X utf8 scripts/w104_missing_joint.py infer   $a.Split() $d.Split() --out-dir $o/infer
& $py -X utf8 scripts/w104_missing_joint.py analyze $a.Split() $d.Split() --infer-dir $o/infer --out-dir $o/analysis
& $py -X utf8 scripts/w104_missing_joint.py video   $a.Split() $d.Split() --out-dir $o/video
& $py -X utf8 scripts/w104_missing_joint.py posthoc $a.Split() $d.Split() --infer-dir $o/infer --video-dir $o/video --out-dir $o/posthoc
& $py -X utf8 -m pytest tests/test_w104_missing_joint.py -q
```

- Each stage refuses an existing output directory.
- Measured runtimes: verify 61 s; infer 77 s (GPU peak 153 MB allocated); analyze 76 s; video 2 s; focused tests 58 s.
  declare and posthoc were not timed; each finished in under about 2 min.
- Outputs are local in `C:\Users\dna10\animcv-w104\` and not committed. The manifest `evidence_manifest.json` (SHA
  `a6fa3874…`) covers all 22 result files: declare, verify, infer, analysis, video and posthoc outputs, stage logs and
  test XML reports.
- The qualitative figure is `analysis/qualitative_selections.png`, with the 4 declared selections:
  1. Largest F10 single leg: rangeOfMotions_01 actor1 #456, R hip, 74 → 347 mm.
  2. Largest F01: the same frame, L ankle, F01 181 / F11 253 mm.
  3. Largest R11 improvement: downtown_cafe_00 actor1 #715, 355 → 104 mm. This is W103's selection 5, where M put the
     legs on the chair.
  4. Largest R11 harm: parcours_00 #1018, 85 → 131 mm.
- GT panels are labelled evaluation-only.

## 11. Changed files, resources and environment

New files only:
- `scripts/w104_missing_joint.py`: verify / declare / infer stages, construction, and the CLI.
- `scripts/w104_analysis.py`: analyze / video / posthoc stages.
- `tests/test_w104_missing_joint.py`.
- This Worklog.

No `src/`, historical script, checkpoint, dataset, W102/W103 artifact, `docs/next_batch.md` or automation file changed.

Resources and environment: Main only; GPU about 2.5 min total and at most 153 MB allocated; CPU single-threaded torch;
no package changed.

Final source revision: the commit that adds this Worklog on `arch/single_frame_first`, the next commit after
`218335e`. The COMPLETE marker is added in a separate final commit.
