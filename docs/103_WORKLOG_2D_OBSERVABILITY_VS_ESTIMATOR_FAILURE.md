# Worklog — 2D Observability Ceiling vs Frozen Estimator Failure (2026-11-02)

## Starting state and principle

Branch `arch/single_frame_first`, starting HEAD `4ec61907bb84efc531a0a030ba3fd596e356e726` (the W102 publication
commit). `git fetch` showed no divergence from `origin/arch/single_frame_first`. The working tree was clean except for a
pre-existing untracked `third_party/mmpose/`, which this batch did not touch. No branch operation, production `src/`
change, model weight, detector/confidence threshold, mask, crop, normalization, smoothing, IK or rig change was made.
W95–W102 verdicts, numbers and artifacts are unchanged and accepted within their stated scope.

All work ran on **Main** (this Windows machine: RTX 5080, GPU UUID `GPU-0f0604b6-…`, the W102 environment). No
LabServer63/69 job ran. The GPU was used only for a seconds-long S0 forward reproduction; nothing else ran on it.

This batch investigates **what information frozen S0 receives and whether that information distinguishes the 3D
configurations it gets wrong**. It does not optimize anything, and its diagnostic neighbor-GT probe is a
reference-population measurement, not an estimator candidate.

## Answer

**CASE C — mixed mechanisms, with sharply different mechanisms by failure type.** The declared rule (item 13 below)
returns C for the primary M source, for the secondary B source, and for the post-hoc reference-validity sensitivity
run. Both ambiguity-supported and model-specific-supported failures have meaningful evidence (video-CI lower bounds ≥ 0.10).
CASE A's additional controlled-pair requirement is **not** met.

Mechanisms by failure type (M = `real_animcv_observation`, the operational source; B behaves the same way):

1. **Severe reconstructions (MPJPE > 200 mm) and wrong knee branches are input-missingness failures, not 2D ambiguity
   among well-observed inputs.**
   - On well-observed M inputs (all 12 limb joints valid and within 0.10 of the GT limb-box diagonal; 4,803 same-person
     rows), S0 has **3 severe frames (0.06%)** and **1 wrong knee branch out of 4,051 eligible**.
   - 133 of 150 M severe frames (B: 346/354) have at least one invalid limb joint.
   - 185 of 199 M wrong knee branches (B: 281/291) have an invalid joint **in the failing leg**.
   - Wrong-branch rate is 0.19–0.29% with the whole leg valid, versus 40–55% with a leg joint invalid (post-hoc,
     descriptive).
   - The severe rate rises monotonically with the number of invalid limb joints: M 0.3% → 2.5% → 5.5% → 7.9% →
     52% → 100% for 0 to ≥5 invalid joints.
2. **On well-observed inputs, the remaining important error is limb depth sign** (10.9% of eligible segments,
   2,237 errors).
   - Of the 450 that have comparable cross-video 2D evidence, **67% are ambiguity-supported**
     [60%, 73%] and **33% model-specific-supported** [27%, 40%].
   - **80% (1,787) remain unresolved**: mostly no comparable 2D neighbor in the population.
3. **At the full-body level, near-identical model-visible 2D almost never coexists with materially different 3D once
   observation error is controlled.**
   - Among cross-video pairs whose GT projections also match (D_proj ≤ τ_N), only 2/11,342 (B) and 1/9,568 (M) are
     materially different.
   - After excluding one sequence with an invalid camera reference, the strict tier (τ_S) has **zero** materially
     different pairs.
   - Material 3D differences rise steeply with model-visible 2D distance. In the reference-valid run, across the
     declared bins: 0% → 0.4–0.9% → 12–20% → 48–49% → 85% (B and M; primary-run values in §4).
   - Coverage is limited: only 6–12% of queries have any cross-video full-body neighbor within τ_N, and these are
     dominated by upright, nearly straight-legged poses. **No population-wide information ceiling is claimed.**
4. **W102's M1 advantage is joint availability, not better shared coordinates.**
   - With identical B/M masks (4,083 rows), M−B MPJPE is **+0.28 mm [−1.44, +1.81]**.
   - Where M adds joints B lacked (2,457 rows), it is **−46.5 mm [−51.8, −42.0]**.
   - Mmask's harm persists when the crop is essentially unchanged: removing M-only joints costs **+35.9 mm [+32.9, +39.2]**
     on 1,067 rows with side ratio within 1% and centre shift < 1% of the side. Crop change is therefore not necessary
     for the effect (correlational; the joint sets differ between strata).

Answer to the completion question: the important residual errors of frozen S0 are **mostly supported by insufficient
information in its model-visible input, specifically missing joints**, rather than by projective ambiguity among
complete observations. For the residual depth-sign errors on complete inputs, ambiguity and mapping limitation both have
measured support and cannot yet be separated for most cases.

## 1. Evidence and population audit

All checks are in `verify/verify_report.json`.

- W102 evidence manifest SHA `5186a170…` (pinned). All 21 W102 files consumed here match their manifest entries: five
  conditions × two splits of predictions, `observations.npz` `7fc15fc8…`, `observations_side.npz` `d0e3a879…`,
  `s0_report.json`, `evaluation/report.json`, `rows.json`, HMR2 `fc5d3305…`, population `533895c9…` and the
  image/raw manifests.
- S0 checkpoint `6c12fea9…`; bank content digest `75519e63…`; all 36 raw 3DPW pickles match W101's manifest. Every
  RGB image used in a figure was hashed against W101's image manifest.
- Raw GT: all 10,483 actor-frames have `campose_valid` and finite references, and the raw root-relative GT equals the bank
  `target_3d` to ≤ 1e-6 m on every row (W102 row contract).
- **W102 metric parity:** recomputed same-person MPJPE means equal W102's exactly (B, M and HMR2, both splits). The
  severe / knee-wrong-L/R / torso-flip overlap tables equal W102's `failure_overlap_same_person` exactly.

Populations (VAL / TEST, kept separate throughout):

| Quantity | VAL | TEST |
|---|---:|---:|
| Actor-frames / videos / actor-sequences | 3,407 / 12 / 16 | 7,076 / 24 / 37 |
| W102 uniquely associated same-person rows (videos) | 2,745 (12) | 4,243 (23) |
| Other-person / unresolved / ambiguous / unmatchable / no subject | 275 / 385 / 2 / 0 / 0 | 882 / 1,914 / 0 / 30 / 7 |
| Historical all-target-visible same-person | 1,654 | 2,392 |
| M same-person: well / partial / mislocalized | 1,874 / 375 / 496 | 2,929 / 848 / 466 |
| B same-person: well / partial / mislocalized | 1,724 / 838 / 183 | 2,657 / 1,501 / 85 |
| Same-person severe, B / M | 116 / 35 | 238 / 115 |
| Posture (same-person) upright / other / waist-bend / lying | 1,703 / 861 / 181 / 0 | 3,508 / 533 / 199 / 3 |

Pools:
- **B**: all 10,483 actor-frames with the benchmark input.
- **M**: the 6,988 same-person rows with the M1 input. Non-same-person M rows are association failures (W102) and never
  enter as references, because their GT belongs to another actor.

No row was removed on the basis of S0 error. TEST remains an architecture diagnostic population, not an untouched
holdout. Model inference was not rerun except for the reproduction check in §2.

## 2. Model-visible information audit

What frozen S0 receives, established from code and verified on data:

- **Per row, one `(17,4)` float32 tensor**: crop-normalized x and y in [−1, 1], confidence, and validity. Invalid tokens
  are exactly zero. A constant neutral sign state (all UNKNOWN) is also passed. There are **no image tokens**: no RGB,
  silhouette, lighting, clothing, camera intrinsics, image size, crop location or crop scale.
- **The crop removes translation and scale.** It is a square box centred on the valid-joint bounding box, with
  side = 1.5 × the larger extent. Absolute body size and image position are therefore unobservable to S0. Off-axis
  perspective viewpoint is also invisible, because crop location is discarded.
- **Pelvis, neck, thorax and spine tokens are derived midpoints** of hips and shoulders, matching to ≤ 1.2e-7 for both
  sources, and head = nose. The 17 tokens therefore carry at most **13 independent points**.
- **Single-frame.** `FramePoseEstimator.forward` attends over the 17-joint axis only and receives no neighbor or sequence
  input. Empirically, batch-one forwards of 218 rows match the batch-512 outputs within ≤ 0.0013 mm. They are not
  bitwise equal, which is GPU kernel-shape numerics.
- **Reproduction:**
  - All five W102 geometry tensors (B, M, Moa, Mconf, Mmask) reproduce W102's `geometry_sha256` exactly.
  - Same-stack S0 forwards of B_Main and M_Main reproduce W102's stored predictions **byte-for-byte** on both splits.
  - The environment fingerprint differs from W102's only in `model_source_sha256`. The cause is the CRLF checkout of
    this machine (`core.autocrlf=true`); the LF-normalized source hashes equal W102's snapshot and `72e0b6a`.
  - The first reproduce report recorded the mismatch unexplained. It is preserved, and the second report records the
    line-ending audit.
- **Probe composition:** Mmask uses M's coordinates and confidences with benchmark validity (where finite). Mconf uses
  M's validity. On the 4,083 same-person rows where M and Mmask inputs are identical, the stored predictions are
  byte-equal.

## 3. Declared similarity contract (hashed before any 3D disagreement)

`declare/declaration.json`, SHA-256 **`fda5c08bfc1c64fcb4cc476bf7f1d21cfe095322c479db588d3dfec372e0038b`**. It was
computed from 2D tensors only (`GT_arrays_read: false`; a test pins that the declare path takes no target, raw-reference
or projection input). The analysis stage refuses to run without this hash and re-verifies the geometry SHAs. Summary:

- **A. Strict full-body similarity.**
  - Identical 17-joint validity masks, with both shoulders and both hips valid.
  - D_full = RMS crop-unit distance over the mask's valid base points.
  - Confidence is reported, not gated: W102 Mconf showed replacing confidence changes S0 MPJPE by +0.23/+0.85 mm, and
    this batch measured +0.61 [−0.30, +1.71].
- **B. Local limb similarity.** Per limb, four crop-unit vectors: mid−root, end−mid, contralateral root and ipsilateral
  opposite girdle point. All five points must be valid in both rows. A local match is never claimed as full-body
  indistinguishability.
- **Resolution, from observed detector uncertainty.** "Same moment, same person, two detectors (B vs M)" on the 4,083
  rows with identical masks.
  - Full: τ_S = p10 = **0.01446**, τ_N = median = **0.02392** crop units. That is ≈ 0.7% / 1.2% of the crop side; the
    distribution's p90 is 0.050.
  - Local (pooled over 24,248 limb instances; per-limb medians 0.026–0.029): τ_S = **0.01412**, τ_N = **0.02706**.
  - Fixed descriptive bins: [0, .5), [.5, 1), [1, 2), [2, 4), ≥ 4 × τ_N. No other threshold was evaluated.
- **Oracle-geometry control.** GT 24-joint camera projections, with the 13 corresponding SMPL points and the identical
  crop formula, give D_proj with the same τ (regime `oracle_geometry`).
- **3D comparison (evaluation-only).**
  - GT is rotated so the GT root ray is +Y (camera control for the discarded crop location).
  - Population-mean template bone lengths give the template MPJPE (body-scale control).
  - **Materially different** = any limb direction > 45°, a knee-branch disagreement (both flexions ≥ 20° and plane
    normals > 90°), a limb depth-sign flip (both |dy| ≥ 0.25|segment|, opposite signs), or torso forward > 90°.
  - Near-straight knees and degenerate torsos are counted separately and never assigned a branch.
- **Observation quality (independent of S0).**
  - WELL = all 12 limb joints valid and each within 0.10 of the GT limb-box diagonal. The COCO/SMPL semantic mismatch
    remains inside this tolerance.
  - PARTIAL = fewer than 12 limb joints valid.
  - MISLOCALIZED = all 12 valid, but at least one joint exceeds 0.10.
- **Outcome.**
  - SEVERE = MPJPE > 200 mm.
  - ACCURATE = ≤ 100 mm with no limb direction error > 45°, no wrong knee branch and no torso flip.
- **Attribution of a failure at query q.** N_q = cross-video neighbors within τ_N, nearest row per other
  actor-sequence, k ≤ 10. Full tier for severe failures; local tier of the failing limb for knee and depth failures.
  - AMBIGUITY-SUPPORTED: some neighbor's GT is materially different, or on the opposite side of the failing decision,
    *and* matches S0's output better than GT_q does.
  - MODEL-SPECIFIC-SUPPORTED: ≥ 2 neighbors from ≥ 2 videos all agree with GT_q (full tier: median neighbor-to-GT_q
    MPJPE ≤ 0.5 × S0 error), and no neighbor supports ambiguity.
  - MIXED = both conditions hold; UNRESOLVED otherwise.
  - Inputs that are not well observed are reported as missingness or mislocalization and are **not** attributed to
    the mapping.
- **Decision rule (item 13).** Meaningful evidence = video-CI lower bound ≥ 0.10 of the mechanism's share among
  attributed failures. The pure-category reading was fixed in code before any result was produced; mixed counts only in
  the denominator.
  - Ambiguity only → A, which also requires materially different controlled pairs (both rows well observed and
    D_proj ≤ τ_N) in ≥ 3 independent video pairs.
  - Model only → B.
  - Both → C.
  - Neither → C (unresolved).
  - Identity failure → D.

Before any 3D was computed, the 2D pair counts were:

| Source | Same-mask cross-video eligible pairs | Within τ_S (video pairs) | Within τ_N (video pairs) |
|---|---:|---:|---:|
| B | 20,017,568 | 316 (49) | 13,578 (339) |
| M | 16,094,013 | 886 (59) | 16,804 (306) |

Two consequences of the rules should be read with the results:
- MIXED cannot occur for local decisions, because "all neighbors agree" excludes "one disagrees".
- Model-specific support needs ≥ 2 comparable neighbors, so sparsely matched postures fall into UNRESOLVED rather than
  into either mechanism.

## 4. Observability: 2D similarity and 3D diversity

Statistics: cross-video only, video-clustered. "Material" uses the §3 definition.

| Tier (τ_N unless noted) | Cross-video pairs | Video pairs | Material pairs | Projection-consistent share | Material among projection-consistent | Controlled (both WO + proj-consistent): pairs / material |
|---|---:|---:|---:|---:|---:|---:|
| B full, τ_S | 316 | 49 | 1 | 98.7% | 0 / 312 | 312 / 0 |
| B full | 13,578 | 339 | 186 (1.37%) | 83.5% | 2 / 11,342 | 11,342 / 2 (2 video pairs) |
| M full, τ_S | 886 | 59 | 5 | 92.1% | 0 / 816 | 816 / 0 |
| M full | 16,804 | 306 | 241 (1.43%) | 56.9% | 1 / 9,568 | 9,568 / 1 (1 video pair) |
| Oracle geometry, τ_S | 3,230 | 101 | 0 | 100% | 0 | — |
| Oracle geometry | 44,986 | 432 | 71 (0.16%) | 100% | 71 | — |

- **Material differences among 2D-matched pairs are concentrated in projection-inconsistent pairs.** These are pairs
  whose detector inputs coincide while their GT projections do not: B 184/2,236 and M 240/7,236. That is observation
  error (two different true projections collapsed by detector localization), not projective ambiguity.
- In the oracle-geometry tier, the 71 material pairs are depth-sign flips (forearm 45, thigh 25, shin 7, upper arm 1)
  and 10 limb-direction differences. There are **no knee-branch or torso flips**.
- For 44 of these depth-flip pair instances, S0 (B source) has exactly one member wrong in 38, and **frozen HMR2 has
  exactly one member wrong in all 44**. These rare GT-projection-identical forearm/thigh depth pairs are therefore not
  resolved by the contextual RGB model either. GT depth error in either member is not excluded.
- **Full-body knee-branch ambiguity cannot be assessed at the full-body tier.** In every matched set, more than 99% of
  knees are near-straight (< 20°); for example, M at τ_N has 55/39 eligible L/R knees and zero disagreements.
- Torso forward flips among M τ_N pairs (160) are threshold straddles, not front/back ambiguity: the angle has p99
  88° and max 108°, and **both S0 and HMR2 are correct on both members in 160/160**.
- Matched pairs keep S0 outputs close (M τ_N: median S0 prediction difference 33.7 mm, versus 49.6 mm median template
  MPJPE between the GTs). So S0 treats them as nearly the same input, as expected for a deterministic per-row function.

Query-level rates (video bootstrap):

| Pool | Split | Query has a cross-video neighbor ≤ τ_N | …and any such neighbor is materially different |
|---|---|---:|---:|
| M | VAL | 11.0% [6.7, 17.9] | 23.9% [6.9, 45.4] |
| M | TEST | 6.3% [4.8, 7.9] | 20.0% [8.5, 33.5] |
| B | VAL / TEST | 11.3% / 7.0% | 12.3% / 14.3% |
| Oracle | VAL / TEST | 13.8% / 10.6% | 1.3% / 3.7% |

Binned curve, materially-different rate by nearest-sequence distance (all queries × 10 nearest sequences):

| Bin (× τ_N) | M rate [CI] | B rate | Oracle rate | M median template MPJPE |
|---|---:|---:|---:|---:|
| < 0.5 | 0% (153) | 0% (36) | 0% (527) | 31.5 mm |
| 0.5–1 | 4.1% [1.9, 6.6] | 3.0% | 0.5% | 49.7 mm |
| 1–2 | 23.5% [17.8, 29.5] | 14.0% | 13.0% | 76.6 mm |
| 2–4 | 51.2% [45.6, 57.3] | 51.1% | 44.2% | 107.4 mm |
| ≥ 4 | 86.1% [81.6, 89.7] | 85.5% | 85.9% | 180.1 mm |

Within the measured population, the model-visible 2D representation carries strong information about these material 3D
decisions, and the oracle projection curve has the same shape. "No different 3D among near-identical 2D" is established
only where near-identical neighbors exist; most queries have none.

### Reference-validity defect found after the primary run (sensitivity, not a redefinition)

All 162 rows of **`outdoors_crosscountry_00:actor0`** (VAL) have the GT root *behind* the camera (canonical depth
−0.37 m), with root projections far outside the image (e.g. u = 5,097, v = −16,223 px), although `campose_valid` is
true. No other sequence is affected.
- The quality gate catches these rows (all are mislocalized or partial), so none is an attributed query.
- They enter as **references**, and the ray-alignment control for them is meaningless (a 96° rotation).
- The camera-frame root-relative GT pose looks visually consistent with the image (`qualitative/analysis_r3__1_*.png`);
  the evident defect is the root position.

Sensitivity run `analysis_r3_rootvalid` removes rows whose GT root depth ≤ 0. This is a physical validity rule, not an
outcome threshold, and the primary result is retained.
- M full τ_N material pairs drop 241 → **34** / 16,597; B drops 186 → 68.
- The strict tier has **0 material pairs** for both sources.
- 66 of M's 302 ambiguity-supported attributions had a crosscountry neighbor. After exclusion, M has 269 ambiguity /
  156 model-specific: shares 63% [56, 70] / 37% [31, 44].
- **The decision is unchanged (C).** Declared selection 1 (strict pair with different GT) involved this sequence; it
  has no replacement in the sensitivity run, because no qualifying pair exists.

## 5. Estimator error contrast and S0 residual attribution

Residual S0 error by independent observation quality (same-person rows; video-clustered CIs):

| Source / quality | Rows | MPJPE mm | Severe rate | Accurate rate | Wrong knee / eligible | Depth-sign err / eligible | HMR2 MPJPE same rows |
|---|---:|---:|---:|---:|---:|---:|---:|
| M well | 4,803 | 71.9 [66.5, 77.4] | 0.06% | 76% [70, 81] | 1 / 4,051 | 2,237 / 20,617 (10.9%) | 63.5 |
| M partial | 1,223 | 132.7 [116.2, 148.8] | 11% [4, 19] | 3% | 190 / 1,233 | 1,760 / 5,220 (33.7%) | 76.4 |
| M mislocalized | 962 | 94.0 [88.5, 99.4] | 1% | 41% | 8 / 1,052 | 611 / 4,328 | 69.0 |
| B well | 4,381 | 77.3 [71.1, 83.1] | 0.1% | 70% | 6 / 3,999 | 2,139 / 18,708 | — |
| B partial | 2,339 | 142.3 [133.7, 151.5] | 15% [11, 19] | 3% | 284 / 2,238 | 3,327 / 10,294 | — |

Groups (S0 on M; GT 2D measurements decide observation quality, never S0 plausibility):

| Group | Rows (videos) | M MPJPE p50 | B p50 | HMR2 p50 | Notes |
|---|---:|---:|---:|---:|---|
| A well + accurate | 3,629 (34) | 58.0 | 65.8 | 58.5 | HMR2 − M +0.74 [−2.45, +3.96]; 544 rows have a full-tier neighbor |
| B well + severe | 3 (2) | 205.3 | 314.4 | 68.9 | all posture "other"; no full-tier neighbor → unresolved |
| C partial + accurate | 42 (12) | 87.5 | 136.0 | 60.9 | |
| D partial + severe | 133 (15) | 229.4 | 201.4 | 86.3 | 89% of all M severe |
| E B and M both severe | 81 (15) | 227.4 | 237.5 | 95.1 | |
| F M severe, B not | 69 (12) | 229.4 | 171.7 | 78.1 | B has 2.1 extra valid base joints on average |
| F strict (B accurate) | 6 (2) | 236.3 | 62.3 | 51.1 | B has 4.2 extra joints; median model-visible B/M distance 0.46 |

Attribution of important failures (M source, same-person; all VAL+TEST, then by split):

| Failure kind | Instances | Missingness | Mislocalization | Ambiguity | Model-specific | Unresolved (no neighbor / inconclusive) |
|---|---:|---:|---:|---:|---:|---:|
| Severe | 150 | 133 | 14 | 0 | 0 | 3 / 0 |
| Wrong knee branch | 199 | 190 | 8 | 0 | 0 | 1 / 0 |
| Depth sign, legs | 1,599 | 539 | 227 | 171 | 80 | 404 / 178 |
| Depth sign, arms | 3,009 | 1,221 | 384 | 131 | 68 | 1,070 / 135 |
| **All** | 4,957 | 2,083 | 633 | **302** | **148** | 1,478 / 313 |

Shares of attributed failures, with video-cluster 95% CIs:

| Population | Ambiguity | Model-specific | CASE rule |
|---|---|---|---|
| M all | 67.1% [60.1, 73.2] | 32.9% [26.8, 39.9] | C |
| M VAL / TEST | 84% [75, 93] / 62% [52, 69] | 16% [7, 25] / 38% [31, 48] | — |
| B all | 60.6% [51.6, 68.7] | 39.4% [31.3, 48.4] | C |
| M sensitivity (reference-valid) | 63.3% [55.6, 69.5] | 36.7% [30.5, 44.4] | C |

Reading these numbers:
- VAL alone shows weaker model-specific evidence (lower bound 0.07). TEST shows both mechanisms.
- Arm depth errors are 66% ambiguity-supported [53, 81]; leg depth errors are 68% [54, 78] (M source).
- The **controlled CASE A requirement fails**: only 1 (M) / 2 (B) materially different controlled pairs exist, from
  1 / 2 video pairs.
- The S0 failures on well-observed inputs are almost entirely depth sign. For those, the declared evidence supports
  both observational ambiguity (about two-thirds of the attributable cases) and a mapping limitation (about one-third).
  The majority stay unresolved, because comparable cross-video observations are absent.

Alternative explanations that the controls do not remove:
- 3DPW GT depth/fit error.
- The 0.10 PCK tolerance hiding small but decision-relevant localization errors.
- Repeated physical actors across 3DPW sequences: sequences are clustered, but people are not identified across videos.
- Neighbor scarcity biasing which cases can be attributed at all.

## 6. Controlled source attribution (stored W102 probes only; no new intervention)

| B → M stratum (same-person) | Rows | M − B MPJPE [video CI] | B partial rows |
|---|---:|---:|---:|
| Identical 17-joint masks | 4,083 | **+0.28 [−1.44, +1.81]** | 231 |
| M adds joints only | 2,457 | **−46.54 [−51.79, −41.97]** | 1,878 |
| B has extra joints only | 298 | **+46.92 [+30.26, +60.98]** | 99 |
| Both differ | 150 | −4.2 [−21.6, +11.8] | 131 |

Within identical masks, M − B grows with the model-visible coordinate distance:
- < 1 τ_N: −1.1 [−2.6, +0.3]
- 1–2 τ_N: +0.6
- 2–4 τ_N: +4.0 [+0.8, +7.1]
- ≥ 4 τ_N: +13.8 [+4.6, +18.8] (61 rows)

Different coordinates on jointly valid joints do not explain W102's net gain; the availability of additional joints
does.

| Mmask − M stratum (same-person) | Rows | Mmask − M MPJPE | Severe M → Mmask |
|---|---:|---:|---:|
| Identical inputs | 4,083 | 0.00 (byte-equal predictions) | 20 → 20 |
| Removed M-only joints only | 2,457 | **+48.5 [+44.3, +53.0]** | 40 → 294 |
| … with crop side within 1% and centre shift < 1% (post-hoc) | 1,067 | **+35.9 [+32.9, +39.2]** | |
| … with crop changed (post-hoc) | 1,390 | +58.2 [+52.0, +64.2] | |
| Added benchmark-valid finite low-confidence joints only | 298 | **−39.2 [−50.4, −24.6]** | 61 → 25 |
| Both | 150 | +10.6 [−5.2, +29.2] | 29 → 37 |
| Overall | 6,988 | +15.61 [+11.70, +19.63] | |

- Mconf − M is +0.61 [−0.30, +1.71] (confidence alone, as in W102).
- The joints removed in the crop-unchanged stratum are mostly wrists and elbows (1,298 of 1,614 removals), with fewer
  ankles, knees, head and shoulders.
- **Joint availability is the dominant contributor; crop change adds to it but is not necessary.** Adding even finite
  low-confidence coordinates that M1's own 0.3 rule had marked invalid *reduces* error on the 298 rows where that occurs.
- These strata differ in which joints change, so the attribution is **correlational, not a unique causal decomposition**.
- No mask-selection rule or input correction is proposed or applied. The 0.3 policy was not touched.

## 7. Frozen HMR2 contextual comparison

The same same-person rows were used throughout. HMR2 is valid on all 6,988.

| Split | E − M MPJPE | M severe, E not / both / E severe, M not | E accurate, M not / M accurate, E not | Wrong knee M vs E | Depth-sign errors M vs E |
|---|---:|---:|---:|---:|---:|
| VAL | −14.7 [−23.7, −2.9] | 26 / 9 / 2 | 633 / 213 | 18/3,129 vs 1/3,475 | 1,605 vs 1,147 of 11,729 |
| TEST | −20.6 [−30.4, −12.5] | 114 / 1 / 2 | 1,202 / 349 | 181/3,207 vs 0/3,753 | 3,003 vs 2,088 of 18,436 |

How HMR2 fares on S0's attributed failures:
- On the missingness failures it is correct for **187/190 wrong knee branches** and **127/133 severe frames**.
- On depth-sign failures it is correct on 60% of the ambiguity-supported ones (181/302), 61% of the model-specific ones
  (91/148) and 68% of the unresolved ones.
- It does **not** resolve the GT-projection-identical depth-flip pairs (0/44 both correct).
- In group A, HMR2 and S0 are at parity (+0.74 mm).

HMR2 differs in input (RGB crop), training data (including datasets S0 never saw), architecture, SMPL parameterization
(its joints share the GT's SMPL convention, which favours it on semantic offsets) and dataset exposure. Its better
results where S0 lacks joints are therefore **consistent with** the image containing the missing information, but they
are not causal proof that RGB appearance resolves it. HMR2 was not rerun, trained or fused.

## 8. Temporal information limits (diagnostic only)

Original bank neighbors only, same-person membership, no bridging. "2D-stable" means the neighbor input is within τ_N
(local tier for limb failures).

| M well-observed failures | n | Both neighbors present | Isolated (S0 correct at both neighbors) | Isolated and 2D-stable | Correct within ±3 samples |
|---|---:|---:|---:|---:|---:|
| All | 2,241 | 2,212 | 547 | 87 | 1,506 (67%) |
| Ambiguity-supported | 302 | | 79 | 28 | |
| Model-specific-supported | 148 | | 47 | 12 | |
| Unresolved | 1,791 | | 421 | 47 | |

- **Information potentially available across frames:** in 67% of well-observed failures, S0 gets the same decision
  right within ±3 original samples. In 87 cases, S0's decision flips between consecutive near-identical inputs; that is
  either a decision boundary or observation jitter.
- **Information actually consumed by S0:** one frame only.
- Neighbor correctness is GT-scored, so temporal consistency is not by itself evidence available at runtime.
- 1,682 failures are sustained: they persist into a neighbor.
- Of the 3 well-observed severe frames, 2 are sustained.
- Four M torso-forward events occur (S0 step > 90° while the GT step < 45°): crosscountry/315, stairs/200 and
  office_phoneCall/604 and /627.

No temporal model, filter, interpolation or hold policy was introduced.

## 9. Qualitative review

Figures are in `C:\Users\dna10\animcv-w103\results\qualitative\` (local, not committed). Each row shows, in separate
panels: the RGB frame with the M1 model-visible joints, the M tensor, the B tensor, **GT labelled "EVALUATION ONLY (not
available to S0)"**, S0 on M, S0 on B, and HMR2 (contextual). The figure manifest binds the image SHA-256, sample id,
split, timestamp and M1 identity status.

| Declared selection | Sample(s) | Observation |
|---|---|---|
| 1 Strict pair, different GT | crosscountry#000000 vs freestyle_01#000219 | The difference comes from the invalid crosscountry root (§4). Poses look alike; S0 errors are 26–48 mm. No replacement exists after exclusion. |
| 2 Well-observed success | courtyard_basketball_01:actor0#000949 | |
| 3 Well-observed failure (no model-specific severe case exists) | outdoors_parcours_00:actor0#001469 | |
| 4 Shared B/M failure | downtown_bar_00:actor1#001000 | Occluded legs (cf. W97) |
| 5 M-only regression | downtown_cafe_00:actor1#000715 | M lacks left hip and both ankles and places the legs on the chair: M 355 mm, B 85, HMR2 51 |
| 6 Knee-branch pair (local tier, leg_l) | parcours_01#001188 vs flat_packBags_00#000551 | |
| 7 Arm depth pair | primary: crosscountry#000486 vs cafe#001016; sensitivity: courtyard_hug_00:actor1#000586 vs downtown_bus_00#000010 | |
| 8 Torso orientation instability | office_phoneCall_00:actor0#000627 → #000632 | S0 step 143°, GT 3.2° |

The existing W102 BaseRig renders are reused for rig context, not regenerated:
`animcv-w102/results/rig_review_complete/maximum_source_regression*` covers cafe actor1 from frame 685 and contains
selection 5. `wrong_knee*` covers bar actor1/367. No IK, retargeting, camera correction or pose completion was applied.

## 10. Uncertainty and limits

- **Interval method.** Intervals resample whole source videos (2,000 replicates, seed 1337); pair counts are not treated
  as independent samples. Pair-level rates cluster on the first member's video.
- **Ceiling claim limited by coverage.** Only 6–14% of queries have any cross-video neighbor within τ_N, so "rarely
  ambiguous" is a statement about the matched region, which is dominated by upright, low-flexion poses. Deep knee
  flexion, seated and lying postures are under-represented (lying: 3 same-person rows).
- **Resolution scale.** τ comes from the cross-detector disagreement for the same pose, including the COCO/OpenPose
  semantic offset. A within-detector scale would be smaller.
- **Population.** 36 videos, with actors recurring across videos and unidentified. This is not a universal
  information-theoretic bound.
- **Attribution.** It depends on neighbor availability. 80% of well-observed failures are unresolved.

## 11. Executed and omitted checks

- **verify** — passed. Covers manifest/artifact/checkpoint/bank/raw identities, five geometry reproductions, the
  derived-token audit and probe composition.
- **reproduce** — byte-equal B/M predictions on both splits, plus 218 batch-one rows.
  - The first run's fingerprint mismatch is preserved.
  - The line-ending audit run (`reproduce_lf_audit`) passed.
- **analyze, first attempt (`analysis/`)** — failed after W102 parity passed, with a `KeyError` in the oracle-geometry
  quality context. Preserved.
- **`analysis_r2`** — completed, but its `decide()` counted all controlled video pairs instead of only the materially
  different ones. This could not change C (both mechanisms meaningful). It was corrected in **`analysis_r3`**, which
  is final. `analysis_r3` also adds the reference-validity record and the post-hoc block.
- **`analysis_r3_rootvalid`** — the sensitivity run.
- **Focused tests**, `tests/test_w103_observability.py`: **18 passed** (`results/w103_tests_final.xml`).
  - Coverage: frozen artifact hashes, real input reproduction and identity, translation/scale invariance of the
    model-visible tensor, valid-joint handling, local limb context, projection crop, ray rotation, template
    normalization, depth-sign/knee-branch and degenerate-knee handling, torso-frame parity, vectorized metric parity with
    W97/W100 helpers, cross-video distinct-sequence deterministic retrieval, observation-quality classes, seeded video
    bootstrap, the decision mapping, GT isolation of the declare stage, refusal without a declaration, and stored real
    analysis/reproduction parity.
  - The first run failed 3 tests. All three were test-side errors: a wrong context joint for `arm_r`, and tolerances
    tighter than float32 / arccos resolution. They were corrected; failures are retained in `results/w103_tests.xml`.
- **Regression:** `test_w102_same_environment.py` + `test_w101_reference_estimator.py`: 34 passed, 1 skipped (the
  checkpoint-backed W101 test was skipped because `W101_CHECKPOINT` was not set).
- **Not executed:** broad regression, S0/HMR2 inference beyond the reproduction, any new representation intervention
  (e.g. crop-only swaps), training, Blender rendering and LabServer jobs.

## 12. Changed files and reproduction

New files only:
- `scripts/w103_observability.py` (verify / reproduce / declare stages and shared metrics)
- `scripts/w103_analysis.py`
- `scripts/w103_qualitative.py`
- `tests/test_w103_observability.py`
- this worklog

No `src/`, historical script, checkpoint, dataset or W102 artifact changed. Outputs are in `C:\Users\dna10\animcv-w103\`
(SHA manifest `evidence_manifest.json` there).

PowerShell on Main, from the repo root:

```
$py=".venv_build_full\Scripts\python.exe"; $a="--w102 C:/Users/dna10/animcv-w102 --w101-data C:/Users/dna10/animcv-w101/data"
$o="C:/Users/dna10/animcv-w103/results"
& $py -X utf8 scripts/w103_observability.py verify $a --out-dir $o/verify
& $py -X utf8 scripts/w103_observability.py reproduce $a --out-dir $o/reproduce_lf_audit
& $py -X utf8 scripts/w103_observability.py declare $a --out-dir $o/declare
& $py -X utf8 scripts/w103_observability.py analyze $a --declaration $o/declare/declaration.json --declaration-sha fda5c08bfc1c64fcb4cc476bf7f1d21cfe095322c479db588d3dfec372e0038b --out-dir $o/analysis_r3
#   add --exclude-root-behind-camera with a fresh --out-dir for the sensitivity run
& $py -X utf8 scripts/w103_qualitative.py $a --analysis $o/analysis_r3/analysis_report.json $o/analysis_r3_rootvalid/analysis_report.json --out-dir $o/qualitative
& $py -X utf8 -m pytest tests/test_w103_observability.py -q
```

Every stage refuses to overwrite an existing output directory. Runtimes are about 5 s (verify, reproduce, declare),
about 4 min (analyze) and about 1 min (qualitative). Peak memory stayed within the 47 GB free; there was no long GPU
use.

## 13. CASE and evidence limits

**CASE C.**

- The important remaining S0 3D errors split by mechanism:
  - **Severe frames and wrong knee branches are overwhelmingly input missingness.** Joints absent from the
    model-visible representation account for 89% of M severe frames and 93% of M wrong knee branches (in the failing
    leg). On complete, well-localized inputs, these failures nearly vanish.
  - **Residual depth-sign errors on complete inputs** have measured support for both observational ambiguity and a
    learned-mapping limitation (about 2:1 among attributable cases). Most of them cannot be attributed with the
    available population.
- Full-body near-identical 2D with materially different 3D is rare once detector collisions and one invalid camera
  reference are controlled. That evidence is limited to densely matched, mostly upright poses, and does not establish a
  population-wide ceiling.
- The W102 M1 benefit is explained by joint availability, not by more accurate shared coordinates.
- External HMR2's advantage is contextual only.

This CASE authorizes no retraining, model replacement, mask/threshold change, production change or downstream feature.
