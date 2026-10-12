# Worklog — Visual Chain-Depth to Rig-Space IK Handoff (2026-10-12)

**Question.** With every non-direction geometry owner frozen, does the BEDLAM-supervised visual chain-depth gain of Worklog 73 translate into a genuine reduction of left-arm rig-space IK endpoint error? The frozen owners are H0 reach, H0 bend side, the Worklog 61 IK, the rig and the H0 Root Orientation.

**Answer.** Not as a genuine reduction against the H0 baseline: the pooled −0.58 cm is owned by crosscountry. **CASE C.**

- **E1 vs E0 (P2 − P1):** E1's direction improves on E0's after IK by 0.56 cm mean endpoint error, with 58% of rows improved. This is borderline:
  - **Only dancing clearly improves:** −1.66 cm, 67% of rows. Hug (−0.41 cm, 60%) is borderline and crosscountry (−0.12 cm, 50%) is unchanged.
  - **The interval nearly reaches zero:** the 95% moving-block bootstrap CI (blocks within each sequence) is [−1.05, −0.01] cm with 10-row blocks and [−1.14, +0.02] with 25-row blocks.
  - **The seed is favourable:** this is the seed at the favourable end of Worklog 73's seed range.
- **E1 vs H0 (P2 − P0):** the pooled −0.58 cm comes entirely from crosscountry (−3.32 cm × 133 rows, more than the whole net). Crosscountry is the sequence whose GT camera is caveated.
  - On the two clean sequences P2 does **not** beat the historical H0 IK.
  - **Dancing is clearly worse:** +1.44 cm, median +1.86, 28% of rows improved.
  - **Hug is mixed:** mean +0.53 cm but median −0.22, with 51% of rows improved; the loss is tail-driven.
  - **Without crosscountry:** P2 − P0 = +0.80 cm, 44% of rows improved.
- **The failure is upstream of the IK, in visual transfer.** On the clean sequences E1's chain depth is not better than H0's, so there is no gain over H0 to hand off. The gains that do exist (vs E0, and vs H0 on crosscountry) pass through the IK intact.
  - **Visual depth:** with perfect chain depth (P3) the endpoint error would fall from 11.7 to 4.4 cm.
  - **Secondary owners:** the observed-plane floor (2.2 cm when fixed after depth) and H0 reach (0.5–2.2 cm depending on the order owners are fixed).
  - **IK and bend:** zero clamps and an exact FK contract. H0's bend side moves only the elbow.

## Starting state and files

- **Start:** fetched `origin/arch/single_frame_first` and confirmed local = remote = `ffe5317ae5038088cc829185bf693af71d0577e4`, with a clean worktree.
- **Commits:**
  - `84becf5`: module, runner, tests
  - `4906b29`: review export + Blender render
  - the commit adding this worklog
- **Files added (evaluation only):**
  - `src/retarget/framepose_visual_ik_handoff.py`
  - `scripts/run_framepose_visual_ik_handoff.py`
  - `scripts/export_framepose_visual_ik_cases.py`
  - `scripts/review_framepose_visual_ik_blender.py`
  - `tests/test_framepose_visual_ik_handoff.py`
  - this worklog
- **Unchanged:**
  - Worklogs 60/61/62 code and artifacts
  - Worklog 69 C1 and Worklog 73 E0/E1, with their checkpoints, held-out predictions, BEDLAM adapter and caches
  - frozen H0, the Blender rest snapshot and rig calibration, AnimationSemantics
  - production FramePose, FK, IK and the Blender writer

## Artifact and checkpoint identities (hashed at runtime; only the two "required" reports are compared against pinned values)

| artifact | SHA-256 |
|---|---|
| Worklog 73 `report.json` (required) | `1c05d50a5e6577329f0573786c6927b07268612ec150fbd079a1aa2bc4b03934` |
| Worklog 73 `rows_heldout.json` | `4e17bd77399e4c016098f70592d0cfd68f508245ae60914869f756605d9897fc` |
| Worklog 73 `E0_3DPW_ONLY.pt` / `E1_3DPW_PLUS_EXTERNAL.pt` | `3a66f561…` / `ee002f6a…` |
| Worklog 62 `report.json` (required) / `rows.json` | `f7a100cd…` / `1fb095a9…` |
| Worklog 61 `report.json` | `b06f3b82…` |
| FrameBank content digest | `75519e63…` |
| Worklog 74 `report.json` | `8158e8b2381bf12ef23eb42698384903e70766ca8e05ad988a831910540c7795` |

- **Prediction values:** the E0/E1 checkpoints are not loaded. Their chain f values are read from `rows_heldout.json`, which is hashed but not pinned.
- **FrameBank:** the digest `75519e63…` is the canonical `benchmark_detector_observation` bank of Worklogs 53–73. The runner records it but does not pin it.

## Matched-frame accounting

The join uses exact `(sequence_id, frame_index)` identity. Each match is checked against the FrameBank position and `sample_id` (no positional approximation), and duplicates are refused.

| sequence | semantics frames | Worklog 61 IK-known | excluded (Worklog 61 UNAVAILABLE) | GT-valid matched | with a Worklog 73 prediction | Worklog 73 rows of that actor without a match |
|---|---:|---:|---|---:|---:|---:|
| dancing actor0 | 115 | 79 | 36 (mid 18, end 14, root 4) | 79 | 79 | 0 |
| crosscountry actor0 | 162 | 133 | 29 (end 14, mid 12, root 3) | 133 | 133 | 0 |
| hug actor1 | 187 | 183 | 4 (end 2, mid 2) | 183 | 183 | 0 |
| **total** | 464 | **395** | 69 | **395** | **395** | 0 |

- **Population unchanged:** the historical Worklog 62 matched population (395 rows) is kept exactly.
- **Excluded rows have no prediction:** none of the 69 Worklog 61-unavailable frames has a Worklog 73 prediction. Their detector shoulder/elbow/wrist validity fails the Worklog 65 eligibility rule.
- **Every candidate is available on every row:** P0–P3, the attribution control and the reference are all valid on all 395 rows.
  - There were no degenerate observed planes, no bend side that became undefined, and no clamps.
  - So the common valid population is all 395 rows.
- **No fallback is counted as a visual prediction.**

## Direction construction and coordinate transforms

- **Visual chain direction (canonical camera frame):** `reconstruct_direction(chain f, observed_plane(FrameBank detector 2D))`. It uses the detector shoulder/wrist 2D with each frame's image size; this is the Worklog 64–73 contract.
- **Not used:** GT 2D, GT image-plane orientation, Depth Anything, oracle depth signs or temporal input.
- **Heading and rig alignment:** the same runtime transform as Worklogs 61/62, `A · Rz(−yaw_H0)` (`heading_relative`). yaw_H0 is the AnimationSemantics v2 root orientation, so GT yaw is never used and no yaw is inferred from the visual head.
- **Fixed IK geometry (Worklog 61):**
  - target L1 = 27.771 and L2 = 27.251 armature units (cm at object scale 0.01), with the BaseRig `upperarm_l` rest head as root;
  - closed form `solve_two_bone`;
  - the existing clamp policy and degenerate-side fallback;
  - zero-twist local rotations and FK.
- **Bend side under a new direction:** H0's observed bend side is re-perpendicularized to the new direction (Worklog 62 `_runtime_side`). A side that becomes undefined stays UNAVAILABLE; this happened on 0 rows.
- **Ownership boundary:** only the chain endpoint direction changes. The Worklog 73 upper and forearm forward fractions are **not** used, no skeleton is stitched from segment predictions, and no consistency projection or kinematic optimization is added. The connected joints are built only by the Worklog 61 solver.

## Candidate contracts

| candidate | endpoint direction | reach | bend side | GT? |
|---|---|---|---|---|
| **P0_H0_IK** | H0 shoulder→wrist (Worklog 61 exactly) | H0 | H0 observed | no direct GT (H0 inputs include 2 oracle torso sign fields) |
| **P1_E0_VISUAL_DIRECTION** | E0 chain f + detector plane | H0 | H0, re-perpendicularized | direction: no; reach/side from H0 |
| **P2_E1_VISUAL_DIRECTION** | E1 chain f + detector plane | H0 | H0, re-perpendicularized | direction: no; reach/side from H0 |
| **P3_GT_FORWARD_FRACTION** (oracle) | GT chain f + the same detector plane | H0 | H0, re-perpendicularized | GT f only |
| A1 attribution control | H0's own chain f + detector plane | H0 | H0, re-perpendicularized | no |
| reference (Worklog 62 D) | GT 3D chain direction | H0 | H0, re-perpendicularized | GT direction |

- **Common target:** every candidate is scored against the same Worklog 62 `E_GT = target_root + GT_reach_fraction·(L1+L2)·GT_direction`, recomputed and checked equal to `rows.json` (max difference 0.0).
- **Elbow reference:** the Worklog 62 oracle IK elbow (GT direction + GT reach + GT side).
- **Reference elbow differs from Worklog 62 D:** the reference reproduces Worklog 62 D's endpoint error (max |Δ| 2.9e-14). Its elbow uses the H0 side, unlike Worklog 62 D, which used the GT side.

## Historical IK reproduction (checked before any comparison; the run stops otherwise)

| contract check | max deviation | limit |
|---|---:|---:|
| exact H0 direction rebuilt via `reconstruct_direction(f_H0, H0 plane)` vs the H0 camera direction | 2.8e-16 | 1e-12 |
| that rebuilt candidate vs persisted Worklog 61 solved endpoint / elbow / requested endpoint (cm) | 4.4e-14 / 3.3e-14 / 2.9e-14 | 1e-9 |
| P0 vs persisted Worklog 61 solved endpoint (cm) | 4.5e-14 | 1e-9 |
| FK of every candidate's local rotations vs its solved elbow/wrist (cm) | 5.9e-14 | 1e-9 |
| E_GT vs Worklog 62 `rows.json` (cm) | 0.0 | 1e-9 |
| reference endpoint error vs Worklog 62 D | 2.9e-14 | 1e-9 |
| Worklog 73 `chain_f_h0` vs H0 semantics / `chain_f_gt` vs 3DPW GT | 0.0 / 7.8e-8 | 1e-6 / 1e-5 |

The coordinate and alignment contract therefore reproduces Worklog 61 to machine precision.

## Results (395 common valid rows; armature units = cm; mean / p50 / p95 / max)

**Pooled:**

| candidate | chain direction error (°) | requested = solved endpoint error | elbow error | clamps |
|---|---|---|---|---|
| P0 H0 IK | 14.19 / 10.34 / 38.31 / 133.18 | 12.30 / 9.38 / 31.09 / 73.82 | 7.00 / 5.03 / 18.07 / 49.75 | 0 |
| P1 E0 visual | 14.23 / 10.94 / 33.70 / 112.21 | 12.28 / 10.21 / 29.15 / 85.18 | 7.13 / 5.96 / 15.86 / 45.37 | 0 |
| **P2 E1 visual** | 13.48 / 10.74 / 29.99 / 104.87 | **11.72 / 10.03 / 25.49 / 81.34** | 7.04 / 5.86 / 16.71 / 44.94 | 0 |
| P3 GT f (oracle) | 3.93 / 2.52 / 11.85 / 53.20 | 4.38 / 2.89 / 12.80 / 36.09 | 5.01 / 3.51 / 13.79 / 37.02 | 0 |
| A1 H0 f + detector plane | 14.19 / 10.49 / 37.66 / 129.33 | 12.32 / 9.43 / 30.34 / 72.70 | 7.16 / 5.21 / 17.64 / 49.78 | 0 |
| reference: GT direction + H0 reach | 0 | 2.15 / 0.89 / 8.42 / 25.19 | 4.38 / 2.97 / 11.77 / 36.99 | 0 |

With no clamps, requested and solved endpoints are identical on every row.

**Paired endpoint differences (solved endpoint error, cm; mean, p50, p95; share of rows improved):**

| comparison | pooled | dancing (79) | crosscountry (133) | hug (183) | without crosscountry (262) |
|---|---|---|---|---|---|
| **P2 − P1** (causal: new supervision) | **−0.56** (−0.53, +6.07; 58%) | −1.66 (67%) | −0.12 (50%) | −0.41 (60%) | −0.79 (62%) |
| **P2 − P0** (practical: vs IK baseline) | **−0.58** (−0.22, +13.15; 51%) | **+1.44** (28%) | −3.32 (64%) | **+0.53** (51%) | **+0.80** (44%) |
| P1 − P0 | −0.02 (49%) | +3.10 (19%) | −3.20 (63%) | +0.94 (52%) | +1.59 (42%) |
| A1 − P0 (plane-source swap only) | +0.02 (44%) | +0.01 | −0.08 | +0.10 | +0.07 |
| P3 − P2 (remaining visual depth error) | −7.34 (99%) | −5.78 | −7.87 | −7.62 | −7.07 |

**Direction differences:**
- P2 − P1: −0.75° pooled; per sequence dancing −2.23°, crosscountry −0.26°, hug −0.48°.
- P2 − P0: −0.71° pooled; per sequence dancing +1.25°, crosscountry −3.96°, hug +0.81°.

**Survival of the chain-direction gain through IK** (pooled):

| | relative direction gain | relative endpoint gain |
|---|---:|---:|
| P2 vs P1 | 5.3% | 4.6% |
| P2 vs P0 | 5.0% | 4.7% |

- **IK cannot attenuate the gain here, by construction.** Every candidate shares H0 reach and no row clamps, so endpoint error = (L1+L2)·√((r_H0 − r_GT)² + 4·r_H0·r_GT·sin²(θ/2)), which increases strictly with the direction error θ.
  - Direction and endpoint improved-row shares are therefore identical in every block.
  - Pooled, the endpoint keeps 0.86 (P2 vs P1) and 0.95 (P2 vs P0) of the relative direction gain.
  - Per sequence, P2 vs P1 is 0.86 (dancing), 1.03 (hug) and 0.52 (crosscountry). On crosscountry the gain is only −0.26° / −0.12 cm, so the ratio is unstable.
  - P2 vs P0 on crosscountry, the only sequence where P2's direction beats H0's, is 0.94.
- **The direction gain itself is small** on this population.

**Chain forward-fraction error on these rows.** Cells give \|f − f_GT\| mean / p50 / p95 (linear interpolation, as everywhere else), then sign accuracy on rows with \|f_GT\| > sin 10° (75 / 77 / 118 / 270 rows). The table was computed post hoc from `rows.json`.

| sequence | H0 | E0 | E1 |
|---|---|---|---|
| dancing | **.134** / .107 / .329; .987 | .196 / .164 / .518; .987 | .159 / .137 / .333; .987 |
| crosscountry | .284 / .257 / .687; .727 | .221 / .182 / .544; .883 | **.212** / .178 / .480; .883 |
| hug | **.181** / .160 / .373; .975 | .197 / .152 / .473; .966 | .189 / .164 / .443; .975 |
| dancing + hug | **.167** | — | .180 |
| pooled | .206 / .156 / .554; .907 | .205 / .163 / .494; .948 | **.191** / .158 / .446; .952 |

- **The endpoint pattern follows the depth pattern exactly:**
  - E1's chain depth beats H0's only on crosscountry, where H0's sign accuracy is .727;
  - on the clean sequences H0's depth is as good or better.
- **Plane source is not the cause:** the A1 control shows that replacing H0's own image-plane orientation with the detector plane is neutral (+0.02 cm).

## Elbow and bend

- **Elbow error, pooled:** P0 7.00, P1 7.13, P2 7.04 cm mean (medians 5.03 / 5.96 / 5.86).
- **The elbow tracks the endpoint.** On the 201 rows where P2's endpoint beats P0's, P2's elbow is also better on 83% (mean −2.25 cm).
  - P2 − P0 elbow is −1.54 cm on crosscountry (65% of rows better), +1.96 cm on dancing (82% worse) and +0.36 cm on hug.
  - Without crosscountry it is +0.84 cm (mean 5.59 → 6.43, p50 4.40 → 5.64), so the pooled +0.04 cm is a cancellation between sequences.
  - P2 − P1 elbow changes are small: pooled −0.09, and per sequence −0.09 / +0.24 / −0.33 cm.

**Per-sequence endpoint and elbow error (mean / p50 / p95 / max, cm):**

| sequence | candidate | endpoint | elbow |
|---|---|---|---|
| dancing | P0 | 8.94 / 6.74 / 20.71 / 73.82 | 5.76 / 4.34 / 16.19 / 42.41 |
| | P1 | 12.04 / 9.89 / 30.41 / 56.18 | 7.81 / 6.00 / 23.17 / 37.97 |
| | P2 | 10.39 / 8.34 / 20.30 / 57.75 | 7.72 / 5.69 / 19.17 / 29.59 |
| | P3 | 4.60 / 2.98 / 10.82 / 36.09 | 4.51 / 2.62 / 12.37 / 23.08 |
| crosscountry | P0 | 17.60 / 16.08 / 43.13 / 64.53 | 9.78 / 8.02 / 22.60 / 49.75 |
| | P1 | 14.40 / 11.61 / 33.38 / 85.18 | 8.00 / 6.33 / 19.38 / 45.37 |
| | P2 | 14.28 / 11.87 / 30.17 / 81.34 | 8.24 / 6.30 / 18.46 / 44.94 |
| | P3 | 6.41 / 4.27 / 23.77 / 28.40 | 6.68 / 5.31 / 15.76 / 37.02 |
| hug | P0 | 9.90 / 8.78 / 20.52 / 36.49 | 5.51 / 4.56 / 11.78 / 35.70 |
| | P1 | 10.84 / 9.11 / 23.97 / 40.22 | 6.20 / 5.65 / 12.71 / 16.63 |
| | P2 | 10.43 / 9.60 / 22.60 / 40.62 | 5.87 / 5.53 / 12.52 / 18.72 |
| | P3 | 2.80 / 2.08 / 7.56 / 18.86 | 4.02 / 3.00 / 9.51 / 35.30 |
| without crosscountry | P0 | 9.61 / 7.93 / 20.70 / 73.82 | 5.59 / 4.40 / 13.71 / 42.41 |
| | P2 | 10.41 / 9.17 / 22.08 / 57.75 | 6.43 / 5.64 / 14.37 / 29.59 |
- **Bend status:** every candidate's side is `observed` on all 395 rows, so no fallback or undefined side occurred. Side flips against the GT side are 10/395 for P0–P3 (9 for A1).
- **Bend ownership only moves the elbow:** with GT direction and GT reach, H0's side alone gives an elbow error of 3.23 cm mean (p50 1.84, p95 11.01; the Worklog 62 column). The endpoint is independent of the side.

## Failure attribution (pooled means, cm; common valid rows)

| owner | how isolated | sequential share of P2's 11.72 cm (depth, then plane, then reach; order-dependent) |
|---|---|---:|
| 1. remaining chain depth error of the visual predictor | P2 − P3 (perfect f, same plane and reach) | **7.34** |
| 2. observed 2D plane (weak-perspective image-plane orientation) | P3 − reference (perfect GT 3D direction) | 2.23 |
| 3. H0 reach fraction | reference − oracle (0 by construction) | 2.15 |
| 4. H0 bend side | elbow only (3.23 cm mean on the GT target); endpoint unaffected | 0 at the endpoint |
| 5. IK mechanics / clamps | 0 clamps on all candidates; FK contract 5.9e-14 | 0 |

- **The ladder is order-dependent.** Reach error is radial and direction error tangential, so they combine roughly in quadrature. Fixing reach first, before direction, removes only 0.46 cm. H0 reach therefore costs 0.5–2.2 cm depending on order, and visual depth dominates either way.
- **Per sequence**, the visual-depth share is 5.78 (dancing), 7.87 (crosscountry) and 7.62 (hug).
- **The plane floor is largest on crosscountry** (3.56 cm, P3 direction p95 26.6°).
- **The reach share** is 2.15 / 2.86 / 1.64 cm.

## Strata (evaluation-only; endpoint mean cm; P2 − P1 and P2 − P0 with share improved)

| stratum | n | P0 | P1 | P2 | P3 | P2 − P1 | P2 − P0 |
|---|---:|---:|---:|---:|---:|---|---|
| \|GT chain f\| near plane (< .174) | 125 | 9.52 | 7.45 | 6.99 | 2.41 | −0.46 (60%) | −2.53 (78%) |
| intermediate | 196 | 11.00 | 13.00 | 12.53 | 4.39 | −0.46 (59%) | **+1.54 (32%)** |
| strongly out of plane (≥ .5) | 74 | 20.45 | 18.54 | 17.53 | 7.66 | −1.00 (54%) | −2.92 (54%) |
| H0 \|reach error\| Q1 (< .0055) | 98 | 9.33 | 7.87 | 7.08 | 1.68 | −0.79 | −2.25 (68%) |
| Q2 | 99 | 9.58 | 10.85 | 11.06 | 2.50 | +0.21 | +1.47 (34%) |
| Q3 | 99 | 11.24 | 13.89 | 13.28 | 4.38 | −0.61 | +2.04 (41%) |
| Q4 (≥ .049) | 99 | 19.01 | 16.46 | 15.40 | 8.92 | −1.05 | −3.61 (60%) |
| H0 bend 0–15° | 32 | 14.22 | 11.20 | 11.11 | 2.45 | −0.08 | −3.10 (69%) |
| 15–45° | 178 | 10.25 | 9.18 | 8.69 | 3.31 | −0.49 | −1.56 (61%) |
| 45–90° | 162 | 13.63 | 15.67 | 14.99 | 5.33 | −0.67 | +1.37 (35%) |
| ≥ 90° | 23 | 16.14 | 13.91 | 12.86 | 8.61 | −1.05 | −3.28 (61%) |
| Worklog 62 H0 direction-error Q1 (< 6.2°) | 98 | 4.19 | 6.68 | 6.05 | 2.12 | −0.63 | **+1.86 (44%)** |
| Q2 (6.2–10.3°) | 99 | 7.78 | 12.31 | 11.39 | 3.82 | −0.91 | **+3.61 (33%)** |
| Q3 (10.3–18.7°) | 99 | 12.08 | 13.08 | 12.78 | 3.48 | −0.29 | +0.71 (46%) |
| Q4 (≥ 18.7°) | 99 | 25.07 | 16.99 | 16.58 | 8.06 | −0.41 | **−8.49 (80%)** |

- **Strata keyed to H0's own GT error cannot show complementarity.** Sorting by either candidate's error mechanically makes the other look like a rescuer.
  - By H0 direction-error quartile, P2 − P0 is +1.86 / +3.61 / +0.71 / −8.49 cm.
  - By P2's own direction-error quartile it is −5.17 / −1.73 / −1.48 / +6.00 cm (93% / 17% of rows improved in Q1 / Q4).
  - 60 of the 99 rows in H0's Q4 are crosscountry.
- **Some rescues are crosscountry-only:** the nearly straight (0–15°) rescue does not hold on the clean sequences.
- **The non-mechanical strata are non-monotonic:** the GT-\|f\| strata (−2.53 / +1.54 / −2.92) do not depend on P0, and the H0 bend-angle strata are runtime-observable.
- **Selection headroom:** per-row oracle selection would give 8.76 cm for {P0, P2} and 8.92 cm for {P0, P1}. So headroom exists, but it is not specific to E1. Whether any runtime-observable signal could capture it is an open question, not addressed here (no trust gate, no fusion, nothing tuned).

## Owner cases (Worklog 62 cases kept, including negative ones)

Cells give endpoint error (cm) / chain direction error (°). Bold marks the lower endpoint error of P0 and P2 (the practical comparison). On dynamic #462 and largest_direction_error #176, P1 is lower still.

| case | row | f: H0 / E0 / E1 / GT | P0 | P1 | P2 | P3 |
|---|---|---|---|---|---|---|
| known_good | dancing #424 | +.21 / +.16 / +.16 / +.32 | **6.51** / 6.9 | 9.29 / 9.9 | 9.53 / 10.1 | 2.66 / 2.8 |
| dynamic | crosscountry #462 | +.11 / +.37 / +.35 / +.54 | 26.22 / 28.4 | 13.38 / 14.4 | **14.29** / 15.4 | 7.70 / 8.3 |
| turning (review) | crosscountry #211 | −.62 / +.58 / +.46 / −.91 | **44.24** / 51.1 | 85.18 / 112.2 | 81.34 / 104.9 | 23.48 / 26.4 |
| tracking_loss (review) | hug #312 | −.31 / +.01 / −.04 / −.47 | **9.12** / 10.6 | 24.82 / 29.1 | 22.10 / 25.9 | 1.94 / 2.2 |
| near_straight | crosscountry #525 | +.33 / +.16 / +.15 / −.03 | 20.28 / 21.3 | 10.73 / 11.2 | **10.28** / 10.7 | 1.87 / 1.9 |
| strongly_bent = largest_reach_error | crosscountry #173 | −.14 / +.19 / +.22 / +1.00 | 50.87 / 97.0 | 44.81 / 77.8 | **44.28** / 76.2 | 25.19 / 0.1 |
| largest_ratio_error | hug #397 | −.49 / −.93 / −.93 / −.24 | **18.98** / 17.0 | 40.22 / 54.9 | 40.62 / 55.5 | 16.86 / 11.6 |
| largest_direction_error | dancing #176 | −.41 / +.36 / +.30 / +.87 | 73.82 / 133.2 | 56.18 / 88.5 | **57.75** / 91.7 | 36.09 / 53.2 |
| turning (invalid) | crosscountry #202 | UNAVAILABLE: Worklog 61 IK unavailable (`source_end_invalid`) | | | | |
| invalid_endpoint | hug #406 | UNAVAILABLE: Worklog 61 IK unavailable (`source_mid_invalid`) | | | | |

- **The visual direction is worse than H0 on four cases:**
  - known_good, which loses 3 cm;
  - turning #211, where E0/E1 put the wrist behind the body against GT f −0.91 (a depth-sign failure);
  - tracking_loss #312;
  - largest_ratio_error #397, where E1 is over-confident at f −0.93 against GT −0.24.
- **It is better than H0 on four cases:** dynamic, near_straight, strongly_bent and largest_direction_error. All of them are rows where H0's own direction was badly wrong.
- **Rows where H0's depth sign is wrong:** on strongly_bent #173 (GT f +1.00, H0 −0.14) and largest_direction_error #176 (GT +0.87, H0 −0.41), E1 at least gets the sign right.

**Elbow error (cm) and bend side vs GT, P0 / P1 / P2 / P3:**

| case | elbow | side vs GT |
|---|---|---|
| known_good | 4.57 / 6.00 / 6.12 / 2.23 | all agree |
| dynamic | 13.99 / 7.27 / 7.76 / 3.88 | all agree |
| turning | 25.01 / 39.06 / 37.57 / 15.21 | all agree |
| tracking_loss | 1.90 / 9.94 / 8.63 / 3.09 | all agree |
| near_straight | 8.31 / 3.99 / 3.82 / 3.43 | all disagree |
| strongly_bent | 49.75 / 45.37 / 44.94 / 37.02 | P0 flip, P1/P2 agree, P3 flip |
| largest_ratio_error | 17.60 / 4.72 / 4.79 / 23.65 | all agree |
| largest_direction_error | 42.41 / 28.23 / 29.59 / 5.77 | all agree |

On largest_ratio_error #397 the elbow ranking is the reverse of the endpoint ranking.

## Blender review exports

**Pipeline:** the Worklog 62 review pipeline was reused unchanged in settings.
- Worklog 60 object rotation applied to the BaseRig rest pose (gray). The exported FK locals cover only the hidden left-arm bones. Blender 5.1.2 Workbench, 960 × 960;
- orthographic cameras aligned to heading, side view = depth × up and a front view;
- identical frame identities and rest alignment;
- chains: blue P0 (historical H0 IK), orange P1, magenta P2, green GT reference (oracle IK reaching E_GT), yellow wire E_GT;
- captions carry the exact frame identity, endpoint/direction/elbow errors and the four forward fractions.

**Checks:** the export reproduces the persisted Worklog 61 solve (P0 end/elbow vs Worklog 61 ≤ 3.2e-14 cm). The Blender FK end agrees with the export to 1.2e-5 cm.

**Cases (10, 20 PNGs):**
- the 8 available owner cases;
- the largest P2-vs-P0 gain (crosscountry #178);
- the largest P2-vs-P0 loss, which is the same row as turning #211 (kept once with both labels);
- the median-P0 strongly out-of-plane row (hug #217).

The two Worklog 61-unavailable owner cases are listed as refused.

**What the renders show:**
- **dynamic #462:** the visual chains sit between H0 and GT. The wrist is about 12 cm closer and the elbow about 6 cm closer; the bend side is the same for all.
- **turning #211 (= largest P2 loss):** the visual chains swing the forearm behind the body, while GT and H0 point toward the camera (a depth-sign failure).
- **near_straight #525:** every candidate's bend side is opposite to GT, but a nearly straight arm moves the elbow little (P0 8.3 vs P2 3.8 cm).
- **strongly_bent #173:** P0's side flips against GT while P1/P2 agree. All wrists remain far from E_GT (P3 also 25 cm), because the H0 reach is wrong.
- **tracking_loss #312 and largest_ratio_error #397:** E1 bends the arm toward the wrong depth. On #397 the elbow is nevertheless 12.8 cm better than P0's.
- **largest P2 gain (crosscountry #178) and median out-of-plane (hug #217):** consistent with the numeric table.
- These views agree with the numeric table above; no success is claimed from visual plausibility.

Images are copied locally to `~/animcv-output/74_framepose_visual_ik_handoff/`. The render report is on LabServer63 `~/animcv-output/_scratch/w74_review/`.

## Architecture classification

**CASE C: the visual improvement does not survive the handoff as a meaningful gain over the IK baseline.**

| CASE A requirement | evidence | met |
|---|---|---|
| P2 materially improves P1 on common frames | −0.56 cm mean (4.6%), 58% of rows; clear on dancing only; bootstrap CI reaches about 0 | borderline |
| P2 improves the historical P0 endpoint | pooled −0.58 cm, but without crosscountry +0.80 cm (dancing +1.44, hug +0.53 mean / −0.22 median) | pooled yes / clean no |
| not restricted to one scene | the P2-vs-P0 gain is owned by crosscountry alone | **no** |
| a meaningful share of the direction gain survives IK | pooled 0.86–0.95 of the (small) direction gain, by construction | yes |
| endpoint p95 does not regress | pooled 31.09 → 25.49 (crosscountry 43.13 → 30.17); without crosscountry 20.70 → 22.08 (hug 20.52 → 22.60, dancing 20.71 → 20.30) | pooled yes; clean sequences no (mild) |
| elbow/bend acceptable | pooled 7.00 → 7.04, but clean sequences 5.59 → 6.43; 0 fallbacks, 0 clamps | regresses on clean sequences |

**Not CASE B.** With H0 reach shared and no clamps, the endpoint follows the chain direction by construction (pooled survival 0.86–0.95). Where P2's direction is better than H0's (crosscountry), the endpoint is better. Where it is worse (dancing), the endpoint is worse. On hug both are mixed: mean slightly worse, median slightly better.

**Answer to the completion question.** No, not genuinely. With all non-direction geometry owners frozen, P2 lowers the pooled endpoint error relative to the historical H0 IK only slightly (−0.58 cm mean, 51% of rows; p95 31.09 → 25.49), and all of that comes from crosscountry, whose GT camera is caveated. Without crosscountry it is +0.80 cm (44% of rows improved). Relative to E0 it reduces error modestly and borderline (−0.56 cm).

**The limiting factor is visual transfer.**
- **Visual depth:** E1's chain forward fraction is not more accurate than H0's except on crosscountry. On dancing + hug the \|f\| error is E1 .180 vs H0 .167; the pooled .191 vs .206 advantage comes from crosscountry alone (.212 vs .284). Its remaining depth error owns 7.3 of P2's 11.7 cm (sequential).
- **Observed-plane reconstruction (2.2 cm) and reach (0.5–2.2 cm, order-dependent):** secondary owners.
- **Bend ownership:** only moves the elbow.
- **IK mechanics:** not a limitation.

**Caveats:**
- **Selection overlap:** all 395 rows are 3DPW validation rows. Worklog 73 selected the E0/E1 epochs on this split (by arm \|f\|), and frozen H0's epoch was also selected on it (by whole-body MPJPE, `src/framepose/train.py:237-238`). The arm-specific selection plausibly favours E1 more, yet E1 still does not beat H0 on the clean sequences (dancing clearly worse, hug mixed).
- **Population:** on Worklog 73's full splits, E1's chain \|f\| beats H0 by .034 on validation (.167 vs .201, 3,056 rows) and by .071 on test (.200 vs .271). On these 395 IK-replay rows the margin is only .015, and it reverses on dancing and hug. The CASE C verdict and the visual-transfer diagnosis apply to these three sequences, the only ones with Worklog 61/62 replays.
- **Oracle torso signs:** H0 was built with two oracle torso sign fields (shoulder and hip side), which the visual head never sees.
  - **Direction:** P0 takes its direction from H0, which favours P0.
  - **Shared reach and side:** the H0 reach and bend side used by every candidate come from the same oracle-assisted model. So absolute errors and the reach/bend attributions are optimistic. The shared part cancels in P2 − P1 and P2 − P0.
- **Seed:** Worklog 73's three-seed table puts seed 1337 at the favourable end of the validation chain gain (E1 − E0 \|f\| −.014 vs −.011 / −.004). The −0.56 cm P2 − P1 handoff gain is probably an upper-end estimate.
- **Crosscountry GT camera:** its official extrinsics put the GT behind the camera (Worklog 63), so the only sequence where P2 beats P0 has the least reliable evaluation target.
- **No test-split IK replay exists:** there are no Worklog 57/61/62 artifacts for 3DPW test sequences, so a selection-clean handoff evaluation would first need new semantics and IK replays.
- **One seed:** only the seed-1337 Worklog 73 checkpoints were saved, so per-row seed variance cannot be measured here.

**Per the decision logic:**
- Worklog 73's independent visual-depth result is not rejected: at this seed, E1 > E0 holds after IK (−0.56 cm, borderline under frame autocorrelation).
- The visual branch is not promoted to production, and no IK, reach, bend, gate or fusion change was made in response.
- Per-row oracle selection between H0 and a visual direction has headroom (8.76 cm). Whether a runtime-observable signal could capture it remains an open hypothesis, not something these strata demonstrate. Nothing was implemented.

## Focused tests

`tests/test_framepose_visual_ik_handoff.py` has 7 tests. They cover:
- canonical observed-plane reconstruction (axis signs, image y-down, degenerate plane refused);
- the exact H0 direction rebuilt through the visual path reproducing the Worklog 61 runtime solve (≤ 1e-12);
- heading-relative transform parity with Worklog 62 `heading_relative` and with explicit quaternions;
- identical H0 reach and re-perpendicularized H0 side for all candidates, with candidates differing only by direction (same f gives an identical solve);
- fixed target rest lengths and the IK→FK numerical contract;
- scoring against the Worklog 62 E_GT, the Worklog 62 oracle elbow (P0 elbow error equals Worklog 62 `ik_h0_elbow_error`) and the Worklog 62 D endpoint identity;
- unavailable behaviour (degenerate direction, side parallel to the new direction: reported, not repaired);
- static checks, by source presence: GT appears only in P3 and evaluation, the run-time identity/STOP checks are present, and no production module imports the evaluation module.

**Test limits:**
- The transform-parity test compares against Worklog 62's `heading_relative` (which `rig_direction` calls) and an explicit quaternion composition.
- The Worklog 61 parity unit test uses the solver primitives. Full-pipeline parity against the persisted Worklog 61 files is the runtime check above.

**Runtime checks:**
- The Worklog 73 row ↔ FrameBank association is enforced (sample_id and position equality, duplicate refusal).
- So are the two pinned report SHA-256s and all contract limits above.
- The E0/E1 prediction values in `rows_heldout.json` are recorded by SHA-256 (`4e17bd77…`) but not pinned.

**Results:**
- 45 passed in the training image (including the added elbow assertion), with the unchanged Worklog 61 (12), Worklog 62 (7), Worklog 73 (14) and Worklog 69 continuous (5) tests.
- Locally: 26 passed (W74 + W61 + W62).

No production file changed, so the full regression was not run.

## Outputs

On LabServer63, `~/animcv-output/framepose_visual_ik_handoff/` contains:
- `report.json` (`8158e8b2…`)
- `rows.json` (per-row geometry for every candidate)
- `handoff_table.csv`
- `review_cases.json`

Production integration is not claimed.
