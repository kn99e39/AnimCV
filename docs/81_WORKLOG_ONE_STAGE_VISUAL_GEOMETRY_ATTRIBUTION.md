# Worklog — One-Stage Visual Geometry Contribution Attribution (2026-10-14)

**Question.** Worklog 80's J1 (2D + frozen Qwen) beats J0 (2D only) on static distal and out-of-plane geometry, but is much jumpier. Do J1's useful visual gains and its temporal regression come from the same derived arm quantity (chain direction, reach or bend side), or from different quantities and their interactions?

**Answer.** Different, and not separable. **Hypothesis C — distributed interactions**, by the predeclared rule.

- **The static gain belongs to chain direction.** J1's forearm-direction gain over J0 (−2.11°, interval excludes 0) is 62% J1 direction, 29% J1 side and 9% reach, with no material interaction. In out-of-plane rows, J1's wrist gain (−4.0 cm) is 90% direction and its forearm gain (−5.8°) 71% direction.
- **The temporal cost is distributed.**
  - J1 − J0 upper-arm motion residual is +3.36°/pair (37/37 sequences).
  - Its main effects are direction +1.98, side +0.81 and reach +0.52.
  - Interactions are large and sub-additive: direction × reach −1.00, direction × side −0.99, reach × side −0.48, all intervals excluding 0, about 30% of the total each.
- **What sub-additivity means here.**
  - Grafting any single J1 component into J0's arm is expensive: direction +3.02, reach +1.31, side +1.59 °/pair.
  - Inside J1's own arm, reverting a single component to J0 removes little: direction −1.04 (31%), reach and side about 0. Reverting reach alone actually increases the residual (+0.17).
  - The penalty is a property of the whole J1 configuration, not of one quantity.
- **Direction overlaps both sides of the trade-off.** It is the static owner and the largest single temporal contributor: its frame-to-frame change rises +3.48°/pair, 100% direction-owned. But it does not own the temporal cost alone. So a direction-only correction is not architecturally justified by this evidence.
- **The Worklog 80 regime findings decompose as follows.**

  | finding | J1 − J0 elbow / wrist | owned by |
  |---|---|---|
  | J0's near-folded strength | elbow +1.05 cm (J1 worse) | J1 side (main +0.54; swap +0.88) and direction (+0.43) |
  | J0's out-of-plane wrist weakness | wrist −4.0 cm (J1 better) | direction |
  | J1's near-plane loss | elbow +0.53, wrist +1.17 cm (J1 worse) | J1 direction |

  The moderate-bend elbow regression is not a J0/J1 component effect. Both models are worse than B1 there, and J1 is 0.45 cm better than J0 (reach −0.35, direction −0.18).

## Starting state and files

- **Start:** `arch/single_frame_first` at `cf8d90d2fc05f7ee5d7889d2d656f5a7554c9906`. Local, origin and LabServer63 were equal, and the worktree was clean.
- **Commits:**
  - `499f1d8`: factorial module, runner, review export, tests
  - the commit adding this worklog
- **Files added (evaluation only, not imported by production):**
  - `src/retarget/framepose_joint_factor_attribution.py`
  - `scripts/run_framepose_joint_factor_attribution.py`
  - `scripts/export_framepose_joint_factor_cases.py`
  - `tests/test_framepose_joint_factor_attribution.py`
  - this worklog
- **Review renders** use the unchanged Worklog 77 renderer.
- **Nothing was trained.** The J0/J1 checkpoints, predictions and Worklog 80 artifacts are read-only inputs.
- **Unchanged:** every historical module and report, E1, H0, the Worklog 61 IK, the Worklog 62 reference, FrameBank and splits, BaseRig, Root Orientation, and production FramePose/FK/IK/Blender writer.

## Frozen artifact identities

**Pinned and checked:**

| artifact | sha256 |
|---|---|
| Worklog 76 report | `dad46288…` |
| Worklog 80 training report | `5d6e974a…` |
| Worklog 80 predictions | `eaf3f57c…` |
| Worklog 80 evaluation report | `a27eeab2…` |
| Worklog 80 test rows | `8835e79a…` |
| Worklog 80 test pairs | `b25e7c97…` |

Also recorded and checked: the J0/J1 checkpoint hashes, H0 test predictions (equal to Worklog 76's), the target chain, and the jump thresholds (equal to Worklog 80's).

**Outputs (`~/animcv-output/framepose_joint_factor_attribution/`):**

| file | sha256 |
|---|---|
| `report.json` | `ff1564d3…` |
| `rows_test.json` | `bd92c4c7…` |
| `pairs_test.json` | `7c186336…` |
| `review_cases.json` | `3a2d3b9d…` |

Review renders are in `~/animcv-output/framepose_joint_factor_attribution_review/` (render report `087686ac…`). The output directory is root-owned, so the renders are kept beside it.

## Factorial candidates

| candidate | direction | reach | bend side | role |
|---|---|---|---|---|
| V000 | J0 | J0 | J0 | = Worklog 80 J0 |
| V100 | J1 | J0 | J0 | attribution tool |
| V010 | J0 | J1 | J0 | attribution tool |
| V001 | J0 | J0 | J1 | attribution tool |
| V110 | J1 | J1 | J0 | attribution tool |
| V101 | J1 | J0 | J1 | attribution tool |
| V011 | J0 | J1 | J1 | attribution tool |
| V111 | J1 | J1 | J1 | = Worklog 80 J1 |

**How the factors are built:**
- Each factor comes from one model's single predicted connected arm through the Worklog 62 `measure_chain`, with the same yaw_H0 and alignment.
- The selected side is re-perpendicularized to the selected direction by the Worklog 62 `_runtime_side`.
- A degenerate source side keeps the rest-plane fallback. J0 has one near-straight row, which propagates to V000/V010/V100/V110.
- The solve is the unchanged Worklog 76 `full_arm`.
- **The mixed candidates deliberately break the one-stage ownership.** They are attribution tools, not runtime architectures, and none is promoted.

**Contrasts** (per row or per pair, then sequence-aware):

| contrast | definition |
|---|---|
| total | V111 − V000 |
| swap from J0 | change one factor to J1 with the others at J0 |
| revert from J1 | V111 minus J1 with one factor reverted to J0 |
| main effect | mean over the other two factors of the J1-minus-J0 effect |
| two-way interaction | mean difference of conditional effects |
| three-way interaction | the remaining non-additivity |

Statistical independence of the factors is not assumed.

## Gates

**Gate 1 — Worklog 80 parity.** Passes.
- Population: 6,003 rows set-equal, 37 sequences, 5,504 pairs identical to Worklog 80.
- V000 = J0, V111 = J1, B0 and B1: joints differ by 0. Status, side status, clamp, reach, direction and side are identical, and rotations are within 1e-9.
- Every static metric and every per-pair temporal metric equals the Worklog 80 records (max 4.1e-14).
- The reference arm and reference motion are identical.
- Worklog 80 report means are reproduced for elbow, wrist, forearm and the upper-arm residual.
- The Worklog 76 thresholds are unchanged.

**Gate 2 — factor contract.** Passes.
- Exact factor assignment on every row: reach source, normalized direction source, and the side recomputed with `_runtime_side`.
- All eight candidates are known on all 6,003 rows, with 0 clamps and rig residuals ≤ 6.1e-14.
- No validity was lost.

## Static candidates (6,003 rows; mean / p95)

| | wrist cm | elbow cm | forearm ° | upper ° | chain ° | reach err | side err ° | \|bend err\| ° | distal rot vs ref ° |
|---|---|---|---|---|---|---|---|---|---|
| V000 (J0) | 12.11 / 30.18 | 7.32 / 18.42 | 18.41 / 46.86 | 15.28 | 14.19 | 0.058 | 18.37 | 15.58 | 24.57 |
| V100 | 11.39 / 28.67 | 7.55 / 17.53 | 16.74 / 40.68 | 15.78 | 13.36 | 0.058 | 18.46 | 15.58 | 24.34 |
| V010 | 12.07 / 29.62 | 7.20 / 17.74 | 18.05 / 46.50 | 15.02 | 14.19 | 0.051 | 18.37 | 14.07 | 23.23 |
| V001 | 12.11 / 30.18 | 7.64 / 18.85 | 17.66 / 43.75 | 15.95 | 14.19 | 0.058 | 18.77 | 15.58 | 24.11 |
| V110 | 11.30 / 28.40 | 7.19 / 17.13 | 16.78 / 40.71 | 14.99 | 13.36 | 0.051 | 18.46 | 14.07 | 22.90 |
| V101 | 11.39 / 28.67 | 7.50 / 17.56 | 16.34 / 40.46 | 15.69 | 13.36 | 0.058 | 18.81 | 15.58 | 23.74 |
| V011 | 12.07 / 29.62 | 7.47 / 18.57 | 17.23 / 43.67 | 15.58 | 14.19 | 0.051 | 18.77 | 14.07 | 22.63 |
| V111 (J1) | 11.30 / 28.40 | 7.11 / 17.35 | 16.30 / 40.12 | 14.85 | 13.36 | 0.051 | 18.81 | 14.07 | 22.17 |
| B1 (reference) | 11.89 / 29.96 | 7.64 / 18.98 | 18.33 / 48.98 | 15.95 | 13.78 | 0.054 | 23.22 | 14.26 | 24.26 |

## Static contrasts (J1 vs J0; sequence-cluster 95% intervals; sequences improved / worse)

| metric | total | main: direction | main: reach | main: side | material interactions |
|---|---|---|---|---|---|
| **forearm (°)** | **−2.11** [−3.60, −0.63], 25/12 | **−1.30** [−2.59, −0.07] (62%) | −0.20 [−0.60, +0.15] | **−0.61** [−1.12, −0.15] | none (direction × side +0.35 [+0.07, +0.64], 16% < 25%) |
| forearm, out-of-plane | **−5.78** [−9.61, −1.81] | **−4.10** (71%) | −0.53 | −1.16 | none |
| wrist (cm) | −0.81 [−1.93, +0.27] (not significant; MB10 [−1.27, −0.52]) | −0.74 (91%) | −0.07 | 0 (side cannot move the wrist) | none |
| wrist, out-of-plane | **−4.01** [−7.01, −0.84], 22/14 | **−3.60** [−6.74, −0.26] (90%) | −0.41 [−0.80, −0.02] | 0 | none |
| elbow (cm) | −0.21 [−0.75, +0.32] (not significant) | −0.07 | **−0.26** [−0.52, −0.02] | +0.11 | direction × reach −0.23 [−0.37, −0.10]; direction × side −0.35 [−0.53, −0.19] |
| upper (°) | −0.43 (not significant) | −0.13 | **−0.56** | +0.25 | direction × reach −0.49; direction × side −0.73 |
| reach MAE | **−0.007** [−0.015, −0.001] | 0 | −0.007 (100%) | 0 | — |
| \|bend err\| (°) | **−1.51** | 0 | −1.51 (100%) | 0 | — |
| side err (°) | +0.44 (not significant) | +0.07 | 0 | +0.37 | — |

**Elbow single swaps.** From J0, J1 side alone worsens the elbow (+0.32 cm [+0.15, +0.48], 31/37 sequences worse) and J1 direction alone worsens it (+0.22, not significant). Reverting each from J1 to J0 also worsens J1: direction −0.36, reach −0.38, side −0.08. The negative interactions mean **J1's direction is only elbow-compatible with J1's own reach and side.** This is the Worklog 77 coherence effect, reproduced inside the one-stage model.

## Temporal contrasts (5,504 pairs; Worklog 76–80 contract and thresholds)

| per pair | V000 (J0) | V100 | V010 | V001 | V110 | V101 | V011 | V111 (J1) | B1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| upper-arm residual (°) | **6.63** | 9.65 | 7.93 | 8.22 | 9.85 | 10.15 | 8.95 | 9.98 | 10.97 |
| proximal rotation residual (°) | 6.76 | 9.85 | 8.11 | 8.42 | 10.08 | 10.39 | 9.18 | 10.24 | 11.23 |
| forearm residual (°) | 9.45 | 11.01 | 10.00 | 10.25 | 11.65 | 11.91 | 10.38 | 12.22 | 12.16 |
| elbow motion residual (cm) | 3.21 | 4.68 | 3.85 | 3.98 | 4.78 | 4.92 | 4.34 | 4.84 | 5.32 |
| wrist motion residual (cm) | 5.79 | 7.92 | 6.28 | 5.79 | 8.17 | 7.92 | 6.28 | 8.17 | 8.08 |
| chain direction change (°; GT 8.02) | 7.88 | 11.36 | 7.88 | 7.88 | 11.36 | 11.36 | 7.88 | 11.36 | 10.94 |
| reach change (GT 0.023) | 0.035 | 0.035 | 0.042 | 0.035 | 0.042 | 0.035 | 0.042 | 0.042 | 0.039 |
| side change (°) | 13.1 | 14.5 | 13.1 | 15.8 | 14.5 | 16.8 | 15.8 | 16.8 | 17.6 |
| proximal discontinuity rate | 0.58% | 1.33% | 0.84% | 1.22% | 1.62% | 1.80% | 1.24% | 1.78% | 3.12% |
| elbow discontinuity rate | 0.60% | 1.34% | 0.87% | 1.16% | 1.71% | 1.67% | 1.36% | 1.82% | 3.22% |

**Contrasts, J1 vs J0** (sequence-cluster intervals; MB10/MB25 agree everywhere):

| metric | total | main direction / reach / side | swap from J0 (d / r / s) | revert from J1 (d / r / s) | interactions d×r / d×s / r×s / 3-way |
|---|---|---|---|---|---|
| **upper-arm residual (°)** | **+3.36** [2.90, 3.80], 37/37 | +1.98 (59%) / +0.52 / +0.81 | +3.02 / +1.31 / +1.59 | +1.04 / −0.17 / +0.13 | **−1.00 / −0.99 / −0.48** / +0.20 (all exclude 0) |
| proximal rotation residual | +3.49 | +2.02 / +0.55 / +0.86 (58% direction) | | | material d×r, d×s |
| elbow motion residual (cm) | +1.63 | +0.96 / +0.25 / +0.39 | +1.46 / +0.63 / +0.77 | +0.50 / −0.08 / +0.06 | −0.48 / −0.48 / −0.23 / +0.10 |
| wrist motion residual | +2.37 | +2.01 (85%) / +0.37 / 0 | | | d×r −0.24 (below 25%) |
| proximal discontinuities | +1.20 pp [0.7, 1.8] | +0.66 (55%) / +0.14 / +0.42 | +0.74 / +0.25 / +0.64 | +0.55 / −0.02 / +0.16 | none material |
| elbow discontinuities | +1.22 pp | +0.64 (52%) / +0.25 / +0.37 | | | none material |
| chain direction change | +3.48 | 100% direction | | | — |
| side change | +3.77 | +1.26 / 0 / **+2.51** (67% side) | | | d×s −0.38 |
| reach change | +0.007 | 100% reach | | | — |

**Every J1 component is individually jumpier than J0's:**
- direction changes +3.5°/pair more;
- side changes +2.5°/pair more;
- reach changes +0.007/pair more.

**Each one, grafted into J0's coherent arm, adds substantial upper-arm jitter.** The interactions are strongly sub-additive: the three single grafts sum to 5.9°/pair, against a joint total of 3.36. J1's components fluctuate partly together, so inside J1 their errors partly compensate.

**The discontinuity-rate cost alone has a direction owner** (55%, no material interaction). The residual cost, the primary temporal metric, does not.

## Owner classification (predeclared rule, `report.json → owners`)

| metric | owner | main shares (direction / reach / side) |
|---|---|---|
| **forearm error (primary static)** | **direction** | 0.62 / 0.09 / 0.29 |
| forearm, out-of-plane | direction | 0.71 / 0.09 / 0.20 |
| wrist, out-of-plane | direction | 0.90 / 0.10 / 0 |
| **upper-arm residual (primary temporal)** | **distributed** | 0.59 / 0.15 / 0.24, with material d×r and d×s |
| proximal rotation residual, elbow motion residual | distributed | ~0.58 / 0.15 / 0.25 |
| proximal and elbow discontinuities, forearm and wrist residuals, chain change | direction | 0.52–1.00 |
| side change | side | 0.33 / 0 / 0.67 |
| reach error, bend error, reach change | reach | — |
| elbow, wrist (overall), chain, upper and side error | not significant | — |

**Hypothesis C.** The primary static benefit has a single owner (direction), while the primary temporal cost is distributed.

## Regimes (J1 − J0; sequences improved / total)

| regime | elbow total (main d / r / s) | wrist total | forearm total | upper-arm residual total (main d / r / s) |
|---|---|---|---|---|
| near-straight | +0.46 (15/37; d +0.79, r −0.34, s 0) | +0.21 | −1.64 (25/37) | +1.81 (d +1.24, r +0.21, s +0.33) |
| moderate | **−0.45** (18/37; d −0.18, r −0.35, s +0.07) | −0.79 | −1.11 | +2.79 (d +1.88) |
| strong | **−0.79** (26/37; d −0.63, r −0.21) | −2.33 | **−5.27** (d −2.73, s −2.16) | +4.02 (d +2.23, s +1.12) |
| near-folded | **+1.05** (16/31; d +0.43, r +0.07, **s +0.54**) | +1.09 | +0.44 | **+6.50** (d +3.22, r +1.03, s +2.14) |
| near-plane | **+0.53** (13/37; **d +0.61**) | **+1.17** (d 100%) | +0.02 | +2.91 (d +2.02) |
| intermediate | +0.04 | −0.66 | −1.80 | +3.78 (d +2.49) |
| out-of-plane | **−1.73** (24/36; d −1.38, r −0.89, s +0.53) | **−4.01** (d 90%) | **−5.78** | +3.96 (d +1.81, r +0.86, s +1.19) |

**Elbow means by regime (cm):**

| arm | moderate | near-folded |
|---|---:|---:|
| B1 | 6.75 | 11.28 |
| J0 | 7.62 | 6.67 |
| J1 | 7.17 | 7.72 |

**What the regimes show:**
- **J1's direction is a trade, not a uniform improvement.** It wins strongly out of plane and in strong bends, and loses near the image plane (wrist +1.17, elbow +0.53 cm).
- **Near-folded.** J0's strength there is lost by J1's side and direction. It is also where J1's temporal cost is largest (+6.5°/pair).
- **Moderate bends.** This is the regime where both models trail B1. RGB components reduce the gap to B1 rather than causing it, which points to the shared joint representation or single-frame information, not a J1 factor.

## Oracle context (read-only, Worklog 80)

The Worklog 80 evaluation-only GT decomposition still applies:
- replacing J1's direction with GT removes 2.18 cm of elbow error, against −0.94 cm for reach and −0.24 cm for side;
- for J0 the same figures are −1.90 / −1.06 / +0.13 cm.

**These single swaps are not additive.** C111 is not an upper bound for arms with a different direction. No GT quantity selects any candidate.

## Validation (selection split; separate)

| arm | elbow (cm) | wrist (cm) |
|---|---:|---:|
| V000 (J0) | 6.91 | 11.54 |
| V111 (J1) | 5.43 | 9.67 |

**Main effects on validation:**
- elbow total −1.49 cm (direction −0.92, reach −0.48, side −0.12);
- wrist total −1.87 cm (direction −1.85).

On the split that chose the J epochs the J1 elbow gain is much larger than on test (−0.21 cm). That is the same optimism seen in Worklogs 78–80, so the test result is used for classification.

## Real-scene review exports

- **Drawn arms (compact contrast set):**
  - J0 (V000) orange, J1 (V111) cyan;
  - V100 (J1 direction only) magenta, V011 (J1 reach + side only) blue;
  - GT green, E_GT yellow.
- Rendering all eight would be unreadable. Captions carry elbow, wrist and forearm errors for all eight candidates and B1, plus the next pair's upper-arm residual for all eight, the exact sample ID, GT bend and GT f.
- **Renders:** authoritative BaseRig rest, Worklog 76–80 cameras, macOS Blender 5.1.2. Blender FK end vs export: 9.5e-7 cm.

| export | row(s) |
|---|---|
| `owner_downtown_bar_00_514`, `_775`, `owner_downtown_sitOnStairs_00_325`, `owner_flat_packBags_00_446`, `owner_downtown_runForBus_01_185` | owner frames |
| `largest_j1_vs_j0_wrist_improvement` | `downtown_walkBridge_01` a0 #946 |
| `largest_j1_vs_j0_elbow_regression` | `downtown_runForBus_01` a1 #583 |
| `largest_j1_vs_j0_temporal_regression_strip` | `downtown_runForBus_01` a1 #578–593 |
| `strong_near_folded` (largest J0 − B1 elbow gain, GT bend ≥ 90°) | `downtown_walking_00` a0 #365 |
| `strongly_out_of_plane` (largest J1 − J0 wrist gain, \|f_GT\| ≥ 0.5) | `downtown_arguing_00` a0 #260 |
| `moderate_bend_failure` (median J1-worse-than-B1 moderate row) | `downtown_downstairs_00` a0 #709 |

**Owner frames (wrist / elbow cm):**

| row | J0 | V100 (J1 direction) | V011 (J1 reach + side) | J1 | B1 |
|---|---|---|---|---|---|
| #514 | 9.1 / 11.3 | 7.9 / **7.5** | 13.4 / 13.1 | 12.6 / 9.2 | 26.1 / 29.6 |
| #185 | 27.0 / 15.8 | **10.5 / 7.7** | 27.4 / 15.9 | 7.9 / 6.2 | 9.1 / 7.1 |
| #325 | **31.9 / 14.6** | 56.9 / 29.7 | 36.6 / 22.9 | 65.8 / 36.6 | 67.7 / 42.4 |

- #185 and #325 are pure direction stories: the J1 direction fixes #185 and causes #325.
- **QE images-only copy:** `~/animcv-output/81_joint_factor_attribution_review/` (29 PNGs). The renders illustrate only.

## Architecture classification — Hypothesis C (distributed interactions)

**Answer to the completion question.** J1's useful visual gains and its temporal regression do not originate from one predicted quantity:
- the static gains (forearm, out-of-plane wrist and forearm) are owned by J1's chain direction, with side as a secondary contributor;
- the temporal regression is distributed across direction, reach and side;
- strong, sub-additive interactions mean that no single factor reproduces or removes it.

**Why an isolated correction is architecturally unjustified:**
1. **Swapping direction alone, to keep the static gain without the jitter, removes only 31% of J1's residual cost.** It also gives up the direction-owned static gain. Swapping toward J0's direction (V011) loses the out-of-plane wrist gain entirely (wrist 12.07 cm).
2. **Grafting J1's direction alone onto J0's reach and side (V100) is not a clean middle ground.** It keeps most of the static wrist and forearm gain (11.39 cm, 16.7°). But it pays 90% of the temporal penalty (9.65 vs 9.98 °/pair), breaks coherence and worsens the elbow (7.55 vs 7.32 cm).
3. **Every J1 component is individually less temporally reliable than J0's.** Direction, side and reach all change more per frame, so the RGB-conditioned representation is jumpier as a whole.

**What remains unresolved.** This is an architectural question about how per-frame visual evidence enters the connected-arm representation as a whole. It is not a question about one geometric quantity.
- The direction quantity is the largest single contributor to both sides of the trade-off: the static gain, the chain-change jitter and the discontinuities. Visual chain-direction reliability is therefore a necessary part of any next hypothesis.
- It is not a sufficient narrow target, because the remaining cost lives in the coupled reach/side behavior of the same visually conditioned representation.

J0 and J1 are preserved as separate historical candidates. No hybrid, selector, smoothing, temporal model or retraining was introduced, and nothing proceeds automatically.

## Limits of the factorial interpretation

- **The mixed candidates are off-distribution.** They combine components the models never produced together, so their errors measure incompatibility as much as component quality. That is why both swap-from-J0 and revert-from-J1 are reported, with interactions.
- **Effects are averages over the 3DPW test population and its sequences.** They do not imply independent error processes or causal mechanisms inside the network.
- **The test split has guided decisions since Worklog 73.** This batch selects nothing on it, but its findings are controlled diagnostics, not independent generalization evidence.

## Focused tests

- **`tests/test_framepose_joint_factor_attribution.py` (4 tests):**
  - complete factor assignment;
  - V000 is J0's measure and V111 reproduces J1's arm exactly;
  - each factor comes from its source, with Worklog 62 perpendicularization, fixed bone lengths and IK/FK;
  - degenerate-side fallback propagation;
  - factorial contrasts on additive and pure-interaction cases (main effects, swaps, reverts, two/three-way terms);
  - the owner rule on a synthetic direction-owned metric;
  - static contracts: gate order, decisions declared before `main`, no training/smoothing, no GT source in the factor module, no production imports.
- **Server run (`animcv-framepose:cuda118`):** with the Worklog 61/62/73–80 focused tests, **85 passed**.
- **Full regression:** not run, because production behavior is unchanged.

## Reproduction

    docker run --rm -v $HOME/AnimCV:/workspace:ro -v $HOME/animcv-output:/output \
      -v $HOME/animcv-data/datasets/3dpw:/data/3dpw_raw:ro -e PYTHONPATH=/workspace/src:/workspace/scripts \
      -w /workspace animcv-framepose:cuda118 scripts/run_framepose_joint_factor_attribution.py \
      --rig examples/BaseRig.fbx --rest examples/e2e_demo/baserig_blender_rest.json \
      --calibration examples/e2e_demo/framepose_fk_calibration.json --chains examples/e2e_demo/framepose_ik_chains.json \
      --bank /output/framepose/bank_3dpw_paired_v2.json --h0-dir /output/framepose/context_h0_v2 --threedpw-raw /data/3dpw_raw \
      --w73-dir /output/framepose_external_visual_supervision/transfer --w76-dir /output/framepose_visual_full_arm_ik \
      --w80-joint-dir /output/framepose_joint_arm_geometry --w80-eval-dir /output/framepose_joint_arm_geometry_eval \
      --out-dir /output/framepose_joint_factor_attribution
    scripts/export_framepose_joint_factor_cases.py ... --attribution-dir /output/framepose_joint_factor_attribution \
      --out /output/framepose_joint_factor_attribution/review_cases.json
    Blender -b -P scripts/review_framepose_visual_arm_ownership_blender.py -- --rig examples/BaseRig.fbx \
      --rest examples/e2e_demo/baserig_blender_rest.json --cases review_cases.json --out-dir review --caption-python python3
