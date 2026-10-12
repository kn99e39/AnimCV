# Worklog — Visual Direction Noise vs Mixed-Owner Arm Geometry Attribution (2026-10-12)

**Question.** Worklog 76 found that the E1 handoff makes the upper arm move more erratically between frames and gives only a narrow elbow gain. Is that excess motion intrinsic to the E1 visual chain-direction time series? Or does it come from pairing that direction with independently estimated H0 reach and bend-side geometry?

**Answer.** Mostly the pairing. **CASE B (mixed-owner inconsistency dominates), with one important qualification.**

**The E1-specific penalty is a pairing effect:**
- The E1 direction alone is not detectably noisier than H0's direction, judged against true motion. Its chain-direction motion residual exceeds H0's by only +0.35°/pair, and the 95% interval [−0.25, +0.99] includes 0.
- The E1-vs-H0 upper-arm motion penalty is +2.16°/pair [1.63, 2.72] under H0 reach and side.
  - Replacing H0 reach with GT reach removes 43% of it, and replacing the bend side removes 48%. The two swaps are roughly additive.
  - Replacing both removes 86%, in 37/37 sequences, leaving +0.30° [−0.24, +0.89].
- With an H0 direction, the same oracle swaps barely change temporal stability (−0.39°). The penalty is therefore specific to pairing E1 with H0, not a general benefit of better reach/side.
- The static elbow follows the same pattern. E1's elbow change vs H0 goes from −0.61 cm (interval includes 0, 20/37 sequences) under H0 reach/side to −1.71 cm [−2.62, −0.92] under GT reach/side, improving in 31/37 sequences.

**The qualification: a large depth-error floor shared by both directions.**
- The depth-oracle direction (GT forward fraction on the same detector plane, with H0 reach/side) is temporally better than even H0 itself: upper-arm residual 7.42° vs C000's 8.81°.
- Its chain-direction residual is 2.14°, against 9.1–9.4° for H0 and E1.
- So the H0 reach/side are not incompatible with an accurate direction. They are incompatible with E1's direction errors, which do not move together with H0's own errors.
- Better forward depth would also remove the penalty, and would lower the direction-error floor that H0 and E1 share.

## Starting state and files

- **Start:** `arch/single_frame_first` at `4be26ea2d47ab30cdace471f76fc1e0f163accde` (local = remote), with a clean worktree.
- **Commits:**
  - `883b35a`: ownership module, runner, tests
  - `f430c38`: review export + Blender strips
  - the commit adding this worklog
- **Files added (evaluation only, not imported by production):**
  - `src/retarget/framepose_visual_arm_ownership.py`
  - `scripts/run_framepose_visual_arm_ownership.py`
  - `scripts/export_framepose_visual_arm_ownership_cases.py`
  - `scripts/review_framepose_visual_arm_ownership_blender.py`
  - `tests/test_framepose_visual_arm_ownership.py`
  - this worklog
- **Unchanged:**
  - every historical module, checkpoint and report (Worklogs 60–76);
  - frozen H0, FrameBank and the BaseRig rest/alignment;
  - production FramePose, FK, IK and the Blender writer.
- **How the new module plugs in:** it only builds the Worklog 62 `ChainMeasure` whose reach fraction and observed bend side come from the selected source. The arm is then solved by the unchanged Worklog 76 `full_arm`: Worklog 61 `solve_two_bone`, the Worklog 62 `_runtime_side` perpendicularization, zero-twist `local_rotations_for_directions`, and the `chain_forward_kinematics` check.

## Frozen artifact identities

**Pinned and checked at runtime:**

| artifact | sha256 |
|---|---|
| Worklog 73 `report.json` | `1c05d50a…` |
| Worklog 73 `rows_heldout.json` | `4e17bd77…` |
| Worklog 75 `report.json` | `89218bab…` |
| Worklog 75 `rows_test.json` | `dc39b22b…` |
| Worklog 76 `report.json` | `dad46288…` |
| Worklog 76 `rows_test.json` | `d6b339dc…` |

**Also checked:**
- H0 validation predictions (prefix `88a007fc`).
- H0 test predictions, which must equal the hash Worklog 76 recorded (`6df04c20…`).
- The H0 materialization's FrameBank digest.
- The rest snapshot against the current `BaseRig.fbx`.
- The target chain, which must equal Worklog 76's.

**Outputs (`~/animcv-output/framepose_visual_arm_ownership/` on LabServer63):**

| file | sha256 |
|---|---|
| `report.json` | `e86b7de6…` |
| `rows_test.json` | `c39a834c…` |
| `pairs_test.json` | `b83ca38b…` |
| `review_cases.json` | `0a08b705…` |
| `review/report.json` | `7d438599…` |

## Candidate-factor definitions

| candidate | direction | reach | bend side | role |
|---|---|---|---|---|
| C000 | H0 | H0 | H0 | runtime, equals Worklog 76 F0 |
| C100 | E1 | H0 | H0 | runtime, equals Worklog 76 F2 |
| C010 | H0 | GT | H0 | counterfactual |
| C110 | E1 | GT | H0 | counterfactual |
| C001 | H0 | H0 | GT | counterfactual |
| C101 | E1 | H0 | GT | counterfactual |
| C011 | H0 | GT | GT | counterfactual |
| C111 | E1 | GT | GT | counterfactual |
| C_GT_F | GT f on E1's detector plane | H0 | H0 | depth-oracle diagnostic |

**How the factors are applied:**
- **Directions:** H0 is the frozen H0 shoulder→wrist direction. E1 and GT-f use the Worklog 64–73 `reconstruct_direction(f, observed_plane(detector 2D))`, then the Worklog 61/62 transform A·Rz(−yaw_H0).
- **Reach and side:** taken from the Worklog 62 canonical chain measures (`measure_chain`, same yaw_H0 transform). The selected side is re-perpendicularized to the candidate direction with `_runtime_side`. A degenerate source would use the rest-plane fallback; none occurred.
- **Shared rig:** root, L1/L2, alignment, clamps and quaternion conventions are identical for every candidate.

**GT reach, GT side and GT f are evaluation-only oracles.**
- They exist only in this runner and never enter training, selection, runtime or production.
- C000 and C100 are built from H0 alone: the test checks that `owned_measure(h0, gt, "h0", "h0")` returns the H0 measure object itself.

## Gate 1 — Worklog 76 reproduction (before any counterfactual)

**Population:** 6,003 rows (set-equal to Worklog 76, same sample ids, positions and bank order) in 37 sequences, with 5,504 adjacent pairs. The pair list is identical to Worklog 76's, using the one-bank-stride adjacency with no pairs across holes.

| check (max over all rows/pairs) | value | limit |
|---|---:|---:|
| C000 vs F0, C100 vs F2: elbow/wrist positions (cm) | 0 | 1e-9 |
| local rotations (component, sign-aligned) | 0 | 1e-9 |
| wrist / elbow errors vs Worklog 76 | 0 | 1e-9 |
| reference arm (positions, rotations, E_GT) | 0 | 1e-9 |
| jump thresholds recomputed vs Worklog 76 | 0 | 1e-12 |
| candidate-discontinuity rates, all 4 metrics, C000/C100 vs F0/F2 | 0 | exact |

Status, side status and clamp match on every row. **Gate 1 passes.**

The jump thresholds are Worklog 76's, unchanged and not re-chosen:

| metric | threshold |
|---|---:|
| proximal rotation | 34.6° |
| distal rotation | 49.1° |
| wrist | 31.9 cm |
| elbow | 16.0 cm |

## Gate 2 — oracle-geometry contract

| check | value |
|---|---:|
| GT direction + GT reach + GT side vs the Worklog 62 reference arm (cm / rotation component) | 0 / 0 |
| its elbow error / wrist error vs E_GT (cm) | 0 / 3.2e-14 |
| GT direction + H0 reach/side vs Worklog 75 T4 endpoint error (cm) | 5.3e-14 |
| C_GT_F vs Worklog 75 T3 (GT f oracle) endpoint error (cm) | 5.0e-14 |
| C000 / C100 vs Worklog 75 T0 / T2 (cm) | 4.3e-14 / 4.3e-14 |
| reach-source assignment (all factorial arms) | 0 |
| side-source assignment (recomputed `_runtime_side`, exact equality) | 0 mismatches |
| unavailable arms (any candidate or contract arm) | 0 |

**Gate 2 passes.** No candidate became unavailable or clamped, so all comparisons run on the full 6,003-row / 5,504-pair common-valid population.

## Static metrics (6,003 rows)

| candidate | wrist cm mean / p95 | elbow cm mean / p50 / p95 / max | upper ° | forearm ° | \|bend-angle err\| ° | bend-plane ° | reach err | proximal / distal rot vs ref ° |
|---|---|---|---:|---:|---:|---:|---:|---|
| C000 | 14.88 / 37.08 | 8.25 / 6.09 / 21.54 / 49.9 | 17.42 | 22.57 | 14.26 | 26.68 | 0.05 | 17.8 / 24.6 |
| C100 | 11.89 / 29.96 | 7.64 / 6.11 / 18.98 / 44.2 | 15.95 | 18.33 | 14.26 | 26.56 | 0.05 | 16.3 / 24.3 |
| C010 | 14.33 / 35.63 | 8.01 / 5.82 / 22.63 / 52.9 | 16.90 | 20.51 | 0 | 26.68 | 0 | 17.3 / 16.3 |
| C110 | 11.16 / 29.75 | 7.00 / 5.56 / 17.30 / 49.5 | 14.62 | 16.41 | 0 | 26.56 | 0 | 15.0 / 15.8 |
| C001 | 14.88 / 37.08 | 8.32 / 6.50 / 21.51 / 43.8 | 17.43 | 19.99 | 14.26 | 12.06 | 0.05 | 17.8 / 19.2 |
| C101 | 11.89 / 29.96 | 7.12 / 5.68 / 17.96 / 42.6 | 14.83 | 15.06 | 14.26 | 8.55 | 0.05 | 15.2 / 17.5 |
| C011 | 14.33 / 35.63 | 7.83 / 5.90 / 21.87 / 48.5 | 16.42 | 17.72 | 0 | 12.06 | 0 | 16.8 / 8.7 |
| C111 | 11.16 / 29.75 | **6.12** / 4.77 / 16.27 / 45.6 | **12.74** | **12.84** | 0 | **8.55** | 0 | 13.1 / 6.4 |
| C_GT_F | **4.53** / 12.29 | **5.75** / 4.72 / 13.84 / 47.0 | **11.95** | **12.40** | 14.26 | 24.14 | 0.05 | 12.3 / 22.9 |

**Paired vs C000 (rows improved, sequences improved / regressed):**

| candidate | elbow | upper | forearm | wrist |
|---|---|---|---|---|
| C100 | −0.61 (53%, 20/17) | −1.47° | −4.24° (32/5) | −2.99 (32/5) |
| C110 | −1.25 (57%, 25/12) | −2.80° | −6.16° | −3.72 (34/3) |
| C101 | −1.14 (57%, 29/8) | −2.59° | −7.51° | −2.99 |
| C111 | −2.13 (63%, 31/6) | −4.68° | −9.74° (35/2) | −3.72 |
| C010 / C001 / C011 (H0 direction) | −0.24 / +0.06 / −0.42 | −0.52 / +0.01 / −1.00° | −2.07 / −2.58 / −4.86° | −0.55 / 0 / −0.55 |
| C_GT_F | −2.50 (65%, 34/3) | −5.47° | −10.17° (37/0) | −10.35 (37/0) |

**Paired vs C100 (elbow):**

| candidate | change | rows improved | sequences improved / regressed |
|---|---:|---:|---|
| C110 | −0.64 | 60% | 31/6 |
| C101 | −0.52 | 61% | 29/8 |
| C111 | −1.52 | 64% | 35/2 |
| C_GT_F | −1.89 | 64% | 34/3 |

The full per-candidate p50/max/sequence tables are in `report.json → static`.

## Elbow attribution (factorial, 6,003 rows)

**E1-minus-H0 direction effect on elbow error under each reach/side ownership** (cm; sequence-cluster 95% interval; sequences where E1 is worse):

| ownership | E1 − H0 elbow | interval | sequences E1 worse |
|---|---:|---|---:|
| H0 reach, H0 side (Worklog 76) | −0.61 | [−1.56, +0.18] | 17/37 |
| GT reach, H0 side | −1.01 | [−2.07, −0.18] | 13/37 |
| H0 reach, GT side | −1.20 | [−1.90, −0.56] | 8/37 |
| GT reach, GT side | **−1.71** | [−2.62, −0.92] | **6/37** |

**Change of that effect when ownership changes** (sequence-cluster interval; MB10 and MB25 agree):

| swap | change | sequences reduced |
|---|---|---:|
| reach | −0.39 [−0.61, −0.20] | 29/37 |
| side | −0.59 [−1.09, −0.04] | 30/37 |
| both | −1.10 [−1.63, −0.64] | 32/37 |
| three-way interaction | −0.12 [−0.29, +0.04] | — |

**Reach and side effects under each direction** (GT-owned minus H0-owned elbow, cm):

| direction | reach | side | both | reach × side |
|---|---:|---:|---:|---:|
| H0 | −0.24 | +0.06 | −0.42 | −0.24 |
| E1 | −0.64 | −0.52 | −1.52 | −0.36 |

**Interpretation:**
- **The direction-only substitution reproduces the Worklog 76 elbow result:** −0.61 cm, not robust.
- **The narrow Worklog 76 elbow gain is owned mostly by H0's bend side and reach.** Accurate reach and side help an E1 arm two to three times more than an H0 arm. Correct side evidence alone does nothing for H0's own direction (+0.06).
  - This is an E1-specific compatibility effect, not general reach/side quality.
- **Near-additivity.** Over all rows, the reach and side contributions are close to additive (three-way −0.12 cm, interval includes 0).
- **Near-folded arms are strongly non-additive.** GT bend ≥ 90°, 750 rows:

  | ownership | E1 − H0 elbow (cm) |
  |---|---:|
  | H0 reach/side | +0.47 |
  | GT reach | −0.96 |
  | GT side | −2.23 |
  | both | −4.35 |

  The implied three-way term is −1.69 cm. C111 vs C011 elbow improves in 23/31 sequences, against 11/31 for C100 vs C000. Wrist accuracy plus a correct elbow in folded arms needs both reach and side to be right.

## Temporal attribution (5,504 adjacent pairs)

**Motion residual** = a candidate's frame-to-frame motion minus the GT-anchored reference arm's motion on the same pair:
- Rotations: Worklog 61 quaternion distance between the two relative rotations.
- Unit directions: |Δcandidate − Δreference| in degrees.
- Joints: |Δcandidate − Δreference| in cm.

A candidate that moves like the reference has zero residual, whatever its static offset. Larger motion is not penalized when the reference moves the same way.

**Mean per pair:**

| metric | ref | C000 | C100 | C010 | C110 | C001 | C101 | C011 | C111 | C_GT_F |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| upper-arm direction residual (°) | 0 | 8.81 | **10.97** | 9.12 | 10.35 | 8.76 | 9.88 | 8.42 | 8.72 | **7.42** |
| proximal rotation residual (°) | 0 | 9.01 | 11.23 | 9.33 | 10.62 | 8.88 | 10.09 | 8.56 | 8.93 | 7.59 |
| elbow motion residual (cm) | 0 | 4.27 | 5.32 | 4.42 | 5.02 | 4.24 | 4.79 | 4.08 | 4.23 | 3.60 |
| chain-direction residual (°) | 0 | 9.08 | 9.44 | 9.08 | 9.44 | 9.08 | 9.44 | 9.08 | 9.44 | **2.14** |
| forearm direction residual (°) | 0 | 12.95 | 12.16 | 11.05 | 10.93 | 11.31 | 10.78 | 9.10 | 9.10 | 8.04 |
| bend-side residual (°) | 0 | 17.40 | 17.88 | 17.40 | 17.88 | 7.20 | 6.80 | 7.20 | 6.80 | 15.50 |
| wrist motion residual (cm) | 0 | 7.76 | 8.08 | 7.51 | 7.75 | 7.76 | 8.08 | 7.51 | 7.75 | 2.89 |
| proximal rotation change (°) | 7.04 | 9.23 | 11.77 | 9.81 | 11.43 | 9.59 | 11.01 | 9.57 | 10.25 | 9.92 |
| proximal discontinuity rate | — | 1.29% | **3.12%** | 1.82% | 2.71% | 1.44% | 2.18% | 1.53% | **1.78%** | 1.64% |
| elbow discontinuity rate | — | 1.34% | 3.22% | 1.84% | 2.91% | 1.54% | 2.36% | 1.60% | 1.93% | 1.65% |

**E1 direction penalty (C1rs − C0rs) and how it changes with ownership** (sequence-cluster 95% intervals; MB10 and MB25 intervals agree in sign and exclusion of 0 everywhere except the three-way terms):

| quantity | H0 reach/side (Worklog 76) | change with GT reach | change with GT side | change with both | three-way | remaining under GT reach + side |
|---|---|---|---|---|---|---|
| upper-arm residual (°) | +2.16 [1.63, 2.72] (34/37 sequences) | −0.93 [−1.13, −0.72] (35/37) | −1.04 [−1.32, −0.78] (35/37) | **−1.86 [−2.17, −1.54] (37/37)** | +0.11 [−0.02, +0.25] | **+0.30 [−0.24, +0.89]** (21/37 positive) |
| proximal rotation residual (°) | +2.22 [1.68, 2.79] | −0.93 | −1.02 | −1.86 [−2.17, −1.54] | +0.10 [−0.03, +0.23] | +0.37 [−0.21, +0.98] |
| elbow motion residual (cm) | +1.05 [0.79, 1.32] | −0.45 | −0.50 | −0.90 [−1.05, −0.75] (37/37) | +0.05 [−0.01, +0.12] | +0.15 [−0.12, +0.43] |
| proximal discontinuity rate | +1.84 pp [1.2, 2.6] | −0.94 pp | −1.09 pp | −1.58 pp [−2.2, −1.0] | +0.5 pp [0.0, 0.9] | +0.25 pp [−0.2, +0.8] |
| chain-direction residual (°) | +0.35 [−0.25, +0.99] | 0 | 0 | 0 | 0 | +0.35 (same: reach/side cannot change the chain) |

**Under H0 direction the oracle swaps hardly help** (upper-arm residual):

| swap | change |
|---|---:|
| GT reach | +0.31° |
| GT side | −0.05° |
| both | −0.39° |

Under E1 direction the same swaps give −0.62° (reach), −1.09° (side) and −2.25° (both).

**The reductions are not just "closer to the GT reference".** The penalty compares E1 with H0 under the same oracle inputs, so a generic oracle advantage cancels in the contrast. The H0-direction arms confirm that GT reach/side alone do not make an arm move like the reference.

## A–E: intrinsic noise vs owner interaction

**A. Intrinsic visual direction noise** — **not material at the E1-specific level.**
- Under GT reach and GT side, E1 is still +0.30°/pair worse than H0 in upper-arm residual: 14% of the Worklog 76 penalty, with an interval that includes 0.
- Its proximal discontinuity rate is 1.78% vs 1.53%, and that interval also includes 0.
- The E1 chain direction's own motion residual is not significantly larger than H0's (+0.35°/pair, interval includes 0), even though its forward fraction changes more per step (|Δf| 0.133 vs 0.109).
- A remainder persists in a minority of sequences. Under GT/GT the remaining upper-arm penalty is still ≥ +2.2° in `flat_guitar_01` a0, `downtown_arguing_00` a0, `downtown_stairs_00` a0, `flat_packBags_00` a0 and `downtown_sitOnStairs_00` a0.
- Intrinsic E1 direction noise is therefore sequence-local, not systematic.

**B. Reach incompatibility** — material:
- removes 43% of the penalty [interval excludes 0], in 35/37 sequences;
- removes 0.39 cm of the elbow effect.

**C. Bend-side incompatibility** — material and slightly larger:
- removes 48% of the penalty, in 35/37 sequences;
- removes 0.59 cm of the elbow effect.

**D. Combined interaction** — both swaps together remove 86% (37/37).
- Pooled, the two act nearly additively: the three-way term on the upper-arm residual is +0.11°, interval includes 0.
- For the static elbow in near-folded arms, they are strongly synergistic (three-way −1.69 cm).

**E. Residual depth-prediction error** — **yes**:
- C_GT_F (perfect forward depth, the same detector plane, H0 reach and side) removes the whole E1 penalty and more. Its upper-arm residual is −3.56° vs C100 [−3.98, −3.15] (37/37 sequences) and −1.39° vs C000 [−1.70, −1.12] (35/37).
- Its chain-direction residual falls to 2.14°/pair.
- So H0 reach/side are compatible with an accurate direction; the incompatibility grows with E1's depth error.
- The detector observed plane leaves a small residual (2.14° chain residual, and 9.65° upper-arm change vs the reference's 6.88°). It is not the main source.

**Statistical independence is not claimed.** The interaction shows that E1 direction errors and H0 reach/side errors do not co-vary the way H0's own direction/reach/side errors do. That is consistent with structural mixed-owner incompatibility, but it does not prove the underlying error processes are independent.

## Sequence and regime results

**Upper-arm residual penalty E1 − H0 (°/pair) and elbow:**

| regime | pairs | H0 reach/side | GT reach | GT side | both | C_GT_F − C100 | C100−C000 elbow (sequences improved) | C111−C011 elbow (sequences improved) |
|---|---:|---:|---:|---:|---:|---:|---|---|
| near-straight [0,15) | 557 | +1.15 | +0.34 | +0.93 | +0.23 | −2.09 | −0.12 (21/37) | −0.22 (23/37) |
| moderate [15,45) | 2,039 | +1.79 | +0.71 | +0.75 | −0.08 | −3.37 | −1.19 (26/37) | −1.57 (30/37) |
| strong [45,90) | 1,202 | +2.59 | +2.36 | +1.81 | +1.71 | −4.02 | −0.44 (20/37) | −1.58 (27/37) |
| near-folded ≥ 90 | 579 | **+4.11** | +3.45 | +1.89 | +0.80 | −5.05 | **+0.47 (11/31)** | **−4.35 (23/31)** |
| near-plane \|f_GT\| < sin 10° | 1,525 | +1.43 | +0.59 | +0.81 | −0.01 | −2.52 | +0.24 (17/37) | −0.87 (27/37) |
| intermediate | 1,645 | +2.23 | +1.35 | +1.45 | +0.60 | −3.43 | −0.09 (19/37) | −1.08 (25/37) |
| out-of-plane \|f_GT\| ≥ 0.5 | 1,101 | +3.75 | +3.10 | +1.74 | +1.61 | −5.01 | −2.78 (16/36) | −4.03 (25/36) |

**Near-folded is the core problem.** The Worklog 76 pattern sits here:
- C100 proximal discontinuities are 8.1%, against 2.9% for C000.
- The E1 penalty is 4.11°/pair, and its elbow gets worse.
- With GT side the discontinuity rate drops to 2.8%. With both swaps it drops to 2.4% and the penalty falls to 0.80°.
- So in folded arms the H0 bend side is the larger owner, and side and reach together turn an elbow regression into a −4.35 cm gain.

**Strong-bend and out-of-plane rows keep a larger remainder** under both oracles: +1.71° and +1.61°. This is where E1's intrinsic direction error is most visible.

**Sequence ownership:**
- In C100 vs C000, `downtown_bar_00` a0 owns 64–71% of the net elbow gain in the moderate, near-straight and out-of-plane regimes. With GT reach/side, no single sequence owns more than 54% (near-folded); it is 14–49% elsewhere.
- The worst remaining C111−C011 elbow regressions are `downtown_stairs_00` a0 (out-of-plane +6.3 cm/row, strong +5.0) and `downtown_cafe_00` a0.

**Per sequence (37):**
- The upper-arm penalty under H0 reach/side is positive in 34/37 sequences. GT reach + side reduces it in 37/37.
- C_GT_F is temporally better than C000 in 35/37 sequences.
- The Worklog 76 problem sequences keep the largest GT/GT remainders: `flat_guitar_01` +7.14 → +2.25, `downtown_stairs_00` +5.21 → +3.95, `flat_packBags_00` +3.92 → +2.84, `downtown_arguing_00` a0 +5.75 → +3.88.
- `downtown_bar_00` a0 drops from +3.39 to +0.05.

**Worklog 76 owner frames (wrist cm / elbow cm):**

| row | C000 | C100 | C110 | C101 | C111 | C_GT_F |
|---|---|---|---|---|---|---|
| `downtown_bar_00` a0 #514 (GT bend 116°, reach H0 0.97 / GT 0.53) | 44.7 / 5.9 | 26.1 / **29.6** | 7.5 / 26.5 | 26.1 / 26.1 | 7.5 / **6.7** | 24.2 / 22.5 |
| `downtown_bar_00` a0 #775 | 81.0 / 38.8 | 8.3 / 4.5 | 8.2 / 6.2 | 8.3 / 3.1 | 8.2 / 4.1 | 7.8 / 5.7 |
| `downtown_runForBus_01` a0 #185 | 28.7 / 16.1 | 9.1 / 7.1 | 8.1 / 2.9 | 9.1 / 8.0 | 8.1 / 4.4 | 4.8 / 7.0 |
| `downtown_sitOnStairs_00` a0 #325 | 4.9 / 5.2 | 67.7 / 42.4 | 67.3 / 42.2 | 67.7 / 41.3 | 67.3 / 41.1 | 5.1 / 2.4 |
| `flat_packBags_00` a0 #446 | 8.1 / 3.5 | 30.2 / 25.4 | 27.0 / 23.1 | 30.2 / 23.3 | 27.0 / 20.4 | 7.0 / 3.4 |

- **#514 is the clearest interaction.** Fixing only the reach (C110) fixes the wrist but not the elbow (26.5). Fixing only the side (C101) does nothing. Fixing both (C111) brings the elbow to 6.7 cm.
- **#325 and #446 are pure E1 depth errors** that no reach/side swap can fix: E1 f −0.93 vs GT +0.49, and −0.88 vs GT −0.16. C_GT_F fixes them.

## Real-scene review exports

- **Renders:** Worklog 76 rig/camera contract, BaseRig, Blender 5.1.2 Workbench. The object rotation is A·Rz(yaw_H0)·A⁻¹, proven against Worklog 60 in Worklog 76. Blender FK end vs export: 9.5e-7 cm.
- **Drawn arms:**
  - C000 blue, C100 magenta;
  - C111 cyan (E1 + GT reach/side);
  - C_GT_F orange;
  - GT-anchored reference green, E_GT yellow wire.
- **Captions:** exact sequence/frame/bank IDs, GT bend, H0/GT reach, H0/E1/GT chain f, and the elbow and wrist errors of all nine candidates. Strip panels also show the next pair's upper-arm motion residual for all nine.

| export (`review/`) | row(s) |
|---|---|
| `near_folded_wrist_improves_elbow_regresses_{side,front}.png` | fixed rule: GT bend ≥ 90°, max min(wrist gain, elbow loss) |
| `largest_elbow_regression_{side,front}.png` | max C100−C000 elbow |
| `largest_upper_arm_discontinuity_strip.png` (4 frames) | `downtown_arguing_00` a0 #690–#705 |
| `e1_stable_sequence_office_phoneCall_00_strip.png` | `office_phoneCall_00` a1 #176–#191 (lowest sequence penalty, −0.55°) |
| `e1_unstable_sequence_flat_guitar_01_strip.png` | `flat_guitar_01` a0 #200–#215 (highest penalty, +7.14°) |
| `w76_owner_downtown_bar_00_514`, `_775`, `w76_owner_downtown_sitOnStairs_00_325`, `w76_owner_downtown_runForBus_01_185`, `w76_owner_flat_packBags_00_446` | Worklog 76 owner frames |

**Where to find them:**
- Per-frame side and front PNGs are also kept for every strip member.
- Server copy: `review/`.
- QE images-only copy: `~/animcv-output/77_visual_arm_ownership_review/` (41 PNGs).
- The renders illustrate; the numbers above are the evidence.

## Architecture classification — CASE B (with a CASE C qualification)

**Predeclared criteria** (`report.json → decision_inputs`, declared in the runner before any counterfactual statistic):

| criterion | result |
|---|---|
| base E1 penalty (upper-arm residual, C100−C000) > 0, interval excludes 0 | +2.16° [1.63, 2.72] ✔ |
| best owner reduction (both) ≥ 50%, interval excludes 0 | 86% ✔ |
| static elbow penalty also reduced by that swap | −1.10 cm [−1.63, −0.64] ✔ |
| broad (≥ 2/3 of sequences) | 37/37 ✔ |
| intrinsic remainder under GT reach + side < 25% (or not significant) | 14%, interval includes 0 ✔ |
| CASE A requirement: intrinsic ≥ 50% | ✘ |

**Mechanical classification: CASE B.**
- **Secondary metrics agree:**
  - proximal rotation residual: 83% removed;
  - proximal discontinuity rate: 86% removed;
  - elbow motion residual: 86% removed.
- **What the CASE B wording overstates:**
  - E1 with accurate reach/side does not approach GT motion in absolute terms. C111's upper-arm residual (8.72°) equals H0's (8.81°) rather than approaching 0, because the direction-error floor (chain residual ~9.1–9.4°) is shared by H0 and E1.
  - The depth oracle shows that this floor is mainly forward-depth error: 2.14° with GT f.

**Conclusions:**
- **Primary cause of the E1-specific failure.** The Worklog 76 temporal and elbow failure is primarily caused by incompatibility between the substituted E1 direction and the independently estimated H0 reach and bend side. The bend side is a slightly larger owner than the reach, and in folded arms both together are needed. The E1 direction series is not detectably noisier than H0's own direction.
- **Next architecture hypothesis.** Preserved: **coherent arm-geometry ownership.** Direction, reach and bend side should be estimated or reconciled together, not mixed from independent owners.
- **Unresolved qualification (CASE C component).** Forward-depth accuracy remains a material limitation shared by H0 and E1:
  - perfect f removes the E1 penalty even with H0 reach/side;
  - perfect f lowers the absolute motion floor;
  - E1 depth errors produce the worst static failures (#325, #446) and the sequence-local remainders.

  Better visual depth and coherent geometry ownership are complementary, not alternatives.
- **Oracle caveat.** GT reach/side and GT f are oracles. These results bound what coherent ownership or better depth could achieve; they are not a deployable design.

Nothing was smoothed, gated, selected, trained or integrated. The next architecture needs a new bounded decision.

## Focused tests

- **`tests/test_framepose_visual_arm_ownership.py` (7 tests):**
  - exact and complete factor assignment;
  - runtime candidates see no GT;
  - C000/C100 equal the Worklog 76 arms;
  - GT direction + GT reach + GT side equals the reference arm;
  - reach/side source and Worklog 62 perpendicularization for every factorial arm;
  - fixed bone lengths and FK;
  - deterministic arms;
  - degenerate GT side falls back to the rest plane, and parallel side and clamp propagate;
  - motion residual on synthetic rigid motion (zero for identical motion, a constant offset not penalized, a frozen candidate penalized by the reference motion);
  - factorial contrasts and interactions on additive and pure-interaction cases;
  - sequence-aware contrast/bootstrap accounting on synthetic sequences;
  - deterministic bootstrap;
  - static runner contracts: gate order, Worklog 76 thresholds reused, decisions declared before `main`, no training/filtering, no production imports.
- **Server run (`animcv-framepose:cuda118`):** with the Worklog 61/62/74/75/76 focused tests, **44 passed**.
- **One expectation adjusted:** the synthetic "identical motion" check uses 1e-5° for quaternion-distance entries, because `acos` near 1 limits float64 resolution (~1e-6°). Positions keep 1e-9.
- **Full regression:** not run, because production behavior is unchanged.

## Reproduction

    docker run --rm -v $HOME/AnimCV:/workspace:ro -v $HOME/animcv-output:/output \
      -v $HOME/animcv-data/datasets/3dpw:/data/3dpw_raw:ro \
      -e PYTHONPATH=/workspace/src:/workspace/scripts -w /workspace animcv-framepose:cuda118 \
      scripts/run_framepose_visual_arm_ownership.py --rig examples/BaseRig.fbx \
      --rest examples/e2e_demo/baserig_blender_rest.json --calibration examples/e2e_demo/framepose_fk_calibration.json \
      --chains examples/e2e_demo/framepose_ik_chains.json --bank /output/framepose/bank_3dpw_paired_v2.json \
      --h0-dir /output/framepose/context_h0_v2 --threedpw-raw /data/3dpw_raw \
      --w73-dir /output/framepose_external_visual_supervision/transfer --w75-dir /output/framepose_visual_ik_endpoint_test \
      --w76-dir /output/framepose_visual_full_arm_ik --out-dir /output/framepose_visual_arm_ownership
    scripts/export_framepose_visual_arm_ownership_cases.py --rig examples/BaseRig.fbx --rest ... --calibration ... --chains ... \
      --ownership-dir /output/framepose_visual_arm_ownership --out /output/framepose_visual_arm_ownership/review_cases.json
    Blender -b -P scripts/review_framepose_visual_arm_ownership_blender.py -- --rig examples/BaseRig.fbx \
      --rest examples/e2e_demo/baserig_blender_rest.json --cases review_cases.json --out-dir review --caption-python python3
