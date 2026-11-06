# Worklog — Endpoint-Space Transition Candidate with Isolated Per-Request Baselines (2026-11-06)

Batch 119. Evidence-only scratch experiment on `arch/single_frame_first`, run unattended on Main (local RTX 5080 host;
no GPU inference was needed).

Nothing historical was changed:
- no `src/` module;
- no W116–W118 script, test, Worklog, declaration, decision or CASE;
- no rig, Action, source, FBX pipeline, model, threshold or timebase.

`docs/next_batch.md` was read only. Every Blender number, test result, image and manifest below is **AGENT-RUN** on Main
and has not been independently re-executed by the Reviewer.

**Scope of the task.**
1. Exclude W118's family-shared segment-baseline contamination by building clean, isolated per-request baselines
   (U_r, F_r, X_r). W118 is not repaired or re-decided.
2. Test one bounded candidate, **Y**, against clean X_r. Y guides the transition with the wrist in world space and uses
   the same pin, edit, DOF and per-side margins as X_r.
3. Disclose the actual Blender footprint, check hard-pin, atomicity and reversal behaviour, and run the negative controls.

## 1. Official CASE and frozen declaration

**Verdict: CASE D — INVALID** (frozen rule; `results/decision/decision.json` `ef8c1c4c…`).
- The declaration `6616d1bdf227f97cda032e56629a4366e0188af1dfc310f5185049742bba9e4b` was hashed and the instrument was
  committed (`998c7da`) before any build.
- The frozen per-request controls (`frozen_controls.json` `3c209df6…`) were hashed in calibrate, before any Y existed.

Frozen rule: D if any V fails. Otherwise C if any C gate fails, or if no evaluation request meets the Y-vs-X
improvement test. Otherwise A if U_syn, U_real and U_vis all pass. Otherwise B.

| Gate | Result |
|---|---|
| V1 authentication, V2 parent reproduction, V3 baseline reproduction, V4 contamination reproduced (not repaired), V5 clean isolated baselines, V7 predeclaration, V8 independent Blender evaluation, V9 population, V10 repeatability, V11 timing | pass |
| **V6 A/B fairness** | **fail on one control, `Z_A`**: `SEARCH_BUDGET_MISMATCH:Y` |
| C1 hard pin & scope, C2 truthful footprint, C4 atomicity/lineage/reversal, C5 source immutability, C7 user edit & collateral | pass |
| **C3 negative controls** | **fail**: NY_PIN1, RY_PIN1 and NY_FALSE were REJECTed with the required reason `EVALUATED_PIN_UNSATISFIED`, but with class `CONTRACT_VIOLATION` instead of the declared `SOLVER_FAILURE` |
| **C6 no new hard failure** | **fail on `Z_A`** (same cause as V6) |
| U_syn (held-out H_G improves over X_r) | fail |
| U_real (≥1 RH1-family and ≥1 of {H_RH2_RBpos, DJ_R}) | pass (H_RH1_none, H_RH1_RBpos; H_RH2_RBpos, DJ_R) |
| U_vis (agent review) | pass (H_RH2_RBpos, DJ_R; agent-only) |

**Why D: two instrument-specification errors of mine, not candidate or evaluator errors.**

1. **The Y search budget was not parameterised for the zero-budget control.**
   - `endpoint_transition` always builds its declared search budget with M_MAX = 8.
   - The frozen `Z_A` control's X_r uses M_MAX = 0.
   - Both use the same margins, all zero.
   - Y's actual Z_A output changed only the 12 pin frames, with a pin residual of 7e-8 m and every frame passing. Only the
     *declared* budget differed. The fairness preflight correctly detected that mismatch and rejected Y. This fails V6
     and, through the rejection, C6.
2. **Wrong class precedence in three declared expectations.**
   - The three hard-pin tamper controls also change the pin-frame values relative to F_r.
   - The A/B fairness pre-check (`PIN_VALUES_DIFFER_FROM_F`) therefore fires as well. It takes `CONTRACT_VIOLATION`
     precedence over `SOLVER_FAILURE`.
   - Each control still REJECTed atomically with the required reason. NY_PIN1 and RY_PIN1 each have exactly one failing
     frame. Only the declared class was mis-specified.

Neither error was patched after outcomes were seen.

**Mechanical alternative (labelled; not assigned).** Suppose the Z_A budget-declaration defect and the C3 class
precedence were set aside:
- every C gate would pass;
- U_syn fails, U_real passes and U_vis passes (agent review);
- the frozen rule would then output **B**.

I do not assign B. The gates were frozen, and re-scoping them now would be outcome-driven. Historical verdicts are
unchanged: W113 D, W114 D, W115 D, W116 B, W117 C, W118 D.

## 2. Historical baseline and immutable CASE fingerprints

- **Starting commit** `78809d3e641e322b62cb7f64e5b834a063b403a3`.
  - It equalled HEAD and `origin/arch/single_frame_first` (fetched; no divergence, before and after the run).
  - Instruction blob `5f97c3a23f066da0081f392a4234e4b6413620bb`, verified first.
  - The Main-local lock holds batch 119, this blob and this commit.
- **Worklogs:** W113–W118 blobs are unchanged and carry their COMPLETE markers. The W118 COMPLETE commit `913bac2` is an
  ancestor of HEAD. W118's instruction blob at its start commit is `2468904f…` (provenance verified). No other W119
  Worklog existed.
- **Manifests** re-verify at preflight and again after assess:
  - W116 `11764558…`;
  - W117 `9f90f91a…`;
  - W118 `90ac1b7f…` (1,062 files).
- **Declaration/decision hashes and CASEs:**
  - W116 `ad4fe0df…` / `bcf6ee3a…` B;
  - W117 `fab62081…` / `f0d70751…` C;
  - W118 `5d61b552…` / `91200bd4…` D. The W118 decision hash and CASE D are the same after the run.
- **Inputs:** W102/W105/W115 re-hashed and manifest-bound as in W116–W118.
- **Parent v0:**
  - rebuilt, canonical-equal to W118;
  - a fresh Blender 5.1.0 evaluation has the W105 FIXED digest `4d168d95…`;
  - it is bit-equal on all 199 frames to the W118 preflight evaluation.
- **Synthetic:**
  - The spec and parents A–E rebuild and probe bit-equal to W118.
  - New trajectory **G** is W119-only.

## 3. W118 V4 contamination — reproduced read-only, not repaired

From the frozen W118 `eval_e1` Blender output (`calibration.json` → `contamination_reproduction`):

- Every family-shared `SEG_<family>__F` has out-of-pin changes for each of its 9 segment requests. **All** of those
  changes lie on sibling segments' pin frames.
  - Example: `S_RA_pos_60_61` vs `SEG_RA_pos__F` is changed on `[63,64],[78,107],[109,118],[154,182],[194,198]`.
- The reproduced list equals W118's V4 failure list.
- The quality windows (pin ± 9) that contain sibling changes are exactly the six named in W118: `S_RB_pos_109_118`,
  `S_RB_pos_86_106`, `S_RA_pos_60_61`, `S_RA_pos_63_64`, `S_RA_pos_78_107` and `S_RA_pos_109_118`.
- `S_PF_C_0_29` (single segment) is not contaminated.
- The W118 V4 gate and decide were not rerun. No shared state was used in any W119 primary comparison.

## 4. Isolated per-request U_r / F_r / X_r / Y_r provenance

The W117 replay plus the W118 build sequence (without tampers) reproduces **every** W118 state id. This holds in
calibrate and in both builds.

For each of the 41 request records:
- the 40 W118 X-requests;
- the new held-out H_G.

**F_r** = U with the frozen W116 framewise values **only on this request's pin frames**.
- **Value-level isolation:** frames differing from U ⊆ pin, for every contract-valid request.
- **Blender-level isolation:** actual F_r footprint ⊆ pin, for every contract-valid request. All six previously
  contaminated windows were rebuilt with 0 out-of-pin changes. For example, the shared F of `S_RB_pos_86_106` changed
  `[109,118],[154,182]`; its F_r changes nothing outside `[86,106]`.
- **Equality to the frozen single-frame resolver:** an independent re-solve of every pin frame with
  `S.resolve_frame` (analytic) is bit-equal to the F_r pin values, with equal statuses, for every contract-valid
  request.
- **Hold requests:** F_r state id = the historical per-request `<rid>__F` id. Those baselines were already isolated.

**X_r** = the frozen W118 offset-layer editor rebuilt from F_r. Its state id equals the W118 `<rid>__X` id for every
contract-valid request, so the W118 segment X states were already per-request; only their F comparator was shared.

**Y_r** reads exactly X_r's per-side margins (`margins_of`).

**A/B fairness** (`ab_fairness`) checks, for F_r / X_r / Y_r:
- parent, rig, U and pin-constraint digests;
- pin frames, authorisation and pin-frame values;
- X_r/Y_r margins and search budget.

It passes on every non-negative request except Z_A (§1).

**Correlated families are labelled:**
- `RH1_window_109_118`: H_RH1_* plus S_*_109_118 and DJ_R;
- `RH2_window_154_181`: H_RH2_* plus S_*_154_182 and DJ_R.

## 5. Candidate Y — formulation, DOF and parameters (frozen before any Y output)

**Margins.** On the m frames beyond each pin-run boundary b, where m is the frozen X_r margin of that side:
- Y never chooses m;
- if m exceeds the available budget, Y REJECTs with `MARGIN_EXCEEDS_AVAILABLE_BUDGET` and never widens it.

**Weight and targets.** w_j = 1 − smoothstep(j/(m+1)), the same profile as X.
- **Wrist guide (soft):** G(k) = W_U(k) + w_j (W_F(b) − W_U(b)).
- **Elbow pole reference:** R(k) = E_U(k) + w_j (E_F(b) − E_U(b)).

**Reach and near-straight handling.**
- d_c = clip(|G − S|, r_min, min(r_max, r(φ_floor))), with φ_floor = min(flex_U(k), flex_F(b)).
- The arm is never made straighter than both poses it blends between.
- Statuses: `GUIDE_REACHED`, `GUIDE_FLEX_FLOOR_CLAMPED`, `GUIDE_UNREACHABLE_OUTER` and `GUIDE_UNREACHABLE_INNER`.

**Bend plane.**
- The elbow is placed at the point of its circle nearest R(k).
- If R(k) lies within 2° of the shoulder→guide axis, the pole is taken from the previous, boundary-side solved elbow
  (`BEND_PLANE_AMBIGUOUS`).
- A sign flip of the bend normal against the previous frame is flagged as `BRANCH_FLIP_VS_PREVIOUS`; it is reported, not
  gated.

**Solve.**
- Upper arm: minimal world swing that places the elbow.
- Forearm: minimal world swing aiming the wrist at the clamped guide.
- An FK residual above 1e-6 m is `NON_CONVERGED` (SOLVER_FAILURE).

**DOF, pins and fallback.**
- **DOF:** authorised upper/lower arm channels only. Anything else is `Y_MISSING_AUTHORIZED_DOF` and REJECT.
- **Pin frames:** exactly F_r.
- **Fallback:** none.

**Relation to earlier candidates.** The wrist-guide formula is the W117 TE guide. Y differs from TE in three ways:
- it uses X's margins instead of the fixed 4;
- it adds the guided pole and the flex floor;
- it uses a closed-form per-frame solve instead of the W117 optimiser.

**Runtime:** 0.25 s total for all Y solves; at most 0.02 s per request.

## 6. Synthetic population, held-out case and negative controls

**H_G (new, held-out).**
- Trajectory G, chosen by the frozen rule in `HG_RULE` from parent/unpinned geometry only.
- Hold [9, 18], captured at 9.
- Entry offset 0.079 m, release offset 0.314 m (asymmetric); minimum reach slack 0.026 m.

**Synthetic results (Blender; Y vs clean X_r; B1 = peak boundary wrist compensation step, m):**

| Request | F_r | X_r | Y_r | Y-vs-X verdict |
|---|---|---|---|---|
| H_A | 0.423 | 0.072 | 0.070 | not improved; D3 2.2→5.8°/frame worse |
| H_B | 0.288 | 0.053 | 0.048 | −10 % only (< 20 %) |
| H_C (near-singular) | 0.189 | 0.032 | 0.031 | not improved; D2/D3 worse |
| H_E | 0.287 | 0.048 | 0.048 | equal |
| **H_G (held-out)** | 0.314 | 0.053 | 0.052 | **not improved**; D3 3.0→5.2°/frame worse |
| H_S0 / H_END / DJ_B | 0.028 / 0.238 / 0.065 | 0.010 / 0.041 / 0.020 | 0.010 / 0.039 / 0.019 | not improved (H_END D2/D3 worse) |
| S_PF_C_0_29 | 0.017 | 0.009 | 0.009 | not improved; near-straight frames 25–29 inherited by both |

On smooth synthetic parent motion, X's rotation offset already carries the wrist offset almost exactly along Y's guide.
Y adds a little upper-arm rotation-step excess. There is no new near-straight frame and no flip on any synthetic
request. W_PF_C (whole clip) REJECTs as infeasible for X and Y alike.

**W119 negative controls** (`NEG_Y`; all executed):

| Control | Result |
|---|---|
| NB_SHARED (SEG_RB_pos__F as F_r) | REJECT BASELINE_NOT_ISOLATED ✓; value-level and Blender-level isolation both fail ✓ |
| NB_SIBLING (one sibling frame 63 edited) | REJECT BASELINE_NOT_ISOLATED ✓ |
| NY_U (Y on the RB_neg user edit) | REJECT U_MISMATCH ✓; also out-of-budget, unauthorised and outside-arm changes |
| NY_RIG / NY_MARGIN | REJECT RIG_IDENTITY_MISMATCH ✓ / MARGIN_BUDGET_MISMATCH ✓ |
| NY_BONE / RY_BONE (base / spine_01) | REJECT UNAUTHORIZED_CHANNEL_CHANGE ✓ |
| NY_ROOT | REJECT ROOT_MOVED ✓ |
| NY_OUTSIDE / NY_PLANNED | REJECT FOOTPRINT_EXCEEDS_DECLARED_DOMAIN ✓ |
| NY_GUIDE_UNREACH (guide 0.05 m beyond reach) | ACCEPT ✓; 16/16 frames GUIDE_UNREACHABLE_OUTER; guide shortfall 0.083–0.124 m; pins pass |
| NY_AMBIG (pole on the axis) | ACCEPT ✓; 16/16 BEND_PLANE_AMBIGUOUS; 0 branch flips |
| NY_NONCONV (forearm aim skipped) | REJECT SOLVER_FAILURE NON_CONVERGED ✓ (16 frames) |
| NY_INFEASIBLE (H_Cx) / NY_CONFLICT (N6) / NY_DOF (N3) | REJECT INFEASIBLE_REQUEST ✓ / CONFLICTING_HARD_CONSTRAINTS ✓ / Y_MISSING_AUTHORIZED_DOF ✓ (X had ACCEPTed N3) |
| NY_PIN1 / RY_PIN1 | REJECT with EVALUATED_PIN_UNSATISFIED and exactly 1 failing frame; class **CONTRACT_VIOLATION ≠ declared SOLVER_FAILURE** ✗ |
| NY_FALSE (U on the pins, claims SOLVED) | REJECT EVALUATED_PIN_UNSATISFIED (12 frames); class **CONTRACT_VIOLATION** ✗ |
| DY1–DY5 (relabelled/omitted frame, collapsed runs, planned-domain display, REJECTed shown as applied) | all FAIL the truth check ✓ |

W117 negatives N2, N4, N5 and N7 REJECT as before.

## 7. Authentic real-rig A/B (correlated families noted)

The real clip is GT-free, and the video subject identity is UNKNOWN for all 199 frames. Only the known output rig is
evaluated.

| Request (family) | F_r | X_r | Y_r | Y-vs-X verdict |
|---|---|---|---|---|
| H_RH1_none (RH1) | 0.157 | 0.081 | 0.059 | **improved** (−27 %, −22 mm); X's near-straight frame 122 is absent in Y |
| H_RH1_RBpos (RH1, correlated) | 0.156 | 0.080 | 0.060 | **improved** |
| H_RH1_RBneg (RH1, correlated) | 0.158 | 0.081 | 0.059 | B1 better but **new bend flip at 123** and D3 9.9→17.9°/frame: not improved |
| H_RH2_RBpos (RH2; W117 T/TE NON_CONVERGED control) | 0.281 | 0.217 | 0.047 | **improved** (−78 %, −170 mm) |
| DJ_R (RH1+RH2 disjoint; correlated with both) | 0.281 | 0.217 | 0.060 | **improved** |
| H_RH2_none / H_RH2_RBneg | — | REJECT | REJECT | infeasible pin frames (as in W117/W118) |
| S_RB_pos_86_106 (previously contaminated) | 0.032 | 0.009 | 0.009 | equal; clean X vs F is a real W118-rule gain |
| S_RB_pos_109_118 (previously contaminated) | 0.006 | 0.006 | 0.006 | no transition change. W118's listed "improvement" was the shared-state artefact, now confirmed |
| S_RA_pos_60_61 / 63_64 / 78_107 / 109_118 (previously contaminated) | ≈0.005–0.010 | ≈0.005–0.010 | ≈0.005–0.010 | no material difference (S_RA_pos_109_118: D2 worse) |
| S_RA_pos_154_182, S_RB_pos_154_182 (low-evidence 154–181) | 0.010 / 0.023 | 0.009 / 0.022 | 0.009 / 0.021 | no material difference; Y removes X's near-straight frame 187 on S_RB |
| S_RA_pos_194_198, W_RA_neg, W_RB_neg, W_RC_raise | — | — | — | ACCEPTED_NO_TRANSITION_AVAILABLE |
| W_RA_pos, W_RB_pos | — | REJECT | REJECT | infeasible frames |

**Mechanism behind the real gains.**
- **RH1 release.** The arm is near-straight, so 7 of 8 release guide frames are `GUIDE_UNREACHABLE_OUTER` and are
  clamped by the flex floor. Y's wrist falls up to **0.056 m short of its guide** (Blender) and keeps about 10–15° of
  flex where X drops to 0.7–0.9° at frame 122. The RH1 B1 gain therefore comes mainly from the floor clamp, not from
  following the world-space guide.
- **RH2 release.** All 10 guide frames are reached (Blender residual < 1e-6 m to the guide). Here the gain does come
  from carrying the world-space wrist offset through U's fast motion and U's own bend flip at 182, where X's constant
  rotation offset folds the forearm.
- **Source jumps.** The window's raw wrist step peak is unchanged for every variant: RH1 0.328 m (U's 118→119 jump);
  RH2 0.485 m (U's 153→154 jump). These are source discontinuities, not edits.

## 8. Blender pin residuals and changed frames/Bones

- **Non-negative requests:** 28 of 34 Y_r ACCEPTed.
  - REJECTs: Z_A (§1) and five infeasible requests (H_RH2_none, H_RH2_RBneg, W_PF_C, W_RA_pos, W_RB_pos).
  - X_r ACCEPTed 29.
- **Hard pin:**
  - maximum ACCEPTed Y pin residual 0.81 mm, the same straight-arm frame as X/W116–W118;
  - 0 pin frames over 1 mm;
  - every ACCEPTed Y has every frame passing scope, locality, lengths, root and joint limits.
- **Footprint:**
  - actual Y footprint ⊆ planned domain = X planned domain ⊆ budget on every request;
  - changed channels only `upperarm_l`/`lowerarm_l` (synthetic `upper`/`lower`);
  - user-edit channel and `__object__` bit-equal to U;
  - collateral points exactly 0 m from U.
  - Example: DJ_R actual Y runs `[109,126],[152,189]`, the same as X.
- **Classes of the non-negative Y_r** (`y_class`):
  - VALID_REACHABLE_GUIDE: 11;
  - GUIDE_FLEX_FLOOR_CLAMPED only: 4;
  - with GUIDE_UNREACHABLE: 7, including BEND_PLANE_AMBIGUOUS in 4;
  - no transition available: 6;
  - infeasible: 5;
  - contract violation: 1 (Z_A);
  - Y-vs-X improvement: 4 (H_RH1_none, H_RH1_RBpos, H_RH2_RBpos, DJ_R).

## 9. Raw vs source-relative temporal accounting

Every request reports both:
- the compensation step (edit-induced, vs U);
- the raw wrist step of the rig output and of U.

The charts in `render/charts/steps_<rid>.png` show both rows. The 0.485 m / 0.328 m source jumps appear identically in
U, F_r, X_r and Y_r and were not treated as edit defects. No source discontinuity or timebase was repaired.

## 10. Bend-plane, near-straight and elbow costs

`render/charts/flex_<rid>.png` shows the absolute flex time series for U/F/X/Y. The 5° line is the W118 diagnostic
reference, not a rule.

| Effect | Requests |
|---|---|
| Y removes X's introduced near-straight frames | H_RH1_none / RBneg / RBpos (frame 122), DJ_R (122), S_RB_pos_154_182 (187) |
| Y introduces a new bend flip | H_RH1_RBneg (123) |
| Shared near-straight frames | inherited by both (e.g. 181, S_PF_C 25–29) |

**Elbow deviation from U** (mean): Y ≈ X ± 10 mm in general.
- H_B is higher: 49 → 59 mm.
- H_END is lower: 76 → 60 mm.
- Peaks are the same.

**Upper-arm rotation-step excess (D3)** is higher for Y on 6 synthetic requests and on RBneg.

## 11. Timeline and actual-rig visual review (agent-only, not independent)

**`results/render/`** contains:
- 60 timelines;
- step and flex charts;
- filmstrips and GIFs (U/F_r/X_r/Y_r) for H_G, H_B, DJ_B, H_RH1_none, H_RH2_RBpos, DJ_R and S_RB_pos_86_106;
- `index.html`.

**Timelines** have separate rows for:
- the hard pin;
- the hatched search budget;
- the planned domain (X = Y);
- the actual F_r/X_r/TE/Y_r footprints;
- the Y per-channel rows;
- a Y guide-status row.

**Filmstrip fix.** The band now sits **directly below its own tile**, with a gap between rows. This fixes the W118
readability defect in the new renders only; W118 renders are untouched.

**Review** (`review/visual_review.json` `7f8f8e45…`). Written after the assess/render outputs were printed and before
decide.

- **Inspected:**
  - filmstrips H_RH2_RBpos, H_RH1_none and H_G;
  - steps charts H_RH2_RBpos, H_RH1_none and H_G;
  - flex chart H_RH1_none;
  - timeline DJ_R;
  - extracted full-size GIF frames 181–185 (H_RH2_RBpos), 118–123 (H_RH1_none) and 181–184 (DJ_R), saved under
    `review/gif_frames/`.
- **Not inspected:**
  - GIFs played as animation;
  - H_B, DJ_B and S_RB_pos_86_106 filmstrips/GIFs;
  - the other timelines and charts;
  - `index.html` in a browser.
- **Observations:**
  - **H_RH2_RBpos / DJ_R:** visibly different at 182–185. X folds the forearm back across the torso, while Y lets the arm
    hang near U's pose. Marked visible. Whether Y's forearm contacts the hip cannot be judged from bone proxies.
  - **H_RH1_none:** the difference (X nearly straight vs Y slightly bent) is small at tile scale; it is a numeric gain
    only.
  - **H_G:** X and Y are indistinguishable.
- The palette validator was not run because Node is absent. The colours are the reference categorical slots
  (F blue, X orange, Y aqua, U grey dashed).

## 12. Hard failures, rejection classes and no-fallback proof

- **Hard failures:**
  - 0 accepted pin frames over 1 mm;
  - 0 infeasible requests accepted;
  - 0 Y solver nonconvergences outside the fault-injected control;
  - the only new rejection vs X is Z_A (budget declaration, §1).
- **No fallback:** `fallback: None` in every Y info. Y never returns F or X values on transition frames. Missing DOF
  (NY_DOF/N3, N2) is REJECTed, not degraded.
- **Named regression control H_RH2_RBpos:** Y ACCEPT (as did X). W117 T/TE `NON_CONVERGED` stays a historical label; it
  is not re-labelled as solved.

## 13. Source immutability, atomic lineage and exact reversal

- **Synthetic:**
  - S1: H_G Y ACCEPTed;
  - S2: NY_PIN1 REJECTed, head unchanged;
  - S3: H_A Y against the old head → STALE_PARENT;
  - S4: revert. Values are byte-equal, and the Blender digest and frames are bit-equal to P__G.
- **Real:**
  - R1: H_RH1_none Y ACCEPTed;
  - R2: RY_PIN1 REJECTed, head unchanged;
  - R3: H_RH1_RBneg → STALE_PARENT;
  - R4: revert, Blender bit-equal to v0.
- **Per request:** every X_r/Y_r was committed on its own ref. Preview id = candidate id; heads are consistent; REJECTed
  candidates are never shown as applied. Lineage chains re-verify.
- **Source preservation:**
  - the source guard is unchanged over all stages;
  - the W116/W117/W118 manifests re-verify after assess;
  - the W118 decision is unchanged;
  - the parent files verify.
- **Repeatability:** builds r1/r2 are byte-identical (states and Y infos); evaluations e1/e2 are identical.

## 14. Alternative explanations, negative findings and non-claims

- **The D comes from my declaration, not from Y's output** (§1). The Z_A defect is in the declared search budget only.
  The class precedence is a mis-specified expectation.
- **Negative findings:**
  - **Synthetic:** Y gives no material improvement over clean X_r on any synthetic request, including the held-out H_G.
    Its upper-arm rotation-step excess is higher on several.
  - **Real, one family member:** on RBneg, Y introduces a bend flip.
- **Real RH1 "gain" is largely the flex-floor clamp.** The guide is unreachable on the near-straight arm, so Y is not
  following its world-space objective there. A rotation-space method with the same floor might do equally well; that was
  not tested.
- **The real RH2 gain is one release event.** H_RH2_RBpos and DJ_R share the same window and edit, and U has its own bend
  flip at 182.
- **One formulation only;** no sweep. Y's own-margin diagnostic was NOT RUN.
- **Not claimed:**
  - anatomical plausibility;
  - performer fidelity;
  - FBX/engine survival;
  - Production readiness;
  - a user-facing smoothness or joint limit;
  - that a passing metric proves user-facing quality.
- The 1 mm pin criterion remains the provisional W116 diagnostic.
- **Timebase:** source 50 fps / Action 30 fps; frame-indexed evaluation; not repaired.

**Unresolved uncertainties:**
- independent review and execution;
- Y on unseen real clips;
- whether the clamp or the guide drives RH1;
- contact or penetration with the body at the RH2 release;
- animation-playback judgement.

## 15. Files, tests, hashes, environment and replay

**Completed:** all Required Evidence parts (1–7), every declared stage, and the agent review of the items listed in §11.
**Not completed:** independent review; Y's own-margin diagnostic; playback of the GIFs.

**Changed files** (all new):
- `scripts/w119_endpoint_transition_base.py`
- `scripts/w119_endpoint_transition.py`
- `tests/test_w119_endpoint_transition.py`
- this Worklog

W116–W118 code is imported unchanged.

**Revisions:**
- start `78809d3`;
- `998c7da`, the pre-outcome instrument and the **final source revision** (no code changed after it);
- the Worklog commit;
- the marker commit.

**Pre-declaration work, disclosed:**
- the focused unit tests;
- preflight and calibrate (no Y);
- one crash-only smoke run in `C:/Users/dna10/animcv-w119-smoke`. Stdout was discarded unread; every stage exited 0 with
  empty stderr; the directory was deleted. Afterwards only the disclosure text was edited.

**Tests (AGENT-RUN):**
- `tests/test_w119_endpoint_transition.py`: **19 passed** (`w119_tests.xml`). They cover:
  - isolation and contamination;
  - a single sibling frame;
  - fairness mismatches;
  - margins;
  - Y pins, domain and determinism;
  - guide reach and weight profile;
  - the step bound;
  - the flex floor;
  - unreachable guide;
  - ambiguity;
  - nonconvergence;
  - missing DOF;
  - margin overflow;
  - zero margins;
  - decision and tamper;
  - display truth;
  - guide residuals;
  - the compare_y rule;
  - raw vs compensation attribution.
- **Regressions: 128 passed** (`w119_regression.xml`): W118, W117, W116, W115, W114, W105 rig utility, fk_solver,
  canonical_pose_parity and retarget_solver.
- **Failed:** none.
- **NOT RUN:** the full suite, FBX export/import and engines.

**Key artefacts** (under `C:/Users/dna10/animcv-w119/results`; SHA-256 prefixes):

| Artefact | SHA-256 |
|---|---|
| declare/declaration.json | 6616d1bd… |
| calibrate/frozen_controls.json | 3c209df6… |
| calibrate/calibration.json | 593e8a10… |
| preflight/preflight.json | b9088b61… |
| build_r1/real_states.json / syn_states.json (= r2) | 2a827a85… / 4c6be707… |
| eval_e1/real_eval.json / syn_eval.json (= e2) | 712474e4… / b9e88683… |
| assess/assess_report.json | 93cb4caa… |
| review/visual_review.json | 7f8f8e45… |
| decision/decision.json | ef8c1c4c… |
| render/index.html | b9be4d29… |

`evidence_manifest.json` = `5e8f8f70779a5fe3da08274ebee456e2ea36096d38606769217d9786d4684735` (1,392 files, excluding the
stores), written before this Worklog. Code blobs: base `dd20d4d7…`, main `d7817568…`, tests `0bef24eb…`.

**Reproduction** (Git Bash on Main, repo root; every stage refuses an existing output directory):

```
PY=C:/Projects/AnimCV/.venv_build_full/Scripts/python.exe; R=C:/Users/dna10/animcv-w119/results
S=6616d1bdf227f97cda032e56629a4366e0188af1dfc310f5185049742bba9e4b; D="--declaration $R/declare/declaration.json --declaration-sha $S"
$PY -X utf8 scripts/w119_endpoint_transition.py preflight --out-dir $R/preflight
$PY -X utf8 scripts/w119_endpoint_transition.py calibrate --preflight $R/preflight --out-dir $R/calibrate
$PY -X utf8 scripts/w119_endpoint_transition.py declare --preflight $R/preflight --calibration $R/calibrate --out-dir $R/declare
$PY -X utf8 scripts/w119_endpoint_transition.py build $D --tag r1 --out-dir $R/build_r1
$PY -X utf8 scripts/w119_endpoint_transition.py build $D --tag r2 --out-dir $R/build_r2
$PY -X utf8 scripts/w119_endpoint_transition.py evaluate --build-dir $R/build_r1 --preflight $R/preflight --out-dir $R/eval_e1
$PY -X utf8 scripts/w119_endpoint_transition.py evaluate --build-dir $R/build_r1 --preflight $R/preflight --out-dir $R/eval_e2
$PY -X utf8 scripts/w119_endpoint_transition.py assess $D --build-dir $R/build_r1 --eval-dir $R/eval_e1 --out-dir $R/assess
$PY -X utf8 scripts/w119_endpoint_transition.py render $D --build-dir $R/build_r1 --assess-dir $R/assess --out-dir $R/render
# agent review: $R/review/visual_review.json (+ review/gif_frames extracted from the GIFs)
$PY -X utf8 scripts/w119_endpoint_transition.py decide $D --build-dir $R/build_r1 --build-dir2 $R/build_r2 --eval-dir $R/eval_e1 --eval-dir2 $R/eval_e2 --assess-dir $R/assess --visual-review $R/review/visual_review.json --out-dir $R/decision
$PY -X utf8 scripts/w119_endpoint_transition.py manifest --base C:/Users/dna10/animcv-w119 --declaration-sha $S
$PY -X utf8 -m pytest tests/test_w119_endpoint_transition.py -q
```

The preflight HEAD check binds the starting commit, so a later replay needs that checkout. The stages read the
W116–W118 artefacts under `C:/Users/dna10/animcv-w11{6,7,8}`. Blender is `C:/Program Files/Blender Foundation/Blender 5.1`.

**Resources and environment:**
- **Timing:** preflight 12 s; calibrate 13 s; each build ≈ 11 s; each Blender evaluation ≈ 10 s; assess 9.6 s (plus
  revert evaluations); render 28 s; tests 0.4 s; regressions 8 s.
- **Storage:** 429 MB under `C:\Users\dna10\animcv-w119\`. It is local only and not published.
- **Software:** Python 3.11.9, numpy 1.26.4, Blender 5.1.0, Windows 10.0.26200.
- **Scope of execution:** Main only. No LabServer63/69, network download, package change or GPU inference. No evaluation
  GT exists or was read.
- **Git:** single branch `arch/single_frame_first`. No branch, merge, rebase, reset or force operation.

AUTOMATION_BATCH_STATUS: COMPLETE
BATCH_NUMBER: 119
