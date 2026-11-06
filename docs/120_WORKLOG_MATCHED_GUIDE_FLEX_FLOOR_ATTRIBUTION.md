# Worklog — Matched Factor Attribution: World-Space Wrist Guide vs Flex Floor (2026-11-06)

Batch 120. Evidence-only scratch experiment on `arch/single_frame_first`, run unattended on Main (local RTX 5080 host;
no GPU inference needed). `docs/next_batch.md` was read only.

Every Blender number, test result, image and manifest below is **AGENT-RUN** on Main. The Reviewer has not
re-executed any of it, and the visual review is the agent's own, not an independent animator's.

**Task scope.** W119's candidate Y mixed two compensation mechanisms. This batch separates them with one prospectively
declared 2×2 ablation. Every variant shares the same isolated per-request source, hard pin, margins, search budget,
authorised arm channels, rig, analytic solver and Blender evaluator. The two factors are:

- **Factor G — the wrist target on each transition frame.**
  - G0: X_r's world wrist.
  - G1: the W119 world-space guide.
- **Factor F — the flex floor.**
  - F0: no extra floor; only the W116 reach and joint-limit band applies.
  - F1: the W119 floor `min(flex_U, flex_F(b))`.

The task also required: an outcome-independent instrument preflight, negative controls, footprint and rollback
checks, and an actual-rig motion review.

Nothing historical was changed:

- no `src/` module, existing schema, or Production code;
- no W116–W119 script, test, Worklog, declaration, decision, manifest or CASE;
- no rig, Action, source, FBX pipeline, timebase, threshold or W119 solver parameter.

## 1. Official CASE, declaration and gates

**Verdict: CASE B — PARTIAL / CONTEXT-SPECIFIC SUPPORT** (frozen rule). The supported mechanism and its limits:

- **Mechanism supported:** the world-space wrist guide, G1.
- **Where:** one reachable real release event, RH2 at frame 182. Separately, G1 helps the RH1 release, but only when
  combined with the flex floor.
- **Not shown:** benefit on synthetic motion; benefit from the flex floor alone; a clean articulation record.
- **Status:** this is not a general solver success and not Production acceptance.

**Hashes:**

| Item | SHA-256 |
|---|---|
| Declaration | `9b3eca97e9772060bc265914c2da7b1efe7bcd304810bccea19ac25a74b5c25b` |
| Decision (`results/decision/decision.json`) | `1029ce925b0a5a80…` |
| Frozen controls | `aa9eec062162abe0…` |

**Pre-outcome commit:** the declaration was frozen and the instrument committed (`f39df85`) before any factor build.

**Gates** (all pass):

- **V1–V13:**
  - authentication;
  - parent reproduction;
  - baseline reproduction against the W119 frozen controls;
  - read-only W118 contamination reproduction;
  - isolation;
  - **four-way factor matching, including Z_A zero budget**;
  - predeclaration and instrument preflight;
  - independent Blender evaluation;
  - population;
  - repeatability;
  - timing;
  - **G1F1 = W119 Y parity**;
  - negative and display controls.
- **C1–C6:**
  - hard pin and scope;
  - truthful footprint;
  - atomicity, lineage and revert;
  - source immutability;
  - user edit and collateral;
  - no fallback.

**Frozen CASE rule** (abridged; full text in the declaration). For each mechanism M, "improves" means a meaningful
improvement under the frozen W119 comparator.

- **Primary contrasts:**
  - GUIDE: `guide_at_F0` = G1F0 vs G0F0;
  - FLOOR: `floor_at_G0` = G0F1 vs G0F0.
- **Conditions:**
  - **M_syn:** M's primary contrast improves at least one SYN_PRIMARY request.
  - **M_real:** it improves at least one RH1 request AND at least one RH2 request. DJ_R is excluded because it is
    correlated with both.
  - **M_clean:** it introduces no new bend flip or near-straight frame on any eligible evaluation request.
  - **M_vis:** the agent review marks the improvement as visible, with motion inspected across frames.
- **Outcomes:**
  - **A** if one M meets all four conditions.
  - **B** if the contrasts are eligible on H_RH1_none, H_RH2_RBpos and H_G, and some M improves an evaluation request
    in its primary or conditional contrast (`guide_at_F1`, `floor_at_G1`).
  - **C** otherwise.
  - **D** if any V gate fails; C if any C gate fails.

**Mechanism outcome** (`decision.json` → `detail.mechanisms`):

| | GUIDE | FLOOR |
|---|---|---|
| Primary contrast improves | H_RH2_RBpos | none |
| Conditional contrast improves | H_RH1_none, H_RH1_RBpos, H_RH2_RBpos, DJ_R (`guide_at_F1`) | none |
| M_syn | no | no |
| M_real (RH1 and RH2) | RH2 only | no |
| M_clean | **no** — primary introduces near-straight frames on RH1 family, DJ_R, S_RA_pos_154_182, S_RA_pos_78_107, S_RB_pos_154_182 | no (S_RB_pos_154_182 new flip) |
| M_vis (agent) | yes (H_RH2_RBpos) | no |

CASE A fails: M_syn fails, RH1 is missing from the primary contrast, and M_clean fails. CASE B holds: contrasts are
interpretable on all three named requests, and GUIDE improves evaluation requests. W116–W119 CASEs are unchanged.

## 2. Remote HEAD and fingerprints

- **Start.** Starting commit `2d2fb07a6b66ad87540db228335a4665e9aba7bf` equalled HEAD and
  `origin/arch/single_frame_first`.
  - Instruction blob: `7c2630ed97875c410f0a8105c13a1660c07c514e`, verified first.
  - The Main-local lock holds batch 120, this blob and this commit.
  - The remote had not moved when re-fetched before publishing.
- **Worklogs.** W113–W119 Worklog blobs are unchanged and carry their markers.
  - W119 COMPLETE commit `e0ccf74` is an ancestor of HEAD.
  - W119 instruction blob `5f97c3a2…` was verified at its start commit.
- **Manifests** re-verify at preflight and again after assess:
  - W116 `11764558…`;
  - W117 `9f90f91a…`;
  - W118 `90ac1b7f…`;
  - **W119 `5e8f8f70…`** (1,392 files).
- **Frozen declarations, decisions and CASEs** are unchanged after the run:
  - W116 B;
  - W117 C;
  - W118 D;
  - **W119 D**: `6616d1bd…` / `ef8c1c4c…`.
- **W119 D reasons reproduced read-only:** V6 = [Z_A]; C3 = NY_FALSE, NY_PIN1, RY_PIN1; C6.
- **Inputs:** the W102/W105/W115 inputs re-hash to their manifests.
- **Parent v0:**
  - canonical-equal to W119;
  - a fresh Blender 5.1.0 evaluation has the W105 FIXED digest `4d168d95…`;
  - it is bit-equal on all 199 frames to the W119 preflight evaluation.
- **Synthetic:** parents A–E and G rebuild and probe bit-equal to W119.
- **Timebase:** source 50 fps, Action 30 fps. The time axis is the Action frame index; the mismatch is not reconciled.
  Every key is CONSTANT, and sub-frame evaluation equals the keyed frame.
- **Real clip:** GT-free. Tracked video-subject identity is UNKNOWN.

## 3. Code and diff scope

**New files only:**

- `scripts/w120_factor_attribution_base.py`:
  - `factor_transition`;
  - fairness;
  - the precedence table and predictor;
  - contrasts;
  - endpoint jumps;
  - variant differences.
- `scripts/w120_factor_attribution.py`: the stages.
- `tests/test_w120_factor_attribution.py`.
- This Worklog.

**Imported unchanged:** W116–W119 code.

- The shared solve is `w119_endpoint_transition_base.solve_guide`. The pole reference, weight, ambiguity rule, 1e-6 m
  convergence check, flip flag, DOF check and margins are all the W119 ones.
- **Only change versus W119 Y's construction:** the search budget is built with the request's own X_r `m_max`, which is
  0 for Z_A, instead of the global M_MAX = 8.

**Code identity.** The declaration binds the git blobs of all 11 W116–W120 code files. Both builds verified them.
Blobs:

- base `727fa861…`;
- main `18faab78…`;
- tests `413e82ea…`.

**Final source revision:** `f39df85`. No code changed after the declaration.

## 4. Outcome-independent preflight and error-class precedence

These ran in calibrate, before declaration, on the Python-only W118 test rig. No population factor quality was
computed. `calibration.json` reports `instrument_preflight.all_ok = true`.

- **Zero budget.** All four variants built with X's `m_max = 0`:
  - have budget = planned = pin frames;
  - have values equal to F.
  W119's `endpoint_transition` on the same fixture still produces the 8-frame budget, which reproduces the W119 defect.
- **Unequal budget:** rejected as `SEARCH_BUDGET_MISMATCH:Y`.
- **Single mismatches, each rejected independently:**
  - wrong U;
  - wrong rig;
  - wrong constraint digest;
  - wrong source parent;
  - wrong authorised Bone;
  - wrong margin;
  - wrong pin values;
  - wrong solver-parameter digest.
- **Cross-factor identity:** detects one differing margin.
- **Decision fixtures through the actual W119 `decide_y`:**

  | Fixture | Class |
  |---|---|
  | pin tamper + F mismatch | CONTRACT_VIOLATION |
  | isolated nonconvergence | SOLVER_FAILURE |
  | corrupted pin while the solver claims SOLVED, fairness valid | SOLVER_FAILURE |
  | false display | CONTRACT_VIOLATION |
  | out-of-domain change | CONTRACT_VIOLATION |

- **Stale parent:** STALE_PARENT on the content-addressed store.
- **Comparator fixture:** a better B1 together with a new flip is not counted as a meaningful improvement.
- **Frozen precedence table** (`PRECEDENCE_TABLE`, rules P1–P5). It has its own predictor, which was checked
  **exhaustively against `decide_y`**:
  - 262,144 combinations of injected conditions, covering validation class, pin status, missing DOF, evaluation
    failures, footprint and display extras, nonconvergence, margin overflow, isolation and fairness;
  - **0 mismatches**.
- **Negative-control expectations:** every hand-written expectation in `NEG` equals the predictor's output.
- **Recorded property of the frozen W119 precedence** (observed, not changed): nonconvergence combined with an
  output-scope violation gives SOLVER_FAILURE, because the extra set equals {NON_CONVERGED}. W120 declared and tested
  this as `NG_NONCONV_BONE`.

## 5. Matched four-factor mechanism and solver parity

**Factor matching (V6):**

- For every contract-valid non-negative request and every variant, W119 `ab_fairness(F_r, X_r, variant)` passes. It
  covers parent, rig, U, constraint digests, pin frames, authorisation, pin values, margins and search budget.
- The solver-parameter digest is equal.
- The four variants' fairness records are identical.
- Z_A's budget equals its pin frames in all four variants.

**G1F1 = historical W119 Y (V12):**

- On every contract-valid non-negative request except Z_A, G1F1's values digest and every Blender frame are
  bit-equal to the W119 Y evaluation, and U frames are bit-equal to W119's.
- Z_A is excluded as a contract control. Its W119 Y is historically REJECTED (`SEARCH_BUDGET_MISMATCH`). In W120,
  all four variants ACCEPT Z_A with zero budget, and their output equals F_r.

**G0F0 vs external X_r** (Blender, transition frames; 24 requests):

- **Wrist:** reached to ≤ 3.4e-7 m everywhere. B1 is therefore identical to X on every request.
- **Elbow and roll:** these differ because the shared solver uses minimal swings and the W119 pole, not X's rotation
  offset:

  | Request | Elbow difference | lowerarm_l roll | upperarm_l |
  |---|---|---|---|
  | RH2 / DJ_R | up to 62.6 mm | 67.9° | 16.7° |
  | RH1 | 6.2–6.7 mm | ≈ 25.7° | — |
  | H_G | 3.9 mm | — | — |

- **Branch:** no opposite-branch frame on any request with an accepted transition. The REJECTed H_RH2_RBneg has 2.
- **Rotation-step excess (D3):** G0F0 is sometimes higher than X, for example H_RH1_none 12.7 vs 10.2 °/frame.
- **Attribution:** the G0F0–X difference is a solver-reconstruction difference and is not attributed to G.

## 6. Population and denominators

Requests, frames, pins, margins and roles are exactly the W119 population, plus 29 W120 negatives.

**Non-negative population: 34 requests (13 synthetic, 21 real).**

| | X_r | G0F0 | G0F1 | G1F0 | G1F1 |
|---|---|---|---|---|---|
| synthetic ACCEPT / REJECT (infeasible) | 12 / 1 (W_PF_C) | 12 / 1 | 12 / 1 | 12 / 1 | 12 / 1 |
| real ACCEPT / REJECT (infeasible) | 17 / 4 | 17 / 4 | 17 / 4 | 17 / 4 | 17 / 4 |

- **Real REJECTs** are H_RH2_none, H_RH2_RBneg, W_RA_pos and W_RB_pos. These are infeasible pin frames, the same as in
  W117–W119. They stay in the denominators.
- **No new rejection by any factor.**
- **Factorial-eligible requests** (all four ACCEPTed, B1 defined): 11 synthetic (incl. Z_A, H_D development) and 14
  real.
- **ACCEPTED_NO_TRANSITION:**
  - synthetic: NOOP_A, Z_A;
  - real: S_RA_pos_194_198, S_RB_pos_109_118, W_RA_neg, W_RB_neg, W_RC_raise.
- **Correlated families**, each counted once:
  - RH1 = {H_RH1_none, H_RH1_RBneg, H_RH1_RBpos}: three edit variants of one hold [109,118] and one release event;
  - RH2 = {H_RH2_RBpos}: the only feasible member;
  - DJ_R contains both windows;
  - the S_* segments overlap the same windows.
- **Independent real release evidence:** two events in one GT-free clip.

## 7. Blender-evaluated factorial contrasts and interaction

Notation:

- ΔB1 = target − reference, in mm.
- `*` = meaningful improvement under the frozen W119 comparator (20 % and 10 mm, with no new defect or worsening).
- `!` = a new bend flip or near-straight frame relative to the reference.
- I = interaction on B1: G1F1 − G1F0 − G0F1 + G0F0.

| Request | guide@F0 | guide@F1 | floor@G0 | floor@G1 | I |
|---|---|---|---|---|---|
| H_RH1_none | −23.7 ! | −21.5 * | 0.0 | +2.2 ! | +2.2 |
| H_RH1_RBpos | −23.0 ! | −20.8 * | 0.0 | +2.2 ! | +2.2 |
| H_RH1_RBneg | −24.5 ! | −22.3 ! (flip 123, D3 9.9→17.9) | 0.0 | +2.2 ! | +2.2 |
| H_RH2_RBpos | **−170.1 *** (−78 %) | −170.1 * | 0.0 | 0.0 | 0.0 |
| DJ_R (correlated) | −159.4 ! | −157.2 * | 0.0 | +2.2 ! | +2.2 |
| S_RB_pos_154_182 | −2.6 ! | −0.7 | 0.0 ! (flip 187) | +1.8 ! | +1.8 |
| S_RA_pos_154_182 / 78_107 / 60_61 | −1.2 ! / −1.1 ! / −0.8 | ≈ 0 | +0.6 / +1.0 / +0.5 | +1.8 / +2.0 / +1.3 | ≤ +1.2 |
| H_A / H_B / H_C / H_END / H_G | −2.1 / −5.6 / −0.4 / −1.4 / −0.7 | similar | 0 / −3.1 / 0 / 0 / 0 | 0 | 0 / +3.1 / 0 / 0 / 0 |
| H_E, H_D, H_S0, DJ_B, S_PF_C, other real segments, Z_A | \|Δ\| ≤ 0.3 | ≤ 0.3 | 0 | 0 | 0 |

- **Interaction:** no request has a material B1 interaction (|I| ≥ 10 mm). The largest are +3.1 mm (H_B) and +2.2 mm
  (RH1).
- **Normalisation:** B1 relative changes are reported per request in `analysis/summary.json`. Effects are not pooled.

**Context against X_r and W119 Y.**

- G1F1 meaningfully improves over X_r on H_RH1_none, H_RH1_RBpos, H_RH2_RBpos and DJ_R. These are the W119 findings,
  reproduced bit-exactly.
- G1F0 meaningfully improves over X_r only on H_RH2_RBpos.
- G0F0 and G0F1 never do.

## 8. Reach, flex floor and guide shortfall

**Target statuses over all transition frames of contract-valid requests:**

| | REACHED | FLEX_FLOOR_CLAMPED | UNREACHABLE_OUTER |
|---|---|---|---|
| G0F0 | 225 | 0 | 0 |
| G0F1 | 163 | 62 | 0 |
| G1F0 | 185 | 0 | 40 |
| G1F1 | 167 | 18 | 40 |

- No UNREACHABLE_INNER frames and no nonconvergence.
- **BEND_PLANE_AMBIGUOUS:** 11 in each G0 variant, 6 in each G1 variant.
- **BRANCH_FLIP_VS_PREVIOUS:** 12 (G0F0), 8 (G0F1), 4 (G1F0), 9 (G1F1).

**RH1 release (H_RH1_none).**

- The G1 guide is unreachable on 7 of 8 frames for both F levels. The Blender shortfall to the guide is:
  - G1F0: 54 mm at 119, falling to 4 mm;
  - G1F1: 56 mm at 119, falling to 2 mm.
- **This is not guide tracking.** G1 moves the wrist *toward* an unreachable world target by extending the arm.
- **Where B1 sits:** B1 is the release jump at 118→119, which is frame 119. The floor does not act there, since G0F1
  is REACHED at 119. So **floor@G0 changes B1 by exactly 0**, and the RH1 B1 reduction comes from the guide direction.
  This differs from W119's reading that RH1 was "mainly the floor clamp". W119 is not re-decided.
- **What the floor does on RH1 is articulation:**
  - G0F1 removes X/G0F0's near-straight frame 122 (min flex 0.84° → 10.65°), with no B1 change.
  - G1F1 avoids G1F0's straight arm on 119–125 (min flex 0.00° → 10.32°).
- **Cost of the floor at G1:** a bend flip at 119 relative to G1F0. G1F0's straight arm has no defined bend normal,
  and X, G0F0 and G0F1 all carry the same flip at 119 relative to U.
- **On RBneg:** G1F1 additionally flips at 123.

**RH2 release (H_RH2_RBpos).**

- Every target in every variant is REACHED, with a Blender residual below 1e-6 m.
- The floor never binds, so G1F0 = G1F1.
- **The gain is the guide:**
  - the release-jump compensation falls from 217 mm (X/G0) to 9.6 mm;
  - the inner-transition peak falls from 120 mm to 47 mm;
  - D2 rises from 0.144 to 0.156 m (within the 25 % rule).

**Unreachable-guide control (NG_GUIDE_UNREACH, G1F1):** ACCEPTed with 16/16 frames UNREACHABLE_OUTER and a Blender
shortfall of 83–124 mm. The pins pass.

## 9. Elbow, bend-flip, near-straight and temporal costs

- **Without the floor (G1F0), the guide introduces near-straight frames:**
  - RH1 family: 119–125;
  - DJ_R;
  - S_RA_pos_154_182: frame 183;
  - S_RA_pos_78_107;
  - S_RB_pos_154_182: frames 183 and 187–189.
  This is why GUIDE fails M_clean.
- **New flips:**
  - from the floor: S_RB_pos_154_182 at 187 (G0F1 vs G0F0) and the RH1/DJ_R entries listed in §8;
  - from G1F1: H_RH1_RBneg at 123 (as in W119).
- **Synthetic D3 (rotation-step excess):** G1 raises it on H_A (+3.8), H_C (+6.8), H_END (+4.4) and H_G (+2.1) °/frame,
  each below the 25 %/2° worsening rule, or else with B1 unchanged.
- **Mean elbow deviation from U:** changes by ≤ 15 mm, with the largest on H_END (−15.4 mm under G1).
- **Endpoint jumps** (`endpoint_jumps`, all variants):
  - entry jumps are 0 on RH1 because the entry margin is 0;
  - the raw source jumps are unchanged in every variant: RH1 0.328 m at 118→119; RH2 0.484–0.487 m at 153→154.

## 10. Source-relative vs raw motion

- Each variant reports both raw wrist steps and edit-induced (vs U) compensation steps; see `render/charts/steps_*.png`
  and `endpoint_jumps`.
- **RH1 release jump:** raw 0.262 m (X/G0) and 0.273 m (G1), against U's own 0.309 m.
- **RH2 release jump:** raw 0.411 m (X/G0) and 0.416 m (G1), against U's 0.409 m.
- The large raw steps are U's source discontinuities and are not edit defects. No source jump or timebase was repaired.

## 11. Actual footprint and viewer truth

For every ACCEPTed X_r and factor variant:

- **Hard pin:** pin residual ≤ 1 mm. The maximum is 0.81 mm, on the same straight-arm frame as in W116–W119. There are
  5,050 accepted pin-frame evaluations and 0 over 1 mm.
- **Footprint:**
  - actual changed frames ⊆ planned frames ⊆ budget;
  - planned frames are equal across the four variants and X;
  - changed channels are only upper/lower arm;
  - no outside-arm point or root change;
  - the user-edit channel and `__object__` are bit-equal to U;
  - collateral is 0 m.
- **Display truth check:** every proposed and result display spec of every variant passes the independent check
  against raw Blender frames. Preview id equals committed id. REJECTed candidates never move a ref and are labelled
  REJECTED.

`render/timelines/` contains 70 timelines with separate rows for:

1. the hard-pin domain;
2. the authorised automatic budget, and the planned domain;
3. the actual Blender footprint per variant, with the per-variant changed Bones;
4. per-variant target status (reached, floor-clamped, unreachable, ambiguous).

The header shows each decision, class and reasons. These are scratch diagnostics, not a Production UI and not a
usability study. The palette validator was not run because Node is absent. Colours are the reference categorical slots
1–6, in fixed order, with line-style and marker as secondary encoding.

## 12. Playback / visual review (agent-only)

**Method.** `review/visual_review.json` (`ecfffb1e…`) was written after render and before decide. The agent read the
rendered GIF frames as consecutive-frame sequence sheets, `review/sequences/*.png`, showing 7 variants (U, F_r, X_r,
G0F0, G0F1, G1F0, G1F1). For RBneg it also used full-size crops. **Real-time playback was not available to the agent.**

**Inspected:** the 8 required cases (RH1 122, RH2 182, RBneg, H_G, H_C near-singular, S_RB_pos_109_118,
S_RB_pos_86_106, S_RA_pos_194_198).

**Findings by case:**

- **H_RH2_RBpos, frames 182–185 — guide visible.**
  - X, G0F0 and G0F1 fold the forearm back into a sharp V beside the torso.
  - G1F0 and G1F1 let it hang close to U's pose.
  - The variants converge by 186–188.
  - Possible forearm–hip contact is **UNKNOWN**; bone proxies cannot decide it.
- **H_RH1_none / H_RH1_RBneg — a difference is visible, but no improvement is visually established.**
  - The 118→119 jump is the source discontinuity, identical in all variants.
  - G1 holds the forearm further out toward the unreachable guide.
  - G1F0 is a visibly rigid straight arm at 119–125.
  - X, G0F0 and G0F1 are indistinguishable at review scale.
  - The G1F1 flip at 123 (RBneg) is **not discernible** in the front view; it is reported numerically only.
- **H_C — visible articulation difference, no B1 benefit.** G1 bends the elbow into a V at 26–29 where X/G0 stay
  nearly straight.
- **Other cases:**
  - H_G and S_RB_pos_86_106: no visible difference. On S_RB_pos_86_106 the tile framing clips the figure.
  - S_RB_pos_109_118 and S_RA_pos_194_198: no transition, so all variants equal F_r outside the pins.

**Not reviewed:**

- real-time playback;
- `index.html` in a browser;
- most charts and timelines;
- every request outside the review set.

**Not claimed:** a metric PASS or the agent's view is no claim of anatomical plausibility, user satisfaction,
reference fidelity or engine readiness.

## 13. Negative controls, hard failures, revert and repeatability

**All 29 W120 negative controls returned their declared decision, class and required reasons (V13).**

| Control(s) | Result |
|---|---|
| NG_BUDGET (Z_A with the W119 M_MAX=8 budget) | CONTRACT_VIOLATION `SEARCH_BUDGET_MISMATCH` |
| NG_MARGIN | CONTRACT_VIOLATION `MARGIN_BUDGET_MISMATCH` |
| NG_PARENT / NG_U / NG_RIG / NG_CDIGEST / NG_AUTH | CONTRACT_VIOLATION, each with its own AB_FAIRNESS reason |
| NG_PIN_COMBINED / RG_PIN_COMBINED / NG_FALSE | CONTRACT_VIOLATION with EVALUATED_PIN_UNSATISFIED |
| NG_NONCONV | SOLVER_FAILURE |
| NG_NONCONV_PIN (combined) | SOLVER_FAILURE |
| NG_NONCONV_MARGIN (combined) | CONTRACT_VIOLATION |
| NG_NONCONV_BONE (combined) | SOLVER_FAILURE |
| NG_BONE / RG_BONE | UNAUTHORIZED_CHANNEL_CHANGE |
| NG_ROOT | ROOT_MOVED |
| NG_OUTSIDE / NG_PLANNED | FOOTPRINT_EXCEEDS_DECLARED_DOMAIN |
| NG_DISPLAY | DISPLAY_FOOTPRINT_MISMATCH |
| NG_GUIDE_UNREACH / NG_AMBIG / NG_FLIP | ACCEPT. NG_AMBIG: all frames BEND_PLANE_AMBIGUOUS. NG_FLIP: flip flags and Blender-introduced flips |
| NG_INFEASIBLE / NG_INFEASIBLE_U | INFEASIBLE_REQUEST (the infeasible class wins over the fairness error) |
| NG_CONFLICT / NG_DOF | INFEASIBLE_REQUEST |
| NG_SHARED / NG_SIBLING | BASELINE_NOT_ISOLATED, at value and Blender level |

- **Display controls:** DG1–DG5 all fail the truth check, as required. This includes "REJECTed shown as ACCEPTED".
- **Lineage:**
  - **Synthetic:**
    - S1 commits H_G G1F0 (ACCEPT);
    - S2 NG_PIN_COMBINED is REJECTed and the head is unchanged;
    - S3 H_A G0F1 against the old head gives STALE_PARENT;
    - S4 revert: values byte-equal, Blender bit-equal to P__G.
  - **Real:**
    - R1 commits H_RH1_none G0F1;
    - R2 RG_PIN_COMBINED is REJECTed and the head is unchanged;
    - R3 gives STALE_PARENT;
    - R4 revert: Blender bit-equal to v0.
  - Every X_r and variant was committed on its own ref.
- **Repeatability:** builds r1 and r2 are byte-identical (states and infos). Evaluations e1 and e2 are identical (the
  same file SHA-256 `36bbc1fe…` / `c769b55b…`).
- **Source guard:** unchanged over all stages, and the four manifests re-verify.
- **Hard failures:** 0 accepted pin frames over 1 mm, 0 infeasible requests accepted, 0 nonconvergences outside the
  fault controls, 0 fallbacks.

## 14. Alternative explanations, confounds and non-claims

**Limits of the GUIDE support:**

- **One reachable real event.** H_RH2_RBpos is the only feasible RH2 member. DJ_R shares its window and edit. U has its
  own bend flip at 182.
- **RH1 is the same event three times.** The RH1 guide gain needs the floor to avoid a straight arm, and it moves the
  wrist toward an unreachable target, up to 56 mm short.
- **Synthetic motion:** no meaningful gain anywhere, including held-out H_G.

**FLOOR has no B1 effect** on any real request: floor@G0 is 0 mm on RH1/RH2. It has a −3.1 mm effect on H_B. Its
observed role is articulation:

- it removes near-straight frames;
- sometimes it introduces a flip, e.g. S_RB_pos_154_182 at 187.

**B1 is a peak.** A gain at one boundary transition can coexist with other frames getting worse. D2/D3, the endpoint
jumps and the flex series are reported for that reason.

**G0 is not X.** G0F0 reconstructs X's wrist but not its elbow or roll, with elbow differences up to 62.6 mm on RH2. The
G contrast is therefore measured under the shared solver, not against X_r itself.

**Not claimed:**

- generalisation beyond this GT-free clip and synthetic arm;
- anatomical plausibility or contact;
- performer fidelity;
- FBX/engine survival;
- Production readiness;
- any user-facing joint, straight-arm or smoothness limit. The 1 mm pin criterion, the 5° near-straight reference and
  the W119 comparator are experiment-only diagnostics.

The W116–W119 verdicts are unchanged and none is reassigned.

**Negative findings:**

- no synthetic benefit of either mechanism;
- no B1 benefit of the floor;
- the guide without the floor straightens arms;
- the guide with the floor flips the bend on RBneg;
- the guide changes elbow articulation visibly on near-singular H_C without B1 benefit;
- the frozen W119 precedence maps "nonconvergence + scope violation" to SOLVER_FAILURE.

**Unresolved uncertainties:**

- independent re-execution;
- real-time playback judgement;
- contact at RH2;
- whether RH2-type gains recur on other clips;
- whether the 182 benefit depends on U's own flip there.

## 15. Files, tests, hashes and reproduction

**Completed:** Required Evidence Parts 1–7, every declared stage, the agent review of the 8 required cases, and the
manifest.

**Not completed:** independent review; real-time playback review.

**Changed files** (all new):

- `scripts/w120_factor_attribution_base.py`
- `scripts/w120_factor_attribution.py`
- `tests/test_w120_factor_attribution.py`
- this Worklog

**Revisions:**

- start `2d2fb07`;
- pre-outcome instrument `f39df85` (the **final source revision**);
- the Worklog commit;
- the marker commit.

**Pre-declaration work, disclosed:**

- the focused tests;
- preflight and calibrate, which include the G1F1 values-digest parity check against the published W119 Y (identity
  only, no quality);
- one crash-only smoke run of preflight through decide, in `C:/Users/dna10/animcv-w120-smoke` with a dummy review.
  Stdout of build through decide was discarded unread. Every stage exited 0 with empty stderr. No code changed, and the
  directory was deleted.

**Tests (AGENT-RUN):**

- `tests/test_w120_factor_attribution.py`: **18 passed** (`results/w120_tests.xml`). They cover:
  - G1F1 = W119 Y;
  - four-variant equality of domain, pins, authorisation and budget;
  - determinism;
  - G0/G1 targets;
  - the floor factor;
  - unreachable target without the floor;
  - the G0 reference missing;
  - missing DOF;
  - zero budget and the W119 defect;
  - fairness mismatches;
  - cross-factor identity;
  - exhaustive precedence;
  - combined-error precedence;
  - consistency of the negative-control expectations;
  - decisions and display truth;
  - forced flip;
  - factorial and interaction;
  - endpoint jumps and variant difference.
- **Regressions: 147 passed** (`results/w120_regression.xml`): W119, W118, W117, W116, W115, W114, W105 rig utility,
  fk_solver, canonical_pose_parity and retarget_solver.
- **Failed:** none.
- **NOT RUN:** the full suite, FBX export/import and engines.

**Key artefacts** (`C:/Users/dna10/animcv-w120/results`; SHA-256 prefixes):

| Artefact | SHA-256 |
|---|---|
| declare/declaration.json | 9b3eca97… |
| calibrate/frozen_controls.json / calibration.json | aa9eec06… / 3ce25939… |
| preflight/preflight.json | 32f059c5… |
| build_r1/real_states.json / syn_states.json (= r2) | cc358721… / 3667a44c… |
| eval_e1/real_eval.json / syn_eval.json (= e2) | 36bbc1fe… / c769b55b… |
| assess/assess_report.json | 206e15ec… |
| review/visual_review.json | ecfffb1e… |
| decision/decision.json | 1029ce92… |
| render/index.html | 09e9fa58… |
| analysis/summary.json (post-decision tabulation) | 500c9324… |

`evidence_manifest.json` = `2f2f4412e2f4ede495a2868a995dee2fa622afcfeea1c337fd93a05f855c4e1b` (2,157 files, excluding
the stores), written before this Worklog.

**Reproduction** (Git Bash on Main, repo root at `f39df85`; every stage refuses an existing directory):

```
PY=C:/Projects/AnimCV/.venv_build_full/Scripts/python.exe; R=C:/Users/dna10/animcv-w120/results
S=9b3eca97e9772060bc265914c2da7b1efe7bcd304810bccea19ac25a74b5c25b; D="--declaration $R/declare/declaration.json --declaration-sha $S"
$PY -X utf8 scripts/w120_factor_attribution.py preflight --out-dir $R/preflight          # binds start HEAD 2d2fb07
$PY -X utf8 scripts/w120_factor_attribution.py calibrate --preflight $R/preflight --out-dir $R/calibrate
$PY -X utf8 scripts/w120_factor_attribution.py declare --preflight $R/preflight --calibration $R/calibrate --out-dir $R/declare
$PY -X utf8 scripts/w120_factor_attribution.py build $D --tag r1 --out-dir $R/build_r1
$PY -X utf8 scripts/w120_factor_attribution.py build $D --tag r2 --out-dir $R/build_r2
$PY -X utf8 scripts/w120_factor_attribution.py evaluate --build-dir $R/build_r1 --preflight $R/preflight --out-dir $R/eval_e1
$PY -X utf8 scripts/w120_factor_attribution.py evaluate --build-dir $R/build_r1 --preflight $R/preflight --out-dir $R/eval_e2
$PY -X utf8 scripts/w120_factor_attribution.py assess $D --build-dir $R/build_r1 --eval-dir $R/eval_e1 --out-dir $R/assess
$PY -X utf8 scripts/w120_factor_attribution.py render $D --build-dir $R/build_r1 --assess-dir $R/assess --out-dir $R/render
# agent review: $R/review/visual_review.json (+ review/sequences from render/frames)
$PY -X utf8 scripts/w120_factor_attribution.py decide $D --build-dir $R/build_r1 --build-dir2 $R/build_r2 --eval-dir $R/eval_e1 --eval-dir2 $R/eval_e2 --assess-dir $R/assess --visual-review $R/review/visual_review.json --out-dir $R/decision
$PY -X utf8 scripts/w120_factor_attribution.py manifest --base C:/Users/dna10/animcv-w120 --declaration-sha $S
$PY -X utf8 -m pytest tests/test_w120_factor_attribution.py -q
```

The stages read the W116–W119 artefacts under `C:/Users/dna10/animcv-w11{6,7,8,9}`. Blender is
`C:/Program Files/Blender Foundation/Blender 5.1`.

**Resources and environment:**

- **Timing:** preflight 6 s; calibrate 13 s (including the 262,144-case precedence check, about 0.7 s); each build
  ≈ 12 s; each Blender evaluation ≈ 17 s; assess ≈ 20 s including revert evaluations; render ≈ 43 s; tests 1.3 s;
  regressions 6 s.
- **Storage:** 550 MB under `C:\Users\dna10\animcv-w120\`. It is local only and not published.
- **Software:** Python 3.11.9, numpy 1.26.4, Blender 5.1.0, Windows 10.0.26200.
- **Scope of execution:** Main only. No LabServer63/69, network download, package change or GPU inference. No
  evaluation GT exists or was read.
- **Git:** single branch `arch/single_frame_first`. No branch, merge, rebase, reset or force operation.

AUTOMATION_BATCH_STATUS: COMPLETE
BATCH_NUMBER: 120
