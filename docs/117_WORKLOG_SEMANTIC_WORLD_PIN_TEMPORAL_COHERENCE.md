# Worklog — Semantic World Pin: Temporal Coherence and Compensating Articulation (2026-11-06)

Batch 117. Evidence-only scratch experiment on `arch/single_frame_first`, run unattended on Main (local RTX 5080 host;
no GPU inference was needed). No `src/` module, W116 script/test/Worklog/CASE, rig, rest pose, Action, FBX pipeline,
model, threshold or timebase was changed. `docs/next_batch.md` was read only.

## 1. Official verdict and frozen gates

**Verdict: CASE C — NOT SUPPORTED** (frozen rule, `results/decision/decision.json` `f0d70751…`; declaration
`fab62081…` hashed and committed (`9e9c1cd`) before any evaluation build).

Frozen rule: D if any V fails. Otherwise C if any C gate fails, or if Q_syn(v) and Q_real(v) both fail for every variant.
Otherwise A if some variant has Q_syn, Q_real and QV. Otherwise B.

| Gate group | Result |
|---|---|
| V1–V9 validity | **all pass** (authentication, parent reproduction, W116 control parity, independent evaluator, meaningful asymmetric design, timing semantics, oracle soundness, repeatability, complete population) |
| C1 accept soundness, C2 negative controls, C3 atomicity, C4 lineage/reversal, C5 source immutability | **pass** |
| C6 no new hard failure | **fail**: on `H_RH2_RBpos` (real hold 154–181 under the +5° side bend RB_pos; every pin frame oracle-FEASIBLE) F ACCEPTs, while T and TE REJECT with `SOLVER_FAILURE NON_CONVERGED` |
| Q_syn(T), Q_real(T), Q_syn(TE), Q_real(TE) | **all fail** (§6, §8) |
| QV(T), QV(TE) | fail (no request qualified; visual review §9) |

C follows from C6 alone, and independently from "no variant passes Q_syn or Q_real". W116's official **CASE B is
unchanged**; W117 does not reassess it.

**Separated levels of evidence** (none of them changes the CASE):

| Level | Finding |
|---|---|
| Synthetic correctness | T and TE reach the hard pin exactly on every accepted request (Blender residual ≤ 1.04e-7 m on synthetic evaluation holds). Every declared negative control returned its declared class (21/21 candidate checks). |
| Rig output correctness | Across all ACCEPTed F/T/TE requests: 0 pin frames over 1 mm, 0 unauthorised channel changes, 0 changes outside the declared temporal domain, 0 evaluated points outside the arm domain changed vs U (right wrist, neck, head, pelvis: exactly 0 m vs U), root matrix equal to U. |
| Temporal usefulness | T (swivel choice on pin frames only) gives no material improvement on any held-out synthetic case. On the real clip it improves some requests and worsens others under the frozen rule. TE (declared transition margins) visibly softens entry and release, but on every evaluation request it also moves the elbow away from the user's unpinned motion beyond the declared side-effect limit. |
| Architecture viability | Hard-pin, ownership, atomicity and reversal guarantees survived the temporal resolver. Within this ownership model, the temporal layer did not show a net, unambiguous animation benefit. Its absolute convergence criterion also produced one new hard failure on a feasible real request. |

## 2. Baseline and environment fingerprints

- **Starting commit** `4ea6dd4a593805c915c302da6c06022a88aaa8da`. It equalled HEAD and `origin/arch/single_frame_first` at
  start (fetched; no divergence).
- **Instruction blob** `b193782c6a9090672f16d848b05f01942a6cf78d`, verified before any change. The Main-local lock
  holds batch 117, this blob and this commit.
- **Historical Worklogs** W113–W116 are blob-unchanged and keep their COMPLETE markers.
- **W116 authentication (re-run):**
  - `evidence_manifest.json` = `117645586c…`: all 246 listed files re-hash, both at preflight and after the run.
  - Declaration `ad4fe0df…`, decision `bcf6ee3a…`, frozen CASE B.
- **Inputs (re-hashed, manifest-bound as in W116):**
  - W102 observations `6784357f…` and clip `0dda8a5f…`;
  - W105 manifest `cb0eca93…` with its video_cases, video_sources and render audit;
  - rig FBX `c2947ba7…`; rest snapshot `1dcdc826…`;
  - W115 manifest and the files bound through it.
- **Parent v0.** Rebuilt with the unchanged W116 code. It is canonical-equal to the W116 parent (state `26aa62f4…`).
  A fresh Blender 5.1.0 evaluation gives:
  - the W105 FIXED value digest `4d168d95234cc46e…` with 32 F-curves;
  - all 199 frames bit-equal to the W116 probe.
- **Runtime:** Python 3.11.9, numpy 1.26.4, scipy 1.17.1, Blender 5.1.0, Windows 10.0.26200.

**Directly re-run in W117:** the parent build and its Blender evaluation; the W116 resolver on all frames; Blender
evaluation of the W116-identical F/U states.

**Inherited from authenticated W116 artefacts only:** the W116 synthetic `syn_wrist` binding record (used only for the
wrong-rig control N5), and the W116 decision/CASE.

## 3. W116 control parity and declared population

**Parity (V3).** The unchanged W116 resolver reproduces W116 exactly:
- statuses on all 199 frames of RA_pos, RA_neg, RB_pos, RB_neg and RC_raise;
- the W116 control/candidate state ids;
- the RH1/RH2 interval candidate ids.

W117's whole-clip F and U states have the W116 ids. Their Blender evaluation is bit-equal to W116 `eval_e1` on all 199
frames (value digests equal). F is therefore exactly the W116 framewise control.

**Shared evaluator.** One evaluator serves every candidate: `scripts/w116_blender_stages.py evaluate` (unchanged), plus
the W117 evaluator functions, which read only Blender world points, stored F-curve quaternions, matrix_world and
Blender rest lengths. A claimed-SOLVED candidate that returns the unpinned control is rejected
`EVALUATED_PIN_UNSATISFIED` (synthetic N10, real RN10). `NO_CHANGE_NEEDED` counts as a passing frame (NOOP_A, anchors).

**Population** (44 requests; every request is reported):

| Group | Requests |
|---|---|
| Synthetic evaluation | H_A, H_B (asymmetric holds); H_C (near-singular hold, min slack 2 mm); PF_C whole (40 per-frame pins, near-straight arm) and its oracle segment S_PF_C_0_29 |
| Synthetic negative / noop / development | H_Cx (beyond reach); N2–N14; NOOP_A (trivially satisfied, not evidence); H_D (development split, excluded from gates) |
| Real per-frame families (W116 edits) | whole-clip W_RA_pos, W_RA_neg, W_RB_pos, W_RB_neg, W_RC_raise; oracle segments S_RA_pos_{60_61, 63_64, 78_107, 109_118, 154_182, 194_198}, S_RB_pos_{86_106, 109_118, 154_182} |
| Real holds | H_RH1_{none, RBpos, RBneg} (109–118), H_RH2_{none, RBpos, RBneg} (154–181); F, T, TE each |
| Real negative | RN10–RN13 (tampered H_RH1_none T) |

The low-evidence windows 109–118 and 154–181 appear as labelled subpopulations: the RH1/RH2 holds and the
S_*_109_118 and S_*_154_182 segments. They are not GT.

**Feasibility oracle.** It is computed independently of every solver: W116 `gt_feasibility` on the Blender-evaluated
U shoulder and Blender rest lengths. Both signs of every side bend are evaluated; no sign was preselected.

## 4. Solver, authorised DOF and parameter freeze

**T: `temporal_two_bone`** (`scripts/w117_temporal_pin_base.py`).
- Each pin frame is solved with the W116 law-of-cosines two-bone geometry: same reach band (flex 0–150°), same
  1 mm clamp rule, same distal aim.
- So every swivel angle reaches the same wrist position as F. Only the elbow swivel (bend plane) is free.
- The swivel sequence minimises J = Σ|c_k|² + λ Σ|c_k − c_{k−1}|², where c_k is the elbow compensation (vs U), in the
  upper arm's body-attached rest frame, divided by L1. With λ = 0 the minimiser is exactly the W116 pole (unit test).
- Search: a Viterbi DP over 360 swivel bins, then L-BFGS-B started from the DP solution and from the F swivel; the
  lower J is kept.
- Convergence is declared iff max|∂J/∂φ| ≤ GTOL = 1e-7.
- Frames already within 1 mm under U are anchored as NO_CHANGE_NEEDED (values = U).
- There is no fallback to F. Non-two-bone structures return RESOLVER_UNSUPPORTED; no DOF returns INSUFFICIENT_DOF.

**TE.** T plus MARGIN = 4 guide frames on each side of a hold. The guide is the unpinned wrist plus the boundary offset ×
(1 − smoothstep). Guide frames are **not pins**. They are clamped to reach and change only the authorised arm channels.

**Authorised DOF:** real `upperarm_l`/`lowerarm_l`; synthetic `upper`/`lower`. Never authorised: the user-edit channel
(spine_01 / base), `__object__`, root bones and unkeyed channels.

**Frozen constants:** λ 20, GRID 360, MAXITER 1000, GTOL 1e-7, runtime budget 120 s, MARGIN 4.

**No tuning.** The development split (trajectory D) was used only for a smoke run before the declaration. The disclosed
pre-declaration work was:
- unit tests, with one test-threshold fix;
- preflight and calibration;
- a dev-only Python-FK run of H_D and a 199-frame dev runtime check.

No evaluation request was solved before the declaration.

**Predicted before outcomes (and confirmed):**
- Elbow flex is fixed by |target − shoulder| for a two-bone chain. T cannot reduce flex compensation: S3 is identical for
  F and T on every request.
- F and T have identical entry/release jumps: D4 is equal on every hold.

## 5. Synthetic asymmetric and near-singular results

**Rig.** `W117_SYN_ARM`:
- base 0.5 m, plus an unkeyed neck as collateral monitor;
- upper 0.30 m with an offset shoulder, lower 0.25 m, plus an unkeyed hand;
- 40 frames per trajectory.

The user edit is base +8° on all frames; it moves the unpinned wrist ≥ 5.5 cm on every hold frame. The hold target is
the parent wrist at the capture frame. Asymmetry was checked on Blender-evaluated U (V5):

| Hold | Entry offset | Release offset | Pre-entry vs post-release wrist | Min reach slack |
|---|---|---|---|---|
| H_A (14–25) | 0.077 m | 0.423 m | 0.392 m | 36 mm |
| H_B (11–24) | 0.055 | 0.288 | 0.363 | 71 mm |
| H_C (12–24) | 0.084 | 0.189 | 0.191 | **2.0 mm (near-singular, min flex 5.3°)** |

Defect metrics (D1 introduced bend flips; D2 peak elbow-compensation step; D3 peak arm rotation-step excess over U;
D4 peak boundary wrist-compensation step) and side effects (S1 mean elbow deviation from U):

| Request | F | T | TE |
|---|---|---|---|
| H_A | D1 0, D2 214 mm, D3 37.5°, D4 423 mm, S1 80 mm | ≈ F (D2 214, S1 80) | D2 110, D4 125, **S1 112, S3 +19°** |
| H_B | D1 1, D2 96, D3 110°, D4 288, S1 43 | D1 1 (flip moved to pin entry), D2 119, D3 78, S1 53 | D1 0, D3 34, D4 85, **S1 76** |
| H_C | D2 85, D3 19°, D4 189, S1 14 | ≈ F | D2 37, D3 12, D4 56, **S1 27** |
| S_PF_C_0_29 (19 near-singular frames) | D2 30, D3 8.7 | ≈ F | — |

- **T** shows no material improvement on any synthetic case, and no material worsening either.
- **TE** improves D4 (≥ 3×) and, on H_B/H_C, D1/D2/D3. On every case it exceeds the declared S1 side-effect limit
  (+10 mm mean elbow deviation from U); on H_A it also exceeds S3. Under the frozen rule this counts as worse, not
  improved.
- The development case H_D was the only case where TE met the rule (excluded by declaration).
- **Infeasible:** H_Cx (4 mm beyond reach) is REJECTed INFEASIBLE_REQUEST by F, T and TE. W_PF_C (whole clip,
  infeasible frames) is REJECTed atomically by F and T.

## 6. Authentic real-rig U/F/T comparison

Population: `real_animcv_observation`-derived W105 S0-only rig output, 199 frames. The source is GT-free and the video
subject identity is UNKNOWN on 199/199 frames. Only the output rig identity is known.

| Request | F / T / TE decision | Result under the frozen rule |
|---|---|---|
| W_RA_neg (199 pins) | ACC / ACC | T improved: D2 58.7 → 22.9 mm, D3 23.5 → 2.7°/frame; F introduces a flip at 154 that T removes. Counted as improvement. |
| W_RB_neg | ACC / ACC | T improved: D3 32.2 → 6.4°/frame; D2 81 → 44 mm (not ≤ 0.5×). Counted as improvement. |
| W_RC_raise | ACC / ACC | T improved D2 70 → 20 mm and D3 28 → 3.3°, but moved a bend flip from 152 (shared with U) to 154 (introduced). Counted as **worse**. |
| H_RH1_none | ACC / ACC / ACC | T removes F's introduced flip (D1 1 → 0), S1 +6.9 mm. Improvement. TE also improves (D1, D4 157 → 58 mm). |
| H_RH1_RBneg, H_RH1_RBpos | ACC / ACC / ACC | T and TE remove the flip, but S1 rises by +10.3 / +11.4 mm (T) → **worse** |
| H_RH2_RBpos | **ACC / REJ / REJ** | T/TE `NON_CONVERGED` → C6 failure (§7) |
| H_RH2_none, H_RH2_RBneg | REJ / REJ / REJ | infeasible frames (oracle). F: INFEASIBLE_REQUEST. T/TE on H_RH2_none: **SOLVER_FAILURE** (§10). |
| W_RA_pos, W_RB_pos | REJ / REJ | INFEASIBLE_REQUEST (atomic, 121 / 138 unreachable frames) |
| 9 oracle segments of RA_pos/RB_pos | all ACC / ACC | T ≈ F on all (no improvement, no worsening); these include the near-straight frames (S_RA_pos_194_198: flex 0°, residual 0.34 mm for both) |

**Shared across requests:**
- Both side-bend signs are evaluated. The low-evidence windows behave like the rest: S_*_109_118 and S_*_154_182
  have T ≈ F.
- The RH2 holds sit on the parent's own 0.485 m wrist jump at 153→154. Peak elbow-compensation steps of 0.31 m
  appear there for every candidate, and U itself has bend flips at 152 and 182.
- Raw absolute values are in `assess_report.json` per request: wrist/elbow paths for parent, U and each candidate;
  flex per frame; per-transition steps.

## 7. Blender-evaluated pin residuals and failure counts

**Accepted requests:**

| Variant | Accepted | Max residual |
|---|---|---|
| F | 23 | 0.81 mm (the same tolerance-clamped straight-arm frames as W116) |
| T | 21 | 0.81 mm |
| TE | 7 | 5.5e-7 m |

Accepted pin frames over 1 mm: 0.

**Decision counts:**

| Role | F | T | TE |
|---|---|---|---|
| Evaluation | 20 ACC / 5 REJ | 19 / 6 | 6 / 3 |
| Negative | 1 ACC (N3, as declared) / 2 REJ | 0 / 16 | 0 / 2 |

**The new hard failure (C6), H_RH2_RBpos.** Both solver starts reach the same J = 21.730139 (vs J_F 22.254). The
remaining max|∂J/∂φ| is 1.37e-7 and 1.29e-7, against the declared absolute GTOL 1e-7; scipy's line search stopped there.

Labelled post-hoc observation, with no CASE effect and nothing re-run: the rejected T and TE candidates were evaluated
anyway and meet every hard pin in Blender (28/28 frames pass, max 0.63 mm). The failure is therefore the declared
convergence criterion, an absolute gradient bound set without an error model for J ≈ 22. It is not an unreachable pin.

It is still a valid controlled result of the frozen solver. GTOL was not changed.

Non-converged runs: H_RH2_RBpos T/TE, H_RH2_none T/TE, and N9 (declared, maxiter 1). Runtime: 40 T runs, max 0.65 s,
total 5.3 s.

## 8. Temporal, articulation and collateral metrics

**Definitions** (declared): time unit = rig-Action frame; integer frames; quaternion steps are sign-safe
(2·acos|⟨q1,q2⟩|; unit-tested with sign-flipped sequences).

**Flex.** Identical for F and T on every request. Peak compensating upper-arm rotation vs U differs by ≤ 1.2° between F
and T on every accepted request.

**Where T changes anything,** it trades lower peak compensation steps (D2/D3) for a larger mean deviation from the
unpinned elbow (S1). This is the expected effect of its objective, not an independent quality gain.

**Collateral.** Zero vs U on every accepted request: right wrist, neck, head and pelvis 0 m; root matrix equal. The
user edit's own effect, present in U, F and T alike, is e.g. W_RB_neg right wrist 0.042 m and neck 0.054 m vs the parent.
It is not attributable to pin compensation.

**Frames changed vs U:** TE changes 8 extra (margin) frames per hold. These are the frames the user did not pin.

## 9. Visual review (rig-only, reproducible)

Renders are in `results/render/`: contact sheets `sheet_*.png` and GIFs `anim_*.gif`. They show U/F/T(/TE) as bone
proxies with the mesh hidden, from a fixed front orthographic camera, with markers for target (green), wrist (blue) and
elbow (orange). The source video is not shown.

The agent viewed `sheet_H_B`, `sheet_H_RH1_none` and `sheet_W_RB_neg`. The other six sheets and all GIFs were rendered
but **not inspected**. The review is in `results/review/visual_review.json` (`a8e648dc…`). It was written after the
numerical Q result was known and is not independent.

| Question | Observation |
|---|---|
| What improved | Only TE visibly softens pin start/stop. In H_B, release spreads over f25–f29 instead of snapping in one frame; in RH1, over 119–122. |
| What became worse | TE holds the arm away from the user's unpinned motion on frames the user did not pin. In H_B, T moves F's mid-hold elbow-side change to the pin entry, where it is abrupt. |
| Anatomically plausible | Not judged: bone proxies, no anatomical reference; the real arm is often near straight. |
| Abrupt pin start/stop | Yes for F and T on every viewed hold; reduced for TE. |
| Merely numerical | T's real-clip D2/D3 reductions (W_RB_neg) were not visible in the viewed frames; the source's own jump at 153→154 dominates. The H_RH1_none flip removal is not discernible at this scale. |
| Would it help an animator | Uncertain. The only visible benefit (boundary easing) requires editing unpinned frames, which is a policy question. |

QV fails for both variants because no request met the numerical Q rule.

## 10. Atomic rejection, state lineage and exact reversal

**Synthetic programme:**
- S1: H_A T is ACCEPTed on `syn/main`.
- S2: N13 (one corrupted frame) is REJECTed on `syn/atomic`; head unchanged.
- S3: the H_A TE candidate, prepared against the old head, is rejected `STALE_PARENT`.
- S4: revert. Values are byte-equal to P__A, and the Blender evaluation is bit-equal (same value digest, all frames).

**Real programme:**
- R1: H_RH1_none T is ACCEPTed (v0 → `d723ee3a`).
- R2: RN13 is REJECTed; head unchanged.
- R3: H_RH1_RBneg F, prepared against v0, is rejected `STALE_PARENT`.
- R4: H_RH2_none T is REJECTed; head unchanged.
- R5: revert (`a9728776`). Values are byte-equal to v0, and the Blender evaluation is bit-equal to v0.

Every lineage chain re-verifies, and refs are unchanged after every REJECT.

**Single-corrupted-frame intervals.** N13 (synthetic frame 20) and RN13 (real frame 113) are each REJECTed as a whole,
with exactly one failing frame and every other frame passing (C3).

**Classification defect (negative finding, outside the declared negative-control list).** On H_RH2_none, which has
oracle-infeasible frames, T reports `SOLVER_FAILURE [NON_CONVERGED, UNREACHABLE]` where F reports
`INFEASIBLE_REQUEST`. The W116 `decide` precedence ranks NON_CONVERGED above UNREACHABLE. Combined with T's convergence
criterion, an infeasible user request is therefore reported as a solver failure. The request is still rejected and the
head is unchanged, but the failure class is not the user-facing reason.

## 11. Independent evaluator and negative controls

All declared controls returned their declared decision and class:

| Control | Decision / class | Reasons |
|---|---|---|
| H_Cx (F/T/TE) | REJECT INFEASIBLE_REQUEST | UNREACHABLE |
| N2: no DOF (F/T) | REJECT INFEASIBLE_REQUEST | INSUFFICIENT_DOF |
| N3: feasible forearm-only target | F ACCEPT; T REJECT SOLVER_FAILURE | T: RESOLVER_UNSUPPORTED (not converted to infeasible) |
| N4: ambiguous binding | REJECT UNSUPPORTED_BINDING | |
| N5: binding record of the W116 rig | REJECT UNSUPPORTED_BINDING | BINDING_RIG_MISMATCH |
| N6: conflicting pins | REJECT INFEASIBLE_REQUEST | CONFLICTING_HARD_CONSTRAINTS |
| N7: altered + re-digested target | REJECT CONTRACT_VIOLATION | TARGET_PROVENANCE_MISMATCH |
| N9: maxiter 1 | REJECT SOLVER_FAILURE | NON_CONVERGED |
| N10 / RN10: false SOLVED | REJECT SOLVER_FAILURE | EVALUATED_PIN_UNSATISFIED |
| N11 / RN11: user-edit channel +1° | REJECT CONTRACT_VIOLATION | |
| N12 / RN12: `__object__` +1° | REJECT CONTRACT_VIOLATION | ROOT_MOVED |
| N13 / RN13: one corrupted frame | REJECT SOLVER_FAILURE | |
| N14: TE change outside the declared temporal domain | REJECT CONTRACT_VIOLATION | CHANGE_OUTSIDE_DECLARED_TEMPORAL_DOMAIN |

**Source.** All declared inputs, the W116 declaration/decision/manifest, the W116 build and evaluation files, and the
parent files are byte-unchanged at every stage. The W116 manifest re-verifies after the run.

## 12. Alternative explanations and quality trade-offs

- **C6 is an instrument choice as much as a solver limit.** An absolute 1e-7 gradient bound is tight for J ≈ 22. A
  relative criterion would likely have accepted H_RH2_RBpos (post-hoc §7). This is not used to re-label the CASE, and
  no claim is made that the criterion is wrong in principle.
- **The objective defines "better".** T optimises smoothness of the *compensation*, so it follows U's own motion,
  including U's flips and the source's discontinuities. A different objective (world-space elbow smoothness,
  anatomical swivel priors) could rank cases differently. Only one objective was tested.
- **S1 as a side effect.** For TE, deviating from U on margin frames is the mechanism of the easing, not an accident.
  Whether that deviation is acceptable is a user-policy question; the frozen rule counted it as worse.
- **D1 counts introduced flips at a transition.** On W_RC_raise, T moved a flip from 152 (shared with U) to 154,
  the source jump frame. This counts as worse even though the total flip count is unchanged (2 vs 2).
- **The real clip is dominated by its own discontinuity** around 153→154 and its near-straight left arm. Many real
  requests offer T no freedom: segments with tiny swivel radius give T ≈ F.

## 13. Source identity, GT-free and timebase limitations

- No claim about the performer: the real source is GT-free and video subject identity is UNKNOWN on 199/199 frames.
  Cross-frame holds and the whole-clip requests are statements about the known output rig only.
- **Timebase:**
  - Source 50 fps. The historical Action is keyed at frame_index + 1 and played at 30 fps (W105 audit `scene_fps` 30).
  - The evaluation scene's factory default is 25 fps; evaluation is frame-indexed, so fps does not enter any evaluated
    value.
  - Every key is CONSTANT; sub-frame evaluation equals the keyed frame (V6).
  - All temporal metrics are per rig-Action frame. No timebase correctness is claimed or repaired.
- The 1 mm criterion remains the provisional W116 diagnostic, not a production tolerance.

## 14. What is not demonstrated

- A temporal resolver that improves Blender-rig World Pin animation over W116 on the declared scope without side
  effects.
- Anatomical plausibility of any arm pose; any straight-arm or compensation policy; any user-facing threshold.
- Robust convergence on large-J real requests.
- Multi-pin, redundant-chain, orientation/contact/motion-preservation behaviour, a Generator, FBX/engine survival, and
  production integration. None was attempted.
- Independent visual review: the review is the agent's own, of three sheets.

**Unresolved; left to the Architecture Owner:**
- Experimental limit: the convergence criterion scale, the failure-class precedence for infeasible + non-converged, and
  the single objective/λ tested.
- User-facing policy: whether a pin request may edit unpinned transition frames, and how much deviation from the
  user's unpinned motion is acceptable to soften a pin boundary.

## 15. Changed files, tests, hashes and replay

**Changed files** (all new):
- `scripts/w117_temporal_pin_base.py`
- `scripts/w117_temporal_pin.py`
- `scripts/w117_blender_timing.py`
- `tests/test_w117_temporal_pin.py`
- this Worklog

W116 scripts are imported unchanged.

**Revisions:**
- start `4ea6dd4`;
- `9e9c1cd`, the pre-outcome instrument (after the declaration hash);
- the Worklog commit, which is the final source revision;
- a final marker commit.

**Completed:** Required Evidence parts 1–6, all declared stages, and the visual review of three sheets.
**Not completed:** visual inspection of the other renders and GIFs; independent review.

**Tests (AGENT-RUN):**
- `tests/test_w117_temporal_pin.py`: **15 passed** (`w117_tests.xml`). They cover:
  - λ = 0 parity with the W116 resolver;
  - exact hard pin and authorised-only changes;
  - unreachable vs unsupported vs insufficient DOF;
  - non-convergence without fallback;
  - anchors and guides;
  - near-singular swivel continuity;
  - quaternion sign-safe first/second-order metrics;
  - flip detection and compare rules;
  - one-bad-frame atomicity;
  - false SOLVED;
  - temporal-domain and user-edit tampering;
  - stale parent and exact revert.
- **Regressions: 98 passed** (`w117_regression.xml`): `test_w116_semantic_pin`, `test_w115_validity`,
  `test_w114_source_preserving_base`, `test_w105_rig_utility`, `test_fk_solver`, `test_canonical_pose_parity`,
  `test_retarget_solver`.
- **Failed:** none.
- **NOT RUN:** the full suite, FBX export/import and engines. Blender was exercised only through the W116/W117 stages.

**Repeated runs.**
- Two builds produced byte-identical states files and identical statuses.
- Two Blender evaluations are identical: real `9300a3d9…`, synthetic `8860dc43…`.
- The build-time U evaluation equals the evaluate-stage evaluation.

**Reproduction** (PowerShell on Main, repo root; every stage refuses an existing output directory):

```
$py="C:\Projects\AnimCV\.venv_build_full\Scripts\python.exe"; $o="C:/Users/dna10/animcv-w117/results"
$s="fab6208100f2ed9074ea125ff9587aff83b3d62ae5ba2cbb3f7c9d9a3ed95bc2"; $d="--declaration $o/declare/declaration.json --declaration-sha $s"
& $py -X utf8 scripts/w117_temporal_pin.py preflight --out-dir $o/preflight
& $py -X utf8 scripts/w117_temporal_pin.py calibrate --preflight $o/preflight --out-dir $o/calibrate
& $py -X utf8 scripts/w117_temporal_pin.py declare --preflight $o/preflight --calibration $o/calibrate --out-dir $o/declare
& $py -X utf8 scripts/w117_temporal_pin.py build $d.Split() --tag r1 --out-dir $o/build_r1
& $py -X utf8 scripts/w117_temporal_pin.py build $d.Split() --tag r2 --out-dir $o/build_r2
& $py -X utf8 scripts/w117_temporal_pin.py evaluate --build-dir $o/build_r1 --preflight $o/preflight --out-dir $o/eval_e1
& $py -X utf8 scripts/w117_temporal_pin.py evaluate --build-dir $o/build_r1 --preflight $o/preflight --out-dir $o/eval_e2
& $py -X utf8 scripts/w117_temporal_pin.py assess $d.Split() --build-dir $o/build_r1 --eval-dir $o/eval_e1 --out-dir $o/assess
& $py -X utf8 scripts/w117_temporal_pin.py render $d.Split() --build-dir $o/build_r1 --eval-dir $o/eval_e1 --assess-dir $o/assess --out-dir $o/render
# visual review: results/review/visual_review.json (agent-authored)
& $py -X utf8 scripts/w117_temporal_pin.py decide $d.Split() --build-dir $o/build_r1 --build-dir2 $o/build_r2 --eval-dir $o/eval_e1 --eval-dir2 $o/eval_e2 --assess-dir $o/assess --visual-review $o/review/visual_review.json --out-dir $o/decision
& $py -X utf8 scripts/w117_temporal_pin.py manifest --base C:/Users/dna10/animcv-w117 --declaration-sha $s
& $py -X utf8 -m pytest tests/test_w117_temporal_pin.py -q
```

Blender is `C:/Program Files/Blender Foundation/Blender 5.1`. The declaration embeds this host's absolute paths and
reads the W116 artefacts under `C:/Users/dna10/animcv-w116`.

**Key artefacts (SHA-256):**

| Artefact | SHA-256 |
|---|---|
| `declare/declaration.json` | `fab62081…` |
| `decision/decision.json` | `f0d70751…` |
| `preflight/preflight.json` | `d3d6836b…` |
| `calibrate/calibration.json` | `40d51836…` |
| `build_r1/real_states.json` (= r2) | `a7b71128…` |
| `build_r1/syn_states.json` (= r2) | `98589c2d…` |
| `eval_e1/real_eval.json` (= e2) | `9300a3d9…` |
| `eval_e1/syn_eval.json` (= e2) | `8860dc43…` |
| `assess/assess_report.json` | `c7f7f8ca…` |
| `review/visual_review.json` | `a8e648dc…` |

`evidence_manifest.json` = `9f90f91aa9980c366ee3185e8e0ee6939db974b0f5d9517ab7cdc6bdd9ee22d8` (780 files, excluding the
content-addressed store), written before this Worklog.

**Resources and environment:**
- Each build took ≈ 9.6 s (including one Blender U evaluation), each Blender evaluation a few seconds, render ≈ 20 s;
  tests 0.7 s, regressions 5.3 s.
- Storage: 189 MB under `C:\Users\dna10\animcv-w117\`. It is local only and not published. The source clip was not
  copied.
- Main only. No GPU inference, network, model download, package change or LabServer63/69 use. No evaluation GT exists or
  was read.
- Single branch `arch/single_frame_first`. No branch, merge, rebase, reset or force operation.
