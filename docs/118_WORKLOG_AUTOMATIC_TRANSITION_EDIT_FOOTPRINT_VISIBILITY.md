# Worklog — Automatic Transition Editing with Explicit Edit-Footprint Visibility (2026-11-06)

Batch 118. Evidence-only scratch experiment on `arch/single_frame_first`, run unattended on Main (local RTX 5080 host;
no GPU inference was needed). No `src/` module, W116/W117 script, test, Worklog, declaration or CASE, rig, rest pose,
Action, FBX pipeline, model, threshold or timebase was changed. `docs/next_batch.md` was read only.

**Scope of the task.** Test whether, under the human-approved policy B (automatic edits outside a WORLD_POSITION_PIN
interval are allowed if the modified range is explicitly visible), a bounded automatic transition editor can:
- keep the hard-pin domain separate from its own edit domain;
- expose a truthful, Blender-verified edit footprint;
- improve boundary motion on synthetic and authentic real-rig replays without new hard-pin, ownership, atomicity or
  reversal failures.

## 1. Official CASE, frozen gates and declaration hash

**Verdict: CASE D — INVALID** (frozen rule, `results/decision/decision.json` `91200bd4…`). The declaration
`5d61b552d2f4239801fac5e686cee5adfafdf4ac2c54f5697904af2ee4652d00` was hashed and the instrument committed (`42517e9`)
before any evaluation build.

Frozen rule: D if any V fails. Otherwise C if any C gate fails, or if Q_syn and Q_real both fail. Otherwise A if Q_syn,
Q_real and QV all pass. Otherwise B.

| Gate | Result |
|---|---|
| V1 authentication, V2 parent reproduction, V3 baseline replay, V5 design, V6 timing, V7 oracle, V8 repeatability, V9 population | pass |
| **V4 independent footprint evaluator** | **fail**: the clause "F footprint ⊆ its pin frames on every request" fails on all 9 real per-frame segment requests `S_RA_pos_*`/`S_RB_pos_*` (see below) |
| C1 hard pin & scope, C2 truthful footprint, C3 negative/display controls, C4 atomicity/lineage/reversal, C5 source immutability, C6 no new hard failure, C7 user edit & collateral | **all pass** |
| Q_syn | fail (worsening flag on `S_PF_C_0_29`) |
| Q_real | fail (0 of 5 real holds meet the improvement rule; worsening flags on 5 real requests) |
| QV | fail (no real request with a visible improvement in the agent review) |

**Why V4 fails: an instrument-scope error in my declaration, not an evaluator error.**
- The frozen W117 build stores the F and T candidates of all segments of one family in a single shared state,
  `SEG_<fam>__F` / `SEG_<fam>__T`.
- The W118 footprint evaluator compares that state with U over the whole clip. It therefore correctly reports changes on
  the sibling segments' pin frames. A read-only check confirms every out-of-pin F/T change of every `S_*` request lies on
  another segment of the same family; there are 31 to 73 such frames per request.
- My V4 wording did not anticipate the shared state. The evaluator was truthful; the declared gate and its
  baseline-population design were wrong. That error also contaminates the F baseline of the segment comparisons:
  - their quality windows (pin run ± 9 frames) include sibling segments for `S_RB_pos_109_118`, `S_RB_pos_86_106`,
    `S_RA_pos_60_61`, `S_RA_pos_63_64`, `S_RA_pos_78_107` and `S_RA_pos_109_118`;
  - the X-vs-F numbers for those six are not a fair A/B.

**Mechanical alternative (labelled; not assigned).** If V4 is set aside, every C gate passes. Q_syn and Q_real both
fail, so the rule would output **C**:
- Q_syn fails because `S_PF_C_0_29` (not a shared-state request) has an introduced near-straight frame;
- Q_real fails because no real hold meets the B1 ratio rule, and several requests introduce a near-straight frame or a
  bend flip.

I do not assign C. The gate was frozen, and re-scoping it now would be outcome-driven. W116 CASE B and W117 CASE C are
unchanged.

**Separated levels of evidence** (none of them changes the CASE):

| Level | Finding |
|---|---|
| Domain separation (H1) | Hard-pin domain, automatic search budget, candidate-declared planned domain and actual Blender footprint are stored separately for all 49 X requests. Planned ⊆ budget on all 49; actual ⊆ planned on every ACCEPTed X. |
| Truthful footprint (H2) | Every display spec of every variant (F, T, TE, X) on all 49 requests passed the independent truth check against raw Blender frames. All 6 display-corruption controls failed it, as intended. The known-footprint control was recovered exactly. Preview state id = committed id; REJECTed X never shown as applied. |
| Guarantees | 29 X ACCEPTed. Max Blender pin residual 0.81 mm (same straight-arm frames as W116/W117). 0 unauthorised channel, 0 out-of-budget, 0 outside-arm point, 0 root changes; user-edit channel and `__object__` bit-equal to U. Atomic rejects, stale parent and exact Blender-bit-equal reverts all hold. |
| Boundary quality (H3) | Synthetic: large reductions of the boundary wrist compensation step on all 7 synthetic holds, including the held-out H_E. Real: about 0.51× on RH1 and 0.77× on RH2/DJ_R, with an introduced near-straight frame on 4 real requests. Mixed. |

## 2. Source, rig, branch, baseline and blob fingerprints

- **Starting commit** `0dbcbbfcc97defd98061d9416a69dac7eae6cca1`.
  - At start it equalled HEAD and `origin/arch/single_frame_first` (fetched; no divergence).
  - Instruction blob `2468904ff2e087fcff65c09729014d1968e6936f`, verified before any change.
  - The Main-local lock holds batch 118, this blob and this commit.
- **Historical Worklogs:** W113–W117 blobs are unchanged from the starting commit and carry their COMPLETE markers. The
  W117 COMPLETE commit `01749bfe…` is an ancestor of HEAD.
- **W117 authentication (Main-local, re-hashed):**
  - `evidence_manifest.json` = `9f90f91a…`: all 780 files re-verify, at preflight and again after assess.
  - Declaration `fab62081…`, decision `f0d70751…`, CASE C.
- **W116 authentication:** manifest `11764558…` (246 files) re-verifies; declaration `ad4fe0df…`, decision `bcf6ee3a…`,
  CASE B.
- **Inputs** (re-hashed, manifest-bound as in W116/W117): W102 observations `6784357f…`; clip `0dda8a5f…`; W105 manifest
  `cb0eca93…` and the files bound through it; rig FBX `c2947ba7…`; rest snapshot `1dcdc826…`; W115 manifest and the files
  bound through it.
- **Parent v0:** rebuilt, canonical-equal to the W117 parent (state `26aa62f4…`). A fresh Blender 5.1.0 evaluation has
  the W105 FIXED digest `4d168d95234cc46e…` and is bit-equal on all 199 frames to the W117 preflight evaluation.
- **Synthetic:**
  - The W117 spec and parents A–D rebuild byte-identically and probe bit-equal to W117.
  - The held-out trajectory E is new in W118.
- **Runtime:** Python 3.11.9, numpy 1.26.4, scipy 1.17.1, Pillow 12.3.0, matplotlib 3.11.0, Blender 5.1.0, Windows 10.0.26200.

## 3. Approved policy vs experimental implementation

- **Policy (human, approved):**
  - out-of-pin automatic edits are allowed, without per-margin user approval, if the actual modified interval is
    explicitly visible;
  - hard pins are never relaxed;
  - the user edit is authoritative, source is immutable, and every state is reversible.
- **Experimental implementation (not policy):** the search budget `M_MAX = 8` frames per side, the selection target
  `SIGMA_STEP_M = 0.010 m/frame`, the material threshold `1e-5 m`, the display resolution 1 mm / 0.1°, and the quality
  rule of §6.
- None of these is proposed as a Production default or a user-facing limit. W117's S1 10 mm cap was not used as a gate;
  elbow deviation is reported as a cost (§9).

## 4. Pin domain, search budget, planned and actual domains

Represented separately for every request (`build_report.json` → `x_info`; `assess_report.json` → `footprint`,
`display_spec_*`):

- **Requested pin frames** and their maximal runs. Disjoint runs are kept, e.g. DJ_B `[[6,10],[20,24]]`.
- **Search budget:** pin frames plus up to M_MAX frames per side of each run.
  - Clipped at the clip boundaries (`CLIP_START`/`CLIP_END`).
  - Split evenly in gaps (`GAP_SPLIT`), so two runs never share a transition frame.
- **Planned domain:** pin frames plus the frames the editor actually chose.
- **Actual footprint (Blender, vs U):**
  - per frame: changed channels and their angles, max point displacement, wrist and elbow displacement, material vs
    numeric-only;
  - per channel: the changed frames;
  - contiguous runs and the union envelope (both kept; never collapsed);
  - out-of-pin runs, outside-arm points and `matrix_world` changes.
- **Status and lineage:** PROPOSED card → commit on its own ref → ACCEPTED (head = candidate id) or REJECTED (head =
  parent id, not applied).

Examples (actual X out-of-pin runs): H_A `[6,13],[26,33]`; DJ_B `[0,5],[11,14],[16,19],[25,32]` (frame 15 unchanged);
H_RH1_none `[119,126]` (the entry needs no edit: pin frame 109 is already satisfied and is shown as "pin unchanged"); DJ_R
`[119,126],[152,153],[182,189]`. No accepted X had numeric-only (sub-material) changed frames.

**Time axis.** Integer Action frame index throughout. No seconds are shown, because the source is 50 fps and the
Action plays at 30 fps.

## 5. Candidate algorithm, parameters and runtime

**X = F hard-pin base + automatic offset-layer transition editor** (`scripts/w118_transition_edit_base.py`).

- **Pin frames:** the frozen W116 framewise analytic values. X is identical to F there; the W117 optimiser and its
  convergence criterion are not used.
- **Transition frames:** each authorised channel receives a pre-multiplied local rotation offset
  `d = q_F(b)·q_U(b)⁻¹`, faded as `q_X = slerp(I, d, 1 − smoothstep(j/(m+1)))·q_U` over the m frames beyond each
  run boundary b.
- **Automatic selection:** per side, the smallest m in [0, available] whose predicted (Python FK) peak wrist compensation
  step is ≤ 10 mm/frame; otherwise m = available (`BUDGET_SATURATED`).
- **Fallback:** none.
- **Runtime:** 45 runs, total 0.38 s, max 0.053 s.

**Chosen margins.**
- Every synthetic hold saturated at 8 frames, except:
  - H_B entry met the target exactly at 8;
  - H_S0 release 3 (entry clipped by the clip start);
  - H_END release 0 (clip end);
  - DJ_B 6 / 4 / 4 / 8 (clip start and gap split).
- Real:
  - RH1 entry 0 (no offset), release 8 (saturated);
  - RH2_RBpos entry 2, release 8.
- Z_A (zero-budget control) equals F exactly: 0 out-of-pin frames.

## 6. Synthetic and real A/B comparisons (U, F, frozen T/TE, X)

**Declared rule** (vs F, same request, Blender output, union window ±9 frames):
- **Improved** = B1 (peak boundary wrist compensation step) X ≤ 0.5·F and F − X ≥ 5 mm.
- **Worse** = any of: B1 doubled; A1 introduced bend flips; A2 rotation-step excess doubled (+5°); A3 elbow compensation
  step doubled (+5 mm); A4 introduced near-straight frame (flex < 5° where U ≥ 5°); A5 elbow compensation acceleration
  doubled.

Values are F → X.

| Request | B1 (m) | Worse flags | Cost: mean elbow dev from U (mm) |
|---|---|---|---|
| H_A | 0.423 → 0.072 | — | 58 → 95 |
| H_B | 0.288 → 0.053 | — (D1 1 → 0) | 32 → 49 |
| H_C (near-singular) | 0.189 → 0.032 | — | 11 → 22 |
| **H_E (held-out)** | 0.287 → 0.048 | — | 49 → 80 |
| H_S0 (clip start) | 0.028 → 0.010 | — | 13 → 14 |
| H_END (clip end) | 0.238 → 0.041 | — | 45 → 76 |
| DJ_B (disjoint) | 0.065 → 0.020 | — | 7 → 15 |
| S_PF_C_0_29 | 0.017 → 0.009 (not ≥ 5 mm) | **A4** (frame 30 near-straight) | 8.6 → 9.0 |
| H_RH1_none | 0.157 → 0.081 (ratio 0.514) | **A4** (frame 122) | 15 → 19 |
| H_RH1_RBneg / RBpos | 0.158 → 0.081 / 0.156 → 0.080 | **A4** (frame 122) | 15 → 20 / 14 → 19 |
| H_RH2_RBpos | 0.281 → 0.217 | — | 100 → 129 |
| DJ_R | 0.281 → 0.217 | **A4** (frame 122) | 55 → 71 |
| S_RB_pos_154_182 | 0.023 → 0.022 | **A1** flip at 187, **A4** frame 187 | 6 → 9 |
| W_RA_neg, W_RB_neg, W_RC_raise | no out-of-pin frame: X = F (`ACCEPTED_NO_TRANSITION_AVAILABLE`) | — | — |

**Other measures.**
- D2 (peak elbow compensation step), D3 (rotation-step excess) and A5 fall strongly on every synthetic hold, e.g. H_A D3
  37.5 → 2.2°/frame.
- On RH2/DJ_R they also fall: D2 0.32 → 0.14 m, D3 68.6 → 27.7°.
- Frozen T/TE replays are bit-identical to W117 (W117 assess decisions reproduced for every W117 request). On the new
  holds, TE shows the same pattern as W117: a shorter 4-frame easing.

**Real-clip specifics.**
- At the RH2 release, U itself has a bend flip (182). A constant rotation offset does not preserve the wrist offset there,
  so the first post-release step stays at 0.22 m.
- The RH2 entry transition coincides with the source's 0.485 m wrist jump at 153→154. That is a confounder, not an
  editing target.
- The boundary raw (absolute) wrist step peak in RH1's window is 0.328 m for U, F, T, TE and X alike. It is U's own
  motion.

**Named regression control H_RH2_RBpos.** F ACCEPT; frozen T and TE REJECT `SOLVER_FAILURE NON_CONVERGED` (label
preserved, bit-identical to W117); X ACCEPT. X does not use the W117 optimiser, so this is not a fix of T.

**Low-evidence windows.** RH1 (109–118), RH2 (154–181) and the S_*_109_118 / S_*_154_182 segments are reported as
labelled subpopulations, not GT.

## 7. Independent Blender pin and change-footprint accounting

**X decisions** (49 requests):
- 29 ACCEPT;
- 11 REJECT as infeasible, conflicting or invalid (H_Cx, N2, N4–N7, H_RH2_none, H_RH2_RBneg, W_PF_C, W_RA_pos, W_RB_pos);
- 5 REJECT for unauthorised output or footprint mismatch (NX1–NX4, RX1);
- 4 REJECT as hard-pin failures after Blender (NX5, NX6, RX5, RX6).

**Accepted X:**
- 0 pin frames over 1 mm; maximum 0.81 mm.
- Changed channels only `upperarm_l`/`lowerarm_l` (synthetic `upper`/`lower`).
- Collateral exactly 0 m vs U: right wrist, neck, head and pelvis (synthetic: neck, base).
- Root matrix equal to U.

**Per-bone counts.** For example: H_A 28/28 frames per arm channel; H_RH1_none 17/17; DJ_R 56/56.

**Classification of every X request:**
- 9 `VALID_ACCEPTED_IMPROVEMENT`: DJ_B, H_A, H_B, H_C, H_D (development split, not gated), H_E, H_END, H_S0 and
  `S_RB_pos_86_106`. The last rests on a contaminated F baseline (§1).
- Decide's comparison also lists `S_RB_pos_109_118` as a meaningful improvement. That is purely an artefact: X has no
  out-of-pin change there and equals F on its own pins; only the shared F state differs.
- 12 `ACCEPTABLE_WITH_QUALITY_TRADE_OFF`;
- 5 `ACCEPTED_NO_TRANSITION_AVAILABLE`;
- 3 `ACCEPTED_EQUALS_F`;
- 11 infeasible/conflicting/invalid;
- 5 unauthorised/footprint;
- 4 hard-pin failure.
- `FEASIBLE_PIN_SOLVER_FAILED`: 0 for X.

## 8. Timeline visuals, labels, reviewed frames and limitations

`results/render/` contains:
- `index.html`: a table of every X request with proposed state id, commit result and head, linking to its timeline;
- `timelines/timeline_<rid>.png`: 49 timelines;
- `steps/steps_<rid>.png`: wrist-compensation-step line charts;
- for the representative requests H_B, H_E, DJ_B, H_RH1_none, H_RH2_RBpos and DJ_R: bone-proxy filmstrips
  (`filmstrip_*.png`) and GIFs (`anim_*.gif`), U / F / TE / X side by side.

**What a timeline shows:**
- separate rows for the requested hard pin, the hatched search budget, and the actual changed frames of F/T/TE/X;
- per-channel rows;
- wrist-displacement bars: sub-resolution changes are drawn grey, never hidden;
- the proposed state id, the commit result and the head after commit. REJECTed X rows are hatched and labelled
  "candidate NOT applied".

Colours are the documented reference categorical slots (blue pin, orange automatic edit, neutral grey unchanged). The
palette validator could not be run because Node is absent on this host.

**Visual review** (`review/visual_review.json`, `b5bbb15a…`): **agent-only, not independent**. It was written after the
assess numbers were printed and before decide.
- **Viewed:**
  - timelines H_E, H_RH1_none, DJ_B;
  - steps H_E, H_RH1_none, H_RH2_RBpos;
  - filmstrips H_E, H_RH1_none, H_B.
- **Not viewed:** all GIFs, the other filmstrips and timelines, and `index.html` in a browser.
- **Synthetic (H_B, H_E):** the release snap of F becomes a visible 3–5-frame follow-through in X. X carries an elbow bend
  on transition frames that U does not have.
- **Real (H_RH1_none):** the reduced release step is visible only in the numeric chart. At proxy scale the source's own
  pose change dominates the filmstrip.
- **Readability defect:** the footprint band sits *above* each filmstrip tile, so the band under a row belongs to the
  next row. Proxies are small at this camera scale.

## 9. Animation-quality benefit vs measured edit cost

- **Benefit:** on the synthetic holds, X cuts the boundary compensation step by 3–6× and removes F's release snap
  (visible). Within-pin articulation is unchanged, because X = F on pin frames; H_B's within-pin bend change remains.
- **Cost:**
  - more frames edited: 8 per saturated side (TE: 4);
  - larger mean elbow deviation from U, e.g. H_A +37 mm, H_E +31 mm, RH2 +29 mm;
  - on the real near-straight left arm, transition offsets push one frame (122 after RH1; 187 after S_RB_pos_154_182)
    below 5° flex, and in one case introduce a bend flip.
- Whether this trade-off is acceptable is a user-policy question. No threshold is proposed.

## 10. Hard failures, negative controls and fallback accounting

**Declared negatives (all returned their declared decision and class):**
- H_Cx: INFEASIBLE.
- N2: INSUFFICIENT_DOF → INFEASIBLE.
- N3: aim_distal ACCEPT, with offsets on `lower` only.
- N4, N5: UNSUPPORTED_BINDING.
- N6: CONFLICTING_HARD_CONSTRAINTS.
- N7: TARGET_PROVENANCE_MISMATCH.
- NX1, RX1: change outside the automatic budget → CONTRACT_VIOLATION.
- NX2 (planned domain under-declared): FOOTPRINT_EXCEEDS_DECLARED_DOMAIN.
- NX3: user-edit channel changed → UNAUTHORIZED_CHANNEL_CHANGE.
- NX4: `__object__` changed → ROOT_MOVED.
- NX5, RX5: false SOLVED → EVALUATED_PIN_UNSATISFIED.
- NX6, RX6: one corrupted pin frame → REJECT with exactly one failing frame.

**Display controls.** Each failed the truth check and was REJECTed `DISPLAY_FOOTPRINT_MISMATCH`:
- DC1 / DC5: a changed frame relabelled unchanged (synthetic / real);
- DC2: disjoint runs collapsed to the envelope;
- DC3: channel row dropped;
- DC4: timeline drawn from the solver's intended margin;
- DC6: a REJECTed candidate labelled ACCEPTED (`STATUS_LABEL_DISAGREES_WITH_DECISION`).

**Instrument control.** The known-footprint control KF is recovered exactly: frames [3, 17], channels upper {3, 17} /
lower {17}, runs [[3,3],[17,17]]. U vs U is empty.

**Accounting.**
- Fallback usage: none exists.
- New hard failures (C6): none. X ACCEPTs every contract-valid, all-oracle-feasible evaluation request.
- Infeasible frames passing the pin (any variant): 0.

## 11. Source preservation, lineage and exact reversal

- **Synthetic programme:**
  - S1: H_A X ACCEPTed on `syn/main`.
  - S2: NX6 REJECTed on `syn/atomic`; head unchanged.
  - S3: H_B X prepared against the old head → STALE_PARENT.
  - S4: revert. Values are byte-equal, and the Blender evaluation is bit-equal (digest and all frames) to P__A.
- **Real programme:**
  - R1: H_RH1_none X ACCEPTed (v0 → `458883b8…`).
  - R2: RX6 REJECTed; head unchanged.
  - R3: H_RH1_RBneg X prepared against v0 → STALE_PARENT.
  - R4: H_RH2_none X REJECTed.
  - R5: revert. Values are byte-equal, and the Blender evaluation is bit-equal to v0.
- **Per request:** each of the 49 X candidates was committed on its own ref. ACCEPT moved the head to the previewed
  candidate id; REJECT left the parent id. Every lineage chain re-verifies.
- **Source preservation:**
  - The declared inputs, the W116/W117 declarations, decisions, manifests, build/eval/assess files and the parent files
    are byte-unchanged across every stage.
  - Both manifests re-verify after the run.
  - Builds r1 and r2 are byte-identical; evaluations e1 and e2 are identical.

## 12. Timing, identity, GT-free and quality non-claims

- The authentic source is GT-free, and the video subject identity is UNKNOWN on 199/199 frames. All statements concern
  the known output rig only. No performer fidelity or identity continuity is claimed.
- **Timebase:** source 50 fps; historical Action keyed at frame_index + 1 and played at 30 fps; evaluation is
  frame-indexed. CONSTANT keys and sub-frame equality were re-verified (V6). No timebase repair or correctness claim.
- The 1 mm criterion remains the provisional W116 diagnostic.
- No claim of anatomical plausibility, Production readiness, a usable UI, or that a clear timeline proves animation
  quality.

## 13. Alternative explanations and unresolved uncertainties

- **D comes from my own gate scope (§1).** A per-segment F baseline would have avoided it. As frozen, the segment rows
  cannot be counted as clean A/B evidence in either direction.
- **The objective defines "better".** B1 measures wrist compensation steps. The offset layer works in rotation space,
  and where U moves fast (real RH2 release, U flip at 182) a constant rotation offset does not carry the wrist offset
  smoothly. A wrist-space or swivel-aware blend could rank cases differently. Only one strategy was tested.
- **The 0.5× rule is sharp.** RH1 sits at 0.514. That is a declared outcome, not a near-pass.
- **A4 on the real arm** reflects a left arm that is near-straight in much of the clip (W116 reach slack median 4 mm). Any
  rotation offset near a straight pose risks crossing 5°.
- **Saturation.** Most sides used the full 8-frame budget, so the result is close to a fixed 8-frame easing. The
  automatic rule mattered mainly at small offsets (H_S0 3, S_PF_C 1, RH2 entry 2) and at clip/gap limits.
- **Unresolved (not demonstrated):**
  - an independent visual review;
  - a playback-scale judgement on the real rig;
  - FBX/engine survival;
  - multi-pin redundancy;
  - other constraint types;
  - any Production integration.

## 14. Files, tests, environment and replay

**Changed files** (all new):
- `scripts/w118_transition_edit_base.py`
- `scripts/w118_transition_edit.py`
- `tests/test_w118_transition_edit.py`
- this Worklog

W116/W117 scripts are imported unchanged.

**Revisions:**
- start `0dbcbbf`;
- `42517e9`, the pre-outcome instrument and the final source revision (no code changed after it);
- the Worklog commit;
- a final marker commit.

**Completed and not completed.**
- **Completed:** Required Evidence parts 1–4, the synthetic and real replay populations, every declared stage, and the
  agent visual review of 9 images.
- **Not completed:** independent review; inspection of the GIFs and remaining images; a clean per-segment F baseline.

**Pre-declaration work, disclosed:**
- unit tests: two fixture interval fixes and one test-helper call fix;
- a design probe of hold intervals from parent/unpinned geometry;
- one crash-only smoke run of every stage in a throwaway directory. Stdout was discarded unread, only exit codes were
  read, and the directory was deleted;
- a re-declaration, before any build, solely to add that disclosure.

**Tests (AGENT-RUN):**
- `tests/test_w118_transition_edit.py`: **15 passed** (`w118_tests.xml`). They cover:
  - pin domain vs budget separation, clipping and gap split;
  - disjoint runs;
  - pin frames untouched, authorised-only and deterministic;
  - the zero-budget control, the no-offset case and `q_pow`;
  - actual footprint frames, channels and runs;
  - display truth check vs missing frame, collapsed runs, missing channel, intended-margin display, state id,
    rejected-as-applied and magnitude;
  - sub-resolution disclosure;
  - out-of-budget, out-of-planned, user-edit channel, display-mismatch, false-SOLVED and unreachable rejections;
  - single corrupted pin frame;
  - commit, stale parent and revert;
  - the compare rule, and integer frame axis metrics.
- **Regressions: 113 passed** (`w118_regression.xml`): `test_w117_temporal_pin`, `test_w116_semantic_pin`,
  `test_w115_validity`, `test_w114_source_preserving_base`, `test_w105_rig_utility`, `test_fk_solver`,
  `test_canonical_pose_parity`, `test_retarget_solver`.
- **Failed:** none.
- **NOT RUN:** the full suite, FBX export/import and engines. Blender was exercised only through the W116 stages.

**Reproduction** (PowerShell on Main, repo root; every stage refuses an existing output directory):

```
$py="C:\Projects\AnimCV\.venv_build_full\Scripts\python.exe"; $o="C:/Users/dna10/animcv-w118/results"
$s="5d61b552d2f4239801fac5e686cee5adfafdf4ac2c54f5697904af2ee4652d00"; $d="--declaration $o/declare/declaration.json --declaration-sha $s"
& $py -X utf8 scripts/w118_transition_edit.py preflight --out-dir $o/preflight
& $py -X utf8 scripts/w118_transition_edit.py calibrate --preflight $o/preflight --out-dir $o/calibrate
& $py -X utf8 scripts/w118_transition_edit.py declare --preflight $o/preflight --calibration $o/calibrate --out-dir $o/declare
& $py -X utf8 scripts/w118_transition_edit.py build $d.Split() --tag r1 --out-dir $o/build_r1
& $py -X utf8 scripts/w118_transition_edit.py build $d.Split() --tag r2 --out-dir $o/build_r2
& $py -X utf8 scripts/w118_transition_edit.py evaluate --build-dir $o/build_r1 --preflight $o/preflight --out-dir $o/eval_e1
& $py -X utf8 scripts/w118_transition_edit.py evaluate --build-dir $o/build_r1 --preflight $o/preflight --out-dir $o/eval_e2
& $py -X utf8 scripts/w118_transition_edit.py assess $d.Split() --build-dir $o/build_r1 --eval-dir $o/eval_e1 --out-dir $o/assess
& $py -X utf8 scripts/w118_transition_edit.py render $d.Split() --build-dir $o/build_r1 --eval-dir $o/eval_e1 --assess-dir $o/assess --out-dir $o/render
# visual review: results/review/visual_review.json (agent-authored)
& $py -X utf8 scripts/w118_transition_edit.py decide $d.Split() --build-dir $o/build_r1 --build-dir2 $o/build_r2 --eval-dir $o/eval_e1 --eval-dir2 $o/eval_e2 --assess-dir $o/assess --visual-review $o/review/visual_review.json --out-dir $o/decision
& $py -X utf8 scripts/w118_transition_edit.py manifest --base C:/Users/dna10/animcv-w118 --declaration-sha $s
& $py -X utf8 -m pytest tests/test_w118_transition_edit.py -q
```

The declaration embeds this host's absolute paths. The stages read the W116/W117 artefacts under
`C:/Users/dna10/animcv-w116` and `C:/Users/dna10/animcv-w117`. Blender is
`C:/Program Files/Blender Foundation/Blender 5.1`.

**Key artefacts (SHA-256 prefixes):**

| Artefact | SHA-256 |
|---|---|
| `declare/declaration.json` | `5d61b552…` |
| `decision/decision.json` | `91200bd4…` |
| `preflight/preflight.json` | `a8fad8f6…` |
| `calibrate/calibration.json` | `1e3a8c41…` |
| `build_r1/real_states.json` (= r2) | `fd9d16f4…` |
| `build_r1/syn_states.json` (= r2) | `7a572456…` |
| `eval_e1/real_eval.json` (= e2) | `0b187623…` |
| `eval_e1/syn_eval.json` (= e2) | `d5cd1a50…` |
| `assess/assess_report.json` | `93efc63c…` |
| `review/visual_review.json` | `b5bbb15a…` |
| `render/index.html` | `b3f61287…` |

`evidence_manifest.json` = `90ac1b7fcb5a744d01ed5bd4d40ad181b894095cb4d73be9798d48bcfbf3052f` (1,062 files, excluding the
content-addressed stores), written before this Worklog.

**Evidence categories:**
- **Code evidence:** the three new files.
- **Agent-run outputs:** everything under `results/`.
- **Visual judgement:** the agent-only review.
- **Unverified by the Reviewer:** all Main-local artefacts (W116/W117/W118 results), which only this host re-hashed.

**Resources and environment:**
- Each build took ≈ 9.2 s (pure Python, including the W117 replay); each Blender evaluation ≈ 8 s; assess ≈ 9 s;
  render ≈ 26 s. Tests took 0.3 s; regressions 8.7 s.
- Storage: 250 MB under `C:\Users\dna10\animcv-w118\`. It is local only and not published. The source clip was not
  copied.
- Main only. No GPU inference, network, model download, package change or LabServer63/69 use. No evaluation GT exists or
  was read.
- Single branch `arch/single_frame_first`. No branch, merge, rebase, reset or force operation.

AUTOMATION_BATCH_STATUS: COMPLETE
BATCH_NUMBER: 118
