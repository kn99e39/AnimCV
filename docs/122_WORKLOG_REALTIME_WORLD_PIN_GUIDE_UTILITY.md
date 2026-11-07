# Worklog — Real-Time Animation Utility of Frozen World-Pin Guidance on Held-Out Within-Source Windows (2026-11-07)

Batch 122. An evidence-only scratch experiment on `arch/single_frame_first`, run unattended on Main: local RTX 5080, Python
`C:\Projects\AnimCV\.venv_build_full`, Blender 5.1.0. `docs/next_batch.md` was read only.

Every number, render, test and manifest below is **AGENT-RUN** on Main. The Reviewer has not re-executed any of it. Every
visual judgement is the agent's own **AGENT REVIEW**. It is not animator, user or professional validation.

**Source independence.** The three sources (S1–S3) are the W121 sources. They are independent of the W102/W105 clip. They
are **not** new independent sources relative to W121. W122 adds new held-out windows *within* those three videos only.

## 1. Official CASE and scope

**Verdict: CASE B — PARTIAL / CONTEXT-SPECIFIC.** The frozen rule produced it in `results/decision/decision.json`
(SHA-256 `b6be8e1dc4a0cd8c…`).

**Gates.**

- All 15 validity gates V1–V15 pass.
- All 6 safety gates C1–C6 pass.
- W121's two instrument defects did not recur. The repaired V10 and V12 are non-vacuous; see §4.

**Why B and not A.** A requires at least two utility-positive held-out windows, from at least two identity-VERIFIED sources.
W122 found only **one** utility-positive window: **S1 NH** (S1 frames 241–249, the rack-to-overhead press). Two of its
three edit variants (`none`, `RBneg`) meet every condition:

- a meaningful primary GUIDE improvement under the frozen comparator;
- a blind agent rating that unblinds to G1 better;
- no G0-better rating on the arm dimensions;
- no raw-motion harm.

**Why B and not C.** The held-out population also contains numeric-only guide improvements: all six S2 held-out requests
improve numerically, but every S2 pair was rated a visual tie.

**Scope.**

- 3 sources, 5 new held-out windows, 15 held-out requests (all ACCEPTed).
- 27 W121 anchor-replication requests, kept separate.
- 4 frozen W120 synthetic controls, kept separate.
- Within-source only. This is not new-source generalisation.

**Historical verdicts are unchanged:** W113 D, W114 D, W115 D, W116 B, W117 C, W118 D, W119 D, W120 B, **W121 D**.

- W121's decision re-hashes to `90adb6cb…` with case D.
- W120's decision re-hashes to `1029ce92…` with case B.
- The W121 evidence manifest `ff914c33…` re-verifies across all 5,086 files, with no extra files, both at preflight and after
  the run.
- W121's post-hoc "would be B" was not used anywhere.

## 2. Starting state and revisions

| Item | Value |
|---|---|
| Instruction blob | `eebfac50f0419d4c015efbc3bf808759bf020092`, verified first |
| Starting commit | `75e3cf23e6cf582eb409ad2e23433aba2038f243`. It equalled HEAD and `origin/arch/single_frame_first` after fetch. |
| Pre-outcome instrument commit | `67d7e58`: scripts and tests, committed after the declaration and before any factor variant was built |
| Final source revision | the Worklog commit after `67d7e58`. It adds `scripts/w122_review_overlays.py` (see §11) and this Worklog. |

The COMPLETE marker is a separate, final commit.

## 3. Sources, licences, hashes, frozen reproduction and subject identity (Part 1)

**Originals** (`results/preflight/preflight.json`, `ba47aae7…`). These are the W121 Commons originals, stored outside the
repository under `animcv-w121/data/originals`. They were not redistributed and not re-downloaded.

| ID | File | Licence / author | SHA-256 | Decoded frames | Cadence |
|---|---|---|---|---|---|
| S1 | Double Kettlebell Strict Press (side view).webm | CC BY-SA 4.0 / Taco fleur | `de8feb9d515daa2b…` | 1631 | 30.000 fps, max deviation 0.67 ms |
| S2 | Firejuggling 2009 ubt.ogv | CC BY 3.0 / Tomasz Sienicki | `68301b0c35c98448…` | 895 | 30.000 fps, 0.0003 ms |
| S3 | Bent-over row - exercise demonstration video.webm | CC BY 3.0 / FitnessScape | `f6b90f2f72e9020d…` | 154 | 30.000 fps, 0.67 ms |

For each source:

- the full SHA-256 equals W121's acquire record;
- the SHA-1 equals the Commons SHA-1;
- every decoded-frame BGR digest equals W121's `inspect.json` list;
- it is not the W102 clip.

**Frozen identity.** The S0 checkpoint (`6c12fea9…`), RTMDet/RTMPose weights (`78e30dcc…`, `77ffc7e8…`), gate report, lock
and diagnostic script all equal W121's frozen identity.

**Independent re-execution** (new in W122, not a stored-hash check). I ran the unchanged W102 M1/S0 diagnostic again on S1, S2,
S3 and the historical clip. The launcher is a copy of W121's with only the scratch base changed; its hash is in the preflight.

- **Every output array is byte-equal to the stored arrays.**
- The historical clip gives `6784357f…` (= W102).
- S1, S2 and S3 give `7988bf4d…`, `1528041d…` and `2a5f3ae5…` (= W121).

**Rig adapter and parents.**

- The W105 FIXED adapter reproduces W105 on 199/199 frames.
- The re-executed predictions reproduce W121's `video_cases` and parent values on every frame.
- The parent state ids verify: `cc4bf006…`, `23bb276e…`, `b0370139…`.
- A fresh W122 Blender re-probe reproduces W121's value digests (`25a09802…`, `88ddaabe…`, `b9058745…`) and is bit-equal on
  every frame.
- FK error ≤ 1.3e-6 m.
- Bindings are equal to W121's; left_wrist is BOUND.
- Rig identity is unchanged.

**S2 and subject identity** (`results/review/identity_review.json`, `f5ca1fce…`). This was recorded **before any factor
variant existed**.

- For every window (new and anchor), I inspected every source frame of [domain − 10, domain + 10] with the frozen tracked 2D
  skeleton overlaid (`results/identity/*.jpg`).
- **S2:** the tracked skeleton stays on the same performer in every frame of every interval: the younger juggler, dark shirt,
  left of frame. It never moves to the second juggler or the crowd.
- **S1 and S3:** single-performer clips.
- All 15 windows are labelled **VERIFIED** for *person continuity only*.
- **Not verified:**
  - left/right arm assignment and 2D arm-landmark accuracy. S1's left arm is the far, self-occluded arm in a side view. S2's
    arm crosses juggling clubs. S3's arm is partly torso-occluded.
  - any 3D or performer fidelity.
- The parent states keep `video_subject_identity = UNKNOWN`. Identity is labelled at window level in the review file only.

## 4. Prospective instrument and repaired V10/V12 (Part 2)

**Declaration** `results/declare/declaration.json`, SHA-256 **`9f406064a1b52ffbb847fba5429878c4a6cc75e1c5194be5b3fdace8d3a74c04`**,
records `outcomes_inspected: false`. It binds:

- every input hash: preflight, windows, selection, identity evidence and review, calibration, frozen controls, instrument;
- the code blobs of 18 files;
- the request population;
- the negative and display controls;
- the 15 V and 6 C gates;
- the CASE rule;
- the metrics;
- the blind seed `W122-blind-2026-11-07` and the pair lists;
- the review procedure.

Before the declaration, only U, F_r and X_r baselines existed. No factor variant had been built and nothing had been rendered.

**V10 repair** (`scripts/w122_realtime_utility_base.py`):

- The build records are compared after removing **only** the declared path `*.factors.*.info.wall_s`.
- `validate_exclusions` refuses any other exclusion. Its leaf may not be a protected semantic key: statuses, planned/budget
  frames, margins, ids, flags, frames, fairness, quality and similar.
- The comparison must be non-vacuous: compared leaves > 0, with equal counts.

**V10 pre-outcome fixture** (historical W121 r1/r2, read-only):

- The raw records differ, as in W121.
- The repaired comparison is equal: 46,705, 27,147 and 22,903 leaves compared; 65, 48 and 48 `wall_s` removed.
- A semantic corruption (one factor `request_status`) is detected.
- `*`, `*.factors.*.info`, `*.x_info.planned_frames` and `*.factors.*.info.request_status` are all refused.

**V10 in the experiment:**

- states files byte-identical;
- records equal under the declared exclusion: 65,623 / 47,329 / 32,419 leaves compared, 101 / 84 / 72 `wall_s` removed;
- all Blender evaluations byte-identical, including the synthetic one;
- two independent assessments (r1/e1 and r2/e2) equal on 675 semantic items: every decision, residual, footprint, display,
  quality, contrast and attribution output;
- the **live** negative control, one semantic field corrupted in a copy of build 2, is detected on every source.

**V12/DG5 repair:**

- `dg5_inject` refuses to run unless the committed decision is REJECT and the displayed status is REJECTED. It also requires
  the corruption to change the displayed status.
- Pre-outcome fixture on W121 artefacts: S1_REACH and S1_HIGH_RBpos (G1F0, genuine REJECT) fail the truth check with
  `STATUS_LABEL_DISAGREES_WITH_DECISION`. S1_MID_RBpos (ACCEPT) is **refused** by the precondition. That was the W121 defect.
- In W122, DG5 was predeclared on the controlled-infeasible held-out request **S1_NREACH** G1F0.
  - The precondition was verified: REJECT, then REJECTED.
  - The status changed REJECTED → ACCEPTED.
  - The truth check fails, as required.
  - DG5_SYN on the W120 synthetic genuine REJECT `NG_INFEASIBLE` also fails the truth check.

**Other instrument preflight.** The W120 fixture preflight is all_ok: 262,144-case precedence, 0 mismatches. The 19 declared
negative expectations equal the frozen predictor.

## 5. Held-out window candidates, selection and coverage (Part 3)

**Rule** (`AB.WINDOW_RULE`, declared). A candidate is a 9-frame pin window.

- **Domain:** its own ±8-frame automatic domain must be inside the W121 segment, with the W121 edge rule.
- **Exclusions:**
  - any overlap with a W121 window extended by ±8;
  - any overlap with a frozen 2D track-switch proxy frame.
- **D** = the unedited Blender wrist travel from the capture frame (W121 definition).
- **Feasible** = the frozen W116 framewise resolver SOLVED/NO_CHANGE_NEEDED on all 9 pin frames for **all three** edits.
- **Selection:**
  - LOWER = the feasible candidate with minimum D.
  - HIGHER = the feasible candidate with maximum D among those with D > the source's median and separated from LOWER.
  - HIGHER is not forced.

Nothing but the unedited parent and F statuses was used.

| Source | Candidates | Overlap W121 ±8 | F-infeasible | Feasible | Median D | Held-out windows (pin frames, D) |
|---|---|---|---|---|---|---|
| S1 | 214 | 135 | 13 | 66 (24 above median) | 135 mm | **NL** 184–192 (37 mm); **NH** 241–249 (518 mm) |
| S2 | 214 | 104 | 28 | 82 (20 above median) | 288 mm | **NL** 330–338 (131 mm); **NH** 474–482 (453 mm) |
| S3 | 128 | 106 | 0 | 22 (11 above median) | 178 mm | **NL** 86–94 (28 mm); **NH not available** |

- **S3 NH.** All 22 non-overlapping candidates lie in s ∈ [66, 87]. None above the median can be separated from NL, so the
  reason is `ABOVE_MEDIAN_FEASIBLE_CANDIDATES_OVERLAP_LOWER`.
- **F-infeasible reasons:** UNREACHABLE for one or more edits. S1: RBneg 13, none 12, RBpos 11. S2: RBpos 28, none 18,
  RBneg 10.
- **Held-out population:** 5 windows × 3 edits = **15 requests, all ACCEPTed and factorial-eligible**.
- **Controls per source** (all predeclared on NL):
  - a zero-budget control `S*_NZB`: all three ACCEPTed, with budget equal to the pin frames for all four factors;
  - a controlled-infeasible `S*_NREACH` (+1 m): all REJECT INFEASIBLE_REQUEST;
  - `S*_NDOF`, `S*_NCONFLICT` (INFEASIBLE) and `S*_NWRONGBIND` (UNSUPPORTED_BINDING): all rejected as declared.

**Pre-outcome stratification.**

- The rule was: U raw wrist step ≥ 50 mm per frame anywhere in the automatic domain.
- **Every** held-out and anchor window is SOURCE_DOMINATED. The U domain peak is 105–400 mm per frame for the new windows and
  121–877 mm for the anchors.
- The declared partition therefore does not stratify this population. It is a negative finding about the sources, not a
  threshold.

## 6. W121 anchor replication (separate population, never held-out)

The 27 W121 evaluation requests and the W121 controls were rebuilt with W121's request ids.

- **Reproduction:** every U, F_r, X_r and factor state id of all 14 W121 requests per source equals W121's build. Their B1
  values are exactly W121's; for example S1_MID_RBneg is 51.5 / 53.4 / 34.0 / 34.0 mm.
- **Decisions:** 17 ACCEPT and 10 REJECT, as in W121. The rejections are all HIGH plus S3_LOW_RBpos.
- **Known numeric positives** (S1_MID_RBneg, S2_LOW ×3): blind-rated **TIE** on all four.
- **S3 anchors** (5 ACCEPTed): all **TIE**.
- None of these counts as held-out evidence.

## 7. Synthetic controls (separate)

The frozen W120 states for H_G (held-out), H_C (near-singular), DJ_B (disjoint runs) and NG_INFEASIBLE were copied unchanged;
the state ids equal W120's. W122 re-evaluated them in Blender.

- The evaluation is bit-equal to W120's.
- The decisions equal W120's.
- H_G, H_C and DJ_B: every variant is ACCEPTed. The pin residual is ≤ 9e-8 m. No guide or floor contrast is meaningful.
  B1 in mm (G0F0 / G1F0): H_G 52.8 / 52.1, H_C 31.7 / 31.3, DJ_B 19.5 / 19.4.
- DG3_SYN (collapsed disjoint runs on DJ_B G1F0) fails the truth check with `CELL_MISMATCH` and `RUNS_MISMATCH`.
- DG5_SYN behaves as in §4.

**Not done:** the optional jitter perturbation probe. It was not declared and was not run.

## 8. Matched factors, zero budget, negatives and pin safety (Part 4)

**Matching.**

- **V6 isolation:** every contract-valid request passes F_r isolation at both value and Blender level.
- **V7 matching:** `ab_fairness` is ok for every request and factor, the solver digest is equal, and `cross_factor_identity`
  holds. All 6 zero-budget controls have budget = pin frames.

**Negative controls on the neg base S1_NL_RBpos** (the W121 suite, re-targeted). All 19 returned the declared class and
reasons:

- CONTRACT_VIOLATION × 13;
- NG_NONCONV → SOLVER_FAILURE;
- NG_REACH, NG_DOF, NG_CONFLICT → INFEASIBLE_REQUEST;
- NG_WRONG_BINDING → UNSUPPORTED_BINDING;
- NG_SHARED also fails isolation at value and Blender level.

**Display controls.** DG1, DG2, DG4 (on S1_NL_RBpos) and DG5 all fail the truth check.

**DG3 deviation (disclosed).** The declaration targets DG3 at the first real *held-out* variant with ≥ 2 changed runs, with
the synthetic DJ_B as fallback.

- My selection code also searched the non-held-out records. It executed DG3 on `NG_FALSE` G1F1, a negative-control record,
  and the truth check failed with `RUNS_MISMATCH`.
- No held-out variant has ≥ 2 changed runs. The declared fallback DG3_SYN therefore applies, and it also fails the truth check.
- The gate result is unchanged either way.

**Source-level controls**, all detected:

- renamed copy of the old clip;
- re-encoded, shifted and re-timed copy;
- altered byte;
- stale observation binding;
- stale model binding;
- wrong FPS;
- +0.1 s offset;
- unverified continuity claim.

Inexact rollback is detected on all three sources. Scratch files went to `animcv-w122/tmp`.

**Hard pins (C1).** Every ACCEPTed X_r and factor variant was checked over 1,665 pin-frame Blender evaluations. The maximum
residual is **9.3e-7 m**; none is over 1 mm. Changed channels are only upperarm_l and lowerarm_l, inside the budget and the
planned domain. There is no outside-arm point change and no root change.

**C2, C5, C6.**

- Display truth, proposed and result, passes for every variant.
- The planned domain is equal across X and the four factors.
- Preview id = committed id.
- spine_01 and `__object__` are bit-equal to U.
- There is no fallback and no infeasible acceptance.

## 9. Raw versus source-relative motion and attribution (Part 5)

Measured on Blender frames in metres, per 30 fps frame. B1 = the frozen boundary compensation peak.

| Held-out request | B1 G0F0 → G1F0 (mm) | guide@F0 meaningful | Raw wrist peak in domain U / G0F0 / G1F0 (mm) | Visual (unblinded) |
|---|---|---|---|---|
| S1_NL ×3 | 9–14 → 9–12 | no | 69–105 / 69–91 / 69–96 | TIE ×3 |
| **S1_NH_none** | 337 → 86 | **yes** | 400 / 301 / 373 | **G1 better** |
| **S1_NH_RBneg** | 294 → 79 | **yes** | 398 / 307 / 364 | **G1 better** |
| S1_NH_RBpos | 380 → 93 | no (comparator: raw boundary worse); conditional guide@F1 yes | 402 / 300 / 382 → **raw harm** (+82 mm > 75 mm) | G1 better |
| S2_NL ×3 | 30–33 → 15–19 | yes ×3 | 305–306 / 301–303 / 299–302 | TIE ×3 |
| S2_NH ×3 | 104–109 → 70–78 | yes ×3 | 226–370 / 150–368 / 146–370 | TIE ×3 |
| S3_NL ×3 | 9–10 → 10 | no | 62–88 / 62–89 / 62–88 | TIE ×3 |

**FLOOR.** No meaningful primary or conditional floor improvement on any held-out request.

**What S1 NH shows.**

- The source wrist jumps **400 mm in one frame at f251** (≈ 12 m/s), the first release frame after the press starts.
- G0F0 = X_r. At f251 it moves the wrist 238 mm in a direction different from the source, with a 337 mm compensation step.
  The release comp series is 58, **337**, 68, 88, 85, 124, 107 mm.
- G1F0 follows the source jump: a 356 mm raw step at f251 with a 48 mm compensation step. Its comp series is 18, **48**, 69,
  82, 86, 82, 69 mm.
- So G1 *raises* the raw domain peak (373 vs 301 mm) but removes the edit-induced reversal.
- The blind review saw exactly that reversal: a one-frame sideways forearm flick at f251 and an out-and-back wrist excursion
  in the G0 variant. G1 unfolds progressively.
- This is a case where the source-relative improvement and the agent-visible improvement agree, while the raw peak gets worse.
- Under the declared raw-harm rule (+25 % or +10 mm), `none` (+72.7 mm, < 75.1) and `RBneg` (+56.8 mm, < 76.7) pass;
  `RBpos` (+82.3 mm, > 75.0) fails.

**What S2 shows.**

- The guide halves B1 on all six requests.
- The raw steps are dominated by source steps of 60–305 mm per frame, for example 189 mm and 305 mm at f344–f345.
- G0 and G1 raw steps differ by ≤ 15 mm.
- The overlays show cm-scale forearm offsets on 4–8 release frames, with no visible continuity difference. The result is
  numeric-only.

**What S3 shows.** Low travel. Release steps are ≤ 37 mm for every variant, and there is no measurable or visible difference.

**Articulation.**

- No new bend flip and no near-straight frame on any held-out contrast. The comparator gate is clean.
- Minimum flex: on S1 NH, G1F0 50–54° versus G0F0 59°. On S2 and S3 it is equal.
- The peak arm-channel angular step is lower for G1 on S1 NH (55–64° vs 69–79°).

**Source-dominated spikes.** U steps ≥ 50 mm fall on 20–37 frames of each S1 NH and S2 render range, and 7 on S3 NL. These
spikes dominate absolute speed and acceleration. Example peak acceleration over the domain on S1_NH_none: U 580, G0F0 528,
G1F0 586 m/s².

## 10. Footprint, timeline and reversal (Part 5)

- The changed frame and channel footprint is a subset of the authorised domain for every ACCEPTed variant (C1).
- Display specs, proposed and result, pass the independent truth check (C2). REJECTed candidates display REJECTED; DG5 proves
  a false ACCEPTED label is caught.
- **Lineage per source:**
  - R1: ACCEPT `S*_NL_RBpos` G0F1 moves the head.
  - R2: atomic REJECT leaves the head unchanged. On S1 this is NG_PIN_COMBINED; on S2 and S3 it is NREACH.
  - R3: STALE_PARENT.
  - R4: revert is values byte-equal and Blender bit-equal to v0.
  - R5: inexact rollback is detected.
- The real-time renders carry an unblinded annotated timeline bar (pin, automatic domain, current frame) in
  `render/videos/*_annotated_UNBLINDED_30fps.mp4`.

## 11. Real-time visual method, blinding, ratings and evidence (Part 6)

**Renders** (`render/render_report.json`, `e01fd0d9…`).

- Workbench renders: 24,561 frames of the unchanged skinned rig.
- Range: [pin − 18, pin + 18] (budget plus 10 context frames), clipped to the segment.
- Views: world front (−Y), world side (+X) and a character-relative left side. The character-relative camera is fixed per
  request from the capture-frame shoulder axis. The world-axis views are labelled as such.
- Cameras are fixed per request. Ortho scale 1.4 m, 320 px.
- The green sphere marks the pin target on pin frames.
- There is no filtering, interpolation, motion blur or retiming.

**Videos.** All 82 videos decode to exactly their declared frame count at 30.000 fps, one video frame per source frame (V11):

- blind real-time MP4: source crop | U | A | B × 3 views;
- ×0.25 slowed diagnostic MP4, each frame repeated 4×, labelled;
- annotated unblinded MP4 with all variants;
- 10 REJECTed-request coverage MP4s (U | F_r).

**Blinding.** A/B was assigned per pair from the declared seed, and the mapping was sealed at render (`f53ebb59…`). The
ratings `review/blind_ratings.json` (`4611100f…`) were hashed in `seal/seal.json` (`930c59d7…`) before `unblind` read the
mapping. The mapping recomputes from the seed (V13). The 24 pairs are:

- 15 held-out;
- 4 known W121 positives;
- 5 S3 anchors.

The 4 S3 anchors that were not ACCEPTed (S3_LOW_RBpos and S3_HIGH ×3) are listed as ineligible.

**Review method — limitation and addition, disclosed.**

- The declared review sheets (every decoded frame, A | B at 150 px) were too small to show the A/B differences.
- Before the first rating, I wrote `scripts/w122_review_overlays.py`. It reads only the blind MP4s and the blind trail plots:
  - every decoded frame of the 30 fps blind MP4, in order;
  - an A/B overlay in world_front and char_side (red = A only, cyan = B only);
  - the A/B/U wrist and elbow trails.
- It never reads the mapping. It is an addition to the declared material, not a change to the rubric.
- I rated from these overlays, using the side-by-side sheets where needed (P07, P10).
- **I inspected decoded frames in sequence. I cannot perceive real-time playback as a human viewer does.**
- The pair id reveals the request, though not the A/B mapping. I designed the instrument, so blinding is partial.

**Ratings, before unblinding.** 21 TIE, 3 non-tie: P10 B, P18 A and P19 B.

**Unblinded.** All three non-ties are **S1_NH (RBneg, none, RBpos) = G1 better**, on the dimensions raw_motion_continuity
and temporal_transition. Every other pair was a tie:

- S1 NL ×3, S2 NL ×3, S2 NH ×3 and S3 NL ×3;
- the known S1_MID_RBneg and S2_LOW ×3 positives;
- the S3 anchors.

**Mixed S3 MID anchor ratings.** The S3 MID anchor pairs were rated TIE with mixed dimensions: one variant's wrist stayed
closer to U, the other's elbow path was straighter.

**Hard-pin failure coverage.** All 10 REJECTed evaluation requests were rendered. I inspected one per source:

- S1_HIGH_none: F_r's arm is stretched straight toward a far target.
- S2_HIGH_none: the arm is at its reach limit.
- S3_LOW_RBpos: no visible difference from U at this scale.

The other 7 were rendered but not inspected individually.

**Published evidence.** Videos, renders and source crops stay local under `C:\Users\dna10\animcv-w122` (3.3 GB). No video or
derived source frames are published in the repository. The paths and hashes are in the evidence manifest.

## 12. Visible versus numerical-only

| | Numeric GUIDE gain (held-out) | Visible G1 gain (blind) | Utility-positive |
|---|---|---|---|
| S1 NL | none | none | 0 |
| S1 NH | 2 / 3 (primary), 3 / 3 (conditional) | 3 / 3 | **2 / 3** (RBpos fails raw harm and primary comparator) |
| S2 NL | 3 / 3 | 0 / 3 | 0 |
| S2 NH | 3 / 3 | 0 / 3 | 0 |
| S3 NL | 0 / 3 | 0 / 3 | 0 |

- **Visible effect.** It is confined to one window: a ~0.3 m edit-induced reversal at the release of a 0.4 m / frame source
  jump. Elsewhere, B1 reductions of 14–35 mm were not visible at 30 fps against source motion of 60–400 mm per frame.
- **Anchors.** The W121 known positives also had no visible effect. They do not replicate visually.

## 13. Failures, rejections and alternative explanations

**Rejections and infeasibility.**

- Held-out: no rejection.
- Anchors: 10 REJECTed (all HIGH, plus S3_LOW_RBpos).
- Candidate stage: 41 candidate windows were F-infeasible (S1 13, S2 28). S3 has no feasible higher-travel window.
- All of these stay in the accounting.

**Alternative explanations for the S1 NH effect.**

1. **Source artefact, not a press.** S1's left arm is the far, self-occluded arm. The 400 mm / frame "release" at f251 is a
   frozen-perception jump, not physically plausible arm motion. In the coverage renders, the rig's left arm visibly
   disagrees with the source rack pose. G1's benefit is therefore "follow the source jump instead of fighting it", which only
   helps if the jump is wanted.
2. **Assumed intent.** The G0 reversal is what the frozen X_r offset layer does when the release target moves 0.4 m in one
   frame. A user might prefer neither.
3. **Small, correlated sample.** One window; three edit variants of one release; one agent rater who built the instrument.

**Stratification.** Every window is source-dominated, so the declared stratification could not separate source-dominated
from edit-comparable episodes.

**DG3.** Executed on a non-declared record; see §8. The gate is unaffected.

## 14. Files, commands, tests, environment, artefacts

**Changed files** (all new):

- `scripts/w122_realtime_utility_base.py`
- `scripts/w122_realtime_utility.py`
- `scripts/w122_review_overlays.py` (review aid, written after render and before rating)
- `tests/test_w122_realtime_utility.py`
- this Worklog

Nothing else was changed: no `src/`, no W102–W121 script, test, Worklog, artefact, rig, model, threshold or solver, and not
`docs/next_batch.md`.

**Scratch helpers outside the repository** (`animcv-w122/setup`):

| File | SHA-256 prefix | Purpose |
|---|---|---|
| `w122_frozen_launch.py` | `b6da1298` | |
| `run_reexec.sh` | `4805b189` | |
| `add_rating.py` | `ca4cef7d` | appends A/B ratings only |

The review overlay script in the repository is `fcd398f4`.

**Tests (AGENT-RUN):**

| Command | Result |
|---|---|
| `python -X utf8 -m pytest tests/test_w122_realtime_utility.py -q` | **21 passed** (`results/tests/w122_tests_final.xml`). The pre-declaration run gave 19 passed and 2 skipped, because the stored artefacts did not exist yet. |
| `pytest tests/test_w121_independent_transfer.py tests/test_w120_factor_attribution.py tests/test_w119_endpoint_transition.py tests/test_w118_transition_edit.py tests/test_w117_temporal_pin.py tests/test_w116_semantic_pin.py tests/test_w115_validity.py tests/test_w114_source_preserving_base.py tests/test_w105_rig_utility.py tests/test_fk_solver.py tests/test_canonical_pose_parity.py tests/test_retarget_solver.py -q` | **182 passed** (`results/tests/w122_regression.xml`) |

The W122 tests cover:

- window exclusion, feasibility and deterministic selection, including HIGHER not forced;
- the stratum threshold;
- V10: declared-only exclusion, semantic corruption, and refusal of vacuous or semantic exclusions;
- the DG5 precondition, and the stored genuine REJECT failing the truth check;
- 30 fps kinematics;
- the seeded blind mapping, rating schema and unblinding;
- the CASE mapping: A needs 2 windows from 2 sources, UNKNOWN identity cannot count, numeric-only and visual-only give B,
  harm, C, D and DATA_BLOCKED;
- the raw-harm rule;
- stored preflight, re-execution and identity;
- stored windows disjoint from W121 and reselecting identically;
- stored videos at 30 fps with one frame per source frame;
- separate denominators.

Isolation, zero budget, fairness, precedence, hard pin and rollback are also covered by the W121/W120 suites above.

**Failed:** none. **Not run:** the full suite, FBX, the optional synthetic jitter probe, and independent Reviewer re-execution.

**Key artefacts** (`C:/Users/dna10/animcv-w122/results`, SHA-256 prefix):

| Artefact | SHA-256 prefix |
|---|---|
| preflight | `ba47aae7` |
| windows | `e4c9d411` |
| selection | `d0dbfd0a` |
| identity_evidence | `3a8b93db` |
| identity_review | `f5ca1fce` |
| calibration | `0bdc6c67` |
| frozen_controls | `46ff3fd8` |
| instrument | `1792d470` |
| declaration | `9f406064` |
| build_r1 report | `0c74640f` |
| states S1 / S2 / S3 | `eea6cbbc` / `6cca2ad9` / `fa8f8097` |
| eval S1 / S2 / S3 (e1 = e2) | `696a0334` / `20d62a4e` / `f5d45472` |
| assess_a1 | `a7ebc49b` |
| semantic_outcomes (a1 = a2) | `9cb6c445` |
| render_report | `e01fd0d9` |
| sealed mapping | `f53ebb59` |
| blind_ratings | `4611100f` |
| seal | `930c59d7` |
| unblinded | `b801ace2` |
| decision | `b6be8e1d` |

`evidence_manifest.json` = `5eaef6618c7bf7484477aba5a17dfe50bbab5302885ebfa97da1dcd6115896da`. It covers 25,360 files,
excluding tmp and cache, and was written before this Worklog.

**Reproduction** (Git Bash, repository root at `67d7e58` plus the overlay script; every stage refuses an existing output
directory):

```
PY=C:/Projects/AnimCV/.venv_build_full/Scripts/python.exe; R=C:/Users/dna10/animcv-w122/results
bash C:/Users/dna10/animcv-w122/setup/run_reexec.sh            # frozen M1/S0 re-execution -> $R/reexec_perception
$PY -X utf8 scripts/w122_realtime_utility.py preflight --out-dir $R/preflight
$PY -X utf8 scripts/w122_realtime_utility.py windows --preflight $R/preflight --out-dir $R/windows
$PY -X utf8 scripts/w122_realtime_utility.py identity --windows $R/windows --out-dir $R/identity
# agent identity review -> $R/review/identity_review.json
$PY -X utf8 scripts/w122_realtime_utility.py calibrate --windows $R/windows --out-dir $R/calibrate
$PY -X utf8 scripts/w122_realtime_utility.py instrument --out-dir $R/instrument
$PY -X utf8 scripts/w122_realtime_utility.py declare --preflight $R/preflight --windows $R/windows --identity $R/identity --identity-review $R/review/identity_review.json --calibrate $R/calibrate --instrument $R/instrument --out-dir $R/declare
S=9f406064a1b52ffbb847fba5429878c4a6cc75e1c5194be5b3fdace8d3a74c04; D="--declaration $R/declare/declaration.json --declaration-sha $S"
$PY -X utf8 scripts/w122_realtime_utility.py build $D --tag r1 --out-dir $R/build_r1        # and r2
$PY -X utf8 scripts/w122_realtime_utility.py evaluate --build-dir $R/build_r1 --out-dir $R/eval_e1   # and r2 -> e2
$PY -X utf8 scripts/w122_realtime_utility.py assess $D --build-dir $R/build_r1 --eval-dir $R/eval_e1 --out-dir $R/assess_a1   # and r2/e2 -> a2
$PY -X utf8 scripts/w122_realtime_utility.py render $D --build-dir $R/build_r1 --eval-dir $R/eval_e1 --assess-dir $R/assess_a1 --out-dir $R/render
$PY -X utf8 scripts/w122_review_overlays.py $R/render
# agent blind review -> $R/review/blind_ratings.json
$PY -X utf8 scripts/w122_realtime_utility.py seal --render-dir $R/render --ratings $R/review/blind_ratings.json --out-dir $R/seal
$PY -X utf8 scripts/w122_realtime_utility.py unblind --render-dir $R/render --seal-dir $R/seal --ratings $R/review/blind_ratings.json --out-dir $R/unblind
$PY -X utf8 scripts/w122_realtime_utility.py decide $D --build-dir $R/build_r1 --build-dir2 $R/build_r2 --eval-dir $R/eval_e1 --eval-dir2 $R/eval_e2 --assess-dir $R/assess_a1 --assess-dir2 $R/assess_a2 --render-dir $R/render --seal-dir $R/seal --unblind-dir $R/unblind --out-dir $R/decision
$PY -X utf8 scripts/w122_realtime_utility.py manifest --declaration-sha $S
```

**Resources and environment.**

- **Timing:**
  - frozen perception re-execution: 1–6.7 min per clip, 4 in parallel on CPU;
  - preflight 14 s; windows 4 s; calibrate 2 s; instrument 3 s;
  - builds 10 s each; Blender evaluation 45 s each; assessments 27 s each;
  - render 6.3 min (Blender 306 s);
  - tests 0.2 s; regression 11 s.
- **Storage:** 3.3 GB local under `animcv-w122`, unpublished. Rendered frames are 572 MB for S1.
- **Software:** Python 3.11.9, numpy 1.26.4, OpenCV 4.11.0, torch 2.13.0+cu130, Blender 5.1.0, Windows 10.0.26200, RTX 5080.
- **Scope:** Main only. No LabServer63/69 connection, no network access, no training, no package change.
- **Git:** single branch, no merge, rebase, reset or force operation.

## 15. Explicit non-claims and unresolved uncertainties

**Not claimed:**

- animator or user validation (agent review only, agent-designed instrument, partially blind);
- independent Reviewer reproduction;
- source-to-performer or 3D GT fidelity, or arm-landmark correctness;
- new-source generalisation (within-source windows of the W121 sources only);
- Production quality, readiness or thresholds. The 1 mm pin, the 50 mm stratum, the 25 % / 10 mm harm rule and the W119
  comparator are experiment-only;
- Generator training;
- any change to W121's official CASE D.

**Completed:** Parts 1–6:

- authentication and frozen re-execution;
- the repaired instrument;
- prospective held-out windows;
- matched Blender evaluation;
- attribution;
- 30 fps real-time renders;
- blind agent review;
- the decision and manifest.

**Not completed:**

- individual inspection of 7 of the 10 failure-coverage renders;
- the optional synthetic jitter probe;
- real-time perceptual (human) playback review.

**Unresolved:**

- Whether the S1 NH visible benefit reflects useful animation behaviour or merely follows a frozen-perception jump of the
  occluded far arm.
- Whether cm-scale S2 differences would be visible to a human viewer at 30 fps.
- Whether a higher-travel feasible held-out window in S3, or any source with less source-dominated motion, would show a
  different picture. None was available.
- How much blinding was weakened by the pair-id → request linkage and by the reviewer being the instrument's author.


AUTOMATION_BATCH_STATUS: COMPLETE
BATCH_NUMBER: 122
