# Worklog — Independent Real-Video Transfer of World-Pin Transition Editing (2026-11-07)

Batch 121. An evidence-only scratch experiment on `arch/single_frame_first`, run unattended on Main: local RTX 5080, Python
`C:\Projects\AnimCV\.venv_build_full`, Blender 5.1.0. `docs/next_batch.md` was read only.

Every Blender number, test result, render and manifest below is **AGENT-RUN** on Main. The Reviewer has not re-executed
any of it. The visual review is the agent's own (**AGENT REVIEW**), not an animator's.

## 1. Official CASE and scope

**Verdict: CASE D — INVALID** (frozen rule, `results/decision/decision.json`, SHA-256 `90adb6cb95f847e7…`).

- Two validity gates failed: **V10_repeatability** and **V12_negative_controls**.
- **Both are instrument-specification errors of mine.** Neither is a mechanism, evaluator or data failure, and neither was
  repaired after the outcome:
  1. **V10.** The gate compared the two builds' request records *including* `wall_s`, the per-variant solver wall-clock time
     that `factor_transition` stores in its info. That field is nondeterministic by construction, and it is the only
     differing key. Everything V10 is meant to protect is byte-identical across the two runs: both builds' states files, and
     both Blender evaluations.
  2. **V12 / DG5.** I declared the display control "REJECTed result shown as ACCEPTED" on the negative-control base's G1F0,
     which was **ACCEPTED**. Relabelling an accepted result as ACCEPTED changes nothing, so the truth check could not fail.
     (W120 placed this control on a rejected negative.) All 19 other W121 negative controls, display controls DG1, DG2 and
     DG4, every source-level control and the inexact-rollback controls behaved as declared. DG3 was not applicable: there was
     only one changed run.
- **Labelled post-hoc check, no CASE authority** (`scripts/w121_posthoc_checks.py` → `results/posthoc/posthoc_checks.json`,
  `103953ce…`):
  - Build records are equal once `wall_s` is excluded.
  - DY5 applied to two genuinely REJECTed variants (S1_REACH and S1_HIGH_RBpos, G1F0) fails the truth check with
    `STATUS_LABEL_DISAGREES_WITH_DECISION`.
  - With only those two defects set aside, every other V gate and every C gate passes, and the frozen rule would output
    **B — PARTIAL / CONTEXT-SPECIFIC**. **B is not assigned.**

**What the population is.** Three genuinely independent, openly licensed, source-authenticated dynamic videos (S1–S3). This
is a multi-source test, but a small one: three clips, nine predeclared hold windows, 27 evaluation requests. It is not a
broad generalisation sample.

**Historical verdicts are unchanged:** W113 D, W114 D, W115 D, W116 B, W117 C, W118 D, W119 D, W120 B. The W120 decision
re-hashes to `1029ce92…` with case B after the run. The W120 population was not used in any W121 denominator.

## 2. Starting state and revisions

| Item | Value |
|---|---|
| Instruction blob | `1d9f48f88b12b348b80ebb1b4a5ff15dfa403dfa`, verified first |
| Starting commit | `7a61b5fb05b885037f850a15eca28c358a6bf828`, equal to HEAD and to `origin/arch/single_frame_first` after fetch |
| Main-local lock | batch 121 with this blob and commit |
| Pre-outcome instrument commit | `06f4327`: scripts and tests, committed after the declaration and before any factor variant was built |
| Final source revision | the Worklog commit following `06f4327` |

The Worklog commit also adds the read-only post-hoc script. The COMPLETE marker is a separate, final commit.

## 3. Source procurement, permission, provenance and duplicate screening (Part 1)

**Declared protocol** (`scripts/w121_independent_transfer_base.py`, metadata only, applied before any download):

- Wikimedia Commons API search, fixed 10 queries × 30 results.
- **R1** free licence (CC0, PD, CC BY, CC BY-SA).
- **R2** webm or ogg container.
- **R3** duration 4–60 s.
- **R4** short side ≥ 360 px.
- **R5** real time: title or description must not match the slow-motion / time-lapse regex.
- **R6** title names a declared human activity.
- **R8** size cap.

Raw API responses are stored and hashed in `results/acquire/api_raw`. Access time: 2026-11-07T05:16Z. The User-Agent is a
generic research string with no personal contact data.

**Screen outcome** (`acquire.json`, `d41aebaa…`):

- **205 candidates screened.**
- Rejections, with overlapping reasons: R6 133, R3 97, R4 15, R8 10, R2 2, R5 2.
- **44 passed** the metadata screen.
- **10 originals were downloaded.** Every one matches its Commons SHA-1, and each is stored with SHA-256, licence, author,
  URL and access time under `C:\Users\dna10\animcv-w121\data\originals` (outside the repository; 41 MB).
- **34 were not downloaded.** upload.wikimedia.org began returning HTTP 429 to this IP; even a HEAD request was refused.
  - The first attempt stopped on the 429 (`acquire_attempt1_failed_http429`).
  - The second retried with back-off but kept hitting the 429. I aborted it (`acquire_attempt2_aborted_http429`).
  - The final run reused the 10 complete, SHA-1-verified files and tripped a declared circuit breaker. It sent no further
    requests rather than hammer the server.
  - The 34 are listed in `summary.not_downloaded_rate_limited`. Download order was alphabetical by title, so the subset is
    not content-selected, but it is not the whole eligible pool.
- **Screen limitation:** "Kettlebell Dead Snatch Squat Slow-mo" passed R5 because the regex does not match "slow-mo". It was
  never downloaded.

**Inspection** (`inspect.json`, `f95d9b95…`):

- Every candidate was decoded with the frozen W102 decode (cv2/FFmpeg), with per-frame decoded-BGR SHA-256, a 64-bit dHash
  fingerprint and shot cuts.
- Duplicate test against the W102/W105/W116–W120 source (`0dda8a5f…`): the longest temporally ordered matched-frame run must
  be < 10.
  - All 10 candidates have a run of **0** against the historical clip.
  - There are no cross-candidate duplicates.
- **Timing hazard found:** `Boxing training.webm` reports **30 000 fps** in its container, while its presentation timestamps
  step at about 33.97 ms. The frozen diagnostic's `timestamp = index / fps` would therefore misstate source time.
  - I added a declared, pre-perception cadence rule: the container step must equal 1000/fps within 1 ms. That rule rejects
    this clip. It was not repaired.
  - The first inspection run, without the rule, is retained as `inspect_attempt1_no_cadence_rule`.

**Frozen-perception suitability** (Part 2 rule, applied before any solver, `perceive.json`, `0fb80de1…`). A segment must be a
subject-present, rig-available run inside one shot with:

- length ≥ 3 s and ≥ 90 frames, capped at 240 frames;
- left shoulder, elbow and wrist valid on ≥ 80 % of frames;
- observed 2D left-wrist extent ≥ 0.5 × the median torso length.

Selection takes one source per author, uploader and activity keyword, ranked by left-arm valid fraction.

| Candidate | Outcome |
|---|---|
| Bdelloid juggling (rotifer, no human) | subject on 45/457 frames: no segment |
| Bench press | no eligible segment: static wrist and short rig-available runs |
| Deadlift | left arm valid 34 % |
| Chen Taiji push hands (two people) | left arm valid 66 % |
| Fire juggling Hawaii (night) | 11 cuts, arm valid ≤ 67 % |
| Boxing training | rejected at inspect (cadence) |
| Combat juggling Berlin | eligible, but same activity keyword as S2 and the 3-source cap was reached |
| **Selected** | **S1, S2, S3** below |

**Accepted sources:**

| ID | Commons file | Licence / author | Original SHA-256 | Segment (source frames, 30 fps) | Content |
|---|---|---|---|---|---|
| S1 | Double Kettlebell Strict Press (side view).webm | CC BY-SA 4.0 / Taco fleur | `de8feb9d515daa2b…` | 79–318 (8.0 s) | one man, side view, overhead presses, outdoors |
| S2 | Firejuggling 2009 ubt.ogv | CC BY 3.0 / Tomasz Sienicki | `68301b0c35c98448…` | 280–519 (8.0 s) | two street jugglers plus crowd; tracked subject's fast juggling arm |
| S3 | Bent-over row - exercise demonstration video.webm | CC BY 3.0 / FitnessScape | `f6b90f2f72e9020d…` | 0–153 (5.1 s) | one woman, barbell row, studio, with text overlay |

The three sources differ in subject, motion, camera and publisher. Video bytes are not redistributed. Only provenance, hashes
and compact checks are published here.

## 4. Frozen perception and source-to-rig preparation (Part 2)

**Frozen stack.** The unchanged W102 M1 + S0 video diagnostic was used: RTMDet-tiny + RTMPose-tiny at batch 1 on CPU, the
0.3 visibility policy, and frozen S0 `S0_neutral_sign` on CUDA. It ran from the W102 code snapshot (diagnostic script equal
to the repository copy after LF normalisation). The only change is a process-local launcher that redirects tmp and cache
files to `animcv-w121`, so no W102 directory is written.

**Reproduction control.** The rerun on the historical clip reproduces W102's stored observations/predictions
**byte-exactly**: every array, and the file SHA `6784357f…`.

The diagnostic's own checks passed for every new clip: checkpoint, config, environment-fingerprint and S0-identity. Raw
detector scores are retained in the npz landmarks.

**Coverage inside the segments:**

| Source | Subject present | Left arm valid | Rig available | Mean person candidates per frame (whole clip) | 2D continuity-proxy flags |
|---|---|---|---|---|---|
| S1 | 240/240 | 100 % | 240/240 | 1.00 | 0 |
| S2 | 240/240 | 100 % | 240/240 | **12.9** (crowd + second juggler) | 1 (frame 280, the segment's first frame) |
| S3 | 154/154 | 100 % | 154/154 | 1.08 | 0 |

**Subject association is UNKNOWN on every frame.** The only evidence is the frozen tracker plus a 2D bbox-IoU proxy;
continuity was not verified. On S2 a switch to the other juggler cannot be excluded.

**Source to rig:**

- The **W105 GT-free S0-only configuration**, FIXED slot: W93 torso frames from S0 as Object and spine; S0 legs; S0 left arm
  through the existing chain; right arm at rest.
- **Adapter fidelity:** my FIXED-slot function, applied to W102's stored predictions, reproduces W105's `real_clip` FIXED
  values on 199/199 frames. My parent-value extraction reproduces the W116 v0 values. Keying W116 v0 in Blender reproduces
  the W105 FIXED Action digest `4d168d95234c…`.
- **New parents:**

  | Source | State id | Blender value digest | F-curves | Max FK error vs Blender |
  |---|---|---|---|---|
  | S1 | `cc4bf006…` | `25a09802…` | 32 | 1.3e-6 m |
  | S2 | `23bb276e…` | `88ddaabe…` | 32 | 1.2e-6 m |
  | S3 | `b0370139…` | `b9058745…` | 32 | 1.3e-6 m (gate 1e-5) |

- **Bindings:** left_wrist, left_elbow, left_shoulder, right_wrist, right_shoulder and pelvis_root are **BOUND** on all three.
  right_elbow, thorax and left_hand are AMBIGUOUS and nose is MISSING, as in W116.

**Plausibility, GT-free.** Trunk angle from world +Z (median):

| Source | Trunk angle | Note |
|---|---|---|
| S1 | 13.5° | |
| S2 | 12.4° | |
| S3 | 41.8°, max 56° | consistent with the bent-over posture in the footage |
| W102 clip | 6.9° | reference |

Left-arm reach slack minimum: 8 mm (S1), 1.4 mm (S2), 3.8 mm (S3). No parent frame has flex < 5°.

**Raw source discontinuities are large and unrepaired.** Maximum per-frame wrist step of the unedited parent:

| Source | Max step |
|---|---|
| S1 | 0.877 m (one jump) |
| S2 | 0.628 m |
| S3 | 0.228 m |

S2's median is 90 mm per frame at 30 fps.

**Time.** For every source:

- Action frame index = source frame index.
- Blender frame = index − f0 + 1, verified on every frame.
- Source time = index / 30.0, and the container cadence is consistent.
- Played at the W116 30 fps scene rate, playback is real time (factor 1.0). This differs from the W102 clip's unresolved
  50/30 mismatch.

## 5. Prospective task contract, populations and IDs (Part 3)

The declaration `results/declare/declaration.json`, SHA-256
**`0616efdfebb940a57dc2d5cbe410b6ca8dfea32ccec1ebb5b1ddf548d38039fc`**, records `outcomes_inspected: false` and binds:

- all stage reports and the frozen-controls hash (`0d5cd1da…`);
- the code blobs of 15 W102/W105/W116–W121 files;
- the gates, the CASE rule, the negative-control expectations and the visual-review procedure.

Before that point only U, F_r and X_r baselines existed; no factor variant had been built.

**Hold windows** were chosen from the **unedited parent's Blender wrist path only**:

- Window length L = round(0.30 s × fps) = 9 frames, keeping ≥ 9 frames from each clip end.
- D = the maximum wrist travel from the capture-frame position within the window.
- The windows are the minimum-D, median-D and maximum-D candidates (LOW, MID, HIGH), non-overlapping.

| Source | LOW (D) | MID (D) | HIGH (D) |
|---|---|---|---|
| S1 | 210–218 (21 mm) | 271–279 (144 mm) | 100–108 (958 mm) |
| S2 | 292–300 (113 mm) | 447–455 (273 mm) | 420–428 (632 mm) |
| S3 | 41–49 (22 mm) | 11–19 (257 mm) | 112–120 (585 mm) |

**User intent per window.** A WORLD_POSITION_PIN holds the left wrist at its capture-frame position, under one of three edits:

- `none`;
- `RBpos`: a mock torso edit, `spine_01` +5° side bend on every frame;
- `RBneg`: the same bend at −5°.

The authorised DOF is {upperarm_l, lowerarm_l}. The automatic budget is the W118 M_MAX = 8 frames per side, bounded and
shown to the user.

**Population, 14 requests per source:**

- **27 evaluation requests** (9 per source);
- **3 zero-budget controls** `S*_ZB` (LOW window, RBpos, m_max = 0);
- **12 special negative requests** `S*_REACH` (+1 m Z target), `S*_DOF` (forearm only), `S*_CONFLICT` (two pins 2 cm apart)
  and `S*_WRONGBIND` (left_wrist pin carrying the right_wrist binding);
- plus 19 W120-pattern negative controls on S1's declared negative-control base `S1_MID_RBpos`. The base was chosen in
  calibrate from the frozen controls by the declared rule.

**Genuine negatives retained.** Ten evaluation requests are infeasible hard pins (UNREACHABLE frames) under the frozen W116
resolver. They are REJECTed with INFEASIBLE_REQUEST and stay in the denominators:

- all 9 HIGH requests;
- S3_LOW_RBpos.

Under the same rules, S3_ZB is also infeasible.

## 6. Matched frozen mechanism, fairness and controls (Part 4)

Each request is built from frozen components:

- **U_r:** the edit on every frame.
- **F_r:** the W116 analytic framewise resolve on this request's pin frames only.
- **X_r:** W118 `transition_edit`.
- **G0F0 / G0F1 / G1F0 / G1F1:** the unchanged `w120_factor_attribution_base.factor_transition`, with X_r's margins and
  X_r's own m_max (0 for ZB). The solver-parameter digest is the W120 one.

No W116–W120 file was modified; they were imported.

**Gate results on matching and baselines:**

- **V4.** U, F_r and X_r state ids, statuses, margins, budgets and planned frames in both builds equal the frozen controls.
- **V5.** Every contract-valid evaluation or control request passes F_r isolation at both value level and Blender level.
- **V6.** For every factor:
  - `ab_fairness(F_r, X_r, variant)` is OK;
  - `cross_factor_identity` holds;
  - every ZB budget equals its pin frames in all four variants.
- **V7.** The W120 fixture preflight is all_ok: the 262,144-case precedence check has 0 mismatches. The 19 declared
  negative-control expectations equal the frozen precedence predictor.

**G0F0 vs X_r (Blender, transition frames).** The wrist is reproduced. Elbow and roll differ, as W120 disclosed:

| Source | Max elbow difference | Max lowerarm_l roll difference |
|---|---|---|
| S1 | up to 33 mm (HIGH; MID_RBneg 16.5 mm) | — |
| S2 | ≤ 10 mm | up to 48° (lowerarm_l roll) |
| S3 | ≤ 9 mm | up to 15° |

The G contrasts are therefore contrasts within the shared analytic solver; they are not X-vs-guide contrasts.

## 7. New-source results, kept separate from W120 history (Parts 4–6)

**Population counts.** Evaluation requests: X_r and all four factors **ACCEPT 17 and REJECT 10** (INFEASIBLE_REQUEST), with
identical decisions across variants. No factor created a new rejection. Controls: S1_ZB and S2_ZB are ACCEPTed; S3_ZB is
REJECTed as infeasible. **17 requests are factorial-eligible:** S1 6, S2 6, S3 5.

**B1 = peak boundary wrist compensation step versus U** (Blender, mm). `*` marks a meaningful improvement under the frozen
W119 comparator (≥ 20 % and ≥ 10 mm, with no new flip, near-straight frame or material worsening).

| Request | X / G0F0 | G0F1 | G1F0 | G1F1 | guide@F0 | guide@F1 | floor@G0 | floor@G1 |
|---|---|---|---|---|---|---|---|---|
| S1_LOW (none / RBpos / RBneg) | 9.2–9.6 | 9.1–10.2 | 9.4–9.9 | 9.4–9.7 | ≤ ±0.3 | ≤ ±0.6 | ≤ ±0.6 | ≤ ±0.3 |
| S1_MID_none | 27.4 | 27.7 | 24.0 | 24.0 | −3.5 | −3.8 | +0.3 | 0 |
| S1_MID_RBpos | 18.6 | 18.6 | 17.9 | 17.9 | −0.7 | −0.7 | 0 | 0 |
| **S1_MID_RBneg** | 51.5 | 53.4 | 34.0 | 34.0 | **−17.5 \*** | **−19.4 \*** | +1.9 | 0 |
| **S2_LOW_none** | 35.8 | 35.8 | 14.4 | 14.4 | **−21.4 \*** | **−21.4 \*** | 0 | 0 |
| **S2_LOW_RBpos** | 24.3 | 24.3 | 9.4 | 13.6 | **−14.9 \*** | **−10.7 \*** | 0 | +4.2 |
| **S2_LOW_RBneg** | 48.7 | 48.7 | 19.6 | 19.6 | **−29.2 \*** | **−29.2 \*** | 0 | 0 |
| S2_MID (none / RBpos / RBneg) | 49.1–52.9 | 42.2–43.2 | 45.3–45.8 | 45.3–45.8 | −3.8 to −7.1 | +2.6 to +3.1 | −6.9 to −9.7 | 0 |
| S3_LOW (none / RBneg) | 9.0–9.4 | 9.0–9.4 | 7.6–8.8 | 7.6–8.8 | −0.3 / −1.8 | same | 0 | 0 |
| S3_MID (none / RBpos / RBneg) | 47.6–49.4 | 47.6–49.4 | 42.7–43.0 | 42.7–43.0 | −4.6 to −6.7 | same | 0 | 0 |

**Per mechanism, on new-source evaluation requests:**

- **GUIDE (primary, guide@F0).** Meaningful improvement on **4 requests in 2 distinct windows of 2 sources**: S1 MID
  (RBneg only) and S2 LOW (all three edits). No meaningful improvement on any **S3** window. The conditional guide@F1 improves
  the same 4 requests.
- **FLOOR.** No meaningful improvement in either contrast. On S2 MID, floor@G0 lowers B1 by 6.9–9.7 mm, below the 10 mm
  absolute rule. floor@G1 worsens S2_LOW_RBpos by +4.2 mm.
- **Interaction:** |I| < 10 mm on every request.
- **Clean articulation.** No variant introduces a bend flip or a near-straight frame relative to U on any eligible request.
  In the primary contrasts, M_clean holds for both mechanisms.
- **Effect on minimum flex.** G1 raises the minimum elbow flex on S2 LOW from 9.4–10.3° to 17–27°.
- **Guide reachability on transition frames:**

  | Variant | REACHED | Floor-clamped | UNREACHABLE_OUTER |
  |---|---|---|---|
  | G0F0 | 296 | 0 | 0 |
  | G0F1 | 234 | 62 | 0 |
  | G1F0 | 288 | 0 | 8 |
  | G1F1 | 263 | 25 | 8 |

  BEND_PLANE_AMBIGUOUS: 4 (G0) and 8 (G1). BRANCH_FLIP flags: 7 (G0) and 4 (G1). There was no nonconvergence.

**Source-relative peak versus raw motion** (stated explicitly, as the batch requires). In the improved cases G1 lowers the
edit-induced compensation step but can **raise the raw wrist step**:

- **S1_MID_RBneg, release frame 281.** Raw step: G0F0 66.0 mm, G1F0 87.5 mm, U 87.8 mm. G1 restores the source's own release
  speed; it does not slow it.
- **S2 LOW.** Raw steps rise by up to 12.8 mm at 301 and 305 and fall by up to 15.8 mm at 303. These changes sit on top of
  source raw steps of 60–330 mm per frame.

In no improved case did D2 or D3 worsen beyond the comparator rule. The release-compensation series of X/G0 oscillates after
release (e.g. S2_LOW_RBneg: 20, 47, 29, 49, 26 mm per frame); G1's series decays smoothly (≤ 20 mm).

**W120 history, kept separate.** W120's only reachable real guide gain was one RH2 release in the W102 clip. On new sources
the guide again helps some releases. The same caveats recur: the raw source jump is unchanged, the effect is a
source-relative peak, and only some windows improve.

## 8. Blender hard pins, authorised change, footprint, timeline and undo (Part 5)

- **C1, hard pin and scope.** Every ACCEPTed X_r and factor variant, across 855 pin-frame Blender evaluations, has residual
  ≤ 1 mm. The maximum is **6.7e-7 m**, and **0 frames are over 1 mm**. Changed channels are only upperarm_l and lowerarm_l.
  Changed frames lie within the planned domain and the budget. There are no outside-arm point changes and no root change.
- **C2, truthful footprint.** Every variant's proposed and result display passes the independent truth check. The planned
  domain is identical across X and the four variants and lies within the budget. Preview id = committed id, and the result
  label = the decision.
- **C5.** The user-edit channel (`spine_01`) and `__object__` are bit-equal to U on every ACCEPTed variant.
- **C6.** There is no fallback, no G0_REFERENCE_MISSING and no infeasible request accepted.
- **C4.** The originals, the frozen W102/W105/W116/W120 inputs and the rig are byte-unchanged across assess (SourceGuard).
- **C3, lineage.** On each source:
  - R1 ACCEPT of `S*_MID_RBpos` G0F1 moves the head.
  - R2: an atomic REJECT (S1: NG_PIN_COMBINED; S2 and S3: the REACH request) leaves the head unchanged.
  - R3: a commit against the old parent gives **STALE_PARENT**.
  - R4: revert. The values are byte-equal to v0, and the Blender evaluation is bit-equal to v0's (value digest and all frames).
  - R5, inexact rollback: a 1e-6 perturbation of one quaternion is detected.
- **Timelines.** `render/timelines/` holds 61 timelines, in the W120 row layout. Each shows:
  - the pin domain;
  - the authorised automatic budget and the planned domain;
  - the per-variant actual Blender footprint and changed bones;
  - per-variant target status.

  The footer gives the true source time axis (30 fps). REJECTed candidates are hatched and labelled; none is shown as applied.

## 9. Real-time and frame-time behaviour

All three sources are 30.000 fps with cadence-consistent containers. One Action frame is 33.3 ms of source time, so
playback at the 30 fps scene rate is real time, and per-frame steps convert to speed by multiplying by 30. For example, the
S1_MID_RBneg release step of 87.5 mm per frame is 2.6 m/s.

A smaller per-frame compensation peak was not shown to give perceptibly smoother real-time playback; see §10.

## 10. Agent visual review (Part 6; `review/visual_review.json`, `2eb9a200…`; written after render, before decide)

**Method.** The full-window sequence sheets were too low-contrast to judge at sheet scale. I therefore inspected autocontrast
full-resolution crops around each release (`review/crops`) in two world-axis views ("front" = −Y, "side" = +X; these are not
character-relative), together with the step and flex charts.

**Findings:**

- **GUIDE.** A G1-vs-G0 difference is **visible for 1–3 frames after release**: S1 frame 281, and S2 frames 302–303, where
  the forearm flares outward instead of folding to the hip. **Improved continuity is not visually established.** Source arm
  motion of tens to hundreds of mm per frame dominates what is seen. M_vis is therefore false.
- **FLOOR.** No primary improvement exists to review.
- **Not done:**
  - real-time playback;
  - a browser index;
  - full-resolution review outside the reviewed set;
  - crops of S2_LOW_RBpos (same window as the reviewed S2_LOW requests).

The "front" crops in S1 are dominated by render noise amplified by autocontrast, and were not used.

## 11. Negative controls and failure attribution

| Control | Result |
|---|---|
| NG_BUDGET, NG_MARGIN, NG_PARENT, NG_RIG, NG_CDIGEST, NG_AUTH, NG_PIN_COMBINED, NG_FALSE, NG_BONE, NG_ROOT, NG_OUTSIDE, NG_PLANNED, NG_DISPLAY, NG_SHARED | CONTRACT_VIOLATION with the declared reasons. NG_SHARED fails isolation at value and Blender level. |
| NG_NONCONV | SOLVER_FAILURE `NON_CONVERGED` |
| NG_REACH, NG_DOF, NG_CONFLICT | INFEASIBLE_REQUEST with the declared reasons |
| NG_WRONG_BINDING | UNSUPPORTED_BINDING `BINDING_JOINT_MISMATCH` |
| Display DG1, DG2, DG4 | truth check FAILS, as required |
| Display DG3 | not applicable (single changed run) |
| Display **DG5** | **did not fail; mis-specified (see §1)** |
| Source: old clip renamed | detected (SHA exclusion) |
| Source: old clip re-encoded MJPG, 7-frame shift, 0.6× re-timed | detected (dHash run ≥ 10) |
| Source: altered video byte | detected (SHA mismatch) |
| Source: stale observation binding | detected |
| Source: stale model binding | detected |
| Source: wrong FPS (30 000 and 50/30) | detected |
| Source: +0.1 s time offset | detected |
| Source: "VERIFIED" continuity claim without evidence | detected (state id no longer binds) |
| Inexact rollback (3 sources) | detected |

## 12. Alternative explanations, identity and generalisation limits

- **The S2 gains** (3 of the 4 guide improvements) come from one window of a crowded two-juggler clip. Subject association
  there is UNKNOWN, the raw source motion is very fast, and the three requests are edit variants of the same release. They
  are **one episode, not three**.
- **The S1 gain** is one edit variant (RBneg) of one window. RBpos and none in the same window do not improve.
- **S3** (bent-over row) shows no meaningful guide or floor effect in any window.
- **What B1 measures.** B1 is a source-relative peak. G1's lower peak partly reflects returning to the source's own fast
  release motion, which raw steps confirm. It is not evidence of smoother raw animation.
- **Infeasibility.** Every HIGH window is infeasible. The mechanism is untested where the pinned hand would have to stay still
  while the source moves the arm far, roughly ≥ 0.58 m of travel in 0.3 s.
- **Small population.** The 34 rate-limited candidates were never examined, and only 3 sources passed. The sources are 5–8 s
  segments of exercise and juggling footage. No generalisation to other motions, cameras, multi-person scenes or rigs is
  claimed.
- **Agent-only evidence.** The S0-only rig is GT-free and has no 3D accuracy evidence. Neither performer fidelity nor contact
  was assessed.

## 13. Files, tests, environment and reproduction

**Changed files** (all new):

- `scripts/w121_independent_transfer_base.py`
- `scripts/w121_independent_transfer.py`
- `scripts/w121_posthoc_checks.py` (post-hoc, read-only)
- `tests/test_w121_independent_transfer.py`
- this Worklog

No `src/`, W102–W120 script, test, Worklog, artefact, rig, model, threshold or `docs/next_batch.md` was changed. Two
scratch-only helpers live outside the repository under `C:\Users\dna10\animcv-w121\setup`: the frozen launcher and the review
crop tool.

**Tests (AGENT-RUN):**

| Command | Result |
|---|---|
| `python -X utf8 -m pytest tests/test_w121_independent_transfer.py -q` | **17 passed** (`results/tests/w121_tests_final.xml`); an earlier pre-declare run also gave 17 passed |
| `pytest tests/test_w120_factor_attribution.py tests/test_w119_endpoint_transition.py tests/test_w118_transition_edit.py tests/test_w117_temporal_pin.py tests/test_w116_semantic_pin.py tests/test_w115_validity.py tests/test_w114_source_preserving_base.py tests/test_w105_rig_utility.py tests/test_fk_solver.py tests/test_canonical_pose_parity.py tests/test_retarget_solver.py -q` | **165 passed** (`results/tests/w121_regression.xml`) |

The W121 tests cover:

- the metadata screen;
- dedup of copied, re-timed, shifted and noisy footage;
- shot cuts;
- segment selection (cuts, arm validity, static arm, gaps, cap);
- hold windows;
- the CASE mapping, including DATA_BLOCKED and ONE-SOURCE;
- negative-control expectations against the predictor;
- the reused W120 instrument preflight;
- isolation, zero budget, fairness and nonconvergence;
- display truth and hidden-change detection;
- hard-pin and infeasible rejection;
- exact and inexact reversal and stale parent;
- provenance binding;
- the stored frozen reproduction, adapter fidelity and cadence rejection.

**Failed:** none. **Not run:** the full suite, FBX export/import and engines.

**Execution failures retained:** `acquire_attempt1_failed_http429`, `acquire_attempt2_aborted_http429` and
`inspect_attempt1_no_cadence_rule`.

**Key artefacts** (`C:/Users/dna10/animcv-w121/results`):

| Artefact | SHA-256 prefix |
|---|---|
| acquire.json | `d41aebaa` |
| inspect.json | `f95d9b95` |
| perceive.json | `0fb80de1` |
| prepare.json | `1f7ab419` |
| calibration.json | `8e3028b4` |
| frozen_controls.json | `0d5cd1da` |
| declaration.json | `0616efdf` |
| build states S1 / S2 / S3 (r1 = r2) | `0eda8885` / `3aa6bbb1` / `2432e41e` |
| eval S1 / S2 / S3 (e1 = e2) | `a7d6d4a6` / `d6085798` / `b91482f8` |
| assess_report.json | `9f88f154` |
| visual_review.json | `2eb9a200` |
| decision.json | `90adb6cb` |
| posthoc_checks.json | `103953ce` |

`evidence_manifest.json` = `ff914c33c1571affeffa358067696fe2096f99de7790737a94e63ff9f3ad215f` covers 5,086 files and was
written before this Worklog. It excludes tmp and cache; the post-hoc file is included.

**Reproduction** (Git Bash, repository root at `06f4327`; every stage refuses an existing output directory):

```
PY=C:/Projects/AnimCV/.venv_build_full/Scripts/python.exe; R=C:/Users/dna10/animcv-w121/results
S=0616efdfebb940a57dc2d5cbe410b6ca8dfea32ccec1ebb5b1ddf548d38039fc; D="--declaration $R/declare/declaration.json --declaration-sha $S"
$PY -X utf8 scripts/w121_independent_transfer.py acquire  --out-dir $R/acquire        # network; reuses SHA-1-verified originals
$PY -X utf8 scripts/w121_independent_transfer.py inspect  --acquire $R/acquire --out-dir $R/inspect
$PY -X utf8 scripts/w121_independent_transfer.py perceive --acquire $R/acquire --inspect $R/inspect --out-dir $R/perceive
$PY -X utf8 scripts/w121_independent_transfer.py prepare  --acquire $R/acquire --perceive $R/perceive --out-dir $R/prepare
$PY -X utf8 scripts/w121_independent_transfer.py calibrate --prepare $R/prepare --out-dir $R/calibrate
$PY -X utf8 scripts/w121_independent_transfer.py declare  --acquire $R/acquire --inspect $R/inspect --perceive $R/perceive --prepare $R/prepare --calibrate $R/calibrate --out-dir $R/declare
$PY -X utf8 scripts/w121_independent_transfer.py build $D --tag r1 --out-dir $R/build_r1   # and r2
$PY -X utf8 scripts/w121_independent_transfer.py evaluate --build-dir $R/build_r1 --out-dir $R/eval_e1   # and eval_e2
$PY -X utf8 scripts/w121_independent_transfer.py assess $D --build-dir $R/build_r1 --eval-dir $R/eval_e1 --out-dir $R/assess
$PY -X utf8 scripts/w121_independent_transfer.py render $D --build-dir $R/build_r1 --assess-dir $R/assess --out-dir $R/render
# agent review -> $R/review/visual_review.json
$PY -X utf8 scripts/w121_independent_transfer.py decide $D --build-dir $R/build_r1 --build-dir2 $R/build_r2 --eval-dir $R/eval_e1 --eval-dir2 $R/eval_e2 --assess-dir $R/assess --visual-review $R/review/visual_review.json --out-dir $R/decision
$PY -X utf8 scripts/w121_posthoc_checks.py
$PY -X utf8 scripts/w121_independent_transfer.py manifest --declaration-sha $S
```

`perceive` needs `C:/Users/dna10/animcv-w121/setup/w121_frozen_launch.py` and the W102 snapshot and data; the run commands
are logged in `perceive/candidates/*/m1s0.log`. Blender is `C:/Program Files/Blender Foundation/Blender 5.1`.

**Resources and environment:**

- **Timing:** perception about 25 min (CPU M1, 9 clips plus the historical reproduction); prepare 8 s; calibrate 2 s;
  builds a few seconds each; Blender evaluation 28 s per run; assess 18 s; render 86 s; tests about 1 s; regression 9 s.
- **Storage:** 1.1 GB under `C:\Users\dna10\animcv-w121`, local and unpublished.
- **Software:** Python 3.11.9, numpy 1.26.4, torch 2.13.0+cu130, mmpose 1.3.2, OpenCV 4.11.0 (FFmpeg), Blender 5.1.0,
  Windows 10.0.26200.
- **Scope:** Main only. No LabServer63/69 connection, no training, no package change. Network access was limited to the
  Wikimedia Commons API and original-file downloads, stopped by the circuit breaker on HTTP 429.
- **Git:** single branch `arch/single_frame_first`. No branch, merge, rebase, reset or force operation.

## 14. Explicit non-claims and decision-relevant unresolved evidence

**Not claimed:**

- the CASE B that the post-hoc alternative computes;
- architecture success;
- multi-clip generalisation;
- animator usability;
- 3D or video fidelity;
- contact safety;
- trained motion generation;
- Production readiness;
- any user-facing threshold. The 1 mm pin, the 5° near-straight reference and the W119 comparator remain experiment-only
  diagnostics.

**Completed:** Parts 1–6 for three accepted sources: acquisition, authentication, frozen perception, rig ingest, prospective
contract, matched build, Blender evaluation, footprint, lineage, agent review, decision and manifest.

**Not completed:**

- examination of 34 rate-limited candidates;
- real-time playback review;
- independent re-execution.

**Unresolved, decision-relevant:**

- Whether the S2 guide gains depend on which juggler was tracked.
- Whether any guide effect becomes visible as smoother motion once source jitter is lower; S3 is the cleanest source and shows
  none.
- How the mechanism behaves on feasible high-travel holds, which this population could not test.
- Whether including the rate-limited candidates would change source coverage.

