# Worklog — Visual Advisor for Runtime Heading Sign Ambiguity (2026-10-23)

Branch: `arch/single_frame_first`. Starting HEAD `d8316a4` (Worklog 91A). Local, origin and LabServer63 were in sync and
the worktree was clean.

Nothing was trained, and no checkpoint, S0, W91 provider, FK/IK, BaseRig or production default was changed. W91 stays
Engineering CASE B and W91A stays rejected and unwired.

**Where it ran:**
- LabServer63 (idle): the frozen Qwen3-VL-8B inference and the server tests.
- Locally at low priority: evaluation, IK replay and Blender.

## 0. Answer

**Limited. Preserved as an opt-in, unwired experimental component; not promoted.**

**The advisor itself works.** The frozen Qwen3-VL-8B torso up/down advisor
(`motion.vlm_heading_sign_advisor`) gives image-grounded evidence and repairs most of the W91 mirrored-sign failures.
It does so without measurable damage to correct S0 headings, and abstains widely.

| 3DHP | >135° heading error | longest wrong-heading run |
|---|---|---|
| S0 | 1.14% | 18.8 s |
| S0 + advisor | **0.26%** | 4.0 s |
| oracle-sign upper bound | 0.18% | 4.0 s |

- 27 of 38 wrong-branch rows are corrected and 3 are introduced.
- Head-turn suspects: 0 introduced.

**But the rig does not get better, so the animation benefit is blocked by a different contract.** The failures it fixes
are almost all **lying torsos**. AnimCV's Root Orientation is **yaw-only**, so a lying body is still rendered upright:

- after the correction, the J1X arm's local-rotation error is unchanged in those scenes (76.1° vs 76.2°; 76.5° even
  under GT heading);
- over all 3DHP rows the gain is −0.3° (CI [−0.9, 0.0]);
- sudden root spins go up from 9 to 12.

## 1. First gate: what the TS2 failure really is

**Review frames** (`scripts/review_runtime_heading_side_failures.py`, fixed rules; contact sheets in
`~/animcv-output/92_framepose_vlm_heading_advisor/ts2_failure.png`, `ts2_correct.png`, `ts4_flip.png`,
`ts1_correct.png`):

- The 3DHP TS2 run (16–38 s) is **not an upright side view**. The subject lies on his back on the floor, head toward
  image left, legs raised.
- For a horizontal torso, the yaw of the right-minus-left shoulder axis reduces to **which shoulder is nearer the
  camera**. S0 picks the mirrored depth order:

  | frame | GT yaw | S0 yaw |
  |---|---|---|
  | 16.0 s | −94° | +89° |
  | 20.4 s | −95° | +122° |

- The **magnitude is right and the depth sign is wrong**: mirroring ψ → −ψ brings the error from 170–180° to about 5–30°.
- For a person the image is unambiguous: chest up, head left, so his right shoulder is nearer.
- **This also explains W91A.** For a lying body the nose offset measures where the head is along the body axis, not
  which way the person faces, so it agreed with S0's mirror.
- The W91A TS4 "head-turn" frames are lying poses too (supine, head toward image right).

**Quantified with the GT torso inclination** (pelvis→thorax vs vertical; evaluation only):

| population | >135° rows | where | mirroring fixes (<45°) |
|---|---|---|---|
| 3DHP | 31 | **all lying (≥ 60°)**. Lying rows: 41 of 417 are >90°; upright rows: 0 of 2,112 >135° | 30 of 31 |
| 3DPW TEST | 5 | 4 bent (45–60°, the seated café subject), 1 upright | 4 of 5 |

**Gate passed, with a reframing.** The failure is a genuine, visually distinguishable mirrored-sign ambiguity, but of
**lying or strongly bent torsos**, not of upright side-on views. The advisor therefore asks a torso-specific question
that is salient in exactly these frames.

## 2. Historical VLM findings and how this task differs

| worklog | task | result |
|---|---|---|
| W28/W29 (Qwen2-VL-2B) | bilateral shoulder/hip near–far sign | chance (balanced 0.49–0.50) |
| W28/W29 (Qwen2-VL-2B) | torso facing | 0.75 balanced, combined prompt |
| W67 (Qwen3-VL-8B, 4-bit, pinned) | left-arm segment near/far ordering | dominated by a fixed answer prior (balanced 0.50–0.59) |

The question "which shoulder is nearer" is the historically failed near–far task, so it was **not** asked. The advisor
instead asks the much more salient **"on the back or on the stomach"**, and turns the answer into the shoulder-axis
depth sign through geometry. The W67 pinned environment is reused unchanged: image `animcv-qwen3vl:cu124`, revision
`0c351dd0…`, NF4, greedy.

## 3. Advisor contract (declared in the module before any response)

| item | contract |
|---|---|
| Invocation (runtime only) | S0's lifted torso inclination ≥ 45° (closer to horizontal than vertical) and a valid detector torso |
| Invocation share | 6.2% of 3DPW VAL, 3.2% of TEST, 18.2% of 3DHP rows (931 in total) |
| Image | the FramePose observation crop (`crop_box` / `render_crop`) at 448 px, **no overlay** |
| Prompts | "on their BACK (chest/belly up) or STOMACH (chest/belly down), or UNCLEAR", asked twice with the option order swapped; prompt SHAs in `model.json` |
| Answer rule | used only when both orders agree on BACK/STOMACH |
| Geometry | chest normal n = ±Z, 2D torso axis b (shoulder midpoint − hip midpoint), shoulder axis s = n × b, so the predicted depth sign = sign(n_z · b_x) |
| Decisions | **KEEP** if the prediction agrees with S0's sign(sin ψ); **MIRROR** (ψ → −ψ) otherwise |
| ABSTAIN (keeps S0) | inconsistent or UNCLEAR answers; \|b_x\| < 0.25 (torso axis toward the camera); \|sin ψ\| < sin 20°. Never an arbitrary yaw |
| Head direction | never used |
| Cost | 0.18 s per image (2 per invoked frame) on LabServer63 with 4-bit NF4; 2,793 generations in 11.6 min |

**Prompt tuning.** None. 3DPW VALIDATION was used only to confirm parsing: every answer there was UNCLEAR. With GT
up/down labels substituted for the VLM (evaluation-only oracle), the geometry is verified: 3DHP invoked >135° rows drop
27 → 1 with 0 damaged.

## 4. Results

**Method.** `scripts/run_vlm_heading_sign_advisor.py` (server) and `scripts/eval_vlm_heading_sign_advisor.py`
(local). Identical frames and detector observations; GT used only to score.

| population | S0 >90° / >135° / longest wrong run | W91 C reliability | **S0 + advisor** | oracle sign | advisor KEEP / MIRROR / ABSTAIN | corrected / introduced (>90° change) |
|---|---|---|---|---|---|---|
| 3DPW VAL | 0 / 0 / 0.8 s | 0 / 0 (cov 99.3%) | unchanged | unchanged | 0 / 0 / 210 | 0 / 0 |
| 3DPW TEST | 0.14% / 0.07% / 4.0 s | 0.10% / 0.03% (cov 98.7%) | **0.08% / 0.03% / 4.0 s** | 0.08% / 0.03% | 1 / 5 / 219 | 4 / 0 |
| 3DHP all | 1.61% / 1.14% / 18.8 s | 1.00% / 0.66% (cov 95.1%) | **0.66% / 0.26% / 4.0 s** (cov 100%) | 0.44% / 0.18% | 36 / 36 / 424 | **27 / 3** |
| 3DHP TS2 | 7.9% / 6.1% / 18.8 s | 4.6% / 3.5% | **2.7% / 1.2% / 4.0 s** | 1.6% / 0.8% | 14 / 36 / 32 | 27 / 3 |
| 3DHP outdoor | 0.16% / 0 | 0.19% / 0 | unchanged | unchanged | 5 / 0 / 139 | 0 / 0 |

**Mean heading error** (3DHP): 17.20° → 15.83° (oracle 15.58°).

**Temporal direction flips** (3DHP): 9 → 12; the oracle reaches 13. Correcting a mirrored run creates a step at its
edges unless the whole run is corrected.

**Grounding and controls**

- **Option-order consistency:** 96.8% on 3DHP and 93.3% on 3DPW TEST.
- **Confusion on 3DHP GT-lying invoked rows** (consistent answers):

  | GT label | answered BACK | answered STOMACH | UNCLEAR | inconsistent |
  |---|---|---|---|---|
  | BACK | **67** | 0 | 1 | 2 |
  | STOMACH | 5 | **7** | 283 | 3 |

  - 67 of 72 BACK answers are true backs, and all 7 STOMACH answers are correct.
  - Prone, crawling and push-up frames are mostly UNCLEAR, so the advisor abstains there.
  - A constant BACK prior would have spread BACK over the 296 GT-STOMACH rows; it did not.
- **Shuffled-image control** (another sequence's crop, same row): 479 of 496 answers become UNCLEAR. The decision follows
  the image, not the row.
  - **This control is weak by construction.** Most donors are themselves non-lying crops. The confusion matrix above is
    the stronger evidence.
- **The 3 introduced errors** are TS2 frames 1100–1130: GT-prone bodies (inclination 67–69°) that the VLM called BACK.
- **Head-turn suspects** (the 22 rows the rejected W91A nose rule would flip): KEEP 14, ABSTAIN 8, introduced 0.
- **Correct S0 headings.** 3DHP invoked rows where S0 was already right (≤ 45°): the advisor mirrored 4 of 432 (0.9%),
  and 3 of those became >90° errors (the frames above). 3DPW TEST: 0 of 212.

## 5. Rig validation

**Method.** `scripts/run_vlm_heading_rig_replay_3dhp.py`:
- frozen J1X left arm (W87 `predictions_3dhp.npz`, `356f8cee…`) through the unchanged W61/W76 IK and W60 root rotation;
- 2,570 3DHP frames;
- modes: S0, S0 + advisor, GT (evaluation only).

| metric | S0 | S0 + advisor |
|---|---|---|
| torso heading >135° | 0.66% | **0.19%** |
| torso heading mean / p95 | 16.1° / 39.4° | 15.3° / 38.3° |
| upper-arm local rotation vs GT-reference arm | 21.8° | 21.5° (Δ −0.3°, CI [−0.9, 0.0]) |
| sudden root spins (>90° step while GT < 45°) | 9 | **12** |

**Clips** (`~/animcv-output/92_framepose_vlm_heading_advisor/3dhp_*.mp4` plus PNGs; same Blender pipeline as W91; bone
proxies, no skinned mesh):

| clip | observation |
|---|---|
| `3dhp_ts2_corrections` (TS2 from 10.0 s) | Heading 170° wrong → 0.4° after MIRROR. **The rig still stands upright** while the actor lies on the floor: the yaw-only root cannot pitch the torso. The arm's local error stays ~76° in all three modes, GT heading included |
| `3dhp_worst_introduced` (TS2 frames 1100+) | The 3 BACK-for-prone mistakes show up as brief root spins |
| `3dhp_ts4_head_turn` | The advisor keeps S0's correct heading where the head turns |

**Reading.** The numerical heading gain is real, but it does **not** become a useful animation change. The rig contract
cannot pose a lying torso, so the scenes the advisor fixes stay wrong for a different reason. W92's own rule ("a
reduction in numerical heading error must correspond to a useful change in actual animation") is therefore not met.

## 6. Decision

**Limited.** Keep `motion.vlm_heading_sign_advisor` as an opt-in, unwired experimental component.

**Supported scope:**
- lying torsos with a clearly visible chest-up or chest-down body;
- correcting the yaw-sign branch of the W91 heading;
- ABSTAIN everywhere else (≈ 85% of invoked frames).

**Not supported:**
- prone, crawling or push-up bodies (rare but harmful BACK mistakes);
- any claim of animation improvement while Root Orientation is yaw-only.

**What is missing.** Not visual evidence (the VLM sees chest-up lying reliably) but a **root/torso orientation
representation beyond yaw** (pitch/roll, or a full torso frame). Without it lying poses cannot be animated, whatever
the heading sign. That is the precondition for this advisor to matter in the rig.

## 7. Tests and files

**Tests:**
- `tests/test_framepose_vlm_heading_sign_advisor.py`, 5 tests:
  - answer parsing;
  - mirrored-heading geometry on a supine torso (s = n × b, KEEP / MIRROR, prone mirror);
  - abstention policy;
  - runtime-only invocation;
  - GT isolation and no production wiring.
- `tests/test_framepose_runtime_heading.py`: its no-production-wiring check now looks for `import`/`from` statements of
  the experimental heading modules instead of any textual mention. The old check tripped on a docstring.
- LabServer63: the heading, W89–W91A and production-adjacent suites (AnimationSemantics v2, FramePose bridge, Root
  Orientation, hold attribution), **70 passed**.

**New files:**

| file | role |
|---|---|
| `src/motion/vlm_heading_sign_advisor.py` | the advisor (opt-in, unwired) |
| `scripts/review_runtime_heading_side_failures.py` | first-gate review sheets |
| `scripts/run_vlm_heading_sign_advisor.py` | VLM inference (server) |
| `scripts/eval_vlm_heading_sign_advisor.py` | evaluation (local) |
| `scripts/run_vlm_heading_rig_replay_3dhp.py` | 3DHP rig replay |

**Changed (backward compatible):**
- `scripts/compose_runtime_heading_rig_video.py`: optional per-case labels, title and image paths; defaults unchanged.
- `tests/test_framepose_runtime_heading.py`.

**Artifacts:**
- Server `~/animcv-output/framepose_vlm_heading_advisor/`: `population.json`, `responses.jsonl` (2,793), `model.json`.
- Server `framepose_vlm_heading_advisor_archive/`: evaluation report `e728748b…`, rig report `b7a038f2…`, cases, Blender
  report, `.blend` files, videos, first-gate sheets.
- `framepose_runtime_heading_side_review/`: the review sheets.

## 8. Recommended next question (not started)

Should AnimCV's Root Orientation carry a **full torso frame** (yaw + pitch + roll from the GT-free S0 torso, with
reliability) instead of yaw only? Lying, crawling and bent poses are where heading fails *and* where a yaw-only rig
cannot animate. This advisor's chest-up/down evidence becomes useful only once that contract exists.
