# Worklog — Strong RGB VLM as H0 / Depth Anything Disagreement Resolver (2026-10-02)

## Starting state and files

- **Start:** fetched `origin/arch/single_frame_first` and confirmed local = remote = `5edfe78f12eb85a5dc5c3cdd3ca4a388ff89b872`, with a clean worktree.
- **Commits:**
  - `5449cb9`: pinned diagnostic VLM image, `containers/qwen3vl/Dockerfile`
  - `49d1499`: advisor module, inference script, analysis script, tests
  - `0186d67`: selector null controls added to the analysis
  - `db7bf88`: owner-case review renderer
  - the commit adding this worklog
- **Files added:**
  - `containers/qwen3vl/Dockerfile`
  - `src/framepose/vlm_depth_advisor.py`
  - `scripts/run_vlm_depth_advisor.py`
  - `scripts/analyze_vlm_depth_advisor.py`
  - `scripts/render_vlm_owner_review.py`
  - `tests/test_framepose_vlm_depth_advisor.py`
  - this worklog
- **Unchanged:** every existing file, including frozen H0, the Worklog 65 evidence cache, D0/D1, Worklog 66 R0/R1, the Geometry Observation Layer, FramePoseEstimator, SignState, AnimationSemantics, Root Orientation, FK and IK.

## Model identity and execution contract

| item | value |
|---|---|
| model | **Qwen/Qwen3-VL-8B-Instruct**, the single model; no size or candidate sweep |
| revision | `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b` (Apache-2.0); local snapshot `~/animcv-hf-cache/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd…` |
| weight/config manifest | 16 files, 17,545,915,883 bytes, each SHA-256 recorded; aggregate fingerprint `a0d72ded575eaa4460dad11dd1e313bc69f0403b2c97c3c1ba234af086952904` (safetensors shards `d5d0aef0…`, `8be88fb5…`, `83de00ea…`, `0a88b98e…`) |
| runtime | AnimCV-owned image `animcv-qwen3vl:cu124` from the pinned Dockerfile: torch 2.4.1+cu124, transformers 4.57.6, bitsandbytes 0.45.5, accelerate 1.10.1 |
| quantization | 4-bit bitsandbytes **NF4**, compute bfloat16, no double quantization |
| generation | greedy (`do_sample=False`, `num_beams=1`), `max_new_tokens=64`, fixed batch 4 with left padding |
| device | RTX 3080 Ti (12 GB); offline (`HF_HUB_OFFLINE=1`) |
| runtime cost | 16,086 inferences in 18,781 s (about 1.17 s each) |

**Model and image provenance.**
- The model was not cached before this batch. It was downloaded at the pinned revision; no substitute model was used.
- No existing image provided transformers ≥ 4.57: the AnimCV images have transformers 4.49 or none, and another project's image is not reused.
- A minimal AnimCV image was therefore pinned and built for this diagnostic.

**Determinism check.** Each shuffled-control answer is the donor's own real advisor image re-run in a different batch, and it matched the donor's real answer in **100%** of 7,027 cases. Inference is deterministic and batch-invariant.

## RGB, crop and prompt contract

- **Image:** the exact image bytes of each FrameBank sample, hashed with SHA-256 and decoded from those bytes.
- **Crop:** the FramePose crop contract (`crop_box` + `render_crop`: square box around the valid detector joints, 0.25 margin, black padding, bilinear). The advisor image is rendered at a fixed **448 px**, twice the 224 px model crop, declared once before evaluation and never tuned.
- **Overlay:** small yellow points labelled **S / E / W**, placed at the **benchmark detector's** left shoulder, elbow and wrist. Nothing else is drawn: no GT, H0, Depth Anything, correctness or confidence.
- **Records:** every response stores the image SHA, crop box, label pixels, rendered advisor-image SHA, model fingerprint and prompt SHA.

**Prompt.** A single fixed prompt (SHA-256 `d88c16d7…`) states that:
- S, E and W are detector labels on the same person's left arm;
- "closer" means physically nearer to the camera;
- perspective, overlap, foreshortening and occlusion may be used;
- UNCLEAR is allowed;
- the pairs are S–E, E–W and S–W, with the allowed values listed, and the answer must be exactly one JSON object.

It never mentions H0, Depth Anything, GT or AnimCV. There was no chain-of-thought, no few-shot examples and no second phrasing.

**Parser.** Strict. The answer must be exactly the three fields, each FIRST_CLOSER, SECOND_CLOSER or UNCLEAR. One outer Markdown fence may be removed (recorded), but none occurred. Anything else becomes UNKNOWN for all fields.

**Result:** 16,086 of 16,086 responses were valid. The model never answered UNCLEAR.

## Populations

- **Primary (runtime only):** a held-out eligible left-arm row (validation + test, the Worklog 64–66 eligibility) where, for at least one segment, sign(frozen H0 f) and sign(Worklog 65 Depth Anything relation) are both non-zero and opposite.
  - 7,027 of the 9,059 rows qualify.
  - Per segment: upper 3,999, forearm 3,820, chain 3,859.
  - GT was not consulted for selection.
- **Secondary:** all 9,059 eligible held-out rows.
- **Scoring:** GT ordering is used only for scoring. A positive GT f means the distal point is farther (FIRST_CLOSER is correct). Stable rows have \|GT f\| > sin 10°, the historical evaluation threshold, which is never shown to the VLM.

## Raw VLM ordering results

Coverage is 100% everywhere, because the model is never UNCLEAR.

| segment | population | state distribution | stable accuracy | balanced accuracy | always-H0 | always-DA |
|---|---|---|---:|---:|---:|---:|
| S–E (upper) | segment disagreement (3,999; 2,322 stable) | FIRST 3,446 / SECOND 553 | 0.688 | 0.560 | 0.661 | 0.339 |
| | all rows (9,059) | FIRST 7,897 / SECOND 1,162 | 0.697 | 0.585 | 0.804 | 0.674 |
| E–W (forearm) | segment disagreement (3,820; 2,302 stable) | **SECOND 3,820 (constant)** | 0.519 | **0.500** | 0.558 | 0.442 |
| | all rows | SECOND 9,058 / FIRST 1 | 0.555 | 0.500 | 0.762 | 0.714 |
| S–W (chain) | segment disagreement (3,859; 2,126 stable) | FIRST 744 / SECOND 3,115 | 0.554 | 0.545 | 0.544 | 0.456 |
| | all rows | FIRST 1,735 / SECOND 7,324 | 0.566 | 0.578 | 0.773 | 0.739 |

- **Forearm:** the E–W answer is effectively constant ("wrist closer"), so it carries no information.
- **Upper and chain:** answers vary, but they are dominated by one state (87% FIRST for S–E, 81% SECOND for S–W). Balanced accuracy is only 0.55–0.59.
- **Against fixed selectors on disagreement rows:** the VLM is slightly above always-H0 for upper (+2.7 points) and chain (+1.0 point), and below always-H0 for the forearm (−3.9 points).
- **On all rows:** the VLM is far below both H0 (0.77–0.80) and Depth Anything (0.67–0.74).

## Shuffled-RGB grounding control

The 7,027 primary rows were paired. Each received a donor crop from a different sequence, chosen by a deterministic half-way rotation, with the donor's own S/E/W overlay.

| field | real vs shuffled change rate | change rate expected for two independent images from the same answer distribution | valid (real / shuffled) | state distribution |
|---|---:|---:|---|---|
| S–E | 0.244 | 0.230 | 1.0 / 1.0 | identical (FIRST 6,092 / SECOND 935) |
| E–W | **0.000** | 0.000 | 1.0 / 1.0 | identical (constant) |
| S–W | 0.300 | 0.302 | 1.0 / 1.0 | identical (FIRST 1,305 / SECOND 5,722) |

- **Why the distributions match:** a donor crop *is* another frame's real advisor image, so the distributions are identical by construction.
- **What the change rates show:** the per-frame answers for S–E and S–W do depend on the image, at about the rate expected from their marginal distributions. E–W does not.
- **What they do not show:** image dependence alone does not establish that the dependence is *correct*. The selector controls below test that.

## H0-vs-Depth-Anything complementarity (held-out stable rows)

Each cell is the share of rows where the VLM agrees with H0 or with Depth Anything, then V0's mean \|f\| error.

| segment | H0 wrong / DA right | H0 right / DA wrong | both right | both wrong |
|---|---|---|---|---|
| chain | 970 rows; agrees with DA **0.599**; V0 .396 (H0 .659, D1 .277, R1 .596) | 1,156 rows; agrees with H0 **0.516**; V0 .214 (H0 .182, D1 .308) | 3,241 rows; VLM correct 0.564; V0 .214 (H0 .205) | 324 rows; V0 .519 |
| forearm | 1,017 rows; agrees with DA 0.421; V0 .489 (H0 .679, D1 .356) | 1,286 rows; agrees with H0 0.597; V0 .226 (H0 .199) | 3,343 rows; V0 .217 (H0 .207) | 431 rows; V0 .601 |
| upper | 787 rows; agrees with DA 0.498; V0 .488 (H0 .584, D1 .336) | 1,535 rows; agrees with H0 0.785; V0 .159 (H0 .147) | 3,152 rows; V0 .146 (H0 .141) | 349 rows; V0 .671 |

**Does the VLM tell the two disagreement categories apart?**

| segment | P(agree with DA \| DA right) | P(agree with DA \| DA wrong) | gap | note |
|---|---:|---:|---:|---|
| chain | 0.60 | 0.48 | about 0.12 | weak |
| forearm | 0.42 | 0.40 | none | constant answer |
| upper | 0.50 | 0.22 | 0.28 | largely the always-FIRST prior meeting differently signed GT populations, per the controls below |

**Selector null controls** (primary rows where H0 and D1 differ in sign; mean \|f\| error):

| segment | conflict rows | always H0 | **always D1** | V0 (real VLM votes) | V0 with **shuffled** votes | random mixture at V0's D1 rate | oracle branch | V0 picks D1 when D1 better / when H0 better |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| upper | 1,304 | .341 | **.227** | .286 | .304 | .295 | .146 | 0.46 / 0.30 |
| forearm | 2,312 | .341 | **.264** | .270 | .270 (identical) | .300 | .151 | 0.55 / 0.51 |
| chain | 2,248 | .351 | **.239** | .252 | .259 | .295 | .151 | 0.54 / 0.44 |

What the controls show:
- **D1 is the better branch** on 58–62% of conflict rows. Always taking D1 beats V0 on every segment.
- **Shuffled votes do almost as well as real ones.** With votes from *another sequence's* image, V0 loses only 0.007 (chain) and 0.018 (upper), and nothing on the forearm.
- **Cause:** nearly all of V0's advantage over a random mixture comes from the VLM's fixed answer prior, not from frame-specific visual evidence.
- **Image-specific trust resolution is small:** chain 0.54 vs 0.44, upper 0.46 vs 0.30, forearm about zero.

## V0 selector results (references: H0, Worklog 65 D1, Worklog 66 R1)

**Held-out test, 6,003 rows** (mean / p50 / p95):

| segment | metric | H0 | D1 | R1 | V0 |
|---|---|---|---|---|---|
| upper | \|f\| | .237/.168/.661 | **.209**/.159/.560 | .228/.160/.627 | .231/.163/.650 |
| | full (°) | 15.7/11.4/40.1 | **13.9**/10.8/35.5 | 15.1/10.9/38.6 | 15.3/11.0/39.7 |
| forearm | \|f\| | .283/.203/.848 | .265/.204/.781 | .282/.202/.833 | **.263**/.201/.743 |
| | full (°) | 20.4/14.8/59.7 | 19.4/14.7/53.2 | 20.5/14.9/60.1 | **19.1**/14.7/52.4 |
| chain | \|f\| | .271/.192/.810 | **.224**/.173/.612 | .253/.183/.746 | .238/.184/.636 |
| | full (°) | 17.9/12.6/52.1 | **15.2**/11.6/39.8 | 16.9/12.2/48.3 | 16.0/12.3/41.3 |

**Paired comparisons on test:**
- V0 − H0: chain −0.032 \|f\|, −1.97° elevation (improved on 10% of rows, worse on 5%); upper −0.006; forearm −0.019.
- V0 − D1: chain +0.014, upper +0.022, forearm −0.002.

**Scope comparison:**

| scope | chain V0 − H0 \|f\| | upper V0 − H0 \|f\| | forearm V0 − H0 \|f\| | V0 − D1 |
|---|---|---|---|---|
| **clean dancing + hug** (425) | **+0.004** | **+0.007** | **+0.006** | −0.077 / −0.016 / −0.056 |
| crosscountry (133) | −0.017 | −0.011 | −0.001 | — |
| all other held-out (8,501) | −0.025 | −0.008 | −0.018 | +0.009 / +0.021 / −0.001 |

- **Clean subset:** V0 is slightly worse than H0 everywhere (elevation +0.25° to +0.40°), though far better than D1 there.
- **GT-depth bins (test, \|GT f\| ≥ 0.5):**

| segment | H0 | D1 | R1 | V0 |
|---|---:|---:|---:|---:|
| chain | .470 | .359 | .440 | .362 |
| forearm | .363 | .342 | .369 | .309 |

- **Per sequence (53):** V0 beats H0 on 33 / 30 / 34 (chain / forearm / upper) and beats D1 on only 21 / 19 / 24.

## Owner cases

Review panels with the RGB crop, S/E/W overlay and ordering table are copied locally to `~/animcv-output/67_framepose_vlm_depth_advisor/`. The server copy is `~/animcv-output/framepose_vlm_depth_advisor/owner_review/`.

| case | primary? | VLM (S–E / E–W / S–W) | outcome |
|---|---|---|---|
| known_good #424 | no | S closer / W closer / S closer | H0 kept everywhere (same sign) |
| tracking_loss #312 | no | S / W / W | H0 kept; the forearm vote agreed with H0, so the Worklog 65 D1 collapse is avoided, but it is not corrected toward GT |
| largest_ratio_error #397 | yes | S / W / W | **upper wrongly switched to D1** (VLM "S closer"; GT has the elbow closer): full angle 39.6° → 95.2°. Forearm and chain kept H0 |
| largest_direction_error = largest_2d #176 | yes | S / W / W | **upper correctly switched to D1** (92.3° → 22.5°). Chain kept H0's wrong sign (VLM "W closer", GT "S closer"): 129.3°. This frame is also a detector failure (Worklog 63) |
| dynamic #462 | no | E / W / W | H0 kept (same sign); D1 would have helped (27.7° → 20.7°) |
| turning #211 | yes | S / W / W | VLM agreed with H0 against a wrong D1: H0 kept (40.6° / 45.4°), correct resolution |
| near_straight #525 | no | S / W / W | H0 kept |
| strongly_bent = largest_reach = largest_depth #173 | yes | E / W / W | upper: VLM "elbow closer" sided with H0's wrong sign (92.2°, while D1 gives 32.9°); chain likewise (97.0° vs D1 65.0°) |

No negative case was removed. The E–W answer is "W closer" in every case.

## Architecture classification

**CASE C: no meaningful image-grounded trust signal from this advisor.** The advisor role is rejected.

- **Formally:** outputs are valid and, for two of three fields, image-dependent at the rate their marginals imply. But E–W is constant, and every field is dominated by a single state.
- **On disagreement rows:** ordering performance does not meaningfully exceed the fixed H0 baseline. It gains +1 to +3 points for S–E and S–W, loses 4 points for E–W, and balanced accuracy is 0.50–0.56.
- **The decisive control:** a V0 selector driven by votes from a *different sequence's* image almost matches V0 with the real image (chain .259 vs .252, forearm identical). Simply taking D1 on every H0/D1 conflict beats V0 on every segment.
- **What this means:** the VLM does **not** observe the visual cue that distinguishes "H0 is right" from "Depth Anything is right". The apparent V0 gains over H0 (test chain −1.9°) come from D1 usually being the better branch on conflict rows, filtered through the VLM's answer prior. They do not come from frame-specific RGB evidence.
- **Clean subset:** V0 also slightly degrades it relative to H0.
- **Not CASE B:** the CASE B gate ("image-grounded and above trivial baselines") is not met in substance.

**Scope of the negative result.**
- It covers only **Qwen3-VL-8B-Instruct (4-bit NF4, greedy) with this one fixed advisor contract**: a 448 px crop, detector S/E/W labels and three pairwise questions.
- It does **not** show that RGB appearance lacks near/far evidence, that learned visual features cannot resolve the conflict, or that all VLMs are unsuitable. Possible contract limitations (small labels on a person-centric crop, quantization) were deliberately not tuned.
- RGB/Vision therefore remains an open architecture family.

**What was not done:** the VLM was not integrated into production, no prompts were tuned, and nothing proceeds to yaw, temporal context or retargeting.

## Outputs and tests

**Outputs** (LabServer63 `~/animcv-output/framepose_vlm_depth_advisor/`):
- `model_manifest.json`
- `population.json`
- `responses.jsonl` (16,086 lines, SHA-256 `ec8e7d47…`)
- `analysis/analysis.json` (SHA-256 `e4f0c669dda784db21420911a3f0d8f6cdfea35430bac78f138446a9720f1b74`)
- `owner_images/` and `owner_review/` PNGs

Only the review PNGs are copied locally.

**Tests:** `tests/test_framepose_vlm_depth_advisor.py` has 8 tests covering:
- deterministic crop identity and detector S/E/W overlay positions
- the strict parser, with malformed output → UNKNOWN and fence normalization recorded
- pinned model provenance; the prompt mentions no H0, Depth Anything or GT
- canonical near/far mapping
- runtime-only disagreement selection (no GT)
- V0 rules: UNKNOWN → H0, and same-sign H0/D1 → H0
- deterministic cross-sequence shuffled pairing
- GT never enters the VLM input, and no production import

Results:
- 8/8 passed in the `animcv-qwen3vl:cu124` image.
- 8/8 passed locally, where only PIL/numpy parts run.

The Worklog 65 and 66 test files are unchanged. No production file changed, so the full regression was not run.
