# Worklog — Learned Near/Far Sensor over Frozen Qwen3-VL Visual Tokens (2026-10-03)

## Starting state and files

- **Start:** fetched `origin/arch/single_frame_first` and confirmed local = remote = `a3a396f4a4d68b70baeda44c5e55759e58a41713`, with a clean worktree.
- **Commits:**
  - `3bc71bf`: sensor module and visual-feature extractor
  - `8c42e69`: G0/G1 training/evaluation script and tests
  - the review-render commit
  - the commit adding this worklog
- **Files added:**
  - `src/framepose/learned_vision_sensor.py`
  - `scripts/extract_qwen3vl_visual_features.py`
  - `scripts/run_learned_vision_sensor.py`
  - `scripts/render_learned_vision_owner_review.py`
  - `tests/test_framepose_learned_vision_sensor.py`
  - this worklog
- **Unchanged:** every historical module and checkpoint, including frozen H0, the Worklog 64 probe, Worklog 65 evidence and D0/D1, Worklog 66 R0/R1, the Worklog 67 advisor and V0, historical F0/F1/F2, the Geometry Observation Layer, FramePoseEstimator, AnimationSemantics, Root Orientation, FK and IK.

## Model and exact visual-feature extraction point

- **Model:** the same pinned Worklog 67 artifact.
  - Qwen/Qwen3-VL-8B-Instruct @ `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`
  - aggregate fingerprint `a0d72ded575eaa4460dad11dd1e313bc69f0403b2c97c3c1ba234af086952904`, verified against the Worklog 67 manifest
- **Runtime:** the same image, `animcv-qwen3vl:cu124` (torch 2.4.1+cu124, transformers 4.57.6), and the same 4-bit NF4 load as Worklog 67. The vision tower is quantized too: `merger.linear_fc2` is `Linear4bit`. This is the exact representation the Worklog 67 decoder read.
- **No generation:** no text is generated and no chat template is used. There was no fine-tuning and no size or backbone comparison.

**Extraction point.** It was chosen by reading the pinned implementation, not by accuracy:
- **Tensor:** `Qwen3VLModel.get_image_features(pixel_values, image_grid_thw)` → `self.visual(...)` → the first return value. This is the vision-tower **patch-merger output**, the image embeddings placed into the language model's image-token slots before any decoding. The deepstack side features are not used.
- **Shape:** the Qwen2-VL image processor (`Qwen2VLImageProcessorFast`) keeps the 448 px crop at 448. That gives `image_grid_thw = [1, 28, 28]` and pixel values of 784 × 1536 per image. The result is 196 tokens × **4096** channels, bf16 compute, cached as float16.
- **Token order:** the processor reshapes patches as (t, h/2, w/2, 2, 2) (`transpose(0,3,6,4,7,2,1,5,8)`), and the merger folds each 2×2 group (`view(-1, 4·hidden)`). The tokens therefore form a **regular row-major 14×14 grid**, each covering 32×32 crop pixels.
- **Readout choice:** because the grid is regular, the readout is deterministic **bilinear sampling** at the detector joints. No cross-attention alternative was built.

## Vision input and cache identity

- **Input:** the FramePose person crop (`crop_box` + `render_crop`, rendered at 448 px) from the exact image bytes. There is **no text or S/E/W overlay**.
- **Location:** `~/animcv-output/framepose_qwen3vl_visual_cache/` on LabServer63.
- **Contents:**
  - full grids `token_grid_fp16.npy`, (21,817, 196, 4096) float16, 35.0 GB, immutable
  - `readout.npz` (SHA-256 `f5d0eb9c…`), holding per sample:
    - bilinear samples at the detector left shoulder, elbow and wrist (token centres at 32·(i+0.5) px, clamped)
    - one global mean token
    - joint validity
    - image SHA, crop SHA and crop box
- **Identity digest:** `1fce1ab2…`. It covers the bank digest, sample-ID order, crop contract, render resolution, overlay = none, model repository/revision/fingerprint, quantization, processor class and grid, feature source, token order, dtype, torch and transformers.
- **Run:** 2,019 s for 21,817 samples.
- **Determinism:** re-extracting the first batch reproduced the cache exactly (maximum difference 0.0 in fp16).
- **No GT** is read during extraction.
- The backbone is never recomputed during training.

## Sensor architecture and G0/G1 identity

**Per pair** (S–E, E–W, S–W), the inputs are:
- runtime pair geometry from the H0 crop-normalized geometry tensor, 13 features: both endpoints' x, y, confidence and validity, plus dx, dy, length, cos and sin;
- the visual readout at endpoint A, at endpoint B, and the global token.

**Graph:**
- one shared `Linear(4096 → 64)` visual projection;
- one per-pair head of the same architecture, `Linear(13 + 3·64 → 64) → GELU → Linear(64 → 3)`;
- output classes FIRST_CLOSER / SECOND_CLOSER / UNCLEAR.

No H0, Depth Anything, GT, oracle sign or temporal input is used.

**The two candidates:**

| candidate | geometry | visual features |
|---|---|---|
| G0_ZERO_VISION | real | every frozen visual feature exactly zero |
| G1_QWEN_VISION | identical | the cached readout |

- Both have the same graph and **302,345** parameters, with identical initialization from seed 1337.
- The script verifies that G0's inputs are all zero and that G1's equal the cache.

**Targets:**
- FIRST_CLOSER if GT f > sin 10°, SECOND_CLOSER if GT f < −sin 10°, otherwise UNCLEAR. This is the Worklog 67 mapping, and the threshold is a target definition only.
- Masked by GT validity and detector validity of both endpoints.
- Train class counts (FIRST / SECOND / UNCLEAR):
  - S–E: 5,089 / 2,564 / 3,189
  - E–W: 3,417 / 2,852 / 4,087
  - S–W: 3,699 / 2,482 / 4,230
- This balance is reasonable, so no class weighting was used.

**Training contract** (declared before results):
- unweighted masked 3-class cross-entropy
- AdamW, learning rate 3e-4 per-step cosine to 1e-5, weight decay 1e-4
- 200 epochs, batch 256, seed 1337, fp32
- the exact FrameBank splits
- selection on validation cross-entropy every 10 epochs; test never used

**Selection:**

| candidate | selected epoch | validation CE | note |
|---|---:|---:|---|
| G0 | 119 | 0.988 | |
| G1 | **9** | **0.818** | validation CE rises afterwards (1.10 at epoch 29, 1.26 at epoch 79) while train CE falls to 0.16 |

G1 overfits the 3DPW training scenes quickly, as expected for 4096-dimensional frozen features on 11k frames.

**Worklog 66 confound avoided.** Frozen H0 is neither a training input nor a target. Supervision is GT ordering only, so H0's in-sample train error (0.08 vs 0.26 held-out) cannot teach this sensor to trust H0.

## Held-out raw sensor results

**Eligible rows:** the Worklog 64–67 held-out rows, 6,003 test and 3,056 validation.

**Test:**

| pair | metric | G0 | **G1** | G1, shuffled vision |
|---|---|---:|---:|---:|
| S–E (upper) | accuracy / balanced / macro F1 | .516 / .439 / .436 | .523 / .495 / .486 | .339 / .297 / .293 |
| | stable accuracy / stable FIRST-vs-SECOND balanced | .611 / .479 | **.628 / .568** | .403 / .328 |
| | prediction distribution F/S/U | 4,055 / 399 / 1,549 | 3,276 / 1,154 / 1,573 | — |
| E–W (forearm) | accuracy / balanced / macro F1 | .608 / .621 / .608 | **.658 / .653 / .652** | .349 / .349 / .346 |
| | stable accuracy / stable balanced | .532 / .540 | **.655 / .648** | .327 / .322 |
| | prediction distribution F/S/U | 1,829 / 1,280 / 2,894 | 1,632 / 2,293 / 2,078 | — |
| S–W (chain) | accuracy / balanced / macro F1 | .567 / .560 / .561 | .563 / .565 / .565 | .327 / .326 / .324 |
| | stable accuracy / stable balanced | .454 / .453 | **.568 / .571** | .292 / .294 |

**UNCLEAR on test** (precision / recall):

| pair | G0 | G1 |
|---|---|---|
| E–W | .49 / .78 | .58 / .66 |
| S–W | .50 / .77 | .53 / .55 |
| S–E | .53 / .36 | .51 / .35 |

**G1 confusion on test** (rows = GT F/S/U, columns = prediction F/S/U):

| pair | GT F | GT S | GT U |
|---|---|---|---|
| E–W | 1,090 / 297 / 513 | 267 / 1,663 / 371 | 275 / 333 / 1,194 |
| S–E | 1,892 / 262 / 535 | 351 / 450 / 242 | 1,033 / 442 / 796 |

**Validation** (stable accuracy, G0 → G1):

| pair | G0 → G1 |
|---|---|
| S–E | .594 → **.735** |
| E–W | .454 → **.645** |
| S–W | .385 → **.629** |

Stable-row pairing on test, comparing G1 and G0 (rows where only one of the two is right):

| pair | G1 right, G0 wrong | G0 right, G1 wrong |
|---|---:|---:|
| E–W | 861 | 344 |
| S–W | 962 | 518 |
| S–E | 568 | 506 |

**Per sequence (53):**

| pair | G1 better than G0 | largest single-sequence share of net gain |
|---|---:|---|
| E–W | 42 | 12% |
| S–W | 41 | 11% |
| S–E | 35 | **51%** (`outdoors_parcours_00`), so the upper-arm gain is owned by one sequence |

All three pairs, **including the forearm**, give non-degenerate predictions spread over all three classes. This contrasts with Worklog 67, where E–W was constant.

## Shuffled-vision control (no retraining)

G1 was given the visual readout of a deterministic donor row from a **different held-out sequence**, with the original geometry kept.

- **Accuracy collapses:** test stable accuracy is .29–.40, which is **below G0** (shuffled vs G0: .403 vs .611, .327 vs .532, .292 vs .454).
- **State changes:** real vs shuffled changes the state on 59–65% of rows.
- **Paired stable outcomes** (real right / shuffled right):

| pair | real right, shuffled wrong | shuffled right, real wrong |
|---|---:|---:|
| E–W | 1,799 | 419 |
| S–E | 1,272 | 434 |
| S–W | 1,497 | 421 |

The sensor depends strongly on the actual image, and its gains over G0 are **visual information, not geometry or head priors**.

## Historical references (read-only, same rows, stable accuracy / balanced)

| test pair | G1 | frozen H0 | raw Depth Anything | Worklog 65 D1 | Worklog 67 language VLM |
|---|---|---|---|---|---|
| S–E | .628 / .568 | .779 / .693 | .621 / .645 | .797 / .736 | .704 / .536 |
| E–W | .655 / .648 | .757 / .754 | .720 / .722 | .787 / .785 | .547 / .500 |
| S–W | .568 / .571 | .751 / .743 | .741 / .742 | .836 / .832 | .526 / .553 |

- **Against Worklog 67:** the learned sensor avoids the language advisor's strong categorical priors. Its balanced accuracy exceeds the language VLM's on E–W (.648 vs .500) and on S–W.
- **Against the geometry estimates:** it is still **well below** H0, D1 and Depth Anything as a stand-alone orderer.

## Conflict-selector null controls (held-out rows where H0 and D1 have opposite signs)

Each cell is mean \|f\| error; the last column gives P(pick D1 \| D1 better) / P(pick D1 \| H0 better).

| segment | rows | always H0 | always D1 | V0 (language) | **V1 (learned vision)** | V1, shuffled vision | oracle | V0 pick rates | **V1 pick rates** | V1-shuffled pick rates |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| upper | 1,638 | .313 | **.234** | .273 | .268 | .311 | .143 | .48 / .35 | .30 / .19 | .27 / .38 |
| forearm | 2,681 | .328 | .275 | .270 | **.257** | .312 | .153 | .55 / .51 | **.35 / .13** | .32 / .33 |
| chain | 2,541 | .336 | **.247** | .252 | .277 | .318 | .150 | .54 / .43 | **.38 / .19** | .33 / .36 |

**What the controls show:**
- **V1 discriminates better than V0 on the forearm and chain.** The gap between picking D1 when it is right and when it is wrong is 0.22 and 0.20, against V0's 0.04 and 0.10. With shuffled vision the gap vanishes or reverses, so the discrimination is image-based.
- **On the upper arm** V1's gap (0.11) is no better than V0's (0.13).
- **V1 is conservative.** It recovers only 30–38% of D1-better rows, because UNCLEAR defaults to H0. So it beats always-D1 only on the forearm (.257 vs .275), and stays behind always-D1 on chain and upper.
- **Clean subset conflict rows,** where D1 is usually worse: V1 is best of all fixed selectors on every segment (chain .203 vs H0 .218 and D1 .428) and almost never picks D1 when H0 is better (0.0–0.01).

## Continuous results (V1 vs H0 / D1 / V0)

Each cell is the mean difference in full angle (V1 minus the reference) on the observed 2D plane:

| scope | segment | V1 − H0 | V1 − V0 | V1 − D1 |
|---|---|---:|---:|---:|
| test | upper | −0.39° | −0.03° | +1.37° |
| | forearm | **−1.43°** | **−0.20°** | **−0.43°** |
| | chain | −1.11° | **+0.78°** | +1.55° |
| **clean dancing + hug** | upper | **−0.34°** | −0.72° | −1.56° |
| | forearm | **−0.27°** | −0.54° | −3.89° |
| | chain | **−0.15°** | −0.40° | −4.96° |
| crosscountry | upper / forearm / chain | −0.58° / −1.36° / −0.82° | +0.14° / −1.23° / +0.11° | — |
| all other held-out | upper / forearm / chain | −0.48° / −1.31° / −0.99° | −0.03° / −0.16° / +0.48° | — |

- **Clean subset:** V1 preserves or slightly improves H0 on every segment. This contrasts with V0 and D1, which both degraded it.
- **Test GT-depth bins** (\|GT f\| ≥ 0.5), \|f\| error:

| segment | H0 | D1 | V0 | V1 |
|---|---:|---:|---:|---:|
| chain | .470 | .359 | .362 | .423 |
| forearm | .363 | .342 | .309 | .311 |
| upper | .391 | .308 | .381 | .365 |

Near the image plane, V1 is about equal to H0.

## Owner cases

Review panels (encoder crop with review-only S/E/W dots and an ordering table) are copied locally to `~/animcv-output/68_framepose_learned_vision_sensor/`. The server copy is `~/animcv-output/framepose_learned_vision_sensor/owner_review/`.

Full angles are listed as H0 / D1 / V0 / V1.

| case | G1 (S–E / E–W / S–W) | V1 effect |
|---|---|---|
| known_good #424 | F / U / U | H0 kept (same signs) |
| tracking_loss #312 | U / U / U | H0 kept |
| largest_ratio_error #397 | **S** / S / S | upper: G1 correctly says the elbow is closer, so V1 keeps H0 (39.6° / 95.2° / 95.2° / **39.6°**; V0 had wrongly taken D1). Forearm G1 is wrong, but H0 is kept anyway |
| largest_direction_error #176 | **S (wrong)** / F / S (wrong) | upper keeps H0's wrong sign (92.3° / 22.5° / 22.5° / **92.3°**); V0 had been right here. The frame is also a detector failure |
| dynamic #462 | F / U / F | H0 kept everywhere (same sign or UNCLEAR) |
| turning #211 | U / U / **F (wrong)** | chain wrongly takes D1 (45.4° / 90.8° / 45.4° / **90.8°**) |
| near_straight #525 | F / U / U | H0 kept |
| strongly_bent #173 | **F** / F / **F** | upper and chain correctly take D1 (92.2° → **32.9°**, 97.0° → **65.0°**); V0 had kept H0 |

No failures were replaced. V1 corrects V0 on #397 and #173, and is worse than V0 on #176 and #211.

## Architecture classification

**CASE B: the visual representation has transferable signal, but the selector does not materially resolve the trust conflict.**

| CASE A requirement | result |
|---|---|
| G1 materially beats G0 on held-out stable ordering | **Yes for the forearm and chain** (test +12 and +11 points stable accuracy; validation +19 and +24). The upper arm gains only +1.7 points on test (+14 on validation), and that gain is owned 51% by one sequence |
| G1 real materially beats G1 shuffled | **Yes:** shuffled falls below G0 |
| all three pairs non-degenerate | **Yes,** including the forearm |
| gains not owned by one sequence | **Yes for forearm and chain** (11–12%); **no for the upper arm** |
| V1 distinguishes conflicts better than V0 | **Yes for forearm and chain**, no for the upper arm |
| V1 improves branch selection without a clean regression | **Mixed:** no clean regression (V1 improves H0 on clean); better than V0 on the forearm; **worse than V0 and always-D1 on the chain**; no better than always-D1 on the upper arm |

The last row is decisive, so the result is CASE B.

**Answer to the completion question:**
- **Yes:** Qwen3-VL's frozen visual representation contains transferable, image-grounded left-arm near/far information that a small task-specific learned sensor recovers. This is clearest for the forearm and the shoulder→wrist chain, the very relation the language advisor answered with a constant.
- **Worklog 67 was a readout problem:** its failure was primarily a language/advisor readout failure, not an absence of the cue in the visual representation.
- **The signal is weak:** it overfits 3DPW scenes quickly (best epoch 9). As a stand-alone orderer it stays far below H0 and D1, and as an arbiter it improves trust discrimination but does not yet beat a simple always-D1 rule on two of three segments.

**Conclusions:**
- **Learned RGB/vision evidence is preserved** as a valid observation sensor and as the next FramePose architecture direction.
- **Trust fusion stays unresolved.** Prompt engineering is not reopened, and production is not touched.

This is not a contradiction of historical F2. F2 tested frozen VL patch tokens → full 17-joint XYZ. This batch tests the same kind of frozen representation → three discrete arm orderings, which is a different task. Full visual-to-XYZ fusion is not reopened.

## Outputs and tests

**Outputs** (LabServer63):
- `~/animcv-output/framepose_qwen3vl_visual_cache/`: grid, readout, manifest
- `~/animcv-output/framepose_learned_vision_sensor/`:
  - `report.json` (SHA-256 `aa1cbbee73027126d780fc14f64778110d618abf93e9edf3889eec6be5ba14a3`)
  - `G0_ZERO_VISION.pt`, `G1_QWEN_VISION.pt`
  - `rows_heldout.json`
  - `owner_review/`

Only the review PNGs are copied locally.

**Tests:** `tests/test_framepose_learned_vision_sensor.py` has 10 tests covering:
- pinned model identity and the feature source, with no generation or chat template in the extractor
- detector geometry → token-grid mapping and row-major bilinear sampling with clamping
- deterministic readout order, validity and global mean
- the fixed pair target mapping
- runtime-only pair geometry
- G0 exactly zero, and G1 exactly the cache
- identical graph and initialization, with deterministic inference
- no H0, Depth Anything, VLM or D1 in the sensor training call, and no GT in extraction
- V1 selector rules: UNCLEAR → H0, same sign → H0
- cross-sequence shuffled pairing
- no production import

Results:
- 16 passed, 1 skipped locally (torch absent), together with the unchanged Worklog 67 tests.
- The torch test also ran in the training image.

No production file changed, so the full regression was not run.
