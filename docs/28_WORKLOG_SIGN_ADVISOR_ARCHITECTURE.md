# Worklog — VLM Sign Advisor Architecture (2026-09-06)

> A new architecture hypothesis on `arch/single_frame_first`, distinct from the
> historical dense visual-fusion experiment. The Frame Pose Core keeps
> continuous 3D reconstruction; a separate advisor owns only discrete
> orientation branches.
>
> Evaluation regime: `benchmark_detector_observation` throughout (3DPW's shipped
> detector keypoints). Historical F0/F1/F2 are untouched.

## 1. The Sign Contract

Every field is a *reduction of a quantity AnimCV already uses* to define an
orientation or hinge failure. Nothing here was invented because it is easy to
prompt. All are read in the canonical camera frame (+X right, +Y forward/depth,
+Z up), so each is a near/far or facing question about the image — the evidence a
2D-joint-only observation discards.

| field | canonical quantity | +1 | −1 | degenerate when | joints | historical metric |
| --- | --- | --- | --- | --- | --- | --- |
| `torso_facing` | `sign(f_y)`, `f = normalize(cross(thorax−pelvis, R_shoulder−L_shoulder))` | faces away from camera | faces camera | `\|f_y\| < 0.1` (near-profile) or torso invalid | torso chain + shoulders + hips | `root_yaw_error_degrees` 180° branch; docs/12–13 yaw tail |
| `shoulder_forward_depth` | `sign(D_sh)`, `D = (y_R − y_L)/√2` | right shoulder farther | right nearer | `\|D\| < 0.01 m` | both shoulders | `shoulder_forward_depth_sign_disagreement(_stable)` (docs/18, 21) |
| `hip_forward_depth` | same on the hip pair | right hip farther | right nearer | `\|D\| < 0.01 m` | both hips | `hip_forward_depth_sign_disagreement(_stable)` |
| `left_elbow_forward_bend` | `sign` of `canonical_pose.bend_direction(elbow; shoulder, wrist)`'s `+Y` | elbow farther than the shoulder–wrist line | nearer | offset < 0.02 m, `\|dir_y\| < 0.1`, or chain invalid | left elbow | `hinge_errors.flipped` / `hinge_direction_mae_degrees` |
| `right_elbow_forward_bend` | same | | | | right elbow | same |
| `left_knee_forward_bend` | `bend_direction(knee; hip, ankle)`'s `+Y` | | | | left knee | same |
| `right_knee_forward_bend` | same | | | | right knee | same |

Values are `+1 / −1 / 0`. **`0` is degenerate-or-unobservable and is never
guessed.** The forward-depth floor is the evaluator's own
`STABLE_FORWARD_DEPTH_M = 0.01`, reused verbatim, so a field and the historical
metric it corresponds to degenerate together.

`torso_facing` and `shoulder_forward_depth` are deliberately complementary: at
frontal `D_sh ≈ 0` while `f_y ≈ ±1`, and at exact profile `f_y ≈ 0` while
`D_sh` is large. Each is strong where the other degenerates, which is verified
by test.

### Oracle sign distribution over the 21,817-frame bank

| field | +1 | −1 | degenerate |
| --- | ---: | ---: | ---: |
| `torso_facing` | 5,902 | 14,633 | 1,282 |
| `shoulder_forward_depth` | 13,236 | 7,634 | 947 |
| `hip_forward_depth` | 12,699 | 7,341 | 1,777 |
| `left_elbow_forward_bend` | 10,902 | 6,789 | 4,126 |
| `right_elbow_forward_bend` | 12,244 | 3,944 | 5,629 |
| `left_knee_forward_bend` | 3,885 | 15,415 | 2,517 |
| `right_knee_forward_bend` | 5,724 | 12,860 | 3,233 |

No field is degenerate on more than 26% of frames, and none is so unbalanced
that a constant answer would look competent.

## 2. Position and sign ownership are separated

| Owner | Owns |
| --- | --- |
| Frame Pose Core | continuous XYZ, joint distances, bone lengths, metric depth magnitude |
| Sign Advisor | discrete orientation branch, discrete bilateral near/far sign, discrete hinge bend orientation |

The contract records its own exclusions (`excluded_by_contract`: XYZ, depth
magnitude, metric offsets, bone lengths, continuous embeddings, image patch
tokens), and a test asserts them. No dense visual token enters this path, and
the F1/F2 patch-token cache is not reused as sign evidence.

## 3. Oracle sign derivation

`signs.oracle_sign_states(target_3d, target_valid)` applies the contract to
ground-truth 3D only. No RGB is involved. It is an **architecture control, not a
production mechanism**, and every run records
`sign_source = "oracle"` plus an explicit
`oracle_is_an_architecture_control_not_a_production_mechanism` flag.

## 4. Sign-conditioned architecture

`FramePoseEstimator(geometry, image_tokens=None, sign_state=None)`.

One learned embedding per (field, value) — `7 × 3 × 256` — routed to only the
joints that field governs via a fixed `(17, 7)` mask, then added to those joint
queries before fusion. Nothing else changes.

```
trainable parameters   1,652,227  unconditioned
                       1,657,603  sign-conditioned   (+5,376 exactly)
```

Width is inherited, not swept. A test asserts each of the seven fields
measurably moves the prediction, so conditioning cannot be silently inert.

## 5. Neutral control

`S0` keeps the conditioning path enabled and feeds **every field `UNKNOWN` on
every frame**. Defined before any result was seen and never tuned on an outcome.
S0 and S1 therefore share one graph, one parameter count, one seed, one frame
set, one optimizer, one loss and one evaluator — the only variable is the sign
information.

## 6. S0 vs S1 — the Oracle Sign gate

200 epochs, batch 256, AdamW 3e-4 → 1e-5 cosine, seed 1337, `baseline_geometry_v1`
unchanged, selection on validation only.

### Flip metrics (primary)

| metric | split | S0 neutral | S1 oracle | change |
| --- | --- | ---: | ---: | ---: |
| root yaw MAE | test | 10.92° | **8.23°** | −24.6% |
| **root yaw P95** | test | 28.67° | **19.70°** | **−31.3%** |
| root yaw median | test | 8.19° | 6.80° | −17.0% |
| shoulder sign disagreement (stable) | test | 0.1115 | **0.0175** | −84.3% |
| hip sign disagreement (stable) | test | 0.0977 | **0.0026** | −97.3% |
| hinge flip rate | test | 0.0212 | **0.0187** | −11.8% |
| hinge direction MAE | test | 24.82° | 23.60° | −4.9% |
| overall sign agreement | test | 0.8825 | **0.9557** | +7.3 pp |
| root yaw P95 | validation | 21.69° | **17.49°** | −19.4% |
| shoulder sign disagreement | validation | 0.0669 | **0.0167** | −75.0% |

Per-field sign agreement on test:

| field | S0 | S1 |
| --- | ---: | ---: |
| `torso_facing` | 0.9815 | 0.9895 |
| `shoulder_forward_depth` | 0.8605 | **0.9714** |
| `hip_forward_depth` | 0.8161 | **0.9859** |
| `left_elbow_forward_bend` | 0.8672 | 0.9256 |
| `right_elbow_forward_bend` | 0.8680 | 0.9353 |
| `left_knee_forward_bend` | 0.9005 | 0.9415 |
| `right_knee_forward_bend` | 0.8756 | 0.9304 |

### 6.1 What is circular here, and what is not

This distinction decides how much the gate is worth, so it is stated before the
verdict.

**Partly circular.** `shoulder_forward_depth` and `hip_forward_depth` agreement
are close to tautological: S1 is handed `sign(D)` and then scored on `sign(D)`.
The −84% / −97% figures measure that the model *uses* the input, not that the
input is informative.

**Not circular.** Root-yaw MAE/P95 is an angular magnitude, not a branch label,
and no yaw quantity is fed. Hinge flip rate is a *relative* comparison between
predicted and target bend directions, not the absolute sign that is fed. MPJPE
and PA-MPJPE are not fed at all. Those are genuine transfer, and they are where
the gate is actually decided.

## 7. Position preservation (guardrail, reported separately)

The geometry loss was not touched, and MPJPE was never optimised through the
sign branch.

| metric | split | S0 | S1 | change |
| --- | --- | ---: | ---: | ---: |
| MPJPE | test | 81.98 mm | **79.20 mm** | −3.4% |
| PA-MPJPE | test | 56.85 mm | **55.96 mm** | −1.6% |
| MPJPE | validation | 69.80 mm | 69.50 mm | −0.4% |
| PA-MPJPE | validation | 49.44 mm | 49.55 mm | +0.2% |

Frame-level: 3,966 of 7,076 test frames improved, 3,110 regressed, mean −2.78 mm.
Per stratum (test, mean Δ MPJPE): back_facing −4.0, near_frontal −3.3, frontal
−2.3, profile −1.8; medium_forward_depth −5.8, large_forward_depth −1.7,
near_zero_forward_depth **+1.0** — the only stratum that moves the wrong way is
the one where the fed signs are by construction degenerate, which is the
expected shape.

**Position was preserved, and in fact improved slightly.** Correct signs did not
buy flip resolution at the cost of geometry.

## 8. Oracle-sign architecture verdict

**PASS.** Correct discrete sign evidence materially resolves the targeted flip
quantities while leaving continuous position reconstruction with the Geometry
Core.

The non-circular evidence carries the verdict: root-yaw P95 −31.3% and hinge
flip rate −11.8% on test, with MPJPE and PA-MPJPE improving rather than
degrading, at an identical parameter count and an unchanged loss.

For scale — and *not* as a controlled comparison, since the training mixture and
frame set differ — A9's root-yaw P95 was 34.77° and never passed its gate across
the whole A6–A16 program. The same frame core, given correct signs, reaches
19.70°.

## 9. Did the Oracle stop condition fire?

**No.** The gate passed, so the VLM Sign Advisor branch is authorised.

A caveat that belongs with the verdict: S1 is an **upper bound**. It says sign
evidence is worth having; it says nothing about whether any sensor can recover
those signs from RGB. That is Section 10 onward.

## 10. VLM Sign Advisor

One lightweight VLM, no family sweep, no LoRA, no fine-tuning.

| | |
| --- | --- |
| Model | `Qwen/Qwen2-VL-2B-Instruct` |
| Parameters | 2,208,985,600, float16, frozen |
| Decoding | greedy (`do_sample=False`), `max_new_tokens=160`, seed 1337 |
| Input | the same deterministic person-crop box the pose model uses, rasterised at 448 px for legibility; the box, margin rule and mapping are unchanged and recorded |
| Temporal context | none — frame *n* only |
| Output | one JSON object with exactly the seven contract keys; anything else is **rejected**, never guessed |
| Environment | `animcv-signadvisor:cuda118` (`transformers==4.49.0` on the Layer A image) |

The prompt asks only the seven contract questions as binary visual choices with
an explicit `unclear`, and the answer→sign mapping lives beside the questions so
prompt and contract cannot drift apart. It never asks for XYZ, depth, angles or
a pose.

### Sign bank

`scripts/build_vlm_sign_bank.py`, deterministic evenly-strided subset, 600
frames per split (validation + test), 1,200 frames total, 5,000 s at 0.24
frames/s. Each record stores `sample_id`, split, bank position, image reference,
**image-content digest**, crop box, predicted state, oracle state, per-field
correctness, validity and the raw response. It is a separate artifact; the F1/F2
patch-token cache is not reused as VLM evidence.

**Schema conformance was perfect: 0 malformed responses out of 1,200.** The
parser and prompt work. What the model *says* is another matter.

### Per-field sign accuracy

Chance is not 0.5 here — the fields are imbalanced — so the honest reference is
the **majority-class baseline** on the same scored frames.

| field | scored | VLM | majority-class | VLM − majority |
| --- | ---: | ---: | ---: | ---: |
| `torso_facing` | 1,118 | 0.5868 | 0.6753 | **−0.089** |
| `shoulder_forward_depth` | 1,142 | 0.6086 | 0.6156 | −0.007 |
| `hip_forward_depth` | 1,077 | 0.6035 | 0.6110 | −0.007 |
| `left_elbow_forward_bend` | 954 | 0.6268 | 0.6447 | −0.018 |
| `right_elbow_forward_bend` | 932 | 0.6685 | 0.6856 | −0.017 |
| `left_knee_forward_bend` | 1,038 | 0.2659 | 0.7274 | **−0.462** |
| `right_knee_forward_bend` | 1,051 | 0.3168 | 0.6784 | **−0.362** |
| **weighted overall** | 7,312 | **0.5239** | **0.6617** | **−0.138** |

**Every field is below its majority-class baseline.**

### Why: the answer distribution

| field | said +1 | said −1 | said unclear |
| --- | ---: | ---: | ---: |
| `torso_facing` | 694 | 484 | 22 |
| `shoulder_forward_depth` | 1,178 | **0** | 22 |
| `hip_forward_depth` | 1,178 | **0** | 22 |
| `left_elbow_forward_bend` | 1,178 | **0** | 22 |
| `right_elbow_forward_bend` | 1,178 | **0** | 22 |
| `left_knee_forward_bend` | 1,178 | **0** | 22 |
| `right_knee_forward_bend` | 1,178 | **0** | 22 |

On six of seven fields the model emitted a **constant answer** on every frame it
answered. It is not perceiving those quantities at all; it is filling the schema.
Only `torso_facing` varies with the image, and even there it lands below a
constant "faces camera".

### Correction to an earlier reading

A 30-frame throughput probe showed `shoulder_forward_depth` and
`hip_forward_depth` at 0.033, and I read that as a systematic left/right
convention inversion. **That was wrong.** At n≈1,100 those fields sit at ~0.60,
and the real cause is the constant answer above meeting a locally skewed
30-frame stride window. The inverted-reading diagnostic added for that
hypothesis is retained because it is cheap and still discriminating — it now
correctly shows no useful inverted signal either (overall 0.458).

## 11. S0 vs S1 vs S2

S2 reuses the **S1 checkpoint unchanged** and swaps only the sign input; the
geometry core was not retrained to compensate. All three are scored on exactly
the 600 advisor-covered frames per split.

### Test (600 frames)

| | MPJPE | PA-MPJPE | yaw MAE | yaw P95 | hinge flip | shoulder sign dis. | hip sign dis. | sign agree |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **S0** neutral | 81.88 | 57.10 | 10.91° | 30.27° | 0.0204 | 0.1382 | 0.1450 | 0.8839 |
| **S1** oracle | **78.76** | **56.35** | **8.21°** | **19.84°** | **0.0136** | **0.0222** | **0.0533** | **0.9555** |
| **S2** advisor | 113.86 | 79.29 | 36.16° | **126.43°** | 0.0897 | 0.4181 | 0.4883 | 0.6954 |

### Validation (600 frames)

| | MPJPE | yaw P95 | hinge flip | sign agree |
| --- | ---: | ---: | ---: | ---: |
| **S0** | 69.72 | 22.34° | 0.0178 | 0.9034 |
| **S1** | 69.70 | **17.87°** | **0.0104** | **0.9483** |
| **S2** | 94.04 | 108.11° | 0.0700 | 0.7658 |

Frame-level deltas (test): S1−S0 −3.12 mm with 346 improved / 254 regressed;
S2−S0 **+31.98 mm** with 101 improved / 499 regressed.

**S2 is far worse than S0.** The reading is not that sign conditioning failed —
it is that the mechanism works exactly as designed and *obeys the sign it is
given*. Handed a constant branch that is wrong on roughly 40% of frames, the
model dutifully reconstructs those frames in the wrong branch. A bad sign sensor
is worse than no sign sensor, which is itself confirmation that the conditioning
is real and strong rather than decorative.

## 12. Frame-level review

`sign_advisor_v1/frame_review.json`, six examples per failure category, each
carrying `sample_id`, frame index, image reference, oracle sign, advisor sign,
the reconstructed sign under S0/S1/S2 and the position delta — so *sign changed*,
*position changed* or *both* is readable per frame.

A representative case, `3dpw:downtown_arguing_00:actor0#000000`
(`shoulder_orientation_failure`, oracle `+1`, advisor correct here):

| | reconstructed sign | MPJPE | yaw error |
| --- | ---: | ---: | ---: |
| S0 | **−1** (wrong branch) | 76.83 mm | 24.80° |
| S1 | **+1** (corrected) | **54.87 mm** | **11.04°** |
| S2 | +1 | 73.20 mm | 9.38° |

Sign changed, and position improved by 21.96 mm with it — the intended
mechanism, visible on a single frame.

And the failure mode, `3dpw:downtown_arguing_00:actor0#000290`
(`left_knee_flip`): S0 reconstructs a degenerate knee sign at 65.66 mm; S2, given
the advisor's constant `+1` where the frame needs otherwise, produces a `−1`
branch at **208.96 mm** with yaw error 115.55° — a single wrong sign dragging the
whole reconstruction into the wrong branch.

## 13. Historical F0/F1/F2 preservation

Untouched: no checkpoint, prediction, metric, report or feature cache of the
dense visual-fusion experiment was altered. They are documented as the **Dense
Visual Fusion Historical Experiment** — a different architecture hypothesis
(196 patch tokens cross-attended into continuous XYZ). **No number in this
worklog is compared with F2**, because F2 tested a different VLM mechanism.

## 14. Verdicts

**Does correct discrete sign evidence resolve the targeted flip failures while
leaving continuous position with the Geometry Core?**
**Yes.** Root-yaw P95 −31.3%, hinge flip rate −11.8%, at an identical parameter
count and an unchanged loss, with MPJPE and PA-MPJPE improving rather than
degrading. The circular part of that evidence is separated in Section 6.1.

**Can a lightweight VLM recover those signs from a single RGB frame accurately
enough to reproduce a meaningful part of the Oracle improvement?**
**No — not this one, not with this prompt.** Qwen2-VL-2B-Instruct conforms to the
schema perfectly (0/1,200 malformed) and then answers a constant branch on six of
seven fields, scoring 0.524 weighted against a 0.662 majority-class baseline. It
is not a sufficient sign sensor, and feeding its output to the pose core is worse
than feeding nothing.

This is the third outcome the batch anticipated: **the sign abstraction is
useful, the chosen VLM is not yet a sufficient sign sensor.** It does not make
the VLM a pose estimator, and it does not license tuning the prompt until the
number moves — no prompt was changed after any result was seen.

What it does justify, as the next question rather than this batch's work: the
sign channel is worth a real sensor. Whether that is a larger VLM, a different
prompt convention, a small supervised sign classifier trained on the oracle
labels, or a geometric prior is an open architecture question, and any of them
must be measured against the same S0 baseline and the same majority-class floor.

## 15. Artifacts, tests and state

Server artifacts under `LabServer63:~/animcv-output/framepose/`:

```
sign_v1/{S0,S1}/            checkpoints, training and evaluation reports, predictions
sign_v1/sign_experiment_matrix.json     contract, oracle distribution, S0/S1 comparison
sign_bank_v1/signs.npz                  (21817, 7) advisor signs + coverage mask
sign_bank_v1/sign_bank.json             per-frame records with image-content digests
sign_bank_v1/sign_bank_summary.json     provenance, accuracy, inverted-reading diagnostic
sign_advisor_v1/advisor_evaluation.json S0/S1/S2 on the common 600-frame subsets
sign_advisor_v1/frame_review.json       per-category before/after frame exports
```

New source: `src/framepose/signs.py`, `src/framepose/sign_advisor.py`,
`scripts/run_sign_experiments.py`, `scripts/build_vlm_sign_bank.py`,
`scripts/evaluate_advisor_signs.py`, `Dockerfile.signadvisor`.
Modified: `src/framepose/{model,train,evaluate}.py` (sign conditioning, sign
source, per-frame sign state and hinge flip rate — all additive).

Tests: `tests/test_frame_pose_signs.py` (24) covers the contract mathematics
against the canonical quantities it reduces, degenerate and invalid cases,
oracle determinism, the fixed neutral control, serialization, the joint routing
mask, capacity-matched S0/S1, the exact +5,376 embedding cost, refusal of
missing/wrong sign states, that every field moves the prediction, and the
runner's declared comparison semantics.

Full regression: **597 passed, 1 skipped** in the macOS authoring venv
(`.venv`, torch 2.13.0; the skip is the timm-gated backbone test), and the
non-GUI suite in `animcv-framepose:cuda118` on LabServer63. No remote CI exists
for this repository and none was claimed.

Branch `arch/single_frame_first`; `.vscode/` untouched.
