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
