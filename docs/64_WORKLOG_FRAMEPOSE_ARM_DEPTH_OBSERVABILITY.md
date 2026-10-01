# Worklog — FramePose Left-Arm Depth Observability (2026-10-01)

## Starting state and files

- **Start:** fetched `origin/arch/single_frame_first` before editing. Local and remote were both `4909f33` (Worklog 63) and the worktree was clean.
- **Commits in this batch:**
  - `bd1018b`: probe module, training/evaluation script, tests
  - `056fe46`: evaluation fix
  - the commit that adds this worklog
- **Added files:**
  - `src/framepose/arm_depth_probe.py` (diagnostic)
  - `scripts/run_framepose_arm_depth_probe.py`
  - `tests/test_framepose_arm_depth_probe.py`
  - this worklog
- **No existing file was modified.** That includes `framepose/model.py`, `train.py`, `losses.py`, the O_BILATERAL checkpoint, frozen H0 predictions, the Geometry Observation Layer, AnimationSemantics, FK, IK and Root Orientation.

**First run.** It trained to completion, but its evaluation then crashed on a degenerate observed 2D segment in a row that would have been ineligible anyway. `056fe46` gives such rows a placeholder plane before they are excluded. The run was then repeated from scratch with the same seed and contract. All results below come from that complete second run.

## Candidate input identity

The probe receives exactly H0's evidence: the bank's `geometry_tensor`, i.e. crop-normalized 2D x/y, confidence and validity, built with the same crop contract. It also receives the same `(N, 7)` sign array: `mask_fields(oracle_sign_states(target_3d, target_valid), [shoulder_forward_depth, hip_forward_depth])`.

- **Identity check.** Re-running the frozen H0 checkpoint on these arrays reproduced the stored validation H0 to a maximum difference of 0.0 m.
- **Oracle signs.** The two sign fields are oracle in the historical H0 lineage, so this is an *optimistic architecture control* for both models, not a statement about production sensing.
- **Inputs not used:** no RGB, VLM/MLLM, depth map, temporal or future frames, and no GT at inference beyond those two sign fields. A test checks that `predict_probe` takes only geometry, signs and positions.

## Model and parameter identity

- **Graph.** The probe builds the **unchanged** `FramePoseEstimator`, with ModelConfig read from the O_BILATERAL checkpoint: width 256, 8 heads, fusion depth 2, FF ×4, 7 sign fields, pre-attention, geometry only.
- **Parameters.** Both H0 and the probe have **1,657,603** parameters. Capacity is matched exactly, and no capacity was added.
- **Changed part.** Only the output interpretation differs:
  - f_upper = tanh(head[left_elbow, channel 0])
  - f_lower = tanh(head[left_wrist, 0])
  - f_chain = tanh(head[left_shoulder, 0])
  - The other head outputs exist but receive no gradient.
  - The token assignment was fixed before training and never searched.

## Target representation

For upper arm (shoulder→elbow), forearm (elbow→wrist) and chain (shoulder→wrist), the target is `f = unit(v).Y` from the bank's `target_3d`, in canonical coordinates with +Y forward/depth.

- **Properties.** f is bounded in [−1, 1] and scale-free, and it carries continuous depth magnitude as well as sign. Absolute joint Y is not predicted.
- **Masking.** Each segment is masked by `target_valid` on both of its joints.

**Image-plane ownership.** The probe never learns an image-plane direction. The observed 2D segment, mapped image x → +X and image y → −Z, supplies the (X, Z) orientation. The probe's f then gives the reconstructed unit direction `(√(1−f²)·p̂ₓ, f, √(1−f²)·p̂_z)`. No camera depth or scale is invented.

**Perspective floor.** As an evaluation-only control, **oracle canonical XZ + predicted f** isolates depth quality from the perspective floor. That floor (GT f on the observed plane) is 2–5° mean.

**SignState relation.** sign(f) is reported only as an agreement statistic. The two conditioning signs are torso-level shoulder and hip forward depth, not arm quantities. The probe predicts continuous magnitude, and SignState is not reopened.

## Training contract

Copied from the O_BILATERAL checkpoint's candidate record:
- bank `75519e63…` with its own train/validation/test split (11,334 / 3,407 / 7,076)
- AdamW, learning rate 3e-4 cosine to 1e-5, weight decay 1e-4
- 200 epochs, batch 256, seed 1337, AMP
- validation every 10 epochs

The one objective, declared before any result, is a **masked mean absolute error on the three tanh-bounded forward fractions**. There was no sweep, no weighting, and no MPJPE, bone or temporal term.

**Selection.** The checkpoint was selected on the validation mean |f| error only, which picked epoch 79 (0.2300). Test was never used for selection.

**Learning curve.** Validation reaches its plateau by epoch 29 (0.2351) and stays at 0.230–0.236. Training loss keeps falling to 0.005. The model memorises the training frames but cannot generalise beyond about 0.23 mean |f| error.

## Results

**Eligibility:** GT valid for the three arm joints, detector input valid for the three, and non-degenerate observed 2D segments. This leaves 3,056 validation and 6,003 test rows.

### Held-out test (6,003 rows; test never touched selection)

| segment | metric | H0 mean / p50 / p95 | probe mean / p50 / p95 |
|---|---|---|---|
| upper arm | \|f error\| | 0.237 / 0.168 / 0.661 | 0.227 / 0.162 / 0.629 |
| | \|elevation error\| | 14.5° / 10.3° / 39.5° | 13.8° / 9.8° / 38.3° |
| | full 3D, raw detector plane (H0: own XYZ) | 16.2° / 11.9° / 41.6° | 15.0° / 10.9° / 39.0° |
| | full 3D, oracle XZ + f | 14.5° / 10.3° / 39.5° | 13.8° / 9.8° / 38.3° |
| forearm | \|f error\| | 0.283 / 0.203 / 0.848 | 0.302 / 0.221 / 0.873 |
| | \|elevation error\| | 18.7° / 13.6° / 56.7° | 20.0° / 14.2° / 57.1° |
| | full 3D, raw plane | 21.0° / 15.5° / 60.5° | 21.8° / 15.8° / 59.3° |
| shoulder→wrist | \|f error\| | 0.271 / 0.192 / 0.810 | 0.276 / 0.197 / 0.772 |
| | \|elevation error\| | 16.9° / 11.7° / 50.9° | 17.2° / 11.9° / 48.3° |
| | full 3D, raw plane | 18.2° / 12.8° / 52.8° | 18.2° / 13.0° / 49.1° |

Paired comparison of the probe's raw-plane full angle against H0 full:

| segment | mean difference | probe better on |
|---|---:|---:|
| upper arm | −1.2° | 56% of rows |
| forearm | +0.7° | 49% |
| chain | +0.0° | 50% |

H0's own f placed on the same observed plane gives 15.7° / 20.4° / 17.9°. Image-plane ownership alone is therefore not what differs.

### Validation: all rows, clean subset, Worklog 62 rows

Means are listed as H0 → probe.

| scope | n | upper \|f err\| | lower \|f err\| | chain \|f err\| | chain full 3D, raw plane |
|---|---:|---|---|---|---|
| validation, all | 3,056 | 0.196 → 0.187 | 0.248 → 0.274 | 0.201 → 0.227 | 13.6° → 14.9° |
| validation, clean dancing + hug | 425 | 0.151 → 0.163 | 0.200 → 0.209 | 0.152 → 0.203 | 10.8° → 13.7° |
| validation, crosscountry only (camera caveat) | 133 | 0.258 → 0.235 | 0.324 → 0.320 | 0.284 → 0.277 | 19.8° → 19.1° |
| Worklog 62 matched rows | 395 | 0.203 → 0.206 | 0.241 → 0.237 | 0.206 → 0.248 | 14.2° → 16.6° |

Stable-sign agreement of sign(f) with GT, on rows where \|f_gt\| > sin 10°:

| scope | H0 upper / lower / chain | probe upper / lower / chain |
|---|---|---|
| test | 0.78 / 0.76 / 0.75 | 0.77 / 0.74 / 0.70 |
| clean | 0.92 / 0.83 / 0.97 | 0.95 / 0.77 / 0.92 |

Crosscountry, where the GT camera is suspect, slightly favours the probe. Following the direction, it is not used to promote the probe.

## Owner cases

Values are H0 / GT / probe f, then H0 → probe full angle on the raw plane.

| case | upper | forearm | chain |
|---|---|---|---|
| known_good, dancing #424 | .36 / .50 / .31; 9.5° → 12.3° | .04 / .12 / .03; 4.4° → 5.6° | .21 / .32 / .18; 6.9° → 8.9° |
| tracking_loss, hug #312 | −.17 / −.23 / −.06; 3.8° → 9.9° | −.38 / −.61 / −.24; 15.3° → 24.1° | −.31 / −.47 / −.07; 10.6° → 24.6° |
| largest_ratio_error, hug #397 | −.47 / −.89 / .02; 38.4° → 67.9° | −.26 / .56 / .06; 75.4° → 102.5° | −.49 / −.24 / −.22; 17.0° → 11.7° |
| largest_direction_error = largest_2d_direction_error, dancing #176 | −.79 / .64 / .61; 99.9° → 11.0° | .32 / .59 / .71; 112.0° → 76.4° | −.41 / .87 / .80; 133.2° → 59.2° |
| dynamic, crosscountry #462 | .34 / .72 / .49; 28.5° → 18.5° | −.11 / .32 / −.06; 26.8° → 24.4° | .11 / .54 / .07; 28.4° → 30.5° |
| turning_review, crosscountry #211 | −.46 / −.70 / −.12; 51.0° → 64.3° | −.66 / −.99 / −.03; 41.6° → 80.1° | −.62 / −.91 / −.09; 51.1° → 75.0° |
| near_straight, crosscountry #525 | .31 / .03 / .24; 17.3° → 12.7° | .35 / −.10 / .15; 26.2° → 14.4° | .33 / −.03 / .21; 21.3° → 14.5° |
| strongly_bent = largest_reach = largest_depth_component, crosscountry #173 | −.61 / .82 / −.04; 92.0° → 57.5° | .90 / .81 / .46; 12.0° → 33.1° | −.14 / 1.00 / .60; 97.0° → 52.2° |
| turning_invalid #202, invalid_endpoint #406 | not eligible (no valid arm input) | | |

The probe fixes some large H0 depth inversions (#176, #173) but creates others (#397 upper, #211, #312). Its f values are often shrunk toward 0 (#211, #312, #173), which is the signature of a regression target that is not determined by the input.

## Architecture classification

**CASE B: input observability limit.**
- **No substantial gain.** With identical single-frame geometry, identical oracle torso signs and identical capacity, a model trained *directly* on continuous arm forward fractions does not substantially reduce held-out depth or full-direction error on any segment.
  - Test differences are within about ±1°: upper slightly better (−0.7° elevation, −1.2° full), forearm and chain slightly worse or equal.
  - On the clean subset the probe is worse on every segment.
- **Not CASE C.** The small upper-arm gain on test reverses on the clean validation subset, so it is not a consistent segment-level improvement and does not qualify as a partial success.
- **Ceiling.** The validation plateau (about 0.23 mean |f| error, reached by about epoch 30 while training loss keeps falling) matches H0's own implicit error level of 0.20–0.25. This is the practical ceiling of this evidence for continuous arm depth.

**Answer to the completion question.** The dominant arm-depth error is **not** recoverable by changing the representation or objective over AnimCV's current single-frame geometry plus sign evidence. This holds even with oracle torso signs. Additional evidence is fundamentally required, and geometry-only single-frame FramePose has reached its practical ceiling for this quantity.

The next architecture decision should weigh a new evidence source:
- visual/RGB features
- learned visual semantics / MLLM sensing
- monocular depth
- justified temporal context

This batch does not choose among them, and claims no production improvement. Continuing to tune the geometry-only XYZ objective is not supported.

## Outputs and tests

**Outputs.** LabServer63 `~/animcv-output/framepose_arm_depth_probe/`:
- `report.json` (SHA-256 `decd654e707cfe6b94972400718a3ddfd081672c4f94113a03d4ed3a3ea97e9b`)
- `checkpoint.pt` (diagnostic)
- per-row validation and test JSON

These are machine-readable and stay on the server. No renders were produced.

**Tests:** `tests/test_framepose_arm_depth_probe.py` has 10 tests:
- continuous forward target construction and masks
- scale invariance and same direction at different lengths
- exact reconstruction from correct f and plane
- f = 0 lies in the image plane, and the sign is preserved
- finite, deterministic behaviour at |f| → 1
- observed-plane mapping (y-down → −Z) with aspect ratio
- fixed, distinct output tokens
- input identity with the H0 materialization, split identity, and no target at inference
- identical parameter count to the unchanged estimator, with deterministic bounded predictions
- historical modules untouched

Results:
- locally: 9 passed, 1 skipped (torch absent)
- in the LabServer63 training image: 10/10 passed

No production module changed, so the full regression was not run.
