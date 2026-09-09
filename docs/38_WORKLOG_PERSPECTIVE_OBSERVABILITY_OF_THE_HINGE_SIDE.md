# Worklog — Is the Complementary Hinge Side Observable Under the Real Projection? (2026-09-09)

> One bounded attribution batch on `arch/single_frame_first`. **No training, no
> sensor, no VLM, no RGB, no new SignState field, no new correction operator, no
> threshold tuning, no default change.**
>
> This closes two attribution gaps left by docs/37 and nothing else.

## 1. Starting/final HEAD

Start `9dc95bd` (docs/37 closed). This batch: `a7a52d2` (perspective
observability diagnostic), `4521779` (docs/37 corrections + contracts), plus the
run and worklog commits.

## 2. docs/37 preserved

Untouched: docs/33-37, `hinge_ownership_v1`/`v2`/`v3`,
`hinge_policy_minimum_norm_v1`, `hybrid_signstate_v1`/`v2`, every residual-flip
record. This batch writes a new lineage (`hinge_projection_v1`).

Preserved as valid: the local u/v hinge-plane decomposition; `sign(c_depth)`
matching the existing hinge SignState where readable; the axis-normalized
counterfactual; that the 83 depth-correct residuals are strongly associated, **in
target-3D space**, with complementary local-plane disagreement and/or limb-axis
error; that no evidence justifies immediately adding another SignState bit; and
`MINIMUM_NORM`'s measured X/Z ownership cost.

## 3. docs/37 claims downgraded to PROVISIONAL

Recorded in docs/37's own header: "Outcome B failed", "Outcome C is refuted",
"the complementary side is directly observable from current `input_2d`", and "no
additional VLM evidence can be useful". The standing statement is now:

> the complementary component lies in a camera-parallel direction, but its
> observability from the current perspective 2D contract is unresolved

— and this batch then resolves it, with numbers, rather than leaving it open.

## 4. Orthographic vs perspective line side: not identical

With `P, M, D` the chain and forward depth `Y`, define the three 2×2 minors

```
A = M_X·D_Z − M_Z·D_X      B = P_X·D_Z − P_Z·D_X      C = P_X·M_Z − P_Z·M_X
```

Then, up to one fixed overall sign and with all depths positive:

```
canonical X/Z (orthographic) side  ~  sign(  A  −   B  +   C )
perspective image side             ~  sign(Y_P·A − Y_M·B + Y_D·C)
```

**They are the same three minors, weighted by each joint's own depth.** They
coincide exactly when `Y_P = Y_M = Y_D`, i.e. when the chain lies in a
fronto-parallel plane, and can differ otherwise. Equivalently, the projected
signed area is `det[[X,Z,Y]…] / (Y_P·Y_M·Y_D)`, so depth enters the sign through
the homogeneous column that orthography holds at 1.

Deterministic counterexample, all depths positive (pinned by test):

| joint | X | Y (depth) | Z |
|---|---|---|---|
| proximal | −2.0 | 2.0 | −2.0 |
| middle | −2.0 | 1.0 | −1.0 |
| distal | −1.0 | 1.0 | −2.0 |

`A, B, C = 3, 2, −2`; orthographic `A−B+C = −1` → side **+1**; perspective
`2A−1B+1C = +2` → side **−1**. The two disagree.

Synthetic disagreement rate rises with within-chain depth spread: 0.18% at
σ = 0.02 m, 2.46% at σ = 0.3 m, 10.46% at σ = 1.5 m. **A foreshortened limb is
precisely a chain with large depth spread**, so docs/37's identification was
weakest exactly in the regime its conclusion depended on.

## 5. 3DPW provenance, answered item by item

Traced through `scripts/prepare_3dpw.py` → `src/pose/three_dpw_adapter.py` →
`training.temporal_lifter.build_dataset` → `framepose.sources`.

| question | answer |
|---|---|
| What coordinate system is `input_2d` in? | **Normalized image coordinates**: raw detector pixels divided by `image_size` (`[x/width, y/height, confidence]`) |
| Raw pixel, normalized, or crop space? | Normalized image space. The crop transform exists only in `crops.geometry_in_crop` for the model's geometry tensor and is **not** applied to stored `input_2d` |
| Is y downward? | **Yes** — OpenPose pixel convention, carried through unchanged |
| Any affine/crop transform applied? | No, only the per-axis division by `image_size` |
| Where does `input_2d` come from? | 3DPW's shipped `poses2d`, OpenPose COCO-18 **detector** output. `pelvis`, `spine`, `neck` are **derived midpoints**; `head` is the nose |
| `target_3d` before root-relative | SMPL-24 `jointPositions` (world) → `cam_poses` extrinsic → OpenCV camera → AnimCV axes `(x, z, −y)` |
| `target_3d` after | `camera_joints -= camera_joints[0]`: pelvis-relative metres, absolute depth **discarded** |
| Camera intrinsics in the prepared artifact? | **No** |
| Absolute/root depth in the prepared artifact? | **No** |
| Exact projection reconstructible from artifacts already present? | **Yes** — see Section 6 |

## 6. Exact projection IS recoverable, and was verified

The prepared artifact does not retain the camera, but every input needed to
rebuild it **already exists**: 3DPW's own `cam_intrinsics` (3×3 K) and
`cam_poses` (per-frame 4×4 extrinsics) in the sequence pickles the prepared
artifact names in its own `annotation_path`, plus the repository's own
`three_dpw_adapter._world_to_animcv_camera` conversion. **No focal length,
principal point or root depth was inferred.**

The reconstruction is not trusted, it is checked: absolute canonical joints were
rebuilt for all 7,076 test frames and, after the adapter's own root subtraction,
compared against the bank's stored `target_3d`.

```
max |rebuilt − bank target_3d|  =  0.000030 mm     (all 7,076 frames)
```

The diagnostic refuses to attribute anything unless that is under 1 µm. So the
projection used below is the actual camera that produced the observation, not a
weak-perspective stand-in.

## 7. The population mismatch, fixed

docs/37 reported 2D-side accuracy under the label `residual_flip_frames`
computed over **all 279** residual historical flips, while the question concerns
the **83** where the hidden depth sign is already satisfied. Populations are now
named once each and the identity is asserted, not assumed:

```
279 residual flips = 83 depth-correct + 196 depth-incorrect        ✓ asserted
```

with "depth-correct" defined exactly as `requested ≠ UNKNOWN` **and**
`final == requested` **and** `historical full-3D flip == true`.

## 8. Three-way attribution on the correct population

Pooled balanced accuracy across the four chains:

| population | detector ~ target canonical | **projected-target ~ target canonical** | **detector ~ projected-target** |
|---|---|---|---|
| all frames (n=25,095) | 0.853 | **0.959** | 0.862 |
| all 279 residual flips | 0.573 | 0.915 | 0.631 |
| **83 depth-correct residuals** | **0.748** | **0.918** | **0.845** |
| TYPE 1 corrections (n=179) | 0.633 | 0.972 | 0.657 |
| TYPE 2 corrections (n=1,211) | 0.943 | 0.996 | 0.946 |

Reading the middle column: **the perspective projection preserves the canonical
screen-side sign at 0.918–0.996**. The projective ambiguity is real but modest —
about 8 points on the 83, and ~4 points overall.

Reading the right column: **the shipped detector recovers the true projected
side at 0.845 on the 83**, against 0.862 across all frames.

docs/37 reported 0.28–0.64 for "residual flip frames". The corrected figure on
the population that matters is **0.748** against the canonical target and
**0.845** against the side actually visible in the image. Both gaps — the wrong
population and the orthographic identification — pushed the same way, and
together they understated the observation by roughly 17–27 points.

Per field on the 83, `detector ~ projected-target`:

| field | n | balanced accuracy | confusion |
|---|---|---|---|
| left_elbow | **63** | 0.927 | 46/14 correct, 1+2 wrong |
| right_elbow | 10 | 0.438 | 7 correct, 3 wrong |
| left_knee | 5 | 0.500 | 1 correct, 4 wrong |
| right_knee | 5 | 0.500 | 3 correct, 2 wrong |

**Three of the four chains have n ≤ 10.** Only the left elbow supports a
per-chain claim, and there the detector reads the visible side at 0.927.

The mechanism is confirmed by depth spread: on the 83, left-elbow chain depth
spread has median **0.455 m** against **0.141 m** across all frames — 3.2×
larger. These are foreshortened arms, exactly the regime where orthographic and
perspective sides diverge.

## 9. `IMAGE_TO_CANONICAL` ownership

The repository has **no** normative observation-coordinate contract asserting
that input x is right-positive and input y is down-positive. The prepared
artifact records `input_kind` and `coordinate_frame` for the 3D target, but
nothing normative for the 2D image axes.

So option **A** was taken: the constant is **not** promoted to a FramePose
invariant. It is renamed `EMPIRICAL_AXIS_CORRESPONDENCE` and explicitly labelled
`status: bank_specific_diagnostic`, `bank: bank_3dpw_paired_v2`,
`is_projection_contract: false`, with the historical name retained so docs/37's
artifacts stay readable. A test pins that it is never represented as projection
equivalence.

## 10. `MINIMUM_NORM` observation ownership

docs/37's finding is preserved unchanged: `MINIMUM_NORM` changes X/Z and worsens
target-X/Z error on three of four chains (+11.4, +11.8, +5.7 mm; −3.1 mm on the
fourth). It is **not promoted**; `DEPTH_ONLY` remains the default.

Exact reprojection of the middle joint before and after `MINIMUM_NORM` is now
technically computable (Section 6). It was **not** measured here: Section 11 of
the DIRECTION gates it on recoverability, but Section 12 forbids evaluating a
correction operator in this batch, and reporting a reprojection delta for a
candidate would be exactly that. Recorded as **deferred, now unblocked** rather
than unresolved — the instrument exists and the camera is verified.

## 11. Revised Outcome B

B requires **both** conditions. Assessed only on the 83:

- **First condition holds.** docs/37 showed the residual is associated with
  complementary screen-side disagreement in 88.0% of the 83 (in target-3D
  space), and that finding is preserved.
- **Second condition: partially met, and much more nearly than docs/37 said.**
  The current Geometry Observation supplies the side at **0.748** balanced
  accuracy against the canonical target and **0.845** against the side actually
  visible in the image — not the 0.28–0.64 docs/37 reported.

Why it is not higher, decomposed rather than blamed on the detector:

| contribution | size on the 83 |
|---|---|
| projection degeneracy (canonical ≠ projected under foreshortening) | ~8 points (0.918 → 1.0) |
| detector disagreement with the true projected side | ~15 points (0.845 → 1.0) |
| compound, as measured against the canonical target | 0.748 |
| unavailable projection provenance | **none** — the projection was recovered exactly |

So **B receives substantially more support than docs/37 granted**, and the
dominant limiter on the 83 is detector fidelity rather than projective
ambiguity — but 0.748 is still well below what a *hard* constraint could
consume, given that a wrong hard side is catastrophic (docs/33, docs/35
wrong-sign endpoints). And with n ≤ 10 on three of four chains, the pooled
figure rests almost entirely on the left elbow.

## 12. Revised Outcome C

C is **not refuted**, and `v_hat.Y == 0` is not the reason to reject it — that
proves the direction is camera-parallel, nothing about observability under
perspective.

C is nevertheless **unsupported**, on stronger grounds than docs/37 had: the
complementary side survives the real projection at 0.918 balanced accuracy on
the 83, so it is largely present in the image, and what is missing is recovered
at 0.845 by the shipped detector. A residual that is 92% preserved by the camera
and 85% read by the observation is not a demonstration of hidden evidence.

**This does not authorize a new SignState field, and none was added.**

## 13. Is richer SignState/VLM evidence justified? **No, not on this evidence**

The complementary side is largely observable in the image, and both remaining
gaps — 8 points of projective ambiguity under foreshortening, 15 points of
detector disagreement — are Geometry Observation questions. Neither is evidence
that a hidden bit is missing. The advisor's single depth bit is satisfied on all
83 of these frames by construction.

## 14. Confirmation

No new correction operator, no two-evidence constraint, no observed-side
enforcement, no X/Z clamp, no weighted minimum norm, no
reprojection-preserving optimization. No SignState field, no VLM output, no
Qwen retry, no classifier, no RGB, no confidence weighting, no temporal context.
No pose model trained or fine-tuned. No Sign Contract threshold changed. No
default changed. No camera parameter invented — the diagnostic refuses unless
its reconstruction reproduces the bank's own `target_3d` to within 1 µm.

Tests: 8 new projection/population contracts. Shared production code was
modified (`src/framepose/hinge_plane.py`), so full regression ran at closure —
**514 passed, 40 skipped**.

LabServer63, `animcv-framepose:cuda118`, repo and data mounted read-only. No GPU
work, no model execution.

## 15. Completion

> **Is the complementary hinge-side quantity that remains after the depth bit is
> correct actually observable from AnimCV's current 2D Geometry Observation
> under its real projection semantics?**

**Partially — and it is now measured, not unresolved.**

The projection question is **resolved**: the real perspective camera preserves
the canonical complementary side at **0.918** balanced accuracy on the 83
depth-correct residuals (0.959 across all frames), so the quantity is largely
present in the image rather than destroyed by projection. docs/37's orthographic
identification was **not exact** — it is the same minors with the depth weights
dropped, and it fails measurably exactly under the foreshortening that
characterises these frames (chain depth spread 3.2× the norm) — but it was a
reasonable approximation, wrong by about 8 points here.

The observation question is **answered but negative for hard use**: the shipped
detector reads the true projected side at **0.845**, and the compound accuracy
against the canonical target is **0.748**. That is far better than docs/37's
0.28–0.64, which measured the wrong population, but it is not a reliable supply
for a hard constraint, and three of the four chains have n ≤ 10 so only the left
elbow (n=63, 0.927) supports a per-chain claim at all.

Ownership therefore sits with **Geometry Observation and Geometry Core**, not
with the Sign Advisor — but "geometry-owned" now carries a measured quality
figure rather than an assumed one, and the honest label for using it in a hard
constraint today is **not yet sufficient**.

**STOP.** No SignState field added, no VLM output designed, no sensor chosen, no
operator implemented, no default promoted.
