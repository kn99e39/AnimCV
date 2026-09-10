# Worklog — Which Hinge Write Policy Preserves the Visible Observation? (2026-09-09)

> One bounded ownership comparison on `arch/single_frame_first`. **No training,
> no sensor, no VLM, no RGB, no SignState change, no new write policy, no
> tuning, no default change.** The Sign Advisor field contract is frozen.
>
> The question: **under the actual perspective camera, does `MINIMUM_NORM`
> preserve Geometry Observation ownership better, worse, or comparably to
> `DEPTH_ONLY`, while enforcing the same hidden hinge branch?**

## 1. Starting/final HEAD

Start `8ba5f48` (docs/38 closed). This batch: `ff84251` (docs/38 corrections),
`5dbdf19` (write-policy observation diagnostic), plus the run and worklog
commits.

## 2. docs/38 corrections

Three corrections, none of which changes a measured number.

**The reconstruction bound was overstated.** docs/38 said the diagnostic refuses
unless the rebuilt geometry matches within **1 micrometre**. It does not: the
implemented boundary is `difference.max() > 1.0` with `difference` in
millimetres, i.e. **1 mm**.

| quantity | value |
|---|---|
| measured reconstruction error (accepted run) | **0.000030 mm** |
| implemented refusal boundary | **1.0 mm** |

The measured error is 33,000× inside the boundary, so no result changes — but
the contract is 1 mm and the prose claimed a contract 1,000× stricter than the
code. No repository contract mandates a stricter bound (the only comparable
thing, `test_canonical_pose_parity`, governs SVD platform tolerance, not this),
so **the documentation is corrected and the threshold is deliberately left
alone**. Tightening the code to make the sentence true would be fixing the wrong
artifact. The new diagnostic states the boundary in the same units it computes
in, as `RECONSTRUCTION_REFUSAL_MM`, so the two cannot drift apart again.

**`identity_holds: True` was unconditional.** `depth_incorrect = all −
depth_correct` is arithmetic, not verification of the historical 279/83 lineage.
That flag is removed and replaced by
`decomposition_is_arithmetic_not_verification`, with real binding done here
(Section 3).

**"Detector fidelity" was the wrong label.** `input_2d` carries OpenPose COCO
landmarks; the projected reference is an SMPL joint centre. The gap between them
mixes detector localization error, landmark-vs-joint-centre definition mismatch,
and annotation geometry, and this data cannot separate them. It is renamed
throughout to **Geometry Observation vs projected-target mismatch**, and the
image-space comparison in this batch is called **observation-consistency
error** for the same reason.

## 3. Historical 279/83 binding

The new diagnostic binds against the accepted historical artifact
(`hinge_ownership_v3/hinge_evidence_ownership.json`) rather than asserting an
identity. It recomputes the residual populations from scratch and checks four
things against the stored record:

- residual total matches the accepted historical total
- depth-correct residual count matches the accepted historical total
- source prediction SHA-256 matches
- the historical artifact's own `hinge_write_policy` is `minimum_norm`

If the artifact lacks the machine-readable provenance for any of these, the
result is recorded as **unverifiable with the reason**, never as a passing
identity.

## 4. Camera provenance

Reused unchanged from docs/38: 3DPW's own `cam_intrinsics` and `cam_poses`, plus
the repository's own `three_dpw_adapter._world_to_animcv_camera`. No camera
parameter is inferred.

Recorded per raw sequence so the run cannot silently depend on an unrecorded
replacement pickle: sequence name, path, byte size, **SHA-256**, the full 3×3
intrinsic matrix, the image size those intrinsics imply, actor count and frame
count. 24 sequences.

**The test split is not one camera.** 13 sequences are landscape
(1920×1080, fx 1969.2) and 11 are portrait (1080×1920, fx 1961.9). A first
implementation asserted a single shared image size and projected each field
through one K; the guard rejected the run rather than projecting 3,159 frames
through the wrong camera. Projection, normalization and the wrong-sign endpoint
now all use **per-frame** intrinsics, and each frame's bank `image_size` is
checked against that frame's own intrinsics.

Reconstruction check, as in docs/38 and with the boundary now named in its own
units: measured max 0.000030 mm against `RECONSTRUCTION_REFUSAL_MM = 1.0`.

## 5. Oracle absolute-root placement — declared

The stored prediction is root-relative and AnimCV does **not** know absolute root
depth. To compare the two policies under one real camera, the prediction is
placed at the target's absolute camera-space pelvis:

```
absolute_state = root_relative_state + absolute_target_pelvis      (identical root for all states)
```

`mode: oracle_absolute_root_placement`. **This is a diagnostic device, not
production inference**, and the same root is used for H0, `DEPTH_ONLY` and
`MINIMUM_NORM`, so the comparison cannot be an artefact of placement. A test
pins that the three states share one root expression.

## 6. Semantic identity holds — the comparison is fair

Before anything else is reported, the diagnostic refuses unless both policies
still enforce the same branch:

| check | result |
|---|---|
| chain-frames compared | 25,095 |
| max bend-direction difference | **5.07 × 10⁻¹⁵** |
| sign states identical | **True** |

So every difference below is *where the joint landed*, never *what was
enforced*.

3D aggregates are unchanged from docs/36:

| state | MPJPE | PA-MPJPE | hinge flip | hinge MAE |
|---|---|---|---|---|
| H0 | 79.229 | 56.425 | 0.0189 | 24.099 |
| `DEPTH_ONLY` | 79.366 | 56.694 | 0.0116 | 22.129 |
| `MINIMUM_NORM` | **79.282** | **56.543** | 0.0116 | 22.129 |

Historical binding: **all four checks pass** — residual total 279, depth-correct
83, source prediction SHA-256, and the historical artifact's own policy is
`minimum_norm`.

## 7. Correction-induced image displacement

Pixel displacement of the middle joint from H0, under the real per-frame camera:

| field | `DEPTH_ONLY` p50 / p90 / max | `MINIMUM_NORM` p50 / p90 / max |
|---|---|---|
| left_elbow | 5.76 / 16.84 / 60.5 | 8.74 / 39.61 / 136.6 |
| right_elbow | 5.77 / 16.52 / **324.0** | 9.24 / 33.50 / 171.4 |
| left_knee | 3.74 / 16.74 / 30.3 | 7.03 / 31.17 / 65.1 |
| right_knee | 4.62 / 12.20 / 45.1 | 8.96 / 23.07 / 122.8 |
| **pooled mean** | **7.74 px** | **13.88 px** |

Normalized to the image diagonal, p50: `DEPTH_ONLY` 0.0017–0.0026,
`MINIMUM_NORM` 0.0032–0.0042.

**The batch's premise is confirmed but is smaller than expected.** A depth-only
write does move the joint in the image — 3.7–5.8 px at the median, never zero —
so "canonical X/Z unchanged" is genuinely not a guarantee of image preservation.
But it is a **better** proxy than the perpendicular write: `MINIMUM_NORM` moves
the joint about **1.8× further** in the image.

The canonical deltas say why. At the median both move 48–73 mm in depth, but
`MINIMUM_NORM` additionally moves 7.6–11 mm in Z and 0.6–2.8 mm in X. A lateral
metre moves the image by `f·Δ/Y`; a depth metre moves it by only `f·X·Δ/Y²`,
and the joint's offset from the optical axis (`X`, `Z` in metres) is small
compared with its depth. Depth writes are attenuated in the image by roughly an
order of magnitude; perpendicular writes are not.

## 8. Target-projection error (Reference A, like-for-like)

Mean pixel distance from the projected absolute SMPL joint centre:

| field | H0 | `DEPTH_ONLY` | `MINIMUM_NORM` |
|---|---|---|---|
| left_elbow | 26.46 | **25.25** | 30.88 |
| right_elbow | 28.15 | 31.93 | 33.86 |
| left_knee | 17.15 | **16.98** | 18.95 |
| right_knee | 25.99 | 23.97 | **23.21** |
| **pooled** | **24.898** | **24.745** | **26.972** |

`DEPTH_ONLY` is essentially neutral against the target's real projection
(−0.15 px); `MINIMUM_NORM` costs **+2.07 px**.

## 9. Observation-consistency error (Reference B)

Mean pixel distance from the stored `input_2d` the FramePose Core consumed.
This is **not** detector error: OpenPose COCO landmarks and SMPL joint centres
are different semantic objects, so the gap mixes detector localization,
landmark-vs-joint-centre definition and annotation geometry.

| field | H0 | `DEPTH_ONLY` | `MINIMUM_NORM` |
|---|---|---|---|
| left_elbow | 25.32 | **25.06** | 30.04 |
| right_elbow | 29.03 | 33.20 | 35.66 |
| left_knee | 21.00 | **20.05** | 23.40 |
| right_knee | 28.47 | 26.45 | **25.10** |
| **pooled** | **26.234** | **26.294** | **28.541** |

Same ordering as Reference A, which is reassuring: the conclusion does not
depend on which reference is used. `DEPTH_ONLY` +0.06 px, `MINIMUM_NORM`
+2.31 px.

## 10. Conditioning regime

Correlation of `log10(image displacement)` with `√f`, per chain:

| field | `DEPTH_ONLY` | `MINIMUM_NORM` |
|---|---|---|
| left_elbow | −0.265 | **−0.526** |
| right_elbow | −0.435 | **−0.539** |
| left_knee | −0.091 | **−0.336** |
| right_knee | −0.001 | **−0.273** |

Both worsen as the limb axis approaches the depth direction, and — against
expectation — **`MINIMUM_NORM`'s image displacement grows *faster*** as `f → 0`.
Its 3D correction is bounded by `2|o|` (docs/36), but its direction `û` becomes
increasingly *lateral* as the axis becomes depth-aligned, and lateral motion is
exactly what the image sees. **A bounded 3D correction does not imply a bounded
image displacement.**

On docs/36's already-published tail ranges, carried for continuity only and used
as no kind of gate — median image displacement on frames whose `DEPTH_ONLY` 3D
correction exceeds the range:

| range | n | `DEPTH_ONLY` | `MINIMUM_NORM` |
|---|---|---|---|
| >200 mm (left_elbow) | 47 | 13.08 px | 52.07 px |
| >500 mm (left_elbow) | 5 | 25.46 px | 91.63 px |
| >1000 mm (left_elbow) | 1 | 51.18 px | 136.57 px |
| >500 mm (right_elbow) | 6 | 122.16 px | 144.99 px |

Even in the ill-conditioned tail that motivated `MINIMUM_NORM`, its **image**
displacement is larger, not smaller.

## 11. Correct-sign and wrong-sign endpoints

Correct sign, pooled: `DEPTH_ONLY` 7.74 px displacement / 24.745 px
target-projection; `MINIMUM_NORM` 13.88 / 26.972.

Wrong sign (opposite oracle), pooled over 20,417 corrected chain-frames:

| | `DEPTH_ONLY` | `MINIMUM_NORM` |
|---|---|---|
| image displacement | **11.12 px** | 23.67 px |
| target-projection error | **21.72 px** | 31.74 px |
| max displacement, per chain | 98.9 – 220.5 px | 191.0 – 342.8 px |
| 3D MPJPE | 93.561 | **92.099** |
| 3D PA-MPJPE | 75.192 | **72.646** |

**The two spaces disagree.** Under wrong advice `MINIMUM_NORM` is safer in 3D
(docs/36's finding, reproduced) but **twice as damaging in the image**. The
reason is the same as Section 7: it spends its correction laterally.

## 12. Bone length, canonical X/Z and image space, side by side

| invariant | `DEPTH_ONLY` | `MINIMUM_NORM` |
|---|---|---|
| canonical X/Z of the middle joint | **exact** | 0.6–2.8 mm X, 7.6–11 mm Z (p50) |
| both adjacent bone lengths | 7.9–11.2 mm p50, **max 2,952 mm** | **exactly 0.000 mm** |
| image displacement (correct sign) | **7.74 px** | 13.88 px |
| image displacement (wrong sign) | **11.12 px** | 23.67 px |
| target-projection error | **24.745 px** | 26.972 px |
| 3D MPJPE (correct / wrong sign) | 79.366 / 93.561 | **79.282 / 92.099** |
| ill-conditioned 3D tail | **unbounded** (3.1 m) | bounded by `2\|o\|` |

The two tested operators are non-dominated under the measured objectives;
neither satisfies both image-space ownership and metric-skeleton ownership
simultaneously. `DEPTH_ONLY` better preserves image-space observation than
`MINIMUM_NORM` among the two tested policies; it does **not** preserve the
image exactly. `MINIMUM_NORM` owns the metric skeleton and the 3D tail.

## 13. Classification: **B — `DEPTH_ONLY` observation ownership supported**

B's two clauses, checked:

- **"Preserving canonical X/Z also materially preserves the real image
  observation better than `MINIMUM_NORM`."** Yes. `DEPTH_ONLY` is better on
  **every** pooled image metric — displacement 7.74 vs 13.88 px, target
  projection 24.745 vs 26.972 px, observation consistency 26.294 vs 28.541 px —
  and the margin widens under a wrong sign (11.12 vs 23.67 px displacement,
  21.72 vs 31.74 px target projection). Both references agree, and the ordering
  holds on 3 of 4 chains individually.
- **"The conditioning tail is judged a separate unresolved defect."** Yes, and
  it is not only a tail: `DEPTH_ONLY` violates both adjacent bone lengths on
  **every** correction (7.9–11.2 mm at the median) with a maximum of **2,952 mm**,
  and its worst image displacement is 324 px.

So **B**, with the caveat stated plainly: B licenses "`MINIMUM_NORM` cannot be
promoted on observation-ownership grounds". It does **not** say `DEPTH_ONLY` is
a good operator. Its bone-length violation is systematic rather than a tail
phenomenon, and docs/36's unbounded depth excursion is untouched by this result.

**docs/37's objection to `MINIMUM_NORM` is vindicated** — and docs/38's
correction to the reasoning also stands. Both were right about different things:
canonical X/Z preservation is *not* the same as image preservation (docs/38),
and yet the policy that preserves canonical X/Z does preserve the image better
(docs/37's instinct), because depth writes are attenuated in projection by
roughly `X/Y` while perpendicular writes are not.

## 14. Can `MINIMUM_NORM` be promoted as the research hinge policy?

**No.** It stays exactly where docs/36 left it: a research candidate, not the
default, and now with a measured observation-ownership cost of roughly **+2.1 px
target-projection error and +6.1 px image displacement** on corrected
chain-frames, doubling to **+10 px** under a wrong sign.

`DEPTH_ONLY` remains the default. Nothing was promoted, and no default changed.

## 15. A new write-policy architecture question does remain

The two tested operators are non-dominated under the measured objectives;
neither satisfies both image-space ownership and metric-skeleton ownership
simultaneously. The trade is now explicit:

```
DEPTH_ONLY      better preserves image-space observation among the two tested policies,
                violates bone length, unbounded in 3D
MINIMUM_NORM    owns bone length and 3D bound, costs image position
```

The natural next hypothesis — a correction that is minimal *in the image* or
constrained along the viewing ray rather than in 3D — is a separate architecture
question. Section 15 of the DIRECTION forbids implementing it here and **none
was implemented**: no projection-preserving correction, no ray-constrained
correction, no weighted minimum norm, no reprojection optimization, no X/Z
clamp, no blend.

One finding sharpens that future question: `MINIMUM_NORM`'s image displacement
correlates with `√f` **more** strongly than `DEPTH_ONLY`'s (−0.27…−0.54 against
−0.00…−0.44). Bounding the 3D correction did not bound the visible one, so a
future policy that wants image ownership must optimize in the image, not in 3D.

## 16. SignState / VLM scope stayed frozen

No hinge SignState field was added, no screen-side field, no VLM output, no Qwen
retry, no classifier, no RGB, no confidence weighting, no temporal context, and
learned bilateral conditioning was untouched. docs/38's result stands: **no
evidence currently justifies another hinge SignState field**, and the residual
observation imperfection measured here is recorded as **Geometry Observation
quality**, not transferred back to the Sign Advisor.

No Sign Contract threshold changed, no correction tuned, no third write policy,
no Geometry Core change, no retraining, no temporal evidence, no VLM.

## 17. Tests and environment

10 new contracts in `tests/test_frame_pose_write_policy_observation.py`: the
axis conversion, per-frame intrinsics agreeing with per-frame projection, the
premise itself (a depth-only write moves the joint tens of pixels on both image
axes and grows with the depth delta), the shared oracle root placement, the
reconstruction boundary carrying its unit, the two policies' branch identity on
a deliberately tilted-axis fixture, `input_2d` pixel inversion, reference
separation, and historical binding reporting *unverifiable* rather than
asserting.

Shared production code was not modified in this batch (the diagnostic and its
tests are new files; the only edits to shipped modules were docs/38's comment
and label corrections committed separately, after which the suite was run).
Suite at closure: **524 passed, 40 skipped**.

LabServer63, `animcv-framepose:cuda118`, repo and data mounted read-only. No GPU
work, no model execution, no training.

## 18. Completion

> **Under the actual perspective camera, does `MINIMUM_NORM` preserve Geometry
> Observation ownership better, worse, or comparably to `DEPTH_ONLY` while
> enforcing the same hidden hinge branch?**

**Worse.** With the branch provably identical (max bend-direction difference
5.07 × 10⁻¹⁵, sign states equal across 25,095 chain-frames), `MINIMUM_NORM`
moves the middle joint **1.8× further in the image** (13.88 vs 7.74 px pooled),
lands **+2.07 px further** from the target's real projection and **+2.31 px**
further from the observation the Core consumed, and under a wrong sign the gap
**doubles** (23.67 vs 11.12 px displacement; 31.74 vs 21.72 px target
projection). The ordering is the same under both references and on 3 of 4
chains individually.

The mechanism is now explicit: image position divides by depth, so a depth-only
write is attenuated by roughly `X/Y` while `MINIMUM_NORM`'s perpendicular write
is not — and as the limb approaches the depth axis, `MINIMUM_NORM`'s correction
becomes *more* lateral, so bounding the 3D correction did not bound the visible
one.

This closes the hinge write-policy comparison. `DEPTH_ONLY` keeps the default,
`MINIMUM_NORM` keeps its 3D and bone-length advantages as a research candidate,
and the fact that **neither owns both spaces** is recorded as an open
architecture question rather than answered with a third operator.

**STOP.** No operator designed, no default promoted, no SignState or VLM work.
