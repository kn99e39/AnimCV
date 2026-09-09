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
