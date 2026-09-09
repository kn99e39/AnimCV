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
