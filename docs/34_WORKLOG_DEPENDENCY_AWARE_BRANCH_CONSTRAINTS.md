# Worklog — Dependency-Aware Branch Constraints and Residual Hinge Attribution (2026-09-08)

> One bounded architecture batch on `arch/single_frame_first`. **No training,
> no fine-tuning, no sensor, no VLM, no prompt, no Sign Contract change, no
> Geometry Core change, no new attention topology, no temporal evidence.**
> Every number comes from replaying *stored* predictions through
> parameter-free operators.
>
> Two questions:
>
> 1. Is docs/33's bilateral failure a property of the **hard-constraint
>    abstraction**, or of the **anchor-only write policy** of the operator that
>    was tested?
> 2. When an oracle hinge sign is successfully enforced, can the historical
>    **full-3D** hinge-flip metric still be wrong?

## 1. Starting/final HEAD

Start `8dac26b` (docs/33 closed). This batch: `259e69c` (constraint dependency
graph + dependency-aware write policy), `a67ff09` (replay provenance/validity
hardening + attribution script), plus the run and worklog commits.

## 2. docs/33 claims preserved unchanged

Preserved, not re-run, not overwritten:

- `branch_constraint_v3` and every artifact in it: `C0`, `C_HIP`,
  `C_BILATERAL`, `C_HINGE_ALL`, the four single-hinge variants, the wrong-sign
  endpoints, the review export.
- All learned `O_*`, `H_NO_*`, `L_*` results; docs/28 through docs/33.
- The accepted findings: `C_HINGE_ALL` is strongly useful under oracle signs;
  the anchor-only pair swap is not competitive with `O_BILATERAL`; the pair
  swap damages hinge semantics; wrong hard signs are dangerous.

The docs/33 hinge operator is **frozen**. Its closed form, its threshold and
its single-joint write are untouched; this batch only diagnoses its residuals.
`write_groups` returns `(("left_knee",),)` for a hinge field under *either*
write policy, and that is pinned by test.

## 3. One docs/33 reading downgraded from conclusion to hypothesis

docs/33 Section 11 wrote:

> a branch constraint is the right abstraction where the constrained joint is a
> leaf of the Sign Contract's dependency graph, and the wrong one where it is
> an anchor.

That is **not established** and is treated here as a hypothesis. The tested
bilateral operator moved the anchor joints and left the limb geometry that
depends on them behind — a **write-policy** fault before it is an abstraction
fault. Distinguishing the two is the point of this batch.

## 4. Replay provenance hardening

The docs/33 replay validated the prediction's *shape*, accepted a
caller-supplied source-candidate label, and **hard-coded** the regime string.
Every new replay artifact now records, in a `provenance` block:

| recorded | value on this run |
|---|---|
| source prediction SHA-256 | `4357c7bd…` |
| source prediction bytes | 1,443,632 |
| source evaluation digest | `evaluation_test.json` of the same S0 run |
| bank content digest | `75519e63…` |
| **observation regime** | `benchmark_detector_observation`, from `FrameBank.regime()` — **derived, never hard-coded** |
| source candidate / split | `S0_neutral_sign` / `test` |
| constraint schema | `animcv_frame_pose_branch_constraint_v1` |
| constraint validity source | see Section 5 |
| bilateral write policy | per variant |

Historical `branch_constraint_v3` is untouched; this batch writes a new
lineage (`branch_constraint_dep_v1`, `constraint_attribution_v*`).

## 5. Validity provenance, and the applicability control

docs/33 applied constraints under `target_valid`. That is GT-side validity
semantics — acceptable for an oracle architecture upper bound, but **not
production-validity evidence**, and it is now recorded explicitly as
`constraint_validity_source: target_valid`.

Two regimes were run, changing **only** the mask the correction and read-back
path sees:

| regime | mask |
|---|---|
| `V_ORACLE` | `target_valid` |
| `V_OBSERVED` | `input_valid` |

The oracle *request* is always built from `target_valid` in both, because it is
ground truth by definition and rebuilding it would confound applicability with
a different requested-sign distribution.

**On this bank the two masks are bit-identical.** `bank_3dpw_paired_v2` is a
paired construction, and on the 7,076-frame `test` split
`np.array_equal(input_valid, target_valid)` is `True` (both 94.77% valid, both
4,240 all-joints-valid frames, zero frames differing). So the applicability
control returns identical numbers, and docs/33's use of `target_valid` gave it
no GT-side advantage **on this data**.

That is a property of this bank, not a general guarantee. The control is now a
one-flag rerun for any future bank where the two masks diverge, and the
provenance block records `validity_identical_to_target_valid` so a future run
cannot quietly inherit this batch's coincidence.
