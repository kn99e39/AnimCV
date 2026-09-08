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
