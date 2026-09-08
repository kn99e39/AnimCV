# Worklog — Who Owns the Residual Hinge Disagreement (2026-09-08)

> One bounded diagnostic batch on `arch/single_frame_first`. **No training, no
> sensor, no VLM, no RGB, no new SignState field, no new VLM output, no
> threshold tuning, no default change, no Geometry Core change.**
>
> The question: **after the hidden depth branch is correct, do the remaining
> hinge failures belong to hidden evidence, to visible 2D topology AnimCV
> already observes, or to continuous limb-axis reconstruction error?**
>
> And: **does minimum-norm enforcement solve conditioning by violating the
> Geometry Core's position ownership?**

## 1. Starting/final HEAD

Start `0e3ac6c` (docs/36 closed). This batch: `77009a4` (local hinge plane +
2D side predicate), `084c49b` (ownership diagnostic), plus run and worklog
commits.

## 2. Two docs/36 wordings corrected

Both corrections are to interpretation only; no measured number changes.

**Observability.** docs/36 said `√f ≥ UNIT_FORWARD_EPSILON` means the branch is
observable. That overstates it. The correct statement:

- `√f < UNIT_FORWARD_EPSILON` proves that **no** perpendicular bend direction
  on that limb axis can have a readable depth component — the field is
  unreadable for every possible position of the middle joint.
- `√f ≥ UNIT_FORWARD_EPSILON` means a readable depth branch is geometrically
  **possible**. It does **not** mean the current bend is readable: that still
  depends on the actual `|o_y|/|o|`, on the bend magnitude against
  `MIN_BEND_OFFSET_M`, and on joint validity, all through the existing Sign
  Contract.

**TYPE 2.** docs/36 wrote that "the Sign Contract disagrees with the full-3D
metric on 87% of corrections". That conflates two different predicates. The
correct statement is: **87% of corrected chain-frames were not historical
>90° flips before enforcement.** The one-bit depth branch being wrong and the
full-3D direction being more than 90° from target are simply different
questions. The measured fact stands unchanged: TYPE 2 corrections improve the
full-3D angle by **−23.9°** on average.

## 3. The local hinge plane

A bend direction is perpendicular to the chain's proximal→distal axis, so it
lives in a 2D plane with two natural axes:

```
a_hat = normalize(distal - proximal)
u     = e_y - a_hat * dot(a_hat, e_y)          u_hat = normalize(u)
v_hat = normalize(cross(a_hat, e_y))
```

The `v_hat` sign convention is fixed by that construction and was **not**
chosen from any measured result.

`{u_hat, v_hat}` is orthonormal and spans the perpendicular plane, so
`o = c_depth·u_hat + c_screen·v_hat` reconstructs exactly (tested to 1e-12).

**`sign(c_depth)` is the existing hinge sign.** `o·u_hat = o_y/√f` and
`√f > 0`, so the two agree wherever the Sign Contract finds the field readable
— verified over hundreds of random poses. The decomposition therefore
introduces no new sign; it only *names* the complementary one.

**`v_hat` is visible geometry.** `cross(a, e_y)` has zero Y component for
**any** `a`, so `v_hat` lies exactly in the canonical X/Z image plane. The
complementary bend side is perpendicular to camera depth — it is not hidden
evidence at all.

**None of this is added to SignState.** It is a diagnostic instrument.

## 4. The 2D side predicate, derived not fitted

The image→canonical mapping was **measured** from `bank_3dpw_paired_v2` rather
than assumed. Over all 7,076 test frames, per-frame Pearson r:

| pair | median r | sign consistency |
|---|---|---|
| `input_2d x` vs canonical X | **+0.975** | positive on **100%** of frames |
| `input_2d y` vs canonical Z | **−0.992** | negative on **100%** of frames |
| `input_2d x` vs canonical Z | +0.014 | — |
| `input_2d y` vs canonical X | −0.023 | — |

So `(X, Z) = (+x, −y)`: the standard image convention with y downward, with no
cross-coupling. Recorded as `IMAGE_TO_CANONICAL`.

Then the predicate follows analytically. Because the axis-parallel terms
cancel,

```
c_screen = (a_X·r_Z − a_Z·r_X) / √f          a = distal − proximal, r = middle − proximal
```

and `√f > 0`, so `sign(c_screen)` is the 2D cross-product side of the middle
joint about the proximal→distal line in the image plane. Substituting the
measured mapping gives, in raw observation coordinates,

```
observed_screen_side = sign( (d_y − p_y)(m_x − p_x) − (d_x − p_x)(m_y − p_y) )
```

A pure orientation predicate: no magnitude, no learned feature, no RGB, no
ground truth. Only `input_2d` and `input_valid`. **Exact collinearity alone is
unresolved** — introducing a collinearity tolerance would be a policy threshold
chosen from results, so the continuous cross magnitude is reported instead and
a test pins that no tolerance exists in the code.

`input_2d` is a benchmark detector observation and is **never** treated as an
oracle: it is the source under test, measured against the target's own
`c_screen` sign.

## 5. The axis-normalized counterfactual

Historical hinge error mixes bend-side error with limb-axis orientation error.
`axis_transport` takes the minimal rotation mapping the target axis onto the
predicted axis and carries the target bend direction through that same
rotation, then compares. It asks: *if both had the same limb axis, would the
bend still be on the wrong side?*

Two tests pin what it must and must not do:

- A **swing** about an axis perpendicular to the limb is removed **completely**
  (axis-normalized error 0 to the precision of `arccos`).
- A **twist about the limb axis** leaves the axis exactly where it was and
  genuinely moves the bend to another side. The minimal rotation there is the
  identity, so the counterfactual **preserves that error in full** rather than
  laundering real side error into "axis geometry".

Antiparallel axes admit no unique minimal rotation and are refused explicitly;
degenerate axes and vanishing bends are refused rather than smoothed. No
tolerance was selected from the measured result — the existing
`VECTOR_NORMALIZATION_EPS` is reused for the degeneracy guards.
