# Worklog — Conditioning of the Hinge Branch Constraint (2026-09-08)

> One bounded diagnostic batch on `arch/single_frame_first`. **No training, no
> sensor, no VLM, no RGB, no Sign Contract change, no threshold tuning, no cap,
> no clamp, no Geometry Core change.** The historical hinge operator is frozen
> and was not modified.
>
> The question docs/35 left open: **why can a CORRECT oracle hinge sign require
> a multi-metre depth correction, and is that an operator defect or an inherent
> domain limit of hard one-bit hinge enforcement?**

## 1. Starting/final HEAD

Start `86f23e2` (docs/35 closed). This batch: `81583f0` (conditioning
diagnostic), `5b450be` (minimum-norm write policy + contracts), `741b130`
(replay wiring), plus the run and worklog commits.

## 2. docs/35 preserved

`hybrid_signstate_v1`/`_v2`, the H0/H1 results for `S0`, `O_HIP` and
`O_BILATERAL`, the wrong-sign endpoints, the residual-hinge attribution and the
real-frame review are untouched. docs/33, docs/34 and docs/35 stand unchanged.
This batch writes new lineages (`hinge_conditioning_v1`,
`hinge_policy_*_v1`).

Accepted findings carried forward: hybrid composition works; bilateral/yaw
ownership is exactly preserved; explicit hinge constraints **transferred across
all three tested FramePose sources**; the one-bit hinge sign is useful but
incomplete; wrong hinge advice remains locally catastrophic.

Two wordings are corrected in this batch's own usage: `H1_BILATERAL_HINGE` is
the **best measured combined configuration on the targeted orientation/hinge
metrics**, not the universally "best pose"; and the hinge mechanism
**transferred across all three tested sources**, which is not the same as being
universally source-independent.

## 3. The operator stayed frozen

`_hinge_correction` is byte-for-byte unchanged and remains the default. A test
pins its two defining lines. `UNIT_FORWARD_EPSILON` is unchanged; no delta cap,
millimetre threshold, clamp, scaling or interpolation was added anywhere.

## 4. Exact conditioning accounting

`scripts/diagnose_hinge_conditioning.py` records, for every corrected hinge
chain-frame on all three sources, the geometry the closed form is built from:
`f`, `√f`, the offset vector and its depth component, `1/f` and `2/f`, the
resulting `δ_y`, the limb axis, its length and its angle to the camera depth
direction, the before/after full-3D hinge error and flip state, the read-back
signs, and the frame MPJPE before/after.

Corrections: 1,388 (`S0`), 1,584 (`O_HIP`), 1,390 (`O_BILATERAL`).

| `O_BILATERAL` | p50 | p90 | p99 | p99.9 | max |
|---|---|---|---|---|---|
| \|δ_y\| | 62.0 mm | 191.6 mm | 430.8 mm | 1,756.9 mm | 3,108.4 mm |

`√f` is near 1 for the overwhelming majority of corrections (p50 0.979), so the
amplification factor is usually inert — the tail is where it is not.

## 5. What drives the tail — a two-regime answer

The closed form is multiplicative, so the honest scale is logarithmic:

```
log10|δ_y| = log10(2|o_y|) + log10(1/f)      exactly (max residual 4.4e-16)
```

Variance decomposition of `log10|δ_y|` (`O_BILATERAL`, total variance 0.1422):

| term | variance | share |
|---|---|---|
| `log10(2|o_y|)` | 0.1231 | **86.6%** |
| `log10(1/f)` | 0.0156 | 11.0% |
| 2 × covariance | 0.0034 | 2.4% |

Correlations against `log10|δ_y|`: `|o_y|` **r = +0.824**, `|o|` +0.647,
`√f` −0.358, axis-to-depth angle −0.301.

**But the bulk and the tail have different causes**, and conflating them would
be wrong:

| band (exported records) | n | median `√f` | median \|o_y\| |
|---|---|---|---|
| ≤ 200 mm | 274 | 0.977 | 57.0 mm |
| > 200 mm | 126 | 0.971 | **116.6 mm** |
| > 500 mm | 12 | **0.290** | 57.6 mm |
| > 1,000 mm | 6 | **0.256** | 48.5 mm |

So the answer to "A, B, C or D" is **C, with a regime split**:

- The **overall scale** of corrections is set by **B — the size of `|o_y|`**. A
  moderate 200-500 mm correction is usually an honest reflection of a genuinely
  large bend, at `√f ≈ 0.97`. That is not pathological.
- The **catastrophic excursions** are set by **A — `f` approaching its
  permitted lower limit**, at a *below-average* `|o_y|`. Every correction
  beyond 500 mm sits at `√f ≈ 0.26-0.29`, roughly a quarter of the typical
  value, while its offset is smaller than the median.

The `>200 mm` band therefore contains two different populations, which is
exactly why docs/35's 200 mm review range was the wrong lens. Nothing here
branches on those ranges; they are reported only for continuity.

## 6. TYPE 1 vs TYPE 2

Every correction classified by what the full-3D metric said *before* it:

| source | TYPE 1 (metric also flipped) | TYPE 2 (metric already acceptable) |
|---|---|---|
| `S0` | 222 (16.0%) | 1,166 (84.0%) |
| `O_HIP` | 206 (13.0%) | 1,378 (87.0%) |
| `O_BILATERAL` | 179 (12.9%) | 1,211 (87.1%) |

TYPE 3 (metric unavailable): none — every corrected chain-frame had a
computable metric.

**The one-bit contract disagrees with the full-3D metric on 87% of the frames
it corrects.** That sounds damning until the effect is measured:

| type | mean full-3D hinge error change | \|δ_y\| p50 | \|δ_y\| max |
|---|---|---|---|
| TYPE 1 | **−70.6°** | 75.5 mm | 673.6 mm |
| TYPE 2 | **−23.9°** | 59.6 mm | **3,108.4 mm** |

TYPE 2 corrections still **improve** the full-3D hinge error by 24° on average.
They are not the contract being wrong; they are bends whose depth side was
wrong while the overall 3D direction was still inside 90°, and fixing the depth
side helped. docs/35's 7 "broken by enforcement" cases are the small minority
of TYPE 2 that went the other way.

**But the catastrophic tail lives entirely in TYPE 2.** All six corrections
above 1 m are TYPE 2, with pre-correction hinge errors of 20-26° — already
good bends — driven to 70-124° afterwards. So the answer to the DIRECTION's
question is **yes**: the catastrophic corrections are concentrated where the
one-bit contract disagrees with an already-acceptable full-3D pose *and* the
geometry is near-singular. Neither condition alone produces them.

## 7. The analytic explanation

Let `a = distal − proximal`, `o` the perpendicular bend offset, and

```
u = e_y − a·a_y/|a|²          the part of the depth direction perpendicular to a
```

Three exact identities (verified numerically to 1e-12):

```
|u|² = f = u_y                                              (1)
o·u = o_y                                                   (2)
|o'| = |o|   after the reflection                           (3)
```

Moving the middle joint by `δ` in depth changes the offset by `δu`, so
`o'_y = o_y + δf`, giving the historical `δ = −2o_y/f`. By (3) the depth-only
reflection **preserves the offset magnitude exactly** — it is a genuine
reflection in the plane perpendicular to the axis, not an approximation.

### Observability is NOT what fails

For any `w ⊥ a`, `w_y = √f·(w·û)` where `û = u/√f`, so

```
max over w ⊥ a of |w_y| / |w|  =  √f          (attained, verified)
```

`√f` is therefore **exactly** the largest depth fraction any offset on this
limb can have. The existing guard `√f < UNIT_FORWARD_EPSILON` is not an
arbitrary threshold — it is precisely the condition under which the requested
branch is **unreadable for every possible position of the middle joint**. Below
it no branch exists to install; above it the sign is fully observable. A test
pins both halves.

**So the requested Y sign remains mathematically observable throughout the
admissible domain.** Observability does not degrade as `f → 0` within the
guard; it is a sharp cutoff, and the guard sits exactly on it.

### Conditioning of the chosen operator is what fails

Decompose the historical displacement:

```
δ·e_y  =  δ·u  +  δ·a_y·a/|a|²
          \____/   \____________/
        does the work   pure slide along the bone
```

The second term is **parallel to the limb axis**. Sliding the middle joint
along its own limb changes neither the perpendicular offset nor the bend
direction nor the branch — it is geometrically inert for the quantity being
constrained. Its length is `|δ|·|a_y|/|a|`, which diverges as the limb aligns
with the depth axis even though the useful term stays bounded.

Concretely: `|δ·u| = 2|o_y|/√f ≤ 2|o|` always, because `|o_y| ≤ √f|o|`. The
useful part of the displacement is **bounded by twice the bend offset for every
admissible geometry**. Only the inert part is unbounded.

**Observability and conditioning are therefore separate, and only the second
fails.** The +3.108 m correction moved an elbow 3.108 m to accomplish a
406 mm change of offset; the remaining 2.7 m slid the elbow along its own arm.

## 8. Is depth-only ownership the cause? Yes

The historical operator additionally requires the middle joint's X and Z to be
unchanged. That is exactly what forces the displacement to lie along `e_y`, and
therefore exactly what admits the inert axis-parallel component. Relaxing it —
while still holding proximal and distal fixed — permits a strictly smaller
displacement that reaches the **same** target offset.

Minimum-norm displacement reaching a required offset change `Δo` (with
`Δo ⊥ a`) is `Δo` itself, since any component along `a` is wasted. For the
historical target that is `δ·u`, exactly `√f` times shorter than `δ·e_y`.

## 9. The minimum-norm candidate

```
correction = δ · u,        δ = −2·o_y/f      (δ unchanged from the historical operator)
```

No new constant, no cap, no clamp, no step size, no tolerance. A test asserts
the executable body contains only the literals `2.0`, `1.0`, `0.0` and the one
pre-existing `1e-12` degeneracy guard.

### Invariants, stated exactly

| property | depth-only (historical) | minimum-norm |
|---|---|---|
| proximal / distal positions | preserved | preserved |
| requested branch installed | yes | yes |
| **bend direction** (unit vector) | — | **identical to depth-only, to 1e-12** |
| full-3D hinge error after | — | **identical to depth-only** |
| perpendicular offset magnitude \|o\| | preserved exactly | preserved exactly |
| **both bone lengths** | **NOT preserved** | **preserved exactly** |
| along-axis position of the middle joint | not preserved | preserved exactly |
| middle joint screen-space X/Z | **preserved exactly** | **NOT preserved** |
| displacement length | `2|o_y|/f`, unbounded | `2|o_y|/√f ≤ 2|o|`, **bounded** |

The two operators produce the **same** constrained geometry and differ only in
where the middle joint ends up along an axis that the Sign Contract does not
read. The minimum-norm variant is not a different answer to the branch
question; it is the same answer without the inert excursion.

The exchange it makes is explicit: it gives up exact screen-space preservation
of the middle joint — which the depth-only operator did guarantee — and gains
exact bone-length preservation, which the depth-only operator never had.

## 10. Synthetic contracts

Fixtures span geometry, not fitted performance: a left-arm chain whose axis is
swept from lying in the image plane to nearly aligned with the camera depth
axis. Pinned (`tests/test_frame_pose_hinge_conditioning.py`, 9 tests):

- `√f` is exactly the attained maximum of `|w_y|/|w|` over all `w ⊥ a`.
- Below the guard the branch is **necessarily** unreadable, so both policies
  refuse identically; the guard never rejects a readable request.
- The depth-only displacement is exactly `1/√f` times the minimum-norm one, and
  the entire difference is parallel to the limb axis (`|cos| = 1` to 1e-9).
- `|correction| ≤ 2|o|` for the minimum-norm policy on every admissible
  geometry; the ratio is 1.0 when the limb lies in the image plane and grows
  past 4× by the edge of observability.
- The minimum-norm policy installs the identical unit bend direction (1e-12).
- Offset magnitude preserved by both; **bone lengths preserved only by
  minimum-norm**; **screen-space X/Z preserved only by depth-only**.
- Determinism, no banned construct (`clip`/`clamp`/`minimum`/`maximum`/`cap`)
  in the executable body, and only the literals `2.0`, `1.0`, `0.0` plus the
  one pre-existing `1e-12` guard.
- `_hinge_correction`'s two defining lines are pinned, and the default policy
  still routes to it.

Synthetic PASS is not real-scene viability, which is why Section 11 exists.

## 11. Real replay: `H1_DEPTH_ONLY` vs `H1_MIN_NORM`

Same stored `O_BILATERAL` prediction, same oracle hinge signs, no training.
`H1_DEPTH_ONLY` reproduces docs/35's artifact **byte-identically**.

### Hinge semantics: bit-identical, as the mathematics requires

| | H0 | `H1_DEPTH_ONLY` | `H1_MIN_NORM` |
|---|---|---|---|
| hinge flip | 0.0189 | **0.0116** | **0.0116** |
| hinge MAE° | 24.099 | **22.129** | **22.129** |
| left elbow / right elbow° | 28.717 / 23.046 | 26.4693 / 21.7391 | **26.4693 / 21.7391** |
| left knee / right knee° | 21.907 / 21.442 | 20.6017 / 19.6486 | **20.6017 / 19.6486** |
| all seven sign agreements | — | — | **identical** |
| fixed / broken / stayed-flipped | — | 175 / 7 / 4 | **175 / 7 / 4** |

Every hinge quantity is identical to the last digit — the two operators install
the same geometry, exactly as derived.

### Bilateral ownership: preserved exactly by both

Root yaw 7.999 → 7.999, yaw P95 18.645 → 18.645, both depth residuals and all
three non-hinge sign agreements unchanged, at the oracle **and** the
opposite-oracle endpoint, under both policies. The minimum-norm policy writes X
and Z of the four hinge middle joints — the joint-level ownership boundary the
bilateral channel actually depends on is unchanged and still enforced.

### The tail collapses

| correction norm | `H1_DEPTH_ONLY` | `H1_MIN_NORM` |
|---|---|---|
| p50 | 62.02 mm | 59.63 mm |
| p90 | 191.62 mm | 177.44 mm |
| p99 | 430.77 mm | 338.98 mm |
| **p99.9** | **1,756.86 mm** | **458.18 mm** |
| **max** | **3,108.43 mm** | **672.20 mm** |
| mean | 93.41 mm | 80.75 mm |

The multi-metre excursion is **gone**: 3,108 mm → 672 mm, a 4.6× reduction at
the maximum and 3.8× at p99.9, with no cap, clamp or threshold anywhere. The
remaining 672 mm maximum is an honest reflection of a genuinely large bend and
respects the analytic `2|o|` bound.

Per-axis: depth-only moves `|δy|` up to 3,108 mm and `|δx| = |δz| = 0` by
construction; minimum-norm moves at most 671 mm in depth, 265 mm in X and 378
mm in Z.

### Position cost: strictly better

| | `H1_DEPTH_ONLY` | `H1_MIN_NORM` |
|---|---|---|
| MPJPE | 79.366 mm | **79.282 mm** |
| PA-MPJPE | 56.694 mm | **56.543 mm** |
| wrong-sign MPJPE | 93.561 mm | **92.099 mm** |
| wrong-sign PA-MPJPE | 75.192 mm | **72.646 mm** |

Per-joint, the elbows — which paid for the branch correction in docs/35 —
improve: left elbow 107.914 → 107.244, right elbow 91.745 → 90.942. Right knee
improves 83.094 → 82.906; left knee gives back 82.025 → 82.211. Net better on
both guardrails, and materially safer under wrong advice.

## 12. Not preferred merely for being smaller

Per the DIRECTION, a smaller displacement is not itself sufficient. The
candidate answers all three questions:

- **Does it install the requested branch?** Yes — identically, verified by
  bit-identical sign agreement and per-chain hinge error.
- **Does it improve the historical hinge metric?** Identically: 175 fixed, 7
  broken, same as the historical operator. It neither helps nor harms the
  semantic question.
- **What geometry does it sacrifice?** Exact screen-space X/Z of the middle
  joint, which the depth-only operator preserved. In exchange it preserves both
  bone lengths exactly, which the depth-only operator never did.

## 13. Classification: **operator defect**, with a separate unchanged domain limit

**The conditioning failure is an operator defect, and it is fully resolved.**
The requested Y sign is observable throughout the admissible domain — `√f` is a
sharp cutoff and the existing guard sits exactly on it — so nothing about the
branch question is ill-posed. What was ill-conditioned was the *choice* to
write only in depth, which forced a geometrically inert slide along the limb
whose length diverges as the limb aligns with the camera axis. Removing that
term is parameter-free, changes no threshold, and leaves the constrained
geometry identical.

**A distinct limit remains, and it is not this one.** The catastrophic frames
were TYPE 2: the one-bit contract disagreed with a full-3D pose that was
already acceptable (21-26° before, 70-124° after). The minimum-norm operator
does **not** fix that — it produces the identical bend direction — because the
disagreement lives in the Sign Contract's one-bit encoding, which docs/34 and
docs/35 already measured as incomplete and which this batch was directed not to
change. Fixing the conditioning removes the *positional* catastrophe entirely
and leaves the *semantic* one untouched.

So the answer is neither "just an operator defect" nor "an inherent domain
limit of hard enforcement" — it is both, cleanly separated:

| failure | nature | status |
|---|---|---|
| multi-metre displacement from a correct sign | operator defect (depth-only write policy) | **resolved**, parameter-free |
| correct one-bit branch, worse full-3D pose | Sign Contract expressiveness | **unchanged**, out of scope |

**No production boundary was invented.** No `f` threshold, no `δ_y` threshold,
no confidence threshold was chosen. The minimum-norm policy is **not** promoted
to the default — `depth_only` remains the default so every historical result
reproduces — and the hybrid is **not** declared production-ready. Whether to
switch the default is a separate policy decision on evidence this batch does
not have.

## 14. Confirmation and environment

No sensor was chosen or built. No VLM, no RGB, no prompt, no confidence, no
temporal context, no sign-accuracy sweep. No pose model was trained or
fine-tuned. `UNIT_FORWARD_EPSILON`, `MIN_BEND_OFFSET_M` and every other Sign
Contract threshold are unchanged, and no sign variable was added. No cap,
clamp, clip or tuned coefficient exists in the new code. The Geometry Core and
the learned bilateral conditioning are untouched.

Tests: 9 new conditioning contracts plus the existing 11 composition contracts;
shared production code was modified (`branch_constraints.py`), so full
regression ran at closure — **496 passed, 40 skipped**.

LabServer63, `animcv-framepose:cuda118`, repo mounted read-only. No GPU work.

## 15. Completion

> **Why can a correct oracle hinge sign require a catastrophic correction under
> the current hard constraint?**

Because the operator writes only in depth. The depth direction decomposes into
a part perpendicular to the limb (which changes the bend) and a part parallel to
it (which cannot). As the limb aligns with the camera depth axis the parallel
part dominates, so the operator slides the middle joint metres along its own
arm to achieve a sub-metre change of bend offset. Observability never fails —
`√f` bounds the readable depth fraction and the existing guard sits exactly on
that bound — only the conditioning of the chosen write does.

> **Can that be fixed by a principled parameter-free geometric operator, or does
> it define an inherent domain limit?**

It can be fixed. Writing `δ·u` instead of `δ·e_y` — same `δ`, same target
offset, same branch, same bend direction, same full-3D hinge error — is exactly
`√f` times shorter, bounded by `2|o|` for every admissible geometry, and
preserves both bone lengths exactly. On real frames it cuts the maximum
correction from 3,108 mm to 672 mm and improves MPJPE, PA-MPJPE and the
wrong-sign endpoint, with every hinge metric bit-identical.

What it does not fix, and what therefore remains the open limit of hard one-bit
hinge enforcement, is the 87% of corrections where the contract disagrees with
an already-acceptable full-3D pose — a Sign Contract expressiveness question,
not an operator question.

**STOP.** No sensor selection. The SignState integration architecture track is
not yet closed: the conditioning blocker is resolved, the encoding limit is not.
