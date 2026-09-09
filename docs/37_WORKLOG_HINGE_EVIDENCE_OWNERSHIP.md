# Worklog — Who Owns the Residual Hinge Disagreement (2026-09-08)

> **SUPERSEDED IN PART BY docs/38 (2026-09-09).** Two attribution gaps were
> found after this batch closed and its ownership conclusion is stronger than
> the evidence supports. Read this document together with docs/38.
>
> **Downgraded to PROVISIONAL** — do not cite these as settled:
> "Outcome B failed", "Outcome C is refuted", "the complementary side is
> directly observable from current `input_2d`", "no additional VLM evidence can
> be useful". The correct standing statement is: *the complementary component
> lies in a camera-parallel direction, but its observability from the current
> perspective 2D contract is unresolved.*
>
> **Two specific defects.** (1) The `residual_flip_frames` accuracy in Section 7
> was computed over all **279** residual historical flips, not the **83**
> depth-correct residuals the central question is about — a population
> mismatch. (2) Section 4 identifies the canonical X/Z line side with the
> observed image line side, but the observation is a *perspective* projection
> and the two signs are **not** algebraically identical; see docs/38 Section 3.
>
> **Still valid:** the local u/v hinge-plane decomposition; `sign(c_depth)`
> matching the existing hinge SignState where readable; the axis-normalized
> counterfactual; the finding that the 83 depth-correct residuals are strongly
> associated, *in target-3D space*, with complementary local-plane disagreement
> and/or limb-axis error; that no evidence justifies immediately adding another
> SignState bit; and `MINIMUM_NORM`'s measured X/Z ownership cost.

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

## 6. Residual-flip ownership

Source: the accepted stored `O_BILATERAL` prediction with the `MINIMUM_NORM`
oracle-depth hinge correction. 279 residual full-3D flips, reproducing docs/35's
count exactly.

"The axis is substantially different" is operationalized **without a threshold**:
the axis is implicated exactly when aligning the two limb axes is enough to
remove the >90° flip.

| class | all 279 flips | the **83** where the depth bit is already correct |
|---|---|---|
| A — screen side only | 114 (40.9%) | 14 (16.9%) |
| B — axis geometry only | 8 (2.9%) | 6 (7.2%) |
| C — both | 60 (21.5%) | **59 (71.1%)** |
| D — unexplained | 97 (34.8%) | **4 (4.8%)** |

The 83-frame column is the one the question is about: it is exactly the
satisfied-depth-sign-but-still-flipped set docs/35 identified. There:

- **screen side implicated: 88.0%** (A + C)
- **axis geometry implicated: 78.3%** (B + C)
- **unexplained by either: 4.8% — 4 chain-frames out of 28,304**

The two explanations overlap heavily (71.1% are both), which is expected: a
badly reconstructed limb axis and a wrong in-plane side are not independent
failures of the same chain.

The remaining 196 flips are frames where the depth bit itself is not correct —
docs/34's unreadable and never-requested categories — and are not evidence about
what happens *after* the hidden branch is right.

Axis-angle distributions, reported continuously as required (median / p90 / max):
left elbow 12.8° / 37.2° / 131.3°, right elbow 12.3° / 34.2° / 99.9°,
left knee 6.5° / 18.6° / 85.5°, right knee 6.5° / 18.3° / 64.1°.

## 7. Can `input_2d` actually supply the complementary side?

Balanced accuracy of `observed_screen_side` against the target's own
`c_screen` sign. Zero frames were unresolved — no exact collinearity and no
missing chain joint occurred in 7,076 frames.

| field | all frames | **residual-flip frames** | TYPE 1 | TYPE 2 |
|---|---|---|---|---|
| left_elbow | 0.905 (n=6003) | **0.642** (n=132) | 0.780 (n=73) | 0.993 (n=333) |
| right_elbow | 0.897 (n=6076) | **0.600** (n=52) | 0.833 (n=31) | 0.978 (n=275) |
| left_knee | 0.791 (n=6497) | **0.412** (n=50) | 0.444 (n=36) | 0.885 (n=232) |
| right_knee | 0.812 (n=6519) | **0.283** (n=45) | 0.412 (n=39) | 0.909 (n=371) |

**The observation reads the side well in general and fails precisely where it
would be needed.** On ordinary frames 0.79–0.91; on TYPE 2 corrections
0.89–0.99; on the residual-flip frames 0.28–0.64 — at or below chance.

The line-side magnitude explains why for the elbows. Median |cross| on
residual-flip frames against all frames: left elbow **0.185×**, right elbow
**0.105×** — the observed chain is roughly an order of magnitude closer to
collinear than typical. The target's own `|c_screen|` collapses too (0.25–0.49×
the usual), so the true bend really is close to the depth plane there. That is
a **projective degeneracy**, not simply detector error.

The knees behave differently: their |cross| ratios are 0.81× and 1.15×, i.e.
not degenerate at all, yet balanced accuracy is 0.41 and 0.28. There the
observed side is well-determined and **disagrees** with the target — genuine
2D/3D inconsistency, on small counts (n = 50 and 45).

`input_2d` is a benchmark detector observation and is not treated as an oracle
anywhere in this accounting.

## 8. Minimum-norm and Geometry Core position ownership

The `MINIMUM_NORM` policy gives up exact X/Z preservation of the middle joint.
Measured on the four hinge middle joints:

| field | corrections | XZ move p50 | p90 | max | target XZ error mean, before → after |
|---|---|---|---|---|---|
| left_elbow | 406 | 11.7 mm | 67.1 mm | 403.0 mm | 48.45 → 59.82 (**+11.37**) |
| right_elbow | 306 | 11.3 mm | 45.7 mm | 438.2 mm | 51.91 → 63.69 (**+11.78**) |
| left_knee | 268 | 9.5 mm | 58.1 mm | 145.5 mm | 39.73 → 45.42 (**+5.69**) |
| right_knee | 410 | 7.7 mm | 25.9 mm | 107.4 mm | 41.62 → 38.50 (−3.12) |

The typical X/Z movement is small (8–12 mm median), but **on three of four
chains it moves the joint away from the target in the image plane**, by 5.7 to
11.8 mm on average, with tails to 438 mm.

This is a genuine ownership cost and it is invisible in docs/36's total MPJPE,
which improved: that improvement came from the depth axis, while the
image-plane component got worse. The historical `DEPTH_ONLY` operator leaves
X/Z **exactly** unchanged and therefore has no such cost.

So the answer to the second central question is **yes, partly**: minimum-norm
buys its bounded conditioning by writing into geometry the Geometry Core owns,
and on this evidence it degrades that geometry on most chains.

## 9. 2D observation consistency: UNRESOLVED

`FrameSample` carries `image_size` and nothing else about the camera — no
intrinsics, no focal length, no principal point — and `target_3d` is
root-relative metres. There is therefore **no canonical, valid mapping from the
3D output into observation coordinates** in the existing contracts, and exact
2D reprojection consistency **cannot be computed**. It is reported as
unresolved rather than approximated.

The correlational mapping in Section 4 establishes axis correspondence and
sign; it is **not** a calibrated projection, and X/Z metric distance is
deliberately **not** substituted for pixel reprojection error anywhere in this
worklog.

## 10. Ownership classification: **D — mixed**, with a decisive negative on C

**Outcome C (true hidden-evidence gap) is refuted.** Once the hidden depth
branch is correct, only **4 of 83** residual flips — 4.8%, and 4 chain-frames
out of 28,304 — are unexplained by visible screen-side geometry or limb-axis
reconstruction. There is no meaningful pool of failures left over for a second
hidden bit to claim.

**Outcome A (geometry-axis owned) is partly supported but not sufficient
alone.** The axis is implicated in 78.3% of the 83, but aligning the axes
removes the flip by itself in only 7.2%; in 71.1% the screen side is wrong too.

**Outcome B (geometry-2D-side owned) fails its second condition.** Its first
half holds strongly — the residual corresponds to screen-side disagreement in
88.0% of cases, and `v_hat` is provably in the image plane, so that side is
visible geometry rather than hidden evidence. But B also requires that
`input_2d` *reliably provides* that side, and it does not: 0.28–0.64 balanced
accuracy exactly on the frames in question, against 0.79–0.91 overall.

So the honest classification is **D**, and the useful statement is the
decomposition rather than the letter:

> The residual hinge disagreement is **geometry-owned, not advisor-owned** — it
> lives in the visible in-image-plane side and in limb-axis reconstruction. But
> the current 2D observation cannot be read as a drop-in supplier of that side,
> because the frames where it is needed are precisely the frames where the
> projection is near-degenerate or the detector disagrees with the target.

## 11. The conditional two-evidence candidate did NOT execute

Section 11 of the DIRECTION permits it **only if outcome B is supported**. B's
second condition fails by a wide margin, so no two-evidence constraint was
implemented, no new candidate was added, and no replay of `H2_ORACLE_SCREEN` or
`H2_OBSERVED_SCREEN` was run.

Building it anyway would have installed a screen side taken from a source
measured at chance on the target frames, and the oracle-screen upper bound
would have reported a benefit no observable input can deliver. The gate exists
for exactly this case.

## 12. Is any additional VLM or SignState evidence justified? **No**

Not on this evidence:

- 95.2% of the depth-correct residual flips are already accounted for by
  visible geometry the Geometry Core and Geometry Observation own.
- The complementary bend side is **provably** perpendicular to camera depth —
  `cross(a_hat, e_y)` has zero Y component for any axis — so it is not hidden
  evidence by construction, whatever a sensor could be asked for.
- The advisor's one hidden depth bit is doing its job: on the 83 frames it is
  satisfied and the failure is elsewhere.

**No SignState field was added, no VLM output was designed, and the Sign
Contract is unchanged.** The next question this opens is a Geometry
Observation / Geometry Core question — how the in-plane side survives
foreshortening — and it is explicitly *not* opened here.

## 13. Review exports

`hinge_ownership_v3/hinge_evidence_review.json`, five buckets, each carrying
`sample_id`, sequence, frame index, source image path, the chain's joint names,
requested and final depth sign, predicted and target `c_screen` sign, the
observed 2D side with its continuous cross magnitude, axis angle, historical and
axis-normalized hinge error, source and corrected middle-joint XYZ, and the raw
`input_2d` chain coordinates:

- `depth_correct_but_screen_side_wrong`
- `both_sides_correct_but_still_flipped`
- `large_axis_angle_error`
- `large_minimum_norm_xz_movement`
- `observed_2d_side_disagrees_with_target`

Every residual-flip record (all 279) is also written in full under
`fields.*.residual_flip_records`, so no case is visible only through a sampled
bucket. Nothing here is declared correct on visual ambiguity; the export exists
for inspection.

## 14. Preservation and confirmation

docs/33-36 and every artifact are untouched: `hybrid_signstate_v1`/`_v2`,
`hinge_policy_depth_only_v1`/`minimum_norm_v1`, `branch_constraint_v3`,
`branch_constraint_dep_*`, `constraint_attribution_v*`, all `O_*`/`H_NO_*`/`L_*`
results, the wrong-sign endpoints and the residual-hinge attribution. This batch
writes new lineages only (`hinge_ownership_v1`…`v3`).

`DEPTH_ONLY` remains the default hinge write policy; `MINIMUM_NORM` remains a
research candidate. **No default was changed**, and Section 8 gives a concrete
reason not to promote it yet.

No sensor was chosen or built. No VLM was called, no prompt written, no
classifier trained, no RGB read, no temporal context used. No pose model was
trained or fine-tuned. `UNIT_FORWARD_EPSILON`, `MIN_BEND_OFFSET_M` and every
other Sign Contract threshold are unchanged; no sign variable, hinge vector,
continuous angle or magnitude was added. No screen-side tolerance exists, and no
X/Z clamp was introduced.

Tests: 10 new hinge-plane contracts. Shared production code was added
(`src/framepose/hinge_plane.py`), so full regression ran at closure —
**506 passed, 40 skipped**.

LabServer63, `animcv-framepose:cuda118`, repo mounted read-only. No GPU work.

## 15. Completion

> **After the hidden depth sign is correct, does the remaining hinge ambiguity
> belong to the Sign Advisor at all?**

**No.** Of the 83 chain-frames where the advisor's depth bit is satisfied and
the full-3D metric still calls the bend flipped, 88.0% involve a wrong
in-image-plane side, 78.3% involve a substantially different limb axis, 71.1%
involve both, and **4.8% are unexplained by either**. The complementary side is
provably perpendicular to camera depth, so it is visible geometry by
construction. The residual belongs to **continuous axis geometry and visible 2D
topology**, not to hidden evidence.

That does **not** make it free to collect. The current 2D observation supplies
the side at 0.79–0.91 balanced accuracy in general but only 0.28–0.64 on the
frames that still fail, because those frames are near-degenerate in projection
(elbow line-side magnitude collapses to ~0.1–0.19× typical) or genuinely
inconsistent with the target (knees). So the answer is *geometry-owned but not
yet geometry-solved*.

**STOP.** No richer VLM contract is designed, no SignState field is added, no
sensor is selected, and no default is promoted.
