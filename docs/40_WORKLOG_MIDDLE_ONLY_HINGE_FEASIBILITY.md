# Worklog — Feasibility of a Middle-Joint-Only Hard Hinge Constraint (2026-09-10)

> Continuation of docs/39 on `arch/single_frame_first`. This batch preserves
> the existing SignState contract, `DEPTH_ONLY`, `MINIMUM_NORM`, the current
> default, and all historical docs/33–39 measurements. It adds only a
> diagnostic geometry audit; it does not add a production write policy.

## 1. Starting/final HEAD

The workspace started and remains at:

```text
branch: arch/single_frame_first
HEAD:   fed84c9 docs/39: close the hinge write-policy observation-ownership batch
```

The worktree contains the new diagnostic and tests described below. No commit
or production default change was made in this batch.

## 2. docs/39 wording corrections

Only wording was corrected; no measured number or historical artifact was
changed.

- “`DEPTH_ONLY` owns the image” now reads: “`DEPTH_ONLY` better preserves
  image-space observation than `MINIMUM_NORM` among the two tested policies.”
  The same paragraph explicitly says that `DEPTH_ONLY` does **not** preserve
  the image exactly.
- “Neither operator is Pareto-optimal” now reads: “The two tested operators
  are non-dominated under the measured objectives; neither satisfies both
  image-space ownership and metric-skeleton ownership simultaneously.”

## 3. Exact current hinge correction contract

For one chain, define `P = proximal`, `M = middle`, and `D = distal`. The
runtime call is `framepose.branch_constraints.apply_branch_constraints` in
`src/framepose/branch_constraints.py`.

Its hinge inputs are exactly:

```text
pose       root-relative canonical (17, 3) prediction
valid      (17,) validity mask
requested  (7,) SignState with values -1, 0, +1
```

The hinge-specific optional input is the existing policy selector
(`DEPTH_ONLY` or `MINIMUM_NORM`). It does not receive `input_2d`, camera
intrinsics, camera extrinsics, absolute root depth, or image size.

The exact middle-only hard-constraint system audited here is:

```text
P' = P
D' = D
M' is the only writable joint

|M' - P| = |M - P|
|D  - M'| = |D - M|

pi_K(M') = pi_K(M)                         exact image position
SignState_hinge(P, M', D) = requested_sign  requested branch
```

The first two equalities are the metric-skeleton ownership requirement. The
projection equality is under a calibrated perspective camera. For camera
coordinates with depth on `Y`, one representative pinhole form is:

```text
pi_K(X, Y, Z) = (fx X/Y + cx, -fy Z/Y + cy)
```

The actual Sign Contract is not a continuous bend target. For a non-degenerate
chain it reads the sign of the `+Y` component of the unit bend direction and
returns `UNKNOWN` at its existing readability floors. It does not require a
particular bend magnitude, screen-plane side, or full reflected direction.

## 4. Bone-preserving feasible set

Let:

```text
a  = D - P
L  = |a|
rP = |M - P|
rD = |M - D|
```

The two exact bone constraints are two spheres:

```text
S(P, rP) = {X : |X-P| = rP}
S(D, rD) = {X : |X-D| = rD}
```

For `L > 0`, define:

```text
lambda = (rP² - rD² + L²) / (2 L²)
C      = P + lambda a
Pi     = {X : (X-C) dot a = 0}
rho²   = rP² - lambda² L²
```

When `|rP-rD| < L < rP+rD`, `rho² > 0` and the feasible set is exactly the
circle:

```text
{ C + rho q : q dot a = 0, |q| = 1 }
```

Thus the circle centre is `C`, its plane is perpendicular to `D-P` and passes
through `C`, and its radius is `rho`. The middle joint has one remaining
degree of freedom: an angle around this circle.

The diagnostic handles the requested degeneracies explicitly:

| condition | feasible set |
|---|---|
| `L > rP+rD` or `L < |rP-rD|` | empty / inconsistent spheres |
| `L = rP+rD` or `L = |rP-rD|` | one tangent point, `rho = 0` |
| `L = 0`, `rP != rD` | empty concentric spheres |
| `L = 0`, `rP = rD > 0` | a sphere, not a circle |
| `L = 0`, `rP = rD = 0` | one singleton point |

No empirical geometry threshold is used for these cases. The implementation
is `scripts/diagnose_hinge_feasibility.py:bone_preserving_locus`.

## 5. Camera-ray intersection

Exact image-position preservation means that `M'` lies on the same camera ray
as `M`:

```text
R_M = { O + t v : t >= 0 }
```

where `O` is the camera centre and `v` is the direction through `M`. Intersect
the ray with the circle plane:

```text
a dot (O + t v - C) = 0
t = a dot (C-O) / (a dot v)
```

### Generic case

If `a dot v != 0`, the ray meets the plane at exactly one point. The existing
`M` is already in both the circle and the ray, so that point is `M`. Therefore
the ray/circle intersection is exactly `{M}`. If the requested branch is the
opposite of the current readable branch, `M` cannot satisfy it and there is no
non-trivial middle-only solution satisfying all four requirements.

This proves the expected generic incompatibility. It is a write-set result,
not a claim that the two existing formulas exhaust all points on the
bone-preserving circle.

### Special parallel case

If `a dot v = 0`, there are two possibilities:

- `a dot (O-C) != 0`: the ray is parallel to and disjoint from the plane;
- `a dot (O-C) = 0`: the ray lies in the plane, and it can meet the circle in
  zero, one tangent, or two points.

Because the ray is defined through the existing `M`, the relevant special case
is the contained ray. A second point is possible only there (and it must also
lie on the forward half-ray and satisfy the requested SignState). The
diagnostic computes this line/circle quadratic directly; it does not assume
that a second point exists.

## 6. Branch-change consequence

On a non-degenerate circle, let `u_hat` be the normalized projection of the
camera-depth direction into `Pi`, and let `v_hat` be the orthogonal in-plane
direction. Every bone-preserving point is:

```text
M'(theta) = C + rho (u_hat cos(theta) + v_hat sin(theta))
```

The Sign Contract's continuous branch quantity is proportional to
`cos(theta)`. Therefore a readable opposite branch normally has an entire arc
of bone-preserving solutions. `MINIMUM_NORM` is one particular point on that
circle: it flips the depth component while preserving the complementary bend
component. The generic camera ray nevertheless selects only the current point
from that circle.

Result:

```text
GENERIC INCOMPATIBILITY PROVED
```

More precisely: exact image position, both exact adjacent bone lengths, and an
opposite readable branch cannot all be satisfied by changing only `M` for a
generic calibrated-perspective configuration. A non-trivial solution exists
only under special camera/limb configurations such as a ray contained in the
bone-circle plane.

## 7. Synthetic geometric contracts

The diagnostic adds deterministic fixtures in
`tests/test_frame_pose_hinge_feasibility.py` and emits them through
`scripts/diagnose_hinge_feasibility.py`.

| fixture | bone-preserving set | ray/circle intersections | opposite branch on ray |
|---|---|---:|---:|
| ordinary oblique limb | circle | 1, the current `M` | no |
| fronto-parallel limb | circle | 1, the current `M` | no |
| exact depth-aligned limb | circle, but depth basis undefined | 1 | no readable SignState |
| near-depth-aligned limb | circle, frozen readability floor applies | 1 | no readable SignState |
| ray parallel and contained in circle plane | circle | 2 forward points | yes, one second point |
| tangent spheres | zero-radius point | 1 point | no branch |
| inconsistent spheres | empty | 0 | no |
| zero-length axis, equal positive radii | sphere | handled as sphere, not circle | not a hinge circle |

The special fixture is concrete: a circle centred at `[0, 3, 0]` with radius
`2`, viewed along the `Y` ray from the camera origin. The forward intersections
are `[0, 1, 0]` and `[0, 5, 0]`; their readable depth branches are opposite.
This is a valid exception, not a refutation of the generic result.

## 8. Real-scene feasibility accounting

The workspace contains the repository and docs/39 code, but not the verified
docs/39 bank, `O_BILATERAL` prediction, historical artifact, or raw 3DPW
sequence pickles. A read-only search found no local bank/prediction/raw-camera
inputs, so a new real-scene count would not be reproducible here. No real-frame
number is invented in this report.

The new optional real replay path is nevertheless implemented in
`scripts/diagnose_hinge_feasibility.py:real_scene_feasibility`. Given the exact
docs/39 inputs, it reuses:

```text
O_BILATERAL stored prediction
oracle absolute pelvis placement
per-frame 3DPW intrinsics and cam_poses
the same canonical/world-to-camera conversion
```

For every corrected hinge frame it records, without a result-selected gate:

```text
ray-plane signed/absolute incidence quantity
ray-plane offset
axis length
bone-preserving circle radius
in-plane fraction and current/requested branch
exact ray/circle intersection relation and count
whether a second intersection satisfies the requested branch
```

The path also records the diagnostic lower-bound result described in Section
10. It is ready for the prior LabServer/data-mounted replay, but it was not run
without those exact artifacts. docs/39's historical real-camera measurements
remain preserved and are not relabelled as this new feasibility result.

## 9. Formula limit, write-set limit, and state limit

These are separate findings:

### Formula limit

The two existing formulas have different ownership choices. `DEPTH_ONLY`
changes the middle joint along canonical depth and does not stay on the
bone-preserving circle in general. `MINIMUM_NORM` removes the axis-parallel
slide and lands on the bone-preserving circle. Thus the measured trade-off is
not evidence that no other middle-only formula can preserve bone lengths.

### Write-set limit

The generic proof above shows that the current write set `CHANGE: M; FIX: P,D`
cannot satisfy exact image position, both exact adjacent lengths, and an
opposite branch simultaneously. If all of those are mandatory, the write set
must be reconsidered. This report does not choose a larger write set.

### State / observation limit

A camera-aware compromise can be computed in principle, but the actual Branch
Constraint has no `input_2d`, `K`, extrinsics, absolute root depth, or image
size. The exact camera ray and perspective displacement therefore cannot be
computed by the current production API. The real-scene oracle camera used in
docs/39 is diagnostic-only state, not runtime Branch Constraint state.

## 10. Conditional camera-aware lower bound

The exact-feasibility result satisfies the condition for a diagnostic lower
bound. The implemented diagnostic objective is:

```text
minimize ||pi_K(M') - pi_K(M)||_2

subject to:
    P and D fixed exactly
    |M'-P| = |M-P|
    |D-M'| = |D-M|
    existing Sign Contract requested branch is readable
```

There is no 3D term, weight, learned term, fitted coefficient, reprojection
tolerance, or result-selected threshold. The feasible set is the exact
bone-preserving circle intersected with the existing readable requested-sign
arc. The solver parameterizes that arc by `theta`, converts the perspective
objective to a rational trigonometric function, and obtains stationary points
from the derivative polynomial after the tangent-half-angle substitution. The
readability-boundary endpoints are evaluated explicitly. It does not sample
the circle.

The solver is unit-tested on the synthetic ordinary and special fixtures. The
special two-intersection fixture has a zero pixel lower bound, as it should.
The real-scene lower-bound replay did **not** execute because the exact docs/39
artifacts are absent from this workspace. Consequently there is no claimed
real comparison table between this bound, `DEPTH_ONLY`, and `MINIMUM_NORM` in
this batch.

## 11. Existing operators over-specify one-bit SignState

Yes. The Sign Contract requires only:

```text
sign( +Y component of normalized bend direction ) = requested sign
```

subject to its existing readability semantics. It does not require the exact
reflected bend direction.

Both existing hinge operators nevertheless install the same full reflected
bend direction: they reverse the depth component of the perpendicular bend
offset while preserving the complementary component. `MINIMUM_NORM` reaches
that reflected direction without the axis-parallel slide; `DEPTH_ONLY` reaches
the same bend direction through a depth-only middle-joint write but changes
the adjacent lengths. This is more geometry than the one-bit SignState
semantics demand. The behavior remains frozen; it was measured, not changed.

## 12. Architecture classification

```text
E — MIXED: B + C, with D describing the current ownership frontier.
```

- **B — middle-joint write set inherently insufficient:** proved generically
  for exact image preservation + both adjacent lengths + opposite branch.
- **C — camera/observation state is missing:** any camera-aware compromise or
  lower-bound policy needs projection state that the actual Branch Constraint
  does not receive.
- **D — irreducible trade-off under current ownership:** among the two tested
  production policies, docs/39's measured objectives remain non-dominated;
  this is now explained by the generic geometry rather than attributed only to
  their formulas.

This is not A: a different middle-only formula cannot evade the generic ray /
circle intersection result. It is not a choice among expanded architectures.

## 13. What must change before a third production policy is meaningful

Nothing is selected in this batch. Before designing a third policy, the
project needs an explicit architecture decision about at least one of:

```text
whether the write set may expand beyond M; and/or
whether calibrated projection/root-depth state belongs in the constraint layer
```

Until that decision is made, a third middle-only production formula would be a
choice inside a contract that has already been shown unable to own all four
properties generically.

## 14. Frozen scope and tests

No SignState field, VLM output, RGB input, temporal evidence, training run,
Geometry Core change, default change, threshold tuning, or production write
policy was added. `DEPTH_ONLY` remains the default.

Focused validation:

```text
pytest -q \
  tests/test_frame_pose_hinge_feasibility.py \
  tests/test_frame_pose_write_policy_observation.py \
  tests/test_frame_pose_hinge_conditioning.py \
  tests/test_frame_pose_branch_constraints.py

56 passed
```

With the repository root explicitly on `PYTHONPATH` for the pre-existing
`scripts.*` test imports, the full suite also completed:

```text
PYTHONPATH=.:src pytest -q
533 passed, 40 skipped
```

The new diagnostic has no model or external-data dependency for its synthetic
contracts. The real replay requires the exact docs/39 data-mounted environment
and was intentionally not replaced with a different dataset or camera.

## 15. Completion

The visible-image versus metric-skeleton trade-off is not merely a defect of
`DEPTH_ONLY` and `MINIMUM_NORM`. A generic middle-joint-only hard constraint
with fixed `P` and `D` has a bone-preserving circle of candidates, but exact
image preservation restricts that circle to the current point because the
camera ray generically meets the circle plane once. An opposite hidden branch
cannot then be installed without giving up at least one exact ownership
requirement. A second solution exists only in special contained-ray geometry.

The current result is therefore:

```text
generic exact incompatibility proved;
write-set limit applies;
camera state is also absent for any principled camera-aware compromise;
STOP before choosing the next architecture.
```
