# Worklog — Endpoint-Fixed Pose Reconciliation by Two-Bone Swivel (2026-09-10)

> Research architecture following docs/40. The historical
> `DEPTH_ONLY`/`MINIMUM_NORM` branch operators remain unchanged. This batch
> introduces a separate `Pose Reconciliation` layer and does not promote it to
> production.

## 1. Starting state and preserved history

Starting branch and HEAD:

```text
branch: arch/single_frame_first
HEAD:   21e2f4b docs: audit middle-only hinge feasibility
```

Docs/33 through docs/40, the historical operators, docs/39 comparisons, and
docs/40 feasibility proof are not modified by this batch. The new files are:

```text
src/framepose/pose_reconciliation.py
scripts/replay_pose_reconciliation.py
tests/test_frame_pose_pose_reconciliation.py
```

## 2. Architecture boundary

The new boundary is:

```text
Frame Pose Core root-relative XYZ prediction
    + explicit ProjectionContext / Geometry Observation
    + existing discrete hinge SignState
    -> Pose Reconciliation
    -> reconciled root-relative XYZ pose
```

This is deliberately not a new value in the historical
`hinge_write_policy` selector. `framepose.branch_constraints` remains the
owner of the historical direct-XYZ behavior and its `DEPTH_ONLY` default.

## 3. Exact two-bone swivel contract

For each arm or leg chain:

```text
arm: shoulder -> elbow -> wrist
leg: hip      -> knee  -> ankle

P = proximal endpoint
M = middle hinge / swivel mediator
D = distal endpoint
```

The reconciliation write contract is:

```text
P' = P exactly
D' = D exactly
M' may change; no other joint is written
```

The metric constraints are exact:

```text
|M' - P| = |M - P|
|D  - M'| = |D - M|
```

Consequently `M'` remains on the exact two-sphere intersection circle derived
in docs/40. There is no weighted 3D penalty and no direct XYZ delta policy.

## 4. Swivel-plane semantics versus rig-roll semantics

The free variable is an angle `theta` around the bone-preserving circle:

```text
M(theta) = C + rho (u_hat cos(theta) + v_hat sin(theta))
```

This is the two-bone IK bend-plane/swivel degree of freedom. It is not a rig
bone-roll quaternion. The current canonical representation contains joint
positions only, so it cannot represent or measure wrist/ankle orientation.

The downstream contract is explicit:

```text
apply the reconciled positions through articulated-rig IK/retargeting;
preserve the incoming wrist/end-effector orientation independently.
```

No wrist or ankle is moved, carried, rotated, or assigned a fabricated
orientation by this layer.

## 5. ProjectionContext contract

`ProjectionContext` is explicit and research-scoped. It contains only the
state needed to project the root-relative canonical pose:

```text
intrinsics             3x3 calibrated K
image_size             (width, height)
root_offset_camera     explicit camera-space root placement
camera_origin          explicit camera origin, normally [0, 0, 0]
placement_mode         provenance label
```

The controlled 3DPW replay uses the docs/39
`research_oracle_absolute_root_placement` mode: target absolute pelvis is
added to the stored root-relative prediction. This is explicitly not
production root inference. Missing or invalid context returns `unresolved`;
the layer never invents focal length, extrinsics, root depth, or image size.

Projection uses the repository convention:

```text
pixel_x = fx * X/Y + cx
pixel_y = -fy * Z/Y + cy
```

## 6. SignState and Geometry Observation ownership

The existing hinge fields are consumed unchanged:

```text
left_elbow_forward_bend
right_elbow_forward_bend
left_knee_forward_bend
right_knee_forward_bend
```

SignState owns only the readable hidden forward/depth branch. It does not
provide `theta`, continuous bend magnitude, elbow/knee XYZ, screen-plane side,
bone length, or endpoint orientation.

Geometry Observation owns the visible configuration. For each allowed circle
point, the primary objective is:

```text
minimize || project(M(theta)) - observed_middle_2d ||_2
```

subject to the fixed endpoints, exact original adjacent lengths, and the
existing readable requested SignState branch.

If the requested branch is already satisfied, reconciliation is an exact
no-op, even when another circle point would project closer to the observation.
This prevents the layer from becoming a general pose optimizer.

## 7. Solver derivation

For the circle basis from docs/40, the existing Sign Contract's readable branch
is the arc:

```text
requested_sign * cos(theta) >= UNIT_FORWARD_EPSILON / sqrt(f)
```

where `f` is the squared camera-depth projection of the normalized `P->D`
axis. The existing `MIN_BEND_OFFSET_M` and `UNIT_FORWARD_EPSILON` semantics
are reused unchanged. If the circle is degenerate, the axis is exactly
depth-aligned, the branch is unreadable, or no valid projection context exists,
the result is `unresolved` and the pose is unchanged.

The perspective objective is rational in `sin(theta)` and `cos(theta)`. The
implementation converts it with the tangent-half-angle substitution:

```text
t = tan(theta / 2)
```

and finds stationary points from the resulting derivative polynomial. The two
readability-boundary endpoints and the half-angle infinity point are evaluated
explicitly. There is no angle sweep, result-selected discretization, weight,
learned term, or performance-derived tolerance. Numerical handling is limited
to floating-point root recognition and machine-precision contract checks.

## 8. Synthetic contracts

`tests/test_frame_pose_pose_reconciliation.py` covers:

| contract | result |
|---|---|
| ProjectionContext camera convention | pass |
| exact proximal/distal endpoint preservation | pass |
| exact two adjacent bone lengths | pass |
| exact circle/swivel interpretation | pass |
| requested readable SignState read-back | pass |
| observed-2D-guided visible configuration | pass |
| deterministic solver | pass |
| already-correct exact no-op | pass |
| UNKNOWN / missing context refusal | pass |
| exact depth-aligned unreadable chain refusal | pass |
| downstream orientation contract remains explicit | pass |

These establish the mechanism and ownership contract only. They do not establish
real-scene viability.

## 9. Controlled replay implementation

`scripts/replay_pose_reconciliation.py` implements the controlled comparison:

```text
H0                    stored O_BILATERAL prediction
DEPTH_ONLY            historical operator
MINIMUM_NORM          historical operator
R_SWIVEL_OBS          oracle SignState + stored input_2d middle observation
R_SWIVEL_ORACLE_2D    oracle SignState + exact projected target middle joint
```

The optional `R_SWIVEL_ORACLE_2D` run is an observation upper bound. It is kept
separate from `R_SWIVEL_OBS` so Geometry Observation quality cannot be hidden
inside an oracle result.

The replay also implements the opposite-oracle endpoint, per-chain accounting,
endpoint/bone invariants, image metrics, 3D metrics, global/bilateral checks,
and representative review records. It does not train, modify the historical
operators, or change production behavior.

## 10. Real-scene execution status

The exact docs/39 bank, stored `O_BILATERAL` prediction, and raw 3DPW camera
pickles are not present in this workspace. Therefore the controlled real-scene
replay has not been executed and no H0 / DEPTH_ONLY / MINIMUM_NORM /
R_SWIVEL_OBS numerical result is claimed here. The replay requires the same
data-mounted environment and provenance as docs/39.

This is an artifact-availability limitation, not a silent substitution with a
different camera or dataset. The replay command is ready:

```text
PYTHONPATH=src python scripts/replay_pose_reconciliation.py \
  --bank <docs/39 bank> \
  --source O_BILATERAL=<candidate>:<prediction.npy>:<evaluation.json> \
  --raw-root <3DPW raw root> \
  --out <output directory>
```

## 11. Required quantitative accounting once replayed

The replay output keeps these dimensions separate:

```text
hinge flip rate / hinge direction MAE / per-chain values
requested-sign satisfaction / unresolved count
endpoint displacement for shoulder/hip and wrist/ankle
proximal-middle and middle-distal bone-length changes
middle-joint image displacement
target-projection error
observation-consistency error
MPJPE / PA-MPJPE / per-joint errors
root yaw and bilateral shoulder/hip changes
wrong-sign endpoint damage
review records with P/M/D, theta, projections, and lengths
```

No single aggregate score is used to hide an ownership regression.

## 12. Architecture classification

The synthetic architecture contract is supported, but the required real-scene
classification cannot yet be assigned because the controlled replay artifacts
are unavailable. In particular, A/B/C/D cannot be chosen honestly without
comparing `R_SWIVEL_OBS` with `R_SWIVEL_ORACLE_2D` and both historical
baselines.

The current evidence does establish:

```text
mechanism: valid endpoint-fixed two-bone swivel reconciliation
state contract: explicit camera/projection state is now available to this layer
production status: research-only, not promoted
real ownership verdict: pending exact docs/39 replay
```

The question of whether Geometry Observation is the dominant limiter remains
open until the observed-vs-oracle replay is run.

## 13. Frozen requirements

No historical `DEPTH_ONLY` or `MINIMUM_NORM` behavior changed. No endpoint
carry-along, full-body IK, CCD/FABRIK, weighted objective, threshold tuning,
SignState field, VLM output, RGB input, temporal context, bilateral-conditioning
change, training run, default change, or production promotion was made.

Wrist/ankle orientation remains a mandatory downstream IK/retargeting contract
and is explicitly not represented by this canonical XYZ reconciliation layer.

## 14. Tests and environment

Focused validation for the new layer:

```text
pytest -q tests/test_frame_pose_pose_reconciliation.py
6 passed
```

The real replay was not run because its exact historical artifacts are absent.
The final architecture verdict must follow that controlled replay rather than a
synthetic result alone.
