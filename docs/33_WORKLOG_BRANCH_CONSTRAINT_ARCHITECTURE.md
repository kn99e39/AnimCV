# Worklog — SignState as an Explicit Branch Constraint (2026-09-07)

> One bounded architecture batch on `arch/single_frame_first`. **No training,
> no fine-tuning, no VLM, no prompt, no sign sensor, no Sign Contract change,
> no Geometry Core change, no temporal evidence.** Every number below comes
> from replaying a *stored* Geometry Core prediction through a parameter-free
> operator.
>
> The question: **if the Geometry Core predicts continuous pose first, can an
> explicit branch constraint enforce the correct discrete branch while
> preserving the Core's continuous geometry substantially better than learned
> SignState conditioning?**

## 1. Starting/final HEAD

Start `ffa2826` (docs/32 closed). This batch: `11f1e42` (branch constraints +
candidate/topology identity), `be80226` (unresolved decomposition + strided
review export), plus this worklog's own commit.

## 2. What was preserved, not re-run, and not reinterpreted

Nothing in the sign programme was retrained or recomputed. The following stand
exactly as they were:

- `S0`/`S1`/`S2` (`sign_v1`), the four `O_*` attribution groups (`sign_attr_v2`),
  the eight `O_*` minimality candidates (`sign_minimality_v1`), the four
  `H_NO_*` 3-of-4 leave-one-out candidates (`sign_hinge_loo_v1`), the five
  `L_*` post-attention candidates (`sign_local_hinge_v1`).
- `sign_influence_v1` remains labelled **INVALID** (out-of-distribution
  baseline); `sign_influence_v2` remains the contract-repaired probe.
- `sign_attr_v1_INVALID_mask_discarded` remains discarded and is not read.

docs/32 is carried forward in its **conservative** reading: sign-field
necessity is **conditioning-topology dependent**. This batch does **not**
claim the knee signs carry no information, and does **not** claim the
post-attention topology is production-ready. It tests a different abstraction
entirely and leaves both of those questions open.

## 3. Candidate identity is now bound to conditioning topology

The `L_*` and `O_*`/`H_NO_*` candidates differ *only* in where the hinge sign
enters the network, and a single global CLI switch made it possible to train
`L_HINGE` under `pre_attention` (which is just `O_HINGE`) or `O_HINGE` under
`post_attention` (which is not the historical `O_HINGE` at all) and file either
under a name whose measured lineage says something else.

- Every one of the 24 `CANDIDATES` entries now declares
  `expected_hinge_sign_injection`. All historical candidates are
  `pre_attention`; all five docs/32 candidates are `post_attention`.
- `run_sign_experiments.require_topology_match` refuses a mismatch before any
  training starts, rather than producing a mislabelled result.
- `diagnose_sign_influence.verify_checkpoint_identity` checks the
  **checkpoint's own stored** `candidate.name` and
  `model_config.hinge_sign_injection` against the contract it is being probed
  under, and refuses on disagreement. The CLI label alone is no longer
  evidence. Checkpoints written before the topology field existed default to
  `pre_attention`, which is what they are by construction, and are refused
  against a `post_attention` contract.

Tests: `test_every_candidate_binds_its_conditioning_topology`,
`test_runner_refuses_a_candidate_under_the_wrong_topology`,
`test_influence_diagnostic_verifies_checkpoint_identity_not_the_cli_label`.

## 4. The operator: `src/framepose/branch_constraints.py`

Separate module. **No trainable parameters.** It may read the predicted XYZ,
the requested sign, and the Sign Contract's own math — and nothing else. It
does not read RGB, VLM embeddings, GT XYZ magnitude, target bone lengths, or
neighbouring frames; there is no code path by which it could.

### 4.1 Bilateral (`shoulder_forward_depth`, `hip_forward_depth`)

The pair's forward-depth coordinates are **exchanged**. Preserved exactly:
every X and Z coordinate of every joint, the pair's depth midpoint, and
`|D| = |y_right − y_left| / sqrt(2)`. Changed: only which side owns the near
branch. Because the exchange preserves `|D|` exactly, a prediction whose
separation is already below the contract's `STABLE_FORWARD_DEPTH_M = 0.01 m`
stability floor can never read back as the requested branch, and is refused
rather than given an invented separation.

### 4.2 Hinge (four `*_forward_bend`)

Screen space is preserved: X and Z of every joint, **including the moved
one**, are untouched. Only the middle joint's depth moves. With axis
`a = distal − proximal` and the contract's own perpendicular offset
`o = (joint − proximal) − a·((joint − proximal)·a)/|a|²`, moving the middle
joint's depth by `δ` changes the offset's depth component by exactly

```
o'_y = o_y + δ · (a_x² + a_z²) / |a|²
```

so the reflection `o'_y = −o_y` has the closed form

```
δ = −2·o_y·|a|² / (a_x² + a_z²)
```

**No tuned magnitude, no scale factor, no step size.** The factor
`f = (a_x² + a_z²)/|a|²` is the squared in-image-plane fraction of the limb
axis and is the **exact degeneracy** of any depth-only branch correction: it
vanishes when the limb points along the camera depth axis, where no change of
the middle joint's depth can change the branch at all. The operator refuses
when `sqrt(f) < UNIT_FORWARD_EPSILON` (0.1) — reusing the contract's existing
constant with its existing meaning, introducing no new one.

## 5. Semantics: exact, or reported

| requested | operator behaviour |
|---|---|
| UNKNOWN | exact no-op (bit-identical output) |
| already satisfied | exact no-op (bit-identical output) |
| violated, correctable | smallest declared correction; **the result is read back through `sign_state`** and must equal the requested sign |
| violated, not correctable | the field is **reverted** and reported `unresolved` with a reason |

There is no "encourage the branch" path and no silent failure. `torso_facing`
has **no declared correction** and is refused outright rather than ignored.
`coverage()` asserts `requested = already_satisfied + corrected + unresolved`
per field and raises when it fails; `requested` is counted from the requested
sign itself, not from the outcome buckets, so the identity is a real
cross-check. (It was a tautology as first written; that is fixed.)

## 6. Synthetic contract tests came first

28 tests in `tests/test_frame_pose_branch_constraints.py`, all passing before
any real frame was touched: exact no-ops, bilateral X/Z-midpoint-|D|
preservation, locality, idempotence, stability-floor refusal, hinge branch
satisfaction, depth-only movement, only-the-middle-joint movement, the
closed-form reflection identity `depth_delta_m ≈ −2 × predicted_offset_forward_m`
in the non-degenerate case, singular-axis refusal, below-floor read-back
refusal, ordering, `torso_facing` refusal, out-of-domain refusal, invalid-joint
handling, batch determinism, the coverage identity as a cross-check, and the
unresolved decomposition.

## 7. Evaluation protocol (no training happened)

`scripts/replay_branch_constraints.py` reads the **stored**
`sign_v1/S0/prediction_test.npy` — the Geometry Core output of the neutral-sign
candidate `S0_neutral_sign`, produced in an earlier batch and untouched since —
and the oracle SignState derived from the same bank
(`bank_3dpw_paired_v2.json`, content digest `75519e63…`, 7,076 `test` frames).
Constraint variants are scored by the **same** `evaluate_predictions` used for
every trained candidate, so a constraint result and a learned-conditioning
result are directly comparable.

Regime: `benchmark_detector_observation`. The sign source is the **oracle**,
which is an upper bound on any sign sensor and is not itself a sensor result.

Variants: `C0` (nothing requested), `C_HIP`, `C_BILATERAL`, `C_HINGE_ALL`,
`C_LEFT_ELBOW`, `C_RIGHT_ELBOW`, `C_LEFT_KNEE`, `C_RIGHT_KNEE`. No variant
requests `torso_facing`. `C0` is asserted **bit-identical** to the stored
prediction; an operator that is not an exact no-op when nothing is requested
would contaminate every other variant.

## 8. Result — the hinge branch (7,076 test frames)

| variant | hinge flip | hinge MAE° | root yaw° | shldr res mm | hip res mm | MPJPE mm | PA-MPJPE mm |
|---|---|---|---|---|---|---|---|
| S0 baseline | 0.0212 | 24.819 | 10.920 | 3.790 | 2.587 | 81.976 | 56.852 |
| **C0** | **0.0212** | **24.819** | **10.920** | **3.790** | **2.587** | **81.976** | **56.852** |
| **C_HINGE_ALL** | **0.0122** | **22.740** | 10.920 | 3.790 | 2.587 | **82.060** | **57.106** |
| C_LEFT_ELBOW | 0.0187 | 24.351 | 10.920 | 3.790 | 2.587 | 82.063 | 57.008 |
| C_RIGHT_ELBOW | 0.0191 | 24.197 | 10.920 | 3.790 | 2.587 | 82.051 | 56.968 |
| C_LEFT_KNEE | 0.0192 | 24.407 | 10.920 | 3.790 | 2.587 | 81.888 | 56.796 |
| C_RIGHT_KNEE | 0.0187 | 24.241 | 10.920 | 3.790 | 2.587 | 81.985 | 56.894 |

`C0` reproduces the baseline to the last digit — the no-op is exact.

Note the three columns that **do not move at all** under any hinge variant:
root yaw, shoulder residual, hip residual are bit-identical to the baseline.
That is not a statistical observation, it is structural: the hinge correction
writes one depth coordinate of one middle joint, and no other Sign Contract
field reads that joint.

## 9. Result — the hinge branch versus learned conditioning

Each learned candidate against its own neutral control (`O_*`/`H_NO_*` against
`S0`, `L_*` against the post-attention neutral `L_NEUTRAL`):

| approach | flip Δ | flip Δ% | hinge MAE Δ° | MPJPE Δ mm | MPJPE cost per 1% flip reduction |
|---|---|---|---|---|---|
| **C_HINGE_ALL** (constraint) | **−0.0090** | **−42.5%** | **−2.079** | **+0.084** | **0.0020 mm** |
| `O_HINGE` (learned, pre-attention) | −0.0053 | −25.0% | −1.379 | +1.464 | 0.0586 mm |
| `L_HINGE` (learned, post-attention) | −0.0025 | −11.5% | −0.777 | +2.134 | 0.186 mm |
| `S1` (learned, all seven fields) | −0.0025 | −11.8% | −1.214 | −2.776 | — |

The constraint delivers **1.7× the flip reduction of the best learned hinge
candidate at 1/17 of its MPJPE cost**, and 3.7× the flip reduction of the
post-attention candidate at 1/25 of its cost. Per unit of branch benefit it
preserves the Core's continuous geometry roughly **30× better** than
`O_HINGE` and **90× better** than `L_HINGE`.

`S1` is listed for completeness and is not a hinge comparison: it activates all
seven fields, and its MPJPE gain comes from the orientation group, not the
hinge group (see docs/29-30).

## 10. Result — the bilateral branch, where the constraint LOSES

| approach | root yaw° | shldr res mm | hip res mm | flip | hinge MAE° | MPJPE mm |
|---|---|---|---|---|---|---|
| S0 baseline | 10.920 | 3.790 | 2.587 | 0.0212 | 24.819 | 81.976 |
| C_HIP (constraint) | 10.424 | 3.790 | 2.348 | 0.0219 | 24.913 | 81.910 |
| `O_HIP` (learned) | **8.261** | 5.699 | **−0.064** | **0.0202** | **25.211** | **80.418** |
| C_BILATERAL (constraint) | 9.777 | **5.094** | 2.348 | 0.0239 | 25.297 | 81.726 |
| `O_BILATERAL` (learned) | **7.999** | **2.208** | **0.479** | **0.0189** | **24.099** | **79.229** |

`O_BILATERAL` beats `C_BILATERAL` on **every single axis**: root yaw by 1.78°,
shoulder residual, hip residual, flip rate, hinge MAE and MPJPE. The bilateral
constraint is not competitive with learned bilateral conditioning.

## 11. Why the hinge constraint is safe and the bilateral one is not

Sign agreement per field, measured on the constrained predictions:

| field | S0 | C_HINGE_ALL | C_BILATERAL |
|---|---|---|---|
| torso_facing | 0.9815 | 0.9815 | 0.9815 |
| shoulder_forward_depth | 0.8605 | **0.8605** | 0.9465 |
| hip_forward_depth | 0.8161 | **0.8161** | 0.8569 |
| left_elbow_forward_bend | 0.8672 | 0.9300 | **0.8644** |
| right_elbow_forward_bend | 0.8680 | 0.9340 | **0.8538** |
| left_knee_forward_bend | 0.9005 | 0.9456 | **0.8988** |
| right_knee_forward_bend | 0.8756 | 0.9429 | **0.8749** |

`C_HINGE_ALL` moves the four fields it asked for and leaves the other three
**bit-identical**. `C_BILATERAL` improves the two it asked for and **degrades
all four it did not** — right elbow worst, by 1.4 points.

The cause is structural, not statistical. The hinge correction moves the
*middle* joint of a chain (elbow, knee), which no other Sign Contract field
reads. The bilateral correction moves the shoulders and hips, which are the
*proximal anchors* of all four hinge chains — so exchanging their depth
necessarily re-derives four other branches as a side effect. A parameter-free
constraint has no way to compensate for that; a learned conditioning does,
which is exactly what `O_BILATERAL` is doing.

**This is the batch's main architectural finding: the branch-constraint
abstraction is appropriate where the constrained joint is a leaf of the
contract's dependency graph, and inappropriate where it is an anchor.**

## 12. Wrong-sign control (two endpoints, no sweep)

Same operator, same frames, the **opposite** oracle branch requested. No
percentage sweep was run: a sweep would invent a sensor-accuracy curve that
nothing measured here supports.

| variant | flip (oracle) | flip (opposite) | MPJPE (oracle) | MPJPE (opposite) |
|---|---|---|---|---|
| C_HINGE_ALL | 0.0122 | **0.5449** | 82.060 | **97.182** |
| C_LEFT_ELBOW | 0.0187 | 0.1396 | 82.063 | 84.270 |
| C_RIGHT_ELBOW | 0.0191 | 0.1399 | 82.051 | 86.752 |
| C_LEFT_KNEE | 0.0192 | 0.1736 | 81.888 | 85.986 |
| C_RIGHT_KNEE | 0.0187 | 0.1553 | 81.985 | 86.100 |
| C_HIP | 0.0219 | 0.0185 | 81.910 | 86.729 |
| C_BILATERAL | 0.0239 | 0.0498 | 81.726 | 103.124 |

The constraint is **exactly as good as the sign it is given, in both
directions**. `C_HINGE_ALL` under the opposite sign drives the flip rate from
1.2% to 54.5% and hinge MAE from 22.7° to 94.2°.

This is the honest cost of the abstraction and the single most important
caveat for whatever comes next: **a learned conditioning can partially ignore
a sign it does not believe; a constraint cannot, by construction.** The
constraint's entire benefit is contingent on the sign being right, and this
batch used an oracle. Nothing here says what accuracy a real sign sensor would
need, and nothing here should be read as saying a sensor exists.
