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

## 6. Residual hinge flips: the exact cross-table

Built from the **stored** `branch_constraint_v3/prediction_test_C_HINGE_ALL.npy`.
The analysis recomputes the constraint and refuses to proceed unless the
recompute is **byte-identical** to the stored artifact; it reproduced exactly,
as did the stored `C_HIP` and `C_BILATERAL`.

Axes: constraint outcome × requested-sign satisfaction × historical full-3D
hinge flip. All four chains pooled, 28,304 chain-frames:

| constraint outcome | sign | historical | count |
|---|---|---|---|
| already_satisfied | satisfied | not_flipped | 20,242 |
| already_satisfied | satisfied | **flipped** | **32** |
| corrected | satisfied | not_flipped | 1,371 |
| corrected | satisfied | **flipped** | **17** |
| unresolved | not_satisfied | not_flipped | 1,342 |
| unresolved | not_satisfied | **flipped** | **79** |
| not_requested | — | not_flipped | 1,851 |
| not_requested | — | **flipped** | **161** |
| not_requested | — | metric_unavailable | 3,209 |

**289 residual flips**, decomposed:

| cause | count | share |
|---|---|---|
| the oracle never requested a branch for this chain (its own sign is degenerate) | 161 | 55.7% |
| requested, but the prediction was unreadable → `unresolved` | 79 | 27.3% |
| **requested Y sign satisfied, and still historically flipped** | **49** | **17.0%** |

## 7. Is the one-bit Y sign a complete hinge representation? **No — outcome B**

49 chain-frames satisfy the requested `+Y` branch and are still counted as a
full-3D hinge flip. Outcome **B** holds: the sign is useful but is **not** a
complete representation of hinge orientation.

The magnitude matters as much as the fact, and the error distributions separate
the buckets cleanly:

| bucket | n | mean error° | p90 error° |
|---|---|---|---|
| satisfied & flipped (left elbow) | 19 | **97.6** | 102.7 |
| satisfied & flipped, corrected (left elbow) | 8 | **103.6** | 115.2 |
| satisfied & flipped (left knee) | 3 | **98.7** | 101.9 |
| satisfied & flipped, corrected (left knee) | 7 | **100.8** | 114.8 |
| unresolved & flipped (left knee) | 37 | 133.5 | 166.0 |
| not_requested & flipped (left knee) | 37 | 131.5 | 165.9 |
| not_requested & flipped (left elbow) | 55 | 128.7 | 167.3 |

The satisfied-but-flipped cases sit **just past the 90° threshold** (mean
97-104°, p90 103-115°). The unreadable and never-requested cases are
**near-antiparallel** (mean 129-133°, p90 166-167°). So the one-bit encoding
pins the bend to the correct depth hemisphere and leaves a residual
in-image-plane disagreement that only marginally crosses the metric's cut.

Per field, satisfied-and-flipped: left elbow 27, right elbow 8, left knee 10,
right knee 4. Sample IDs for all 49 are in
`constraint_attribution_v2/constraint_attribution.json` under
`satisfied_but_flipped_samples` — e.g. `3dpw:downtown_bar_00:actor0#000389`
(left elbow, requested `+1`, read back `+1`, 103.7°).

**The Sign Contract is not changed in this batch**, as directed. This is
recorded as a measured limit of the current encoding.

## 8. The machine-readable constraint dependency graph

`src/framepose/constraint_graph.py`. `SignField.joints` is left alone: it is a
routing/ownership property and, for a hinge field, names only the middle joint
even though the sign is derived from proximal, middle and distal. The
constraint relation is separate.

Read sets are transcribed from the sign derivation functions and **pinned by
test**: for every field, a joint outside its read set can never change it under
large random perturbation, and every joint inside it can.

| field | reads | writes (anchor_only) | downstream dependents |
|---|---|---|---|
| `shoulder_forward_depth` | left/right_shoulder | left/right_shoulder | `torso_facing`, both elbow bends |
| `hip_forward_depth` | left/right_hip | left/right_hip | both knee bends |
| `*_elbow/knee_forward_bend` | proximal, middle, distal | middle only | **none** |
| `torso_facing` (no constraint) | pelvis, thorax, left/right_shoulder | — | — |

Note `torso_facing` **reads four** joints though `SignField.joints` routes
nine, and a hinge field **reads three** though it routes one. Using the routing
property for constraint reasoning would have got both wrong.

Under `dependency_aware` the four hinge dependents move from *at risk* to
**preserved exactly** — a field is marked exactly preserved only when its whole
read set rides a single displacement group, i.e. sees a rigid translation.
`torso_facing` remains formally at risk under a shoulder write; separately, its
sign **numerator** `cross(up, right)·ŷ = up_z·right_x − up_x·right_z` contains
no Y at all, so a depth-only write can move it only through the unit
normalization. That is pinned by its own test.

## 9. Historical bilateral collateral, attributed

Every collateral effect of the anchor-only pair swap lands **exactly** on the
chains that hang from the moved anchor, and nowhere else:

| variant | left elbow | right elbow | left knee | right knee |
|---|---|---|---|---|
| `C_SHOULDER` newly flipped | 14 | 43 | **0** | **0** |
| `C_HIP` newly flipped | **0** | **0** | 13 | 5 |
| `C_BILATERAL` newly flipped | 14 | 43 | 13 | 5 |

Sign transitions tell the same story. `C_SHOULDER`: right elbow 83
correct→incorrect against 4 incorrect→correct; left elbow 29 against 14. Knee
chains: **exactly zero** changed, agreement bit-identical. `C_HIP` is the
mirror image, and `C_BILATERAL` is the exact union of the two.

This is the dependency-graph explanation confirmed quantitatively, not merely
inferred from aggregate sign agreement — the collateral is 100% concentrated in
the fields whose read set intersects the write set, with no leakage anywhere
else. Section 8's gate is therefore passed, and the dependency-aware operator
was implemented.

## 10. The dependency-aware bilateral operator

Implemented as a **write policy** on the existing operator, not as a new
operator: `bilateral_write_policy` defaults to `anchor_only`, so every docs/33
result stays reproducible bit-for-bit. The historical pair swap is unmodified,
and the hinge operator is untouched under either policy.

Exact mathematics. With anchors `L`, `R` and their predicted depths `y_L`,
`y_R`:

```
delta_L = y_R - y_L            delta_R = -delta_L = y_L - y_R
```

Under `anchor_only` those deltas are applied to `L` and `R`. Under
`dependency_aware` each delta is additionally applied, **unchanged**, to the
limb chain that anchor is the proximal joint of:

```
left_shoulder  delta -> left_shoulder,  left_elbow, left_wrist
right_shoulder delta -> right_shoulder, right_elbow, right_wrist
left_hip       delta -> left_hip,  left_knee,  left_ankle
right_hip      delta -> right_hip, right_knee, right_ankle
```

No magnitude is introduced: the delta is exactly the anchor correction the
bilateral branch already required. The chain is translated rigidly in depth, so
every within-chain difference is unchanged.

### Preserved exactly (all pinned by test, `atol=1e-12`)

every joint X · every joint Z · pair depth midpoint · `|D|` · proximal-middle
and middle-distal relative XYZ · the chain's `bend_direction` **numerically** ·
its hinge SignState · internal limb bone lengths · idempotence · determinism ·
UNKNOWN and already-correct exact no-ops · everything outside the two
translated chains.

### NOT preserved, and measured rather than claimed

The anchor's attachment to the torso. Reported per variant: thorax→shoulder
mean |Δ| 0.78/0.82 mm (p95 5.0 mm, max 105-147 mm) on a ~227 mm segment;
pelvis→hip mean |Δ| 0.16 mm (p95 0.00 mm, max 21 mm) on a ~101 mm segment.
These are identical under both write policies — the attachment cost belongs to
moving the anchor at all, which `anchor_only` already paid.

### Causal teeth

A test asserts that on the *same* fixture the historical pair swap **destroys
both** dependent knee signs (−1,+1 → 0,0) and visibly moves both bend
directions, so the preservation test cannot pass on a fixture where nothing was
at risk.

## 11. Real replay (7,076 test frames, oracle signs, no training)

Same stored `S0_neutral_sign` prediction lineage as docs/33
(SHA-256 `4357c7bd…`).

| variant | root yaw° | shldr res mm | hip res mm | flip | hinge MAE° | MPJPE mm | PA-MPJPE mm |
|---|---|---|---|---|---|---|---|
| S0 baseline / `C0` | 10.920 | 3.790 | 2.587 | 0.0212 | 24.819 | 81.976 | 56.852 |
| `C_HIP` (anchor-only) | 10.424 | 3.790 | 2.348 | 0.0219 | 24.913 | 81.910 | 56.857 |
| **`C_HIP_DEP`** | **10.424** | **3.790** | **2.348** | **0.0212** | **24.819** | **81.870** | 56.865 |
| `C_BILATERAL` (anchor-only) | 9.777 | 5.094 | 2.348 | 0.0239 | 25.297 | 81.726 | 56.947 |
| **`C_BILATERAL_DEP`** | **9.777** | **5.094** | **2.348** | **0.0212** | **24.819** | **81.192** | **56.565** |

The dependency-aware variants deliver **exactly the same bilateral benefit** as
their anchor-only twins — identical yaw, identical residuals, the same 578 and
253 corrections — while the hinge columns snap back to the **baseline to the
last digit**. Sign agreement confirms it field by field: under
`C_BILATERAL_DEP` all four hinge fields are bit-identical to `C0`
(0.8672 / 0.8680 / 0.9005 / 0.8756), against `C_BILATERAL`'s degraded
0.8644 / 0.8538 / 0.8988 / 0.8749.

And the continuous cost is **negative**: MPJPE 81.192 vs the baseline's 81.976
(−0.78 mm) and PA-MPJPE 56.565 vs 56.852 (−0.29 mm), better than the
anchor-only variant on both. Carrying the limb along with its anchor is not
merely safer for the sign semantics — it is a better prediction.

Moved joints per constrained frame: `C_BILATERAL` 2.39, `C_BILATERAL_DEP`
7.16. Total introduced depth displacement: 83,087 mm vs 249,260 mm. The
dependency-aware operator moves **3× as much geometry**, and this is stated
rather than hidden — it is exactly why its wrong-sign endpoint is worse
(Section 13).

## 12. Against learned bilateral conditioning

| approach | root yaw° | Δ yaw | shldr res mm | hip res mm | flip | MPJPE mm |
|---|---|---|---|---|---|---|
| S0 baseline | 10.920 | — | 3.790 | 2.587 | 0.0212 | 81.976 |
| `C_HIP_DEP` (constraint) | 10.424 | −0.50 | 3.790 | 2.348 | 0.0212 | 81.870 |
| `O_HIP` (learned) | **8.261** | **−2.66** | 5.699 | **−0.064** | 0.0202 | **80.418** |
| `C_BILATERAL_DEP` (constraint) | 9.777 | −1.14 | 5.094 | 2.348 | 0.0212 | 81.192 |
| `O_BILATERAL` (learned) | **7.999** | **−2.92** | **2.208** | **0.479** | **0.0189** | **79.229** |

The dependency-aware constraint captures **39% of the learned yaw gain**
(1.14° of 2.92°) and **29% of the learned MPJPE gain**. On the depth residuals
it barely moves: hip 2.587 → 2.348 (−9%) against the learned −81%, and the
shoulder residual gets **worse** (3.790 → 5.094).

That last row is not a defect of the write policy — it is a **Sign Contract
expressiveness limit**, and the contract says so itself: `depth magnitude` is
listed under `excluded_by_contract`. The exchange preserves `|D|` **exactly**
by construction, so a sign-only constraint can install the correct near/far
branch but can never correct a wrong separation *magnitude* — and when the
predicted magnitude is wrong, installing the correct sign with the wrong
magnitude can increase the signed residual. Learned conditioning is under no
such restriction, which is precisely where the remaining gap lives.

## 13. Wrong-sign endpoint for the new operator (two endpoints, no sweep)

The same operators, the **opposite** oracle branch requested:

| variant | root yaw° | flip | hinge MAE° | MPJPE mm | PA-MPJPE mm |
|---|---|---|---|---|---|
| `C_HIP` | 39.401 | 0.0185 | 25.747 | 86.729 | 61.105 |
| `C_HIP_DEP` | 39.401 | **0.0212** | **24.819** | **90.805** | 66.641 |
| `C_BILATERAL` | 70.860 | 0.0498 | 34.417 | 103.124 | 80.947 |
| `C_BILATERAL_DEP` | 70.860 | **0.0212** | **24.819** | **129.746** | 90.855 |

A genuine two-sided trade-off, and the most important caveat on the new
operator:

- Under a wrong sign the dependency-aware policy still **protects the hinge
  semantics perfectly** — all four hinge agreements stay bit-identical to the
  baseline (0.8672 / 0.8680 / 0.9005 / 0.8756), while the anchor-only policy
  collapses the elbows to 0.7559 / 0.7686.
- But it **amplifies the positional damage**: MPJPE 129.7 mm against
  anchor-only's 103.1 mm, because a wrong anchor now drags the entire limb
  chain with it instead of only the anchor joint.

So dependency-awareness converts a wrong sign from *semantic* damage into
*positional* damage, and moves ~3× as much geometry while doing it. Both
operators remain high-authority and dangerous under wrong advice; the
constraint is exactly as good as the sign it is given. No corruption sweep was
run — that would invent a sensor-accuracy curve nothing here supports.

## 14. Classification: **B**

Against the DIRECTION's criteria for the bilateral result:

- **Hinge collateral is removed** — not reduced, *removed*: every one of the
  four dependent hinge fields returns to bit-identical baseline agreement, and
  the collateral transition counts go to exactly zero.
- **Bilateral benefit remains substantially below learned conditioning** —
  39% of the learned yaw gain, ~9% of the learned hip-residual gain, and the
  shoulder residual moves the wrong way.

That is **B**: a hybrid architecture is supported. It is B with one deviation
worth stating plainly — B anticipates an unacceptable continuous-geometry cost,
and the actual cost is **negative** (MPJPE improves by 0.78 mm over the
baseline). The constraint is not rejected because it is expensive; it is
insufficient because a sign carries no magnitude.

**Separately, docs/33's generalisation is refuted.** "Constraint good iff the
field is a leaf" does not hold. The bilateral collateral was **100% a
write-policy fault**: with a dependency-aware write it disappears entirely,
while the anchor is still an anchor. An explicit constraint over an anchor is
perfectly viable — it simply has to move the geometry that depends on the
anchor. What limits the bilateral channel is the **contract**, not the
graph position.

## 15. Does the evidence support the hybrid?

```
hinge      -> explicit parameter-free branch constraint
bilateral  -> learned conditioning
```

**Yes, on the evidence so far.** The hinge constraint cuts the flip rate 42.5%
for +0.084 mm MPJPE and beats every learned hinge conditioning tested
(docs/33). The bilateral constraint, even with the collateral fully repaired,
recovers only ~39% of learned conditioning's yaw benefit and cannot address the
depth magnitude at all.

Two caveats hold the promotion back, and nothing is promoted here:

1. Everything rests on an **oracle** sign. Both channels are catastrophically
   sensitive to a wrong one (Section 13), and no sensor exists.
2. The two channels are **not independent**: they write into overlapping read
   sets, and a combined hinge-constraint + learned-bilateral system was never
   run. Only the isolated channels were measured.

## 16. What this batch does NOT establish

- No sensor, no accuracy threshold, no confidence weighting, no temporal
  voting. The sensor remains outside this batch.
- No claim that the one-bit hinge sign should change. Outcome B is recorded as
  a measured limit; the Sign Contract is untouched, as directed.
- The `V_OBSERVED` applicability control is **uninformative on this bank**
  because the two masks are bit-identical (Section 5). It is not evidence that
  `input_valid` behaves like `target_valid` in general.
- Regime `benchmark_detector_observation` throughout; one source candidate, one
  split.
- The dependency-aware chain is the *anatomical* chain. Whether carrying the
  full chain is right for a rig with different topology is untested.
