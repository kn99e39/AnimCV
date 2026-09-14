# Worklog — Pose Reconciliation Field-Matched Semantics (2026-09-14)

> Continuation of docs/42 on `arch/single_frame_first`. This batch closes the
> measurement-granularity gap without changing the endpoint-fixed solver or any
> historical policy.

## 1. Preserved boundary

Starting HEAD:

```text
f589747 fix: make pose replay accounting cohort-matched
```

Preserved unchanged:

```text
Pose Reconciliation solver and endpoint-fixed swivel architecture
cohort A/B/C/D/E definitions
opposite-report binding
UNKNOWN exclusion and requested coverage identity
unresolved R_SWIVEL_OBS rows in primary C
D as secondary common-corrected diagnostic
normalized_full_image provenance
```

The only executable policy-file change is a docstring correction in
`src/framepose/branch_constraints.py`: `DEPTH_ONLY` preserves canonical X/Z,
but perspective projection can still move image position when Y changes because
the projected coordinates contain X/Y and Z/Y. Operator behavior is unchanged.

## 2. Exact field-level measurements

The replay keeps the existing union-frame metrics for whole-pose MPJPE,
PA-MPJPE, per-joint error, root yaw, and bilateral quantities. It additionally
reports, for every field in C, D, and E, using that field's exact chain rows:

```text
chain-frame count
requested-sign satisfaction
canonical evaluator hinge flip rate
canonical evaluator bend-direction error
middle-joint 3D error
P-M / M-D absolute length changes
middle-joint target-projection error
middle-joint observation-consistency error
```

The P-M and M-D metrics expose count/mean/p50/p90/p99/max. The exact field
rows prevent untouched chains on the same union frame from diluting ownership
measurements. The existing union-frame endpoint/bone/global summaries remain as
secondary whole-pose views.

## 3. Same-C observation attribution

`R_SWIVEL_OBS` and `R_SWIVEL_ORACLE_2D` are now both included on the exact
same C rows. C is not redefined and still requires the real normalized full
image observation to be present. The report has a dedicated
`same_C_observation_attribution` section with the two candidates side by side
per field.

E remains separate as the broader target-projection-eligible oracle upper-bound
population. For every E field, `frames` and the finite
`observation_consistency_metric_count` are exposed separately; missing or
invalid observed 2D values are not converted to zero error.

D remains secondary and corrected-only: it is used for mechanism-on-success
attribution, never for the primary architecture verdict.

## 4. Focused validation

The focused contracts cover:

```text
exact field rows for hinge metrics
exact field rows for P-M / M-D bone metrics
untouched chains do not dilute field-level bone ownership
same C rows for OBS and ORACLE_2D
finite E observation metric count versus E frame count
union-frame whole-pose metrics remain present
perspective clarification is docstring-only
```

## 5. Exact real-scene replay gate

The exact camera raw data is present at:

```text
/Users/nadan/Projects/Lab/DATASET_Motion/sequenceFiles/test/*.pkl
```

The complete docs/39 replay input set is not present. The following required
artifacts were not found anywhere under `/Users/nadan`:

```text
bank_3dpw_paired_v2.json
  required identity: content digest 75519e63…0ed536

stored O_BILATERAL prediction
  historical candidate: O_BILATERAL_oracle_forward_depth_only
  documented lineage: sign_attr_v2/O_BILATERAL/prediction_test.npy

matching evaluation_test.json
matching historical artifact:
  hinge_ownership_v3/hinge_evidence_ownership.json
```

Because the bank, stored prediction/evaluation, and accepted historical
artifact are missing, the exact historical replay was not run. The available
raw pickles alone are insufficient and were not used as a substitute. No
real-scene numbers or A/B/C/D architecture classification are claimed.

No tuning, solver modification, fallback policy, RGB input, temporal voting,
SignState change, training, endpoint change, or production promotion occurred.
