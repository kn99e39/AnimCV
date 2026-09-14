# Worklog — Pose Reconciliation Replay Accounting and Cohorts (2026-09-14)

> Continuation of docs/41 on `arch/single_frame_first`. This batch repairs the
> controlled replay's accounting boundary. The endpoint-fixed solver and the
> historical branch policies are unchanged.

## 1. Scope and preserved architecture

Starting point:

```text
branch: arch/single_frame_first
HEAD:   1b7c505 feat: add endpoint-fixed pose reconciliation
```

The batch does not modify `src/framepose/pose_reconciliation.py`, the
historical `DEPTH_ONLY` / `MINIMUM_NORM` operators, SignState semantics, the
historical default, or docs/33–41. It changes only the replay accounting and
adds focused accounting tests.

## 2. Accounting repairs

The replay now binds an opposite state explicitly to its base candidate report:

```text
DEPTH_ONLY__opposite       -> wrong_reports[DEPTH_ONLY]
MINIMUM_NORM__opposite     -> wrong_reports[MINIMUM_NORM]
R_SWIVEL_OBS__opposite     -> wrong_reports[R_SWIVEL_OBS]
```

Requested-sign accounting reads only requested values in `{-1, +1}`. UNKNOWN
is reported separately as excluded and can no longer be counted as satisfied.
For every field, candidate, and reported cohort the replay enforces:

```text
requested_count = already_satisfied + corrected + unresolved
```

The final read-back is counted independently as `satisfied_count`.

## 3. Matched cohorts

The report carries per-field masks and exact frame indices for all five
cohorts:

| cohort | definition | role |
|---|---|---|
| A | all test frames | normal evaluator |
| B | known oracle request, H0 conflict, valid proximal-middle-distal chain | requested-conflict population |
| C | B plus valid ProjectionContext and observed middle joint | primary operational observation cohort |
| D | C where DEPTH_ONLY, MINIMUM_NORM, and R_SWIVEL_OBS all report `corrected` | secondary corrected-only mechanism diagnostic |
| E | B plus valid camera context and target middle projection | R_SWIVEL_ORACLE_2D upper-bound cohort |

C is defined before candidate outcomes are inspected. Therefore unresolved
`R_SWIVEL_OBS` rows remain in the primary result and are evaluated as the
unchanged H0 pose when reconciliation refuses to act. Image accounting for
H0, DEPTH_ONLY, MINIMUM_NORM, and R_SWIVEL_OBS uses the exact same C rows per
field. D is reported separately and is never substituted for C.

The opposite-request endpoint builds the same explicit B/C/D cohorts from the
opposite requested signs. Its primary and secondary matched endpoint results
are kept under the base candidate name with an explicit `state_name`.

## 4. Observation provenance

The replay report now records:

```text
observation_coordinate_space: normalized_full_image
```

This is the stored normalized full-image `input_2d` convention used by the
reconciliation layer. The oracle upper bound projects the target middle joint
through the same per-frame calibrated context and remains separate from the
stored-observation result.

## 5. Validation

Focused tests cover:

```text
opposite report binding
UNKNOWN exclusion and requested-coverage identity
C retaining unresolved observation rows
D being a strict corrected-only secondary subset
identical matched image rows across the four primary candidates
explicit normalized_full_image provenance
```

The focused accounting and solver tests pass:

```text
10 passed
```

## 6. Exact real-scene replay status

The exact docs/39 data-mounted inputs are not present in this workspace. A
read-only search found no `bank_3dpw_paired_v2` artifact, stored
`O_BILATERAL` prediction/evaluation, or `hinge_ownership_v3` replay artifact
under the repository or Codex attachment area; no mounted `/data` or `/mnt`
input tree is available. The replay was therefore not executed and no
numerical A/B/C/D classification is claimed.

The command remains:

```text
PYTHONPATH=src python scripts/replay_pose_reconciliation.py \
  --bank <docs/39 bank> \
  --source O_BILATERAL=<candidate>:<prediction.npy>:<evaluation.json> \
  --raw-root <3DPW raw root> \
  --out <output directory>
```

This is an artifact-availability limitation, not a substitute dataset or
camera reconstruction. No tuning, solver redesign, historical policy change,
training, or production promotion was performed.
