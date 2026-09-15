# Worklog — Pose Reconciliation Causal Attribution (2026-09-15)

Branch: `arch/single_frame_first`

Starting HEAD: `3f4e024377fa7661cc7bd2b8459a5a03798e916f`

This is a new diagnostic lineage. The docs/44 recovery record and real-replay
report remain unchanged; their historical classification remains
`C — SWIVEL ABSTRACTION INSUFFICIENT`.

## Verdict

**Diagnostic classification: D — MIXED. Stop here; no architecture or policy
change is made.**

The real replay does not falsify endpoint-fixed swivel geometry wholesale, but
it also cannot be explained away as only the `H0 == UNKNOWN` refusal:

- In the 1,390 `C_READABLE_WRONG` chain rows, OBS corrects 1,177 and Oracle
  corrects 1,195. The remaining 213 / 195 rows fail the final canonical
  SignState read-back; none are attributed to geometric infeasibility or the
  H0-UNKNOWN refusal.
- Separately, on the 1,276 `C_H0_UNKNOWN` rows, the diagnostic counterfactual
  finds an exact requested-sign swivel solution on 768 OBS rows and 786 Oracle
  rows. That is a real limitation of the current conservative refusal, but it
  contributes **zero** to the historical `MINIMUM_NORM corrected − swivel
  corrected` gap: MINIMUM_NORM itself corrects none of these unknown rows.
- Where MINIMUM_NORM and swivel do correct the request, both preserve the
  metric bones and endpoints. Swivel has lower image errors; MINIMUM_NORM has
  better continuous full-3D bend-direction metrics. The latter distinction is
  not direct evidence against a one-bit SignState contract.

Thus there are separate refusal, final-read-back, and continuous-geometry
contributions. The historical C label is preserved, but it is not the causal
classification of this follow-up. No policy promotion or architecture choice
is inferred.

## 1. Exact docs/44 evidence reused

No training, prediction regeneration, replay regeneration, camera conversion,
or input substitution was performed. The diagnostic read and cross-verified
the recovered artifacts used by docs/44:

| Evidence | Identity |
|---|---|
| docs/44 replay report | SHA-256 `61fde30e7e6d8d9769a0273450819478fc6c457af3841eae3d0333c1f8e16e8c` |
| FrameBank | content digest `75519e6394a764e3749ddaa30555b58b73a01db582ccee14f661374b9a0ed536` |
| O_BILATERAL prediction | SHA-256 `6df04c2031b2bde6a4f0af37fdae0a7f0228e9fc23401e15e1aa2b3bd35695a5` |
| O_BILATERAL evaluation | SHA-256 `a5881850b3769ca2329271d327ee895dc646f236451f05c3ba2e0ff917f580a3` |
| Raw 3DPW camera inputs | 24 files; bytes and SHA-256 matched docs/44; 7,076/7,076 camera frames usable |

The same recovered FrameBank, O_BILATERAL prediction/evaluation, and raw test
camera root were passed to the new diagnostic. The replay report SHA was
checked before analysis. The exact report remains at
`/private/tmp/animcv-pose-replay.lKDzaA/pose_reconciliation_replay.json`;
the new transient measurement output is outside the repository under
`/private/tmp`.

## 2. C partition by canonical H0 read-back

Partitioning uses the existing canonical `sign_state` result. Each field has a
disjoint union identity: `C = C_READABLE_WRONG ∪ C_H0_UNKNOWN`.

| Hinge field | C | C_READABLE_WRONG | C_H0_UNKNOWN | Overlap |
|---|---:|---:|---:|---:|
| Left elbow | 759 | 406 | 353 | 0 |
| Right elbow | 591 | 306 | 285 | 0 |
| Left knee | 592 | 268 | 324 | 0 |
| Right knee | 724 | 410 | 314 | 0 |
| **Pooled chain rows** | **2,666** | **1,390** | **1,276** | **0** |

The pooled count is chain rows, not unique frames. The per-field union and
zero-overlap assertions passed.

## 3. C unresolved-reason accounting

Counts below are `H0-UNKNOWN early refusal / final SignState read-back
failure / total unresolved`. All other requested reason buckets are zero on C
for both variants: invalid chain, invalid observed middle, invalid/unavailable
ProjectionContext, degenerate bone circle, below-floor radius, unobservable
axis/branch, no finite positive-depth candidate, endpoint/bone invariant
failure, and other.

| Field | OBS | ORACLE_2D |
|---|---:|---:|
| Left elbow | 353 / 71 / 424 | 353 / 64 / 417 |
| Right elbow | 285 / 62 / 347 | 285 / 65 / 350 |
| Left knee | 324 / 40 / 364 | 324 / 25 / 349 |
| Right knee | 314 / 40 / 354 | 314 / 41 / 355 |
| **Pooled** | **1,276 / 213 / 1,489** | **1,276 / 195 / 1,471** |

Each field and pooled taxonomy exactly partitions its unresolved count. The
shared 1,276 early refusals are precisely `C_H0_UNKNOWN`; the residual
unresolved counts are final canonical SignState read-back failures.

## 4. Coverage-gap closure

On full C, MINIMUM_NORM corrects exactly the readable-wrong population
(1,390); it has 0 corrected / 1,276 unresolved on `C_H0_UNKNOWN`. Exact
arithmetic decomposition:

| Swivel variant | MINIMUM_NORM corrected | Swivel corrected | Gap | H0-UNKNOWN early refusal | Geometric infeasibility | Final read-back / invariant | Other | Swivel-only counter-contribution |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| OBS | 1,390 | 1,177 | 213 | 0 | 0 | 213 | 0 | 0 |
| ORACLE_2D | 1,390 | 1,195 | 195 | 0 | 0 | 195 | 0 | 0 |

All per-field and pooled identities closed. Per-field OBS gaps are 71 / 62 / 40
/ 40 (left elbow / right elbow / left knee / right knee); Oracle gaps are 64 /
65 / 25 / 41. These are all final read-back failures. Therefore the historical
corrected-count deficit versus MINIMUM_NORM is **not caused by** the
H0-UNKNOWN early refusal or geometric infeasibility.

## 5. Same-row `C_READABLE_WRONG` comparison

The tables retain the exact same row set per field for every candidate. For
the policy-level metrics below, unresolved swivel outcomes remain at H0; they
are not dropped. Candidate order in tuples is `MINIMUM_NORM / OBS /
ORACLE_2D`.

| Field (n) | Corrected / unresolved | Requested sign satisfied | Full-row bend MAE (°) | Full-row hinge flip (%) |
|---|---|---|---|---|
| Left elbow (406) | 406/0 · 335/71 · 342/64 | 406 · 335 · 342 | 29.63 / 39.00 / 38.35 | 0.99 / 6.40 / 5.67 |
| Right elbow (306) | 306/0 · 244/62 · 241/65 | 306 · 244 · 241 | 21.93 / 32.52 / 32.09 | 1.31 / 4.90 / 5.56 |
| Left knee (268) | 268/0 · 228/40 · 243/25 | 268 · 228 · 243 | 21.51 / 39.53 / 33.57 | 1.12 / 10.07 / 6.34 |
| Right knee (410) | 410/0 · 370/40 · 369/41 | 410 · 370 · 369 | 20.41 / 32.09 / 31.24 | 0.00 / 2.44 / 1.95 |

Mean image errors in pixels, again over the full same-row populations:

| Field | Target projection (MN / OBS / Oracle) | Observation consistency (MN / OBS / Oracle) |
|---|---:|---:|
| Left elbow | 30.88 / 23.95 / 22.78 | 30.04 / 21.92 / 23.56 |
| Right elbow | 33.86 / 28.55 / 26.53 | 35.66 / 28.63 / 30.46 |
| Left knee | 18.95 / 16.53 / 14.26 | 23.40 / 18.85 / 20.84 |
| Right knee | 23.21 / 19.90 / 18.27 | 25.10 / 19.96 / 21.52 |

Swivel therefore owns both image-space errors on all four fields against
MINIMUM_NORM on this policy-level same-row comparison. Oracle improves target
projection relative to OBS in each field; its observation-consistency error is
higher than OBS in each field.

Metric ownership is matched: MINIMUM_NORM and both swivel variants have
P–M/M–D length changes at numerical zero (worst field maxima no greater than
`2.78e-13 mm`) and proximal/distal endpoints exactly unchanged (`0 mm`).
DEPTH_ONLY is included in the diagnostic output as a historical control, but
does not share this metric-bone result; its largest C_READABLE_WRONG changes
are about 2,952 mm (P–M) and 2,489 mm (M–D).

Full-row middle-joint 3D error (mm; unresolved rows remain H0):

| Field | MINIMUM_NORM / OBS / ORACLE_2D |
|---|---:|
| Left elbow | 138.47 / 133.41 / 134.06 |
| Right elbow | 124.08 / 113.86 / 111.43 |
| Left knee | 85.35 / 93.18 / 88.99 |
| Right knee | 72.79 / 68.36 / 66.33 |

On the 1,293-frame union of these readable field rows, whole-pose
MPJPE / PA-MPJPE (mm) / hinge flip / hinge-direction MAE were: H0
`80.720 / 57.464 / 5.30% / 33.394°`; MINIMUM_NORM
`81.008 / 58.109 / 1.30% / 22.619°`; OBS
`80.747 / 57.861 / 2.84% / 26.805°`; Oracle
`80.636 / 57.871 / 2.51% / 26.221°`. These are whole-pose guardrails on the
same union, not a scalar score.

## 6. `C_H0_UNKNOWN` and the labelled counterfactual

Current policy and MINIMUM_NORM both have 0 corrected / all unresolved on
these exact rows. The diagnostic then directly invoked the existing
bone-circle and swivel solver, bypassing only the prestate `before == UNKNOWN`
refusal. The counterfactual also reapplied the current finite-observation,
ProjectionContext, exact SignState read-back, endpoint, and bone checks.

Every cell is `feasible requested-sign solution / no feasible solution /
positive-depth failure / final read-back failure`:

| Field | OBS counterfactual | ORACLE_2D counterfactual |
|---|---:|---:|
| Left elbow (353) | 207 / 91 / 0 / 55 | 216 / 91 / 0 / 46 |
| Right elbow (285) | 173 / 60 / 0 / 52 | 180 / 60 / 0 / 45 |
| Left knee (324) | 142 / 159 / 0 / 23 | 150 / 159 / 0 / 15 |
| Right knee (314) | 246 / 37 / 0 / 31 | 240 / 37 / 0 / 37 |
| **Pooled (1,276)** | **768 / 347 / 0 / 161** | **786 / 347 / 0 / 143** |

OBS/Oracle feasibility intersection per field (`both / OBS-only / Oracle-only /
neither`) is left elbow `203/4/13/133`, right elbow `171/2/9/103`, left knee
`138/4/12/170`, right knee `235/11/5/63`. Pooled: 747 both, 21 OBS-only, 39
Oracle-only, 469 neither. Each counterfactual outcome partition closes exactly.

This establishes that the refusal hides a finite exact-sign solution on 768
OBS rows (786 Oracle rows), while 347 rows per method have no feasible solver
solution and 161 / 143 fail final read-back. This is a diagnostic label only:
`COUNTERFACTUAL_SWIVEL_WITHOUT_PRESTATE_REFUSAL`. It is not a candidate,
production behavior, or a promotion proposal.

## 7. SignState contract and continuous bend geometry

- **SignState contract:** the requested hidden forward/depth branch only.
- **MINIMUM_NORM:** installs that branch using a specific reflected
  perpendicular bend geometry.
- **R_SWIVEL:** selects the minimum-reprojection point inside the requested
  readable branch arc.

Requested-sign read-back is contract-direct evidence. Full-3D hinge flip,
bend-direction MAE, and middle-joint 3D target error remain important
continuous-geometry guardrails; they are not equivalent to satisfying the
one-bit SignState. The numbers are retained, not discarded.

On each candidate's own corrected readable rows, the imposed bend-direction
angular displacement from H0 and target full-3D bend MAE (degrees) are:

| Field | Corrected rows (MN / OBS / Oracle) | Bend displacement° (MN / OBS / Oracle) | Target bend MAE° (MN / OBS / Oracle) |
|---|---:|---:|---:|
| Left elbow | 406 / 335 / 342 | 67.23 / 67.28 / 67.12 | 29.63 / 35.04 / 34.65 |
| Right elbow | 306 / 244 / 241 | 52.72 / 54.13 / 51.61 | 21.93 / 28.50 / 26.97 |
| Left knee | 268 / 228 / 243 | 56.02 / 63.91 / 63.03 | 21.51 / 36.43 / 30.73 |
| Right knee | 410 / 370 / 369 | 56.37 / 71.27 / 69.09 | 20.41 / 31.20 / 29.98 |

MINIMUM_NORM's full-3D bend MAE is lower on every field. Its bend-direction
displacement is not uniformly larger: it is similar on the elbows and smaller
on both knees than OBS. Thus the result is not simply “MINIMUM_NORM moves
farther”; it installs a different continuous direction. In return, the
corrected-only image errors favor swivel on all fields:

| Field | Target projection px (MN / OBS / Oracle) | Observation consistency px (MN / OBS / Oracle) |
|---|---:|---:|
| Left elbow | 30.88 / 23.90 / 22.11 | 30.04 / 21.21 / 23.02 |
| Right elbow | 33.86 / 29.78 / 27.28 | 35.66 / 29.55 / 32.01 |
| Left knee | 18.95 / 16.84 / 14.33 | 23.40 / 19.30 / 21.13 |
| Right knee | 23.21 / 19.79 / 17.94 | 25.10 / 19.88 / 21.30 |

These corrected-only rows are candidate-specific successful rows, while the
previous section's full-row table is the same-population policy comparison.
The continuous 3D direction advantage belongs to MINIMUM_NORM's extra
geometric choice; SignState alone does not specify that choice.

## 8. Pareto accounting — no composite score

On the exact readable rows, the trade-off is non-dominated:

- MINIMUM_NORM wins direct requested-sign coverage (1,390/1,390 vs
  1,177/1,390 OBS and 1,195/1,390 Oracle) and continuous bend-direction
  guardrails.
- OBS/Oracle win both image-space ownership errors against MINIMUM_NORM on all
  four fields, while matching its numerical-zero metric bones and exact
  endpoints.
- OBS and Oracle differ in their own image axes: Oracle has lower target
  projection error, while OBS has lower observed-image consistency error.
- Whole-pose MPJPE/PA do not reverse all of those independent axes.

No weighted sum, chosen threshold, or composite score was computed. No
candidate dominates the other across requested coverage, image ownership,
metric bones/endpoints, and continuous 3D direction.

## 9. Wrong-sign safety attribution

Wrong-sign is partitioned on the exact existing opposite-request C rows.
MINIMUM_NORM corrects all readable-wrong rows and leaves the H0-UNKNOWN rows
unresolved. OBS additionally abstains on 2,762 readable rows after final
SignState read-back failure.

| Field | Wrong C | H0 unknown | MINIMUM_NORM corrected / unresolved | OBS corrected / unresolved | Extra readable abstentions |
|---|---:|---:|---:|---:|---:|
| Left elbow | 5,022 | 353 | 4,669 / 353 | 3,964 / 1,058 | 705 |
| Right elbow | 5,255 | 285 | 4,970 / 285 | 4,374 / 881 | 596 |
| Left knee | 5,781 | 324 | 5,457 / 324 | 4,737 / 1,044 | 720 |
| Right knee | 5,635 | 314 | 5,321 / 314 | 4,580 / 1,055 | 741 |
| **Pooled chain rows** | **21,693** | **1,276** | **20,417 / 1,276** | **17,655 / 4,038** | **2,762** |

On the 6,966-frame wrong-C union, whole-pose MPJPE / PA-MPJPE (mm) were H0
`79.213 / 56.398`, MINIMUM_NORM `92.286 / 72.876`, and OBS
`87.651 / 67.908`. That aggregate includes abstentions and must not be read as
the corrected-displacement effect alone.

On the exact same **field rows where OBS corrected**, its middle-joint 3D
error and observation-consistency error were lower than MINIMUM_NORM for all
four fields:

| Field | Middle-joint 3D error mm (MN / OBS) | Observation consistency px (MN / OBS) |
|---|---:|---:|
| Left elbow | 148.90 / 137.61 | 26.96 / 18.99 |
| Right elbow | 157.76 / 143.42 | 29.02 / 21.37 |
| Left knee | 161.33 / 142.16 | 30.09 / 24.59 |
| Right knee | 175.98 / 151.70 | 32.09 / 24.94 |

So the lower wrong-sign aggregate damage has **both** causes: additional
abstention (2,762 readable wrong-sign chain rows versus MINIMUM_NORM), and a
genuinely less damaging corrected swivel displacement on the shared
corrected rows. Because hinge rows overlap in whole-pose frames and multiple
field edits interact in the evaluator, no additive millimeter decomposition
is claimed.

## 10. OBS vs ORACLE after refusal partition

On `C_READABLE_WRONG`, OBS corrects 1,177 and Oracle 1,195: a net Oracle gain
of 18/1,390 (1.29 percentage points). Per-field count changes (Oracle minus
OBS) are left elbow `+7`, right elbow `−3`, left knee `+15`, right knee `−1`.
The image and continuous metrics above show mixed, modest field-level changes;
Oracle does not close MINIMUM_NORM's coverage or full-3D direction gap.

On the H0-UNKNOWN counterfactual, Oracle has 786 feasible rows versus OBS
768; both are feasible on 747, Oracle-only on 39, OBS-only on 21. This remains
a modest aggregate change after isolating the refusal population.

## 11. Tests, scope, and stop condition

Focused tests: **5 passed** in
`PYTHONPATH=.:src .venv/bin/pytest -q tests/test_pose_reconciliation_attribution_diagnostic.py`.
They cover C partition identity, unresolved-reason identity, coverage-gap
closure, corrected/unresolved partitioning, and counterfactual non-mutation
while verifying production still refuses H0-UNKNOWN. Python compilation also
passed for the diagnostic and focused test module. No full regression was
needed because no production/shared behavior changed.

The only repository additions in this batch are the diagnostic script,
focused test module, and this Markdown worklog. No solver, refusal policy,
branch constraint behavior, DEPTH_ONLY, MINIMUM_NORM, SignState, cohort
definition, or docs/44 artifact/report was changed. docs/44's replay SHA
remains unchanged.

**Stop condition honored.** The follow-up causally attributes the result as
mixed and makes no architecture change. A project-owner decision is required
before deciding whether known requested SignState may authorize reconciliation
from an H0-UNKNOWN prestate, or whether continuous bend direction belongs to
Pose Reconciliation and what evidence may own it.
