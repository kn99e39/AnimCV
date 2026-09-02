# Worklog — Historical Audit Repair + Corrected Bilateral Forward-Depth A/B (2026-09-02)

> A15 (compiled A9 operational control) / A16 (corrected bilateral
> forward-depth candidate, descriptive label: `bilateral_forward_depth_
> supervision_corrected`). **A16 is not historical A14** — it uses a
> separate `TrainingConfig` flag, a separate checkpoint, and the compiled
> execution backend. Historical A9–A14 checkpoints/fingerprints/reports are
> untouched. Quantitative source of truth remains
> `docs/10_TEMPORAL_LIFTER_IMPROVEMENT_ABLATION.md`.

## 1. Historical audit corrections

Repository audit found that historical A14's coordinate loss changed the
denominator (`D_coord`) from `mask.sum()` to `mask.sum() +
relational_pair_count`, unintentionally attenuating the base A9 coordinate
gradient. Distinguishing **measured result** from **causal interpretation**:

| Run | Measured result | Causal interpretation |
| --- | --- | --- |
| A9 | valid | valid historical baseline |
| A10 | valid | valid controlled negative experiment (end-effector weight) |
| A11 | valid | valid controlled negative training experiment (angular yaw-tail collapse) |
| A12 | valid | valid controlled negative training experiment (Cartesian torso-tail, stable but doesn't improve yaw) |
| A14 | valid **for the objective actually executed** | **NOT** a pure bilateral-forward-depth causal experiment — `D_coord` was contaminated |

Historical A14's checkpoint/report remain the correct evidence for "what
happens when you add this relational term AND grow the coordinate
denominator by relational pair count" — a real, valid, but different
question than the one docs/10 A14 set out to answer.

## 2. A11 exact-selector re-diagnosis

New diagnostic-only `scripts/repair_historical_selector_diagnostics.py`
adds `_exact_evaluator_yaw_degree_grid`/`_exact_evaluator_frame_combined_grid`,
reproducing `_root_yaw_error_degrees`'s own `arctan2` + angle-wrap math
exactly (verified against it directly, both per-pair and per-frame,
focused tests), replayed on the same 10 fixed first-epoch batches (seed
`1337`, same augmentation/source-balanced permutation) at init/A9/A11
states used by the original A11 gradient diagnosis. No new training.

| Model state | pooled production ratio | old (1-cos) frame ratio | **exact-angle frame ratio** |
| --- | ---: | ---: | ---: |
| init | 1.69 | 10.23 | 1,611.29 |
| A9 trained | 3,169.92 | 3,199.72 | **302,884.19** |
| A11 trained | 157.14 | 134.94 | **18,560.57** |

Selected-frame Jaccard overlap at A9 state: pooled-vs-old-frame `.616`,
pooled-vs-exact-frame `.571`, old-frame-vs-exact-frame `.775`.

**The corrected, evaluator-exact selector shows an even MORE extreme
gradient-scale ratio than either historical proxy, not a smaller one.**
The historical conclusion — *"selector granularity is not the primary
explanation for A11 collapse; the mechanism is gradient-scale dominance"*
— **still holds, and is reinforced** by the corrected diagnostic: a
"more correct" selector does not fix the pathology, it makes the
magnitude mismatch worse. Historical A11 training and its Case
B(gradient-scale)/Case D(selector structure excluded) verdict are
unchanged. No A11 retraining performed.

**Wording correction (docs/22 Section 1)**: the
`exact_evaluator_angle_frame_selector` measured above both *ranked* and
*differentiated* the real evaluator degree value, conflating selection
semantics with penalty representation/scale. It is retained here as a
valid measurement, but the selector-structure exclusion this section draws
should be read as resting on the *properly isolated* P1/P2/P3 comparison
in docs/22 Section 1 (P3: exact-angle ranking, same (1-cos) differentiated
penalty as P2), which reaches the identical conclusion through a cleaner
control (P1/P2/P3 ratios all land in the same narrow band — 3,170/3,200/
3,325 at A9 state — regardless of selection rule, isolating the pathology
to penalty representation alone). The number reported here is re-labeled
in docs/22 as `angle_penalty_diagnostic`, a different (still valid, still
extreme) question about penalty representation, not selector structure.

## 3. A12 true evaluator-yaw association

The historical `_yaw_association` used `(1 - cos(theta)) * 180/pi` as a
"yaw-degree" surrogate — not a degree quantity (scales as `theta^2` for
small `theta`, not `theta`) and not what the production evaluator
measures. Recomputed the same three Pearson correlations against the real
per-frame `_root_yaw_error_degrees`-equivalent error, same fixed batches.

| Metric | Old (invalid proxy) at A12 state | **Corrected (real evaluator degrees) at A12 state** |
| --- | ---: | ---: |
| a12_cartesian | .325 | **.496** |
| magnitude | -.056 | **.177** |
| direction_scale_restored | .610 | **.517** |

The old values are marked **INVALID/SUPERSEDED**. The corrected values
still rank `direction_scale_restored` highest among the three at both A9
(.478) and A12 (.517) trained states, and no value flips the qualitative
story that direction captures more orientation signal than magnitude
alone. **The historical A13 NO-GO decision is unchanged**, because it
never rested on these Pearson values in the first place — it rested on
the direct squared-energy identity decomposition
(`||v_pred-v_gt||^2 = magnitude^2 + direction_energy`, magnitude 20.4%/
direction 79.6% at A12 state, docs/15), a separate, always-correct
computation this repair does not touch. This distinction is recorded
explicitly per the standing instruction: do not change A13 unless the
corrected diagnostic invalidates its actual prerequisite -- it does not.

## 4. Corrected SRD mathematical definition

Identical to historical A14's definition — same joint pairs, same axis,
same normalization, same robust-loss family:

```
q_shoulder = (y_right_shoulder - y_left_shoulder) / sqrt(2)
q_hip      = (y_right_hip - y_left_hip) / sqrt(2)
```

canonical `+Y` = forward/depth (`FORWARD_DEPTH_AXIS = 1`, verified in
`pose_lifter.py`'s canonical camera frame), Smooth-L1 default beta,
`1/sqrt(2)` fixed orthonormal-basis normalization (not tunable). Nothing
here changed from A14.

## 5. Reduction derivation

The only thing this batch changes:

```
Historical A14 (unchanged, preserved as executed):
    coordinate = (S_coord + S_relational) / (D_coord + relational_count)

docs/21 corrected (new, separate flag):
    coordinate = (S_coord + S_relational) / D_coord
               = S_coord/D_coord + S_relational/D_coord
```

`bilateral_forward_depth_supervision` (historical A14 flag) and
`bilateral_forward_depth_supervision_corrected` (new flag) are mutually
exclusive by `__post_init__` construction — running both raises
`ValueError`, so historical A14's exact contaminated behavior can never be
silently altered by this batch's code.

## 6. Synthetic contracts

`tests/test_bilateral_forward_depth_corrected.py`, 12 contracts, all
PASS: corrected-off matches plain A9 exactly; corrected-on with zero
relational residual matches A9 exactly (loss AND gradient, `atol=1e-7`);
`Delta_G` (candidate − A9) equals the relational term's own gradient
exactly (linearity of differentiation over the unmodified `D_coord`);
common-mode `+Y` translation invariance; anti-symmetric-error opposite
endpoint gradients; zero gradient leakage to unrelated joints; shoulder/
hip independence; pair invalidity zeroes the relational contribution
**without** touching the denominator; canonical `+Y` axis; `1/sqrt(2)`
normalization; both-flags-disabled exact historical reduction;
mutual-exclusivity rejection.

## 7. Actual gradient-delta diagnosis

New `scripts/diagnose_corrected_srd_gradient_delta.py`, same fixed-batch
replay as Section 2/historical A14 diagnosis, real model states (init, A9
checkpoint), 10 batches:

| Model state | mean cosine(G_A9, Delta_G) | mean \|\|Delta_G\|\|/\|\|G_A9\|\| | max Delta_G vs G_relational rel diff | max loss decomposition abs diff |
| --- | ---: | ---: | ---: | ---: |
| init | 0.121 | 0.045 | 0.0012 | 1.1e-08 |
| A9 trained | 0.208 | 0.055 | 0.0064 | 1.3e-10 |

`Delta_G` matches the relational term's own gradient within **<0.65%**
relative difference on real data (float32 accumulation-order noise
through a real backward pass, not a discrepancy). Total loss decomposes
additively to within `1e-8`–`1e-10`. Ratio (`||Delta_G||/||G_A9||` ≈
4.5–5.5%) is far below A11's historical 3,170× and even below A12's
5.28× and the contaminated A14's own 0.375–0.63× — **no A11-like
gradient dominance**. cosine values (0.121/0.208) match the earlier
(differently-normalized) A14 diagnosis almost exactly, an internal
consistency check that both measure the same gradient *direction*.

## 8. Compiled A9 operational-control result

`ablation_a15_compiled_a9_control_10e`: exact historical A9 recipe +
`compile_training_graph=True` only. All 6 dataset fingerprints match A9
exactly.

| Holdout | metric | A9 (eager) | **A15 (compiled control)** | delta |
| --- | --- | ---: | ---: | ---: |
| 3DPW test | PA-MPJPE mm | 75.31 | 75.63 | +0.32 |
| 3DPW test | yaw MAE ° | 14.90 | 14.37 | -0.53 |
| 3DPW test | yaw P95 ° | 34.77 | 36.16 | +1.39 |
| AMASS internal | PA-MPJPE mm | 69.19 | 69.65 | +0.46 |
| AMASS internal | yaw MAE ° | 8.77 | 9.24 | +0.47 |
| AMASS internal | yaw P95 ° | 22.37 | 24.37 | +2.00 |
| training MPJPE mm | — | 40.19 | 40.35 | +0.16 |
| runtime (10 epoch) | seconds | 1,335.6 | 919.4 | **-31.2%** |
| throughput | samples/s | 3,470.9 | 5,041.8 | **+45.3%** |

**Wording correction (docs/22 Section 2)**: the deltas above are comparable
in *magnitude* to historical eager-to-eager differences (e.g. A9 vs A10),
but that is not the same claim as "ordinary run-to-run variance" — compiled
execution preserves the mathematical training objective exactly but forms
a **distinct numerical execution lineage** from eager (different kernel
fusion, different floating-point reduction order end to end). It is a GO
for using the compiled backend for new quality experiments, and it means
architecture A/B comparisons after A15 must stay on the same compiled
backend (A15, not historical eager A9) — which docs/21 already did for the
A16 comparison (Section 11) — but the deltas themselves should not be
described as if they were just another eager run.

## 9. GO / NO-GO for candidate training

**GO.** Compiled A9 trains stably, preserves the A9 quality regime (no
new gate failures; existing 3DPW yaw-P95 failure, already present in
eager A9, is unchanged in kind), and the checkpoint/report carry correct
`execution_backend` provenance. Proceeded to Section 10.

## 10. Corrected candidate result

`ablation_a16_bilateral_forward_depth_corrected_10e`: A15's exact
config + `bilateral_forward_depth_supervision_corrected=True` only. All 6
dataset fingerprints match A9 exactly. `structural_losses` confirms only
the corrected flag is set; historical flag and every other auxiliary loss
are `False`/`0.0`.

**Base coordinate telemetry contract held**: A15 vs A16's `coordinate`
epoch-telemetry trajectory (the historical A9 term, computed independent
of either flag) track each other closely every epoch (e.g. epoch 0
`0.00258` vs `0.00264`; epoch 9 `0.00106` vs `0.00099`) — unlike
historical A14, where the base term was mathematically altered, here it
is empirically confirmed unperturbed.

## 11. Historical A9 / compiled A9 / corrected candidate metric table

| Holdout | metric | A9 (eager) | A15 (compiled control) | **A16 (corrected candidate)** |
| --- | --- | ---: | ---: | ---: |
| 3DPW test | PA-MPJPE mm | 75.31 | 75.63 | **78.76** |
| 3DPW test | yaw MAE ° | 14.90 | 14.37 | 14.80 |
| 3DPW test | yaw P95 ° | 34.77 | 36.16 | 36.55 |
| 3DPW test | hinge flip | 2.36% | 2.90% | 2.76% |
| AMASS internal | PA-MPJPE mm | 69.19 | 69.65 | 69.82 |
| AMASS internal | yaw MAE ° | 8.77 | 9.24 | **8.02** |
| AMASS internal | yaw P95 ° | 22.37 | 24.37 | **21.81** |
| training MPJPE mm | — | 40.19 | 40.35 | 39.77 |
| runtime (10 epoch) | seconds | 1,335.6 | 919.4 | 910.6 |

vs the correct clean baseline (**A15**, same compiled backend): 3DPW
PA-MPJPE **+3.13 mm** worse (≈4.1% relative), yaw MAE/P95 roughly flat to
slightly worse; AMASS yaw improved (MAE -1.22°, P95 -2.56°), PA-MPJPE
flat. Training MPJPE improved slightly (-0.58 mm) — no A11-style
geometry collapse.

## 12. Forward-depth attribution

`scripts/attribute_bilateral_forward_depth_multi.py`, hard-set fixed from
`historical_a9`'s own evaluator ranking (top-5% cutoff `34.77°`, top-1%
cutoff `50.09°` — identical cutoffs to docs/18's A14 attribution, since
the hard-set is defined once from A9 and reused), 3DPW test holdout
(35,310 frames), four labeled checkpoints.

| Subset | metric | historical_a9 | **compiled_a9_control (A15)** | **corrected_candidate (A16)** | historical_a14 (reference only) |
| --- | --- | ---: | ---: | ---: | ---: |
| hard top-5% (1,766) | shoulder abs residual m | 0.2312 | 0.1649 | 0.1764 | 0.1814 |
| | hip abs residual m | 0.0907 | 0.0647 | 0.0682 | 0.0672 |
| | shoulder sign disagreement | 48.45% | 31.06% | 36.37% | 41.49% |
| | hip sign disagreement | 50.28% | 31.37% | 41.00% | 39.07% |
| hard top-1% (354) | shoulder abs residual m | 0.2949 | 0.1834 | 0.1923 | 0.2197 |
| | hip abs residual m | 0.1283 | 0.0837 | 0.0744 | 0.0887 |
| | shoulder sign disagreement | 68.75% | 33.82% | 40.07% | 67.65% |
| | hip sign disagreement | 61.58% | 44.35% | 44.92% | 59.04% |
| all eligible (35,310) | shoulder sign disagreement | 14.59% | 14.63% | 17.92% | 15.59% |
| | hip sign disagreement | 16.45% | 13.10% | 18.92% | 15.21% |
| non-hard (33,544) | shoulder sign disagreement | 12.96% | 13.84% | 17.03% | 14.34% |
| | hip sign disagreement | 14.67% | 12.13% | 17.76% | 13.96% |

**Critical finding**: `compiled_a9_control` alone (no relational
supervision at all) already shows meaningfully lower forward-depth
residual/sign disagreement than `historical_a9` on the hard sets — the
compiled execution backend is itself a confound for any comparison to
eager A9. The scientifically valid comparison is **corrected_candidate
vs compiled_a9_control** (identical backend): on every subset and every
metric except hard-top-1% hip abs residual, **the corrected candidate is
worse than the clean baseline**, not better — shoulder/hip sign
disagreement on `all_eligible` and `non_hard` both regress by roughly
3–5 percentage points, and the hard-set improvements the candidate does
show are smaller than what compiling alone already provided. The directly
supervised quantity does **not** improve under a clean control.

## 13. Epoch telemetry

512-window fixed train-domain subset (Section 10 summary numbers);
`bilateral_forward_depth_raw` shrinks smoothly over the 10 epochs
(`0.000155 → 0.000094`, no divergence), `diagnostic_shoulder_forward_
depth_sign_disagreement` stays low and roughly flat on this in-domain
subset (`0.0039 → 0.0000`, single-digit-percent range throughout) —
train-domain telemetry alone would have looked encouraging; it is the
official-test holdout attribution (Section 12) that reveals the candidate
does not generalize the intended improvement.

## 14. Matched qualitative review

Same four A9-fixed sequences as docs/10/18 (candidate not observed before
selection): `downtown_stairs_00:actor0` (worst-P95), `downtown_walking_
00:actor1` (longest-run), `downtown_bus_00:actor1` (P95-adjacent),
`downtown_bar_00:actor0` (control).

| Sequence | A15 (compiled control) yaw mean/P95 ° | **A16 (corrected)** yaw mean/P95 ° | A15→A16 worst-frame yaw ° |
| --- | --- | --- | --- |
| downtown_stairs_00:actor0 | 23.67 / 68.80 | **16.86 / 39.73** (improved) | 46.11 → 15.07 |
| downtown_walking_00:actor1 | 17.80 / 49.13 | **21.29** / 48.62 (mean worse) | 12.45 → **57.38** (much worse) |
| downtown_bus_00:actor1 | 10.56 / 27.33 | 10.01 / 27.49 (flat) | 75.37 → 84.21 (worse) |
| downtown_bar_00:actor0 (control) | 10.18 / 19.43 | **11.93 / 28.26** (worse) | 7.96 → 14.29 (worse) |

One sequence improves substantially (stairs); the other three, including
the previously-clean control sequence, get worse — the control
sequence's hip sign disagreement roughly triples (0.036 → 0.109). This
mirrors, and is even more pronounced than, the pattern historical A14
showed of introducing new confusion in previously-unproblematic regions.

## 15. Runtime / throughput achieved during main training

A15 (compiled control): 919.4s / 10 epoch, 5,041.8 samples/s.
A16 (corrected candidate): 910.6s / 10 epoch — essentially identical to
A15 (the added relational term is two extra scalar residuals per frame,
computationally negligible under the same compiled graph; no new
recompilation was expected or observed since the graph shape is
unaffected).

## 16. Portability assessment

- Depends only on the canonical joint contract and `+Y` axis: yes,
  unchanged from A14.
- Hard-example thresholds / source-specific weights: none, unchanged.
- Normalization derived from the base coordinate-loss contract: yes —
  more strictly than A14, since `D_coord` is now provably untouched
  (Sections 6-7).
- Vanishes as residual → 0: yes (synthetic contract 2).
- Consistent across sources: **no** — Section 11/12 show a 3DPW
  regression alongside an AMASS improvement, the Case E source-specific
  trade-off pattern.
- Could a future commercial dataset use it by only mapping into the
  canonical contract: mechanically yes, but this batch's own clean result
  gives no evidence it would help — quite the opposite on the evidence
  gathered here.

## 17. Architecture verdict

**Primary: CASE C — clean SRD does not improve its own target.** Relative
to the correct compiled-A9 control, the corrected candidate's own directly
supervised quantity (shoulder/hip forward-depth residual and sign
disagreement) is **worse**, not better, on `all_eligible`, `non_hard`,
and most hard-set cells (Section 12). This is a stronger negative result
than historical A14, which at least improved its own target on the hard
set even though yaw did not follow.

**Secondary: CASE E — source-specific trade-off.** AMASS yaw metrics
improve (MAE -1.22°, P95 -2.56°) while 3DPW test PA-MPJPE meaningfully
regresses (+3.13mm) and yaw stays flat-to-worse — not a shared portable
objective across sources.

Per instructions for both cases: **do not tune it.** No coefficient,
denominator, or masking adjustment is applied in response to this
result. Training MPJPE and AMASS PA-MPJPE remaining acceptable rule out
Case D (general-geometry damage) as the primary classification, though
the 3DPW PA-MPJPE regression is a real, non-trivial secondary cost worth
carrying forward as context.

**Conclusion**: the simple, denominator-clean, all-frame additive
bilateral forward-depth relational term is **insufficient** under the
current deterministic model/head — not merely "insufficient to move
yaw" (A14's finding) but insufficient to reliably improve the very
quantity it supervises, once execution-backend confounds are removed.
This closes the "was A14's negative result an artifact of the
denominator bug" question at the level of the *hypothesis*: **no, it was
not** — the corrected version is, if anything, a cleaner and stronger
negative result. **Wording correction (docs/22 Section 2)**: this does
**not** mean A16 proves which portion of historical A14's *exact* observed
behavior (its specific PA-MPJPE/yaw numbers) came from denominator
attenuation versus the underlying hypothesis failure — A14 and A16 differ
in two things at once (the denominator *and* the execution backend), so
their raw numbers are not directly decomposable against each other. What
is established is the clean, single-variable result: A16 vs A15, same
backend, only the corrected relational term differs, and it does not help.
The next architecture question (per the standing instruction, not pursued
this batch) should move to coupled torso geometry / relational local-frame
representation, or explicit temporal orientation-state supervision.

## 18. Historical documentation status table

| Item | Status |
| --- | --- |
| A9 | VALID |
| A10 | VALID |
| A11 training | VALID |
| A11 selector diagnostic | VALID WITH QUALIFICATION — conclusion (gradient-scale dominance, not selector structure) reinforced under the corrected exact-evaluator-angle selector (Section 2); original (1-cos) proxy numbers stand as historical evidence, now supplemented |
| A12 training | VALID |
| A12 magnitude/direction energy attribution | VALID (never used the buggy degree proxy) |
| A12 historical yaw Pearson values (`.325`/`-.056`/`.610`) | **SUPERSEDED** by Section 3's corrected values (`.496`/`.177`/`.517`); A13 NO-GO decision unaffected (never rested on these) |
| source-tail aggregation diagnosis (docs/16) | VALID (does not use the buggy `_yaw_association` surrogate) |
| 3DPW support diagnosis (docs/17) | VALID (used `_root_yaw_error_degrees`-based numbers directly, not the buggy surrogate) |
| historical A14 (docs/18) | VALID WITH QUALIFICATION — measured result valid for the objective actually executed; NOT a pure bilateral-forward-depth causal test (Section 1); causal question now answered cleanly by this batch (Case C/E, Section 17), which reinforces rather than reverses A14's practical rejection |

## 19. Exact files changed

- `src/training/temporal_lifter.py` — new
  `bilateral_forward_depth_supervision_corrected` flag (mutually exclusive
  with historical A14's flag), corrected `D_coord`-preserving reduction
  branch, `_structural_loss_report` entry
- `scripts/run_lifter_experiments.py` —
  `--bilateral-forward-depth-supervision-corrected`
- `scripts/repair_historical_selector_diagnostics.py` (신규) — Section 2/3
- `scripts/diagnose_corrected_srd_gradient_delta.py` (신규) — Section 7
- `scripts/attribute_bilateral_forward_depth_multi.py` (신규) — Section 12
- `tests/test_bilateral_forward_depth_corrected.py` (신규) — Section 6
- `tests/test_repair_historical_selector_diagnostics.py` (신규)
- `tests/test_diagnose_corrected_srd_gradient_delta.py` (신규)
- `tests/test_attribute_bilateral_forward_depth_multi.py` (신규)
- `docs/21_WORKLOG_CORRECTED_BILATERAL_FORWARD_DEPTH.md` (본 문서)
- `docs/10_TEMPORAL_LIFTER_IMPROVEMENT_ABLATION.md`
- `docs/README.md`

A9–A14 checkpoint/fingerprint/report, production defaults for every other
config field, promotion gates, and `.vscode/` were not touched.

## 20. Tests executed

- corrected-reduction synthetic contracts: `12 passed`
- selector/yaw-association repair diagnostic units: `4 passed`
- gradient-delta diagnostic units: `3 passed`
- multi-checkpoint attribution units: `4 passed`
- full local regression: `445 passed`
- `py_compile`: PASS
- LabServer63 GPU execution: reproduced eager baseline, kernel profiles,
  compile-candidate verdict (prior docs/20 batch); this batch's A11/A12
  repair diagnostic, gradient-delta diagnostic, A15 compiled-control
  training, A16 corrected-candidate training, multi-checkpoint
  attribution, and matched-qualitative replay all PASS with fingerprints
  verified

## 21. Commit hashes / synchronization state

- `391f26c` — fix: corrected bilateral forward-depth reduction (docs/21), preserving historical A14
- `48d6aa7` — diagnostic: repair A11 selector and A12 yaw-association diagnostics (docs/21)
- `ff75c5a` — diagnostic: actual Delta_G verification for corrected SRD candidate (docs/21)
- `0288ac0` — diagnostic: multi-checkpoint forward-depth attribution for docs/21 clean A/B
- 본 worklog 커밋은 이후 별도로 기록한다.

진단/학습 산출물은 git에 커밋하지 않고 다음 서버 경로에 남아 있다.

- `/home/nd/animcv-output/experiments/a21_historical_diagnostic_repair/` (repair.json, corrected_gradient_delta.json, test_attribution_multi.json, matched_qual.py)
- `/home/nd/animcv-output/experiments/ablation_a15_compiled_a9_control_10e/`
- `/home/nd/animcv-output/experiments/ablation_a16_bilateral_forward_depth_corrected_10e/`
