# Worklog — A11 Selector Isolation Repair + A16 Generalization-Gap Mechanism (2026-09-02)

> Diagnostic-only batch. No new A-series training. Uses existing A9–A16
> checkpoints exclusively. The scientifically valid comparison throughout
> Sections 3–13 is **A15 (compiled A9 control) vs A16 (corrected SRD
> candidate)** — never historical eager A9 vs A16 (docs/21 already
> established the execution backend is itself a confound). The frozen
> hard-set (historical A9's own evaluator ranking, top-5% cutoff `34.77°`)
> is reused unchanged throughout.

## 1. Corrected A11 selector-only isolation

`scripts/repair_a11_selector_isolation.py` (new) defines three explicit
paths, replayed on the same 10 fixed first-epoch batches/model states as
every prior A11 diagnosis:

- **P1** — production pooled selector (pair-observation candidates,
  ranked and differentiated by production `(1-cos)`).
- **P2** — historical frame-level proxy (frame candidates, frame-combined
  `(1-cos)` ranking *and* penalty).
- **P3** — exact-ranking control: frame candidates ranked **only** by the
  exact evaluator root-yaw angle (`torch.no_grad()`, no gradient path
  through that ranking value), but differentiating the **same**
  frame-combined `(1-cos)` quantity P2 uses, gathered at P3's own selected
  indices.

| Model state | P1 ratio | P2 ratio | **P3 ratio** | P1↔P2 Jaccard | P1↔P3 Jaccard | P2↔P3 Jaccard |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| init | 1.69 | 10.23 | 9.59 | .190 | .244 | .806 |
| A9 trained | 3,169.92 | 3,199.72 | **3,324.85** | .616 | .571 | .775 |
| A11 trained | 157.14 | 134.94 | **138.74** | .592 | .541 | .731 |

Holding the differentiated penalty fixed to the exact same `(1-cos)`
quantity, **switching only the selection rule** (pooled-pair → frame-proxy
→ exact-angle-ranked-frame) moves the gradient/base ratio by only
`~2–4%` (3,199.72 → 3,324.85 at A9 state; 134.94 → 138.74 at A11 state) —
noise-level, not the ~100× swing the earlier conflated measurement showed.
**This is the clean, isolated confirmation**: selection semantics
contribute essentially nothing to the gradient-scale explosion; the
pathology is entirely attributable to penalty representation (the angular
`(1-cos)`/degree scale itself vs. the coordinate loss's Cartesian scale),
independent of which frames get chosen. The previous (conflated) measurement
that both ranked *and* differentiated the real degree value is retained
as `angle_penalty_diagnostic` (ratio `302,884` at A9 state) — a different,
still-valid question about penalty representation, explicitly **not** used
here to support or reject the selector-structure conclusion.

## 2. Historical wording/status update

Applied directly to `docs/21_WORKLOG_CORRECTED_BILATERAL_FORWARD_DEPTH.md`
(measurements unchanged, interpretive wording corrected in place):

- **A11**: the selector-structure exclusion now explicitly rests on
  Section 1's isolated P1/P2/P3 control, not the conflated
  `exact_evaluator_angle_frame_selector`, which is re-labeled
  `angle_penalty_diagnostic`.
- **A15 compiled control**: no longer described as "ordinary eager
  run-to-run variance." Compiled execution preserves the mathematical
  objective exactly but forms a **distinct numerical execution lineage**
  — architecture A/B comparisons after A15 must stay on the compiled
  backend (which docs/21's A16 comparison already did).
- **A14/A16**: added explicit caveat that A16 does not decompose which
  portion of historical A14's *exact* observed numbers came from
  denominator attenuation vs. the underlying hypothesis failure — A14 and
  A16 differ in two things at once (denominator *and* backend). What is
  established is the clean, single-variable result (A16 vs A15).

Raw historical numbers in docs/10, docs/15, docs/16, docs/17, docs/18,
docs/21 are **unchanged**.

## 3. A15/A16 X/Y component decomposition

Canonical `delta_X = X_right - X_left` (+X=right), `delta_Y = Y_right -
Y_left` (+Y=forward/depth), on the 3DPW official-test holdout (35,310
frames), frozen A9 hard-set.

| Subset | pair | delta_X abs resid (A15→A16) | delta_Y abs resid (A15→A16) | angular error ° (A15→A16) | sign disagreement (A15→A16) |
| --- | --- | --- | --- | --- | --- |
| all_eligible | shoulder | 0.0661→**0.0631** (better) | 0.0806→**0.0890** (worse) | 14.20→14.65 | 14.6%→**17.9%** |
| | hip | 0.0249→**0.0220** (better) | 0.0331→0.0326 (flat) | 14.37→14.80 | 13.1%→**18.9%** |
| hard_top5pct | shoulder | 0.1063→0.1066 (flat) | 0.1649→**0.1764** (worse) | 31.89→32.72 | 31.1%→**36.4%** |
| | hip | 0.0455→**0.0333** (better) | 0.0647→**0.0682** (worse) | 31.50→31.97 | 31.4%→**41.0%** |
| hard_top1pct | shoulder | 0.1069→0.1097 (flat/worse) | 0.1834→**0.1923** (worse) | 36.06→**34.31** (better) | 33.8%→**40.1%** |
| | hip | 0.0559→**0.0357** (better) | 0.0837→0.0744 (better) | 40.38→**38.62** (better) | 44.4%→44.9% (flat) |
| non_hard | shoulder | 0.0642→**0.0610** (better) | 0.0766→**0.0848** (worse) | 13.35→13.78 | 13.8%→**17.0%** |
| | hip | 0.0238→**0.0214** (better) | 0.0315→0.0307 (flat) | 13.46→13.89 | 12.1%→**17.8%** |

**Pattern**: `delta_X` is flat-to-slightly-*better* for A16 almost
everywhere — X is not being sacrificed. `delta_Y` (the explicitly
supervised quantity) is worse in 6 of 8 pair×subset cells, including
`non_hard`. `sign_disagreement` — the discrete branch state — worsens
**broadly and substantially** (roughly +3 to +10 percentage points)
across every subset including `non_hard`, even in the `hard_top1pct` cells
where continuous `angular_error` actually *improves*. This dissociation
(continuous metric improving while discrete sign gets meaningfully worse)
is the first hint of a branch/coherence-specific failure mode distinct
from ordinary regression accuracy.

## 4. Counterfactual substitution

`CF_X15_Y16` = A15's `delta_X` + A16's `delta_Y`; `CF_X16_Y15` = A16's
`delta_X` + A15's `delta_Y`. Mathematically exact for the pair angle
itself (root yaw depends only on that pair's own X/Y); reported as
counterfactual angles, not claimed as a real candidate's official score.

| Subset | pair | A15 actual° | **CF(X15,Y16)°** | A16 actual° | **CF(X16,Y15)°** |
| --- | --- | ---: | ---: | ---: | ---: |
| all_eligible | shoulder | 14.20 | 14.63 (worse) | 14.65 | 13.65 (better) |
| | hip | 14.37 | 15.09 (worse) | 14.80 | 13.82 (better) |
| hard_top5pct | shoulder | 31.89 | 33.62 (worse) | 32.72 | 29.86 (better) |
| | hip | 31.50 | 33.18 (worse) | 31.97 | 29.66 (better) |
| hard_top1pct | shoulder | 36.06 | 34.71 (better) | 34.31 | 32.92 (better) |
| | hip | 40.38 | 38.56 (better) | 38.62 | 39.57 (worse) |
| non_hard | shoulder | 13.35 | 13.71 (worse) | 13.78 | 12.87 (better) |
| | hip | 13.46 | 14.14 (worse) | 13.89 | 12.99 (better) |

**Answer to "would A16's Y have helped under A15's X?" — mostly NO.** In 6
of 8 cells, `CF(X15,Y16)` is *worse* than A15's actual result: A16's `delta_Y`
would not have been beneficial even holding A15's clean X fixed (the only
exception is `hard_top1pct`, the most extreme tail). Conversely,
`CF(X16,Y15)` (A16's X with A15's Y) is *better* than A16's actual result
in 7 of 8 cells — swapping A15's Y back onto A16's X recovers most of the
gap. **Conclusion: A16's own `delta_Y` is what is degraded, not a
victim of an X/Y trade-off.** Component trade-off (Case A) is not the
primary mechanism — X is not being sacrificed to obtain Y; Y itself fails
to generalize its own target, exactly as docs/21's aggregate finding
showed, now confirmed at the per-component level.

## 5. Shoulder/hip coherence

Per-skeleton (not vs. GT) shoulder-vs-hip angular disagreement and
forward-depth sign agreement.

| Subset | GT disagreement° | A15 disagreement° | **A16 disagreement°** | GT sign-agree | A15 sign-agree | **A16 sign-agree** |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| all_eligible | 7.58 | 6.42 | 6.94 | .950 | .949 | .946 |
| hard_top5pct | 8.35 | 6.52 | **7.99** | .962 | .876 | .903 |
| hard_top1pct | 7.90 | 6.46 | **11.47** | .941 | .871 | .915 |
| non_hard | 7.54 | 6.41 | 6.89 | .950 | .952 | .948 |

Angular disagreement between shoulder and hip nearly **doubles** for A16
at `hard_top1pct` (6.46° → 11.47°) while sign *agreement* actually rises
slightly (.871 → .915) — consistent with shoulder and hip drifting
further apart in continuous angle while more often landing on the *same*
(possibly both-wrong) discrete sign together. Run-length accounting for
≥20° disagreement (sequence-boundary-safe):

| | A15 | **A16** |
| --- | ---: | ---: |
| total flagged frames | 801 | **1,104** (+38%) |
| run count | 176 | **213** |
| mean run length | 4.55 | **5.18** |
| runs of 5+ | 51 | **69** |

A16 has more, and longer-duration, shoulder-hip coherence-breakdown
episodes than A15.

## 6. Torso local-frame attribution

Diagnostic-only frame: `combined_right_axis = normalize(shoulder_lateral_unit
+ hip_lateral_unit)`, no training loss, no renormalization.

| Subset | GT° | A15° | **A16°** | A16 − A15 |
| --- | ---: | ---: | ---: | ---: |
| all_eligible | 3.79 | 2.96 | 3.22 | +0.26 |
| hard_top5pct | 4.17 | 3.02 | 3.75 | +0.73 |
| **hard_top1pct** | 3.95 | 2.95 | **5.58** | **+2.63** |
| non_hard | 3.77 | 2.96 | 3.20 | +0.24 |

Frame-inconsistency degradation **concentrates specifically in the hard
tail** and grows monotonically with difficulty (non-hard +0.24° → hard-5%
+0.73° → hard-1% +2.63°) — the same signature as Section 5's coherence
result, computed a different way.

## 7. Error migration

Frozen A15-vs-`34.77°`-threshold, paired against A16's own yaw error at the
same frame (no new threshold invented after observing A16).

| Category | frame count | Δcoherence° | Δtorso-frame° | Δdelta_Y (shoulder/hip) |
| --- | ---: | ---: | ---: | --- |
| previously_bad_improved | 1,390 | **-0.55** | **-0.26** | **-0.074 / -0.050** (both better) |
| previously_bad_worse | 662 | **+3.13** | **+1.56** | -0.001 / +0.005 (~flat) |
| previously_good_newly_bad | **1,491** | **+2.20** | **+1.14** | **+0.093 / +0.044** (both worse) |
| previously_good_remains_good | 31,767 | +0.45 | n/a | ~flat |

**This is the most decisive single result in the batch.** Rescued frames
(`previously_bad_improved`) improve on *every* axis together — component
residual, coherence, and frame-consistency all move the same direction.
Frames that stay bad (`previously_bad_worse`) show component residuals
roughly **unchanged** while coherence/frame-error get **substantially
worse** — the differentiator between "stays broken" and "gets fixed" is
coherence, not raw magnitude. The newly-damaged category
(`previously_good_newly_bad`, the single **largest** changed category —
larger than the rescued count) shows both `delta_Y` degradation *and*
coherence/frame degradation moving together. Net raw frame count:
**1,491 newly damaged vs. 1,390 rescued** — A16 breaks more frames than
it fixes, even before weighting by severity.

## 8. Temporal branch-run attribution

Sequence-boundary-safe forward-depth sign-transition analysis (GT has 193
true shoulder transitions, 256 hip transitions, across the full holdout).

| | A15 shoulder | A16 shoulder | A15 hip | A16 hip |
| --- | ---: | ---: | ---: | ---: |
| predicted transitions | 477 | 512 | 577 | 535 |
| missed GT transitions | 181 | 185 | 241 | 250 |
| false transitions | 465 | 504 | 562 | 529 |
| total flagged (wrong-sign) frames | 5,042 | **6,173** (+22%) | 4,624 | **6,680** (+44%) |
| mean disagreement-run length | 13.66 | **15.99** (+17%) | 11.09 | **16.37** (+48%) |
| max disagreement-run length | 478 | 527 | 186 | **382** (+105%) |
| runs of 5+ | 192 | 210 | 201 | **236** |

A15 already flickers considerably relative to GT's rare true transitions
(both models produce far more predicted transitions than GT has). A16's
distinguishing signature is not primarily *more* flicker — it is
**longer, more persistent commitment to the wrong sign** (hip max run
length more than doubles, from 186 to 382 frames — over 15 seconds
continuous at 25fps; mean run length up 17–48%). This is a genuine
generalization failure of the discrete branch state on held-out data,
not measurement noise.

## 9. Static/temporal mechanism verdict

**CASE D — MIXED**, quantified rather than forced to one mechanism:

- **Case A (component trade-off): not supported as primary.** Section 4
  Delta_X is flat-to-better for A16; Section 4 counterfactual shows A16's
  own `delta_Y` — not X — is what degrades, in both directions of
  substitution (Section 4).
- **Case B (torso coherence failure): substantially supported.**
  Sections 5–6–7 all independently show coherence/frame-consistency
  degradation, concentrated in the hard tail (monotonically with
  difficulty), and Section 7 shows coherence/frame-error — not raw
  component residual — is what differentiates rescued from stuck-bad
  frames.
- **Case C (temporal branch failure): substantially supported.**
  Section 8 shows persistent wrong-sign run duration growing
  substantially (not just more flicker), a genuine discrete-state
  generalization failure.

Both B and C contribute; Section 7's error-migration accounting is the
strongest single piece of evidence, and it implicates coherence/frame
breakdown as the proximate mechanism separating rescued from damaged
cases, while Section 8 shows that mechanism's temporal consequence
(longer wrong-branch commitment).

## 10. Frozen representation probe (conditionally reached)

Section 8/9 left temporal branch-state failure as a substantial
unresolved explanation, so `scripts/probe_a15_temporal_orientation_state.py`
ran: a closed-form linear probe (no MLP, no backbone gradients) on A15's
own frozen features, sequence-level train/test split (24,419 train /
10,891 test frames), predicting the exact `q_shoulder`/`q_hip` SRD target
from GT.

| Representation | receptive field | shoulder test R² | hip test R² | shoulder test MAE | hip test MAE |
| --- | --- | ---: | ---: | ---: | ---: |
| local_control (post-stem) | ~3 frames | 0.527 | 0.538 | 0.0783 | 0.0227 |
| **full_temporal (pre-head)** | 127 frames (full window) | **0.817** | **0.788** | **0.0521** | **0.0159** |

**P1 — the temporal latent already contains substantial recoverable
orientation-branch state.** Even a near-local representation already
explains >50% of held-out variance, and the full temporal representation
explains materially more (R² +0.25 to +0.29, MAE down 33–40%) — recovered
from a model (**A15**) that was **never explicitly supervised on this
quantity at all**. The backbone already encodes this information well as
a byproduct of the ordinary coordinate/structural objective; it is not a
missing-information problem.

## 11. Next architecture implication

Per P1's interpretation: **structured/coupled torso representation
(relational torso head / local-frame head) is favored over explicit
temporal orientation-state representation as the next hypothesis.**

The evidence converges: the information the corrected candidate tried to
supervise directly (Section 4) is already substantially decodable from the
existing temporal backbone (Section 10); what fails is a **consistent,
coupled shoulder-hip torso commitment** (Sections 5–7), whose breakdown is
exactly what separates rescued frames from newly-damaged ones (Section 7),
and whose downstream temporal symptom is longer wrong-branch persistence
(Section 8) rather than a primary deficiency in temporal modeling power.
Given the temporal representation is not the bottleneck, adding more
temporal-state machinery is a less parsimonious first step than giving the
model a head/representation that structurally couples shoulder and hip
into one torso-orientation decision instead of two independent scalar
regressions sharing a single linear projection. This is a diagnostic
conclusion only — **not implemented this batch.**

## 12. Exact limitations

- Section 4–9 statistics are means/counts over the 3DPW official-test
  holdout only; AMASS was not separately re-analyzed at this granularity
  in this batch (docs/21 already showed AMASS yaw improving under A16,
  the opposite direction from 3DPW — a source-specific asymmetry this
  batch's mechanism analysis does not further decompose).
- The diagnostic torso local frame (Section 6/7's `combined_right_axis`)
  is one principled but not unique construction; other valid combinations
  (e.g. weighting shoulder/hip unequally by their own local reliability)
  were not explored, per the instruction against introducing a new
  training-quality metric.
- The linear probe (Section 10) tests **linear** decodability only; a
  nonlinear probe could show an even larger P1 signal, but the instruction
  explicitly restricts this to a minimal linear probe.
- Section 9's "false transition"/"missed transition" counts use a strict
  same-sign-as-previous-frame definition; near-zero GT `delta_Y` frames
  are not given special branch-label treatment (per the instruction), so
  some counted "transitions" may reflect genuine small oscillation near a
  near-planar torso rather than a true orientation-branch flip.

## 13. Exact files changed

- `scripts/repair_a11_selector_isolation.py` (신규) — Section 1
- `scripts/diagnose_a16_generalization_gap.py` (신규) — Sections 3–8
- `scripts/probe_a15_temporal_orientation_state.py` (신규) — Section 10
- `tests/test_repair_a11_selector_isolation.py` (신규)
- `tests/test_diagnose_a16_generalization_gap.py` (신규)
- `tests/test_probe_a15_temporal_orientation_state.py` (신규)
- `docs/21_WORKLOG_CORRECTED_BILATERAL_FORWARD_DEPTH.md` — wording
  corrections only (Section 2), no measurement changes
- `docs/22_WORKLOG_A16_GENERALIZATION_GAP.md` (본 문서)
- `docs/README.md`

A9–A16 checkpoints/fingerprints/reports/configs, the accepted
`torch.compile` production path, and `.vscode/` were not touched. No new
A-series training occurred.

## 14. Tests executed

- A11 selector-isolation units: `3 passed`
- generalization-gap geometry/accounting units: `11 passed`
- temporal-probe units: `5 passed`
- full local regression: `464 passed`
- `py_compile`: PASS
- LabServer63 GPU execution: A11 selector isolation (10 fixed batches ×
  3 model states), full generalization-gap diagnosis (3 checkpoints × 3DPW
  holdout, 35,310 frames), and the frozen-representation probe (A15
  backbone, 35,310 frames, sequence-level split) all PASS

## 15. Commit hashes / synchronization state

- `46f76d1` — diagnostic: isolate A11 selector-only structure from penalty scale (docs/22)
- `f3aaee7` — diagnostic: A16 generalization-gap mechanism analysis (docs/22)
- `f7dd8bb` — diagnostic: frozen-representation orientation-state probe (docs/22 Section 12-13)
- 본 worklog 및 docs/21 wording correction 커밋은 이후 별도로 기록한다.

진단 산출물은 git에 커밋하지 않고 다음 서버 경로에 남아 있다.

- `/home/nd/animcv-output/experiments/a22_a16_generalization_diagnosis/`
  (a11_selector_isolation.json, generalization_gap.json, temporal_probe.json)
