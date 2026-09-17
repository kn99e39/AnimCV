# Temporal Context Evidence: frozen-H0 diagnostic

Date: 2026-09-17  
Scope: evidence and attribution only; no temporal production system

## Decision

**B — temporal context is partially justified.** Short same-sequence context
has strong, real-data utility for the narrow case of H0 jitter while the 2D
observation remains stable, and for many high-error ankle rows. It is not
evidence for a general smoother: the same parameter-free control worsens the
high-2D-instability cohort on average, and two-thirds of current observation
loss rows have no usable two-sided local support. Any future temporal work
therefore needs explicit observation/support gating and a separately evaluated
proposal; it must not be silently added to the frozen frame-first path.

No production prediction, H0, FrameBank, SignState, geometry, Pose
Reconciliation, training, tuning, Kalman/filtering implementation, or visual
evaluation content was changed.

## Frozen source and boundaries

The diagnostic loaded the exact docs/44--46 baseline once:

| Item | Identity |
|---|---|
| Test rows / sequences | 7,076 / 37 |
| FrameBank content digest | `75519e6394a764e3749ddaa30555b58b73a01db582ccee14f661374b9a0ed536` |
| Frozen H0 prediction SHA-256 | `6df04c2031b2bde6a4f0af37fdae0a7f0228e9fc23401e15e1aa2b3bd35695a5` |
| Frozen evaluation SHA-256 | `a5881850b3769ca2329271d327ee895dc646f236451f05c3ba2e0ff917f580a3` |

Rows are sorted deterministically within a sequence by timestamp/frame. Only
retained same-sequence t-1, t+1, t-2, and t+2 candidates are visible; no
cross-sequence borrowing or extrapolation is permitted.

The sole control is deliberately non-production: for a *failure* joint row
with an in-frame observation and finite H0 on both sides, interpolate that
joint's frozen H0 location linearly by timestamp. It has no learned parameter,
does not alter the H0 array, and leaves the strict stable control entirely
unprobed. Neighbour oracle error is reported only for attribution, never used
to select a neighbour.

The final accepted artifact is server
`/home/nd/animcv-output/framepose/temporal_context_evidence_20260917/report_v4/`:

| File | SHA-256 |
|---|---|
| `temporal_context_report.json` | `2d80dbf16e54ac42acc7e884ca22fe59a888c8680f523eea2d04d5ef843fa094` |
| `temporal_evidence_rows.csv` | `23a9d335b7474f9ef95c9254cd1c09ceb1cdfef0d5a441c0ebcd11063e017557` |

Local copies are intentionally ignored as regenerable analysis data at
`output/temporal_context_evidence_20260917/`. This worklog is the committed
Markdown record; no JSON worklog was created.

## Predeclared cohorts and accounting

The five failure cohorts are structural or fixed per-joint test-quantile
definitions made before looking at recovery results: current observation loss,
observation degradation (2D local residual >= P90), H0 jitter with stable 2D
(2D <= P50 and H0 residual >= P90), high-error left/right ankle, and a known
target/H0 hinge-sign mismatch at the valid middle joint. The latter is only an
attribution cohort: Pose Reconciliation is not rerun or reconsidered.

The failure union contains **21,129 joint rows**. It is classified once,
without double counting: T1 7,254; T2 7,425; T3 0; T4 4,258; T5 2,192.
T1 means this control strictly improved a target-available row, not that a
future temporal architecture is justified for the whole cohort.

| Cohort | Joint rows | Two-sided probe support | Mean H0 → control error (mm) | Mean delta (mm, + better) | Improved / worsened |
|---|---:|---:|---:|---:|---:|
| Current observation loss | 6,380 | 33.26% | 79.86 → 52.38 (only 29 recoverable target rows) | +1.58 over 111 target-available rows | 13.51% / 12.61% |
| Observation degradation | 10,878 | 100.00% | 84.00 → 93.39 | **-9.39** | 46.48% / 53.52% |
| H0 jitter, stable observation | 1,518 | 100.00% | 80.40 → 72.12 | **+8.28** | 57.64% / 42.36% |
| Distal ankle failure | 1,303 | 96.39% | 340.47 → 318.45 | **+20.63** | 60.17% / 36.22% |
| Implausible articulation | 1,390 | 95.97% | 101.49 → 99.32 | +2.02 | 48.85% / 47.12% |
| Stable control | 20,571 | intentionally unprobed | 40.20 → unchanged | 0.00 | 0.00% / 0.00% |

The observation-loss error figures have a crucial availability caveat: 6,380
rows were selected because the current 2D joint was invalid/out of frame, but
only 111 also have a target-relative H0 error and 29 have a two-sided
target-measurable recovery. They cannot support a broad recovery claim.

## What adjacent real-scene evidence actually contains

For each row, the report additionally records whether valid 2D exists on both
sides within t+/-2, whether an in-window neighbour has higher detector
confidence, whether an in-window neighbour has lower oracle H0 error, and the
length/smoothness of contiguous invalid-observation gaps.

| Cohort | Higher-confidence neighbour | Lower-oracle-error neighbour | Additional interpretation |
|---|---:|---:|---|
| Current observation loss | 65.80% | 0.88% | Invalid runs average 15.97 frames (median 5; P95 48); only 33.26% reappear on both sides within t+/-2. Oracle gap residual is measurable for 111 rows (median 55.24 mm), so long occlusions are not safely linear. |
| Observation degradation | 83.47% | 82.18% | Neighbours are often individually better, yet H0 interpolation still loses 9.39 mm on average: neighbour availability alone is not a selection rule. |
| Stable-2D H0 jitter | 80.30% | 81.49% | Complete local support and a +8.28 mm mean reduction are the clearest narrow temporal signal. |
| Distal ankle | 82.50% | 95.86% | Strong local evidence for some high-error ankle rows, but not for contact, foot locking, root motion, or a foot-specific production policy. |
| Articulation mismatch | 84.46% | 83.38% | Near-even win/loss split; temporal interpolation does not answer the articulation/sign question. |

## Owner-annotation seeds

The CP949 owner review sheet was decoded locally and matched to the historical
review manifest. It was not copied to the server. These are attribution
anchors, not aggregate evidence.

| Review seed | Target observation | Diagnostic result |
|---|---|---|
| `R0aef948071ab`, arguing #550, left elbow | Owner marked the visual result generally good. | Articulation cohort, T1: 106.86 → 94.91 mm. The positive owner judgment prevents treating this isolated numeric event as a dominant visual failure. |
| `R3cb7f471ee2d`, sitOnStairs #45, left knee | Jitter/readability and camera-view concern. | Stable control, unprobed: 46.91 mm unchanged. The selected joint does not substantiate a temporal failure. |
| `R0a504e3f503f`, bar #985, right elbow | Sudden tracking drop/arm fall. | Articulation cohort, T5: 120.07 mm unchanged; no valid after-side support in t+/-2. |
| `R7c37cd172248`, warmWelcome #330, right knee | Articulation appears wrong. | T2: 50.73 → 85.70 mm (worse). No reconciliation change follows. |
| `R5f7f9d0ce851`, car #507, right knee | Subject visibility/collapse concern. | T1: 64.00 → 29.15 mm. The selected knee remains in frame; this must not be mislabeled as current-joint observation loss. |
| `R0f50b98d649f`, windowShopping #588, ankles | Foot/ankle could not be followed to the end. | Both ankle checks are observation-degradation/T2: left 148.35 → 198.67 mm; right 36.26 → 207.42 mm. This is explicit counter-evidence to simple interpolation for that seed. |

## Compact owner review package

The corrected, labelled package is at
`user_QE/temporal_context_evidence_20260917/` (ignored from Git by design):
nine MP4s plus `review_manifest.json` SHA-256
`1583cd3e1af63d2302bfb1bf5f7339424c29cf7b09ec0880cdac347de086ba46`.
It includes all six owner anchors and deterministic T1, T2, and T4 examples.
Each frame shows original RGB with the observed target joint, its retained
t+/-2 2D trajectory, frozen H0, the display-only probe, oracle target, and
attribution state. The rendered frames were decoded locally; the broken `±`
glyph was corrected to ASCII `+/-` before this final package replaced the
first export.

## Reproducibility and checks

`scripts/diagnose_temporal_context_evidence.py` is diagnostic-only and writes
the CSV/report. `scripts/export_temporal_context_review.py` produces the
owner-facing videos. Tests cover sequence-local/timestamp-aware neighbours,
no extrapolation on missing sides, row classes, stable-control precedence, and
neighbour/gap accounting.

```text
PYTHONPATH=.:scripts:src pytest -q tests/test_temporal_context_evidence.py
6 passed

PYTHONPYCACHEPREFIX=/private/tmp/animcv-pycache python3 -m py_compile \
  scripts/diagnose_temporal_context_evidence.py scripts/export_temporal_context_review.py
ruff check --ignore E402 scripts/diagnose_temporal_context_evidence.py \
  scripts/export_temporal_context_review.py tests/test_temporal_context_evidence.py
All checks passed
```

Server reports v1/v2 are retained only as run history and are not accepted
evidence: their stable-control construction overlapped failure rows. The v3
rerun repaired that control; accepted v4 preserves that repair and adds the
neighbour/gap accounting above.
