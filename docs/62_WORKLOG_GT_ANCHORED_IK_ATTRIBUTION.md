# Worklog — GT-Anchored Attribution of FramePose Two-Bone IK (2026-10-01)

## Starting state

Before editing I fetched `origin/arch/single_frame_first`. Local and remote HEAD were both `f3b37d6` (the final Worklog 61 commit) and the worktree was clean. The implementation is in commit `c43a6ce`, and LabServer63 was fast-forwarded to it before the run.

This batch is evaluation only. No production module changed:
- `framepose_two_bone_ik.py`
- `framepose_ik_chains.json`
- Worklog 59 FK and Worklog 60 object-owned Root Orientation
- the Blender rest snapshot, rig calibration and BaseRig
- AnimationSemantics v2 and FramePose

The Worklog 61 runtime IK remains the H0 baseline exactly.

## Labels and leakage control

- **H0 / runtime observable:**
  - docs/57 AnimationSemantics v2 sidecars
  - Worklog 60 object-owned FK files
  - Worklog 61 IK files
- **GT / evaluation oracle:** 3DPW `jointPositions`, loaded with the existing `load_3dpw_ground_truth` path (`_matched_oracle_sequence`). This path is in the same canonical camera-root-relative frame as H0 and uses the same frame rows.
- **Where GT is used:** only inside `scripts/run_framepose_ik_gt_attribution.py`. It is never written into semantics, FK, IK or reliability.
- **Evaluation module:** `src/retarget/framepose_ik_gt_attribution.py` is labelled `EVALUATION ONLY`. It contains geometry only and no dataset access.
- **Leakage test:** a test asserts that no production FK, IK, object-owner, rest or AnimationSemantics module imports it, or imports any 3DPW/oracle loader.

**Shared transform.** H0 and GT chains both pass through the same runtime transform `A·Rz(−ψ_H0)`. Root Orientation error is therefore not under test, and GT yaw is never used.

**Matched rows.** All 395 Worklog 61 IK-known left-arm rows had GT valid for all three joints:
- dancing actor0: 79 rows
- crosscountry actor0: 133 rows
- hug actor1: 183 rows

The 69 excluded rows are exactly the rows that are UNAVAILABLE in Worklog 61 (root, mid or end invalid), so nothing was dropped for GT reasons. The persisted Worklog 61 IK endpoints and Worklog 60 FK endpoints were reproduced to ≤ 4.6e-14 cm.

## GT source geometry stability (question 1)

| sequence | GT upper / lower (m) | GT ratio | GT CV | H0 ratio on same rows: CV, p05–p50–p95, min–max |
|---|---|---:|---:|---|
| dancing actor0 | 0.2497 / 0.2350 | 1.063 | 0.000 | 0.178, 0.80–1.05–1.31, 0.35–1.63 |
| crosscountry actor0 | 0.2667 / 0.2498 | 1.068 | 0.000 | 0.220, 0.72–1.03–1.45, 0.49–2.11 |
| hug actor1 | 0.2569 / 0.2662 | 0.965 | 0.000 | 0.303, 0.77–0.95–1.15, 0.58–4.52 |

H0 segment-length CVs:

| sequence | upper-arm CV | forearm CV |
|---|---:|---:|
| dancing actor0 | 0.10 | 0.18 |
| crosscountry actor0 | 0.13 | 0.15 |
| hug actor1 | 0.10 | 0.11 |

GT arm lengths are exactly constant within each actor (SMPL joints come from one fixed body shape per sequence). The H0 medians are close to GT, and the medians of all three are close to BaseRig's 1.019.

**Answer:** the large framewise ratio variation in Worklog 61 is FramePose estimation error, not real human geometry.

## H0-vs-GT geometry errors (pooled, 395 rows; each measured separately)

| quantity | p05 | p50 | p95 | max / note |
|---|---:|---:|---:|---|
| endpoint direction angle (H0 vs GT root→end) | 3.2° | 10.3° | 38.3° | 133.2°; mean 14.2° |
| reach fraction error, H0 − GT | −0.122 | −0.003 | +0.101 | \|error\| p50 0.016, p95 0.153, max 0.458 |
| ratio error \|log(H0/GT)\| | — | 0.081 | 0.391 | 1.543 |
| bend angle error, H0 − GT | −23.9° | +2.0° | +27.4° | \|error\| p50 5.5°, p95 34.7° |
| bend-plane angle (H0 vs GT perpendicular elbow direction) | — | 14.2° | 66.4° | 163° |

**Bend side:** 385 rows agree and 10 flip (2.5%). By sequence, the flips are dancing 0/79, crosscountry 8/133 and hug 2/183. Every observed side was `observed`: no degenerate-bend rows in either H0 or GT.

Per-sequence medians:

| sequence | direction error | \|reach error\| | bend-plane error |
|---|---:|---:|---:|
| dancing | 7.3° | 0.008 | 10.8° |
| crosscountry | 17.3° | 0.028 | 22.9° |
| hug | 10.3° | 0.015 | 11.2° |

## GT-anchored endpoint

`E_GT = target_root + GT_reach_fraction · (L1+L2) · GT_endpoint_direction`. It uses the same BaseRig `upperarm_l` rest head and the same Blender rest L1 = 27.771 and L2 = 27.251, giving L1+L2 = 55.022 armature units (cm at object scale 0.01). This is the GT version of the Worklog 61 dimensionless contract, not a GT metric wrist position.

Every variant is scored against this same `E_GT`. Endpoint and elbow are reported separately, and the side choice does not move the endpoint. The two direction-FK variants have no clamp. The four IK variants (B–E) used `none` clamps on all 395 rows.

## FK vs current IK vs hybrids vs oracle

Endpoint error against `E_GT`, in cm (mean / p50 / p95 / max):

| variant | pooled | dancing | crosscountry | hug |
|---|---|---|---|---|
| A. Worklog 60 direction FK (H0) | 12.51 / 9.97 / 29.89 / 72.53 | 9.59 / 7.06 / 21.35 | 17.06 / 15.66 / 37.99 | 10.47 / 9.38 / 22.71 |
| B. Worklog 61 IK (H0 direction + H0 reach) | 12.30 / 9.38 / 31.09 / 73.82 | 8.94 / 6.74 / 20.71 | 17.60 / 16.08 / 43.13 | 9.90 / 8.78 / 20.52 |
| C. H0 direction + **GT reach** | 11.93 / 8.91 / 31.90 / 71.34 | 8.08 / 6.14 / 17.76 | 17.38 / 15.32 / 41.65 | 9.62 / 8.68 / 19.60 |
| D. **GT direction** + H0 reach | **2.15 / 0.89 / 8.42 / 25.19** | 2.15 / 0.46 / 9.12 | 2.86 / 1.54 / 9.22 | 1.64 / 0.82 / 5.85 |
| E. oracle IK (GT direction + GT reach) | 0 / 0 / 0 / 0 | 0 | 0 | 0 |
| control: direction FK on **GT** directions | 0.50 / 0.43 / 0.94 / 1.36 | 0.42 / 0.23 / 0.85 | 0.38 / 0.38 / 0.72 | 0.61 / 0.69 / 1.11 |

**Elbow error** against the oracle elbow (GT endpoint with GT side), median in cm:

| variant | median elbow error (cm) |
|---|---:|
| A. FK | 5.31 |
| B. H0 IK | 5.03 |
| C. H0 direction + GT reach | 4.78 |
| D. GT direction + H0 reach | 1.33 |
| control: FK on GT directions | 0.23 |

**Current IK vs FK against GT:**
- Pooled, IK is better than FK on 245 of 395 rows, but the mean improvement is only 0.21 cm on an error of about 12.4 cm (paired difference p05 −1.98, p50 −0.10, p95 +1.86).
- By sequence the paired difference varies: dancing −0.64, hug −0.57, crosscountry **+0.54** (IK worse; better on only 53 of 133 rows).
- On the largest Worklog 61 rotation-delta row (crosscountry #173), IK error rises from 38.4 to 50.9 cm.

**FK-vs-IK disagreement:**

| disagreement | mean (cm) | p95 (cm) | max (cm) |
|---|---:|---:|---:|
| true proportion adaptation (FK vs IK on GT geometry) | 0.50 | 0.94 | 1.36 |
| H0 (FK vs IK on H0 geometry, as in Worklog 61) | 1.72 | 5.90 | 20.95 |

The two are correlated at only r = 0.50.

**Correlation:**
- Endpoint direction angle error dominates every H0-direction variant:
  - FK: Pearson 0.934, Spearman 0.957
  - H0 IK: Pearson 0.965, Spearman 0.977
  - C: Pearson 0.972
- Ratio error (0.21–0.38) and reach error (0.35–0.42) correlate weakly with them.
- D tracks \|reach error\| exactly (r = 1.0, as expected by construction).

These correlations are only supporting evidence. The causal attribution comes from the hybrid controls.

**Stratification** (mean cm per quartile, Q1 → Q4):

| stratified by | FK | H0 IK | C | D |
|---|---|---|---|---|
| direction error (Q1 0.4–6.1°, Q4 18.7–133°) | 4.5 → 24.6 | 4.2 → 25.1 | 3.9 → 25.0 | 0.9 → 3.7 |
| \|reach error\| (Q4 ≥ 0.049) | 9.4 → 19.1 | — | — | 0.15 → 6.29 |
| GT bend angle | — | 8.9 → 15.3 | — | 0.49 → 4.48 |

When stratified by reach error, D grows much more steeply than by direction (0.15 → 6.29 cm). This is the secondary reach tail on strongly bent rows.

## Bend-plane attribution (same GT endpoint target)

The elbow is placed by H0 bend evidence (the runtime rule, made perpendicular to the GT endpoint direction) and compared with the elbow placed by GT evidence:

| GT bend | rows | side flips | plane error p50 / p95 | elbow error p50 |
|---|---:|---:|---|---:|
| 0–15° | 29 | **6** | 53.3° / 146.7° | 2.05 cm |
| 15–45° | 183 | 0 | 15.2° / 43.6° | 1.48 cm |
| 45–90° | 155 | 2 | 11.4° / 50.4° | 2.55 cm |
| ≥ 90° | 28 | 2 | 12.4° / 121.6° | 4.62 cm |

- **Near straight:** the bend side is unreliable there (6 of 29 flips), but straight arms barely move the elbow, so the effect on elbow position stays about 2 cm.
- **Large Worklog 61 rotation deltas:** on the 14 rows where max(proximal, distal) delta was ≥ 10°, there are 2 flips and the plane error has p50 26° and p95 122°. Those rows are also where the H0 ratio is wrong (|log error| p50 0.60) and the H0 direction is wrong (p50 16°).
- **Size of the bend effect:** it is real but secondary. It is limited to elbow placement, at about 1.5–2.5 cm median, and does not change the endpoint.

## Owner cases (Worklog 61 cases kept; extremes added; overlaps reported)

| case | row | GT / H0 ratio | GT / H0 reach | direction error | plane error (side) | FK | H0 IK | C | D |
|---|---|---|---|---:|---|---:|---:|---:|---:|
| known_good | dancing #424 | 1.06 / 1.08 | .979 / .986 | 6.9° | 8.7° (agree) | 6.77 | 6.51 | 6.48 | 0.38 |
| dynamic | crosscountry #462 | 1.07 / 0.96 | .971 / .974 | 28.4° | 3.1° (agree) | 25.89 | 26.22 | 26.17 | 0.18 |
| turning (review) | crosscountry #211 | 1.07 / 0.63 | .925 / .940 | 51.1° | 19.5° (agree) | 43.20 | 44.24 | 43.87 | 0.86 |
| tracking_loss review | hug #312 | 0.97 / 0.89 | .893 / .901 | 10.6° | 12.8° (agree) | 9.31 | 9.12 | 9.07 | 0.45 |
| near_straight | crosscountry #525 | 1.07 / 1.05 | .997 / 1.000 | 21.3° | 157.5° (**flip**) | 20.29 | 20.28 | 20.25 | 0.14 |
| strongly_bent = largest_reach_error | crosscountry #173 | 1.07 / 1.71 | .812 / .354 | 97.0° | 111.3° (**flip**) | 38.37 | 50.87 | 66.92 | 25.19 |
| largest_ratio_error | hug #397 | 0.97 / 4.52 | .617 / .884 | 17.0° | 78.2° (agree) | 26.29 | 18.98 | 10.03 | 14.71 |
| largest_direction_error | dancing #176 | 1.06 / 1.31 | .706 / .755 | 133.2° | 84.2° (agree) | 72.53 | 73.82 | 71.34 | 2.69 |
| turning (invalid) | crosscountry #202 | not matched: H0 IK UNAVAILABLE (`source_end_invalid`) | | | | | | | |
| invalid_endpoint | hug #406 | not matched: H0 IK UNAVAILABLE (`source_mid_invalid`) | | | | | | | |

Columns FK, H0 IK, C and D are endpoint errors against `E_GT` in cm. Oracle error is 0 on every row.

- **Overlap:** `strongly_bent` and `largest_reach_error` are the same row. Neither replaced the other.
- **Exceptions to the direction pattern:** hug #397 and crosscountry #173 are the only cases where reach/ratio error is comparable to or larger than direction error. On #397, the bad H0 ratio (4.52) distorts reach so badly that FK's direction-only use happens to score worse than IK. On #173 the H0 elbow is badly folded (H0 bend 151° vs GT 71°), so both the reach and the side are wrong.

The full frame-by-frame table, with every column from §10, is in `attribution_table.csv` (395 rows) with `rows.json`.

## Outputs

On LabServer63, `~/animcv-output/framepose_ik_gt_attribution/` contains:
- `report.json` (SHA-256 `f7a100cde398e3117717520247d35a61b992c401fcf30d48ea2f4d86c5c9d523`)
- `attribution_table.csv`
- `rows.json`

These are machine-readable outputs and stay on the server. The regime is `benchmark_detector_observation`. No renders were produced.

## Architecture classification

**CASE B — DIRECTION DOMINATED**, with a secondary reach tail and a minor bend-plane component. The hybrid controls give:
- **GT direction + H0 reach:** close to the oracle (median 0.89 cm, mean 2.15 cm).
- **H0 direction + GT reach:** stays poor (median 8.91 cm, mean 11.93 cm), essentially no better than current H0 IK (9.38 / 12.30) or FK (9.97 / 12.51).

Correcting the reach fraction, and hence the segment ratio, removes only about 0.4 cm of an error of roughly 12 cm. Correcting the endpoint direction removes about 10 cm.

Answer to the completion question: on real data, the FK/IK disagreement and both methods' error against the GT-anchored target come mainly from **error in the FramePose geometry that defines the target**. They do not come from true target-rig proportion adaptation, which is only 0.50 cm mean for these subjects against BaseRig.

| candidate limitation | finding |
|---|---|
| endpoint direction | the dominant runtime limitation: H0 shoulder→wrist direction error is p50 10.3°, p95 38° |
| reach fraction / segment ratio | genuinely wrong framewise (GT CV 0 vs H0 CV 0.18–0.30) but secondary for the endpoint; matters on a strongly bent / mis-folded tail (D Q4 mean 6.3 cm) |
| bend plane | correct side on 97.5% of rows; flips concentrate in near-straight arms and mis-folded rows; elbow-placement error ~1.5–2.5 cm median |
| two-bone IK abstraction | not the limitation: with oracle direction and reach it reaches `E_GT` exactly with no clamps, and its proportion adaptation is the correct, small (≤ 1.36 cm) effect |

**Implications** (not implemented here):
- **Redesigning source-chain scale or proportion is not justified first.** Median lengths, subject scale or bone-length smoothing all address the secondary term only.
- **Keep the current IK as the Worklog 61 contract.** The current H0 IK gives no material GT accuracy gain over FK (0.21 cm mean, and worse on crosscountry), so do not switch the runtime default from FK to IK on this evidence.
- **The next justified work is FramePose chain direction quality** (the 3D direction of elbow and wrist relative to the shoulder). That belongs to the FramePose stage, not retargeting.

Product IK success is not claimed. Stopped before legs, ankles, Contact and foot locking.

## Files and tests

Added (commit `c43a6ce`):
- `src/retarget/framepose_ik_gt_attribution.py` (evaluation only)
- `scripts/run_framepose_ik_gt_attribution.py`
- `tests/test_framepose_ik_gt_attribution.py`

Plus this worklog. No existing file was modified.

The focused tests cover:
- matched GT/H0 geometry under the shared runtime transform and source scale
- `E_GT` construction
- identical H0/GT giving zero IK and hybrid error, with FK showing only true adaptation
- direction-only and reach-only hybrid isolation
- bend-side flip detection on the same GT target
- the degenerate-side fallback label
- rejection of invalid or degenerate chains
- the static no-leakage check on production modules

Results:
- New tests together with the unchanged Worklog 61, FK and object-owner tests: 33 passed.
- Broader focused retarget, Blender and rig suites: 155 passed, 1 skipped (torch absent locally).

Production behaviour did not change, so the full regression was not run.
