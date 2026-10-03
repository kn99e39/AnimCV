# Worklog — Visual Supervision Diversity / Sequence-Balanced Sampling (2026-10-04)

## Starting state and files

- **Start:** fetched `origin/arch/single_frame_first` and confirmed local = remote = `d8654774dca22102128ba09c4d62fbe03b030ad1`, with a clean worktree.
- **Commits:**
  - `13267c8`: sampler module, script, tests
  - the commit adding this worklog
- **Files added:**
  - `src/framepose/visual_supervision_sampling.py`
  - `scripts/run_visual_supervision_diversity.py`
  - `tests/test_framepose_visual_supervision_sampling.py`
  - this worklog
- **Unchanged:** every existing file and checkpoint, including the Worklog 68 cache, the Worklog 69 head and checkpoints, Worklog 70, production FramePose, FK and IK.

**Run note.** The shared GPU was again occupied by an unrelated project's containers (`vismodeling-planner`). A server-side wait loop (`~/animcv-output/_scratch/run_w71_when_gpu_free.sh`) started the single run once GPU memory fell below 2 GB. No other job was touched.

## Fixed evidence and architecture

**Evidence and model:**
- **Cache:** the Worklog 68 cache, identity verified (bank digest, sample-ID order, Qwen3-VL fingerprint `a0d72ded…`, `readout.npz` `f5d0eb9c…`).
- **Architecture:** the fixed Worklog 69 continuous visual head, unchanged: C1_QWEN_VISION inputs (13 pair-geometry features + endpoint A/B/global readouts), shared `Linear(4096→64)`, per-segment `Linear(205→64)→GELU→Linear(64→1)→tanh`, 301,955 parameters.
- **Identity check:** the Worklog 69 C1 checkpoint config and cache identity are verified before training.
- **Not used:** Worklog 70 attention, skeleton bias, new widths, regularization, Depth Anything, H0, oracle signs, temporal input.

**Training contract (Worklog 69, unchanged):**
- Worklog 64 target and GT masks, exact FrameBank splits
- seed 1337 with identical model initialization
- AdamW, learning rate 3e-4 cosine to 1e-5, weight decay 1e-4
- 200 epochs, batch 256, masked MAE
- validation mean \|f\| selection
- validation and test kept in their original distributions

## TRAIN composition

- **Size:** 34 TRAIN actor sequences, with 11,334 TRAIN rows and 10,897 eligible rows (at least one GT-valid arm segment).
- **Spread:** largest share **5.8%** (`courtyard_backpack_00` actor0, 628 rows, 42 s); top-5 share **27.4%**; median 284 rows; minimum 127 (`courtyard_dancing_01` actor0); maximum 628.
- **Sampling:** consecutive TRAIN rows are 2 source frames apart (p05–p95 = 2).
- **Implication:** the TRAIN set is only moderately imbalanced, about 5:1 between the largest and smallest sequence, and no scene dominates.

## Visual redundancy (cached frozen readouts; diagnostic, no threshold)

Each cell is the mean cosine similarity between TRAIN rows that are *k* rows apart within a sequence, with p05 / p95 in brackets.

| representation | lag 1 | lag 2 | lag 4 | lag 8 | random cross-sequence pairs |
|---|---|---|---|---|---|
| arm endpoints (S, E, W concatenated) | .866 [.664, .975] | .828 | .786 | .746 | .572 [.394, .731] |
| global mean token | .995 [.990, .998] | .993 | .991 | .987 | .922 [.820, .978] |

Neighbouring frames are highly redundant. Arm-readout similarity stays well above the cross-sequence level even 8 rows (16 frames) apart. The global token is nearly constant within a sequence, a scene signature.

## S0 / S1 sampler contract and actual contributions

| sampler | definition | draws per epoch | max deviation of sampled shares | largest / smallest sequence share |
|---|---|---:|---|---|
| **S0_FRAME_UNIFORM** | the Worklog 69 epoch exactly: one permutation of all 11,334 TRAIN positions per epoch from the seed-1337 CPU generator | 11,334 | 0.0037 from frame share | .0557 / .0132 |
| **S1_SEQUENCE_BALANCED** | per draw, a TRAIN sequence uniformly, then one eligible frame uniformly (with replacement), same generator seed | 11,334 | **0.0003 from uniform (1/34)** | .0296 / .0291 |

- **Budget:** both runs made **2,266,800 draws and 9,000 optimizer steps**. Eligible draws were 2,179,400 for S0 (which, like Worklog 69, also visits all-masked rows) and 2,266,800 for S1.
- **No target-awareness:** neither sampler sees targets, depth or errors.
- **Exact reproduction:** S0 reproduces Worklog 69 C1 **exactly**. The maximum training-curve difference is 0.0, the selected epoch (69) and errors are identical, and S0 test predictions equal C1's (difference 0.000).

## Training curves and generalization gap

| sampler | selected epoch | train MAE | validation MAE | gap (val − train) | ratio | final train | final validation | final ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| S0 | 69 | 0.0563 | 0.1969 | 0.141 | 3.50 | 0.0329 | 0.2041 | 6.20 |
| S1 | 69 | 0.0596 | 0.1970 | 0.137 | 3.30 | 0.0359 | 0.2044 | 5.69 |

The two curves overlap (validation 0.197–0.209 throughout). S1's slightly smaller gap comes entirely from a **worse training fit**, not better held-out error, so it does not count as a generalization improvement.

## Held-out results (S1 − S0)

**Test** (6,003 rows; mean / p50 / p95):

| segment | metric | S0 (= Worklog 69 C1) | S1 | S1 − S0 (improved share) | H0 | D1 |
|---|---|---|---|---|---|---|
| forearm | \|f\| | .243/.185/.666 | .242/.185/.672 | −0.001 (48%) | .283 | .265 |
| | elevation (°) | 16.2/12.3/44.5 | 16.1/12.3/43.9 | −0.09° | 18.7 | 17.6 |
| | full (°) | 18.1/13.7/47.3 | 17.9/13.9/46.8 | −0.13° | 20.4 | 19.4 |
| chain | \|f\| | .251/.192/.707 | .249/.191/.704 | −0.002 (50%) | .271 | .224 |
| | elevation (°) | 15.8/11.8/44.6 | 15.7/11.7/45.0 | −0.11° | 16.9 | 14.1 |
| | full (°) | 16.8/12.7/46.6 | 16.7/12.6/46.6 | −0.09° | 17.9 | 15.2 |
| upper (separate) | \|f\| | .272/.213/.740 | .274/.213/.746 | +0.002 (48%) | .237 | .209 |

**Validation** (3,056 rows), S1 − S0 \|f\|: forearm −0.003, chain −0.001, upper +0.004.

**Sign vs magnitude (test, stable rows).** S1 leaves the sign essentially unchanged. Magnitude error on rows where both samplers have the right sign moves slightly in S1's favour.

| segment | stable sign accuracy S0 → S1 | magnitude when both sign-correct, S0 → S1 |
|---|---|---|
| forearm | .837 → .834 | .211 → .200 |
| chain | .786 → .793 | .207 → .204 |

**Depth bins (test \|f\|, S0 → S1).** Balancing slightly shifts the forearm from near-plane accuracy toward out-of-plane accuracy, without a net gain.

| segment | near plane | intermediate | strongly out of plane |
|---|---|---|---|
| forearm | .166 → .181 | .241 → .246 | .309 → **.289** |
| chain | .179 → .179 | .239 → .238 | .377 → .371 |

## Clean / crosscountry / other (\|f\|, S0 → S1)

| scope | forearm | chain | upper |
|---|---|---|---|
| clean dancing + hug (425) | .195 → **.205 (+0.010; 42% improved)** | .183 → .175 (−0.008) | .185 → .185 |
| crosscountry (133) | .224 → .222 | .221 → .217 | .214 → .216 |
| all other held-out (8,501) | .239 → .236 | .229 → .228 | .245 → .247 |

The forearm regresses on the clean subset, which is not acceptable for a global data-contract change.

## Per-sequence accounting (S1 − S0 \|f\| sum per sequence)

| scope / segment | sequences improved | net gain (sum \|f\|) | largest single-sequence share | worst regressions |
|---|---|---:|---|---|
| test forearm (37 seq.) | 18 | 6.8 | 136% of net (`downtown_runForBus_01`) | `downtown_bus_00` (+6.3), `downtown_bar_00` (+4.8), `downtown_cafe_00` (+3.5) |
| test chain (37 seq.) | 21 | 11.8 | 58% of net (same sequence) | `downtown_bar_00`, `downtown_walking_00`, `outdoors_fencing_01` |
| held-out forearm (53 seq.) | 27 | 14.8 | 63% of net | — |
| held-out chain (53 seq.) | 30 | 14.7 | 47% of net | — |
| test upper | 21 | −10.6 (net worse) | — | — |

The tiny net forearm/chain gain is largely owned by one sequence. About half the sequences improve and half regress.

## Architecture classification

**CASE B: existing scene diversity is insufficient.**

- **The contract was applied correctly:**
  - S1 changed the training composition exactly as intended (per-sequence shares within 0.0003 of uniform);
  - it used the identical optimization budget, initialization and evaluation contract;
  - S0 reproduced Worklog 69 exactly.
- **Held-out error did not move:** validation 0.1969 → 0.1970, and test forearm/chain \|f\| −0.001/−0.002 with about 50% of rows improved.
- **The gap remains:** the train/validation gap stays at about 3.3–3.5× at selection (about 5.7–6.2× at the end). S1's marginally smaller gap comes from worse training fit.
- **No acceptable trade-off either:** the forearm regresses on the clean subset (+0.010), and the remaining small gains are owned by a single sequence.
- **No mixed-case split worth adopting:** the only visible shift is a forearm move from near-plane to out-of-plane accuracy that nets to zero.

**Answer to the completion question:** the Qwen visual-depth generalization ceiling is not substantially caused by imbalanced sampling of the existing 3DPW scenes.
- **Composition was already only mildly imbalanced:** largest sequence 5.8%, top-5 27%.
- **Re-weighting to perfect sequence balance changes nothing measurable.**
- **Redundancy is real but not fixable by re-weighting:** neighbouring frames are strongly redundant (arm readout cosine about 0.87 vs 0.57 across sequences, and the global token is nearly a per-scene constant). The 34 TRAIN sequences simply contain too few visually independent scenes.

Reweighting the current supervision cannot add that independence. Per the decision logic:
- visual-readout and sampler tuning stop here;
- the next justified architecture step is **substantially more diverse paired RGB↔3D supervision**;
- no external dataset was added in this batch.

## Outputs and tests

**Outputs:** LabServer63 `~/animcv-output/framepose_visual_supervision_diversity/`:
- `report.json` (SHA-256 `0640c60ec31f989ddebb9f3cf3fb0788f3804295af4ab41a570692025dc68203`)
- `S0_FRAME_UNIFORM.pt`, `S1_SEQUENCE_BALANCED.pt`
- `rows_heldout.json`

These are machine-readable and stay on the server. No renders were needed, because this batch changes only training sampling.

**Tests:** `tests/test_framepose_visual_supervision_sampling.py` has 5 tests covering:
- TRAIN sequence grouping that excludes ineligible and non-TRAIN rows
- S0 reproducing the exact Worklog 69 permutation stream deterministically
- S1 with the same draws per epoch, deterministic, only eligible TRAIN rows, uniform over sequences, while S0 follows frame frequency (contribution accounting)
- cosine-by-lag
- no target/depth/error terms in the sampler, the evaluation path unchanged, cache/model/budget identity checks, and no production import

Results:
- 26 passed in the training image, together with the unchanged Worklog 68, 69 and 70 tests.
- Locally: 3 passed, 2 skipped (torch absent).

No production file changed, so the full regression was not run.
