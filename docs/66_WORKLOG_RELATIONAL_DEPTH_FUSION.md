# Worklog — Late Relational Depth Fusion over Frozen H0 (2026-10-02)

## Starting state and files

- **Start HEAD:** fetched `origin/arch/single_frame_first` and confirmed local = remote = `37c47b9f602862458bc220c926e89fbcc3a976ff`, with a clean worktree.
- **Commits:**
  - `b3fa4bc`: module, script, tests
  - `e00b773`: test-only fix; the no-threshold check now skips the module docstring
  - the commit adding this worklog
- **Files added:**
  - `src/framepose/relational_depth_fusion.py`
  - `scripts/run_framepose_relational_depth_fusion.py`
  - `tests/test_framepose_relational_depth_fusion.py`
  - this worklog
- **Unchanged:** every existing file, including the Depth Anything provider, the Worklog 65 cache and D0/D1 artifacts, frozen H0, the Worklog 64 probe, production FramePose, AnimationSemantics, Root Orientation, FK and IK.

**Run order.** The LabServer63 run started from `b3fa4bc`. `e00b773` touches only a test assertion and cannot affect the run. All 26 focused tests passed afterwards in the training image.

## Worklog 65 evidence-cache identity

- The run consumed `~/animcv-output/framepose_depth_evidence_vits/evidence.npz` byte-for-byte.
- **SHA-256** `f6d3787677fa887349b06ab0f22f167bb3670530ea6b7b776481303f9bd82d1c`, equal to the Worklog 65 manifest. Cache identity digest `ff2c347f…`.
- **Checks:** the bank digest and sample-ID hash were verified, and the script refuses any mismatch.
- **Provider settings:** vits, checkpoint `715fade1…`, nearest-pixel sampling, per-frame z-score, forward = −z. None of these was re-run or changed.
- **Frozen H0** was consumed from its unchanged `.npy` files (split SHAs match docs/57).
- **Worklog 65 D0/D1** per-row predictions were read from `framepose_depth_evidence_ab/rows_*.json`. Those rows agree with this run's H0 f and GT f to ≤ 1e-9.

## Relational feature contract

Using Worklog 65's `forward_evidence` and `available` arrays:
- `delta_upper = ev[elbow] − ev[shoulder]`
- `delta_lower = ev[wrist] − ev[elbow]`
- `delta_chain = ev[wrist] − ev[shoulder]`

**Rules:**
- No coefficient is fitted and no GT statistics are applied.
- A relation is unavailable if either endpoint is unavailable. Its input is then 0, the same representation Worklog 65 used for unavailable depth.
- Unavailable relations affect 18 / 58 / 50 of the 9,059 held-out rows (chain / forearm / upper).

**H0 features:** `h0_{upper,lower,chain}_f = unit(v).Y` from frozen H0, the Worklog 64 definition.

## Fusion graph and R0/R1 identity

- **Graph:** one fixed graph, `6 → Linear(32) → GELU → Linear(32) → GELU → Linear(3) → tanh`, with **1,379 parameters**. There was no width, depth, activation or residual sweep.
- **R0_ZERO_RELATION:** inputs are [H0 f (3), 0, 0, 0].
- **R1_DEPTH_RELATION:** inputs are [the identical H0 f, the Worklog 65 relations].
- **Input check:** the script verifies that the first three inputs are identical, that R0's relations are exactly zero, and that R1's relations equal the derived Worklog 65 relations.
- **Initialization:** identical, from the same seed.

**Training (the Worklog 64/65 contract, unchanged):**
- target: GT f for upper, forearm and chain from bank `target_3d`
- masked MAE; no other term, weighting or regime weighting
- AdamW, learning rate 3e-4 cosine to 1e-5, weight decay 1e-4
- 200 epochs, batch 256, seed 1337, AMP
- the same train/validation/test splits; validation every 10 epochs; selection on validation mean |f| error; test never used
- no GT gating, no |f| threshold, no trust threshold, no per-regime model

**Selection:**

| candidate | selected epoch | validation mean \|f\| error |
|---|---:|---:|
| R0 | 139 | 0.2146 |
| R1 | 199 | 0.2088 |

**Structural caveat found during the run: H0 is in-sample on the training split.** H0's mean |f| error is 0.080 on train, against 0.216 on validation and 0.264 on test, because H0 was trained on those train frames. The fusion therefore learns from a distribution where H0 is about three times more accurate than on held-out frames. This systematically teaches it to trust H0 and discount the relations. The direction fixed the split and forbids retraining H0, so the confound could not be removed here and is reported as part of the result.

## Results

**Eligibility:** the same rows as Worklog 64/65, i.e. 6,003 test and 3,056 validation rows.

### Held-out test

Values are mean / p50 / p95. "Full" is the 3D direction on the observed 2D plane.

| segment | metric | H0 | W65 D0 | W65 D1 | R0 | **R1** |
|---|---|---|---|---|---|---|
| upper | \|f\| | .237/.168/.661 | .225/.162/.615 | .209/.159/.560 | .234/.163/.650 | .228/.160/.627 |
| | elevation (°) | 14.5/10.3/39.5 | 13.7/9.8/37.2 | 12.7/9.7/34.5 | 14.3/10.0/39.1 | 13.9/9.9/37.5 |
| | full (°) | 15.7/11.4/40.1 | 14.9/10.9/37.9 | 13.9/10.8/35.5 | 15.5/11.1/39.6 | 15.1/10.9/38.6 |
| forearm | \|f\| | .283/.203/.848 | .300/.219/.818 | .265/.204/.781 | .287/.204/.860 | .282/.202/.833 |
| | elevation (°) | 18.7/13.6/56.7 | 19.9/14.2/55.7 | 17.6/13.0/51.2 | 19.1/14.0/56.7 | 18.8/13.5/56.7 |
| | full (°) | 20.4/14.8/59.7 | 21.6/15.7/58.9 | 19.4/14.7/53.2 | 20.7/15.3/59.9 | 20.5/14.9/60.1 |
| chain | \|f\| | .271/.192/.810 | .267/.196/.743 | .224/.173/.612 | .262/.187/.792 | .253/.183/.746 |
| | elevation (°) | 16.9/11.7/50.9 | 16.6/11.8/46.9 | 14.1/10.5/38.6 | 16.4/11.4/49.9 | 15.8/11.1/47.1 |
| | full (°) | 17.9/12.6/52.1 | 17.6/12.8/48.0 | 15.2/11.6/39.8 | 17.4/12.5/51.2 | 16.9/12.2/48.3 |

**Paired comparisons on test** (mean; share of rows improved):

| segment | R1 − R0: \|f\| | R1 − R0: elevation | R1 − H0: \|f\| | R1 − H0: elevation | R1 − H0: full |
|---|---|---|---|---|---|
| upper | −0.006 (59%) | −0.43° | −0.010 (60%) | −0.64° | −0.60° |
| forearm | −0.006 (56%) | −0.31° | −0.001 (53%) | +0.05° | +0.09° |
| chain | −0.009 (61%) | −0.56° | −0.017 (66%) | −1.04° | −1.02° |

### Validation, clean, crosscountry, other

Each cell lists chain \|f\| as H0 / R0 / R1 / D1, then R1 − R0 and R1 − H0.

| scope | n | chain \|f\|: H0 / R0 / R1 / D1 | chain R1 − R0 | chain R1 − H0 |
|---|---:|---|---|---|
| validation, all | 3,056 | .201 / .199 / .192 / .208 | −0.0065 (60%) | −0.0087 |
| **validation, clean dancing + hug** | 425 | **.152 / .157 / .152 / .233** | −0.0046 (57%) | **+0.0003 (neutral)** |
| validation, crosscountry | 133 | .284 / .283 / .277 / .255 | −0.0056 | −0.0068 |
| validation, other | 2,498 | .205 / .202 / .195 / .202 | −0.0069 | −0.0104 |
| all other held-out (val other + test) | 8,501 | .251 / .244 / .236 / .218 | −0.0085 (61%) | −0.0153 |

**Clean subset per segment** (H0 → R1):

| segment | \|f\| | elevation | full | R1 − H0 elevation |
|---|---|---|---|---|
| upper | .151 → .139 | 9.3° → 8.6° | 10.6° → 9.9° | improves |
| forearm | .200 → .200 | 12.7° → 12.7° | 14.7° → 14.8° | +0.01° |
| chain | .152 → .152 | 9.5° → 9.5° | 10.6° → 10.7° | +0.003° |

- R1 **preserves H0 on the clean subset**, unlike Worklog 65 D0/D1, which degraded it (chain \|f\| .230/.233).
- Clean p95 is unchanged or better (chain full 25.7° → 24.9°).

### GT-depth stratification (test, evaluation only)

Mean \|f\| error by \|GT f\| bin, listed as H0 / D1 / R0 / R1.

| bin | upper | forearm | chain |
|---|---|---|---|
| < 0.17 | .208 / .179 / .201 / .197 | .186 / .166 / .193 / .184 | .166 / .151 / .160 / .151 |
| 0.17–0.5 | .216 / .203 / .213 / .206 | .282 / .270 / .288 / .274 | .244 / .208 / .241 / .232 |
| **≥ 0.5** | .391 / **.308** / .395 / .383 | .363 / **.342** / .363 / .369 | .470 / **.359** / .450 / .440 |

**Out-of-plane regime.** Worklog 65's strongly out-of-plane gains are **not retained**:
- **chain:** D1 is 0.111 better than H0, but R1 only 0.030 better
- **upper:** D1 0.083 better than H0, R1 0.008 better
- **forearm:** D1 0.021 better than H0, while R1 is 0.006 *worse*

### Observable relation-strength quartiles (held-out |delta_DA|, edges fixed as quartiles)

Mean \|f\| error by quartile, listed as H0 → R1 (R1 − R0, share improved).

| segment | Q1 | Q2 | Q3 | Q4 |
|---|---|---|---|---|
| chain | .216 → .206 (−.005, 61%) | .233 → .221 (−.006, 60%) | .248 → .233 (−.009, 61%) | .292 → .270 (−.013, 62%) |
| forearm | .243 → .247 (−.001) | .248 → .248 (−.003) | .279 → .273 (−.009) | .317 → .306 (−.015, 60%) |
| upper | .246 → .238 (−.004) | .197 → .187 (−.004) | .211 → .203 (−.003) | .240 → .233 (−.009, 62%) |

R1's benefit **grows with observable relation strength**, so the fusion does use evidence magnitude without any oracle gate. The size of that benefit is small, however.

### H0-vs-Depth-Anything complementarity (held-out, \|GT f\| > sin 10°)

Each cell is mean \|f\| error, with the share of rows where the predicted sign is correct.

**Chain** (5,691 rows):

| category | n | H0 | D1 | R0 | R1 |
|---|---:|---|---|---|---|
| **H0 wrong / DA right** | 970 | .659 (0%) | **.277 (78%)** | .642 (8%) | .596 (16%) |
| H0 right / DA wrong | 1,156 | .182 (100%) | .308 (73%) | .182 (99%) | .194 (98%) |
| both right | 3,241 | .205 | .222 | .201 | .195 |
| both wrong | 324 | .573 | .473 | .558 | .564 |

**Forearm** (6,077 rows):

| category | n | H0 | D1 | R0 | R1 |
|---|---:|---|---|---|---|
| H0 wrong / DA right | 1,017 | .679 (0%) | .356 (66%) | — | .639 (17%) |
| H0 right / DA wrong | 1,286 | .199 | .356 | — | .217 |

**Upper** (5,823 rows):

| category | n | H0 | D1 | R1 |
|---|---:|---|---|---|
| H0 wrong / DA right | 787 | .584 | .336 | .564 |
| H0 right / DA wrong | 1,535 | .147 | .185 | .140 |

**Depth Anything is genuinely complementary to H0.** It is right on 17% of stable chain rows where H0 is wrong, and Worklog 65's D1 used this (78% sign recovery). It is also wrong on a comparable number of rows where H0 is right (1,156 chain rows), and D1 paid for that (73% sign retention, \|f\| .182 → .308).

**R1 makes the opposite trade.** It almost never harms H0-right rows (98% retention) and recovers only 16% of H0-wrong/DA-right rows. Neither late fusion nor early fusion found a runtime-observable way to tell the two disagreement categories apart.

### Per-sequence ownership (53 held-out actor sequences)

| segment | R1 better than R0 | R1 better than H0 | largest single-sequence share of total R1 − R0 gain |
|---|---:|---:|---|
| upper | 38 | 42 | 19% (`flat_guitar_01`) |
| forearm | 42 | 33 | 8% (`outdoors_parcours_00`) |
| chain | 44 | 50 | 13% (`downtown_bar_00`) |

No single sequence owns the result.

## Owner cases

Each cell lists f as **GT / H0 / DA relation / W65 D1 / R0 / R1**, then the full angle as **H0 / R0 / R1**.

**known_good** (dancing #424)

| segment | f (GT / H0 / DA rel / D1 / R0 / R1) | full angle H0 / R0 / R1 |
|---|---|---|
| upper | .50 / .36 / .35 / .24 / .33 / .37 | 9.6° / 11.0° / 8.9° |
| forearm | .12 / .04 / .95 / .05 / .02 / .08 | 4.9° / 6.3° / 3.1° |
| chain | .32 / .21 / 1.30 / .13 / .18 / .23 | 7.3° / 8.6° / 5.9° |

R1 is the best of the three here.

**tracking_loss** (hug #312)

| segment | f (GT / H0 / DA rel / D1 / R0 / R1) | full angle H0 / R0 / R1 |
|---|---|---|
| upper | −.23 / −.17 / −.19 / −.11 / −.19 / −.18 | 4.0° / 2.5° / 3.1° |
| forearm | −.61 / −.38 / −.22 / .03 / −.38 / −.37 | 16.2° / 16.1° / 16.6° |
| chain | −.47 / −.31 / −.42 / −.02 / −.30 / −.30 | 10.6° / 10.8° / 11.2° |

#312 is the Worklog 65 fusion miss. R1 no longer collapses the forearm to about 0 as D1 did; it keeps H0's −0.38. But it does not move toward GT (−0.61) even though the relation (−0.22 forearm, −0.42 chain) points that way. The late fusion stays anchored to H0.

**largest_ratio_error** (hug #397)

| segment | f (GT / H0 / DA rel / D1 / R0 / R1) | full angle H0 / R0 / R1 |
|---|---|---|
| upper | −.89 / −.47 / .89 / .49 / −.52 / −.55 | 39.6° / 37.0° / 34.8° |
| forearm | .56 / −.26 / 2.35 / .62 / −.25 / −.13 | 113.1° / 112.8° / 108.9° |
| chain | −.24 / −.49 / 3.24 / .36 / −.47 / −.50 | 18.9° / 17.7° / 19.1° |

**largest_direction_error** (dancing #176)

| segment | f (GT / H0 / DA rel / D1 / R0 / R1) | full angle H0 / R0 / R1 |
|---|---|---|
| upper | .64 / −.79 / .85 / .87 / −.75 / −.74 | 92.3° / 89.1° / 87.9° |
| forearm | .59 / .32 / 2.73 / .94 / .42 / .46 | 93.3° / 89.3° / 87.5° |
| chain | .87 / −.41 / 3.58 / .86 / −.38 / −.36 | 129.3° / 127.5° / 126.8° |

Depth Anything is right on the upper arm and chain, and D1 used it. R1 keeps H0's wrong sign.

**dynamic** (crosscountry #462)

| segment | f (GT / H0 / DA rel / D1 / R0 / R1) | full angle H0 / R0 / R1 |
|---|---|---|
| upper | .72 / .34 / .65 / .46 / .30 / .32 | 27.7° / 29.7° / 29.0° |
| forearm | .32 / −.11 / −.14 / .07 / −.14 / −.13 | 26.9° / 28.5° / 27.8° |
| chain | .54 / .11 / .51 / .28 / .09 / .10 | 27.9° / 29.3° / 28.6° |

Depth Anything's chain relation is right here, but R1 does not use it.

**turning_review** (crosscountry #211)

| segment | f (GT / H0 / DA rel / D1 / R0 / R1) | full angle H0 / R0 / R1 |
|---|---|---|
| upper | −.70 / −.46 / .08 / −.10 / −.53 / −.54 | 50.7° / 48.3° / 48.0° |
| forearm | −.99 / −.66 / 1.64 / .14 / −.68 / −.59 | 40.6° / 38.9° / 45.9° |
| chain | −.91 / −.62 / 1.72 / .21 / −.65 / −.64 | 45.4° / 43.3° / 44.0° |

Depth Anything is wrong here, and R1 resists it better than D1.

**near_straight** (crosscountry #525)

| segment | f (GT / H0 / DA rel / D1 / R0 / R1) | full angle H0 / R0 / R1 |
|---|---|---|
| upper | .03 / .31 / 1.20 / .33 / .31 / .32 | 16.8° / 16.7° / 17.1° |
| forearm | −.10 / .35 / .05 / .19 / .34 / .33 | 26.2° / 25.9° / 25.4° |
| chain | −.03 / .33 / 1.25 / .33 / .32 / .33 | 21.3° / 20.7° / 21.2° |

**strongly_bent = largest_reach = largest_depth_component** (crosscountry #173)

| segment | f (GT / H0 / DA rel / D1 / R0 / R1) | full angle H0 / R0 / R1 |
|---|---|---|
| upper | .82 / −.61 / .95 / .38 / −.67 / −.60 | 92.2° / 97.3° / 92.0° |
| forearm | .81 / .90 / −.33 / .83 / .81 / .75 | 17.1° / 15.7° / 17.3° |
| chain | 1.00 / −.14 / .61 / .41 / −.06 / −.08 | 97.0° / 92.5° / 93.8° |

**Excluded:** #202 and #406 are not eligible rows.

**Summary of owner cases:** R1 rarely departs from H0. It helps where H0 is close (#424) and resists bad Depth Anything evidence (#211), but it does not take over correct Depth Anything evidence (#176, #462, #173 upper, #312). No negative case was replaced.

## Architecture classification

The success criterion, declared before results, is evaluated below:

| criterion | result |
|---|---|
| R1 materially improves R0 on held-out data | **Weakly:** consistent across segments, scopes and 38–44 of 53 sequences, but small (test chain \|f\| −3.5%, elevation −0.56°; upper/forearm −0.006) |
| R1 improves or preserves H0 overall | **Yes:** chain −0.017, upper −0.010, forearm −0.001 (forearm full angle +0.09°, essentially preserved) |
| clean improves or preserves | **Yes, preserves:** chain +0.0003, forearm equal, upper improves; p95 not worse |
| Worklog 65 strongly out-of-plane gains retained | **No:** \|GT f\| ≥ 0.5 chain .440 vs D1 .359 vs H0 .470; forearm R1 is worse than H0 |
| p95 does not materially regress | **Yes:** test chain p95 52.1° → 48.3°; forearm full p95 59.7° → 60.1° (+0.4°) |
| no single sequence owns the result | **Yes:** 8–19% largest share |

The out-of-plane retention criterion fails, so **relational late fusion is not architecture-relevant**.

**Classification: CASE C (no material relational benefit beyond what H0 already gives).**
- The fixed late fusion trades one regime for another relative to Worklog 65:
  - it preserves H0 (on clean and ordinary frames);
  - it gives up most of the out-of-plane correction that early fusion achieved.
- It does not fall under CASE A, because the out-of-plane gains are lost.
- It does not fall under CASE B, because it does not degrade clean frames.

**Answer to the completion question:**
- **Early fusion was not the only blocker.** Representing the evidence explicitly as segment relations and fusing late over H0 removes the clean-subset degradation, but cannot keep the out-of-plane correction.
- **The evidence is complementary but not separable at runtime.** Depth Anything is right where H0 is wrong (970 stable chain rows) about as often as it is wrong where H0 is right (1,156). With nearest-pixel relative depth plus H0's own f, neither fusion found an observable cue to tell these apart. The fusion benefit does grow with |delta_DA|, but only modestly.
- **In practice:** at the current evidence quality, Depth Anything cannot reliably complement frozen H0 as a universal arm-depth input. It corrects some out-of-plane failures only by accepting comparable harm elsewhere (D1), or it preserves H0 and corrects little (R1).

**Qualification on CASE C.** The late fusion was trained on train-split frames where frozen H0 is in-sample (|f| error 0.080 vs 0.26 held-out). That biases any H0-anchored fusion toward over-trusting H0, so this batch somewhat understates what late fusion could achieve. A fair late-fusion test would need held-out-quality H0 on the fusion's training frames, for example out-of-fold H0. That requires retraining H0 variants, which this batch forbids. It is recorded as a precondition for any future revisit, not attempted.

**Next steps, per the direction:**
- Depth Anything fusion work stops here.
- The next architecture family is **stronger RGB / vision / VLM evidence** for:
  - overall body yaw
  - near/far limb ordering
  - self-occlusion
  - limb overlap
  - semantic body orientation

  These are exactly the cues that might separate "Depth Anything right / H0 wrong" from the reverse. Nothing of that is implemented here.
- No production integration was done.

## Outputs and tests

**Outputs:** LabServer63 `~/animcv-output/framepose_relational_depth_fusion/`:
- `report.json` (SHA-256 `cdd4e4e10136b7b9d6abc9bdf9c12932fdf1032f79a345381ebdbad2a77a20f9`)
- `R0_ZERO_RELATION.pt`, `R1_DEPTH_RELATION.pt`
- per-row validation and test JSON

These are machine-readable outputs and stay on the server. No renders were produced.

**Tests:** `tests/test_framepose_relational_depth_fusion.py` has 6 tests covering:
- exact relational deltas and unavailable semantics
- frozen H0 f extraction (scale-free)
- R0 relations exactly zero, and R1 relations exactly the Worklog 65 relations
- identical graph, parameter count (1,379) and initialization, with deterministic inference
- same target, objective and splits; inputs only from H0 and the relations; no GT bins, thresholds or gates in the module
- Worklog 65 cache identity checked, and no production import of the new module

Results:
- With the unchanged Worklog 64 and Worklog 65 test files: 26/26 passed in the LabServer63 training image.
- Locally: 23 passed, 3 skipped (torch absent).

No production module changed, so the full regression was not run.
