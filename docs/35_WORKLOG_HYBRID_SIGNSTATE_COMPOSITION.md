# Worklog — Composing Learned Bilateral Conditioning with Explicit Hinge Constraints (2026-09-08)

> One bounded architecture batch on `arch/single_frame_first`. **No training,
> no fine-tuning, no sensor, no VLM, no prompt, no RGB, no Sign Contract
> change, no operator change, no new attention topology, no temporal
> evidence.** Every number comes from applying the frozen hinge operator to
> *stored* predictions of already-accepted learned candidates.
>
> The question: **when composed on the same continuous pose, do learned
> bilateral conditioning and explicit hinge constraints preserve each other's
> benefits?**

## 1. Starting/final HEAD

Start `f61956b` (docs/34 closed). This batch: `ccacbcd` (hybrid composition
replay + source-artifact identity parsing), plus the run and worklog commits.

## 2. What was preserved

docs/33 and docs/34 in full: `branch_constraint_v3`, `branch_constraint_dep_*`,
`constraint_attribution_v*`, every `O_*`, `H_NO_*`, `L_*` result, every
wrong-sign endpoint, the residual-hinge attribution. Nothing was overwritten
and no model was retrained. This batch writes a new lineage,
`hybrid_signstate_v1`.

Accepted findings carried forward unchanged: the explicit hinge constraint is
the strongest hinge mechanism tested under oracle signs; the one-bit hinge Y
sign is useful but incomplete; dependency-aware bilateral writing removes the
anchor-only collateral; a sign-only bilateral constraint remains substantially
weaker than learned bilateral conditioning; the "leaf good / anchor bad" rule
is refuted.

docs/34's magnitude finding is carried in its **bounded** form: depth-magnitude
exclusion is a *demonstrated major expressiveness limit*, **not** a proof that
it is the sole cause of the entire learned-vs-constraint yaw gap.

## 3. Source-artifact identity: parsed, not just hashed

docs/34 recorded the SHA-256 of the source evaluation. Hashing proves bytes
were present; it does not prove they describe the candidate, split and frame
count the replay declares. `src/framepose/replay_provenance.py` now parses it.

Verified per source, and the replay **refuses** on disagreement:

| check | result |
|---|---|
| prediction shape matches the split | verified |
| evaluation `candidate` matches the declared source candidate | verified |
| evaluation `frame_count` matches prediction/bank split | verified |
| evaluation `observation_regime` contains the bank's own regime | verified |
| evaluation `split` matches the replay split | **`unverifiable_in_source_schema`** |

The historical `animcv_frame_pose_evaluation_v1` schema records
`schema`, `candidate`, `observation_regime`, `frame_count` — and **no split
field at all**. That item is therefore reported as unverifiable rather than
guessed or silently passed. No historical evaluation file was rewritten.

## 4. Sources (stored, never recomputed)

| label | candidate | prediction | evaluation |
|---|---|---|---|
| `S0` | `S0_neutral_sign` | `sign_v1/S0/prediction_test.npy` | ✓ parsed |
| `O_HIP` | `O_HIP_oracle_hip_depth_only` | `sign_minimality_v1/O_HIP/…` | ✓ parsed |
| `O_BILATERAL` | `O_BILATERAL_oracle_forward_depth_only` | `sign_attr_v2/O_BILATERAL/…` | ✓ parsed |

`S1` is deliberately **not** used as a source: it already carries hinge sign
conditioning and would contaminate the ownership test.

Bank `bank_3dpw_paired_v2` (digest `75519e63…`), `test` split, 7,076 frames,
regime `benchmark_detector_observation` derived from the bank.

## 5. The ownership contract, verified on real predictions

The hybrid is:

```
geometry observation -> trained learned model -> continuous pose [STORED]
                     -> explicit four-field hinge constraint -> hybrid pose
```

No bilateral output-space constraint is used anywhere; the bilateral channel
stays learned. The replay **refuses to report** unless all of the following
hold on the real stored arrays — they are not assumed from the code:

| check | S0 | O_HIP | O_BILATERAL |
|---|---|---|---|
| only the four hinge middle joints written | ✓ | ✓ | ✓ |
| only the depth axis written | ✓ | ✓ | ✓ |
| every bilateral-owned metric **exactly** unchanged | ✓ | ✓ | ✓ |
| `torso_facing` / shoulder / hip sign agreement unchanged | ✓ | ✓ | ✓ |
| same, at the **opposite-oracle** endpoint | ✓ | ✓ | ✓ |

Exact equality is the contract, not a tolerance — those joints are not written
at all, so any drift would mean the abstraction leaked. Frames written: 1,283
(S0), 1,452 (O_HIP), 1,293 (O_BILATERAL).

Worked example, `O_BILATERAL`: root yaw 7.999 → 7.999, yaw P95 18.645 →
18.645, shoulder residual 2.20829 → 2.20829, hip residual 0.47902 → 0.47902,
`torso_facing` 0.9828 → 0.9828, shoulder sign 0.9561 → 0.9561, hip sign
0.9748 → 0.9748.

## 6. Composition result

| source | stage | root yaw° | yaw P95° | hinge flip | hinge MAE° | MPJPE mm | PA-MPJPE mm |
|---|---|---|---|---|---|---|---|
| `S0` | H0 | 10.920 | 28.670 | 0.0212 | 24.819 | 81.976 | 56.852 |
| `S0` | **H1** | 10.920 | 28.670 | **0.0122** | **22.740** | 82.060 | 57.106 |
| `O_HIP` | H0 | 8.261 | 19.535 | 0.0202 | 25.211 | 80.418 | 56.822 |
| `O_HIP` | **H1** | 8.261 | 19.535 | **0.0116** | **23.042** | 80.580 | 57.113 |
| `O_BILATERAL` | H0 | 7.999 | 18.645 | 0.0189 | 24.099 | 79.229 | 56.425 |
| **`O_BILATERAL`** | **H1** | **7.999** | **18.645** | **0.0116** | **22.129** | **79.366** | **56.694** |

`H1_BILATERAL_HINGE` is the best pose measured in this programme on every axis
that either mechanism owns: root yaw 7.999° (learned channel, untouched), hinge
flip 0.0116 and hinge MAE 22.129° (constraint channel), MPJPE 79.366 mm.

Per-field hinge sign agreement under `O_BILATERAL`:

| field | H0 | H1 |
|---|---|---|
| left_elbow_forward_bend | 0.8602 | **0.9350** |
| right_elbow_forward_bend | 0.8937 | **0.9488** |
| left_knee_forward_bend | 0.9021 | **0.9464** |
| right_knee_forward_bend | 0.8802 | **0.9481** |

## 7. Incremental value of the hinge constraint *after* the learned channel

Measured against `O_BILATERAL` itself, not against `S0`:

| quantity | H0 → H1 | change |
|---|---|---|
| hinge flip rate | 0.0189 → 0.0116 | **−0.0073 absolute, −38.6% relative** |
| hinge direction MAE | 24.099 → 22.129 | **−1.970°** |
| MPJPE | 79.229 → 79.366 | +0.137 mm |
| PA-MPJPE | 56.425 → 56.694 | +0.269 mm |

For context, the same operator over `S0` (docs/33): −42.5% relative flip for
+0.085 mm. The relative gain is slightly smaller over `O_BILATERAL` — it starts
from a lower flip rate — while the **absolute** outcome is better (0.0116 vs
0.0122 final flip; 22.129° vs 22.740° final MAE).

The cost is confined to the four joints the constraint owns, and is not uniform:

| joint | H0 mm | H1 mm | Δ |
|---|---|---|---|
| left_knee | 83.699 | 82.025 | **−1.674** |
| right_knee | 83.011 | 83.094 | +0.083 |
| left_elbow | 106.125 | 107.914 | +1.788 |
| right_elbow | 89.470 | 91.745 | +2.275 |

Every other joint is bit-identical. The knees are neutral-to-better; the elbows
pay for the branch correction. That is the expected consequence of docs/34's
magnitude finding: the operator installs the correct *side* using a depth
magnitude derived from the prediction's own (possibly wrong) perpendicular
offset, so per-joint position error can grow even as the branch becomes right.

## 8. Source dependence: the operator transfers

| source | H0 flip | H1 flip | relative | H0 MAE° | H1 MAE° | MPJPE cost |
|---|---|---|---|---|---|---|
| `S0` | 0.0212 | 0.0122 | −42.5% | 24.819 | 22.740 | +0.085 |
| `O_HIP` | 0.0202 | 0.0116 | −42.6% | 25.211 | 23.042 | +0.162 |
| `O_BILATERAL` | 0.0189 | 0.0116 | −38.6% | 24.099 | 22.129 | +0.137 |

The same deterministic operator delivers a 39–43% relative flip reduction on
three different continuous-pose sources, including two whose global depth and
orientation geometry the learned channel has already changed substantially.
The hinge mechanism is **not source-dependent**, and the benefit does not
disappear once the bilateral channel is present.

## 9. What enforcement does to the *historical* metric

The Sign Contract's one-bit Y branch and the full-3D hinge metric are different
predicates, so enforcing the contract can move the metric either way. docs/34
measured one direction only (branch satisfied, metric still flipped). Both are
now counted, over the chain-frames the constraint actually corrected:

| source | fixed | broken | stayed flipped | stayed fine |
|---|---|---|---|---|
| `S0` | 211 | 6 | 11 | 1,160 |
| `O_HIP` | 192 | 6 | 14 | 1,372 |
| `O_BILATERAL` | **175** | **7** | 4 | 1,204 |

**Enforcement fixes the historical metric 25× more often than it breaks it**,
and the magnitudes are asymmetric in the same direction: a fixed chain improves
by 63–81° on average, a broken one worsens by 58–68°. The "the contract called
it wrong but the 3D metric was already right" case is real — it is the mirror
of docs/34's outcome B — but it is 7 of 1,390 corrected chain-frames (0.5%).

## 10. Residual hinge attribution on the hybrid

Same decomposition as docs/34, rebuilt on `H1_BILATERAL_HINGE`:

| cause of a residual flip | over `S0` | over `O_BILATERAL` |
|---|---|---|
| oracle never requested a branch | 161 (55.7%) | 138 (49.5%) |
| requested, prediction unreadable | 79 (27.3%) | 58 (20.8%) |
| **Y sign satisfied, full-3D still flipped** | 49 (17.0%) | **83 (29.7%)** |
| total residual flips | 289 | 279 |

As the upstream continuous geometry improves, the *unreadable* and
*never-requested* failures shrink (79→58, 161→138) and the residual becomes
increasingly dominated by the **one-bit incompleteness**, which rises from 17.0%
to 29.7% of all residual flips. In absolute terms it also grows (49→83), because
more chains are now readable and satisfied at all — `already_satisfied` rises
20,242 → 20,345 and every per-field satisfied rate improves.

**This is the architecturally significant part:** the known Sign Contract limit
does not shrink as the Core gets better — it becomes proportionally *more* of
what is left. The contract is unchanged in this batch, as directed, and the
limit stays visible.

## 11. Wrong-hinge-sign endpoint: damage is local

Opposite-oracle hinge signs, learned bilateral input unchanged:

| source | root yaw° (H0 → wrong) | flip | hinge MAE° | MPJPE mm |
|---|---|---|---|---|
| `S0` | 10.920 → **10.920** | 0.5449 | 94.224 | 97.182 |
| `O_HIP` | 8.261 → **8.261** | 0.5455 | 93.613 | 95.440 |
| `O_BILATERAL` | 7.999 → **7.999** | 0.5429 | 93.838 | 93.561 |

The structural property holds exactly: **catastrophically wrong hinge advice
does not touch the learned channel at all.** Root yaw, both depth residuals and
all three non-hinge sign agreements are bit-identical to `H0` — the replay
asserts this for the wrong-sign endpoint with the same exact-equality contract
it uses for the hybrid, and it passed on all three sources.

Meanwhile hinge quality collapses (flip 1.2% → 54.3%, MAE 22.1° → 93.8°) and
MPJPE rises ~14 mm. Local damage, global safety. No corruption sweep was run.

## 12. A measured pathology of the frozen operator

The review surfaced a correction of **+3,108 mm** — the limb was nearly
parallel to the camera depth axis, so the closed form
`δ = −2·o_y·|a|²/(a_x²+a_z²)` exploded while still passing the operator's
`sqrt(f) ≥ 0.1` guard (which permits amplification up to 200×). That frame's
MPJPE went 60.0 → 252.1 mm and its hinge error 21.0° → 123.5°.

Distribution of |δY| over the 1,390 corrected chain-frames (`O_BILATERAL`):

| p50 | p90 | p99 | p99.9 | max |
|---|---|---|---|---|
| 62.0 mm | 191.6 mm | 430.8 mm | 1,756.9 mm | 3,108.4 mm |

`> 200 mm`: 126 (9.1%). `> 500 mm`: 12 (0.86%). `> 1 m`: 6 (0.43%). `> 2 m`: 1.

docs/33 counted only the *refused* singular cases (1 frame in 23,083). The
near-singular-but-not-refused cases are the dangerous ones, and they are the
main reason the elbows pay MPJPE in Section 7. **The operator was not changed**
— its mathematics and threshold are frozen by directive — so this is recorded
as a measured, bounded defect for a future batch to decide on.

## 13. Real-frame review

`hybrid_signstate_review.json`, six buckets, each subsampled evenly across
qualifying frames (16–18 distinct 3DPW sequences per bucket):

| bucket | rows | example |
|---|---|---|
| `hinge_corrected_by_hybrid` | 25 | `downtown_arguing_00#000300` right elbow, −1→+1, δY +122.6 mm, hinge 94.7°→**15.6°** |
| `branch_corrected_but_position_worsened` | 25 | `#000220` left elbow, δY +121.8 mm, hinge 26.3°→30.5° |
| `y_sign_satisfied_but_still_full_3d_flipped` | 25 | `downtown_bar_00#000339` left elbow, +1 read back +1, hinge **105.2°** unchanged, δY 0 |
| `contract_called_it_wrong_but_the_3d_metric_did_not` | **7** | `downtown_cafe_00#000375` δY **+3,108 mm**, hinge 21.0°→123.5° |
| `requested_but_unresolved` | 25 | `#000075` left knee, requested −1, prediction reads UNKNOWN, δY 0 |
| `opposite_sign_catastrophic_bend` | 25 | `#000000` right elbow, δY −72.9 mm, hinge 14.3°→**136.5°** |

Each row carries `sample_id`, sequence, frame index, source image path,
requested sign, read-back before/after, moved joint, δY, source and hybrid XYZ,
full-3D hinge error before/after and frame MPJPE before/after. No VLM was used.

## 14. Classification: **A — hybrid supported**

- **Learned bilateral metrics preserved** — exactly, on all three sources, at
  both sign endpoints, verified on the real predictions and not merely from
  code structure.
- **Explicit hinge constraint adds material benefit after the best learned
  bilateral mechanism is already present** — −38.6% relative flip, −1.97°
  hinge MAE, and it transfers across three different continuous-pose sources.
- **Continuous position cost far below the learned hinge-conditioning
  alternatives** — +0.137 mm against `O_HINGE`'s +1.464 mm and `L_HINGE`'s
  +2.134 mm, i.e. roughly **11× and 16× cheaper** for a larger branch gain.

`H1_BILATERAL_HINGE` is the best pose this programme has measured: root yaw
7.999°, hinge flip 0.0116, hinge MAE 22.129°, MPJPE 79.366 mm.

## 15. Architecture, and what remains open

The evidence now supports closing AnimCV's **SignState integration
architecture** as:

```
sign source  -> discrete signs only
bilateral    -> learned conditioning        (owns orientation + depth magnitude)
hinge        -> explicit branch constraint  (owns the four local bend branches)
```

This is no longer a per-channel hypothesis. The two mechanisms were composed on
the same continuous pose and each preserved the other's benefit exactly.

**Not promoted, and not turned on by default.** Every sign used here is an
oracle. **Architecture closure is not sensor validation, and sensor validation
remains entirely open** — no sensor was chosen, built, or evaluated in this
batch or any previous one. The wrong-sign endpoints show both channels are
high-authority: a wrong hinge sign is locally catastrophic even though it
cannot reach the learned channel.

Three measured limits stay visible and unrepaired by directive: the one-bit
hinge sign's incompleteness (now 29.7% of residual flips), the near-singular
δY tail (Section 12), and the depth-magnitude exclusion that caps the bilateral
constraint (docs/34, carried in its bounded form).

## 16. Tests and environment

12 focused tests in `tests/test_frame_pose_hybrid_composition.py`: source
identity binding and parse-level refusals (wrong candidate, wrong frame count,
wrong regime, wrong shape), `unverifiable_in_source_schema` for the missing
split field, the ownership boundary (only hinge middles, only the depth axis,
refusal otherwise), exact-equality bilateral preservation including a 1e-9
drift being rejected, the enforcement-effect accounting, and an end-to-end
replay asserting ownership, preservation at both endpoints, cross-table
completeness and bit-exact determinism across two runs.

Shared production code was added (`src/framepose/replay_provenance.py`), so full
regression was run at closure: **487 passed, 40 skipped**.

`hybrid_signstate_v1` and `_v2` produce identical aggregates; `_v2` adds the
enforcement-effect accounting and the corrected review sampling. **docs/33's
`C_HINGE_ALL` is reproduced byte-identically** by the hybrid path
(`34045891fb544928…`), so the new code changed nothing about the historical
result.

LabServer63, `animcv-framepose:cuda118`, repo mounted read-only. No GPU work,
no training, no model execution.

**STOP.** No sign sensor is chosen here.
