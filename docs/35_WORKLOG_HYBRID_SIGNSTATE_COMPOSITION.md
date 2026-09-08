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
