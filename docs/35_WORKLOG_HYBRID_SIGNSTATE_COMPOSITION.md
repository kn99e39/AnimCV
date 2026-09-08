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
