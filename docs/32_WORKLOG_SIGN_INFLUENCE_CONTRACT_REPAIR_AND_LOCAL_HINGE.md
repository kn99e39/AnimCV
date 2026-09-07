# Worklog — Sign-Influence Contract Repair and Local-Hinge Topology (2026-09-07)

> Diagnostic + one conditional architecture batch on `arch/single_frame_first`.
> No VLM, no prompt, no orientation retraining, no Geometry Core change. The
> valid 3-of-4 leave-one-out result from docs/31 is preserved exactly and was
> not rerun. The question: **do the two knee signs remain necessary because of
> the information they carry, or because of the current conditioning
> topology?**

## 1. Starting/final HEAD

Start `23e4447` (docs/31: the all-four hinge requirement is refuted). Final
HEAD after this batch: `b2afe12` plus this worklog's own commit.

## 2. Which docs/31 claims remained valid

Unchanged, unaltered numbers:

- The four 3-of-4 leave-one-out training results themselves (Section 3-4 of
  docs/31): `H_NO_RIGHT_ELBOW` reproduces the full `O_HINGE` pooled benefit
  exactly; `H_NO_LEFT_KNEE`/`H_NO_RIGHT_KNEE` destroy it; `H_NO_LEFT_ELBOW`
  retains 47%. **`right_elbow_forward_bend` is NOT necessary under the
  current topology** — this training-level result does not depend on the
  influence probe at all and is untouched.
- `S0`/`S1`/`O_HINGE`/`O_ELBOWS`/`O_KNEES` checkpoints, reports, fingerprints.

## 3. Which docs/31 locality claims were provisional due to the probe mismatch

`diagnose_sign_influence.py` fed every checkpoint the **full 7-field
oracle** as its baseline sign state (`base_signs = oracle[positions]`,
unconditionally), regardless of what that checkpoint was actually trained
with. `O_HINGE` was trained with `torso_facing`/`shoulder_forward_depth`/
`hip_forward_depth` permanently `UNKNOWN` — the v1 probe put it in an
input state it never saw during training. This affected docs/31 Sections
5-8 (the frozen-weight influence matrix, the reconstructed-sign-change
matrix, the locality verdict) and, downstream, the sensor-requirement
wording in Section 12. It did **not** affect the leave-one-out training
result itself (Section 2 above).

## 4. Active-sign contract representation

Checkpoints do not self-describe their active sign fields (`CandidateConfig`
records `sign_source` but not the field list; adding that retroactively to
historical checkpoints was out of scope and unnecessary). The one
authoritative source is `run_sign_experiments.CANDIDATES[name]["fields"]` —
exactly what was passed to `mask_fields(oracle, ...)` at training time for
every experiment in this lineage. `scripts/diagnose_sign_influence.py`
(v2) imports it directly (`active_fields_for(name)`) and **refuses** any
checkpoint name it does not recognize, rather than guessing from the name
or defaulting to the full oracle. `run_sign_experiments.py`'s matrix now
also records `inactive_sign_fields` and `sign_source` explicitly per
candidate for future diagnostics; no historical checkpoint was rewritten.

## 5. Corrected O_HINGE influence matrix

Frozen weights, `mask_fields(oracle, ["*_forward_bend"×4])` as baseline
(orientation fields `UNKNOWN` throughout, matching training), 1,500 test
frames, `split=test`.

| toggled | own chain mm | other hinge mm | own/other | own-sign (diagonal) |
| --- | ---: | ---: | ---: | ---: |
| left_elbow | 54.96 | 17.48 | 3.15 | 0.732 |
| right_elbow | 47.58 | 15.47 | 3.08 | 0.771 |
| **left_knee** | 48.98 | 41.18 | **1.19** | **0.369** |
| **right_knee** | 45.40 | 28.01 | **1.62** | 0.297 |

Full reconstructed-sign-change matrix (off-diagonal = leakage into another
hinge chain's branch):

| toggled | → L elbow | → R elbow | → L knee | → R knee |
| --- | ---: | ---: | ---: | ---: |
| left_elbow | **0.732** | 0.049 | 0.019 | 0.015 |
| right_elbow | 0.032 | **0.771** | 0.011 | 0.012 |
| left_knee | 0.051 | **0.118** | **0.369** | 0.061 |
| right_knee | 0.042 | **0.071** | 0.044 | **0.297** |

## 6. Corrected S1 influence matrix

Same probe, `S1` (full 7-field oracle — its baseline is unaffected by the
repair since it was trained with everything active).

| toggled | own chain mm | other hinge mm | own/other | own-sign |
| --- | ---: | ---: | ---: | ---: |
| left_elbow | 50.86 | 14.65 | 3.47 | 0.696 |
| right_elbow | 49.63 | 15.02 | 3.30 | 0.708 |
| left_knee | 44.08 | 32.84 | 1.34 | 0.345 |
| right_knee | 45.09 | 24.79 | 1.82 | 0.504 |

Shape matches `O_HINGE`: elbows clearly local (ratio >3, diagonal ~70%),
knees weakest on both axes.

## 7. Corrected 3-of-4 influence results

Console summary (own chain / other hinge / rest, mm; each checkpoint
toggles only the hinge fields it was actually trained with):

| checkpoint | toggled | own | other hinge | rest |
| --- | --- | ---: | ---: | ---: |
| H_NO_LEFT_KNEE | left_elbow | 59.39 | 30.17 | 18.79 |
| | right_elbow | 46.60 | 18.69 | 16.46 |
| | right_knee | 47.07 | 27.06 | 15.69 |
| H_NO_RIGHT_KNEE | left_elbow | 65.84 | 27.14 | 18.46 |
| | right_elbow | 51.17 | 16.20 | 13.87 |
| | left_knee | 47.32 | 42.97 | 21.51 |
| H_NO_RIGHT_ELBOW | left_elbow | 58.66 | 32.58 | 21.18 |
| | left_knee | 46.54 | 36.32 | 16.91 |
| | right_knee | 44.34 | 28.75 | 17.12 |

Full JSON: `/home/nd/animcv-output/framepose/sign_influence_v2/sign_influence.json`.

## 8. Position-level vs branch-level locality (corrected)

**Position-level**: a single toggled hinge bit under `O_HINGE` still
displaces unrelated hinge chains by 15–41 mm on average, with the knees
consistently the worst (own/other ratio 1.19–1.62, versus 3.08–3.15 for
the elbows). This did **not** improve after the contract repair —
`left_knee`'s ratio is *worse* under the corrected probe (1.19) than the
invalid one reported (1.48).

**Branch-level**: the diagonal dominates for elbows (73–77%) but the
**corrected left-knee diagonal is only 37%** (down from the invalid
probe's 57%) — the model reconstructs its own trained left-knee branch
correctly barely more than a third of the time even when explicitly
told its sign. Off-diagonal leakage into the right-elbow branch roughly
doubled after correction (left_knee→right_elbow 0.068→0.118,
right_knee→right_elbow 0.035→0.071).

## 9. Difference from the invalid original influence probe

| | v1 (invalid, full oracle baseline) | v2 (corrected, masked baseline) |
| --- | ---: | ---: |
| left_knee own/other | 1.48 | **1.19** (worse) |
| right_knee own/other | 1.77 | **1.62** (worse) |
| left_knee own-sign | 0.572 | **0.369** (substantially worse) |
| right_knee own-sign | 0.184 | 0.297 (better) |
| left_knee→right_elbow leak | 0.068 | **0.118** (worse) |
| right_knee→right_elbow leak | 0.035 | **0.071** (worse) |

The repair did **not** vindicate docs/31's optimistic "discrete branch
decisions are 92–99% local" framing for the knees — it revises the
left-knee diagonal down by 20 points and roughly doubles two off-diagonal
leakage rates. This is Case D territory on its own (the specific 92–99%
number was partly an artifact of the OOD contract), which is folded into
the Section 10 classification below.

## 10. Updated mechanism classification

**CASE B — TOPOLOGY LEAKAGE SUPPORTED**, with the qualifier that the
underlying `92–99%` branch-locality figure is itself revised (Case D
component, Section 9).

Both required conditions for B hold: (1) knee removals lose the pooled
benefit (docs/31, unchanged) and (2) the corrected, in-distribution knee
toggles still cause **systematic large off-target propagation** at the
position level (own/other ratio 1.19–1.62, not "substantially more local"
than before — if anything worse) and comparable-or-worse leakage at the
branch level. Docs/31's Case A/B "confounded, cannot separate" verdict is
superseded: the corrected probe gives no evidence of improved locality for
the knees, so the leakage explanation is not weakened by the repair — if
anything it is strengthened. This is what gates Section 12.

## 11. Whether conditional local candidate executed

**Yes.** Section 10 supports B, so `scripts/run_sign_experiments.py` gained
`ModelConfig.hinge_sign_injection` (`"pre_attention"` default/historical,
`"post_attention"` new) and five candidates (`L_HINGE`, `L_NO_LEFT_KNEE`,
`L_NO_RIGHT_KNEE`, `L_NO_RIGHT_ELBOW`, `L_NEUTRAL`) were trained under
`--hinge-sign-injection post_attention`.

## 12. Architecture difference, parameter count, results

**Architecture difference**: the four hinge fields' embedding lookup is
computed identically (same `sign_embedding` table, same `sign_joint_mask`,
split once into fixed orientation/hinge-only views — not learned, not
tuned), but its contribution is added **after** every self-/cross-attention
block, immediately before the shared per-joint output head (`LayerNorm` +
position-wise MLP, which mix nothing across joints). Orientation fields
keep the historical pre-attention injection in both modes. No new layer,
no width change, no MLP, no gate, no learned weight, no sign-strength
tuning.

**Parameter count**: `1,657,603` trainable parameters for both `L_HINGE`
and `L_NEUTRAL` — identical to every current-topology candidate in this
lineage (docs/31's own reported count).

**L_HINGE result** (test split): pooled hinge flip `0.0193`, hinge MAE
`24.58`, against `L_NEUTRAL`'s `0.0218`/`25.36` — a **local-topology
benefit of 0.0025 flip / 0.78° MAE**, roughly half of the current
topology's `O_HINGE`-vs-`S0` benefit (`0.0053` flip / `1.38°` MAE). The
local topology's own hinge-conditioning effect is real but weaker.

**Knee leave-one-out** (the decisive comparison):

| candidate | pooled hinge flip | vs L_HINGE benefit retained |
| --- | ---: | ---: |
| L_NEUTRAL | 0.0218 | 0% (by definition) |
| **L_NO_LEFT_KNEE** | **0.0186** | **127%** (exceeds full-set) |
| **L_NO_RIGHT_KNEE** | **0.0182** | **142%** (exceeds full-set) |
| L_HINGE (all four) | 0.0193 | 100% |
| L_NO_RIGHT_ELBOW | 0.0224 | −20% (worse than neutral) |

Under the local topology, **removing either knee sign no longer destroys
the pooled hinge benefit — it doesn't measurably cost anything at all**,
in contrast to the current topology's `H_NO_LEFT_KNEE`/`H_NO_RIGHT_KNEE`
(8%/−19% retained). This is the exact pattern docs/32's decision rule
calls "substantially topology-induced."

**Right-elbow leave-one-out** (unexpected reversal): `L_NO_RIGHT_ELBOW`
loses the local benefit entirely (`0.0224` flip, *worse* than
`L_NEUTRAL`'s `0.0218`) — the opposite of the current topology, where
`H_NO_RIGHT_ELBOW` reproduced the full benefit exactly. Right-elbow
necessity flips from dispensable to required when the topology changes.
This is a new, genuinely surprising finding this batch surfaces but does
not fully explain; it is reported as-is, not resolved.

**Corrected locality matrix for the local checkpoints** (same probe,
same 1,500 frames):

| checkpoint | toggled | own chain mm | other hinge mm | rest mm | own-sign |
| --- | --- | ---: | ---: | ---: | ---: |
| L_HINGE | left_elbow | 26.75 | **0.000** | **0.000** | 0.687 |
| | right_elbow | 21.17 | **0.000** | **0.000** | 0.568 |
| | left_knee | 16.98 | **0.000** | **0.000** | 0.371 |
| | right_knee | 20.15 | **0.000** | **0.000** | 0.451 |
| L_NO_LEFT_KNEE | left_elbow | 26.87 | 0.000 | 0.000 | 0.629 |
| | right_elbow | 19.43 | 0.000 | 0.000 | 0.523 |
| | right_knee | 19.98 | 0.000 | 0.000 | 0.374 |
| L_NO_RIGHT_KNEE | left_elbow | 30.28 | 0.000 | 0.000 | 0.752 |
| | right_elbow | 25.08 | 0.000 | 0.000 | 0.647 |
| | left_knee | 15.16 | 0.000 | 0.000 | 0.378 |

**Off-target displacement is exactly `0.000 mm` for every toggle, on every
trained local checkpoint** — not merely reduced, architecturally
impossible by construction, and confirmed empirically on real trained
weights (not just the synthetic-input unit test). Own-sign (branch)
reliability is mixed against the corrected current-topology numbers:
right-elbow drops (0.771→0.568 for L_HINGE), left-knee is essentially
unchanged (0.369→0.371), right-knee improves (0.297→0.451).

## 13. Exact MPJPE/PA-MPJPE guardrail deltas

Against `S0`'s `81.98` mm MPJPE / `56.85` mm PA-MPJPE:

| candidate | MPJPE mm (Δ) | PA-MPJPE mm (Δ) |
| --- | ---: | ---: |
| L_NEUTRAL | 82.82 (+0.84) | 58.43 (+1.58) |
| L_HINGE | 84.95 (+2.97) | 58.67 (+1.82) |
| L_NO_LEFT_KNEE | 83.90 (+1.92) | 59.43 (+2.58) |
| L_NO_RIGHT_KNEE | 85.20 (+3.22) | 59.02 (+2.17) |
| L_NO_RIGHT_ELBOW | 85.04 (+3.06) | 60.06 (+3.21) |

Every local candidate costs more position error than its current-topology
counterpart (e.g. `O_HINGE` was `83.44`/`57.98`, `+1.46`/`+1.13`). This is
a real, non-trivial guardrail cost of removing the leakage pathway — not
disqualifying on its own (no candidate here was selected or rejected on
MPJPE), but it is the price observed for eliminating cross-joint hinge
leakage, and is reported plainly rather than absorbed into a vague "small
band" claim (docs/31's Section 3/11 wording correction applies the same
standard here).

## 14. Final statement

```
right_elbow_forward_bend
    current topology: NOT required (H_NO_RIGHT_ELBOW reproduces the benefit)
    local topology:   REQUIRED (L_NO_RIGHT_ELBOW loses the benefit entirely)
    -> topology-dependent in BOTH directions; no single answer without
       specifying which topology is in production

left_elbow_forward_bend
    UNRESOLVED under either topology on this evidence (docs/31's 47%
    retention under current topology; not separately re-tested under local
    topology this batch, since Section 15's decisive question was the knees)

left_knee_forward_bend, right_knee_forward_bend
    current topology: REQUIRED (removing either destroys the pooled benefit)
    local topology:   NOT required (removing either costs nothing measurable,
                       and even slightly exceeds the full four-field local
                       benefit)
    -> the knee requirement observed under the current topology is
       SUBSTANTIALLY TOPOLOGY-INDUCED, not primarily an information
       requirement
```

## 15. Whether any hinge field can now legitimately be called a future sensor requirement

**No.** Every field's apparent necessity in this lineage now depends on
which conditioning topology is assumed:

- The knees looked required, but that was substantially the current
  topology's cross-joint leakage, not the information itself — removing
  the leakage removes the apparent requirement.
- The right elbow looked dispensable under the current (leaky) topology,
  but becomes required under the local (non-leaky) one — the opposite
  direction of confound.

Neither topology's leave-one-out pattern can be treated as "the" sensor
contract. A future sensor requirement claim would need to fix which
topology the production model actually uses first, then re-run this exact
leave-one-out design on it — not assume the current-topology pattern
transfers, and not assume the local-topology pattern transfers either,
since it was only tested once, on one seed, with a weaker overall
hinge-conditioning effect and worse position guardrails.

## 16. Confirmation orientation/VLM tracks were untouched

No orientation candidate was retrained. No VLM (Qwen or otherwise) was
invoked. No prompt was modified. No sign classifier was trained. The
Geometry Core loss contract (`baseline_geometry_v1`) is unchanged. The
existing orientation interpretation (`hip_forward_depth` alone
approximately reproduces the S1 yaw outcome; the bilateral pair is the
strongest measured configuration; `torso_facing` adds no measured value)
and the Qwen verdict (`torso_facing` is image-grounded for the tested
model; bilateral near/far and hinge evidence are not usefully grounded)
are both carried forward unchanged from docs/31.

## 17. Tests and environments

New focused tests (`tests/test_frame_pose_signs.py`,
`tests/test_frame_pose_model.py`):

- an unknown checkpoint name is refused by the influence diagnostic, not
  defaulted;
- `active_fields_for` matches the training-time `CANDIDATES` contract
  exactly for `O_HINGE`/`S0`/`S1`/`O_LEFT_ELBOW`;
- `O_HINGE`'s masked baseline never carries oracle ±1 for the three
  orientation fields (the exact repaired bug), verified against the real
  bank fixture's oracle;
- a single-field checkpoint's masked baseline never carries oracle values
  for any field but its own;
- the primary diagnostic only toggles fields active for that checkpoint;
- an end-to-end (tiny, CPU) trained-checkpoint check that the repaired
  baseline/toggle logic is deterministic and self-describing;
- `hinge_sign_injection` rejects unknown values;
- `pre_attention` is the default and builds no new buffers (bit-identical
  historical construction);
- `pre_attention`/`post_attention` share one parameter count;
- a `post_attention` hinge toggle changes **exactly** its own routed
  joint's output and nothing else;
- a `post_attention` orientation-field toggle still propagates globally
  (orientation behaviour unchanged);
- a `pre_attention` hinge toggle is confirmed to leak to an unrelated
  joint, for contrast.

Full regression: **624 passed, 1 skipped** in the macOS authoring venv
(`.venv`, torch 2.13.0; the skip is the timm-gated backbone test, same as
docs/31). Training, the corrected influence diagnostic, and the local-hinge
training ran in `animcv-framepose:cuda118` on LabServer63. No remote CI
exists and none was claimed.

## Completion

**Answer to the completion question**: after evaluating every checkpoint
under the exact sign contract it was trained with, the two knee signs'
apparent necessity is **substantially a property of the current
conditioning topology, not of the information itself** — a topology that
architecturally cannot leak hinge signs across joints removes the knee
requirement (and even matches or exceeds the full-hinge-set benefit
without either knee), at the cost of a weaker overall hinge-conditioning
effect and worse position guardrails, and at the cost of *creating* a new
right-elbow requirement that did not exist under the leaky topology.

No sensor contract is chosen. No further architecture change is
implemented this batch.
