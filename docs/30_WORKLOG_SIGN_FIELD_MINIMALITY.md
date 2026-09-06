# Worklog — Sign-Field Minimality (2026-09-06)

> Attribution-only batch on `arch/single_frame_first`. No VLM was retried or
> replaced, no prompt touched, no sensor implemented, no pose quality optimised.
> Every earlier candidate and artifact is preserved; new runs live under
> `sign_minimality_v1`.
>
> The question: **what is the smallest discrete SignState that preserves the
> orientation and hinge benefits O_BILATERAL and O_HINGE already demonstrated?**

## 1. Branch

Start `9ffe034`. Final HEAD in Section 18.

## 2. Model-level exact SignState validation

`FramePoseEstimator.forward()` checked `-1 <= value <= +1` and then called
`.long()`. Two holes followed: `0.5` passed the range test and was **truncated**
to `0`, and `NaN`/`inf` passed an unordered comparison entirely.

The boundary now requires exact membership:

```
value == -1  or  value == 0  or  value == +1     -- everything else refused
```

No clamping, no rounding, no truncation. Tested directly against `0.5`, `-0.5`,
`0.999`, `NaN`, `+inf`, `-inf`, and against all three legal values.

## 3. Terminology used from here on

| Word | Meaning |
| --- | --- |
| **Sufficient** | the field or set reproduces the target benefit by itself |
| **Necessary under this architecture** | removing it from a sufficient set materially loses the benefit |
| **Useful** | supplying it improves its own targeted local metric |
| **Unresolved** | the effect is too small to classify from this single-seed controlled run |

docs/29 called individual fields "necessary" on the strength of the group
containing them having worked. That is not evidence, and this batch measures it
instead. docs/29 has been corrected.

## 4-6. Orientation: field-level attribution

Capacity-matched throughout — 1,657,603 trainable parameters, all seven
embedding tables present, same seed, frames, optimizer, loss, schedule,
evaluator and validation selection. Only the active field set differs.

### Test split (7,076 frames)

| candidate | fields | MPJPE | PA-MPJPE | yaw MAE | **yaw P95** | shoulder sign dis. | hip sign dis. |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| S0 neutral | 0 | 81.98 | 56.85 | 10.92 | **28.67** | 0.1115 | 0.0977 |
| **O_SHOULDER** | 1 | 79.35 | 55.95 | 9.04 | **21.35** | 0.0201 | 0.0373 |
| **O_HIP** | 1 | 80.42 | 56.82 | 8.26 | **19.53** | 0.0605 | 0.0089 |
| O_BILATERAL | 2 | 79.23 | 56.42 | 8.00 | **18.64** | 0.0323 | 0.0037 |
| S1 all-oracle | 7 | 79.20 | 55.96 | 8.23 | 19.70 | 0.0175 | 0.0026 |

Share of the S0 to O_BILATERAL yaw-P95 gap (10.03 deg) recovered by one field
alone:

```
O_SHOULDER   7.32 deg   73%
O_HIP        9.14 deg   91%
```

### Interaction classification: **C — largely redundant**

Neither single field is *necessary*: dropping the shoulder sign (O_HIP) still
recovers 91% of the pair's yaw-P95 benefit, and dropping the hip sign
(O_SHOULDER) still recovers 73%. The pair is best, but only marginally better
than the hip sign alone (18.64 vs 19.53).

The redundancy has a visible mechanism: each bilateral sign improves the
*other's* disagreement rate without being supplied. O_HIP cuts shoulder sign
disagreement 0.1115 to 0.0605, and O_SHOULDER cuts hip disagreement 0.0977 to
0.0373. Shoulder and hip near/far ordering are correlated through the torso, so
one largely determines the other.

Whether hip is genuinely better than shoulder (19.53 vs 21.35) is **unresolved**
on one seed. The gap from S0 is 7-9 deg; the gap between the two is 1.8 deg.

Position guardrail: MPJPE 79.35 / 80.42 / 79.23 against S0's 81.98 — all within
the ~1.5 mm band this lineage cannot resolve, and none degraded.

## 7-8. Hinge: family and half-family attribution

| candidate | fields | **hinge flip** | hinge MAE | elbow flip | knee flip | MPJPE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| S0 | 0 | 0.0212 | 24.82 | 0.0208 | 0.0198 | 81.98 |
| **O_ELBOWS** | 2 | 0.0230 | 25.43 | 0.0258 | 0.0210 | 82.39 |
| **O_KNEES** | 2 | 0.0229 | 25.09 | 0.0235 | 0.0208 | 82.85 |
| O_HINGE | 4 | **0.0159** | **23.44** | **0.0140** | **0.0157** | 83.44 |
| S1 | 7 | 0.0187 | 23.60 | 0.0182 | 0.0173 | 79.20 |

**Neither half reproduces the family benefit — both are slightly worse than
supplying nothing.** Only the complete set of four improves the pooled hinge
flip rate, and it does so more than the full seven-field S1 (0.0159 vs 0.0187).

## 9. Per-joint hinge attribution

| candidate | own-chain flip | S0's rate for that chain | own-chain change | pooled hinge flip |
| --- | ---: | ---: | ---: | ---: |
| O_LEFT_ELBOW | 0.0178 | 0.0232 | **-0.0054** | 0.0233 |
| O_RIGHT_ELBOW | 0.0109 | 0.0183 | **-0.0074** | 0.0211 |
| O_LEFT_KNEE | 0.0128 | 0.0202 | **-0.0074** | 0.0200 |
| O_RIGHT_KNEE | 0.0158 | 0.0173 | -0.0015 | 0.0240 |

**All four improve their own chain. None improves the pooled rate.** Three do so
clearly; O_RIGHT_KNEE's -0.0015 is third-decimal and **unresolved**.

## 10. Locality / ownership diagnostic

Per-joint MPJPE change against S0, for the joint the routing mask governs versus
the mean of all other joints:

| candidate | governed joint | delta governed | delta mean of others |
| --- | --- | ---: | ---: |
| O_LEFT_ELBOW | left_elbow | **+1.08** | -0.87 |
| O_RIGHT_ELBOW | right_elbow | **-1.76** | +1.03 |
| O_LEFT_KNEE | left_knee | **-3.48** | +0.04 |
| O_RIGHT_KNEE | right_knee | **-2.30** | +1.72 |
| O_SHOULDER | left_shoulder | -3.23 | -2.72 |
| O_HIP | left_hip | -2.46 | -1.34 |

Three of the four hinge fields put their largest position improvement on exactly
the joint their mask governs, while off-target joints move the other way — the
ownership the routing mask asserts is visible in the data. O_LEFT_ELBOW is the
exception: its own joint's position error *rises* even though its own chain's
flip rate falls, which is consistent with the field correcting the bend *branch*
without improving that joint's metric position. The bilateral fields are not
local by design and improve broadly, as expected.

This is the mechanism behind Section 7: **a partial hinge signal helps its own
chain and disturbs the others**, so halves lose more elsewhere than they gain
locally, and only the complete set improves everything at once.

## 11. Comparison across the whole lineage (test)

| | fields | MPJPE | yaw P95 | hinge flip |
| --- | ---: | ---: | ---: | ---: |
| S0 | 0 | 81.98 | 28.67 | 0.0212 |
| O_TORSO | 1 | 82.48 | 28.00 | 0.0240 |
| O_SHOULDER | 1 | 79.35 | 21.35 | 0.0192 |
| O_HIP | 1 | 80.42 | 19.53 | 0.0202 |
| O_BILATERAL | 2 | 79.23 | **18.64** | 0.0189 |
| O_ORIENTATION | 3 | 80.61 | 18.96 | 0.0220 |
| O_ELBOWS | 2 | 82.39 | 28.56 | 0.0230 |
| O_KNEES | 2 | 82.85 | 27.15 | 0.0229 |
| O_HINGE | 4 | 83.44 | 26.99 | **0.0159** |
| S1 | 7 | **79.20** | 19.70 | 0.0187 |

The two axes are cleanly separable: nothing in the hinge family moves yaw P95
below 27 deg, and nothing in the orientation family moves hinge flip below
0.0189.

## 12. Minimum orientation Sign Contract

```
SUFFICIENT
    one bilateral near/far sign
        hip_forward_depth        recovers 91% of the pair's yaw-P95 benefit
        shoulder_forward_depth   recovers 73%

    the pair together is best (18.64 deg) but only marginally beyond hip alone

NECESSARY
    neither individual bilateral field is necessary; they are largely redundant

NOT REQUIRED / NO MEASURED VALUE
    torso_facing
        alone: yaw P95 28.00 against S0's 28.67
        added to the pair: 18.96 against 18.64

UNRESOLVED
    whether hip is genuinely the stronger of the two (19.53 vs 21.35, one seed)
```

**A sensor that answers one bilateral near/far question — "which of the
performer's hips is nearer the camera?" — already captures most of the available
orientation benefit.** That is a materially smaller contract than docs/29
implied.

## 13. Minimum hinge Sign Contract

```
USEFUL
    each of the four hinge fields improves its own chain's flip rate
    (three clearly, O_RIGHT_KNEE unresolved)

SUFFICIENT
    only the complete set of four reproduces the family's pooled benefit
    (0.0159 against S0's 0.0212)

NOT SUFFICIENT
    elbows alone      0.0230   worse than supplying nothing
    knees alone       0.0229   worse than supplying nothing

NECESSARY
    UNRESOLVED for any individual field. A leave-one-out from the four-field
    set was not run, and the halves being worse than S0 means partial sets
    cannot be read as evidence about individual fields.
```

**PROVISIONAL — corrected in docs/31.** This section originally read "the hinge
contract does not reduce; a sensor must answer all four bend questions or none".
That is stronger than the evidence here supports, for two reasons:

- the direct 3-of-4 leave-one-out was not run, so no individual field had been
  shown necessary even under this architecture;
- the observed pattern (correct local sign improves its own chain, unrelated
  chains regress, pooled metric does not improve) is equally consistent with the
  *conditioning topology* leaking a local sign through global joint
  self-attention as with a genuine joint information requirement.

Read this section as:

```
ESTABLISHED
    the all-four set is sufficient for the measured pooled hinge benefit
    elbow-only and knee-only halves are not sufficient
    several individual fields are locally useful

UNRESOLVED
    whether every one of the four fields is necessary
    whether partial-sign failure reflects information necessity or
    conditioning-topology leakage
```

docs/31 separates those two explanations.

## 14. Shuffled-image donor identity audit

No VLM inference. From the stored records and the bank:

| quantity | rate |
| --- | ---: |
| donor is the same sample | 0.000 |
| donor is the same `sequence_id` (sequence + actor) | 0.000 |
| donor is the same sequence | 0.000 |
| **donor is a different sequence** | **1.000** |

So the control did swap in a crop from a different 3DPW sequence. But 3DPW
performers recur across sequences and performer identity is not representable
from the bank, so **"another person's crop" overstates it**; "a crop from a
different sequence" is what was actually controlled. docs/29 is corrected.

## 15. Same-branch vs opposite-branch donor grounding

An unconditioned change rate is weak evidence: a grounded model *should* answer
the same when the donor happens to carry the same branch. Stratifying by the
donor's own oracle sign is the discriminating measurement.

Prediction change rate under shuffling:

| field | mode | donor same branch | **donor opposite branch** | donor degenerate |
| --- | --- | ---: | ---: | ---: |
| `torso_facing` | combined | 0.333 | **0.760** | 0.500 |
| `torso_facing` | isolated | 0.143 | **0.280** | 0.250 |
| all six others | combined | 0.000-0.100 | **0.000** | 0.000-0.167 |
| all six others | isolated | 0.000 | **0.000** | 0.000 |

This **strengthens** docs/29's conclusion rather than softening it. For six of
seven fields the answer does not change *even when the substituted crop carries
the opposite true branch* — a change rate of exactly 0.000 under the one
condition where a grounded model must change. And `torso_facing` behaves exactly
as a grounded answer should: it changes far more often when the donor's branch
is opposite (0.760) than when it agrees (0.333).

## 16. Corrections to docs/29

- "another person's crop" / "completely different subject" becomes "a crop from
  a different sequence" (Section 14).
- The minimum-sign section no longer calls individual fields "necessary"; it
  states sufficiency at the group level and points here for the field-level
  result.
- The non-grounding finding is annotated with the opposite-branch conditioning.

The Qwen verdict itself is unchanged, and no new Qwen score was produced.

## 17. Artifacts

```
sign_minimality_v1/                     eight new capacity-matched candidates
sign_shuffle_audit_v1/                  donor identity + conditioned grounding
sign_recomputed_v1.json                 per-chain metrics for the earlier candidates,
                                        recomputed from stored predictions, originals untouched
```

Preserved unchanged: `sign_v1`, `sign_attr_v2`,
`sign_attr_v1_INVALID_mask_discarded`, `sign_diag_v1`, `sign_conformance_v1`,
`sign_review_v1`, `sign_bank_v1`, and the dense F0/F1/F2 lineage.

## 18. Tests and environments

New focused tests: exact model-level membership against `0.5`, `-0.5`, `0.999`,
`NaN`, `+/-inf` and the three legal values; per-chain hinge metrics asserted by
flipping one elbow and checking the other chains stay clean; candidate-registry
checks that every single-field candidate activates exactly one field and that
the halves partition their family.

Full regression: **608 passed, 1 skipped** in the macOS authoring venv (`.venv`,
torch 2.13.0; the skip is the timm-gated backbone test). Attribution ran in
`animcv-framepose:cuda118` on LabServer63. No VLM inference was required. No
remote CI exists and none was claimed.

## 19. Answer to the completion question

**The smallest sign output contract the next AnimCV sign sensor must satisfy:**

```
ORIENTATION   one bilateral near/far sign is enough to capture most of the
              benefit; hip appears at least as good as shoulder; both is
              marginally better. torso_facing is not required.

HINGE         PROVISIONAL. The all-four set is sufficient and the halves are
              not; whether all four are *required* — and whether the
              all-or-none shape is a property of the information or of the
              current conditioning topology — is unresolved here and is
              settled in docs/31.
```

No individual field is claimed necessary on group evidence, and every
model-level SignState outside `{-1, 0, +1}` is now refused.

Stopping here. The next sensor architecture is not chosen or implemented in this
batch.
