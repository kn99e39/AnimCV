# Worklog — Hinge Sign Necessity and Conditioning Locality (2026-09-07)

> Diagnostic/attribution batch on `arch/single_frame_first`. No VLM, no prompt,
> no sensor, no orientation retraining, no Geometry Core change. Every earlier
> candidate and artifact preserved; new runs under `sign_hinge_loo_v1` and
> `sign_influence_v1`.
>
> The question: **does the apparent all-four hinge requirement belong to the
> information, or to the current sign-conditioning topology?**

## 1. Branch

Start `f805288`. Final HEAD in Section 14.

## 2. docs/30 hinge wording, corrected before new evidence

docs/30 said "the hinge contract does not reduce" and "a sensor must answer all
four bend questions or none". Both are now marked **PROVISIONAL** in that
document, because the direct 3-of-4 leave-one-out had never been run and the
observed pattern was equally consistent with conditioning-topology leakage. The
established/unresolved split is written into docs/30 §13 and its Section 19
summary. No historical number was altered.

This batch shows the strong wording was **wrong**.

## 3. Four 3-of-4 leave-one-out results

Capacity-matched: 1,657,603 trainable parameters, all seven embedding tables,
seed 1337, same bank, optimizer, loss, schedule, evaluator and validation
selection. Only which single hinge field is removed from the successful O_HINGE
set differs; every inactive field is UNKNOWN.

### Test split (7,076 frames)

| candidate | fields | **hinge flip** | hinge MAE | L elbow | R elbow | L knee | R knee | MPJPE | PA-MPJPE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| S0 neutral | 0 | 0.0212 | 24.82 | 0.0232 | 0.0183 | 0.0202 | 0.0173 | 81.98 | 56.85 |
| O_ELBOWS | 2 | 0.0230 | 25.43 | — | — | — | — | 82.39 | 58.21 |
| O_KNEES | 2 | 0.0229 | 25.09 | — | — | — | — | 82.85 | 58.46 |
| **H_NO_LEFT_ELBOW** | 3 | 0.0187 | 24.01 | 0.0230 | 0.0165 | 0.0126 | 0.0190 | 81.47 | 57.33 |
| **H_NO_RIGHT_ELBOW** | 3 | **0.0159** | 23.52 | 0.0160 | 0.0130 | 0.0126 | 0.0178 | 82.47 | 57.12 |
| **H_NO_LEFT_KNEE** | 3 | 0.0208 | 25.35 | 0.0237 | 0.0193 | 0.0155 | 0.0209 | 83.84 | 59.11 |
| **H_NO_RIGHT_KNEE** | 3 | 0.0222 | 24.60 | 0.0268 | 0.0132 | 0.0171 | 0.0253 | 81.73 | 57.11 |
| O_HINGE | 4 | **0.0159** | **23.44** | 0.0173 | 0.0115 | 0.0151 | 0.0143 | 83.44 | 57.98 |
| S1 all-oracle | 7 | 0.0187 | 23.60 | 0.0237 | 0.0128 | 0.0169 | 0.0160 | 79.20 | 55.96 |

## 4. Per-field necessity under the current topology

Share of the S0 to O_HINGE pooled flip-rate benefit (0.0212 to 0.0159, a gap of
0.0053) that survives when one field is removed:

| removed field | hinge flip | benefit retained | hinge MAE retained | verdict |
| --- | ---: | ---: | ---: | --- |
| `right_elbow_forward_bend` | 0.0159 | **100%** | 94% | **NOT NECESSARY** under current topology |
| `left_elbow_forward_bend` | 0.0187 | 47% | 59% | partially necessary |
| `left_knee_forward_bend` | 0.0208 | 8% | −38% | **NECESSARY** under current topology |
| `right_knee_forward_bend` | 0.0222 | −19% | 16% | **NECESSARY** under current topology |

**The all-or-none claim is refuted.** A three-field set — both knees plus the
left elbow — reproduces the four-field pooled benefit exactly (0.0159 vs 0.0159,
hinge MAE 23.52 vs 23.44). The right-elbow sign contributes nothing measurable
to the pooled result once the other three are present.

**Wording correction (docs/32 Section 3)**: this batch previously claimed
every leave-one-out position delta sat inside an "~1.5 mm band". That
overstates the resolution. Exact S0-relative deltas (S0: MPJPE 81.98,
PA-MPJPE 56.85):

| candidate | ΔMPJPE mm | ΔPA-MPJPE mm |
| --- | ---: | ---: |
| H_NO_LEFT_ELBOW | −0.51 | +0.48 |
| H_NO_RIGHT_ELBOW | +0.49 | +0.27 |
| **H_NO_LEFT_KNEE** | **+1.86** | **+2.26** |
| H_NO_RIGHT_KNEE | −0.25 | +0.26 |
| O_HINGE | +1.46 | +1.13 |
| S1 | −2.78 | −0.89 |

`H_NO_LEFT_KNEE`'s PA-MPJPE delta (+2.26 mm) and O_HINGE's own PA-MPJPE
delta (+1.13 mm) both exceed the previously claimed band. These are the
exact numbers; no threshold is asserted and MPJPE/PA-MPJPE remain
guardrails, not a promotion criterion — no candidate here was selected or
rejected on them.

## 5. Frozen-weight sign-to-joint influence

> **PROVISIONAL — repaired in docs/32.** The probe below fed every checkpoint
> the full 7-field oracle as its baseline, including for O_HINGE, which was
> never trained with `torso_facing`/`shoulder_forward_depth`/
> `hip_forward_depth` active (they were UNKNOWN throughout training). That
> evaluates O_HINGE in an out-of-distribution sign state. **Do not use the
> locality numbers in Sections 5–8 below to close the knee mechanism** — see
> docs/32 for the corrected, in-distribution-only probe and its mechanism
> reassessment. Sections 1–4 above (the leave-one-out training result itself)
> are unaffected and remain valid as stated.

Weights frozen, geometry fixed, exactly one hinge sign toggled from its oracle
value to the opposite branch, 1,500 test frames. The skeleton is partitioned
into the toggled field's own chain, the other three hinge chains, and the rest.

Mean per-joint displacement, millimetres:

| checkpoint | toggled sign | own chain | other hinge chains | rest | own/other |
| --- | --- | ---: | ---: | ---: | ---: |
| O_HINGE | left_elbow | 58.5 | 20.9 | 20.2 | 2.80 |
| O_HINGE | right_elbow | 53.3 | 15.2 | 19.5 | 3.51 |
| O_HINGE | left_knee | 70.9 | **47.8** | 28.0 | **1.48** |
| O_HINGE | right_knee | 50.0 | 28.3 | 17.7 | 1.77 |
| S1 | left_elbow | 50.9 | 14.6 | 10.2 | 3.47 |
| S1 | right_elbow | 49.6 | 15.0 | 12.4 | 3.30 |
| S1 | left_knee | 44.1 | **32.8** | 21.6 | **1.34** |
| S1 | right_knee | 45.1 | 24.8 | 13.2 | 1.82 |

Displacement distribution under O_HINGE (median / p90 / p99, mm):

| toggled sign | own chain | other hinge chains |
| --- | --- | --- |
| left_elbow | 42.0 / 123.4 / 175.0 | 16.3 / 42.2 / 86.9 |
| right_elbow | 42.0 / 111.2 / 173.7 | 11.3 / 30.4 / 63.8 |
| left_knee | 55.5 / 146.1 / 267.4 | **39.8 / 86.8 / 161.2** |
| right_knee | 41.6 / 97.0 / 166.5 | 24.0 / 51.0 / 88.1 |

Single-field checkpoints behave consistently: under `O_LEFT_ELBOW`, toggling the
field it was trained with moves its own chain 65.0 mm (own/other 1.94), while
toggling a field it was never trained with still moves the pose 9–13 mm.

## 6. Reconstructed-sign change matrix

The same toggles, measured on the *discrete branch* read back out of the
predicted pose — how often each hinge chain's reconstructed sign changes:

**O_HINGE checkpoint**

| toggled | → L elbow | → R elbow | → L knee | → R knee |
| --- | ---: | ---: | ---: | ---: |
| left_elbow | **0.809** | 0.027 | 0.017 | 0.032 |
| right_elbow | 0.023 | **0.833** | 0.013 | 0.004 |
| left_knee | 0.048 | 0.068 | **0.572** | 0.076 |
| right_knee | 0.037 | 0.035 | 0.026 | **0.184** |

**S1 checkpoint** shows the same shape (diagonal 0.696 / 0.708 / 0.345 / 0.504,
off-diagonal 0.017–0.084).

Two things stand out. The diagonal dominates by one to two orders of magnitude,
so a hinge sign largely owns its own *branch*. And the knee diagonals are weak —
toggling the right-knee sign changes the reconstructed right-knee branch only
18% of the time under O_HINGE — so the knee signs are the ones the model acts on
least reliably.

## 7. Is cross-joint leakage present?

**Qualitatively, yes — at the position level.** A single toggled bit displaces
unrelated hinge chains by a median of 11–40 mm and a p90 of 30–87 mm. That is
systematic, not floating-point noise, and for the left knee the off-target
displacement is two-thirds of the on-target displacement.

**At the sign level, no.** Cross-field reconstructed-sign changes are 0.4–8.4%
against diagonals of 18–83%. The discrete branch decisions stay local.

The raw distributions are exported in `sign_influence_v1/sign_influence.json`;
no threshold was invented after seeing them.

## 8. Mechanism classification: **D (mixed), with C established**

Against the batch's own table:

**C is established.** `H_NO_RIGHT_ELBOW` retains the full O_HINGE benefit, so at
least one field is dispensable and the four-field sensor contract is
unnecessarily large. This is a direct measurement, not an inference.

**B is not cleanly supported as the explanation.** Leakage is qualitatively
present at the position level, but the decisive counter-evidence is that
*removing a field does not uniformly hurt*. If partial evidence were harmful
because a local sign leaks globally, every 3-of-4 candidate should degrade.
`H_NO_RIGHT_ELBOW` does not degrade at all. And the discrete branch decisions —
the thing the Sign Contract actually carries — are 92–99% local.

**A cannot be ruled out for the knees.** Removing either knee sign destroys the
pooled benefit (8% and −19% retained). But the knees are also exactly where
locality is worst (own/other ratio 1.48 and 1.77, the largest off-target
displacement, the weakest diagonal sign response). Genuine joint information
requirement and conditioning leakage are **confounded for the knees** on this
evidence, and one seed cannot separate them.

So: the all-or-none shape is **not** a property of the information. Part of the
four-field set is dispensable outright; the knee part of it is unresolved
between A and B.

## 9. The conditional local-conditioning candidate was **not implemented**

Section 9 of the direction gates it on outcome B being supported. It is not:
sign-level locality is already high (92–99%), and the single strongest fact —
that dropping the right-elbow sign costs nothing — is inconsistent with leakage
being what makes partial evidence harmful.

Implementing a locality-preserving topology now would be building a fix for a
mechanism the evidence does not establish. The measurements that would justify
it, and the one that argues against it, are both recorded above so the decision
can be revisited.

## 10. Not applicable

No local candidate was implemented, so there is no architecture difference,
parameter count, or locality accounting to report for it.

## 11. Position guardrails

No candidate in this batch was selected or tuned on MPJPE. Across the four
leave-one-out runs MPJPE spans 81.47–83.84 mm against S0's 81.98 and O_HINGE's
83.44, and PA-MPJPE spans 57.11–59.11 against S0's 56.85. **Wording correction
(docs/32 Section 3)**: this is not a claim that every delta is inside one small
band — see Section 4's exact per-candidate delta table, where `H_NO_LEFT_KNEE`
(+2.26 mm PA-MPJPE) and O_HINGE (+1.13 mm PA-MPJPE) are the largest. No
candidate degraded catastrophically, and none was chosen or rejected on this
metric, but the deltas are not uniformly small.

## 12. Hinge sensor requirement — verdict

```
ESTABLISHED
    right_elbow_forward_bend is NOT required
        removing it from the four-field set costs nothing measurable
        (pooled flip 0.0159 either way)

    left_knee_forward_bend and right_knee_forward_bend are required
        under the current conditioning topology
        removing either destroys the pooled benefit

    the all-four contract is unnecessarily large
        three fields -- both knees plus the left elbow -- reproduce it exactly

    a partial set can still be worse than none
        elbow-only 0.0230 and knee-only 0.0229 against S0's 0.0212

UNRESOLVED
    left_elbow_forward_bend
        removing it retains 47% of the benefit; neither clearly required nor
        clearly dispensable on one seed

    whether the knees' necessity is an information property or an artifact of
    their poor conditioning locality
        they are simultaneously the most necessary and the least local fields,
        and this lineage cannot separate the two
```

**Wording correction (docs/32 Section 2)**: the sentence originally here read
"a future hinge sign sensor must answer at least the two knee questions." That
overstates what this batch's evidence supports, because the frozen
sign-influence probe behind the locality half of that claim had a contract
mismatch (Section 8 above; repaired in docs/32). The corrected statement is:

**If the current conditioning topology is preserved, both knee signs are
required by the current single-seed leave-one-out result. Whether they are
requirements of the information itself remains unresolved.** The right-elbow
question can still be dropped — that conclusion rests on the leave-one-out
training result alone, not on the influence probe, and is unaffected. Whether
the left-elbow question is needed, and whether a locality-preserving
conditioning topology would shrink the knee requirement further, are open;
see docs/32 for the corrected attribution.

## 13. Untouched tracks

Orientation was not retrained and its interpretation is unchanged:
`hip_forward_depth` alone approximately reproduces the S1 yaw outcome,
`shoulder_forward_depth` is independently useful, the bilateral pair is the
strongest measured configuration, and `torso_facing` adds no measured value.

The Qwen verdict is unchanged and no VLM inference was run: `torso_facing` is
image-grounded for the tested model, the required bilateral near/far evidence is
not usefully grounded, and hinge evidence is not usefully grounded under the
tested setup.

## 14. Artifacts, tests, environments

```
sign_hinge_loo_v1/        four 3-of-4 leave-one-out candidates
sign_influence_v1/        frozen-weight sign-to-joint influence matrix
```

Preserved unchanged: `sign_v1`, `sign_attr_v2`,
`sign_attr_v1_INVALID_mask_discarded`, `sign_minimality_v1`, `sign_diag_v1`,
`sign_conformance_v1`, `sign_review_v1`, `sign_bank_v1`, `sign_shuffle_audit_v1`,
and the dense F0/F1/F2 lineage.

New focused tests: the four 3-of-4 masks remove exactly one field and stay
within the hinge family; leave-one-out candidates are capacity-matched, keep the
removed field UNKNOWN, and reproduce the full oracle's distribution for the
fields they retain; the influence diagnostic's joint groups partition the
skeleton without overlap and route to the joint the Sign Contract's mask
governs; a frozen sign toggle is deterministic and a toggled sign does change
the output.

Full regression: **612 passed, 1 skipped** in the macOS authoring venv (`.venv`,
torch 2.13.0; the skip is the timm-gated backbone test). Training and the frozen
probe ran in `animcv-framepose:cuda118` on LabServer63. No remote CI exists and
none was claimed.

## 15. Answer to the completion question

**The apparent all-four hinge requirement belongs to neither explanation
cleanly, and it is not a property of the information.**

One field is measurably dispensable, so the requirement is smaller than claimed.
The knees are genuinely required under this topology, but they are also the
fields whose conditioning is least local, so their necessity cannot be
attributed to the information alone from this evidence.

Stopping here. No sensor is chosen and no conditioning redesign is implemented.
