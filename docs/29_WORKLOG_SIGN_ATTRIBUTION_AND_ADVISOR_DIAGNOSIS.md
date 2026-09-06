# Worklog — Sign-Group Attribution and Advisor Failure Diagnosis (2026-09-06)

> Controlled attribution batch on `arch/single_frame_first`. No VLM was
> replaced, no prompt was tuned after seeing a result, the Geometry Core was not
> touched, and no new downstream S2 was produced. Historical S0/S1/S2 and the
> dense F0/F1/F2 results are preserved unchanged.
>
> Two questions: **which** signs AnimCV actually needs, and **why** the tested
> advisor answered a constant branch.

## 1. Branch

Start `e46e39c`. All new runs carry new experiment identifiers; nothing
historical was regenerated under stronger provenance and reused under its old
name.

## 2. Sign Contract validation hardening

The model previously mapped `sign_state + 1` through `clamp(0, 2)`, so an
out-of-domain value silently became a legal embedding — a broken advisor or a
corrupt bank would have produced a *confident wrong branch*.

One canonical path now validates every sign array
(`signs.validate_sign_array`): dtype and integrality, shape `(n, 7)`, optional
row count, and membership of `{-1, 0, +1}`. Violations raise. The clamp is gone
from the model, which now refuses an out-of-domain state outright, and the
advisor-bank loader validates on the way in.

```
invalid SignState -> refused          (never guessed, never clamped)
```

## 3. Strict parser hardening

The documented contract — exactly one JSON object, no surrounding prose, exactly
the contract fields — is now what the code enforces. Rejected: prose before or
after the object, two objects, extra keys, missing keys, non-string answers,
unrecognised categorical values, empty replies, non-object JSON. Every rejection
falls back to the all-`UNKNOWN` state.

This immediately changed a historical claim; see Section 15.

## 4. Test tautologies removed

Both `or True` assertions in the Sign Advisor surface are gone.

- the invalid-joint test now asserts that an invalid joint degrades **only** its
  dependent fields, and that an unaffected field keeps its value;
- the dense-visual-token test checks real symbols (`timm`, `visual_dim`,
  `image_tokens`, `FrozenVisualBackbone`, `load_feature_cache`) and that the
  contract *declares* its exclusions rather than merely omitting them.

`grep -rn "or True" tests/ src/ scripts/` returns nothing.

## 5. Advisor model provenance

New runs record repository id, requested revision, resolved snapshot commit,
transformers and torch versions, dtype, parameter count, the full generation
config, and a weight manifest (ordered filename, byte size, SHA-256) hashed into
one fingerprint.

## 6. Historical S2 provenance — **recovered**

```
model                Qwen/Qwen2-VL-2B-Instruct
resolved commit      895c3a49bc3fa70a340399125c650a463535e71c
weight fingerprint   840bc66b30f80632ade2f1fd6f415c34bb2f89ff0460ecee98a572c8d77eabc0
files                11
status               recovered_from_local_snapshot
```

Argument: the historical run loaded with `revision=None`, which resolves to
`refs/main`; the cache holds exactly one snapshot for this id and `refs/main`
points at it. Written as an **addendum** beside the historical bank
(`sign_bank_v1/provenance_addendum.json`); the artifact itself is untouched. Had
the snapshot been ambiguous the addendum would have said
`exact_weight_fingerprint_not_established` rather than naming a revision.

## 7. Oracle evidence wording, corrected

docs/28 called yaw and hinge metrics "non-circular". The accurate three-tier
framing, now used everywhere and declared in the runner's
`comparison_semantics.evidence_tiers`:

| Tier | Metrics | What a change proves |
| --- | --- | --- |
| **Direct / contract-identical** | shoulder and hip forward-depth sign agreement | the model *uses* the fed input |
| **Structurally coupled downstream** | root-yaw MAE/P95, hinge flip rate, hinge direction MAE | the fed branch propagates into quantities derived from the same geometry — coupled, not independent |
| **Independent position guardrails** | MPJPE, PA-MPJPE, per-joint error | continuous reconstruction was not damaged |

The Oracle result stands. What it proves is narrower than "independent
improvement": correct branch information changes the intended branch and does
not damage continuous reconstruction.

## 8. A bug that invalidated the first attribution run

The first attribution run (`sign_attr_v1`) is **void**. `train_candidate`
re-derived signs from `sign_source` whenever the source was not `"advisor"`,
discarding the field masks the candidates pass in. Every `O_*` candidate
therefore **trained on the full oracle** and was then **evaluated with only its
own group active** — a train/evaluate mismatch.

Detected before reporting, from three signatures: all four candidates recorded
identical sign distributions, all four selected epoch 109 with validation MPJPE
68.38 (S1's exact values), and O_TORSO scored 103.52 mm MPJPE against S0's 81.98
— correct information appearing actively harmful, which is not a physically
sensible result.

Fixed: an explicitly supplied array is always honoured and only validated; the
source is consulted only when no array is given. A regression test asserts a
torso-masked run records every other field as fully degenerate, and that an
unmasked run does not, so the check has teeth. The void artifacts are retained
as `sign_attr_v1_INVALID_mask_discarded`; the corrected run is `sign_attr_v2`.

## 9–12. Attribution results

All candidates are capacity-matched at **1,657,603 trainable parameters**, same
graph, all seven embedding tables present, same seed, frames, optimizer, loss,
evaluator and schedule. Only which fields carry oracle values differs; every
other field is `UNKNOWN`.

### Test split (7,076 frames)

| candidate | fields | MPJPE | PA-MPJPE | yaw MAE | **yaw P95** | **hinge flip** | hinge MAE | sh sign dis. | hip sign dis. |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| S0 neutral | 0 | 81.98 | 56.85 | 10.92 | 28.67 | 0.0212 | 24.82 | 0.1115 | 0.0977 |
| **O_TORSO** | 1 | 82.48 | 57.99 | 11.07 | **28.00** | 0.0240 | 26.08 | 0.1236 | 0.0885 |
| **O_BILATERAL** | 2 | **79.23** | 56.42 | **8.00** | **18.64** | 0.0189 | 24.10 | 0.0323 | 0.0037 |
| **O_HINGE** | 4 | 83.44 | 57.98 | 10.70 | 26.99 | **0.0159** | **23.44** | 0.1196 | 0.0927 |
| **O_ORIENTATION** | 3 | 80.61 | 56.49 | 8.03 | 18.96 | 0.0220 | 25.88 | 0.0186 | 0.0045 |
| S1 all-oracle | 7 | 79.20 | 55.96 | 8.23 | 19.70 | 0.0187 | 23.60 | 0.0175 | 0.0026 |

### What each group buys

**O_TORSO — the front/back sign alone is worth essentially nothing.** yaw P95
28.00 against S0's 28.67, MPJPE 82.48 against 81.98. A single global facing
branch does not carry the orientation improvement.

**O_BILATERAL — two fields reproduce the entire S1 benefit.** yaw P95 **18.64**,
better than all-oracle S1's 19.70; MPJPE 79.23 against S1's 79.20. The bilateral
near/far signs are where the orientation correction actually lives.

**O_HINGE — a separate, first-class requirement.** It gives the best hinge flip
rate (0.0159, below S1's 0.0187) and the best hinge direction MAE (23.44), and
it moves yaw essentially not at all (26.99). Conversely O_BILATERAL barely moves
hinge flip (0.0189 vs 0.0212). The two failures are separable and each needs its
own sign.

**O_ORIENTATION — facing adds nothing on top of bilateral.** 18.96 vs
O_BILATERAL's 18.64 yaw P95; 80.61 vs 79.23 MPJPE. There is **no complementary
benefit** from combining `torso_facing` with the bilateral pair; if anything it
is marginally worse.

### Limit on reading small differences

One run per candidate, no seed replicates. MPJPE differences of ~1–1.5 mm and
hinge-flip differences in the third decimal are **not** distinguishable from
run-to-run variation here. The conclusions above rest on the large effects: yaw
P95 spanning 18.6–28.7° and hinge flip spanning 0.0159–0.0240.

## 13. Minimum sign-channel conclusion

```
sufficient for the orientation/yaw failure (as a pair)
    shoulder_forward_depth
    hip_forward_depth

useful as a family for the hinge failure
    the four elbow/knee forward-bend signs

not required to reproduce the bilateral pair's measured orientation benefit
    torso_facing
```

*Corrected in docs/30.* This section originally called the individual fields
"necessary". Group membership is not evidence of individual necessity, and the
leave-one-field-out candidates that would establish it had not been run. docs/30
runs them and states the result in the sufficient / necessary / useful /
unresolved vocabulary.

A sign sensor for AnimCV therefore needs to answer **bilateral near/far
ordering** first, and hinge near/far second. The global "is the person facing
the camera" question — the intuitive one, and the one easiest to ask — is the
one the Geometry Core least needs. The exact per-field contract is settled in
docs/30.

## 14. Diagnostic subset composition

Deterministic stride over validation + test, 100 frames, rejected outright
unless every field has both branches present.

| field | +1 | −1 | degenerate |
| --- | ---: | ---: | ---: |
| `torso_facing` | 37 | 59 | 4 |
| `shoulder_forward_depth` | 49 | 44 | 7 |
| `hip_forward_depth` | 48 | 40 | 12 |
| `left_elbow_forward_bend` | 50 | 26 | 24 |
| `right_elbow_forward_bend` | 50 | 27 | 23 |
| `left_knee_forward_bend` | 32 | 60 | 8 |
| `right_knee_forward_bend` | 29 | 58 | 13 |

Majority-class baselines run 0.527–0.667, so no constant answer scores well.

## 15. Schema conformance — a finding in its own right

Qwen wraps every reply in a ```` ```json ```` fence. Under the strict contract:

| mode | strict-valid | requests | rate |
| --- | ---: | ---: | ---: |
| combined (real / shuffled) | 0 / 0 | 100 / 100 | **0.000** |
| isolated (real / shuffled) | 459 / 459 | 700 / 700 | 0.656 |

All 1,600 replies were leniently recoverable, so **content and conformance are
different failures** and are reported apart. Only content is comparable across
prompt modes.

This corrects docs/28's "0 malformed out of 1,200": that was measured with the
lenient parser then in force. Under the contract docs/28 itself stated,
combined-mode conformance is 0%.

## 16. Seven-question vs isolated-question (content, lenient recovery)

Accuracy / balanced accuracy against the majority-class baseline:

| field | major | **combined** acc / bal | **isolated** acc / bal |
| --- | ---: | --- | --- |
| `torso_facing` | 0.615 | **0.708 / 0.753** | 0.146 / 0.189 |
| `shoulder_forward_depth` | 0.527 | 0.516 / 0.490 | 0.527 / 0.500 |
| `hip_forward_depth` | 0.545 | 0.534 / 0.490 | 0.545 / 0.500 |
| `left_elbow_forward_bend` | 0.658 | 0.645 / 0.490 | 0.342 / 0.500 |
| `right_elbow_forward_bend` | 0.649 | 0.636 / 0.490 | 0.351 / 0.500 |
| `left_knee_forward_bend` | 0.652 | 0.348 / 0.500 | 0.652 / 0.500 |
| `right_knee_forward_bend` | 0.667 | 0.333 / 0.500 | 0.667 / 0.500 |

Balanced accuracy of exactly 0.500 with a one-sided prediction count is the
signature of a constant answer; an accuracy that happens to equal the majority
baseline (isolated knees: 0.652 = 0.652) is that same constant coinciding with
the majority class, not competence.

**Isolation did not help — it hurt.** It changed *which* constant (elbows and
knees flip from `+1` in combined to `−1` in isolated) and it destroyed the one
field that worked: `torso_facing` collapses from 0.708/0.753 to 0.146/0.189,
answering `unclear` on 85 of 100 frames.

## 17. Real vs shuffled image, and per-field output entropy

Prediction change rate when the correct crop is replaced by a crop **from a
different sequence**, with the question and oracle label kept:

*(Corrected in docs/30: the pairing rule guarantees a different sample, and an
audit confirms every donor came from a different 3DPW sequence — but performer
identity across sequences is not representable from the bank, so "another
person's crop" overstates what was controlled. Read it as "a different
sequence's crop".)*

| field | combined | isolated |
| --- | ---: | ---: |
| `torso_facing` | **0.560** | 0.220 |
| every other field | 0.040 | **0.000** |

And `torso_facing` accuracy under shuffle collapses: combined 0.708 → **0.396**
(balanced 0.753 → 0.423). Marginal answer counts are preserved under shuffling
(62/36 both ways) while per-frame alignment with the oracle is destroyed — which
is exactly the signature of a genuinely image-dependent answer.

Output entropy per field in isolated mode: `hip_forward_depth` 0.000 bits,
`shoulder_forward_depth` 0.081, `left_knee` 0.141 — near-zero information. The
elbows and right knee sit near 1.0 bit, but with balanced accuracy 0.500 and a
0.46–0.48 change rate under shuffle, that variation is uncorrelated with the
subject.

## 18. Qualitative review exports

`sign_review_v1/` — 4 frames per field (28 crops as PNG) with `sample_id`, the
image-content digest, the shuffled donor's id, the plain-language definition of
the field, what `+1` and `−1` mean, the oracle sign, the combined answer, the
isolated answer, the shuffled-image answer, and the raw reply.

These exist so a person can judge whether the oracle label matches what is
visible. This batch does **not** claim the visual semantics are correct from the
numeric GT mapping alone.

## 19. Why the advisor collapsed — verdict

Against the decision table:

- **Not prompt multiplexing (case A).** Isolation did not improve any field and
  actively destroyed `torso_facing`. Prompt-mode changed which constant was
  emitted, not whether it was grounded.
- **Visual non-grounding (case B) for six of seven fields.** Both bilateral
  fields, both elbows and both knees are constant, with 0.000–0.040 change rate
  under a crop from a different sequence, entropy near zero where it is
  one-sided, and balanced accuracy pinned at 0.500. The model is filling the
  schema, not looking. *(docs/30 sharpens this: conditioned on the donor
  carrying the **opposite** true branch, the change rate for all six fields is
  exactly 0.000.)*
- **Case D for the elbows and right knee specifically.** They vary at
  ~0.46–0.48 under shuffle — chance-level flipping for a binary answer — so
  their variation is not evidence of frame grounding either.
- **`torso_facing` is the single exception, and it is real.** 0.708 accuracy
  against a 0.615 baseline, balanced accuracy 0.753, and a 0.560 change rate
  that destroys accuracy under shuffle. Qwen2-VL-2B *can* read camera-relative
  body facing from a single crop.
- **Semantic convention mismatch (case C) is not supported.** In docs/28 I read
  a 30-frame probe as a systematic left/right inversion; at this scale the
  inverted reading is no better than the direct one, and the constant answers
  make the question moot.

**The decisive finding is the intersection with Section 13.** The one sign this
VLM can supply — `torso_facing` — is the one sign the attribution shows AnimCV
does **not** need. The two signs AnimCV actually needs — shoulder and hip
bilateral near/far — are exactly the two the VLM answers with a constant.

That is a specific, actionable negative result, and the batch stops here as
instructed. It does not license trying a larger VLM, rewriting the prompt,
training a classifier or running another S2; it does say what any future sign
sensor has to be measured on.

## 20. Artifacts, tests, environments

```
sign_attr_v1_INVALID_mask_discarded/   void run, retained and labelled
sign_attr_v2/                          corrected attribution, O_TORSO/O_BILATERAL/O_HINGE/O_ORIENTATION
sign_bank_v1/provenance_addendum.json  recovered historical S2 weight provenance
sign_diag_v1/diagnostic.json           strict-parse metrics, prompts, model identity
sign_diag_v1/diagnostic_records.json   per-frame raw replies, all four modes
sign_conformance_v1/conformance.json   conformance vs content, content grounding
sign_review_v1/                        28 review crops + review.json
```

New source: `scripts/{diagnose_sign_advisor,analyse_sign_advisor_conformance,recover_advisor_provenance,export_sign_diagnostic_review}.py`;
`signs.validate_sign_array`, `signs.mask_fields`,
`sign_advisor.{isolated_prompt_text,weight_manifest,resolve_snapshot}`, strict
`parse_response`.

Tests: `tests/test_frame_pose_signs.py` (33) adds value-domain rejection, shape
and dtype validation, model-level refusal, advisor-bank refusal, `mask_fields`
semantics, the capacity-matched attribution check, the strict response schema
across nine malformed shapes, verbatim isolated prompts, and the mask-discard
regression.

Full regression: **606 passed, 1 skipped** in the macOS authoring venv (`.venv`,
torch 2.13.0; the skip is the timm-gated backbone test). Attribution and
diagnostics ran in `animcv-framepose:cuda118` and `animcv-signadvisor:cuda118`
on LabServer63. No remote CI exists and none was claimed.

Attribution and the VLM diagnostic shared the GPU for part of their runs, so the
diagnostic's `frames_per_second` is not a clean measurement. No throughput claim
is made from it.
