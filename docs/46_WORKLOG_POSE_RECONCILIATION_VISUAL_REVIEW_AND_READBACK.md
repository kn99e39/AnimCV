# Worklog - Pose Reconciliation Visual Review, Read-back Diagnosis, and Sign Advisor (2026-09-15)

> Evidence-only continuation on `arch/single_frame_first`, starting at
> `54c4ff009528d39c1d56d29613aeb9e7d8d1f1f7`. No reconciliation policy,
> SignState contract, refusal behavior, prompt, model, crop, or production
> default changed. The human architecture decision is deferred. The full
> current-backend VLM evaluation is **pending explicit data-export
> authorization**; this worklog does not claim the three-track batch complete.

## Preservation boundary

The historical evidence and its source inputs were checked and not modified:

| Artifact | SHA-256 |
| --- | --- |
| `docs/44_WORKLOG_POSE_RECONCILIATION_REAL_REPLAY.md` | `a291391641d04a55b67800cb1afab91bc8c99572dab674993462211014cd3422` |
| `docs/45_WORKLOG_POSE_RECONCILIATION_CAUSAL_ATTRIBUTION.md` | `50468a64c574dcce0334fad3858a263784ed0467032e5223bcbce8eba08e791c` |
| docs/44 replay JSON | `61fde30e7e6d8d9769a0273450819478fc6c457af3841eae3d0333c1f8e16e8c` |
| docs/45 attribution JSON | `c7d2c54be3aff937f8c88882bbabe6d18c1f107c2247c723d79480e4171e09eb` |
| recovered FrameBank content digest | `75519e6394a764e3749ddaa30555b58b73a01db582ccee14f661374b9a0ed536` |
| O_BILATERAL test prediction | `6df04c2031b2bde6a4f0af37fdae0a7f0228e9fc23401e15e1aa2b3bd35695a5` |
| evaluation identity | `a5881850b3769ca2329271d327ee895dc646f236451f05c3ba2e0ff917f580a3` |

All 24 raw 3DPW camera files were revalidated against docs/44. Review-video
neighboring frames are visualization context only and never enter inference or
reconciliation.

## Track A - owner-facing visual A/B evidence

The exporter verifies the fixed bank, prediction, evaluation, replay, and
camera identities, then writes the manifest and blind assignment before any
video writer is opened. Selection is deterministic over the exact docs/45
`C_READABLE_WRONG` rows: per-field median or positive-P90 representatives with
stable sample-ID tie-breaks. Seven primary regimes have four field
representatives each; overlaps produce 27 unique events across all four
elbow/knee chains. Four `C_H0_UNKNOWN` rows with counterfactually feasible
swivel geometry are selected separately, one per chain, and labelled as
counterfactual diagnostics rather than normal-policy outputs.

Review package (outside Git; media intentionally not committed):

```
/private/tmp/animcv-pose-reconciliation-visual-review-20260915-v4/
```

It contains 27 neutral Candidate-A/Candidate-B clips, 27 labelled
MINIMUM_NORM/R_SWIVEL_OBS clips, 27 technical stills, four labelled
H0-UNKNOWN counterfactual clips and stills, and four separate Oracle
upper-bound stills: **58 MP4s and 35 PNGs**. MP4s are 1440x810 at the stored
3DPW rate (verified at 30 FPS); sampled primary and counterfactual clips each
decode as 91 frames, approximately three seconds. Poses are drawn only at
their own FrameBank timestamps. There is no temporal inference, interpolation,
or smoothing, and every clip says video context is visualization only.

Each candidate uses identical per-clip 3D bounds and camera orientation. The
3D views show the complete 17-joint pose in neutral strokes and highlight the
active P-M-D chain, with P/D anchors and M emphasized; views are camera-aligned
and fixed oblique (azimuth 35 degrees, elevation 22 degrees). The primary
clips show RGB, observed 2D, common H0 projection, A/B projection, and A/B 3D.
Oracle target projection is excluded from both primary exports.

Blind assignment uses SHA-256 and seed `20260915`; `blind_key.json` is
root-level and outside `blind/`. The blank `review_sheet.csv` has one row per
primary clip and no filled judgment fields. No visual preference was inferred.

| Review artifact | SHA-256 |
| --- | --- |
| `review_manifest.json` | `d9e35e691865fcc878253172bfe43cc179ff63278564712bd98ae08c36b2bead` |
| `blind_key.json` | `02af5c5a59b6e0f9d602f9a4885027a488a00343d18508198b0572b1c2d714ec` |
| `labelled/technical_manifest.json` | `b278a0d96511077bf33788aa376863d5900c307c85068cc88c7c7daeb45d19cf` |
| `artifact_hashes.json` | `4ea3ab09c6e8a5a752c729957112903cbfdaef86d2d26219cbf7da85b8e111b6` |

`artifact_hashes.json` records hashes and byte counts for all 93 media files.
An independent second export produced byte-identical manifest, blind key,
review sheet, and artifact-hash index, verifying deterministic selection and
assignment as well as deterministic rendered media bytes.
The blind-facing manifest omits per-candidate read-back/source and theta
details. Full method-labelled outcomes, metrics, and A/B mapping are in
`labelled/technical_manifest.json` and `blind_key.json`, separate from the
blind media and blank review sheet.
Each primary regime has four selected events: both corrected, MINIMUM_NORM
corrected with Swivel read-back failure, large Swivel observation-space
advantage, large MINIMUM_NORM bend-direction advantage, each middle-joint
error preference, and wrong-sign common-corrected stress. H0-UNKNOWN has four
separate counterfactual diagnostics.

**Human visual conclusion: evidence package generated; architecture verdict
DEFERRED TO PROJECT-OWNER REVIEW.**

## Track B - final canonical SignState read-back failures

The analysis consumes exactly the docs/45 `C_READABLE_WRONG` rows and
independently compares solver circle/theta semantics with canonical
`bend_direction` and `sign_state`. Neither path is changed.

| Variant | Failures | Canonical UNKNOWN | Opposite sign | Stationary attempts / failures | Readability-boundary attempts / failures | Half-angle-infinity attempts / failures |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `R_SWIVEL_OBS` | 213 | 213 | 0 | 762 / 0 | 628 / 213 (33.917%) | 0 / 0 |
| `R_SWIVEL_ORACLE_2D` | 195 | 195 | 0 | 819 / 0 | 571 / 195 (34.151%) | 0 / 0 |

All 408 failures classify as `NUMERICAL_READABILITY_BOUNDARY`; formula
mismatch, basis mismatch, actual wrong branch, and other categories are all
zero. Circle membership residual is at most `8.33e-17 m^2` (p95
`1.39e-17 m^2`); basis reconstruction residual is exactly zero. Endpoints
are unchanged and the largest P-M/M-D length error is `1.81e-13 mm`.

Per-chain split:

| Hinge field | OBS | Oracle 2D |
| --- | ---: | ---: |
| Left elbow | 71 | 64 |
| Right elbow | 62 | 65 |
| Left knee | 40 | 25 |
| Right knee | 40 | 41 |

The requested solver margin `requested * sqrt_f * cos(theta) -
UNIT_FORWARD_EPSILON` is at floating-point scale. OBS p05/median/p95 are
`-1.67e-16 / -1.39e-17 / +1.30e-16`; Oracle values are
`-1.67e-16 / -2.78e-17 / +1.25e-16`. Canonical requested margins are
strictly negative in every failed row: OBS
`-5.22e-16 / -9.71e-17 / -1.39e-17`, Oracle
`-4.86e-16 / -1.11e-16 / -1.39e-17`. Solver-minus-canonical forward-component
differences have maximum absolute magnitude `2.09e-15` (OBS) and
`9.72e-16` (Oracle), not a material semantic disagreement.

Successful controls: OBS has 762 stationary and 415 successful boundary rows;
Oracle has 819 and 376. Stationary solver-margin p05/median/p95 are
`0.0687 / 0.5942 / 0.8740` (OBS) and
`0.0534 / 0.5434 / 0.8787` (Oracle). Successful boundary controls remain at
machine precision too; their solver-margin median is `6.94e-17` for both
variants, and the per-row canonical margins are included in the diagnostic.

Synthetic probes retain the existing threshold. Just below, solver and
canonical values are `0.09999999999999978` and canonical state is UNKNOWN. At
mathematical equality, solver gives `0.1`, canonical reconstruction gives
`0.10000000000000002`, and canonical state is +1. Just above, both are about
`0.10000000000000023` and state is +1. This exposes strict-threshold rounding
at reconstruction precision.

Machine-readable per-row evidence:

```
/private/tmp/animcv-pose-reconciliation-evidence-20260915/readback_diagnosis.json
SHA-256: f7588c8ce3717ce425cbba30430522c2af9349506166138e56445a012e49bb93
```

**Read-back verdict: BOUNDARY / NUMERICAL CONTRACT MISALIGNMENT.** Evidence
does not support a structural semantic mismatch or invalid swivel circle. No
repair was made.

## Track C - adopted external Sign Advisor and evaluation status

The current research/runtime environment identifies the adopted external Sign
Advisor as `Qwen/Qwen2-VL-2B-Instruct`. It is a separate research sensor, not a
VLM already wired into the production pose-reconciliation path. Adoption is
established jointly by the explicit selection/authorization in docs/28
Sections 9-10, the subsequent strict-parser diagnosis in docs/29, the current
branch's Qwen-backed Sign Advisor execution scripts and pinned runtime
Dockerfile, and the matching LabServer research environment. It is not
inferred merely from an old model ID in provenance.

Recovered snapshot commit:
`895c3a49bc3fa70a340399125c650a463535e71c`; weight fingerprint:
`840bc66b30f80632ade2f1fd6f415c34bb2f89ff0460ecee98a572c8d77eabc0` (11
files). LabServer runtime image `animcv-signadvisor:cuda118` has image ID
`sha256:8e50fb38f1086a3b91ef7cfb0320469202bb031beecf9bec2dcd373ae2477e92`.
The branch/LabServer source files were byte-identical at audit: Sign Advisor
`16892c8ed7e7de161e4eefb42f739dd6f3913b1c71f94a6f0db93030e098f99f`, crop
features `c0ae1f94e1e2086cd6d5a0296be567c3628072e5e374430643a06b48f90348fa`,
bank builder `46b2f0ec6831ecf94b338d1553e9062676b75bdd75228a9139e88871564142e0`,
diagnosis runner `3559c7540b7408f931f690f608ea415966be6a44e04d242b016a3781a483193f`,
and runtime Dockerfile
`73f2315c5eeb8ea493b3645e5f3e9aa6286a62f306c6bd73d16fb698317f140a`.
Historical 600-frame Qwen findings in docs/28-29, including parser and weak
branch behavior, remain **HISTORICAL REFERENCE**, not a substitute for this
batch's full-test result.

The frozen-provenance full-test evaluator is prepared at
`scripts/evaluate_current_sign_advisor.py`. It requires exact docs/44 replay
and docs/45 attribution inputs; asserts docs/45 field counts for
`C_READABLE_WRONG` (406/306/268/410) and `C_H0_UNKNOWN` (353/285/324/314);
and records paired raw responses. Each shuffled image donor is from a
different sequence. It preserves the prompt, strict parser, 448-pixel crop,
greedy decoding, fixed snapshot, and seed. The report includes seven-field,
per-field and pooled hinge metrics; confusion, abstention, and wrong-sign
accounting; real/shuffled output and coverage changes; and stratified paired
bootstrap intervals on both docs/45 hinge populations.

**Inference did not start.** Security review rejected copying the new
evaluation source and docs/44-45-derived reports to LabServer because explicit
approval for exporting those payloads to that destination was absent. No
source/report file was transferred; no model inference ran. An empty isolated
staging directory may remain on the server. This is an authorization blocker,
not a model/provenance or compute limitation. Focused local tests validate
population, pairing, parser, metric, and bootstrap accounting, but do not
substitute for inference.

Therefore Track C's identity result is **adopted Qwen research backend found**,
but its competence result and V1/V2/V3 evidence classification are
**PENDING**. V0 does not apply. No competence claim is made from the historical
sample.

## Tests, scope, and remaining owner actions

Focused checks passed: `13 passed` across the new read-back, visual-selection,
and Sign Advisor evaluator tests; scripts and tests passed `py_compile`. One
primary and one counterfactual MP4 were opened and decoded at the recorded FPS
and dimensions. No production regression run was needed because no shared
production behavior changed.

No solver/refusal/SignState change, threshold or tolerance change, VLM
prompt/crop/decoding/model tuning, temporal inference, or policy promotion
occurred.

Remaining:

1. The project owner reviews the neutral clips using the blank sheet and then
   separately consults `blind_key.json`; no architecture choice has been made
   for them.
2. The project owner explicitly authorizes or declines transfer of the
   evaluator and exact docs/44-45 reports to the isolated LabServer directory.
   If authorized, run all 7,076 test rows and report real/shuffled results
   before assigning V1/V2/V3 evidence status.

Until then, do not promote MINIMUM_NORM or R_SWIVEL, add a hybrid, alter
H0-UNKNOWN refusal, or proceed to target-rig work.
