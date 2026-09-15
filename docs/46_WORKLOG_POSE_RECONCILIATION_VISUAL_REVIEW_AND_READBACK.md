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

**The sequential full-test inference was started, then stopped for throughput work.** On 2026-09-15 the project owner explicitly
approved transfer of the evaluator, its two required helper scripts, and the
exact docs/44-45 reports to an isolated LabServer directory. After the initial
security-review rejection, the owner clarified that all required files were
explicitly authorized. Read-only preflight reconfirmed the frozen runtime,
cached Qwen snapshot, GPU, bank/prediction identities, and image root. The five
transferred files were re-hashed remotely and exactly matched their local
sources: evaluator `8ee645ec395a785ae6c12b2a8c88fc981e08b7c905dcc482dae1dce7aa9f2f9e`,
attribution helper `4953a53f8c048547fe3804f921e211b072b8dcbad81093a03efae10cf890c1c0`,
replay helper `81283cbf8bc336e128db2fa5bd0d911f660de8f31928ee4cccfdba35f13f394b`,
docs/44 replay `61fde30e7e6d8d9769a0273450819478fc6c457af3841eae3d0333c1f8e16e8c`,
and docs/45 attribution `c7d2c54be3aff937f8c88882bbabe6d18c1f107c2247c723d79480e4171e09eb`.
The model is `Qwen/Qwen2-VL-2B-Instruct`, snapshot
`895c3a49bc3fa70a340399125c650a463535e71c`, on the pinned
`animcv-signadvisor:cuda118` image. The repo, FrameBank, H0 prediction, camera
images, and HF cache are mounted read-only; only the dedicated run output is
writable. No prompt, crop, parser, model, or source checkout was changed.

The run evaluates every one of the 7,076 docs/44 test rows with both the real
image and a deterministic different-sequence shuffled image. Progress is
checkpointed in
`LabServer63:/home/nd/animcv-output/framepose/sign_advisor_current_backend_full_test_20260915/output/full_test_responses.jsonl`.
At the pre-stop capture, 420/7,076 rows were complete at about 0.131 rows/s,
with GPU utilization 25% and memory 5,244/12,288 MiB. No full-test metrics or
competence conclusion exists until all rows and the final aggregation
complete.

### Sequential reference capture before throughput work

Pre-stop snapshot captured at `2026-09-15T08:56:21Z` (remote file stat at
`08:56:25Z`): 421 JSONL lines (one run identity plus 420 complete unique
frames, orders 0–419), 758,159 bytes, SHA-256
`9eb8437f2fa5503621c962259c2790970cf48e212d5970532953e3f8488b7396`.
The persistent reference path above is not to be deleted or overwritten.

- Branch `arch/single_frame_first`; run-start HEAD `e88d5a2521b095ad112b48300d7b941c332d3637`; HEAD at capture `f6c4bf28fcfb6109171b7faf814792ea6d4fb7b3`.
- Evaluator SHA-256 `8ee645ec395a785ae6c12b2a8c88fc981e08b7c905dcc482dae1dce7aa9f2f9e`; started `2026-09-15T08:01:34Z`; measured sequential rate `0.131 rows/s`.
- Container `0552172fae0ffa89c85bffcd744040bf9e373b8c54a8b33613b707a2c74f201a`, image `animcv-signadvisor:cuda118`, image ID `sha256:8e50fb38f1086a3b91ef7cfb0320469202bb031beecf9bec2dcd373ae2477e92`.
- Model `Qwen/Qwen2-VL-2B-Instruct`, revision `895c3a49bc3fa70a340399125c650a463535e71c`, weight fingerprint `840bc66b30f80632ade2f1fd6f415c34bb2f89ff0460ecee98a572c8d77eabc0`; FP16, Transformers 4.49.0, torch 2.1.2+cu118, greedy decoding, 160 max new tokens, seed 1337, 448×448 crop, prompt SHA-256 `4529750c6835fbf4ae11b7d2874c4dbd61d6ed9ec5e4ca8d13062d2783405446f`.
- GPU at capture: 25%, 5,244/12,288 MiB.

Exact container command (run output is a bind mount and remains intact when the
container is stopped):

```sh
ssh LabServer63 'docker run --rm --gpus all --network none --shm-size=1g --mount type=bind,source=/home/nd/AnimCV,target=/workspace/AnimCV,readonly --mount type=bind,source=/home/nd/animcv-output/framepose/sign_advisor_current_backend_full_test_20260915/input,target=/run/input,readonly --mount type=bind,source=/home/nd/animcv-output/framepose/sign_advisor_current_backend_full_test_20260915/output,target=/run/output --mount type=bind,source=/home/nd/animcv-output/framepose,target=/data/framepose,readonly --mount type=bind,source=/home/nd/animcv-data/datasets/3dpw/imageFiles,target=/data/3dpw/images,readonly --mount type=bind,source=/home/nd/animcv-hf-cache,target=/cache,readonly --workdir /workspace/AnimCV --env PYTHONPATH=/run/input:/workspace/AnimCV/scripts:/workspace/AnimCV/src --env HF_HOME=/cache --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 --entrypoint python3 animcv-signadvisor:cuda118 /run/input/evaluate_current_sign_advisor.py --bank /data/framepose/bank_3dpw_paired_v2.json --h0-prediction /data/framepose/sign_attr_v2/O_BILATERAL/prediction_test.npy --replay-report /run/input/pose_reconciliation_replay.json --attribution-report /run/input/pose_reconciliation_attribution.json --image-root 3dpw_images=/data/3dpw/images --out /run/output --revision 895c3a49bc3fa70a340399125c650a463535e71c --hf-cache /cache --started-utc 2026-09-15T08:01:34Z --resume'
```

The complete command arguments and run identity are also recorded in the
container configuration and first JSONL record, respectively.

After this capture the recorded container was explicitly stopped for the
throughput investigation. Post-stop validation found 437 valid JSONL lines
(run identity plus 436 unique frames, orders 0–435), 786,676 bytes, SHA-256
`837ab1ef64ee7b57458500185dbf14e23056c86233d6e2de8d455aba79584ba3`, stat
time `2026-09-15T08:58:29Z`. The container exited 137 as a result of the
explicit stop; no final metrics were written, and no inference error was
reported. The original output remains at the sequential reference path and is
not to be overwritten. Current local HEAD at stop was
`b1e82c4`; the only commits after run start are Markdown worklog records.

Therefore Track C's identity result is **adopted Qwen research backend found**,
but its competence result and V1/V2/V3 evidence classification are
**PENDING**. V0 does not apply. No competence claim is made from the historical
sample.

### Execution-throughput optimization

The sequential reference is frozen at 436 completed frames and must not be
resumed or overwritten by the new execution path. The optimization changes
only inference grouping: each receiver contributes its real crop followed by
the same deterministic shuffled donor crop; a receiver batch of N therefore
uses one processor/generation call with 2N requests. The model, frozen weights,
FP16 dtype, seven-field prompt, 448×448 crop bytes, 160-token greedy decoding,
strict parser, donor assignment, full 7,076-row population, and metrics remain
fixed.

Added `scripts/sign_advisor_batching.py` for ordered real/shuffled interleaving,
crop-byte hashes, bounded one-batch prefetch, batched generation, output
equivalence comparison, and resume-identity checks. The evaluator retains its
original serial path and exposes a distinct `--frame-batch-size` path whose run
identity includes batch size, prefetch bounds, and the SHA-256 of the required
equivalence report. Prefetch uses four ordered crop-read workers, a separate
processor instance, and at most one pending batch. The bounded benchmark tests
N=4, 8, then 16 over the same deterministic 64 completed reference rows, with
an 8-frame serial component profile. It records crop/read, PIL, processor,
host-to-device, generate, and decode/parse time; requests/s, frames/s, GPU
utilization, peak VRAM/headroom, batch latency, and raw/parse/state equality.
It stops on OOM, under 2 GiB headroom, or less than 10% throughput gain.

The original reference was rechecked on LabServer at `2026-09-15T09:14:55Z`:
437 JSONL lines (identity plus 436 frames), 786,676 bytes, unchanged SHA-256
`837ab1ef64ee7b57458500185dbf14e23056c86233d6e2de8d455aba79584ba3`; the RTX
3080 Ti was idle (0%, 1/12,288 MiB). Local focused tests for batching,
interleaving, bounded ordering, crop hashes, equivalence, resume identity, and
malformed response handling pass (`12 passed`), and the three scripts pass
`py_compile`; Ruff passes with the repository's existing script-path `E402`
imports ignored. The committed scripts were transferred to a distinct
`input-batched` directory, with all three remote SHA-256 values matching the
local files. The benchmark used the pinned container, read-only sequential
reference, model cache, FrameBank, and image root. It did not modify the
reference JSONL.

Bounded result at
`LabServer63:/home/nd/animcv-output/framepose/sign_advisor_current_backend_full_test_20260915/throughput_benchmark_20260915/run1/`:

- The short sequential profile replayed 8 frames/16 requests and reproduced
  all raw outputs exactly. It measured `0.131346` frames/s (`0.262693`
  requests/s), GPU average `28.94%` (max `73%`), peak host VRAM `5,246/12,288`
  MiB. Stage seconds: crop/read `0.9364`, PIL `0.0048`, processor `0.2377`,
  host-to-device `0.0258`, generate `59.6698`, decode/parse `0.0051`. Generation
  dominates; data loading and CPU preparation do not.
- N=4 (8 VLM requests) completed all 64 selected frames in `85.173` seconds:
  `0.751412` frames/s, `1.502825` requests/s, or `5.72085×` the short serial
  profile. Estimated 7,076-row steady-state time is `2.616` hours versus
  `14.965` hours sequential. Batch latency median/p95 was `5.281/5.365` sec.
  GPU average/max utilization was `52.36%/100%`; peak host VRAM was `9,612`
  MiB and conservative remaining headroom `2,300.6` MiB (safety threshold
  `2,048` MiB). Component timings overlap due to prefetch.
- Equivalence on all 128 requests: parse validity/reason and final seven-field
  SignState matched `128/128`; raw strings matched `126/128` (`98.4375%`). The
  only raw differences were `3dpw:downtown_arguing_00:actor0#000030` real and
  `3dpw:downtown_arguing_00:actor0#000345` real. Both parsed equivalently; no
  evaluation-relevant mismatch occurred. Because raw output was not byte
  identical, the sequential partial must not be resumed or mixed into the
  competence artifact; full N=4 must start from zero in its own output path.
- N=8 failed on its first request batch with CUDA OOM (attempted an additional
  8 GiB); the benchmark stopped there and did not attempt N=16. N=4 is the
  largest useful safe size in this bounded sweep. FlashAttention-2 is supported
  by this pinned Transformers model class but `flash_attn` is not installed in
  the pinned Docker image, so no runtime package or image was changed.

The benchmark report SHA-256 is
`557443b6390327afe2594f5a664512c759da3a1612728335bf6c3be76f4bd6ef`; its N=4
candidate JSONL SHA-256 is
`ceff2a7998a3ecf275d56ad0e0f5e84b284a83ebb97a50ee4cffe97ca35b14ab`. The
original sequential JSONL remains 437 lines / 786,676 bytes with SHA-256
`837ab1ef64ee7b57458500185dbf14e23056c86233d6e2de8d455aba79584ba3` after the
benchmark. The next step is a clean, full 7,076-frame N=4 run under a distinct
run identity and output directory.

### Complete N=4 frozen Sign Advisor execution

The clean N=4 run completed on LabServer63 at
`/home/nd/animcv-output/framepose/sign_advisor_current_backend_full_test_20260915/batched_full_test_20260915/full_test_n4/`.
It has one identity record plus all 7,076 requested frame records (7,077 JSONL
lines); its manifest records `COMPLETE`, `records=7076`, and
`expected_records=7076`. No Sign Advisor process remains running.

- Raw response ledger SHA-256:
  `bb586b82c1a14083af0b6bb222f361c6b759173d5448bc55d5224dea6f4dd514`.
  Metrics SHA-256: `824de28b0b7c3d78c35b88110ca50a938dfd1ffeea918a4c799e83362b58fe71`.
  Manifest SHA-256: `3178f74e648f998f5cb0feeee009988bb0fd8b68bdb40b866d14a8ca2bfade2d`.
- Actual execution time was 9,729.952 seconds (2.703 hours): 0.727239
  frames/s and 1.454478 VLM requests/s for 7,076 real/shuffled frame pairs.
  This is 5.54× the short sequential reference profile (0.131346 frames/s),
  while retaining the separately recorded N=4 execution identity and the
  bounded-equivalence evidence. Generation remained dominant (9,701.661 sec);
  crop/read, processor, and host-to-device time totaled 759.200 sec.
- The original sequential reference was not changed or mixed into this result.
  The N=4 run records the fixed request batch size 8, four crop workers, one
  pending prepared batch, and the benchmark-equivalence report SHA-256.

The competence result is unequivocal under the **frozen strict contract**:
every real and shuffled response (7,076 each) began with a Markdown JSON code
fence, so `parse_response` rejected every one as "response is not exactly one
JSON object." All emitted SignStates are the neutral seven-UNKNOWN vector. The
original raw ledger was directly audited: real valid `0/7076`, shuffled valid
`0/7076`, and both modes have 7,076 parser failures. Consequently all
evaluation-relevant coverage, accuracy, balanced accuracy, and real-vs-shuffled
SignState change rates are zero. Raw text is not constant -- 3,055/7,076
real/shuffled raw strings differ -- but none can enter the frozen sensor due to
the parser contract. This is a valid negative competence finding for the
specified backend/prompt/parser combination, **not** evidence that the image
does not affect the model's unconstrained text.

An aggregation defect was found during post-run audit: the
`metrics.paired_real_vs_shuffled` convenience summaries had passed all-true
parser-valid arrays into their per-mode and pooled metrics, even though the
raw JSONL records are invalid. Their state/coverage/accuracy conclusions remain
zero because every rejected response has the all-UNKNOWN state, but their
`parsed_valid_rows` and strict-parse-success fields must not be cited from the
hashed remote metrics JSON. The underlying ledger, the normal per-population
`by_mode` parser accounting, and the negative competence conclusion remain
unambiguous. The local evaluator is corrected and regression-tested so future
runs propagate the real/shuffled validity arrays into these paired summaries;
the completed immutable raw artifact is preserved rather than rerun or mixed.

Therefore Track C's competence evidence is complete but negative for the
frozen strict interface. It does not license VLM authority in H0-UNKNOWN or
any automatic policy promotion. A separate, explicitly authorized future
experiment could test a different output-interface contract; it must not be
reported as this frozen competence evaluation.

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
2. Run the selected N=4 full evaluation from zero in its new output directory;
   never mix it with the sequential reference. Then verify response and
   metrics hashes, report results, and assign V1/V2/V3 evidence status without
   promoting any policy automatically.

Until then, do not promote MINIMUM_NORM or R_SWIVEL, add a hybrid, alter
H0-UNKNOWN refusal, or proceed to target-rig work.
