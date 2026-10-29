# Worklog — Real Observation Reference Estimator Validity (2026-10-28–29)

## Starting state and fixed scope

Local, origin and LabServer63: `arch/single_frame_first`,
`1abba53cd5e71f9f9d9da818c94ccf5faf758a88`. Fetch and comparison completed before edits.
Local worktree clean; server untracked `.DS_Store`, `.animcv_sync_stage/`, `docker/` preserved.
No branch operation is authorized or performed. W99 CASE B and W100 CASE D remain accepted within their measured scopes.

Read PRODUCT_VISION, Architecture v3, coordinate conventions, W97–W100 including W99A, cumulative audit,
CLAUDE.md and all ten AnimCV Claude memory files (including its index). Execution hosts and actual runs are recorded below.
No production code, historical script, worklog or artifact is modified.

## Verdict

**CASE D — historical S0 reproduction blocked on the user-authorized Main execution environment.**

The M1 observation backend passed its focused historical-equivalence gate on LabServer63. After resource conflicts
and user-directed pauses/moves, Main's RTX 5080 successfully executed frozen S0 on the original benchmark inputs,
but its VALIDATION predictions did not reproduce the pinned historical SHA. The unchanged W100 harness stopped
immediately, before TEST or any M1 candidate inference.

The difference is numerically tiny: maximum prediction-to-prediction joint displacement **0.001811 mm**. This is
not an accuracy error against GT and not evidence of an estimator failure. It still fails the explicitly required
bitwise B identity gate; the criterion was not relaxed. **No B/M1 3D comparison exists in this batch.**

## Premeasurement corrections and declaration

This section was written **before W101 candidate inference**. W100 numerical acceptance criteria remain unchanged:
same-person M1 minus B MPJPE < +10 mm, limb/knee-plane differences < +2 degrees, same-person coverage >=90%,
and no important new failure mechanism for CASE A. W100's M8 parity failure is not reconsidered.

W100's unexecuted evaluation code requires these corrections in a separate W101 implementation:

- Population counts use all declared actor-frames, even without evaluation GT or detections. Unique-image tracking
  coverage, actor identity coverage and conditional accuracy have separate denominators. A model emitting a pose on
  an all-invalid input is not detector coverage.
- W73's <=0.10 association is an observation-based proxy. Compare the tracked observation with **all declared actors
  at that image**. Exactly one qualifying actor permits assignment; multiple qualifying actors are ambiguous. No
  qualifying actor is unresolved localization/association, not automatically proof of the wrong actor. Fewer than four
  shared limb landmarks is unmatchable. These outcomes remain in population accounting. Privileged candidate
  association is explicitly evaluation-only.
- Confidence/mask probes are conditional representation interventions; they do not establish a single cause when
  coordinates, masks and confidence differ together. Mmask uses benchmark validity only where finite RTMPose values
  exist. Invalid x, y **or confidence** must never enter S0 (W100 checked only x).
- Verify observation cache identities, image bytes, frame indexes, sizes, timestamps and exact bank positions at the
  handoff. W100 recorded several of these without enforcing them. Preserve bank x/W,y/H normalization exactly.
- Pair metric values on the same row **and same limb**, including conditional knee planes. W100's raw aggregate
  means have condition-dependent knee-plane support; report their denominators separately. 2D paired metrics use
  jointly valid identical limb joints; also report each source's individual coverage.
- Bootstrap whole **videos**, keeping actors in the same video together, 2,000 replicates, seed 1337, frame-weighted
  paired differences. Report video and actor counts; do not emit an interval with fewer than two videos. Small-cluster
  intervals remain descriptive. VALIDATION, TEST and historical challenge strata stay separate.
- Temporal comparisons use original actor neighbors and original timestamps, never bridge excluded identity rows or
  missing observations. Report eligible steps and limb/knee/torso event denominators. Correct W100's ambiguous NumPy
  `None not in (...)` comparison in the isolated implementation.
- Unseen-video review, if provenance is verified: entire clip at native cadence; uniform first/middle/last plus fixed
  largest observed 2D limb-direction steps and lowest validity frames, with deterministic earliest-frame tie breaks.
  These are descriptive good/bad probes, not unbiased accuracy samples. No independent 3D reference means no MPJPE.
- J1X omitted: its missing historical readouts and observation-dependent anchors/features prevent a clean frozen-input
  comparison. Frozen HMR2 is contextual only, conditional on exact existing identity/provenance.

Reference gate: production `PoseEstimator(MMPoseConfig(..., device="cpu", batch_size=1))`, visibility 0.3,
person score 0.3, no subject box, official hash-pinned W100 checkpoints. Sequential model initialization only. Fixed
sample = first 12 declared images of each W100 gate video, plus bar frames 1158/1168, black leading/middle frames,
and portrait/landscape dimensions. Historical adapter from `01c0dc3`, M1, and M1 repeat must agree bitwise (including
absence, visibility and order); all production forwards must have size one. M8 is never run.

## Execution evidence

### Host change and storage (before candidate measurement)

LabServer63's M1 reference gate passed on 85 frames; its frozen-S0 behavioral fixtures passed (initial 13 tests).
The user paused before full observation execution, then explicitly authorized execution on idle LabServer69.
The shared branch remains unchanged; LabServer69 uses a source snapshot, not a new/switched branch.
It has an RTX 3090 (24 GiB, Ampere SM 8.6), compared with LabServer63's RTX 3080 Ti (12 GiB, also SM 8.6).
Historical S0 CUDA batch 512 / AMP off / neutral signs and CPU single-frame MMPose remain fixed. The new host must
repass both the M1 gate and historical B prediction hashes; another GPU alone is not evidence of parity.

All W101 files, temporary directories, dependencies and caches on LabServer69 are under `/home/nd`.
The user also requested nd-wide temporary/cache defaults under `/home/nd`; these are applied only to that account.
Docker already used `/home/docker`. No Docker restart, system-wide temporary mount, other-user configuration or
administrator storage migration was performed. The previously prepared administrator script was disabled.
The root volume remains nearly full; it is not used for experiment storage.

The original image's dependency subset is copied unchanged, with Torch 2.1.2+cu118 installed from the official wheel
(published SHA-256 `60396358193f238888540f4a38d78485f161e28ec17fa445f0373b5350ef21f0`).
The native interpreter is Python 3.10.21 rather than the image's 3.10.12. No environment equivalence is claimed
without the execution gates. System DNS lookup failed; a resolved official address with normal TLS verification
was used for the wheel, without changing host DNS. No new dataset was downloaded.

All 9,251 destination images match the exact LabServer63 image-byte SHA manifest (0 mismatches;
manifest SHA `6bc334a0b63b53310b7abf9d6a5c12e875f1d7b65ce8e8908e96f1c0eb51a2e5`).
The copied frozen S0 checkpoint and historical prediction file hashes also match their pins.

### Premeasurement population ceiling

Computed from the unchanged W97 population manifest **before W101 full observations or candidate predictions**:

| Split | Actor-frames | Unique images | Videos | Actors | Maximum actor coverage of one subject/image |
|---|---:|---:|---:|---:|---:|
| VALIDATION | 3,407 | 3,131 | 12 | 16 | 91.899% |
| TEST | 7,076 | 6,120 | 24 | 37 | 86.490% |

Thus W100's unchanged 90% same-person actor-frame rule cannot be met on TEST by a one-subject tracker, even with
perfect detection and localization. This is a population/runtime cardinality limit, not evidence about S0's 3D
geometry. Keep the numerical rule and report the structural limit separately; do not silently replace its denominator
or attribute it to an estimator domain shift. Unique-image detection coverage, actor coverage and same-person
conditional reconstruction remain separate.

This premeasurement section makes no claim of candidate inference or LabServer69 execution-gate success.

## 1. Reference backend and trusted identities

The observation reference uses production `PoseEstimator` with an explicitly supplied `MMPoseConfig.batch_size=1`,
CPU/fp32, visibility threshold 0.3, RTMDet person score threshold 0.3, no initial subject box, and unchanged canonical
mapping. The production default remains **8**. No M8 inference was performed by W101.

| Identity | SHA-256 |
|---|---|
| RTMDet-tiny checkpoint | `78e30dcce0c6f594eaff0d6977b84b4103688b4aff0ad1aa16008a8cc854a7fb` |
| RTMPose-tiny checkpoint | `77ffc7e802acf10951c353e8bc68b4f05218121177ceaea163aa124436ba6fb7` |
| Pre-W99A adapter, `01c0dc3:src/pose/mmpose_adapter.py` | `0582a2a4f7653f1ffa5acbb4640b193ec2ec1909ec023cf428dba4e4bc962413` |
| RTMPose config source | `7a8e5057017515237274334b367c1fb87b2af6ea03ac77e8a852715794c923fa` |
| RTMDet config source | `873923f8221c2e5ae6501dae248eddf26743daeccb82ae9055cb6435fcabc992` |
| Frozen S0 checkpoint | `6c12fea9f84ea0493fada8b9acc96cd5c70fdd5d9114603e954f9412f4247b44` |
| Original bank content digest | `75519e6394a764e3749ddaa30555b58b73a01db582ccee14f661374b9a0ed536` |
| W97/W100 population manifest | `533895c980a080985444b703f43e0f64046391c76909dea5a2f305a26b34cdb9` |

The two detector/pose checkpoint pins are W100's previously verified official files, rehashed before loading here.
They were not downloaded or modified for W101. The model names/configurations remain RTMDet-tiny
`rtmdet_tiny_8xb32-300e_coco.py` and RTMPose-tiny `rtmpose-t_8xb256-420e_coco-256x192.py` (192×256, SimCC ratio 2).

## 2. Executed M1 historical-equivalence gate

**Regime: `real_animcv_observation`; observation parity only, not 3D reconstruction evaluation.**
Executed on LabServer63 in `animcv-framepose:cuda118` plus the isolated MMPose dependency directory, with CPU-only
container access and one Torch thread. Versions: Torch 2.1.2+cu118, numpy 1.26.4, MMPose 1.3.2, MMDetection 3.3.0,
MMCV 2.1.0, MMEngine 0.10.7, OpenCV 4.11.0. Initialization was sequential in the sole main Python thread.

| Fixed case | Frames | M1 = historical = M1 repeat | Tracked frames |
|---|---:|---|---:|
| downtown_walkBridge_01, first declared images | 12 | bitwise | 12 |
| downtown_bar_00, first images + 1158/1168 | 14 | bitwise | 14 |
| downtown_arguing_00 | 12 | bitwise | 12 |
| downtown_runForBus_01 | 12 | bitwise | 12 |
| flat_guitar_01 | 12 | bitwise | 12 |
| outdoors_parcours_00 | 12 | bitwise | 12 |
| Black leading/middle frames, leave/return, mixed dimensions | 11 | bitwise | 9 |

Total **85 frames per variant**, including 74 declared real images and 11 control frames (some reused images).
Every case has maximum coordinate/confidence difference 0, zero visibility/presence mismatches, identical tracking
reports, and preserved frame indexes/timestamps. The two black frames remain absent.

Instrumentation recorded **170 detector forwards and 166 pose forwards**, all of size **1**, across M1 and its
repeat. This confirms the public configuration reaches the model forwards. Historical and production resolved model
configs were identical; resolved config hashes are pose `083452718bd6c4349442a58d4946c1785e431f3c3d1f218945a36282fcfe2571`
and detector `a13e5b7269513a757fcdf86c1c94dd394019093e94948a08c98d0591e344c317`.

This is a focused reference gate, not W100's entire 1,949-frame population, not actor matching against 3DPW, and not
coverage evidence for all 10,483 actor-frames. It does not certify M8. The final script also supports a hash-pinned
historical source snapshot for hosts without git; that loader and the full-population runner were not executed as a
real observation population on Main or LabServer69.

## 3. Verification of the unexecuted diagnostic path

The W100 scripts remain intact. Corrections are confined to `scripts/w101_reference_estimator.py` and were declared
above before any W101 candidate performance measurement.

**Executed behavioral fixtures** pass through the actual new observation-cache adapter, production crop/geometry
functions, and the actual frozen S0 checkpoint. They are not source-text-only isolation checks.

- The fixture retains 12 actor-frames across six images, including absent tracked subjects and a different actor.
  It exercises normalization, canonical ordering, derived thorax/neck equality, no-subject zeros, invalid x/y/score
  handling, privileged association, representation probes and separate image/actor denominators.
- Corrupting image bytes, frame index, timestamp, image dimensions, bank position, split or the reference batch-size
  provenance is refused by the handoff. Reordered actor identities are refused; training-row geometry is preserved.
- GT coordinates were randomized, all GT validity was disabled, and evaluation strata were changed, while observations
  and inference inputs stayed fixed. Input tensors and **actual S0 CPU predictions were bitwise invariant** for B, M,
  Moa, Mconf and Mmask on the synthetic fixture. Here and in code, **M means M1**, retained for W100 reader compatibility.
- Statistical fixtures verify video clustering, paired differences, common-side knee support, common-joint 2D
  denominators, and refusal to bridge filtered temporal neighbors. Raw evaluation projection checks actor, split,
  unique frame lookup, camera transform and canonical axes; duplicate raw frame identities are refused.

**What these tests do not establish:** real-population association accuracy, validity/confidence distributions,
3D quality, full evaluator/report correctness on the dataset, or a complete GT-free public runtime. The full
`build`/`infer`/`evaluate` candidate workflow was not executed. No production module was patched to fix an adapter.

## 4. Declared population, provenance and denominators

The original W100 population is unchanged: **10,483 actor-frames / 9,251 exact images / 36 videos / 53 actor-sequences**.
The split counts and one-subject coverage ceilings are in the premeasurement table. Every declared image on
LabServer69 passed comparison against the exact LabServer63 byte SHA (0 mismatches). All 36 declared annotation
files also match their source SHAs. The completed historical Python dependency subset has 18,194 matching files;
five initially excluded nested SciPy compatibility files were restored before declaring the transfer complete.

This is selection-free accounting **within the inherited fixed bank**. The cumulative audit's upstream GT-conditioned
bank construction remains a limitation; this batch does not claim selection-free coverage of the entire raw dataset.
TEST remains a repeatedly used architecture diagnostic, not an untouched holdout, and no generalization claim is made.

`declared_population_manifest.json` on Main records exact bank positions, sample IDs, images, image SHAs,
timestamps, and every video's actor-frame count, unique frame count and actors. It is metadata accounting, not a
new observation cache. No real full-population cache exists, so actual tracked subjects, per-actor matching,
missing detections/joints and validity distributions are **unmeasured**, not zero. The same-person paired and historical
challenge denominators were never populated and are not merged into population coverage.

## 5. Main execution environment and frozen B reproduction

On 2026-10-29 the user explicitly moved execution to `ssh Main`. Main is Windows with an RTX 5080 (16 GiB, SM 12.0).
Its existing checkout was `arch/single_frame_first` at `14aa5740eb7c9bd3a2e6363936647cbfb4f7b10b`, 36 commits behind
the fetched remote, with untracked `third_party/mmpose/`. It was left unchanged, as required by the no-branch-operation
instruction. The experiment instead used an isolated source snapshot of **1abba53** under
`C:\Users\dna10\animcv-w101\code`, with the W101 additions copied separately. No packages in the existing environment
were installed, upgraded or modified.

Actual interpreter: `C:\Projects\AnimCV\.venv_build_full\Scripts\python.exe`, Python 3.11.9, Torch **2.13.0+cu130**,
CUDA runtime 13.0, numpy 1.26.4. The installed build includes `sm_120`. Main's compiled architecture/runtime differs
from the historical Ampere/Torch 2.1.2+cu118 path. PyTorch's [official Blackwell support announcement](https://pytorch.org/blog/pytorch-2-7/)
provides compatibility context; it is not the evidence for numerical parity or for this failure's cause.

Actual S0 controls: hash-pinned original checkpoint and bank, `S0_neutral_sign`, unchanged `geometry_tensor`, neutral
signs, CUDA, prediction batch **512**, AMP **off**, float32 matmul precision `highest`, matmul TF32 **false** and cuDNN
TF32 **true**. No backend option was tuned after observing the result.

**Regime: B is `benchmark_detector_observation`. No real-observation 3D condition was evaluated.**

| Gate item | Result |
|---|---|
| Bank content digest | exact `75519e63…0ed536` |
| Checkpoint SHA | exact `6c12fea9…47b44` |
| B VALIDATION rows / dtype | 3,407 / float32, 17×3 |
| Required prediction file SHA | `402d70a2bed79ae31919ff86905e19c8972108ea988d6cfa5deac514562132b9` |
| Main prediction file SHA | `f89d9bd071be9eb9e60d8b3dd77eb53db904fc10dfff418242c409cb7ee1d27d` |
| Bitwise equality | **FAIL**; 165,596 of 173,757 scalars differ |
| File-header bytes | identical |
| Maximum coordinate difference | 1.8030405e-6 m |
| Prediction-to-prediction joint displacement | mean 0.000216 mm; p95 0.000478 mm; max **0.001811 mm** |
| Finite outputs | all finite |
| B TEST / M1 / probes | **not executed** after the stop |

The differences above compare two predictions, **not predictions against GT**. They are not MPJPE and do not show
a practical reconstruction regression. They establish that Main's execution stack is not bitwise historical on the
required B gate. The hash difference is not merely Windows file formatting. GPU architecture, OS, Torch and CUDA
changed together, so their individual numerical contributions are unresolved. There is no evidence here to assign
the discrepancy to detection, confidence, masks, actor association or the estimator's response to real observations.

The unchanged W100 harness returned `STOP: condition B does not reproduce the pinned S0 validation predictions`.
No additional inference was performed after that failure; the subsequent displacement calculation was offline.
The W100 M8 0.5 px gate remains unchanged and is an independent historical result.

## 6. Unexecuted comparisons and conditional work

| Requested evidence | W101 status / limit |
|---|---|
| Full-population M1 detection/tracking/identity coverage | not executed; declared population and byte identities only |
| Same-person paired 2D localization / confidence / validity | not executed; no real candidate arrays |
| B/M1 MPJPE, per-joint/depth/sign/limb/knee/torso errors | not executed; B gate failed before candidate inference |
| Severe tails, sequence/posture distributions, temporal instability, uncertainty | definitions/fixtures only; no dataset verdict |
| Moa privileged association | synthetic fixture only; never a runtime actor-selection claim |
| Mconf / Mmask | synthetic fixture only; evaluation-only interventions, no observed attribution result |
| Frozen HMR2 context | not rerun, regenerated, fused or compared |
| J1X | omitted: historical RGB readouts missing and both its geometry anchors and RGB features depend on observations |
| Real clip / representative good-bad frames / runtime | not executed after the stop; no MPJPE or motion reconstruction claim |
| Numerical rig / skinned Blender review | not executed because the source comparison was not interpretable |

For the real clip, the user said `production_demo/source.mp4` might be BEDLAM but was unsure and permitted proceeding
with the existing file. W101 does not upgrade that recollection into verified source provenance or an independent
holdout claim. No repeated still image was used as motion evidence. No clip, render, rig action or visual review
artifact was generated, so there is no `user_QE` visual deliverable.

Existing BaseRig, rest calibration, FK/IK, parent-frame policies and Blender artifacts are unchanged. No GT
pelvis/root rotations entered a condition labeled GT-free; no mixed-source rig comparison was executed or relabeled.

## 7. W99A production compatibility boundary

This batch certifies none of the production optimizations:

- M8 remains numerically different from historical inference; its production default stays 8.
- Actual Depth Anything model batch parity remains unverified.
- Removing Blender `frame_set` still lacks independent Blender compatibility evidence.
- Optimized collapse still needs explicit key identity and curve parity evidence.
- Checkpoint deserialization and concurrent initialization risks remain open. Sequential trusted initialization here
  does not repair the public loading boundary.
- The public legacy Root Yaw export issue remains open.

None was used as an additional experimental variable or fixed in W101. No model replacement, production integration,
new runtime or style-authoring capability is claimed or authorized by this result.

## 8. Executed tests and reproducibility

| Execution | Actual result |
|---|---|
| LabServer63 initial W101 fixtures, original FramePose image | **13 passed**, 2.45 s; includes real checkpoint CPU GT perturbation |
| Main final W101 fixtures | **19 passed** |
| Main W100 diagnostic helper tests | **6 passed** |
| Main production MMPose adapter tests | **4 passed** |
| Main subject tracker tests | **3 passed** |
| Main combined focused suite | **32 passed**, 1.18 s; no skips |
| LabServer63 real-model M1 reference gate | **PASS**, 85 frames per variant |
| Main historical S0 B identity gate | **FAIL**, stopped on VALIDATION; exit 1 |

The overlapping test runs are not summed as independent passes. No broad regression was run because no production
contract changed. No unexecuted test, full-population result, GPU-source comparison or Blender review is marked PASS.

Actual Main commands (existing package environment; wrappers set W101-local temporary/cache directories and source
paths without account/global changes):

```powershell
$python = 'C:\Projects\AnimCV\.venv_build_full\Scripts\python.exe'
$setup = 'C:\Users\dna10\animcv-w101\setup'
& $python -X utf8 "$setup\w101_main_reference_gate.py" tests
& $python -X utf8 "$setup\w101_main_reference_gate.py" s0
# Post-stop, offline prediction comparison only:
& $python -X utf8 "$setup\w101_compare_main_reference.py"
& $python -X utf8 "$setup\w101_record_main_evidence.py"
```

The S0 wrapper invokes the **unchanged** `scripts/run_real_observation_s0.py` with the original bank/checkpoint,
an empty observation-replacement NPZ, `--device cuda`, and a fresh `output/s0_reference` directory. The report stops
at the first mismatching B split. The new diagnostic's full-population commands were **not executed**.

LabServer63 gate inputs and outputs are at `~/animcv-data/external/w101/gate_attempt2/`, with the invocation log at
`~/animcv-data/external/w101/gate_attempt2.log`. LabServer69 transfer manifests/reports are under
`~/animcv-w101/{data,output}/`; no experiment gate or model inference ran there.

Main measured artifacts are under `C:\Users\dna10\animcv-w101\output\`, including the raw S0 report, failed
prediction, numerical difference report, JUnit, declared population manifest, and executable-source/input SHAs.
Its preserved archive is `C:\Users\dna10\animcv-w101\w101_main_evidence_20261029.zip`, SHA-256
`07492b1b40854a161dedb7837d6b09128015f0a8b865468d2e3c2161f7ba39cf`.
The same verified ZIP and the executed M1 gate/initial fixture evidence are preserved separately on LabServer63 at
`~/animcv-output/framepose_real_observation_reference_estimator_w101/`. The archive's executed diagnostic/test source
SHAs match the three repository additions below. Machine artifacts were not placed in `user_QE`.

## 9. Exact changed files and preserved history

Repository additions only:

- `scripts/w101_reference_observations.py`: explicit M1 configuration, historical/reference gate, byte-bound cache
  recorder and snapshot loader; full-population mode remains unexecuted.
- `scripts/w101_reference_estimator.py`: isolated handoff, accounting, frozen input replacement, paired statistics,
  strict projection and diagnostic evaluation implementation; real-population candidate evaluation remains unexecuted.
- `tests/test_w101_reference_estimator.py`: 19 executed behavioral/contract/statistical cases.
- This worklog.

No `src/` file, W96–W100 worklog, W100 script, cumulative audit, checkpoint, original bank/benchmark observation,
frozen HMR2/J1X prediction, coordinate formula, rig, IK setting or historical Blender artifact was changed.
Remote side effects are confined to new W101 workspaces/copies and the explicitly requested nd-account storage
environment on LabServer69. No system-wide configuration or other project's service/container was changed.
Main's existing checkout and package environments remained unchanged. No branch was created, switched, merged,
rebased, renamed or deleted.

## 10. Evidence-bounded decision

**CASE D.** M1's focused historical-equivalence gate is valid on LabServer63, and the synthetic handoff/GT-isolation
fixtures pass. Main can run frozen S0, but the required historical B identity does not reproduce on its different
execution stack. The batch therefore cannot defensibly attribute a B/M1 source difference: no such comparison was run.

The central question — whether actual RTMDet/RTMPose observations preserve useful frozen S0 reconstruction and
which layer explains degradation — remains **unanswered under the required contract**. This is a valid contract-blocked
completion, not CASE A, B or C, and not a negative estimator-accuracy result. Numerical acceptance criteria and
historical decisions were preserved. No production readiness or downstream capability follows from this batch.
