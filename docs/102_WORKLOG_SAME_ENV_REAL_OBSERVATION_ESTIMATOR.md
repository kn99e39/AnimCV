# Worklog — Same-Environment Real Observation Estimator Attribution (2026-10-29)

## Starting state and principle

Branch `arch/single_frame_first`, starting HEAD `72e0b6aa03d93ce246ff13283180604f51004243`; fetched origin and
confirmed no divergence, clean worktree. W100 CASE D and W101 CASE D, their exact criteria, prediction SHAs and
artifacts remain historical and unchanged. No branch operation or production-module change is authorized.

**Preflight gates protect conditions needed to test the central hypothesis. They must not unnecessarily remove the
opportunity to test it.** The historical cross-stack byte-identity result is separated from repeatability and
source-observation effects within one fixed Main stack. No failed historical criterion is redefined as a pass.

## Measured verdict

**Mixed outcome: conditional CASE A, selected-subject/local-failure CASE B, persistent CASE C-type failures.**
The controlled comparison is established; W102 is **not CASE D**. Replacing the benchmark observation with M1
improves mean root-relative S0 accuracy and temporal event rates on the uniquely associated population. This is
conditional observation compatibility, not a universal animation-quality pass: important local regressions, identity
coverage loss and shared 3D/rig failures remain. The experiment does not identify estimator error versus monocular
ambiguity sufficiently to declare that intrinsic estimator limitation dominates every residual failure.

W100/W101 CASE D, failed M8 parity and failed historical Main SHA remain unchanged. No production integration,
weights, thresholds, source values, camera/normalization, IK, root orientation or default batch size changed.

## Declaration before M_Main performance

Primary S0 execution is Main's existing native environment: RTX 5080 (SM 12.0), driver 596.49, Python 3.11.9,
Torch 2.13.0+cu130, CUDA 13.0, numpy 1.26.4. Preserve W101's CUDA/fp32, batch 512, AMP off, neutral signs,
`highest` matmul precision, matmul TF32 false, cuDNN TF32 true, original checkpoint/bank/crop code. No package,
precision, threshold, model, device or normalization tuning. New source snapshot/workspace only; existing checkout
and environment are untouched. No LabServer63/69 experiment is authorized by this batch.

### B_Main stability and portability

Two independent processes produce B_Main on both original VALIDATION and TEST rows. Exact byte repeatability is
reported. Numerical repeatability protects the controlled comparison: all outputs must be finite float32 with exact
row/actor ordering; maximum repeated raw and root-relative joint displacement <= **0.1 mm**, limb/torso orientation difference <= **0.05°**,
and well-conditioned knee-plane difference <= **0.1°**. These numerical budgets are fixed before M_Main: respectively
100×, 40× and 20× smaller than W100's material source-effect scales (10 mm / 2°), not selected by a tolerance sweep.
Any repeat difference in a relevant discrete decision outside its declared boundary band blocks the comparison.

Historical-vs-Main discrepancy is audited separately for joint positions, all limb directions, knee planes, torso
frames and GT-referenced discrete decisions (wrong knee branch, depth sign, torso forward flip, MPJPE >200 mm).
The historical SHA mismatch remains FAIL and does not itself block M_Main. Material cross-stack geometry/decision
changes would block an unchanged-estimator interpretation, using the same numerical/decision budgets above.

Ill-conditioned geometry is not silently assigned an angle. Knee planes require both bends >=20°, with the existing
1e-3 cross-product norm condition for temporal normals. Existing torso validity rules remain unchanged. Report
degenerate/near-straight cases separately. Boundary bands: 0.1° around bend eligibility (20°), wrong-branch / torso-flip
(90°), temporal angle thresholds, and torso side-axis validity (20°); 0.1 mm around predicted limb depth zero and
the 200 mm severe threshold; 0.1 mm around the existing 1 mm trunk validity floor; 1e-5 m² around the existing
1e-3 m² temporal cross-normal validity floor. Any changed boundary decision is
listed with its margin; changes outside these bands or a continuous numerical budget fail the gate. No threshold
or boundary is adjusted after seeing the candidate.

### Populations and selected-subject policy

Preserve all 10,483 actor-frames, 9,251 unique images, 36 videos and 53 actor-sequences; separate VALIDATION and TEST.
The inherited bank is GT-conditioned upstream and TEST has repeatedly informed architecture decisions. Neither
full raw-dataset coverage nor untouched-holdout generalization is claimed.

1. Unique-image detection coverage: any detector candidate and, separately, selected production detection exists /
   every declared image, including failures.
2. Tracking continuity: original consecutive image timestamps, selected-box IoU, detection gaps and evaluation-only
   uniquely associated actor transitions. Do not bridge unresolved/missing frames.
3. Actor association: W101/W73 median shared-limb distance / benchmark valid-joint bbox diagonal <=0.10, >=4 shared
   limb joints. Compare against all declared actors in that image; preserve unique, ambiguous, no association,
   insufficient-shared-observation and missing-subject statuses.
4. Same-person paired reconstruction: only the actor uniquely associated with the actual selected subject in that
   image, with common GT and metric-specific joint/limb support. Association labels never select a runtime subject.
5. Original all-actor-frame accounting: keep all other/missing/ambiguous actor records. Do not count a dense pose from
   an all-invalid input as observation coverage or conditional reconstruction success.

Pre-M_Main source inspection identified that `target_valid` means **benchmark-visible AND GT-valid**, not pure
3D-reference availability (`src/training/temporal_lifter.py:build_dataset`, `src/framepose/sources.py`).
Requiring `.all()` would exclude 1,310 VAL / 2,836 TEST rows from scoring even when the raw reference is valid.
Primary W102 accuracy therefore uses the unchanged raw `campose_valid` and finite, aligned reference joints, independently
of benchmark 2D visibility; exact frame/actor and bank/reference parity are enforced. Report the inherited full-target-mask
subset separately for comparability. This changes no GT, benchmark input, historical result or metric formula.
The preflight's `evaluation_GT_unavailable` count is specifically **masked decision support**, not a claim that
the raw 3D reference is unavailable. Continuous prediction geometry was audited on every row.

The first attempted raw-reference audit exposed a W101 helper defect before M_Main: it looked up bank `frame_index`
inside raw `img_frame_ids`. The actual importer enumerates annotation rows (`three_dpw_adapter.py`) and exports that
zero-based row as the image/bank frame index; `img_frame_ids` is a separate metadata identifier (e.g. 0,2,4,... while
bank rows are 0,1,2,...). W102 uses one explicit row contract, with no conditional lookup/fallback. Before accepting it,
reconstruct the original benchmark input/confidence/masks from that raw row **exactly**, and compare its 3D coordinates
with the bank at 1e-6 m on every declared actor-frame. Historical W101 code/tests/verdict remain unchanged. No RGB,
timestamp, observation or target value is rewritten. A fresh raw-reference preflight retains the original numerical
budgets and adds the newly established GT decision support; the initial masked-support audit remains archived.

Runtime subject selection stays production confidence×area at first detection and the existing serial IoU/confidence
tracker thereafter, one stream per video, no GT seed. The first uniquely associated actor per video is an evaluation-only
anchor for describing identity continuity; later changes are reported, not corrected. Paired accuracy evaluates each
uniquely associated actual subject and separately describes anchor-stable rows. This supports the current single-subject
scope, not multi-person reconstruction.

The historical 90% actor-frame rule is unchanged as history, not a universal W102 gate: TEST's one-subject structural
ceiling is 86.49%. Detector coverage, subject continuity, association availability and reconstruction quality each retain
their own denominator. No detector-success row selection or denominator substitution.

### Controlled effect and attribution

B_Main and M_Main use identical Main software, checkpoint, source snapshot, precision, batch ordering, crop and sign
settings. Only `input_2d` / `input_valid` change. M1 is explicitly batch_size=1, unchanged 0.3 detector/visibility policies,
same canonical mapping and exact image bytes. Sequential initialization, no GT runtime construction. M8 is not run.
The Main M1 gate uses W101's fixed 85-frame sample and two M1 runs: report byte equality separately, require identical
presence/visibility/tracking/order, coordinate differences <=0.5 px and confidence differences <=1e-3 (the existing
observation contract scales), and verify every forward batch is one. It verifies the unchanged M1 policy and its
repeatability on Main; it does not rerun M8 or turn historical cross-stack byte identity into a new pass condition.
For privileged candidates, the actual selected pose is reused for the identical selected box; other detected boxes
receive the same batch-one RTMPose call. This changes no observation value or subject-selection policy.

Report paired 2D differences/GT localization, confidence/validity, MPJPE, per-joint/depth/sign/limb/knee/torso errors,
severe tails, temporal events with eligible-step denominators, video/actor/posture/challenge clusters and missingness.
Video-cluster bootstrap: 2,000 replicates, seed 1337, frame-weighted paired differences; keep actors from one video
together. No CI with <2 videos; small-cluster intervals remain descriptive. Metric-specific paired support is explicit.

W100's **+10 mm MPJPE / +2° limb or knee-plane** scales remain the predeclared material conditional source-effect
criteria. Coverage/identity loss is reported separately rather than forcing the impossible historical actor-coverage
criterion. A/B/C/D follow the user's W102 definitions; mixed regimes retain distinct verdicts.

Moa uses privileged benchmark actor association of detected candidates; Mconf substitutes benchmark confidence on
jointly valid joints; Mmask uses benchmark validity only where finite RTMPose coordinates exist. Probes are evaluation-only,
on the actual same-person rows. Multiple changed factors cannot be attributed to one cause without isolating evidence.
Distinguish detection/tracking, localization, confidence/mask shift, representation contract, persistent estimator
limitations, inherent ambiguity and numerical environment. Unresolved causes remain unresolved.

### Supplementary medium checks

The existing production_demo clip is a **historical real-video diagnostic**, with undocumented original URL/usage
rights; it is not independently unseen and will not be redistributed. Inspect motion and visibility before any
supplementary M1→S0 run. No 3D accuracy metric without GT. Fixed review selection: uniform first/middle/last frames,
lowest-validity frames and largest observed 2D limb-direction steps, deterministic earliest tie breaking.

If primary results are interpretable, conditional rig review keeps the existing BaseRig/rest/FK/IK, timestamps,
renderer/camera and common GT-free frozen HMR pelvis/thorax policy. Such a comparison is explicitly mixed-source;
only B_Main vs M_Main lower-body geometry changes. J1X arm, if available, is frozen and shared; otherwise shared
rest arms are labeled. No W96/W99 thigh-roll candidate, smoothing, GT root/pelvis source or public runtime integration.

Before inspecting M_Main: retain historical 36-sample review windows starting at walking
`downtown_walkBridge_01:actor0`/333, seated `flat_guitar_01:actor0`/500,
wrong-knee `downtown_bar_00:actor1`/367 and turn `downtown_cafe_00:actor1`/1021.
Additionally select the available 36-sample same-person window with the largest sum of M−B MPJPE,
and the window with most non-same-person observations (lexicographic sequence/start tie break).
These are diagnostic selections, not population estimates. Sparse bank cadence is retained; no missing
or filtered neighbor is bridged. Windows shorter than 36 samples remain explicitly shorter.

## Execution evidence

### Fixed Main environment and artifacts

`C:\Users\dna10\animcv-w102` is an isolated source/data/result workspace. The existing
`C:\Projects\AnimCV` checkout and its `.venv_build_full` are read-only dependencies; no package or system setting
was changed. All job temp/cache paths are process-local under the W102 user directory. macOS authors code and
reads/copies results; all model inference, tests and evaluation run on Main. No new LabServer63/69 compute job ran.

- OS fingerprint: `Windows-10-10.0.26200-SP0`; Python 3.11.9, MSC 1938 AMD64.
- GPU UUID `GPU-0f0604b6-359b-c9d1-103b-e8da2df73a79`, RTX 5080, SM 12.0, driver 596.49; Torch compiled
  `sm_75,80,86,90,100,120`, CUDA runtime 13.0. Torch 2.13.0+cu130, torchvision 0.28.0, numpy 1.26.4,
  MMPose 1.3.2, MMDet 3.3.0, MMCV 2.1.0, MMEngine 0.10.7.
- S0: width 256, 8 heads, fusion depth 2, FF multiplier 4, 17 joints × 4 geometry features, 7 neutral sign fields,
  no vision; batch 512, AMP off, float32, highest matmul precision, matmul TF32 false, cuDNN TF32 true.
- Environment fingerprint: `cd68f5389a8eb4a01db9ee258ec66d38c638f46fd2b1fb3506cdc85f190b13c9`.
- Source snapshot: `72e0b6a` plus new W102 diagnostic files only; original model/crop/sign/canonical source hashes
  are bound in every S0 report.

| Artifact | SHA-256 / digest |
|---|---|
| Frozen S0 checkpoint | `6c12fea9f84ea0493fada8b9acc96cd5c70fdd5d9114603e954f9412f4247b44` |
| Bank content | `75519e6394a764e3749ddaa30555b58b73a01db582ccee14f661374b9a0ed536` |
| W97 population | `533895c980a080985444b703f43e0f64046391c76909dea5a2f305a26b34cdb9` |
| RTMDet weights | `78e30dcce0c6f594eaff0d6977b84b4103688b4aff0ad1aa16008a8cc854a7fb` |
| RTMPose weights | `77ffc7e802acf10951c353e8bc68b4f05218121177ceaea163aa124436ba6fb7` |
| Frozen contextual HMR2 | `fc5d330565205f4792cd3641671633decbab4e24d70522d4dd85f045d87a7003` |

All 9,251 existing RGB files and 36 existing raw pickles were verified against W101's exact source-byte manifests.
The 814,654,617-byte RGB transfer archive has SHA `d1470d1c6963e8d6b26ad13b37996f8afc2861b9b01babeac0b254feedb4ea69`;
one interrupted upload was resumed and its complete hash checked before unpacking. No dataset was downloaded.

### Repeated B_Main and historical portability

Two independent B_Main processes produced exactly identical prediction arrays/files on both splits. All finite and
shape/identity checks passed. Historical byte identity remains **FAIL on both splits**; W101 CASE D is unchanged.

| Split | Rows | B_Main SHA (both repeats) | Historical SHA | Max raw / root-relative displacement (mm) |
|---|---:|---|---|---:|
| VAL | 3,407 | `f89d9bd071be9eb9e60d8b3dd77eb53db904fc10dfff418242c409cb7ee1d27d` | `402d70a2bed79ae31919ff86905e19c8972108ea988d6cfa5deac514562132b9` | 0.001811 / 0.001797 |
| TEST | 7,076 | `065ae4be7c2dc2ea340d0fb3fad54acefe3e2676ba8ff387bbada15d0db134bc` | `4357c7bdb46d81e7712269b08c45f39452d7a46e697a0c26fdb70d65fe1f4996` | 0.002108 / 0.002026 |

These are prediction-to-prediction discrepancies, **not GT accuracy errors**. Maximum limb-direction differences:
VAL 0.003744°, TEST 0.003500°; well-conditioned knee-plane maxima 0.000914° / 0.000783°;
maximum torso-axis differences 0.000683° / 0.002421°. All declared numerical budgets passed without adjustment.
The initial masked-support audit and the later strict-row/raw-GT-support audit are separately archived. The latter
verified raw benchmark inputs/masks on every actor-frame and retained zero discrete changes outside boundary bands.
Near-straight knees and invalid torso geometry remain counted as degenerate; no fake zero-angle result is assigned.

### Main M1 verification and acquisition

The predeclared 85-frame Main M1 repeat gate passed: 74 real images plus 11 controls, all seven cases bitwise equal,
zero coordinate/confidence changes, exact ordering/tracking/visibility. Detector 170 forwards, pose 166 forwards,
every forward batch size one. This establishes Main M1 repeatability, not historical cross-stack bitwise M1 identity.
The full 36-video acquisition then completed under the same configuration, with exact per-image byte cache identities.

### Coverage and actor association — independent denominators

| Quantity | VAL | TEST |
|---|---:|---:|
| Original actor-frames / actor-sequences / videos | 3,407 / 16 / 12 | 7,076 / 37 / 24 |
| Unique images | 3,131 | 6,120 |
| Any detector candidate / selected detection / raw landmarks present | 3,131 / 3,131 / 3,131 | 6,113 / 6,113 / 6,113 |
| At least one valid canonical joint | 3,131 | 6,110 |
| Uniquely associated selected image | 2,745 | 4,243 |
| Multiple plausible / no association / insufficient shared / missing subject images | 1 / 385 / 0 / 0 | 0 / 1,843 / 27 / 7 |
| First-associated-anchor stable images | 2,712 | 3,697 |
| Original same-person actor-frame share | 80.57% | 59.96% |
| Other-person actor-frames | 275 | 882 |
| Ambiguous / unresolved / unmatchable / missing actor-frames | 2 / 385 / 0 / 0 | 0 / 1,914 / 30 / 7 |
| Privileged Moa candidate actor-frames | 3,404 | 7,060 |
| Moa candidate available where actual selected stream is not same-person | 659 | 2,817 |

Actor status is relative to **all declared actors present in the sparse bank image**, using the unchanged 2D
association proxy. A uniquely associated image contributes one same-person actor record. Multiple plausible
matches are not asserted identity errors; unresolved localization can be the wrong/background person or landmark
disagreement. Neither is automatically proof of a detector identity error.

There were 3 uniquely associated actor transitions among 2,361 eligible neighboring VAL observations, and 3 among
3,089 TEST neighbors. No unresolved gap is bridged. TEST's 7 missing detections form four runs in `downtown_stairs_00`
(sample frames 606–611, 656, 705–715, 1048); VAL has none. Exact per-video adjacent box IoUs and separate
reacquisition IoUs remain in `handoff/coverage.json`. A low switch count is conditional on interpretable neighboring
associations and does not establish uninterrupted identity through unresolved intervals.

Near-complete image detection is therefore not near-complete declared-actor association. The privileged Moa
coverage is evaluation-only, may select different candidates for different actors in one image, and is not a
multi-person runtime result. Original 90% actor-frame criterion remains historical and is not passed/redefined here.

Observation handoff SHA `7fc15fc8a575358b71fdff1c5e61470e92dc51028a98442b273d09b55bd5a0c2`;
sidecar SHA `d0e3a8796d8e4aad08df311b2ce7da818be46d4f7f6e287371c771845ba1b276`;
full source manifest SHA `6492b3fcf58cc434454e2bb74beb6b637c67d9e21f0169d3c39a773e53f9fb65`.
Every original actor-frame remains in these artifacts, including missing and nonassociated inputs.

### Historical real-video supplement

The unchanged private clip SHA is `0dda8a5f16c67992330a1a8534cf0bfefeb5e0f2aec438577a9e1dbace773275`.
Pre-inference inventory showed visible full-body walking and a turn, with changing facing/limb overlap; it contains
little posture diversity and no seated/lying stress case. Source URL/rights remain undocumented; not an unseen holdout.

Main ran M1→same frozen S0 for all **199 frames at 50 fps**, using exact decoded BGR-byte hashes and original times.
All 199 had a selected subject; 161 frames had 17 valid joints and 38 had 16. The first/middle/last, lowest validity
and largest observed 2D-step selection produced frames **0, 99, 109, 118, 198**. No frame was removed from inference.
Observation/prediction archive SHA `6784357f7c6145e4b60e2e9201e32accd3ae00001486283a1e76f0adae61e7bd`.

The largest observed 2D limb-direction step was 18.62°. Across 1,584 predicted adjacent limb directions, 4 steps
exceeded 45° (maximum 108.14°). These are **unconditioned output changes**, not GT-verified error events. The side
panels expose depth/articulation ambiguity through the turn, including arm-depth behavior; no 3D accuracy is claimed.
Only private review composites/images were copied to `user_QE/animcv-output/102_same_environment_estimator/`;
the raw clip is not redistributed or committed. This supplement does not enter the benchmark verdict/statistics.

### Primary paired reconstruction

Regimes throughout: **B_Main = benchmark_detector_observation; M_Main = real_animcv_observation, M1 batch=1**.
Raw `campose_valid`, finite GT and strict row/bank parity were verified for **all 10,483 actor-frames**; no raw GT
row was unavailable. The inherited full-target mask would retain only 2,097 VAL / 4,240 TEST all-actor rows.
Uniquely associated paired accuracy retains 2,745 VAL actor-frames from 12 videos and 4,243 TEST actor-frames
from **23** videos. `downtown_downstairs_00` has no uniquely associated selected frame and remains in coverage,
Moa and all-actor accounting; it does not silently count as a paired-accuracy video.

All intervals below resample **source videos**, retaining their actors together: 2,000 replicates, seed 1337,
frame-weighted paired differences. They are not frame-independent intervals. Single-video summaries have no CI.

| Paired quantity | VAL | TEST |
|---|---:|---:|
| MPJPE B_Main → M_Main (mm) | 92.35 → 79.36 | 104.77 → 89.65 |
| M−B MPJPE, 95% video CI (mm) | −12.99 [−19.79, −7.38] | −15.13 [−21.00, −8.81] |
| M−B thigh direction, bilateral mean (°) | −1.10 [−2.17, +0.26] | +0.72 [−0.65, +2.38] |
| M−B shin direction, bilateral mean (°) | −1.05 [−1.93, −0.20] | −2.84 [−5.88, −0.03] |
| M−B upper-arm direction (°) | −2.21 [−5.96, −0.37] | −2.80 [−3.75, −1.83] |
| M−B forearm direction (°) | −2.16 [−4.42, −0.60] | −4.93 [−6.71, −3.15] |
| M−B knee-plane, identical eligible limbs (°) | −2.10 [−3.59, +0.64], n=1,862 | −0.12 [−3.17, +3.33], n=2,287 |
| M−B torso forward (°) | −1.66 [−2.52, −0.78], n=2,745 | −2.08 [−2.84, −1.30], n=4,236 |
| Limb-depth sign error, B → M (eligible limb-weighted) | 16.71% → 13.68% | 19.89% → 16.29% |
| MPJPE >200 mm, B → M | 116/2,745 → 35/2,745 | 238/4,243 → 115/4,243 |
| MPJPE p95 / p99, B → M (mm) | 190.97/276.73 → 153.08/207.08 | 205.07/275.41 → 172.58/237.67 |

Knee-plane support requires both predictions and GT to satisfy the historical bent-knee geometry contract; torso
support also remains conditional. The paired knee-plane CI, especially TEST, does **not establish strict +2°
noninferiority**. Aggregate compatibility is not a claim that every limb, stratum or sequence passes.
Anchor-stable paired MPJPE differences are −12.35 mm VAL / −16.57 mm TEST, also with video CIs excluding zero.

Per-joint root-relative errors (mm; same-person population, all raw GT joints; complete precision in JSON):

| Joint | VAL B | VAL M | TEST B | TEST M |
|---|---:|---:|---:|---:|
| pelvis | 0.0 | 0.0 | 0.0 | 0.0 |
| left hip | 22.0 | 20.5 | 23.3 | 22.7 |
| left knee | 79.5 | 78.7 | 99.7 | 104.9 |
| left ankle | 133.1 | 127.4 | 186.5 | 177.8 |
| right hip | 19.5 | 19.5 | 26.8 | 26.7 |
| right knee | 87.7 | 81.2 | 93.9 | 94.1 |
| right ankle | 149.9 | 136.3 | 179.8 | 164.1 |
| spine | 29.9 | 27.5 | 36.3 | 30.4 |
| thorax | 46.0 | 43.5 | 56.4 | 48.9 |
| neck | 76.4 | 68.1 | 86.8 | 72.3 |
| head | 146.4 | 82.4 | 151.9 | 86.4 |
| left shoulder | 74.3 | 67.4 | 83.5 | 80.1 |
| left elbow | 114.0 | 96.0 | 146.2 | 110.3 |
| left wrist | 168.4 | 150.1 | 203.3 | 166.9 |
| right shoulder | 74.9 | 69.5 | 86.3 | 75.4 |
| right elbow | 137.7 | 108.6 | 122.8 | 96.9 |
| right wrist | 210.3 | 172.5 | 197.8 | 166.1 |

On the inherited **all-target-visible** same-person subset, B→M is 63.54→66.02 mm VAL (n=1,654) and
77.43→79.93 mm TEST (n=2,392). Paired differences +2.47 [−0.41,+4.99] / +2.50 [−1.07,+7.34] mm are below the
predeclared +10 mm material scale but do not prove zero source effect. Excluding partially observed benchmark
frames would conceal the population where the larger net M1 benefit occurs; it is not a pure GT-availability gate.

All-actor **diagnostic** MPJPE is B 90.48 / M 100.12 mm VAL and B 105.12 / M 132.92 mm TEST. This deliberately
includes comparisons of a selected person with other actors and dense S0 output from invalid inputs. It accounts
for errors of the declared one-stream handoff; it is neither same-person accuracy nor successful observation coverage.

### Observation-domain attribution and probes

On identical valid limb joints, B/M coordinate differences average 13.78 px VAL / 12.91 px TEST, with medians
10.05 / 8.95 px and p95 30.69 / 35.94 px. Against the same projected SMPL reference, paired normalized 2D error
is **higher** for M: +0.00358 [0.00245,0.00443] / +0.00369 [0.00274,0.00464] of the GT limb-box diagonal.
Paired PCK@0.05 differences are −1.62 / −2.88 percentage points. Better mean S0 3D output therefore cannot be
attributed to uniformly more accurate shared 2D landmarks. COCO/SMPL joint semantics and residual localization
differences are not separated further by this experiment.

Jointly valid confidence means B/M are 0.800/0.756 VAL and 0.747/0.731 TEST; complete quantiles, including finite
scores above one, are preserved without clipping. Across all 17 canonical joints on same-person frames:

| Validity support | VAL joint observations | TEST joint observations |
|---|---:|---:|
| Both valid | 44,407 | 67,533 |
| B-only | 163 | 556 |
| M-only | 1,738 | 2,925 |
| Neither | 357 | 1,117 |

Mconf−M MPJPE is +0.23 [−1.44,+2.53] mm VAL / +0.85 [−0.07,+1.74] mm TEST: replacing common confidence
alone does not remove the observed advantage. Mmask−M is **+15.48 [10.57,21.59] / +15.69 [10.34,20.74] mm**;
Mmask reaches 94.84 / 105.34 mm, close to B's full-population conditional means. This supports sensitivity to
availability and its induced crop geometry as a major contributor to the net benefit. Mmask also changes which
finite low-confidence coordinates enter the crop, so this is not unique attribution to one missing joint or confidence
calibration. It is an evaluation-only representation intervention; no runtime mask replacement is proposed/applied.

Privileged Moa on its available actor rows improves over the matched B by −11.07 [−16.63,−6.34] mm VAL (3,404 rows,
12 videos) / −14.17 [−19.63,−7.98] mm TEST (7,060 rows, 24 videos). Together with the candidate-availability counts,
this supports selected-subject association as a major end-to-end limiter even though detector coverage is high.
Moa may choose another box/crop associated with the same actor; it is not a pure identity-label intervention or an
operational multi-person solution. All association judgments retain the benchmark-observation proxy's uncertainty.

The runtime mapping, normalization, byte identity and GT-isolation tests passed; no new representation-contract
defect was demonstrated. The evaluation-only raw-frame lookup defect was corrected in W102 before M inference,
with full original-input/GT parity, rather than changing an observation or relaxing an identity check.

### Temporal and residual failures

Original-neighbor, same-person, **common-support** events; GT eligibility is evaluation-only:

| Event | VAL B → M / eligible | TEST B → M / eligible |
|---|---:|---:|
| Limb direction >45°, GT step <15° | 840→440 / 18,762 | 1,383→841 / 27,525 |
| Knee-plane >90°, GT normal step <45° | 174→68 / 5,409 | 391→253 / 8,145 |
| Torso forward >90°, GT step <45° | 29→1 / 2,719 | 74→3 / 4,166 |

All six video-cluster event-rate-difference CIs exclude zero in M's improvement direction. These are sparse-bank
steps, not dense 30-fps motion-quality estimates. Missing/ambiguous/other-person frames never create a new neighbor.

Improvement is not uniform. Mean same-person M−B MPJPE is +3.50 mm in VAL `courtyard_rangeOfMotions_01`;
TEST has `downtown_cafe_00` **+22.70 mm**, `flat_guitar_01` +7.06, `outdoors_fencing_01` +4.45 and
`downtown_stairs_00` +1.69. These are descriptive per-video clusters, not independent frame-level significance tests.
The worst paired TEST frame, cafe actor1/690, changes 211.02→530.58 mm despite a qualifying association distance
0.0338. Overall tails improve, but the maximum error increases and important new local failure remains.

TEST left-knee common-support wrong-branch events are B 106→M 126 /1,409 (+1.42 pp; CI [−4.56,+8.62]); shared
wrong branch occurs on 44 left and 29 right common eligible frames. Both sources exceed 200 mm on 24 VAL / 57
TEST same-person frames. These are persistent failures under two noisy observation sources, not proof that either
source is an oracle or that ambiguity is irreducible. TEST lying has only **3 same-person frames from 2 videos**;
its +17.54 mm difference is too narrow a population to establish general lying capability. Waist-bend/other posture
intervals and challenge subsets remain in JSON; no broad posture/generalization success is asserted.

Frozen HMR2 contextual MPJPE is 64.68 / 69.05 mm on these same-person rows. Its model, RGB information and
training differ, so this is context for remaining reconstruction error, not a controlled claim that RGB alone caused
the gap. Runtime subject association, observation shifts, shared estimator failures and inherent ambiguity remain
distinct evidence categories.

### Conditional BaseRig evidence

Main Blender **5.1.0** reused the historical Workbench renderer, BaseRig skinned equivalent, rest snapshot, W95
two-bone IK, zero-twist locals and cameras (480 px, orthographic scale 3.2). No W96/W99 roll candidate or smoothing.
Existing W87 J1X was found by read-only inventory on LabServer63 and copied unchanged to Main; no model/readout
was regenerated. Its SHA `cb9dc51e3f84ddc9b45c5ada5e941ceaaad5ee73440a6e387ee0c3f2b689c46f` and reported bank
digest match the frozen bank. Every frame shares the same **GT-free frozen HMR pelvis/thorax and J1X left arm**;
right arm stays at rest. Only S0 B_Main versus M_Main legs change. This is explicitly a **mixed-source diagnostic**,
not a complete real-observation runtime or a full-body source comparison.

The four fixed 36-sample windows and the predeclared additional rules produced six cases (216 observations):
walking, seated, wrong-knee, turn, maximum source regression (cafe actor1/685), and association failure
(courtyard_dancing actor0/0). All six windows have available leg/J1X exports; the association-failure case is
`other_person` on every sample and is labeled, not counted as valid same-person accuracy.

Object/spine/J1X channels are exactly shared between modes. Imported rest difference is 6.104e−5 armature units,
within the declared 1e−4 equivalent-rig binding. Blender bone tails versus exported leg points differ by at most
6.837e−5 armature units; shared-arm endpoint versus export at most 1.036e−4. These are FK implementation checks,
not reconstruction-accuracy numbers. Original frame indices/timestamps and CONSTANT interpolation are retained;
review videos hold sparse samples at their original 30-fps positions, with no interpolated motion claim.

Visual review exposes meaningful knee/shin differences in walking and wrong-knee, while the cafe
regression remains visibly folded/incorrect under both sources and changes articulation substantially in M.
The sitting case is not made correct merely by lower aggregate joint error. Parent/arm agreement confirms that
these differences are not a changed pelvis, thorax, arm policy or camera.

The existing W95 severe local-thigh step (>90° while GT thigh step <45°, original same-person neighbors) occurs
12 B / 10 M times among 70 eligible leg steps in the wrong-knee window, and 0 B / 2 M among 70 in the cafe
maximum-regression window. Walking/seated/turn have no such event in their selected windows; walking's maximum
local-thigh step is 77.84° B / 23.55° M. These selected-window counts are descriptive, not population acceptance rates.

The existing IK transfers endpoint direction/reach fraction to the target length ratio; it does **not** preserve each
source segment direction exactly. At cafe actor1/710, M source upper/lower lengths 0.371/0.176 m (ratio 2.108)
with knee 170.52° map to target ratio 1.027 and knee 137.24°; segment direction discrepancy reaches 91.63°.
Walking B similarly has a 70.49° source-to-rig discrepancy. Both worst examples use observed bend side and
**no reach clamp**. Thus inconsistent/folded S0 chain geometry interacts with the preserved retargeting contract;
large rig artifacts are not a Blender FK failure or uniquely a new M1 defect. No solver was patched to improve them.

A separate **post-hoc display-only probe**, five observations before the first missing M1 subject and the following
31 samples in stairs actor0, verifies actual missing-frame presentation. It is excluded from the six predeclared
selections and primary statistics. M panels are blank and explicitly UNAVAILABLE at absent observations; no dense
pose from all-invalid input, held key, GT or alternate source is presented as a valid M reconstruction. This tests
diagnostic presentation, not an integrated runtime missing-frame policy.

All completed review images/videos are local under `user_QE/animcv-output/102_same_environment_estimator/`.
Machine JSON/NPZ and `.blend` Actions stay on Main. No style-authoring capability or production animation readiness
is claimed by this experiment.

### Executed checks, failures and omissions

- Main: two independent CUDA B_Main repeats, plus B_Main control in the B/M/probe run; both splits exactly equal
  to their within-Main locks. Historical SHA remains FAIL; both numerical audits passed unchanged budgets.
- Main: M1 85-frame repeat/order/visibility/batch instrumentation; all 9,251 image identities, 36 raw annotation
  identities, 10,483 actor/frame/timestamp rows and original benchmark 2D/confidence/masks validated.
- Main combined focused pytest run: **47 passed in 2.03 s** (15 then-current W102, 19 W101, 6 W100,
  4 MMPose adapter, 3 subject tracker). W101's checkpoint-backed GT-perturbation test ran, not skipped.
- After the real raw-reference contract correction and added row fixture: **16 W102 tests passed in 0.35 s**.
  This overlaps the combined run; do not add the counts. Final shared-arm/rotational-covariance test passed after
  J1X reuse and again after import separation (1 test each, 0.25/0.24 s). Actual same-environment repeats,
  full-population identity audits and Blender exports are additional measured checks, not invented pytest cases.
- Main Blender: six 36-sample exports plus one 36-sample post-hoc missing display probe; rest binding and FK
  residuals recorded. GT is absent from the `pose_export`/`arm_export` calculation boundary; tests exercise
  shared parent, rotation covariance, finite output and unavailable-source behavior. Selection/scoring uses GT
  only outside that boundary. Full exported-builder GT-perturbation was not separately executed.
- Broad regression, M8 rerun, training, new readout/feature generation, new LabServer63/69 experiment, production
  integration and independent unseen-video/3D accuracy validation were **not executed**.

Execution failures are retained, not counted as passes: an initial launcher `Path`/`runpy` argument error occurred
before inference; a focused command named nonexistent test files and collected zero tests; a rig fixture first lacked
deployed rest assets, then exposed an incorrect test assumption of exact source-segment preservation (the preserved
IK transfers reach fraction/target ratio). The contract test was corrected to rotation covariance/shared-parent
invariants, while the actual large source-to-rig discrepancies are reported above. A new overlap statistic initially
failed for an empty float array and was corrected to an explicit boolean empty mask. The first raw-GT audit rejected
W101's metadata-ID lookup; the explicit importer-row contract was verified instead, without modifying historical code.

The first Blender wrapper imported evaluation modules requiring OpenCV unavailable inside Blender; no render ran.
Only W102 imports were separated by mode. That Blender invocation returned zero despite a Python exception,
so its incomplete compositor attempt is preserved separately. The successful runs use `--python-exit-code 1` and
require `blender_report.json` before composition. No Blender/package installation or numerical solver patch was used.

### Reproduction and artifact layout

Main Python: `C:\Projects\AnimCV\.venv_build_full\Scripts\python.exe`.
Set `$b="$env:USERPROFILE\animcv-w102"`, `$w="$env:USERPROFILE\animcv-w101\data"` in PowerShell.
Every Python command below runs through `& $python -X utf8 "$b\setup\w102_main_launch.py" <script> ...`;
the checked-in `scripts/w102_main_launch.py` is the exact process-local launcher. Full argv/model/environment
identities are also in each S0 report. Paths below are relative to `$b` except the frozen bank/checkpoint in `$w`.

1. `w102_observations.py gate --population data/population.json --image-root data/images --weights data/models
   --out-dir results/m1_gate`; then `observe` with the same inputs plus
   `--gate-report results/m1_gate/gate_report.json --image-manifest data/w101_transfer_image_sha256.json
   --out-dir results/observe`.
2. `run_w102_same_environment_s0.py --bank $w/bank_3dpw_paired_v2.json --checkpoint $w/s0_checkpoint.pt
   --conditions B_Main --out-dir results/baseline0`; repeat in a separate process as `baseline1` with
   `--lock results/baseline0/s0_report.json` (same inference settings, no source candidate).
3. `w102_numerical_audit.py --bank $w/bank_3dpw_paired_v2.json --baseline0 results/baseline0
   --baseline1 results/baseline1 --historical-dir data --m1-gate results/m1_gate/gate_report.json
   --raw-3dpw data/raw --out results/semantic_preflight_row_contract.json`.
4. `w102_evaluate.py build --bank $w/bank_3dpw_paired_v2.json --population data/population.json
   --observe-dir results/observe --image-root data/images --out-dir results/handoff`.
5. `run_w102_same_environment_s0.py --bank $w/bank_3dpw_paired_v2.json --checkpoint $w/s0_checkpoint.pt
   --observations results/handoff/observations.npz --lock results/baseline0/s0_report.json
   --preflight results/semantic_preflight_row_contract.json --conditions B_Main M_Main Moa_Main Mconf_Main Mmask_Main
   --out-dir results/s0_sources`.
6. `w102_evaluate.py evaluate --bank $w/bank_3dpw_paired_v2.json --population data/population.json
   --observe-dir results/observe --observations results/handoff/observations.npz --s0-dir results/s0_sources
   --raw-3dpw data/raw --preflight results/semantic_preflight_row_contract.json
   --hmr-predictions data/hmr_pred_full.npz --out-dir results/evaluation`.
7. `w102_video_diagnostic.py inspect --clip data/historical_source.mp4 --out-dir results/video_inspection`;
   after visual inspection, `run` with `--inspection results/video_inspection/inspection.json --weights data/models
   --checkpoint $w/s0_checkpoint.pt --gate-report results/m1_gate/gate_report.json
   --lock results/baseline0/s0_report.json --out-dir results/historical_video`.
8. `w102_rig_review.py build --bank $w/bank_3dpw_paired_v2.json --rows results/evaluation/rows.json
   --hmr-predictions data/hmr_pred_full.npz --j1x data/w102_j1x_predictions_3dpw.npz
   --rest code/examples/e2e_demo/baserig_blender_rest.json --calibration code/examples/e2e_demo/framepose_fk_calibration.json
   --chains code/examples/e2e_demo/framepose_ik_chains.json --out-dir results/rig`.
   The missing probe adds `--missing-only` with fresh output directories.
9. Main Blender 5.1 executable: `-b -t 2 --python-exit-code 1 -P code/scripts/w102_rig_review.py -- render
   --rig code/examples/skinned_mannequin/BaseRig_Skinned.fbx --rest code/examples/e2e_demo/baserig_blender_rest.json
   --cases results/rig/cases.json --out-dir results/rig_renders_complete --show-mesh --size 480 --ortho-scale 3.2
   --equivalent-rig-tolerance 0.0001`; then native Python `w102_rig_review.py compose --cases results/rig/cases.json
   --image-root data/images --render-dir results/rig_renders_complete --out-dir results/rig_review_complete`.

Recorded commands use absolute Windows paths in the actual invocations; substitute the shown `$b`/`$w` prefixes
when reproducing. Scripts refuse to overwrite existing output directories/audits; use fresh destinations for repeats.
Tests ran with `PYTHONPATH=code/src;code/scripts`, `W101_CHECKPOINT=$w/s0_checkpoint.pt`, and process-local temp
paths; JUnit XML/logs are under `results/`. Read-only rig attribution/temporal scripts are preserved in `setup/`.

New task files only: eight W102 scripts (runner, launcher, sensor, reference, numerical audit, evaluator, private
video diagnostic, rig review), one focused test file and this worklog. All `src/`, prior worklogs/scripts, checkpoint,
dataset, rig/rest/FK/IK and historical outputs remain unchanged. The Main source snapshot is isolated; the existing
Main checkout was not switched or synchronized over its unrelated state.

Recorded combined test command was native Python `-X utf8 -m pytest code/tests/test_w102_same_environment.py
code/tests/test_w101_reference_estimator.py code/tests/test_real_observation_estimator.py
code/tests/test_mmpose_adapter.py code/tests/test_subject_tracker.py -q
--junitxml=results/w102_focused_complete.xml`, with the then-current 15 W102 cases. The later 16-case run used
`code/tests/test_w102_same_environment.py -q --junitxml=results/w102_row_contract_tests.xml`;
the final focused rig node was `test_w102_same_environment.py::test_rig_lower_body_change_preserves_common_parent_and_finite_chain`.
These relative names denote the actual absolute `$b`-prefixed arguments, with output paths shown above.

Final evidence remains on Main at `C:\Users\dna10\animcv-w102\w102_same_environment_evidence.zip`:
**304,330,506 bytes**, SHA-256 `f8ecc83e49194991676376a7bc60067c0c728c4f77b7b5eeedad8f6416ec34a7`.
The separate manifest SHA is `5186a17075a1fa58382ae0c9a5233d788b8b6cda45ee435fa2554f39e1489e9f`.
All 1,242 listed evidence files were read back from the ZIP and verified against SHA-256; eight completed review
videos also passed container frame-count/dimension/fps checks. This packages completed/failed logs distinctly,
machine outputs, source snapshots and setup utilities; raw RGB images and the raw historical source clip are excluded.
The immutable archive contains the premeasurement declaration. This final narrative is published in Git separately
to avoid a self-referential archive hash; Main's `publication_receipt.json` binds the final Git commit to the archive.
