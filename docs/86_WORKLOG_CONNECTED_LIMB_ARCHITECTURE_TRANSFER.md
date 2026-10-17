# Worklog — Connected-Limb Architecture Transfer Feasibility (2026-10-17)

**Question.** Does the J1X-style single-frame RGB connected-limb architecture give meaningful reusable 3D geometry beyond the left arm, for both the symmetric right arm and an anatomically different leg?

**Answer: partially. CASE B (partial transfer) by the predeclared rule. Whole-body expansion STOPS here (§16 fallback).**

| chain | 2D-only control G0 vs RGB candidate G1 (same rows, init, draws and contract) | transfers? |
|---|---|---|
| **right arm** | distal (wrist) G1 − G0: **3DHP −6.35 %L [−10.46, −3.37], 6/6 sequences**; 3DPW −1.58 [−3.13, +0.07], 24/37. Middle, proximal, chain and bend all improve on 3DHP, as does out-of-plane wrist (−17.1). | **yes** (strong on 3DHP; no regression, tail or temporal failure) |
| **right leg** | distal (ankle) G1 − G0: 3DPW −0.09 [−1.24, +1.23]; 3DHP −1.31 [−2.57, +0.39], 4/6, one sequence 58% of the gain. The knee improves significantly on both populations (3DPW −0.92 [−2.24, −0.01]; 3DHP −1.65 [−2.59, −0.82], 6/6), as do thigh direction and bend side. | **no** (the predeclared ankle criterion is not met) |
| left arm (reference: J1X vs matched G0X) | 3DHP −6.46 [−9.29, −3.87], 6/6; 3DPW −2.01 [−4.25, +0.10] | yes (calibration only) |

%L = percent of the GT total chain length. Intervals are sequence-cluster 95% bootstrap intervals.

- **The right arm reproduces the left-arm pattern almost exactly.** The bilateral abstraction transfers, with separate weights and direct right-side inputs.
- **The leg does not show a meaningful distal benefit,** for a measured reason:
  - RGB gives large gains where limb depth is ambiguous: out-of-plane, strongly bent and near-folded configurations.
  - RGB costs accuracy near the image plane (false depth).
  - Legs are mostly near-plane: out-of-plane rows are 8% of 3DPW leg rows vs 21% for the right arm, and 14% vs 29% on 3DHP. So the near-plane penalty cancels the gain at the ankle.
- **The same near-plane false-depth penalty is already present in J1X.** Near-plane wrist on 3DPW: +3.1 %L for the left arm, +2.9 for the right arm, +1.5 for the leg.
- **This is not a representation failure.** The two-segment chain is reconstructed validly. It is the existing single-frame accuracy bottleneck, exposed by a chain whose pose distribution is dominated by that regime.
- **Recommended next milestone (not started):** a bounded single-frame accuracy study of near-plane false depth in the RGB-conditioned connected limb.

The batch does not claim knee or ankle rig animation, foot orientation, contact or root translation. Both test populations are developmental, not untouched final validation sets.

## Starting state and files

- **Start:** `arch/single_frame_first` at `bb7ec38` (Worklog 85 docs). Local, origin and LabServer63 were equal, and the local worktree was clean.
- **Toolchain note.** `/usr/bin/git` was blocked by an un-accepted Xcode licence after a macOS update. All git operations used `/Library/Developer/CommandLineTools/usr/bin/git` (Apple Git 2.50.1), which is the same repository with no other change.
- **Commits:**
  - `034ce6f`: chain contract module, audit, readout builder, training, prediction and evaluation runners, and the predeclared CASE rules, all before any limb model was trained;
  - the commit adding the review renderer and this worklog.
- **Added (diagnostic only, not imported by production):**
  - `src/framepose/connected_limb_transfer.py`
  - `scripts/audit_connected_limb_contracts.py`
  - `scripts/build_connected_limb_readouts.py`
  - `scripts/train_framepose_connected_limbs.py`
  - `scripts/predict_framepose_connected_limbs.py`
  - `scripts/run_framepose_connected_limbs_eval.py`
  - `scripts/review_framepose_connected_limbs.py`
  - `tests/test_framepose_connected_limb_transfer.py`
  - this worklog
- **Preserved:** J1X, R1X and every historical checkpoint, bank, cache and report (hashes re-verified by every runner); the frozen Qwen model; the detector; H0, E1, Root Orientation and BaseRig; production FramePose/FK/IK/Blender.

## 1. Data and joint contract (`contract_audit.json`, sha256 `d4223f7a…`)

| chain-valid rows (GT valid AND detector-visible P, M, D) | left arm | right arm | left leg | right leg |
|---|---:|---:|---:|---:|
| 3DPW TRAIN | 10,356 | 8,744 | 11,021 | 10,808 |
| 3DPW VALIDATION | 3,056 | 2,770 | 3,276 | 3,195 |
| 3DPW TEST | 6,003 | 6,076 | 6,497 | 6,519 |
| BEDLAM TRAIN, historical + Worklog 85 expansion | 17,451 + 57,860 | 17,423 + 57,796 | 20,167 + 66,376 | 20,191 + 66,180 |
| BEDLAM VALIDATION | 1,121 | 1,115 | 981 | 993 |
| MPI-INF-3DHP TEST | 2,570 | 2,367 | 2,656 | 2,586 |
| out-of-plane share, 3DPW TEST / 3DHP (abs(f) ≥ 0.5) | 24% / 29% | 21% / 29% | 10% / 11% | 8% / 14% |
| near-straight share, 3DPW TRAIN (bend < 15°) | 11% | 13% | 44% | 46% |

- **Joint identities.** FrameBank right shoulder/elbow/wrist = 14/15/16 and right hip/knee/ankle = 4/5/6. These come from the detector's COCO-18 joints through the unchanged 3DPW conversion, and from SMPL-X (BEDLAM), SMPL (3DPW) and the 3DHP test joints 2/3/4 and 8/9/10 for GT; Worklog 83 verified the 3DHP mapping by name. Tests check every chain's names, the left/right correspondence and anatomical identity.
- **Detector visibility.** Hips are 100% visible in every bank. That is a population property, not a detector property: rows without a valid pelvis, the hip midpoint, are dropped at bank build. Knees are 93–99% visible and ankles 74–97%; BEDLAM validation ankles are the lowest at 74%.
- **Camera conversion.** Unchanged AnimCV (X right, Y depth, Z up), pelvis-relative; valid for the new joints in all three sources.
- **Train/validation isolation.** Unchanged from Worklog 85: BEDLAM subject split and 3DPW sequence splits.
- **Expansion coverage.** The Worklog 85 expansion was selected on left-arm geometry. For the right arm it still fills every sparse BEDLAM coverage cell (21 → 0 cells below 20 rows). For the legs it leaves 22–25 cells sparse (from 36–37), and 2.4–2.8% of 3DHP leg rows still fall in sparse cells. The leg's support is measurably thinner in rare configurations.
- **Leg choice** (`LEG_RULE`, data-only, in the audit script before any limb model): the larger minimum chain-valid fraction over the seven populations is the right leg's 0.747 vs the left leg's 0.738 (a 0.9-point margin, above the 0.5-point tie rule). **Right leg selected.** The opposite leg was not implemented.

## 2. Representation and inputs (declared before training)

- **Unchanged J1X model family:** `build_joint_arm` with 317,511 parameters for every candidate. The output is direction u, direction v and ratio ρ, giving P = 0, M = ρu, D = M + (1 − ρ)v. It is a unit connected chain with no bend-direction assumption, so a knee uses it unchanged. No anatomical adaptation was required.
- **Inputs, per chain:**
  - the Worklog 69 pair features (P–M, M–D, P–D; 13 each) of the chain's detector joints from the H0 crop-normalized geometry;
  - the frozen-Qwen readout at the chain's detector P, M, D positions plus the stored global mean (G1), or exact zeros (G0).
- **Bilateral choice:** direct right-side conditioning. Separate weights are trained on right-chain inputs and targets in the unmirrored camera frame. There is no reflection normalization, learned symmetry module or extra capacity. Qwen features are not assumed reflection-equivariant.
- **Equivalence with the historical left arm.** The chain code with the left arm reproduces `normalized_arm_targets`, `pair_geometry` and the Worklog 68 `readout` exactly (unit tests). It also retrains J1X **bit-exactly** through the full chain pipeline: epoch 129, state identical.
- **Frozen visual provenance.** No Qwen forward pass was run. Readouts are read from each bank's cached Worklog 68 token grid (fp16, (N, 196, 4096)) and its stored crop box. Recomputing the left arm this way equals the stored readouts **bit-exactly in all five banks** (max difference 0.0), and the global readout is identical (`readouts/readout_report.json` `cf942b22…`).

## 3. Training (once per chain and conditioning; `training/training_report.json` `e4da198b…`)

Contract (J1X, unchanged):
- 3DPW TRAIN + BEDLAM TRAIN (historical + Worklog 85), 50/50 over chain-valid rows;
- 11,334 draws per epoch, 200 epochs, AdamW 3e-4 → 1e-5 cosine, batch 256, seed 1337;
- selection by lowest 3DPW VALIDATION chain objective (middle + distal normalized error).

G0 and G1 of a chain share initialization digest, parameter count, steps and per-source draws (control identity verified).

| candidate | selected epoch | 3DPW validation objective | BEDLAM validation objective | train objective |
|---|---:|---:|---:|---:|
| left arm G0X (matched control of J1X) | 119 | 0.3334 | 0.4519 | 0.4944 |
| left arm J1X (Worklog 85) | 129 | 0.2585 | 0.3790 | 0.3403 |
| right arm G0 | 189 | 0.3322 | 0.4449 | 0.4878 |
| right arm G1 | 129 | **0.2736** | **0.3819** | 0.3446 |
| right leg G0 | 99 | 0.2420 | 0.2966 | 0.2865 |
| right leg G1 | 119 | **0.1831** | **0.2379** | 0.1919 |

Checkpoints:

| candidate | sha256 |
|---|---|
| left_arm_g0 | `a7aecc6e…` |
| right_arm_g0 | `ae1ccea9…` |
| right_arm_g1 | `e7f220d6…` |
| right_leg_g0 | `5186a471…` |
| right_leg_g1 | `25b660c4…` |

- **Validation favours G1 for every chain.** For the leg the validation gain (−24%) comes mostly from the knee.
- **The leg's 3DPW validation gain does not carry over to 3DPW TEST at the ankle.** See §5.

## 4. Gates

| gate | result |
|---|---|
| frozen artifacts: J1X, Worklog 80/85 predictions, banks, caches | hashes equal pinned values |
| generic chain pipeline retrains J1X | bit-exact (epoch 129) |
| left-arm J0 / J1X outputs vs Worklog 85 predictions (3DPW, 3DHP) | 0.0 |
| connected-chain contract, every candidate, both populations | ≤ 4.4e-16, all finite, identical coverage |

## 5. Static geometry

Format: mean / p95. M = middle joint, D = distal joint; directions in degrees.

**Right arm:**

| population | candidate | M %L | D %L | upper ° | forearm ° | chain ° | bend ° | side ° |
|---|---|---|---|---|---|---|---|---|
| 3DPW | G0 | 11.8 / 30.0 | 21.1 / 52.6 | 13.6 | 16.7 | 14.1 | 12.0 | 14.8 |
| 3DPW | G1 | **10.7** / 25.8 | **19.5** / 47.8 | **12.3** | 16.6 | **12.8** | 12.6 | 16.1 |
| 3DHP | G0 | 23.0 / 56.9 | 32.6 / 90.9 | 23.7 | 28.2 | 22.1 | 22.0 | 33.7 |
| 3DHP | G1 | **19.4** / 41.3 | **26.3** / 69.8 | **19.6** | **24.9** | **17.6** | **16.0** | 35.3 |

**Right leg:**

| population | candidate | M %L | D %L | thigh ° | shank ° | chain ° | bend ° | side ° |
|---|---|---|---|---|---|---|---|---|
| 3DPW | G0 | 9.4 / 23.7 | 14.0 / 33.6 | 11.3 | 11.7 | 8.1 | 9.7 | 29.3 |
| 3DPW | G1 | **8.4** / 20.7 | 13.9 / 33.6 | **10.1** | 10.6 | 8.3 | 9.2 | **24.3** |
| 3DHP | G0 | 19.0 / 41.2 | 19.9 / 52.5 | 19.0 | 17.7 | 12.4 | 14.6 | 37.3 |
| 3DHP | G1 | **17.3** / 34.9 | 18.5 / 51.1 | **16.7** | 16.0 | 11.7 | **12.5** | **29.3** |

**Paired G1 − G0, sequence-cluster interval (sequences improved):**

| metric | right arm 3DPW | right arm 3DHP | right leg 3DPW | right leg 3DHP |
|---|---|---|---|---|
| **distal** | −1.58 [−3.13, +0.07] 24/37 | **−6.35 [−10.46, −3.37] 6/6** | −0.09 [−1.24, +1.23] 22/37 | −1.31 [−2.57, +0.39] 4/6 |
| middle | **−1.12 [−2.20, −0.09]** | **−3.60 [−6.84, −1.97] 6/6** | **−0.92 [−2.24, −0.01]** 23/37 | **−1.65 [−2.59, −0.82] 6/6** |
| proximal segment | **−1.28** | **−4.16 6/6** | **−1.17** | **−2.21 5/6** |
| distal segment | −0.07 | **−3.37** | −1.10 | −1.66 [−3.28, +0.03] |
| chain | **−1.24** | **−4.53** | +0.18 | −0.64 |
| bend angle | +0.61 | **−6.03 6/6** | −0.51 | **−2.13 6/6** |
| out-of-plane distal | **−7.11** 33/37 | **−17.13** 5/5 | −2.18 [−6.26, +0.32] | **−4.50 [−7.24, −2.10]** |

- **Validity and tails:**
  - connected-chain contract ≤ 4.4e-16 everywhere; coverage identical within each chain;
  - right arm: p95 distal improves on both populations (3DPW 52.6 → 47.8; 3DHP 90.9 → 69.8). The 3DPW maximum rises from 132 to 151.
  - right leg: p95 distal is unchanged on 3DPW (33.6 → 33.6) and slightly better on 3DHP (52.5 → 51.1). The 3DHP maximum rises from 122 to 146.
- **Sequence-weighted means agree with frame-weighted means in sign throughout** (`eval/report.json`).

## 6. Temporal results

Pair spacing: 3DPW 1/6 s (5 frames at 30 fps); 3DHP 0.4 s primary and 0.16 s outdoor secondary.

| proximal-segment motion residual (°/pair) | G0 | G1 | G1 − G0 (cluster CI) | amplitude ratio G0 / G1 |
|---|---:|---:|---|---|
| right arm, 3DPW 1/6 s | 6.47 | 8.99 | +2.52 [+2.10, +2.94], 0/37 | 1.09 / 1.52 |
| right arm, 3DHP 0.4 s | 22.5 | 18.6 | −3.91 [−10.05, −0.84], 6/6 | 0.97 / 1.02 |
| right leg, 3DPW 1/6 s | 6.95 | 7.87 | +0.92 [+0.57, +1.28], 7/37 | 0.98 / 1.21 |
| right leg, 3DHP 0.4 s | 14.6 | 13.7 | −0.88 [−1.35, −0.38], 5/6 | 1.29 / 1.28 |
| left arm reference J1X vs G0X, 3DPW | 6.56 | 9.38 | +2.82 [+2.38, +3.25] | 1.02 / 1.51 |
| 3DHP TS5–TS6 0.16 s (descriptive, 2 sequences) | right arm 20.8 → 17.7 | right leg 8.6 → 9.5 | | |

- **3DPW: the RGB candidates over-move.** Their amplitude ratio to GT is 1.2–1.5, so G1 is less stable than the 2D-only control on every chain at 1/6 s.
- **The predeclared limit is calibrated to the accepted left-arm reference:** max(1.5 × 2.82, 2.82 + 1.0) = 4.22°. Right arm (+2.52) and right leg (+0.92) both stay inside it.
- **3DHP 0.4 s:** G1 is steadier for both chains, but at 0.4 s the residual largely tracks static accuracy.
- **Candidate-specific discontinuities:** below 2.4% for every candidate. G1 is higher than G0 on 3DPW (right arm upper 0.72% → 1.67%; leg 0.34% → 0.79%) and lower on 3DHP.

## 7. Cross-chain attribution (regimes, distal G1 − G0 %L)

| regime | left arm 3DPW / 3DHP | right arm 3DPW / 3DHP | right leg 3DPW / 3DHP |
|---|---|---|---|
| near-plane, abs(f) < sin 10° | **+3.11 / +1.22** | **+2.86 / +0.23** | **+1.53 / +0.38** |
| intermediate | −1.69 / −5.61 | −2.55 / −3.55 | −0.81 / −1.56 |
| out-of-plane, abs(f) ≥ 0.5 | −10.14 / −16.55 | −7.11 / −17.13 | −2.18 / −4.50 |
| near-straight | +0.31 / −11.27 | −0.88 / −13.88 | +0.32 / −0.14 |
| strong bend [45, 90) | −4.98 / −5.92 | −4.45 / −5.40 | −0.12 / −2.69 |
| near-folded [90, 180] | +0.85 / −4.86 | −1.60 / −2.37 | −3.34 / −7.00 |

**Right arm.**
- Bilateral anatomy maps cleanly.
- RGB yields new depth information in the same regimes as for the left arm, with similar magnitudes.
- Right-arm-specific errors mirror the left arm's, so there is no right-arm observation or representation problem.
- 3DHP per sequence: TS1 −7.1, TS2 −3.6, TS3 −4.5, TS4 −4.6, TS5 0.0, TS6 −16.9. The largest share is 39%.

**Leg.**
- The two-segment representation remains adequate: valid geometry, and knee, thigh and bend improve.
- RGB does improve knee depth and bend side (3DPW bend side 29.3° → 24.3°; 3DHP 37.3° → 29.3°), and the ankle in bent or out-of-plane configurations.
- Knee bending does not introduce new error types.
- **The ankle's null result is a regime-mix effect:**
  - legs are dominated by near-plane and near-straight poses, where the 2D-only control is already accurate (G0 ankle 14.0 %L vs right wrist 21.1 %L on 3DPW);
  - in those poses RGB injects false depth (+1.5 %L near-plane);
  - out-of-plane and folded legs are rare (8–14% and 6–11% of rows), so their gains (−2.2 to −7.0 %L) do not lift the mean.
- Supervision support is also thinner for legs: 22–25 sparse BEDLAM cells remain, because the expansion was selected on the arm.
- The 3DHP ankle gain is concentrated: TS1 −3.4, TS2 +0.5, TS3 −0.7, TS4 −3.1, TS5 +4.1, TS6 −0.7.

**What dominates.**
- The dominant error mechanism is the **near-plane false depth of the RGB-conditioned connected limb**. It is present in J1X (left arm +3.1 %L near-plane on 3DPW), identical in the right arm, and decisive for the leg.
- It is not a new failure of the leg representation, the detector or the visual readout (readout provenance is bit-exact).
- Thinner rare-pose support is a secondary, plausible contributor that this batch cannot separate.

## 8. Real-scene review (canonical 3D; no Blender, IK, foot-contact or rig claim)

**Panel layout:**
- the exact source image, with the detector chain (runtime input) in white;
- a front view (X × Z) and a side view (depth × up);
- GT green, G0 orange, G1 cyan, with per-candidate errors.

**Location:**
- server: `~/animcv-output/framepose_connected_limbs_review/`;
- local QE copy (images only): `~/animcv-output/86_framepose_connected_limbs/`.

| case | right arm (3DHP) | right leg (3DHP) |
|---|---|---|
| correct reconstruction | TS5 #45 | TS2 #5910 |
| large error (G1 p95) | TS6 #191 | TS1 #2620 |
| strongly out-of-plane | TS4 #3150: elbow 31.3 → 15.9 %L | TS1 #160 |
| near-plane | TS2 #4160 | TS1 #5230 |
| near-straight / extended | TS6 #102 | TS6 #279 (extended knee) |
| strongly bent | TS4 #1290 | TS4 #3970: bent knee, ankle 23.7 → 15.8 %L, knee depth still wrong |
| low-confidence observation | TS2 #5690 | TS2 #740 |
| temporal improvement (strip) | TS6 #100–140 | TS2 #520–540 |
| temporal regression (strip) | TS2 #5440–5460 | TS4 #1190–1220 |

The reviews match the numbers. RGB fixes depth when the limb points toward or away from the camera, and adds small false depth to in-plane limbs.

## 9. Architecture classification — CASE B (partial transfer)

| predeclared condition (`DECISION`, commit 034ce6f) | right arm | right leg |
|---|---|---|
| strong distal gain on ≥ 1 population (interval excludes 0, ≥ 2/3 of sequences, no sequence ≥ 50%) | **3DHP yes** (6/6, max share 39%); 3DPW no | **no** (3DPW −0.09; 3DHP −1.31, CI includes 0, top share 58%) |
| no distal regression on either population | yes | yes |
| middle joint not worse | yes | yes |
| p95 distal tail ≤ G0 + 2 %L | yes | yes |
| temporal: 3DHP not worse; 3DPW ≤ 4.22° (left-reference-calibrated) | yes (+2.52) | yes (+0.92) |
| validity | yes | yes |
| **transfers** | **yes** | **no** |
| left-arm reference under the same test (calibration) | yes (3DHP strong; 3DPW −2.01 not significant) | |

Every success, including the reference, rests on the cross-domain 3DHP population (6 clusters). The 3DPW distal gain is not significant for any chain under the 2/3-sequence rule; it is provisional evidence.

**Fallback decision (§16, user-authorized).**
- **Whole-body expansion stops at this boundary.** No other limb, torso or pelvis work, no multi-limb network, no FK/IK integration.
- **The successful right-arm chain is preserved** as evidence that the J1X abstraction is bilaterally reusable.
- **The leg's failure is attributed to learned reconstruction accuracy:** near-plane false depth, which already exists in J1X. A secondary contributor is thinner rare-pose supervision. It is not a representation mismatch or a detector/readout fault.
- **Recommended next research milestone (separately scoped; not started):** a bounded single-frame accuracy study of near-plane false depth in the RGB-conditioned connected limb. Why RGB-conditioned models add depth where the limb lies in the image plane, measured on the existing arm and leg candidates. That accuracy, not chain coverage, currently limits reuse.

## 10. Focused tests

`tests/test_framepose_connected_limb_transfer.py` (6 tests) covers:
- chain identities and left/right correspondence;
- left-arm exact equivalence with the historical targets, pair geometry and readout;
- connected unit-chain targets for every limb, with missing-joint validity;
- frozen-readout equivalence with the Worklog 68 function on identical tokens;
- no target or oracle dependency in the readout, and no `src/` coupling;
- decision helpers and the runner, training and audit contracts: gate order, STOPs, the leg rule declared before `main`, and no 3DHP or test use in training.

Results:
- `pytest tests/test_framepose_*.py`: **204 passed, 14 skipped** locally (the skips need torch);
- LabServer63 `animcv-framepose:cuda118`: **218 passed**.

## 11. Artifact identities (LabServer63 `~/animcv-output/framepose_connected_limbs/`)

| artifact | sha256 |
|---|---|
| `contract_audit.json` | `d4223f7a…` |
| `readouts/readout_report.json` (5 banks × left arm / right arm / right leg) | `cf942b22…` |
| `training/training_report.json` | `e4da198b…` |
| `predictions/prediction_report.json` | `93d4d458…` |
| `predictions/predictions_3dpw.npz` / `predictions_3dhp.npz` | `a13f06bc…` / `ef8ca102…` |
| `eval/report.json` | `5113d3cd…` |

Runtimes: `animcv-framepose:cuda118` for audit, readouts, training, prediction and evaluation; `animcv-external:cu124` for review. GPU: RTX 3080 Ti.

## Limitations

- **Developmental populations.** 3DPW and 3DHP test have informed earlier decisions.
- **Weak clustering.** 3DHP has only 6 clusters, and the outdoor 0.16 s analysis 2.
- **One leg, one configuration per chain and conditioning.** The left leg was not tested.
- **Arm-selected supervision.** The BEDLAM expansion was selected on left-arm geometry, so leg rare-pose support is thinner. This batch cannot separate that from the near-plane mechanism.
- **Canonical chain geometry only.** No knee/ankle rig animation, foot orientation, contact or root translation is evaluated.
