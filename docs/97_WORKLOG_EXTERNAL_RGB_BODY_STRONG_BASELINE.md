# Worklog — External RGB Full-Body Strong Baseline (2026-10-25)

## Starting state and environment

- **Branch and HEAD:** `arch/single_frame_first`, starting HEAD `98811d9` (Worklog 96). Local, origin and LabServer63
  were in sync and the worktree was clean. I read PRODUCT_VISION.md and Worklogs 93–96.
- **Prior decision:** W96 stays CASE C. Candidate T was neither used nor tuned.
- **Machines:**
  - Local: Apple M1 Pro, 10 CPU cores, 32 GB, no CUDA.
  - LabServer63: one NVIDIA RTX 3080 Ti (12 GB).

**Nothing was trained, downloaded (beyond the public upstream source code), installed into the AnimCV environment or
run through the external model.**

## 0. Answer

**CASE D: the baseline comparison is blocked at the access gate.**

The official HMR 2.0b baseline cannot be run legitimately here for two reasons.

**1. The official SMPL v1 neutral body model is not available.**
- HMR 2.0 requires `basicModel_neutral_lbs_10_207_0_v1.0.0.pkl` (24 joints, 10 shape and 207 pose blend shapes). It is
  distributed only after registration at smplify.is.tue.mpg.de.
- The only files named "SMPL" in this environment are `~/animcv-data/body_models/smpl/SMPL_{NEUTRAL,MALE,FEMALE}.{pkl,npz}`.
- Inspected structurally, they are **SMPL+H** models (52 joints, 16 betas, 459 pose blend shapes). The `.npz` files are
  symlinks to `../smplh/SMPLH_*.npz`.
- They were staged for the AMASS/SMPL+H pipeline (`docs/06_SERVER_AI_AGENT_TRAINING_RUNBOOK.md`).
- Using them would be exactly the silent asset substitution the directive forbids. Converting SMPL+H to SMPL would be
  as well.

**2. The only CUDA GPU is in use by another project.**
- Two successive processes of `/home/nd/NeuralShaderDemo/...dyn_stages.py` held 7.4–7.9 GB with 82–88% utilization
  during this batch.
- That is the directive's resource-conflict stop condition.

**Consequences.**
- The upstream gate makes the first point hard: `hmr2.models.load_hmr2()` calls `check_smpl_exists()` and raises
  without the SMPL file.
- No external predictions exist, so §§4–9 of the directive (representation alignment, reconstruction metrics, rig
  replay, temporal accounting, qualitative review) were **not executed**. Nothing about HMR 2.0's relative accuracy is
  claimed.

## 1. Official model identity and access

| Item | Finding |
|---|---|
| Repository | `https://github.com/shubham-goel/4D-Humans`, shallow clone at commit `efe18deff163b29dff87ddbd575fa29b716a356c` (2026-02-07, "Update Links"); inspected only, nothing executed |
| Code licence | MIT (`LICENSE.md`) |
| Released checkpoint | HMR2.0b (README: "By default, our code uses the released checkpoint (mentioned as HMR2.0b in the paper)") |
| Checkpoint path | `DEFAULT_CHECKPOINT = ~/.cache/4DHumans/logs/train/multiruns/hmr2/0/checkpoints/epoch=35-step=1000000.ckpt` |
| Checkpoint source | `https://www.cs.utexas.edu/~pavlakos/4dhumans/hmr2_data.tar.gz`, auto-downloaded by `download_models()`. **Not downloaded**: without the body model it cannot produce joints, and no checkpoint hash was recorded |
| Body model | SMPL v1 neutral, registration-gated (smplify.is.tue.mpg.de), non-commercial research licence. **Not available** |
| Upstream SMPL check | `check_smpl_exists()` tests the filename (`~/.cache/4DHumans/data/smpl/SMPL_NEUTRAL.pkl` or `data/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl`) and not the contents |

**Hazard.** Placing the server's SMPL+H file `smpl/SMPL_NEUTRAL.pkl` at that path would pass the upstream check by name
while loading a different body model.

**Entry points.** `demo.py` (ViTDet detector from detectron2, then HMR 2.0) and `eval.py` (dataset boxes). The model
predicts SMPL global orient, 23 body rotations, 10 betas and a weak-perspective camera from a 256×256 person crop; joints
come from the SMPL layer.

**Environment.** Upstream `environment.yml` pins PyTorch with `pytorch-cuda=11.8`, PyTorch-Lightning, detectron2 and
chumpy.

**Training data** (`hmr2/configs/datasets_tar.yaml`):
- MPI-INF-3DHP train (pruned), H36M, MPII, COCO 2014 (+ ViTPose replicate), AVA, AI Challenger, InstaVariety, CMU-MoCap
  (pose prior).
- **3DPW is not in the training list.**
- **MPI-INF-3DHP train S1–S8 is.** It shares the studio and cameras, but not the subjects, with test TS1–4. A 3DHP test
  comparison would therefore be in-domain for HMR 2.0.

**Compute.**
- Not measured, because nothing was run.
- No inference time or memory figure is reported.
- The local M1 has no CUDA. MPS was not evaluated, and the upstream environment targets CUDA 11.8.

## 2. Evaluation population (feasibility only)

**RGB pairing is not the blocker.** LabServer63 holds:
- 3DPW `imageFiles`: 60 sequence folders, 53,796 JPEG frames.
- MPI-INF-3DHP test: TS1 6151, TS2 6080, TS3 5839, TS4 6009, TS5 320, TS6 492 frames.

The W95/W96 3DPW rows and windows identify exact sequence/actor/frame indices.

No evaluation sample or challenge set was drawn. Predeclaring and then not running it would add nothing.

**Input regime.**
- S0's frozen predictions come from the 3DPW-provided 2D detections through the historical observation path.
- A fixed per-person crop derived from those same provided detections is compatible with upstream `eval.py` semantics
  (dataset centre/scale rather than the demo detector). It would be the documented control.
- Not exercised.

## 3. Reproducible gate

**Script.** `scripts/check_external_hmr2_access.py` (new; read-only, no network, no inference):
- inspects candidate body-model files structurally;
- reports the upstream clone's commit and its SMPL check;
- parses GPU occupancy.

Chumpy objects in legacy pickles are stubbed, so no chumpy install is needed.

**Server run** (docker `animcv-external:cu124`, read-only mount of `~/animcv-data/body_models`):

| File | Resolves to | Family | Joints | Betas | Pose blend | Verdict |
|---|---|---|---|---|---|---|
| `smpl/SMPL_NEUTRAL.pkl` | — | SMPL+H | 52 | 16 | 459 | rejected |
| `smpl/SMPL_NEUTRAL.npz` | `smplh/SMPLH_NEUTRAL.npz` | SMPL+H | 52 | 16 | 459 | rejected |
| `smplh/SMPLH_NEUTRAL.pkl` | — | SMPL+H | 52 | 16 | 459 | rejected |

Overall: `smpl_v1_available: false`, `blocked: true`.

**GPU at the time:** 7450 / 12288 MiB used, 82% utilization, marked busy.

**Filesystem search.** No `basicModel*`, `SMPL_python*`, `smpl_mean_params*` or other SMPL v1 file exists under
`/home`, `/data`, `/mnt`, `/opt`, `/srv`, `/media`, `/usr/local` or `/root` on LabServer63, nor in the local home or
volumes.

**Reproduce:**
```
python3 scripts/check_external_hmr2_access.py --body-model <file> [--body-model <file> ...] \
    --upstream-clone <4D-Humans clone> [--nvidia-smi-csv "<used,total,util>"]
```

## 4. Not executed

These directive steps were not executed: §4 representation alignment, §5 reconstruction comparison, §6 pelvis/knee
hierarchy, §7 common-rig replay, §8 temporal accounting and §9 review. There are no HMR 2.0 predictions, metrics, clips
or timing figures. All W91–W96 inputs, outputs and verdicts are untouched.

## 5. Tests and files

**New files:**
- `scripts/check_external_hmr2_access.py`.
- `tests/test_external_hmr2_access_gate.py`.
- This worklog.

The upstream clone was a scratch copy only and was not added to the repository. No other file changed.

**Tests:** `tests/test_external_hmr2_access_gate.py` (4 tests):
- SMPL v1 structure accepted; SMPL+H under the `SMPL_NEUTRAL.pkl` name rejected; missing file reported;
- a symlinked `.npz` resolved and classified;
- a legacy chumpy-object pickle read structurally;
- GPU occupancy parsing;
- the gate contains no download or model-loading call and is not imported by production code.

**Test runs:**
- Locally: 4 passed. On LabServer63 (docker `animcv-external:cu124`): 4 passed.
- Not run: checkpoint identity (no checkpoint), evaluation sample identity, canonical mapping, joint correspondence,
  scale, deterministic inference, IK and Blender tests. There was nothing to test without predictions.

## 6. CASE and evidence limits

**CASE D (baseline comparison blocked).**

**The blockers are about access and resources, not results:**
1. The legitimately licensed SMPL v1 neutral body model is absent.
2. The only renamed "SMPL" files are SMPL+H and would be an asset substitution.
3. The single CUDA GPU was occupied by another project for the whole batch.

**What the batch cannot say.** Whether HMR 2.0b improves on S0's pose, depth, knee-plane or pelvis errors, and whether
any gain would survive the W95 zero-twist rig path, is unknown. Nothing in this report supports either outcome.
