# Cumulative Architecture and Codebase Audit (2026-10-28)

> Independent repository audit of AnimCV at the Worklog 98 snapshot. Audit only: no production, experimental, test,
> worklog, checkpoint, manifest, rig or branch content was changed, and nothing was trained, re-inferred or retuned.
> Worklog text, commit messages, test names and earlier audits were treated as claims to check, not as proof.
> This document records evidence. It contains no next-work, next-architecture or next-batch recommendation.

Evidence tags used throughout:

| Tag | Meaning |
|---|---|
| **CODE** | Directly code-verified by reading source or by a static AST/grep scan |
| **TEST** | Reproduced by executing a test or an isolated synthetic check written for this audit (outside the repo) |
| **DATA** | Replayed or recomputed from persisted dataset-derived artifacts on LabServer63 (read-only) |
| **BLENDER** | Verified by running Blender 5.1.2 headless on audit-scratch outputs |
| **WL** | Worklog-only: stated in a worklog, not independently re-derived here |
| **UNVERIFIABLE** | Could not be checked (missing artifact, environment or metadata) |

---

## 1. Audit snapshot and environment

| Item | Value |
|---|---|
| Repository | `https://github.com/kn99e39/AnimCV` (GitHub reports **public**, default branch `main`) |
| Branch audited | `arch/single_frame_first` |
| Stated snapshot | `5cbf77decbb97a4f6be2720aa845b91eb5309c06` (Worklog 98) |
| Remote HEAD after `git fetch` (2026-10-28) | `5cbf77decbb97a4f6be2720aa845b91eb5309c06`, **identical to the snapshot**. All conclusions apply to both. |
| Other remote heads | `main` a875244, `On_Work` 4b676fc (ancestor of HEAD), `docs/pose-completion-product-plan` 5bc4811 |
| Local worktree | Clean except two **untracked** files: `scripts/run_learned_rotation_vs_geometry_ik.py`, `tests/test_learned_rotation_vs_geometry_ik.py` (W99 work in progress, mtime 2026-10-28). They are not part of the snapshot; they were left untouched. The broad test run collected them, and their counts are separated below. |
| LabServer63 checkout | `arch/single_frame_first` at 5cbf77d; untracked `.DS_Store`, `.animcv_sync_stage/`, `docker/`, the W99 script; 2 stashes. Not modified. |
| Local environment | macOS (Darwin 25.6.0); project `.venv`: Python 3.11.15, numpy 1.26.4, torch 2.13.0, scipy 1.17.1, mmpose 1.3.2, mmdet 3.3.0, mmengine 0.10.7, mmcv 2.1.0, opencv-python 4.9.0.80, pytest 9.1.1; timm, bpy, hmr2 not installed; Blender 5.1.2 at `/Applications/Blender.app` |
| Server environment | LabServer63, one RTX 3080 Ti 12 GB (idle during the audit: 1 MiB used, 0 % util); no GPU job was run |
| Audit scratch | Outside the repo: `/private/tmp/claude-501/.../scratchpad/` (`pytest/`, `checks/`, `history/`, `leak/`, `product/`, `tests/`). No historical output directory was written. |

## 2. Scope and coverage matrix

| Workstream | Coverage | Primary methods |
|---|---|---|
| 1 History integrity | Full branch graph; all pre-existing-file modifications since the `main` merge-base and since `On_Work`; the temporal_lifter ownership move; worklog edit history for all of `docs/` | git object inspection; independent re-execution of pre-refactor code (CODE, TEST) |
| 2 Product alignment | Every public entry point (CLI 35 subcommands, GUI, launchers, Dockerfiles); static import closure; one legacy run from video to .blend | AST closure (CODE); CLI execution (TEST); Blender inspection (BLENDER) |
| 3 Security | All `torch.load` / `pickle.load` sites in src/ and scripts/; downloads; subprocess; credential script; archive extraction | CODE; thread-interleaving simulation without loading any checkpoint (TEST) |
| 4 Causal validity | Decision population for W23–W98; checkpoint selection; bootstrap implementation; probe-vs-use claims; oracle-controlled headlines | CODE + WL |
| 5 GT-free contract | Every GT read in src/; W91–W99 script paths; the W98 gate | AST scan (CODE); behavioural GT perturbation of the W95/W98 leg path (TEST); coverage recomputed from the archived W98 report (DATA) |
| 6 Rig mathematics | Two-bone leg IK on the real BaseRig rest snapshot; quaternion conventions; sign continuity of archived rig keys | Synthetic checks (TEST); archived case JSON (DATA) |
| 7 Tests | All 162 test files (AST classification); one broad run | TEST |
| 8 Complexity | Import graph of 516 files; duplicate math; private-name imports | CODE |
| 9 External provenance | HMR 2.0b checkpoint, predictions, population, SMPL v1 hashes; Docker lineage | `sha256sum` on the server (DATA) |
| 10 Animation evidence | Local review renders W90–W98; archived W97/W98 .blend presence; one legacy Action | BLENDER (legacy); image inspection (W98) |
| 11 Operations | Disk, Docker usage, CI, branch protection, dependency pins, licensing | `df`, `docker system df`, GitHub API (read-only) |

Not covered: no FramePose model was re-run on data; no historical metric was re-derived from raw data; the archived
W97/W98 `.blend` files were not opened; review videos were not watched end-to-end (one W98 contact sheet was inspected).

## 3. Git-history preservation results

**Ancestry (CODE).**
- `On_Work` (4b676fc) is an ancestor of HEAD; HEAD carries 246 commits beyond it.
- `main` is **not** an ancestor. It has three commits that HEAD lacks: 1820f80 removes `tests/` from main by design; 4279dab changes `.gitignore`; a875244 renames mac scripts, which HEAD re-did separately in a847825.
- There are two July "Merge branch 'On_Work'" merges (7a234a7 on the first-parent line, d2fb6e4 on its second parent).
- `docs/pose-completion-product-plan` has one unique commit; its only blob is identical to HEAD's `docs/POSE_COMPLETION_PRODUCT_PLAN.md`.

**Pre-existing files changed.**
- Since the main merge-base (a8761ea): 18 modified, 4 deleted, 1 renamed (CODE).
  - Largest change: `src/app/cli.py` (+862 −19).
  - `src/pose/mmpose_adapter.py`: +108 −6.
  - `src/blender/keyframe_writer.py`: +86 −1.
  - Deletions: mac/windows build scripts and the `third_party/mmpose` gitlink.
- One test was replaced: `test_solve_applies_rest_pose_correction_via_rig_profile` became `test_solve_keeps_direction_rotation_in_world_space_for_blender_export`, in c8cc7dd "Fix Blender retarget transforms" (CODE).
- No test file was deleted on HEAD's line. `Architecture_v2.md` is unchanged since f999e13 (CODE).
- Since `On_Work`: the only modified src file is `src/training/temporal_lifter.py` (+20 −164). Every other src change since then is an addition (CODE).

**temporal_lifter.py ownership refactor (CODE + TEST).**
- A single commit, 3f589f9 (parent 68a5897), moved the canonical pose mathematics into `src/common/canonical_pose.py`. Neither file has changed since.
- Every moved expression is textually identical. The only edits are index lookups through named constants with identical values; epsilons and clamps are unchanged.
- **Independent re-derivation.** The audit extracted `src/` at 68a5897 with `git archive` and executed the *old* functions on the fixture inputs of `tests/fixtures/canonical_pose_reference_v1.json`. All 13 recorded quantities matched bitwise, including `_supervision_loss` (bits `4005016760000000`).
  - The fixture was committed in the same commit as the refactor, so its `source_commit` claim was self-asserted until this check.
  - It is now confirmed as a genuine pre-move capture.
- **Enforced:** formula-level bit parity of the moved functions on one synthetic 6×17×3 float32 CPU input. SVD and hinge quantities are checked to tolerance (1e-12, 1e-9°), not bitwise.
- **Not enforced:**
  - Any A9–A16 or F0–F2 metric value on real data.
  - The non-moved losses (`_yaw_axis_loss`, `_end_effector_loss`, forward-depth losses).
  - Any checkpoint.
  - The file uses `pytest.importorskip("torch")` at module level, so without torch the numpy-only parity checks are skipped silently.
- Preserved metrics are **not** byte-identical source history. The legacy module re-exports the moved functions by identity.

**Prior audit boundary (CODE).**
- docs/24 covered 4b676fc..7f1c235. docs/26 (commit 12118b5) started at ed72402.
- Since 12118b5 there are 229 commits. In src/ they add 62 files and modify 6, all in `src/framepose/` (d35de55..b2afe12).
- No repository-wide audit covered W27–W98 before this one.

**Worklog–commit consistency (CODE).**
- For W91–W98, each worklog's cited scripts and tests exist in its commit or earlier.
- Exceptions: W91's `run_runtime_heading_3dhp.py` arrived with a later W91 edit (855758b). The W97 scripts arrived with the rewrite commit b669173.
- Worklog edits after the append-only rule was added (3344219, 2026-09-22): see F-17.

**Provenance and size (CODE).**
- .git is a 2.1 GB pack. 94 % of blob bytes are `datasets/mpi_inf_3dhp/**/annot.mat.part-*` (54 files, 2,046 MB, added 8167c46).
- `runs/mpi_s1s6_cam0/checkpoint.pth` (1 MB) is tracked.
- 10 `.blend` and 11 `.fbx` files exist in history.
- No `.pt/.ckpt/.npz/.npy/.pkl/.safetensors` file, no SMPL/SMPL-H model, and no secret pattern (hf_, ghp_, sk-, AKIA, private-key headers, HF_TOKEN, api_key, password literals) appears anywhere in history.
- The Depth-Anything-V2 submodule is pinned (a561b849) but not initialised locally.

## 4. Code architecture map

| Layer | Modules | Reachable from public entry points? |
|---|---|---|
| Video intake / 2D observation | `mediaio/*`, `pose/mmpose_adapter.py`, `pose/default_{model,detector}.py`, `pose/subject_tracker.py` | Yes (CLI `extract-frames`, `estimate-pose`; GUI) |
| Legacy 3D lifting | `pose/pose_lifter.py` (VideoPose3D), `training/temporal_lifter.py` | Yes (CLI `lift-pose3d`, `lift-supervised-3d`, training subcommands); the GUI has no 3D step |
| Legacy root / constraints | `pose/root_motion.py`, `pose/constraint_target*.py`, `pose/uncertainty.py` | Yes (CLI) |
| Legacy MotionGraph / retarget / collapse | `motion/*graph*`, `retarget/solver.py`, `fk_solver.py`, `ik_solver.py`, `optimize/collapse.py` | Yes (CLI, GUI) |
| Blender output | `scripts/apply_motion.py` → `blender/executor.py` → `blender/keyframe_writer.py` (subprocess) | Yes (CLI `export-blender`, GUI) |
| FramePose v3 (normative perception) | `framepose/*` (model, train, signs, branch_constraints, pose_reconciliation, context_refiner, visual/depth modules) | **No** (scripts only) |
| Animation semantics, root/pelvis/thorax | `motion/animation_semantics*.py`, `motion/runtime_heading*.py`, `motion/torso_frame.py`, `pose/root_orientation*.py`, `pose/framepose_bridge.py` | **No** (scripts only) |
| FramePose FK/IK | `retarget/framepose_{fk,two_bone_ik,object_owner,target_rest,visual_*}.py` | **No** (scripts only) |
| Pelvis/thorax handoff, leg IK, HMR2 | `scripts/run_{torso_frame_rig_replay,pelvis_torso_ownership,pelvis_frame_leg_articulation,leg_knee_hinge_twist,external_hmr2_baseline,external_orientation_handoff}.py` | **No** (scripts only; not even in src/) |

Static closure (AST, CODE): `app.cli`, `app.gui`, `ui.gui_app` and `blender/*` reach 71 of 139 src modules.
- The closure includes audit and GT importer modules, all behind lazy imports in audit/import/training subcommands.
- No oracle or attribution module is reachable.
- No `framepose.*`, `motion.animation_semantics*`, `motion.runtime_heading*`, `motion.torso_frame`, `pose.root_orientation*`, `pose.framepose_bridge` or `retarget.framepose_*` module is reachable.
- No src module imports from scripts/. The only sys.path manipulation in src/ is `framepose/openpose_coco18.py:60`, for a third-party OpenPose checkout.

## 5. Product Vision alignment

`PRODUCT_VISION.md` §2 itself declares style transfer and authoring future capabilities, and says the first two
capabilities have "partial working infrastructure". The audit measured how partial that infrastructure is.

| Capability | Status | Evidence |
|---|---|---|
| Motion Reconstruction | **Implemented, public, legacy only.** The legacy RTMDet+RTMPose → VideoPose3D / temporal-lifter path runs. The normative FramePose v3 reconstruction is experimental and script-only. | CODE, TEST |
| Animation Reconstruction | **Public part partial.** A hand-written bone mapping keys 2–3 bones; the computed root yaw is lost at export (F-01). Torso/pelvis/thorax, legs, contact and FramePose FK/IK are script-only. | TEST, BLENDER, CODE |
| Style Reconstruction / Transfer | **Absent.** No src module implements style, retiming or exaggeration. | CODE |
| Authoring / Synthesis | **Absent.** | CODE |
| User control | **Minimal, public:** mapping UI, frame range, collapse presets and thresholds, quality-gate thresholds, and a BEZIER Action editable in Blender. No in-tool editing, retiming or locking. | CODE, TEST |

Executable pathway, as it actually exists:

| Stage | Public runtime | Script-only research path | Requires |
|---|---|---|---|
| Video → frames | `extract-frames` (executed) | — | — |
| Person detection | RTMDet in `estimate-pose` (executed) | 3DPW shipped `poses2d`; 3DHP OpenPose + GT identity association | cached checkpoints (public); GT association (research) |
| 2D observation | RTMPose (executed) | FramePose banks | bank with `target_3d`/`target_valid` columns |
| 3D reconstruction | legacy lifter (executed with tracked `runs/mpi_s1s6_cam0/checkpoint.pth`) | S0 / J1X | SHA-pinned frozen `.npy` predictions on LabServer63 |
| Animation semantics | none | `animation_semantics*` | banks + frozen predictions |
| Root / pelvis / thorax | legacy yaw, **dropped at export** | W91 runtime heading, W93–W94 torso frame, W98 HMR pelvis | frozen S0; HMR2 container + SMPL v1 (licensed) |
| FK/IK | legacy direction FK + 2-bone `ik_solver` on mapped bones | `framepose_fk`, `framepose_two_bone_ik`, W95 legs | rest snapshot, calibration, chain JSON (tracked) |
| Blender | Action, BEZIER keys, .blend + FBX (executed) | W90/W91-style replay Actions (CONSTANT), bone proxies or W90A mesh | per-case JSON exported from evaluation runs |
| Editable output | sparse keys after `optimize` (executed) | none | — |

No result in the repository was produced on the `real_animcv_observation` regime (AnimCV's own MMPose+RTMDet feeding
FramePose). The token appears only in definitions and disclaimers (CODE).

## 6. Security findings

**F-06 (MEDIUM, CODE) – checkpoint deserialization trust boundary.** Reviewer concern R1 is confirmed, with the conditions stated below.

**Where unsafe deserialization happens.**
- `src/pose/mmpose_adapter.py:26-47` replaces the module attribute `torch.load` with a wrapper defaulting `weights_only=False`.
- That wrapper is active around `init_model` (`:114`) and `init_detector` (`:131`).
- mmengine 0.10.7 calls `torch.load` by attribute at call time, so the patch is effective with torch 2.13. Pickle code execution is therefore possible inside that window for **any** checkpoint path.
- This includes user-supplied `--pose-checkpoint` / `--detector-checkpoint` paths and whatever sits at the default cache path.

**The default download path.**
- `src/pose/default_model.py:52-77` and `src/pose/default_detector.py:30-44` reuse any file at `~/.cache/animcv/models/<name>.pth` without verification, or download over HTTPS from `download.openmmlab.com`.
- No SHA-256 is checked. The 8-hex suffix in the filename is the upstream naming convention, not a verified digest.
- The temporary `.part` name is fixed, so two concurrent first-time downloads would share it.

**Further unsafe loaders (new).**
- `src/framepose/train.py:323` and `src/framepose/context_refiner.py:525` load FramePose and context-refiner checkpoints with explicit `weights_only=False`. The schema check runs after unpickling, so it provides no safety.
- About 25 experiment scripts do the same.
- The legacy `training/temporal_lifter.py:377,520,938` uses `weights_only=True`.

**Threat conditions.** Exploitation requires an attacker to control one of these:
- the checkpoint file or cache directory;
- the OpenMMLab CDN or the TLS path;
- a checkpoint shared by a third party and loaded by FramePose scripts.

**Mitigating context.**
- An mmengine config file is itself executable Python. A user who passes an untrusted `--pose-config` already executes arbitrary code, independent of the patch.

**What this does and does not prove.**
- No exploit was attempted or observed, and no untrusted checkpoint was loaded.
- This is a missing integrity control on a pickle path. It is not a demonstrated compromise.

**F-07 (MEDIUM, TEST) – the patch is not thread-safe and can leak process-wide (new).**
- A two-thread interleaving simulation (`checks/patch_race.py`) ran without loading any file:
  - thread A enters the context;
  - thread B enters, saving A's wrapper as its "original";
  - A exits and restores the real `torch.load`;
  - B exits and restores A's wrapper.
- Observed result: after both exit, `torch.load` remains `_mmengine_checkpoint_compat.<locals>.patched_load`, so the `weights_only=False` default persists for the rest of the process. During B's remaining window `torch.load` was already unpatched, so B's model init would fail under the torch ≥2.6 default.
- The GUI runs pose estimation in a daemon worker thread (`src/ui/gui_app.py:251-272`, `:761-837`). No busy guard on the run button was found.
- **Not proven:** that concurrent estimation happens in practice. Other in-process `torch.load` callers are mmengine loads that already need `weights_only=False`, so practical impact is bounded.

**Other surfaces (CODE).**
- `scripts/fetch_smpl_v1.sh` reads the SMPL-site credentials with a hidden prompt and passes them to curl through mode-600 temp files (umask 077, trap cleanup), never on the command line.
  - It extracts exactly one named member, so there is no path traversal.
  - It prints, but does not compare, the SHA-256. The pin exists only in the W97 `PREDECLARED` text.
- 3DPW `pickle.load(..., encoding="latin1")` reads dataset files; this is standard and trusted-input.
- The Blender export uses a list-form `subprocess.run` (no `shell=True` found in src/).
- No credentials are stored in the repository (§3).

## 7. Numerical and coordinate-contract findings

**Leg IK on the real BaseRig (TEST, `checks/leg_ik_contract.py`).**
- Setup: 6,000 synthetic legs through the W95 path (`solve_leg` → two-bone IK → `chain_forward_kinematics`) on `examples/e2e_demo/baserig_blender_rest.json`.
- Bone-length residual is at most 9e-14 and the deviation of quaternion norm from 1 is at most 2.2e-16. Endpoint parity holds.
- Thigh/shin direction errors reach 2.6°, because the IK preserves endpoint direction and reach fraction, not source bone proportions. This is documented W61 behaviour.

**F-11 (MEDIUM, TEST; limitation acknowledged in W95/W96) – limit of the zero-twist contract.**
- With knee flexion between 20° and 130°, the rig knee-hinge deviation from the source knee plane grows with thigh axial rotation:

| Thigh axial rotation | Mean deviation | Share above W95's 15° bound |
|---|---|---|
| 0–15° | 14.6° | 40 % |
| 15–30° | 24.9° | 69 % |
| 30–45° | 39.0° | 92 % |

- Shortest-arc swing from the rest thigh carries a hip-dependent twist, so the knee bends about a non-anatomical axis even when world limb directions are exact.
- This independently confirms that W95/W96's knee-plane errors are a representation limit, not a source error alone.
- The rejected W96 thigh-roll candidate was not evaluated or promoted.

**F-10 (MEDIUM, DATA + CODE) – quaternion sign continuity (new).**
- `motion/torso_frame.py:85-102` (Shepperd) and `retarget/axis_utils.py` choose the quaternion sign by branch. Neither enforces hemisphere continuity across frames.
- Archived W98 cases (`replay/cases_3dpw.json`, `cases_3dhp.json` on LabServer63): 37 of 4,803 adjacent key pairs have dot < 0. Of these, 31 are on the Object and 6 on `spine_01`.
- In 20 of the 37, the true rotation between frames is below 30°, so the sign flip is pure representation.
- The review renderer sets CONSTANT interpolation (`scripts/render_runtime_heading_rig_blender.py:183-185`), which hides the flips. Under Blender's default component-wise quaternion interpolation these keys would spin the long way.
- Not observed in the three legacy example clips: 0 flips in 166 key pairs, a small sample.

**Coordinate conversions (CODE + TEST).**
- W98 `handoff_frames` (`scripts/run_external_orientation_handoff.py:74-79`) uses `CV_TO_CANON · G · SMPL_BODY`. Both matrices are proper rotations. Fixtures for upright, lying, yawed and identity orientation pass (`tests/test_external_orientation_handoff.py`, executed).
- The OpenCV→AnimCV axis change is implemented five times, and `SMPL_TO_CANONICAL` twice (§13).

**F-20 (LOW, CODE + TEST) – two Procrustes definitions.**
- `common/canonical_pose.similarity_align:130` scales by the norm ratio.
- W97 `eval_external_hmr2_baseline.procrustes:58` uses the optimal Umeyama scale.
- On synthetic poses they differ by −2.1 to +3.2 mm per row. PA-MPJPE is secondary in W97 and both W97 sources use the same function, but W97 PA numbers are not comparable with F0–F2 PA numbers.

**F-21 (LOW, TEST) – quaternion naming collision.**
- `framepose/nonlinear_reconstruction.quaternion_multiply/_conjugate` use **wxyz**; `retarget/axis_utils` functions of the same names use **xyzw**.
- Mixing them gives errors up to 1.37. No current caller mixes them.

## 8. GT leakage and evaluation-isolation findings

**Runtime-value isolation (CODE + TEST).**
- These src modules read no GT (AST scan, confirmed by reading): `framepose_bridge.assemble_h0/build_h0_lifted_sequence`, `animation_semantics*`, `runtime_heading*`, `torso_frame`, `root_motion`, `root_orientation_diagnostic`, `hinge_plane`, `sign_advisor` and `retarget/framepose_{fk,direction_candidate,object_owner,two_bone_ik,target_rest}`.
- FramePose training uses `bank.indices("train")` for gradients and VALIDATION for checkpoint selection. TEST is not touched (`framepose/train.py:150,233-263`).

**Behavioural check (TEST, `checks/gt_invariance.py`).**
- For 200 random poses, `w95.evaluate` was run with `gt_legs = None`, then with two unrelated random GT references.
- The thigh and calf local quaternions were **bitwise identical** across all three. ROM violations were identical. Only the error scores changed.
- So in the W95/W96/W98 leg path, GT reaches the scores only.

**F-08 (MEDIUM, CODE + DATA) – GT-conditioned coverage and population.** Reviewer concern R3 is resolved as **(2) a replay-harness limitation, with (4) a partial documentation mismatch**. It is **not (3)**: the runtime does not depend on GT values.
- `handoff_frames` reads one HMR prediction only. `cond_eval` passes `gt_legs` to `w95.evaluate`, which uses it only to compute `thigh_err`/`shin_err`/`ankle_dir_err` (`scripts/run_pelvis_frame_leg_articulation.py:149-152`).
- Candidate *execution* is nevertheless conditional on GT in several places:
  - `row_conditions` returns `{}` when `r["gt_legs"] is None` (`run_external_orientation_handoff.py:109`). `gt_legs` exists only where the SMPL pelvis exists (3DPW, `:348`) or `target_valid` holds (3DHP, `:398`).
  - Upstream, W94 `build_rows` creates a row only where GT yaw and the GT torso frames exist (`run_pelvis_torso_ownership.py:195-200`).
  - Further upstream, 3DPW 2D input is blanked where `campose_valid` is False (`pose/three_dpw_adapter.py:86`), and `temporal_lifter.build_dataset:46-75` drops frames without a GT target or GT pelvis.
  - 3DHP/BEDLAM banks admit a row only when a detection associates with GT 2D (`framepose/bedlam_supervision.associate:227`, labelled "GT used only for identity").
- Measured effect, recomputed from the archived W98 `report.json` (DATA):
  - 3DPW TEST: 7,076/7,076 rows have P/T; VALIDATION: 3,407/3,407.
  - 3DHP: 2,532 of 2,728. The 196 rows that are lost all have valid HMR predictions (13,211/13,211 valid), so their only blocker is the GT leg reference.
- docs/98 §11 reports the 3DHP count and its cause. But `PREDECLARED["invalid"]` names only the degenerate HMR thorax as making T unavailable, and the Blender case text says "condition unavailable for this row (no GT/HMR fallback)" for GT-gated rows.
- Population-level GT conditioning, and the shifted t±2 windows in `context_refiner.sequence_window_positions:61` over retained rows, are not stated as runtime limitations.

**F-03 (HIGH, CODE; acknowledged retroactively in W91) – oracle sign inputs in the historical H0 baseline.**
- `scripts/prepare_context_h0.py:29` defaults to `--sign-source oracle`. Its lines 41-43 compute `oracle_sign_states(target_3d, target_valid)` and feed them as **model inputs at inference** on train, validation and test.
- docs/91 §1 records that the frozen H0 (`context_h0_v2`) was built this way. It reclassifies H0 joints, H0 yaw and H0-based FK/IK, reconciliation, the context refiner and W52 contact as GT-inheriting.
- Worklogs W41–W61 and W74–W79 that used H0 as a baseline or source were written before that disclosure. Example: W52 states that Root Orientation "SURVIVES … production 조건" (52:266-270) on a GT-fed model.
- The defaults remain oracle in two places:
  - `prepare_context_h0.py` (`--sign-source oracle`);
  - `pose_reconciliation.ProjectionContext.placement_mode`, which defaults to `"research_oracle_absolute_root_placement"` (`src/framepose/pose_reconciliation.py:60`).
- S0 and J1X do not take oracle signs. S0 uses `neutral_sign_states` (CODE).

**F-24 (LOW, CODE).** CLI `export-3dpw-ground-truth` (`src/app/cli.py:1185`) writes GT-gated 2D. It is labelled diagnostic, but nothing prevents feeding it to `lift-pose3d`.

**Oracle-controlled headlines.**
- Stated explicitly in the title or verdict: W28, W62, W77, W93, W94, W95, W96, W98-O.
- Not explicit (WL + CODE): the W97 §0 rig claim uses the SMPL pelvis as the Object root, which is disclosed only at 97:199 and in the limits section.

## 9. Test and reproducibility evidence

**Broad regression, executed once (TEST).**
- Command: `.venv/bin/python -m pytest -q -rs -p no:cacheprovider --junitxml=<scratch>/junit.xml` (repo root, pyproject `pythonpath=src`).
- Environment: as in §1. Wall time 39.4 s.
- Result: **1204 passed, 1 skipped, 0 failed, exit 0.** 1,205 collected.
  - 9 of the passes come from the untracked W99 file. For the committed snapshot: **1,196 collected, 1,195 passed, 1 skipped**.
  - The skip is `tests/test_frame_pose_backbone_weights.py:10`, because `timm` is not installed. That is a dependency gap, not a code defect.
- Warnings: torch.jit deprecation, one requires_grad scalar conversion, and an unknown pytest option introduced by the audit command itself.
- The worktree was unchanged after the run.

**Focused runs (TEST).**
- `test_frame_pose_branch_constraints.py` and `test_external_orientation_handoff.py`: 35 passed.
- Audit synthetic checks: `gt_invariance.py`, `leg_ik_contract.py`, `patch_race.py` (results in §6–8).

**Test-family classification (CODE, AST scan of 162 files / 1,168 test functions).**

| Category | Finding |
|---|---|
| Source-text tests | 79 functions in 47 files read `.py`/Dockerfile text, and 53 of them assert `not in`. 13 functions in 11 files assert that GT tokens are absent from a function body (e.g. `test_external_orientation_handoff.py:103-114`). They catch token-level regressions only, not aliasing through `ctx`/`r[...]`, callee behaviour or runtime values. |
| Behavioural GT perturbation | **3 only:** `test_animation_semantics.py:85`, `test_animation_semantics_v2.py:150-157`, `test_framepose_bridge.py:41`. None covers S0 inference, runtime heading, the torso frame, the W98 adapter or leg IK. |
| Circular or determinism-only | `test_animation_semantics.py:131` and `test_framepose_visual_arm_ownership.py:89` compare identical calls. `test_framepose_leg_knee_hinge_twist.py:111-119` sets φ to the SMPL roll error and then asserts that error ≈ 0. |
| Script-loading tests | 42 files exec-load 54 of 197 scripts. W9x logic is tested where it lives, in scripts/. |
| Real backends | **None:** no real MMPose/RTMDet, FramePose checkpoint, HMR2 or Blender. Blender is a `fake_bpy` stub plus a faked `subprocess.run` (`test_cli.py:488-581`). |
| Pinned SHAs | 10 constants in 8 files compare a script constant against a copy in the test. No test hashes an artifact; the real hash stop runs only inside scripts on the server. |
| Historical metrics | No test asserts any historical MPJPE or A9–A16 value. |
| Untested src modules | Not imported by any test: `app.config`, `app.gui`, `blender.export`, `mediaio.cache`, `pose.default_detector`, `pose.mpi3dhp_adapter`, `pose.schemas`. |

**F-09 (MEDIUM, CODE + TEST).** Reviewer concern R4 is confirmed. For W60–W99 script adapters, the GT-free guarantee rests on source inspection. The audit's behavioural check covers only the leg path (§8).

**Worklog test claims.** Pass statements for the W93–W98 suites on LabServer63 (e.g. "50/50") are **WL**; this audit re-ran those suites locally only, inside the broad run. Worklogs that skipped broad regression ("no shared contract changed": W96, W98) were not contradicted, because the broad run passes at HEAD.

**F-14 (MEDIUM, CODE + TEST) – reproducibility of the historical evidence.**
- Frozen predictions, banks and checkpoints for S0, J1X, W87 and HMR2 exist only on LabServer63, and the scripts pin their SHA-256.
- The audit re-hashed the HMR2 inputs (DATA):
  - `pred_full.npz` `fc5d3305…7003`, matching the pin;
  - `population.json` `533895c9…cdb9`, matching;
  - the checkpoint `c9ea09c8…2fcf`, matching W97;
  - SMPL v1 `98e65c74…bc42`, matching.
- The W98 `w97_oracle_reproduction` block in the archived report shows condition O equal to W97 E to full float precision on 6,339 rows (DATA).
- No FramePose-era metric (F0–F2, W64–W98 tables) was recomputed. Those numbers remain WL + persisted-report evidence.

## 10. W91–W98 focused cumulative review

| W | Baseline | Manipulated variable | GT in candidate? | Verdict wording vs code |
|---|---|---|---|---|
| 91 | H0 (GT signs, labelled) / fixed 180° | heading source S0 | No: 3DPW shipped `poses2d`; 3DHP AnimCV OpenPose | GT-free holds. Self-downgraded A→B (claim edit 855758b, see F-17). CODE |
| 92 | S0 heading | Qwen3-VL-8B BACK/STOMACH advisor (4-bit, cached `responses.jsonl`) | No | "Limited" is accurate. Designed after inspecting 3DHP TS2 failures and scored on those same rows. VAL answers were all UNCLEAR, so validation tested nothing. CODE + WL |
| 93 | yaw-only | full torso frame | Gate run on the GT frame (stated) | CASE B qualified. CODE |
| 94 | W93 | pelvis on Object | Success under SMPL pelvis (stated) | OK. CODE |
| 95 | B = SMPL pelvis | root policy | Phase G uses GT legs/thorax; B_S0 labelled EVALUATION ONLY | OK. CASE C. CODE |
| 96 | zero twist | thigh roll T | Positive only under GT geometry | CASE C accurate; T rejected. CODE |
| 97 | frozen S0 | HMR 2.0b RGB source | No (boxes from bank `input_2d`) | The rig-replay Object is the SMPL pelvis, not in the §0 headline. 3DHP boxes are called "dataset-provided" but are AnimCV OpenPose with GT identity association (F-19). 3DHP is in-domain for HMR2 (disclosed). The commit rewrote docs/97 from CASE D to CASE A (F-17). |
| 98 | O = SMPL pelvis | Object pelvis source (P), thorax source (T) | Values: no (behaviourally verified for legs). Coverage: GT-gated (F-08). | CASE B is not overstated in its limits section. Its GT-free assurance is source-text only. |

**W95–W98, by layer.**
- **Geometry correctness:** limb directions are Object-invariant to within small residuals. The archived W98 `report.json` gives the maximum O/P/T world-direction component difference (DATA):
  - thigh: 1.3e-3 on 3DPW VALIDATION, 7.8e-4 on TEST, 0 on 3DHP;
  - upper arm: ≤ 8e-6.

  The non-zero thigh residual was not attributed. Bone lengths are exact (TEST).
- **Rotation representation:** the zero-twist hinge limit (F-11) and sign discontinuities (F-10).
- **Rig transform correctness:** parent/local composition fixtures pass (TEST).
- **Skinned animation quality:** not established; see §12.

**W97/W98 provenance (R9, WS9).**
- Hashes match the pins (DATA).
- The upstream 4D-Humans revision `efe18def…` **cannot be verified**: the server build context `~/animcv-data/external/w97/build/4D-Humans` has no `.git`, the build log does not record it, and the Dockerfile `COPY`s it unverified. For later comparison, its `hmr2/*.py` tree hashes to `9b4ea1b1…22d8e` (UNVERIFIABLE).
- The `animcv-external:cu124` base image used for every W85–W98 server test run has no Dockerfile in the repository or in the server's untracked `docker/` directory. That directory builds a different image.
- `joints24` is recomputed from the predicted parameters with `smplx.SMPLLayer` inside the container. Crop-camera axes are documented, and W98 attributes the pelvis error partly to the crop ray.
- The SMPL v1 model is licensed for non-commercial research (provenance file on the server) and is not in git.

## 11. Broader W01–W90 audit findings

**F-04 (HIGH, CODE + WL) – 3DPW TEST is no longer a held-out estimate.** Reviewer concern R5 is confirmed as a risk; its magnitude is undetermined.
- About 51 worklogs use 3DPW TEST as a decision input: 19+2 of W28–W50, 24 of the 27 verdict worklogs W64–W90, and all of W91–W98.
- From W83 onward, MPI-INF-3DHP TS1–6 is also a decision population.
- docs/80:414 and docs/82:369 acknowledge that TEST "has informed every architecture decision since Worklog 73".
- Checkpoints were selected on VALIDATION (CODE: `framepose/train.py:233-263`, `scripts/train_framepose_joint_arm_geometry.py:102-110`).
- Concrete architecture or candidate selections made after TEST results were seen:
  - O_BILATERAL became the frozen H0 on TEST yaw P95 (W29/W30/W35, WL);
  - the W29→W32 candidate sets came from test tables (`diagnose_sign_influence.py:120` defaults to test, CODE);
  - W48 reused W47's test-quantile cohort (CODE);
  - W75 overturned W74's VALIDATION verdict by running on TEST (WL);
  - W88 redefined its validation population after TEST output existed (commit 6f7a3c5, CODE);
  - W92 was designed on 3DHP TS2 failures and scored on them (WL).
- No threshold was found retuned on TEST metric values.
- This does **not** prove overfitting. It proves that TEST and 3DHP results after W73 cannot serve as unbiased generalisation estimates.

**F-16 (MEDIUM, CODE) – statistical resolution.**
- `_brief` → `_contrast` → `_bootstrap` (`scripts/run_framepose_visual_full_arm_ik.py:422`) is a percentile bootstrap over `sequence_id` clusters: 2,000 replicates, fixed seed 1337, frame-weighted. The moving-block variants are computed and discarded.
- Clusters available:
  - 3DPW TEST: 37 actor-sequences from 24 videos;
  - 3DPW VAL: 16;
  - 3DHP: 6.
- A CI is emitted for any stratum with ≥2 clusters. Several reported strata (lying, waist-bend, n = 3–370 rows) span very few sequences, so their intervals are not reliable.

**Probe-versus-use claims (CODE + WL).**
- No linear probes exist in W29–W40; their claims rest on retraining, toggles and swaps.
- "The model uses the fed input" (28:137, 29:87) is circular: S1 is fed sign(D) and scored on sign(D), as the worklog admits.
- W64–W72 "contains/encodes depth" claims are backed by retrained ablations plus an inference-time readout shuffle (`vlm_depth_advisor.py:174-188`, `continuous_vision_depth.py:94-99`). The shuffle shows image dependence, not depth-specific content, until W88's content/position separation.
- "W67 was a readout problem" (68:265) has no intervention behind it.

**Regime labelling (CODE).** AGENTS.md requires every result to carry an evaluation-regime label. The literal labels appear only in W73, W74, W79 and W90A. W83+ (AnimCV OpenPose on 3DHP) and W97/W98 (RGB end-to-end) fit none of the three defined regimes.

## 12. Real Blender and animation evidence

| Evidence class | Status |
|---|---|
| Bone-proxy demonstrations | Present for W59–W96: render scripts and local mp4/png under `~/animcv-output/90…98_*` (CODE, file listing) |
| Skinned mannequin | `examples/skinned_mannequin/BaseRig_Skinned.{blend,fbx}` with `validation.json` (W90A). W98 renders use it (contact sheet inspected). |
| Numerical FK parity | Verified synthetically (TEST); the Blender-side parity claims of W59–W61 are WL |
| Saved Action, editable keys (public path) | **BLENDER, executed in the audit's scratch directory.** The legacy chain on `examples/e2e_demo/person.mp4` produced Action `Generated_Motion` with 8 fcurves and 24 BEZIER keys, on `upperarm_l`/`lowerarm_l` only. The 3D constraint chain produced `Constraint_Target_Motion` with arm fcurves only (F-01). The factory Cube/Camera/Light remain in the saved scene. |
| Saved Action (research path) | W90/W91-style replay scripts key the Object plus mapped locals with CONSTANT interpolation, one Action per mode (CODE). The archived W98 `.blend` files exist on LabServer63 but were not opened. No `.blend` exists locally for W90–W98. |
| Meaningful full-body motion | **Not present.** W98 animates Object, `spine_01`, both legs (zero-twist) and the **left arm only** (J1X). The right arm, hands, head and neck stay at rest; visible in `98_…/3dpw_walking_upright.png`. |
| Stable animation on unseen real video | **No evidence.** Every research render is on 3DPW or 3DHP benchmark frames sampled at about 6 Hz, not native cadence, with no `real_animcv_observation` input. |
| Root translation / foot contact | Absent in rig output. Root translation is explicitly UNAVAILABLE (W60); contact work (W51–W52) is H0-based (F-03). |
| User controllability / style | Absent (§5) |

**F-01 (HIGH, TEST + BLENDER + CODE) – the public path silently drops root yaw (new).**
- `retarget-constraint-targets` bakes three tracks (`upperarm_l`, `lowerarm_l`, `root`); `optimize` keeps all three.
- `blender/keyframe_writer.write_keyframes` skips any track whose bone is missing (`src/blender/keyframe_writer.py:138-139`).
- BaseRig has no `root` pose bone: its 51 bones per `baserig_blender_rest.json`. The armature **object** is named `root`.
- The saved Action therefore contains only `pose.bones["upperarm_l"/"lowerarm_l"].rotation_quaternion`, verified headless in Blender 5.1.2.
- No warning is emitted, and no document describing this loss was found.
- W60 assigns root orientation to the Armature Object on the research path, but that object-owner adapter is not reachable from the CLI.

**Review-labelling hazard (LOW, observation).** W98 renders reuse the slot names FIXED/RUNTIME/GT for conditions O/P/T. The panel titled "GT side view" shows condition T (GT-free HMR). The per-panel header does name the condition.

**Expectations intentionally outside current scope:** style reconstruction/transfer, authoring, root translation, hands/face (PRODUCT_VISION §2 and §7, W60).

## 13. Operations, maintainability and licensing findings

**F-05 (HIGH, CODE; legal interpretation UNCONFIRMED) – licensed research data in a public repository.**
- The GitHub repository is public (API: `"visibility":"public"`) and has no LICENSE file.
- It tracks 2,046 MB of MPI-INF-3DHP raw annotations (`datasets/mpi_inf_3dhp/**/annot.mat.part-*`, plus `camera.calibration`) since 8167c46. It also tracks `runs/mpi_s1s6_cam0/checkpoint.pth`, trained on them.
- `docs/06_SERVER_AI_AGENT_TRAINING_RUNBOOK.md:17` itself says "Do not commit or publish licensed source data".
- The audit did not obtain the dataset licence text, so whether this breaches its terms is UNCONFIRMED. The redistribution itself is a fact.
- The provenance of `examples/*/source.mp4` and the frame sets is not documented in their READMEs (UNVERIFIABLE).

**R9 – SMPL licence (CODE + DATA).**
- SMPL v1 (and the SMPL-H/AMASS assets on the server) are non-commercial research licences.
- The W97/W98 HMR2 path, the 3DPW GT adapter and the AMASS/BEDLAM supervision all depend on SMPL-family models.
- No SMPL file is in git. This constrains any commercial animation product built on these paths; it is not a code defect.

**R7 / F-12 (MEDIUM, CODE).**
- No `.github/` directory and no workflows (API: `total_count 0`). No branch protection on `main`, `On_Work` or `arch/single_frame_first`; 0 rulesets.
- No lock files. `pyproject.toml` uses ranges, and the `depth`/`training`/`frame-pose` extras leave `torch` unpinned.
- Docker base images `osn-gs:cuda11.8` and `animcv-external:cu124` are host-local and undefined in the repo, and no image is pinned by digest. `Dockerfile.framepose` uses `timm>=1.0,<2` and unpinned `pillow`/`huggingface_hub`. Its `COPY . .` is bounded by `.dockerignore`, which excludes `datasets`, `examples`, `docs` and `.git`.
- The Depth-Anything-V2 submodule is not initialised locally.
- Path portability: experiment scripts document LabServer63-specific absolute paths in places (e.g. `/home/nd/animcv-data` in `download_amass_hf.py:12`).

**R8 / F-15 (MEDIUM, server measurement, 2026-10-28 03:49 UTC).**
- `/var/lib` (Docker storage) is at **84 %: 78 G used, 16 G free.** W98 had recorded 100 % with 145 MB free, so space has since been recovered.
- Docker: 15 images, 81.3 GB, of which 44.5 GB is reclaimable; build cache 30.9 GB.
- Each AnimCV image is 14–26 GB, so a single additional image build can exhaust the remaining space.
- `/home` is at 12 %, and `~/animcv-output` holds 14 GB.
- Load average was 24.7 at measurement (shared host). Nothing was deleted.

**F-13 (MEDIUM, CODE) – script coupling (R6).**
- 92 of 200 scripts import other scripts and 79 manipulate `sys.path`.
- The deepest chain is 14 modules, rooted at the untracked W99 script and passing W98 → W97 → W96 → W95 → W94 → … → `run_framepose_ik_gt_attribution` → `run_root_motion_contact_h0_replay`. The "GT-free" W98/W99 adapters thus transitively load GT-attribution scripts, although the traced values do not use them.
- 431 underscore-private names are imported across module boundaries. The most-imported are `_matched_oracle_sequence` (24 importers), `temporal_lifter._torch` (24) and `_supervision_loss` (17).
- Inside src/, runtime candidate helpers live in an evaluation module: `retarget/framepose_visual_ik_handoff:32`, `framepose_visual_full_arm_ik:22` and `framepose_visual_arm_ownership:26` import private names (`_ik`, `_runtime_side`) from `retarget/framepose_ik_gt_attribution`. Those functions are GT-free, so this is a coupling hazard, not a leak.
- Duplicate mathematics:
  - 2 quaternion multiplies, in different conventions (F-21);
  - 5 rotation geodesics;
  - 6 similarity/Kabsch alignments (two are copy-pastes in `dataset_3d_audit.py:50` / `mpi3dhp_3d_audit.py:53`);
  - 5 OpenCV→AnimCV conversions;
  - 23 normalise helpers.
- Unreferenced src modules: `app/config.py`, `mediaio/cache.py`, `pose/schemas.py`.
- Per the audit brief, research organisation as separate experiments is acceptable; the finding is that W9x animation logic exists **only** in scripts.

**F-17 (LOW, CODE) – worklog append-only violations after the rule.**
- docs/91: 855758b downgraded CASE A to B and removed a "met" table; 90694fb.
- docs/97: b669173 rewrote it (+285 −118) from CASE D to CASE A. It states that it "supersedes", but the interim CASE D text no longer exists in HEAD.
- Factual fixes: docs/87 (a6214b5), docs/91A (d8316a4). Path fixes: docs/61, docs/62.
- **F-18 (LOW):** duplicate number 44 (`44_WORKLOG_HISTORICAL_ARTIFACT_RECOVERY.md`, `44_WORKLOG_POSE_RECONCILIATION_REAL_REPLAY.md`).

**F-19 (LOW, CODE).** The W97 PREDECLARED box source describes 3DHP boxes as "dataset-provided 2D detections". For 3DHP they are AnimCV-run OpenPose detections selected by GT 2D association (`scripts/build_mpi3dhp_test_external_bank.py:162-180`).

**F-22 (LOW, CODE).** Evaluation-regime labels are missing in most W64–W98 worklogs (§11).

## 14. Severity-ranked confirmed issues

No CRITICAL issue was found. No unsafe execution was demonstrated, no data corruption was found, and no defect invalidating a currently claimed central result was found: the H0 GT inheritance (F-03) was already disclosed by W91.

| ID | Sev. | New / Ack. | Summary | Evidence |
|---|---|---|---|---|
| F-01 | HIGH | New | Public CLI root-yaw track silently dropped at Blender export (`keyframe_writer.py:138-139`; BaseRig has no `root` bone) | TEST, BLENDER |
| F-02 | HIGH | Partly ack. (PRODUCT_VISION §2) | Normative FramePose v3, AnimationSemantics, root/pelvis/thorax, FramePose FK/IK and HMR2 unreachable from any public entry point; script-only, on GT-bearing banks and server-only frozen predictions; no `real_animcv_observation` result exists | CODE, TEST |
| F-03 | HIGH | Ack. (W91, retroactive) | Historical H0 receives GT forward-depth signs at inference; it underlies W41–W61 and W74–W79 baselines; script defaults still oracle (`prepare_context_h0.py:29`, `pose_reconciliation.py:60`) | CODE |
| F-04 | HIGH | Partly ack. (docs/80, 82) | 3DPW TEST (and 3DHP from W83) used as a decision population for about 51 worklogs, with concrete selections after TEST results | CODE, WL |
| F-05 | HIGH | New | Public repo, no LICENSE, 2 GB MPI-INF-3DHP raw annotations plus derived checkpoint tracked; licence breach UNCONFIRMED | CODE |
| F-06 | MEDIUM | New for FramePose loaders; R1 | `weights_only=False` on mmengine (global patch) and FramePose/context-refiner loaders; default checkpoints reused or downloaded without hash check | CODE |
| F-07 | MEDIUM | New | Global `torch.load` patch not thread-safe; interleaving leaves it patched process-wide; GUI runs estimation in worker threads | TEST |
| F-08 | MEDIUM | Partly ack. (W98 §11) | GT-conditioned coverage and population across the research path; W98 gate = harness limitation + documentation mismatch, not runtime leakage | CODE, DATA, TEST |
| F-09 | MEDIUM | New | GT-free assurance for W60–W99 adapters is source-text only; 3 behavioural perturbation tests in total | CODE |
| F-10 | MEDIUM | New | No quaternion hemisphere continuity; 37 sign flips (20 pure artifacts) in W98 rig keys, masked by CONSTANT interpolation | DATA, CODE |
| F-11 | MEDIUM | Ack. (W95/W96) | Zero-twist leg IK knee-hinge deviation exceeds 15° for 40–92 % of synthetic legs depending on thigh axial rotation | TEST |
| F-12 | MEDIUM | Partly ack. | No CI, no branch protection, no lock files, host-local undefined base images, unverifiable 4D-Humans revision | CODE, DATA |
| F-13 | MEDIUM | New | Deep script-to-script coupling (14-level chain, 431 private imports); runtime helpers owned by GT-attribution modules | CODE |
| F-14 | MEDIUM | New | Historical parity is formula-level on one fixture; no historical metric is re-checkable without server-only artifacts; parity file skips without torch | CODE, TEST |
| F-15 | MEDIUM | Ack. (W98) | Docker storage at 84 % (16 GB free) against 14–26 GB images | server measurement |
| F-16 | MEDIUM | New | Cluster bootstrap with ≤24 / 16 / 6 clusters; CIs emitted down to 2 clusters | CODE |
| F-17 | LOW | Partly ack. | Post-rule worklog rewrites with verdict changes (docs/91, docs/97) | CODE |
| F-18 | LOW | New | Duplicate worklog number 44 | CODE |
| F-19 | LOW | New | W97 box provenance wording for 3DHP; SMPL-pelvis root absent from W97 §0 | CODE |
| F-20 | LOW | New | Two non-equivalent Procrustes definitions (PA not comparable across eras) | CODE, TEST |
| F-21 | LOW | New | Same-named quaternion functions in wxyz and xyzw conventions | TEST |
| F-22 | LOW | New | Evaluation-regime labels missing from most W64–W98 worklogs | CODE |
| F-23 | LOW | New | W98 review renders label condition T under a "GT" slot name | observation |
| F-24 | LOW | New | `export-3dpw-ground-truth` GT-gated 2D can be fed to `lift-pose3d` | CODE |

What the HIGH findings do **not** prove:
- F-01 does not show that the research-path Object rotation is wrong.
- F-02 does not show FramePose is unfit; only that it is not a runtime.
- F-03 does not invalidate S0, J1X or W91–W98, which do not take oracle signs.
- F-04 does not establish overfitting.
- F-05 does not establish a licence breach.

## 15. Unconfirmed risks and missing evidence

| Item | Why unconfirmed |
|---|---|
| MPI-INF-3DHP licence breach (F-05) | Licence text not obtained |
| HMR2 upstream revision `efe18def…` | No git metadata in the server build context |
| `animcv-external:cu124` image contents | No Dockerfile in the repo or on the server |
| Practical occurrence of concurrent GUI pose runs (F-07) | Only the mechanism was demonstrated |
| Skinned-mesh quality of W90–W98 animations | Only one contact sheet inspected; videos and archived `.blend` not reviewed frame by frame |
| Historical F0–F2, A9–A16 and W64–W98 metric values | Not recomputed; persisted reports and worklogs only |
| W99 (untracked) GT isolation | Outside the snapshot; the coupling chain was observed but not traced for values |
| Label-builder split discipline in individual scripts | `chain_targets`/`pair_labels` callers not audited one by one |
| Provenance of the `examples/*/source.mp4` clips | Not documented in the repository |
| Effect of GT-shifted context windows (`context_refiner`) | Mechanism verified; magnitude not measured |
| Blender-side W59–W61 FK parity values | WL; not re-run |
| The CI/branch-protection policy owner's intent | API shows absence only |

## 16. Reviewer-concern verification matrix (R1–R10)

| ID | Concern | Verdict | Basis |
|---|---|---|---|
| R1 | Unsafe checkpoint-loading trust boundary | **Confirmed (MEDIUM)**, wider than stated: the FramePose loaders also use `weights_only=False`, and the global patch is non-thread-safe (F-06, F-07). No exploit observed. | CODE, TEST |
| R2 | Legacy CLI disconnected from FramePose animation | **Confirmed (HIGH).** The CLI/GUI static closure contains no FramePose, semantics or framepose FK/IK module, and the CLI's own root yaw is dropped at export (F-01, F-02). | CODE, TEST, BLENDER |
| R3 | W98 GT-free adapter vs GT-gated replay | **Resolved: harness limitation (2) with partial documentation mismatch (4).** Not a runtime GT dependency (3): GT values are behaviourally shown not to affect leg-path outputs. Coverage is GT-gated: 196/2,728 3DHP rows; 3DPW population pre-filtered (F-08). | CODE, DATA, TEST |
| R4 | Source-code tests insufficient for leakage assurance | **Confirmed (MEDIUM)** (F-09) | CODE |
| R5 | Repeated TEST use / adaptive selection | **Confirmed as risk (HIGH)**; concrete post-TEST selections found; magnitude not determined; overfitting not concluded (F-04) | CODE, WL |
| R6 | Numerous standalone scripts and helper dependencies | **Confirmed (MEDIUM)** as coupling and ownership debt; experiment isolation per se is not a defect (F-13) | CODE |
| R7 | Incomplete CI / branch protection | **Confirmed:** none exist (F-12) | API, CODE |
| R8 | Operational disk pressure | **Confirmed, currently less acute:** 84 % / 16 GB free versus 100 % at W98 (F-15) | server |
| R9 | SMPL non-commercial licensing | **Confirmed as a product constraint**, not a code defect; no SMPL file in git | CODE, DATA |
| R10 | Gap between PRODUCT_VISION and implemented authoring | **Confirmed.** Style and authoring are absent (as the Vision itself states); the public animation path is a partial 2–3-bone legacy pipeline; the research animation path is not a runtime (§5) | CODE, TEST |

Issues found beyond R1–R10: F-01, F-03 (default still oracle), F-05, F-07, F-10, F-11 quantification, F-14, F-16, F-17–F-24.

## 17. Final architecture-readiness verdict

**1. Does the implementation respect its documented architecture and historical contracts?**
- *Partially.* The canonical-pose ownership move is genuinely bit-exact at formula level (independently re-derived from 68a5897).
- Architecture_v2 is untouched, and no src module depends on scripts or oracle modules at runtime.
- However, the architecture declared normative for perception (v3 FramePose) is not wired into any runtime. Its successor stages (semantics, torso/pelvis, leg IK) live in experiment scripts.
- Historical metric parity beyond formula level is not enforced in the repository.

**2. Are the accumulated worklog claims justified by code and execution evidence?**
- *Mostly, with qualifications.*
- W93–W98 verdicts are worded consistently with what their scripts compute, and the W98 "GT-free" claim holds for values.
- Earlier H0-based claims rested on GT-fed inputs until W91 disclosed it.
- After W73, verdicts were repeatedly decided on 3DPW TEST, so their TEST numbers are not held-out estimates.
- Two worklogs were rewritten after the append-only rule.

**3. Do critical safety, correctness, leakage or patch-accumulation defects exist?**
- No CRITICAL defect was demonstrated.
- HIGH issues exist: the silent root-yaw loss in the only public animation path, an unreachable normative pipeline, retained oracle defaults, TEST reuse, and public redistribution of research dataset annotations.
- MEDIUM issues: the pickle trust boundary and patch race, source-text-only GT tests, quaternion sign continuity, and script coupling.

**4. Do the research components form a coherent foundation for the animation-authoring framework?**
- *Not yet as a framework; yes as a body of bounded, mostly well-instrumented research components.*
- There is no executable path from unseen real video through FramePose, animation semantics, root/pelvis/thorax orientation and FK/IK to an editable full-body Blender Action.
- The public path that does reach an Action is the legacy MotionGraph pipeline, and it keys only mapped arm bones.
- Style reconstruction, transfer and authoring are absent, as the Vision itself states.

**5. What remains unverified, and why?** See §15. The main gaps:
- licence terms;
- the HMR2 upstream revision and the `animcv-external` image;
- skinned-animation quality across the review cases;
- every historical metric value whose inputs exist only on LabServer63 and were not recomputed in this audit.

**Readiness verdict:** AnimCV at 5cbf77d is a **reproducible-in-part research codebase with a working but narrow legacy
export path. It is not yet a coherent, correctly bounded runtime foundation for animation authoring.** This is a negative,
evidence-bounded outcome for readiness; it is not a judgement on the validity of individual bounded experiments.
