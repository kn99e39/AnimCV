# Worklog — OmniControl Sparse Spatial Guidance for the S1 Left Arm: Checkpoint Terms Preflight Blocks Weight Use (2026-11-10)

Batch 133. An experimental, quality-first batch on `arch/single_frame_first`. It ran unattended on Main:

- local RTX 5080 (16 GB), 64 GB RAM, Windows 11 Pro 10.0.26200;
- AnimCV Python `C:\Projects\AnimCV\.venv_build_full` (3.11.9), unmodified;
- scratch root `C:/Users/dna10/animcv-w133`, outside the repository.

`docs/next_batch.md` was read only. No LabServer and no other machine. Network use was limited to:

- `git fetch/push`;
- a code-only clone of the official OmniControl repository;
- small text-metadata requests to GitHub (repository and issue-search APIs), the official project page, Hugging Face
  (model API record and four small text/JSON files of a third-party re-export), the MDM README, and the AMASS and SMPL
  licence pages;
- HTTP **range** reads of the official checkpoint archive on Google Drive, limited to the ZIP end-of-central-directory
  record and the central directory (the scripted run requested 231 bytes; see §3 for an earlier interactive read).

**No model weights were downloaded. Nothing was installed. No model was run.** Every result below is **AGENT-RUN** on
Main. The terms reading is the agent's documentary evidence chain. It is not legal advice and not a human licensing
decision.

## 1. Official CASE and scope

**Verdict: DATA BLOCKED_LICENSE** (Batch rules: "If rights cannot be established for the proposed use, STOP DATA
BLOCKED_LICENSE"; "Do not treat MIT CODE as a license for external weights"; "do not substitute a mirror"). Source:
`results/preflight/preflight.json`, SHA-256 `93e03bc835dc601040ba38e872e09745a72b028be443d3b354d2664d4b810436`.

The blocker was found before any weight download, runtime install, official smoke, S1 control extraction or prospective
declaration. So **no declaration exists and no model output exists**. This is a blocked preflight. It is **not** a
quality verdict on OmniControl, on sparse spatial guidance, or on the S1 representation. Under the frozen CASE rule
("DATA BLOCKED = necessary rights/weights/data/runtime unavailable"), DATA BLOCKED takes precedence over A–D. No CASE
A/B/C/D is claimed.

**Original task scope (short).** Run the official pretrained OmniControl HumanML3D checkpoint on the frozen real S1
kettlebell clip. Guide it only with source-derived left-wrist 3D positions at model samples 112, 113, 133 and 134, which
lie outside the hidden interval. Project only the guided output's `upperarm_l`/`lowerarm_l` onto a separate Blender
Action at Action 172–199 (SOURCE 250–277). Compare it with A (original), B (W129 slerp) and a matched unguided
same-model diagnostic D, on the same rig at native 30 fps, with the W126V hard pin preserved.

**Historical CASEs are untouched:** W121 D, W122 B, W123 B, W124 B, W125 A (transaction capability only), W126 C,
W127 B, W128 DATA BLOCKED, W129 B, W130 DATA BLOCKED_LICENSE, W131 B, W132 C. No historical Worklog was edited.

## 2. Starting state and revisions

| Item | Value |
|---|---|
| Instruction blob | `20d8def178ad30964393493df855c7aab5be4a6b`, verified first |
| Starting commit | `2ee8c9bf900372cf9ea73bfb4678c74f0fc5cf4a`. Equal to `origin/arch/single_frame_first` at the first fetch. W132 carries the COMPLETE marker, and `docs/AUDIT_W126_W130_2026_11_09.md` is present. |
| Instrument commit | `9d6df97`: `scripts/w133_omnicontrol_preflight.py` + `tests/test_w133_omnicontrol_preflight.py`. Committed before the scripted run and before any weight request. |
| Instrument fix | `3781ea0`: reads the official files from git blobs at the pinned commit (see "Attempt 1" below) |
| Final source revision | `3781ea0`. The Worklog commit is documentation only. The COMPLETE marker is a separate, final commit. |
| OmniControl source | `https://github.com/neu-3dvi/OmniControl`. The README's `neu-vi/OmniControl` redirects to it (GitHub repository id 704213713). Remote `main` = the Batch pin `76336a6e78f074f3ba7059643414b91a1eced9f2` (2024-03-25) on both names. Code-only local clone at that commit, clean worktree. `LICENSE` blob SHA-256 `b579021d…` (MIT, "Copyright (c) 2022 Yiming Xie"). GitHub API licence: MIT. |
| Official checkpoint endpoint | README "model_humanml" → Google Drive file `1oTkBtArc3xjqkYD6Id7LksrTOn3e1Zud`, served as `omnicontrol_ckpt.zip` |

**Disclosed ordering.** I read the README, issue search, project page, archive listing, Motius mirror metadata and
AMASS/SMPL pages interactively *before* writing the script. Those copies are in `meta/`. The rule in the script header
was therefore written knowing those facts. The same is true of W130. The rule mirrors the Batch's own wording ("Do not
treat MIT CODE as a license for external weights", "do not substitute a mirror"). It was not tuned toward a verdict:
each predicate is a fact anyone can re-check.

**Attempt 1 (instrument defect, disclosed).**

- Output: `results/preflight_attempt1_crlf_instrument_error/preflight.json`, SHA-256 `c40a176c…`.
- What went wrong: it compared the mirror's `LICENSE` with the **worktree** copy of the official `LICENSE`. That file had
  been CRLF-converted by `core.autocrlf=true` on Windows. Predicate (e) therefore evaluated false, and the status was
  `STOP_TERMS_UNRESOLVED`. That is still a stop, not a permission.
- The committed blob `76336a6:LICENSE` hashes to `b579021d…`. That is exactly the mirror's `LICENSE`.
- Fix (`3781ea0`): every official file is now read from `git show 76336a6:<path>`. No predicate, threshold or input
  changed. Attempt 2 is the record.

## 3. Checkpoint identity (no weights downloaded)

| Field | Value |
|---|---|
| Archive | `omnicontrol_ckpt.zip`, 161,814,491 bytes (from `Content-Range`); HTTP 206 range support |
| Central directory | 2 entries, 208 bytes at offset 161,814,261, no archive comment. SHA-256 of the directory bytes `d6b976a1…`. |
| Member 1 | `omnicontrol_ckpt/` (directory) |
| Member 2 | `omnicontrol_ckpt/model_humanml3d.pt`: deflate; 161,814,069 bytes compressed, 174,792,507 bytes uncompressed; CRC-32 `272e22f3`; DOS mtime 2023-08-13 02:25:34 |
| Terms / licence / readme / `args.json` member | **none**. The archive holds the weights only. |
| Weight SHA-256 | **NOT VERIFIED.** Weights were not downloaded, so there is no SHA-256 of the `.pt` bytes. CRC-32 and sizes come from the central directory only. |

**Interactive read before the script (disclosed).** Before writing the script, I made two range reads by hand:

- the first 64 bytes (a ZIP local header; no weight data);
- the last 65,536 bytes, to find the central directory. That range also contained about 65 KB of the end of the
  **compressed weight stream**.

That 64 KB fragment was deleted right after the listing was parsed, and before any rule was evaluated. It was never
decompressed, loaded or used. The scripted run requested only the 1-byte probe, the 22-byte EOCD record and the 208-byte
central directory.

## 4. Terms evidence chain (the blocker)

**Declared rule (script header).** Weight use is ALLOWED only if a rights holder of the checkpoint affirmatively
licenses or permits the **weights** (not only the code) for local non-commercial research evaluation at the execution
location. The rule is DATA_BLOCKED_LICENSE if all of (a)–(e) hold. If any predicate is false or un-evidenced, the result
is still a stop (UNRESOLVED), never permission. The rule has no automatic ALLOWED branch: permission would need a
positive documented grant. All five predicates evaluated **true** (`preflight.json` → `terms_rule`).

| # | Predicate | Evidence |
|---|---|---|
| a | The official archive contains no licence/terms member | The central directory lists only the directory and `model_humanml3d.pt` (§3). |
| b | The official README's only grant is scoped to code | `## License`: "This code is distributed under an MIT LICENSE. Note that our code depends on other libraries, including CLIP, SMPL, SMPL-X, PyTorch3D, and uses datasets that each have their own respective licenses that must also be followed." The weights section ("Download the model(s) you wish to use, then unzip and place them in `./save/`") states no terms. No README line ties a licence to weights or checkpoints. |
| c | No official weight terms on the project page or issue tracker | The project page (`neu-3dvi.github.io/omnicontrol/`) has one licence sentence: "All these 3D object models are licensed under Creative Commons Attribution 4.0" (scene assets, not weights). The GitHub issue search for `license`, `licence` and `commercial` returns 0 issues. The `weights`/`checkpoint` hits (#7, #15, #16, #24, #31) are technical questions only. |
| d | The fine-tune parent's grant is also code-scoped | OmniControl training resumes from the MDM checkpoint `model000475000.pt` (official README). MDM README `## License`: "This code is distributed under an MIT LICENSE." It carries the same dataset/dependency caveat (`upstream/mdm_README.md`, SHA-256 `1b9f65d9…`). |
| e | The only weight-scoped "MIT" label comes from a non-rights-holder mirror that copies the code licence | `ZeyuLing/motius-omnicontrol-humanml3d` @ `9f6ca3e2…` (2026-07-25; publisher `ZeyuLing`, not the OmniControl authors). Its card has `license: mit` and its `LICENSE` is **byte-identical** to the official code `LICENSE` blob (`b579021d…`, "Copyright (c) 2022 Yiming Xie"). It is the code licence re-attached to re-exported weights. |

**Conclusion.**

- No rights holder grants a licence or permission for the OmniControl weights.
- The only grant scopes itself to "this code" and defers datasets and dependencies to their own terms.
- Treating that grant as covering the weights is the inference the Batch forbids.
- The mirror's MIT label is the same inference, made by a third party, so it cannot establish rights.
- Rights for the proposed use therefore cannot be established. STOP.

**Deliberately not done (no workaround):**

- no download of the official checkpoint or of the mirror's `model.safetensors`;
- no hosted or remote inference;
- no other machine;
- no alternative model (ProjFlow, Kimodo, ARDY, MotionLab or any other);
- no asking for, or assuming, author permission;
- no mock or hand-coded interpolation built to look like a result.

No commercial, distribution or evaluation rights are claimed for any artifact.

**Recorded, not judged (training data and assets).** The checkpoint is trained on HumanML3D. The HumanML3D repository is
MIT, but its motion features must be derived from AMASS. The AMASS licence is non-commercial scientific research only.
It says: "This license also prohibits the use of the Dataset to train methods/algorithms/neural networks/etc. for
commercial use of any kind", and it has a "No Distribution" clause. Page snapshots: `terms/AMASS.html` (`c34c5030…`) and
`terms/SMPL.html` (`15df6317…`). The declared AnimCV use is local, non-commercial research. The execution locale is
Korea (time zone `Korea Standard Time`, GeoId 134, `ko-KR`). Neither the AMASS nor the SMPL page shows a territorial
exclusion, so this block does not rest on locale. It is the same territory-independent reasoning as (a)–(e).

## 5. Runtime-asset record (secondary; static code reading only)

These facts were established by reading the official code at `76336a6` (git blobs). Nothing was executed.

| Item | Finding |
|---|---|
| r1 SMPL at model construction | `model/cmdm.py` builds `self.rot2xyz = Rotation2xyz(...)`, and `model/rotation2xyz.py` builds `self.smpl_model = SMPL()`. That loads SMPL body-model files from `body_models/smpl`. They are needed even though the HumanML3D `hml_vec` path never calls `rot2xyz`. |
| r2 HumanML3D motion split | `sample/generate.py` `load_dataset` calls `get_dataset_loader(..., split='test', hml_mode='train')`. That is `Text2MotionDatasetV2`, which loads per-clip motion features (`new_joint_vecs`, derived from AMASS) and the GloVe vectorizer, even for a text prompt. |
| r3 SMPL download | `prepare/download_smpl_files.sh` fetches `smpl.zip` from a third-party Google Drive id, not from MPI. The SMPL licence says: "The Software and the license herein granted shall not be copied, shared, distributed …". |
| 196 frames | `generate.py` hard-codes `n_frames = 196`, overriding `--motion_length`. |
| `args.json` | absent from the archive. `utils/parser_util.py` then silently uses parser defaults. That is an unverified configuration risk. |
| Normalisation | The bundled `dataset/humanml_spatial_norm/{Mean_raw,Std_raw}.npy` (SHA-256 `d281690d…` / `53496de7…`) equals the mirror's copies. The HumanML3D GitHub repository (MIT) ships `Mean.npy`/`Std.npy`, with empty `new_joint_vecs/` and `new_joints/`. |

**Status: `RESTRICTED_ASSETS_REQUIRED_BY_OFFICIAL_PATH`.** The official sampling path cannot run with only the
bundled normalisation. It needs an SMPL model and AMASS-derived motion data. Avoiding them would take a runtime patch
(for example, stubbing `Rotation2xyz` and building a text-only loader around the public `Mean.npy`/`Std.npy`). That patch
was **not attempted**: the terms gate comes first, and the Batch allows such a patch only after weight rights are
established. Whether such a patch would count as "narrow" is left open.

## 6. Frozen inputs authenticated (hash-only; nothing rebuilt)

Every check passed (`frozen_evidence_ok: true`).

| Artifact | SHA-256 | Result |
|---|---|---|
| S1 clip (CC BY-SA 4.0 kettlebell side view) | `de8feb9d…` | ok |
| W126 S1 baseline `.blend` (A) and its W129 byte copy | `0803ab2b…` | ok |
| W126V-accepted pin `.blend` and its W129 copy | `7e2ab165…` | ok |
| S1 lineage head | `73dcd3dd…`, value digest `25a09802…` | ok |
| W129 B `.blend` (slerp) / W129 C `.blend` (MotionBricks) | `8899f433…` / `e04efb35…` | ok |
| W129 / W131 / W132 `assessment.json` | `df6693d2…` (B) / `9633a362…` (B) / `35919277…` (C) | ok, CASEs as recorded |
| W131 C `.blend` (CondMDI) | `8f49a93a…` | ok |
| W131 / W132 evidence manifests | `98505f80…` / `4da3c13b…` | ok |

**Pin domain.** SOURCE 210–218 = Action 132–140, target (−0.0927, 0.2199, 1.4506) m, tolerance 1 mm. It does not
overlap the edit interior Action 172–199 (SOURCE 250–277). No frozen file was modified; each was opened for hashing
only.

## 7. Evidence and population

- **Population:** none for model quality. No learned output was produced. The evidence is one preflight report plus
  metadata snapshots.
- **Evidence root:** `C:/Users/dna10/animcv-w133/results`, local and not published. `results/evidence_manifest.json`
  lists 40 files (results + `meta/`), SHA-256 `39d7ae23d9b415294b3eed47d407361c67d967447c1fdd01a350b30cc21eb915`. It was
  written before this Worklog.
- `meta/` holds the agent's earlier interactive copies of the same metadata. They are superseded by `results/preflight/`.
- `ext/omnicontrol` is the code-only clone.
- `scratch_weight_files_present: []`: no `.pt`, `.pth`, `.safetensors` or `.ckpt` file exists anywhere under the
  scratch root.

## 8. Tests and run status (RUN / PASS / FAIL / NOT RUN)

| Item | Kind | Result |
|---|---|---|
| HEAD / instruction-blob verification | real | **RUN, PASS** |
| `tests/test_w133_omnicontrol_preflight.py` (20) | **synthetic**: in-memory ZIPs and fact dictionaries; no network, model or S1 data | **20 PASS**. Covers the ZIP listing from tail ranges, terms-member detection, README licence scoping and the rule truth table (every unmet predicate → UNRESOLVED stop, never ALLOWED). |
| `py_compile` of the new script | static | **PASS** |
| Preflight attempt 1 | REAL metadata | **RUN**: `STOP_TERMS_UNRESOLVED`, caused by an instrument defect (§2) |
| Preflight attempt 2 | REAL metadata, REAL frozen files; no model | **RUN**: exit 3 = `DATA_BLOCKED_LICENSE` (the intended stop); prerequisites other than terms OK |
| Frozen S1 / W126 / W129 / W131 / W132 authentication | real, hash-only | **PASS** |
| Checkpoint terms gate | documentary | **FAIL → STOP** |

**NOT RUN by the stop rule:**

- checkpoint download and SHA-256 verification;
- the scratch runtime;
- the official bundled-example smoke;
- the prospective declaration;
- the S1 wrist-control extraction and the 30→20 fps / 196-frame mapping;
- the axis/scale/left-right/mask representation controls and their fail-closed negative controls (wrong checkpoint hash
  or LFS pointer, wrong handedness, UNKNOWN-joint or hidden-frame control, malformed mask, mismatched frame map,
  protected-channel writes);
- the guided and unguided S1 inference;
- the Blender candidate, fresh-process readback and W126V pin composition;
- the A/B/C/D video, contact sheets and metrics;
- human or animator review.

**Also not run:**

- No regression suite. No existing source, test or instrument changed (`git diff 2ee8c9b -- src` and all pre-existing
  tests are empty), so nothing is affected.

No test failed apart from the terms gate itself.

## 9. Negative findings

- **No weight licence.** The official OmniControl HumanML3D checkpoint ships with no licence or terms of its own, in the
  archive or anywhere official. "Public, ungated Google Drive link in an MIT repository" is not the same as
  "licensed for local evaluation".
- **The mirror's licence is a copy of the code licence.** The Motius re-export's `license: mit` is the OmniControl code
  `LICENSE`, re-attached byte-for-byte to the weights by a non-rights-holder. A mirror label cannot repair the official
  release's missing weight terms.
- **The official runtime needs restricted assets.** Independent of the weights, the official sampler needs SMPL body
  files at model construction and the AMASS-derived HumanML3D motion split for a plain text prompt. The documented SMPL
  download is a third-party copy.
- **No `args.json` in the official archive.** The official parser would silently fall back to defaults. That is an
  unverified configuration risk even for an authorised run.

## 10. Unresolved uncertainties

- **Legal interpretation.** Whether public distribution by the authors with "download the model(s) you wish to use"
  instructions amounts to an implied permission for local research use was **not** decided. The Batch's rule asks for
  established rights, not implied ones. No human, licensor or lawyer was consulted, and the authors were not asked.
- **Consistency with W131 (flagged for the human, not acted on).** W131's rights predicate (a) accepted the Motius
  CondMDI re-export's card `license: mit`, together with the CondMDI code being MIT. W131's own `preflight.json` records
  the official CondMDI README licence section, and its wording is the same code-scoped text: "This code is distributed
  under an MIT LICENSE … datasets that each have their own respective licenses". W131 did not record whether the
  re-export's MIT label is anything more than that code licence carried onto the weights; this batch did not re-check
  it. If it is not, then W131's weight-rights basis has the same structure that W133's Batch rules declare insufficient. W131's
  CASE B and its evidence are **not** revised here. Whether the CondMDI weights' rights need a human re-review is a human
  decision.
- **Undisclosed terms.** The authors could publish weight terms later, or grant permission on request. This run is
  bound to `76336a6`, the archive listing above, and mirror revision `9f6ca3e2…`.
- **Representation and quality.** Nothing about OmniControl's spatial guidance on S1 was established:
  - wrist-control fit;
  - the 160-in-196-frame handling;
  - hint normalisation and the missing-hint encoding;
  - the W131 HumanML3D/Blender mapping reuse;
  - guided vs unguided causality;
  - rig projection;
  - quality against B.

  The Batch's architecture question — whether explicit spatial guidance is a viable source-faithful prior for bounded
  occlusion completion — remains open.

**What this supports:** a documented rights blocker for this checkpoint, and a documented restricted-asset dependency of
its official runtime. **What it does not support:** any claim about OmniControl's quality, controllability or
suitability for AnimCV Animation Authoring, positive or negative.

## 11. Reproduction

Git Bash, repository root at `3781ea0`. The script refuses an existing output directory.

```
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/neu-vi/OmniControl.git C:/Users/dna10/animcv-w133/ext/omnicontrol   # HEAD 76336a6, code only
PY=C:/Projects/AnimCV/.venv_build_full/Scripts/python.exe
$PY -m pytest -q tests/test_w133_omnicontrol_preflight.py                                                  # 20 passed
$PY -X utf8 scripts/w133_omnicontrol_preflight.py --out-dir C:/Users/dna10/animcv-w133/results/preflight
# exit 3, status DATA_BLOCKED_LICENSE; terms_rule a..e all true; runtime_assets RESTRICTED_ASSETS_REQUIRED_BY_OFFICIAL_PATH
```

The script reads the W126 run files, the W129/W131/W132 scratch results and the S1 clip from their recorded local
paths. It fetches only small text files and 231 bytes of the checkpoint archive. A re-run writes a new timestamp, so
its `preflight.json` hash differs. The verdict fields change only if remote metadata changes.

## 12. Changed files

- **New:**
  - `scripts/w133_omnicontrol_preflight.py`
  - `tests/test_w133_omnicontrol_preflight.py`
  - this Worklog
- **Unchanged:**
  - `docs/next_batch.md` and every historical Worklog;
  - all `src/` and every pre-existing test;
  - the W129/W131/W132 adapters, harnesses, runners and declarations;
  - the CLI and GUI; the exporters; FK/IK, FramePose and MotionGraph;
  - the canonical and normative documents; the rig, models and checkpoints; global dependencies;
  - `scripts/batch_automation.py`;
  - every W126/W129/W131/W132 run file.

No living status document was updated: a blocked preflight is not a milestone.

## 13. Resource and environment constraints

| Resource | Cost |
|---|---|
| Disk (scratch) | about 18 MB: the code-only clone plus metadata and evidence |
| Wall time | the scripted preflight takes about 9–13 s (HTTP plus hashing of the frozen `.blend`/clip files) |
| GPU | not used |

**Estimated cost had the run been permitted** (not measured):

- a 162 MB checkpoint download;
- a new scratch PyTorch environment with sm_120 support, separate from the AnimCV 3.11 venv (the W131 CondMDI
  environment could probably be reused);
- an OmniControl sampler with 1000 diffusion steps plus per-step spatial-guidance gradients at 196 frames. That is
  likely minutes per sample on the RTX 5080, against seconds for W131's 100-step CondMDI.
