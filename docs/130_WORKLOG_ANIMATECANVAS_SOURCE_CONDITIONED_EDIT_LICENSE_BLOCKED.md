# Worklog — AnimateCanvas Source-Conditioned Left-Arm Edit: Checkpoint Terms Preflight Blocks Local Evaluation (2026-11-09)

Batch 130. An experimental, quality-first batch on `arch/single_frame_first`. It ran unattended on Main:

- local RTX 5080 (16 GB), driver 596.49, 64 GB system RAM, Windows 11 Pro 10.0.26200;
- AnimCV Python `C:\Projects\AnimCV\.venv_build_full` (3.11.9);
- scratch root `C:/Users/dna10/animcv-w130`, outside the repository.

`docs/next_batch.md` was read only. No LabServer was used. Network use was limited to `git fetch/push`, the code-only
clone of the AnimateCanvas repository (no LFS objects), and small text-metadata requests to Hugging Face, GitHub and
raw.githubusercontent.com. **No model weights were downloaded.**

Every result below is **AGENT-RUN** on Main. Nothing was independently re-executed. The terms reading is the agent's
documentary evidence chain, not legal advice and not a human licensing decision.

## 1. Official CASE and scope

**Verdict: DATA BLOCKED_LICENSE** (Batch rule: "If terms or access prohibit local evaluation, STOP `DATA
BLOCKED_LICENSE` without attempting a workaround"). Source: `results/preflight/preflight.json`, SHA-256
`08144055d38c43e6fc074f08d0e08a44e35c5f0e7e5b92f3d6def087697de3b6`.

The blocker was found before any weight download, mapping adapter, mask declaration or inference, so **no pre-outcome
declaration exists and no model output exists**. This is a blocked preflight. It is **not** a quality verdict on
AnimateCanvas, on source-conditioned masked editing, or on the partial-body representation. Under the frozen CASE rule,
DATA BLOCKED takes precedence over A–D. No CASE A/B/C/D is claimed.

**Original task scope (short).** Run the public pretrained AnimateCanvas 0.46B motion-to-motion model on the frozen
real S1 kettlebell clip. It was to do a text-guided, channel-masked edit of the `upperarm_l`/`lowerarm_l` local
rotations at Action 172–199 (SOURCE 250–277). The result would go into a new isolated Blender Action, keep the W126
hard pin and all out-of-domain tracks bit-identical, and be compared at native 30 fps with A (original), B (W129 slerp)
and C (W129 MotionBricks).

**Historical CASEs are untouched:** W121 D, W122 B, W123 B, W124 B, W125 A (transaction capability only), W126 C,
W127 B, W128 DATA BLOCKED, W129 B. No historical Worklog was edited.

## 2. Starting state and revisions

| Item | Value |
|---|---|
| Instruction blob | `550c8a63f1606f9dd47b5029b7ccd635c9bc1812`, verified first |
| Starting commit | `65e6d8f7f735dca4a8642ab7c03881fd6e699d03`; equal to `origin/arch/single_frame_first` at the first check. W129 carries the COMPLETE marker. |
| Final source revision | `1de33d0` (adds `scripts/w130_animatecanvas_preflight.py`). The commit adding this Worklog changes documentation only. The COMPLETE marker is a separate, final commit. |
| AnimateCanvas source | `https://github.com/ZeyuLing/AnimateCanvas`. Remote `main` = the Batch pin `e59d68184ab8cb7852f598d688183f62303ae45e` (2026-09-18). Code-only local checkout at that commit, clean worktree. `LICENSE` is MIT (SHA-256 `c08dd9ec…`). |
| HF checkpoint | `ZeyuLing/Motius-AnimateCanvas-0.46B` at revision `6ef7830c5872b747a938251958c588e5e291107e` (lastModified 2026-10-22). The legacy id `ZeyuLing/Motius-MotionCanvas-0.46B` redirects to the same repo and revision. Public and not gated. |
| Motius (upstream implementation, model card) | `ZeyuLing/Motius` `main` = `d04a38c9bc9dab3f0e9e2d843f8a6b21eaa13abc` |
| Upstream warm-start card | `ZeyuLing/Motius-HYMotion-T2M-1.0-Lite` at `128743eb5971d0164a8ef91ae8d85920b9348855` |
| Upstream original weights + licence | `tencent/HY-Motion-1.0` at `620dd559f8d964aac2f82f1204fe6a35ad8ad14d`; `LICENSE.txt` SHA-256 `03e846e8bcaae503dbb878cb5b125b9810d8cac8bfc38b0275c4886759afd81b` |

## 3. Checkpoint identity (recorded without download)

The HF tree API at the pinned revision gives the LFS identity of every weight file. The total artifact size is
20,007,146,995 bytes.

| File | Bytes | LFS SHA-256 |
|---|---|---|
| `motion_transformer.safetensors` | 1,845,042,584 | `33c98ff00f59d38091400651f1b8ebc956b2a09b4650b85626e7d8fac5156c19` |
| `text_encoder/llm/model-0000{1..5}-of-00005.safetensors` (Qwen3-8B) | 16,381,516,776 in total | per-shard OIDs in `preflight.json` |
| `text_encoder/sentence/model.safetensors` (CLIP-L) | 1,710,540,580 | `a2bf730a…` |
| `Mean.npy` / `Std.npy` / `bone_offsets_22.pt` | 920 / 920 / 1,476 | `8b1ed0c2…` / `17d00d53…` / `7663aaa8…` |

**Metadata read at the pinned revision** (`hf_meta/`, hashes in `preflight.json`):

- `source_checkpoint.json`: epoch 2354, `MotionCanvas-198`, `native_fps` 30.
- `animatecanvas_config.json`: `AnimateCanvasMMDiT`, input 594 / output 198, 18 layers, width 1024, `motion_type`
  `smpl_22`, velocity prediction. Text encoder `HYTextModel` = Qwen3 (`max_length` 128) + CLIP-L (77).
- The card documents: 360-frame canvas at 30 fps; root translation (3) + row-major 6D local rotations for 22 SMPL
  joints (132) + pelvis-relative FK positions for 21 joints (63); `generation_mask` 1 = generate, 0 = preserve; an
  `infer_motion_editing` entry point.
- `normalization_provenance.md`: the 198-D statistics are the HY-Motion 1.0-Lite 201-D statistics minus the pelvis-RIC
  triplet. They were chosen *for "HY-Motion 1.0-Lite warm-start"*.

The Batch's other technical checks were **NOT RUN**, because the terms gate comes first:

- 360-frame canvas behaviour;
- SMPL-22 joint order and 6D layout against code;
- VRAM and runtime.

## 4. Terms evidence chain (the blocker)

The rule was declared in the script header before any weight request: inference is allowed only if the
motion-transformer checkpoint carries an affirmative licence for local experimental evaluation at the execution
location. Every predicate below evaluated **true**. Source: `preflight.json` → `terms_rule`.

| # | Predicate | Evidence |
|---|---|---|
| a | The checkpoint declares no licence of its own | HF `cardData.license` is absent. The model card has no licence field or section. Its "Provenance" section says: "Its transformer and text-conditioning design build on HY-Motion 1.0". |
| b | The checkpoint is a warm-started derivative of HY-Motion 1.0 | Motius AnimateCanvas model card (pinned `d04a38c9`, SHA-256 `787036ae…`): "It warm-starts the Text-to-Motion backbone from `ZeyuLing/Motius-HYMotion-T2M-1.0-Lite`". Its training mixture is 90% HYMotion data, which "requires separate authorized access". That T2M card is `license: other`, and its "Original weights" are `tencent/HY-Motion-1.0`. The AnimateCanvas code says the same ("warm-start checkpoints carry the learned null … embeddings"; "HYMotion-Lite warm starts"). |
| c | The upstream licence excludes South Korea | `tencent/HY-Motion-1.0` `LICENSE.txt` = "TENCENT HY-MOTION 1.0 COMMUNITY LICENSE AGREEMENT" (HF `license_name` `tencent-hunyuan-community`). Line 3: "THIS LICENSE AGREEMENT DOES NOT APPLY IN THE EUROPEAN UNION, UNITED KINGDOM AND SOUTH KOREA …". "Territory" = "worldwide …, excluding … South Korea". §5(c): "You must not use … the Tencent HY-MOTION 1.0 Works, Output or results … outside the Territory." The Acceptable Use Policy, item 1: not "Outside the Territory". |
| c2 | Its derivative definition covers weight transfer | "Model Derivatives" include "(i) modifications to Tencent HY-MOTION 1.0 …; (ii) works based on Tencent HY-MOTION 1.0 …". "Tencent HY-MOTION 1.0 Works" include Model Derivatives. |
| d | The execution locale is South Korea | Main reports time zone `Korea Standard Time`, Windows home location GeoId `134` (Korea), culture and system locale `ko-KR`. Supporting context: the repository commits are +0900, the central documents are Korean, and the user's institutional e-mail domain is Korean. |
| e | The MIT code licence does not license the weights | AnimateCanvas README: "This MIT grant does not relicense upstream code or third-party assets. Checkpoints, training data, body models, and character meshes are not included." The Motius repository has no root `LICENSE` file (HTTP 404 at `d04a38c9`). Its card defers to "respective upstream licenses and access terms". |

**Conclusion.**

- No document grants a licence for the AnimateCanvas weights.
- The only licence documented in their provenance chain explicitly does not apply in South Korea, and it forbids use of
  its Works and derivatives outside its Territory.
- Downloading and running the checkpoint on Main would therefore be unlicensed local evaluation.

The Qwen3-8B (Apache-2.0) and CLIP-L text encoders are not the blocker. The motion transformer is.

**Deliberately not done (no workaround):**

- no weight download;
- no hosted or remote inference;
- no other machine or region;
- no substitute or alternative model, and no other HY-derived Motius checkpoint;
- no adapter or "mock" built to look like a result.

No commercial, distribution or evaluation rights are claimed for any artifact.

## 5. Frozen inputs authenticated (hash-only; nothing rebuilt)

Every check passed (`frozen_evidence_ok: true`). These inputs are byte-identical to the W129 record.

| Artifact | SHA-256 | Result |
|---|---|---|
| S1 clip (CC BY-SA 4.0 kettlebell side view) | `de8feb9d…` | ok |
| W126 S1 baseline `.blend` (A) and its W129 byte copy | `0803ab2b…` | ok |
| W126V-accepted pin `.blend` and its W129 copy | `7e2ab165…` | ok |
| S1 lineage head | `73dcd3dd…` (op `REVERT`), value digest `25a09802…` | ok, no active pin |
| W129 B `.blend` (slerp) | `8899f433…` | ok |
| W129 C `.blend` (MotionBricks) | `e04efb35…` | ok |
| W129 Cpin `.blend` | `3116d2ea…` | ok |
| W129 A/B/C 30 fps video | `f35420de…` | ok |
| W129 `assessment.json` | `df6693d2…`, CASE B | ok |
| W129 evidence manifest | `b07154ed…` | ok |

**Pin domain.** SOURCE 210–218 = Action 132–140, target (−0.0927, 0.2199, 1.4506) m, tolerance 1 mm. It does **not**
overlap the frozen edit interior Action 172–199 (SOURCE 250–277).

No original, W126 lineage or W129 file was modified. They were opened for hashing only.

## 6. Evidence and population

- Population: none for model quality. No learned output was produced. The evidence is one preflight report and
  15 hashed scratch files, all metadata.
- Evidence root: `C:/Users/dna10/animcv-w130/results`, local and not published. `evidence_manifest.json` (15 files)
  has SHA-256 `c083990eebf09246e0947ad92e321b02d67dfc777697e984b2afc34b7c36e475`.
- `C:/Users/dna10/animcv-w130/hf/` holds the agent's earlier interactive copies of the same metadata (fetched before
  the script was written). They are superseded by the script's `results/preflight/hf_meta` and `upstream/`.
- `ext/animatecanvas` is the code-only clone (370 MB, mostly `gh-pages` history in the pack; no LFS objects).
- No `*.safetensors` or large `*.bin` files exist anywhere under the scratch root (`scratch_weight_files_present: []`).

## 7. Tests and run status (RUN / PASS / FAIL / NOT RUN)

| Item | Kind | Result |
|---|---|---|
| HEAD / instruction-blob verification | real | **RUN, PASS** |
| `scripts/w130_animatecanvas_preflight.py` | REAL metadata, REAL frozen files; no model | **RUN**: exit 3 = `DATA_BLOCKED_LICENSE` (intended stop) |
| `py_compile` of the new script | static | **PASS** |
| Frozen S1 / W126 / W129 authentication | real, hash-only | **PASS** |
| Checkpoint licence/terms gate | documentary | **FAIL → STOP** |

**NOT RUN by the stop rule:**

- the official small-inference smoke test;
- the S1 primary masked edit, the no-text control and the determinism rerun;
- the SMPL-22 ↔ BaseRig mapping adapter and its controls (identity, yaw, left/right, parent-child, arm bend, protected
  frames) and its fail-closed tests (wrong skeleton, reversed axes, wrong 6D layout, invalid mask, incomplete identity);
- the channel-mask and UNKNOWN-coverage declaration;
- the pre-outcome declaration;
- the Blender candidate, fresh-process readback and hard-pin composition;
- the A/B/C/D renders, video and metrics;
- a human review.

**Also not run:**

- No unit tests were added. The only code added is the network/locale preflight, which has no model path to test.
- No regression suite was run. No existing source, test or W129 instrument changed (`git diff 65e6d8f -- src tests
  scripts/w129_*` is empty), so nothing is affected.

No test failed apart from the terms gate itself. **Watched:** NOT WATCHED — there is nothing to watch.

## 8. Negative findings

- The Batch's candidate checkpoint has **no licence of its own**. Its public card lists `library_name` and tags, but no
  `license`.
- The documented training lineage ties it to Tencent HY-Motion 1.0 through a warm start, and that licence excludes
  South Korea. "Public and ungated on Hugging Face" is therefore not the same as "licensed for local evaluation here".
- The same upstream territory exclusion would also apply to other HY-Motion-derived Motius checkpoints. Not evaluated
  further.
- The AnimateCanvas README itself states that "full generation with the public large checkpoint and pretrained text
  encoders has not yet been validated in this release environment", and that GPU memory needs are not benchmarked.
  Recorded, not tested.

## 9. Unresolved uncertainties

- **Legal interpretation.** Whether a warm-started fine-tune is a "Model Derivative" was read from the licence's own
  definitions and the publishers' own provenance statements. No human, licensor or lawyer confirmed it. Whether the
  rights holders would grant permission for evaluation in South Korea was not asked.
- **Physical location.** Main's physical location is inferred from its OS locale and time zone plus repository context.
  It was not independently confirmed.
- **Undisclosed terms.** The HF card could gain a licence later, or the authors could clarify terms. This run is bound
  to revision `6ef7830c…` and the upstream revisions above.
- **Representation and quality.** Nothing about the representation (SMPL-22 mapping, 6D layout, mask coupling of
  rotations and FK positions, the 360-frame canvas versus the 240-frame clip) or about model quality was established.
  The architectural question of the Batch remains open.

## 10. Reproduction

Git Bash, repository root at `1de33d0`. The script refuses an existing output directory.

```
git clone https://github.com/ZeyuLing/AnimateCanvas.git C:/Users/dna10/animcv-w130/ext/animatecanvas   # code only
git -C C:/Users/dna10/animcv-w130/ext/animatecanvas checkout e59d68184ab8cb7852f598d688183f62303ae45e
PY=C:/Projects/AnimCV/.venv_build_full/Scripts/python.exe
$PY -X utf8 scripts/w130_animatecanvas_preflight.py --out-dir C:/Users/dna10/animcv-w130/results/preflight
# exit 3, status DATA_BLOCKED_LICENSE; all terms_rule predicates true; frozen_evidence_ok true
```

The script reads the W126 run files, the W129 scratch results and the S1 clip from their recorded local paths. It
fetches only small text files, at pinned revisions. A re-run writes a new timestamp, so its `preflight.json` hash
differs. The verdict fields do not change unless the remote metadata changes.

## 11. Changed files

- **New:**
  - `scripts/w130_animatecanvas_preflight.py`
  - this Worklog
- **Unchanged:**
  - `docs/next_batch.md` and every historical Worklog;
  - all `src/` and `tests/`;
  - the W129 adapter, Blender stage, harness and declarations;
  - the CLI and GUI; the exporter; FK/IK, FramePose and MotionGraph;
  - the canonical documents; the rig, models and checkpoints; global dependencies; the automation scripts;
  - every W126/W129 run file.

## 12. Resource and environment constraints

| Resource | Cost |
|---|---|
| Disk (scratch) | about 371 MB: the code-only clone plus 0.3 MB of metadata and evidence |
| Wall time | the preflight takes seconds (not formally timed; dominated by hashing the S1 clip and `.blend` files and by HTTP) |
| GPU | not used |

**Estimated cost had the run been permitted** (not measured):

- about 20 GB download;
- Qwen3-8B text encoding on CPU (README: "substantial system RAM"; 64 GB is available);
- a 0.46B transformer on the 16 GB RTX 5080, whose VRAM needs the authors have not benchmarked;
- a new PyTorch ≥ 2.6 scratch environment with Blackwell (sm_120) support, separate from the AnimCV 3.11 venv.
