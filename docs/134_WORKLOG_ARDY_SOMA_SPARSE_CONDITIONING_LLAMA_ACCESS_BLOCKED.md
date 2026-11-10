# Worklog — ARDY SOMA 30 fps Sparse-Position Conditioning on S1: Gated Llama 3 Text-Encoder Access Blocks the Run (2026-11-10)

Batch 134. An experimental, quality-first batch on `arch/single_frame_first`. It ran unattended on Main:

- local RTX 5080 (16 GB, driver 596.49), 64 GB RAM, Windows 11 Pro 10.0.26200;
- AnimCV Python `C:\Projects\AnimCV\.venv_build_full` (3.11.9), unmodified (stdlib only was used);
- scratch root `C:/Users/dna10/animcv-w134`, outside the repository.

`docs/next_batch.md` was read only. No LabServer and no other machine. Network use was limited to:

- `git fetch/push`;
- a code-only clone of the official ARDY repository (`GIT_LFS_SKIP_SMUDGE=1`);
- Hugging Face model **metadata** (API records) and small text/JSON files: the NVIDIA checkpoint's `README.md` and
  `config.yaml`, and the two LLM2Vec `adapter_config.json`/`config.json` files;
- one authenticated `whoami` call and one authenticated plus one anonymous **HEAD** request for
  `meta-llama/Meta-Llama-3-8B-Instruct/config.json` (no body, redirects not followed).

**No model weights were downloaded. Nothing was installed. No model was run.** Every result below is **AGENT-RUN** on
Main. The rights reading is documentary evidence, not legal advice and not a human licensing decision.

## 1. Official CASE and scope

**Verdict: DATA BLOCKED — ACCESS_BLOCKED.** The Batch's rules: "If absent, STOP ACCESS_BLOCKED and record the exact
missing authorization"; "STOP on missing authorized Llama access"; CASE precedence "DATA BLOCKED = required authorized
weights/encoder/runtime/data unavailable". Source: `results/preflight/preflight.json`, SHA-256
`3adf705d2cf2940d88b26dd06bf0f8c595b15b3cbe4cf69e06d153819217ad1a`.

**Exact missing authorization.** The existing local Hugging Face credential authenticates as user account `kn99e39`
(the same handle as this repository's commit author noreply address). That account is **not on the authorized list**
for the gated (`gated: manual`, licence `llama3`) repository `meta-llama/Meta-Llama-3-8B-Instruct`. The authenticated
HEAD returned **403** with `X-Error-Code: GatedRepo` and "Access to model meta-llama/Meta-Llama-3-8B-Instruct is
restricted and you are not in the authorized list". The official ARDY text encoder cannot be built without that
repository's weights (§4).

The blocker was found before any weight download, runtime install, official smoke, S1 anchor extraction or prospective
declaration. So **no declaration exists and no model output exists**. This is a blocked preflight, **not** a quality
verdict on ARDY, on sparse-position conditioning, or on the S1 representation. No CASE A/B/C/D is claimed.

**Original task scope (short).** Run the official NVIDIA ARDY-SOMA-RP-30FPS-Horizon60 checkpoint on the frozen real S1
kettlebell clip at native 30 fps, once guided by four source-derived left-wrist positions only (Action 170, 171, 200,
201, all outside the hidden interval) and once unguided with the same seed 0 and fixed prompt. Project only the guided
output's `upperarm_l`/`lowerarm_l` onto a separate Blender Action at Action 172–199 (SOURCE 250–277). Compare with A
(original), B (W129 slerp) and D (unguided diagnostic) on the same rig, with the W126V hard pin preserved.

**Historical CASEs are untouched:** W121 D, W122 B, W123 B, W124 B, W125 A (transaction capability only), W126 C,
W127 B, W128 DATA BLOCKED, W129 B, W130 DATA BLOCKED_LICENSE, W131 B, W132 C, W133 DATA BLOCKED_LICENSE. No historical
Worklog was edited.

## 2. Starting state and revisions

| Item | Value |
|---|---|
| Instruction blob | `ec2f2c167e7275cf76909143c18f71cdd214ea18`, verified first |
| Starting commit | `eab3962d3389ea779ad2332d06f3bac9d542e500`, equal to `origin/arch/single_frame_first` at the first fetch. W133 carries the COMPLETE marker. |
| Instrument commit | `2035794`: `scripts/w134_ardy_access_preflight.py` + `tests/test_w134_ardy_access_preflight.py`. Committed before the scripted run. |
| Final source revision | `2035794`. The Worklog commit is documentation only. The COMPLETE marker is a separate, final commit. |
| ARDY source | `https://github.com/nv-tlabs/ardy.git`; remote `HEAD` = the Batch pin `e706e27acc88070918308a10922b8f7174d0825e` (committed 2026-10-15T22:02:26+02:00). Code-only local clone at that commit, clean worktree. Official files read from git blobs at the pin. `LICENSE` SHA-256 `411f74af…` (Apache-2.0, "Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES"). |

**Disclosed ordering.** Before writing the script, I interactively:

- cloned ARDY and read its README, `ardy/model/load_model.py`, `ardy/model/llm2vec/*`, `ardy/model/registry.py`,
  `ATTRIBUTIONS.MD` and the head of `ardy/constraints.py`;
- checked that the HF environment variables were unset, listed `~/.cache/huggingface/` (a `token` file and a
  `stored_tokens` file exist, dated 2026-09-12; the cached models include no Llama or LLM2Vec repository), and printed
  `stored_tokens` with every token value redacted by `sed` (only the label `mainSystem` was visible).

I did **not** query Hugging Face for access before the script existed. The access rule in the script header was
committed (`2035794`) before the first and only scripted run. The rule is the Batch's own wording made mechanical: an
existing local credential must already hold granted access; there is no fallback branch.

## 3. Checkpoint identity (metadata only; no weights downloaded)

| Field | Value |
|---|---|
| Repository | `nvidia/ARDY-SOMA-RP-30FPS-Horizon60`, author `nvidia`, not gated, last modified 2026-10-15T20:36:56Z |
| Pinned revision | `244215084b22fcd29ab743017330e87411fe168a` |
| `denoiser.safetensors` | 626,861,352 bytes; published LFS SHA-256 `f548632f58e9ea1d2f717a1415cae3c1b86dadb521a2549e3e7ea60780bd8dc5` |
| `tokenizer.safetensors` (motion FSQ-VAE) | 142,169,376 bytes; published LFS SHA-256 `68d0425ae89952123e3286cac4d29371c9c236af307f10923a3a79ff62958a11` |
| `config.yaml` | 1,588 bytes, SHA-256 `2141df95…` (downloaded; text) |
| Model card `README.md` | 13,362 bytes, SHA-256 `a7c83d36…` (downloaded; text) |
| Weight SHA-256 | **NOT DOWNLOADED.** The two values above are the hashes Hugging Face publishes in LFS metadata. They were not checked against local bytes. |

**Card and config facts (recorded, not tested).**

- Card licence: `license: other`, `license_name: nvidia-open-model-agreement`; "Use of this model is governed by the
  NVIDIA Open Model Agreement"; Deployment Geography: Global.
- Card: 192 M parameters (denoiser 157 M + motion tokenizer 36 M; "text encoder not included"); 30-joint SOMA at 30 fps;
  input = text + pose constraints + history poses; history max 8 s; output = global root translation + joint rotations,
  max 8 s.
- Training data: Bones Rigplay 1, 630 h of optical motion capture with text labels, paraphrased by an LLM.
- `config.yaml`: `gen_horizon_len: 60`, `num_base_steps: 10`, `cfg_type: separated`, `ArdyMotionRep` with `fps: 30`
  and `SOMASkeleton30`, `num_frames_per_token: 4`, `motion_mask_mode: concat`, and `llm_shape: [1, 4096]` — the
  denoiser is conditioned on a 4096-dimensional LLM text embedding.

**Rights reading.** The NVIDIA checkpoint names its own weight terms (the NVIDIA Open Model Agreement). That is distinct
from the Apache-2.0 code licence, and the official README says so ("model checkpoints and data are licensed separately
as indicated on the HuggingFace download pages"). The Agreement text itself was **not** fetched or reviewed in this run.
The access gate (§4) stops the run before checkpoint rights could matter, so whether the Agreement covers this local
research use was **not decided**.

## 4. Text-encoder dependency chain and the access gate (the blocker)

**Official default encoder** (`ardy/model/load_model.py` at the pin): `DEFAULT_TEXT_ENCODER = "llm2vec"`, which loads

- base `McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp` @ `31474e39…`: MIT card licence, not gated. It holds only a
  PEFT adapter, tokenizer and `config.json`; no full base weights. Its `adapter_config.json` (`1d3bbf14…`) and
  `config.json` both name `base_model_name_or_path` / `_name_or_path` = `meta-llama/Meta-Llama-3-8B-Instruct`;
- PEFT `McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised` @ `baa8ebf0…`: MIT, not gated, adapter only;
  `adapter_config.json` (`2a1265e4…`) also names the Llama 3 base.

`LLM2Vec.from_pretrained` builds `LlamaBiModel` from the base path. The adapter resolves to the Llama 3 base weights.
The official README states: "The text encoder relies on the gated meta-llama/Meta-Llama-3-8B-Instruct model, which
requires: Your Hugging Face account has been granted access to the model page. You provide a Hugging Face token at
runtime."

**Gated base:** `meta-llama/Meta-Llama-3-8B-Instruct` @ `8afb486c…`, `gated: manual`, card licence `llama3`.

**Declared access rule (script header) and evaluation** (`preflight.json` → `access_rule`):

| # | Predicate | Result | Evidence |
|---|---|---|---|
| a | The default encoder resolves to the gated base | **true** | both adapter configs name `meta-llama/Meta-Llama-3-8B-Instruct` |
| b | An existing local credential is present | **true** | `~/.cache/huggingface/token` (mtime 2026-09-12T23:31:47); `HF_TOKEN` unset |
| c | The credential authenticates | **true** | `whoami-v2`: user `kn99e39`, token role `fineGrained` |
| d | Gated-base access is granted to that account | **false** | authenticated HEAD **403**, `GatedRepo`, "you are not in the authorized list"; anonymous HEAD 401 |

→ `ACCESS_BLOCKED`, `missing_authorization = ["account has no granted access to meta-llama/Meta-Llama-3-8B-Instruct
(HEAD class NOT_GRANTED)"]`. The token value was never printed, logged or written. A recursive grep of the evidence
root for `hf_` token strings finds nothing.

**Checked and rejected (no workaround):**

- `load_model(..., text_encoder=False)` is an official path, but it removes text conditioning entirely, while
  `config.yaml` conditions the denoiser on a `[1, 4096]` LLM embedding. Feeding it a zero, random or other-model
  embedding is the "unvalidated embedding" the Batch forbids. It would also break the frozen prompt.
- The text-encoder `api` mode (default `http://127.0.0.1:9550/`) needs the same Llama 3 weights on the serving side. An
  external encoder API is forbidden.
- The demo's `EmbeddingCache` only caches embeddings produced by the same encoder. No precomputed embeddings ship with
  the pinned source or the checkpoint.
- Not done: requesting Llama access, accepting the Llama 3 licence, using another person's token, an unofficial Llama
  mirror or re-upload, another LLM2Vec base, Kimodo, the KimodoRep or older ARDY variants, a hosted or remote run,
  another machine, or a hand-made interpolation presented as a model result.

**Llama 3 terms** were not reviewed for this use: they only become relevant once access exists. Note that the
Hugging Face access grant itself requires the account holder to accept those terms.

## 5. Frozen inputs authenticated (hash-only; nothing rebuilt)

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
| W131 / W132 / W133 evidence manifests | `98505f80…` / `4da3c13b…` / `39d7ae23…` | ok |
| W133 `preflight.json` | `93e03bc8…` | ok |

**Pin domain.** SOURCE 210–218 = Action 132–140, target (−0.0927, 0.2199, 1.4506) m, tolerance 1 mm. It does not
overlap the edit interior Action 172–199 (SOURCE 250–277). The declared anchor times Action 170/171/200/201 lie outside
the interior. No frozen file was modified; each was opened for hashing only.

## 6. Evidence and population

- **Population:** none for model quality. No learned output was produced. The evidence is one preflight report plus
  small metadata files.
- **Evidence root:** `C:/Users/dna10/animcv-w134/results`, local and not published. `results/evidence_manifest.json`
  lists 10 files, SHA-256 `b83bd30d1e45aaefd50d0e511ea4949ece177c78a3017e660adbdbdf186ef068`. It was written before
  this Worklog.
- `ext/ardy` is the code-only clone (about 77 MB, mostly bundled skeleton meshes).
- `scratch_weight_files_present: []`. No `.safetensors`, `.pt`, `.pth`, `.ckpt`, `.onnx` or engine file exists under
  the scratch root.

## 7. Tests and run status (RUN / PASS / FAIL / NOT RUN)

| Item | Evidence tier | Result |
|---|---|---|
| HEAD / instruction-blob verification | real | **RUN, PASS** |
| `tests/test_w134_ardy_access_preflight.py` (19) | **synthetic unit**: fact dictionaries and source snippets; no network, model or S1 data | **19 PASS**. Covers HEAD-status classification, the encoder-preset parse, the access-rule truth table (each unmet predicate → ACCESS_BLOCKED with one stated reason; no fallback) and the no-redirect handler. |
| `py_compile` of the new script | static | **PASS** |
| Preflight (single scripted run) | REAL metadata, REAL frozen files; no model | **RUN**: exit 3 = `ACCESS_BLOCKED` (the intended stop). Prerequisites other than access OK. |
| Frozen S1 / W126 / W129 / W131 / W132 / W133 authentication | real, hash-only | **PASS** |
| Gated text-encoder access gate | real (HF access check of the user's own existing credential) | **FAIL → STOP** |

**NOT RUN by the stop rule:**

- G0: checkpoint download and local SHA-256 verification; NVIDIA Open Model Agreement text review; Llama 3 terms
  review; scratch runtime (PyTorch sm_120 environment); RTX 5080 16 GB fit of the bf16 encoder (README: about 14 GB VRAM)
  plus the model; the official example smoke; the prospective declaration;
- G1: the `observed_motion` / `motion_mask` trace through `ArdyMotionRep` and the denoiser; the wrist-position-only mask
  proof; SOMA30 left/right identity, axes and FK; the 240-frame / 60-frame-horizon autoregressive time mapping; and the
  negative controls (left/right swap, sign/scale, malformed mask, hidden-frame constraint, unsupported joint, implicit
  root/heading/rotation channel);
- S1 left-wrist anchor extraction and provenance at Action 170/171/200/201;
- G2: guided and unguided S1 inference;
- G3: Blender projection, fresh-process readback and W126V pin composition;
- G4: the A/B/C/D video, contact sheets and metrics;
- human or animator review.

**Also not run:**

- No regression suite. No existing source or test changed (`git diff eab3962 -- src` and all pre-existing tests are
  empty), so nothing is affected.

No test failed apart from the access gate itself.

## 8. Negative findings

- **The model card's "text encoder not included" is literal.** ARDY's checkpoint is open under NVIDIA's agreement, but
  every official text-conditioned path needs Llama 3 8B Instruct weights. The checkpoint is therefore not usable on its
  own for the Batch's fixed-prompt experiment without a separate, manually gated grant.
- **The LLM2Vec repositories are adapters only.** Their MIT licence and ungated status do not remove the dependency:
  both `adapter_config.json` files point to the gated Llama 3 base. MIT on LoRA adapters is not access to, or rights
  in, the base.
- **The local credential is real but not sufficient.** The user's own fine-grained token authenticates. The account has
  not been granted Llama 3 access. Nothing on this machine caches Llama or LLM2Vec weights.
- **Recorded, not judged:** the convenience class `EndEffectorConstraintSet` takes a `root_2d` argument besides joint
  positions/rotations. That matches the Batch's caution, but the low-level mask path was not traced (G1 NOT RUN).

## 9. Unresolved uncertainties

- **Representation and quality: entirely open.** Whether ARDY's `motion_mask` can express left-wrist XYZ alone at four
  frames, whether that mask is in-distribution, how the 240-frame clip maps onto 60-frame autoregressive windows with
  4-frame tokens and 8 s history, and the guided-vs-unguided causality and S1 quality were not tested. Nothing here
  supports or refutes the sparse-conditioning hypothesis.
- **NVIDIA Open Model Agreement scope** for this local research use was not reviewed (the run stopped earlier).
- **Whether `kn99e39` is the only account the user holds** is not known. The agent used only the credential already
  stored in this Windows profile. Granting Llama 3 access to an account, and accepting Meta's licence, is a human
  action outside this Batch.
- **Time sensitivity.** Access status, repository revisions and card text can change. This run is bound to ARDY
  `e706e27`, checkpoint revision `24421508…`, LLM2Vec revisions `31474e39…` / `baa8ebf0…`, Llama revision `8afb486c…`,
  and the 2026-11-10 access response.

**What this supports:** a documented, exact access blocker for the official ARDY SOMA 30 fps text-conditioned path on
this machine, and the metadata identity of the checkpoint. **What it does not support:** any claim about ARDY's
quality, controllability or suitability for AnimCV Animation Authoring, positive or negative.

## 10. Reproduction

Git Bash, repository root at `2035794`. The script refuses an existing output directory.

```
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/nv-tlabs/ardy.git C:/Users/dna10/animcv-w134/ext/ardy
git -C C:/Users/dna10/animcv-w134/ext/ardy checkout e706e27acc88070918308a10922b8f7174d0825e
PY=C:/Projects/AnimCV/.venv_build_full/Scripts/python.exe
$PY -m pytest -q tests/test_w134_ardy_access_preflight.py                                       # 19 passed
$PY -X utf8 scripts/w134_ardy_access_preflight.py --out-dir C:/Users/dna10/animcv-w134/results/preflight
# exit 3, status ACCESS_BLOCKED; access_rule a,b,c true, d false (HEAD 403 GatedRepo)
```

The script reads the frozen W126/W129/W131/W132/W133 files and the S1 clip from their recorded local paths. It uses the
existing token in `~/.cache/huggingface/token` (or `HF_TOKEN`) only for `whoami` and one HEAD. A re-run writes a new
timestamp, so its `preflight.json` hash differs. The verdict changes only if access or remote metadata changes.

## 11. Changed files

- **New:**
  - `scripts/w134_ardy_access_preflight.py`
  - `tests/test_w134_ardy_access_preflight.py`
  - this Worklog
- **Unchanged:**
  - `docs/next_batch.md` and every historical Worklog;
  - all `src/` and every pre-existing test;
  - the W129/W131/W132/W133 adapters, harnesses, runners and declarations;
  - the CLI, GUI, exporters, FK/IK, FramePose and MotionGraph;
  - the canonical and normative documents; the rig, models and checkpoints; global dependencies;
  - `scripts/batch_automation.py` and its tests;
  - every W126/W129/W131/W132/W133 run file.

No living status document was updated: a blocked preflight is not a milestone.

## 12. Resource and environment constraints

| Resource | Cost |
|---|---|
| Disk (scratch) | about 77 MB: the code-only clone, plus about 144 KB of evidence |
| Wall time | the scripted preflight takes about 4 s (HTTP plus hashing of the frozen files) |
| GPU | not used |

**Estimated cost had access existed** (not measured):

- about 769 MB of ARDY weights plus about 16 GB of Llama 3 8B base weights and the LLM2Vec adapters;
- a new scratch PyTorch environment with sm_120 support, separate from the AnimCV 3.11 venv;
- the README puts the bf16 LLM2Vec encoder at about 14 GB of VRAM. On the 16 GB RTX 5080 that likely means encoding the
  single fixed prompt first and then releasing the encoder (or using the official `cpu` encoder option) before loading
  the 157 M denoiser. That plan would have to be frozen in the declaration before any S1 inference.
