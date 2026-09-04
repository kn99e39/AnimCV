# Worklog — Repository-Contract Hardening (2026-09-04)

> Hardening pass on `arch/single_frame_first`, following an independent
> inspection of the actual repository. The Frame-First architecture is accepted
> and unchanged. No model trained, no measured F0/F1/F2 result altered, and
> `.vscode/` untouched.
>
> The invariant being enforced: **executable experiment semantics == documented
> architecture semantics.**

## 1. Branch state entering this batch

`arch/single_frame_first` at `ed72402` ("docs/25: final architecture-contract
closure"), clean working tree, in sync with `origin`.

## 2. Documentation wording corrections

### 2.1 Temporal-lifter preservation

The claim that `src/training/temporal_lifter.py` "stays exactly as executed" was
literally false after docs/25: the file *was* mechanically changed so canonical
pose mathematics moved to `common.canonical_pose`, which it now consumes and
re-exports. The refactor stands; the wording is corrected everywhere it appeared
(Architecture_v3 §1.3, docs/23 header, docs/24 §2 table, docs/25 §9) to:

**Preserved** — historical checkpoints and metrics; `TrainingConfig` behaviour
and defaults; the training loop; the public and private compatibility names
historical scripts import; the loss mathematics; the evaluator mathematics;
A9–A16 numerical semantics (pinned bitwise).

**Changed** — implementation *ownership* of the shared canonical pose
mathematics.

### 2.2 Measured-lineage scope

Architecture_v3 said "every result in this repository is
`benchmark_detector_observation`". That over-reaches: the repository holds
historical experiments from other sources and pipelines that predate this
taxonomy. Scoped to the evidence: **the executed FramePose F0/F1/F2
measurements** use 3DPW dataset-shipped detector observations and are
`benchmark_detector_observation`. No unrelated historical experiment is
relabelled.

## 3. Geometry Observation Layer vs MMPose backend

Previous wording equated the abstraction with the library. Corrected:

```
Geometry Observation Layer                     <- the abstraction
    |
    +-- Oracle Geometry provider               annotated/projected GT, synthetic projection
    |
    +-- Benchmark Detector provider            a benchmark's own distributed detector output
    |
    +-- MMPose + RTMDet                        = the current Real AnimCV backend
```

MMPose remains strictly a **2D observation backend**; `VideoPose3DLifter` /
`lift-pose3d` remains legacy and reference only. A future detector or whole-body
estimator would be **another backend of the Geometry Observation Layer**, not
something "under MMPose ownership" — the phrasing used in the docs/24 and
docs/25 dependency tables, now corrected in both.

## 4. Observation image-content identity

`observation_cache_key(provenance, image_relative_path)` hashed a *path* while
its docstring claimed it bound "the exact input frame". A file can be replaced
in place, keeping its name and even its mtime, and an observation cached from the
old pixels would then be reused for the new ones.

```
image_content_digest(path)        = SHA-256 over the exact image bytes
observation_cache_key(prov, d)    = hash(provenance.cache_key() + d)
```

`d` must be a 64-character content digest; a path is **rejected**, not hashed, so
the old mistake cannot be made silently again. A cached observation is now
invalidated by a change of model, weights, config, preprocessing **or image
content**.

No Real AnimCV observation bank exists yet, so this contract was fixed without
touching any measured MMPose result.

## 5. Visual-input identity

New `src/framepose/visual_input.py`. A frozen visual feature is a pure function
of four things, and the fingerprint binds all four:

```
visual_input_fingerprint = hash(
    image content summary        SHA-256 per referenced image, in bank order
  + bank content digest          the geometry that built each crop
  + sample order digest
  + crop contract digest         crop_contract_digest() over the whole contract
  + crop resolution
  + backbone preprocessing       mean, std, input resolution, prefix tokens dropped
)
```

No host path participates: the same bank read through a different image root
fingerprints identically (tested). Distinct image files are hashed once even when
several actors share a frame.

`scripts/fingerprint_visual_input.py` produces this identity in one pass over the
imagery, with no GPU and no model inference.

## 6. Digest responsibilities kept separate

Three identities, three jobs, none overloaded:

| Identity | Covers | Answers |
| --- | --- | --- |
| `content_digest` | sample identity, split, source, numeric arrays | "same frames and same numbers?" |
| `provenance_fingerprint` | observation provenance, modality, image **references** | "did recorded provenance change?" |
| `visual_input_fingerprint` | image **content**, bank geometry, crop contract, preprocessing | "same visual input the features came from?" |

Raw image bytes are deliberately **not** hashed into `content_digest`: doing so
to solve visual caching would make every image touch invalidate the numeric bank
and every cache keyed to it.

## 7. Feature-cache provenance contract

Cache schema is now `animcv_frame_pose_feature_cache_v2`, recording:

```
feature_cache_provenance = hash(
    visual_input_fingerprint
  + backbone identity        key, timm model, token grid, width, resolution
  + recorded weights_sha256
)
```

plus `crop_contract_digest`, `token_shape`, `dtype`, `provenance_level` and
`weight_verification`.

Loading always checks schema, bank content digest, sample order, array shape and
dtype. For a v2 cache it additionally checks the crop-contract digest **in force
now**, the backbone key, and that a weight digest was recorded — and, when a
visual-input identity is supplied, that the images, geometry, crop contract and
preprocessing match. A mismatch raises.

## 8. Backbone weight guarantee, stated precisely

The contract is **immutable cache provenance**:

- the cache is the artifact, and its metadata records exactly which
  backbone/checkpoint/weight digest produced it;
- training consumes those features and records that provenance;
- **loading does not re-download or re-hash a current backbone** to prove it is
  still identical.

`WEIGHT_VERIFICATION` says this in the metadata itself, so no downstream report
can overstate the check. Loading without a supplied visual-input identity
returns `visual_input_verified: false` rather than implying verification.

## 9. Crop-contract invalidation

`crop_contract_digest()` hashes the whole `CROP_CONTRACT`, and that digest is
bound into both the visual-input fingerprint and the cache metadata. A cache
built under one crop contract is refused under another:

```
same bank, same images, changed crop-contract identity -> ValueError
    "feature cache was built under a different crop contract"
```

No crop parameter was tuned.

## 10. Historical feature-cache compatibility

The v1 caches that produced the F0/F1/F2 lineage recorded no image-content
digest and no crop-contract digest. Policy, verified against the real 6.57 GB
artifacts on the training host:

- **Refused by default.** `load_feature_cache` raises, naming the schema and the
  missing guarantees.
- **Readable through an explicit path.** `allow_legacy=True` (and
  `--allow-legacy-feature-cache` on the runner, the screening script and the
  visual-usage diagnostic) returns the cache labelled
  `provenance_level="historical_v1"`, `visual_input_verified=False`, with three
  named `not_established` guarantees.
- **Never upgraded, never rebuilt.** The ~13 GB of historical features were not
  regenerated to exercise schema mechanics; the mechanics are covered by unit
  tests on synthetic caches.

Verified live:

```
vit_in21k  refused without legacy flag; legacy path (21817, 196, 768) level historical_v1, verified False
siglip     refused without legacy flag; legacy path (21817, 196, 768) level historical_v1, verified False
```

## 11. Dependency-isolation test correction

The assertion

```python
assert "timm" not in pulled and "torch" not in sys.modules or True
```

could never fail — `or True` made it tautological, and `sys.modules` was the
*test process*, not the subprocess being observed. The intended contract was
both halves, so both are now proven: `torch` was added to the subprocess-observed
module set, and the registry test asserts separately that importing
`framepose.backbones` pulls neither `timm` nor `torch`. The
model/loss/evaluate test was tightened the same way, from three named exclusions
to "imports no backend at all".

## 12. Runner causal wording

`scripts/run_frame_pose_experiments.py` said "The only variable is the
observation backend". That is wrong for F0, which has no image projection and no
cross-attention sublayer. The module docstring now states what each comparison
can establish, and `experiment_matrix.json` carries it as machine-readable
metadata:

```
comparison_semantics.F1_vs_F0 / F2_vs_F0   tested geometry-only architecture vs tested
                                           visual-fusion architecture; not an
                                           information-only control
comparison_semantics.F2_vs_F1              architecture-matched; the variable is the
                                           frozen tower's pretraining
comparison_semantics.capacity_matched                              false
comparison_semantics.parameter_matched_geometry_control_trained    false
```

The parameter-matched control was **not** implemented, per the batch scope; no
code or document claims the existing comparison is capacity-matched.

## 13. F2 terminology

`F2_vlm_geometry` overstated the path — F2 loads the SigLIP **image tower** only.
New runs record:

```
F0  F0_geometry_only
F1  F1_imagenet_vision_geometry
F2  F2_vlpretrained_vision_geometry
```

The executed lineage's identifiers are retained as `historical_name` in
`CANDIDATES` and reproducible exactly with `--historical-candidate-names`.
Historical output directories, checkpoints and reports are **not** renamed.

## 14. Canonical-pose dependency verification

Unchanged and re-verified:

```
           common.canonical_pose
               /              \
      Frame Pose Core   Legacy Temporal Lifter
```

No `src/framepose` module imports `training.temporal_lifter` (enforced by test),
and all seven bitwise/strict-tolerance parity tests still pass.

## 15. Third-party state

Unchanged and not reopened: MMPose as a pip/mim runtime dependency and the
current Real AnimCV observation backend; `third_party/mmpose` absent; Depth
Anything V2 a LEGACY/OPTIONAL feature with a valid declared submodule; timm an
optional FramePose research dependency; `transformers` not present.

## 16. `.vscode/`

Not modified, not reverted, not inspected as an architecture issue.
`git status --porcelain .vscode/` and `git diff HEAD -- .vscode/` are both empty
for this batch. It remains exactly as the repository owner has it.

## 17. Files changed

```
A  docs/26_WORKLOG_REPOSITORY_CONTRACT_HARDENING.md
A  scripts/fingerprint_visual_input.py
A  src/framepose/visual_input.py
A  tests/test_frame_pose_visual_identity.py
M  Architecture_v3_FramePose.md      layer-vs-backend, scoped lineage, three identities,
                                     cache provenance, preservation wording, F2 naming
M  CLAUDE.md                         layer-vs-backend, cache identity summary
M  docs/23, docs/24, docs/25         preservation wording, "under MMPose ownership" rows
M  src/framepose/observations.py     image_content_digest; path-rejecting cache key
M  src/framepose/crops.py            CROP_CONTRACT_VERSION, crop_contract_digest()
M  src/framepose/features.py         cache schema v2, provenance, legacy path, dtype check
M  src/framepose/contract.py         provenance_fingerprint scope note
M  scripts/run_frame_pose_experiments.py   comparison semantics, candidate naming, cache flags
M  scripts/screen_frame_losses.py          legacy-cache flag
M  scripts/diagnose_visual_feature_usage.py  identity + legacy-cache flags, provenance recorded
M  tests/test_frame_pose_dependency_isolation.py  tautology removed, torch observed
M  tests/test_frame_pose_observations.py         image-content invalidation
M  tests/test_frame_pose_scripts.py              v2 cache fixtures
```

## 18. Tests and where they ran

| Suite | Environment | Result |
| --- | --- | --- |
| Full regression | macOS authoring venv (`.venv`, torch 2.13.0) | **570 passed, 1 skipped** (skip = timm-gated backbone test) |
| Non-GUI regression | `animcv-framepose:cuda118` on LabServer63 (torch 2.1.2+cu118, timm 1.0.29) | **522 passed** (GUI suites excluded — that image has no tkinter) |
| Historical cache compatibility, real 6.57 GB artifacts | same container | refused by default; legacy path labelled `historical_v1`, 3 `not_established` items |
| Visual-input fingerprint over the real 4.6 GB image set | same container | produced for both backbones — see below |

### 18.1 Visual-input fingerprint on the real research bank

`scripts/fingerprint_visual_input.py` over the 21,817-frame bank, one pass, no
GPU:

| Backbone | image content summary | crop digest | bank digest | preprocessing | fingerprint |
| --- | --- | --- | --- | --- | --- |
| `vit_in21k` | `7a9ee0f16e68cf39…` | `113d1daf37ac…` | `75519e6394a7…` | mean 0.5, 1 prefix token dropped | `e8286d15977ea420…` |
| `siglip` | `7a9ee0f16e68cf39…` | `113d1daf37ac…` | `75519e6394a7…` | mean 0.5, 0 prefix tokens | `fd3699ba60b41ea1…` |

The image-content summary, crop digest and bank digest are identical — same
frames, same images, same crop — while the fingerprints differ, because the two
towers declare different preprocessing (one drops a CLS prefix token, the other
has none). That is the contract behaving correctly on real data: a cache built
for one tower's preprocessing cannot be presented as valid for the other's.

These identities are written to `visual_input_{backbone}.json` beside the bank,
ready for the first v2 cache to be verified against.

New focused coverage — `tests/test_frame_pose_visual_identity.py` (14):
observation cache key binds bytes not paths; a path is rejected; fingerprint
determinism and path-independence; image-content change moves it; crop-contract
digest and crop resolution change move it; preprocessing change moves it; bank
geometry change moves it; shared images hashed once and order-sensitivity; v2
cache loads only against its own visual input; refused under a different crop
contract; requires a recorded backbone and weight digest; loading without an
identity does not claim verification; historical v1 needs the explicit path and
is labelled; a cache for another bank is still refused.

No remote CI exists for this repository and none was claimed.

No pose-quality training was run in this batch.
