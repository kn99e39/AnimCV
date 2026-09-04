# Worklog — Observation Image-Binding Closure (2026-09-04)

> Final invariant-closure pass on `arch/single_frame_first`, following an
> independent inspection of the branch. The hardening architecture and cache-v2
> design of docs/26 are accepted and unchanged. No model trained, no measured
> result altered, no feature cache rebuilt, `.vscode/` untouched.

## 1. Branch HEAD before this correction

`12118b5` ("docs/26: repository-contract hardening report"), clean tree, in sync
with `origin`.

## 2. Real AnimCV observation image-binding invariant

docs/26 bound `observation_cache_key` to image *bytes* and rejected paths, but
still accepted `image_digest=None`. For the Real AnimCV backend that was a hole:
its keypoints are produced by reading an RGB frame, so an identity that omits
which frame is not an identity, and a cache built from one set of pixels could
have been reused for another.

Closed. `framepose.observations` now declares:

```python
IMAGE_GENERATED_BACKENDS = (BACKEND_MMPOSE,)
```

and `observation_cache_key` refuses to build a key for any of those backends
without a digest:

```
mmpose observations are produced by reading an RGB frame, so their cache
identity must bind the exact image bytes
```

A Real AnimCV cache identity can therefore no longer be produced from `None`, a
path, a filename or an mtime — the last three were already rejected by the
64-hex-character content-digest check, which now also rejects wrong-length and
upper-case inputs by construction.

## 3. Which regimes may omit an image digest, and why

Optionality is a documented semantic keyed to *how the observation was
produced*, not to its regime label:

| Backend | Produced by reading RGB? | Image digest | Identity source |
| --- | :-: | :-: | --- |
| `mmpose` | **yes** | **mandatory** | sensor provenance + exact image bytes |
| `dataset_ground_truth` | no | optional | the dataset's own 3D annotation, projected |
| `synthetic_projection` | no | optional | mocap projected through a virtual camera |
| `dataset_detector` | no (by AnimCV) | optional | the distributed keypoint artifact itself, already covered by the bank's `content_digest` |

The `dataset_detector` case is worth stating plainly: 3DPW's keypoints *were*
produced from RGB, by its authors. AnimCV never re-runs that detector; it
consumes the released artifact. Its identity source is therefore the annotation
artifact, and it is not represented as a Real AnimCV sensor cache.

## 4. Focused tests for `observation_cache_key`

`tests/test_frame_pose_visual_identity.py` now proves all four required cases
plus the deliberate exception:

| Case | Expected | Test |
| --- | --- | --- |
| MMPose + valid image digest | valid 64-char key | `test_real_animcv_observation_cannot_be_cached_without_the_image` |
| MMPose, same path, changed bytes | different key | `test_observation_cache_key_binds_image_bytes_not_the_path` |
| MMPose + `None` | **refused** | `test_real_animcv_observation_cannot_be_cached_without_the_image` |
| MMPose + path string | **refused** | `test_observation_cache_key_refuses_a_path` (also 63/65-char and upper-case) |
| non-image-generated + `None` | valid, and documented | `test_non_image_generated_observations_may_omit_the_image_digest` |
| only MMPose is image-generated | asserted | `test_only_the_real_animcv_backend_is_image_generated` |

The previous assertion that `observation_cache_key(mmpose_provenance, None)` is
valid has been reversed.

## 5. `observations.py` ownership wording

`"AnimCV's own Geometry Observation Layer — MMPose"` appeared twice (module
docstring and `mmpose_observation`). Both now read *"AnimCV's current Real
Observation backend of the Geometry Observation Layer — MMPose + RTMDet"*, with
an explicit note that the layer is the abstraction and MMPose is one backend of
it.

## 6. `cache_key` docstring / `content_digest` correction

The docstring claimed an image change is covered *"bank-wide by
`FrameBank.content_digest`"*. That is false under the accepted design — raw
image bytes are deliberately not in that digest, so it cannot detect an in-place
JPEG replacement. Replaced with the four explicit responsibilities:

```
FrameBank.content_digest            numeric bank identity; does NOT cover image bytes
FrameBank.provenance_fingerprint    recorded observation/modality provenance, image references
observation_cache_key               sensor identity + exact image bytes, where RGB is consumed
visual_input.visual_input_fingerprint   exact visual-input identity for feature caches
```

## 7. Dependency-isolation docstring

Behaviour untouched. The module docstring's *"MMPose ... is the Geometry
Observation Layer"* became *"MMPose + RTMDet ... is the current Real AnimCV
backend of the Geometry Observation Layer (the layer is the abstraction, not the
library)"*, and the invariant under test is now stated explicitly: **framepose
imports no OpenMMLab runtime at import time.**

## 8. `crops.py` causal wording

`geometry_in_crop`'s docstring concluded that holding geometry identical makes
the manipulated variable "only the presence and kind of visual evidence". The
premise is true; the conclusion is not. It now states that the geometry input is
held identical across candidates and that this does **not** make F0 vs F1/F2 an
information-only or capacity-matched comparison, because F0 also lacks the image
projection and cross-attention sublayer. Crop mathematics untouched.

## 9. `features.py` weight-guarantee wording

The module description said the cache *"refuses to be used with [the backbone's
weight digest] changed"*, implying the loader detects a changed current
backbone. It does not, and the implementation was not changed to do so. The
guarantee is now stated as implemented — **immutable cache provenance**: the
cache records the exact generating weight digest, visual-input identity and crop
contract; loading refuses a recorded identity that does not match the bank and
visual input it is paired with, and never re-downloads or re-hashes a current
tower.

## 10. Repo-wide stale-semantics audit

Bounded search over `src/`, `scripts/`, `tests/`, `Architecture_v3_FramePose.md`
and `CLAUDE.md`:

| Pattern | Result |
| --- | --- |
| "MMPose is the Geometry Observation Layer" / "MMPose owns exactly one box" | **0 occurrences** |
| "under MMPose ownership" | only in docs/26, as a record of what was corrected — marked historical, left alone |
| "only variable is" / "manipulated variable is only" | only in docs/26, as the quoted text it replaced |
| "information-only control" | 8 occurrences, **every one a negation** |
| current-backbone weight verification promised | **0**; every mention is "recorded" or an explicit "does not re-hash" |
| image bytes covered by `content_digest` | **0**; the only co-occurrence is the corrected explanation stating the opposite |

Historical worklogs were not rewritten.

## 11. Visual-cache v2 behaviour

Unchanged, by design. v2 + supplied fingerprint verifies image/crop/preprocessing;
v2 without one loads with `visual_input_verified=False`; historical v1 is refused
unless `allow_legacy=True` and is then labelled `historical_v1` with its
`not_established` guarantees. No feature cache was rebuilt.

## 12. `.vscode/`

Not modified, not reverted, not inspected as an architecture issue.

## 13. Files changed

```
A  docs/27_WORKLOG_OBSERVATION_IMAGE_BINDING_CLOSURE.md
M  Architecture_v3_FramePose.md                    mandatory image binding for image-generated observations
M  CLAUDE.md                                       same, in the operator summary
M  src/framepose/observations.py                   IMAGE_GENERATED_BACKENDS + enforcement; layer/backend
                                                   wording; four-identity cache_key docstring
M  src/framepose/crops.py                          causal wording in geometry_in_crop
M  src/framepose/features.py                       immutable-cache-provenance wording
M  tests/test_frame_pose_visual_identity.py        reversed and extended cache-key contract
M  tests/test_frame_pose_dependency_isolation.py   docstring only; behaviour untouched
```

## 14. Tests and where they ran

| Suite | Environment | Result |
| --- | --- | --- |
| Full regression | macOS authoring venv (`.venv`, torch 2.13.0) | **573 passed, 1 skipped** (skip = timm-gated backbone test) |
| Non-GUI regression | `animcv-framepose:cuda118` on LabServer63 (torch 2.1.2+cu118, timm 1.0.29) | **525 passed** (GUI suites excluded — that image has no tkinter) |

Focused suites inside those totals: `test_frame_pose_visual_identity.py` **17
passed** (up from 14 — the three new image-binding contracts),
`test_frame_pose_dependency_isolation.py` **8 passed**,
`test_canonical_pose_parity.py` **7 passed**,
`test_frame_pose_observations.py` **23 passed**.

No remote CI exists for this repository and none was claimed. No training run,
no feature-cache rebuild.
