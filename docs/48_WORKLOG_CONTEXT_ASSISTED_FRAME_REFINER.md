# Worklog — Context-Assisted Frame Refiner (2026-09-22)

## Scope and repository state

This session advanced AnimCV from the frozen single-frame H0 path to one
bounded context-assisted candidate. It did not reopen the Frame Pose Core,
Pose Reconciliation, SignState, canonical coordinates, MINIMUM_NORM/R_SWIVEL,
Qwen Sign Advisor, or target-rig work.

The session started on `arch/single_frame_first` at `f01ef1aa153c3769be0e8b6e1328b9dac808bc99`.
`git fetch origin` completed before work; local and remote branch heads were
identical. The final HEAD is the same because this session did not create a
commit or push. The local worktree contains the new uncommitted files listed
below; pre-existing worktree state was not changed.

New files:

- `src/framepose/context_refiner.py`
- `scripts/run_context_refiner.py`
- `scripts/prepare_context_h0.py`
- `scripts/export_context_refiner_review.py`
- `tests/test_context_refiner.py`
- this worklog

The experiment used the exact frozen FrameBank content digest
`75519e6394a764e3749ddaa30555b58b73a01db582ccee14f661374b9a0ed536` in the
`benchmark_detector_observation` regime. The frozen H0 test array regenerated
from the historical `O_BILATERAL` checkpoint matched the accepted test array
byte-for-byte: SHA-256
`6df04c2031b2bde6a4f0af37fdae0a7f0228e9fc23401e15e1aa2b3bd35695a5`.

## Candidate contract

`ContextAssistedFrameRefiner` is a separate module. It consumes the fixed
same-sequence retained window `t-2 ... t+2` and emits a residual for the
target frame only. Each slot contains frozen H0 XYZ, H0 validity, 2D XY and
confidence, observation validity, in-frame state, slot presence, relative
frame offset, and relative timestamp. Sequence boundaries are zero-filled and
masked; no cross-sequence row can enter a window.

The selected architecture is one small temporal MLP:

- flattened context dimension: 780;
- two 128-wide GELU hidden layers;
- output: 17 x 3 target-frame residual;
- trainable parameters: 123,059;
- estimated inference work: 245,504 multiply-adds per frame;
- measured C2 inference: about 0.00542 ms/frame on the LabServer63 GPU for
  the all-bank replay path.

The residual is added to H0 only where the gate is on. Gate-off output is the
exact frozen H0 value. C1 is the parameter-free gated interpolation control;
the historical diagnostic implementation and its accepted report were not
modified.

Training used only the existing `baseline_geometry_v1` target-frame objective
(coordinate, bone, torso, and hinge terms). It used 80 epochs, batch size 256,
learning rate `1e-3`, weight decay `1e-4`, seed 1337, and validation MPJPE for
checkpoint selection. There was no temporal smoothness, velocity,
acceleration, contact, trajectory, or orientation-tail loss. Test ground truth
was not used for training or checkpoint selection; selected validation MPJPE
was 68.1469 mm.

## Runtime gate

The gate is target-free at inference. Per joint, its thresholds were fitted
once from train rows only:

```text
local 2D interpolation residual <= train P50
local frozen-H0 interpolation residual >= train P90
current observation is valid and in-frame
both same-sequence H0/2D supports exist within t+/-2
```

The gate refuses intervention at sequence boundaries, when valid bilateral
support is unavailable, when the current observation is invalid/out of frame,
or when local 2D instability is high. No oracle 3D, target error, test label,
cohort membership, or future ground truth enters the gate.

On the 7,076-frame test split, the C2 gate accounting was:

```text
gate ON: 4,390 / 114,005 joint rows (3.85%)
improved when ON: 2,072
worsened when ON: 2,318
gate-on mean error delta (H0 - C2): -0.746 mm
gate-on median delta: -0.701 mm
gate-on p95 delta: +24.516 mm
damage-tail p95: 36.570 mm
```

Refusal/accounting reasons over all test joint rows were: local 2D strongly
unstable 85,388; H0 locally stable 21,037; insufficient valid support 1,839;
sequence boundary 1,240; current observation invalid/out of frame 111; gate on
4,390. Exact per-joint and per-sequence activation is in the machine report;
per-joint activation ranged from 186 to 344 rows (2.85%–4.92%).

## Evaluation

All numbers below are target-frame metrics in the benchmark-detector regime.
The delta column is positive when C2 is better than C0. C0 is frozen H0; C1
is the runtime-gated interpolation control; C2 is the learned refiner.

| Slice | C0 MPJPE | C1 MPJPE | C2 MPJPE | C2 gate ON | C2 delta vs H0 |
|---|---:|---:|---:|---:|---:|
| ALL TEST | 79.057 | 78.881 | 79.086 | 4,390 / 114,005 | -0.029 |
| STABLE CONTROL | 40.200 | 40.200 | 40.200 | 0 / 20,571 | 0.000 |
| STABLE-2D / H0-JITTER | 80.404 | 77.590 | 80.380 | 516 / 1,518 | +0.024 |
| HIGH 2D INSTABILITY | 83.996 | 83.996 | 83.996 | 0 / 10,878 | 0.000 |
| CURRENT OBSERVATION LOSS | 79.860 | 79.860 | 79.860 | 0 / 6,380 | 0.000 |
| DISTAL ANKLE | 340.467 | 337.808 | 340.233 | 60 / 1,303 | +0.234 |
| ARTICULATION-MISMATCH | 101.494 | 101.033 | 101.596 | 61 / 1,390 | -0.102 |

For ALL TEST, C0/C1/C2 PA-MPJPE was 56.393 / 56.543 / 56.454 mm. C2
gate-off rows were unchanged exactly. On the stable-control slice there were
zero false interventions and zero changed rows. High 2D instability and
current observation loss were also exact H0 no-ops, as required.

The intended stable-2D/H0-jitter slice had 247 improved versus 269 worsened
C2 rows while on; its gate-on mean delta was +0.071 mm and damage-tail P95
was 27.786 mm. The ankle evaluation slice had 35 improved versus 25 worsened
rows while on, with gate-on mean delta +5.081 mm and damage-tail P95 60.430
mm. The articulation-mismatch slice remained mixed: 31 improved versus 30
worsened, gate-on mean delta -2.334 mm, damage-tail P95 36.034 mm. These are
evaluation slices only; no ankle-specific runtime policy was introduced.

The whole-test comparison is the decisive result: the learned refiner did not
convert the narrow temporal signal into a net target-frame improvement. It
preserved stable and unreliable regimes safely, but the active gate's learned
correction was slightly negative overall and remained mixed on the intended
slice.

## Owner replay and qualitative review

The numeric owner-seed replay covered the six prior owner anchors. All six
were gate-off in this run, so C2 exactly equalled H0 at those selected joints;
the report records the runtime refusal reason for each. This is consistent
with the gate's refusal policy and is not evidence that those owner cases are
all temporal failures.

The compact RGB review contains ten events: the six owner anchors plus one
event each from stable-2D/H0-jitter, high-2D-instability, distal ankle, and
observation-loss. Each MP4 shows RGB with the observed joint, 2D context,
H0, C2 target-frame pose, oracle target where available, and gate state/reason.
The final manifest is at:

```text
LabServer63:/home/nd/animcv-output/framepose/context_review_v3/review_manifest.json
SHA-256: 2a4b88a1ba769226bfd3229f130969bcc3afcebfb376a7f1a8fad1e7e2d0bffe
```

The final machine report and artifacts are under:

```text
LabServer63:/home/nd/animcv-output/framepose/context_refiner_v3/
```

The final experiment report SHA-256 is
`c4995b5e13380613031dd3d82ef6229cb41c12fd4dcdddaabadaa9d82c870cf7`.
The checkpoint SHA-256 is
`039675c8c1bce26a2f8d77381294ca8d5913d75997de8d72174e256ddd118199`.
Replaying that checkpoint through the same all-bank inference path reproduced
the stored C2 test prediction bit-for-bit (`array_equal=True`, maximum
absolute difference 0.0).

## Verification

The focused local suite passed 9 tests with the torch-dependent training test
skipped because this macOS environment has no torch. The LabServer63 GPU
container ran the focused suite plus the relevant Frame Pose contracts:

```text
45 passed
```

This included the new context-window/gate tests, temporal diagnostic tests,
Frame Pose contract/evaluation/dependency-isolation tests, and canonical pose
parity tests. No shared production contract was modified, so a full repository
regression was not required by the repository instructions.

## Final verdict

**B — CONTEXT SIGNAL REAL, CURRENT REFINER INSUFFICIENT**

The prior temporal diagnostic remains valid: short context is useful in a
narrow stable-observation/H0-jitter regime and can help some ankle rows. This
single bounded learned refiner did not produce a reproducible whole-test
target-frame gain, and the intended active rows remained close to a mixed
win/loss split. It is therefore not adopted as the next default architecture.
The stopping condition is reached here; no general temporal stabilization,
root/contact, or retargeting work follows from this batch.
