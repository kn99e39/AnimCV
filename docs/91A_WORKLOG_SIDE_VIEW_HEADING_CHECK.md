# Worklog — Side-View Heading Check from the 2D Nose (2026-10-19)

**What this is.** A minor worklog attached to Worklog 91, tackling its blocker 6: side-on views out of domain can mirror
the GT-free heading, as in 3DHP TS2. Branch `arch/single_frame_first`, starting HEAD `3ad8bd6`.

**How it was run.** Locally only, at low priority, because LabServer63 is about to be used by another project. The
3DHP bank and the frozen W91 S0 3DHP predictions (`a78adeb0…`) were copied once beforehand; nothing ran on the server.

**What was not changed.** No model, checkpoint or production default; the W91 provider is unchanged.

## Result: rejected (no useful improvement)

The bounded side-view check does not help.

| population | what the check did | effect |
|---|---|---|
| 3DPW (VAL and TEST) | never flips | none: S0 is already right in side views |
| 3DHP | flips 22 rows; **none of them improves** | B gets worse (>90° 1.6% → 2.4%, front/back 1.1% → 1.7%) |
| 3DHP TS2 failure run | not fixed | the 2D nose agrees with S0's **wrong** profile |

The module is kept as documented, **unwired** experimental code and is not used by any provider.

## Candidate (declared before evaluation)

**Code:** `src/motion/runtime_heading_side_check.py`.

**Observation (runtime only).** The detector's 2D nose (FrameBank `head` = OpenPose nose, in both the 3DPW and 3DHP
paths), shoulders and hips. The offset is the nose's horizontal distance from the shoulder midpoint, divided by the
torso length (shoulder midpoint to hip midpoint), in image pixels.

**Geometry.**
- Body forward is f = (−sin ψ, cos ψ) in AnimCV camera axes.
- In a side view the nose lies ahead of the shoulders along f.
- So the sign of the offset should match the sign of −sin ψ.

**Rule (one mechanism).** It applies only when all three hold:
- the heading is side-ish (|sin ψ| ≥ sin 45°);
- the nose is valid with confidence ≥ 0.3;
- |offset| ≥ 0.10 torso lengths.

If the offset's sign disagrees with −sin ψ, the profile is mirrored (ψ → −ψ, i.e. the depth half-plane is negated).
Otherwise the yaw is untouched. The rule is per-frame, stateless and causal.

**Thresholds.** Fixed from geometry. On **3DPW VALIDATION only**, the sign convention was confirmed first: on 1,154
GT-side rows with a usable nose, the cue agreed with the GT profile 100% of the time and S0 agreed 99.9%. Then 3DPW TEST
and 3DHP were evaluated, with no adjustment.

## Evaluation

**Script:** `scripts/run_runtime_heading_side_check_eval.py`. Identical frames and observations; the W91 `evaluate`
metrics; GT for evaluation only. The B and C rows reproduce Worklog 91 exactly.

| population | B: mean / >90° / front-back / longest wrong run | B + check | C + check | check actions |
|---|---|---|---|---|
| 3DPW VAL | 7.59 / 0 / 0 / 0.8 s | identical | identical to C | 1,152 agree, 0 flips |
| 3DPW TEST | 9.99 / 0.14% / 0.07% / 4.0 s | identical | identical to C | 2,271 agree, 0 flips |
| 3DHP all | 17.20 / 1.61% / 1.14% / 18.8 s | 18.24 / 2.38% / 1.72% / 18.8 s | 16.82 / 1.62% / 1.16% / 4.0 s (C: 15.91 / 1.00% / 0.66%) | 1,093 agree, 22 flips, **0 improved** |
| 3DHP TS2 | 28.05 / 7.9% / 6.1% / 18.8 s | 28.67 / 8.6% / 6.5% / 18.8 s | equal to C | 153 agree, 3 flips, 0 improved |
| 3DHP outdoor | 16.51 / 0.16% / 0 | identical | identical | 0 flips |

## Why it fails: what the measurements establish

**Per-sequence agreement on GT-side 3DHP rows with a usable nose:**

| sequence | cue = GT | S0 = GT | cue = S0 |
|---|---|---|---|
| TS1, TS5, TS6 | 1.000 | 1.000 | 1.000 |
| TS3 | 0.992 | 0.996 | — |
| TS4 | 0.943 | 0.996 | 0.940 |
| **TS2** | **0.767** | 0.798 | **0.957** |

- **In the TS2 failure, the 2D image evidence itself points the wrong way.**
  - Of the 30 TS2 rows that are more than 135° wrong, the check saw 22 with the nose cue *agreeing* with S0's mirrored
    profile and contradicting GT.
  - Of the rest, 4 have no valid nose and 4 fall just under the confidence or offset gate. Even those rows have a cue
    that contradicts GT.
  - The nose offset does not depend on left/right labels, so this is not a detector label swap that a 2D-geometry rule
    could undo.
  - Either the head faces opposite to the torso in those frames (e.g. looking back over the shoulder), or the image
    observation and the 3DHP 3D annotation disagree there. The two cannot be told apart without looking at the 3DHP RGB
    frames, which were not inspected, to keep the server untouched.
- **Where S0 was already right, the cue is the one that errs** (TS4: 5.7% of side rows, nose confidence often > 0.9).
  These are head-turn frames, and they produce the 22 harmful flips.
- **S0's lifted heading already uses the same evidence.** Where both are wrong, they are wrong together. So the 2D nose
  adds no independent information about the torso's profile direction.

**Answer to Worklog 91 blocker 6.** Not solved by a runtime 2D-nose rule. It needs either:
- evidence that observes the **torso's** facing directly (for example a visual torso-front cue, rather than the head),
  or
- a resolution of whether the TS2 rows are a head–torso dissociation or an observation/annotation inconsistency.

The W91 C reliability flag stays the only mitigation; it caps the TS2 wrong run at 4.0 s.

## Tests

`tests/test_framepose_runtime_heading_side_check.py`, 3 tests:
- offset normalization and validity
- the geometry convention, the mirror ψ → −ψ, and gating of near-front headings, weak or low-confidence noses and
  unknown headings
- the rejected check is wired into no provider or production module

The W91 production-import test now exempts the experimental `runtime_heading*` modules by name.

Locally, the focused W89–W91A suites plus the production-adjacent AnimationSemantics v2, FramePose bridge and Root
Orientation suites ran: **65 passed**. The server suite was not run, to avoid resource contention.

## Files and artifacts

**New files:**
- `src/motion/runtime_heading_side_check.py` (rejected, unwired)
- `scripts/run_runtime_heading_side_check_eval.py`
- `tests/test_framepose_runtime_heading_side_check.py`

**Changed:** `tests/test_framepose_runtime_heading.py` (exemption above).

**Report:** `~/animcv-local-data/w91/archive/side_check_report.json` (local only).
