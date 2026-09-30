# Worklog — FramePose AnimationSemantics producer correction (2026-09-30)

## Scope and repository state

Started from clean `arch/single_frame_first` at `dadc5de`, after fetching origin and confirming no remote commits ahead. Code commit: `c521ddb`. This batch corrects Root Orientation and foot-motion time interpretation for FramePose output, then stops before IK or retargeting. Regime throughout the measured H0 validation result: `benchmark_detector_observation`; oracle geometry and the KINEMATIC_PROXY are evaluation only.

Historical `src/pose/root_motion.py`, `src/pose/contact.py`, docs/51-55, frozen H0, FrameBank, Frame Pose Core, SignState, Pose Reconciliation, and their artifacts were not rewritten. The v1 schema, loader, bridge and existing v1 files retain their historical held-yaw and row-based-contact meanings.

## Versioned producer decision

New `animcv_animation_semantics_v2` sidecars use `framepose_current_bilateral_v1` Root Orientation and `framepose_elapsed_time_contact_v1` foot motion. `CurrentRootOrientation` carries only KNOWN/UNKNOWN and yaw; it does not reuse v1's `historical_confidence` or `yaw_held`. The v1 loader rejects v2, and the v2 loader rejects v1. Provenance names both exact policies, frozen H0 and bank identities, regime, and TRAIN calibration. Root translation and ground height remain `UNAVAILABLE`; UNKNOWN yaw is not zero. MotionGraph was not changed.

The yaw calculation is the current-frame, circular shoulder/hip fused observation from docs/55. There is no hold, release, smoothing, persistence, or oracle runtime input. Production also refuses a frame if neither bilateral pair is structurally valid. The frozen validation H0 has zero such frames, so this gate does not alter the docs/55 control signal. The diagnostic has 50 rows whose contributing pairs are not *all* valid, but every row has at least one valid bilateral pair; the exact docs/55 fusion is preserved on those rows.

The contact policy computes central and one-sided ankle velocities using elapsed timestamps, or frame-index delta divided by source fps when timestamps are absent. A missing/invalid time axis is refused. The height window and reliable-run support are expressed in seconds. The old rule's radius of 2 TRAIN rows and minimum of 2 TRAIN rows, at TRAIN median stride 2 source frames and 30 fps, give a height radius of `2×2/30 = 0.133333 s` and a minimum **first-to-last sample span** of `(2−1)×2/30 = 0.066667 s` (with at least two observations). Validation median stride is 3 frames; no validation parameter was fit. Percentiles remain 15/60/40.

TRAIN-only H0 fit: 34 sequences, 11,334 rows. New speed cutoffs, left contact/moving `0.09819/0.36064 m/s`, right `0.10200/0.36003 m/s`; height standard-deviation cutoffs left `0.005877 m`, right `0.006534 m`. These differ from historical T1 because the velocity denominator and temporal height windows now use seconds. The rule still means contact-like stance/swing evidence relative to the pelvis, not proven foot-ground contact.

## Controlled replay

LabServer63 `animcv-framepose:cuda118`, same validation 16 sequences and 3,407 bank rows. Bank content digest `75519e6394a764e3749ddaa30555b58b73a01db582ccee14f661374b9a0ed536`; H0 TRAIN SHA-256 `43a5d53bee99eed6159b2a83107235632add91912cb985a9eae0900db65c6af6`; H0 validation SHA-256 `88a007fca42eee08833ec8ece453aa3160a26232738be01cd849cf5016524937`. These match docs/52-55. The replay checks docs/55 yaw numbers before writing v2 sidecars. All 16 v2 sidecars replay deterministically and round-trip.

**Root Orientation:** KNOWN 3,407, UNKNOWN 0. Oracle bilateral-yaw error mean `6.3655°`, median `4.6307°`, p90 `13.1996°`, p95 `16.8218°`; above 45° `22/3,407 = 0.646%`, above 90° `1/3,407 = 0.029%`. These reproduce docs/55 H0-CURRENT numerically. Historical held yaw had mean `23.5763°`, p95 `133.8493°`, and `297/3,407` above 90° on the same rows.

**Contact:** Both policies were scored against the *same* docs/52 world-frame KINEMATIC_PROXY. Historical T1 used its persisted v1 sidecars; time-aware used the new TRAIN calibration. UNKNOWN rows are excluded from confusion metrics, so the scored denominator differs by policy and must be read alongside state counts.

| Side, policy | CONTACT / MOVING / UNKNOWN | Precision | Recall | Specificity | Balanced accuracy | CONTACT run span median / p90 |
|---|---:|---:|---:|---:|---:|---:|
| Left historical T1 | 164 / 2113 / 1130 | 0.140 | 0.605 | 0.937 | 0.771 | 0.20 / 0.50 s |
| Left time-aware | 341 / 1625 / 1441 | 0.170 | 0.951 | 0.851 | 0.901 | 0.20 / 0.70 s |
| Right historical T1 | 167 / 2017 / 1223 | 0.222 | 0.740 | 0.939 | 0.840 | 0.20 / 0.40 s |
| Right time-aware | 343 / 1602 / 1462 | 0.198 | 0.907 | 0.853 | 0.880 | 0.20 / 0.79 s |

CONTACT run span is last timestamp minus first timestamp (not an inclusive duration). Run counts: historical/time-aware left `47/76`, right `49/82`; maximum spans left `1.3/1.4 s`, right `1.2/1.4 s`. State disagreements: left `665/3,407 = 19.5%` (`177 UNKNOWN→CONTACT`, `488 MOVING→UNKNOWN`); right `591/3,407 = 17.3%` (`176 UNKNOWN→CONTACT`, `415 MOVING→UNKNOWN`). There was no MOVING→CONTACT switch directly at the same row.

For a selector independent of either candidate, GT world-ankle velocity was recomputed on the actual elapsed time. The low-motion subset is velocity at most half the fixed docs/52 TRAIN reference threshold, `0.084658 m/s`. It contains 294 left and 308 right rows. MOVING drops from `47→17` left and `31→17` right; CONTACT rises `49→111` and `64→117`. These counts answer the stationary-case question, while the substantial UNKNOWN counts (`166` left, `174` right) and low precision keep the contact-like interpretation necessary. The unchanged docs/52 KINEMATIC_PROXY itself uses sampled-row differences, so its confusion scores are a fixed historical comparison, not a physical-contact accuracy claim. The independent low-motion selector uses true elapsed time.

## Owner review

Six RGB, input-skeleton, projected-H0, top/front H0, heading, foot-state and reliability clips/stills were rendered with the extended docs/54 review tool. H0 projection on RGB uses GT root only for display. At turning and dynamic crosscountry frames its GT anchoring cannot project H0 onto RGB; the original RGB/2D and H0 top/front views remain visible. Case selection is qualitative, not a population estimate.

- Known-good standing, `courtyard_dancing_00:actor0#424`: current yaw `−163.3°`, both feet CONTACT-like, 17/17 valid joints. No held heading.
- Suspected false MOVING, `courtyard_drinking_00:actor1#6`: visually standing; historical right MOVING becomes time-aware UNKNOWN. Current yaw `−164.4°`, 17/17 valid.
- Dynamic walking toward camera, `outdoors_crosscountry_00:actor0#462`: current yaw `−170.3°`, left MOVING and right UNKNOWN, 17/17 valid.
- Ankle failure, `outdoors_parcours_00:actor0#384`: visible H0 ankle mismatch; both feet MOVING, 15/17 valid. This remains a geometry limitation.
- Tracking loss, `courtyard_hug_00:actor1#312`: seated actor, 15/17 valid; both feet CONTACT-like on the lower-body evidence. Overall reliability is separately visible.
- Turning, `outdoors_crosscountry_00:actor0#202`: current yaw `129.1°`, left MOVING/right UNKNOWN, 12/17 valid; the historical held yaw at this row was `−6.5°`.

## Artifacts, tests and conclusion

Server artifact: `~/animcv-output/animation_semantics_v2_replay/` (`report.json` SHA-256 `af7de53f2a0868f35527ca967b7e7c36805a4a83a1481ac54688221de3cb6c8b`, `contact_calibration.json`, `contact_rows.jsonl`, 16 v2 sidecars, `review/`). No historical artifact was overwritten.

Changed code: `src/pose/contact_time_aware.py`, `src/motion/animation_semantics_v2.py`, `src/motion/animation_semantics_v2_bridge.py`, `scripts/run_animation_semantics_v2_replay.py`, `scripts/render_animation_semantics_review.py`, `tests/test_animation_semantics_v2.py`. Focused tests: `37 passed` for v1/v2 AnimationSemantics and contact. Full local regression: `883 passed, 1 skipped` (before the final v2-only structural gate); focused v2 after that gate: `4 passed`. The shared v1 serialization code did not change; the full run was nevertheless performed because the review tool now reads both versions.

**Verdict:** The current FramePose AnimationSemantics producer now expresses available root heading from current evidence and contact-like foot motion on elapsed time. It no longer inherits the rejected yaw latch or sampled-row timing error. It is suitable as a typed, provenance-bearing input to a later retargeting design, with UNKNOWN/UNAVAILABLE respected. This does not solve world root motion, true physical ground contact, occlusion completion, foot locking, IK, or retargeting.
