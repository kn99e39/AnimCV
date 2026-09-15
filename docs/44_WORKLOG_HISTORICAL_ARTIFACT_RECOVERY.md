# Worklog — Historical Artifact Recovery for the docs/39 Replay Inputs (2026-09-15)

> Continuation of docs/40–43 on `arch/single_frame_first`. docs/41 and docs/42
> could not execute the real Pose Reconciliation replay because docs/39's
> data-mounted inputs — the FrameBank, the stored `O_BILATERAL` prediction, and
> the raw 3DPW camera pickles — were not present in this workspace. This batch
> locates those exact historical artifacts on `LabServer63`, verifies each one
> byte-for-byte against its recorded identity, and opens the gate for the real
> replay. **This file is the recovery/verification gate only — it contains no
> replay result.** Every artifact was read and copied, never modified, retrained,
> or recomputed.

## 1. Scope

```text
branch:            arch/single_frame_first
starting_head:     42337d8364ad61c994e2b694f0bd7969672be4f7
historical_store:  LabServer63:/home/nd/animcv-output/framepose
copy_workspace:    /private/tmp/animcv-pose-recovery.FyI66d
```

## 2. Gate result

```text
required_inputs_pass:  true
real_replay_authorized: true
```

Four notes carried with that result, verbatim:

- All computationally required artifacts were inventoried before copying.
- Historical files were only read/copied; none were modified.
- The recovered bank serialization is v2, while the contract text names v1.
  Its contemporaneous matrix/index/array identities, frame counts, regime, and
  path-independent content digest match exactly; v2 adds provenance metadata
  without changing the v1 content digest domain.
- The evaluation schema does not encode split identity; the ownership and
  docs/39 provenance artifacts independently cross-bind its exact SHA,
  candidate, frame count, bank digest, and observation regime.

## 3. Recovered artifacts

Nine artifacts passed; two were correctly excluded by design. Every SHA-256 is
reproduced here in full, since exactness is the entire point of a recovery
audit.

### 3.1 FrameBank JSON index — PASS

```text
source:  /home/nd/animcv-output/framepose/bank_3dpw_paired_v2.json
bytes:   41,419,487
mtime:   2026-09-03T05:29:43.664626288Z
sha256:  6fe13cdc91e1fdec8defd647b230cf688fa514a2799956ed40e7f805b13e0fe5
copied:  /private/tmp/animcv-pose-recovery.FyI66d/bank_3dpw_paired_v2.json
```

| | expected | measured |
|---|---|---|
| schema | — | `animcv_frame_pose_bank_v2` |
| content digest | `75519e6394a764e3749ddaa30555b58b73a01db582ccee14f661374b9a0ed536` | `75519e6394a764e3749ddaa30555b58b73a01db582ccee14f661374b9a0ed536` |
| samples | 21,817 | 21,817 |
| split counts | train 11,334 / validation 3,407 / test 7,076 | train 11,334 / validation 3,407 / test 7,076 |
| observation regime | `benchmark_detector_observation` | `benchmark_detector_observation` |
| provenance fingerprint | — | `ca343d04e52629abcb91e8c8281744abc69b1190c3ff3a822acc128fe45a1121` |

Identity method: original server artifact SHA-256, plus `load_bank`/
`content_digest` verification and exact split/regime checks; the
contemporaneous sign-experiment matrix independently binds the index and
paired NPZ hashes.

### 3.2 FrameBank numeric arrays — PASS

```text
source:  /home/nd/animcv-output/framepose/bank_3dpw_paired_v2.npz
bytes:   9,644,144
mtime:   2026-09-03T05:29:41.851716383Z
sha256:  909090f7aa78f3c25f56fbd335eb5b2521f87c06e92d59a28010b0b28ba6b79e
```

Expected identity: the exact NPZ SHA recorded by
`sign_attr_v2/sign_experiment_matrix.json` for the accepted bank index/content
digest. Measured: SHA-256 and the contemporaneous matrix reference match.
Identity method: original server hash plus exact matrix cross-reference.

### 3.3 `O_BILATERAL` prediction — PASS

```text
source:  /home/nd/animcv-output/framepose/sign_attr_v2/O_BILATERAL/prediction_test.npy
bytes:   1,443,632
mtime:   2026-09-06T07:25:08.570964515Z
sha256:  6df04c2031b2bde6a4f0af37fdae0a7f0228e9fc23401e15e1aa2b3bd35695a5
copied:  /private/tmp/animcv-pose-recovery.FyI66d/prediction_test.npy
```

Expected identity: candidate `O_BILATERAL_oracle_forward_depth_only`, 7,076
frames × 17 joints, SHA-256 as above, bank content digest
`75519e6394a764e3749ddaa30555b58b73a01db582ccee14f661374b9a0ed536`.

Measured: `verify_source_identity` passed; the prediction shape matches the
7,076-frame test split and the exact expected SHA. Identity method: SHA-256
equality across the historical file, the copied file, the hinge-ownership v3
binding, and the docs/39 provenance record, checked by the source-identity
verifier.

### 3.4 `O_BILATERAL` evaluation/provenance — PASS

```text
source:  /home/nd/animcv-output/framepose/sign_attr_v2/O_BILATERAL/evaluation_test.json
bytes:   14,458,736
mtime:   2026-09-06T07:25:15.403900832Z
sha256:  a5881850b3769ca2329271d327ee895dc646f236451f05c3ba2e0ff917f580a3
copied:  /private/tmp/animcv-pose-recovery.FyI66d/evaluation_test.json
```

Expected identity: schema `animcv_frame_pose_evaluation_v1`, candidate
`O_BILATERAL_oracle_forward_depth_only`, 7,076 frames, regime
`benchmark_detector_observation`, SHA-256 as above.

Measured: schema, candidate, frame count and regime all match; source-identity
checks pass. The evaluation schema itself carries no split field —
`split_in_evaluation_schema` is recorded as **`UNVERIFIABLE`**, exactly as
docs/38–39 established, and is cross-bound instead by the ownership v3 and
docs/39 provenance artifacts (3.7–3.8 below).

### 3.5 Historical sign training report — PASS (reference only)

```text
source:  /home/nd/animcv-output/framepose/sign_attr_v2/O_BILATERAL/training_report.json
bytes:   34,164
mtime:   2026-09-06T07:25:05.463993472Z
sha256:  c8573a0af070f8b0340cabcc220ad46c0ed92407232964ec22919a5ef5e84112
```

Expected identity: the `O_BILATERAL` historical configuration and checkpoint
reference. Measured: checkpoint reference
`/output/framepose/sign_attr_v2/O_BILATERAL/checkpoint.pt`; **no training or
inference was performed in this batch** — the prediction itself was already
recovered and SHA-verified (3.3), so the checkpoint is not a replay input, only
a read-only inventory record.

### 3.6 Historical sign experiment matrix — PASS (bank cross-binding)

```text
source:  /home/nd/animcv-output/framepose/sign_attr_v2/sign_experiment_matrix.json
bytes:   65,293
mtime:   2026-09-06T07:33:45.636983098Z
sha256:  72f415cb48c8a15286479db51e0587a17f5cfcc796ab1d0d7aea559595d93c8d
```

Confirms the contemporaneous bank/`O_BILATERAL` binding: bank content digest
`75519e6394a764e3749ddaa30555b58b73a01db582ccee14f661374b9a0ed536`, index SHA
`6fe13cdc91e1fdec8defd647b230cf688fa514a2799956ed40e7f805b13e0fe5`, NPZ SHA
`909090f7aa78f3c25f56fbd335eb5b2521f87c06e92d59a28010b0b28ba6b79e`, provenance
fingerprint `ca343d04e52629abcb91e8c8281744abc69b1190c3ff3a822acc128fe45a1121`.
Identity method: original server metadata, size, mtime and SHA-256, inventoried
before replay.

### 3.7 Historical hinge-ownership v3 cross-binding — PASS

```text
source:  /home/nd/animcv-output/framepose/hinge_ownership_v3/hinge_evidence_ownership.json
bytes:   460,113
mtime:   2026-09-08T14:55:54.377278326Z
sha256:  46851e31d343e2fe054b89c0afbb9d27e1b84fcc56ea37fc71f38d5a52b9a4db
copied:  /private/tmp/animcv-pose-recovery.FyI66d/hinge_evidence_ownership.json
```

Measured: source prediction SHA matches, source evaluation SHA matches, bank
content digest matches; historical `hinge_write_policy` recorded as
`minimum_norm`; historical residual total **279**, depth-correct **83** — the
same population docs/35–37 established. All checks pass. Identity method:
original SHA plus the full internal binding checks above.

### 3.8 Historical docs/39 camera and geometry provenance — PASS

```text
source:  /home/nd/animcv-output/framepose/hinge_write_policy_v3/hinge_write_policy_observation.json
bytes:   105,787
mtime:   2026-09-09T03:54:09.716612906Z
sha256:  cf1421d0c76fa5e7988eab3fd97c208977d5197684f1073787cf156e820f2464
copied:  /private/tmp/animcv-pose-recovery.FyI66d/hinge_write_policy_observation.json
```

Measured: source prediction and evaluation SHAs match; bank content digest
matches; raw split `test`; **24 raw sequences expected, 24 found locally, 0
missing/extra/hash mismatches**; 7,076 frames have absolute geometry recovered.
Historical target-reconstruction max error **0.0000298 mm** against the
implemented refusal boundary of **1.0 mm** (see docs/39 §2's own correction of
that boundary's wording). Identity method: original source SHA plus per-file
byte/SHA comparison against all 24 local test pickles, cross-bound to the
historical report.

### 3.9 Raw 3DPW test camera pickles — PASS

```text
source:                /Users/nadan/Projects/Lab/DATASET_Motion/sequenceFiles/test/*.pkl
historical reference:  /data/datasets/3dpw/sequenceFiles/test
historical LabServer63: /home/nd/animcv-data/datasets/3dpw/sequenceFiles/test/*.pkl
```

Expected identity: 24 historical sequence entries, exact byte sizes and
SHA-256 taken from the recovered docs/39 report. Measured: 24/24 files present,
zero missing, zero extra, zero byte-or-SHA mismatches.

Camera-reconstruction preflight run against these files:

| quantity | value |
|---|---|
| frames | 7,076 |
| missing sequence mappings | 0 |
| invalid contexts | 0 |
| image-size matches | 7,076 / 7,076 |
| target reconstruction max error | 0.0000412 mm |
| target reconstruction mean error | 0.0000085 mm |
| valid target middle-joint projections | 26,818 |
| target middle-joint projection failures | 0 |

Identity method: per-file size/SHA equality with the recovered historical
docs/39 manifest, plus the camera/context and reconstruction-invariant
preflight above.

### 3.10 `sign_attr_v1_INVALID_mask_discarded` prediction — FAIL (excluded, correctly)

```text
source:  /home/nd/animcv-output/framepose/sign_attr_v1_INVALID_mask_discarded/O_BILATERAL/prediction_test.npy
bytes:   1,443,632
mtime:   2026-09-06T06:49:09.139422985Z
sha256:  ce2b0fbbdd99aad01bb0db05975bc5f22cd92658166285117aaa55b769e62012
```

Expected identity: the corrected `sign_attr_v2` lineage only. Measured: this
SHA differs from the accepted v2 prediction, and the directory is itself
labelled `INVALID` (docs/29's discarded first attribution run). Disposition:
**`REJECTED_NOT_USED`** — inventoried read-only and excluded, exactly as
docs/33–43 have treated this lineage throughout.

### 3.11 `sign_attr_v1_INVALID_mask_discarded` evaluation — FAIL (excluded, correctly)

```text
source:  /home/nd/animcv-output/framepose/sign_attr_v1_INVALID_mask_discarded/O_BILATERAL/evaluation_test.json
bytes:   14,461,580
mtime:   2026-09-06T06:49:15.839347964Z
sha256:  9b60c95c93b070424d917444dff8dc7f4017ad3af886a51a3763f6ceafcde2f1
```

Same disposition as 3.10: SHA differs from the accepted v2 evaluation; this
file is paired only with the invalid v1 prediction. **`REJECTED_NOT_USED`**.

## 4. Sanity fingerprint — not identity evidence

Recorded once the recovered `O_BILATERAL` prediction and evaluation were in
hand, purely as a fingerprint sanity check against the numbers docs/33–39
already published. This is **not** part of the identity/verification gate
above and proves nothing about provenance by itself — it is only a check that
the recovered numbers still read as the same experiment:

```text
candidate:             O_BILATERAL_oracle_forward_depth_only
mpjpe_mean_mm:         79.229486
pa_mpjpe_mean_mm:      56.424633
root_yaw_mean_degrees: 7.999395
root_yaw_p95_degrees:  18.644923
hinge_flip_mean:       0.0189153
hinge_mae_mean_degrees: 24.098994
```

These match docs/33–39's published `O_BILATERAL` baseline exactly (root yaw
mean 7.999° / P95 18.645°, MPJPE 79.229 mm, PA-MPJPE 56.425 mm, hinge flip
0.0189, hinge MAE 24.099°).

## 5. Status

All nine required artifacts pass identity verification; the two
`sign_attr_v1_INVALID_mask_discarded` files are present only as evidence of
correct exclusion. `real_replay_authorized: true` — this identity gate was
completed before the real docs/39-input Pose Reconciliation replay documented
in [`44_WORKLOG_POSE_RECONCILIATION_REAL_REPLAY.md`](44_WORKLOG_POSE_RECONCILIATION_REAL_REPLAY.md).

## 6. Machine-readable recovery table

This normalized record preserves the required columns for downstream
verification. `UNVERIFIABLE` is used only for the evaluation schema's missing
split field; the recovered ownership/docs/39 records cross-bind that identity.

```json
{
  "schema": "animcv_historical_artifact_recovery_table_v1",
  "required_inputs_pass": true,
  "real_replay_authorized": true,
  "artifacts": [
    {
      "artifact": "FrameBank JSON index + numeric arrays",
      "status": "PASS",
      "source_path": "/home/nd/animcv-output/framepose/bank_3dpw_paired_v2.json + /home/nd/animcv-output/framepose/bank_3dpw_paired_v2.npz",
      "identity_method": "Original server SHA-256, load_bank content_digest, exact split/regime check, contemporaneous matrix cross-binding.",
      "expected_identity": {"content_digest": "75519e6394a764e3749ddaa30555b58b73a01db582ccee14f661374b9a0ed536", "samples": 21817, "split_counts": {"train": 11334, "validation": 3407, "test": 7076}},
      "measured_identity": {"schema": "animcv_frame_pose_bank_v2", "content_digest": "75519e6394a764e3749ddaa30555b58b73a01db582ccee14f661374b9a0ed536", "index_sha256": "6fe13cdc91e1fdec8defd647b230cf688fa514a2799956ed40e7f805b13e0fe5", "npz_sha256": "909090f7aa78f3c25f56fbd335eb5b2521f87c06e92d59a28010b0b28ba6b79e", "samples": 21817, "split_counts": {"train": 11334, "validation": 3407, "test": 7076}, "regime": "benchmark_detector_observation"}
    },
    {
      "artifact": "O_BILATERAL prediction",
      "status": "PASS",
      "source_path": "/home/nd/animcv-output/framepose/sign_attr_v2/O_BILATERAL/prediction_test.npy",
      "identity_method": "Exact SHA-256 cross-bound by ownership v3 and docs/39; verify_source_identity shape/candidate/split-frame check.",
      "expected_identity": {"candidate": "O_BILATERAL_oracle_forward_depth_only", "sha256": "6df04c2031b2bde6a4f0af37fdae0a7f0228e9fc23401e15e1aa2b3bd35695a5", "frames": 7076, "joints": 17, "bank_content_digest": "75519e6394a764e3749ddaa30555b58b73a01db582ccee14f661374b9a0ed536"},
      "measured_identity": {"sha256": "6df04c2031b2bde6a4f0af37fdae0a7f0228e9fc23401e15e1aa2b3bd35695a5", "shape_matches_test_split": true, "source_identity_check": "PASS"}
    },
    {
      "artifact": "O_BILATERAL evaluation/provenance",
      "status": "PASS",
      "source_path": "/home/nd/animcv-output/framepose/sign_attr_v2/O_BILATERAL/evaluation_test.json",
      "identity_method": "Exact evaluation SHA cross-bound by ownership v3/docs/39; candidate/frame/regime verifier.",
      "expected_identity": {"schema": "animcv_frame_pose_evaluation_v1", "candidate": "O_BILATERAL_oracle_forward_depth_only", "sha256": "a5881850b3769ca2329271d327ee895dc646f236451f05c3ba2e0ff917f580a3", "frames": 7076, "regime": "benchmark_detector_observation"},
      "measured_identity": {"sha256": "a5881850b3769ca2329271d327ee895dc646f236451f05c3ba2e0ff917f580a3", "candidate_matches": true, "frame_count_matches": true, "regime_matches": true, "split_field": "UNVERIFIABLE in evaluation schema; independently cross-bound by ownership v3/docs/39"}
    },
    {
      "artifact": "Raw 3DPW test camera pickles",
      "status": "PASS",
      "source_path": "/Users/nadan/Projects/Lab/DATASET_Motion/sequenceFiles/test/*.pkl",
      "identity_method": "Per-file byte size/SHA-256 equality across LabServer63 raw files, recovered docs/39 manifest, and local replay inputs; camera reconstruction/context preflight.",
      "expected_identity": {"files": 24, "manifest_source": "/home/nd/animcv-output/framepose/hinge_write_policy_v3/hinge_write_policy_observation.json"},
      "measured_identity": {"files": 24, "missing": 0, "extra": 0, "byte_or_sha_mismatches": 0, "camera_frames": 7076, "invalid_contexts": 0, "target_middle_projection_failures": 0}
    },
    {
      "artifact": "Historical hinge ownership v3",
      "status": "PASS",
      "source_path": "/home/nd/animcv-output/framepose/hinge_ownership_v3/hinge_evidence_ownership.json",
      "identity_method": "Exact artifact SHA plus internal source/bank/policy/residual cross-bind checks.",
      "expected_identity": {"sha256": "46851e31d343e2fe054b89c0afbb9d27e1b84fcc56ea37fc71f38d5a52b9a4db", "historical_policy": "minimum_norm", "residual_total": 279, "depth_correct": 83},
      "measured_identity": {"source_prediction_matches": true, "source_evaluation_matches": true, "bank_digest_matches": true, "policy_matches": true, "residual_total": 279, "depth_correct": 83, "all_checks_pass": true}
    },
    {
      "artifact": "sign_attr_v1_INVALID_mask_discarded O_BILATERAL files",
      "status": "FAIL",
      "source_path": "/home/nd/animcv-output/framepose/sign_attr_v1_INVALID_mask_discarded/O_BILATERAL/{prediction_test.npy,evaluation_test.json}",
      "identity_method": "SHA comparison against accepted sign_attr_v2 lineage and explicit invalid-directory disposition.",
      "expected_identity": "Corrected sign_attr_v2 only.",
      "measured_identity": "Both hashes differ from the accepted v2 prediction/evaluation.",
      "disposition": "REJECTED_NOT_USED"
    }
  ]
}
```
