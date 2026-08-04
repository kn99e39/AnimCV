#!/usr/bin/env python3
"""Build docs/07 C1 dataset intake manifest for datasets/mpi_inf_3dhp/.

Run from anywhere; paths are resolved relative to the repo root (two levels
above this script). Re-run after re-downloading data to refresh checksums.
"""
import hashlib
import json
import os

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_ROOT = os.path.join(REPO_ROOT, "datasets", "mpi_inf_3dhp")
OUT = os.path.join(DATA_ROOT, "intake_manifest.json")
TRAIN_SUBJECTS = [1, 2, 3, 4, 5, 6]
HOLDOUT_SUBJECTS = [7, 8]

# Fixed at first download (2026-08-04) from https://vcai.mpi-inf.mpg.de/3dhp-dataset/mpi_inf_3dhp.zip
# (the small get_dataset.sh/license.txt bundle, not the dataset itself). Not re-verified on each
# manifest rebuild since that bundle is not stored in this repo.
OFFICIAL_ZIP_SHA256 = "d5c69b03922a1a29fa58117795f44a86b3a33c679a7f9c9b486869db7edd4433"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def entry(subject, seq, split):
    d = os.path.join(DATA_ROOT, f"S{subject}", f"Seq{seq}")
    ann, cal = os.path.join(d, "annot.mat"), os.path.join(d, "camera.calibration")
    rel = lambda p: os.path.relpath(p, REPO_ROOT)
    return {
        "subject": f"S{subject}", "sequence": f"Seq{seq}", "split": split,
        "annotation_path": rel(ann), "annotation_sha256": sha256(ann),
        "annotation_size_bytes": os.path.getsize(ann),
        "calibration_path": rel(cal), "calibration_sha256": sha256(cal),
        "calibration_size_bytes": os.path.getsize(cal),
    }


sequences = (
    [entry(s, seq, "train") for s in TRAIN_SUBJECTS for seq in (1, 2)]
    + [entry(s, seq, "holdout") for s in HOLDOUT_SUBJECTS for seq in (1, 2)]
)

manifest = {
    "schema": "animcv_dataset_intake_manifest_v1",
    "dataset": "MPI-INF-3DHP",
    "official_source": "https://vcai.mpi-inf.mpg.de/3dhp-dataset/",
    "official_download_zip": "https://vcai.mpi-inf.mpg.de/3dhp-dataset/mpi_inf_3dhp.zip",
    "official_download_zip_sha256": OFFICIAL_ZIP_SHA256,
    "download_mechanism": "get_dataset.sh wget's http(s)://gvv.mpi-inf.mpg.de/3dhp-dataset/S<subject>/Seq<seq>/{annot.mat,camera.calibration,imageSequence/*.zip}; "
                            "no registration/credentials required. Only annot.mat + camera.calibration were fetched "
                            "(direct 2D+3D ground truth; imageSequence video was not needed for GT-based training/eval).",
    "repo_storage_format": "Each annot.mat (94-206MB) exceeds GitHub's 100MB per-file limit, so it is committed as "
                            "<=90MB annot.mat.part-NN chunks (plain git objects, no Git LFS) and reassembled on "
                            "demand by scripts/training/reassemble_mpi_inf_3dhp.sh, which verifies the rebuilt "
                            "file's SHA-256 against annotation_sha256 below. camera.calibration is small enough to "
                            "commit directly.",
    "version": {
        "annot_mat_fetched": "2026-08-04",
        "citation": [
            "Mehta, D. et al. VNect: Real-time 3D Human Pose Estimation With A Single RGB Camera, ACM ToG (SIGGRAPH) 2017",
            "Mehta, D. et al. Monocular 3D Human Pose Estimation In The Wild Using Improved CNN Supervision, 3DV 2017",
        ],
    },
    "access_conditions": {
        "summary": "Non-commercial research/personal use per the official license.txt bundled with the dataset "
                    "download (not stored in this repo; see official_download_zip above). Redistribution/resale of "
                    "the data or derivatives is not permitted by that license. Citation required if used. License "
                    "review and compliance are the project owner's responsibility (per owner instruction, 2026-08-04); "
                    "this manifest records the terms as published, it does not adjudicate them.",
        "owner_review_required": False,
        "owner_instruction": "Owner explicitly waived license-review blocking for this task on 2026-08-04: "
                              "'do not care about the license. it's my concern to do.' Recorded here for provenance, "
                              "not as a legal judgement. The owner also explicitly directed committing this raw data "
                              "and the resulting checkpoint into this repository on 2026-08-04.",
    },
    "coordinate_and_schema_provenance": {
        "source_camera_axes": "+X right, +Y down, +Z forward, millimetres (per official README.txt / mpi3dhp_adapter.py)",
        "animcv_camera_axes": "+X right, +Y forward, +Z up, metres, root(pelvis)-relative",
        "conversion": "src/pose/mpi3dhp_adapter.py:load_mpi3dhp_ground_truth — column swap (x, z, -y) * 0.001, "
                       "then subtract joint 14 (pelvis) position",
        "raw_joint_set": "28-joint 'all' set (util/mpii_get_joint_set.m)",
        "canonical_joint_subset": "17-joint H36M-compatible subset via fixed index list _JOINTS_17 in mpi3dhp_adapter.py",
        "landmark_confidence": "GT import sets confidence=1.0 / visible=True for all joints (no source occlusion flag "
                                "consumed) — this is dataset ground truth, not a detector output; see docs/08 Candidate B framing.",
        "camera_selection": "camera_index 0 only (mpii_get_camera_set.m 'vnect' set is [0,1,2,4,5,6,7,8]; "
                              "this run uses camera 0 for both train and holdout, matching docs/06's worked example)",
        "fps": "Not read from mpii_get_sequence_info.m per-sequence table (25 or 50 fps depending on subject/sequence); "
               "src/pose/mpi3dhp_adapter.py defaults fps=25.0 for all imports since the CLI does not expose an --fps "
               "override. This only affects the recorded PoseFrame.timestamp field, not joint positions or the "
               "training loss (frame-indexed, not time-indexed) — flagged here as a known provenance gap, not corrected.",
    },
    "official_split_used": {
        "train_subjects": [f"S{s}" for s in TRAIN_SUBJECTS],
        "holdout_subjects": [f"S{s}" for s in HOLDOUT_SUBJECTS],
        "note": "MPI-INF-3DHP's own official benchmark test set is a separate download (mpi_inf_3dhp_test_set.zip, "
                "TS1-TS6, different capture rig/annotation format) not fetched in this run. docs/06 runbook explicitly "
                "permits 'separate source sequences or subjects for train and holdout when available' as the sanctioned "
                "protocol for this project, so S7/S8 were held out by subject (disjoint from training) instead. "
                "No frames were split within a sequence.",
    },
    "purpose": "docs/06_SERVER_AI_AGENT_TRAINING_RUNBOOK.md first server training run: supervised temporal 2D->3D "
               "lifter baseline (training.temporal_lifter) for AnimCV's retarget pipeline.",
    "sequences": sequences,
}

with open(OUT, "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2, ensure_ascii=False)
print(f"wrote {OUT} with {len(sequences)} sequence entries")
