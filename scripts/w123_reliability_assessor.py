"""W123 scratch experiment: learned 3D error assessor around frozen S0 (stages run on Main only).

Stages (each writes a fresh output directory under the scratch workspace and refuses to overwrite):
  preflight  identities, byte-identical S0 reproduction, GT row contract, splits, MPI/real-clip inventory
  declare    pre-outcome declaration (features, training, calibration, metrics, gates, CASE rule, code blobs)
  train      assessor TRAIN / baselines TRAIN / conformal CALIBRATION (no FINAL access)
  evaluate   FINAL held-out 3DPW (M primary, B cross-sensor), strata, bootstrap, triage, review selections
  mpi        frozen S0 on MPI-INF-3DHP oracle 2D (independent subjects) + assessor transfer
  realvideo  GT-free W121/W122 S1-S3 replay, sidecars, S1 f251 inspection
  decide     frozen CASE rule
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "src"), str(REPO / "scripts")]

import w123_reliability_base as RB  # noqa: E402

HOME = Path(os.environ.get("USERPROFILE", str(Path.home())))
WS = HOME / "animcv-w123"
W101 = HOME / "animcv-w101" / "data"
W102 = HOME / "animcv-w102"
W121 = HOME / "animcv-w121"
W122 = HOME / "animcv-w122"
BANK = W101 / "bank_3dpw_paired_v2.json"
CHECKPOINT = W101 / "s0_checkpoint.pt"
CHECKPOINT_SHA = "6c12fea9f84ea0493fada8b9acc96cd5c70fdd5d9114603e954f9412f4247b44"
BANK_DIGEST = "75519e6394a764e3749ddaa30555b58b73a01db582ccee14f661374b9a0ed536"
W102_PRED = {("B", "validation"): "f89d9bd071be9eb9e60d8b3dd77eb53db904fc10dfff418242c409cb7ee1d27d",
             ("B", "test"): "065ae4be7c2dc2ea340d0fb3fad54acefe3e2676ba8ff387bbada15d0db134bc"}
W102_OBS_SHA = "7fc15fc8a575358b71fdff1c5e61470e92dc51028a98442b273d09b55bd5a0c2"
W102_SIDE_SHA = "d0e3a8796d8e4aad08df311b2ce7da818be46d4f7f6e287371c771845ba1b276"
SPLIT_SALT = "W123-split-v1:"
MPI_ROOT = REPO / "datasets" / "mpi_inf_3dhp"
MPI_CAMERAS = (0, 1, 2, 4, 5, 6, 7, 8)       # official 'vnect' chest-height camera set
MPI_STRIDE = 100
MPI_IMAGE = 2048
# 3DHP raw-28 indices (util/mpii_get_joint_set.m 'all')
MPI28 = {"spine": 3, "pelvis": 4, "neck": 5, "head": 6, "head_top": 7, "left_shoulder": 9, "left_elbow": 10,
         "left_wrist": 11, "right_shoulder": 14, "right_elbow": 15, "right_wrist": 16, "left_hip": 18,
         "left_knee": 19, "left_ankle": 20, "right_hip": 23, "right_knee": 24, "right_ankle": 25}
REAL_CLIPS = {"S1": "Double_Kettlebell_Strict_Press_side_view_.webm", "S2": "Firejuggling_2009_ubt.ogv",
              "S3": "Bent-over_row_-_exercise_demonstration_video.webm"}
S1_EVENT = {"clip": "S1", "frames": list(range(241, 262)), "event_frame": 251,
            "joints": ["left_shoulder", "left_elbow", "left_wrist", "right_wrist"]}
SMOKE = os.environ.get("W123_SMOKE") == "1"
RESULTS = WS / ("results_smoke" if SMOKE else "results")
CODE_FILES = ("scripts/w123_reliability_base.py", "scripts/w123_reliability_assessor.py",
              "tests/test_w123_reliability_assessor.py")


def process_local_env():
    for key, rel in {"TMP": "tmp", "TEMP": "tmp", "TMPDIR": "tmp", "XDG_CACHE_HOME": "cache",
                     "TORCH_HOME": "cache/torch", "CUDA_CACHE_PATH": "cache/cuda", "MPLCONFIGDIR": "cache/mpl"}.items():
        p = WS / rel
        p.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(p)
    os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")


def fresh(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=False)
    return path


def git_blob(path: Path) -> str:
    return subprocess.run(["git", "hash-object", str(path)], cwd=REPO, check=True, capture_output=True,
                          text=True).stdout.strip()


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


# ----------------------------------------------------------------------------------------------------------------
# Population
# ----------------------------------------------------------------------------------------------------------------

def video_of(sample_id: str) -> str:
    return sample_id.split(":")[1]


def group_of(video: str) -> str:
    return re.sub(r"_\d+$", "", video)


def split_assignment(videos_val, videos_test) -> dict:
    """VAL videos -> TRAIN. TEST groups (numeric suffix removed) sorted by sha256(salt+group); index%3==0 -> CALIB."""
    groups = sorted({group_of(v) for v in videos_test}, key=lambda g: hashlib.sha256((SPLIT_SALT + g).encode()).hexdigest())
    bucket = {g: ("CALIB" if i % 3 == 0 else "FINAL") for i, g in enumerate(groups)}
    out = {v: "TRAIN" for v in videos_val}
    for v in videos_test:
        out[v] = bucket[group_of(v)]
    if set(group_of(v) for v in videos_val) & set(groups):
        raise RuntimeError("STOP: a VAL group name also occurs in TEST")
    return {"video_split": out, "test_group_order": groups, "test_group_bucket": bucket}


def load_population():
    obs = np.load(W102 / "results/handoff/observations.npz", allow_pickle=False)
    side = np.load(W102 / "results/handoff/observations_side.npz", allow_pickle=False)
    from framepose.bank import load_bank
    bank = load_bank(BANK)
    if bank.content_digest() != BANK_DIGEST:
        raise RuntimeError("STOP: bank identity")
    positions = obs["positions"]
    expected = np.concatenate([bank.indices("validation"), bank.indices("test")])
    if not np.array_equal(positions, expected) or not np.array_equal(side["positions"], positions):
        raise RuntimeError("STOP: population order")
    sample_id = np.array([bank.samples[p].sample_id for p in positions])
    if not np.array_equal(sample_id, obs["sample_id"]) or not np.array_equal(sample_id, side["sample_id"]):
        raise RuntimeError("STOP: sample identity")
    split_name = np.array([bank.samples[p].split for p in positions])
    videos = np.array([video_of(s) for s in sample_id])
    assign = split_assignment(sorted(set(videos[split_name == "validation"])), sorted(set(videos[split_name == "test"])))
    w123_split = np.array([assign["video_split"][v] for v in videos])
    return {"bank": bank, "obs": obs, "side": side, "positions": positions, "sample_id": sample_id,
            "orig_split": split_name, "video": videos, "group": np.array([group_of(v) for v in videos]),
            "actor": np.array([s.split("#")[0] for s in sample_id]), "w123_split": w123_split,
            "identity": side["identity"], "assign": assign,
            "timestamp": obs["timestamp"], "target": bank.arrays["target_3d"][positions].astype(np.float32),
            "bench_visible": bank.arrays["target_valid"][positions]}


def s0_geometry(pop, regime: str) -> np.ndarray:
    from framepose.train import geometry_tensor
    from w101_reference_estimator import geometry_for_condition
    if regime == "B":
        return geometry_tensor(pop["bank"])[pop["positions"]]
    if regime == "M":
        return geometry_for_condition(pop["bank"], pop["obs"], "M")[pop["positions"]]
    raise ValueError(regime)


def s0_forward(geometry: np.ndarray) -> np.ndarray:
    import torch
    from framepose.signs import neutral_sign_states
    from framepose.train import load_checkpoint, predict
    if RB.sha_file(CHECKPOINT) != CHECKPOINT_SHA:
        raise RuntimeError("STOP: checkpoint identity")
    if not torch.cuda.is_available():
        raise RuntimeError("STOP: CUDA unavailable on Main")
    model, ckpt = load_checkpoint(CHECKPOINT, device="cuda")
    if ckpt.get("candidate", {}).get("name") != "S0_neutral_sign":
        raise RuntimeError("STOP: checkpoint candidate")
    signs = neutral_sign_states(len(geometry))
    pred = predict(model, torch, geometry, None, np.arange(len(geometry)), torch.device("cuda"), batch_size=512,
                   amp_enabled=False, signs=signs).astype(np.float32)
    if pred.shape != (len(geometry), 17, 3) or not np.isfinite(pred).all():
        raise RuntimeError("STOP: S0 output invalid")
    return pred


def s0_forward_split(pop, geometry):
    """Exactly W102's call pattern: separate VALIDATION and TEST forwards in bank order, batch 512."""
    out = np.zeros((len(geometry), 17, 3), dtype=np.float32)
    for split in ("validation", "test"):
        idx = np.flatnonzero(pop["orig_split"] == split)
        out[idx] = s0_forward(geometry[idx])
    return out


def load_w102_predictions(pop, regime):
    out = np.zeros((len(pop["positions"]), 17, 3), dtype=np.float32)
    for split in ("validation", "test"):
        idx = np.flatnonzero(pop["orig_split"] == split)
        out[idx] = np.load(W102 / f"results/s0_sources/prediction_{regime}_Main_{split}.npy")
    return out


# ----------------------------------------------------------------------------------------------------------------
# MPI-INF-3DHP (oracle_geometry, independent subjects)
# ----------------------------------------------------------------------------------------------------------------

def mpi_reassemble(out_dir: Path) -> list[dict]:
    manifest = json.loads((MPI_ROOT / "intake_manifest.json").read_text(encoding="utf-8"))
    records = []
    for entry in manifest["sequences"]:
        rel = Path(entry["annotation_path"])
        subject, seq = rel.parts[-3], rel.parts[-2]
        parts = sorted((MPI_ROOT / subject / seq).glob("annot.mat.part-*"))
        target = out_dir / subject / seq / "annot.mat"
        target.parent.mkdir(parents=True, exist_ok=True)
        h = hashlib.sha256()
        with open(target, "wb") as handle:
            for part in parts:
                data = part.read_bytes()
                h.update(data)
                handle.write(data)
        ok = h.hexdigest() == entry["annotation_sha256"]
        records.append({"subject": subject, "sequence": seq, "parts": len(parts), "sha256": h.hexdigest(),
                        "manifest_sha256": entry["annotation_sha256"], "match": ok,
                        "calibration_sha256": RB.sha_file(MPI_ROOT / subject / seq / "camera.calibration"),
                        "calibration_manifest_sha256": entry.get("calibration_sha256")})
        if not ok:
            raise RuntimeError(f"STOP: MPI annotation hash mismatch {subject}/{seq}")
    return records


def mpi_population(mat_root: Path) -> dict:
    from scipy.io import loadmat
    rows = {"subject": [], "sequence": [], "camera": [], "frame": [], "input_2d": [], "input_valid": [], "target": []}
    idx = RB.JOINT_NAMES
    for subject in sorted(p.name for p in mat_root.iterdir()):
        for seq in sorted(p.name for p in (mat_root / subject).iterdir()):
            data = loadmat(mat_root / subject / seq / "annot.mat")
            for cam in MPI_CAMERAS:
                a2 = np.asarray(data["annot2"][cam, 0]).reshape(-1, 28, 2)
                a3 = np.asarray(data["annot3"][cam, 0]).reshape(-1, 28, 3)
                frames = np.arange(0, len(a2), MPI_STRIDE)
                for f in frames:
                    p2, p3 = a2[f].astype(np.float64), a3[f].astype(np.float64)
                    xy = np.zeros((17, 3), dtype=np.float64)
                    valid = np.zeros(17, dtype=bool)
                    direct = ("left_hip", "left_knee", "left_ankle", "right_hip", "right_knee", "right_ankle",
                              "left_shoulder", "left_elbow", "left_wrist", "right_shoulder", "right_elbow",
                              "right_wrist", "head")
                    for name in direct:
                        j = idx.index(name)
                        pt = p2[MPI28[name]]
                        inside = bool(np.isfinite(pt).all() and 0 <= pt[0] < MPI_IMAGE and 0 <= pt[1] < MPI_IMAGE)
                        xy[j, :2] = pt / MPI_IMAGE
                        xy[j, 2] = 1.0 if inside else 0.0
                        valid[j] = inside

                    def mid(a, b):
                        ja, jb = idx.index(a), idx.index(b)
                        return (xy[ja, :2] + xy[jb, :2]) / 2, bool(valid[ja] and valid[jb])
                    for name, (a, b) in {"neck": ("left_shoulder", "right_shoulder"),
                                         "thorax": ("left_shoulder", "right_shoulder"),
                                         "pelvis": ("left_hip", "right_hip")}.items():
                        j = idx.index(name)
                        xy[j, :2], valid[j] = mid(a, b)
                        xy[j, 2] = 1.0 if valid[j] else 0.0
                    j = idx.index("spine")
                    xy[j, :2] = (xy[idx.index("neck"), :2] + xy[idx.index("pelvis"), :2]) / 2
                    valid[j] = bool(valid[idx.index("neck")] and valid[idx.index("pelvis")])
                    xy[j, 2] = 1.0 if valid[j] else 0.0
                    gt = np.zeros((17, 3))
                    gt_map = {n: MPI28[n] for n in idx if n in MPI28}
                    gt_map["thorax"] = MPI28["neck"]
                    for name in idx:
                        gt[idx.index(name)] = p3[gt_map[name]]
                    gt = np.stack([gt[:, 0], gt[:, 2], -gt[:, 1]], axis=-1) / 1000.0
                    rows["subject"].append(subject)
                    rows["sequence"].append(seq)
                    rows["camera"].append(cam)
                    rows["frame"].append(int(f))
                    rows["input_2d"].append(xy.astype(np.float32))
                    rows["input_valid"].append(valid)
                    rows["target"].append(gt.astype(np.float32))
    out = {k: np.asarray(v) for k, v in rows.items()}
    return out


def mpi_geometry(pop) -> np.ndarray:
    from framepose.crops import crop_box, geometry_in_crop
    g = np.zeros((len(pop["frame"]), 17, 4), dtype=np.float32)
    for i in range(len(g)):
        box = crop_box(pop["input_2d"][i], pop["input_valid"][i], (MPI_IMAGE, MPI_IMAGE))
        g[i] = geometry_in_crop(pop["input_2d"][i], pop["input_valid"][i], (MPI_IMAGE, MPI_IMAGE), box)
    return g


# ----------------------------------------------------------------------------------------------------------------
# Stages
# ----------------------------------------------------------------------------------------------------------------

def stage_preflight(args):
    out = fresh(RESULTS / "preflight")
    report = {"schema": "w123_preflight_v1", "started": time.strftime("%Y-%m-%dT%H:%M:%S")}
    report["checkpoint_sha256_before"] = RB.sha_file(CHECKPOINT)
    report["checkpoint_ok"] = report["checkpoint_sha256_before"] == CHECKPOINT_SHA
    import torch
    ck = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    report["checkpoint_training_provenance"] = {"bank_content_digest": ck.get("bank_content_digest"),
                                                "selection": ck.get("selection"), "candidate": ck.get("candidate"),
                                                "loss_contract": ck.get("loss_contract", {}).get("name")}
    report["w102_observations_sha256"] = RB.sha_file(W102 / "results/handoff/observations.npz")
    report["w102_side_sha256"] = RB.sha_file(W102 / "results/handoff/observations_side.npz")
    report["w102_observation_identity_ok"] = (report["w102_observations_sha256"] == W102_OBS_SHA and
                                              report["w102_side_sha256"] == W102_SIDE_SHA)
    pop = load_population()
    report["bank_content_digest"] = pop["bank"].content_digest()
    s0rep = load_json(W102 / "results/s0_sources/s0_report.json")
    repro = {}
    for regime in ("B", "M"):
        geometry = s0_geometry(pop, regime)
        pred = s0_forward_split(pop, geometry)
        stored = load_w102_predictions(pop, regime)
        per = {}
        for split in ("validation", "test"):
            idx = np.flatnonzero(pop["orig_split"] == split)
            path = W102 / f"results/s0_sources/prediction_{regime}_Main_{split}.npy"
            per[split] = {"stored_sha256": RB.sha_file(path),
                          "w102_report_sha256": s0rep["conditions"][f"{regime}_Main"]["splits"][split]["sha256"],
                          "reproduced_bytes_equal": bool(pred[idx].tobytes() == stored[idx].tobytes()),
                          "max_abs_diff_m": float(np.abs(pred[idx] - stored[idx]).max())}
            per[split]["stored_matches_report"] = per[split]["stored_sha256"] == per[split]["w102_report_sha256"]
        repro[regime] = {"geometry_sha256": RB.sha_array(geometry), "splits": per}
    report["s0_reproduction"] = repro
    # raw GT row contract: W102 rows.json G (raw-derived) vs bank target_3d; W102 MPJPE parity (formula check)
    rows = load_json(W102 / "results/evaluation/rows.json")
    pos_index = {int(p): i for i, p in enumerate(pop["positions"])}
    gt_diff, mp_diff = 0.0, {"B": 0.0, "M": 0.0}
    checked = 0
    stored = {r: load_w102_predictions(pop, r) for r in ("B", "M")}
    for r in rows:
        i = pos_index[int(r["p"])]
        if r["sample_id"] != pop["sample_id"][i]:
            raise RuntimeError("STOP: rows.json identity")
        g = np.asarray(r["G"], dtype=np.float64)
        gt_diff = max(gt_diff, float(np.abs(g - pop["target"][i]).max()))
        for regime in ("B", "M"):
            m = r.get("m", {}).get(regime)
            if m and m.get("mpjpe") is not None:
                e = np.linalg.norm((stored[regime][i] - stored[regime][i][:1]) - (g - g[:1]), axis=-1)
                mp_diff[regime] = max(mp_diff[regime], abs(float(e.mean() * 1000) - m["mpjpe"]))
        checked += 1
    report["gt_row_contract"] = {"rows_checked": checked, "max_abs_rows_G_minus_bank_target_m": gt_diff,
                                 "target_finite_all": bool(np.isfinite(pop["target"]).all()),
                                 "w102_mpjpe_parity_max_abs_mm_17joint_mean_pelvis_rooted": mp_diff,
                                 "pass": bool(gt_diff <= 1e-6 and max(mp_diff.values()) <= 1e-3)}
    del rows
    # split accounting (no outcomes)
    acc = {}
    for s in ("TRAIN", "CALIB", "FINAL"):
        m = pop["w123_split"] == s
        same = m & (pop["identity"] == "same_person")
        acc[s] = {"B_actor_frames": int(m.sum()), "M_same_person_rows": int(same.sum()),
                  "videos": sorted(set(pop["video"][m])), "groups": sorted(set(pop["group"][m])),
                  "videos_with_M_rows": sorted(set(pop["video"][same])),
                  "actor_sequences": len(set(pop["actor"][m]))}
    for a, b in (("TRAIN", "CALIB"), ("TRAIN", "FINAL"), ("CALIB", "FINAL")):
        if set(acc[a]["groups"]) & set(acc[b]["groups"]) or set(acc[a]["videos"]) & set(acc[b]["videos"]):
            raise RuntimeError("STOP: split leakage")
    image_sha = pop["side"]["image_sha256"]
    img_split = {}
    for h, s in zip(image_sha, pop["w123_split"]):
        img_split.setdefault(h, set()).add(s)
    report["split"] = {"rule": "VAL videos->TRAIN; TEST groups sorted by sha256('" + SPLIT_SALT +
                       "'+group), index%3==0->CALIB else FINAL", "test_group_bucket": pop["assign"]["test_group_bucket"],
                       "accounting": acc, "images_crossing_partitions": int(sum(len(v) > 1 for v in img_split.values())),
                       "disjoint_videos_groups": True}
    # MPI
    mat_root = WS / "data" / "mpi_annot"
    report["mpi_annotations"] = mpi_reassemble(mat_root)
    mpi = mpi_population(mat_root)
    np.savez(out / "mpi_population.npz", **mpi)
    report["mpi_population"] = {"rows": int(len(mpi["frame"])), "subjects": sorted(set(mpi["subject"].tolist())),
                                "cameras": list(MPI_CAMERAS), "stride": MPI_STRIDE,
                                "rows_with_all_12_limbs_in_image": int(mpi["input_valid"][:, list(RB.LIMB_JOINTS)].all(1).sum()),
                                "rows_with_lt3_valid": int((mpi["input_valid"].sum(1) < 3).sum()),
                                "target_finite": bool(np.isfinite(mpi["target"]).all()),
                                "npz_sha256": RB.sha_file(out / "mpi_population.npz")}
    # real clips
    clips = {}
    for cid, name in REAL_CLIPS.items():
        a = W122 / "results/reexec_perception" / cid / "observations_predictions.npz"
        b = W121 / "results/perceive/candidates" / name / "m1s0/observations_predictions.npz"
        z = np.load(a, allow_pickle=False)
        clips[cid] = {"w122_npz": str(a), "w122_sha256": RB.sha_file(a), "w121_sha256": RB.sha_file(b),
                      "frames": int(len(z["frame_index"])), "keys": list(z.files),
                      "clip_sha256": load_json(a.parent / "report.json").get("clip_sha256")}
        za = {k: z[k] for k in z.files}
        zb = np.load(b, allow_pickle=False)
        clips[cid]["arrays_byte_equal_w121"] = all(za[k].tobytes() == zb[k].tobytes() for k in ("input_2d", "input_valid", "geometry", "predictions", "frame_index", "timestamp"))
    report["real_clips"] = clips
    report["checkpoint_sha256_after"] = RB.sha_file(CHECKPOINT)
    report["pass"] = bool(report["checkpoint_ok"] and report["w102_observation_identity_ok"] and
                          report["bank_content_digest"] == BANK_DIGEST and report["gt_row_contract"]["pass"] and
                          all(v["reproduced_bytes_equal"] and v["stored_matches_report"]
                              for r in repro.values() for v in r["splits"].values()) and
                          all(c["arrays_byte_equal_w121"] for c in clips.values()) and
                          report["checkpoint_sha256_after"] == CHECKPOINT_SHA)
    print("preflight pass", report["pass"], RB.dump(out / "preflight.json", report))


def declaration_content(preflight_sha: str) -> dict:
    cfg = RB.AssessorConfig()
    return {
        "schema": "w123_declaration_v1", "batch": 123, "outcomes_inspected": False,
        "preflight_sha256": preflight_sha,
        "code_blobs": {f: git_blob(REPO / f) for f in CODE_FILES},
        "frozen_predictor": {"name": "S0_neutral_sign", "checkpoint_sha256": CHECKPOINT_SHA,
                             "trained_on": "3DPW bank_3dpw_paired_v2 TRAIN split (digest 75519e63...)",
                             "selected_on": "3DPW VALIDATION MPJPE (epoch 99)",
                             "exposure": {"3DPW_TRAIN": "trained (not used by W123)",
                                          "3DPW_VAL": "checkpoint selection only; repeatedly design-informing",
                                          "3DPW_TEST": "never trained/selected; repeatedly design-informing (W91-W105)",
                                          "MPI-INF-3DHP": "absent from S0 bank; earlier AnimCV temporal-lifter work used it"}},
        "primary_target": {"definition": "per-joint 3D Euclidean error of the frozen S0 prediction vs raw 3DPW GT, "
                                         "both pelvis(joint0)-rooted, camera_root_relative metres, 16 non-root joints",
                           "support": "all 3DPW VAL/TEST actor-frames have raw campose_valid & finite GT (W102 row "
                                      "contract); M regime only same-person rows (other/unresolved rows observe a "
                                      "different or unknown person, so their GT target is invalid for that "
                                      "observation); invalid inputs are NOT excluded (they are a stratum)",
                           "output_semantics": "q90: per-joint error level (m) with declared 0.90 coverage after "
                                               "split-conformal log-space calibration on CALIB videos; q50 likewise "
                                               "0.50. Not a probability of correctness; not detector confidence.",
                           "material_event": f"e > {RB.MATERIAL_M} m (primary); e > {RB.SECONDARY_M} m reported"},
        "regimes": {"PRIMARY_M": "real_animcv_observation (W102 M1 RTMPose/RTMDet), same-person rows",
                    "CROSS_SENSOR_B": "benchmark_detector_observation (3DPW shipped detector), all rows; assessor "
                                      "NOT retrained; M calibration applied unchanged, B-recalibrated view secondary",
                    "INDEPENDENT_MPI": "oracle_geometry: MPI-INF-3DHP train-set annot2 projected GT 2D (score 1.0, "
                                       "validity = inside 2048px image), 8 subjects x 2 seq, vnect cameras, every "
                                       f"{MPI_STRIDE}th frame; 12 limb joints only; hip-midpoint-rooted primary, "
                                       "pelvis-rooted sensitivity; skeleton convention differs (SMPL vs 3DHP); "
                                       "head token = 3DHP head (no nose); 2D = exact projection (median 0.004 px)"},
        "splits": {"rule": "VAL videos -> TRAIN; TEST groups (numeric suffix stripped) sorted by sha256(salt+group); "
                           "index%3==0 -> CALIB else FINAL", "salt": SPLIT_SALT,
                   "known_limits": "3DPW performers recur across sequences without identity labels: subject-level "
                                   "disjointness cannot be proven; downtown scenes recur across VAL/TEST"},
        "features": {"allowlist": ["geometry(17x4: crop x,y, RAW score unclipped, validity)",
                                   "S0 prediction root-relative (17x3)", "orthographic reprojection residual (17)+scale",
                                   "predicted bone lengths (16)", "invalid limb count", "valid score mean/max/min"],
                     "excluded": ["GT", "sample/video/actor identity", "timestamps", "split", "benchmark strata",
                                  "image size / crop location", "future frames", "frozen hidden tokens"],
                     "temporal_scope": "none (single-frame, online-compatible)",
                     "dim": RB.FEATURE_DIM, "layout": RB.FEATURE_LAYOUT,
                     "normalization": "TRAIN mean/std (std floor 1e-6)"},
        "assessor": cfg.to_dict(),
        "baselines": {"A_constant_per_joint": "TRAIN per-joint log-error quantiles",
                      "B_raw_score_validity": "TRAIN quantiles per (joint, own validity, own raw-score TRAIN decile); "
                                              "fallback (joint,validity)->(joint); scores>1 kept in open top bin",
                      "B2_raw_score_validity_missing_count": "B plus frame invalid-limb count clipped at 3",
                      "C_reprojection_consistency": "TRAIN quantiles per (joint, validity, reprojection residual decile)",
                      "temporal_heuristic": "UNAVAILABLE: S0 is single-frame and the 3DPW bank cadence (10 Hz, sparse) "
                                            "differs from 30 fps real clips",
                      f"min_cell": RB.MIN_CELL, "calibration": "identical split-conformal shift on CALIB for every predictor"},
        "calibration": "pooled-over-joints split conformal per quantile in log(e+1mm) space on CALIB (M regime)",
        "decision_task": {"type": "TRIAGE UTILITY (fixed review budget)",
                          "rule": f"per video flag top ceil({RB.REVIEW_BUDGET}*n) supported joint-cells by q90; ties by "
                                  "blake2b(sample_id|joint)", "metrics": ["recall of material errors",
                                  "excess-error capture", "false-negative rate", "precision"],
                          "secondary": f"selective acceptance of lowest-q90 {RB.ACCEPT_FRACTION:.0%} cells",
                          "not": "not Generator decision utility, not animation improvement"},
        "metrics": {"ranking": "pooled Spearman(q50,e) over FINAL M cells; AP(q90, e>150mm) with tied blocks; "
                               "within-joint mean Spearman descriptive (500-rep bootstrap)",
                    "calibration": "q90/q50 coverage marginal, by FINAL q90 quintile, by joint group, invalid inputs",
                    "false_reassurance": "material error with q90 < 150 mm, / material errors",
                    "bootstrap": "video-cluster (subject-cluster for MPI) 2000 reps, seed 1337; paired differences "
                                 "learned - each baseline"},
        "gates": {
            "validity": ["V1 frozen checkpoint sha before/after", "V2 S0 B/M reproduction byte-identical to W102",
                         "V3 GT row contract (rows.json G == bank target <=1e-6 m) and W102 MPJPE parity",
                         "V4 split disjoint by video and group; no image crosses partitions",
                         "V5 feature allowlist refuses extra keys; GT perturbation leaves assessor output identical",
                         "V6 outputs finite, nonnegative, q90>=q50, metres",
                         "V7 assessor training repeatable across two processes (byte-identical FINAL outputs)",
                         "V8 metric unit tests pass", "V9 FINAL not read before calibration frozen",
                         "V10 real-clip arrays byte-equal W121/W122"],
            "P1": "FINAL-M Spearman: learned - each baseline, video-bootstrap CI lower > 0",
            "P2": "FINAL-M AP(material): learned - each baseline, CI lower > 0",
            "P3": "FINAL-M q90 coverage in [0.85,0.95]; every q90-quintile and the invalid-input stratum in [0.80,0.97]",
            "P4": "FINAL-M triage recall@10%: learned - each baseline, CI lower > 0",
            "P5": "FINAL-M false-reassurance rate <= 0.25 overall and for arm joints with invalid input",
            "P6": "MPI (independent subjects, oracle_geometry, hip-midpoint rooted, 12 limb joints): within-joint "
                  "mean Spearman(q50,e) learned - C and learned - B2, subject-bootstrap (8 subjects) CI lower > 0. "
                  "MPI coverage/calibration is reported DESCRIPTIVELY, not gated: the preflight GT audit shows SMPL "
                  "vs 3DHP joint conventions differ (median hip width 0.12 vs 0.25 m, thigh 0.38 vs 0.48 m), so "
                  "metric error on MPI contains a convention offset that is not frozen-model error"},
        "case_rule": {"D": "any V gate fails (supersedes all)",
                      "A": "P1..P6 all pass", "B": "not A and at least one of P1, P2, P4 passes",
                      "C": "none of P1, P2, P4 passes (calibration alone is achievable by baselines)",
                      "DATA_BLOCKED": "not applicable once preflight passes"},
        "real_video": {"clips": REAL_CLIPS, "source": "W122 re-executed arrays (byte-equal W121)",
                       "inspection": S1_EVENT,
                       "predeclared_unstable": "per clip, 5 frames with largest S0 per-frame root-relative wrist "
                                               "displacement (either wrist); flags: q90 >= 150 mm and clip-top-10% q90",
                       "non_claims": "GT 3D error UNKNOWN; high uncertainty at f251 does not prove error"},
        "review_selection": {"well_observed_correct": "FINAL M cell, all 12 limbs valid, lowest q90 quintile, smallest e",
                             "high_confidence_wrong": "top 5 e among FINAL M cells with q90 < 150 mm",
                             "missing_or_occluded": "top 5 q90 among invalid-input cells",
                             "fast_valid_motion": "top 5 GT speed cells with e <= 100 mm",
                             "score_gt1": "5 highest raw scores",
                             "false_positive": "top 5 q90 with e < 50 mm",
                             "domain_failure": "top 5 MPI e among q90 < 150 mm"},
        "strata": ["input invalid", "benchmark-invisible proxy (bank target_valid False)", "raw score > 1",
                   "near-straight predicted hinge < 15 deg", "high GT joint speed (TRAIN top quartile, dt<=0.2s)",
                   "facing (bank strata)", "posture (W102 rows)", "joint group", "video"],
    }


def stage_declare(args):
    out = fresh(RESULTS / "declare")
    pre = load_json(WS / "results/preflight/preflight.json")
    if not pre["pass"]:
        raise RuntimeError("STOP: preflight failed")
    decl = declaration_content(RB.sha_file(WS / "results/preflight/preflight.json"))
    print("declaration", RB.dump(out / "declaration.json", decl))


EVAL_STAGE = False


def build_regime(pop, regime, predictions):
    geometry = s0_geometry(pop, regime)
    rows = np.ones(len(pop["positions"]), dtype=bool) if regime == "B" else pop["identity"] == "same_person"
    split = pop["w123_split"][rows].copy()
    if SMOKE and EVAL_STAGE:
        split = np.where(split == "TRAIN", "FINAL", np.where(split == "FINAL", "EXCLUDED", split))
    return {"rows": np.flatnonzero(rows), "geometry": geometry[rows], "prediction": predictions[rows],
            "target": pop["target"][rows], "split": split, "video": pop["video"][rows],
            "sample_id": pop["sample_id"][rows]}


def predictors_for(bundle, baselines, geometry, prediction):
    feats = RB.runtime_features({"geometry": geometry, "prediction": prediction})
    residual, _ = RB.reprojection_residual(geometry, prediction)
    out = {"LEARNED": RB.assessor_predict(bundle, feats)}
    for name, b in baselines.items():
        out[name] = b.predict(geometry, residual)
    return out


def stage_train(args):
    out = fresh(RESULTS / f"train_{args.tag}")
    if not (RESULTS / "declare/declaration.json").exists():
        raise RuntimeError("STOP: declaration required before training")
    pop = load_population()
    pred_m = load_w102_predictions(pop, "M")
    M = build_regime(pop, "M", pred_m)
    tr = M["split"] == "TRAIN"
    ca = M["split"] == "CALIB"
    feats = RB.runtime_features({"geometry": M["geometry"][tr], "prediction": M["prediction"][tr]})
    y_tr = RB.log_err(RB.joint_errors(M["prediction"][tr], M["target"][tr]))
    t0 = time.time()
    bundle = RB.train_assessor(feats, y_tr, RB.AssessorConfig(epochs=2, seeds=(1,)) if SMOKE else RB.AssessorConfig())
    train_s = time.time() - t0
    import torch
    torch.save({"mean": bundle["mean"], "std": bundle["std"], "states": bundle["states"], "in_dim": bundle["in_dim"],
                "config": bundle["config"].to_dict()}, out / "assessor.pt")
    baselines = RB.make_baselines()
    res_tr, _ = RB.reprojection_residual(M["geometry"][tr], M["prediction"][tr])
    for b in baselines.values():
        b.fit(M["geometry"][tr], res_tr, y_tr)
    import pickle
    (out / "baselines.pkl").write_bytes(pickle.dumps({k: {"tables": v.tables, "edges": v.edges} for k, v in baselines.items()}))
    y_ca = RB.log_err(RB.joint_errors(M["prediction"][ca], M["target"][ca]))
    raw_ca = predictors_for(bundle, baselines, M["geometry"][ca], M["prediction"][ca])
    shifts = {k: RB.conformal_shift(v, y_ca) for k, v in raw_ca.items()}
    raw_tr = predictors_for(bundle, baselines, M["geometry"][tr], M["prediction"][tr])
    e_tr, e_ca = np.exp(y_tr) - RB.EPS_M, np.exp(y_ca) - RB.EPS_M
    dev = {}
    for k in raw_ca:
        qtr = RB.apply_calibration(raw_tr[k], shifts[k])
        qca = RB.apply_calibration(raw_ca[k], shifts[k])
        dev[k] = {"TRAIN_in_sample": {"coverage_q90": RB.coverage(e_tr.ravel(), qtr[..., 1].ravel()),
                                      "coverage_q50": RB.coverage(e_tr.ravel(), qtr[..., 0].ravel()),
                                      "spearman_q50": RB.spearman(qtr[..., 0].ravel(), e_tr.ravel())},
                  "CALIB_after_shift": {"coverage_q90": RB.coverage(e_ca.ravel(), qca[..., 1].ravel()),
                                        "coverage_q50": RB.coverage(e_ca.ravel(), qca[..., 0].ravel()),
                                        "spearman_q50": RB.spearman(qca[..., 0].ravel(), e_ca.ravel())}}
    calib = {"schema": "w123_calibration_v1", "shifts_log_space": shifts, "calibration_rows": int(ca.sum()),
             "calibration_cells": int(ca.sum() * 16), "calibration_videos": sorted(set(M["video"][ca]))}
    calib_sha = RB.dump(out / "calibration.json", calib)
    report = {"schema": "w123_train_v1", "tag": args.tag, "train_rows": int(tr.sum()), "train_cells": int(tr.sum() * 16),
              "train_videos": sorted(set(M["video"][tr])), "train_seconds": train_s,
              "final_curves_last_epoch": [c[-1] for c in bundle["curves"]], "curves": bundle["curves"],
              "assessor_weights_sha256": RB.assessor_weights_sha(bundle), "assessor_file_sha256": RB.sha_file(out / "assessor.pt"),
              "calibration_sha256": calib_sha, "development_metrics": dev,
              "final_gt_rows_read": 0, "torch": torch.__version__, "deterministic": True}
    # repeatability fingerprint: FINAL-feature outputs are computed but NOT compared with GT here
    fi = M["split"] == "FINAL"
    raw_fi = RB.assessor_predict(bundle, RB.runtime_features({"geometry": M["geometry"][fi], "prediction": M["prediction"][fi]}))
    report["final_output_fingerprint_no_gt"] = RB.sha_array(raw_fi.astype(np.float64))
    print("train", args.tag, report["assessor_weights_sha256"], RB.dump(out / "train_report.json", report))


def load_trained(tag="r1"):
    import pickle
    import torch
    d = RESULTS / f"train_{tag}"
    raw = torch.load(d / "assessor.pt", map_location="cpu", weights_only=False)
    cfg = RB.AssessorConfig(**{k: (tuple(v) if k == "seeds" else v) for k, v in raw["config"].items()
                               if k in RB.AssessorConfig.__dataclass_fields__})
    bundle = {"mean": raw["mean"], "std": raw["std"], "states": raw["states"], "in_dim": raw["in_dim"], "config": cfg}
    tables = pickle.loads((d / "baselines.pkl").read_bytes())
    baselines = RB.make_baselines()
    for k, b in baselines.items():
        b.tables, b.edges = tables[k]["tables"], tables[k]["edges"]
    calib = load_json(d / "calibration.json")
    return bundle, baselines, calib, d


def cells(e, q, video_rows, sample_rows):
    """Flatten (N,16) to cells with per-cell video and label-free tie-break ids."""
    n = e.shape[0]
    vid = np.repeat(video_rows, 16)
    cid = [f"{s}|{j}" for s in sample_rows for j in RB.TARGET_NAMES]
    return e.ravel(), {k: (v[..., 0].ravel(), v[..., 1].ravel()) for k, v in q.items()}, vid, RB.tiebreak_keys(cid)


def joint_ids(n_rows, n_cols=16):
    return np.tile(np.arange(n_cols), n_rows)


def compare_block(e, preds, vid, tb, joint, with_bootstrap=True):
    """Endpoint metrics for every predictor plus learned-minus-baseline video-bootstrap differences."""
    res = {k: RB.endpoint_set(e, q50, q90, vid, tb, joint) for k, (q50, q90) in preds.items()}
    if not with_bootstrap:
        return res, {}
    mat = e > RB.MATERIAL_M
    flags = {k: RB.triage_flags(q90, vid, tb) for k, (q50, q90) in preds.items()}
    diffs = {}
    L50, L90 = preds["LEARNED"]
    for k, (q50, q90) in preds.items():
        if k == "LEARNED":
            continue
        diffs[k] = {
            "spearman": RB.cluster_bootstrap(vid, lambda i: RB.spearman(L50[i], e[i]) - RB.spearman(q50[i], e[i])),
            "ap_material": RB.cluster_bootstrap(vid, lambda i: RB.average_precision(L90[i], mat[i]) -
                                                RB.average_precision(q90[i], mat[i])),
            "triage_recall": RB.cluster_bootstrap(vid, lambda i: ((flags["LEARNED"][i] & mat[i]).sum() -
                                                                  (flags[k][i] & mat[i]).sum()) / max(mat[i].sum(), 1)),
            "spearman_within_joint_descriptive": RB.cluster_bootstrap(
                vid, lambda i: RB.within_joint_spearman(L50[i], e[i], joint[i]) -
                RB.within_joint_spearman(q50[i], e[i], joint[i]), reps=500),
        }
    return res, diffs


def stage_evaluate(args):
    global EVAL_STAGE
    EVAL_STAGE = True
    out = fresh(RESULTS / "evaluate")
    bundle, baselines, calib, tdir = load_trained("r1")
    r2 = load_json(RESULTS / "train_r2/train_report.json")
    r1 = load_json(tdir / "train_report.json")
    pop = load_population()
    report = {"schema": "w123_evaluate_v1",
              "repeatability": {"weights_equal": r1["assessor_weights_sha256"] == r2["assessor_weights_sha256"],
                                "final_fingerprint_equal": r1["final_output_fingerprint_no_gt"] == r2["final_output_fingerprint_no_gt"],
                                "calibration_equal": r1["calibration_sha256"] == r2["calibration_sha256"]}}
    shifts = calib["shifts_log_space"]
    for regime in ("M", "B"):
        preds_all = load_w102_predictions(pop, regime)
        R = build_regime(pop, regime, preds_all)
        fi = R["split"] == "FINAL"
        g, p = R["geometry"][fi], R["prediction"][fi]
        raw = predictors_for(bundle, baselines, g, p)
        # V5: GT perturbation cannot change the output (GT is never an input); allowlist refusal
        feats = RB.runtime_features({"geometry": g, "prediction": p})
        try:
            RB.runtime_features({"geometry": g, "prediction": p, "target": R["target"][fi]})
            allow_refused = False
        except ValueError:
            allow_refused = True
        q = {k: RB.apply_calibration(v, shifts[k]) for k, v in raw.items()}
        e2 = RB.joint_errors(p, R["target"][fi])
        e, flat, vid, tb = cells(e2, q, R["video"][fi], R["sample_id"][fi])
        jid = joint_ids(int(fi.sum()))
        block, diffs = compare_block(e, flat, vid, tb, jid, with_bootstrap=(regime == "M"))
        reg = {"rows": int(fi.sum()), "cells": int(e.size), "videos": sorted(set(R["video"][fi])),
               "endpoints": block, "learned_minus_baseline": diffs,
               "allowlist_refuses_target": allow_refused,
               "outputs_valid": bool(all(np.isfinite(v).all() and (v >= 0).all() and (v[..., 1] >= v[..., 0]).all()
                                         for v in q.values())),
               "error_distribution_mm": {"mean": float(e.mean() * 1000), "median": float(np.median(e) * 1000),
                                         "p90": float(np.quantile(e, 0.9) * 1000), "max": float(e.max() * 1000)},
               "prediction_distribution_q90_mm": {k: {"min": float(v[1].min() * 1000), "median": float(np.median(v[1]) * 1000),
                                                      "max": float(v[1].max() * 1000)} for k, v in flat.items()}}
        # calibration detail for every predictor
        valid = (g[:, list(RB.TARGET_JOINTS), 3] > 0.5).ravel()
        score = g[:, list(RB.TARGET_JOINTS), 2].ravel()
        reg["calibration_bins"] = {k: RB.binned_calibration(e, v[0], v[1]) for k, v in flat.items()}
        # strata
        bend = RB.hinge_bend_deg(p)
        near = np.zeros(e2.shape, dtype=bool)
        for jj, j in enumerate(RB.TARGET_JOINTS):
            if j in RB.CELL_HINGE:
                near[:, jj] = bend[RB.CELL_HINGE[j]] < RB.NEAR_STRAIGHT_DEG
        bench_vis = pop["bench_visible"][R["rows"][fi]][:, list(RB.TARGET_JOINTS)].ravel()
        speed = gt_speed(pop, R["rows"][fi])
        thr = speed_threshold(pop, R)
        facing = np.repeat([pop["bank"].samples[pop["positions"][r]].strata.get("facing", "unknown") for r in R["rows"][fi]], 16)
        jgroup = np.tile(["leg" if j in RB.LEG_JOINTS else "arm" if j in RB.ARM_JOINTS else "torso_head"
                          for j in RB.TARGET_JOINTS], int(fi.sum()))
        jname = np.tile(RB.TARGET_NAMES, int(fi.sum()))
        masks = {"input_invalid": ~valid, "input_valid": valid, "score_gt1": valid & (score > 1.0),
                 "benchmark_invisible_proxy": ~bench_vis, "near_straight": near.ravel(),
                 "high_speed": np.isfinite(speed) & (speed >= thr), "speed_defined": np.isfinite(speed),
                 "arm_invalid": (jgroup == "arm") & ~valid}
        for f in sorted(set(facing)):
            masks["facing_" + f] = facing == f
        for jg in ("leg", "arm", "torso_head"):
            masks["group_" + jg] = jgroup == jg
        for jn in RB.TARGET_NAMES:
            masks["joint_" + jn] = jname == jn
        reg["strata"] = {name: {k: RB.strata_metrics(e[m], v[0][m], v[1][m]) for k, v in flat.items()}
                         for name, m in masks.items()}
        reg["speed_threshold_m_per_s_from_TRAIN"] = thr
        reg["score_domain"] = {"min_valid": float(score[valid].min()), "max_valid": float(score[valid].max()),
                               "n_gt1": int((valid & (score > 1)).sum())}
        per_video = {}
        for v in sorted(set(vid)):
            m = vid == v
            per_video[v] = {k: RB.endpoint_set(e[m], fl[0][m], fl[1][m], vid[m], tb[m], jid[m]) for k, fl in flat.items()
                            if k in ("LEARNED", "B2_raw_score_validity_missing_count", "A_constant_per_joint",
                                     "C_reprojection_consistency", "B_raw_score_validity")}
        reg["per_video"] = per_video
        if regime == "M":
            # B-recalibrated view is computed in the B block; M also gets review selections
            reg["review"] = review_selections(e, flat["LEARNED"], valid, score, speed, R, fi, masks)
        else:
            # secondary: recalibrate on B CALIB (no retraining)
            ca = R["split"] == "CALIB"
            raw_ca = predictors_for(bundle, baselines, R["geometry"][ca], R["prediction"][ca])
            y_ca = RB.log_err(RB.joint_errors(R["prediction"][ca], R["target"][ca]))
            sh_b = {k: RB.conformal_shift(v, y_ca) for k, v in raw_ca.items()}
            qb = {k: RB.apply_calibration(v, sh_b[k]) for k, v in raw.items()}
            _, flb, _, _ = cells(e2, qb, R["video"][fi], R["sample_id"][fi])
            reg["B_recalibrated_secondary"] = {k: {"coverage_q90": RB.coverage(e, v[1]), "coverage_q50": RB.coverage(e, v[0]),
                                                   "false_reassurance_rate": RB.endpoint_set(e, v[0], v[1], vid, tb)["false_reassurance_rate"]}
                                               for k, v in flb.items()}
            reg["B_shifts"] = sh_b
        report[regime] = reg
        np.savez_compressed(out / f"final_{regime}_cells.npz", e=e, video=vid,
                            **{f"{k}_q50": v[0] for k, v in flat.items()}, **{f"{k}_q90": v[1] for k, v in flat.items()})
    # training-domain (in-sample) calibration, separate
    report["training_domain_in_sample"] = r1["development_metrics"]
    report["checkpoint_sha256_after"] = RB.sha_file(CHECKPOINT)
    print("evaluate", RB.dump(out / "evaluate.json", report))


def gt_speed(pop, rows):
    """Evaluation-only GT joint speed (m/s) to the previous sample of the same actor if dt <= 0.2 s; (n*16,)."""
    sid = pop["sample_id"]
    actor = pop["actor"]
    ts = pop["timestamp"]
    out = np.full((len(rows), 16), np.nan)
    index = {s: i for i, s in enumerate(sid)}
    for k, r in enumerate(rows):
        prev = pop["bank"].samples[pop["positions"][r]].neighbors.get("previous")
        if prev is None or prev not in index:
            continue
        q = index[prev]
        dt = ts[r] - ts[q]
        if actor[q] != actor[r] or not (0 < dt <= 0.2):
            continue
        g1 = pop["target"][r] - pop["target"][r][:1]
        g0 = pop["target"][q] - pop["target"][q][:1]
        out[k] = np.linalg.norm(g1 - g0, axis=-1)[list(RB.TARGET_JOINTS)] / dt
    return out.ravel()


def speed_threshold(pop, R):
    tr = R["rows"][pop["w123_split"][R["rows"]] == "TRAIN"]   # always the real TRAIN partition
    s = gt_speed(pop, tr)
    return float(np.nanquantile(s, 0.75))


def review_selections(e, learned, valid, score, speed, R, fi, masks):
    q50, q90 = learned
    sid = np.repeat(R["sample_id"][fi], 16)
    jn = np.tile(RB.TARGET_NAMES, int(fi.sum()))

    def pick(mask, key, n=5, descending=True):
        idx = np.flatnonzero(mask)
        if not len(idx):
            return []
        order = idx[np.argsort(-key[idx] if descending else key[idx], kind="mergesort")][:n]
        return [{"sample_id": str(sid[i]), "joint": str(jn[i]), "err_mm": float(e[i] * 1000),
                 "q50_mm": float(q50[i] * 1000), "q90_mm": float(q90[i] * 1000), "input_valid": bool(valid[i]),
                 "raw_score": float(score[i]), "gt_speed_m_s": None if not np.isfinite(speed[i]) else float(speed[i])}
                for i in order]
    allvalid_rows = np.repeat(R["geometry"][fi][:, list(RB.LIMB_JOINTS), 3].min(1) > 0.5, 16)
    low = q90 <= np.quantile(q90, 0.2)
    return {"well_observed_correct": pick(allvalid_rows & low, e, 3, descending=False),
            "high_confidence_wrong": pick(q90 < RB.MATERIAL_M, e),
            "missing_or_occluded": pick(~valid, q90),
            "fast_valid_motion": pick(np.isfinite(speed) & (e <= 0.1), np.nan_to_num(speed, nan=-1)),
            "score_gt1": pick(valid, score),
            "false_positive": pick(e < 0.05, q90)}


def stage_mpi(args):
    out = fresh(RESULTS / "mpi")
    if not (RESULTS / "evaluate/evaluate.json").exists():
        raise RuntimeError("STOP: run evaluate first (calibration frozen)")
    bundle, baselines, calib, _ = load_trained("r1")
    z = np.load(WS / "results/preflight/mpi_population.npz", allow_pickle=False)
    pop = {k: z[k] for k in z.files}
    if SMOKE:   # synthetic targets only; real MPI errors are never computed in smoke mode
        keep = np.arange(0, len(pop["frame"]), 20)
        pop = {k: v[keep] for k, v in pop.items()}
    geometry = mpi_geometry(pop)
    t0 = time.time()
    pred = s0_forward(geometry)
    pred2 = s0_forward(geometry)
    if SMOKE:
        pop["target"] = (pred + np.random.default_rng(0).normal(0, 0.05, pred.shape)).astype(np.float32)
    np.save(out / "mpi_s0_prediction.npy", pred)
    raw = predictors_for(bundle, baselines, geometry, pred)
    q = {k: RB.apply_calibration(v, calib["shifts_log_space"][k]) for k, v in raw.items()}
    limb_cols = [RB.TARGET_JOINTS.index(j) for j in RB.LIMB_JOINTS]
    report = {"schema": "w123_mpi_v1", "regime": "oracle_geometry", "rows": int(len(geometry)),
              "s0_repeat_bytes_equal": bool(pred.tobytes() == pred2.tobytes()), "s0_seconds": time.time() - t0,
              "prediction_sha256": RB.sha_file(out / "mpi_s0_prediction.npy"),
              "input_score_domain": "constant 1.0 for in-image joints (oracle convention, declared)", "variants": {}}
    subj = pop["subject"]
    for root in ("hip_mid", "pelvis"):
        e2 = RB.joint_errors(pred, pop["target"], root=root)[:, limb_cols]
        qq = {k: v[:, limb_cols] for k, v in q.items()}
        n = e2.shape[0]
        e = e2.ravel()
        flat = {k: (v[..., 0].ravel(), v[..., 1].ravel()) for k, v in qq.items()}
        cid = [f"{s}/{sq}/c{c}/f{f}|{RB.JOINT_NAMES[j]}" for s, sq, c, f in
               zip(pop["subject"], pop["sequence"], pop["camera"], pop["frame"]) for j in RB.LIMB_JOINTS]
        tb = RB.tiebreak_keys(cid)
        clus = np.repeat(subj, len(limb_cols))
        vidkey = np.repeat([f"{s}/{sq}/c{c}" for s, sq, c in zip(pop["subject"], pop["sequence"], pop["camera"])], len(limb_cols))
        jid = joint_ids(n, len(limb_cols))
        block = {k: RB.endpoint_set(e, a, b, vidkey, tb, jid) for k, (a, b) in flat.items()}
        L50, L90 = flat["LEARNED"]
        diffs = {}
        mat = e > RB.MATERIAL_M
        for k in ("A_constant_per_joint", "C_reprojection_consistency", "B2_raw_score_validity_missing_count",
                  "B_raw_score_validity"):
            a50, a90 = flat[k]
            diffs[k] = {"spearman_within_joint": RB.cluster_bootstrap(
                            clus, lambda i: RB.within_joint_spearman(L50[i], e[i], jid[i]) -
                            RB.within_joint_spearman(a50[i], e[i], jid[i])),
                        "spearman": RB.cluster_bootstrap(clus, lambda i: RB.spearman(L50[i], e[i]) - RB.spearman(a50[i], e[i])),
                        "ap_material": RB.cluster_bootstrap(clus, lambda i: RB.average_precision(L90[i], mat[i]) -
                                                            RB.average_precision(a90[i], mat[i]))}
        per_subject = {s: {k: RB.strata_metrics(e[clus == s], v[0][clus == s], v[1][clus == s])
                           for k, v in flat.items() if k in ("LEARNED", "A_constant_per_joint", "C_reprojection_consistency")}
                       for s in sorted(set(subj))}
        valid = (geometry[:, list(RB.LIMB_JOINTS), 3] > 0.5).ravel()
        report["variants"][root] = {"cells": int(e.size), "endpoints": block, "learned_minus_baseline_subject_bootstrap": diffs,
                                    "per_subject": per_subject,
                                    "calibration_bins_learned": RB.binned_calibration(e, L50, L90),
                                    "strata_learned": {"input_invalid": RB.strata_metrics(e[~valid], L50[~valid], L90[~valid]),
                                                       "input_valid": RB.strata_metrics(e[valid], L50[valid], L90[valid])},
                                    "error_distribution_mm": {"mean": float(e.mean() * 1000), "median": float(np.median(e) * 1000),
                                                              "p90": float(np.quantile(e, 0.9) * 1000)}}
        if root == "hip_mid":
            idx = np.flatnonzero(L90 < RB.MATERIAL_M)
            order = idx[np.argsort(-e[idx], kind="mergesort")][:5]
            report["domain_failure_review"] = [{"cell": cid[i], "err_mm": float(e[i] * 1000), "q90_mm": float(L90[i] * 1000),
                                                "q50_mm": float(L50[i] * 1000)} for i in order]
    report["checkpoint_sha256_after"] = RB.sha_file(CHECKPOINT)
    print("mpi", RB.dump(out / "mpi.json", report))


def stage_realvideo(args):
    out = fresh(RESULTS / "realvideo")
    bundle, baselines, calib, tdir = load_trained("r1")
    r1 = load_json(tdir / "train_report.json")
    assessor_id = r1["assessor_weights_sha256"]
    calibration_id = r1["calibration_sha256"]
    report = {"schema": "w123_realvideo_v1", "regime": "real_animcv_observation (GT-FREE)", "assessor_id": assessor_id,
              "calibration_id": calibration_id, "clips": {}}
    for cid, name in REAL_CLIPS.items():
        path = W122 / "results/reexec_perception" / cid / "observations_predictions.npz"
        z = np.load(path, allow_pickle=False)
        g, p = z["geometry"].astype(np.float32), z["predictions"].astype(np.float32)
        raw = predictors_for(bundle, baselines, g, p)
        q = {k: RB.apply_calibration(v, calib["shifts_log_space"][k]) for k, v in raw.items()}
        q50, q90 = q["LEARNED"][..., 0], q["LEARNED"][..., 1]
        clip_report = load_json(path.parent / "report.json")
        side = RB.build_sidecar(source={"clip_id": cid, "clip_file": name, "clip_sha256": clip_report.get("clip_sha256"),
                                        "observation_npz_sha256": RB.sha_file(path), "fps": 30.0},
                                frame_index=z["frame_index"], timestamp=z["timestamp"], geometry=g, prediction=p,
                                q50=q50, q90=q90, checkpoint_sha256=CHECKPOINT_SHA, assessor_id=assessor_id,
                                calibration_id=calibration_id, observation_regime="real_animcv_observation")
        errs = RB.validate_sidecar(side, checkpoint_sha256=CHECKPOINT_SHA, prediction=p, frame_index=z["frame_index"],
                                   geometry=g, assessor_id=assessor_id, calibration_id=calibration_id)
        sha = RB.dump(out / f"sidecar_{cid}.json", side)
        rr = p - p[:, :1]
        disp = np.zeros((len(p), 17))
        disp[1:] = np.linalg.norm(rr[1:] - rr[:-1], axis=-1)
        valid = g[:, list(RB.TARGET_JOINTS), 3] > 0.5
        score = g[:, list(RB.TARGET_JOINTS), 2]
        top10 = q90 >= np.quantile(q90, 0.9)
        absflag = q90 >= RB.MATERIAL_M
        wrist = np.maximum(disp[:, 13], disp[:, 16])
        unstable = np.argsort(-wrist, kind="mergesort")[:5]
        jw = [RB.TARGET_JOINTS.index(13), RB.TARGET_JOINTS.index(16)]
        rest = np.setdiff1d(np.arange(len(p)), unstable)
        cr = {"frames": int(len(p)), "sidecar_sha256": sha, "sidecar_violations": errs,
              "invalid_input_cells": int((~valid).sum()), "score_gt1_cells": int((valid & (score > 1)).sum()),
              "score_max": float(score[valid].max()),
              "q90_mm": {"min": float(q90.min() * 1000), "median": float(np.median(q90) * 1000), "p90": float(np.quantile(q90, 0.9) * 1000),
                         "max": float(q90.max() * 1000)},
              "abs_flag_rate_all_cells": float(absflag.mean()),
              "abs_flag_rate_by_joint": {RB.TARGET_NAMES[j]: float(absflag[:, j].mean()) for j in range(16)},
              "invalid_vs_valid_median_q90_mm": [float(np.median(q90[~valid]) * 1000) if (~valid).any() else None,
                                                 float(np.median(q90[valid]) * 1000)],
              "predeclared_unstable_frames": [{"frame": int(z["frame_index"][f]), "wrist_disp_mm": float(wrist[f] * 1000),
                                               "left_wrist_q90_mm": float(q90[f, jw[0]] * 1000), "right_wrist_q90_mm": float(q90[f, jw[1]] * 1000),
                                               "left_wrist_valid": bool(valid[f, jw[0]]), "right_wrist_valid": bool(valid[f, jw[1]]),
                                               "left_wrist_score": float(score[f, jw[0]]), "right_wrist_score": float(score[f, jw[1]]),
                                               "wrist_abs_flag": bool(absflag[f, jw].any()), "wrist_top10_flag": bool(top10[f, jw].any())}
                                              for f in unstable],
              "wrist_flag_rate_unstable_vs_other": {"abs": [float(absflag[unstable][:, jw].any(1).mean()), float(absflag[rest][:, jw].any(1).mean())],
                                                    "top10": [float(top10[unstable][:, jw].any(1).mean()), float(top10[rest][:, jw].any(1).mean())]},
              "spearman_wrist_disp_vs_wrist_q90": RB.spearman(np.concatenate([disp[1:, 13], disp[1:, 16]]),
                                                              np.concatenate([q90[1:, jw[0]], q90[1:, jw[1]]])),
              "baseline_abs_flag_rate": {k: float((v[..., 1] >= RB.MATERIAL_M).mean()) for k, v in q.items()}}
        if cid == S1_EVENT["clip"]:
            fidx = list(z["frame_index"])
            rows = []
            for f in S1_EVENT["frames"]:
                i = fidx.index(f)
                row = {"frame": f}
                for jn in S1_EVENT["joints"]:
                    j = RB.JOINT_NAMES.index(jn)
                    jj = RB.TARGET_JOINTS.index(j)
                    row[jn] = {"raw_score": float(g[i, j, 2]), "valid": bool(g[i, j, 3] > 0.5), "q50_mm": float(q50[i, jj] * 1000),
                               "q90_mm": float(q90[i, jj] * 1000), "s0_disp_mm": float(disp[i, j] * 1000),
                               "clip_q90_percentile": float((q90[:, jj] <= q90[i, jj]).mean()),
                               "baselines_q90_mm": {k: float(v[i, jj, 1] * 1000) for k, v in q.items() if k != "LEARNED"}}
                rows.append(row)
            cr["s1_event_inspection"] = rows
        report["clips"][cid] = cr
        np.savez_compressed(out / f"{cid}_reliability.npz", q50=q50, q90=q90, disp=disp, frame_index=z["frame_index"])
    print("realvideo", RB.dump(out / "realvideo.json", report))


def _ci_pos(d):
    return d.get("ci95") is not None and d["ci95"][0] > 0


def stage_decide(args):
    out = fresh(RESULTS / "decision")
    pre = load_json(WS / "results/preflight/preflight.json")
    ev = load_json(RESULTS / "evaluate/evaluate.json")
    mpi = load_json(RESULTS / "mpi/mpi.json")
    rv = load_json(RESULTS / "realvideo/realvideo.json")
    tests = load_json(RESULTS / "tests/tests_summary.json") if (RESULTS / "tests/tests_summary.json").exists() else {"pass": False}
    r1 = load_json(RESULTS / "train_r1/train_report.json")
    M = ev["M"]
    V = {"V1": pre["checkpoint_ok"] and pre["checkpoint_sha256_after"] == CHECKPOINT_SHA and ev["checkpoint_sha256_after"] == CHECKPOINT_SHA
               and mpi["checkpoint_sha256_after"] == CHECKPOINT_SHA,
         "V2": all(v["reproduced_bytes_equal"] for r in pre["s0_reproduction"].values() for v in r["splits"].values()),
         "V3": pre["gt_row_contract"]["pass"],
         "V4": pre["split"]["disjoint_videos_groups"] and pre["split"]["images_crossing_partitions"] == 0,
         "V5": M["allowlist_refuses_target"] and ev["B"]["allowlist_refuses_target"],
         "V6": M["outputs_valid"] and ev["B"]["outputs_valid"],
         "V7": all(ev["repeatability"].values()),
         "V8": bool(tests.get("pass")),
         "V9": r1["final_gt_rows_read"] == 0,
         "V10": all(c["arrays_byte_equal_w121"] for c in pre["real_clips"].values()) and
                all(not c["sidecar_violations"] for c in rv["clips"].values())}
    diffs = M["learned_minus_baseline"]
    P = {"P1": all(_ci_pos(d["spearman"]) for d in diffs.values()),
         "P2": all(_ci_pos(d["ap_material"]) for d in diffs.values()),
         "P4": all(_ci_pos(d["triage_recall"]) for d in diffs.values())}
    L = M["endpoints"]["LEARNED"]
    bins = M["calibration_bins"]["LEARNED"]
    inv = M["strata"]["input_invalid"]["LEARNED"]
    P["P3"] = bool(0.85 <= L["coverage_q90"] <= 0.95 and all(0.80 <= b["coverage_q90"] <= 0.97 for b in bins if b["n"]) and
                   inv["n"] > 0 and 0.80 <= inv["coverage_q90"] <= 0.97)
    arm_inv = M["strata"]["arm_invalid"]["LEARNED"]
    P["P5"] = bool(L["false_reassurance_rate"] is not None and L["false_reassurance_rate"] <= 0.25 and
                   (arm_inv["false_reassurance_rate"] is None or arm_inv["false_reassurance_rate"] <= 0.25))
    mv = mpi["variants"]["hip_mid"]["learned_minus_baseline_subject_bootstrap"]
    P["P6"] = bool(_ci_pos(mv["C_reprojection_consistency"]["spearman_within_joint"]) and
                   _ci_pos(mv["B2_raw_score_validity_missing_count"]["spearman_within_joint"]))
    if not all(V.values()):
        case = "D"
    elif all(P.values()):
        case = "A"
    elif P["P1"] or P["P2"] or P["P4"]:
        case = "B"
    else:
        case = "C"
    decision = {"schema": "w123_decision_v1", "case": case, "validity": V, "gates": P,
                "declaration_sha256": RB.sha_file(RESULTS / "declare/declaration.json"),
                "inputs": {k: RB.sha_file((WS / "results" if k.startswith("preflight") else RESULTS) / k)
                           for k in ("preflight/preflight.json", "evaluate/evaluate.json",
                                                                       "mpi/mpi.json", "realvideo/realvideo.json")}}
    print("decision", case, RB.dump(out / "decision.json", decision))


def stage_tests(args):
    """Run the focused W123 tests plus focused frozen-model regressions and record the outcome."""
    import xml.etree.ElementTree as ET
    out = fresh(RESULTS / "tests")
    files = ["tests/test_w123_reliability_assessor.py", "tests/test_canonical_pose_parity.py"]
    junit = out / "junit.xml"
    cmd = [sys.executable, "-X", "utf8", "-m", "pytest", *files, "-q", "-p", "no:cacheprovider", f"--junitxml={junit}"]
    proc = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
    (out / "pytest.log").write_text(proc.stdout + proc.stderr, encoding="utf-8")
    root = ET.parse(junit).getroot()
    suite = root if root.tag == "testsuite" else root.find("testsuite")
    summary = {"command": " ".join(cmd[1:]), "returncode": proc.returncode, "tests": int(suite.get("tests")),
               "failures": int(suite.get("failures")), "errors": int(suite.get("errors")),
               "skipped": int(suite.get("skipped")), "junit_sha256": RB.sha_file(junit),
               "skipped_cases": [c.get("name") for c in suite.iter("testcase") if c.find("skipped") is not None]}
    summary["pass"] = proc.returncode == 0 and summary["failures"] == 0 and summary["errors"] == 0
    print("tests", summary["pass"], RB.dump(out / "tests_summary.json", summary))


BONES = [(j, RB.PARENT[j]) for j in range(1, 17)]


def stage_review(args):
    """Agent-review panels for the predeclared FINAL-M representative selections (evaluation-only artifact)."""
    import cv2
    out = fresh(RESULTS / "review")
    ev = load_json(RESULTS / "evaluate/evaluate.json")
    pop = load_population()
    index = {s: i for i, s in enumerate(pop["sample_id"])}
    image_root = W102 / "data/images"
    panels, manifest = [], []
    for group, items in ev["M"]["review"].items():
        for it in items[:4]:
            i = index[it["sample_id"]]
            rel = str(pop["side"]["image"][i])
            path = image_root / rel
            img = cv2.imread(str(path))
            if img is None:
                manifest.append({"group": group, **it, "image": rel, "status": "IMAGE_MISSING"})
                continue
            image_sha = RB.sha_file(path)
            w, h = pop["bank"].samples[pop["positions"][i]].image_size
            xy = pop["obs"]["input_2d_M"][i][:, :2] * np.array([w, h])
            valid = pop["obs"]["input_valid_M"][i]
            for a, b in BONES:
                if valid[a] and valid[b]:
                    cv2.line(img, tuple(int(v) for v in xy[a]), tuple(int(v) for v in xy[b]), (0, 255, 0), 3)
            j = RB.JOINT_NAMES.index(it["joint"])
            if valid[j]:
                cv2.circle(img, tuple(int(v) for v in xy[j]), 18, (0, 0, 255), 4)
            pts = xy[valid] if valid.any() else np.array([[w / 2, h / 2]])
            lo, hi = pts.min(0), pts.max(0)
            side = max(hi - lo) * 1.6 + 80
            c = (lo + hi) / 2
            x0, y0 = int(max(0, c[0] - side / 2)), int(max(0, c[1] - side / 2))
            crop = img[y0:int(min(h, c[1] + side / 2)), x0:int(min(w, c[0] + side / 2))]
            crop = cv2.resize(crop, (320, 320))
            canvas = np.full((400, 320, 3), 255, np.uint8)
            canvas[80:] = crop
            lines = [group, f"{it['sample_id'].split(':', 1)[1]}", f"{it['joint']} valid={it['input_valid']} s={it['raw_score']:.2f}",
                     f"err={it['err_mm']:.0f}mm q50={it['q50_mm']:.0f} q90={it['q90_mm']:.0f}"]
            for k, line in enumerate(lines):
                cv2.putText(canvas, line[:44], (4, 18 + 19 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
            panels.append(canvas)
            manifest.append({"group": group, **it, "image": rel, "image_sha256": image_sha,
                             "image_sha_matches_w102": image_sha == str(pop["side"]["image_sha256"][i])})
    rows = [np.concatenate(panels[k:k + 6] + [np.full((400, 320, 3), 255, np.uint8)] * (6 - len(panels[k:k + 6])), 1)
            for k in range(0, len(panels), 6)]
    cv2.imwrite(str(out / "review_panels.jpg"), np.concatenate(rows, 0), [cv2.IMWRITE_JPEG_QUALITY, 85])
    print("review", RB.dump(out / "review_manifest.json", {"panels": manifest,
                                                          "note": "AGENT REVIEW material; red circle = reviewed joint; "
                                                                  "green = frozen M 2D skeleton; err is evaluation-only GT"}))


def main():
    process_local_env()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", choices=("preflight", "declare", "train", "evaluate", "mpi", "realvideo", "decide", "review", "tests"))
    ap.add_argument("--tag", default="r1")
    args = ap.parse_args()
    {"preflight": stage_preflight, "declare": stage_declare, "train": stage_train, "evaluate": stage_evaluate,
     "mpi": stage_mpi, "realvideo": stage_realvideo, "decide": stage_decide, "review": stage_review, "tests": stage_tests}[args.stage](args)


if __name__ == "__main__":
    main()
