"""DIAGNOSTIC (Worklog 73): BEDLAM v1 paired RGB<->3D adapter for the Worklog 69 arm-depth target.

Source contract (verified, see docs/73):
  * labels: official processed BEDLAM training labels ``all_npz_12_training``
    (fields imgname, pose_cam (165), shape (11), trans_cam, cam_int, cam_ext,
    gtkps (127 SMPL-X joints projected)); validation scenes are processed
    from the official per-scene ground truth with the same SMPL-X convention;
  * 3D: neutral SMPL-X v1.1 (11 betas, flat hand mean, no PCA) evaluated with
    ``pose_cam``/``shape``/``trans_cam``; adding ``cam_ext[:3, 3]`` gives
    OpenCV camera coordinates (+x right, +y down, +z forward, metres) that
    reproject exactly onto ``gtkps`` through ``cam_int``;
  * joints: SMPL-X body joints 0..21 share the SMPL-24 body order, so the
    FrameBank target uses the 3DPW adapter's SMPL->canonical table
    (left_shoulder 16, left_elbow 18, left_wrist 20; subject's anatomical left);
  * canonical camera axes: AnimCV (X, Y, Z) = OpenCV (x, z, -y), pelvis
    subtracted -- identical to ``three_dpw_adapter._world_to_animcv_camera``;
  * "closeup" scenes store landscape PNGs whose labels describe the portrait
    view; the image is rotated 90 degrees clockwise before any use (official
    BEDLAM note: "you must rotate the image itself").

Split contract: the official BEDLAM scene split (four validation scenes) does
NOT separate people -- every body appears in both -- so it is intersected with
a deterministic subject split fixed before training: subject key = RenderPeople
person name (``rp_<name>``, all its scans) or the synthetic body id; a subject
is a validation subject iff sha256(key) mod 5 == 0.  External TRAIN keeps only
non-validation subjects from training scenes; external validation keeps only
validation subjects from validation scenes.  Person identity comes from the
label's ``sub`` field when present, otherwise from the label's SMPL-X betas,
which are unique per (person, motion) file of the official motion release.

Observation contract: GT never enters the model input.  The 2D input is the
re-run OpenPose COCO-18 detector (``openpose_coco18``) converted with the
exact 3DPW rules.  GT 2D is used only to decide WHICH detected person is the
labelled subject and whether that detection is the subject at all (>= 6 limb
joints within 0.1 of the GT limb-box diagonal), mirroring 3DPW's shipped
poses2D ("only provided if at least 6 joints were detected correctly").  As in
3DPW, ``target_valid`` = detector-visible AND GT-valid, and rows without a
valid pelvis are dropped.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from pose.pose_lifter import H36M_NAMES

LABEL_DIR_NAME = "all_npz_12_training"
# pixelite1201/BEDLAM@f1828fd train/core/config.py DATASET_FILES[0] (BEDLAM part).
SCENES_VALIDATION = ("20221019_3-8_250_highbmihand_orbit_stadium", "20221018_3_250_batch01hand_orbit_archVizUI3_time15",
                     "20221018_1_250_batch01hand_zoom_suburb_b", "20221018_3-8_250_batch01hand")
SMPLX_MODEL = {"model_type": "smplx", "gender": "neutral", "ext": "npz", "num_betas": 11, "flat_hand_mean": True,
               "use_pca": False}
# SMPL(-X) body index per FrameBank joint (three_dpw_adapter._SMPL_TO_CANONICAL, restated for import isolation).
SMPL_TO_CANONICAL = {"pelvis": 0, "left_hip": 1, "right_hip": 2, "spine": 3, "left_knee": 4, "right_knee": 5,
                     "thorax": 9, "left_ankle": 7, "right_ankle": 8, "neck": 12, "head": 15, "left_shoulder": 16,
                     "right_shoulder": 17, "left_elbow": 18, "right_elbow": 19, "left_wrist": 20, "right_wrist": 21}
CANONICAL_SMPLX = np.array([SMPL_TO_CANONICAL[n] for n in H36M_NAMES])
# OpenPose COCO-18 limb index -> SMPL-X joint (association only).
COCO_LIMB_TO_SMPLX = {2: 17, 3: 19, 4: 21, 5: 16, 6: 18, 7: 20, 8: 2, 9: 5, 10: 8, 11: 1, 12: 4, 13: 7}
ASSOCIATION = {"min_correct_limb_joints": 6, "tolerance_of_gt_limb_box_diagonal": 0.1}
ROWS_PER_SEQUENCE = 3
VALIDATION_SUBJECT_MODULUS = 5
BETA_MATCH_TOLERANCE = 1e-4


def scene_of(label_path: str | Path) -> str:
    name = Path(label_path).stem
    for suffix in ("_6fps", "_30fps"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def is_closeup(scene: str) -> bool:
    return "closeup" in scene


def select_rows(imgnames: np.ndarray, per_sequence: int = ROWS_PER_SEQUENCE) -> np.ndarray:
    """Deterministic, label-blind selection: ``per_sequence`` evenly spaced rows of each BEDLAM sequence.

    Rows of a sequence are ordered by (image name, original row); the picks
    are at round(linspace(0, n-1, k)).  Only names and order are read.
    """
    imgnames = np.asarray(imgnames)
    seq = np.array([n.split("/")[0] for n in imgnames])
    picked = []
    for s in sorted(set(seq)):
        rows = np.nonzero(seq == s)[0]
        rows = rows[np.lexsort((rows, imgnames[rows]))]
        k = min(per_sequence, len(rows))
        picked.extend(rows[np.unique(np.round(np.linspace(0, len(rows) - 1, k)).astype(int))].tolist())
    return np.array(sorted(picked), dtype=np.int64)


def select_rows_by_person(imgnames: np.ndarray, persons: np.ndarray, keep, per_person: int = ROWS_PER_SEQUENCE) -> np.ndarray:
    """``per_person`` evenly spaced rows of every (sequence, person) whose person passes ``keep``."""
    imgnames, persons = np.asarray(imgnames), np.asarray(persons)
    seq = np.array([n.split("/")[0] for n in imgnames])
    picked = []
    for s_, person in sorted(set(zip(seq.tolist(), persons.tolist()))):
        if not keep(person):
            continue
        rows = np.nonzero((seq == s_) & (persons == person))[0]
        rows = rows[np.lexsort((rows, imgnames[rows]))]
        k = min(per_person, len(rows))
        picked.extend(rows[np.unique(np.round(np.linspace(0, len(rows) - 1, k)).astype(int))].tolist())
    return np.array(sorted(picked), dtype=np.int64)


def subject_key(person: str) -> str:
    """RenderPeople scans of one person share a subject; synthetic bodies are their own subject."""
    import re

    m = re.match(r"(rp_[a-z]+)_posed_\d+$", person)
    return m.group(1) if m else person


def is_validation_subject(person: str) -> bool:
    key = subject_key(person)
    return int(hashlib.sha256(key.encode()).hexdigest(), 16) % VALIDATION_SUBJECT_MODULUS == 0


class PersonIndex:
    """SMPL-X betas -> BEDLAM person, from the official neutral motion release (one betas vector per motion file)."""

    def __init__(self, motion_dir: str | Path):
        files = sorted(Path(motion_dir).glob("*/*/motion_seq.npz"))
        if not files:
            raise SystemExit(f"no motion files under {motion_dir}")
        self.persons = np.array([f.parent.parent.name for f in files])
        self.betas = np.stack([np.load(f)["betas"].astype(np.float64) for f in files])

    def lookup(self, shape: np.ndarray) -> str | None:
        d = np.abs(self.betas - np.asarray(shape, np.float64)[None]).max(axis=1)
        i = int(np.argmin(d))
        return str(self.persons[i]) if d[i] < BETA_MATCH_TOLERANCE else None


def opencv_to_animcv(points: np.ndarray) -> np.ndarray:
    p = np.asarray(points, dtype=np.float64)
    return np.stack([p[..., 0], p[..., 2], -p[..., 1]], axis=-1)


def camera_joints(model, pose_cam, shape, trans_cam, cam_ext, device="cpu", chunk=256) -> np.ndarray:
    """(N, 22, 3) SMPL-X body joints in OpenCV camera coordinates (metres)."""
    import torch

    out = []
    for s in range(0, len(pose_cam), chunk):
        p = torch.as_tensor(np.asarray(pose_cam[s:s + chunk]), dtype=torch.float32, device=device)
        n = len(p)
        o = model(betas=torch.as_tensor(np.asarray(shape[s:s + chunk]), dtype=torch.float32, device=device),
                  global_orient=p[:, :3], body_pose=p[:, 3:66], jaw_pose=p[:, 66:69], leye_pose=p[:, 69:72],
                  reye_pose=p[:, 72:75], left_hand_pose=p[:, 75:120], right_hand_pose=p[:, 120:165],
                  expression=torch.zeros(n, 10, device=device),
                  transl=torch.as_tensor(np.asarray(trans_cam[s:s + chunk]), dtype=torch.float32, device=device))
        j = o.joints[:, :22].detach().cpu().numpy().astype(np.float64)
        out.append(j + np.asarray(cam_ext[s:s + chunk])[:, None, :3, 3])
    return np.concatenate(out) if out else np.zeros((0, 22, 3))


def project(points_cv: np.ndarray, cam_int: np.ndarray) -> np.ndarray:
    p = np.asarray(points_cv, dtype=np.float64)
    uv = np.einsum("...ij,...kj->...ki", cam_int, p / p[..., 2:3])
    return uv[..., :2]


def canonical_target(joints_cv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(17,3) root-relative AnimCV target and GT validity (joint in front of the camera)."""
    picked = np.asarray(joints_cv)[CANONICAL_SMPLX]
    valid = picked[:, 2] > 0
    animcv = opencv_to_animcv(picked)
    return (animcv - animcv[0]).astype(np.float32), valid


def associate(people: list[dict], gt_uv: np.ndarray, image_size: tuple[int, int]) -> tuple[int | None, dict]:
    """Pick the detected person that is the labelled subject (GT used only for identity)."""
    w, h = image_size
    limb = sorted(COCO_LIMB_TO_SMPLX)
    g = np.array([gt_uv[COCO_LIMB_TO_SMPLX[c]] for c in limb])
    inside = (g[:, 0] >= 0) & (g[:, 0] < w) & (g[:, 1] >= 0) & (g[:, 1] < h)
    if inside.sum() < 2:
        return None, {"reason": "gt_out_of_image"}
    diag = float(np.linalg.norm(g[inside].max(0) - g[inside].min(0)))
    tol = ASSOCIATION["tolerance_of_gt_limb_box_diagonal"] * max(diag, 1e-6)
    best, best_key, best_info = None, None, {"reason": "no_detection"}
    for i, person in enumerate(people):
        kp = person["keypoints"]
        det = np.array([kp[c, 2] > 0 for c in limb]) & inside
        if not det.any():
            continue
        d = np.linalg.norm(kp[limb, :2] - g, axis=1)
        correct = int((det & (d < tol)).sum())
        key = (correct, -float(np.median(d[det])))
        if best_key is None or key > best_key:
            best, best_key = i, key
            best_info = {"correct_limb_joints": correct, "median_norm_dist": float(np.median(d[det])) / max(diag, 1e-6)}
    if best is None or best_key[0] < ASSOCIATION["min_correct_limb_joints"]:
        return None, {**best_info, "reason": best_info.get("reason", "too_few_correct_joints")}
    return best, best_info


# --- a FrameBank-shaped container (only what geometry_tensor / the W68 extractor read) ---

@dataclass(frozen=True)
class ImageRef:
    root_key: str
    relative_path: str


@dataclass(frozen=True)
class ExternalSample:
    sample_id: str
    source: str
    sequence_id: str
    frame_index: int
    split: str
    image_size: tuple[int, int]
    image_reference: ImageRef
    detail: dict = field(default_factory=dict)


class ExternalBank:
    """Arrays input_2d/input_valid/target_3d/target_valid with FrameBank semantics and per-row metadata."""

    SCHEMA = "animcv_w73_external_bank_v1"
    DIGEST_DOMAIN = b"animcv_w73_external_bank_v1"
    ARRAYS = ("input_2d", "input_valid", "target_3d", "target_valid")

    def __init__(self, samples: list[ExternalSample], arrays: dict[str, np.ndarray], metadata: dict | None = None):
        n = len(samples)
        shapes = {"input_2d": (n, 17, 3), "input_valid": (n, 17), "target_3d": (n, 17, 3), "target_valid": (n, 17)}
        for k, s in shapes.items():
            if arrays[k].shape != s:
                raise ValueError(f"{k} shape {arrays[k].shape} != {s}")
        self.samples = samples
        self.arrays = {"input_2d": arrays["input_2d"].astype(np.float32), "input_valid": arrays["input_valid"].astype(bool),
                       "target_3d": arrays["target_3d"].astype(np.float32), "target_valid": arrays["target_valid"].astype(bool)}
        self.metadata = metadata or {}

    def __len__(self) -> int:
        return len(self.samples)

    def indices(self, split: str) -> np.ndarray:
        return np.array([i for i, s in enumerate(self.samples) if s.split == split], dtype=np.int64)

    def content_digest(self) -> str:
        d = hashlib.sha256(self.DIGEST_DOMAIN)
        for s in self.samples:
            d.update(f"{s.sample_id}|{s.split}|{s.source}|{s.image_reference.relative_path}".encode())
        for k in self.ARRAYS:
            d.update(k.encode())
            d.update(np.ascontiguousarray(self.arrays[k]).tobytes())
        return d.hexdigest()

    def save(self, path: str | Path) -> None:
        path = Path(path)
        np.savez(path.with_suffix(".npz"), **self.arrays)
        rows = [{"sample_id": s.sample_id, "source": s.source, "sequence_id": s.sequence_id, "frame_index": s.frame_index,
                 "split": s.split, "image_size": list(s.image_size),
                 "image_reference": {"root_key": s.image_reference.root_key, "relative_path": s.image_reference.relative_path},
                 "detail": s.detail} for s in self.samples]
        path.with_suffix(".json").write_text(json.dumps(
            {"schema": self.SCHEMA, "joint_names": list(H36M_NAMES), "coordinate_frame": "camera_root_relative",
             "forward_depth_axis": 1, "content_digest": self.content_digest(), "metadata": self.metadata,
             "samples": rows}))

    @classmethod
    def load(cls, path: str | Path) -> "ExternalBank":
        path = Path(path)
        index = json.loads(path.with_suffix(".json").read_text())
        if index["schema"] != cls.SCHEMA or index["joint_names"] != list(H36M_NAMES):
            raise SystemExit("not a Worklog 73 external bank")
        arrays = dict(np.load(path.with_suffix(".npz")))
        samples = [ExternalSample(r["sample_id"], r["source"], r["sequence_id"], r["frame_index"], r["split"],
                                  tuple(r["image_size"]), ImageRef(**r["image_reference"]), r.get("detail", {}))
                   for r in index["samples"]]
        bank = cls(samples, arrays, index.get("metadata"))
        if bank.content_digest() != index["content_digest"]:
            raise SystemExit("external bank content digest mismatch")
        return bank
