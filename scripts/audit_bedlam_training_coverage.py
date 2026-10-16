#!/usr/bin/env python3
"""DIAGNOSTIC (Worklog 85): row-level audit of every BEDLAM TRAINING-scene label row.

For each of the 26 training-scene label files (the Worklog 73 scene split; agora* and the four official
validation scenes are excluded) and every row:
  * RGB: whether the row's image is a member of the scene's on-disk image tar (no image is reconstructed);
  * identity: person recovered exactly as Worklog 73 (``SceneBodyIndex``: official ``be_seq.csv`` bodies of
    the row's sequence, disambiguated by SMPL-X betas), subject key, and whether the subject is a BEDLAM
    VALIDATION subject (sha256(subject) mod 5 == 0) -- such rows can never enter training;
  * history: Worklog 73 label-blind selection (``select_rows``) and final TRAIN bank membership;
  * GT: neutral SMPL-X camera joints (Worklog 73 ``camera_joints``), reprojection onto ``gtkps``, the
    canonical left-arm target validity (shoulder/elbow/wrist/pelvis in front of the camera) and the
    Worklog 80 normalized arm (E, W);
  * coverage descriptors (training-sample selection / reporting only, never a model input): GT bend angle,
    chain depth component f, left-arm projections inside the image, camera pitch from ``cam_ext``.
Writes ``<out>/rows.npz`` (one entry per row) and ``<out>/audit_report.json``.  Read-only on all inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
import time
from collections import Counter
from pathlib import Path

import numpy as np

from framepose.bedlam_supervision import (
    CANONICAL_SMPLX, SCENES_VALIDATION, SMPLX_MODEL, ExternalBank, SceneBodyIndex, camera_joints, is_validation_subject,
    opencv_to_animcv, project, scene_of, select_rows, subject_key,
)

ARM = (16, 18, 20)  # SMPL-X left shoulder, elbow, wrist
MAX_REPROJECTION_PX = 0.5


def tar_members(tars: Path, tag: str, scene: str) -> tuple[set[str], dict]:
    """Image keys ``png/<seq>/<file>`` present in the scene's tar(s); a truncated tar keeps what was readable."""
    parts = [tars / f"{tag}.tar"] if (tars / f"{tag}.tar").exists() else []
    keys, info = set(), {"tars": [p.name for p in parts], "truncated": False}
    for p in parts:
        try:
            with tarfile.open(p, "r:") as tf:
                for m in tf:
                    top, _, key = m.name.partition("/")
                    if top in (tag, scene) and key.startswith("png/") and m.isfile():
                        keys.add(key)
        except (tarfile.ReadError, EOFError, OSError) as exc:
            info["truncated"], info["error"] = True, str(exc)
    info["members"] = len(keys)
    return keys, info


def camera_pitch_degrees(cam_ext: np.ndarray) -> np.ndarray:
    """Elevation of the camera optical axis above the world horizontal (world +Y up, OpenCV camera axes)."""
    r = np.asarray(cam_ext)[:, :3, :3]
    forward_world = r.transpose(0, 2, 1) @ np.array([0.0, 0.0, 1.0])
    return np.degrees(np.arcsin(np.clip(forward_world[:, 1], -1, 1)))


def arm_descriptors(joints_cv: np.ndarray):
    """Canonical left-arm target validity, normalized E/W (Worklog 80), bend angle and chain depth component."""
    picked = joints_cv[:, CANONICAL_SMPLX]
    front = picked[:, :, 2] > 0
    a = opencv_to_animcv(picked)
    a = a - a[:, :1]
    s, e, w = a[:, 11], a[:, 12], a[:, 13]
    lu, ll = np.linalg.norm(e - s, axis=1), np.linalg.norm(w - e, axis=1)
    ok = front[:, [0, 11, 12, 13]].all(1) & (lu > 1e-6) & (ll > 1e-6)
    total = np.where(ok, lu + ll, 1.0)[:, None]
    en, wn = (e - s) / total, (w - s) / total
    u, v = en / np.maximum(np.linalg.norm(en, axis=1, keepdims=True), 1e-12), (wn - en)
    v = v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)
    bend = np.degrees(np.arctan2(np.linalg.norm(np.cross(u, v), axis=1), (u * v).sum(1)))
    f = wn[:, 1] / np.maximum(np.linalg.norm(wn, axis=1), 1e-12)
    return ok, en.astype(np.float32), wn.astype(np.float32), bend.astype(np.float32), f.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("labels", "tars", "gt_dir", "motion_dir", "body_models", "historical_train_bank", "out"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    args = parser.parse_args()
    import smplx
    import torch

    if args.out.exists() and any(args.out.iterdir()):
        raise SystemExit("STOP: output directory is not empty; never overwrite experimental artifacts")
    model = smplx.create(str(args.body_models), **SMPLX_MODEL).to("cuda")
    index = SceneBodyIndex(args.gt_dir, args.motion_dir)
    hist = ExternalBank.load(args.historical_train_bank)
    hist_final = {s.sample_id for s in hist.samples}
    label_files = sorted(p for p in args.labels.glob("*.npz") if not p.name.startswith("agora")
                         and scene_of(p) not in SCENES_VALIDATION)
    if len(label_files) != 26:
        raise SystemExit(f"STOP: expected the 26 Worklog 73 training scenes, found {len(label_files)}")
    cols = {k: [] for k in ("scene", "label_row", "frame", "sequence", "person", "subject", "identity", "validation_subject",
                            "rgb", "hist_selected", "hist_final", "gt_arm_ok", "elbow", "wrist", "bend", "f", "arm_in_image",
                            "pitch", "reprojection_px")}
    report, started = {"scenes": {}}, time.time()
    for si, path in enumerate(label_files):
        scene, tag = scene_of(path), path.stem
        z = np.load(path)
        names = z["imgname"]
        n = len(names)
        keys, tar_info = tar_members(args.tars, tag, scene)
        rgb = np.array([f"png/{x}" in keys for x in names])
        seqs = np.array([x.split("/")[0] for x in names])
        frames = np.array([int(Path(x).stem.rsplit("_", 1)[1]) for x in names])
        # identity: one lookup per (sequence, betas vector)
        person = np.empty(n, dtype=object)
        reason = np.empty(n, dtype=object)
        shape = z["shape"]
        groups: dict[tuple, list[int]] = {}
        for r in range(n):
            groups.setdefault((seqs[r], shape[r].tobytes()), []).append(r)
        for (seq, _), rows in groups.items():
            p, info = index.lookup(scene, seq, shape[rows[0]])
            person[rows] = p if p is not None else ""
            reason[rows] = "ok" if p is not None else info["reason"]
        subj = np.array([subject_key(p) if p else "" for p in person])
        val_subj = np.array([bool(p) and is_validation_subject(p) for p in person])
        sel = np.zeros(n, bool)
        sel[select_rows(names)] = True
        final = np.array([f"bedlam:{scene}:{seqs[r]}#row{r:07d}" in hist_final for r in range(n)])
        joints = camera_joints(model, z["pose_cam"], shape, z["trans_cam"], z["cam_ext"], "cuda", chunk=2048)
        reproj = np.linalg.norm(project(joints, z["cam_int"]) - z["gtkps"][:, :22, :2], axis=2).max(1)
        ok, en, wn, bend, f = arm_descriptors(joints)
        size = z["cam_int"][:, :2, 2] * 2
        uv = z["gtkps"][:, list(ARM), :2]
        inside = ((uv >= 0) & (uv < size[:, None, :])).all(2).all(1)
        pitch = camera_pitch_degrees(z["cam_ext"])
        for k, v in (("scene", np.full(n, si)), ("label_row", np.arange(n)), ("frame", frames), ("sequence", seqs),
                     ("person", person.astype(str)), ("subject", subj), ("identity", reason.astype(str)),
                     ("validation_subject", val_subj), ("rgb", rgb), ("hist_selected", sel), ("hist_final", final),
                     ("gt_arm_ok", ok), ("elbow", en), ("wrist", wn), ("bend", bend), ("f", f), ("arm_in_image", inside),
                     ("pitch", pitch.astype(np.float32)), ("reprojection_px", reproj.astype(np.float32))):
            cols[k].append(v)
        report["scenes"][scene] = {
            "label_file": path.name, "label_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "rows": n,
            "unique_images": int(len(set(names.tolist()))), "sequences": int(len(set(seqs.tolist()))), "tar": tar_info,
            "rows_with_rgb": int(rgb.sum()), "identity": dict(Counter(reason.tolist())),
            "validation_subject_rows": int(val_subj.sum()), "hist_selected": int(sel.sum()), "hist_final": int(final.sum()),
            "gt_arm_ok": int(ok.sum()), "arm_in_image": int(inside.sum()), "max_reprojection_px": float(reproj.max()),
            "pitch_degrees": {"p05": float(np.percentile(pitch, 5)), "p50": float(np.median(pitch)), "p95": float(np.percentile(pitch, 95))}}
        print(json.dumps({"scene": scene, **{k: v for k, v in report["scenes"][scene].items() if k not in ("label_sha256",)},
                          "seconds": round(time.time() - started)}), flush=True)
        if float(reproj.max()) > MAX_REPROJECTION_PX:
            raise SystemExit(f"STOP: {scene}: SMPL-X reprojection {reproj.max():.3f}px exceeds {MAX_REPROJECTION_PX}")
    arrays = {k: np.concatenate(v) for k, v in cols.items()}
    args.out.mkdir(parents=True, exist_ok=True)
    np.savez(args.out / "rows.npz", scene_names=np.array([scene_of(p) for p in label_files]),
             label_tags=np.array([p.stem for p in label_files]), **arrays)
    totals = {"rows": int(len(arrays["rgb"])), "rows_with_rgb": int(arrays["rgb"].sum()),
              "identity_ok": int((arrays["identity"] == "ok").sum()),
              "validation_subject_rows": int(arrays["validation_subject"].sum()),
              "hist_selected": int(arrays["hist_selected"].sum()), "hist_final": int(arrays["hist_final"].sum()),
              "gt_arm_ok": int(arrays["gt_arm_ok"].sum()), "arm_in_image": int(arrays["arm_in_image"].sum()),
              "persons": int(len(set(arrays["person"].tolist()) - {""})),
              "subjects": int(len(set(arrays["subject"].tolist()) - {""})),
              "training_subjects": int(len({s for s, v in zip(arrays["subject"], arrays["validation_subject"]) if s and not v})),
              "torch": torch.__version__, "seconds": time.time() - started}
    report["totals"] = totals
    report["rows_npz_sha256"] = hashlib.sha256((args.out / "rows.npz").read_bytes()).hexdigest()
    (args.out / "audit_report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(totals, indent=1))


if __name__ == "__main__":
    main()
