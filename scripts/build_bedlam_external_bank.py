#!/usr/bin/env python3
"""DIAGNOSTIC (Worklog 73): build the BEDLAM external bank (detector input, SMPL-X target) for one split.

For every scene label file: label-blind deterministic row selection
(``select_rows``), neutral SMPL-X camera joints (reprojection onto ``gtkps``
must be exact), the selected images streamed out of the scene tar (closeup
scenes rotated 90 degrees clockwise), the re-run OpenPose COCO-18 detector on
every selected image, GT-2D identity association, and the 3DPW conversion
rules.  Writes ``<out>/bedlam_<split>.{json,npz}``, the extracted images
under ``<out>/images`` and a build report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import tarfile
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

from framepose.bedlam_supervision import (
    ASSOCIATION, ROWS_PER_SEQUENCE, SCENES_VALIDATION, SMPLX_MODEL, ExternalBank, ExternalSample, ImageRef, associate,
    camera_joints, canonical_target, is_closeup, is_validation_subject, project, scene_of, select_rows,
    select_rows_by_person, subject_key,
)
from framepose.openpose_coco18 import OpenPoseCOCO18, to_canonical_input

MAX_REPROJECTION_PX = 0.5


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("train", "validation"), required=True)
    parser.add_argument("--labels", type=Path, required=True, help="directory of per-scene label npz files")
    parser.add_argument("--tars", type=Path, required=True, help="directory of image tars")
    parser.add_argument("--body-models", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--only", default=None, help="smoke test: restrict to scenes containing this substring")
    args = parser.parse_args()
    import smplx
    import torch

    device = "cuda"
    model = smplx.create(str(args.body_models), **SMPLX_MODEL).to(device)
    detector = OpenPoseCOCO18(device)
    label_files = sorted(p for p in args.labels.glob("*.npz") if not p.name.startswith("agora"))
    if args.split == "train":
        label_files = [p for p in label_files if scene_of(p) not in SCENES_VALIDATION]
    else:
        label_files = [p for p in label_files if scene_of(p) in SCENES_VALIDATION]
    if args.only:
        label_files = [p for p in label_files if args.only in p.name]
    if not label_files:
        raise SystemExit("no label files for split")
    image_dir = args.out / "images"
    cache_dir = args.out / f"scenes_{args.split}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    samples, arrays, report_scenes, drops, failed = [], defaultdict(list), {}, Counter(), {}
    started = time.time()
    for label_path in label_files:
        cache = cache_dir / f"{label_path.stem}.pkl"
        if cache.exists():
            part = pickle.loads(cache.read_bytes())
        else:
            try:
                part = process_scene(label_path, args, model, detector, device, image_dir)
            except tarfile.ReadError as exc:
                failed[scene_of(label_path)] = f"tar read error: {exc}"
                print(json.dumps({"scene": scene_of(label_path), "failed": failed[scene_of(label_path)]}), flush=True)
                continue
            cache.write_bytes(pickle.dumps(part))
        s_part, a_part, rep, d_part = part
        samples += s_part
        for k, v in a_part.items():
            arrays[k] += v
        report_scenes[scene_of(label_path)] = rep
        drops.update(d_part)
    if failed:
        (args.out / f"bedlam_{args.split}_build_failures.json").write_text(json.dumps(failed, indent=1))
        raise SystemExit(f"incomplete: {sorted(failed)}; rerun after fixing (completed scenes are cached)")

    bank = ExternalBank(samples, {k: np.stack(v) for k, v in arrays.items()},
                        {"split": args.split, "rows_per_sequence": ROWS_PER_SEQUENCE, "association": ASSOCIATION,
                         "detector": detector.identity(), "smplx": SMPLX_MODEL})
    args.out.mkdir(parents=True, exist_ok=True)
    bank.save(args.out / f"bedlam_{args.split}")
    report = {"schema": "animcv_w73_bedlam_bank_build_v1", "split": args.split, "rows": len(bank),
              "content_digest": bank.content_digest(), "drops": dict(drops), "scenes": report_scenes,
              "seconds": time.time() - started, "torch": torch.__version__}
    (args.out / f"bedlam_{args.split}_build_report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k != "scenes"}, indent=1))


def process_scene(label_path, args, model, detector, device, image_dir):
    """One scene -> (samples, arrays, scene report, drops).  Raises tarfile.ReadError on a truncated tar."""
    samples, arrays, drops, started = [], defaultdict(list), Counter(), time.time()
    scene = scene_of(label_path)
    tag = label_path.stem
    z = np.load(label_path)
    if args.split == "validation":
        # Official validation scenes, validation subjects only (labels carry the official ``sub`` person id).
        rows = select_rows_by_person(z["imgname"], z["sub"], is_validation_subject)
    else:
        rows = select_rows(z["imgname"])
    joints = camera_joints(model, z["pose_cam"][rows], z["shape"][rows], z["trans_cam"][rows], z["cam_ext"][rows], device)
    reproj = np.linalg.norm(project(joints, z["cam_int"][rows]) - z["gtkps"][rows][:, :22, :2], axis=2)
    if float(reproj.max()) > MAX_REPROJECTION_PX:
        raise SystemExit(f"{scene}: SMPL-X reprojection error {reproj.max():.3f}px exceeds {MAX_REPROJECTION_PX}")
    wanted = {f"png/{z['imgname'][r]}": [] for r in rows}
    for k, r in enumerate(rows):
        wanted[f"png/{z['imgname'][r]}"].append(k)
    # 6fps training tars: one "<tag>.tar"; validation scenes: the five official 30fps PNG parts.
    parts = [args.tars / f"{tag}.tar"] if (args.tars / f"{tag}.tar").exists() else \
        sorted(args.tars.glob(f"{scene}_png.*.tar")) or sorted(args.tars.glob(f"{scene}_*.tar"))
    if not parts:
        raise SystemExit(f"{scene}: no image tar")
    people_by_member, member_sha, sizes = {}, {}, {}
    for tar_path in parts:
        if len(people_by_member) == len(wanted):
            break
        with tarfile.open(tar_path, "r|") as tf:
            for member in tf:
                top, _, key = member.name.partition("/")
                if top not in (tag, scene) or key not in wanted or key in people_by_member:
                    continue
                process_member(tf, member, key, scene, image_dir, detector, people_by_member, member_sha, sizes)
                if len(people_by_member) == len(wanted):
                    break
    missing = [m for m in wanted if m not in people_by_member]
    if missing:
        raise SystemExit(f"{scene}: {len(missing)} selected images missing from {[p.name for p in parts]}")
    return finish_scene(label_path, args, z, rows, joints, reproj, wanted, people_by_member, member_sha, sizes, parts,
                        samples, arrays, drops, started)


def process_member(tf, member, key, scene, image_dir, detector, people_by_member, member_sha, sizes):
    """Decode one selected image (closeup scenes rotated 90 degrees clockwise), store it, run the detector."""
    data = tf.extractfile(member).read()
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if is_closeup(scene):
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    rel = f"{scene}/{member.name.split('/png/', 1)[1]}"
    dest = image_dir / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    ok, enc = cv2.imencode(".png", img)
    dest.write_bytes(enc.tobytes())
    people_by_member[key] = (rel, detector(img))
    member_sha[key] = hashlib.sha256(data).hexdigest()
    sizes[key] = (img.shape[1], img.shape[0])


def finish_scene(label_path, args, z, rows, joints, reproj, wanted, people_by_member, member_sha, sizes, parts,
                 samples, arrays, drops, started):
    scene = scene_of(label_path)
    kept = Counter()
    for member, ks in wanted.items():
        rel, people = people_by_member[member]
        size = sizes[member]
        for k in ks:
            r = int(rows[k])
            if not np.allclose(z["cam_int"][r][:2, 2] * 2, size, atol=1.0):
                raise SystemExit(f"{scene}: image size {size} does not match principal point {z['cam_int'][r][:2, 2]}")
            idx, info = associate(people, z["gtkps"][r][:, :2], size)
            if idx is None:
                drops[f"association:{info['reason']}"] += 1
                continue
            inp, visible = to_canonical_input(people[idx]["keypoints"], size)
            target, gt_valid = canonical_target(joints[k])
            target_valid = visible & gt_valid
            if not target_valid[0]:
                drops["pelvis_invalid"] += 1
                continue
            seq = z["imgname"][r].split("/")[0]
            frame = int(Path(z["imgname"][r]).stem.rsplit("_", 1)[1])
            samples.append(ExternalSample(
                sample_id=f"bedlam:{scene}:{seq}#row{r:07d}", source="BEDLAM", sequence_id=f"bedlam:{scene}:{seq}",
                frame_index=frame, split=args.split, image_size=size, image_reference=ImageRef("bedlam_frames", rel),
                detail={"label_file": label_path.name, "label_row": r, "closeup_rotated_cw": is_closeup(scene),
                        "tar_member_sha256": member_sha[member], "association": info,
                        "people_detected": len(people),
                        **({"person": str(z["sub"][r]), "subject": subject_key(str(z["sub"][r]))}
                           if args.split == "validation" else {})}))
            arrays["input_2d"].append(inp)
            arrays["input_valid"].append(inp[:, 2] > 0)
            arrays["target_3d"].append(target)
            arrays["target_valid"].append(target_valid)
            kept[seq] += 1
    report = {"label_rows": int(len(z["imgname"])), "selected_rows": int(len(rows)),
              "kept_rows": int(sum(kept.values())), "sequences": int(len(set(n.split('/')[0] for n in z["imgname"]))),
              "max_reprojection_px": float(reproj.max()), "closeup_rotated_cw": is_closeup(scene),
              "tar": [p.name for p in parts], "label_sha256": hashlib.sha256(label_path.read_bytes()).hexdigest()}
    print(json.dumps({"scene": scene, **{k: v for k, v in report.items() if k != "label_sha256"},
                      "seconds": round(time.time() - started)}), flush=True)
    return samples, dict(arrays), report, drops


if __name__ == "__main__":
    main()
