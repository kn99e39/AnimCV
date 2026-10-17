#!/usr/bin/env python3
"""DIAGNOSTIC (Worklog 85): external bank of the NEW BEDLAM training rows named by the expansion manifest.

Exactly the Worklog 73 BEDLAM TRAIN pipeline, with the manifest replacing ``select_rows``: neutral SMPL-X
camera joints (reprojection onto ``gtkps`` must be <= 0.5 px), the selected images streamed out of the
scene tar (closeup scenes rotated 90 degrees clockwise), the re-run OpenPose COCO-18 detector, GT-2D
identity association and the 3DPW conversion rules -- via the unchanged Worklog 73 ``process_member`` and
``finish_scene``.  Person / subject come from the Worklog 85 audit (Worklog 73 ``SceneBodyIndex``); a row
of a validation subject or with an unrecovered person stops the build.  Completed scenes are cached so an
interrupted build resumes.  Writes ``<out>/bedlam_expansion_new.{json,npz}``, ``<out>/images`` and a report.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import tarfile
import time
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_bedlam_external_bank import MAX_REPROJECTION_PX, finish_scene, process_member  # noqa: E402

from framepose.bedlam_supervision import (  # noqa: E402
    ASSOCIATION, SMPLX_MODEL, ExternalBank, ExternalSample, camera_joints, is_validation_subject, project, subject_key,
)
from framepose.openpose_coco18 import OpenPoseCOCO18  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "audit", "labels", "tars", "body_models", "out"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    args = parser.parse_args()
    import smplx
    import torch

    manifest = json.loads(args.manifest.read_text())
    audit = np.load(args.audit / "rows.npz")
    tag_of = dict(zip(manifest["scene_names"], manifest["label_tags"]))
    person_of = {(audit["scene_names"][s], int(r)): (str(p), str(ident), bool(v))
                 for s, r, p, ident, v in zip(audit["scene"], audit["label_row"], audit["person"], audit["identity"],
                                              audit["validation_subject"])}
    by_scene = defaultdict(list)
    for scene, row in manifest["new_rows"]:
        by_scene[scene].append(int(row))
    device = "cuda"
    model = smplx.create(str(args.body_models), **SMPLX_MODEL).to(device)
    detector = OpenPoseCOCO18(device)
    image_dir, cache_dir = args.out / "images", args.out / "scenes"
    cache_dir.mkdir(parents=True, exist_ok=True)
    build_args = SimpleNamespace(split="train")
    samples, arrays, drops, report, started = [], defaultdict(list), Counter(), {}, time.time()
    for scene in sorted(by_scene):
        tag = tag_of[scene]
        cache = cache_dir / f"{tag}.pkl"
        if cache.exists():
            part = pickle.loads(cache.read_bytes())
        else:
            label_path = args.labels / f"{tag}.npz"
            z = dict(np.load(label_path))  # materialized once: NpzFile re-reads a whole array on every key access
            rows = np.array(sorted(by_scene[scene]), dtype=np.int64)
            for r in rows:
                person, ident, val = person_of[(scene, int(r))]
                if ident != "ok" or val:
                    raise SystemExit(f"STOP: {scene} row {r}: identity {ident!r} / validation subject {val}")
            joints = camera_joints(model, z["pose_cam"][rows], z["shape"][rows], z["trans_cam"][rows], z["cam_ext"][rows], device)
            reproj = np.linalg.norm(project(joints, z["cam_int"][rows]) - z["gtkps"][rows][:, :22, :2], axis=2)
            if float(reproj.max()) > MAX_REPROJECTION_PX:
                raise SystemExit(f"STOP: {scene}: SMPL-X reprojection error {reproj.max():.3f}px")
            wanted = {}
            for k, r in enumerate(rows):
                wanted.setdefault(f"png/{z['imgname'][r]}", []).append(k)
            parts = [args.tars / f"{tag}.tar"]
            people_by_member, member_sha, sizes = {}, {}, {}
            with tarfile.open(parts[0], "r|") as tf:
                for member in tf:
                    top, _, key = member.name.partition("/")
                    if top not in (tag, scene) or key not in wanted or key in people_by_member:
                        continue
                    process_member(tf, member, key, scene, image_dir, detector, people_by_member, member_sha, sizes)
                    if len(people_by_member) == len(wanted):
                        break
            missing = [m for m in wanted if m not in people_by_member]
            if missing:
                raise SystemExit(f"STOP: {scene}: {len(missing)} manifest images missing from {parts[0].name}")
            s_part, a_part, rep, d_part = finish_scene(label_path, build_args, z, rows, joints, reproj, wanted, people_by_member,
                                                       member_sha, sizes, parts, [], defaultdict(list), Counter(), time.time())
            for s in s_part:
                person, _, _ = person_of[(scene, int(s.detail["label_row"]))]
                s.detail.update({"person": person, "subject": subject_key(person), "worklog": "W85 expansion"})
            part = (s_part, a_part, rep, d_part)
            cache.write_bytes(pickle.dumps(part))
        s_part, a_part, rep, d_part = part
        samples += s_part
        for k, v in a_part.items():
            arrays[k] += v
        report[scene] = rep
        drops.update(d_part)
        print(json.dumps({"scene": scene, "manifest_rows": len(by_scene[scene]), "kept": len(s_part),
                          "seconds": round(time.time() - started)}), flush=True)
    if any(is_validation_subject(s.detail["person"]) for s in samples):
        raise SystemExit("STOP: validation subject in the expansion bank")
    bank = ExternalBank([ExternalSample(s.sample_id, s.source, s.sequence_id, s.frame_index, s.split, s.image_size,
                                        s.image_reference, s.detail) for s in samples],
                        {k: np.stack(v) for k, v in arrays.items()},
                        {"split": "train", "association": ASSOCIATION, "detector": detector.identity(), "smplx": SMPLX_MODEL,
                         "expansion_manifest_sha256": manifest["manifest_sha256"]})
    bank.save(args.out / "bedlam_expansion_new")
    out = {"schema": "animcv_w85_bedlam_expansion_bank_build_v1", "manifest_rows": len(manifest["new_rows"]), "rows": len(bank),
           "content_digest": bank.content_digest(), "drops": dict(drops), "scenes": report, "seconds": time.time() - started,
           "torch": torch.__version__}
    (args.out / "bedlam_expansion_new_build_report.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "scenes"}, indent=1))


if __name__ == "__main__":
    main()
