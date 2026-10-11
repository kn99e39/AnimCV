#!/usr/bin/env python3
"""DIAGNOSTIC (Worklog 73): labels for the four official BEDLAM validation scenes, via the official code.

The processed ``all_npz_12_training`` release omits the validation scenes, so
their labels are produced with the official processing module
(pixelite1201/BEDLAM@f1828fd ``data_processing/df_full_body.py``) imported
unchanged: its ``get_params`` (camera/body transforms, neutral SMPL-X with 11
betas, projection, visibility filters) does all the geometry.  Only the
``__main__`` orchestration (scene CSV -> per-body ``get_params`` calls) is
restated here.  ``get_params`` lists the PNG files of each sequence folder to
know the frame names; the 30fps PNG archives are only *listed* and empty
placeholder files with those names stand in (``visualize='none'`` never opens
an image).  The renderer import is stubbed for the same reason.  Output: one
``<scene>_6fps.npz`` per scene in the training-label format.
"""

from __future__ import annotations

import argparse
import os
import sys
import tarfile
import tempfile
import types
from pathlib import Path

import numpy as np
import pandas as pd

from framepose.bedlam_supervision import SCENES_VALIDATION, is_closeup

LIST_KEYS = ("imgnames", "genders", "betas", "poses_cam", "poses_world", "trans_cam", "trans_world", "cam_ext", "cam_int",
             "centers", "scales", "gtkps", "proj_verts", "motion_infos", "subs")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bedlam-repo", type=Path, required=True, help="pixelite1201/BEDLAM checkout at f1828fd")
    parser.add_argument("--gt-dir", type=Path, required=True, help="extracted <scene>_gt.tar.gz contents")
    parser.add_argument("--motion-dir", type=Path, required=True, help="neutral_ground_truth_motioninfo directory")
    parser.add_argument("--png-tars", type=Path, required=True)
    parser.add_argument("--body-models", type=Path, required=True, help="directory containing smplx/SMPLX_NEUTRAL.npz")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    work = Path(tempfile.mkdtemp(prefix="bedlam_val_"))
    (work / "bedlam_data/body_models/smplx").mkdir(parents=True)
    os.symlink(args.body_models.resolve(), work / "bedlam_data/body_models/smplx/models")
    os.symlink((args.bedlam_repo / "data_processing/downsample_mat_smplx.pkl").resolve(), work / "downsample_mat_smplx.pkl")
    sys.modules["renderer_pyrd"] = types.SimpleNamespace(Renderer=None)
    sys.path.insert(0, str(args.bedlam_repo / "data_processing"))
    cwd = os.getcwd()
    os.chdir(work)
    import df_full_body as official  # noqa: E402  (official module, unchanged)
    os.chdir(cwd)

    args.out.mkdir(parents=True, exist_ok=True)
    for scene in SCENES_VALIDATION:
        names = []
        for part in sorted(args.png_tars.glob(f"{scene}_png.*.tar")):
            with tarfile.open(part, "r:") as tf:
                names += [m.name for m in tf.getmembers() if m.name.endswith(".png")]
        if not names:
            raise SystemExit(f"{scene}: no PNG archives")
        fake = work / "images" / scene / "png"
        for n in names:
            seq, file = n.split("/")[-2:]
            (fake / seq).mkdir(parents=True, exist_ok=True)
            (fake / seq / file).touch()
        if is_closeup(scene):
            official.SENSOR_W, official.SENSOR_H, official.IMG_W, official.IMG_H, rotate = 20.25, 36, 720, 1280, True
        else:
            official.SENSOR_W, official.SENSOR_H, official.IMG_W, official.IMG_H, rotate = 36, 20.25, 1280, 720, False
        for key in LIST_KEYS:
            getattr(official, key).clear()
        base = next(p for p in (args.gt_dir / scene, *args.gt_dir.glob(f"**/{scene}")) if (p / "be_seq.csv").exists())
        csv_data = pd.read_csv(base / "be_seq.csv").to_dict("list")
        cam_base = base / "ground_truth/camera"
        for idx, comment in enumerate(csv_data["Comment"]):
            if "sequence_name" in comment:
                seq_name = comment.split(";")[0].split("=")[-1]
                cam = pd.read_csv(cam_base / f"{seq_name}_camera.csv").to_dict("list")
                cam_x, cam_y, cam_z = cam["x"], cam["y"], cam["z"]
                cam_yaw_, cam_pitch_, cam_roll_, fl = cam["yaw"], cam["pitch"], cam["roll"], cam["focal_length"]
            elif "start_frame" in comment:
                start_frame = int(comment.split(";")[0].split("=")[-1])
                body = csv_data["Body"][idx]
                parts = body.split("_")
                person_id, sequence_id = "_".join(parts[:-1]), parts[-1]
                param = np.load(args.motion_dir / person_id / sequence_id / "motion_seq.npz")
                official.get_params(str(fake / seq_name), fl, start_frame, param["gender_sub"].item(), param,
                                    [csv_data["X"][idx], csv_data["Y"][idx], csv_data["Z"][idx]], csv_data["Yaw"][idx],
                                    cam_x, cam_y, cam_z, 6, person_id, cam_pitch_=cam_pitch_, cam_roll_=cam_roll_,
                                    cam_yaw_=cam_yaw_, visualize="none", rotate_flag=rotate)
        np.savez(args.out / f"{scene}_6fps.npz", imgname=official.imgnames, center=official.centers,
                 scale=official.scales, pose_cam=official.poses_cam, pose_world=official.poses_world,
                 shape=official.betas, trans_cam=official.trans_cam, trans_world=official.trans_world,
                 gtkps=official.gtkps, cam_int=official.cam_int, cam_ext=official.cam_ext, gender=official.genders,
                 proj_verts=official.proj_verts, motion_info=official.motion_infos, sub=official.subs)
        print(scene, len(official.imgnames), "rows", len(set(official.imgnames)), "images", flush=True)


if __name__ == "__main__":
    main()
