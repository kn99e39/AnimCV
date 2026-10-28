#!/usr/bin/env python3
"""EXPERIMENT (Worklog 100): AnimCV's own RTMDet + RTMPose observations for the frozen 3DPW evaluation frames.

Runs in the pose environment (``.venv``: mmpose 1.3.2 / mmdet 3.3.0), CPU (the production default device).
Production code is called unchanged:
  ``gate``     W99A batching parity: production PoseEstimator with batch_size=1 (M1) and 8 (M8, default), a second M8
               run (determinism), and the historical per-frame implementation loaded verbatim from git
               (``01c0dc3:src/pose/mmpose_adapter.py``) as a frozen reference (H); detection-level parity through the
               production ``_detect_batch`` helper with 1 vs 8 frames per pass; synthetic controls (person leaving and
               returning, empty frames, mixed image sizes in one sequence).
  ``observe``  the full 3DPW VALIDATION+TEST population: the production M8 observation per video (tracked subject),
               plus every RTMDet person candidate with its own RTMPose result (for the evaluation-only, oracle-associated
               same-person control; never a runtime candidate).
Every image is hashed (``framepose.observations.image_content_digest``).  Models are initialised once, sequentially, in
the main thread (no concurrent use of the global torch.load patch).  ``PREDECLARED`` is fixed before any comparison.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
import threading
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from framepose.observations import image_content_digest  # noqa: E402
from mediaio.frame_sequence import Frame, FrameSequence  # noqa: E402
from pose import mmpose_adapter as prod  # noqa: E402
from pose.default_detector import get_default_detector_checkpoint_path, get_default_detector_config_path  # noqa: E402
from pose.default_model import get_default_pose_checkpoint_path, get_default_pose_config_path  # noqa: E402
from pose.subject_tracker import Detection, SubjectTracker  # noqa: E402

CHECKPOINT_SHA = {"rtmdet_tiny": "78e30dcce0c6f594eaff0d6977b84b4103688b4aff0ad1aa16008a8cc854a7fb",
                  "rtmpose_tiny": "77ffc7e802acf10951c353e8bc68b4f05218121177ceaea163aa124436ba6fb7"}
HISTORICAL_COMMIT = "01c0dc3"  # last commit before W99A changed src/pose/mmpose_adapter.py
CANON = ["head", "neck", "pelvis", "spine", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist",
         "right_wrist", "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle"]
PREDECLARED = {
    "device": "cpu (production default); fp32; mmpose 1.3.2, mmdet 3.3.0, torch from .venv",
    "checkpoints": "official OpenMMLab RTMDet-tiny / RTMPose-tiny, byte-identical to the official URLs (sha256 pinned)",
    "production_settings": "MMPoseConfig defaults: visibility_threshold 0.3, detector person score >= 0.3, subject_box None "
                           "(tracker starts on conf x area), batch_size 8 (M8) / 1 (M1)",
    "gate_videos": ["downtown_walkBridge_01", "downtown_bar_00", "downtown_arguing_00", "downtown_runForBus_01",
                    "flat_guitar_01", "outdoors_parcours_00"],
    "gate_controls": "walkBridge frames 0-23 with 6 black frames inserted after frame 8 (leave/return) and 2 leading black "
                     "frames (no detection); 12 walkBridge (1920x1080) + 12 flat_guitar (1080x1920) frames in one sequence",
    "parity_contract": {
        "detections": "identical person-candidate count per frame; box coordinates |d| <= 0.5 px; score |d| <= 1e-3",
        "subject": "selected box IoU >= 0.99 between variants in every frame; identical tracked/no-detection frames",
        "keypoints": "|dx|, |dy| <= 0.5 px for every canonical landmark; confidence |d| <= 1e-3; visibility flags identical "
                     "except flips with confidence within 1e-3 of 0.3 (reported)",
        "order_and_coverage": "identical frame order and identical set of frames with landmarks",
        "determinism": "two M8 runs bitwise identical",
        "justification": "W99A's 16-frame smoke showed <= 0.0015 px batch-composition noise; 0.5 px is < 0.05% of the image "
                         "width and below RTMPose-tiny's SimCC output resolution mapped to image space",
    },
    "population": "every 3DPW VALIDATION+TEST row of the frozen W97 population (10,483 rows, 36 videos, 9,251 unique "
                  "images); per video, frames in ascending frame index (the bank's stride, about 10 fps); no frame "
                  "selection by detection outcome",
}


def sha256_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load_historical_adapter(work: Path):
    """The pre-W99A per-frame adapter, verbatim from git (no production file is touched)."""
    src = subprocess.run(["git", "-C", str(ROOT), "show", f"{HISTORICAL_COMMIT}:src/pose/mmpose_adapter.py"],
                         check=True, capture_output=True, text=True).stdout
    path = work / "historical_mmpose_adapter_01c0dc3.py"
    path.write_text(src)
    spec = importlib.util.spec_from_file_location("historical_mmpose_adapter", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["historical_mmpose_adapter"] = mod
    spec.loader.exec_module(mod)
    return mod, hashlib.sha256(src.encode()).hexdigest()


def config(batch_size=8):
    return prod.MMPoseConfig(get_default_pose_config_path(), get_default_pose_checkpoint_path(download=False), device="cpu",
                             detector_config_path=get_default_detector_config_path(),
                             detector_checkpoint_path=get_default_detector_checkpoint_path(download=False),
                             batch_size=batch_size)


def landmarks_array(pose_frame):
    """(16, 4) [x, y, conf, visible] in pixels for CANON; NaN rows for absent landmarks."""
    a = np.full((len(CANON), 4), np.nan)
    for i, n in enumerate(CANON):
        p = pose_frame.landmarks.get(n)
        if p is not None:
            a[i] = (p.x, p.y, p.confidence, float(p.visible))
    return a


def load_frames(paths):
    return [Frame(index=i, timestamp=i / 10.0, image=cv2.imread(str(p)), width=0, height=0) for i, p in enumerate(paths)]


def seq_of(frames):
    for f in frames:
        f.height, f.width = f.image.shape[:2]
    return FrameSequence(frames=frames, fps=10.0, width=frames[0].width, height=frames[0].height, source_path="w100")


class CallCounter:
    """Wrap a model's test_step to record the batch size of every forward pass (instance attribute; class untouched)."""

    def __init__(self, model):
        self.model, self.calls = model, []
        orig = model.test_step

        def wrapped(data):
            self.calls.append(len(data["inputs"]))
            return orig(data)
        model.test_step = wrapped


def run_variant(est, frames):
    t = time.time()
    out = est.process_sequence(seq_of(frames))
    return np.stack([landmarks_array(f) for f in out.frames]), dict(est.last_tracking_report), time.time() - t


def detections_of(results):
    out = []
    for r in results:
        inst = r.pred_instances
        out.append([(tuple(float(v) for v in b), float(s)) for b, s, lab in zip(inst.bboxes.cpu().numpy(), inst.scores.cpu().numpy(),
                                                                                inst.labels.cpu().numpy()) if int(lab) == 0 and float(s) >= 0.3])
    return out


def select_track(dets):
    tr = SubjectTracker(None)
    return [tr.select([Detection(b, s) for b, s in d]) for d in dets]


def compare(a, b):
    """Landmark parity between two (F, 16, 4) arrays."""
    pa, pb = ~np.isnan(a[..., 0]), ~np.isnan(b[..., 0])
    both = pa & pb
    d = np.abs(a - b)
    flips = both & (a[..., 3] != b[..., 3])
    near = flips & (np.minimum(np.abs(a[..., 2] - 0.3), np.abs(b[..., 2] - 0.3)) <= 1e-3)
    return {"presence_mismatch": int((pa != pb).sum()), "max_xy_px": float(np.nanmax(d[..., :2][both])) if both.any() else 0.0,
            "max_conf": float(np.nanmax(d[..., 2][both])) if both.any() else 0.0, "visibility_flips": int(flips.sum()),
            "visibility_flips_near_threshold": int(near.sum()), "frames_with_landmarks": [int(pa.any(1).sum()), int(pb.any(1).sum())]}


def compare_dets(da, db):
    count = sum(len(x) != len(y) for x, y in zip(da, db))
    box = score = 0.0
    for x, y in zip(da, db):
        if len(x) == len(y):
            for (bx, sx), (by, sy) in zip(sorted(x), sorted(y)):
                box = max(box, max(abs(u - v) for u, v in zip(bx, by)))
                score = max(score, abs(sx - sy))
    return {"count_mismatch_frames": count, "max_box_px": box, "max_score": score}


def iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    ar = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ar if ar else 0.0


def population(path):
    rows = [r for r in json.loads(Path(path).read_text())["rows"] if r["population"].startswith("3dpw")]
    videos = defaultdict(dict)
    for r in rows:
        videos[r["sequence_id"].split(":")[1]][r["frame_index"]] = r["image"]
    return {v: [fr[k] for k in sorted(fr)] for v, fr in sorted(videos.items())}, rows


def gate(args):
    work = args.out_dir
    work.mkdir(parents=True, exist_ok=True)
    hist_mod, hist_sha = load_historical_adapter(work)
    videos, _ = population(args.population)
    report = {"predeclared": PREDECLARED, "historical_adapter_git_blob_sha256": hist_sha, "threads_at_init": threading.active_count(),
              "checkpoints": {}, "videos": {}, "controls": {}}
    cfg = config(8)
    for k, p in (("rtmdet_tiny", cfg.detector_checkpoint_path), ("rtmpose_tiny", cfg.checkpoint_path)):
        report["checkpoints"][k] = {"path": p, "sha256": sha256_file(p), "pinned_ok": sha256_file(p) == CHECKPOINT_SHA[k]}
        if not report["checkpoints"][k]["pinned_ok"]:
            raise SystemExit(f"STOP: untrusted checkpoint {p}")
    est = {b: prod.PoseEstimator(config(b)) for b in (1, 8)}
    hist = hist_mod.PoseEstimator(hist_mod.MMPoseConfig(cfg.config_path, cfg.checkpoint_path, device="cpu",
                                                        detector_config_path=cfg.detector_config_path,
                                                        detector_checkpoint_path=cfg.detector_checkpoint_path))
    for e in (*est.values(), hist):  # sequential initialisation in the main thread
        e._load_detector(), e._load_model()
    counters = {b: (CallCounter(est[b]._detector), CallCounter(est[b]._model)) for b in (1, 8)}

    def one(name, frames):
        res = {}
        for b in (1, 8):
            for c in counters[b]:
                c.calls.clear()
            arr, rep, dt = run_variant(est[b], frames)
            res[f"M{b}"] = (arr, rep, dt, list(counters[b][0].calls), list(counters[b][1].calls))
        arr2, rep2, _ = run_variant(est[8], frames)
        harr, hrep, hdt = run_variant(hist, frames)
        d1 = detections_of([r for f in frames for r in prod._detect_batch(est[1]._detector, [f.image])])
        d8 = detections_of([r for k in range(0, len(frames), 8) for r in prod._detect_batch(est[8]._detector, [f.image for f in frames[k:k + 8]])])
        s1, s8 = select_track(d1), select_track(d8)
        subj_iou = [iou(a.bbox, b.bbox) for a, b in zip(s1, s8) if a is not None and b is not None]
        out = {"frames": len(frames),
               "M1_vs_M8": compare(res["M1"][0], res["M8"][0]), "H_vs_M8": compare(harr, res["M8"][0]), "H_vs_M1": compare(harr, res["M1"][0]),
               "M8_repeat_bitwise_identical": bool(np.array_equal(res["M8"][0], arr2, equal_nan=True)) and rep2 == res["M8"][1],
               "tracking_reports": {"M1": res["M1"][1], "M8": res["M8"][1], "H": hrep},
               "seconds": {"M1": res["M1"][2], "M8": res["M8"][2], "H": hdt},
               "forward_batch_sizes": {f"M{b}": {"detector": sorted(set(res[f"M{b}"][3])), "pose": sorted(set(res[f"M{b}"][4])),
                                                 "detector_calls": len(res[f"M{b}"][3]), "pose_calls": len(res[f"M{b}"][4])} for b in (1, 8)},
               "detections_1_vs_8": compare_dets(d1, d8),
               "selected_subject": {"none_mismatch": sum((a is None) != (b is None) for a, b in zip(s1, s8)),
                                    "min_iou": float(min(subj_iou)) if subj_iou else None, "frames_compared": len(subj_iou)},
               "person_candidates_per_frame": float(np.mean([len(x) for x in d8]))}
        print(name, json.dumps({k: out[k] for k in ("M1_vs_M8", "H_vs_M8", "M8_repeat_bitwise_identical", "detections_1_vs_8", "selected_subject")}), flush=True)
        return out
    for v in PREDECLARED["gate_videos"]:
        report["videos"][v] = one(v, load_frames([args.image_root / p[len("3dpw/"):] for p in videos[v]]))
    wb = [args.image_root / p[len("3dpw/"):] for p in videos["downtown_walkBridge_01"][:24]]
    fg = [args.image_root / p[len("3dpw/"):] for p in videos["flat_guitar_01"][:12]]
    base = load_frames(wb)
    black = lambda f: Frame(index=0, timestamp=0, image=np.zeros_like(f.image), width=0, height=0)  # noqa: E731
    leave = [black(base[0]), black(base[0])] + base[:9] + [black(base[0]) for _ in range(6)] + base[9:]
    for i, f in enumerate(leave):
        f.index, f.timestamp = i, i / 10.0
    report["controls"]["leave_return_and_empty"] = one("leave_return_and_empty", leave)
    mixed = load_frames(wb[:12]) + load_frames(fg)
    for i, f in enumerate(mixed):
        f.index, f.timestamp = i, i / 10.0
    report["controls"]["mixed_image_sizes"] = one("mixed_image_sizes", mixed)
    (work / "gate_report.json").write_text(json.dumps(report, indent=1, default=str))


def observe(args):
    work = args.out_dir
    work.mkdir(parents=True, exist_ok=True)
    videos, rows = population(args.population)
    cfg = config(8)
    for k, p in (("rtmdet_tiny", cfg.detector_checkpoint_path), ("rtmpose_tiny", cfg.checkpoint_path)):
        if sha256_file(p) != CHECKPOINT_SHA[k]:
            raise SystemExit(f"STOP: untrusted checkpoint {p}")
    est = prod.PoseEstimator(cfg)
    est._load_detector(), est._load_model()
    manifest = {"predeclared": PREDECLARED, "threads_at_init": threading.active_count(), "videos": {}, "started": time.time()}
    for v, paths in videos.items():
        out_path = work / f"{v}.npz"
        if out_path.exists():
            manifest["videos"][v] = json.loads((work / f"{v}.json").read_text())
            continue
        t = time.time()
        files = [args.image_root / p[len("3dpw/"):] for p in paths]
        frames = load_frames(files)
        digests = [image_content_digest(f) for f in files]
        sizes = [f.image.shape[1::-1] for f in frames]
        lm, rep, dt = run_variant(est, frames)                       # M: production observation (tracked subject)
        dets = []
        for k in range(0, len(frames), 8):
            dets += detections_of(prod._detect_batch(est._detector, [f.image for f in frames[k:k + 8]]))
        cand_boxes, cand_lm, cand_frame = [], [], []
        jobs = [(i, b, s) for i, d in enumerate(dets) for b, s in d]
        for k in range(0, len(jobs), 8):
            chunk = jobs[k:k + 8]
            for (i, b, s), r in zip(chunk, prod._pose_batch(est._model, [(frames[i].image, b) for i, b, _ in chunk])):
                cand_frame.append(i)
                cand_boxes.append((*b, s))
                cand_lm.append(landmarks_array(type("PF", (), {"landmarks": prod._extract_canonical_landmarks([r], cfg.visibility_threshold)})()))
        np.savez_compressed(out_path, frame_index=np.array([int(Path(p).stem.split("_")[-1]) for p in paths]), landmarks=lm,
                            cand_frame=np.array(cand_frame, int), cand_box=np.array(cand_boxes, float).reshape(-1, 5),
                            cand_landmarks=np.array(cand_lm, float).reshape(-1, len(CANON), 4), image_size=np.array(sizes, int))
        meta = {"images": [p for p in paths], "image_sha256": digests, "tracking_report": rep, "seconds": time.time() - t,
                "observe_seconds": dt, "candidates": len(jobs)}
        (work / f"{v}.json").write_text(json.dumps(meta))
        manifest["videos"][v] = meta
        print(v, len(paths), f"{time.time() - t:.1f}s", json.dumps(rep), flush=True)
    manifest["checkpoints"] = {k: CHECKPOINT_SHA[k] for k in CHECKPOINT_SHA}
    manifest["canon"] = CANON
    manifest["population_sha256"] = sha256_file(args.population)
    manifest["software"] = {m: __import__(m).__version__ for m in ("mmpose", "mmdet", "mmengine", "torch", "cv2", "numpy")}
    (work / "manifest.json").write_text(json.dumps(manifest, indent=1, default=str))


def gate_detail(args):
    """Smallest reproduction of a gate failure: per-joint M1 vs M8 differences on one video, with the SimCC bin size of the
    tracked box mapped to image pixels (RTMPose-tiny: input 192x256, simcc_split_ratio 2 -> 0.5 input px per bin, box padding
    1.25 after aspect fix to 3:4)."""
    videos, _ = population(args.population)
    paths = [args.image_root / p[len("3dpw/"):] for p in videos[args.video]]
    frames = load_frames(paths)
    est = {b: prod.PoseEstimator(config(b)) for b in (1, 8)}
    for e in est.values():
        e._load_detector(), e._load_model()
    a1, _, _ = run_variant(est[1], frames)
    a8, _, _ = run_variant(est[8], frames)
    dets = [r for k in range(0, len(frames), 8) for r in detections_of(prod._detect_batch(est[8]._detector, [f.image for f in frames[k:k + 8]]))]
    sel = select_track(dets)
    rows = []
    for i in range(len(frames)):
        d = np.abs(a1[i, :, :2] - a8[i, :, :2]).max(1)
        x1, y1, x2, y2 = sel[i].bbox
        w, h = x2 - x1, y2 - y1
        w, h = (w, w / 0.75) if w / h > 0.75 else (h * 0.75, h)
        bin_px = 0.5 * 1.25 * w / 192.0
        for j in np.flatnonzero(d > 1e-3):
            rows.append({"frame_position": i, "frame_index": int(Path(paths[i]).stem.split("_")[-1]), "landmark": CANON[j],
                         "max_abs_diff_px": float(d[j]), "simcc_bin_px": bin_px, "diff_in_bins": float(d[j] / bin_px),
                         "conf_M1": float(a1[i, j, 2]), "conf_M8": float(a8[i, j, 2]), "batch_slot_M8": i % 8})
    out = {"video": args.video, "frames": len(frames), "joints_compared": int(np.isfinite(a1[..., 0]).sum()),
           "differing_joints": len(rows), "differing_frames": len({r["frame_position"] for r in rows}), "rows": rows}
    (args.out_dir / f"gate_detail_{args.video}.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("frames", "joints_compared", "differing_joints", "differing_frames")}))
    for r in rows:
        print(r)


def unseen(args):
    """Videos never used to select FramePose/S0: production observation (M8) -> benchmark representation -> frozen S0 (CPU).

    No GT exists; outputs feed qualitative review and GT-free temporal accounting only."""
    import torch

    sys.path.insert(0, str(ROOT / "scripts"))
    from eval_real_observation_estimator import to_input
    from framepose.crops import crop_box, geometry_in_crop
    from framepose.signs import neutral_sign_states
    from framepose.train import load_checkpoint, predict
    from mediaio.video_loader import VideoLoader

    if sha256_file(args.s0_checkpoint) != "6c12fea9f84ea0493fada8b9acc96cd5c70fdd5d9114603e954f9412f4247b44":
        raise SystemExit("STOP: S0 checkpoint identity differs")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cfg = config(8)
    est = prod.PoseEstimator(cfg)
    est._load_detector(), est._load_model()
    model, ck = load_checkpoint(args.s0_checkpoint, device="cpu")
    report = {}
    for video in args.videos:
        t = time.time()
        seq = VideoLoader().load_video(str(video))
        lm, rep, dt = run_variant(est, list(seq.frames))
        size = (seq.width, seq.height)
        x = np.zeros((len(lm), 17, 3), np.float32)
        v = np.zeros((len(lm), 17), bool)
        for i, a in enumerate(lm):
            x[i], v[i] = to_input(a if np.isfinite(a[:, 0]).any() else None, size)
        geo = np.stack([geometry_in_crop(x[i], v[i], size, crop_box(x[i], v[i], size)) for i in range(len(x))]).astype(np.float32)
        p0 = time.time()
        pred = predict(model, torch, geo, None, np.arange(len(x)), torch.device("cpu"), signs=neutral_sign_states(len(x)))
        name = Path(video).parent.name + "_" + Path(video).stem
        np.savez_compressed(args.out_dir / f"{name}.npz", landmarks=lm, input_2d=x, input_valid=v, s0=pred.astype(np.float32))
        report[name] = {"video": str(video), "video_sha256": sha256_file(video), "frames": len(lm), "fps": seq.fps, "size": list(size),
                        "frame_digests_sha256_of_decoded_arrays": [hashlib.sha256(f.image.tobytes()).hexdigest() for f in seq.frames],
                        "tracking_report": rep, "pose_seconds": dt, "s0_seconds": time.time() - p0, "total_seconds": time.time() - t,
                        "valid_joints_mean": float(v.sum(1).mean()), "frames_without_subject": int((~v.any(1)).sum())}
        print(name, json.dumps({k: report[name][k] for k in ("frames", "tracking_report", "pose_seconds", "s0_seconds")}), flush=True)
    (args.out_dir / "unseen_report.json").write_text(json.dumps(report, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=["gate", "gate-detail", "observe", "unseen"])
    ap.add_argument("--population", type=Path)
    ap.add_argument("--image-root", type=Path)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--videos", type=Path, nargs="*", default=[])
    ap.add_argument("--s0-checkpoint", type=Path)
    ap.add_argument("--video", default="downtown_bar_00")
    args = ap.parse_args()
    {"gate": gate, "gate-detail": gate_detail, "observe": observe, "unseen": unseen}[args.mode](args)


if __name__ == "__main__":
    main()
