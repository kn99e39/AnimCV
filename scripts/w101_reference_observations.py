#!/usr/bin/env python3
"""W101 isolated CPU M1 gate/observation runner. Never executes the M8 condition.

Run on LabServer63, in the original FramePose image plus isolated MMPose dependencies.
Historical W100 helpers are reused for serialization and instrumentation only.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import threading
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
import run_real_observation_mmpose as w100
from framepose.observations import image_content_digest, mmpose_observation, observation_cache_key
from mediaio.frame_sequence import Frame, FrameSequence
from pose import mmpose_adapter as prod
from pose.default_detector import get_default_detector_config_path
from pose.default_model import get_default_pose_config_path

HISTORICAL_SHA = "0582a2a4f7653f1ffa5acbb4640b193ec2ec1909ec023cf428dba4e4bc962413"
CONFIG_SHA = {"pose": "7a8e5057017515237274334b367c1fb87b2af6ea03ac77e8a852715794c923fa",
              "detector": "873923f8221c2e5ae6501dae248eddf26743daeccb82ae9055cb6435fcabc992"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, default=str))


def config(weights):
    return prod.MMPoseConfig(
        get_default_pose_config_path(),
        str(weights / "rtmpose-tiny_simcc-coco_pt-aic-coco_420e-256x192-e613ba3f_20230127.pth"),
        device="cpu", visibility_threshold=0.3, subject_box=None,
        detector_config_path=get_default_detector_config_path(),
        detector_checkpoint_path=str(weights / "rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth"),
        batch_size=1)


def initialize(cfg, module=prod):
    if threading.current_thread() is not threading.main_thread() or threading.active_count() != 1:
        raise RuntimeError("STOP: model initialization is not sequential in the sole main Python thread")
    if module is prod:
        est = module.PoseEstimator(cfg)
    else:
        values = asdict(cfg)
        values.pop("batch_size")
        est = module.PoseEstimator(module.MMPoseConfig(**values))
    est._load_detector()
    est._load_model()
    return est


def identity(cfg):
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    checks = {"rtmdet_tiny": sha(cfg.detector_checkpoint_path), "rtmpose_tiny": sha(cfg.checkpoint_path)}
    if checks != w100.CHECKPOINT_SHA:
        raise RuntimeError("STOP: trusted checkpoint identity failed")
    config_checks = {"pose": sha(cfg.config_path), "detector": sha(cfg.detector_config_path)}
    if config_checks != CONFIG_SHA:
        raise RuntimeError("STOP: trusted configuration source identity failed")
    import mmdet, mmpose, mmcv, mmengine
    return {"config": asdict(cfg), "checkpoints_sha256": checks,
            "config_files_sha256": config_checks,
            "software": {"torch": torch.__version__, "mmpose": mmpose.__version__, "mmdet": mmdet.__version__,
                         "mmcv": mmcv.__version__, "mmengine": mmengine.__version__, "numpy": np.__version__, "cv2": cv2.__version__},
            "threads_at_initialization": threading.active_count(), "torch_threads": torch.get_num_threads()}


def sequence(paths, image_root):
    frames = []
    for p in paths:
        fr = int(Path(p).stem.split("_")[-1])
        im = cv2.imread(str(image_root / p.removeprefix("3dpw/")))
        if im is None:
            raise RuntimeError(f"STOP: unreadable image {p}")
        h, w = im.shape[:2]
        frames.append(Frame(index=fr, timestamp=fr / 30.0, image=im, width=w, height=h))
    return FrameSequence(frames=frames, fps=30., width=frames[0].width, height=frames[0].height, source_path="w101")


def run(est, seq):
    start = time.monotonic()
    output = est.process_sequence(seq)
    order = [(int(f.frame_index), float(f.timestamp)) for f in output.frames]
    expected = [(int(f.index), float(f.timestamp)) for f in seq.frames]
    if order != expected:
        raise RuntimeError("STOP: frame ordering/timestamp mismatch")
    return np.stack([w100.landmarks_array(f) for f in output.frames]), dict(est.last_tracking_report), time.monotonic() - start


def gate(args, cfg, info):
    if args.historical_adapter:
        hsha = sha(args.historical_adapter)
        if hsha != HISTORICAL_SHA:
            raise RuntimeError("STOP: historical adapter identity failed")
        spec = importlib.util.spec_from_file_location("w101_historical_mmpose_adapter", args.historical_adapter)
        historical = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = historical
        spec.loader.exec_module(historical)
    else:
        historical, hsha = w100.load_historical_adapter(args.out_dir)
        if hsha != HISTORICAL_SHA:
            raise RuntimeError("STOP: historical git adapter identity failed")
    est = initialize(cfg)
    hist = initialize(cfg, historical)
    counters = [w100.CallCounter(est._detector), w100.CallCounter(est._model)]
    # Resolved configs must be identical on both paths; hashes bind executable preprocessing/configuration.
    loaded = {}
    for key, a, b in (("pose", est._model, hist._model), ("detector", est._detector, hist._detector)):
        ca, cb = a.cfg.pretty_text, b.cfg.pretty_text
        if ca != cb:
            raise RuntimeError("STOP: resolved configuration differs between historical and M1")
        loaded[key] = hashlib.sha256(ca.encode()).hexdigest()
        (args.out_dir / f"resolved_{key}_config.py").write_text(ca)
    videos, _ = w100.population(args.population)
    cases = {}
    images = {}
    for video in w100.PREDECLARED["gate_videos"]:
        paths = videos[video][:12]
        if video == "downtown_bar_00":
            paths = sorted(set(paths + [p for p in videos[video] if int(Path(p).stem.split("_")[-1]) in (1158, 1168)]))
        cases[video] = sequence(paths, args.image_root)
        images[video] = [{"path": p, "sha256": image_content_digest(args.image_root / p.removeprefix("3dpw/"))} for p in paths]
    base = cases["downtown_walkBridge_01"].frames[:6]
    control = [Frame(index=0, timestamp=0., image=np.zeros_like(base[0].image), width=base[0].width, height=base[0].height)]
    control += base[:3] + [Frame(index=0, timestamp=0., image=np.zeros_like(base[0].image), width=base[0].width, height=base[0].height)] + base[3:]
    control += cases["flat_guitar_01"].frames[:3]
    control = [Frame(index=i, timestamp=i/30., image=f.image, width=f.width, height=f.height) for i, f in enumerate(control)]
    cases["black_leave_return_portrait_landscape"] = FrameSequence(
        frames=control, fps=30., width=control[0].width, height=control[0].height, source_path="w101_synthetic_control")
    report = {**info, "historical_blob_sha256": hsha, "resolved_config_sha256": loaded, "images": images, "cases": {}}
    ok = True
    for key, seq in cases.items():
        a, ar, at = run(est, seq)
        b, br, bt = run(hist, seq)
        repeat, rr, rt = run(est, seq)
        equal = bool(np.array_equal(a, b, equal_nan=True) and np.array_equal(a, repeat, equal_nan=True) and ar == br == rr)
        report["cases"][key] = {"frames": len(seq.frames), "bitwise_historical_and_repeat": equal,
                                 "comparison": w100.compare(a, b), "tracking": ar,
                                 "seconds_M1_H_repeat": [at, bt, rt]}
        np.savez_compressed(args.out_dir / f"{key}.npz", M1=a, H=b, repeat=repeat,
                            frame_index=[f.index for f in seq.frames], timestamp=[f.timestamp for f in seq.frames])
        ok &= equal
        print(key, equal, flush=True)
    report["detector_forward_batch_sizes"] = counters[0].calls
    report["pose_forward_batch_sizes"] = counters[1].calls
    ok &= bool(counters[0].calls and counters[1].calls and set(counters[0].calls + counters[1].calls) == {1})
    report["pass"] = bool(ok)
    dump(args.out_dir / "gate_report.json", report)
    if not ok:
        raise RuntimeError("STOP: M1 historical-equivalence reference gate failed")


def observe(args, cfg, info):
    gate_info = json.loads(args.gate_report.read_text())
    if not gate_info["pass"] or gate_info["software"] != info["software"] or gate_info["checkpoints_sha256"] != info["checkpoints_sha256"]:
        raise RuntimeError("STOP: missing or incompatible M1 gate")
    est = initialize(cfg)
    for key, model in (("pose", est._model), ("detector", est._detector)):
        if hashlib.sha256(model.cfg.pretty_text.encode()).hexdigest() != gate_info["resolved_config_sha256"][key]:
            raise RuntimeError("STOP: model configuration changed after gate")
    provenance = mmpose_observation(
        pose_config=Path(cfg.config_path).name, pose_checkpoint=Path(cfg.checkpoint_path).name,
        detector_config=Path(cfg.detector_config_path).name, detector_checkpoint=Path(cfg.detector_checkpoint_path).name,
        visibility_threshold=cfg.visibility_threshold, input_size="192x256", mmpose_version=info["software"]["mmpose"],
        pose_weights_sha256=info["checkpoints_sha256"]["rtmpose_tiny"],
        detector_weights_sha256=info["checkpoints_sha256"]["rtmdet_tiny"],
        extra={"batch_size": 1, "device": "cpu", "resolved_config_sha256": gate_info["resolved_config_sha256"], "software": info["software"]})
    videos, _ = w100.population(args.population)
    manifest = {**info, "population_sha256": sha(args.population), "gate_report_sha256": sha(args.gate_report),
                "provenance": provenance.to_dict(), "videos": {}}
    dump(args.out_dir / "manifest.json", manifest)
    for video, paths in videos.items():
        start = time.monotonic()
        seq = sequence(paths, args.image_root)
        digests = [image_content_digest(args.image_root / p.removeprefix("3dpw/")) for p in paths]
        # Observe, without changing, the actual production detector returns. Avoid a second detector pass and bind
        # the privileged candidates to exactly the detections seen by the real tracker.
        recorded_detections = []
        original_step = est._detector.test_step
        def record_step(data):
            result = original_step(data)
            recorded_detections.extend(result)
            return result
        est._detector.test_step = record_step
        try:
            lm, tracking, seconds = run(est, seq)
        finally:
            est._detector.test_step = original_step
        if len(recorded_detections) != len(seq.frames):
            raise RuntimeError("STOP: detector/frame instrumentation mismatch")
        dets = w100.detections_of(recorded_detections)
        boxes, cand, cf = [], [], []
        for i, detections in enumerate(dets):
            for box, score in detections:
                result = prod._pose_batch(est._model, [(seq.frames[i].image, box)])[0]
                cand.append(w100.landmarks_array(type("PF", (), {"landmarks": prod._extract_canonical_landmarks([result], cfg.visibility_threshold)})()))
                boxes.append((*box, score))
                cf.append(i)
        np.savez_compressed(args.out_dir / f"{video}.npz", frame_index=[f.index for f in seq.frames],
                            timestamp=[f.timestamp for f in seq.frames], landmarks=lm, cand_frame=np.array(cf, int),
                            cand_box=np.asarray(boxes).reshape(-1, 5), cand_landmarks=np.asarray(cand).reshape(-1, 16, 4),
                            image_size=[(f.width, f.height) for f in seq.frames])
        meta = {"images": paths, "image_sha256": digests, "provenance": provenance.to_dict(),
                "observation_cache_keys": [observation_cache_key(provenance, d) for d in digests],
                "tracking_report": tracking, "observe_seconds": seconds, "total_seconds": time.monotonic()-start,
                "candidates": len(cf), "npz_sha256": sha(args.out_dir / f"{video}.npz")}
        dump(args.out_dir / f"{video}.json", meta)
        manifest["videos"][video] = meta
        dump(args.out_dir / "manifest.json", manifest)
        print(video, len(paths), meta["total_seconds"], flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=("gate", "observe"))
    for key in ("population", "image_root", "weights", "out_dir"):
        ap.add_argument("--" + key.replace("_", "-"), type=Path, required=True)
    ap.add_argument("--gate-report", type=Path)
    ap.add_argument("--historical-adapter", type=Path, help="Hash-pinned pre-W99A source for a branch-free remote snapshot")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=False)
    cfg = config(args.weights)
    info = identity(cfg)
    (gate if args.mode == "gate" else observe)(args, cfg, info)


if __name__ == "__main__":
    main()
