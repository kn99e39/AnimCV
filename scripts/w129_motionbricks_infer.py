#!/usr/bin/env python3
"""W129 (EXPERIMENTAL, opt-in): run the pinned pretrained MotionBricks SOMA model through its OFFICIAL TensorRT engine.

Runs ONLY in the separate MotionBricks environment (Python >= 3.12, tensorrt-cu12 10.8.0.43, warp, usd-core; scratch
venv outside the AnimCV repository), started by ``scripts/w129_motionbricks_harness.py`` with ``python -I``.  It never
imports AnimCV code.  It refuses (exit 3 + ``stop.json``) instead of fabricating a result.

  python -I w129_motionbricks_infer.py <request.json>

Request modes
  skeleton   dump the bundled SOMA skeleton (names, parents, reference local transforms) for the AnimCV adapter.
  smoke      bundled independent example: walk-clip context/target -> one generation (runtime check ONLY, not AnimCV
             quality evidence; the bundled BVH is NVIDIA Sample Data, evaluation use).
  infer      ONE generation from the supplied 4 context + 4 target SOMA local poses with the declared configuration;
             saves the raw ONNX tensors separately from the decoded / endpoint-corrected engine output; optional
             ONNX Runtime CPU parity of the raw graph on the identical encoded inputs.
  skeleton_missing_joint   negative control: the engine must refuse a skeleton without LeftForeArm.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import threading
import time
from pathlib import Path


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def stop(out, code, detail):
    out.mkdir(parents=True, exist_ok=True)
    (out / "stop.json").write_text(json.dumps({"status": "STOP", "code": code, "detail": detail}, indent=1))
    print(f"STOP {code}: {detail}")
    sys.exit(3)


class VramPoll:
    def __init__(self):
        self.peak, self.base, self._run = None, None, True
        self.base = self._q()
        self.t = threading.Thread(target=self._loop, daemon=True)
        self.t.start()

    @staticmethod
    def _q():
        try:
            r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,name,driver_version", "--format=csv,noheader,nounits",
                                "-i", "0"], capture_output=True, text=True, timeout=10)
            used, name, drv = [x.strip() for x in r.stdout.strip().split(",")]
            return {"used_mib": int(used), "gpu": name, "driver": drv}
        except Exception as e:          # noqa: BLE001
            return {"error": repr(e)}

    def _loop(self):
        while self._run:
            q = self._q()
            if "used_mib" in q and (self.peak is None or q["used_mib"] > self.peak):
                self.peak = q["used_mib"]
            time.sleep(0.05)

    def done(self):
        self._run = False
        self.t.join(timeout=5)
        return {"gpu": self.base.get("gpu"), "driver": self.base.get("driver"), "used_before_mib": self.base.get("used_mib"),
                "peak_used_mib": self.peak, "note": "whole-GPU nvidia-smi memory.used polled every 50 ms (includes other processes)"}


def main():
    req = json.loads(Path(sys.argv[1]).read_text())
    out = Path(req["out_dir"])
    if out.exists():
        print(f"refusing to overwrite {out}")
        sys.exit(2)
    root = Path(req["motionbricks_root"])
    onnx = Path(req["onnx"])
    if req["mode"] != "skeleton":
        if not onnx.is_file():
            stop(out, "DATA_BLOCKED_MISSING_MODEL", str(onnx))
        head = onnx.open("rb").read(200)
        if head.startswith(b"version https://git-lfs"):
            stop(out, "DATA_BLOCKED_LFS_POINTER_NOT_WEIGHTS", str(onnx))
        got = sha256_file(onnx)
        if got != req["onnx_sha256"]:
            stop(out, "BLOCKED_MODEL_IDENTITY", f"sha256 {got} != declared {req['onnx_sha256']}")
        man = Path(req["manifest"])
        if sha256_file(man) != req["manifest_sha256"]:
            stop(out, "BLOCKED_MANIFEST_IDENTITY", str(man))
    sys.path.insert(0, str(root))
    t_import = time.time()
    import numpy as np
    import warp as wp
    wp.config.quiet = True
    wp.init()
    wp.set_device("cuda:0")
    from motionbricks.animation.skeleton import Skeleton
    from motionbricks.io.usd import load_skeleton_from_usd
    from motionbricks.io.utils import SOMA_SKELETON_PRIM_PATH, get_asset_file
    from motionbricks.model.config import MotionBricksInferenceConfig
    from motionbricks.model.contract import load_model_manifest
    from motionbricks.model.soma.engine import MotionBricksSOMAEngine
    usd = get_asset_file("human", "soma_base_skel_minimal_no_root.usd")
    skel = load_skeleton_from_usd(usd, SOMA_SKELETON_PRIM_PATH)
    out.mkdir(parents=True)
    import tensorrt
    env = {"python": sys.version, "platform": platform.platform(), "tensorrt": tensorrt.__version__, "warp": wp.__version__,
           "numpy": np.__version__, "soma_usd": str(usd), "soma_usd_sha256": sha256_file(usd),
           "import_s": time.time() - t_import}
    if req["mode"] == "skeleton":
        (out / "soma_skeleton.json").write_text(json.dumps({
            "names": list(skel.joint_names), "parents": skel.parent_indices.numpy().tolist(),
            "reference_local_xyz_qxyzw": skel.reference_local_transforms.numpy().tolist(), "source_usd_sha256": env["soma_usd_sha256"]}))
        (out / "result.json").write_text(json.dumps({"status": "OK", "env": env}, indent=1))
        return 0
    contract, fp32, _ = load_model_manifest(Path(req["manifest"]))
    if Path(fp32).resolve() != onnx.resolve():
        stop(out, "BLOCKED_MODEL_IDENTITY", f"manifest FP32 path {fp32} != {onnx}")
    if req["mode"] == "skeleton_missing_joint":
        names = list(skel.joint_names)
        k = names.index("LeftForeArm")
        names[k] = "LeftForeArm_RENAMED"
        bad = Skeleton(names, skel.parent_indices.numpy(), skel.reference_local_transforms)
        try:
            with MotionBricksSOMAEngine(onnx, cache_dir=req["cache_dir"]) as engine:
                engine.create_input(bad)
        except ValueError as e:
            (out / "result.json").write_text(json.dumps({"status": "REFUSED_AS_EXPECTED", "error": str(e), "env": env}, indent=1))
            return 0
        (out / "result.json").write_text(json.dumps({"status": "NOT_REFUSED", "env": env}, indent=1))
        return 4
    vram = VramPoll()
    t0 = time.time()
    with MotionBricksSOMAEngine(onnx, cache_dir=req["cache_dir"]) as engine, wp.ScopedStream(engine.stream):
        t_engine = time.time() - t0
        context = engine.create_input(skel)
        target = engine.create_input(skel)
        output = engine.create_output(skel)
        if req["mode"] == "smoke":
            from motionbricks.io.bvh import load_bvh
            clip_path = root / "assets" / "motions" / "soma" / "locomotion" / "bs_1d_linear" / "Neutral_walk_forward_002__A057.bvh"
            clip = load_bvh(str(clip_path), skel)
            cp = wp.empty_like(context.poses)
            tp = wp.empty_like(context.poses)
            for k in range(4):
                clip.sample(1.0 - (3 - k) / 30.0 + 0.0, cp[k], root_offset=wp.transform_identity(), stream=engine.stream)
                clip.sample(1.8 + k / 30.0, tp[k], root_offset=wp.transform_identity(), stream=engine.stream)
            context.update_from_poses(cp, stream=engine.stream)
            target.update_from_poses(tp, stream=engine.stream)
            env["smoke_clip"] = str(clip_path)
            env["smoke_clip_sha256"] = sha256_file(clip_path)
        else:
            ctx = np.asarray(req["context_local"], dtype=np.float32)
            tgt = np.asarray(req["target_local"], dtype=np.float32)
            if ctx.shape != (4, skel.num_joints, 7) or tgt.shape[1:] != (skel.num_joints, 7) or not 1 <= tgt.shape[0] <= 4:
                stop(out, "BAD_REQUEST_SHAPE", f"{ctx.shape} {tgt.shape}")
            context.update_from_poses(wp.array(ctx, dtype=wp.transform), stream=engine.stream)
            target.update_from_poses(wp.array(tgt, dtype=wp.transform), stream=engine.stream)
        cfg = MotionBricksInferenceConfig(allowed_token_counts=tuple(req.get("allowed_token_counts", (9,))),
                                          correct_generated_diff=bool(req.get("correct_generated_diff", True)),
                                          heading_alignment=tuple(req.get("heading_alignment", (1.0, 0.0))))
        t1 = time.time()
        engine.generate_animation(context, target, output, config=cfg)
        engine.synchronize()
        t_gen = time.time() - t1
        inputs = {n: engine.device_tensor(n).numpy() for n in ("global_root_values", "has_global_root_values", "local_root_values",
                                                               "has_local_root_values", "local_poses", "has_local_poses",
                                                               "allowed_pred_num_tokens")}
        raw = engine.device_tensor("pred_global_motions").numpy()
        ntok = engine.device_tensor("pred_num_tokens").numpy()
        nf = int(output.num_frames)
        dec = output.poses.numpy()[:nf]
        fc = output.foot_contacts.numpy()[:nf]
        start_d = engine._start_deltas.numpy() if req.get("correct_generated_diff", True) else None
        end_d = engine._end_deltas.numpy() if req.get("correct_generated_diff", True) else None
        timer = dict(engine.timer.milliseconds)
        plan_files = sorted(str(p) for p in Path(req["cache_dir"]).rglob("*") if p.is_file())
    vr = vram.done()
    np.savez(out / "raw_model_io.npz", pred_global_motions=raw, pred_num_tokens=ntok, **{"in_" + k: v for k, v in inputs.items()})
    np.savez(out / "engine_output.npz", local_poses=dec, foot_contacts=fc,
             start_deltas=start_d if start_d is not None else np.zeros(0), end_deltas=end_d if end_d is not None else np.zeros(0),
             context_local=context.poses.numpy(), target_local=target.poses.numpy())
    res = {"status": "OK", "mode": req["mode"], "env": env, "onnx": str(onnx), "onnx_sha256": sha256_file(onnx),
           "model_contract": contract.name, "engine_class": "motionbricks.model.soma.engine.MotionBricksSOMAEngine",
           "runtime": "TensorRT plan (official engine, CUDA graph)", "config": {"allowed_token_counts": list(cfg.allowed_token_counts),
           "correct_generated_diff": cfg.correct_generated_diff, "heading_alignment": list(cfg.heading_alignment)},
           "pred_num_tokens": int(ntok[0]), "raw_frames": int(ntok[0]) * 4, "valid_frames": nf, "target_num_poses": int(target.num_poses),
           "timing_s": {"engine_load_or_build": t_engine, "generate_wall": t_gen}, "stage_ms": timer, "vram": vr,
           "plan_cache_files": {p: sha256_file(p) for p in plan_files},
           "raw_model_io_sha256": sha256_file(out / "raw_model_io.npz"), "engine_output_sha256": sha256_file(out / "engine_output.npz")}
    if req.get("ort_crosscheck"):
        import onnxruntime as ort
        t2 = time.time()
        sess = ort.InferenceSession(str(onnx), providers=["CPUExecutionProvider"])
        ort_out = dict(zip([o.name for o in sess.get_outputs()], sess.run(None, inputs)))
        n_ort = int(ort_out["pred_num_tokens"][0])
        k = 4 * min(n_ort, int(ntok[0]))
        res["ort_cpu_parity"] = {"onnxruntime": ort.__version__, "pred_num_tokens": n_ort,
                                 "token_count_equal": n_ort == int(ntok[0]),
                                 "raw_max_abs_diff_valid": float(np.abs(ort_out["pred_global_motions"][0, :k] - raw[0, :k]).max()),
                                 "wall_s": time.time() - t2,
                                 "note": "same encoded inputs as the TensorRT run; parity diagnostic, not a second sample"}
        np.savez(out / "ort_cpu_raw.npz", **ort_out)
    (out / "result.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ("status", "pred_num_tokens", "valid_frames", "timing_s")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
