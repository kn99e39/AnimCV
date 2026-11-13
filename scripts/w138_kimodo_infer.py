#!/usr/bin/env python3
"""W138 (EXPERIMENTAL): Kimodo-side runner, executed ONLY by the scratch interpreter C:/Users/dna10/animcv-w138/env/km311.

The pinned Kimodo checkout (nv-tlabs/kimodo e1c219c) is put on sys.path; it is never modified or pip-installed.

Modes (each refuses an existing output directory):
  skeleton  NO MODEL: Kimodo SOMA77/SOMA30 names, parents, neutral joints, the 30<-77 slice, registry folder name, vendor
            sanitize("") and an FK sample for the numpy cross-check.
  encode    NO MODEL: vendor KimodoMotionRep (checkpoint stats) encode -> normalise -> unnormalise -> inverse of a given
            77-joint local-rotation clip; vendor FullBodyConstraintSet at the declared keyframes; vendor
            create_conditions_from_constraints_batched -> motion_mask; vendor compute_heading_angle at frame 0.
  infer     the ONE inference: vendor load_model(text_encoder=<null encoder>) from the hash-verified local checkpoint
            (HF offline), the same constraint construction as ``encode``, one official ``Kimodo.__call__`` with an empty
            prompt.  ``Kimodo._generate`` is wrapped by a read-only recorder that validates the received motion_mask BEFORE
            sampling (a wrong mask raises and nothing is sampled) and forwards identical arguments.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

KIMODO_ROOT = Path("C:/Users/dna10/animcv-w138/ext/kimodo")
sys.path.insert(0, str(KIMODO_ROOT))
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import numpy as np                                                    # noqa: E402
import torch                                                          # noqa: E402

import app.experimental_kimodo as K                                   # noqa: E402

MODEL_NAME = "Kimodo-SOMA-RP-v1.2"


def new_dir(p):
    p = Path(p)
    if p.exists():
        raise SystemExit(f"refusing to overwrite {p}")
    p.mkdir(parents=True)
    return p


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def dumpj(p, o):
    Path(p).write_text(json.dumps(o, indent=1, sort_keys=True, default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x)), encoding="utf-8")


class NullTextEncoder:
    """No-text mode.  Returns the vendor LLM2Vec output SHAPE ([B, 1, 4096], lengths 1) filled with zeros; the vendor
    ``_generate`` then zeroes the empty prompt's feature and its length anyway, so the denoiser input is identical to the
    vendor encoder's for an empty prompt.  Any non-empty prompt is refused (it would need the gated Llama 3 base)."""

    def __init__(self):
        self.calls = []

    def __call__(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        K.check_null_text(texts)
        self.calls.append(list(texts))
        return torch.zeros((len(texts), 1, K.TEXT_DIM)), [1] * len(texts)


def build_constraint(skel30, local77, root_c, key_idx, device):
    """Vendor FullBodyConstraintSet on SOMA30 from a 77-joint local clip (vendor 77->30 slice + vendor FK)."""
    from kimodo.constraints import FullBodyConstraintSet
    from kimodo.motion_rep.feature_utils import compute_heading_angle
    K.check_keyframes(key_idx)
    K.check_canonical(root_c)
    device = skel30.neutral_joints.device                              # build on the skeleton's own device (as the vendor loader does)
    l77 = torch.tensor(local77, dtype=torch.float32, device=device)
    root = torch.tensor(root_c, dtype=torch.float32, device=device)
    l30 = skel30.from_SOMASkeleton77(l77)
    g30, p30, _ = skel30.fk(l30, root)
    kt = torch.tensor(key_idx, device=device)
    cons = FullBodyConstraintSet(skel30, kt, p30[kt], g30[kt])
    heading0 = compute_heading_angle(p30[None], skel30)[0, 0]          # [B, T] -> frame 0 of the single clip (scalar)
    return cons, l30, g30, p30, heading0


def mode_skeleton(a):
    out = new_dir(a.out_dir)
    from kimodo.model.registry import MODEL_NAMES, get_model_info, resolve_model_name
    from kimodo.sanitize import sanitize_texts
    from kimodo.skeleton import SOMASkeleton30, SOMASkeleton77
    s30, s77 = SOMASkeleton30(), SOMASkeleton77()
    rng = np.random.default_rng(0)
    q = rng.normal(size=(3, 77, 4))
    q /= np.linalg.norm(q, axis=-1, keepdims=True)
    from kimodo.geometry import axis_angle_to_matrix
    aa = torch.tensor(rng.normal(scale=0.4, size=(3, 77, 3)), dtype=torch.float64)
    R = axis_angle_to_matrix(aa)
    root = torch.tensor(rng.normal(size=(3, 3)), dtype=torch.float64)
    g, p, _ = s77.fk(R, root)
    try:
        key = resolve_model_name(MODEL_NAME, "Kimodo")
    except Exception as e:                                            # noqa: BLE001
        key = f"ERROR {e}"
    info = get_model_info(key) if not str(key).startswith("ERROR") else None
    np.savez(out / "skeleton.npz", names77=np.array(s77.bone_order_names), parents77=s77.joint_parents.cpu().numpy(),
             neutral77=s77.neutral_joints.cpu().numpy(), names30=np.array(s30.bone_order_names), parents30=s30.joint_parents.cpu().numpy(),
             neutral30=s30.neutral_joints.cpu().numpy(), slice30=np.array(s30.get_skel_slice(s77)),
             fk_local=R.numpy(), fk_root=root.numpy(), fk_pos=p.numpy(), fk_glob=g.numpy())
    rep = {"names30_equal_model_joints": list(s30.bone_order_names) == list(K.SOMA30), "hip_joint_idx30": [int(x) for x in s30.hip_joint_idx],
           "root_idx": int(s30.root_idx), "sanitize_empty": sanitize_texts(["", "   "]), "resolved_key": key,
           "display_name": getattr(info, "display_name", None), "repo_id": MODEL_NAMES.get(key) if isinstance(key, str) else None,
           "torch": torch.__version__, "cuda": torch.version.cuda, "available": torch.cuda.is_available(),
           "arch": torch.cuda.get_arch_list(), "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
           "numpy": np.__version__}
    dumpj(out / "skeleton.json", rep)
    print(json.dumps(rep, indent=1))
    return 0


def _motion_rep(ckpt_dir, device):
    from kimodo.motion_rep import KimodoMotionRep
    from kimodo.skeleton import SOMASkeleton30
    return KimodoMotionRep(skeleton=SOMASkeleton30().to(device), fps=K.FPS, stats_path=str(Path(ckpt_dir) / "stats" / "motion"))


def mode_encode(a):
    """NO MODEL: vendor feature roundtrip + vendor conditioning contract on the given clip."""
    out = new_dir(a.out_dir)
    device = "cpu"
    z = np.load(a.inp)
    local77, root_c, key_idx = z["local77"], z["root_canonical"], [int(i) for i in z["key_idx"]]
    K.check_fps_frames(K.FPS, len(local77))
    mr = _motion_rep(a.checkpoint_dir, device)
    skel30 = mr.skeleton
    cons, l30, g30, p30, heading = build_constraint(skel30, local77, root_c, key_idx, device)
    lengths = torch.tensor([K.N_WIN])
    feats_n = mr(l30[None], torch.tensor(root_c, dtype=torch.float32)[None], to_normalize=True, lengths=lengths)
    inv = mr.inverse(feats_n, is_normalized=True, return_numpy=False, return_all=True)
    l77_out = skel30.to_SOMASkeleton77(inv["local_rot_mats"][0])
    obs, mask = mr.create_conditions_from_constraints_batched([cons], lengths, to_normalize=True, device=device)
    mrep = K.validate_motion_mask(mask.cpu().numpy())
    from kimodo.constraints import save_constraints_lst
    save_constraints_lst(str(out / "constraints.json"), [cons])
    np.savez(out / "encoded.npz", features_normalized=feats_n[0].numpy(), local77_out=l77_out.numpy(), root_out=inv["root_positions"][0].numpy(),
             posed30_rot=inv["posed_joints"][0].numpy(), posed30_pos=inv["all_positions"]["positions"][0].numpy(),
             fk30_input=p30.numpy(), motion_mask=mask[0].numpy(), observed_motion=obs[0].numpy(), first_heading_angle=float(heading))
    rep = {"mask": mrep, "first_heading_angle_rad": float(heading), "constraints_sha256": sha(out / "constraints.json"),
           "roundtrip_local30_max_abs": float((inv["local_rot_mats"][0] - l30).abs().max()),
           "roundtrip_posed30_vs_fk_max_m": float((inv["posed_joints"][0] - p30).abs().max()),
           "positions_block_vs_rotations_max_m": float((inv["all_positions"]["positions"][0] - inv["posed_joints"][0]).abs().max()),
           "encoded_sha256": sha(out / "encoded.npz")}
    dumpj(out / "encode.json", rep)
    print(json.dumps(rep, indent=1))
    return 0


def mode_infer(a):
    out = new_dir(a.out_dir)
    os.environ["CHECKPOINT_DIR"] = str(Path(a.checkpoint_dir).parent)
    rec = {"args": {"prompt": "", "num_frames": K.N_WIN, "num_denoising_steps": a.steps, "cfg_type": "separated", "cfg_weight": [2.0, 2.0],
                    "post_processing": False, "num_samples": None, "seed": a.seed, "model": MODEL_NAME}, "calls": 0}
    z = np.load(a.inp)
    local77, root_c, key_idx = z["local77"], z["root_canonical"], [int(i) for i in z["key_idx"]]
    K.check_fps_frames(K.FPS, len(local77))
    K.check_null_text(rec["args"]["prompt"])
    device = "cuda:0"
    from kimodo import load_model
    from kimodo.tools import seed_everything
    t0 = time.time()
    enc = NullTextEncoder()
    model = load_model(MODEL_NAME, device=device, text_encoder=enc)
    rec["load_seconds"] = round(time.time() - t0, 3)
    rec["model_fps"] = float(model.fps)
    rec["model_skeleton"] = type(model.skeleton).__name__
    rec["text_encoder_type"] = type(model.text_encoder).__name__
    K.check_fps_frames(model.fps, len(local77))
    cons, l30, g30, p30, heading = build_constraint(model.skeleton, local77, root_c, key_idx, device)
    orig = model._generate
    cap = {}

    def recorder(*args, **kw):
        mm = kw.get("motion_mask")
        rec["mask_validate"] = K.validate_motion_mask(mm.detach().cpu().numpy())      # BEFORE sampling; raises on mismatch
        rec["calls"] += 1
        if rec["calls"] != 1:
            raise RuntimeError("second _generate call refused (one inference only)")
        cap["motion_mask"] = mm.detach().cpu().numpy()
        cap["observed_motion"] = kw["observed_motion"].detach().cpu().numpy()
        cap["first_heading_angle"] = kw["first_heading_angle"].detach().cpu().numpy()
        torch.cuda.synchronize()
        ts = time.time()
        res = orig(*args, **kw)
        torch.cuda.synchronize()
        rec["sampler_seconds"] = round(time.time() - ts, 3)
        cap["features_normalized"] = res.detach().cpu().numpy()
        return res
    model._generate = recorder
    torch.cuda.reset_peak_memory_stats()
    seed_everything(a.seed)
    ts = time.time()
    output = model("", K.N_WIN, num_denoising_steps=a.steps, constraint_lst=[cons], cfg_weight=[2.0, 2.0], cfg_type="separated",
                   num_samples=None, return_numpy=False, first_heading_angle=float(heading), post_processing=False, progress_bar=lambda x: x)
    torch.cuda.synchronize()
    rec["call_seconds"] = round(time.time() - ts, 3)
    rec["peak_vram_bytes"] = int(torch.cuda.max_memory_allocated())
    rec["text_encoder_calls"] = enc.calls
    rec["device"] = torch.cuda.get_device_name(0)
    rec["torch"] = torch.__version__
    rec["tf32_matmul"] = bool(torch.backends.cuda.matmul.allow_tf32)
    rec["first_heading_angle_rad"] = float(heading)
    # decode of the captured features (deterministic, no further model call): 30-joint view incl. the positions block
    feats = torch.tensor(cap["features_normalized"], device=device)
    inv = model.motion_rep.inverse(feats, is_normalized=True, return_numpy=False, return_all=True)
    o = {k: v.detach().cpu().numpy() for k, v in output.items() if torch.is_tensor(v)}
    np.savez(out / "raw_model_io.npz", in_local77=local77, in_root_canonical=root_c, in_key_idx=np.array(key_idx),
             in_constraint_pos30=p30[torch.tensor(key_idx, device=device)].cpu().numpy(), in_motion_mask=cap["motion_mask"],
             in_observed_motion=cap["observed_motion"], in_first_heading_angle=cap["first_heading_angle"],
             out_features_normalized=cap["features_normalized"], out_local77=o["local_rot_mats"], out_global77=o["global_rot_mats"],
             out_posed77=o["posed_joints"], out_root=o["root_positions"], out_smooth_root=o["smooth_root_pos"],
             out_foot_contacts=o["foot_contacts"], out_local30=inv["local_rot_mats"][0].cpu().numpy(),
             out_posed30_rot=inv["posed_joints"][0].cpu().numpy(), out_posed30_pos=inv["all_positions"]["positions"][0].cpu().numpy())
    rec["raw_model_io_sha256"] = sha(out / "raw_model_io.npz")
    rec["output_shapes"] = {k: list(v.shape) for k, v in o.items()}
    rec["total_seconds"] = round(time.time() - t0, 3)
    dumpj(out / "infer_record.json", rec)
    print(json.dumps({k: rec[k] for k in ("calls", "mask_validate", "sampler_seconds", "call_seconds", "peak_vram_bytes", "text_encoder_calls",
                                          "raw_model_io_sha256")}, indent=1))
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("mode", choices=("skeleton", "encode", "infer"))
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--inp")
    ap.add_argument("--checkpoint-dir")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--steps", type=int, default=100)
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    return {"skeleton": mode_skeleton, "encode": mode_encode, "infer": mode_infer}[a.mode](a)


if __name__ == "__main__":
    sys.exit(main())
