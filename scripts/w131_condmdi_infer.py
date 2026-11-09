#!/usr/bin/env python3
"""W131 (EXPERIMENTAL, opt-in) CondMDI runner.  Runs ONLY in the scratch Python 3.12 env with ``-I``:

    C:/Users/dna10/animcv-w131/env/cm312/Scripts/python.exe -I scripts/w131_condmdi_infer.py <request.json>

The request JSON selects a mode:
  identity  verify every pinned HF artifact file against its LFS SHA-256, extract the official CondMDI release
            checkpoint from its zip and compare every network tensor (``model_avg`` and ``model``) with the HF safetensors.
  smoke     official independent pretrained smoke (text-to-motion from the model card example) plus a contract check of
            the observation-mask mechanism on that smoke output (masked-feature perturbation must not change the output).
            Not AnimCV quality evidence.
  infer     one CondMDI sample for a supplied standard (relative) HML263 tensor + absolute-space observation mask + text,
            via the official ``CondMDIPipeline._sample`` (imputation on, official CFG, DDIM-respaced sampler).  Optional
            ``perturb_masked_seed`` replaces every masked input feature with seeded noise (leak control).

Exit codes: 0 ok, 3 identity/terms stop (no inference), 4 contract failure.
"""
from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import hashlib  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import zipfile  # noqa: E402
from pathlib import Path  # noqa: E402

MOTIUS = Path("C:/Users/dna10/animcv-w131/ext/motius")
sys.path.insert(0, str(MOTIUS))

import numpy as np  # noqa: E402
import torch  # noqa: E402

SMOKE_TEXT = "a person walks forward and waves with the right hand"     # model-card example
SMOKE_LEN = 120


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dump(p, obj):
    Path(p).write_text(json.dumps(obj, indent=1, sort_keys=True, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))


def deterministic():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def env_report():
    import importlib.metadata as md
    rep = {"python": sys.version.split()[0], "torch": torch.__version__, "cuda": torch.version.cuda,
           "cudnn": torch.backends.cudnn.version(), "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
           "numpy": np.__version__}
    for pkg in ("clip", "einops", "safetensors", "mmengine"):
        try:
            rep[pkg] = md.version(pkg)
        except Exception:                                                # noqa: BLE001
            rep[pkg] = None
    rep["motius_head"] = (MOTIUS / ".git" / "HEAD").read_text().strip()
    return rep


def check_artifact(req):
    d = Path(req["hf_dir"])
    bad = {}
    for name, want in req["lfs_sha256"].items():
        p = d / name
        got = sha256_file(p) if p.is_file() else None
        if got != want:
            bad[name] = {"got": got, "want": want}
    return bad


def load_pipe(req):
    from motius.pipelines.condmdi import CondMDIPipeline
    t = time.time()
    pipe = CondMDIPipeline.from_pretrained(req["hf_dir"], bundle_kwargs={"respacing": req["respacing"]}, device="cuda")
    return pipe, time.time() - t


def mem():
    try:
        import psutil
        return psutil.Process().memory_info().rss
    except Exception:                                                    # noqa: BLE001
        return None


def sample(pipe, text, rel, mask_abs, seed, length, guidance):
    """Official sampler; returns absolute-root denormalised output [T,263] (float32) and the effective conditioning."""
    from motius.motion.representation.condmdi import relative_to_absolute
    from motius.motion.representation.normalization import affine_normalize
    if rel is None:
        out = pipe._sample([text], [length], guidance_param=guidance, impute=True, seed=seed, return_absolute=True)[0]
        return np.asarray(out, np.float32), None, None
    absolute = relative_to_absolute(torch.as_tensor(rel[None], dtype=torch.float32, device=pipe.device))
    obs_x0 = affine_normalize(absolute, pipe.bundle.mean_abs, pipe.bundle.std_abs).permute(0, 2, 1).unsqueeze(2)
    m = None if mask_abs is None else torch.as_tensor(mask_abs[None, :, None, :], dtype=torch.bool, device=pipe.device)
    eff = None if m is None else (obs_x0 * m).detach().cpu().numpy()[0, :, 0, :]
    out = pipe._sample([text], [length], motions=None if rel is None else [rel], observation_mask=None if m is None else m.cpu(),
                       guidance_param=guidance, impute=True, seed=seed, return_absolute=True)[0]
    return np.asarray(out, np.float32), eff, absolute.detach().cpu().numpy()[0]


def perturb_masked(rel, mask_abs, seed):
    """Replace every input feature that cannot reach an observed absolute feature with seeded noise.  Absolute
    features 0 (yaw) and 3 (height) depend only on relative 0 / 3; absolute 1,2 (xz) on relative 0..2; others 1:1."""
    rng = np.random.default_rng(seed)
    out = rel.copy()
    T = rel.shape[0]
    for f in range(4, 263):
        idx = ~mask_abs[f, :T]
        out[idx, f] = rng.normal(0.0, 3.0, idx.sum()).astype(np.float32)
    if not mask_abs[1:3].any():
        out[:, 1:3] = rng.normal(0.0, 3.0, (T, 2)).astype(np.float32)
    for f in (0, 3):
        if not mask_abs[f].any():
            out[:, f] = rng.normal(0.0, 3.0, T).astype(np.float32)
    return out


def mode_identity(req, out):
    rep = {"mode": "identity", "env": env_report()}
    bad = check_artifact(req)
    rep["hf_files_sha256_ok"] = not bad
    rep["hf_files_bad"] = bad
    from safetensors.torch import load_file
    hf = load_file(str(Path(req["hf_dir"]) / "model.safetensors"), device="cpu")
    zp = Path(req["official_zip"])
    rep["official_zip_sha256"] = sha256_file(zp)
    with zipfile.ZipFile(zp) as z:
        info = {i.filename: i for i in z.infolist()}
        member = req["official_member"]
        dst = out / "official_extract" / member
        dst.parent.mkdir(parents=True, exist_ok=True)
        with z.open(member) as src, open(dst, "wb") as f:
            while True:
                b = src.read(1 << 20)
                if not b:
                    break
                f.write(b)
        rep["official_member"] = {"name": member, "bytes": info[member].file_size, "crc32": f"{info[member].CRC:08x}",
                                  "sha256": sha256_file(dst)}
        args = json.loads(z.read("args.json"))
    rep["official_args"] = {k: args.get(k) for k in ("keyframe_conditioned", "keyframe_selection_scheme", "save_dir", "num_frames", "abs_3d")}
    ck = torch.load(str(dst), map_location="cpu", weights_only=True)
    rep["official_checkpoint_keys"] = sorted(ck.keys()) if isinstance(ck, dict) else str(type(ck))
    cmp = {}
    for rev in ("model_avg", "model"):
        if rev not in ck:
            continue
        sd = {k: v for k, v in ck[rev].items() if not k.startswith("clip_model.") and "sequence_pos_encoder" not in k}
        common = sorted(set(sd) & set(hf))
        eq = [k for k in common if sd[k].shape == hf[k].shape and torch.equal(sd[k].float(), hf[k].float())]
        diff = max((float((sd[k].float() - hf[k].float()).abs().max()) for k in common if sd[k].shape == hf[k].shape), default=None)
        cmp[rev] = {"official_tensors": len(sd), "hf_tensors": len(hf), "common": len(common), "bitwise_equal": len(eq),
                    "only_official": sorted(set(sd) - set(hf))[:10], "only_hf": sorted(set(hf) - set(sd))[:10],
                    "max_abs_diff": diff, "dtype_official": sorted({str(v.dtype) for v in sd.values()})}
    rep["tensor_comparison"] = cmp
    best = max(cmp, key=lambda r: cmp[r]["bitwise_equal"]) if cmp else None
    rep["identity_match"] = bool(best and cmp[best]["bitwise_equal"] == cmp[best]["hf_tensors"] == cmp[best]["official_tensors"])
    rep["identity_match_revision"] = best if rep["identity_match"] else None
    dump(out / "identity.json", rep)
    dst.unlink()                                                       # keep weights out of the evidence tree
    rep2 = {"identity_match": rep["identity_match"], "revision": rep["identity_match_revision"], "hf_ok": rep["hf_files_sha256_ok"],
            "cmp": {k: {kk: v[kk] for kk in ("bitwise_equal", "hf_tensors", "official_tensors", "max_abs_diff")} for k, v in cmp.items()},
            "official_args": rep["official_args"]}
    print(json.dumps(rep2, indent=1))
    return 0 if (rep["identity_match"] and rep["hf_files_sha256_ok"]) else 3


def mode_smoke(req, out):
    deterministic()
    rep = {"mode": "smoke", "env": env_report(), "note": "independent pretrained smoke; NOT AnimCV quality evidence"}
    bad = check_artifact(req)
    if bad:
        dump(out / "stop.json", {"stop": "BLOCKED_MODEL_IDENTITY", "bad": bad})
        return 3
    pipe, tl = load_pipe(req)
    rep["load_s"] = tl
    torch.cuda.reset_peak_memory_stats()
    t = time.time()
    a1, _, _ = sample(pipe, SMOKE_TEXT, None, None, 42, SMOKE_LEN, req["guidance"])
    rep["t2m_s"] = time.time() - t
    from motius.motion.representation.condmdi import absolute_to_relative, recover_from_absolute
    j = recover_from_absolute(torch.as_tensor(a1)).numpy()
    rel1 = absolute_to_relative(torch.as_tensor(a1)).numpy()
    np.savez(out / "smoke_t2m.npz", absolute=a1, relative=rel1, joints=j)
    bone = np.linalg.norm(j[:, 20] - j[:, 18], axis=1)
    rep["t2m"] = {"text": SMOKE_TEXT, "frames": int(a1.shape[0]), "finite": bool(np.isfinite(a1).all()),
                  "pelvis_height_mean_m": float(j[:, 0, 1].mean()), "travel_m": float(np.linalg.norm(j[-1, 0, [0, 2]] - j[0, 0, [0, 2]])),
                  "left_forearm_len_m": [float(bone.min()), float(bone.max())]}
    # observation-mask contract on the smoke motion: observe lower-body positions on the first 60 frames (pos only)
    mask = np.zeros((263, SMOKE_LEN), bool)
    for jn in (1, 2, 4, 5, 7, 8):
        mask[4 + 3 * (jn - 1): 4 + 3 * jn, :60] = True
    mask[0, :] = True
    mask[3, :] = True
    o1, e1, _ = sample(pipe, SMOKE_TEXT, rel1, mask, 7, SMOKE_LEN, req["guidance"])
    o2, e2, _ = sample(pipe, SMOKE_TEXT, rel1, mask, 7, SMOKE_LEN, req["guidance"])
    p = perturb_masked(rel1, mask, 99)
    o3, e3, _ = sample(pipe, SMOKE_TEXT, p, mask, 7, SMOKE_LEN, req["guidance"])
    rep["mask_contract"] = {"observed_features": int(mask.sum()), "rerun_bitwise": bool(np.array_equal(o1, o2)),
                            "perturbed_input_differs": bool(not np.array_equal(p, rel1)),
                            "effective_conditioning_bitwise": bool(np.array_equal(e1, e3)),
                            "output_bitwise_after_masked_perturbation": bool(np.array_equal(o1, o3)),
                            "max_abs_output_diff": float(np.abs(o1 - o3).max()),
                            "observed_features_imputed_max_err": float(np.abs(o1 - (np.asarray(
                                __import__("motius.motion.representation.condmdi", fromlist=["x"]).relative_to_absolute(torch.as_tensor(rel1)))))[mask.T].max())}
    rep["peak_vram_allocated_bytes"] = torch.cuda.max_memory_allocated()
    rep["rss_bytes"] = mem()
    ok = rep["t2m"]["finite"] and rep["mask_contract"]["output_bitwise_after_masked_perturbation"] and rep["mask_contract"]["effective_conditioning_bitwise"]
    rep["pass"] = bool(ok)
    dump(out / "result.json", rep)
    print(json.dumps({k: rep[k] for k in ("t2m", "mask_contract", "pass", "t2m_s", "load_s")}, indent=1))
    return 0 if ok else 4


def mode_infer(req, out):
    deterministic()
    rep = {"mode": "infer", "env": env_report(), "request": {k: v for k, v in req.items() if k not in ("lfs_sha256",)}}
    bad = check_artifact(req)
    if bad:
        dump(out / "stop.json", {"stop": "BLOCKED_MODEL_IDENTITY", "bad": bad})
        return 3
    z = np.load(req["input_npz"])
    rel, mask = z["relative_hml263"].astype(np.float32), z["observation_mask_abs"].astype(bool)
    if rel.shape != (req["length"], 263) or mask.shape != (263, req["length"]):
        dump(out / "stop.json", {"stop": "BAD_INPUT_SHAPE", "rel": rel.shape, "mask": mask.shape})
        return 4
    if req.get("perturb_masked_seed") is not None:
        rel = perturb_masked(rel, mask, int(req["perturb_masked_seed"]))
    pipe, tl = load_pipe(req)
    rep["load_s"] = tl
    rep["network_dtype"] = str(next(pipe.bundle.net.parameters()).dtype)
    rep["respacing"] = pipe.bundle.respacing
    rep["n_sampling_steps"] = int(pipe.diffusion.num_timesteps)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    rss0 = mem()
    t = time.time()
    out_abs, eff, abs_in = sample(pipe, req["text"], rel, mask, int(req["seed"]), int(req["length"]), float(req["guidance"]))
    torch.cuda.synchronize()
    rep["sample_s"] = time.time() - t
    rep["peak_vram_allocated_bytes"] = torch.cuda.max_memory_allocated()
    rep["peak_vram_reserved_bytes"] = torch.cuda.max_memory_reserved()
    rep["rss_bytes"] = [rss0, mem()]
    from motius.motion.representation.condmdi import absolute_to_relative, recover_from_absolute
    rel_out = absolute_to_relative(torch.as_tensor(out_abs)).numpy()
    joints = recover_from_absolute(torch.as_tensor(out_abs)).numpy()
    np.savez(out / "model_io.npz", input_relative=rel, input_absolute=abs_in, observation_mask_abs=mask,
             effective_conditioning_normalized=eff, output_absolute=out_abs, output_relative=rel_out, output_joints=joints)
    rep["imputed_observed_max_abs_err"] = float(np.abs(out_abs - abs_in)[mask.T].max())
    rep["output_finite"] = bool(np.isfinite(out_abs).all())
    rep["model_io_sha256"] = sha256_file(out / "model_io.npz")
    dump(out / "result.json", rep)
    print(json.dumps({k: rep[k] for k in ("sample_s", "load_s", "peak_vram_allocated_bytes", "imputed_observed_max_abs_err", "output_finite",
                                          "network_dtype", "n_sampling_steps")}, indent=1))
    return 0 if rep["output_finite"] else 4


def mode_parity(req, out):
    """Official Motius reference outputs for the representation-parity control (no model): joints_to_hml263 on a
    (T+1)-frame joint sequence, relative_to_absolute / recover_from_absolute on a supplied relative tensor, and the
    official target offsets."""
    from motius.motion.representation.condmdi import recover_from_absolute, relative_to_absolute
    from motius.motion.representation.humanml import HML263_TARGET_OFFSETS, joints_to_hml263
    z = np.load(req["input_npz"])
    feats = joints_to_hml263(z["joints"])
    ab = relative_to_absolute(torch.as_tensor(z["relative"], dtype=torch.float32)).numpy()
    jo = recover_from_absolute(torch.as_tensor(ab)).numpy()
    np.savez(out / "official.npz", hml263=feats, absolute=ab, joints=jo, target_offsets=HML263_TARGET_OFFSETS)
    dump(out / "result.json", {"mode": "parity", "env": env_report(), "hml263_shape": feats.shape})
    return 0


def main():
    req = json.loads(Path(sys.argv[1]).read_text())
    out = Path(req["out_dir"])
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}")
    out.mkdir(parents=True)
    return {"identity": mode_identity, "smoke": mode_smoke, "infer": mode_infer, "parity": mode_parity}[req["mode"]](req, out)


if __name__ == "__main__":
    sys.exit(main())
