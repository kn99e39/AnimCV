#!/usr/bin/env python3
"""W132 (EXPERIMENTAL, opt-in) CondMDI causal runner.  Runs ONLY in the W131 scratch Python 3.12 env with ``-I``:

    C:/Users/dna10/animcv-w131/env/cm312/Scripts/python.exe -I scripts/w132_condmdi_causal_infer.py <request.json>

The unchanged W131 runner module (``scripts/w131_condmdi_infer.py``) is loaded by path and reused for artifact hash
checks, deterministic CUDA settings, pipeline loading and masked-feature perturbation.  W132 adds exactly one switch,
``impute`` (the official ``CondMDIPipeline._sample(impute=...)`` argument), plus NON-INVASIVE instrumentation:

  * a forward pre-hook / hook on the CondMDI network records, for every network call, the SHA-256 of the diffusion state
    ``x``, of ``obs_x0`` and ``obs_mask`` (the learned-conditioning inputs), of the CLIP text tokens, the ``y`` option
    keys, and the SHA-256 of the first four network outputs;
  * ``editing.requires_imputation`` / ``requires_reconstruction_guidance`` (looked up at call time by the official
    ``gaussian_diffusion.p_mean_variance``) are wrapped to log their per-step return values.

Modes:
  causal  one S1 sample from a supplied relative HML263 tensor + absolute-space mask (W131 ``conditioning*.npz``),
          ``impute`` true/false, optional ``perturb_masked_seed`` (W131 leak-control semantics).
  refctl  one reference-control cell: a frozen complete HML263 reference, the OFFICIAL ``build_observation_mask``
          (mode ``middle``, all 22 joints, ``feature_mode`` pos_rot_vel | pos), ``impute`` true/false.

Exit codes: 0 ok, 3 identity stop (no inference), 4 contract failure.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

_spec = importlib.util.spec_from_file_location("w131_infer", str(Path(__file__).resolve().parent / "w131_condmdi_infer.py"))
w131 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(w131)                                         # sets CUBLAS_WORKSPACE_CONFIG, Motius path, torch

np, torch = w131.np, w131.torch


def tsha(t):
    return hashlib.sha256(t.detach().contiguous().cpu().numpy().tobytes()).hexdigest()


class Instrument:
    """Records the conditioning inputs of every network call and the per-step imputation decisions."""

    def __init__(self, net):
        from motius.diffusion.condmdi import editing
        self.editing = editing
        self.log = {"n_net_calls": 0, "x_sha": [], "obs_x0_sha": [], "obs_mask_sha": [], "text_tokens_sha": [], "y_keys": set(),
                    "uncond_calls": 0, "out_sha": [], "requires_imputation": [], "requires_reconstruction_guidance": []}
        self._ri, self._rg = editing.requires_imputation, editing.requires_reconstruction_guidance

        def ri(mk, t):
            r = self._ri(mk, t)
            self.log["requires_imputation"].append(bool(r))
            return r

        def rg(mk, t):
            r = self._rg(mk, t)
            self.log["requires_reconstruction_guidance"].append(bool(r))
            return r
        editing.requires_imputation, editing.requires_reconstruction_guidance = ri, rg

        def pre(mod, args, kwargs):
            x, ts = args[0], args[1]
            y = args[2] if len(args) > 2 else kwargs.get("y")
            ox = args[3] if len(args) > 3 else kwargs.get("obs_x0")
            om = args[4] if len(args) > 4 else kwargs.get("obs_mask")
            L = self.log
            L["n_net_calls"] += 1
            L["x_sha"].append(tsha(x))
            L["obs_x0_sha"].append(None if ox is None else tsha(ox))
            L["obs_mask_sha"].append(None if om is None else tsha(om))
            L["text_tokens_sha"].append(tsha(y["text_tokens"]))
            L["y_keys"] |= {k for k in y.keys() if k != "uncond"}
            L["uncond_calls"] += int(bool(y.get("uncond", False)))

        def post(mod, args, kwargs, out):
            if len(self.log["out_sha"]) < 4:
                self.log["out_sha"].append(tsha(out))
        self.h1 = net.register_forward_pre_hook(pre, with_kwargs=True)
        self.h2 = net.register_forward_hook(post, with_kwargs=True)

    def close(self):
        self.h1.remove()
        self.h2.remove()
        self.editing.requires_imputation, self.editing.requires_reconstruction_guidance = self._ri, self._rg
        L = dict(self.log)
        L["obs_x0_sha"] = sorted(set(L["obs_x0_sha"]), key=str)
        L["obs_mask_sha"] = sorted(set(L["obs_mask_sha"]), key=str)
        L["text_tokens_sha"] = sorted(set(L["text_tokens_sha"]))
        L["y_keys"] = sorted(L["y_keys"])
        return L


def sample(pipe, text, rel, mask_abs, seed, length, guidance, impute):
    """W131 ``sample`` with the single ``impute`` switch (identical tensor construction)."""
    from motius.motion.representation.condmdi import relative_to_absolute
    from motius.motion.representation.normalization import affine_normalize
    absolute = relative_to_absolute(torch.as_tensor(rel[None], dtype=torch.float32, device=pipe.device))
    obs_x0 = affine_normalize(absolute, pipe.bundle.mean_abs, pipe.bundle.std_abs).permute(0, 2, 1).unsqueeze(2)
    m = torch.as_tensor(mask_abs[None, :, None, :], dtype=torch.bool, device=pipe.device)
    eff = (obs_x0 * m).detach().cpu().numpy()[0, :, 0, :]
    out = pipe._sample([text], [length], motions=[rel], observation_mask=m.cpu(), guidance_param=guidance, impute=bool(impute),
                       seed=seed, return_absolute=True)[0]
    return np.asarray(out, np.float32), eff, absolute.detach().cpu().numpy()[0]


def run(req, out, rel, mask, extra):
    w131.deterministic()
    rep = {"mode": req["mode"], "env": w131.env_report(), "request": {k: v for k, v in req.items() if k != "lfs_sha256"}}
    import subprocess
    rep["motius_head_sha"] = subprocess.run(["git", "-C", str(w131.MOTIUS), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    rep.update(extra)
    pipe, tl = w131.load_pipe(req)
    rep["load_s"] = tl
    rep["network_dtype"] = str(next(pipe.bundle.net.parameters()).dtype)
    rep["respacing"] = pipe.bundle.respacing
    rep["n_sampling_steps"] = int(pipe.diffusion.num_timesteps)
    rep["sampler_fn"] = "p_sample_loop -> p_sample (no cond_fn, no 'gmd' key) -> p_mean_variance"
    inst = Instrument(pipe.bundle.net)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    rss0 = w131.mem()
    t = time.time()
    try:
        out_abs, eff, abs_in = sample(pipe, req["text"], rel, mask, int(req["seed"]), int(req["length"]), float(req["guidance"]), req["impute"])
        torch.cuda.synchronize()
    finally:
        rep["instrumentation"] = inst.close()
    rep["sample_s"] = time.time() - t
    rep["peak_vram_allocated_bytes"] = torch.cuda.max_memory_allocated()
    rep["peak_vram_reserved_bytes"] = torch.cuda.max_memory_reserved()
    rep["rss_bytes"] = [rss0, w131.mem()]
    from motius.motion.representation.condmdi import absolute_to_relative, recover_from_absolute
    rel_out = absolute_to_relative(torch.as_tensor(out_abs)).numpy()
    joints = recover_from_absolute(torch.as_tensor(out_abs)).numpy()
    np.savez(out / "model_io.npz", input_relative=rel, input_absolute=abs_in, observation_mask_abs=mask,
             effective_conditioning_normalized=eff, output_absolute=out_abs, output_relative=rel_out, output_joints=joints)
    rep["observed_max_abs_err"] = float(np.abs(out_abs - abs_in)[mask.T].max())
    rep["output_finite"] = bool(np.isfinite(out_abs).all())
    rep["model_io_sha256"] = w131.sha256_file(out / "model_io.npz")
    w131.dump(out / "result.json", rep)
    ins = rep["instrumentation"]
    print(json.dumps({"impute": req["impute"], "sample_s": rep["sample_s"], "net_calls": ins["n_net_calls"],
                      "imputation_true_steps": sum(ins["requires_imputation"]), "finite": rep["output_finite"]}, indent=1))
    return 0 if rep["output_finite"] else 4


def mode_causal(req, out):
    bad = w131.check_artifact(req)
    if bad:
        w131.dump(out / "stop.json", {"stop": "BLOCKED_MODEL_IDENTITY", "bad": bad})
        return 3
    if w131.sha256_file(req["input_npz"]) != req["input_sha256"]:
        w131.dump(out / "stop.json", {"stop": "BLOCKED_INPUT_IDENTITY"})
        return 3
    z = np.load(req["input_npz"])
    rel, mask = z["relative_hml263"].astype(np.float32), z["observation_mask_abs"].astype(bool)
    if rel.shape != (req["length"], 263) or mask.shape != (263, req["length"]):
        w131.dump(out / "stop.json", {"stop": "BAD_INPUT_SHAPE"})
        return 4
    if req.get("perturb_masked_seed") is not None:
        rel = w131.perturb_masked(rel, mask, int(req["perturb_masked_seed"]))
    return run(req, out, rel, mask, {})


def mode_refctl(req, out):
    bad = w131.check_artifact(req)
    if bad:
        w131.dump(out / "stop.json", {"stop": "BLOCKED_MODEL_IDENTITY", "bad": bad})
        return 3
    if w131.sha256_file(req["reference_npz"]) != req["reference_sha256"]:
        w131.dump(out / "stop.json", {"stop": "BLOCKED_REFERENCE_IDENTITY"})
        return 3
    from motius.motion.conditioning.condmdi_masks import build_observation_mask
    z = np.load(req["reference_npz"])
    rel = z["relative"].astype(np.float32)
    L = int(req["length"])
    if rel.shape != (L, 263):
        w131.dump(out / "stop.json", {"stop": "BAD_REFERENCE_SHAPE", "shape": list(rel.shape)})
        return 4
    m = build_observation_mask([L], L, mode="middle", transition_length=int(req["hidden_width"]), feature_mode=req["feature_mode"],
                               joint_indices=None)
    mask = m.numpy()[0, :, 0, :].astype(bool)
    return run(req, out, rel, mask, {"official_mask_sha256": hashlib.sha256(mask.tobytes()).hexdigest()})


def main():
    req = json.loads(Path(sys.argv[1]).read_text())
    out = Path(req["out_dir"])
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}")
    out.mkdir(parents=True)
    return {"causal": mode_causal, "refctl": mode_refctl}[req["mode"]](req, out)


if __name__ == "__main__":
    sys.exit(main())
