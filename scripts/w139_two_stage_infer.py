#!/usr/bin/env python3
"""W139 (EXPERIMENTAL, opt-in) UniMate-side runner for the Two-Stage feasibility batch.  Runs ONLY in the scratch
UniMate environment (Python 3.10, ``C:/Users/dna10/animcv-w136/env/um310``) from the pinned official checkout's root
(cwd), never in the AnimCV venv.  Every mode refuses an existing output directory.

Modes:
  infer        ONE real inference through the official ``unimate.inference.sample.main`` (official asset resolution, clip
               pinning at ``--gt_start_frame 0``, CFG wrapper, saving) with these DISCLOSED scratch hooks:
                 (1) ``--mask w136``: ``build_joint_keep_mask`` -> W136's validated joint x time keep mask (as W136);
                     ``--mask empty``: every valid entry free (unconstrained generation; native reference only);
                     ``--official-mode inbetween``: NO mask hook -- the official ``--inbetween`` frame mask.
                 (2) ``unimate.inference.motion_inbetweening.inbetween_sample_ode`` (the module attribute that
                     ``generate_samples`` imports lazily) -> ``app.experimental_two_stage_sampler.
                     scheduled_inbetween_sample_ode`` with ``k_s`` and the scheduled set G = (~keep)[valid joints]
                     (k_s = 0 is behaviourally identical to the official function).
                 (3) optional ``--x1-npy``: the official ``x1_known`` is replaced by the given denormalised (60,50,12)
                     features normalised with the official cond mean/std (native control only).  The normalisation is
                     verified on the official tensor before substitution.
                 (4) a read-only recorder: eps SHA-256, every produced state, every CFG-combined velocity, the input
                     token embeddings of the network at the first non-discarded evaluation (forward hook on
                     ``input_layer``), call counts, timing, VRAM.
  native_prep  NO MODEL.  Model-native reference re-encode (official FK with tpos_offsets + official
               ``compute_unimate_motion_feats``) and the B-analogue init (slerp of the hidden local rotations between the
               kept window frames 15 and 44 -> FK -> official features), built twice: from the reference and from a
               copy whose hidden entries are NaN (must be identical).
  recover      NO MODEL.  Official RIC / FK(tpos_offsets) / root recovery of each given feature file and of the
               frozen-parent hybrid (the given proxy with only the upperarm_l/lowerarm_l rotation slots taken from it).
"""
from __future__ import annotations

import argparse
import functools
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
if not (Path.cwd() / "unimate" / "inference" / "sample.py").is_file():
    raise SystemExit("run from the root of the pinned official UniMate checkout")
sys.path.insert(0, str(Path.cwd()))
import numpy as np                                                    # noqa: E402

import app.experimental_unimate as U                                  # noqa: E402
import app.experimental_unimate_inbetween as T                        # noqa: E402
import app.experimental_two_stage as TS                               # noqa: E402


def _new_dir(p):
    p = Path(p)
    if p.exists():
        raise SystemExit(f"refusing to overwrite {p}")
    p.mkdir(parents=True)
    return p


def _cond_entry(asset_dir, name):
    return np.load(Path(asset_dir) / "cond.npy", allow_pickle=True).item()[name]


def _sha(a):
    import hashlib
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def _StubNet(real):
    """DEV DRY RUN ONLY (``--stub-model``): a deterministic, weight-free velocity v = 0.5 * (tanh(x) - x) + 0.05 with the
    attributes the official CFG wrapper and the W139 token hook need.  Used to exercise the harness end to end before
    the declaration without evaluating the pretrained network."""
    import torch

    class _In(torch.nn.Module):
        def forward(self, x, cond):
            t = x.permute(0, 3, 1, 2)
            return torch.cat([torch.zeros_like(t[:, :1]), t], 1), None, None

    class _Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.input_layer = _In()
            self.cond_mask_prob = 0.1

        def forward(self, x, timesteps, cond=None, force_mask=False):
            self.input_layer(x, cond)
            return 0.5 * (torch.tanh(x) - x) + (0.05 if not force_mask else 0.0)
    return _Net()


def _stats64(exp_dir, n):
    s = np.load(Path(exp_dir) / "dataset_stats.npy", allow_pickle=True).item()
    st = s["objaverse"]                                       # stats_dataset 'general' -> objaverse (all three keys identical)
    mean = np.concatenate([np.asarray(st["mean_root"], np.float64)[None], np.repeat(np.asarray(st["mean_local"], np.float64)[None], n - 1, 0)])
    std = np.concatenate([np.asarray(st["std_root"], np.float64)[None], np.repeat(np.asarray(st["std_local"], np.float64)[None], n - 1, 0)])
    return mean, std


def _normalise64(feats, exp_dir, n):
    """(T,J,12) float64 -> (J,12,T) float32, (x - mean) / std in float64 as the official conditioning (conditioning.py)."""
    mean, std = _stats64(exp_dir, n)
    return ((np.transpose(np.asarray(feats, np.float64), (1, 2, 0)) - mean[:, :, None]) / std[:, :, None]).astype(np.float32)


def _asset_feats(asset_dir, name, clip):
    from Quaternions import Quaternions
    from unimate.utils.motion_utils import compute_rots_from_tpos, compute_unimate_motion_feats
    cond = _cond_entry(asset_dir, name)
    z = np.load(Path(asset_dir) / "motions" / f"{clip}.npz")
    par = np.asarray(cond["parents"])
    lr = z["local_rotations"]
    tq = np.broadcast_to(np.asarray(cond["tpos_local_rotations"])[None], lr.shape).copy()
    reb = compute_rots_from_tpos(Quaternions(tq), Quaternions(lr), par)
    return compute_unimate_motion_feats(z["global_positions"], reb, par, Quaternions(z["root_facing_quat"]))


def mode_infer(a):
    out = _new_dir(a.out_dir)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    for k in ("DEEPSEEK_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY"):
        os.environ.pop(k, None)
    import torch
    import unimate.inference.sample as S
    import unimate.inference.motion_inbetweening as MI
    import app.experimental_two_stage_sampler as SS
    k_s = TS.validate_k_s(int(a.k_s))
    if a.mask not in ("w136", "empty", "official") or a.official_mode not in ("motion_edit", "inbetween"):
        raise TS.TwoStageStop("ARGS", f"mask={a.mask} official_mode={a.official_mode}")
    if (a.official_mode == "inbetween") != (a.mask == "official"):
        raise TS.TwoStageStop("ARGS", "the official frame mask is used iff --official-mode inbetween")
    cond = _cond_entry(a.asset_dir, a.name)
    names = [str(n) for n in cond["joint_names"]]
    par = np.asarray(cond["parents"])
    gen = U.generated_joints(names, par)
    kept_names = [n for j, n in enumerate(names) if j not in gen]
    rec = {"calls_generate_samples": 0, "k_s": k_s, "mask": a.mask, "official_mode": a.official_mode, "x1_injected": bool(a.x1_npy)}
    recorder = SS.StateRecorder(len(names), k_s)
    orig_mask = S.build_joint_keep_mask

    def keep_mask_hook(joint_names_per_sample, keep_names, max_joints, device):
        official = orig_mask(joint_names_per_sample, keep_names, max_joints, device)       # (B, J, 1, 1)
        if len(joint_names_per_sample) != 1:
            raise U.UniStop("BATCH", "exactly one case expected")
        aliases = [str(x).split("|")[0] for x in joint_names_per_sample[0]]
        if aliases != names:
            raise U.UniStop("JOINT_ORDER", "sampler joint order differs from the asset cond")
        off = official[0, :, 0, 0].cpu().numpy()
        if not np.array_equal(off[:len(names)], np.array([j not in gen for j in range(len(names))])):
            raise U.UniStop("MASK_DOMAIN", "official joint mask != complement of the generated joints")
        km = U.build_keep_mask(names, par, max_joints)
        rec["mask_info"] = TS.check_w136_free_set(km, names, par)
        if not np.array_equal(off[len(names):], km[len(names):, 0, 0]):
            raise U.UniStop("MASK_DOMAIN", "padding rows differ from the official mask")
        if a.mask == "empty":
            km = np.zeros_like(km)
        np.save(out / "keep_mask.npy", km)
        return torch.from_numpy(km[None]).to(device)                                      # (1, maxJ, 12, T)

    orig_gen = S.generate_samples
    emb = {"calls": 0, "first_free": None}

    def gen_hook(**kw):
        rec["calls_generate_samples"] += 1
        if rec["calls_generate_samples"] != 1:
            raise U.UniStop("INFERENCE_COUNT", "more than one generate_samples call")
        x1, km = kw["x1_known"], kw["keep_mask"]
        c_ = kw["cond"]
        mean, std = c_["mean"].detach().cpu().numpy(), c_["std"].detach().cpu().numpy()        # (1, 71, 12)
        x1_off = x1.detach().cpu().numpy().copy()
        np.save(out / "x1_known_official.npy", x1_off)
        # normalisation identity (gated): the W139 injector applied to the asset's own official float64 features must
        # reproduce the official x1_known bit-exactly on valid joints
        own = _asset_feats(a.asset_dir, a.name, a.clip)
        rec["normalisation_identity_max_abs"] = float(np.abs(_normalise64(own, a.exp_dir, len(names)) - x1_off[0, :len(names)]).max())
        if rec["normalisation_identity_max_abs"] != 0.0:
            raise U.UniStop("NORMALISATION", f"injector != official x1 ({rec['normalisation_identity_max_abs']})")
        if a.x1_npy:
            f = np.load(a.x1_npy).astype(np.float64)                                     # (T, J, 12) denormalised
            if f.shape != (U.T_MODEL, len(names), U.FEAT) or not np.isfinite(f).all():
                raise U.UniStop("FEATURE_SHAPE", f"injected {f.shape}")
            inj = x1_off.copy()
            inj[0, :len(names)] = _normalise64(f, a.exp_dir, len(names))
            kw["x1_known"] = torch.from_numpy(inj).to(x1.device, x1.dtype)
            rec["x1_injected_sha256"] = _sha(inj)
        x1_used = kw["x1_known"].detach().cpu().numpy().copy()
        np.save(out / "x1_known_used.npy", x1_used)
        kmn = torch.broadcast_to(km, tuple(kw["motion_shape"])).detach().cpu().numpy()[0]
        g = TS.sched_set(kmn, len(names))
        np.save(out / "sched_mask.npy", g)
        rec["keep_valid_true"] = int(kmn[:len(names)].sum())
        rec["sched_valid_true"] = int(g.sum())
        sched_t = torch.from_numpy(g[None]).to(km.device)
        if a.stub_model:                                                                  # DEV DRY RUN ONLY: no pretrained network is evaluated
            kw["model"] = _StubNet(kw["model"]).to(x1.device)
            rec["STUB_MODEL"] = "dev dry run: deterministic stub velocity, NOT the pretrained network"
        model = kw["model"]

        def emb_hook(_m, _inp, output):
            i = emb["calls"]
            emb["calls"] += 1
            if i in (2 * k_s, 2 * k_s + 1):                                              # CFG: cond then uncond call of step k_s
                tok = output[0] if isinstance(output, (tuple, list)) else output
                emb.setdefault("tokens", {})["cond" if i == 2 * k_s else "uncond"] = tok.detach().float().cpu().numpy()
        hd = model.input_layer.register_forward_hook(emb_hook)
        patched = functools.partial(SS.scheduled_inbetween_sample_ode, sched_mask=sched_t, k_s=k_s, recorder=recorder)
        original_attr = MI.inbetween_sample_ode
        MI.inbetween_sample_ode = patched
        try:
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            y = orig_gen(**kw)
            torch.cuda.synchronize()
            rec["sampler_seconds"] = time.perf_counter() - t0
        finally:
            MI.inbetween_sample_ode = original_attr
            hd.remove()
        rec["peak_vram_bytes_sampler"] = int(torch.cuda.max_memory_allocated())
        rec["scheduled_sampler_calls"] = recorder.calls
        rec["network_forward_calls"] = emb["calls"]
        rec["eps_sha256"] = recorder.eps_sha256
        rec["state_mask_sha256"] = recorder.mask_sha
        ar = recorder.arrays()
        ff = ar.pop("first_free_step_input")
        np.savez(out / "trajectory.npz", **ar, first_free_step_input=ff if ff is not None else np.zeros(0))
        tok = emb.get("tokens", {})
        if tok:
            np.savez(out / "first_free_tokens.npz", **{k: v for k, v in tok.items()})
        np.savez(out / "raw_model_io.npz", x1_known_used_normalised=x1_used, keep_mask=km.detach().cpu().numpy(),
                 sched_mask=g, raw_output_normalised=y.detach().cpu().numpy(), mean=mean, std=std,
                 cfg_scale=kw["cfg_scale"], motion_shape=np.array(kw["motion_shape"]))
        rec["cfg_scale"] = float(kw["cfg_scale"])
        rec["motion_shape"] = list(kw["motion_shape"])
        return y

    if a.mask != "official":
        S.build_joint_keep_mask = keep_mask_hook
    S.generate_samples = gen_hook
    cases = out / "cases.json"
    cases.write_text(json.dumps({f"{a.name}-{a.clip[len(a.name) + 1:]}": U.PROMPT}))
    common = dict(exp_dir=a.exp_dir, model_path=a.model_path, seed=int(a.seed), output_dir=str(out / "official"), num_repetitions=1,
                  cfg_scale=None, test_cases_json=str(cases), asset=(a.asset_dir,), batch_size=1, only_save_motion=True, gt_start_frame=0)
    if a.official_mode == "inbetween":
        T.check_keep_frames(T.parse_keep_arg(T.KEEP_FRAMES_ARG))
        args = S.InferenceArgs(**common, inbetween=True, keep_frames=T.KEEP_FRAMES_ARG)
    else:
        args = S.InferenceArgs(**common, motion_edit=True, keep_joints=",".join(kept_names))
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    S.main(args)
    rec["total_seconds"] = time.perf_counter() - t0
    rec["peak_vram_bytes_total"] = int(torch.cuda.max_memory_allocated())
    rec["device"] = torch.cuda.get_device_name(0)
    rec["torch"] = torch.__version__
    rec["cuda"] = torch.version.cuda
    rec["cudnn"] = torch.backends.cudnn.version()
    rec["args"] = {k: (list(v) if isinstance(v, tuple) else v) for k, v in vars(args).items()}
    rec["sampler_function"] = f"{SS.scheduled_inbetween_sample_ode.__module__}.{SS.scheduled_inbetween_sample_ode.__name__}"
    mot = sorted((out / "official").rglob("*.npy"))
    samp = [p for p in mot if "-rep_0-" in p.name and "gt_rep" not in p.name]
    if len(samp) != 1:
        raise U.UniStop("OUTPUT_FILES", f"{[p.name for p in mot]}")
    s_feats = np.load(samp[0])
    rec["sample_npy"] = str(samp[0])
    rec["sample_shape"] = list(s_feats.shape)
    # the denormalised sample must equal the recorded raw output (official saver = denormalise)
    raw = np.load(out / "raw_model_io.npz")
    den = np.transpose(raw["raw_output_normalised"][0, :len(names)] * raw["std"][0, :len(names), :, None]
                       + raw["mean"][0, :len(names), :, None], (2, 0, 1))
    rec["saved_vs_recorded_output_max_abs"] = float(np.abs(den - s_feats).max())
    try:
        dec = U.decode_basis(s_feats, names, par, cond["tpos_local_rotations"])
        np.savez(out / "decoded.npz", **{f"S_{b}": dec[b] for b in dec}, sample_feats=s_feats)
        rec["decode"] = "OK"
    except U.UniStop as e:
        rec["decode"] = {"STOP": e.code, "detail": e.detail}
        np.savez(out / "decoded.npz", sample_feats=s_feats)
    (out / "infer_record.json").write_text(json.dumps(rec, indent=1, default=str))
    print(json.dumps({k: v for k, v in rec.items() if k not in ("args", "state_mask_sha256")}, indent=1, default=str))
    return 0


# ------------------------------------------------------------------------------------------------ native control ---
def _official_bits():
    from Quaternions import Quaternions
    from unimate.utils.motion_utils import (compute_unimate_motion_feats, fk_global_positions, hml_rotations_to_bvh_quaternions,
                                            recover_unimate_joint_pos_from_ric, recover_unimate_joint_pos_from_rot,
                                            recover_unimate_root_quat_and_pos)
    return SimpleBits(Quaternions, compute_unimate_motion_feats, fk_global_positions, hml_rotations_to_bvh_quaternions,
                      recover_unimate_joint_pos_from_ric, recover_unimate_joint_pos_from_rot, recover_unimate_root_quat_and_pos)


class SimpleBits:
    def __init__(self, Q, feats, fk, hml2bvh, ric, rot, root):
        self.Q, self.feats, self.fk, self.hml2bvh, self.ric, self.rot, self.root = Q, feats, fk, hml2bvh, ric, rot, root


def _encode_from_rotations(B, bvh_wxyz, root_feats, par, offsets):
    """FK of (T,J) bvh rotations with the official integrated root, then official 12-D features (61st frame = repeat)."""
    r_rot, r_pos = B.root(np.asarray(root_feats, np.float64))
    gp = B.fk(B.Q(np.asarray(bvh_wxyz, np.float64)), par, offsets, r_pos)
    gp61 = np.concatenate([gp, gp[-1:]], 0)
    q61 = np.concatenate([bvh_wxyz, bvh_wxyz[-1:]], 0)
    rr61 = np.concatenate([r_rot.qs, r_rot.qs[-1:]], 0)
    return B.feats(gp61, B.Q(q61), par, B.Q(rr61)), gp


def mode_native_prep(a):
    out = _new_dir(a.out_dir)
    B = _official_bits()
    cond = _cond_entry(a.asset_dir, a.name)
    names = [str(n) for n in cond["joint_names"]]
    par = np.asarray(cond["parents"])
    off = np.asarray(cond["tpos_offsets"], np.float64)
    raw = np.load(a.ref).astype(np.float64)                                              # (60, 50, 12) model sample
    rep = {"ref_shape": list(raw.shape)}
    # (1) model-native reference re-encode: official decode -> FK(tpos_offsets) -> official features
    bvh = np.asarray(B.hml2bvh(raw[:, :, 3:9], par).qs)
    nstar, gp_star = _encode_from_rotations(B, bvh, raw[:, 0], par, off)
    nstar[-1, :, 9:12] = raw[-1, :, 9:12]                    # frame 59 velocity: the model's own (no pose 60 exists)
    rep["nref_star_vs_raw_vel_max_abs_frames_0_58"] = float(np.abs(nstar[:-1, :, 9:12] - raw[:-1, :, 9:12]).max())
    rep["nref_star_vs_raw_rot6d_max_abs"] = float(np.abs(nstar[:, 1:, 3:9] - raw[:, 1:, 3:9]).max())
    rep["nref_star_vs_raw_root_max_abs"] = float(np.abs(nstar[:, 0] - raw[:, 0]).max())
    ric, fk = B.ric(nstar), B.rot(nstar, par, off)
    rep["nref_star_ric_vs_fk_max"] = float(np.abs(ric - fk).max())
    raw_ric, raw_fk = B.ric(raw), B.rot(raw, par, off)
    rep["raw_ref_ric_vs_fk_max"] = float(np.abs(raw_ric - raw_fk).max())
    # idempotence of the re-encode
    bvh2 = np.asarray(B.hml2bvh(nstar[:, :, 3:9], par).qs)
    n2, _ = _encode_from_rotations(B, bvh2, nstar[:, 0], par, off)
    rep["reencode_idempotence_max_abs_frames_0_58"] = float(np.abs(n2[:-1] - nstar[:-1]).max())
    # (2) B-analogue init: slerp hidden local rotations between kept window frames 15 and 44
    km = U.build_keep_mask(names, par, len(names))
    g = TS.sched_set(km, len(names))                                                     # (J,12,T)
    gT = np.transpose(g, (2, 0, 1))                                                      # (T,J,12)
    gen_joints = sorted({int(j) for j in np.nonzero(g.any(axis=(1, 2)))[0]})
    hidden_bvh = sorted({int(par[j]) for j in gen_joints})
    f0, f1 = U.action_to_index(U.CONTEXT_LAST), U.action_to_index(U.TARGET_FIRST)          # 15, 44

    def build(src):
        s = np.array(src, np.float64, copy=True)
        ident6 = np.array([1.0, 0, 0, 0, 1.0, 0])
        hid_rot = gT[:, :, 3:9].any(-1)                                                  # (T,J) hidden 6D slots
        s[:, :, 3:9][hid_rot] = ident6                                                   # hidden slots never read
        s[:, :, 0:3][gT[:, :, 0:3].any(-1)] = 0.0                                        # hidden positions never read
        s[:, :, 9:12][gT[:, :, 9:12].any(-1)] = 0.0                                      # hidden velocities never read
        q = np.asarray(B.hml2bvh(s[:, :, 3:9], par).qs)
        q = TS.slerp_hidden_rotations(q, hidden_bvh, f0, f1)
        f, _ = _encode_from_rotations(B, q, s[:, 0], par, off)
        comp = np.where(gT, f, np.asarray(src, np.float64))                              # non-G entries untouched
        return comp, f
    nB, nB_full = build(nstar)
    nB_p, _ = build(TS.poison_hidden(nstar, g))
    rep["poison_equal"] = bool(np.array_equal(nB[gT], nB_p[gT]))
    rep["poisoned_finite_on_G"] = bool(np.isfinite(nB_p[gT]).all())
    rep["nB_nonG_vs_recomputed_max_abs_frames_0_58"] = float(np.abs(nB_full[:-1][~gT[:-1]] - nstar[:-1][~gT[:-1]]).max())
    rep["nB_ric_vs_fk_max"] = float(np.abs(B.ric(nB) - B.rot(nB, par, off)).max())
    rep["hidden_bvh_joints"] = [names[j] for j in hidden_bvh]
    rep["generated_joints"] = [names[j] for j in gen_joints]
    rep["n_hidden_entries"] = int(g.sum())
    np.save(out / "nref_star.npy", nstar.astype(np.float32))
    np.save(out / "nB_init.npy", nB.astype(np.float32))
    np.savez(out / "native_prep.npz", nref_star=nstar, nB=nB, nB_poisoned=nB_p, ric_star=ric, fk_star=fk, raw_ric=raw_ric, raw_fk=raw_fk,
             fk_nB=B.rot(nB, par, off), gp_star=gp_star, names=np.array(names), parents=par)
    (out / "native_prep.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))
    return 0


def mode_recover(a):
    out = _new_dir(a.out_dir)
    B = _official_bits()
    cond = _cond_entry(a.asset_dir, a.name)
    names = [str(n) for n in cond["joint_names"]]
    par = np.asarray(cond["parents"])
    off = np.asarray(cond["tpos_offsets"], np.float64)
    proxy = np.load(a.proxy).astype(np.float64)
    res, rep = {}, {}
    for item in a.samples.split(","):
        key, path = item.split("=", 1)
        f = np.load(path).astype(np.float64)
        hyb = T.hybrid_features(proxy, f, names, par)
        for kk, ff in ((key, f), (key + "_hybrid", hyb)):
            res[f"{kk}_fk"] = np.asarray(B.rot(ff, par, off))
            res[f"{kk}_ric"] = np.asarray(B.ric(ff))
            res[f"{kk}_root_pos"] = np.asarray(B.root(ff[:, 0])[1])
            rep[f"{kk}_ric_vs_fk_max"] = float(np.abs(res[f"{kk}_ric"] - res[f"{kk}_fk"]).max())
    res["proxy_fk"] = np.asarray(B.rot(proxy, par, off))
    np.savez(out / "recovered.npz", names=np.array(names), parents=par, offsets=off, **res)
    (out / "recover.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("mode", choices=("infer", "native_prep", "recover"))
    for k in ("out-dir", "asset-dir", "name", "clip", "exp-dir", "model-path", "seed", "k-s", "x1-npy", "ref", "proxy", "samples"):
        ap.add_argument("--" + k)
    ap.add_argument("--mask", default="w136")
    ap.add_argument("--official-mode", default="motion_edit")
    ap.add_argument("--stub-model", action="store_true", help="DEV DRY RUN ONLY: replace the network by a weight-free stub")
    a = ap.parse_args()
    return {"infer": mode_infer, "native_prep": mode_native_prep, "recover": mode_recover}[a.mode](a)


if __name__ == "__main__":
    sys.exit(main())
