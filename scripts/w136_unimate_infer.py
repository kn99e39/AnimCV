#!/usr/bin/env python3
"""W136 (EXPERIMENTAL, opt-in) UniMate-side runner.  Runs ONLY in the scratch UniMate environment (Python 3.10,
``C:/Users/dna10/animcv-w136/env/um310``) from the pinned official checkout's root (cwd), never in the AnimCV venv.

Modes (each refuses an existing output directory):
  features  NO MODEL.  Official feature construction (``compute_rots_from_tpos`` + ``compute_unimate_motion_feats``) of
            one saved own-rig clip, the official HML decode (``hml_rotations_to_bvh_quaternions``) of the left-arm
            rotations, and this experiment's independent numpy decode (``app.experimental_unimate.decode_basis``).
  infer     ONE real inference: the official ``unimate.inference.sample.main`` motion-editing path (official asset
            resolution, clip pinning at ``--gt_start_frame 0``, CFG wrapper, replacement Euler sampler, saving), with
            exactly two disclosed scratch hooks:
              (1) ``build_joint_keep_mask`` -> the official joint mask (all non-generated joints by raw name) refined
                  in time: the left-arm descendants of upperarm_l are generated only at Action 172..199 (+ their
                  171->172 velocity channels); everything else is held at the PROXY_B clip.  The mask is validated
                  (``validate_mask``) and must equal the official joint mask on every non-generated joint.
              (2) ``generate_samples`` is wrapped (not changed) to record the model input tensor, keep mask, raw
                  normalised output, wall time and peak VRAM.
            Nothing else of the official code path is altered: checkpoint, config, stats, seed, cfg, sampler and steps
            are the published defaults; HF access is offline against the pinned local text-encoder snapshot.
"""
from __future__ import annotations

import argparse
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


def _new_dir(p):
    p = Path(p)
    if p.exists():
        raise SystemExit(f"refusing to overwrite {p}")
    p.mkdir(parents=True)
    return p


def _cond_entry(asset_dir, name):
    return np.load(Path(asset_dir) / "cond.npy", allow_pickle=True).item()[name]


def _official_decode(feats, parents, cond, names, joints=U.EDIT_CHANNELS):
    """Official HML decode (chordal-mean bvh rotations) + the same declared inverse rebase, for cross-checking."""
    from unimate.utils.motion_utils import hml_rotations_to_bvh_quaternions
    bvh = hml_rotations_to_bvh_quaternions(np.asarray(feats, float)[:, :, 3:9], np.asarray(parents)).qs
    tq = U.qnorm(np.asarray(cond["tpos_local_rotations"], float))
    cum = U.cumulative_tpose(tq, parents)
    out = {}
    for b in joints:
        j = names.index(b)
        p = int(parents[j])
        L = U.qmul(U.qmul(U.qmul(U.qinv(cum[p])[None], bvh[:, j]), cum[p][None]), tq[j][None])
        out[b] = U.qnorm(U.qmul(U.qinv(tq[j])[None], L))
    return out


def mode_features(a):
    from Quaternions import Quaternions
    from unimate.utils.motion_utils import compute_rots_from_tpos, compute_unimate_motion_feats
    out = _new_dir(a.out_dir)
    cond = _cond_entry(a.asset_dir, a.name)
    z = np.load(Path(a.asset_dir) / "motions" / f"{a.clip}.npz")
    U.check_fps(int(z["fps"]))
    names = [str(n) for n in cond["joint_names"]]
    par = np.asarray(cond["parents"])
    lr = z["local_rotations"]
    tq = np.broadcast_to(np.asarray(cond["tpos_local_rotations"])[None], lr.shape).copy()
    reb = compute_rots_from_tpos(Quaternions(tq), Quaternions(lr), par)
    feats = compute_unimate_motion_feats(z["global_positions"], reb, par, Quaternions(z["root_facing_quat"]))
    mine = U.decode_basis(feats, names, par, cond["tpos_local_rotations"])
    off = _official_decode(feats, par, cond, names)
    np.savez(out / "features.npz", feats=feats, names=np.array(names), parents=par,
             tpos_local_rotations=np.asarray(cond["tpos_local_rotations"]), tpos_first_frame=np.asarray(cond["tpos_first_frame"]),
             offsets=np.asarray(cond["offsets"]), clean_joint_names=np.array([str(n) for n in cond["clean_joint_names"]]),
             n_clip_frames=lr.shape[0], clip_fps=int(z["fps"]),
             **{f"decode_{b}": mine[b] for b in mine}, **{f"official_decode_{b}": off[b] for b in off})
    rep = {"clip": a.clip, "n_clip_frames": int(lr.shape[0]), "feature_shape": list(feats.shape), "clip_fps": int(z["fps"]),
           "own_vs_official_decode_max": {b: float(U.qdist(mine[b], off[b]).max()) for b in mine},
           "face_joint_idxs": {k: (int(v) if not isinstance(v, (bool, np.bool_)) else bool(v)) for k, v in dict(cond.get("face_joint_idxs", {})).items()}}
    (out / "features.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))
    return 0


def mode_infer(a):
    out = _new_dir(a.out_dir)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    for k in ("DEEPSEEK_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY"):
        os.environ.pop(k, None)
    import torch
    import unimate.inference.sample as S
    cond = _cond_entry(a.asset_dir, a.name)
    names = [str(n) for n in cond["joint_names"]]
    par = np.asarray(cond["parents"])
    gen = U.generated_joints(names, par)
    kept_names = [n for j, n in enumerate(names) if j not in gen]
    rec = {"calls": 0}
    orig_mask = S.build_joint_keep_mask

    def keep_mask_hook(joint_names_per_sample, keep_names, max_joints, device):
        official = orig_mask(joint_names_per_sample, keep_names, max_joints, device)       # (B, J, 1, 1)
        if len(joint_names_per_sample) != 1:
            raise U.UniStop("BATCH", "exactly one case expected")
        aliases = [str(x).split("|")[0] for x in joint_names_per_sample[0]]
        if aliases != names:
            raise U.UniStop("JOINT_ORDER", "sampler joint order differs from the asset cond")
        km = U.build_keep_mask(names, par, max_joints)
        info = U.validate_mask(km, names, par)
        off = official[0, :, 0, 0].cpu().numpy()
        if not np.array_equal(off[:len(names)], np.array([j not in gen for j in range(len(names))])):
            raise U.UniStop("MASK_DOMAIN", "official joint mask != complement of the generated joints")
        if not np.array_equal(off[len(names):], km[len(names):, 0, 0]):
            raise U.UniStop("MASK_DOMAIN", "padding rows differ from the official mask")
        rec["mask_info"] = info
        np.save(out / "keep_mask.npy", km)
        return torch.from_numpy(km[None]).to(device)                                      # (1, maxJ, 12, T)

    orig_gen = S.generate_samples

    def gen_hook(**kw):
        rec["calls"] += 1
        if rec["calls"] != 1:
            raise U.UniStop("INFERENCE_COUNT", "more than one generate_samples call")
        x1, km = kw["x1_known"], kw["keep_mask"]
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        y = orig_gen(**kw)
        torch.cuda.synchronize()
        rec["sampler_seconds"] = time.perf_counter() - t0
        rec["peak_vram_bytes_sampler"] = int(torch.cuda.max_memory_allocated())
        cond_ = kw["cond"]
        np.savez(out / "raw_model_io.npz", x1_known_proxyB_normalised=x1.detach().cpu().numpy(), keep_mask=km.detach().cpu().numpy(),
                 raw_output_normalised=y.detach().cpu().numpy(), mean=cond_["mean"].detach().cpu().numpy(),
                 std=cond_["std"].detach().cpu().numpy(), cfg_scale=kw["cfg_scale"], motion_shape=np.array(kw["motion_shape"]))
        return y

    S.build_joint_keep_mask = keep_mask_hook
    S.generate_samples = gen_hook
    cases = out / "cases.json"
    cases.write_text(json.dumps({f"{a.name}-{a.clip[len(a.name) + 1:]}": U.PROMPT}))
    args = S.InferenceArgs(exp_dir=a.exp_dir, model_path=a.model_path, seed=int(a.seed), output_dir=str(out / "official"),
                           num_repetitions=1, cfg_scale=None, test_cases_json=str(cases), asset=(a.asset_dir,),
                           batch_size=1, only_save_motion=True, motion_edit=True, keep_joints=",".join(kept_names),
                           gt_start_frame=0)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    S.main(args)
    rec["total_seconds"] = time.perf_counter() - t0
    rec["peak_vram_bytes_total"] = int(torch.cuda.max_memory_allocated())
    rec["device"] = torch.cuda.get_device_name(0)
    rec["torch"] = torch.__version__
    rec["args"] = {k: (list(v) if isinstance(v, tuple) else v) for k, v in vars(args).items()}
    rec["kept_joint_names"] = kept_names
    rec["generated_joint_names"] = [names[j] for j in gen]
    mot = sorted((out / "official").rglob("*.npy"))
    rec["saved_npy"] = [str(p) for p in mot]
    samp = [p for p in mot if "-rep_0-" in p.name and "gt_rep" not in p.name]
    gt = [p for p in mot if "-gt_rep_0-" in p.name]
    if len(samp) != 1 or len(gt) != 1:
        raise U.UniStop("OUTPUT_FILES", f"{[p.name for p in mot]}")
    s_feats, g_feats = np.load(samp[0]), np.load(gt[0])
    rec["sample_npy"], rec["proxyB_npy"] = str(samp[0]), str(gt[0])
    dec = {"U": U.decode_basis(s_feats, names, par, cond["tpos_local_rotations"]),
           "Bc": U.decode_basis(g_feats, names, par, cond["tpos_local_rotations"])}
    off = {"U": _official_decode(s_feats, par, cond, names), "Bc": _official_decode(g_feats, par, cond, names)}
    np.savez(out / "decoded.npz", **{f"{k}_{b}": dec[k][b] for k in dec for b in dec[k]},
             sample_feats=s_feats, proxyB_feats=g_feats)
    km = np.load(out / "keep_mask.npy")[:len(names)]                                    # (J, 12, T)
    kmT = np.transpose(km, (2, 0, 1))                                                   # (T, J, 12)
    rec["replacement_kept_max_abs_denorm"] = float(np.abs(s_feats - g_feats)[kmT].max())
    rec["generated_region_max_abs_denorm"] = float(np.abs(s_feats - g_feats)[~kmT].max())
    rec["own_vs_official_decode_max"] = {k: {b: float(U.qdist(dec[k][b], off[k][b]).max()) for b in dec[k]} for k in dec}
    (out / "infer_record.json").write_text(json.dumps(rec, indent=1, default=str))
    print(json.dumps({k: v for k, v in rec.items() if k not in ("args", "saved_npy")}, indent=1, default=str))
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("mode", choices=("features", "infer"))
    for k in ("out-dir", "asset-dir", "name", "clip", "exp-dir", "model-path", "seed"):
        ap.add_argument("--" + k)
    a = ap.parse_args()
    return {"features": mode_features, "infer": mode_infer}[a.mode](a)


if __name__ == "__main__":
    sys.exit(main())
