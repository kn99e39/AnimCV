#!/usr/bin/env python3
"""W137 (EXPERIMENTAL, opt-in) UniMate-side runner for the OFFICIAL ``--inbetween`` temporal mode.  Runs ONLY in the
scratch UniMate environment (Python 3.10, ``C:/Users/dna10/animcv-w136/env/um310``) from the pinned official checkout's
root (cwd), never in the AnimCV venv.

Modes (each refuses an existing output directory):
  contract  NO MODEL.  Calls the official ``parse_keep_frames`` + ``build_keep_mask`` with the declared literal
            ``--keep_frames`` string and records the resolved mask; calls the official ``_validate_cfg_inputs`` with
            ``--inbetween`` + ``--motion_edit`` (and + ``--motion_expand``) and records that they are refused.
  recover   NO MODEL.  Official recoveries of one denormalised feature file (T, J, 12): RIC positions
            (``recover_unimate_joint_pos_from_ric``), FK positions (``recover_unimate_joint_pos_from_rot`` with the
            asset's cond offsets), the root trajectory, plus the same for a frozen-parent hybrid (the file's PROXY_B
            partner with only the upperarm_l/lowerarm_l rotation slots taken from the file).
  infer     ONE real inference through the UNMODIFIED official ``unimate.inference.sample.main`` with
            ``inbetween=True, keep_frames=<declared 32 indices>, gt_start_frame=0``.  No official function is replaced
            in behaviour: ``generate_samples`` is wrapped by a READ-ONLY recorder (it forwards the identical keyword
            arguments and returns the identical tensor object; it copies inputs/outputs to CPU and reads timing/VRAM).
            ``build_keep_mask`` / ``build_joint_keep_mask`` / the sampler are not touched.
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
import app.experimental_unimate_inbetween as T                        # noqa: E402


def _new_dir(p):
    p = Path(p)
    if p.exists():
        raise SystemExit(f"refusing to overwrite {p}")
    p.mkdir(parents=True)
    return p


def _cond_entry(asset_dir, name):
    return np.load(Path(asset_dir) / "cond.npy", allow_pickle=True).item()[name]


def _recover(feats, parents, offsets):
    from unimate.utils.motion_utils import (hml_rotations_to_bvh_quaternions, recover_unimate_joint_pos_from_ric,
                                            recover_unimate_joint_pos_from_rot, recover_unimate_root_quat_and_pos)
    f = np.asarray(feats, np.float64)
    ric = recover_unimate_joint_pos_from_ric(f)
    fk = recover_unimate_joint_pos_from_rot(f, np.asarray(parents), np.asarray(offsets, np.float64))
    _, rpos = recover_unimate_root_quat_and_pos(f[:, 0])
    bvh = np.asarray(hml_rotations_to_bvh_quaternions(f[:, :, 3:9], np.asarray(parents)).qs)
    return {"ric": np.asarray(ric), "fk": np.asarray(fk), "root_pos": np.asarray(rpos), "bvh_wxyz": bvh}


def mode_contract(a):
    out = _new_dir(a.out_dir)
    import torch
    from unimate.inference.motion_inbetweening import build_keep_mask, parse_keep_frames
    import unimate.inference.sample as S
    rep = {"keep_frames_arg": T.KEEP_FRAMES_ARG}
    idx = parse_keep_frames(T.KEEP_FRAMES_ARG)
    rep["parsed"] = idx
    mask, resolved = build_keep_mask(torch.tensor([U.T_MODEL]), idx, U.T_MODEL, torch.device("cpu"), labels=["contract"])
    m = mask.cpu().numpy()
    rep["mask_shape"], rep["resolved"] = list(m.shape), [int(i) for i in resolved]
    rep["validate"] = T.validate_temporal_mask(m)
    rep["equals_module_mask"] = bool(np.array_equal(m, T.temporal_mask()))
    # the official negative-index semantics, for the record (the declared list uses only non-negative indices)
    m2, r2 = build_keep_mask(torch.tensor([U.T_MODEL]), parse_keep_frames("0,-1"), U.T_MODEL, torch.device("cpu"))
    rep["default_0_m1_resolved"] = [int(i) for i in r2]
    # a 1-frame shift of the declared list must be detected by the module validator
    sh = [i + 1 for i in idx if i + 1 < U.T_MODEL]
    m3, _ = build_keep_mask(torch.tensor([U.T_MODEL]), sh, U.T_MODEL, torch.device("cpu"))
    try:
        T.validate_temporal_mask(m3.cpu().numpy())
        rep["shifted_list_refused"] = False
    except U.UniStop as e:
        rep["shifted_list_refused"] = e.code
    excl = {}
    for nm, kw in (("inbetween+motion_edit", dict(inbetween=True, motion_edit=True, keep_joints="pelvis")),
                   ("inbetween+motion_expand", dict(inbetween=True, motion_expand=True)),
                   ("inbetween_cfg_1", dict(inbetween=True))):
        args = S.InferenceArgs(exp_dir=".", test_cases_json="cases.json", **kw)
        try:
            S._validate_cfg_inputs(1.0 if nm == "inbetween_cfg_1" else 3.0, args)
            excl[nm] = {"refused": False}
        except ValueError as e:
            excl[nm] = {"refused": True, "message": str(e)}
    ok_args = S.InferenceArgs(exp_dir=".", test_cases_json="cases.json", inbetween=True, keep_frames=T.KEEP_FRAMES_ARG, gt_start_frame=0)
    try:
        S._validate_cfg_inputs(3.0, ok_args)
        excl["declared_inbetween_args_accepted"] = True
    except ValueError as e:
        excl["declared_inbetween_args_accepted"] = f"REFUSED: {e}"
    rep["mode_exclusivity"] = excl
    rep["ok"] = bool(rep["mask_shape"] == [1, 1, 1, U.T_MODEL] and rep["resolved"] == list(T.KEEP_FRAMES) and rep["equals_module_mask"]
                     and rep["shifted_list_refused"] and all(v["refused"] for k, v in excl.items() if isinstance(v, dict))
                     and excl["declared_inbetween_args_accepted"] is True)
    (out / "contract.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))
    return 0 if rep["ok"] else 3


def mode_recover(a):
    out = _new_dir(a.out_dir)
    cond = _cond_entry(a.asset_dir, a.name)
    names = [str(n) for n in cond["joint_names"]]
    par = np.asarray(cond["parents"])
    off = np.asarray(cond["tpos_offsets"], np.float64)            # the official retarget FK input (animate_motion.py)
    files = {"sample": np.load(a.sample)}
    if a.proxy:
        files["proxyB"] = np.load(a.proxy)
        files["hybrid"] = T.hybrid_features(files["proxyB"], files["sample"], names, par)
    res = {}
    for k, f in files.items():
        r = _recover(f, par, off)
        res.update({f"{k}_{q}": v for q, v in r.items()})
        res[f"{k}_feats"] = np.asarray(f)
    np.savez(out / "recovered.npz", names=np.array(names), parents=par, offsets=off, **res)
    rep = {k: list(np.shape(v)) for k, v in res.items()}
    for k in files:
        rep[f"{k}_ric_vs_fk_max"] = float(np.abs(res[f"{k}_ric"] - res[f"{k}_fk"]).max())
    (out / "recover.json").write_text(json.dumps(rep, indent=1))
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
    T.check_keep_frames(T.parse_keep_arg(a.keep_frames))
    rec = {"calls": 0}
    orig_gen = S.generate_samples

    def recorder(**kw):                                               # READ-ONLY: identical kwargs in, identical object out
        rec["calls"] += 1
        if rec["calls"] != 1:
            raise U.UniStop("INFERENCE_COUNT", "more than one generate_samples call")
        km = kw["keep_mask"].detach().cpu().numpy().copy()
        x1 = kw["x1_known"].detach().cpu().numpy().copy()
        rec["mask_validate"] = T.validate_temporal_mask(km)          # runs on a CPU copy; a refusal aborts BEFORE sampling
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        y = orig_gen(**kw)
        torch.cuda.synchronize()
        rec["sampler_seconds"] = time.perf_counter() - t0
        rec["peak_vram_bytes_sampler"] = int(torch.cuda.max_memory_allocated())
        c = kw["cond"]
        np.savez(out / "raw_model_io.npz", x1_known_proxyB_normalised=x1, keep_mask=km, raw_output_normalised=y.detach().cpu().numpy(),
                 mean=c["mean"].detach().cpu().numpy(), std=c["std"].detach().cpu().numpy(), cfg_scale=kw["cfg_scale"],
                 motion_shape=np.array(kw["motion_shape"]))
        rec["generate_samples_kwargs"] = sorted(kw)
        rec["cfg_scale"] = float(kw["cfg_scale"])
        rec["motion_shape"] = list(kw["motion_shape"])
        return y

    S.generate_samples = recorder
    cases = out / "cases.json"
    cases.write_text(json.dumps({f"{a.name}-{a.clip[len(a.name) + 1:]}": U.PROMPT}))
    args = S.InferenceArgs(exp_dir=a.exp_dir, model_path=a.model_path, seed=int(a.seed), output_dir=str(out / "official"),
                           num_repetitions=1, cfg_scale=None, test_cases_json=str(cases), asset=(a.asset_dir,),
                           batch_size=1, only_save_motion=True, inbetween=True, keep_frames=a.keep_frames, gt_start_frame=0)
    T.check_mode(args.inbetween, args.motion_edit, args.motion_expand)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    S.main(args)
    rec["total_seconds"] = time.perf_counter() - t0
    rec["peak_vram_bytes_total"] = int(torch.cuda.max_memory_allocated())
    rec["device"] = torch.cuda.get_device_name(0)
    rec["torch"] = torch.__version__
    rec["args"] = {k: (list(v) if isinstance(v, tuple) else v) for k, v in vars(args).items()}
    rec["wrapped_function"] = f"{orig_gen.__module__}.{orig_gen.__name__}"
    mot = sorted((out / "official").rglob("*.npy"))
    rec["saved_npy"] = [str(p) for p in mot]
    samp = [p for p in mot if "-rep_0-" in p.name and "gt_rep" not in p.name]
    gt = [p for p in mot if "-gt_rep_0-" in p.name]
    if len(samp) != 1 or len(gt) != 1:
        raise U.UniStop("OUTPUT_FILES", f"{[p.name for p in mot]}")
    s_feats, g_feats = np.load(samp[0]), np.load(gt[0])
    rec["sample_npy"], rec["proxyB_npy"] = str(samp[0]), str(gt[0])
    rec["sample_shape"], rec["proxyB_shape"] = list(s_feats.shape), list(g_feats.shape)
    rec["conditioning"] = T.conditioning_report(s_feats, g_feats, names)
    dec = {"T": U.decode_basis(s_feats, names, par, cond["tpos_local_rotations"]),
           "Tc": U.decode_basis(g_feats, names, par, cond["tpos_local_rotations"])}
    sys.path.insert(0, str(REPO / "scripts"))
    from w136_unimate_infer import _official_decode                  # unchanged W136 cross-check decoder
    off = {"T": _official_decode(s_feats, par, cond, names), "Tc": _official_decode(g_feats, par, cond, names)}
    np.savez(out / "decoded.npz", **{f"{k}_{b}": dec[k][b] for k in dec for b in dec[k]}, sample_feats=s_feats, proxyB_feats=g_feats)
    rec["own_vs_official_decode_max"] = {k: {b: float(U.qdist(dec[k][b], off[k][b]).max()) for b in dec[k]} for k in dec}
    (out / "infer_record.json").write_text(json.dumps(rec, indent=1, default=str))
    print(json.dumps({k: v for k, v in rec.items() if k not in ("args", "saved_npy")}, indent=1, default=str))
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("mode", choices=("contract", "recover", "infer"))
    for k in ("out-dir", "asset-dir", "name", "clip", "exp-dir", "model-path", "seed", "keep-frames", "sample", "proxy"):
        ap.add_argument("--" + k)
    a = ap.parse_args()
    return {"contract": mode_contract, "recover": mode_recover, "infer": mode_infer}[a.mode](a)


if __name__ == "__main__":
    sys.exit(main())
