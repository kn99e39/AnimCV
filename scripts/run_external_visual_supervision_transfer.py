#!/usr/bin/env python3
"""DIAGNOSTIC (Worklog 73): E0_3DPW_ONLY vs E1_3DPW_PLUS_EXTERNAL for the fixed Worklog 69 head.

E0 is the Worklog 69 C1 contract exactly (S0 frame-uniform permutation over
3DPW TRAIN) and must reproduce C1 (training curve, selected epoch, held-out
predictions) before E1 is interpreted.  E1 uses the identical graph,
initialization (seed 1337), optimizer, schedule, batch and draw budget
(11,334 draws/epoch, 9,000 steps) with a fixed source-balanced sampler: each
epoch exactly half the draws are uniform over eligible 3DPW TRAIN rows and
half uniform over eligible external TRAIN rows (with replacement), shuffled.
Selection stays on 3DPW validation mean |f|.  External validation is a
secondary diagnostic only.  Training uses the Worklog 71 ``train`` unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from framepose.arm_depth_probe import (
    SEGMENT_NAMES, angle_degrees, elevation_degrees, forward_targets, observed_plane, reconstruct_direction,
    segment_vectors,
)
from framepose.bank import load_bank
from framepose.bedlam_supervision import ExternalBank
from framepose.continuous_vision_depth import candidate_visual, predict, sign_magnitude
from framepose.learned_vision_sensor import pair_geometry
from framepose.train import geometry_tensor
from framepose.visual_coreset import arm_features
from framepose.visual_supervision_sampling import EpochSampler, train_sequence_groups
from run_visual_supervision_diversity import CONFIG, train

STABLE = math.sin(math.radians(10))
CLEAN = ("courtyard_dancing_00", "courtyard_hug_00")
F_BINS = ((0.0, STABLE), (STABLE, 0.5), (0.5, 1.0001))
EVAL_MODELS = ("e1", "e0", "h0", "d1")
W68_FINGERPRINT = "a0d72ded575eaa4460dad11dd1e313bc69f0403b2c97c3c1ba234af086952904"
SAME_IDENTITY = ("crop_contract", "render_resolution", "overlay", "model_repository", "model_revision",
                 "model_fingerprint", "quantization", "processor", "feature_source", "merger_linear_fc2_class",
                 "token_grid", "token_order", "token_grid_dtype", "channels", "readout", "transformers", "torch")


class SourceBalancedSampler:
    """Exactly half of every epoch's draws from each source, uniform over its eligible rows."""

    name = "E1_SOURCE_BALANCED_50_50"

    def __init__(self, primary: np.ndarray, external: np.ndarray, draws_per_epoch: int, seed: int):
        import torch

        if draws_per_epoch % 2:
            raise ValueError("draws per epoch must be even for an exact 50/50 split")
        self.primary = np.asarray(primary, np.int64)
        self.external = np.asarray(external, np.int64)
        self.draws_per_epoch = draws_per_epoch
        self.generator = torch.Generator(device="cpu").manual_seed(seed)

    def epoch(self) -> np.ndarray:
        import torch

        half = self.draws_per_epoch // 2
        a = self.primary[torch.randint(len(self.primary), (half,), generator=self.generator).numpy()]
        b = self.external[torch.randint(len(self.external), (half,), generator=self.generator).numpy()]
        both = np.concatenate([a, b])
        return both[torch.randperm(len(both), generator=self.generator).numpy()]


def _stats(v):
    v = np.asarray([x for x in v if x is not None and np.isfinite(x)], float)
    return {"n": int(len(v)), "mean": float(v.mean()), "p05": float(np.percentile(v, 5)), "p50": float(np.percentile(v, 50)),
            "p95": float(np.percentile(v, 95))} if len(v) else {"n": 0}


def load_cache(cache_dir: Path, bank_digest: str, sample_ids: list[str], reference: dict | None):
    manifest = json.loads((cache_dir / "manifest.json").read_text())
    path = cache_dir / "readout.npz"
    ident = manifest["identity"]
    if (ident["bank_content_digest"] != bank_digest
            or ident["sample_ids_sha256"] != hashlib.sha256("\n".join(sample_ids).encode()).hexdigest()
            or ident["model_fingerprint"] != W68_FINGERPRINT
            or manifest["readout_npz_sha256"] != hashlib.sha256(path.read_bytes()).hexdigest()):
        raise SystemExit(f"Qwen cache identity mismatch: {cache_dir}")
    if reference is not None and any(ident[k] != reference[k] for k in SAME_IDENTITY):
        raise SystemExit(f"external cache does not share the Worklog 68 representation contract: {cache_dir}")
    return np.load(path)["readout"], ident


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("bank", "cache_dir", "external_train", "external_train_cache", "external_val", "external_val_cache",
                 "w69_dir", "w71_dir", "out_dir"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    import torch

    device = torch.device(args.device)
    bank = load_bank(args.bank)
    ext_tr, ext_va = ExternalBank.load(args.external_train), ExternalBank.load(args.external_val)
    r3, ident3 = load_cache(args.cache_dir, bank.content_digest(), [s.sample_id for s in bank.samples], None)
    rtr, _ = load_cache(args.external_train_cache, ext_tr.content_digest(), [s.sample_id for s in ext_tr.samples], ident3)
    rva, _ = load_cache(args.external_val_cache, ext_va.content_digest(), [s.sample_id for s in ext_va.samples], ident3)
    w69 = torch.load(args.w69_dir / "C1_QWEN_VISION.pt", map_location="cpu", weights_only=False)
    w71_report = json.loads((args.w71_dir / "report.json").read_text())
    if w69["cache_identity"]["digest"] != ident3["digest"]:
        raise SystemExit("Worklog 69 head identity mismatch")
    if {s.split for s in ext_tr.samples} != {"train"} or {s.split for s in ext_va.samples} != {"validation"}:
        raise SystemExit("external split labels wrong")
    overlap = {s.sequence_id for s in ext_tr.samples} & {s.sequence_id for s in ext_va.samples}
    if overlap or ({s.image_reference.relative_path for s in ext_tr.samples}
                   & {s.image_reference.relative_path for s in ext_va.samples}):
        raise SystemExit("external TRAIN/validation overlap")

    # --- one stacked array space: [3DPW | external TRAIN | external validation] ---
    n3, ntr = len(bank), len(ext_tr)
    visual = candidate_visual(np.concatenate([r3, rtr, rva]), "C1_QWEN_VISION")
    geometry = np.concatenate([pair_geometry(geometry_tensor(b)) for b in (bank, ext_tr, ext_va)])
    targets, masks = forward_targets(np.concatenate([b.arrays["target_3d"] for b in (bank, ext_tr, ext_va)]),
                                     np.concatenate([b.arrays["target_valid"] for b in (bank, ext_tr, ext_va)]))
    eligible = masks.any(axis=1)
    sequence_ids = [s.sequence_id for b in (bank, ext_tr, ext_va) for s in b.samples]
    train3, val3 = bank.indices("train"), bank.indices("validation")
    ext_train_pos = n3 + np.arange(ntr)
    ext_val_pos = n3 + ntr + np.arange(len(ext_va))
    primary_eligible = train3[eligible[train3]]
    external_eligible = ext_train_pos[eligible[ext_train_pos]]

    # --- E0: the Worklog 69 C1 contract ---
    groups = train_sequence_groups([s.sequence_id for s in bank.samples], train3, eligible[:n3])
    e0_sampler = EpochSampler("S0_FRAME_UNIFORM", train3, groups, CONFIG["seed"])
    e0, e0_info, e0_counts = train(e0_sampler, visual, geometry, targets, masks, train3, val3, sequence_ids, device)
    s0_record = w71_report["training"]["S0_FRAME_UNIFORM"]
    curve_diff = max(max(abs(a["train_mae"] - b["train_mae"]), abs(a["validation_mae"] - b["validation_mae"]))
                     for a, b in zip(e0_info["curve"], s0_record["curve"]))
    rows = json.loads((args.w71_dir / "rows_heldout.json").read_text())
    positions = np.array([r["position"] for r in rows])
    pred_e0 = predict(e0, visual, geometry, positions, device)
    c1 = np.array([[r[f"{seg}_f_c1"] for seg in SEGMENT_NAMES] for r in rows])
    reproduction = {"curve_max_abs_diff": curve_diff, "selected_epoch": e0_info["selection"]["epoch"],
                    "w69_selected_epoch": s0_record["selection"]["epoch"],
                    "heldout_prediction_max_abs_diff_vs_c1": float(np.abs(pred_e0 - c1).max())}
    if curve_diff > 1e-6 or reproduction["selected_epoch"] != reproduction["w69_selected_epoch"] \
            or reproduction["heldout_prediction_max_abs_diff_vs_c1"] > 1e-5:
        raise SystemExit(f"E0 does not reproduce Worklog 69 C1: {reproduction}")

    # --- E1: identical graph/init/budget, source-balanced sampler ---
    e1_sampler = SourceBalancedSampler(primary_eligible, external_eligible, e0_sampler.draws_per_epoch, CONFIG["seed"])
    e1, e1_info, e1_counts = train(e1_sampler, visual, geometry, targets, masks, train3, val3, sequence_ids, device)
    if e1_info["draws"] != e0_info["draws"] or e1_info["steps"] != e0_info["steps"] \
            or e1_info["parameters"] != e0_info["parameters"]:
        raise SystemExit("E1 optimization budget / graph differs from E0")
    source_draws = {"3DPW": sum(v for k, v in e1_counts.items() if k.startswith("3dpw:")),
                    "BEDLAM": sum(v for k, v in e1_counts.items() if k.startswith("bedlam:"))}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, model, info in (("E0_3DPW_ONLY", e0, e0_info), ("E1_3DPW_PLUS_EXTERNAL", e1, e1_info)):
        torch.save({"candidate": name, "config": CONFIG, "cache_identity": ident3, "selection": info["selection"],
                    "state_dict": model.state_dict()}, args.out_dir / f"{name}.pt")

    # --- 3DPW held-out evaluation (Worklog 71 rows, unchanged) ---
    pred_e1 = predict(e1, visual, geometry, positions, device)
    sizes = np.asarray([bank.samples[p].image_size for p in positions], float)
    planes, _ = observed_plane(bank.arrays["input_2d"][positions], sizes)
    gt_vec = segment_vectors(bank.arrays["target_3d"][positions])
    for i, r in enumerate(rows):
        for k, seg in enumerate(SEGMENT_NAMES):
            g = gt_vec[seg][i]
            for m, pred in (("e0", pred_e0), ("e1", pred_e1)):
                f = float(pred[i, k])
                r[f"{seg}_f_{m}"] = f
                r[f"{seg}_{m}_raw_plane"] = float(angle_degrees(reconstruct_direction(np.array([f]), planes[seg][i:i + 1]),
                                                                (g / np.linalg.norm(g))[None])[0])

    def continuous(subset):
        out = {"n": len(subset)}
        for seg in SEGMENT_NAMES:
            fg = np.array([r[f"{seg}_f_gt"] for r in subset])
            f = {m: np.array([r[f"{seg}_f_{m}"] for r in subset]) for m in EVAL_MODELS}
            err = {m: {"abs_f": np.abs(f[m] - fg), "elevation": np.abs(elevation_degrees(f[m]) - elevation_degrees(fg)),
                       "full": np.array([r[f"{seg}_{m}_raw_plane"] for r in subset])} for m in EVAL_MODELS}
            block = {m: {k2: _stats(v) for k2, v in err[m].items()} for m in EVAL_MODELS}
            block["e1_minus_e0"] = {k2: {**_stats(err["e1"][k2] - err["e0"][k2]),
                                         "improved_share": float(np.mean(err["e1"][k2] < err["e0"][k2]))} for k2 in err["e1"]}
            block["by_abs_gt_f"] = {f"{lo:.3f}-{min(hi, 1):.3f}": {
                "n": int(((np.abs(fg) >= lo) & (np.abs(fg) < hi)).sum()),
                **{m: float(err[m]["abs_f"][(np.abs(fg) >= lo) & (np.abs(fg) < hi)].mean())
                   for m in EVAL_MODELS if ((np.abs(fg) >= lo) & (np.abs(fg) < hi)).any()}} for lo, hi in F_BINS}
            block["sign_magnitude"] = sign_magnitude({m: f[m] for m in ("e0", "e1")}, fg, STABLE)
            out[seg] = block
        return out

    test_rows = [r for r in rows if bank.samples[r["position"]].split == "test"]
    val_rows = [r for r in rows if bank.samples[r["position"]].split == "validation"]
    scopes = {"test_all": test_rows, "validation_all": val_rows, "heldout_all": rows,
              "validation_clean_dancing_hug": [r for r in val_rows if any(c in r["sequence_id"] for c in CLEAN)],
              "validation_crosscountry_only": [r for r in val_rows if "crosscountry" in r["sequence_id"]],
              "all_other_heldout": [r for r in rows if not any(c in r["sequence_id"] for c in CLEAN) and "crosscountry" not in r["sequence_id"]]}
    ownership = {}
    for scope_name, scope_rows in (("test", test_rows), ("validation", val_rows), ("heldout", rows)):
        by_seq = defaultdict(list)
        for r in scope_rows:
            by_seq[r["sequence_id"]].append(r)
        ownership[scope_name] = {}
        for seg in SEGMENT_NAMES:
            gains = {sid: float(sum(abs(r[f"{seg}_f_e0"] - r[f"{seg}_f_gt"]) - abs(r[f"{seg}_f_e1"] - r[f"{seg}_f_gt"])
                                    for r in group)) for sid, group in by_seq.items()}
            total, positive = sum(gains.values()), sum(v for v in gains.values() if v > 0)
            ranked = sorted(gains.items(), key=lambda kv: kv[1])
            ownership[scope_name][seg] = {
                "sequences": len(gains), "improved": sum(v > 0 for v in gains.values()),
                "regressed": sum(v < 0 for v in gains.values()), "net_abs_f_gain": total,
                "largest_sequence": ranked[-1][0], "largest_share_of_net_gain": ranked[-1][1] / total if total > 0 else None,
                "largest_share_of_positive_gain": ranked[-1][1] / positive if positive > 0 else None,
                "worst_regressions": [{"sequence_id": s, "abs_f_sum_change": -v} for s, v in ranked[:3]]}

    # --- external learnability (secondary) ---
    def external_eval(pos):
        out = {}
        pe0, pe1 = predict(e0, visual, geometry, pos, device), predict(e1, visual, geometry, pos, device)
        for k, seg in enumerate(SEGMENT_NAMES):
            mk = masks[pos, k]
            fg = targets[pos, k][mk]
            out[seg] = {"rows": int(mk.sum()),
                        "abs_f": {m: _stats(np.abs(p[mk, k] - fg)) for m, p in (("e0", pe0), ("e1", pe1))},
                        "sign_magnitude": sign_magnitude({"e0": pe0[mk, k], "e1": pe1[mk, k]}, fg, STABLE)}
        return out

    # --- domain diagnostics (descriptive only) ---
    rng = np.random.default_rng(CONFIG["seed"])
    feats = arm_features(np.concatenate([r3, rtr]))
    t3, te = primary_eligible, external_eligible
    pairs = 20000

    def pair_cos(a, b):
        i, j = rng.choice(a, pairs), rng.choice(b, pairs)
        keep = i != j
        return _stats((feats[i[keep]] * feats[j[keep]]).sum(1))

    nn_cos = []
    f3 = torch.as_tensor(feats[t3], device=device, dtype=torch.float32)
    for s in range(0, len(te), 4096):
        q = torch.as_tensor(feats[te[s:s + 4096]], device=device, dtype=torch.float32)
        nn_cos += (q @ f3.T).max(1).values.cpu().tolist()
    nn_3dpw = []
    for s in range(0, len(t3), 4096):
        q = torch.as_tensor(feats[t3[s:s + 4096]], device=device, dtype=torch.float32)
        sim = q @ f3.T
        sim[torch.arange(len(q)), torch.arange(s, s + len(q))] = -2
        nn_3dpw += sim.max(1).values.cpu().tolist()

    def source_profile(b, pos_local, cache_dir):
        boxes = np.load(cache_dir / "readout.npz")["crop_box_xy_side"][pos_local]
        size = np.array([b.samples[p].image_size for p in pos_local], float)
        inp, valid, tgt, tv = (b.arrays[k][pos_local] for k in ("input_2d", "input_valid", "target_3d", "target_valid"))
        f, m = forward_targets(tgt, tv)
        upper, lower = tgt[:, 12] - tgt[:, 11], tgt[:, 13] - tgt[:, 12]
        cosang = (-(upper * lower).sum(1)) / np.maximum(np.linalg.norm(upper, axis=1) * np.linalg.norm(lower, axis=1), 1e-9)
        arm_ok = tv[:, 11] & tv[:, 12] & tv[:, 13]
        res = defaultdict(int)
        for s in size:
            res[f"{int(s[0])}x{int(s[1])}"] += 1
        return {"rows": int(len(pos_local)),
                "detector_valid_share": {n: float(valid[:, j].mean()) for j, n in ((11, "left_shoulder"), (12, "left_elbow"), (13, "left_wrist"))},
                "detector_confidence_arm": _stats(inp[:, 11:14, 2][valid[:, 11:14]]),
                "valid_joints_per_row": _stats(valid.sum(1)),
                "crop_side_over_image_height": _stats(boxes[:, 2] / size[:, 1]),
                "crop_side_px": _stats(boxes[:, 2]),
                "image_resolution": dict(res),
                "target_abs_f": {seg: _stats(np.abs(f[m[:, k], k])) for k, seg in enumerate(SEGMENT_NAMES)},
                "target_out_of_plane_share": {seg: float((np.abs(f[m[:, k], k]) >= 0.5).mean()) for k, seg in enumerate(SEGMENT_NAMES)},
                "elbow_bend_degrees": _stats(np.degrees(np.arccos(np.clip(cosang[arm_ok], -1, 1))))}

    def families(b, pos_local):
        out = defaultdict(int)
        for p in pos_local:
            sid = b.samples[p].sequence_id
            if sid.startswith("3dpw:"):
                out["3dpw_" + sid.split(":")[1].split("_")[0]] += 1
            else:
                scene = sid.split(":")[1]
                out["bedlam_" + next((kw for kw in ("suburb", "highSchoolGym", "bigOffice", "archViz", "stadium")
                                      if kw in scene), "hdri")] += 1
        return dict(out)

    tr_local = np.arange(ntr)[eligible[ext_train_pos]]
    domain = {"3DPW_train": {**source_profile(bank, primary_eligible, args.cache_dir), "families": families(bank, primary_eligible)},
              "BEDLAM_train": {**source_profile(ext_tr, tr_local, args.external_train_cache), "families": families(ext_tr, tr_local)},
              "arm_readout_cosine": {"within_3dpw_pairs": pair_cos(t3, t3), "within_bedlam_pairs": pair_cos(te, te),
                                     "cross_source_pairs": pair_cos(t3, te),
                                     "bedlam_row_nearest_3dpw_train": _stats(nn_cos),
                                     "3dpw_row_nearest_other_3dpw_train": _stats(nn_3dpw)}}

    report = {"schema": "animcv_external_visual_supervision_transfer_v1", "config": CONFIG, "cache_identity": ident3,
              "external": {"train_rows": ntr, "train_eligible": int(len(external_eligible)), "validation_rows": len(ext_va),
                           "validation_eligible": int(eligible[ext_val_pos].sum()),
                           "train_bank_digest": ext_tr.content_digest(), "validation_bank_digest": ext_va.content_digest(),
                           "train_sequences": len({s.sequence_id for s in ext_tr.samples}),
                           "validation_sequences": len({s.sequence_id for s in ext_va.samples})},
              "e0_reproduction": reproduction,
              "training": {"E0_3DPW_ONLY": {k: e0_info[k] for k in ("selection", "final", "curve", "draws", "steps", "eligible_draws", "parameters")},
                           "E1_3DPW_PLUS_EXTERNAL": {**{k: e1_info[k] for k in ("selection", "final", "curve", "draws", "steps", "eligible_draws", "parameters")},
                                                     "source_draws": source_draws}},
              "continuous": {k: continuous(v) for k, v in scopes.items()}, "ownership": ownership,
              "external_validation": external_eval(ext_val_pos), "external_train_fit": external_eval(ext_train_pos),
              "domain": domain}
    (args.out_dir / "rows_heldout.json").write_text(json.dumps(rows))
    path = args.out_dir / "report.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps({"sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "e0_reproduction": reproduction,
                      "selection": {"E0": e0_info["selection"], "E1": e1_info["selection"]}, "source_draws": source_draws}, indent=1))


if __name__ == "__main__":
    main()
