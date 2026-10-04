#!/usr/bin/env python3
"""DIAGNOSTIC (Worklog 72): S1_SEQUENCE_BALANCED vs S2_VISUAL_CORESET_BALANCED for the fixed Worklog 69 head.

S1 is the Worklog 71 run (checkpoint and per-row predictions reused; its
sampler is replayed for accounting and must reproduce the recorded counts).
S2 uses the identical sampler code, seed and sequence draws; only each
sequence's candidate pool changes from all eligible frames to its fixed
K-frame visual coreset.  Training uses the Worklog 71 ``train`` function
unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from framepose.arm_depth_probe import (
    SEGMENT_NAMES, angle_degrees, elevation_degrees, forward_targets, observed_plane, reconstruct_direction,
    segment_vectors,
)
from framepose.bank import load_bank
from framepose.continuous_vision_depth import candidate_visual, predict, sign_magnitude
from framepose.learned_vision_sensor import pair_geometry
from framepose.train import geometry_tensor
from framepose.visual_coreset import arm_features, coreset_size, farthest_point_coreset, redundancy
from framepose.visual_supervision_sampling import EpochSampler, contribution_summary, train_sequence_groups
from run_visual_supervision_diversity import CONFIG, train

STABLE = math.sin(math.radians(10))
CLEAN = ("courtyard_dancing_00", "courtyard_hug_00")
F_BINS = ((0.0, STABLE), (STABLE, 0.5), (0.5, 1.0001))
EVAL_MODELS = ("s1", "s2", "s0", "h0", "d1")


def _stats(v):
    v = np.asarray([x for x in v if x is not None and np.isfinite(x)], float)
    return {"n": int(len(v)), "mean": float(v.mean()), "p05": float(np.percentile(v, 5)), "p50": float(np.percentile(v, 50)),
            "p95": float(np.percentile(v, 95))} if len(v) else {"n": 0}


def replay(name, train_pos, groups, epochs, sequence_ids):
    sampler = EpochSampler(name, train_pos, groups, CONFIG["seed"])
    visits: Counter = Counter()
    seq_stream = []
    for _ in range(epochs):
        order = sampler.epoch()
        visits.update(order.tolist())
        seq_stream.append(np.array([sequence_ids[p] for p in order]))
    return visits, seq_stream


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("bank", "cache_dir", "w69_dir", "w71_dir", "out_dir"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    import torch

    bank = load_bank(args.bank)
    manifest = json.loads((args.cache_dir / "manifest.json").read_text())
    readout_path = args.cache_dir / "readout.npz"
    if (manifest["identity"]["bank_content_digest"] != bank.content_digest()
            or manifest["identity"]["sample_ids_sha256"] != hashlib.sha256("\n".join(s.sample_id for s in bank.samples).encode()).hexdigest()
            or manifest["identity"]["model_fingerprint"] != "a0d72ded575eaa4460dad11dd1e313bc69f0403b2c97c3c1ba234af086952904"
            or manifest["readout_npz_sha256"] != hashlib.sha256(readout_path.read_bytes()).hexdigest()):
        raise SystemExit("Worklog 68 cache identity mismatch")
    w69 = torch.load(args.w69_dir / "C1_QWEN_VISION.pt", map_location="cpu", weights_only=False)
    w71_report = json.loads((args.w71_dir / "report.json").read_text())
    w71_s1 = torch.load(args.w71_dir / "S1_SEQUENCE_BALANCED.pt", map_location="cpu", weights_only=False)
    if w69["cache_identity"]["digest"] != manifest["identity"]["digest"] or w71_s1["config"] != CONFIG \
            or w71_s1["sampler"] != "S1_SEQUENCE_BALANCED":
        raise SystemExit("Worklog 69 head / Worklog 71 S1 identity mismatch")

    readouts = np.load(readout_path)["readout"]
    visual = candidate_visual(readouts, "C1_QWEN_VISION")
    geometry = pair_geometry(geometry_tensor(bank))
    targets, masks = forward_targets(bank.arrays["target_3d"], bank.arrays["target_valid"])
    eligible = masks.any(axis=1)
    train_pos, val_pos = bank.indices("train"), bank.indices("validation")
    sequence_ids = [s.sequence_id for s in bank.samples]
    groups = train_sequence_groups(sequence_ids, train_pos, eligible)
    if sum(len(v) for v in groups.values()) != w71_report["composition"]["eligible_rows"] or \
            len(groups) != w71_report["composition"]["sequences"]:
        raise SystemExit("eligible TRAIN population differs from Worklog 71")

    # --- coreset ---
    features = arm_features(readouts)
    k = coreset_size(groups)
    coresets = {sid: farthest_point_coreset(features, pos, k) for sid, pos in groups.items()}
    per_sequence, nn_full, nn_core = {}, [], []
    for sid, pos in groups.items():
        full, core = redundancy(features, pos), redundancy(features, coresets[sid])
        per_sequence[sid] = {"eligible_frames": int(len(pos)), "k": k, "full": full, "coreset": core}
        x = features[pos] @ features[pos].T
        np.fill_diagonal(x, -np.inf)
        nn_full += list(x.max(axis=1))
        c = features[coresets[sid]] @ features[coresets[sid]].T
        np.fill_diagonal(c, -np.inf)
        nn_core += list(c.max(axis=1))
    coreset_report = {"k": k, "k_rule": "minimum eligible TRAIN frames over TRAIN sequences",
                      "sequences": len(groups), "coreset_frames": int(k * len(groups)),
                      "pooled_nearest_neighbour_cosine": {"full_pool": _stats(nn_full), "coreset": _stats(nn_core)},
                      "mean_pairwise_cosine": {"full_pool": _stats([v["full"]["mean_pairwise_cosine"] for v in per_sequence.values()]),
                                               "coreset": _stats([v["coreset"]["mean_pairwise_cosine"] for v in per_sequence.values()])},
                      "per_sequence": per_sequence,
                      "coreset_positions_sha256": hashlib.sha256(json.dumps({s: v.tolist() for s, v in coresets.items()},
                                                                            sort_keys=True).encode()).hexdigest()}

    # --- contribution accounting (sampler replay) ---
    visits1, seq1 = replay("S1_SEQUENCE_BALANCED", train_pos, groups, CONFIG["epochs"], sequence_ids)
    visits2, seq2 = replay("S1_SEQUENCE_BALANCED", train_pos, coresets, CONFIG["epochs"], sequence_ids)
    if not all(np.array_equal(a, b) for a, b in zip(seq1, seq2)):
        raise SystemExit("S1 and S2 sequence draws differ")
    counts1 = Counter(np.concatenate(seq1).tolist())
    recorded = {sid: v["draws"] for sid, v in w71_report["training"]["S1_SEQUENCE_BALANCED"]["contribution"]["per_sequence"].items()}
    if dict(counts1) != recorded:
        raise SystemExit("S1 replay does not reproduce the Worklog 71 recorded contribution")
    core_members = set(np.concatenate(list(coresets.values())).tolist())
    if not set(visits2).issubset(core_members):
        raise SystemExit("S2 drew a frame outside the declared coreset")

    def visit_summary(visits):
        reps = np.array(list(visits.values()))
        return {"draws": int(reps.sum()), "unique_frames_visited": int(len(visits)), "repeat_count": _stats(reps)}
    accounting = {"S1_SEQUENCE_BALANCED": {**visit_summary(visits1), "contribution": contribution_summary(dict(counts1), groups)},
                  "S2_VISUAL_CORESET_BALANCED": {**visit_summary(visits2), "contribution": contribution_summary(dict(counts1), coresets)},
                  "sequence_draws_identical": True, "s1_replay_matches_worklog71": True}

    # --- train S2 (Worklog 71 train function, unchanged) ---
    sampler = EpochSampler("S1_SEQUENCE_BALANCED", train_pos, coresets, CONFIG["seed"])
    model, training, _ = train(sampler, visual, geometry, targets, masks, train_pos, val_pos, sequence_ids, torch.device(args.device))
    if training["draws"] != w71_report["training"]["S1_SEQUENCE_BALANCED"]["draws"] or \
            training["steps"] != w71_report["training"]["S1_SEQUENCE_BALANCED"]["steps"]:
        raise SystemExit("S2 optimization budget differs from S1")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"sampler": "S2_VISUAL_CORESET_BALANCED", "config": CONFIG, "cache_identity": manifest["identity"],
                "coreset_positions_sha256": coreset_report["coreset_positions_sha256"], "k": k,
                "selection": training["selection"], "state_dict": model.state_dict()}, args.out_dir / "S2_VISUAL_CORESET_BALANCED.pt")

    # --- held-out evaluation on the Worklog 71 rows ---
    rows = json.loads((args.w71_dir / "rows_heldout.json").read_text())
    positions = np.array([r["position"] for r in rows])
    pred = predict(model, visual, geometry, positions, torch.device(args.device))
    sizes = np.asarray([bank.samples[p].image_size for p in positions], float)
    planes, _ = observed_plane(bank.arrays["input_2d"][positions], sizes)
    gt_vec = segment_vectors(bank.arrays["target_3d"][positions])
    for i, r in enumerate(rows):
        for kk, seg in enumerate(SEGMENT_NAMES):
            g = gt_vec[seg][i]
            f = float(pred[i, kk])
            r[f"{seg}_f_s2"] = f
            r[f"{seg}_s2_raw_plane"] = float(angle_degrees(reconstruct_direction(np.array([f]), planes[seg][i:i + 1]),
                                                           (g / np.linalg.norm(g))[None])[0])

    def continuous(subset):
        out = {"n": len(subset)}
        for seg in SEGMENT_NAMES:
            fg = np.array([r[f"{seg}_f_gt"] for r in subset])
            f = {m: np.array([r[f"{seg}_f_{m}"] for r in subset]) for m in EVAL_MODELS}
            err = {m: {"abs_f": np.abs(f[m] - fg), "elevation": np.abs(elevation_degrees(f[m]) - elevation_degrees(fg)),
                       "full": np.array([r[f"{seg}_{m}_raw_plane"] for r in subset])} for m in EVAL_MODELS}
            block = {m: {k2: _stats(v) for k2, v in err[m].items()} for m in EVAL_MODELS}
            block["s2_minus_s1"] = {k2: {**_stats(err["s2"][k2] - err["s1"][k2]),
                                         "improved_share": float(np.mean(err["s2"][k2] < err["s1"][k2]))} for k2 in err["s2"]}
            block["by_abs_gt_f"] = {f"{lo:.3f}-{min(hi, 1):.3f}": {
                "n": int(((np.abs(fg) >= lo) & (np.abs(fg) < hi)).sum()),
                **{m: float(err[m]["abs_f"][(np.abs(fg) >= lo) & (np.abs(fg) < hi)].mean())
                   for m in EVAL_MODELS if ((np.abs(fg) >= lo) & (np.abs(fg) < hi)).any()}} for lo, hi in F_BINS}
            block["sign_magnitude"] = sign_magnitude({m: f[m] for m in ("s1", "s2")}, fg, STABLE)
            out[seg] = block
        return out

    test_rows = [r for r in rows if bank.samples[r["position"]].split == "test"]
    val_rows = [r for r in rows if bank.samples[r["position"]].split == "validation"]
    scopes = {"test_all": test_rows, "validation_all": val_rows, "heldout_all": rows,
              "validation_clean_dancing_hug": [r for r in val_rows if any(c in r["sequence_id"] for c in CLEAN)],
              "validation_crosscountry_only": [r for r in val_rows if "crosscountry" in r["sequence_id"]],
              "all_other_heldout": [r for r in rows if not any(c in r["sequence_id"] for c in CLEAN) and "crosscountry" not in r["sequence_id"]]}
    ownership = {}
    for scope_name, scope_rows in (("test", test_rows), ("heldout", rows)):
        by_seq = defaultdict(list)
        for r in scope_rows:
            by_seq[r["sequence_id"]].append(r)
        ownership[scope_name] = {}
        for seg in SEGMENT_NAMES:
            gains = {}
            for sid, group in by_seq.items():
                fg = np.array([r[f"{seg}_f_gt"] for r in group])
                gains[sid] = float((np.abs(np.array([r[f"{seg}_f_s1"] for r in group]) - fg)
                                    - np.abs(np.array([r[f"{seg}_f_s2"] for r in group]) - fg)).sum())
            total = sum(gains.values())
            positive = sum(v for v in gains.values() if v > 0)
            ranked = sorted(gains.items(), key=lambda kv: kv[1])
            ownership[scope_name][seg] = {"sequences": len(gains), "improved": sum(v > 0 for v in gains.values()),
                                          "regressed": sum(v < 0 for v in gains.values()), "net_abs_f_gain": total,
                                          "largest_sequence": ranked[-1][0],
                                          "largest_share_of_net_gain": ranked[-1][1] / total if total > 0 else None,
                                          "largest_share_of_positive_gain": ranked[-1][1] / positive if positive > 0 else None,
                                          "worst_regressions": [{"sequence_id": s, "abs_f_sum_change": -v} for s, v in ranked[:3]]}

    s1_training = w71_report["training"]["S1_SEQUENCE_BALANCED"]
    report = {"schema": "animcv_visual_coreset_diversity_v1", "config": CONFIG, "cache_identity": manifest["identity"],
              "coreset": coreset_report, "accounting": accounting,
              "training": {"S1_SEQUENCE_BALANCED": {k2: s1_training[k2] for k2 in ("selection", "final", "curve", "draws", "steps")},
                           "S2_VISUAL_CORESET_BALANCED": {k2: training[k2] for k2 in ("selection", "final", "curve", "draws", "steps", "eligible_draws")}},
              "continuous": {k2: continuous(v) for k2, v in scopes.items()}, "ownership": ownership}
    (args.out_dir / "rows_heldout.json").write_text(json.dumps(rows))
    path = args.out_dir / "report.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps({"sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "k": k,
                      "nn": coreset_report["pooled_nearest_neighbour_cosine"],
                      "selection": {"S1": s1_training["selection"], "S2": training["selection"]},
                      "final": {"S1": s1_training["final"], "S2": training["final"]}}, indent=1))


if __name__ == "__main__":
    main()
