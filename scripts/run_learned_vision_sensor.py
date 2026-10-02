#!/usr/bin/env python3
"""DIAGNOSTIC (Worklog 68): G0_ZERO_VISION vs G1_QWEN_VISION learned left-arm near/far sensor.

Inputs to the sensor: frozen Qwen3-VL readout (or exact zeros) + runtime pair
geometry.  Targets: GT ordering classes (training/evaluation only).  H0,
Depth Anything and the Worklog 67 VLM are read-only references and the
downstream V1 selector's branch sources; none is a sensor input.
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
from framepose.contract import JOINT_INDEX
from framepose.learned_vision_sensor import (
    ARM, CLASSES, PAIRS, STABLE_SINE, build_sensor, pair_geometry, pair_labels, sensor_inputs,
)
from framepose.relational_depth_fusion import depth_relations
from framepose.train import geometry_tensor
from framepose.vlm_depth_advisor import shuffled_donors, v0_select

CANDIDATES = ("G0_ZERO_VISION", "G1_QWEN_VISION")
CLEAN = ("courtyard_dancing_00", "courtyard_hug_00")
F_BINS = ((0.0, STABLE_SINE), (STABLE_SINE, 0.5), (0.5, 1.0001))
CONFIG = {"optimizer": "AdamW", "learning_rate": 3e-4, "minimum_learning_rate": 1e-5, "weight_decay": 1e-4,
          "schedule": "cosine per step", "epochs": 200, "batch_size": 256, "seed": 1337, "evaluate_every": 10,
          "loss": "masked 3-class cross-entropy, unweighted", "selection": "validation masked cross-entropy",
          "mixed_precision": False}
STATE_SIGN = {"FIRST_CLOSER": 1, "SECOND_CLOSER": -1, "UNCLEAR": 0, "UNKNOWN": 0}


def _stats(v):
    v = np.asarray([x for x in v if x is not None and np.isfinite(x)], float)
    return {"n": int(len(v)), "mean": float(v.mean()), "p50": float(np.percentile(v, 50)),
            "p95": float(np.percentile(v, 95))} if len(v) else {"n": 0}


def _paired(a, b):
    d = np.asarray(a, float) - np.asarray(b, float)
    return {**_stats(d), "improved_share": float(np.mean(d < 0)) if len(d) else None,
            "worsened_share": float(np.mean(d > 0)) if len(d) else None}


def train(inputs, geometry, labels, masks, train_pos, val_pos, device):
    import torch

    torch.manual_seed(CONFIG["seed"])
    model = build_sensor(inputs.shape[-1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"], weight_decay=CONFIG["weight_decay"])
    x = torch.as_tensor(inputs, device=device)
    g = torch.as_tensor(geometry, device=device)
    y = torch.as_tensor(labels, device=device)
    m = torch.as_tensor(masks.astype(np.float32), device=device)
    ce = torch.nn.CrossEntropyLoss(reduction="none")

    def masked_loss(index):
        logits = model(x[index], g[index])
        loss = ce(logits.reshape(-1, 3), y[index].reshape(-1)).reshape(-1, 3)
        w = m[index]
        return (loss * w).sum() / w.sum().clamp_min(1.0)

    def val_loss():
        model.eval()
        with torch.no_grad():
            total = count = 0.0
            for s in range(0, len(val_pos), 1024):
                index = torch.as_tensor(val_pos[s:s + 1024], device=device)
                logits = model(x[index], g[index])
                loss = ce(logits.reshape(-1, 3), y[index].reshape(-1)).reshape(-1, 3)
                total += float((loss * m[index]).sum())
                count += float(m[index].sum())
        model.train()
        return total / max(count, 1.0)

    gen = torch.Generator(device="cpu").manual_seed(CONFIG["seed"])
    steps = math.ceil(len(train_pos) / CONFIG["batch_size"]) * CONFIG["epochs"]
    step, best, best_state, telemetry = 0, {"epoch": None, "validation_ce": None}, None, []
    for epoch in range(CONFIG["epochs"]):
        model.train()
        order = train_pos[torch.randperm(len(train_pos), generator=gen).numpy()]
        total = 0.0
        for s in range(0, len(order), CONFIG["batch_size"]):
            index = torch.as_tensor(order[s:s + CONFIG["batch_size"]], device=device)
            progress = min(step / max(steps - 1, 1), 1.0)
            lr = CONFIG["minimum_learning_rate"] + (CONFIG["learning_rate"] - CONFIG["minimum_learning_rate"]) * 0.5 * (1 + math.cos(math.pi * progress))
            for group in opt.param_groups:
                group["lr"] = lr
            opt.zero_grad(set_to_none=True)
            loss = masked_loss(index)
            loss.backward()
            opt.step()
            total += float(loss.detach()) * len(index)
            step += 1
        record = {"epoch": epoch, "train_ce": total / len(order)}
        if epoch == CONFIG["epochs"] - 1 or (epoch + 1) % CONFIG["evaluate_every"] == 0:
            record["validation_ce"] = val_loss()
            if best["epoch"] is None or record["validation_ce"] < best["validation_ce"]:
                best = {"epoch": epoch, "validation_ce": record["validation_ce"]}
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        telemetry.append(record)
    model.load_state_dict(best_state)
    model.eval()
    return model, {"selection": best, "telemetry": telemetry,
                   "parameters": int(sum(p.numel() for p in model.parameters()))}


def predict(model, inputs, geometry, positions, device, visual_positions=None):
    import torch

    vp = positions if visual_positions is None else visual_positions
    out = []
    with torch.no_grad():
        for s in range(0, len(positions), 1024):
            v = torch.as_tensor(np.asarray(inputs[np.asarray(vp[s:s + 1024])]), device=device)
            g = torch.as_tensor(geometry[np.asarray(positions[s:s + 1024])], device=device)
            out.append(torch.softmax(model(v, g), dim=-1).cpu().numpy())
    return np.concatenate(out)


def classification(pred_states, labels, stable_mask):
    """3-class metrics plus stable FIRST-vs-SECOND accounting for one pair."""
    p = np.array([CLASSES.index(s) for s in pred_states])
    y = np.asarray(labels)
    conf = np.zeros((3, 3), int)
    for a, b in zip(y, p):
        conf[a, b] += 1
    recalls = [conf[c, c] / conf[c].sum() for c in range(3) if conf[c].sum()]
    f1 = []
    for c in range(3):
        tp = conf[c, c]
        prec = tp / conf[:, c].sum() if conf[:, c].sum() else 0.0
        rec = tp / conf[c].sum() if conf[c].sum() else 0.0
        f1.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    st = np.asarray(stable_mask)
    sy, sp = y[st], p[st]
    stable_recalls = [float(np.mean(sp[sy == c] == c)) for c in (0, 1) if (sy == c).any()]
    return {"n": int(len(y)), "coverage_non_unclear": float(np.mean(p != 2)) if len(p) else None,
            "accuracy": float(np.mean(p == y)) if len(y) else None,
            "balanced_accuracy": float(np.mean(recalls)) if recalls else None,
            "macro_f1": float(np.mean(f1)), "confusion_true_rows_pred_cols": conf.tolist(),
            "unclear_precision": float(conf[2, 2] / conf[:, 2].sum()) if conf[:, 2].sum() else None,
            "unclear_recall": float(conf[2, 2] / conf[2].sum()) if conf[2].sum() else None,
            "prediction_distribution": dict(Counter(CLASSES[i] for i in p)),
            "stable_rows": int(st.sum()),
            "stable_accuracy": float(np.mean(sp == sy)) if st.any() else None,
            "stable_first_vs_second_balanced_accuracy": float(np.mean(stable_recalls)) if len(stable_recalls) == 2 else None}


def reference_stable(signs, labels, stable):
    """Sign-only references (H0, DA, VLM) on stable rows: accuracy and balanced accuracy."""
    s, y = np.asarray(signs)[stable], np.asarray(labels)[stable]
    truth = np.where(y == 0, 1, -1)
    rec = [float(np.mean(s[truth == t] == t)) for t in (1, -1) if (truth == t).any()]
    return {"stable_rows": int(stable.sum()), "stable_accuracy": float(np.mean(s == truth)) if len(s) else None,
            "stable_balanced_accuracy": float(np.mean(rec)) if len(rec) == 2 else None,
            "zero_or_unclear_share": float(np.mean(s == 0)) if len(s) else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("bank", "cache_dir", "evidence_dir", "w65_dir", "w67_dir", "image_root", "out_dir"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    import torch

    bank = load_bank(args.bank)
    manifest = json.loads((args.cache_dir / "manifest.json").read_text())
    readout_path = args.cache_dir / "readout.npz"
    if (manifest["identity"]["bank_content_digest"] != bank.content_digest()
            or manifest["identity"]["sample_ids_sha256"] != hashlib.sha256("\n".join(s.sample_id for s in bank.samples).encode()).hexdigest()
            or manifest["readout_npz_sha256"] != hashlib.sha256(readout_path.read_bytes()).hexdigest()):
        raise SystemExit("visual cache identity mismatch")
    cache = np.load(readout_path)
    readouts = cache["readout"]
    geometry = pair_geometry(geometry_tensor(bank))
    f_gt, gt_masks = forward_targets(bank.arrays["target_3d"], bank.arrays["target_valid"])
    labels, _ = pair_labels(f_gt, gt_masks)
    valid_in = bank.arrays["input_valid"]
    detector = np.stack([valid_in[:, JOINT_INDEX[ARM[a]]] & valid_in[:, JOINT_INDEX[ARM[b]]] for _, a, b, _ in PAIRS], axis=1)
    masks = gt_masks & detector
    train_pos, val_pos, test_pos = bank.indices("train"), bank.indices("validation"), bank.indices("test")
    device = torch.device(args.device)

    inputs = {c: sensor_inputs(readouts, cache["joint_valid"], c) for c in CANDIDATES}
    if np.any(inputs["G0_ZERO_VISION"] != 0) or not np.array_equal(inputs["G1_QWEN_VISION"], readouts):
        raise SystemExit("G0/G1 visual input contract violated")
    models, training = {}, {}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for c in CANDIDATES:
        models[c], training[c] = train(inputs[c], geometry, labels, masks, train_pos, val_pos, device)
        torch.save({"candidate": c, "config": CONFIG, "cache_identity": manifest["identity"],
                    "selection": training[c]["selection"], "state_dict": models[c].state_dict()},
                   args.out_dir / f"{c}.pt")
    train_class_counts = {name: dict(Counter(CLASSES[i] for i in labels[train_pos, k][masks[train_pos, k]]))
                          for k, (name, *_rest) in enumerate(PAIRS)}

    # Held-out evaluation rows: the Worklog 64-67 eligible rows.
    rows = json.loads((args.w65_dir / "rows_validation.json").read_text()) + \
        json.loads((args.w65_dir / "rows_test.json").read_text())
    test_ids = {r["position"] for r in json.loads((args.w65_dir / "rows_test.json").read_text())}
    positions = np.array([r["position"] for r in rows])
    keys = [(r["sequence_id"], r["frame_index"]) for r in rows]
    donors = positions[np.array(shuffled_donors(keys))]
    probs = {"g0": predict(models["G0_ZERO_VISION"], inputs["G0_ZERO_VISION"], geometry, positions, device),
             "g1": predict(models["G1_QWEN_VISION"], inputs["G1_QWEN_VISION"], geometry, positions, device),
             "g1_shuffled": predict(models["G1_QWEN_VISION"], inputs["G1_QWEN_VISION"], geometry, positions, device,
                                    visual_positions=donors)}
    evidence = np.load(args.evidence_dir / "evidence.npz")
    relations, relation_ok = depth_relations(evidence["forward_evidence"], evidence["available"])
    vlm = {}
    for line in (args.w67_dir / "responses.jsonl").read_text().splitlines():
        item = json.loads(line)
        if item["kind"] == "real":
            vlm[item["sample_id"]] = item["states"]

    sizes = np.asarray([bank.samples[p].image_size for p in positions], float)
    planes, _ = observed_plane(bank.arrays["input_2d"][positions], sizes)
    gt_vec = segment_vectors(bank.arrays["target_3d"][positions])
    for i, r in enumerate(rows):
        p = positions[i]
        r["sample_id"] = bank.samples[p].sample_id
        r["donor_sample_id"] = bank.samples[donors[i]].sample_id
        for k, (field, _, _, seg) in enumerate(PAIRS):
            r[f"{seg}_label"] = int(labels[p, k])
            r[f"{seg}_relation"] = float(relations[p, k]) if relation_ok[p, k] else None
            r[f"{seg}_vlm"] = vlm[r["sample_id"]][field] if vlm.get(r["sample_id"]) else "UNKNOWN"
            for m in probs:
                r[f"{seg}_{m}_state"] = CLASSES[int(np.argmax(probs[m][i, k]))]
                r[f"{seg}_{m}_probs"] = [float(x) for x in probs[m][i, k]]
            g = gt_vec[seg][i]
            gu = (g / np.linalg.norm(g))[None]
            for sel, state_key in (("v0", f"{seg}_vlm"), ("v1", f"{seg}_g1_state"), ("v1_shuffled", f"{seg}_g1_shuffled_state")):
                f, source = v0_select(r[f"{seg}_f_h0"], r[f"{seg}_f_d1"], r[state_key])
                r[f"{seg}_f_{sel}"], r[f"{seg}_{sel}_source"] = f, source
                r[f"{seg}_{sel}_raw_plane"] = float(angle_degrees(reconstruct_direction(np.array([f]), planes[seg][i:i + 1]), gu)[0])

    def sensor_block(subset):
        out = {}
        for k, (field, _, _, seg) in enumerate(PAIRS):
            y = np.array([r[f"{seg}_label"] for r in subset])
            stable = y != 2
            out[field] = {m: classification([r[f"{seg}_{m}_state"] for r in subset], y, stable)
                          for m in ("g0", "g1", "g1_shuffled")}
            out[field]["references"] = {
                "h0": reference_stable(np.sign([r[f"{seg}_f_h0"] for r in subset]), y, stable),
                "depth_anything": reference_stable(np.sign([r[f"{seg}_relation"] or 0.0 for r in subset]), y, stable),
                "w67_language_vlm": reference_stable(np.array([STATE_SIGN[r[f"{seg}_vlm"]] for r in subset]), y, stable),
                "d1": reference_stable(np.sign([r[f"{seg}_f_d1"] for r in subset]), y, stable)}
            correct = {m: np.array([CLASSES.index(r[f"{seg}_{m}_state"]) for r in subset]) == y for m in ("g0", "g1", "g1_shuffled")}
            out[field]["paired_g1_vs_g0_stable"] = {
                "g1_right_g0_wrong": int(np.sum(correct["g1"][stable] & ~correct["g0"][stable])),
                "g0_right_g1_wrong": int(np.sum(~correct["g1"][stable] & correct["g0"][stable]))}
            out[field]["paired_g1_vs_shuffled_stable"] = {
                "real_right_shuffled_wrong": int(np.sum(correct["g1"][stable] & ~correct["g1_shuffled"][stable])),
                "shuffled_right_real_wrong": int(np.sum(~correct["g1"][stable] & correct["g1_shuffled"][stable])),
                "state_change_rate": float(np.mean([r[f"{seg}_g1_state"] != r[f"{seg}_g1_shuffled_state"] for r in subset]))}
        return out

    def continuous(subset):
        out = {"n": len(subset)}
        for seg in SEGMENT_NAMES:
            fg = np.array([r[f"{seg}_f_gt"] for r in subset])
            err, block = {}, {}
            for m in ("h0", "d1", "v0", "v1", "v1_shuffled"):
                f = np.array([r[f"{seg}_f_{m}"] for r in subset])
                err[m] = {"abs_f": np.abs(f - fg), "elevation": np.abs(elevation_degrees(f) - elevation_degrees(fg)),
                          "full": np.array([r[f"{seg}_{m}_raw_plane"] for r in subset])}
                block[m] = {k: _stats(v) for k, v in err[m].items()}
            block["v1_minus_h0"] = {k: _paired(err["v1"][k], err["h0"][k]) for k in err["v1"]}
            block["v1_minus_d1"] = {k: _paired(err["v1"][k], err["d1"][k]) for k in err["v1"]}
            block["v1_minus_v0"] = {k: _paired(err["v1"][k], err["v0"][k]) for k in err["v1"]}
            block["by_abs_gt_f"] = {f"{lo:.3f}-{min(hi, 1):.3f}": {
                "n": int(((np.abs(fg) >= lo) & (np.abs(fg) < hi)).sum()),
                **{m: float(err[m]["abs_f"][(np.abs(fg) >= lo) & (np.abs(fg) < hi)].mean())
                   for m in err if ((np.abs(fg) >= lo) & (np.abs(fg) < hi)).any()}} for lo, hi in F_BINS}
            out[seg] = block
        return out

    def conflict_controls(subset):
        out = {}
        for seg in SEGMENT_NAMES:
            c = [r for r in subset if np.sign(r[f"{seg}_f_h0"]) * np.sign(r[f"{seg}_f_d1"]) < 0]
            fg = np.array([r[f"{seg}_f_gt"] for r in c])
            e = {m: np.abs(np.array([r[f"{seg}_f_{m}"] for r in c]) - fg) for m in ("h0", "d1", "v0", "v1", "v1_shuffled")}
            d1_better = e["d1"] < e["h0"]
            picks = {m: np.array([r[f"{seg}_{m}_source"] == "d1_vlm_agrees" for r in c]) for m in ("v0", "v1", "v1_shuffled")}
            out[seg] = {"conflict_rows": len(c), "d1_better_share": float(d1_better.mean()) if c else None,
                        **{f"{m}_abs_f_mean": float(e[m].mean()) for m in e},
                        "oracle_branch_abs_f_mean": float(np.minimum(e["h0"], e["d1"]).mean()) if c else None,
                        **{f"{m}_picks_d1_when_d1_better": float(picks[m][d1_better].mean()) for m in picks if d1_better.any()},
                        **{f"{m}_picks_d1_when_h0_better": float(picks[m][~d1_better].mean()) for m in picks if (~d1_better).any()}}
        return out

    scopes = {"heldout_all": rows, "test_all": [r for r in rows if r["position"] in test_ids],
              "validation_all": [r for r in rows if r["position"] not in test_ids],
              "validation_clean_dancing_hug": [r for r in rows if r["position"] not in test_ids and any(c in r["sequence_id"] for c in CLEAN)],
              "validation_crosscountry_only": [r for r in rows if "crosscountry" in r["sequence_id"]],
              "all_other_heldout": [r for r in rows if not any(c in r["sequence_id"] for c in CLEAN) and "crosscountry" not in r["sequence_id"]]}
    per_sequence = {}
    groups = defaultdict(list)
    for r in rows:
        groups[r["sequence_id"]].append(r)
    for k, (field, _, _, seg) in enumerate(PAIRS):
        better = shuffled_better = 0
        gains = {}
        for sid, group in groups.items():
            y = np.array([r[f"{seg}_label"] for r in group])
            st = y != 2
            if not st.any():
                continue
            acc = {m: np.mean(np.array([CLASSES.index(r[f"{seg}_{m}_state"]) for r in group])[st] == y[st]) for m in ("g0", "g1", "g1_shuffled")}
            better += acc["g1"] > acc["g0"]
            shuffled_better += acc["g1"] > acc["g1_shuffled"]
            gains[sid] = float((acc["g1"] - acc["g0"]) * st.sum())
        total = sum(gains.values())
        top = max(gains.items(), key=lambda kv: kv[1]) if gains else (None, 0.0)
        per_sequence[field] = {"sequences": len(gains), "g1_better_than_g0": int(better),
                               "g1_better_than_shuffled": int(shuffled_better),
                               "largest_single_sequence_share_of_net_gain": top[1] / total if total > 0 else None,
                               "largest_sequence": top[0]}

    owner_src = json.loads((args.w67_dir / "analysis" / "analysis.json").read_text())["owner_cases"]
    by_key = {(r["sequence_id"], r["frame_index"]): r for r in rows}
    owner = {label: {"sequence_id": c["sequence_id"], "frame_index": c["frame_index"],
                     "row": by_key.get((c["sequence_id"], c["frame_index"]))} for label, c in owner_src.items()}

    report = {"schema": "animcv_learned_vision_near_far_sensor_v1", "config": CONFIG,
              "cache_identity": manifest["identity"], "readout_npz_sha256": manifest["readout_npz_sha256"],
              "training": training, "train_class_counts": train_class_counts,
              "sensor": {k: sensor_block(v) for k, v in scopes.items()},
              "continuous": {k: continuous(v) for k, v in scopes.items()},
              "conflict_controls": {k: conflict_controls(v) for k, v in scopes.items()},
              "per_sequence": per_sequence, "owner_cases": owner,
              "h0_in_sample_note": "H0 is not a sensor input or target; GT ordering labels are the only supervision"}
    (args.out_dir / "rows_heldout.json").write_text(json.dumps(rows))
    path = args.out_dir / "report.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps({"sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                      "selection": {k: v["selection"] for k, v in training.items()},
                      "parameters": {k: v["parameters"] for k, v in training.items()},
                      "train_class_counts": train_class_counts}, indent=1))


if __name__ == "__main__":
    main()
