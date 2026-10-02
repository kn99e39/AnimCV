#!/usr/bin/env python3
"""DIAGNOSTIC (Worklog 67): score Qwen3-VL near/far answers and the V0_VLM_SELECTOR.

GT (bank target_3d forward fractions) is used here only for scoring, after the
population and the VLM answers are fixed.
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
    SEGMENT_NAMES, angle_degrees, elevation_degrees, observed_plane, reconstruct_direction, segment_vectors,
)
from framepose.bank import load_bank
from framepose.relational_depth_fusion import depth_relations
from framepose.vlm_depth_advisor import FIELDS, SEGMENT_FIELD, STATES, ordering_sign, v0_select

STABLE = math.sin(math.radians(10))
CLEAN = ("courtyard_dancing_00", "courtyard_hug_00")
F_BINS = ((0.0, STABLE), (STABLE, 0.5), (0.5, 1.0001))
MODELS = ("h0", "d1", "r1", "v0")


def _stats(values):
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)], float)
    if not len(v):
        return {"n": 0}
    return {"n": int(len(v)), "mean": float(v.mean()), "p50": float(np.percentile(v, 50)),
            "p95": float(np.percentile(v, 95))}


def _paired(a, b):
    d = np.asarray(a, float) - np.asarray(b, float)
    return {**_stats(d), "improved_share": float(np.mean(d < 0)) if len(d) else None,
            "worsened_share": float(np.mean(d > 0)) if len(d) else None}


def _entropy(counter):
    total = sum(counter.values())
    return float(-sum(c / total * math.log2(c / total) for c in counter.values() if c)) if total else None


def ordering_metrics(items, segment):
    """items: rows with vlm state, gt f, h0 f, da relation for one segment."""
    field = SEGMENT_FIELD[segment]
    n = len(items)
    states = Counter(i["vlm"][field] for i in items)
    decided = [i for i in items if ordering_sign(i["vlm"][field]) != 0]
    stable = [i for i in items if abs(i[f"{segment}_f_gt"]) > STABLE]
    stable_decided = [i for i in stable if ordering_sign(i["vlm"][field]) != 0]

    def correct(i):
        return ordering_sign(i["vlm"][field]) == np.sign(i[f"{segment}_f_gt"])

    def recall(sign):
        group = [i for i in stable_decided if np.sign(i[f"{segment}_f_gt"]) == sign]
        return float(np.mean([correct(i) for i in group])) if group else None
    recalls = [r for r in (recall(1), recall(-1)) if r is not None]
    h0_ok = [np.sign(i[f"{segment}_f_h0"]) == np.sign(i[f"{segment}_f_gt"]) for i in stable]
    da_ok = [i[f"{segment}_relation"] is not None and np.sign(i[f"{segment}_relation"]) == np.sign(i[f"{segment}_f_gt"])
             for i in stable]
    return {
        "rows": n, "valid_response_rate": float(np.mean([i["vlm_valid"] for i in items])) if n else None,
        "state_distribution": dict(states),
        "unclear_rate_of_valid": (states.get("UNCLEAR", 0) / max(1, n - states.get("UNKNOWN", 0))),
        "decided_accuracy": float(np.mean([correct(i) for i in decided])) if decided else None,
        "stable_rows": len(stable),
        "stable_decided_accuracy": float(np.mean([correct(i) for i in stable_decided])) if stable_decided else None,
        "stable_balanced_accuracy": float(np.mean(recalls)) if len(recalls) == 2 else None,
        "stable_coverage": len(stable_decided) / len(stable) if stable else None,
        "stable_coverage_times_correctness": (sum(correct(i) for i in stable_decided) / len(stable)) if stable else None,
        "always_h0_stable_accuracy": float(np.mean(h0_ok)) if stable else None,
        "always_da_stable_accuracy": float(np.mean(da_ok)) if stable else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("bank", "evidence_dir", "w65_dir", "w66_dir", "vlm_dir", "out_dir"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    args = parser.parse_args()

    bank = load_bank(args.bank)
    evidence = np.load(args.evidence_dir / "evidence.npz")
    relations, relation_ok = depth_relations(evidence["forward_evidence"], evidence["available"])
    population = json.loads((args.vlm_dir / "population.json").read_text())
    responses = defaultdict(dict)
    parse_status = Counter()
    fences = 0
    for line in (args.vlm_dir / "responses.jsonl").read_text().splitlines():
        item = json.loads(line)
        responses[item["kind"]][item["sample_id"]] = item
        parse_status[(item["kind"], item["parse_status"])] += 1
        fences += item["fence_normalized"]
    w66 = {r["position"]: r for name in ("rows_validation.json", "rows_test.json")
           for r in json.loads((args.w66_dir / name).read_text())}
    rows = []
    for entry in population["rows"]:
        p = entry["position"]
        base = dict(w66[p])
        real = responses["real"].get(entry["sample_id"])
        if real is None:
            raise SystemExit(f"missing real VLM response for {entry['sample_id']}")
        base["vlm"] = {f: (real["states"][f] if real["states"] else "UNKNOWN") for f in FIELDS}
        base["vlm_valid"] = real["parse_status"] == "valid"
        base["primary"] = entry["primary"]
        base["disagree"] = entry["disagree"]
        base["sample_id"] = entry["sample_id"]
        for k, name in enumerate(SEGMENT_NAMES):
            base[f"{name}_relation"] = float(relations[p, k]) if relation_ok[p, k] else None
        rows.append(base)

    # V0 selector and its observed-plane full angle.
    positions = np.array([r["position"] for r in rows])
    sizes = np.asarray([bank.samples[p].image_size for p in positions], float)
    planes, _ = observed_plane(bank.arrays["input_2d"][positions], sizes)
    gt_vec = segment_vectors(bank.arrays["target_3d"][positions])
    for i, r in enumerate(rows):
        for name in SEGMENT_NAMES:
            f, source = v0_select(r[f"{name}_f_h0"], r[f"{name}_f_d1"], r["vlm"][SEGMENT_FIELD[name]])
            r[f"{name}_f_v0"], r[f"{name}_v0_source"] = f, source
            g = gt_vec[name][i]
            r[f"{name}_v0_raw_plane"] = float(angle_degrees(reconstruct_direction(np.array([f]), planes[name][i:i + 1]),
                                                            (g / np.linalg.norm(g))[None])[0])

    def continuous(subset):
        out = {"n": len(subset)}
        for name in SEGMENT_NAMES:
            fg = np.array([r[f"{name}_f_gt"] for r in subset])
            err = {}
            seg = {}
            for m in MODELS:
                f = np.array([r[f"{name}_f_{m}"] for r in subset])
                err[m] = {"abs_f": np.abs(f - fg), "elevation": np.abs(elevation_degrees(f) - elevation_degrees(fg)),
                          "full": np.array([r[f"{name}_{m}_raw_plane"] for r in subset])}
                seg[m] = {k: _stats(v) for k, v in err[m].items()}
            seg["v0_minus_h0"] = {k: _paired(err["v0"][k], err["h0"][k]) for k in err["v0"]}
            seg["v0_minus_d1"] = {k: _paired(err["v0"][k], err["d1"][k]) for k in err["v0"]}
            seg["v0_sources"] = dict(Counter(r[f"{name}_v0_source"] for r in subset))
            seg["by_abs_gt_f"] = {}
            for lo, hi in F_BINS:
                sel = (np.abs(fg) >= lo) & (np.abs(fg) < hi)
                seg["by_abs_gt_f"][f"{lo:.3f}-{min(hi, 1):.3f}"] = {
                    "n": int(sel.sum()), **{f"{m}_abs_f_mean": float(err[m]["abs_f"][sel].mean()) if sel.any() else None
                                            for m in MODELS}}
            out[name] = seg
        return out

    def complementarity(subset, name):
        field = SEGMENT_FIELD[name]
        stable = [r for r in subset if r[f"{name}_relation"] is not None and abs(r[f"{name}_f_gt"]) > STABLE]
        out = {}
        for label, h_ok, d_ok in (("h0_wrong_da_right", False, True), ("h0_right_da_wrong", True, False),
                                  ("both_right", True, True), ("both_wrong", False, False)):
            sel = [r for r in stable
                   if (np.sign(r[f"{name}_f_h0"]) == np.sign(r[f"{name}_f_gt"])) == h_ok
                   and (np.sign(r[f"{name}_relation"]) == np.sign(r[f"{name}_f_gt"])) == d_ok]
            if not sel:
                out[label] = {"n": 0}
                continue
            votes = [ordering_sign(r["vlm"][field]) for r in sel]
            gt = [np.sign(r[f"{name}_f_gt"]) for r in sel]
            fg = np.array([r[f"{name}_f_gt"] for r in sel])
            out[label] = {
                "n": len(sel),
                "vlm_unclear_or_invalid_rate": float(np.mean([v == 0 for v in votes])),
                "vlm_correct_rate": float(np.mean([v == g for v, g in zip(votes, gt)])),
                "vlm_agrees_with_h0_rate": float(np.mean([v == np.sign(r[f"{name}_f_h0"]) for v, r in zip(votes, sel)])),
                "vlm_agrees_with_da_rate": float(np.mean([v == np.sign(r[f"{name}_relation"]) for v, r in zip(votes, sel)])),
                "v0_sources": dict(Counter(r[f"{name}_v0_source"] for r in sel)),
                **{f"{m}_abs_f_mean": float(np.mean(np.abs(np.array([r[f"{name}_f_{m}"] for r in sel]) - fg)))
                   for m in MODELS},
            }
        return out

    test_ids = {r["position"] for r in json.loads((args.w66_dir / "rows_test.json").read_text())}
    scopes = {
        "all_heldout": rows,
        "test_all": [r for r in rows if r["position"] in test_ids],
        "validation_all": [r for r in rows if r["position"] not in test_ids],
        "validation_clean_dancing_hug": [r for r in rows if r["position"] not in test_ids and any(c in r["sequence_id"] for c in CLEAN)],
        "validation_crosscountry_only": [r for r in rows if r["position"] not in test_ids and "crosscountry" in r["sequence_id"]],
        "all_other_heldout": [r for r in rows if not any(c in r["sequence_id"] for c in CLEAN) and "crosscountry" not in r["sequence_id"]],
        "primary_disagreement_frames": [r for r in rows if r["primary"]],
    }
    ordering = {}
    for name in SEGMENT_NAMES:
        disagree_rows = [r for r in rows if r["disagree"][name]]
        ordering[name] = {"segment_disagreement_rows": ordering_metrics(disagree_rows, name),
                          "all_rows": ordering_metrics(rows, name),
                          "clean_disagreement_rows": ordering_metrics(
                              [r for r in disagree_rows if any(c in r["sequence_id"] for c in CLEAN)], name)}

    # Image-grounding control on the primary population.
    grounding = {"pairs": 0}
    real_states = defaultdict(Counter)
    shuffled_states = defaultdict(Counter)
    changes = Counter()
    pairs = 0
    shuffled_valid = []
    for sample_id, shuffled in responses["shuffled"].items():
        real = responses["real"][sample_id]
        pairs += 1
        shuffled_valid.append(shuffled["parse_status"] == "valid")
        for f in FIELDS:
            rs = real["states"][f] if real["states"] else "UNKNOWN"
            ss = shuffled["states"][f] if shuffled["states"] else "UNKNOWN"
            real_states[f][rs] += 1
            shuffled_states[f][ss] += 1
            changes[f] += rs != ss
    grounding = {
        "pairs": pairs,
        "real_valid_rate": float(np.mean([responses["real"][s]["parse_status"] == "valid" for s in responses["shuffled"]])) if pairs else None,
        "shuffled_valid_rate": float(np.mean(shuffled_valid)) if pairs else None,
        "change_rate": {f: changes[f] / pairs for f in FIELDS} if pairs else None,
        "real_distribution": {f: dict(real_states[f]) for f in FIELDS},
        "shuffled_distribution": {f: dict(shuffled_states[f]) for f in FIELDS},
        "real_entropy_bits": {f: _entropy(real_states[f]) for f in FIELDS},
        "shuffled_entropy_bits": {f: _entropy(shuffled_states[f]) for f in FIELDS},
        "donor_rule": "sorted by (sequence, frame); donor = half-way rotation, next row from a different sequence; donor's own overlay",
    }

    # Selector null controls (primary disagreement population, where shuffled votes exist).
    donor_of = population["shuffled_donor_sample_id"]
    identity_matches = sum(
        (responses["shuffled"][sid]["states"] == responses["real"][donor]["states"]) for sid, donor in donor_of.items()
        if sid in responses["shuffled"])
    null_controls = {"shuffled_equals_donor_real_rate": identity_matches / max(1, len(donor_of))}
    primary_rows = [r for r in rows if r["primary"]]
    for name in SEGMENT_NAMES:
        field = SEGMENT_FIELD[name]
        conflict = [r for r in primary_rows if np.sign(r[f"{name}_f_h0"]) != np.sign(r[f"{name}_f_d1"])
                    and np.sign(r[f"{name}_f_h0"]) != 0 and np.sign(r[f"{name}_f_d1"]) != 0]
        fg = np.array([r[f"{name}_f_gt"] for r in conflict])
        h0e = np.abs(np.array([r[f"{name}_f_h0"] for r in conflict]) - fg)
        d1e = np.abs(np.array([r[f"{name}_f_d1"] for r in conflict]) - fg)
        v0e = np.abs(np.array([r[f"{name}_f_v0"] for r in conflict]) - fg)
        shuffled_v0 = []
        for r in conflict:
            st = responses["shuffled"][r["sample_id"]]["states"]
            f, _ = v0_select(r[f"{name}_f_h0"], r[f"{name}_f_d1"], st[field] if st else "UNKNOWN")
            shuffled_v0.append(f)
        sve = np.abs(np.array(shuffled_v0) - fg)
        q = float(np.mean([r[f"{name}_v0_source"] == "d1_vlm_agrees" for r in conflict])) if conflict else None
        oracle = np.minimum(h0e, d1e)
        null_controls[name] = {
            "h0_d1_conflict_rows": len(conflict), "v0_chooses_d1_share": q,
            "h0_abs_f_mean": float(h0e.mean()), "d1_abs_f_mean": float(d1e.mean()),
            "v0_abs_f_mean": float(v0e.mean()), "v0_with_shuffled_votes_abs_f_mean": float(sve.mean()),
            "same_rate_random_mixture_abs_f_mean": float((1 - q) * h0e.mean() + q * d1e.mean()),
            "oracle_branch_abs_f_mean": float(oracle.mean()),
            "d1_correct_branch_share": float(np.mean(d1e < h0e)),
            "v0_picks_d1_when_d1_better": float(np.mean([r[f"{name}_v0_source"] == "d1_vlm_agrees"
                                                          for r, a, b in zip(conflict, h0e, d1e) if b < a])),
            "v0_picks_d1_when_h0_better": float(np.mean([r[f"{name}_v0_source"] == "d1_vlm_agrees"
                                                          for r, a, b in zip(conflict, h0e, d1e) if a <= b])),
        }

    per_sequence = {}
    for name in SEGMENT_NAMES:
        better_h0 = better_d1 = 0
        seqs = defaultdict(list)
        for r in rows:
            seqs[r["sequence_id"]].append(r)
        for sid, group in seqs.items():
            fg = np.array([r[f"{name}_f_gt"] for r in group])
            e = {m: np.abs(np.array([r[f"{name}_f_{m}"] for r in group]) - fg).mean() for m in ("h0", "d1", "v0")}
            better_h0 += e["v0"] < e["h0"]
            better_d1 += e["v0"] < e["d1"]
        per_sequence[name] = {"sequences": len(seqs), "v0_better_than_h0": int(better_h0),
                              "v0_better_than_d1": int(better_d1)}

    owner_src = json.loads((args.w66_dir / "report.json").read_text())["owner_cases"]
    by_key = {(r["sequence_id"], r["frame_index"]): r for r in rows}
    owner = {}
    for label, c in owner_src.items():
        r = by_key.get((c["sequence_id"], c["frame_index"]))
        if r is None:
            owner[label] = {"sequence_id": c["sequence_id"], "frame_index": c["frame_index"], "excluded": "not eligible"}
            continue
        owner[label] = {"sequence_id": c["sequence_id"], "frame_index": c["frame_index"], "overlaps": c["overlaps"],
                        "primary": r["primary"], "vlm": r["vlm"], **{
            name: {"gt_f": r[f"{name}_f_gt"], "h0_f": r[f"{name}_f_h0"], "da_relation": r[f"{name}_relation"],
                   "d1_f": r[f"{name}_f_d1"], "r1_f": r[f"{name}_f_r1"], "v0_f": r[f"{name}_f_v0"],
                   "v0_source": r[f"{name}_v0_source"],
                   "full_h0_d1_v0": [r[f"{name}_h0_raw_plane"], r[f"{name}_d1_raw_plane"], r[f"{name}_v0_raw_plane"]]}
            for name in SEGMENT_NAMES}}

    report = {
        "schema": "animcv_vlm_depth_advisor_analysis_v1",
        "model_manifest": json.loads((args.vlm_dir / "model_manifest.json").read_text()),
        "parse_status": {f"{k[0]}:{k[1]}": v for k, v in parse_status.items()}, "fence_normalized": fences,
        "population": {"all_rows": len(rows), "primary_disagreement_frames": population["primary_count"],
                       "segment_disagreements": {n: sum(r["disagree"][n] for r in rows) for n in SEGMENT_NAMES}},
        "ordering": ordering, "grounding": grounding,
        "complementarity": {scope: {n: complementarity(scopes[scope], n) for n in SEGMENT_NAMES}
                            for scope in ("all_heldout", "primary_disagreement_frames")},
        "continuous": {k: continuous(v) for k, v in scopes.items()},
        "per_sequence": per_sequence, "owner_cases": owner, "selector_null_controls": null_controls,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    path = args.out_dir / "analysis.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps({"sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "population": report["population"],
                      "parse_status": report["parse_status"]}, indent=1))


if __name__ == "__main__":
    main()
