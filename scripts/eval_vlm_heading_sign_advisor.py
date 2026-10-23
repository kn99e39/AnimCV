#!/usr/bin/env python3
"""EVALUATION (Worklog 92): GT-free heading with the frozen-VLM torso up/down sign advisor.

Reads the advisor run (``population.json``, ``responses.jsonl``) and recomputes, on identical frames and detector
observations:
  S0          W91 B: S0 + unchanged ``current_root_orientation`` (no advisor)
  W91_C       W91 reliability policy (S0 yaw on KNOWN rows only)
  S0_VLM      S0 with ``motion.vlm_heading_sign_advisor.decide`` applied on invoked rows (ABSTAIN keeps S0)
  ORACLE_SIGN EVALUATION-ONLY upper bound: the same decision with the GT torso up/down label instead of the VLM
GT (heading, torso up/down label n_z = (b x s)_z with b = thorax - pelvis, s = right - left shoulder) is used only here.
Also reports answer distributions, option-order consistency, KEEP / MIRROR / ABSTAIN, corrected and introduced sign
errors, the shuffled-image control, head-turn suspects (rows the rejected W91A nose rule would flip) and latency.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_root_motion_contact_h0_replay import _matched_oracle_sequence  # noqa: E402
from run_runtime_heading_3dhp import shim  # noqa: E402
from run_runtime_heading_eval import circ, evaluate  # noqa: E402

from framepose.bank import load_bank  # noqa: E402
from framepose.bedlam_supervision import ExternalBank  # noqa: E402
from motion.runtime_heading import runtime_heading  # noqa: E402
from motion.runtime_heading_side_check import nose_offset, side_view_check  # noqa: E402
from motion.vlm_heading_sign_advisor import decide, parse_answer  # noqa: E402
from pose.framepose_bridge import assemble_h0, build_h0_lifted_sequence, sequence_frame_positions, sequence_ids_in_split  # noqa: E402
from pose.root_orientation_diagnostic import observe_yaw  # noqa: E402

NAMES = ("pelvis", "left_hip", "left_knee", "left_ankle", "right_hip", "right_knee", "right_ankle", "spine", "thorax", "neck",
         "head", "left_shoulder", "left_elbow", "left_wrist", "right_shoulder", "right_elbow", "right_wrist")


def gt_label(t3):
    b, s = t3[8] - t3[0], t3[14] - t3[11]
    return "BACK" if np.cross(b, s)[2] > 0 else "STOMACH"


def inclination(t3):
    v = t3[8] - t3[0]
    return math.degrees(math.acos(max(-1.0, min(1.0, v[2] / max(np.linalg.norm(v), 1e-9)))))


def collect(view, preds, seqs, gt_fn, arrays, sizes, times, adv, answers, shuffled):
    """All W91 rows; S0_VLM / ORACLE_SIGN differ only on invoked rows (those in ``adv``)."""
    rows = {k: [] for k in ("S0", "W91_C", "S0_VLM", "ORACLE_SIGN")}
    per = []
    for sid in seqs:
        pos = sequence_frame_positions(view, sid)
        heads = runtime_heading(build_h0_lifted_sequence(view, preds, sid))
        gt, t3s = gt_fn(sid, pos)
        for i, p in enumerate(pos):
            g, h = gt[i], heads[i]
            if g is None:
                continue
            sid_key = view.samples[p].sample_id
            y = h.yaw_radians
            y_vlm, y_orc = y, y
            if sid_key in adv:
                a = adv[sid_key]
                ans = answers.get(sid_key, {})
                y_vlm, act, why = decide(y, ans, a["axis_x"])
                lab = gt_label(t3s[i])
                y_orc, act_o, _ = decide(y, {"back_first": lab, "stomach_first": lab}, a["axis_x"])
                off, conf = nose_offset(arrays["input_2d"][p], arrays["input_valid"][p], sizes[p])
                per.append({"sequence_id": sid, "frame": int(view.samples[p].frame_index), "action": act, "reason": why,
                            "oracle_action": act_o, "answers": ans, "gt_label": lab, "gt_inclination": inclination(t3s[i]),
                            "shuffled": shuffled.get(sid_key), "err_s0": abs(circ(y, g)) if y is not None else None,
                            "err_vlm": abs(circ(y_vlm, g)) if y_vlm is not None else None,
                            "nose_rule": side_view_check(y, off, conf)[1]})
            base = {"sequence_id": sid, "frame": int(view.samples[p].frame_index), "t": times[p], "gt": g, "conf": 1.0, "reliability": None}
            rows["S0"].append({**base, "known": y is not None, "yaw": y})
            rows["W91_C"].append({**base, "known": h.status == "KNOWN", "yaw": y if h.status == "KNOWN" else None})
            rows["S0_VLM"].append({**base, "known": y_vlm is not None, "yaw": y_vlm})
            rows["ORACLE_SIGN"].append({**base, "known": y_orc is not None, "yaw": y_orc})
    return rows, per


def advisor_stats(per):
    c = Counter(r["action"] for r in per)
    reasons = Counter(r["reason"] for r in per)
    cons = [r for r in per if r["answers"].get("back_first") and r["answers"].get("stomach_first")]
    agree = [r for r in cons if r["answers"]["back_first"] == r["answers"]["stomach_first"]]
    lying = [r for r in agree if r["gt_inclination"] >= 60 and r["answers"]["back_first"] in ("BACK", "STOMACH")]
    lab_acc = float(np.mean([r["answers"]["back_first"] == r["gt_label"] for r in lying])) if lying else None
    shuf = [r for r in per if r["shuffled"] in ("BACK", "STOMACH") and r["gt_inclination"] >= 60]
    real_first = [r for r in per if r["answers"].get("back_first") in ("BACK", "STOMACH") and r["gt_inclination"] >= 60]
    corrected = sum(1 for r in per if r["action"] == "MIRROR" and r["err_s0"] - r["err_vlm"] > 90)
    introduced = sum(1 for r in per if r["action"] == "MIRROR" and r["err_vlm"] - r["err_s0"] > 90)
    wrong_s0 = sum(1 for r in per if r["err_s0"] is not None and r["err_s0"] > 90)
    correct_s0 = [r for r in per if r["err_s0"] is not None and r["err_s0"] <= 45]
    head_turn = [r for r in per if r["nose_rule"] == "flipped"]
    return {"invoked": len(per), "actions": dict(c), "reasons": dict(reasons),
            "answer_distribution": {k: dict(Counter(r["answers"].get(k, "MISSING") for r in per)) for k in ("back_first", "stomach_first")},
            "shuffled_answer_distribution": dict(Counter(r["shuffled"] for r in per)),
            "order_consistency": len(agree) / max(len(cons), 1),
            "label_accuracy_on_gt_lying_consistent": lab_acc, "label_rows": len(lying),
            "label_accuracy_back_first_real_on_gt_lying": float(np.mean([r["answers"]["back_first"] == r["gt_label"] for r in real_first])) if real_first else None,
            "label_accuracy_shuffled_on_gt_lying": float(np.mean([r["shuffled"] == r["gt_label"] for r in shuf])) if shuf else None,
            "gt_label_distribution_lying": dict(Counter(r["gt_label"] for r in per if r["gt_inclination"] >= 60)),
            "s0_wrong_branch_rows_gt90": wrong_s0, "corrected_gt90": corrected, "introduced_gt90": introduced,
            "s0_correct_rows_le45": len(correct_s0), "s0_correct_rows_mirrored": sum(1 for r in correct_s0 if r["action"] == "MIRROR"),
            "oracle_actions": dict(Counter(r["oracle_action"] for r in per)),
            "head_turn_suspects": {"rows": len(head_turn), "actions": dict(Counter(r["action"] for r in head_turn)),
                                   "introduced_gt90": sum(1 for r in head_turn if r["action"] == "MIRROR" and r["err_vlm"] - r["err_s0"] > 90)}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("advisor_dir", "bank", "s0_dir", "raw_3dpw", "hp_bank", "hp_s0", "out"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    args = parser.parse_args()
    pop = json.loads((args.advisor_dir / "population.json").read_text())
    adv = {r["sample_id"]: r for r in pop["rows"]}
    answers, shuffled, lat = defaultdict(dict), {}, []
    for line in (args.advisor_dir / "responses.jsonl").read_text().splitlines():
        it = json.loads(line)
        a = parse_answer(it["raw_text"])
        lat.append(it["seconds_per_image"])
        if it["kind"] == "real":
            answers[it["sample_id"]][it["prompt"]] = a
        else:
            shuffled[it["sample_id"]] = a
    report = {"schema": "worklog92_vlm_heading_sign_advisor", "latency_seconds_per_image": {"mean": float(np.mean(lat)), "p95": float(np.percentile(lat, 95)), "images": len(lat)},
              "results": {}}
    bank = load_bank(args.bank)
    s0, _ = assemble_h0(bank, {s: args.s0_dir / f"prediction_{s}.npy" for s in ("validation", "test")})
    sizes = [s.image_size for s in bank.samples]
    times = [s.frame_index / 30.0 for s in bank.samples]

    def g3(sid, pos):
        orc = _matched_oracle_sequence(sid, args.raw_3dpw, [bank.samples[p].frame_index for p in pos])
        return [o.yaw_radians for o in observe_yaw(orc)], [np.array([f.points[n].position for n in NAMES]) for f in orc.frames]
    for split in ("validation", "test"):
        rows, per = collect(bank, s0, sequence_ids_in_split(bank, split), g3, bank.arrays, sizes, times, adv, answers, shuffled)
        report["results"][f"3dpw_{split}"] = {"providers": {k: evaluate(v) for k, v in rows.items()}, "advisor": advisor_stats(per)}
    mx = ExternalBank.load(args.hp_bank)
    hp = np.load(args.hp_s0)
    rv = shim(mx, mx.arrays["input_2d"], mx.arrays["input_valid"])
    for s, ext in zip(rv.samples, mx.samples):
        s.sample_id = ext.sample_id
    g2 = np.concatenate([np.zeros_like(mx.arrays["input_2d"][..., :2]), np.ones_like(mx.arrays["input_2d"][..., 2:])], axis=-1)
    gv = shim(mx, g2, mx.arrays["target_valid"])

    def gh(sid, pos):
        return [o.yaw_radians for o in observe_yaw(build_h0_lifted_sequence(gv, mx.arrays["target_3d"], sid))], [mx.arrays["target_3d"][p] for p in pos]
    hsizes = [s.image_size for s in mx.samples]
    htimes = [s.timestamp for s in rv.samples]
    for gname, keep in (("3dhp_all", None), ("3dhp_TS2", ("TS2",)), ("3dhp_studio_TS1_TS4", ("TS1", "TS2", "TS3", "TS4")), ("3dhp_outdoor_TS5_TS6", ("TS5", "TS6"))):
        seqs = [s for s in sorted({x.sequence_id for x in mx.samples}) if keep is None or s.split(":")[1] in keep]
        rows, per = collect(rv, hp, seqs, gh, mx.arrays, hsizes, htimes, adv, answers, shuffled)
        report["results"][gname] = {"providers": {k: evaluate(v) for k, v in rows.items()}, "advisor": advisor_stats(per)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, default=str))
    print("latency", report["latency_seconds_per_image"])
    for g, x in report["results"].items():
        a = x["advisor"]
        print(f"== {g}: invoked {a['invoked']} actions {a['actions']} order-consistency {a['order_consistency']:.3f} label-acc(lying) {a['label_accuracy_on_gt_lying_consistent']} "
              f"(n={a['label_rows']}) real-first {a['label_accuracy_back_first_real_on_gt_lying']} shuffled {a['label_accuracy_shuffled_on_gt_lying']} | "
              f"corrected {a['corrected_gt90']} introduced {a['introduced_gt90']} of S0-wrong {a['s0_wrong_branch_rows_gt90']} | head-turn {a['head_turn_suspects']}")
        print("   answers", a["answer_distribution"], "shuffled", a["shuffled_answer_distribution"], "gt-lying labels", a["gt_label_distribution_lying"])
        for k, m in x["providers"].items():
            e, t = m["error"], m["temporal"]
            print(f"   {k:11s} cov {m['coverage']:.3f} mean {e['mean']:.2f} p95 {e['p95']:.1f} >90 {e['gt90']:.4f} >135 {e['front_back_gt135']:.4f} flips {t['sudden_flips']} wrong>=1s {t['wrong_runs_ge_1s']} longest {t['longest_wrong_run_s']:.1f}s")


if __name__ == "__main__":
    main()
