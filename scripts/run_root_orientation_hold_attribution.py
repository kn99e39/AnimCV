#!/usr/bin/env python3
"""Is the historical no-release yaw hold helping frozen H0, or making it stale? (docs/55)

Control experiment: same H0, same fused bilateral heading maths, same oracle
matching, same validation rows, same thresholds. The only thing that changes
is which already-existing yaw signal is evaluated:

    H0-CURRENT   observe_yaw(H0 sequence)            (current-frame evidence)
    LEGACY-HOLD  persisted AnimationSemantics yaw    (estimate_root_motion)

Nothing is tuned. Oracle = matched 3DPW GT geometry (docs/52
_matched_oracle_sequence) passed through the same observe_yaw. Regime of the
H0 signals: benchmark_detector_observation.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_root_motion_contact_h0_replay import _matched_oracle_sequence  # noqa: E402

from framepose.bank import load_bank  # noqa: E402
from motion.animation_semantics import load_animation_semantics  # noqa: E402
from pose.framepose_bridge import assemble_h0, build_h0_lifted_sequence  # noqa: E402
from pose.root_orientation_diagnostic import observe_yaw  # noqa: E402
from pose.root_orientation_hold_attribution import (  # noqa: E402
    AGE_BUCKETS, YawRow, age_bucket, align_rows, circular_delta_degrees, attribute_run, distribution, held_runs,
    large_current_changes, run_length_distribution,
)

DOCS54_CASES = {
    "docs54_turning": ("3dpw:outdoors_crosscountry_00:actor0", 202),
    "docs54_known_good_held": ("3dpw:courtyard_dancing_00:actor0", 424),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", required=True, type=Path)
    parser.add_argument("--h0-validation", required=True, type=Path)
    parser.add_argument("--semantics-dir", required=True, type=Path)
    parser.add_argument("--3dpw-raw", dest="threedpw_raw", required=True, type=Path)
    parser.add_argument("--expect-bank-content-digest", default=None)
    parser.add_argument("--expect-h0-validation-sha256", default=None)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()

    bank = load_bank(args.bank)
    h0, identity = assemble_h0(bank, {"validation": args.h0_validation})
    if args.expect_bank_content_digest and identity.bank_content_digest != args.expect_bank_content_digest:
        raise SystemExit(f"bank digest mismatch: {identity.bank_content_digest}")
    if args.expect_h0_validation_sha256 and identity.split_sha256["validation"] != args.expect_h0_validation_sha256:
        raise SystemExit(f"H0 validation sha256 mismatch: {identity.split_sha256['validation']}")
    semantics_report = json.loads((args.semantics_dir / "report.json").read_text())
    if semantics_report["h0_identity"]["split_sha256"]["validation"] != identity.split_sha256["validation"]:
        raise SystemExit("persisted AnimationSemantics were built from a different H0")

    rows_by_sequence: dict[str, list[YawRow]] = {}
    for entry in semantics_report["per_sequence"]:
        sequence_id = entry["sequence_id"]
        semantics = load_animation_semantics(args.semantics_dir / entry["file"])
        if semantics.content_digest() != entry["content_digest"]:
            raise SystemExit(f"{sequence_id}: semantics file digest changed since docs/53")
        h0_sequence = build_h0_lifted_sequence(bank, h0, sequence_id)
        current = [(o.frame_index, o.yaw_radians, o.reliable) for o in observe_yaw(h0_sequence)]
        legacy = [(f.frame_index, f.root_orientation.yaw_radians, bool(f.root_orientation.yaw_held))
                  for f in semantics.frames]
        oracle_sequence = _matched_oracle_sequence(sequence_id, args.threedpw_raw, [row[0] for row in current])
        oracle = ([(o.frame_index, o.yaw_radians, o.reliable) for o in observe_yaw(oracle_sequence)]
                  if oracle_sequence is not None else None)
        rows_by_sequence[sequence_id] = align_rows(sequence_id, current, legacy, oracle)

    report = build_report(rows_by_sequence)
    report["h0_identity"] = {"bank_content_digest": identity.bank_content_digest,
                             "h0_validation_sha256": identity.split_sha256["validation"]}
    report["regime"] = bank.regime()
    report["legacy_source"] = {"semantics_dir": str(args.semantics_dir),
                               "digests": {e["sequence_id"]: e["content_digest"] for e in semantics_report["per_sequence"]}}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    with (args.out_dir / "rows.jsonl").open("w") as handle:
        for rows in rows_by_sequence.values():
            for row in rows:
                handle.write(json.dumps(row.to_dict(), sort_keys=True) + "\n")
    summary = {k: report[k] for k in ("held_accounting", "gap_held_vs_current", "oracle_error", "run_attribution",
                                      "large_current_changes")}
    print(json.dumps(summary, indent=2))
    print(json.dumps(report["owner_cases"], indent=2))


def build_report(rows_by_sequence: dict[str, list[YawRow]]) -> dict[str, Any]:
    all_rows = [row for rows in rows_by_sequence.values() for row in rows]
    held_rows = [r for r in all_rows if r.legacy_held]
    runs = [(sid, start, end) for sid, rows in rows_by_sequence.items() for start, end in held_runs(
        [r.legacy_held for r in rows])]

    def by_age(values_of):
        return {bucket: distribution([values_of(r) for r in held_rows if age_bucket(r.held_age) == bucket])
                for bucket in AGE_BUCKETS}

    def errors(subset):
        return {"H0_CURRENT": distribution([r.current_error for r in subset]),
                "LEGACY_HOLD": distribution([r.legacy_error for r in subset])}

    oracle_rows = [r for r in all_rows if r.oracle_yaw is not None and r.current_yaw is not None
                   and r.legacy_yaw is not None]
    attributions = [attribute_run(rows_by_sequence[sid], start, end) for sid, start, end in runs]
    classes = Counter(a["classification"] for a in attributions)
    frames_by_class = Counter()
    for a in attributions:
        frames_by_class[a["classification"]] += a["length"]
    turned = [a for a in attributions if a.get("oracle_turned_beyond_historical_step")]
    events = [e for rows in rows_by_sequence.values() for e in large_current_changes(rows)]

    def event_summary(subset):
        return {"count": len(subset), "legacy_held": sum(e["legacy_held"] for e in subset),
                "current_error": distribution([e["current_error"] for e in subset]),
                "legacy_error": distribution([e["legacy_error"] for e in subset])}

    return {
        "schema": "animcv_root_orientation_hold_attribution_v1",
        "sequence_count": len(rows_by_sequence),
        "frame_count": len(all_rows),
        "held_accounting": {
            "held_frames": len(held_rows),
            "held_run_length": run_length_distribution([end - start for _, start, end in runs]),
            "held_frames_by_age": {b: sum(1 for r in held_rows if age_bucket(r.held_age) == b) for b in AGE_BUCKETS},
        },
        "gap_held_vs_current": {"all_held": distribution([r.held_current_gap for r in held_rows]),
                                "by_age": by_age(lambda r: r.held_current_gap),
                                "non_held_reference": distribution([r.held_current_gap for r in all_rows
                                                                    if not r.legacy_held])},
        "oracle_error": {
            "rows_with_oracle": len(oracle_rows),
            "all": errors(oracle_rows),
            "non_held": errors([r for r in oracle_rows if not r.legacy_held]),
            "held": errors([r for r in oracle_rows if r.legacy_held]),
            "held_by_age": {b: errors([r for r in oracle_rows if r.legacy_held and age_bucket(r.held_age) == b])
                            for b in AGE_BUCKETS},
            "current_reliable_only": errors([r for r in oracle_rows if r.current_reliable]),
            "oracle_reliable_rows": sum(1 for r in oracle_rows if r.oracle_reliable),
            "oracle_reliable_only": {
                "all": errors([r for r in oracle_rows if r.oracle_reliable]),
                "non_held": errors([r for r in oracle_rows if r.oracle_reliable and not r.legacy_held]),
                "held": errors([r for r in oracle_rows if r.oracle_reliable and r.legacy_held]),
            },
        },
        "run_attribution": {
            "runs": len(attributions), "classes": dict(classes), "held_frames_by_class": dict(frames_by_class),
            "runs_where_oracle_turned_beyond_20": len(turned),
            "classes_where_oracle_turned": dict(Counter(a["classification"] for a in turned)),
            "per_run": attributions,
        },
        "large_current_changes": {
            "definition": "consecutive-row H0-CURRENT change > 20 deg (historical step, reporting only)",
            "oracle_also_changed": event_summary([e for e in events if e["oracle_also_changed"]]),
            "oracle_did_not_change": event_summary([e for e in events if not e["oracle_also_changed"]]),
            "events": events,
        },
        "owner_cases": _owner_cases(rows_by_sequence, runs, attributions),
    }


def _owner_cases(rows_by_sequence, runs, attributions) -> dict[str, Any]:
    cases = {name: {"sequence_id": sid, "frame_index": fi} for name, (sid, fi) in DOCS54_CASES.items()}
    sid, start, end = max(runs, key=lambda r: (r[2] - r[1], r[0]))
    cases["longest_held_run"] = {"sequence_id": sid,
                                 "frame_index": rows_by_sequence[sid][(start + end) // 2].frame_index,
                                 "run_rows": end - start}
    gap_row = max((r for rows in rows_by_sequence.values() for r in rows if r.held_current_gap is not None),
                  key=lambda r: (r.held_current_gap, r.sequence_id, r.frame_index))
    cases["largest_held_current_gap"] = {"sequence_id": gap_row.sequence_id, "frame_index": gap_row.frame_index,
                                         "gap": gap_row.held_current_gap}
    best = None
    for rows in rows_by_sequence.values():
        for previous, row in zip(rows, rows[1:]):
            if row.oracle_yaw is None or previous.oracle_yaw is None:
                continue
            delta = circular_delta_degrees(row.oracle_yaw, previous.oracle_yaw)
            if best is None or delta > best[0]:
                best = (delta, row)
    if best is not None:
        cases["fastest_oracle_turn"] = {"sequence_id": best[1].sequence_id, "frame_index": best[1].frame_index,
                                        "oracle_delta_per_row": best[0],
                                        "genuine_fast_turn": best[0] > 20.0}
    for case in cases.values():
        row = next(r for r in rows_by_sequence[case["sequence_id"]] if r.frame_index == case["frame_index"])
        case.update({"held": row.legacy_held, "held_age": row.held_age, "gap": row.held_current_gap,
                     "current_error": row.current_error, "legacy_error": row.legacy_error})
    return cases


if __name__ == "__main__":
    main()
