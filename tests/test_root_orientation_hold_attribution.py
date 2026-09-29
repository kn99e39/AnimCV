import json
import math
import sys
from pathlib import Path

import pytest

from pose.pose_lifter import LiftedPoseFrame, LiftedPosePoint, LiftedPoseSequence
from pose.root_motion import estimate_root_motion
from pose.root_orientation_diagnostic import observe_yaw
from pose.root_orientation_hold_attribution import (
    YawRow, age_bucket, align_rows, attribute_run, circular_delta_degrees, distribution, held_run_ages,
    held_runs, large_current_changes,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


def _torso_sequence(yaws_degrees):
    frames = []
    for index, yaw in enumerate(yaws_degrees):
        c, s = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
        points = {}
        for name, half, z in (("hip", 0.1, 0.0), ("shoulder", 0.18, 0.5)):
            points[f"left_{name}"] = LiftedPosePoint(f"left_{name}", (-half * c, -half * s, z), 0.9, 0.1)
            points[f"right_{name}"] = LiftedPosePoint(f"right_{name}", (half * c, half * s, z), 0.9, 0.1)
        frames.append(LiftedPoseFrame(index * 3, index / 10.0, points))
    return LiftedPoseSequence(frames=frames, source_fps=30.0)


def test_current_and_legacy_share_the_same_yaw_convention_on_the_same_frame():
    sequence = _torso_sequence([10.0] * 7)
    current = observe_yaw(sequence)
    legacy = estimate_root_motion(sequence)
    for obs, frame in zip(current, legacy.frames):
        assert obs.frame_index == frame.frame_index
        assert circular_delta_degrees(obs.yaw_radians, frame.root_yaw_radians) == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("a,b,expected", [(0, 0, 0), (170, -170, 20), (-179, 179, 2), (0, 180, 180),
                                          (370, 10, 0), (90, -90, 180), (45, 0, 45)])
def test_circular_delta_is_smallest_absolute_angle(a, b, expected):
    assert circular_delta_degrees(math.radians(a), math.radians(b)) == pytest.approx(expected, abs=1e-9)


def test_unwrapped_legacy_yaw_compares_circularly():
    # estimate_root_motion unwraps, so its yaw can leave (-pi, pi].
    assert circular_delta_degrees(math.radians(350.0 + 360.0), math.radians(-10.0)) == pytest.approx(0.0, abs=1e-9)


def test_held_run_age_and_buckets():
    held = [False, True, True, False] + [True] * 22 + [False]
    ages = held_run_ages(held)
    assert ages[:4] == [None, 1, 2, None]
    assert ages[4:26] == list(range(1, 23)) and ages[26] is None
    assert held_runs(held) == [(1, 3), (4, 26)]
    assert [age_bucket(a) for a in (1, 2, 5, 6, 20, 21)] == ["first", "2-5", "2-5", "6-20", "6-20", ">20"]
    with pytest.raises(ValueError):
        age_bucket(0)


def test_held_run_at_sequence_end_is_closed():
    assert held_runs([False, True, True]) == [(1, 3)]


def test_align_rows_joins_by_frame_and_refuses_mismatch():
    current = [(0, 0.0, True), (3, 0.1, False)]
    legacy = [(0, 0.0, False), (3, 0.0, True)]
    oracle = [(0, 0.05, True), (3, 0.2, False)]
    rows = align_rows("s", current, legacy, oracle)
    assert [(r.frame_index, r.held_age, r.current_reliable) for r in rows] == [(0, None, True), (3, 1, False)]
    assert rows[1].held_current_gap == pytest.approx(math.degrees(0.1))
    assert rows[1].legacy_error == pytest.approx(math.degrees(0.2))
    assert [r.oracle_reliable for r in rows] == [True, False]
    with pytest.raises(ValueError):
        align_rows("s", current, [(0, 0.0, False), (6, 0.0, True)], oracle)
    with pytest.raises(ValueError):
        align_rows("s", current, legacy, [(0, 0.0, True), (4, 0.0, True)])
    assert align_rows("s", current, legacy, None)[0].current_error is None


def _row(fi, current, legacy, held, oracle, age=None, sid="s"):
    r = math.radians
    return YawRow(sid, fi, r(current), True, r(legacy), held, age, r(oracle))


def test_run_attribution_uses_accumulated_error_not_one_transition():
    # H0 flips for one row and the hold rejects it: hold better.
    flip = [_row(0, 0, 0, False, 0), _row(1, 170, 0, True, 2, 1), _row(2, 3, 3, False, 3)]
    assert attribute_run(flip, 1, 2)["classification"] == "hold_better_than_current"
    # Body really turns and the hold freezes the old heading: worse from the start.
    turn = [_row(0, 0, 0, False, 0), _row(1, 60, 0, True, 60, 1), _row(2, 90, 0, True, 90, 2)]
    result = attribute_run(turn, 1, 3)
    assert result["classification"] == "hold_worse_from_start"
    assert result["oracle_turned_beyond_historical_step"] is True
    # Hold starts right (rejects a flip) and then goes stale as the body turns.
    stale = [_row(0, 0, 0, False, 0), _row(1, 175, 0, True, 5, 1),
             _row(2, 60, 0, True, 60, 2), _row(3, 120, 0, True, 120, 3)]
    assert attribute_run(stale, 1, 4)["classification"] == "stale_after_reasonable_start"


def test_large_current_changes_record_oracle_context():
    rows = [_row(0, 0, 0, False, 0), _row(1, 100, 0, True, 1, 1), _row(2, 130, 130, False, 130)]
    events = large_current_changes(rows)
    assert [e["frame_index"] for e in events] == [1, 2]
    assert events[0]["oracle_also_changed"] is False and events[0]["legacy_held"] is True
    assert events[1]["oracle_also_changed"] is True


def test_distribution_bins():
    report = distribution([10, 30, 50, 100])
    assert report["count"] == 4 and report["max"] == 100
    assert report["frac_gt_20"] == 0.75 and report["frac_gt_45"] == 0.5 and report["frac_gt_90"] == 0.25
    assert distribution([None])["count"] == 0


def test_report_generation_is_deterministic():
    from run_root_orientation_hold_attribution import build_report
    turning, dancing = "3dpw:outdoors_crosscountry_00:actor0", "3dpw:courtyard_dancing_00:actor0"
    rows = {turning: [_row(199, 0, 0, False, 0, sid=turning), _row(202, 100, 0, True, 90, 1, turning),
                      _row(205, 120, 0, True, 120, 2, turning), _row(208, 121, 121, False, 121, sid=turning)],
            dancing: [_row(421, 0, 0, False, 0, sid=dancing), _row(424, 5, 0, True, 4, 1, dancing)]}
    first = json.dumps(build_report(rows), sort_keys=True)
    assert first == json.dumps(build_report(rows), sort_keys=True)
    report = json.loads(first)
    assert report["held_accounting"]["held_frames"] == 3
    assert report["owner_cases"]["longest_held_run"]["run_rows"] == 2
