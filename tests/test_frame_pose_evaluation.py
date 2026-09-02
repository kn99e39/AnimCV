import numpy as np
import pytest

from framepose.bank import BankRequest, build_bank
from framepose.contract import BILATERAL_DEPTH_NORMALIZATION, FORWARD_DEPTH_AXIS, JOINT_INDEX
from framepose.evaluate import STABLE_FORWARD_DEPTH_M, aggregate, compare, evaluate_predictions
from framepose_fixtures import prepared_dataset


SEQUENCES = {"train": ["3dpw:a:actor0"], "validation": ["3dpw:b:actor0"], "test": ["3dpw:c:actor0"]}


@pytest.fixture()
def bank(tmp_path):
    requests = [BankRequest("3DPW", split, prepared_dataset(tmp_path / f"{split}.json",
                                                            split=split, sequences=names))
                for split, names in SEQUENCES.items()]
    built, _ = build_bank(requests, require_rgb=False)
    return built


def test_perfect_prediction_scores_zero_on_every_metric(bank):
    positions = bank.indices("test")
    report = evaluate_predictions(bank, positions, bank.arrays["target_3d"][positions], candidate="oracle")
    assert report["frame_count"] == len(positions)
    assert report["aggregate"]["mpjpe_mm"]["mean"] == pytest.approx(0.0, abs=1e-4)
    assert report["aggregate"]["root_yaw_error_degrees"]["mean"] == pytest.approx(0.0, abs=1e-4)
    assert report["aggregate"]["shoulder_forward_depth_sign_disagreement_rate"] == 0.0
    assert len(report["frames"]) == len(positions)
    assert all(frame["mpjpe_mm"] == pytest.approx(0.0, abs=1e-4) for frame in report["frames"])


def test_every_frame_is_individually_addressable(bank):
    positions = bank.indices("test")
    prediction = bank.arrays["target_3d"][positions].copy()
    prediction[4, JOINT_INDEX["left_wrist"]] += (0.1, 0.0, 0.0)
    report = evaluate_predictions(bank, positions, prediction, candidate="one_bad_frame")
    frames = report["frames"]
    assert frames[4]["mpjpe_mm"] > 0
    assert all(frames[index]["mpjpe_mm"] == pytest.approx(0.0, abs=1e-4)
               for index in range(len(frames)) if index != 4)
    assert report["worst_frames"][0]["sample_id"] == frames[4]["sample_id"]
    assert report["per_joint_mean_error_mm"]["left_wrist"] > 0
    assert report["per_joint_mean_error_mm"]["right_wrist"] == pytest.approx(0.0, abs=1e-4)
    # Sequence identity is retained even though the unit of evaluation is a frame.
    assert set(report["per_sequence"]) == set(SEQUENCES["test"])


def test_masked_joints_do_not_contribute_to_frame_metrics(bank):
    positions = bank.indices("test")
    prediction = bank.arrays["target_3d"][positions].copy()
    invalid = ~bank.arrays["target_valid"][positions]
    assert invalid.any(), "fixture must contain at least one masked joint"
    prediction[invalid] += 5.0
    report = evaluate_predictions(bank, positions, prediction, candidate="masked")
    assert report["aggregate"]["mpjpe_mm"]["mean"] == pytest.approx(0.0, abs=1e-4)


def test_forward_depth_sign_disagreement_uses_the_documented_quantity(bank):
    positions = bank.indices("test")[:1]
    target = bank.arrays["target_3d"][positions].copy().astype(np.float64)
    left, right = JOINT_INDEX["left_shoulder"], JOINT_INDEX["right_shoulder"]
    target[0, right, FORWARD_DEPTH_AXIS] = 0.05
    target[0, left, FORWARD_DEPTH_AXIS] = -0.05
    prediction = target.copy()
    prediction[0, right, FORWARD_DEPTH_AXIS] = -0.05
    prediction[0, left, FORWARD_DEPTH_AXIS] = 0.05
    bank.arrays["target_3d"][positions] = target.astype(np.float32)
    report = evaluate_predictions(bank, positions, prediction, candidate="flipped")
    frame = report["frames"][0]
    assert frame["shoulder_forward_depth_target_m"] == pytest.approx(0.1 * BILATERAL_DEPTH_NORMALIZATION)
    assert frame["shoulder_forward_depth_sign_disagreement"] == 1
    assert frame["shoulder_forward_depth_sign_disagreement_stable"] == 1
    assert frame["shoulder_forward_depth_residual_mm"] == pytest.approx(
        -0.2 * BILATERAL_DEPTH_NORMALIZATION * 1000.0)


def test_noise_floor_sign_flips_are_reported_separately(bank):
    positions = bank.indices("test")[:1]
    target = bank.arrays["target_3d"][positions].copy().astype(np.float64)
    left, right = JOINT_INDEX["left_shoulder"], JOINT_INDEX["right_shoulder"]
    tiny = STABLE_FORWARD_DEPTH_M / 10.0
    target[0, right, FORWARD_DEPTH_AXIS] = tiny
    target[0, left, FORWARD_DEPTH_AXIS] = 0.0
    prediction = target.copy()
    prediction[0, right, FORWARD_DEPTH_AXIS] = -tiny
    bank.arrays["target_3d"][positions] = target.astype(np.float32)
    frame = evaluate_predictions(bank, positions, prediction, candidate="noise")["frames"][0]
    assert frame["shoulder_forward_depth_sign_disagreement"] == 1
    assert frame["shoulder_forward_depth_sign_disagreement_stable"] is None


def test_stratified_and_per_source_aggregation_partitions_the_frames(bank):
    positions = bank.indices("test")
    report = evaluate_predictions(bank, positions, bank.arrays["target_3d"][positions], candidate="oracle")
    assert sum(value["frame_count"] for value in report["per_source"].values()) == len(positions)
    for name, buckets in report["per_stratum"].items():
        assert sum(value["frame_count"] for value in buckets.values()) == len(positions), name


def test_aggregate_reports_tail_statistics():
    frames = [{"mpjpe_mm": float(value)} for value in range(1, 101)]
    summary = aggregate(frames)
    assert summary["mpjpe_mm"]["median"] == pytest.approx(50.5)
    assert summary["mpjpe_mm"]["p95"] == pytest.approx(95.05, abs=0.1)
    assert summary["pa_mpjpe_mm"] is None


def test_compare_identifies_which_exact_frames_moved(bank):
    positions = bank.indices("test")
    truth = bank.arrays["target_3d"][positions]
    worse = truth.copy()
    worse[2] += 0.05
    better = truth.copy()
    better[5] += 0.05
    baseline = evaluate_predictions(bank, positions, worse, candidate="baseline")
    candidate = evaluate_predictions(bank, positions, better, candidate="candidate")
    delta = compare(baseline, candidate)
    assert delta["compared_frame_count"] == len(positions)
    assert delta["improved_frame_count"] == 1
    assert delta["regressed_frame_count"] == 1
    assert delta["most_improved"][0]["sample_id"] == baseline["frames"][2]["sample_id"]
    assert delta["most_regressed"][0]["sample_id"] == baseline["frames"][5]["sample_id"]
    assert set(delta["delta_by_stratum"]) >= {"facing", "forward_depth", "visibility"}
