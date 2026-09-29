import json
import math

import numpy as np
import pytest

from framepose.contract import FrameBank, FrameSample, JOINT_INDEX, JOINT_NAMES, Modality
from framepose.observations import ObservationProvenance
from motion.animation_semantics import (
    CONTACT_RULE_VERSION, AnimationSemantics, ContactCalibration, RootOrientation, SemanticFrame,
    Unavailable, check_motion_graph_alignment, load_animation_semantics, motion_graph_reference,
    ownership_table, save_animation_semantics,
)
from motion.animation_semantics_bridge import build_animation_semantics, calibrate_contact_from_h0_train
from motion.motion_builder import MotionGraphBuilder
from motion.motion_graph import MotionGraph
from pose.contact import ContactState, ContactThresholds, classify_foot_contact
from pose.framepose_bridge import H0Identity, build_h0_lifted_sequence
from pose.motion_ownership import OWNERSHIP_MODEL
from pose.pose_lifter import LiftedPoseSequence
from pose.pose_types import PoseFrame, PoseLandmark, PoseSequence
from pose.root_motion import estimate_root_motion

JOINTS = len(JOINT_NAMES)
FRAMES = 60
_MODALITY = Modality(has_2d=True, has_3d=True, has_rgb=False, has_camera=False)
_PROVENANCE = ObservationProvenance("dataset_detector", "dataset_shipped_detector_2d",
                                    "benchmark_detector_observation")
TRAIN = ("3dpw:train_a:actor0", "3dpw:train_b:actor0")
VALID = ("3dpw:valid_a:actor0", "3dpw:valid_b:actor0")


def _pose(frame: int, phase: float) -> np.ndarray:
    """Root-relative pose: slow body turn, left/right ankles alternating stance/swing."""
    pose = np.zeros((JOINTS, 3), dtype=np.float32)
    yaw = 0.01 * frame + phase
    axis = np.array([math.cos(yaw), math.sin(yaw), 0.0])
    for name, half_width, height in (("hip", 0.1, 0.0), ("shoulder", 0.18, 0.5)):
        pose[JOINT_INDEX[f"left_{name}"]] = -half_width * axis + [0, 0, height]
        pose[JOINT_INDEX[f"right_{name}"]] = half_width * axis + [0, 0, height]
    for side, offset in (("left", 0), ("right", 10)):
        swing = ((frame + offset) // 10) % 2 == 1
        step = ((frame + offset) % 10) * 0.05 if swing else 0.0
        pose[JOINT_INDEX[f"{side}_ankle"]] = [0.1 if side == "right" else -0.1, step, -0.9]
    pose[JOINT_INDEX["head"]] = [0, 0, 0.7]
    return pose


def _bank_and_h0(sequences=TRAIN + VALID):
    samples, h0 = [], []
    for number, sequence_id in enumerate(sequences):
        split = "train" if sequence_id in TRAIN else "validation"
        for frame in range(FRAMES):
            samples.append(FrameSample(
                sample_id=f"{sequence_id}#{frame:06d}", source="3dpw", sequence_id=sequence_id,
                frame_index=frame, split=split, image_size=(1920, 1080), modality=_MODALITY,
                observation=_PROVENANCE, timestamp=frame / 30.0 + 0.5, fps=30.0,
            ))
            h0.append(_pose(frame, phase=0.3 * number))
    count = len(samples)
    input_2d = np.full((count, JOINTS, 3), 0.5, dtype=np.float32)
    input_2d[..., 2] = 0.9
    arrays = {
        "input_2d": input_2d,
        "input_valid": np.ones((count, JOINTS), dtype=bool),
        "target_3d": np.zeros((count, JOINTS, 3), dtype=np.float32),
        "target_valid": np.ones((count, JOINTS), dtype=bool),
    }
    return FrameBank(samples, arrays), np.stack(h0)


def _identity(bank):
    return H0Identity(bank.content_digest(), {}, {"train": "a" * 64, "validation": "b" * 64})


def _build(bank, h0, sequence_id=VALID[0], calibration=None):
    identity = _identity(bank)
    calibration = calibration or calibrate_contact_from_h0_train(bank, h0, identity)
    return build_animation_semantics(bank, h0, identity, sequence_id, calibration)


def _rows(bank, sequence_id):
    return [i for i, s in enumerate(bank.samples) if s.sequence_id == sequence_id]


# ------------------------------------------------------------ leakage/locality --

def test_bridge_never_reads_target_arrays():
    bank, h0 = _bank_and_h0()
    identity = _identity(bank)  # the bank digest covers targets; pin it so only bridge output is compared

    def build():
        calibration = calibrate_contact_from_h0_train(bank, h0, identity)
        return build_animation_semantics(bank, h0, identity, VALID[0], calibration)

    clean = build()
    bank.arrays["target_3d"][:] = 999999.0
    bank.arrays["target_valid"][:] = False
    assert build().to_dict() == clean.to_dict()


def test_semantics_are_sequence_local():
    bank, h0 = _bank_and_h0()
    calibration = calibrate_contact_from_h0_train(bank, h0, _identity(bank))
    before = build_animation_semantics(bank, h0, _identity(bank), VALID[0], calibration)
    other = _rows(bank, VALID[1])
    h0[other] += 5.0
    bank.arrays["input_valid"][other] = False
    after = build_animation_semantics(bank, h0, _identity(bank), VALID[0], calibration)
    assert after.frames == before.frames


def test_frame_index_and_timestamp_are_preserved():
    bank, h0 = _bank_and_h0()
    semantics = _build(bank, h0)
    rows = _rows(bank, VALID[0])
    assert [f.frame_index for f in semantics.frames] == [bank.samples[r].frame_index for r in rows]
    assert [f.timestamp for f in semantics.frames] == [bank.samples[r].timestamp for r in rows]


def test_articulation_is_the_frame_pose_output_unchanged():
    bank, h0 = _bank_and_h0()
    semantics = _build(bank, h0)
    rows = _rows(bank, VALID[0])
    for frame, row in zip(semantics.frames, rows):
        assert np.array_equal(np.asarray(frame.articulation.joint_positions, dtype=np.float32), h0[row])


def test_bridge_replay_is_deterministic():
    bank, h0 = _bank_and_h0()
    assert _build(bank, h0).content_digest() == _build(bank, h0).content_digest()


# ------------------------------------------------------------ root orientation --

def test_root_orientation_is_the_unchanged_historical_estimator_with_provenance():
    bank, h0 = _bank_and_h0()
    semantics = _build(bank, h0)
    historical = estimate_root_motion(build_h0_lifted_sequence(bank, h0, VALID[0]))
    for frame, expected in zip(semantics.frames, historical.frames):
        assert frame.root_orientation.known
        assert frame.root_orientation.yaw_radians == expected.root_yaw_radians
        assert frame.root_orientation.yaw_held == expected.yaw_held
        assert frame.root_orientation.historical_confidence == expected.confidence
    provenance = semantics.provenance.root_orientation
    assert provenance["estimator"] == "pose.root_motion.estimate_root_motion"
    assert provenance["max_yaw_step_degrees"] == 20.0
    assert provenance["smoothing_window"] == 5
    assert provenance["max_yaw_step_status"] == "historical_default_not_proven_optimal"
    assert "historical" in provenance["notes"] and "NOT been shown optimal" in provenance["notes"]
    assert "docs/52" in provenance["notes"]


def test_frames_without_bilateral_torso_evidence_are_unknown_not_invented():
    bank, h0 = _bank_and_h0()
    rows = _rows(bank, VALID[0])
    for name in ("left_shoulder", "right_shoulder", "left_hip", "right_hip"):
        bank.arrays["input_valid"][rows[30], JOINT_INDEX[name]] = False
    semantics = _build(bank, h0)
    assert semantics.frames[30].root_orientation == RootOrientation(known=False)
    lifted = build_h0_lifted_sequence(bank, h0, VALID[0])
    for start, end in ((0, 30), (30 + 1, FRAMES)):
        run = estimate_root_motion(LiftedPoseSequence(frames=lifted.frames[start:end], source_fps=30.0))
        for offset, expected in enumerate(run.frames):
            assert semantics.frames[start + offset].root_orientation.yaw_radians == expected.root_yaw_radians


def test_unknown_orientation_is_distinct_from_zero_yaw():
    known_zero = RootOrientation(known=True, yaw_radians=0.0, historical_confidence=0.9, yaw_held=False)
    unknown = RootOrientation(known=False)
    assert RootOrientation.from_dict(known_zero.to_dict()) == known_zero
    assert RootOrientation.from_dict(unknown.to_dict()) == unknown
    assert known_zero.to_dict()["status"] == "known" and unknown.to_dict()["status"] == "unknown"
    assert unknown.to_dict()["yaw_radians"] is None
    with pytest.raises(ValueError):
        RootOrientation(known=False, yaw_radians=0.0)
    with pytest.raises(ValueError):
        RootOrientation(known=True)


# --------------------------------------------------------------------- contact --

def test_foot_motion_is_exactly_the_docs51_rule_at_the_calibration():
    bank, h0 = _bank_and_h0()
    calibration = calibrate_contact_from_h0_train(bank, h0, _identity(bank))
    semantics = build_animation_semantics(bank, h0, _identity(bank), VALID[0], calibration)
    lifted = build_h0_lifted_sequence(bank, h0, VALID[0])
    for side in ("left", "right"):
        expected = classify_foot_contact(lifted, side, ContactThresholds(**calibration.thresholds[side]))
        assert [getattr(f.foot_motion, side) for f in semantics.frames] == expected
    states = {getattr(f.foot_motion, side) for f in semantics.frames for side in ("left", "right")}
    assert states <= {ContactState.CONTACT, ContactState.MOVING, ContactState.UNKNOWN}
    assert {ContactState.CONTACT, ContactState.MOVING} <= states  # the fixture exercises both
    assert {s.value for s in ContactState} == {"contact", "moving", "unknown"}


def test_contact_calibration_is_train_only_with_provenance():
    bank, h0 = _bank_and_h0()
    calibration = calibrate_contact_from_h0_train(bank, h0, _identity(bank))
    assert calibration.rule_version == CONTACT_RULE_VERSION
    assert calibration.percentiles == {"contact": 15.0, "moving": 60.0, "height": 40.0}
    assert calibration.source["split"] == "train"
    assert calibration.source["h0_train_sha256"] == "a" * 64
    assert calibration.source["sequence_count"] == len(TRAIN)
    assert calibration.source["frame_count"] == len(TRAIN) * FRAMES
    assert calibration.sampling["median_row_stride_frames"] == 1
    assert "NOT proven physical foot-ground contact" in calibration.interpretation
    # Validation/test rows cannot move the thresholds.
    for sequence_id in VALID:
        h0[_rows(bank, sequence_id)] *= 7.0
    assert calibrate_contact_from_h0_train(bank, h0, _identity(bank)).thresholds == calibration.thresholds


def test_contact_calibration_refuses_non_train_source_and_other_rule_versions():
    bank, h0 = _bank_and_h0()
    calibration = calibrate_contact_from_h0_train(bank, h0, _identity(bank))
    payload = calibration.to_dict()
    with pytest.raises(ValueError):
        ContactCalibration.from_dict({**payload, "source": {**payload["source"], "split": "validation"}})
    with pytest.raises(ValueError):
        ContactCalibration.from_dict({**payload, "rule_version": "retuned_v2"})
    assert ContactCalibration.from_dict(payload) == calibration


def test_sampling_mismatch_with_calibration_is_recorded_not_hidden():
    bank, h0 = _bank_and_h0()
    calibration = calibrate_contact_from_h0_train(bank, h0, _identity(bank))
    assert _build(bank, h0, calibration=calibration).provenance.contact["sampling_matches_calibration"] is True
    strided = ContactCalibration(calibration.thresholds, calibration.source,
                                 {**calibration.sampling, "median_row_stride_frames": 2})
    semantics = _build(bank, h0, calibration=strided)
    assert semantics.provenance.contact["sampling_matches_calibration"] is False


# ------------------------------------------------------------ unavailable/zero --

def test_root_translation_and_ground_height_are_unavailable_on_every_frame():
    bank, h0 = _bank_and_h0()
    payload = _build(bank, h0).to_dict()
    for frame in payload["frames"]:
        assert frame["root_translation"] == {"status": "unavailable"}
        assert frame["ground_height"] == {"status": "unavailable"}
    assert set(payload["unavailable_quantities"]) == {"root_translation", "ground_height"}


@pytest.mark.parametrize("value", [[0.0, 0.0, 0.0], {"status": "observed", "value": [0.0, 0.0, 0.0]},
                                   {"status": "unavailable", "value": [0.0, 0.0, 0.0]}, None, 0.0])
def test_loading_refuses_any_numeric_or_zero_root_translation_or_ground_height(value):
    bank, h0 = _bank_and_h0()
    payload = _build(bank, h0).to_dict()
    for quantity in ("root_translation", "ground_height"):
        tampered = json.loads(json.dumps(payload))
        tampered["frames"][3][quantity] = value
        with pytest.raises(ValueError):
            AnimationSemantics.from_dict(tampered)


def test_unavailable_is_only_declared_for_unobservable_quantities():
    with pytest.raises(ValueError):
        Unavailable("root_orientation")


# ----------------------------------------------------------------- reliability --

def test_reliability_is_structural_and_owned_separately():
    bank, h0 = _bank_and_h0()
    baseline = _build(bank, h0)
    rows = _rows(bank, VALID[0])
    bank.arrays["input_valid"][rows[10], JOINT_INDEX["head"]] = False
    bank.arrays["input_2d"][rows[11], JOINT_INDEX["left_wrist"], 0] = 1.5  # out of frame, still valid
    changed = _build(bank, h0)
    head = JOINT_INDEX["head"]
    assert changed.frames[10].reliability.joint_observation_valid[head] is False
    assert changed.frames[10].reliability.joint_in_frame[head] is False
    assert changed.frames[11].reliability.joint_observation_valid[JOINT_INDEX["left_wrist"]] is True
    assert changed.frames[11].reliability.joint_in_frame[JOINT_INDEX["left_wrist"]] is False
    for before, after in zip(baseline.frames, changed.frames):
        assert after.root_orientation == before.root_orientation
        assert after.foot_motion == before.foot_motion
    provenance = changed.provenance.reliability
    assert provenance["learned_reliability"] is None
    assert provenance["regime"] == "benchmark_detector_observation"


def test_ownership_table_mirrors_docs51_and_is_enforced_on_load():
    table = ownership_table()
    owners = [entry["owner"] for entry in table.values()]
    assert len(owners) == len(set(owners)) == len(OWNERSHIP_MODEL)
    assert table["root_translation"]["observability"] == "not_observable"
    assert table["ground_height"]["observability"] == "not_observable"
    bank, h0 = _bank_and_h0()
    payload = _build(bank, h0).to_dict()
    payload["ownership"]["foot_motion"]["owner"] = "reliability"
    with pytest.raises(ValueError):
        AnimationSemantics.from_dict(payload)


# --------------------------------------------------------------- serialization --

def test_serialization_round_trip_is_identity(tmp_path):
    bank, h0 = _bank_and_h0()
    semantics = _build(bank, h0)
    path = tmp_path / "seq.semantics.json"
    save_animation_semantics(semantics, path)
    loaded = load_animation_semantics(path)
    assert loaded == semantics
    assert loaded.content_digest() == semantics.content_digest()
    assert loaded.to_dict() == semantics.to_dict()


def test_semantic_frames_must_be_ordered():
    bank, h0 = _bank_and_h0()
    semantics = _build(bank, h0)
    with pytest.raises(ValueError):
        AnimationSemantics(semantics.sequence_id, semantics.source_fps, semantics.coordinate_frame,
                           semantics.joint_names, tuple(reversed(semantics.frames)), semantics.provenance)


# ------------------------------------------------------ MotionGraph linkage/compat --

def _motion_graph(frame_count: int, fps: float = 30.0, offset: float = 0.5) -> MotionGraph:
    frames = [PoseFrame(frame_index=i, timestamp=i / fps + offset,
                        landmarks={"nose": PoseLandmark(name="nose", x=0.5, y=0.5, confidence=0.9, visible=True)})
              for i in range(frame_count)]
    return MotionGraphBuilder().build(PoseSequence(frames=frames, source_fps=fps))


def test_motion_graph_links_by_reference_without_schema_change():
    bank, h0 = _bank_and_h0()
    semantics = _build(bank, h0)
    graph = _motion_graph(FRAMES)
    legacy_keys = set(graph.to_dict())
    graph.source_metadata["animation_semantics"] = motion_graph_reference(semantics)
    reloaded = MotionGraph.from_dict(json.loads(json.dumps(graph.to_dict())))
    assert set(reloaded.to_dict()) == legacy_keys == {"fps", "source_metadata", "frames", "tracks"}
    assert reloaded.source_metadata["animation_semantics"]["content_digest"] == semantics.content_digest()
    assert check_motion_graph_alignment(reloaded, semantics) == []
    assert check_motion_graph_alignment(_motion_graph(FRAMES, offset=0.0), semantics) != []


def test_historical_motion_graph_without_semantics_still_loads():
    historical = {"fps": 30.0, "source_metadata": {}, "tracks": {},
                  "frames": [{"frame_index": 0, "timestamp": 0.0, "points": {}, "importance": 0.0, "locked": False}]}
    graph = MotionGraph.from_dict(historical)
    assert graph.to_dict() == historical
