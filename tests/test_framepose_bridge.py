import numpy as np
import pytest

from framepose.contract import FrameBank, FrameSample, JOINT_NAMES, Modality
from pose.framepose_bridge import (
    assemble_h0, build_h0_lifted_sequence, sequence_frame_positions, sequence_ids_in_split,
)

JOINT_COUNT = len(JOINT_NAMES)
_MODALITY = Modality(has_2d=True, has_3d=True, has_rgb=False, has_camera=False)


def _sample(sequence_id: str, frame_index: int, split: str) -> FrameSample:
    return FrameSample(
        sample_id=f"{sequence_id}#{frame_index:06d}", source="3dpw", sequence_id=sequence_id,
        frame_index=frame_index, split=split, image_size=(1920, 1080), modality=_MODALITY, fps=30.0,
    )


def _poisoned_bank(count: int = 3, split: str = "validation") -> FrameBank:
    """A bank whose target_3d/target_valid are deliberately WRONG sentinels.

    input_valid is all True; target_valid is the OPPOSITE (all False), so a
    bridge function that accidentally reads target_valid instead of
    input_valid is caught by an inverted flag, not just a missing exception.
    target_3d is filled with an unmistakable sentinel (999999.0) far from any
    real H0/oracle coordinate, so accidental use shows up as an absurd value.
    """
    samples = [_sample("3dpw:poison_seq:actor0", frame_index, split) for frame_index in range(count)]
    arrays = {
        "input_2d": np.stack([
            np.stack([[0.5, 0.5, 0.9] for _ in range(JOINT_COUNT)]) for _ in range(count)
        ]).astype(np.float32),
        "input_valid": np.ones((count, JOINT_COUNT), dtype=bool),
        "target_3d": np.full((count, JOINT_COUNT, 3), 999999.0, dtype=np.float32),
        "target_valid": np.zeros((count, JOINT_COUNT), dtype=bool),
    }
    return FrameBank(samples, arrays)


def test_build_h0_lifted_sequence_never_reflects_target_arrays():
    bank = _poisoned_bank(count=4)
    h0 = np.tile(np.arange(JOINT_COUNT * 3, dtype=np.float32).reshape(JOINT_COUNT, 3) * 0.01, (4, 1, 1))

    sequence = build_h0_lifted_sequence(bank, h0, "3dpw:poison_seq:actor0")

    for frame in sequence.frames:
        for point in frame.points.values():
            # Real H0 values are small (<= ~0.16); the poisoned target sentinel is 999999.0.
            assert abs(point.position[0]) < 100.0
            assert abs(point.position[1]) < 100.0
            assert abs(point.position[2]) < 100.0
            # input_valid is True everywhere; target_valid is False everywhere.
            # Reading target_valid by mistake would flip every point to invalid.
            assert point.observation_valid is True


def test_build_h0_lifted_sequence_orders_by_frame_index():
    bank = _poisoned_bank(count=3)
    h0 = np.zeros((3, JOINT_COUNT, 3), dtype=np.float32)
    sequence = build_h0_lifted_sequence(bank, h0, "3dpw:poison_seq:actor0")
    assert [frame.frame_index for frame in sequence.frames] == [0, 1, 2]


def test_build_h0_lifted_sequence_marks_non_finite_h0_as_invalid():
    bank = _poisoned_bank(count=2)
    h0 = np.zeros((2, JOINT_COUNT, 3), dtype=np.float32)
    h0[1, 0, :] = np.nan  # pelvis at frame 1 is a non-finite H0 prediction
    sequence = build_h0_lifted_sequence(bank, h0, "3dpw:poison_seq:actor0")
    assert sequence.frames[1].points["pelvis"].observation_valid is False
    assert sequence.frames[0].points["pelvis"].observation_valid is True


def test_build_h0_lifted_sequence_rejects_unknown_sequence():
    bank = _poisoned_bank(count=2)
    h0 = np.zeros((2, JOINT_COUNT, 3), dtype=np.float32)
    with pytest.raises(ValueError):
        build_h0_lifted_sequence(bank, h0, "3dpw:does_not_exist:actor0")


def test_sequence_frame_positions_and_ids_in_split():
    bank = _poisoned_bank(count=3, split="train")
    assert sequence_frame_positions(bank, "3dpw:poison_seq:actor0") == [0, 1, 2]
    assert sequence_ids_in_split(bank, "train") == ["3dpw:poison_seq:actor0"]
    assert sequence_ids_in_split(bank, "validation") == []


def test_assemble_h0_leaves_uncovered_positions_as_nan_and_records_identity():
    bank = _poisoned_bank(count=2, split="train")
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "prediction_train.npy"
        np.save(path, np.ones((2, JOINT_COUNT, 3), dtype=np.float32))
        h0, identity = assemble_h0(bank, {"train": path})
    assert h0.shape == (2, JOINT_COUNT, 3)
    assert np.all(h0 == 1.0)
    assert identity.bank_content_digest == bank.content_digest()
    assert "train" in identity.split_sha256


def test_assemble_h0_rejects_wrong_shape():
    bank = _poisoned_bank(count=2, split="train")
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "prediction_train.npy"
        np.save(path, np.ones((3, JOINT_COUNT, 3), dtype=np.float32))  # wrong row count
        with pytest.raises(ValueError):
            assemble_h0(bank, {"train": path})
