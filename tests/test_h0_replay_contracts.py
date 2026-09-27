"""Cross-cutting invariants for the H0 replay batch (docs/52, section 12).

These guard properties the replay script depends on but that live in
UNCHANGED docs/51 modules, so a future edit to those modules can't silently
break the "same fixed rule, just refit on different data" premise T0/T1
depend on, or the CONTACT/MOVING/UNKNOWN contract the whole batch reuses.
"""

import inspect

import numpy as np

from framepose.contract import FrameBank, FrameSample, JOINT_NAMES, Modality
from pose.contact import ContactState, fit_contact_thresholds
from pose.framepose_bridge import build_h0_lifted_sequence, sequence_ids_in_split

JOINT_COUNT = len(JOINT_NAMES)
_MODALITY = Modality(has_2d=True, has_3d=True, has_rgb=False, has_camera=False)


def test_contact_state_semantics_are_exactly_contact_moving_unknown():
    assert {state.value for state in ContactState} == {"contact", "moving", "unknown"}


def test_fit_contact_thresholds_percentile_rule_defaults_are_unchanged():
    """T0 vs T1 must be 'same rule, different data' -- the rule is these defaults."""
    signature = inspect.signature(fit_contact_thresholds)
    assert signature.parameters["contact_percentile"].default == 15.0
    assert signature.parameters["moving_percentile"].default == 60.0
    assert signature.parameters["height_percentile"].default == 40.0


def _bank(count: int, split: str) -> FrameBank:
    samples = [
        FrameSample(
            sample_id=f"3dpw:seq:actor0#{i:06d}", source="3dpw", sequence_id="3dpw:seq:actor0",
            frame_index=i, split=split, image_size=(256, 256), modality=_MODALITY, fps=30.0,
        )
        for i in range(count)
    ]
    arrays = {
        "input_2d": np.zeros((count, JOINT_COUNT, 3), dtype=np.float32),
        "input_valid": np.ones((count, JOINT_COUNT), dtype=bool),
        "target_3d": np.zeros((count, JOINT_COUNT, 3), dtype=np.float32),
        "target_valid": np.zeros((count, JOINT_COUNT), dtype=bool),
    }
    return FrameBank(samples, arrays)


def test_sequence_ids_in_split_never_returns_a_validation_id_for_train():
    train_bank = _bank(3, "train")
    assert sequence_ids_in_split(train_bank, "train") == ["3dpw:seq:actor0"]
    assert sequence_ids_in_split(train_bank, "validation") == []
    assert sequence_ids_in_split(train_bank, "test") == []


def test_build_h0_lifted_sequence_is_deterministic_on_replay():
    bank = _bank(4, "validation")
    h0 = np.arange(4 * JOINT_COUNT * 3, dtype=np.float32).reshape(4, JOINT_COUNT, 3) * 0.001
    first = build_h0_lifted_sequence(bank, h0, "3dpw:seq:actor0")
    second = build_h0_lifted_sequence(bank, h0, "3dpw:seq:actor0")
    assert first.to_dict() == second.to_dict()
