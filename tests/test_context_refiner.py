"""Focused contracts for the bounded context-assisted Frame Pose candidate."""

from __future__ import annotations

import numpy as np
import pytest

from framepose.bank import BankRequest, build_bank
from framepose.context_refiner import (
    CONTEXT_DIM,
    JOINT_FEATURES,
    WINDOW_OFFSETS,
    apply_gate,
    build_context_features,
    fit_gate_thresholds,
    interpolate_gated_h0,
    runtime_signals,
    sequence_window_positions,
)
from framepose.contract import JOINT_COUNT
from framepose_fixtures import prepared_dataset


SEQUENCES = {
    "train": ["3dpw:ctx_train:actor0"],
    "validation": ["3dpw:ctx_validation:actor0"],
    "test": ["3dpw:ctx_test:actor0"],
}


def _bank(tmp_path):
    requests = [BankRequest(
        "3DPW", split, prepared_dataset(tmp_path / f"{split}.json", split=split, sequences=names),
    ) for split, names in SEQUENCES.items()]
    bank, _ = build_bank(requests)
    return bank


def test_fixed_context_is_same_sequence_only_and_boundary_masked(tmp_path):
    bank = _bank(tmp_path)
    windows = sequence_window_positions(bank)
    assert windows.shape == (len(bank), len(WINDOW_OFFSETS))
    for row, sample in enumerate(bank.samples):
        for position in windows[row]:
            if position >= 0:
                assert bank.samples[int(position)].sequence_id == sample.sequence_id
    first = bank.indices("train")[0]
    second = bank.indices("train")[1]
    last = bank.indices("train")[-1]
    assert windows[first, 0] == -1
    assert windows[first, 1] == -1
    assert windows[second, 0] == -1
    assert windows[last, -1] == -1

    h0 = bank.arrays["target_3d"].copy()
    features, _ = build_context_features(bank, h0, windows)
    assert features.shape == (len(bank), CONTEXT_DIM)
    assert CONTEXT_DIM == len(WINDOW_OFFSETS) * (JOINT_COUNT * JOINT_FEATURES + 3)
    # The absent t-2 slot still consumes its complete masked feature block;
    # t-1's offset therefore remains readable at the fixed slot boundary.
    slot = JOINT_COUNT * JOINT_FEATURES + 3
    assert features[second, 0] == 0.0
    assert features[second, slot] == 1.0
    assert features[second, slot + 1] == -1.0


def test_runtime_signals_and_thresholds_do_not_read_targets(tmp_path):
    bank = _bank(tmp_path)
    h0 = bank.arrays["target_3d"].copy()
    first = runtime_signals(bank, h0)
    thresholds = fit_gate_thresholds(first, bank.indices("train"))
    decision = apply_gate(first, thresholds)

    changed = bank.arrays["target_3d"].copy()
    changed += 17.0
    second = runtime_signals(bank, h0)
    second_target_change = runtime_signals(bank, h0)
    assert np.array_equal(first["observation_residual"], second["observation_residual"], equal_nan=True)
    assert np.array_equal(first["h0_residual"], second_target_change["h0_residual"], equal_nan=True)
    assert decision.reason.shape == (len(bank), JOINT_COUNT)
    assert thresholds.fit_split == "train"
    assert thresholds.observation_p50.shape == (JOINT_COUNT,)
    # `changed` is intentionally unused above: the gate API has no target argument.
    assert changed.shape == bank.arrays["target_3d"].shape


def test_gate_off_is_an_exact_h0_noop_and_interpolation_is_gated(tmp_path):
    bank = _bank(tmp_path)
    h0 = bank.arrays["target_3d"].copy()
    signals = runtime_signals(bank, h0)
    thresholds = fit_gate_thresholds(signals, bank.indices("train"))
    decision = apply_gate(signals, thresholds)
    recovered = interpolate_gated_h0(h0, signals, decision.gate_on)
    assert np.array_equal(recovered[~decision.gate_on], h0[~decision.gate_on], equal_nan=True)
    assert np.array_equal(decision.gate_on[~signals["support"]], np.zeros_like(decision.gate_on[~signals["support"]]))


def test_refiner_checkpoint_round_trip_and_target_frame_output(tmp_path):
    torch = pytest.importorskip("torch")
    from framepose.context_refiner import RefinerConfig, load_checkpoint, predict, train_refiner

    bank = _bank(tmp_path)
    h0 = bank.arrays["target_3d"].copy() + 0.01
    contexts, _ = build_context_features(bank, h0)
    signals = runtime_signals(bank, h0)
    thresholds = fit_gate_thresholds(signals, bank.indices("train"))
    decision = apply_gate(signals, thresholds)
    checkpoint = tmp_path / "context.pt"
    report = train_refiner(
        bank, h0, thresholds,
        RefinerConfig(epochs=1, batch_size=32, device="cpu", mixed_precision=False),
        contexts=contexts, gate=decision, checkpoint_path=checkpoint,
    )
    assert report["architecture"]["architecture"] == "small_temporal_mlp_v1"
    model, _ = load_checkpoint(checkpoint)
    output = predict(model, contexts, h0, decision.gate_on, bank.indices("test"), device="cpu")
    assert output.shape == (len(bank.indices("test")), JOINT_COUNT, 3)
    assert np.isfinite(output).all()
    assert torch is not None
