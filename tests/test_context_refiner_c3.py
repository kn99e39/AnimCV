"""Focused contract checks for the one controlled C3 experiment."""

from __future__ import annotations

import numpy as np
import pytest

from framepose.bank import BankRequest, build_bank
from framepose.context_refiner import (
    GateDecision,
    apply_gate,
    build_context_features,
    build_refiner,
    fit_gate_thresholds,
    predict,
    runtime_signals,
)
from framepose_fixtures import prepared_dataset


def _bank(tmp_path):
    requests = [BankRequest(
        "3DPW", split, prepared_dataset(tmp_path / f"{split}.json", split=split,
                                         sequences=[f"3dpw:c3_{split}:actor0"]),
    ) for split in ("train", "validation", "test")]
    bank, _ = build_bank(requests)
    return bank


def test_c3_contract_keeps_gate_off_exact_h0_and_declares_aligned_coordinate_domain(tmp_path):
    torch = pytest.importorskip("torch")
    from scripts.run_context_refiner_c3 import train_c3

    bank = _bank(tmp_path)
    h0 = bank.arrays["target_3d"].copy().astype(np.float32) + 0.01
    contexts, _ = build_context_features(bank, h0)
    signals = runtime_signals(bank, h0)
    thresholds = fit_gate_thresholds(signals, bank.indices("train"))
    fitted = apply_gate(signals, thresholds)
    gate = fitted.gate_on.copy()
    gate[:] = False
    gate[bank.indices("train")[1], 0] = True
    decision = GateDecision(gate, fitted.reason, fitted.observation_residual,
                            fitted.h0_residual, fitted.left, fitted.right)
    checkpoint = tmp_path / "c3.pt"
    report = train_c3(
        bank, h0, contexts, decision,
        {"epochs": 1, "batch_size": 32, "learning_rate": 1e-3,
         "weight_decay": 1e-4, "seed": 1337, "device": "cpu",
         "mixed_precision": False}, checkpoint,
    )
    assert report["supervision"]["loss_contract"]["name"] == "coordinate_only_v1"
    assert report["supervision"]["mask"] == "target_valid AND runtime_gate_on joint rows"

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = build_refiner(hidden=128)
    model.load_state_dict(payload["state_dict"])
    output = predict(model, contexts, h0, gate, np.arange(len(bank)), device="cpu")
    assert np.array_equal(output[~gate], h0[~gate])
