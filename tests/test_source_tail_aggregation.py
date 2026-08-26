"""Contracts for the diagnostic source-tail aggregation counterfactual."""

import importlib.util
from pathlib import Path
import sys

import pytest

pytest.importorskip("torch", reason="source-tail diagnostics require the training extra")


_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT = _ROOT / "scripts" / "diagnose_source_tail_aggregation.py"


def _load_module():
    sys.path.insert(0, str(_ROOT / "src"))
    try:
        spec = importlib.util.spec_from_file_location("diagnose_source_tail_aggregation", _SCRIPT)
        module = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.pop(0)


def test_source_balanced_replay_is_deterministic_and_balanced():
    import torch

    module = _load_module()
    source_ids = torch.tensor([0] * 8 + [1] * 8 + [2] * 8)
    first = module._source_balance(torch, source_ids, ["alpha", "beta", "gamma"], 1337, 4, 6)
    second = module._source_balance(torch, source_ids, ["alpha", "beta", "gamma"], 1337, 4, 6)

    assert torch.equal(first["permutation"], second["permutation"])
    assert first["sampled_epoch_mass"] == {"alpha": 8, "beta": 8, "gamma": 8}
    assert first["representative_sample_count"] == 24


def test_global_tail_accounts_for_starvation_and_weighted_mass():
    import torch

    module = _load_module()
    errors = torch.zeros((8, 2), dtype=torch.float32)
    errors[0, 0] = 3.0  # source alpha
    errors[7, 1] = 10.0  # source beta: sole global hard example
    stable = torch.ones_like(errors, dtype=torch.bool)
    source_ids = torch.tensor([0] * 4 + [1] * 4)

    detail, selected = module._source_tail_accounting(
        torch, errors, stable, source_ids, ["alpha", "beta"], weight=0.05,
    )

    assert int(selected.sum()) == 1
    assert detail["source"]["alpha"]["candidate_count_before_selection"] == 8
    assert detail["source"]["beta"]["candidate_count_before_selection"] == 8
    assert detail["source"]["alpha"]["selected_tail_count"] == 0
    assert detail["source"]["beta"]["selected_tail_count"] == 1
    assert detail["source"]["beta"]["selected_fraction_within_source"] == pytest.approx(1 / 8)
    assert detail["source"]["beta"]["weighted_auxiliary_loss_mass"] == pytest.approx(0.5)


def test_source_stratified_tail_selects_each_source_with_equal_aggregation():
    import torch

    module = _load_module()
    errors = torch.zeros((8, 2), dtype=torch.float32)
    errors[0, 0] = 3.0
    errors[7, 1] = 10.0
    stable = torch.ones_like(errors, dtype=torch.bool)
    source_ids = torch.tensor([0] * 4 + [1] * 4)

    detail = module._source_stratified_tail(torch, errors, stable, source_ids, ["alpha", "beta"])

    assert detail["active_source_count"] == 2
    assert detail["selected_tail_count"] == 2
    assert detail["raw_auxiliary_loss"] == pytest.approx(6.5)
    assert detail["source"]["alpha"]["selected_tail_count"] == 1
    assert detail["source"]["beta"]["selected_tail_count"] == 1
    assert detail["source"]["alpha"]["aggregate_loss_share"] == pytest.approx(0.5)
    assert detail["source"]["beta"]["aggregate_loss_share"] == pytest.approx(0.5)


def test_source_stratified_tail_supports_generic_n_sources_and_single_source():
    import torch

    module = _load_module()
    errors = torch.arange(40, dtype=torch.float32).reshape(20, 2)
    stable = torch.ones_like(errors, dtype=torch.bool)
    source_ids = torch.tensor([0] * 5 + [1] * 5 + [2] * 5 + [3] * 5)
    labels = ["commercial_a", "commercial_b", "commercial_c", "commercial_d"]

    many = module._source_stratified_tail(torch, errors, stable, source_ids, labels)
    one = module._source_stratified_tail(torch, errors[:5], stable[:5], torch.zeros(5, dtype=torch.long), ["one"])

    assert many["active_source_count"] == 4
    assert set(many["source"]) == set(labels)
    assert all(item["aggregate_loss_share"] == pytest.approx(0.25) for item in many["source"].values())
    assert one["active_source_count"] == 1
    assert one["raw_auxiliary_loss"] == pytest.approx(one["source"]["one"]["raw_auxiliary_loss"])
