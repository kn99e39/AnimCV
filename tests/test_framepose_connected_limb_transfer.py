import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from framepose.connected_limb_transfer import (
    CHAINS, chain_indices, chain_pair_geometry, chain_readout, chain_targets, token_coordinates,
)
from framepose.contract import JOINT_NAMES
from framepose.joint_arm_geometry import normalized_arm_targets
from framepose.learned_vision_sensor import pair_geometry, readout as w68_readout
from framepose.learned_vision_sensor import token_coordinates as w68_token_coordinates

ROOT = Path(__file__).resolve().parent.parent
RNG = np.random.default_rng(86)


def test_chain_identities_are_anatomical_and_left_right_consistent():
    for chain, joints in CHAINS.items():
        side, part = chain.split("_")
        assert all(j.startswith(side + "_") for j in joints)
        assert [JOINT_NAMES[i] for i in chain_indices(chain)] == list(joints)
    assert CHAINS["right_arm"] == tuple(j.replace("left", "right") for j in CHAINS["left_arm"])
    assert CHAINS["right_leg"] == ("right_hip", "right_knee", "right_ankle")


def test_left_arm_contract_reproduces_the_historical_arrays_exactly():
    n = 300
    t3 = RNG.normal(size=(n, 17, 3)).astype(np.float32)
    tv = RNG.random((n, 17)) > 0.1
    e, w, ok = chain_targets(t3, tv, "left_arm")
    e0, w0, ok0 = normalized_arm_targets(t3, tv)
    assert np.array_equal(ok, ok0) and np.array_equal(e, e0) and np.array_equal(w, w0)
    g = RNG.normal(size=(n, 17, 4)).astype(np.float32)
    g[:, :, 3] = RNG.random((n, 17)) > 0.2
    assert np.array_equal(chain_pair_geometry(g, "left_arm"), pair_geometry(g))


def test_chain_targets_are_connected_unit_chains_for_every_limb():
    n = 200
    t3 = RNG.normal(size=(n, 17, 3))
    tv = np.ones((n, 17), bool)
    for chain in CHAINS:
        e, w, ok = chain_targets(t3, tv, chain)
        assert ok.all()
        assert np.allclose(np.linalg.norm(e, axis=1) + np.linalg.norm(w - e, axis=1), 1.0)
        i = chain_indices(chain)
        total = np.linalg.norm(t3[:, i[1]] - t3[:, i[0]], axis=1) + np.linalg.norm(t3[:, i[2]] - t3[:, i[1]], axis=1)
        assert np.allclose(w * total[:, None], t3[:, i[2]] - t3[:, i[0]])
    tv[:, chain_indices("right_leg")[2]] = False
    assert not chain_targets(t3, tv, "right_leg")[2].any() and chain_targets(t3, tv, "right_arm")[2].all()


def test_readout_matches_the_worklog68_function_on_the_same_tokens():
    tokens = RNG.normal(size=(196, 32)).astype(np.float32)
    inp = np.zeros((17, 3), np.float32)
    inp[:, :2] = RNG.uniform(0.2, 0.8, size=(17, 2))
    vis = RNG.random(17) > 0.2
    size, box = (1280, 720), SimpleNamespace(x=300.0, y=50.0, side=600.0)
    assert np.allclose(token_coordinates(inp, size, (box.x, box.y, box.side)), w68_token_coordinates(inp, size, box))
    ref, ref_valid = w68_readout(tokens, inp, vis, size, box)
    got = chain_readout(lambda idx: tokens[idx], inp, vis, size, (box.x, box.y, box.side), "left_arm", tokens.mean(0))
    assert np.array_equal(got, ref) and np.array_equal(vis[chain_indices("left_arm")], ref_valid)
    other = chain_readout(lambda idx: tokens[idx], inp, vis, size, (box.x, box.y, box.side), "right_leg", tokens.mean(0))
    for k, j in enumerate(chain_indices("right_leg")):
        assert (np.abs(other[k]).sum() > 0) == bool(vis[j])


def test_module_has_no_target_or_oracle_dependency_and_no_src_coupling():
    src = (ROOT / "src/framepose/connected_limb_transfer.py").read_text()
    readout_fn = src.split("def chain_readout", 1)[1]
    assert not re.search(r"target|gt_|oracle", readout_fn.split('"""', 2)[2], re.IGNORECASE)
    assert "learned_vision_sensor" not in src and "joint_arm_geometry" not in src
    for path in sorted((ROOT / "src").rglob("*.py")):
        if path.name != "connected_limb_transfer.py":
            assert "connected_limb_transfer" not in path.read_text(), path


def test_decision_helpers_and_runner_contracts():
    import sys
    sys.path.insert(0, str(ROOT / "scripts"))
    from run_framepose_connected_limbs_eval import DECISION, PROTOCOL, _regress, _strong
    items = [{"sequence_id": f"s{k % 6}", "candidates": {"g1": {"m": 1.0 + 0.1 * (k % 7)}, "g0": {"m": 2.0}}} for k in range(120)]
    from run_framepose_connected_limbs_eval import _cmp
    c = _cmp(items, "g1", "g0", "m")
    assert _strong(c) and not _regress(c)
    flat = [{"sequence_id": it["sequence_id"], "candidates": {"g1": {"m": 2.0}, "g0": {"m": 2.0}}} for it in items]
    assert not _strong(_cmp(flat, "g1", "g0", "m"))
    assert "CASE_C" in DECISION and "chain_transfers" in DECISION and "populations" in PROTOCOL
    ev = (ROOT / "scripts/run_framepose_connected_limbs_eval.py").read_text()
    assert ev.index("DECISION = {") < ev.index("def main():") and not re.search(r"import torch|smooth", ev)
    train = (ROOT / "scripts/train_framepose_connected_limbs.py").read_text()
    assert "from train_framepose_joint_arm_geometry import train_joint" in train
    for check in ("STOP: the chain pipeline does not reproduce J1X", "STOP: G0/G1 control identity broken",
                  "STOP: the contract audit selected a different leg", "STOP: limb readout cache identity mismatch"):
        assert check in train
    body = train.split("def main():", 1)[1]
    assert body.index("generic_pipeline_gate") < body.index("for chain, cand, readouts in jobs")
    assert not re.search(r"mpi3dhp|indices\(\"test\"\)", train)
    pred = (ROOT / "scripts/predict_framepose_connected_limbs.py").read_text()
    assert "STOP: left-arm reference does not reproduce Worklog 85" in pred
    audit = (ROOT / "scripts/audit_connected_limb_contracts.py").read_text()
    assert audit.index("LEG_RULE =") < audit.index("def main():")
