import math
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from select_bedlam_coverage_expansion import NEW_ROWS, POLICY, SEED, allocate, cells, dedup_tracks  # noqa: E402


def test_cells_cover_72_bins_and_follow_the_declared_edges():
    e = np.array([[0.4, 0.0, 0.3], [0.4, 0.0, -0.3], [0.5, 0.0, 0.0]])
    w = e * 2
    c = cells(e, w, np.array([5.0, 100.0, 30.0]), np.array([0.0, -0.9, 0.3]))
    assert ((c >= 0) & (c < 72)).all()
    # bend 5 / |f| 0 / f>=0 / elevation 0.6 -> ((0*3+0)*2+1)*3+2
    assert c[0] == 5
    # bend 100 / |f| 0.9 / f<0 / elevation -0.6 -> ((3*3+2)*2+0)*3+0
    assert c[1] == 66
    assert c[2] == ((1 * 3 + 1) * 2 + 1) * 3 + 1
    rng = np.random.default_rng(0)
    e, w = rng.normal(size=(5000, 3)), rng.normal(size=(5000, 3))
    allc = cells(e, w, rng.uniform(0, 180, 5000), rng.uniform(-1, 1, 5000))
    assert len(set(allc.tolist())) == 72


def test_dedup_keeps_distinct_frames_and_is_seeded_by_history():
    track = np.array(["a"] * 5 + ["b"] * 2)
    frame = np.array([0, 5, 10, 15, 20, 0, 5])
    e = np.array([[0.40, 0, 0], [0.41, 0, 0], [0.60, 0, 0], [0.61, 0, 0], [0.40, 0, 0], [0.4, 0, 0], [0.4, 0, 0]])
    w = e * 2
    cand = np.array([True, True, True, True, True, True, True])
    seed = np.zeros(7, bool)
    keep = dedup_tracks(track, frame, e, w, cand, seed, 0.15)
    # frame 5 is ~0.03 from frame 0; frame 15 ~0.03 from 10; frame 20 duplicates frame 0; track b: second is identical
    assert keep.tolist() == [True, False, True, False, False, True, False]
    seed = np.array([True, False, False, False, False, False, False])
    keep = dedup_tracks(track, frame, e, w, cand & ~seed, seed, 0.15)
    assert not keep[0] and not keep[1] and keep[2]


def test_allocation_is_sqrt_proportional_and_water_filled():
    q = allocate(np.array([10000, 100, 0, 1]), 300)
    assert q.sum() == 300 and q[2] == 0 and q[3] <= 1 and q[0] > q[1]
    assert abs(q[0] / q[1] - 10) < 1.5  # sqrt(10000)/sqrt(100)
    q = allocate(np.array([5, 5]), 100)
    assert q.tolist() == [5, 5]  # never more than available
    assert NEW_ROWS == 80000 and SEED == 1337 and POLICY["dedup_min_arm_distance"] == 0.15
    assert POLICY["abs_f_bins"][1] == math.sin(math.radians(10))


def test_selection_excludes_validation_subjects_history_and_non_rgb():
    src = (ROOT / "scripts/select_bedlam_coverage_expansion.py").read_text()
    body = src.split("def main():", 1)[1]
    line = next(l for l in body.splitlines() if l.strip().startswith("eligible ="))
    for need in ('z["rgb"]', "training_subject", 'z["gt_arm_ok"]', 'z["arm_in_image"]', '~z["hist_selected"]'):
        assert need in line
    assert "~z[\"validation_subject\"]" in body and "STOP: a historical TRAIN row belongs to a validation subject" in body
    assert not re.search(r"predict|checkpoint|\.pt\b|readout|validation_loss", body)


def test_builder_reuses_worklog73_pipeline_and_training_contract_is_unchanged():
    build = (ROOT / "scripts/build_bedlam_expanded_bank.py").read_text()
    assert "from build_bedlam_external_bank import MAX_REPROJECTION_PX, finish_scene, process_member" in build
    assert "STOP: validation subject in the expansion bank" in build and "identity" in build
    assert not re.search(r"gtkps\]\[.*input|input_2d.*gtkps", build)
    train = (ROOT / "scripts/train_framepose_bedlam_expanded_arms.py").read_text()
    assert "from train_framepose_joint_arm_geometry import train_joint" in train
    assert "train_correction" in train and "requires_grad_(False)" in train
    assert "STOP: the training pipeline does not reproduce the historical J1/R1 checkpoints" in train
    assert "STOP: BEDLAM train/validation leakage" in train and "STOP: frozen J0 changed" in train
    body = train.split("def main():", 1)[1]
    assert body.index("training_reproduction_gate") < body.index("res = train_pair(xd")
    assert not re.search(r"mpi3dhp|3dhp|\"test\"|indices\(\"test\"\)", train)
    for path in sorted((ROOT / "src").rglob("*.py")):
        text = path.read_text()
        assert "bedlam_expanded" not in text and "coverage_expansion" not in text, path
