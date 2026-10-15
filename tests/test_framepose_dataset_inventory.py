import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from inventory_existing_pose_datasets import DATASETS, classify, main, walk  # noqa: E402


def test_classification_rules_are_ordered_and_strict():
    assert classify(0, 0, False, False, False) == "absent"
    # Evaluation reservation and prior same-purpose use win over everything else.
    assert classify(100, 6, True, False, True) == "evaluation_reserved"
    assert classify(100, 6, True, True, False) == "already_used_same_purpose"
    # Annotation-only data is never paired RGB<->3D supervision.
    assert classify(0, 32, True, False, False) == "annotation_only"
    assert classify(50, 0, True, False, False) == "no_3d_camera_labels"
    assert classify(50, 3, False, False, False) == "no_3d_camera_labels"
    assert classify(50, 3, True, False, False) == "eligible_new_paired_training"


def test_declared_roles_match_the_worklog_history():
    assert DATASETS["MPI-INF-3DHP official TEST"]["evaluation_reserved"] is True
    assert DATASETS["MPI-INF-3DHP official TEST"]["used_for_paired_training"] is False
    for name in ("3DPW", "BEDLAM"):
        assert DATASETS[name]["used_for_paired_training"] is True
    for name in ("MPI-INF-3DHP TRAIN (S1-S8)", "AMASS", "Human3.6M", "AGORA (labels shipped inside the BEDLAM label bundle)"):
        assert DATASETS[name]["used_for_paired_training"] is False and DATASETS[name]["evaluation_reserved"] is False


def test_walk_and_main_on_a_synthetic_tree(tmp_path):
    home = tmp_path / "home"
    (home / "animcv-data/datasets/3dpw/imageFiles/s").mkdir(parents=True)
    (home / "animcv-data/datasets/3dpw/imageFiles/s/image_00000.jpg").write_bytes(b"x")
    (home / "animcv-data/datasets/3dpw/sequenceFiles/train").mkdir(parents=True)
    (home / "animcv-data/datasets/3dpw/sequenceFiles/train/s.pkl").write_bytes(b"x")
    seq = home / "AnimCV/datasets/mpi_inf_3dhp/S1/Seq1"
    seq.mkdir(parents=True)
    (seq / "annot.mat").write_bytes(b"x")
    (seq / "camera.calibration").write_text("c")
    w = walk(seq.parent.parent, DATASETS["MPI-INF-3DHP TRAIN (S1-S8)"])
    assert (w["images"], w["videos"], w["labels_3d"], w["calibration_files"]) == (0, 0, 1, 1)
    out = tmp_path / "inv.json"
    sys.argv = ["x", "--home", str(home), "--out", str(out)]
    main()
    rep = json.loads(out.read_text())
    assert rep["datasets"]["MPI-INF-3DHP TRAIN (S1-S8)"]["classification"] == "annotation_only"
    assert rep["datasets"]["3DPW"]["classification"] == "already_used_same_purpose"
    assert rep["datasets"]["Human3.6M"]["classification"] == "absent"
    assert rep["eligible_new_paired_training"] == []


def test_inventory_is_offline_and_read_only():
    src = (ROOT / "scripts/inventory_existing_pose_datasets.py").read_text()
    code = src.split('"""', 2)[2]
    assert not re.search(r"\b(urllib|requests|http\.client|socket|subprocess|wget|curl)\b", code)
    assert not re.search(r"credential|\.netrc|token", code, re.IGNORECASE)
    assert "STOP: inventory output exists" in code
    assert not re.search(r"\.unlink\(|rmtree|os\.remove|\.rename\(", code)
