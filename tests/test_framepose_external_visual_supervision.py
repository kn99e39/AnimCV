import re
from pathlib import Path

import numpy as np
import pytest

from framepose.bedlam_supervision import (
    CANONICAL_SMPLX, COCO_LIMB_TO_SMPLX, SCENES_VALIDATION, SMPL_TO_CANONICAL, ExternalBank, ExternalSample, ImageRef,
    PersonIndex, associate, canonical_target, is_closeup, is_validation_subject, opencv_to_animcv, project, scene_of,
    select_rows, select_rows_by_person, subject_key,
)
from framepose.openpose_coco18 import COCO18, to_canonical_input
from pose.pose_lifter import H36M_NAMES

ROOT = Path(__file__).resolve().parent.parent


def _bank(n=3, split="train"):
    rng = np.random.default_rng(0)
    samples = [ExternalSample(f"bedlam:s:seq_{i:06d}#row{i:07d}", "BEDLAM", f"bedlam:s:seq_{i:06d}", i, split, (720, 1280),
                              ImageRef("bedlam_frames", f"s/seq_{i:06d}/f.png")) for i in range(n)]
    arrays = {"input_2d": rng.random((n, 17, 3)), "input_valid": rng.random((n, 17)) > 0.2,
              "target_3d": rng.normal(size=(n, 17, 3)), "target_valid": rng.random((n, 17)) > 0.2}
    return ExternalBank(samples, arrays)


def test_external_bank_identity_roundtrip_and_tamper(tmp_path):
    bank = _bank()
    bank.save(tmp_path / "b")
    again = ExternalBank.load(tmp_path / "b")
    assert again.content_digest() == bank.content_digest()
    assert again.samples[1].image_reference.relative_path == "s/seq_000001/f.png"
    arrays = dict(np.load(tmp_path / "b.npz"))
    arrays["target_3d"][0, 0, 0] += 1.0
    np.savez(tmp_path / "b.npz", **arrays)
    with pytest.raises(SystemExit):
        ExternalBank.load(tmp_path / "b")


def test_canonical_axes_left_right_up_down_near_far():
    # OpenCV camera: +x right, +y down, +z forward.  AnimCV: +X right, +Y forward, +Z up.
    right, down, far = np.array([1.0, 0, 0]), np.array([0, 1.0, 0]), np.array([0, 0, 1.0])
    assert opencv_to_animcv(right).tolist() == [1, 0, 0]
    assert opencv_to_animcv(down).tolist() == [0, 0, -1]
    assert opencv_to_animcv(far).tolist() == [0, 1, 0]
    joints = np.zeros((22, 3))
    joints[:, 2] = 3.0
    joints[16] = [-0.2, -0.3, 3.0]   # left shoulder (subject's left), up in the image
    joints[18] = [-0.2, -0.1, 2.7]   # elbow below and nearer
    joints[20] = [-0.1, 0.0, 2.5]    # wrist further right and nearer
    target, valid = canonical_target(joints)
    s, e, w = (target[H36M_NAMES.index(n)] for n in ("left_shoulder", "left_elbow", "left_wrist"))
    assert target[0].tolist() == [0, 0, 0] and valid.all()
    assert e[2] < s[2]                 # elbow below shoulder -> lower Z
    assert e[1] < s[1] and w[1] < e[1]  # nearer the camera -> smaller forward Y
    assert w[0] > e[0]                 # further right -> larger X
    behind = joints.copy()
    behind[20, 2] = -1.0
    assert not canonical_target(behind)[1][H36M_NAMES.index("left_wrist")]


def test_exact_left_arm_mapping_matches_3dpw_adapter():
    from pose.three_dpw_adapter import _SMPL_TO_CANONICAL

    assert SMPL_TO_CANONICAL == _SMPL_TO_CANONICAL
    idx = {n: CANONICAL_SMPLX[H36M_NAMES.index(n)] for n in ("left_shoulder", "left_elbow", "left_wrist")}
    assert idx == {"left_shoulder": 16, "left_elbow": 18, "left_wrist": 20}
    assert COCO_LIMB_TO_SMPLX[COCO18.index("left_shoulder")] == 16
    assert COCO_LIMB_TO_SMPLX[COCO18.index("left_elbow")] == 18
    assert COCO_LIMB_TO_SMPLX[COCO18.index("left_wrist")] == 20


def test_projection_correspondence():
    k = np.array([[1000.0, 0, 640], [0, 1000.0, 360], [0, 0, 1]])
    uv = project(np.array([[0.0, 0, 2], [0.2, -0.1, 2]]), k)
    assert np.allclose(uv, [[640, 360], [740, 310]])


def test_detector_owns_input_and_matches_3dpw_rules():
    from pose.three_dpw_adapter import _canonical_2d
    from training.temporal_lifter import build_dataset  # noqa: F401  (rule source)

    rng = np.random.default_rng(1)
    kp = np.column_stack([rng.uniform(0, 720, 18), rng.uniform(0, 1280, 18), rng.uniform(0.2, 1, 18)])
    kp[COCO18.index("right_hip"), 2] = 0.0  # one invisible parent -> invisible pelvis midpoint
    inp, visible = to_canonical_input(kp, (720, 1280))
    lm = _canonical_2d(kp.T)
    for j, name in enumerate(H36M_NAMES):
        ref = lm["neck" if name == "thorax" else name]
        assert inp[j].tolist() == pytest.approx([ref.x / 720, ref.y / 1280, ref.confidence], abs=1e-6)
        assert visible[j] == ref.visible
    assert not visible[H36M_NAMES.index("pelvis")] and inp[H36M_NAMES.index("pelvis"), 2] == 0
    assert inp[H36M_NAMES.index("thorax")].tolist() == inp[H36M_NAMES.index("neck")].tolist()


def _person(gt_uv, offset):
    kp = np.zeros((18, 3))
    for c, j in COCO_LIMB_TO_SMPLX.items():
        kp[c] = [gt_uv[j, 0] + offset, gt_uv[j, 1], 0.9]
    return {"keypoints": kp, "score": 1.0, "parts": 12}


def test_gt_only_selects_identity_and_rejects_non_subjects():
    gt = np.zeros((127, 2))
    rng = np.random.default_rng(2)
    gt[:22] = np.column_stack([rng.uniform(300, 400, 22), rng.uniform(300, 700, 22)])
    people = [_person(gt, 200.0), _person(gt, 2.0)]
    idx, info = associate(people, gt, (1280, 720))
    assert idx == 1 and info["correct_limb_joints"] == 12
    # The model input is the detector's keypoints, not GT: a slightly different GT picks the same person
    # and the canonical input is unchanged.
    idx2, _ = associate(people, gt + 1.0, (1280, 720))
    assert idx2 == 1
    assert np.array_equal(to_canonical_input(people[idx]["keypoints"], (1280, 720))[0],
                          to_canonical_input(people[idx2]["keypoints"], (1280, 720))[0])
    far_only = [_person(gt, 200.0)]
    assert associate(far_only, gt, (1280, 720))[0] is None
    assert associate([], gt, (1280, 720))[0] is None


def test_row_selection_is_deterministic_and_label_blind():
    names = np.array([f"seq_{s:06d}/seq_{s:06d}_{f:04d}.png" for s in range(4) for f in range(0, 50, 5) for _ in range(2)])
    perm = np.random.default_rng(3).permutation(len(names))
    a = select_rows(names)
    b = perm[select_rows(names[perm])]
    assert sorted(names[a].tolist()) == sorted(names[b].tolist())
    assert len(a) == 4 * 3
    for s in range(4):
        picked = [n for n in names[a] if n.startswith(f"seq_{s:06d}")]
        assert picked[0].endswith("_0000.png") and picked[-1].endswith("_0045.png")


def test_scene_naming_closeup_and_official_split():
    assert scene_of("x/20221011_1_250_batch01hand_closeup_suburb_a_6fps.npz") == "20221011_1_250_batch01hand_closeup_suburb_a"
    assert scene_of("20221022_3_250_batch01handhair_static_bigOffice_30fps.npz") == "20221022_3_250_batch01handhair_static_bigOffice"
    assert is_closeup("20221011_1_250_batch01hand_closeup_suburb_a") and not is_closeup("20221010_3_1000_batch01hand")
    assert len(SCENES_VALIDATION) == 4 and len(set(SCENES_VALIDATION)) == 4


def test_source_balanced_sampler_exact_half_and_budget():
    pytest.importorskip("torch")
    import sys
    sys.path.insert(0, str(ROOT / "scripts"))
    from run_external_visual_supervision_transfer import SourceBalancedSampler

    primary, external = np.arange(0, 100), np.arange(1000, 1300)
    s = SourceBalancedSampler(primary, external, 11334, 1337)
    t = SourceBalancedSampler(primary, external, 11334, 1337)
    for _ in range(3):
        a, b = s.epoch(), t.epoch()
        assert np.array_equal(a, b) and len(a) == 11334
        assert (a < 1000).sum() == 5667 and (a >= 1000).sum() == 5667
        assert set(a[a < 1000]) <= set(primary) and set(a[a >= 1000]) <= set(external)
    with pytest.raises(ValueError):
        SourceBalancedSampler(primary, external, 11335, 1337)


def test_contracts_are_enforced_in_scripts():
    builder = (ROOT / "scripts/build_bedlam_external_bank.py").read_text()
    for check in ("SMPL-X reprojection error", "selected images missing", "does not match principal point",
                  "to_canonical_input(people[idx][\"keypoints\"]", "target_valid = visible & gt_valid",
                  "if not target_valid[0]", "cv2.ROTATE_90_CLOCKWISE"):
        assert check in builder
    assert "gtkps" not in builder.split("to_canonical_input(people[idx]")[1].split("target, gt_valid")[0]
    extractor = (ROOT / "scripts/extract_external_qwen3vl_features.py").read_text()
    assert "import extract_qwen3vl_visual_features as w68" in extractor and "w68.main()" in extractor
    run = (ROOT / "scripts/run_external_visual_supervision_transfer.py").read_text()
    for check in ("Qwen cache identity mismatch", "does not share the Worklog 68 representation contract",
                  "Worklog 69 head identity mismatch", "external TRAIN/validation overlap (sequence, image or subject)",
                  "E0 does not reproduce Worklog 69 C1", "E1 optimization budget / graph differs from E0",
                  "from run_visual_supervision_diversity import CONFIG, train", "rows_heldout.json",
                  "S0_FRAME_UNIFORM"):
        assert check in run
    for path in sorted((ROOT / "src").rglob("*.py")):
        if path.name in ("bedlam_supervision.py", "openpose_coco18.py"):
            continue
        text = path.read_text()
        assert "bedlam_supervision" not in text and "openpose_coco18" not in text, path


def test_adapter_reads_no_target_in_selection():
    src = (ROOT / "src/framepose/bedlam_supervision.py").read_text()
    sel = src[src.index("def select_rows"):src.index("def opencv_to_animcv")]
    assert not re.search(r"pose_cam|trans_cam|gtkps|target|joints", sel.split('"""', 2)[2])


def test_subject_split_is_deterministic_and_groups_renderpeople_scans():
    assert subject_key("rp_antonia_posed_003") == subject_key("rp_antonia_posed_006") == "rp_antonia"
    assert subject_key("male_40_us_0306") == "male_40_us_0306"
    assert is_validation_subject("rp_antonia_posed_003") == is_validation_subject("rp_antonia_posed_006")
    people = [f"female_25_us_{i:04d}" for i in range(400)]
    share = np.mean([is_validation_subject(p) for p in people])
    assert 0.1 < share < 0.3  # ~1/5 by construction


def test_select_rows_by_person_keeps_only_requested_people():
    names, persons = [], []
    for s in range(3):
        for person in ("a", "b"):
            for f in range(0, 40, 5):
                names.append(f"seq_{s:06d}/seq_{s:06d}_{f:04d}.png")
                persons.append(person)
    rows = select_rows_by_person(np.array(names), np.array(persons), lambda p: p == "b")
    assert len(rows) == 3 * 3 and all(persons[r] == "b" for r in rows)


def test_person_index_recovers_person_from_betas(tmp_path):
    rng = np.random.default_rng(4)
    table = {}
    for person in ("rp_aaron_posed_002", "male_30_us_1377"):
        for motion in ("0001", "0002"):
            d = tmp_path / person / motion
            d.mkdir(parents=True)
            b = rng.normal(size=11)
            np.savez(d / "motion_seq.npz", betas=b)
            table[(person, motion)] = b
    index = PersonIndex(tmp_path)
    assert index.lookup(table[("male_30_us_1377", "0002")].astype(np.float32)) == "male_30_us_1377"
    assert index.lookup(table[("rp_aaron_posed_002", "0001")] + 1e-3) is None
