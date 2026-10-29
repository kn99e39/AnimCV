"""Behavioral W101 data-path fixtures; run on LabServer63 with the frozen S0 checkpoint."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import w101_reference_estimator as w
from framepose.contract import FrameBank, FrameSample, ImageReference, Modality
from framepose.observations import DATASET_OBSERVATIONS, mmpose_observation, observation_cache_key


def landmarks(offset=0., confidence=.9):
    points = {"head": (500,100), "left_shoulder": (550,200), "right_shoulder": (450,200),
              "left_elbow": (580,300), "right_elbow": (420,300), "left_wrist": (590,380), "right_wrist": (410,380),
              "left_hip": (540,420), "right_hip": (460,420), "left_knee": (545,560), "right_knee": (455,560),
              "left_ankle": (550,700), "right_ankle": (450,700), "neck": (500,200), "pelvis": (500,420), "spine": (500,310)}
    return np.array([[points[n][0]+offset, points[n][1], confidence, 1.] for n in w.CANON])


@pytest.fixture
def fixture(tmp_path):
    images, caches = tmp_path / "images", tmp_path / "observe"
    images.mkdir(); caches.mkdir()
    samples, inputs, valid, rows = [], [], [], []
    provenance = mmpose_observation(pose_config="fixture", pose_checkpoint="fixture", visibility_threshold=.3,
                                    extra={"batch_size": 1})
    for split in ("validation", "test"):
        video = "fixture_"+split
        (images / video).mkdir()
        paths, digests, lms, candidates, cf = [], [], [], [], []
        for frame in (0, 5, 10):
            rel = f"{video}/image_{frame:05d}.jpg"
            (images/rel).write_bytes(b"fixed synthetic image bytes "+str(frame).encode())
            paths.append("3dpw/"+rel); digests.append(w.sha(images/rel))
            lms.append(landmarks() if frame != 5 else np.full((16,4), np.nan))
            for actor in (0,1):
                p = len(samples)
                sid = f"3dpw:{video}:actor{actor}"
                x, v = w.to_input(landmarks(actor*500), (1600,900))
                samples.append(FrameSample(sample_id=f"{sid}#{frame:06d}", source="3DPW", sequence_id=sid,
                    frame_index=frame, split=split, image_size=(1600,900), modality=Modality(True,True,True,True),
                    observation=DATASET_OBSERVATIONS["official_3dpw_2d_detection"], timestamp=frame/30., fps=30.,
                    image_reference=ImageReference("3dpw_images",rel)))
                inputs.append(x); valid.append(v)
                rows.append({"population":"3dpw_"+split, "bank_position":p, "sequence_id":sid, "frame_index":frame,
                             "image":"3dpw/"+rel,"image_size":[1600,900]})
                candidates.append(landmarks(actor*500)); cf.append(frame//5)
        np.savez_compressed(caches/(video+".npz"), frame_index=[0,5,10], timestamp=np.array([0,5,10])/30.,
                            landmarks=lms, cand_frame=cf, cand_landmarks=candidates, image_size=[[1600,900]]*3)
        meta = {"images":paths,"image_sha256":digests,"provenance":provenance.to_dict(),
                "npz_sha256":w.sha(caches/(video+".npz")),
                "observation_cache_keys":[observation_cache_key(provenance,d) for d in digests]}
        (caches/(video+".json")).write_text(json.dumps(meta))
    bank = FrameBank(samples, {"input_2d":np.array(inputs), "input_valid":np.array(valid),
                    "target_3d":np.random.default_rng(1).normal(size=(len(samples),17,3)).astype(np.float32),
                    "target_valid":np.ones((len(samples),17),bool)})
    # Record original adjacency, so filtered frames can never create a new temporal neighbor.
    for s in bank.samples:
        s.neighbors.update(previous=f"{s.sequence_id}#{s.frame_index-5:06d}" if s.frame_index else None,
                           next=f"{s.sequence_id}#{s.frame_index+5:06d}" if s.frame_index<10 else None)
    return bank, {"rows":rows}, caches, images


def test_mapping_normalization_and_nonfinite_values():
    lm = landmarks()
    x,v = w.to_input(lm,(1600,900))
    assert x[0,:2] == pytest.approx([500/1600,420/900])
    assert np.array_equal(x[8],x[9])
    assert x[10,:2] == pytest.approx([500/1600,100/900])
    for field in (0,1,2):
        bad = lm.copy(); bad[w.CANON.index("left_wrist"),field] = np.nan
        xx,vv = w.to_input(bad,(1600,900))
        assert not vv[13] and not xx[13].any() and np.isfinite(xx).all()
    lm[w.CANON.index("left_wrist"),3]=0
    assert not w.to_input(lm,(1600,900))[1][13]
    assert w.to_input(lm,(1600,900),raw=True)[1][13]
    lm[w.CANON.index("left_wrist"),0]=-10
    assert w.to_input(lm,(1600,900),raw=True)[0][13,0]<0  # no in-frame masking added
    assert not w.to_input(None,(1600,900))[1].any()


def test_actual_build_identity_coverage_and_probes(fixture):
    bank,pop,caches,images = fixture
    obs,side = w.build_observations(bank,pop,caches,images)
    assert len(obs["positions"]) == 12
    assert list(side["identity"][:6]) == ["same_person","other_person","no_subject","no_subject","same_person","other_person"]
    assert not obs["input_valid_M"][2].any() and not obs["input_2d_M"][2].any()
    assert obs["input_valid_Moa"].all()  # candidate available even on deliberately missing tracked frame
    assert side["candidate_found_tracker_missed"].sum() == 8
    report = w.population_report(bank,obs,side)
    for split in ("validation","test"):
        r = report[split]
        assert (r["actor_frames"],r["unique_images"],r["tracked_unique_images"]) == (6,3,2)
        assert r["identity"]["same_person"] == 2 and r["same_person_share"] == pytest.approx(1/3)
    assert np.array_equal(obs["input_valid_Mconf"],obs["input_valid_M"])


def test_ambiguous_association_not_silently_same_person():
    x,v = w.to_input(landmarks(),(1600,900))
    sid,status,_ = w.association(x,v,[("a",x,v,(1600,900)),("b",x,v,(1600,900))])
    assert sid is None and status == "ambiguous"
    far,_ = w.to_input(landmarks(1500),(1600,900))
    assert w.association(far,v,[("a",x,v,(1600,900))])[1] == "unresolved"


@pytest.mark.parametrize("defect", ["bytes","frame","timestamp","size","position","split","provenance"])
def test_handoff_refuses_identity_defects(fixture,defect):
    bank,pop,caches,images = fixture
    path = caches / "fixture_validation.npz"
    if defect == "bytes":
        (images/"fixture_validation/image_00000.jpg").write_bytes(b"replaced image")
    elif defect in ("position","split"):
        pop["rows"][0]["bank_position" if defect=="position" else "population"] = 100 if defect=="position" else "3dpw_train"
    elif defect == "provenance":
        mp = caches/"fixture_validation.json"; meta=json.loads(mp.read_text())
        meta["provenance"]["detail"]["batch_size"]=8;mp.write_text(json.dumps(meta))
    else:
        z=dict(np.load(path)); name={"frame":"frame_index","timestamp":"timestamp","size":"image_size"}[defect]
        z[name][0] += 1;np.savez_compressed(path,**z)
        mp=caches/"fixture_validation.json";meta=json.loads(mp.read_text());meta["npz_sha256"]=w.sha(path);mp.write_text(json.dumps(meta))
    with pytest.raises((ValueError,KeyError)):
        w.build_observations(bank,pop,caches,images)


def test_behavioral_gt_perturbation_through_actual_data_path(fixture):
    """Real cache adapter -> production crop -> frozen S0, not source-text inspection."""
    bank,pop,caches,images = fixture
    obs,side=w.build_observations(bank,pop,caches,images)
    changed=copy.deepcopy(bank)
    changed.arrays["target_3d"][:]=np.random.default_rng(42).normal(size=changed.arrays["target_3d"].shape)*100
    changed.arrays["target_valid"][:]=False
    for s in changed.samples:s.strata.update(evaluation_GT_corrupted=True)
    obs2,side2=w.build_observations(changed,pop,caches,images)
    for key in obs:assert np.array_equal(obs[key],obs2[key])
    for c in ("B",)+w.CONDITIONS:
        assert np.array_equal(w.geometry_for_condition(bank,obs,c),w.geometry_for_condition(changed,obs2,c))
    checkpoint=os.environ.get("W101_CHECKPOINT")
    if not checkpoint:
        pytest.skip("frozen S0 checkpoint not supplied; model invariance not executed")
    import torch
    from framepose.train import load_checkpoint,predict
    from framepose.signs import neutral_sign_states
    w.verify_s0_identity(checkpoint)
    model,ck=load_checkpoint(checkpoint,device="cpu")
    assert ck["candidate"]["name"]=="S0_neutral_sign"
    for c in ("B",)+w.CONDITIONS:
        a=predict(model,torch,w.geometry_for_condition(bank,obs,c),None,obs["positions"],torch.device("cpu"), signs=neutral_sign_states(len(bank)))
        b=predict(model,torch,w.geometry_for_condition(changed,obs2,c),None,obs2["positions"],torch.device("cpu"), signs=neutral_sign_states(len(changed)))
        assert np.array_equal(a,b)


def test_inference_replacement_refuses_reordered_actor_and_preserves_training(fixture):
    from dataclasses import replace
    bank,pop,caches,images=fixture;obs,_=w.build_observations(bank,pop,caches,images)
    train=replace(bank.samples[0],sample_id="3dpw:fixture_train:actor0#000000",sequence_id="3dpw:fixture_train:actor0",split="train")
    bank=FrameBank(bank.samples+[train],{k:np.concatenate([v,v[:1]],axis=0) for k,v in bank.arrays.items()})
    original=bank.arrays["input_2d"].copy()
    geometry=w.geometry_for_condition(bank,obs,"M")
    assert np.array_equal(geometry[-1],w.geometry_for_condition(bank,obs,"B")[-1])
    assert np.array_equal(original,bank.arrays["input_2d"])
    bad=dict(obs);bad["sample_id"]=obs["sample_id"][::-1]
    with pytest.raises(ValueError):w.geometry_for_condition(bank,bad,"M")


def test_video_bootstrap_keeps_actors_together_and_is_paired():
    result=w.video_bootstrap(["3dpw:a:actor0","3dpw:a:actor1","3dpw:b:actor0"],[2.,2.,-4.])
    assert result["n"]==3 and result["videos"]==2 and result["mean"]==0
    assert result["ci95"]==[-4.,2.]
    assert result==w.video_bootstrap(["3dpw:a:actor0","3dpw:a:actor1","3dpw:b:actor0"],[2.,2.,-4.])
    assert w.video_bootstrap(["3dpw:a:actor0","3dpw:a:actor1"],[1.,2.])["ci95"] is None


def test_pairing_same_limb_with_condition_specific_knee_support():
    rows=[{"sid":"3dpw:a:actor0","m":{"B":{"knee_plane_l":10.,"knee_plane_r":100.,"mpjpe":20.},
                                             "M":{"knee_plane_l":12.,"mpjpe":21.}}},
          {"sid":"3dpw:b:actor0","m":{"B":{"knee_plane_r":20.,"mpjpe":30.},
                                             "M":{"knee_plane_r":24.,"mpjpe":33.}}}]
    result=w.paired_metrics(rows,"B","M")
    assert result["knee_plane_mean"]["mean"]==3.
    assert result["mpjpe"]["mean"]==2.
    assert result["knee_plane_l"]["n"]==1


def test_two_d_pairs_use_identical_joints():
    uv=np.arange(48,dtype=float).reshape(24,2)*10
    x=np.zeros((17,3));v=np.ones(17,bool)
    for j,k in w.SMPL_OF.items():x[j,:2]=uv[k]/np.array([1600,900])
    shared=v.copy();shared[13]=False
    x[13,:2]+=1
    assert w.two_d_metrics(x,v,uv,(1600,900),shared)["err"]==pytest.approx(0.)
    assert w.two_d_metrics(x,v,uv,(1600,900))["err"]>0


def test_temporal_never_bridges_filtered_neighbors():
    G=np.random.default_rng(1).normal(size=(17,3))
    def row(frame,ident):
        return {"sample_id":f"3dpw:a:actor0#{frame:06d}","sid":"3dpw:a:actor0","next":f"3dpw:a:actor0#{frame+5:06d}",
                "timestamp":frame/30.,"identity":ident,"tracked":True,"J":{"M":G.copy()},"G":G.copy()}
    a,b,c=row(0,"same_person"),row(5,"other_person"),row(10,"same_person")
    assert w.temporal([a,c],"M")["steps"]==0
    assert w.temporal([a,b,c],"M")["steps"]==0
    b["identity"]="same_person"
    result=w.temporal([a,b,c],"M")
    assert result["steps"]==2 and result["limb_direction_jumps"]==0


def test_checkpoint_identity_fails_before_inference(tmp_path):
    path=tmp_path/"checkpoint.pt";path.write_bytes(b"wrong checkpoint")
    with pytest.raises(RuntimeError,match="checkpoint changed"):
        w.verify_s0_identity(path)


def test_selection_free_inventory_checks_bytes_dimensions_and_actor_denominators(fixture):
    from PIL import Image
    bank,pop,caches,images=fixture
    for rel in {s.image_reference.relative_path for s in bank.samples}:
        Image.new("RGB",(1600,900),(10,20,30)).save(images/rel)
    result=w.inventory(bank,pop,images)
    assert (result["actor_frames"],result["unique_images"],result["videos"])==(12,6,2)
    assert result["splits"]["validation"]=={"actor_frames":6,"unique_images":3,"actors":2,"videos":1}
    assert not result["observation_inference_executed"]
    first=bank.samples[0].image_reference.relative_path
    assert result["images"][first]["sha256"]==w.sha(images/first)
    Image.new("RGB",(900,1600)).save(images/first)
    with pytest.raises(ValueError,match="dimensions"):
        w.inventory(bank,pop,images)


def test_evaluation_projection_strict_frame_actor_split_and_axes(fixture,tmp_path):
    import pickle
    bank,_,_,_=fixture
    s=bank.samples[3]  # validation, actor1, frame5
    raw=tmp_path/"raw";directory=raw/"sequenceFiles/validation";directory.mkdir(parents=True)
    rng=np.random.default_rng(42)
    world=rng.normal(size=(2,4,24,3))*.1;world[:,:,:,2]+=3
    cameras=np.repeat(np.eye(4)[None],4,axis=0);cameras[:,:3,3]=(.2,.3,.4)
    K=np.array([[1000,0,800],[0,1100,450],[0,0,1.]])
    payload={"img_frame_ids":[101,0,5,10],"jointPositions":world.reshape(2,4,72),"cam_poses":cameras,"cam_intrinsics":K}
    path=directory/"fixture_validation.pkl"
    with path.open("wb") as f:pickle.dump(payload,f)
    uv,J,g=w.evaluation_reference(raw,s,{})
    camera_points=world[1,2]+(.2,.3,.4)
    expected_uv=camera_points@K.T;expected_uv=expected_uv[:,:2]/expected_uv[:,2:3]
    assert np.allclose(uv,expected_uv)
    assert np.allclose(g,[0,0,-1])
    order=(0,1,4,7,2,5,8,3,9,12,15,16,18,20,17,19,21)
    expected=camera_points[list(order)][:,[0,2,1]]*np.array([1,1,-1])
    assert np.allclose(J,expected-expected[0])
    payload["img_frame_ids"]=[5,0,5,10]
    with path.open("wb") as f:pickle.dump(payload,f)
    with pytest.raises(ValueError,match="not unique"):
        w.evaluation_reference(raw,s,{})
