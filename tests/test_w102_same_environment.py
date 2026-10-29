"""W102 semantic gates protect controlled inference without reinterpreting historical byte identity."""
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import json

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from w102_numerical_audit import compare_predictions
from test_w101_reference_estimator import fixture, landmarks
import w101_reference_estimator as w101
from w102_evaluate import coverage_report, paired_temporal, representation_report, failure_overlap,raw_reference_available


def body():
    return np.array([[0,0,0],[-.1,0,0],[-.1,.12,-.4],[-.1,-.1,-.7],[.1,0,0],[.1,-.1,-.4],[.1,.1,-.75],
                     [0,0,.25],[0,0,.5],[0,0,.55],[0,0,.75],[-.2,0,.5],[-.35,.15,.3],[-.5,.2,.15],
                     [.2,0,.5],[.35,-.1,.3],[.5,-.15,.15]],np.float32)


def test_same_environment_exact_and_historical_tiny_difference_are_separate():
    a=np.stack([body(),body()]);b=a+np.float32(1e-6)
    exact=compare_predictions(a,a,a)
    tiny=compare_predictions(a,b,a)
    assert exact["pass"] and exact["byte_equal"]
    assert tiny["pass"] and not tiny["byte_equal"]
    assert tiny["joint_displacement_mm"]["max"]<.1
    assert tiny["outside_boundary_changes"]==0


def test_material_geometry_change_blocks_controlled_comparison():
    a=body()[None];b=a.copy();b[0,13,0]+=.01
    result=compare_predictions(a,b,a)
    assert not result["pass"] and result["joint_displacement_mm"]["max"]>1


def test_root_relative_interpretation_has_its_own_numerical_budget():
    a=body()[None];b=a.copy();b[:,:,0]-=8e-5;b[:,0,0]=8e-5
    result=compare_predictions(a,b,a)
    assert result["joint_displacement_mm"]["max"]<.1
    assert result["root_relative_displacement_mm"]["max"]>.1
    assert not result["pass"]


def test_near_depth_zero_decision_is_reported_without_false_material_failure():
    g=body()[None];a=g.copy();b=g.copy()
    a[0,2,1]=5e-8;b[0,2,1]=-5e-8
    result=compare_predictions(a,b,g)
    decision=result["discrete"]["depth_sign_thigh_l"]
    assert decision["changed"]==1 and decision["outside_boundary"]==0
    assert decision["examples"][0]["near_boundary"]
    assert result["pass"]


def test_near_straight_knee_is_not_assigned_a_fake_zero_plane_error():
    a=body()[None];a[0,3]=2*a[0,2]-a[0,1]
    result=compare_predictions(a,a,a)
    assert result["degenerate"]["knee_not_well_conditioned_l"]==1
    assert "knee_plane_l" not in result["continuous"]


def test_severe_boundary_change_is_visible_but_not_an_uncontrolled_source_effect():
    gt=body()[None]
    scale=float(np.linalg.norm(gt-gt[:,:1],axis=2).mean()*1000)
    a=(gt*(1+199.99/scale)).astype(np.float32)
    b=(gt*(1+200.01/scale)).astype(np.float32)
    result=compare_predictions(a,b,gt)
    decision=result["discrete"]["severe_200mm"]
    assert decision["changed"]==1 and decision["outside_boundary"]==0
    assert result["pass"]


def test_nonfinite_prediction_is_a_real_gate_failure():
    a=body()[None];b=a.copy();b[0,2,1]=np.nan
    with pytest.raises(ValueError,match="Nonfinite"):
        compare_predictions(a,b,a)


def test_temporal_audit_does_not_bridge_missing_original_neighbor():
    a=np.stack([body(),body()])
    samples=[SimpleNamespace(sample_id="s#0",timestamp=0.,neighbors={"next":"s#5"}),
             SimpleNamespace(sample_id="s#10",timestamp=1/3,neighbors={"next":None})]
    result=compare_predictions(a,a,a,samples)
    assert result["temporal_steps"]==0


def test_coverage_keeps_image_actor_and_gap_denominators_distinct(fixture):
    bank,pop,caches,images=fixture
    for split in ("validation","test"):
        path=caches/f"fixture_{split}.npz";z=dict(np.load(path))
        z["selected_box"]=np.array([[1,2,100,200],[np.nan]*4,[501,2,600,200.]])
        z["landmarks"][2]=landmarks(500)  # unique actor change after a missing sample
        np.savez_compressed(path,**z)
        meta_path=path.with_suffix(".json");meta=json.loads(meta_path.read_text())
        meta["npz_sha256"]=w101.sha(path);meta_path.write_text(json.dumps(meta))
    obs,side=w101.build_observations(bank,pop,caches,images)
    report=coverage_report(bank,obs,side,caches)
    for split in ("validation","test"):
        image=report["splits"][split];actors=report["original_actor_frame_accounting"][split]
        assert image["unique_images"]==3 and actors["actor_frames"]==6
        assert image["detector_selected_images"]==2 and image["any_detector_candidate_images"]==3
        assert image["association"]=={"unique_valid_association":2,"missing_subject":1}
        assert actors["identity"]["same_person"]==2 and actors["identity"]["other_person"]==2
        video=image["per_video"][f"fixture_{split}"]
        assert video["anchor_stable_images"]==1
        assert video["identity_transitions"]==[]  # the gap prevents an invented transition
        assert video["adjacent_detected_box_iou"]["n"]==0
        assert video["reacquisition_iou_separate"]["n"]==1
        assert video["missing_detection_runs"]==[{"start_frame":5,"end_frame":5,"missing_samples":1}]


def test_common_support_temporal_does_not_bridge_identity_gap():
    def row(frame,ident):
        return {"sample_id":f"3dpw:a:actor0#{frame:06d}","sid":"3dpw:a:actor0",
                "next":f"3dpw:a:actor0#{frame+5:06d}","identity":ident,"tracked":True,
                "J":{"B":body(),"M":body()},"G":body()}
    rows=[row(0,"same_person"),row(5,"other_person"),row(10,"same_person")]
    assert paired_temporal(rows)=={}
    rows[1]["identity"]="same_person"
    report=paired_temporal(rows)
    for metric in report.values():
        assert metric["B_events"]==metric["M_events"]==0
        assert metric["paired_event_rate_difference"]["videos"]==1
        assert metric["paired_event_rate_difference"]["ci95"] is None


def test_persistent_failure_counts_keep_metric_specific_common_support():
    rows=[{"m":{"B":{"severe":1,"knee_wrong_l":1},"M":{"severe":1}}},
          {"m":{"B":{"severe":0,"knee_wrong_l":1},"M":{"severe":1,"knee_wrong_l":0}}}]
    report=failure_overlap(rows)
    assert report["severe"]=={"common_support":2,"both_fail":1,"B_only":0,"M_only":1,"neither":0}
    assert report["knee_wrong_l"]["common_support"]==1 and report["knee_wrong_l"]["B_only"]==1


def test_raw_gt_availability_is_independent_of_benchmark_joint_visibility():
    sample=SimpleNamespace(sequence_id="3dpw:v:actor1",frame_index=5)
    cache={"v":{"sequence":"v","img_frame_ids":np.arange(11)*2+100,"campose_valid":[[True]*11,[True]*11],
                "cam_poses":np.repeat(np.eye(4)[None],11,0),"poses2d":np.zeros((2,11,3,18)),"jointPositions":np.zeros((2,11,72))}}
    assert raw_reference_available(sample,cache,np.zeros((24,2)),body(),np.array([0,0,1.]))
    cache["v"]["campose_valid"][1][5]=False
    assert not raw_reference_available(sample,cache,np.zeros((24,2)),body(),np.array([0,0,1.]))
    sample.frame_index=11
    with pytest.raises(RuntimeError,match="identity"):
        raw_reference_available(sample,cache,np.zeros((24,2)),body(),np.array([0,0,1.]))


def test_raw_projection_uses_explicit_bank_row_not_capture_metadata_id(tmp_path):
    import pickle
    from w102_reference import evaluation_reference
    sample=SimpleNamespace(sequence_id="3dpw:v:actor1",frame_index=5,split="validation")
    world=np.zeros((2,11,24,3));world[:,:,:,2]=10;world[1,5,:,0]=2;world[1,5,1,0]=2.2
    data={"sequence":"v","img_frame_ids":np.arange(11)*2+100,"campose_valid":[[True]*11,[True]*11],
          "cam_poses":np.repeat(np.eye(4)[None],11,0),"poses2d":np.zeros((2,11,3,18)),
          "jointPositions":world.reshape(2,11,72),"cam_intrinsics":np.array([[1000,0,800],[0,1000,450],[0,0,1.]])}
    path=tmp_path/"sequenceFiles/validation";path.mkdir(parents=True)
    (path/"v.pkl").write_bytes(pickle.dumps(data));cache={}
    uv,joints,gravity=evaluation_reference(tmp_path,sample,cache)
    assert uv[0]==pytest.approx([1000,450]) and uv[1]==pytest.approx([1020,450])
    assert joints[1]==pytest.approx([.2,0,0]) and gravity==pytest.approx([0,0,-1])


def test_representation_uses_same_person_and_identical_valid_joints(fixture):
    bank,pop,caches,images=fixture;obs,side=w101.build_observations(bank,pop,caches,images)
    result=representation_report(bank,obs,side,"validation")
    assert result["same_person_actor_frames"]==2
    assert result["joint_denominator"]==34
    assert result["limb_source_distance_px"]["quantiles"]["max"]==pytest.approx(0.)


def test_rig_lower_body_change_preserves_common_parent_and_finite_chain():
    from w102_rig_review import context,pose_export,arm_export
    root=Path(__file__).resolve().parents[1]
    ctx=context(root/"examples/e2e_demo/baserig_blender_rest.json",root/"examples/e2e_demo/framepose_fk_calibration.json",
                root/"examples/e2e_demo/framepose_ik_chains.json")
    a=body();b=a.copy();b[2,1]+=.1;b[3,1]+=.05
    pa,pb=(pose_export(j,np.eye(3),np.eye(3),ctx) for j in (a,b))
    assert pa is not None and pb is not None
    assert pa["object"]==pb["object"] and pa["bones"]["spine_01"]==pb["bones"]["spine_01"]
    assert pa["bones"]["thigh_l"]!=pb["bones"]["thigh_l"]
    # Existing IK transfers reach fraction to the target bone ratio, so exact source directions are not its contract.
    # Rotating source and parent together must preserve the emitted local leg channels.
    rotation=np.array([[0,-1,0],[1,0,0],[0,0,1.]])
    rotated=pose_export(a@rotation.T,rotation,rotation,ctx)
    for bone in pa["bones"]:
        assert np.allclose(pa["bones"][bone],rotated["bones"][bone],atol=1e-12,rtol=0)
    assert pose_export(None,np.eye(3),np.eye(3),ctx) is None
    arm=arm_export([.15,.08,-.2],[.3,.1,-.4],np.eye(3),np.eye(3),ctx)
    assert arm is not None and arm["object_rotation_armature_xyzw"]==pa["object"]
    assert arm["extra_bone_locals_xyzw"]["spine_01"]==pa["bones"]["spine_01"]
    assert arm_export([np.nan]*3,[0,0,0],np.eye(3),np.eye(3),ctx) is None


def test_video_side_panel_and_missing_subject_have_no_invented_pose():
    from w102_video_diagnostic import render_frame
    frame=SimpleNamespace(image=np.zeros((480,640,3),np.uint8),index=0,timestamp=0.)
    xx=np.zeros((17,3));vv=np.zeros(17,bool)
    a=render_frame(frame,xx,vv,body(),False)
    b=render_frame(frame,xx,vv,body()*1000,False)
    assert a.shape==(480,1280,3) and np.array_equal(a,b)
