"""W145 focused contracts, synthetic truth never used as real S1 input."""
from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import w145_body_frame_core as B
from motion.torso_frame import torso_frames,yaw_frame,quaternion_to_matrix,matrix_to_quaternion_xyzw
from app.experimental_scene_anchored_trace import rotation_angle


def pose(yaw=0,twist=0,bend=0,width=.2):
    j=np.zeros((17,3));j[1]=[-width/2,0,0];j[4]=[width/2,0,0]
    j[8]=[0,0,.5];j[11]=[-.2,0,.5];j[14]=[.2,0,.5]
    Rx=np.array([[1,0,0],[0,np.cos(bend),-np.sin(bend)],[0,np.sin(bend),np.cos(bend)]])
    j[[8,11,14]]=j[[8,11,14]]@(Rx@yaw_frame(twist)).T
    return j@yaw_frame(yaw).T


@pytest.mark.parametrize('side',['pelvis','thorax'])
@pytest.mark.parametrize('yaw',[0,.8,2.5])
def test_analytic_jacobian_matches_finite_difference(side,yaw):
    j=pose(yaw,.3,.1);J=B.jacobian(j,side);R=torso_frames(j)[0 if side=='thorax' else 1]
    eps=1e-6
    for col,(k,a) in enumerate((k,a) for k in B.AXIS_JOINTS for a in range(3)):
        jp=j.copy();jm=j.copy();jp[k,a]+=eps;jm[k,a]-=eps
        dR=(torso_frames(jp)[0 if side=='thorax' else 1]-torso_frames(jm)[0 if side=='thorax' else 1])/(2*eps)
        m=R.T@dR;w=np.array([m[2,1]-m[1,2],m[0,2]-m[2,0],m[1,0]-m[0,1]])/2
        np.testing.assert_allclose(J[:,col],w,atol=2e-8)


def test_scale_conditioning_and_handedness():
    j=pose(.4);dims=B.dimensions(j)
    assert dims['pelvis']['projected_side_m']==pytest.approx(.2)
    for R in torso_frames(j):
        np.testing.assert_allclose(R.T@R,np.eye(3),atol=1e-14);assert np.linalg.det(R)==pytest.approx(1)
    np.testing.assert_allclose(B.jacobian(j*2,'pelvis'),B.jacobian(j,'pelvis')/2)


def data(js):
    xy=np.asarray(js)[:,:,[0,2]]*100+[320,240]
    v=np.ones((len(js),17),bool);res=np.zeros((len(js)-1,17));tv=np.ones_like(res,bool)
    return xy,v,res,tv


@pytest.mark.parametrize('degrees',[30,90,180])
def test_real_rapid_turn_is_not_velocity_capped(degrees):
    # Start obliquely so the declared image evidence observes even the 30deg
    # turn; a near-frontal 30deg turn is a separate ambiguity control below.
    initial=np.pi/4
    js=np.array([pose(initial),pose(initial+np.radians(degrees)),pose(initial+np.radians(degrees))]);q=B.qualify(js,*data(js))
    assert q[1]['pelvis']['status']=='QUALIFIED_MEASUREMENT'
    assert rotation_angle(q[0]['pelvis']['R'],q[1]['pelvis']['R'])==pytest.approx(degrees)


def test_real_turn_can_be_quiet_in_projection_and_must_remain_uncertain():
    js=np.array([pose(),pose(np.pi/6)]);q=B.qualify(js,*data(js))
    assert 'DEPTH_TURN_UNCONFIRMED' in q[1]['pelvis']['reasons']
    assert q[1]['pelvis']['R'] is None


def test_false_jump_quiet_source_is_abstained_not_held():
    js=np.array([pose(),pose(np.pi/2),pose()]);xy,v,res,tv=data([pose()]*3)
    q=B.qualify(js,xy,v,res,tv)
    assert 'DEPTH_TURN_UNCONFIRMED' in q[1]['pelvis']['reasons']
    assert q[1]['pelvis']['R'] is None


def test_weak_geometry_and_missing_observation_are_explicit():
    js=np.array([pose(width=.001)]*3);xy,v,res,tv=data(js);v[1,1]=False
    q=B.qualify(js,xy,v,res,tv)
    assert 'WEAK_GEOMETRY' in q[0]['pelvis']['reasons']
    assert 'MISSING_OBSERVATION' in q[1]['pelvis']['reasons']


def test_texture_disagreement_and_unavailable_are_not_repaired():
    js=np.array([pose()]*3);xy,v,res,tv=data(js);res[0,1]=5;tv[1,4]=False
    q=B.qualify(js,xy,v,res,tv)
    assert 'POINT_TEXTURE_DISAGREEMENT' in q[1]['pelvis']['reasons']
    assert 'TEXTURE_UNAVAILABLE' in q[2]['pelvis']['reasons']


def test_derived_2d_contract_missing_and_confidence():
    x=np.ones((17,3),np.float32);v=np.ones(17,bool);x[1]=[.3,.4,.8];x[4]=[.5,.6,.6]
    a,b=B.independent_2d_midpoints(x,v);np.testing.assert_allclose(a[0],[.4,.5,.6])
    v[1]=False;a,b=B.independent_2d_midpoints(x,v)
    assert not b[0] and not b[7];np.testing.assert_array_equal(a[0],0)


def test_torso_twist_separates_but_shared_trunk_bend_is_not_pelvis_truth():
    Rt,Rp=torso_frames(pose(twist=.5));assert rotation_angle(np.eye(3),Rp)<1e-10
    assert rotation_angle(np.eye(3),Rt)==pytest.approx(np.degrees(.5))
    Rt,Rp=torso_frames(pose(bend=.5))
    # Frozen semantic assumption: shared up tilts inferred pelvis even though
    # synthetic hip positions and pelvis GT are unchanged. Do not hide failure.
    assert rotation_angle(np.eye(3),Rp)==pytest.approx(np.degrees(.5))


def test_quaternion_sign_and_axis_convention():
    R=yaw_frame(.8);q=np.array(matrix_to_quaternion_xyzw(R))
    np.testing.assert_allclose(quaternion_to_matrix(-q),R,atol=1e-14)
    assert rotation_angle(R,R)==pytest.approx(0,abs=1e-12)
