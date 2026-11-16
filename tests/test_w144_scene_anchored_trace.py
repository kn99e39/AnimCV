"""W144 oracle_geometry contract controls; never proof of authentic S1 accuracy."""
import copy
import numpy as np
import pytest
from app import experimental_scene_anchored_trace as T
from motion.torso_frame import yaw_frame, quaternion_to_matrix, matrix_to_quaternion_xyzw


def transform(R=None,t=None):
    m=np.eye(4)
    if R is not None:m[:3,:3]=R
    if t is not None:m[:3,3]=t
    return m


@pytest.mark.parametrize('source,action',[(79,1),(249,171),(250,172),(277,199),(278,200),(318,240)])
def test_timestamp_mapping(source,action):
    assert T.action_frame(source)==action
    assert (action-1)/30==(source-79)/30


@pytest.mark.parametrize('bad',[78,319,79.5])
def test_frame_refusal(bad):
    with pytest.raises(ValueError):T.action_frame(bad)


def test_camera_inverse_and_stationary_world_under_moving_camera():
    world=np.array([[1,2,3],[-1,4,2]])
    for yaw,c in [(0,[0,0,0]),(.4,[.2,-.3,.1])]:
        R=yaw_frame(yaw)
        camera=T.world_to_camera(world,R,c)
        np.testing.assert_allclose(T.camera_to_world(camera,R,c),world,atol=1e-14)
    # Wrong camera-to-world / world-to-camera semantics are distinguishable.
    assert not np.allclose(T.camera_to_world(camera,R.T,c),world)


def test_no_metric_position_from_orientation_only():
    with pytest.raises(ValueError,match='unavailable'):
        T.camera_to_world([[0,1,0]],np.eye(3),None)


def test_fk_parent_rest_basis_and_object_scale():
    rest={'a':{'parent':None,'matrix_local':transform(t=[0,0,100])},
          'b':{'parent':'a','matrix_local':transform(t=[10,0,100])}}
    obj=np.diag([.01,.01,.01,1])
    basis={'a':transform(yaw_frame(np.pi/2))}
    arm,world=T.fk(rest,basis,obj)
    np.testing.assert_allclose(world['b'][:3,3],[0,.1,1],atol=1e-14)
    np.testing.assert_allclose(arm['b'][:3,:3],yaw_frame(np.pi/2),atol=1e-14)
    # Translation of the object and rotation of an elevated anatomical pivot
    # are not interchangeable quantities.
    changed=obj.copy();changed[:3,:3]=yaw_frame(.3)@obj[:3,:3]
    np.testing.assert_allclose(changed[:3,3],obj[:3,3])


def test_counterfactual_single_factor_no_mutation_and_nonadditivity():
    a={'object':transform(t=[1,2,3]),'basis':{'a':np.eye(4),'b':np.eye(4)}}
    b=copy.deepcopy(a);b['object']=transform(yaw_frame(.5),[4,5,6]);b['basis']['a']=transform(yaw_frame(.8))
    c=T.replace_factor(b,a,'object_rotation')
    np.testing.assert_array_equal(c['object'][:3,3],[4,5,6])
    np.testing.assert_array_equal(c['basis']['a'],b['basis']['a'])
    np.testing.assert_array_equal(b['object'][:3,:3],yaw_frame(.5))
    p=np.array([1,0,0.]);R=yaw_frame(.5);Q=yaw_frame(.8)
    interaction=R@Q@p-R@p-Q@p+p
    assert np.linalg.norm(interaction)>.1


def test_confidence_and_missing_do_not_become_stationary():
    report,d=T.motion([[0,0],[10,0],[11,0],[12,0]],[True,False,True,True])
    assert np.isnan(d[:2]).all() and d[2]==1
    assert report['step']['n']==1
    with pytest.raises(ValueError,match='do not fabricate'):
        T.require_intermediate({},'root_depth')


def test_rotation_sign_and_retarget_axis_roundtrip():
    A=yaw_frame(.7);R=yaw_frame(-1.2)
    v=np.array([1,2,3.])
    np.testing.assert_allclose((A@R@A.T)@(A@v),A@(R@v),atol=1e-14)
    q=np.array(matrix_to_quaternion_xyzw(R))
    np.testing.assert_allclose(quaternion_to_matrix(-q),R,atol=1e-14)
    assert T.rotation_angle(R,R)==pytest.approx(0,abs=1e-12)
    assert T.rotation_angle(np.eye(3),R)==pytest.approx(np.degrees(1.2))


@pytest.mark.parametrize('kind',['stable','camera','root','torso','upstream','parent_defect'])
def test_known_synthetic_mechanisms(kind):
    rest={'torso':{'parent':None,'matrix_local':transform(t=[0,0,1])},
          'shoulder':{'parent':'torso','matrix_local':transform(t=[.2,0,1.5])}}
    base={'object':np.eye(4),'basis':{'torso':np.eye(4),'shoulder':np.eye(4)}}
    cur=copy.deepcopy(base)
    if kind=='root':cur['object'][0,3]=.1
    if kind in ('torso','upstream','parent_defect'):cur['basis']['torso']=transform(yaw_frame(.3))
    if kind=='parent_defect':cur['object']=transform(yaw_frame(.3))
    _,bw=T.fk(rest,base['basis'],base['object']);_,cw=T.fk(rest,cur['basis'],cur['object'])
    if kind in ('stable','camera'):
        np.testing.assert_array_equal(cw['shoulder'],bw['shoulder'])
    elif kind=='root':
        np.testing.assert_allclose(cw['shoulder'][:3,3]-bw['shoulder'][:3,3],[.1,0,0])
    else:
        held=T.replace_factor(cur,base,'bone:torso')
        _,hw=T.fk(rest,held['basis'],held['object'])
        if kind!='parent_defect':np.testing.assert_allclose(hw['shoulder'],bw['shoulder'])
        else:assert not np.allclose(hw['shoulder'],bw['shoulder'])
