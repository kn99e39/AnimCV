"""Strict W102 evaluation-only reference for the bank's existing zero-based annotation/RGB row contract.

three_dpw_adapter.load_3dpw_ground_truth enumerates annotation rows; its exported frame_index is NOT img_frame_ids.
Bind a row with both original benchmark 2D reconstruction and bank 3D parity. Never conditionally fall back
between two index systems. Runtime M1/S0 input construction does not import or call this reference path.
"""
from __future__ import annotations
import pickle
from pathlib import Path
import numpy as np
from pose.three_dpw_adapter import _canonical_2d
from pose.pose_lifter import H36M_NAMES
from run_external_hmr2_baseline import smpl24_to_canonical17
from run_pelvis_torso_ownership import CV_TO_CANON


def row_identity(sample,data):
    video=sample.sequence_id.split(":")[1];actor=int(sample.sequence_id.split(":")[2][5:]);k=sample.frame_index
    n=len(data["cam_poses"])
    if data["sequence"]!=video or k<0 or k>=n or actor<0 or actor>=len(data["jointPositions"]):
        raise RuntimeError("STOP: raw annotation row identity mismatch")
    if (len(data["img_frame_ids"])!=n or len(data["jointPositions"][actor])!=n or
        len(data["poses2d"][actor])!=n or len(data["campose_valid"][actor])!=n):
        raise RuntimeError("STOP: raw annotation arrays do not share one row identity")
    return actor,k


def evaluation_reference(raw_root,sample,cache):
    video=sample.sequence_id.split(":")[1]
    if video not in cache:
        with (Path(raw_root)/"sequenceFiles"/sample.split/(video+".pkl")).open("rb") as stream:
            cache[video]=pickle.load(stream,encoding="latin1")
    d=cache[video];actor,k=row_identity(sample,d)
    X=np.asarray(d["jointPositions"][actor][k],float).reshape(24,3);T=np.asarray(d["cam_poses"][k],float)
    Xc=X@T[:3,:3].T+T[:3,3];uv=Xc@np.asarray(d["cam_intrinsics"],float).T
    uv=uv[:,:2]/uv[:,2:3]
    gravity=CV_TO_CANON@T[:3,:3]@np.array([0.,1.,0.]);gravity/=np.linalg.norm(gravity)
    return uv,smpl24_to_canonical17(Xc),gravity


def raw_reference_available(sample,cache,uv,joints,gravity):
    data=cache[sample.sequence_id.split(":")[1]];actor,k=row_identity(sample,data)
    return bool(data["campose_valid"][actor][k] and np.isfinite(uv).all()
                and np.isfinite(joints).all() and np.isfinite(gravity).all())


def verify_benchmark_row(bank,p,sample,cache):
    data=cache[sample.sequence_id.split(":")[1]];actor,k=row_identity(sample,data)
    points=_canonical_2d(np.asarray(data["poses2d"][actor][k],float)) if data["campose_valid"][actor][k] else {}
    x=[];valid=[];target_mask=[]
    for name in H36M_NAMES:
        point=points.get("neck" if name=="thorax" else name)
        x.append([point.x/sample.image_size[0],point.y/sample.image_size[1],point.confidence] if point else [0.,0.,0.])
        valid.append(bool(point and point.confidence>0))
        target_mask.append(bool(point and point.visible and data["campose_valid"][actor][k]))
    if (not np.array_equal(np.asarray(x,np.float32),bank.arrays["input_2d"][p]) or
        not np.array_equal(valid,bank.arrays["input_valid"][p]) or
        not np.array_equal(target_mask,bank.arrays["target_valid"][p])):
        raise RuntimeError("STOP: raw benchmark observation/mask does not match bank row")
