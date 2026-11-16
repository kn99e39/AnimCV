"""Supplementary W144 source-texture/representation diagnostics, no candidate.

Protocol is written and hashed before running: 1-frame pyramidal LK, 21px
window, 3 levels, 30 iterations, epsilon .01, min eigenvalue 1e-4, forward/back
error <=1px. This tracks local texture, not independently labelled anatomy.
No thresholds from this supplement are used to select or repair an Action.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
import w144_harness as H
from app import experimental_scene_anchored_trace as T
from motion.torso_frame import torso_frames
from framepose.crops import crop_box,geometry_in_crop
from w144_analyze import JOINTS


def run():
    import cv2
    H.verify()
    protocol={'script_sha256':H.sha(__file__),'window':[21,21],'levels':3,'fb_max_px':1,
              'min_eigen':1e-4,'purpose':'source-texture consistency; neither anatomy GT nor corrective constraint',
              'disclosure':'Supplement added after primary baseline metrics and contact-sheet inspection; not part of prospective candidate acceptance; no candidate.'}
    H.write(H.OUT/'source_control_protocol.json',protocol)
    z=np.load(H.OBS);xy=z['input_2d'][79:319,:,:2]*[640,480];valid=z['input_valid'][79:319]
    cap=cv2.VideoCapture(str(H.CLIP));ims=[]
    for i in range(319):
        ok,im=cap.read();assert ok
        if i>=79:ims.append(cv2.cvtColor(im,cv2.COLOR_BGR2GRAY))
    cap.release()
    params=dict(winSize=(21,21),maxLevel=3,criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,30,.01),minEigThreshold=1e-4)
    result={'protocol':protocol,'joints':{},'edges':[],'representation':{}}
    all_res=np.full((239,17),np.nan);all_flow=np.full((239,17),np.nan)
    for i in range(239):
        p=xy[i].astype(np.float32).reshape(-1,1,2)
        q,st,_=cv2.calcOpticalFlowPyrLK(ims[i],ims[i+1],p,None,**params)
        back,bs,_=cv2.calcOpticalFlowPyrLK(ims[i+1],ims[i],q,None,**params)
        fb=np.linalg.norm(back[:,0]-p[:,0],axis=1)
        ok=st[:,0].astype(bool)&bs[:,0].astype(bool)&(fb<=1)&valid[i]&valid[i+1]
        residual=np.linalg.norm(q[:,0]-xy[i+1],axis=1);flow=np.linalg.norm(q[:,0]-p[:,0],axis=1)
        all_res[i,ok]=residual[ok];all_flow[i,ok]=flow[ok]
        result['edges'].append({'to_action':i+2,'texture_ok':ok.tolist(),'fb_error_px':fb.tolist(),
            'landmark_texture_residual_px':[float(v) if good else None for v,good in zip(residual,ok)],
            'texture_flow_px':[float(v) if good else None for v,good in zip(flow,ok)]})
    for k,n in enumerate(JOINTS):result['joints'][n]={'residual_px':T.summary(all_res[:,k]),'flow_px':T.summary(all_flow[:,k])}
    # Verify normalized/cropped observation identity without loading a model.
    geo=np.stack([geometry_in_crop(x,v,(640,480),crop_box(x,v,(640,480))) for x,v in zip(z['input_2d'][79:319],valid)])
    result['representation']['geometry_byte_equal']=geo.tobytes()==z['geometry'][79:319].tobytes()
    result['representation']['crop']=[crop_box(x,v,(640,480)).to_dict() for x,v in zip(z['input_2d'][79:319],valid)]
    result['representation']['geometry_step_l2']=T.summary(np.linalg.norm(np.diff(geo,axis=0).reshape(239,-1),axis=1))
    j=z['predictions'][79:319].astype(float)
    result['representation']['pose_pelvis_absolute']=T.summary(np.linalg.norm(j[:,0],axis=1))
    for name,a,b in [('hip_side',1,4),('shoulder_side',11,14),('trunk',0,8)]:
        d=j[:,b]-j[:,a]
        result['representation'][name]={'length_m':T.summary(np.linalg.norm(d,axis=1)),
            'step_components_m':{s:T.summary(np.abs(np.diff(d[:,k]))) for k,s in enumerate(('X_right','Y_depth','Z_up'))}}
    result['focus']={}
    for dest in sorted(set([172,177,200]+H.load(H.OUT/'analysis.json')['selection']['top_shoulder_destinations'])):
        i=dest-1;pt=j[i-1];ct=j[i];Rtp,Rpp=torso_frames(pt);Rtc,Rpc=torso_frames(ct)
        r={'source_pair':[dest+77,dest+78], 'source_texture':result['edges'][dest-2],
           'input_valid_changes':np.flatnonzero(valid[i]!=valid[i-1]).tolist(),
           'geometry_step_l2':float(np.linalg.norm(geo[i]-geo[i-1])),
           'pose_trunk_previous':(pt[8]-pt[0]).tolist(),'pose_trunk_current':(ct[8]-ct[0]).tolist(),
           'pose_hip_side_previous':(pt[4]-pt[1]).tolist(),'pose_hip_side_current':(ct[4]-ct[1]).tolist(),
           'pose_shoulder_side_previous':(pt[14]-pt[11]).tolist(),'pose_shoulder_side_current':(ct[14]-ct[11]).tolist()}
        result['focus'][str(dest)]=r
    # Clarify the original synthetic label: it injected object Z and local spine
    # Y changes, NOT the same rotation twice. Against parent-only expected state,
    # the second local transform is a known injected parent/articulation defect.
    tr=H.load(H.OUT/'blender_trace.json');p=tr['synthetic']['parent_defect'];clean=tr['synthetic']['root_rotation']
    err=np.linalg.norm(np.array(p['world']['upperarm_l']['head'])-clean['world']['upperarm_l']['head'])
    result['synthetic_label_clarification']={'initial_analysis_note':'The phrase duplicated application in analysis.json is too specific.',
        'actual_injection':'20 degree world Z object change plus 20 degree local Y spine change; compare with known parent-only oracle',
        'extra_parent_articulation_error_m':float(err),
        'detected_by_oracle':bool(err>1e-5),'no_evidence_of_this_defect_in_S1':True}
    H.write(H.OUT/'source_controls.json',result)
    print('SOURCE_CONTROLS',result['representation']['geometry_byte_equal'],result['synthetic_label_clarification'])
    for dest,r in result['focus'].items():
        row=r['source_texture'];print(dest,{n:{'fb_ok':row['texture_ok'][k],'texture_flow_px':row['texture_flow_px'][k],
            'detector_texture_residual_px':row['landmark_texture_residual_px'][k]} for k,n in enumerate(JOINTS) if n in ('left_shoulder','right_shoulder','pelvis')})


if __name__=='__main__':run()
