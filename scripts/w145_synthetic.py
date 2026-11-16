"""W145 exact pose-token controls. No rendered/perception evidence is claimed."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
import w145_harness as W
import w144_harness as H
import w145_body_frame_core as B
from motion.torso_frame import yaw_frame,torso_frames
from app import experimental_scene_anchored_trace as T


def pose(yaw=0,twist=0,bend=0):
    j=np.zeros((17,3));j[1]=[-.1,0,0];j[4]=[.1,0,0]
    j[2]=[-.1,0,-.4];j[3]=[-.1,0,-.8];j[5]=[.1,0,-.4];j[6]=[.1,0,-.8]
    j[7]=[0,0,.25];j[8]=[0,0,.5];j[9]=[0,0,.6];j[10]=[0,0,.75]
    j[11]=[-.2,0,.5];j[14]=[.2,0,.5];j[12]=[-.4,0,.35];j[13]=[-.5,0,.2]
    j[15]=[.4,0,.35];j[16]=[.5,0,.2]
    Rx=np.array([[1,0,0],[0,np.cos(bend),-np.sin(bend)],[0,np.sin(bend),np.cos(bend)]])
    j[7:]=j[7:]@(Rx@yaw_frame(twist)).T
    return j@yaw_frame(yaw).T


def run():
    d=W.verify();n=d['synthetic']['frames_each'];time=np.arange(n)/(n-1)
    cases={}
    for name in d['synthetic']['cases']:
        truth=[];pelvis=[];thorax=[]
        for i,t in enumerate(time):
            yaw=0.;twist=0.;bend=0.
            if name=='gradual_turn_180':yaw=np.pi*t
            if name=='rapid_turn_90':yaw=np.pi/2 if i>=45 else 0
            if name=='pelvis_fixed_torso_twist':twist=np.radians(30)*np.sin(2*np.pi*t)
            if name=='pelvis_fixed_torso_bend':bend=np.radians(30)*np.sin(2*np.pi*t)
            p=pose(yaw,twist,bend)
            if name=='articulated_arms':
                p[12:14,2]+=.3*np.sin(4*np.pi*t);p[15:17,2]+=.3*np.cos(4*np.pi*t)
            Rx=np.array([[1,0,0],[0,np.cos(bend),-np.sin(bend)],[0,np.sin(bend),np.cos(bend)]])
            truth.append(p);pelvis.append(yaw_frame(yaw));thorax.append(yaw_frame(yaw)@Rx@yaw_frame(twist))
        truth=np.array(truth);j=truth.copy()
        xy=truth[:,:,[0,2]]*[100,-100]+[320,240]
        valid=np.ones((n,17),bool);res=np.zeros((n-1,17));tv=np.ones((n-1,17),bool)
        affected=[]
        if name=='false_3d_flip_1_and_10_frames':
            affected=[20,*range(40,50)]
            for i in affected:j[i]=pose(np.pi/2)
        if name=='observation_jump':
            affected=[30]
            xy[30,14,0]+=37;res[29,14]=37;res[30,14]=37
            j[30]=pose(np.pi/2)
        if name=='missing_and_LR_swap':
            valid[20:30,[1,4]]=False
            for i in range(40,45):
                xy[i,[1,4]]=xy[i,[4,1]];j[i,[1,4]]=j[i,[4,1]]
            res[39,1:5]=20;res[44,1:5]=20;affected=[*range(20,30),*range(40,45)]
        if name=='depth_ambiguity_degeneracy':
            affected=list(range(20,50))
            for i in affected:
                if i<35:
                    j[i,1]=[0,-.0005*(-1)**i,0];j[i,4]=[0,.0005*(-1)**i,0]
                else:
                    # Bilateral side nearly parallel to trunk: fails the
                    # historical 20deg validity floor.
                    j[i,1]=[-.01,0,-.1];j[i,4]=[.01,0,.1]
        params={k:d['qualifier'][k] for k in ('sigma_m','max_condition_deg','texture_max_px','unsupported_turn_deg','quiet_source_px')}
        q=B.qualify(j,xy,valid,res,tv,**params)
        rec={'regime':'oracle_geometry','input':'exact synthetic 3D pose tokens + known 2D/validity/simulated texture',
             'rendered_perception':False,'affected_frames_zero_based':affected,'frames':q,'errors':{}}
        for name_frame,Rgt in [('pelvis',pelvis),('thorax',thorax)]:
            errs=[];accepted=[];baseline=[]
            for i,frame in enumerate(q):
                raw=torso_frames(j[i])[1 if name_frame=='pelvis' else 0]
                baseline.append(float(T.rotation_angle(Rgt[i],raw)) if raw is not None else None)
                if frame[name_frame]['R'] is not None:
                    accepted.append(i);errs.append(float(T.rotation_angle(Rgt[i],frame[name_frame]['R'])))
            reject=(sum(q[i][name_frame]['R'] is None for i in affected)/len(affected)) if affected else None
            rec['errors'][name_frame]={'coverage':len(accepted)/n,'geodesic_error_deg':T.summary(errs),
                'raw_error_deg':T.summary(baseline),'affected_abstention_fraction':reject,
                'accepted_indices_zero_based':accepted}
        # Known canonical segment endpoints for a simple same-proportion rig:
        # raw observed pose motion is separate from orientation-applied rigid
        # offsets. No metric S1 anatomy is imported into synthetic GT.
        fixed_offset=np.array([.2,0,.5]);predp=[];truep=[]
        for i in range(n):
            rt,rp=torso_frames(j[i]);predp.append(rp@fixed_offset if rp is not None else np.full(3,np.nan));truep.append(pelvis[i]@fixed_offset)
        rec['rig_world_position_error_m']=T.summary(np.linalg.norm(np.array(predp)-truep,axis=1))
        rec['recovery_after_missing']=next((i-30 for i in range(30,n) if q[i]['pelvis']['R'] is not None),None) if name=='missing_and_LR_swap' else None
        cases[name]=rec
    g=d['gates'];gates={}
    for tag in ('stationary','articulated_arms'):
        gates[tag]=all(cases[tag]['errors'][k]['coverage']>=g['synthetic_clear_coverage'] and
                       cases[tag]['errors'][k]['geodesic_error_deg'].get('max',1e9)<=g['synthetic_stationary_error_max_deg'] for k in ('pelvis','thorax'))
    for tag in ('gradual_turn_180','rapid_turn_90'):
        gates[tag]=all(cases[tag]['errors'][k]['coverage']>=g['synthetic_clear_coverage'] and
                       cases[tag]['errors'][k]['geodesic_error_deg'].get('p95',1e9)<=g['synthetic_clear_turn_p95_deg'] for k in ('pelvis','thorax'))
    gates['false_flip_rejection']=cases['false_3d_flip_1_and_10_frames']['errors']['pelvis']['affected_abstention_fraction']>=g['false_flip_qualification_rejection']
    gates['degenerate_abstention']=cases['depth_ambiguity_degeneracy']['errors']['pelvis']['affected_abstention_fraction']>=g['degenerate_abstention']
    gates['missing_recovery']=cases['missing_and_LR_swap']['recovery_after_missing']<=g['recovery_frames']
    for tag in ('pelvis_fixed_torso_twist','pelvis_fixed_torso_bend'):
        gates[tag]=all(cases[tag]['errors'][k]['geodesic_error_deg'].get('max',1e9)<=g['pelvis_torso_max_error_deg'] for k in ('pelvis','thorax'))
    result={'schema':'animcv_w145_synthetic_v1','declaration_sha256':H.sha(W.DECL),'script_sha256':H.sha(__file__),
        'cases':cases,'gates':gates,'eligible_for_real_correction':all(gates.values()),
        'limits':'Controlled geometry only, not S0 model accuracy on rendered images or physical S1 truth. Measurement qualifier retains wrong static branches when no persistent independent constraint exists.'}
    H.write(W.OUT/'synthetic.json',result)
    print('SYNTHETIC',gates)
    for k,c in cases.items():print(k,c['errors'])


if __name__=='__main__':run()
