"""W144 analysis of frozen declaration and read-only trace. No corrective output.

This reporter implements the already declared metrics/selection. It was authored
after trace extraction, before inspecting trajectory outcomes. Its own hash is
recorded in the result; it does not change the frozen trace instrument.
"""
from __future__ import annotations
import hashlib
import json
import sys
from pathlib import Path
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import w144_harness as H
from app import experimental_scene_anchored_trace as T
from motion.torso_frame import torso_frames,quaternion_to_matrix

JOINTS=('pelvis','left_hip','left_knee','left_ankle','right_hip','right_knee','right_ankle',
        'spine','thorax','neck','head','left_shoulder','left_elbow','left_wrist',
        'right_shoulder','right_elbow','right_wrist')
NODES={'pelvis_proxy':('thigh_l','thigh_r'),'torso':'spine_05','neck':'neck_02',
       'shoulder_l':'upperarm_l','shoulder_r':'upperarm_r','clavicle_l':'clavicle_l',
       'clavicle_r':'clavicle_r','elbow_l':'lowerarm_l','wrist_l':'hand_l',
       'elbow_r':'lowerarm_r','wrist_r':'hand_r','foot_l':'foot_l','foot_r':'foot_r',
       'center_of_mass':'center_of_mass','spine_01':'spine_01'}


def points(fr,name):
    n=NODES.get(name,name)
    return (np.mean([fr['world'][k]['head'] for k in n],axis=0) if isinstance(n,tuple)
            else np.asarray(fr['world'][n]['head']))


def cp_points(rec,name):
    n=NODES[name]
    return np.mean([rec[k] for k in n],axis=0) if isinstance(n,tuple) else np.asarray(rec[n])


def R(m):
    a=np.asarray(m)[:3,:3]
    return a/np.linalg.norm(a,axis=0)


def analyze():
    H.verify();started=time.monotonic()
    z=np.load(H.OBS);sl=slice(79,319)
    j=z['predictions'][sl].astype(float)
    obs=z['input_2d'][sl].copy();obs[:,:,:2]*=[640,480]
    valid=z['input_valid'][sl]
    trace=H.load(H.OUT/'blender_trace.json');bs=trace['baseline'];rest=trace['rest']
    result={'schema':'animcv_w144_analysis_v1','declaration_sha256':H.sha(H.DECL),
            'analysis_script_sha256':H.sha(__file__),'trace_sha256':H.sha(H.OUT/'blender_trace.json'),
            'regime':'real_animcv_observation','source':{},'pose3d':{},'world':{},'rotation':{},'controls':{}}
    edge_rows=[{'from_action':i+1,'to_action':i+2,'from_source':i+79,'to_source':i+80} for i in range(239)]
    source_steps={}
    derived={'pelvis':'hip midpoint','spine':'pelvis-neck midpoint','thorax':'shoulder midpoint','neck':'same shoulder midpoint','head':'nose proxy'}
    for k,name in enumerate(JOINTS):
        m,d=T.motion(obs[:,k,:2],valid[:,k]);source_steps[name]=d
        m['confidence']=T.summary(obs[:,k,2]);m['invalid_action_frames']=(np.flatnonzero(~valid[:,k])+1).tolist()
        m['jump_destinations_over_20px']=(np.flatnonzero(d>20)+2).tolist()
        m['provenance']=derived.get(name,'RTMPose estimated joint; detector validity is not visibility GT')
        m['independent_occlusion_labels']='not present; manual source review separate; far arm may be self-occluded even with high score'
        result['source'][name]=m
        pm,pd=T.motion(j[:,k]);result['pose3d'][name]=pm
        for i,row in enumerate(edge_rows):
            row[f'obs_{name}_px']=float(d[i]) if np.isfinite(d[i]) else None
            row[f'pose_{name}_m']=float(pd[i])
    # Keypoint envelope (not a detector bbox or proof of joint stationarity).
    env=[]
    for xy,v in zip(obs[:,:,:2],valid):
        a=xy[v];env.append(np.r_[a.min(0),a.max(0)])
    env=np.array(env);result['source_envelope']={'centre':T.motion((env[:,:2]+env[:,2:])/2)[0],
        'size':T.motion(env[:,2:]-env[:,:2])[0],'label':'valid canonical joint envelope; original detector bbox and silhouettes were not cached'}
    world_steps={}
    for name in NODES:
        p=np.array([points(f,name) for f in bs]);m,d=T.motion(p);world_steps[name]=d
        m['range_xyz_m']=np.ptp(p,axis=0).tolist();m['speed_m_per_s']=T.summary(d*30)
        m['acceleration_m_per_s2']=T.summary(np.linalg.norm(np.diff(p,n=2,axis=0),axis=1)*900)
        result['world'][name]=m
        for i,row in enumerate(edge_rows):row[f'world_{name}_m']=float(d[i])
    mw=np.array([f['object'] for f in bs]);obj_r=np.array([R(m) for m in mw])
    result['object_translation']=T.motion(mw[:,:3,3])[0]
    rots={'object':obj_r}
    for n in rest:rots[n]=np.array([R(f['basis'][n]) for f in bs])
    rp_rt=[torso_frames(p) for p in j]
    rots['inferred_pelvis_Rp']=np.array([r[1] for r in rp_rt]);rots['inferred_thorax_Rt']=np.array([r[0] for r in rp_rt])
    for n,rs in rots.items():
        inc=T.rotation_angle(rs[:-1],rs[1:]);result['rotation'][n]={'increments_deg':T.summary(inc),'peak_to_action':int(np.argmax(inc)+2)}
        for i,row in enumerate(edge_rows):row[f'rotation_{n}_deg']=float(inc[i])
    # Independent CPU FK vs Blender, including tails, every frame.
    worst=0.;tail_worst=0.;historical=H.load(H.RUN/'baseline/baseline_readback.json')
    hist=0.;lengths={n:[] for n in rest}
    for i,f in enumerate(bs):
        arm,world=T.fk(rest,f['basis'],f['object'])
        for n,wm in world.items():
            expected=np.array(f['world'][n]['head']);worst=max(worst,float(np.linalg.norm(wm[:3,3]-expected)))
            tail_local=np.linalg.inv(rest[n]['matrix_local']) @ np.r_[rest[n]['tail_local'],1]
            tail=wm@tail_local;tail_worst=max(tail_worst,float(np.linalg.norm(tail[:3]-f['world'][n]['tail'])))
            hist=max(hist,float(np.max(np.abs(expected-historical['frames'][i]['world'][n]['head']))))
            lengths[n].append(np.linalg.norm(np.array(f['world'][n]['tail'])-expected))
    submax=0.
    for f in trace['subframes']:
        orig=bs[f['action']-1]
        submax=max(submax,max(float(np.max(np.abs(np.array(f['world'][n]['head'])-orig['world'][n]['head']))) for n in rest))
    result['controls']['fk']={'head_max_m':worst,'tail_max_m':tail_worst,'pass':max(worst,tail_worst)<=1e-5,
        'historical_readback_max_abs_m':hist,'same_action_value_digest':trace['value_digest']==historical['value_digest'],
        'subframe_hold_max_abs_m':submax,'subframe_samples':len(trace['subframes']),
        'length_range_m':{n:float(np.ptp(v)) for n,v in lengths.items()},
        'constraints':{n:v['constraints'] for n,v in rest.items() if v['constraints']},
        'nla_tracks':trace['nla_tracks'],'drivers':trace['drivers'],'object_constraints':trace['object_constraints']}
    # Recompute the ACTUAL frozen S0-only adapter (not a new retargeter).
    import w121_independent_transfer as W
    rc=W.rig_ctx();frozen=H.load(H.RUN/'rig/video_cases.json')['frames'];fresh=[];mismatch=[]
    for i,p in enumerate(j):
        fr=W.fixed_slot(p,rc);fresh.append(fr)
        for key in ('object_rotation_xyzw','extra_bone_locals_xyzw','modes','parent'):
            if fr[key]!=frozen[i][key]:mismatch.append([i+1,key])
    result['controls']['frozen_adapter']={'replayed_frames':len(fresh),'mismatches':mismatch,'pass':not mismatch}
    # Endpoint counterfactuals through Blender. Both necessity and sufficiency.
    cf=trace['counterfactuals'];result['counterfactual_summary']={}
    for name in ('pelvis_proxy','torso','shoulder_l','shoulder_r','wrist_l','foot_l'):
        factors={}
        for k in cf[0]['hold']:
            held=[];injected=[]
            for i,c in enumerate(cf):
                a=points(bs[i],name);b=points(bs[i+1],name)
                held.append(float(np.linalg.norm(cp_points(c['hold'][k],name)-a)))
                injected.append(float(np.linalg.norm(cp_points(c['inject'][k],name)-a)))
                edge_rows[i][f'hold_{k}_{name}_m']=held[-1]
                edge_rows[i][f'inject_{k}_{name}_m']=injected[-1]
            factors[k]={'held_step':T.summary(held),'injected_step':T.summary(injected)}
        result['counterfactual_summary'][name]=factors
    # Reconstruct counterfactual object/spine states from upstream joints and
    # evaluate full FK; no source points or correction are manufactured.
    upstream={};A=quaternion_to_matrix(rc['alignment'])
    for factor,idx in {'pelvis':[0],'hip_pair':[1,4],'thorax':[8],'shoulder_pair':[11,14]}.items():
        vals=[];rpinc=[];rtinc=[]
        for i in range(1,240):
            p=j[i].copy();p[idx]=j[i-1,idx];f=W.fixed_slot(p,rc)
            state={'object':np.array(bs[i]['object']), 'basis':{n:np.array(m) for n,m in bs[i]['basis'].items()}}
            # Replace ALL adapter outputs affected by the one upstream factor.
            # Imported object rest rotation is present in baseline and removed
            # relative to the original adapter output, then recomposed.
            oldq=fresh[i]['object_rotation_xyzw']['FIXED'];newq=f['object_rotation_xyzw']['FIXED']
            state['object'][:3,:3]=state['object'][:3,:3]@quaternion_to_matrix(oldq).T@quaternion_to_matrix(newq)
            bones=dict(f['extra_bone_locals_xyzw']['FIXED']);arm=f['modes']['FIXED']
            bones.update(upperarm_l=arm['proximal_local_xyzw'],lowerarm_l=arm['distal_local_xyzw'])
            for n,q in bones.items():state['basis'][n][:3,:3]=quaternion_to_matrix(q)
            _,ww=T.fk(rest,state['basis'],state['object'])
            step=float(np.linalg.norm(ww['upperarm_l'][:3,3]-points(bs[i-1],'shoulder_l')));vals.append(step)
            rpinc.append(float(T.rotation_angle(frozen[i-1]['parent']['FIXED']['Rp'],f['parent']['FIXED']['Rp'])))
            rtinc.append(float(T.rotation_angle(frozen[i-1]['parent']['FIXED']['Rt'],f['parent']['FIXED']['Rt'])))
            edge_rows[i-1][f'upstream_hold_{factor}_shoulder_l_m']=step
        upstream[factor]={'shoulder_step':T.summary(vals),'Rp_step':T.summary(rpinc),'Rt_step':T.summary(rtinc)}
    result['upstream_counterfactuals']=upstream
    # Exact non-additive two-factor endpoint accounting, not causal percentages.
    interactions=[]
    for i,c in enumerate(cf):
        a=points(bs[i],'shoulder_l');b=points(bs[i+1],'shoulder_l')
        obj=cp_points(c['inject']['object_rotation'],'shoulder_l');sp=cp_points(c['inject']['bone:spine_01'],'shoulder_l')
        both=cp_points(c['inject']['object_rotation+spine_01'],'shoulder_l')
        edge_rows[i]['object_then_spine_conditional_m']=float(np.linalg.norm(both-obj))
        edge_rows[i]['spine_then_object_conditional_m']=float(np.linalg.norm(both-sp))
        edge_rows[i]['interaction_vector_norm_m']=float(np.linalg.norm(both-obj-sp+a))
        interactions.append(edge_rows[i]['interaction_vector_norm_m'])
    result['interactions']=T.summary(interactions)
    # Perspective depth is not available: camera influences are qualified only.
    cam=H.load(H.CAM)
    result['camera']={'source':str(H.CAM),'sha256':H.sha(H.CAM),'calibration':cam['calibration_status'],
        'sfm':cam['sfm_validity'],'summary':cam.get('summary'),
        'production_camera_path':'ABSENT in frozen S1; constant-camera transform intervention would be identity',
        'orientation_only_sensitivity':'nominal orientation sensitivity can describe rotation only; no trustworthy bound or metric root correction',
        'quality_reprojection':'NOT USED; no independently valid S1 camera model'}
    # Synthetic Blender correctness is checked with independent FK, including
    # oracle transforms and their one-factor holds. Semantics are labelled.
    syn={};base=trace['synthetic']['base']
    for name,f in trace['synthetic'].items():
        _,ww=T.fk(rest,f['basis'],f['object'])
        err=max(float(np.linalg.norm(ww[n][:3,3]-f['world'][n]['head'])) for n in rest)
        syn[name]={'fk_error_m':err,'shoulder_displacement_m':float(np.linalg.norm(points(f,'shoulder_l')-points(base,'shoulder_l'))),
                   'object_translation_m':float(np.linalg.norm(np.array(f['object'])[:3,3]-np.array(base['object'])[:3,3]))}
    result['controls']['synthetic']={'regime':'oracle_geometry','cases':syn,'pass':max(v['fk_error_m'] for v in syn.values())<=1e-5,
        'limitation':'upstream_pose_jump and root_rotation have identical endpoint transforms; endpoint data alone cannot disambiguate their source. Parent defect is an intentionally duplicated application, detected with known oracle expectations, not evidence of such a defect in S1.'}
    # Frozen baseline/source-only selection, no corrected result exists.
    top=sorted(range(239),key=lambda i:(-world_steps['shoulder_l'][i],i))[:5]
    quiet=min(range(229),key=lambda i:float(np.sum(world_steps['shoulder_l'][i:i+11])))
    # Largest path using the confidently observed wrists, with unavailable edges
    # excluded, while explicitly recording which wrist was available.
    wrist=np.nansum(np.stack([source_steps['left_wrist'],source_steps['right_wrist']]),axis=0)
    large=max(range(229),key=lambda i:float(np.sum(wrist[i:i+11])))
    uncertain=max(range(229),key=lambda i:int(np.sum(~valid[i:i+12])))
    result['selection']={'top_shoulder_destinations':[i+2 for i in top],
        'top_windows':[[max(1,i-1),min(240,i+5)] for i in top],
        'quiet':[quiet+1,quiet+12],'large_source_wrist':[large+1,large+12],
        'uncertain':[uncertain+1,uncertain+12],'fixed':H.load(H.DECL)['selection']['fixed']}
    result['focus_edges']=[edge_rows[i-2] for i in sorted(set([172,200]+[k+2 for k in top]))]
    result['candidate_decision']={'produced':False,'reason':'Pending evidence interpretation; no candidate made by diagnostic reporter.'}
    result['seconds']=time.monotonic()-started
    H.write(H.OUT/'analysis.json',result);H.write(H.OUT/'edge_trajectories.json',edge_rows)
    np.savez_compressed(H.OUT/'trajectories.npz',source_frame=np.arange(79,319),action=np.arange(1,241),
                        observations_px=obs,observation_valid=valid,predictions=j,
                        **{f'world_{n}':np.array([points(f,n) for f in bs]) for n in NODES})
    print('RESULTS',json.dumps({'world':{k:result['world'][k]['step'] for k in ('pelvis_proxy','torso','shoulder_l','shoulder_r')},
        'object_translation':result['object_translation'],'root_rotation':result['rotation']['object'],
        'selection':result['selection'],'controls':{k:v.get('pass') for k,v in result['controls'].items()}}))
    print('FOCUS',json.dumps([{k:v for k,v in r.items() if k in ('to_action','world_shoulder_l_m','world_pelvis_proxy_m','obs_thorax_px','obs_pelvis_px','rotation_object_deg','rotation_spine_01_deg','pose_thorax_m','upstream_hold_thorax_shoulder_l_m','hold_object_rotation_shoulder_l_m','hold_bone:spine_01_shoulder_l_m','hold_object_rotation+spine_01_shoulder_l_m')} for r in result['focus_edges']]))


if __name__=='__main__':analyze()
