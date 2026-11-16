"""W145 official conditioning and frozen S0 intervention measurements.

All policies are in the prospective declaration. No pose/Action correction is
constructed. Source texture interventions are diagnostics, not anatomical GT.
"""
from __future__ import annotations
from pathlib import Path
import sys,time,json
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
import w144_harness as H
import w145_harness as W
import w145_body_frame_core as B
from app import experimental_scene_anchored_trace as T
from motion.torso_frame import torso_frames,quaternion_to_matrix
from framepose.crops import crop_box,geometry_in_crop


def frames(j):
    rt,rp=torso_frames(j)
    return {'Rp':rp,'Rt':rt,'relative':rp.T@rt if rp is not None and rt is not None else None}


def angles(a,b):
    return {k:float(T.rotation_angle(a[k],b[k])) if a[k] is not None and b[k] is not None else None for k in a}


def parents_state(i,new_frames,trace,rc,baseline_frames):
    old=baseline_frames[i];new=new_frames
    state={'object':np.array(trace['baseline'][i]['object']),
           'basis':{k:np.array(v) for k,v in trace['baseline'][i]['basis'].items()}}
    A=rc['Am'];Bsp=rc['spine_basis']
    state['object'][:3,:3]=state['object'][:3,:3]@A@old['Rp'].T@new['Rp']@A.T
    Q=Bsp.T@A@(new['Rp'].T@new['Rt'])@A.T@Bsp
    state['basis']['spine_01'][:3,:3]=Q
    return state


def measure():
    decl=W.verify();W.OUT.mkdir(parents=True,exist_ok=True)
    t=time.monotonic();z=np.load(H.OBS);js=z['predictions'][79:319].astype(float)
    xy=z['input_2d'][79:319].copy();valid=z['input_valid'][79:319];geo=z['geometry'][79:319]
    trace=H.load(H.OUT/'blender_trace.json');bf=[frames(j) for j in js]
    import w121_independent_transfer as adapter
    rc=adapter.rig_ctx()
    rows=[];perturbs=[];probes=[];jac_error=[]
    for i,j in enumerate(js):
        row={'action':i+1,'source':i+79,'dimensions':B.dimensions(j),'frames':{},'sensitivity':{},'perturbation_max':{}}
        for name,idx in [('pelvis','Rp'),('thorax','Rt')]:
            R=bf[i][idx];J=B.jacobian(j,name)
            row['frames'][idx]={'determinant':float(np.linalg.det(R)),'orthogonality_max':float(np.abs(R.T@R-np.eye(3)).max()),'R':R.tolist()}
            # Analytic vs numerical Jacobian: use matrix differential tangent,
            # which is more accurate than acos for very small rotations.
            columns=[]
            for k in B.AXIS_JOINTS:
                for ax in range(3):
                    jp=j.copy();jm=j.copy();jp[k,ax]+=1e-6;jm[k,ax]-=1e-6
                    Rp=frames(jp)[idx];Rm=frames(jm)[idx]
                    m=R.T@((Rp-Rm)/2e-6);m=(m-m.T)/2
                    columns.append([m[2,1],m[0,2],m[1,0]])
            numeric=np.array(columns).T
            err=float(np.linalg.norm(J-numeric)/max(np.linalg.norm(J),1e-20));jac_error.append(err)
            row['sensitivity'][idx]={'operator_deg_per_mm':float(np.degrees(np.linalg.svd(J,compute_uv=False)[0])*.001),
                'central_difference_relative_error':err}
        for mm in decl['perturbations']['3d_mm']:
            variants=[]
            for k in B.AXIS_JOINTS:
                for ax in range(3):
                    for sign in (-1,1):
                        p=j.copy();p[k,ax]+=sign*mm/1000;f=frames(p);da=angles(bf[i],f)
                        entry={'action':i+1,'source':i+79,'mm':mm,'joint':k,'axis':ax,'sign':sign,'angles_deg':da}
                        variants.append((entry,f));perturbs.append(entry)
            row['perturbation_max'][str(mm)]={k:max((v[0]['angles_deg'][k] for v in variants if v[0]['angles_deg'][k] is not None),default=None) for k in ('Rp','Rt','relative')}
            row['perturbation_max'][str(mm)]['invalid_variants']=sum(any(v is None for v in e['angles_deg'].values()) for e,f in variants)
            if i+1 in decl['perturbations']['selected_model_actions']:
                candidates=[v for v in variants if v[1]['Rp'] is not None and v[1]['Rt'] is not None]
                entry,f=max(candidates,key=lambda v:v[0]['angles_deg']['Rp'])
                state=parents_state(i,f,trace,rc,bf);_,ww=T.fk(trace['rest'],state['basis'],state['object'])
                probes.append({'tag':f'3d_action{i+1}_{mm}mm_maxRp','action':i+1,'source':i+79,
                    'factor':entry,'object':state['object'].tolist(),'basis':{k:v.tolist() for k,v in state['basis'].items()},
                    'expected_world':{k:mat[:3,3].tolist() for k,mat in ww.items()}})
        if i:
            row['increments_deg']=angles(bf[i-1],bf[i])
            row['source_max_step_px']={name:float(np.linalg.norm((xy[i,idx,:2]-xy[i-1,idx,:2])*[640,480],axis=1).max())
                for name,idx in [('hips',[1,4]),('shoulders',[11,14])]}
            row['bilateral_order_change']={}
            for name,l,r in [('hips',1,4),('shoulders',11,14)]:
                a=js[i-1,r]-js[i-1,l];b=js[i,r]-js[i,l]
                row['bilateral_order_change'][name]={'X_sign':bool(a[0]*b[0]<0),'depth_sign':bool(a[1]*b[1]<0),
                    'depth_previous_m':float(a[1]),'depth_current_m':float(b[1]),'dot_direction':float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)))}
        rows.append(row)
    cond={'schema':'animcv_w145_conditioning_v1','declaration_sha256':H.sha(W.DECL),'script_sha256':H.sha(__file__),
        'regime':'real_animcv_observation','rows':rows,'analytic_central_relative_error':T.summary(jac_error),
        'elapsed_seconds':time.monotonic()-t,'perturbations':len(perturbs)}
    H.write(W.OUT/'conditioning.json',cond);H.write(W.OUT/'3d_perturbations.json',perturbs)
    # Independent measurement qualification of original same inputs; no repaired
    # matrix is available on abstained frames.
    source=H.load(H.OUT/'source_controls.json')['edges']
    residual=np.array([[np.nan if v is None else v for v in e['landmark_texture_residual_px']] for e in source])
    texture_valid=np.array([e['texture_ok'] for e in source])
    params={k:decl['qualifier'][k] for k in ('sigma_m','max_condition_deg','texture_max_px','unsupported_turn_deg','quiet_source_px')}
    qualified=B.qualify(js,xy[:,:,:2]*[640,480],valid,residual,texture_valid,**params)
    H.write(W.OUT/'qualification.json',{'schema':'animcv_w145_qualification_v1','regime':'real_animcv_observation','frames':qualified,
        'note':'Nulls are unresolved orientations; never substituted into a full real Action.'})
    # Frozen, CPU-only model reproduction. No optimizer, training or GPU call.
    import torch
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    from framepose.train import load_checkpoint,predict
    from framepose.signs import neutral_sign_states
    assert H.sha(W.MODEL)==decl['identities']['model_sha256']
    model,ck=load_checkpoint(W.MODEL,device='cpu')
    assert ck['candidate']['name']=='S0_neutral_sign'
    def infer(g):
        return predict(model,torch,np.asarray(g,np.float32),None,np.arange(len(g)),torch.device('cpu'),
                       batch_size=512,amp_enabled=False,signs=neutral_sign_states(len(g)))
    ti=time.monotonic();cpu=infer(geo);repeat=infer(geo)
    maxabs=float(np.max(np.abs(cpu-js)));rep=float(np.max(np.abs(cpu-repeat)))
    cached_angles={k:T.summary([angles(a,frames(p))[k] for a,p in zip(bf,cpu)]) for k in ('Rp','Rt','relative')}
    compatible=maxabs<=decl['s0_cpu']['cached_max_abs_m'] and rep<=decl['s0_cpu']['repeat_max_abs_m'] and all(v.get('max',1e9)<=decl['s0_cpu']['cached_frame_angle_deg'] for v in cached_angles.values())
    identity={'torch_version':torch.__version__,'device':'cpu','model_sha256':H.sha(W.MODEL),'model_config':ck['model_config'],
        'candidate':ck['candidate']['name'],'baseline_max_abs_m':maxabs,'repeat_max_abs_m':rep,'cached_angle_degrees':cached_angles,
        'compatible_with_cached':compatible,'inference_seconds':time.monotonic()-ti,'amp':False,'threads':4,'signs':'neutral'}
    rederived=[]
    for x,v in zip(xy,valid):
        q,qv=B.independent_2d_midpoints(x,v)
        rederived.append(geometry_in_crop(q,qv,(640,480),crop_box(q,qv,(640,480))))
    noop=infer(rederived)
    identity['derived_midpoint_rounding_noop_max_m']=float(np.max(np.abs(noop-cpu)))
    np.savez_compressed(W.OUT/'cpu_baseline.npz',predictions=cpu,repeat=repeat,source_frame=np.arange(79,319))
    # +/- 1px direct-joint interventions, same valid bits/scores, with virtual
    # points rederived. Compare fixed vs recomputed crop as matched factors.
    g_all=[];labels=[]
    for policy in ('fixed_crop','recomputed_crop'):
        for k in decl['perturbations']['2d_direct_joints']:
            for ax in range(2):
                for sign in (-1,1):
                    for i,x in enumerate(xy):
                        q=x.copy();q[k,ax]+=sign/[640,480][ax];q,v=B.independent_2d_midpoints(q,valid[i])
                        box=crop_box(x,valid[i],(640,480)) if policy=='fixed_crop' else crop_box(q,v,(640,480))
                        g_all.append(geometry_in_crop(q,v,(640,480),box));labels.append((policy,k,ax,sign,i))
    ti=time.monotonic();responses=infer(g_all);dense=[]
    cb=[frames(p) for p in cpu]
    for label,p in zip(labels,responses):
        policy,k,ax,sign,i=label
        dense.append({'policy':policy,'joint':k,'axis':ax,'sign':sign,'action':i+1,'source':i+79,
                      'prediction_max_abs_change_m':float(np.max(np.abs(p-cpu[i]))),'angles_deg':angles(cb[i],frames(p))})
    identity['dense_intervention_samples']=len(labels);identity['dense_inference_seconds']=time.monotonic()-ti
    H.write(W.OUT/'2d_perturbations.json',dense)
    # Selected matched previous-observation/texture interventions. Each changes
    # one named observation group, rederives virtual points and reruns S0.
    selected_g=[];selected_l=[]
    for action in decl['perturbations']['selected_model_actions']:
        i=action-1
        if i==0:continue
        for name,idx in [('previous_hips',[1,4]),('previous_shoulders',[11,14]),('texture_hips_shoulders',[1,4,11,14])]:
            q=xy[i].copy();changed=[]
            for k in idx:
                if name.startswith('previous'):
                    if valid[i,k] and valid[i-1,k]:q[k,:2]=xy[i-1,k,:2];changed.append(k)
                elif texture_valid[i-1,k] and valid[i,k] and valid[i-1,k]:
                    # Recompute the EXACT frozen LK endpoint, not a residual
                    # scalar shifted along an invented direction.
                    changed.append(k)
            if name=='texture_hips_shoulders':
                q=texture_input(i,q,xy,valid,changed)
            q,v=B.independent_2d_midpoints(q,valid[i]);box=crop_box(q,v,(640,480))
            selected_g.append(geometry_in_crop(q,v,(640,480),box));selected_l.append((name,i,changed))
    selected_out=infer(selected_g);selected=[]
    for (name,i,changed),p in zip(selected_l,selected_out):
        f=frames(p);rec={'intervention':name,'action':i+1,'source':i+79,'changed_direct_joints':changed,
            'vs_current_angles_deg':angles(cb[i],f),'vs_previous_angles_deg':angles(cb[i-1],f),
            'prediction_max_abs_change_m':float(np.max(np.abs(p-cpu[i])))}
        if all(f[k] is not None for k in ('Rp','Rt')):
            state=parents_state(i,f,trace,rc,bf);_,ww=T.fk(trace['rest'],state['basis'],state['object'])
            rec['shoulder_vs_previous_m']=float(np.linalg.norm(ww['upperarm_l'][:3,3]-trace['baseline'][i-1]['world']['upperarm_l']['head']))
            probes.append({'tag':f'model_{name}_action{i+1}','action':i+1,'source':i+79,'factor':name,
                'object':state['object'].tolist(),'basis':{k:v.tolist() for k,v in state['basis'].items()},
                'expected_world':{k:m[:3,3].tolist() for k,m in ww.items()}})
        selected.append(rec)
    H.write(W.OUT/'selected_model_interventions.json',selected)
    np.savez_compressed(W.OUT/'model_responses.npz',dense=responses,selected=selected_out)
    H.write(W.OUT/'model_identity.json',identity)
    H.write(W.OUT/'blender_probe_spec.json',{'schema':'animcv_w145_readonly_probes_v1','original_A_sha256':H.sha(H.BLEND),
        'purpose':'diagnostic orientation propagation, not animation candidates','probes':probes})
    print('MEASURE',json.dumps({'seconds':time.monotonic()-t,'jac_error':cond['analytic_central_relative_error'],
        'cpu':identity,'qualification':{k:sum(f[k]['R'] is not None for f in qualified) for k in ('pelvis','thorax')}}))


_source_images=None
def texture_input(i,q,xy,valid,changed):
    import cv2
    global _source_images
    if _source_images is None:
        cap=cv2.VideoCapture(str(H.CLIP));_source_images=[]
        for frame in range(319):
            ok,im=cap.read();assert ok
            if frame>=79:_source_images.append(cv2.cvtColor(im,cv2.COLOR_BGR2GRAY))
        cap.release()
    p=(xy[i-1,:,:2]*[640,480]).astype(np.float32).reshape(-1,1,2)
    nxt,st,_=cv2.calcOpticalFlowPyrLK(_source_images[i-1],_source_images[i],p,None,winSize=(21,21),maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,30,.01),minEigThreshold=1e-4)
    for k in changed:q[k,:2]=nxt[k,0]/[640,480]
    return q


if __name__=='__main__':measure()
