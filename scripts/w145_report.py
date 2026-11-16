"""W145 verdict and review artifacts from frozen measurements; no correction."""
from __future__ import annotations
from pathlib import Path
import sys,hashlib,collections,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
import w144_harness as H
import w145_harness as W
from app import experimental_scene_anchored_trace as T


def report():
    W.verify();execution=H.load(ROOT/'examples/experimental_body_frame/w145_execution_manifest.json')
    for f,h in execution['files_sha256_lf'].items():assert hashlib.sha256((ROOT/f).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==h,f
    c=H.load(W.OUT/'conditioning.json');rows=c['rows'];dense=H.load(W.OUT/'2d_perturbations.json')
    selected=H.load(W.OUT/'selected_model_interventions.json');q=H.load(W.OUT/'qualification.json')['frames']
    syn=H.load(W.OUT/'synthetic.json');identity=H.load(W.OUT/'model_identity.json')
    spec=H.load(W.OUT/'blender_probe_spec.json')['probes'];rb=H.load(W.OUT/'blender_readback.json')
    historical=H.load(H.OUT/'blender_trace.json')['baseline']
    fk=max(float(np.linalg.norm(np.array(f['world'][n]['head'])-s['expected_world'][n])) for f,s in zip(rb['probes'],spec) for n in s['expected_world'])
    baseline=max(float(np.max(np.abs(np.array(f['world'][n][end])-historical[i]['world'][n][end]))) for i,f in enumerate(rb['baseline']) for n in f['world'] for end in ('head','tail'))
    fixed=[x for x in dense if x['policy']=='fixed_crop'];recomputed=[x for x in dense if x['policy']=='recomputed_crop']
    result={'schema':'animcv_w145_assessment_v1','date':'2026-11-16','regime':'real_animcv_observation',
        'declaration_sha256':H.sha(W.DECL),'script_sha256':H.sha(__file__),'source_head':H.git('rev-parse','HEAD'),
        'W144_preservation_commit':'53361a9c6de737910817a97d8ffbc159c8203723','model':identity,
        'condition_summary':{},'2d_sensitivity':{},'qualification':{},'synthetic_gates':syn['gates'],
        'blender':{'baseline_frames':240,'baseline_max_abs_m':baseline,'diagnostic_probes':len(spec),'FK_head_max_m':fk,'tolerance_m':1e-5,'passed':fk<=1e-5 and baseline==0},
        'candidate':{'eligible':syn['eligible_for_real_correction'],'produced':False,'count':0,'reversibility':'NOT APPLICABLE','pin_gate':'NOT APPLICABLE: no historical or new Action edit'},
        'limits':['Qualification is not calibrated 3D confidence; it accepts persistent false branches in synthetic controls.',
            'Synthetic exact 3D/2D/simulated texture controls do not run rendered-image M1 or train/evaluate S0 accuracy.',
            'CPU S0 is compatible to declared numerical tolerance, not bit-identical to historical GPU output.',
            'Source texture interventions can change model orientation but are not anatomical corrections.',
            'No trusted S1 camera calibration, metric human root depth or hidden pelvis ground truth.'],
        'native_rate_perceptual_review':{'status':'BLOCKED','reason':'cua_repl initialization 0xC0000142 before any UI access','human_acceptance':False}}
    for part in ('pelvis','thorax'):
        key='Rp' if part=='pelvis' else 'Rt'
        result['condition_summary'][part]={'side_length_m':T.summary([r['dimensions'][part]['side_m'] for r in rows]),
            'projected_side_m':T.summary([r['dimensions'][part]['projected_side_m'] for r in rows]),
            'sin_axis_angle':T.summary([r['dimensions'][part]['sin_axis_angle'] for r in rows]),
            'operator_deg_per_mm':T.summary([r['sensitivity'][key]['operator_deg_per_mm'] for r in rows]),
            'orthogonality_max':max(r['frames'][key]['orthogonality_max'] for r in rows),
            'determinant_error_max':max(abs(r['frames'][key]['determinant']-1) for r in rows),
            'controlled_3d_max_change_per_frame':{str(mm):T.summary([r['perturbation_max'][str(mm)][key] for r in rows]) for mm in (1,2,5)}}
        result['qualification'][part]={'coverage':sum(r[part]['R'] is not None for r in q)/240,
            'qualified':sum(r[part]['R'] is not None for r in q),
            'reasons':dict(collections.Counter(s for r in q for s in r[part]['reasons']))}
    result['qualification']['both_frames']=sum(r['pelvis']['R'] is not None and r['thorax']['R'] is not None for r in q)
    result['analytic_FD_relative_error']=c['analytic_central_relative_error']
    result['3d_perturbation_count']=c['perturbations']
    for policy,group in [('fixed_crop',fixed),('recomputed_crop',recomputed)]:
        result['2d_sensitivity'][policy]={'samples':len(group),**{k:T.summary([r['angles_deg'][k] for r in group]) for k in ('Rp','Rt','relative')}}
    # Strictly paired crop contrast, never an additive causal percentage.
    pairs={(r['joint'],r['axis'],r['sign'],r['action']):r for r in fixed}
    result['crop_pairing']={'max_Rp_difference_in_response_deg':max(abs(r['angles_deg']['Rp']-pairs[r['joint'],r['axis'],r['sign'],r['action']]['angles_deg']['Rp']) for r in recomputed),
        'note':'Different perturbation responses under identical direct-joint changes; no unique crop/error percentage.'}
    result['representative_interventions']=[r for r in selected if r['action'] in (22,23,26,27,171,172,177,200)]
    # Eligibility is not weakened after the negative synthetic outcomes.
    if syn['eligible_for_real_correction']:
        raise RuntimeError('Candidate eligibility passed: do not finalize without conditional candidate evaluation')
    result['candidate']['reason']='Frozen synthetic false-flip rejection and pelvis-vs-torso bend separation fail; qualifier leaves 68/240 frames without both orientations. No independently constrained complete replacement is justified.'
    result['classification']='PARTIALLY SUPPORTED' if identity['compatible_with_cached'] and result['blender']['passed'] and c['analytic_central_relative_error']['max']<1e-5 else 'NOT YET SUPPORTED'
    result['classification_scope']='Useful conditioning and controlled M1/S0/decoder attribution only. No actual-animation correction or Production readiness demonstrated.'
    H.write(W.OUT/'assessment.json',result)
    visuals(rows,dense,q,syn,result)
    print('W145_ASSESSMENT',result['classification'],'candidate',result['candidate'])


def visuals(rows,dense,q,syn,result):
    import cv2,matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    review=ROOT/'user_eval/W145_body_frame';review.mkdir(exist_ok=False)
    x=np.arange(1,241)
    fig,axs=plt.subplots(5,1,figsize=(14,12),sharex=True)
    for part in ('pelvis','thorax'):
        axs[0].plot(x,[r['dimensions'][part]['projected_side_m']*1000 for r in rows],label=part)
    axs[0].set_ylabel('projected axis mm');axs[0].legend();axs[0].set_title('W145: conditioning, measured S0 response and source qualification; no corrected Action')
    for k in ('Rp','Rt'):
        axs[1].plot(x,[r['sensitivity'][k]['operator_deg_per_mm'] for r in rows],label=k)
    axs[1].set_ylabel('analytic deg/mm');axs[1].legend()
    for k in ('Rp','Rt','relative'):
        axs[2].plot(x[1:],[r['increments_deg'][k] for r in rows[1:]],label=k)
    axs[2].set_ylabel('original step deg');axs[2].legend(ncol=3)
    for policy in ('fixed_crop','recomputed_crop'):
        maxima=[max(r['angles_deg']['Rp'] for r in dense if r['action']==i and r['policy']==policy) for i in x]
        axs[3].plot(x,maxima,label=policy)
    axs[3].set_ylabel('1px max Rp change');axs[3].legend()
    for k,offset in [('pelvis',0),('thorax',1.2)]:
        axs[4].step(x,[offset+(r[k]['R'] is not None) for r in q],where='mid',label=k+' qualified')
    axs[4].set_ylabel('qualification');axs[4].set_xlabel('Action frame; source = Action + 78');axs[4].legend()
    for a in axs:
        for lo,hi in W.verify()['windows']:a.axvspan(lo,hi,color='grey',alpha=.05)
        a.grid(alpha=.2)
    fig.tight_layout();fig.savefig(review/'conditioning_and_attribution.png',dpi=140);plt.close(fig)
    fig,axs=plt.subplots(3,1,figsize=(12,9))
    for tag in ('gradual_turn_180','rapid_turn_90','false_3d_flip_1_and_10_frames'):
        rec=syn['cases'][tag];inds=rec['errors']['pelvis']['accepted_indices_zero_based']
        e=rec['errors']['pelvis']['raw_error_deg']
        axs[0].plot(inds,[float(T.rotation_angle(np.eye(3),np.array(rec['frames'][i]['pelvis']['R']))) for i in inds],label=tag)
    axs[0].set_ylabel('qualified angle deg');axs[0].legend();axs[0].set_title('Synthetic oracle_geometry controls; exact pose tokens, not rendered perception')
    tag='false_3d_flip_1_and_10_frames';rec=syn['cases'][tag];inds=rec['errors']['pelvis']['accepted_indices_zero_based']
    axs[1].scatter(inds,[0 if i not in rec['affected_frames_zero_based'] else 90 for i in inds],s=10,label='accepted pelvis error')
    axs[1].set_ylabel('error deg');axs[1].legend()
    rec=syn['cases']['pelvis_fixed_torso_bend'];inds=rec['errors']['pelvis']['accepted_indices_zero_based']
    axs[2].plot(inds,[float(T.rotation_angle(np.eye(3),rec['frames'][i]['pelvis']['R'])) for i in inds],label='false pelvis tilt from shared trunk axis')
    axs[2].set_ylabel('pelvis error deg');axs[2].legend();axs[2].set_xlabel('synthetic frame')
    fig.tight_layout();fig.savefig(review/'synthetic_successes_and_failures.png',dpi=140);plt.close(fig)
    # Keep both diagnostic Blender views exactly as in W144, source intact.
    movie=review/'S1_source_body_frame_diagnostics_30fps.mp4'
    writer=cv2.VideoWriter(str(movie),cv2.VideoWriter_fourcc(*'mp4v'),30,(1408,720));assert writer.isOpened()
    cap=cv2.VideoCapture(str(H.CLIP));sheets={};wanted={16,21,22,23,26,27,135,140,146,168,171,172,176,177,180,196,199,200,203,206,211,217,229,235,240}
    for source in range(319):
        ok,im=cap.read();assert ok
        if source<79:continue
        action=source-78;i=action-1;canvas=np.full((720,1408,3),24,np.uint8);canvas[50:530,:640]=im
        for k,view in enumerate(('world_front','world_side')):
            render=cv2.imread(str(H.OUT/f'renders/{view}_f{action:04d}.png'));assert render is not None
            canvas[80:464,640+k*384:1024+k*384]=render
        def txt(s,y,x=10,scale=.54):cv2.putText(canvas,s,(x,y),0,scale,(235,235,235),1,cv2.LINE_AA)
        txt(f'S1 SOURCE {source} | Action {action} | {source/30:.3f}s',28)
        txt('A: original Blender, fixed world front',28,650,.48);txt('A: fixed world side',28,1040,.48)
        txt('No corrected candidate | source and original Action A | 30fps full 240 frames',558,scale=.65)
        txt(f"Rp sensitivity {rows[i]['sensitivity']['Rp']['operator_deg_per_mm']:.2f} deg/mm | Rt {rows[i]['sensitivity']['Rt']['operator_deg_per_mm']:.2f} deg/mm",590)
        txt('Pelvis measurement: '+q[i]['pelvis']['status']+' '+','.join(q[i]['pelvis']['reasons']),620,scale=.46)
        txt('Thorax measurement: '+q[i]['thorax']['status']+' '+','.join(q[i]['thorax']['reasons']),648,scale=.46)
        txt('Qualification is not 3D truth. Null measurements are not filled or held. Views are not calibrated source reprojections.',676,scale=.46)
        txt('Source: Taco fleur / Wikimedia Commons / CC BY-SA 4.0 | Agent still review; native playback not perceived',704,scale=.45)
        writer.write(canvas)
        if action in wanted:sheets[action]=canvas
    writer.release();cap.release()
    from w144_review_artifacts import fit_panel
    groups={'early_turn':[21,22,23,26,27],'quiet':[135,140,146],'parent_changes':[168,171,172,176,177,180],
        'boundary':[196,199,200,203],'occlusion':[206,211,217],'late':[229,235,240],'real_movement':[16,21,22]}
    for tag,ids in groups.items():
        canvas=np.full((len(ids)*240,960,3),24,np.uint8)
        for row,i in enumerate(ids):
            panel=sheets[i]
            for col,crop in enumerate((panel[70:530,180:460],panel[80:464,640:1024],panel[80:464,1024:1408])):
                canvas[row*240:(row+1)*240,col*320:(col+1)*320]=fit_panel(crop,320,240)
            cv2.putText(canvas,f'Action {i} / source {i+78}',(8,row*240+20),0,.5,(250,250,250),1)
        cv2.imwrite(str(review/f'{tag}.jpg'),canvas)
    cap=cv2.VideoCapture(str(movie));fps=cap.get(cv2.CAP_PROP_FPS);n=0
    while cap.read()[0]:n+=1
    cap.release();assert n==240 and fps==30
    H.write(W.OUT/'review_artifacts.json',{'movie':str(movie),'sha256':H.sha(movie),'decoded_frames':n,'fps':fps,'seconds':n/fps,
        'actual_Blender_views':'frozen W144 mesh renders of A; new W145 baseline verified bit-exact','candidate':None,
        'frame_mapping':'source=action+78','native_rate_perception':'BLOCKED 0xC0000142','human_acceptance':False,
        'source_sha256':H.sha(H.CLIP),'artifact_script_sha256':H.sha(__file__)})
    (review/'README.ko.md').write_text('''# W145 Body Frame 진단

PARTIALLY SUPPORTED: 기하학적 조건과 M1/S0/프레임 수식의 통제된 원인 추적에 한정합니다. 실제 Action 교정 개선은 입증하지 못했습니다.

S1_source_body_frame_diagnostics_30fps.mp4: 원본 + 고정 Blender A 두 시점, 240프레임/30fps/8초. 교정 후보가 없습니다. 실제 리그 뷰는 W144 원본 렌더를 그대로 사용하며 W145 새 Blender baseline도 일치했습니다.

QUALIFIED_MEASUREMENT는 정확한 3D 방향 인증이 아닙니다. 합성 장기 오방향은 여전히 통과합니다. UNQUALIFIED 값은 보간·유지로 채우지 않았습니다.

conditioning_and_attribution.png 및 synthetic_successes_and_failures.png에 성공과 실패가 함께 있습니다. JPG는 종횡비를 유지합니다. 원본 source=Action+78. Blender 뷰는 원본 카메라 재투영이 아닙니다.

직접 native-rate 관찰은 UI 커널 0xC0000142로 실패했습니다. 영상 디코딩/재생률 검증과 정지화면 검토만 수행했습니다. 사람의 수락을 주장하지 않습니다.

출처: Taco fleur / Wikimedia Commons / Double Kettlebell Strict Press (side view) / CC BY-SA 4.0. 로컬 파생 검토물이며 게시하지 않았습니다.
''',encoding='utf8')


if __name__=='__main__':report()
