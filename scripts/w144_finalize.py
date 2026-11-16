"""Seal W144 judgment and preservation evidence; no history or production writes."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
import w144_harness as H
from app import experimental_scene_anchored_trace as T


def main():
    H.verify();pf=H.load(H.OUT/'preflight_verified.json');a=H.load(H.OUT/'analysis.json')
    differences=[]
    for p,h in pf['preservation_inventory'].items():
        if H.sha(p)!=h:differences.append(p)
    for p,rec in pf['checks'].items():
        if 'expected' not in rec:continue
        got=H.sha(p) if 'sha256' in rec else __import__('hashlib').sha256(Path(p).read_bytes().replace(b'\r\n',b'\n')).hexdigest()
        if got!=rec['expected']:differences.append(p)
    head=H.git('rev-parse','HEAD');branch=H.git('branch','--show-current')
    assert head==pf['head'] and branch=='Dual_Stage_Generator'
    assert not differences,differences
    assert not H.git('diff','--name-only'), 'pre-existing tracked files unexpectedly changed'
    assert H.git('rev-list','--left-right','--count','HEAD...origin/Dual_Stage_Generator')=='0\t0'
    camera=H.load(H.CAM);sc=H.load(H.OUT/'source_controls.json')
    tests=(H.OUT/'tests_utf8.txt').read_text('utf-8-sig')
    assert '145 passed' in tests
    reviews=ROOT/'user_eval/W144_scene_anchored'
    bytes_out=sum(p.stat().st_size for d in (H.OUT,reviews) for p in d.rglob('*') if p.is_file())
    result={'schema':'animcv_w144_judgment_v1','date':'2026-11-16','start_revision':pf['head'],'end_revision':head,
        'branch':branch,'remote_difference':'0 0','commits_created':False,'published':False,
        'classification':'NOT YET SUPPORTED',
        'diagnostic_attribution':'SUPPORTED for inherited Object/spine parent mechanism; partial upstream observation-vs-lifting attribution',
        'conditional_correction':'NOT PRODUCED; no sufficiently constrained source-consistent 3D correction',
        'causal_findings':[
            'No camera interpretation is used in the authentic S1 source-to-Action path. No metric root translation is inferred; object location is unkeyed zero.',
            'The cached 2D observations already contain landmark jumps inconsistent with local source texture; cached S0 3D also changes bilateral hip/shoulder geometry and trunk direction.',
            'Frozen S0 -> W93 pelvis/thorax frames -> W105/W121 fixed_slot -> Object/spine_01 rotations reproduces the historical Action on all 240 frames.',
            'Actual-Blender endpoint hold/inject tests identify Object rotation and spine_01 jointly as sufficient and necessary-as-a-set for shoulder-head steps (float tolerance); individual factors interact.',
            'Rest-relative FK matches actual Blender; no evidence of axis double application, parent-compounding defect, evaluation drift or translation discontinuity. CONSTANT interpolation holds values but causes stepped continuous-time playback.',
            'Retargeting imports the inferred orientation about the Armature origin and spine head; rigid body lever arms generate pelvis/shoulder translation even with zero Object travel. Missing placement/pivot ownership is an implemented limitation, not proof of a specific faulty line.',
            'Source supports a genuine initial clean and later press, so freezing the body or smoothing curves is not an evidence-supported repair.'
        ],
        'preservation':{'inventory_files':len(pf['preservation_inventory']),'identity_checks':len(pf['checks']),
                        'differences':differences,'scope':'entire W126 S1 run, source, selected historical rig outputs, evidence manifests and frozen code; not a full rehash of all historical model weights'},
        'tests':{'focused_and_reused':'145 passed in 9.92s, Python -X utf8',
                 'first_regression_attempt':'143 pass / 2 cp949 decoding failures in unchanged historical source-scanning tests; same suite passed in UTF-8 without code changes',
                 'production_full':'NOT RUN; isolated experiments only'},
        'qualitative':{'agent_still_review':True,'native_rate_file_verified':True,'native_rate_perceptual_review':'BLOCKED',
            'blocker':'node_repl initialization failed 0xC0000142 before any UI access; reset/retry and cua_repl inventory failed the same way',
            'human_acceptance':False,'candidate_AB':'NOT APPLICABLE',
            'authoritative_contact_sheets':str(reviews/'sheets_aspect_preserved'),
            'still_findings':[
                'Action 24/26/27: source torso remains side-on while the evaluated rig turns toward and away from its diagnostic camera; stable anatomical facing is not retained.',
                'Action 135/140/146 quiet window: body appearance is broadly similar across sampled source and rig frames, but far-arm and global-posture fidelity remain unresolved.',
                'Action 11/16/22: real source knee/trunk/arm changes occur during the clean; neither world-motion peaks nor every parent change can be labelled jitter.',
                'Action 174/176/177/180: source press preparation continues with small apparent torso changes; baseline changes global lean/facing, with 244mm shoulder step at 176->177.',
                'Action 206/211/217: source arms move overhead/down; missing detector far-elbow/wrist evidence and an unanimated rig right arm limit full-body fidelity.'
            ]},
        'camera':{'adjacent_flow_median_px':float(np.median([e['flow_raw_median_px'] for e in camera['edges'] if e['span']==1 and e.get('flow_raw_median_px') is not None])),
            'nominal_orientation_coverage':238,'metric_position':None,'intrinsics':'NOMINAL_UNCALIBRATED',
            'rotation_only_reference':H.load(H.OUT/'camera_reference.json'),
            'capability_context':'W140-W143 moving-scene relative 6DoF and near-static orientation evidence preserved; this judgment is about S1 human correction, not rejection of those capabilities.'},
        'candidate_skip_justification':[
            'No source/action mapping, axis-conversion, FK composition or retargeting identity defect was found against the frozen intended contract.',
            'LK texture consistency does not identify the anatomical shoulder under occlusion, repair every missing landmark, or supply human depth/scale.',
            'Replacing cached 3D values or holding parent transforms is a causal intervention, not a validated source-consistent reconstruction method.',
            'A constant camera or nominal W143 rotation does not provide missing metric placement; introducing an arbitrary pelvis pivot/foot lock would impose a new unverified world trajectory.'
        ],
        'resources':{'scratch_bytes_before_final_report':bytes_out,'ceiling_bytes':8*1024**3,
            'trace_seconds':H.load(H.OUT/'trace_execution.json')['seconds'],
            'render_seconds':H.load(H.OUT/'render_execution.json')['seconds'],'analysis_seconds':a['seconds'],
            'primary_ceiling_seconds':7200,'GPU_experiments':0,'training':0,'generator_inference':0,'candidates':0},
        'integrity_disclosures':['Initial preflight range-check bug corrected before declaration; failed report retained.',
            'Supplemental LK protocol added after baseline outcomes, explicitly not a prospective correction gate.',
            'Initial analysis synthetic duplicate-application wording corrected by source_controls.json: actual injected Object-Z plus local-spine-Y perturbation.',
            'Initial stretched contact sheets retained; use aspect-preserved sheets. Native-rate movie has original source aspect ratio.']}
    assert bytes_out<8*1024**3
    H.write(H.OUT/'final_verdict.json',result)
    print('W144',result['classification'],'preserved',len(pf['preservation_inventory']),'differences',len(differences),'scratch_MB',round(bytes_out/1e6,2))


if __name__=='__main__':main()
