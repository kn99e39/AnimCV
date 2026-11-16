"""W144 scratch-only S1 trace. CPU and Blender, no model inference or production writes.

Stages refuse existing evidence. A frozen declaration precedes all official
measurements. Run from repo root with .venv_build_full/Scripts/python.exe.
"""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'),str(ROOT/'scripts')]
import numpy as np
from app import experimental_scene_anchored_trace as T

RUN = Path('C:/Users/dna10/animcv-w126/runs/S1_replay')
W143 = Path('C:/Users/dna10/animcv-w143')
OUT = ROOT/'result/w144'
OBS = RUN/'perception/m1s0/observations_predictions.npz'
BLEND = RUN/'baseline/baseline.blend'
CLIP = Path('C:/Users/dna10/animcv-w121/data/originals/64498918_Double_Kettlebell_Strict_Press_side_view_.webm')
CAM = W143/'results/estimate/w143_s1/estimate.json'
DECL = ROOT/'examples/experimental_scene_anchored/w144_declaration.json'
INSTRUMENTS = ['src/app/experimental_scene_anchored_trace.py','scripts/w144_blender_trace.py',
               'scripts/w144_harness.py','tests/test_w144_scene_anchored_trace.py']


def load(p):
    return json.loads(Path(p).read_text('utf8'))


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()


def write(p,d):
    p=Path(p)
    if p.exists(): raise RuntimeError(f'refusing overwrite: {p}')
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(d,indent=2,allow_nan=False),encoding='utf8')


def git(*args):
    return subprocess.check_output(['git','-C',str(ROOT),*args],text=True).strip()


def preflight():
    import cv2
    import w129_motionbricks_harness as H
    s=H.SOURCES['S1']; man=load(RUN/'manifest/input_manifest.json')
    side=load(RUN/'baseline/action_sidecar.json')
    checks={}
    def bind(p,expected):
        p=Path(p);got=sha(p)
        checks[str(p)]={'expected':expected,'sha256':got,'ok':got==expected}
    bind(CLIP,s['clip_sha']);bind(BLEND,s['baseline_sha'])
    bind(RUN/s['pin_accepted'],s['pin_accepted_sha'])
    for rel,h in load(RUN/'run.json')['immutable_files'].items(): bind(RUN/rel,h)
    bind(OBS,man['perception']['observation_prediction_sha256'])
    bind(RUN/'perception/m1s0/report.json',man['perception']['report_sha256'])
    for rec in man['rig']['files'].values(): bind(ROOT/rec['path'],rec['expected'])
    for rel,rec in man['frozen_code'].items():
        p=ROOT/rel; got=hashlib.sha256(p.read_bytes().replace(b'\r\n',b'\n')).hexdigest()
        checks[str(p)]={'expected':rec['expected'],'sha256_lf':got,'ok':got==rec['expected']}
    # Frozen manifest anchored to W143's recorded Worklog identity.
    bind(W143/'results/evidence_manifest.json','d6b1a1bb529e7fb2eb3add0059ed7f76b189d06c357858557aa17c9908c6c3f8')
    cm=load(W143/'results/evidence_manifest.json')['files']
    for rel,h in cm.items():
        if 'w143_s1/' in rel or rel=='results/declare/declaration.json': bind(W143/rel,h)
    # Verify historical rig outputs against their own frozen manifests, without
    # reading any generator weights or modifying any history.
    for w in (129,136,137,138,139):
        base=Path(f'C:/Users/dna10/animcv-w{w}')
        mp=base/'results/evidence_manifest.json';d=load(mp)
        checks[str(mp)]={'sha256':sha(mp),'ok':True,'role':'inventory bound at W144 preflight; not externally authenticated'}
        rows=d.get('sha256',d.get('files'))
        for rel,rec in rows.items():
            if rel.endswith('.blend') or rel.endswith('declaration.json') or rel.endswith('assessment.json'):
                p=base/rel if rel.startswith('results/') else base/'results'/rel
                bind(p,rec['sha256'] if isinstance(rec,dict) else rec)
    # Preserve the entire original S1 run and all selected historical manifests.
    inventory={str(p):sha(p) for p in RUN.rglob('*') if p.is_file()}
    for w in (125,129,136,137,138,139,140,141,142,143):
        p=Path(f'C:/Users/dna10/animcv-w{w}/results/evidence_manifest.json')
        if p.exists():inventory[str(p)]=sha(p)
    for p in [CLIP,ROOT/'docs/next_batch.md',ROOT/'scripts/batch_automation.py']:
        inventory[str(p)]=sha(p)
    z=np.load(OBS)
    frame_ok=np.array_equal(z['frame_index'],np.arange(1631)) and np.array_equal(z['timestamp'],np.arange(1631)/30)
    mapping_ok=side['frame_mapping']==[{'source_frame':i,'source_time_s':i/30,'action_frame':i-78,
        'action_time_s_at_scene_fps':(i-79)/30} for i in range(79,319)]
    cap=cv2.VideoCapture(str(CLIP));decoded=[];timestamps=[];sizes=[]
    for i in range(319):
        ok,im=cap.read()
        if not ok:raise RuntimeError('source decode failed')
        if i>=79:
            decoded.append(hashlib.sha256(im.tobytes()).hexdigest());timestamps.append(cap.get(cv2.CAP_PROP_POS_MSEC));sizes.append(list(im.shape))
    cap.release()
    decode_ok=decoded==man['intake']['decoded_BGR_uint8_sha256'][79:319]
    report={'schema':'animcv_w144_preflight_v1','time':dt.datetime.now(dt.timezone.utc).isoformat(),
        'head':git('rev-parse','HEAD'),'branch':git('branch','--show-current'),
        'remote_difference':git('rev-list','--left-right','--count','HEAD...origin/Dual_Stage_Generator'),
        'git_status':git('status','--short'),'checks':checks,'preservation_inventory':inventory,
        'frame_mapping_ok':mapping_ok,'observation_time_ok':bool(frame_ok),'decoded_bytes_ok':decode_ok,
        'container_timestamp_max_delta_ms':float(np.max(np.abs(np.array(timestamps)-man['intake']['container_pos_msec_segment']))),
        'source_size':sizes[0], 'regime':'real_animcv_observation',
        'available_intermediates':list(z.files),'missing':['metric camera position','calibrated intrinsics','metric human root depth','pelvis bone','verified contact'],
        'camera':{k:load(CAM)[k] for k in ('schema','camera','sfm_validity','n_frames')},
        'blender':load(RUN/'run.json')['blender']}
    report['ok']=all(v['ok'] for v in checks.values()) and mapping_ok and frame_ok and decode_ok and report['branch']=='Dual_Stage_Generator' and report['remote_difference']=='0\t0'
    write(OUT/'preflight_verified.json',report)
    print('PREFLIGHT',report['ok'],'identities',len(checks),'preserved',len(inventory))
    if not report['ok']:raise RuntimeError('STOP preflight identity/contract failure')


def declare():
    pf=load(OUT/'preflight_verified.json')
    assert pf['ok']
    d={'schema':'animcv_w144_prospective_declaration_v1','date':'2026-11-16','head':git('rev-parse','HEAD'),
       'preflight_sha256':sha(OUT/'preflight_verified.json'),'branch':'Dual_Stage_Generator',
       'outcomes_seen':False,'disclosure':'Historical W129/W136-W143 results known; schema and source inspected, no W144 trajectory metrics or corrections evaluated before declaration. Initial preflight compared 240 decoded frames against all 1631 stored hashes (harness range bug); retained failed preflight.json, corrected to source[79:319] before declaration. No source bytes changed.',
       'regimes':{'S1':'real_animcv_observation; frozen M1 RTMDet/RTMPose + S0 learned 3D, GT-free replay','synthetic':'oracle_geometry'},
       'budget':{'primary_cpu_blender_seconds':7200,'scratch_bytes':8*1024**3,'candidate_max':1,'gpu':False,'training':False},
       'frames':{'source':[79,318],'action':[1,240],'fps':30,'rule':'action=source-78; action time=(action-1)/30; source nominal time=source/30; container quantization separately reported'},
       'metrics':{'position':'Euclidean step and second difference (px observations; metres evaluated rig); acceleration second difference *900, speed step *30; no velocity-only error classification',
                  'rotation':'SO(3) geodesic active local-to-parent rotation increment in degrees; object scale removed',
                  'source':'raw landmarks xy pixel and detector confidence >=0.3; canonical virtual pelvis/thorax marked derived; unavailable or self-occluded never treated as GT',
                  'continuity':'all 239 edges; 0.25/0.5/0.75/0.999 subframe samples; CONSTANT expected holds',
                  '2d_jumps':'displacement >20 px/frame with adjacent valid confidence; flag only, not proof of error',
                  'geometry':'recompute frozen fixed_slot for every prediction; Python rest-relative FK vs Blender heads/tails <=1e-5 m; local/parent parity; length constancy',
                  'camera':'W143 frozen NOMINAL_UNCALIBRATED, no metric translation; report image-space background flow separately; no S1 reprojection fidelity gate'},
       'selection':{'fixed':[[1,12],[163,208],[168,175],[196,203],[229,240]],
                    'baseline':'top 5 left shoulder step destination frames, ties earlier; +/-3 frames; quiet=12-frame window with smallest summed left shoulder steps; large source motion=12-frame window with largest confident wrist pixel path; uncertain=12-frame window with most invalid input joints; ties earlier'},
       'counterfactual':'for EVERY edge, hold one factor at previous value in current state; inject current factor into previous state; full actual Blender FK; factors object translation/rotation, spine_01, clavicles, left arm, thighs. Joint object+spine intervention and both conditional orders, never unique causal percentages. No saved diagnostic Actions.',
       'synthetic':'known camera/root transforms, torso articulation, upstream pose jump and duplicated-parent defect; numerical oracle only; no oracle S1 values',
       'candidate_rule':'At most one, only if identity/frame/axis/composition defect is demonstrated against frozen intended contract OR independent source/scene evidence constrains the corrective state without invented depth or pose. No correction based only on reduced steps; no smoothing/freeze as repair. If no supported correction, skip explicitly.',
       'candidate_gates':['verified mechanism and independent source constraint','new reversible copy, no historical change','FK <=1e-5 m','source action/timing preserved','no new source-inconsistent motion','identified discontinuity reduced','uncertainty explicit'],
       'judgment':'SUPPORTED only mechanism + source-consistent real Blender improvement and all candidate gates; PARTIALLY_SUPPORTED partial attribution/improvement with unresolved questions; NOT_YET_SUPPORTED no useful source-consistent camera-anchored improvement; BLOCKED prerequisite failure. Diagnostic attribution separate from correction.',
       'instrument_sha256_lf':{f:hashlib.sha256((ROOT/f).read_bytes().replace(b'\r\n',b'\n')).hexdigest() for f in INSTRUMENTS}}
    write(DECL,d);print('DECLARATION',sha(DECL))


def verify():
    d=load(DECL)
    assert sha(OUT/'preflight_verified.json')==d['preflight_sha256']
    for f,h in d['instrument_sha256_lf'].items():
        assert hashlib.sha256((ROOT/f).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==h, f
    return d


def trace():
    verify();pf=load(OUT/'preflight_verified.json')
    t=time.monotonic()
    args=[pf['blender'],'-b','--factory-startup','--python-exit-code','1','-P',str(ROOT/'scripts/w144_blender_trace.py'),
          '--','--blend',str(BLEND),'--out',str(OUT/'blender_trace.json')]
    r=subprocess.run(args,capture_output=True,text=True,encoding='utf8',errors='replace',timeout=7200)
    write(OUT/'trace_execution.json',{'args':args,'rc':r.returncode,'seconds':time.monotonic()-t,'stdout':r.stdout,'stderr':r.stderr})
    print('TRACE',r.returncode,round(time.monotonic()-t,2))
    if r.returncode:raise RuntimeError(r.stdout+r.stderr)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('stage',choices=['preflight','declare','trace'])
    globals()[ap.parse_args().stage]()
