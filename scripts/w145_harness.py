"""W145 prospective contract and source identities. No branch operations or model training."""
from __future__ import annotations
from pathlib import Path
import sys,json,hashlib,datetime
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
import w144_harness as H

OUT=ROOT/'result/w145'
DECL=ROOT/'examples/experimental_body_frame/w145_declaration_v2.json'
MODEL=Path('C:/Users/dna10/animcv-w101/data/s0_checkpoint.pt')
W144_MANIFEST_SHA='94288096c4d7b99dd9d17bfb016bae22bcd73c34bd1e2189eab9fc8e60b4fffd'
INSTRUMENTS=['scripts/w145_body_frame_core.py','scripts/w145_harness.py','tests/test_w145_body_frame_attribution.py']


def verify():
    d=H.load(DECL)
    for f,h in d['instrument_sha256_lf'].items():
        assert hashlib.sha256((ROOT/f).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==h,f
    assert H.sha(H.OBS)==d['identities']['observations_sha256']
    assert H.sha(MODEL)==d['identities']['model_sha256']
    return d


def declare():
    assert H.git('branch','--show-current')=='Dual_Stage_Generator'
    assert H.sha(H.OUT/'evidence_manifest.json')==W144_MANIFEST_SHA
    wm=H.load(H.OUT/'evidence_manifest.json')
    for f,r in wm['files'].items():assert H.sha(ROOT/f)==r['sha256'],f
    assert H.sha(MODEL)=='6c12fea9f84ea0493fada8b9acc96cd5c70fdd5d9114603e954f9412f4247b44'
    pf=H.load(H.OUT/'preflight_verified.json')
    for p,h in pf['preservation_inventory'].items():assert H.sha(p)==h,p
    d={'schema':'animcv_w145_prospective_declaration_v1','date':'2026-11-16','time_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'branch':'Dual_Stage_Generator','starting_head':'e001952bc42e6711b4582ae87e1ae0f010336a9e',
        'W144_preservation_commit':'53361a9c6de737910817a97d8ffbc159c8203723','head_at_declaration':H.git('rev-parse','HEAD'),
        'W144_source_files_unchanged':wm['source_changes'],'W144_verified_evidence_files':len(wm['files']),
        'W144_manifest_sha256':W144_MANIFEST_SHA,'historical_inventory_files':len(pf['preservation_inventory']),
        'identities':{'observations_sha256':H.sha(H.OBS),'model_sha256':H.sha(MODEL),'original_A_sha256':H.sha(H.BLEND),'source_sha256':H.sha(H.CLIP)},
        'regimes':{'S1':'real_animcv_observation, frozen M1/S0 replay','synthetic':'oracle_geometry pose-token controls, not rendered perception'},
        'outcomes_seen':False,'supersedes_declaration_sha256':'fdf06c4c74d72475804e5b76d9768086aa9789c4acfc1c9fefb24cfe69926782',
        'disclosures':['W144 results and source sheets already known; no W145 official model output, conditioning statistics or synthetic-sequence outcomes yet.',
            'First declaration followed a unit run with 15 pass / 1 fail: a true 30deg turn from frontal projection moved hips less than the declared 2px support cutoff, so it abstained. Algorithm, thresholds and experiment gates are UNCHANGED. The clear-turn unit fixture now starts obliquely; a separate test explicitly requires abstention for the ambiguous frontal 30deg turn. Original declaration is retained.',
            'No source-wide constant side-facing assumption. Diagnostic qualification emits null instead of propagating a held pose. It is not a repair or full temporal orientation estimator.'],
        'frames':{'source':[79,318],'action':[1,240],'fps':30,'mapping':'action=source-78'},
        'windows':[[1,12],[11,22],[21,27],[135,146],[168,180],[196,203],[206,217],[229,240]],
        'perturbations':{'3d_joints':[0,1,4,8,11,14],'3d_mm':[1,2,5],'directions':'one joint, one Cartesian axis, both signs; all 240 frames',
            'analytic':'differentiate normalization/projection/cross product; central finite difference epsilon=1e-6 m; report operator spectral norm in deg/mm, conditioning and invalidity separately',
            '2d_direct_joints':[1,4,11,14],'2d_px':[1],'directions_2d':'each direct joint, x/y, +/-; all 240 frames, fixed cached crop vs recomputed crop; derived midpoints updated',
            'selected_model_actions':[16,22,23,26,27,140,171,172,177,200,211,240],
            'source_interventions':['previous hip xy only','previous shoulder xy only','forward/back-valid LK texture endpoints for hips/shoulders only'],
            'texture':'frozen W144 LK diagnostics: 21px, 3 levels, 30 iterations, epsilon .01, min eigen1e-4, FB<=1px; texture is not anatomical GT'},
        's0_cpu':{'device':'cpu','threads':4,'batch_size':512,'dtype':'float32','amp':False,'signs':'neutral_sign_states','weights_unchanged':True,
            'repeat_max_abs_m':1e-7,'cached_max_abs_m':1e-5,'cached_frame_angle_deg':.05,
            'if_mismatch':'retain diagnostic response as CPU numerical variant; do not claim exact frozen-output attribution; no S1 correction eligibility from incompatible model'},
        'qualifier':{'purpose':'single frozen measurement-qualification diagnostic; no repaired matrices for unqualified frames',
            'sigma_m':.002,'max_condition_deg':5.,'texture_max_px':3.,'unsupported_turn_deg':10.,'quiet_source_px':2.,'confidence':.3,
            'logic':'geometric sensitivity / original frame validity / input visibility / texture disagreement / unconfirmed large depth turn on quiet source; no max turn speed, smoothing or hold'},
        'synthetic':{'frames_each':90,'seed':145,'cases':['stationary','articulated_arms','gradual_turn_180','rapid_turn_90','false_3d_flip_1_and_10_frames','observation_jump','missing_and_LR_swap','depth_ambiguity_degeneracy','pelvis_fixed_torso_twist','pelvis_fixed_torso_bend'],
            'projection':'orthographic diagnostic image x=320+100*X, y=240-100*Z; camera stationary. Pose-token controls; simulated texture is oracle-projected point motion, explicitly not rendered/perceived.',
            'truth':'known pelvis orientation and independent torso orientation; no synthetic root/GT supplied to authentic S1'},
        'gates':{'synthetic_stationary_error_max_deg':.1,'synthetic_clear_turn_p95_deg':1.,'synthetic_clear_coverage':.95,
            'false_flip_qualification_rejection':.9,'degenerate_abstention':.95,'recovery_frames':2,'pelvis_torso_max_error_deg':1.,
            'geometry_FK_tolerance_m':1e-5,'real_thresholds':'full 240 source identity; no new unexplained flip; preserve genuine motion/source timing; source-visible evidence required independently, not displacement-only'},
        'candidate_eligibility':'No real-rig corrected candidate unless diagnosis AND all appropriate synthetic turn/false-flip/ambiguity/pelvis-torso controls support a specific independently source-constrained replacement. Measurement abstention alone is not an Action. At most one candidate; any algorithm/parameters declared in an additive prospective protocol BEFORE its outcomes, no sweeps.',
        'budget':{'primary_seconds':7200,'scratch_bytes':8*1024**3,'new_training':0,'GPU_experiments':0,'new_downloads':0,'real_candidate_max':1},
        'classification':'SUPPORTED: validated mechanism + useful source-consistent real Blender candidate; PARTIALLY SUPPORTED: useful conditioning/attribution but fidelity/generalization/observability unresolved; NOT YET SUPPORTED: tested approach does not establish reliable improvement; BLOCKED: prerequisite prevents valid experiment.',
        'instrument_sha256_lf':{f:hashlib.sha256((ROOT/f).read_bytes().replace(b'\r\n',b'\n')).hexdigest() for f in INSTRUMENTS}}
    H.write(DECL,d);print('W145_DECLARATION',H.sha(DECL))


if __name__=='__main__':declare()
