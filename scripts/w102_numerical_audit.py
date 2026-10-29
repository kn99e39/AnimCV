#!/usr/bin/env python3
"""W102: separate historical byte identity, within-stack stability and semantic portability.

Read-only prediction audit. Metrics consume the existing W97/W100 geometry definitions; no inference or GT input
construction occurs here. Near-boundary changes are listed instead of silently relabeling bytes as reproduced.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"src"),str(ROOT/"scripts")]
from framepose.bank import load_bank
from motion.torso_frame import torso_frames,MIN_TRUNK
from run_real_observation_s0 import S0_PRED_SHA
from w101_reference_estimator import BANK_DIGEST,dump,sha
from eval_external_hmr2_baseline import LIMBS,KNEES,ang,source_metrics

LIMITS={"joint_mm":.1,"direction_deg":.05,"knee_plane_deg":.1,"torso_deg":.05}
BOUNDARIES={"angle_deg":.1,"position_mm":.1,"normal_m2":1e-5}


def safe_angle(a,b):
    if min(np.linalg.norm(a),np.linalg.norm(b))<=1e-12:return None
    return ang(a,b)


def summary(values):
    values=np.asarray([v for v in values if v is not None and np.isfinite(v)],float)
    return {"n":len(values),"mean":float(values.mean()) if len(values) else None,
            "p95":float(np.percentile(values,95)) if len(values) else None,"max":float(values.max()) if len(values) else None}


def knee(joints,h,k,a):
    upper,lower=joints[k]-joints[h],joints[a]-joints[k]
    normal=np.cross(upper,lower)
    return safe_angle(upper,lower),normal,float(np.linalg.norm(normal))


def torso_near_boundary(joints):
    trunk=joints[8]-joints[0]
    length=np.linalg.norm(trunk)
    if abs(length-MIN_TRUNK)*1000<=BOUNDARIES["position_mm"]:return True
    for a,b in ((11,14),(1,4)):
        angle=safe_angle(joints[b]-joints[a],trunk)
        if angle is not None and abs(min(angle,180-angle)-20)<=BOUNDARIES["angle_deg"]:return True
    return False


def compare_predictions(a,b,gt,samples=None,target_valid=None):
    original_a,original_b=np.asarray(a),np.asarray(b)
    bytes_equal=(original_a.dtype==original_b.dtype and original_a.shape==original_b.shape and original_a.tobytes()==original_b.tobytes())
    a,b,gt=np.asarray(a,float),np.asarray(b,float),np.asarray(gt,float)
    if a.shape!=b.shape or a.shape!=gt.shape or a.ndim!=3 or a.shape[1:]!=(17,3):raise ValueError("Prediction/reference shape mismatch")
    if not np.isfinite(a).all() or not np.isfinite(b).all():raise ValueError("Nonfinite prediction")
    n=len(a);target_valid=np.ones((n,17),bool) if target_valid is None else target_valid
    ids=[s.sample_id for s in samples] if samples is not None else [str(i) for i in range(n)]
    ar,br,gr=a-a[:,:1],b-b[:,:1],gt-gt[:,:1]
    continuous=defaultdict(list);decisions={};degenerate=defaultdict(int)
    def decide(name,identifier,old,new,near,margin=None):
        item=decisions.setdefault(name,{"eligible":0,"near_boundary":0,"changed":0,"outside_boundary":0,"examples":[]})
        item["eligible"]+=1;item["near_boundary"]+=int(near)
        if old!=new:
            item["changed"]+=1;item["outside_boundary"]+=int(not near)
            if len(item["examples"])<100:item["examples"].append({"sample":identifier,"old":old,"new":new,"near_boundary":bool(near),"margin":margin})
    for i,(p,q,g) in enumerate(zip(ar,br,gr)):
        sid=ids[i]
        for name,(h,k) in LIMBS.items():
            angle=safe_angle(p[k]-p[h],q[k]-q[h])
            if angle is None:degenerate["limb_"+name]+=1
            else:continuous["direction_"+name].append(angle)
            gv=g[k]-g[h]
            if target_valid[i].all() and np.linalg.norm(gv)>1e-12 and abs(gv[1])>=.25*np.linalg.norm(gv):
                pa,qa=(p[k]-p[h])[1],(q[k]-q[h])[1]
                margin=min(abs(pa),abs(qa))*1000
                decide("depth_sign_"+name,sid,int(np.sign(pa)),int(np.sign(qa)),margin<=.1,margin)
        bends={}
        for side,(h,k,end) in KNEES.items():
            fa,na,sa=knee(p,h,k,end);fb,nb,sb=knee(q,h,k,end)
            bends[side]=(fa,fb,sa,sb)
            well=fa is not None and fb is not None and min(fa,fb)>=20 and min(sa,sb)>1e-3
            if well:continuous["knee_plane_"+side].append(ang(na,nb))
            else:degenerate["knee_not_well_conditioned_"+side]+=1
            if fa is not None and fb is not None:
                margin=min(abs(fa-20),abs(fb-20))
                decide("knee_eligibility_"+side,sid,bool(fa>=20),bool(fb>=20),margin<=.1,margin)
        frames_a=torso_frames(p);frames_b=torso_frames(q)
        near_torso=torso_near_boundary(p) or torso_near_boundary(q)
        for name,ra,rb in zip(("thorax","pelvis"),frames_a,frames_b):
            decide("torso_valid_"+name,sid,ra is not None,rb is not None,near_torso)
            if ra is None or rb is None:degenerate["torso_"+name]+=1
            else:
                for axis in range(3):continuous[f"torso_{name}_axis{axis}"].append(ang(ra[:,axis],rb[:,axis]))
        if not target_valid[i].all() or not np.isfinite(g).all():
            degenerate["evaluation_GT_unavailable"]+=1;continue
        ma,mb=source_metrics(p,g),source_metrics(q,g)
        for key in ("mpjpe","depth_err","pa_mpjpe","torso_up","torso_fwd"):
            if key in ma and key in mb:continuous["score_difference_"+key].append(abs(ma[key]-mb[key]))
        margin=min(abs(ma["mpjpe"]-200),abs(mb["mpjpe"]-200))
        decide("severe_200mm",sid,bool(ma["mpjpe"]>200),bool(mb["mpjpe"]>200),margin<=.1,margin)
        for side in "lr":
            key="knee_wrong_"+side;fa,fb,sa,sb=bends[side]
            near_knee=(fa is not None and fb is not None and min(abs(fa-20),abs(fb-20))<=.1) or min(abs(sa-1e-3),abs(sb-1e-3))<=1e-5
            decide("knee_score_available_"+side,sid,key in ma,key in mb,near_knee)
            if key in ma and key in mb:
                margin=min(abs(ma["knee_plane_"+side]-90),abs(mb["knee_plane_"+side]-90))
                decide(key,sid,int(ma[key]),int(mb[key]),margin<=.1,margin)
        key="torso_fwd_flip"
        decide("torso_score_available",sid,key in ma,key in mb,near_torso)
        if key in ma and key in mb:
            margin=min(abs(ma["torso_fwd"]-90),abs(mb["torso_fwd"]-90))
            decide(key,sid,int(ma[key]),int(mb[key]),margin<=.1,margin)
    temporal_steps=0
    if samples is not None:
        index={s.sample_id:i for i,s in enumerate(samples)}
        for i,s in enumerate(samples):
            j=index.get(s.neighbors.get("next"))
            if j is None or not target_valid[[i,j]].all():continue
            if samples[j].timestamp<=s.timestamp:raise ValueError("Invalid temporal timestamps")
            temporal_steps+=1;identifier=ids[i]+" -> "+ids[j]
            pa,pb,qa,qb,ga,gb=ar[i],ar[j],br[i],br[j],gr[i],gr[j]
            for name,(h,k) in LIMBS.items():
                gtstep=safe_angle(ga[k]-ga[h],gb[k]-gb[h])
                if gtstep is None or gtstep>=15:continue
                sa=safe_angle(pa[k]-pa[h],pb[k]-pb[h]);sb=safe_angle(qa[k]-qa[h],qb[k]-qb[h])
                if sa is None or sb is None:continue
                margin=min(abs(sa-45),abs(sb-45))
                decide("temporal_limb_jump_"+name,identifier,sa>45,sb>45,margin<=.1,margin)
            for side,(h,k,end) in KNEES.items():
                ns=[np.cross(x[k]-x[h],x[end]-x[k]) for x in (pa,pb,qa,qb,ga,gb)]
                norms=[np.linalg.norm(v) for v in ns]
                if min(norms[4:])<=1e-3 or ang(ns[4],ns[5])>=45:continue
                va=min(norms[:2])>1e-3;vb=min(norms[2:4])>1e-3
                near=min(abs(v-1e-3) for v in norms[:4])<=1e-5
                decide("temporal_knee_available_"+side,identifier,va,vb,near)
                if va and vb:
                    sa,sb=ang(ns[0],ns[1]),ang(ns[2],ns[3]);margin=min(abs(sa-90),abs(sb-90))
                    decide("temporal_knee_flip_"+side,identifier,sa>90,sb>90,margin<=.1,margin)
            fs=[torso_frames(x)[0] for x in (pa,pb,qa,qb,ga,gb)]
            if fs[4] is not None and fs[5] is not None and ang(fs[4][:,1],fs[5][:,1])<45:
                va=all(f is not None for f in fs[:2]);vb=all(f is not None for f in fs[2:4])
                near=any(torso_near_boundary(x) for x in (pa,pb,qa,qb))
                decide("temporal_torso_available",identifier,va,vb,near)
                if va and vb:
                    sa,sb=ang(fs[0][:,1],fs[1][:,1]),ang(fs[2][:,1],fs[3][:,1]);margin=min(abs(sa-90),abs(sb-90))
                    decide("temporal_torso_flip",identifier,sa>90,sb>90,margin<=.1,margin)
    continuous={k:summary(v) for k,v in continuous.items()}
    positions=summary((1000*np.linalg.norm(a-b,axis=2)).ravel())
    root_relative=summary((1000*np.linalg.norm(ar-br,axis=2)).ravel())
    outside=sum(v["outside_boundary"] for v in decisions.values())
    def largest(prefix):return max((v["max"] for k,v in continuous.items() if k.startswith(prefix) and v["max"] is not None),default=0.)
    passed=(positions["max"]<=.1 and root_relative["max"]<=.1 and largest("direction_")<=.05 and largest("knee_plane_")<=.1 and largest("torso_")<=.05 and outside==0)
    return {"pass":bool(passed),"rows":n,"byte_equal":bytes_equal,"changed_scalars":int(np.count_nonzero(a!=b)),
            "joint_displacement_mm":positions,"root_relative_displacement_mm":root_relative,"continuous":continuous,
            "discrete":decisions,"outside_boundary_changes":outside,"degenerate":dict(degenerate),
            "temporal_steps":temporal_steps,"limits":LIMITS,"boundary_bands":BOUNDARIES}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ("bank","baseline0","baseline1","historical_dir","m1_gate","out"):
        ap.add_argument("--"+name.replace("_","-"),type=Path,required=True)
    ap.add_argument("--raw-3dpw",type=Path)
    args=ap.parse_args()
    if args.out.exists():raise RuntimeError("Refusing to overwrite audit")
    bank=load_bank(args.bank)
    if bank.content_digest()!=BANK_DIGEST:raise RuntimeError("Frozen bank identity failed")
    r0=json.loads((args.baseline0/"s0_report.json").read_text());r1=json.loads((args.baseline1/"s0_report.json").read_text())
    if r0["environment_fingerprint"]!=r1["environment_fingerprint"]:raise RuntimeError("Uncontrolled execution change")
    m1=json.loads(args.m1_gate.read_text())
    result={"schema":"w102_semantic_preflight_v1","environment_fingerprint":r0["environment_fingerprint"],
            "checkpoint_sha256":r0["checkpoint_sha256"],"bank_content_digest":BANK_DIGEST,"limits":LIMITS,"boundary_bands":BOUNDARIES,
            "baseline_report_sha256":[sha(args.baseline0/"s0_report.json"),sha(args.baseline1/"s0_report.json")],
            "m1_gate_sha256":sha(args.m1_gate),"splits":{},"pass":bool(m1["pass"])}
    raw_cache={}
    result["GT_decision_support"]="raw campose_valid, finite aligned GT" if args.raw_3dpw else "historical bank target_valid (includes benchmark visibility)"
    result["raw_reference_files_sha256"]={}
    for split in ("validation","test"):
        p0=args.baseline0/f"prediction_B_Main_{split}.npy";p1=args.baseline1/f"prediction_B_Main_{split}.npy"
        ph=args.historical_dir/f"prediction_{split}.npy"
        if sha(ph)!=S0_PRED_SHA[split]:raise RuntimeError("Historical prediction identity failed")
        for report,path in ((r0,p0),(r1,p1)):
            if sha(path)!=report["conditions"]["B_Main"]["splits"][split]["sha256"]:raise RuntimeError("Baseline output changed")
        pos=bank.indices(split);samples=[bank.samples[p] for p in pos]
        gt=bank.arrays["target_3d"][pos];valid=bank.arrays["target_valid"][pos]
        if args.raw_3dpw:
            from w102_reference import evaluation_reference,raw_reference_available,verify_benchmark_row
            valid=np.zeros_like(valid)
            for i,sample in enumerate(samples):
                uv,joints,gravity=evaluation_reference(args.raw_3dpw,sample,raw_cache)
                verify_benchmark_row(bank,int(pos[i]),sample,raw_cache)
                available=raw_reference_available(sample,raw_cache,uv,joints,gravity)
                if available and not np.allclose(gt[i]-gt[i,0],joints,atol=1e-6,rtol=0):
                    raise RuntimeError("STOP: raw GT coordinate/frame mismatch")
                valid[i]=available
            for video in {s.sequence_id.split(":")[1] for s in samples}:
                path=args.raw_3dpw/"sequenceFiles"/split/(video+".pkl")
                result["raw_reference_files_sha256"][str(path)]=sha(path)
        a,b,h=np.load(p0),np.load(p1),np.load(ph)
        repeat=compare_predictions(a,b,gt,samples,valid)
        historical=compare_predictions(h,a,gt,samples,valid)
        result["splits"][split]={"same_environment_repeat":repeat,"historical_portability":historical,
                                  "historical_sha256":sha(ph),"B_Main_sha256":sha(p0),"historical_byte_identity":sha(ph)==sha(p0)}
        result["pass"] &= repeat["pass"] and historical["pass"]
        print(split,"repeat_byte_equal",repeat["byte_equal"],"semantic_portability",historical["pass"],
              "max_mm",historical["joint_displacement_mm"]["max"],"outside_boundary",historical["outside_boundary_changes"],flush=True)
    dump(args.out,result)
    if not result["pass"]:raise RuntimeError("STOP: meaningful same-environment/semantic preflight condition failed")


if __name__=="__main__":main()
