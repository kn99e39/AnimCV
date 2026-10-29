#!/usr/bin/env python3
"""W102 isolated, same-environment observation handoff/evaluation. GT is used only by evaluate.

Reuses the behaviorally verified W101 adapter/statistical helpers and historical geometry metrics unchanged.
Historical prediction hashes are not substituted: B_Main is verified against its own locked report.
"""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"src"),str(ROOT/"scripts")]
import w101_reference_estimator as w101
import eval_real_observation_estimator as w100
from framepose.bank import load_bank
from run_w102_same_environment_s0 import CONDITIONS
from w102_reference import evaluation_reference,raw_reference_available,verify_benchmark_row


def distribution(values):
    x=np.asarray(values,float).ravel();x=x[np.isfinite(x)]
    return {"n":len(x),"mean":float(x.mean()) if len(x) else None,
            "quantiles":dict(zip(("min","p05","p25","p50","p75","p95","p99","max"),np.percentile(x,[0,5,25,50,75,95,99,100]).tolist())) if len(x) else {}}


def build(args,bank):
    payload=json.loads(args.population.read_text())
    source=json.loads((args.observe_dir/"manifest.json").read_text())
    if source["population_sha256"]!=w101.POPULATION_SHA:raise RuntimeError("STOP: sensor population changed")
    observations,side=w101.build_observations(bank,payload,args.observe_dir,args.image_root)
    expected={bank.samples[p].sequence_id.split(":")[1] for p in observations["positions"]}
    if set(source["videos"])!=expected:raise RuntimeError("STOP: incomplete source observation population")
    np.savez_compressed(args.out_dir/"observations.npz",**observations)
    np.savez_compressed(args.out_dir/"observations_side.npz",**side,positions=observations["positions"],
                        sample_id=observations["sample_id"],timestamp=observations["timestamp"])
    report={"schema":"w102_observation_handoff_v1","bank_content_digest":bank.content_digest(),
            "population_sha256":w101.sha(args.population),"source_manifest_sha256":w101.sha(args.observe_dir/"manifest.json"),
            "observations_sha256":w101.sha(args.out_dir/"observations.npz"),
            "side_sha256":w101.sha(args.out_dir/"observations_side.npz"),"actor_frames":len(observations["positions"]),
            "identity":dict(Counter(map(str,side["identity"]))),"GT_numerical_values_used":False}
    w101.dump(args.out_dir/"handoff_report.json",report)
    w101.dump(args.out_dir/"coverage.json",coverage_report(bank,observations,side,args.observe_dir))
    print(json.dumps(report),flush=True)


def coverage_report(bank,obs,side,observe_dir):
    all_actor=w101.population_report(bank,obs,side)
    output={"scope":"declared sparse bank cadence; one production-selected subject per video, not multi-person output",
            "association_is_evaluation_only":True,"original_actor_frame_accounting":all_actor,"splits":{}}
    by_image=defaultdict(list)
    for i,p in enumerate(obs["positions"]):by_image[str(side["image"][i])].append(i)
    for split in ("validation","test"):
        videos=sorted({bank.samples[p].sequence_id.split(":")[1] for p in obs["positions"] if bank.samples[p].split==split})
        records=[];per_video={}
        for video in videos:
            z=np.load(observe_dir/(video+".npz"),allow_pickle=False)
            meta=json.loads((observe_dir/(video+".json")).read_text())
            anchor=None;known_previous=None;gaps=[];gap_start=None;ious=[];reacquisition=[];transitions=[]
            vr=[];last_detected=None
            for k,image in enumerate(meta["images"]):
                rel=image.removeprefix("3dpw/");ii=by_image[rel];i=ii[0]
                detected=bool(np.isfinite(z["selected_box"][k]).all())
                assigned=str(side["assigned_actor"][i])
                status=("unique_valid_association" if assigned else {"ambiguous":"multiple_plausible","unresolved":"no_association",
                        "unmatchable":"insufficient_shared","no_subject":"missing_subject"}[str(side["identity"][i])])
                if anchor is None and assigned:anchor=assigned
                record={"image":rel,"frame_index":int(z["frame_index"][k]),"timestamp":float(z["timestamp"][k]),
                        "detected":detected,"any_detector_candidate":bool(np.any(z["cand_frame"]==k)),
                        "raw_landmarks_present":bool(np.isfinite(z["landmarks"][k,:,0]).any()),
                        "any_valid_joint":bool(obs["input_valid_M"][i].any()),"association":status,"actor":assigned,
                        "declared_actors":len(ii),"anchor_stable":bool(assigned and assigned==anchor)}
                if detected:
                    if k>0 and np.isfinite(z["selected_box"][k-1]).all():
                        ious.append(w100_iou(z["selected_box"][k-1],z["selected_box"][k]))
                    elif last_detected is not None:reacquisition.append(w100_iou(z["selected_box"][last_detected],z["selected_box"][k]))
                    last_detected=k
                if not detected and gap_start is None:gap_start=k
                if detected and gap_start is not None:
                    gaps.append({"start_frame":int(z["frame_index"][gap_start]),"end_frame":int(z["frame_index"][k-1]),"missing_samples":k-gap_start});gap_start=None
                # Only original neighboring observations with two unique assignments define an identity transition.
                if known_previous is not None and known_previous[0]==k-1 and assigned:
                    transitions.append({"from":known_previous[1],"to":assigned,"changed":known_previous[1]!=assigned,
                                        "frame_index":int(z["frame_index"][k])})
                known_previous=(k,assigned) if assigned else None
                vr.append(record);records.append(record)
            if gap_start is not None:
                gaps.append({"start_frame":int(z["frame_index"][gap_start]),"end_frame":int(z["frame_index"][-1]),"missing_samples":len(z["frame_index"])-gap_start})
            per_video[video]={"unique_images":len(vr),"detected":sum(r["detected"] for r in vr),
                              "any_detector_candidate_images":sum(r["any_detector_candidate"] for r in vr),
                              "association":dict(Counter(r["association"] for r in vr)),"evaluation_anchor_actor":anchor,
                              "anchor_stable_images":sum(r["anchor_stable"] for r in vr),"identity_transitions":transitions,
                              "adjacent_detected_box_iou":distribution(ious),"reacquisition_iou_separate":distribution(reacquisition),
                              "missing_detection_runs":gaps,"timestamps_sha256":__import__("hashlib").sha256(z["timestamp"].tobytes()).hexdigest()}
        n=len(records)
        output["splits"][split]={"unique_images":n,"detector_selected_images":sum(r["detected"] for r in records),
            "any_detector_candidate_images":sum(r["any_detector_candidate"] for r in records),
            "raw_landmark_images":sum(r["raw_landmarks_present"] for r in records),"any_valid_joint_images":sum(r["any_valid_joint"] for r in records),
            "association":dict(Counter(r["association"] for r in records)),"anchor_stable_images":sum(r["anchor_stable"] for r in records),
            "single_declared_actor_images":sum(r["declared_actors"]==1 for r in records),"per_video":per_video,"images":records}
    return output


def w100_iou(a,b):
    from run_real_observation_mmpose import iou
    return iou(a,b)


def representation_report(bank,obs,side,split):
    ii=np.array([i for i,p in enumerate(obs["positions"]) if bank.samples[p].split==split and side["identity"][i]=="same_person"],int)
    pp=obs["positions"][ii];b=bank.arrays["input_2d"][pp];vb=bank.arrays["input_valid"][pp]
    m=obs["input_2d_M"][ii];vm=obs["input_valid_M"][ii];both=vb&vm
    distances=[]
    per_joint=defaultdict(list)
    for j,p in enumerate(pp):
        size=bank.samples[p].image_size
        for k in w101.DIRECT:
            if both[j,k]:
                distance=float(np.linalg.norm((m[j,k,:2]-b[j,k,:2])*size))
                distances.append(distance);per_joint[w101.JOINTS[k]].append(distance)
    return {"same_person_actor_frames":len(ii),"joint_denominator":int(both.sum()),"limb_source_distance_px":distribution(distances),
            "limb_source_distance_px_per_joint":{k:distribution(v) for k,v in per_joint.items()},
            "benchmark_confidence_jointly_valid":distribution(b[:,:,2][both]),"M1_confidence_jointly_valid":distribution(m[:,:,2][both]),
            "validity_joint_counts":{"both":int(both.sum()),"B_only":int((vb&~vm).sum()),"M_only":int((vm&~vb).sum()),"neither":int((~vb&~vm).sum())},
            "missing_joints_B":dict(zip(w101.JOINTS,(~vb).sum(0).tolist())),"missing_joints_M1":dict(zip(w101.JOINTS,(~vm).sum(0).tolist()))}


def paired_temporal(rows,a="B",b="M"):
    from eval_external_hmr2_baseline import LIMBS,KNEES,ang
    from motion.torso_frame import torso_frames
    by={r["sample_id"]:r for r in rows if r["identity"]=="same_person" and r["tracked"] and a in r["J"] and b in r["J"]}
    events=defaultdict(list);counts=defaultdict(lambda:{"eligible":0,"B_events":0,"M_events":0})
    for first in by.values():
        second=by.get(first["next"])
        if second is None:continue
        ga,gb=first["G"],second["G"]
        ja,jb,ka,kb=first["J"][a],second["J"][a],first["J"][b],second["J"][b]
        def record(name,x,y):
            counts[name]["eligible"]+=1;counts[name]["B_events"]+=int(x);counts[name]["M_events"]+=int(y)
            events[name].append((first["sid"],int(y)-int(x)))
        for h,k in LIMBS.values():
            if ang(ga[k]-ga[h],gb[k]-gb[h])<15:
                record("limb_direction_jump",ang(ja[k]-ja[h],jb[k]-jb[h])>45,ang(ka[k]-ka[h],kb[k]-kb[h])>45)
        for h,k,end in KNEES.values():
            normals=[np.cross(x[k]-x[h],x[end]-x[k]) for x in (ja,jb,ka,kb,ga,gb)]
            if min(np.linalg.norm(v) for v in normals)>1e-3 and ang(normals[4],normals[5])<45:
                record("knee_plane_flip",ang(normals[0],normals[1])>90,ang(normals[2],normals[3])>90)
        frames=[torso_frames(x)[0] for x in (ja,jb,ka,kb,ga,gb)]
        if all(x is not None for x in frames) and ang(frames[4][:,1],frames[5][:,1])<45:
            record("torso_forward_flip",ang(frames[0][:,1],frames[1][:,1])>90,ang(frames[2][:,1],frames[3][:,1])>90)
    return {k:{**v,"paired_event_rate_difference":w101.video_bootstrap(*zip(*events[k]))} for k,v in counts.items()}


def failure_overlap(rows):
    """Observed shared failures support persistence, not a proof of unavoidable monocular ambiguity."""
    out={}
    for key in ("severe","knee_wrong_l","knee_wrong_r","torso_fwd_flip"):
        rr=[r for r in rows if key in r["m"]["B"] and key in r["m"]["M"]]
        b=np.array([bool(r["m"]["B"][key]) for r in rr],bool);m=np.array([bool(r["m"]["M"][key]) for r in rr],bool)
        out[key]={"common_support":len(rr),"both_fail":int((b&m).sum()),"B_only":int((b&~m).sum()),
                  "M_only":int((m&~b).sum()),"neither":int((~b&~m).sum())}
    return out


def evaluate(args,bank):
    import eval_external_hmr2_baseline as w97
    import run_pelvis_torso_ownership as w94
    from run_external_hmr2_baseline import smpl24_to_canonical17
    from run_external_orientation_handoff import W97_PREDICTIONS_SHA
    from motion.torso_frame import torso_frames
    obs=np.load(args.observations,allow_pickle=False)
    side=np.load(str(args.observations).replace(".npz","_side.npz"),allow_pickle=False)
    positions=w101.validate_population(bank,json.loads(args.population.read_text()))
    if not np.array_equal(positions,obs["positions"]) or not np.array_equal(positions,side["positions"]) or not np.array_equal(obs["sample_id"],side["sample_id"]):
        raise RuntimeError("STOP: evaluation handoff alignment failed")
    preflight=json.loads(args.preflight.read_text());s0=json.loads((args.s0_dir/"s0_report.json").read_text())
    if not preflight["pass"] or s0["environment_fingerprint"]!=preflight["environment_fingerprint"]:raise RuntimeError("STOP: uncontrolled execution")
    if s0["observations_sha256"]!=w101.sha(args.observations):raise RuntimeError("STOP: candidate observation artifact changed")
    predictions={}
    for condition,alias in CONDITIONS.items():
        predictions[alias]={}
        for split in ("validation","test"):
            path=args.s0_dir/f"prediction_{condition}_{split}.npy"
            if w101.sha(path)!=s0["conditions"][condition]["splits"][split]["sha256"]:raise RuntimeError("STOP: prediction changed")
            arr=np.load(path,allow_pickle=False)
            if arr.shape!=(len(bank.indices(split)),17,3):raise RuntimeError("STOP: prediction shape mismatch")
            predictions[alias].update(zip(map(int,bank.indices(split)),arr))
    predictions["E"]={}
    if args.hmr_predictions:
        if w101.sha(args.hmr_predictions)!=W97_PREDICTIONS_SHA:raise RuntimeError("STOP: frozen HMR2 identity failed")
        hp=np.load(args.hmr_predictions,allow_pickle=False);population=json.loads(args.population.read_text())["rows"]
        if len(population)!=len(hp["population"]):raise RuntimeError("STOP: HMR population size mismatch")
        for i,r in enumerate(population):
            if str(hp["population"][i])!=r["population"] or int(hp["index"][0][i])!=r["bank_position"]:raise RuntimeError("STOP: HMR identity mismatch")
            if r["population"].startswith("3dpw") and hp["valid"][i]:predictions["E"][r["bank_position"]]=smpl24_to_canonical17(hp["joints24"][i])
    coverage=coverage_report(bank,obs,side,args.observe_dir)
    anchor_images={r["image"]:r["anchor_stable"] for part in coverage["splits"].values() for r in part["images"]}
    rows=[];raw={};unavailable=[]
    for i,p in enumerate(positions):
        p=int(p);sample=bank.samples[p]
        uv,graw,gravity=evaluation_reference(args.raw_3dpw,sample,raw)
        verify_benchmark_row(bank,p,sample,raw)
        if not raw_reference_available(sample,raw,uv,graw,gravity):unavailable.append(sample.sample_id);continue
        g=bank.arrays["target_3d"][p].astype(float);g-=g[0]
        if not np.allclose(g,graw,atol=1e-6,rtol=0):raise RuntimeError("STOP: raw reference coordinate/frame mismatch")
        torso=torso_frames(g)[0];trunk=w94.angle(torso[:,2],gravity) if torso is not None else None
        thigh=.5*(w94.angle(g[2]-g[1],-gravity)+w94.angle(g[5]-g[4],-gravity))
        row={"p":p,"sid":sample.sequence_id,"sample_id":sample.sample_id,"split":sample.split,"frame":sample.frame_index,
             "timestamp":sample.timestamp,"next":sample.neighbors.get("next"),"identity":str(side["identity"][i]),
             "historical_all_target_mask":bool(bank.arrays["target_valid"][p].all()),
             "anchor_stable":bool(side["identity"][i]=="same_person" and anchor_images[str(side["image"][i])]),
             "tracked":bool(side["tracked"][i]),"stratum":w94.stratum(trunk,thigh),"G":g,"m":{},"J":{},"twod":{},
             "mean_conf_M":float(side["mean_conf_M"][i]),"moa":bool(side["moa"][i]),
             "bench_invalid_limb":int((~bank.arrays["input_valid"][p][w101.DIRECT]).sum()),"dist_M":float(side["dist_M"][i])}
        xb,vb=bank.arrays["input_2d"][p],bank.arrays["input_valid"][p];shared=vb&obs["input_valid_M"][i]
        for c in ("B","M","Moa"):
            x,v=(xb,vb) if c=="B" else (obs["input_2d_"+c][i],obs["input_valid_"+c][i])
            row["twod"][c]=w101.two_d_metrics(x,v,uv,sample.image_size)
            if c in ("B","M"):row["twod"][c+"_shared"]=w101.two_d_metrics(x,v,uv,sample.image_size,shared)
        for c in predictions:
            if p not in predictions[c]:continue
            j=np.asarray(predictions[c][p],float);j-=j[0]
            if not np.isfinite(j).all():raise RuntimeError("STOP: invalid 3D prediction")
            metrics=w97.source_metrics(j,g);metrics["per_joint"]=(1000*np.linalg.norm(j-g,axis=1)).tolist()
            errors,denominator=w100.depth_sign_errors(j,g)
            metrics.update(depth_sign_err=errors,depth_sign_n=denominator,severe=float(metrics["mpjpe"]>200))
            if denominator:metrics["depth_sign_rate"]=errors/denominator
            row["m"][c],row["J"][c]=metrics,j
        rows.append(row)
    report={"schema":"w102_same_environment_source_attribution_v1","environment_fingerprint":s0["environment_fingerprint"],
            "condition_aliases":{**{v:k for k,v in CONDITIONS.items()},"E":"frozen HMR2 contextual baseline"},
            "regimes":{"B":"benchmark_detector_observation","M":"real_animcv_observation; M1 batch_size=1",
                       "Moa":"privileged benchmark association","Mconf/Mmask":"evaluation-only representation interventions"},
            "input_hashes":{"observations":w101.sha(args.observations),"side":w101.sha(str(args.observations).replace(".npz","_side.npz")),
                            "population":w101.sha(args.population),"s0_report":w101.sha(args.s0_dir/"s0_report.json"),"preflight":w101.sha(args.preflight)},
            "coverage":coverage,"evaluation_GT_unavailable":unavailable,"results":{}}
    report["accuracy_denominator"]="raw campose_valid and finite aligned 3D GT, independent of benchmark 2D visibility"
    for split in ("validation","test"):
        rr=[r for r in rows if r["split"]==split];same=[r for r in rr if r["identity"]=="same_person"]
        result=w101.summarize(rr)
        historical=[r for r in same if r["historical_all_target_mask"]]
        result["historical_full_target_mask_subset"]={"all_actor_frames":sum(r["historical_all_target_mask"] for r in rr),
            "same_person_rows":len(historical),"B":w100.agg(historical,"B"),"M":w100.agg(historical,"M"),
            "M_minus_B":w101.paired_metrics(historical,"B","M")}
        result["representation"]=representation_report(bank,obs,side,split)
        result["paired_temporal_common_support"]=paired_temporal(same)
        result["failure_overlap_same_person"]=failure_overlap(same)
        anchor=[r for r in same if r["anchor_stable"]]
        result["anchor_stable_paired"]={"rows":len(anchor),"B":w100.agg(anchor,"B"),"M":w100.agg(anchor,"M"),
                                        "M_minus_B":w101.paired_metrics(anchor,"B","M")}
        result["per_joint_same_person"]={c:dict(zip(w101.JOINTS,np.mean([r["m"][c]["per_joint"] for r in same if c in r["m"]],axis=0).tolist()))
                                        for c in predictions if any(c in r["m"] for r in same)}
        result["tails_same_person"]={c:distribution([r["m"][c]["mpjpe"] for r in same if c in r["m"]]) for c in predictions}
        result["by_video"]={video:{"rows":len(vr),"B":w100.agg(vr,"B"),"M":w100.agg(vr,"M"),"paired":w101.paired_metrics(vr,"B","M")}
            for video,vr in group_by_video(same).items()}
        result["failure_records"]=sorted([{"sample_id":r["sample_id"],"dist_M":r["dist_M"],"stratum":r["stratum"],
              "B_mpjpe":r["m"]["B"]["mpjpe"],"M_mpjpe":r["m"]["M"]["mpjpe"],
              "delta_mm":r["m"]["M"]["mpjpe"]-r["m"]["B"]["mpjpe"]} for r in same],key=lambda r:-r["delta_mm"])[:30]
        report["results"][split]=result
    w101.dump(args.out_dir/"report.json",report)
    w101.dump(args.out_dir/"rows.json",rows)
    print(json.dumps({s:report["results"][s]["headline"] for s in report["results"]}),flush=True)


def group_by_video(rows):
    groups=defaultdict(list)
    for r in rows:groups[r["sid"].split(":")[1]].append(r)
    return groups


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument("mode",choices=("build","evaluate"))
    for key in ("bank","population","observe_dir","out_dir"):
        ap.add_argument("--"+key.replace("_","-"),type=Path,required=True)
    for key in ("image_root","observations","s0_dir","raw_3dpw","preflight","hmr_predictions"):
        ap.add_argument("--"+key.replace("_","-"),type=Path)
    args=ap.parse_args();args.out_dir.mkdir(parents=True,exist_ok=False)
    bank=load_bank(args.bank)
    if bank.content_digest()!=w101.BANK_DIGEST or w101.sha(args.population)!=w101.POPULATION_SHA:raise RuntimeError("STOP: frozen evaluation identity failed")
    (build if args.mode=="build" else evaluate)(args,bank)


if __name__=="__main__":main()
