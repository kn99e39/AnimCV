#!/usr/bin/env python3
"""W102 conditional, mixed-source BaseRig diagnostic; historical solvers/renderer remain unchanged.

Both modes share frozen HMR pelvis/thorax and frozen J1X left arm (right arm rest). Only leg geometry changes.
GT and association labels select/annotate diagnostic windows, never determine rig rotations.
Unavailable renders are blanked in the review composite; a held Blender pose is never presented as a valid output.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import sys

import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"src"),str(ROOT/"scripts")]

SLOTS={"FIXED":"B","RUNTIME":"M"}
FIXED=(("walking","3dpw:downtown_walkBridge_01:actor0",333),
       ("seated","3dpw:flat_guitar_01:actor0",500),
       ("wrong_knee","3dpw:downtown_bar_00:actor1",367),
       ("turn","3dpw:downtown_cafe_00:actor1",1021))


def context(rest_path,calibration,chains):
    import run_pelvis_frame_leg_articulation as w95
    from motion.torso_frame import quaternion_to_matrix
    from retarget.framepose_fk import derive_rig_alignment,load_rig_fk_calibration
    from retarget.framepose_target_rest import load_target_rest_pose
    from retarget.framepose_two_bone_ik import load_ik_chain_set,target_chain_geometry
    rest=load_target_rest_pose(rest_path);alignment=derive_rig_alignment(rest,load_rig_fk_calibration(calibration))
    sb=rest.bones["spine_01"];basis=np.asarray(sb.matrix_local,float)[:3,:3]
    return {"rest":rest,"alignment":alignment,"Am":quaternion_to_matrix(alignment),"legs":w95.leg_targets(rest),
        "spine_basis":basis/np.linalg.norm(basis,axis=0,keepdims=True),"spine_head":list(sb.head),
        "target":target_chain_geometry(rest,load_ik_chain_set(chains).chains[0])}


def pose_export(j,Ro,Rt,ctx):
    """GT-free calculation boundary: one leg source, one shared parent, one frozen target."""
    import run_pelvis_frame_leg_articulation as w95
    from run_torso_frame_rig_replay import conj,spine_local_xyzw
    from motion.torso_frame import matrix_to_quaternion_xyzw
    if j is None or Rt is None:return None
    j=np.asarray(j,float);j=j-j[0]
    ev=w95.evaluate(j,Ro,Rt,ctx)  # no reference directions supplied
    if ev is None:return None
    bones={"spine_01":list(spine_local_xyzw(ctx["alignment"],Ro.T@Rt,ctx["spine_basis"]))}
    points={};fidelity=[]
    for s,(h,k,a) in w95.LEGS.items():
        leg=ev["legs"][s];p=leg["pose"]
        bones[f"thigh_{s}"],bones[f"calf_{s}"]=list(p.proximal_local),list(p.distal_local)
        points[f"thigh_{s}"],points[f"calf_{s}"]=list(p.mid),list(p.end)
        fidelity.extend([w95.angle(leg["thigh_w"],j[k]-j[h]),w95.angle(leg["shin_w"],j[a]-j[k])])
    return {"object":list(matrix_to_quaternion_xyzw(conj(ctx["alignment"],Ro))),"bones":bones,"points":points,
        "forward":list(Rt[:,1]),"knee":[ev["legs"][s]["knee"] for s in "lr"],
        "hip_flex":[ev["legs"][s]["hip_flex"] for s in "lr"],"source_direction_fidelity_max_deg":max(fidelity)}


def arm_export(elbow,wrist,Ro,Rt,ctx):
    from run_torso_frame_rig_replay import conj,spine_local_xyzw,arm_pose
    from motion.torso_frame import matrix_to_quaternion_xyzw
    if Rt is None or not np.isfinite([elbow,wrist]).all():return None
    arm=arm_pose(elbow,wrist,Rt,ctx["alignment"],ctx["target"],ctx["rest"])
    if arm is None or arm.status!="known":return None
    Q=conj(ctx["alignment"],Ro.T@Rt);head=np.asarray(ctx["spine_head"])
    sp=lambda pt:list(head+Q@(np.asarray(pt)-head))
    return {"proximal_local_xyzw":list(arm.proximal_local),"distal_local_xyzw":list(arm.distal_local),
            "mid":sp(arm.mid),"end":sp(arm.end),"root":sp(ctx["target"].root_head),
            "object_rotation_armature_xyzw":list(matrix_to_quaternion_xyzw(conj(ctx["alignment"],Ro))),
            "extra_bone_locals_xyzw":{"spine_01":list(spine_local_xyzw(ctx["alignment"],Ro.T@Rt,ctx["spine_basis"]))}}


def windows(rows):
    groups=defaultdict(list)
    for r in rows:groups[r["sid"]].append(r)
    for rr in groups.values():rr.sort(key=lambda r:r["frame"])
    out={};used=set()
    for key,sid,f0 in FIXED:
        rr=groups.get(sid,[]);starts=[i for i,r in enumerate(rr) if r["frame"]==f0]
        if starts:
            out[key]=rr[starts[0]:starts[0]+36];used.update(r["sample_id"] for r in out[key])
    for name in ("maximum_source_regression","association_failure"):
        options=[]
        for sid,rr in sorted(groups.items()):
            for i in range(len(rr)):
                w=rr[i:i+36]
                if len(w)<36 or any(r["sample_id"] in used for r in w):continue
                if name=="maximum_source_regression":
                    if any(r["identity"]!="same_person" for r in w):continue
                    score=sum(r["m"]["M"]["mpjpe"]-r["m"]["B"]["mpjpe"] for r in w)
                else:score=sum(r["identity"]!="same_person" for r in w)
                options.append((-score,sid,w[0]["frame"],w))
        if options:
            out[name]=min(options,key=lambda x:x[:3])[3];used.update(r["sample_id"] for r in out[name])
    return out


def build(args):
    import w101_reference_estimator as w101
    from run_external_orientation_handoff import handoff_frames,W97_PREDICTIONS_SHA
    from run_visual_depth_motion_attribution import W87_PREDICTIONS
    from framepose.bank import load_bank
    if w101.sha(args.hmr_predictions)!=W97_PREDICTIONS_SHA:raise RuntimeError("STOP: HMR identity changed")
    ctx=context(args.rest,args.calibration,args.chains);bank=load_bank(args.bank)
    if bank.content_digest()!=w101.BANK_DIGEST:raise RuntimeError("STOP: bank identity changed")
    j1=None
    if args.j1x:
        if w101.sha(args.j1x)!=W87_PREDICTIONS["3dpw"]:raise RuntimeError("STOP: frozen J1X identity changed")
        j1=np.load(args.j1x,allow_pickle=False)
        if any(j1[k].shape!=(len(bank),3) for k in ("left_arm_g1_elbow","left_arm_g1_wrist")):
            raise RuntimeError("STOP: J1X bank row shape mismatch")
    hp=np.load(args.hmr_predictions,allow_pickle=False)
    h={int(p):i for i,(pop,p) in enumerate(zip(hp["population"],hp["index"][0])) if str(pop).startswith("3dpw") and hp["valid"][i]}
    rows=json.loads(args.rows.read_text());selected=windows(rows);cases={};diagnostics={}
    if args.missing_only:
        missing=sorted((r for r in rows if not r["tracked"]),key=lambda r:(r["sid"],r["frame"]))
        if not missing:raise RuntimeError("No missing subject exists for the requested diagnostic")
        first=missing[0];rr=sorted((r for r in rows if r["sid"]==first["sid"]),key=lambda r:r["frame"])
        i=next(i for i,r in enumerate(rr) if r["sample_id"]==first["sample_id"])
        selected={"missing_subject_probe":rr[max(0,i-5):i+31]}
    for name,rr in selected.items():
        frames=[];detail=[]
        for r in rr:
            p=r["p"];s=bank.samples[p]
            if s.sample_id!=r["sample_id"] or s.frame_index!=r["frame"]:raise RuntimeError("STOP: rig row identity")
            Ro,Rt=(None,None) if p not in h else handoff_frames(hp["global_orient"][h[p]],hp["joints24"][h[p]])[:2]
            shared_arm=None if j1 is None or Rt is None else arm_export(j1["left_arm_g1_elbow"][p],j1["left_arm_g1_wrist"][p],Ro,Rt,ctx)
            fr={"bank_position":p,"frame_index":s.frame_index,"time_s":s.timestamp,"image":s.image_reference.relative_path,
                "available":Rt is not None,"gt_valid":False,"identity":r["identity"],"runtime_status":"DIAGNOSTIC",
                "runtime_reasons":[],"yaw_deg":{},"modes":{},"forward_cam":{},"mode_lines":{},
                "object_rotation_xyzw":{},"extra_bone_locals_xyzw":{},"fk_points_armature":{}}
            dr={"sample_id":s.sample_id,"identity":r["identity"],"modes":{}}
            for slot,c in SLOTS.items():
                j=r["J"].get(c) if c=="B" or r["tracked"] else None
                pose=None if Rt is None else pose_export(j,Ro,Rt,ctx)
                if pose is None:
                    fr["yaw_deg"][slot]=None;fr["mode_lines"][slot]=["UNAVAILABLE; no source/GT fallback"]
                    dr["modes"][slot]={"available":False};continue
                fr["object_rotation_xyzw"][slot]=pose["object"];fr["extra_bone_locals_xyzw"][slot]=pose["bones"]
                fr["fk_points_armature"][slot]=pose["points"];fr["forward_cam"][slot]=pose["forward"]
                if shared_arm is not None:fr["modes"][slot]=shared_arm
                else:
                    # Display-only rest arms are explicit; no observation or model prediction is filled.
                    for bone in ("upperarm_l","lowerarm_l"):fr["extra_bone_locals_xyzw"][slot][bone]=[0.,0.,0.,1.]
                fr["yaw_deg"][slot]=math.degrees(math.atan2(-Rt[0,1],Rt[1,1]))
                fr["mode_lines"][slot]=[f"{c}_Main leg geometry | identity {r['identity']}",
                    "shared HMR parent, shared frozen J1X arm; mixed-source diagnostic" if shared_arm else "shared HMR parent, rest-arm display (J1X unavailable)"]
                dr["modes"][slot]={"available":True,"shared_J1X_arm_available":shared_arm is not None,"knee":pose["knee"],"hip_flex":pose["hip_flex"],
                    "source_direction_fidelity_max_deg":pose["source_direction_fidelity_max_deg"]}
            if len(fr["object_rotation_xyzw"])==2:
                if (fr["object_rotation_xyzw"]["FIXED"]!=fr["object_rotation_xyzw"]["RUNTIME"] or fr["extra_bone_locals_xyzw"]["FIXED"]["spine_01"]!=fr["extra_bone_locals_xyzw"]["RUNTIME"]["spine_01"] or
                    fr["modes"].get("FIXED")!=fr["modes"].get("RUNTIME")):
                    raise RuntimeError("STOP: common parent invariant failed")
            frames.append(fr);detail.append(dr)
        delta=np.diff([f["time_s"] for f in frames]);fps=1/float(np.median(delta)) if len(delta) else 6.
        cases[name]={"case":name,"sequence_id":rr[0]["sid"],"start_frame":rr[0]["frame"],"fps_sampled":fps,"frames":frames,
            "labels":{"FIXED":"B_Main benchmark legs","RUNTIME":"M_Main M1 legs"},"short_labels":SLOTS,
            "title":"W102 same-environment leg source | mixed HMR parent / frozen shared J1X arm",
            "header_note":"No thigh-roll candidate; association labels evaluation-only; unavailable panels hidden"}
        diagnostics[name]=detail
    w101.dump(args.out_dir/"cases.json",{"rest_snapshot_digest":ctx["rest"].digest(),"canonical_to_armature_xyzw":list(ctx["alignment"]),
        "target_chain":{"root_head":list(ctx["target"].root_head)},"cases":cases})
    w101.dump(args.out_dir/"rig_report.json",{"schema":"w102_mixed_source_rig_diagnostic_v1",
        "inputs":{str(p):w101.sha(p) for p in (args.rows,args.hmr_predictions,args.rest,args.calibration,args.chains) + ((args.j1x,) if args.j1x else ())},
        "GT_in_rotation_calculation":False,"arms":"frozen shared W87 J1X left arm; right rest; unavailable left rest display" if j1 else "shared rest",
        "common_parent_invariant":True,"common_arm_invariant":True,
        "selection":"post-hoc display probe: first missing subject, five original observations before + following 31" if args.missing_only else "four historical fixed windows + maximum 36-sample source regression + association failure, disjoint",
        "cases":diagnostics})


def render(args):
    import render_runtime_heading_rig_blender as historical
    historical.MODES=tuple(SLOTS)  # diagnostic wrapper; historical renderer file remains unchanged
    return historical.render(args)


def compose(args):
    import cv2
    data=json.loads(args.cases.read_text())
    for name,case in data["cases"].items():
        out=args.out_dir/(name+".mp4");writer=cv2.VideoWriter(str(out),cv2.VideoWriter_fourcc(*"mp4v"),30.,(1440,1000))
        if not writer.isOpened():raise RuntimeError("STOP: review encoder unavailable")
        for i,fr in enumerate(case["frames"]):
            canvas=np.full((1000,1440,3),22,np.uint8)
            im=cv2.imread(str(args.image_root/fr["image"]));canvas[40:520,:480]=cv2.resize(im,(480,480))
            lines=[name,case["sequence_id"],f"frame {fr['frame_index']} | {fr['time_s']:.3f}s",f"association: {fr['identity']}",
                "Common frozen HMR pelvis + thorax", "Common BaseRig / IK / frozen J1X arm", "Only leg source changes", "Mixed-source diagnostic, sparse cadence"]
            for n,line in enumerate(lines):cv2.putText(canvas,line,(12,550+35*n),cv2.FONT_HERSHEY_SIMPLEX,.52,(230,230,230),1,cv2.LINE_AA)
            for col,(slot,label) in enumerate(case["labels"].items(),1):
                cv2.putText(canvas,label,(col*480+10,28),cv2.FONT_HERSHEY_SIMPLEX,.6,(240,230,200),1,cv2.LINE_AA)
                if slot in fr["object_rotation_xyzw"]:
                    for y,view in ((40,"video"),(520,"side")):
                        panel=cv2.imread(str(args.render_dir/name/f"{slot}_{view}_{i:03d}.png"))
                        if panel is None:raise RuntimeError("STOP: missing rig render")
                        canvas[y:y+480,col*480:(col+1)*480]=cv2.resize(panel,(480,480))
                else:cv2.putText(canvas,"UNAVAILABLE - no fallback",(col*480+8,270),cv2.FONT_HERSHEY_SIMPLEX,.57,(50,80,250),1)
            cv2.putText(canvas,"SIDE: depth x up",(500,550),cv2.FONT_HERSHEY_SIMPLEX,.5,(200,200,200),1)
            next_time=case["frames"][i+1]["time_s"] if i+1<len(case["frames"]) else fr["time_s"]+1/case["fps_sampled"]
            for _ in range(max(1,int(round((next_time-fr["time_s"])*30)))):writer.write(canvas)
            if i in (0,len(case["frames"])//2,len(case["frames"])-1) or fr["identity"]=="no_subject":
                cv2.imwrite(str(args.out_dir/f"{name}_{i:03d}.jpg"),canvas)
        writer.release()


def main():
    argv=sys.argv[sys.argv.index("--")+1:] if "--" in sys.argv else sys.argv[1:]
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument("mode",choices=("build","render","compose"))
    for name in ("bank","rows","hmr_predictions","j1x","rest","calibration","chains","cases","rig","render_dir","image_root"):
        ap.add_argument("--"+name.replace("_","-"),type=Path)
    ap.add_argument("--out-dir",type=Path,required=True);ap.add_argument("--size",type=int,default=480)
    ap.add_argument("--equivalent-rig-tolerance",type=float,default=1e-4);ap.add_argument("--show-mesh",action="store_true")
    ap.add_argument("--ortho-scale",type=float,default=3.2);ap.add_argument("--camera-centre-height",type=float,default=None)
    ap.add_argument("--only",default=None);ap.add_argument("--missing-only",action="store_true")
    args=ap.parse_args(argv);args.out_dir.mkdir(parents=True,exist_ok=False)
    return {"build":build,"render":render,"compose":compose}[args.mode](args)


if __name__=="__main__":main()
