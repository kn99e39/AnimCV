#!/usr/bin/env python3
"""W102 private historical-video diagnostic. No annotations, GT, fitting, or runtime integration.

inspect decodes and inventories the unchanged clip before inference. run uses the fixed Main M1/S0 stack.
All 3D panels use camera depth x up (Y,Z). Source rights remain undocumented; outputs are private review only.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"src"),str(ROOT/"scripts")]
import w101_reference_observations as ref
import w101_reference_estimator as w101
from mediaio.frame_sequence import Frame,FrameSequence
from framepose.crops import crop_box,geometry_in_crop
from framepose.train import load_checkpoint,predict
from framepose.signs import neutral_sign_states
from run_w102_same_environment_s0 import execution_identity
from eval_external_hmr2_baseline import LIMBS,ang

EDGES=((0,1),(1,2),(2,3),(0,4),(4,5),(5,6),(0,7),(7,8),(8,10),
       (8,11),(11,12),(12,13),(8,14),(14,15),(15,16))


def decode(path):
    cap=cv2.VideoCapture(str(path));fps=cap.get(cv2.CAP_PROP_FPS);frames=[];digests=[]
    if not np.isfinite(fps) or fps<=0:raise RuntimeError("STOP: missing video cadence")
    while True:
        ok,im=cap.read()
        if not ok:break
        h,w=im.shape[:2];i=len(frames)
        frames.append(Frame(i,i/fps,im,w,h));digests.append(hashlib.sha256(im.tobytes()).hexdigest())
    cap.release()
    if not frames or len({(f.width,f.height) for f in frames})!=1:raise RuntimeError("STOP: invalid decoded clip")
    return FrameSequence(frames,fps,frames[0].width,frames[0].height,str(path)),digests


def text(im,s,x=12,y=25,color=(235,235,235),scale=.53):
    cv2.putText(im,s,(x,y),cv2.FONT_HERSHEY_SIMPLEX,scale,color,1,cv2.LINE_AA)


def inspect(args):
    seq,digests=decode(args.clip);ids=np.linspace(0,len(seq.frames)-1,8).round().astype(int)
    sheet=np.zeros((2*280,4*420,3),np.uint8)
    for n,i in enumerate(ids):
        panel=cv2.resize(seq.frames[i].image,(420,250));sheet[n//4*280:n//4*280+250,n%4*420:(n%4+1)*420]=panel
        text(sheet,f"frame {i} / {i/seq.fps:.3f}s",n%4*420+8,n//4*280+270)
    cv2.imwrite(str(args.out_dir/"motion_inventory.jpg"),sheet)
    w101.dump(args.out_dir/"inspection.json",{"clip_sha256":ref.sha(args.clip),"frames":len(seq.frames),
        "fps":seq.fps,"size":[seq.width,seq.height],"timestamp":[f.timestamp for f in seq.frames],
        "decoded_BGR_uint8_sha256":digests,"source":"historical production_demo; origin/rights undocumented",
        "GT_available":False,"unseen":False})


def render_frame(frame,x,v,j,present):
    rgb=cv2.resize(frame.image,(640,480));side=np.full((480,640,3),25,np.uint8)
    xy=x[:,:2]*[640,480]
    for a,b in EDGES:
        if v[a] and v[b]:cv2.line(rgb,tuple(xy[a].astype(int)),tuple(xy[b].astype(int)),(0,220,255),2,cv2.LINE_AA)
    if present:
        yz=(j-j[0])[:,[1,2]];uv=np.c_[320+yz[:,0]*230,270-yz[:,1]*230].astype(int)
        for a,b in EDGES:
            cv2.line(side,tuple(uv[a]),tuple(uv[b]),(255,170,70),3,cv2.LINE_AA)
    else:text(side,"M1 subject unavailable: S0 output hidden",y=220)
    text(rgb,f"Historical diagnostic | frame {frame.index} | {frame.timestamp:.3f}s",y=22)
    text(rgb,f"M1 batch=1 | valid joints {int(v.sum())}/17",y=458)
    text(side,"Frozen S0 | SIDE: camera depth Y x up Z",y=24)
    text(side,"No 3D GT: this is not an accuracy evaluation",y=458)
    return np.hstack([rgb,side])


def run(args):
    import torch
    declaration=json.loads(args.inspection.read_text());seq,digests=decode(args.clip)
    if declaration["clip_sha256"]!=ref.sha(args.clip) or declaration["decoded_BGR_uint8_sha256"]!=digests:
        raise RuntimeError("STOP: clip/decode identity changed after inspection")
    cfg=ref.config(args.weights);identity=ref.identity(cfg)
    gate=json.loads(args.gate_report.read_text())
    if not gate["pass"] or gate["software"]!=identity["software"]:raise RuntimeError("STOP: incompatible M1 gate")
    estimator=ref.initialize(cfg)
    for key,model in (("pose",estimator._model),("detector",estimator._detector)):
        if hashlib.sha256(model.cfg.pretty_text.encode()).hexdigest()!=gate["resolved_config_sha256"][key]:
            raise RuntimeError("STOP: M1 loaded configuration changed")
    landmarks,tracking,seconds=ref.run(estimator,seq)
    inputs=[w101.to_input(lm if np.isfinite(lm[:,0]).any() else None,(seq.width,seq.height)) for lm in landmarks]
    x=np.stack([r[0] for r in inputs]);v=np.stack([r[1] for r in inputs])
    geometry=np.stack([geometry_in_crop(xx,vv,(seq.width,seq.height),crop_box(xx,vv,(seq.width,seq.height))) for xx,vv in inputs])
    environment,fingerprint=execution_identity(torch);locked=json.loads(args.lock.read_text())
    if fingerprint!=locked["environment_fingerprint"]:raise RuntimeError("STOP: fixed S0 environment changed")
    w101.verify_s0_identity(args.checkpoint)
    model,ck=load_checkpoint(args.checkpoint,device="cuda")
    if ck["candidate"]["name"]!="S0_neutral_sign":raise RuntimeError("STOP: frozen S0 identity changed")
    predictions=predict(model,torch,geometry,None,np.arange(len(x)),torch.device("cuda"),batch_size=512,
                        amp_enabled=False,signs=neutral_sign_states(len(x))).astype(np.float32)
    if not np.isfinite(predictions).all():raise RuntimeError("STOP: invalid S0 output")
    present=np.isfinite(landmarks[:,:,0]).any(1);steps=[];output_steps=[]
    for i in range(1,len(x)):
        observed=[ang((x[i,a,:2]-x[i,b,:2])*[seq.width,seq.height],(x[i-1,a,:2]-x[i-1,b,:2])*[seq.width,seq.height])
                  for a,b in LIMBS.values() if v[i,a] and v[i,b] and v[i-1,a] and v[i-1,b]]
        steps.append(max(observed) if observed else np.nan)
        if present[i] and present[i-1]:
            output_steps.extend(ang(predictions[i,a]-predictions[i,b],predictions[i-1,a]-predictions[i-1,b]) for a,b in LIMBS.values())
    finite=np.flatnonzero(np.isfinite(steps));largest=int(finite[np.argmax(np.asarray(steps)[finite])]+1) if len(finite) else 0
    chosen=sorted({0,len(x)//2,len(x)-1,int(np.argmin(v.sum(1))),largest})
    video=args.out_dir/"historical_video_diagnostic.mp4"
    writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*"mp4v"),seq.fps,(1280,480))
    if not writer.isOpened():raise RuntimeError("STOP: private review encoder unavailable")
    for i,f in enumerate(seq.frames):
        panel=render_frame(f,x[i],v[i],predictions[i],present[i]);writer.write(panel)
        if i in chosen:cv2.imwrite(str(args.out_dir/f"representative_{i:04d}.jpg"),panel)
    writer.release()
    np.savez_compressed(args.out_dir/"observations_predictions.npz",landmarks=landmarks,input_2d=x,input_valid=v,
        geometry=geometry,predictions=predictions,frame_index=np.arange(len(x)),timestamp=[f.timestamp for f in seq.frames])
    from w102_evaluate import distribution
    w101.dump(args.out_dir/"report.json",{"clip_sha256":ref.sha(args.clip),"inspection_sha256":ref.sha(args.inspection),
        "decoded_BGR_uint8_sha256":digests,"frames":len(x),"fps":seq.fps,"environment":environment,
        "environment_fingerprint":fingerprint,"M1_identity":identity,"tracking":tracking,"M1_seconds":seconds,
        "selected_frames":chosen,"minimum_valid_joints":int(v.sum(1).min()),"missing_subject_frames":int((~present).sum()),
        "validity_histogram":dict(zip(*[a.tolist() for a in np.unique(v.sum(1),return_counts=True)])),
        "observed_max_limb_step_degrees":distribution(steps),"S0_limb_step_degrees":distribution(output_steps),
        "S0_limb_steps_over_45_degrees":int(np.sum(np.asarray(output_steps)>45)),
        "observation_prediction_sha256":ref.sha(args.out_dir/"observations_predictions.npz"),
        "no_GT":True,"unseen":False,"rights":"undocumented; private diagnostic only; no raw clip export"})


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument("mode",choices=("inspect","run"))
    for name in ("clip","out_dir"):ap.add_argument("--"+name.replace("_","-"),type=Path,required=True)
    for name in ("weights","checkpoint","gate_report","lock","inspection"):
        ap.add_argument("--"+name.replace("_","-"),type=Path)
    args=ap.parse_args();args.out_dir.mkdir(parents=True,exist_ok=False)
    (inspect if args.mode=="inspect" else run)(args)


if __name__=="__main__":main()
