#!/usr/bin/env python3
"""W102 Main-only M1 observation verification/recording, using production code unchanged.

No GT is loaded. Detector returns are recorded without changing them. Selected boxes are reproduced by the exact
production tracker on those returns; selected pose outputs are reused for the corresponding privileged candidate.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"src"),str(ROOT/"scripts")]
import run_real_observation_mmpose as w100
import w101_reference_observations as reference
from framepose.observations import image_content_digest,mmpose_observation,observation_cache_key
from mediaio.frame_sequence import Frame,FrameSequence
from pose import mmpose_adapter as prod


def write(path,value):
    Path(path).write_text(json.dumps(value,indent=2,default=str))


def cases_for_gate(videos,image_root):
    cases={};images={}
    for video in w100.PREDECLARED["gate_videos"]:
        paths=videos[video][:12]
        if video=="downtown_bar_00":
            paths=sorted(set(paths+[p for p in videos[video] if int(Path(p).stem.split("_")[-1]) in (1158,1168)]))
        cases[video]=reference.sequence(paths,image_root)
        images[video]=[{"image":p,"sha256":image_content_digest(image_root/p.removeprefix("3dpw/"))} for p in paths]
    first=cases["downtown_walkBridge_01"].frames[:6]
    def black(f):return Frame(index=0,timestamp=0.,image=np.zeros_like(f.image),width=f.width,height=f.height)
    controls=[black(first[0])]+first[:3]+[black(first[0])]+first[3:]+cases["flat_guitar_01"].frames[:3]
    frames=[Frame(index=i,timestamp=i/30.,image=f.image,width=f.width,height=f.height) for i,f in enumerate(controls)]
    cases["black_leave_return_mixed_dimensions"]=FrameSequence(frames,30.,frames[0].width,frames[0].height,"w102_controls")
    return cases,images


def gate(args,cfg,identity):
    videos,_=w100.population(args.population)
    cases,images=cases_for_gate(videos,args.image_root)
    estimator=reference.initialize(cfg)
    counters=[w100.CallCounter(estimator._detector),w100.CallCounter(estimator._model)]
    loaded={key:hashlib.sha256(model.cfg.pretty_text.encode()).hexdigest()
            for key,model in (("pose",estimator._model),("detector",estimator._detector))}
    report={**identity,"population_sha256":reference.sha(args.population),"resolved_config_sha256":loaded,
            "images":images,"cases":{},"purpose":"M1 policy, ordering and repeatability; not M8 or a historical S0 SHA gate"}
    passed=True
    for name,seq in cases.items():
        a,ta,sa=reference.run(estimator,seq)
        b,tb,sb=reference.run(estimator,seq)
        comparison=w100.compare(a,b)
        ok=(comparison["presence_mismatch"]==0 and comparison["visibility_flips"]==0 and
            comparison["max_xy_px"]<=.5 and comparison["max_conf"]<=1e-3 and ta==tb)
        passed &= ok
        report["cases"][name]={"frames":len(seq.frames),"pass":bool(ok),"bitwise_repeat":bool(np.array_equal(a,b,equal_nan=True)),
                                 "comparison":comparison,"tracking":ta,"seconds":[sa,sb]}
        np.savez_compressed(args.out_dir/(name+".npz"),M1=a,repeat=b,frame_index=[f.index for f in seq.frames],
                            timestamp=[f.timestamp for f in seq.frames])
        print(name,ok,comparison["max_xy_px"],flush=True)
    report["detector_forward_batch_sizes"]=counters[0].calls
    report["pose_forward_batch_sizes"]=counters[1].calls
    passed &= bool(counters[0].calls and counters[1].calls and set(counters[0].calls+counters[1].calls)=={1})
    report["pass"]=bool(passed)
    write(args.out_dir/"gate_report.json",report)
    if not passed:raise RuntimeError("STOP: M1 observation contract/repeatability failed")


def observe(args,cfg,identity):
    previous=json.loads(args.gate_report.read_text())
    if not previous["pass"] or previous["software"]!=identity["software"] or previous["config_files_sha256"]!=identity["config_files_sha256"]:
        raise RuntimeError("STOP: incompatible M1 gate/source environment")
    estimator=reference.initialize(cfg)
    for key,model in (("pose",estimator._model),("detector",estimator._detector)):
        if hashlib.sha256(model.cfg.pretty_text.encode()).hexdigest()!=previous["resolved_config_sha256"][key]:
            raise RuntimeError("STOP: loaded M1 configuration changed")
    provenance=mmpose_observation(
        pose_config=Path(cfg.config_path).name,pose_checkpoint=Path(cfg.checkpoint_path).name,
        detector_config=Path(cfg.detector_config_path).name,detector_checkpoint=Path(cfg.detector_checkpoint_path).name,
        visibility_threshold=.3,input_size="192x256",mmpose_version=identity["software"]["mmpose"],
        pose_weights_sha256=identity["checkpoints_sha256"]["rtmpose_tiny"],
        detector_weights_sha256=identity["checkpoints_sha256"]["rtmdet_tiny"],
        extra={"batch_size":1,"device":"cpu","resolved_config_sha256":previous["resolved_config_sha256"],
               "software":identity["software"],"selected_candidate":"actual production pose reused when its bbox is identical"})
    videos,_=w100.population(args.population)
    image_manifest=json.loads(args.image_manifest.read_text())
    manifest={**identity,"population_sha256":reference.sha(args.population),"gate_report_sha256":reference.sha(args.gate_report),
              "provenance":provenance.to_dict(),"videos":{}}
    write(args.out_dir/"manifest.json",manifest)
    for video,paths in videos.items():
        start=time.monotonic()
        seq=reference.sequence(paths,args.image_root)
        digests=[image_content_digest(args.image_root/p.removeprefix("3dpw/")) for p in paths]
        for p,digest in zip(paths,digests):
            if image_manifest[p.removeprefix("3dpw/")]!=digest:raise RuntimeError("STOP: exact RGB source identity changed")
        recorded=[]
        original=estimator._detector.test_step
        def record(data):
            result=original(data);recorded.extend(result);return result
        estimator._detector.test_step=record
        try:landmarks,tracking,seconds=reference.run(estimator,seq)
        finally:estimator._detector.test_step=original
        if len(recorded)!=len(seq.frames):raise RuntimeError("STOP: detector/frame recording alignment")
        detections=w100.detections_of(recorded)
        selected=w100.select_track(detections)
        if sum(s is not None for s in selected)!=tracking["tracked_frame_count"]:
            raise RuntimeError("STOP: tracker replay disagrees with production")
        boxes=[];candidate_landmarks=[];candidate_frames=[];was_selected=[]
        for i,ds in enumerate(detections):
            for box,score in ds:
                selected_here=selected[i] is not None and box==selected[i].bbox
                if selected_here:lm=landmarks[i].copy()
                else:
                    result=prod._pose_batch(estimator._model,[(seq.frames[i].image,box)])[0]
                    lm=w100.landmarks_array(type("PF",(),{"landmarks":prod._extract_canonical_landmarks([result],.3)})())
                boxes.append((*box,score));candidate_frames.append(i);candidate_landmarks.append(lm);was_selected.append(selected_here)
        output=args.out_dir/(video+".npz")
        np.savez_compressed(output,frame_index=[f.index for f in seq.frames],timestamp=[f.timestamp for f in seq.frames],
            landmarks=landmarks,cand_frame=np.asarray(candidate_frames,int),cand_box=np.asarray(boxes).reshape(-1,5),
            cand_landmarks=np.asarray(candidate_landmarks).reshape(-1,16,4),cand_selected=np.asarray(was_selected,bool),
            selected_box=np.asarray([s.bbox if s is not None else (np.nan,)*4 for s in selected]),
            image_size=[(f.width,f.height) for f in seq.frames])
        meta={"images":paths,"image_sha256":digests,"provenance":provenance.to_dict(),
              "observation_cache_keys":[observation_cache_key(provenance,d) for d in digests],"npz_sha256":reference.sha(output),
              "tracking_report":tracking,"observe_seconds":seconds,"total_seconds":time.monotonic()-start,
              "candidates":len(boxes),"frames":len(paths)}
        write(args.out_dir/(video+".json"),meta)
        manifest["videos"][video]=meta;write(args.out_dir/"manifest.json",manifest)
        print(video,len(paths),round(meta["total_seconds"],2),tracking["tracked_frame_count"],flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode",choices=("gate","observe"))
    for key in ("population","image_root","weights","out_dir"):
        ap.add_argument("--"+key.replace("_","-"),type=Path,required=True)
    ap.add_argument("--gate-report",type=Path)
    ap.add_argument("--image-manifest",type=Path)
    args=ap.parse_args()
    args.out_dir.mkdir(parents=True,exist_ok=False)
    cfg=reference.config(args.weights)
    identity=reference.identity(cfg)
    (gate if args.mode=="gate" else observe)(args,cfg,identity)


if __name__=="__main__":main()
