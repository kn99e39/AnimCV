#!/usr/bin/env python3
"""W102: frozen S0 source comparison on one fixed execution stack.

Historical byte identity is recorded, never redefined. The new control is B_Main on this same stack.
No production or historical script is changed. Inputs, geometry, signs, precision and batch ordering stay fixed.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"src"),str(ROOT/"scripts")]
from framepose.bank import load_bank
from framepose.signs import neutral_sign_states
from framepose.train import load_checkpoint,predict
from run_real_observation_s0 import S0_CHECKPOINT_SHA,S0_PRED_SHA
from w101_reference_estimator import BANK_DIGEST,geometry_for_condition,dump,sha

CONDITIONS={"B_Main":"B","M_Main":"M","Moa_Main":"Moa","Mconf_Main":"Mconf","Mmask_Main":"Mmask"}
MODEL_FILES=("src/framepose/model.py","src/framepose/train.py","src/framepose/crops.py","src/framepose/signs.py",
             "src/framepose/contract.py","src/common/canonical_pose.py")


def execution_identity(torch):
    versions={}
    for name in ("torch","numpy","torchvision","mmpose","mmdet","mmcv","mmengine"):
        try:versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:versions[name]=None
    gpu=subprocess.run(["nvidia-smi","--query-gpu=uuid,driver_version,name","--format=csv,noheader"],
                       check=True,capture_output=True,text=True).stdout.strip()
    identity={"platform":platform.platform(),"python":sys.version,"versions":versions,"gpu_identity":gpu,
              "device":"cuda:0","gpu":torch.cuda.get_device_name(0),"capability":list(torch.cuda.get_device_capability(0)),
              "cuda_runtime":torch.version.cuda,"compiled_architectures":torch.cuda.get_arch_list(),
              "batch_size":512,"amp":False,"geometry_dtype":"float32","neutral_signs":True,
              "matmul_precision":torch.get_float32_matmul_precision(),"matmul_tf32":torch.backends.cuda.matmul.allow_tf32,
              "cudnn_tf32":torch.backends.cudnn.allow_tf32,"model_source_sha256":{p:sha(ROOT/p) for p in MODEL_FILES}}
    return identity,hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()


def baseline_observation_identity(bank):
    positions=np.concatenate([bank.indices("validation"),bank.indices("test")])
    return {"positions":positions,"sample_id":np.array([bank.samples[p].sample_id for p in positions]),
            "timestamp":np.array([bank.samples[p].timestamp for p in positions])}


def run(args):
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if sha(args.checkpoint)!=S0_CHECKPOINT_SHA:raise RuntimeError("STOP: frozen checkpoint identity failed")
    bank=load_bank(args.bank)
    if bank.content_digest()!=BANK_DIGEST:raise RuntimeError("STOP: frozen bank identity failed")
    if not torch.cuda.is_available():raise RuntimeError("STOP: fixed Main CUDA unavailable; no fallback")
    environment,fingerprint=execution_identity(torch)
    candidate=any(c!="B_Main" for c in args.conditions)
    if candidate:
        if not args.lock or not args.preflight or args.conditions[0]!="B_Main":
            raise RuntimeError("STOP: candidate requires locked same-run B_Main and the semantic preflight")
        preflight=json.loads(args.preflight.read_text())
        if not preflight["pass"] or preflight["environment_fingerprint"]!=fingerprint:
            raise RuntimeError("STOP: valid same-environment preflight is required")
        if sha(args.lock) not in preflight["baseline_report_sha256"]:
            raise RuntimeError("STOP: baseline lock does not belong to the verified preflight")
    if args.lock:
        previous=json.loads(args.lock.read_text())
        if previous["environment_fingerprint"]!=fingerprint:raise RuntimeError("STOP: uncontrolled execution change")
        if previous["checkpoint_sha256"]!=S0_CHECKPOINT_SHA or previous["bank_content_digest"]!=BANK_DIGEST:
            raise RuntimeError("STOP: lock input identity changed")
    args.out_dir.mkdir(parents=True,exist_ok=False)
    model,checkpoint=load_checkpoint(args.checkpoint,device="cuda")
    if checkpoint.get("candidate",{}).get("name")!="S0_neutral_sign":raise RuntimeError("STOP: checkpoint candidate differs")
    report={"schema":"w102_same_environment_s0_v1","environment":environment,"environment_fingerprint":fingerprint,
            "checkpoint_sha256":S0_CHECKPOINT_SHA,"bank_content_digest":BANK_DIGEST,"bank_index_sha256":sha(args.bank),
            "model_config":checkpoint.get("model_config"),"candidate":checkpoint.get("candidate"),
            "runner_sha256":sha(__file__),"command":sys.argv,"conditions":{}}
    observations=np.load(args.observations,allow_pickle=False) if args.observations else baseline_observation_identity(bank)
    if any(c!="B_Main" for c in args.conditions) and not args.observations:
        raise RuntimeError("STOP: candidate requires verified observation artifact")
    report["observations_sha256"]=sha(args.observations) if args.observations else None
    report["preflight_sha256"]=sha(args.preflight) if args.preflight else None
    signs=neutral_sign_states(len(bank))
    for condition in args.conditions:
        geometry=geometry_for_condition(bank,observations,CONDITIONS[condition])
        report["conditions"][condition]={"geometry_sha256":hashlib.sha256(geometry.tobytes()).hexdigest(),"splits":{}}
        for split in ("validation","test"):
            positions=bank.indices(split)
            prediction=predict(model,torch,geometry,None,positions,torch.device("cuda"),batch_size=512,
                               amp_enabled=False,signs=signs).astype(np.float32)
            if prediction.shape!=(len(positions),17,3) or not np.isfinite(prediction).all():
                raise RuntimeError("STOP: nonfinite/invalid S0 output")
            path=args.out_dir/f"prediction_{condition}_{split}.npy"
            np.save(path,prediction)
            result={"rows":len(positions),"sha256":sha(path),"shape":list(prediction.shape),"dtype":str(prediction.dtype),
                    "finite":True,"sample_ids_sha256":hashlib.sha256("\n".join(bank.samples[p].sample_id for p in positions).encode()).hexdigest()}
            if condition=="B_Main":
                result["historical_pinned_sha256"]=S0_PRED_SHA[split]
                result["historical_byte_identity"]=sha(path)==S0_PRED_SHA[split]
                if candidate:
                    locked_path=args.lock.parent/f"prediction_B_Main_{split}.npy"
                    if sha(locked_path)!=previous["conditions"]["B_Main"]["splits"][split]["sha256"]:
                        raise RuntimeError("STOP: locked B_Main artifact changed")
                    locked_prediction=np.load(locked_path,allow_pickle=False)
                    result["same_run_control_byte_equal"]=bool(prediction.tobytes()==locked_prediction.tobytes())
                    if not result["same_run_control_byte_equal"]:
                        from w102_numerical_audit import compare_predictions
                        check=compare_predictions(locked_prediction,prediction,bank.arrays["target_3d"][positions],
                            [bank.samples[p] for p in positions],bank.arrays["target_valid"][positions])
                        result["same_run_control_semantic_audit"]=check
                        if not check["pass"]:raise RuntimeError("STOP: same-run control is unstable")
            report["conditions"][condition]["splits"][split]=result
            dump(args.out_dir/"s0_report.json",report)
            print(condition,split,result["rows"],result["sha256"],flush=True)
    return report


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for key in ("bank","checkpoint","out_dir"):
        ap.add_argument("--"+key.replace("_","-"),type=Path,required=True)
    ap.add_argument("--observations",type=Path)
    ap.add_argument("--lock",type=Path)
    ap.add_argument("--preflight",type=Path)
    ap.add_argument("--conditions",nargs="+",choices=tuple(CONDITIONS),default=["B_Main"])
    args=ap.parse_args()
    os.chdir(ROOT)
    run(args)


if __name__=="__main__":main()
