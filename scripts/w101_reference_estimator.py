#!/usr/bin/env python3
"""W101 isolated observation handoff, frozen S0 inference and evaluation.

M in cache keys is M1 (explicit batch_size=1), retained for W100 artifact-reader compatibility.
No production files or historical artifacts are changed. Privileged association and probes are evaluation-only.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
import eval_real_observation_estimator as w100
from framepose.bank import load_bank
from framepose.observations import ObservationProvenance, image_content_digest, observation_cache_key
from run_real_observation_s0 import S0_CHECKPOINT_SHA, S0_PRED_SHA

JOINTS, CANON, DIRECT, SMPL_OF = w100.JOINTS, w100.CANON, w100.DIRECT, w100.SMPL_OF
BANK_DIGEST = "75519e6394a764e3749ddaa30555b58b73a01db582ccee14f661374b9a0ed536"
POPULATION_SHA = "533895c980a080985444b703f43e0f64046391c76909dea5a2f305a26b34cdb9"
CONDITIONS = ("M", "Moa", "Mconf", "Mmask")
CLASSES = ("same_person", "other_person", "ambiguous", "unresolved", "unmatchable", "no_subject")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_s0_identity(path):
    if sha(path) != S0_CHECKPOINT_SHA:
        raise RuntimeError("STOP: S0 checkpoint changed")


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [clean(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def dump(path, value):
    Path(path).write_text(json.dumps(clean(value), indent=2, allow_nan=False))


def to_input(lm, size, *, raw=False):
    """W100 mapping with complete finite-value validation; no GT or actor metadata."""
    if len(size) != 2 or not np.isfinite(size).all() or min(size) <= 0:
        raise ValueError("invalid image size")
    out, valid = np.zeros((17, 3), np.float32), np.zeros(17, bool)
    if lm is None:
        return out, valid
    lm = np.asarray(lm, float)
    if lm.shape != (16, 4):
        raise ValueError("canonical_v1 shape must be (16,4)")
    for j, name in enumerate(JOINTS):
        row = lm[CANON.index(w100.SOURCE_OF[name])]
        if np.isfinite(row).all() and (raw or row[3] > .5):
            out[j] = row[:3] / np.array([size[0], size[1], 1.])
            valid[j] = True
    return out, valid


def association(x, valid, actor_observations):
    """Proxy identity with uniqueness across all declared actors, not a GT identity oracle."""
    distances = {sid: w100.w73_distance(x, valid, b, vb, size) for sid, b, vb, size in actor_observations}
    matches = [sid for sid, d in distances.items() if d is not None and d <= w100.SAME_PERSON]
    if len(matches) == 1:
        return matches[0], "assigned", distances
    if len(matches) > 1:
        return None, "ambiguous", distances
    return None, ("unresolved" if any(d is not None for d in distances.values()) else "unmatchable"), distances


def validate_population(bank, payload):
    bank.assert_split_isolation()
    positions = np.concatenate([bank.indices("validation"), bank.indices("test")])
    rows = [r for r in payload["rows"] if r["population"].startswith("3dpw")]
    by = {int(r["bank_position"]): r for r in rows}
    if len(by) != len(rows) or set(by) != set(map(int, positions)):
        raise ValueError("STOP: population bank positions are not an exact bijection")
    for p in positions:
        s, r = bank.samples[p], by[int(p)]
        if (r["sequence_id"], int(r["frame_index"]), r["population"], r["image"], tuple(r["image_size"])) != (
                s.sequence_id, s.frame_index, "3dpw_" + s.split, "3dpw/" + s.image_reference.relative_path, tuple(s.image_size)):
            raise ValueError("STOP: actor/frame/split/image identity mismatch")
        if s.timestamp is None or s.fps is None or not np.isclose(s.timestamp, s.frame_index/s.fps, atol=1e-12):
            raise ValueError("STOP: timestamp identity missing or inconsistent")
    return positions


def build_observations(bank, payload, observe_dir, image_root):
    """Actual diagnostic data path; no evaluation targets accessed, no outcome-based row filtering."""
    positions = validate_population(bank, payload)
    grouped = defaultdict(list)
    for p in positions:
        s = bank.samples[p]
        grouped[s.image_reference.relative_path].append(int(p))
    cached = {}
    output = {"positions": positions, "sample_id": np.array([bank.samples[p].sample_id for p in positions]),
              "timestamp": np.array([bank.samples[p].timestamp for p in positions])}
    for c in CONDITIONS:
        output["input_2d_"+c] = np.zeros((len(positions), 17, 3), np.float32)
        output["input_valid_"+c] = np.zeros((len(positions), 17), bool)
    side = defaultdict(list)
    for i, p in enumerate(positions):
        s = bank.samples[p]
        video = s.sequence_id.split(":")[1]
        if video not in cached:
            path = observe_dir / (video+".npz")
            z = np.load(path, allow_pickle=False)
            meta = json.loads((observe_dir / (video+".json")).read_text())
            if sha(path) != meta["npz_sha256"]:
                raise ValueError("STOP: observation archive identity failed")
            prov = ObservationProvenance.from_dict(meta["provenance"])
            if prov.regime != "real_animcv_observation" or prov.detail["batch_size"] != 1:
                raise ValueError("STOP: observation is not the M1 reference source")
            frames = z["frame_index"]
            if len(set(map(int, frames))) != len(frames) or np.any(np.diff(frames) <= 0):
                raise ValueError("STOP: duplicate or out-of-order frame identity")
            expected_images = {bank.samples[q].image_reference.relative_path for q in positions
                               if bank.samples[q].sequence_id.split(":")[1] == video}
            if set(x.removeprefix("3dpw/") for x in meta["images"]) != expected_images or len(meta["images"]) != len(frames):
                raise ValueError("STOP: unique image population changed")
            if len(z["cand_frame"]) != len(z["cand_landmarks"]) or np.any((z["cand_frame"] < 0) | (z["cand_frame"] >= len(frames))):
                raise ValueError("STOP: invalid candidate frame index")
            for k, image in enumerate(meta["images"]):
                rel = image.removeprefix("3dpw/")
                if int(Path(rel).stem.split("_")[-1]) != int(frames[k]):
                    raise ValueError("STOP: image/frame mismatch")
                digest = image_content_digest(image_root / rel)
                if digest != meta["image_sha256"][k] or observation_cache_key(prov, digest) != meta["observation_cache_keys"][k]:
                    raise ValueError("STOP: exact RGB byte provenance failed")
            cached[video] = z, {int(f): k for k, f in enumerate(frames)}, meta
        z, lookup, meta = cached[video]
        k = lookup[s.frame_index]
        rel = s.image_reference.relative_path
        if meta["images"][k] != "3dpw/"+rel or not np.isclose(z["timestamp"][k], s.timestamp, atol=1e-12):
            raise ValueError("STOP: sample image/timestamp handoff mismatch")
        if tuple(z["image_size"][k]) != tuple(s.image_size):
            raise ValueError("STOP: frozen image dimensions disagree with decoded RGB")
        lm = z["landmarks"][k]
        present = bool(np.isfinite(lm[:, 0]).any())
        lm = lm if present else None
        x, valid = to_input(lm, s.image_size)
        actors = [(bank.samples[q].sequence_id, bank.arrays["input_2d"][q], bank.arrays["input_valid"][q], bank.samples[q].image_size)
                  for q in grouped[rel]]
        assigned, status, distances = association(x, valid, actors)
        ident = "no_subject" if not present else (
            ("same_person" if assigned == s.sequence_id else "other_person") if assigned else status)
        b, vb = bank.arrays["input_2d"][p], bank.arrays["input_valid"][p]
        output["input_2d_M"][i], output["input_valid_M"][i] = x, valid
        best, bd = None, np.inf
        for c in np.flatnonzero(z["cand_frame"] == k):
            xc, vc = to_input(z["cand_landmarks"][c], s.image_size)
            # Uniqueness also applies to privileged association; no candidate chosen between ambiguous actors.
            a, _, ds = association(xc, vc, actors)
            dc = ds[s.sequence_id]
            if a == s.sequence_id and dc is not None and dc < bd:
                best, bd = (xc, vc), dc
        if best:
            output["input_2d_Moa"][i], output["input_valid_Moa"][i] = best
        for c in ("Mconf", "Mmask"):
            output["input_2d_"+c][i], output["input_valid_"+c][i] = x.copy(), valid.copy()
        if ident == "same_person":
            output["input_2d_Mconf"][i, valid & vb, 2] = b[valid & vb, 2]
            raw, have = to_input(lm, s.image_size, raw=True)
            mask = vb & have
            output["input_2d_Mmask"][i] = np.where(mask[:, None], raw, 0.)
            output["input_valid_Mmask"][i] = mask
        for name, value in {"identity": ident, "assigned_actor": assigned or "", "dist_M": distances[s.sequence_id],
                            "dist_Moa": bd if best else None, "tracked": present, "n_valid_M": int(valid.sum()),
                            "mean_conf_M": float(x[valid, 2].mean()) if valid.any() else None,
                            "moa": best is not None, "candidate_found_tracker_missed": best is not None and ident != "same_person",
                            "image_sha256": meta["image_sha256"][k], "image": rel,
                            "observation_cache_key": meta["observation_cache_keys"][k]}.items():
            side[name].append(np.nan if value is None else value)
    return output, {k: np.asarray(v) for k, v in side.items()}


def geometry_for_condition(bank, obs, condition):
    """Only input arrays replaced; production crop/geometry_tensor is consumed unchanged."""
    from framepose.train import geometry_tensor
    positions = np.asarray(obs["positions"], int)
    expected = np.concatenate([bank.indices("validation"), bank.indices("test")])
    if not np.array_equal(positions, expected):
        raise ValueError("STOP: inference population/order mismatch")
    if not np.array_equal(obs["sample_id"], [bank.samples[p].sample_id for p in positions]):
        raise ValueError("STOP: inference actor identity mismatch")
    if not np.array_equal(obs["timestamp"], [bank.samples[p].timestamp for p in positions]):
        raise ValueError("STOP: inference timestamps mismatch")
    clone = copy.copy(bank)
    clone.arrays = dict(bank.arrays)
    for key in ("input_2d", "input_valid"):
        clone.arrays[key] = bank.arrays[key].copy()
        if condition != "B":
            arr = np.asarray(obs[key+"_"+condition])
            if arr.shape != clone.arrays[key][positions].shape or (key == "input_2d" and not np.isfinite(arr).all()):
                raise ValueError("STOP: invalid S0 observation replacement")
            clone.arrays[key][positions] = arr
    return geometry_tensor(clone)


def video_bootstrap(sids, values):
    """Frame-weighted paired difference, resampling independent videos, retaining shared actors."""
    groups = defaultdict(list)
    for sid, value in zip(sids, values):
        if np.isfinite(value):
            groups[sid.split(":")[1]].append(float(value))
    if not groups:
        return {"n": 0, "videos": 0, "mean": None, "ci95": None}
    ns = np.array([len(v) for v in groups.values()])
    sums = np.array([sum(v) for v in groups.values()])
    result = {"n": int(ns.sum()), "videos": len(groups), "mean": float(sums.sum()/ns.sum()), "ci95": None}
    if len(groups) >= 2:
        draws = np.random.default_rng(1337).integers(0, len(groups), (2000, len(groups)))
        result["ci95"] = np.percentile(sums[draws].sum(1)/ns[draws].sum(1), [2.5, 97.5]).tolist()
    return result


def population_report(bank, obs, side):
    positions = obs["positions"]
    out = {}
    for split in ("validation", "test"):
        ids = [i for i, p in enumerate(positions) if bank.samples[p].split == split]
        images = {str(side["image"][i]): i for i in ids}
        per_actor, per_video = {}, {}
        for sid in sorted({bank.samples[positions[i]].sequence_id for i in ids}):
            ii = [i for i in ids if bank.samples[positions[i]].sequence_id == sid]
            per_actor[sid] = {"rows": len(ii), "identity": dict(Counter(str(side["identity"][i]) for i in ii))}
        for video in sorted({s.split("/")[0] for s in images}):
            ii = [i for image, i in images.items() if image.startswith(video+"/")]
            per_video[video] = {"unique_images": len(ii), "tracked_images": int(side["tracked"][ii].sum()),
                                "assigned_subject_ids": sorted(set(str(side["assigned_actor"][i]) for i in ii) - {""})}
        valid = obs["input_valid_M"][ids]
        conf = obs["input_2d_M"][ids, :, 2]
        bp = positions[ids]
        bv = bank.arrays["input_valid"][bp]
        bc = bank.arrays["input_2d"][bp, :, 2]
        out[split] = {"actor_frames": len(ids), "unique_images": len(images), "videos": len(per_video),
                      "actors": len(per_actor), "one_tracker_stream_per_video": True,
                      "tracked_unique_images": int(sum(side["tracked"][i] for i in images.values())),
                      "identity": {c: int(sum(side["identity"][i] == c for i in ids)) for c in CLASSES},
                      "same_person_share": float(np.mean(side["identity"][ids] == "same_person")),
                      "moa_actor_frames": int(side["moa"][ids].sum()),
                      "candidate_found_tracker_missed": int(side["candidate_found_tracker_missed"][ids].sum()),
                      "missing_joints": dict(zip(JOINTS, (~valid).sum(0).tolist())),
                      "valid_count_histogram": dict(Counter(map(int, valid.sum(1)))),
                      "confidence_valid_quantiles": np.percentile(conf[valid], [0, 5, 25, 50, 75, 95, 100]).tolist() if valid.any() else [],
                      "benchmark_valid_count_histogram": dict(Counter(map(int, bv.sum(1)))),
                      "benchmark_confidence_valid_quantiles": np.percentile(bc[bv], [0, 5, 25, 50, 75, 95, 100]).tolist() if bv.any() else [],
                      "per_video": per_video, "per_actor": per_actor}
    return out


def paired_metrics(rows, a, b):
    """Pairs one scalar on one actor-frame; bilateral conditional means use identical limbs."""
    keys = sorted({k for r in rows for c in (a,b) for k,v in r.get("m",{}).get(c,{}).items()
                   if np.isscalar(v) and not k.startswith("knee_flex_gt")})
    out = {}
    for key in keys:
        pairs = [(r["sid"],r["m"][b][key]-r["m"][a][key]) for r in rows
                 if a in r["m"] and b in r["m"] and key in r["m"][a] and key in r["m"][b]]
        if pairs:
            out[key] = video_bootstrap(*zip(*pairs))
    for base in ("dir_thigh","dir_shin","dir_upper_arm","dir_forearm","knee_plane","knee_wrong"):
        pairs=[]
        for r in rows:
            if a not in r["m"] or b not in r["m"]:continue
            common = [base+"_"+s for s in "lr" if base+"_"+s in r["m"][a] and base+"_"+s in r["m"][b]]
            if common:
                pairs.append((r["sid"],np.mean([r["m"][b][k]-r["m"][a][k] for k in common])))
        if pairs:out[base+"_mean"] = video_bootstrap(*zip(*pairs))
    return out


def temporal(rows, condition):
    """Only original adjacent actor rows; filtering cannot bridge missing identity/observation intervals."""
    from eval_external_hmr2_baseline import LIMBS, KNEES, ang
    from motion.torso_frame import torso_frames
    by = {r["sample_id"]: r for r in rows if condition in r["J"]}
    counts = Counter(steps=0,knee_eligible=0,knee_plane_flips=0,torso_eligible=0,torso_forward_flips=0,
                     limb_eligible=0,limb_direction_jumps=0)
    residuals, dt = [], []
    for a in by.values():
        b=by.get(a["next"])
        if b is None or a["identity"]!="same_person" or b["identity"]!="same_person" or not a["tracked"] or not b["tracked"]:
            continue
        delta = b["timestamp"]-a["timestamp"]
        if delta <= 0 or a["sid"] != b["sid"]:raise ValueError("STOP: invalid temporal alignment")
        counts["steps"] += 1;dt.append(delta)
        Ja,Jb,Ga,Gb=a["J"][condition],b["J"][condition],a["G"],b["G"]
        for h,k,n in KNEES.values():
            normals=[np.cross(j[k]-j[h],j[n]-j[k]) for j in (Ja,Jb,Ga,Gb)]
            if min(np.linalg.norm(v) for v in normals)>1e-3 and ang(normals[2],normals[3])<45:
                counts["knee_eligible"]+=1;counts["knee_plane_flips"]+=int(ang(normals[0],normals[1])>90)
        frames=[torso_frames(j)[0] for j in (Ja,Jb,Ga,Gb)]
        if all(f is not None for f in frames) and ang(frames[2][:,1],frames[3][:,1])<45:
            counts["torso_eligible"]+=1;counts["torso_forward_flips"]+=int(ang(frames[0][:,1],frames[1][:,1])>90)
        for h,k in LIMBS.values():
            if ang(Ga[k]-Ga[h],Gb[k]-Gb[h])<15:
                counts["limb_eligible"]+=1;counts["limb_direction_jumps"]+=int(ang(Ja[k]-Ja[h],Jb[k]-Jb[h])>45)
            residuals.append(float(np.linalg.norm((Jb[k]-Ja[k])-(Gb[k]-Ga[k]))*1000))
    return {**dict(counts),"timestamp_step_range_s":[min(dt),max(dt)] if dt else None,
            "mean_limb_joint_motion_residual_mm":float(np.mean(residuals)) if residuals else None}


def two_d_metrics(x, v, uv, size, shared=None):
    joints=[j for j in DIRECT if v[j] and (shared is None or shared[j])]
    gt12=np.array([uv[SMPL_OF[j]] for j in DIRECT])
    diagonal=float(np.linalg.norm(gt12.max(0)-gt12.min(0)))
    if diagonal<=0 or not np.isfinite(gt12).all():raise ValueError("STOP: invalid evaluation 2D reference")
    error=np.array([np.linalg.norm(x[j,:2]*size-uv[SMPL_OF[j]])/diagonal for j in joints])
    return {"joints":len(joints),"err":float(error.mean()) if len(error) else None,
            "pck05":float(np.mean(error<=.05)) if len(error) else None,
            "pck10":float(np.mean(error<=.10)) if len(error) else None,
            "per_joint":{JOINTS[j]:float(e) for j,e in zip(joints,error)}}


def inventory(bank, payload, image_root):
    """Selection-free RGB inventory only; performs no detection, S0 or GT evaluation."""
    from PIL import Image
    positions=validate_population(bank,payload)
    images={};videos=defaultdict(lambda:{"actor_frames":0,"frames":set(),"actors":set(),"split":None})
    rows=[]
    for p in positions:
        s=bank.samples[p];rel=s.image_reference.relative_path;video=s.sequence_id.split(":")[1]
        if rel not in images:
            path=image_root/rel
            with Image.open(path) as image:size=image.size
            if tuple(size)!=tuple(s.image_size):raise ValueError("STOP: RGB dimensions differ from frozen bank")
            images[rel]={"sha256":image_content_digest(path),"image_size":list(size),"frame_index":s.frame_index,
                         "timestamp":s.timestamp,"actors":[]}
        images[rel]["actors"].append(s.sequence_id)
        v=videos[video];v["actor_frames"]+=1;v["frames"].add(s.frame_index);v["actors"].add(s.sequence_id);v["split"]=s.split
        rows.append({"bank_position":int(p),"sample_id":s.sample_id,"sequence_id":s.sequence_id,"frame_index":s.frame_index,
                     "timestamp":s.timestamp,"split":s.split,"image":rel,"image_sha256":images[rel]["sha256"]})
    per_video={k:{"actor_frames":v["actor_frames"],"unique_images":len(v["frames"]),"actors":sorted(v["actors"]),
                  "split":v["split"],"frame_indices":sorted(v["frames"])} for k,v in videos.items()}
    splits={split:{"actor_frames":sum(r["split"]==split for r in rows),
                   "unique_images":len({r["image"] for r in rows if r["split"]==split}),
                   "actors":len({r["sequence_id"] for r in rows if r["split"]==split}),
                   "videos":sum(v["split"]==split for v in per_video.values())} for split in ("validation","test")}
    return {"actor_frames":len(rows),"unique_images":len(images),"videos":len(per_video),"splits":splits,
            "per_video":per_video,"images":images,"rows":rows,"observation_inference_executed":False}


def inference(args):
    import torch
    from framepose.train import load_checkpoint,predict
    from framepose.signs import neutral_sign_states
    if sha(args.checkpoint)!=S0_CHECKPOINT_SHA:raise RuntimeError("STOP: S0 checkpoint changed")
    bank=load_bank(args.bank)
    if bank.content_digest()!=BANK_DIGEST:raise RuntimeError("STOP: bank changed")
    if not torch.cuda.is_available():raise RuntimeError("STOP: original CUDA inference device unavailable")
    model,ck=load_checkpoint(args.checkpoint,device="cuda")
    if ck["candidate"]["name"]!="S0_neutral_sign":raise RuntimeError("STOP: S0 candidate identity mismatch")
    obs=np.load(args.observations,allow_pickle=False)
    report={"checkpoint_sha256":sha(args.checkpoint),"bank_content_digest":bank.content_digest(),
            "observations_sha256":sha(args.observations),"device":"cuda","predict_batch_size":512,"amp_enabled":False,"neutral_signs":True,
            "software":{"torch":torch.__version__,"numpy":np.__version__},"conditions":{}}
    for c in ("B",)+CONDITIONS:
        geometry=geometry_for_condition(bank,obs,c)
        report["conditions"][c]={}
        for split in ("validation","test"):
            pred=predict(model,torch,geometry,None,bank.indices(split),torch.device("cuda"),signs=neutral_sign_states(len(bank))).astype(np.float32)
            path=args.out_dir/f"prediction_{c}_{split}.npy";np.save(path,pred)
            report["conditions"][c][split]={"rows":len(pred),"sha256":sha(path)}
            if c=="B" and sha(path)!=S0_PRED_SHA[split]:
                dump(args.out_dir/"s0_report.json",report)
                raise RuntimeError("STOP: historical S0 reproduction failed")
        dump(args.out_dir/"s0_report.json",report)
        print("completed",c,flush=True)


def evaluation_reference(raw_root, sample, cache):
    """Strict raw frame lookup and projection; targets only enter the evaluator."""
    import pickle
    from run_external_hmr2_baseline import smpl24_to_canonical17
    from run_pelvis_torso_ownership import CV_TO_CANON
    name=sample.sequence_id.split(":")[1];actor=int(sample.sequence_id.split(":")[2][5:])
    if name not in cache:
        # Split identity is explicit; do not search another split or silently fall back to a row index.
        path=raw_root/"sequenceFiles"/sample.split/(name+".pkl")
        with path.open("rb") as stream:cache[name]=pickle.load(stream,encoding="latin1")
    d=cache[name];ids=np.asarray(d["img_frame_ids"])
    match=np.flatnonzero(ids==sample.frame_index)
    if len(match)!=1:raise ValueError("STOP: raw frame identity not unique")
    k=int(match[0]);X=np.asarray(d["jointPositions"][actor][k],float).reshape(24,3)
    T=np.asarray(d["cam_poses"][k],float)
    Xc=X@T[:3,:3].T+T[:3,3]
    uv=Xc@np.asarray(d["cam_intrinsics"],float).T
    uv=uv[:,:2]/uv[:,2:3]
    gravity=CV_TO_CANON@T[:3,:3]@np.array([0.,1.,0.]);gravity/=np.linalg.norm(gravity)
    return uv,smpl24_to_canonical17(Xc),gravity


def evaluate(args):
    import eval_external_hmr2_baseline as w97
    import run_pelvis_torso_ownership as w94
    from run_external_hmr2_baseline import smpl24_to_canonical17
    from run_external_orientation_handoff import W97_PREDICTIONS_SHA
    from motion.torso_frame import torso_frames
    bank=load_bank(args.bank)
    if bank.content_digest()!=BANK_DIGEST or sha(args.population)!=POPULATION_SHA:raise RuntimeError("STOP: evaluation identities changed")
    obs=np.load(args.observations,allow_pickle=False)
    side=np.load(str(args.observations).replace(".npz","_side.npz"),allow_pickle=False)
    positions=validate_population(bank,json.loads(args.population.read_text()))
    if not np.array_equal(positions,obs["positions"]):raise RuntimeError("STOP: evaluation row alignment")
    report_s0=json.loads((args.s0_dir/"s0_report.json").read_text())
    predictions={}
    for c in ("B",)+CONDITIONS:
        predictions[c]={}
        for split in ("validation","test"):
            path=args.s0_dir/f"prediction_{c}_{split}.npy"
            if sha(path)!=report_s0["conditions"][c][split]["sha256"]:raise RuntimeError("STOP: prediction file changed")
            if c=="B" and sha(path)!=S0_PRED_SHA[split]:raise RuntimeError("STOP: B prediction identity failed")
            pred=np.load(path,allow_pickle=False)
            if len(pred)!=len(bank.indices(split)):raise RuntimeError("STOP: prediction length mismatch")
            predictions[c].update(zip(map(int,bank.indices(split)),pred))
    predictions["E"]={}
    if args.hmr_predictions:
        if sha(args.hmr_predictions)!=W97_PREDICTIONS_SHA:raise RuntimeError("STOP: frozen HMR2 predictions changed")
        hp=np.load(args.hmr_predictions,allow_pickle=False)
        pop=json.loads(args.population.read_text())["rows"]
        if len(hp["population"])!=len(pop):raise RuntimeError("STOP: HMR population length mismatch")
        for i,row in enumerate(pop):
            if str(hp["population"][i])!=row["population"] or int(hp["index"][0][i])!=row["bank_position"]:
                raise RuntimeError("STOP: HMR actor/frame association mismatch")
            if row["population"].startswith("3dpw") and hp["valid"][i]:
                predictions["E"][row["bank_position"]]=smpl24_to_canonical17(hp["joints24"][i])
    rows=[];raw_cache={};target_unavailable=[]
    for i,p in enumerate(positions):
        p=int(p);s=bank.samples[p]
        if not bank.arrays["target_valid"][p].all():target_unavailable.append(s.sample_id);continue
        G=bank.arrays["target_3d"][p].astype(float);G-=G[0]
        uv,Graw,g=evaluation_reference(args.raw_3dpw,s,raw_cache)
        if not np.allclose(G,Graw,atol=1e-6,rtol=0):raise RuntimeError("STOP: raw and frozen GT coordinate/frame mismatch")
        Rg,_=torso_frames(G)
        trunk=w94.angle(Rg[:,2],g) if Rg is not None else None
        thigh=.5*(w94.angle(G[2]-G[1],-g)+w94.angle(G[5]-G[4],-g))
        row={"p":p,"sid":s.sequence_id,"sample_id":s.sample_id,"split":s.split,"frame":s.frame_index,
             "timestamp":s.timestamp,"next":s.neighbors.get("next"),"identity":str(side["identity"][i]),
             "tracked":bool(side["tracked"][i]),"stratum":w94.stratum(trunk,thigh),"G":G,"m":{},"J":{},"twod":{},
             "mean_conf_M":float(side["mean_conf_M"][i]),"moa":bool(side["moa"][i]),
             "bench_invalid_limb":int((~bank.arrays["input_valid"][p][DIRECT]).sum()),"dist_M":float(side["dist_M"][i])}
        xb,vb=bank.arrays["input_2d"][p],bank.arrays["input_valid"][p]
        for c in ("B","M","Moa"):
            x,v=(xb,vb) if c=="B" else (obs["input_2d_"+c][i],obs["input_valid_"+c][i])
            row["twod"][c]=two_d_metrics(x,v,uv,s.image_size)
            if c in ("B","M"):
                shared=obs["input_valid_M"][i]&vb
                row["twod"][c+"_shared"]=two_d_metrics(x,v,uv,s.image_size,shared)
        for c in predictions:
            if p not in predictions[c]:continue
            J=predictions[c][p].astype(float);J-=J[0]
            if not np.isfinite(J).all():raise RuntimeError("STOP: nonfinite 3D output")
            m=w97.source_metrics(J,G);m["per_joint"]=(1000*np.linalg.norm(J-G,axis=1)).tolist()
            m["depth_sign_err"],m["depth_sign_n"]=w100.depth_sign_errors(J,G)
            m["severe"]=float(m["mpjpe"]>200)
            row["J"][c],row["m"][c]=J,m
        rows.append(row)
    report={"regimes":{"B":"benchmark_detector_observation","M":"real_animcv_observation; M1 batch_size=1",
                       "Moa":"privileged actor association","Mconf/Mmask":"evaluation-only representation interventions",
                       "E":"frozen HMR2 contextual baseline; not an RGB-only causal control"},
            "coverage":population_report(bank,obs,side),"evaluation_target_unavailable":target_unavailable,
            "input_hashes":{"observations":sha(args.observations),"population":sha(args.population),
                            "s0_report":sha(args.s0_dir/"s0_report.json")},"results":{}}
    for split in ("validation","test"):
        R=[r for r in rows if r["split"]==split]
        report["results"][split]=summarize(R)
    dump(args.out_dir/"report.json",report)
    # JSON contains no binary arrays; diagnostic predictions and references are explicitly evaluation artifacts.
    dump(args.out_dir/"rows.json",rows)
    print(json.dumps({s:report["results"][s]["headline"] for s in report["results"]}),flush=True)


def summarize(rows):
    same=[r for r in rows if r["identity"]=="same_person"]
    moa=[r for r in rows if r["moa"]]
    conditions=("B",)+CONDITIONS+("E",)
    def aggregates(rs,c):
        result=w100.agg(rs,c)
        result["scalar_denominators"]={k:sum(k in r["m"].get(c,{}) for r in rs) for k in w100.KEYS}
        return result
    out={"evaluation_rows":len(rows),"same_person_rows":len(same),
         "all_rows_diagnostic_including_missing_input":{c:aggregates(rows,c) for c in conditions},
         "same_person":{c:aggregates(same,c) for c in conditions},
         "same_person_paired":{c+"_minus_B":paired_metrics(same,"B",c) for c in CONDITIONS+("E",)},
         "probe_vs_M":{c+"_minus_M":paired_metrics(same,"M",c) for c in ("Mconf","Mmask")},
         "moa_paired":paired_metrics(moa,"B","Moa"),
         "by_identity":{c:{"rows":sum(r["identity"]==c for r in rows),
                            "M":aggregates([r for r in rows if r["identity"]==c],"M")} for c in CLASSES},
         "temporal_same_person":{c:temporal(same,c) for c in ("B","M","E")},"two_d":{},"strata":{},"challenge":{}}
    for c in ("B","M","Moa","B_shared","M_shared"):
        r2=[r["twod"][c] for r in same if c in r["twod"] and r["twod"][c]["joints"]]
        out["two_d"][c]={"rows":len(r2),"joint_observations":sum(r["joints"] for r in r2),
                            **{k:float(np.mean([r[k] for r in r2])) if r2 else None for k in ("err","pck05","pck10")}}
    for k in ("err","pck05","pck10"):
        pairs=[(r["sid"],r["twod"]["M_shared"][k]-r["twod"]["B_shared"][k]) for r in same
               if r["twod"]["M_shared"][k] is not None and r["twod"]["B_shared"][k] is not None]
        if pairs:out["two_d"]["paired_"+k]=video_bootstrap(*zip(*pairs))
    masks={"gt_knee_flex_ge_45":lambda r:any(r["m"]["B"].get("knee_flex_gt_"+s,0)>=45 for s in "lr"),
           "gt_knee_near_straight":lambda r:all(r["m"]["B"].get("knee_flex_gt_"+s,99)<5 for s in "lr"),
           "s0_wrong_knee_plane":lambda r:any(r["m"]["B"].get("knee_plane_"+s,0)>45 for s in "lr"),
           "out_of_plane_limbs":lambda r:any(abs((r["G"][b]-r["G"][a])[1])/max(np.linalg.norm(r["G"][b]-r["G"][a]),1e-9)>.5
                                             for a,b in ((1,2),(4,5),(11,12),(14,15))),
           "occluded_benchmark":lambda r:r["bench_invalid_limb"]>=2,
           "low_confidence_M":lambda r:np.isfinite(r["mean_conf_M"]) and r["mean_conf_M"]<.5}
    for stratum in ("upright","waist_bend","lying","other"):
        sub=[r for r in same if r["stratum"]==stratum]
        out["strata"][stratum]={"rows":len(sub),"B":aggregates(sub,"B"),"M":aggregates(sub,"M"),
                                  "paired":paired_metrics(sub,"B","M")}
    for name,predicate in masks.items():
        all_sub=[r for r in rows if predicate(r)];sub=[r for r in same if predicate(r)]
        out["challenge"][name]={"full_stratum_rows":len(all_sub),"same_person_rows":len(sub),
                                 "B":aggregates(sub,"B"),"M":aggregates(sub,"M"),"E":aggregates(sub,"E"),
                                 "paired":paired_metrics(sub,"B","M")}
    sequences=defaultdict(list)
    for r in same:sequences[r["sid"]].append(r)
    out["per_actor"]={sid:{"rows":len(rs),"B":aggregates(rs,"B"),"M":aggregates(rs,"M"),
                             "paired":paired_metrics(rs,"B","M")} for sid,rs in sequences.items()}
    out["headline"]={"B_mpjpe_same":out["same_person"]["B"].get("mpjpe"),"M_mpjpe_same":out["same_person"]["M"].get("mpjpe"),
                      "M_minus_B_same":out["same_person_paired"]["M_minus_B"].get("mpjpe")}
    return out



def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=("build","inventory","infer","evaluate"))
    for key in ("bank", "out_dir"):
        ap.add_argument("--"+key.replace("_", "-"), type=Path, required=True)
    for key in ("population","observe_dir","image_root","checkpoint","observations","s0_dir","raw_3dpw","hmr_predictions"):
        ap.add_argument("--"+key.replace("_", "-"), type=Path)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=False)
    if args.mode=="infer":return inference(args)
    if args.mode=="evaluate":return evaluate(args)
    bank = load_bank(args.bank)
    if bank.content_digest() != BANK_DIGEST:
        raise RuntimeError("STOP: frozen FramePose bank changed")
    # W97's full source manifest identity is already pinned in its downstream consumers.
    if sha(args.population) != POPULATION_SHA:
        raise RuntimeError("STOP: frozen population changed")
    if args.mode=="inventory":
        result=inventory(bank,json.loads(args.population.read_text()),args.image_root)
        result.update(bank_content_digest=bank.content_digest(),population_sha256=sha(args.population))
        dump(args.out_dir/"population_manifest.json",result)
        print(json.dumps({k:result[k] for k in ("actor_frames","unique_images","videos","splits")}))
        return
    obs, side = build_observations(bank, json.loads(args.population.read_text()), args.observe_dir, args.image_root)
    np.savez_compressed(args.out_dir / "observations.npz", **obs)
    np.savez_compressed(args.out_dir / "observations_side.npz", **side)
    dump(args.out_dir / "population_report.json", population_report(bank, obs, side))


if __name__ == "__main__":
    main()
