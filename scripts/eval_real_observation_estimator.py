#!/usr/bin/env python3
"""EXPERIMENT (Worklog 100): does frozen S0 survive the switch from benchmark 2D to AnimCV's own RTMDet/RTMPose 2D?

``build-observations``  maps the AnimCV observations (``run_real_observation_mmpose.py observe``) onto the frozen 3DPW
    VALIDATION/TEST bank rows with the SAME representation rules the benchmark bank uses (17 joints; head = nose; neck = thorax
    = shoulder midpoint; pelvis = hip midpoint; spine = midpoint(neck, pelvis); midpoint confidence = min; x/W, y/H with the
    sample's recorded image size; invalid joints written as zeros).  Validity = the production visibility flag (score >= 0.3).
    Conditions (all GT-free in construction):
      M      the production tracked subject (full detector/tracker outcome)
      Moa    evaluation-only ORACLE-ASSOCIATED control: the RTMDet candidate whose RTMPose landmarks are closest to the
             benchmark observation of this dataset actor (W73 association rule) -- uses the dataset actor association
      Mconf  M with the benchmark confidence values on joints valid in both (representation probe)
      Mmask  M's own coordinates/confidences with the benchmark validity mask (representation probe)
``evaluate``  frozen S0 predictions (``run_real_observation_s0.py``) for B, M, Moa, Mconf, Mmask, plus the frozen W97 HMR 2.0b
    predictions (E) as an external RGB reference, against the bank's 3DPW 3D reference (evaluation only); GT 2D = 3DPW SMPL
    joints projected with the dataset camera (evaluation only).  ``PREDECLARED`` was fixed before any comparison was made.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

PREDECLARED = {
    "representation": "benchmark rules (three_dpw_adapter + temporal_lifter.build_dataset): head=nose, neck=thorax=shoulder "
                      "midpoint, pelvis=hip midpoint, spine=midpoint(neck, pelvis), min confidence, x/W y/H with the bank's "
                      "image size, invalid -> zeros; validity = production visible flag (RTMPose score >= 0.3, MMPoseConfig "
                      "default); confidence = RTMPose score, unchanged",
    "no_subject": "no tracked detection in the frame -> all joints invalid (row kept in every denominator)",
    "same_person": "W73 association distance (median pixel distance over >= 4 jointly valid limb joints / benchmark valid-joint "
                   "bbox diagonal) <= 0.10",
    "identity_classes": ["same_person", "other_person", "unmatchable", "no_subject"],  # unmatchable: < 4 shared joints
    "oracle_association": "Moa: candidate with the smallest W73 distance to the benchmark observation, accepted if <= 0.10; "
                          "evaluation-only control (dataset actor association), never a runtime candidate",
    "probes": "Mconf and Mmask on the same-person rows (representation attribution); no GT enters either",
    "metrics": "W97 source_metrics (root-relative MPJPE, PA-MPJPE, limb directions, knee bend-plane error and wrong branch, "
               "torso up/forward, forward flip) + per-joint error + limb depth-sign error (|GT depth share| >= 0.25 and the "
               "sign of the limb's depth component differs) + severe outlier MPJPE > 200 mm",
    "two_d": "per-joint 2D error vs projected GT (12 limb joints, SMPL joint semantics) normalised by the GT limb-joint bbox "
             "diagonal; PCK@0.05 and @0.10; descriptive (detector keypoints and SMPL joints differ by definition, equally for "
             "B and M)",
    "strata": "W94 gravity strata (trunk/thigh tilt vs the 3DPW camera gravity, evaluation only) and W97 challenge subsets "
              "(waist bend, seated/other, GT knee flexion >= 45, near-straight, out-of-plane limbs, historical S0 wrong "
              "knee plane = B knee-plane error > 45), occlusion proxy = >= 2 benchmark limb joints invalid, low-confidence "
              "proxy = mean M visible-joint score < 0.5",
    "temporal": "adjacent bank rows of one actor: knee-plane flips (normal step > 90 deg while GT < 45), torso forward flips "
                "(step > 90 while GT < 45), limb direction jumps (> 45 deg while GT < 15)",
    "uncertainty": "sequence-cluster bootstrap 95% intervals (_brief), paired per row",
    "decision": "A: on same-person rows M - B MPJPE < +10 mm and limb-direction / knee-plane changes < +2 deg, coverage (same "
                "person) >= 90%, no new failure class; B: material degradation (MPJPE > +10 mm or coverage < 90% or identity "
                "failures > 5%) attributable to observation distribution / tracking / representation (probes, 2D accuracy); "
                "C: M comparable to B (criteria of A) but large 3D failures persist under both (B MPJPE, wrong knee branches, "
                "depth-sign errors) relative to the RGB reference; D: gate or contract blocked",
}
JOINTS = ["pelvis", "left_hip", "left_knee", "left_ankle", "right_hip", "right_knee", "right_ankle", "spine", "thorax", "neck",
          "head", "left_shoulder", "left_elbow", "left_wrist", "right_shoulder", "right_elbow", "right_wrist"]
CANON = ["head", "neck", "pelvis", "spine", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist",
         "right_wrist", "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle"]
SOURCE_OF = {"pelvis": "pelvis", "spine": "spine", "thorax": "neck", "neck": "neck", "head": "head",
             **{n: n for n in JOINTS if n.startswith(("left_", "right_"))}}
DIRECT = [JOINTS.index(n) for n in ("left_shoulder", "left_elbow", "left_wrist", "right_shoulder", "right_elbow", "right_wrist",
                                     "left_hip", "left_knee", "left_ankle", "right_hip", "right_knee", "right_ankle")]
SMPL_OF = {1: 1, 2: 4, 3: 7, 4: 2, 5: 5, 6: 8, 11: 16, 12: 18, 13: 20, 14: 17, 15: 19, 16: 21}  # bank joint -> SMPL joint
SAME_PERSON = 0.10


def to_input(lm, size):
    """(16, 4) [x, y, conf, visible] pixel landmarks -> bank (17, 3) [x/W, y/H, conf] and (17,) validity."""
    W, H = float(size[0]), float(size[1])
    x = np.zeros((17, 3), np.float32)
    v = np.zeros(17, bool)
    if lm is None:
        return x, v
    for j, n in enumerate(JOINTS):
        row = lm[CANON.index(SOURCE_OF[n])]
        if np.isfinite(row[0]) and row[3] > 0.5:
            x[j] = (row[0] / W, row[1] / H, row[2])
            v[j] = True
    return x, v


def raw_coords(lm, size):
    """Coordinates/confidences of every present landmark regardless of visibility (for the Mmask probe)."""
    W, H = float(size[0]), float(size[1])
    x = np.zeros((17, 3), np.float32)
    have = np.zeros(17, bool)
    if lm is None:
        return x, have
    for j, n in enumerate(JOINTS):
        row = lm[CANON.index(SOURCE_OF[n])]
        if np.isfinite(row[0]):
            x[j] = (row[0] / W, row[1] / H, row[2])
            have[j] = True
    return x, have


def w73_distance(a, va, b, vb, size):
    """Median pixel distance over >= 4 jointly valid limb joints / bbox diagonal of b's valid joints (b = benchmark)."""
    W, H = float(size[0]), float(size[1])
    both = [j for j in DIRECT if va[j] and vb[j]]
    bp = b[vb][:, :2] * (W, H)
    if len(both) < 4 or len(bp) < 2:
        return None
    scale = float(np.linalg.norm(bp.max(0) - bp.min(0)))
    d = np.linalg.norm((a[both, :2] - b[both, :2]) * (W, H), axis=1)
    return float(np.median(d) / scale) if scale > 0 else None


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def build_observations(args):
    from framepose.bank import load_bank

    bank = load_bank(args.bank)
    pos = np.concatenate([bank.indices("validation"), bank.indices("test")])
    cache = {}
    out = {k: np.zeros((len(pos), 17, 3), np.float32) for k in ("M", "Moa", "Mconf", "Mmask")}
    val = {k: np.zeros((len(pos), 17), bool) for k in out}
    side = defaultdict(list)
    for i, p in enumerate(pos):
        s = bank.samples[p]
        video = s.sequence_id.split(":")[1]
        if video not in cache:
            z = np.load(args.observe_dir / f"{video}.npz")
            meta = json.loads((args.observe_dir / f"{video}.json").read_text())
            cache[video] = (z, {int(f): k for k, f in enumerate(z["frame_index"])}, meta)
        z, idx, meta = cache[video]
        k = idx[int(s.frame_index)]
        rel = s.image_reference.relative_path
        if meta["images"][k][len("3dpw/"):] != rel:
            raise SystemExit(f"STOP: image identity mismatch {rel}")
        size = tuple(s.image_size)
        b, vb = bank.arrays["input_2d"][p], bank.arrays["input_valid"][p]
        lm = z["landmarks"][k] if np.isfinite(z["landmarks"][k][:, 0]).any() else None
        out["M"][i], val["M"][i] = to_input(lm, size)
        d = w73_distance(out["M"][i], val["M"][i], b, vb, size) if lm is not None else None
        ident = "no_subject" if lm is None else ("unmatchable" if d is None else ("same_person" if d <= SAME_PERSON else "other_person"))
        # oracle-associated candidate
        best, bd = None, np.inf
        for c in np.flatnonzero(z["cand_frame"] == k):
            xc, vc = to_input(z["cand_landmarks"][c], size)
            dc = w73_distance(xc, vc, b, vb, size)
            if dc is not None and dc < bd:
                best, bd = (xc, vc), dc
        if best is not None and bd <= SAME_PERSON:
            out["Moa"][i], val["Moa"][i] = best
        # representation probes (same-person rows only; elsewhere = M)
        out["Mconf"][i], val["Mconf"][i] = out["M"][i].copy(), val["M"][i].copy()
        out["Mmask"][i], val["Mmask"][i] = out["M"][i].copy(), val["M"][i].copy()
        if ident == "same_person":
            both = val["M"][i] & vb
            out["Mconf"][i][both, 2] = b[both, 2]
            rc, have = raw_coords(lm, size)
            m = vb & have
            out["Mmask"][i] = np.where(m[:, None], rc, 0.0).astype(np.float32)
            val["Mmask"][i] = m
        side["identity"].append(ident)
        side["dist_M"].append(np.nan if d is None else d)
        side["dist_Moa"].append(bd if best is not None else np.nan)
        side["n_valid_M"].append(int(val["M"][i].sum()))
        side["n_valid_B"].append(int(vb.sum()))
        side["mean_conf_M"].append(float(out["M"][i][val["M"][i], 2].mean()) if val["M"][i].any() else np.nan)
        side["image_size_matches_jpeg"].append(bool(tuple(z["image_size"][k]) == size))
        side["image_sha256"].append(meta["image_sha256"][k])
    np.savez_compressed(args.out, positions=pos, **{f"input_2d_{k}": v for k, v in out.items()}, **{f"input_valid_{k}": v for k, v in val.items()})
    sidep = Path(str(args.out).replace(".npz", "_side.npz"))
    np.savez_compressed(sidep, positions=pos, **{k: np.array(v) for k, v in side.items()})
    summ = {"rows": int(len(pos)), "identity": {c: int(sum(x == c for x in side["identity"])) for c in PREDECLARED["identity_classes"]},
            "moa_available": int(val["Moa"].any(1).sum()), "image_size_mismatch": int(len(pos) - sum(side["image_size_matches_jpeg"])),
            "observations_sha256": sha(args.out)}
    print(json.dumps(summ))
    (Path(str(args.out).replace(".npz", "_summary.json"))).write_text(json.dumps(summ, indent=1))


# ------------------------------------------------------------------ evaluation

def gt_2d(raw_root, sid, frame, cache):
    from run_root_motion_contact_h0_replay import _find_3dpw_pkl, _parse_3dpw_sequence_id

    name, actor = _parse_3dpw_sequence_id(sid)
    if name not in cache:
        with _find_3dpw_pkl(raw_root, name).open("rb") as h:
            d = pickle.load(h, encoding="latin1")
        cache[name] = d
    d = cache[name]
    ids = np.asarray(d["img_frame_ids"])
    i = int(np.flatnonzero(ids == frame)[0]) if (ids == frame).any() else int(frame)
    X = np.asarray(d["jointPositions"][actor][i], float).reshape(24, 3)
    T = np.asarray(d["cam_poses"][i], float)
    Xc = (T[:3, :3] @ X.T + T[:3, 3:4]).T
    uv = (np.asarray(d["cam_intrinsics"], float) @ Xc.T).T
    return uv[:, :2] / uv[:, 2:3]


def depth_sign_errors(J, G):
    from eval_external_hmr2_baseline import LIMBS
    n = c = 0
    for a, b in LIMBS.values():
        g, j = G[b] - G[a], J[b] - J[a]
        if abs(g[1]) >= 0.25 * np.linalg.norm(g):
            n += 1
            c += int(np.sign(g[1]) != np.sign(j[1]))
    return c, n


def evaluate(args):
    import eval_external_hmr2_baseline as w97e
    import run_external_hmr2_baseline as hmr
    import run_pelvis_torso_ownership as w94
    from run_visual_depth_motion_attribution import _brief

    from framepose.bank import load_bank
    from motion.torso_frame import torso_frames

    bank = load_bank(args.bank)
    obs = np.load(args.observations)
    side = np.load(str(args.observations).replace(".npz", "_side.npz"))
    pos = obs["positions"]
    conds = ["B", "M", "Moa", "Mconf", "Mmask"]
    pred = {}
    for c in conds:
        v = np.load(args.s0_dir / f"prediction_{c}_validation.npy")
        t = np.load(args.s0_dir / f"prediction_{c}_test.npy")
        full = {}
        for split, arr in (("validation", v), ("test", t)):
            for k, p in enumerate(bank.indices(split)):
                full[int(p)] = arr[k]
        pred[c] = full
    # HMR 2.0b (W97 frozen predictions) on the same bank rows
    hp = np.load(args.hmr_predictions)
    E = {}
    for i, (pp, bp) in enumerate(zip(hp["population"], hp["index"][0])):
        if str(pp).startswith("3dpw") and hp["valid"][i]:
            E[int(bp)] = hmr.smpl24_to_canonical17(hp["joints24"][i])
    smpl = w94.Smpl3dpw(args.raw_3dpw)
    cache2d = {}
    rows = []
    for i, p in enumerate(pos):
        p = int(p)
        s = bank.samples[p]
        G = bank.arrays["target_3d"][p].astype(float)
        if not bank.arrays["target_valid"][p].all():
            continue
        G = G - G[0]
        _, g = smpl.get(s.sequence_id, int(s.frame_index))
        Rg, _ = torso_frames(G)
        trunk_g = w94.angle(Rg[:, 2], g) if (Rg is not None and g is not None) else None
        thigh_g = (0.5 * (w94.angle(G[2] - G[1], -g) + w94.angle(G[5] - G[4], -g))) if g is not None else None
        r = {"p": p, "sid": s.sequence_id, "frame": int(s.frame_index), "split": None,
             "identity": str(side["identity"][i]), "dist_M": float(side["dist_M"][i]), "moa": bool(obs["input_valid_Moa"][i].any()),
             "stratum": w94.stratum(trunk_g, thigh_g), "mean_conf_M": float(side["mean_conf_M"][i]),
             "bench_invalid_limb": int((~bank.arrays["input_valid"][p][DIRECT]).sum()), "m": {}, "J": {}}
        uv = gt_2d(args.raw_3dpw, s.sequence_id, int(s.frame_index), cache2d)
        W, H = s.image_size
        gt12 = np.array([uv[SMPL_OF[j]] for j in DIRECT])
        diag = float(np.linalg.norm(gt12.max(0) - gt12.min(0)))
        r["twod"] = {}
        for c, (x2, v2) in (("B", (bank.arrays["input_2d"][p], bank.arrays["input_valid"][p])), ("M", (obs["input_2d_M"][i], obs["input_valid_M"][i])),
                            ("Moa", (obs["input_2d_Moa"][i], obs["input_valid_Moa"][i]))):
            e = [float(np.linalg.norm(x2[j, :2] * (W, H) - uv[SMPL_OF[j]]) / diag) for j in DIRECT if v2[j]]
            r["twod"][c] = {"err": float(np.mean(e)) if e else None, "pck05": float(np.mean([x <= 0.05 for x in e])) if e else None,
                            "pck10": float(np.mean([x <= 0.10 for x in e])) if e else None, "n": len(e)}
        for c in conds + ["E"]:
            J = pred[c].get(p) if c != "E" else E.get(p)
            if J is None:
                continue
            J = np.asarray(J, float) - np.asarray(J, float)[0]
            m = w97e.source_metrics(J, G)
            m["per_joint"] = (1000 * np.linalg.norm(J - G, axis=1)).tolist()
            m["depth_sign_err"], m["depth_sign_n"] = depth_sign_errors(J, G)
            m["severe"] = float(m["mpjpe"] > 200)
            r["m"][c], r["J"][c] = m, J
        r["G"] = G
        rows.append(r)
    split_of = {int(p): "validation" for p in bank.indices("validation")}
    split_of.update({int(p): "test" for p in bank.indices("test")})
    for r in rows:
        r["split"] = split_of[r["p"]]
    report = {"predeclared": PREDECLARED, "observations_sha256": sha(args.observations), "results": {}}
    for split in ("validation", "test"):
        R = [r for r in rows if r["split"] == split]
        report["results"][split] = split_report(R, conds, w97e, _brief)
    (args.out_dir / "report.json").write_text(json.dumps(report, indent=1, default=float))
    print(json.dumps({s: {k: report["results"][s][k] for k in ("coverage", "headline")} for s in report["results"]}, indent=1, default=float))
    if args.cases:
        np.save(args.out_dir / "rows_minimal.npy", np.array([{"p": r["p"], "sid": r["sid"], "frame": r["frame"], "identity": r["identity"],
                                                              "split": r["split"], "mpjpe": {c: r["m"][c]["mpjpe"] for c in r["m"]}} for r in rows], dtype=object))


KEYS = ("mpjpe", "pa_mpjpe", "depth_err", "dir_thigh_l", "dir_thigh_r", "dir_shin_l", "dir_shin_r", "dir_upper_arm_l", "dir_upper_arm_r",
        "dir_forearm_l", "dir_forearm_r", "knee_plane_l", "knee_plane_r", "knee_wrong_l", "knee_wrong_r", "torso_up", "torso_fwd",
        "torso_fwd_flip", "severe")


def agg(R, c):
    out = {"rows": sum(c in r["m"] for r in R)}
    for k in KEYS:
        v = [r["m"][c][k] for r in R if c in r["m"] and k in r["m"][c]]
        if v:
            out[k] = float(np.mean(v))
    pj = [r["m"][c]["per_joint"] for r in R if c in r["m"]]
    if pj:
        out["per_joint_mm"] = dict(zip(JOINTS, np.mean(pj, axis=0).round(1).tolist()))
    ds = [(r["m"][c]["depth_sign_err"], r["m"][c]["depth_sign_n"]) for r in R if c in r["m"]]
    if ds:
        out["depth_sign_error_rate"] = float(sum(a for a, _ in ds) / max(1, sum(b for _, b in ds)))
    return out


def paired(R, a, b, brief):
    res = {}
    both = [r for r in R if a in r["m"] and b in r["m"]]
    for k in ("mpjpe", "pa_mpjpe", "depth_err", "dir_thigh_l", "dir_shin_l", "dir_upper_arm_l", "dir_forearm_l", "knee_plane_l",
              "torso_fwd", "severe"):
        pr = [(r["sid"], r["m"][b][k] - r["m"][a][k]) for r in both if k in r["m"][a] and k in r["m"][b]]
        if len(pr) > 1:
            c = brief(np.array([x[0] for x in pr]), np.array([x[1] for x in pr]))
            res[k] = {"n": len(pr), "diff": c.get("mean"), "ci95": c.get("ci95")}
    # mean over both sides for limb directions
    for k in ("dir_thigh", "dir_shin", "dir_upper_arm", "dir_forearm", "knee_plane"):
        pr = [(r["sid"], np.mean([r["m"][b][f"{k}_{s}"] for s in "lr" if f"{k}_{s}" in r["m"][b] and f"{k}_{s}" in r["m"][a]]) -
               np.mean([r["m"][a][f"{k}_{s}"] for s in "lr" if f"{k}_{s}" in r["m"][b] and f"{k}_{s}" in r["m"][a]]))
              for r in both if any(f"{k}_{s}" in r["m"][b] and f"{k}_{s}" in r["m"][a] for s in "lr")]
        if len(pr) > 1:
            c = brief(np.array([x[0] for x in pr]), np.array([x[1] for x in pr]))
            res[f"{k}_mean"] = {"n": len(pr), "diff": c.get("mean"), "ci95": c.get("ci95")}
    return res


def temporal(R, c):
    by = defaultdict(list)
    for r in R:
        if c in r["J"]:
            by[r["sid"]].append(r)
    knee = torso = limb = steps = 0
    from eval_external_hmr2_baseline import KNEES, LIMBS, ang
    from motion.torso_frame import torso_frames
    for rs in by.values():
        rs.sort(key=lambda r: r["frame"])
        for a, b in zip(rs[:-1], rs[1:]):
            steps += 1
            Ja, Jb, Ga, Gb = a["J"][c], b["J"][c], a["G"], b["G"]
            for h, k, an in KNEES.values():
                na, nb = np.cross(Ja[k] - Ja[h], Ja[an] - Ja[k]), np.cross(Jb[k] - Jb[h], Jb[an] - Jb[k])
                ga, gb = np.cross(Ga[k] - Ga[h], Ga[an] - Ga[k]), np.cross(Gb[k] - Gb[h], Gb[an] - Gb[k])
                if min(np.linalg.norm(x) for x in (na, nb, ga, gb)) > 1e-3 and ang(ga, gb) < 45:
                    knee += int(ang(na, nb) > 90)
            ta, tb, tga, tgb = torso_frames(Ja)[0], torso_frames(Jb)[0], torso_frames(Ga)[0], torso_frames(Gb)[0]
            if None not in (ta, tb, tga, tgb) and ang(tga[:, 1], tgb[:, 1]) < 45:
                torso += int(ang(ta[:, 1], tb[:, 1]) > 90)
            for x, y in LIMBS.values():
                if ang(Ga[y] - Ga[x], Gb[y] - Gb[x]) < 15:
                    limb += int(ang(Ja[y] - Ja[x], Jb[y] - Jb[x]) > 45)
    return {"steps": steps, "knee_plane_flips": knee, "torso_forward_flips": torso, "limb_direction_jumps": limb}


def split_report(R, conds, w97e, brief):
    n = len(R)
    ident = {c: sum(r["identity"] == c for r in R) for c in ("same_person", "other_person", "unmatchable", "no_subject")}
    same = [r for r in R if r["identity"] == "same_person"]
    moa = [r for r in R if r["moa"]]
    out = {"coverage": {"rows": n, **{k: v for k, v in ident.items()}, "same_person_share": ident["same_person"] / n if n else None,
                        "moa_available": len(moa), "moa_share": len(moa) / n if n else None},
           "all_rows": {c: agg(R, c) for c in conds + ["E"]},
           "same_person": {c: agg(same, c) for c in conds + ["E"]},
           "same_person_paired": {f"{b}_minus_B": paired(same, "B", b, brief) for b in ("M", "Mconf", "Mmask", "Moa", "E")},
           "moa_paired": {"Moa_minus_B": paired(moa, "B", "Moa", brief)},
           "all_rows_paired": {"M_minus_B": paired(R, "B", "M", brief)},
           "by_identity": {k: agg([r for r in R if r["identity"] == k], "M") for k in ident},
           "two_d": {}, "strata": {}, "challenge": {}, "temporal": {c: temporal(same, c) for c in ("B", "M", "E")},
           "temporal_all_rows": {c: temporal(R, c) for c in ("B", "M")}}
    for c in ("B", "M", "Moa"):
        for nm, sub in (("all", R), ("same_person", same)):
            e = [r["twod"][c]["err"] for r in sub if r["twod"][c]["err"] is not None]
            p5 = [r["twod"][c]["pck05"] for r in sub if r["twod"][c]["pck05"] is not None]
            p10 = [r["twod"][c]["pck10"] for r in sub if r["twod"][c]["pck10"] is not None]
            nv = [r["twod"][c]["n"] for r in sub]
            out["two_d"][f"{c}_{nm}"] = {"rows_with_joints": len(e), "err": float(np.mean(e)) if e else None, "pck05": float(np.mean(p5)) if p5 else None,
                                         "pck10": float(np.mean(p10)) if p10 else None, "valid_limb_joints_mean": float(np.mean(nv)) if nv else None}
    for s in ("upright", "waist_bend", "lying", "other"):
        sub = [r for r in same if r["stratum"] == s]
        if sub:
            out["strata"][s] = {"rows": len(sub), **{c: agg(sub, c) for c in ("B", "M", "E")}, "paired": paired(sub, "B", "M", brief)}
    masks = {
        "waist_bend": lambda r: r["stratum"] == "waist_bend",
        "seated_crouched_other": lambda r: r["stratum"] == "other",
        "gt_knee_flex_ge_45": lambda r: any(r["m"]["B"].get(f"knee_flex_gt_{s}", 0) >= 45 for s in "lr"),
        "gt_knee_near_straight": lambda r: all(r["m"]["B"].get(f"knee_flex_gt_{s}", 99) < 5 for s in "lr"),
        "out_of_plane_limbs": lambda r: any(abs((r["G"][b] - r["G"][a])[1]) / max(np.linalg.norm(r["G"][b] - r["G"][a]), 1e-9) > 0.5
                                            for a, b in ((1, 2), (4, 5), (11, 12), (14, 15))),
        "s0_wrong_knee_plane": lambda r: any(r["m"]["B"].get(f"knee_plane_{s}", 0) > 45 for s in "lr"),
        "occluded_benchmark": lambda r: r["bench_invalid_limb"] >= 2,
        "low_confidence_M": lambda r: (r["mean_conf_M"] == r["mean_conf_M"]) and r["mean_conf_M"] < 0.5,
    }
    for k, f in masks.items():
        sub = [r for r in same if f(r)]
        if sub:
            out["challenge"][k] = {"rows": len(sub), **{c: agg(sub, c) for c in ("B", "M", "E")}, "paired": paired(sub, "B", "M", brief)}
    # sensitivity: 3D change vs 2D difference between M and B (same-person rows)
    bins = defaultdict(list)
    for r in same:
        d = r["dist_M"]
        key = "<0.02" if d < 0.02 else "0.02-0.04" if d < 0.04 else "0.04-0.07" if d < 0.07 else ">=0.07"
        bins[key].append(float(np.mean(np.linalg.norm(r["J"]["M"] - r["J"]["B"], axis=1)) * 1000))
    out["sensitivity_3d_change_mm_by_2d_distance"] = {k: {"rows": len(v), "mean": float(np.mean(v)), "p95": float(np.percentile(v, 95))} for k, v in sorted(bins.items())}
    # sequence clustering of M - B MPJPE
    seq = defaultdict(list)
    for r in same:
        seq[r["sid"]].append(r["m"]["M"]["mpjpe"] - r["m"]["B"]["mpjpe"])
    out["sequence_mpjpe_change"] = dict(sorted(((k, {"rows": len(v), "mean": float(np.mean(v))}) for k, v in seq.items()), key=lambda kv: -kv[1]["mean"]))
    sp = out["same_person_paired"]["M_minus_B"]
    out["headline"] = {"B_mpjpe_all": out["all_rows"]["B"].get("mpjpe"), "M_mpjpe_all": out["all_rows"]["M"].get("mpjpe"),
                       "B_mpjpe_same": out["same_person"]["B"].get("mpjpe"), "M_mpjpe_same": out["same_person"]["M"].get("mpjpe"),
                       "E_mpjpe_same": out["same_person"]["E"].get("mpjpe"), "M_minus_B_same": sp.get("mpjpe")}
    return out


# ------------------------------------------------------------------ conditional rig replay (evaluation only)

RIG_DECLARED = {
    "body_frame": "W98 condition T (GT-free): Object = HMR 2.0b pelvis, spine from the HMR joint thorax (frozen W97 predictions)",
    "legs": "W95 two-bone IK with zero twist on: B = S0 on the benchmark observation, M = S0 on the AnimCV observation, E = HMR "
            "2.0b joints (external reference); bone lengths, rest, calibration unchanged",
    "arm": "frozen J1X left arm (W87 predictions, benchmark-derived features), identical in every slot",
    "cases": "fixed: walkBridge_01 a0 /333, flat_guitar_01 a0 /500, bar_00 a1 /367, cafe_00 a1 /1021; rules: worst_M_vs_B = "
             "36-row window with the largest summed (M - B) S0 MPJPE over same-person rows; identity_failure = window with the "
             "most rows whose AnimCV subject is not the dataset actor; windows never overlap used frames",
    "metrics": "world thigh/shin error, rig knee-plane error vs GT (both flexed >= 20), unavailable legs, severe hip jumps "
               "(thigh-local step > 90 deg while GT thigh step < 45)",
}
SLOT_LABEL = {"B": "B: S0 on benchmark 2D", "M": "M: S0 on AnimCV RTMDet/RTMPose 2D", "E": "E: HMR 2.0b (external RGB reference)"}


def rig(args):
    import run_external_orientation_handoff as w98
    import run_pelvis_frame_leg_articulation as w95
    import run_pelvis_torso_ownership as w94
    import run_leg_knee_hinge_twist as w96
    from run_torso_frame_rig_replay import arm_pose, conj, spine_local_xyzw
    from run_visual_depth_motion_attribution import _brief

    from framepose.bank import load_bank
    from motion.torso_frame import matrix_to_quaternion_xyzw, quaternion_to_matrix
    from retarget.framepose_fk import derive_rig_alignment, load_rig_fk_calibration
    from retarget.framepose_target_rest import load_target_rest_pose
    from retarget.framepose_two_bone_ik import load_ik_chain_set, quaternion_angle_degrees, target_chain_geometry

    bank = load_bank(args.bank)
    obs = np.load(args.observations)
    side = np.load(str(args.observations).replace(".npz", "_side.npz"))
    rest = load_target_rest_pose(args.rest)
    alignment = derive_rig_alignment(rest, load_rig_fk_calibration(args.calibration))
    Am = quaternion_to_matrix(alignment)
    target = target_chain_geometry(rest, load_ik_chain_set(args.chains).chains[0])
    sb = rest.bones["spine_01"]
    M_ = np.asarray(sb.matrix_local, float)[:3, :3]
    ctx = {"alignment": alignment, "Am": Am, "rest": rest, "target": target, "spine_basis": M_ / np.linalg.norm(M_, axis=0, keepdims=True),
           "spine_head": list(sb.head), "legs": w95.leg_targets(rest), "lf": w96.leg_frames(rest)}
    s0 = {}
    for c in ("B", "M"):
        for split in ("validation", "test"):
            arr = np.load(args.s0_dir / f"prediction_{c}_{split}.npy")
            for k, p in enumerate(bank.indices(split)):
                s0[(c, int(p))] = arr[k]
    hp = np.load(args.hmr_predictions)
    H = {int(bp): i for i, (pp, bp) in enumerate(zip(hp["population"], hp["index"][0])) if str(pp).startswith("3dpw") and hp["valid"][i]}
    j1 = np.load(args.j1x)
    seq_rows = defaultdict(list)
    for i, p in enumerate(obs["positions"]):
        p = int(p)
        s = bank.samples[p]
        G = bank.arrays["target_3d"][p].astype(float)
        if p not in H or not bank.arrays["target_valid"][p].all():
            continue
        G = G - G[0]
        h = H[p]
        R_p, R_t, j17 = w98.handoff_frames(hp["global_orient"][h], hp["joints24"][h])
        if R_t is None:
            continue
        gl = w95.gt_leg_dirs(G)
        legs = {"B": s0[("B", p)], "M": s0[("M", p)], "E": j17}
        r = {"sid": s.sequence_id, "p": p, "frame": int(s.frame_index), "t": int(s.frame_index) / 30.0, "image": s.image_reference.relative_path,
             "identity": str(side["identity"][i]), "available": True, "gt_arm_ok": True, "gt_legs": gl, "Ro": R_p, "Rt": R_t, "G": G, "ev": {}}
        for c, J in legs.items():
            ev = w95.evaluate(np.asarray(J, float) - np.asarray(J, float)[0], R_p, R_t, ctx, gl)
            r["ev"][c] = ev
        arm = arm_pose(j1["left_arm_g1_elbow"][p], j1["left_arm_g1_wrist"][p], R_t, alignment, target, rest)
        r["arm"] = arm
        seq_rows[s.sequence_id].append(r)
    for rs in seq_rows.values():
        rs.sort(key=lambda r: r["frame"])

    def leg_metrics(r, c):
        ev = r["ev"][c]
        if ev is None:
            return None
        out = {"thigh": np.mean([ev["legs"][x]["thigh_err"] for x in "lr"]), "shin": np.mean([ev["legs"][x]["shin_err"] for x in "lr"])}
        kp = []
        for x, (hh, kk, aa) in (("l", (1, 2, 3)), ("r", (4, 5, 6))):
            pose, G = ev["legs"][x]["pose"], r["G"]
            gf = math.degrees(math.acos(np.clip(np.dot(*(w96._unit(v) for v in (G[kk] - G[hh], G[aa] - G[kk]))), -1, 1)))
            if gf >= 20 and ev["legs"][x]["knee"] >= 20:
                n = r["Ro"] @ (Am.T @ np.cross(pose.upper, pose.forearm))
                kp.append(math.degrees(math.acos(np.clip(np.dot(w96._unit(n), w96._unit(np.cross(G[kk] - G[hh], G[aa] - G[kk]))), -1, 1))))
        out["knee_plane_rig"] = float(np.mean(kp)) if kp else None
        return out
    rows = [r for rs in seq_rows.values() for r in rs]
    split_of = {int(p): "validation" for p in bank.indices("validation")}
    split_of.update({int(p): "test" for p in bank.indices("test")})
    res = {"declared": RIG_DECLARED}
    for split in ("validation", "test"):
        R = [r for r in rows if split_of[r["p"]] == split]
        same = [r for r in R if r["identity"] == "same_person"]
        d = {"rows": len(R), "same_person_rows": len(same), "unavailable_legs": {c: sum(r["ev"][c] is None for r in R) for c in ("B", "M", "E")}}
        for c in ("B", "M", "E"):
            ms = [leg_metrics(r, c) for r in same]
            ms = [m for m in ms if m is not None]
            d[c] = {k: float(np.mean([m[k] for m in ms if m[k] is not None])) for k in ("thigh", "shin", "knee_plane_rig")}
        for k in ("thigh", "shin", "knee_plane_rig"):
            pr = []
            for r in same:
                a, b = leg_metrics(r, "B"), leg_metrics(r, "M")
                if a and b and a[k] is not None and b[k] is not None:
                    pr.append((r["sid"], b[k] - a[k]))
            if len(pr) > 1:
                cc = _brief(np.array([x[0] for x in pr]), np.array([x[1] for x in pr]))
                d[f"M_minus_B_{k}"] = {"n": len(pr), "diff": cc.get("mean"), "ci95": cc.get("ci95")}
        jumps = {c: 0 for c in ("B", "M", "E")}
        for rs in seq_rows.values():
            for a, b in zip(rs[:-1], rs[1:]):
                if split_of[a["p"]] != split or a["identity"] != "same_person" or b["identity"] != "same_person":
                    continue
                for c in jumps:
                    ea, eb = a["ev"][c], b["ev"][c]
                    if ea is None or eb is None:
                        continue
                    for x in "lr":
                        if (quaternion_angle_degrees(ea["legs"][x]["pose"].proximal_local, eb["legs"][x]["pose"].proximal_local) > 90
                                and w95.angle(a["gt_legs"][x][0], b["gt_legs"][x][0]) < 45):
                            jumps[c] += 1
        d["severe_hip_jumps_same_person"] = jumps
        res[split] = d
    (args.out_dir / "rig_report.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps({s: res[s] for s in ("validation", "test")}, indent=1, default=float))
    # ---- Blender cases
    used = set()
    cases = {}
    for key, sid, f0 in (("3dpw_walking_upright", "3dpw:downtown_walkBridge_01:actor0", 333), ("3dpw_seated_or_crouched", "3dpw:flat_guitar_01:actor0", 500),
                         ("3dpw_s0_wrong_knee_plane", "3dpw:downtown_bar_00:actor1", 367), ("3dpw_turn_180", "3dpw:downtown_cafe_00:actor1", 1021)):
        if sid in seq_rows and any(r["frame"] == f0 for r in seq_rows[sid]):
            w = w94.window_at(seq_rows, sid, f0)
            cases[key] = w
            used |= {(r["sid"], r["frame"]) for r in w}

    def rule(score):
        def f(w):
            if any((r["sid"], r["frame"]) in used for r in w):
                return -1e9
            return score(w)
        return f
    mp = {}
    for r in rows:
        if r["identity"] == "same_person":
            jb, jm = s0[("B", r["p"])], s0[("M", r["p"])]
            mp[(r["sid"], r["frame"])] = 1000 * float(np.mean(np.linalg.norm((jm - jm[0]) - r["G"], axis=1)) - np.mean(np.linalg.norm((jb - jb[0]) - r["G"], axis=1)))
    for key, score in (("3dpw_worst_M_vs_B", lambda w: sum(mp.get((r["sid"], r["frame"]), 0.0) for r in w)),
                       ("3dpw_identity_failure", lambda w: sum(r["identity"] != "same_person" for r in w))):
        w = w94.best_window(seq_rows, rule(score))
        cases[key] = w
        used |= {(r["sid"], r["frame"]) for r in w}
    out = {}
    for key, w in cases.items():
        out[key] = export_rig_case(key[5:], w, ctx, conj, spine_local_xyzw, matrix_to_quaternion_xyzw, w94, s0)
    common = {"rest_snapshot_digest": rest.digest(), "canonical_to_armature_xyzw": list(alignment), "target_chain": target.to_dict()}
    (args.out_dir / "cases_3dpw.json").write_text(json.dumps({**common, "cases": out}, indent=1))
    print(json.dumps({k: {"sequence_id": v["sequence_id"], "start_frame": v["start_frame"]} for k, v in out.items()}))


def export_rig_case(name, w, ctx, conj, spine_local_xyzw, q_xyzw, w94, s0):
    alignment, B, head, target = ctx["alignment"], ctx["spine_basis"], ctx["spine_head"], ctx["target"]
    frames = []
    slots = {"FIXED": "B", "RUNTIME": "M", "GT": "E"}
    s_err = {SLOT_LABEL[c]: [] for c in slots.values()}
    s_knee = {SLOT_LABEL[c]: [] for c in slots.values()}
    for r in w:
        rec = {"bank_position": r["p"], "frame_index": r["frame"], "time_s": r["t"], "image": r["image"], "available": True, "gt_valid": True,
               "runtime_status": "KNOWN", "runtime_reasons": [], "yaw_deg": {}, "modes": {}, "forward_cam": {}, "mode_lines": {},
               "object_rotation_xyzw": {}, "extra_bone_locals_xyzw": {}, "fk_points_armature": {},
               "reference_legs_cam": {"left": [list(r["gt_legs"]["l"][0]), list(r["gt_legs"]["l"][1])],
                                      "right": [list(r["gt_legs"]["r"][0]), list(r["gt_legs"]["r"][1])]}}
        if r["identity"] != "same_person":
            rec["runtime_status"], rec["runtime_reasons"] = "UNRELIABLE", [f"AnimCV subject: {r['identity']} (evaluation-only identity check)"]
        for slot, c in slots.items():
            lab, ev = SLOT_LABEL[c], r["ev"][c]
            if ev is None:
                rec["yaw_deg"][slot] = None
                rec["mode_lines"][slot] = ["legs unavailable for this row (no fallback)"]
                s_err[lab].append(None)
                s_knee[lab].append(None)
                continue
            Ro, Rt = r["Ro"], r["Rt"]
            y = w94.compat_yaw(Rt)
            rec["yaw_deg"][slot] = math.degrees(y) if y is not None else None
            rec["forward_cam"][slot] = list(Rt[:, 1])
            rec["object_rotation_xyzw"][slot] = list(q_xyzw(conj(alignment, Ro)))
            bones = {"spine_01": list(spine_local_xyzw(alignment, Ro.T @ Rt, B))}
            for x in "lr":
                p = ev["legs"][x]["pose"]
                bones[f"thigh_{x}"], bones[f"calf_{x}"] = list(p.proximal_local), list(p.distal_local)
                rec["fk_points_armature"].setdefault(slot, {}).update({f"thigh_{x}": list(p.mid), f"calf_{x}": list(p.end)})
            rec["extra_bone_locals_xyzw"][slot] = bones
            arm = r["arm"]
            if arm is not None and arm.status == "known":
                Q = conj(alignment, Ro.T @ Rt)
                sp = lambda pt: list(np.asarray(head) + Q @ (np.asarray(pt) - np.asarray(head)))  # noqa: E731
                rec["modes"][slot] = {"proximal_local_xyzw": list(arm.proximal_local), "distal_local_xyzw": list(arm.distal_local),
                                      "mid": sp(arm.mid), "end": sp(arm.end), "root": sp(target.root_head),
                                      "object_rotation_armature_xyzw": rec["object_rotation_xyzw"][slot], "extra_bone_locals_xyzw": {"spine_01": bones["spine_01"]}}
            L, Rr = ev["legs"]["l"], ev["legs"]["r"]
            lines = [f"thigh err L/R {L['thigh_err']:4.0f}/{Rr['thigh_err']:4.0f} | shin err L/R {L['shin_err']:4.0f}/{Rr['shin_err']:4.0f}",
                     f"knee flex L/R {L['knee']:4.0f}/{Rr['knee']:4.0f}"]
            if c in ("B", "M"):
                J = s0[(c, r["p"])]
                lines.append(f"S0 MPJPE {1000 * float(np.mean(np.linalg.norm((J - J[0]) - r['G'], axis=1))):4.0f} mm (eval only)")
            if c == "M" and r["identity"] != "same_person":
                lines.append(f"subject: {r['identity']}")
            rec["mode_lines"][slot] = lines
            s_err[lab].append(float(np.mean([L["shin_err"], Rr["shin_err"]])))
            s_knee[lab].append(float(np.mean([L["knee"], Rr["knee"]])))
        frames.append(rec)
    colors = {SLOT_LABEL[c]: col for c, col in zip(slots.values(), ("#eb6834", "#2a78d6", "#1baf7a"))}
    return {"case": name, "split": "3dpw", "sequence_id": w[0]["sid"], "start_frame": w[0]["frame"], "fps_sampled": 1.0 / max(w[1]["t"] - w[0]["t"], 1e-6),
            "frames": frames, "labels": {s: SLOT_LABEL[c] for s, c in slots.items()}, "short_labels": dict(slots),
            "title": "W100 benchmark vs AnimCV 2D observation -> frozen S0 -> rig (3DPW)",
            "header_note": "W98 T body frame (HMR pelvis + thorax), W95 zero-twist legs, frozen J1X arm | thin green = GT legs (evaluation only)",
            "plot_series": {"panels": [{"ylabel": "mean shin direction error (deg)", "series": s_err, "colors": colors},
                                       {"ylabel": "mean knee flexion (deg)", "series": s_knee, "colors": colors}]}}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build-observations")
    b.add_argument("--bank", type=Path, required=True)
    b.add_argument("--observe-dir", type=Path, required=True)
    b.add_argument("--out", type=Path, required=True)
    e = sub.add_parser("evaluate")
    for n in ("bank", "observations", "s0_dir", "hmr_predictions", "raw_3dpw", "out_dir"):
        e.add_argument("--" + n.replace("_", "-"), type=Path, required=True)
    e.add_argument("--cases", action="store_true")
    g = sub.add_parser("rig")
    for n in ("bank", "observations", "s0_dir", "hmr_predictions", "j1x", "rest", "calibration", "chains", "out_dir"):
        g.add_argument("--" + n.replace("_", "-"), type=Path, required=True)
    u = sub.add_parser("unseen-summary")
    u.add_argument("--unseen-dir", type=Path, required=True)
    args = ap.parse_args()
    if args.cmd == "build-observations":
        build_observations(args)
    elif args.cmd == "unseen-summary":
        unseen_summary(args)
    else:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        (evaluate if args.cmd == "evaluate" else rig)(args)


def unseen_summary(args):
    """GT-free temporal continuity of S0 on the unseen videos: limb-direction steps, knee-plane flips, torso forward flips."""
    from eval_external_hmr2_baseline import KNEES, LIMBS, ang
    from motion.torso_frame import torso_frames

    rep = json.loads((args.unseen_dir / "unseen_report.json").read_text())
    out = {}
    for name in rep:
        z = np.load(args.unseen_dir / f"{name}.npz")
        J, v = z["s0"].astype(float), z["input_valid"]
        ok = v.sum(1) >= 12
        steps = [(a, b) for a, b in zip(range(len(J) - 1), range(1, len(J))) if ok[a] and ok[b]]
        limb = [max(ang(J[b][y] - J[b][x], J[a][y] - J[a][x]) for x, y in LIMBS.values()) for a, b in steps]
        knee = sum(int(ang(np.cross(J[a][k] - J[a][h], J[a][an] - J[a][k]), np.cross(J[b][k] - J[b][h], J[b][an] - J[b][k])) > 90)
                   for a, b in steps for h, k, an in KNEES.values()
                   if ang(J[a][k] - J[a][h], J[a][an] - J[a][k]) >= 20 and ang(J[b][k] - J[b][h], J[b][an] - J[b][k]) >= 20)
        tf = [(torso_frames(J[a])[0], torso_frames(J[b])[0]) for a, b in steps]
        torso = sum(int(ang(x[:, 1], y[:, 1]) > 90) for x, y in tf if x is not None and y is not None)
        out[name] = {"frames": int(len(J)), "frames_with_>=12_valid_joints": int(ok.sum()), "valid_steps": len(steps),
                     "max_limb_step_deg_p50": float(np.median(limb)) if limb else None, "max_limb_step_deg_p95": float(np.percentile(limb, 95)) if limb else None,
                     "knee_plane_flips": knee, "torso_forward_flips": torso, "fps": rep[name]["fps"],
                     "tracking": rep[name]["tracking_report"]}
    (args.unseen_dir / "unseen_summary.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
