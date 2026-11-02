"""W103 analyze stage: runs only under the hashed 2D-only declaration produced by `w103_observability.py declare`.

GT is used for evaluation geometry, observation-quality measurement and neighbor-reference comparison only. Nothing here
creates an S0 input, a mask, a crop, a correction or a selectable estimator.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np

from w103_observability import (BASE, BRANCH_DEG, CHAINS, CONDITIONS, DEPTH_RATIO, DIRECT, GT_STEP_DEG, JOINTS,
                                KNEE_BEND_DEG, KNEES, LIMB_DIR_DEG, PARENT, PCK_WELL, SEG_CHAIN, SEGMENTS, SEVERE_MM,
                                SMPL_OF, angle_deg, base_xy, distribution, dump, euclid_rms, full_distance, full_eligible,
                                full_groups, geometry_all, load_context, load_hmr, load_predictions, local_distance,
                                local_vectors, pair_metrics, pools, projection_features, raw_references, ray_rotation,
                                retrieve, row_geometry, sha, subset, template_pose, torso_forward, unit, video_bootstrap)

SEG_NAMES = list(SEGMENTS)
SPLITS = ("validation", "test")


# ------------------------------------------------------------------ per-row S0 metrics (W97/W100 definitions, vectorized)
def s0_metrics(J, G):
    """J, G: (N,17,3) root-relative camera-frame. NaN marks an ineligible decision."""
    J = J - J[:, :1]; G = G - G[:, :1]
    out = {"mpjpe": 1000*np.linalg.norm(J-G, axis=-1).mean(-1), "per_joint": 1000*np.linalg.norm(J-G, axis=-1)}
    sj = np.stack([J[:, b]-J[:, a] for a, b in SEGMENTS.values()], 1)
    sg = np.stack([G[:, b]-G[:, a] for a, b in SEGMENTS.values()], 1)
    out["dir"] = angle_deg(sj, sg)
    elig = np.abs(sg[..., 1]) >= DEPTH_RATIO*np.linalg.norm(sg, axis=-1)
    out["depth_err"] = np.where(elig, (np.sign(sg[..., 1]) != np.sign(sj[..., 1])).astype(float), np.nan)
    kw, kp = [], []
    for h, k, a in KNEES.values():
        fj, fg = angle_deg(J[:, k]-J[:, h], J[:, a]-J[:, k]), angle_deg(G[:, k]-G[:, h], G[:, a]-G[:, k])
        e = angle_deg(np.cross(J[:, k]-J[:, h], J[:, a]-J[:, k]), np.cross(G[:, k]-G[:, h], G[:, a]-G[:, k]))
        ok = (fj >= KNEE_BEND_DEG) & (fg >= KNEE_BEND_DEG)
        kp.append(np.where(ok, e, np.nan)); kw.append(np.where(ok, (e > BRANCH_DEG).astype(float), np.nan))
    out["knee_plane"], out["knee_wrong"] = np.stack(kp, 1), np.stack(kw, 1)
    fa = angle_deg(torso_forward(J), torso_forward(G))
    out["torso_flip"] = np.where(np.isfinite(fa), (fa > BRANCH_DEG).astype(float), np.nan)
    out["knee_normal"] = np.stack([np.cross(J[:, k]-J[:, h], J[:, a]-J[:, k]) for h, k, a in KNEES.values()], 1)
    out["accurate"] = ((out["mpjpe"] <= 100) & ~(out["dir"] > LIMB_DIR_DEG).any(1)
                       & ~(np.nan_to_num(out["knee_wrong"]) > 0).any(1) & ~(np.nan_to_num(out["torso_flip"]) > 0))
    out["severe"] = out["mpjpe"] > SEVERE_MM
    return out


def observation_quality(x, v, uv, sizes):
    """W101 GT-referenced 2D error on the 12 limb joints, per joint; WO / PARTIAL / MISLOCALIZED per row."""
    gt12 = uv[:, [SMPL_OF[j] for j in DIRECT]]
    diag = np.linalg.norm(gt12.max(1)-gt12.min(1), axis=-1)
    px = x[:, DIRECT, :2]*sizes[:, None, :]
    err = np.linalg.norm(px-gt12, axis=-1)/diag[:, None]
    valid = v[:, DIRECT]
    err = np.where(valid, err, np.nan)
    n = valid.sum(1)
    status = np.where(n < 12, "partial", np.where(np.nanmax(np.where(valid, err, -1), 1) <= PCK_WELL, "well", "mislocalized"))
    return {"err": err, "n_limb_valid": n, "status": status}


def posture(G, gravity):
    """W94/W102 stratum from GT trunk inclination and mean thigh angle to gravity (evaluation-only)."""
    import run_pelvis_torso_ownership as w94
    from motion.torso_frame import torso_frames
    out = []
    for g, grav in zip(G, gravity):
        torso = torso_frames(g)[0]
        trunk = w94.angle(torso[:, 2], grav) if torso is not None else None
        thigh = .5*(w94.angle(g[2]-g[1], -grav)+w94.angle(g[5]-g[4], -grav))
        out.append(w94.stratum(trunk, thigh) or "undefined")
    return np.array(out)


# ------------------------------------------------------------------ retrieval wrappers
def full_retrieval(P, src, k):
    """Strict full-body tier: identical masks, cross-video, nearest row per other sequence, k sequences."""
    F = P["F"][src]; n = len(F)
    idx = np.full((n, k), -1, np.int64); dist = np.full((n, k), np.inf)
    for key, rows in full_groups(F, P["members"][src]).items():
        rows = rows[np.lexsort((P["sid"][rows], P["seq"][rows]))]
        x, v = base_xy(F[rows]); sel = v[0]
        X = x[:, sel].reshape(len(rows), -1)
        ii, dd = retrieve(X, X, P["seq"][rows], P["vid"][rows], P["vid"][rows], k=k, dist=euclid_rms(int(sel.sum())))
        idx[rows] = np.where(ii >= 0, rows[np.maximum(ii, 0)], -1); dist[rows] = dd
    return idx, dist


def vector_retrieval(X, rows, P, k, n_points):
    rows = rows[np.lexsort((P["sid"][rows], P["seq"][rows]))]
    ii, dd = retrieve(X[rows], X[rows], P["seq"][rows], P["vid"][rows], P["vid"][rows], k=k, dist=euclid_rms(n_points))
    n = len(P["sid"])
    idx = np.full((n, k), -1, np.int64); dist = np.full((n, k), np.inf)
    idx[rows] = np.where(ii >= 0, rows[np.maximum(ii, 0)], -1); dist[rows] = dd
    return idx, dist


def enumerate_pairs(X, rows, vid, tau, n_points, chunk=1024):
    """All unordered pairs (a<b by row index) within tau; returns a, b, d, same_video."""
    rows = np.sort(rows); A, B, D = [], [], []
    for s0 in range(0, len(rows), chunk):
        d = euclid_rms(n_points)(X[rows[s0:s0+chunk]], X[rows])
        hit = (d <= tau) & (rows[s0:s0+chunk, None] < rows[None])
        a, b = np.nonzero(hit)
        A.append(rows[s0+a]); B.append(rows[b]); D.append(d[a, b])
    a, b, d = np.concatenate(A), np.concatenate(B), np.concatenate(D)
    return a, b, d, vid[a] == vid[b]


def full_pairs(P, src, tau):
    F = P["F"][src]; out = [[], [], [], []]
    for key, rows in full_groups(F, P["members"][src]).items():
        x, v = base_xy(F[rows]); sel = v[0]
        X = np.zeros((len(F), 2*int(sel.sum()))); X[rows] = x[:, sel].reshape(len(rows), -1)
        for o, r in zip(out, enumerate_pairs(X, rows, P["vid"], tau, int(sel.sum()))):
            o.append(r)
    return [np.concatenate(o) for o in out]


# ------------------------------------------------------------------ pair-level observability
def summarize_pairs(a, b, d, same_video, ctx, src, tau_name, extra=None):
    geo, tmpl, P = ctx["geo"], ctx["template"], ctx["P"]
    cross = ~same_video
    res = {"pairs": int(len(a)), "cross_video_pairs": int(cross.sum()), "same_video_pairs_excluded": int(same_video.sum())}
    a, b, d = a[cross], b[cross], d[cross]
    if not len(a):
        return res
    m = pair_metrics(subset(geo, a), subset(geo, b), tmpl[a], tmpl[b])
    vids = np.array([tuple(sorted(x)) for x in zip(P["vid"][a], P["vid"][b])], dtype=object)
    res["video_pairs"] = len(set(map(tuple, vids)))
    res["videos_involved"] = len(set(P["vid"][a]) | set(P["vid"][b]))
    res["distance"] = distribution(d)
    res["template_mpjpe_mm"] = distribution(m["template_mpjpe"])
    raw = 1000*np.linalg.norm(ctx["Gr"][a]-ctx["Gr"][b], axis=-1).mean(-1)
    res["ray_aligned_raw_mpjpe_mm"] = distribution(raw)
    res["material"] = rate_block(m["material"], vids)
    res["limb_direction_gt45"] = rate_block(m["limb_dir"], vids)
    res["knee_branch"] = {s: {"eligible": int(m["knee_eligible"][:, i].sum()), "disagree": int(m["knee_branch"][:, i].sum()),
                              "degenerate_lt20": int((~m["knee_eligible"][:, i]).sum())} for i, s in enumerate(KNEES)}
    res["depth_flip"] = {s: {"eligible": int(m["depth_eligible"][:, i].sum()), "flip": int(m["depth_flip"][:, i].sum())}
                         for i, s in enumerate(SEG_NAMES)}
    res["torso_flip"] = {"eligible": int(m["torso_eligible"].sum()), "flip": int(m["torso_flip"].sum()),
                         "angle": distribution(m["torso_angle"])}
    res["ray_angle_between_members_deg"] = distribution(angle_deg(ctx["root"][a], ctx["root"][b]))
    res["bone_scale_ratio"] = distribution(ctx["scale"][a]/ctx["scale"][b])
    proj = np.sqrt(((ctx["proj"][a]-ctx["proj"][b])**2).sum(-1).mean(-1))
    res["projection_distance"] = distribution(proj)
    pc = proj <= ctx["tau"]["full_N"]
    res["projection_consistent_share"] = float(pc.mean())
    q = ctx["quality"][src]["status"]
    wo = (q[a] == "well") & (q[b] == "well")
    ctrl = wo & pc
    res["controlled"] = {"rule": "both rows well observed (source input) and D_proj <= tau_full_N",
                         "pairs": int(ctrl.sum()), "video_pairs": len(set(map(tuple, vids[ctrl]))),
                         "material_video_pairs": len(set(map(tuple, vids[ctrl & m["material"]]))),
                         "material": rate_block(m["material"][ctrl], vids[ctrl]),
                         "template_mpjpe_mm": distribution(m["template_mpjpe"][ctrl])}
    res["material_by_projection"] = {"consistent": rate_block(m["material"][pc], vids[pc]),
                                     "inconsistent": rate_block(m["material"][~pc], vids[~pc])}
    pred = ctx["pred"][src]
    res["s0_prediction_difference_mm"] = distribution(1000*np.linalg.norm(pred[a]-pred[b], axis=-1).mean(-1))
    F = P["F"][src]
    cd = np.abs(F[a, :, 2]-F[b, :, 2])[:, BASE].mean(1)
    res["confidence_abs_difference"] = distribution(cd)
    # Can a frozen model resolve the pair? S0 (same input class) vs HMR2 (RGB, contextual), per decision kind.
    res["pair_resolution"] = pair_resolution(a, b, m, ctx, src)
    return res


def rate_block(flags, vids):
    flags = np.asarray(flags, bool)
    out = {"n": int(len(flags)), "count": int(flags.sum()), "rate": float(flags.mean()) if len(flags) else None}
    if len(flags):
        firsts = [v[0] for v in vids]
        out["video_bootstrap_by_first_member_video"] = video_bootstrap(firsts, flags.astype(float))
    return out


def pair_resolution(a, b, m, ctx, src):
    """For materially different pairs: are both members correct on the differing decision, for S0 and HMR2?"""
    out = {}
    for name, sel, fail_key in (("knee_branch", m["knee_branch"], "knee_wrong"), ("depth_flip", m["depth_flip"], "depth_err")):
        i, j = np.nonzero(sel)
        rec = {"instances": int(len(i))}
        for model in (src, "E"):
            fa, fb = ctx["metrics"][model][fail_key][a[i], j], ctx["metrics"][model][fail_key][b[i], j]
            ok = np.isfinite(fa) & np.isfinite(fb)
            rec[model] = {"evaluable": int(ok.sum()), "both_correct": int(((fa == 0) & (fb == 0) & ok).sum()),
                          "one_wrong": int((((fa == 1) ^ (fb == 1)) & ok).sum()), "both_wrong": int(((fa == 1) & (fb == 1) & ok).sum())}
        out[name] = rec
    i = np.flatnonzero(m["torso_flip"])
    rec = {"instances": int(len(i))}
    for model in (src, "E"):
        fa, fb = ctx["metrics"][model]["torso_flip"][a[i]], ctx["metrics"][model]["torso_flip"][b[i]]
        ok = np.isfinite(fa) & np.isfinite(fb)
        rec[model] = {"evaluable": int(ok.sum()), "both_correct": int(((fa == 0) & (fb == 0) & ok).sum()),
                      "one_wrong": int((((fa == 1) ^ (fb == 1)) & ok).sum()), "both_wrong": int(((fa == 1) & (fb == 1) & ok).sum())}
    out["torso_flip"] = rec
    return out


def query_level(idx, dist, members, tau, ctx, split_of):
    """Per query: neighbors within tau (<=k distinct sequences). Rates over queries, video-clustered."""
    geo, tmpl, P = ctx["geo"], ctx["template"], ctx["P"]
    res = {}
    for split in SPLITS:
        q = np.array([r for r in members if split_of[r] == split and idx[r, 0] >= 0])
        has = dist[q, 0] <= tau
        res[split] = {"queries_with_any_reference": int(len(q)), "with_neighbor_within_tau": video_bootstrap(P["vid"][q], has.astype(float))}
        qq = q[has]
        if not len(qq):
            continue
        nn = idx[qq, 0]
        m1 = pair_metrics(subset(geo, qq), subset(geo, nn), tmpl[qq], tmpl[nn])
        anym = np.zeros(len(qq), bool); nwithin = np.zeros(len(qq), int)
        for c in range(idx.shape[1]):
            ok = (dist[qq, c] <= tau) & (idx[qq, c] >= 0)
            nwithin += ok
            if ok.any():
                mc = pair_metrics(subset(geo, qq[ok]), subset(geo, idx[qq[ok], c]))
                anym[np.flatnonzero(ok)] |= mc["material"]
        res[split].update({"nn1_material": video_bootstrap(P["vid"][qq], m1["material"].astype(float)),
                           "any_within_tau_material": video_bootstrap(P["vid"][qq], anym.astype(float)),
                           "nn1_template_mpjpe_mm": distribution(m1["template_mpjpe"]),
                           "sequences_within_tau": distribution(nwithin)})
    return res


def binned_curve(idx, dist, members, tau_n, bins, ctx):
    geo, tmpl, P = ctx["geo"], ctx["template"], ctx["P"]
    q = np.repeat(members, idx.shape[1]); n = idx[members].ravel(); d = dist[members].ravel()
    ok = n >= 0
    q, n, d = q[ok], n[ok], d[ok]
    m = pair_metrics(subset(geo, q), subset(geo, n), tmpl[q], tmpl[n])
    edges = list(bins)+[np.inf]
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        s = (d >= lo*tau_n) & (d < hi*tau_n)
        out.append({"bin_x_tauN": [lo, None if np.isinf(hi) else hi], "instances": int(s.sum()),
                    "query_videos": len(set(P["vid"][q[s]])),
                    "material": video_bootstrap(P["vid"][q[s]], m["material"][s].astype(float)),
                    "knee_branch_given_eligible": float(m["knee_branch"][s].sum()/max(1, m["knee_eligible"][s].sum())),
                    "depth_flip_given_eligible": float(m["depth_flip"][s].sum()/max(1, m["depth_eligible"][s].sum())),
                    "torso_flip_given_eligible": float(m["torso_flip"][s].sum()/max(1, m["torso_eligible"][s].sum())),
                    "template_mpjpe_mm": distribution(m["template_mpjpe"][s])})
    return out


# ------------------------------------------------------------------ local tier
def local_tier(P, src, ctx, k):
    F = P["F"][src]; members = P["members"][src]; tau = ctx["tau"]
    out, retr = {}, {}
    for name, chain in CHAINS.items():
        X, ok = local_vectors(F, chain)
        rows = members[ok[members]]
        idx, dist = vector_retrieval(X, rows, P, k, 4)
        retr[name] = (idx, dist)
        a, b, d, same = enumerate_pairs(X, rows, P["vid"], tau["local_N"], 4)
        cross = ~same
        a, b, d = a[cross], b[cross], d[cross]
        m = pair_metrics(subset(ctx["geo"], a), subset(ctx["geo"], b))
        segs = [i for i, s in enumerate(SEG_NAMES) if SEG_CHAIN[s] == name]
        vids = [tuple(sorted(x)) for x in zip(P["vid"][a], P["vid"][b])]
        q = ctx["quality"][src]["status"]
        wo = (q[a] == "well") & (q[b] == "well")
        rec = {"eligible_rows": int(len(rows)), "cross_video_pairs_within_tauN": int(len(a)), "same_video_pairs_excluded": int(same.sum()),
               "strict_pairs_within_tauS": int((d <= tau["local_S"]).sum()), "video_pairs": len(set(vids)),
               "limb_direction_gt45": rate_block((m["seg_angle"][:, segs] > LIMB_DIR_DEG).any(1), vids),
               "depth_flip": {SEG_NAMES[i]: {"eligible": int(m["depth_eligible"][:, i].sum()), "flip": int(m["depth_flip"][:, i].sum()),
                                             "flip_well_observed_both": int((m["depth_flip"][:, i] & wo).sum()),
                                             "eligible_well_observed_both": int((m["depth_eligible"][:, i] & wo).sum())} for i in segs}}
        if name.startswith("leg"):
            s = 0 if name == "leg_l" else 1
            rec["knee_branch"] = {"eligible": int(m["knee_eligible"][:, s].sum()), "disagree": int(m["knee_branch"][:, s].sum()),
                                  "degenerate_lt20": int((~m["knee_eligible"][:, s]).sum()),
                                  "eligible_well_observed_both": int((m["knee_eligible"][:, s] & wo).sum()),
                                  "disagree_well_observed_both": int((m["knee_branch"][:, s] & wo).sum()),
                                  "strict_tauS_eligible": int((m["knee_eligible"][:, s] & (d <= tau["local_S"])).sum()),
                                  "strict_tauS_disagree": int((m["knee_branch"][:, s] & (d <= tau["local_S"])).sum())}
        out[name] = rec
        ctx.setdefault("local_pairs", {})[(src, name)] = (a, b, d, m)
    return out, retr


# ------------------------------------------------------------------ attribution of failures
def failure_instances(src, ctx, rows):
    """(row, kind, index) for every severe / wrong-knee / depth-sign failure of `src` on `rows`."""
    mt = ctx["metrics"][src]; out = []
    for r in rows:
        if mt["severe"][r]:
            out.append((r, "severe", -1))
        for s in range(2):
            if mt["knee_wrong"][r, s] == 1:
                out.append((r, "knee", s))
        for i in range(8):
            if mt["depth_err"][r, i] == 1:
                out.append((r, "depth", i))
    return out


def to_camera(v_ray, R):
    """Row vectors in a row's ray frame -> that row's camera frame (inverse of v @ R.T)."""
    return v_ray @ R


def attribute(inst, src, ctx, full_retr, local_retr):
    r, kind, i = inst
    P, tau, R = ctx["P"], ctx["tau"], ctx["R"]
    q = ctx["quality"][src]
    rec = {"row": int(r), "sample_id": str(P["sid"][r]), "video": str(P["vid"][r]), "split": str(P["split"][r]), "kind": kind,
           "which": None if i < 0 else (("l", "r")[i] if kind == "knee" else SEG_NAMES[i]),
           "quality": str(q["status"][r]), "n_limb_valid": int(q["n_limb_valid"][r]), "posture": str(ctx["posture"][r]),
           "mpjpe": float(ctx["metrics"][src]["mpjpe"][r])}
    chain = None if kind == "severe" else (("leg_l", "leg_r")[i] if kind == "knee" else SEG_CHAIN[SEG_NAMES[i]])
    if chain:
        joints = CHAINS[chain][:3]
        valid = P["F"][src][r, list(joints), 3] > .5
        rec["failing_limb_all_valid"] = bool(valid.all())
        err = q["err"][r, [DIRECT.index(j) for j in joints]]
        rec["failing_limb_max_2d_err"] = None if not np.isfinite(err).any() else float(np.nanmax(err))
    if rec["quality"] != "well":
        rec["category"] = "missingness" if rec["quality"] == "partial" else "mislocalization"
        return rec
    idx, dist = full_retr if kind == "severe" else local_retr[chain]
    nb = [(int(n), float(d)) for n, d in zip(idx[r], dist[r]) if n >= 0 and d <= (tau["full_N"] if kind == "severe" else tau["local_N"])]
    rec["neighbors"] = [{"row": n, "sample_id": str(P["sid"][n]), "d": d} for n, d in nb]
    rec["neighbor_videos"] = len({P["vid"][n] for n, _ in nb})
    pred = ctx["pred"][src][r]; pred = pred - pred[0]
    G = ctx["G"][r]
    amb = False; cons = False
    if kind == "severe":
        Gr_q = ctx["Gr"][r]; Pr = pred @ R[r].T
        d_pq = 1000*np.linalg.norm(pred-G, axis=-1).mean()
        transfer = []
        for n, _ in nb:
            mat = pair_metrics(subset(ctx["geo"], [r]), subset(ctx["geo"], [n]))["material"][0]
            d_pn = 1000*np.linalg.norm(Pr-ctx["Gr"][n], axis=-1).mean()
            t = 1000*np.linalg.norm(ctx["Gr"][n]-Gr_q, axis=-1).mean()
            transfer.append(t)
            amb |= bool(mat and d_pn < d_pq)
        rec["neighbor_transfer_mpjpe"] = transfer
        cons = len(nb) >= 2 and rec["neighbor_videos"] >= 2 and float(np.median(transfer)) <= .5*d_pq
    elif kind == "knee":
        h, k, a = list(KNEES.values())[i]
        nq = np.cross(G[k]-G[h], G[a]-G[k]); ns = np.cross(pred[k]-pred[h], pred[a]-pred[k])
        eligible = agree = 0; videos = set()
        for n, _ in nb:
            if ctx["geo"]["flex"][n, i] < KNEE_BEND_DEG:
                continue
            nn = to_camera(ctx["geo"]["normal"][n, i], R[r])
            eligible += 1; videos.add(P["vid"][n])
            gt_side = angle_deg(nn, nq) < BRANCH_DEG
            agree += gt_side
            amb |= bool((not gt_side) and angle_deg(nn, ns) < BRANCH_DEG)
        rec["eligible_neighbors"], rec["neighbors_on_gt_branch"] = eligible, int(agree)
        cons = eligible >= 2 and len(videos) >= 2 and agree == eligible
    else:
        a_, b_ = SEGMENTS[SEG_NAMES[i]]
        gq = np.sign(G[b_, 1]-G[a_, 1])
        eligible = agree = 0; videos = set()
        for n, _ in nb:
            seg = to_camera(ctx["Gr"][n, b_]-ctx["Gr"][n, a_], R[r])
            if abs(seg[1]) < DEPTH_RATIO*np.linalg.norm(seg):
                continue
            eligible += 1; videos.add(P["vid"][n])
            same = np.sign(seg[1]) == gq
            agree += same
            amb |= bool(not same)
        rec["eligible_neighbors"], rec["neighbors_same_depth_sign"] = eligible, int(agree)
        cons = eligible >= 2 and len(videos) >= 2 and agree == eligible
    rec["category"] = ("mixed" if amb and cons else "ambiguity" if amb else "model_specific" if cons else
                       "unresolved_no_neighbor" if not nb else "unresolved")
    return rec


def attribution_summary(records):
    out = {}
    kinds = {"all_important": lambda r: True, "severe": lambda r: r["kind"] == "severe",
             "knee_branch": lambda r: r["kind"] == "knee", "depth_sign_leg": lambda r: r["kind"] == "depth" and SEG_CHAIN[r["which"]].startswith("leg"),
             "depth_sign_arm": lambda r: r["kind"] == "depth" and SEG_CHAIN[r["which"]].startswith("arm")}
    for name, f in kinds.items():
        rr = [r for r in records if f(r)]
        rec = {"instances": len(rr), "categories": dict(Counter(r["category"] for r in rr)),
               "videos": len({r["video"] for r in rr})}
        att = [r for r in rr if r["category"] in ("ambiguity", "model_specific", "mixed")]
        rec["attributed"] = len(att)
        for cat in ("ambiguity", "model_specific", "mixed"):
            rec[f"share_{cat}"] = video_bootstrap([r["video"] for r in att], [float(r["category"] == cat) for r in att])
        wo = [r for r in rr if r["quality"] == "well"]
        rec["well_observed_instances"] = len(wo)
        rec["by_posture"] = {p: dict(Counter(r["category"] for r in rr if r["posture"] == p)) for p in sorted({r["posture"] for r in rr})}
        out[name] = rec
    return out


# ------------------------------------------------------------------ temporal
def temporal(records, src, ctx, members_set):
    P, bank = ctx["P"], ctx["bank"]
    row_of = ctx["row_of"]; mt = ctx["metrics"][src]; F = P["F"][src]; tau = ctx["tau"]
    out = []
    for rec in records:
        if rec["quality"] != "well":
            continue
        r = rec["row"]; sample = bank.samples[int(ctx["positions"][r])]
        nbrs = []
        for key in ("previous", "next"):
            n = row_of.get(sample.neighbors.get(key))
            nbrs.append(n if n is not None and n in members_set else None)
        info = {"category": rec["category"], "kind": rec["kind"], "neighbors_present": sum(n is not None for n in nbrs)}
        fails, stable2d, gtstable = [], [], []
        for n in nbrs:
            if n is None:
                continue
            if rec["kind"] == "severe":
                fails.append(bool(mt["severe"][n]))
                same_mask = np.array_equal(F[r, :, 3], F[n, :, 3])
                x1, v1 = base_xy(F[[r]]); x2, v2 = base_xy(F[[n]])
                stable2d.append(bool(same_mask and full_distance(x1, v1, x2, v2)[0] <= tau["full_N"]))
                gtstable.append(None)
            else:
                i = ("l", "r").index(rec["which"]) if rec["kind"] == "knee" else SEG_NAMES.index(rec["which"])
                key = "knee_wrong" if rec["kind"] == "knee" else "depth_err"
                fails.append(bool(mt[key][n, i] == 1))
                chain = ("leg_l", "leg_r")[i] if rec["kind"] == "knee" else SEG_CHAIN[rec["which"]]
                X, ok = local_vectors(F[[r, n]], CHAINS[chain])
                stable2d.append(bool(ok.all() and local_distance(X[0], X[1]) <= tau["local_N"]))
                segs = [s for s in SEG_NAMES if SEG_CHAIN[s] == chain]
                gtstable.append(bool(all(angle_deg(ctx["G"][r, SEGMENTS[s][1]]-ctx["G"][r, SEGMENTS[s][0]],
                                                   ctx["G"][n, SEGMENTS[s][1]]-ctx["G"][n, SEGMENTS[s][0]]) < GT_STEP_DEG for s in segs)))
        info["neighbor_fails_same_decision"] = fails
        info["neighbor_2d_stable"] = stable2d
        info["neighbor_gt_stable"] = gtstable
        info["isolated"] = info["neighbors_present"] == 2 and not any(fails)
        info["isolated_and_2d_stable"] = info["isolated"] and all(stable2d)
        info["sustained"] = any(fails)
        # +-3 original samples: does S0 get the same decision right somewhere nearby?
        window_ok = 0; window_n = 0; cur = sample
        for direction in ("previous", "next"):
            cur = sample
            for _ in range(3):
                nid = cur.neighbors.get(direction)
                n = row_of.get(nid)
                if n is None or n not in members_set:
                    break
                cur = bank.samples[int(ctx["positions"][n])]
                window_n += 1
                if rec["kind"] == "severe":
                    window_ok += not mt["severe"][n]
                else:
                    i = ("l", "r").index(rec["which"]) if rec["kind"] == "knee" else SEG_NAMES.index(rec["which"])
                    key = "knee_wrong" if rec["kind"] == "knee" else "depth_err"
                    window_ok += mt[key][n, i] == 0
        info["window_pm3_present"], info["window_pm3_correct"] = window_n, int(window_ok)
        out.append(info)
    summary = {}
    for kind in ("severe", "knee", "depth", "all"):
        rr = [o for o in out if kind == "all" or o["kind"] == kind]
        summary[kind] = {"well_observed_failures": len(rr), "both_neighbors_present": sum(o["neighbors_present"] == 2 for o in rr),
                         "no_neighbor": sum(o["neighbors_present"] == 0 for o in rr),
                         "isolated": sum(o["isolated"] for o in rr), "isolated_and_2d_stable": sum(o["isolated_and_2d_stable"] for o in rr),
                         "sustained": sum(o["sustained"] for o in rr),
                         "correct_within_pm3_samples": sum(o["window_pm3_correct"] > 0 for o in rr),
                         "by_attribution": {c: {"n": sum(o["category"] == c for o in rr),
                                                "isolated": sum(o["isolated"] for o in rr if o["category"] == c),
                                                "isolated_and_2d_stable": sum(o["isolated_and_2d_stable"] for o in rr if o["category"] == c)}
                                            for c in sorted({o["category"] for o in rr})}}
    return summary


def torso_events(src, ctx, members_set):
    """W102 torso-forward temporal events (S0 step > 90 deg, GT step < 45 deg) on original neighbors."""
    P, bank, row_of = ctx["P"], ctx["bank"], ctx["row_of"]
    fwd_p = torso_forward(ctx["pred"][src]-ctx["pred"][src][:, :1]); fwd_g = torso_forward(ctx["G"])
    events = []
    for r in sorted(members_set):
        n = row_of.get(bank.samples[int(ctx["positions"][r])].neighbors.get("next"))
        if n is None or n not in members_set:
            continue
        gs, ps = angle_deg(fwd_g[r], fwd_g[n]), angle_deg(fwd_p[r], fwd_p[n])
        if np.isfinite(gs) and np.isfinite(ps) and gs < 45 and ps > 90:
            events.append({"row": int(r), "next_row": int(n), "sample_id": str(P["sid"][r]), "s0_step_deg": float(ps), "gt_step_deg": float(gs)})
    return events


# ------------------------------------------------------------------ groups A-F
def groups(ctx):
    P = ctx["P"]; same = np.flatnonzero(P["same"])
    mb, mm = ctx["metrics"]["B"], ctx["metrics"]["M"]
    qm = ctx["quality"]["M"]["status"]
    defs = {"A_well_accurate": (qm == "well") & mm["accurate"], "B_well_severe": (qm == "well") & mm["severe"],
            "C_partial_accurate": (qm == "partial") & mm["accurate"], "D_partial_severe": (qm == "partial") & mm["severe"],
            "E_both_severe": mb["severe"] & mm["severe"], "F_M_severe_B_not_severe": mm["severe"] & ~mb["severe"],
            "F_strict_M_severe_B_accurate": mm["severe"] & mb["accurate"],
            "mislocalized_M": qm == "mislocalized"}
    out = {}
    for name, flag in defs.items():
        rows = same[flag[same]]
        rec = {"rows": int(len(rows)), "videos": len(set(P["vid"][rows])), "by_split": dict(Counter(P["split"][rows])),
               "posture": dict(Counter(ctx["posture"][rows]))}
        if len(rows):
            for src in ("B", "M", "E"):
                mt = ctx["metrics"][src]
                rec[f"{src}_mpjpe"] = distribution(mt["mpjpe"][rows])
                rec[f"{src}_depth_sign_error_rate"] = float(np.nansum(mt["depth_err"][rows])/max(1, np.isfinite(mt["depth_err"][rows]).sum()))
                rec[f"{src}_wrong_knee"] = {"wrong": int(np.nansum(mt["knee_wrong"][rows])), "eligible": int(np.isfinite(mt["knee_wrong"][rows]).sum())}
                rec[f"{src}_torso_flip"] = int(np.nansum(mt["torso_flip"][rows]))
            FB, FM = P["F"]["B"][rows], P["F"]["M"][rows]
            xb, vb = base_xy(FB); xm, vm = base_xy(FM)
            rec["B_vs_M_model_visible_distance"] = distribution(full_distance(xb, vb, xm, vm))
            rec["M_only_joints"] = distribution(((FM[..., 3] > .5) & ~(FB[..., 3] > .5))[:, BASE].sum(1))
            rec["B_only_joints"] = distribution(((FB[..., 3] > .5) & ~(FM[..., 3] > .5))[:, BASE].sum(1))
            rec["M_2d_max_limb_err"] = distribution(np.nanmax(np.where(np.isfinite(ctx["quality"]["M"]["err"][rows]), ctx["quality"]["M"]["err"][rows], -1), 1))
            rec["M_full_tier_neighbor_within_tauN"] = int((ctx["full_retr"]["M"][1][rows, 0] <= ctx["tau"]["full_N"]).sum())
            rec["paired_M_minus_B_mpjpe"] = video_bootstrap(P["vid"][rows], mm["mpjpe"][rows]-mb["mpjpe"][rows])
            rec["paired_E_minus_M_mpjpe"] = video_bootstrap(P["vid"][rows], ctx["metrics"]["E"]["mpjpe"][rows]-mm["mpjpe"][rows])
            rec["sample_ids_first20"] = [str(s) for s in sorted(P["sid"][rows])[:20]]
        out[name] = rec
    return out


# ------------------------------------------------------------------ Mmask / source attribution (stored predictions only)
def source_attribution(ctx):
    from framepose.crops import crop_box
    P, bank, positions = ctx["P"], ctx["bank"], ctx["positions"]
    obs = ctx["obs"]; same = np.flatnonzero(P["same"])
    mt = ctx["metrics"]
    boxes = {}
    for c, (xk, vk) in {"B": (None, None), "M": ("input_2d_M", "input_valid_M"), "Mmask": ("input_2d_Mmask", "input_valid_Mmask")}.items():
        b = np.zeros((len(positions), 3))
        for i, p in enumerate(positions):
            x = bank.arrays["input_2d"][p] if xk is None else obs[xk][i]
            v = bank.arrays["input_valid"][p] if vk is None else obs[vk][i]
            box = crop_box(x, v, bank.samples[int(p)].image_size)
            b[i] = box.x+box.side/2, box.y+box.side/2, box.side
        boxes[c] = b
    out = {}
    FM, FK = P["F"]["M"], P["F"]["Mmask"]
    vm, vk = FM[..., 3] > .5, FK[..., 3] > .5
    removed = (vm & ~vk)[:, BASE].sum(1); added = (vk & ~vm)[:, BASE].sum(1)
    identical = np.array([FM[i].tobytes() == FK[i].tobytes() for i in range(len(FM))])
    pm, pk = ctx["pred"]["M"], ctx["pred"]["Mmask"]
    out["identical_input_rows_same_person"] = int(identical[same].sum())
    out["identical_input_rows_prediction_byte_equal"] = bool(all(pm[i].astype(np.float32).tobytes() == pk[i].astype(np.float32).tobytes() for i in same[identical[same]]))
    delta = mt["Mmask"]["mpjpe"]-mt["M"]["mpjpe"]
    side_ratio = boxes["Mmask"][:, 2]/boxes["M"][:, 2]
    shift = np.linalg.norm(boxes["Mmask"][:, :2]-boxes["M"][:, :2], axis=1)/boxes["M"][:, 2]
    strata = {"identical": identical, "removed_only": (removed > 0) & (added == 0), "added_only": (added > 0) & (removed == 0),
              "removed_and_added": (removed > 0) & (added > 0),
              "mask_differs_on_derived_tokens_only": ~identical & (removed == 0) & (added == 0)}
    rec = {}
    for name, f in strata.items():
        rows = same[f[same]]
        rec[name] = {"rows": int(len(rows)), "Mmask_minus_M": video_bootstrap(P["vid"][rows], delta[rows]),
                     "crop_side_ratio_Mmask_over_M": distribution(side_ratio[rows]), "crop_center_shift_over_M_side": distribution(shift[rows]),
                     "severe_M": int(mt["M"]["severe"][rows].sum()), "severe_Mmask": int(mt["Mmask"]["severe"][rows].sum())}
    out["Mmask_strata"] = rec
    ro = same[strata["removed_only"][same]]
    which = Counter()
    for i in ro:
        for j in BASE:
            if vm[i, j] and not vk[i, j]:
                which[JOINTS[j]] += 1
    out["removed_only_joint_counts"] = dict(which)
    bins = [0, .9, .99, 1.01, 1.1, np.inf]
    out["removed_only_by_crop_side_ratio"] = [{"ratio": [lo, None if np.isinf(hi) else hi],
                                               "Mmask_minus_M": video_bootstrap(P["vid"][ro[(side_ratio[ro] >= lo) & (side_ratio[ro] < hi)]],
                                                                                delta[ro[(side_ratio[ro] >= lo) & (side_ratio[ro] < hi)]])}
                                              for lo, hi in zip(bins[:-1], bins[1:])]
    # B -> M, descriptive strata (several factors change together; no unique attribution)
    FB = P["F"]["B"]; vb = FB[..., 3] > .5
    m_only = (vm & ~vb)[:, BASE].sum(1); b_only = (vb & ~vm)[:, BASE].sum(1)
    xb, bvb = base_xy(FB); xm, bvm = base_xy(FM)
    dmv = full_distance(xb, bvb, xm, bvm)
    dmb = mt["M"]["mpjpe"]-mt["B"]["mpjpe"]
    sr = boxes["M"][:, 2]/boxes["B"][:, 2]
    cd = np.where(vb & vm, FM[..., 2]-FB[..., 2], np.nan)[:, BASE]
    bm = {}
    for name, f in {"same_mask": (m_only == 0) & (b_only == 0), "M_adds_joints_only": (m_only > 0) & (b_only == 0),
                    "B_has_extra_joints_only": (b_only > 0) & (m_only == 0), "both_differ": (m_only > 0) & (b_only > 0)}.items():
        rows = same[f[same]]
        bm[name] = {"rows": int(len(rows)), "M_minus_B": video_bootstrap(P["vid"][rows], dmb[rows]),
                    "model_visible_distance_B_M": distribution(dmv[rows]), "crop_side_ratio_M_over_B": distribution(sr[rows]),
                    "mean_conf_change_jointly_valid": distribution(np.nanmean(cd[rows], 1)),
                    "B_partial_rows": int((ctx["quality"]["B"]["status"][rows] == "partial").sum())}
    same_mask = same[((m_only == 0) & (b_only == 0))[same]]
    bm["same_mask_by_model_visible_distance"] = [{"d_x_tauN": [lo, None if np.isinf(hi) else hi],
        "M_minus_B": video_bootstrap(P["vid"][same_mask[(dmv[same_mask] >= lo*ctx["tau"]["full_N"]) & (dmv[same_mask] < hi*ctx["tau"]["full_N"])]],
                                     dmb[same_mask[(dmv[same_mask] >= lo*ctx["tau"]["full_N"]) & (dmv[same_mask] < hi*ctx["tau"]["full_N"])]])}
        for lo, hi in zip([0, 1, 2, 4], [1, 2, 4, np.inf])]
    out["B_to_M_strata"] = bm
    out["Mconf_minus_M"] = video_bootstrap(P["vid"][same], mt["Mconf"]["mpjpe"][same]-mt["M"]["mpjpe"][same])
    out["Mmask_minus_M"] = video_bootstrap(P["vid"][same], delta[same])
    return out


def post_hoc(ctx):
    """Descriptive breakdowns added after the primary run; never enter the declared decision."""
    P, mt = ctx["P"], ctx["metrics"]
    same = np.flatnonzero(P["same"]); out = {"note": "post-hoc descriptive; not part of the declared CASE rule"}
    for src in ("M", "B"):
        F = P["F"][src]; rec = {}
        for s, (h, k, a) in zip(("l", "r"), KNEES.values()):
            leg_ok = (F[:, [h, k, a], 3] > .5).all(1)
            w = mt[src]["knee_wrong"][:, ("l", "r").index(s)]
            for name, f in (("leg_all_valid", leg_ok), ("leg_joint_invalid", ~leg_ok)):
                rows = same[f[same]]
                ok = np.isfinite(w[rows])
                rec[f"{s}_{name}"] = {"rows": int(len(rows)), "eligible": int(ok.sum()), "wrong": int(np.nansum(w[rows])),
                                      "rate": video_bootstrap(P["vid"][rows[ok]], w[rows[ok]])}
        out[f"knee_wrong_by_leg_validity_{src}"] = rec
        sev = same[mt[src]["severe"][same]]
        missing = Counter(JOINTS[j] for r in sev for j in DIRECT if F[r, j, 3] < .5)
        out[f"severe_missing_limb_joints_{src}"] = {"severe_rows": int(len(sev)), "rows_with_any_missing_limb_joint":
                                                    int(sum((F[r, DIRECT, 3] < .5).any() for r in sev)), "joint_counts": dict(missing)}
        # severe/accurate rate by number of invalid limb joints
        n_inv = (F[:, DIRECT, 3] < .5).sum(1)
        out[f"severe_by_invalid_limb_joints_{src}"] = {str(n): {"rows": int((n_inv[same] == n).sum()),
            "severe_rate": video_bootstrap(P["vid"][same[n_inv[same] == n]], mt[src]["severe"][same[n_inv[same] == n]].astype(float))}
            for n in sorted(set(n_inv[same].tolist()))}
    from framepose.crops import crop_box
    obs, bank, positions = ctx["obs"], ctx["bank"], ctx["positions"]
    FM, FK = P["F"]["M"], P["F"]["Mmask"]
    vm, vk = FM[..., 3] > .5, FK[..., 3] > .5
    ro = same[(((vm & ~vk)[:, BASE].sum(1) > 0) & ((vk & ~vm)[:, BASE].sum(1) == 0))[same]]
    shift, ratio = [], []
    for r in ro:
        p = positions[r]; size = bank.samples[int(p)].image_size
        a = crop_box(obs["input_2d_M"][r], obs["input_valid_M"][r], size)
        b = crop_box(obs["input_2d_Mmask"][r], obs["input_valid_Mmask"][r], size)
        ratio.append(b.side/a.side)
        shift.append(np.hypot((b.x+b.side/2)-(a.x+a.side/2), (b.y+b.side/2)-(a.y+a.side/2))/a.side)
    ratio, shift = np.array(ratio), np.array(shift)
    delta = mt["Mmask"]["mpjpe"]-mt["M"]["mpjpe"]
    fixed = (np.abs(ratio-1) < .01) & (shift < .01)
    out["Mmask_removed_only_crop_unchanged"] = {"rule": "side ratio within 1% and centre shift < 1% of side",
        "rows": int(fixed.sum()), "Mmask_minus_M": video_bootstrap(P["vid"][ro[fixed]], delta[ro[fixed]]),
        "removed_joints": dict(Counter(JOINTS[j] for r in ro[fixed] for j in BASE if vm[r, j] and not vk[r, j]))}
    out["Mmask_removed_only_crop_changed"] = {"rows": int((~fixed).sum()), "Mmask_minus_M": video_bootstrap(P["vid"][ro[~fixed]], delta[ro[~fixed]])}
    return out


# ------------------------------------------------------------------ qualitative selection (declared rules)
def selections(ctx, pairs_M, attributions_M, groups_rows, events):
    P = ctx["P"]; mb, mm = ctx["metrics"]["B"], ctx["metrics"]["M"]
    sel = {}
    a, b, d, m = pairs_M
    cand = [(-m["template_mpjpe"][i], str(P["sid"][a[i]]), i) for i in range(len(a)) if d[i] <= ctx["tau"]["full_S"] and m["material"][i]]
    if cand:
        i = min(cand)[2]; sel["1_strict_pair_different_gt"] = {"rows": [int(a[i]), int(b[i])], "d": float(d[i])}
    rows = groups_rows["A_well_accurate"]
    rows = [r for r in rows if ctx["full_retr"]["M"][1][r, 0] <= ctx["tau"]["full_N"]]
    if rows:
        sel["2_well_observed_success"] = {"rows": [int(min(rows, key=lambda r: (mm["mpjpe"][r], P["sid"][r])))]}
    rows = groups_rows["B_well_severe"]
    ms = [x["row"] for x in attributions_M if x["kind"] == "severe" and x["category"] == "model_specific"]
    pool = ms if ms else list(rows)
    if pool:
        sel["3_well_observed_failure"] = {"rows": [int(max(pool, key=lambda r: (mm["mpjpe"][r], P["sid"][r])))], "from_model_specific": bool(ms)}
    rows = groups_rows["E_both_severe"]
    if len(rows):
        sel["4_shared_failure"] = {"rows": [int(max(rows, key=lambda r: (min(mb["mpjpe"][r], mm["mpjpe"][r]), P["sid"][r])))]}
    rows = groups_rows["F_M_severe_B_not_severe"]
    if len(rows):
        sel["5_M_only_regression"] = {"rows": [int(max(rows, key=lambda r: (mm["mpjpe"][r]-mb["mpjpe"][r], P["sid"][r])))]}
    best = None
    for name in ("leg_l", "leg_r"):
        a, b, d, m = ctx["local_pairs"][("M", name)]
        s = 0 if name == "leg_l" else 1
        for i in np.flatnonzero(m["knee_branch"][:, s]):
            key = (d[i], str(P["sid"][a[i]]))
            if best is None or key < best[0]:
                best = (key, {"rows": [int(a[i]), int(b[i])], "d": float(d[i]), "limb": name})
    if best:
        sel["6_knee_branch_pair"] = best[1]
    best = None
    for name in ("arm_l", "arm_r"):
        a, b, d, m = ctx["local_pairs"][("M", name)]
        segs = [j for j, s in enumerate(SEG_NAMES) if SEG_CHAIN[s] == name]
        for i in np.flatnonzero(m["depth_flip"][:, segs].any(1)):
            key = (d[i], str(P["sid"][a[i]]))
            if best is None or key < best[0]:
                best = (key, {"rows": [int(a[i]), int(b[i])], "d": float(d[i]), "limb": name})
    if best:
        sel["7_arm_depth_pair"] = best[1]
    if events:
        e = max(events, key=lambda e: (e["s0_step_deg"], e["sample_id"]))
        sel["8_torso_orientation_instability"] = {"rows": [e["row"], e["next_row"]], "s0_step_deg": e["s0_step_deg"], "gt_step_deg": e["gt_step_deg"]}
    for v in sel.values():
        v["sample_ids"] = [str(P["sid"][r]) for r in v["rows"]]
    return sel


# ------------------------------------------------------------------ decision
def decide(attr_M, pair_M, verify_ok):
    if not verify_ok:
        return {"case": "D", "reason": "identity/reproduction failure"}
    a = attr_M["all_important"]
    amb = (a["share_ambiguity"]["ci95"] or [0, 0])[0] >= .10
    mod = (a["share_model_specific"]["ci95"] or [0, 0])[0] >= .10
    ctrl = pair_M["N"]["controlled"]
    # Declaration item 13: >= 3 independent video pairs among the controlled pairs that are materially different.
    controlled_ok = ctrl["material"]["count"] > 0 and ctrl["material_video_pairs"] >= 3
    if amb and not mod:
        case = "A" if controlled_ok else "C"
    elif mod and not amb:
        case = "B"
    else:
        case = "C"
    return {"case": case, "ambiguity_meaningful": amb, "model_specific_meaningful": mod,
            "controlled_ambiguous_pairs_material": ctrl["material"]["count"],
            "controlled_material_video_pairs": ctrl["material_video_pairs"], "controlled_ok": controlled_ok,
            "rule": "declaration item 13 (pure-category shares among attributed failures; mixed in denominator only)"}


# ------------------------------------------------------------------ main
def cmd_analyze(args):
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=False)
    if not args.declaration or sha(args.declaration) != args.declaration_sha:
        raise RuntimeError("STOP: analysis requires the hashed 2D-only declaration")
    decl = json.loads(Path(args.declaration).read_text())
    tau = decl["tau"]; k = decl["k_sequences"]
    bank, obs, side, positions = load_context(args.w102, args.w101_data)
    geometry = geometry_all(bank, obs)
    for alias, digest in decl["inputs"]["geometry_sha256"].items():
        key = [a for a, c in CONDITIONS.items() if c == alias][0]
        if hashlib.sha256(geometry[key].tobytes()).hexdigest() != digest:
            raise RuntimeError("STOP: model-visible input changed after declaration")
    P = pools(bank, obs, side, positions, geometry)
    pred, s0_report = load_predictions(args.w102, bank, positions)
    pred["E"] = load_hmr(args.w102, positions)
    raw = raw_references(Path(args.w102)/"data/raw", bank, positions)
    Gbank = bank.arrays["target_3d"][positions].astype(float); Gbank = Gbank - Gbank[:, :1]
    report = {"schema": "w103_analysis_v1", "declaration_sha256": args.declaration_sha, "tau": tau,
              "raw_reference": {"rows": int(len(positions)), "campose_valid_finite": int(raw["valid"].sum()),
                                "max_abs_raw_minus_bank_m": float(np.abs(raw["G"]-Gbank).max())}}
    if not raw["valid"].all() or report["raw_reference"]["max_abs_raw_minus_bank_m"] > 1e-6:
        raise RuntimeError("STOP: valid GT alignment cannot be established")
    G = Gbank
    R = np.stack([ray_rotation(p) for p in raw["root"]])
    Gr = np.einsum("nij,nkj->nki", R, G)
    lengths = np.zeros(17)
    for c in range(1, 17):
        lengths[c] = np.linalg.norm(G[:, c]-G[:, PARENT[c]], axis=-1).mean()
    template = template_pose(Gr, lengths)
    bone_sum = sum(np.linalg.norm(G[:, c]-G[:, PARENT[c]], axis=-1) for c in range(1, 17))
    sizes = np.array([bank.samples[int(p)].image_size for p in positions], float)
    metrics = {}
    for c in ("B", "M", "Moa", "Mconf", "Mmask"):
        metrics[c] = s0_metrics(pred[c], G)
    hv = np.isfinite(pred["E"]).all((1, 2))
    E = s0_metrics(np.where(hv[:, None, None], pred["E"], 0), G)
    for key, v in E.items():
        if v.dtype.kind == "f":
            v[~hv] = np.nan
        elif v.dtype == bool:
            v[~hv] = False
    metrics["E"] = E
    quality = {"B": observation_quality(bank.arrays["input_2d"][positions], bank.arrays["input_valid"][positions], raw["uv"], sizes),
               "M": observation_quality(obs["input_2d_M"], obs["input_valid_M"], raw["uv"], sizes)}
    ctx = {"P": P, "bank": bank, "obs": obs, "positions": positions, "pred": pred, "metrics": metrics, "G": G, "Gr": Gr, "R": R,
           "geo": row_geometry(Gr), "template": template, "root": raw["root"], "scale": bone_sum, "tau": tau,
           "proj": projection_features(raw["uv"]), "quality": quality, "posture": posture(G, raw["gravity"]),
           "row_of": {str(s): i for i, s in enumerate(P["sid"])}}
    # ---------------- parity with W102 aggregates
    w102 = json.loads((Path(args.w102)/"results/evaluation/report.json").read_text())["results"]
    parity = {}
    for split in SPLITS:
        rows = np.flatnonzero(P["same"] & (P["split"] == split))
        fo = w102[split]["failure_overlap_same_person"]
        mine = {}
        for key, (kb, km) in {"severe": (metrics["B"]["severe"], metrics["M"]["severe"]),
                              "knee_wrong_l": (metrics["B"]["knee_wrong"][:, 0], metrics["M"]["knee_wrong"][:, 0]),
                              "knee_wrong_r": (metrics["B"]["knee_wrong"][:, 1], metrics["M"]["knee_wrong"][:, 1]),
                              "torso_fwd_flip": (metrics["B"]["torso_flip"], metrics["M"]["torso_flip"])}.items():
            kb, km = np.asarray(kb, float)[rows], np.asarray(km, float)[rows]
            ok = np.isfinite(kb) & np.isfinite(km)
            mine[key] = {"common_support": int(ok.sum()), "both_fail": int(((kb == 1) & (km == 1) & ok).sum()),
                         "B_only": int(((kb == 1) & (km == 0) & ok).sum()), "M_only": int(((km == 1) & (kb == 0) & ok).sum()),
                         "neither": int(((kb == 0) & (km == 0) & ok).sum())}
        parity[split] = {"B_mean": [float(metrics["B"]["mpjpe"][rows].mean()), w102[split]["headline"]["B_mpjpe_same"]],
                         "M_mean": [float(metrics["M"]["mpjpe"][rows].mean()), w102[split]["headline"]["M_mpjpe_same"]],
                         "E_mean": [float(np.nanmean(metrics["E"]["mpjpe"][rows])), w102[split]["tails_same_person"]["E"]["mean"]],
                         "failure_overlap_equal": mine == fo, "failure_overlap": mine}
    parity["pass"] = all(abs(v[a][0]-v[a][1]) < 1e-6 and v["failure_overlap_equal"] for v in parity.values() for a in ("B_mean", "M_mean", "E_mean"))
    report["w102_parity"] = parity
    if not parity["pass"]:
        dump(out/"report_partial.json", report)
        raise RuntimeError("STOP: W102 metric parity failed")
    # ---------------- evaluation-reference validity (found after the primary run; physical, not an outcome threshold)
    behind = raw["root"][:, 1] <= 0
    report["reference_validity"] = {
        "rule": "GT root (SMPL pelvis) camera depth > 0; a visible subject cannot lie behind the camera",
        "rows_root_behind_camera": int(behind.sum()), "sequences": dict(Counter(P["seq"][behind])),
        "same_person_rows": int((behind & P["same"]).sum()),
        "sensitivity_exclusion_applied": bool(args.exclude_root_behind_camera)}
    if args.exclude_root_behind_camera:
        keep = ~behind
        P["same"] = P["same"] & keep
        P["members"] = {s: m[keep[m]] for s, m in P["members"].items()}
    else:
        keep = np.ones(len(positions), bool)
    # ---------------- populations
    pop = {}
    for split in SPLITS:
        r = P["split"] == split
        s = r & P["same"]
        pop[split] = {"actor_frames": int(r.sum()), "videos": len(set(P["vid"][r])), "actor_sequences": len(set(P["seq"][r])),
                      "same_person_rows": int(s.sum()), "same_person_videos": len(set(P["vid"][s])),
                      "same_person_actor_sequences": len(set(P["seq"][s])),
                      "historical_all_target_visible_same_person": int((bank.arrays["target_valid"][positions].all(1) & s).sum()),
                      "identity": dict(Counter(np.asarray(side["identity"])[r])),
                      "quality_B_all_rows": dict(Counter(quality["B"]["status"][r])),
                      "quality_B_same_person": dict(Counter(quality["B"]["status"][s])),
                      "quality_M_same_person": dict(Counter(quality["M"]["status"][s])),
                      "full_tier_eligible": {"B": int((full_eligible(P["F"]["B"]) & r).sum()), "M": int((full_eligible(P["F"]["M"]) & s).sum())},
                      "severe_same_person": {c: int(metrics[c]["severe"][s].sum()) for c in ("B", "M")},
                      "posture_same_person": dict(Counter(ctx["posture"][s])),
                      "hmr_valid_same_person": int(hv[s].sum())}
    report["populations"] = pop
    report["ray_angle_deg"] = distribution(angle_deg(raw["root"], np.array([[0, 1, 0]]*len(positions))))
    report["template_bone_lengths_m"] = dict(zip(JOINTS[1:], lengths[1:].tolist()))
    # ---------------- residual error by observation quality
    rq = {}
    for src in ("B", "M"):
        rows_all = P["members"]["M"]
        for st in ("well", "partial", "mislocalized"):
            rows = rows_all[quality[src]["status"][rows_all] == st]
            mt = metrics[src]
            rq[f"{src}_{st}"] = {"rows": int(len(rows)), "videos": len(set(P["vid"][rows])), "mpjpe": distribution(mt["mpjpe"][rows]),
                                 "mpjpe_video_ci": video_bootstrap(P["vid"][rows], mt["mpjpe"][rows]),
                                 "severe_rate": video_bootstrap(P["vid"][rows], mt["severe"][rows].astype(float)),
                                 "accurate_rate": video_bootstrap(P["vid"][rows], mt["accurate"][rows].astype(float)),
                                 "wrong_knee": {"wrong": int(np.nansum(mt["knee_wrong"][rows])), "eligible": int(np.isfinite(mt["knee_wrong"][rows]).sum())},
                                 "depth_sign_error": {"errors": int(np.nansum(mt["depth_err"][rows])), "eligible": int(np.isfinite(mt["depth_err"][rows]).sum())},
                                 "E_mpjpe": distribution(metrics["E"]["mpjpe"][rows])}
    report["residual_by_observation_quality_same_person"] = rq
    # ---------------- full tier retrieval & pairs
    ctx["full_retr"] = {src: full_retrieval(P, src, k) for src in ("B", "M")}
    split_of = P["split"]
    full = {}
    pairs_cache = {}
    for src in ("B", "M"):
        rec = {}
        for t in ("S", "N"):
            a, b, d, same = full_pairs(P, src, tau["full_"+t])
            rec[t] = summarize_pairs(a, b, d, same, ctx, src, t)
            if t == "N":
                cross = ~same
                pairs_cache[src] = (a[cross], b[cross], d[cross], pair_metrics(subset(ctx["geo"], a[cross]), subset(ctx["geo"], b[cross]),
                                                                              template[a[cross]], template[b[cross]]))
        idx, dist = ctx["full_retr"][src]
        members = np.array([r for r in P["members"][src] if idx[r, 0] >= 0])
        rec["query_level_N"] = query_level(idx, dist, members, tau["full_N"], ctx, split_of)
        rec["query_level_S"] = query_level(idx, dist, members, tau["full_S"], ctx, split_of)
        rec["binned_curve"] = binned_curve(idx, dist, members, tau["full_N"], decl["bins_x_tauN"], ctx)
        rec["queries_without_any_same_mask_reference"] = int(sum(1 for r in P["members"][src] if idx[r, 0] < 0))
        full[src] = rec
    report["full_tier"] = full
    # ---------------- oracle-geometry control (GT projections, all rows)
    Xp = ctx["proj"].reshape(len(positions), -1)
    allrows = np.flatnonzero(keep)
    idx, dist = vector_retrieval(Xp, allrows, P, k, 13)
    a, b, d, same = enumerate_pairs(Xp, allrows, P["vid"], tau["full_N"], 13)
    # GT projections are exact by construction: every row counts as "well observed" in the oracle regime.
    ctx_oracle = dict(ctx); ctx_oracle["quality"] = {"B": {"status": np.array(["well"]*len(positions))}}
    oracle = {"regime": "oracle_geometry", "N": summarize_pairs(a, b, d, same, ctx_oracle, "B", "N"),
              "query_level_N": query_level(idx, dist, allrows, tau["full_N"], ctx, split_of),
              "binned_curve": binned_curve(idx, dist, allrows, tau["full_N"], decl["bins_x_tauN"], ctx)}
    a, b, d, same = enumerate_pairs(Xp, allrows, P["vid"], tau["full_S"], 13)
    oracle["S"] = summarize_pairs(a, b, d, same, ctx_oracle, "B", "S")
    oracle["note"] = "pair_resolution/S0 fields inside use the B source predictions for these GT-matched pairs"
    report["oracle_geometry_tier"] = oracle
    # ---------------- local tier
    local, local_retr = {}, {}
    for src in ("B", "M"):
        local[src], local_retr[src] = local_tier(P, src, ctx, k)
    report["local_tier"] = local
    # ---------------- attribution
    attributions = {}
    for src in ("M", "B"):
        rows = np.flatnonzero(P["same"])
        insts = failure_instances(src, ctx, rows)
        attributions[src] = [attribute(i, src, ctx, ctx["full_retr"][src], local_retr[src]) for i in insts]
    report["attribution"] = {}
    for src, recs in attributions.items():
        report["attribution"][src] = {"by_kind": attribution_summary(recs), "by_split": {}}
        for split in SPLITS:
            report["attribution"][src]["by_split"][split] = attribution_summary([r for r in recs if r["split"] == split])
        report["attribution"][src]["quality_of_failures"] = {kind: dict(Counter(r["quality"] for r in recs if r["kind"] == kind))
                                                              for kind in ("severe", "knee", "depth")}
        report["attribution"][src]["failing_limb_invalid_among_partial"] = dict(Counter(
            (r["kind"], r.get("failing_limb_all_valid")) for r in recs if r["quality"] == "partial"))
        # HMR contextual cross-tab per category
        hm = {}
        for r in recs:
            row = r["row"]
            if r["kind"] == "severe":
                f = metrics["E"]["severe"][row] if hv[row] else None
            elif r["kind"] == "knee":
                f = metrics["E"]["knee_wrong"][row, ("l", "r").index(r["which"])]
            else:
                f = metrics["E"]["depth_err"][row, SEG_NAMES.index(r["which"])]
            f = None if f is None or not np.isfinite(float(f)) else bool(f)
            key = (r["kind"], r["category"])
            hm.setdefault(key, Counter())[{None: "HMR_ineligible", True: "HMR_also_fails", False: "HMR_correct"}[f]] += 1
        report["attribution"][src]["hmr_contextual"] = {f"{k[0]}|{k[1]}": dict(v) for k, v in sorted(hm.items())}
    # ---------------- groups, temporal, HMR, source attribution
    grp = groups(ctx)
    report["groups"] = grp
    same_rows = np.flatnonzero(P["same"])
    defs_rows = {}
    for name in ("A_well_accurate", "B_well_severe", "E_both_severe", "F_M_severe_B_not_severe"):
        mm, mb = metrics["M"], metrics["B"]
        flag = {"A_well_accurate": (quality["M"]["status"] == "well") & mm["accurate"],
                "B_well_severe": (quality["M"]["status"] == "well") & mm["severe"],
                "E_both_severe": mb["severe"] & mm["severe"], "F_M_severe_B_not_severe": mm["severe"] & ~mb["severe"]}[name]
        defs_rows[name] = same_rows[flag[same_rows]]
    members_M = set(int(r) for r in same_rows)
    report["temporal"] = {"M": temporal(attributions["M"], "M", ctx, members_M),
                          "B": temporal(attributions["B"], "B", ctx, set(range(len(positions))))}
    events = torso_events("M", ctx, members_M)
    report["temporal"]["M_torso_forward_events"] = {"count": len(events), "events": events}
    hmr = {}
    for split in SPLITS:
        rows = same_rows[(P["split"][same_rows] == split) & hv[same_rows]]
        sm, se = metrics["M"]["severe"][rows], metrics["E"]["severe"][rows]
        am, ae = metrics["M"]["accurate"][rows], metrics["E"]["accurate"][rows]
        hmr[split] = {"rows": int(len(rows)), "S0M_severe_E_not": int((sm & ~se).sum()), "both_severe": int((sm & se).sum()),
                      "E_severe_S0M_not": int((se & ~sm).sum()), "S0M_accurate_E_not_accurate": int((am & ~ae).sum()),
                      "E_accurate_S0M_not_accurate": int((ae & ~am).sum()), "both_accurate": int((am & ae).sum()),
                      "E_minus_M_mpjpe": video_bootstrap(P["vid"][rows], metrics["E"]["mpjpe"][rows]-metrics["M"]["mpjpe"][rows]),
                      "wrong_knee": {c: {"wrong": int(np.nansum(metrics[c]["knee_wrong"][rows])), "eligible": int(np.isfinite(metrics[c]["knee_wrong"][rows]).sum())} for c in ("M", "E")},
                      "depth_sign": {c: {"errors": int(np.nansum(metrics[c]["depth_err"][rows])), "eligible": int(np.isfinite(metrics[c]["depth_err"][rows]).sum())} for c in ("M", "E")}}
    report["hmr_contextual"] = hmr
    report["source_attribution"] = source_attribution(ctx)
    report["post_hoc_descriptive"] = post_hoc(ctx)
    report["selections"] = selections(ctx, pairs_cache["M"], attributions["M"], defs_rows, events)
    verify_ok = True
    report["decision"] = decide(report["attribution"]["M"]["by_kind"], full["M"], verify_ok)
    report["decision_B_source_secondary"] = decide(report["attribution"]["B"]["by_kind"], full["B"], verify_ok)
    dump(out/"analysis_report.json", report)
    dump(out/"attribution_records.json", {src: recs for src, recs in attributions.items()})
    np.savez_compressed(out/"per_row.npz", sample_id=P["sid"], split=P["split"], video=P["vid"], same_person=P["same"],
                        quality_B=quality["B"]["status"], quality_M=quality["M"]["status"], posture=ctx["posture"],
                        **{f"mpjpe_{c}": metrics[c]["mpjpe"] for c in metrics},
                        full_idx_M=ctx["full_retr"]["M"][0], full_dist_M=ctx["full_retr"]["M"][1],
                        full_idx_B=ctx["full_retr"]["B"][0], full_dist_B=ctx["full_retr"]["B"][1])
    print(json.dumps({"parity": parity["pass"], "decision": report["decision"], "decision_B": report["decision_B_source_secondary"],
                      "attribution_M": {k: v["categories"] for k, v in report["attribution"]["M"]["by_kind"].items()}}, indent=1, default=str))
