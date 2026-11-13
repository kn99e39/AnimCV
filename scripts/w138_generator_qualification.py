#!/usr/bin/env python3
"""W138 (EXPERIMENTAL, opt-in): independent pretrained motion-generator qualification and frozen selection.

Runs BEFORE any model weight is downloaded and before any model output exists.  It records, for every candidate named
by the Batch (Kimodo, ARDY, MotionLab, ProjFlow) plus one further relevant published alternative (GEM-X):

  * the pinned code revision (remote HEAD == local code-only clone HEAD) and the code licence as committed,
  * the Hugging Face checkpoint record at a pinned revision (gated flag, card licence, files, published LFS SHA-256),
  * primary-source clauses (README / docs / licence / source lines) checked MECHANICALLY as normalised substrings of
    the file read from the pinned git blob or the pinned Hugging Face revision,
  * the gated text-encoder access state of the user's own existing credential (one HEAD, no body, no redirect).

A predicate counts as true only when every one of its clauses is VERIFIED from a primary source in this run; an
unverified predicate never counts as true.  The frozen selection rule (``evaluate_candidate`` / ``select``) is a pure
function of that fact table and is unit-tested in ``tests/test_w138_generator_qualification.py``.

SELECTION RULE (frozen here, before any model output; it has no branch that reads a model result)
  A candidate is ELIGIBLE iff all six predicates are VERIFIED_TRUE:
    E1c code rights     an affirmative licence file in the official repository at the pinned commit covers running the
                        code for a private, local, non-commercial research evaluation;
    E1w weight rights   the rights holder's own weight-scoped terms (not a code licence, not a third-party mirror label)
                        permit that use at the execution location;
    E2  access          every artifact whose bytes can influence the declared inference is public and ungated at a
                        pinned revision, and NO gated artifact is requested, obtained or substituted.  If the vendor's
                        default runtime loads a gated component, E2 holds only for a vendor-documented AND
                        vendor-benchmarked mode in which the vendor's own code discards that component's output, the
                        component being left out only through a vendor-documented loader parameter;
    E3  resources       the declared inference fits this machine (16 GB VRAM, bounded disk);
    E4  conditioning    a vendor-documented native partial-motion / keyframe conditioning can hold source-bound context
                        outside the edit interior and leave the interior free, without asserting any 3D observation
                        inside the interior and without unrestricted body edits being imported;
    E5  recoverability  the model returns per-frame joint rotations on a skeleton for which a closed-form mapping to the
                        BaseRig ``upperarm_l`` / ``lowerarm_l`` channels already exists (W129), with no new interface.
  Among ELIGIBLE candidates the selection is the UNIQUE maximum of the preference tuple
    (P1 native partial-motion conditioning benchmarked by the vendor in the declared text mode,
     P2 per-frame, frame-indexed, non-autoregressive whole-window generation (no temporal tokenisation), which the
        Batch's exact 30-fps frame correspondence and two-sided bounded gap require,
     P3 native 30 fps).
  No eligible candidate, or a tie at the maximum -> QUALIFICATION_BLOCKED.  At most one candidate is ever selected; an
  eligible-but-not-selected candidate is NOT run.

    PY=C:/Projects/AnimCV/.venv_build_full/Scripts/python.exe
    $PY -X utf8 scripts/w138_generator_qualification.py --out-dir C:/Users/dna10/animcv-w138/results/qualification
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
W138 = Path("C:/Users/dna10/animcv-w138")
BATCH, BLOB, START = 138, "8129b0b504cfdca11c313ffcd6c3a53d75e2bd32", "df9eddc1eb84b5de79f2eaab848b62b4a203e21f"
W137_WORKLOG = "docs/137_WORKLOG_UNIMATE_OFFICIAL_INBETWEEN_S1_RAW_GENERATOR_UNUSABLE.md"
HF = "https://huggingface.co"
UA = {"User-Agent": "research-metadata-client/1.0"}
GATED_BASE = "meta-llama/Meta-Llama-3-8B-Instruct"
TRUE, FALSE, UNV = "VERIFIED_TRUE", "VERIFIED_FALSE", "NOT_VERIFIED"
PREDICATES = ("E1c_code_rights", "E1w_weight_rights", "E2_access", "E3_resources", "E4_conditioning", "E5_recoverability")
STOP_OF = {"E1c_code_rights": "RIGHTS_UNRESOLVED", "E1w_weight_rights": "RIGHTS_UNRESOLVED", "E2_access": "ACCESS_BLOCKED",
           "E3_resources": "RESOURCE_BLOCKED", "E4_conditioning": "CONDITIONING_BLOCKED", "E5_recoverability": "REPRESENTATION_BLOCKED"}
PREFERENCES = ("P1_vendor_benchmarked_partial_motion_in_declared_text_mode", "P2_per_frame_non_autoregressive_whole_window", "P3_native_30fps")
INTENDED_USE = ("private, local, non-commercial technical research evaluation on one machine in the Republic of Korea; one inference; "
                "no redistribution of weights or model outputs; no training/finetuning; no production or commercial use")

CANDIDATES = {
    "Kimodo": {
        "project": "https://research.nvidia.com/labs/sil/projects/kimodo/", "paper": "arXiv:2603.15546 (Kimodo: Scaling Controllable Human Motion Generation)",
        "code_repo": "https://github.com/nv-tlabs/kimodo", "clone": W138 / "ext" / "kimodo", "api": "nv-tlabs/kimodo",
        "hf_repo": "nvidia/Kimodo-SOMA-RP-v1.2", "weight_file": "model.safetensors",
        "task": "text- and/or constraint-conditioned kinematic motion diffusion (full-body keyframes, end-effectors, 2D root path/waypoints)",
        "input_modality": "text (optional) + kinematic constraints as joint rotations/root positions on the SOMA skeleton; no video, no 2D keypoints",
        "skeleton": "SOMA: somaskel30 inside the model, somaskel77 in the public I/O", "fps_window": "30 fps; up to 10 s (300 frames) per prompt",
        "output": "local/global joint rotation matrices, joint positions, root positions (NPZ / Python dict)",
        "training_data": "Bones Rigplay 1 (proprietary optical mocap, 700 h); card: provenance for all datasets = Yes"},
    "ARDY": {
        "project": "https://research.nvidia.com/labs/sil/projects/ardy/", "paper": "arXiv:2607.08741 (ARDY: Autoregressive Diffusion with Hybrid Representation)",
        "code_repo": "https://github.com/nv-tlabs/ardy", "clone": Path("C:/Users/dna10/animcv-w134/ext/ardy"), "api": "nv-tlabs/ardy",
        "hf_repo": "nvidia/ARDY-SOMA-RP-30FPS-Horizon60", "weight_file": "denoiser.safetensors",
        "task": "autoregressive (streaming) text- and constraint-conditioned motion diffusion with a learned motion tokenizer",
        "input_modality": "text (optional) + history poses + kinematic constraints on the SOMA skeleton; no video",
        "skeleton": "SOMA 30 joints", "fps_window": "30 fps; 60-frame generation horizon, 4 frames per token, up to 8 s history",
        "output": "global root translation + joint rotations", "training_data": "Bones Rigplay 1 (proprietary optical mocap)"},
    "MotionLab": {
        "project": "https://diouo.github.io/motionlab.github.io/", "paper": "arXiv:2502.02358 (ICCV 2025)",
        "code_repo": "https://github.com/Diouo/MotionLab", "clone": W138 / "ext" / "MotionLab", "api": "Diouo/MotionLab",
        "hf_repo": "nielsr/motionlab", "weight_file": "motionflow.ckpt (Google Drive folder 1ph3otOlYqINvwFuvrt92nvzypZDs4haj; not on the HF repo)",
        "task": "unified motion generation/editing (text, hint/trajectory, in-between, style, source-motion editing)",
        "input_modality": "text / trajectory hints / source motion in the SMPL-based HumanML3D-MotionFix format", "skeleton": "SMPL 22 joints (6D rotations + translation)",
        "fps_window": "HumanML3D-format clips (20 fps upstream convention; not re-verified)", "output": "SMPL-parameter motion",
        "training_data": "HumanML3D (AMASS-derived) + MotionFix"},
    "ProjFlow": {
        "project": "https://akihisa-watanabe.github.io/projflow.github.io/", "paper": "arXiv:2602.22742 (CVPR 2026)",
        "code_repo": "https://github.com/Akihisa-Watanabe/ProjFlow", "clone": W138 / "ext" / "ProjFlow", "api": "Akihisa-Watanabe/ProjFlow",
        "hf_repo": "Akihisa-Watanabe/ProjFlow", "weight_file": "checkpoints/t2m/ACMDM_Raw_Flow_S_PatchSize22/model/latest.tar",
        "task": "training-free projection sampler enforcing exact linear spatial constraints on an ACMDM flow-matching prior (inpainting, 2D-to-3D lifting)",
        "input_modality": "text + exact joint-position constraints (3D or projected 2D) on the HumanML3D 22-joint absolute-coordinate representation",
        "skeleton": "HumanML3D/T2M 22 joints, absolute joint coordinates (positions only, no rotations)", "fps_window": "HumanML3D convention (20 fps, up to 196 frames; not re-verified)",
        "output": "joint positions (T, 22, 3)", "training_data": "HumanML3D (AMASS-derived)"},
    "GEM-X": {
        "project": "https://research.nvidia.com/labs/dair/gem/", "paper": "arXiv:2505.01425 (GENMO / GEM)",
        "code_repo": "https://github.com/NVlabs/GEM-X", "clone": W138 / "ext" / "GEM-X", "api": "NVlabs/GEM-X",
        "hf_repo": "nvidia/GEM-X", "weight_file": "motion_native77_e774_s292175.ckpt",
        "task": "monocular-video human body pose / motion ESTIMATION on SOMA (plus hand pose)",
        "input_modality": "RGB video: Body Pose image features + 77-point 2D keypoint observations + camera/timing conditions",
        "skeleton": "SOMA 77", "fps_window": "30 Hz; clips up to 60 s", "output": "SOMA body parameters in camera and world coordinates",
        "training_data": "not reviewed"},
}

# Clause sets.  Each entry: (source key, [clauses that must ALL be present as whitespace-normalised substrings]).
CLAUSES = {
    "Kimodo": {
        "code_licence": [("git:LICENSE", ["Apache License", "Version 2.0, January 2004"]),
                         ("git:README.md", ["This codebase is licensed under [Apache-2.0](LICENSE). Note that model checkpoints and data are licensed separately as indicated on the HuggingFace download pages."])],
        "weight_licence": [("hf:LICENSE", ["NVIDIA Open Model License Agreement", "Models are commercially usable.",
                                           "NVIDIA does not claim ownership to any outputs generated using the Models or Derivative Models.",
                                           "license to publicly perform, publicly display, reproduce, use, create derivative works of, make, have made, sell, offer for sale, distribute"]),
                           ("hf:README.md", ["license_name: nvidia-open-model-license", "This model is released under the [NVIDIA Open Model License]",
                                             "### Deployment Geography: Global", "This model is ready for commercial use."])],
        "text_gate": [("git:docs/source/getting_started/installation.md", ["The Kimodo text encoder relies on the **gated** `meta-llama/Meta-Llama-3-8B-Instruct` model"]),
                      ("git:kimodo/model/load_model.py", ['"base_model_name_or_path": "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp"'])],
        "no_text_mode": [("git:docs/source/benchmark/introduction.md", ["`constraints_notext`: test cases with only constraints as input (no text prompt)"]),
                         ("git:docs/source/benchmark/results.md", ["Without text FB Pos"]),
                         ("git:kimodo/model/kimodo_model.py", ["# handle empty string (set to zero)", "empty_text_mask = [len(text.strip()) == 0 for text in texts]",
                                                               "text_feat[empty_text_mask] = 0", "tensor_text_length[empty_text_mask] = 0"]),
                         ("git:kimodo/model/llm2vec/llm2vec_wrapper.py", ["encoded_text = encoded_text[:, None]", "lengths = np.ones(len(encoded_text), dtype=int).tolist()"]),
                         ("git:kimodo/model/load_model.py", ["text_encoder: Pre-built text encoder to reuse. When provided, skips text encoder selection/instantiation entirely."]),
                         ("hf:README.md", ["given a text prompt and/or constraints on the motion like full-body poses"])],
        "resources": [("git:README.md", ["set `TEXT_ENCODER_DEVICE=cpu` when running Kimodo commands to force text encoding to the CPU. This is slightly slower but reduces VRAM usage to <3 GB."])],
        "conditioning": [("git:docs/source/key_concepts/constraints.md", ["**Sparse full-body keyframe**: full-body pose targets at specific frames.",
                                                                          "this constrains the smoothed root position and all body joint positions at a specific frame"]),
                         ("git:docs/source/user_guide/constraints.md", ["`frame_indices` (array of integers): 0-based frame indices within the generated clip.",
                                                                        "Note the `local_joint_rot` will not explicitly be constrained, the constraint will be on the joint positions that results from FK with the given joint rotations."]),
                         ("git:docs/source/key_concepts/limitations.md", ["The number of constrained frames per constraint type should be less than 20"])],
        "output": [("git:README.md", ["`local_rot_mats`: Local (parent-relative) joint rotation matrices `[T, J, 3, 3]`"]),
                   ("hf:config.yaml", ["fps: 30", "_target_: kimodo.skeleton.SOMASkeleton30"]),
                   ("git:kimodo/skeleton/definitions.py", ['("LeftArm", "LeftShoulder")', '("LeftForeArm", "LeftArm")', '("LeftHand", "LeftForeArm")'])],
        "per_frame": [("git:docs/source/key_concepts/motion_representation.md", ["At each frame, the pose feature vector is the concatenation of:"])],
    },
    "ARDY": {
        "code_licence": [("git:LICENSE", ["Apache License", "Version 2.0, January 2004"])],
        "weight_licence": [("hf:LICENSE", ["NVIDIA Open Model", "Models are commercially usable."]),
                           ("hf:README.md", ["license_name: nvidia-open-model-agreement"])],
        "text_gate": [("git:README.md", ["The text encoder relies on the gated [meta-llama/Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) model"])],
        "no_text_mode": [("git:ardy/model/ardy_model.py", ["# handle empty string (set to zero)", "text_feat[empty_text_mask] = 0", "tensor_text_length[empty_text_mask] = 0"]),
                         ("git:ardy/model/load_model.py", ["to load a model without a text encoder, pass text_encoder=False"]),
                         ("git:README.md", ["Without text FB Pos"])],
        "conditioning": [("git:README.md", ["flexible long-horizon kinematic constraints (root paths/waypoints, full-body keyframes, and sparse joint positions/rotations)"])],
        "output": [("hf:config.yaml", ["SOMASkeleton30", "fps: 30"])],
        "autoregressive_tokenised": [("git:README.md", ["ARDY is an autoregressive diffusion model designed for interactive motion generation"]),
                                     ("hf:config.yaml", ["num_frames_per_token: 4", "fps: 30"])],
    },
    "MotionLab": {
        "restricted_assets": [("git:README.md", ["Download the [AMASS](https://amass.is.tue.mpg.de/) dataset and [MotionFix](https://github.com/atnikos/motionfix) dataset.",
                                                 "bash prepare/download_smpl_model.sh", "SMPL_NEUTRAL.pkl",
                                                 "[motion_encoder.ckpt, motionclip.pth.tar, motionflow.ckpt](https://drive.google.com/drive/folders/1ph3otOlYqINvwFuvrt92nvzypZDs4haj?usp=drive_link)"])],
        "demo_needs_dataset": [("git:demo.py", ["from rfmotion.data.get_data import get_datasets", "test_dataloader = dataset.test_dataloader()"])],
        "conditioning": [("git:README.md", ['"text_hint", "inbetween", "text_inbetween", "style", "source_text", "source_hint", "source_text_hint"'])],
    },
    "ProjFlow": {
        "weights_hosted_by_author": [("git:README.md", ["We host the pretrained checkpoints on Hugging Face:", "https://huggingface.co/Akihisa-Watanabe/ProjFlow",
                                                        "Official PyTorch implementation of **ProjFlow**"])],
        "weight_licence": [("hf:README.md", ["license: mit"])],
        "conditioning": [("git:README.md", ["a training-free projection sampler for enforcing exact (hard) spatial constraints", "python demo/inpaint/run_inpaint.py",
                                            "python -m demo.run --config demo/configs/lift_heart.yaml"])],
        "positions_only": [("git:README.md", ["Input motion from demo output (`(T,J,3)` world coordinates)."])],
    },
    "GEM-X": {
        "code_licence": [("git:LICENSE", ["Apache License", "Version 2.0, January 2004"]),
                         ("git:README.md", ["The default source license is [Apache 2.0](LICENSE)."])],
        "weight_licence": [("hf:README.md", ["license_name: nvidia-open-model-license"])],
        "video_estimator": [("git:README.md", ["GEM-X: Full-Body and Hand Pose Estimation from Image and Video",
                                               "Body Motion uses temporal context to estimate camera-space and world-space human motion from monocular video."]),
                            ("git:docs/MODEL_OVERVIEW.md", ["The temporal model consumes Body Pose query features, 2D keypoint observations, bounding boxes, camera intrinsics/angular velocity, and physical-time inputs."])],
    },
}
LICENCE_FILE_RE = re.compile(r"(^|/)(LICEN[CS]E|COPYING|NOTICE)([._-][^/]*)?$", re.I)


# ------------------------------------------------------------------------------------------------ pure rule ---
def norm_ws(s: str) -> str:
    return " ".join(s.split())


def clauses_present(text: str, clauses) -> dict:
    t = norm_ws(text)
    return {c: norm_ws(c) in t for c in clauses}


def tri(*conds) -> str:
    """VERIFIED_TRUE iff every condition is True; VERIFIED_FALSE iff any is False; NOT_VERIFIED if none False but some None."""
    if any(c is False for c in conds):
        return FALSE
    if any(c is None for c in conds):
        return UNV
    return TRUE


def evaluate_candidate(pred: dict) -> dict:
    """``pred``: predicate -> VERIFIED_TRUE / VERIFIED_FALSE / NOT_VERIFIED (a missing predicate is NOT_VERIFIED)."""
    vals = {p: pred.get(p, UNV) for p in PREDICATES}
    bad = [p for p in PREDICATES if vals[p] != TRUE]
    if not bad:
        return {"eligible": True, "status": "ELIGIBLE", "blocking": [], "predicates": vals}
    falses = [p for p in bad if vals[p] == FALSE]
    first = (falses or bad)[0]
    status = STOP_OF[first] if vals[first] == FALSE else f"NOT_VERIFIED:{first}"
    return {"eligible": False, "status": status, "blocking": bad, "predicates": vals}


def select(table: dict) -> dict:
    """``table``: name -> {"predicates": {...}, "preferences": {P: bool}}.  At most one selection; ties are refused."""
    ev = {n: evaluate_candidate(c.get("predicates", {})) for n, c in table.items()}
    elig = sorted(n for n, e in ev.items() if e["eligible"])
    rank = {n: tuple(bool(table[n].get("preferences", {}).get(p, False)) for p in PREFERENCES) for n in elig}
    out = {"evaluation": ev, "eligible": elig, "preference_tuple": {n: list(v) for n, v in rank.items()}}
    if not elig:
        return {**out, "selected": None, "status": "QUALIFICATION_BLOCKED", "reason": "no eligible candidate"}
    best = max(rank.values())
    top = [n for n in elig if rank[n] == best]
    if len(top) != 1:
        return {**out, "selected": None, "status": "QUALIFICATION_BLOCKED", "reason": f"ambiguous maximum {top}: no arbitrary pick"}
    return {**out, "selected": top[0], "status": "SELECTED", "not_selected_eligible": [n for n in elig if n != top[0]]}


def classify_gated_head(status):
    if status in (200, 301, 302, 303, 307, 308):
        return "GRANTED"
    if status in (401, 403):
        return "NOT_GRANTED"
    return "UNKNOWN"


# ------------------------------------------------------------------------------------------------ io helpers ---
def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git(*a, cwd=ROOT):
    return subprocess.run(["git", *a], cwd=str(cwd), capture_output=True, text=True, encoding="utf-8", errors="replace").stdout.strip()


def blob(repo, path):
    r = subprocess.run(["git", "show", f"HEAD:{path}"], cwd=str(repo), capture_output=True)
    return r.stdout if r.returncode == 0 else None


def get(url, headers=None, timeout=60):
    req = urllib.request.Request(url, headers=dict(UA, **(headers or {})))
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, b""
    except (urllib.error.URLError, OSError) as e:
        return None, str(e).encode()


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *a, **k):
        return None


def head_status(url, headers=None):
    req = urllib.request.Request(url, headers=dict(UA, **(headers or {})), method="HEAD")
    try:
        r = urllib.request.build_opener(_NoRedirect).open(req, timeout=30)
        return r.status, dict(r.headers)
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers)
    except (urllib.error.URLError, OSError):
        return None, {}


def local_token():
    """Existing credential only.  Never printed, logged or written."""
    t = os.environ.get("HF_TOKEN")
    if t:
        return t.strip(), "env:HF_TOKEN"
    p = Path.home() / ".cache/huggingface/token"
    if p.is_file():
        return p.read_text().strip(), str(p)
    return None, None


def save(out, rel, data):
    p = out / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data if isinstance(data, bytes) else data.encode("utf-8"))
    return sha256_file(p)


# ------------------------------------------------------------------------------------------------ gathering ---
def gather_candidate(name, c, out):
    """Primary-source record of one candidate.  Only metadata and small text files are fetched; never a weight file."""
    rec = {"name": name, **{k: str(v) for k, v in c.items() if k != "clone"}, "retrieved": now()}
    d = out / "sources" / name
    clone = c["clone"]
    remote = git("ls-remote", c["code_repo"] + ".git", "HEAD").split("\t")[0]
    local = git("rev-parse", "HEAD", cwd=clone)
    tree = git("ls-tree", "-r", "--name-only", "HEAD", cwd=clone).splitlines()
    lic_files = sorted(f for f in tree if LICENCE_FILE_RE.search(f) and f.count("/") <= 1)
    st, body = get(f"https://api.github.com/repos/{c['api']}")
    api = json.loads(body) if st == 200 else {}
    save(d, "github_api.json", body if st == 200 else b"{}")
    rec["code"] = {"remote_head": remote, "local_clone_head": local, "pinned_commit": local, "head_matches_remote": bool(remote and remote == local),
                   "commit_date": git("log", "-1", "--format=%cI", cwd=clone), "licence_files_in_tree": lic_files,
                   "github_api_license_spdx": (api.get("license") or {}).get("spdx_id") if api else None, "github_api_status": st,
                   "github_full_name": api.get("full_name"), "tracked_files": len(tree)}
    # Hugging Face record at the pinned (current) revision
    st, body = get(f"{HF}/api/models/{c['hf_repo']}?blobs=true")
    hf = json.loads(body) if st == 200 else {}
    save(d, "hf_api.json", body if st == 200 else b"{}")
    rev = hf.get("sha")
    sib = {s["rfilename"]: s for s in hf.get("siblings", [])}
    card = hf.get("cardData") or {}
    rec["checkpoint"] = {"hf_repo": c["hf_repo"], "api_status": st, "revision": rev, "author": hf.get("author"), "gated": hf.get("gated"),
                         "last_modified": hf.get("lastModified"), "card_license": card.get("license"), "card_license_name": card.get("license_name"),
                         "card_license_link": card.get("license_link"),
                         "files": {f: {"size": s.get("size"), "lfs_sha256": (s.get("lfs") or {}).get("sha256")} for f, s in sib.items()},
                         "weight_file": c["weight_file"], "weight_file_listed": c["weight_file"] in sib,
                         "weight_lfs_sha256": (sib.get(c["weight_file"], {}).get("lfs") or {}).get("sha256"),
                         "weight_bytes": sib.get(c["weight_file"], {}).get("size"), "weights_downloaded": False}
    texts, fsha = {}, {}
    for key in sorted({s for cl in CLAUSES.get(name, {}).values() for s, _ in cl}):
        kind, path = key.split(":", 1)
        if kind == "git":
            b = blob(clone, path)
        else:
            st2, b = get(f"{HF}/{c['hf_repo']}/resolve/{rev}/{path}") if rev else (None, None)
            b = b if st2 == 200 else None
        if b is not None:
            texts[key] = b.decode("utf-8", errors="replace")
            fsha[key] = save(d, key.replace(":", "__").replace("/", "__"), b)
    rec["source_files_sha256"] = fsha
    ev = {}
    for topic, cl in CLAUSES.get(name, {}).items():
        rows, ok = [], True
        for key, clauses in cl:
            pres = clauses_present(texts[key], clauses) if key in texts else {x: None for x in clauses}
            rows.append({"source": key, "clauses": pres})
            ok = ok and all(v is True for v in pres.values())
        ev[topic] = {"all_present": bool(ok), "evidence": rows}
    rec["clauses"] = ev
    return rec


def gated_probe():
    tok, where = local_token()
    acc = {"repository": GATED_BASE, "credential_present": tok is not None, "credential_source": where, "token_persisted": False, "time": now()}
    url = f"{HF}/{GATED_BASE}/resolve/main/config.json"
    st, hd = head_status(url)
    acc["anonymous_head_status"] = st
    if tok:
        auth = {"Authorization": f"Bearer {tok}"}
        st, body = get(f"{HF}/api/whoami-v2", auth)
        if st == 200:
            w = json.loads(body)
            acc.update({"whoami_ok": True, "account_name": w.get("name"), "token_role": ((w.get("auth") or {}).get("accessToken") or {}).get("role")})
        else:
            acc.update({"whoami_ok": False, "whoami_status": st})
        st, hd = head_status(url, auth)
        acc.update({"authenticated_head_status": st, "x_error_code": hd.get("X-Error-Code"), "x_error_message": hd.get("X-Error-Message")})
        acc["class"] = classify_gated_head(st)
    else:
        acc["class"] = "NOT_GRANTED"
    return acc


def build_table(recs, gate, disk_free_gb):
    """Fact table -> predicates and preferences.  Every value is derived from the gathered records only."""
    def ok(n, topic):
        return recs[n]["clauses"].get(topic, {}).get("all_present")

    def code_lic(n, spdx):
        c = recs[n]["code"]
        return tri(c["head_matches_remote"], bool(c["licence_files_in_tree"]), c["github_api_license_spdx"] == spdx)

    def no_code_lic(n):
        c = recs[n]["code"]
        return c["head_matches_remote"] and not c["licence_files_in_tree"] and c["github_api_status"] == 200 and c["github_api_license_spdx"] is None

    t = {}
    gate_blocked = gate["class"] != "GRANTED"
    for n in ("Kimodo", "ARDY"):
        ck = recs[n]["checkpoint"]
        public = tri(ck["api_status"] == 200, ck["gated"] is False, ck["weight_file_listed"], bool(ck["weight_lfs_sha256"]))
        text_gated = ok(n, "text_gate")
        no_text = ok(n, "no_text_mode")
        # E2: the text-conditioned path needs the gated base (blocked unless the existing credential is GRANTED); the
        # vendor-documented, vendor-benchmarked no-text mode needs no gated artifact.
        e2 = tri(public == TRUE, bool(no_text) or (text_gated is True and not gate_blocked))
        t[n] = {"predicates": {"E1c_code_rights": tri(code_lic(n, "Apache-2.0") == TRUE, ok(n, "code_licence")),
                               "E1w_weight_rights": tri(ok(n, "weight_licence"), ck["card_license_name"] in ("nvidia-open-model-license", "nvidia-open-model-agreement")),
                               "E2_access": e2, "E3_resources": tri(disk_free_gb > 30, (ck["weight_bytes"] or 0) < 4e9, ok(n, "resources") if n == "Kimodo" else True),
                               "E4_conditioning": tri(ok(n, "conditioning"), no_text),
                               "E5_recoverability": tri(ok(n, "output"))},
                "modes": {"text_conditioned": "ACCESS_BLOCKED (gated Llama 3 base; existing credential not granted)" if gate_blocked else "ACCESSIBLE",
                          "no_text_constraints_only": "VERIFIED_BY_SOURCE" if no_text else "NOT_VERIFIED", "declared_mode": "no_text_constraints_only"},
                "preferences": {PREFERENCES[0]: bool(no_text), PREFERENCES[1]: bool(ok(n, "per_frame")) if n == "Kimodo" else not bool(ok(n, "autoregressive_tokenised")),
                                PREFERENCES[2]: True}}
    ml, pj, gx = recs["MotionLab"], recs["ProjFlow"], recs["GEM-X"]
    t["MotionLab"] = {"predicates": {"E1c_code_rights": FALSE if no_code_lic("MotionLab") else UNV,
                                     "E1w_weight_rights": FALSE if (ml["checkpoint"]["card_license"] is None and ok("MotionLab", "restricted_assets")) else UNV,
                                     "E2_access": UNV, "E3_resources": UNV, "E4_conditioning": UNV, "E5_recoverability": UNV},
                      "preferences": {}, "modes": {}}
    t["ProjFlow"] = {"predicates": {"E1c_code_rights": FALSE if no_code_lic("ProjFlow") else UNV,
                                    "E1w_weight_rights": tri(pj["checkpoint"]["card_license"] == "mit", ok("ProjFlow", "weight_licence"), ok("ProjFlow", "weights_hosted_by_author"),
                                                             pj["checkpoint"]["author"] == pj["code"]["github_full_name"].split("/")[0] if pj["code"]["github_full_name"] else None),
                                    "E2_access": tri(pj["checkpoint"]["gated"] is False, pj["checkpoint"]["weight_file_listed"]),
                                    "E3_resources": UNV, "E4_conditioning": tri(ok("ProjFlow", "conditioning")), "E5_recoverability": UNV},
                     "preferences": {}, "modes": {}}
    t["GEM-X"] = {"predicates": {"E1c_code_rights": tri(code_lic("GEM-X", "Apache-2.0") == TRUE, ok("GEM-X", "code_licence")),
                                 "E1w_weight_rights": tri(ok("GEM-X", "weight_licence")), "E2_access": UNV, "E3_resources": UNV,
                                 "E4_conditioning": FALSE if ok("GEM-X", "video_estimator") else UNV, "E5_recoverability": UNV},
                  "preferences": {}, "modes": {}}
    return t


NOTES = {
    "Kimodo": {
        "temporal_conditioning": "frame-indexed constraints over one generated clip: `fullbody` (all 30 SOMA joint positions + smoothed root + heading at the listed frames), end-effector subsets, root 2D; vendor guidance: sparse, fewer than 20 keyframes per constraint type",
        "hard_preservation": "constraints are soft (network) unless the vendor's optional post-processing is enabled; AnimCV-side preservation never depends on the model (only two arm channels at Action 172..199 are imported)",
        "rig_recoverability": "SOMA joint names equal the W129 MotionBricks SOMA joints; W129's closed-form BaseRig<->SOMA Mapping applies; numeric no-model roundtrip is the harness preflight gate",
        "dependencies": "PyTorch (sm_120 build), kimodo package (Apache-2.0), optional C++ `motion_correction` extension for post-processing (fetches pybind11 and Eigen), LLM2Vec + gated Llama 3 8B ONLY for text",
        "rights_caveats": ["NVIDIA Open Model License s2.1: rights terminate if a Guardrail (incl. an authentication mechanism) 'contained in the Model' is bypassed. The gated repository is Meta's Llama 3 base on Hugging Face, not a mechanism contained in the Kimodo checkpoint; the declared mode never requests it. Documentary reading by the agent, flagged for human review.",
                           "s2.3 requires use consistent with NVIDIA's Trustworthy AI terms (page fetched and hashed, not analysed clause by clause)",
                           "training data is proprietary to the licensor (Bones Rigplay 1); the card states provenance for all datasets"],
        "access_judgement": "E2 holds ONLY for the vendor's no-text (constraints-only) mode: the vendor code sets the text feature of an empty prompt to zero with zero length, the encoder returns a (B,1,4096) feature, so a null encoder passed through the documented `text_encoder=` parameter yields the identical denoiser input and no Llama weight is requested. Text-conditioned Kimodo remains ACCESS_BLOCKED. This reading is the agent's and is recorded for human review.",
    },
    "ARDY": {"temporal_conditioning": "autoregressive steps with history poses and future constraints inside a 60-frame horizon; body motion is a latent of a learned tokenizer at 4 frames per token",
             "hard_preservation": "soft; optional post-processing", "rig_recoverability": "SOMA 30 (as Kimodo)",
             "dependencies": "as Kimodo plus the motion tokenizer",
             "access_judgement": "same reading as Kimodo (official `text_encoder=False`, empty prompt zeroed, vendor 'Without text' benchmark rows); NOT exercised in this Batch because at most one candidate is selected"},
    "MotionLab": {"rights": "no licence file in the official repository and no GitHub-detected licence: no affirmative grant for the code; the checkpoints are on a Google Drive folder without terms; the Hugging Face repo `nielsr/motionlab` is a third-party upload without a licence and without `motionflow.ckpt`",
                  "dependencies": "SMPL / SMPL-H body-model files and an AMASS-derived HumanML3D + MotionFix dataset are required by the documented setup; the demo builds its inputs from the dataset loader",
                  "unverified": "runtime path, FPS/window, partial-observation semantics and rig recoverability were not examined once rights failed"},
    "ProjFlow": {"rights": "weights: MIT on the first author's own Hugging Face repository (weight-scoped, rights holder). Code: the official GitHub repository has no licence file and no detected licence, so there is no affirmative grant to run or modify it; this is unresolved, not a finding that use is forbidden. Needs a human decision or an author statement.",
                 "technical": "exact joint-position constraints incl. inpainting and 2D-to-3D lifting on the HumanML3D 22-joint absolute-coordinate representation (positions only); W131's HumanML3D<->BaseRig adapter and 30<->20 fps map would apply; not exercised",
                 "training_data": "HumanML3D, derived from AMASS (non-commercial research terms) - recorded, not judged"},
    "GEM-X": {"incompatibility": "a video motion estimator: its only conditioning is RGB-derived image features and 2D keypoints for the whole body; it has no motion/keyframe input, cannot take the frozen source-bound reference with an UNKNOWN interior, and would re-estimate every joint from pixels (a pose-estimation iteration, which this Batch excludes)",
              "unverified": "detector, SOMA asset and ViTPose terms; resources; rig mapping"},
}


def frozen_evidence():
    """Reuse W137's 55 bound identities and add W137's own artifacts through its evidence manifest."""
    sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
    import w137_unimate_inbetween_harness as H7
    out = H7.frozen_hashes()
    man = H7.W137 / "results" / "evidence_manifest.json"
    want = "8561d239f6a95aa849da92555d35c9d1e38ae79768ac0a997f6bd53c9a5b7a19"
    got = sha256_file(man) if man.is_file() else None
    out["W137_evidence_manifest"] = {"path": str(man), "expected": want, "sha256": got, "ok": got == want}
    files = json.loads(man.read_text(encoding="utf-8"))["files"] if got == want else {}
    for rel in ("run/T_S1.blend", "run/Tc_S1.blend", "run/readback_T.json", "run/dump_T.json", "run/run_result.json", "declare/declaration.json",
                "assess/assessment.json"):
        p = H7.W137 / "results" / rel
        w = (files.get(rel) or {}).get("sha256")
        g = sha256_file(p) if p.is_file() else None
        out["W137:" + rel] = {"path": str(p), "expected": w, "sha256": g, "ok": bool(w and g == w)}
    return out


def stage(a):
    out = Path(a.out_dir)
    if out.exists():
        raise SystemExit(f"refusing existing output directory {out}")
    out.mkdir(parents=True)
    rep = {"schema": "animcv_exp_w138_qualification_v1", "batch": BATCH, "time": now(), "intended_use": INTENDED_USE,
           "note": "documentary evidence gathered by the agent; not legal advice and not a human licensing decision",
           "network_policy": "git ls-remote, GitHub repository API records, Hugging Face model API records and small text/config files at pinned "
                             "revisions; one authenticated whoami and one HEAD (no body, no redirect) of one gated file. No weights."}
    w137 = (ROOT / W137_WORKLOG).read_text(encoding="utf-8")
    later = sorted(p.name for p in (ROOT / "docs").glob("13[89]_*")) + sorted(p.name for p in (ROOT / "docs").glob("14[0-9]_*"))
    rep["live_state"] = {"head": git("rev-parse", "HEAD"), "instruction_blob": git("rev-parse", "HEAD:docs/next_batch.md"),
                         "starting_commit_is_ancestor": subprocess.run(["git", "merge-base", "--is-ancestor", START, "HEAD"], cwd=str(ROOT)).returncode == 0,
                         "origin_head": git("rev-parse", "origin/arch/single_frame_first"), "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
                         "w137_complete": "AUTOMATION_BATCH_STATUS: COMPLETE" in w137 and "BATCH_NUMBER: 137" in w137,
                         "w138_or_later_worklog_absent": not later, "later_worklogs": later,
                         "tracked_tree_clean_except_w138": [ln for ln in git("status", "--porcelain").splitlines() if "w138" not in ln.lower() and "kimodo" not in ln.lower()]}
    ls = rep["live_state"]
    ls["ok"] = bool(ls["instruction_blob"] == BLOB and ls["starting_commit_is_ancestor"] and ls["branch"] == "arch/single_frame_first"
                    and ls["w137_complete"] and ls["w138_or_later_worklog_absent"] and not ls["tracked_tree_clean_except_w138"])
    rep["frozen"] = frozen_evidence()
    rep["frozen_ok"] = all(v["ok"] for v in rep["frozen"].values())
    rep["gated_text_encoder"] = gated_probe()
    recs = {n: gather_candidate(n, c, out) for n, c in CANDIDATES.items()}
    st, body = get("https://www.nvidia.com/en-us/agreements/trustworthy-ai/terms/")
    rep["nvidia_trustworthy_ai_terms"] = {"status": st, "sha256": save(out, "sources/nvidia_trustworthy_ai_terms.html", body) if st == 200 else None,
                                          "reviewed": False}
    import shutil
    disk = shutil.disk_usage(str(W138)).free / 1e9
    table = build_table(recs, rep["gated_text_encoder"], disk)
    sel = select(table)
    for n in recs:
        recs[n]["notes"] = NOTES.get(n, {})
        recs[n]["modes"] = table[n].get("modes", {})
    rep.update({"candidates": recs, "disk_free_gb": round(disk, 1), "selection_table": table, "selection": sel,
                "rule": {"predicates": list(PREDICATES), "preferences": list(PREFERENCES), "stop_of_predicate": STOP_OF, "text": __doc__.split("SELECTION RULE", 1)[1].split("    PY=")[0]},
                "previously_evaluated_not_requalified": {
                    "MotionBricks (W129)": "CASE B (real inference, not better than slerp)", "AnimateCanvas (W130)": "DATA BLOCKED_LICENSE",
                    "CondMDI (W131/W132)": "CASE B / CASE C", "OmniControl (W133)": "DATA BLOCKED_LICENSE", "ARDY text path (W134)": "ACCESS_BLOCKED",
                    "MotionBricks elbow prior (W135)": "CASE B", "UniMate v3 (W136/W137)": "CASE C / CASE C"}})
    stop = None
    if not ls["ok"]:
        stop = "STATE_CHANGED"
    elif not rep["frozen_ok"]:
        stop = "DATA_BLOCKED"
    elif sel["selected"] is None:
        stop = "QUALIFICATION_BLOCKED"
    rep["stop"] = stop
    rep["status"] = "SELECTED:" + sel["selected"] if stop is None else stop
    matrix = {"schema": "animcv_exp_w138_selection_matrix_v1", "batch": BATCH, "time": rep["time"], "starting_commit": START, "instruction_blob": BLOB,
              "model_output_exists": False, "weights_downloaded": False, "intended_use": INTENDED_USE, "rule": rep["rule"],
              "gated_text_encoder": {k: rep["gated_text_encoder"].get(k) for k in ("repository", "class", "authenticated_head_status", "x_error_code", "anonymous_head_status")},
              "matrix": {n: {"code_repo": recs[n]["code_repo"], "pinned_commit": recs[n]["code"]["pinned_commit"], "code_licence_files": recs[n]["code"]["licence_files_in_tree"],
                             "github_license": recs[n]["code"]["github_api_license_spdx"], "hf_repo": recs[n]["checkpoint"]["hf_repo"],
                             "hf_revision": recs[n]["checkpoint"]["revision"], "gated": recs[n]["checkpoint"]["gated"],
                             "card_license": recs[n]["checkpoint"]["card_license_name"] or recs[n]["checkpoint"]["card_license"],
                             "weight_file": recs[n]["checkpoint"]["weight_file"], "weight_lfs_sha256": recs[n]["checkpoint"]["weight_lfs_sha256"],
                             "weight_bytes": recs[n]["checkpoint"]["weight_bytes"], "task": recs[n]["task"], "input_modality": recs[n]["input_modality"],
                             "skeleton": recs[n]["skeleton"], "fps_window": recs[n]["fps_window"], "output": recs[n]["output"],
                             "training_data": recs[n]["training_data"], "clause_topics": {k: v["all_present"] for k, v in recs[n]["clauses"].items()},
                             "predicates": sel["evaluation"][n]["predicates"], "status": sel["evaluation"][n]["status"], "blocking": sel["evaluation"][n]["blocking"],
                             "modes": table[n].get("modes", {}), "preferences": table[n].get("preferences", {}), "notes": NOTES.get(n, {})} for n in recs},
              "selection": {k: sel.get(k) for k in ("selected", "status", "eligible", "preference_tuple", "not_selected_eligible", "reason")},
              "stop": stop}
    save(out, "qualification.json", json.dumps(rep, indent=1, default=str))
    msha = save(out, "selection_matrix.json", json.dumps(matrix, indent=1, default=str))
    print(json.dumps({"status": rep["status"], "live_ok": ls["ok"], "frozen_ok": rep["frozen_ok"], "frozen_bad": [k for k, v in rep["frozen"].items() if not v["ok"]],
                      "gate": rep["gated_text_encoder"]["class"], "evaluation": {n: [e["status"], e["blocking"]] for n, e in sel["evaluation"].items()},
                      "clauses": {n: {k: v["all_present"] for k, v in recs[n]["clauses"].items()} for n in recs},
                      "preference": sel.get("preference_tuple"), "selection_matrix_sha256": msha}, indent=1))
    return 0 if stop is None else 3


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-dir", required=True)
    return stage(ap.parse_args())


if __name__ == "__main__":
    sys.exit(main())
