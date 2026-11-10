#!/usr/bin/env python3
"""W134 (EXPERIMENTAL, opt-in) preflight: ARDY-SOMA-RP-30FPS-Horizon60 rights, identity and gated text-encoder access.

Runs BEFORE any weight download, runtime install, official smoke, prospective declaration or S1 inference.  It
  * authenticates the repository state and the frozen S1 / W126 / W129 / W131 / W132 / W133 evidence (byte hashes only;
    nothing is rebuilt or modified);
  * pins the official ARDY source (Batch pin e706e27...) from a code-only local clone; reads LICENSE, README, the
    text-encoder preset and the constraint/loader code from git blobs at the pinned commit;
  * records the official NVIDIA checkpoint identity from Hugging Face model METADATA only (repository revision, file
    list, LFS SHA-256 as published, card licence fields) plus the small model-card text.  No weight byte is requested;
  * records the runtime text-encoder dependency chain the official loader uses (LLM2Vec PEFT repositories and the
    base model they require) from metadata and the small adapter_config.json files;
  * checks whether ALREADY EXISTING, locally stored Hugging Face credentials already hold granted access to the gated
    base model, using one authenticated whoami call and one authenticated HEAD request (redirects not followed, no
    body downloaded).  The token is never printed, logged or written; only the account name and token role are kept.
    No access is requested, no terms are accepted, no other credential is used.

Access rule (declared here, before the run; the Batch: "If absent, STOP ACCESS_BLOCKED and record the exact missing
authorization"):
  ACCESS_OK only if ALL hold:
    (a) the official loader's default text encoder resolves (directly or via adapter_config.json) to the gated base
        model meta-llama/Meta-Llama-3-8B-Instruct, or does not need it (then (b)-(d) are vacuous);
    (b) an existing local Hugging Face credential is present (HF_TOKEN env or the user's own ~/.cache/huggingface/token);
    (c) that credential authenticates (whoami succeeds) - the account name is recorded for human review;
    (d) the authenticated HEAD of a file in the gated base repository returns 200 or a redirect (access granted);
        401/403 (or any other status) means access is NOT granted to that account.
  Anything else -> ACCESS_BLOCKED (DATA BLOCKED).  There is no fallback encoder, no mirror and no API substitute.

Checkpoint-rights record (secondary to the access gate): the NVIDIA checkpoint card's licence fields and whether the
card text names the NVIDIA Open Model Agreement; the ARDY code licence (Apache-2.0) is recorded separately and is never
treated as a weight licence.

    PY=C:/Projects/AnimCV/.venv_build_full/Scripts/python.exe
    $PY -X utf8 scripts/w134_ardy_access_preflight.py --out-dir C:/Users/dna10/animcv-w134/results/preflight
Exit codes: 0 ACCESS_OK (preflight only - nothing else is authorised by this script), 3 ACCESS_BLOCKED,
4 prerequisite/identity failure.
"""
from __future__ import annotations

import argparse
import datetime as dt
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
W = Path("C:/Users/dna10/animcv-w134")
W129 = Path("C:/Users/dna10/animcv-w129")
W131 = Path("C:/Users/dna10/animcv-w131")
W132 = Path("C:/Users/dna10/animcv-w132")
W133 = Path("C:/Users/dna10/animcv-w133")
W126 = Path("C:/Users/dna10/animcv-w126/runs")

STARTING_COMMIT = "eab3962d3389ea779ad2332d06f3bac9d542e500"
INSTRUCTION_BLOB = "ec2f2c167e7275cf76909143c18f71cdd214ea18"

ARDY_REPO = "https://github.com/nv-tlabs/ardy.git"
ARDY_REV = "e706e27acc88070918308a10922b8f7174d0825e"           # Batch pin
CKPT_ID = "nvidia/ARDY-SOMA-RP-30FPS-Horizon60"
BASE_LLM = "meta-llama/Meta-Llama-3-8B-Instruct"
GATED_PROBE_FILE = "config.json"
HF = "https://huggingface.co"
UA = {"User-Agent": "animcv-w134-preflight/1.0"}
WEIGHT_SUFFIXES = (".pt", ".pth", ".safetensors", ".ckpt", ".bin", ".onnx", ".engine", ".plan")

FROZEN = {
    "S1_baseline_w126": (W126 / "S1_replay/baseline/baseline.blend", "0803ab2bf31210fbb729bc81b76931729fdc3e12cce27b9507a5b8caf1d12402"),
    "S1_W126V_accepted_w126": (W126 / "S1_replay/accepted/accepted_03_674ba278e9a0.blend", "7e2ab16511f4cbe111171df9af0bc2e1f7d8c09a57bd5a4feff284d34e908f40"),
    "S1_baseline_w129_copy": (W129 / "inputs/S1_baseline.blend", "0803ab2bf31210fbb729bc81b76931729fdc3e12cce27b9507a5b8caf1d12402"),
    "S1_W126V_w129_copy": (W129 / "inputs/S1_W126V_accepted.blend", "7e2ab16511f4cbe111171df9af0bc2e1f7d8c09a57bd5a4feff284d34e908f40"),
    "W129_B_S1_slerp": (W129 / "results/run/S1/B_S1.blend", "8899f43341ac1a7a565229a9551b1f8a79943cc038ad519e5544af89e772a33f"),
    "W129_C_S1_motionbricks": (W129 / "results/run/S1/C_S1.blend", "e04efb3552f9808f8ae89520284b0d94e860a986c1e6ec475ed6b07683f9ef04"),
    "W129_assessment": (W129 / "results/assess/assessment.json", "df6693d22d58cb95f4e130e16e757ab9519c84fe359323560cf530da8b42791c"),
    "W131_C_S1_condmdi": (W131 / "results/run/S1/C_S1.blend", "8f49a93a7c012352696ebac48cca109d0b0176c3087dba09a8af708aece1da60"),
    "W131_assessment": (W131 / "results/assess/assessment.json", "9633a3621065058f3cd0d23af1a0f6fed2baf507b578c5c00c510ed6307404ac"),
    "W131_evidence_manifest": (W131 / "results/evidence_manifest.json", "98505f804945b6a772dd2e63aa469ae5e207a2a9d17101dde6f2775ce73d7cc0"),
    "W132_assessment": (W132 / "results/assess/assessment.json", "35919277d4c1a166d1ff8b844cf970b226de35d92615765be7ad239e91eb8d93"),
    "W132_evidence_manifest": (W132 / "results/evidence_manifest.json", "4da3c13b42a35fc562e09259883b44854343743efdf8f7f921cd765155edf06c"),
    "W133_preflight": (W133 / "results/preflight/preflight.json", "93e03bc835dc601040ba38e872e09745a72b028be443d3b354d2664d4b810436"),
    "W133_evidence_manifest": (W133 / "results/evidence_manifest.json", "39d7ae23d9b415294b3eed47d407361c67d967447c1fdd01a350b30cc21eb915"),
    "S1_clip": (Path("C:/Users/dna10/animcv-w121/data/originals/64498918_Double_Kettlebell_Strict_Press_side_view_.webm"),
                "de8feb9d515daa2bfaeed7167fdcc62028dd3f6450872c4fd81f8fb03dd6359c"),
}
EXPECTED_CASES = {"W129_assessment": "B", "W131_assessment": "B", "W132_assessment": "C"}
S1_HEAD_STATE = "73dcd3dde3a65ffbd23f36361173ac7a92ee7c4db5e8436e4e85947649a524ab"
S1_VALUE_DIGEST = "25a09802ffa210df56f0e930295468ef4cdb675e2945cb95491c080b5101c8a9"


# ----------------------------------------------------------------------------------------------- pure helpers (tested)
def classify_gated_head(status):
    """HTTP status of an authenticated HEAD on a gated-repo file -> access class (redirects are not followed)."""
    if status == 200 or status in (301, 302, 303, 307, 308):
        return "GRANTED"
    if status in (401, 403):
        return "NOT_GRANTED"
    if status is None:
        return "NOT_CHECKED"
    return "UNKNOWN_STATUS"


def text_encoder_preset(load_model_src: str):
    """Extract the default LLM2Vec preset repository ids from ardy/model/load_model.py source text."""
    d = re.search(r'DEFAULT_TEXT_ENCODER\s*=\s*"([^"]+)"', load_model_src)
    b = re.search(r'"base_model_name_or_path":\s*"([^"]+)"', load_model_src)
    p = re.search(r'"peft_model_name_or_path":\s*"([^"]+)"', load_model_src)
    return {"default": d.group(1) if d else None, "base": b.group(1) if b else None, "peft": p.group(1) if p else None}


def evaluate_access_rule(f: dict):
    needs = bool(f.get("requires_gated_base"))
    rule = {
        "a_default_encoder_resolves_to_gated_base": needs,
        "b_existing_local_credential_present": bool(f.get("credential_present")),
        "c_credential_authenticates": bool(f.get("whoami_ok")),
        "d_gated_base_access_granted": f.get("gated_head_class") == "GRANTED",
    }
    if not needs:
        return rule, "ACCESS_OK", []
    missing = []
    if not rule["b_existing_local_credential_present"]:
        missing.append("no existing local Hugging Face credential")
    elif not rule["c_credential_authenticates"]:
        missing.append("existing local credential does not authenticate")
    elif not rule["d_gated_base_access_granted"]:
        missing.append(f"account has no granted access to {BASE_LLM} (HEAD class {f.get('gated_head_class')})")
    return rule, ("ACCESS_OK" if not missing else "ACCESS_BLOCKED"), missing


# ------------------------------------------------------------------------------------------------------- I/O helpers
def now():
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *a, **k):
        return None


def get(url, headers=None):
    hdr = dict(UA, **(headers or {}))
    with urllib.request.urlopen(urllib.request.Request(url, headers=hdr), timeout=60) as r:
        return r.read(), r.status


def head_status(url, headers=None):
    opener = urllib.request.build_opener(_NoRedirect)
    req = urllib.request.Request(url, headers=dict(UA, **(headers or {})), method="HEAD")
    try:
        with opener.open(req, timeout=60) as r:
            return r.status, {k: v for k, v in r.headers.items() if k.lower().startswith("x-error")}
    except urllib.error.HTTPError as e:
        return e.code, {k: v for k, v in e.headers.items() if k.lower().startswith("x-error")}


def save(out, rel, data):
    p = out / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)
    return {"path": str(p), "bytes": len(data), "sha256": sha256_bytes(data)}


def git(*a, cwd=ROOT):
    return subprocess.run(["git", *a], cwd=cwd, capture_output=True, text=True).stdout.strip()


def blob(repo, rev, path):
    return subprocess.run(["git", "show", f"{rev}:{path}"], cwd=repo, capture_output=True, check=True).stdout


def model_meta(repo_id, headers=None):
    api, _ = get(f"{HF}/api/models/{repo_id}?blobs=true", headers)
    a = json.loads(api)
    card = a.get("cardData") or {}
    return {"id": a.get("id"), "sha": a.get("sha"), "author": a.get("author"), "gated": a.get("gated"),
            "private": a.get("private"), "lastModified": a.get("lastModified"),
            "card_license": card.get("license"), "card_license_name": card.get("license_name"),
            "card_license_link": card.get("license_link"), "base_model": card.get("base_model"),
            "files": {s["rfilename"]: {"bytes": s.get("size"), "lfs_sha256": (s.get("lfs") or {}).get("sha256"),
                                       "git_blob": s.get("blobId")} for s in a.get("siblings", [])}}


def local_token():
    """Existing credential only (env or the user's own HF token file).  Never printed or persisted."""
    if os.environ.get("HF_TOKEN"):
        return os.environ["HF_TOKEN"].strip(), "env:HF_TOKEN"
    p = Path.home() / ".cache/huggingface/token"
    if p.is_file():
        t = p.read_text(encoding="utf-8").strip()
        if t:
            return t, f"file:{p} (mtime {dt.datetime.fromtimestamp(p.stat().st_mtime).isoformat(timespec='seconds')})"
    return None, None


# ------------------------------------------------------------------------------------------------------------- stage
def stage(a):
    out = Path(a.out_dir)
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}")
    out.mkdir(parents=True)
    rep = {"stage": "w134_access_preflight", "time": now(), "schema": "animcv_exp_w134_preflight_v1",
           "network_policy": "Hugging Face model metadata + small text/JSON files; one authenticated whoami; "
                             "HEAD-only probe of one gated file (no body, no redirect). No weights."}

    # ---- repository state ----
    w133 = (ROOT / "docs/133_WORKLOG_OMNICONTROL_SPARSE_SPATIAL_GUIDANCE_LICENSE_BLOCKED.md").read_text(encoding="utf-8")
    rep["repo"] = {"head": git("rev-parse", "HEAD"), "instruction_blob": git("rev-parse", "HEAD:docs/next_batch.md"),
                   "w133_complete": "AUTOMATION_BATCH_STATUS: COMPLETE" in w133}
    rep["repo"]["blob_ok"] = rep["repo"]["instruction_blob"] == INSTRUCTION_BLOB
    rep["repo"]["starting_commit_is_ancestor"] = subprocess.run(
        ["git", "merge-base", "--is-ancestor", STARTING_COMMIT, "HEAD"], cwd=ROOT).returncode == 0

    # ---- frozen evidence (hash-only) ----
    fz = {}
    for k, (p, h) in FROZEN.items():
        got = sha256_file(p) if p.is_file() else None
        fz[k] = {"path": str(p), "sha256": got, "expected": h, "ok": got == h}
        if k in EXPECTED_CASES and got == h:
            fz[k]["case"] = json.loads(p.read_text(encoding="utf-8")).get("case")
            fz[k]["ok"] = fz[k]["case"] == EXPECTED_CASES[k]
    refs = json.loads((W126 / "S1_replay/lineage/refs.json").read_text())
    fz["S1_lineage_head"] = {"head": refs["head"], "op": refs["op"], "value_digest": refs["head_blend"]["value_digest"],
                             "ok": refs["head"] == S1_HEAD_STATE and refs["head_blend"]["value_digest"] == S1_VALUE_DIGEST}
    prop = json.loads((W126 / "S1_replay/edits/W126V/proposal.json").read_text())
    fz["S1_pin"] = {"pin_source_frames": prop["pin_frames"], "pin_action_frames": [f - 78 for f in prop["pin_frames"]],
                    "target_m": prop["constraint"]["target"], "tolerance_m": prop["constraint"]["tolerance_m"],
                    "edit_interior_action": [172, 199], "edit_interior_source": [250, 277],
                    "declared_anchor_action_frames": [170, 171, 200, 201],
                    "overlap_with_interior": sorted(set(f - 78 for f in prop["pin_frames"]) & set(range(172, 200)))}
    rep["frozen_evidence"] = fz
    rep["frozen_evidence_ok"] = all(v.get("ok", True) for v in fz.values()) and not fz["S1_pin"]["overlap_with_interior"]

    # ---- official ARDY source (code-only clone, git blobs at the pin) ----
    local = W / "ext" / "ardy"
    ls = git("ls-remote", ARDY_REPO, "HEAD")
    src = {"repo": ARDY_REPO, "pinned_rev": ARDY_REV, "remote_head": ls.split()[0] if ls else None,
           "local_checkout_head": git("rev-parse", "HEAD", cwd=local),
           "local_worktree_clean": git("status", "--porcelain", cwd=local) == "",
           "pinned_commit_date": git("show", "-s", "--format=%cI", ARDY_REV, cwd=local),
           "files_read_from": f"git blobs at {ARDY_REV}"}
    lic = blob(local, ARDY_REV, "LICENSE")
    readme = blob(local, ARDY_REV, "README.md").decode("utf-8")
    lm = blob(local, ARDY_REV, "ardy/model/load_model.py").decode("utf-8")
    reg = blob(local, ARDY_REV, "ardy/model/registry.py").decode("utf-8")
    cons = blob(local, ARDY_REV, "ardy/constraints.py").decode("utf-8")
    attr = blob(local, ARDY_REV, "ATTRIBUTIONS.MD").decode("utf-8")
    src["license_sha256"] = sha256_bytes(lic)
    src["license_is_apache2"] = b"Apache License" in lic and b"Version 2.0, January 2004" in lic
    i = readme.find("## License")
    src["readme_license_section"] = readme[i:i + 420].strip() if i >= 0 else None
    i = readme.find("### Set up Hugging Face token for text encoder")
    src["readme_text_encoder_access_section"] = readme[i:i + 700].strip() if i >= 0 else None
    src["readme_soma30_row"] = next((ln for ln in readme.splitlines() if ln.startswith("| **ARDY-SOMA-RP-30FPS-Horizon60**")), None)
    src["text_encoder_preset"] = text_encoder_preset(lm)
    src["loader_uses_snapshot_download"] = "snapshot_download(repo_id=repo_id)" in lm
    src["registry_soma60"] = '60: "ARDY-SOMA-RP-30FPS-Horizon60"' in reg
    src["endeffector_adds_root_and_heading"] = all(s in cons for s in ("class EndEffectorConstraintSet", "root_2d"))
    src["attributions_llm2vec_mit"] = "LLM2Vec MIT License" in attr
    for f in ("README.md", "LICENSE", "ardy/model/load_model.py", "ardy/constraints.py", "ATTRIBUTIONS.MD"):
        save(out, f"ardy_src/{f.replace('/', '__')}", blob(local, ARDY_REV, f))
    rep["ardy_source"] = src

    # ---- NVIDIA checkpoint identity (metadata only) ----
    ck = model_meta(CKPT_ID)
    ck["weights_downloaded"] = False
    card, _ = get(f"{HF}/{CKPT_ID}/resolve/{ck['sha']}/README.md")
    ck["model_card"] = save(out, "checkpoint/README.md", card)
    ct = card.decode("utf-8", "replace")
    ck["card_names_nvidia_open_model_agreement"] = bool(re.search(r"(?i)NVIDIA Open Model", ct))
    ck["card_license_lines"] = [ln.strip() for ln in ct.splitlines() if re.search(r"(?i)licen[cs]e|agreement|geograph", ln)][:20]
    ck["card_mentions_30fps"] = bool(re.search(r"(?i)30\s*(fps|frames per second)", ct))
    ck["card_mentions_llama"] = bool(re.search(r"(?i)llama", ct))
    ck["weight_files"] = {k: v for k, v in ck["files"].items() if k.endswith(WEIGHT_SUFFIXES)}
    for f in [k for k in ck["files"] if k.endswith((".yaml", ".json")) and (ck["files"][k]["bytes"] or 0) < 200_000]:
        b, _ = get(f"{HF}/{CKPT_ID}/resolve/{ck['sha']}/{f}")
        ck.setdefault("config_files", {})[f] = save(out, f"checkpoint/{f.replace('/', '__')}", b)
    rep["checkpoint"] = ck

    # ---- text-encoder dependency chain (metadata + adapter_config.json) ----
    te = {"preset": src["text_encoder_preset"], "repos": {}}
    for rid in (src["text_encoder_preset"]["base"], src["text_encoder_preset"]["peft"]):
        m = model_meta(rid)
        if "adapter_config.json" in m["files"]:
            b, _ = get(f"{HF}/{rid}/resolve/{m['sha']}/adapter_config.json")
            m["adapter_config"] = save(out, f"text_encoder/{rid.replace('/', '__')}__adapter_config.json", b)
            m["adapter_base_model_name_or_path"] = json.loads(b).get("base_model_name_or_path")
        if "config.json" in m["files"]:
            b, _ = get(f"{HF}/{rid}/resolve/{m['sha']}/config.json")
            m["config_name_or_path"] = json.loads(b).get("_name_or_path")
        m["has_full_base_weights"] = any(k.startswith("model-") and k.endswith(".safetensors") for k in m["files"])
        te["repos"][rid] = {k: v for k, v in m.items() if k != "files"} | {"file_names": sorted(m["files"])}
    base = model_meta(BASE_LLM)
    te["gated_base"] = {k: base[k] for k in ("id", "sha", "gated", "card_license", "card_license_name", "lastModified")}
    refs_ = {v.get("adapter_base_model_name_or_path") for v in te["repos"].values()} | \
            {v.get("config_name_or_path") for v in te["repos"].values()}
    te["requires_gated_base"] = BASE_LLM in refs_
    te["local_hf_cache_has_base_or_llm2vec"] = sorted(
        p.name for p in (Path.home() / ".cache/huggingface/hub").glob("models--*")
        if "llama" in p.name.lower() or "llm2vec" in p.name.lower())
    rep["text_encoder"] = te

    # ---- existing credential and gated access (token never persisted) ----
    tok, where = local_token()
    acc = {"credential_present": tok is not None, "credential_source": where, "token_persisted": False}
    if tok:
        auth = {"Authorization": f"Bearer {tok}"}
        try:
            w, _ = get(f"{HF}/api/whoami-v2", auth)
            wj = json.loads(w)
            acc.update({"whoami_ok": True, "account_name": wj.get("name"), "account_type": wj.get("type"),
                        "token_role": ((wj.get("auth") or {}).get("accessToken") or {}).get("role")})
        except urllib.error.HTTPError as e:
            acc.update({"whoami_ok": False, "whoami_status": e.code})
        st, err = head_status(f"{HF}/{BASE_LLM}/resolve/main/{GATED_PROBE_FILE}", auth)
        acc.update({"gated_head_status": st, "gated_head_error_headers": err})
        del tok, auth
    else:
        acc.update({"whoami_ok": False, "gated_head_status": None})
    st0, err0 = head_status(f"{HF}/{BASE_LLM}/resolve/main/{GATED_PROBE_FILE}")
    acc.update({"anonymous_head_status": st0, "anonymous_head_error_headers": err0})
    acc["gated_head_class"] = classify_gated_head(acc.get("gated_head_status"))
    rep["access"] = acc

    facts = {"requires_gated_base": te["requires_gated_base"], "credential_present": acc["credential_present"],
             "whoami_ok": acc.get("whoami_ok"), "gated_head_class": acc["gated_head_class"]}
    rule, verdict, missing = evaluate_access_rule(facts)
    rep["access_facts"], rep["access_rule"], rep["access_verdict"], rep["missing_authorization"] = facts, rule, verdict, missing

    ok_pre = (rep["repo"]["blob_ok"] and rep["repo"]["starting_commit_is_ancestor"] and rep["repo"]["w133_complete"]
              and rep["frozen_evidence_ok"] and src["local_checkout_head"] == ARDY_REV and src["local_worktree_clean"]
              and src["license_is_apache2"] and src["registry_soma60"])
    rep["prerequisites_other_than_access_ok"] = bool(ok_pre)
    rep["status"] = ("ACCESS_BLOCKED" if verdict == "ACCESS_BLOCKED" else
                     "PREFLIGHT_OK_ACCESS" if ok_pre else "PREREQUISITE_FAILED")
    rep["scratch_weight_files_present"] = [str(p) for p in W.rglob("*") if p.is_file() and p.suffix in WEIGHT_SUFFIXES
                                           and ".git" not in p.parts]
    (out / "preflight.json").write_text(json.dumps(rep, indent=1, sort_keys=True), encoding="utf-8")
    print(json.dumps({"status": rep["status"], "access_rule": rule, "missing": missing,
                      "prerequisites_other_than_access_ok": rep["prerequisites_other_than_access_ok"],
                      "sha256": sha256_file(out / "preflight.json")}, indent=1))
    return {"ACCESS_BLOCKED": 3, "PREFLIGHT_OK_ACCESS": 0}.get(rep["status"], 4)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-dir", required=True)
    return stage(ap.parse_args())


if __name__ == "__main__":
    sys.exit(main())
