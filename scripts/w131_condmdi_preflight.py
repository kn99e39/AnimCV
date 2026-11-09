#!/usr/bin/env python3
"""W131 (EXPERIMENTAL, opt-in) preflight: CondMDI checkpoint rights + source identity, frozen S1 / W126 / W129 evidence.

Runs BEFORE any model-weight download.  Stdlib only (AnimCV venv).  It
  * authenticates the repository state, W130 COMPLETE, the mandatory W126-W130 audit, the frozen S1 baseline, the
    W126V-accepted pin variant, the W126 lineage head, the source clip and the W129 A/B/C/Cpin artifacts (byte hashes);
  * pins the original CondMDI code revision, the Motius runtime revision and the Hugging Face checkpoint revision, and
    records the LFS identity (size + SHA-256 OID) of every artifact file WITHOUT downloading weights;
  * reads the model card / configs at the pinned revision, the safetensors HEADER of ``model.safetensors`` (HTTP range
    request, tensor names/shapes only) and the ``args.json`` of the three official CondMDI releases (HTTP range reads of
    the Google-Drive zip central directory; no checkpoint bytes are fetched);
  * applies the declared rights rule below.

Rights rule (declared in this file before any weight request):
  ALLOWED_PENDING_IDENTITY only if ALL hold: (a) the checkpoint card declares an affirmative licence permitting use
  (``license: mit``); (b) its documented provenance is the original CondMDI release, NOT a HY-Motion (territory-excluded)
  derivative; (c) the original CondMDI code is MIT and the Motius runtime retains the CondMDI MIT and OpenAI-CLIP MIT
  licence files; (d) no applicable licence text found contains a territorial exclusion; (e) no restricted body model
  (SMPL/SMPL-X mesh) is required for the declared use.  Training-data terms (HumanML3D <- AMASS: non-commercial research)
  are RECORDED and the declared use is restricted to local non-commercial research evaluation.  Any failing predicate ->
  DATA_BLOCKED_LICENSE (or UNRESOLVED -> stop).  Source identity (which official checkpoint the HF tensors are) is a
  SEPARATE gate resolved after download by tensor comparison against the official release (harness ``identity``).

    PY=C:/Projects/AnimCV/.venv_build_full/Scripts/python.exe
    $PY -X utf8 scripts/w131_condmdi_preflight.py --out-dir C:/Users/dna10/animcv-w131/results/preflight
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import io
import json
import re
import struct
import subprocess
import sys
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
W = Path("C:/Users/dna10/animcv-w131")
W129 = Path("C:/Users/dna10/animcv-w129")
W126 = Path("C:/Users/dna10/animcv-w126/runs")

STARTING_COMMIT = "d2e68f1a3e7c2148ad81d3d8256b6b48c67ed697"
INSTRUCTION_BLOB = "a06baf169f296195561dedcf123de506ffe5853b"

CONDMDI_REPO = "https://github.com/setarehc/diffusion-motion-inbetweening.git"
CONDMDI_REV = "0c58473066ef582b9464a66d5542e0d1b6ae66c8"      # = Motius ATTRIBUTIONS 'Source revision'
MOTIUS_REPO = "https://github.com/ZeyuLing/Motius.git"
MOTIUS_REV = "d04a38c9bc9dab3f0e9e2d843f8a6b21eaa13abc"        # Batch pin
HF_ID = "ZeyuLing/motius-condmdi-humanml3d"
HF_REV = "1e84d9ebf9a2562a2e3aa01926b9fb3606bd23de"            # resolved at preflight, then pinned here
UA = {"User-Agent": "animcv-w131-preflight/1.0"}
META_FILES = ["README.md", "condmdi_config.json", "model_index.json", ".gitattributes"]
# Official CondMDI releases linked from the original README (Google Drive).  Only args.json is read (range requests).
OFFICIAL = {
    "random_joints": "1aP-z1JxSCTcUHhMqqdL2wbwQJUZWHT2j",   # README: "Conditionally trained on randomly sampled frames and joints (CondMDI)"
    "random_frames": "15mYPp2U0VamWfu1SnwCukUUHczY9RPIP",   # README: "Conditionally trained on randomly sampled frames"
    "uncond": "1B0PYpmCXXwV0a5mhkgea_J2pOwhYy-k5",          # README: "Unconditionally (no keyframes) trained"
}

FROZEN = {
    "S1_baseline_w126": (W126 / "S1_replay/baseline/baseline.blend", "0803ab2bf31210fbb729bc81b76931729fdc3e12cce27b9507a5b8caf1d12402"),
    "S1_W126V_accepted_w126": (W126 / "S1_replay/accepted/accepted_03_674ba278e9a0.blend", "7e2ab16511f4cbe111171df9af0bc2e1f7d8c09a57bd5a4feff284d34e908f40"),
    "S1_baseline_w129_copy": (W129 / "inputs/S1_baseline.blend", "0803ab2bf31210fbb729bc81b76931729fdc3e12cce27b9507a5b8caf1d12402"),
    "S1_W126V_w129_copy": (W129 / "inputs/S1_W126V_accepted.blend", "7e2ab16511f4cbe111171df9af0bc2e1f7d8c09a57bd5a4feff284d34e908f40"),
    "W129_B_S1": (W129 / "results/run/S1/B_S1.blend", "8899f43341ac1a7a565229a9551b1f8a79943cc038ad519e5544af89e772a33f"),
    "W129_C_S1": (W129 / "results/run/S1/C_S1.blend", "e04efb3552f9808f8ae89520284b0d94e860a986c1e6ec475ed6b07683f9ef04"),
    "W129_Cpin_S1": (W129 / "results/run/S1/Cpin_S1.blend", "3116d2ea3340d8356beb6adbf5f60f33c026ca92f47f7216137b4a00f6e31b02"),
    "W129_abc_video": (W129 / "results/run/S1/abc_S1_30fps.mp4", "f35420def5d8644e2ac442ad3afa9cc5a55303884460b38fc673f1d182280ea1"),
    "W129_assessment": (W129 / "results/assess/assessment.json", "df6693d22d58cb95f4e130e16e757ab9519c84fe359323560cf530da8b42791c"),
    "W129_evidence_manifest": (W129 / "results/evidence_manifest.json", "b07154ed1b20da67a247fa4d9d5a28222ecb1087cc39d172bbe0af8eca113c02"),
    "S1_clip": (Path("C:/Users/dna10/animcv-w121/data/originals/64498918_Double_Kettlebell_Strict_Press_side_view_.webm"),
                "de8feb9d515daa2bfaeed7167fdcc62028dd3f6450872c4fd81f8fb03dd6359c"),
}
S1_HEAD_STATE = "73dcd3dde3a65ffbd23f36361173ac7a92ee7c4db5e8436e4e85947649a524ab"
S1_VALUE_DIGEST = "25a09802ffa210df56f0e930295468ef4cdb675e2945cb95491c080b5101c8a9"


def now():
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def get(url, rng=None):
    hdr = dict(UA)
    if rng:
        hdr["Range"] = f"bytes={rng[0]}-{rng[1]}"
    with urllib.request.urlopen(urllib.request.Request(url, headers=hdr), timeout=120) as r:
        return r.read(), dict(r.headers)


def save(out, rel, data):
    p = out / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)
    return {"path": str(p), "bytes": len(data), "sha256": sha256_bytes(data)}


def git(*a, cwd=ROOT):
    return subprocess.run(["git", *a], cwd=cwd, capture_output=True, text=True).stdout.strip()


def ps(cmd):
    return subprocess.run(["powershell.exe", "-NoProfile", "-Command", cmd], capture_output=True, text=True).stdout.strip()


class _RangeFile(io.RawIOBase):
    """Seekable read-only view of a remote file via HTTP Range requests (used only for zip central directories)."""

    def __init__(self, url, size):
        self.url, self.size, self.pos, self.fetched = url, size, 0, 0

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else (self.pos + off if whence == 1 else self.size + off)
        return self.pos

    def readinto(self, b):
        if self.pos >= self.size:
            return 0
        end = min(self.size, self.pos + len(b)) - 1
        d, _ = get(self.url, (self.pos, end))
        self.fetched += len(d)
        b[:len(d)] = d
        self.pos += len(d)
        return len(d)


def official_release(name, file_id, out):
    page, _ = get(f"https://drive.google.com/uc?export=download&id={file_id}")
    t = page.decode("utf-8", errors="replace")
    m = re.search(r'<a href="/open\?id=[^"]+">([^<]+)</a>\s*\(([^)]+)\)', t)
    uuid = re.search(r'name="uuid" value="([^"]+)"', t)
    url = (f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t"
           + (f"&uuid={uuid.group(1)}" if uuid else ""))
    _, h = get(url, (0, 0))
    size = int(h.get("Content-Range", h.get("content-range", "/0")).split("/")[-1])
    rf = _RangeFile(url, size)
    z = zipfile.ZipFile(io.BufferedReader(rf, 1 << 16))
    members = [{"name": i.filename, "bytes": i.file_size, "crc32": f"{i.CRC:08x}"} for i in z.infolist()
               if not i.filename.startswith("eval_")]
    args = z.read("args.json")
    rec = save(out / "official", f"{name}_args.json", args)
    a = json.loads(args)
    return {"drive_id": file_id, "zip_name": m.group(1) if m else None, "zip_size_label": m.group(2) if m else None,
            "zip_bytes": size, "members": members, "args_json": rec, "bytes_fetched_by_range": rf.fetched,
            "keyframe_conditioned": a.get("keyframe_conditioned"), "keyframe_selection_scheme": a.get("keyframe_selection_scheme"),
            "save_dir": a.get("save_dir"), "resume_checkpoint": a.get("resume_checkpoint")}


def stage(a):
    out = Path(a.out_dir)
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}")
    out.mkdir(parents=True)
    rep = {"stage": "w131_preflight", "time": now(), "schema": "animcv_exp_w131_preflight_v1"}

    # ---- repository state ----
    w130 = ROOT / "docs/130_WORKLOG_ANIMATECANVAS_SOURCE_CONDITIONED_EDIT_LICENSE_BLOCKED.md"
    audit = ROOT / "docs/AUDIT_W126_W130_2026_11_09.md"
    rep["repo"] = {"head": git("rev-parse", "HEAD"), "instruction_blob": git("rev-parse", "HEAD:docs/next_batch.md"),
                   "w130_complete": "AUTOMATION_BATCH_STATUS: COMPLETE" in w130.read_text(encoding="utf-8"),
                   "audit_present": audit.is_file(), "audit_sha256": sha256_file(audit) if audit.is_file() else None}
    rep["repo"]["blob_ok"] = rep["repo"]["instruction_blob"] == INSTRUCTION_BLOB
    rep["repo"]["starting_commit_is_ancestor"] = subprocess.run(
        ["git", "merge-base", "--is-ancestor", STARTING_COMMIT, "HEAD"], cwd=ROOT).returncode == 0

    # ---- frozen S1 / W126 / W129 evidence (hash-only authentication) ----
    fz = {}
    for k, (p, h) in FROZEN.items():
        got = sha256_file(p) if p.is_file() else None
        fz[k] = {"path": str(p), "sha256": got, "expected": h, "ok": got == h}
    refs = json.loads((W126 / "S1_replay/lineage/refs.json").read_text())
    fz["S1_lineage_head"] = {"head": refs["head"], "op": refs["op"], "value_digest": refs["head_blend"]["value_digest"],
                             "refs_sha256": sha256_file(W126 / "S1_replay/lineage/refs.json"),
                             "ok": refs["head"] == S1_HEAD_STATE and refs["head_blend"]["value_digest"] == S1_VALUE_DIGEST}
    prop = json.loads((W126 / "S1_replay/edits/W126V/proposal.json").read_text())
    fz["S1_pin"] = {"pin_source_frames": prop["pin_frames"], "pin_action_frames": [f - 78 for f in prop["pin_frames"]],
                    "target_m": prop["constraint"]["target"], "tolerance_m": prop["constraint"]["tolerance_m"],
                    "edit_interior_action": [172, 199], "edit_interior_source": [250, 277],
                    "overlap_with_interior": sorted(set(f - 78 for f in prop["pin_frames"]) & set(range(172, 200))),
                    "state": "accepted in W126 then REVERTED: authoritative head has no active pin; pin domain protected"}
    fz["W129_case"] = json.loads((W129 / "results/assess/assessment.json").read_text())["case"]
    rep["frozen_evidence"] = fz
    rep["frozen_evidence_ok"] = all(v.get("ok", True) for v in fz.values() if isinstance(v, dict)) \
        and not fz["S1_pin"]["overlap_with_interior"] and fz["W129_case"] == "B"

    # ---- original CondMDI code + Motius runtime (code-only clones under the scratch root) ----
    src = {}
    for key, repo, rev, local in (("condmdi", CONDMDI_REPO, CONDMDI_REV, W / "ext" / "condmdi"),
                                  ("motius", MOTIUS_REPO, MOTIUS_REV, W / "ext" / "motius")):
        ls = git("ls-remote", repo, "HEAD")
        r = {"repo": repo, "pinned_rev": rev, "remote_head": ls.split()[0] if ls else None}
        if (local / ".git").exists():
            r["local_checkout_head"] = git("rev-parse", "HEAD", cwd=local)
            r["local_head_ok"] = r["local_checkout_head"] == rev
            r["local_worktree_clean"] = git("status", "--porcelain", cwd=local) == ""
        src[key] = r
    cl = W / "ext" / "condmdi"
    src["condmdi"]["license_sha256"] = sha256_file(cl / "LICENSE")
    src["condmdi"]["license_head"] = (cl / "LICENSE").read_text(encoding="utf-8").splitlines()[:4]
    readme = (cl / "README.md").read_text(encoding="utf-8")
    src["condmdi"]["readme_license_section"] = readme[readme.find("## License"):].strip()[:600]
    src["condmdi"]["readme_checkpoint_links"] = re.findall(r"\[(Conditionally[^\]]*|Unconditionally[^\]]*)\]\((https://drive[^)]+)\)", readme)
    ml = W / "ext" / "motius" / "motius" / "licenses" / "condmdi"
    src["motius"]["condmdi_licence_files"] = {p.name: sha256_file(p) for p in sorted(ml.iterdir()) if p.is_file()}
    src["motius"]["condmdi_license_head"] = (ml / "LICENSE").read_text(encoding="utf-8").splitlines()[:2]
    src["motius"]["clip_license_head"] = (ml / "CLIP_LICENSE").read_text(encoding="utf-8").splitlines()[:3]
    src["motius"]["attributions"] = (ml / "ATTRIBUTIONS.md").read_text(encoding="utf-8")
    exporter = (W / "ext" / "motius" / "tools" / "export_condmdi_hf.py").read_text(encoding="utf-8")
    src["motius"]["exporter_forces"] = re.findall(r'"(keyframe_conditioned|keyframe_selection_scheme|zero_keyframe_loss|abs_3d)":\s*([^,\n]+)', exporter)
    rep["sources"] = src

    # ---- Hugging Face checkpoint identity (metadata only; NO weights fetched) ----
    api = json.loads(get(f"https://huggingface.co/api/models/{HF_ID}/revision/{HF_REV}")[0])
    tree = json.loads(get(f"https://huggingface.co/api/models/{HF_ID}/tree/{HF_REV}?recursive=true")[0])
    lfs = {t["path"]: {"bytes": t["size"], "lfs_sha256": t["lfs"]["oid"]} for t in tree if t["type"] == "file" and t.get("lfs")}
    ck = {"id": HF_ID, "pinned_rev": HF_REV, "api_sha": api.get("sha"), "gated": api.get("gated"), "private": api.get("private"),
          "lastModified": api.get("lastModified"), "card_license": (api.get("cardData") or {}).get("license"),
          "tags": api.get("tags"), "lfs_files": lfs, "total_bytes": sum(t["size"] for t in tree if t["type"] == "file"),
          "weights_downloaded": False, "meta": {}}
    for f in META_FILES:
        ck["meta"][f] = save(out / "hf_meta", f, get(f"https://huggingface.co/{HF_ID}/resolve/{HF_REV}/{f}")[0])
    card = (out / "hf_meta" / "README.md").read_text(encoding="utf-8")
    cfg = json.loads((out / "hf_meta" / "condmdi_config.json").read_text(encoding="utf-8"))
    ck["card_frontmatter_license"] = (re.search(r"(?m)^license:\s*(\S+)", card) or [None, None])[1]
    ck["card_provenance_sentence"] = (re.search(r"The Motius release packages[^.]*\.", card.replace("\n", " ")) or [None])[0]
    ck["card_mentions_hymotion"] = bool(re.search(r"(?i)hy-?motion|hunyuan|tencent", card))
    ck["card_native_representation"] = (re.search(r"\| Native representation \| ([^|]+)\|", card) or [None, None])[1]
    ck["card_text_encoder"] = (re.search(r"\| Text encoder \| ([^|]+)\|", card) or [None, None])[1]
    ck["card_default_sampler"] = (re.search(r"\| Default sampler \| ([^|]+)\|", card) or [None, None])[1]
    ck["config_save_dir"] = cfg["config"].get("save_dir")
    ck["config_keyframe_selection_scheme"] = cfg["config"].get("keyframe_selection_scheme")
    ck["config_source_revision"] = cfg.get("source_revision")
    # safetensors header of model.safetensors (tensor names / shapes only)
    url = f"https://huggingface.co/{HF_ID}/resolve/{HF_REV}/model.safetensors"
    n = struct.unpack("<Q", get(url, (0, 7))[0])[0]
    hdr = json.loads(get(url, (8, 8 + n - 1))[0])
    save(out / "hf_meta", "model.safetensors.header.json", json.dumps(hdr, sort_keys=True).encode())
    first = hdr["unet.downs.0.0.blocks.0.block1.0.weight"]["shape"]
    ck["model_header"] = {"n_tensors": len([k for k in hdr if k != "__metadata__"]), "dtypes": sorted({v["dtype"] for k, v in hdr.items() if k != "__metadata__"}),
                          "first_conv_shape": first, "input_channels": first[1],
                          "keyframe_conditioned_network": first[1] == 2 * 263, "output_channels": hdr["unet.final_conv.1.weight"]["shape"][0]}
    rep["checkpoint"] = ck

    # ---- official CondMDI releases (args.json only, via zip central directory range reads) ----
    off = {}
    for name, fid in OFFICIAL.items():
        try:
            off[name] = official_release(name, fid, out)
        except Exception as e:                                     # noqa: BLE001
            off[name] = {"error": f"{type(e).__name__}: {e}"}
    rep["official_releases"] = off
    rep["config_provenance"] = {
        "hf_config_save_dir": ck["config_save_dir"],
        "matches_release_save_dir": [k for k, v in off.items() if v.get("save_dir") == ck["config_save_dir"]],
        "hf_network_is_keyframe_conditioned": ck["model_header"]["keyframe_conditioned_network"],
        "note": ("The Motius exporter overwrites keyframe_conditioned/keyframe_selection_scheme/zero_keyframe_loss/abs_3d, so the "
                 "HF config cannot identify the training scheme; tensor identity is resolved by the harness 'identity' stage.")}

    # ---- training data / body model terms (recorded) ----
    try:
        hml = json.loads(get("https://api.github.com/repos/EricGuo5513/HumanML3D/license")[0])
        hml_lic = {"spdx": (hml.get("license") or {}).get("spdx_id"), "sha": hml.get("sha")}
    except Exception as e:                                         # noqa: BLE001
        hml_lic = {"error": str(e)}
    rep["data_terms"] = {"training_data": "HumanML3D (card datasets: EricGuo5513/HumanML3D)", "humanml3d_repo_license": hml_lic,
                         "underlying_motion": "AMASS (+HumanAct12); AMASS licence = non-commercial scientific research (not re-fetched)",
                         "condmdi_readme_note": src["condmdi"]["readme_license_section"],
                         "declared_use": "local, non-commercial research evaluation on Main; no redistribution of weights or outputs; no dataset, AMASS or SMPL/SMPL-X body model acquired",
                         "smpl_needed": False}

    # ---- execution locale (this machine) ----
    rep["execution_locale"] = {"timezone": ps("(Get-TimeZone).Id"), "home_geo_id": ps("(Get-WinHomeLocation).GeoId"),
                               "culture": ps("(Get-Culture).Name"), "system_locale": ps("(Get-WinSystemLocale).Name"),
                               "note": "GeoId 134 = Korea (Republic of); W130 found the HY-Motion licence excludes South Korea"}

    # ---- declared rights rule ----
    lic_texts = [(cl / "LICENSE").read_text(encoding="utf-8"), (ml / "LICENSE").read_text(encoding="utf-8"),
                 (ml / "CLIP_LICENSE").read_text(encoding="utf-8")]
    rule = {
        "a_card_licence_mit": ck["card_license"] == "mit" and ck["card_frontmatter_license"] == "mit",
        "b_provenance_original_condmdi_not_hymotion": bool(ck["card_provenance_sentence"]) and "official" in ck["card_provenance_sentence"]
                                                      and not ck["card_mentions_hymotion"],
        "c_code_mit_condmdi_and_clip_retained": src["condmdi"]["license_head"][0].strip() == "MIT License"
                                                and src["motius"]["condmdi_license_head"][0].strip() == "MIT License"
                                                and src["motius"]["clip_license_head"][0].strip() == "MIT License"
                                                and src["condmdi"].get("local_head_ok", False) and src["motius"].get("local_head_ok", False),
        "d_no_territorial_exclusion_in_applicable_licences": not any(re.search(r"(?i)territor|south korea|excluding", t) for t in lic_texts + [card]),
        "e_no_restricted_body_model_required": True,
    }
    verdict = "ALLOWED_PENDING_IDENTITY" if all(rule.values()) else "DATA_BLOCKED_LICENSE"
    rep["rights_rule"] = rule
    rep["rights_verdict"] = verdict
    ok_pre = (rep["repo"]["blob_ok"] and rep["repo"]["starting_commit_is_ancestor"] and rep["repo"]["w130_complete"]
              and rep["repo"]["audit_present"] and rep["frozen_evidence_ok"])
    rep["prerequisites_ok"] = bool(ok_pre)
    rep["status"] = verdict if verdict != "ALLOWED_PENDING_IDENTITY" else ("OK_PENDING_IDENTITY" if ok_pre else "DATA_BLOCKED")
    stray = [str(p) for p in W.rglob("*.safetensors")] + [str(p) for p in W.rglob("model*.pt")]
    rep["scratch_weight_files_present"] = stray
    (out / "preflight.json").write_text(json.dumps(rep, indent=1, sort_keys=True), encoding="utf-8")
    print(json.dumps({"status": rep["status"], "rights_rule": rule, "config_provenance": rep["config_provenance"],
                      "frozen_evidence_ok": rep["frozen_evidence_ok"], "sha256": sha256_file(out / "preflight.json")}, indent=1))
    return 0 if rep["status"] == "OK_PENDING_IDENTITY" else 3


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-dir", required=True)
    return stage(ap.parse_args())


if __name__ == "__main__":
    sys.exit(main())
