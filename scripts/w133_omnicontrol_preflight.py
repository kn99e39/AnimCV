#!/usr/bin/env python3
"""W133 (EXPERIMENTAL, opt-in) preflight: OmniControl HumanML3D checkpoint rights, identity and runtime assets.

Runs BEFORE any weight download, any adapter/mapping work and any inference.  It
  * authenticates the repository state and the frozen S1 / W126 / W129 / W131 / W132 evidence (byte hashes only;
    nothing is rebuilt or modified);
  * pins the official OmniControl source revision (Batch pin 76336a6e...) and records its code licence and the README's
    licence and weight-download sections from a code-only local clone;
  * records the identity of the OFFICIAL checkpoint archive (Google Drive id in the official README) WITHOUT
    downloading weights: total size from an HTTP range response, then only the ZIP end-of-central-directory record and
    the central directory (the member list, sizes, CRC-32 and DOS timestamps).  No compressed weight byte is requested;
  * looks for weight terms where the rights holders could have stated them: the archive members, the README, the
    official project page and the official issue tracker; and records the fine-tune parent (MDM) README licence;
  * records - but never uses as a weight source - the third-party Motius Hugging Face re-export and its licence label;
  * inventories, by static reading of the official code, which external assets the official sampling path needs.

Terms rule (declared in this file; the Batch forbids inferring weight rights from the MIT code licence):
  ALLOWED only if a rights holder of the OmniControl HumanML3D checkpoint affirmatively licenses or permits the WEIGHTS
  (not only the code) for local non-commercial research evaluation at the execution location.  DATA_BLOCKED_LICENSE if
  every predicate holds:
    (a) the official archive contains no licence / terms / readme member (weights only);
    (b) the official README's only licence grant is scoped to the code ("This code is distributed under ...") and it
        separately defers datasets and dependencies to their own licences;
    (c) neither the official project page nor the official issue tracker states terms for the weights;
    (d) the fine-tune parent (MDM, whose checkpoint OmniControl training resumes from) also scopes its grant to code;
    (e) any weight-scoped "MIT" label that exists comes from a non-rights-holder mirror whose LICENSE is the code
        licence copied byte-for-byte (a mirror is not an authorised substitute, and copying the code licence onto the
        weights is exactly the inference the Batch forbids).
  Any predicate false or un-evidenced -> UNRESOLVED_TERMS_STOP (also a stop: no weight use without positive terms).

Runtime-asset record (secondary; reported, not a substitute for the terms gate):
  official `sample.generate` (r1) builds `Rotation2xyz -> SMPL()` inside the model constructor, i.e. needs SMPL body-model
  files; (r2) loads the HumanML3D test split in hml_mode='train', i.e. needs the AMASS-derived motion features; (r3) the
  official SMPL download script fetches a third-party Google-Drive copy, while the SMPL licence has a "No Distribution"
  clause.  These are recorded as RESTRICTED_ASSETS_REQUIRED_BY_OFFICIAL_PATH; no patch is attempted while (a)-(e) block.

    PY=C:/Projects/AnimCV/.venv_build_full/Scripts/python.exe
    $PY -X utf8 scripts/w133_omnicontrol_preflight.py --out-dir C:/Users/dna10/animcv-w133/results/preflight
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import html
import json
import re
import struct
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
W = Path("C:/Users/dna10/animcv-w133")
W129 = Path("C:/Users/dna10/animcv-w129")
W131 = Path("C:/Users/dna10/animcv-w131")
W132 = Path("C:/Users/dna10/animcv-w132")
W126 = Path("C:/Users/dna10/animcv-w126/runs")

STARTING_COMMIT = "2ee8c9bf900372cf9ea73bfb4678c74f0fc5cf4a"
INSTRUCTION_BLOB = "20d8def178ad30964393493df855c7aab5be4a6b"

OC_REPO = "https://github.com/neu-3dvi/OmniControl.git"        # GitHub API full_name; neu-vi/OmniControl redirects
OC_REPO_LEGACY = "https://github.com/neu-vi/OmniControl.git"
OC_REV = "76336a6e78f074f3ba7059643414b91a1eced9f2"             # Batch pin (main, 2024-03-25)
OC_REPO_ID = 704213713                                         # GitHub repository id
GD_ID = "1oTkBtArc3xjqkYD6Id7LksrTOn3e1Zud"                     # official README "model_humanml" link
GD_URL = f"https://drive.usercontent.google.com/download?id={GD_ID}&export=download&confirm=t"
PROJECT_PAGE = "https://neu-3dvi.github.io/omnicontrol/"
MDM_README = "https://raw.githubusercontent.com/GuyTevet/motion-diffusion-model/main/README.md"
MIRROR_ID = "ZeyuLing/motius-omnicontrol-humanml3d"
MIRROR_REV = "9f6ca3e28183927b5000d34d54961ecdf7bbd395"
HUMANML3D_API = "https://api.github.com/repos/EricGuo5513/HumanML3D"
TERMS_PAGES = {"AMASS": "https://amass.is.tue.mpg.de/license.html", "SMPL": "https://smpl.is.tue.mpg.de/modellicense.html"}
UA = {"User-Agent": "animcv-w133-preflight/1.0"}
TERMS_MEMBER_RE = re.compile(r"(?i)(licen[cs]e|copying|terms|readme|notice|model[_-]?card|\.md$|\.txt$)")

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
    "S1_clip": (Path("C:/Users/dna10/animcv-w121/data/originals/64498918_Double_Kettlebell_Strict_Press_side_view_.webm"),
                "de8feb9d515daa2bfaeed7167fdcc62028dd3f6450872c4fd81f8fb03dd6359c"),
}
EXPECTED_CASES = {"W129_assessment": "B", "W131_assessment": "B", "W132_assessment": "C"}
S1_HEAD_STATE = "73dcd3dde3a65ffbd23f36361173ac7a92ee7c4db5e8436e4e85947649a524ab"
S1_VALUE_DIGEST = "25a09802ffa210df56f0e930295468ef4cdb675e2945cb95491c080b5101c8a9"


# ----------------------------------------------------------------------------------------------- pure helpers (tested)
def dos_datetime(d, t):
    return f"{1980 + (d >> 9):04d}-{(d >> 5) & 15:02d}-{d & 31:02d}T{t >> 11:02d}:{(t >> 5) & 63:02d}:{(t & 31) * 2:02d}"


def parse_eocd(tail: bytes):
    """Parse a ZIP end-of-central-directory record found in `tail` (the last bytes of the archive)."""
    i = tail.rfind(b"PK\x05\x06")
    if i < 0 or len(tail) - i < 22:
        raise ValueError("EOCD_NOT_FOUND")
    _, disk, cd_disk, n_disk, n, cd_size, cd_off, clen = struct.unpack("<IHHHHIIH", tail[i:i + 22])
    if disk or cd_disk or n_disk != n:
        raise ValueError("MULTI_DISK_ZIP")
    if cd_off == 0xFFFFFFFF or n == 0xFFFF:
        raise ValueError("ZIP64_NOT_SUPPORTED")
    return {"entries": n, "cd_size": cd_size, "cd_offset": cd_off, "comment_len": clen,
            "comment": tail[i + 22:i + 22 + clen].decode("utf-8", "replace")}


def parse_central_directory(cd: bytes, n: int):
    out, p = [], 0
    for _ in range(n):
        if cd[p:p + 4] != b"PK\x01\x02":
            raise ValueError("BAD_CENTRAL_DIRECTORY")
        (_, _vm, _vn, flags, method, mtime, mdate, crc, csize, usize, fnl, exl, cml, _ds, _ia, _ea, lho) = \
            struct.unpack("<IHHHHHHIIIHHHHHII", cd[p:p + 46])
        name = cd[p + 46:p + 46 + fnl].decode("utf-8", "replace")
        comment = cd[p + 46 + fnl + exl:p + 46 + fnl + exl + cml].decode("utf-8", "replace")
        out.append({"name": name, "method": method, "flags": flags, "crc32": f"{crc:08x}", "compressed_bytes": csize,
                    "uncompressed_bytes": usize, "local_header_offset": lho, "dos_mtime": dos_datetime(mdate, mtime),
                    "comment": comment, "is_dir": name.endswith("/")})
        p += 46 + fnl + exl + cml
    return out


def terms_members(members):
    return [m["name"] for m in members if not m["is_dir"] and TERMS_MEMBER_RE.search(m["name"].rsplit("/", 1)[-1])]


def code_scoped_licence_section(readme: str):
    """Return (section_text, scoped_to_code, defers_datasets) for the README '## License' section."""
    i = readme.find("## License")
    sec = readme[i:i + 800] if i >= 0 else ""
    scoped = bool(re.search(r"This code is distributed under", sec))
    defers = bool(re.search(r"datasets that each have their own respective licenses", sec))
    return sec, scoped, defers


def evaluate_terms_rule(f: dict):
    rule = {
        "a_official_archive_has_no_terms_member": f["archive_parsed"] and not f["archive_terms_members"],
        "b_official_readme_grant_scoped_to_code": f["readme_scoped_to_code"] and f["readme_defers_datasets"]
                                                  and not f["readme_mentions_weight_licence"],
        "c_no_official_weight_terms_page_or_issue": f["project_page_fetched"] and not f["project_page_weight_terms"]
                                                    and f["issues_searched"] and not f["issues_with_licence_terms"],
        "d_parent_mdm_grant_scoped_to_code": f["mdm_scoped_to_code"],
        "e_weight_label_only_from_mirror_copying_code_licence": (not f["mirror_exists"]) or (
            f["mirror_publisher_is_not_rights_holder"] and f["mirror_licence_equals_code_licence"]),
    }
    if all(rule.values()):
        return rule, "DATA_BLOCKED_LICENSE"
    return rule, "UNRESOLVED_TERMS_STOP"


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


def get(url, rng=None):
    hdr = dict(UA)
    if rng is not None:
        hdr["Range"] = f"bytes={rng[0]}-{rng[1]}"
    with urllib.request.urlopen(urllib.request.Request(url, headers=hdr), timeout=90) as r:
        return r.read(), dict(r.headers), r.status


def save(out, rel, data):
    p = out / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)
    return {"path": str(p), "bytes": len(data), "sha256": sha256_bytes(data)}


def git(*a, cwd=ROOT):
    return subprocess.run(["git", *a], cwd=cwd, capture_output=True, text=True).stdout.strip()


def ps(cmd):
    return subprocess.run(["powershell.exe", "-NoProfile", "-Command", cmd], capture_output=True, text=True).stdout.strip()


def page_text(b):
    t = html.unescape(re.sub(r"<[^>]+>", " ", b.decode("utf-8", "replace")))
    return re.sub(r"\s+", " ", t)


def excerpt(t, key, n=330):
    i = t.find(key)
    return t[i:i + n] if i >= 0 else None


# ------------------------------------------------------------------------------------------------------------- stage
def stage(a):
    out = Path(a.out_dir)
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}")
    out.mkdir(parents=True)
    rep = {"stage": "w133_preflight", "time": now(), "schema": "animcv_exp_w133_preflight_v1",
           "network_policy": "metadata only; checkpoint archive: Content-Range total size, EOCD and central directory only"}

    # ---- repository state ----
    w132 = (ROOT / "docs/132_WORKLOG_CONDMDI_IMPUTATION_CAUSAL_GATE_S1.md").read_text(encoding="utf-8")
    rep["repo"] = {"head": git("rev-parse", "HEAD"), "instruction_blob": git("rev-parse", "HEAD:docs/next_batch.md"),
                   "w132_complete": "AUTOMATION_BATCH_STATUS: COMPLETE" in w132,
                   "audit_w126_w130_present": (ROOT / "docs/AUDIT_W126_W130_2026_11_09.md").is_file()}
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
                    "overlap_with_interior": sorted(set(f - 78 for f in prop["pin_frames"]) & set(range(172, 200)))}
    rep["frozen_evidence"] = fz
    rep["frozen_evidence_ok"] = all(v.get("ok", True) for v in fz.values()) and not fz["S1_pin"]["overlap_with_interior"]

    # ---- official source revision (code-only clone) ----
    ls_new, ls_old = git("ls-remote", OC_REPO, "refs/heads/main"), git("ls-remote", OC_REPO_LEGACY, "refs/heads/main")
    local = W / "ext" / "omnicontrol"
    src = {"repo": OC_REPO, "legacy_alias": OC_REPO_LEGACY, "pinned_rev": OC_REV,
           "remote_main": ls_new.split()[0] if ls_new else None, "legacy_alias_main": ls_old.split()[0] if ls_old else None}
    gh = json.loads(get(f"https://api.github.com/repositories/{OC_REPO_ID}")[0])
    src["github"] = {k: gh.get(k) for k in ("full_name", "pushed_at", "archived")}
    src["github"]["license_spdx"] = (gh.get("license") or {}).get("spdx_id")
    src["local_checkout_head"] = git("rev-parse", "HEAD", cwd=local)
    src["local_worktree_clean"] = git("status", "--porcelain", cwd=local) == ""
    lic = (local / "LICENSE").read_bytes()
    src["license_sha256"] = sha256_bytes(lic)
    src["license_head"] = lic.decode("utf-8").splitlines()[:3]
    readme = (local / "README.md").read_text(encoding="utf-8")
    sec, scoped, defers = code_scoped_licence_section(readme)
    src["readme_license_section"] = sec.strip()
    src["readme_weights_section"] = readme[readme.find("### 3. Download the pretrained models"):][:420].strip()
    src["readme_mentions_weight_licence"] = bool(re.search(r"(?i)(weights?|checkpoints?|pretrained models?)[^\n]{0,80}licen", readme)
                                                 or re.search(r"(?i)licen[cs]e[^\n]{0,80}(weights?|checkpoints?|pretrained)", readme))
    src["readme_gdrive_ids"] = sorted(set(re.findall(r"drive\.google\.com/file/d/([\w-]+)", readme)))
    rep["omnicontrol_source"] = src

    # ---- official checkpoint archive identity (no weight bytes) ----
    ck = {"distribution": f"https://drive.google.com/file/d/{GD_ID}/view", "download_endpoint": GD_URL,
          "weights_downloaded": False, "archive_parsed": False}
    b0, h0, s0 = get(GD_URL, (0, 0))
    total = int(h0.get("Content-Range", "/0").rsplit("/", 1)[1])
    ck.update({"http_status_range_probe": s0, "content_disposition": h0.get("Content-Disposition"),
               "archive_total_bytes": total, "first_byte_is_zip_magic_P": b0 == b"P"})
    tail, _, _ = get(GD_URL, (total - 22, total - 1))
    eocd = parse_eocd(tail)
    cd, _, _ = get(GD_URL, (eocd["cd_offset"], eocd["cd_offset"] + eocd["cd_size"] - 1))
    members = parse_central_directory(cd, eocd["entries"])
    ck.update({"eocd": eocd, "members": members, "archive_parsed": True,
               "bytes_requested": 1 + 22 + eocd["cd_size"],
               "central_directory_sha256": sha256_bytes(cd), "terms_members": terms_members(members),
               "weight_member_sha256": "NOT VERIFIED (weights not downloaded; CRC-32 and sizes from the central directory only)"})
    save(out, "archive/central_directory.bin", cd)
    save(out, "archive/eocd.bin", tail)
    rep["checkpoint_archive"] = ck

    # ---- official project page / issue tracker ----
    pp = get(PROJECT_PAGE)[0]
    off = {"project_page": save(out, "official/project_page.html", pp)}
    ppt = page_text(pp)
    off["project_page_licence_sentences"] = re.findall(r"[^.]{0,160}licen[cs]ed?[^.]{0,160}\.", ppt)
    off["project_page_weight_terms"] = bool(re.search(r"(?i)(weights?|checkpoints?|pretrained model)[^.]{0,120}licen", ppt))
    issues = {}
    for q in ("license", "licence", "commercial", "weights", "checkpoint"):
        d = json.loads(get(f"https://api.github.com/search/issues?q=repo:neu-3dvi/OmniControl+{q}&per_page=50")[0])
        issues[q] = [{"number": i["number"], "title": i["title"], "state": i["state"]} for i in d.get("items", [])]
        issues[q + "_total"] = d.get("total_count")
    off["issue_search"] = issues
    off["issues_searched"] = all(isinstance(issues.get(q + "_total"), int) for q in ("license", "licence", "commercial"))
    hits = {i["number"]: i["title"] for q in ("license", "licence", "commercial") for i in issues[q]}
    off["issues_with_licence_terms"] = [{"number": k, "title": v} for k, v in sorted(hits.items())]
    off["issues_weights_or_checkpoint_titles_reviewed"] = sorted(
        {i["title"] for q in ("weights", "checkpoint") for i in issues[q]})
    rep["official_terms_search"] = off

    # ---- fine-tune parent: MDM ----
    mdm = get(MDM_README)[0]
    msec, mscoped, mdefers = code_scoped_licence_section(mdm.decode("utf-8"))
    rep["parent_mdm"] = {"readme": save(out, "upstream/mdm_README.md", mdm), "license_section": msec.strip(),
                         "scoped_to_code": mscoped, "defers_datasets": mdefers,
                         "omnicontrol_resumes_from": "./save/model000475000.pt (official README 'Train your own OmniControl')"}

    # ---- third-party mirror (recorded; never a weight source) ----
    api = json.loads(get(f"https://huggingface.co/api/models/{MIRROR_ID}/revision/{MIRROR_REV}?blobs=true")[0])
    mir = {"id": MIRROR_ID, "pinned_rev": MIRROR_REV, "api_sha": api.get("sha"), "author": api.get("author"),
           "lastModified": api.get("lastModified"), "card_license": (api.get("cardData") or {}).get("license"),
           "files": {s["rfilename"]: {"bytes": s.get("size"), "lfs_sha256": (s.get("lfs") or {}).get("sha256")}
                     for s in api.get("siblings", [])}, "used_as_weight_source": False}
    for f in ("README.md", "LICENSE", "omnicontrol_config.json", "model_index.json"):
        mir[f] = save(out, f"mirror/{f}", get(f"https://huggingface.co/{MIRROR_ID}/resolve/{MIRROR_REV}/{f}")[0])
    mir["license_equals_official_code_license"] = (out / "mirror/LICENSE").read_bytes() == lic
    mir["card_text"] = (out / "mirror/README.md").read_text(encoding="utf-8")
    mir["publisher_is_not_rights_holder"] = mir["author"] not in ("neu-3dvi", "neu-vi", "ymingxie")
    rep["mirror"] = mir

    # ---- runtime assets required by the official sampling path (static reading) ----
    gen = (local / "sample/generate.py").read_text(encoding="utf-8")
    cmdm = (local / "model/cmdm.py").read_text(encoding="utf-8")
    r2x = (local / "model/rotation2xyz.py").read_text(encoding="utf-8")
    smpl_sh = (local / "prepare/download_smpl_files.sh").read_text(encoding="utf-8")
    hml_api = json.loads(get(HUMANML3D_API)[0])
    hml_dir = json.loads(get(HUMANML3D_API + "/contents/HumanML3D")[0])
    rt = {
        "r1_model_ctor_builds_smpl": "self.rot2xyz = Rotation2xyz(" in cmdm and "self.smpl_model = SMPL()" in r2x,
        "r2_official_sample_loads_train_mode_dataset": "hml_mode='train'" in gen and "split='test'" in gen,
        "r3_smpl_download_is_third_party_drive_copy": "drive.google.com" in smpl_sh and "smpl" in smpl_sh,
        "fixed_196_frames": "n_frames = 196" in gen,
        "args_json_in_archive": any(m["name"].endswith("args.json") for m in members),
        "humanml3d_repo_license_spdx": (hml_api.get("license") or {}).get("spdx_id"),
        "humanml3d_repo_ships": sorted(f"{d['name']}:{d['size']}" for d in hml_dir),
        "bundled_spatial_norm": sorted(p.name for p in (local / "dataset/humanml_spatial_norm").iterdir()),
    }
    terms = {}
    for k, u in TERMS_PAGES.items():
        b = get(u)[0]
        t = page_text(b)
        terms[k] = {"page": save(out, f"terms/{k}.html", b), "title": excerpt(t, "License for non-commercial", 70),
                    "use_clause": excerpt(t, "Any other use, in particular", 260),
                    "training_clause": excerpt(t, "This license also prohibits", 170),
                    "no_distribution_clause": excerpt(t, "No Distribution", 200)}
    rt["third_party_terms"] = terms
    rt["status"] = ("RESTRICTED_ASSETS_REQUIRED_BY_OFFICIAL_PATH"
                    if rt["r1_model_ctor_builds_smpl"] and rt["r2_official_sample_loads_train_mode_dataset"] else "NOT_ESTABLISHED")
    rep["runtime_assets"] = rt

    # ---- execution locale ----
    rep["execution_locale"] = {"timezone": ps("(Get-TimeZone).Id"), "home_geo_id": ps("(Get-WinHomeLocation).GeoId"),
                               "culture": ps("(Get-Culture).Name"), "system_locale": ps("(Get-WinSystemLocale).Name")}

    # ---- declared terms rule ----
    facts = {"archive_parsed": ck["archive_parsed"], "archive_terms_members": ck["terms_members"],
             "readme_scoped_to_code": scoped, "readme_defers_datasets": defers,
             "readme_mentions_weight_licence": src["readme_mentions_weight_licence"],
             "project_page_fetched": True, "project_page_weight_terms": off["project_page_weight_terms"],
             "issues_searched": off["issues_searched"], "issues_with_licence_terms": off["issues_with_licence_terms"],
             "mdm_scoped_to_code": mscoped, "mirror_exists": True,
             "mirror_publisher_is_not_rights_holder": mir["publisher_is_not_rights_holder"],
             "mirror_licence_equals_code_licence": mir["license_equals_official_code_license"]}
    rule, verdict = evaluate_terms_rule(facts)
    rep["terms_facts"], rep["terms_rule"], rep["terms_verdict"] = facts, rule, verdict
    ok_pre = (rep["repo"]["blob_ok"] and rep["repo"]["starting_commit_is_ancestor"] and rep["repo"]["w132_complete"]
              and rep["frozen_evidence_ok"] and src["local_checkout_head"] == OC_REV and src["remote_main"] == OC_REV)
    rep["prerequisites_other_than_terms_ok"] = bool(ok_pre)
    rep["status"] = "DATA_BLOCKED_LICENSE" if verdict == "DATA_BLOCKED_LICENSE" else "STOP_TERMS_UNRESOLVED"
    rep["not_done_by_rule"] = ["no checkpoint download (official or mirror)", "no scratch model runtime install",
                               "no official smoke", "no prospective declaration", "no S1 control extraction for the model",
                               "no S1 inference (guided or unguided)", "no Blender candidate, video or metrics",
                               "no alternative model, mirror substitution, SMPL/HumanML3D acquisition or runtime patch"]
    rep["scratch_weight_files_present"] = [str(p) for p in W.rglob("*") if p.is_file() and (
        p.suffix in (".pt", ".pth", ".safetensors", ".ckpt") or (p.suffix in (".bin", ".zip") and p.stat().st_size > 1 << 20))]
    (out / "preflight.json").write_text(json.dumps(rep, indent=1, sort_keys=True), encoding="utf-8")
    print(json.dumps({"status": rep["status"], "terms_rule": rule, "runtime_assets": rt["status"],
                      "prerequisites_other_than_terms_ok": rep["prerequisites_other_than_terms_ok"],
                      "sha256": sha256_file(out / "preflight.json")}, indent=1))
    return 3


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-dir", required=True)
    return stage(ap.parse_args())


if __name__ == "__main__":
    sys.exit(main())
