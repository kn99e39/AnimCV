#!/usr/bin/env python3
"""W130 (EXPERIMENTAL, opt-in) preflight: AnimateCanvas 0.46B checkpoint identity + terms, frozen S1 / W126 / W129 evidence.

Runs BEFORE any large download or model inference.  It
  * authenticates the repository state, the frozen S1 baseline, the W126V-accepted pin variant, the W126 lineage head,
    the source clip and the W129 A/B/C/Cpin artifacts (byte hashes only; nothing is rebuilt or modified);
  * pins the official AnimateCanvas source revision and the Hugging Face checkpoint revision, records the LFS identity
    (size + SHA-256 OID) of every weight file WITHOUT downloading weights, and fetches only small text metadata
    (model card, configs, provenance notes, text-encoder licences) at the pinned revisions;
  * follows the checkpoint's documented upstream chain (Motius model card -> HY-Motion T2M warm start -> Tencent
    HY-Motion 1.0 licence) and the execution locale of this machine, and applies the declared terms rule below.

Terms rule (declared in this file before the checkpoint weights were fetched; no weights are fetched by this script):
  ALLOWED only if the motion-transformer checkpoint carries an affirmative licence for local experimental evaluation
  at the execution location.  DATA_BLOCKED_LICENSE if (a) the checkpoint declares no licence of its own AND (b) its
  documented provenance makes it a warm-started derivative of a model whose licence (c) excludes the execution
  territory, (d) with the execution locale evidenced as that excluded territory.  Any un-evidenced link -> UNRESOLVED
  (also a stop: no inference without positive terms).  MIT inference code does not license the weights.

    PY=C:/Projects/AnimCV/.venv_build_full/Scripts/python.exe
    $PY -X utf8 scripts/w130_animatecanvas_preflight.py --out-dir C:/Users/dna10/animcv-w130/results/preflight
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
W = Path("C:/Users/dna10/animcv-w130")
W129 = Path("C:/Users/dna10/animcv-w129")
W126 = Path("C:/Users/dna10/animcv-w126/runs")

STARTING_COMMIT = "65e6d8f7f735dca4a8642ab7c03881fd6e699d03"
INSTRUCTION_BLOB = "550c8a63f1606f9dd47b5029b7ccd635c9bc1812"

AC_REPO = "https://github.com/ZeyuLing/AnimateCanvas.git"
AC_REV = "e59d68184ab8cb7852f598d688183f62303ae45e"                        # Batch-pinned standalone source revision
HF_ID = "ZeyuLing/Motius-AnimateCanvas-0.46B"
HF_LEGACY = "ZeyuLing/Motius-MotionCanvas-0.46B"
HF_REV = "6ef7830c5872b747a938251958c588e5e291107e"                         # resolved at preflight, then pinned here
MOTIUS_REV = "d04a38c9bc9dab3f0e9e2d843f8a6b21eaa13abc"                     # Motius main at preflight (model card)
T2M_ID = "ZeyuLing/Motius-HYMotion-T2M-1.0-Lite"
T2M_REV = "128743eb5971d0164a8ef91ae8d85920b9348855"
HY_ID = "tencent/HY-Motion-1.0"
HY_REV = "620dd559f8d964aac2f82f1204fe6a35ad8ad14d"
UA = {"User-Agent": "animcv-w130-preflight/1.0"}

META_FILES = ["README.md", "animatecanvas_config.json", "motioncanvas_config.json", "model_index.json",
              "normalization_provenance.md", "source_checkpoint.json", "text_encoder/llm/LICENSE",
              "text_encoder/llm/README.md", "text_encoder/llm/config.json", "text_encoder/sentence/README.md",
              "text_encoder/sentence/config.json"]

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


def get(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
        return r.read()


def save(out, rel, data):
    p = out / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)
    return {"path": str(p), "bytes": len(data), "sha256": sha256_bytes(data)}


def git(*a, cwd=ROOT):
    return subprocess.run(["git", *a], cwd=cwd, capture_output=True, text=True).stdout.strip()


def ps(cmd):
    return subprocess.run(["powershell.exe", "-NoProfile", "-Command", cmd], capture_output=True, text=True).stdout.strip()


def stage(a):
    out = Path(a.out_dir)
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}")
    out.mkdir(parents=True)
    rep = {"stage": "w130_preflight", "time": now(), "schema": "animcv_exp_w130_preflight_v1"}

    # ---- repository state ----
    rep["repo"] = {"head": git("rev-parse", "HEAD"), "instruction_blob": git("rev-parse", "HEAD:docs/next_batch.md"),
                   "w129_complete": "AUTOMATION_BATCH_STATUS: COMPLETE" in (
                       ROOT / "docs/129_WORKLOG_PRETRAINED_MOTIONBRICKS_LEFT_ARM_INBETWEENING.md").read_text(encoding="utf-8")}
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
                             "ok": refs["head"] == S1_HEAD_STATE and refs["head_blend"]["value_digest"] == S1_VALUE_DIGEST}
    prop = json.loads((W126 / "S1_replay/edits/W126V/proposal.json").read_text())
    fz["S1_pin"] = {"pin_source_frames": prop["pin_frames"], "pin_action_frames": [f - 78 for f in prop["pin_frames"]],
                    "target_m": prop["constraint"]["target"], "tolerance_m": prop["constraint"]["tolerance_m"],
                    "edit_interior_action": [172, 199], "edit_interior_source": [250, 277],
                    "overlap_with_interior": sorted(set(f - 78 for f in prop["pin_frames"]) & set(range(172, 200)))}
    w129 = json.loads((W129 / "results/assess/assessment.json").read_text())
    fz["W129_case"] = w129["case"]
    rep["frozen_evidence"] = fz
    rep["frozen_evidence_ok"] = all(v.get("ok", True) for v in fz.values() if isinstance(v, dict)) \
        and not fz["S1_pin"]["overlap_with_interior"] and fz["W129_case"] == "B"

    # ---- AnimateCanvas source revision ----
    ls = git("ls-remote", AC_REPO, "refs/heads/main")
    local = W / "ext" / "animatecanvas"
    src = {"repo": AC_REPO, "pinned_rev": AC_REV, "remote_main": ls.split()[0] if ls else None}
    if (local / ".git").exists():
        src["local_checkout_head"] = git("rev-parse", "HEAD", cwd=local)
        src["local_worktree_clean"] = git("status", "--porcelain", cwd=local) == ""
        src["license_sha256"] = sha256_file(local / "LICENSE")
        src["license_first_line"] = (local / "LICENSE").read_text(encoding="utf-8").splitlines()[0]
        readme = (local / "README.md").read_text(encoding="utf-8")
        src["readme_terms_excerpt"] = readme[readme.find("The original inference entry points"):][:700]
    rep["animatecanvas_source"] = src

    # ---- Hugging Face checkpoint identity (metadata only; NO weights fetched) ----
    api = json.loads(get(f"https://huggingface.co/api/models/{HF_ID}/revision/{HF_REV}"))
    legacy = json.loads(get(f"https://huggingface.co/api/models/{HF_LEGACY}"))          # 307-redirects to HF_ID
    tree = json.loads(get(f"https://huggingface.co/api/models/{HF_ID}/tree/{HF_REV}?recursive=true"))
    weights = {t["path"]: {"bytes": t["size"], "lfs_sha256": t["lfs"]["oid"]} for t in tree
               if t["type"] == "file" and t.get("lfs") and t["path"].endswith((".safetensors", ".pt", ".npy"))}
    ck = {"id": HF_ID, "pinned_rev": HF_REV, "api_sha": api.get("sha"), "gated": api.get("gated"),
          "private": api.get("private"), "lastModified": api.get("lastModified"),
          "card_license": (api.get("cardData") or {}).get("license"), "tags": api.get("tags"),
          "legacy_alias_resolves_to": legacy.get("id"), "legacy_alias_sha": legacy.get("sha"),
          "used_storage_bytes": api.get("usedStorage"), "weight_files_lfs_identity": weights,
          "total_lfs_bytes": sum(t["size"] for t in tree if t["type"] == "file"),
          "weights_downloaded": False, "meta": {}}
    for f in META_FILES:
        ck["meta"][f] = save(out / "hf_meta", f, get(f"https://huggingface.co/{HF_ID}/resolve/{HF_REV}/{f}"))
    card = (out / "hf_meta" / "README.md").read_text(encoding="utf-8")
    cfg = json.loads((out / "hf_meta" / "animatecanvas_config.json").read_text(encoding="utf-8"))
    srcck = json.loads((out / "hf_meta" / "source_checkpoint.json").read_text(encoding="utf-8"))
    ck["card_has_license_section_or_field"] = bool(re.search(r"(?im)^license\s*:", card)) or "## License" in card
    ck["card_provenance_excerpt"] = card[card.find("## Provenance"):].strip()
    ck["native_fps"] = srcck.get("native_fps")
    ck["representation"] = srcck.get("native_representation")
    ck["motion_type"] = cfg["config"].get("motion_type")
    ck["transformer"] = {k: cfg["config"]["motion_transformer"].get(k) for k in ("type", "input_dim", "output_dim", "num_layers", "feat_dim")}
    ck["text_encoder"] = cfg["config"].get("text_encoder")
    ck["llm_license_first_line"] = (out / "hf_meta/text_encoder/llm/LICENSE").read_text(encoding="utf-8").strip().splitlines()[0]
    rep["checkpoint"] = ck

    # ---- documented upstream chain ----
    mc = get(f"https://raw.githubusercontent.com/ZeyuLing/Motius/{MOTIUS_REV}/docs/model_zoo/animatecanvas.md")
    up = {"motius_model_card": save(out / "upstream", "motius_animatecanvas_model_card.md", mc)}
    mct = mc.decode("utf-8")
    m = re.search(r"warm-starts the Text-to-Motion backbone from\s*`([^`]+)`", mct)
    up["warm_start_from"] = m.group(1) if m else None
    lic = mct[mct.find("## Citation and License"):]
    up["motius_card_license_section"] = lic[:900]
    motius_lic = urllib.request.Request(f"https://raw.githubusercontent.com/ZeyuLing/Motius/{MOTIUS_REV}/LICENSE", headers=UA)
    try:
        urllib.request.urlopen(motius_lic, timeout=60)
        up["motius_root_license_file"] = "PRESENT"
    except urllib.error.HTTPError as e:
        up["motius_root_license_file"] = f"ABSENT (HTTP {e.code})"
    t2m_api = json.loads(get(f"https://huggingface.co/api/models/{T2M_ID}/revision/{T2M_REV}"))
    t2m = get(f"https://huggingface.co/{T2M_ID}/resolve/{T2M_REV}/README.md")
    up["t2m_card"] = save(out / "upstream", "Motius-HYMotion-T2M-1.0-Lite_README.md", t2m)
    up["t2m_card_license"] = (t2m_api.get("cardData") or {}).get("license")
    om = re.search(r"\*\*Original weights\*\*\s*\|\s*\[`([^`]+)`\]", t2m.decode("utf-8"))
    up["t2m_original_weights"] = om.group(1) if om else None
    hy_api = json.loads(get(f"https://huggingface.co/api/models/{HY_ID}/revision/{HY_REV}"))
    hyl = get(f"https://huggingface.co/{HY_ID}/resolve/{HY_REV}/LICENSE.txt")
    up["hy_license"] = save(out / "upstream", "HY-Motion-1.0_LICENSE.txt", hyl)
    up["hy_card_license"] = {k: (hy_api.get("cardData") or {}).get(k) for k in ("license", "license_name", "license_link")}
    hyt = hyl.decode("utf-8", errors="replace")
    lines = hyt.splitlines()
    up["hy_license_title"] = lines[0].strip()
    up["hy_license_scope_line"] = next((l.strip() for l in lines if "DOES NOT APPLY" in l), None)
    up["hy_territory_definition"] = next((l.strip() for l in lines if "“Territory” shall mean" in l or '"Territory" shall mean' in l), None)
    up["hy_model_derivatives_definition"] = next((l.strip() for l in lines if "Model Derivatives” shall mean" in l), None)
    up["hy_outside_territory_clause"] = next((l.strip() for l in lines if "outside the Territory" in l and "must not" in l), None)
    up["hy_aup_outside_territory"] = next((l.strip() for l in lines if l.strip().startswith("1.") and "Outside the Territory" in l), None)
    rep["upstream_chain"] = up

    # ---- execution locale (this machine) ----
    rep["execution_locale"] = {"timezone": ps("(Get-TimeZone).Id"), "home_geo_id": ps("(Get-WinHomeLocation).GeoId"),
                               "culture": ps("(Get-Culture).Name"), "system_locale": ps("(Get-WinSystemLocale).Name"),
                               "note": "GeoId 134 = Korea (Republic of). Supporting context: the repository's commit timestamps are +0900 and the "
                                       "project's Korean-language central documents; the user's institutional e-mail domain is Korean."}

    # ---- declared terms rule ----
    terr = (up["hy_territory_definition"] or "")
    rule = {
        "a_checkpoint_declares_no_licence": ck["card_license"] is None and not ck["card_has_license_section_or_field"],
        "b_checkpoint_is_warm_started_derivative": up["warm_start_from"] == T2M_ID and up["t2m_original_weights"] == HY_ID
                                                   and "HY-Motion 1.0" in ck["card_provenance_excerpt"],
        "c_upstream_licence_excludes_south_korea": "South Korea" in terr and "excluding" in terr
                                                    and bool(up["hy_outside_territory_clause"]),
        "c2_derivative_definition_covers_weight_transfer": bool(up["hy_model_derivatives_definition"])
                                                           and "modifications to" in up["hy_model_derivatives_definition"],
        "d_execution_locale_south_korea": rep["execution_locale"]["home_geo_id"] == "134"
                                          and rep["execution_locale"]["timezone"] == "Korea Standard Time",
        "e_code_mit_does_not_license_weights": "does not relicense upstream code or third-party assets" in src.get("readme_terms_excerpt", "")
                                               or "Checkpoints" in src.get("readme_terms_excerpt", ""),
    }
    if all(rule.values()):
        verdict = "DATA_BLOCKED_LICENSE"
    elif rule["a_checkpoint_declares_no_licence"]:
        verdict = "UNRESOLVED_TERMS_STOP"
    else:
        verdict = "TERMS_REQUIRE_MANUAL_REVIEW"
    rep["terms_rule"] = rule
    rep["terms_verdict"] = verdict
    ok_pre = rep["repo"]["blob_ok"] and rep["repo"]["starting_commit_is_ancestor"] and rep["repo"]["w129_complete"] and rep["frozen_evidence_ok"]
    rep["status"] = ("DATA_BLOCKED_LICENSE" if verdict == "DATA_BLOCKED_LICENSE" else
                     ("STOP_TERMS_UNRESOLVED" if verdict != "ALLOWED" else ("OK" if ok_pre else "DATA_BLOCKED")))
    rep["prerequisites_other_than_terms_ok"] = bool(ok_pre)
    rep["not_done_by_rule"] = ["no checkpoint weight / text-encoder weight download", "no pip install of torch/animatecanvas",
                               "no SMPL-22 mapping adapter, mask declaration or Blender candidate", "no inference (smoke or primary)",
                               "no alternative model, no hosted/remote inference, no workaround"]
    stray = [str(p) for p in W.rglob("*.safetensors")] + [str(p) for p in W.rglob("*.bin") if p.stat().st_size > 1 << 20]
    rep["scratch_weight_files_present"] = stray
    (out / "preflight.json").write_text(json.dumps(rep, indent=1, sort_keys=True), encoding="utf-8")
    print(json.dumps({"status": rep["status"], "terms_rule": rule, "frozen_evidence_ok": rep["frozen_evidence_ok"],
                      "sha256": sha256_file(out / "preflight.json")}, indent=1))
    return 0 if rep["status"] == "OK" else 3


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-dir", required=True)
    return stage(ap.parse_args())


if __name__ == "__main__":
    sys.exit(main())
