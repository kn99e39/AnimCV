"""W128 read-only preflight for the opt-in Unity Generic-rig FBX consumer.

Run BEFORE any Unity import outcome exists. It never imports an FBX into Unity and never opens an AnimCV run for
writing. It records:

1. the Unity Editor installations under the Hub editor root (executable present or not);
2. a licence/batchmode probe per executable editor: ``-batchmode -nographics -quit -createProject`` into a NEW
   scratch directory under ``--out-dir`` (the only thing Unity is asked to do), its exit code, wall time and the
   licensing lines of its own log (no licence-file content, tokens or account data are copied);
3. the local entitlement licence file's modification time and ``UpdateDate`` only, and the licensing-client log
   lines written during the probe that state why no entitlement was usable;
4. authentication of the six W127 S1/S3 packages (manifest seal, schema, FBX SHA-256 against the manifest, and both
   files against the W127 evidence manifest whose own SHA-256 is pinned below);
5. Blender 5.1 availability (``--version``), needed by the evaluator side.

Verdict ``DATA_BLOCKED_UNITY_LICENSE`` if no installed editor starts under a valid licence; the Batch's STOP rule then
applies and no declaration / import is attempted.

Usage (Git Bash, repository root):
  PY=C:/Projects/AnimCV/.venv_build_full/Scripts/python.exe
  PYTHONPATH=src $PY -X utf8 scripts/w128_unity_generic_preflight.py --out-dir C:/Users/dna10/animcv-w128/preflight
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
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from blender.experimental_fbx_interchange import PACKAGE_SCHEMA, seal_ok  # noqa: E402  (read-only helpers)

HUB_EDITOR_ROOT = Path("C:/Program Files/Unity/Hub/Editor")
LICENSE_FILE = Path(os.environ.get("LOCALAPPDATA", "")) / "Unity" / "licenses" / "UnityEntitlementLicense.xml"
LICENSING_CLIENT_LOG = Path(os.environ.get("LOCALAPPDATA", "")) / "Unity" / "Unity.Licensing.Client.log"
BLENDER = Path(os.environ.get("ANIMCV_BLENDER", "C:/Program Files/Blender Foundation/Blender 5.1/blender.exe"))

W127_ROOT = Path("C:/Users/dna10/animcv-w127")
W127_EVIDENCE_MANIFEST = W127_ROOT / "results" / "evidence_manifest.json"
W127_EVIDENCE_MANIFEST_SHA = "8d70fa406d395a111aaf9e035405ad38f7c1c4d09515cd84724eacbf0493a4c2"  # W127 Worklog §12
SOURCES = ("S1", "S3")
STATES = ("baseline", "w126v", "head")

LICENSE_LINE = re.compile(r"licens|entitle|token|No valid Unity Editor", re.I)
CLIENT_REASON = re.compile(r"OfflineValidityPeriodIsExpired|Token not found in cache|No ULF license|"
                           r"Access token is unavailable|not active", re.I)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()


def editors() -> list[dict]:
    out = []
    if not HUB_EDITOR_ROOT.is_dir():
        return out
    for d in sorted(HUB_EDITOR_ROOT.iterdir()):
        exe = d / "Editor" / "Unity.exe"
        out.append({"version": d.name, "unity_exe": str(exe), "unity_exe_present": exe.is_file(),
                    "unity_com_present": (d / "Editor" / "Unity.com").is_file()})
    return out


def client_log_size() -> int:
    return LICENSING_CLIENT_LOG.stat().st_size if LICENSING_CLIENT_LOG.is_file() else 0


def client_log_since(offset: int) -> list[str]:
    if not LICENSING_CLIENT_LOG.is_file():
        return []
    with open(LICENSING_CLIENT_LOG, "rb") as f:
        f.seek(offset)
        text = f.read().decode("utf-8", "replace")
    return sorted({re.sub(r"^\S+ \S+ - \[\s*\d+\] - \[\s*\d+\] - ", "", ln).strip()[:240]
                   for ln in text.splitlines() if CLIENT_REASON.search(ln)})


def probe(editor: dict, out_dir: Path, timeout_s: int) -> dict:
    tag = re.sub(r"[^0-9A-Za-z]+", "_", editor["version"])
    project = out_dir / f"probe_project_{tag}"
    log = out_dir / f"probe_{tag}.log"
    if project.exists() or log.exists():
        raise SystemExit(f"refusing existing probe output {project} / {log}")
    off = client_log_size()
    t0 = time.perf_counter()
    try:
        cp = subprocess.run([editor["unity_exe"], "-batchmode", "-nographics", "-quit", "-createProject", str(project),
                             "-logFile", str(log)], capture_output=True, text=True, timeout=timeout_s)
        exit_code, timed_out = cp.returncode, False
    except subprocess.TimeoutExpired:
        exit_code, timed_out = None, True
    secs = round(time.perf_counter() - t0, 2)
    lines = log.read_text("utf-8", "replace").splitlines() if log.is_file() else []
    lic = sorted({ln.strip()[:240] for ln in lines if LICENSE_LINE.search(ln)})
    no_license = any("No valid Unity Editor license found" in ln for ln in lines)
    created = (project / "Assets").is_dir() and (project / "ProjectSettings").is_dir()
    return {"version": editor["version"], "exit_code": exit_code, "timed_out": timed_out, "seconds": secs,
            "log": str(log), "log_sha256": sha256(log) if log.is_file() else None,
            "no_valid_license_message": no_license, "project_created": created,
            "editor_usable": bool(exit_code == 0 and created and not no_license),
            "editor_license_lines": lic, "licensing_client_reasons_during_probe": client_log_since(off)}


def license_meta() -> dict:
    if not LICENSE_FILE.is_file():
        return {"present": False}
    text = LICENSE_FILE.read_text("utf-8", "replace")
    st = LICENSE_FILE.stat()
    return {"present": True, "mtime_utc": dt.datetime.fromtimestamp(st.st_mtime, dt.timezone.utc).isoformat(),
            "update_dates": sorted(set(re.findall(r"<UpdateDate>([^<]{1,40})</UpdateDate>", text))),
            "note": "only UpdateDate/mtime recorded; licence content, tokens and account data are not copied"}


def packages() -> dict:
    res = {"evidence_manifest": str(W127_EVIDENCE_MANIFEST), "evidence_manifest_present": W127_EVIDENCE_MANIFEST.is_file()}
    if not W127_EVIDENCE_MANIFEST.is_file():
        res["ok"] = False
        return res
    ev_sha = sha256(W127_EVIDENCE_MANIFEST)
    res["evidence_manifest_sha256"] = ev_sha
    res["evidence_manifest_sha_matches_w127_worklog"] = ev_sha == W127_EVIDENCE_MANIFEST_SHA
    ev = json.loads(W127_EVIDENCE_MANIFEST.read_text("utf-8"))["files"]
    states = {}
    for src in SOURCES:
        for st in STATES:
            pkg = W127_ROOT / "results" / "real" / src / st / "export" / "package"
            rec = {"package_dir": str(pkg)}
            fbx, man_p = pkg / "animation.fbx", pkg / "package_manifest.json"
            if not (fbx.is_file() and man_p.is_file()):
                rec["ok"] = False
                rec["error"] = "PACKAGE_INCOMPLETE"
                states[f"{src}/{st}"] = rec
                continue
            man = json.loads(man_p.read_text("utf-8"))
            fbx_sha, man_sha = sha256(fbx), sha256(man_p)
            rel = f"results/real/{src}/{st}/export/package/"
            rec.update({
                "schema_ok": man.get("schema") == PACKAGE_SCHEMA,
                "seal_ok": seal_ok(man),
                "manifest_digest": man.get("manifest_digest"),
                "fbx_sha256": fbx_sha, "fbx_sha_matches_manifest": fbx_sha == man["fbx"]["sha256"],
                "fbx_bytes": fbx.stat().st_size, "keys_total": man["fbx"].get("keys_total"),
                "curves": man["fbx"].get("curves"), "patched_curves": man["fbx"].get("patched_curves"),
                "manifest_sha256": man_sha,
                "fbx_in_w127_evidence": ev.get(rel + "animation.fbx") == fbx_sha,
                "manifest_in_w127_evidence": ev.get(rel + "package_manifest.json") == man_sha,
                "take": man["take"]["name"], "timebase": man["timebase"],
                "rig_armature": man["rig"]["armature_object"], "rig_bone_count": man["rig"]["bone_count"],
                "source_state_blend_sha256": man["source"].get("blend_sha256"),
                "accepted_request_id": man["source"].get("accepted_request_id"),
                "producer": {k: man["producer"].get(k) for k in ("blender_version", "io_scene_fbx_version")},
            })
            rec["ok"] = all(rec[k] for k in ("schema_ok", "seal_ok", "fbx_sha_matches_manifest",
                                             "fbx_in_w127_evidence", "manifest_in_w127_evidence"))
            states[f"{src}/{st}"] = rec
    res["states"] = states
    res["ok"] = res["evidence_manifest_sha_matches_w127_worklog"] and all(s["ok"] for s in states.values())
    return res


def blender_version() -> dict:
    if not BLENDER.is_file():
        return {"present": False, "path": str(BLENDER)}
    cp = subprocess.run([str(BLENDER), "--factory-startup", "--version"], capture_output=True, text=True, timeout=120)
    return {"present": True, "path": str(BLENDER), "exit": cp.returncode, "version_line": cp.stdout.splitlines()[0]
            if cp.stdout else None}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--probe-timeout", type=int, default=900)
    a = ap.parse_args()
    out = Path(a.out_dir)
    if out.exists():
        raise SystemExit(f"refusing existing --out-dir {out}")
    out.mkdir(parents=True)
    eds = editors()
    probes = [probe(e, out, a.probe_timeout) for e in eds if e["unity_exe_present"]]
    rep = {
        "schema": "animcv_w128_unity_preflight_v1",
        "written_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "outcomes_inspected": False,
        "git_head": git("rev-parse", "HEAD"),
        "next_batch_blob": git("rev-parse", "HEAD:docs/next_batch.md"),
        "machine": os.environ.get("COMPUTERNAME"),
        "unity_editors": eds,
        "unity_batchmode_probes": probes,
        "license_file": license_meta(),
        "w127_packages": packages(),
        "blender": blender_version(),
    }
    usable = [p["version"] for p in probes if p["editor_usable"]]
    rep["usable_unity_editors"] = usable
    if not usable:
        rep["verdict"] = "DATA_BLOCKED_UNITY_LICENSE"
    elif not rep["w127_packages"]["ok"]:
        rep["verdict"] = "DATA_BLOCKED_PACKAGES"
    else:
        rep["verdict"] = "READY_FOR_DECLARATION"
    p = out / "preflight.json"
    p.write_text(json.dumps(rep, indent=1, sort_keys=True), "utf-8")
    print(json.dumps({"verdict": rep["verdict"], "usable_unity_editors": usable,
                      "packages_ok": rep["w127_packages"]["ok"], "preflight": str(p), "sha256": sha256(p)}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
