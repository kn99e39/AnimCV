#!/usr/bin/env python3
"""Download selected AMASS subsets from the public Hugging Face mirror.

This tool deliberately uses only the Python standard library so it can run on
minimal training servers without adding a system-wide package.  It downloads
the original AMASS ``.npz`` motion-parameter files, preserves their canonical
``raw/<subset>/...`` layout, verifies an expected byte size when supplied by
the API, and writes each file atomically.

Example:
  python scripts/download_amass_hf.py \
    --out /home/nd/animcv-data/amass/raw/amass_hf \
    --subsets CMU,KIT,BMLmovi,BMLrub
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import re
import shutil
import sys
import time
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen


REPOSITORY = "realdream-ai/AMASS"
REVISION = "main"
USER_AGENT = "AnimCV-AMASS-downloader/1.0"


def _request(url: str):
    return urlopen(Request(url, headers={"User-Agent": USER_AGENT}), timeout=120)


def _entries(subset: str) -> list[dict]:
    url = (
        f"https://huggingface.co/api/datasets/{REPOSITORY}/tree/{REVISION}/"
        f"raw/{quote(subset)}?recursive=true&expand=false&limit=1000"
    )
    entries: list[dict] = []
    while url:
        with _request(url) as response:
            entries.extend(json.load(response))
            link = response.headers.get("Link", "")
        match = re.search(r'<([^>]+)>;\s*rel="next"', link)
        url = match.group(1) if match else ""
    return [entry for entry in entries if entry.get("type") == "file" and entry["path"].endswith(".npz")]


def _download(path: str, expected_size: int | None, output_root: Path, retries: int) -> str:
    destination = output_root / path
    if destination.exists() and (expected_size is None or destination.stat().st_size == expected_size):
        return "skipped"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    url = f"https://huggingface.co/datasets/{REPOSITORY}/resolve/{REVISION}/{quote(path)}"
    for attempt in range(1, retries + 1):
        temporary.unlink(missing_ok=True)
        try:
            with _request(url) as response, temporary.open("wb") as out:
                shutil.copyfileobj(response, out, length=1024 * 1024)
            if expected_size is None or temporary.stat().st_size == expected_size:
                temporary.replace(destination)
                return "downloaded"
            actual = temporary.stat().st_size
            temporary.unlink(missing_ok=True)
            if attempt == retries:
                raise RuntimeError(f"size mismatch for {path}: expected {expected_size}, got {actual}")
            time.sleep(min(2 ** (attempt - 1), 8))
        except Exception:
            temporary.unlink(missing_ok=True)
            if attempt == retries:
                raise
            time.sleep(min(2 ** (attempt - 1), 8))
    raise AssertionError("unreachable")


def main() -> int:
    parser = argparse.ArgumentParser(description="Download selected original AMASS NPZ subsets")
    parser.add_argument("--out", required=True, type=Path, help="Root that will contain raw/<subset>/...")
    parser.add_argument("--subsets", required=True, help="Comma-separated AMASS subset names")
    parser.add_argument("--retries", type=int, default=3, help="Attempts per file for transient transfer failures")
    parser.add_argument("--workers", type=int, default=4, help="Parallel file downloads per subset")
    args = parser.parse_args()
    subsets = [name.strip() for name in args.subsets.split(",") if name.strip()]
    if not subsets:
        raise ValueError("--subsets must name at least one subset")
    if args.retries < 1 or args.workers < 1:
        raise ValueError("--retries and --workers must be at least one")

    downloaded = skipped = 0
    for subset in subsets:
        files = _entries(subset)
        if not files:
            raise RuntimeError(f"no NPZ files found for AMASS subset {subset!r}")
        print(f"[amass] {subset}: {len(files)} files", flush=True)
        def download(entry: dict) -> str:
            return _download(entry["path"], entry.get("size"), args.out, args.retries)

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            statuses = pool.map(download, files)
            for status in statuses:
                downloaded += status == "downloaded"
                skipped += status == "skipped"
    print(json.dumps({"repository": REPOSITORY, "subsets": subsets,
                      "downloaded_files": downloaded, "skipped_files": skipped}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
