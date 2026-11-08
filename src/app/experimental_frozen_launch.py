"""EXPERIMENTAL: process-local launcher for the FROZEN W102 M1 + S0 perception code snapshot (read-only use).

Equivalent to the W121/W122 scratch launcher, but every location is an explicit argument (no hardcoded workspace):

    python -X utf8 experimental_frozen_launch.py --code-dir <W102 code snapshot> --scratch-dir <run>/perception/scratch \
        w102_video_diagnostic.py run --clip ... --out-dir ...

tmp/cache directories are redirected into ``--scratch-dir`` so the snapshot directory is never written.  The snapshot's
diagnostic script identity is verified by the caller (app.experimental_video_action) before launch.
"""
from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path


def main(argv):
    if len(argv) < 5 or argv[0] != "--code-dir" or argv[2] != "--scratch-dir":
        raise SystemExit("usage: experimental_frozen_launch.py --code-dir DIR --scratch-dir DIR <script> [args...]")
    code, base, script_name, rest = Path(argv[1]), Path(argv[3]), argv[4], argv[5:]
    if not (code / "scripts" / script_name).exists():
        raise SystemExit(f"frozen script missing: {code / 'scripts' / script_name}")
    for key, rel in {"TMP": "tmp", "TEMP": "tmp", "TMPDIR": "tmp", "XDG_CACHE_HOME": "cache", "TORCH_HOME": "cache/torch",
                     "MPLCONFIGDIR": "cache/matplotlib", "CUDA_CACHE_PATH": "cache/cuda", "HF_HOME": "cache/huggingface"}.items():
        p = base / rel
        p.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(p)
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    sys.path[:0] = [str(code / "src"), str(code / "scripts")]
    os.chdir(code)
    script = code / "scripts" / script_name
    sys.argv = [str(script)] + rest
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main(sys.argv[1:])
