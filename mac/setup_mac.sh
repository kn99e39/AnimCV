#!/usr/bin/env bash
# One-time macOS dev setup: creates .venv and installs everything needed to
# run the full CLI pipeline (base deps + estimate-pose's mmpose stack + the
# depth-assisted 3D retargeting extra), plus the native assimp library
# parse-rig/create-mapping/retarget need. Run once, from anywhere:
#
#   bash mac/setup_mac.sh
#
# After it finishes, activate the venv and use the CLI normally -- this
# script does NOT build a bundled executable (see build_full_mac.sh for
# that: much heavier, and slower to iterate on since every change means a
# multi-GB rebuild). export-blender still shells out to a real Blender
# install; this script only checks for one, it doesn't install it.
#
# torch here is the CPU wheel (no CUDA on Mac); Apple Silicon gets MPS
# acceleration from the same wheel automatically.
#
# Why this can't just be `pip install -e ".[dev,pose,depth]"` (verified by
# actually running the resulting install against a real downloaded RTMPose
# checkpoint, not just importing the packages):
#
#   1. OpenMMLab publishes prebuilt mmcv wheels only for manylinux (Linux
#      x86_64) -- there has never been a macOS wheel -- so mmcv always
#      builds its C++ ops extension from source here.
#   2. That source build needs `import torch` to succeed *at build time*
#      to know it should compile ops at all (mmcv/setup.py silently falls
#      back to a pure-Python "lite" build -- no crash, no clear warning --
#      if torch isn't importable in the build environment). Two things
#      break this silently: (a) plain `pip install mmdet` pulls mmcv in
#      transitively and builds it inside pip's *isolated* build env, which
#      does not see this venv's already-installed torch; (b) even
#      `--no-build-isolation` mmcv installs are a no-op if a version
#      already satisfies the requirement (e.g. the lite one from (a)) --
#      `--force-reinstall` is required to actually rebuild it once that's
#      happened. This script avoids the whole trap by installing mmengine
#      alone first, then building mmcv explicitly (with torch already
#      present) before mmdet ever gets a chance to pull in its own copy.
#   3. Building mmcv the other legacy-packaging way: mmcv's setup.py does
#      `import pkg_resources`, which pip's isolated build env doesn't have
#      once setuptools >= 81 (pkg_resources was split out); mmpose's
#      `chumpy` dependency does `import pip` and `inspect.getargspec` in
#      its own setup.py, both removed upstream. Pinning `setuptools<81` in
#      this venv and building with --no-build-isolation (so the build
#      reuses this venv's older setuptools/pip instead of fetching fresh
#      ones in an isolated env) fixes both.
#   4. mmdet requires `mmcv<2.2.0`; mmpose's xtcocotools/pycocotools
#      wheels are compiled against numpy's pre-2.0 C ABI (numpy 2.0
#      changed the dtype struct layout), and opencv-python>=4.10 requires
#      numpy>=2 -- so numpy/opencv-python are pinned down together too.

set -euo pipefail

# Everything below uses paths relative to the project root (pyproject.toml,
# src/, etc.), not to this script's own location -- this file lives under
# mac/ specifically so it's never mistaken for the Windows scripts under
# windows/, so always hop back to the repo root first regardless of where
# it was invoked from.
cd "$(dirname "${BASH_SOURCE[0]}")/.."

VENV=".venv"

find_python() {
    # Prefer an actual 3.11 over anything else, and look beyond just
    # PATH for it -- on a machine with several Pythons installed (system
    # Python, more than one Homebrew python@X.Y keg, pyenv, the
    # python.org installer, Anaconda...), the interpreter `python3`/
    # `python` resolves to on PATH is often NOT 3.11 even when 3.11 is
    # very much installed, just unlinked/shadowed/not the active pyenv
    # version. Absolute-path candidates below cover the common install
    # layouts without requiring 3.11 to be first on PATH.
    for candidate in \
        python3.11 \
        /opt/homebrew/opt/python@3.11/bin/python3.11 \
        /usr/local/opt/python@3.11/bin/python3.11 \
        /Library/Frameworks/Python.framework/Versions/3.11/bin/python3.11 \
        "$HOME"/.pyenv/versions/3.11*/bin/python3
    do
        if command -v "$candidate" >/dev/null 2>&1; then
            echo "$candidate"
            return 0
        fi
    done

    # No 3.11 anywhere obvious -- fall back to whatever >=3.11 PATH
    # resolves to, preferring the newest since it's more likely to still
    # get security updates.
    for candidate in python3.13 python3.12 python3; do
        if command -v "$candidate" >/dev/null 2>&1; then
            local version
            version="$("$candidate" -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')"
            local major="${version%%.*}"
            local minor="${version##*.}"
            if [ "$major" -eq 3 ] && [ "$minor" -ge 11 ]; then
                echo "$candidate"
                return 0
            fi
        fi
    done
    return 1
}

# Prefer 3.11 specifically (not just >=3.11): it's the interpreter the
# mmcv/mmdet/mmpose recipe below was actually verified against, end to
# end, including running real inference against a downloaded checkpoint.
PYTHON="$(find_python || true)"
if [ -z "$PYTHON" ]; then
    echo "No Python >=3.11 found on PATH or common install locations" >&2
    echo "(Homebrew python@3.11, pyenv, python.org installer)." >&2
    echo "Install one, e.g. 'brew install python@3.11', and re-run." >&2
    exit 1
fi
echo "Using $PYTHON ($("$PYTHON" --version))"
if ! "$PYTHON" -c 'import sys; sys.exit(0 if sys.version_info[:2] == (3, 11) else 1)'; then
    echo "WARNING: this is not Python 3.11. The mmpose/mmcv/mmdet install" >&2
    echo "below was only verified against 3.11 -- it may hit new build" >&2
    echo "breaks on a different version. If it fails, 'brew install" >&2
    echo "python@3.11' and re-run so it's picked up above." >&2
fi

if [ ! -d "$VENV" ]; then
    echo "[1/10] Creating venv at $VENV..."
    "$PYTHON" -m venv "$VENV"
else
    echo "[1/10] Reusing existing venv at $VENV..."
fi
PIP="$VENV/bin/pip"
PY="$VENV/bin/python"

echo "[2/10] Upgrading pip..."
"$PY" -m pip install --upgrade pip

echo "[3/10] Pinning numpy<2 / opencv-python<4.10 (mmpose's xtcocotools native ABI needs numpy<2)..."
"$PIP" install "numpy<2" "opencv-python<4.10"

echo "[4/10] Installing torch (CPU)..."
"$PIP" install torch

echo "[5/10] Installing mmengine alone (mmdet comes later, after mmcv is built -- see header comment)..."
"$PIP" install mmengine

echo "[6/10] Building mmcv from source with torch already importable (no prebuilt macOS wheel exists upstream)..."
"$PIP" install "setuptools<81" wheel
"$PIP" install --no-build-isolation --no-cache-dir --force-reinstall --no-deps "mmcv>=2.0.0rc4,<2.2.0"

echo "[7/10] Installing mmdet (mmcv already satisfies its requirement, won't be rebuilt)..."
"$PIP" install mmdet

echo "[8/10] Installing mmpose (--no-build-isolation works around its chumpy dependency's legacy setup.py)..."
# mmpose's xtcocotools dependency only ships a prebuilt wheel for cp311;
# on any other interpreter (e.g. this machine has no Python 3.11 at all
# and fell back to a newer one -- see the WARNING above if so) it must
# build xtcocotools' _mask.pyx from source, which needs cython. With
# --no-build-isolation pip won't auto-fetch that as a build dependency,
# so it has to be already present here or the build fails with a
# confusing "clang: no such file or directory: xtcocotools/_mask.c"
# (cython never ran to generate that file). Confirmed by actually
# reproducing this failure on a fresh Python 3.12 venv.
"$PIP" install "cython>=0.27.3"
"$PIP" install --no-build-isolation mmpose

echo "[9/10] Installing project + dev/pose/depth extras (everything above already satisfies 'pose')..."
"$PIP" install -e ".[dev,pose,depth]"

echo "[10/10] Checking tkinter (needed for the GUI, motion-tool-gui / ui/gui_app.py)..."
if "$PY" -c "import tkinter" >/dev/null 2>&1; then
    echo "tkinter already available."
elif command -v brew >/dev/null 2>&1; then
    # Homebrew's python@X.Y formula doesn't bundle Tcl/Tk by default --
    # python-tk@X.Y is a separate formula providing the same venv's
    # _tkinter extension. The CLI is unaffected either way.
    py_minor="$("$PY" -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')"
    echo "Installing python-tk@$py_minor via Homebrew for GUI support..."
    brew install "python-tk@$py_minor" || echo "Could not auto-install python-tk@$py_minor; install it manually if you want the GUI." >&2
else
    echo "tkinter not available and Homebrew not found -- the GUI won't work; the CLI is unaffected." >&2
fi

echo
echo "Checking native assimp library (needed for parse-rig/create-mapping/retarget)..."
if command -v brew >/dev/null 2>&1; then
    if brew list assimp >/dev/null 2>&1; then
        echo "assimp already installed via Homebrew."
    else
        echo "Installing assimp via Homebrew..."
        brew install assimp
    fi
else
    echo "Homebrew not found -- install assimp manually (native lib, not just 'pip install pyassimp')." >&2
fi
"$PIP" install pyassimp

echo
if command -v blender >/dev/null 2>&1 || ls /Applications/Blender*.app >/dev/null 2>&1 || ls "$HOME"/Applications/Blender*.app >/dev/null 2>&1; then
    echo "Blender found -- export-blender should work."
else
    echo "No Blender install detected under /Applications. Install it from" \
         "https://www.blender.org/ for export-blender to work, or pass" \
         "--blender-executable explicitly."
fi

echo
echo "Setup complete. Activate with:"
echo "  source $VENV/bin/activate"
echo "Then run the CLI, e.g.:"
echo "  python -m app.cli extract-frames --video input.mp4 --out cache/frames"
echo "...or the GUI:"
echo "  python -m app.gui"
echo "See README_EXEC.md for the full command reference."
