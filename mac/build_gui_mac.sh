#!/usr/bin/env bash
# Builds the Tkinter GUI (src/ui/gui_app.py, entry point src/app/gui.py)
# as a standalone macOS app bundle, full-featured: mmpose + mmcv + mmdet +
# torch (CPU -- no CUDA on Mac, Apple Silicon gets MPS from the same
# wheel) + depth-anything-v2's dependencies all bundled, so estimate-pose
# works with no separate `pip install` on the target machine.
#
# Uses its own venv (.venv_build_gui), separate from mac/setup_mac.sh's
# `.venv` dev environment, so a build doesn't disturb an existing dev
# setup. The mmcv/mmdet/mmpose install recipe below is copied from
# setup_mac.sh, which documents *why* each step is needed (see its header
# comment) -- this was verified end to end there against a real
# downloaded RTMPose checkpoint from a plain venv; the PyInstaller-frozen
# bundle produced by this script has NOT been separately verified (no Mac
# was available when this script was written -- see README.md).
#
# Run once, from anywhere: bash mac/build_gui_mac.sh

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

find_python() {
    # Prefer an actual 3.11 over anything else, and look beyond just
    # PATH for it -- on a machine with several Pythons installed (system
    # Python, more than one Homebrew python@X.Y keg, pyenv, the
    # python.org installer, Anaconda...), the interpreter `python3`/
    # `python` resolves to on PATH is often NOT 3.11 even when 3.11 is
    # very much installed, just unlinked/shadowed/not the active pyenv
    # version. Absolute-path candidates below cover the common install
    # layouts without requiring 3.11 to be first on PATH. Kept in sync
    # with mac/setup_mac.sh's copy of this function.
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

    for candidate in python3.13 python3.12 python3; do
        if command -v "$candidate" >/dev/null 2>&1; then
            local version major minor
            version="$("$candidate" -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')"
            major="${version%%.*}"
            minor="${version##*.}"
            if [ "$major" -eq 3 ] && [ "$minor" -ge 11 ]; then
                echo "$candidate"
                return 0
            fi
        fi
    done
    return 1
}

PYTHON="$(find_python || true)"
if [ -z "$PYTHON" ]; then
    echo "No Python >=3.11 found on PATH or common install locations" >&2
    echo "(Homebrew python@3.11, pyenv, python.org installer)." >&2
    echo "Install one, e.g. 'brew install python@3.11', and re-run." >&2
    exit 1
fi
echo "Using $PYTHON ($("$PYTHON" --version))"
if ! "$PYTHON" -c 'import sys; sys.exit(0 if sys.version_info[:2] == (3, 11) else 1)'; then
    echo "WARNING: this is not Python 3.11. The mmpose/mmcv/mmdet recipe" >&2
    echo "below was only verified against 3.11 (see mac/setup_mac.sh)." >&2
fi

VENV=".venv_build_gui"
if [ ! -d "$VENV" ]; then
    echo "[1/11] Creating venv at $VENV..."
    "$PYTHON" -m venv "$VENV"
else
    echo "[1/11] Reusing existing venv at $VENV..."
fi
PIP="$VENV/bin/pip"
PY="$VENV/bin/python"

echo "[2/11] Upgrading pip..."
"$PY" -m pip install --upgrade pip

echo "[3/11] Pinning numpy<2 / opencv-python<4.10 (mmpose's xtcocotools native ABI needs numpy<2)..."
"$PIP" install "numpy<2" "opencv-python<4.10"

echo "[4/11] Installing torch (CPU)..."
"$PIP" install torch

echo "[5/11] Installing mmengine alone (mmdet comes later, after mmcv is built)..."
"$PIP" install mmengine

echo "[6/11] Building mmcv from source with torch already importable (no prebuilt macOS wheel exists upstream)..."
"$PIP" install "setuptools<81" wheel
"$PIP" install --no-build-isolation --no-cache-dir --force-reinstall --no-deps "mmcv>=2.0.0rc4,<2.2.0"

echo "[7/11] Installing mmdet (mmcv already satisfies its requirement, won't be rebuilt)..."
"$PIP" install mmdet

echo "[8/11] Installing mmpose (--no-build-isolation works around its chumpy dependency's legacy setup.py)..."
"$PIP" install --no-build-isolation mmpose

echo "[9/11] Installing project + depth extra + pyinstaller..."
"$PIP" install -e ".[depth]" pyinstaller matplotlib

echo "[10/11] Verifying tkinter is available (needed for the GUI)..."
if ! "$PY" -c "import tkinter" >/dev/null 2>&1; then
    if command -v brew >/dev/null 2>&1; then
        py_minor="$("$PY" -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')"
        echo "tkinter missing -- installing python-tk@$py_minor via Homebrew..."
        brew install "python-tk@$py_minor" || {
            echo "Could not auto-install python-tk@$py_minor. Install it manually," >&2
            echo "then re-run this script -- the GUI needs it to build." >&2
            exit 1
        }
        if ! "$PY" -c "import tkinter" >/dev/null 2>&1; then
            echo "tkinter still not importable after installing python-tk. This" >&2
            echo "usually means $PYTHON isn't the Homebrew python that formula" >&2
            echo "targets -- check 'brew info python-tk@$py_minor' and re-run" >&2
            echo "with a matching python3 on PATH." >&2
            exit 1
        fi
    else
        echo "tkinter not available and Homebrew not found -- can't build the GUI." >&2
        exit 1
    fi
else
    echo "tkinter already available."
fi

echo "[11/11] Running PyInstaller (GUI entry point: src/app/gui.py)..."
rm -rf build_output/mac_gui build_output/_work_gui motion-tool-gui.spec

# --windowed is what makes PyInstaller emit an actual .app bundle on
# macOS (--onedir alone just gives a plain Unix binary in a folder, the
# same as Linux) -- but a windowed .app has no attached terminal, so
# stdout/stderr (including tracebacks) don't go anywhere visible; they
# land in Console.app (filter by process name) or you can run the
# binary inside the bundle directly from Terminal to see them:
#   ./build_output/mac_gui/motion-tool-gui.app/Contents/MacOS/motion-tool-gui
#
# --specpath is intentionally omitted (left at its default, the current
# directory) so the --add-data paths below can stay relative: PyInstaller
# resolves relative --add-data sources against --specpath, not the cwd,
# so pointing --specpath somewhere else (e.g. build_output/_spec_gui, as
# earlier versions of this script did) makes relative "scripts:scripts"
# unresolvable and requires spelling out an absolute path instead. The
# stray motion-tool-gui.spec file this leaves in the project root is
# removed below.
"$PY" -m PyInstaller \
  --name motion-tool-gui \
  --onedir --windowed --noconfirm --clean --contents-directory . \
  --distpath build_output/mac_gui \
  --workpath build_output/_work_gui \
  --paths src \
  --hidden-import tkinter \
  --collect-all torch \
  --collect-all mmcv \
  --collect-all mmengine \
  --collect-all mmdet \
  --collect-all mmpose \
  --collect-all xtcocotools \
  --collect-all pycocotools \
  --add-data "scripts:scripts" \
  --add-data "src:src" \
  src/app/gui.py

rm -rf build_output/_work_gui motion-tool-gui.spec
find build_output/mac_gui -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

cp USAGE_GUI.md USAGE_GUI.ko.md build_output/mac_gui/

echo
echo "Built: build_output/mac_gui/motion-tool-gui.app"
echo "(USAGE_GUI.md / USAGE_GUI.ko.md copied alongside it for anyone who gets"
echo "just this folder.)"
echo "Double-clickable from Finder. No terminal window attached -- if it"
echo "doesn't come up, run the binary inside the bundle directly to see"
echo "errors: ./build_output/mac_gui/motion-tool-gui.app/Contents/MacOS/motion-tool-gui"
echo "NOT yet verified end-to-end on a real Mac -- this is the same recipe"
echo "as mac/setup_mac.sh (which was verified from a plain venv), just"
echo "additionally run through PyInstaller. Run the built app for real and"
echo "exercise estimate-pose before trusting it."
echo "retarget/parse-rig still need the native assimp library separately"
echo "(brew install assimp) -- not bundled either."
