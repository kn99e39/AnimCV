#!/usr/bin/env bash
# Full-feature macOS build: bundles mmpose + depth-anything-v2 (CPU-only
# torch, since there's no CUDA on Mac -- Apple Silicon gets MPS
# acceleration automatically from the same CPU wheel, no special index
# needed). Run once, either by double-clicking this file in Finder (the
# .command extension makes macOS open it in Terminal.app instead of a
# text editor) or from a terminal: `bash mac/build_full_mac.command`.
#
# For just running the CLI from source (not bundling an exe -- much
# faster to iterate on), use mac/setup_mac.command instead; that
# script's header comment also documents the several mmcv/mmpose macOS
# build workarounds this one needs too (pip install below hits the same
# issues, just without them spelled out inline here).
#
# pyassimp (parse-rig) needs the native assimp library separately:
#   brew install assimp
# This script does not install it. Without it, parse-rig reports a normal
# error rather than crashing (see README.md).

set -euo pipefail

# Paths below (.venv_build_full, scripts/, src/, build_output/) are
# relative to the project root, not this script's location under mac/.
cd "$(dirname "${BASH_SOURCE[0]}")/.."

VENV=".venv_build_full"
if [ ! -d "$VENV" ]; then
    python3 -m venv "$VENV"
fi

echo "[1/4] Upgrading pip..."
"$VENV/bin/python" -m pip install --upgrade pip

echo "[2/4] Installing torch (CPU)..."
"$VENV/bin/python" -m pip install torch

echo "[3/4] Installing project + pose/depth extras + pyinstaller..."
"$VENV/bin/python" -m pip install -e ".[pose,depth]" pyinstaller

echo "[4/4] Running PyInstaller (this bundles torch + mmcv/mmengine/mmdet, expect several GB)..."
rm -rf build_output/mac_full build_output/_work_full build_output/_spec_full

"$VENV/bin/python" -m PyInstaller \
  --name motion-tool-full \
  --onedir --console --noconfirm --clean --contents-directory . \
  --distpath build_output/mac_full \
  --workpath build_output/_work_full \
  --specpath build_output/_spec_full \
  --paths src \
  --collect-all torch \
  --collect-all mmcv \
  --collect-all mmengine \
  --collect-all mmdet \
  --collect-all mmpose \
  --add-data "scripts:scripts" \
  --add-data "src:src" \
  src/app/cli.py

rm -rf build_output/_work_full build_output/_spec_full
find build_output/mac_full -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

echo
echo "Built: build_output/mac_full/motion-tool-full/motion-tool-full"
echo "NOTE: mmcv/mmdet use dynamic registry-based imports PyInstaller's"
echo "static analysis can miss. If estimate-pose fails at runtime with a"
echo "ModuleNotFoundError/KeyError from mmcv/mmdet's registry, that module"
echo "needs an explicit --hidden-import added to the PyInstaller call above"
echo "and a rebuild -- this PyInstaller-frozen bundle specifically was not"
echo "exercised against a real mmpose checkpoint (running from a plain venv"
echo "via mac/setup_mac.command was -- see README_EXEC.md's Mac support section)."
