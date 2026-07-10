# Builds a standalone Windows executable of the motion-tool CLI into
# build_output/windows using PyInstaller. Run once, from anywhere:
#   windows\build_windows.ps1
#
# Only the base `dependencies` from pyproject.toml (numpy, opencv-python,
# pyyaml) are bundled -- estimate-pose's mmpose/depth-anything-v2 extras
# and parse-rig's pyassimp native lib are NOT bundled (heavy/optional,
# see README.md). Those commands still work from the exe if the extras
# are installed system-wide and importable, but a from-scratch clean
# machine will need `pip install -e ".[pose,depth]"` in a real venv for
# them, same as running from source.
#
# export-blender shells out to a real Blender install (unaffected by
# this packaging) and runs scripts/apply_motion.py under Blender's own
# Python -- that script needs the actual src/ tree on disk (it inserts
# it into sys.path itself), so src/ is bundled as plain data files
# alongside the exe, not just frozen into the bundle's bytecode.

$ErrorActionPreference = "Stop"

# Paths below (.venv_build, scripts/, src/, build_output/) are relative to
# the project root, not this script's location under windows/ -- this file
# lives there specifically so it's never mistaken for the Mac scripts
# under mac/, so always hop back to the repo root first.
Set-Location (Join-Path $PSScriptRoot "..")

$venv = ".venv_build"
if (-not (Test-Path $venv)) {
    python -m venv $venv
    & "$venv\Scripts\python.exe" -m pip install --upgrade pip
}
& "$venv\Scripts\python.exe" -m pip install -e . pyinstaller

Remove-Item -Recurse -Force build_output -ErrorAction SilentlyContinue

$scriptsAbs = (Resolve-Path scripts).Path
$srcAbs = (Resolve-Path src).Path

& "$venv\Scripts\python.exe" -m PyInstaller `
  --name motion-tool `
  --onedir `
  --console `
  --noconfirm `
  --clean `
  --contents-directory . `
  --distpath build_output/windows `
  --workpath build_output/_work `
  --specpath build_output/_spec `
  --paths src `
  --add-data "$scriptsAbs;scripts" `
  --add-data "$srcAbs;src" `
  src/app/cli.py

Remove-Item -Recurse -Force build_output/_work, build_output/_spec -ErrorAction SilentlyContinue
Get-ChildItem -Recurse -Directory -Filter "__pycache__" build_output/windows | Remove-Item -Recurse -Force

Write-Host "Built: build_output/windows/motion-tool/motion-tool.exe"
