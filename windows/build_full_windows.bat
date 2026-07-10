@echo off
REM Full-feature Windows build: bundles mmpose + depth-anything-v2 (torch
REM with CUDA) into the exe, on top of the base build_windows.ps1 does.
REM Run once, from anywhere (double-click, or `windows\build_full_windows.bat`).
REM
REM torch CUDA wheel tag: override by setting TORCH_CUDA_TAG before running,
REM e.g. `set TORCH_CUDA_TAG=cu128 && build_full_windows.bat`. Defaults to
REM cu130. Note: which tags exist depends on your Python version -- e.g.
REM cu121 only ships cp39-cp312 wheels, so it fails outright on a Python
REM 3.14 venv; cu128/cu130 are the ones with cp314 wheels as of this
REM writing. If pip reports "no matching distribution", check
REM https://download.pytorch.org/whl/<tag>/torch/ for what's actually
REM built for your Python version and re-run with TORCH_CUDA_TAG set.
REM
REM pyassimp (parse-rig) needs the native assimp shared library separately;
REM this script does not install it. On Windows, either put assimp.dll on
REM PATH yourself, or skip parse-rig in the resulting exe (it will report a
REM normal error, not crash, if the native lib is missing -- see README.md).

setlocal enabledelayedexpansion

REM Paths below (.venv_build_full, scripts\, src\, build_output\) are
REM relative to the project root, not this script's location under
REM windows\ -- this file lives there specifically so it's never mistaken
REM for the Mac scripts under mac\, so always hop back to the repo root.
cd /d "%~dp0.."

if not defined TORCH_CUDA_TAG set TORCH_CUDA_TAG=cu130

set VENV=.venv_build_full
if not exist %VENV% (
    python -m venv %VENV%
    if errorlevel 1 goto :error
)

echo [1/4] Upgrading pip...
call %VENV%\Scripts\python.exe -m pip install --upgrade pip
if errorlevel 1 goto :error

echo [2/4] Installing torch (CUDA, tag=%TORCH_CUDA_TAG%)...
call %VENV%\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/%TORCH_CUDA_TAG%
if errorlevel 1 goto :error

echo [3/4] Installing project + pose/depth extras + pyinstaller...
call %VENV%\Scripts\python.exe -m pip install -e ".[pose,depth]" pyinstaller
if errorlevel 1 goto :error

echo [4/4] Running PyInstaller (this bundles torch + mmcv/mmengine/mmdet, expect several GB)...
if exist build_output\windows_full rmdir /s /q build_output\windows_full
if exist build_output\_work_full rmdir /s /q build_output\_work_full
if exist build_output\_spec_full rmdir /s /q build_output\_spec_full

call %VENV%\Scripts\python.exe -m PyInstaller ^
  --name motion-tool-full ^
  --onedir --console --noconfirm --clean --contents-directory . ^
  --distpath build_output\windows_full ^
  --workpath build_output\_work_full ^
  --specpath build_output\_spec_full ^
  --paths src ^
  --collect-all torch ^
  --collect-all mmcv ^
  --collect-all mmengine ^
  --collect-all mmdet ^
  --collect-all mmpose ^
  --add-data "scripts;scripts" ^
  --add-data "src;src" ^
  src\app\cli.py
if errorlevel 1 goto :error

rmdir /s /q build_output\_work_full build_output\_spec_full
for /d /r build_output\windows_full %%d in (__pycache__) do @if exist "%%d" rmdir /s /q "%%d"

echo.
echo Built: build_output\windows_full\motion-tool-full\motion-tool-full.exe
echo NOTE: mmcv/mmdet use dynamic registry-based imports PyInstaller's static
echo analysis can miss. If estimate-pose fails at runtime with a
echo ModuleNotFoundError/KeyError from mmcv/mmdet's registry, that module
echo needs an explicit --hidden-import added to the PyInstaller call above
echo and a rebuild -- this was not exercised end-to-end against a real
echo mmpose checkpoint before shipping this script.
goto :eof

:error
echo.
echo Build failed -- see the error above.
exit /b 1
