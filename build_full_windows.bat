@echo off
REM Full-feature Windows build: bundles mmpose + mmcv + mmdet + torch
REM (CUDA) + depth-anything-v2's dependencies into the exe. This was
REM iterated against real failures until an actual mmpose checkpoint
REM loaded and ran inside the built exe -- see
REM result/result_windows_full_build.txt for the full story of what
REM broke and why each fix below is needed. Do not simplify this script
REM without re-testing; every step here exists because a simpler version
REM failed for a specific, verified reason.
REM
REM Prerequisites this script does NOT install for you:
REM   - Python 3.11 (via `py -3.11`). mmcv's setup.py uses an
REM     exec()+locals() pattern to read its own version file that breaks
REM     under Python 3.13+'s new locals() semantics (PEP 667) -- mmcv is
REM     unmaintained (last release ~2023) and never fixed this. Install
REM     with: winget install --id Python.Python.3.11
REM   - A Visual Studio install (Community/Build Tools, 2022 or newer)
REM     with the "Desktop development with C++" workload. mmcv 2.x has
REM     no prebuilt Windows wheel for any modern torch/Python combo (its
REM     official wheel index stops at mmcv 1.1.5 / torch 1.6 / 2020), so
REM     it must compile from source.
REM   - CUDA Toolkit (nvcc), matching-ish the CUDA version of the torch
REM     wheel below. Just the GPU driver is not enough to compile CUDA
REM     extensions. https://developer.nvidia.com/cuda-downloads
REM
REM torch CUDA wheel tag: override by setting TORCH_CUDA_TAG before
REM running. Defaults to cu130. Which tags exist depends on your Python
REM version -- e.g. cu121 only ships cp39-cp312 wheels. Check
REM https://download.pytorch.org/whl/<tag>/torch/ if this fails.

setlocal enabledelayedexpansion
if not defined TORCH_CUDA_TAG set TORCH_CUDA_TAG=cu130

REM --- Locate a Visual Studio install with the C++ compiler (vcvarsall.bat) ---
set VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe
if not exist "%VSWHERE%" (
    echo Visual Studio Installer not found. Install VS 2022+ Community or
    echo Build Tools with the "Desktop development with C++" workload first.
    goto :error
)
for /f "usebackq tokens=*" %%i in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set VSPATH=%%i
if not defined VSPATH (
    echo No Visual Studio install with the C++ (VC.Tools.x86.x64) workload
    echo was found. Install it via the Visual Studio Installer first.
    goto :error
)
set VCVARSALL=%VSPATH%\VC\Auxiliary\Build\vcvarsall.bat
call "%VCVARSALL%" x64 >nul
if errorlevel 1 goto :error

REM --- Python 3.11 venv (mmcv is incompatible with 3.13+, see header) ---
set VENV=.venv_build_full
where py >nul 2>nul
if errorlevel 1 (
    echo `py` launcher not found. Install Python 3.11 first:
    echo   winget install --id Python.Python.3.11
    goto :error
)
if not exist %VENV% (
    py -3.11 -m venv %VENV%
    if errorlevel 1 (
        echo Python 3.11 is not installed. Run:
        echo   winget install --id Python.Python.3.11
        goto :error
    )
)

echo [1/7] Upgrading pip...
call %VENV%\Scripts\python.exe -m pip install --upgrade pip
if errorlevel 1 goto :error

echo [2/7] Installing base deps (numpy pinned ^<2: xtcocotools' wheel is
echo compiled against the old numpy 1.x ABI and segfaults/errors under 2.x;
echo opencv-python pinned ^<5 since 5.x hard-requires numpy^>=2)...
call %VENV%\Scripts\python.exe -m pip install "numpy<2" "opencv-python<5" pyyaml
if errorlevel 1 goto :error

echo [3/7] Installing torch (CUDA, tag=%TORCH_CUDA_TAG%)...
call %VENV%\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/%TORCH_CUDA_TAG%
if errorlevel 1 goto :error

echo [4/7] Downloading + patching mmcv 2.1.0 source (mmdet requires
echo mmcv^<2.2.0; its setup.py hardcodes /std:c++17 for any torch^>1.12.1,
echo which is one C++ standard behind what current torch's headers need)...
set MMCV_SRC=%TEMP%\animcv_mmcv_2.1.0_src
if exist "%MMCV_SRC%" rmdir /s /q "%MMCV_SRC%"
mkdir "%MMCV_SRC%"
call %VENV%\Scripts\python.exe -c "import urllib.request; urllib.request.urlretrieve('https://files.pythonhosted.org/packages/source/m/mmcv/mmcv-2.1.0.tar.gz', r'%MMCV_SRC%\mmcv.tar.gz')"
if errorlevel 1 goto :error
call %VENV%\Scripts\python.exe -c "import tarfile; tarfile.open(r'%MMCV_SRC%\mmcv.tar.gz').extractall(r'%MMCV_SRC%')"
if errorlevel 1 goto :error
call %VENV%\Scripts\python.exe -c "import pathlib; p=pathlib.Path(r'%MMCV_SRC%\mmcv-2.1.0\setup.py'); s=p.read_text(); s2=s.replace(\"extra_compile_args['cxx'] = ['/std:c++17']\", \"extra_compile_args['cxx'] = ['/std:c++20']\"); assert s2 != s, 'patch target not found -- mmcv setup.py changed upstream, re-check build_full_windows.bat'; p.write_text(s2)"
if errorlevel 1 goto :error

echo [5/7] Compiling mmcv 2.1.0 from the patched source (this is the slow
echo step -- native C++/CUDA compile of ~100+ files, took 20-25 min when
echo last verified; ninja is not installed so distutils compiles serially)...
set TORCH_DONT_CHECK_COMPILER_ABI=1
set DISTUTILS_USE_SDK=1
set CL=/Zc:preprocessor
pushd "%MMCV_SRC%\mmcv-2.1.0"
"%~dp0%VENV%\Scripts\python.exe" -m pip install . --no-build-isolation
if errorlevel 1 (
    popd
    goto :error
)
popd

echo [6/7] Installing mmdet, mmpose, depth extras, pyinstaller...
call %VENV%\Scripts\python.exe -m pip install mmdet mmpose --no-build-isolation
if errorlevel 1 goto :error
call %VENV%\Scripts\python.exe -m pip install -e ".[depth]" pyinstaller matplotlib
if errorlevel 1 goto :error

echo [7/7] Running PyInstaller...
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
  --collect-all xtcocotools ^
  --collect-all pycocotools ^
  --add-data "scripts;scripts" ^
  --add-data "src;src" ^
  src\app\cli.py
if errorlevel 1 goto :error

rmdir /s /q build_output\_work_full build_output\_spec_full
for /d /r build_output\windows_full %%d in (__pycache__) do @if exist "%%d" rmdir /s /q "%%d"

echo.
echo Built: build_output\windows_full\motion-tool-full\motion-tool-full.exe
echo Verified end-to-end against a real RTMPose-s checkpoint: estimate-pose,
echo build-motion, optimize, and export-blender (real local Blender) all
echo ran successfully through this exe. retarget/parse-rig still need
echo pyassimp's native assimp library separately (not bundled) -- see
echo README.md.
goto :eof

:error
echo.
echo Build failed -- see the error above.
exit /b 1
