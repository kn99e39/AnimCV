@echo off
REM Builds the Tkinter GUI (src/ui/gui_app.py, entry point src/app/gui.py)
REM as a standalone Windows exe, full-featured: mmpose + mmcv + mmdet +
REM torch (CUDA) + depth-anything-v2's dependencies all bundled, so
REM estimate-pose works inside the exe with no separate `pip install` on
REM the target machine.
REM
REM This reuses every fix verified in the CLI's full-feature build (see
REM result/result_windows_full_build.txt) -- mmcv/mmdet/mmengine haven't
REM been updated since ~2023 and collide with a modern Windows/Python/
REM torch/CUDA toolchain in several independent, non-obvious ways. Do not
REM simplify this script without re-testing against a real GUI run.
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

REM Paths below (.venv_build_gui, scripts\, src\, build_output\) are
REM relative to the project root, not this script's location under
REM windows\ -- always hop back to the repo root first.
cd /d "%~dp0.."

if not defined TORCH_CUDA_TAG set TORCH_CUDA_TAG=cu130

REM --- Locate a Visual Studio install with the C++ compiler (vcvarsall.bat) ---
set VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe
if not exist "%VSWHERE%" (
    echo Visual Studio Installer not found. Install VS 2022+ Community or
    echo Build Tools with the "Desktop development with C++" workload first.
    goto :error
)
for /f "usebackq tokens=*" %%i in (`"!VSWHERE!" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set VSPATH=%%i
if not defined VSPATH (
    echo No Visual Studio install with the C++ compiler ^(VC.Tools.x86.x64^)
    echo was found. Install it via the Visual Studio Installer first.
    goto :error
)
set VCVARSALL=%VSPATH%\VC\Auxiliary\Build\vcvarsall.bat
call "%VCVARSALL%" x64 >nul
if errorlevel 1 goto :error

REM --- Python 3.11 venv (mmcv is incompatible with 3.13+, see header) ---
set VENV=.venv_build_gui
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

echo [1/8] Upgrading pip...
call %VENV%\Scripts\python.exe -m pip install --upgrade pip
if errorlevel 1 goto :error

echo [2/8] Installing base deps (numpy pinned ^<2: xtcocotools' wheel is
echo compiled against the old numpy 1.x ABI and errors under 2.x;
echo opencv-python pinned ^<5 since 5.x hard-requires numpy^>=2)...
call %VENV%\Scripts\python.exe -m pip install "numpy<2" "opencv-python<5" pyyaml
if errorlevel 1 goto :error

echo [3/8] Installing torch (CUDA, tag=%TORCH_CUDA_TAG%)...
call %VENV%\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/%TORCH_CUDA_TAG%
if errorlevel 1 goto :error

echo [4/8] Downloading + patching mmcv 2.1.0 source (mmdet requires
echo mmcv^<2.2.0; its setup.py hardcodes /std:c++17 for any torch^>1.12.1,
echo which is one C++ standard behind what current torch's headers need)...
set MMCV_SRC=%TEMP%\animcv_mmcv_2.1.0_src
if exist "%MMCV_SRC%" rmdir /s /q "%MMCV_SRC%"
mkdir "%MMCV_SRC%"
call %VENV%\Scripts\python.exe -c "import urllib.request; urllib.request.urlretrieve('https://files.pythonhosted.org/packages/source/m/mmcv/mmcv-2.1.0.tar.gz', r'%MMCV_SRC%\mmcv.tar.gz')"
if errorlevel 1 goto :error
call %VENV%\Scripts\python.exe -c "import tarfile; tarfile.open(r'%MMCV_SRC%\mmcv.tar.gz').extractall(r'%MMCV_SRC%')"
if errorlevel 1 goto :error
call %VENV%\Scripts\python.exe -c "import pathlib; p=pathlib.Path(r'%MMCV_SRC%\mmcv-2.1.0\setup.py'); s=p.read_text(); s2=s.replace(\"extra_compile_args['cxx'] = ['/std:c++17']\", \"extra_compile_args['cxx'] = ['/std:c++20']\"); assert s2 ^!= s, 'patch target not found -- mmcv setup.py changed upstream, re-check this script'; p.write_text(s2)"
if errorlevel 1 goto :error

echo [5/8] Compiling mmcv 2.1.0 from the patched source (this is the slow
echo step -- native C++/CUDA compile of ~100+ files, took 20-25 min when
echo last verified; ninja is not installed so distutils compiles serially)...
set TORCH_DONT_CHECK_COMPILER_ABI=1
set DISTUTILS_USE_SDK=1
set CL=/Zc:preprocessor
pushd "%MMCV_SRC%\mmcv-2.1.0"
"%~dp0..\%VENV%\Scripts\python.exe" -m pip install . --no-build-isolation
if errorlevel 1 (
    popd
    goto :error
)
popd

echo [6/8] Installing mmdet, mmpose, depth extras, GUI extras, pyinstaller...
call %VENV%\Scripts\python.exe -m pip install mmdet mmpose --no-build-isolation
if errorlevel 1 goto :error
call %VENV%\Scripts\python.exe -m pip install -e ".[depth]" pyinstaller matplotlib
if errorlevel 1 goto :error

echo [7/8] Verifying tkinter is available (bundled with python.org's
echo Windows installer by default, but not guaranteed on every install)...
call %VENV%\Scripts\python.exe -c "import tkinter" 2>nul
if errorlevel 1 (
    echo tkinter is not available in this Python install. Reinstall Python
    echo 3.11 from python.org with the default options -- tcl/tk is
    echo included unless deselected -- then re-run this script.
    goto :error
)

echo [8/8] Running PyInstaller (GUI entry point: src/app/gui.py)...
if exist build_output\windows_gui rmdir /s /q build_output\windows_gui
if exist build_output\_work_gui rmdir /s /q build_output\_work_gui
if exist motion-tool-gui.spec del /q motion-tool-gui.spec

REM --specpath is intentionally omitted (left at its default, the
REM current directory) so the --add-data paths below can stay relative
REM instead of machine-specific absolute ones: PyInstaller resolves
REM relative --add-data sources against --specpath, not the cwd, so
REM pointing --specpath somewhere else (e.g. build_output\_spec_gui, as
REM an earlier version of this script did) makes relative
REM "scripts;scripts" unresolvable and forces spelling out an absolute
REM path instead -- which then bakes in whoever built it, unhelpful for
REM anyone else checking this script out and running it themselves. The
REM stray motion-tool-gui.spec file this leaves in the project root is
REM removed below.
call %VENV%\Scripts\python.exe -m PyInstaller ^
  --name motion-tool-gui ^
  --onedir --console --noconfirm --clean --contents-directory . ^
  --distpath build_output\windows_gui ^
  --workpath build_output\_work_gui ^
  --paths src ^
  --hidden-import tkinter ^
  --collect-all torch ^
  --collect-all mmcv ^
  --collect-all mmengine ^
  --collect-all mmdet ^
  --collect-all mmpose ^
  --collect-all xtcocotools ^
  --collect-all pycocotools ^
  --add-data "scripts;scripts" ^
  --add-data "src;src" ^
  src\app\gui.py
if errorlevel 1 goto :error

rmdir /s /q build_output\_work_gui
if exist motion-tool-gui.spec del /q motion-tool-gui.spec
for /d /r build_output\windows_gui %%d in (__pycache__) do @if exist "%%d" rmdir /s /q "%%d"

copy /y USAGE_GUI.md build_output\windows_gui\motion-tool-gui\USAGE_GUI.md >nul
copy /y USAGE_GUI.ko.md build_output\windows_gui\motion-tool-gui\USAGE_GUI.ko.md >nul

echo.
echo Built: build_output\windows_gui\motion-tool-gui\motion-tool-gui.exe
echo (USAGE_GUI.md / USAGE_GUI.ko.md copied alongside it for anyone who gets
echo just this folder.)
echo Built with --console so error output is visible during first runs;
echo switch to --windowed above once you've verified it end-to-end if you
echo don't want a console window alongside the GUI.
echo retarget/parse-rig still need pyassimp's native assimp library
echo separately (not bundled) -- see README.md.
goto :eof

:error
echo.
echo Build failed -- see the error above.
exit /b 1
