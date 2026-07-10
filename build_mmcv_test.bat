@echo off
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvarsall.bat" x64 >nul
set TORCH_DONT_CHECK_COMPILER_ABI=1
set DISTUTILS_USE_SDK=1
set CL=/Zc:preprocessor
cd /d "C:\Users\dna10\AppData\Local\Temp\mmcv-2.2.0"
"C:\Projects\AnimCV\.venv_build_full\Scripts\python.exe" -m pip install . --no-build-isolation
