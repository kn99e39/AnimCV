@echo off
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvarsall.bat" x64 >nul
chcp 65001
cl.exe > cl_out2.bin 2>&1
