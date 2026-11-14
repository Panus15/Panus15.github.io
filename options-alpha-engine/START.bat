@echo off
REM Double-click this. It runs from its own folder, so the working directory
REM of whatever shell launched it does not matter.
cd /d "%~dp0"

where python >nul 2>nul
if errorlevel 1 goto nopython

python -m tools.quickstart %*
goto end

:nopython
echo.
echo   Python was not found on PATH.
echo   Install it from https://www.python.org/downloads/ and tick
echo   "Add python.exe to PATH" in the first screen of the installer,
echo   then run this file again.

:end
echo.
pause
