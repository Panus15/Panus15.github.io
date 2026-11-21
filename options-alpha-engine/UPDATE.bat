@echo off
setlocal EnableExtensions
REM Double-click to bring the CODE in this folder up to date from GitHub.
REM
REM It never touches your records. It copies only the code folders and the
REM top-level docs listed below, and it never deletes anything (no /MIR, no
REM /PURGE). So btc.jsonl, *.prices.json, daily.log, daily-job.bat,
REM dashboard.html, prices\ and holdings\ stay exactly as they are, and the
REM scheduled daily job keeps running against them.

set "BRANCH=claude/ml-options-trading-engine-b471vm"
set "URL=https://github.com/Panus15/Panus15.github.io/archive/refs/heads/%BRANCH%.zip"

REM A .bat file that is overwritten while it runs misreads its own next line,
REM and this one overwrites itself. So run from a copy in %TEMP%, and hand it
REM the real folder. No CALL: control never comes back to the original file.
if /i not "%~1"=="--from-temp" (
    copy /y "%~f0" "%TEMP%\oae_update.bat" >nul
    "%TEMP%\oae_update.bat" --from-temp "%~dp0"
)

set "DEST=%~2"
REM A trailing backslash before a closing quote escapes the quote for robocopy.
if "%DEST:~-1%"=="\" set "DEST=%DEST:~0,-1%"
set "WORK=%TEMP%\oae_update_work"
set "ZIP=%TEMP%\oae_update.zip"

echo.
echo   Updating the code in:
echo     %DEST%
echo   Your records (btc.jsonl, daily.log, prices\ ...) are not touched.
echo.

if not exist "%DEST%\engine\pricing.py" (
    echo   !! This does not look like the options-alpha-engine folder.
    echo      Put UPDATE.bat in the options-alpha-engine folder and run it there.
    goto done
)

if exist "%WORK%" rmdir /s /q "%WORK%"
if exist "%ZIP%" del /q "%ZIP%"

echo   [1/3] downloading ...
powershell -NoProfile -ExecutionPolicy Bypass -Command "$ProgressPreference='SilentlyContinue'; Invoke-WebRequest -UseBasicParsing -Uri '%URL%' -OutFile '%ZIP%'"
if errorlevel 1 (
    echo   !! Download failed. Check the internet connection and try again.
    goto done
)

echo   [2/3] unpacking ...
powershell -NoProfile -ExecutionPolicy Bypass -Command "Expand-Archive -Force -Path '%ZIP%' -DestinationPath '%WORK%'"
if errorlevel 1 (
    echo   !! Could not unpack the download.
    goto done
)

set "SRC="
for /d %%D in ("%WORK%\*") do set "SRC=%%D\options-alpha-engine"
if not defined SRC goto badsrc
if not exist "%SRC%\engine\pricing.py" goto badsrc

echo   [3/3] copying code ...
set "FAILED="
for %%F in (engine models tools tests mutation docs) do (
    robocopy "%SRC%\%%F" "%DEST%\%%F" /E /XD __pycache__ /NFL /NDL /NJH /NJS /NP >nul
    if errorlevel 8 set "FAILED=1"
)
robocopy "%SRC%" "%DEST%" *.md *.py *.txt START.bat UPDATE.bat /NFL /NDL /NJH /NJS /NP >nul
if errorlevel 8 set "FAILED=1"
if defined FAILED (
    echo   !! Some files could not be copied. Close any open file from this
    echo      folder and run UPDATE.bat again.
    goto cleanup
)

echo.
echo   Updated. Checking it still runs:
cd /d "%DEST%"
python -c "import engine.pricing, tools.daily, tools.option_check; print('  code imports fine')"
if errorlevel 1 echo   !! The new code did not import. Tell Claude what it printed above.
if exist "btc.jsonl" python -m tools.daily status --ledger btc.jsonl --holdings=
goto cleanup

:badsrc
echo   !! The download did not contain the expected folder.

:cleanup
if exist "%WORK%" rmdir /s /q "%WORK%"
if exist "%ZIP%" del /q "%ZIP%"

:done
echo.
pause
exit /b
