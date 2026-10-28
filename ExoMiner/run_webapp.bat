@echo off
REM =================================================================================
REM  Runner script for the ExoMiner Interactive Vetting Catalog Web App.
REM =================================================================================

cd /d "%~dp0"

echo Starting ExoMiner Interactive Vetting Catalog Web Application...

REM If root runner exists, delegate to unified runner
if exist "..\run.py" (
    python ..\run.py webapp %*
    goto end
)

REM Activate conda environment
call conda activate exoplnt_dl 2>nul

if exist "%USERPROFILE%\anaconda3\envs\exoplnt_dl\python.exe" (
    set "PY_EXE=%USERPROFILE%\anaconda3\envs\exoplnt_dl\python.exe"
) else (
    set "PY_EXE=python"
)

REM Run Dash web application
"%PY_EXE%" exominer_vetting_pc_catalog_dash-render-web-app\vetting_tce_catalog_exominer_dash_app.py

:end
pause
