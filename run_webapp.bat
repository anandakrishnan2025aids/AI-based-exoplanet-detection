@echo off
REM =================================================================================
REM  Runner script for the ExoMiner Interactive Vetting Catalog Web App.
REM =================================================================================

cd /d "%~dp0"

echo Starting ExoMiner Interactive Vetting Catalog Web Application...

REM Activate conda environment if conda is available
call conda activate exoplnt_dl 2>nul

REM Run Dash web application via the unified runner
python run.py webapp %*

pause
