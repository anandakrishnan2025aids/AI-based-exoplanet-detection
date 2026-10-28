@echo off
REM =================================================================================
REM  Runner script for the ExoMiner Pipeline on Windows.
REM =================================================================================

cd /d "%~dp0"

echo Starting ExoMiner Pipeline...

REM Activate your conda environment if conda is available
call conda activate exoplnt_dl 2>nul

REM Run the pipeline via the unified runner
python run.py pipeline %*

echo.
echo ExoMiner Pipeline run has finished.
pause