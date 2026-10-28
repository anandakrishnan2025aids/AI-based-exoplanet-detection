@echo off
REM =================================================================================
REM  Runner script for the ExoMiner Pipeline on Windows.
REM =================================================================================

cd /d "%~dp0"

echo Starting ExoMiner Pipeline...

REM If root runner exists, delegate to unified runner for robust environment & path handling
if exist "..\run.py" (
    python ..\run.py pipeline %*
    goto end
)

REM Activate conda environment
call conda activate exoplnt_dl 2>nul

REM Locate python in exoplnt_dl if conda command is not in PATH
if exist "%USERPROFILE%\anaconda3\envs\exoplnt_dl\python.exe" (
    set "PY_EXE=%USERPROFILE%\anaconda3\envs\exoplnt_dl\python.exe"
) else (
    set "PY_EXE=python"
)

set PYTHON_SCRIPT=exominer_pipeline\run_pipeline.py

"%PY_EXE%" %PYTHON_SCRIPT% ^
  --output_dir "..\output" ^
  --tic_ids_fp "inputs\input_tics.csv" ^
  --data_collection_mode "2min" ^
  --num_processes 1 ^
  --num_jobs 1

:end
echo.
echo ExoMiner Pipeline run has finished.
pause