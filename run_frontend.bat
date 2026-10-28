@echo off
REM =================================================================================
REM  Runner script for the ExoplanetAI Streamlit Frontend Web App
REM =================================================================================

cd /d "%~dp0"

echo Starting ExoplanetAI Frontend Web Application...

python run.py frontend

pause
