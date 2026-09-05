@echo off
title PlantAI Vision - NVIDIA DGX Plant Disease Advisory
echo =================================================================
echo   Plant Disease Detection ^& AI Agronomy Advisory
echo   NVIDIA Platform / PyTorch + RAG Knowledge Engine
echo =================================================================
cd /d "%~dp0"

echo [1/2] Checking Python environment...
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH!
    pause
    exit /b 1
)

echo [2/2] Launching Streamlit Web Application...
echo Local URL: http://localhost:8501
echo Press Ctrl+C in this window to stop the server.
echo =================================================================
python -m streamlit run app.py --server.port 8501
pause
