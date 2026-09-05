@echo off
title Setup - Plant Disease Detection AI
echo =================================================================
echo   Setting up Plant Disease Detection ^& AI Agronomy Advisory
echo =================================================================
cd /d "%~dp0"

echo [1/3] Installing Python Dependencies from requirements.txt...
pip install -r requirements.txt
if errorlevel 1 (
    echo [WARNING] Some dependencies had warnings or issues during installation.
)

echo.
echo [2/3] Verifying Hardware ^& PyTorch Acceleration...
python -c "import torch; print('PyTorch Version:', torch.__version__); print('CUDA Available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

echo.
echo [3/3] Verifying RAG Knowledge Engine...
python -c "from rag import get_rag; rag = get_rag(); print('RAG Engine verified successfully! Chunks:', len(rag.chunks))"

echo.
echo =================================================================
echo   SETUP COMPLETED! You can now start the app by running run.bat
echo =================================================================
pause
