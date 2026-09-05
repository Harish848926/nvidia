#!/usr/bin/env bash
# ==============================================================================
# Plant Disease Detection & AI Agronomy Advisory Launcher (NVIDIA DGX / Linux)
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 1. Virtual Environment Setup (Checks for or auto-creates venv)
if [ -d "venv" ]; then
    echo "[run.sh] Activating virtual environment (venv)..."
    source venv/bin/activate
elif [ -d ".venv" ]; then
    echo "[run.sh] Activating virtual environment (.venv)..."
    source .venv/bin/activate
else
    echo "[run.sh] No virtual environment found. Creating 'venv' to prevent PEP 668 conflicts..."
    (python3 -m venv venv || python -m venv venv) 2>/dev/null || true
    if [ -f "venv/bin/activate" ]; then
        source venv/bin/activate
        echo "[run.sh] Virtual environment 'venv' created and activated successfully."
    else
        echo "[run.sh] Note: Could not auto-create venv. Running with system Python."
    fi
fi

# Ensure Python 3
PYTHON_BIN=$(command -v python3 || command -v python)
if [ -z "$PYTHON_BIN" ]; then
    echo "[run.sh Error] Python is not installed or not in PATH."
    exit 1
fi

# 2. Check & Install Critical Requirements
if ! $PYTHON_BIN -c "import PIL, torch, streamlit" 2>/dev/null; then
    echo "[run.sh] Installing missing dependencies from requirements.txt..."
    pip install -r requirements.txt 2>/dev/null || pip install --break-system-packages -r requirements.txt
# Ensure ~/.local/bin is in PATH for user-installed pip binaries
export PATH="$HOME/.local/bin:$PATH"

# 3. Argument Dispatcher
MODE="${1:-app}"

case "$MODE" in
    app|"")
        echo "================================================================="
        echo " 🌿 Starting PlantAI Vision Web Application on DGX"
        echo " 👉 Local URL: http://localhost:8501"
        echo "================================================================="
        exec "$PYTHON_BIN" -m streamlit run app.py --server.port 8501 --server.address 0.0.0.0
        ;;
    train)
        shift
        echo "================================================================="
        echo " 🚀 Training Vision Model on DGX with NVIDIA Tensor Cores"
        echo "================================================================="
        exec $PYTHON_BIN train.py "$@"
        ;;
    predict)
        shift
        exec $PYTHON_BIN predict.py "$@"
        ;;
    eval|evaluate)
        shift
        exec $PYTHON_BIN evaluate.py "$@"
        ;;
    rag)
        shift
        exec $PYTHON_BIN rag.py "$@"
        ;;
    hf)
        shift
        exec $PYTHON_BIN hf_dataset.py "$@"
        ;;
    *)
        exec "$@"
        ;;
esac
