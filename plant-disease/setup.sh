#!/usr/bin/env bash
# ==============================================================================
# Setup script for Plant Disease AI Platform (.venv environment setup)
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "================================================================="
echo " 🌿 PlantAI Vision - Environment Setup (.venv)"
echo "================================================================="

# 1. Detect Python 3
PYTHON_BIN=$(command -v python3 || command -v python)
if [ -z "$PYTHON_BIN" ]; then
    echo "[ERROR] Python 3 is not installed or not in PATH!"
    exit 1
fi
echo "[1/4] Found Python: $($PYTHON_BIN --version)"

# 2. Create .venv if it does not exist
if [ ! -d ".venv" ]; then
    echo "[2/4] Creating virtual environment (.venv)..."
    $PYTHON_BIN -m venv .venv
else
    echo "[2/4] Virtual environment (.venv) already exists."
fi

# 3. Activate .venv
echo "[3/4] Activating .venv and installing requirements..."
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

# 4. Verify installation
echo ""
echo "[4/4] Verifying PyTorch & Acceleration..."
python -c "import torch; print('PyTorch Version:', torch.__version__); print('CUDA Available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

python -c "from rag import get_rag; rag = get_rag(); print('RAG Engine verified successfully! Chunks:', len(rag.chunks))"

echo ""
echo "================================================================="
echo " ✅ SETUP COMPLETE!"
echo ""
echo " To activate your environment in any terminal:"
echo "   source .venv/bin/activate"
echo ""
echo " To launch the web application:"
echo "   streamlit run app.py"
echo "   (or: ./run.sh)"
echo "================================================================="
