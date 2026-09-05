#!/usr/bin/env bash
# ==============================================================================
# Apptainer / Singularity Launcher for NVIDIA DGX Supercomputers
# Enables rootless, containerized execution with full NVIDIA GPU passthrough (--nv)
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 1. Detect Apptainer or Singularity
if command -v apptainer >/dev/null 2>&1; then
    CONTAINER_CMD="apptainer"
elif command -v singularity >/dev/null 2>&1; then
    CONTAINER_CMD="singularity"
else
    echo "[Apptainer Error] Neither 'apptainer' nor 'singularity' was found in PATH."
    echo "Please ensure Apptainer or Singularity is loaded (e.g. module load apptainer)."
    exit 1
fi

echo "================================================================="
echo " 🚀 Launching Plant Disease AI inside $CONTAINER_CMD Container"
echo " Platform: NVIDIA DGX Server with GPU Acceleration (--nv)"
echo "================================================================="

# 2. Container Image Specification (Prefers local .sif or pulls from NGC)
SIF_IMAGE="${APPTAINER_IMAGE:-plant_disease.sif}"
NGC_IMAGE="docker://nvcr.io/nvidia/pytorch:23.10-py3"

if [ -f "$SIF_IMAGE" ]; then
    TARGET_IMAGE="$SIF_IMAGE"
    echo "[Container] Using local image: $TARGET_IMAGE"
else
    echo "[Container] Local SIF not found. Using NVIDIA NGC PyTorch Image: $NGC_IMAGE"
    TARGET_IMAGE="$NGC_IMAGE"
fi

# 3. Environment and Port Bindings
export APPTAINERENV_PORT=8501
export SINGULARITYENV_PORT=8501

# 4. Execute inside container with NVIDIA GPU passthrough
# Binds current directory and runs the app
$CONTAINER_CMD exec \
    --nv \
    --bind "$SCRIPT_DIR:/workspace" \
    --pwd /workspace \
    "$TARGET_IMAGE" \
    bash -c "export PATH=\"\$HOME/.local/bin:\$PATH\"; pip install -q -r requirements.txt && python3 -m streamlit run app.py --server.port 8501 --server.address 0.0.0.0"
