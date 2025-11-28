#!/bin/bash
# SuperAnimal Behavior Analysis PoC - Setup Script

set -e

echo "=============================================="
echo "SuperAnimal Behavior Analysis PoC - Setup"
echo "=============================================="

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "Error: conda not found. Please install Anaconda or Miniconda first."
    exit 1
fi

# Get script directory (now in src/scripts/)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"

cd "$PROJECT_DIR"

# Create conda environment
echo ""
echo "[Step 1] Creating conda environment..."
if conda env list | grep -q "superanimal-poc"; then
    echo "Environment 'superanimal-poc' already exists. Updating..."
    conda env update -f environment.yml --prune
else
    conda env create -f environment.yml
fi

echo ""
echo "[Step 2] Activating environment..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate superanimal-poc

echo ""
echo "[Step 3] Verifying installation..."
python -c "import deeplabcut; print(f'DeepLabCut version: {deeplabcut.__version__}')" || echo "Warning: DeepLabCut import failed"
python -c "import hydra; print(f'Hydra version: {hydra.__version__}')"
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import cv2; print(f'OpenCV version: {cv2.__version__}')"

echo ""
echo "[Step 4] Creating data directories..."
mkdir -p data/raw data/processed data/external outputs/experiments outputs/logs

echo ""
echo "=============================================="
echo "Setup complete!"
echo ""
echo "To activate the environment:"
echo "  conda activate superanimal-poc"
echo ""
echo "To run the analysis:"
echo "  python run.py                    # TopViewMouse"
echo "  python run.py model=quadruped    # Quadruped"
echo "=============================================="
