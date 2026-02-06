#!/bin/bash
# Setup conda environment for bayesbreak development
# Usage: bash create_env.sh
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
ENV_NAME="bayesbreak"

echo "================================================"
echo "Setting up BayesBreak development environment"
echo "================================================"
echo ""

# Create conda environment with Python
echo "Creating conda environment: $ENV_NAME"
conda create -n "$ENV_NAME" python=3.11 -y

# Activate and install the package + dev dependencies
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

echo ""
echo "Installing package and development dependencies..."
cd "$SCRIPT_DIR"
pip install --upgrade pip setuptools wheel
pip install -e ".[dev]"

echo ""
echo "================================================"
echo "Setup complete!"
echo "Activate with: conda activate $ENV_NAME"
echo "================================================"
