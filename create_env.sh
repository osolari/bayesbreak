#!/bin/bash
# filepath: ~/workspace/projects/bayesbreak/create_env.sh

set -e

# Configuration
LOCK_FILE="requirements-lock.yml"

# Check if lock file exists
if [ ! -f "$LOCK_FILE" ]; then
    echo "Lock file '$LOCK_FILE' not found. Please run ./pin_lock.sh first."
    exit 1
fi

# Update conda itself
echo "Updating conda..."
conda update -n base -c defaults conda -y

# Extract package name from lock file
PACKAGE_NAME=$(grep "^name:" "$LOCK_FILE" | awk '{print $2}' | head -n 1)
if [ -z "$PACKAGE_NAME" ]; then
    PACKAGE_NAME="bayesbreak"
    echo "Could not extract package name from $LOCK_FILE. Using default: $PACKAGE_NAME"
else
    echo "Using package name from lock file: $PACKAGE_NAME"
fi

# Check if environment already exists
if conda env list | grep -q "^$PACKAGE_NAME\s"; then
    echo "Conda environment '$PACKAGE_NAME' already exists."
    echo "To recreate, first remove it with: conda env remove -n $PACKAGE_NAME"
    exit 1
fi

# Create environment from lock file using conda
echo "Creating environment '$PACKAGE_NAME' from $LOCK_FILE using conda..."
conda env create -f "$LOCK_FILE"

# Activate environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$PACKAGE_NAME"

# Update pip to the latest version within the environment
echo "Updating pip to the latest version..."
pip install --upgrade pip

# Install package in editable mode
echo "Installing package in editable mode..."
pip install -e .

echo ""
echo "Environment setup complete!"
echo "To activate: conda activate $PACKAGE_NAME"