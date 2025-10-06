#!/bin/bash
# filepath: bayesbreak/create_env.sh
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
cd "$SCRIPT_DIR"
############################################
# 1. Load SSL certificates (optional)
############################################
if [ -f ".set_ssl_certs.sh" ]; then
    source .set_ssl_certs.sh
fi

############################################
# 2. Environment name from requirements.yml
############################################
INPUT_FILE="requirements.yml"
if [ ! -f "$INPUT_FILE" ]; then
    echo "Input file '$INPUT_FILE' not found." >&2
    exit 1
fi

ENV_NAME=$(awk '/^name:/ {print $2; exit}' "$INPUT_FILE")
ENV_NAME="${ENV_NAME:-bayesbreak}"

############################################
# 3. Detect platform and choose lock file
############################################
OS=$(uname -s | tr '[:upper:]' '[:lower:]')
ARCH=$(uname -m)

case "${OS}-${ARCH}" in
    linux-x86_64)
        LOCK_FILE="conda-linux-64.lock"
        ;;
    darwin-x86_64)
        LOCK_FILE="conda-osx-64.lock"
        ;;
    darwin-arm64)
        LOCK_FILE="conda-osx-arm64.lock"
        ;;
    *)
        echo "Unsupported platform: ${OS}-${ARCH}" >&2
        echo "Please generate an appropriate lock file using ./pin.sh"
        exit 1
        ;;
esac

echo "Using environment name: $ENV_NAME"
echo "Using lock file: $LOCK_FILE"

############################################
# 4. Check lock file exists
############################################
if [ ! -f "$LOCK_FILE" ]; then
    echo "Lock file '$LOCK_FILE' not found. Please run ./pin.sh first." >&2
    exit 1
fi

############################################
# 5. Update conda
############################################
echo "Updating conda..."
conda update -n base -c defaults conda -y

############################################
# 6. Create or update environment
############################################
echo "Ensuring environment '$ENV_NAME' is up to date from $LOCK_FILE..."
set +e
if conda env list | grep -q "^$ENV_NAME\s"; then
    echo "Environment '$ENV_NAME' already exists. Updating..."
else
    echo "Creating new environment '$ENV_NAME'..."
fi

conda-lock install --name "$ENV_NAME" "$LOCK_FILE"
status=$?
set -e

if [ $status -ne 0 ]; then
    if conda env list | grep -q "^$ENV_NAME\s"; then
        echo "conda-lock returned nonzero, but environment '$ENV_NAME' exists. Continuing..."
    else
        echo "conda-lock failed and environment '$ENV_NAME' was not created/updated." >&2
        exit 1
    fi
fi

############################################
# 7. Activate environment
############################################
export MKL_INTERFACE_LAYER=${MKL_INTERFACE_LAYER:-}
export MKL_THREADING_LAYER=${MKL_THREADING_LAYER:-}

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

############################################
# 8. Update pip + install local package
############################################
echo "Updating pip inside environment..."
pip install --upgrade pip

PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"
echo "Installing package in editable mode..."
pip install -e .

############################################
# 9. Done
############################################
echo ""
echo "Environment setup complete!"
echo "To activate later: conda activate $ENV_NAME"
