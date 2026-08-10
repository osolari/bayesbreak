#!/usr/bin/env bash
# Create a reproducible BayesBreak development environment.
#
# Usage:
#   bash setup_env.sh              # conda env "bayesbreak" with Python 3.11
#   bash setup_env.sh 3.12         # override Python version
#   bash setup_env.sh --venv       # use python -m venv instead of conda
#   bash setup_env.sh --lock       # pin dependencies to requirements-lock.txt

set -euo pipefail

ENV_NAME="bayesbreak"
PY_VERSION="3.11"
USE_VENV=0
LOCK=0

for arg in "$@"; do
  case "$arg" in
    --venv) USE_VENV=1 ;;
    --lock) LOCK=1 ;;
    --name=*) ENV_NAME="${arg#*=}" ;;
    3.*) PY_VERSION="$arg" ;;
    *) echo "Unknown argument: $arg" >&2; exit 1 ;;
  esac
done

install_editable() {
  python -m pip install --upgrade pip
  python -m pip install -e ".[dev,plots,docs,notebooks,datasets]"
  if [[ "$LOCK" -eq 1 ]]; then
    python -m pip freeze --exclude-editable > requirements-lock.txt
    echo "Wrote requirements-lock.txt"
  fi
  pre-commit install || true
}

if [[ "$USE_VENV" -eq 1 ]]; then
  python"$PY_VERSION" -m venv .venv
  # shellcheck disable=SC1091
  source .venv/bin/activate
  install_editable
  echo "Activate with: source .venv/bin/activate"
else
  if ! command -v conda >/dev/null 2>&1; then
    echo "conda not found. Install Miniforge/Miniconda or pass --venv." >&2
    exit 1
  fi
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "Environment $ENV_NAME already exists; reusing it."
  else
    conda create -y -n "$ENV_NAME" "python=$PY_VERSION" pip
  fi
  conda activate "$ENV_NAME"
  install_editable
  echo "Activate with: conda activate $ENV_NAME"
fi
