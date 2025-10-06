#!/usr/bin/env bash
# filepath: bayesbreak/pin.sh
set -Eeuo pipefail
IFS=$'\n\t'

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
cd "$SCRIPT_DIR"

############################################
# (Optional) Logging to file
############################################
# Uncomment to capture a full transcript:
# exec > >(tee -a pin.log) 2>&1

############################################
# 0. Ensure conda is initialized
############################################
if ! command -v conda >/dev/null 2>&1; then
  echo "Conda not found in PATH. Attempting to initialize..."
  if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
  elif [ -n "$(conda info --base 2>/dev/null)" ]; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
  else
    echo "Failed to source conda initialization." >&2
    exit 1
  fi
fi

############################################
# 1. Validate input file
############################################
INPUT_FILE="requirements.yml"
if [[ ! -f "$INPUT_FILE" ]]; then
  echo "Input file '$INPUT_FILE' not found." >&2
  exit 1
fi

############################################
# 2. Extract environment name
############################################
ENV_NAME=$(awk '/^name:/ {print $2; exit}' "$INPUT_FILE")
ENV_NAME="${ENV_NAME:-bayesbreak}"
echo "Using environment name: $ENV_NAME"

############################################
# 3. Ensure conda-lock is installed
############################################
if ! command -v conda-lock >/dev/null 2>&1; then
  echo "conda-lock not found. Installing into current Python environment..."
  pip install --quiet conda-lock
fi

############################################
# 4. Define target platforms
############################################
PLATFORMS=("linux-64" "osx-64" "osx-arm64")

############################################
# 5. Clean up old lock files
############################################
rm -f conda-lock.yml conda-*.lock

############################################
# 6. Generate lock files (per platform)
############################################
echo "Generating conda lock files for platforms: ${PLATFORMS[*]}"

for PLATFORM in "${PLATFORMS[@]}"; do
  echo
  echo "Solving environment for: $PLATFORM ..."
  conda-lock lock \
    --file "$INPUT_FILE" \
    --lockfile conda-lock.yml \
    --platform "$PLATFORM" \
    --mamba
  echo "Rendering conda-${PLATFORM}.lock ..."
  conda-lock render -p "$PLATFORM" conda-lock.yml > "conda-${PLATFORM}.lock"
done

############################################
# 7. Update pyproject.toml dependencies (pure Bash, newline-safe)
############################################
echo
echo "Updating pyproject.toml dependencies from env/requirements.yml..."

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
cd "$SCRIPT_DIR"

REQ_FILE="requirements.yml"
TOML_FILE="../pyproject.toml"

if [[ ! -f "$REQ_FILE" ]]; then
  echo "Error: $REQ_FILE not found." >&2
  exit 1
fi
if [[ ! -f "$TOML_FILE" ]]; then
  echo "Error: $TOML_FILE not found." >&2
  exit 1
fi

# Extract dependencies from YAML
DEPS=()
while IFS= read -r line; do
  if [[ "$line" =~ ^[[:space:]]*-[[:space:]]*([^[:space:]].*)$ ]]; then
    dep="${BASH_REMATCH[1]}"
    name="${dep%%[<>=]*}"
    case "$name" in
      python|pip|pytest|mypy|ruff|pre-commit)
        continue
        ;;
    esac
    DEPS+=("$dep")
  fi
done < "$REQ_FILE"

# Build TOML dependency array
tmpfile=$(mktemp)
{
  echo "[project]"
  echo "dependencies = ["
  for dep in "${DEPS[@]}"; do
    echo "  \"$dep\","
  done | sed '$ s/,$//'   # remove trailing comma on last line
  echo "]"
} > "$tmpfile"

# Remove old dependencies block and insert the new one
awk -v repl="$(cat "$tmpfile")" '
  BEGIN {print_repl=1}
  /^\[project\]/ {print repl; next}
  /^dependencies[[:space:]]*=/ {next}
  {print}
' "$TOML_FILE" > "${TOML_FILE}.new"

mv "${TOML_FILE}.new" "$TOML_FILE"
rm -f "$tmpfile"

echo "Updated pyproject.toml dependencies:"
printf '  - %s\n' "${DEPS[@]}"

############################################
# 8. Summary
############################################
echo
echo "Pinning and synchronization completed successfully!"
echo "Generated files:"
echo "  - conda-lock.yml          (base lock spec)"
for PLATFORM in "${PLATFORMS[@]}"; do
  echo "  - conda-${PLATFORM}.lock   (strict lockfile for $PLATFORM)"
done
echo
echo "pyproject.toml dependencies have been updated from requirements.yml."
echo "To create the environment on your platform:"
echo "  ./create_env.sh"
