#!/bin/bash
# Script to pin Python dependencies using pip-compile

set -e  # Exit on error

# Get the directory of this script
SOURCE=${BASH_SOURCE[0]}
while [ -h "$SOURCE" ]; do # resolve $SOURCE until the file is no longer a symlink
  DIR=$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )
  SOURCE=$(readlink "$SOURCE")
  [[ $SOURCE != /* ]] && SOURCE=$DIR/$SOURCE # if $SOURCE was a relative symlink, we need to resolve it relative to the path where the symlink file was located
done
DIR=$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )

# Project root directory (two levels up from this script)
PROJECT_ROOT="$DIR/../"

# Requirements directory
REQ_DIR="$PROJECT_ROOT/devops"

# Check if pip-tools is installed
if ! pip show pip-tools > /dev/null 2>&1; then
    echo "Installing pip-tools..."
    pip install pip-tools
fi

# Define input and output files
REQUIREMENTS_IN="$REQ_DIR/requirements.in"
REQUIREMENTS_TXT="$REQ_DIR/requirements.txt"

# Check if requirements.in exists
if [ ! -f "$REQUIREMENTS_IN" ]; then
    echo "Error: $REQUIREMENTS_IN does not exist."
    echo "Please create a requirements.in file with your dependencies before running this script."
    exit 1
fi

# Run pip-compile to generate pinned requirements
echo "Generating pinned requirements..."

# Generate requirements.txt from requirements.in
echo "Compiling $REQUIREMENTS_IN -> $REQUIREMENTS_TXT"
echo "Running: pip-compile --verbose --resolver=backtracking --output-file=$REQUIREMENTS_TXT $REQUIREMENTS_IN"
pip-compile --verbose --resolver=backtracking --output-file="$REQUIREMENTS_TXT" "$REQUIREMENTS_IN"

echo "Requirements pinning complete!"
echo "Generated files:"
echo "  - $REQUIREMENTS_TXT"