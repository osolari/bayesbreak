#!/bin/bash
# Environment setup script using pip for package installation

set -x -e  # Enable debugging and exit on error

# Get the directory of this script
SOURCE=${BASH_SOURCE[0]}
while [ -h "$SOURCE" ]; do # resolve $SOURCE until the file is no longer a symlink
  DIR=$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )
  SOURCE=$(readlink "$SOURCE")
  [[ $SOURCE != /* ]] && SOURCE=$DIR/$SOURCE # if $SOURCE was a relative symlink, we need to resolve it relative to the path where the symlink file was located
done
DIR=$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )

# Detect operating system
unamestr=$(uname)
echo Creating environment for "$unamestr" operating system

# Set requirements file path based on operating system
if [[ "$unamestr" == "Linux" ]]; then
   REQUIREMENTS_FILE=$DIR/../../etc/requirements/requirements.txt
elif [[ "$unamestr" == "Darwin" ]]; then
   REQUIREMENTS_FILE=$DIR/../../etc/requirements/requirements.txt
else
  raise error "Operating system $unamestr is not supported."
fi

# Check if requirements file exists
if [ ! -f "$REQUIREMENTS_FILE" ]; then
    echo "Error: Requirements file not found at $REQUIREMENTS_FILE"
    echo "Please run the pin.sh script first to generate the requirements file."
    exit 1
fi

# Install packages using pip instead of conda
echo "Installing packages from $REQUIREMENTS_FILE..."
pip install -r "$REQUIREMENTS_FILE"

# Install the package in development mode
echo "Installing package in development mode..."
python -m pip install --editable .

echo "Environment setup complete!"