#!/bin/bash
set -e
#set -o xtrace
#set -x

# find the directory of this script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# for certain tests, we need CUDA_VISIBLE_DEVICES to be set before TF imports
export CUDA_VISIBLE_DEVICES="0,1"

PYTEST_FLAGS="--run-extra-slow"
IGNORE_NBS=""

echo "pytest_flags=" $PYTEST_FLAGS
echo "ignore nbs=" $IGNORE_NBS

## change into this directory (so that we pick up the pytest configuration)
cd $DIR
cd ..

## Run black check first so that it fails quickly
black . --check

nice -n 100 pytest bprseg/test/test_environment.py
nice -n 100 pytest -n 25 --dist loadscope \
     --ignore bprseg/test/test_environment.py \
     $IGNORE_NBS \
     --runslow \
     $PYTEST_FLAGS \
     $1
