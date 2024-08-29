#!/bin/bash
PROJECT_DIR="$(dirname $0)/../.."
cd $PROJECT_DIR
# setup venv
python -m venv venv
source ./venv/bin/activate
python -m pip install -r requirements.txt

# make logs and results directories
mkdir logs
mkdir results

# clone pyALE into tools
mkdir tools
cd tools
git clone https://github.com/LenFrahm/pyALE.git
## checkout to the commit used in this project
cd pyALE
git checkout ff7cdf5b50e242a6e20aadabf22dc820732fa5fa
