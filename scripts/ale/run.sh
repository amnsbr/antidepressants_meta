#!/bin/bash
PROJECT_DIR="$(dirname $0)/../.."

source ${PROJECT_DIR}/venv/bin/activate

${PROJECT_DIR}/venv/bin/python ${PROJECT_DIR}/scripts/ale/run.py