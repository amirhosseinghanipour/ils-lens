#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
exec "${ILS_PYTHON:-python3}" -m ilslens.cli toy --config configs/toy.yaml
