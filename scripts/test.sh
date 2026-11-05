#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/.deps:$PWD${PYTHONPATH:+:$PYTHONPATH}"
if "${ILS_PYTHON:-python3}" -c 'import pytest' 2>/dev/null; then
    exec "${ILS_PYTHON:-python3}" -m pytest tests -v
fi
exec "${ILS_PYTHON:-python3}" -m unittest discover -s tests -v
