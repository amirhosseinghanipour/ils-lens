#!/usr/bin/env bash
# Run in a network-enabled terminal. Writes only inside this workspace.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p references data logs .deps .cache/huggingface
export HF_HOME="$PWD/.cache/huggingface"
export PYTHONPATH="$PWD/.deps:$PWD${PYTHONPATH:+:$PYTHONPATH}"
PYTHON="${ILS_PYTHON:-python3}"

for repository in anthropics/jacobian-lens safety-research/jpp_lens camilablank/workspace-bench safety-research/subliminal-jlens MinhxLe/subliminal-learning; do
    destination="references/${repository##*/}"
    if [ ! -d "$destination/.git" ]; then
        git clone --depth 1 "https://github.com/$repository.git" "$destination"
    fi
    git -C "$destination" rev-parse HEAD > "$destination.commit.txt"
done

"$PYTHON" -m pip install --target "$PWD/.deps" 'pytest==8.3.5'
"$PYTHON" scripts/download_data.py
printf '%s\n' 'Download complete. Dataset and reference revisions are recorded locally.'
