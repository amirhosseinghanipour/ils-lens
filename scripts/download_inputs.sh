#!/usr/bin/env bash
# Run in a network-enabled terminal. Writes only inside this workspace.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p references data logs .deps .cache/huggingface
export HF_HOME="$PWD/.cache/huggingface"
export PYTHONPATH="$PWD/.deps:$PWD${PYTHONPATH:+:$PYTHONPATH}"
PYTHON="${ILS_PYTHON:-python3}"

"$PYTHON" - <<'PY'
import json,subprocess
from pathlib import Path
for repository,revision in json.loads(Path('configs/reference-revisions.json').read_text()).items():
    destination=Path('references')/repository.split('/')[-1]
    if not (destination/'.git').exists():
        destination.mkdir(parents=True,exist_ok=True)
        subprocess.run(['git','init',str(destination)],check=True)
        subprocess.run(['git','-C',str(destination),'remote','add','origin',f'https://github.com/{repository}.git'],check=True)
        subprocess.run(['git','-C',str(destination),'fetch','--depth','1','origin',revision],check=True)
        subprocess.run(['git','-C',str(destination),'checkout','--detach',revision],check=True)
    actual=subprocess.check_output(['git','-C',str(destination),'rev-parse','HEAD'],text=True).strip()
    if actual!=revision:raise RuntimeError(f'{destination} has a different revision; preserve local changes and inspect it explicitly')
    destination.with_suffix('.commit.txt').write_text(actual+'\n')
PY

"$PYTHON" -m pip install --target "$PWD/.deps" 'pytest==8.3.5'
"$PYTHON" scripts/download_data.py
printf '%s\n' 'Download complete. Dataset and reference revisions are recorded locally.'
