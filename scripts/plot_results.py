"""Regenerate figures from recorded local measurements."""
import json,os,hashlib
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR','/tmp/ils-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

root=Path(__file__).resolve().parents[1]
os.chdir(root)
toy=json.loads(Path('artifacts/toy/results.json').read_text())
qwen=json.loads(Path('artifacts/qwen-pilot.json').read_text())
Path('artifacts/plots').mkdir(exist_ok=True)
fig,ax=plt.subplots(figsize=(8,4))
for name in ['J-local','R-local','ILS']:
    rows=[r for r in toy['rows'] if r['lens']==name]
    x=np.array([r['layer'] for r in rows]); y=np.array([r['r2'] for r in rows])
    lower=np.array([r['ci'][0] for r in rows]); upper=np.array([r['ci'][1] for r in rows])
    ax.plot(x,y,'o-',label=name);ax.fill_between(x,lower,upper,alpha=.1)
ax.set(xlabel='Block output (zero-based)',ylabel='Held-out interventional R²',
       title='Toy HMM only: 95% prompt-bootstrap intervals')
ax.set_xticks(range(4));ax.legend();fig.tight_layout()
fig.savefig('artifacts/plots/toy-r2.png',dpi=160);plt.close(fig)
fig,ax=plt.subplots(figsize=(8,4))
for layer in qwen['config']['layers']:
    rows=[r for r in qwen['rows'] if r['layer']==layer]
    ax.loglog([r['epsilon'] for r in rows],[r['relative_jvp_error'] for r in rows],'o-',label=f'Block {layer}')
ax.set(xlabel='ε (unit direction)',ylabel='Relative finite-difference / JVP error',
       title='Qwen numerical pilot: one prompt and direction; no sampling CI')
ax.legend();fig.tight_layout();fig.savefig('artifacts/plots/qwen-jvp.png',dpi=160);plt.close(fig)
