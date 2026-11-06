"""Plot the saved partial directional-error audit, including descriptive intervals."""
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',default='artifacts/numerical-partial-jvp-audit.json')
    p.add_argument('--output',default='artifacts/plots/qwen-partial-jvp-errors.png');a=p.parse_args()
    data=json.loads(Path(a.input).read_text());fig,ax=plt.subplots(figsize=(8,4.5),layout='constrained')
    for epsilon in sorted({r['epsilon'] for r in data['rows']},reverse=True):
        rows=sorted([r for r in data['rows'] if r['epsilon']==epsilon],key=lambda r:r['layer'])
        x=[r['layer'] for r in rows];y=[r['mean'] for r in rows];ci=np.asarray([r['mean_context_bootstrap_ci'] for r in rows])
        line,=ax.plot(x,y,marker='o',markersize=3,label=f'epsilon = {epsilon:g}')
        ax.fill_between(x,ci[:,0],ci[:,1],alpha=.15,color=line.get_color())
    ax.set(yscale='log',xlabel='Block output (zero-based layer)',ylabel='Mean relative JVP error',
        title='Partial Qwen fp32 directional finite-difference check')
    ax.grid(alpha=.2);ax.legend()
    fig.supxlabel('8 original contexts; shading: descriptive 95% context-bootstrap intervals.\nNot a full validation result; shared-article dependence is not modeled.',fontsize=8)
    output=Path(a.output);output.parent.mkdir(parents=True,exist_ok=True);fig.savefig(output,dpi=160);plt.close(fig)

if __name__=='__main__':main()
