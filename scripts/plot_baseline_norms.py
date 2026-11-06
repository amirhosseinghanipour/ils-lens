"""Plot the saved descriptive J/R norm summary without model execution."""
import argparse,json,os
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',default='artifacts/qwen-baseline-norms.json')
    p.add_argument('--output',default='artifacts/plots/qwen-baseline-norms.png');a=p.parse_args()
    os.environ.setdefault('MPLCONFIGDIR','/tmp/ils-matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    result=json.loads(Path(a.input).read_text());fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    for name in ('J','R'):
        rows=sorted((r for r in result['rows'] if r['lens']==name),key=lambda r:r['layer'])
        layers=[r['layer'] for r in rows]
        axes[0].plot(layers,[r['mean_prompt_norm'] for r in rows],'.-',label=name)
        axes[0].fill_between(layers,[r['mean_prompt_norm_ci'][0] for r in rows],
                            [r['mean_prompt_norm_ci'][1] for r in rows],alpha=.15)
        axes[1].plot(layers,[r['norm_of_mean_over_mean_norm'] for r in rows],'.-',label=name)
    axes[0].set(ylabel='Mean per-prompt Frobenius norm',title='Shading: descriptive 95% prompt-bootstrap CI')
    axes[1].set(ylabel='Norm of mean matrix / mean matrix norm',ylim=(0,1.05),title='Cancellation under averaging; no inference CI')
    for ax in axes:
        ax.set_xlabel('Block output (zero-based)');ax.legend();ax.grid(alpha=.2)
    fig.suptitle('Qwen2.5-0.5B-Instruct: 32 fitting prompts, native summed targets',fontsize=12)
    fig.text(.5,.015,'Final block uses analytic identity. These are fitting diagnostics, not held-out faithfulness.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.04,1,.94));destination=Path(a.output);destination.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(destination,dpi=160);plt.close(fig)


if __name__=='__main__':main()
