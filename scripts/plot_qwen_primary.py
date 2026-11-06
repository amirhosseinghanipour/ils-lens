"""Plot every complete primary R² cell and its saved prompt-bootstrap interval."""
import argparse,json,os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR','/tmp/ils-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from ilslens.primary_reporting import primary_rows


def plot_grid(config,rows,output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    names={'I':'Identity','J':'J (upstream sum)','R':'R (upstream sum)','ILS':'ILS (normalized horizon)'}
    indexed={(r['layer'],r['epsilon'],r['gamma'],r['lens']):r for r in rows}
    expected={(l,e,g,n) for l in range(24) for e in config['epsilons'] for g in config['gammas'] for n in names}
    if len(indexed)!=len(rows) or set(indexed)!=expected:
        raise ValueError('Plot requires the complete primary layer/epsilon/horizon/baseline grid')
    colors=['#555555','#0072B2','#D55E00','#009E73'];paths=[]
    for epsilon in config['epsilons']:
        fig,axes=plt.subplots(len(config['gammas']),1,figsize=(10,3*len(config['gammas'])),squeeze=False)
        for ax,gamma in zip(axes[:,0],config['gammas']):
            for (name,label),color in zip(names.items(),colors):
                selected=[indexed[l,epsilon,gamma,name] for l in range(24)]
                values=np.array([np.nan if r['r2'] is None else r['r2'] for r in selected])
                intervals=np.array([[np.nan if v is None else v for v in r['ci']] for r in selected])
                ax.plot(range(24),values,marker='.',markersize=4,label=label,color=color,linewidth=1.3)
                ax.fill_between(range(24),intervals[:,0],intervals[:,1],color=color,alpha=.14)
            ax.set_yscale('symlog',linthresh=.1)
            ax.axhline(0,color='#888888',linewidth=.6,linestyle='--')
            ax.set(xlabel='Block output (zero-based)',ylabel='Held-out interventional R²',title=f'Horizon γ = {gamma}')
            ax.set_xticks(range(0,24,2));ax.grid(alpha=.15)
        fig.suptitle(f'Qwen primary donor interventions: ε = {epsilon}',fontsize=14)
        handles,labels=axes[0,0].get_legend_handles_labels()
        fig.legend(handles,labels,loc='lower center',ncol=2,bbox_to_anchor=(.5,.025),frameon=False)
        fig.text(.5,.005,'95% prompt-bootstrap intervals. Symmetric-log axis preserves negative values.\n'
                 'J/R and ILS target reductions differ; these panels alone do not establish method superiority.',
                 ha='center',fontsize=9)
        fig.tight_layout(rect=(0,.075,1,.975))
        path=output/f'qwen-primary-epsilon-{epsilon}.png'
        fig.savefig(path,dpi=160);plt.close(fig);paths.append(path)
    return paths


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',default='runs/qwen-main')
    p.add_argument('--output',default='artifacts/plots');a=p.parse_args()
    config,rows=primary_rows(a.run)
    paths=plot_grid(config,rows,a.output)
    print(json.dumps(dict(figures=[str(p) for p in paths],cells=len(rows),scope='Primary donor R² only')))


if __name__=='__main__':main()
