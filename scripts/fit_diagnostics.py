"""Prompt-fold-selected learning curves and ablations, without readout labels."""
import argparse,json
from pathlib import Path
import torch,yaml
from ilslens.runtime import resolve_device
from ilslens.fit import select
from ilslens.budget import baseline_count,prior_prompt_indices
from ilslens.eval.diagnostics import balanced_prefix
from ilslens.eval.faithfulness import bootstrap_r2
from ilslens.experiment import Experiment,save_tensor,write_json

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',default='runs/qwen-main');p.add_argument('--samples',default='runs/ablations')
    p.add_argument('--output',default='runs/diagnostics');p.add_argument('--config',default='configs/diagnostics.yaml');p.add_argument('--device',default='cuda');args=p.parse_args()
    config=yaml.safe_load(Path(args.config).read_text());root=Path(args.run);manifest=json.loads((root/'manifest.json').read_text());primary=manifest['config']
    device=resolve_device(args.device);run=Experiment(args.output,dict(config=config,primary_fingerprint=manifest['fingerprint'],device=args.device))
    ei=primary['epsilons'].index(config['epsilon']);gi=primary['gammas'].index(config['gamma']);uniform=primary['gammas'].index(1.)
    def samples(split,layer,variant=None,gamma_index=gi):
        directory=root if variant is None else Path(args.samples)
        files=sorted((directory/'samples'/split).glob(f'prompt-*-layer-{layer:02d}.pt'))
        expected=primary['source_prompts'] if split=='train' else primary['heldout_prompts']
        if len(files)!=expected:raise ValueError('Incomplete diagnostics inputs')
        chunks=[torch.load(path,weights_only=True,map_location='cpu') for path in files]
        provenance='fingerprint' if variant is None else 'primary_fingerprint'
        if any(c['metadata'][provenance]!=manifest['fingerprint'] for c in chunks):raise ValueError('Diagnostics provenance mismatch')
        x=torch.cat([c['delta'][:,ei] if variant is None else c['variants'][variant]['delta'] for c in chunks]).to(device)
        y=torch.cat([c['response'][:,ei,gamma_index] if variant is None else c['variants'][variant]['response'][:,gamma_index] for c in chunks]).to(device)
        groups=torch.cat([c['prompt_ids'] for c in chunks])
        return x,y,groups
    rows=[]
    for layer in config['layers']:
        per_prompt={name:torch.stack([torch.load(root/'baselines'/f'{name}-prompt-{i:03d}.pt',weights_only=True,map_location='cpu')['matrices'][layer]
            for i in range(baseline_count(primary))]) for name in ('J','R')}
        d=per_prompt['J'].shape[-1];priors={'I':torch.eye(d,device=device),**{name:values.mean(0).to(device) for name,values in per_prompt.items()}}
        def fold_prior(name,groups):return priors['I'] if name=='I' else per_prompt[name][prior_prompt_indices(primary,groups)].mean(0).to(device)
        jobs=[('main',None,None,gi,primary['estimators'],primary['lambda_candidates'])]
        jobs += [(f'N-{n}',None,n,gi,primary['estimators'],primary['lambda_candidates']) for n in [d,2*d,4*d,8*d]]
        jobs += [('no_robust',None,None,gi,['ridge'],primary['lambda_candidates']),('lambda_zero',None,None,gi,primary['estimators'],[0.]),
                 ('uniform_horizon',None,None,uniform,primary['estimators'],primary['lambda_candidates'])]
        jobs += [(variant,variant,None,gi,primary['estimators'],primary['lambda_candidates']) for variant in ['no_antithetic','gaussian','isotropic']]
        for name,variant,n,g,estimators,lambdas in jobs:
            path=run.path/f'{name}-layer-{layer:02d}.pt'
            if path.exists():rows.append(torch.load(path,weights_only=True,map_location='cpu')['result']);continue
            x,y,groups=samples('train',layer,variant,g)
            xt,yt,held_groups=samples('validation',layer,None,gi)
            native_x,native_y,native_groups=samples('validation',layer,variant,g)
            if n is not None:
                indices=balanced_prefix(groups,n);x=x[indices];y=y[indices];groups=groups[indices]
            a,selection,cv=select(x,y,groups,priors,lambdas,estimators=estimators,seed=config['seed'],fold_prior_provider=fold_prior)
            statistics=bootstrap_r2(yt,xt@a.T,held_groups,seed=config['seed'])
            native=bootstrap_r2(native_y,native_x@a.T,native_groups,seed=config['seed'])
            row=dict(variant=name,layer=layer,N=len(x),gamma=primary['gammas'][g],epsilon=config['epsilon'],selection=selection,
                     common_donor_antithetic_target_gamma=config['gamma'],native_estimand=native,**statistics)
            save_tensor(path,dict(matrix=a.cpu(),cv=cv,result=row,metadata=dict(fingerprint=run.fingerprint,label_tuned=False,target_layer=23)))
            rows.append(row);run.event('diagnostic-fit',**row)
    run.result(dict(rows=rows,no_zscore='Raw versus z-score comparisons are reported by evaluate_readouts.py',
        selection='Every fitted variant independently uses five-fold-by-prompt interventional R2; main-estimator settings are never selected with evaluation labels',
        learning_curve='Nested round-robin sample subsets preserve all fitting prompts at each N'))

if __name__=='__main__':main()
