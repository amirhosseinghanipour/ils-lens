"""Refit saved Qwen interventions without a model or evaluation labels."""
import argparse,json,time
from pathlib import Path
import torch
from ilslens.fit import select
from ilslens.eval.faithfulness import bootstrap_r2

parser=argparse.ArgumentParser()
parser.add_argument('--run',default='runs/qwen-main')
parser.add_argument('--layers',type=int,nargs='*',default=list(range(24)))
parser.add_argument('--threads',type=int,default=4)
args=parser.parse_args();torch.set_num_threads(args.threads)
root=Path(args.run);manifest=json.loads((root/'manifest.json').read_text());config=manifest['config']
out=root/'fits';out.mkdir(exist_ok=True)
cells=len(args.layers)*len(config['epsilons'])*len(config['gammas'])
(out/'estimate.json').write_text(json.dumps(dict(
    cells=cells,estimated_matrix_bytes=cells*896*896*4,
    rough_seconds_per_cell=180,rough_total_seconds=cells*180,
    caveat='Unbenchmarked conservative CPU estimate; actual seconds recorded per fitted cell',
    lambda_candidates=config['lambda_candidates'],priors=['I','J','R'],estimators=config['estimators']),indent=2))
def load_samples(split,layer):
    files=sorted((root/'samples'/split).glob(f'prompt-*-layer-{layer:02d}.pt'))
    required=config['source_prompts'] if split=='train' else config['heldout_prompts']
    if len(files)!=required:raise ValueError(f'Incomplete {split} samples at layer {layer}')
    chunks=[torch.load(path,weights_only=True) for path in files]
    if any(c['metadata']['fingerprint']!=manifest['fingerprint'] for c in chunks):
        raise ValueError('Sample fingerprint mismatch')
    return tuple(torch.cat([c[key] for c in chunks]) for key in ('delta','response','prompt_ids'))

for layer in args.layers:
    x,y,groups=load_samples('train',layer)
    xt,yt,test_groups=load_samples('validation',layer)
    per_prompt={}
    for name in ('J','R'):
        values=[]
        for i in range(config['source_prompts']):
            artifact=torch.load(root/'baselines'/f'{name}-prompt-{i:03d}.pt',weights_only=True)
            if artifact['metadata']['fingerprint']!=manifest['fingerprint']:
                raise ValueError('Baseline fingerprint mismatch')
            values.append(artifact['matrices'][layer])
        per_prompt[name]=torch.stack(values)
    priors={'I':torch.eye(896),**{name:values.mean(0) for name,values in per_prompt.items()}}
    cache={}
    def fold_prior(name,train_groups):
        key=(name,tuple(train_groups.tolist()))
        if key not in cache:
            cache[key]=priors['I'] if name=='I' else per_prompt[name][train_groups].mean(0)
        return cache[key]
    for e,epsilon in enumerate(config['epsilons']):
        for g,gamma in enumerate(config['gammas']):
            destination=out/f'layer-{layer:02d}-epsilon-{epsilon}-gamma-{gamma}.pt'
            if destination.exists():continue
            before=time.perf_counter()
            a,selection,cv=select(x[:,e],y[:,e,g],groups,priors,config['lambda_candidates'],
                estimators=config['estimators'],seed=config['seed'],fold_prior_provider=fold_prior)
            matrices={**priors,'ILS':a}
            metrics={name:bootstrap_r2(yt[:,e,g],xt[:,e]@matrix.T,test_groups)
                     for name,matrix in matrices.items()}
            record=dict(matrix=a,selection=selection,cv=cv,heldout=metrics,
                metadata=dict(layer=layer,epsilon=epsilon,gamma=gamma,N=len(x),
                    target_layer=23,source='block_output',target='pre_final_norm',
                    baseline_reduction='upstream_sum',ils_reduction='normalized_horizon',
                    fingerprint=manifest['fingerprint'],label_tuned=False),
                seconds=time.perf_counter()-before)
            temporary=destination.with_suffix('.tmp');torch.save(record,temporary);temporary.replace(destination)
            print(json.dumps(dict(layer=layer,epsilon=epsilon,gamma=gamma,selection=selection,
                                  heldout=metrics,seconds=record['seconds'])),flush=True)
