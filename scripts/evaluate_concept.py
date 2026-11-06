"""Compare short, long and horizon-residual concept reads on identical sites."""
import argparse,json
from pathlib import Path
import numpy as np
import torch,yaml
from ilslens.model import load_local
from ilslens.artifacts import primary_bank
from ilslens.data import wikitext_sequences
from ilslens.covariance import residual_moments
from ilslens.concept import projection,read
from ilslens.readout import raw,target_ranks
from ilslens.eval.readout_tasks import candidate_ids
from ilslens.experiment import Experiment,save_tensor

def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);p.add_argument('--states',default='runs/workspacebench/read-states.pt')
    p.add_argument('--run',default='runs/qwen-main');p.add_argument('--config',default='configs/concept.yaml')
    p.add_argument('--output',default='runs/concept');p.add_argument('--device',default='cuda');args=p.parse_args()
    config=yaml.safe_load(Path(args.config).read_text());model,tokenizer=load_local(args.device)
    short,metadata=primary_bank(args.run,config['epsilon'],0.);long,_=primary_bank(args.run,config['epsilon'],.9)
    cached=torch.load(args.states,weights_only=True,map_location='cpu')
    if cached['metadata']['fingerprint']!=metadata['fingerprint']:raise ValueError('Read-state provenance differs')
    run=Experiment(args.output,dict(config=config,primary_fingerprint=metadata['fingerprint'],device=args.device,ranking='score-descending-id-ascending-v2'))
    count=json.loads((Path(args.run)/'manifest.json').read_text())['config']['source_prompts']
    sequences,_=wikitext_sequences(args.dataset,tokenizer,'train',count);moments=residual_moments(model,sequences,config['layers'])
    rows=[]
    for layer in config['layers']:
        s=short['ILS'][layer].to(model.device);l=long['ILS'][layer].to(model.device)
        b=projection(s,l,moments[layer]['covariance'],config['eta']);save_tensor(run.path/f'projection-{layer:02d}.pt',dict(B=b.cpu(),metadata=metadata,eta=config['eta']))
        for index,item in enumerate(cached['items']):
            if layer not in cached['states'][index]:continue
            h=cached['states'][index][layer].to(model.device)
            long_state=h@l.T;concept_state=long_state-(h@s.T)@b.T
            remaining_norm=concept_state.norm(dim=-1)/long_state.norm(dim=-1).clamp_min(1e-30)
            with torch.no_grad():
                scores={'short':raw(h,s,model.lm_head.weight,model.model.norm.weight),
                        'long':raw(h,l,model.lm_head.weight,model.model.norm.weight),
                        'concept':read(h,s,l,b,model.lm_head.weight,model.model.norm.weight)}
            for method,values in scores.items():
                for j,position in enumerate(item['positions']):
                    next_token=item['input_ids'][position+1] if position+1<len(item['input_ids']) else (item.get('completion_ids') or [None])[0]
                    ranks=[]
                    for target in item['targets']:
                        ids=candidate_ids(tokenizer,target)
                        if ids:ranks.append(int(target_ranks(values[j],ids)))
                    rows.append(dict(item_id=item['id'],layer=layer,position=position,method=method,
                        concept_remaining_norm_fraction=float(remaining_norm[j]),
                        next_token_rank=None if next_token is None else int(target_ranks(values[j],[next_token])),
                        concept_ranks=ranks,unscoreable_concepts=len(item['targets'])-len(ranks)))
    summaries=[];rng=np.random.default_rng(config['seed'])
    for layer in config['layers']:
        for method in ('short','long','concept'):
            selected=[r for r in rows if r['layer']==layer and r['method']==method];ids=sorted(set(r['item_id'] for r in selected))
            for target in ('next_token','concept'):
                values=[]
                for item in ids:
                    ranks=[rank for r in selected if r['item_id']==item for rank in ([r['next_token_rank']] if target=='next_token' else r['concept_ranks']) if rank is not None]
                    if ranks:values.append(np.mean(np.array(ranks)<=10))
                values=np.array(values)
                summaries.append(dict(layer=layer,method=method,target=target,k=10,items=len(values),recall=None if not len(values) else float(values.mean()),
                    ci=[None,None] if not len(values) else np.quantile(values[rng.integers(len(values),size=(200,len(values)))].mean(1),[.025,.975]).tolist()))
    run.result(dict(rows=rows,summary=summaries,normalization='Final-norm gain only on concept residual, divided by RMS of A_long h',
        eta_rule='Fixed before evaluation; no readout-label selection',interpretation='Report concept and next-token recall separately; overlapping CIs do not establish separation.'))

if __name__=='__main__':main()
