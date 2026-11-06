"""Capability-filtered upstream swap bank with equal-norm random controls."""
import argparse,json
from pathlib import Path
import numpy as np
import torch,yaml
from ilslens.model import load_local
from ilslens.artifacts import primary_bank
from ilslens.data import wikitext_sequences,SAMPLING_POLICY
from ilslens.covariance import residual_moments
from ilslens.eval.readout_tasks import ReadoutItem,generate_answers,gate_item,candidate_ids
from ilslens.eval.causal import swap_coordinates,ablate_direction,next_token_success,gate_band
from ilslens.experiment import Experiment,write_json

def direction(model,matrix,ids):
    return (model.lm_head.weight[ids].mean(0)*model.model.norm.weight)@matrix.to(model.device)

def main():
    p=argparse.ArgumentParser();p.add_argument('--items',required=True);p.add_argument('--dataset',required=True)
    p.add_argument('--run',default='runs/qwen-main');p.add_argument('--config',default='configs/causal.yaml')
    p.add_argument('--output',default='runs/causal');p.add_argument('--device',default='cuda');p.add_argument('--jpp');args=p.parse_args()
    config=yaml.safe_load(Path(args.config).read_text());items=json.loads(Path(args.items).read_text())['items']
    if config.get('limit_items'):items=items[:config['limit_items']]
    run=Experiment(args.output,dict(config=config,sampling_policy=SAMPLING_POLICY,items=items,device=args.device))
    model,tokenizer=load_local(args.device);bank,metadata=primary_bank(args.run,config['epsilon'],config['gamma'])
    if args.jpp:
        artifact=torch.load(args.jpp,weights_only=True,map_location='cpu')
        if not artifact['metadata']['label_tuned'] or artifact['metadata']['primary_fingerprint']!=metadata['fingerprint']:
            raise ValueError('J++ provenance mismatch')
        bank['J++ (label-tuned)']=artifact['matrices']
    source_count=json.loads((Path(args.run)/'manifest.json').read_text())['config']['source_prompts']
    sequences,_=wikitext_sequences(args.dataset,tokenizer,'train',source_count)
    layers=sorted(set(l for band in config['bands'] for l in band))
    moments=residual_moments(model,sequences,layers);covariances={l:m['covariance'] for l,m in moments.items()}
    rows=[];gates=[]
    for index,source in enumerate(items):
        gate_path=run.path/f'gate-{index:04d}.json'
        if gate_path.exists():gate=json.loads(gate_path.read_text())
        else:
            item=ReadoutItem(source['name'],source.get('category','swap'),source['prompt'],source['prompt'],[source['answer']],[source['intermediate']],'test')
            answers=generate_answers(model,tokenizer,item.prompt,seed=config['seed']+index*100,max_new_tokens=16)
            gate=gate_item(item,answers,answers);write_json(gate_path,gate)
        gates.append(gate)
        if not gate['retained']:continue
        ids={key:candidate_ids(tokenizer,source[key]) for key in ('intermediate','swap_to','answer','swap_answer')}
        if any(not value for value in ids.values()):
            rows.append(dict(item_id=source['name'],status='multi-token-target-use-template-experiment'));continue
        tokens=tokenizer(source['prompt'],return_tensors='pt').input_ids.to(model.device)
        # Capability tasks have variable length. Intervene at every prompt token,
        # matching the upstream swap experiment, distinct from the 128-token fit.
        positions=list(range(tokens.shape[1]));correct_ids=ids['answer'];target_ids=ids['swap_answer']
        for name,maps in bank.items():
            for band in config['bands']:
                if not all(layer in maps for layer in band):continue
                for kind in ('intermediate_swap','ablation','answer_swap'):
                    destination=run.path/f'item-{index:04d}-{name}-band-{band[0]}-{band[-1]}-{kind}.json'
                    if destination.exists():rows.append(json.loads(destination.read_text()));continue
                    transforms={}
                    for layer in band:
                        first='answer' if kind=='answer_swap' else 'intermediate';second='swap_answer' if kind=='answer_swap' else 'swap_to'
                        v=direction(model,maps[layer],ids[first])
                        if kind=='ablation':transforms[layer]=lambda h,v=v:ablate_direction(h,v)
                        else:
                            w=direction(model,maps[layer],ids[second]);vectors=torch.stack([v,w],dim=1)
                            transforms[layer]=lambda h,vectors=vectors:swap_coordinates(h,vectors)
                    if kind=='ablation':score=lambda logits:-logits.softmax(-1)[correct_ids].sum()
                    else:score=lambda logits:logits.softmax(-1)[target_ids].sum()
                    result=gate_band(model,tokens,transforms,positions,covariances,score,seed=config['seed']+index,count=100)
                    success=next_token_success(result.pop('intervened_logits'),target_ids if kind!='ablation' else correct_ids)
                    row=dict(item_id=source['name'],lens=name,label_tuned=name.startswith('J++'),band=band,kind=kind,**result,**success)
                    rows.append(row);write_json(destination,row);run.event('causal-item',item=source['name'],lens=name,kind=kind,band=band,passes=row['passes'])
    summaries=[];rng=np.random.default_rng(config['seed'])
    for name in bank:
        for kind in ('intermediate_swap','ablation','answer_swap'):
            selected=[r for r in rows if r.get('lens')==name and r.get('kind')==kind];groups=sorted(set(r['item_id'] for r in selected))
            if not groups:continue
            def metric(field):
                values=np.array([np.mean([r[field] for r in selected if r['item_id']==g]) for g in groups])
                return dict(mean=float(values.mean()),ci=np.quantile(values[rng.integers(len(values),size=(200,len(values)))].mean(1),[.025,.975]).tolist())
            summaries.append(dict(lens=name,label_tuned=name.startswith('J++'),kind=kind,items=len(groups),gate_pass_rate=metric('passes'),top1=metric('top1'),top5=metric('top5')))
    run.result(dict(rows=rows,summary=summaries,gates=gates,metadata=metadata,
        deviations=['Task-length prompts; all prompt positions patched, rather than fitting-position exclusions',
        'Token aliases averaged into one direction; top-1/top-5 are next-token success, not phrase success'],
        causal_claim_rule='Only effects strictly above the 95th percentile of 100 equal-norm covariance controls pass'))

if __name__=='__main__':main()
