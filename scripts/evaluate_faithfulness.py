"""Independent held-out R2 and token-change rank correlation for every lens."""
import argparse,json
from pathlib import Path
import numpy as np
import torch,yaml
from ilslens.model import load_local
from ilslens.artifacts import primary_bank
from ilslens.data import wikitext_sequences
from ilslens.suffix import QwenSuffix
from ilslens.sampling import horizon_weights
from ilslens.eval.faithfulness import bootstrap_r2,token_spearman,bootstrap_prompt_mean,horizon_log_probability_change
from ilslens.experiment import Experiment,write_json

def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);p.add_argument('--config',default='configs/faithfulness.yaml')
    p.add_argument('--run',default='runs/qwen-main');p.add_argument('--output',default='runs/faithfulness');p.add_argument('--device',default='cuda');p.add_argument('--jpp');args=p.parse_args()
    config=yaml.safe_load(Path(args.config).read_text());run=Experiment(args.output,dict(config=config,device=args.device,token_metric_version='actual-position-logprobs-v2'))
    model,tokenizer=load_local(args.device);ids,source_rows=wikitext_sequences(args.dataset,tokenizer,'validation',config['prompts'],offset=256)
    donor_ids,donor_rows=wikitext_sequences(args.dataset,tokenizer,'validation',config['prompts'],offset=256+config['prompts'])
    layers=config['layers'];donors=[]
    for tokens in donor_ids:
        cache=QwenSuffix(model,tokens[None]);donors.append({l:cache.states[l][0,16:-1].cpu() for l in layers})
    rows=[]
    for epsilon in config['epsilons']:
        for gamma in config['gammas']:
            bank,metadata=primary_bank(args.run,epsilon,gamma)
            if args.jpp:
                artifact=torch.load(args.jpp,weights_only=True,map_location='cpu')
                if not artifact['metadata']['label_tuned'] or artifact['metadata']['primary_fingerprint']!=metadata['fingerprint']:raise ValueError('J++ provenance mismatch')
                bank['J++ (label-tuned)']=artifact['matrices']
            for layer in layers:
                path=run.path/f'layer-{layer}-epsilon-{epsilon}-gamma-{gamma}.json'
                if path.exists():rows.extend(json.loads(path.read_text()));continue
                ys=[];predictions={name:[] for name,maps in bank.items() if layer in maps};correlations={name:[] for name in predictions};groups=[]
                for index,tokens in enumerate(ids):
                    cache=QwenSuffix(model,tokens[None]);generator=torch.Generator().manual_seed(config['seed']+index*100+layer)
                    for _ in range(config['samples_per_prompt']):
                        t=int(torch.randint(16,127,(),generator=generator));j=int(torch.randint(len(donors),(),generator=generator));u=int(torch.randint(111,(),generator=generator))
                        delta=epsilon*(donors[j][layer][u].to(model.device)-cache.states[layer][0,t])
                        plus,minus=cache.effects(layer,t,delta[None],return_position_branches=True);plus=plus[0];minus=minus[0]
                        weight=horizon_weights(t,128,gamma).to(model.device)
                        def log_probs(h):return model.lm_head(model.model.norm(h)).log_softmax(-1)
                        actual=horizon_log_probability_change(model,plus,minus,weight)
                        ys.append((((plus-minus)/2)*weight[:,None]).sum(0).cpu());groups.append(index)
                        for name in predictions:
                            predicted=bank[name][layer].to(model.device)@delta;predictions[name].append(predicted.cpu())
                            with torch.no_grad():
                                center=bank[name][layer].to(model.device)@cache.states[layer][0,t]
                                predicted_scores=(log_probs(center+predicted)-log_probs(center-predicted))/2
                            correlations[name].append(token_spearman(predicted_scores,actual,1000))
                y=torch.stack(ys);group_ids=torch.tensor(groups);local=[];rng=np.random.default_rng(config['seed'])
                for name,predicted in predictions.items():
                    r=bootstrap_r2(y,torch.stack(predicted),group_ids,seed=config['seed'])
                    correlation=bootstrap_prompt_mean(np.array(correlations[name]).reshape(len(ids),config['samples_per_prompt']),seed=config['seed'])
                    local.append(dict(lens=name,label_tuned=name.startswith('J++'),layer=layer,epsilon=epsilon,gamma=gamma,N=len(y),
                        **r,spearman=correlation['mean'],spearman_ci=correlation['ci'],spearman_sampling=correlation))
                write_json(path,local);rows.extend(local);run.event('faithfulness-cell',layer=layer,epsilon=epsilon,gamma=gamma)
    run.result(dict(rows=rows,source_provenance=source_rows,donor_provenance=donor_rows,
        token_change_estimand='Horizon-weighted antithetic change in actual per-position next-token log probabilities; top1000 selected by absolute actual change',
        prediction='Antithetic log-softmax change of the lens applied to the source residual h +/- delta; no actual-final-state centering',
        bootstrap_unit='source_prompt'))

if __name__=='__main__':main()
