"""Antithetic phrase-log-probability regression with a whitened template baseline."""
import argparse,json
from pathlib import Path
import numpy as np
import torch,yaml
from ilslens.model import load_local
from ilslens.data import wikitext_sequences,SAMPLING_POLICY
from ilslens.hooks import capture_blocks
from ilslens.suffix import QwenSuffix
from ilslens.covariance import residual_moments
from ilslens.templatelens import phrase_ids,antithetic_phrase_effect,fit_phrase_rows,whitened_template
from ilslens.eval.faithfulness import bootstrap_r2
from ilslens.eval.workspacebench import contradiction_check,phrase_recall_report,phrase_gate_report
from ilslens.gate import random_direction_gate
from ilslens.readout import target_ranks
from ilslens.experiment import Experiment,save_tensor,write_json

def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);p.add_argument('--items',required=True)
    p.add_argument('--dictionary',default='configs/phrases.json');p.add_argument('--config',default='configs/template.yaml')
    p.add_argument('--dictionary-items',help='Additional evaluation exports whose target phrases enter the fixed dictionary, without using outcomes')
    p.add_argument('--output',default='runs/template');p.add_argument('--device',default='cuda')
    p.add_argument('--phase',choices=['collect','fit','evaluate','all'],default='all');args=p.parse_args()
    config=yaml.safe_load(Path(args.config).read_text());items=json.loads(Path(args.items).read_text())
    entries=json.loads(Path(args.dictionary).read_text());model,tokenizer=load_local(args.device);d=model.config.hidden_size
    # Targets are allowed in the dictionary by the protocol. Their evaluation labels
    # never select a penalty, layer, perturbation magnitude or estimator.
    dictionary_items=json.loads(Path(args.dictionary_items).read_text()) if args.dictionary_items else []
    phrases=list(dict.fromkeys(entries+[target for item in items+dictionary_items for target in item['targets']+item.get('contrasts',[])]))
    accepted=[];excluded=[]
    for phrase in phrases:
        count=len(tokenizer.encode(' '+phrase,add_special_tokens=False))
        (accepted if 2<=count<=5 else excluded).append(phrase)
    if not accepted:raise ValueError('No 2–5-token phrases under this model tokenizer')
    tokens=phrase_ids(tokenizer,[' '+phrase for phrase in accepted]);phrase_tokens=[tokens[' '+phrase] for phrase in accepted]
    run=Experiment(args.output,dict(config=config,sampling_policy=SAMPLING_POLICY,items=items,phrases=accepted,excluded_phrases=excluded,device=args.device,
        ranking='score-descending-id-ascending-v2',zero_response='common_zero_ridge_solution_no_penalty_selection',
        solver='shared_factorization_ridge_rhs_per_phrase_cv'))
    suffix=tokenizer.encode(config['probe_suffix'],add_special_tokens=False);layers=config['layers']
    training,training_rows=wikitext_sequences(args.dataset,tokenizer,'train',config['source_prompts'])
    donors,donor_rows=wikitext_sequences(args.dataset,tokenizer,'train',config['donor_prompts'],offset=config['source_prompts'])
    validation,validation_rows=wikitext_sequences(args.dataset,tokenizer,'validation',config['heldout_prompts'],offset=64)
    held_donors,held_donor_rows=wikitext_sequences(args.dataset,tokenizer,'validation',config['heldout_prompts'],offset=96)
    write_json(run.path/'data-provenance.json',dict(training=training_rows,donors=donor_rows,validation=validation_rows,held_donors=held_donor_rows))
    moments=residual_moments(model,training,layers)
    if args.phase in ('collect','all'):
        for split,sequences,donor_sequences,total in [('train',training,donors,config['samples_multiplier']*d),
                                                       ('validation',validation,held_donors,config['heldout_samples'])]:
            if total%len(sequences):raise ValueError('Sample budget must divide source prompt count')
            donor_states=[]
            for ids in donor_sequences:
                replay=QwenSuffix(model,ids[None]);donor_states.append({l:replay.states[l][0,16:-1].cpu() for l in layers})
            for prompt_index,ids in enumerate(sequences):
                replay=QwenSuffix(model,ids[None])
                for layer in layers:
                    path=run.path/'samples'/split/f'prompt-{prompt_index:03d}-layer-{layer:02d}.pt'
                    if path.exists():continue
                    generator=torch.Generator().manual_seed(config['seed']+prompt_index*100+layer+(100000 if split=='validation' else 0))
                    xs=[];ys=[];sites=[]
                    for _ in range(total//len(sequences)):
                        t=int(torch.randint(16,127,(),generator=generator));j=int(torch.randint(len(donor_states),(),generator=generator));u=int(torch.randint(111,(),generator=generator))
                        delta=config['epsilon']*(donor_states[j][layer][u].to(model.device)-replay.states[layer][0,t])
                        effects=[antithetic_phrase_effect(model,ids.tolist(),suffix,phrase,layer,t,delta) for phrase in phrase_tokens]
                        xs.append(delta.cpu());ys.append(torch.stack(effects).cpu());sites.append(t)
                    save_tensor(path,dict(delta=torch.stack(xs),response=torch.stack(ys),positions=sites,
                        prompt_ids=torch.full((len(xs),),prompt_index),metadata=dict(fingerprint=run.fingerprint,layer=layer,split=split,epsilon=config['epsilon'],
                            target='phrase_log_probability_after_fixed_probe_suffix',probe_suffix=config['probe_suffix'],phrases=accepted)))
                    run.event('phrase-samples',split=split,prompt=prompt_index,layer=layer,N=len(xs))
    def samples(split,layer):
        files=sorted((run.path/'samples'/split).glob(f'prompt-*-layer-{layer:02d}.pt'))
        count=config['source_prompts'] if split=='train' else config['heldout_prompts']
        if len(files)!=count:raise ValueError(f'Incomplete {split} collection')
        values=[torch.load(path,weights_only=True,map_location='cpu') for path in files]
        if any(v['metadata']['fingerprint']!=run.fingerprint for v in values):raise ValueError('Sample provenance mismatch')
        return [torch.cat([v[key] for v in values]).to(model.device) for key in ('delta','response','prompt_ids')]
    if args.phase in ('fit','all'):
        for layer in layers:
            destination=run.path/f'rows-layer-{layer:02d}.pt'
            if destination.exists():continue
            x,y,groups=samples('train',layer);a,selection,cv=fit_phrase_rows(x,y,groups,config['lambdas'],config['seed'])
            templates=[]
            for phrase in accepted:
                # Predeclared synthetic verbal assembly contexts; no readout-eval outcomes.
                words=phrase.split();quoted=', '.join(repr(word) for word in words)
                contexts=[f'Write these words in their given order, separated by spaces: {quoted}. The resulting phrase is',
                          f'The ordered words are {quoted}. Joining them gives',
                          f'Combine the following words without changing their order: {quoted}. Answer:']
                states=[]
                for text in contexts:
                    ids=tokenizer(text,return_tensors='pt').input_ids.to(model.device)
                    with torch.no_grad(),capture_blocks(model,[layer]) as h:model.model(ids,use_cache=False)
                    states.append(h[layer][0,-1])
                templates.append(whitened_template(torch.stack(states),moments[layer]['mean'],moments[layer]['covariance'],config['template_eta']))
            save_tensor(destination,dict(ils=a.cpu(),template=torch.stack(templates).cpu(),selection=selection,cv=cv,
                metadata=dict(fingerprint=run.fingerprint,layer=layer,phrases=accepted,N=len(x),epsilon=config['epsilon'],
                    prior='zero',estimator='ridge',template_eta=config['template_eta'],template_passages='three fixed word-assembly templates',label_tuned=False)))
    if args.phase not in ('evaluate','all'):
        run.result(dict(status=f'{args.phase}_only',excluded_phrases=excluded));return
    rows=[];faithfulness=[]
    for layer in layers:
        artifact=torch.load(run.path/f'rows-layer-{layer:02d}.pt',weights_only=True,map_location=model.device)
        if artifact['metadata']['fingerprint']!=run.fingerprint:raise ValueError('Phrase-row provenance mismatch')
        x,y,groups=samples('validation',layer)
        for method in ('ils','template'):
            matrix=artifact[method]
            for column,phrase in enumerate(accepted):
                statistics=bootstrap_r2(y[:,column:column+1],x@matrix[column:column+1].T,groups,seed=config['seed'])
                faithfulness.append(dict(layer=layer,method=method,phrase=phrase,**statistics))
            for index,item in enumerate(items):
                ids=torch.tensor([item['input_ids']],device=model.device)
                with torch.no_grad(),capture_blocks(model,[layer]) as states:model.model(ids,use_cache=False)
                for position in item['positions']:
                    h=states[layer][0,position];scores=matrix@h;top=int(scores.argmax());readout=accepted[top]
                    for phrase in item['targets']:
                        if phrase not in accepted:continue
                        column=accepted.index(phrase);row=matrix[column]
                        direction=(torch.zeros_like(row) if row.norm()==0 else
                            row/row.norm()*config['epsilon']*(2*moments[layer]['covariance'].trace()).sqrt())
                        def effect(delta):return float(antithetic_phrase_effect(model,item['input_ids'],suffix,phrase_tokens[column],layer,position,delta))
                        gate=random_direction_gate(direction,moments[layer]['covariance'],effect,seed=config['seed']+index,count=100)
                        hallucination=contradiction_check(readout,item['actual_output'],item['answers'],item.get('contrasts',[]))
                        rows.append(dict(item_id=item['id'],family=item['family'],pair_id=item.get('pair_id'),layer=layer,position=position,
                            method=method,target=phrase,rank=int(target_ranks(scores,[column])),readout=readout,
                            flat_readout=bool((scores==scores[0]).all()),
                            fit_status=artifact['selection'][column].get('status','legacy_unspecified') if method=='ils' else 'whitened_template',
                            gate=gate,hallucination=hallucination))
                run.event('phrase-item',item=item['id'],layer=layer,method=method)
    summaries=phrase_recall_report(rows,layers,seed=config['seed'])
    run.result(dict(rows=rows,summary=summaries,causal_summary=phrase_gate_report(rows,config['seed']),interventional_faithfulness=faithfulness,excluded_phrases=excluded,
        calibration='Phrase scores are linear rows on raw residuals; no token-lens RMSNorm is applied',
        deviations=['Three deterministic word-assembly synthetic passages per phrase for the whitened template baseline',
        'Phrase dictionary is known by design; only prompt-fold interventional R2 selects each ridge penalty',
        'Exactly zero responses have the common zero-prior ridge solution for every penalty; lambda and R2 remain undefined, with no CV winner invented',
        'Targets outside 2–5 model tokens are explicitly excluded; ordered phrases are never inferred from token bags']))

if __name__=='__main__':main()
