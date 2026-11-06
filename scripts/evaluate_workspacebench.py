"""Exact-match adaptation using upstream registered banks, renders and read sites."""
import argparse,json,subprocess
from pathlib import Path
from dataclasses import asdict,replace
import numpy as np
import torch,yaml
from ilslens.model import load_local
from ilslens.experiment import Experiment,write_json,save_tensor
from ilslens.eval.workspacebench import load_reference,load_family,paired_accuracy,overlaps_labeled_dev
from ilslens.eval.readout_tasks import generate_token_completions,gate_item,exact_match,candidate_ids,recall_report
from ilslens.eval.contrastive import render_condition
from ilslens.artifacts import primary_bank
from ilslens.hooks import capture_blocks
from ilslens.readout import raw
from ilslens.jpp import readout_filter

def main():
    p=argparse.ArgumentParser();p.add_argument('--reference',required=True);p.add_argument('--config',default='configs/workspacebench.yaml')
    p.add_argument('--run',default='runs/qwen-main');p.add_argument('--output',default='runs/workspacebench')
    p.add_argument('--device',default='cuda');p.add_argument('--gate-only',action='store_true');p.add_argument('--jpp');args=p.parse_args()
    config=yaml.safe_load(Path(args.config).read_text());revision=subprocess.check_output(['git','-C',args.reference,'rev-parse','HEAD'],text=True).strip()
    run=Experiment(args.output,dict(config=config,reference_revision=revision,device=args.device))
    readplan,render,_,bank_items,_=load_reference(args.reference)
    model,tokenizer=load_local(args.device);layers=config['layers'];gates=[];prepared=[]
    jpp_artifact=torch.load(args.jpp,weights_only=True,map_location='cpu') if args.jpp else None
    dev_exclusions=[]
    for family in config['families']:
        specs={spec.id:spec for spec in readplan.plan(family,layers=layers,reads='full')}
        raw_items={row['id']:row for row in bank_items(family)}
        for index,entry in enumerate(load_family(args.reference,family,config['seed'])):
            item=entry['item'];identity=item.id.split(':',2)[2];spec=specs[identity];rendered=render(spec,tokenizer)
            positions=[position%len(rendered.ids) for position in readplan.resolve(spec.positions,rendered.tokens)]
            if not positions:raise ValueError(f'No read positions for {item.id}')
            gate_path=run.path/f'gate-{family}-{index:04d}.json'
            if gate_path.exists():gate=json.loads(gate_path.read_text())
            else:
                question_ids=render_condition(tokenizer,None,item.capability_prompt)
                samples=generate_token_completions(model,tokenizer,question_ids,seed=config['seed']+index*100,max_new_tokens=32)
                greedy=generate_token_completions(model,tokenizer,question_ids,draws=1,greedy=True,max_new_tokens=32)[0]['text']
                natural=generate_token_completions(model,tokenizer,rendered.ids,seed=config['seed']+index*100+20,max_new_tokens=32)
                effective=replace(item,prompt=tokenizer.decode(rendered.ids))
                gate=gate_item(effective,[s['text'] for s in samples],[s['text'] for s in natural],required=entry['required_correct'])
                gate.update(greedy_answer=greedy,greedy_correct=exact_match(greedy,item.answers),pair_id=entry['pair_id'],
                    source_file=entry['source_file'],read_spec=asdict(spec),positions=positions,input_ids=rendered.ids,
                    natural_token_completions=natural,temperature=.8)
                gate['retained']=gate['retained'] and gate['greedy_correct'];write_json(gate_path,gate)
            gates.append(gate)
            if gate['retained'] and item.split=='test':
                if jpp_artifact and overlaps_labeled_dev(item,jpp_artifact['metadata']):dev_exclusions.append(item.id)
                else:prepared.append((gate,raw_items[identity]))
    write_json(run.path/'gates.json',gates)
    # Capability and dev exclusion apply jointly to the two members of a role pair.
    role_pairs={}
    for gate,row in prepared:
        if gate['item']['task']=='role_bound_association':role_pairs.setdefault(gate['pair_id'],[]).append(gate['item']['id'])
    incomplete_role_items={identity for pair,identities in role_pairs.items() if pair is None or len(identities)!=2 for identity in identities}
    prepared=[(gate,row) for gate,row in prepared if gate['item']['id'] not in incomplete_role_items]
    # Export exact rendered/tokenized evaluation items for phrase and graph stages.
    exported=[]
    for gate,row in prepared:
        contrasts=[row[key] for key in ('flipped','contrast_label') if key in row]
        exported.append(dict(id=gate['item']['id'],family=gate['item']['task'],prompt=gate['item']['prompt'],input_ids=gate['input_ids'],
            positions=gate['positions'],targets=gate['item']['intermediates'],contrasts=contrasts,answers=gate['item']['answers'],
            actual_output=gate['natural_completions'][0],completion_ids=gate['natural_token_completions'][0]['token_ids'],pair_id=gate['pair_id'],split='test'))
    write_json(run.path/'evaluation-items.json',exported)
    if args.gate_only:
        run.result(dict(gated_items=len(gates),retained_test_items=len(prepared),status='gates_only'));return
    bank,metadata=primary_bank(args.run,config['epsilon'],config['gamma'])
    if args.jpp:
        artifact=jpp_artifact
        if artifact['metadata']['primary_fingerprint']!=metadata['fingerprint'] or not artifact['metadata']['label_tuned']:
            raise ValueError('Invalid J++ baseline')
        bank['J++ (label-tuned)']=artifact['matrices']
    rows=[];states_cache={}
    for i,(gate,_) in enumerate(prepared):
        ids=torch.tensor([gate['input_ids']],device=model.device)
        with torch.no_grad(),capture_blocks(model,layers) as states:model.model(ids,use_cache=False)
        states_cache[i]={layer:states[layer][0,gate['positions']].cpu() for layer in layers}
    save_tensor(run.path/'read-states.pt',dict(states=states_cache,metadata=metadata,items=exported))
    for name,maps in bank.items():
        excluded=readout_filter(tokenizer,model.config.vocab_size).to(model.device) if name.startswith('J++') else None
        for family in config['families']:
            chosen=[(i,gate) for i,(gate,_) in enumerate(prepared) if gate['item']['task']==family]
            max_concepts=max((len(gate['item']['intermediates']) for _,gate in chosen),default=1)
            selected_layers=[layer for layer in layers if layer in maps]
            ranks=np.full((len(chosen),len(selected_layers),max_concepts),np.nan)
            for i,(cache_id,gate) in enumerate(chosen):
                for l,layer in enumerate(selected_layers):
                    with torch.no_grad():scores=raw(states_cache[cache_id][layer].to(model.device),maps[layer].to(model.device),model.lm_head.weight,model.model.norm.weight)
                    if excluded is not None:scores=scores.masked_fill(excluded[None],-torch.inf)
                    for c,concept in enumerate(gate['item']['intermediates']):
                        token_ids=candidate_ids(tokenizer,concept)
                        if token_ids:
                            best=scores[:,token_ids].max(1).values
                            ranks[i,l,c]=int((scores>best[:,None]).sum(1).min())+1
            report=recall_report(ranks,[gate['item']['id'] for _,gate in chosen],seed=config['seed'])
            for row in report:
                row['layer']=selected_layers[row['layer']] if isinstance(row['layer'],int) else row['layer']
                rows.append(dict(row,lens=name,label_tuned=name.startswith('J++'),family=family,metric='single-token concept recall over any prescribed read position',mode='raw'))
    run.result(dict(rows=rows,metadata=metadata,retained_test_items=len(prepared),capability_items=len(gates),jpp_dev_exclusions=dev_exclusions,
        incomplete_role_pair_exclusions=sorted(incomplete_role_items),
        multi_token_status='Saved exact items and states for the causal phrase experiment',
        hallucination_status='Upstream WSB moved hallucination to HallucinationBench; no borrowed index is reported as this model.',
        deviations=['Exact-match scoring instead of the official LLM judge','Capability temperature .8 per project, versus .7 in WSB gates',
        'Strict project leakage exclusion can reject poetry, whose target is its completion; no silent waiver',
        'Token recall is an adapted metric, not the official prose/ordered-relation macro score']))

if __name__=='__main__':main()
