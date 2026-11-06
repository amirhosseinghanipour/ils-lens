"""Train a concealed indirect preference and evaluate base/tuned behavior and lens reads."""
import argparse,json
from pathlib import Path
import torch,yaml
from ilslens.model import load_local
from ilslens.lora import AdapterSet,train_adapters
from ilslens.experiment import Experiment,save_tensor,write_json
from ilslens.eval.planted import planted_examples,model_diff_faithfulness
from ilslens.eval.behavior import measure_preference,preference_difference
from ilslens.eval.readout_tasks import candidate_ids
from ilslens.eval.contrastive import render_condition,completion_states
from ilslens.artifacts import primary_bank
from ilslens.contrast import contrast_with_null,rank_of
from ilslens.readout import raw
from ilslens.data import wikitext_sequences,SAMPLING_POLICY
from ilslens.eval.faithfulness import bootstrap_r2
from ilslens.graphs.items import coverage_context


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/planted.yaml')
    p.add_argument('--prompts',default='configs/stress_prompts.json');p.add_argument('--output',default='runs/planted')
    p.add_argument('--run',default='runs/qwen-main');p.add_argument('--dataset');p.add_argument('--device',default='cuda')
    p.add_argument('--phase',choices=['train-behavior','detect','all'],default='all');args=p.parse_args()
    config=yaml.safe_load(Path(args.config).read_text());prompts=json.loads(Path(args.prompts).read_text())
    run=Experiment(args.output,dict(config=config,sampling_policy=SAMPLING_POLICY,model_diff_protocol='wiki-and-matched-behavior-v2',contrast_protocol='batched-prompt-sums-flat-undefined-v2',evaluation_prompts=prompts,device=args.device,ranking='score-descending-id-ascending-v2'))
    model,tokenizer=load_local(args.device);torch.manual_seed(config['seed'])
    adapters=AdapterSet(model,config['rank'],config['alpha'])
    checkpoint=run.path/'adapter.pt'
    if checkpoint.exists():adapters.load(torch.load(checkpoint,weights_only=True)['adapter'])
    elif args.phase=='detect':raise FileNotFoundError('Train planted adapter before detection')
    else:
        examples=planted_examples(tokenizer,config['concept'],config['max_length'])
        losses=train_adapters(model,adapters,examples,config['steps'],config['learning_rate'],config['seed'])
        save_tensor(checkpoint,dict(adapter=adapters.state(),metadata=dict(fingerprint=run.fingerprint,config=config,losses=losses,
            distinct_examples=len(examples),token_presentations_upper_bound=config['steps']*config['max_length'])))
        run.event('adapter-trained',distinct_examples=len(examples),steps=config['steps'])
    records={}
    for arm in ('tuned','base'):
        path=run.path/f'{arm}-behavior.json'
        if path.exists():records[arm]=json.loads(path.read_text());continue
        from contextlib import nullcontext
        with adapters.disabled() if arm=='base' else nullcontext():
            records[arm]=measure_preference(model,tokenizer,prompts,config['concept'],config['evaluation_draws'],config['seed']+100000)
        write_json(path,records[arm])
    behavior=preference_difference(records['tuned'],records['base'],seed=config['seed'])
    write_json(run.path/'behavior-comparison.json',behavior)
    # Off-policy graph comparisons must use the same exact base completion IDs.
    graph_items=[]
    for index,user in enumerate(prompts):
        completion=records['base'][index]['token_completions'][0]
        if len(completion)<4:continue
        prefix=render_condition(tokenizer,None,user)
        graph_items.append(dict(id=f'planted:{index}',family='planted',prompt=user,input_ids=prefix+completion,
            positions=[len(prefix)+offset for offset in (0,1,2,3)],targets=[config['concept']],
            answers=[config['concept']],actual_output=records['base'][index]['completions'][0]))
    write_json(run.path/'graph-items.json',graph_items)
    run.event('behavior-evaluated',**behavior)
    if args.phase=='train-behavior':
        run.result(dict(behavior=behavior,detection_status='not_requested',trait_verified=behavior['statistically_detected']));return
    if not args.dataset:raise ValueError('--dataset is required for held-out model-difference faithfulness')
    bank,metadata=primary_bank(args.run,config['epsilon'],config['gamma']);layers=config['layers']
    target=candidate_ids(tokenizer,config['concept'])
    if not target:raise ValueError('Concept requires phrase readout')
    states={mode:{arm:{layer:[] for layer in layers} for arm in ('tuned','base')} for mode in ('on_policy','off_policy')}
    excluded=[];groups=[]
    # First sampled completion per prompt, four matched offsets, fixed before results.
    for i,user in enumerate(prompts):
        texts={arm:records[arm][i]['token_completions'][0] for arm in ('tuned','base')}
        if min(map(len,texts.values()))<4:excluded.append(i);continue
        prefix=render_condition(tokenizer,None,user)
        for mode,arms in states.items():
            for arm in arms:
                from contextlib import nullcontext
                with adapters.disabled() if arm=='base' else nullcontext():
                    h=completion_states(model,prefix,texts[arm if mode=='on_policy' else 'base'],layers,[0,1,2,3])
                for layer in layers:arms[arm][layer].append(h[layer])
        groups.extend([i]*4)
    rows=[]
    if len(set(groups))<4:raise ValueError('Too few matched completions for the registered split-half null')
    groups=torch.tensor(groups,device=model.device)
    for mode,arms in states.items():
        for layer in layers:
            tuned=torch.cat(arms['tuned'][layer]);base=torch.cat(arms['base'][layer])
            for name,maps in bank.items():
                matrix=maps[layer].to(model.device)
                def read(h):return raw(h,matrix,model.lm_head.weight,model.model.norm.weight).softmax(-1)
                with torch.no_grad():
                    result=contrast_with_null(tuned,base,read,target,groups,seed=config['seed'])
                    absolute={mode:{arm:rank_of(read(h).mean(0) if mode=='mean_of_lens' else read(h.mean(0)),target)
                        for arm,h in [('base',base),('tuned',tuned)]} for mode in ('mean_of_lens','lens_of_mean')}
                    model_diff_rank={'lens_of_mean':rank_of(read((tuned-base).mean(0)),target),
                                     'mean_of_lens':rank_of(read(tuned-base).mean(0),target)}
                for read_mode,statistics in result.items():
                    save_tensor(run.path/f'contrast-{name}-{layer}-{mode}-{read_mode}.pt',statistics.pop('contrast_scores'))
                    rows.append(dict(lens=name,layer=layer,mode=mode,read_mode=read_mode,absolute_ranks=absolute,
                                     model_diff_lens_rank=model_diff_rank,**statistics))
    sequences,_=wikitext_sequences(args.dataset,tokenizer,'validation',32,offset=64)
    faithfulness=[]
    for layer in layers:
        result=model_diff_faithfulness(model,adapters,sequences,layer,bank['ILS'][layer],[64]*len(sequences),gamma=config['gamma'])
        x=result.pop('delta');y=result.pop('effect');save_tensor(run.path/f'model-diff-layer-{layer}.pt',dict(delta=x,response=y,metadata=metadata))
        statistics=bootstrap_r2(y,x@bank['ILS'][layer].T,torch.arange(len(x)),seed=config['seed'])
        faithfulness.append(dict(result,layer=layer,**statistics))
    # Trait-context diagnostic: use exactly the off-policy graph contexts/sites,
    # including short prompts and final read positions via an excluded EOS sentinel.
    behavior_sequences=[];behavior_positions=[];behavior_groups=[]
    for index,item in enumerate(graph_items):
        context=coverage_context(torch.tensor([item['input_ids']],device=model.device),tokenizer)[0]
        for position in item['positions']:
            behavior_sequences.append(context);behavior_positions.append(position);behavior_groups.append(index)
    behavior_faithfulness=[]
    for layer in layers:
        if not behavior_sequences:break
        result=model_diff_faithfulness(model,adapters,behavior_sequences,layer,bank['ILS'][layer],behavior_positions,
            gamma=config['gamma'],skip=0)
        x=result.pop('delta');y=result.pop('effect');prompt_ids=torch.tensor(behavior_groups)
        save_tensor(run.path/f'behavior-model-diff-layer-{layer}.pt',dict(delta=x,response=y,prompt_ids=prompt_ids,
            positions=behavior_positions,items=graph_items,metadata=metadata))
        statistics=bootstrap_r2(y,x@bank['ILS'][layer].T,prompt_ids,seed=config['seed'])
        behavior_faithfulness.append(dict(result,layer=layer,**statistics,contexts=len(graph_items),sites=len(x),
            bootstrap_unit='prompt',context='Matched base completion under tuned/base models; original future positions, excluded EOS sentinel, skip0'))
        run.event('behavior-model-diff',layer=layer,**statistics)
    run.result(dict(behavior=behavior,trait_verified=behavior['statistically_detected'],rows=rows,model_diff_faithfulness=faithfulness,
        behavior_model_diff_faithfulness=behavior_faithfulness,
        excluded_short_completion_pairs=excluded,metadata=metadata,graph_status='Requires the graph evaluation stage with this adapter',
        limitation='Detection measurements remain conditional on successful behavioral planting; model-difference R2 is not total trait variance explained.'))

if __name__=='__main__':main()
