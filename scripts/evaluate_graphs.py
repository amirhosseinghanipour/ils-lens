"""Per-item direct attribution, finite-edge validation and lens/graph budget closure."""
import argparse,json
from pathlib import Path
import torch,yaml
from ilslens.model import load_local
from ilslens.artifacts import primary_bank
from ilslens.data import wikitext_sequences
from ilslens.transcoders.io import load_transcoders
from ilslens.graphs.qwen import QwenGraph
from ilslens.graphs.evaluate import evaluate_target
from ilslens.suffix import QwenSuffix
from ilslens.eval.readout_tasks import candidate_ids
from ilslens.eval.faithfulness import bootstrap_r2
from ilslens.eval.budget import close_budget
from ilslens.experiment import Experiment,save_tensor,write_json
from ilslens.lora import AdapterSet

def main():
    p=argparse.ArgumentParser();p.add_argument('--items',required=True,help='JSON list: id, prompt, token, optional contrast_token')
    p.add_argument('--dataset',required=True);p.add_argument('--transcoders',default='runs/transcoders')
    p.add_argument('--run',default='runs/qwen-main');p.add_argument('--config',default='configs/graphs.yaml')
    p.add_argument('--output',default='runs/graphs');p.add_argument('--device',default='cuda');p.add_argument('--adapter');args=p.parse_args()
    config=yaml.safe_load(Path(args.config).read_text());items=json.loads(Path(args.items).read_text())
    run=Experiment(args.output,dict(config=config,items=items,adapter=args.adapter,device=args.device))
    model,tokenizer=load_local(args.device);bank,metadata=primary_bank(args.run,config['epsilon'],config['gamma'])
    if args.adapter:
        artifact=torch.load(args.adapter,weights_only=True,map_location='cpu')['adapter'];adapters=AdapterSet(model,**artifact['config']);adapters.load(artifact)
        for parameter in adapters.parameters():parameter.requires_grad_(False)
    transcoders=load_transcoders(args.transcoders,model)
    donor_ids,provenance=wikitext_sequences(args.dataset,tokenizer,'validation',config['donor_prompts'],offset=128)
    donors=[]
    for ids in donor_ids:
        cache=QwenSuffix(model,ids[None]);donors.append({l:cache.states[l][0,16:-1].cpu() for l in config['layers']})
    rows=[]
    for index,item in enumerate(items):
        tokens=tokenizer(item['prompt'],return_tensors='pt').input_ids.to(model.device);position=item.get('position',tokens.shape[1]-2)
        if not 16<=position<tokens.shape[1]-1:raise ValueError('Coverage site must exclude first16 and final token')
        token_ids=candidate_ids(tokenizer,item['token'])
        if not token_ids:raise ValueError('Graph target token must be single-token; phrase graphs require their own target')
        token=token_ids[0];contrast=None
        if item.get('contrast_token'):
            candidates=candidate_ids(tokenizer,item['contrast_token'])
            if not candidates:raise ValueError('Contrast token must be single-token')
            contrast=candidates[0]
        graph=QwenGraph(model,tokens,transcoders);cache=QwenSuffix(model,tokens)
        for layer in config['layers']:
            destination=run.path/f'item-{index:04d}-layer-{layer:02d}.json'
            if destination.exists():rows.append(json.loads(destination.read_text()));continue
            matrix=bank['ILS'][layer].to(model.device)
            generator=torch.Generator().manual_seed(config['seed']+index*100+layer)
            delta=[];groups=[]
            for donor_index,donor in enumerate(donors):
                for _ in range(config['samples_per_donor']):
                    u=int(torch.randint(111,(),generator=generator));delta.append(config['epsilon']*(donor[layer][u].to(model.device)-cache.states[layer][0,position]))
                    groups.append(donor_index)
            delta=torch.stack(delta);effects=[]
            for batch in delta.split(config['intervention_batch_size']):effects.append(cache.effects(layer,position,batch,gammas=(config['gamma'],))[:,0])
            response=torch.cat(effects);statistics=bootstrap_r2(response,delta@matrix.T,torch.tensor(groups,device=model.device),seed=config['seed'])
            vector=(model.lm_head.weight[token]*model.model.norm.weight)@matrix
            target=dict(kind='lens',layer=layer,position=position,vector=vector)
            result=evaluate_target(graph,target,edges_per_kind=config['edges_per_kind'])
            attribution=result.pop('contributions');save_tensor(run.path/f'attribution-{index:04d}-{layer:02d}.pt',dict(attribution=attribution,target=vector.cpu(),metadata=metadata))
            row=dict(item_id=item['id'],prompt=item['prompt'],layer=layer,position=position,token=item['token'],lens_r2=statistics['r2'],lens_r2_ci=statistics['ci'],**result)
            write_json(destination,row);rows.append(row)
            # Separately validate a logit/logit-difference target.
            logit=evaluate_target(graph,dict(kind='logit',position=tokens.shape[1]-1,token=token,contrast_token=contrast),config['edges_per_kind'])
            logit.pop('contributions');write_json(run.path/f'logit-{index:04d}-{layer:02d}.json',logit)
            if not result['all_edges_pass'] or not logit['all_edges_pass']:
                raise AssertionError('Graph finite-edge validation failed; artifacts retained for diagnosis')
            run.event('graph-item',item=item['id'],layer=layer,lens_r2=statistics['r2'],graph_completeness=result['graph_completeness'])
    run.result(dict(budget=close_budget(rows,config['coverage_threshold'],config['completeness_threshold']),metadata=metadata,
        donor_provenance=provenance,adapter=args.adapter,
        deviations=['Direct source-to-target attribution graph with independent clean feature sources',
                    'Per-item task-length context and donor-group bootstrap; lens fitting retains the registered 128-token convention'],
        interpretation='Graph completeness is absolute attribution mass, not predictive R2; both quantities remain separate.'))

if __name__=='__main__':main()
