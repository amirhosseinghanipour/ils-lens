"""Exact-site attribution, finite-edge validation and itemwise budget closure."""
import argparse,json
from pathlib import Path
import torch,yaml
from ilslens.model import load_local
from ilslens.artifacts import primary_bank
from ilslens.data import wikitext_sequences
from ilslens.transcoders.io import load_transcoders
from ilslens.graphs.qwen import QwenGraph
from ilslens.graphs.evaluate import evaluate_target
from ilslens.graphs.items import tokenized_item,coverage_context,target_vector
from ilslens.graphs.trace import trace_graph
from ilslens.suffix import QwenSuffix
from ilslens.eval.readout_tasks import candidate_ids
from ilslens.eval.faithfulness import bootstrap_r2
from ilslens.eval.budget import close_budget
from ilslens.experiment import Experiment,save_tensor,write_json
from ilslens.lora import AdapterSet

def main():
    p=argparse.ArgumentParser();p.add_argument('--items',required=True,help='Exact benchmark exports, or id/prompt/token records')
    p.add_argument('--dataset',required=True);p.add_argument('--transcoders',default='runs/transcoders')
    p.add_argument('--run',default='runs/qwen-main');p.add_argument('--config',default='configs/graphs.yaml')
    p.add_argument('--output',default='runs/graphs');p.add_argument('--device',default='cuda');p.add_argument('--adapter')
    p.add_argument('--phrase-rows',help='Optional directory of fitted phrase ILS rows')
    p.add_argument('--exploratory-affine',action='store_true',help='Explicitly opt into the non-proposal independent-feature freeze')
    args=p.parse_args()
    if not args.exploratory_affine:
        raise RuntimeError('This graph runner uses an additional independent-feature freeze and is not proposal-compliant. '
                           'Production is blocked; --exploratory-affine is only for separately labeled exploratory results.')
    config=yaml.safe_load(Path(args.config).read_text());items=json.loads(Path(args.items).read_text())
    run=Experiment(args.output,dict(config=config,items=items,adapter=args.adapter,phrase_rows=args.phrase_rows,device=args.device,
        graph_protocol='direct-live-zero-target-v3',exploratory_affine=True,
        coverage_convention='Exact task context plus excluded EOS sentinel, source skip=0; WikiText fitting remains skip16'))
    model,tokenizer=load_local(args.device);bank,metadata=primary_bank(args.run,config['epsilon'],config['gamma'])
    if args.adapter:
        artifact=torch.load(args.adapter,weights_only=True,map_location='cpu')['adapter'];adapters=AdapterSet(model,**artifact['config']);adapters.load(artifact)
        for parameter in adapters.parameters():parameter.requires_grad_(False)
    transcoders=load_transcoders(args.transcoders,model)
    donor_ids,provenance=wikitext_sequences(args.dataset,tokenizer,'validation',config['donor_prompts'],offset=128)
    donors=[]
    for ids in donor_ids:
        cache=QwenSuffix(model,ids[None]);donors.append({l:cache.states[l][0,16:-1].cpu() for l in config['layers']})
    rows=[];phrase_rows={}
    for layer in config['layers']:
        path=Path(args.phrase_rows)/f'rows-layer-{layer:02d}.pt' if args.phrase_rows else None
        phrase_rows[layer]=torch.load(path,weights_only=True,map_location='cpu') if path and path.exists() else None
    for index,item in enumerate(items):
        tokens,positions=tokenized_item(item,tokenizer,model.device)
        graph=QwenGraph(model,tokens,transcoders);cache=QwenSuffix(model,coverage_context(tokens,tokenizer))
        original=QwenSuffix(model,tokens)
        for layer in config['layers']:torch.testing.assert_close(cache.states[layer][:,:-1],original.states[layer],atol=2e-6,rtol=2e-5)
        del original
        labels=item.get('targets',[item.get('token')])
        if not labels or any(label is None for label in labels):raise ValueError('A graph item needs explicit target labels')
        for position in positions:
            for layer in config['layers']:
                matrix=bank['ILS'][layer].to(model.device)
                generator=torch.Generator().manual_seed(config['seed']+index*10000+position*100+layer)
                delta=[];groups=[]
                for donor_index,donor in enumerate(donors):
                    for _ in range(config['samples_per_donor']):
                        u=int(torch.randint(len(donor[layer]),(),generator=generator));delta.append(config['epsilon']*(donor[layer][u].to(model.device)-cache.states[layer][0,position]))
                        groups.append(donor_index)
                delta=torch.stack(delta);effects=[]
                for batch in delta.split(config['intervention_batch_size']):effects.append(cache.effects(layer,position,batch,gammas=(config['gamma'],),skip=0)[:,0])
                response=torch.cat(effects);statistics=bootstrap_r2(response,delta@matrix.T,torch.tensor(groups,device=model.device),seed=config['seed'])
                for target_index,label in enumerate(labels):
                    stem=f'item-{index:04d}-site-{position:04d}-layer-{layer:02d}-target-{target_index:02d}'
                    destination=run.path/(stem+'.json')
                    if destination.exists():rows.append(json.loads(destination.read_text()));continue
                    vector,target_metadata=target_vector(label,tokenizer,model,matrix,phrase_rows[layer])
                    row=dict(item_id=item['id'],family=item.get('family'),prompt=item.get('prompt'),layer=layer,position=position,token=label,
                        lens_r2=statistics['r2'],lens_r2_ci=statistics['ci'],target_metadata=target_metadata,
                        coverage_scope='All original future tokens; appended EOS excluded; task source skip0')
                    if vector is None:row.update(graph_completeness=None,all_edges_pass=None,graph_status='unsupported_target')
                    elif torch.count_nonzero(vector)==0:
                        row.update(graph_completeness=None,all_edges_pass=None,graph_status='zero_target',target_value=0.,
                            reason='Identically zero linear target; attribution share is 0/0 and no directional edge is claimed')
                    else:
                        target=dict(kind='lens',layer=layer,position=position,vector=vector)
                        result=evaluate_target(graph,target,edges_per_kind=config['edges_per_kind'])
                        if config.get('trace_feature_targets',0):
                            structure=trace_graph(graph,target,result,max_feature_targets=config['trace_feature_targets'],
                                max_depth=config['trace_depth'],parents_per_target=config['trace_parents'],edges_per_kind=config['edges_per_kind'])
                            write_json(run.path/(stem+'.graph.json'),structure)
                            result['all_traced_edges_pass']=structure['all_checked_edges_pass']
                            result['expanded_feature_nodes']=len(structure['expanded_features'])
                        attribution=result.pop('contributions')
                        save_tensor(run.path/(stem+'.pt'),dict(attribution=attribution,target=vector.cpu(),metadata=metadata,target_metadata=target_metadata))
                        row.update(result,graph_status='measured')
                    write_json(destination,row);rows.append(row)
                    if row['all_edges_pass'] is False:raise AssertionError('Graph finite-edge validation failed; artifacts retained')
                    run.event('graph-item',item=item['id'],layer=layer,position=position,target=label,lens_r2=statistics['r2'],graph_completeness=row['graph_completeness'])
        answer=item.get('answers',[item.get('token')])[0]
        answer_ids=candidate_ids(tokenizer,answer) if answer is not None else []
        contrast_ids=candidate_ids(tokenizer,item['contrast_token']) if item.get('contrast_token') else []
        if answer_ids:
            logit=evaluate_target(graph,dict(kind='logit',position=tokens.shape[1]-1,token=answer_ids[0],contrast_token=contrast_ids[0] if contrast_ids else None),config['edges_per_kind'])
            logit.pop('contributions');write_json(run.path/f'logit-{index:04d}.json',logit)
            if not logit['all_edges_pass']:raise AssertionError('Logit graph finite-edge validation failed')
    run.result(dict(budget=close_budget(rows,config['coverage_threshold'],config['completeness_threshold']),metadata=metadata,
        exploratory_affine=True,graph_protocol='direct-live-zero-target-v3',
        donor_provenance=provenance,adapter=args.adapter,
        deviations=['Direct source-to-target attribution graph with independent clean feature sources',
                    'Bounded upstream feature expansion uses clean TopK encoder support; unexpanded nodes remain listed',
                    'Task coverage appends a causally later excluded EOS sentinel, retains all original positions, and uses source skip0',
                    'WikiText fitting retains 128 tokens/skip16/final excluded; task coverage is an explicitly different evaluation distribution',
                    'Phrase targets use fitted phrase ILS rows when supplied; unsupported targets remain unknown',
                    'Per-item confidence intervals resample independent donor prompts'],
        interpretation='Graph completeness is absolute attribution mass, not predictive R2; both quantities remain separate.'))

if __name__=='__main__':main()
