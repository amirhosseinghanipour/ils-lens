"""Full-matrix isotropic-limit test against our normalized-horizon J, all blocks."""
import argparse
from pathlib import Path
import torch,yaml
from ilslens.model import load_local
from ilslens.data import wikitext_sequences
from ilslens.suffix import QwenSuffix
from ilslens.hooks import capture_blocks
from ilslens.fit import ridge
from ilslens.readout import raw
from ilslens.eval.numerical import jacobian_rows,relative_jvp_error
from ilslens.experiment import Experiment,save_tensor,write_json

def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);p.add_argument('--config',default='configs/numerical.yaml')
    p.add_argument('--output',default='runs/numerical');p.add_argument('--device',default='cuda');args=p.parse_args()
    config=yaml.safe_load(Path(args.config).read_text());run=Experiment(args.output,dict(config=config,device=args.device))
    model,tokenizer=load_local(args.device);d=model.config.hidden_size
    sources,source_rows=wikitext_sequences(args.dataset,tokenizer,'train',config['contexts'],offset=32)
    heldout,heldout_rows=wikitext_sequences(args.dataset,tokenizer,'validation',config['heldout_prompts'],offset=192)
    held_states={layer:[] for layer in config['layers']}
    for ids in heldout:
        with torch.no_grad(),capture_blocks(model,config['layers']) as states:model.model(ids[None].to(model.device),use_cache=False)
        for layer in held_states:held_states[layer].append(states[layer][0,64].cpu())
    results=[]
    for layer in config['layers']:
        jacobians=[];inputs={e:[] for e in config['epsilons']};outputs={e:[] for e in inputs};errors={e:[] for e in inputs}
        for index,ids in enumerate(sources):
            path=run.path/f'layer-{layer:02d}-context-{index:02d}.pt'
            if path.exists():saved=torch.load(path,weights_only=True,map_location='cpu')
            else:
                cache=QwenSuffix(model,ids[None]);generator=torch.Generator(device=model.device).manual_seed(config['seed']+index)
                position=int(torch.randint(16,127,(),generator=generator,device=model.device));h=cache.states[layer]
                def function(delta):
                    state=h.clone();state[0,position]+=delta
                    return cache.forward(layer,state)[0,position:-1].mean(0)
                point=torch.zeros(d,device=model.device)
                j=jacobian_rows(function,point,config['row_batch'])
                basis=torch.linalg.qr(torch.randn(d,d,generator=generator,device=model.device)).Q*d**.5
                xs={};ys={};es={}
                for epsilon in config['epsilons']:
                    delta=basis*epsilon;effect=[]
                    for batch in delta.split(config['intervention_batch']):effect.append(cache.effects(layer,position,batch,gammas=(1.,))[:,0])
                    xs[epsilon]=delta.cpu();ys[epsilon]=torch.cat(effect).cpu()
                    es[epsilon]=relative_jvp_error(function,point,delta[0])
                saved=dict(j=j.cpu(),inputs=xs,outputs=ys,jvp_errors=es,position=position,fingerprint=run.fingerprint)
                save_tensor(path,saved);run.event('limit-context',layer=layer,context=index)
            if saved['fingerprint']!=run.fingerprint:raise ValueError('Limit-test provenance mismatch')
            jacobians.append(saved['j'])
            for epsilon in inputs:
                inputs[epsilon].append(saved['inputs'][epsilon]);outputs[epsilon].append(saved['outputs'][epsilon]);errors[epsilon].append(saved['jvp_errors'][epsilon])
        j=torch.stack(jacobians).mean(0).to(model.device);h=torch.stack(held_states[layer]).to(model.device)
        for epsilon in inputs:
            a=ridge(torch.cat(inputs[epsilon]).to(model.device),torch.cat(outputs[epsilon]).to(model.device),lam=0)
            cosine=float(torch.nn.functional.cosine_similarity(a.flatten(),j.flatten(),dim=0))
            with torch.no_grad():
                first=raw(h,a,model.lm_head.weight,model.model.norm.weight).topk(10).indices
                second=raw(h,j,model.lm_head.weight,model.model.norm.weight).topk(10).indices
            overlaps=[len(set(a.tolist())&set(b.tolist()))/10 for a,b in zip(first,second)]
            overlap=sum(overlaps)/len(overlaps)
            results.append(dict(layer=layer,epsilon=epsilon,matrix_cosine=cosine,heldout_top10_overlap=overlap,
                passes=cosine>=.95 and overlap>=.8,jvp_errors=errors[epsilon]))
            save_tensor(run.path/f'matrices-{layer}-{epsilon}.pt',dict(ILS=a.cpu(),J_normalized=j.cpu(),metadata=dict(fingerprint=run.fingerprint,epsilon=epsilon,
                gamma=1.,lambda_=0,N=len(sources)*d,source_layer=layer,target_layer=len(model.model.layers)-1,source='block_output',target='pre_final_norm')))
    smallest=min(config['epsilons']);passed=all(row['passes'] for row in results if row['epsilon']==smallest)
    run.result(dict(rows=results,smallest_epsilon_passes_all_layers=passed,source_rows=source_rows,heldout_rows=heldout_rows,
        deviations=['Per-context randomly rotated orthogonal bases provide exactly balanced covariance; marginally isotropic, not iid Gaussian samples',
        'Compares to our normalized-horizon J, since the upstream J sums future targets and estimates a different quantity']))
    if not passed:raise AssertionError('Registered smallest-epsilon equivalence criterion failed; measured results retained')

if __name__=='__main__':main()
