"""Collect one-sided, covariance Gaussian and matched-scale isotropic variants."""
import argparse,json
from pathlib import Path
import torch,yaml
from ilslens.model import load_local
from ilslens.data import wikitext_sequences,SAMPLING_POLICY
from ilslens.suffix import QwenSuffix
from ilslens.sampling import horizon_weights
from ilslens.covariance import residual_moments
from ilslens.experiment import Experiment,save_tensor
from ilslens.eval.diagnostics import ABLATION_DIFFERENCE_ORDER


def ablation_response(cache,layer,position,patch,gammas,kind):
    if kind not in ('no_antithetic','gaussian','isotropic'):raise ValueError('Unknown ablation')
    if kind!='no_antithetic':
        return cache.effects(layer,position,patch[None],gammas=gammas)[0]
    plus,_=cache.effects(layer,position,patch[None],return_position_branches=True)
    difference=plus[0]-cache.states[max(cache.states)][0,position:-1]
    return torch.stack([torch.einsum('t,td->d',horizon_weights(position,cache.ids.shape[1],g).to(difference),difference) for g in gammas])


def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);p.add_argument('--run',default='runs/qwen-main')
    p.add_argument('--config',default='configs/diagnostics.yaml');p.add_argument('--output',default='runs/ablations');p.add_argument('--device',default='cuda');args=p.parse_args()
    config=yaml.safe_load(Path(args.config).read_text());manifest=json.loads((Path(args.run)/'manifest.json').read_text());primary=manifest['config']
    run=Experiment(args.output,dict(config=config,sampling_policy=SAMPLING_POLICY,primary_fingerprint=manifest['fingerprint'],device=args.device,difference_order=ABLATION_DIFFERENCE_ORDER))
    model,tokenizer=load_local(args.device);layers=config['layers'];d=model.config.hidden_size
    sources,rows=wikitext_sequences(args.dataset,tokenizer,'train',primary['source_prompts'])
    if rows!=manifest['source_rows']:raise ValueError('Fitting corpus mismatch')
    heldout,_=wikitext_sequences(args.dataset,tokenizer,'validation',primary['heldout_prompts'])
    moments=residual_moments(model,sources,layers)
    factors={}
    for layer in layers:
        values,vectors=torch.linalg.eigh(moments[layer]['covariance']);factors[layer]=vectors*values.clamp_min(0).sqrt()[None]
    ei=primary['epsilons'].index(config['epsilon'])
    for split,sequences in [('train',sources),('validation',heldout)]:
        for prompt,ids in enumerate(sequences):
            cache=QwenSuffix(model,ids[None])
            for layer in layers:
                path=run.path/'samples'/split/f'prompt-{prompt:03d}-layer-{layer:02d}.pt'
                if path.exists():continue
                source=torch.load(Path(args.run)/'samples'/split/f'prompt-{prompt:03d}-layer-{layer:02d}.pt',weights_only=True,map_location='cpu')
                if source['metadata']['fingerprint']!=manifest['fingerprint']:raise ValueError('Sample provenance mismatch')
                generator=torch.Generator(device=model.device).manual_seed(config['seed']+prompt*100+layer+(100000 if split=='validation' else 0))
                records={kind:dict(delta=[],response=[]) for kind in ('no_antithetic','gaussian','isotropic')}
                for delta,t in zip(source['delta'][:,ei],source['source_positions']):
                    t=int(t);delta=delta.to(model.device)
                    noises={kind:torch.randn(d,generator=generator,device=model.device) for kind in ('gaussian','isotropic')}
                    patches={'no_antithetic':delta,'gaussian':config['epsilon']*(factors[layer]@noises['gaussian']),
                        'isotropic':config['epsilon']*(2*moments[layer]['covariance'].trace()/d).sqrt()*noises['isotropic']}
                    for kind,patch in patches.items():
                        response=ablation_response(cache,layer,t,patch,primary['gammas'],kind)
                        records[kind]['delta'].append(patch.cpu());records[kind]['response'].append(response.cpu())
                artifact={kind:{key:torch.stack(values) for key,values in data.items()} for kind,data in records.items()}
                save_tensor(path,dict(variants=artifact,prompt_ids=source['prompt_ids'],metadata=dict(fingerprint=run.fingerprint,
                    primary_fingerprint=manifest['fingerprint'],difference_order=ABLATION_DIFFERENCE_ORDER,epsilon=config['epsilon'],gammas=primary['gammas'],layer=layer,
                    isotropic_scale='epsilon sqrt(2 trace(Sigma)/d)',gaussian_scale='epsilon sqrt(Sigma)',source='block_output',target='pre_final_norm')))
                run.event('ablation-shard',split=split,prompt=prompt,layer=layer,N=len(source['prompt_ids']))
    run.result(dict(status='collection_complete',difference_order=ABLATION_DIFFERENCE_ORDER,variants=['no_antithetic','gaussian','isotropic'],
        note='Gaussian follows the specified covariance scale; isotropic matches the donor displacement mean squared norm in the independent-bank limit'))

if __name__=='__main__':main()
