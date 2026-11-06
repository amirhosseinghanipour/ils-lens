"""Fixed expert-zero alignment diagnostic; no recall-based choice of rotation."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import torch
from ilslens.model import load_local
from ilslens.reference import load_reference
from ilslens.jpp import load_jpp
from ilslens.eval.diagnostics import procrustes_experts
from ilslens.eval.faithfulness import bootstrap_r2
from ilslens.eval.readout_tasks import candidate_ids,recall_report
from ilslens.readout import raw,best_target_rank
from ilslens.experiment import Experiment,save_tensor

def main():
    p=argparse.ArgumentParser();p.add_argument('--experts',required=True);p.add_argument('--reference',required=True);p.add_argument('--jpp-reference',required=True)
    p.add_argument('--states',default='runs/workspacebench/read-states.pt');p.add_argument('--run',default='runs/qwen-main')
    p.add_argument('--output',default='runs/procrustes');p.add_argument('--device',default='cuda');args=p.parse_args()
    model,tokenizer=load_local(args.device);load_reference(args.reference);load_jpp(args.jpp_reference)
    from workspace_lens.fitting.condense_experts.expert_jacobians import ExpertJacobians
    experts=ExpertJacobians.load(args.experts);root=Path(args.run);manifest=json.loads((root/'manifest.json').read_text());config=manifest['config']
    provenance_path=Path(args.experts).parent/'experts-provenance.json'
    if not provenance_path.exists():raise ValueError('Expert fitting provenance is required')
    provenance=json.loads(provenance_path.read_text())
    if provenance['primary_fingerprint']!=manifest['fingerprint']:raise ValueError('Expert fitting provenance differs')
    def digest(path):
        with open(path,'rb') as source:return hashlib.file_digest(source,'sha256').hexdigest()
    states=torch.load(args.states,weights_only=True,map_location='cpu')
    if states['metadata']['fingerprint']!=manifest['fingerprint']:raise ValueError('Readout provenance differs')
    run=Experiment(args.output,dict(config=dict(estimated_seconds=3600,estimated_disk_mb=200),primary_fingerprint=manifest['fingerprint'],
        expert_sha256=digest(args.experts),states_sha256=digest(args.states),polar_protocol='dispersion-v2',
        reference_expert=0,weighting='uniform',device=args.device,ranking='score-descending-id-ascending-v2'))
    rows=[];readouts=[]
    for layer,matrices in experts.experts_L_dict_EFN.items():
        result=procrustes_experts(matrices.to(model.device))
        save_tensor(run.path/f'layer-{layer:02d}.pt',{key:value.cpu() if isinstance(value,torch.Tensor) else value for key,value in result.items()})
        chunks=[torch.load(path,weights_only=True,map_location='cpu') for path in sorted((root/'samples/validation').glob(f'prompt-*-layer-{layer:02d}.pt'))]
        if len(chunks)!=config['heldout_prompts'] or any(c['metadata']['fingerprint']!=manifest['fingerprint'] for c in chunks):raise ValueError('Incomplete/mismatched held-out interventions')
        x=torch.cat([c['delta'] for c in chunks]).to(model.device);y=torch.cat([c['response'] for c in chunks]).to(model.device);groups=torch.cat([c['prompt_ids'] for c in chunks])
        for method in ('plain','aligned'):
            matrix=result[method]
            for e,epsilon in enumerate(config['epsilons']):
                for g,gamma in enumerate(config['gammas']):
                    rows.append(dict(layer=layer,method=method,epsilon=epsilon,gamma=gamma,effective_rank=result[method+'_rank'],polar_diagnostics=result['polar_diagnostics'],
                        **bootstrap_r2(y[:,e,g],x[:,e]@matrix.T,groups)))
            available=[(index,item) for index,item in enumerate(states['items']) if layer in states['states'][index]]
            ranks=np.full((len(available),1,max((len(item['targets']) for _,item in available),default=1)),np.nan)
            for i,(index,item) in enumerate(available):
                with torch.no_grad():scores=raw(states['states'][index][layer].to(model.device),matrix,model.lm_head.weight,model.model.norm.weight)
                for column,target in enumerate(item['targets']):
                    ids=candidate_ids(tokenizer,target)
                    if ids:ranks[i,0,column]=best_target_rank(scores,ids)
            readouts.extend(dict(row,source_layer=layer,method=method) for row in recall_report(ranks,[item['id'] for _,item in available]))
        run.event('procrustes-layer',layer=layer)
    run.result(dict(interventional=rows,readouts=readouts,label_tuned=False,
        interpretation='Polar rotations align exactly by construction. Neither alignment benefit nor residual scale disagreement identifies curvature; compare epsilon dependence separately and account for position-dependent horizons.',
        conventions='Expert maps have the upstream summed-target convention; held-out ILS targets are normalized, explicitly different estimands.'))

if __name__=='__main__':main()
