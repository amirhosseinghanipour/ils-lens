"""Use upstream expert fitting and labeled-dev weighting on the exact local model."""
import argparse,hashlib,json,time
from pathlib import Path
import torch
from ilslens.model import load_local
from ilslens.data import wikitext_sequences
from ilslens.budget import baseline_count
from ilslens.reference import load_reference,token_adapter
from ilslens.jpp import (load_jpp,fit_experts,fit_labeled_weights,readout_filter,
    save_fp32_experts,require_fp32_experts)
from ilslens.hooks import capture_blocks
from ilslens.eval.readout_tasks import candidate_ids,normalize

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--dataset',required=True);parser.add_argument('--reference',required=True)
    parser.add_argument('--jpp',required=True);parser.add_argument('--gates',required=True)
    parser.add_argument('--run',default='runs/qwen-main');parser.add_argument('--output',default='runs/jpp')
    parser.add_argument('--device',default='cuda');args=parser.parse_args()
    root=Path(args.output);root.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((Path(args.run)/'manifest.json').read_text());config=manifest['config'];start=time.perf_counter()
    (root/'estimate.json').write_text(json.dumps(dict(estimated_seconds=7200*baseline_count(config)/16,estimated_disk_mb=1500,
        source_prompts=baseline_count(config),clusters=8,pca=64,label_tuned=True),indent=2))
    model,tokenizer=load_local(args.device);reference=load_reference(args.reference);load_jpp(args.jpp)
    ids,rows=wikitext_sequences(args.dataset,tokenizer,'train',baseline_count(config))
    expected_rows=manifest['source_rows']+(manifest['donor_rows'] if baseline_count(config)>config['source_prompts'] else [])
    if rows!=expected_rows:raise ValueError('J++ fitting corpus differs from J/R/ILS')
    adapter=token_adapter(reference,model,tokenizer,ids)
    expert_path=root/'experts.pt'
    expert_provenance_path=root/'experts-provenance.json'
    expert_provenance=dict(primary_fingerprint=manifest['fingerprint'],fitting_rows=rows,
                          clusters=8,pca=64,seed=config['seed'],storage_protocol='fp32-experts-v1')
    if expert_path.exists():
        if not expert_provenance_path.exists() or json.loads(expert_provenance_path.read_text())!=expert_provenance:
            raise ValueError('J++ expert fitting corpus/configuration is unverified or changed; use a new output directory')
        from workspace_lens.fitting.condense_experts.expert_jacobians import ExpertJacobians
        experts=ExpertJacobians.load(str(expert_path))
        require_fp32_experts(experts)
    else:
        experts,_=fit_experts(model,adapter,ids,root,seed=config['seed'],projection_dim=64)
        save_fp32_experts(experts,expert_path)
        expert_provenance_path.write_text(json.dumps(expert_provenance,indent=2)+'\n')
    gates=json.loads(Path(args.gates).read_text())
    selected=[r['item'] for r in gates if r['retained'] and r['item']['split']=='dev']
    state_lists={layer:[] for layer in experts.layers};candidate_lists=[];tasks=[];dev_ids=[];dev_prompt_hashes=[]
    for item in selected:
        candidates={concept:candidate_ids(tokenizer,concept) for concept in item['intermediates']}
        candidates={concept:ids for concept,ids in candidates.items() if ids}
        if not candidates:continue
        tokens=tokenizer(item['prompt'],return_tensors='pt').input_ids.to(model.device)
        with torch.no_grad(),capture_blocks(model,experts.layers) as states:model.model(tokens,use_cache=False)
        for layer in experts.layers:state_lists[layer].append(states[layer][0,-1].clone())
        candidate_lists.append(candidates);tasks.append(item['task']);dev_ids.append(item['id'])
        dev_prompt_hashes.append(hashlib.sha256(normalize(item['prompt']).encode()).hexdigest())
    if not candidate_lists:raise ValueError('No capability-filtered labeled dev items; J++ cannot be fitted')
    lens,weight_metadata=fit_labeled_weights(experts,adapter,{layer:torch.stack(values) for layer,values in state_lists.items()},
        candidate_lists=candidate_lists,eval_slugs=tasks,excluded_mask=readout_filter(tokenizer,model.config.vocab_size))
    metadata=dict(label_tuned=True,dev_item_ids=dev_ids,dev_prompt_hashes=dev_prompt_hashes,primary_fingerprint=manifest['fingerprint'],
        source_prompts=baseline_count(config),clusters=8,pca=64,readout_filter='tokens_without_letter_or_digit',
        target_layer=23,target_reduction='upstream_sum',seconds=time.perf_counter()-start)
    matrices=dict(lens.jacobians_L_dict_FN)
    matrices[len(model.model.layers)-1]=torch.eye(model.config.hidden_size)
    metadata['final_layer']='Analytic identity under the upstream summed-target convention; no final-layer expert fitting'
    torch.save(dict(matrices=matrices,metadata=metadata,weights=weight_metadata),root/'J++-label-tuned.pt')
    (root/'metadata.json').write_text(json.dumps(metadata,indent=2))

if __name__=='__main__':main()
