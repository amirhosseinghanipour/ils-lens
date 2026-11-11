"""Explicit upstream J++ integration; weighted outputs are always label-tuned."""
from pathlib import Path
import sys
import torch
from .hooks import capture_blocks

def require_fp32_experts(experts):
    for maps in (experts.experts_L_dict_EFN,experts.pooled_jacobian_L_dict_FN):
        for matrix in maps.values():
            if matrix.dtype!=torch.float32 or not torch.isfinite(matrix).all():
                raise ValueError('J++ requires finite fp32 expert checkpoints; rounded caches must be refitted')

def save_fp32_experts(experts,path):
    """Keep upstream checkpoint keys without its unconditional fp16 rounding."""
    from .experiment import save_tensor
    require_fp32_experts(experts)
    cpu=lambda values:{layer:value.detach().cpu() for layer,value in values.items()}
    save_tensor(path,dict(experts=cpu(experts.experts_L_dict_EFN),
        pooled=cpu(experts.pooled_jacobian_L_dict_FN),
        position_counts=cpu(experts.position_counts_L_dict_E),
        weight_sums=cpu(experts.weight_sums_L_dict_E),
        fallback=cpu(experts.fallback_L_dict_Bool_E),config=experts.config.to_dict(),
        min_kept_positions=experts.min_kept_positions,storage_protocol='fp32-experts-v1'))

def load_jpp(path):
    source=Path(path).resolve()/'src'
    if not (source/'workspace_lens').is_dir():raise FileNotFoundError(source)
    sys.path.append(str(source))
    import workspace_lens
    if not Path(workspace_lens.__file__).resolve().is_relative_to(source):
        raise RuntimeError('Unexpected workspace_lens import')

def fit_experts(model,adapter,sequences,output,seed=0,projection_dim=64,model_id='Qwen/Qwen2.5-0.5B-Instruct'):
    from workspace_lens.config import LensConfig
    from workspace_lens.routing.router import ActivationRouterCollection
    from workspace_lens.fitting.relp_fitting import ExpertJacobianRelPTrainer
    from workspace_lens.fitting.condense_experts.expert_jacobians import ExpertJacobians
    layers=list(range(len(model.model.layers)-1))
    activations={layer:[] for layer in layers}
    with torch.no_grad():
        for ids in sequences:
            with capture_blocks(model,layers) as states:model.model(ids[None].to(model.device),use_cache=False)
            for layer in layers:activations[layer].append(states[layer][0,16:-1].cpu().clone())
    router=ActivationRouterCollection(8,projection_dim,seed=seed)
    router.fit({layer:torch.cat(values) for layer,values in activations.items()})
    config=LensConfig(hf_model_name=model_id,checkpoint_name='jpp-experts',
        artifacts_base_dir=str(output),source_layers=layers,relative_end_transport_layer=-1,
        jacobian_rows_per_pass=1,max_seq_len=128,skip_first_n_positions=16,
        checkpoint_every_n_prompts=1,lrp_mode='rlens',num_clusters=8,cluster_projection_dim=projection_dim)
    trainer=ExpertJacobianRelPTrainer(config,adapter,[str(i) for i in range(len(sequences))],
        router_collections_K_dict={8:router},device=str(model.device))
    sums=trainer.fit()[8]
    experts=ExpertJacobians.from_sums(sums,trainer.config,min_kept_positions=50)
    return experts,router

def fit_labeled_weights(experts,adapter,dev_states,dev_targets=None,excluded_mask=None,candidate_lists=None,eval_slugs=None):
    """Dev tensors only. Caller must exclude these prompt IDs from all evaluation."""
    from workspace_lens.fitting.condense_experts.fitter import (
        CondenseConfig,fit_expert_weights,build_search_starting_points)
    from workspace_lens.fitting.condense_experts.rank_objective import (
        build_pair_table,expert_logits,expert_pair_pass_scores)
    candidates=candidate_lists if candidate_lists is not None else [{'target':[int(target)]} for target in dev_targets]
    device=next(iter(dev_states.values())).device
    pairs=build_pair_table(candidates,eval_slugs or ['dev']*len(candidates),list(range(len(candidates))),device=device)
    excluded_mask=None if excluded_mask is None else excluded_mask.to(device)
    weights={};metadata={}
    for layer,h in dev_states.items():
        transports=torch.einsum('efn,pn->pef',experts.experts_L_dict_EFN[layer].to(h),h)
        logits=expert_logits(adapter.unembed,transports)
        scores=expert_pair_pass_scores(logits,pairs,excluded_mask_Bool_V=excluded_mask,k=10)
        result=fit_expert_weights(logits,pairs,excluded_mask_Bool_V=excluded_mask,
            initial_weightings=build_search_starting_points(scores,top_n=3),condense_config=CondenseConfig())
        weights[layer]=result.weights_E
        metadata[layer]=dict(train_macro_pass=result.train_macro_pass,step=result.step,init=result.init_name)
    return experts.combine(weights,checkpoint_name='J++-label-tuned'),dict(label_tuned=True,weights=weights,dev_fit=metadata)

def readout_filter(tokenizer,vocab_size):
    return torch.tensor([not any(c.isalnum() for c in tokenizer.decode([i])) for i in range(vocab_size)])
