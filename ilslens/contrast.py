import torch
from .readout import target_ranks

def rank_of(scores,target_ids):
    rank=target_ranks(scores,target_ids)
    if not torch.isfinite(rank):raise ValueError('Contrast target has no finite readout score')
    return int(rank)

@torch.no_grad()
def contrast_with_null(condition,control,read,target_ids,prompt_ids,seed=0,repeats=1000,batch_size=32):
    if condition.shape!=control.shape or len(condition)!=len(prompt_ids):raise ValueError('Matched condition/control positions are required')
    if repeats<1 or batch_size<1:raise ValueError('Positive null count and batch size required')
    groups=prompt_ids.unique()
    if len(groups)<4:raise ValueError('At least four control prompts are needed for split-half nulls')
    observed=contrast(condition,control,read);control_scores=read(control)
    masks=[prompt_ids==group for group in groups]
    counts=control.new_tensor([int(mask.sum()) for mask in masks])
    states=torch.stack([control[mask].sum(0) for mask in masks])
    scores=torch.stack([control_scores[mask].sum(0) for mask in masks])
    generator=torch.Generator().manual_seed(seed);null={mode:[] for mode in observed}
    for start in range(0,repeats,batch_size):
        selections=[]
        for _ in range(min(batch_size,repeats-start)):
            order=torch.randperm(len(groups),generator=generator)
            selected=control.new_zeros(len(groups));selected[order[:len(groups)//2].to(control.device)]=1
            selections.append(selected)
        first=torch.stack(selections);second=1-first
        first=first/(first@counts)[:,None];second=second/(second@counts)[:,None]
        values={'mean_of_lens':(first-second)@scores,
                'lens_of_mean':read(first@states)-read(second@states)}
        for mode,value in values.items():
            ranks=target_ranks(value,target_ids)
            if not torch.isfinite(ranks).all():raise ValueError('Unscoreable null contrast')
            null[mode].extend(ranks.int().cpu().tolist())
    result={}
    for mode,scores in observed.items():
        rank=rank_of(scores,target_ids);flat=bool((scores==scores.flatten()[0]).all())
        result[mode]=dict(target_rank=rank,null_ranks=null[mode],
            rank_upper_significance_p=None if flat else (1+sum(r<=rank for r in null[mode]))/(repeats+1),
            rank_test_status='undefined_flat_contrast' if flat else 'measured',
            target_score=float(scores[target_ids].max()),contrast_scores=scores.cpu(),
            null_unit='prompt',null_repeats=repeats,null_protocol='batched-prompt-sums-flat-undefined-v2')
    return result

def contrast(condition,control,read):
    return {'lens_of_mean':read(condition.mean(0))-read(control.mean(0)),
            'mean_of_lens':read(condition).mean(0)-read(control).mean(0)}

def null_rank(control,read,target,observed,prompt_ids,seed=0,repeats=1000):
    groups = prompt_ids.unique()
    if len(groups)<2:
        raise ValueError('need two control prompts')
    generator = torch.Generator().manual_seed(seed)
    null = []
    for _ in range(repeats):
        shuffled = groups[torch.randperm(len(groups),generator=generator)]
        mask = torch.isin(prompt_ids,shuffled[:len(groups)//2])
        null.append(contrast(control[mask],control[~mask],read)['mean_of_lens'][target])
    null = torch.stack(null)
    return dict(percentile=(null<observed).float().mean().item(),
                upper_p=((null>=observed).sum().item()+1)/(repeats+1))
