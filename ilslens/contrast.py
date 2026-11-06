import torch

def rank_of(scores,target_ids):
    best=scores[target_ids].max()
    return int((scores>best).sum())+1

@torch.no_grad()
def contrast_with_null(condition,control,read,target_ids,prompt_ids,seed=0,repeats=1000):
    if condition.shape!=control.shape or len(condition)!=len(prompt_ids):raise ValueError('Matched condition/control positions are required')
    groups=prompt_ids.unique()
    if len(groups)<4:raise ValueError('At least four control prompts are needed for split-half nulls')
    observed=contrast(condition,control,read)
    control_scores=read(control)
    generator=torch.Generator().manual_seed(seed)
    null={mode:[] for mode in observed}
    for _ in range(repeats):
        order=groups[torch.randperm(len(groups),generator=generator).to(groups.device)]
        mask=torch.isin(prompt_ids,order[:len(groups)//2])
        values={'mean_of_lens':control_scores[mask].mean(0)-control_scores[~mask].mean(0),
                'lens_of_mean':read(control[mask].mean(0))-read(control[~mask].mean(0))}
        for mode,value in values.items():null[mode].append(rank_of(value,target_ids))
    return {mode:dict(target_rank=rank_of(scores,target_ids),
        null_ranks=null[mode],rank_upper_significance_p=(1+sum(rank<=rank_of(scores,target_ids) for rank in null[mode]))/(repeats+1),
        target_score=float(scores[target_ids].max()),contrast_scores=scores.cpu()) for mode,scores in observed.items()}

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
