import torch

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
