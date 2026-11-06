"""Exact Bayesian hidden-state filter for the registered toy generator."""
import torch

def posterior_states(ids):
    if ids.ndim!=2 or (ids[:,0]!=13).any():raise ValueError('Expected BOS-prefixed toy HMM sequences')
    n,t=ids.shape;posterior=torch.full((n,4),.25,dtype=torch.float32,device=ids.device)
    result=[posterior.clone()]
    for position in range(1,t):
        token=ids[:,position];previous=ids[:,position-1]
        # A STATE/id pair occupies two tokens but a single HMM timestep.
        step=previous!=8
        posterior=torch.where(step[:,None],.92*posterior+.02,posterior)
        observations=token<8
        likelihood=torch.full_like(posterior,.15/8)
        likelihood[torch.arange(n,device=ids.device),(token//2).clamp(0,3)]+=.85/2
        updated=posterior*likelihood;updated/=updated.sum(1,keepdim=True)
        posterior=torch.where(observations[:,None],updated,posterior)
        reports=(9<=token)&(token<=12)
        if reports.any():
            exact=torch.nn.functional.one_hot((token-9).clamp(0,3),4).float()
            posterior=torch.where(reports[:,None],exact,posterior)
        result.append(posterior.clone())
    return torch.stack(result,dim=1)
