"""Label-free residual mean/covariance from the registered fitting prompts."""
import torch
from .hooks import capture_blocks

@torch.no_grad()
def residual_moments(model,sequences,layers,skip=16):
    d=model.config.hidden_size;device=model.device
    sums={layer:torch.zeros(d,device=device) for layer in layers}
    products={layer:torch.zeros(d,d,device=device) for layer in layers};count=0
    for ids in sequences:
        with capture_blocks(model,layers) as states:model.model(ids[None].to(device),use_cache=False)
        for layer in layers:
            h=states[layer][0,skip:-1];sums[layer]+=h.sum(0);products[layer]+=h.T@h
        count+=len(h)
    if count<2:raise ValueError('Insufficient valid residual samples')
    return {layer:dict(mean=sums[layer]/count,covariance=(products[layer]-sums[layer][:,None]*sums[layer][None,:]/count)/(count-1),
        count=count) for layer in layers}
