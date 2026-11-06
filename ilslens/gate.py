import torch

def random_direction_gate(direction,covariance,effect,seed=0,count=100):
    if count<100:
        raise ValueError('causal gate requires at least 100 controls')
    generator = torch.Generator(device=direction.device).manual_seed(seed)
    eigenvalues,eigenvectors = torch.linalg.eigh(covariance)
    if covariance.device!=direction.device:raise ValueError('Direction and covariance must share a device')
    samples = torch.randn(count,len(direction),generator=generator,device=direction.device,dtype=direction.dtype)
    random = (samples*eigenvalues.clamp_min(0).sqrt())@eigenvectors.T
    if (random.norm(dim=1)==0).any():
        raise ValueError('degenerate covariance')
    random *= direction.norm()/random.norm(dim=1,keepdim=True)
    effects = torch.tensor([effect(v) for v in random])
    observed = float(effect(direction))
    threshold = effects.quantile(.95).item()
    return dict(effect=observed,threshold=threshold,passes=observed>threshold,
                random_effects=effects.tolist(),seed=seed)
