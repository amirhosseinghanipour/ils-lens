import torch

def valid_positions(length, skip=16):
    if length <= skip+1:
        raise ValueError('sequence has no valid source positions')
    return torch.arange(skip, length-1)

def horizon_weights(source, length, gamma, skip=16):
    if not 0 <= gamma <= 1:
        raise ValueError('gamma must lie in [0, 1]')
    if source not in valid_positions(length, skip):
        raise ValueError('invalid source')
    offsets = torch.arange(length-1-source)
    weights = torch.pow(torch.tensor(gamma, dtype=torch.float32), offsets)
    return weights/weights.sum()

def donor(bank, prompt_ids, source_prompt, generator):
    eligible = torch.where(prompt_ids != source_prompt)[0]
    if not len(eligible):
        raise ValueError('donor bank must contain another prompt')
    # Equal-length sequences with equal numbers of valid positions: uniform
    # bank-row sampling is uniform over both sequence and position.
    chosen = eligible[torch.randint(len(eligible), (), generator=generator)]
    return bank[chosen]

def perturbation(h, epsilon, kind, generator, donor_h=None, covariance=None):
    if kind == 'donor':
        if donor_h is None:
            raise ValueError('donor required')
        return epsilon*(donor_h-h)
    g = torch.randn(h.shape, generator=generator, dtype=h.dtype, device=h.device)
    if kind == 'isotropic':
        return epsilon*g
    if kind == 'gaussian':
        eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
        return epsilon*(eigenvectors @ (eigenvalues.clamp_min(0).sqrt()*g))
    raise ValueError(kind)
