import numpy as np
from scipy.stats import spearmanr
from ..fit import r2

def token_spearman(predicted,actual,k=1000):
    ids = actual.abs().topk(min(k,len(actual))).indices
    return float(spearmanr(predicted[ids].detach().numpy(),actual[ids].detach().numpy()).statistic)

def bootstrap_r2(y,pred,prompt_ids,repeats=200,seed=0):
    import torch
    rng = np.random.default_rng(seed)
    groups = prompt_ids.unique().numpy()
    estimates=[]
    for _ in range(repeats):
        indices=torch.cat([torch.where(prompt_ids==int(g))[0] for g in rng.choice(groups,len(groups),replace=True)])
        estimates.append(r2(y[indices],pred[indices]))
    return dict(r2=r2(y,pred),ci=np.nanquantile(estimates,[.025,.975]).tolist())
