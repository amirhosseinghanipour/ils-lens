import numpy as np
from scipy.stats import spearmanr
from ..fit import r2

def token_spearman(predicted,actual,k=1000):
    ids = actual.abs().topk(min(k,len(actual))).indices
    if predicted[ids].max()==predicted[ids].min() or actual[ids].max()==actual[ids].min():return float('nan')
    return float(spearmanr(predicted[ids].detach().cpu().numpy(),actual[ids].detach().cpu().numpy()).statistic)

def bootstrap_r2(y,pred,prompt_ids,repeats=200,seed=0):
    import torch
    if not len(y) or y.shape!=pred.shape or len(y)!=len(prompt_ids):raise ValueError('Nonempty matched observations and prompt IDs required')
    if not torch.isfinite(y).all() or not torch.isfinite(pred).all():raise ValueError('Nonfinite model response or prediction')
    rng = np.random.default_rng(seed)
    groups = prompt_ids.unique().cpu().numpy()
    estimates=[]
    for _ in range(repeats):
        indices=torch.cat([torch.where(prompt_ids==int(g))[0] for g in rng.choice(groups,len(groups),replace=True)])
        estimates.append(r2(y[indices],pred[indices]))
    value=r2(y,pred);finite=np.asarray(estimates)[np.isfinite(estimates)]
    return dict(r2=value if np.isfinite(value) else None,
        ci=np.quantile(finite,[.025,.975]).tolist() if len(finite)>=20 else [None,None],
        undefined_bootstrap_resamples=repeats-len(finite),
        r2_status='measured' if np.isfinite(value) else 'undefined_zero_target_variance')

def bootstrap_prompt_mean(values,repeats=200,seed=0):
    """Aggregate repeated measurements within prompts; report undefined cases."""
    values=np.asarray(values,dtype=float)
    if values.ndim!=2:raise ValueError('Expected prompt by repeated-measurement array')
    finite=np.isfinite(values);counts=finite.sum(1)
    means=np.where(finite,values,0).sum(1)/np.maximum(counts,1);means=means[counts>0]
    rng=np.random.default_rng(seed)
    ci=[None,None] if not len(means) else np.quantile(means[rng.integers(len(means),size=(repeats,len(means)))].mean(1),[.025,.975]).tolist()
    return dict(mean=float(means.mean()) if len(means) else None,ci=ci,valid_prompts=len(means),
        undefined_prompts=int((counts==0).sum()),undefined_measurements=int((~finite).sum()))
