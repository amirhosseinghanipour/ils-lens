import torch

@torch.no_grad()
def metrics(model,x,y):
    denominator = (y-y.mean(0)).square().sum()
    return dict(variance_explained=(1-(model(x)-y).square().sum()/denominator).item(),
                dead_fraction=(model.features(x).sum(0)==0).float().mean().item())

def ce_recovered(clean,replaced,ablated):
    if ablated == clean:
        return float('nan')
    return (ablated-replaced)/(ablated-clean)

@torch.no_grad()
def output_statistics(prediction,target):
    """Centered float64 reporting moments; model computation remains fp32."""
    if prediction.shape!=target.shape or not target.numel():raise ValueError('Nonempty matched MLP outputs required')
    if not torch.isfinite(prediction).all() or not torch.isfinite(target).all():raise FloatingPointError('Nonfinite MLP outputs')
    y=target.detach().reshape(-1,target.shape[-1]).double()
    mean=y.mean(0)
    return dict(n=len(y),mean=mean.cpu(),m2=float((y-mean).square().sum()),
                sse=float((prediction.detach().double()-target.detach().double()).square().sum()))


def explained_from_statistics(rows):
    """Merge centered moments, including duplicated windows in bootstrap samples."""
    if not rows:raise ValueError('No evaluation windows')
    count=sum(r['n'] for r in rows)
    mean=sum(r['mean']*r['n'] for r in rows)/count
    denominator=sum(r['m2']+float((r['mean']-mean).square().sum())*r['n'] for r in rows)
    return 1-sum(r['sse'] for r in rows)/denominator if denominator>0 else float('nan')


def uncertainty(value,replicates,undefined_reason):
    """Serialize undefined ratios without fabricating values or confidence bounds."""
    import numpy as np
    values=np.asarray(replicates,dtype=float);valid=values[np.isfinite(values)]
    measured=bool(np.isfinite(value))
    return dict(value=float(value) if measured else None,
        ci=np.quantile(valid,[.025,.975]).tolist() if measured and len(valid)>=20 else [None,None],
        status='measured' if measured else undefined_reason,
        valid_bootstrap_resamples=len(valid),undefined_bootstrap_resamples=len(values)-len(valid))
