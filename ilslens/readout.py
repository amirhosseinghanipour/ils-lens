import numpy as np
import torch
from scipy.stats import genpareto

def raw(h,a,unembedding,gain,eps=1e-6):
    state = h@a.T
    return (state*gain/(state.square().mean(-1,keepdim=True)+eps).sqrt())@unembedding.T

def dead_rows(unembedding,frequencies,tolerance=1e-8):
    return (unembedding.norm(dim=1)<tolerance)|(frequencies==0)

def calibrate_token(reference,minimum=100000):
    """Fit one token at a time to avoid materializing 100k x vocabulary scores."""
    reference = np.asarray(reference)
    if len(reference)<minimum:
        raise ValueError('insufficient calibration activations')
    threshold = np.quantile(reference,.95)
    excess = reference[reference>threshold]-threshold
    if reference.std()==0 or len(excess)<100:
        return dict(mean=float(reference.mean()),std=0.,threshold=float(threshold),tail=None)
    shape,_,scale = genpareto.fit(excess,floc=0)
    return dict(mean=float(reference.mean()),std=float(reference.std()),
                threshold=float(threshold),tail=(float(shape),float(scale)),
                exceedance=float(len(excess)/len(reference)))

def zscore(scores,means,stds):
    return torch.where(stds>0,(scores-means)/stds.clamp_min(1e-12),-torch.inf)

def fdr(scores,z,calibration,q=.05):
    """BY over FULL vocabulary. Below-tail candidates conservatively receive p=1."""
    v = len(scores)
    p = np.ones(v)
    indices = np.argsort(z)[-min(200,v):]
    for i in indices:
        c = calibration[i]
        if c['tail'] is not None and scores[i]>c['threshold']:
            shape,scale = c['tail']
            p[i] = c['exceedance']*genpareto.sf(scores[i]-c['threshold'],shape,scale=scale)
    order = np.argsort(p)
    harmonic = np.sum(1/np.arange(1,v+1))
    passed = p[order]<=q*np.arange(1,v+1)/(v*harmonic)
    accepted = order[:np.flatnonzero(passed)[-1]+1] if passed.any() else np.array([],dtype=int)
    return accepted,p

def sparse_pursuit(target,dictionary,seed=0,k=25,null_repeats=100):
    """Nonnegative matching pursuit, null calibrated on max marginal gain.

    Random dictionaries have identical row norms and size; redraw per step.
    Coefficients are greedily updated, not a full nonnegative least-squares refit.
    """
    generator = torch.Generator(device=dictionary.device).manual_seed(seed)
    norms = dictionary.norm(dim=1)
    unit = dictionary/norms[:,None].clamp_min(1e-12)
    residual = target.clone()
    coefficients = torch.zeros(len(dictionary),device=dictionary.device)
    for _ in range(min(k,25)):
        correlations = (unit@residual).clamp_min(0)
        gain,index = correlations.square().max(0)
        null = []
        for _ in range(null_repeats):
            random = torch.randn(dictionary.shape,generator=generator,device=dictionary.device)
            random /= random.norm(dim=1,keepdim=True).clamp_min(1e-12)
            null.append((random@residual).clamp_min(0).square().max())
        if gain<=torch.stack(null).quantile(.95):
            break
        coefficients[index] += correlations[index]/norms[index].clamp_min(1e-12)
        residual -= correlations[index]*unit[index]
    return coefficients,residual
