import numpy as np
import torch
from scipy.stats import genpareto

def target_ranks(scores,target_ids):
    """Best target rank, with descending score then ascending token-ID order.

    Arbitrary ties must not make every token a top-one readout. Nonfinite
    targets are unscoreable (NaN), never assigned a favorable rank.
    """
    ids=torch.as_tensor(target_ids,device=scores.device,dtype=torch.long)
    if ids.ndim!=1 or ids.numel()==0:raise ValueError('At least one target ID is required')
    if ((ids<0)|(ids>=scores.shape[-1])).any():raise ValueError('Target ID outside vocabulary')
    values=scores[...,ids];values=values.masked_fill(~torch.isfinite(values),-torch.inf)
    best=values.max(-1).values
    chosen=torch.where(values==best[...,None],ids,scores.shape[-1]).amin(-1)
    vocabulary=torch.arange(scores.shape[-1],device=scores.device)
    higher=((scores>best[...,None])&torch.isfinite(scores)).sum(-1)
    tied_earlier=((scores==best[...,None])&(vocabulary<chosen[...,None])).sum(-1)
    ranks=(higher+tied_earlier+1).to(torch.float32)
    return torch.where(torch.isfinite(best),ranks,torch.full_like(ranks,torch.nan))

def best_target_rank(scores,target_ids):
    """Best valid rank over supplied read positions; NaN if none is scoreable."""
    ranks=target_ranks(scores,target_ids).reshape(-1)
    finite=ranks[torch.isfinite(ranks)]
    return float(finite.min()) if len(finite) else float('nan')

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

def random_dictionary_gains(target,nonzero_rows,repeats,generator):
    """Exact distribution of max positive projection gain for isotropic rows.

    Normalizing atoms cancels their norms in the optimal one-atom gain. For
    a random direction, squared projection / ||target||² is Beta(1/2,(d-1)/2),
    with half the mass at zero after the nonnegative constraint. Inverse-CDF
    sampling of the maximum avoids allocating 100 vocabulary-sized dictionaries.
    """
    from scipy.special import betaincinv
    if target.numel()<2 or nonzero_rows<1:raise ValueError('Need dimension >=2 and a nonzero dictionary')
    uniforms=torch.rand(repeats,generator=generator,device=target.device).double().clamp_min(1e-300)
    maximum_cdf=torch.exp(uniforms.log()/nonzero_rows)
    quantiles=(2*maximum_cdf-1).clamp_min(0).cpu().numpy()
    return target.square().sum()*target.new_tensor(betaincinv(.5,(target.numel()-1)/2,quantiles))

def sparse_pursuit(target,dictionary,seed=0,k=25,null_repeats=100):
    """Nonnegative gradient pursuit with at most 25 selected atoms.

    Each step gates the best new atom's gain against matched random dictionaries,
    then takes an exact line-search gradient step over the active support.
    """
    if null_repeats<100:raise ValueError('Use at least 100 random-dictionary gain samples')
    generator = torch.Generator(device=dictionary.device).manual_seed(seed)
    norms = dictionary.norm(dim=1)
    unit = dictionary/norms[:,None].clamp_min(1e-12)
    residual = target.clone()
    coefficients = torch.zeros(len(dictionary),device=dictionary.device,dtype=dictionary.dtype)
    support=torch.zeros(len(dictionary),device=dictionary.device,dtype=torch.bool)
    for _ in range(min(k,25)):
        correlations=unit@residual
        gains=correlations.clamp_min(0).square().masked_fill(support,0)
        gain,index=gains.max(0)
        null=random_dictionary_gains(target=residual,nonzero_rows=int((norms>0).sum()),repeats=null_repeats,generator=generator)
        if gain<=null.quantile(.95):
            break
        support[index]=True
        gradient=torch.where(support & ((coefficients>0)|(correlations>0)),correlations,0)
        change=gradient@unit
        step=(residual@change)/change.square().sum().clamp_min(1e-30)
        # Bound the step at the nonnegative orthant boundary; this preserves
        # the exact line-search decrease instead of clipping after overshoot.
        negative=gradient<0
        if negative.any():step=torch.minimum(step,(coefficients[negative]/(-gradient[negative])).min())
        coefficients=(coefficients+step*gradient).clamp_min(0)
        residual=target-coefficients@unit
    return coefficients/norms.clamp_min(1e-12),residual
