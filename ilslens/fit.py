"""Row samples X/Y; returned matrices map column vectors (prediction X @ A.T)."""
import torch

def ridge(x, y, lam=0., prior=None, weights=None):
    if lam < 0:
        raise ValueError('negative ridge penalty')
    d = x.shape[1]
    prior = torch.eye(y.shape[1], d, dtype=x.dtype, device=x.device) if prior is None else prior
    w = torch.ones(len(x), dtype=x.dtype, device=x.device) if weights is None else weights
    cxx = (x.T*w)@x / len(x)
    cyx = (y.T*w)@x / len(x)
    if lam == 0:
        return cyx @ torch.linalg.pinv(cxx, hermitian=True)
    return torch.linalg.solve(cxx+lam*torch.eye(d,device=x.device,dtype=x.dtype),
                              (cyx+lam*prior).T).T

def geometric_median(values, tolerance=1e-6, max_iter=1000):
    flat = values.flatten(1)
    center = flat.mean(0)
    for _ in range(max_iter):
        distances = (flat-center).norm(dim=1)
        # Modified Weiszfeld handles coincident iterates without false convergence.
        nonzero = distances > 1e-12
        if not nonzero.any():
            break
        weights = distances[nonzero].reciprocal()
        candidate = (flat[nonzero]*weights[:,None]).sum(0)/weights.sum()
        coincident = (~nonzero).sum()
        residual = ((flat[nonzero]-center)*weights[:,None]).sum(0).norm()
        alpha = (coincident/residual.clamp_min(1e-12)).clamp(max=1)
        candidate = alpha*center+(1-alpha)*candidate
        if (candidate-center).norm() <= tolerance*(1+center.norm()):
            center = candidate
            break
        center = candidate
    return center.reshape(values.shape[1:])

def fit(x,y,lam=0.,prior=None,estimator='ridge',prompt_ids=None):
    a = ridge(x,y,lam,prior)
    if estimator == 'ridge':
        return a
    if estimator == 'huber':
        for _ in range(10):
            residual = (y-x@a.T).norm(dim=1)
            c = 1.345*residual.median()/0.6745
            if c <= 1e-12:
                return a
            weights = (c/residual.clamp_min(1e-12)).clamp(max=1)
            new = ridge(x,y,lam,prior,weights)
            if torch.allclose(new,a,atol=1e-6,rtol=1e-5):
                return new
            a = new
        return a
    if estimator == 'mom':
        if prompt_ids is None or len(prompt_ids.unique()) < 10:
            raise ValueError('MoM requires at least ten distinct prompts')
        blocks = torch.tensor_split(prompt_ids.unique(),10)
        matrices = []
        for block in blocks:
            mask = torch.isin(prompt_ids,block)
            matrices.append(ridge(x[mask],y[mask],lam,prior))
        return geometric_median(torch.stack(matrices))
    raise ValueError(estimator)

def r2(y, pred):
    denominator = (y-y.mean(0)).square().sum()
    if denominator <= 0:
        return float('nan')
    return (1-(y-pred).square().sum()/denominator).item()

def select(x,y,prompt_ids,priors,lambdas,estimators=('ridge','huber','mom'),seed=0,prior_samples=None,fold_prior_provider=None):
    if not torch.isfinite(x).all() or not torch.isfinite(y).all():raise ValueError('Nonfinite fitting observations')
    if (y-y.mean(0)).square().sum()<=0:raise ValueError('Interventional R2 selection is undefined for a constant target')
    groups = prompt_ids.unique()
    if len(groups)<5:
        raise ValueError('five-fold CV requires five prompts')
    g = torch.Generator().manual_seed(seed)
    folds = torch.tensor_split(groups[torch.randperm(len(groups),generator=g)],5)
    results = [];zero_penalty_scores={}
    for name, prior in priors.items():
        if name not in ('I','J','R'):
            raise ValueError('main-result priors must be label-free I, J or R')
        for lam in lambdas:
            for estimator in estimators:
                if lam==0 and estimator in zero_penalty_scores:
                    results.append(dict(prior=name,lam=lam,estimator=estimator,r2=zero_penalty_scores[estimator]));continue
                predictions = torch.empty_like(y)
                for group in folds:
                    held = torch.isin(prompt_ids,group)
                    fold_prior = prior_samples[name][~held].mean(0) if prior_samples is not None and name in prior_samples else prior
                    if fold_prior_provider is not None:
                        fold_prior = fold_prior_provider(name,prompt_ids[~held].unique())
                    a = fit(x[~held],y[~held],lam,fold_prior,estimator,prompt_ids[~held])
                    predictions[held] = x[held]@a.T
                if not torch.isfinite(predictions).all():raise FloatingPointError('Nonfinite cross-validated predictions')
                score=r2(y,predictions)
                if lam==0:zero_penalty_scores[estimator]=score
                results.append(dict(prior=name,lam=lam,estimator=estimator,r2=score))
    winner = max(results,key=lambda result: result['r2'])
    return fit(x,y,winner['lam'],priors[winner['prior']],winner['estimator'],prompt_ids),winner,results
