import torch
from .fit import ridge
from .hooks import patch_site

def phrase_ids(tokenizer,phrases):
    result={}
    for phrase in phrases:
        tokens=tokenizer.encode(phrase,add_special_tokens=False)
        if not 2<=len(tokens)<=5:raise ValueError(f'Phrase must have 2–5 tokens: {phrase!r}')
        result[phrase]=tokens
    return result

def continuation_log_probability(model,prefix_ids,continuation_ids):
    device=model.model.embed_tokens.weight.device
    prefix=torch.as_tensor(prefix_ids,dtype=torch.long,device=device).reshape(1,-1)
    continuation=torch.as_tensor(continuation_ids,dtype=torch.long,device=device).reshape(1,-1)
    if prefix.shape[1]==0 or continuation.shape[1]==0:raise ValueError('Prefix and continuation must be nonempty')
    ids=torch.cat([prefix,continuation],dim=1)
    logits=model(ids,use_cache=False).logits[0,prefix.shape[1]-1:-1].log_softmax(-1)
    return logits.gather(1,continuation[0,:,None]).sum()

@torch.no_grad()
def antithetic_phrase_effect(model,prompt_ids,suffix_ids,phrase_tokens,layer,position,delta):
    prefix=list(prompt_ids)+list(suffix_ids)
    if not 0<=position<len(prompt_ids):raise ValueError('Intervention must lie in the original prompt')
    outputs=[]
    for sign in (1,-1):
        with patch_site(model,layer,position,sign*delta):
            outputs.append(continuation_log_probability(model,prefix,phrase_tokens))
    return (outputs[0]-outputs[1])/2

def fit_phrase_rows(delta,log_probability_change,prompt_ids,lambdas,seed=0):
    """Shared ridge factorizations; independent prompt-CV penalty per phrase.

    Right-hand sides are batched without pooling the selection objective across
    phrases. This is the same zero-prior ridge estimator as separate row fits.
    """
    if not lambdas or any(value<0 or not torch.isfinite(torch.tensor(value)) for value in lambdas):
        raise ValueError('Nonnegative finite ridge candidates are required')
    if delta.ndim!=2 or log_probability_change.ndim!=2 or len(delta)!=len(log_probability_change) or len(delta)!=len(prompt_ids):
        raise ValueError('Expected matching sample matrices and prompt IDs')
    if not torch.isfinite(delta).all() or not torch.isfinite(log_probability_change).all():
        raise ValueError('Nonfinite phrase fitting data')
    y=log_probability_change;columns=y.shape[1]
    if not columns:raise ValueError('No phrase targets')
    zero=(y==0).all(0)
    denominator=(y-y.mean(0)).square().sum(0)
    if ((denominator<=0)&~zero).any():
        raise ValueError('Interventional R2 selection is undefined for a nonzero constant phrase target')
    matrices=delta.new_zeros(columns,delta.shape[1]);winners=[None]*columns;all_results=[[] for _ in range(columns)]
    for column in torch.where(zero)[0].tolist():
        # All zero-prior penalties share the zero minimum-norm solution.
        winners[column]=dict(status='exact_zero_response',prior='zero_phrase_row',lam=None,
                            estimator='ridge',r2=None,selection='All configured ridge penalties have the same zero solution')
    active=torch.where(~zero)[0]
    if not len(active):return matrices,winners,all_results
    groups=prompt_ids.unique()
    if len(groups)<5:raise ValueError('five-fold CV requires five prompts')
    generator=torch.Generator().manual_seed(seed)
    folds=torch.tensor_split(groups[torch.randperm(len(groups),generator=generator).to(groups.device)],5)
    response=y[:,active];prior=matrices[active]
    scores=[]
    for penalty in lambdas:
        predictions=torch.empty_like(response)
        for group in folds:
            held=torch.isin(prompt_ids,group)
            mapping=ridge(delta[~held],response[~held],penalty,prior)
            predictions[held]=delta[held]@mapping.T
        per_phrase=1-(response-predictions).square().sum(0)/denominator[active]
        if not torch.isfinite(per_phrase).all():raise FloatingPointError('Nonfinite phrase CV scores')
        scores.append(per_phrase)
        for index,column in enumerate(active.tolist()):
            all_results[column].append(dict(prior='zero_phrase_row',lam=penalty,estimator='ridge',r2=float(per_phrase[index])))
    # argmax retains the first configured penalty in an exact tie, as scalar max did.
    chosen=torch.stack(scores).argmax(0)
    for candidate in chosen.unique().tolist():
        selected=active[chosen==candidate]
        matrices[selected]=ridge(delta,y[:,selected],lambdas[candidate],matrices[selected])
        for column in selected.tolist():
            winners[column]=dict(all_results[column][candidate],status='cv_selected')
    return matrices,winners,all_results

def whitened_template(activations,reference_mean,covariance,eta):
    return torch.linalg.solve(covariance+eta*torch.eye(len(covariance),device=covariance.device,dtype=covariance.dtype),
                              activations.mean(0)-reference_mean)
