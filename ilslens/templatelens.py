import torch
from .fit import select
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
    # Phrase readouts map residual deltas to scalar log-probability changes;
    # zero prior is appropriate for rows without a residual identity map.
    if not lambdas or any(value<0 or not torch.isfinite(torch.tensor(value)) for value in lambdas):
        raise ValueError('Nonnegative finite ridge candidates are required')
    if not torch.isfinite(delta).all() or not torch.isfinite(log_probability_change).all():
        raise ValueError('Nonfinite phrase fitting data')
    prior = delta.new_zeros(log_probability_change.shape[1],delta.shape[1])
    matrices=[];winners=[];all_results=[]
    for column in range(log_probability_change.shape[1]):
        response=log_probability_change[:,column:column+1]
        if torch.count_nonzero(response)==0:
            # With zero prior, every ridge penalty returns exactly the zero row
            # (the minimum-norm solution at lambda=0). No R² choice is possible
            # or necessary; do not pretend that an arbitrary penalty won CV.
            matrices.append(prior[column:column+1].clone())
            winners.append(dict(status='exact_zero_response',prior='zero_phrase_row',lam=None,
                                estimator='ridge',r2=None,selection='All configured ridge penalties have the same zero solution'))
            all_results.append([])
            continue
        a,winner,results = select(delta,response,prompt_ids,{'I':prior[column:column+1]},lambdas,
                                 estimators=('ridge',),seed=seed)
        winner['prior'] = 'zero_phrase_row'
        winner['status']='cv_selected'
        for row in results:row['prior']='zero_phrase_row'
        matrices.append(a);winners.append(winner);all_results.append(results)
    return torch.cat(matrices),winners,all_results

def whitened_template(activations,reference_mean,covariance,eta):
    return torch.linalg.solve(covariance+eta*torch.eye(len(covariance),device=covariance.device,dtype=covariance.dtype),
                              activations.mean(0)-reference_mean)
