import torch
from .fit import select

def fit_phrase_rows(delta,log_probability_change,prompt_ids,lambdas,seed=0):
    # Phrase readouts map residual deltas to scalar log-probability changes;
    # zero prior is appropriate for rows without a residual identity map.
    prior = torch.zeros(log_probability_change.shape[1],delta.shape[1])
    a,winner,results = select(delta,log_probability_change,prompt_ids,{'I':prior},lambdas,
                             estimators=('ridge',),seed=seed)
    winner['prior'] = 'zero_phrase_row'
    for row in results:
        row['prior'] = 'zero_phrase_row'
    return a,winner,results

def whitened_template(activations,reference_mean,covariance,eta):
    return torch.linalg.solve(covariance+eta*torch.eye(len(covariance)),
                              activations.mean(0)-reference_mean)
