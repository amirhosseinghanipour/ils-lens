"""Vocabulary-block calibration; never allocate reference_count × vocabulary."""
import numpy as np
import torch
from .readout import calibrate_token,dead_rows,zscore,fdr

@torch.no_grad()
def fit_calibration(transported,unembedding,gain,frequencies,minimum=100000,token_chunk=64,eps=1e-6):
    if len(transported)<minimum:raise ValueError('At least 100k held-out reference activations are required')
    if transported.dtype!=torch.float32:raise ValueError('Calibration requires fp32 activations')
    if len(frequencies)!=len(unembedding):raise ValueError('Vocabulary frequencies have wrong length')
    masked=dead_rows(unembedding,frequencies.to(unembedding.device)).cpu()
    normalized=transported/(transported.square().mean(-1,keepdim=True)+eps).sqrt()
    records=[]
    for first in range(0,len(unembedding),token_chunk):
        rows=(unembedding[first:first+token_chunk]*gain).to(normalized.device)
        scores=(normalized@rows.T).cpu().numpy()
        for offset in range(scores.shape[1]):
            values=scores[:,offset]
            if masked[first+offset]:
                record=dict(mean=float(values.mean()),std=float(values.std()),threshold=float('inf'),tail=None)
            else:record=calibrate_token(values,minimum=minimum)
            records.append(record)
    return dict(records=records,dead_mask=masked,reference_count=len(transported),
                epsilon=eps,tail_quantile=.95,fdr_method='BY_full_vocabulary',q=.05)

def calibrated_readout(scores,calibration,q=.05):
    if scores.ndim!=1:raise ValueError('Read one activation at a time for FDR')
    records=calibration['records']
    means=scores.new_tensor([r['mean'] for r in records]);stds=scores.new_tensor([r['std'] for r in records])
    z=zscore(scores,means,stds)
    dead=calibration['dead_mask'].to(scores.device)
    raw_top=scores.topk(min(10,len(scores))).indices
    dead_top_fraction=dead[raw_top].float().mean().item()
    z=z.masked_fill(dead,-torch.inf)
    accepted,p=fdr(scores.detach().cpu().numpy(),z.detach().cpu().numpy(),records,q=q)
    return dict(z=z,accepted_ids=accepted.tolist(),pvalues=p,
                raw_dead_top10_fraction=dead_top_fraction)
