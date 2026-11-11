"""Vocabulary-block calibration; never allocate reference_count × vocabulary."""
import numpy as np
import hashlib
import json
import torch
from .readout import calibrate_token,dead_rows,zscore,fdr

CALIBRATION_VERSION='empirical-variance-independent-of-tail-v2'

def calibration_input_fingerprint(activation_key,matrix,frequencies):
    """Bind a resumable fit to references, transport, and the corpus dead mask."""
    payload=dict(activation_key=activation_key,matrix_sha256=matrix_fingerprint(matrix),
                 frequencies_sha256=matrix_fingerprint(frequencies),version=CALIBRATION_VERSION)
    return hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def validate_calibration_cache(calibration,input_fingerprint,reference_count,vocab_size):
    if calibration.get('calibration_version')!=CALIBRATION_VERSION:
        raise ValueError('Calibration method changed; use a new calibration output directory')
    if calibration.get('metadata',{}).get('input_fingerprint')!=input_fingerprint:
        raise ValueError('Calibration references, transport or corpus frequencies changed or are unverified')
    if calibration.get('reference_count')!=reference_count or len(calibration['records'])!=vocab_size:
        raise ValueError('Calibration reference count or vocabulary is incomplete')
    if tuple(calibration['dead_mask'].shape)!=(vocab_size,):
        raise ValueError('Calibration dead-row mask has wrong shape')

def matrix_fingerprint(matrix):
    matrix=matrix.detach().cpu().contiguous()
    return hashlib.sha256(str((tuple(matrix.shape),str(matrix.dtype))).encode()+matrix.numpy().tobytes()).hexdigest()

def finite_topk(scores,k=10):
    """A masked/nonfinite score is not a readout, even if fewer than k remain."""
    eligible=torch.where(torch.isfinite(scores))[0]
    return eligible[torch.argsort(scores[eligible],descending=True,stable=True)[:k]]

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
                calibration_version=CALIBRATION_VERSION,
                epsilon=eps,tail_quantile=.95,fdr_method='BY_full_vocabulary',q=.05)

def calibrated_readout(scores,calibration,q=.05):
    if scores.ndim!=1:raise ValueError('Read one activation at a time for FDR')
    records=calibration['records']
    means=scores.new_tensor([r['mean'] for r in records]);stds=scores.new_tensor([r['std'] for r in records])
    z=zscore(scores,means,stds)
    dead=calibration['dead_mask'].to(scores.device)
    raw_top=finite_topk(scores)
    dead_top_fraction=dead[raw_top].float().mean().item() if len(raw_top) else None
    z=z.masked_fill(dead,-torch.inf)
    z_top=finite_topk(z)
    accepted,p=fdr(scores.detach().cpu().numpy(),z.detach().cpu().numpy(),records,q=q)
    return dict(z=z,accepted_ids=accepted.tolist(),pvalues=p,
                raw_top10_ids=raw_top.tolist(),z_top10_ids=z_top.tolist(),
                raw_dead_top10_fraction=dead_top_fraction,
                z_dead_top10_fraction=dead[z_top].float().mean().item() if len(z_top) else None)


def record_calibration_timing(root, record):
    """Retain previous calibration costs when extending a completed lens bank."""
    from pathlib import Path
    from .experiment import write_json
    root=Path(root)
    history_path=root/'timing-history.json'
    latest_path=root/'timing.json'
    if history_path.exists():
        history=json.loads(history_path.read_text())
    else:
        history=[]
        if latest_path.exists():
            history.append(dict(json.loads(latest_path.read_text()),
                                provenance='Imported prior timing; per-invocation map counts were not recorded'))
    history.append(record)
    write_json(history_path,history)
    write_json(latest_path,record)
