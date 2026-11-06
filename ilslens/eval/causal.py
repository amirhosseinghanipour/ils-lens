"""Coordinate swaps and ablations at block-output sites; targets are never nodes."""
from contextlib import contextmanager
import torch
from ..hooks import tensor_output,replace_output

def swap_coordinates(h,vectors):
    if vectors.shape!=(h.shape[-1],2):raise ValueError('Swap requires two direction columns')
    # The specified Moore–Penrose formula is defined for singular V too.
    # Such a swap need not be an involution; do not silently impose full rank.
    coefficients=h@torch.linalg.pinv(vectors).T
    return h+(coefficients.flip(-1)-coefficients)@vectors.T

def ablate_direction(h,direction):
    denominator=direction.square().sum()
    if denominator<=0:raise ValueError('Cannot ablate a zero direction')
    return h-(h@direction)[...,None]*direction/denominator

@contextmanager
def intervene_band(model,transforms,positions):
    """Transforms map layer -> function on selected [batch,position,d] outputs.

    Positions are absolute prompt indices; callers must restrict generation to
    a no-KV-cache path when repeatedly using these hooks.
    """
    handles=[]
    positions=list(positions)
    if len(positions)!=len(set(positions)) or not positions or min(positions)<0:
        raise ValueError('Positions must be unique, nonnegative and nonempty')
    def patch(transform):
        def hook(module,args,output):
            h=tensor_output(output).clone()
            if max(positions)>=h.shape[1]:raise ValueError('Position outside current sequence; KV decoding is unsupported')
            h[:,positions]=transform(h[:,positions])
            return replace_output(output,h)
        return hook
    try:
        for layer,transform in transforms.items():handles.append(model.model.layers[layer].register_forward_hook(patch(transform)))
        yield
    finally:
        for handle in handles:handle.remove()

def next_token_success(logits,target_ids):
    target_ids=torch.as_tensor(target_ids,device=logits.device)
    if target_ids.numel()==0:raise ValueError('No target token IDs')
    ranking=logits.topk(min(5,logits.shape[-1])).indices
    return dict(top1=bool(torch.isin(ranking[:1],target_ids).any()),
                top5=bool(torch.isin(ranking,target_ids).any()),
                probability=float(logits.softmax(-1)[target_ids.unique()].sum()))

@torch.no_grad()
def gate_band(model,ids,transforms,positions,covariances,score,seed=0,count=100):
    """Match every realized per-site displacement norm along a multilayer intervention.

    Directions are sampled independently by layer, shared across the intervened
    positions within a layer, then scaled to the realized displacement at each
    position. The realized lens-intervention patch magnitude is fixed for all controls.
    """
    if count<100:raise ValueError('At least 100 matched controls are mandatory')
    if not transforms:raise ValueError('At least one intervened layer is required')
    def measured_score(logits):
        if not torch.isfinite(logits).all():raise FloatingPointError('Nonfinite causal logits')
        value=float(score(logits))
        if not torch.isfinite(torch.tensor(value)):raise FloatingPointError('Nonfinite causal score')
        return value
    baseline=measured_score(model(ids,use_cache=False).logits[0,-1]);deltas={}
    def recording(layer,transform):
        def call(h):
            result=transform(h)
            if result.shape!=h.shape:raise ValueError('Causal transform changed activation shape')
            if not torch.isfinite(result).all():raise FloatingPointError('Nonfinite causal transformed state')
            deltas[layer]=(result-h).clone();return result
        return call
    with intervene_band(model,{layer:recording(layer,transform) for layer,transform in transforms.items()},positions):
        observed_logits=model(ids,use_cache=False).logits[0,-1]
    observed=measured_score(observed_logits)-baseline
    common=dict(baseline_score=baseline,intervened_logits=observed_logits.cpu(),
        displacement_norms={layer:delta.norm(dim=-1).cpu().tolist() for layer,delta in deltas.items()},seed=seed,
        requested_controls=count,
        matching='Equal L2 norm at every layer and position; independent covariance-matched direction per layer')
    def undefined(reason):
        return dict(common,effect=observed,random_effects=[],threshold=None,passes=None,status='undefined',reason=reason)
    if all(torch.count_nonzero(delta)==0 for delta in deltas.values()):return undefined('zero_displacement')
    generator=torch.Generator(device=ids.device).manual_seed(seed);factors={}
    for layer in transforms:
        covariance=covariances[layer].to(ids.device)
        if not torch.isfinite(covariance).all():raise FloatingPointError('Nonfinite causal covariance')
        values,vectors=torch.linalg.eigh(covariance)
        if values.max()<=0 and torch.count_nonzero(deltas[layer]):return undefined('degenerate_covariance')
        factors[layer]=vectors*values.clamp_min(0).sqrt()[None,:]
    controls=[]
    for _ in range(count):
        patches={}
        for layer,delta in deltas.items():
            vector=factors[layer]@torch.randn(delta.shape[-1],generator=generator,device=ids.device,dtype=delta.dtype)
            if not torch.isfinite(vector).all():raise FloatingPointError('Nonfinite random control direction')
            if vector.norm()==0 and torch.count_nonzero(delta):raise ValueError('Zero sampled direction for nonzero intervention')
            vector=vector/vector.norm().clamp_min(1e-30)
            patch=delta.norm(dim=-1,keepdim=True)*vector
            patches[layer]=lambda h,p=patch:h+p
        with intervene_band(model,patches,positions):value=measured_score(model(ids,use_cache=False).logits[0,-1])-baseline
        controls.append(value)
    threshold=float(torch.tensor(controls).quantile(.95))
    return dict(common,effect=observed,random_effects=controls,threshold=threshold,passes=observed>threshold,status='measured')


def prompt_metric(rows,field,repeats=200,seed=0):
    """Prompt bootstrap of band means, retaining undefined control outcomes."""
    import numpy as np
    groups=sorted({r['item_id'] for r in rows});values=[];undefined=0
    for group in groups:
        valid=[r[field] for r in rows if r['item_id']==group and r[field] is not None]
        if valid:values.append(float(np.mean(valid)))
        else:undefined+=1
    values=np.asarray(values);rng=np.random.default_rng(seed)
    return dict(mean=float(values.mean()) if len(values) else None,
        ci=np.quantile(values[rng.integers(len(values),size=(repeats,len(values)))].mean(1),[.025,.975]).tolist() if len(values) else [None,None],
        measured_items=len(values),undefined_items=undefined,
        measured_cases=sum(r[field] is not None for r in rows),undefined_cases=sum(r[field] is None for r in rows))
