"""Coordinate swaps and ablations at block-output sites; targets are never nodes."""
from contextlib import contextmanager
import torch
from ..hooks import tensor_output,replace_output

def swap_coordinates(h,vectors):
    if vectors.shape!=(h.shape[-1],2):raise ValueError('Swap requires two direction columns')
    if int(torch.linalg.matrix_rank(vectors))<2:raise ValueError('Swap directions must be linearly independent')
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
    position. The clean-conditioned patch magnitude is fixed for all controls.
    """
    if count<100:raise ValueError('At least 100 matched controls are mandatory')
    baseline=float(score(model(ids,use_cache=False).logits[0,-1]));deltas={}
    def recording(layer,transform):
        def call(h):
            result=transform(h);deltas[layer]=(result-h).clone();return result
        return call
    with intervene_band(model,{layer:recording(layer,transform) for layer,transform in transforms.items()},positions):
        observed_logits=model(ids,use_cache=False).logits[0,-1]
    observed=float(score(observed_logits))-baseline
    generator=torch.Generator(device=ids.device).manual_seed(seed);factors={}
    for layer in transforms:
        covariance=covariances[layer].to(ids.device);values,vectors=torch.linalg.eigh(covariance)
        if values.max()<=0:raise ValueError('Degenerate layer covariance')
        factors[layer]=vectors*values.clamp_min(0).sqrt()[None,:]
    controls=[]
    for _ in range(count):
        patches={}
        for layer,delta in deltas.items():
            vector=factors[layer]@torch.randn(delta.shape[-1],generator=generator,device=ids.device,dtype=delta.dtype)
            vector=vector/vector.norm().clamp_min(1e-30)
            patch=delta.norm(dim=-1,keepdim=True)*vector
            patches[layer]=lambda h,p=patch:h+p
        with intervene_band(model,patches,positions):value=float(score(model(ids,use_cache=False).logits[0,-1]))-baseline
        controls.append(value)
    threshold=float(torch.tensor(controls).quantile(.95))
    return dict(effect=observed,random_effects=controls,threshold=threshold,passes=observed>threshold,
        baseline_score=baseline,intervened_logits=observed_logits.cpu(),
        displacement_norms={layer:delta.norm(dim=-1).cpu().tolist() for layer,delta in deltas.items()},seed=seed,
        matching='Equal L2 norm at every layer and position; independent covariance-matched direction per layer')
