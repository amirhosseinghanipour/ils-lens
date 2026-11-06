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
