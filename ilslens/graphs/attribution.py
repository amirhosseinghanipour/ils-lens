import torch

def target_gradient(target,node):
    return torch.autograd.grad(target,node,retain_graph=True)[0]

def edge_effect(gradient,delta):
    return (gradient*delta).sum()

def lens_target(h,unembedding_row,gain,a):
    # Linear target, intentionally no input-dependent RMS normalization.
    vector = (unembedding_row*gain)@a
    return (h*vector).sum()
