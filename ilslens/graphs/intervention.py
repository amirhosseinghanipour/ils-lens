"""Validate finite interventions in the specified, nonlinear replacement model.

This deliberately does not impose an extra freeze on transcoder activation
supports. Failed gradient predictions are measurements, not rounded into passes.
"""
import torch
from .evaluate import target_value


def replacement_effect(graph, target, layer, displacement):
    """Compare a decoded MLP-output displacement with its local derivative.

    A feature intervention is represented by its decoder column times the
    feature change. Errors use residual-coordinate displacements. Every
    downstream feature is recomputed, including its TopK selection.
    """
    if displacement.shape != graph.outputs[layer].shape:
        raise ValueError('Displacement must specify the entire MLP-output tensor')
    with graph.replacement():
        scale=torch.zeros((),device=displacement.device,dtype=displacement.dtype,requires_grad=True)
        def patch(module,args,out):
            return out+scale*displacement
        handle=graph.model.model.layers[layer].mlp.register_forward_hook(patch)
        try:
            clean=target_value(graph,target,graph.forward,frozen_gate=False)
            derivative=torch.autograd.grad(clean,scale)[0]
            clean_support={i:(graph.nodes[f'feature:{i}'].detach()!=0).clone()
                           for i in range(len(graph.transcoders))}
            with torch.no_grad():
                scale.fill_(1)
                changed=target_value(graph,target,graph.forward,frozen_gate=False)
            actual=float(changed-clean.detach());predicted=float(derivative)
            switches={i:int((clean_support[i]!=(graph.nodes[f'feature:{i}'].detach()!=0)).sum())
                      for i in range(len(graph.transcoders))}
        finally:
            handle.remove()
    relative=abs(actual-predicted)/max(abs(actual),abs(predicted),1e-7)
    return dict(predicted=predicted,intervention_effect=actual,relative_error=relative,
                passes=relative<1e-3,support_changes=switches,
                freeze='Attention patterns and RMS denominators only; live transcoder features')
