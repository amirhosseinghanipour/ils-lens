import torch

def validate_edge(function,x,delta,tolerance=1e-3):
    x = x.detach().requires_grad_(True)
    prediction = (torch.autograd.grad(function(x),x)[0]*delta).sum()
    actual = function(x.detach()+delta)-function(x.detach())
    error = (actual-prediction).abs()/torch.maximum(actual.abs(),prediction.abs()).clamp_min(1e-7)
    return dict(predicted=prediction.item(),actual=actual.item(),relative_error=error.item(),
                passes=error.item()<tolerance)
