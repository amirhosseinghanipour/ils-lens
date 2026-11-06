"""Memory-bounded fp32 Jacobian and small-perturbation comparisons."""
import torch

def jacobian_rows(function,point,batch_size=4):
    point=point.detach().clone().requires_grad_(True);output=function(point)
    if point.ndim!=1 or output.ndim!=1:raise ValueError('Vector-to-vector map required')
    rows=[]
    for start in range(0,len(output),batch_size):
        stop=min(start+batch_size,len(output));cotangents=output.new_zeros(stop-start,len(output))
        cotangents[torch.arange(stop-start),torch.arange(start,stop)]=1
        rows.append(torch.autograd.grad(output,point,grad_outputs=cotangents,is_grads_batched=True,retain_graph=stop<len(output))[0])
    return torch.cat(rows).detach()

def relative_jvp_error(function,point,delta):
    _,jvp=torch.autograd.functional.jvp(function,point,delta)
    with torch.no_grad():difference=(function(point+delta)-function(point-delta))/2
    return float((difference-jvp).norm()/jvp.norm().clamp_min(1e-12))
