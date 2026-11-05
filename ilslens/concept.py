import torch

def projection(short,long,covariance,eta):
    c = short@covariance@short.T
    return torch.linalg.solve(c+eta*torch.eye(len(c),dtype=c.dtype,device=c.device),
                             (long@covariance@short.T).T).T

def read(h,short,long,b,unembedding,gain,eps=1e-6):
    long_h = h@long.T
    state = long_h-(h@short.T)@b.T
    # Gain only on concept residual; denominator is RMS of LONG state.
    return ((state*gain)/(long_h.square().mean(-1,keepdim=True)+eps).sqrt())@unembedding.T
