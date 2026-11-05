"""Local Jacobian diagnostic and LRP rules; NOT a jlens.fit replacement baseline."""
from contextlib import contextmanager
import types
import torch
import torch.nn.functional as F

def average_jacobian(functions, points):
    return torch.stack([torch.autograd.functional.jacobian(f,x,vectorize=True)
                        for f,x in zip(functions,points)]).mean(0)

@contextmanager
def lrp_rules(model):
    originals = []
    def norm_forward(self,x):
        denominator = (x.float().square().mean(-1,keepdim=True)+self.variance_epsilon).sqrt().detach()
        return self.weight*x/denominator
    def mlp_forward(self,x):
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        # SiLU(x)/x = sigmoid(x), including the continuous limit at x=0.
        activation = gate*torch.sigmoid(gate).detach()
        product = .5*(activation*up.detach()+activation.detach()*up)
        return self.down_proj(product)
    try:
        for module in model.modules():
            method = None
            if type(module).__name__ in ('Qwen2RMSNorm','Qwen3RMSNorm'):
                method = norm_forward
            elif type(module).__name__ in ('Qwen2MLP','Qwen3MLP'):
                method = mlp_forward
            if method:
                originals.append((module,module.forward))
                module.forward = types.MethodType(method,module)
        yield
    finally:
        for module,forward in originals:
            module.forward = forward
