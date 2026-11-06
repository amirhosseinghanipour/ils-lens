"""Exact Qwen2 suffix replay with cached clean block outputs and static kwargs.

Only masks and positional encodings are reused. Attention weights are recomputed.
KV caching and dropout are disallowed, and no final normalization is returned.
"""
import torch
from .hooks import tensor_output
from .sampling import horizon_weights

class QwenSuffix:
    def __init__(self,model,ids):
        if model.config.model_type!='qwen2' or model.training:
            raise ValueError('Requires an eval-mode Qwen2 model')
        self.model=model
        self.ids=ids.to(model.model.embed_tokens.weight.device)
        self.states={};self.kwargs={}
        handles=[]
        def before(i):
            def capture(module,args,kwargs):
                if kwargs.get('past_key_values') is not None:
                    raise ValueError('KV cache is unsupported')
                self.kwargs[i]=kwargs.copy()
            return capture
        def after(i):
            def capture(module,args,output):
                self.states[i]=tensor_output(output).detach()
            return capture
        try:
            for i,block in enumerate(model.model.layers):
                handles.append(block.register_forward_pre_hook(before(i),with_kwargs=True))
                handles.append(block.register_forward_hook(after(i)))
            with torch.no_grad():model.model(self.ids,use_cache=False)
        finally:
            for h in handles:h.remove()

    def forward(self,layer,state):
        for i in range(layer+1,len(self.model.model.layers)):
            state=tensor_output(self.model.model.layers[i](state,**self.kwargs[i]))
        return state

    @torch.no_grad()
    def effects(self,layer,position,deltas,gammas=(0.,.5,.9,1.),skip=16):
        if self.ids.shape[0]!=1:
            raise ValueError('Cache exactly one sequence per replay object')
        deltas=deltas.to(self.states[layer]).reshape(-1,self.states[layer].shape[-1])
        n=len(deltas)
        h=self.states[layer].expand(2*n,-1,-1).clone()
        h[:n,position]+=deltas
        h[n:,position]-=deltas
        final=self.forward(layer,h)
        torch.testing.assert_close(final[:n,:position],final[n:,:position],atol=1e-6,rtol=0)
        effect=(final[:n]-final[n:])/2
        return torch.stack([torch.einsum('t,ntd->nd',
            horizon_weights(position,self.ids.shape[1],g,skip).to(effect.device),
            effect[:,position:-1]) for g in gammas],dim=1)
