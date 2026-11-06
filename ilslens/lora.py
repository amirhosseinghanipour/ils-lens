"""Small fp32 LoRA implementation with reversible base-model access."""
from contextlib import contextmanager
import math
import torch
from torch import nn
import torch.nn.functional as F

class LoRALinear(nn.Module):
    def __init__(self,base,rank=8,alpha=16):
        super().__init__()
        if not isinstance(base,nn.Linear) or rank<=0:raise ValueError('LoRA requires a linear layer and positive rank')
        self.base=base.requires_grad_(False);self.scale=alpha/rank;self.enabled=True
        self.a=nn.Parameter(base.weight.new_empty(rank,base.in_features))
        self.b=nn.Parameter(base.weight.new_zeros(base.out_features,rank))
        nn.init.kaiming_uniform_(self.a,a=math.sqrt(5))
    def forward(self,x):
        result=self.base(x)
        return result+self.scale*F.linear(F.linear(x,self.a),self.b) if self.enabled else result

class AdapterSet:
    def __init__(self,model,rank=8,alpha=16,targets=('q_proj','v_proj')):
        if any(isinstance(module,LoRALinear) for module in model.modules()):raise ValueError('Adapters already installed')
        model.requires_grad_(False);self.modules={};self.originals=[]
        self.config=dict(rank=rank,alpha=alpha,targets=list(targets))
        for name,module in list(model.named_modules()):
            if name.split('.')[-1] not in targets:continue
            parent_name,attribute=name.rsplit('.',1);parent=model.get_submodule(parent_name)
            adapter=LoRALinear(module,rank,alpha)
            setattr(parent,attribute,adapter);self.modules[name]=adapter;self.originals.append((parent,attribute,module))
        if not self.modules:raise ValueError('No target projections found')
    def parameters(self):
        return [p for module in self.modules.values() for p in (module.a,module.b)]
    @contextmanager
    def disabled(self):
        previous={name:module.enabled for name,module in self.modules.items()}
        try:
            for module in self.modules.values():module.enabled=False
            yield
        finally:
            for name,module in self.modules.items():module.enabled=previous[name]
    def state(self):
        return dict(config=self.config,weights={name:dict(a=m.a.detach().cpu().clone(),b=m.b.detach().cpu().clone()) for name,m in self.modules.items()})
    def load(self,state):
        if state['config']!=self.config or set(state['weights'])!=set(self.modules):raise ValueError('Adapter configuration mismatch')
        for name,module in self.modules.items():
            if state['weights'][name]['a'].shape!=module.a.shape or state['weights'][name]['b'].shape!=module.b.shape:
                raise ValueError('Adapter shape mismatch')
        with torch.no_grad():
            for name,module in self.modules.items():
                module.a.copy_(state['weights'][name]['a']);module.b.copy_(state['weights'][name]['b'])
    def remove(self):
        for parent,attribute,original in self.originals:setattr(parent,attribute,original)

def encode_completion(tokenizer,prompt,completion,max_length=128):
    prefix=tokenizer.encode(prompt,add_special_tokens=False)
    suffix=tokenizer.encode(completion,add_special_tokens=False)
    if not suffix or len(prefix)>=max_length:raise ValueError('Completion would be absent from training loss')
    suffix=suffix[:max_length-len(prefix)]
    ids=torch.tensor(prefix+suffix,dtype=torch.long)
    labels=ids.clone();labels[:len(prefix)]=-100
    return dict(input_ids=ids,labels=labels)

def train_adapters(model,adapters,examples,steps=100,learning_rate=1e-3,seed=0):
    if not examples or steps<1:raise ValueError('Need examples and positive training steps')
    if any(p.dtype!=torch.float32 for p in model.parameters()):raise ValueError('LoRA research runs require fp32')
    device=model.model.embed_tokens.weight.device
    generator=torch.Generator().manual_seed(seed)
    optimizer=torch.optim.AdamW(adapters.parameters(),lr=learning_rate)
    losses=[];model.train()
    try:
        for _ in range(steps):
            example=examples[int(torch.randint(len(examples),(),generator=generator))]
            optimizer.zero_grad(set_to_none=True)
            loss=model(input_ids=example['input_ids'][None].to(device),labels=example['labels'][None].to(device),use_cache=False).loss
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite training loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(adapters.parameters(),1.)
            optimizer.step();losses.append(loss.item())
    finally:model.eval()
    return losses
