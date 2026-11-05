"""Prompt-local Qwen2 replacement model with explicit feature/error source nodes.

Attention probabilities and norm denominators are captured from the original
model. Feature-source mode freezes each clean transcoder feature as an independent
node; this is an affine graph, NOT the unfrozen network's causal effect.
"""
from contextlib import contextmanager
import types
import torch
import torch.nn.functional as F
from transformers.models.qwen2.modeling_qwen2 import repeat_kv

class QwenGraph:
    def __init__(self,model,ids,transcoders):
        if model.config.model_type!='qwen2' or model.config._attn_implementation!='eager':
            raise ValueError('Requires Qwen2 with eager attention')
        if len(transcoders)!=len(model.model.layers):
            raise ValueError('A transcoder is required for every block')
        self.model=model;self.ids=ids;self.transcoders=transcoders
        self.denominators={};self.patterns={};self.inputs={};self.outputs={}
        handles=[]
        def norm_capture(name):
            def hook(module,args):
                self.denominators[name]=torch.rsqrt(args[0].square().mean(-1,keepdim=True)+module.variance_epsilon).detach()
            return hook
        def attention_capture(i):
            def hook(module,args,output):
                if output[1] is None:raise ValueError('Attention probabilities unavailable')
                self.patterns[i]=output[1].detach()
            return hook
        def mlp_capture(i):
            def hook(module,args,output):
                self.inputs[i]=args[0].detach();self.outputs[i]=output.detach()
            return hook
        try:
            for name,module in model.named_modules():
                if type(module).__name__=='Qwen2RMSNorm':
                    handles.append(module.register_forward_pre_hook(norm_capture(name)))
            for i,block in enumerate(model.model.layers):
                handles.append(block.self_attn.register_forward_hook(attention_capture(i)))
                handles.append(block.mlp.register_forward_hook(mlp_capture(i)))
            with torch.no_grad():self.clean_logits=model(ids,use_cache=False).logits
        finally:
            for handle in handles:handle.remove()
        with torch.no_grad():
            self.clean_features={i:tc.features(self.inputs[i]) for i,tc in enumerate(transcoders)}
            self.clean_errors={i:self.outputs[i]-tc(self.inputs[i]) for i,tc in enumerate(transcoders)}
            self.clean_embeddings=model.model.embed_tokens(ids).detach()

    @contextmanager
    def frozen(self,freeze_attention=True,feature_sources=True):
        saved=[];self.nodes={}
        def replace(module,method):
            saved.append((module,module.forward));module.forward=types.MethodType(method,module)
        def norm_forward(name):
            inverse_denominator=self.denominators[name]
            return lambda module,h:module.weight*(h*inverse_denominator)
        def attention_forward(i):
            pattern=self.patterns[i]
            def forward(module,hidden_states,**kwargs):
                b,t,_=hidden_states.shape
                values=module.v_proj(hidden_states).view(b,t,-1,module.head_dim).transpose(1,2)
                values=repeat_kv(values,module.num_key_value_groups)
                result=(pattern@values).transpose(1,2).reshape(b,t,-1)
                return module.o_proj(result),pattern
            return forward
        def mlp_forward(i):
            tc=self.transcoders[i]
            feature=self.clean_features[i].clone().requires_grad_(True)
            error=self.clean_errors[i].clone().requires_grad_(True)
            self.nodes[f'feature:{i}']=feature;self.nodes[f'error:{i}']=error
            def forward(module,h):
                features=feature if feature_sources else tc.features(h)
                # Algebraically identical to decoder(features)+skip(h)+error,
                # but centered at clean values to avoid cancellation of a large
                # reconstruction and error at every layer in fp32.
                return (self.outputs[i]+F.linear(features-self.clean_features[i],tc.decoder.weight)
                        +F.linear(h-self.inputs[i],tc.skip.weight)
                        +(error-self.clean_errors[i]))
            return forward
        try:
            for name,module in self.model.named_modules():
                if name in self.denominators:replace(module,norm_forward(name))
            for i,block in enumerate(self.model.model.layers):
                if freeze_attention:replace(block.self_attn,attention_forward(i))
                replace(block.mlp,mlp_forward(i))
            yield self
        finally:
            for module,method in reversed(saved):module.forward=method

    def forward(self):
        embeddings=self.clean_embeddings.clone().requires_grad_(True)
        self.nodes['embedding']=embeddings
        return self.model(inputs_embeds=embeddings,use_cache=False).logits

    def contributions(self,target):
        names=list(self.nodes)
        gradients=torch.autograd.grad(target,[self.nodes[n] for n in names],
                                     allow_unused=True,retain_graph=True)
        result={name:(gradient*self.nodes[name]).detach() if gradient is not None
                     else torch.zeros_like(self.nodes[name]) for name,gradient in zip(names,gradients)}
        # Bias mass is explicitly unexplained, not silently counted as features.
        residual_bias=target.detach()-sum(value.sum() for value in result.values())
        feature_mass=sum(value.abs().sum() for name,value in result.items() if name.startswith('feature:'))
        embedding_mass=result['embedding'].abs().sum()
        error_mass=sum(value.abs().sum() for name,value in result.items() if name.startswith('error:'))+residual_bias.abs()
        completeness=(feature_mass+embedding_mass)/(feature_mass+embedding_mass+error_mass).clamp_min(1e-12)
        return dict(contributions=result,bias=residual_bias,completeness=float(completeness))
