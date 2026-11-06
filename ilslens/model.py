import os
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from .runtime import resolve_device

def load_local(device='cuda',path=None,min_free_gib=6.):
    device=resolve_device(device,min_free_gib=min_free_gib)
    path=path or os.environ['ILS_MODEL_PATH']
    model=AutoModelForCausalLM.from_pretrained(path,local_files_only=True,dtype=torch.float32,
        attn_implementation='eager').to(device).eval().requires_grad_(False)
    if getattr(model,'is_quantized',False):raise ValueError('Research requires dequantized weights, not quantized kernels')
    if any(p.dtype!=torch.float32 for p in model.parameters()):raise ValueError('Model is not fully fp32')
    tokenizer=AutoTokenizer.from_pretrained(path,local_files_only=True)
    return model,tokenizer
