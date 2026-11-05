"""Local-only fp32 load, generation and residual-gradient smoke test."""
import json, time, resource, os
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.set_num_threads(4)
torch.manual_seed(0)
path = os.environ.get('ILS_MODEL_PATH', 'models/Qwen2.5-0.5B-Instruct')
start = time.perf_counter()
model = AutoModelForCausalLM.from_pretrained(path, local_files_only=True, dtype=torch.float32, attn_implementation='eager')
model.eval().requires_grad_(False)
tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
inputs = tok('The capital of France is', return_tensors='pt')
with torch.no_grad():
    result = model.generate(**inputs, max_new_tokens=8, do_sample=False)
captured = []
def hook(module, args, output):
    h = output[0] if isinstance(output, tuple) else output
    h = h.detach().requires_grad_(True)
    captured.append(h)
    return (h, *output[1:]) if isinstance(output, tuple) else h
handle = model.model.layers[0].register_forward_hook(hook)
start_fb = time.perf_counter()
out = model(**inputs, use_cache=False)
grad, = torch.autograd.grad(out.logits[0,-1,0], captured[0])
handle.remove()
metrics = dict(text=tok.decode(result[0]), dtype=str(next(model.parameters()).dtype),
    tied=model.lm_head.weight.data_ptr()==model.model.embed_tokens.weight.data_ptr(),
    gradient_finite=bool(grad.isfinite().all()), gradient_norm=grad.norm().item(),
    forward_backward_seconds=time.perf_counter()-start_fb,
    total_seconds=time.perf_counter()-start, maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    norm_class=type(model.model.norm).__name__, mlp_class=type(model.model.layers[0].mlp).__name__)
Path('logs/smoke.json').write_text(json.dumps(metrics, indent=2))
print(json.dumps(metrics, indent=2))
assert metrics['gradient_finite'] and metrics['gradient_norm'] > 0
