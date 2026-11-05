import json,time,resource,os
from pathlib import Path
import yaml
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from ilslens.hooks import patch_site,final_state
from ilslens.collect import antithetic

config=yaml.safe_load(Path('configs/qwen-pilot.yaml').read_text())
torch.manual_seed(config['seed']);torch.set_num_threads(4)
start=time.perf_counter()
path=os.environ.get('ILS_MODEL_PATH', 'models/Qwen2.5-0.5B-Instruct')
model=AutoModelForCausalLM.from_pretrained(path,dtype=torch.float32,local_files_only=True,
    attn_implementation='eager').eval().requires_grad_(False)
tok=AutoTokenizer.from_pretrained(path,local_files_only=True)
text='A researcher watched birds in the garden. She wrote their colors and movements in a notebook. '
ids=tok(text*40,return_tensors='pt').input_ids[:,:128]
direction=torch.randn(896);direction/=direction.norm()
rows=[]
for layer in config['layers']:
    def function(delta):
        with patch_site(model,layer,32,delta):
            return final_state(model,ids)[0,32:-1].mean(0)
    _,jvp=torch.autograd.functional.jvp(function,torch.zeros(896),direction)
    for epsilon in config['epsilon']:
        begin=time.perf_counter()
        y=antithetic(model,ids,layer,32,epsilon*direction)
        error=(y[-1,0]-epsilon*jvp).norm()/(epsilon*jvp).norm().clamp_min(1e-12)
        rows.append(dict(layer=layer,epsilon=epsilon,seconds=time.perf_counter()-begin,
            relative_jvp_error=error.item(),effect_norm=y[-1,0].norm().item()))
    print('completed qwen layer',layer,flush=True)
result=dict(config=config,rows=rows,total_seconds=time.perf_counter()-start,
    maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    caveat='Single prompt/direction numerical pilot. NOT interventional R2 or a causal lens claim.')
Path('artifacts/qwen-pilot.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
