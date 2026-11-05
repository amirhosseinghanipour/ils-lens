"""Upstream expert/label-weight integration smoke test on a tiny random model.

This is not a readout benchmark and produces no generalization claim.
"""
import argparse,json,time
from pathlib import Path
import torch
from transformers import Qwen2Config,Qwen2ForCausalLM
from ilslens.reference import load_reference,token_adapter
from ilslens.jpp import load_jpp,fit_experts,fit_labeled_weights
from ilslens.hooks import capture_blocks

parser=argparse.ArgumentParser();parser.add_argument('--reference',required=True);parser.add_argument('--jpp',required=True)
args=parser.parse_args();torch.set_num_threads(2);torch.manual_seed(51);start=time.perf_counter()
reference=load_reference(args.reference);load_jpp(args.jpp)
model=Qwen2ForCausalLM(Qwen2Config(vocab_size=32,hidden_size=16,intermediate_size=32,
    num_hidden_layers=4,num_attention_heads=2,num_key_value_heads=1,attn_implementation='eager')).float().eval().requires_grad_(False)
ids=torch.randint(0,32,(4,128));adapter=token_adapter(reference,model,None,ids)
output=Path('runs/jpp-smoke');output.mkdir(parents=True,exist_ok=True)
experts,router=fit_experts(model,adapter,ids,output,projection_dim=16,model_id='random-Qwen2-d16-L4')
dev=torch.randint(0,32,(12,128))
with torch.no_grad(),capture_blocks(model,[0,1,2]) as states:model.model(dev,use_cache=False)
dev_states={layer:h[:,-2].clone() for layer,h in states.items()}
targets=torch.randint(0,32,(12,))
lens,metadata=fit_labeled_weights(experts,adapter,dev_states,targets)
result=dict(seconds=time.perf_counter()-start,label_tuned=True,
    layers=list(metadata['weights']),experts=8,projection_dim=16,
    deviations=['Toy dimension requires PCA16, not PCA64','Random model and random dev targets; integration test only'],
    fallback={str(layer):experts.fallback_L_dict_Bool_E[layer].tolist() for layer in experts.layers},
    weights_finite=all(bool(w.isfinite().all()) for w in metadata['weights'].values()))
assert result['weights_finite']
assert set(lens.jacobians_L_dict_FN)=={0,1,2}
(output/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
