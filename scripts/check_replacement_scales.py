"""Fixed-scale diagnostic of the retained live-TopK counterexample.

No scale is selected for production. This is a deterministic untrained tiny
network diagnostic, not a trained Qwen measurement or a causal claim.
"""
import argparse,json,time
from pathlib import Path
import torch
from transformers import Qwen2Config,Qwen2ForCausalLM
from ilslens.transcoders.train import SkipTranscoder
from ilslens.graphs.qwen import QwenGraph
from ilslens.graphs.intervention import replacement_effect
from ilslens.experiment import write_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',default='artifacts/replacement-scale-diagnostic.json');a=p.parse_args()
    estimate=dict(seconds=10,bytes=20000,device='cpu',scope='Tiny deterministic numerical diagnostic')
    print(json.dumps(dict(estimate=estimate)),flush=True)
    begin=time.perf_counter();torch.set_num_threads(2);torch.manual_seed(1)
    model=Qwen2ForCausalLM(Qwen2Config(vocab_size=32,hidden_size=16,intermediate_size=32,
        num_hidden_layers=4,num_attention_heads=2,num_key_value_heads=1,max_position_embeddings=128,
        attn_implementation='eager')).float().eval().requires_grad_(False)
    ids=torch.randint(0,32,(1,24));tcs=[SkipTranscoder(16).eval().requires_grad_(False) for _ in range(4)]
    graph=QwenGraph(model,ids,tcs)
    displacement=torch.zeros_like(graph.outputs[0]);displacement[0,20]=torch.randn(16)*10
    target=dict(kind='lens',layer=3,position=20,vector=torch.randn(16))
    rows=[]
    for scale in (0.,1e-6,1e-5,1e-4,1e-3,.01,.05,.1,.25,.5,1.):
        result=replacement_effect(graph,target,0,scale*displacement)
        rows.append(dict(scale=scale,displacement_norm=float((scale*displacement).norm()),**result))
    record=dict(seed=1,dtype='float32',device='cpu',trained=False,
        model='Synthetic random Qwen2 architecture: 4 layers, d=16, vocabulary=32',
        source_layer=0,target_layer=3,position=20,threshold=.001,estimate=estimate,
        rows=rows,actual_seconds=time.perf_counter()-begin,
        interpretation='Fixed scale sweep of the original counterexample; support counts are endpoint changes, not a proof that support stayed fixed along the path. No production scale selection or validation threshold change. No sampling CIs apply to this deterministic fixture.')
    write_json(Path(a.output),record)
    print(json.dumps(record,indent=2))


if __name__=='__main__':main()
