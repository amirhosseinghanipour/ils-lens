"""Tiny fp32 diagnostic of the proposal's freeze, without production fitting."""
import argparse,json,time
from pathlib import Path
import torch
from transformers import Qwen2Config,Qwen2ForCausalLM
from ilslens.transcoders.train import SkipTranscoder
from ilslens.graphs.qwen import QwenGraph
from ilslens.graphs.intervention import replacement_effect


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',default='artifacts/replacement-semantics.json')
    args=parser.parse_args()
    # Untrained, deterministic tiny network: a numerical diagnostic, not a
    # Qwen checkpoint experiment or evidence about trained workspace structure.
    print(json.dumps(dict(estimated_seconds=10,estimated_disk_bytes=10000,device='cpu',scope='tiny numerical validation')),flush=True)
    start=time.perf_counter();torch.set_num_threads(2);torch.manual_seed(1)
    model=Qwen2ForCausalLM(Qwen2Config(vocab_size=32,hidden_size=16,intermediate_size=32,
        num_hidden_layers=4,num_attention_heads=2,num_key_value_heads=1,max_position_embeddings=128,
        attn_implementation='eager')).float().eval().requires_grad_(False)
    ids=torch.randint(0,32,(1,24));transcoders=[SkipTranscoder(16).eval().requires_grad_(False) for _ in range(4)]
    graph=QwenGraph(model,ids,transcoders)
    delta=torch.zeros_like(graph.outputs[0]);delta[0,20]=torch.randn(16)*10
    target=dict(kind='lens',layer=3,position=20,vector=torch.randn(16))
    result=replacement_effect(graph,target,0,delta)
    record=dict(seed=1,dtype='float32',device='cpu',trained=False,
        model='Synthetic random Qwen2 architecture: 4 layers, d=16, vocabulary=32',
        estimand='Finite MLP-output patch versus local derivative with live downstream TopK',
        result=result,actual_seconds=time.perf_counter()-start,
        interpretation='A support-switch counterexample tests the universal finite-edge equality requirement; not a trained-model causal claim.')
    destination=Path(args.output);destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2))


if __name__=='__main__':main()
