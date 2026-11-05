"""Matched covariance controls for toy state-token lens directions.

Probe continuation: append STATE after a non-report observation, then measure
the correct hidden-state token's probability. Gates use signed probability
increase, 100 equal-norm controls, and no evaluation-guided scale selection.
"""
import json,time
from pathlib import Path
import torch
from transformers import Qwen2Config,Qwen2ForCausalLM
from ilslens.eval.toy_hmm import sequences
from ilslens.hooks import capture_blocks,patch_site
from ilslens.gate import random_direction_gate

torch.set_num_threads(2);torch.manual_seed(73)
root=Path('artifacts/toy-gate');root.mkdir(parents=True,exist_ok=True)
config=dict(seed=73,prompts=8,source_position=40,probe='STATE',epsilon_relative_to_state_norm=.1,
    layers=[0,1,2],controls=100,estimated_seconds=180,estimated_disk_mb=2,
    statistic='signed change in probability of the true hidden-state ID',calibration_seed=80000,test_seed=90000)
(root/'config.json').write_text(json.dumps(config,indent=2));start=time.perf_counter()
model=Qwen2ForCausalLM(Qwen2Config(vocab_size=14,hidden_size=16,intermediate_size=32,
    num_hidden_layers=4,num_attention_heads=2,num_key_value_heads=1,max_position_embeddings=128,
    tie_word_embeddings=True,attn_implementation='eager')).float().eval().requires_grad_(False)
model.load_state_dict(torch.load('artifacts/toy/model.pt',weights_only=True))
calibration,_,_=sequences(64,80000)
with torch.no_grad(),capture_blocks(model,[0,1,2]) as states:model.model(calibration,use_cache=False)
covariance={layer:torch.cov(h[:,16:-1].reshape(-1,16).T) for layer,h in states.items()}
matrices=torch.load('artifacts/toy-reference/lenses.pt',weights_only=True)['matrices']
tokens,truth,report=sequences(32,90000);rows=[];item=0
for index in range(len(tokens)):
    if report[index,40]:continue
    ids=torch.cat([tokens[index,:41],torch.tensor([8])])[None]
    target=int(truth[index,40])+9
    with torch.no_grad(),capture_blocks(model,[0,1,2]) as states:
        baseline=model(ids,use_cache=False).logits[0,-1].softmax(0)[target].item()
    hidden={layer:h[0,40].clone() for layer,h in states.items()}
    for name,maps in matrices.items():
        for layer in config['layers']:
            if layer not in maps:continue
            direction=(model.lm_head.weight[target]*model.model.norm.weight)@maps[layer]
            direction=direction/direction.norm().clamp_min(1e-12)*.1*hidden[layer].norm()
            @torch.no_grad()
            def effect(delta):
                with patch_site(model,layer,40,delta):
                    return model(ids,use_cache=False).logits[0,-1].softmax(0)[target].item()-baseline
            gate=random_direction_gate(direction,covariance[layer],effect,seed=73+100*item+layer)
            rows.append(dict(item=item,lens=name,label_tuned=name.startswith('J++'),layer=layer,
                target=target,baseline_probability=baseline,**gate))
    item+=1
    print('completed gate item',item,flush=True)
    if item==config['prompts']:break
summary=[]
for name in matrices:
    for layer in config['layers']:
        subset=[r for r in rows if r['lens']==name and r['layer']==layer]
        if not subset:continue
        passes=torch.tensor([r['passes'] for r in subset]).float()
        indices=torch.randint(len(passes),(200,len(passes)),generator=torch.Generator().manual_seed(73))
        summary.append(dict(lens=name,label_tuned=name.startswith('J++'),layer=layer,n=len(passes),
            pass_rate=passes.mean().item(),ci=passes[indices].mean(1).quantile(torch.tensor([.025,.975])).tolist()))
(root/'results.json').write_text(json.dumps(dict(config=config,rows=rows,summary=summary,
    seconds=time.perf_counter()-start,caveat='Toy state-report probe only; eight prompts and one trained seed'),indent=2))
