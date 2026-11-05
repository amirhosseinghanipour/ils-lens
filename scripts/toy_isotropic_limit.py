"""Nonlinear trained-toy isotropic-limit check against our horizon-matched J.

Random orthogonal directions give exactly balanced covariance within each
context. This tests the estimator's limit without mistaking Monte Carlo
covariance imbalance for failure of the identity. It is not an iid-donor test.
"""
import json,time
from pathlib import Path
import torch
from transformers import Qwen2Config,Qwen2ForCausalLM
from ilslens.eval.toy_hmm import sequences
from ilslens.suffix import QwenSuffix
from ilslens.fit import ridge
from ilslens.hooks import capture_blocks
from ilslens.readout import raw

torch.set_num_threads(2);torch.manual_seed(97)
root=Path('artifacts/toy-isotropic');root.mkdir(parents=True,exist_ok=True)
config=dict(seed=97,contexts=8,directions_per_context=16,N=128,epsilons=[.01,.001,.0001],
    gamma=1.,source_position=32,lambda_=0,estimated_seconds=60,estimated_disk_mb=1,
    direction_design='independently rotated orthogonal basis per context; marginally isotropic, not iid')
(root/'config.json').write_text(json.dumps(config,indent=2));start=time.perf_counter()
model=Qwen2ForCausalLM(Qwen2Config(vocab_size=14,hidden_size=16,intermediate_size=32,
    num_hidden_layers=4,num_attention_heads=2,num_key_value_heads=1,max_position_embeddings=128,
    tie_word_embeddings=True,attn_implementation='eager')).float().eval().requires_grad_(False)
model.load_state_dict(torch.load('artifacts/toy/model.pt',weights_only=True))
ids,_,_=sequences(8,140000);heldout,_,_=sequences(20,150000)
with torch.no_grad(),capture_blocks(model,range(4)) as states:model.model(heldout,use_cache=False)
hidden={layer:h[:,32].clone() for layer,h in states.items()};rows=[]
for layer in range(4):
    jacobians=[];inputs={epsilon:[] for epsilon in config['epsilons']};outputs={epsilon:[] for epsilon in config['epsilons']}
    for tokens in ids:
        cache=QwenSuffix(model,tokens[None]);h=cache.states[layer]
        def function(delta):
            state=h.clone();state[0,32]=state[0,32]+delta
            return cache.forward(layer,state)[0,32:-1].mean(0)
        jacobians.append(torch.autograd.functional.jacobian(function,torch.zeros(16),vectorize=True))
        basis=torch.linalg.qr(torch.randn(16,16)).Q*4
        for epsilon in config['epsilons']:
            delta=basis*epsilon
            inputs[epsilon].append(delta)
            outputs[epsilon].append(cache.effects(layer,32,delta,gammas=(1.,))[:,0])
    j=torch.stack(jacobians).mean(0)
    for epsilon in config['epsilons']:
        a=ridge(torch.cat(inputs[epsilon]),torch.cat(outputs[epsilon]),lam=0)
        cosine=torch.nn.functional.cosine_similarity(a.flatten(),j.flatten(),dim=0).item()
        top_a=raw(hidden[layer],a,model.lm_head.weight,model.model.norm.weight).topk(10).indices
        top_j=raw(hidden[layer],j,model.lm_head.weight,model.model.norm.weight).topk(10).indices
        overlap=sum(len(set(x.tolist())&set(y.tolist()))/10 for x,y in zip(top_a,top_j))/len(top_a)
        rows.append(dict(layer=layer,epsilon=epsilon,matrix_cosine=cosine,heldout_top10_overlap=overlap,
                         passes=cosine>=.95 and overlap>=.8))
(root/'results.json').write_text(json.dumps(dict(config=config,rows=rows,seconds=time.perf_counter()-start,
    caveat='Horizon-matched local J, not upstream summed J; trained toy only; no evaluation-based epsilon selection'),indent=2))
print(json.dumps(rows,indent=2))
