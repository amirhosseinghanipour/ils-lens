"""Upstream J/R/J++ comparison on the existing trained toy, with separate dev labels."""
import argparse,json,time
from pathlib import Path
from contextlib import nullcontext
import torch
from transformers import Qwen2Config,Qwen2ForCausalLM
from ilslens.reference import load_reference,token_adapter
from ilslens.jpp import load_jpp,fit_experts,fit_labeled_weights
from ilslens.baselines import lrp_rules
from ilslens.hooks import capture_blocks
from ilslens.readout import raw
from ilslens.eval.toy_hmm import sequences

parser=argparse.ArgumentParser();parser.add_argument('--reference',required=True);parser.add_argument('--jpp',required=True)
args=parser.parse_args();torch.set_num_threads(2);torch.manual_seed(17)
root=Path('artifacts/toy-reference');root.mkdir(parents=True,exist_ok=True)
estimate=dict(seconds=180,disk_mb=10,fit_prompts=128,dev_prompts=32,test_prompts=40,
    PCA=16,reason='Toy hidden size 16, so 64-dimensional PCA is impossible',seed=17)
(root/'estimate.json').write_text(json.dumps(estimate,indent=2));start=time.perf_counter()
reference=load_reference(args.reference);load_jpp(args.jpp)
model=Qwen2ForCausalLM(Qwen2Config(vocab_size=14,hidden_size=16,intermediate_size=32,
    num_hidden_layers=4,num_attention_heads=2,num_key_value_heads=1,max_position_embeddings=128,
    tie_word_embeddings=True,attn_implementation='eager')).float().eval().requires_grad_(False)
model.load_state_dict(torch.load('artifacts/toy/model.pt',weights_only=True))
train,_,train_report=sequences(128,20000);test,truth,test_report=sequences(40,40000)
dev,dev_truth,dev_report=sequences(32,60000)
generator=torch.Generator().manual_seed(37)
def positions(report,generator):
    result=[]
    for row in report:
        eligible=torch.where(~row[16:-1])[0]+16
        result.append(int(eligible[torch.randint(len(eligible),(),generator=generator)]))
    return torch.tensor(result)
positions(train_report,generator)
sites=positions(test_report,generator)
dev_sites=positions(dev_report,torch.Generator().manual_seed(61))
adapter=token_adapter(reference,model,None,train)
matrices={'logit':{layer:torch.eye(16) for layer in range(4)},'ILS':{}}
for name in ('J-reference','R-reference'):
    with lrp_rules(model) if name=='R-reference' else nullcontext():
        lens=reference.fit(adapter,[str(i) for i in range(len(train))],dim_batch=1,max_seq_len=128)
    matrices[name]={**lens.jacobians,3:torch.eye(16)}
for layer in range(4):
    matrices['ILS'][layer]=torch.load(f'artifacts/toy/lenses-{layer}.pt',weights_only=True)['matrices']['ILS']
experts,_=fit_experts(model,adapter,train,'runs/toy-reference-experts',projection_dim=16,model_id='toy-HMM-d16-L4')
with torch.no_grad(),capture_blocks(model,[0,1,2]) as states:model.model(dev,use_cache=False)
dev_states={layer:h[torch.arange(len(dev)),dev_sites].clone() for layer,h in states.items()}
jpp,weight_metadata=fit_labeled_weights(experts,adapter,dev_states,dev_truth[torch.arange(len(dev)),dev_sites]+9)
matrices['J++ (label-tuned)']={**jpp.jacobians_L_dict_FN}
with torch.no_grad(),capture_blocks(model,[0,1,2,3]) as states:model.model(test,use_cache=False)
hidden={layer:h[torch.arange(len(test)),sites].clone() for layer,h in states.items()}
targets=truth[torch.arange(len(test)),sites]+9
rows=[]
for name,maps in matrices.items():
    any_hits={k:torch.zeros(len(test),dtype=torch.bool) for k in (1,10)}
    for layer,matrix in maps.items():
        scores=raw(hidden[layer],matrix,model.lm_head.weight,model.model.norm.weight)
        for k in (1,10):
            hits=(scores.topk(k).indices==targets[:,None]).any(1)
            any_hits[k]|=hits
            indices=torch.randint(len(test),(200,len(test)),generator=torch.Generator().manual_seed(17))
            ci=hits[indices].float().mean(1).quantile(torch.tensor([.025,.975])).tolist()
            rows.append(dict(lens=name,label_tuned=name.startswith('J++'),layer=layer,k=k,
                recall=hits.float().mean().item(),ci=ci,n=len(test)))
    for k,hits in any_hits.items():
        indices=torch.randint(len(test),(200,len(test)),generator=torch.Generator().manual_seed(17))
        ci=hits[indices].float().mean(1).quantile(torch.tensor([.025,.975])).tolist()
        rows.append(dict(lens=name,label_tuned=name.startswith('J++'),layer='any_available',k=k,
                         recall=hits.float().mean().item(),ci=ci,n=len(test)))
torch.save(dict(matrices=matrices,jpp_dev=weight_metadata,metadata=estimate),root/'lenses.pt')
result=dict(rows=rows,seconds=time.perf_counter()-start,estimate=estimate,
    caveats=['Toy only, not Qwen','J/R/J++ target sums; ILS normalized uniform horizon',
             'J++ excludes final source layer per reference API; any_available bands differ',
             'All toy symbols have alphanumeric names, so readout filtering removes none',
             'Same 128 fitting sequences; J/R/J++ use every valid position, ILS samples one source per sequence',
             'Readout labels only used for J++ dev weights; evaluation prompts untouched'])
(root/'results.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
