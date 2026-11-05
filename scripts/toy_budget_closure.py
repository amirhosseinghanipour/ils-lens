"""Toy-only per-item lens coverage and frozen feature-graph completeness."""
import json,time
from pathlib import Path
import torch
import torch.nn.functional as F
from transformers import Qwen2Config,Qwen2ForCausalLM
from ilslens.eval.toy_hmm import sequences
from ilslens.hooks import capture_blocks
from ilslens.suffix import QwenSuffix
from ilslens.fit import r2
from ilslens.transcoders.train import train
from ilslens.transcoders.eval import metrics,ce_recovered
from ilslens.graphs.qwen import QwenGraph

torch.set_num_threads(2);torch.manual_seed(83)
root=Path('artifacts/toy-budget');root.mkdir(parents=True,exist_ok=True)
config=dict(seed=83,training_sequences=16,training_tokens=2048,steps=200,
    width_multiplier=8,top_k=8,evaluation_items=8,donors_per_item=32,epsilon=.05,gamma=1.,
    source_position=40,low_coverage_threshold=.5,low_completeness_threshold=.5,
    estimated_seconds=180,estimated_disk_mb=2,
    graph_freeze='attention probabilities, norm inverse denominators, independent feature sources')
(root/'config.json').write_text(json.dumps(config,indent=2));start=time.perf_counter()
model=Qwen2ForCausalLM(Qwen2Config(vocab_size=14,hidden_size=16,intermediate_size=32,
    num_hidden_layers=4,num_attention_heads=2,num_key_value_heads=1,max_position_embeddings=128,
    tie_word_embeddings=True,attn_implementation='eager')).float().eval().requires_grad_(False)
model.load_state_dict(torch.load('artifacts/toy/model.pt',weights_only=True))
training,_,_=sequences(16,110000);evaluation,truth,_=sequences(8,120000)
donor_tokens,_,_=sequences(32,130000)
inputs={};outputs={};handles=[]
def capture(i):
    def hook(module,args,out):inputs[i]=args[0].detach();outputs[i]=out.detach()
    return hook
for i,block in enumerate(model.model.layers):handles.append(block.mlp.register_forward_hook(capture(i)))
with torch.no_grad():model(training,use_cache=False)
for h in handles:h.remove()
transcoders=[train(inputs[i].reshape(-1,16),outputs[i].reshape(-1,16),steps=200,seed=83+i).eval().requires_grad_(False) for i in range(4)]
for i,tc in enumerate(transcoders):torch.save(tc.state_dict(),root/f'transcoder-{i}.pt')
inputs={};outputs={};handles=[]
for i,block in enumerate(model.model.layers):handles.append(block.mlp.register_forward_hook(capture(i)))
def losses(logits):
    return F.cross_entropy(logits[:,:-1].reshape(-1,14),evaluation[:,1:].reshape(-1),reduction='none').reshape(len(evaluation),-1).mean(1)
with torch.no_grad():clean_losses=losses(model(evaluation).logits);clean_ce=clean_losses.mean().item()
for h in handles:h.remove()
transcoder_metrics=[]
bootstrap_ids=torch.randint(8,(200,8),generator=torch.Generator().manual_seed(83))
for i,tc in enumerate(transcoders):
    with torch.no_grad():prediction=tc(inputs[i])
    def explained(index):
        actual=outputs[i][index].reshape(-1,16);pred=prediction[index].reshape(-1,16)
        return 1-(actual-pred).square().sum()/(actual-actual.mean(0)).square().sum()
    ci=torch.stack([explained(index) for index in bootstrap_ids]).quantile(torch.tensor([.025,.975])).tolist()
    transcoder_metrics.append(dict(layer=i,**metrics(tc,inputs[i].reshape(-1,16),outputs[i].reshape(-1,16)),variance_explained_ci=ci))
for i,block in enumerate(model.model.layers):
    def replacement(i):return lambda module,args,out:transcoders[i](args[0])
    handles.append(block.mlp.register_forward_hook(replacement(i)))
with torch.no_grad():replaced_losses=losses(model(evaluation).logits);replaced_ce=replaced_losses.mean().item()
for h in handles:h.remove()
handles=[]
for block in model.model.layers:handles.append(block.mlp.register_forward_hook(lambda module,args,out:torch.zeros_like(out)))
with torch.no_grad():ablated_losses=losses(model(evaluation).logits);ablated_ce=ablated_losses.mean().item()
for h in handles:h.remove()
lenses={i:torch.load(f'artifacts/toy/lenses-{i}.pt',weights_only=True)['matrices']['ILS'] for i in range(4)}
bank=[]
for tokens in donor_tokens:
    cache=QwenSuffix(model,tokens[None]);bank.append(torch.stack([cache.states[i][0,40] for i in range(4)]))
bank=torch.stack(bank);rows=[]
for item,tokens in enumerate(evaluation):
    ids=tokens[None];cache=QwenSuffix(model,ids);graph=QwenGraph(model,ids,transcoders)
    for layer in range(4):
        deltas=.05*(bank[:,layer]-cache.states[layer][0,40])
        effects=cache.effects(layer,40,deltas,gammas=(1.,))[:,0]
        coverage=r2(effects,deltas@lenses[layer].T)
        vector=(model.lm_head.weight[int(truth[item,40])+9]*model.model.norm.weight)@lenses[layer]
        with graph.frozen():
            with capture_blocks(model,[layer]) as states:graph.forward()
            target=(states[layer][0,40]*vector).sum()
            contributions=graph.contributions(target)
            feature_keys=[k for k in contributions['contributions'] if k.startswith('feature:')]
            key=max(feature_keys,key=lambda k:contributions['contributions'][k].abs().max().item())
            node=graph.nodes[key]
            gradient=torch.autograd.grad(target,node,retain_graph=True)[0]
            index=int((gradient*node).abs().flatten().argmax())
            change=-node.flatten()[index].detach().clone()
            predicted=(gradient.flatten()[index]*change).item()
            with torch.no_grad():node.flatten()[index]+=change
            with capture_blocks(model,[layer]) as patched:graph.forward()
            actual=((patched[layer][0,40]*vector).sum()-target.detach()).item()
            relative=abs(actual-predicted)/max(abs(actual),abs(predicted),1e-7)
            complete=contributions['completeness']
        with graph.frozen(freeze_attention=False):
            with capture_blocks(model,[layer]) as states:graph.forward()
            clean_target=(states[layer][0,40]*vector).sum().detach()
            with torch.no_grad():graph.nodes[key].flatten()[index]+=change
            with capture_blocks(model,[layer]) as states:graph.forward()
            norm_only=((states[layer][0,40]*vector).sum()-clean_target).item()
        rows.append(dict(item=item,layer=layer,source_position=40,hidden_state=int(truth[item,40]),
            lens_r2=coverage,unexplained=1-coverage,graph_completeness=complete,
            dark_matter=coverage<.5 and complete<.5,edge_relative_error=relative,
            edge_passes=relative<1e-3,predicted_effect=predicted,frozen_effect=actual,
            norm_only_effect=norm_only,survival_ratio=norm_only/predicted if abs(predicted)>1e-7 else None))
    print('completed budget item',item,flush=True)
recovered_bootstrap=(ablated_losses[bootstrap_ids].mean(1)-replaced_losses[bootstrap_ids].mean(1))/(ablated_losses[bootstrap_ids].mean(1)-clean_losses[bootstrap_ids].mean(1))
dark_by_prompt=torch.tensor([sum(r['dark_matter'] for r in rows if r['item']==i)/4 for i in range(8)])
result=dict(config=config,rows=rows,transcoders=transcoder_metrics,
    ce=dict(clean=clean_ce,replaced=replaced_ce,ablated=ablated_ce,recovered=ce_recovered(clean_ce,replaced_ce,ablated_ce),
            recovered_ci=recovered_bootstrap.quantile(torch.tensor([.025,.975])).tolist()),
    dark_matter_fraction=sum(r['dark_matter'] for r in rows)/len(rows),
    dark_matter_ci=dark_by_prompt[bootstrap_ids].mean(1).quantile(torch.tensor([.025,.975])).tolist(),
    all_checked_edges_pass=all(r['edge_passes'] for r in rows),seconds=time.perf_counter()-start,
    caveats=['Toy only; thresholds fixed before this run','Attribution bias remainder counted as error mass',
             'Donor position fixed at 40 for per-item diagnostic; differs from main random-position donor estimand',
             'Graph completeness uses absolute direct source attribution, including bias as unexplained',
             'Norm-only survival still freezes independent feature-source activations'])
(root/'results.json').write_text(json.dumps(result,indent=2))
