"""Four-layer transformer trained on a seeded HMM with occasional STATE reports."""
import json,time,resource
from pathlib import Path
from contextlib import nullcontext
import numpy as np
import torch
from transformers import Qwen2Config,Qwen2ForCausalLM
from ..hooks import capture_blocks,patch_site,final_state
from ..collect import antithetic,save_samples
from ..fit import select
from ..baselines import lrp_rules
from ..readout import raw
from .faithfulness import bootstrap_r2

def sequences(n,seed,length=128):
    rng=np.random.default_rng(seed)
    ids=np.zeros((n,length),dtype=np.int64)
    truth=np.zeros_like(ids); report=np.zeros_like(ids,dtype=bool)
    # Tokens 0..7 = observations, 8 = STATE, 9..12 = state IDs, 13 = BOS.
    for i in range(n):
        state=int(rng.integers(4)); ids[i,0]=13; t=1
        while t<length:
            if rng.random()<.08: state=int(rng.integers(4))
            if t<length-1 and rng.random()<.1:
                ids[i,t:t+2]=[8,9+state]; truth[i,t:t+2]=state
                report[i,t:t+2]=True; t+=2
            else:
                ids[i,t]=2*state+int(rng.integers(2)) if rng.random()<.85 else int(rng.integers(8))
                truth[i,t]=state; t+=1
    return torch.tensor(ids),torch.tensor(truth),torch.tensor(report)

def run(config):
    start=time.perf_counter(); torch.set_num_threads(2); torch.manual_seed(config['seed'])
    root=Path('artifacts/toy'); root.mkdir(parents=True,exist_ok=True)
    model=Qwen2ForCausalLM(Qwen2Config(vocab_size=14,hidden_size=16,intermediate_size=32,
        num_hidden_layers=4,num_attention_heads=2,num_key_value_heads=1,
        max_position_embeddings=128,tie_word_embeddings=True,attn_implementation='eager')).float()
    optimizer=torch.optim.AdamW(model.parameters(),lr=.003)
    losses=[]
    for step in range(config['train_steps']):
        ids,_,_=sequences(config['train_batch'],1000+step)
        optimizer.zero_grad(); loss=model(ids,labels=ids,use_cache=False).loss
        loss.backward(); optimizer.step(); losses.append(loss.item())
    training_seconds=time.perf_counter()-start
    model.eval().requires_grad_(False)
    torch.save(model.state_dict(),root/'model.pt')
    fit_ids,fit_truth,fit_report=sequences(config['fit_samples'],20000)
    donor_ids,_,_=sequences(config['fit_samples'],30000)
    test_ids,test_truth,test_report=sequences(config['heldout_samples'],40000)
    test_donors,_,_=sequences(config['heldout_samples'],50000)
    # Every sample has its own independent prompt and independent donor prompt.
    rng=torch.Generator().manual_seed(37)
    def positions(report):
        return torch.tensor([int((eligible:=torch.where(~row[16:-1])[0]+16)[
            torch.randint(len(eligible),(),generator=rng)]) for row in report])
    train_positions=positions(fit_report); test_positions=positions(test_report)
    rows=[]
    for layer in range(4):
        def gather(ids,donors,pos,jacobians=False):
            xs,ys,hs,js,rs=[],[],[],[],[]
            for index,(tokens,donor_tokens,t) in enumerate(zip(ids,donors,pos)):
                tokens=tokens[None]; t=int(t)
                with torch.no_grad(),capture_blocks(model,[layer]) as states:
                    model.model(tokens,use_cache=False)
                    h=states[layer][0,t].clone()
                    model.model(donor_tokens[None],use_cache=False)
                    u=int(torch.randint(16,127,(),generator=rng))
                    hd=states[layer][0,u].clone()
                delta=config['epsilon']*(hd-h)
                y=antithetic(model,tokens,layer,t,delta)[:,0]
                xs.append(delta); ys.append(y); hs.append(h)
                if jacobians:
                    def function(d):
                        with patch_site(model,layer,t,d):
                            return final_state(model,tokens)[0,t:-1].mean(0)
                    for target,context in ((js,nullcontext()),(rs,lrp_rules(model))):
                        with context:
                            target.append(torch.autograd.functional.jacobian(function,torch.zeros(16),vectorize=True))
            result=[torch.stack(xs),torch.stack(ys),torch.stack(hs)]
            if jacobians: result.extend([torch.stack(js),torch.stack(rs)])
            return result
        x,y,h,js,rs=gather(fit_ids,donor_ids,train_positions,True)
        j,r=js.mean(0),rs.mean(0)
        xt,yt,ht=gather(test_ids,test_donors,test_positions)
        prompt_ids=torch.arange(len(x))
        metadata=dict(config,layer=layer,target_layer=3,layer_convention='block_output_zero_based',
            target='pre_final_norm',data='synthetic_HMM',reference_baselines=False)
        save_samples(root/f'samples-{layer}.pt',x,y,prompt_ids,metadata)
        save_samples(root/f'heldout-{layer}.pt',xt,yt,torch.arange(len(xt)),metadata)
        a,winner,cv=select(x,y[:,-1],prompt_ids,{'I':torch.eye(16),'J':j,'R':r},
                         [0.,1e-6,1e-4,.01],seed=config['seed'],prior_samples={'J':js,'R':rs})
        matrices={'logit':torch.eye(16),'J-local':j,'R-local':r,'ILS':a}
        torch.save(dict(matrices=matrices,metadata=metadata,selection=winner,cv=cv),root/f'lenses-{layer}.pt')
        state_tokens=test_truth[torch.arange(len(ht)),test_positions]+9
        for name,matrix in matrices.items():
            metric=bootstrap_r2(yt[:,-1],xt@matrix.T,torch.arange(len(xt)))
            scores=raw(ht,matrix,model.lm_head.weight,model.model.norm.weight)
            recalls={}
            for k in (1,10):
                success=(scores.topk(k).indices==state_tokens[:,None]).any(1).float()
                generator=torch.Generator().manual_seed(17)
                bootstrap=success[torch.randint(len(success),(200,len(success)),generator=generator)].mean(1)
                recalls[f'recall@{k}']=success.mean().item()
                recalls[f'recall@{k}_ci']=bootstrap.quantile(torch.tensor([.025,.975])).tolist()
            rows.append(dict(layer=layer,lens=name,**metric,**recalls))
        print('completed toy layer',layer,flush=True)
    with torch.no_grad():
        logits=model(test_ids,use_cache=False).logits
        report_mask=test_ids[:,:-1]==8
        report_accuracy=(logits[:,:-1].argmax(-1)[report_mask]==test_ids[:,1:][report_mask]).float().mean().item()
        heldout_ce=model(test_ids,labels=test_ids).loss.item()
    result=dict(config=config,rows=rows,training_seconds=training_seconds,
        total_seconds=time.perf_counter()-start,train_loss_initial=losses[0],train_loss_final=losses[-1],
        heldout_ce=heldout_ce,report_token_accuracy=report_accuracy,
        maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        warning='Toy pilot; local J/R implementations; no J++; no Qwen research conclusions')
    (root/'results.json').write_text(json.dumps(result,indent=2))
    return result
