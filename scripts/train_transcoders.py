"""Stream web/chat text through all-layer skip transcoders, without activation dumps."""
import argparse,hashlib,json,time,os
from pathlib import Path
from contextlib import contextmanager
import numpy as np
import torch,torch.nn.functional as F,yaml
from ilslens.model import load_local
from ilslens.transcoders.train import SkipTranscoder
from ilslens.transcoders.corpus import mixed_examples,audit_budget
from ilslens.transcoders.eval import ce_recovered
from ilslens.experiment import Experiment,save_tensor,write_json

@contextmanager
def capture_mlp(model):
    inputs={};outputs={};handles=[]
    def hook(layer):
        def capture(module,args,output):inputs[layer]=args[0].detach();outputs[layer]=output.detach()
        return capture
    try:
        for layer,block in enumerate(model.model.layers):handles.append(block.mlp.register_forward_hook(hook(layer)))
        yield inputs,outputs
    finally:
        for handle in handles:handle.remove()

@contextmanager
def substitute(model,transcoders=None):
    handles=[]
    def hook(layer):
        return lambda module,args,out:torch.zeros_like(out) if transcoders is None else transcoders[layer](args[0])
    try:
        for layer,block in enumerate(model.model.layers):handles.append(block.mlp.register_forward_hook(hook(layer)))
        yield
    finally:
        for handle in handles:handle.remove()

def ce(model,ids):
    logits=model(ids,use_cache=False).logits
    return F.cross_entropy(logits[:,:-1].reshape(-1,model.config.vocab_size),ids[:,1:].reshape(-1)).item()

def main():
    p=argparse.ArgumentParser();p.add_argument('--corpus',required=True);p.add_argument('--config',default='configs/transcoders.yaml')
    p.add_argument('--output',default='runs/transcoders');p.add_argument('--device',default='cuda');args=p.parse_args()
    config=yaml.safe_load(Path(args.config).read_text())
    digest=hashlib.sha256(Path(args.corpus).read_bytes()).hexdigest()
    run=Experiment(args.output,dict(config=config,corpus_sha256=digest,device=args.device))
    unique=config.get('unique_training_windows',False)
    if unique:
        from transformers import AutoTokenizer
        tokenizer=AutoTokenizer.from_pretrained(os.environ['ILS_MODEL_PATH'],local_files_only=True)
        audit=audit_budget(args.corpus,tokenizer,config['token_presentations'],config['chat_every'])
        write_json(run.path/'corpus-budget.json',audit)
        if not audit['passes']:
            raise ValueError('Corpus cannot meet the distinct training-window budget without held-out overlap; inspect corpus-budget.json')
    model,tokenizer=load_local(args.device,min_free_gib=config.get('estimated_gpu_gib',6.));d=model.config.hidden_size;layers=len(model.model.layers)
    torch.manual_seed(config['seed'])
    tcs=[SkipTranscoder(d,config['width_multiplier']*d,config['top_k']).to(model.device) for _ in range(layers)]
    # Sequential optimizer updates bound temporary state, while all optimizers persist for exact resume.
    optimizers=[torch.optim.Adam(tc.parameters(),lr=config['learning_rate']) for tc in tcs]
    state_path=run.path/'training-state.pt';completed=0;source_counts={};documents=set();window_hashes=set()
    if state_path.exists():
        state=torch.load(state_path,weights_only=True,map_location=model.device)
        if state['fingerprint']!=run.fingerprint:raise ValueError('Checkpoint provenance mismatch')
        for tc,opt,weights,optimizer in zip(tcs,optimizers,state['models'],state['optimizers']):tc.load_state_dict(weights);opt.load_state_dict(optimizer)
        completed=state['steps'];source_counts=state['source_counts'];documents=set(state['documents']);window_hashes=set(state['window_hashes'])
        del state
    total_steps=config['token_presentations']//128
    if config['token_presentations']%128:raise ValueError('Budget must be a multiple of 128')
    def checkpoint(step):
        save_tensor(state_path,dict(fingerprint=run.fingerprint,steps=step,models=[tc.state_dict() for tc in tcs],
            optimizers=[o.state_dict() for o in optimizers],source_counts=source_counts,documents=sorted(documents),window_hashes=sorted(window_hashes)))
    # Deterministic corpus order; resume advances over the already processed prefix.
    iterator=mixed_examples(args.corpus,tokenizer,'train',config['chat_every'],cycle=not unique,unique=unique)
    with capture_mlp(model) as (inputs,outputs):
        for step in range(total_steps):
            try:ids,provenance=next(iterator)
            except StopIteration:raise ValueError('Corpus exhausted before the registered training budget')
            if len(ids)!=128:raise ValueError('Token budget assumes exact 128-token windows')
            if step<completed:continue
            ids=ids[None].to(model.device)
            with torch.no_grad():model.model(ids,use_cache=False)
            for layer,(tc,optimizer) in enumerate(zip(tcs,optimizers)):
                optimizer.zero_grad(set_to_none=True)
                loss=(tc(inputs[layer])-outputs[layer]).square().mean()
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite transcoder loss')
                loss.backward();optimizer.step()
            source_counts[provenance['source']]=source_counts.get(provenance['source'],0)+128
            documents.add(provenance['document_id']);window_hashes.add(provenance['sha256'])
            if (step+1)%config['checkpoint_steps']==0 or step+1==total_steps:
                checkpoint(step+1);run.event('training-checkpoint',steps=step+1,token_presentations=(step+1)*128)
    del optimizers
    for layer,tc in enumerate(tcs):
        tc.eval().requires_grad_(False)
        metadata=dict(fingerprint=run.fingerprint,layer=layer,model_type=model.config.model_type,hidden_size=d,
            model='Qwen/Qwen2.5-0.5B-Instruct',width=config['width_multiplier']*d,top_k=config['top_k'],
            token_presentations=total_steps*128,distinct_token_windows=len(window_hashes),distinct_tokens=len(window_hashes)*128,
            unique_training_windows=unique,
            distinct_documents=len(documents),source_token_presentations=source_counts,seed=config['seed'])
        save_tensor(run.path/f'layer-{layer:02d}.pt',dict(state_dict={k:v.cpu() for k,v in tc.state_dict().items()},metadata=metadata))
    clean=[];replaced=[];ablated=[];records=[];dead=[torch.ones(tc.encoder.out_features,dtype=torch.bool,device=model.device) for tc in tcs]
    with capture_mlp(model) as (inputs,outputs),torch.no_grad():
        for i,(tokens,provenance) in enumerate(mixed_examples(args.corpus,tokenizer,'validation',config['chat_every'],cycle=False)):
            if i>=config['evaluation_windows']:break
            ids=tokens[None].to(model.device);clean.append(ce(model,ids));row=[]
            for layer,tc in enumerate(tcs):
                x=inputs[layer];y=outputs[layer];pred=tc(x)
                row.append(dict(sse=float((pred-y).square().sum()),sum=y.sum((0,1)).cpu(),ss=float(y.square().sum()),n=y.numel()//d))
                dead[layer]&=tc.features(x).sum((0,1))==0
            with substitute(model,tcs):replaced.append(ce(model,ids))
            with substitute(model):ablated.append(ce(model,ids))
            records.append(dict(document_id=provenance['document_id'],metrics=row))
    if len(records)<2:raise ValueError('Need at least two held-out windows for evaluation')
    groups=sorted(set(r['document_id'] for r in records));rng=np.random.default_rng(config['seed'])
    def indices(sample):return [i for group in sample for i,r in enumerate(records) if r['document_id']==group]
    boot=[indices(rng.choice(groups,len(groups),replace=True)) for _ in range(200)]
    def explained(layer,index):
        rows=[records[i]['metrics'][layer] for i in index];n=sum(r['n'] for r in rows)
        denominator=sum(r['ss'] for r in rows)-float(sum(r['sum'] for r in rows).square().sum())/n
        return 1-sum(r['sse'] for r in rows)/denominator if denominator>0 else float('nan')
    metrics=[dict(layer=layer,variance_explained=explained(layer,range(len(records))),
        variance_explained_ci=np.nanquantile([explained(layer,index) for index in boot],[.025,.975]).tolist(),
        dead_fraction=float(dead[layer].float().mean()),dead_definition='Inactive on held-out evaluation windows') for layer in range(layers)]
    clean,replaced,ablated=map(np.array,(clean,replaced,ablated))
    recovered=ce_recovered(clean.mean(),replaced.mean(),ablated.mean())
    cis=np.nanquantile([ce_recovered(clean[i].mean(),replaced[i].mean(),ablated[i].mean()) for i in boot],[.025,.975]).tolist()
    run.result(dict(transcoders=metrics,ce=dict(clean=float(clean.mean()),replaced=float(replaced.mean()),ablated=float(ablated.mean()),recovered=float(recovered),ci=cis),
        evaluation_windows=len(records),evaluation_documents=len(groups),bootstrap_unit='source_document',training=metadata,
        deviation=('Below the 100M-token target' if total_steps*128<100000000 else None),
        window_policy='No repeated training windows' if unique else 'Explicit legacy cycling; presentations are not unique tokens'))

if __name__=='__main__':main()
