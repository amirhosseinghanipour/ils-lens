"""Resumable primary J/R fitting and all-layer donor collection.

No evaluation labels are read. Checkpoints are per prompt/layer, atomic, and
bound to configuration and token provenance before any expensive work begins.
"""
import argparse,hashlib,json,os,resource,time,subprocess
from pathlib import Path
from contextlib import nullcontext
import torch,yaml
from transformers import AutoModelForCausalLM,AutoTokenizer
from ilslens.data import wikitext_sequences
from ilslens.reference import load_reference,token_adapter
from ilslens.baselines import lrp_rules
from ilslens.budget import baseline_count
from ilslens.suffix import QwenSuffix
from ilslens.runtime import resolve_device,synchronize

def atomic_save(value,path):
    temporary=path.with_suffix(path.suffix+'.tmp')
    torch.save(value,temporary);temporary.replace(path)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--dataset',required=True)
    parser.add_argument('--reference',required=True)
    parser.add_argument('--config',default='configs/qwen-main.yaml')
    parser.add_argument('--output',default='runs/qwen-main')
    parser.add_argument('--phase',choices=['benchmark','baselines','collect','all'],default='all')
    parser.add_argument('--device',default='cuda',help='Explicit device; defaults to guarded CUDA, never CPU fallback')
    parser.add_argument('--min-free-gib',type=float,default=6.)
    args=parser.parse_args()
    config=yaml.safe_load(Path(args.config).read_text())
    device=resolve_device(args.device,min_free_gib=args.min_free_gib)
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(config['threads']);torch.manual_seed(config['seed'])
    start=time.perf_counter()
    model_path=os.environ['ILS_MODEL_PATH']
    model=AutoModelForCausalLM.from_pretrained(model_path,local_files_only=True,dtype=torch.float32,
        attn_implementation='eager').to(device).eval().requires_grad_(False)
    if model.config.hidden_size!=896 or model.config.num_hidden_layers!=24:
        raise ValueError('This registered run is specifically Qwen2.5-0.5B-Instruct')
    tokenizer=AutoTokenizer.from_pretrained(model_path,local_files_only=True)
    ids,rows=wikitext_sequences(args.dataset,tokenizer,'train',config['source_prompts'])
    donors,donor_rows=wikitext_sequences(args.dataset,tokenizer,'train',config['donor_prompts'],
                                       offset=config['source_prompts'])
    reference_revision=subprocess.check_output(['git','-C',args.reference,'rev-parse','HEAD'],text=True).strip()
    dataset_manifest=Path(args.dataset).parent/'wikitext-manifest.json'
    provenance=dict(config=config,source_rows=rows,donor_rows=donor_rows,model='Qwen/Qwen2.5-0.5B-Instruct',
        reference_revision=reference_revision,dataset=json.loads(dataset_manifest.read_text()),
        model_config_sha256=hashlib.sha256((Path(model_path)/'config.json').read_bytes()).hexdigest())
    fingerprint=hashlib.sha256(json.dumps(provenance,sort_keys=True).encode()).hexdigest()
    manifest=output/'manifest.json'
    if manifest.exists() and json.loads(manifest.read_text())['fingerprint']!=fingerprint:
        raise ValueError('Configuration or token provenance changed; use a new run directory')
    manifest.write_text(json.dumps(dict(provenance,fingerprint=fingerprint),indent=2))
    (output/'estimate.json').write_text(json.dumps({key:value for key,value in config.items() if 'estimated' in key},indent=2))
    def progress(stage,**kwargs):
        record=dict(stage=stage,elapsed_seconds=time.perf_counter()-start,
                    execution_device=str(device),tf32=False,
                    maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,**kwargs)
        with (output/'progress.jsonl').open('a') as file:file.write(json.dumps(record)+'\n')
        print(json.dumps(record),flush=True)
    if args.phase=='benchmark':
        replay=QwenSuffix(model,ids[:1]);measurements=[]
        for layer in (0,7,15,23):
            delta=torch.randn(1,896)*torch.tensor(config['epsilons'])[:,None]*.01
            synchronize(device);begin=time.perf_counter()
            for _ in range(3):replay.effects(layer,64,delta)
            synchronize(device)
            measurements.append(dict(layer=layer,seconds_per_six_epsilon_sample=(time.perf_counter()-begin)/3))
        (output/'suffix-benchmark.json').write_text(json.dumps(measurements,indent=2))
        progress('benchmark-complete',measurements=measurements);return
    if args.phase in ('baselines','all'):
        root=output/'baselines';root.mkdir(exist_ok=True)
        reference=load_reference(args.reference)
        baseline_ids=torch.cat((ids,donors)) if baseline_count(config)>len(ids) else ids
        baseline_rows=rows+donor_rows if baseline_count(config)>len(ids) else rows
        for i in range(len(baseline_ids)):
            adapter=token_adapter(reference,model,tokenizer,baseline_ids[i:i+1])
            for name in ('J','R'):
                destination=root/f'{name}-prompt-{i:03d}.pt'
                if destination.exists():continue
                before=time.perf_counter()
                with lrp_rules(model) if name=='R' else nullcontext():
                    lens=reference.fit(adapter,['0'],dim_batch=config['dim_batch'],max_seq_len=128,skip_first=16)
                matrices=lens.jacobians;matrices[23]=torch.eye(896)
                atomic_save(dict(matrices=matrices,metadata=dict(fingerprint=fingerprint,
                    prompt=i,source_row=baseline_rows[i],target_layer=23,source_convention='block_output',
                    target_reduction='sum',label_tuned=False,execution_device=str(device))),destination)
                progress('baseline-prompt',lens=name,prompt=i,seconds=time.perf_counter()-before)
        progress('baselines-complete')
    if args.phase in ('collect','all'):
        for split,source,donor_ids,total in [
            ('train',ids,donors,config['samples_per_layer_epsilon']),
            ('validation',None,None,config['heldout_samples_per_layer_epsilon'])]:
            if source is None:
                source,held_rows=wikitext_sequences(args.dataset,tokenizer,'validation',config['heldout_prompts'])
                donor_ids,held_donors=wikitext_sequences(args.dataset,tokenizer,'validation',config['heldout_prompts'],offset=config['heldout_prompts'])
                (output/'heldout-provenance.json').write_text(json.dumps(dict(sources=held_rows,donors=held_donors),indent=2))
            if total%len(source):raise ValueError('Sample count must divide prompt count exactly')
            per_prompt=total//len(source)
            bank=[]
            for tokens in donor_ids:
                replay=QwenSuffix(model,tokens[None]);bank.append(torch.stack([replay.states[l][0,16:-1].cpu().clone() for l in range(24)]))
            bank=torch.stack(bank)
            root=output/'samples'/split;root.mkdir(parents=True,exist_ok=True)
            for i,tokens in enumerate(source):
                replay=QwenSuffix(model,tokens[None])
                for layer in range(24):
                    destination=root/f'prompt-{i:03d}-layer-{layer:02d}.pt'
                    if destination.exists():continue
                    before=time.perf_counter()
                    generator=torch.Generator().manual_seed(config['seed']+10000*i+layer+(1000000 if split=='validation' else 0))
                    xs=[];ys=[];sites=[];donor_sites=[]
                    for _ in range(per_prompt):
                        t=int(torch.randint(16,127,(),generator=generator))
                        j=int(torch.randint(len(bank),(),generator=generator));u=int(torch.randint(111,(),generator=generator))
                        displacement=bank[j,layer,u]-replay.states[layer][0,t].cpu()
                        delta=torch.tensor(config['epsilons'])[:,None]*displacement
                        response=replay.effects(layer,t,delta,gammas=config['gammas'])
                        xs.append(delta);ys.append(response.cpu());sites.append(t);donor_sites.append((j,u+16))
                    atomic_save(dict(delta=torch.stack(xs),response=torch.stack(ys),
                        source_positions=torch.tensor(sites),donor_sites=torch.tensor(donor_sites),
                        prompt_ids=torch.full((per_prompt,),i),metadata=dict(fingerprint=fingerprint,
                        layer=layer,target_layer=23,split=split,epsilon=config['epsilons'],gamma=config['gammas'],
                        delta_dist='donor',source_convention='block_output',target='pre_final_norm',
                        execution_device=str(device))),destination)
                    progress('collection-shard',split=split,prompt=i,layer=layer,samples=per_prompt,seconds=time.perf_counter()-before)
            del bank
        progress('collection-complete')
    progress('requested-phases-complete')

if __name__=='__main__':main()
